from time import sleep
from nam.train import full as _full
from pathlib import Path as _Path
import json as _json
from nam.util import timestamp as _timestamp
from nam.train.full import _create_callbacks
from nam.train.lightning_module import LightningModule as _LightningModule
from nam.train.checkpoint import validated_trainer_options as _validated_trainer_options
from nam.data import Split as _Split, init_dataset as _init_dataset, wav_to_tensor as _wav_to_tensor
from torch.utils.data import DataLoader as _DataLoader, TensorDataset as _TensorDataset
import pytorch_lightning as _pl
from warnings import warn as _warn
from copy import deepcopy as _deepcopy
import torch as _torch
from nam.util import filter_warnings as _filter_warnings
from pytorch_lightning.utilities.warnings import (
    PossibleUserWarning as _PossibleUserWarning,
)
import os as _os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as _np
import torchaudio as _ta


_torch.set_float32_matmul_precision("medium")

def ensure_outdir(outdir: str) -> _Path:
    outdir = _Path(outdir, _timestamp() + "_active_learner")
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir

def train_ensemble_member(rank, model_config, aggregated_data_config, learning_config, outdir, i, managed_ckpts_list, old_ckpt_paths=None):
    """Train a single ensemble member on a specific GPU."""
    _torch.cuda.set_device(rank)
    device = _torch.device(f"cuda:{rank}")
    model_config_local = _deepcopy(model_config)
    if old_ckpt_paths is not None:
        model_config_local["checkpoint-path"] = old_ckpt_paths[rank] 
        print(f"Finetuning ensemble member {rank} from checkpoint {model_config_local['checkpoint-path']}")
        learning_config["trainer"]["max_epochs"] = 5
        print(f"Reduced number of epochs. Finetuning ensemble member {rank} for 5 epochs.")
    model = _LightningModule.init_from_config(model_config)
    model.to(device)

    root_dir = outdir / _Path(f"round_{i}_ensemble_member_{rank}")
    root_dir.mkdir(parents=True, exist_ok=False)
    ckpt_path = root_dir / _Path(f"for_main_thread.ckpt")

    aggregated_data_config["common"]["nx"] = model.net.receptive_field

    if aggregated_data_config["train"]["outputs"] == []:
        print("\n\nNo training data. This should NOT happen as we're using starter data for first round. \n\n")
        return

    from nam import data, shared_in_multi_out
    data.register_dataset_initializer("shared_in_multi_out", shared_in_multi_out.SharedInputMultiOutputDataset.init_from_config)

    dataset_validation = _init_dataset(aggregated_data_config, _Split.VALIDATION)
    model.net.sample_rate = dataset_validation.sample_rate
    
    aggregated_dataset = _init_dataset(aggregated_data_config, _Split.TRAIN)
    trainer_options = dict(learning_config["trainer"])
    trainer_options.pop("devices", None)
    trainer_options = _validated_trainer_options(
        trainer_options,
        context="learning_config.trainer",
    )
    trainer_fit_kwargs = _validated_trainer_options(
        learning_config.get("trainer_fit_kwargs", {}),
        context="learning_config.trainer_fit_kwargs",
    )
    trainer = _pl.Trainer(
        callbacks=_create_callbacks(learning_config),
        default_root_dir=root_dir,
        **trainer_options,
        devices=[rank],
        enable_progress_bar=True
    )

    train_dataloader = _DataLoader(
        aggregated_dataset, **learning_config["train_dataloader"]
    )

    val_dataloader = _DataLoader(
        dataset_validation, **learning_config["val_dataloader"]
    )
    
    with _filter_warnings("ignore", category=_PossibleUserWarning):
        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            **trainer_fit_kwargs
        )

    trainer.save_checkpoint(ckpt_path)
    managed_ckpts_list.append(ckpt_path)

def find_optimal_g(x_path, out_dir, ensemble_models_list, num_restarts, num_opt_steps, opt_lr, nx, ny, batch_size, round_idx, main_device, model_type, mel):
    x = _wav_to_tensor(x_path)
    g_dim = ensemble_models_list[0][1].net.global_condition_size
    best_gs = []

    # Pre-build mel spectrogram transforms if requested (mirror _mel_mrstft_loss params)
    mel_transforms = None
    if mel:
        window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        mel_bins = [5, 10, 20, 40, 80, 160, 320]
        sample_rate = ensemble_models_list[0][1].net.sample_rate
        mel_transforms = []
        for win_len, n_mels in zip(window_lengths, mel_bins):
            hop_length = win_len // 4
            mel_t = _ta.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win_len,
                hop_length=hop_length,
                win_length=win_len,
                n_mels=n_mels,
                power=1.0,
                normalized=False,
            ).to(main_device)
            mel_transforms.append(mel_t)

    # Chop x into chunks
    chunks = x.unfold(0, ny, ny-nx).contiguous()
    chunk_dataset = _TensorDataset(chunks)
    chunk_loader = _DataLoader(chunk_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for i in range(num_restarts):
        current_g_latent = _torch.randn(g_dim, requires_grad=True)
        optimizer = _torch.optim.Adam([current_g_latent], lr=opt_lr)
        losses = []

        for step in tqdm(range(num_opt_steps), desc=f"Restart {i}"):
            optimizer.zero_grad()
            g_for_model = _torch.sigmoid(current_g_latent).unsqueeze(0)
            ensemble_outputs = []
            for batch in chunk_loader:
                chunk_batch = batch[0]  # Shape: (batch_size, ny)
                current_batch_size = chunk_batch.size(0)
                g_batch = g_for_model.expand(current_batch_size, -1)

                model_outputs = []
                for rank, model in ensemble_models_list:
                    chunk_batch_cuda = chunk_batch.to(f"cuda:{rank}")
                    g_batch_cuda = g_batch.to(f"cuda:{rank}")
                    out = model(chunk_batch_cuda, g_batch_cuda)  # (batch_size, ?)
                    model_outputs.append(out.to(main_device))

                stacked = _torch.stack(model_outputs, dim=0)  # (num_models, batch_size, ?)
                ensemble_outputs.append(stacked)

            ensemble_outputs = _torch.cat(ensemble_outputs, dim=1)  # (num_models, total_chunks, ?)
            if not mel:
                variance_wave = _torch.var(ensemble_outputs, dim=0)
                disagreement_score = variance_wave.mean()
            else:
                variance_wave = _torch.var(ensemble_outputs, dim=0).mean()
                M, Btot, _Lsig = ensemble_outputs.shape
                mel_var_scalars = []
                if mel_transforms is None:
                    raise RuntimeError("mel_transforms not initialized while mel=True")
                for mel_t in mel_transforms:
                    mel_feats_per_model = []
                    for m in range(M):
                        spec = mel_t(ensemble_outputs[m])
                        spec = _torch.log10(spec + 1e-5)
                        mel_feats_per_model.append(spec)
                    mel_stack = _torch.stack(mel_feats_per_model, dim=0)
                    mel_var_scalars.append(mel_stack.var(dim=0).mean())
                variance_mel = _torch.stack(mel_var_scalars).mean()
                disagreement_score = 6.2e-05 * variance_mel + variance_wave

            loss = -disagreement_score

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        g = _torch.sigmoid(current_g_latent).detach().cpu().numpy()
        best_gs.append((g, disagreement_score.item()))

        # Plot
        plt.figure(figsize=(12, 8))
        plt.plot(losses, label=f"Restart {i} (Final g: {str(best_gs[i][0])} Final Disagreement: {best_gs[i][1]:.6f})")
        
        plt.xlabel("Optimization Step")
        plt.ylabel("Loss (-Disagreement Score)")
        plt.title("Loss Curve During Optimization of 'g'")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = out_dir / _Path(f"round{round_idx} plots") / _Path(f"loss_plot_restart_{i+1}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(plot_path)
            print(f"Loss plot saved to {plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")


    return best_gs

def cluster_gs(best_gs_unclustered, out_dir, max_num, round_idx):

    # from sklearn.cluster import HDBSCAN
    # clusterer = HDBSCAN(min_cluster_size=2,
    #                     min_samples=2,
    #                         metric='euclidean') 

    # # Perform clustering 
    # cluster_labels = clusterer.fit_predict([g for g, _ in best_gs_unclustered])
    # zipped = zip([g for g, _ in best_gs_unclustered], [disagreement_score for _, disagreement_score in best_gs_unclustered], cluster_labels) 
    # clsuter_dict = {}
    # for g, disagreement_score, cluster_label in zipped:
    #     if cluster_label not in clsuter_dict:
    #         clsuter_dict[cluster_label] = []
    #     clsuter_dict[cluster_label].append((g, disagreement_score))

    def group_similar_vectors(g_list, threshold=0.2):
        g_arr = _np.array(g_list)
        n = len(g_list)
        similarity_matrix = _np.ones((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if not _np.all(_np.abs(g_arr[i] - g_arr[j]) <= threshold):
                # if np.linalg.norm(g_arr[i] - g_arr[j]) > threshold:
                    similarity_matrix[i, j] = False
        
        groups = []
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue
            group = [i]
            for j in range(i + 1, n):
                if similarity_matrix[i, j] and j not in assigned:
                    group.append(j)
                    assigned.add(j)
            assigned.add(i)
            groups.append(group)
        
        return groups
    
    clusters = group_similar_vectors([g for g, _ in best_gs_unclustered])
    clsuter_dict = {}
    for cluster_label, group in enumerate(clusters):
        clsuter_dict[cluster_label] = []
        for idx in group:
            g, disagreement_score = best_gs_unclustered[idx]
            clsuter_dict[cluster_label].append((g, disagreement_score))
    

    # log cluster_dict in a readable format
    logging_path = out_dir / _Path(f"round{round_idx} plots") / _Path(f"clustered_gs.txt")
    with open(logging_path, "w") as f:
        f.write("Clustered g's and disagreement scores:\n")
        for cluster_label, g_disagreement_pairs in clsuter_dict.items():
            f.write(f"Cluster {cluster_label}:\n")
            for g, disagreement_score in g_disagreement_pairs:
                f.write(f"  g: {g}, disagreement score: {disagreement_score}\n")

    best_gs = []
    for cluster_label, g_disagreement_pairs in clsuter_dict.items():
        best_g = max(g_disagreement_pairs, key=lambda x: x[1])
        best_gs.append(best_g)

    best_gs.sort(key=lambda x: x[1], reverse=True)
    best_gs = best_gs[:max_num]
    
    with open(logging_path, "a") as f:
        f.write("\nBest g's from each cluster:\n")
        for g, disagreement_score in best_gs:
            f.write(f"  g: {g}, disagreement score: {disagreement_score}\n")
    print(f"Clustered g's and disagreement scores logged to {logging_path}")

    return best_gs
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Active Learner")

    parser.add_argument("--current-round-idx", type=int, help="Current round index. Starts at 0")
    parser.add_argument("--saved-ckpts-paths", type=str, nargs='+', default=None, help="Paths to saved checkpoints from previous rounds.")
    parser.add_argument("--output-dir", type=_Path, default=_Path("outputs/automated_active_learning_RNN"), help="Output directory for the active learner. Recommended to always make one yourself.")
    parser.add_argument("--starter-data-config-path", type=_Path, default=_Path("training_data/nam_train_data/active_leaner_starter_10.json"), help="Path to the starter data config JSON")
    parser.add_argument("--model-config-path", type=_Path, default=_Path("default_config_files/models/lstm.json"), help="Path to the model config JSON")
    parser.add_argument("--learning-config-path", type=_Path, default=_Path("default_config_files/learning/RNN-active-learning.json"), help="Path to the learning config JSON")
    parser.add_argument("--x-path-for-g-opt", type=_Path, default=_Path("training_data/nam_train_data/input.wav"), help="Path to the input WAV file for g optimization")
    parser.add_argument("--num-restarts-per-round", type=int, default=10, help="Number of restarts per round")
    parser.add_argument("--ensemble-size", type=int, default=4, help="Size of the ensemble")

    parser.add_argument("--training-ny", type=int, default=22050, help="Length of each sample in a batch for training")
    
    parser.add_argument("--g-opt-batch-size", type=int, default=96*4, help="Batch size for g optimization")
    parser.add_argument("--g-opt-ny", type=int, default=8192, help="Length of each sample in a batch for g optimization")
    parser.add_argument("--max-gs-per-round", type=int, default=10, help="Maximum number of g's to select per round")
    parser.add_argument("--use-mel-in-g-opt", action="store_true", help="Whether to use mel spectrogram in g optimization")

    args = parser.parse_args()

    CURRENT_ROUND_IDX = args.current_round_idx
    assert CURRENT_ROUND_IDX is not None, "Current round index must be provided"
    SAVED_CKPTS_PATHS = args.saved_ckpts_paths
    PROVIDED_OUTDIR = args.output_dir
    BASE_OUTPUT_DIR = _Path("outputs")
    STARTER_DATA_CONFIG_PATH = args.starter_data_config_path
    MODEL_CONFIG_PATH = args.model_config_path
    LEARNING_CONFIG_PATH = args.learning_config_path
    X_PATH_FOR_G_OPT = args.x_path_for_g_opt
    NUM_RESTARTS_PER_ROUND = args.num_restarts_per_round
    ENSEMBLE_SIZE = args.ensemble_size
    NUM_VALIDATION = 50
    MAX_GS_PER_ROUND = args.max_gs_per_round
    USE_MEL_IN_G_OPT = args.use_mel_in_g_opt

    # Set last GPU as main device
    if _torch.cuda.device_count() >= ENSEMBLE_SIZE + 1:
        MAIN_DEVICE = f"cuda:{ENSEMBLE_SIZE}"
    else:
        raise RuntimeError(
            f"Not enough GPUs available. {ENSEMBLE_SIZE + 1} GPUs are required, but only {_torch.cuda.device_count()} are available."
        )


    ### Boilerplate

    manager = _torch.multiprocessing.Manager()

    if CURRENT_ROUND_IDX == 0:
        if PROVIDED_OUTDIR: # possible to provide an outdir also for round 0
            outdir = PROVIDED_OUTDIR
            if not outdir.exists():
                print(f"Provided outdir {PROVIDED_OUTDIR} does not exist. Making a new outdir.")
                PROVIDED_OUTDIR.mkdir(parents=True, exist_ok=False)
        else:
            outdir = ensure_outdir(BASE_OUTPUT_DIR)
    else:
        outdir = PROVIDED_OUTDIR
        if not outdir.exists():
            raise RuntimeError(f"Last outdir {outdir} does not exist. It is needed since current round index > 0.")
        
    with open(MODEL_CONFIG_PATH, "r") as fp:
        model_config = _json.load(fp)
        model_type = model_config["net"]["name"]
    if model_type not in ["LSTM", "WaveNet"]:
        raise RuntimeError(f"Unsupported model type {model_type}. Only LSTM and WaveNet are supported for active learning.")
   
    with open(LEARNING_CONFIG_PATH, "r") as fp:
        learning_config = _json.load(fp)

    if CURRENT_ROUND_IDX == 0:
        with open(STARTER_DATA_CONFIG_PATH, "r") as fp:
            data_config = _json.load(fp)
        # drop data_config.train/validation.outputs.for_reference
        for s in ["train", "validation"]:
            data_config[s]["outputs"] = [
                {k: v for k, v in output.items() if k != "for_reference"}
                for output in data_config[s]["outputs"]
            ]

        data_config["validation"]["outputs"] = data_config["validation"]["outputs"][:NUM_VALIDATION]
        aggregated_data_config = _deepcopy(data_config)

    else:
        with open(_Path(outdir, f"aggregated_data_config_{CURRENT_ROUND_IDX-1}.json"), "r") as fp:
            aggregated_data_config = _json.load(fp)
            print("Current total data points: ", len(aggregated_data_config["train"]["outputs"]))

    aggregated_data_config["train"]["ny"] = args.training_ny
    if model_type == "LSTM":
        aggregated_data_config["validation"]["ny"] = args.training_ny
    # For LSTM, set the val ny as well, otherwise validation is slow

    for basename, config in (
        ("data", aggregated_data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(_Path(outdir, f"config_{basename}.json"), "w") as fp:
            _json.dump(config, fp, indent=4)


    ### One round of active learning loop, then just terminate
        
    # 1. Train ensemble members
    print(f"Starting active learner round {CURRENT_ROUND_IDX}...")
    if PROVIDED_OUTDIR: 
        print(f"Using provided outdir: {outdir}")
    else:
        print(f"Using new outdir: {outdir}")

    print(f"Using active learning inputs dir: {outdir}/active_learning_inputs")

    print("Training ensemble members.")


    if SAVED_CKPTS_PATHS is not None:
        print(f"Using saved checkpoints from previous rounds: {SAVED_CKPTS_PATHS}")
        ckpt_paths = SAVED_CKPTS_PATHS
        if len(ckpt_paths) != ENSEMBLE_SIZE:
            raise RuntimeError(f"Expected {ENSEMBLE_SIZE} checkpoint paths, but got {len(ckpt_paths)}")
    else:
        managed_ckpts_list = manager.list()
        _torch.multiprocessing.spawn(
            train_ensemble_member,
            args=(model_config, aggregated_data_config, learning_config, outdir, CURRENT_ROUND_IDX, managed_ckpts_list, None),
            nprocs=ENSEMBLE_SIZE,
            join=True,
        )
        ckpt_paths = [pred for pred in managed_ckpts_list] 


    # 2. Find optimal next g's
    print("Finding optimal g's...")

    loaded_models = []
    assert len(ckpt_paths) >= ENSEMBLE_SIZE, f"Expected {ENSEMBLE_SIZE} checkpoint paths, but got {len(ckpt_paths)}"
    for rank in range(ENSEMBLE_SIZE):
        ckpt_path = ckpt_paths[rank]
        if not _os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint {ckpt_path} does not exist")
        model = _LightningModule.load_from_safe_checkpoint(
            ckpt_path,
            **_LightningModule.parse_config(model_config),
        )
        model.to(f"cuda:{rank}")
        # Set model to train mode for g optimization. 
        # The model will not be updated, but the CuDNN backend for RNNs requires it.
        # For wavenet, this is not necessary and we can use eval mode.
        if model_type == "LSTM":
            model.train()
        elif model_type == "WaveNet":
            model.eval()
        loaded_models.append((rank, model))
    
    best_gs_unclustered = find_optimal_g(
        x_path = X_PATH_FOR_G_OPT,
        out_dir = outdir,
        ensemble_models_list=loaded_models,
        num_restarts=NUM_RESTARTS_PER_ROUND,
        num_opt_steps=300,
        opt_lr=0.02,
        nx=loaded_models[0][1].net.receptive_field, 
        ny=args.g_opt_ny,
        batch_size=args.g_opt_batch_size,
        round_idx=CURRENT_ROUND_IDX,
        main_device=MAIN_DEVICE,
        model_type=model_type,
        mel=USE_MEL_IN_G_OPT
    )

    # 3. cluster the g's 

    best_gs = cluster_gs(best_gs_unclustered, outdir, MAX_GS_PER_ROUND, CURRENT_ROUND_IDX)

    # 4. Write selected g's to aggregated data config
    
    best_gs.sort(key=lambda x: x[1], reverse=True)
    selected_gs = [g for g, _ in best_gs]
    new_config_parts = [
        {"g_vector": g.tolist(), "index": idx + CURRENT_ROUND_IDX * NUM_RESTARTS_PER_ROUND} for idx, g in enumerate(selected_gs)
    ]
    active_learning_inputs_dir = _Path(outdir, f"active_learning_inputs")
    _os.makedirs(active_learning_inputs_dir, exist_ok=True)
    aggregated_data_config["train"]["outputs"] = aggregated_data_config["train"]["outputs"] + [
        {
            "g_vector": item["g_vector"],
            "y_path": f"{active_learning_inputs_dir}/{item['index']:04d}.wav",
        }
        for item in new_config_parts
    ]

    prompt = (
        f"Please provide {len(new_config_parts)} input audio files in {active_learning_inputs_dir} for the selected g's:\n"
        + "\n".join(
            f"{item['index']:04d}.wav: {item['g_vector']}"
            for item in new_config_parts
        )
        + "\n\n"
    )
    print(prompt)

    new_points_path = active_learning_inputs_dir / _Path("new_points.json")
    with open(new_points_path, "w") as fp:
        _json.dump(new_config_parts, fp, indent=4)
    print(f"New points config written to {new_points_path}. The local machine should read this file via ssh and upload the new data points, unless data collection is manual.")

    with open(_Path(outdir, f"aggregated_data_config_{CURRENT_ROUND_IDX}.json"), "w") as fp:
        _json.dump(aggregated_data_config, fp, indent=4)

    # 5. Plot the distribution of g-vector component values so far. Hacky
    g_vectors = [output["g_vector"] for output in aggregated_data_config["train"]["outputs"][11:]] # skips starter data. TODO: make this more robust
    all_g_values = _np.array(g_vectors).flatten()

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_g_values, bins=20, edgecolor='black')
    plt.title('Distribution of g-vector Component Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(outdir / _Path(f"round{CURRENT_ROUND_IDX} plots") / _Path(f"g_vector_distribution.png"))
