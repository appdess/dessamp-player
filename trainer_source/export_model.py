import argparse
import json
import os
from pathlib import Path
import torch
from nam.train.lightning_module import LightningModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str, help="Directory containing checkpoints and config")
    parser.add_argument("output_dir", type=str, help="Directory to save exported model")
    parser.add_argument("--input-level", type=float, help="Input level in dBu")
    parser.add_argument("--output-level", type=float, help="Output level in dBu")
    args = parser.parse_args()
    
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find config
    config_path = ckpt_dir / "config_model.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        return

    with open(config_path, "r") as f:
        model_config = json.load(f)
        
    # Find best checkpoint
    # Assuming standard lightning naming or just pick the one with lowest val_loss if possible
    # Or just use the one provided by user if they pass a file
    # But usually custom_train_full saves 'config_model.json' in value dir.
    
    # List .ckpt files recursively
    ckpts = [p for p in ckpt_dir.rglob("*.ckpt")]
    if not ckpts:
        print(f"No .ckpt found in {ckpt_dir}")
        return
        
    # Sort by modification time or name?
    # Trainer usually saves 'last.ckpt' or similar.
    # filename="{epoch:04d}_{step}_{ESR:.3e}_{MSE:.3e}"
    # I should try to find the one with lowest ESR (if in name)
    
    # Find best checkpoint based on ESR if available
    best_ckpt = None
    best_esr = float("inf")
    best_metrics = {}
    
    import re
    # Pattern: epoch=XXXX_step=XXXX_ESR=X.XXXe-XX_MSE=X.XXXe-XX.ckpt
    pattern = re.compile(r"ESR=([0-9]+\.[0-9]+e[-+][0-9]+)_MSE=([0-9]+\.[0-9]+e[-+][0-9]+)")
    
    for ckpt in ckpts:
        name = ckpt.name
        match = pattern.search(name)
        if match:
            try:
                esr = float(match.group(1))
                mse = float(match.group(2))
                if esr < best_esr:
                    best_esr = esr
                    best_ckpt = ckpt
                    best_metrics = {"ESR": esr, "MSE": mse, "Checkpoint": name}
            except ValueError:
                continue
                
    if best_ckpt is None:
        # Fallback to latest modified
        print("Could not parse ESR from filenames, selecting latest...")
        best_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
        best_metrics = {"Checkpoint": best_ckpt.name, "Note": "Selected by timestamp (parsing failed)"}
    else:
        print(f"Selected best checkpoint: {best_ckpt.name} (ESR={best_esr:.6f})")

    # Save metrics
    metrics_path = out_dir / "model_stats.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=4)
    print(f"Saved stats to {metrics_path}")

    # Load model
    print("Loading model...")
    model = LightningModule.load_from_safe_checkpoint(
        str(best_ckpt),
        **LightningModule.parse_config(model_config)
    )
    model.eval()
    model.cpu()
    
    # Export
    print(f"Exporting to {out_dir}")
    from nam.models.metadata import UserMetadata
    user_metadata = UserMetadata(
        input_level_dbu=args.input_level,
        output_level_dbu=args.output_level
    )
    model.net.export(out_dir, user_metadata=user_metadata)
    print("Done")

if __name__ == "__main__":
    main()
