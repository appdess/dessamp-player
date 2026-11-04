import json
import os
import argparse
from pathlib import Path

# Paths are derived from this script so the generator works from any checkout.
ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT_DIR / "trainer_source"
INPUT_WAV = TRAINER_DIR / "training_data/input.wav"

# Data folders for different amp channels
DATA_FOLDERS = {
    "blue": ROOT_DIR / "NeuralAmpModeler/resources/models/parametric-training-test-files",
    "red": ROOT_DIR / "NeuralAmpModeler/resources/models/parametric-training-test-files/5153-red-training-files"
}

OUTPUT_DATA_CONFIG = TRAINER_DIR / "parametric_data.json"
OUTPUT_MODEL_CONFIG = TRAINER_DIR / "parametric_wavenet.json"

def get_gain_from_filename(filename):
    # Example: 5153-blue-Gain-2#01.wav -> 2
    try:
        parts = filename.split("Gain-")
        if len(parts) > 1:
            gain_str = parts[1].split("#")[0].split(".")[0] # Handle #01.wav or just .wav
            return int(gain_str)
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate training configs for parametric NAM")
    parser.add_argument("--channel", type=str, choices=["blue", "red"], default="blue",
                        help="Amp channel to train (blue or red)")
    args = parser.parse_args()
    
    DATA_DIR = DATA_FOLDERS[args.channel]
    print(f"Using data folder for '{args.channel}' channel: {DATA_DIR}")
    
    # 1. Generate Data Config
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".wav")])
    outputs = []
    
    print(f"Found {len(files)} re-amped files.")
    
    for f in files:
        gain = get_gain_from_filename(f)
        if gain is not None:
            # Map Gain 2 -> 0.2
            normalized_gain = gain / 10.0
            entry = {
                "y_path": str(DATA_DIR / f),
                "g_vector": [normalized_gain]
            }
            outputs.append(entry)
            print(f"Added {f}: Gain {gain} -> {normalized_gain}")
        else:
            print(f"Skipping {f} (Could not parse gain)")
            
    if not outputs:
        print("No output files found!")
        return

    # Create nested structure with train/val split
    # Using all files for both train and validation due to small dataset size
    data_config = {
        "type": "shared_in_multi_out",
        "common": {
            "nx": 64, # Placeholder
            "input_gain": 0.0,
            "delay": 0,
            "x_path": str(INPUT_WAV)
        },
        "train": {
            "outputs": outputs
        },
        "validation": {
            "outputs": outputs
        }
    }
    
    with open(OUTPUT_DATA_CONFIG, "w") as f:
        json.dump(data_config, f, indent=4)
    print(f"Wrote {OUTPUT_DATA_CONFIG}")

    # 2. Generate Model Config
    # Load default to get structure or just define it
    # I'll define a standard structure based on wavenet.json viewed earlier
    # BUT with global_condition_size = 1
    
    model_config = {
        "net": {
            "name": "WaveNet",
            "global_condition_size": 1,
            "config": {
                "layers_configs": [
                    {
                        "condition_size": 1,
                        "global_condition_size": 1,
                        "input_size": 1,
                        "channels": 16,
                        "head_size": 8,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False
                    },
                    {
                        "condition_size": 1,
                        "global_condition_size": 1,
                        "input_size": 16,
                        "channels": 8,
                        "head_size": 1,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True
                    }
                ],
                "head_scale": 0.02
            }
        },
        "optimizer": {
            "lr": 0.004
        },
        "lr_scheduler": {
            "class": "ExponentialLR",
            "kwargs": {
                "gamma": 0.993
            }
        }
    }

    with open(OUTPUT_MODEL_CONFIG, "w") as f:
        json.dump(model_config, f, indent=4)
    print(f"Wrote {OUTPUT_MODEL_CONFIG}")

if __name__ == "__main__":
    main()
