import itertools
import json
import argparse
import os
import subprocess

# Define the parameter values
batch_size = [128]
learning_rate = [0.001, 0.0001]
epochs = [10]
hidden_size = [86, 128]
num_hidden_layers = [1, 4]
num_attention_heads = [2, 4, 8]
dropout = [0.1, 0.2]

# Generate all configurations
configurations = list(itertools.product(
    batch_size,
    learning_rate,
    epochs,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    dropout
))

# Convert configurations into dictionaries
config_list = [
    {
        "batch_size": config[0],
        "learning_rate": config[1],
        "epochs": config[2],
        "hidden_size": config[3],
        "num_hidden_layers": config[4],
        "num_attention_heads": config[5],
        "dropout": config[6]
    }
    for config in configurations
]

# Split configurations into chunks for parallel runs
splits = [[] for _ in range(5)]
for i, config in enumerate(config_list):
    splits[i % 5].append(config)

# Map output path parameter to directory
output_path_map = {
    "rc": "./outputs/fine_tuning_rc",
    "np": "./outputs/fine_tuning_np",
    "lp": "./outputs/fine_tuning_lp",
    "mp": "./outputs/fine_tuning_mp",
    "sp": "./outputs/fine_tuning_sp"
}

def execute_cli(config, output_dir):
    """Runs the cli.py script with the given configuration."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "hyperparameters.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Ensure all required parameters are included in the CLI command
    cli_command = [
        "python", "cli.py",
        "--output_path", output_dir,
        "--batch_size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--dropout", str(config["dropout"]),
        "--lr", str(config["learning_rate"]),
        "--hidden_size", str(config["hidden_size"]),
        "--layers", str(config["num_hidden_layers"]),
        "--heads", str(config["num_attention_heads"]),
        "--model_type", "mamba",
        "--dataset_id", "physionet2012"
    ]

    print(f"Running configuration: {config}")  # Debug

    try:
        subprocess.run(cli_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running configuration {config}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning configurations.")
    parser.add_argument("--split", type=str, required=True, choices=output_path_map.keys())
    args = parser.parse_args()

    output_path = output_path_map[args.split]
    split_index = list(output_path_map.keys()).index(args.split)
    selected_split = splits[split_index]

    for i, config in enumerate(selected_split):
        output_dir = os.path.join(output_path, f"conf_{i + 1}")
        execute_cli(config, output_dir)