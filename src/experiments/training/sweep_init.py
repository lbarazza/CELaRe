# Leonardo Barazza, acse-lb1223

import yaml
import wandb
import tyro
from dataclasses import dataclass

@dataclass
class SweepArgs:
    config_file: str  # Path to the YAML configuration file
    project: str      # Name of the W&B project
    entity: str       # Name of the W&B entity

def main(sweep_args: SweepArgs):
    # Load the sweep configuration from the YAML file
    with open(sweep_args.config_file) as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=sweep_args.project, entity=sweep_args.entity)

    # Print the sweep ID
    print(f"Sweep ID: {sweep_id}")

if __name__ == "__main__":
    sweep_args = tyro.cli(SweepArgs)
    main(sweep_args)
