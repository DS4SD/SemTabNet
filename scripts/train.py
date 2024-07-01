import argparse
from pathlib import Path
from src.models.t5.trainer import T5Trainer

def main():
    parser = argparse.ArgumentParser(
        description="Train T5 Model"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Path to the run_config file for current run",
        required=True
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        help="Specify type of model (t5 or other)",
        required=True
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory path.",
        required=True
    )    
    args = parser.parse_args()

    if(args.model_type == "t5"):
        trainer = T5Trainer(
            run_config_file = Path(args.config_file),
            output_dir = Path(args.output_dir)
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Train the model
    trained_model = trainer.train()

if __name__ == "__main__":
    main()
