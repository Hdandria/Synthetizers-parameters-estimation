import argparse
import pandas as pd
import wandb


def export_run_data(run_path: str, output_csv: str):
    """
    Exports history data from a wandb run to a CSV file.
    """
    api = wandb.Api()
    run = api.run(run_path)

    # scan_history() retrieves all logged data points
    history = run.scan_history()
    df = pd.DataFrame(history)

    # Save as CSV
    df.to_csv(output_csv, index=False)
    print(f"Successfully exported history from {run_path} to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export wandb run history to CSV.")
    parser.add_argument(
        "--run_path",
        type=str,
        default="paindespistes-t-l-com-paris/synth-prediction/myn9w8e5",
        help="The wandb run path (entity/project/run_id)",
    )
    parser.add_argument(
        "--output", type=str, default="run_data.csv", help="Path to the output CSV file"
    )

    args = parser.parse_args()
    export_run_data(args.run_path, args.output)
