from fraudbench.config import PROJECT_ROOT, load_config
from fraudbench.data import load_splits
from fraudbench.training import save_processed

if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "training.json")
    splits, _ = load_splits(config)
    destination = PROJECT_ROOT / "preprocessed_df" / "model_data.csv.gz"
    save_processed(splits, destination)
    print(f"Saved {destination}")
