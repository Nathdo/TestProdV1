from pathlib import Path
import yaml
import logging

from ingestion import read_data
from preprocessing import split_data, scale_data

logging.basicConfig(level=logging.INFO)

# Data Path
# Path(__file__) => Absolute file Address
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "riceClassification.csv"

# Config Path
YAML_PATH = BASE_DIR / "config.yaml"
with open(file=YAML_PATH, mode="r") as f:
    config = yaml.safe_load(f)


if __name__ == "__main__":
    logging.info("Read Data...")
    data = read_data(path=DATA_PATH)
    logging.info("Split Data...")
    train_data, val_data, test_data = split_data(data=data, config=config)
    logging.info("Scaled Data...")
    train_data_scaled, val_data_scaled, test_data_scaled = scale_data(
        train_data=train_data, val_data=val_data, test_data=test_data
    )

    print()
    print("Data shape", data.shape)
    print("Training Data shape", train_data.shape)
    print(train_data.head(3))
    print()
    print(train_data_scaled.head(3))
