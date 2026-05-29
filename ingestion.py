import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

# Create Path -----------------
BASE_DIR = Path.cwd().parent
DATA_PATH = BASE_DIR / 'data' / 'riceClassification.csv'
YAML_PATH = BASE_DIR / 'config.yaml'

# Connect YAML file for config
with open(file = YAML_PATH, mode = 'r') as f: config = yaml.safe_load(f)

def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH).drop('id', axis = 1)
    