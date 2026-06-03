import pandas as pd

def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path).drop('id', axis = 1)



