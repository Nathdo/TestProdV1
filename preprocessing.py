import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler


def split_data(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''  
    Split Data into Training, Validation & Test Set
    '''
    train_data, val_data = train_test_split(data, 
                                        stratify = data.Class, 
                                        test_size = config['split']['VALID_SIZE'], 
                                        random_state = 42)

    val_data, test_data = train_test_split(val_data, 
                                        stratify = val_data.Class, 
                                        test_size = config['split']['TEST_SIZE'], 
                                        random_state = 42)
                
    return train_data, val_data, test_data



def scale_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Standardization => Z = (X - Mu) / Std
    '''
    cols = train_data.drop('Class', axis = 1).columns

    scaler = StandardScaler()

    train_scaled_numpy = scaler.fit_transform(train_data[cols])     # The Scaler calculates the average and std of the Train AND normalizes the Train
    val_scaled_numpy = scaler.transform(val_data[cols])             # The Scaler uses the STORED AVG to normalize val & test
    test_scaled_numpy = scaler.transform(test_data[cols])

    # From 2D numpy Arrays to pandas DataFrame
    train_data_scaled = pd.DataFrame(train_scaled_numpy, columns = cols, index = train_data.index)
    val_data_scaled = pd.DataFrame(val_scaled_numpy, columns = cols, index = val_data.index)
    test_data_scaled = pd.DataFrame(test_scaled_numpy, columns = cols, index = test_data.index)

    # Adding again Class column
    train_data_scaled['Class'] = train_data['Class']
    val_data_scaled['Class'] = val_data['Class']
    test_data_scaled['Class'] = test_data['Class']

    return train_data_scaled, val_data_scaled, test_data_scaled

    