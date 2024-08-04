import pandas as pd

def load_data(file_path='H:/My Drive/BISI II/Data Science/Term Assignments/UCLA - Neural_Networks_Solution/data/Admission.csv') -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Implement data preprocessing steps here
    return df