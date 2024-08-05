import pandas as pd

def load_data(file_path='H:/My Drive/BISI II/Data Science/Term Assignments/UCLA - Neural_Networks_Solution/data/Admission.csv') -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_data(data):
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    return data