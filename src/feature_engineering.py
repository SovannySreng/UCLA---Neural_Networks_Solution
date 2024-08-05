

import pandas as pd

def create_features(data):
    data = pd.get_dummies(data, columns=['University_Rating', 'Research'])
    return data

def split_data(data):
    x = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']
    return x, y