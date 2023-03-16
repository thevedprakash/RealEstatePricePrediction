import pandas as pd

def load_data(path):
    df = pd.read_csv(path, on_bad_lines='skip')
    print('data uploaded!')
    return df

df = load_data('/home/ris/PycharmProjects/pythonProject/USA-housing/dags/utils/train.py')

