import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print('data uploaded!')
    return df

if __name__ == "__main__":
    path = '/home/ris/PycharmProjects/pythonProject/USA-housing/data/train_validate.csv'
    df = load_data(path)
    print(df.columns)