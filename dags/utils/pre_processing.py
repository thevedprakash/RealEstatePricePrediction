from sklearn.preprocessing import LabelEncoder
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils.load import load_data

def sanity_check(df,mode='train'):
    '''
      This function perform sanity and check create a dataframe.
      Input:
        df : Dataframe which require sanity-check
        mode : train or predict
      return : None
    '''
    if mode == 'train':
      print('Percentage of missing values column wise\n:', df.isnull().sum()/len(df))
    # sold_date must be datetime object
    df['sold_date'] = pd.to_datetime(df['sold_date'])
    print('sanity check done!')
    return df


def handling_missing_values(df):

    df.dropna(inplace=True)
    columns_to_dropped = ['full_address', 'street', "city", 'status', 'zip_code',
                          "sold_date"]  # reasons states in EDA part.
    df.drop(columns_to_dropped, axis=1, inplace=True)
    print('handling of missing values done!')
    return df

def remove_state_outlier(value):
    if value not in ['New Jersey','Connecticut','New York','Pennsylvania','Massachusetts']:
        return 'New Jersey'
    else:
        return value

def remove_bed_outlier(value):
    if value not in [3,4,2,5]:
        return 3
    else:
        return value

def remove_bath_outlier(value):
    if value not in [1,2,3,4]:
        return 2
    else:
        return value


def handling_outliers(df, mode = 'train'):
    if mode == 'train':
        df = df[(df["price"] < 800000) & (df["house_size"] < 5000)]  # reason explained in EDA part

    df['state'] = df['state'].apply(remove_state_outlier)
    df['bath'] = df['bath'].apply(remove_bath_outlier)
    df['bed'] = df['bed'].apply(remove_bed_outlier)

    print('handling of outliers done!')
    return df


def handling_categorical_cols(df):
    """
    This function encodes a categorical column based on the basis of their order label.
    input:
        df : Input DataFrame in which encoding has to be created
        col : Column name which has to be encoded
    return:
          label encoded dict for column
    """

    object_columns = df.select_dtypes(object).columns
    for col in object_columns:
        le = LabelEncoder()
        le.fit(df[col])
        encoded_dict = dict(zip((le.classes_),le.transform(le.classes_)))
        df[col] = df[col].replace(encoded_dict)
    print('encoding of categorical columns done!')


    return df, encoded_dict

def filter_predictor_columns(df):
    '''
    This function filters predictor columns from the incoming Data
    '''
    predictor_columns = ['bed', 'bath', 'acre_lot', 'state', 'house_size']
    return df[predictor_columns]

def pre_process(df, target):
    df = sanity_check(df)
    df = handling_missing_values(df)
    df = handling_outliers(df, mode = 'train')
    df, encoded_dict= handling_categorical_cols(df)
    X = filter_predictor_columns(df)
    y = df[target]
    return X, y, encoded_dict

df = load_data('/home/ris/PycharmProjects/pythonProject/USA-housing/data/train_validate.csv')
pre_process(df, 'price')