from load import load_data
from pre_processing import sanity_check, handling_missing_values, handling_outliers, handling_categorical_cols
from pre_processing import remove_bed_outlier, remove_bath_outlier, remove_state_outlier, filter_predictor_columns
import pickle
import joblib

import warnings
warnings.filterwarnings("ignore")




def encode_predict_input(df, encoded_dict):
    '''
    This function encodes categorical values with same values as training encoded values.
    Input:
        df : DataFrame
        encoded_dict : Category encoded dictionary
    returns :None
    '''
    encoded_cols = ["state"]

    label_dict = encoded_dict
    for col in encoded_cols:
        df[col] = df[col].replace(label_dict)
    return df



def preprocess_and_predict(df, encoded_dict):
    '''
      This function takes in new dataframe or row of observation and generate all features
    Input :
        df : DataFrame or row of observation
        encoded_dict : Dictonary created while training for Categorical Encoded Value.
    '''
    df = sanity_check(df, mode='predict')
    df = handling_missing_values(df)
    df = handling_outliers(df, mode = 'predict')
    df = encode_predict_input(df, encoded_dict)

    X = filter_predictor_columns(df)
    return X


if __name__ == "__main__":
    print("Loading the TestData.")
    # Load data (deserialize)
    with open('/home/ris/PycharmProjects/pythonProject/USA-housing/models/encoded.pickle', 'rb') as handle:
        encoded_dict = pickle.load(handle)

    print('encoded dict train_data',encoded_dict)

    model_path = "/home/ris/PycharmProjects/pythonProject/USA-housing/models/knearest.pickle"
    saved_model = joblib.load(model_path)

    test_path = '/home/ris/PycharmProjects/pythonProject/USA-housing/data/test.csv'
    test_df = load_data(test_path)
    test_input = preprocess_and_predict(test_df, encoded_dict)


    print(test_input.head())
    saved_model.predict(test_input)
    print(saved_model.predict(test_input))