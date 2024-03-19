import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocessing(df):
    """ imputes missing values for swimmingpool -> they get filled up with """    
    # _____imputing missing values for swimmingpool____
    # Define the imputer to replace missing values with 0.0
    constant_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
    # Define the column containing missing values
    columns_with_missing_values = ['swimming_pool']
    # Impute missing values in X_train
    df[columns_with_missing_values] = constant_imputer.fit_transform(df[columns_with_missing_values])

    # _____one-hot encoding for kitchen_type____
    df = pd.get_dummies(df, columns=["kitchen_type"], prefix="kitchen_type")

    # _____one-hot encoding for state_of_building____
    df = pd.get_dummies(df, columns=["state_of_building"], prefix="state_of_building")

    # _____one-hot encoding for property_subtype____
    df = pd.get_dummies(df, columns=["property_subtype"], prefix="property_subtype")

    # _____one-hot encoding for province____
    df = pd.get_dummies(df, columns=["province"], prefix="province")

    return df