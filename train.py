import pandas as pd
import numpy as np
import os
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from preprocessing import preprocessing


def preprocess_X(df,prop):
 
    # Define features
    target_column_to_drop = ['price']
    X = df.drop(columns=target_column_to_drop, axis=1)

    if prop =='house':
        # Define columns to drop    
        columns_to_drop = ['property_id', 'latitude', 'longitude', 'property_type', 'type_of_sale', 'fully_equipped_kitchen', 'locality_name', 'main_city']
        X = X.drop(columns=columns_to_drop, axis=1)
    elif prop =='app':
        columns_to_drop = ['property_id', 'latitude', 'longitude', 'property_type', 'type_of_sale', 'fully_equipped_kitchen', 'locality_name', 'main_city', 'surface_of_good']
        X = X.drop(columns=columns_to_drop, axis=1)
        print(type(X))

    X = preprocessing(X)
    return X


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # _____imputing -1 for missing values for number_of_rooms, terrace_area, garden_area, furnished, garden, terrace, number_of_facades____
    constant_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
    colums_to_impute = ['number_of_rooms', 'terrace_area', 'garden_area', 'furnished', 'garden', 'terrace', 'number_of_facades']
    X_train[colums_to_impute] = constant_imputer.fit_transform(X_train[colums_to_impute])
    X_test[colums_to_impute] = constant_imputer.fit_transform(X_test[colums_to_impute])

    param_grid = {
    'max_depth': [10, 15, 20, 25, 30, 40],
    'n_estimators': [100, 200]
    }

    grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=3, verbose=3)
    grid.fit(X, y)
    
    best_max_depth = grid.best_params_['max_depth']
    best_n_estimators = grid.best_params_['n_estimators']
    
    model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth)
    model.fit(X_train, y_train)
    
    #____show the choosen max_depth____
    print("randomforest choosen max_depth: ", best_max_depth)

    #____show the choosen n_estimators____
    print("randomforest choosen n_estimators: ", best_n_estimators)

    #____show the score____
    print("randomforest training score: ", model.score(X_train, y_train))

    #____test the model____
    y_prediction = model.predict(X_test)

    #____show the score____
    print("randomforest testscore: ", model.score(X_test, y_test))

    mse = MSE(y_test, y_prediction)
    rmse = math.sqrt(mse)
    print("rmse: ",rmse)

    print("cross val score: ", cross_val_score(model, X_train, y_train).mean())

    return model


# _____Get the current directory_____
current_dir = os.getcwd()


# _____Define relative file path for clean house input file____
clean_huis_te_koop_path = os.path.join(current_dir, "data", "clean_house.csv")

#____Creation pickle file for house____
df_house = pd.read_csv(clean_huis_te_koop_path, sep=",")
y = df_house['price']
X = preprocess_X(df_house, prop='house')
model = train_model(X, y)
print("For HOUSE rfr_house_model.pkl has been created")
with open('rfr_house_model.pkl', 'wb') as file:
    pickle.dump(model, file=file)


# # _____Define relative file path for clean app input file____
# clean_apartement_te_koop_path = os.path.join(current_dir, "data", "clean_app.csv")

# #____Creation pickle file for app____
# df_app = pd.read_csv(clean_apartement_te_koop_path, sep=",")
# y = df_house['price']
# X = define_X_and_y(df_app, prop='app')
# model = train_model(X, y)
# print("For APP rfr_app_model.pkl has been created")
# with open('rfr_app_model.pkl', 'wb') as file:
#     pickle.dump(model, file=file)