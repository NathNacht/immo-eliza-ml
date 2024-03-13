import pandas as pd

df_house = pd.read_csv('./data/clean_house.csv')
df_app = pd.read_csv('./data/clean_house.csv')


def dtype_conversions(df):
    # List of columns to convert to integer
    columns_to_convert = ['number_of_rooms', 'living_area', 'fully_equipped_kitchen', 'furnished', 
                      'terrace', 'terrace_area', 'garden', 'garden_area', 'surface_of_good', 
                      'number_of_facades', 'swimming_pool']

    # Convert specified columns to integer type
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce').fillna(pd.NA)
    df[columns_to_convert] = df[columns_to_convert].astype('Int64')
    return df


def drop_unnecessary_columns(df):
    """ This function drops unwanted columns from the dataset for both houses and apartments"""
    # List of columns to drop
    columns_to_drop = ['property_id', 'latitude', 'longitude', 'property_type', 'type_of_sale']
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df


def drop_unnecessary_columns_app(df):
    """ This function drops unwanted columns from the dataset for apartments"""
    # List of columns to drop
    app_columns_to_drop = ['surface_of_good']
    df.drop(app_columns_to_drop, axis=1, inplace=True)
    return df


print("--- Data type conversions of the house and app dataset ---")
house = dtype_conversions(df_house)
app = dtype_conversions(df_app)

# print(house.dtypes)
# print(app.dtypes)

print("--- Drop unnecessary columns ---")
drop_unnecessary_columns(df_house)
drop_unnecessary_columns(df_app)
drop_unnecessary_columns_app(df_app)

print(house.dtypes)
print(app.dtypes)