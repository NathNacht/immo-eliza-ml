

def preprocessing(df, prop):
    """ Preprocesses features for house and app
        Imputes missing values for swimmingpool -> they get filled up with 0
        Label encoding for a few categorical features
        Removing columns that are not needed
    """
    # Define features
    target_column_to_drop = ['price']
    X = df.drop(columns=target_column_to_drop, axis=1)

    if prop =='house':
        # Define columns to drop    
        columns_to_drop = ['property_id', 'latitude', 'longitude', 'property_type', 'type_of_sale', 'fully_equipped_kitchen', 'locality_name', 'main_city']
    elif prop =='app':
        columns_to_drop = ['property_id', 'latitude', 'longitude', 'property_type', 'type_of_sale', 'fully_equipped_kitchen', 'locality_name', 'main_city', 'surface_of_good']

    X = X.drop(columns=columns_to_drop, axis=1)

    return X