# Model card

## Project context

This model was created to predict the price of a house or apartment in Belgium starting from a set of given features.

## Data

* Dataset Name apartments: clean_app.csv
* Dataset Name houses: clean_houses.csv

The fields of the raw files are as follows:

* property_id
* locality_name
* postal_code
* latitude
* longitude
* property_type (house or apartment)
* property_subtype (bungalow, chalet, mansion, ...)
* price
* type_of_sale (note: exclude life sales)
* number_of_rooms (Number of rooms)
* living_area (Living area (area in m²))
* kitchen_type
* fully_equipped_kitchen (0/1)
* furnished (0/1)
* open_fire (0/1)
* terrace
* terrace_area (area in m² or null if no terrace)
* garden
* garden_area (area in m² or null if no garden)
* surface_of_good
* number_of_facades
* swimming_pool (0/1)
* state_of_building (new, to be renovated, ...)
* main_city
* province

## Model details

* Model Name: RandomForestRegressor
Model Version: v0.1
Model Type: Tree-based model
Date: 20-03-2024
Author: Nathalie Nachtergaele

* Other model tested: Linear Regression


## Performance

Performance metrics for the various models tested

Metrics:
| Property Type               | Test/Training | model type | extra info    | Mean R²   | RMSE      | CV_score |
|-----------------------------|---------------|------------|---------------|-----------|-----------|----------|
| House                       | Training      | LR         | with outliers | 0.38      |           |          |
| House                       | Test          | LR         | with outliers | 0.46      | 211804    | 0.26     |
| House                       | Training      | RF         | with outliers | 0.96      |           |          |
| House                       | Test          | RF         | with outliers | 0.73      | 150468    | 0.72     |
| Appartment                  | Training      | LR         | with outliers | 0.43      |           |          |
| Appartment                  | Test          | LR         | with outliers | 0.39      | 273950    | 0.42     |
| Appartment                  | Training      | RF         | with outliers | 0.96      |           |          |
| Appartment                  | Test          | RF         | with outliers | 0.69      | 194765    | 0.77     |

Performance was tuned using cross-validation, keeping outliers in the training set or removing them.

## Limitations

Prediction only works on well structured data (check data/prediction/new_app_data.csv and new_house_data.csv for the structure)

## Usage

* To train the model, run the train.py script 
```bash
$ python3 train.py
```

* Two model .pkl files will be created (one for houses and one for apartments)

* Run the predict.py script to generate predictions
For apartments
```bash
$ python3 predict.py -i "data/predictions/new_app_data.csv" -o "data/predictions/app_predicted.csv" -p "app"
```
For Houses
```bash
$ python3 predict.py -i "data/predictions/new_house_data.csv" -o "data/predictions/house_predicted.csv" -p "house"
```

## Maintainers

Nathalie Nachtergaele