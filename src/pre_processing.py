import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 


def fillNullValues(df):
    # Fill "unknown" for missing values of color and director_name
    df.color.fillna(value="unknown", inplace=True)
    df.director_name.fillna(value="unknown", inplace=True)
    
    numeric_features = df._get_numeric_data().columns.values.tolist()
    numeric_features.remove("title_year")
    
    # Imputer : Fill the missing values with the median value for all the numeric features.
    imp=SimpleImputer(missing_values = np.nan,strategy="median")
    imp = imp.fit(df[numeric_features])
    df[numeric_features] = imp.transform(df[numeric_features])

    # Standard Scaler : Scale all the numeric values to the range of values such that mean = 0 and variance = 1
    scl=StandardScaler()
    df[numeric_features]=scl.fit_transform(df[numeric_features])
    return df

if __name__ == "__main__":
    df = pd.read_csv("movie_metadata/movie_metadata.csv")
    print(df.isnull().sum())
    #print(df.info())
    df = fillNullValues(df)
    print("###########################################################################################")
    print(df.isnull().sum())
    print(df)

