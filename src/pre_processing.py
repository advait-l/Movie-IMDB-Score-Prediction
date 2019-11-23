import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import StandardScaler

def Encoder(df):
    labelencoder = LabelEncoder()
    df['color'] = labelencoder.fit_transform(df['color'])
    print(df.head(10))
    onehotencoder = OneHotEncoder(categorical_features=[1])
    df = onehotencoder.fit_transform(df).toarray()
    return df

def remove_duplicates(df):
    df.drop_duplicates(subset="movie_title", keep= 'first', inplace = True)
    return df

def fillNullValues(df):
    # Fill "unknown" for missing values of color and director_name
    df.color.fillna(value="unknown", inplace=True)
    df.director_name.fillna(value="unknown", inplace=True)

    
    #numeric_features = df._get_numeric_data().columns.values.tolist()
    #numeric_features.remove("title_year")
    


    #numeric_features = df._get_numeric_data().columns.values.tolist()
    #numeric_features.remove("title_year")


    # Imputer : Fill the missing values with the median value for all the numeric features.
    #imp=SimpleImputer(missing_values = np.nan,strategy="median")
    #imp = imp.fit(df[numeric_features])
    #df[numeric_features] = imp.transform(df[numeric_features])

    # Standard Scaler : Scale all the numeric values to the range of values such that mean = 0 and variance = 1
    #scl=StandardScaler()
    #df[numeric_features]=scl.fit_transform(df[numeric_features])
    return df

if __name__ == "__main__":
    df = pd.read_csv('../movie_metadata/processed_data.csv')
    #print(df.isnull().sum())
    #print(df.info())
    df = fillNullValues(df)
    #print("###########################################################################################")
    #print(df.isnull().sum())
    #df = remove_duplicates(df)
    df = Encoder(df)
    print(df)



    #df.to_csv('../movie_metadata/processed_data.csv')