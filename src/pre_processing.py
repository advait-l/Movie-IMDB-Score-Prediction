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
    #onehotencoder = OneHotEncoder(categorical_features=[])
    #df = onehotencoder.fit_transform(df).toarray()
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


def rating(df):
    df['content_rating'].replace(to_replace = ['M', 'GP', 'TV-PG','TV-Y7'] , value = 'PG', inplace = True)
    df['content_rating'].replace(to_replace = ['X'] , value = 'NC-17', inplace = True)
    df['content_rating'].replace(to_replace = ['Approved','Not Rated', 'Passed', 'Unrated','TV-MA'] , value = 'R', inplace = True)
    df['content_rating'].replace(to_replace = ['TV-G','TV-Y'] , value = 'G', inplace = True)
    df['content_rating'].replace(to_replace = 'TV-14' , value = 'PG-13', inplace = True)

    return df
if __name__ == "__main__":
    
    #When Running for the first time
    #df = pd.read_csv('movie_metadata/movie_metadata.csv')

    #When updating the code
    df = pd.read_csv('movie_metadata/processed_data.csv')
    
    #df = fillNullValues(df)
    #df = remove_duplicates(df)
    #df = Encoder(df)
    #df = drop(df)
    df = rating(df)
    print(df)
    x = df.content_rating.unique()
    print(x)
    print(len(x))
    df.to_csv('movie_metadata/processed_data.csv')