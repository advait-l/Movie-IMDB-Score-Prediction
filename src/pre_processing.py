import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import StandardScaler


def remove_duplicates(df):
    df.drop_duplicates(subset="movie_title", keep= 'first', inplace = True)
    return df

def ContentRating(df):
    df['content_rating'].replace(to_replace = ['M', 'GP', 'TV-PG','TV-Y7'] , value = 'PG', inplace = True)
    df['content_rating'].replace(to_replace = ['X'] , value = 'NC-17', inplace = True)
    df['content_rating'].replace(to_replace = ['Approved','Not Rated', 'Passed', 'Unrated','TV-MA'] , value = 'R', inplace = True)
    df['content_rating'].replace(to_replace = ['TV-G','TV-Y'] , value = 'G', inplace = True)
    df['content_rating'].replace(to_replace = 'TV-14' , value = 'PG-13', inplace = True)
    df['content_rating'].fillna('R',inplace = True)
    
    labelencoder = LabelEncoder()
    df['content_rating'] = labelencoder.fit_transform(df['content_rating'])

    print(df)
    i = df.columns.get_loc('content_rating') 
    onehotencoder = OneHotEncoder(categorical_features=[i])
    df = onehotencoder.fit_transform(df).toarray()
    return df



if __name__ == "__main__":
    
    #When Running for the first time
    df = pd.read_csv('movie_metadata/movie_metadata.csv')

    #When updating the code
    #df = pd.read_csv('movie_metadata/processed_data.csv')
    
    df = df.drop(columns=["aspect_ratio","language","color", "plot_keywords", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres","movie_imdb_link"])
    df = remove_duplicates(df)
    df = ContentRating(df)
    print(df)
    x = df.content_rating.unique()
    print(x)
    print(len(x))
    df.to_csv('movie_metadata/processed_data.csv')