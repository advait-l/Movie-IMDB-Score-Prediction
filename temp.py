import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def handlingValues(df):
    # Dropped null valued rows of gross and budget columns
    df = df.dropna(axis=0, subset=["gross", "budget"])

    # Dropped columns aspect_ratio, language, color, plot keywords, director's name, actors' name, genres, movie-imdb-link
    df = df.drop(columns=["aspect_ratio","language","color", "plot_keywords", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres","movie_imdb_link"])
    
    # Countries other than USA and UK labelled as others
    country = ["USA", "UK"]
    other = []
    for  i in df["country"]:
        if(i not in country):
            other.append(i)
    df["country"] = df["country"].replace(to_replace = other, value="Other")

    # Face number 0s filled with mean
    df["facenumber_in_poster"] = df["facenumber_in_poster"].replace(0, np.nan)
    
    # Filled null values with the mean of the column 
    df["facenumber_in_poster"] = df["facenumber_in_poster"].fillna(df["facenumber_in_poster"].mean())
    df["num_critic_for_reviews"] = df["num_critic_for_reviews"].fillna(df["num_critic_for_reviews"].mean())
    df["duration"] = df["duration"].fillna(df["duration"].mean())
    df["actor_1_facebook_likes"] = df["actor_1_facebook_likes"].fillna(df["actor_1_facebook_likes"].mean())
    df["actor_2_facebook_likes"] = df["actor_2_facebook_likes"].fillna(df["actor_2_facebook_likes"].mean())
    df["actor_3_facebook_likes"] = df["actor_3_facebook_likes"].fillna(df["actor_3_facebook_likes"].mean())

    # Profit column
    df["profit"] = df["gross"] - df["budget"]

    return df

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
    #onehotencoder = OneHotEncoder(categorical_features=[i])
    #df = onehotencoder.fit_transform(df).toarray()

    df = pd.get_dummies(df, columns=['content_rating'])

    return df

if __name__ == "__main__":
    df = pd.read_csv("movie_metadata/movie_metadata.csv")
    print(df.isnull().sum())
    #print(df.info())
    df = handlingValues(df)
    df = remove_duplicates(df)
    df = ContentRating(df)
    print(df.isnull().sum())
    print(df)
    df.to_csv('movie_metadata/processed_data.csv')
