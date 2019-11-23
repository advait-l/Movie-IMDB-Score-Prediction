import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def imdbScoreHistogram(df):
    labels = df["imdb_score"]
    plt.hist(labels, bins=20)
    plt.title("Distribution of the IMDB ratings")
    plt.savefig("visualizations/IMDB-Score-Histogram.png")
    sns.pairplot(df)

def pairplot(df2):

    sns.pairplot(df2).savefig("visualizations/pairplots.png")
    # print( df.head())
    # sns.pairplot(data = df[["gross", "aspect_ratio"]], hue="Survived", dropna=True)
        # sns.pairplot(df)
def pairplots_specific(df2):

    sns.pairplot(df2, vars = ["gross", "aspect_ratio"]).savefig("visualizations/pairplot1.png")


if __name__ == "__main__":
    df = pd.read_csv("movie_metadata/movie_metadata.csv")
    df2 = pd.read_csv("movie_metadata/movie_metadata.csv")
    df2.dropna(inplace = True)
    imdbScoreHistogram(df)
    pairplot(df2)
    pairplots_specific(df2)
