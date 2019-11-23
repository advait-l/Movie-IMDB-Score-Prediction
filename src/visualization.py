import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def imdbScoreHistogram(df):
    labels = df["imdb_score"]
    plt.hist(labels, bins=20)
    plt.title("Distribution of the IMDB ratings")
    plt.savefig("visualizations/IMDB-Score-Histogram.png")

if __name__ == "__main__":
    df = pd.read_csv("movie_metadata/movie_metadata.csv")
    imdbScoreHistogram(df)
