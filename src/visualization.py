import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# import pandas.rpy.common as com

def imdbScoreHistogram(df):
    labels = df["imdb_score"]
    plt.hist(labels, bins=20)
    plt.title("Distribution of the IMDB ratings")
    plt.savefig("visualizations/IMDB-Score-Histogram.png")

def language_histogram(df):
    language = df2["language"]
    figure = pd.Series(language).value_counts().plot(kind='bar', figsize=(12, 10)).get_figure()
    figure.savefig("visualizations/Language-Histogram.png")

def Country_histogram(df):
    country = df2["country"]
    figure = pd.Series(country).value_counts().plot(kind='bar', figsize=(12, 10)).get_figure()
    figure.savefig("visualizations/Country-Histogram.png")

def Colour_histogram(df):
    colour = df2["color"]
    figure = pd.Series(colour).value_counts().plot(kind='bar', figsize=(12, 14)).get_figure()
    figure.savefig("visualizations/Colour-Histogram.png")

def Content_rating_histogram(df):
    content_ratings = df2["content_rating"]
    figure = pd.Series(content_ratings).value_counts().plot('bar').get_figure()
    figure.savefig("visualizations/Content_ratings-Histogram.png")
def face_num_histogram(df):
    labels = df["facenumber_in_poster"]
    plt.hist(labels, bins=20)
    plt.title("Distribution of the facenumbers")
    plt.savefig("visualizations/Facenumber-Histogram.png")

def pairplot(df2):

    sns.pairplot(df2).savefig("visualizations/pairplots.png")
    # print( df.head())
    # sns.pairplot(data = df[["gross", "aspect_ratio"]], hue="Survived", dropna=True)
    #sns.pairplot(df)
def pairplots_specific(df2):

    sns.pairplot(df2, vars = ["gross", "aspect_ratio"]).savefig("visualizations/pairplot1.png")

def correlation_matrix(df2):
    # corr = df2.corr()
    # sns.set(context = "paper", font = "monospace")
    # svm = sns.heatmap(corr,linewidths=.5)
    # figure = svm.get_figure()
    # figure.savefig("visualizations/corr_matrix.png")

    # f = plt.figure(figsize=(19, 15))
    # plt.matshow(df2.corr(), fignum=f.number)
    # plt.xticks(range(df2.shape[1]), df2.columns, fontsize=14)
    # plt.yticks(range(df2.shape[1]), df2.columns, fontsize=14, rotation = 90)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.title('Correlation Matrix', fontsize=16)
    # plt.savefig("visualizations/Correlation-Matrix.png")

    # load the R package ISL
    # calculate the correlation matrix
    corr = df2.corr()
    plt.figure(figsize=(18, 18))
    svm = sns.heatmap(corr, annot = True, vmin=-1, vmax = 1, cmap='coolwarm')
    figure = svm.get_figure()
    figure.savefig("visualizations/corr_matrix.png")

if __name__ == "__main__":
    df = pd.read_csv("movie_metadata/movie_metadata.csv")
    df2 = pd.read_csv("movie_metadata/movie_metadata.csv")
    df2.dropna(inplace = True)

    # correlation_matrix(df2)
    # Colour_histogram(df2)
    # language_histogram(df2)
    # face_num_histogram(df2)
    correlation_matrix(df2)
    # Country_histogram(df2)
    # Content_rating_histogram(df2)
    # imdbScoreHistogram(df)

    # print(sum(duplicated(df)))
