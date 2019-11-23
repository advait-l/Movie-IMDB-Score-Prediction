import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(x1,y1,x2,y2):
    regr = linear_model.LinearRegression()
    regr.fit(x1, y1)
    imdb_predict = regr.predict(x2)
    print("Coefficients : \n", regr.coef_)
    print("Mean-squared-error : " + str(mean_squared_error(y2, imdb_predict)))
    print("Variance : " + str(r2_score(y2, imdb_predict)))

if __name__=="__main__":
    df = pd.read_csv('../movie_metadata/processed_data.csv')
    df_train, df_test = train_test_split(df, test_size = 0.2)
    y_train = df_train["imdb_score"]
    x_train = df_train.drop(columns = ["imdb_score", "movie_title"])
    y_test = df_test["imdb_score"]
    x_test = df_test.drop(columns = ["imdb_score", "movie_title"])
    linear_regression(x_train, y_train, x_test, y_test)
    print(df)
