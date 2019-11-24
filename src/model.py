import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def linear_regression(x1,y1,x2,y2):
    regr = linear_model.LinearRegression()
    regr.fit(x1, y1)
    imdb_predict = regr.predict(x2)
    print("Coefficients : \n", regr.coef_)
    print("Mean-squared-error : " + str(mean_squared_error(y2, imdb_predict)))
    print("Variance : " + str(r2_score(y2, imdb_predict)))


def PolyReg(df):
    y = df.imdb_score.values
    X = df.drop(["imdb_score","movie_title"], axis = 1)
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)
    
    lin2 = LinearRegression()

    poly =  PolynomialFeatures(degree = 2)
    X_poly = poly.fit_transform(X)

    lin2.fit(X_poly, y)

    pred = lin2.predict(poly.fit_transform(X_test))
    print("Accuracy : " + str(lin2.score(poly.fit_transform(X_test),y_test)))
    


if __name__=="__main__":
    df = pd.read_csv('movie_metadata/processed_data.csv')
    df.drop(['Unnamed: 0'], axis= 1, inplace= True)
    
    
    #linear_regression(x_train, y_train, x_test, y_test)

    PolyReg(df)
    #print(df)
