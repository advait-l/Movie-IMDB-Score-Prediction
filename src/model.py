import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.metrics import  mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def linear_regression(x1,y1,x2,y2):
    regr = linear_model.LinearRegression()
    regr.fit(x1, y1)
    imdb_predict = regr.predict(x2)
    print("Coefficients : \n", regr.coef_)
    print("Mean-squared-error : " + str(mean_squared_error(y2, imdb_predict)))
    print("Variance : " + str(r2_score(y2, imdb_predict)))

def random_forest(x1,y1,x2,y2):
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(x1, np.ravel(y1, order='C'))
    rfcpred = rfc.predict(x2)
    cnf_matrix = metrics.confusion_matrix(y2, rfcpred)
    print(cnf_matrix)
    print("Accuracy", metrics.accuracy_score(y2,rfcpred))

def gradient_boosting(x1,y1,x2,y2):
    gbcl = GradientBoostingClassifier(n_estimators=50, learning_rate=0.09, max_depth=5)
    gbcl = gbcl.fit(x1, np.ravel(y1, order='C'))
    gbcl_pred = gbcl.predict(x2)
    cnf_matrix = metrics.confusion_matrix(y2, gbcl_pred)
    print(cnf_matrix)
    print("Accuracy", metrics.accuracy_score(y2,gbcl_pred))


if __name__=="__main__":
    df = pd.read_csv('../movie_metadata/processed_data.csv')
    df_train, df_test = train_test_split(df, test_size = 0.2)
    y_train = df_train["quality"]
    x_train = df_train.drop(columns = ["quality", "movie_title"])
    y_test = df_test["quality"]
    x_test = df_test.drop(columns = ["quality", "movie_title"])
    
    #linear_regression(x_train, y_train, x_test, y_test)
    random_forest(x_train, y_train, x_test, y_test)
    gradient_boosting(x_train, y_train, x_test, y_test)

    print(df)
