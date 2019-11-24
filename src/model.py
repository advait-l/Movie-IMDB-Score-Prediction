import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.metrics import  mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib

def random_forest(x1,y1,x2,y2):
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(x1, np.ravel(y1, order='C'))
    rfcpred = rfc.predict(x2)
    cnf_matrix = metrics.confusion_matrix(y2, rfcpred)
    #print(cnf_matrix)

    joblib.dump(rfc,'pickle/RF.pkl')

    return metrics.accuracy_score(y2,rfcpred)

def gradient_boosting(x1,y1,x2,y2):
    gbcl = GradientBoostingClassifier(n_estimators=50, learning_rate=0.09, max_depth=5)
    gbcl = gbcl.fit(x1, np.ravel(y1, order='C'))
    gbcl_pred = gbcl.predict(x2)
    cnf_matrix = metrics.confusion_matrix(y2, gbcl_pred)
    #print(cnf_matrix)

    joblib.dump(gbcl,'pickle/gbcl.pkl')

    return metrics.accuracy_score(y2,gbcl_pred)


if __name__=="__main__":
    rf_accuracy = []
    gb_accuracy = []
    for i in range(10):
        df = pd.read_csv('movie_metadata/processed_data.csv')
        df_train, df_test = train_test_split(df, test_size = 0.2)
        y_train = df_train["quality"]
        x_train = df_train.drop(columns = ["quality", "movie_title"])
        y_test = df_test["quality"]
        x_test = df_test.drop(columns = ["quality", "movie_title"])
        print("Iteration : " + str(i))
        rf_accuracy.append(random_forest(x_train, y_train, x_test, y_test))
        gb_accuracy.append(gradient_boosting(x_train, y_train, x_test, y_test))
    print("RF max : " + str(max(rf_accuracy)))
    print("RF mean : " + str(sum(rf_accuracy)/len(rf_accuracy)))
    print("GB max : " + str(max(gb_accuracy)))
    print("GB mean : " + str(sum(gb_accuracy)/len(gb_accuracy)))
