from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

def dt_fit(X_train, y_train, task='Classification'):
    if task == 'Classification':
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def logistic_regression_fit(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def svm_fit(X_train, y_train, task='Classification'):
    if task == 'Classification':
        model = SVC(random_state=42)
    else:
        model = SVR()
    model.fit(X_train, y_train)
    return model

def rf_fit(X_train, y_train, task='Classification'):
    if task == 'Classification':
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def knn_fit(X_train, y_train, task='Classification'):
    if task == 'Classification':
        model = KNeighborsClassifier()
    else:
        model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    return model

def naive_bayes_fit(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def ann_fit(X_train, y_train, task='Classification'):
    if task == 'Classification':
        model = MLPClassifier(random_state=42, max_iter=500)
    else:
        model = MLPRegressor(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    return model

def linreg_fit(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def kmeans_fit(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr = classification_report(y_test, y_pred, output_dict=True)
    return {"accuracy": acc, "confusion_matrix": cm, "classification_report": cr}

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

