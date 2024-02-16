import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

def load_data(file_path):
    df = pd.read_excel(file_path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.rename(columns={'drugsID':'drug_id', 'order_numbers': 'amount', 'city': 'district'}, inplace=True)
    df.set_index('time_step', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def feature_engineering(df):
    buffer = pd.Series(df.index)
    df['month'] = buffer.dt.month
    df['year'] = buffer.dt.year
    df['season'] = buffer.dt.month % 4 + 1
    df = pd.get_dummies(df, columns=['district'], drop_first=False)
    df['drug_id'] = pd.Categorical(df['drug_id']).codes
    return df

def split_data(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=0)

def train_model(X_train, y_train):
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

def tune_model(X, y):
    random_grid = {
        'loss': ['squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'],
        'learning_rate': [0.1, 0.03, 0.003],
        'max_iter': [50, 100, 200, 500],
        'max_leaf_nodes': [7, 14, 21, 28, 31, 50],
        'max_depth': [-1, 3, 5],
        'min_samples_leaf': [1, 2, 4, 10, 20],
        'l2_regularization': [0.0, 0.1, 0.5, 1.0],
    }
    hgbr = HistGradientBoostingRegressor(random_state=42)
    hgbr_random = RandomizedSearchCV(
        estimator=hgbr,
        param_distributions=random_grid,
        n_iter=100,
        cv=5,
        scoring='r2',
        verbose=10,
        random_state=42,
        n_jobs=-1
    )
    hgbr_random.fit(X, y)
    return hgbr_random

def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    print("Train")
    print_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    print("Test")
    print_score(y_test, test_pred)

# Modify the print_score function to return the computed scores
def print_score(y_test, y_predict):
    r2 = r2_score(y_test, y_predict)
    n = len(y_test)
    p = 1
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mae = mean_absolute_error(y_test, y_predict)
    
    return {
        'R-squared': r2,
        'Adjusted R-Squared': adjusted_r2,
        'RMSE': rmse,
        'MAE': mae
    }


