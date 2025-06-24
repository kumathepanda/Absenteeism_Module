import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle

class CustomScaler(BaseEstimator,TransformerMixin):
    def __init__(self,columns , copy=True,with_mean=True,with_std=True):
        self.columns = columns
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.mean_ = None
        self.var_ = None
        
    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_= np.mean(X[self.columns])
        self.var_= np.var(X[self.columns])
        return self
    
    def transform(self,X,y=None,copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]),columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_scaled,X_not_scaled],axis=1)[init_col_order]


class absenteeism_model:

    def __init__(self, model_file, scaler_file):
        with open(model_file, "rb") as mf, open(scaler_file, "rb") as sf:
            self.reg = pickle.load(mf)
            self.scaler = pickle.load(sf)
            self.data = None

    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter=",")
        self.df_with_pred = df.copy()

        df = df.drop("ID", axis=1)
        df["Absenteeism in hours"] = np.nan  # Fix

        reason_columns = pd.get_dummies(df["Reason for Absence"], drop_first=True,dtype="int")
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        df.drop("Reason for Absence", axis=1, inplace=True)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education',
                        'Children', 'Pets', 'Absenteeism in hours',
                        'reason_type_1', 'reason_type_2', 'reason_type_3', 'reason_type_4']
        df.columns = column_names

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df["Month Values"] = df['Date'].dt.month
        df["Week of Day"] = df['Date'].dt.weekday
        df.drop("Date", axis=1, inplace=True)

        df["Education"] = df["Education"].map({1: 0, 2: 1, 3: 1, 4: 1})
        df.fillna(value=0, inplace=True)
        df.drop("Absenteeism in hours", axis=1, inplace=True)

        df.drop(["Daily Work Load Average", "Distance to Work", "Week of Day"], axis=1, inplace=True)

        self.data_preprocessed = df.copy()
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        if self.data is not None:
            return self.reg.predict_proba(self.data)[:, 1]

    def predicted_output_category(self):
        if self.data is not None:
            return self.reg.predict(self.data)

    def predicted_outputs(self):
        if self.data is not None:
            self.data_preprocessed["Probability"] = self.reg.predict_proba(self.data)[:, 1]
            self.data_preprocessed["Predictions"] = self.reg.predict(self.data)
            return self.data_preprocessed






