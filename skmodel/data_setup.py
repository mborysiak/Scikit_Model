import pandas as pd 
import numpy as np

class DataSetup:

    def __init__(self, data):
        self.data = data

    def Xy_split(self, y_metric, to_drop=[]):
        """Split train dataset into X and y by excluding non-numeric values

        Args:
            df (pandas.DataFrame): DataFrame containing samples features and target variable
            y_metric (str): Column name of target variable to split into y

        Returns:
            X (pandas.DataFrame): Features matrix with numeric columns
            y (pandas.DataFrame): Target vector containing corresponding measure
        """ 
        # create the y variable based on input metric, if it exists
        if y_metric in self.data.columns: 
            to_drop.append(y_metric)
            self.X = self.data.drop(to_drop, axis=1)
            self.y = self.data[y_metric]
        else: 
            self.y = None
            print(f'{y_metric} not in dataframe to create y')

        return self.X, self.y


    def Xy_split_list(self, y_metric, col_list):
        """Split train dataset into X and y by passing list of desired columns

        Args:
            df (pandas.DataFrame): DataFrame containing samples features and target variable
            y_metric (str): Column name of target variable to split into y
            col_list (str): List of column names to keep within X dataframe

        Returns:
            X (pandas.DataFrame): Features matrix with selected columns
            y (pandas.DataFrame): Target vector containing corresponding measure
        """ 
        # create the X variable based on column list
        self.X = self.data[col_list]
        
        # create the y variable based on input metric, if it exists
        if y_metric in self.data.columns: 
            self.y = self.data[y_metric]
        else: 
            self.y = None
            print(f'{y_metric} not in dataframe to create y')

        return self.X, self.y

    def return_X_y(self):
        return self.X, self.y


    def count_null(self):
        """Count null values for each column in DataFrame

        Args:
            df (pandas.DataFrame): DataFrame to count nulls in each column
        """    
        print('Null Counts:', self.X.isnull().sum()[self.X.isnull().sum() > 0])