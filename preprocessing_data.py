# this file works for the dataset.excel in current directory

import ast
import pandas as pd
import numpy as np


class Preprocessing:
    CATEGORY_FILE_PATH = './category.txt'

    def __init__(self):
        self.category_unique = dict()

    def load_categories(self):
        """load self.category_unique from file"""
        f = open(self.CATEGORY_FILE_PATH, "r")
        data = f.read()
        self.category_unique = ast.literal_eval(data)

    def save_categories(self):
        """save self.category_unique in file"""
        f = open(self.CATEGORY_FILE_PATH, "w")
        f.write(str(self.category_unique))

    def make_categories_data(self, df: pd.DataFrame, features: list):
        """
        fill self.category_unique
        :param df: dataframe of dataset
        :param features: a list of name of features that want to behave like categorical data
        :return: None
        """
        for each in features:
            values = list(df[each].unique().toarray())
            values.append('other')
            self.category_unique[each] = values

    def delete_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        delete features
        :param df: dataframe of dataset
        :param features: a list of name of features that want to delete form dataframe
        :return: dataframe
        """
        for each in features:
            df.drop(each, axis='columns', inplace=True)
        return df

    def convert_date_to_year(self, df: pd.DataFrame, column_name: str, delete_date_col=True) -> pd.DataFrame:
        """
        convert date column to year of date
        :param df: dataframe of dataset
        :param column_name: name of date column
        :param delete_date_col: a boolean to delete the df[column_name]
        :return: dataframe
        """
        new_name = column_name + " Year"
        df[new_name] = pd.DatetimeIndex(df[column_name]).year
        if delete_date_col:
            df = self.delete_features(df, [column_name, ])
        return df

    def add_between_two_date(self, df: pd.DataFrame, start_col_name: str, finish_col_name: str) -> pd.DataFrame:
        """
        :param df: dataframe of dataset
        :param start_col_name: the name of start date column
        :param finish_col_name: the name of finish date column
        :return: dataframe with days feature
        """
        df['days'] = (df[finish_col_name] - df[start_col_name]) / np.timedelta64(1, 'D')
        return df

    def 