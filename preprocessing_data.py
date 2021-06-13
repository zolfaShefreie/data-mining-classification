# this file works for the dataset.excel in current directory

import ast
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split


def get_stage(close_value, product, stage, df):
    if stage != "In Progress":
        return stage
    min_value = df[df['Product'] == product][df['Stage'] == "Won"].quantile(.1)['Close_Value']
    max_value = df[df['Product'] == product][df['Stage'] == "Won"].quantile(.9)['Close_Value']
    if min_value < close_value < max_value:
        return "Won"
    return "Lost"



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

    def make_categories_data(self, df: pd.DataFrame, features: list, add_other=True):
        """
        fill self.category_unique
        :param df: dataframe of dataset
        :param features: a list of name of features that want to behave like categorical data
        :return: None
        """
        for each in features:
            values = list(df[each].unique().toarray())
            if add_other:
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

    def change_unexpected_values(self, df: pd.DataFrame, col_name: str):
        """
        set unexpected_values (categorical feature values) to other
        :param df:
        :param col_name:
        :return:
        """
        unexpected_values = df[df[col_name] not in self.category_unique[col_name]].unique()
        if unexpected_values:
            df[col_name] = df[col_name].replace(unexpected_values, 'other')
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

    def fill_nan_by_mean_group(self, df: pd.DataFrame, group_cols: list, nan_col: str, index_name: str) -> pd.DataFrame:
        """
        :param df:
        :param group_cols: a list of col_name that want to be group and get mean
        :param nan_col: the name of col that have nan
        :param index_name: the name if index col
        :return: result
        """
        group_df = df.groupby(group_cols)[nan_col].mean().reset_index()
        group_df.columns = group_cols + ['mean_value', ]
        merged_df = pd.merge(df, group_df, on=group_cols)
        merged_df.sort_values(by=[index_name], inplace=True)
        merged_df.reset_index(inplace=True)
        df[nan_col].fillna(merged_df['mean_value'], inplace=True)
        return df

    def replace_category_data(self, apply_func, col_name: str, arg_cols: list, df: pd.DataFrame) -> pd.DataFrame:
        """
        this function just write for our dataset
        :param apply_func: the function
        :param col_name:
        :param arg_cols: a list of col_names for sending as args
        :return: result
        """
        df[col_name] = df.apply(lambda row: apply_func(row[arg_cols[0]], row[arg_cols[1]], row[arg_cols[2]], df),
                                axis=1)
        return df

    def label_encode(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        :param df:
        :param col_name:
        :return: result
        """
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.category_unique[col_name])
        df[col_name] = label_encoder.transform(df[col_name])
        return df

    def one_to_hot_encode(self, df: pd.DataFrame, col_name: str):
        one_to_hot = preprocessing.OneHotEncoder()
        one_to_hot.fit(np.array(self.category_unique[col_name]).reshape(-1, 1))

        encode_df = pd.DataFrame(one_to_hot.transform(df[[col_name]]).toarray())
        encode_df.columns = encode_df.get_feature_names()

        df = df.join(encode_df)
        df = self.delete_features(df, [col_name, ])
        return df

    def hash_encode(self, df: pd.DataFrame, col_name: str):
        hashEncoder = FeatureHasher(n_features=6, input_type='string')
        hashEncoder.fit(self.category_unique[col_name])

        encode_df = pd.DataFrame(hashEncoder.transform(df[col_name]).toarray())
        encode_df.columns = [col_name + str(i) for i in range(hashEncoder.n_features)]
        df = df.join(encode_df)
        df = self.delete_features(df, [col_name, ])
        return df

    def get_df_of_categories(self, df: pd.DataFrame) -> list:
        """
        this function just write for our dataset
        :param df:
        :return:
        """
        choices = list()

        remind = pd.DataFrame()
        close_years = list(df['Close Date Year'].unique())
        create_years = list(df['Created Date Year'].unique())
        stages = list(df['Stage'].unique())
        agents = list(df['SalesAgentEmailID'].unique())
        products = list(df['Product'].unique())

        for product in products:
            pro_df = df[df['Product'] == product]
            for stage in stages:
                pro_st_df = pro_df[pro_df['Stage'] == stage]
                for create_year in create_years:
                    p_st_c_df = pro_st_df[pro_st_df['Created Date Year'] == create_year]
                    for close_year in close_years:
                        p_s_c2_df = p_st_c_df[p_st_c_df['Close Date Year'] == close_year]
                        for agent in agents:
                            a_p_s_c2_df = p_s_c2_df[p_s_c2_df['SalesAgentEmailID'] == agent]
                            d = pd.DataFrame()
                            d = pd.concat([d, a_p_s_c2_df])
                            if len(d) > 1:
                                choices.append(d)
                            elif len(d) == 1:
                                remind = pd.concat([d, remind])
        choices.append(remind)
        return choices

    def split_base_categories(self, df, choices, target_col: str, index_col: str):
        val_x_data = pd.DataFrame()
        val_y_data = list()
        train_x_data = pd.DataFrame()
        train_y_data = list()
        for choice in choices:
            x = choice.drop(target_col, axis=1)
            x = x.drop(index_col, axis=1)
            y = choice[target_col]
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=0.8,
                                                                random_state=42)
            train_x_data = train_x_data.append(x_train)
            train_y_data += (y_train.tolist())
            val_x_data = val_x_data.append(x_test)
            val_y_data += y_test.tolist()

        train_x_data.reset_index(inplace=True)
        val_x_data.reset_index(inplace=True)

        return train_x_data, val_x_data, train_y_data, val_y_data

    def preprocess_train_val_data(self, df):
        # this function just write for our dataset
        df = self.add_between_two_date(df, 'Created Date', 'Close Date')
        df = self.convert_date_to_year(df, 'Created Date')
        df = self.convert_date_to_year(df, 'Close Date')
        df = self.delete_features(df, ['Customer', 'SalesAgentEmailID', 'ContactEmailID'])
        df = self.fill_nan_by_mean_group(df, ['Product', 'Stage'], 'Close_Value', 'Index')
        df = self.replace_category_data(get_stage, 'Stage', ['Close_Value', 'Product', 'Stage'], df)
        self.make_categories_data(df, ['Product', 'Agent'])
        self.make_categories_data(df, ['Stage', ], False)
        df = self.label_encode(df, 'Stage')
        choices = self.get_df_of_categories(df)
        train_x_data, val_x_data, train_y_data, val_y_data = self.split_base_categories(df, choices, 'Stage', 'Index')
        train_x_data = self.one_to_hot_encode(train_x_data, 'Product')
        val_x_data = self.one_to_hot_encode(val_x_data, 'Product')
        train_x_data = self.hash_encode(train_x_data, 'Agent')
        val_x_data = self.hash_encode(val_x_data, 'Agent')
        self.save_categories()
        return train_x_data, val_x_data, train_y_data, val_y_data

    def preprocess_test_data(self, df):
        self.load_categories()
        if 'Stage' in df.columns:
            df = self.delete_features(df, ["Stage"])
        df = self.change_unexpected_values(df, "Product")
        df = self.change_unexpected_values(df, "Agent")
        df = self.add_between_two_date(df, 'Created Date', 'Close Date')
        df = self.convert_date_to_year(df, 'Created Date')
        df = self.convert_date_to_year(df, 'Close Date')
        df = self.delete_features(df, ['Customer', 'SalesAgentEmailID', 'ContactEmailID'])
        df = self.one_to_hot_encode(df, 'Product')
        df = self.one_to_hot_encode(df, 'Agent')
        return df
