import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import pandas as pd


def imputing_with_mean(data_csv, table_cols_name):
    data_with_mean_values = data_csv.copy()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    for i in table_cols_name:
        if isinstance(data_with_mean_values[i][0], str):
            continue
        data_with_mean_values[i] = imputer.fit_transform(data_with_mean_values[i].values.reshape(-1, 1))
    data_with_mean_values.head()
    return data_with_mean_values


def imputing_with_median(data_csv, table_cols_name):
    data_with_median_values = data_csv.copy()
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    for i in table_cols_name:
        if isinstance(data_with_median_values[i][0], str):
            continue
        data_with_median_values[i] = imputer.fit_transform(data_with_median_values[i].values.reshape(-1, 1))
    data_with_median_values.head()
    return data_with_median_values


def imputing_with_most_frequent(data_csv, table_cols_name):
    data_with_most_frequent_value = data_csv.copy()
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for i in table_cols_name:
        if isinstance(data_with_most_frequent_value[i][0], str):
            continue
        data_with_most_frequent_value[i] = imputer.fit_transform(data_with_most_frequent_value[i].values.reshape(-1, 1))
    data_with_most_frequent_value.head()
    return data_with_most_frequent_value


def imputing_with_zero(data_csv, table_cols_name):
    data_with_zero_value = data_csv.copy()
    imputer = SimpleImputer(missing_values=np.nan, fill_value=0, strategy='constant')
    for i in table_cols_name:
        if isinstance(data_with_zero_value[i][0], str):
            continue
        data_with_zero_value[i] = imputer.fit_transform(data_with_zero_value[i].values.reshape(-1, 1))
    data_with_zero_value.head()
    return data_with_zero_value


def imputing_with_knn(data_csv, table_cols_name):
    data_with_numric_column = data_csv.copy()
    le = preprocessing.LabelEncoder()
    catgory_column_list = data_with_numric_column.select_dtypes(include=['object']).columns.tolist()
    for column in catgory_column_list:
        data_with_numric_column[column] = le.fit_transform(data_with_numric_column[column])
    data_with_knn_value = data_with_numric_column
    impute_knn = KNNImputer(n_neighbors=2)
    data_with_knn_value = pd.DataFrame(impute_knn.fit_transform(data_with_knn_value),
                                       columns=data_with_knn_value.columns)
    data_with_knn_value.head(20)
    return data_with_knn_value, data_with_numric_column


def imputing_with_covariance(data_with_numric_column):
    data_with_corr_value = data_with_numric_column
    impute_it = IterativeImputer()
    data_with_corr_value = pd.DataFrame(impute_it.fit_transform(data_with_corr_value),
                                        columns=data_with_corr_value.columns)
    data_with_corr_value.head()
    return data_with_corr_value