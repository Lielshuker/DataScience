import pandas as pd
import random
import numpy as np


def loadData(path):
    data_csv = pd.read_csv(path)
    data = data_csv.copy()
    row_count, column_count = data_csv.shape
    table_cols_name = data_csv.columns
    data_csv.head()
    return data_csv, data, row_count, column_count, table_cols_name


def remove_values_from_dataset(data_csv, row_count, table_cols_name):
    dataset_len = len(data_csv)
    five_percent = int((dataset_len * 5) / 100)
    for j in range(len(data_csv.columns)):
        for i in range(five_percent):
            row = random.randint(0, row_count - 1)
            data_csv.loc[row, table_cols_name[j]] = np.nan
    data_csv.head()
    return data_csv

