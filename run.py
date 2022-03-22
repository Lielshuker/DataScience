import LoadDataset as loadDataset
import Imputation as imputation
import BestMethod as bestMethod
dataset_path1 = './datasets/housing.csv'

# this function loading the dataset, removing 5% of the dataset and try 6 imputation methods
# after this running function that finding the best method.
def find_best_imputation_method(path=dataset_path1):
    # first, load the dataset
    data_csv, data, row_count, column_count, table_cols_name = loadDataset.loadData(path=path)
    # secondly, remove 5% of each feature.
    data_after_remove_value = loadDataset.remove_values_from_dataset(data_csv, row_count, table_cols_name)

    # trying different imputations
    data_with_mean_values = imputation.imputing_with_mean(data_after_remove_value, table_cols_name)
    data_with_median_values = imputation.imputing_with_median(data_after_remove_value, table_cols_name)
    data_with_most_frequent_value = imputation.imputing_with_most_frequent(data_after_remove_value, table_cols_name)
    data_with_zero_value = imputation.imputing_with_zero(data_after_remove_value, table_cols_name)
    data_with_knn_value, data_with_numric_column = imputation.imputing_with_knn(data_after_remove_value,
                                                                                table_cols_name)
    data_with_corr_value = imputation.imputing_with_zero(data_after_remove_value, table_cols_name)

    # create list of those datasets
    datasets = [data, data_after_remove_value, data_with_mean_values, data_with_median_values,
                data_with_most_frequent_value,
                data_with_knn_value, data_with_zero_value, data_with_corr_value]
    datasets_names = ['data', 'data_csv', 'data_with_mean_values', 'data_with_median_values',
                      'data_with_most_frequent_value', 'data_with_knn_value', 'data_with_zero_value',
                      'data_with_corr_value']

    # after the imputation we want to find the best imputation method
    best = bestMethod.find_best_method(table_cols_name, data, datasets, datasets_names)
    return best, datasets, datasets_names
