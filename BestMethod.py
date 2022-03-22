from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import ks_2samp


# using fitter library to find the datasets distribution, using sum square error to the calculating
def best_fit_dest(column_name, datasets, datasets_names):
    data_best_dest = ''
    same_dist_summary = []
    for i, dataset in enumerate(datasets):
        if datasets_names[i] == 'data_csv':
            continue
        try:
            f = Fitter(dataset[column_name].values, distributions=get_common_distributions())
            f.fit()
            d = f.summary()
            best_dist = f.get_best(method='sumsquare_error')
            if datasets_names[i] == 'data':
                data_best_dest = {'best': best_dist, 'd': d}
                continue

            if next(iter(best_dist)) == next(iter(data_best_dest['best'])):
                same_dist_summary.append([datasets_names[i], d])
        except:
            pass
    return same_dist_summary, data_best_dest


# calculate which distribution is the most similar to the original
def mme(same_dist_summary, data_best_dest):
    col = next(iter(data_best_dest['best']))
    best_mme = data_best_dest['d'].loc[col, 'sumsquare_error']
    first = 0
    best = ''
    for i, dist in enumerate(same_dist_summary):
        x = dist[1].loc[col, 'sumsquare_error']
        if i == 0:
            first = abs(x - best_mme)
            best = dist

        closet = abs(x - best_mme)

        if closet < first:
            first = closet
            best = dist
    return best


# if the fitter return few distribution we calculate the similar distribution
def chose_best_data(col, datasets, datasets_names):
    same_dist_summary, data_best_dest = best_fit_dest(col, datasets, datasets_names)
    if len(same_dist_summary) == 0:
        return ''
    elif len(same_dist_summary) == 1:
        return same_dist_summary[0][0]
    else:
        return mme(same_dist_summary, data_best_dest)[0]


# using ks test to check the new data that its distribution is the most close to the original distribution
def ks_test(col, data, datasets, datasets_names):
    max_pvalue = 0
    best_dataset = ''
    for i, dataset in enumerate(datasets):
        if datasets_names[i] == 'data_csv' or datasets_names[i] == 'data':
            continue
        try:
            res, pval = ks_2samp(data[col], dataset[col])
            if pval > max_pvalue:
                max_pvalue = pval
                best_dataset = datasets_names[i]
        except:
            pass
    return best_dataset


# calculating the skewness of the new data and compering to the original to find the best method
def skewness(col, data, datasets, datasets_names):
    first = 0
    best_data = ''
    try:
        data_skewness = data[col].skew()
    except:
        return ''
    for i, dataset in enumerate(datasets):
        if datasets_names[i] == 'data_csv' or datasets_names[i] == 'data':
            continue
        x = dataset[col].skew()
        if i == 2:
            first = abs(x - data_skewness)
            best_data = datasets_names[i]
        closet = abs(x - data_skewness)
        if closet < first:
            first = closet
            best_data = datasets_names[i]

    return best_data


# calculating the kurtosis of the new data and compering to the original to find the best method
def kurtosis(col, data, datasets, datasets_names):
    first = 0
    best_data = ''
    try:
        data_kurt = data[col].kurt()
    except:
        return ''
    for i, dataset in enumerate(datasets):
        if datasets_names[i] == 'data_csv' or datasets_names[i] == 'data':
            continue
        x = dataset[col].kurt()
        if i == 2:
            first = abs(x - data_kurt)
            best_data = datasets_names[i]
        closet = abs(x - data_kurt)
        if closet < first:
            first = closet
            best_data = datasets_names[i]

    return best_data


# running all those methods, inserting to list and find the most frequent one.
def find_best_method(table_cols_name, data, datasets, datasets_names):
    best_by_ks_test = []
    best_by_dist = []
    best_by_skewness = []
    best_by_kurtosis = []

    for col in table_cols_name:
        best_by_dist.append(chose_best_data(col, datasets, datasets_names))
        best_by_ks_test.append(ks_test(col, data, datasets, datasets_names))
        best_by_skewness.append(skewness(col, data, datasets, datasets_names))
        best_by_kurtosis.append(kurtosis(col, data, datasets, datasets_names))

    best = []
    best.append(max(set(best_by_skewness), key=best_by_skewness.count))
    best.append(max(set(best_by_kurtosis), key=best_by_kurtosis.count))
    best.append(max(set(best_by_dist), key=best_by_dist.count))
    best.append(max(set(best_by_ks_test), key=best_by_ks_test.count))

    return max(set(best), key=best.count)





