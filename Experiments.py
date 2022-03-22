from sklearn import metrics
import run as run
import numpy as np
import matplotlib.pyplot as plt


dataset_path1 = './datasets/housing.csv'
dataset_path2 = './datasets/NFLX.csv'
dataset_path3 = './datasets/fish.csv'
dataset_path4 = './datasets/Pokemon.csv'

def test(path):
    best, datasets, datasets_names = run.find_best_imputation_method(path)
    index = datasets_names.index('data')
    original_dataset = datasets[0]
    for i, col in enumerate(original_dataset):
        if isinstance(original_dataset[col][0], str) or isinstance(original_dataset[col][0], np.bool_):
            continue
        accuracy = []
        for j,dataset in enumerate(datasets):
            if datasets_names[j] == 'data_csv' or datasets_names[j] == 'data':
                continue
            y_true = np.array(np.round(original_dataset[col]))
            y_prod = np.array(np.round(dataset[col]))
            cm = metrics.confusion_matrix(y_true, y_prod)
            accuracy.append(metrics.accuracy_score(y_true, y_prod))
            print(metrics.accuracy_score(y_true, y_prod))
            print(datasets_names[j])
        plt.plot([i.strip('data_with_').strip('_value').strip('_values') for i in datasets_names[2:]], accuracy, label=col)

    plt.title(path.strip('./datasets').strip('.csv'))
    plt.show()

test(dataset_path1)
test(dataset_path2)
test(dataset_path3)
test(dataset_path4)