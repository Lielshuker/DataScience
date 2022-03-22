import run as run

if __name__ == '__main__':
    # running the code on one dataset - housing and return the best imputation
    # for experiments there is Experiments script
    best = run.find_best_imputation_method()[0]
    print(best)




