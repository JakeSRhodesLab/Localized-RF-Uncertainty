# File to run all prediction intervals for regression methods

# TODO: Currently a direct copy from the run_classification.py file.
# Update for regression module

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))



# TODO: Direct copy from RF-Imputation paper; rewrite for use in uncertainty quantification.
from methods import regression_methods
from datasets import load_regression_sets
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import pickle

import traceback

from rfgap import (
    save_to_pkl,
    load_from_pkl,
    is_in_interval,
    get_coverage,
    get_width_stats
)


# Load datasets
regression_datasets = load_regression_sets()
data_idx = regression_datasets.keys()

# Random seeds and missing percentages
np.random.seed(42)
random_states = np.random.randint(1, 1000000, 10)
prox_methods = ['rfgap', 'oob', 'original']


# Output directory
output_dir = Path("results/regression")
output_dir.mkdir(parents=True, exist_ok=True)


# Main processing function
# TODO: Update input parameters
def process_dataset(idx, random_state, **kwargs):
    name = regression_datasets[idx]['name']
    print(name)

    # TODO: Review read in and prepare data
    X = regression_datasets[idx]['X']
    X = pd.get_dummies(X)
    y = pd.Series(pd.Categorical(regression_datasets[idx]['y']).codes, name='class')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    n = regression_datasets[idx]['n_samples']
    d = regression_datasets[idx]['n_features']

    # TODO: Dynamically determine the methods to use based on the dataset
    ks = [1, 5, 10, 20, 50, 100, 200, 500]
    levels = [0.7, 0.8, 0.9, 0.95, 0.99]


    for prox_method in prox_methods:
        for k in ks:
            for level in levels:
                print('  ', prox_method, k)

                
                qualitative_dir = output_dir / 'qualitative'
                quantitative_dir = output_dir / 'quantitative'
                qualitative_dir.mkdir(parents=True, exist_ok=True)
                quantitative_dir.mkdir(parents=True, exist_ok=True)

                qual_fname = qualitative_dir / f"{name}_{prox_method}_k{k}_rs{random_state}.pkl"
                quant_fname = quantitative_dir / f"{name}_{prox_method}_k{k}_rs{random_state}.pkl"

                # TODO: Get separate file names for each method


                # Check if both files already exist
                if qual_fname.exists() and quant_fname.exists():
                    print(f"    Skipping {prox_method}, k={k} (cached)")
                    continue

                try:



                    # TODO: Update the below for the regression methods
                    # rf, plot_results, base_results = get_classification_results(
                    #     X_train, y_train, X_test=X_test, y_test=y_test,
                    #     prox_method=prox_method, random_state=random_state, k=k
                    # )

                    rf_method = regression_methods['rf']
                    qrf_method = regression_methods['qrf']
                    zrf_method = regression_methods['zrf']

                    rf, rf_plot_results, rf_base_results = rf_method(
                        X_train, y_train, X_test=X_test, y_test=y_test,
                        prox_method=prox_method, random_state=random_state, k=k,
                        level=level, **kwargs
                    )

                    qrf, qrf_plot_results, qrf_base_results = qrf_method(
                        X_train, y_train, X_test=X_test, y_test=y_test,
                        prox_method=prox_method, random_state=random_state, k=k,
                        level=level, **kwargs
                    )

                    zrf, zrf_plot_results, zrf_base_results = zrf_method(
                        X_train, y_train, X_test=X_test, y_test=y_test,
                        prox_method=prox_method, random_state=random_state, k=k,
                        level=level, **kwargs
                    )

                    plot_results['name'] = name
                    plot_results['n_features'] = d
                    plot_results['n_samples'] = n

                    base_results['name'] = name
                    base_results['n_features'] = d
                    base_results['n_samples'] = n


                    save_to_pkl(plot_results, qual_fname)
                    save_to_pkl(base_results, quant_fname)

                except Exception as e:
                    error_message = f"Error processing {name} with method {prox_method} and k={k}, rs={random_state}:\n"
                    with open("run_classification_errors.txt", "a") as error_file:
                        error_file.write(error_message)
                        traceback.print_exc(file=error_file)
                    print(error_message)

# Parallel execution
Parallel(n_jobs=-2)(
    delayed(process_dataset)(idx, random_state)
    for idx in data_idx
    for random_state in random_states
)
