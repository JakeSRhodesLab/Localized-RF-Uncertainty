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
    y = pd.Series(regression_datasets[idx]['y'], name='response')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    n = regression_datasets[idx]['n_samples']
    d = regression_datasets[idx]['n_features']

    # TODO: Dynamically determine the methods to use based on the dataset
    ks = [1, 5, 10, 20, 50, 100, 200, 500, 'all', 'auto']
    levels = [0.7, 0.8, 0.9, 0.95, 0.99]


    for prox_method in prox_methods:
        for k in ks:
            for level in levels:
                print('  ', prox_method, k)

                
                qualitative_dir = output_dir / 'qualitative'
                quantitative_dir = output_dir / 'quantitative'
                qualitative_dir.mkdir(parents=True, exist_ok=True)
                quantitative_dir.mkdir(parents=True, exist_ok=True)

                qual_fname = qualitative_dir / f"{name}_{prox_method}_k{k}_level{level}_rs{random_state}.pkl"
                quant_fname = quantitative_dir / f"{name}_{prox_method}_k{k}_level{level}_rs{random_state}.pkl"

                # TODO: Get separate file names for each method


                # Check if both files already exist
                # Currently not checking as different file names are used for each method
                # Check if all result files for each method exist
                qrf_qual = qual_fname.with_name(qual_fname.stem + f'_qrf{qual_fname.suffix}')
                zrf_qual = qual_fname.with_name(qual_fname.stem + f'_zrf{qual_fname.suffix}')
                rf_qual = qual_fname.with_name(qual_fname.stem + f'_rf{qual_fname.suffix}')
                qrf_quant = quant_fname.with_name(quant_fname.stem + f'_qrf{quant_fname.suffix}')
                zrf_quant = quant_fname.with_name(quant_fname.stem + f'_zrf{quant_fname.suffix}')
                rf_quant = quant_fname.with_name(quant_fname.stem + f'_rf{quant_fname.suffix}')

                if (
                    qrf_qual.exists() and zrf_qual.exists() and rf_qual.exists() and
                    qrf_quant.exists() and zrf_quant.exists() and rf_quant.exists()
                ):
                    print(f"    Skipping {prox_method}, k={k} (cached)")
                    continue

                try:


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

                    qrf_plot_results['name'] = name
                    qrf_plot_results['n_features'] = d
                    qrf_plot_results['n_samples'] = n

                    zrf_plot_results['name'] = name
                    zrf_plot_results['n_features'] = d
                    zrf_plot_results['n_samples'] = n

                    rf_plot_results['name'] = name
                    rf_plot_results['n_features'] = d
                    rf_plot_results['n_samples'] = n

                    qrf_base_results['name'] = name
                    qrf_base_results['n_features'] = d
                    qrf_base_results['n_samples'] = n

                    zrf_base_results['name'] = name
                    zrf_base_results['n_features'] = d
                    zrf_base_results['n_samples'] = n

                    rf_base_results['name'] = name
                    rf_base_results['n_features'] = d
                    rf_base_results['n_samples'] = n


                    save_to_pkl(qrf_plot_results, qual_fname.with_name(qual_fname.stem + f'_qrf{qual_fname.suffix}'))
                    save_to_pkl(zrf_plot_results, qual_fname.with_name(qual_fname.stem + f'_zrf{qual_fname.suffix}'))
                    save_to_pkl(rf_plot_results, qual_fname.with_name(qual_fname.stem + f'_rf{qual_fname.suffix}'))
                    save_to_pkl(qrf_base_results, quant_fname.with_name(quant_fname.stem + f'_qrf{quant_fname.suffix}'))
                    save_to_pkl(zrf_base_results, quant_fname.with_name(quant_fname.stem + f'_zrf{quant_fname.suffix}'))
                    save_to_pkl(rf_base_results, quant_fname.with_name(quant_fname.stem + f'_rf{quant_fname.suffix}'))
                    

                except Exception as e:
                    error_message = f"Error processing {name} with method {prox_method} and k={k}, rs={random_state}:\n"
                    with open("run_regression_errors.txt", "a") as error_file:
                        error_file.write(error_message)
                        traceback.print_exc(file=error_file)
                    print(error_message)

# Parallel execution
Parallel(n_jobs=-2)(
    delayed(process_dataset)(idx, random_state)
    for idx in data_idx
    for random_state in random_states
)
