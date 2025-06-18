# File to compute trust scores for classification methods


# Add the parent directory to sys.path to import the methods module

"""
For each classification method, we need to return the actual scores (e.g., trust), the accuracy drops, n_drops,
and the AUC for the rejection curve...

May need to retain accuracy drops and n_drops separately from numerical results for use in plots.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))



# TODO: Direct copy from RF-Imputation paper; rewrite for use in uncertainty quantification.
from methods import get_classification_results
from datasets import load_classification_sets
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
classification_datasets = load_classification_sets()
data_idx = classification_datasets.keys()

# Random seeds and missing percentages
np.random.seed(42)
random_states = np.random.randint(1, 1000000, 10)
prox_methods = ['rfgap', 'oob', 'original']


# Output directory
output_dir = Path("results/classification")
output_dir.mkdir(parents=True, exist_ok=True)


# Main processing function
# TODO: Update input parameters
def process_dataset(idx, random_state):
    name = classification_datasets[idx]['name']
    print(name)

    # TODO: Review read in and prepare data
    X = classification_datasets[idx]['X']
    X = pd.get_dummies(X)
    y = pd.Series(pd.Categorical(classification_datasets[idx]['y']).codes, name='class')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    n = classification_datasets[idx]['n_samples']
    d = classification_datasets[idx]['n_features']

    # TODO: Dynamically determine the methods to use based on the dataset
    ks = [1, 5, 10, 20, 50, 100, 200, 500, 'all', 'auto']


    for prox_method in prox_methods:
        for k in ks:
            print('  ', prox_method, k)

            
            qualitative_dir = output_dir / 'qualitative'
            quantitative_dir = output_dir / 'quantitative'
            qualitative_dir.mkdir(parents=True, exist_ok=True)
            quantitative_dir.mkdir(parents=True, exist_ok=True)

            qual_fname = qualitative_dir / f"{name}_{prox_method}_k{k}_rs{random_state}.pkl"
            quant_fname = quantitative_dir / f"{name}_{prox_method}_k{k}_rs{random_state}.pkl"


            # Check if both files already exist
            if qual_fname.exists() and quant_fname.exists():
                print(f"    Skipping {prox_method}, k={k} (cached)")
                continue

            try:



                # NOT READY, DOUBLE CHECK FUNCTIONS AND INPUTS
                rf, plot_results, base_results = get_classification_results(
                    X_train, y_train, X_test=X_test, y_test=y_test,
                    prox_method=prox_method, random_state=random_state, k=k
                )

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
