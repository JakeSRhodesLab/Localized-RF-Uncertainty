import openml
import os
import pandas as pd

reg_idx = openml.study.get_suite(353).data
reg_idx = openml.datasets.list_datasets(data_id = reg_idx, output_format = 'dataframe')
reg_idx = reg_idx[reg_idx['NumberOfInstancesWithMissingValues'] == 0]
# reg_idx = reg_idx[reg_idx['NumberOfInstances'] < 5000]
# reg_idx = reg_idx[reg_idx['NumberOfFeatures'] < 100]
reg_idx.sort_values(by = 'NumberOfInstances', ascending = True, inplace = True)
reg_idx = reg_idx.did.values.tolist()


def load_regression_sets():
    """Load regression datasets from OpenML.

    Returns:
        list: A list of tuples containing the dataset ID and the dataset name.
    """
    datasets = {}
    for i, did in enumerate(reg_idx):
        try:
            dataset = openml.datasets.get_dataset(did, download_data = True, download_qualities = False, download_features_meta_data = False)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            y = y.astype(float)
            X = pd.get_dummies(X, drop_first=False)
            bool_cols = X.select_dtypes(bool).columns
            X[bool_cols] = X[bool_cols].astype(int)
            

            X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.dtype == 'float64' else x)
            datasets[i] = {
                'did': did,
                'name': dataset.name,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'y': y,
                'X': X
            }
        except Exception as e:
            print(f"Error loading dataset {did}: {e}")
    return datasets