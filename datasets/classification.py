import openml
import os
import pandas as pd

class_idx = openml.study.get_suite('OpenML-CC18').data
class_idx = openml.datasets.list_datasets(data_id = class_idx, output_format = 'dataframe')
class_idx = class_idx[class_idx['NumberOfInstancesWithMissingValues'] == 0]
# class_idx = class_idx[class_idx['NumberOfInstances'] < 5000]
# class_idx = class_idx[class_idx['NumberOfFeatures'] < 100]
class_idx.sort_values(by = 'NumberOfInstances', ascending = True, inplace = True)
class_idx = class_idx.did.values.tolist()



def load_classification_sets():
    """Load classification datasets from OpenML.

    Returns:
        list: A list of tuples containing the dataset ID and the dataset name.
    """
    datasets = {}
    for i, did in enumerate(class_idx):
        try:
            dataset = openml.datasets.get_dataset(did, download_data = True, download_qualities = False, download_features_meta_data = False)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
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

