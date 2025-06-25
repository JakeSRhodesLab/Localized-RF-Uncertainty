from rfgap import RFGAP

# TODO: Verify scaling of RF-GAP and how it affects AUC etc.

# All of these results come from the same rf model.
def get_classification_results(X, y, X_test=None, y_test=None, prox_method = 'rfgap', 
                  random_state=None, k=5, **kwargs):

    rf = RFGAP(y=y, random_state=random_state, oob_score=True, **kwargs)
    rf.fit(X, y)
    
    rf.get_diff_proba(x_test=X_test, y_test=y_test) # Not dependent on prox_method
    rf.get_nonconformity(k=k, x_test=X_test, y_test=y_test)
    rf.get_instance_classification_expectation(x_test = X_test, y_test = y_test)
    rf.get_tree_conformity(X, y, x_test = X_test, y_test = y_test)

    # Has all results in case needed for plots
    plot_results = {
        'prox_method': prox_method,
        'conformity_k': k,
        'random_state': random_state,
        'oob_score_': rf.oob_score_,
        'diff_proba_auc': rf.diff_proba_auc,
        'diff_proba_auc_test': rf.diff_proba_auc_test,
        'conformity_auc': rf.conformity_auc,
        'conformity_auc_test': rf.conformity_auc_test,
        'diff_proba_accuarcy_drop': rf.diff_proba_accuarcy_drop,
        'diff_proba_n_drop': rf.diff_proba_n_drop,
        'diff_proba_accuracy_drop_test': rf.diff_proba_accuarcy_drop_test,
        'diff_proba_n_drop_test': rf.diff_proba_n_drop_test,
        'conformity_accuracy_drop': rf.conformity_accuracy_drop,
        'conformity_n_drop': rf.conformity_n_drop,
        'conformity_accuracy_drop_test': rf.conformity_accuracy_drop_test,
        'conformity_n_drop_test': rf.conformity_n_drop_test,
        'ice_auc': rf.ice_auc,
        'ice_auc_test': rf.ice_auc_test,
        'ice_accuracy_drop': rf.ice_accuracy_drop,
        'ice_n_drop': rf.ice_n_drop,
        'ice_accuracy_drop_test': rf.ice_accuracy_drop_test,
        'ice_n_drop_test': rf.ice_n_drop_test,
        'tree_conformity': rf.tree_conformity,
        'tree_conformity_test': rf.tree_conformity_test,
        'tree_conformity_accuracy_drop': rf.tree_conformity_accuracy_drop,
        'tree_conformity_accuracy_drop_test': rf.tree_conformity_accuracy_drop_test,
        'tree_conformity_n_drop': rf.tree_conformity_n_drop,
        'tree_conformity_n_drop_test': rf.tree_conformity_n_drop_test,
        'tree_conformity_auc': rf.tree_conformity_auc,
        'tree_conformity_auc_test': rf.tree_conformity_auc_test
    }
    
    # Just results for tables
    base_results = {
        'prox_method': prox_method,
        'conformity_k': k,
        'random_state': random_state,
        'oob_score_': rf.oob_score_,
        'diff_proba_auc': rf.diff_proba_auc,
        'diff_proba_auc_test': rf.diff_proba_auc_test,
        'conformity_auc': rf.conformity_auc,
        'conformity_auc_test': rf.conformity_auc_test,
        'ice_auc': rf.ice_auc,
        'ice_auc_test': rf.ice_auc_test,
        'tree_conformity_auc': rf.tree_conformity_auc,
        'tree_conformity_auc_test': rf.tree_conformity_auc_test
    }
    
    # Maybe different return values?
    return rf, plot_results, base_results