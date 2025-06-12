from rfgap import RFGAP

# TODO: Determine the common set of inputs for each method

# Include X, y, X_test, random_state, ...?
def nonconformity_rfgap(X, y, X_test = None, random_state = None, k = 5, **kwargs):
    rf = RFGAP(y = y, prox_method = 'rfgap', matrix_type = 'dense', 
               random_state = random_state, oob_score = True, **kwargs)

    rf.fit(X)

    rf.get_nonconformity(k = k, X_test = X_test)


    # WHAT TO ACTUALLY RETURN?

def nonconformity_original():
    pass

def nonconformity_oob():
    pass



# FROM OLD REPO:

        # rf.get_diff_proba()
        # rf.get_nonconformity()
        # rf.get_nonconformity_original()
        # rf.get_nonconformity_rfgap()
        # rf.get_trust_scores()

        # arc = {
        #     'conformity-accuarcy-drop': rf.conformity_accuarcy_drop,
        #     'conformity-drop': rf.conformity_n_drop,
        #     'conformity-quantiles': rf.conformity_quantiles,
        #     'conformity-rfgap-accuarcy-drop': rf.conformity_rfgap_accuarcy_drop,
        #     'conformity-rfgap-drop': rf.conformity_rfgap_n_drop,
        #     'conformity-rfgap-quantiles': rf.conformity_quantiles_rfgap,
        #     'diff-proba-accuarcy-drop': rf.diff_proba_accuarcy_drop,
        #     'diff-proba-drop': rf.diff_proba_n_drop,
        #     'diff-proba-quantiles': rf.diff_proba_quantiles,
        #     'original-conformity-accuarcy-drop': rf.conformity_original_accuarcy_drop,
        #     'original-conformity-drop': rf.conformity_original_n_drop,
        #     'original-conformity-quantiles': rf.conformity_quantiles_original,
        #     'trust-accuarcy-drop': rf.trust_accuarcy_drop,
        #     'trust-drop': rf.trust_n_drop,
        #     'trust-quantiles': rf.trust_quantiles
        # }