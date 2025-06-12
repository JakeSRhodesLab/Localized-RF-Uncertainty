from rfgap import RFGAP


# Again, what to return? What is needed for the plots/tables/scores?
def probadiff():
    pass



# def get_diff_proba(self, x_test = None):
#     """
#     Calculates the differences between the two highest predict_proba values for all out-of-bag (OOB) and test points.
#     Returns:
#     - diff_proba_oob: Array of differences between the two highest predict_proba values for OOB points.
#     - diff_proba_test: Array of differences between the two highest predict_proba values for test points.
#     """

#     if not self.oob_score:
#         raise ValueError('Model must be fit with oob_score = True to calculate differences in predict_proba values')
    

#     self.oob_proba = self.oob_decision_function_

#     # Calculate differences for OOB points
#     self.diff_proba_oob = np.abs(np.diff(np.sort(self.oob_proba, axis=1)[:, -2:], axis = 1)).squeeze()
#     self.diff_proba_quantiles = np.quantile(self.diff_proba_oob, np.linspace(0, 0.99, 100))

#     # Calculate differences for test points if available
#     if x_test is not None:
#         test_proba = self.predict_proba(x_test)
#         self.diff_proba_test = np.abs(np.diff(np.sort(test_proba, axis=1)[:, -2:], axis=1)).squeeze()


#     self.diff_proba_auc, self.diff_proba_accuarcy_drop, self.diff_proba_n_drop = self.accuracy_rejection_curve_area(self.diff_proba_quantiles, self.diff_proba_oob)