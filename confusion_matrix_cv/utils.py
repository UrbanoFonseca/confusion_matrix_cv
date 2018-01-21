#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:33:52 2018

@author: ml
"""

from sklearn.metrics import confusion_matrix
import pandas as pd
from confusion_matrix_cv.confusion_matrix_cv import ConfusionMatrixCV


def output_confusions_results(dict_of_estimators, X, Y):
	# Outputs a DataFrame from where the results of the Confusion Matrix metrics
	# are stored.
	confusion_matrix_cv_metrics = ['sensitivity', 'specificity', 'precision',
									'fprate', 'F-Score', 'Likelihoods', 'Youden', 'DP']
	results = pd.DataFrame(columns=confusion_matrix_cv_metrics)

	for name, estimator in dict_of_estimators.items():
		cnf = ConfusionMatrixCV()
		predicted_y_bin = estimator.predict(X)
		cnf.convert_cm(pd.DataFrame(confusion_matrix(Y, predicted_y_bin)))
		cnf.calculate_metrics()
		metrics = [cnf.sensitivity, cnf.specificity, cnf.precision, cnf.fprate,
		cnf.F, cnf.likelihoods, cnf.youden, cnf.DP]
		results.loc[name] = metrics

	return results
