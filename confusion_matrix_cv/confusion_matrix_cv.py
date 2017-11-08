import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import cross_validate


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]



class ConfusionMatrixCV():
	
	def __init__(self):
		self.tp = None
		self.fn = None
		self.fp = None
		self.tn = None


	def convert_cm(self, confusion_matrix):
		self.tp = confusion_matrix.loc[0,0]
		self.fn = confusion_matrix.loc[0,1]
		self.fp = confusion_matrix.loc[1,0]
		self.tn = confusion_matrix.loc[1,1]
		return tp, fn, fp, tn

	def calculate_metrics(self):
		self.sensitivity =  self.tp / (self.tp + self.fn) 
		self.recall = self.sensitivity
		self.specificity =  self.tn / (self.fp + self.tn)
		self.precision = self.tp / (self.tp + self.fp)
		self.fprate = self.fp / (self.fp + self.tn)
		self.accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
		self.AUC = (self.sensitivity + self.specificity) / 2
		self.F = 2 / ((1/self.precision)+(1/self.recall))

		# ROC
		self.ROC = [self.sensitivity, self.fprate]

		# Youden's index
		self.youden = self.sensitivity - (1-self.specificity)

		# Likelihoods
		roplus = self.sensitivity / (1-self.specificity)
		rominus = (1-self.sensitivity) / self.specificity
		self.likelihoods = roplus, rominus


		# Discriminant Power
		X = self.sensitivity / (1-self.sensitivity)
		Y = self.specificity / (1-self.specificity)
		self.DP = np.sqrt(3) * (np.log(X) + np.log(Y)) / np.pi



	def cross_validate(self, model, X, Y, cv=None, n_jobs=1):
		# Returns test dataframes:
		# TP , FN
		# FP , TN

		confusion_matrix_scorers = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
									'fp': make_scorer(fp), 'fn': make_scorer(fn)}

		confusion_results = cross_validate(model, X, Y, cv=cv,
										   n_jobs=n_jobs,
										   scoring=confusion_matrix_scorers)

		test_keys = [k for k in confusion_results if 'test' in k]
		test_results = [np.mean(confusion_results[i]) for i in test_keys]

		self.test_cm = pd.DataFrame([[test_results[0], test_results[3]],[test_results[2], test_results[1]]])        

		self.convert_cm(self.test_cm)

		self.calculate_metrics()

		return self.test_cm

