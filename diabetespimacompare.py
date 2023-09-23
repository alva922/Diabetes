# Import key libraries
import scikitplot as skplt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
# Load input data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y 
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=7)
# train model
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# evaluate performance
yhat = model.predict(X_test)
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % score)
Y_test_probs = model.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, Y_test_probs,
                       title="ROC Curve", figsize=(12,6));
skplt.metrics.plot_precision_recall_curve(y_test, Y_test_probs,
                       title="Precision-Recall Curve", figsize=(12,6));

lr_probas = model.fit(X_train, y_train).predict_proba(X_test)
probas_list = [lr_probas]
clf_names = ['XGB']

skplt.metrics.plot_calibration_curve(y_test,
                                     probas_list,
                                     clf_names, n_bins=15,
                                     figsize=(12,6)
                                     );
Y_probas = model.predict_proba(X_test)

skplt.metrics.plot_ks_statistic(y_test, Y_probas, figsize=(10,6));

skplt.metrics.plot_cumulative_gain(y_test, Y_probas, figsize=(10,6));

skplt.metrics.plot_lift_curve(y_test, Y_probas, figsize=(10,6));

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, y_pred)

from sklearn.metrics import hamming_loss
hamming_loss(y_test, y_pred)

from sklearn import metrics
metrics.fbeta_score(y_test, y_pred, average='macro', beta=0.5)

metrics.precision_score(y_test, y_pred, average='macro')

metrics.recall_score(y_test, y_pred, average='micro')

metrics.f1_score(y_test, y_pred, average='weighted')

from sklearn.metrics import jaccard_score
jaccard_score(y_test, y_pred)

from sklearn.metrics import log_loss
log_loss(y_test, y_pred)

# https://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_pred)

from sklearn.metrics import zero_one_loss
zero_one_loss(y_test, y_pred)

from sklearn.metrics import brier_score_loss
brier_score_loss(y_test, y_pred)


