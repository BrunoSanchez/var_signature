#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ml_experiment.py
#
#  Copyright 2018 Bruno S <bruno@oac.unc.edu.ar>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


def experiment(clf, x, y, nfolds=10, printing=False, multiclass=False):
    skf = StratifiedKFold(n_splits=nfolds)
    probabilities = None # np.array([])
    predictions = np.array([])
    y_testing = np.array([])

    for train, test in skf.split(x, y):

        x_train = x[train]
        y_train = y[train]
        clf.fit(x_train, y_train)

        x_test = x[test]
        y_test = y[test]
        pr = clf.predict(x_test)
        probs = clf.predict_proba(x_test)  #[:, 0]

        probabilities = (
            probs if probabilities is None else
            np.vstack([probabilities, probs]))
        predictions = np.hstack([predictions, pr])
        y_testing = np.hstack([y_testing, y_test])

    results = {}
    results['y_test'] = y_testing
    results['model'] = clf
    results['probabilities'] = probabilities
    results['predictions'] = predictions
    results['confusion_matrix'] = metrics.confusion_matrix(y_testing, predictions)
        
    if printing:
        print metrics.classification_report(y_testing, predictions)
    if not multiclass:
        fpr, tpr, thresholds = metrics.roc_curve(y_testing, 1.-probabilities[:, 0])
        prec_rec_curve = metrics.precision_recall_curve(y_testing, 1.- probabilities[:, 0])
        roc_auc = metrics.auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
        results['thresh'] = thresholds
        results['roc_auc'] = roc_auc
        results['prec_rec_curve'] = prec_rec_curve

    return results