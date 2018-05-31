#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ml_test_no1.py
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
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn import metrics

import ml_experiment

dbf_c = np.loadtxt('atlas_db/features_c_DBF.txt.gz')
dbf_o = np.loadtxt('atlas_db/features_o_DBF.txt.gz')

mira_c = np.loadtxt('atlas_db/features_c_MIRA.txt.gz')
mira_o = np.loadtxt('atlas_db/features_o_MIRA.txt.gz')


X = np.vstack([mira_o, dbf_o])
Y = np.vstack([np.ones(mira_o.shape[0]), np.zeros(dbf_o.shape[0])])

X_dev, X_final_test, y_dev, y_final_test = \
    train_test_split(X, Y, test_size=0.25)

imp1 = Imputer(missing_values=np.nan, strategy="mean", axis=0)
imp1.fit(X_dev)
x = imp1.transform(X_dev)

clf = RandomForestClassifier(n_estimators=1000, max_features=10, n_jobs=4)
results = ml_experiment.experiment(clf, x, y_dev)

clf.fit(x, y_dev)

pr = clf.predict(imp1.transform(X_final_test))
probs = clf.predict_proba(imp1.transform(X_final_test))
print metrics.classification_report(y_final_test, pr)

print metrics.confusion_matrix(y_final_test, pr)

# Results:
             #~ precision    recall  f1-score   support

        #~ 0.0       1.00      1.00      1.00      2817
        #~ 1.0       0.99      1.00      1.00      1884

#~ avg / total       1.00      1.00      1.00      4701

#~ [[2807   10]
 #~ [   6 1878]]
