"""
For making the script run:
- set the 8_JeDiS folder as the current directory
- run: python ./Question3/run_3_JeDiS.py

NOTE: the data is supposed to be in JeDiS_HW1/data/*.csv
"""
# Import libraries
import sys
import os

# Set directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from funcs.funcs import *
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import time
from sklearn.metrics import confusion_matrix

# Load data set
df = pd.read_csv('data/Letters_Q_O.csv')
columns = list(df.columns[:-1])

# Standardize features 
df[columns] = preprocessing.StandardScaler().fit_transform(df[columns])

# Get train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1939671)

# Define hyperparameters and kernel
C = 5
gamma = 0.15
kernel = 'rbf'
q = 2

# Process dataframes to have them ready to apply the SVM
X, y = process_df(train_df, ['Q', 'O'])
X_test, y_test = process_df(test_df, ['Q', 'O'])

start = time.time()

# Fit SVM with the MVP method
svm = SVMDecomposition(X, y, C=C, gamma=gamma, kernel=kernel)
svm.fit(working_set_size=2, max_iters=5000, stop_thr=1e-3, tol=1e-3)

stop = time.time()

# Calculate metrics
train_acc = svm.eval(X, y)
test_acc = svm.eval(X_test, y_test)
time = round((stop-start), 2)
num_it = svm.i

# Calculate initial and final values for the objective funtion
alpha_init = np.zeros(len(svm.alpha))
e = np.ones((X.shape[0], 1))

# Calculate Q matrix to use when evaluating the objective function
K = rbf(X, X, 0.1) 
Q = np.outer(svm.y, svm.y) * K

# Evaluate the objective function
init_obj = 0.5 * alpha_init.T @ Q @ alpha_init - e.T @ alpha_init
fin_obj = 0.5 * svm.alpha.T @ Q @ svm.alpha - e.T @ svm.alpha

print('C: ', C)
print('gamma: ', gamma)
print('kernel: ', kernel)
print('Classification Rate on Training Set:', round(train_acc, 3)*100, '%')
print('Classification Rate on Test Set:', round(test_acc, 3)*100, '%')
matrix = confusion_matrix(y_test, svm.pred(X_test)).ravel()
print('Confusion Matrix: \n', matrix.reshape((2, 2)))
print('Computational Time:', time, ' s')
print('Number of optimizations:', num_it)
print('m(α) - M(α):', svm.diff_ma_Ma)
