"""
For making the script run:
- set the 8_JeDiS folder as the current directory
- run: python ./Question1/complete_code.py

NOTE: the data is supposed to be in JeDiS_HW1/data/*.csv
"""

import sys
import os

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

def process_df(df):
    X = np.array(df.drop(['letter'], axis=1))
    y_letter = np.array(df[['letter']])

    y = []
    for i in y_letter:
        if i[0] == 'Q':
            y.append(-1)
        else:
            y.append(1)
    y = np.array(y)
    return X, y

df = pd.read_csv('data/Letters_Q_O.csv')
columns = list(df.columns[:-1])

df[columns] = preprocessing.StandardScaler().fit_transform(df[columns])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1939671)

grid_C = [1, 5, 10, 20, 30, 40, 50]
grid_gamma = [1, 2, 3, 4, 5]
grid_kernel = ['polynomial']


# Run hyperparameter tunning for Polynomial Kernel
iterables = [grid_C, grid_gamma, grid_kernel]
min_loss = 1000
k_fold = 5

kf5 = KFold(n_splits=k_fold, shuffle=False)

list_ = [['C', 'Gamma', 'Kernel', 'Validation Loss']]

for t in itertools.product(*iterables):
    
    val_loss = 0
    val_loss_sklearn = 0
    
    C = t[0]
    gamma = t[1]
    kernel = t[2]
    
    print('C: ', C)
    print('gamma: ', gamma)
    print('kernel: ', kernel)
    
    for train_index, test_index in kf5.split(train_df):
        
        X_, y_ = process_df(train_df.iloc[train_index])
        X_val, y_val = process_df(train_df.iloc[test_index])

        svm = SVM(X_, y_, C=C, gamma=gamma, kernel=kernel)
        svm.fit(tol=1e-5, fix_intercept=False)
        
        val_loss += svm.eval(X_val, y_val)
        
        # sklearn implementation
        clf = SVC(kernel = 'poly', degree=gamma, C=C)
        clf.fit(X_, y_)
        val_loss_sklearn += clf.score(X_val, y_val)
        
    val_loss = val_loss/k_fold
    val_loss_sklearn = val_loss_sklearn/k_fold
    print('Accuracy: ', val_loss)
    print('Accuracy Sklearn: ', val_loss_sklearn)
    print('=======================')
    print('')


# Run hyperparameter tunning for RBF Kernel
grid_C = [1, 5, 10, 20, 30, 40, 50]
grid_gamma = [0.01, 0.05, 0.1, 0.13, 0.15, 0.2]
grid_kernel = ['rbf']

iterables = [grid_C, grid_gamma, grid_kernel]
min_acc = 0
k_fold = 5

kf5 = KFold(n_splits=k_fold, shuffle=False)

for t in itertools.product(*iterables):
    
    val_loss = 0
    val_loss_sklearn = 0
    
    C = t[0]
    gamma = t[1]
    kernel = t[2]
    
    print('C: ', C)
    print('gamma: ', gamma)
    print('kernel: ', kernel)
    
    for train_index, test_index in kf5.split(train_df):
        
        X_, y_ = process_df(train_df.iloc[train_index])
        X_val, y_val = process_df(train_df.iloc[test_index])
        
        start = time.time()
        
        svm = SVM(X_, y_, C=C, gamma=gamma, kernel=kernel)
        svm.fit(tol=1e-5, fix_intercept=False)
        
        stop = time.time()
        
        val_loss += svm.eval(X_val, y_val)
        
    val_loss = val_loss/k_fold
    print('Accuracy: ', val_loss)
    print('=======================')
    print('')
    if val_loss > min_acc:
        print('inside')
        X, y = process_df(train_df)
        X_test, y_test = process_df(test_df)
        val_acc = val_loss
        train_acc = svm.eval(X, y)
        test_acc = svm.eval(X_test, y_test)
        best_time = round((stop-start), 2)
        num_it = svm.fit_sol['iterations']
        min_acc = val_loss
        KKT = KKT_violations(svm.alpha, svm.y, svm.X, svm.w, svm.bias, svm.C)
        alpha_init = np.zeros(len(svm.alpha))
        fin_obj = 0.5*np.dot(np.dot(svm.alpha.T, svm.P), 
                             svm.alpha) + np.dot(svm.q.T, svm.alpha)
        init_obj = 0.5*np.dot(np.dot(alpha_init.T, svm.P), 
                              alpha_init) + np.dot(svm.q.T, alpha_init)
