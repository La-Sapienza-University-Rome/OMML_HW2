"""
For making the script run:
- set the 8_JeDiS folder as the current directory
- run: python ./Question4/run_4_JeDiS.py

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
from sklearn.metrics import confusion_matrix

df_Q_O = pd.read_csv('data/Letters_Q_O.csv')
df_D = pd.read_csv('data/Letter_D.csv')
df = pd.concat([df_D, df_Q_O])

columns = list(df.columns[:-1])

df[columns] = preprocessing.StandardScaler().fit_transform(df[columns])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1939671)

C = 5
gamma = 0.15
kernel = 'rbf'

start = time.time()

multi_svm = MultiSVM(train_df, C, gamma, kernel)
multi_svm.fit()

stop = time.time()

X_train, y_train = split_X_y(train_df)
X_test, y_test = split_X_y(test_df)

train_acc = multi_svm.eval(X_train, y_train)
test_acc = multi_svm.eval(X_test, y_test)
time = round((stop-start), 2)
num_it = multi_svm.iter

print('C: ', C)
print('gamma: ', gamma)
print('kernel: ', kernel)
print('Classification Rate on Training Set:', round(train_acc, 3)*100, '%')
print('Classification Rate on Test Set:', round(test_acc, 3)*100, '%')
#confusion_matrix(y_test=y_test, y_fit=svm.pred(X_test))
matrix = confusion_matrix(y_test, multi_svm.pred(X_test)).ravel()
print('Confusion Matrix: \n', matrix.reshape((3, 3)))
print('Computational Time:', time, ' s')
print('Number of optimizations:', num_it)
print('Difference between m and M:', multi_svm.m_M())