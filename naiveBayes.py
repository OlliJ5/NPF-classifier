import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

df = pd.read_csv('npf_train.csv')

#Wrangle data
class2 = np.array([1]*df.shape[0])
class2[df['class4']=='nonevent'] = 0
df['class2'] = class2
df = df.drop(['id', 'date', 'class4'], axis=1)


#Set X and y
y = df['class2']
X = df.drop(['class2'], axis=1)


#Create model and predict
gnb = GaussianNB()
#scores = cross_val_score(gnb, X, y, cv=3)

#performance measures, accuracy estimations, stats etc.
#print("Scores using cross validation", scores)

#Classification accuracy and standard deviation
#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#Confusion matrix
kf = KFold(n_splits=3)

for train_index, test_index in kf.split(X):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X.loc[train_index], X.loc[test_index]
     y_train, y_test = y[train_index], y[test_index]

     gnb.fit(X_train, y_train)
     y_pred = gnb.predict(X_test)
     print(confusion_matrix(y_test, y_pred))

     print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))