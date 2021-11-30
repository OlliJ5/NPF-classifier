import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('npf_train.csv')
print('test')

class2 = np.array([1]*df.shape[0])
class2[df['class4']=='nonevent'] = 0

df['class2'] = class2
df = df.drop(['id', 'date', 'class4'], axis=1)

#print(df.head())

y = df['class2']
#print(y)
X = df.drop(['class2'], axis=1)
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print(X_train)
# print(X_test)

# print(y_train)
# print(y_test)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
