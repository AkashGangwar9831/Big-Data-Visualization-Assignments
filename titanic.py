# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:02:33 2018

@author: Akash
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import tree


from sklearn.metrics import accuracy_score


df = pd.read_csv('titanic.csv')

df = df[[ 'Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = df.dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.25)

model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

acc=accuracy_score(y_test, y_predict)

print(acc)