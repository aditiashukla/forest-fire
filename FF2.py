import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
 
import graphviz 

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
 
 
df = pd.read_csv('C:/Users/Anupam Shukla/Desktop/FFdata.csv')
 
X = df.drop('CLASS', axis=1).values
y = df['CLASS'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, stratify = y)
 
 
 
# clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=4)
 
clf = clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
print(clf)
 
print('Training Accuracy = {}'.format(metrics.accuracy_score(y_train, y_train_pred)))
print('Training Confusion = \n{}'.format(metrics.confusion_matrix(y_train, y_train_pred, ['Y','N'])))
 
 
 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['T','H','CO'], 
                                class_names=['Y','N'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("FF-train")
 
 
 
y_test_pred = clf.predict(X_test)
 
print('Testing Accuracy = {}'.format(metrics.accuracy_score(y_test, y_test_pred)))
#print('Testing Confusion = \n{}'.format(metrics.confusion_matrix(y_test, y_test_pred, ['0','1'])))