import graphviz
from sklearn import tree
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import pickle
from sklearn.externals import joblib

balance_data = pd.read_csv('C:/Users/Anupam Shukla/Downloads/dtaset.csv')

ip = ['T','H','CO']
X = balance_data[ip]
Y = balance_data['Decision'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

clf_gini = tree.DecisionTreeClassifier(criterion = "gini", 
random_state = 100,max_depth=3, min_samples_leaf=5) 
clf_gini = clf_gini.fit(X_train,Y_train)
y_pred_gini = clf_gini.predict(X_test) 
print("Confusion Matrix: ", 
	confusion_matrix(Y_test, y_pred_gini)) 
print ("Accuracy : ", 
	accuracy_score(Y_test,y_pred_gini)*100) 
print("Report : ", 
	classification_report(Y_test,y_pred_gini))
c_val_score_current=cross_val_score(DecisionTreeClassifier(),X,Y,cv=5)
print(c_val_score_current)

shizz= "c:/Users/Anupam Shukla/Desktop/theshizz.pkl"
joblib.dump(clf_gini,shizz)


dot_data = tree.export_graphviz(clf_gini, out_file=None, feature_names=['T','H','CO'],class_names=[0,1], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('C:/Users/Anupam Shukla/Downloads/dtaset')