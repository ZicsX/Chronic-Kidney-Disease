import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import graphviz
import sklearn.datasets as dtst

df = pd.read_csv('data.csv')

Y = df['class'].values

X = df
del X['class']
del X['Unnamed: 0']

cols = X.columns
numeric = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

emp = None

for j in cols:
    
	if emp is None:
        
		emp = pd.DataFrame(X[j], columns=[j])

	else:
        
		emp = emp.join(X[j])

    
	if not j in numeric:
        
		emp = pd.get_dummies(emp, columns=[j])
X = emp.values



encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)



model = DecisionTreeClassifier()

model.fit(X_train,Y_train)

score = model.score(X_test, Y_test)


ypred = model.predict(X_test)

print('Accuracy using Decision Tree Classifier: ', score*100, '%')

print(classification_report(Y_test,ypred))
print(confusion_matrix(Y_test,ypred))

