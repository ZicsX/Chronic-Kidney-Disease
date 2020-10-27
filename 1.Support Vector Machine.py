import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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


from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() # sklearn.preprocessing.LabelEncoder
encoder.fit(Y)
print(Y)
Y = encoder.transform(Y)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)


model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy using Support Veator Machine: ',model.score(X_test,y_test)*100,"%")

from sklearn.metrics import accuracy_score
print('accuracy_score: ',accuracy_score(y_test, y_pred)*100,"%")
print("classification_report = ")
print(classification_report(y_test,y_pred))
print("confusion_matrix = ")
print(confusion_matrix(y_test,y_pred))
