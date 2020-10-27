# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import numpy as np
from statistics import median
import joblib


dataset = pd.read_csv('dataset.csv')

X = dataset.iloc[:, 0:24].values
y = dataset.iloc[:, 24].values

def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

for i in range(0,399):
    if y[i] == 'ckd':
        y[i] = 1
    else:
        y[i] = 0
        
y = y.astype(int)

for a in range(0, 399):
    if X[a][5] == 'normal':
        X[a][5] = 0
    if X[a][5] == 'abnormal':
        X[a][5] = 1
        
for a in range(0, 399):
    if X[a][6] == 'normal':
        X[a][6] = 0
    if X[a][6] == 'abnormal':
        X[a][6] = 1
        
for a in range(0, 399):
    if X[a][7] == 'notpresent':
        X[a][7] = 0
    if X[a][7] == 'present':
        X[a][7] = 1
        
for a in range(0, 399):
    if X[a][8] == 'notpresent':
        X[a][8] = 0
    if X[a][8] == 'present':
        X[a][8] = 1
        
for a in range(0, 399):
    for b in range(18, 24):
        if X[a][b] == 'yes' or X[a][b] == 'good':
            X[a][b] = 0
        if X[a][b] == 'no' or X[a][b] == 'poor':
            X[a][b] = 1
    
for a in range(0,399):
    for b in range(0, 24):
        if(isinstance(X[a][b], int)):
            X[a][b] = float(X[a][b])
        elif(isinstance(X[a][b], str)):
            if(is_float(X[a][b])):
                X[a][b] = float(X[a][b])
                
totals = [0] * 24
added = [0] * 24           
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float)):
            totals[b] += X[a][b]
            added[b] += 1
            
averages = [0] * 24          
for a in range(0, 24):
    averages[a] = totals[a] / added[a]
 
c = 0
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float) == 0):
            X[a][b] = averages[b]
            c += 1
    
X = X.astype(float)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


print('CLASSIFICATION BY Logistic Regression')

print("Logistic Regression")

from sklearn.linear_model import LogisticRegression
lg_classifier = LogisticRegression()
lg_classifier.fit(X_train, y_train)


# Export the model to a file
joblib.dump(lg_classifier, 'lg_classifier.joblib')
print('Model trained and saved')

 
print('Testing And Model Accuracy')

predictions = lg_classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
print("Confusion Matrix")
print(pd.crosstab(y_test, predictions, rownames=['Label'], colnames=['Predicted'], margins=True))

print("Classification Report")
print(classification_report(y_test,predictions))

print("Logistic Regression accuracy_score")
print(accuracy_score(y_test, predictions))

print("F1 Score")
print(f1_score(y_test, predictions, average='binary'))
