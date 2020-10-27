import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.backend import eval
from keras import optimizers

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


def neural(X):
    
	model = Sequential()
    
	model.add(Dense(150, input_dim=X.shape[1], activation='sigmoid'))

	model.add(Dense(100, activation='sigmoid'))
    
	model.add(Dense(50, activation='sigmoid'))
    
	model.add(Dense(8, activation='sigmoid'))
    
	model.add(Dense(1, activation='sigmoid'))
    
	return model
    
model = neural(X_train)

#adam = optimizers.SGD(lr=0.01, momentum=0.001)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=150, batch_size=10)


score = model.evaluate(X_test, Y_test)
prediction = model.predict(X_test)
#############
binarizer = Binarizer(threshold=0.50).fit(prediction)
binary = binarizer.transform(prediction)
#############
y_test = Y_test
y_pred = binary



print('Accuracy using Back Propagation: ', score*100, '%')

print("classification_report = ")
print(classification_report(y_test,y_pred))
print("confusion_matrix = ")
print(confusion_matrix(y_test,y_pred))

print('learning rate: =')
print(eval(model.optimizer.lr))

#print('momentum: ', eval(model.optimizer.momentum))
