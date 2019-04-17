import sys
import numpy
import matplotlib
import pandas
import sklearn


print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('Python: {}'.format(pandas.__version__))
print('Python: {}'.format(sklearn.__version__))
import numpy as np
from sklearn import preprocessing
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

#loading the data
names = ['id', 'clump_thickness','uniform_cell_size','uniform_cell_shape',
         'marginal_adhesion','signle_epithelial_size','bare_nuclei',
		 'bland_chromatin','normal_nucleoli','mitoses','class']
		 
df = pd.read_csv("data1.csv", names=names)

#preprocess the data
df.replace('?', -99999, inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)

#print the shape of dataset
print(df.shape)

#do dataset visualization
print(df.loc[6])
print(df.describe())

#plot histograms for each variable
df.hist(figsize = (10,10))
plt.show()

#create scatter plot matrix
scatter_matrix(df, figsize = (18,18))
plt.show()

# create X and Y datasets for training
X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])
""""
X_train = cross_validation.train_test_split(X,Y,test_size = 0.2)
X_test = cross_validation.train_test_split(X,Y,test_size = 0.2)
Y_train = cross_validation.train_test_split(X,Y,test_size = 0.2)
Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)
"""
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
#specify testing options
seed = 8
scoring = 'accuracy'

#define the models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors =5)))
models.append(('SVM',SVC()))


#evaluation each model in turn
result = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train , Y_train, cv=kfold, scoring=scoring)
	result.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
#make prediction on valid dataset
for name, model in models:
	model.fit(X_train,Y_train)
	predictions = model.predict(X_test)
	print(name)
	print(accuracy_score(Y_test,predictions))
	print(classification_report(Y_test,predictions))
    
clf = SVC()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)

example = np.array([[4,2,1,1,1,2,3,2,10]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)
print(prediction)