# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13 ].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_encoder_X1 = LabelEncoder()
X[:,1] = label_encoder_X1.fit_transform(X[:,1])
label_encoder_X2 = LabelEncoder()
X[:,2] = label_encoder_X2.fit_transform(X[:,2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:,1:]

# Split between train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Make the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Dropout to avoid overfitting
classifier = Sequential()

# Add Input layer and first hidden layer with dropout
classifier.add(Dense( output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate=0.1))
# Add the second hidden layer
classifier.add(Dense( output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

# Add the output layer
classifier.add(Dense( output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN -> Apply stochastic gradient descent
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit ANN to the training set
classifier.fit(X_train,y_train,epochs=100, batch_size=10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting for an example
X_eg = np.array([600,'France','Male',40,3,60000,2,1,1,50000 ])
X_eg = X_eg.reshape(1,-1)
X_eg[:,1] = label_encoder_X1.transform(X_eg[:,1])
X_eg[:,2] = label_encoder_X2.transform(X_eg[:,2])
X_eg = one_hot_encoder.transform(X_eg).toarray()
X_eg = X_eg[:,1:]
X_eg = sc.transform(X_eg)
y_eg_pred = classifier.predict(X_eg)

# K-Fold Cross validation with Keras wrapper of cross validation by scikit learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense( output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense( output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense( output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

# Hyperparameter tuning of ANN with Grid Search CV

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense( output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense( output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense( output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile( optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# remove  batch_size and epochs as these are hyperparameters w ewant to tune with grid search
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
parameters = {'batch_size':[25,32],
              'epochs':[100,300],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring = 'accuracy',
                           cv = 3)
grid_search = grid_search.fit(X_train, y_train)
best_prameters=grid_search.best_params_
best_accuracy=grid_search.best_score_