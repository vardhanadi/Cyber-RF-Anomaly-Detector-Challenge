import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA =  pd.read_csv("Cyber-RF_Anomaly_Detector_Challenge_Dataset_TrainingSet_80.csv")
DATA.head()

DATA['Class'].value_counts() # Anomalous = 381, nornmal = 349

# Performing PCA to pick 20 most important colums
DATA['Class'] = DATA['Class'].map({'anomalous' : 1, 'normal': 0})
DATA.head()

X = DATA.iloc[:,0:52].astype(float)
X = np.asanyarray(X)
X  = np.reshape(X, (730,52,1))

Y = DATA.iloc[:, -1]
Y = np.asanyarray(Y)
Y = np.reshape(Y, (730,1))
#------------------ Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

from tensorflow.keras.utils import to_categorical
Y_train_enc = to_categorical(Y_train)
Y_test_enc = to_categorical(Y_test)

print('Training data shape : ', X_train.shape)
print('Training label shape : ',  Y_train_enc.shape)
print('Testing data shape : ', X_test.shape)
print('Testing label shape : ',  Y_test_enc.shape)

#----------------- Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(32, activation = 'relu', input_dim = (52,1)))
model.add(LSTM(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation= 'softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train_enc,epochs=1000, verbose=2) # hide the output because we have so many epochs

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
import seaborn as sns

y_prob = model.predict(X_test) # Probabilities
y_classes = y_prob.argmax(axis=-1) #Convert probabilities to classes
print("Accuracy on Test Set: ", accuracy_score(Y_test, y_classes)) #Accuracy on test set

precision  =  precision_score(Y_test, y_classes)
print("Precision: ", precision)

f_score = f1_score(Y_test, y_classes)
print("F1 Score: ", f_score)

cf = confusion_matrix(Y_test, y_classes)
sns.heatmap(cf, annot=True)
plt.show()



