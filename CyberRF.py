import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA =  pd.read_csv("Cyber-RF_Anomaly_Detector_Challenge_Dataset_TrainingSet_80.csv")
DATA.head()

DATA['Class'].value_counts() # Anomalous = 381, nornmal = 349
len(DATA.columns) # 53 COlumns 

# Performing PCA to pick 20 most important colums
DATA['Class'] = DATA['Class'].map({'anomalous' : 1, 'normal': 0})
DATA.head()

X = DATA.iloc[:,0:52].astype(float)
Y = DATA.iloc[:, -1]

#--------------------Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# learning the statistical parameters for each of the data and transforming
Scaled_X = scaler.fit_transform(X)
np.set_printoptions(precision=3)
print(Scaled_X[0:5,:])


#-------------- Split the dataset to train the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(Scaled_X, Y, test_size = 0.2, random_state=2)

from tensorflow.keras.utils import to_categorical
Y_train_enc = to_categorical(Y_train)
Y_test_enc = to_categorical(Y_test)

print('Training data shape : ', X_train.shape, Y_train.shape)
print('Testing data shape : ', X_test.shape, Y_test.shape)
print('Training label shape : ',  Y_train_enc.shape)
print('Testing label shape : ',  Y_test_enc.shape)

#------------ Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
import seaborn as sns

model = Sequential()
model.add(Dense(32, activation = 'tanh', input_shape = (52,2)))
model.add(Dense(64, activation = 'tanh'))
model.add(Dense(32, activation = 'tanh'))
model.add(Dense(2, activation= 'softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train_enc, validation_data = (X_test,Y_test) ,epochs=500, verbose=2) # hide the output because we have so many epochs

 precision  =  precision_score(Y_test, y_classes)
print("Precision: ", precision)

f_score = f1_score(Y_test, y_classes)
print("F1 Score: ", f_score)

cf = confusion_matrix(Y_test, y_classes)
sns.heatmap(cf, annot=True)
plt.show()

