#DONEZO FUCK THIS FILE
#FILE IS GREAT AS OF FEB 26TH
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import numpy as np
import scikitplot as skplt
import statistics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

os.chdir('D:\\RESEARCH\\metadata')


df = pd.read_csv('D:\\RESEARCH\\metadata\\ModelData.csv', low_memory=False)
#print(df.head())

df['electric_heat'] = [
    1 if typ == 'Yes' else 0 for typ in df ['in.hvac_has_zonal_electric_heating']
]

df.drop('in.hvac_has_zonal_electric_heating', axis=1, inplace=True)

#print(df.head())

print("-----------------")
X = df[['January','February','March','April','May','June','July','August','September','October','November','December','January Again?']]
""" print("Features Head:")
print(X.head())
print("Features Sample:")
print(X.sample(10))
print("-----------------") """

Y = df[['electric_heat']]
""" print("Labels Head:")
print(Y.head())
print("Labels Sample:")
print(Y.sample(10)) """

rus = RandomUnderSampler(sampling_strategy = 'not minority')
X_res, Y_res = rus.fit_resample(X,Y)

#print(X_res)
#print(Y_res)

X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=1, stratify=Y_res)

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

tf.random.set_seed(1)


model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(13, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model_1.fit(X_train_scaled, Y_train, epochs=800, batch_size = 8)
#test_loss, test_acc = model_1.evaluate(X_test, Y_test)
#print(test_acc)
predictions = model_1.predict(X_test_scaled)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
print(confusion_matrix(Y_test, prediction_classes))
print(f'Accuracy: {accuracy_score(Y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(Y_test, prediction_classes):.2f}')
print(f'Recall: {recall_score(Y_test, prediction_classes):.2f}')


y_hat = model_1.predict(X_test_scaled)
y_hat = [0 if val <0.5 else 1 for val in y_hat]
print(accuracy_score(Y_test, y_hat))
print(recall_score(Y_test, y_hat))
print(precision_score(Y_test, y_hat))

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.plot(
    np.arange(1, 801), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 801), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 801), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 801), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()







""" class_count_1, class_count_0 = Y.value_counts()
print(class_count_0)
print(class_count_1)
total = class_count_0 + class_count_1
print(total)

weight_for_0 = (1/class_count_0) * (total/2.0)
weight_for_1 = (1/class_count_1) * (total/2.0)

print(weight_for_0)
print(weight_for_1)

#print(len(X))
#set 11,827 samples for training, 2956 for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1, stratify=Y)

classcount0, classcount1 = Y_train.value_counts()
print(classcount0) 
print(classcount1) 
classycount0, classycount1 = Y_test.value_counts()
print(classycount0) 
print(classycount1) 

#smote = SMOTE(sampling_strategy='minority')
#X_sm, Y_sm = smote.fit_resample(X_train, Y_train)

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

tf.random.set_seed(1)


model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(13, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model_1.fit(X_train_scaled, Y_train, epochs=200, batch_size = 2, class_weight = {0: weight_for_0, 1: weight_for_1})
#test_loss, test_acc = model_1.evaluate(X_test, Y_test)
#print(test_acc)
predictions = model_1.predict(X_test_scaled)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
print(confusion_matrix(Y_test, prediction_classes))
print(f'Accuracy: {accuracy_score(Y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(Y_test, prediction_classes):.2f}')
print(f'Recall: {recall_score(Y_test, prediction_classes):.2f}')


y_hat = model_1.predict(X_test_scaled)
y_hat = [0 if val <0.5 else 1 for val in y_hat]
print(accuracy_score(Y_test, y_hat))
print(recall_score(Y_test, y_hat))
print(precision_score(Y_test, y_hat))

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.plot(
    np.arange(1, 201), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 201), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 201), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 201), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()

#this model has a high recall and low EVERYTHING ELSE. according to tensorflow api this is normal with
#highly imbalanced data. talk about the pros and cons in the paper and why i decided to go with recall """