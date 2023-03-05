#THIS FILE IS DONE AND WORKS AND IM SO HAPPY

#CONFIRMED DONE AS OF FEB 26TH (CAN STILL BE OPTIMIZED)


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
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.chdir('D:\\RESEARCH\\metadata')

df = pd.read_csv('D:\\RESEARCH\\metadata\\ModelData.csv', low_memory=False)

df['cooling_system'] =[
    0 if typ == 'None' else 1 for typ in df['in.hvac_cooling_type']
]

df.drop('in.hvac_cooling_type', axis=1, inplace=True)
print(df.head())

print("-----------------")
X = df[['January','February','March','April','May','June','July','August','September','October','November','December','January Again?']]
print("Features Head:")
print(X.head())
print("Features Sample:")
print(X.sample(10))
print("-----------------")

Y = df[['cooling_system']]
print("Labels Head:")
print(Y.head())
print("Labels Sample:")
print(Y.sample(10))

print(len(X))
#set 11,827 samples for training, 2956 for testing
X_train, Y_train = X[:11827], Y[:11827]
X_test, Y_test = X[11827:], Y[11827:]

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

smote = SMOTE(sampling_strategy='minority')
X_sm, Y_sm = smote.fit_resample(X_train_scaled, Y_train)

tf.random.set_seed(42)


model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
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

history = model_1.fit(X_sm, Y_sm, epochs=100, batch_size = 32)
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

y_hat = model_1.predict(X_test_scaled)
y_hat = [0 if val <0.5 else 1 for val in y_hat]
print(accuracy_score(Y_test, y_hat))
print(recall_score(Y_test, y_hat))
print(precision_score(Y_test, y_hat))

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.plot(
    np.arange(1, 101), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 101), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 101), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 101), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()






""" print(X.shape, Y.shape)
print(Y.loc[Y.index[2]])
print(Y.value_counts())

class_count_1, class_count_0 = Y.value_counts()
print(class_count_1)
print(class_count_0)


smote = SMOTE(sampling_strategy='minority')
X_sm, Y_sm = smote.fit_resample(X, Y)

print(X_sm)
print(Y_sm)

class_count_1, class_count_0 = Y_sm.value_counts()
print(class_count_1)
print(class_count_0)


#tf.random.set_seed(36)
model_1 = tf.keras.Sequential([
    #tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(250, input_dim = 13, activation="relu"),
    tf.keras.layers.Dense(250, activation="relu"),
    #tf.keras.layers.Dense(150, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                metrics=['accuracy'])
history = model_1.fit(X_sm, Y_sm, epochs=50)
test_loss, test_acc = model_1.evaluate(X_test, Y_test, batch_size = 32)
print(test_acc)   """








  