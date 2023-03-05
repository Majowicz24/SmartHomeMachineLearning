#THIS ONE IS PRETTY GOOD, MAYBE COULD USE SOME TUNING
#DONE AS OF FEB 26TH, JUST TUNING LAYERS AND NEURONS BB


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
from sklearn.model_selection import train_test_split

os.chdir('D:\\RESEARCH\\metadata')
df = pd.read_csv('D:\\RESEARCH\\metadata\\ModelData.csv', low_memory=False)


df['electric_cooking'] =[
    0 if typ == 0 else 1 for typ in df['out.electricity.range_oven.energy_consumption']
]

df.drop('out.electricity.range_oven.energy_consumption', axis=1, inplace=True)

print(df.head())

print("-----------------")
X = df[['January','February','March','April','May','June','July','August','September','October','November','December','January Again?']]
print("Features Head:")
print(X.head())
print("Features Sample:")
print(X.sample(10))
print("-----------------")

Y = df[['electric_cooking']]
print("Labels Head:")
print(Y.head())
print("Labels Sample:")
print(Y.sample(10))

class_count_1, class_count_0 = Y.value_counts()
print(class_count_1)
print(class_count_0)

X_train, X_test = train_test_split(X, random_state=42)
Y_train, Y_test = train_test_split(Y, random_state=42)


smote = SMOTE(sampling_strategy='minority')
X_sm_train, Y_sm_train = smote.fit_resample(X_train, Y_train) 

#print(X_sm)
#print(Y_sm)

#class_count_1, class_count_0 = Y_sm.value_counts()
#print(class_count_1)
#print(class_count_0)


scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_sm_train)
X_test_scaled = scalar.transform(X_test)

tf.random.set_seed(42)


model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
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

history = model_1.fit(X_train_scaled, Y_sm_train, epochs=100, batch_size = 32)
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
