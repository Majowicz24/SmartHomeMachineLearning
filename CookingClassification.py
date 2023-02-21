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

history = model_1.fit(X_sm, Y_sm, epochs=100, batch_size = 16)
#test_loss, test_acc = model_1.evaluate(X_test, Y_test)
#print(test_acc)

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
