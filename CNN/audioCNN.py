import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pydot
import pydotplus
from tensorflow import keras
from keras.utils import plot_model


from numpy import load

train0 = []
train1 = []
test1 = []
test0 = []

for i in range(1,379):
    train0.append(((cv2.imread('Specs/tn ('+str(i)+').png',cv2.IMREAD_GRAYSCALE))/255))

for i in range(1,379):
    train1.append(((cv2.imread('Specs/ty ('+str(i)+').png',cv2.IMREAD_GRAYSCALE))/255))

for i in range(1,163):
    test1.append(((cv2.imread('Specs/y ('+str(i)+').png',cv2.IMREAD_GRAYSCALE))/255))

for i in range(1,163):
    test0.append(((cv2.imread('Specs/n ('+str(i)+').png',cv2.IMREAD_GRAYSCALE))/255))


x_train = train1 + train0
y_train = []
for i in range(756):
    if(i<378):
        y_train.append(1)
    else:
        y_train.append(0)


x_test = test1 + test0
y_test = []
for i in range(324):
    if(i<162):
        y_test.append(1)
    else:
        y_test.append(0)
        
originalYtest = []
originalYtest.append(y_test)


class_names = ["Not", "Yes"]

x_test = np.expand_dims(x_test, axis=-1)
x_train = np.expand_dims(x_train, axis=-1)


y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=2), tf.keras.utils.to_categorical(y_test, num_classes=2)


def calcAccuracy(preds, trues):
    print(preds)
    print(trues)
    correct=0
    n = len(preds)
    for i in range(n):
        if(preds[i]==trues[0][i]):
            correct=correct+1
    return correct/n

results = []
greaterAccuracy = 0.0
epch = 25
for i in range(100):

    model = keras.models.Sequential([
      keras.layers.Input((256,86,1)),
      keras.layers.Conv2D(8, (3,3), padding="same", activation="relu"),
      keras.layers.MaxPool2D(),
      keras.layers.Conv2D(8, (3,3), padding="same", activation="relu"),
      keras.layers.MaxPool2D(),
      keras.layers.Flatten(),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(2, activation="softmax")
    ])

    model.compile("sgd",loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, epochs=epch, use_multiprocessing=True, verbose=0)
    eval = model.evaluate(x_test, y_test)

    predictions = model.predict(x_test)
    y_classes = predictions.argmax(axis=-1)
    print(y_classes)
    accuracy = calcAccuracy(y_classes, originalYtest)

    
    if accuracy > greaterAccuracy:
        greaterAccuracy = accuracy
        print("----- Change in Accuracy: "+str(greaterAccuracy))
        
        
    currentResult = (str(i)+": Evaluation [loss, accuracy]: ", eval, "||| -> "+str(accuracy))
    results.append(currentResult)
    print(currentResult)
    epch=epch+1
    
print(" ************** ------------------------- ***************")

for i in results:
    print(i)
print("Greatest Accuracy: "+str(greaterAccuracy))
with open(r'results.txt', 'w') as fp:
    for item in results:
        fp.write(str(item)+"\n")
    print('Done')