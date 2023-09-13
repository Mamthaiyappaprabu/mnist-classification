# CONVOLUTIONAL DEEP NEURAL NETWORK FOR DIGIT CLASSIFICATION :

## AIM :

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## PROBLEM STATEMENT AND DATASET :

The MNIST (Modified National Institute of Standards and Technology) database is a large database of handwritten numbers or digits that are used for training various image processing s!
ystems. The dataset also widely used for training and testing in the field of machine learning.

![Screenshot 2023-09-13 112924](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/5117f743-1b8d-46d4-8f75-d93644401b72)

## NEURAL NETWORK MODEL :

![Screenshot 2023-09-12 114853](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/f5623856-42d9-4a37-ada0-364c31696b2c)

## DESIGN STEPS :

### STEP 1 :
Import tensorflow and preprocessing libraries
### STEP 2 :
Download and load the dataset
### STEP 3 :
Scale the dataset between it's min and max values
### STEP 4 :
Using one hot encode, encode the categorical values
### STEP 5 :
Split the data into train and test
### STEP 6 :
Build the convolutional neural network model
### STEP 7 :
Train the model with the training data and Plot the performance plot
Write your own steps
### STEP 8 :
Evaluate the model with the testing data and Fit the model and predict the single input
## PROGRAM :
### NAME : MAMTHA I
### REG NO : 212222230076
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)


metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

```
## PREDICTION FOR A SINGLE INPUT :
```
img = image.load_img('/content/img.png')

type(img)

img = image.load_img('/content/img.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

```

## OUTPUT :


### TRAINING LOSS VS VALIDATION LOSS AND ACCURACY VS VAL_ACCURACY :

![Screenshot 2023-09-13 114403](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/5d5f4b81-cc75-474d-b0d2-36e9f164c3cd)

![Screenshot 2023-09-13 114414](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/550b0bd9-3496-4c15-b525-0dd3513af8d2)

### CLASSIFICATION REPORT :
![Screenshot 2023-09-13 114634](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/edd8f393-b24b-4b98-ab5b-cc47530fd0cb)

### CONFUSION MATRIX :

![Screenshot 2023-09-13 114647](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/0723f75b-c2d6-4cc4-a634-ba4a38d2662d)


### NEW SAMPLE DATA PREDICTION :
![Screenshot 2023-09-13 114740](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/3db9ee73-c8ef-4ece-8d0b-d41d8003b045)
![Screenshot 2023-09-13 114748](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/b0fad0cc-2aa6-47de-abb9-e850f78f9b0e)

![Screenshot 2023-09-13 114756](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/56d3f72f-93ab-4b5a-892e-eb9a2eacaaca)

![Screenshot 2023-09-13 114829](https://github.com/Mamthaiyappaprabu/mnist-classification/assets/119393563/484daf2f-2b67-4103-9e41-2ee5e090919d)


## RESULT :
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
