#training model for clutter code
# vgg16 for good and bad website image classification
# To explore mobilenetv2
# To explore VGG16

from sklearn.metrics import confusion_matrix
from keras.models import load_model
from tensorflow.keras import layers, models
from keras.utils.vis_utils import plot_model
import os  # all below imported packeages are important
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
import json
from skimage import color
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# get the path/directory

imagelist = []
filelist = []
labellist = []
featurelist = []
fimagelist = []
scaler = MinMaxScaler()
folder_dir = "C://aditi/competitions/hackerx4/clutterDataset/train/moreClutter"
#jpeg, png, webp  

""" for filename in os.listdir(folder_dir):
    if (filename.endswith("webp")):
        print(filename)
        filename1 = filename.replace('webp', 'jpg')
        os.rename(os.path.join(folder_dir, filename),
                  os.path.join(folder_dir, filename1))
        print(filename1) """

""" k = 1
for filename in os.listdir(folder_dir):
    print(filename)
    img = cv2.imread(os.path.join(folder_dir, filename))
    if img is not None:
        img1 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        norm_image = cv2.normalize(
            img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imagelist.append(norm_image)
        filelist.append(filename)
        label = 0  # 'bad_website'
        labellist.append(label)
        print(label)
        k = k+1

print(len(imagelist))
print(len(labellist))
print(k)  # 125 bad website images
k1 = 1

folder_dir = "C://aditi/competitions/hackerx4/clutterDataset/train/lessClutter"

for filename in os.listdir(folder_dir):
    print(filename)
    img = cv2.imread(os.path.join(folder_dir, filename))
    if img is not None:
        img1 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        norm_image = cv2.normalize(
            img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imagelist.append(norm_image)
        filelist.append(filename)
        label = 1  # 'good_website'
        labellist.append(label)
        print(label)
        k1 = k1+1

print(len(imagelist))
print(len(labellist))
print(k1)  # 136 bad website images
#cv2.imshow('image', imagelist[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()#all 3 statements must be required while using imshow
# cv2.waitKey(1)
#labellist1 = [int(i) for i in labellist]
y = np.array(labellist)  # convert list into array
# x=np.array(featurelist)
x = np.array(imagelist)

np.savez("C:\\aditi\\competitions\\hackerx4\\x_images_arrays_forClutter", x)
np.savez("C:\\aditi\\competitions\\hackerx4\\y_labels_forClutter", y) """ 

""" """ 
x1 = np.load("C:\\aditi\\competitions\\hackerx4\\x_images_arrays_forClutter.npz")
y1 = np.load("C:\\aditi\\competitions\\hackerx4\\y_labels_forClutter.npz")

print(x1.files)
print(y1.files)

x2 = x1['arr_0']
y2 = y1['arr_0']
print(x2.shape)
print(y2.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y2, test_size=0.20, random_state=0, stratify=y2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


def to_categorical1(y, num_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not num_classes:
        num_classes = np.max(y)
    Y = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        Y[i, y[i]-1] = 1.
    return Y


y_train1 = to_categorical1(y_train, num_classes=2)
y_test1 = to_categorical1(y_test, num_classes=2)

print(y_train1[1:5])
print(y_test1[1:5])

#print("loading VGG16 with imagenet weights")
print("loading MobileNetV2 with imagenet weights")
# Loading VGG16 model
#base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Not trainable weights
base_model.summary()


flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(
    150, activation='selu', kernel_initializer="lecun_normal")
bn1 = layers.BatchNormalization()
dp1 = layers.Dropout(rate=0.2)
dense_layer_2 = layers.Dense(
    150, activation='selu', kernel_initializer="lecun_normal")
bn2 = layers.BatchNormalization()
dp2 = layers.Dropout(rate=0.2)
dense_layer_3 = layers.Dense(
    150, activation='selu', kernel_initializer="lecun_normal")
bn3 = layers.BatchNormalization()
dp3 = layers.Dropout(rate=0.2)
dense_layer_4 = layers.Dense(
    150, activation='selu', kernel_initializer="lecun_normal")
bn4 = layers.BatchNormalization()
dp4 = layers.Dropout(rate=0.2)
prediction_layer = layers.Dense(2, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    bn1,
    dp1,
    dense_layer_2,
    bn2,
    dp2,
    dense_layer_3,
    bn3,
    dp3,
    dense_layer_4,
    bn4,
    dp4,
    prediction_layer
])

print(model.summary())

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)


es = EarlyStopping(monitor='val_accuracy', mode='max',
                   patience=3,  restore_best_weights=True)

mc = ModelCheckpoint('C://aditi/competitions/hackerx4/best_model_mobilenet_forClutter.hdf5',
                     monitor='val_loss', mode='min', verbose=1, save_best_only=True)

print("training/fine tuning mobilenet last layers for website good_bad_Clutter dataset")

hist = model.fit(x_train, y_train1, epochs=15, validation_split=0.2, batch_size=32, callbacks=[es, mc])

saved_model = load_model('C://aditi/competitions/hackerx4/best_model_mobilenet_forClutter.hdf5')
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()

_, train_acc = saved_model.evaluate(x_train, y_train1, verbose=0)
_, test_acc = saved_model.evaluate(x_test, y_test1, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


y_pred = saved_model.predict(x_test)
y_pred2 = np.argmax(y_pred, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
# Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(y_test2, y_pred2, normalize='pred')
print(result)

print(y_test2[1:5])
print(y_pred2[1:5])
