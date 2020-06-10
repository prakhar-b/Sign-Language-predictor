# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:54:17 2020

@author: prakhar
"""

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import pickle
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics
# from sklearn.externals import joblib
# from sklearn_export import Export

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()


# grab the list of images that we'll be describing
print("[INFO] describing images...")
#imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images(r"C:\Users\prakhar\Documents\GitHub\Sign-Language\DatasetforKNN"))
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
    pixels = image_to_feature_vector(gray)
    hist = extract_color_histogram(image)

	# update the raw images, features, and labels matricies,
	# respectively
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

	# show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=5,
	n_jobs=-1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
#
## train and evaluate a k-NN classifer on the histogram
## representations
##print("[INFO] evaluating histogram accuracy...")
##model = KNeighborsClassifier(n_neighbors=2,
##	n_jobs=-1)
##model.fit(trainFeat, trainLabels)
##acc = model.score(testFeat, testLabels)
##print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


y_pred = model.predict(testRI)
conf_mat = confusion_matrix(testLabels, y_pred, labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
accu_score = sklearn.metrics.accuracy_score(testLabels, y_pred, normalize=True, sample_weight=None)
prec_score = sklearn.metrics.precision_score(testLabels, y_pred, average = None)

df_cm = pd.DataFrame(conf_mat, index = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                  columns = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
plt.figure(figsize = (30,21))
sn.heatmap(df_cm, annot=True)
classification_report = sklearn.metrics.classification_report(testLabels,y_pred,labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])


# filename = 'modelk5.pkl'
# pickle.dump(model, open(filename, 'wb'))

