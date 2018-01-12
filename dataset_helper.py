# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:06:20 2017

@author: JT
"""

import os
import numpy as np
import pickle
import pandas as pd
from scipy import misc
import cv2

# Define the paths to data
traindata_path = "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
testdata_path = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/"
test_ground_truth_csv_path = "GTSRB_Final_Test_GT/GT-final_test.csv"
#print(os.listdir(traindata_path))
#print(os.listdir(testdata_path))

# Create a container for the training images and the training class id's
training_images = []
training_classes = []
# Create a container for the test images and the test class id's
test_images = []
test_classes = []
# Define containers for the original image shapes
image_shapes = []
test_image_shapes = []

# Define the height and width of the obtained images
h = 32
w = 32

# Traverse through the training data folder structure
for folder in os.listdir(traindata_path):
    subfolder_path = traindata_path +"/"+folder+"/"
    subfolder = os.listdir(subfolder_path)
    # Get the name of the .csv-file containing the image class-id information
    csvname = [fname for fname in subfolder if fname.endswith("csv")][0]
    #print(csvname)
    #print(subfolder_path)
    csv_in = pd.read_csv(subfolder_path+csvname, delimiter=";")
    #print(csv_in.Filename)
    for image_fname,classid in zip(csv_in.Filename, csv_in.ClassId):
        #print(image_fname + "  " + str(classid))
        # Read the .ppm image data
        image_original = misc.imread(subfolder_path+image_fname)
        # Save the shape of the original image
        image_shapes.append(image_original.shape)
        # Resize the image
        image = cv2.resize(image_original, (h, w))
        # Append the obtained image, class-id pair into the containers
        training_images.append(image)
        training_classes.append(classid)

# Merge the training images and correct class id's into one container
trainset = [np.array(training_images), np.array(training_classes)]
# Save the training set as a pickle
pickle.dump(trainset, open("trainingset.p", "wb"))
print("Training set generated and saved successfully")
       

# Load the .csv-file containing the image class-id information
csv_in = pd.read_csv(test_ground_truth_csv_path, delimiter=";")
#print(csv_in)
# Traverse through the test data
for image_fname,classid in zip(csv_in.Filename, csv_in.ClassId):
    #print(image_fname+"  "+str(classid))
    # Read the .ppm image data
    image_original = misc.imread(testdata_path+image_fname)
    # Save the shape of the original image
    test_image_shapes.append(image_original.shape)
    # Resize the image
    image = cv2.resize(image_original, (h, w))
    # Append the obtained image, class-id pair into the containers
    test_images.append(image)
    test_classes.append(classid)
    
# Merge the test images and the correct class id's into one container
testset = [np.array(test_images), np.array(test_classes)]

# Save the test set as a pickle
pickle.dump(testset, open("testset.p", "wb"))
print("Test set generated and saved successfully")

# Save also the image shapes for further processing
pickle.dump(np.array(image_shapes), open("training_image_shapes.p", "wb"))
pickle.dump(np.array(test_image_shapes), open("test_image_shapes.p", "wb"))

# Print out some statistics
print("Number of training images: {}".format(len(image_shapes)))
print("Number of test images: {}".format(len(test_image_shapes)))
print("Mean training image height: {}".format(np.mean([x[0] for x in image_shapes])))
print("Mean training image width: {}".format(np.mean([x[1] for x in image_shapes])))
print("Mean test image height: {}".format(np.mean([x[0] for x in test_image_shapes])))
print("Mean test image width: {}".format(np.mean([x[1] for x in test_image_shapes])))
# Print out some more statistics
print("Median training image height: {}".format(np.median([x[0] for x in image_shapes])))
print("Median training image width: {}".format(np.median([x[1] for x in image_shapes])))
print("Median test image height: {}".format(np.median([x[0] for x in test_image_shapes])))
print("Median test image width: {}".format(np.median([x[1] for x in test_image_shapes])))

print("alles gut")
