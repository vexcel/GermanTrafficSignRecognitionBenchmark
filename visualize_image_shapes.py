# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 02:10:32 2017

@author: JT
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Close all the plots
plt.close('all')

# Load the image shape data
training_shapes = pickle.load(open("training_image_shapes.p", "rb"))
test_shapes = pickle.load(open("test_image_shapes.p", "rb"))

# Extract the widths and heights of the images
train_heights = [x[0] for x in training_shapes]
train_widths = [x[1] for x in training_shapes]
test_heights = [x[0] for x in test_shapes]
test_widths = [x[1] for x in test_shapes]

# Visualize the histograms
plt.figure()
plt.hist(train_heights, bins=40, density=True)
plt.title("Training heights")

plt.figure()
plt.hist(train_widths, bins=40, density=True)
plt.title("Training widths")

plt.figure()
plt.hist(test_heights, bins=40, density=True)
plt.title("Test heights")

plt.figure()
plt.hist(test_widths, bins=40, density=True)
plt.title("Test widths")

# Print out some statistics
print("Number of training images: {}".format(len(training_shapes)))
print("Number of test images: {}".format(len(test_shapes)))
print("Mean training image height: {}".format(np.mean([x[0] for x in training_shapes])))
print("Mean training image width: {}".format(np.mean([x[1] for x in training_shapes])))
print("Mean test image height: {}".format(np.mean([x[0] for x in test_shapes])))
print("Mean test image width: {}".format(np.mean([x[1] for x in test_shapes])))
# Print out some more statistics
print("Median training image height: {}".format(np.median([x[0] for x in training_shapes])))
print("Median training image width: {}".format(np.median([x[1] for x in training_shapes])))
print("Median test image height: {}".format(np.median([x[0] for x in test_shapes])))
print("Median test image width: {}".format(np.median([x[1] for x in test_shapes])))
# Print even more statistics
print("Training image height, standard deviation: {}".format(np.std([x[0] for x in training_shapes])))
print("Training image width, standard deviation: {}".format(np.std([x[1] for x in training_shapes])))
print("Test image height, standard deviation: {}".format(np.std([x[0] for x in test_shapes])))
print("Test image width, standard deviation: {}".format(np.std([x[1] for x in test_shapes])))

print("alles gut")
