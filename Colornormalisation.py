# https://youtu.be/_GAhbrGHaVo (watch this video)

# Uses Reinhard color transfer method to change colour of stain slides or pretty much anything to the same color family. AKA normalise the colours
# I have adapted this to my situation and added a further step of getting the pixel values in the range of 0 and 1.
"""
Reinhard color transfer
Based on the paper: https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

This approach is suitable for stain normalization of pathology images where
the 'look and feel' of all images can be normalized to a template image.
This can be a good preprocessing step for machine learning and deep learning
of pathology images.

"""

import numpy as np
import cv2
import os

#input output directories
# input_dir = "input_images/"
input_dir = "train_images/"
input_image_list = os.listdir(input_dir)

# output_dir = "output_images/"
output_dir = "train_0-1/"

# Calculates mean and standard deviation of an image
def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2)) #rounding up mean value
    x_std = np.hstack(np.around(x_std, 2)) #rounding up standard deviation
    return x_mean, x_std

#Select location for template image and extract color pallete

# template_img = cv2.imread('template_images/sunset_template.jpg') #example given by original programmer
template_img = cv2.imread('template-image/fdfa9f4db8b4.tif') #image used as base color reference
template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB) #convert image to LAB color space
template_mean, template_std = get_mean_and_std(template_img) #Calculate mean and std deviation for template

#Get total number of input images and initialise a counter
total_files = len(input_image_list)
processed_files = 0

#process every image one at a time
for img in (input_image_list):
    print(f"Processing file: {img} ({processed_files}/{total_files})")
    processed_files += 1

    #load and convert image to LAB color space
    input_img = cv2.imread(input_dir + img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)

    #Calculate mean and standard deviation
    img_mean, img_std = get_mean_and_std(input_img)

    #Get image height width and channels
    height, width, channel = input_img.shape

    #loop through each pixel and apply Reinhard color transfer
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                x = input_img[i, j, k]
                x = ((x - img_mean[k]) * (template_std[k] / img_std[k])) + template_mean[k]
                x = round(x)
                # boundary check
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                input_img[i, j, k] = x

    # Normalize pixel values (makes the image look grey)
    input_img = input_img.astype(np.float32) / 255.0

    #convert the LAB image back to BGR color space and save the processed image
    input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_dir+img, input_img)
