from sys import argv
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt


def stripping(file_path):
    """
    Reads an image, perform skull stripping and saves the result as new image 

    Parameters
    ----------
    file_path : name of an image

    Returns
    -------
    None

    """

    #Read in image
    img  = cv2.imread(file_path)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = np.ones((3,3), dtype = 'uint8') 
    gray = cv2.erode(gray, se, iterations = 1)
    #Threshold the image to binary using Otsu's method
    ret, img_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(img_thresh)
    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0] 
    #Get label of largest component by area
    largest = np.argmax(marker_area) + 1 #Add 1 since you dropped zero above  
    #Get pixels which correspond to the brain
    brain_mask = markers == largest
    brain_out = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask == False] = (0,0,0)
    image_name = os.path.splitext(file_path)[0]
    cv2.imwrite(image_name + "_1.jpg", brain_out)



if __name__ == '__main__':

    stripping(argv[1])
    

    


