from sys import argv
import cv2
import numpy as np
from sklearn.metrics import jaccard_score


def get_image(file_path):
    """
    Reads an image and returns it with its LAB color space version

    Parameters
    ----------
    file_path : name of an image

    Returns
    -------
    img : loaded image tranformed into binary 1D array

    """
    try:
        img = cv2.imread(file_path, 0)
        
    except:
        print("File is not a valid picture")
    img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1]
    return img.flatten()

def dice(segmented, true):
    """
    Calculates Dice metric between segmented labels and ground truth mask

    Parameters
    ----------
    segmented : image with labels
    true : image with ground truth labels

    Returns
    -------
    int: Dice index

    """
    intersection = 0
    true_score = 0
    segmented_score = 0
    for i in range(len(true)):
        if (true[i] == 1) and (segmented[i] == 1) :
            intersection += 1
        if true[i] == 1:
            true_score += 1
        if segmented[i] == 1:
            segmented_score += 1
    return np.round((2 * intersection) / (true_score + segmented_score), 2)
    

    
if __name__ == '__main__':
 
    true = get_image(argv[1])
    segmented = get_image(argv[2])
    
    print("Wskaźnik Jaccard'a wynosi: ", np.round(jaccard_score(true, segmented), 2))
    print("Wskaźnik Dice'a wynosi: ", dice(true, segmented))
