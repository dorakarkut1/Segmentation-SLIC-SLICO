import os
import cv2
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from skimage.segmentation import slic,mark_boundaries
from skimage import color, io, img_as_ubyte
from skimage.future.graph import rag_mean_color, cut_threshold



def get_image(file_path):
    """
    Reads an image and returns it 

    Parameters
    ----------
    file_path : name of an image if in the same directory or path to the image

    Returns
    -------
    image : loaded image

    """

    try:
        img = io.imread(file_path)
    except:
        print("File is not a valid picture") 
    
    image = img_as_ubyte(img)

    return image
    
def create_superpixel(image,amount,image_name):
    """
    
    Creates superpixels of the specified size from the given photo.
    Saves series of images with labels every 10 iterations

    Parameters
    ----------
    image : image to segment
    amount : number of desired superpixels
    image_name : name for saving 

    Returns
    -------
    final_label_average :  labeled image, color of superpixel is even out

    """

    path = os.path.join(os.getcwd())
    os.makedirs(os.path.join(path, "data/" + image_name + "_SLIC"), exist_ok=True)
    number_of_iterations = 1

    for i in range(1,4):
        number_of_iterations = number_of_iterations*i
        labels = slic(image, n_segments=amount, max_iter=number_of_iterations, enforce_connectivity=True,  slic_zero=False,start_label=1)
        image_with_boundaries = mark_boundaries(image, labels, (0, 0, 0))
        image_with_boundaries = img_as_ubyte(image_with_boundaries)
        io.imsave("data/" + image_name + "_SLIC/" + str(number_of_iterations) +"_SLIC.jpg", image_with_boundaries)
    
    labels = slic(image, n_segments=amount, max_iter=200, enforce_connectivity=True,  slic_zero=False,start_label=1)
    rag = rag_mean_color(image, labels)
    final_labels = cut_threshold(labels, rag,25)
    final_label_average = color.label2rgb(final_labels, image, kind='avg', bg_label=0)
    image_with_boundaries = mark_boundaries(image, labels, (0, 0, 0))

    plt.figure()
    plt.imshow(image_with_boundaries, cmap='gray'), plt.axis('off')
    plt.show()

    io.imsave("data/" + image_name + "_SLIC/" + image_name +"_final.jpg", final_label_average)
    io.imsave("data/" + image_name + "_SLIC/" +"200_SLIC.jpg", image_with_boundaries)
    return final_label_average
    

def post_processing(Image,image_name):
    """
    
    Creates binary image from given image in grayscale. 

    Parameters
    ----------
    Image : image to transform
    image_name : name for saving 

    Returns
    -------
    None

    """

    kernel = np.ones((7,7), dtype='uint8') 
    #Image = 255-Image  
    #Image = cv2.morphologyEx(Image, cv2.MORPH_OPEN, kernel) # use for cleaning noises
    #Image = cv2.morphologyEx(Image, cv2.MORPH_CLOSE, kernel) # use for cleaning noises
    #Image = remove_small_objects(Image, min_size= 10000) # use if object has bigger noise (e.g. hole)
    binary = cv2.threshold(Image, 100, 255, cv2.THRESH_BINARY)[1]
    binary_bitwise = 255-binary #if image has to be inverted

    plt.subplot(1, 2, 2), plt.imshow(binary_bitwise, cmap='gray'), plt.title('Result-bitwise'), plt.axis("off")
    plt.subplot(1, 2, 1), plt.imshow(binary, cmap='gray'), plt.title('Result'), plt.axis("off")
    io.imsave("data/" + image_name + "_SLIC/" + image_name +"_final2.jpg", binary)
    plt.show()

def main(file_path,amount_superpixel):
    
    """
    
    Runs all functions, creates a set of segmented images and final binary image and saves them into folder   

    Parameters
    ----------
    file_path: name of an image if in the same directory or path to the image
    amount_superpixel : number of desired superpixels

    """
    image_name = file_path.split(".")[0]
    image = get_image(file_path)
    segmented = create_superpixel(image, amount_superpixel, image_name)
    post_processing(segmented,image_name)
    

    

if __name__ == '__main__':

    main("01_1.jpg", 100)
    #main(argv[1],int(argv[2]))

