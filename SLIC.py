from sys import argv
import os
from matplotlib import pyplot as plt
from skimage.segmentation import slic,mark_boundaries
from skimage import color, io, img_as_ubyte
from skimage.future.graph import rag_mean_color, cut_threshold





def get_image(file_path):
    """
    Reads an image and returns it with its LAB color space version

    Parameters
    ----------
    file_path : name of an image

    Returns
    -------
    img : loaded image
    lab_image : image in LAB color space

    """
    try:
        img = io.imread(file_path)
    except:
        print("File is not a valid picture") 
    
    image = img_as_ubyte(img)
    #image = color.bgr2rgb(image)
    return image


def show_img(img):
    width = 8.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img/255)
    
def create_superpixel(image,amount,image_name):
    """
    
    Creates superpixels of the specified size from the given photo.
    Saves series of images with labels every 10 iterations

    Parameters
    ----------
    image : image to segment
    size : an average superpixel size measured in pixels

    Returns
    -------
    labels :  the segmentation labeling of the image after all iterations

    """


    path = os.path.join(os.getcwd())
    os.makedirs(os.path.join(path, "data/" + image_name + "_SLIC"), exist_ok=True)
    number_of_iterations = 1
    for i in range(1,5):
        number_of_iterations = number_of_iterations*i
        labels = slic(image, n_segments=amount, max_iter=number_of_iterations, enforce_connectivity=True,  slic_zero=False,start_label=1)
        image_with_boundaries = mark_boundaries(image, labels, (0, 0, 0))

        image_with_boundaries = img_as_ubyte(image_with_boundaries)
        io.imsave("data/" + image_name + "_SLIC/" + str(number_of_iterations) +"_SLIC.jpg", image_with_boundaries)
    
    labels = slic(image, n_segments=amount, max_iter=200, enforce_connectivity=True,  slic_zero=False,start_label=1)
    rag = rag_mean_color(image, labels)
    final_labels = cut_threshold(labels, rag,25)
    plt.figure()
    plt.imshow(final_labels, cmap='gray'), plt.axis('off')
    final_label_rgb = color.label2rgb(final_labels, image, kind='avg', bg_label=0)
    plt.figure()
    plt.imshow(final_label_rgb), plt.axis('off')
    image_with_boundaries = mark_boundaries(image, labels, (0, 0, 0))
    plt.figure()
    plt.imshow(image_with_boundaries, cmap='gray'), plt.axis('off')
    io.imsave("data/" + image_name + "_SLIC/" + image_name+"_final5.jpg", final_label_rgb)
    io.imsave("data/" + image_name + "_SLIC/" +"200_SLIC.jpg", image_with_boundaries)
    

def main(file_path,amount_superpixel):
    
    """
    
    Runs all functions, creates a new label image and saves it.

    Parameters
    ----------
    file_path: name of an image
    size : number of desired superpixels

    """
    image_name = file_path.split(".")[0]
    image = get_image(file_path)
    create_superpixel(image, amount_superpixel,image_name)
    

    

if __name__ == '__main__':

    main("01.jpg", 100)
    #main(argv[1],int(argv[2]))

