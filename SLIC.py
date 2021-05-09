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
    io.imsave("data/" + image_name + "_SLIC/" + image_name +"_final5.jpg", final_label_average)
    io.imsave("data/" + image_name + "_SLIC/" +"200_SLIC.jpg", image_with_boundaries)
    return final_label_average
    

def post_processing(I,file_path):
    
    se = np.ones((7,7), dtype='uint8') 
    #close = cv2.inRange(I, 0, 255)
    I = 255-I
    Img = I > 0

    close = cv2.threshold(I, 40, 255, cv2.THRESH_BINARY)[1]
    #opens = cv2.morphologyEx(close, cv2.MORPH_OPEN, se)
    #close = cv2.morphologyEx(opens, cv2.MORPH_CLOSE, se)
    #close = cv2.morphologyEx(opens, cv2.MORPH_CLOSE, se)

    #clean = remove_small_objects(Img,min_size= 10000)
    image_name = file_path.split(".")[0]

    close_bitwise = 255-close
    
    cv2.imwrite(image_name+ "2.jpg", close_bitwise)

    plt.subplot(1, 2, 2), plt.imshow(close_bitwise, cmap='gray'), plt.title('Result-bitwise'), plt.axis("off")
    plt.subplot(1, 2, 1), plt.imshow(close, cmap='gray'), plt.title('Result'), plt.axis("off")
    plt.show()

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
    segmented = create_superpixel(image, amount_superpixel, image_name)
    

    

if __name__ == '__main__':

    main("01.jpg", 100)
    #main(argv[1],int(argv[2]))

