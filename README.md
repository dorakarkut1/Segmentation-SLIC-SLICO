# Segmentation-SLIC-SLICO
A Python program for medical images segmentation using SLIC and SLICO. It was tested for MRI images of the brain with tumor and images of skin cancer. If you want to use it for brain images you need to use STRIPPING.py script first to extract skull from brain. 

A script named STRIPPING.py contains the code needed to perform skull extraction for MRI images of the brain. It should be run from the command line by giving the path to the source image. The resulting image will be saved in the same folder as the source image.

A script named SLIC.py contains the code needed to perform the image segmentation. Image processing is automatically performed after completing segmentation. The script should be run from the command line by specifying the path to the source image and the number of segments of the resulting image. Results of individual stages
segmentation will be saved in the prepared DATA folder. Script SLICO.py is similar to SLIC.py. The only difference is that you do not specify number of segments.

If you want to compare your result to mask prepared by specialist you can use METRICS.py. It should be run from the command line by giving the path to the source image and mask. It returns Dice and Jaccard coefficient. 

Packages used in scripts: OpenCV , Numpy , Matplotlib  and Scikit - image. The required Python version is at least 3.7.

