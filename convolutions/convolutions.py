from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):

    # grab the dimestions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # to ensure that the dimention of input image and output image are the same
    # padding (replicate, zero padding)
    pad = (kW - 1) // 2 # if kw=3 then pad=1 --> allocate 1 row for replicate/zero padding
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # apply the actual convolution to our image

    # loop over the input image
    # sliding the kernel from left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # ROI:
            # distance y: 3 --> k
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # perform the actual convolution
            k = (roi * K).sum()
            # store the convolved value in the output
            output[y - pad, x - pad] = k

    # finish up the convolved method

    # rescale the output image
    # bring our output image back into the range [0,255] --> rescale_intensity function of scikit-image 
    output = rescale_intensity(output, in_range=(0, 255))
    # convert image back to an unsigned 8-bit integer data type
    output = (output * 255).astype("uint8")

    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# define two kernels used for blurring and smoothing an image

# 1/S --> S is the total number of entries in the matrix
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
        
# Laplacian kernel used to detect edge-like regions
laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

# The Sobel kernels to detect edge-like regions along both the x and y axis
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# construct an emboss kernel
emboss = np.array((
    [-2, -1, 2],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

# construct the kernel bank, a list of kernels we’re going to apply
# using both our custom ‘convole‘ function and OpenCV’s ‘filter2D‘ function
kernelbank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss))

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, K) in kernelbank:
    # apply the kernel to the grayscale image
    # using both ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    # show the output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    