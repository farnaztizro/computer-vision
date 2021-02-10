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
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # apply the actual convolution to our image