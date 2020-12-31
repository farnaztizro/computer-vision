import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # method used when resizing
        self.width = width  #The target width of our input image after resizing
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)    