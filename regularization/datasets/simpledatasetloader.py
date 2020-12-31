import numpy as np #numerical processing
import cv2
import os # we can extract the names of subdirectories in image paths.

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []   
        labels = []

        for(i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # Now image is loaded from disk, we can preprocess it (if necessary)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    #apply the loop
                    image = p.preprocess(image)
        
            # while image has beem preprocessed we update the data and lable lists
            data.append(image)
            labels.append(label)        

            # printing updates, returning 2-tuple of data and lables
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))

        return(np.array(data), np.array(labels))        
