from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pre_processing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="of jobs for knn distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# step 1

print("[INFO] loading images...")
# grabs the file paths to all images in our dataset
imagePaths = list(paths.list_images(args["dataset"]))

# initialize our SimplePreprocessor used to resize each image to 32×32 pixels 
sp = SimplePreprocessor(32, 32)
# supplying our instantiated SimplePreprocessor as an argument
# implying that sp will be applied to every image in the dataset
sdl = SimpleDatasetLoader(preprocessors=[sp])
# load the dataset from disk
(data, labels) = sdl.load(imagePaths, verbose = 500)
# reshape the data matrix
data = data.reshape((data.shape[0], 3072))

print("[INFO] feautures matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))


# step 2

# encode the labels as integer
le = LabelEncoder()
labels = le.fit_transform(labels)
# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# step 3,4

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
# initialize the KNeighborsClassifier class
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
# trains the classifier
model.fit(trainX, trainY)
# evaluate our classifier
print(classification_report(testY, model.predict(testX), target_names=le.classes_))