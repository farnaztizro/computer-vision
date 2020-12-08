import numpy as np
import cv2

# initializes the list of target class labels for the “Animals” dataset.
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# initialize our weight matrix and bias vector
W = np.random.randn(3, 3072)
b = np.random.randn(3)

#  load our example image from disk
orig = cv2.imread("D:\computer_vision\cat.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

# compute the output class label scores by applying our scoring function
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 

# display our output image
cv2.imshow("image", orig)
cv2.waitKey(0)
