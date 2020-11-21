import cv2
image = cv2.imread("D:\pictures\gushi\Telegram\bitmoji.png")
print(image.shape)
cv2.imshow("Image",image)
cv2.waitKey(0)