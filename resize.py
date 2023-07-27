import cv2
img = cv2.imread('image/kinan.jpg',0)
img = cv2.resize(img, (1400,1600))
cv2.imwrite('image/kinan-small.jpg',img)   