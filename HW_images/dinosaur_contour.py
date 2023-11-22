import cv2

image = cv2.imread('images/8.jpg', 0)
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Original image', image)
cv2.imshow('Thresh', thresh)

for contour in contours:
    print(contour)
    area = cv2.contourArea(contour)
    cv2.drawContours(image, contour, -1, (0, 255, 0), 3)

cv2.imshow('Image with Contours', image)

cv2.waitKey()
cv2.destroyAllWindows()
