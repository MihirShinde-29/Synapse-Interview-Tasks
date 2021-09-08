import cv2
import numpy as np

shapes = cv2.imread('shapes.jpg')

while True:
    img2 = shapes.copy()
    img = cv2.GaussianBlur(shapes, (7, 7), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 23, 22)
    kernel = np.ones((5, 5))
    img = cv2.dilate(img, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxarea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxarea:
            maxarea = area
            maincnt = cnt
    cv2.drawContours(img2, maincnt, -1, (255, 0, 255), 7)
    cv2.imshow('frame', img2)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()