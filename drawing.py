import cv2

def on_trackbar(x):
    pass

cv2.namedWindow('Canny')

cv2.createTrackbar('low threshold', 'Canny', 0, 1000, on_trackbar)
cv2.createTrackbar('high threshold', 'Canny', 0, 1000, on_trackbar)

cv2.setTrackbarPos('low threshold', 'Canny', 50)
cv2.setTrackbarPos('high threshold', 'Canny', 150)

img_gray = cv2.imread('yeon000.jpg', cv2.IMREAD_GRAYSCALE)

while(1):
    low = cv2.getTrackbarPos('low threshold', 'Canny')
    high = cv2.getTrackbarPos('high threshold', 'Canny')

    img_canny = cv2.Canny(img_gray, low, high)

    cv2.imshow('Canny', img_canny)

    if cv2.waitKey(1) & 0xFF == 13:
        cv2.imwrite("result.jpg", img_canny)
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

img_result = cv2.imread('result.jpg')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_result = cv2.dilate(img_result, kernel, iterations = 1)

img_r = 255 - img_canny
cv2.imwrite("r.jpg", img_r)
cv2.imshow('result', img_result)
cv2.waitKey()
