import cv2
import numpy as np

img = cv2.imread("green.bmp")

frame = img

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define lower and uppper limits of what we call "brown"
img_lo=np.array([0, 255, 0])
img_hi=np.array([0,255,0])

# Mask image to only select browns
mask=cv2.inRange(rgb,img_lo,img_hi)

# Change image to red where we found brown
# frame[mask>0]=(237,28,36) #001 빨
frame[mask>0]=(36,28,237) #001 빨
cv2.imwrite("001.jpg", frame)
# frame[mask>0]=(255,127,39) #002 주
frame[mask>0]=(39,127,255) #002 주
cv2.imwrite("002.jpg", frame)
# frame[mask>0]=(255,242,0) #003 노
frame[mask>0]=(0,242,255) #003 노
cv2.imwrite("003.jpg", frame)
# frame[mask>0]=(34,177, 76) #004 초
frame[mask>0]=(76,177, 34) #004 초
cv2.imwrite("004.jpg", frame)
# frame[mask>0]=(0, 162, 232) #005 파
frame[mask>0]=(232, 162, 0) #005 파
cv2.imwrite("005.jpg", frame)
# frame[mask>0]=(63, 72, 204) #006 남
frame[mask>0]=(204, 72, 63) #006 남
cv2.imwrite("006.jpg", frame)
# frame[mask>0]=(163, 73, 164) #007 보
frame[mask>0]=(164, 73, 163) #007 보
cv2.imwrite("007.jpg", frame)
# frame[mask>0]=(136, 0, 21) #008 갈
frame[mask>0]=(21, 0, 136) #008 갈
cv2.imwrite("008.jpg", frame)
# frame[mask>0]=(185, 122, 87) #009 연갈
frame[mask>0]=(87, 122, 185) #009 연갈
cv2.imwrite("009.jpg", frame)
# frame[mask>0]=(255, 174, 201) #010 분홍
frame[mask>0]=(201, 174, 255) #010 분홍
cv2.imwrite("010.jpg", frame)
# frame[mask>0]=(255, 201, 14) #011 감귤
frame[mask>0]=(14, 201, 255) #011 감귤
cv2.imwrite("011.jpg", frame)
# frame[mask>0]=(239, 228, 176) #012 베이지
frame[mask>0]=(176, 228, 239) #012 베이지
cv2.imwrite("012.jpg", frame)
# frame[mask>0]=(181, 230, 29) #013 라임
frame[mask>0]=(29, 230, 181) #013 라임
cv2.imwrite("013.jpg", frame)
# frame[mask>0]=(153, 217, 234) #014 하늘
frame[mask>0]=(234, 217, 153) #014 하늘
cv2.imwrite("014.jpg", frame)
# frame[mask>0]=(112, 146, 190) #015 연남
frame[mask>0]=(190, 146, 112) #015 연남
cv2.imwrite("015.jpg", frame)
# frame[mask>0]=(200, 191, 231) #016 연보
frame[mask>0]=(231, 191, 200) #016 연보
cv2.imwrite("016.jpg", frame)
# frame[mask>0]=(255, 255, 255) #000 오리진
frame[mask>0]=(255, 255, 255) #000 오리진
cv2.imwrite("000.jpg", frame)

