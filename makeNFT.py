import cv2
from matplotlib import pyplot as plt
import skimage
from sklearn.cluster import KMeans
from numpy import linalg as LA
import numpy as np
import copy

def printI(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)

def printI2(i1, i2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(cv2.cvtColor(i1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(cv2.cvtColor(i2, cv2.COLOR_BGR2RGB))

def pixelate(img, w, h):
    height, width = img.shape[:2]

    # Resize input to "pixelated" size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]

    return imgC


def segmentImgClrRGB(img, k):
    imgC = np.copy(img)

    h = img.shape[0]
    w = img.shape[1]

    imgC.shape = (img.shape[0] * img.shape[1], 3)

    # 5. Run k-means on the vectorized responses X to get a vector of labels (the clusters);
    #
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_

    # 6. Reshape the label results of k-means so that it has the same size as the input image
    #   Return the label image which we call idx
    kmeans.shape = (h, w)

    return kmeans

def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)

img_origin = cv2.imread('12800.png')

printI(img_origin)

# img64 = pixelate(img_origin, 64, 64)
img128 = pixelate(img_origin,64,64)

# printI2(img_origin, img64)

# img_result = kMeansImage(img_origin, 5)
# img_result_64 = kMeansImage(img64, 5)
img_result_128 = kMeansImage(img128, 5)
# cv2.imshow('origin', img_origin)
# cv2.imshow('img64', img64)
# cv2.imshow('img128', img128)
# cv2.imshow('img_kmeans', img_result)
# cv2.imshow('img64_kmeans', img_result_64)
# cv2.imshow('img128_kmeans', img_result_128)
cv2.imwrite('128.jpg', img_result_128)
# cv2.imwrite('64.jpg', img_result_64)

######################################

'''
img = cv2.imread('128.jpg', cv2.IMREAD_COLOR)
b,g,r = cv2.split(img) # channel split
bbg_img = cv2.merge((b,b,g))
rbr_img = cv2.merge((r,b,r))
bbb_img = cv2.merge((b,b,b))
rbb_img = cv2.merge((r,b,b))
rrr_img = cv2.merge((r,r,r))
# ggg_img = cv2.merge((g,g,g))

cv2.imshow('original',img)
cv2.imshow('gbr',bbg_img)
cv2.imshow('rbr',rbr_img)
cv2.imshow('bbb',bbb_img)
cv2.imshow('rbb',rbb_img)
cv2.imshow('rrr',rrr_img)
# cv2.imshow('ggg',ggg_img)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('original.jpg',img)
cv2.imwrite('gbr.jpg',bbg_img)
cv2.imwrite('rbr.jpg',rbr_img)
cv2.imwrite('bbb.jpg',bbb_img)
cv2.imwrite('rbb.jpg',rbb_img)
cv2.imwrite('rrr.jpg',rrr_img)

img_red = copy.deepcopy(img)
img_green = copy.deepcopy(img)
img_blue = copy.deepcopy(img)
img_magenta = copy.deepcopy(img)
img_yellow = copy.deepcopy(img)
img_cyan = copy.deepcopy(img)

img_red[:,:,(0,1)] = 0
img_green[:,:,(0,2)] = 0
img_blue[:,:,(1,2)] = 0
img_magenta[:,:,1] = 0
img_yellow[:,:,0] = 0
img_cyan[:,:,2] = 0

cv2.imwrite('red.jpg',img_red)
cv2.imwrite('green.jpg',img_green)
cv2.imwrite('blue.jpg',img_blue)
cv2.imwrite('magenta.jpg',img_magenta)
cv2.imwrite('cyan.jpg',img_cyan)
cv2.imwrite('yellow.jpg',img_yellow)
'''