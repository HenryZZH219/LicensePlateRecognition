#from tensorflow.keras import layers, losses, models
#from tensorflow import keras
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import math


def randomRotation(img):
    imgInfo = img.shape
    height= imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]
    angle=np.random.randint(-3,3)
    scale=np.random.randint(97,103)/100
    m1=np.random.randint(48,52)/100
    m2=np.random.randint(48,52)/100
# 定义一个旋转矩阵
    matRotate = cv2.getRotationMatrix2D((height*m1, width*m2), angle, scale) # mat rotate 1 center 2 angle 3 缩放系数

    dst = cv2.warpAffine(img, matRotate, (width,height))
    return dst



def randomShift(image):
    #def randomShift(image, xoffset, yoffset=None):
        """
        对图像进行平移操作
        :param image: PIL的图像image
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        :return: 翻转之后的图像
        """
        imgInfo = image.shape
        height= imgInfo[0]
        width = imgInfo[1]
        random_xoffset = np.random.randint(0, height*0.05)
        random_yoffset = np.random.randint(0, width*0.05)
        return image.offset(xoffset = random_xoffset, yoffset = random_yoffset)
        #return image.offset(random_xoffset)
def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    return image


    # 读取数据集
path = './p/'  # 车牌号数据集路径(车牌图片宽240，高80)
pic_name = sorted(os.listdir(path))
n = len(pic_name)

for i in range(n):
    print("正在读取第%d张图片" % i)
    img = cv2.imdecode(np.fromfile(path + pic_name[i], dtype=np.uint8), -1)  # cv2.imshow无法读取中文路径图片，改用此方式
    img = cv2.resize(img,(240, 80))
    ###############
    img = randomRotation(img)
    #img = randomShift(img)
    r=np.random.randint(4,8)
    img = cv2.blur(img,(r,r))
    img = gaussian_noise(img)
    #cv2.imshow("1",img)
    #cv2.waitKey()
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,imGray = cv2.threshold(imGray,127,255,cv2.THRESH_BINARY)
    #cv2.imshow("1",imGray)
    #cv2.waitKey()
    cv2.imencode('.jpg', imGray)[1].tofile("./tt./"+pic_name[i]) 