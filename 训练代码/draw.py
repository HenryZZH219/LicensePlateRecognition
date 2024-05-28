import numpy as np
import os
import cv2
import pickle
img_src=cv2.imread('./test/1.jpg')
bgr_img=cv2.imread('./test/2.jpg')

gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
th, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(bgr_img, contours, -1, (0, 0, 255), 3)
 
bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
 
for bbox in bounding_boxes:
     [x , y, w, h] = bbox
     cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

ww=int(w/2)
hh=int(h/2)
for i in range(x,x+ww,1):
    if(binary[y,i]==255):
        cv2.circle(bgr_img,(i,y),5,(0,0,255),-1)
        l0=(i,y)
        break 
for i in range(x,x+ww,1):
    if(binary[y+h-1,i]==255):
        cv2.circle(bgr_img,(i,y+h),5,(0,0,255),-1)
        l1=(i,y+h)
        break 
for i in range(y,y+hh,1):
    if(binary[i,x]==255):
        cv2.circle(bgr_img,(x,i),5,(0,0,255),-1)
        l0=(x,i)
        break 
for i in range(y,y+hh,1):
    if(binary[i,x+w-1]==255):
        cv2.circle(bgr_img,(x+w,i),5,(0,0,255),-1)
        l2=(x+w,i)
        break 


for i in range(x+w,x+ww,-1):
    if(binary[y,i]==255):
        cv2.circle(bgr_img,(i,y),5,(0,0,255),-1)
        l2=(i,y)
        break 
for i in range(x+w,x+ww,-1):
    if(binary[y+h-1,i]==255):
        cv2.circle(bgr_img,(i,y+h),5,(0,0,255),-1)
        l3=(i,y+h)
        break 
for i in range(y+h,y+hh,-1):
    if(binary[i,x]==255):
        cv2.circle(bgr_img,(x,i),5,(0,0,255),-1)
        l1=(x,i)
        break 
for i in range(y+h,y+hh,-1):
    if(binary[i,x+w-1]==255):
        cv2.circle(bgr_img,(x+w,i),5,(0,0,255),-1)
        l3=(x+w,i)
        break

p0 = np.float32([l0, l1, l2, l3])  # 左上角，左下角，右上角，右下角，p0和p1中的坐标顺序对应，以进行转换矩阵的形成
p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # 我们所需的长方形
transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # 进行车牌矫正 
imGray = cv2.cvtColor(lic, cv2.COLOR_BGR2GRAY)
ret,imGray = cv2.threshold(imGray,127,255,cv2.THRESH_BINARY)
cv2.imshow("name", imGray)
cv2.waitKey(0)
