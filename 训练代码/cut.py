import cv2

img=cv2.imread("./img./5.jpg")
img=img[108:175,83:209]

gray=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
pic=cv2.Canny(img,200,300)
cv2.imshow('1',pic)
cv2.waitKey()