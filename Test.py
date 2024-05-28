# -*- coding:utf-8 -*-
# author: DuanshengLiu
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers, losses, models
from tensorflow import keras
import cv2
import pickle
from PIL import Image, ImageFont, ImageDraw
import argparse


def unet_predict(unet, img_src_path):
    img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
    # img_src=cv2.imread(img_src_path)
    if img_src.shape != (512, 512, 3):
        img_src = cv2.resize(img_src, dsize=(512, 512), interpolation=cv2.INTER_AREA)[:, :, :3]  # dsize=(宽度,高度),[:,:,:3]是防止图片为4通道图片，后续无法reshape
    img_src = img_src.reshape(1, 512, 512, 3)  # 预测图片shape为(1,512,512,3)

    img_mask = unet.predict(img_src)  # 归一化除以255后进行预测
    img_src = img_src.reshape(512, 512, 3)  # 将原图reshape为3维
    img_mask = img_mask.reshape(512, 512, 3)  # 将预测后图片reshape为3维
    img_mask = img_mask / np.max(img_mask) * 255  # 归一化后乘以255
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # 三个通道保持相同
    img_mask = img_mask.astype(np.uint8)  # 将img_mask类型转为int型

    return img_src, img_mask



def locate_and_correct(img_src, img_mask):
    """
    该函数通过cv2对img_mask进行边缘检测，获取车牌区域的边缘坐标(存储在contours中)和最小外接矩形4个端点坐标,
    再从车牌的边缘坐标中计算出和最小外接矩形4个端点最近的点即为平行四边形车牌的四个端点,从而实现车牌的定位和矫正
    :param img_src: 原始图片
    :param img_mask: 通过u_net进行图像分隔得到的二值化图片，车牌区域呈现白色，背景区域为黑色
    :return: 定位且矫正后的车牌
    """
    # cv2.imshow('img_mask',img_mask)
    # cv2.waitKey(0)
    # ret,thresh = cv2.threshold(img_mask[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #二值化
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)
    try:
        contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:  # 防止opencv版本不一致报错
        ret, contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):  # contours1长度为0说明未检测到车牌
        # print("未检测到车牌")
        return [], []
    else:
        Lic_img = []
        img_src_copy = img_src.copy()  # img_src_copy用于绘制出定位的车牌轮廓
        for ii, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)  # 获取最小外接矩形
            img_cut_mask = img_mask[y:y + h, x:x + w]  # 将标签车牌区域截取出来
            # cv2.imshow('img_cut_mask',img_cut_mask)
            # cv2.waitKey(0)
            # print('w,h,均值,宽高比',w,h,np.mean(img_cut_mask),w/h)
            # contours中除了车牌区域可能会有宽或高都是1或者2这样的小噪点，
            # 而待选车牌区域的均值应较高，且宽和高不会非常小，因此通过以下条件进行筛选
            if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:
                rect = cv2.minAreaRect(cont)  # 针对坐标点获取带方向角的最小外接矩形，中心点坐标，宽高，旋转角度
                box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标
                # cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 2)
                # cv2.drawContours(img_mask, [box], 0, (0, 255, 0), 2)
                # cv2.imshow('img_mask',img_mask)
                # cv2.waitKey(0)
                cont = cont.reshape(-1, 2).tolist()
                # 由于转换矩阵的两组坐标位置需要一一对应，因此需要将最小外接矩形的坐标进行排序，最终排序为[左上，左下，右上，右下]
                box = sorted(box, key=lambda xy: xy[0])  # 先按照左右进行排序，分为左侧的坐标和右侧的坐标
                box_left, box_right = box[:2], box[2:]  # 此时box的前2个是左侧的坐标，后2个是右侧的坐标
                box_left = sorted(box_left, key=lambda x: x[1])  # 再按照上下即y进行排序，此时box_left中为左上和左下两个端点坐标
                box_right = sorted(box_right, key=lambda x: x[1])  # 此时box_right中为右上和右下两个端点坐标
                box = np.array(box_left + box_right)  # [左上，左下，右上，右下]
                # print(box)
                x0, y0 = box[0][0], box[0][1]  # 这里的4个坐标即为最小外接矩形的四个坐标，接下来需获取平行(或不规则)四边形的坐标
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]

                def point_to_line_distance(X, Y):
                    if x2 - x0:
                        k_up = (y2 - y0) / (x2 - x0)  # 斜率不为无穷大
                        d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
                    else:  # 斜率无穷大
                        d_up = abs(X - x2)
                    if x1 - x3:
                        k_down = (y1 - y3) / (x1 - x3)  # 斜率不为无穷大
                        d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
                    else:  # 斜率无穷大
                        d_down = abs(X - x1)
                    return d_up, d_down

                d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
                for each in cont:  # 计算cont中的坐标与矩形四个坐标的距离以及到上下两条直线的距离，对距离和进行权重的添加，成功计算选出四边形的4个顶点坐标
                    x, y = each[0], each[1]
                    dis0 = (x - x0) ** 2 + (y - y0) ** 2
                    dis1 = (x - x1) ** 2 + (y - y1) ** 2
                    dis2 = (x - x2) ** 2 + (y - y2) ** 2
                    dis3 = (x - x3) ** 2 + (y - y3) ** 2
                    d_up, d_down = point_to_line_distance(x, y)
                    weight = 0.975
                    if weight * d_up + (1 - weight) * dis0 < d0:  # 小于则更新
                        d0 = weight * d_up + (1 - weight) * dis0
                        l0 = (x, y)
                    if weight * d_down + (1 - weight) * dis1 < d1:
                        d1 = weight * d_down + (1 - weight) * dis1
                        l1 = (x, y)
                    if weight * d_up + (1 - weight) * dis2 < d2:
                        d2 = weight * d_up + (1 - weight) * dis2
                        l2 = (x, y)
                    if weight * d_down + (1 - weight) * dis3 < d3:
                        d3 = weight * d_down + (1 - weight) * dis3
                        l3 = (x, y)

                # print([l0,l1,l2,l3])
                # for l in [l0, l1, l2, l3]:
                #     cv2.circle(img=img_mask, color=(0, 255, 255), center=tuple(l), thickness=2, radius=2)
                # cv2.imshow('img_mask',img_mask)
                # cv2.waitKey(0)
                p0 = np.float32([l0, l1, l2, l3])  # 左上角，左下角，右上角，右下角，p0和p1中的坐标顺序对应，以进行转换矩阵的形成
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # 我们所需的长方形
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
                lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # 进行车牌矫正
                # cv2.imshow('lic',lic)
                # cv2.waitKey(0)
                Lic_img.append(lic)
                #cv2.drawContours(img_src_copy, [np.array([l0, l1, l3, l2])], -1, (0, 255, 0), 2)  # 在img_src_copy上绘制出定位的车牌轮廓，(0, 255, 0)表示绘制线条为绿色
                cv2.circle(img_src_copy,l0,5,(0,0,255),-1)
                cv2.circle(img_src_copy,l1,5,(0,0,255),-1)
                cv2.circle(img_src_copy,l2,5,(0,0,255),-1)
                cv2.circle(img_src_copy,l3,5,(0,0,255),-1)
    return img_src_copy, Lic_img
def cnn_predict(cnn, Lic_img):
    characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    Lic_pred = []
    Char=[]
    for lic in Lic_img:
        lic_pred = cnn.predict(lic.reshape(1, 80, 240, 3))  # 预测形状应为(1,80,240,3)
        lic_pred = np.array(lic_pred).reshape(7, 65)  # 列表转为ndarray，形状为(7,65)
        if len(lic_pred[lic_pred >= 0.0]) >= 4:  # 统计其中预测概率值大于80%以上的个数，大于等于4个以上认为识别率高，识别成功
            chars = ''
            for arg in np.argmax(lic_pred, axis=1):  # 取每行中概率值最大的arg,将其转为字符
                chars += characters[arg]
            Char.append(chars)
            chars = chars[0:2] + '·' + chars[2:]
            Lic_pred.append((lic, chars))  # 将车牌和识别结果一并存入Lic_pred
            
    return Lic_pred,Char

class Draw:
    _font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/zh_cn_92.ttf"), 95)
    ]
    x=cv2.imread(os.path.join(os.path.dirname(__file__), "res/blue_bg.png"))
    _bg = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "res/blue_bg.png")), (440, 140))

    def __call__(self, plate):
        if len(plate) != 7:
            print("ERROR: Invalid length")
            return None
        fg = self._draw_fg(plate)
        return cv2.cvtColor(cv2.bitwise_or(fg, self._bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch):
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 95, 140), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill = (255, 255, 255),
            font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (440, 140), (0, 0, 0)))
        offset = 15
        img[0:140, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 12
        img[0:140, offset:offset+45] = self._draw_char(plate[1])
        offset = offset + 45 + 34
        for i in range(2, len(plate)):
            img[0:140, offset:offset+45] = self._draw_char(plate[i])
            offset = offset + 45 + 12
        return img

if __name__ == '__main__':

    unet=keras.models.load_model('unet_test.h5')
    cnn = keras.models.load_model('chepaishibie.h5')
    while 1:
        a = input("Enter the file name: ")
        
        #img=cv2.imread('./img/7.jpg')
        img_src, img_mask=unet_predict(unet, a)
        cv2.imshow('./test/1.jpg',img_src)
        cv2.waitKey()
        cv2.imshow('./test/2.jpg',img_mask)
        cv2.waitKey()
        img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
        cv2.imshow('./test/3.jpg',img_src_copy)
        cv2.waitKey()
        cv2.imshow('./test/4.jpg',Lic_img[0])
        cv2.waitKey()

        
        Lic_img=Lic_img[0].reshape(1, 80, 240, 3)
        cp1,cp2=cnn_predict(cnn, Lic_img)
        print(cnn_predict(cnn, Lic_img))

        parser = argparse.ArgumentParser(description="Generate a black plate.")
        parser.add_argument("plate", help="license plate number (default: 京A12345)", type=str, nargs="?", default=cp2[0])
        args = parser.parse_args()

        draw = Draw()
        plate = draw(args.plate)
        b,g,r = cv2.split(plate)
        plate = cv2.merge((r,g,b))
        cv2.imshow("./test/5.jpg",plate)
        cv2.waitKey()
