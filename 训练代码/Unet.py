# -*- coding:utf-8 -*-
# author: DuanshengLiu
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers, losses, models
from tensorflow import keras
import cv2
import pickle

def unet_train():
    height = 512
    width = 512
    # path = 'D:/desktop/unet_datasets/'
    # input_name = os.listdir(path + 'train_image')
    # n = len(input_name)
    # print(n)
    X_train, y_train = [], []
    # for i in range(n):
    #     print("正在读取第%d张图片" % i)
    #     img = cv2.imread(path + 'train_image/%d.png' % i)
    #     label = cv2.imread(path + 'train_label/%d.png' % i)
    #     X_train.append(img)
    #     y_train.append(label)

    i=0
    cur_dir=os.getcwd()
    labels_dir=os.path.join(cur_dir,'VOCdevkit/VOC2007/JPEGImages')
    for parent, dirnames, filenames in os.walk(labels_dir): # 分别得到根目录，子目录和根目录下文件   
        for file_name in filenames:
            if file_name.endswith("jpg"):
                img=cv2.imread('./VOCdevkit/VOC2007/JPEGImages./'+file_name)
                img=cv2.resize(img,(height,width))
                #cv2.imshow('1',img)
                #cv2.waitKey()
                X_train.append(img)
                label=cv2.imread('./VOCdevkit/VOC2007/labels./'+file_name)
                label=cv2.resize(label,(height,width))
                ret,label=cv2.threshold(label,127,255,cv2.THRESH_BINARY)
                y_train.append(label)
                #cv2.imshow('2',label)
                #cv2.waitKey()
                i=i+1
    print(i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    inpt = layers.Input(shape=(height, width, 3))
    conv1 = Conv2d_BN(inpt, 8, (3, 3))
    conv1 = Conv2d_BN(conv1, 8, (3, 3))
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    conv2 = Conv2d_BN(conv2, 16, (3, 3))
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    conv3 = Conv2d_BN(conv3, 32, (3, 3))
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    concat1 = layers.concatenate([conv4, convt1], axis=3)
    concat1 = layers.Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 64, (3, 3))
    conv6 = Conv2d_BN(conv6, 64, (3, 3))

    convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    concat2 = layers.concatenate([conv3, convt2], axis=3)
    concat2 = layers.Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 32, (3, 3))
    conv7 = Conv2d_BN(conv7, 32, (3, 3))

    convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    concat3 = layers.concatenate([conv2, convt3], axis=3)
    concat3 = layers.Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16, (3, 3))
    conv8 = Conv2d_BN(conv8, 16, (3, 3))

    convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    concat4 = layers.concatenate([conv1, convt4], axis=3)
    concat4 = layers.Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8, (3, 3))
    conv9 = Conv2d_BN(conv9, 8, (3, 3))
    conv9 = layers.Dropout(0.5)(conv9)
    outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

    model = models.Model(inpt, outpt)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    print("开始训练u-net")
    history=model.fit(X_train, y_train, epochs=100, batch_size=5,shuffle=True,validation_split=0.05)
    #model.fit(X_train, y_train, epochs=100, batch_size=15)#epochs和batch_size看个人情况调整，batch_size不要过大，否则内存容易溢出
    #我11G显存也只能设置15-20左右，我训练最终loss降低至250左右，acc约95%左右
    model.save('unet_test.h5')
    print('unet.h5保存成功!!!')
    with open('history.txt','wb') as file_pi:
        pickle.dump(history.history,file_pi)


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
                cv2.drawContours(img_src_copy, [np.array([l0, l1, l3, l2])], -1, (0, 0, 255), 2)  # 在img_src_copy上绘制出定位的车牌轮廓，(0, 255, 0)表示绘制线条为绿色
    return img_src_copy, Lic_img


if __name__ == '__main__':
    #unet_train()
    unet=keras.models.load_model('unet_test.h5')
    img=cv2.imread('./img/7.jpg')
    img_src, img_mask=unet_predict(unet, './img/5.jpg')
    cv2.imwrite('./test/1.jpg',img_src)
    cv2.imwrite('./test/2.jpg',img_mask)
    img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
    cv2.imwrite('./test/3.jpg',img_src_copy)
    cv2.imwrite('./test/4.jpg',Lic_img[0])

