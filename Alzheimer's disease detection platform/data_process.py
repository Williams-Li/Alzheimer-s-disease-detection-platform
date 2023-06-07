import cv2
import glob
import numpy as np
import pandas as pd


def rotate(angle=45, scale=0.7, hight=None, width=None):
    '''
    # 图片旋转函数,angle：,scale：
    :param angle: 旋转角度
    :param scale: 图像比例
    :param hight: 原图高度
    :param width: 圆度宽度
    :return: 旋转后的图像
    '''
    rotation = cv2.getRotationMatrix2D(center=(hight*0.5, width*0.5), angle=angle, scale=scale)    # 仿射矩阵
    return rotation


def get_data(path='data/train/*/*.png'):
    file_names = glob.glob(path)                      # 获取所有照片路径
    X = []       # 用于存储样本自变量
    y = []       # 用于存储样本标签
    for i in file_names:
        image = cv2.imread(i)                          # 读取数据
        image = cv2.resize(image, (227, 227))          # 将数据缩放至指定大小
        hight, width = image.shape[0], image.shape[1]  # 图像尺寸

        image_flip = cv2.flip(image, flipCode=-1)      # 图像增强一：图像翻转
        rotate45 = cv2.warpAffine(image, rotate(45, 0.7, hight, width), dsize=(hight, width))     # 图像增强二：图像旋转与缩放
        rotate90 = cv2.warpAffine(image, rotate(90, 1.2, hight, width), dsize=(hight, width))     # 图像增强三：图像旋转与缩放

        X.extend([                       # 保存中间数据（样本自变量）
            image[:, :, 0:1],            # 原图
            image_flip[:, :, 0:1],       # 水平垂直翻转图
            rotate45[:, :, 0:1],         # 旋转45度图
            rotate90[:, :, 0:1]          # 旋转90度图
        ])
        y.extend([i.split('\\')[1]] * 4)     # 保存样本标签

    X = np.array(X, dtype=np.float64)                  # 将数据转换为数组形式
    y = pd.Series(y).map({'AD': 0, 'CN': 1}).values    # 用数字表示样本标签
    return X, y


def get_data_test(path='data/test/*/*.png'):
    file_names = glob.glob(path)                       # 获取所有照片路径
    X = []       # 用于存储样本自变量
    for i in file_names:
        image = cv2.imread(i)                          # 读取数据
        image = cv2.resize(image, (227, 227))          # 将数据缩放至指定大小

        X.extend([                       # 保存中间数据（样本自变量）
            image[:, :, 0:1]             # 原图
        ])

    X = np.array(X, dtype=np.float64)                  # 将数据转换为数组形式
    return X
