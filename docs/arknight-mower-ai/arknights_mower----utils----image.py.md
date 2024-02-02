# `arknights-mower\arknights_mower\utils\image.py`

```py
# 导入必要的模块
from typing import Union
import cv2
import numpy as np
from . import typealias as tp
from .log import logger, save_screenshot

# 将字节数据转换为图像数据
def bytes2img(data: bytes, gray: bool = False) -> Union[tp.Image, tp.GrayImage]:
    """ bytes -> image """
    if gray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.cvtColor(
            cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )

# 将图像数据转换为字节数据
def img2bytes(img) -> bytes:
    """ bytes -> image """
    return cv2.imencode('.png', img)[1]

# 从文件加载图像数据
def loadimg(filename: str, gray: bool = False) -> Union[tp.Image, tp.GrayImage]:
    """ load image from file """
    logger.debug(filename)
    if gray:
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

# 对图像进行二值化处理
def thres2(img: tp.GrayImage, thresh: int) -> tp.GrayImage:
    """ binarization of images """
    _, ret = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return ret

# 将彩色图像转换为灰度图像
def rgb2gray(img: tp.Image) -> tp.GrayImage:
    """ change image from rgb to gray """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 将区域范围转换为切片
def scope2slice(scope: tp.Scope) -> tp.Slice:
    # 将输入的坐标范围转换为另一种形式的坐标范围
    if scope is None:
        # 如果输入为空，则返回一个切片对象，表示全部范围
        return slice(None), slice(None)
    # 返回一个切片对象，表示输入坐标范围的转换结果
    return slice(scope[0][1], scope[1][1]), slice(scope[0][0], scope[1][0])
# 定义一个函数，用于裁剪图片
def cropimg(img: tp.Image, scope: tp.Scope) -> tp.Image:
    """ crop image """
    # 返回裁剪后的图片
    return img[scope2slice(scope)]

# 定义一个函数，用于保存图片
def saveimg(img, folder='failure'):
    # 保存转换为字节流的图片截图
    save_screenshot(
        img2bytes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        # 保存到指定文件夹和图片尺寸的子文件夹中
        subdir=f'{folder}/{img.shape[0]}x{img.shape[1]}',
    )
```