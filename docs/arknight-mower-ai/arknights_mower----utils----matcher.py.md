# `arknights-mower\arknights_mower\utils\matcher.py`

```
# 从 __future__ 模块中导入 annotations 特性
from __future__ import annotations

# 导入所需的模块和库
import pickle
import traceback
from typing import Optional, Tuple
import cv2
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

# 导入自定义模块
from .. import __rootdir__
from . import typealias as tp
from .image import cropimg
from .log import logger

# 设置匹配器调试模式
MATCHER_DEBUG = False
# 定义 FLANN 算法中的索引类型
FLANN_INDEX_KDTREE = 0
# 定义良好匹配距离的限制
GOOD_DISTANCE_LIMIT = 0.7
# 创建 SIFT 特征提取器对象
SIFT = cv2.SIFT_create()

# 从文件中加载 SVM 模型
with open(f'{__rootdir__}/models/svm.model', 'rb') as f:
    SVC = pickle.loads(f.read())

# 定义函数，计算图像哈希值
def getHash(data: list[float]) -> tp.Hash:
    """ calc image hash """
    # 计算数据的平均值
    avreage = np.mean(data)
    # 根据数据的平均值生成哈希值
    return np.where(data > avreage, 1, 0)

# 定义函数，计算两个哈希值之间的汉明距离
def hammingDistance(hash1: tp.Hash, hash2: tp.Hash) -> int:
    """ calc Hamming distance between two hash """
    return np.count_nonzero(hash1 != hash2)

# 定义函数，计算图像的哈希值并计算汉明距离
def aHash(img1: tp.GrayImage, img2: tp.GrayImage) -> int:
    """ calc image hash """
    # 将图像缩放为 8x8 大小并展平
    data1 = cv2.resize(img1, (8, 8)).flatten()
    data2 = cv2.resize(img2, (8, 8)).flatten()
    # 计算图像的哈希值
    hash1 = getHash(data1)
    hash2 = getHash(data2)
    # 计算汉明距离
    return hammingDistance(hash1, hash2)

# 定义类，实现图像匹配功能
class Matcher(object):
    """ image matching module """

    def __init__(self, origin: tp.GrayImage) -> None:
        # 初始化匹配器，记录原始图像的形状
        logger.debug(f'Matcher init: shape ({origin.shape})')
        self.origin = origin
        # 初始化 SIFT 特征提取器
        self.init_sift()

    def init_sift(self) -> None:
        """ get SIFT feature points """
        # 使用 SIFT 算法检测原始图像的特征点和描述符
        self.kp, self.des = SIFT.detectAndCompute(self.origin, None)
    # 定义一个方法，用于匹配图像
    def match(self, query: tp.GrayImage, draw: bool = False, scope: tp.Scope = None, judge: bool = True,prescore = 0.0) -> Optional(tp.Scope):
        """ check if the image can be matched """
        # 调用 score 方法，获取匹配得分
        rect_score = self.score(query, draw, scope)  # get matching score
        # 如果得分为空，表示匹配失败
        if rect_score is None:
            return None  # failed in matching
        else:
            rect, score = rect_score

        # 如果预设得分不为0且实际得分大于等于预设得分，则匹配成功
        if prescore != 0.0 and score[3] >= prescore:
            logger.debug(f'match success: {score}')
            return rect
        # 使用 SVC 判断得分是否在合法范围内
        if judge and not SVC.predict([score])[0]:  # numpy.bool_
            logger.debug(f'match fail: {score}')
            return None  # failed in matching
        else:
            # 如果预设得分大于0且实际得分小于预设得分，则匹配失败
            if prescore>0 and score[3]<prescore:
                logger.debug(f'score is not greater than {prescore}: {score}')
                return None
            logger.debug(f'match success: {score}')
            return rect  # success in matching
```