# `arknights-mower\arknights_mower\utils\character_recognize.py`

```
# 导入必要的模块和库
from __future__ import annotations
import traceback
from copy import deepcopy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from .. import __rootdir__
from ..data import agent_list,ocr_error
from . import segment
from .image import saveimg
from .log import logger
from .recognize import RecognizeError
from ..ocr import ocrhandle

# 计算多边形的中心点
def poly_center(poly):
    return (np.average([x[0] for x in poly]), np.average([x[1] for x in poly]))

# 判断点是否在多边形内部
def in_poly(poly, p):
    return poly[0, 0] <= p[0] <= poly[2, 0] and poly[0, 1] <= p[1] <= poly[2, 1]

# 初始化字符映射字典
char_map = {}
# 深拷贝干员列表并按长度排序
agent_sorted = sorted(deepcopy(agent_list), key=len)
# 初始化原始图像、关键点和描述子
origin = origin_kp = origin_des = None
# 定义 FLANN 索引类型和距离限制
FLANN_INDEX_KDTREE = 0
GOOD_DISTANCE_LIMIT = 0.7
# 创建 SIFT 对象
SIFT = cv2.SIFT_create()

# 初始化 SIFT 特征点识别
def agent_sift_init():
    global origin, origin_kp, origin_des
    if origin is None:
        logger.debug('agent_sift_init')
        # 设置图像大小和网格数量
        height = width = 2000
        lnum = 25
        cell = height // lnum
        # 创建空白图像
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        # 加载字体和字符集
        font = ImageFont.truetype(
            f'{__rootdir__}/fonts/SourceHanSansSC-Bold.otf', size=30, encoding='utf-8'
        )
        chars = sorted(list(set(''.join([x for x in agent_list])))
        # 确保字符数量不超过网格容量
        assert len(chars) <= (lnum - 2) * (lnum - 2)
        # 在图像上绘制字符并建立字符映射
        for idx, char in enumerate(chars):
            x, y = idx % (lnum - 2) + 1, idx // (lnum - 2) + 1
            char_map[(x, y)] = char
            ImageDraw.Draw(img).text(
                (x * cell, y * cell), char, (255, 255, 255), font=font
            )
        # 转换图像为灰度图
        origin = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        # 提取图像的关键点和描述子
        origin_kp, origin_des = SIFT.detectAndCompute(origin, None)

# 使用 SIFT 特征点识别干员名称
def sift_recog(query, resolution, draw=False,bigfont = False):
    agent_sift_init()
    # 大号字体修改参数
    # 如果使用大字体，则创建 SIFT 对象并设置参数
    if bigfont:
        SIFT = cv2.SIFT_create(
            contrastThreshold=0.1,
            edgeThreshold=20
        )
    # 否则，创建默认的 SIFT 对象
    else:
        SIFT = cv2.SIFT_create()
    # 将查询图像转换为灰度图像
    query = cv2.cvtColor(np.array(query), cv2.COLOR_RGB2GRAY)
    # 获取查询图像的高度和宽度
    height, width = query.shape

    # 根据分辨率调整查询图像的大小
    multi = 2 * (resolution / 1080)
    query = cv2.resize(query, (int(width * multi), int(height * multi)))
    # 使用 SIFT 对象检测并计算查询图像的关键点和描述符
    query_kp, query_des = SIFT.detectAndCompute(query, None)

    # 构建基于 Flann 的匹配器
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 使用 Flann 匹配器进行特征点匹配
    matches = flann.knnMatch(query_des, origin_des, k=2)

    # 根据 Lowe's ratio test 存储所有良好的匹配
    good = []
    for x, y in matches:
        if x.distance < GOOD_DISTANCE_LIMIT * y.distance:
            good.append(x)

    # 如果需要绘制匹配结果，则绘制并显示
    if draw:
        result = cv2.drawMatches(query, query_kp, origin, origin_kp, good, None)
        plt.imshow(result, 'gray')
        plt.show()

    # 初始化一个空字典用于存储字符计数
    count = {}

    # 遍历良好的匹配点，根据特征点位置映射到字符，并统计出现次数
    for x in good:
        x, y = origin_kp[x.trainIdx].pt
        c = char_map[(int(x) // 80, int(y) // 80)]
        count[c] = count.get(c, 0) + 1

    # 初始化最佳匹配结果和最高得分
    best = None
    best_score = 0
    # 遍历按代理排序的字符集
    for x in agent_sorted:
        score = 0
        # 计算每个字符集中字符在查询图像中出现的次数总和
        for c in set(x):
            score += count.get(c, -1)
        # 更新最佳匹配结果和最高得分
        if score > best_score:
            best = x
            best_score = score

    # 记录调试信息，包括字符计数和最佳匹配结果
    logger.debug(f'segment.sift_recog: {count}, {best}')

    # 返回最佳匹配结果
    return best
def agent(img, draw=False):
    """
    识别干员总览界面的干员名称
    """
    # 捕获异常并记录日志
    except Exception as e:
        logger.debug(traceback.format_exc())
        # 保存图片并抛出识别错误
        saveimg(img, 'failure_agent')
        raise RecognizeError(e)

def agent_name(__img, height, draw: bool = False):
    # 将图像转换为灰度图
    query = cv2.cvtColor(np.array(__img), cv2.COLOR_RGB2GRAY)
    # 获取图像的高度和宽度
    h, w= query.shape
    # 设置图像的尺寸为原来的4倍
    dim = (w*4, h*4)
    # 调整图像大小
    resized = cv2.resize(__img, dim, interpolation=cv2.INTER_AREA)
    # 使用 OCR 模型预测图像中的文本
    ocr = ocrhandle.predict(resized)
    name = ''
    try:
        # 如果 OCR 结果不为空并且在干员列表中，且不是特定的干员，则将其作为干员名称
        if len(ocr) > 0 and ocr[0][1] in agent_list and ocr[0][1] not in ['砾', '陈']:
            name = ocr[0][1]
        # 如果 OCR 结果不为空并且在 OCR 错误字典中，则使用对应的正确名称
        elif len(ocr) > 0 and ocr[0][1] in ocr_error.keys():
            name = ocr_error[ocr[0][1]]
        else:
            # 使用 SIFT 算法识别干员名称
            res = sift_recog(__img, height, draw,bigfont=True)
            # 如果识别结果不为空并且在干员列表中，则将其作为干员名称
            if (res is not None) and res in agent_list:
                name = res
            else:
                # 抛出识别错误异常
                raise Exception(f"识别错误: {res}")
    except Exception as e:
        # 如果发生异常，记录警告日志并保存图片
        if len(ocr)>0:
            logger.warning(e)
            logger.warning(ocr[0][1])
            saveimg(__img, 'failure_agent')
    # 返回干员名称
    return name
```