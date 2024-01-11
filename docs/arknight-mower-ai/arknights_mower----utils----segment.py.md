# `arknights-mower\arknights_mower\utils\segment.py`

```
# 导入必要的模块和类
from __future__ import annotations
import traceback
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ..data import agent_list
from ..ocr import ocrhandle
from . import detector
from . import typealias as tp
from .log import logger
from .recognize import RecognizeError

# 自定义异常类，用于表示洪水检查失败
class FloodCheckFailed(Exception):
    pass

# 定义函数，根据给定的坐标返回一个矩形的四个顶点坐标
def get_poly(x1: int, x2: int, y1: int, y2: int) -> tp.Rectangle:
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    return np.array([ [ x1, y1 ], [ x1, y2 ], [ x2, y2 ], [ x2, y1 ]])

# 定义信用交易所特供的图像分割算法函数
def credit(img: tp.Image, draw: bool = False) -> list[ tp.Scope ]:
    """
    信用交易所特供的图像分割算法
    """
    # 捕获异常并记录日志，然后抛出自定义的识别错误
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)

# 定义公招特供的图像分割算法函数
def recruit(img: tp.Image, draw: bool = False) -> list[ tp.Scope ]:
    """
    公招特供的图像分割算法
    """
    # 捕获异常并记录日志，然后抛出自定义的识别错误
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)

# 定义基建布局的图像分割算法函数
def base(img: tp.Image, central: tp.Scope, draw: bool = False) -> dict[ str, tp.Rectangle ]:
    """
    基建布局的图像分割算法
    """
    # 捕获异常并记录日志，然后抛出自定义的识别错误
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)

# 定义进驻总览的图像分割算法函数
def worker(img: tp.Image, draw: bool = False) -> tuple[ list[ tp.Rectangle ], tp.Rectangle, bool ]:
    """
    进驻总览的图像分割算法
    """
    # 尝试执行以下代码块，捕获可能发生的异常
    try:
        # 获取图像的高度、宽度和通道数
        height, width, _ = img.shape

        # 初始化左右边界
        left, right = 0, width
        # 寻找右边界，直到像素值小于100
        while np.max(img[ :, right - 1 ]) < 100:
            right -= 1
        # 寻找左边界，直到像素值小于100
        while np.max(img[ :, left ]) < 100:
            left += 1

        # 初始化 x0 坐标
        x0 = right - 1
        # 寻找 x0 坐标，直到绿色通道的平均像素值小于100
        while np.average(img[ :, x0, 1 ]) >= 100:
            x0 -= 1
        x0 -= 2

        # 初始化分割线列表和移除模式标志
        seg = [ ]
        remove_mode = False
        pre, st = int(img[ 0, x0, 1 ]), 0
        # 遍历图像的每一行像素
        for y in range(1, height):
            # 更新移除模式标志
            remove_mode |= int(img[ y, x0, 0 ]) - int(img[ y, x0, 1 ]) > 40
            # 判断是否需要进行分割
            if np.ptp(img[ y, x0 ]) <= 1 or int(img[ y, x0, 0 ]) - int(img[ y, x0, 1 ]) > 40:
                now = int(img[ y, x0, 1 ])
                # 判断是否需要进行分割
                if abs(now - pre) > 20:
                    if now < pre and st == 0:
                        st = y
                    elif now > pre and st != 0:
                        seg.append((st, y))
                        st = 0
                pre = now
            elif st != 0:
                seg.append((st, y))
                st = 0
        # 输出分割线列表
        logger.debug(f'segment.worker: seg {seg}')

        # 获取移除按钮的多边形
        remove_button = get_poly(x0 - 10, x0, seg[ 0 ][ 0 ], seg[ 0 ][ 1 ])
        # 移除第一个分割线
        seg = seg[ 1: ]

        # 寻找第二个分割线的 x1 坐标
        for i in range(1, len(seg)):
            if seg[ i ][ 1 ] - seg[ i ][ 0 ] > 9:
                x1 = x0
                while img[ seg[ i ][ 1 ] - 3, x1 - 1, 2 ] < 100:
                    x1 -= 1
                break

        # 初始化结果列表
        ret = [ ]
        # 遍历分割线列表，获取多边形
        for i in range(len(seg)):
            if seg[ i ][ 1 ] - seg[ i ][ 0 ] > 9:
                ret.append(get_poly(x1, x0, seg[ i ][ 0 ], seg[ i ][ 1 ]))

        # 如果需要绘制图像
        if draw:
            # 绘制多边形
            cv2.polylines(img, ret, True, (255, 0, 0), 10, cv2.LINE_AA)
            # 显示图像
            plt.imshow(img)
            plt.show()

        # 输出结果列表
        logger.debug(f'segment.worker: {[ x.tolist() for x in ret ]}')
        # 返回结果列表、移除按钮和移除模式标志
        return ret, remove_button, remove_mode

    # 捕获异常并记录日志
    except Exception as e:
        logger.debug(traceback.format_exc())
        # 抛出识别错误
        raise RecognizeError(e)
# 定义一个函数用于干员总览的图像分割算法，参数img为输入图像，draw表示是否绘制结果，默认为False
def agent(img, draw=False):
    # 捕获异常并记录异常信息
    except Exception as e:
        # 调试模式下记录异常堆栈信息
        logger.debug(traceback.format_exc())
        # 抛出识别错误并传递异常信息
        raise RecognizeError(e)


# 定义一个函数用于识别未在工作中的干员，参数img为输入图像，draw表示是否绘制结果，默认为False
def free_agent(img, draw=False):
    # 该函数暂时没有实现任何功能，可以根据需求进行补充
    try:
        # 获取图像的高度、宽度和通道数
        height, width, _ = img.shape
        # 设置分辨率为图像的高度
        resolution = height
        # 初始化左右边界
        left, right = 0, width

        # 异形屏适配，找到不透明度大于100的最右侧像素列
        while np.max(img[ :, right - 1 ]) < 100:
            right -= 1
        # 异形屏适配，找到不透明度大于100的最左侧像素列
        while np.max(img[ :, left ]) < 100:
            left += 1

        # 去除左侧干员详情，找到左侧干员详情的右边界
        x0 = left + 1
        while not (img[ height - 10, x0 - 1, 0 ] > img[ height - 10, x0, 0 ] + 10 and abs(
                int(img[ height - 10, x0, 0 ]) - int(img[ height - 10, x0 + 1, 0 ])) < 5):
            x0 += 1

        # 获取分割结果
        ret = agent(img, draw)
        # 获取起点和终点
        st = ret[ -2 ][ 2 ]
        ed = ret[ 0 ][ 1 ]

        # 收集 y 坐标并初步筛选
        y_set = set()
        __ret = [ ]
        for poly in ret:
            # 获取每个干员框的图像
            __img = img[ poly[ 0, 1 ]:poly[ 2, 1 ], poly[ 0, 0 ]:poly[ 2, 0 ] ]
            # 添加干员框的上下边界到 y_set 集合中
            y_set.add(poly[ 0, 1 ])
            y_set.add(poly[ 2, 1 ])
            # 去除空白的干员框
            if 80 <= np.min(__img):
                logger.debug(f'drop(empty): {poly.tolist()}')
                continue
            # 去除被选中的蓝框
            elif np.count_nonzero(__img[ :, :, 0 ] >= 224) == 0 or np.count_nonzero(__img[ :, :, 0 ] == 0) > 0:
                logger.debug(f'drop(selected): {poly.tolist()}')
                continue
            __ret.append(poly)
        ret = __ret

        # 对 y 坐标进行排序
        y1, y2, y4, y5 = sorted(list(y_set))
        y0 = height - y5
        y3 = y0 - y2 + y5

        ret_free = [ ]
        for poly in ret:
            # 将干员框的上边界和下边界调整为 y0 和 y3
            poly[ :, 1 ][ poly[ :, 1 ] == y1 ] = y0
            poly[ :, 1 ][ poly[ :, 1 ] == y4 ] = y3
            __img = img[ poly[ 0, 1 ]:poly[ 2, 1 ], poly[ 0, 0 ]:poly[ 2, 0 ] ]
            # 如果干员框不在值班状态，则添加到 ret_free 列表中
            if not detector.is_on_shift(__img):
                ret_free.append(poly)

        # 如果 draw 为 True，则显示处理后的图像
        if draw:
            __img = img.copy()
            cv2.polylines(__img, ret_free, True, (255, 0, 0), 3, cv2.LINE_AA)
            plt.imshow(__img)
            plt.show()

        # 记录日志并返回处理后的干员框列表、起点和终点
        logger.debug(f'segment.free_agent: {[ x.tolist() for x in ret_free ]}')
        return ret_free, st, ed
    # 捕获任何异常，并记录异常的堆栈信息
    except Exception as e:
        # 使用日志记录器记录异常的堆栈信息
        logger.debug(traceback.format_exc())
        # 抛出自定义的识别错误，并将原始异常作为参数传递
        raise RecognizeError(e)
```