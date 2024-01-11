# `arknights-mower\arknights_mower\utils\recognize.py`

```
# 从 __future__ 模块中导入 annotations 特性
from __future__ import annotations

# 导入时间模块
import time
# 导入类型提示模块中的 List 和 Optional 类型
from typing import List, Optional

# 导入 OpenCV 模块
import cv2
# 导入 NumPy 模块，并使用 np 别名
import numpy as np

# 从上一级目录中导入 __rootdir__ 模块
from .. import __rootdir__
# 从当前目录中导入 config、detector、typealias、device、image、log、matcher、scene 模块
from . import config, detector
from . import typealias as tp
from .device import Device
from .image import bytes2img, cropimg, loadimg, thres2
from .log import logger, save_screenshot
from .matcher import Matcher
from .scene import Scene, SceneComment


# 定义 RecognizeError 类，继承自 Exception 类
class RecognizeError(Exception):
    pass


# 定义 Recognizer 类
class Recognizer(object):

    # 初始化方法
    def __init__(self, device: Device, screencap: bytes = None) -> None:
        # 设备属性赋值
        self.device = device
        # 调用 start 方法
        self.start(screencap)

    # start 方法
    def start(self, screencap: bytes = None, build: bool = True) -> None:
        """ init with screencap, build matcher  """
        # 初始化重试次数
        retry_times = config.MAX_RETRYTIME
        # 循环，直到重试次数为 0
        while retry_times > 0:
            try:
                # 如果传入了截屏数据，则使用传入的数据，否则调用设备的截屏方法获取数据
                if screencap is not None:
                    self.screencap = screencap
                else:
                    self.screencap = self.device.screencap()
                # 将截屏数据转换为图像数据
                self.img = bytes2img(self.screencap, False)
                # 将截屏数据转换为灰度图像数据
                self.gray = bytes2img(self.screencap, True)
                # 获取图像的高度、宽度和通道数
                self.h, self.w, _ = self.img.shape
                # 如果 build 为 True，则创建 Matcher 对象，否则为 None
                self.matcher = Matcher(self.gray) if build else None
                # 设置场景为未定义
                self.scene = Scene.UNDEFINED
                return
            # 捕获 OpenCV 错误
            except cv2.error as e:
                # 记录警告日志
                logger.warning(e)
                # 重试次数减一
                retry_times -= 1
                # 休眠 1 秒
                time.sleep(1)
                # 继续下一次循环
                continue
        # 如果重试次数为 0，则抛出运行时错误
        raise RuntimeError('init Recognizer failed')

    # update 方法
    def update(self, screencap: bytes = None, rebuild: bool = True) -> None:
        """ rebuild matcher """
        # 调用 start 方法
        self.start(screencap, rebuild)

    # color 方法
    def color(self, x: int, y: int) -> tp.Pixel:
        """ get the color of the pixel """
        # 返回图像指定位置的像素颜色
        return self.img[y][x]

    # save_screencap 方法
    def save_screencap(self, folder):
        # 保存截屏数据
        save_screenshot(self.screencap, subdir=f'{folder}/{self.h}x{self.w}')
    # 检查当前场景是否全黑
    def is_black(self) -> None:
        """ check if the current scene is all black """
        return np.max(self.gray[:, 105:-105]) < 16

    # 查找导航按钮
    def nav_button(self):
        """ find navigation button """
        return self.find('nav_button', thres=128, scope=((0, 0), (100+self.w//4, self.h//10)))

    # 查找元素是否出现在画面中
    def find(self, res: str, draw: bool = False, scope: tp.Scope = None, thres: int = None, judge: bool = True, strict: bool = False,score = 0.0) -> tp.Scope:
        """
        查找元素是否出现在画面中

        :param res: 待识别元素资源文件名
        :param draw: 是否将识别结果输出到屏幕
        :param scope: ((x0, y0), (x1, y1))，提前限定元素可能出现的范围
        :param thres: 是否在匹配前对图像进行二值化处理
        :param judge: 是否加入更加精确的判断
        :param strict: 是否启用严格模式，未找到时报错
        :param score: 是否启用分数限制，有些图片精确识别需要提高分数阈值

        :return ret: 若匹配成功，则返回元素在游戏界面中出现的位置，否则返回 None
        """
        logger.debug(f'find: {res}')
        res = f'{__rootdir__}/resources/{res}.png'

        if thres is not None:
            # 对图像二值化处理
            res_img = thres2(loadimg(res, True), thres)
            gray_img = cropimg(self.gray, scope)
            matcher = Matcher(thres2(gray_img, thres))
            ret = matcher.match(res_img, draw=draw, judge=judge,prescore=score)
        else:
            res_img = loadimg(res, True)
            matcher = self.matcher
            ret = matcher.match(res_img, draw=draw, scope=scope, judge=judge,prescore=score)
        if strict and ret is None:
            raise RecognizeError(f"Can't find '{res}'") 
        return ret
    # 定义一个方法，用于查找元素是否出现在画面中，并返回分数
    def score(self, res: str, draw: bool = False, scope: tp.Scope = None, thres: int = None) -> Optional[List[float]]:
        """
        查找元素是否出现在画面中，并返回分数

        :param res: 待识别元素资源文件名
        :param draw: 是否将识别结果输出到屏幕
        :param scope: ((x0, y0), (x1, y1))，提前限定元素可能出现的范围
        :param thres: 是否在匹配前对图像进行二值化处理

        :return ret: 若匹配成功，则返回元素在游戏界面中出现的位置，否则返回 None
        """
        # 打印调试信息，显示待识别元素资源文件名
        logger.debug(f'find: {res}')
        # 拼接资源文件的完整路径
        res = f'{__rootdir__}/resources/{res}.png'

        if thres is not None:
            # 如果指定了二值化处理的阈值，则对图像进行二值化处理
            res_img = thres2(loadimg(res, True), thres)
            # 裁剪灰度图像
            gray_img = cropimg(self.gray, scope)
            # 创建匹配器对象
            matcher = Matcher(thres2(gray_img, thres))
            # 获取匹配分数
            score = matcher.score(res_img, draw=draw, only_score=True)
        else:
            # 如果没有指定二值化处理的阈值，则直接加载图像
            res_img = loadimg(res, True)
            # 使用预先创建的匹配器对象
            matcher = self.matcher
            # 获取匹配分数
            score = matcher.score(res_img, draw=draw, scope=scope, only_score=True)
        # 返回匹配分数
        return score
```