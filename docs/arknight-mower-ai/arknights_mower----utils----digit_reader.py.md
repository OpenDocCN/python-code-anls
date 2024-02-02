# `arknights-mower\arknights_mower\utils\digit_reader.py`

```py
# 导入需要的库
import cv2 as cv
import numpy as np
from pathlib import Path
import os
from .image import loadimg
from .. import __rootdir__

# 创建数字识别类
class DigitReader:
    # 初始化方法
    def __init__(self, template_dir=None):
        # 如果未提供模板目录，则使用默认路径
        if not template_dir:
            template_dir = Path(os.path.dirname(os.path.abspath(__file__))) / Path("templates")
        # 如果提供的模板目录不是 Path 对象，则转换为 Path 对象
        if not isinstance(template_dir, Path):
            template_dir = Path(template_dir)
        # 初始化时间模板和无人机模板
        self.time_template = []
        self.drone_template = []
        # 遍历数字 0 到 9
        for i in range(10):
            # 加载时间模板图片并添加到时间模板列表中
            self.time_template.append(
                loadimg(f'{__rootdir__}/resources/orders_time/{i}.png', True)
            )
            # 加载无人机模板图片并添加到无人机模板列表中
            self.drone_template.append(
                loadimg(f'{__rootdir__}/resources/drone_count/{i}.png', True)
            )

    # 获取无人机数字的方法
    def get_drone(self, img_grey):
        # 从灰度图像中提取无人机部分
        drone_part = img_grey[32:76, 1144:1225]
        # 初始化结果字典
        result = {}
        # 遍历数字 0 到 9
        for j in range(10):
            # 使用模板匹配方法进行匹配
            res = cv.matchTemplate(
                drone_part,
                self.drone_template[j],
                cv.TM_CCORR_NORMED,
            )
            # 设置匹配阈值
            threshold = 0.95
            # 获取匹配位置
            loc = np.where(res >= threshold)
            # 遍历匹配位置
            for i in range(len(loc[0])):
                offset = loc[1][i]
                accept = True
                # 检查是否已经存在相似的匹配结果
                for o in result:
                    if abs(o - offset) < 5:
                        accept = False
                        break
                # 如果是新的匹配结果，则添加到结果字典中
                if accept:
                    result[loc[1][i]] = j
        # 将结果字典按键排序并转换为数字
        l = [str(result[k]) for k in sorted(result)]
        return int("".join(l))
    # 获取灰度图像的数字部分
    digit_part = img_grey[510:543, 499:1920]
    # 初始化结果字典
    result = {}
    # 遍历0到9的数字模板
    for j in range(10):
        # 使用模板匹配函数匹配数字部分和数字模板，返回匹配结果
        res = cv.matchTemplate(
            digit_part,
            self.time_template[j],
            cv.TM_CCOEFF_NORMED,
        )
        # 设置匹配阈值
        threshold = 0.85
        # 获取匹配结果大于阈值的位置
        loc = np.where(res >= threshold)
        # 遍历匹配位置
        for i in range(len(loc[0])):
            # 获取匹配位置的横坐标
            x = loc[1][i]
            # 初始化接受标志
            accept = True
            # 遍历结果字典
            for o in result:
                # 如果当前位置与已有位置的差值小于5，则不接受
                if abs(o - x) < 5:
                    accept = False
                    break
            # 如果接受当前位置
            if accept:
                # 如果结果字典为空
                if len(result) == 0:
                    # 裁剪数字部分，保留当前位置及后续116个像素
                    digit_part = digit_part[:, loc[1][i] - 5 : loc[1][i] + 116]
                    # 计算偏移量
                    offset = loc[1][0] - 5
                    # 对所有匹配位置进行偏移
                    for m in range(len(loc[1])):
                        loc[1][m] -= offset
                # 将当前位置及对应的数字添加到结果字典中
                result[loc[1][i]] = j
    # 将结果字典按键排序，将数字拼接成时间格式返回
    l = [str(result[k]) for k in sorted(result)]
    return f"{l[0]}{l[1]}:{l[2]}{l[3]}:{l[4]}{l[5]}"
```