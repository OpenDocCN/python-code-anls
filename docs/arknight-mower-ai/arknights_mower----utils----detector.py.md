# `arknights-mower\arknights_mower\utils\detector.py`

```
import cv2  # 导入 OpenCV 库
import numpy as np  # 导入 NumPy 库

from .. import __rootdir__  # 导入上级目录的 __rootdir__ 模块
from . import typealias as tp  # 导入当前目录下的 typealias 模块，并重命名为 tp
from .image import loadimg  # 从 image 模块中导入 loadimg 函数
from .log import logger  # 从 log 模块中导入 logger 对象
from .matcher import Matcher  # 从 matcher 模块中导入 Matcher 类


def confirm(img: tp.Image) -> tp.Coordinate:  # 定义 confirm 函数，参数为 tp.Image 类型，返回值为 tp.Coordinate 类型
    """
    检测是否出现确认界面
    """
    height, width, _ = img.shape  # 获取图像的高度、宽度和通道数

    # 4 scan lines: left, right, up, down
    left, right = width // 4 * 3 - 10, width // 4 * 3 + 10  # 计算左右扫描线的范围
    up, down = height // 2 - 10, height // 2 + 10  # 计算上下扫描线的范围

    # the R/G/B must be the same for a single pixel in the specified area
    if (img[up:down, left:right, :-1] != img[up:down, left:right, 1:]).any():  # 检查指定区域内的像素的 R/G/B 是否相同
        return None  # 如果不相同，则返回 None

    # the pixel average of the specified area must be in the vicinity of 55
    if abs(np.mean(img[up:down, left:right]) - 55) > 5:  # 检查指定区域内像素的平均值是否在 55 附近
        return None  # 如果不在附近，则返回 None

    # set a new scan line: up
    up = 0  # 初始化上扫描线的位置
    for i in range(down, height):  # 遍历指定区域的下方像素
        for j in range(left, right):  # 遍历指定区域的左右像素
            if np.ptp(img[i, j]) != 0 or abs(img[i, j, 0] - 13) > 3:  # 检查像素的峰峰值和蓝色通道值是否符合条件
                break  # 如果不符合条件，则跳出循环
            elif j == right-1:  # 如果遍历到指定区域的最右侧像素
                up = i  # 更新上扫描线的位置
        if up:  # 如果上扫描线的位置已经更新
            break  # 跳出循环
    if up == 0:  # 如果上扫描线的位置未更新
        return None  # 返回 None

    # set a new scan line: down
    down = 0  # 初始化下扫描线的位置
    for i in range(up, height):  # 遍历指定区域的上方像素
        for j in range(left, right):  # 遍历指定区域的左右像素
            if np.ptp(img[i, j]) != 0 or abs(img[i, j, 0] - 13) > 3:  # 检查像素的峰峰值和蓝色通道值是否符合条件
                down = i  # 更新下扫描线的位置
                break  # 跳出循环
        if down:  # 如果下扫描线的位置已经更新
            break  # 跳出循环
    if down == 0:  # 如果下扫描线的位置未更新
        return None  # 返回 None

    # detect successful
    point = (width // 2, (up + down) // 2)  # 计算检测到的坐标点
    logger.debug(f'detector.confirm: {point}')  # 记录调试信息
    return point  # 返回坐标点


def infra_notification(img: tp.Image) -> tp.Coordinate:  # 定义 infra_notification 函数，参数为 tp.Image 类型，返回值为 tp.Coordinate 类型
    """
    检测基建内是否存在蓝色通知
    前置条件：已经处于基建内
    """
    height, width, _ = img.shape  # 获取图像的高度、宽度和通道数

    # set a new scan line: right
    right = width  # 初始化右扫描线的位置
    while np.max(img[:, right-1]) < 100:  # 遍历图像的垂直像素列，直到找到像素值大于 100 的位置
        right -= 1  # 更新右扫描线的位置
    right -= 1  # 调整右扫描线的位置

    # set a new scan line: up
    up = 0  # 初始化上扫描线的位置
    for i in range(height):  # 遍历图像的水平像素行
        if img[i, right, 0] < 100 < img[i, right, 1] < img[i, right, 2]:  # 检查指定位置像素的颜色是否符合条件
            up = i  # 更新上扫描线的位置
            break  # 跳出循环
    # 如果上边界为0，则返回空
    if up == 0:
        return None

    # 设置一个新的扫描线：下边界
    down = 0
    # 遍历从上边界到图像高度的范围
    for i in range(up, height):
        # 如果当前像素的红色通道值小于100且绿色通道值大于100且蓝色通道值大于100
        if not (img[i, right, 0] < 100 < img[i, right, 1] < img[i, right, 2]):
            # 将下边界设置为当前像素的位置，并跳出循环
            down = i
            break
    # 如果下边界仍为0，则返回空
    if down == 0:
        return None

    # 检测成功，计算检测点的坐标
    point = (right - 10, (up + down) // 2)
    # 记录检测点的信息
    logger.debug(f'detector.infra_notification: {point}')
    # 返回检测点的坐标
    return point
def announcement_close(img: tp.Image) -> tp.Coordinate:
    """
    检测「关闭公告」按钮
    """
    height, width, _ = img.shape

    # 4 scan lines: left, right, up, down
    up, down = 0, height // 4
    left, right = width // 4 * 3, width

    sumx, sumy, cnt = 0, 0, 0
    for i in range(up, down):
        line_cnt = 0
        for j in range(left, right):
            # 检查像素点是否满足条件
            if np.ptp(img[i, j]) == 0 and abs(img[i, j, 0] - 89) < 3:  # condition
                sumx += i
                sumy += j
                cnt += 1
                line_cnt += 1

                # 当一行中满足条件的像素点数量达到100时，返回空值
                if line_cnt >= 100:
                    return None

                # 当满足条件的像素点数量达到2000时，返回检测成功的坐标
                if cnt >= 2000:
                    # detect successful
                    point = (sumy // cnt, sumx // cnt)
                    logger.debug(f'detector.announcement_close: {point}')
                    return point

    return None


def visit_next(img: tp.Image) -> tp.Coordinate:
    """
    检测「访问下位」按钮
    """
    height, width, _ = img.shape

    # 设置新的扫描线: 右边界
    right = width
    while np.max(img[:, right-1]) < 100:
        right -= 1
    right -= 1

    # 设置新的扫描线: 上边界
    up = 0
    for i in range(height):
        if img[i, right, 0] > 150 > img[i, right, 1] > 40 > img[i, right, 2]:
            up = i
            break
    if up == 0:
        return None

    # 设置新的扫描线: 下边界
    down = 0
    for i in range(up, height):
        if not (img[i, right, 0] > 150 > img[i, right, 1] > 40 > img[i, right, 2]):
            down = i
            break
    if down == 0:
        return None

    # 检测成功，返回坐标
    point = (right - 10, (up + down) // 2)
    logger.debug(f'detector.visit_next: {point}')
    return point


on_shift = loadimg(f'{__rootdir__}/resources/agent_on_shift.png', True)
# 从指定路径加载图像，并转换为灰度图像
distracted = loadimg(f'{__rootdir__}/resources/distracted.png', True)
resting = loadimg(f'{__rootdir__}/resources/agent_resting.png', True)

# 检测干员是否正在工作中
def is_on_shift(img: tp.Image) -> bool:
    # 创建图像匹配器对象
    matcher = Matcher(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    # 如果匹配到正在工作的图像，则返回 True
    if matcher.match(on_shift, judge=False) is not None:
        return True
    # 如果匹配到休息的图像，则返回 True
    if matcher.match(resting, judge=False) is not None:
        return True
    # 如果匹配到分心的图像，则返回 False
    if matcher.match(distracted, judge=False) is not None:
        return False
    # 计算图像宽度的 70% 并进行像素比较
    width = img.shape[1]
    __width = int(width * 0.7)
    left_up = np.count_nonzero(np.all(img[0, :__width] <= 62, axis=1) & np.all(30 <= img[0, :__width], axis=1)) / __width
    # 记录调试信息
    logger.debug(f'is_on_shift: {left_up}')
    # 返回是否在工作的判断结果
    return left_up > 0.3
```