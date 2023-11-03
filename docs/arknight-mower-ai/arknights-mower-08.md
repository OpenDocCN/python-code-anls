# ArknightMower源码解析 8

# `/opt/arknights-mower/arknights_mower/utils/datetime.py`

这段代码定义了两个函数，分别名为`the_same_day`和`the_same_time`，它们接受两个参数，均为`datetime`类型。这两个函数用于比较两个日期或时间是否相同。

函数`the_same_day`的作用是判断两个日期是否相同。具体来说，如果两个参数`a`和`b`中的任何一个为`None`，则返回False。否则，函数将返回`a`和`b`的年份和月份必须相同，且年份必须相同。

函数`the_same_time`的作用是判断两个时间是否相同。具体来说，如果两个参数`a`和`b`中的任何一个为`None`，则返回False。否则，函数将返回两个时间之间的差距（以秒为单位）小于1.5。

总之，这两个函数用于比较日期和时间是否相同，是Python中非常常用的函数。


```
from datetime import datetime
import pytz


def the_same_day(a: datetime = None, b: datetime = None) -> bool:
    if a is None or b is None:
        return False
    return a.year == b.year and a.month == b.month and a.day == b.day


def the_same_time(a: datetime = None, b: datetime = None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b).total_seconds() < 1.5


```

这段代码定义了一个名为 `get_server_weekday` 的函数，它使用了 `datetime.now` 函数来获取当前日期，并将其转换为指定的时区，使用了 `pytz` 模块来处理时区。

函数的作用是返回当前日期是星期几，如果当前日期是星期天，则返回 0，否则返回 1。函数返回的值类型是整数类型。

该函数的输出将是 0 或 1，而不是函数本身。


```
def get_server_weekday():
    return datetime.now(pytz.timezone('Asia/Dubai')).weekday()

```

# `/opt/arknights-mower/arknights_mower/utils/detector.py`

This is a Python class that uses the OpenCV library that implements the确认(confirmation) and contour detection algorithm.

The confirm method takes an image and returns the top-left corner of the detected confirmation (corner)坐标。

The corner detection algorithm uses the following steps:

1. Check for the presence of a pre-defined number of scan lines in the image.
2. Iterate through扫描 lines.
3. Check if the pixel value at the center of the scan line is within a certain range for a single pixel in the specified area.
4. Check if the average pixel value of the specified area is within a certain distance from the specified "確定のカインient" value.
5. Set the current scan line to the center of the detected corner.

If the corner is detected, it is returned as a tuple of the coordinates (width // 2, height // 2).

If the corner is not detected within the pre-defined time or if the parameters of the `confirm` method are not correct, an None is returned.


```
import cv2
import numpy as np

from .. import __rootdir__
from . import typealias as tp
from .image import loadimg
from .log import logger
from .matcher import Matcher


def confirm(img: tp.Image) -> tp.Coordinate:
    """
    检测是否出现确认界面
    """
    height, width, _ = img.shape

    # 4 scan lines: left, right, up, down
    left, right = width // 4 * 3 - 10, width // 4 * 3 + 10
    up, down = height // 2 - 10, height // 2 + 10

    # the R/G/B must be the same for a single pixel in the specified area
    if (img[up:down, left:right, :-1] != img[up:down, left:right, 1:]).any():
        return None

    # the pixel average of the specified area must be in the vicinity of 55
    if abs(np.mean(img[up:down, left:right]) - 55) > 5:
        return None

    # set a new scan line: up
    up = 0
    for i in range(down, height):
        for j in range(left, right):
            if np.ptp(img[i, j]) != 0 or abs(img[i, j, 0] - 13) > 3:
                break
            elif j == right-1:
                up = i
        if up:
            break
    if up == 0:
        return None

    # set a new scan line: down
    down = 0
    for i in range(up, height):
        for j in range(left, right):
            if np.ptp(img[i, j]) != 0 or abs(img[i, j, 0] - 13) > 3:
                down = i
                break
        if down:
            break
    if down == 0:
        return None

    # detect successful
    point = (width // 2, (up + down) // 2)
    logger.debug(f'detector.confirm: {point}')
    return point


```

这段代码定义了一个名为`infra_notification`的函数，它接受一个名为`img`的二维图像作为参数，并返回一个名为`point`的二维坐标。

函数的作用是检测图像中是否包含 blue通知（即设施内存在某种指示性的颜色）。为了达到这个目的，函数在前置条件满足（即图片已经位于设施内）的情况下，从图片的右上角开始，扫描并处理每一列的像素值，直到找到第一个像素值大于100的位置。接下来，函数将从上到下扫描，处理与之前扫描行同一列的像素值。通过这种方式，如果图片中某个位置的三个相邻像素都小于100，那么函数就可以判定该位置一定存在 blue通知，函数将返回该位置的坐标。如果循环遍历所有位置仍然没有找到 blue通知，函数将返回 None。


```
def infra_notification(img: tp.Image) -> tp.Coordinate:
    """
    检测基建内是否存在蓝色通知
    前置条件：已经处于基建内
    """
    height, width, _ = img.shape

    # set a new scan line: right
    right = width
    while np.max(img[:, right-1]) < 100:
        right -= 1
    right -= 1

    # set a new scan line: up
    up = 0
    for i in range(height):
        if img[i, right, 0] < 100 < img[i, right, 1] < img[i, right, 2]:
            up = i
            break
    if up == 0:
        return None

    # set a new scan line: down
    down = 0
    for i in range(up, height):
        if not (img[i, right, 0] < 100 < img[i, right, 1] < img[i, right, 2]):
            down = i
            break
    if down == 0:
        return None

    # detect successful
    point = (right - 10, (up + down) // 2)
    logger.debug(f'detector.infra_notification: {point}')
    return point


```

这段代码定义了一个名为"announcement_close"的函数，其输入参数是一个名为"img"的二维图像对象。这个函数返回一个名为"coordinate"的元组类型，包含一个或多个坐标点，表示检测到的关闭按钮位置。

具体来说，这个函数的实现过程如下：

1. 读取输入图像的尺寸高度和每行扫描线的数量。
2. 定义左右和上下扫描线的位置，初始化为0。
3. 循环遍历图像的每一行，统计该行扫描线上与关闭按钮的距离为0的像素个数、该行扫描线上的像素个数和该行扫描线上的像素个数之和。
4. 如果统计结果满足条件，就统计到该行的像素个数。
5. 如果统计结果达到2000，就认为检测到关闭按钮，返回检测到的坐标点。
6. 如果以上步骤都未能检测到关闭按钮，返回None。

这个函数的作用是用于图像识别领域中的关闭按钮检测，可以帮助开发者判断图像中是否包含关闭按钮，并根据按钮位置给出相关信息。


```
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
            if np.ptp(img[i, j]) == 0 and abs(img[i, j, 0] - 89) < 3:  # condition
                sumx += i
                sumy += j
                cnt += 1
                line_cnt += 1

                # the number of pixels meeting the condition in one line reaches 100
                if line_cnt >= 100:
                    return None

                # the number of pixels meeting the condition reaches 2000
                if cnt >= 2000:
                    # detect successful
                    point = (sumy // cnt, sumx // cnt)
                    logger.debug(f'detector.announcement_close: {point}')
                    return point

    return None


```

这段代码是一个函数 `visit_next(img: tp.Image) -> tp.Coordinate`，它用于检测图像中的「访问下位」按钮。

函数接收一个图像对象 `img`，并返回一个坐标 `point`，表示按钮的位置。

以下是函数的实现细节：

1. 函数首先获取图像的高度、宽度和元素数量。
2. 创建一个新的扫描线，指向右下角。
3. 循环遍历新的扫描线，直到检测到高度元素中的最大值大于 100。
4. 循环遍历新的扫描线，直到检测到高度元素中的最大值大于 40 且第二高元素大于 40 且第二高元素大于高度元素中的最大值。
5. 如果以上条件都不满足，则返回 None，表示没有检测到按钮。
6. 如果检测到按钮，则返回按钮的位置。

该函数可以被理解为在图像中找到按钮的位置，并返回该位置的坐标。


```
def visit_next(img: tp.Image) -> tp.Coordinate:
    """
    检测「访问下位」按钮
    """
    height, width, _ = img.shape

    # set a new scan line: right
    right = width
    while np.max(img[:, right-1]) < 100:
        right -= 1
    right -= 1

    # set a new scan line: up
    up = 0
    for i in range(height):
        if img[i, right, 0] > 150 > img[i, right, 1] > 40 > img[i, right, 2]:
            up = i
            break
    if up == 0:
        return None

    # set a new scan line: down
    down = 0
    for i in range(up, height):
        if not (img[i, right, 0] > 150 > img[i, right, 1] > 40 > img[i, right, 2]):
            down = i
            break
    if down == 0:
        return None

    # detect successful
    point = (right - 10, (up + down) // 2)
    logger.debug(f'detector.visit_next: {point}')
    return point


```

这段代码定义了一个名为 `is_on_shift` 的函数，用于判断一个 GR2D 图像(即一个图片)是否显示了干员在工作中的状态。

首先，代码使用 `loadimg` 函数从 `__rootdir__` 目录下加载了三个 GR2D 图像，分别命名为 `on_shift.png`、`distracted.png` 和 `agent_resting.png`。

接着，代码定义了一个函数 `is_on_shift`，该函数接受一个 GR2D 图像作为参数，并返回一个布尔值，表示该图像是否显示了干员在工作中的状态。

函数中使用了 `Matcher` 类，它是一个 `cv2.Cv2Matcher` 类的实例，用于匹配图像中的像素。`Matcher` 类接受两个参数：要匹配的图像和回查的阈值，回查指的是一个处理函数，用于确定是否匹配。在这里， `Matcher` 类用于检测一个名为 `on_shift` 的图像是否与 `on_shift` 本身或者一个名为 `resting` 的图像是否匹配。如果 `is_on_shift` 检测到匹配到了 `on_shift` 或者 `resting`，函数将返回 `True`；否则返回 `False`。

此外，代码还定义了一个 `is_on_shift` 的函数，它的实现与上述函数类似，只是检测的图片是 `distracted` 而不是 `on_shift`。


```
on_shift = loadimg(f'{__rootdir__}/resources/agent_on_shift.png', True)
distracted = loadimg(f'{__rootdir__}/resources/distracted.png', True)
resting = loadimg(f'{__rootdir__}/resources/agent_resting.png', True)

def is_on_shift(img: tp.Image) -> bool:
    """
    检测干员是否正在工作中
    """
    matcher = Matcher(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    if matcher.match(on_shift, judge=False) is not None:
        return True
    if matcher.match(resting, judge=False) is not None:
        return True
    if matcher.match(distracted, judge=False) is not None:
        return False
    width = img.shape[1]
    __width = int(width * 0.7)
    left_up = np.count_nonzero(np.all(img[0, :__width] <= 62, axis=1) & np.all(30 <= img[0, :__width], axis=1)) / __width
    logger.debug(f'is_on_shift: {left_up}')
    return left_up > 0.3

```

# `/opt/arknights-mower/arknights_mower/utils/digit_reader.py`



This is a Python class that uses OpenCV to perform various operations on a drone image.

The drone image is first converted to grayscale and then a template is created for a standard drone model. The `get_time()` method then takes an image and returns the time information (e.g. GPS location) based on the drone model.

The `drone_template` class sets the template for the drone model.
css
drone_template = {
   'A': [3, 2, 1, 2, 1],
   'B': [2, 3, 3, 2, 3],
   'C': [4, 5, 6, 7, 8],
   'D': [1, 2, 3, 4, 5],
   'E': [2, 3, 4, 5, 6],
   'F': [3, 4, 5, 6, 7],
   'G': [2, 3, 4, 5, 6],
   'H': [4, 5, 6, 7, 8],
   'I': [3, 4, 5, 6, 7]
}

The `res_template` class creates the result of the matching process.
less
res_template = {
   'A': [6, 5, 3, 2, 1],
   'B': [7, 8, 6, 3, 2],
   'C': [10, 9, 8, 5, 6],
   'D': [11, 12, 13, 14, 15],
   'E': [12, 13, 14, 15, 16],
   'F': [13, 14, 15, 16, 17],
   'G': [14, 15, 16, 17, 18],
   'H': [15, 16, 17, 18, 19],
   'I': [16, 17, 18, 19, 20]
}

The class also defines some utility methods such as `get_time()` and `drone_part()`, which are used in the `res_template` class.


```
import cv2 as cv
import numpy as np
from pathlib import Path
import os
from .image import loadimg
from .. import __rootdir__


class DigitReader:
    def __init__(self, template_dir=None):
        if not template_dir:
            template_dir = Path(os.path.dirname(os.path.abspath(__file__))) / Path("templates")
        if not isinstance(template_dir, Path):
            template_dir = Path(template_dir)
        self.time_template = []
        self.drone_template = []
        for i in range(10):
            self.time_template.append(
                loadimg(f'{__rootdir__}/resources/orders_time/{i}.png', True)
            )
            self.drone_template.append(
                loadimg(f'{__rootdir__}/resources/drone_count/{i}.png', True)
            )

    def get_drone(self, img_grey):
        drone_part = img_grey[32:76, 1144:1225]
        result = {}
        for j in range(10):
            res = cv.matchTemplate(
                drone_part,
                self.drone_template[j],
                cv.TM_CCORR_NORMED,
            )
            threshold = 0.95
            loc = np.where(res >= threshold)
            for i in range(len(loc[0])):
                offset = loc[1][i]
                accept = True
                for o in result:
                    if abs(o - offset) < 5:
                        accept = False
                        break
                if accept:
                    result[loc[1][i]] = j
        l = [str(result[k]) for k in sorted(result)]
        return int("".join(l))

    def get_time(self, img_grey):
        digit_part = img_grey[510:543, 499:1920]
        result = {}
        for j in range(10):
            res = cv.matchTemplate(
                digit_part,
                self.time_template[j],
                cv.TM_CCOEFF_NORMED,
            )
            threshold = 0.85
            loc = np.where(res >= threshold)
            for i in range(len(loc[0])):
                x = loc[1][i]
                accept = True
                for o in result:
                    if abs(o - x) < 5:
                        accept = False
                        break
                if accept:
                    if len(result) == 0:
                        digit_part = digit_part[:, loc[1][i] - 5 : loc[1][i] + 116]
                        offset = loc[1][0] - 5
                        for m in range(len(loc[1])):
                            loc[1][m] -= offset
                    result[loc[1][i]] = j
        l = [str(result[k]) for k in sorted(result)]
        return f"{l[0]}{l[1]}:{l[2]}{l[3]}:{l[4]}{l[5]}"

```

# `/opt/arknights-mower/arknights_mower/utils/email.py`

这段代码使用了Jinja2框架来生成动态HTML模板。它主要做了以下几件事情：

1. 导入需要的库：Jinja2、os、sys。
2. 设置Arknights Mower模板目录。如果没有在系统环境变量中设置模板目录，则默认为当前工作目录。
3. 检查当前系统是否为Python，如果是，则执行系统补丁。
4. 遍历电子邮件文件夹。
5. 如果当前系统不是Python，或者模板目录不存在，则创建模板目录。
6. 加载模板目录下的所有文件。


```
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import sys


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    template_dir = os.path.join(
        sys._MEIPASS,
        "arknights_mower",
        "__init__",
        "templates",
        "email",
    )
else:
    template_dir = os.path.join(
        os.getcwd(),
        "arknights_mower",
        "templates",
        "email",
    )

```

这段代码是使用Python的包 Environmental as Environmental to create an instance of an Environment object.这个Instance用来自环境的加载器、自动escape以及获取指定文件的模板。

进一步解析：

1. `loader=FileSystemLoader(template_dir)`: 从指定的模板目录中加载模板。

2. `autoescape=select_autoescape()`: 从 "select_autoescape()" 函数中选择正确的自动escape 类型。

3. `env.get_template("task.html")`: 加载名为 "task.html" 的模板。

4. `env.get_template("maa.html")`: 加载名为 "maa.html" 的模板。

5. `env.get_template("recruit_template.html")`: 加载名为 "recruit_template.html" 的模板。

6. `env.get_template("recruit_rarity.html")`: 加载名为 "recruit_rarity.html" 的模板。

7. `task_template = env.get_template("task.html")`: 将上面获取的模板保存为 `task_template`。

8. `maa_template = env.get_template("maa.html")`: 将上面获取的模板保存为 `maa_template`。

9. `recruit_template = env.get_template("recruit_template.html")`: 将上面获取的模板保存为 `recruit_template`。

10. `recruit_rarity = env.get_template("recruit_rarity.html")`: 将上面获取的模板保存为 `recruit_rarity`。


```
env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=select_autoescape(),
)

task_template = env.get_template("task.html")
maa_template = env.get_template("maa.html")
recruit_template = env.get_template("recruit_template.html")
recruit_rarity = env.get_template("recruit_rarity.html")

```

# `/opt/arknights-mower/arknights_mower/utils/image.py`

这段代码定义了一个名为 "bytes2img" 的函数，它接受一个名为 "data" 的字节数组，并返回一个名为 "image" 的图像类型。

函数有两个参数，一个是灰度模式（True）另一个是黑色图像模式（False）。函数内部使用 "cv2" 库中的 "imdecode" 函数将字节数组转换为图像，如果参数 "gray" 是True，则函数使用灰度模式。

函数返回一个 "Union" 类型的对象，"Union" 是一个类型兼容的类型，允许函数返回多种类型的之一。在这个函数中，它允许返回两个参数之一，"image" 或者 "grayImage"。


```
from typing import Union

import cv2
import numpy as np

from . import typealias as tp
from .log import logger, save_screenshot


def bytes2img(data: bytes, gray: bool = False) -> Union[tp.Image, tp.GrayImage]:
    """ bytes -> image """
    if gray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.cvtColor(
            cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )


```

这段代码定义了两个函数，分别是 `img2bytes` 和 `loadimg`。这两个函数的主要作用是图像处理和加载。

1. `img2bytes` 函数接收一个图像对象（img），并将其转换成字节数组。具体操作是通过调用 `cv2.imencode()` 函数，并传递一个表示输出图像类型的参数 `.png`。`cv2.imencode()` 函数将图像转换成指定格式的图像，并返回一个字节数组，其中第一个参数表示图像类型，第二个参数表示转换方式。在这里，第一个参数是 `.png`，表示将图像转换成 PNG 格式的字节数组。

2. `loadimg` 函数接收一个图像文件的文件名（filename），并返回一个图像对象。如果 `gray` 为 `True`，函数使用 `cv2.imread()` 函数将图像从文件中读取，并返回一个灰度图像。否则，函数使用 `cv2.cvtColor()` 函数将灰度图像转换为彩色图像，并返回一个彩色图像。

3. `thres2` 函数接收一个灰度图像对象（img），和一个阈值（thresh），并返回一个新的灰度图像。具体操作是通过调用 `cv2.threshold()` 函数，将图像阈值以下的部分设置为黑色，并将阈值以上（不包括阈值）的部分设置为白色。函数返回一个新的灰度图像，其中黑色部分被设置为 `thresh`，白色部分被设置为 `255 - thresh`。


```
def img2bytes(img) -> bytes:
    """ bytes -> image """
    return cv2.imencode('.png', img)[1]


def loadimg(filename: str, gray: bool = False) -> Union[tp.Image, tp.GrayImage]:
    """ load image from file """
    logger.debug(filename)
    if gray:
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def thres2(img: tp.GrayImage, thresh: int) -> tp.GrayImage:
    """ binarization of images """
    _, ret = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return ret


```

该函数定义了一个名为`thres0`的函数，它接受一个图像`img`和一个阈值`thresh`作为参数，并返回一个图像`ret`。

函数的主要目的是对传入的图像进行处理，主要通过删除所有不满足条件的像素值，具体实现如下：

1. 首先，函数将输入的图像`img`的轮廓复制出来，以便于后续的操作。

2. 如果图像`img`的轮廓有3个维度，那么函数会执行以下操作：

  a. 创建一个新的 image 变量`ret`，与输入图像`img`保持相同的大小。

  b. 将图像`img`转换为灰度图像，这样所有像素的值都只关心像素的亮度，而不关心像素的颜色。

  c. 遍历图像`img`的每个像素点，计算出该像素点是否小于等于阈值`thresh`。如果该像素点满足条件，则将其设置为0，否则将其保留。

  d. 根据计算得到的条件，对图像`ret`中的像素点进行筛选，保留满足条件的像素点。

  e. 最后，函数返回处理后的图像`ret`。

函数的作用是保留图像中所有满足条件的像素点，从而实现去除图像中不符合要求的像素点的功能。


```
# def thres0(img: tp.Image, thresh: int) -> tp.Image:
#     """ delete pixel, filter: value > thresh """
#     ret = img.copy()
#     if len(ret.shape) == 3:
#         # ret[rgb2gray(img) <= thresh] = 0
#         z0 = ret[:, :, 0]
#         z1 = ret[:, :, 1]
#         z2 = ret[:, :, 2]
#         _ = (z0 <= thresh) | (z1 <= thresh) | (z2 <= thresh)
#         z0[_] = 0
#         z1[_] = 0
#         z2[_] = 0
#     else:
#         ret[ret <= thresh] = 0
#     return ret


```

这段代码定义了三个函数，函数的作用是：

1. `thres0(img: tp.Image, thresh: int) -> tp.Image` 是一个非空函数，接收一个 tp.Image 类型的图像和一个整数类型的阈值作为参数。函数的作用是删除图像中所有像素值大于阈值的像素，然后返回处理后的图像。
2. `rgb2gray(img: tp.Image) -> tp.GrayImage` 是一个非空函数，接收一个 tp.Image 类型的图像作为参数。函数的作用是在图像中把所有像素从 RGB 颜色空间转换为灰度颜色空间。
3. `scope2slice(scope: tp.Scope) -> tp.Slice` 是一个非空函数，接收一个 tp.Scope 类型的对象作为参数。函数的作用是在给定scope对象的范围内，返回一个左闭右开区间的切片。其中，切片区间的左端点是scope对象的左边界，右端点是scope对象的右边界，左闭右开表示左端点不包含在切片范围内，右闭左开表示右端点不包含在切片范围内。


```
# def thres0(img: tp.Image, thresh: int) -> tp.Image:  # not support multichannel image
#     """ delete pixel which > thresh """
#     _, ret = cv2.threshold(img, thresh, 255, cv2.THRESH_TOZERO)
#     return ret


def rgb2gray(img: tp.Image) -> tp.GrayImage:
    """ change image from rgb to gray """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def scope2slice(scope: tp.Scope) -> tp.Slice:
    """ ((x0, y0), (x1, y1)) -> ((y0, y1), (x0, x1)) """
    if scope is None:
        return slice(None), slice(None)
    return slice(scope[0][1], scope[1][1]), slice(scope[0][0], scope[1][0])


```

这两段代码定义了一个名为 "cropimg" 的函数和一个名为 "saveimg" 的函数。

"cropimg" 函数接收一个名为 "img" 的 Image 对象和一个名为 "scope" 的 Scope 对象。函数的作用是返回一个名为 "img" 的新 Image 对象，它仅包含 `scope` 中的图像区域。函数的具体实现是，先将 `img` 对象缩放到 `scope` 范围内，然后返回 `img` 对象中 `scope2slice` 方法返回的图像区域。

"saveimg" 函数接收一个名为 "img" 的 Image 对象和一个名为 "folder" 的字符串参数，用于保存截图到指定的文件夹中。函数的具体实现是，将 `img` 对象转换为 bytes 类型，然后使用 `cv2.cvtColor` 函数将其转换为 RGB 颜色空间。接着，使用 `subdir` 参数指定文件保存目录，并将转换后的图像保存到指定的文件夹中。


```
def cropimg(img: tp.Image, scope: tp.Scope) -> tp.Image:
    """ crop image """
    return img[scope2slice(scope)]


def saveimg(img, folder='failure'):
    save_screenshot(
        img2bytes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        subdir=f'{folder}/{img.shape[0]}x{img.shape[1]}',
    )

```

# `/opt/arknights-mower/arknights_mower/utils/log.py`

这段代码的作用是创建一个日志记录器（logger），用于记录当前目录（Path.current）中文件的访问日志。

具体来说，它首先引入了logging库、os库、sys库、threading库和time库。然后，通过importing模块的方式，将colorlog库也导入进来，以便能够使用颜色编码显示日志信息。

接着，定义了一个BASIC_FORMAT和COLOR_FORMAT两种日志格式，分别用于打印日志信息和不带颜色的日志信息。其中，BASIC_FORMAT使用了当前日期、时间、相对路径、线性索引号和函数名称作为参数；而COLOR_FORMAT则使用了颜色编码和更多的打印信息，可以在不丢失任何信息的情况下打印日志。

接下来，通过旋转文件的方式（利用time.time()函数获取当前时间），在每次文件访问时创建一个新的日志记录（file.pyc中的create_log_file()函数），并使用RotatingFileHandler类将日志记录保存在一个文件中。这个文件每天只会保存最近7天的日志，而剩下的日志则会被删除，以便腾出文件空间。

最后，通过创建一个BasicFormatter对象和一个RotatingFileHandler对象，设置日志记录器的格式为BASIC_FORMAT，然后将日志记录器与当前目录（Path.current）的文件访问进行关联。这样，每当文件被访问时，就会创建一个新的日志记录，并使用BASIC_FORMAT格式来打印日志信息。


```
import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
from . import config

BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(relativepath)s:%(lineno)d - %(funcName)s - %(message)s'
COLOR_FORMAT = '%(log_color)s%(asctime)s - %(levelname)s - %(relativepath)s:%(lineno)d - %(funcName)s - %(message)s'
DATE_FORMAT = None
basic_formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
```

这段代码定义了一个名为 "color_formatter" 的类，该类实现了 Python 标准库中的 "colorlog" 模块中的 "ColoredFormatter" 类，用于将日志信息与颜色相关联。

该代码还定义了一个名为 "PackagePathFilter" 的类，该类实现了 "logging.Filter" 类，用于在日志记录中过滤文件路径。

具体来说，代码中的 "color_formatter = colorlog.ColoredFormatter(COLOR_FORMAT, DATE_FORMAT)" 行代码创建了一个 "ColoredFormatter" 对象，其中 "COLOR_FORMAT" 和 "DATE_FORMAT" 是颜色和日期格式，分别用于在日志信息中使用颜色和日期格式。这些格式信息将在后续的 "logging.StreamHandler" 实例中使用，以便将日志信息与颜色相关联。

而 "PackagePathFilter = logging.Filter()" 行代码创建了一个 "logging.Filter" 对象，该对象用于在日志记录中过滤文件路径。具体来说，该类中的 "filter" 方法将在每个日志记录中检查路径是否以 "package" 为前缀。如果是，则该方法将记录的 "relativepath" 属性设置为路径相对于当前包的路径，并将 "abs_sys_paths" 数组中的路径与 "path" 比较，以获取与 "path" 比较的最后一个路径。这段代码将在每个日志记录中使用 "colorlog" 模块中的 "ColoredFormatter" 将日志信息与颜色相关联，并在 "logging.StreamHandler" 实例中使用颜色。


```
color_formatter = colorlog.ColoredFormatter(COLOR_FORMAT, DATE_FORMAT)


class PackagePathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


```



这段代码定义了一个名为`MaxFilter`的类，用于处理日志记录。该类包含了一个`__init__`方法和一个`filter`方法。

在`__init__`方法中，该类指定了一个最大级别，即在处理日志记录时最高允许记录的级别。

在`filter`方法中，该类根据传入的`record`对象，检查其日志级别是否小于或等于定义的最大级别。如果是，则返回`True`，表示继续处理该记录。否则，返回`False`，停止处理记录。

接下来，定义了一个名为`Handler`的类，该类继承自`logging.StreamHandler`类。该类包含了一个`__init__`方法和一个`emit`方法。

在`__init__`方法中，该类创建了一个`pipe`对象，用于从源代码流中读取日志记录。

在`emit`方法中，该类将`record`对象转换为JSON格式，并将其作为JSON对象发送到源代码流的结尾。


```
class MaxFilter(object):
    def __init__(self, max_level: int) -> None:
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= self.max_level:
            return True


class Handler(logging.StreamHandler):
    def __init__(self, pipe):
        logging.StreamHandler.__init__(self)
        self.pipe = pipe

    def emit(self, record):
        record = f'{record.message}'
        self.pipe.send({'type':'log','data':record})


```

这段代码定义了两个日志处理器，分别是chlr和ehlr。这两个处理器都是基于系统标准输出的流处理器，但是它们在输出形式和级别上有所不同。

chlr将输出格式设置为和一个颜色格式器，然后设置日志级别为INFO。chlr还添加了一个名为MaxFilter的过滤器，它会捕获所有日志级别为INFO的输出，并将这些输出传递给ehlr。同时，chlr还添加了一个名为PackagePathFilter的过滤器，它会捕获所有包含"/path/to/package"的输出，并将这些输出传递给ehlr。

ehlr将输出格式设置为和一个颜色格式器，然后设置日志级别为WARNING。ehlr还添加了一个名为PackagePathFilter的过滤器，它会捕获所有包含"/path/to/package"的输出，并将这些输出传递给logger。

最后，logger创建了一个日志实例，设置了一个DEBUG级别的级别，并将chlr和ehlr添加为它的两个处理器。这样做后，每个输出都会经过chlr和ehlr的过滤，然后输出到logger。


```
chlr = logging.StreamHandler(stream=sys.stdout)
chlr.setFormatter(color_formatter)
chlr.setLevel('INFO')
chlr.addFilter(MaxFilter(logging.INFO))
chlr.addFilter(PackagePathFilter())

ehlr = logging.StreamHandler(stream=sys.stderr)
ehlr.setFormatter(color_formatter)
ehlr.setLevel('WARNING')
ehlr.addFilter(PackagePathFilter())

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
logger.addHandler(chlr)
logger.addHandler(ehlr)


```

这段代码是一个函数 `init_fhlr`，它用于初始化一个日志文件。函数有两个参数，一个是 `pipe`，它是传递给函数的第二个参数，另一个是 `None`，表示函数可以不使用传递的管道。

函数首先检查 `config.LOGFILE_PATH` 是否为空。如果是，函数不做任何操作，直接返回。如果 `config.LOGFILE_PATH` 不是空，函数会创建一个文件夹并设置为初始目录，以便创建日志文件。

接下来，函数创建一个名为 `runtime.log` 的文件，并使用 `RotatingFileHandler` 类来写入日志文件。 `RotatingFileHandler` 类是一种用于定期将文件内容旋转并清除空间以便重新写入的文件处理器。在这里，文件被写入到一个名为 `runtime.log` 的文件中，文件大小为 10KB，最大文件大小为 1GB，备份文件数量为 `config.LOGFILE_AMOUNT`，以便在达到文件大小上限时备份和滚动。

接下来，函数设置日志文件的格式和使用 `basic_formatter` 作为格式化字符串。 `basic_formatter` 是一个简单的字符串格式化器，它将时间戳和日志级别组合在一起。

函数还设置日志文件的级别为 `DEBUG`，以确保记录下来的日志信息具有更高的详细程度。

最后，函数还添加了一个 `PackagePathFilter` 过滤器，用于确保在写入日志时只记录特定的包的运行时日志。

函数最后通过 `logger.addHandler` 方法将创建的 `RotatingFileHandler` 和过滤器添加到正在运行的日志应用程序中。如果传递给函数的 `pipe` 参数为 `None`，则函数将直接写入日志文件而不是使用管道流来写入日志。


```
def init_fhlr(pipe=None) -> None:
    """ initialize log file """
    if config.LOGFILE_PATH is None:
        return
    folder = Path(config.LOGFILE_PATH)
    folder.mkdir(exist_ok=True, parents=True)
    fhlr = RotatingFileHandler(
        folder.joinpath('runtime.log'),
        encoding='utf8',
        maxBytes=10 * 1024 * 1024,
        backupCount=config.LOGFILE_AMOUNT,
    )
    fhlr.setFormatter(basic_formatter)
    fhlr.setLevel('DEBUG')
    fhlr.addFilter(PackagePathFilter())
    logger.addHandler(fhlr)
    if pipe is not None:
        wh = Handler(pipe)
        wh.setLevel(logging.INFO)
        logger.addHandler(wh)


```

这段代码定义了两个函数，分别是 `set_debug_mode()` 和 `save_screenshot()`。

1. `set_debug_mode()` 函数用于设置调试模式。函数的原型为 `def set_debug_mode() -> None:`，表示返回一个 None 类型的函数。函数内部先判断 `config.DEBUG_MODE` 是否为真，如果是，则执行一系列操作，包括初始化 FHLR 对象、将日志存储到指定的日志文件等等。最后，函数内部创建一个 `config.LOGFILE_PATH` 指向的文件，并将其路径添加到 `config.DEBUG_MODE` 的设置中。

2. `save_screenshot()` 函数用于保存 screenshot。函数的原型为 `def save_screenshot(img: bytes, subdir: str = '') -> None:`，表示接收一个 bytes 类型的图像数据（即 screenshot）和一个子目录参数（可选，默认为当前工作目录）。函数内部首先检查 `config.SCREENSHOT_PATH` 是否已经被创建，如果没有，则创建一个子目录并创建一个保存 screenshot 的文件夹。接着，函数创建一个文件夹，并将 `config.SCREENSHOT_PATH` 和 `subdir` 作为参数传递给文件夹创建操作。然后，函数使用 `list()` 方法遍历 `config.SCREENSHOT_PATH` 目录下的所有文件，并将它们按名称排序。接下来，如果 `subdir` 参数中文件数量超过 `config.SCREENSHOT_MAXNUM`，函数会将列表的前 `config.SCREENSHOT_MAXNUM` 个文件从列表中删除。然后，函数使用 `time.strftime()` 方法创建一个文件名，并使用 `with` 语句打开一个写入文件，将 `img` 数据写入文件中。最后，函数输出保存 screenshot 的文件名。


```
def set_debug_mode() -> None:
    """ set debud mode on """
    if config.DEBUG_MODE:
        logger.info(f'Start debug mode, log is stored in {config.LOGFILE_PATH}')
        init_fhlr()


def save_screenshot(img: bytes, subdir: str = '') -> None:
    """ save screenshot """
    if config.SCREENSHOT_PATH is None:
        return
    folder = Path(config.SCREENSHOT_PATH).joinpath(subdir)
    folder.mkdir(exist_ok=True, parents=True)
    if subdir != '-1' and len(list(folder.iterdir())) > config.SCREENSHOT_MAXNUM:
        screenshots = list(folder.iterdir())
        screenshots = sorted(screenshots, key=lambda x: x.name)
        for x in screenshots[: -config.SCREENSHOT_MAXNUM]:
            logger.debug(f'remove screenshot: {x.name}')
            x.unlink()
    filename = time.strftime('%Y%m%d%H%M%S.png', time.localtime())
    with folder.joinpath(filename).open('wb') as f:
        f.write(img)
    logger.debug(f'save screenshot: {filename}')


```

这段代码定义了一个名为 "log_sync" 的类，该类继承自 Python 标准库中的 "threading.Thread" 类。

这个类的实现中包含一个 "__init__" 方法和一个 "__del__" 方法，以及一个 "run" 方法。

"__init__" 方法接收两个参数，一个是包含要运行的进程名，另一个是包含从子进程中接收数据的管道号。在方法中，创建了两个变量，一个是 self.process，另一个是 self.pipe，然后使用 `super().__init__(daemon=True)` 来调用父类的 "__init__" 方法并设置 self.process 为运行进程的名称，self.pipe 为管道号。

"__del__" 方法中，使用 self.pipe.close() 来关闭管道并释放资源。

"run" 方法中包含一个无限循环，该循环从管道中读取一行数据并打印到日誌文件中。在循环的每次迭代中，使用 self.pipe.readline().strip() 方法读取一行的数据，然后使用 Python 标准库中的 "logger.debug" 函数来打印该数据并包含一个带有 self.process 变量的字符串。


```
class log_sync(threading.Thread):
    """ recv output from subprocess """

    def __init__(self, process: str, pipe: int) -> None:
        self.process = process
        self.pipe = os.fdopen(pipe)
        super().__init__(daemon=True)

    def __del__(self) -> None:
        self.pipe.close()

    def run(self) -> None:
        while True:
            line = self.pipe.readline().strip()
            logger.debug(f'{self.process}: {line}')



```

# `/opt/arknights-mower/arknights_mower/utils/matcher.py`

这段代码是一个带有未来时注语的函数，它表示当前导入的函数都使用未来时。接着，它导入了pickle模块、traceback模块和typing模块，这三个模块用于异步操作、模块导入和类型声明。

接着，它导入了cv2模块、numpy模块和sklearn模块，这三个模块用于图像处理、数值计算和机器学习。

接下来，它定义了一个名为"image"的函数，这个函数通过cv2模块的cv2.imread()函数读取一张图片，然后使用sklearn中的create_dataset()函数将图片转换成一个数据集。

紧接着，它定义了一个名为"structural_similarity"的函数，这个函数使用skimage中的compare_ssim()函数计算两个图像之间的结构相似性。

最后，定义了一个名为"export_images"的函数，这个函数将上面定义的"image"和"structural_similarity"函数的结果存储到两个变量中，然后使用pickle模块将结果存储到硬盘上。


```
from __future__ import annotations

import pickle
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

from .. import __rootdir__
from . import typealias as tp
from .image import cropimg
```

这段代码的作用是实现一个图像特征匹配和分割的算法。

首先，从 .log 文件中导入了一个logger，用于在匹配过程中输出调试信息。

然后，定义了一个匹配器_debug，表示在匹配时输出调试信息，但不会对匹配结果产生影响。

接着，定义了一个FLANN_INDEX_KDTREE，表示要使用的FLANN库中用于构建索引的层级树节点数。

然后，定义了一个GOOD_DISTANCE_LIMIT，表示用于SIFT算法中目标检测的距离上限，用于设置分割阈值以避免分割过于密集。

接下来，从 ./log 目录下创建了一个名为 models/svm.model 的模型文件，并使用 pickle 库读取该文件中的内容，以便将其用于SIFT算法的训练和预测。

接着，定义了一个名为 getHash 的函数，用于计算给定数据列表的哈希值，以便用于FLANN库中的 SIFT 算法。

最后，实现了 FLANN 算法中的 SIFT 算法，以及用于图像特征匹配和分割的函数。


```
from .log import logger

MATCHER_DEBUG = False
FLANN_INDEX_KDTREE = 0
GOOD_DISTANCE_LIMIT = 0.7
SIFT = cv2.SIFT_create()
with open(f'{__rootdir__}/models/svm.model', 'rb') as f:
    SVC = pickle.loads(f.read())


def getHash(data: list[float]) -> tp.Hash:
    """ calc image hash """
    avreage = np.mean(data)
    return np.where(data > avreage, 1, 0)


```

This is a Python function that appears to perform image matching and analysis on KAKAI退格数据集中的训练样本。其接受四个参数：query（查询图像，2D张量，可以是坐标或者张量，这里以张量形式接收），rect（矩形区域，可以是已知的矩形区域或者是用户输入的矩形区域，以numpy数组形式接收），orientation（orientation，指的是对象的朝向，可以是北东、南西、北南、东南西北中的一个或者多个，以字符串形式接收），score（得分，可以是integer、float或者double，以float形式接收）。函数返回四个结果：good_matches_rate（正确匹配的样本数占总样本数的比例，可以是float或者integer，以float形式接收），good_area_rate（正确匹配的矩形区域面积与总面积的比率，可以是float或者integer，以float形式接收），hash（两个图像的相似度，以float形式接收），ssim（两个图像的结构相似度，以float形式接收，这里采用multichannel=True的方式计算），以及根据函数计算出来的其他得分。函数使用了OpenCV的cropimg、cv2.resize、plt.imshow、compare_ssim等函数，以及AdaHash算法。


```
def hammingDistance(hash1: tp.Hash, hash2: tp.Hash) -> int:
    """ calc Hamming distance between two hash """
    return np.count_nonzero(hash1 != hash2)


def aHash(img1: tp.GrayImage, img2: tp.GrayImage) -> int:
    """ calc image hash """
    data1 = cv2.resize(img1, (8, 8)).flatten()
    data2 = cv2.resize(img2, (8, 8)).flatten()
    hash1 = getHash(data1)
    hash2 = getHash(data2)
    return hammingDistance(hash1, hash2)


class Matcher(object):
    """ image matching module """

    def __init__(self, origin: tp.GrayImage) -> None:
        logger.debug(f'Matcher init: shape ({origin.shape})')
        self.origin = origin
        self.init_sift()

    def init_sift(self) -> None:
        """ get SIFT feature points """
        self.kp, self.des = SIFT.detectAndCompute(self.origin, None)

    def match(self, query: tp.GrayImage, draw: bool = False, scope: tp.Scope = None, judge: bool = True,prescore = 0.0) -> Optional(tp.Scope):
        """ check if the image can be matched """
        rect_score = self.score(query, draw, scope)  # get matching score
        if rect_score is None:
            return None  # failed in matching
        else:
            rect, score = rect_score

        if prescore != 0.0 and score[3] >= prescore:
            logger.debug(f'match success: {score}')
            return rect
        # use SVC to determine if the score falls within the legal range
        if judge and not SVC.predict([score])[0]:  # numpy.bool_
            logger.debug(f'match fail: {score}')
            return None  # failed in matching
        else:
            if prescore>0 and score[3]<prescore:
                logger.debug(f'score is not greater than {prescore}: {score}')
                return None
            logger.debug(f'match success: {score}')
            return rect  # success in matching

    def score(self, query: tp.GrayImage, draw: bool = False, scope: tp.Scope = None, only_score: bool = False) -> Optional(Tuple[tp.Scope, tp.Score]):
        """ scoring of image matching """
        try:
            # if feature points is empty
            if self.des is None:
                logger.debug('feature points is None')
                return None

            # specify the crop scope
            if scope is not None:
                ori_kp, ori_des = [], []
                for _kp, _des in zip(self.kp, self.des):
                    if scope[0][0] <= _kp.pt[0] and scope[0][1] <= _kp.pt[1] and _kp.pt[0] <= scope[1][0] and _kp.pt[1] <= scope[1][1]:
                        ori_kp.append(_kp)
                        ori_des.append(_des)
                logger.debug(
                    f'match crop: {scope}, {len(self.kp)} -> {len(ori_kp)}')
                ori_kp, ori_des = np.array(ori_kp), np.array(ori_des)
            else:
                ori_kp, ori_des = self.kp, self.des

            # if feature points is less than 2
            if len(ori_kp) < 2:
                logger.debug('feature points is less than 2')
                return None

            # the height & width of query image
            h, w = query.shape

            # the feature point of query image
            qry_kp, qry_des = SIFT.detectAndCompute(query, None)

            # build FlannBasedMatcher
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(qry_des, ori_des, k=2)

            # store all the good matches as per Lowe's ratio test
            good = []
            for x, y in matches:
                if x.distance < GOOD_DISTANCE_LIMIT * y.distance:
                    good.append(x)
            good_matches_rate = len(good) / len(qry_des)

            # draw all the good matches, for debug
            if draw:
                result = cv2.drawMatches(
                    query, qry_kp, self.origin, ori_kp, good, None)
                plt.imshow(result, 'gray')
                plt.show()

            # if the number of good matches no more than 4
            if len(good) <= 4:
                logger.debug(
                    f'not enough good matches are found: {len(good)} / {len(qry_des)}')
                return None

            # get the coordinates of good matches
            qry_pts = np.float32(
                [qry_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            ori_pts = np.float32(
                [ori_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # calculated transformation matrix and the mask
            M, mask = cv2.findHomography(qry_pts, ori_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # if transformation matrix is None
            if M is None:
                logger.debug('calculated transformation matrix failed')
                return None

            # calc the location of the query image
            quad = np.float32([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
            quad = cv2.perspectiveTransform(quad, M)  # quadrangle
            quad_points = qp = np.int32(quad).reshape(4, 2).tolist()

            # draw the result, for debug
            if draw or MATCHER_DEBUG:
                cv2.polylines(self.origin, [np.int32(quad)],
                              True, 0, 2, cv2.LINE_AA)
                draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                                   matchesMask=matchesMask, flags=2)
                result = cv2.drawMatches(query, qry_kp, self.origin, ori_kp,
                                         good, None, **draw_params)
                plt.imshow(result, 'gray')
                plt.show()

            # if quadrangle is not rectangle
            if max(abs(qp[0][0] - qp[1][0]), abs(qp[2][0] - qp[3][0]), abs(qp[0][1] - qp[3][1]), abs(qp[1][1] - qp[2][1])) > 30:
                logger.debug(f'square is not rectangle: {qp}')
                return None

            # make quadrangle rectangle
            rect = [(np.min(quad[:, 0, 0]), np.min(quad[:, 0, 1])),
                    (np.max(quad[:, 0, 0]), np.max(quad[:, 0, 1]))]

            # if rectangle is too small
            if rect[1][0] - rect[0][0] < 10 or rect[1][1] - rect[0][1] < 10:
                logger.debug(f'rectangle is too small: {rect}')
                return None

            # measure the rate of good match within the rectangle (x-axis)
            better = filter(
                lambda m:
                    rect[0][0] < ori_kp[m.trainIdx].pt[0] < rect[1][0] and rect[0][1] < ori_kp[m.trainIdx].pt[1] < rect[1][1], good)
            better_kp_x = [qry_kp[m.queryIdx].pt[0] for m in better]
            if len(better_kp_x):
                good_area_rate = np.ptp(better_kp_x) / w
            else:
                good_area_rate = 0

            # rectangle: float -> int
            rect = np.array(rect, dtype=int).tolist()
            rect_img = cropimg(self.origin, rect)

            # if rect_img is too small
            if np.min(rect_img.shape) < 10:
                logger.debug(f'rect_img is too small: {rect_img.shape}')
                return None

            # transpose rect_img
            rect_img = cv2.resize(rect_img, query.shape[::-1])

            # draw the result
            if draw or MATCHER_DEBUG:
                plt.subplot(1, 2, 1)
                plt.imshow(query, 'gray')
                plt.subplot(1, 2, 2)
                plt.imshow(rect_img, 'gray')
                plt.show()

            # calc aHash between query image and rect_img
            hash = 1 - (aHash(query, rect_img) / 32)

            # calc ssim between query image and rect_img
            ssim = compare_ssim(query, rect_img, multichannel=True)

            # return final rectangle and four dimensions of scoring
            if only_score:
                return (good_matches_rate, good_area_rate, hash, ssim)
            else:
                return rect, (good_matches_rate, good_area_rate, hash, ssim)

        except Exception as e:
            logger.error(e)
            logger.debug(traceback.format_exc())

```

# `/opt/arknights-mower/arknights_mower/utils/operators.py`



This is a Python implementation of the司降服务异常处理系统，用于处理宿舍房间安排中的异常情况。

首先，我们需要定义一些类和函数：

- `Country` 类，用于管理国家。
- `City` 类，用于管理城市。
- `CountryCity` 类，用于管理国家-城市组合。
- `Dorm` 类，用于表示宿舍房间。
- `Operator` 类，用于表示优先级。
- `Priority` 类，用于表示优先级。
- `AssignDorm` 函数，用于分配宿舍房间。
- `Print` 函数，用于打印输出。

接下来，我们可以编写一些方法来处理宿舍房间安排中的异常情况。例如，我们可以编写一个方法来检查分配宿舍房间时是否会产生满休息室的异常情况：

python
class中国家：
   def __init__(self, max_resting_count):
       self.max_resting_count = max_resting_count

   def assign_country(self, name):
       pass  # 此处编写代码，分配国家

class消化道：
   def __init__(self):
       self.name = ''
       self.time = None

   def __repr__(self):
       return f"Country: {self.name}, Time: {self.time}"


class 房间：
   def __init__(self, name):
       self.name = name
       self.time = None

   def __repr__(self):
       return f"Room: {self.name}, Time: {self.time}"


class Priority:
   def __init__(self):
       self.name = ''
       self.resting_priority = ''

   def is_high(self):
       return self.resting_priority == 'high'


class Admin:
   def __init__(self, max_resting_count):
       self.max_resting_count = max_resting_count


class RestingCount:
   def __init__(self):
       self.count = 0


class AssignDorm:
   def __init__(self, dorm, max_resting_count):
       self.dorm = dorm
       self.max_resting_count = max_resting_count

   def assign(self):
       if self.max_resting_count <= self.max_resting_count:
           pass  # 如果没有满休息室，直接返回
       else:
           min_resting_count = self.max_resting_count - 1
           if self.dorm[min_resting_count] is None or self.dorm[min_resting_count].time is None:
               self.dorm[min_resting_count] =下一个未被分配的宿舍房间
               self.max_resting_count = min_resting_count
           else:
               self.max_resting_count = min_resting_count
               self.dorm[min_resting_count].time = 


def print_dorms(dorms):
   for dorm in dorms:
       print(f"{dorm.name}: {dorm.time}")


def assign_dorm(name, max_resting_count):
   dorm = 新的宿舍房间(name)
   try:
       dorm.time = datetime.datetime.now()
       ret = 0
       while ret < max_resting_count:
           for idx in range(max_resting_count - 1, -1, -1):
               # 检查指定的宿舍房间是否为空
               if idx == 0 or idx == max_resting_count:
                   pass
               else:
                   dorm = 目标宿舍房间(dorm)
                   if idx < len(dorms) and dorms[idx].name != dorm.name:
                       break
                   else:
                       ret += 1
                       break
                   if idx < max_resting_count:
                       dorms.pop(idx)
                       break
                   else:
                       break
           if idx < max_resting_count:
                       break
           else:
               break
       dorm.time = None
       ret += 1
       print_dorms(dorms)
   except Exception as e:
       print(f"Error: {e}")


上述代码实现了上述代码的主要功能。如果分配宿舍房间时，有满休息室的情况，


```
from datetime import datetime, timedelta
from ..data import agent_list
from ..solvers.record import save_action_to_sqlite_decorator
from ..utils.log import logger

class Operators(object):
    config = None
    operators = None
    exhaust_agent = []
    exhaust_group = []
    groups = None
    dorm = []
    max_resting_count = 4
    plan = None

    def __init__(self, config, max_resting_count, plan):
        self.config = config
        self.operators = {}
        self.groups = {}
        self.exhaust_agent = []
        self.exhaust_group = []
        self.dorm = []
        self.max_resting_count = max_resting_count
        self.workaholic_agent = []
        self.plan = plan
        self.run_order_rooms = {}
        self.clues = []

    def __repr__(self):
        return f'Operators(operators={self.operators})'

    def init_and_validate(self):
        for room in self.plan.keys():
            for idx, data in enumerate(self.plan[room]):
                if data["agent"] not in agent_list and data['agent'] != 'Free':
                    return f'干员名输入错误: 房间->{room}, 干员->{data["agent"]}'
                if data["agent"] in ['龙舌兰', '但书']:
                    return f'高效组不可用龙舌兰，但书 房间->{room}, 干员->{data["agent"]}'
                if data["agent"] == '菲亚梅塔' and idx == 1:
                    return f'菲亚梅塔不能安排在2号位置 房间->{room}, 干员->{data["agent"]}'
                if data["agent"] == 'Free' and not room.startswith('dorm'):
                    return f'Free只能安排在宿舍 房间->{room}, 干员->{data["agent"]}'
                if data["agent"] in self.operators and data["agent"] != "Free":
                    return f'高效组干员不可重复 房间->{room},{self.operators[data["agent"]].room}, 干员->{data["agent"]}'
                self.add(Operator(data["agent"], room, idx, data["group"], data["replacement"], 'high',
                                  operator_type="high"))
        missing_replacements = []
        for room in self.plan.keys():
            if room.startswith("dorm") and len(self.plan[room]) != 5:
                return f'宿舍 {room} 人数少于5人'
            for idx, data in enumerate(self.plan[room]):
                # 菲亚梅塔替换组做特例判断
                if "龙舌兰" in data["replacement"] and "但书" in data["replacement"]:
                    return f'替换组不可同时安排龙舌兰和但书 房间->{room}, 干员->{data["agent"]}'
                if "菲亚梅塔" in data["replacement"]:
                    return f'替换组不可安排菲亚梅塔 房间->{room}, 干员->{data["agent"]}'
                r_count = len(data["replacement"])
                if "龙舌兰" in data["replacement"] or "但书" in data["replacement"]:
                    r_count -= 1
                if r_count <= 0 and data['agent'] != 'Free' and (not room.startswith("dorm")):
                    missing_replacements.append(data["agent"])
                for _replacement in data["replacement"]:
                    if _replacement not in agent_list and data['agent'] != 'Free':
                        return f'干员名输入错误: 房间->{room}, 干员->{_replacement}'
                    if data["agent"] != '菲亚梅塔':
                        # 普通替换
                        if _replacement in self.operators and self.operators[_replacement].is_high():
                            return f'替换组不可用高效组干员: 房间->{room}, 干员->{_replacement}'
                        self.add(Operator(_replacement, ""))
                    else:
                        if _replacement not in self.operators:
                            return f'菲亚梅塔替换不在高效组列: 房间->{room}, 干员->{_replacement}'
                        if _replacement in self.operators and not self.operators[_replacement].is_high():
                            return f'菲亚梅塔替换只能高效组干员: 房间->{room}, 干员->{_replacement}'
                        if _replacement in self.operators and self.operators[_replacement].group != '':
                            return f'菲亚梅塔替换不可分组: 房间->{room}, 干员->{_replacement}'
        # 判定替换缺失
        if "菲亚梅塔" in missing_replacements:
            return f'菲亚梅塔替换缺失'
        if '菲亚梅塔' in self.operators:
            for _agent in missing_replacements[:]:
                if _agent in self.operators['菲亚梅塔'].replacement[:-1]:
                    missing_replacements.remove(_agent)
        if len(missing_replacements):
            return f'以下干员替换组缺失：{",".join(missing_replacements)}'
        dorm_names = [k for k in self.plan.keys() if k.startswith("dorm")]
        dorm_names.sort(key=lambda d: d, reverse=False)
        added = []
        # 竖向遍历出效率高到低
        for dorm in dorm_names:
            free_found = False
            for _idx, _dorm in enumerate(self.plan[dorm]):
                if _dorm['agent'] == 'Free' and _idx <= 1:
                    return f'宿舍必须安排2个宿管'
                if _dorm['agent'] != 'Free' and free_found:
                    return f'Free必须连续且安排在宿管后'
                if _dorm['agent'] == 'Free' and not free_found and (dorm + str(_idx)) not in added and len(
                        added) < self.max_resting_count:
                    self.dorm.append(Dormitory((dorm, _idx)))
                    added.append(dorm + str(_idx))
                    free_found = True
                    continue
            if not free_found:
                return f'宿舍必须安排至少一个Free'
        # VIP休息位用完后横向遍历
        for dorm in dorm_names:
            for _idx, _dorm in enumerate(self.plan[dorm]):
                if _dorm['agent'] == 'Free' and (dorm + str(_idx)) not in added:
                    self.dorm.append(Dormitory((dorm, _idx)))
                    added.append(dorm + str(_idx))
        if len(self.dorm) < self.max_resting_count:
            return f'宿舍Free总数 {len(self.dorm)}小于最大分组数 {self.max_resting_count}'
        # low_free 的排序
        # self.dorm[self.max_resting_count:len(self.dorm)] = sorted(
        #     self.dorm[self.max_resting_count:len(self.dorm)],
        #     key=lambda k: (k.position[0], k.position[1]), reverse=True)
        # 跑单
        for x, y in self.plan.items():
            if not x.startswith('room'): continue
            if any(('但书' in obj['replacement'] or '龙舌兰' in obj['replacement']) for obj in y):
                self.run_order_rooms[x] = {}
        # 判定分组排班可能性
        current_high = self.available_free()
        current_low = self.available_free('low')
        for key in self.groups:
            high_count = 0
            low_count = 0
            _replacement = []
            for name in self.groups[key]:
                _candidate = next(
                    (r for r in self.operators[name].replacement if r not in _replacement and r not in ['龙舌兰', '但书']),
                    None)
                if _candidate is None:
                    return f'{key} 分组无法排班,替换组数量不够'
                else:
                    _replacement.append(_candidate)
                if self.operators[name].resting_priority == 'high':
                    high_count += 1
                else:
                    low_count += 1
            if high_count > current_high or low_count > current_low:
                return f'{key} 分组无法排班,宿舍可用高优先{current_high},低优先{current_low}->分组需要高优先{high_count},低优先{low_count}'

    def get_current_room(self, room, bypass=False, current_index=None):
        room_data = {v.current_index: v for k, v in self.operators.items() if v.current_room == room}
        res = [obj['agent'] for obj in self.plan[room]]
        not_found = False
        for idx, op in enumerate(res):
            if idx in room_data:
                res[idx] = room_data[idx].name
            else:
                res[idx] = ''
                if current_index is not None and idx not in current_index:
                    continue
                not_found = True
        if not_found and not bypass:
            return None
        else:
            return res

    def predict_fia(self, operators, fia_mood, hours=240):
        recover_hours = (24 - fia_mood) / 2
        for agent in operators:
            agent.mood -= agent.depletion_rate * recover_hours
            if agent.mood < 0.0:
                return False
        if recover_hours >= hours or 0 < recover_hours < 1:
            return True
        operators.sort(key=lambda x: (x.mood - x.lower_limit) / (x.upper_limit - x.lower_limit), reverse=False)
        fia_mood = operators[0].mood
        operators[0].mood = 24
        return self.predict_fia(operators, fia_mood, hours - recover_hours)

    def reset_dorm_time(self):
        for name in self.operators.keys():
            agent = self.operators[name]
            if agent.room.startswith("dorm"):
                agent.time_stamp = None

    @save_action_to_sqlite_decorator
    def update_detail(self, name, mood, current_room, current_index, update_time=False):
        agent = self.operators[name]
        if update_time:
            if agent.time_stamp is not None and agent.mood > mood:
                agent.depletion_rate = (agent.mood - mood) * 3600 / (
                    (datetime.now() - agent.time_stamp).total_seconds())
            agent.time_stamp = datetime.now()
        # 如果移出宿舍，则清除对应宿舍数据 且重新记录高效组心情
        if agent.current_room.startswith('dorm') and not current_room.startswith('dorm') and agent.is_high():
            self.refresh_dorm_time(agent.current_room, agent.current_index, {'agent': ''})
            if update_time:
                self.time_stamp = datetime.now()
            else:
                self.time_stamp = None
            agent.depletion_rate = 0
        if self.get_dorm_by_name(name)[0] is not None and not current_room.startswith('dorm') and agent.is_high():
            _dorm = self.get_dorm_by_name(name)[1]
            _dorm.name = ''
            _dorm.time = None
        agent.current_room = current_room
        agent.current_index = current_index
        agent.mood = mood
        # 如果是高效组且没有记录时间，则返还index
        if agent.current_room.startswith('dorm') and agent.is_high():
            for dorm in self.dorm:
                if dorm.position[0] == current_room and dorm.position[1] == current_index and dorm.time is None:
                    return current_index
        if agent.name == "菲亚梅塔" and (
                self.operators["菲亚梅塔"].time_stamp is None or self.operators["菲亚梅塔"].time_stamp < datetime.now()):
            return current_index

    def refresh_dorm_time(self, room, index, agent):
        for idx, dorm in enumerate(self.dorm):
            # Filter out resting priority low
            # if idx >= self.max_resting_count:
            #     break
            if dorm.position[0] == room and dorm.position[1] == index:
                # 如果人为高效组，则记录时间
                _name = agent['agent']
                if _name in self.operators.keys() and self.operators[_name].is_high():
                    dorm.name = _name
                    _agent = self.operators[_name]
                    # 如果干员有心情上限，则按比例修改休息时间
                    if _agent.mood != 24:
                        sec_remaining = (_agent.upper_limit - _agent.mood) * (
                            (agent['time'] - _agent.time_stamp).total_seconds()) / (24 - _agent.mood)
                        dorm.time = _agent.time_stamp + timedelta(seconds=sec_remaining)
                    else:
                        dorm.time = agent['time']
                else:
                    dorm.name = ''
                    dorm.time = None
                break

    def correct_dorm(self):
        for idx, dorm in enumerate(self.dorm):
            if dorm.name != "" and dorm.name in self.operators.keys():
                op = self.operators[dorm.name]
                if not (dorm.position[0] == op.current_room and dorm.position[1] == op.current_index):
                    self.dorm[idx].name = ""
                    self.dorm[idx].time = None

    def get_refresh_index(self, room, plan):
        ret = []
        for idx, dorm in enumerate(self.dorm):
            # Filter out resting priority low
            if idx >= self.max_resting_count:
                break
            if dorm.position[0] == room:
                for i, _name in enumerate(plan):
                    if _name in self.operators.keys() and self.operators[_name].is_high() and self.operators[
                        _name].resting_priority == 'high' and not self.operators[_name].room.startswith('dorm'):
                        ret.append(i)
                break
        return ret

    def get_dorm_by_name(self, name):
        for idx, dorm in enumerate(self.dorm):
            if dorm.name == name:
                return idx, dorm
        return None, None

    def add(self, operator):
        if operator.name not in agent_list:
            return
        if operator.name in self.config.keys() and 'RestingPriority' in self.config[operator.name].keys():
            operator.resting_priority = self.config[operator.name]['RestingPriority']
        if operator.name in self.config.keys() and 'ExhaustRequire' in self.config[operator.name].keys():
            operator.exhaust_require = self.config[operator.name]['ExhaustRequire']
        if operator.name in self.config.keys() and 'RestInFull' in self.config[operator.name].keys():
            operator.rest_in_full = self.config[operator.name]['RestInFull']
        if operator.name in self.config.keys() and 'LowerLimit' in self.config[operator.name].keys():
            operator.lower_limit = self.config[operator.name]['LowerLimit']
        if operator.name in self.config.keys() and 'UpperLimit' in self.config[operator.name].keys():
            operator.upper_limit = self.config[operator.name]['UpperLimit']
        if operator.name in self.config.keys() and 'Workaholic' in self.config[operator.name].keys():
            operator.workaholic = self.config[operator.name]['Workaholic']
        self.operators[operator.name] = operator
        # 需要用尽心情干员逻辑
        if (operator.exhaust_require or operator.group in self.exhaust_group) \
                and operator.name not in self.exhaust_agent:
            self.exhaust_agent.append(operator.name)
            if operator.group != '':
                self.exhaust_group.append(operator.group)
        # 干员分组逻辑
        if operator.group != "":
            if operator.group not in self.groups.keys():
                self.groups[operator.group] = [operator.name]
            else:
                self.groups[operator.group].append(operator.name)
        if operator.workaholic and operator.name not in self.workaholic_agent:
            self.workaholic_agent.append(operator.name)

    def available_free(self, free_type='high'):
        ret = 0
        if free_type == 'high':
            idx = 0
            for dorm in self.dorm:
                if dorm.name == '' or (dorm.name in self.operators.keys() and not self.operators[dorm.name].is_high()):
                    ret += 1
                elif dorm.time is not None and dorm.time < datetime.now():
                    logger.info("检测到房间休息完毕，释放Free位")
                    dorm.name = ''
                    ret += 1
                if idx == self.max_resting_count - 1:
                    break
                else:
                    idx += 1
        else:
            idx = self.max_resting_count
            for i in range(idx, len(self.dorm)):
                dorm = self.dorm[i]
                # 释放满休息位
                # TODO 高效组且低优先可以相互替换
                if dorm.name == '' or (dorm.name in self.operators.keys() and not self.operators[dorm.name].is_high()):
                    dorm.name = ''
                    ret += 1
                elif dorm.time is not None and dorm.time < datetime.now():
                    logger.info("检测到房间休息完毕，释放Free位")
                    dorm.name = ''
                    ret += 1
        return ret

    def assign_dorm(self, name):
        is_high = self.operators[name].resting_priority == 'high'
        if is_high:
            _room = next(obj for obj in self.dorm if
                         obj.name not in self.operators.keys() or not self.operators[obj.name].is_high())
        else:
            _room = None
            for i in range(self.max_resting_count, len(self.dorm)):
                if self.dorm[i].name == '':
                    _room = self.dorm[i]
                    break
        _room.name = name
        return _room

    def print(self):
        ret = "{"
        op = []
        dorm = []
        for k, v in self.operators.items():
            op.append("'" + k + "': " + str(vars(v)))
        ret += "'operators': {" + ','.join(op) + "},"
        for v in self.dorm:
            dorm.append(str(vars(v)))
        ret += "'dorms': [" + ','.join(dorm) + "]}"
        return ret


```

This is a class called `Operator` that represents a group of students in a dormitory. It has several attributes including the student's name, the room they are in, their group number, their replacement, their resting priority, their current room, their exhaust require, their mood, their upper limit, their lower limit, their operator type, their depletion rate, and their time stamp.

It also has two methods, `__init__` and `__repr__`, that are used for initialization and representation.

The `__init__` method is used to initialize the attributes of the `Operator` object. It takes in the student's name, the room they are in, their group number, their replacement, their resting priority, their current room, their exhaust require, their mood, their upper limit, their lower limit, their operator type, their depletion rate, and their time stamp.

The `__repr__` method is used for representation and is generated by `repr()` function. It returns a string that summarizes the attributes of the `Operator` object.

You can use this `Operator` class to check if a student is in a certain room, if they are not, you can set the room to be in the dorm, and you can also set the mood of the student, among other things.


```
class Dormitory(object):

    def __init__(self, position, name='', time=None):
        self.position = position
        self.name = name
        self.time = time

    def __repr__(self):
        return f"Dormitory(position={self.position},name='{self.name}',time='{self.time}')"


class Operator(object):
    time_stamp = None
    depletion_rate = 0
    workaholic = False

    def __init__(self, name, room, index=-1, group='', replacement=[], resting_priority='low', current_room='',
                 exhaust_require=False,
                 mood=24, upper_limit=24, rest_in_full=False, current_index=-1, lower_limit=0, operator_type="low",
                 depletion_rate=0, time_stamp=None):
        self.name = name
        self.room = room
        self.operator_type = operator_type
        self.index = index
        self.group = group
        self.replacement = replacement
        self.resting_priority = resting_priority
        self.current_room = current_room
        self.exhaust_require = exhaust_require
        self.upper_limit = upper_limit
        self.rest_in_full = rest_in_full
        self.mood = mood
        self.current_index = current_index
        self.lower_limit = lower_limit
        self.depletion_rate = depletion_rate
        self.time_stamp = time_stamp

    def is_high(self):
        return self.operator_type == 'high'

    def need_to_refresh(self, h=2, r=""):
        # 是否需要读取心情
        if self.operator_type == 'high':
            if self.time_stamp is None or (
                    self.time_stamp is not None and self.time_stamp + timedelta(hours=h) < datetime.now()) or (
                    r.startswith("dorm") and not self.room.startswith("dorm")):
                return True
        return False

    def not_valid(self):
        if self.workaholic:
            return False
        if self.operator_type == 'high':
            if not self.room.startswith("dorm") and self.current_room.startswith("dorm"):
                if self.mood == -1 or self.mood == 24:
                    return True
                else:
                    return False
            return self.need_to_refresh(2.5) or self.current_room != self.room or self.index != self.current_index
        return False

    def current_mood(self):
        predict = self.mood
        if self.time_stamp is not None:
            predict = self.mood - self.depletion_rate * (datetime.now() - self.time_stamp).total_seconds() / 3600
        if 0 <= predict <= 24:
            return predict
        else:
            return self.mood

    def __repr__(self):
        return f"Operator(name='{self.name}', room='{self.room}', index={self.index}, group='{self.group}', replacement={self.replacement}, resting_priority='{self.resting_priority}', current_room='{self.current_room}',exhaust_require={self.exhaust_require},mood={self.mood}, upper_limit={self.upper_limit}, rest_in_full={self.rest_in_full}, current_index={self.current_index}, lower_limit={self.lower_limit}, operator_type='{self.operator_type}',depletion_rate={self.depletion_rate},time_stamp='{self.time_stamp}')"

```