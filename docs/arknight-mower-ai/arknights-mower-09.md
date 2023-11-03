# ArknightMower源码解析 9

# `/opt/arknights-mower/arknights_mower/utils/param.py`

这段代码定义了一个名为`ParamError`的类，该类继承自`ValueError`类，用于表示在函数中参数值无效的情况。

该类包含一个参数`args`，是一个元组参数，用于传递给函数的不同部分的参数。在函数内部，使用一个字典`params`来存储已经识别出的参数。

`parse_operation_params`函数用于将传递给它的参数进行解析，并返回一个元组，其中包含参数的级别、时间戳、药方、起源和消除状态。

该函数首先对参数进行遍历，并检查参数中是否包含`-`操作符。如果是，就尝试从参数中提取出一个整数。如果是`r`操作符，就检查药方是否为零，如果是，就尝试从参数中提取出一个整数。如果是`R`操作符，就检查起源是否为零，如果是，就尝试从参数中提取出一个整数。如果是`e`操作符，就检查消除是否为零，如果是，就尝试从参数中提取出一个整数。如果是`E`操作符，就尝试从参数中提取出一个整数，并检查消除是否为零。

如果参数中包含其他操作符或没有有效的参数，就尝试从参数中提取出有效的参数，并将其存储到`params`字典中。如果发生异常，就通过`raise ParamError`方法抛出参数错误，并返回参数级别、时间戳、药方、起源和消除状态中的最低值。


```
from .typealias import ParamArgs


class ParamError(ValueError):
    """ 参数错误 """


def parse_operation_params(args: ParamArgs = []):
    level = None
    times = -1
    potion = 0
    originite = 0
    eliminate = 0

    try:
        for p in args:
            if p[0] == '-':
                val = -1
                if len(p) > 2:
                    val = int(p[2:])
                if p[1] == 'r':
                    assert potion == 0
                    potion = val
                elif p[1] == 'R':
                    assert originite == 0
                    originite = val
                elif p[1] == 'e':
                    assert eliminate == 0
                    eliminate = 1
                elif p[1] == 'E':
                    assert eliminate == 0
                    eliminate = 2
            elif p.find('-') == -1:
                assert times == -1
                times = int(p)
            else:
                assert level is None
                level = p
    except Exception:
        raise ParamError
    return level, times, potion, originite, eliminate


```

这段 Python 代码定义了一个名为 `operation_times` 的函数，它接受一个参数 `args`，这个参数是一个参数列表。函数内部使用了一个名为 `parse_operation_params` 的函数来解析参数列表中的参数，并将解析得到的参数存储在变量中，最后返回解析得到的参数数量。

具体来说，函数 `operation_times` 接受一个参数 `args`，这个参数是一个包含多个参数的列表。函数内部使用 `parse_operation_params` 函数来解析 `args` 参数列表中的每个参数，并将解析得到的参数存储在两个整数变量 `_` 和 `times` 中。解析完成后，函数内部创建了一个新的整数变量 `_`，该变量被赋值为 `len(args)`，这样 `_` 变量将存储参数列表的长度。最后，函数返回 `_`，即参数列表的长度。

这个函数的作用是获取传递给它的参数列表，并返回其中参数的数量。


```
def operation_times(args: ParamArgs = []) -> int:
    _, times, _, _, _ = parse_operation_params(args)
    return times

```

# `/opt/arknights-mower/arknights_mower/utils/pipe.py`

这段代码定义了一个名为`push_operators`的函数，它接受一个函数作为参数(函数内部定义了一个内部函数`func_dec`)，并将内部函数作为参数传入。

内部函数`func_dec`的作用是创建一个新的函数，这个新函数在接收参数`s`和`*args`之后，与原函数`func`执行相同的操作并返回结果。然后，它检查`Pipe`对象是否尚存(因为代码中使用了`if`语句)，如果`Pipe`尚存，则创建一个`Pipe.conn`对象，并发送一个包含`s.op_data.operators`的JSON数据到该连接，如果`Pipe`不存在，则创建一个新的`Pipe`对象并设置`conn`属性。最后，它返回新函数。

整个函数的作用是创建一个新的函数，允许将`func`中定义的操作符及其传递给新函数的参数一起传递给新函数。这个新函数可以在需要时被调用，无论`func`是否正在执行。


```
def push_operators(func):
    def func_dec(s,*args):
        r=func(s,*args)
        if Pipe is not None and Pipe.conn is not None:
            # print('发送数据')
            # print(s.op_data.operators)
            Pipe.conn.send({'type':'operators','data':s.op_data.operators})
        return r
    return func_dec


class Pipe:
    conn = None



```

# `/opt/arknights-mower/arknights_mower/utils/priority_queue.py`

这段代码定义了一个名为 PriorityQueue 的类，该类实现了一个基于 heapq 的优先队列。该类的构造函数创建了一个空队列，并提供了 push 和 pop 方法来添加元素到队列中或从队列中移除元素。

具体来说，push 方法将一个数据对象添加到队列的末尾，可以使用 heapq.heappush 函数实现，这个函数会按照堆升序的顺序将数据对象添加到队列中。pop 方法从队列的头部取出一个数据对象，并按照堆降序的顺序删除该数据对象，同样可以使用 heapq.heappop 函数实现，这个函数会从队列的头部取出一个数据对象并按照堆降序的顺序删除该数据对象。

该类的实例化方式是通过调用 PriorityQueue 的 constructor，并传入一些参数来创建一个 priority queue，例如：

python
pq = PriorityQueue()


这将创建一个名为 pq 的 priority queue，可以像下面这样使用该实例化出的 pq 对象：

python
pq.push(1)
pq.push(2)
pq.push(3)

print(pq.pop())  # 输出 2
print(pq.pop())  # 输出 3
print(pq.pop())  # 输出 1


这将输出 [1, 2, 3]，说明在 pq 中的优先级顺序是 3 > 2 > 1。


```
import heapq


class PriorityQueue(object):
    """
    基于 heapq 实现的优先队列
    """

    def __init__(self):
        self.queue = []

    def push(self, data):
        heapq.heappush(self.queue, data)

    def pop(self):
        if len(self.queue) == 0:
            return None
        return heapq.heappop(self.queue)

```

# `/opt/arknights-mower/arknights_mower/utils/recognize.py`

这段代码是一个Python程序，它从未来的趋势中导入了一个名为“annotations”的模块。然后，它导入了time模块、cv2.pycv2模块（用于处理计算机视觉中的图像）和一个名为“typing”的模块（用于定义可交换类型变量）。

接下来，它从名为“typing”的模块中导入了一个名为“List”的类型，然后从名为“Optional”的模块中导入了一个类型。

紧接着，它从名为“__rootdir__”的目录中导入了一个名为“config”的配置文件，然后从名为“detector”的目录中导入了一个名为“detector”的函数。

接下来，它从名为“tp”的命名空间中导入了一个名为“tp”的类型，然后从名为“Device”的目录中导入了一个名为“Device”的类。

接着，它从名为“bytes2img”的目录中导入了一个名为“bytes2img”的函数，然后从名为“cropimg”的目录中导入了一个名为“cropimg”的函数。

然后，它从名为“loadimg”的目录中导入了一个名为“loadimg”的函数，然后从名为“thres2”的目录中导入了一个名为“thres2”的函数。

接下来，它从名为“logger”的目录中导入了一个名为“logger”的函数，然后从名为“save_screenshot”的目录中导入了一个名为“save_screenshot”的函数。

最后，它从名为“Matcher”的目录中导入了一个名为“Matcher”的类，然后从函数开始定义了这些函数。


```
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .. import __rootdir__
from . import config, detector
from . import typealias as tp
from .device import Device
from .image import bytes2img, cropimg, loadimg, thres2
from .log import logger, save_screenshot
from .matcher import Matcher
```



This is a class called `Recognizer` which is used for image recognition. It has several methods including `find`, `score`, and `crop`.

The `find` method takes an image resource file name and attempts to find the specified element in the image using the `Matcher` class. It supports various thresholding methods to make the recognition process more accurate. The threshold值 can be specified to control the sensitivity of the recognition.

The `score` method takes an image resource file name and attempts to find the specified element in the image. It can also control the drawing output by setting `draw=True`. It takes the recognition image and the threshold value as inputs and returns the score of the recognition.

The `crop` method takes an image resource file name and a two-dimensional cropping scope (x, y). It crops the image to the specified area of the image. It can also be used to resize the image.

The `__init__` method sets the root directory of the image resources and defines the `find`, `score`, and `crop` methods. The `__rootdir__` attribute is a directive to tell Python to look for image resources in the specified directory.


```
from .scene import Scene, SceneComment


class RecognizeError(Exception):
    pass


class Recognizer(object):

    def __init__(self, device: Device, screencap: bytes = None) -> None:
        self.device = device
        self.start(screencap)

    def start(self, screencap: bytes = None, build: bool = True) -> None:
        """ init with screencap, build matcher  """
        retry_times = config.MAX_RETRYTIME
        while retry_times > 0:
            try:
                if screencap is not None:
                    self.screencap = screencap
                else:
                    self.screencap = self.device.screencap()
                self.img = bytes2img(self.screencap, False)
                self.gray = bytes2img(self.screencap, True)
                self.h, self.w, _ = self.img.shape
                self.matcher = Matcher(self.gray) if build else None
                self.scene = Scene.UNDEFINED
                return
            except cv2.error as e:
                logger.warning(e)
                retry_times -= 1
                time.sleep(1)
                continue
        raise RuntimeError('init Recognizer failed')

    def update(self, screencap: bytes = None, rebuild: bool = True) -> None:
        """ rebuild matcher """
        self.start(screencap, rebuild)

    def color(self, x: int, y: int) -> tp.Pixel:
        """ get the color of the pixel """
        return self.img[y][x]

    def save_screencap(self, folder):
        save_screenshot(self.screencap, subdir=f'{folder}/{self.h}x{self.w}')

    def get_scene(self) -> int:
        """ get the current scene in the game """
        if self.scene != Scene.UNDEFINED:
            return self.scene
        if self.find('connecting', scope=((self.w//2, self.h//10*8), (self.w//4*3, self.h))) is not None:
            self.scene = Scene.CONNECTING
        elif self.find('index_nav', thres=250, scope=((0, 0), (100+self.w//4, self.h//10))) is not None:
            self.scene = Scene.INDEX
        elif self.find('nav_index') is not None:
            self.scene = Scene.NAVIGATION_BAR
        elif self.find('login_new',score= 0.8) is not None:
            self.scene = Scene.LOGIN_NEW
        elif self.find('login_bilibili_new',score= 0.8) is not None:
            self.scene = Scene.LOGIN_NEW_B
        elif self.find('close_mine') is not None:
            self.scene = Scene.CLOSE_MINE
        elif self.find('check_in') is not None:
            self.scene = Scene.CHECK_IN
        elif self.find('materiel_ico') is not None:
            self.scene = Scene.MATERIEL
        elif self.find('read_mail') is not None:
            self.scene = Scene.MAIL
        elif self.find('loading') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading2') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading3') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading4') is not None:
            self.scene = Scene.LOADING
        elif self.is_black():
            self.scene = Scene.LOADING
        elif self.find('ope_plan') is not None:
            self.scene = Scene.OPERATOR_BEFORE
        elif self.find('ope_select_start') is not None:
            self.scene = Scene.OPERATOR_SELECT
        elif self.find('ope_agency_going') is not None:
            self.scene = Scene.OPERATOR_ONGOING
        elif self.find('ope_elimi_finished') is not None:
            self.scene = Scene.OPERATOR_ELIMINATE_FINISH
        elif self.find('ope_finish') is not None:
            self.scene = Scene.OPERATOR_FINISH
        elif self.find('ope_recover_potion_on') is not None:
            self.scene = Scene.OPERATOR_RECOVER_POTION
        elif self.find('ope_recover_originite_on') is not None:
            self.scene = Scene.OPERATOR_RECOVER_ORIGINITE
        elif self.find('double_confirm') is not None:
            if self.find('network_check') is not None:
                self.scene = Scene.NETWORK_CHECK
            else:
                self.scene = Scene.DOUBLE_CONFIRM
        elif self.find('ope_firstdrop') is not None:
            self.scene = Scene.OPERATOR_DROP
        elif self.find('ope_eliminate') is not None:
            self.scene = Scene.OPERATOR_ELIMINATE
        elif self.find('ope_elimi_agency_panel') is not None:
            self.scene = Scene.OPERATOR_ELIMINATE_AGENCY
        elif self.find('ope_giveup') is not None:
            self.scene = Scene.OPERATOR_GIVEUP
        elif self.find('ope_failed') is not None:
            self.scene = Scene.OPERATOR_FAILED
        elif self.find('friend_list_on') is not None:
            self.scene = Scene.FRIEND_LIST_ON
        elif self.find('credit_visiting') is not None:
            self.scene = Scene.FRIEND_VISITING
        elif self.find('infra_overview') is not None:
            self.scene = Scene.INFRA_MAIN
        elif self.find('infra_todo') is not None:
            self.scene = Scene.INFRA_TODOLIST
        elif self.find('clue') is not None:
            self.scene = Scene.INFRA_CONFIDENTIAL
        elif self.find('arrange_check_in') or self.find('arrange_check_in_on') is not None:
            self.scene = Scene.INFRA_DETAILS
        elif self.find('infra_overview_in') is not None:
            self.scene = Scene.INFRA_ARRANGE
        elif self.find('arrange_confirm') is not None:
            self.scene = Scene.INFRA_ARRANGE_CONFIRM
        elif self.find('friend_list') is not None:
            self.scene = Scene.FRIEND_LIST_OFF
        elif self.find("mission_trainee_on") is not None:
            self.scene = Scene.MISSION_TRAINEE
        elif self.find('mission_daily_on') is not None:
            self.scene = Scene.MISSION_DAILY
        elif self.find('mission_weekly_on') is not None:
            self.scene = Scene.MISSION_WEEKLY
        elif self.find('terminal_pre') is not None:
            self.scene = Scene.TERMINAL_MAIN
        elif self.find('open_recruitment') is not None:
            self.scene = Scene.RECRUIT_MAIN
        elif self.find('recruiting_instructions') is not None:
            self.scene = Scene.RECRUIT_TAGS
        elif self.find('agent_token') is not None:
            self.scene = Scene.RECRUIT_AGENT
        elif self.find('agent_token_1080_1440') is not None:
            self.scene = Scene.RECRUIT_AGENT
        elif self.find('agent_token_900_1440') is not None:
            self.scene = Scene.RECRUIT_AGENT
        elif self.find('agent_unlock') is not None:
            self.scene = Scene.SHOP_CREDIT
        elif self.find('shop_credit_2') is not None:
            self.scene = Scene.SHOP_OTHERS
        elif self.find('shop_cart') is not None:
            self.scene = Scene.SHOP_CREDIT_CONFIRM
        elif self.find('shop_assist') is not None:
            self.scene = Scene.SHOP_ASSIST
        elif self.find('login_logo') is not None and self.find('hypergryph') is not None:
            if self.find('login_awake') is not None:
                self.scene = Scene.LOGIN_QUICKLY
            elif self.find('login_account') is not None:
                self.scene = Scene.LOGIN_MAIN
            elif self.find('login_iknow') is not None:
                self.scene = Scene.LOGIN_ANNOUNCE
            else:
                self.scene = Scene.LOGIN_MAIN_NOENTRY
        elif self.find('register') is not None:
            self.scene = Scene.LOGIN_REGISTER
        elif self.find('login_loading') is not None:
            self.scene = Scene.LOGIN_LOADING
        elif self.find('login_iknow') is not None:
            self.scene = Scene.LOGIN_ANNOUNCE
        elif self.find('12cadpa') is not None:
            if self.find('cadpa_detail') is not None:
                self.scene = Scene.LOGIN_CADPA_DETAIL
            else:
                self.scene = Scene.LOGIN_START
        elif detector.announcement_close(self.img) is not None:
            self.scene = Scene.ANNOUNCEMENT
        elif self.find('skip') is not None:
            self.scene = Scene.SKIP
        elif self.find('upgrade') is not None:
            self.scene = Scene.UPGRADE
        elif detector.confirm(self.img) is not None:
            self.scene = Scene.CONFIRM
        elif self.find('login_verify') is not None:
            self.scene = Scene.LOGIN_INPUT
        elif self.find('login_captcha') is not None:
            self.scene = Scene.LOGIN_CAPTCHA
        elif self.find('login_connecting') is not None:
            self.scene = Scene.LOGIN_LOADING
        elif self.find('main_theme') is not None:
            self.scene = Scene.TERMINAL_MAIN_THEME
        elif self.find('episode') is not None:
            self.scene = Scene.TERMINAL_EPISODE
        elif self.find('biography') is not None:
            self.scene = Scene.TERMINAL_BIOGRAPHY
        elif self.find('collection') is not None:
            self.scene = Scene.TERMINAL_COLLECTION
        elif self.find('login_bilibili') is not None:
            self.scene = Scene.LOGIN_BILIBILI
        elif self.find('loading6') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading7') is not None:
            self.scene = Scene.LOADING
        elif self.find('arrange_order_options_scene') is not None:
            self.scene = Scene.INFRA_ARRANGE_ORDER
        else:
            self.scene = Scene.UNKNOWN
            self.device.check_current_focus()
        # save screencap to analyse
        if config.SCREENSHOT_PATH is not None:
            self.save_screencap(self.scene)
        logger.info(f'Scene: {self.scene}: {SceneComment[self.scene]}')
        return self.scene

    def get_infra_scene(self)-> int:
        if self.scene != Scene.UNDEFINED:
            return self.scene
        if self.find('connecting', scope=((self.w//2, self.h//10*8), (self.w//4*3, self.h))) is not None:
            self.scene = Scene.CONNECTING
        elif self.find('double_confirm') is not None:
            if self.find('network_check') is not None:
                self.scene = Scene.NETWORK_CHECK
            else:
                self.scene = Scene.DOUBLE_CONFIRM
        elif self.find('infra_overview') is not None:
            self.scene = Scene.INFRA_MAIN
        elif self.find('infra_todo') is not None:
            self.scene = Scene.INFRA_TODOLIST
        elif self.find('clue') is not None:
            self.scene = Scene.INFRA_CONFIDENTIAL
        elif self.find('arrange_check_in') or self.find('arrange_check_in_on') is not None:
            self.scene = Scene.INFRA_DETAILS
        elif self.find('infra_overview_in') is not None:
            self.scene = Scene.INFRA_ARRANGE
        elif self.find('arrange_confirm') is not None:
            self.scene = Scene.INFRA_ARRANGE_CONFIRM
        elif self.find('arrange_order_options_scene') is not None:
            self.scene = Scene.INFRA_ARRANGE_ORDER
        elif self.find('loading') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading2') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading3') is not None:
            self.scene = Scene.LOADING
        elif self.find('loading4') is not None:
            self.scene = Scene.LOADING
        elif self.find('index_nav', thres=250, scope=((0, 0), (100+self.w//4, self.h//10))) is not None:
            self.scene = Scene.INDEX
        elif self.is_black():
            self.scene = Scene.LOADING
        else:
            self.scene = Scene.UNKNOWN
            self.device.check_current_focus()
        # save screencap to analyse
        if config.SCREENSHOT_PATH is not None:
            self.save_screencap(self.scene)
        logger.info(f'Scene: {self.scene}: {SceneComment[self.scene]}')
        return self.scene

    def is_black(self) -> None:
        """ check if the current scene is all black """
        return np.max(self.gray[:, 105:-105]) < 16

    def nav_button(self):
        """ find navigation button """
        return self.find('nav_button', thres=128, scope=((0, 0), (100+self.w//4, self.h//10)))

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

    def score(self, res: str, draw: bool = False, scope: tp.Scope = None, thres: int = None) -> Optional[List[float]]:
        """
        查找元素是否出现在画面中，并返回分数

        :param res: 待识别元素资源文件名
        :param draw: 是否将识别结果输出到屏幕
        :param scope: ((x0, y0), (x1, y1))，提前限定元素可能出现的范围
        :param thres: 是否在匹配前对图像进行二值化处理

        :return ret: 若匹配成功，则返回元素在游戏界面中出现的位置，否则返回 None
        """
        logger.debug(f'find: {res}')
        res = f'{__rootdir__}/resources/{res}.png'

        if thres is not None:
            # 对图像二值化处理
            res_img = thres2(loadimg(res, True), thres)
            gray_img = cropimg(self.gray, scope)
            matcher = Matcher(thres2(gray_img, thres))
            score = matcher.score(res_img, draw=draw, only_score=True)
        else:
            res_img = loadimg(res, True)
            matcher = self.matcher
            score = matcher.score(res_img, draw=draw, scope=scope, only_score=True)
        return score

```

# `/opt/arknights-mower/arknights_mower/utils/recruit.py`

这段代码是一个Python函数，名为`filter_result`，属于`recruit.py`文件。其目的是对给定的 `tag_list` 和 `result_list` 进行处理，并将处理结果返回。

具体来说，这段代码实现了一个自定义的过滤函数，接收两个参数：`tag_list` 和 `result_list`，分别表示要过滤的标签列表和结果列表。函数内部首先对传入的参数进行处理，然后使用一系列条件判断，将符合条件的元素添加到 `temp_list` 中。最后，对 `temp_list` 进行处理，并将处理结果返回。

函数的实现中，通过调用 `logger.debug` 函数来输出日志信息，输出格式为：`tag`（标签） `level`（级别） `产阶级`（bors） 。这里，`temp_list` 中的元素是按照 `level` 升序排序的，所以，`temp_list` 中的第一个元素，一定是最小的。


```
#!Environment yolov8_Env
# -*- coding: UTF-8 -*-
"""
@Project ：arknights-mower 
@File    ：recruit.py
@Author  ：EightyDollars
@Date    ：2023/8/13 19:12
"""
from arknights_mower.utils.log import logger


def filter_result(tag_list, result_list, type=0):
    """
    temp_list
    {"tags": tag,
     "level":item['level'],
     "opers":item['opers']}
    """
    temp_list = []
    for tag in tag_list:
        logger.debug(tag)
        for result_dict in result_list:
            for item in result_dict["result"]:
                '''高资'''
                if type == 0:
                    if tag == result_dict['tags'] and item['level'] == result_dict['level']:
                        temp_list.append(item)
                elif type == 1:
                    if tag == item['tags']:
                        temp_list.append(
                            {"tags": tag,
                             "level": item['level'],
                             "opers": item['opers']})

    # 筛选好干员和对应tag存入返回用于后续jinja传输
    # logger.debug(temp_list)
    return temp_list

```

# `/opt/arknights-mower/arknights_mower/utils/scene.py`

这段代码的作用是创建了一个Scene类和一个SceneComment字典。

具体来说，首先从一个名为`data`的模块中导入了`scene_list`变量。然后，定义了一个`Scene`类，该类没有定义任何方法。接着，定义了一个`SceneComment`字典，该字典存储了各个场景的ID和评论。

接着，使用一个循环遍历`scene_list`中的每个场景，并将其ID存储在`id`属性中。循环还调用了`setattr`函数，该函数用于将场景ID映射到ID对象上。这样，当场景ID被调用时，程序会自动调用`Scene`类中定义的方法来获取场景对象。

最后，将场景ID和评论存储到`SceneComment`字典中。这样，每当场景ID被调用时，程序都会返回该场景对象的评论。


```
from ..data import scene_list


class Scene:
    pass


SceneComment = {}


for scene in scene_list.keys():
    id = int(scene)
    label = scene_list[scene]['label']
    comment = scene_list[scene]['comment']
    setattr(Scene, label, id)
    SceneComment[id] = comment

```

# `/opt/arknights-mower/arknights_mower/utils/scheduler_task.py`

这段代码定义了一个名为 `SchedulerTask` 的类，用于在命令行游戏 `SwordCraft` 中执行定时任务。这个类的实例可以包含以下信息：

- `time`：任务计划开始的时间，可以是当前时间的一个小时后开始，也可以是指定时间。
- `task_plan`：一个字典，用于存储任务的相关信息，如地图的 ID、目标单位的 ID 等。
- `task_type`：任务的类型，可以是 `Attack`、`Defend` 或 `Waiting` 之一。
- `meta_flag`：一个布尔值，表示是否执行 `Attack` 任务时考虑 `Waiting` 阶段。

在 `__init__` 方法中，首先检查 `time` 是否为空，如果是，则将当前时间设置为任务开始的时间。否则，将 `SchedulerTask` 实例的 `time` 属性设置为 `time`。接下来，将 `task_plan`、`task_type` 和 `meta_flag` 设置为传进来的参数。

`time_offset` 方法用于在指定的时间之后执行任务。它返回一个 `SchedulerTask` 实例，该实例的 `time` 属性增加了指定时间。

`__str__` 方法返回一个字符串，表示 `SchedulerTask` 实例的元组，其中包括 `time`、`task_plan`、`task_type` 和 `meta_flag`。

`__eq__` 方法用于比较两个 `SchedulerTask` 实例是否相等。如果它们具有相同的 `task_type`、`meta_flag` 和 `time_offset`，则返回 `True`，否则返回 `False`。


```
from datetime import datetime, timedelta
import copy
from arknights_mower.utils.datetime import the_same_time


class SchedulerTask:
    time = None
    type = ''
    plan = {}
    meta_flag = False

    def __init__(self, time=None, task_plan={}, task_type='', meta_flag=False):
        if time is None:
            self.time = datetime.now()
        else:
            self.time = time
        self.plan = task_plan
        self.type = task_type
        self.meta_flag = meta_flag

    def time_offset(self, h):
        after_offset = copy.deepcopy(self)
        after_offset.time += timedelta(hours=h)
        return after_offset

    def __str__(self):
        return f"SchedulerTask(time='{self.time}',task_plan={self.plan},task_type='{self.type}',meta_flag={self.meta_flag})"

    def __eq__(self, other):
        if isinstance(other, SchedulerTask):
            return self.type == other.type and self.plan == other.plan and the_same_time(self.time,
                                                                                         other.time) and self.meta_flag == other.meta_flag
        return False

```

# `/opt/arknights-mower/arknights_mower/utils/segment.py`

这段代码是一个带有未来时注解的函数，从Python 2.7开始引入。它主要是导入自在未来可能要引入的一些模块和函数。接下来，它导入了一个名为“traceback”的模块，以及一个名为“cv2”的模块，用于从摄像头获取图像。然后，它导入了一个名为“numpy”的模块，用于从列表中索引，接着导入了一个名为“torchvision”的模块，可能用于图像处理和计算机视觉任务。

接着，它从名为“agent_list”的函数中导入了列表“agent_list”，从名为“ocrhandle”的函数中导入了函数“ocrhandle”，从名为“detector”的函数中导入了函数“detector”，从名为“typealias”的模块中导入了类型定义“tp”，从名为“logger”的函数中导入了函数“logger”，从名为“recognize”的函数中导入了函数“recognize”。

此外，它还导入了一个名为“cv2”的模块，用于在摄像头中获取图像，接着从“RecognizeError”的函数中导入了函数“RecognizeError”。

最后，它定义了一个函数“__main__”，这个函数似乎是执行一系列图像处理和计算机视觉任务的通用步骤。


```
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


```

This appears to be a Python function named `segment_image` that is used for image segmentation. It takes in an input image and a number of parameters such as the minimum and maximum size of the image, the threshold for the minimum region of interest, and the minimum and maximum number of regions in the image.

The function first uses the numpy library to calculate the average of the pixel values in the up and down channels of the image, and then uses the `And` operator to check if the average is greater than or equal to 250. If this condition is met, the function sets the `flag` variable to `True` and increments the `up_1` counter.

The function then calculates the minimum and maximum values of the pixel values in the up and down channels, and uses these values to determine the minimum and maximum number of regions in the image.

The function also uses the `counter` function from the itertools module to keep track of the number of regions in the image.

Finally, the function uses the OpenCV library to draw the regions in the image and returns the image.

It is important to note that the function is not very readable and it is not clear what it does without the parameters passed.


```
class FloodCheckFailed(Exception):
    pass


def get_poly(x1: int, x2: int, y1: int, y2: int) -> tp.Rectangle:
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    return np.array([ [ x1, y1 ], [ x1, y2 ], [ x2, y2 ], [ x2, y1 ] ])


def credit(img: tp.Image, draw: bool = False) -> list[ tp.Scope ]:
    """
    信用交易所特供的图像分割算法
    """
    try:
        height, width, _ = img.shape

        left, right = 0, width
        while np.max(img[ :, right - 1 ]) < 100:
            right -= 1
        while np.max(img[ :, left ]) < 100:
            left += 1

        def average(i: int) -> int:
            num, sum = 0, 0
            for j in range(left, right):
                if img[ i, j, 0 ] == img[ i, j, 1 ] and img[ i, j, 1 ] == img[ i, j, 2 ]:
                    num += 1
                    sum += img[ i, j, 0 ]
            return sum // num

        def ptp(j: int) -> int:
            maxval = -999999
            minval = 999999
            for i in range(up_1, up_2):
                minval = min(minval, img[ i, j, 0 ])
                maxval = max(maxval, img[ i, j, 0 ])
            return maxval - minval

        up_1 = 0
        flag = False
        while not flag or average(up_1) >= 250:
            flag |= average(up_1) >= 250  # numpy.bool_
            up_1 += 1

        up_2 = up_1
        flag = False
        while not flag or average(up_2) < 220:
            flag |= average(up_2) < 220
            up_2 += 1

        down = height - 1
        while average(down) < 180:
            down -= 1

        right = width - 1
        while ptp(right) < 50:
            right -= 1

        left = 0
        while ptp(left) < 50:
            left += 1

        split_x = [ left + (right - left) // 5 * i for i in range(0, 6) ]
        split_y = [ up_1, (up_1 + down) // 2, down ]

        ret = [ ]
        for y1, y2 in zip(split_y[ :-1 ], split_y[ 1: ]):
            for x1, x2 in zip(split_x[ :-1 ], split_x[ 1: ]):
                ret.append(((x1, y1), (x2, y2)))

        if draw:
            for y1, y2 in zip(split_y[ :-1 ], split_y[ 1: ]):
                for x1, x2 in zip(split_x[ :-1 ], split_x[ 1: ]):
                    cv2.polylines(img, [ get_poly(x1, x2, y1, y2) ],
                                  True, 0, 10, cv2.LINE_AA)
            plt.imshow(img)
            plt.show()

        logger.debug(f'segment.credit: {ret}')
        return ret

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)


```

This is a Python function that appears to perform image segmentation. The function is called `segment_image` and takes an image as input. It uses the OpenCV library to retrieve information about the objects in the image and to draw a polygon around those objects. The function assumes that the input image is in RGB format and has a polygon that contains objects that are distinct from the background.

The function has several parameters:

* `img`: The input image
* `height`: The height of the input image
* `width`: The width of the input image
* `average(img)`: The average value of the pixel values in the input image
* `adj_x(up)`: The adjusted x-value for the upper edge of the polygon
* `adj_y(down)`: The adjusted y-value for the lower edge of the polygon
* `polygon_threshold`: The minimum number of pixels in a polygon that must be present in order to consider it to be a distinct object
* `draw`: Whether to draw a polygon around the detected objects
* `logger`: A logger for debugging purposes

The function first initializes the variables that it needs for the segmentation process. It then loops through the input image and extracts the pixel values at each location in the image.

For each pixel, the function checks whether the pixel value is greater than the threshold value for a polygon to be considered an object. If the pixel value is greater than the threshold, the function draws a polygon around the pixel according to the rules specified in the `segment_image.recruit` function.

Finally, the function returns the polygon image.

It is important to note that this function assumes that the input image is in RGB format and that the polygon threshold is set appropriately to avoid false positives. Additionally, the function may have other bugs or issues that need to be addressed before it can be used for production purposes.


```
def recruit(img: tp.Image, draw: bool = False) -> list[ tp.Scope ]:
    """
    公招特供的图像分割算法
    """
    try:
        height, width, _ = img.shape
        left, right = width // 2 - 100, width // 2 - 50

        def adj_x(i: int) -> int:
            if i == 0:
                return 0
            sum = 0
            for j in range(left, right):
                for k in range(3):
                    sum += abs(int(img[ i, j, k ]) - int(img[ i - 1, j, k ]))
            return sum // (right - left)

        def adj_y(j: int) -> int:
            if j == 0:
                return 0
            sum = 0
            for i in range(up_2, down_2):
                for k in range(3):
                    sum += abs(int(img[ i, j, k ]) - int(img[ i, j - 1, k ]))
            return int(sum / (down_2 - up_2))

        def average(i: int) -> int:
            sum = 0
            for j in range(left, right):
                sum += np.sum(img[ i, j, :3 ])
            return sum // (right - left) // 3

        def minus(i: int) -> int:
            s = 0
            for j in range(left, right):
                s += int(img[ i, j, 2 ]) - int(img[ i, j, 0 ])
            return s // (right - left)

        up = 0
        while minus(up) > -100:
            up += 1
        while not (adj_x(up) > 80 and minus(up) > -10 and average(up) > 210):
            up += 1
        up_2, down_2 = up - 90, up - 40

        left = 0
        while np.max(img[ :, left ]) < 100:
            left += 1
        left += 1
        while adj_y(left) < 50:
            left += 1

        right = width - 1
        while np.max(img[ :, right ]) < 100:
            right -= 1
        while adj_y(right) < 50:
            right -= 1

        split_x = [ left, (left + right) // 2, right ]
        down = height - 1
        split_y = [ up, (up + down) // 2, down ]

        ret = [ ]
        for y1, y2 in zip(split_y[ :-1 ], split_y[ 1: ]):
            for x1, x2 in zip(split_x[ :-1 ], split_x[ 1: ]):
                ret.append(((x1, y1), (x2, y2)))

        if draw:
            for y1, y2 in zip(split_y[ :-1 ], split_y[ 1: ]):
                for x1, x2 in zip(split_x[ :-1 ], split_x[ 1: ]):
                    cv2.polylines(img, [ get_poly(x1, x2, y1, y2) ],
                                  True, 0, 10, cv2.LINE_AA)
            plt.imshow(img)
            plt.show()

        logger.debug(f'segment.recruit: {ret}')
        return ret

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)


```

This appears to be a Python function that performs image segmentation using the OpenCV library. It takes in two arguments: an image (in the format of a NumPy array) and a mask (in the format of a binary NumPy array).

The function first converts the mask to a binary form using `numpy.where()`. It then uses the `get_poly()` function to convert the mask to a polygon list.

The function then loops through the polygon list and performs a polygon check to determine if the polygon should be included in the segmented image. If the polygon is included, it is added to a list of segmented rooms.

Finally, the function checks if the image should be drawn on top of the segmented rooms. If it should be, it creates a polygon list from the rooms and uses the `cv2.polylines()` function to draw the polyggon on top of the image. Finally, it returns the image segmented according to the rooms.

This function may have potential issues with accurate segmentation, as the room shapes are not well defined.


```
def base(img: tp.Image, central: tp.Scope, draw: bool = False) -> dict[ str, tp.Rectangle ]:
    """
    基建布局的图像分割算法
    """
    try:
        ret = {}

        x1, y1 = central[ 0 ]
        x2, y2 = central[ 1 ]
        alpha = (y2 - y1) / 160
        x1 -= 170 * alpha
        x2 += 182 * alpha
        y1 -= 67 * alpha
        y2 += 67 * alpha
        central = get_poly(x1, x2, y1, y2)
        ret[ 'central' ] = central

        for i in range(1, 5):
            y1 = y2 + 25 * alpha
            y2 = y1 + 134 * alpha
            if i & 1:
                dormitory = get_poly(x1, x2 - 158 * alpha, y1, y2)
            else:
                dormitory = get_poly(x1 + 158 * alpha, x2, y1, y2)
            ret[ f'dormitory_{i}' ] = dormitory

        x1, y1 = ret[ 'dormitory_1' ][ 0 ]
        x2, y2 = ret[ 'dormitory_1' ][ 2 ]

        x1 = x2 + 419 * alpha
        x2 = x1 + 297 * alpha
        factory = get_poly(x1, x2, y1, y2)
        ret[ f'factory' ] = factory

        y2 = y1 - 25 * alpha
        y1 = y2 - 134 * alpha
        meeting = get_poly(x1 - 158 * alpha, x2, y1, y2)
        ret[ f'meeting' ] = meeting

        y1 = y2 + 25 * alpha
        y2 = y1 + 134 * alpha
        y1 = y2 + 25 * alpha
        y2 = y1 + 134 * alpha
        contact = get_poly(x1, x2, y1, y2)
        ret[ f'contact' ] = contact

        y1 = y2 + 25 * alpha
        y2 = y1 + 134 * alpha
        train = get_poly(x1, x2, y1, y2)
        ret[ f'train' ] = train

        for floor in range(1, 4):
            x1, y1 = ret[ f'dormitory_{floor}' ][ 0 ]
            x2, y2 = ret[ f'dormitory_{floor}' ][ 2 ]
            x2 = x1 - 102 * alpha
            x1 = x2 - 295 * alpha
            if floor & 1 == 0:
                x2 = x1 - 24 * alpha
                x1 = x2 - 295 * alpha
            room = get_poly(x1, x2, y1, y2)
            ret[ f'room_{floor}_3' ] = room
            x2 = x1 - 24 * alpha
            x1 = x2 - 295 * alpha
            room = get_poly(x1, x2, y1, y2)
            ret[ f'room_{floor}_2' ] = room
            x2 = x1 - 24 * alpha
            x1 = x2 - 295 * alpha
            room = get_poly(x1, x2, y1, y2)
            ret[ f'room_{floor}_1' ] = room

        if draw:
            polys = list(ret.values())
            cv2.polylines(img, polys, True, (255, 0, 0), 10, cv2.LINE_AA)
            plt.imshow(img)
            plt.show()

        logger.debug(f'segment.base: {ret}')
        return ret

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)

```

This is a Python module that uses the OpenCV library to perform image segmentation on an input image. The module has two functions, `segment_worker` and `remove_button`, which are responsible for the main image processing logic and the button to remove segments from the image, respectively.

The `segment_worker` function takes an input image, a threshold value, and a polygon that is detected using the polygon detected by a previous segmentation worker. It starts by finding the x-coordinate of the polygon that has the highest score using the `get_poly` function from the `segment_py` module. Then, it checks if the polygon has more than 9 units of difference in the x direction than the threshold value. If the polygon does not meet the threshold, it resets the `st` variable to 0 and returns. Otherwise, it adds the pixel at the current x-coordinate and the corresponding y-coordinate to the `ret` list, which is a list of polygons detected by the current worker. Finally, it checks if the `draw` parameter is `True`, and if so, it adds lines to the image and displays it using Matplotlib.

The `remove_button` function takes a list of polygons detected by the current worker and returns the x-coordinates of the polygons that have been removed. It first finds the intersection of all the polygons in the `ret` list using the `tolist` method and then returns the list of x-coordinates of the polygons that have been removed.

The `segment_worker` function uses the `segment_py` module to detect polygons in the input image. It also uses the `get_poly` function to get the x- and y-coordinates of each polygon detected by the current worker. This allows the current worker to compare the newly detected polygons to the previously detected ones to determine if they belong to the same object or not.


```
def worker(img: tp.Image, draw: bool = False) -> tuple[ list[ tp.Rectangle ], tp.Rectangle, bool ]:
    """
    进驻总览的图像分割算法
    """
    try:
        height, width, _ = img.shape

        left, right = 0, width
        while np.max(img[ :, right - 1 ]) < 100:
            right -= 1
        while np.max(img[ :, left ]) < 100:
            left += 1

        x0 = right - 1
        while np.average(img[ :, x0, 1 ]) >= 100:
            x0 -= 1
        x0 -= 2

        seg = [ ]
        remove_mode = False
        pre, st = int(img[ 0, x0, 1 ]), 0
        for y in range(1, height):
            remove_mode |= int(img[ y, x0, 0 ]) - int(img[ y, x0, 1 ]) > 40
            if np.ptp(img[ y, x0 ]) <= 1 or int(img[ y, x0, 0 ]) - int(img[ y, x0, 1 ]) > 40:
                now = int(img[ y, x0, 1 ])
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
        # if st != 0:
        #     seg.append((st, height))
        logger.debug(f'segment.worker: seg {seg}')

        remove_button = get_poly(x0 - 10, x0, seg[ 0 ][ 0 ], seg[ 0 ][ 1 ])
        seg = seg[ 1: ]

        for i in range(1, len(seg)):
            if seg[ i ][ 1 ] - seg[ i ][ 0 ] > 9:
                x1 = x0
                while img[ seg[ i ][ 1 ] - 3, x1 - 1, 2 ] < 100:
                    x1 -= 1
                break

        ret = [ ]
        for i in range(len(seg)):
            if seg[ i ][ 1 ] - seg[ i ][ 0 ] > 9:
                ret.append(get_poly(x1, x0, seg[ i ][ 0 ], seg[ i ][ 1 ]))

        if draw:
            cv2.polylines(img, ret, True, (255, 0, 0), 10, cv2.LINE_AA)
            plt.imshow(img)
            plt.show()

        logger.debug(f'segment.worker: {[ x.tolist() for x in ret ]}')
        return ret, remove_button, remove_mode

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)


```

This is a Python implementation of the ORC (Object Recognition) algorithm. It is used to recognize objects in images and videos. The code is divided into several段， each of which performs a specific task.

The `segmentation.py` module contains the main function for recognizing objects. This function, when called, takes an input image, a starting X-coordinate, an ending Y-coordinate, and a hyperparameter `gap`, and returns the recognized agent names and the bounding boxes around each recognized agent.

The `encoding.py` module contains the encoder used to extract features from the input image. It takes the input image and the recognized agent names as input and outputs a one-hot encoded image, which is the image in which each recognized agent is represented as a binary word.

The `utils.py` module contains utility functions for working with images, such as sorting an image by its element, and extracting bounding boxes around recognized agents.

The `img_proc.py` module contains image processing functions, such as resizing an image and converting it to a binary image.

The `demo.py` module contains a demonstration of how to use the `segmentation.py` module to recognize objects in an input image.

The `python_ocr.py` module contains the ORC recognition algorithm. This algorithm is responsible for recognizing the objects in the input image and outputs the recognized agent names and the bounding boxes around each recognized agent.

The `python_ocr.py` module has two functions, `detect_objects.py` and `recognize_objects.py`, which perform the main tasks of the ORC algorithm.

The `detect_objects.py` function takes an input image and returns a list of recognized agent names and the bounding boxes around each recognized agent.

The `recognize_objects.py` function takes an input image and a hyperparameter `gap` and returns the recognized agent names and the bounding boxes around each recognized agent. It also


```
def agent(img, draw=False):
    """
    干员总览的图像分割算法
    """
    try:
        height, width, _ = img.shape
        resolution = height
        left, right = 0, width

        # 异形屏适配
        while np.max(img[ :, right - 1 ]) < 100:
            right -= 1
        while np.max(img[ :, left ]) < 100:
            left += 1

        # 去除左侧干员详情
        x0 = left + 1
        while not (img[ height - 10, x0 - 1, 0 ] > img[ height - 10, x0, 0 ] + 10 and abs(
                int(img[ height - 10, x0, 0 ]) - int(img[ height - 10, x0 + 1, 0 ])) < 5):
            x0 += 1

        # ocr 初步识别干员名称
        ocr = ocrhandle.predict(img[ :, x0:right ])

        # 收集成功识别出来的干员名称识别结果，提取 y 范围，并将重叠的范围进行合并
        segs = [ (min(x[ 2 ][ 0 ][ 1 ], x[ 2 ][ 1 ][ 1 ]), max(x[ 2 ][ 2 ][ 1 ], x[ 2 ][ 3 ][ 1 ]))
                 for x in ocr if x[ 1 ] in agent_list ]
        while True:
            _a, _b = None, None
            for i in range(len(segs)):
                for j in range(len(segs)):
                    if i != j and (
                            segs[ i ][ 0 ] <= segs[ j ][ 0 ] <= segs[ i ][ 1 ] or segs[ i ][ 0 ] <= segs[ j ][ 1 ] <=
                            segs[ i ][ 1 ]):
                        _a, _b = segs[ i ], segs[ j ]
                        break
                if _b is not None:
                    break
            if _b is not None:
                segs.remove(_a)
                segs.remove(_b)
                segs.append((min(_a[ 0 ], _b[ 0 ]), max(_a[ 1 ], _b[ 1 ])))
            else:
                break
        segs = sorted(segs)

        # 计算纵向的四个高度，[y0, y1] 是第一行干员名称的纵向坐标范围，[y2, y3] 是第二行干员名称的纵向坐标范围
        y0 = y1 = y2 = y3 = None
        for x in segs:
            if x[ 1 ] < height // 2:
                y0, y1 = x
            else:
                y2, y3 = x
        if y0 is None or y2 is None:
            raise RecognizeError
        hpx = y1 - y0  # 卡片上干员名称的高度
        logger.debug((segs, [ y0, y1, y2, y3 ]))

        # 预计算：横向坐标范围集合
        x_set = set()
        for x in ocr:
            if x[ 1 ] in agent_list and (y0 <= x[ 2 ][ 0 ][ 1 ] <= y1 or y2 <= x[ 2 ][ 0 ][ 1 ] <= y3):
                # 只考虑矩形右边端点
                x_set.add(x[ 2 ][ 1 ][ 0 ])
                x_set.add(x[ 2 ][ 2 ][ 0 ])
        x_set = sorted(x_set)
        logger.debug(x_set)

        # 排除掉一些重叠的范围，获得最终的横向坐标范围
        gap = 160 * (resolution / 1080)  # 卡片宽度下限
        x_set = [ x_set[ 0 ] ] + \
                [ y for x, y in zip(x_set[ :-1 ], x_set[ 1: ]) if y - x > gap ]
        gap = [ y - x for x, y in zip(x_set[ :-1 ], x_set[ 1: ]) ]
        logger.debug(sorted(gap))
        gap = int(np.median(gap))  # 干员卡片宽度
        for x, y in zip(x_set[ :-1 ], x_set[ 1: ]):
            if y - x > gap:
                gap_num = round((y - x) / gap)
                for i in range(1, gap_num):
                    x_set.append(int(x + (y - x) / gap_num * i))
        x_set = sorted(x_set)
        if x_set[ -1 ] - x_set[ -2 ] < gap:
            # 如果最后一个间隔不足宽度则丢弃，避免出现「梅尔」只露出一半识别成「梅」算作成功识别的情况
            x_set = x_set[ :-1 ]
        while np.min(x_set) > 0:
            x_set.append(np.min(x_set) - gap)
        while np.max(x_set) < right - x0:
            x_set.append(np.max(x_set) + gap)
        x_set = sorted(x_set)
        logger.debug(x_set)

        # 获得所有的干员名称对应位置
        ret = [ ]
        for x1, x2 in zip(x_set[ :-1 ], x_set[ 1: ]):
            if 0 <= x1 + hpx and x0 + x2 + 5 <= right:
                ret += [ get_poly(x0 + x1 + hpx, x0 + x2 + 5, y0, y1),
                         get_poly(x0 + x1 + hpx, x0 + x2 + 5, y2, y3) ]

        # draw for debug
        if draw:
            __img = img.copy()
            cv2.polylines(__img, ret, True, (255, 0, 0), 3, cv2.LINE_AA)
            plt.imshow(__img)
            plt.show()

        logger.debug(f'segment.agent: {[ x.tolist() for x in ret ]}')
        return ret, ocr

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)


```

This is a Python implementation of the segmentation algorithm you described, using the OpenCV library and the Detection Transform library. The code is divided into different functions, each of which performs a specific task.

The `RecognizeError` class is raised if any of the inputs or conditions inside the loop are not met.


```
def free_agent(img, draw=False):
    """
    识别未在工作中的干员
    """
    try:
        height, width, _ = img.shape
        resolution = height
        left, right = 0, width

        # 异形屏适配
        while np.max(img[ :, right - 1 ]) < 100:
            right -= 1
        while np.max(img[ :, left ]) < 100:
            left += 1

        # 去除左侧干员详情
        x0 = left + 1
        while not (img[ height - 10, x0 - 1, 0 ] > img[ height - 10, x0, 0 ] + 10 and abs(
                int(img[ height - 10, x0, 0 ]) - int(img[ height - 10, x0 + 1, 0 ])) < 5):
            x0 += 1

        # 获取分割结果
        ret = agent(img, draw)
        st = ret[ -2 ][ 2 ]  # 起点
        ed = ret[ 0 ][ 1 ]  # 终点

        # 收集 y 坐标并初步筛选
        y_set = set()
        __ret = [ ]
        for poly in ret:
            __img = img[ poly[ 0, 1 ]:poly[ 2, 1 ], poly[ 0, 0 ]:poly[ 2, 0 ] ]
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

        y1, y2, y4, y5 = sorted(list(y_set))
        y0 = height - y5
        y3 = y0 - y2 + y5

        ret_free = [ ]
        for poly in ret:
            poly[ :, 1 ][ poly[ :, 1 ] == y1 ] = y0
            poly[ :, 1 ][ poly[ :, 1 ] == y4 ] = y3
            __img = img[ poly[ 0, 1 ]:poly[ 2, 1 ], poly[ 0, 0 ]:poly[ 2, 0 ] ]
            if not detector.is_on_shift(__img):
                ret_free.append(poly)

        if draw:
            __img = img.copy()
            cv2.polylines(__img, ret_free, True, (255, 0, 0), 3, cv2.LINE_AA)
            plt.imshow(__img)
            plt.show()

        logger.debug(f'segment.free_agent: {[ x.tolist() for x in ret_free ]}')
        return ret_free, st, ed

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise RecognizeError(e)

```

# `/opt/arknights-mower/arknights_mower/utils/simulator.py`

这段代码的作用是实现了一个模拟器，可以模拟Nox和MuMu12两种不同的情况。具体来说，代码中定义了一个名为Simulator_Type的枚举类型，用于指定要模拟的模拟器类型。然后，定义了一个名为Restart_Simulator的函数，用于重启指定的模拟器。

在函数中，首先根据模拟器类型调用对应枚举类型的方法，并传入相应的参数。如果模拟器类型为Simulator_Type.Nox，那么会执行以下命令：


Nox.exe -clone:Nox_{data["index"]} -quit -wait 2


如果模拟器类型为Simulator_Type.MuMu12，那么会执行以下命令：


MuMuManager.exe api -v shutdown_player {data["index"]} -wait 25


其中，参数{data["index"]}表示要关闭的模拟器的ID，后面的参数指定了要关闭的命令。如果参数中指定的ID不存在，函数会输出一条警告信息。

在执行完命令后，函数都会等待一段时间，然后继续等待2秒钟，以确保模拟器已经正常关闭。如果模拟器无法正常关闭，函数会输出一条警告信息并停止执行。


```
import subprocess
from enum import Enum
from arknights_mower.utils.log import logger
import time


class Simulator_Type(Enum):
    Nox = "夜神"
    MuMu12 = "MuMu12"


def restart_simulator(data):
    index = data["index"]
    simulator_type = data["name"]
    cmd = ""
    if simulator_type in [Simulator_Type.Nox.value, Simulator_Type.MuMu12.value]:
        if simulator_type == Simulator_Type.Nox.value:
            cmd = "Nox.exe"
            if index >= 0:
                cmd += f' -clone:Nox_{data["index"]}'
            cmd += " -quit"
        elif simulator_type == Simulator_Type.MuMu12.value:
            cmd = "MuMuManager.exe api -v "
            if index >= 0:
                cmd += f'{data["index"]} '
            cmd += "shutdown_player"
        exec_cmd(cmd, data["simulator_folder"])
        logger.info(f'开始关闭{simulator_type}模拟器，等待2秒钟')
        time.sleep(2)
        if simulator_type == Simulator_Type.Nox.value:
            cmd = cmd.replace(' -quit', '')
        elif simulator_type == Simulator_Type.MuMu12.value:
            cmd = cmd.replace(' shutdown_player', ' launch_player')
        exec_cmd(cmd, data["simulator_folder"])
        logger.info(f'开始启动{simulator_type}模拟器，等待25秒钟')
        time.sleep(25)
    else:
        logger.warning(f"尚未支持{simulator_type}重启/自动启动")


```

这段代码定义了一个名为 `exec_cmd` 的函数，它接受两个参数 `cmd` 和 `folder_path`。函数的作用是在给定的文件夹路径中运行 `cmd` 命令，并将输出和错误信息返回给调用者。

函数内部使用了 `subprocess` 模块，该模块提供了方便的系统调用。函数使用 `subprocess.Popen` 函数来创建一个进程对象，并传递给 `cmd` 和 `folder_path` 参数。`shell=True` 参数表示运行 `cmd` 命令时使用当前目录，`cwd=folder_path` 参数用于指定当前目录。

函数还使用 `subprocess.PIPE` 对象来从 `cmd` 的标准输出接收输出信息，使用 `subprocess.PIPE` 对象来从 `cmd` 的标准错误接收错误信息，并使用 `universal_newlines=True` 参数来确保输出和错误信息连接在一起。

如果 `cmd` 运行的时间超过 2 秒钟，函数将使用 `subprocess.TimeoutExpired` 异常来捕获。在这种情况下，函数将杀死进程并返回。


```
def exec_cmd(cmd, folder_path):
    try:
        process = subprocess.Popen(cmd, shell=True, cwd=folder_path, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()

```

# `/opt/arknights-mower/arknights_mower/utils/solver.py`

这段代码的作用是定义了一个名为“邮件推送”的类，通过调用邮件推送类中的方法，实现发送电子邮件的功能。下面按照代码的作用，逐步解释：

1. 从“__future__”导入“annotations”模块，这样就可以使用在未来可以定义的新类型的代码。

2. 导入“smtplib”和“email.mime.text”模块，用于实现发送电子邮件所需的SMTP和MIME库。

3. 导入“time”和“traceback”模块，用于在程序出现问题时查找和记录错误信息。

4. 从“abc”模块中导入“ abstractmethod”，用于定义邮件推送类的抽象方法。

5. 从“..utils”模块中导入“tp”类型定义，可能是用于定义邮件主题等需要从外部传入的配置参数。

6. 从“device”模块中导入“Device”和“KeyCode”类型定义，可能是用于定义设备类型和与设备相关的函数和方法。

7. 从“log”模块中导入“logger”函数，用于记录错误信息，并可以进行日志输出。

8. 从“recognize”模块中导入“RecognizeError”和“Recognizer”类型定义，可能是用于实现语音识别等功能的类。

9. 最后定义了“EmailPush”类，其中包含了所有邮件推送所需的功能和属性，包括设置邮件主题、发送邮件、处理异常等。


```
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import time
import traceback
from abc import abstractmethod

from ..utils import typealias as tp
from . import config, detector
from .device import Device, KeyCode
from .log import logger
from .recognize import RecognizeError, Recognizer, Scene


```



This is a class that can be used to send email to specified accounts. It has the following methods:

- `send_email(body, subject, subtype, retry_times)`: Sends an email with the specified body, subject, subtype, and number of retries. The body should be a string that contains the email content. This method uses the `MIMEMultipart` and `MIMEText` classes to create the email message and attach the email body.
- `wait_for_scene(scene, method, wait_count, sleep_time)`: Waits for the specified scene to appear. This method takes three arguments: the scene to wait for, the method to use (either `'get_infra_scene'` or `'scene'`), the number of waits, and the amount of time to sleep before checking for the scene. It uses the `get_infra_scene` method to check if the scene has been specified, and the `scene` method to check if the scene is currently active. If the scene is active, it returns `True`.

This class can be used as follows:
scss
wait_send = WaitForScene(場景， 'get_infra_scene')
wait_send.send_email('你是中国吗', '我来自88Dollars', 'plain')


You can also use the `send_email` method to send email:
makefile
send_email.send_email('你是中国吗', '我来自88Dollars', 'plain')



```
class StrategyError(Exception):
    """ Strategy Error """
    pass


class BaseSolver:
    """ Base class, provide basic operation """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        # self.device = device if device is not None else (recog.device if recog is not None else Device())
        if device is None and recog is not None:
            raise RuntimeError
        self.device = device if device is not None else Device()
        self.recog = recog if recog is not None else Recognizer(self.device)
        self.device.check_current_focus()
        self.recog.update()

    def run(self) -> None:
        retry_times = config.MAX_RETRYTIME
        result = None
        while retry_times > 0:
            try:
                result = self.transition()
                if result:
                    return result
            except RecognizeError as e:
                logger.warning(f'识别出了点小差错 qwq: {e}')
                self.recog.save_screencap('failure')
                retry_times -= 1
                self.sleep(3)
                continue
            except StrategyError as e:
                logger.error(e)
                logger.debug(traceback.format_exc())
                return
            except Exception as e:
                raise e
            retry_times = config.MAX_RETRYTIME

    @abstractmethod
    def transition(self) -> bool:
        # the change from one state to another is called transition
        return True  # means task completed

    def get_color(self, pos: tp.Coordinate) -> tp.Pixel:
        """ get the color of the pixel """
        return self.recog.color(pos[0], pos[1])

    def get_pos(self, poly: tp.Location, x_rate: float = 0.5, y_rate: float = 0.5) -> tp.Coordinate:
        """ get the pos form tp.Location """
        if poly is None:
            raise RecognizeError('poly is empty')
        elif len(poly) == 4:
            # tp.Rectangle
            x = (poly[0][0] * (1 - x_rate) + poly[1][0] * (1 - x_rate) +
                 poly[2][0] * x_rate + poly[3][0] * x_rate) / 2
            y = (poly[0][1] * (1 - y_rate) + poly[3][1] * (1 - y_rate) +
                 poly[1][1] * y_rate + poly[2][1] * y_rate) / 2
        elif len(poly) == 2 and isinstance(poly[0], (list, tuple)):
            # tp.Scope
            x = poly[0][0] * (1 - x_rate) + poly[1][0] * x_rate
            y = poly[0][1] * (1 - y_rate) + poly[1][1] * y_rate
        else:
            # tp.Coordinate
            x, y = poly
        return (int(x), int(y))

    def sleep(self, interval: float = 1, rebuild: bool = True) -> None:
        """ sleeping for a interval """
        time.sleep(interval)
        self.recog.update(rebuild=rebuild)

    def input(self, referent: str, input_area: tp.Scope, text: str = None) -> None:
        """ input text """
        logger.debug(f'input: {referent} {input_area}')
        self.device.tap(self.get_pos(input_area))
        time.sleep(0.5)
        if text is None:
            text = input(referent).strip()
        self.device.send_text(str(text))
        self.device.tap((0, 0))

    def find(self, res: str, draw: bool = False, scope: tp.Scope = None, thres: int = None, judge: bool = True,
             strict: bool = False, score=0.0) -> tp.Scope:
        return self.recog.find(res, draw, scope, thres, judge, strict, score)

    def tap(self, poly: tp.Location, x_rate: float = 0.5, y_rate: float = 0.5, interval: float = 1,
            rebuild: bool = True) -> None:
        """ tap """
        pos = self.get_pos(poly, x_rate, y_rate)
        self.device.tap(pos)
        if interval > 0:
            self.sleep(interval, rebuild)

    def tap_element(self, element_name: str, x_rate: float = 0.5, y_rate: float = 0.5, interval: float = 1,
                    rebuild: bool = True,
                    draw: bool = False, scope: tp.Scope = None, judge: bool = True, detected: bool = False) -> bool:
        """ tap element """
        if element_name == 'nav_button':
            element = self.recog.nav_button()
        else:
            element = self.find(element_name, draw, scope, judge=judge)
        if detected and element is None:
            return False
        self.tap(element, x_rate, y_rate, interval, rebuild)
        return True

    def swipe(self, start: tp.Coordinate, movement: tp.Coordinate, duration: int = 100, interval: float = 1,
              rebuild: bool = True) -> None:
        """ swipe """
        end = (start[0] + movement[0], start[1] + movement[1])
        self.device.swipe(start, end, duration=duration)
        if interval > 0:
            self.sleep(interval, rebuild)

    def swipe_only(self, start: tp.Coordinate, movement: tp.Coordinate, duration: int = 100,
                   interval: float = 1) -> None:
        """ swipe only, no rebuild and recapture """
        end = (start[0] + movement[0], start[1] + movement[1])
        self.device.swipe(start, end, duration=duration)
        if interval > 0:
            time.sleep(interval)

    # def swipe_seq(self, points: list[tp.Coordinate], duration: int = 100, interval: float = 1, rebuild: bool = True) -> None:
    #     """ swipe with point sequence """
    #     self.device.swipe(points, duration=duration)
    #     if interval > 0:
    #         self.sleep(interval, rebuild)

    # def swipe_move(self, start: tp.Coordinate, movements: list[tp.Coordinate], duration: int = 100, interval: float = 1, rebuild: bool = True) -> None:
    #     """ swipe with start and movement sequence """
    #     points = [start]
    #     for move in movements:
    #         points.append((points[-1][0] + move[0], points[-1][1] + move[1]))
    #     self.device.swipe(points, duration=duration)
    #     if interval > 0:
    #         self.sleep(interval, rebuild)

    def swipe_noinertia(self, start: tp.Coordinate, movement: tp.Coordinate, duration: int = 100, interval: float = 1,
                        rebuild: bool = False) -> None:
        """ swipe with no inertia (movement should be vertical) """
        points = [start]
        if movement[0] == 0:
            dis = abs(movement[1])
            points.append((start[0] + 100, start[1]))
            points.append((start[0] + 100, start[1] + movement[1]))
            points.append((start[0], start[1] + movement[1]))
        else:
            dis = abs(movement[0])
            points.append((start[0], start[1] + 100))
            points.append((start[0] + movement[0], start[1] + 100))
            points.append((start[0] + movement[0], start[1]))
        self.device.swipe_ext(points, durations=[200, dis * duration // 100, 200])
        if interval > 0:
            self.sleep(interval, rebuild)

    def back(self, interval: float = 1, rebuild: bool = True) -> None:
        """ send back keyevent """
        self.device.send_keyevent(KeyCode.KEYCODE_BACK)
        self.sleep(interval, rebuild)

    def scene(self) -> int:
        """ get the current scene in the game """
        return self.recog.get_scene()

    def get_infra_scene(self) -> int:
        """ get the current scene in the infra """
        return self.recog.get_infra_scene()

    def is_login(self):
        """ check if you are logged in """
        return not (self.scene() // 100 == 1 or self.scene() // 100 == 99 or self.scene() == -1)

    def login(self):
        """
        登录进游戏
        """
        retry_times = config.MAX_RETRYTIME
        while retry_times and not self.is_login():
            try:
                if self.scene() == Scene.LOGIN_START:
                    self.tap((self.recog.w // 2, self.recog.h - 10), 3)
                elif self.scene() == Scene.LOGIN_NEW:
                    self.tap(self.find('login_new', score=0.8))
                elif self.scene() == Scene.LOGIN_NEW_B:
                    self.tap(self.find('login_bilibili_new', score=0.8))
                elif self.scene() == Scene.LOGIN_QUICKLY:
                    self.tap_element('login_awake')
                elif self.scene() == Scene.LOGIN_MAIN:
                    self.tap_element('login_account', 0.25)
                elif self.scene() == Scene.LOGIN_REGISTER:
                    self.back(2)
                elif self.scene() == Scene.LOGIN_CAPTCHA:
                    exit()
                    # self.back(600)  # TODO: Pending
                elif self.scene() == Scene.LOGIN_INPUT:
                    input_area = self.find('login_username')
                    if input_area is not None:
                        self.input('Enter username: ', input_area, config.USERNAME)
                    input_area = self.find('login_password')
                    if input_area is not None:
                        self.input('Enter password: ', input_area, config.PASSWORD)
                    self.tap_element('login_button')
                elif self.scene() == Scene.LOGIN_ANNOUNCE:
                    self.tap_element('login_iknow')
                elif self.scene() == Scene.LOGIN_LOADING:
                    self.waiting_solver(Scene.LOGIN_LOADING)
                elif self.scene() == Scene.LOADING:
                    self.waiting_solver(Scene.LOADING)
                elif self.scene() == Scene.CONNECTING:
                    self.waiting_solver(Scene.CONNECTING)
                elif self.scene() == Scene.CONFIRM:
                    self.tap(detector.confirm(self.recog.img))
                elif self.scene() == Scene.LOGIN_MAIN_NOENTRY:
                    self.waiting_solver(Scene.LOGIN_MAIN_NOENTRY)
                elif self.scene() == Scene.LOGIN_CADPA_DETAIL:
                    self.back(2)
                elif self.scene() == Scene.LOGIN_BILIBILI:
                    self.tap_element('login_bilibili_entry')
                elif self.scene() == Scene.NETWORK_CHECK:
                    self.tap_element('double_confirm', 0.2)
                elif self.scene() == Scene.UNKNOWN:
                    raise RecognizeError('Unknown scene')
                else:
                    raise RecognizeError('Unanticipated scene')
            except RecognizeError as e:
                logger.warning(f'识别出了点小差错 qwq: {e}')
                self.recog.save_screencap('failure')
                retry_times -= 1
                self.sleep(3)
                continue
            except Exception as e:
                raise e
            retry_times = config.MAX_RETRYTIME

        if not self.is_login():
            raise StrategyError

    def get_navigation(self):
        """
        判断是否存在导航栏，若存在则打开
        """
        retry_times = config.MAX_RETRYTIME
        while retry_times:
            if self.scene() == Scene.NAVIGATION_BAR:
                return True
            elif not self.tap_element('nav_button', detected=True):
                return False
            retry_times -= 1

    def back_to_infrastructure(self):
        self.back_to_index()
        self.tap_element('index_infrastructure')

    def back_to_index(self):
        """
        返回主页
        """
        logger.info('back to index')
        retry_times = config.MAX_RETRYTIME
        pre_scene = None
        while retry_times and self.scene() != Scene.INDEX:
            try:
                if self.get_navigation():
                    self.tap_element('nav_index')
                elif self.scene() == Scene.CLOSE_MINE:
                    self.tap_element('close_mine')
                elif self.scene() == Scene.CHECK_IN:
                    self.tap_element('check_in')
                elif self.scene() == Scene.ANNOUNCEMENT:
                    self.tap(detector.announcement_close(self.recog.img))
                elif self.scene() == Scene.MATERIEL:
                    self.tap_element('materiel_ico')
                elif self.scene() // 100 == 1:
                    self.login()
                elif self.scene() == Scene.CONFIRM:
                    self.tap(detector.confirm(self.recog.img))
                elif self.scene() == Scene.LOADING:
                    self.waiting_solver(Scene.LOADING)
                elif self.scene() == Scene.CONNECTING:
                    self.waiting_solver(Scene.CONNECTING)
                elif self.scene() == Scene.SKIP:
                    self.tap_element('skip')
                elif self.scene() == Scene.OPERATOR_ONGOING:
                    self.sleep(10)
                elif self.scene() == Scene.OPERATOR_FINISH:
                    self.tap((self.recog.w // 2, 10))
                elif self.scene() == Scene.OPERATOR_ELIMINATE_FINISH:
                    self.tap((self.recog.w // 2, 10))
                elif self.scene() == Scene.DOUBLE_CONFIRM:
                    self.tap_element('double_confirm', 0.8)
                elif self.scene() == Scene.NETWORK_CHECK:
                    self.tap_element('double_confirm', 0.2)
                elif self.scene() == Scene.MAIL:
                    mail = self.find('mail')
                    mid_y = (mail[0][1] + mail[1][1]) // 2
                    self.tap((mid_y, mid_y))
                elif self.scene() == Scene.INFRA_ARRANGE_CONFIRM:
                    self.tap((self.recog.w // 3, self.recog.h - 10))
                elif self.scene() == Scene.UNKNOWN:
                    raise RecognizeError('Unknown scene')
                elif pre_scene is None or pre_scene != self.scene():
                    pre_scene = self.scene()
                    self.back()
                else:
                    raise RecognizeError('Unanticipated scene')
            except RecognizeError as e:
                logger.warning(f'识别出了点小差错 qwq: {e}')
                self.recog.save_screencap('failure')
                retry_times -= 1
                self.sleep(3)
                continue
            except Exception as e:
                raise e
            retry_times = config.MAX_RETRYTIME

        if self.scene() != Scene.INDEX:
            raise StrategyError

    def back_to_reclamation_algorithm(self):
        self.recog.update()
        while self.find('index_terminal') is None:
            if self.scene() == Scene.UNKNOWN:
                self.device.exit('com.hypergryph.arknights')
            self.back_to_index()
        logger.info('导航至生息演算')
        self.tap_element('index_terminal', 0.5)
        self.tap((self.recog.w * 0.2, self.recog.h * 0.8), interval=0.5)

    def to_sss(self, sss_type, ec_type=3):
        self.recog.update()
        # 导航去保全派驻
        retry = 0
        self.back_to_index()
        self.tap_element('index_terminal', 0.5)
        self.tap((self.recog.w * 0.7, self.recog.h * 0.95), interval=0.2)
        self.tap((self.recog.w * 0.85, self.recog.h * 0.5), interval=0.2)
        if sss_type == 1:
            self.tap((self.recog.w * 0.2, self.recog.h * 0.3), interval=5)
        else:
            self.tap((self.recog.w * 0.4, self.recog.h * 0.6), interval=5)
        loop_count = 0
        ec_chosen_step = -99
        choose_team = False
        while self.find('end_sss', score=0.8) is None and loop_count < 8:
            if loop_count == ec_chosen_step + 2 or self.find('sss_team_up') is not None:
                choose_team = True
                logger.info("选择小队")
            elif self.find('choose_ss_ec') is not None and not choose_team:
                if ec_type == 1:
                    self.tap((self.recog.w * 0.3, self.recog.h * 0.5), interval=0.2)
                elif ec_type == 2:
                    self.tap((self.recog.w * 0.5, self.recog.h * 0.5), interval=0.2)
                else:
                    self.tap((self.recog.w * 0.7, self.recog.h * 0.5), interval=0.2)
                ec_chosen_step = loop_count
                logger.info(f"选定导能单元:{ec_type + 1}")
            self.tap((self.recog.w * 0.95, self.recog.h * 0.95), interval=(0.2 if not choose_team else 10))
            self.recog.update()
            loop_count += 1
        if loop_count == 8:
            return "保全派驻导航失败"

    def waiting_solver(self, scenes, wait_count=20, sleep_time=3):
        """需要等待的页面解决方法。触发超时重启会返回False
        """
        while wait_count > 0:
            self.sleep(sleep_time)
            if self.scene() != scenes and self.get_infra_scene() != scenes:
                return True
            wait_count -= 1
        logger.warning("同一等待界面等待超时，重启方舟。")
        self.device.exit(self.package_name)
        time.sleep(3)
        self.device.check_current_focus()
        return False

    def wait_for_scene(self, scene, method, wait_count=10, sleep_time=1):
        """等待某个页面载入
        """
        while wait_count > 0:
            self.sleep(sleep_time)
            if method == "get_infra_scene":
                if self.get_infra_scene() == scene:
                    return True
            elif method == "scene":
                if self.scene() == scene:
                    return True
            wait_count -= 1
        raise Exception("等待超时")

    # 邮件发送 EightyDollars
    def send_email(self, body='', subject='', subtype='plain', retry_times=3):
        if 'mail_enable' in self.email_config.keys() and self.email_config['mail_enable'] == 0:
            logger.info('邮件功能未开启')
            return

        msg = MIMEMultipart()
        msg.attach(MIMEText(body, subtype))
        msg['Subject'] = self.email_config['subject'] + subject
        msg['From'] = self.email_config['account']

        while retry_times > 0:
            try:
                s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
                # 登录邮箱
                s.login(self.email_config['account'], self.email_config['pass_code'])
                # 开始发送
                s.sendmail(self.email_config['account'], self.email_config['receipts'], msg.as_string())
                logger.info("邮件发送成功")
                break
            except Exception as e:
                logger.error("邮件发送失败")
                logger.exception(e)
                retry_times -= 1
                time.sleep(3)

```