# ArknightMower源码解析 7

# `arknights_mower/solvers/schedule.py`

这段代码的作用是定义了一个自定义的 `datetime` 类，用于在 Python 3600+ 中处理日期和时间。

具体来说，这段代码实现了以下功能：

1. 定义了一个名为 `DateTime` 的类，继承自 `datetime` 类。
2. 实现了一个名为 `datetime_ object` 的方法，该方法使用 `datetime` 类提供的 API，创建了一个 `datetime` 类的对象，并将其作为参数传递给 `__init__` 方法，用于初始化该对象。
3. 实现了一个名为 `datetime_ object_比较` 的方法，该方法使用 Python 的 `cmp_to_key` 函数比较两个 `datetime` 对象，并将结果存储在 `__key__` 方法中，用于在 `cmp_to_key` 函数中使用。
4. 导入了一系列自定义的函数和类，包括 `collections.abc` 中的 `Callable` 类型，用于创建可调用的函数；`ruamel.yaml` 中的 `yaml_object` 类，用于将 YAML 对象转换为 Python 对象；`the_same_day`、`Device`、`logger`、`operation_times`、`PriorityQueue` 和 `Recognizer` 类，用于实现日期和时间的处理和相关的优先级队列、计时器等功能。
5. 导入了一个 `config` 类，用于读取配置文件中的参数，并将其存储在类的 `__init__` 方法中。
6. 导入了一个 `the_same_day` 类，用于比较两个日期是否相同。
7. 导入了一个 `Device` 类，用于管理设备的连接和操作。
8. 导入了一个 `logger` 类，用于记录操作日志。
9. 导入了一个 `operation_times` 类，用于记录操作的次数，并支持对次数进行统计。
10. 导入了一个 `PriorityQueue` 类，用于实现优先级队列。
11. 导入了一个 `Recognizer` 类，用于实现语音识别功能。
12. 在 `utils.datetime` 类中定义了一系列处理日期和时间的函数，包括 `the_same_day`、`format_date`、`convert_to_date`、`convert_to_time` 等，用于实现对日期和时间的处理。
13. 在 `utils.device` 类中定义了一系列管理设备连接和操作的函数，包括 `connect_device`、`disconnect_device` 等，用于实现对设备的操作。
14. 在 `utils.logger` 类中定义了一系列记录日志的函数，包括 `log_info`、`log_warning`、`log_error` 等，用于记录日志信息。
15. 在 `utils.param` 类中定义了一系列对参数进行操作的函数，包括 `operation_times`、`parse_operation_params` 等，用于实现对参数的操作。
16. 在 `utils.priority_queue` 类中定义了一系列实现优先级队列的函数，包括 `put`、`get`、`compare_to` 等，用于实现对优先级队列的操作。
17. 在 `utils.recognize` 类中定义了一系列使用语音识别功能的方法，包括 `recognize_speech` 等，用于实现对语音识别的功能。


```py
import datetime
import time
from collections.abc import Callable
from functools import cmp_to_key
from pathlib import Path

from ruamel.yaml import yaml_object

from ..utils import config
from ..utils.datetime import the_same_day
from ..utils.device import Device
from ..utils.log import logger
from ..utils.param import operation_times, parse_operation_params
from ..utils.priority_queue import PriorityQueue
from ..utils.recognize import Recognizer
```

这段代码是一个基于 Python 的程序，主要作用是定义了一个名为 `operation_one` 的函数，属于 `Operation` 类。

它使用了 `BaseSolver` 和 `ParamArgs` 类，从 `utils.solver` 包中获取了 `OpeSolver` 类，从 `utils.typealias` 包中获取了 `ParamArgs` 类，从 `utils.yaml` 包中获取了 `yaml` 类。

`task_priority` 变量定义了一个映射，将任务优先级映射到数字，数字越小越低优先级越高。

`operation_one` 函数接收两个参数，一个是参数 `args`，另一个是设备 `device`。函数内部首先调用 `OpeSolver` 类的 `run` 方法，传递参数 `level`、`device`、`level`、`arg1` 和 `arg2`，以及两个可选的参数 `eliminate` 和 `force_full_battle`。函数内部使用解析出的参数调用 `run` 方法，并获取执行计划的最后一步，即 `plan[1]`，如果该值为 0，说明任务完成，返回 `True`，否则继续执行。

函数内部还使用 `for` 循环遍历执行计划中的所有计划，判断每个计划的优先级是否高于当前任务。如果当前计划高于所有优先级，则返回 `False`。


```py
from ..utils.solver import BaseSolver
from ..utils.typealias import ParamArgs
from ..utils.yaml import yaml
from .operation import OpeSolver

task_priority = {'base': 0, 'recruit': 1, 'mail': 2,
                 'credit': 3, 'shop': 4, 'mission': 5, 'operation': 6}


class ScheduleLogError(ValueError):
    """ Schedule log 文件解析错误 """


def operation_one(args: ParamArgs = [], device: Device = None) -> bool:
    """
    只为 schedule 模块使用的单次作战操作
    目前不支持使用源石和体力药

    返回值表示该次作战是否成功
    完成剿灭不算成功
    """
    level, _, _, _, eliminate = parse_operation_params(args)
    remain_plan = OpeSolver(device).run(level, 1, 0, 0, eliminate)
    for plan in remain_plan:
        if plan[1] != 0:
            return False
    return True


```

以上是一个Python语言写的Armed Server的Finish和Operation类，其中包含了Finish和Operation的相关逻辑。


```py
@yaml_object(yaml)
class Task(object):
    """
    单个任务
    """

    def __init__(self, tag: str = '', cmd: Callable = None, args: ParamArgs = [], device: Device = None):
        self.cmd = cmd
        self.cmd_args = args
        self.tag = tag
        self.last_run = None
        self.idx = None
        self.pending = False
        self.total = 1
        self.finish = 0
        self.device = device

        # per_hour 任务的第一次执行将在启动脚本后的一个小时之后
        if tag == 'per_hour':
            self.last_run = datetime.datetime.now()
        if cmd.__name__ == 'operation':
            self.total = operation_times(args)
            assert self.total != 0

    @classmethod
    def to_yaml(cls, representer, data):
        last_run = ''
        if data.last_run is not None:
            last_run = data.last_run.strftime('%Y-%m-%d %H:%M:%S')
        return representer.represent_mapping('task',
                                             {'tag': data.tag,
                                              'cmd': data.cmd.__name__,
                                              'cmd_args': data.cmd_args,
                                              'last_run': last_run,
                                              'idx': data.idx,
                                              'pending': data.pending,
                                              'total': data.total,
                                              'finish': data.finish})

    def __lt__(self, other):
        if task_priority[self.cmd.__name__] != task_priority[other.cmd.__name__]:
            return task_priority[self.cmd.__name__] < task_priority[other.cmd.__name__]
        return self.idx < other.idx

    def load(self, last_run: str = '', idx: int = 0, pending: bool = False, total: int = 1, finish: int = 0):
        if last_run == '':
            self.last_run = None
        else:
            self.last_run = datetime.datetime.strptime(
                last_run, '%Y-%m-%d %H:%M:%S')
        self.idx = idx
        self.pending = pending
        self.total = total
        self.finish = finish

    def reset(self):
        if self.tag != 'per_hour':
            self.last_run = None
        self.pending = False
        self.finish = 0

    def set_idx(self, idx: int = None):
        self.idx = idx

    def start_up(self) -> bool:
        return self.tag == 'start_up'

    def need_run(self, now: datetime.datetime = datetime.datetime.now()) -> bool:
        if self.pending:
            return False
        if self.start_up():
            if self.last_run is not None:
                return False
            self.pending = True
            self.last_run = now
            return True
        if self.tag[:4] == 'day_':
            # 同一天 and 跑过了
            if self.last_run is not None and the_same_day(now, self.last_run):
                return False
            # 还没到时间
            if now.strftime('%H:%M') < self.tag.replace('_', ':')[4:]:
                return False
            self.pending = True
            self.last_run = now
            return True
        if self.tag == 'per_hour':
            if self.last_run + datetime.timedelta(hours=1) <= now:
                self.pending = True
                self.last_run = now
                return True
            return False
        return False

    def run(self) -> bool:
        logger.info(f'task: {self.cmd.__name__} {self.cmd_args}')
        if self.cmd.__name__ == 'operation':
            if operation_one(self.cmd_args, self.device):
                self.finish += 1
                if self.finish == self.total:
                    self.finish = 0
                    self.pending = False
                    return True
            return False
        self.cmd(self.cmd_args, self.device)
        self.pending = False
        return True


```

这段代码实现了一个 Task Scheduler，可以安排任务的计划、执行和卸载。

具体来说，它实现了以下功能：

1. 定义了 Task 类，其中包含标签（tag）、命令（cmd）和参数（args），以及设备（device）属性。
2. 实现了 add_task 方法，用于添加任务。
3. 实现了 per_run 方法，用于在运行时处理任务的优先级。
4. 实现了 run 方法，用于启动计划、执行任务和更新标签。
5. 实现了 new_day 方法，用于在每天开始时运行所有任务一次。
6. 实现了 transition 方法，用于在每天结束时切换到下一天的任务。
7. 实现了 dump_to_disk 方法，用于将任务数据保存到磁盘。

此外，还实现了森川流的算法，用于生成优先级。


```py
def cmp_for_init(task_a: Task = None, task_b: Task = None) -> int:
    if task_a.start_up() and task_b.start_up():
        return 0

    if task_a.start_up():
        return -1

    if task_b.start_up():
        return 1
    return 0


@yaml_object(yaml)
class ScheduleSolver(BaseSolver):
    """
    按照计划定时、自动完成任务
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)
        self.tasks = []
        self.pending_list = PriorityQueue()
        self.device = device
        self.last_run = None
        self.schedule_log_path = Path(
            config.LOGFILE_PATH).joinpath('schedule.log')

    @classmethod
    def to_yaml(cls, representer, data):
        return representer.represent_mapping('Schedule', {'last_run': data.last_run.strftime('%Y-%m-%d %H:%M:%S'),
                                                          'tasks': data.tasks})

    def dump_to_disk(self):
        with self.schedule_log_path.open('w', encoding='utf8') as f:
            yaml.dump(self, f)
        logger.info('计划已存档')

    def load_from_disk(self, cmd_list=None, matcher: Callable = None) -> bool:
        if cmd_list is None:
            cmd_list = []
        try:
            with self.schedule_log_path.open('r', encoding='utf8') as f:
                data = yaml.load(f)
            self.last_run = datetime.datetime.strptime(
                data['last_run'], '%Y-%m-%d %H:%M:%S')
            for task in data['tasks']:
                cmd = matcher(task['cmd'], cmd_list)
                if cmd is None:
                    raise ScheduleLogError
                new_task = Task(
                    task['tag'], cmd, task['cmd_args'], self.device
                )
                new_task.load(
                    task['last_run'], task['idx'], task['pending'], task['total'], task['finish']
                )
                self.tasks.append(new_task)
                if new_task.pending:
                    self.pending_list.push(new_task)
        except Exception:
            return False
        logger.info('发现中断的计划，将继续执行')
        return True

    def add_task(self, tag: str = '', cmd: Callable = None, args: ParamArgs = []):
        task = Task(tag, cmd, args, self.device)
        self.tasks.append(task)

    def per_run(self):
        """
        这里是为了处理优先级相同的情况，对于优先级相同时，我们依次考虑：
        1. start_up 优先执行
        2. 按照配置文件的顺序决定先后顺序

        sort 是稳定排序，详见:
        https://docs.python.org/3/library/functions.html#sorted
        """
        self.tasks.sort(key=cmp_to_key(cmp_for_init))
        for idx, task in enumerate(self.tasks):
            task.set_idx(idx)

    def run(self):
        logger.info('Start: 计划')

        super().run()

    def new_day(self):
        for task in self.tasks:
            task.reset()
        self.pending_list = PriorityQueue()

    def transition(self) -> None:
        while True:
            now = datetime.datetime.now()
            if self.last_run is not None and the_same_day(self.last_run, now) is False:
                self.new_day()
            self.last_run = now
            for task in self.tasks:
                if task.need_run(now):
                    self.pending_list.push(task)

            task = self.pending_list.pop()
            if task is not None and task.run() is False:
                self.pending_list.push(task)

            self.dump_to_disk()
            time.sleep(60)

```

# `arknights_mower/solvers/shop.py`

This is a class that inherits from the AbstractRecognizer. It inherits the functionality of the AbstractRecognizer, but with some added features and changes in behavior.

The Recognizer class has a `__call__` method that initializes the instance and starts the recognition. The recognition is triggered when the `snapshot` attribute is accessed, and the instance uses the `SegmentationCanvas` to analyze the image data.

The Scene class defines the different scenes the Recognizer can be in, and每种场景有不同的 logic. For example, the `Scene.LOADING` scene means the image is being loaded from disk, while the `Scene.CONNECTING` scene means the image is being connected to a remote device.

The `tap_element` and `get_navigation` methods are added to the scene hierarchy, which allows the Recognizer to interact with the game environment.

The `shop_credit` method is also added to the scene hierarchy, which allows the Recognizer to buy items from the game store. This method checks if the user has enough currency, and then buys the item and returns the success or failure message.

Overall, this class implements the basic functionality of an image recognition system, which recognizes items in an image by highlighting the corresponding region in the image data. It also adds some additional features to the scene hierarchy and adds the ability to connect to a remote device.


```py
from __future__ import annotations

from ..data import shop_items
from ..ocr import ocr_rectify, ocrhandle
from ..utils import segment
from ..utils.device import Device
from ..utils.image import scope2slice
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Scene
from ..utils.solver import BaseSolver, Recognizer


class ShopSolver(BaseSolver):
    """
    自动使用信用点购买物资
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    def run(self, priority: list[str] = None) -> None:
        """
        :param priority: list[str], 使用信用点购买东西的优先级, 若无指定则默认购买第一件可购买的物品
        """
        self.priority = priority
        self.buying = None
        logger.info('Start: 商店')
        logger.info('购买期望：%s' % priority if priority else '无，购买到信用点用完为止')
        super().run()

    def transition(self) -> bool:
        if self.scene() == Scene.INDEX:
            self.tap_element('index_shop')
        elif self.scene() == Scene.SHOP_OTHERS:
            self.tap_element('shop_credit_2')
        elif self.scene() == Scene.SHOP_CREDIT:
            collect = self.find('shop_collect')
            if collect is not None:
                self.tap(collect)
            else:
                return self.shop_credit()
        elif self.scene() == Scene.SHOP_CREDIT_CONFIRM:
            if self.find('shop_credit_not_enough') is None:
                self.tap_element('shop_cart')
            elif len(self.priority) > 0:
                # 移除优先级中买不起的物品
                self.priority.remove(self.buying) 
                logger.info('信用点不足，放弃购买%s，看看别的...' % self.buying)
                self.back()
            else:
                return True
        elif self.scene() == Scene.SHOP_ASSIST:
            self.back()
        elif self.scene() == Scene.MATERIEL:
            self.tap_element('materiel_ico')
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.get_navigation():
            self.tap_element('nav_shop')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

    def shop_credit(self) -> bool:
        """ 购买物品逻辑 """
        segments = segment.credit(self.recog.img)
        valid = []
        for seg in segments:
            if self.find('shop_sold', scope=seg) is None:
                scope = (seg[0], (seg[1][0], seg[0][1] + (seg[1][1]-seg[0][1])//4))
                img = self.recog.img[scope2slice(scope)]
                ocr = ocrhandle.predict(img)
                if len(ocr) == 0:
                    raise RecognizeError
                ocr = ocr[0]
                if ocr[1] not in shop_items:
                    ocr[1] = ocr_rectify(img, ocr, shop_items, '物品名称')
                valid.append((seg, ocr[1]))
        logger.info(f'商店内可购买的物品：{[x[1] for x in valid]}')
        if len(valid) == 0:
            return True
        priority = self.priority
        if priority is not None:
            valid.sort(
                key=lambda x: 9999 if x[1] not in priority else priority.index(x[1]))
            if valid[0][1] not in priority:
                return True
        logger.info(f'实际购买顺序：{[x[1] for x in valid]}')
        self.buying = valid[0][1]
        self.tap(valid[0][0], interval=3)

```

# `arknights_mower/solvers/__init__.py`

这段代码定义了7个不同的求解器类，包括基于构建的构造求解器、信证求解器、邮件求解器、任务求解器、操作求解器、招聘求解器和计划求解器。这些求解器类可以用来解决各种不同类型的数学问题。


```py
from .base_construct import BaseConstructSolver
from .credit import CreditSolver
from .mail import MailSolver
from .mission import MissionSolver
from .operation import OpeSolver
from .recruit import RecruitSolver
from .schedule import ScheduleSolver
from .shop import ShopSolver

```

# Templates

## config.yaml

配置文件模板


# `arknights_mower/utils/asst.py`

该代码使用了多面骰，即CTypes库，用于在操作系统上执行Python字节码。具体而言，它们导入了以下模块：

- os：用于从操作系统获取文件路径等信息。
- pathlib：用于处理文件路径相关的任务。
- platform：用于获取当前操作系统相关的信息。
- json：用于解析JSON格式的数据。

接着，它们定义了一个名为InstanceOptionType的枚举类型，该类型有 touch_type 和 deployment_with_pause 两个成员变量，分别表示是否触摸屏幕和是否在部署时暂停执行。

然后，它们导入了ctypes库的支点，即CTypes.py库，以及创建了一个名为 Instance 的函数实例。该函数可以执行以下操作：

- 在操作系统中打开一个可执行文件，并设置触摸屏幕类型为实例选项类型中的 touch_type 对应的值。
- 在操作系统中打开一个可执行文件，并设置是否在部署时暂停执行为实例选项类型中的 deployment_with_pause 对应的值。

这些操作的具体实现可能因操作系统而异，但它们是在给定的系统上执行动作，以获取或设置特定实例选项类型的值。


```py
import ctypes
import os
import pathlib
import platform
import json

from typing import Union, Dict, List, Any, Type, Optional
from enum import Enum, IntEnum, unique, auto

JSON = Union[Dict[str, Any], List[Any], int, str, float, bool, Type[None]]


class InstanceOptionType(IntEnum):
    touch_type = 2
    deployment_with_pause = 3


```

It looks like this is a Python class that has some utility functions for working withAsyncio objects. Here is a summary of what this class does:

* Asst.__lib.AsstSetInstanceOption.restype: returns a ctypes.c_void_p pointer that is a void pointer to an AsstInstanceOption struct.
* Asst.__lib.AsstSetInstanceOption.argtypes: returns a tuple of the required argument types for the AsstInstanceOption struct. The first return type is ctypes.c_void_p, which means that the struct pointer should be a ctypes.c_void_p value. The second return type is an integer, and the third return type is a ctypes.c_int, which means that the struct should have an integer field. The fourth return type is a ctypes.c_char_p, which means that the struct should have a char field.
* Asst.__lib.AsstConnect.restype: returns a ctypes.c_bool pointer.
* Asst.__lib.AsstConnect.argtypes: returns a tuple of the required argument types for the AsstConnect struct. The first return type is ctypes.c_void_p, which means that the connect function should return a ctypes.c_void_p value. The second return type is a ctypes.c_char_p, which means that the connect function should return a ctypes.c_char_p value. The third return type is a ctypes.c_char_p, which means that the connect function should return a ctypes.c_char_p value. The fourth return type is an integer, and the fifth return type is ctypes.c_int, which means that the connect function should return an integer value.
* Asst.__lib.AsstAppendTask.restype: returns an integer value.
* Asst.__lib.AsstAppendTask.argtypes: returns a tuple of the required argument types for the AsstAppendTask struct. The first return type is an integer, and the second return type is a ctypes.c_void_p, which means that the appending task should return a ctypes.c_void_p value.
* Asst.__lib.AsstSetTaskParams.restype: returns a ctypes.c_bool pointer.
* Asst.__lib.AsstSetTaskParams.argtypes: returns a tuple of the required argument types for the AsstTaskParams struct. The first return type is ctypes.c_void_p, which means that the setting


```py
class Asst:
    CallBackType = ctypes.CFUNCTYPE(
        None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    """
    回调函数，使用实例可参照 my_callback
    :params:
        ``param1 message``: 消息类型
        ``param2 details``: json string
        ``param3 arg``:     自定义参数
    """

    @staticmethod
    def load(path: Union[pathlib.Path, str], incremental_path: Optional[Union[pathlib.Path, str]] = None, user_dir: Optional[Union[pathlib.Path, str]] = None) -> bool:
        """
        加载 dll 及资源
        :params:
            ``path``:    DLL及资源所在文件夹路径
            ``incremental_path``:   增量资源所在文件夹路径
            ``user_dir``:   用户数据（日志、调试图片等）写入文件夹路径
        """

        platform_values = {
            'windows': {
                'libpath': 'MaaCore.dll',
                'environ_var': 'PATH'
            },
            'darwin': {
                'libpath': 'libMaaCore.dylib',
                'environ_var': 'DYLD_LIBRARY_PATH'
            },
            'linux': {
                'libpath': 'libMaaCore.so',
                'environ_var': 'LD_LIBRARY_PATH'
            }
        }
        lib_import_func = None

        platform_type = platform.system().lower()
        if platform_type == 'windows':
            lib_import_func = ctypes.WinDLL
        else:
            lib_import_func = ctypes.CDLL

        Asst.__libpath = pathlib.Path(path) / platform_values[platform_type]['libpath']
        try:
            os.environ[platform_values[platform_type]['environ_var']] += os.pathsep + str(path)
        except KeyError:
            os.environ[platform_values[platform_type]['environ_var']] = os.pathsep + str(path)
        Asst.__lib = lib_import_func(str(Asst.__libpath))
        Asst.__set_lib_properties()

        ret: bool = True
        if user_dir:
            ret &= Asst.__lib.AsstSetUserDir(str(user_dir).encode('utf-8'))

        ret &= Asst.__lib.AsstLoadResource(str(path).encode('utf-8'))
        if incremental_path:
            ret &= Asst.__lib.AsstLoadResource(
                str(incremental_path).encode('utf-8'))

        return ret

    def __init__(self, callback: CallBackType = None, arg=None):
        """
        :params:
            ``callback``:   回调函数
            ``arg``:        自定义参数
        """

        if callback:
            self.__ptr = Asst.__lib.AsstCreateEx(callback, arg)
        else:
            self.__ptr = Asst.__lib.AsstCreate()

    def __del__(self):
        Asst.__lib.AsstDestroy(self.__ptr)
        self.__ptr = None

    def set_instance_option(self, option_type: InstanceOptionType, option_value: str):
        """
        设置额外配置
        参见${MaaAssistantArknights}/src/MaaCore/Assistant.cpp#set_instance_option
        :params:
            ``externa_config``: 额外配置类型
            ``config_value``:   额外配置的值
        :return: 是否设置成功
        """
        return Asst.__lib.AsstSetInstanceOption(self.__ptr,
                                                int(option_type), option_value.encode('utf-8'))


    def connect(self, adb_path: str, address: str, config: str = 'General'):
        """
        连接设备
        :params:
            ``adb_path``:       adb 程序的路径
            ``address``:        adb 地址+端口
            ``config``:         adb 配置，可参考 resource/config.json
        :return: 是否连接成功
        """
        return Asst.__lib.AsstConnect(self.__ptr,
                                      adb_path.encode('utf-8'), address.encode('utf-8'), config.encode('utf-8'))

    TaskId = int

    def append_task(self, type_name: str, params: JSON = {}) -> TaskId:
        """
        添加任务
        :params:
            ``type_name``:  任务类型，请参考 docs/集成文档.md
            ``params``:     任务参数，请参考 docs/集成文档.md
        :return: 任务 ID, 可用于 set_task_params 接口
        """
        return Asst.__lib.AsstAppendTask(self.__ptr, type_name.encode('utf-8'), json.dumps(params, ensure_ascii=False).encode('utf-8'))

    def set_task_params(self, task_id: TaskId, params: JSON) -> bool:
        """
        动态设置任务参数
        :params:
            ``task_id``:  任务 ID, 使用 append_task 接口的返回值
            ``params``:   任务参数，同 append_task 接口，请参考 docs/集成文档.md
        :return: 是否成功
        """
        return Asst.__lib.AsstSetTaskParams(self.__ptr, task_id, json.dumps(params, ensure_ascii=False).encode('utf-8'))

    def start(self) -> bool:
        """
        开始任务
        :return: 是否成功
        """
        return Asst.__lib.AsstStart(self.__ptr)

    def stop(self) -> bool:
        """
        停止并清空所有任务
        :return: 是否成功
        """
        return Asst.__lib.AsstStop(self.__ptr)

    def running(self) -> bool:
        """
        是否正在运行
        :return: 是否正在运行
        """
        return Asst.__lib.AsstRunning(self.__ptr)

    @staticmethod
    def log(level: str, message: str) -> None:
        '''
        打印日志
        :params:
            ``level``:      日志等级标签
            ``message``:    日志内容
        '''

        Asst.__lib.AsstLog(level.encode('utf-8'), message.encode('utf-8'))

    def get_version(self) -> str:
        """
        获取DLL版本号
        : return: 版本号
        """
        return Asst.__lib.AsstGetVersion().decode('utf-8')

    @staticmethod
    def __set_lib_properties():
        Asst.__lib.AsstSetUserDir.restype = ctypes.c_bool
        Asst.__lib.AsstSetUserDir.argtypes = (
            ctypes.c_char_p,)

        Asst.__lib.AsstLoadResource.restype = ctypes.c_bool
        Asst.__lib.AsstLoadResource.argtypes = (
            ctypes.c_char_p,)

        Asst.__lib.AsstCreate.restype = ctypes.c_void_p
        Asst.__lib.AsstCreate.argtypes = ()

        Asst.__lib.AsstCreateEx.restype = ctypes.c_void_p
        Asst.__lib.AsstCreateEx.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p,)

        Asst.__lib.AsstDestroy.argtypes = (ctypes.c_void_p,)

        Asst.__lib.AsstSetInstanceOption.restype = ctypes.c_bool
        Asst.__lib.AsstSetInstanceOption.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,)

        Asst.__lib.AsstConnect.restype = ctypes.c_bool
        Asst.__lib.AsstConnect.argtypes = (
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,)

        Asst.__lib.AsstAppendTask.restype = ctypes.c_int
        Asst.__lib.AsstAppendTask.argtypes = (
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)

        Asst.__lib.AsstSetTaskParams.restype = ctypes.c_bool
        Asst.__lib.AsstSetTaskParams.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)

        Asst.__lib.AsstStart.restype = ctypes.c_bool
        Asst.__lib.AsstStart.argtypes = (ctypes.c_void_p,)

        Asst.__lib.AsstStop.restype = ctypes.c_bool
        Asst.__lib.AsstStop.argtypes = (ctypes.c_void_p,)

        Asst.__lib.AsstRunning.restype = ctypes.c_bool
        Asst.__lib.AsstRunning.argtypes = (ctypes.c_void_p,)

        Asst.__lib.AsstGetVersion.restype = ctypes.c_char_p

        Asst.__lib.AsstLog.restype = None
        Asst.__lib.AsstLog.argtypes = (
            ctypes.c_char_p, ctypes.c_char_p)


```

这段代码定义了一个名为Message的枚举类型，并创建了一个枚举类型类。该类定义了回调消息的各种枚举值，如InternalError、ConnectionInfo、AllTasksCompleted、TaskChainError、TaskChainStart、TaskChainCompleted、TaskChainExtraInfo、TaskChainStopped、SubTaskError和SubTaskStart等。在这些枚举值中，有一些具有特殊的意义，如TaskChainError、TaskChainStart和TaskChainCompleted等，它们被设置为auto()，意味着它们将自动在代码中进行初始化。


```py
@unique
class Message(Enum):
    """
    回调消息
    请参考 docs/回调消息.md
    """
    InternalError = 0

    InitFailed = auto()

    ConnectionInfo = auto()

    AllTasksCompleted = auto()

    TaskChainError = 10000

    TaskChainStart = auto()

    TaskChainCompleted = auto()

    TaskChainExtraInfo = auto()

    TaskChainStopped = auto()

    SubTaskError = 20000

    SubTaskStart = auto()

    SubTaskCompleted = auto()

    SubTaskExtraInfo = auto()

    SubTaskStopped = auto()
```

# `arknights_mower/utils/character_recognize.py`

这段代码是一个Python文件中的函数，主要作用是定义了一个名为“__future__”的模块，该模块引入了两个数据类型：在未来和属性。具体来说，这个模块中定义了两个函数，分别是：from __future__ import annotations 和 from __future__ import sq厂。

1. from __future__ import annotations：该函数允许在函数内部使用未来数据类型。通过这个函数，可以指定将来会出现的类和函数，并对其进行定义。这个函数通常用于当前文件中的某个函数，但并不包含任何数据。

2. from __future__ import sq厂：该函数允许使用Python 2.x中定义的函数和数据类型。通过这个函数，可以指定现在就有的类和函数，并对其进行定义。这个函数通常用于当前文件中的一个类，但并不包含任何数据。

通过这两个函数，我们可以看出代码的用途是将未来的代码和定义的数据类型引入到当前文件中，以便我们可以使用这些数据类型和方法。


```py
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
```

这段代码定义了两个函数和一个名为“poly_center”的函数指针。函数指针“poly_center”接受一个多边形（poly）作为参数，并返回该多边形的几何中心点。函数“in_poly”接受一个多边形和一个点作为参数，并返回一个布尔值，表示该点是否在传入的多边形内。

此外，代码还定义了一个名为“char_map”的 dictionary，用于存储已经识别出的所有文字。定义了一个名为“agent_sorted”的 list，用于存储经过训练的代理程序列表。该列表按照其处理文字的时长(即其处理时间)从小到大排序。

最后，定义了一个名为“origin_kp”的变量作为起始位置，一个名为“origin_des”的变量作为起始描述(即描述该位置的起始状态)，以及一个未定义的“origin_update”函数。


```py
from .recognize import RecognizeError
from ..ocr import ocrhandle


def poly_center(poly):
    return (np.average([x[0] for x in poly]), np.average([x[1] for x in poly]))


def in_poly(poly, p):
    return poly[0, 0] <= p[0] <= poly[2, 0] and poly[0, 1] <= p[1] <= poly[2, 1]


char_map = {}
agent_sorted = sorted(deepcopy(agent_list), key=len)
origin = origin_kp = origin_des = None

```

这段代码的作用是创建一个基于FLANN算法的SIFT特征点检测器，并设置一些参数。

首先，FLANN_INDEX_KDTREE和GOOD_DISTANCE_LIMIT是FLANN算法的参数，用于设置FLANN树搜索的节点数量和点之间最短的距离。

然后，SIFT是一个用于点检测的FLANN算法实例，已经使用FLANN_INDEX_KDTREE和GOOD_DISTANCE_LIMIT参数训练好。

接下来的代码是在origin变量上执行的，将origin图像转换为灰度图像并执行SIFT特征点检测。如果之前已经执行过该操作，则直接使用之前的结果。然后，执行一些计算，将检测到的特征点保存在origin_kp和origin_des变量中。

最后，使用这些特征点来训练FLANN算法。


```py
FLANN_INDEX_KDTREE = 0
GOOD_DISTANCE_LIMIT = 0.7
SIFT = cv2.SIFT_create()


def agent_sift_init():
    global origin, origin_kp, origin_des
    if origin is None:
        logger.debug('agent_sift_init')

        height = width = 2000
        lnum = 25
        cell = height // lnum

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)

        font = ImageFont.truetype(
            f'{__rootdir__}/fonts/SourceHanSansSC-Bold.otf', size=30, encoding='utf-8'
        )
        chars = sorted(list(set(''.join([x for x in agent_list]))))
        assert len(chars) <= (lnum - 2) * (lnum - 2)

        for idx, char in enumerate(chars):
            x, y = idx % (lnum - 2) + 1, idx // (lnum - 2) + 1
            char_map[(x, y)] = char
            ImageDraw.Draw(img).text(
                (x * cell, y * cell), char, (255, 255, 255), font=font
            )

        origin = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        origin_kp, origin_des = SIFT.detectAndCompute(origin, None)


```

The function `segment_recog` performs image segmentation using the SIFT algorithm and the Flann-based match algorithm. It takes an input image and a resolution, and returns the segmentation mask for the specified region.

The `cv2.cvtColor` function converts the input image from the RGB color space to the grayscale color space.

The `cv2.resize` function resizes the input image to a multiple of the specified resolution.

The `SIFT.detectAndCompute` function detects and computes feature points in the resized image using the SIFT algorithm.

The `cv2.resize` function resizes the input image to a multiple of the specified resolution.

The `cv2.FlannBasedMatcher` function builds a Flann-based match algorithm to compute匹配 scores between the input image and the template image.

The `cv2.drawMatches` function draws match rectangles around the detected matches.

The function returns the segmentation mask for the specified region.


```py
def sift_recog(query, resolution, draw=False,bigfont = False):
    """
    使用 SIFT 提取特征点识别干员名称
    """
    agent_sift_init()
    # 大号字体修改参数
    if bigfont:
        SIFT = cv2.SIFT_create(
            contrastThreshold=0.1,
            edgeThreshold=20
        )
    else:
        SIFT = cv2.SIFT_create()
    query = cv2.cvtColor(np.array(query), cv2.COLOR_RGB2GRAY)
    # the height & width of query image
    height, width = query.shape

    multi = 2 * (resolution / 1080)
    query = cv2.resize(query, (int(width * multi), int(height * multi)))
    query_kp, query_des = SIFT.detectAndCompute(query, None)

    # build FlannBasedMatcher
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_des, origin_des, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for x, y in matches:
        if x.distance < GOOD_DISTANCE_LIMIT * y.distance:
            good.append(x)

    if draw:
        result = cv2.drawMatches(query, query_kp, origin, origin_kp, good, None)
        plt.imshow(result, 'gray')
        plt.show()

    count = {}

    for x in good:
        x, y = origin_kp[x.trainIdx].pt
        c = char_map[(int(x) // 80, int(y) // 80)]
        count[c] = count.get(c, 0) + 1

    best = None
    best_score = 0
    for x in agent_sorted:
        score = 0
        for c in set(x):
            score += count.get(c, -1)
        if score > best_score:
            best = x
            best_score = score

    logger.debug(f'segment.sift_recog: {count}, {best}')

    return best


```

This is a Python script that performs character recognition using the Tesseract OCR engine. It performs the following tasks:

1. Listen to an AI bot (Line ReQparse, current version), which is capable of recognizing characters in a text.
2. For each character, it checks if it is recognized by the AI bot. If it is not recognized, it sends a "failure" message to the AI bot. If it is recognized, it sends a "success" message to the AI bot.
3. If the AI bot recognizes a character, it sends the character and its position to the AI bot.
4. If the AI bot does not recognize anything, it will print "No new text" to indicate that there is no input.
5. If the AI bot recognizes multiple characters, it will send the recognized characters to the AI bot.
6. If the AI bot recognizes a character but the character is not in its vocabulary, it will print a message and continue with the next character.
7. If the AI bot recognizes multiple characters that are in its vocabulary, it will add them to its recognized characters.

The script uses the `RecognizeError` class to handle errors, such as when the AI bot does not recognize a character. It also uses the `cv2` library to display images.


```py
def agent(img, draw=False):
    """
    识别干员总览界面的干员名称
    """
    try:
        height, width, _ = img.shape
        resolution = height
        left, right = 0, width

        # 异形屏适配
        while np.max(img[:, right - 1]) < 100:
            right -= 1
        while np.max(img[:, left]) < 100:
            left += 1

        # 去除左侧干员详情
        x0 = left + 1
        while not (
            img[height - 10, x0 - 1, 0] > img[height - 10, x0, 0] + 10
            and abs(int(img[height - 10, x0, 0]) - int(img[height - 10, x0 + 1, 0])) < 5
        ):
            x0 += 1

        # 获取分割结果
        ret, ocr = segment.agent(img, draw)

        # 确定位置后开始精确识别
        ret_succ = []
        ret_fail = []
        ret_agent = []
        for poly in ret:
            found_ocr, fx = None, 0
            for x in ocr:
                cx, cy = poly_center(x[2])
                if in_poly(poly, (cx + x0, cy)) and cx > fx:
                    fx = cx
                    found_ocr = x
            __img = img[poly[0, 1]: poly[2, 1], poly[0, 0]: poly[2, 0]]
            try:
                if found_ocr is not None:
                    x = found_ocr
                    if len(x[1]) == 3 and x[1][0] == "休" and x[1][2] == "斯":
                        x[1] = "休谟斯"
                    if x[1] in agent_list and x[1] not in ['砾', '陈']:  # ocr 经常会把这两个搞错
                        ret_agent.append(x[1])
                        ret_succ.append(poly)
                        continue
                    res = sift_recog(__img, resolution, draw)
                    if (res is not None) and res in agent_list:
                        ret_agent.append(res)
                        ret_succ.append(poly)
                        continue
                    logger.debug(
                        f'干员名称识别异常：{x[1]} 为不存在的数据，请报告至 https://github.com/Konano/arknights-mower/issues'
                    )
                    saveimg(__img, 'failure_agent')
                    raise Exception(x[1])
                else:
                    if 80 <= np.min(__img):
                        continue
                    res = sift_recog(__img, resolution, draw)
                    if res is not None:
                        ret_agent.append(res)
                        ret_succ.append(poly)
                        continue
                    logger.warning(f'干员名称识别异常：区域 {poly.tolist()}')
                    saveimg(__img, 'failure_agent')
                    raise Exception("启动 Plan B")
                ret_fail.append(poly)
                raise Exception("启动 Plan B")
            except Exception as e:
                # 大哥不行了，二哥上！
                _msg = str(e)
                ret_fail.append(poly)
                if "Plan B" not in _msg:
                    if _msg in ocr_error.keys():
                        name = ocr_error[_msg]
                    elif "Off" in _msg:
                        name = 'U-Official'
                    else:
                        continue
                    ret_agent.append(name)
                    ret_succ.append(poly)
                    continue
        if len(ret_fail):
            saveimg(img, 'failure')
            if draw:
                __img = img.copy()
                cv2.polylines(__img, ret_fail, True, (255, 0, 0), 3, cv2.LINE_AA)
                plt.imshow(__img)
                plt.show()

        logger.debug(f'character_recognize.agent: {ret_agent}')
        logger.debug(f'character_recognize.agent: {[x.tolist() for x in ret]}')
        return list(zip(ret_agent, ret_succ))

    except Exception as e:
        logger.debug(traceback.format_exc())
        saveimg(img, 'failure_agent')
        raise RecognizeError(e)

```

这段代码定义了一个名为`agent_name`的函数，接受一个名为`__img`的图像对象，以及一个名为`height`的图像高度参数和一个名为`draw`的布尔参数，用于绘制图像时是否保留边框。

函数的主要作用是通过对图像进行处理，提取出物体的检测结果，然后根据设定的阈值和分类器匹配，最终输出物体的名称。

具体来说，函数首先将输入的图像转换为灰度图像，然后对图像进行尺寸扩充，接着对图像进行OCR(光学字符识别)处理，如果识别结果有输出，则执行以下操作：

1. 如果图像中检测到物体，并且该物体不在预设的`agent_list`中，则执行以下操作：

  1.1. 尝试使用SIFT(基于SIFT特征的图像匹配算法)对图像进行匹配，如果匹配结果仍然存在，则继续检测物体。

  1.2. 如果SIFT匹配结果不存在或者不是物体，则画出物体的边界框并添加到图像中。

  1.3. 如果匹配结果仍然不存在，则返回错误信息并保存图像。

如果函数在执行过程中出现任何异常，例如图像处理失败、OCR识别出错、SIFT匹配失败等，则相应的日志信息也会记录下来。


```py
def agent_name(__img, height, draw: bool = False):
    query = cv2.cvtColor(np.array(__img), cv2.COLOR_RGB2GRAY)
    h, w= query.shape
    dim = (w*4, h*4)
    # resize image
    resized = cv2.resize(__img, dim, interpolation=cv2.INTER_AREA)
    ocr = ocrhandle.predict(resized)
    name = ''
    try:
        if len(ocr) > 0 and ocr[0][1] in agent_list and ocr[0][1] not in ['砾', '陈']:
            name = ocr[0][1]
        elif len(ocr) > 0 and ocr[0][1] in ocr_error.keys():
            name = ocr_error[ocr[0][1]]
        else:
            res = sift_recog(__img, height, draw,bigfont=True)
            if (res is not None) and res in agent_list:
                name = res
            else:
                raise Exception(f"识别错误: {res}")
    except Exception as e:
        if len(ocr)>0:
            logger.warning(e)
            logger.warning(ocr[0][1])
            saveimg(__img, 'failure_agent')
    return name

```

# `arknights_mower/utils/conf.py`

这段代码是一个Python脚本，主要作用是读取一个.yml格式的配置文件，将其解析为字典，并将字典存储为一个.yml格式的临时配置文件。

具体来说，代码首先导入了os、json、pathlib和ruamel.yaml库，然后定义了一个名为__get_temp_conf`的函数，该函数通过读取一个名为`./templates/conf.yml`的文件，并将其解析为yaml格式的字典，返回给调用者。

接着，代码定义了一个名为save_conf`的函数，该函数接受一个conf字典和一个文件路径参数，将其存储为.yml格式的文件。通过with语句打开一个文件，并使用yaml.dump函数将conf字典写入文件中。

最后，代码通过Path类对象将当前工作目录设置为包含`./templates/conf.yml`文件的目录，并在程序启动时调用__get_temp_conf`函数，读取并返回初始的conf字典。


```py
import os
import json
from pathlib import Path
from ruamel import yaml
from flatten_dict import flatten, unflatten
from .. import __rootdir__


def __get_temp_conf():
    with Path(f'{__rootdir__}/templates/conf.yml').open('r', encoding='utf8') as f:
        return yaml.load(f,Loader=yaml.Loader)


def save_conf(conf, path="./conf.yml"):
    with Path(path).open('w', encoding='utf8') as f:
        yaml.dump(conf, f, allow_unicode=True)


```

这段代码定义了一个名为`load_conf`的函数，它接受一个参数`path`，它的默认值为当前工作目录中的临时配置文件('.conf.yml')。

函数首先调用一个名为`__get_temp_conf`的函数，并将它的返回值存储在变量`temp_conf`中。然后，函数判断给定的路径是否为文件，如果是，就创建一个空配置文件并保存已经定义的配置，最后返回temp_conf。如果路径不是文件，就创建一个空配置文件，并将temp_conf的值保存到配置文件中。

如果给定的路径是文件，函数使用`os.path.isfile`函数检查文件是否存在，如果不存在，就创建一个空配置文件并写入temp_conf的内容。如果文件已存在，就使用`with`语句打开文件，并使用`yaml.load`函数将文件内容加载到`conf`字典中。如果conf是空的，函数将创建一个空字典，并将temp_conf的内容复制到conf中。然后，函数使用`unflatten`函数将conf的列表内容扁平化，并将扁平化的内容存储在temp_conf中。

最后，函数将temp_conf存储在函数自身中，并返回temp_conf，以便在需要时使用。


```py
def load_conf(path="./conf.yml"):
    temp_conf = __get_temp_conf()
    if not os.path.isfile(path):
        open(path, 'w')  # 创建空配置文件
        save_conf(temp_conf, path)
        return temp_conf
    else:
        with Path(path).open('r', encoding='utf8') as c:
            conf = yaml.load(c, Loader=yaml.Loader)
            if conf is None:
                conf = {}
            flat_temp = flatten(temp_conf)
            flat_conf = flatten(conf)
            flat_temp.update(flat_conf)
            temp_conf = unflatten(flat_temp)
            return temp_conf


```

这段代码是一个Python定义函数，名为 "__get_temp_plan()"，它使用了一个与文件操作密切相关的`with`语句来打开一个名为`plan.json`的JSON文件，并将其读取为Python对象。

函数体中，首先读取文件内容并将其转换为JSON对象，然后使用该对象创建一个临时计划。接着，如果要保存的文件路径指定的文件存在，则创建一个新文件并将文件内容写入其中。最后，如果新版本的计划存储在`conf`键中，那么将新旧版本的内容进行更新，并在返回新版本的计划时，将`conf`键中的内容存储为新的计划。


```py
def __get_temp_plan():
    with open(f'{__rootdir__}/templates/plan.json', 'r') as f:
        return json.loads(f.read())


def load_plan(path="./plan.json"):
    temp_plan = __get_temp_plan()
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            json.dump(temp_plan, f)  # 创建空json文件
        return temp_plan
    with open(path, 'r', encoding='utf8') as fp:
        plan = json.loads(fp.read())
        if 'conf' not in plan.keys():  # 兼容旧版本
            temp_plan['plan1'] = plan
            return temp_plan
        # 获取新版本的
        tmp = temp_plan['conf']
        tmp.update(plan['conf'])
        plan['conf'] = tmp
        return plan


```

这段代码定义了一个名为 `write_plan` 的函数，它接受一个名为 `plan` 的参数，以及一个名为 `path` 的字符串参数，默认值为当前工作目录（"./plan.json"）。

函数的作用是将 `plan` 对象写入到一个名为 `write_plan.json` 的文件中，如果该文件不存在，则创建一个新的文件。

具体来说，函数使用 `with` 语句打开一个名为 `write_plan.json` 的文件，使用 `json.dump` 函数将 `plan` 对象写入到文件中，同时使用 `ensure_ascii=False` 参数来忽略可以将对象中的所有属性设置为小写的不良效果。这样，即使 `plan` 对象中存在一些不是字母或数字的字符，函数也可以安全地写入文件中。

函数的完整实现可以确保 `write_plan.json` 文件中的内容是一个 JSON 对象，可以根据需要进行调用，例如：

python
write_plan({
   "name": "John",
   "age": 30,
   "is_student": False
})

print(json.dumps(write_plan({
   "name": "John",
   "age": 30,
   "is_student": False
})))


这段代码将 `write_plan` 函数的作用解释为：创建一个名为 `write_plan.json` 的文件，并将一个名为 `plan` 的字典对象写入到该文件中。


```py
def write_plan(plan, path="./plan.json"):
    with open(path, 'w', encoding='utf8') as c:
        json.dump(plan, c, ensure_ascii=False)

```

# `arknights_mower/utils/config.py`

这段代码是一个Python语义注释，它定义了一个函数 `from __future__ import annotations`。该注释表示该函数将来的定义，但不会输出任何内容。

接下来的三行导入 `shutil`、`sys` 和 `typing` 模块。

接着，定义了一个函数 `from collections.abc import Mapping`，该函数表示从 `collections.abc` 模块中导入一个名为 `Mapping` 的类型。

然后，定义了一个函数 `from typing import Any`，用于定义一个名为 `Any` 的类型。

接下来，定义了一个函数 `from ruamel.yaml.comments import CommentedSeq`，该函数表示从 `ruamel.yaml.comments` 模块中导入一个名为 `CommentedSeq` 的类型。

最后，定义了一个函数 `__main__`，表示程序的入口点。函数的实现与 `__init__` 函数类似，但在 `__main__` 函数中需要进行一些额外的操作，如设置 `sys.path`、调用 `tp.configure` 函数等。这些操作是为了确保程序能够在不同的环境中正常运行。


```py
from __future__ import annotations

import shutil
import sys
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from pathlib import Path
from typing import Any

from ruamel.yaml.comments import CommentedSeq

from .. import __pyinstall__, __rootdir__, __system__
from . import typealias as tp
```

这段代码是一个Python脚本，它导入了`.yaml`模块，从而能够使用`yaml`模块中定义的YAML数据结构。

该脚本定义了两个变量：`VERSION_SUPPORTED_MIN`和`VERSION_SUPPORTED_MAX`。这些变量表示YAML模块中最低和最高支持的主机版的版本号。

此外，脚本还定义了一个名为`__ydoc`的属性。如果没有初始化，它的值为`None`。

接着，脚本定义了一个名为`BASE_CONSTRUCT_PLAN`的字典类型，它包含了一些子计划的实例。

最后，脚本定义了一个名为`OPE_PLAN`的列表类型，它包含了一些Ops计划的实例。

接下来，脚本实现了一个名为`__dig_mapping`的函数，它接收一个路径参数。这个函数将路径切分成多个部分，并从左到右遍历父节点。对于每个节点，函数检查当前节点是否属于当前节点，如果是，函数将当前节点存储在`current_map`中。如果不是，函数将引发KeyError。

最后，脚本导入了自定义的`from .yaml import yaml`语句，以便在需要时动态导入YAML模块。


```py
from .yaml import yaml

# The lowest version supported
VERSION_SUPPORTED_MIN = 1
VERSION_SUPPORTED_MAX = 1


__ydoc = None

BASE_CONSTRUCT_PLAN: dict[str, tp.BasePlan]
OPE_PLAN: list[tp.OpePlan]


def __dig_mapping(path: str):
    path = path.split('/')
    parent_maps = path[:-1]
    current_map = __ydoc
    for idx, k in enumerate(parent_maps):
        if k not in current_map:
            raise KeyError(path)
        current_map = current_map[k]
        if not isinstance(current_map, Mapping):
            raise TypeError(
                'config key %s is not a mapping' % '/'.join(path[: idx + 1])
            )
    return current_map, path[-1]


```

这两位作者创建了一个名为 `__get` 的函数，以及一个名为 `__get_list` 的函数。这两个函数的作用是获取一个序列（例如列表）的指定路径下的元素，并在路径找不到元素时返回一个默认值。

`__get__` 函数的作用是：
1. 如果给定的路径和默认值都是有效的，那么返回该路径下的第一个元素。
2. 如果给定的路径或默认值不正确，那么返回一个默认值。

`__get_list__` 函数的作用是：
1. 如果给定的路径和默认值都是有效的，那么返回该路径下的所有元素。
2. 如果给定的路径或默认值不正确，那么返回一个默认值（即列表类型）。
3. 如果给定的路径对应的用户地图（例如 `path`）是有效的，并且路径中的元素都是 `CommentedSeq` 类型，那么返回这些元素的列表。否则，返回一个空列表。


```py
def __get(path: str, default: Any = None):
    try:
        current_map, k = __dig_mapping(path)
    except (KeyError, TypeError) as e:
        return default
    if current_map is None or k not in current_map or current_map[k] is None:
        return default
    return current_map[k]


def __get_list(path: str, default: Any = list()):
    item = __get(path)
    if item is None:
        return default
    elif not isinstance(item, CommentedSeq):
        return [item]
    else:
        return list(item)


```

这段代码定义了一个内联函数 `__set(path: str, value: Any)`，用于设置一个文件的配置文件中的某个键的值。

具体来说，它实现了以下操作：

1. 尝试从当前目录(可能是应用程序的根目录)中映射出文件路径，以便在配置文件中查找对应的键。
2. 如果找到了对应的键，就尝试从配置文件中读取键值对，并将其存储在当前的映射中。
3. 如果尝试过程中出现了一个 `KeyError` 或 `TypeError`，就返回，表明无法继续执行。

函数的参数包括两个参数：要设置的文件路径和要设置的配置文件中的键值对。函数内部使用了 `__dig_mapping(path)` 函数来尝试从当前目录中找到对应的游戏配置文件路径，并使用 `yaml.load_all()` 函数来读取配置文件中的所有内容。

另外，函数 `build_config(path: str, module: bool)` 用于构建应用程序的配置文件。它读取了一个名为 `config.yaml` 的模板文件，并将其内容赋值给一个名为 `__config` 的变量。然后在函数内部使用 `__set` 函数来设置游戏配置文件中的各种键的值，这些键的值都是通过调用 `__config` 变量中的键得出的。由于 `__config` 是一个全局变量，因此它的值是在函数第一次被调用时初始化的。


```py
def __set(path: str, value: Any):
    try:
        current_map, k = __dig_mapping(path)
    except (KeyError, TypeError):
        return
    current_map[k] = value


def build_config(path: str, module: bool) -> None:
    """ build config via template """
    global __ydoc
    with Path(f'{__rootdir__}/templates/config.yaml').open('r', encoding='utf8') as f:
        loader = yaml.load_all(f)
        next(loader)  # discard first document (used for comment)
        __ydoc = next(loader)
    init_debug(module)
    __set('debug/logfile/path', str(LOGFILE_PATH.resolve()))
    __set('debug/screenshot/path', str(SCREENSHOT_PATH.resolve()))
    with Path(path).open('w', encoding='utf8') as f:
        yaml.dump(__ydoc, f)


```

这段代码定义了两个函数，分别是`load_config()`和`save_config()`。这两个函数的功能是加载和保存config文件中的配置信息。

`load_config()`函数接受一个参数`path`，它是一个字符串，表示config文件所在的路径。函数内部使用了一个全局变量`__ydoc`和一个名为`PATH`的全局变量，显然从文件路径中读取config文件的内容，并使用`with`语句打开文件进行读取。在文件读取完成后，将读取得到的配置信息存储到`__ydoc`中，然后调用一个名为`init_config()`的内部函数来初始化config变量。最后，如果当前的`version`值小于`MIN`或者大于`MAX`，则会抛出一个`RuntimeError`。

`save_config()`函数的接受参数与`load_config()`相反，它是一个空字符串，表示config文件应该保存到哪个文件中。函数使用一个`with`语句打开文件进行写入，并将config信息存储到文件中。


```py
def load_config(path: str) -> None:
    """ load config from PATH """
    global __ydoc, PATH
    PATH = Path(path)
    with PATH.open('r', encoding='utf8') as f:
        __ydoc = yaml.load(f)
    if not VERSION_SUPPORTED_MIN <= __get('version', 1) <= VERSION_SUPPORTED_MAX:
        raise RuntimeError('The current version of the config file is not supported')
    init_config()


def save_config() -> None:
    """ save config into PATH """
    global PATH
    with PATH.open('w', encoding='utf8') as f:
        yaml.dump(__ydoc, f)


```

This appears to be a Python script that retrieves configuration settings from the environment, including debugging settings such as the amount of log file output and the maximum number of retries. It also includes settings related to OCR, such as the API key and the maximum number of screenshot attempts.

The script has the following settings:

* LOGFILE\_AMOUNT: The number of log file output lines to include in the screenshot.
* SCREENSHOT\_PATH: The path to the directory containing the screenshot images.
* SCREENSHOT\_MAXNUM: The maximum number of screenshot images to create.
* MAX\_RETRYTIME: The maximum number of retries for the OCR API.
* OCR\_APIKEY: The API key for the OCR API.
* BASE\_CONSTRUCT\_PLAN: The base construct plan.
* SCHEDULE\_PLAN: The schedule plan.
* RECRUIT\_PRIORITY: The priority for recruitments.
* SHOP\_PRIORITY: The priority for shops.
* OPE\_TIMES: The number of operations to perform in the OCR process.
* OPE\_POTION: The potion to use in the OCR process.
* OPE\_ORIGINITE: The origin of the OCR image.
* OPE\_ELIMINATE: The operation to perform to eliminate the OCR image.
* OPE\_PLAN: The plan for the OCR image.
* TAP\_TO\_LAUNCH: A dictionary mapping FAPI applications to the corresponding极易类分数.

The script uses the `__get` method to retrieve the values of these settings, meaning that if the setting is defined multiple times in the script, it will return the first occurrence. For example, `__get('debug/logfile/amount')` will return the value of the `amount` setting, regardless of how many times it has been defined.


```py
def init_config() -> None:
    """ init config from __ydoc """
    global ADB_BINARY, ADB_DEVICE, ADB_CONNECT
    ADB_BINARY = __get('device/adb_binary', [])
    ADB_DEVICE = __get('device/adb_device', [])
    ADB_CONNECT = __get('device/adb_connect', [])
    if shutil.which('adb') is not None:
        ADB_BINARY.append(shutil.which('adb'))

    global ADB_BUILDIN
    ADB_BUILDIN = None

    global ADB_SERVER_IP, ADB_SERVER_PORT, ADB_SERVER_TIMEOUT
    ADB_SERVER_IP = __get('device/adb_server/ip', '127.0.0.1')
    ADB_SERVER_PORT = __get('device/adb_server/port', 5037)
    ADB_SERVER_TIMEOUT = __get('device/adb_server/timeout', 5)

    global ADB_CONTROL_CLIENT
    ADB_CONTROL_CLIENT = __get('device/adb_control_client', 'scrcpy')

    global MNT_TOUCH_DEVICE
    MNT_TOUCH_DEVICE = __get('device/mnt_touch_device', None)

    global MNT_PORT
    MNT_PORT = __get('device/mnt_port', 20937)

    global MNT_COMPATIBILITY_MODE
    MNT_COMPATIBILITY_MODE = __get('device/mnt_compatibility_mode', False)

    global USERNAME, PASSWORD
    USERNAME = __get('account/username', None)
    PASSWORD = __get('account/password', None)

    global APPNAME
    APPNAME = __get('app/package_name', 'com.hypergryph.arknights') 

    global APP_ACTIVITY_NAME
    APP_ACTIVITY_NAME = __get('app/activity_name','com.u8.sdk.U8UnityContext')

    global DEBUG_MODE, LOGFILE_PATH, LOGFILE_AMOUNT, SCREENSHOT_PATH, SCREENSHOT_MAXNUM
    DEBUG_MODE = __get('debug/enabled', False)
    LOGFILE_PATH = __get('debug/logfile/path', None)
    LOGFILE_AMOUNT = __get('debug/logfile/amount', 3)
    SCREENSHOT_PATH = __get('debug/screenshot/path', None)
    SCREENSHOT_MAXNUM = __get('debug/screenshot/max_total', 20)

    global MAX_RETRYTIME
    MAX_RETRYTIME = __get('behavior/max_retry', 5)

    global OCR_APIKEY
    OCR_APIKEY = __get('ocr/ocr_space_api', 'c7431c9d7288957')

    global BASE_CONSTRUCT_PLAN
    BASE_CONSTRUCT_PLAN = __get('arrangement', None)

    global SCHEDULE_PLAN
    SCHEDULE_PLAN = __get('schedule', None)

    global RECRUIT_PRIORITY, SHOP_PRIORITY
    RECRUIT_PRIORITY = __get('priority/recruit', None)
    SHOP_PRIORITY = __get('priority/shop', None)

    global OPE_TIMES, OPE_POTION, OPE_ORIGINITE, OPE_ELIMINATE, OPE_PLAN
    OPE_TIMES = __get('operation/times', -1)
    OPE_POTION = __get('operation/potion', 0)
    OPE_ORIGINITE = __get('operation/originite', 0)
    OPE_ELIMINATE = int(__get('operation/eliminate', 0))  # convert bool to int
    OPE_PLAN = __get('operation/plan', None)
    if OPE_PLAN is not None:
        OPE_PLAN = [x.split(',') for x in OPE_PLAN]
        OPE_PLAN = [[x[0], int(x[1])] for x in OPE_PLAN]

    global TAP_TO_LAUNCH
    TAP_TO_LAUNCH = {}


```

这段代码定义了一个名为 `update_ope_plan` 的函数，用于更新操作计划。该函数接收一个名为 `plan` 的列表参数，其中包含操作计划的元素。函数首先将 `OPEN_PLAN` 变量更新为输入的计划，然后遍历该计划并将其打印出来。接着，函数创建了一个新的 `OPERATION_PLAN` 列表，与输入的计划列表相同，用于保存更新后的操作计划。最后，函数调用了一个名为 `save_config` 的函数，该函数将 `OPERATION_PLAN` 列表写入到一个配置文件中。

接下来是定义了一个名为 `init_debug` 的函数，该函数用于初始化日志文件和截图文件。函数根据操作系统类型执行不同的操作，如果操作系统是 Windows，则将日志文件保存到 `arknights-mower` 目录中，并将截图文件保存到 `arknights-mower` 目录中。如果操作系统是 Linux，则将日志文件保存到 `/var/log/arknights-mower` 目录中，并将截图文件保存到 `/var/log/arknights-mower/screenshot` 目录中。如果操作系统是 Darwin，则将日志文件保存到 `/var/log/arknights-mower` 目录中，并将截图文件保存到 `/var/log/arknights-mower/screenshot` 目录中。如果输入的操作系统不支持上述操作，函数将引发 `NotImplementedError`。

最后，在 `main` 函数中，我们创建了一个 `OperationPlan` 类，该类继承自 `tp.OpePlan` 类。我们创建了一个名为 `tp.OperationPlan` 的类，该类实现了 `OpePlan` 类，用于定义操作计划的元素。我们还创建了一个名为 `update_ope_plan` 的函数，用于实现更新操作计划的功能。


```py
def update_ope_plan(plan: list[tp.OpePlan]) -> None:
    """ update operation plan """
    global OPE_PLAN
    OPE_PLAN = plan
    print([f'{x[0]},{x[1]}' for x in OPE_PLAN])
    __set('operation/plan', [f'{x[0]},{x[1]}' for x in OPE_PLAN])
    # TODO 其他参数也应该更新
    save_config()


def init_debug(module: bool) -> None:
    """ init LOGFILE_PATH & SCREENSHOT_PATH """
    global LOGFILE_PATH, SCREENSHOT_PATH
    if __pyinstall__:
        LOGFILE_PATH = Path(sys.executable).parent.joinpath('log')
        SCREENSHOT_PATH = Path(sys.executable).parent.joinpath('screenshot')
    elif module:
        if __system__ == 'windows':
            LOGFILE_PATH = Path.home().joinpath('arknights-mower')
            SCREENSHOT_PATH = Path.home().joinpath('arknights-mower/screenshot')
        elif __system__ == 'linux':
            LOGFILE_PATH = Path('/var/log/arknights-mower')
            SCREENSHOT_PATH = Path('/var/log/arknights-mower/screenshot')
        elif __system__ == 'darwin':
            LOGFILE_PATH = Path('/var/log/arknights-mower')
            SCREENSHOT_PATH = Path('/var/log/arknights-mower/screenshot')
        else:
            raise NotImplementedError(f'Unknown system: {__system__}')
    else:
        LOGFILE_PATH = __rootdir__.parent.joinpath('log')
        SCREENSHOT_PATH = __rootdir__.parent.joinpath('screenshot')


```

这段代码是一个Python脚本，名为“init_adb_buildin”，它定义了一个名为“init_adb_buildin”的函数，该函数返回一个名为“Path”的匿名类型。

函数的作用是在脚本运行时初始化ADB_BUILDIN和ADB_BUILDIN_DIR，其中ADB_BUILDIN是Android Debug Bridge（ADB）构建目录的根目录。

具体来说，函数根据操作系统类型来设置ADB_BUILDIN_DIR。如果运行在Windows上，则设置为用户主目录中的“adb-buildin”目录；如果运行在Linux或Darwin上，则设置为用户主目录中的“arknights-mower”目录。如果操作系统不被指定，则会引发一个名为“Unknown system”的异常。

函数的实现基于以下假设：

1. 如果运行在Windows上，则直接使用操作系统提供的一个名为“adb-buildin”的目录作为ADB_BUILDIN_DIR的值。
2. 如果运行在Linux或Darwin上，则需要根据操作系统类型设置ADB_BUILDIN_DIR。


```py
def init_adb_buildin() -> Path:
    """ init ADB_BUILDIN & ADB_BUILDIN_DIR """
    global ADB_BUILDIN_DIR, ADB_BUILDIN
    ADB_BUILDIN = None
    if __pyinstall__:
        ADB_BUILDIN_DIR = Path(sys.executable).parent.joinpath('adb-buildin')
    elif __system__ == 'windows':
        ADB_BUILDIN_DIR = Path.home().joinpath('arknights-mower/adb-buildin')
    elif __system__ == 'linux':
        ADB_BUILDIN_DIR = Path.home().joinpath('.arknights-mower')
    elif __system__ == 'darwin':
        ADB_BUILDIN_DIR = Path.home().joinpath('.arknights-mower')
    else:
        raise NotImplementedError(f'Unknown system: {__system__}')
    return ADB_BUILDIN_DIR


```

`init_config()` 函数在机器学习或深度学习任务中通常用于初始化配置文件。配置文件通常包含程序参数、层次结构、配置需求等信息。在这个函数中，可能是在读取或创建一个配置文件，然后根据需要对其进行修改或设置。具体作用可能因项目需求而异。


```py
init_config()

```