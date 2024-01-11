# `arknights-mower\arknights_mower\utils\config.py`

```
# 导入必要的模块
from __future__ import annotations
import shutil
import sys
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from pathlib import Path
from typing import Any

# 导入自定义模块
from ruamel.yaml.comments import CommentedSeq
from .. import __pyinstall__, __rootdir__, __system__
from . import typealias as tp
from .yaml import yaml

# 定义最低支持的版本号和最高支持的版本号
VERSION_SUPPORTED_MIN = 1
VERSION_SUPPORTED_MAX = 1

# 初始化全局变量
__ydoc = None

# 定义全局变量
BASE_CONSTRUCT_PLAN: dict[str, tp.BasePlan]
OPE_PLAN: list[tp.OpePlan]

# 定义函数，用于递归查找映射
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

# 定义函数，用于获取指定路径的值
def __get(path: str, default: Any = None):
    try:
        current_map, k = __dig_mapping(path)
    except (KeyError, TypeError) as e:
        return default
    if current_map is None or k not in current_map or current_map[k] is None:
        return default
    return current_map[k]

# 定义函数，用于获取指定路径的列表值
def __get_list(path: str, default: Any = list()):
    item = __get(path)
    if item is None:
        return default
    elif not isinstance(item, CommentedSeq):
        return [item]
    else:
        return list(item)

# 定义函数，用于设置指定路径的值
def __set(path: str, value: Any):
    try:
        current_map, k = __dig_mapping(path)
    except (KeyError, TypeError):
        return
    current_map[k] = value

# 定义函数，用于构建配置
def build_config(path: str, module: bool) -> None:
    """ build config via template """
    global __ydoc
    with Path(f'{__rootdir__}/templates/config.yaml').open('r', encoding='utf8') as f:
        loader = yaml.load_all(f)
        next(loader)  # discard first document (used for comment)
        __ydoc = next(loader)
    # 初始化调试模块
    init_debug(module)
    # 设置调试日志文件路径
    __set('debug/logfile/path', str(LOGFILE_PATH.resolve()))
    # 设置调试截图路径
    __set('debug/screenshot/path', str(SCREENSHOT_PATH.resolve()))
    # 打开指定路径的文件，以写入模式，编码为utf8
    with Path(path).open('w', encoding='utf8') as f:
        # 将__ydoc对象以yaml格式写入文件
        yaml.dump(__ydoc, f)
# 从指定路径加载配置文件
def load_config(path: str) -> None:
    """ load config from PATH """
    # 声明要使用的全局变量
    global __ydoc, PATH
    # 将路径字符串转换为 Path 对象
    PATH = Path(path)
    # 打开配置文件，使用 utf8 编码方式读取
    with PATH.open('r', encoding='utf8') as f:
        # 将配置文件内容加载为 yaml 格式
        __ydoc = yaml.load(f)
    # 检查配置文件的版本是否在支持的范围内，如果不是则抛出异常
    if not VERSION_SUPPORTED_MIN <= __get('version', 1) <= VERSION_SUPPORTED_MAX:
        raise RuntimeError('The current version of the config file is not supported')
    # 初始化配置
    init_config()


# 将配置保存到文件中
def save_config() -> None:
    """ save config into PATH """
    # 声明要使用的全局变量
    global PATH
    # 打开配置文件，使用 utf8 编码方式写入
    with PATH.open('w', encoding='utf8') as f:
        # 将配置内容以 yaml 格式写入文件
        yaml.dump(__ydoc, f)


# 从 __ydoc 中初始化配置
def init_config() -> None:
    """ init config from __ydoc """
    # 声明要使用的全局变量
    global ADB_BINARY, ADB_DEVICE, ADB_CONNECT
    # 从 __ydoc 中获取 adb_binary、adb_device、adb_connect 的值
    ADB_BINARY = __get('device/adb_binary', [])
    ADB_DEVICE = __get('device/adb_device', [])
    ADB_CONNECT = __get('device/adb_connect', [])
    # 如果系统中存在 adb 命令，则将其路径添加到 ADB_BINARY 中
    if shutil.which('adb') is not None:
        ADB_BINARY.append(shutil.which('adb'))

    # 初始化其他全局变量
    # ...

    # 依次初始化其他全局变量
    # ...
    # 获取调试模式是否启用的配置，如果未配置则默认为 False
    DEBUG_MODE = __get('debug/enabled', False)
    # 获取日志文件路径的配置，如果未配置则默认为 None
    LOGFILE_PATH = __get('debug/logfile/path', None)
    # 获取日志文件数量的配置，如果未配置则默认为 3
    LOGFILE_AMOUNT = __get('debug/logfile/amount', 3)
    # 获取截图路径的配置，如果未配置则默认为 None
    SCREENSHOT_PATH = __get('debug/screenshot/path', None)
    # 获取最大截图数量的配置，如果未配置则默认为 20
    SCREENSHOT_MAXNUM = __get('debug/screenshot/max_total', 20)
    
    # 设置全局最大重试次数，获取配置，如果未配置则默认为 5
    global MAX_RETRYTIME
    MAX_RETRYTIME = __get('behavior/max_retry', 5)
    
    # 设置全局 OCR API 密钥，获取配置，如果未配置则默认为 'c7431c9d7288957'
    global OCR_APIKEY
    OCR_APIKEY = __get('ocr/ocr_space_api', 'c7431c9d7288957')
    
    # 设置基础构建计划，获取配置，如果未配置则默认为 None
    global BASE_CONSTRUCT_PLAN
    BASE_CONSTRUCT_PLAN = __get('arrangement', None)
    
    # 设置调度计划，获取配置，如果未配置则默认为 None
    global SCHEDULE_PLAN
    SCHEDULE_PLAN = __get('schedule', None)
    
    # 设置招募优先级和商店优先级，获取配置，如果未配置则默认为 None
    global RECRUIT_PRIORITY, SHOP_PRIORITY
    RECRUIT_PRIORITY = __get('priority/recruit', None)
    SHOP_PRIORITY = __get('priority/shop', None)
    
    # 设置操作次数、理智消耗、源石消耗、代理指挥消耗、操作计划，获取配置，如果未配置则默认为相应的默认值
    global OPE_TIMES, OPE_POTION, OPE_ORIGINITE, OPE_ELIMINATE, OPE_PLAN
    OPE_TIMES = __get('operation/times', -1)
    OPE_POTION = __get('operation/potion', 0)
    OPE_ORIGINITE = __get('operation/originite', 0)
    OPE_ELIMINATE = int(__get('operation/eliminate', 0))  # 将布尔值转换为整数
    OPE_PLAN = __get('operation/plan', None)
    # 如果操作计划不为 None，则将其转换为列表形式
    if OPE_PLAN is not None:
        OPE_PLAN = [x.split(',') for x in OPE_PLAN]
        OPE_PLAN = [[x[0], int(x[1])] for x in OPE_PLAN]
    
    # 设置点击启动的配置，初始化为空字典
    global TAP_TO_LAUNCH
    TAP_TO_LAUNCH = {}
# 更新操作计划
def update_ope_plan(plan: list[tp.OpePlan]) -> None:
    # 声明全局变量 OPE_PLAN
    global OPE_PLAN
    # 将传入的计划赋值给全局变量 OPE_PLAN
    OPE_PLAN = plan
    # 打印 OPE_PLAN 中每个元素的第一和第二个值
    print([f'{x[0]},{x[1]}' for x in OPE_PLAN])
    # 调用 __set 函数，更新操作计划
    __set('operation/plan', [f'{x[0]},{x[1]}' for x in OPE_PLAN])
    # TODO 其他参数也应该更新
    # 保存配置
    save_config()


# 初始化调试模式
def init_debug(module: bool) -> None:
    # 声明全局变量 LOGFILE_PATH, SCREENSHOT_PATH
    global LOGFILE_PATH, SCREENSHOT_PATH
    # 如果是打包后的程序
    if __pyinstall__:
        # 设置日志文件路径为执行文件的父目录下的 log 文件夹
        LOGFILE_PATH = Path(sys.executable).parent.joinpath('log')
        # 设置截图文件路径为执行文件的父目录下的 screenshot 文件夹
        SCREENSHOT_PATH = Path(sys.executable).parent.joinpath('screenshot')
    # 如果是模块
    elif module:
        # 根据系统设置日志文件路径和截图文件路径
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
        # 设置日志文件路径为根目录的父目录下的 log 文件夹
        LOGFILE_PATH = __rootdir__.parent.joinpath('log')
        # 设置截图文件路径为根目录的父目录下的 screenshot 文件夹
        SCREENSHOT_PATH = __rootdir__.parent.joinpath('screenshot')


# 初始化内置 ADB
def init_adb_buildin() -> Path:
    # 声明全局变量 ADB_BUILDIN_DIR, ADB_BUILDIN
    global ADB_BUILDIN_DIR, ADB_BUILDIN
    # 将 ADB_BUILDIN 置为 None
    ADB_BUILDIN = None
    # 如果是打包后的程序
    if __pyinstall__:
        # 设置 ADB 内置目录为执行文件的父目录下的 adb-buildin 文件夹
        ADB_BUILDIN_DIR = Path(sys.executable).parent.joinpath('adb-buildin')
    # 根据系统设置 ADB 内置目录
    elif __system__ == 'windows':
        ADB_BUILDIN_DIR = Path.home().joinpath('arknights-mower/adb-buildin')
    elif __system__ == 'linux':
        ADB_BUILDIN_DIR = Path.home().joinpath('.arknights-mower')
    elif __system__ == 'darwin':
        ADB_BUILDIN_DIR = Path.home().joinpath('.arknights-mower')
    else:
        raise NotImplementedError(f'Unknown system: {__system__}')
    return ADB_BUILDIN_DIR
# 调用初始化配置函数，用于初始化程序的配置
init_config()
```