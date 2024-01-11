# `arknights-mower\arknights_mower\utils\asst.py`

```
# 导入 ctypes 模块，用于调用 C 语言的函数库
import ctypes
# 导入 os 模块，用于与操作系统交互
import os
# 导入 pathlib 模块，用于操作文件路径
import pathlib
# 导入 platform 模块，用于获取平台信息
import platform
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 Union、Dict、List、Any、Type、Optional 类型提示
from typing import Union, Dict, List, Any, Type, Optional
# 导入 Enum、IntEnum、unique、auto 枚举类型
from enum import Enum, IntEnum, unique, auto

# 定义 JSON 类型别名
JSON = Union[Dict[str, Any], List[Any], int, str, float, bool, Type[None]]

# 定义 InstanceOptionType 枚举类型，继承自 IntEnum
class InstanceOptionType(IntEnum):
    touch_type = 2
    deployment_with_pause = 3

# 定义 Asst 类
class Asst:
    # 定义回调函数类型 CFUNCTYPE
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
    # 定义一个函数，用于加载 DLL 及资源
    def load(path: Union[pathlib.Path, str], incremental_path: Optional[Union[pathlib.Path, str]] = None, user_dir: Optional[Union[pathlib.Path, str]] = None) -> bool:
        """
        加载 dll 及资源
        :params:
            ``path``:    DLL及资源所在文件夹路径
            ``incremental_path``:   增量资源所在文件夹路径
            ``user_dir``:   用户数据（日志、调试图片等）写入文件夹路径
        """

        # 定义不同平台下的 DLL 路径和环境变量
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

        # 获取当前操作系统类型
        platform_type = platform.system().lower()
        # 根据操作系统类型选择不同的 DLL 导入函数
        if platform_type == 'windows':
            lib_import_func = ctypes.WinDLL
        else:
            lib_import_func = ctypes.CDLL

        # 设置 DLL 路径
        Asst.__libpath = pathlib.Path(path) / platform_values[platform_type]['libpath']
        # 将 DLL 路径添加到环境变量中
        try:
            os.environ[platform_values[platform_type]['environ_var']] += os.pathsep + str(path)
        except KeyError:
            os.environ[platform_values[platform_type]['environ_var']] = os.pathsep + str(path)
        # 导入 DLL
        Asst.__lib = lib_import_func(str(Asst.__libpath))
        # 设置 DLL 属性
        Asst.__set_lib_properties()

        # 初始化返回值
        ret: bool = True
        # 如果有用户数据文件夹路径，则设置用户数据文件夹路径
        if user_dir:
            ret &= Asst.__lib.AsstSetUserDir(str(user_dir).encode('utf-8'))

        # 加载资源文件
        ret &= Asst.__lib.AsstLoadResource(str(path).encode('utf-8'))
        # 如果有增量资源文件夹路径，则加载增量资源文件
        if incremental_path:
            ret &= Asst.__lib.AsstLoadResource(
                str(incremental_path).encode('utf-8'))

        # 返回加载结果
        return ret
    # 初始化函数，接受回调函数和自定义参数
    def __init__(self, callback: CallBackType = None, arg=None):
        """
        :params:
            ``callback``:   回调函数
            ``arg``:        自定义参数
        """
        # 如果有回调函数，则使用回调函数和自定义参数创建对象指针
        if callback:
            self.__ptr = Asst.__lib.AsstCreateEx(callback, arg)
        # 否则，只使用默认方式创建对象指针
        else:
            self.__ptr = Asst.__lib.AsstCreate()

    # 析构函数，销毁对象指针
    def __del__(self):
        Asst.__lib.AsstDestroy(self.__ptr)
        self.__ptr = None

    # 设置额外配置的函数
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

    # 连接设备的函数
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

    # 任务 ID 的类型定义
    TaskId = int

    # 添加任务的函数
    def append_task(self, type_name: str, params: JSON = {}) -> TaskId:
        """
        添加任务
        :params:
            ``type_name``:  任务类型，请参考 docs/集成文档.md
            ``params``:     任务参数，请参考 docs/集成文档.md
        :return: 任务 ID, 可用于 set_task_params 接口
        """
        return Asst.__lib.AsstAppendTask(self.__ptr, type_name.encode('utf-8'), json.dumps(params, ensure_ascii=False).encode('utf-8'))
    # 动态设置任务参数
    def set_task_params(self, task_id: TaskId, params: JSON) -> bool:
        """
        :params:
            ``task_id``:  任务 ID, 使用 append_task 接口的返回值
            ``params``:   任务参数，同 append_task 接口，请参考 docs/集成文档.md
        :return: 是否成功
        """
        return Asst.__lib.AsstSetTaskParams(self.__ptr, task_id, json.dumps(params, ensure_ascii=False).encode('utf-8'))

    # 开始任务
    def start(self) -> bool:
        """
        :return: 是否成功
        """
        return Asst.__lib.AsstStart(self.__ptr)

    # 停止并清空所有任务
    def stop(self) -> bool:
        """
        :return: 是否成功
        """
        return Asst.__lib.AsstStop(self.__ptr)

    # 是否正在运行
    def running(self) -> bool:
        """
        :return: 是否正在运行
        """
        return Asst.__lib.AsstRunning(self.__ptr)

    # 打印日志
    @staticmethod
    def log(level: str, message: str) -> None:
        '''
        :params:
            ``level``:      日志等级标签
            ``message``:    日志内容
        '''
        Asst.__lib.AsstLog(level.encode('utf-8'), message.encode('utf-8'))

    # 获取DLL版本号
    def get_version(self) -> str:
        """
        : return: 版本号
        """
        return Asst.__lib.AsstGetVersion().decode('utf-8')
    # 设置 AsstSetUserDir 函数的返回类型为布尔型
    Asst.__lib.AsstSetUserDir.restype = ctypes.c_bool
    # 设置 AsstSetUserDir 函数的参数类型为字符指针
    Asst.__lib.AsstSetUserDir.argtypes = (
        ctypes.c_char_p,)

    # 设置 AsstLoadResource 函数的返回类型为布尔型
    Asst.__lib.AsstLoadResource.restype = ctypes.c_bool
    # 设置 AsstLoadResource 函数的参数类型为字符指针
    Asst.__lib.AsstLoadResource.argtypes = (
        ctypes.c_char_p,)

    # 设置 AsstCreate 函数的返回类型为无类型指针
    Asst.__lib.AsstCreate.restype = ctypes.c_void_p
    # 设置 AsstCreate 函数的参数类型为空
    Asst.__lib.AsstCreate.argtypes = ()

    # 设置 AsstCreateEx 函数的返回类型为无类型指针
    Asst.__lib.AsstCreateEx.restype = ctypes.c_void_p
    # 设置 AsstCreateEx 函数的参数类型为两个无类型指针
    Asst.__lib.AsstCreateEx.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p,)

    # 设置 AsstDestroy 函数的参数类型为无类型指针
    Asst.__lib.AsstDestroy.argtypes = (ctypes.c_void_p,)

    # 设置 AsstSetInstanceOption 函数的返回类型为布尔型
    Asst.__lib.AsstSetInstanceOption.restype = ctypes.c_bool
    # 设置 AsstSetInstanceOption 函数的参数类型为一个无类型指针和两个字符指针
    Asst.__lib.AsstSetInstanceOption.argtypes = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,)

    # 设置 AsstConnect 函数的返回类型为布尔型
    Asst.__lib.AsstConnect.restype = ctypes.c_bool
    # 设置 AsstConnect 函数的参数类型为一个无类型指针和三个字符指针
    Asst.__lib.AsstConnect.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,)

    # 设置 AsstAppendTask 函数的返回类型为整型
    Asst.__lib.AsstAppendTask.restype = ctypes.c_int
    # 设置 AsstAppendTask 函数的参数类型为一个无类型指针和两个字符指针
    Asst.__lib.AsstAppendTask.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)

    # 设置 AsstSetTaskParams 函数的返回类型为布尔型
    Asst.__lib.AsstSetTaskParams.restype = ctypes.c_bool
    # 设置 AsstSetTaskParams 函数的参数类型为一个无类型指针、整型和字符指针
    Asst.__lib.AsstSetTaskParams.argtypes = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)

    # 设置 AsstStart 函数的返回类型为布尔型
    Asst.__lib.AsstStart.restype = ctypes.c_bool
    # 设置 AsstStart 函数的参数类型为一个无类型指针
    Asst.__lib.AsstStart.argtypes = (ctypes.c_void_p,)

    # 设置 AsstStop 函数的返回类型为布尔型
    Asst.__lib.AsstStop.restype = ctypes.c_bool
    # 设置 AsstStop 函数的参数类型为一个无类型指针
    Asst.__lib.AsstStop.argtypes = (ctypes.c_void_p,)

    # 设置 AsstRunning 函数的返回类型为布尔型
    Asst.__lib.AsstRunning.restype = ctypes.c_bool
    # 设置 AsstRunning 函数的参数类型为一个无类型指针
    Asst.__lib.AsstRunning.argtypes = (ctypes.c_void_p,)

    # 设置 AsstGetVersion 函数的返回类型为字符指针
    Asst.__lib.AsstGetVersion.restype = ctypes.c_char_p

    # 设置 AsstLog 函数的返回类型为空
    Asst.__lib.AsstLog.restype = None
    # 设置 AsstLog 函数的参数类型为两个字符指针
    Asst.__lib.AsstLog.argtypes = (
        ctypes.c_char_p, ctypes.c_char_p)
# 定义一个枚举类，用于表示回调消息
@unique
class Message(Enum):
    """
    回调消息
    请参考 docs/回调消息.md
    """
    # 内部错误消息
    InternalError = 0

    # 初始化失败消息
    InitFailed = auto()

    # 连接信息消息
    ConnectionInfo = auto()

    # 所有任务完成消息
    AllTasksCompleted = auto()

    # 任务链错误消息
    TaskChainError = 10000

    # 任务链开始消息
    TaskChainStart = auto()

    # 任务链完成消息
    TaskChainCompleted = auto()

    # 任务链额外信息消息
    TaskChainExtraInfo = auto()

    # 任务链停止消息
    TaskChainStopped = auto()

    # 子任务错误消息
    SubTaskError = 20000

    # 子任务开始消息
    SubTaskStart = auto()

    # 子任务完成消息
    SubTaskCompleted = auto()

    # 子任务额外信息消息
    SubTaskExtraInfo = auto()

    # 子任务停止消息
    SubTaskStopped = auto()
```