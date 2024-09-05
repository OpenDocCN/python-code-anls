# `.\yolov8\ultralytics\utils\__init__.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import contextlib
import importlib.metadata
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

from ultralytics import __version__

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))  # 获取环境变量 RANK 的整数值，默认为 -1
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # 获取环境变量 LOCAL_RANK 的整数值，默认为 -1，用于 PyTorch Elastic运行

# Other Constants
ARGV = sys.argv or ["", ""]  # 获取命令行参数列表，若为空则初始化为包含两个空字符串的列表
FILE = Path(__file__).resolve()  # 获取当前脚本文件的绝对路径
ROOT = FILE.parents[1]  # 获取当前脚本文件的父目录的父目录路径，即 YOLO 的根目录
ASSETS = ROOT / "assets"  # 默认图像文件目录的路径
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"  # 默认配置文件路径
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # YOLO 多进程线程数，至少为 1，最多为 CPU 核心数减 1
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  # 全局自动安装模式，默认为 True
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # 全局详细模式，默认为 True
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm 进度条显示格式，如果详细模式开启则使用指定格式，否则为 None
LOGGING_NAME = "ultralytics"  # 日志记录器名称
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # 操作系统类型的布尔值
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 架构的布尔值
PYTHON_VERSION = platform.python_version()  # Python 版本号
TORCH_VERSION = torch.__version__  # PyTorch 版本号
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # torchvision 的版本号，比直接导入速度更快
HELP_MSG = """
    Usage examples for running YOLOv8:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.yaml')  # 从头开始构建一个新的模型
        model = YOLO("yolov8n.pt")  # 加载预训练模型（推荐用于训练）

        # Use the model
        results = model.train(data="coco8.yaml", epochs=3)  # 训练模型
        results = model.val()  # 在验证集上评估模型性能
        results = model('https://ultralytics.com/images/bus.jpg')  # 对图像进行预测
        success = model.export(format='onnx')  # 将模型导出为 ONNX 格式
"""
    # 使用命令行界面 (CLI)：
    
    YOLOv8 的 'yolo' CLI 命令遵循以下语法：
    
        yolo TASK MODE ARGS
    
        其中   TASK (可选) 可以是 [detect, segment, classify] 中的一个
              MODE (必需) 可以是 [train, val, predict, export] 中的一个
              ARGS (可选) 是任意数量的自定义 'arg=value' 对，如 'imgsz=320'，用于覆盖默认设置。
                  可以在 https://docs.ultralytics.com/usage/cfg 或通过 'yolo cfg' 查看所有 ARGS。
    
    - 训练一个检测模型，使用 coco8.yaml 数据集，模型为 yolov8n.pt，训练 10 个 epochs，初始学习率为 0.01
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
    
    - 使用预训练的分割模型预测 YouTube 视频，图像尺寸为 320：
        yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
    
    - 在批量大小为 1 和图像尺寸为 640 的情况下，验证预训练的检测模型：
        yolo detect val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640
    
    - 将 YOLOv8n 分类模型导出为 ONNX 格式，图像尺寸为 224x128 (不需要 TASK 参数)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
    
    - 运行特殊命令：
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
    
    文档链接：https://docs.ultralytics.com
    社区链接：https://community.ultralytics.com
    GitHub 仓库链接：https://github.com/ultralytics/ultralytics
# 设置和环境变量

# 设置 Torch 的打印选项，包括行宽、精度和默认配置文件
torch.set_printoptions(linewidth=320, precision=4, profile="default")

# 设置 NumPy 的打印选项，包括行宽和浮点数格式
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5

# 设置 OpenCV 的线程数为 0，以防止与 PyTorch DataLoader 的多线程不兼容
cv2.setNumThreads(0)

# 设置 NumExpr 的最大线程数为 NUM_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)

# 设置 CUBLAS 的工作空间配置为 ":4096:8"，用于确定性训练
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 设置 TensorFlow 的最小日志级别为 "3"，以在 Colab 中抑制冗长的 TF 编译器警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 设置 Torch 的 C++ 日志级别为 "ERROR"，以抑制 "NNPACK.cpp could not initialize NNPACK" 警告
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# 设置 Kineto 的日志级别为 "5"，以在计算 FLOPs 时抑制冗长的 PyTorch 分析器输出
os.environ["KINETO_LOG_LEVEL"] = "5"


class TQDM(tqdm_original):
    """
    自定义的 Ultralytics tqdm 类，具有不同的默认参数设置。

    Args:
        *args (list): 传递给原始 tqdm 的位置参数。
        **kwargs (any): 关键字参数，应用自定义默认值。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化自定义的 Ultralytics tqdm 类，具有不同的默认参数设置。

        注意，这些参数在调用 TQDM 时仍然可以被覆盖。
        """
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # 逻辑 'and' 操作，使用默认值（如果传递）。
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # 如果传递，则覆盖默认值。
        super().__init__(*args, **kwargs)


class SimpleClass:
    """
    Ultralytics SimpleClass 是一个基类，提供更简单的调试和使用方法，包括有用的字符串表示、错误报告和属性访问方法。
    """

    def __str__(self):
        """返回对象的人类可读字符串表示形式。"""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # 对于子类，仅显示模块和类名
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """返回对象的机器可读字符串表示形式。"""
        return self.__str__()

    def __getattr__(self, attr):
        """自定义属性访问错误消息，提供有用的信息。"""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """
    Ultralytics IterableSimpleNamespace 是 SimpleNamespace 的扩展类，添加了可迭代功能并支持与 dict() 和 for 循环一起使用。
    """
    # 返回一个迭代器，迭代命名空间的属性键值对
    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    # 返回对象的可读字符串表示，每行显示属性名=属性值
    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    # 自定义属性访问错误消息，提供帮助信息
    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    # 返回指定键的值，如果不存在则返回默认值
    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)
# 定义一个函数 plt_settings，用作装饰器，临时设置绘图函数的 rc 参数和后端

def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    # 如果未提供 rcparams，则使用默认字典设置 {"font.size": 11}
    if rcparams is None:
        rcparams = {"font.size": 11}

    # 定义装饰器函数 decorator
    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        # 定义 wrapper 函数，用于设置 rc 参数和后端，调用原始函数，并恢复设置
        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            # 获取当前的后端
            original_backend = plt.get_backend()
            # 如果指定的后端与当前不同，则关闭所有图形，并切换到指定的后端
            if backend.lower() != original_backend.lower():
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # 使用指定的 rc 参数上下文管理器
            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            # 如果使用了不同的后端，则关闭所有图形，并恢复原始后端
            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)

            return result

        return wrapper

    return decorator


# 定义函数 set_logging，为给定名称设置日志记录，支持 UTF-8 编码，并确保在不同环境中的兼容性
def set_logging(name="LOGGING_NAME", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different
    environments.
    """
    # 根据 verbose 参数设置日志级别，多 GPU 训练中考虑到 RANK 的情况
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    # 配置控制台（stdout）的编码为 UTF-8，以确保兼容性
    formatter = logging.Formatter("%(message)s")  # Default formatter
    # 如果在 Windows 环境下，并且 sys.stdout 具有 "encoding" 属性，并且编码不是 "utf-8"
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        # 定义一个定制的日志格式化器 CustomFormatter
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                # 返回格式化后的日志记录，包括表情符号处理
                return emojis(super().format(record))

        try:
            # 尝试重新配置 stdout 使用 UTF-8 编码（如果支持）
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # 对于不支持 reconfigure 的环境，用 TextIOWrapper 包装 stdout
            elif hasattr(sys.stdout, "buffer"):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                # 创建自定义格式化器以应对非 UTF-8 环境
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            # 如果出现异常，创建适应非 UTF-8 环境的自定义格式化器
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    # 创建并配置流处理器 StreamHandler，使用适当的格式化器和日志级别
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # 设置日志记录器 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    # 禁止日志传播到父记录器
    logger.propagate = False

    # 返回配置好的 logger 对象
    return logger
# 设置日志记录器
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # 在全局定义日志记录器，用于 train.py、val.py、predict.py 等
# 设置 "sentry_sdk" 和 "urllib3.connectionpool" 的日志级别为 CRITICAL + 1
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """返回与平台相关的安全的字符串版本，支持表情符号。"""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """
    用于确保函数或方法的线程安全执行的装饰器类。可以作为装饰器使用，以确保如果从多个线程调用装饰的函数，
    则只有一个线程能够执行该函数。

    Attributes:
        lock (threading.Lock): 管理对装饰函数访问的锁对象。

    Example:
        ```py
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # 在此处编写代码
        ```py
    """

    def __init__(self):
        """初始化装饰器类，用于函数或方法的线程安全执行。"""
        self.lock = threading.Lock()

    def __call__(self, f):
        """执行函数或方法的线程安全装饰器。"""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """应用线程安全性到装饰的函数或方法。"""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


def yaml_save(file="data.yaml", data=None, header=""):
    """
    将数据保存为 YAML 格式到文件中。

    Args:
        file (str, optional): 文件名。默认为 'data.yaml'。
        data (dict): 要保存的数据，以 YAML 格式。
        header (str, optional): 要添加的 YAML 头部。

    Returns:
        (None): 数据保存到指定文件中。
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # 如果父目录不存在，则创建父目录
        file.parent.mkdir(parents=True, exist_ok=True)

    # 将路径对象转换为字符串
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # 将数据以 YAML 格式写入文件
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    """
    从文件中加载 YAML 格式的数据。

    Args:
        file (str, optional): 文件名。默认为 'data.yaml'。
        append_filename (bool): 是否将 YAML 文件名添加到 YAML 字典中。默认为 False。

    Returns:
        (dict): YAML 格式的数据和文件名。
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    # 使用指定的编码打开文件，忽略解码错误，返回文件对象 f
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # 将文件内容读取为字符串 s

        # 如果字符串 s 中存在不可打印字符，则通过正则表达式去除特殊字符
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # 使用 yaml.safe_load() 加载字符串 s，转换为 Python 字典；若文件为空则返回空字典
        data = yaml.safe_load(s) or {}
        
        # 如果 append_filename 为真，则将文件名以字符串形式添加到字典 data 中
        if append_filename:
            data["yaml_file"] = str(file)
        
        # 返回加载后的数据字典 data
        return data
# 漂亮地打印一个 YAML 文件或 YAML 格式的字典
def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    # 如果 yaml_file 是字符串或 Path 对象，使用 yaml_load 加载 YAML 文件
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    # 将 yaml_dict 转换成 YAML 格式的字符串，不排序键，允许 Unicode，宽度为无限大
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=float("inf"))
    # 记录打印信息到日志，包含文件名（加粗黑色），以及 YAML 格式的字符串
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# 默认配置字典，从 DEFAULT_CFG_PATH 加载 YAML 文件得到
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
# 将值为字符串且为 "None" 的项改为 None
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
# 获取默认配置字典的键集合
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
# 用 DEFAULT_CFG_DICT 创建一个 IterableSimpleNamespace 对象作为默认配置
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    """
    从系统中读取设备型号信息，并缓存以便快速访问。被 is_jetson() 和 is_raspberrypi() 使用。

    Returns:
        (str): 如果成功读取，返回设备型号文件内容，否则返回空字符串。
    """
    # 尝试打开 "/proc/device-tree/model" 文件，读取并返回其内容
    with contextlib.suppress(Exception):
        with open("/proc/device-tree/model") as f:
            return f.read()
    # 如果发生异常（如文件不存在），返回空字符串
    return ""


def is_ubuntu() -> bool:
    """
    检查当前操作系统是否为 Ubuntu。

    Returns:
        (bool): 如果操作系统是 Ubuntu，则返回 True，否则返回 False。
    """
    # 尝试打开 "/etc/os-release" 文件，检查是否包含 "ID=ubuntu" 的行
    with contextlib.suppress(FileNotFoundError):
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    # 如果文件未找到或未包含 "ID=ubuntu" 行，则返回 False
    return False


def is_colab():
    """
    检查当前脚本是否运行在 Google Colab 笔记本环境中。

    Returns:
        (bool): 如果运行在 Colab 笔记本中，则返回 True，否则返回 False。
    """
    # 检查环境变量中是否包含 "COLAB_RELEASE_TAG" 或 "COLAB_BACKEND_VERSION"
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """
    检查当前脚本是否运行在 Kaggle 内核中。

    Returns:
        (bool): 如果运行在 Kaggle 内核中，则返回 True，否则返回 False。
    """
    # 检查当前工作目录是否为 "/kaggle/working"，并且检查 KAGGLE_URL_BASE 是否为 "https://www.kaggle.com"
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """
    检查当前脚本是否运行在 Jupyter Notebook 中。在 Colab、Jupyterlab、Kaggle、Paperspace 环境中验证通过。

    Returns:
        (bool): 如果运行在 Jupyter Notebook 中，则返回 True，否则返回 False。
    """
    # 尝试导入 IPython 中的 get_ipython 函数，如果导入成功并返回不为 None，则说明在 Jupyter 环境中
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    # 如果导入过程中发生异常，则返回 False
    return False


def is_docker() -> bool:
    """
    判断当前脚本是否运行在 Docker 容器中。

    Returns:
        (bool): 如果运行在 Docker 容器中，则返回 True，否则返回 False。
    """
    # 尝试打开 "/proc/self/cgroup" 文件，检查是否包含 "docker" 字符串
    with contextlib.suppress(Exception):
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    # 如果发生异常（如文件不存在），返回 False
    return False


def is_raspberrypi() -> bool:
    """
    判断当前设备是否为 Raspberry Pi。

    Returns:
        (bool): 如果设备为 Raspberry Pi，则返回 True，否则返回 False。
    """
    # 检查当前 Python 环境是否运行在树莓派上，通过检查设备模型信息
    # 返回值为布尔类型：如果在树莓派上运行则返回 True，否则返回 False
    """
    检查 Python 环境的设备模型信息中是否包含"Raspberry Pi"
    """
    return "Raspberry Pi" in PROC_DEVICE_MODEL
def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson Nano or Jetson Orin device by checking the device model
    information.

    Returns:
        (bool): True if running on a Jetson Nano or Jetson Orin, False otherwise.
    """
    return "NVIDIA" in PROC_DEVICE_MODEL  # 检查 PROC_DEVICE_MODEL 是否包含 "NVIDIA"，表示运行在 Jetson Nano 或 Jetson Orin


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    with contextlib.suppress(Exception):
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"  # 检查环境变量 YOLO_OFFLINE 是否为 "True"
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):  # 检查 Cloudflare 和 Google DNS
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # 获取模块的规范
    spec = importlib.util.find_spec(filepath)

    # 返回规范不为 None 且 origin 不为 None（表示是一个包）
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    """
    Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def is_git_dir():
    """
    Determines whether the current file is part of a git repository. If the current file is not part of a git
    repository, returns False.

    Returns:
        (bool): True if the current file is part of a git repository, False otherwise.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return True
    return False
    # 检查全局变量 GIT_DIR 是否为 None
    return GIT_DIR is not None
def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str | None): The origin URL of the git repository or None if not git directory.
    """
    # 检查当前是否在 Git 仓库中
    if IS_GIT_DIR:
        # 使用 subprocess 模块调用 git 命令获取远程仓库的 URL
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            # 将获取到的字节流解码成字符串，并去除首尾的空白字符
            return origin.decode().strip()


def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name or None if not a git directory.
    """
    # 检查当前是否在 Git 仓库中
    if IS_GIT_DIR:
        # 使用 subprocess 模块调用 git 命令获取当前分支名
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            # 将获取到的字节流解码成字符串，并去除首尾的空白字符
            return origin.decode().strip()


def get_default_args(func):
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    # 使用 inspect 模块获取函数的签名信息
    signature = inspect.signature(func)
    # 构建并返回参数名与默认参数值组成的字典
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    # 检查当前操作系统是否为 Ubuntu
    if is_ubuntu():
        # 尝试打开 /etc/os-release 文件并匹配版本号信息
        with contextlib.suppress(FileNotFoundError, AttributeError):
            with open("/etc/os-release") as f:
                # 使用正则表达式搜索并提取版本号
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]


def get_user_config_dir(sub_dir="Ultralytics"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    # 根据不同的操作系统返回相应的用户配置目录
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        # 如果不支持当前操作系统，抛出 ValueError 异常
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # 对于 GCP 和 AWS Lambda，只有 /tmp 目录可写入
    if not is_dir_writeable(path.parent):
        # 如果父目录不可写，输出警告信息并使用备选路径 /tmp 或当前工作目录
        LOGGER.warning(
            f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # 如果路径不存在，则创建相应的子目录
    path.mkdir(parents=True, exist_ok=True)

    return path


# Define constants (required below)
PROC_DEVICE_MODEL = read_device_model()  # is_jetson() and is_raspberrypi() depend on this constant
# 检查当前是否在线
ONLINE = is_online()

# 检查当前是否在Google Colab环境中
IS_COLAB = is_colab()

# 检查当前是否在Docker容器中
IS_DOCKER = is_docker()

# 检查当前是否在NVIDIA Jetson设备上
IS_JETSON = is_jetson()

# 检查当前是否在Jupyter环境中
IS_JUPYTER = is_jupyter()

# 检查当前是否在Kaggle环境中
IS_KAGGLE = is_kaggle()

# 检查当前代码是否安装为pip包
IS_PIP_PACKAGE = is_pip_package()

# 检查当前是否在树莓派环境中
IS_RASPBERRYPI = is_raspberrypi()

# 获取当前Git仓库的目录
GIT_DIR = get_git_dir()

# 检查当前目录是否是Git仓库
IS_GIT_DIR = is_git_dir()

# 获取用户配置目录，默认为环境变量YOLO_CONFIG_DIR，或者使用系统默认配置目录
USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # Ultralytics settings dir

# 设置配置文件路径为用户配置目录下的settings.yaml文件
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>>    # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>>     # Code block here
        >>>     pass
    """
    # 定义 TryExcept 类，用于处理异常，可以作为装饰器或上下文管理器使用

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose
        # 初始化 TryExcept 类，可以设置错误消息和详细输出选项

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        # 进入上下文管理器时执行的方法，初始化实例
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        # 定义退出上下文管理器时的行为，如果需要，打印错误消息
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
            # 如果设置了详细输出并且出现了异常，打印错误消息
        return True
        # 返回 True 表示已经处理了异常，不会向上层抛出异常
class Retry(contextlib.ContextDecorator):
    """
    Retry class for function execution with exponential backoff.

    Can be used as a decorator to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>>     # Replace with function logic that may raise exceptions
        >>>     return True
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times  # 设置重试次数
        self.delay = delay  # 设置初始重试延迟时间
        self._attempts = 0  # 记录当前重试次数

    def __call__(self, func):
        """Decorator implementation for Retry with exponential backoff."""

        def wrapped_func(*args, **kwargs):
            """Applies retries to the decorated function or method."""
            self._attempts = 0  # 重试次数初始化为0
            while self._attempts < self.times:  # 循环直到达到最大重试次数
                try:
                    return func(*args, **kwargs)  # 调用被装饰的函数或方法
                except Exception as e:
                    self._attempts += 1  # 增加重试次数计数
                    print(f"Retry {self._attempts}/{self.times} failed: {e}")  # 打印重试失败信息
                    if self._attempts >= self.times:  # 如果达到最大重试次数
                        raise e  # 抛出异常
                    time.sleep(self.delay * (2**self._attempts))  # 按指数增加的延迟时间

        return wrapped_func


def threaded(func):
    """
    Multi-threads a target function by default and returns the thread or function result.

    Use as @threaded decorator. The function runs in a separate thread unless 'threaded=False' is passed.
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function based on 'threaded' kwarg and returns the thread or function result."""
        if kwargs.pop("threaded", True):  # 如果未指定 'threaded' 参数或为 True
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)  # 创建线程对象
            thread.start()  # 启动线程
            return thread  # 返回线程对象
        else:
            return func(*args, **kwargs)  # 直接调用函数或方法

    return wrapper


def set_sentry():
    """
    Initialize the Sentry SDK for error tracking and reporting. Only used if sentry_sdk package is installed and
    sync=True in settings. Run 'yolo settings' to see and update settings YAML file.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)

    The function also configures Sentry SDK to ignore KeyboardInterrupt and FileNotFoundError
    exceptions and to exclude events with 'out of memory' in their exception message.
    """
    # 设置 Sentry 事件的自定义标签和用户信息
    def before_send(event, hint):
        """
        根据特定的异常类型和消息修改事件，然后发送给 Sentry。

        Args:
            event (dict): 包含错误信息的事件字典。
            hint (dict): 包含额外错误信息的字典。

        Returns:
            dict: 修改后的事件字典，如果不发送事件到 Sentry 则返回 None。
        """
        # 如果 hint 中包含异常信息
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            # 如果异常类型是 KeyboardInterrupt、FileNotFoundError，或者异常消息包含"out of memory"
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # 不发送事件

        # 设置事件的标签
        event["tags"] = {
            "sys_argv": ARGV[0],  # 系统参数列表中的第一个参数
            "sys_argv_name": Path(ARGV[0]).name,  # 第一个参数的文件名部分
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",  # 安装来源是 git、pip 还是其他
            "os": ENVIRONMENT,  # 环境变量中的操作系统信息
        }
        return event  # 返回修改后的事件字典

    # 如果满足一系列条件，则配置 Sentry
    if (
        SETTINGS["sync"]  # 同步设置为 True
        and RANK in {-1, 0}  # 运行时的进程等级为 -1 或 0
        and Path(ARGV[0]).name == "yolo"  # 第一个参数的文件名为 "yolo"
        and not TESTS_RUNNING  # 没有正在运行的测试
        and ONLINE  # 处于联机状态
        and IS_PIP_PACKAGE  # 安装方式为 pip
        and not IS_GIT_DIR  # 不是从 git 安装
    ):
        # 如果 sentry_sdk 包未安装，则返回
        try:
            import sentry_sdk  # 导入 sentry_sdk 包
        except ImportError:
            return

        # 初始化 Sentry SDK
        sentry_sdk.init(
            dsn="https://5ff1556b71594bfea135ff0203a0d290@o4504521589325824.ingest.sentry.io/4504521592406016",  # Sentry 项目的 DSN
            debug=False,  # 调试模式设为 False
            traces_sample_rate=1.0,  # 所有跟踪数据采样率设为 100%
            release=__version__,  # 使用当前应用版本号
            environment="production",  # 环境设置为生产环境
            before_send=before_send,  # 设置发送前的处理函数为 before_send
            ignore_errors=[KeyboardInterrupt, FileNotFoundError],  # 忽略的错误类型列表
        )
        # 设置 Sentry 用户信息，使用 SHA-256 匿名化的 UUID 哈希
        sentry_sdk.set_user({"id": SETTINGS["uuid"]})
class SettingsManager(dict):
    """
    Manages Ultralytics settings stored in a YAML file.

    Args:
        file (str | Path): Path to the Ultralytics settings YAML file. Default is USER_CONFIG_DIR / 'settings.yaml'.
        version (str): Settings version. In case of local version mismatch, new default settings will be saved.
    """
    def __init__(self, file=SETTINGS_YAML, version="0.0.4"):
        """
        Initialize the SettingsManager with default settings, load and validate current settings from the YAML
        file.
        """
        import copy  # 导入用于深拷贝对象的模块
        import hashlib  # 导入用于哈希计算的模块

        from ultralytics.utils.checks import check_version  # 导入版本检查函数
        from ultralytics.utils.torch_utils import torch_distributed_zero_first  # 导入分布式训练相关的函数

        root = GIT_DIR or Path()  # 设置根目录为环境变量GIT_DIR的值或当前路径
        datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.parent) else root).resolve()  # 设置数据集根目录路径

        self.file = Path(file)  # 将传入的文件路径转换为Path对象
        self.version = version  # 设置版本号
        self.defaults = {  # 设置默认配置字典
            "settings_version": version,
            "datasets_dir": str(datasets_root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # 计算当前机器的唯一标识符
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,  # 各种集成配置
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard": True,
            "wandb": True,
        }
        self.help_msg = (
            f"\nView settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        super().__init__(copy.deepcopy(self.defaults))  # 调用父类构造函数并使用深拷贝的默认配置

        with torch_distributed_zero_first(RANK):  # 在分布式环境中，仅主节点执行以下操作
            if not self.file.exists():  # 如果配置文件不存在，则保存默认配置
                self.save()

            self.load()  # 载入配置文件中的设置
            correct_keys = self.keys() == self.defaults.keys()  # 检查载入的配置键与默认配置键是否一致
            correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))  # 检查各项设置的类型是否与默认设置一致
            correct_version = check_version(self["settings_version"], self.version)  # 检查配置文件中的版本与当前版本是否一致
            if not (correct_keys and correct_types and correct_version):
                LOGGER.warning(
                    "WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem "
                    f"with your settings or a recent ultralytics package update. {self.help_msg}"
                )
                self.reset()  # 将设置重置为默认值

            if self.get("datasets_dir") == self.get("runs_dir"):  # 如果数据集目录与运行目录相同，则发出警告
                LOGGER.warning(
                    f"WARNING ⚠️ Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                    f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                    f"Please change one to avoid possible issues during training. {self.help_msg}"
                )

    def load(self):
        """Loads settings from the YAML file."""  # 从YAML文件加载设置
        super().update(yaml_load(self.file))  # 调用父类方法，使用yaml_load函数加载配置文件内容
    def save(self):
        """将当前设置保存到YAML文件中。"""
        # 调用yaml_save函数，将当前对象转换为字典并保存到文件中
        yaml_save(self.file, dict(self))

    def update(self, *args, **kwargs):
        """更新当前设置中的一个设置值。"""
        # 遍历关键字参数kwargs，检查每个设置项的有效性
        for k, v in kwargs.items():
            # 如果设置项不在默认设置中，则引发KeyError异常
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            # 获取默认设置项k的类型
            t = type(self.defaults[k])
            # 如果传入的值v不是预期的类型t，则引发TypeError异常
            if not isinstance(v, t):
                raise TypeError(f"Ultralytics setting '{k}' must be of type '{t}', not '{type(v)}'. {self.help_msg}")
        # 调用父类的update方法，更新设置项
        super().update(*args, **kwargs)
        # 更新后立即保存设置到文件
        self.save()

    def reset(self):
        """将设置重置为默认值并保存。"""
        # 清空当前设置
        self.clear()
        # 使用默认设置更新当前设置
        self.update(self.defaults)
        # 保存更新后的设置到文件
        self.save()
# 发出弃用警告的函数，用于提示已弃用的参数，建议使用更新的参数
def deprecation_warn(arg, new_arg):
    # 使用 LOGGER 发出警告消息，指出已弃用的参数和建议的新参数
    LOGGER.warning(
        f"WARNING ⚠️ '{arg}' is deprecated and will be removed in in the future. " f"Please use '{new_arg}' instead."
    )


# 清理 URL，去除授权信息，例如 https://url.com/file.txt?auth -> https://url.com/file.txt
def clean_url(url):
    # 使用 Path 对象将 URL 转换成 POSIX 格式的字符串，并替换 Windows 下的 "://" 为 "://"
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    # 对 URL 解码并按 "?" 进行分割，保留问号前的部分作为最终的清理后的 URL
    return urllib.parse.unquote(url).split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth


# 将 URL 转换为文件名，例如 https://url.com/file.txt?auth -> file.txt
def url2file(url):
    # 清理 URL 后使用 Path 对象获取文件名部分作为结果
    return Path(clean_url(url)).name


# 在 utils 初始化过程中运行以下代码 ------------------------------------------------------------------------------------

# 检查首次安装步骤
PREFIX = colorstr("Ultralytics: ")  # 设置日志前缀
SETTINGS = SettingsManager()  # 初始化设置管理器
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  # 全局数据集目录
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  # 全局权重目录
RUNS_DIR = Path(SETTINGS["runs_dir"])  # 全局运行目录
# 确定当前环境，根据不同情况设置 ENVIRONMENT 变量
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()  # 检查是否正在运行测试
set_sentry()  # 初始化 Sentry 错误监控

# 应用 Monkey Patch
from ultralytics.utils.patches import imread, imshow, imwrite, torch_load, torch_save

torch.load = torch_load  # 覆盖默认的 torch.load 函数
torch.save = torch_save  # 覆盖默认的 torch.save 函数
if WINDOWS:
    # 对于 Windows 平台，应用 cv2 的补丁以支持图像路径中的非 ASCII 和非 UTF 字符
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
```