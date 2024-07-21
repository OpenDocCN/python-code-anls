# `.\pytorch\torch\testing\_internal\common_utils.py`

```
# 忽略类型检查错误，这通常用于告知类型检查工具在此文件中忽略特定的类型错误
mypy: ignore-errors

r"""Importing this file must **not** initialize CUDA context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no CUDA calls shall be made, including torch.cuda.device_count(), etc.

torch.testing._internal.common_cuda.py can freely initialize CUDA context when imported.
"""

# 引入 argparse 库，用于解析命令行参数
import argparse
# 引入 contextlib 库，提供了一些与上下文管理器一起使用的实用函数
import contextlib
# 引入 copy 库，用于对象的深拷贝操作
import copy
# 引入 ctypes 库，用于调用动态链接库中的 C 函数
import ctypes
# 引入 errno 库，提供了与操作系统错误码相关的符号常量
import errno
# 引入 functools 库，提供了一些高阶函数，如函数装饰器
import functools
# 引入 gc 库，Python 的垃圾回收机制接口
import gc
# 引入 inspect 库，用于解析 Python 对象的结构信息
import inspect
# 引入 io 库，提供了 Python 的核心工具，用于处理流数据
import io
# 引入 json 库，用于解析 JSON 数据
import json
# 引入 logging 库，Python 标准库中的日志记录工具
import logging
# 引入 math 库，Python 标准数学函数库
import math
# 引入 operator 库，提供了 Python 中常见的运算符函数
import operator
# 引入 os 库，提供了与操作系统交互的功能
import os
# 引入 platform 库，用于访问底层操作系统的平台标识符
import platform
# 引入 random 库，生成伪随机数的模块
import random
# 引入 re 库，提供了正则表达式的支持
import re
# 引入 shutil 库，提供了高级的文件操作支持
import shutil
# 引入 signal 库，Python 标准信号处理模块
import signal
# 引入 socket 库，提供了网络通信的基本功能
import socket
# 引入 subprocess 库，用于创建新的进程并与其通信
import subprocess
# 引入 sys 库，提供了与 Python 解释器相关的变量和函数
import sys
# 引入 tempfile 库，用于创建临时文件和目录
import tempfile
# 引入 threading 库，提供了多线程编程的支持
import threading
# 引入 time 库，提供了时间操作的函数
import time
# 引入 types 库，包含了与 Python 类型操作相关的函数
import types
# 引入 unittest 库，Python 标准单元测试框架
import unittest
# 引入 warnings 库，用于处理警告信息
import warnings
# 从 collections.abc 中引入 Mapping 和 Sequence 类型
from collections.abc import Mapping, Sequence
# 从 contextlib 中引入 closing 和 contextmanager 上下文管理器
from contextlib import closing, contextmanager
# 从 copy 中引入 deepcopy 函数，用于对象的深拷贝操作
from copy import deepcopy
# 从 dataclasses 中引入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 从 enum 中引入 Enum 类，用于创建枚举类型
from enum import Enum
# 从 functools 中引入 partial 和 wraps 函数，用于创建偏函数和装饰器
from functools import partial, wraps
# 从 itertools 中引入 product 和 chain 函数，用于迭代器操作
from itertools import product, chain
# 从 pathlib 中引入 Path 类，用于操作文件路径
from pathlib import Path
# 从 statistics 中引入 mean 函数，用于计算平均值
from statistics import mean
# 从 typing 中引入多个类型，用于类型注解
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
# 从 unittest.mock 中引入 MagicMock 类，用于创建模拟对象
from unittest.mock import MagicMock

# 引入 expecttest 库，用于测试时的期望结果匹配
import expecttest
# 引入 numpy 库，Python 数值计算的核心库
import numpy as np

# 引入 __main__ 模块，忽略导入时的类型检查错误
import __main__  # type: ignore[import]
# 引入 torch 库，PyTorch 深度学习框架
import torch
# 引入 torch.backends.cudnn 模块，PyTorch 的 cuDNN 后端支持
import torch.backends.cudnn
# 引入 torch.backends.mkl 模块，PyTorch 的 MKL 后端支持
import torch.backends.mkl
# 引入 torch.backends.mps 模块，PyTorch 的 MPS 后端支持
import torch.backends.mps
# 引入 torch.backends.xnnpack 模块，PyTorch 的 XNNPACK 后端支持
import torch.backends.xnnpack
# 引入 torch.cuda 模块，PyTorch 的 CUDA 支持接口
import torch.cuda
# 从 torch 中引入 Tensor 类型，PyTorch 的张量类型
from torch import Tensor
# 从 torch._C 中引入 ScriptDict 和 ScriptList 类型，忽略属性定义时的类型检查错误
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
# 引入 torch._utils_internal 模块中的 get_writable_path 函数
from torch._utils_internal import get_writable_path
# 从 torch.nn 中引入多个类，PyTorch 的神经网络模块
from torch.nn import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
# 从 torch.onnx 中引入 register_custom_op_symbolic 和 unregister_custom_op_symbolic 函数
from torch.onnx import (
    register_custom_op_symbolic,
    unregister_custom_op_symbolic,
)
# 从 torch.testing 中引入 make_tensor 函数，用于创建测试用张量
from torch.testing import make_tensor
# 从 torch.testing._comparison 中引入多个类，用于张量比较和测试
from torch.testing._comparison import (
    BooleanPair,
    NonePair,
    NumberPair,
    Pair,
    TensorLikePair,
)
# 从 torch.testing._comparison 中引入 not_close_error_metas 函数，用于测试不接近的错误信息
from torch.testing._comparison import not_close_error_metas
# 从 torch.testing._internal.common_dtype 中引入 get_all_dtypes 函数，获取所有数据类型
from torch.testing._internal.common_dtype import get_all_dtypes
# 从 torch.utils._import_utils 中引入 _check_module_exists 函数，检查模块是否存在
from torch.utils._import_utils import _check_module_exists
# 从 torch.utils 中引入 _pytree 模块，树结构操作工具
import torch.utils._pytree as pytree

# 尝试导入 pytest 库，设置是否成功的标志
try:
    import pytest
    has_pytest = True
except ImportError:
    has_pytest = False

# 定义 freeze_rng_state 函数，用于冻结随机数生成器的状态
def freeze_rng_state(*args, **kwargs):
    return torch.testing._utils.freeze_rng_state(*args, **kwargs)


# 类 TestEnvironment，用于管理测试环境的标志和设置
class TestEnvironment:
    # repro_env_vars 字典，用于存储在测试失败时用于重现的环境变量和其值
    # 键为环境变量名，值为非默认且未隐含设置的环境变量值
    repro_env_vars: dict = {}

    # TODO: 扩展此类以处理除布尔标志外的任意设置？
    # 定义一个静态方法，用于设置整个测试套件中可用的标志，
    # 根据指定的环境变量确定其值。
    #
    # Args:
    #     name (str): 标志的名称。将设置具有此名称的全局变量，
    #         以便在整个测试套件中方便访问。
    #     env_var (str): 主要环境变量的名称，用于确定此标志的值。
    #         如果为None或环境变量未设置，则将使用默认值，除非另有规定（参见implied_by_fn）。默认值为None。
    #     default (bool): 如果环境变量未设置且未被暗示，则用于标志的默认值。默认值为False。
    #     include_in_repro (bool): 表示是否应在测试失败时的重现命令中包含此标志
    #         （即它是否可能与重现测试失败有关）。默认值为True。
    #     enabled_fn (Callable): 可调用对象，根据环境变量值和默认值返回标志是否应启用的结果。
    #         默认为Lambda表达式，要求如果默认情况下为"0"则禁用，如果默认情况下为"1"则启用。
    #     implied_by_fn (Callable): 一个Thunk，返回一个布尔值，指示此标志是否由其主要环境变量设置之外的某些东西启用。
    #         例如，如果另一个环境变量的值意味着启用该标志，则此功能很有用。默认为返回False的Lambda表达式，表示没有任何影响。
    @staticmethod
    def def_flag(
        name,
        env_var=None,
        default=False,
        include_in_repro=True,
        enabled_fn=lambda env_var_val, default: (
            (env_var_val != "0") if default else (env_var_val == "1")),
        implied_by_fn=lambda: False,
    ):
        # 默认情况下，标志的启用状态为默认值
        enabled = default
        # 如果指定了环境变量，则尝试获取其值并根据enabled_fn确定标志的最终启用状态
        if env_var is not None:
            env_var_val = os.getenv(env_var)
            enabled = enabled_fn(env_var_val, default)
        # 检查是否有其他因素暗示标志的启用状态
        implied = implied_by_fn()
        # 将暗示的启用状态也考虑在内
        enabled = enabled or implied
        # 如果需要在重现命令中包含此标志，并且环境变量已指定，并且标志的实际启用状态与默认状态不同且没有被暗示，则将其加入到重现环境变量字典中
        if include_in_repro and (env_var is not None) and (enabled != default) and not implied:
            TestEnvironment.repro_env_vars[env_var] = env_var_val
        
        # 为了方便起见，全局导出标志
        assert name not in globals(), f"duplicate definition of flag '{name}'"
        globals()[name] = enabled

    # 返回一个字符串前缀，可用于设置任何测试中需要显式设置以匹配此测试套件实例的环境变量。
    # 示例："PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_ROCM=1"
    @staticmethod
    def repro_env_var_prefix() -> str:
        return " ".join([f"{env_var}={value}"
                         for env_var, value in TestEnvironment.repro_env_vars.items()])
# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)

# 禁用 Torch 的全局标志
torch.backends.disable_global_flags()

# 文件路径模式的前缀，默认为 "file://"，在 Windows 下会变为 "file:///"
FILE_SCHEMA = "file://"
if sys.platform == 'win32':
    FILE_SCHEMA = "file:///"

# NB: 这个标志在语义上与其他标志不同，设置环境变量为任何非空值都会使其为真：
#   CI=1, CI="true", CI=0 等都会使标志为真。
#   CI= 和未设置 CI 都会使标志为假。
# GitHub 将 CI 的值设置为 "true" 以启用该标志。
# 定义一个名为 "IS_CI" 的标志，根据环境变量 "CI" 的值确定，不包含在可重现性中
TestEnvironment.def_flag("IS_CI", env_var="CI", include_in_repro=False,
                         enabled_fn=lambda env_var_value, _: bool(env_var_value))

# 定义一个名为 "IS_SANDCASTLE" 的标志，根据环境变量 "SANDCASTLE" 和 TW_JOB_USER 的值确定，不包含在可重现性中
TestEnvironment.def_flag(
    "IS_SANDCASTLE",
    env_var="SANDCASTLE",
    implied_by_fn=lambda: os.getenv("TW_JOB_USER") == "sandcastle",
    include_in_repro=False)

# 检查是否在 FBCode 环境中，默认根据 torch._utils_internal.IS_FBSOURCE 的值确定
_is_fbcode_default = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and
    torch._utils_internal.IS_FBSOURCE
)
TestEnvironment.def_flag("IS_FBCODE", env_var="PYTORCH_TEST_FBCODE",
                         default=_is_fbcode_default,
                         include_in_repro=False)

# 定义一个名为 "IS_REMOTE_GPU" 的标志，根据环境变量 "PYTORCH_TEST_REMOTE_GPU" 的值确定，不包含在可重现性中
TestEnvironment.def_flag("IS_REMOTE_GPU", env_var="PYTORCH_TEST_REMOTE_GPU",
                         include_in_repro=False)

# 定义一个名为 "DISABLE_RUNNING_SCRIPT_CHK" 的标志，根据环境变量 "PYTORCH_DISABLE_RUNNING_SCRIPT_CHK" 的值确定，不包含在可重现性中
TestEnvironment.def_flag(
    "DISABLE_RUNNING_SCRIPT_CHK",
    env_var="PYTORCH_DISABLE_RUNNING_SCRIPT_CHK",
    include_in_repro=False)

# NB: 默认启用，除非在 fbcode 环境中。
# 定义一个名为 "PRINT_REPRO_ON_FAILURE" 的标志，根据环境变量 "PYTORCH_PRINT_REPRO_ON_FAILURE" 和 IS_FBCODE 的值确定，不包含在可重现性中
TestEnvironment.def_flag("PRINT_REPRO_ON_FAILURE", env_var="PYTORCH_PRINT_REPRO_ON_FAILURE",
                         default=(not IS_FBCODE), include_in_repro=False)  # noqa: F821

# 默认禁用的测试文件和慢速测试文件名
DEFAULT_DISABLED_TESTS_FILE = '.pytorch-disabled-tests.json'
DEFAULT_SLOW_TESTS_FILE = '.pytorch-slow-tests.json'

# 初始化空的禁用测试和慢速测试字典
disabled_tests_dict = {}
slow_tests_dict = {}

# 如果存在对应的环境变量，则加载禁用测试和慢速测试的 JSON 文件内容到相应的字典中
def maybe_load_json(filename):
    if os.path.isfile(filename):
        with open(filename) as fp:
            return json.load(fp)
    # 如果文件不存在，记录警告日志
    log.warning("Attempted to load json file '%s' but it does not exist.", filename)
    return {}

# 如果存在环境变量 "SLOW_TESTS_FILE"，则加载其指定的 JSON 文件到 slow_tests_dict
if os.getenv("SLOW_TESTS_FILE", ""):
    slow_tests_dict = maybe_load_json(os.getenv("SLOW_TESTS_FILE", ""))
# 如果存在环境变量 "DISABLED_TESTS_FILE"，则加载其指定的 JSON 文件到 disabled_tests_dict
if os.getenv("DISABLED_TESTS_FILE", ""):
    disabled_tests_dict = maybe_load_json(os.getenv("DISABLED_TESTS_FILE", ""))

# 定义支持的本地设备列表
NATIVE_DEVICES = ('cpu', 'cuda', 'xpu', 'meta', torch._C._get_privateuse1_backend_name())

# 检查当前平台是否为 Jetson 系列设备
check_names = ['orin', 'concord', 'galen', 'xavier', 'nano', 'jetson', 'tegra']
IS_JETSON = any(name in platform.platform() for name in check_names)

# 在 Jetson 设备上执行内存清理的装饰器函数
def gcIfJetson(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if IS_JETSON:
            gc.collect()  # 执行 Python 垃圾回收
            torch.cuda.empty_cache()  # 清空 CUDA 设备上的缓存
        fn(*args, **kwargs)
    return wrapper

# 尝试通过检查堆栈来提取当前的测试函数
# 如果失败，则返回 None
def extract_test_fn() -> Optional[Callable]:
    # 尝试获取当前调用栈信息
    try:
        # 获取当前调用栈的详细信息
        stack = inspect.stack()
        # 遍历调用栈中的每个帧信息
        for frame_info in stack:
            # 获取帧对象
            frame = frame_info.frame
            # 检查帧的局部变量中是否包含名为 "self" 的变量
            if "self" not in frame.f_locals:
                continue
            # 获取 "self" 变量的值
            self_val = frame.f_locals["self"]
            # 检查 "self" 变量是否为 unittest.TestCase 的实例
            if isinstance(self_val, unittest.TestCase):
                # 获取测试用例的标识符
                test_id = self_val.id()
                # 根据标识符获取测试用例的名称
                test_name = test_id.split('.')[2]
                # 获取测试用例方法的实际函数对象
                test_fn = getattr(self_val, test_name).__func__
                # 返回测试用例的函数对象
                return test_fn
    # 捕获任何异常并忽略
    except Exception:
        pass
    # 如果未找到适合的测试用例函数，则返回 None
    return None
# 用于调试目的的跟踪输入数据类，包含索引、值和类型描述信息
@dataclass
class TrackedInput:
    index: int
    val: Any
    type_desc: str

# 尝试从测试函数中提取跟踪输入信息，返回一个 TrackedInput 对象
# 使用 TrackedInputIter 类插入这些信息。
def get_tracked_input() -> Optional[TrackedInput]:
    # 提取当前测试函数
    test_fn = extract_test_fn()
    if test_fn is None:
        return None
    # 如果测试函数没有 tracked_input 属性，返回 None
    if not hasattr(test_fn, "tracked_input"):
        return None
    # 返回测试函数中的 tracked_input 属性
    return test_fn.tracked_input

# 清除测试函数中的 tracked_input 属性
def clear_tracked_input():
    # 提取当前测试函数
    test_fn = extract_test_fn()
    if test_fn is None:
        return
    # 如果测试函数没有 tracked_input 属性，直接返回
    if not hasattr(test_fn, "tracked_input"):
        return None
    # 将测试函数中的 tracked_input 属性设置为 None
    test_fn.tracked_input = None

# 包装一个迭代器，用于跟踪迭代器产生的最新值，以便调试目的
# 跟踪的值存储在测试函数上。
class TrackedInputIter:
    def __init__(self, child_iter, input_type_desc, callback=lambda x: x):
        # 构造函数，初始化子迭代器、输入类型描述和回调函数
        self.child_iter = enumerate(child_iter)
        # 输入类型描述，描述正在跟踪的内容（例如 "示例输入"、"错误输入"）
        self.input_type_desc = input_type_desc
        # 回调函数，在每个迭代项上运行以获取要跟踪的内容
        self.callback = callback
        # 提取当前测试函数
        self.test_fn = extract_test_fn()

    def __iter__(self):
        # 返回迭代器自身
        return self

    def __next__(self):
        # 获取子迭代器的下一个元素
        input_idx, input_val = next(self.child_iter)
        # 创建 TrackedInput 对象，设置为测试函数的 tracked_input 属性
        self._set_tracked_input(
            TrackedInput(
                index=input_idx, val=self.callback(input_val), type_desc=self.input_type_desc
            )
        )
        # 返回迭代项的值
        return input_val

    def _set_tracked_input(self, tracked_input: TrackedInput):
        # 如果当前测试函数不存在，直接返回
        if self.test_fn is None:
            return
        # 如果测试函数没有 tracked_input 属性，直接返回
        if not hasattr(self.test_fn, "tracked_input"):
            return
        # 设置测试函数的 tracked_input 属性为指定的 tracked_input 对象
        self.test_fn.tracked_input = tracked_input

class _TestParametrizer:
    """
    参数化测试函数的装饰器类，生成基于原始通用测试的一组新测试，
    每个测试专门针对特定的测试输入集。例如，针对一组操作进行参数化，
    将生成每个操作的测试函数。

    参数化 / 如何参数化的决定应由每个派生类实现。

    装饰器在细节上向测试函数添加一个 'parametrize_fn' 属性。此函数
    预期稍后由以下之一调用：
      * 通过 instantiate_device_type_tests() 实例化特定设备类型的测试。请注意，
        对于这种情况，无需显式参数化设备类型，因为这将单独处理。
      * 通过 instantiate_parametrized_tests() 实例化设备不可知的参数化测试。

    如果将装饰器应用于已经具有 'parametrize_fn' 属性的测试函数，则将创建
    一个新的复合 'parametrize_fn'，生成带有参数的测试。

    """
    """
    generated by the old and new parametrize_fns. This allows for convenient composability of decorators.
    """
    # 将新旧的 parametrize_fns 组合，以便方便地组合装饰器。
    def _parametrize_test(self, test, generic_cls, device_cls):
        """
        Parametrizes the given test function across whatever dimension is specified by the derived class.
        Tests can be parametrized over any arbitrary dimension or combination of dimensions, such as all
        ops, all modules, or all ops + their associated dtypes.

        Args:
            test (fn): Test function to parametrize over
            generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
            device_cls (class): Device-specialized test class object (e.g. TestFooCPU); set to None
                if the tests are not part of a device-specific set

        Returns:
            Generator object returning 4-tuples of:
                test (fn): Parametrized test function; must support a device arg and args for any params
                test_name (str): Parametrized suffix for the test (e.g. opname_int64); will be appended to
                    the base name of the test
                param_kwargs (dict): Param kwargs to pass to the test (e.g. {'op': 'add', 'dtype': torch.int64})
                decorator_fn (callable): Callable[[Dict], List] for list of decorators to apply given param_kwargs
        """
        # 抛出未实现错误，表明该函数应该在派生类中被重写实现
        raise NotImplementedError

    def __call__(self, fn):
        # 检查函数 fn 是否有 parametrize_fn 属性
        if hasattr(fn, 'parametrize_fn'):
            # 如果有，将现有的 parametrize_fn 保存为 old_parametrize_fn
            old_parametrize_fn = fn.parametrize_fn
            # 设置新的 parametrize_fn 为 _parametrize_test 方法
            new_parametrize_fn = self._parametrize_test
            # 组合新旧 parametrize_fn，并将结果保存为 fn.parametrize_fn
            fn.parametrize_fn = compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn)
        else:
            # 如果没有，直接将 fn.parametrize_fn 设置为 _parametrize_test 方法
            fn.parametrize_fn = self._parametrize_test
        # 返回被修饰的函数 fn
        return fn
# 组合两个 parametrize_fn 函数，生成一个新的 parametrize_fn 函数，用于处理参数化测试的组合
def compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn):
    """
    Returns a parametrize_fn that parametrizes over the product of the parameters handled
    by the given parametrize_fns. Each given parametrize_fn should each have the signature
    f(test, generic_cls, device_cls).

    The test names will be a combination of the names produced by the parametrize_fns in
    "<new_name>_<old_name>" order. This order is done to match intuition for constructed names
    when composing multiple decorators; the names will be built in top to bottom order when stacking
    parametrization decorators.

    Args:
        old_parametrize_fn (callable) - First parametrize_fn to compose.
        new_parametrize_fn (callable) - Second parametrize_fn to compose.
    """

    # 定义一个组合函数 composite_fn，接受 test、generic_cls、device_cls 三个参数
    def composite_fn(test, generic_cls, device_cls,
                     old_parametrize_fn=old_parametrize_fn,
                     new_parametrize_fn=new_parametrize_fn):
        # 调用 old_parametrize_fn 处理当前的 test，并获取其返回的列表
        old_tests = list(old_parametrize_fn(test, generic_cls, device_cls))
        # 遍历 old_tests 中的每一个元组
        for (old_test, old_test_name, old_param_kwargs, old_dec_fn) in old_tests:
            # 对 old_test 应用 new_parametrize_fn，获取新的测试、测试名、参数字典和装饰器函数
            for (new_test, new_test_name, new_param_kwargs, new_dec_fn) in \
                    new_parametrize_fn(old_test, generic_cls, device_cls):
                # 检查是否有重复的参数在两个 parametrize_fn 中都有定义
                redundant_params = set(old_param_kwargs.keys()).intersection(new_param_kwargs.keys())
                if redundant_params:
                    # 如果存在重复参数，则抛出运行时错误
                    raise RuntimeError('Parametrization over the same parameter by multiple parametrization '
                                       f'decorators is not supported. For test "{test.__name__}", the following parameters '
                                       f'are handled multiple times: {redundant_params}')
                # 合并两个 parametrize_fn 的参数字典
                full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
                # 根据新旧测试名构建合并后的测试名
                merged_test_name = '{}{}{}'.format(new_test_name,
                                                   '_' if old_test_name != '' and new_test_name != '' else '',
                                                   old_test_name)

                # 定义一个合并装饰器函数，将 old_dec_fn 和 new_dec_fn 合并处理参数
                def merged_decorator_fn(param_kwargs, old_dec_fn=old_dec_fn, new_dec_fn=new_dec_fn):
                    return list(old_dec_fn(param_kwargs)) + list(new_dec_fn(param_kwargs))

                # 返回合并后的测试、测试名、参数字典和合并装饰器函数的元组
                yield (new_test, merged_test_name, full_param_kwargs, merged_decorator_fn)

    return composite_fn


# 实例化已被 parametrize_fn 装饰的测试函数，用特定名称的参数化测试替换通用测试
def instantiate_parametrized_tests(generic_cls):
    """
    Instantiates tests that have been decorated with a parametrize_fn. This is generally performed by a
    decorator subclass of _TestParametrizer. The generic test will be replaced on the test class by
    parametrized tests with specialized names. This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```
    """

    # 这里是一个占位函数，没有添加具体的代码内容
    Args:
        generic_cls (class): 包含测试的通用测试类对象（例如 TestFoo）
    """
    # 遍历通用测试类对象的所有属性名
    for attr_name in tuple(dir(generic_cls)):
        # 获取属性名对应的属性对象
        class_attr = getattr(generic_cls, attr_name)
        # 如果属性对象没有 parametrize_fn 属性，跳过当前循环
        if not hasattr(class_attr, 'parametrize_fn'):
            continue

        # 从测试类中移除通用测试
        delattr(generic_cls, attr_name)

        # 添加参数化测试到测试类中
        def instantiate_test_helper(cls, name, test, param_kwargs):
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                test(self, **param_kwargs)

            # 断言测试类中没有重复定义同名的测试
            assert not hasattr(generic_cls, name), f"Redefinition of test {name}"
            # 将实例化后的测试方法设置为测试类的新属性
            setattr(generic_cls, name, instantiated_test)

        # 遍历 parametrize_fn 方法返回的测试信息
        for (test, test_suffix, param_kwargs, decorator_fn) in class_attr.parametrize_fn(
                class_attr, generic_cls=generic_cls, device_cls=None):
            # 构建完整的测试方法名称
            full_name = f'{test.__name__}_{test_suffix}'

            # 根据参数关键字应用装饰器函数
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            # 实例化测试方法并添加到测试类中
            instantiate_test_helper(cls=generic_cls, name=full_name, test=test, param_kwargs=param_kwargs)
    
    # 返回更新后的通用测试类对象
    return generic_cls
class subtest:
    """
    用于测试参数化的显式子测试案例。
    允许显式命名各个子测试案例，并将装饰器应用于参数化测试。

    Args:
        arg_values (iterable): 参数值的可迭代对象（例如 range(10)）或参数值元组的可迭代对象
            （例如 [(1, 2), (3, 4)]）。
        name (str): 可选的测试名称。
        decorators (iterable): 应用于生成测试的装饰器的可迭代对象。
    """
    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []


class parametrize(_TestParametrizer):
    """
    用于应用通用测试参数化的装饰器。

    此装饰器的接口模仿 `@pytest.mark.parametrize`。
    与 pytest 的基本用法相同。第一个参数应为包含测试参数名称的字符串，用逗号分隔，
    第二个参数应为返回单个值或值元组的可迭代对象（对于多个参数的情况）。

    除了基本用法外，此装饰器还提供了一些 pytest 不具备的额外功能。

    1. 参数化测试最终会生成 unittest 测试类的生成测试函数。
    因此，此装饰器额外负责命名这些测试函数。默认的测试名称由测试的基本名称以及每个参数名称和值组成
    （例如 "test_bar_x_1_y_foo"），但可以使用 `name_fn` 或 `subtest` 结构定义自定义名称（参见下文）。

    2. 装饰器特别处理类型为 `subtest` 的参数值，允许更精细的控制测试名称和执行。
    特别地，它可用于标记具有显式测试名称或应用任意装饰器的子测试（见下面的示例）。

    Examples::

        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize("x, y", [subtest((1, 2), name='double'),
                              subtest((1, 3), name='triple', decorators=[unittest.expectedFailure]),
                              subtest((1, 4), name='quadruple')])
        def test_baz(self, x, y):
            ...

    要实际实例化参数化测试，应调用其中一个 instantiate_parametrized_tests() 或
    instantiate_device_type_tests()。前者适用于测试类的情况
    """
    def __init__(self, arg_str, arg_values, name_fn=None):
        # 初始化方法，接受参数名称字符串和参数值列表，以及可选的命名函数
        self.arg_names: List[str] = [s.strip() for s in arg_str.split(',') if s != '']
        # 将参数名称字符串按逗号分隔，并去除空格，存储为参数名称列表
        self.arg_values = arg_values
        # 存储参数值列表或参数值元组
        self.name_fn = name_fn
        # 存储用于生成子测试名称的可选函数

    def _formatted_str_repr(self, idx, name, value):
        """ Returns a string representation for the given arg that is suitable for use in test function names. """
        # 根据给定的参数值生成适合用于测试函数名称的字符串表示
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        # 如果值是 torch 数据类型，则返回其名称
        elif isinstance(value, torch.device):
            return str(value)
        # 如果值是 torch 设备类型，则返回其字符串表示
        # 无法使用 isinstance 避免循环导入
        elif type(value).__name__ in {'OpInfo', 'ModuleInfo'}:
            return value.formatted_name
        # 如果值的类型名称为 'OpInfo' 或 'ModuleInfo'，则返回其格式化名称
        elif isinstance(value, (int, float, str)):
            return f"{name}_{str(value).replace('.', '_')}"
        # 如果值是整数、浮点数或字符串，则返回格式化后的名称
        else:
            return f"{name}{idx}"
        # 否则返回名称加索引的形式作为默认字符串表示

    def _default_subtest_name(self, idx, values):
        # 默认的子测试名称生成方法，根据参数名称和对应值生成名称字符串
        return '_'.join([self._formatted_str_repr(idx, a, v) for a, v in zip(self.arg_names, values)])
        # 使用参数名称和对应值调用 _formatted_str_repr 方法生成字符串，并连接成一个用下划线分隔的字符串

    def _get_subtest_name(self, idx, values, explicit_name=None):
        # 获取子测试名称的方法，根据给定条件生成子测试名称字符串
        if explicit_name:
            subtest_name = explicit_name
            # 如果有显式名称，直接使用显式名称
        elif self.name_fn:
            subtest_name = self.name_fn(*values)
            # 如果存在命名函数，使用命名函数生成名称
        else:
            subtest_name = self._default_subtest_name(idx, values)
            # 否则使用默认的子测试名称生成方法生成名称
        return subtest_name
        # 返回生成的子测试名称字符串
    # 定义一个用于参数化测试的私有方法，接受测试函数、通用类和设备类作为参数
    def _parametrize_test(self, test, generic_cls, device_cls):
        # 如果参数名列表为空，则不需要额外的参数进行测试
        if len(self.arg_names) == 0:
            # 没有额外参数需要添加到测试中
            test_name = ''
            # 返回一个生成器，生成包含测试函数、测试名称为空字符串、空字典和空装饰器列表的元组
            yield (test, test_name, {}, lambda _: [])
        else:
            # 否则，遍历参数值列表，参数名与参数值应该一一对应
            values = check_exhausted_iterator = object()
            for idx, values in enumerate(self.arg_values):
                maybe_name = None

                decorators = []
                # 如果参数值是一个子测试实例
                if isinstance(values, subtest):
                    sub = values
                    # 使用子测试的参数值和名称
                    values = sub.arg_values
                    maybe_name = sub.name

                    # 包装测试函数，用子测试的装饰器装饰
                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorators = sub.decorators
                    gen_test = test_wrapper
                else:
                    gen_test = test

                # 如果参数名数量大于1，则将参数值列表转换为列表；否则将其作为单个值的列表
                values = list(values) if len(self.arg_names) > 1 else [values]
                # 如果参数值数量与参数名数量不匹配，则抛出运行时错误
                if len(values) != len(self.arg_names):
                    raise RuntimeError(f'Expected # values == # arg names, but got: {len(values)} '
                                       f'values and {len(self.arg_names)} names for test "{test.__name__}"')

                # 创建参数关键字字典，将参数名与参数值一一对应
                param_kwargs = dict(zip(self.arg_names, values))

                # 获取子测试的名称，如果没有显式提供名称则使用默认生成的名称
                test_name = self._get_subtest_name(idx, values, explicit_name=maybe_name)

                # 定义一个返回装饰器列表的装饰函数
                def decorator_fn(_, decorators=decorators):
                    return decorators

                # 返回生成器，生成包含测试函数、测试名称、参数关键字字典和装饰函数的元组
                yield (gen_test, test_name, param_kwargs, decorator_fn)

            # 如果 values 还是 check_exhausted_iterator，说明参数值为空，抛出值错误
            if values is check_exhausted_iterator:
                raise ValueError(f'{test}: An empty arg_values was passed to @parametrize. '
                                 'Note that this may result from reuse of a generator.')
class decorateIf(_TestParametrizer):
    """
    Decorator for applying parameter-specific conditional decoration.
    Composes with other test parametrizers (e.g. @modules, @ops, @parametrize, etc.).

    Examples::

        @decorateIf(unittest.skip, lambda params: params["x"] == 2)
        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["x"] == 3 and params["y"] == "baz"
        )
        def test_bar(self, x, y):
            ...

        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["op"].name == "add" and params["dtype"] == torch.float16
        )
        @ops(op_db)
        def test_op_foo(self, device, dtype, op):
            ...

        @decorateIf(
            unittest.skip,
            lambda params: params["module_info"].module_cls is torch.nn.Linear and \
                params["device"] == "cpu"
        )
        @modules(module_db)
        def test_module_foo(self, device, dtype, module_info):
            ...

    Args:
        decorator: Test decorator to apply if the predicate is satisfied.
        predicate_fn (Callable): Function taking in a dict of params and returning a boolean
            indicating whether the decorator should be applied or not.
    """
    def __init__(self, decorator, predicate_fn):
        self.decorator = decorator  # 存储传入的装饰器函数或类
        self.predicate_fn = predicate_fn  # 存储传入的条件判断函数

    def _parametrize_test(self, test, generic_cls, device_cls):

        # Leave test as-is and return the appropriate decorator_fn.
        def decorator_fn(params, decorator=self.decorator, predicate_fn=self.predicate_fn):
            if predicate_fn(params):  # 根据传入的参数调用条件判断函数，决定是否应用装饰器
                return [decorator]
            else:
                return []

        @wraps(test)
        def test_wrapper(*args, **kwargs):
            return test(*args, **kwargs)  # 保持测试函数不变，并返回包装后的测试函数

        test_name = ''  # 初始化测试名称为空字符串
        yield (test_wrapper, test_name, {}, decorator_fn)  # 返回装饰后的测试函数、测试名称、空字典和装饰器函数


class ProfilingMode(Enum):
    LEGACY = 1  # 定义枚举值，表示不同的性能分析模式
    SIMPLE = 2
    PROFILING = 3

def cppProfilingFlagsToProfilingMode():
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)  # 启用 Torch 的性能分析执行器
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)  # 获取当前图形执行优化状态
    torch._C._jit_set_profiling_executor(old_prof_exec_state)  # 恢复原始性能分析执行器状态
    torch._C._get_graph_executor_optimize(old_prof_mode_state)  # 恢复原始图形执行优化状态

    if old_prof_exec_state:
        if old_prof_mode_state:
            return ProfilingMode.PROFILING  # 如果启用了性能分析执行器并且图形执行优化也启用，则返回 PROFILING 模式
        else:
            return ProfilingMode.SIMPLE  # 如果启用了性能分析执行器但图形执行优化未启用，则返回 SIMPLE 模式
    else:
        return ProfilingMode.LEGACY  # 如果未启用性能分析执行器，则返回 LEGACY 模式

@contextmanager
def enable_profiling_mode_for_profiling_tests():
    if GRAPH_EXECUTOR == ProfilingMode.PROFILING:  # 如果全局变量 GRAPH_EXECUTOR 的值为 PROFILING 模式
        old_prof_exec_state = torch._C._jit_set_profiling_executor(True)  # 启用 Torch 的性能分析执行器
        old_prof_mode_state = torch._C._get_graph_executor_optimize(True)  # 获取当前图形执行优化状态
    try:
        yield  # 返回执行上下文
    finally:
        # 最终执行块，无论如何都会执行的代码块
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            # 如果当前的执行模式是性能分析模式
            # 恢复旧的性能执行器状态
            torch._C._jit_set_profiling_executor(old_prof_exec_state)
            # 恢复旧的性能优化模式状态
            torch._C._get_graph_executor_optimize(old_prof_mode_state)
@contextmanager
def enable_profiling_mode():
    # 保存当前的 JIT 分析执行器状态，并设置为启用分析模式
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    # 保存当前的图执行优化状态，并设置为获取图执行优化信息
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield  # 执行上下文管理器代码块
    finally:
        # 恢复之前保存的 JIT 分析执行器状态
        torch._C._jit_set_profiling_executor(old_prof_exec_state)
        # 恢复之前保存的图执行优化状态
        torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def num_profiled_runs(num_runs):
    # 保存当前的 JIT 分析运行次数，并设置为指定的运行次数
    old_num_runs = torch._C._jit_set_num_profiled_runs(num_runs)
    try:
        yield  # 执行上下文管理器代码块
    finally:
        # 恢复之前保存的 JIT 分析运行次数
        torch._C._jit_set_num_profiled_runs(old_num_runs)

func_call = torch._C.ScriptFunction.__call__
meth_call = torch._C.ScriptMethod.__call__

def prof_callable(callable, *args, **kwargs):
    if 'profile_and_replay' in kwargs:
        del kwargs['profile_and_replay']
        # 如果在参数中发现 'profile_and_replay'，并且 GRAPH_EXECUTOR 为 PROFILING 模式，则启用分析模式测试
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            with enable_profiling_mode_for_profiling_tests():
                callable(*args, **kwargs)
                return callable(*args, **kwargs)

    return callable(*args, **kwargs)

def prof_func_call(*args, **kwargs):
    # 使用 prof_callable 函数处理 func_call 的调用
    return prof_callable(func_call, *args, **kwargs)

def prof_meth_call(*args, **kwargs):
    # 使用 prof_callable 函数处理 meth_call 的调用
    return prof_callable(meth_call, *args, **kwargs)

# 重载 ScriptFunction 的 __call__ 方法，使用 prof_func_call 函数
torch._C.ScriptFunction.__call__ = prof_func_call  # type: ignore[method-assign]
# 重载 ScriptMethod 的 __call__ 方法，使用 prof_meth_call 函数
torch._C.ScriptMethod.__call__ = prof_meth_call  # type: ignore[method-assign]

def _get_test_report_path():
    # 允许用户覆盖测试报告文件的位置，因为分布式测试会使用不同配置多次运行同一测试文件
    override = os.environ.get('TEST_REPORT_SOURCE_OVERRIDE')
    test_source = override if override is not None else 'python-unittest'
    return os.path.join('test-reports', test_source)

# 检查是否通过 run_test.py 运行
is_running_via_run_test = "run_test.py" in getattr(__main__, "__file__", "")
# 创建参数解析器，根据是否通过 run_test.py 运行决定是否添加帮助信息
parser = argparse.ArgumentParser(add_help=not is_running_via_run_test, allow_abbrev=False)
parser.add_argument('--subprocess', action='store_true',
                    help='whether to run each test in a subprocess')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
parser.add_argument('--jit-executor', '--jit_executor', type=str)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--test-bailouts', '--test_bailouts', action='store_true')
parser.add_argument('--use-pytest', action='store_true')
parser.add_argument('--save-xml', nargs='?', type=str,
                    const=_get_test_report_path(),
                    default=_get_test_report_path() if IS_CI else None)  # noqa: F821
parser.add_argument('--discover-tests', action='store_true')
parser.add_argument('--log-suffix', type=str, default="")
parser.add_argument('--run-parallel', type=int, default=1)
parser.add_argument('--import-slow-tests', type=str, nargs='?', const=DEFAULT_SLOW_TESTS_FILE)
parser.add_argument('--import-disabled-tests', type=str, nargs='?', const=DEFAULT_DISABLED_TESTS_FILE)
parser.add_argument('--rerun-disabled-tests', action='store_true')
parser.add_argument('--pytest-single-test', type=str, nargs=1)

# 仅在命令行参数中包含 -h 或 --help 时运行，用于显示单元测试和解析器的帮助信息
def run_unittest_help(argv):
    unittest.main(argv=argv)

if '-h' in sys.argv or '--help' in sys.argv:
    # 创建一个线程来运行帮助函数，并启动该线程
    help_thread = threading.Thread(target=run_unittest_help, args=(sys.argv,))
    help_thread.start()
    # 等待帮助线程执行完成
    help_thread.join()

# 解析命令行参数，获取程序运行所需的设置
args, remaining = parser.parse_known_args()

# 根据 args.jit_executor 的值设置 GRAPH_EXECUTOR 的模式
if args.jit_executor == 'legacy':
    GRAPH_EXECUTOR = ProfilingMode.LEGACY
elif args.jit_executor == 'profiling':
    GRAPH_EXECUTOR = ProfilingMode.PROFILING
elif args.jit_executor == 'simple':
    GRAPH_EXECUTOR = ProfilingMode.SIMPLE
else:
    # 根据默认设置推断出 PROFILING 模式的 cppProfilingFlagsToProfilingMode() 函数返回值
    GRAPH_EXECUTOR = cppProfilingFlagsToProfilingMode()

# 根据命令行参数设置是否重新运行禁用的测试
RERUN_DISABLED_TESTS = args.rerun_disabled_tests

# 设置导入慢速测试的文件名
SLOW_TESTS_FILE = args.import_slow_tests

# 设置导入禁用测试的文件名
DISABLED_TESTS_FILE = args.import_disabled_tests

# 设置日志后缀
LOG_SUFFIX = args.log_suffix

# 设置是否并行运行测试
RUN_PARALLEL = args.run_parallel

# 设置测试中断处理方式
TEST_BAILOUTS = args.test_bailouts

# 设置是否使用 Pytest 运行测试
USE_PYTEST = args.use_pytest

# 设置单个 Pytest 测试的名称
PYTEST_SINGLE_TEST = args.pytest_single_test

# 设置是否发现测试
TEST_DISCOVER = args.discover_tests

# 设置是否在子进程中运行测试
TEST_IN_SUBPROCESS = args.subprocess

# 设置是否保存 XML 测试结果
TEST_SAVE_XML = args.save_xml

# 设置测试重复运行次数
REPEAT_COUNT = args.repeat

# 设置随机种子
SEED = args.seed

# 如果 expecttest.ACCEPT 未被设置过，则设置为 args.accept
if not getattr(expecttest, "ACCEPT", False):
    expecttest.ACCEPT = args.accept

# 将 sys.argv[0] 和剩余未解析的参数组成 UNITTEST_ARGS
UNITTEST_ARGS = [sys.argv[0]] + remaining

# 设置 Torch 的随机种子
torch.manual_seed(SEED)

# 设置 CI 环境下的测试前缀路径
CI_TEST_PREFIX = str(Path(os.getcwd()))

# 设置 CI 环境下 Pytorch 根目录路径
CI_PT_ROOT = str(Path(os.getcwd()).parent)

# 设置 CI 环境下 Functorch 根目录路径
CI_FUNCTORCH_ROOT = str(os.path.join(Path(os.getcwd()).parent, "functorch"))

# 等待子进程结束，并处理可能出现的异常情况
def wait_for_process(p, timeout=None):
    try:
        return p.wait(timeout=timeout)
    except KeyboardInterrupt:
        # 给进程一些时间处理 KeyboardInterrupt
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            return exit_status
        else:
            # 如果超时或无法获取退出状态，则强制杀死进程
            p.kill()
            raise
    except subprocess.TimeoutExpired:
        # 发送 SIGINT 信号给进程，让其有机会输出已收集的错误信息
        p.send_signal(signal.SIGINT)
        exit_status = None
        try:
            exit_status = p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if exit_status is not None:
            return exit_status
        else:
            # 如果无法获取退出状态，则强制杀死进程
            p.kill()
        raise
    except:  # noqa: B001,E722, 从 Python 核心库中复制的异常处理方式
        # 发生任何异常情况时，强制杀死进程
        p.kill()
        raise
    finally:
        # 最终始终调用 p.wait() 确保进程退出
        p.wait()

# 执行 shell 命令，并可指定工作目录、环境变量、输出流、超时等参数
def shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # 下面这段酷炫的代码片段是从 Python 3 核心库 subprocess.call 复制而来的
    # 只有以下改动：
    #   1. 添加了用于处理 SIGINT 的 `except KeyboardInterrupt` 块。
    #   2. 在 Python 2 中，subprocess.Popen 不返回上下文管理器，因此我们在 `finally` 块中使用 `p.wait()` 来使代码可移植。
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    assert not isinstance(command, str), "Command to shell should be a list or tuple of tokens"
    # 使用 subprocess.Popen 启动子进程，传入的参数包括命令列表、统一的新行模式、工作目录、环境变量、标准输出和标准错误
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    # 调用 wait_for_process 函数等待进程完成，并指定超时时间
    return wait_for_process(p, timeout=timeout)
# 定义一个函数 retry_shell，用于执行 shell 命令，支持重试操作，并返回执行结果码和是否重试的标志
def retry_shell(
    command,         # 要执行的 shell 命令
    cwd=None,        # 执行命令的当前工作目录
    env=None,        # 执行命令时使用的环境变量
    stdout=None,     # 标准输出流，用于输出命令执行的信息
    stderr=None,     # 标准错误流，用于输出命令执行的错误信息
    timeout=None,    # 命令执行的超时时间
    retries=1,       # 命令执行失败时的重试次数
    was_rerun=False, # 是否已经重试过的标志
) -> Tuple[int, bool]:  # 返回一个元组，包含执行结果码和是否重试的布尔值
    # 检查重试次数是否为非负数，如果为负数则抛出异常
    assert retries >= 0, f"Expecting non negative number for number of retries, got {retries}"
    
    try:
        # 执行 shell 命令，并获取退出码
        exit_code = shell(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout
        )
        # 如果命令执行成功或者已经没有重试次数，则直接返回退出码和重试标志
        if exit_code == 0 or retries == 0:
            return exit_code, was_rerun
        
        # 输出命令执行失败的信息，准备进行重试
        print(
            f"Got exit code {exit_code}, retrying (retries left={retries})",
            file=stdout,
            flush=True,
        )
    
    except subprocess.TimeoutExpired:
        # 如果命令执行超时，并且已经没有重试次数，则返回超时退出码和重试标志
        if retries == 0:
            print(
                f"Command took >{timeout // 60}min, returning 124",
                file=stdout,
                flush=True,
            )
            return 124, was_rerun
        
        # 输出命令执行超时的信息，准备进行重试
        print(
            f"Command took >{timeout // 60}min, retrying (retries left={retries})",
            file=stdout,
            flush=True,
        )
    
    # 递归调用自身，进行命令重试，并返回结果
    return retry_shell(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        retries=retries - 1,
        was_rerun=True,
    )


# 递归地发现测试用例，如果参数是 unittest.TestCase 类型，则直接返回其列表，否则遍历参数中的每个元素
def discover_test_cases_recursively(suite_or_case):
    if isinstance(suite_or_case, unittest.TestCase):
        return [suite_or_case]
    rc = []
    for element in suite_or_case:
        print(element)  # 打印当前元素，用于调试
        rc.extend(discover_test_cases_recursively(element))  # 递归调用自身，获取所有测试用例
    return rc


# 获取测试用例的名称列表，返回形式为 ['module.test_case']，只保留每个测试用例的最后两级名称
def get_test_names(test_cases):
    return ['.'.join(case.id().split('.')[-2:]) for case in test_cases]


# 打印当前模块中所有测试用例的名称
def _print_test_names():
    suite = unittest.TestLoader().loadTestsFromModule(__main__)  # 从当前模块加载测试套件
    test_cases = discover_test_cases_recursively(suite)  # 递归获取所有测试用例
    for name in get_test_names(test_cases):
        print(name)  # 打印测试用例的名称


# 将列表分成 nchunks 个子列表，返回一个包含子列表的列表
def chunk_list(lst, nchunks):
    return [lst[i::nchunks] for i in range(nchunks)]


# 清理文件名，例如将 'distributed/pipeline/sync/skip/test_api.py' 转换为 'distributed.pipeline.sync.skip.test_api'
def sanitize_test_filename(filename):
    # 如果文件名以 CI_TEST_PREFIX 开头，将其截断到第一个 '/' 后面的部分
    if filename.startswith(CI_TEST_PREFIX):
        filename = filename[len(CI_TEST_PREFIX) + 1:]
    strip_py = re.sub(r'.py$', '', filename)  # 去除文件名末尾的 '.py' 后缀
    return re.sub('/', r'.', strip_py)  # 将 '/' 替换为 '.'


# 对测试套件中的测试用例进行扩展名检查，返回是否所有测试用例扩展名检查都成功的布尔值
def lint_test_case_extension(suite):
    succeed = True
    # 遍历测试套件中的每一个测试用例或测试套件
    for test_case_or_suite in suite:
        # 将当前遍历到的对象视为测试用例
        test_case = test_case_or_suite
        
        # 检查当前对象是否为unittest的TestSuite实例
        if isinstance(test_case_or_suite, unittest.TestSuite):
            # 获取第一个测试用例对象
            first_test = test_case_or_suite._tests[0] if len(test_case_or_suite._tests) > 0 else None
            
            # 如果第一个测试用例存在且也是一个TestSuite，则递归调用lint_test_case_extension函数
            if first_test is not None and isinstance(first_test, unittest.TestSuite):
                return succeed and lint_test_case_extension(test_case_or_suite)
            
            # 将当前对象更新为第一个测试用例对象
            test_case = first_test

        # 如果当前测试用例对象存在
        if test_case is not None:
            # 从测试用例的ID中提取测试类名
            test_class = test_case.id().split('.', 1)[1].split('.')[0]
            
            # 如果当前测试用例不是TestCase的实例，输出错误信息并标记测试失败
            if not isinstance(test_case, TestCase):
                err = "This test class should extend from torch.testing._internal.common_utils.TestCase but it doesn't."
                print(f"{test_class} - failed. {err}")
                succeed = False
    
    # 返回最终的测试结果，表示所有测试用例是否成功执行
    return succeed
def get_report_path(argv=UNITTEST_ARGS, pytest=False):
    # 根据传入的参数获取测试文件名，并清理文件名
    test_filename = sanitize_test_filename(argv[0])
    # 构建保存测试报告的路径
    test_report_path = TEST_SAVE_XML + LOG_SUFFIX
    # 在路径中添加测试文件名
    test_report_path = os.path.join(test_report_path, test_filename)
    # 如果是 pytest 测试，则调整路径中的文件名前缀，并创建必要的目录结构
    if pytest:
        test_report_path = test_report_path.replace('python-unittest', 'python-pytest')
        os.makedirs(test_report_path, exist_ok=True)
        # 在路径中添加随机生成的唯一标识，并以 xml 格式保存
        test_report_path = os.path.join(test_report_path, f"{test_filename}-{os.urandom(8).hex()}.xml")
        return test_report_path  # 返回最终的测试报告路径
    # 如果不是 pytest 测试，则仅创建必要的目录结构
    os.makedirs(test_report_path, exist_ok=True)
    return test_report_path  # 返回最终的测试报告路径


def sanitize_pytest_xml(xml_file: str):
    # 对 pytest 生成的 xml 文件进行处理，使其更类似于 unittest 生成的 xml 格式
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    # 遍历 xml 文件中的每个 'testcase' 元素
    for testcase in tree.iter('testcase'):
        full_classname = testcase.attrib.get("classname")
        if full_classname is None:
            continue
        # 解析类名和文件名，将类名规范化并更新 'classname' 和 'file' 属性
        regex_result = re.search(r"^(test\.)?(?P<file>.*)\.(?P<classname>[^\.]*)$", full_classname)
        if regex_result is None:
            continue
        classname = regex_result.group("classname")
        file = regex_result.group("file").replace(".", "/")
        testcase.set("classname", classname)
        testcase.set("file", f"{file}.py")
    tree.write(xml_file)  # 更新后的 xml 写回文件


def get_pytest_test_cases(argv: List[str]) -> List[str]:
    # 定义 pytest 的测试收集插件类
    class TestCollectorPlugin:
        def __init__(self):
            self.tests = []

        def pytest_collection_finish(self, session):
            # 收集所有测试用例的相对路径
            for item in session.items:
                self.tests.append(session.config.cwd_relative_nodeid(item.nodeid))

    test_collector_plugin = TestCollectorPlugin()
    import pytest
    # 运行 pytest 收集测试用例，返回测试用例的路径列表
    pytest.main(
        [arg for arg in argv if arg != '-vv'] + ['--collect-only', '-qq', '--use-main-module'],
        plugins=[test_collector_plugin]
    )
    return test_collector_plugin.tests  # 返回收集到的测试用例路径列表


def run_tests(argv=UNITTEST_ARGS):
    # 导入测试文件
    if SLOW_TESTS_FILE:
        # 如果设置了慢速测试文件并且文件存在，则加载其中的测试列表
        if os.path.exists(SLOW_TESTS_FILE):
            with open(SLOW_TESTS_FILE) as fp:
                global slow_tests_dict
                slow_tests_dict = json.load(fp)
                # 将环境变量设置为使 pytest-xdist 子进程仍然可以访问它们
                os.environ['SLOW_TESTS_FILE'] = SLOW_TESTS_FILE
        else:
            # 如果慢速测试文件不存在，则发出警告
            warnings.warn(f'slow test file provided but not found: {SLOW_TESTS_FILE}')
    if DISABLED_TESTS_FILE:
        # 如果设置了禁用测试文件并且文件存在，则加载其中的测试列表
        if os.path.exists(DISABLED_TESTS_FILE):
            with open(DISABLED_TESTS_FILE) as fp:
                global disabled_tests_dict
                disabled_tests_dict = json.load(fp)
                # 设置环境变量以供 pytest-xdist 子进程访问
                os.environ['DISABLED_TESTS_FILE'] = DISABLED_TESTS_FILE
        else:
            # 如果禁用测试文件不存在，则发出警告
            warnings.warn(f'disabled test file provided but not found: {DISABLED_TESTS_FILE}')
    # 确定测试启动机制
    if TEST_DISCOVER:
        # 如果设置了TEST_DISCOVER，则打印测试名称并返回
        _print_test_names()
        return

    # 在运行测试之前，进行代码检查以确保每个测试类都是TestCase的子类
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    if not lint_test_case_extension(suite):
        # 如果代码检查未通过，退出程序并返回状态码1
        sys.exit(1)

    # 如果设置了TEST_IN_SUBPROCESS
    if TEST_IN_SUBPROCESS:
        other_args = []
        # 如果定义了DISABLED_TESTS_FILE，则添加参数"--import-disabled-tests"
        if DISABLED_TESTS_FILE:
            other_args.append("--import-disabled-tests")
        # 如果定义了SLOW_TESTS_FILE，则添加参数"--import-slow-tests"
        if SLOW_TESTS_FILE:
            other_args.append("--import-slow-tests")
        # 如果使用PYTEST，则添加参数"--use-pytest"
        if USE_PYTEST:
            other_args.append("--use-pytest")
        # 如果设置了RERUN_DISABLED_TESTS，则添加参数"--rerun-disabled-tests"
        if RERUN_DISABLED_TESTS:
            other_args.append("--rerun-disabled-tests")
        # 如果设置了TEST_SAVE_XML，则添加参数'--save-xml'和对应的文件名参数args.save_xml
        if TEST_SAVE_XML:
            other_args += ['--save-xml', args.save_xml]

        # 获取测试用例列表
        test_cases = (
            get_pytest_test_cases(argv) if USE_PYTEST else
            [case.id().split('.', 1)[1] for case in discover_test_cases_recursively(suite)]
        )

        failed_tests = []

        # 遍历每个测试用例全名
        for test_case_full_name in test_cases:

            # 构建执行测试的命令
            cmd = (
                [sys.executable] + [argv[0]] + other_args + argv[1:] +
                (["--pytest-single-test"] if USE_PYTEST else []) +
                [test_case_full_name]
            )
            string_cmd = " ".join(cmd)

            # 设置超时时间
            timeout = None if RERUN_DISABLED_TESTS else 15 * 60

            # 重试执行命令
            exitcode, _ = retry_shell(cmd, timeout=timeout, retries=0 if RERUN_DISABLED_TESTS else 1)

            # 如果退出码不为0，记录失败的测试用例
            if exitcode != 0:
                # 对于特定测试用例（例如'TestDistBackendWithSpawn'），添加相关的环境变量以支持分布式测试
                if 'TestDistBackendWithSpawn' in test_case_full_name:
                    backend = os.environ.get("BACKEND", "")
                    world_size = os.environ.get("WORLD_SIZE", "")
                    env_prefix = f"BACKEND={backend} WORLD_SIZE={world_size}"
                    string_cmd = env_prefix + " " + string_cmd
                # 打印出导致失败的测试命令以便复现
                print(f"Test exited with non-zero exitcode {exitcode}. Command to reproduce: {string_cmd}")
                failed_tests.append(test_case_full_name)

            # 断言没有失败的测试用例
            assert len(failed_tests) == 0, "{} unit test(s) failed:\n\t{}".format(
                len(failed_tests), '\n\t'.join(failed_tests))

    # 如果设置了RUN_PARALLEL > 1，则并行运行测试用例
    elif RUN_PARALLEL > 1:
        # 递归地发现所有测试用例
        test_cases = discover_test_cases_recursively(suite)
        # 将测试用例分成多个批次，每个批次包含RUN_PARALLEL个测试用例
        test_batches = chunk_list(get_test_names(test_cases), RUN_PARALLEL)
        processes = []
        # 对每个批次创建一个子进程
        for i in range(RUN_PARALLEL):
            command = [sys.executable] + argv + [f'--log-suffix=-shard-{i + 1}'] + test_batches[i]
            processes.append(subprocess.Popen(command, universal_newlines=True))
        failed = False
        # 等待所有子进程执行完毕，并检查是否有进程返回非0退出码
        for p in processes:
            failed |= wait_for_process(p) != 0
        # 断言没有测试分片失败
        assert not failed, "Some test shards have failed"
    elif USE_PYTEST:
        # 如果使用 pytest 运行测试，则准备 pytest 的参数列表
        pytest_args = argv + ["--use-main-module"]
        # 如果需要保存 XML 测试报告，获取报告路径并打印提示信息
        if TEST_SAVE_XML:
            test_report_path = get_report_path(pytest=True)
            print(f'Test results will be stored in {test_report_path}')
            # 添加 pytest 的选项以生成 XML 报告
            pytest_args.append(f'--junit-xml={test_report_path}')
        # 如果有指定单个测试用例，只运行该测试用例
        if PYTEST_SINGLE_TEST:
            pytest_args = PYTEST_SINGLE_TEST + pytest_args[1:]

        import pytest
        # 设置环境变量以禁用彩色输出
        os.environ["NO_COLOR"] = "1"
        # 运行 pytest 并获取退出码
        exit_code = pytest.main(args=pytest_args)
        # 如果需要保存 XML 测试报告，对报告进行处理以消除敏感信息
        if TEST_SAVE_XML:
            sanitize_pytest_xml(test_report_path)

        # 如果不禁用重新运行失败的测试，根据退出码决定退出状态
        if not RERUN_DISABLED_TESTS:
            # 退出码为 5 表示没有找到测试，这在某些测试配置不运行某些文件的情况下会发生
            sys.exit(0 if exit_code == 5 else exit_code)
        else:
            # 在禁用重新运行失败测试模式下，仅记录测试报告并始终返回成功状态码
            sys.exit(0)
    elif TEST_SAVE_XML is not None:
        # 为了非 CI 环境不需要安装 xmlrunner，因此在此处导入
        import xmlrunner  # type: ignore[import]
        from xmlrunner.result import _XMLTestResult  # type: ignore[import]

        class XMLTestResultVerbose(_XMLTestResult):
            """
            添加测试输出的详细信息：
            默认情况下，测试摘要只打印 'skip'，
            但我们希望还能打印跳过的原因。
            GH issue: https://github.com/pytorch/pytorch/issues/69014

            这与 unittest_xml_reporting<=3.2.0,>=2.0.0 兼容
            (3.2.0 是当前最新版本)
            """
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def addSkip(self, test, reason):
                # 在添加跳过测试的信息时，增加对跳过原因的详细输出
                super().addSkip(test, reason)
                for c in self.callback.__closure__:
                    if isinstance(c.cell_contents, str) and c.cell_contents == 'skip':
                        # 这条消息将打印在测试摘要中；
                        # 它代表在闭包中捕获的 `verbose_str`
                        c.cell_contents = f"skip: {reason}"

            def printErrors(self) -> None:
                # 打印错误信息时，增强显示测试失败的详细信息
                super().printErrors()
                self.printErrorList("XPASS", self.unexpectedSuccesses)

        # 获取 XML 测试报告的保存路径
        test_report_path = get_report_path()
        # 检查是否指定了详细输出选项
        verbose = '--verbose' in argv or '-v' in argv
        if verbose:
            print(f'Test results will be stored in {test_report_path}')
        # 运行 unittest 并使用 xmlrunner.XMLTestRunner 作为测试运行器
        unittest.main(argv=argv, testRunner=xmlrunner.XMLTestRunner(
            output=test_report_path,
            verbosity=2 if verbose else 1,
            resultclass=XMLTestResultVerbose))
    elif REPEAT_COUNT > 1:
        # 如果需要重复运行测试多次
        for _ in range(REPEAT_COUNT):
            # 如果有测试失败，则返回非零退出码
            if not unittest.main(exit=False, argv=argv).result.wasSuccessful():
                sys.exit(-1)
    else:
        # 否则，正常运行 unittest 测试套件
        unittest.main(argv=argv)
# 判断当前操作系统是否为 Linux
IS_LINUX = sys.platform == "linux"
# 判断当前操作系统是否为 Windows
IS_WINDOWS = sys.platform == "win32"
# 判断当前操作系统是否为 macOS
IS_MACOS = sys.platform == "darwin"
# 判断当前机器是否为 PowerPC 架构
IS_PPC = platform.machine() == "ppc64le"
# 判断当前机器是否为 x86 架构
IS_X86 = platform.machine() in ('x86_64', 'i386')
# 判断当前机器是否为 ARM64 架构
IS_ARM64 = platform.machine() in ('arm64', 'aarch64')

# 检查当前系统是否支持 AVX-512 VNNI 指令集
def is_avx512_vnni_supported():
    # 如果不是 Linux 系统，则不支持 AVX-512 VNNI
    if sys.platform != 'linux':
        return False
    # 读取 /proc/cpuinfo 文件内容，查看是否包含 "vnni" 关键字
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return "vnni" in lines

# 存储当前系统是否支持 AVX-512 VNNI 指令集的结果
IS_AVX512_VNNI_SUPPORTED = is_avx512_vnni_supported()

# 如果是 Windows 系统，定义一个临时文件名生成器的上下文管理器
if IS_WINDOWS:
    @contextmanager
    def TemporaryFileName(*args, **kwargs):
        # 在 Windows 上，NamedTemporaryFile 会打开文件，且文件无法多次打开，所以需要手动关闭和删除
        if 'delete' in kwargs:
            if kwargs['delete'] is not False:
                raise UserWarning("only TemporaryFileName with delete=False is supported on Windows.")
        else:
            kwargs['delete'] = False
        # 创建临时文件，并在退出时手动删除
        f = tempfile.NamedTemporaryFile(*args, **kwargs)
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)
# 如果不是 Windows 系统，定义另一种临时文件名生成器的上下文管理器
else:
    @contextmanager  # noqa: T484
    def TemporaryFileName(*args, **kwargs):
        # 使用 tempfile.NamedTemporaryFile 创建临时文件，并在退出时自动删除
        with tempfile.NamedTemporaryFile(*args, **kwargs) as f:
            yield f.name

# 如果是 Windows 系统，定义一个临时目录名生成器的上下文管理器
if IS_WINDOWS:
    @contextmanager
    def TemporaryDirectoryName(suffix=None):
        # 在 Windows 上，TemporaryDirectory 创建的目录可能会过早删除，所以先使用 mkdtemp 创建目录，退出时手动删除
        try:
            dir_name = tempfile.mkdtemp(suffix=suffix)
            yield dir_name
        finally:
            shutil.rmtree(dir_name)
# 如果不是 Windows 系统，定义另一种临时目录名生成器的上下文管理器
else:
    @contextmanager  # noqa: T484
    def TemporaryDirectoryName(suffix=None):
        # 使用 tempfile.TemporaryDirectory 创建临时目录，并在退出时自动删除
        with tempfile.TemporaryDirectory(suffix=suffix) as d:
            yield d

# 检查当前文件系统编码是否为 UTF-8
IS_FILESYSTEM_UTF8_ENCODING = sys.getfilesystemencoding() == 'utf-8'

# 检查是否安装了 numpy 库
TEST_NUMPY = _check_module_exists('numpy')
# 检查是否安装了 fairseq 库
TEST_FAIRSEQ = _check_module_exists('fairseq')
# 检查是否安装了 scipy 库
TEST_SCIPY = _check_module_exists('scipy')
# 检查是否支持 MKL 加速（torch.backends.mkl.is_available()）
TEST_MKL = torch.backends.mkl.is_available()
# 检查是否支持 MPS 加速（torch.backends.mps.is_available()）
TEST_MPS = torch.backends.mps.is_available()
# 检查是否支持 XPU 加速（torch.xpu.is_available()）
TEST_XPU = torch.xpu.is_available()
# 检查是否支持 HPU 加速（通过 torch.hpu.is_available() 判断）
TEST_HPU = True if (hasattr(torch, "hpu") and torch.hpu.is_available()) else False
# 检查是否支持 CUDA 加速（torch.cuda.is_available()）
TEST_CUDA = torch.cuda.is_available()
# 获取私有扩展库的模块对象
custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name(), None)
# 判断私有扩展库是否可用
custom_device_is_available = hasattr(custom_device_mod, "is_available") and custom_device_mod.is_available()
# 检查私有扩展库是否可用
TEST_PRIVATEUSE1 = True if custom_device_is_available else False
# 获取私有扩展库的设备类型
TEST_PRIVATEUSE1_DEVICE_TYPE = torch._C._get_privateuse1_backend_name()
# 检查是否安装了 numba 库
TEST_NUMBA = _check_module_exists('numba')
# 检查是否安装了 transformers 库
TEST_TRANSFORMERS = _check_module_exists('transformers')
# 检查是否安装了 dill 库
TEST_DILL = _check_module_exists('dill')

# 检查是否安装了 librosa 库，并且当前系统不是 ARM64 架构
TEST_LIBROSA = _check_module_exists('librosa') and not IS_ARM64
TEST_OPT_EINSUM = _check_module_exists('opt_einsum')

TEST_Z3 = _check_module_exists('z3')

# 定义一个函数，接受一个字符串参数 x，并返回其按逗号分隔后的列表，如果 x 为空则返回空列表
def split_if_not_empty(x: str):
    return x.split(",") if len(x) != 0 else []

# 检查环境变量 'PYTORCH_TESTING_DEVICE_EXCEPT_FOR' 中是否包含 "cpu"，并将结果存储在 NOTEST_CPU 变量中
NOTEST_CPU = "cpu" in split_if_not_empty(os.getenv('PYTORCH_TESTING_DEVICE_EXCEPT_FOR', ''))

# 如果 TEST_DILL 不可用，则跳过测试
skipIfNoDill = unittest.skipIf(not TEST_DILL, "no dill")

# 设置测试环境标志，对应不同的环境变量
TestEnvironment.def_flag("NO_MULTIPROCESSING_SPAWN", env_var="NO_MULTIPROCESSING_SPAWN")
TestEnvironment.def_flag("TEST_WITH_ASAN", env_var="PYTORCH_TEST_WITH_ASAN")
TestEnvironment.def_flag("TEST_WITH_DEV_DBG_ASAN", env_var="PYTORCH_TEST_WITH_DEV_DBG_ASAN")
TestEnvironment.def_flag("TEST_WITH_TSAN", env_var="PYTORCH_TEST_WITH_TSAN")
TestEnvironment.def_flag("TEST_WITH_UBSAN", env_var="PYTORCH_TEST_WITH_UBSAN")
TestEnvironment.def_flag("TEST_WITH_ROCM", env_var="PYTORCH_TEST_WITH_ROCM")

# TODO: Remove PYTORCH_MIOPEN_SUGGEST_NHWC once ROCm officially supports NHWC in MIOpen
# See #64427
# 检查环境变量 'PYTORCH_MIOPEN_SUGGEST_NHWC' 是否为 '1'，设置 TEST_WITH_MIOPEN_SUGGEST_NHWC 变量为布尔值
TEST_WITH_MIOPEN_SUGGEST_NHWC = os.getenv('PYTORCH_MIOPEN_SUGGEST_NHWC', '0') == '1'

# 启用运行较慢的测试（默认禁用）
TestEnvironment.def_flag("TEST_WITH_SLOW", env_var="PYTORCH_TEST_WITH_SLOW")

# 禁用非慢速测试（默认启用）
# 通常与 TEST_WITH_SLOW 结合使用，仅运行慢速测试
TestEnvironment.def_flag("TEST_SKIP_FAST", env_var="PYTORCH_TEST_SKIP_FAST")

# 启用交叉引用测试，额外进行计算并与常规计算结果进行交叉验证
# 默认情况下不运行这些测试
TestEnvironment.def_flag("TEST_WITH_CROSSREF", env_var="PYTORCH_TEST_WITH_CROSSREF")

# 检查是否满足运行 CUDA 图的条件，并设置 TEST_CUDA_GRAPH 变量
TEST_CUDA_GRAPH = TEST_CUDA and (not TEST_SKIP_CUDAGRAPH) and (
    (torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 11) or
    (torch.version.hip and float(".".join(torch.version.hip.split(".")[0:2])) >= 5.3)
)

# 如果 TEST_CUDA 可用且环境变量 'NUM_PARALLEL_PROCS' 存在，则根据环境设置 CUDA 运行时内存分配
if TEST_CUDA and 'NUM_PARALLEL_PROCS' in os.environ:
    num_procs = int(os.getenv("NUM_PARALLEL_PROCS", "2"))
    gb_available = torch.cuda.mem_get_info()[1] / 2 ** 30
    # 其他库每个进程大约占用不到 1 GB 的空间
    # 设置每个进程可用显存比例
    torch.cuda.set_per_process_memory_fraction(round((gb_available - num_procs * .85) / gb_available / num_procs, 2))

# 定义一个装饰器函数 skipIfCrossRef，用于在不满足条件时跳过交叉引用测试
def skipIfCrossRef(fn):
    @wraps(fn)
    # 定义一个装饰器函数 `wrapper`，接受任意位置参数和关键字参数
    def wrapper(*args, **kwargs):
        # 如果 TEST_WITH_CROSSREF 变量为真，则抛出 unittest.SkipTest 异常，指示测试不支持与 crossref 运行
        if TEST_WITH_CROSSREF:  # noqa: F821
            raise unittest.SkipTest("test doesn't currently with crossref")
        else:
            # 否则，调用被装饰的函数 fn，并传递所有参数和关键字参数
            fn(*args, **kwargs)
    # 返回装饰器函数 wrapper
    return wrapper
class CrossRefMode(torch.overrides.TorchFunctionMode):
    # 定义一个继承自 TorchFunctionMode 的子类 CrossRefMode
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 实现 __torch_function__ 方法，用于自定义 Torch 函数的行为
        kwargs = kwargs or {}
        # 如果 kwargs 为 None，则初始化为空字典
        r = func(*args, **kwargs)
        # 调用传入的函数 func，并传入 args 和 kwargs 参数，将结果保存到 r 中
        return r
        # 返回 func 的执行结果

# Run PyTorch tests with TorchDynamo
TestEnvironment.def_flag("TEST_WITH_TORCHINDUCTOR", env_var="PYTORCH_TEST_WITH_INDUCTOR")
# 定义一个名为 TEST_WITH_TORCHINDUCTOR 的测试环境标志，对应的环境变量为 PYTORCH_TEST_WITH_INDUCTOR

# AOT_EAGER not tested in ci, useful for debugging
TestEnvironment.def_flag("TEST_WITH_AOT_EAGER", env_var="PYTORCH_TEST_WITH_AOT_EAGER")
# 定义一个名为 TEST_WITH_AOT_EAGER 的测试环境标志，对应的环境变量为 PYTORCH_TEST_WITH_AOT_EAGER

TestEnvironment.def_flag("TEST_WITH_TORCHDYNAMO", env_var="PYTORCH_TEST_WITH_DYNAMO",
                         implied_by_fn=lambda: TEST_WITH_TORCHINDUCTOR or TEST_WITH_AOT_EAGER)  # noqa: F821
# 定义一个名为 TEST_WITH_TORCHDYNAMO 的测试环境标志，对应的环境变量为 PYTORCH_TEST_WITH_DYNAMO，
# 根据 implied_by_fn 函数来设定其隐含值，条件是 TEST_WITH_TORCHINDUCTOR 或 TEST_WITH_AOT_EAGER

if TEST_WITH_TORCHDYNAMO:  # noqa: F821
    # 如果 TEST_WITH_TORCHDYNAMO 为真
    import torch._dynamo
    # 导入 torch._dynamo 模块
    # Do not spend time on helper functions that are called with different inputs
    torch._dynamo.config.accumulated_cache_size_limit = 8
    # 设置 torch._dynamo.config 中 accumulated_cache_size_limit 的值为 8
    # Do not log compilation metrics from unit tests
    torch._dynamo.config.log_compilation_metrics = False
    # 设置 torch._dynamo.config 中 log_compilation_metrics 的值为 False
    if TEST_WITH_TORCHINDUCTOR:  # noqa: F821
        # 如果 TEST_WITH_TORCHINDUCTOR 为真
        import torch._inductor.config
        # 导入 torch._inductor.config 模块
        torch._inductor.config.fallback_random = True
        # 设置 torch._inductor.config 中 fallback_random 的值为 True


def xpassIfTorchDynamo(func):
    # 定义一个函数 xpassIfTorchDynamo，根据 TEST_WITH_TORCHDYNAMO 决定是否返回 func 或 unittest.expectedFailure(func)
    return func if TEST_WITH_TORCHDYNAMO else unittest.expectedFailure(func)  # noqa: F821


def xfailIfTorchDynamo(func):
    # 定义一个函数 xfailIfTorchDynamo，根据 TEST_WITH_TORCHDYNAMO 决定是否返回 unittest.expectedFailure(func) 或 func
    return unittest.expectedFailure(func) if TEST_WITH_TORCHDYNAMO else func  # noqa: F821


def skipIfTorchDynamo(msg="test doesn't currently work with dynamo"):
    """
    Usage:
    @skipIfTorchDynamo(msg)
    def test_blah(self):
        ...
    """
    assert isinstance(msg, str), "Are you using skipIfTorchDynamo correctly?"

    def decorator(fn):
        # 定义一个装饰器 decorator
        if not isinstance(fn, type):
            # 如果 fn 不是类
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # 定义一个 wrapper 函数，用于包装 fn
                if TEST_WITH_TORCHDYNAMO:  # noqa: F821
                    # 如果 TEST_WITH_TORCHDYNAMO 为真
                    raise unittest.SkipTest(msg)
                    # 抛出一个跳过测试的异常，信息为 msg
                else:
                    fn(*args, **kwargs)
                    # 否则调用 fn，并传入 args 和 kwargs
            return wrapper
            # 返回 wrapper 函数

        assert isinstance(fn, type)
        # 断言 fn 是一个类
        if TEST_WITH_TORCHDYNAMO:  # noqa: F821
            # 如果 TEST_WITH_TORCHDYNAMO 为真
            fn.__unittest_skip__ = True
            # 设置 fn 的 __unittest_skip__ 属性为 True
            fn.__unittest_skip_why__ = msg
            # 设置 fn 的 __unittest_skip_why__ 属性为 msg

        return fn
        # 返回 fn

    return decorator
    # 返回 decorator 函数


def skipIfTorchInductor(msg="test doesn't currently work with torchinductor",
                        condition=TEST_WITH_TORCHINDUCTOR):  # noqa: F821
    # 定义一个跳过测试装饰器，根据 TEST_WITH_TORCHINDUCTOR 决定是否跳过测试
    def decorator(fn):
        # 定义一个装饰器 decorator
        if not isinstance(fn, type):
            # 如果 fn 不是类
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # 定义一个 wrapper 函数，用于包装 fn
                if condition:
                    # 如果 condition 为真
                    raise unittest.SkipTest(msg)
                    # 抛出一个跳过测试的异常，信息为 msg
                else:
                    fn(*args, **kwargs)
                    # 否则调用 fn，并传入 args 和 kwargs
            return wrapper
            # 返回 wrapper 函数

        assert isinstance(fn, type)
        # 断言 fn 是一个类
        if condition:
            # 如果 condition 为真
            fn.__unittest_skip__ = True
            # 设置 fn 的 __unittest_skip__ 属性为 True
            fn.__unittest_skip_why__ = msg
            # 设置 fn 的 __unittest_skip_why__ 属性为 msg

        return fn
        # 返回 fn

    return decorator
    # 返回 decorator 函数


def serialTest(condition=True):
    """
    Decorator for running tests serially.  Requires pytest
    """
    # 串行测试的装饰器，需要 pytest 支持
    def decorator(fn):
        # 定义一个装饰器 decorator
        if has_pytest and condition:
            # 如果有 pytest 并且 condition 为真
            return pytest.mark.serial(fn)
            # 使用 pytest 的标记 serial 来标记 fn
        return fn
        # 否则直接返回 fn
    return decorator
# 定义一个装饰器函数 unMarkDynamoStrictTest，用于将类的 dynamo_strict 属性设置为 False
def unMarkDynamoStrictTest(cls=None):
    # 定义装饰器函数 decorator，接收一个类作为参数，并将其 dynamo_strict 属性设置为 False
    def decorator(cls):
        cls.dynamo_strict = False
        return cls

    # 如果没有传入参数 cls，则返回 decorator 函数本身
    if cls is None:
        return decorator
    else:
        # 否则，将传入的 cls 作为参数传递给 decorator 函数，并返回其结果
        return decorator(cls)


# 定义一个装饰器函数 markDynamoStrictTest，用于将测试标记为“strict”模式
def markDynamoStrictTest(cls_or_func=None, nopython=False):
    """
    Marks the test as 'strict'. In strict mode, we reset before and after the
    test, and run without suppress errors.

    Args:
    - nopython: if we should run torch._dynamo.optimize with nopython={True/False}.
    """
    # 定义装饰器函数 decorator，根据参数 cls_or_func 的类型分别设置类属性或者包装函数
    def decorator(cls_or_func):
        # 如果传入的 cls_or_func 是一个类
        if inspect.isclass(cls_or_func):
            # 设置该类的 dynamo_strict 属性为 True，并设置 dynamo_strict_nopython 属性为传入的 nopython 值
            cls_or_func.dynamo_strict = True
            cls_or_func.dynamo_strict_nopython = nopython
            return cls_or_func

        # 否则，传入的是一个函数
        fn = cls_or_func

        # 定义包装函数 wrapper，用于在运行函数前后进行 reset，并禁用错误抑制
        @wraps(fn)
        def wrapper(*args, **kwargs):
            torch._dynamo.reset()
            with unittest.mock.patch("torch._dynamo.config.suppress_errors", False):
                fn(*args, **kwargs)
            torch._dynamo.reset()
        return wrapper

    # 如果没有传入参数 cls_or_func，则返回 decorator 函数本身
    if cls_or_func is None:
        return decorator
    else:
        # 否则，将传入的 cls_or_func 作为参数传递给 decorator 函数，并返回其结果
        return decorator(cls_or_func)


# 定义一个函数 skipRocmIfTorchInductor，用于根据条件跳过测试
def skipRocmIfTorchInductor(msg="test doesn't currently work with torchinductor on the ROCm stack"):
    # 调用 skipIfTorchInductor 函数，传入消息和条件 TEST_WITH_ROCM and TEST_WITH_TORCHINDUCTOR
    return skipIfTorchInductor(msg=msg, condition=TEST_WITH_ROCM and TEST_WITH_TORCHINDUCTOR)  # noqa: F821


# 定义一个装饰器函数 skipIfLegacyJitExecutor，用于根据条件跳过使用旧 JIT executor 的测试
def skipIfLegacyJitExecutor(msg="test doesn't currently work with legacy JIT executor"):
    # 定义装饰器函数 decorator，根据函数类型或 GRAPH_EXECUTOR 的值来设置跳过测试的标记或包装函数
    def decorator(fn):
        # 如果 fn 不是一个类，则定义包装函数 wrapper
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # 如果 GRAPH_EXECUTOR 是 ProfilingMode.LEGACY，则跳过测试
                if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        # 如果 fn 是一个类
        assert isinstance(fn, type)
        # 根据 GRAPH_EXECUTOR 的值设置 unittest 的跳过标记和原因
        if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn

    return decorator


# 定义一个条件变量 TEST_WITH_TV，用于确定是否启用翻译验证
TEST_WITH_TV = os.getenv('PYTORCH_TEST_WITH_TV') == '1'

# 如果 TEST_WITH_TV 为 True，则设置 torch.fx.experimental._config.translation_validation 为 True
if TEST_WITH_TV:
    torch.fx.experimental._config.translation_validation = True

# 定义一个函数 disable_translation_validation_if_dynamic_shapes，用于根据条件禁用翻译验证
# 当 dynamic_shapes 和 translation_validation 结合时，某些测试需要禁用翻译验证
def disable_translation_validation_if_dynamic_shapes(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch._dynamo.config.dynamic_shapes:
            # 当 dynamic_shapes 为 True 时，禁用翻译验证
            torch.fx.experimental._config.translation_validation = False
        return fn(*args, **kwargs)
    return wrapper


# 决定是否启用 CUDA 内存泄漏检查
# CUDA 内存泄漏检查很昂贵，因此我们不希望在每个测试用例/配置上都执行它
# 如果该变量为 True，则跳过 CUDA 内存泄漏检查；否则执行检查。
# 设置一个测试环境的标志，用于检查 CUDA 内存泄漏
TestEnvironment.def_flag("TEST_CUDA_MEM_LEAK_CHECK", env_var="PYTORCH_TEST_CUDA_MEM_LEAK_CHECK")

# NumPy 数据类型到 PyTorch 数据类型的映射字典
numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.uint16     : torch.uint16,
    np.uint32     : torch.uint32,
    np.uint64     : torch.uint64,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

# 将 NumPy 数据类型映射为对应的 PyTorch 数据类型，考虑到 NumPy 中的类型可能是类而非实例
def numpy_to_torch_dtype(np_dtype):
    try:
        return numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return numpy_to_torch_dtype_dict[np_dtype.type]

# 检查给定的 NumPy 数据类型是否存在对应的 PyTorch 数据类型
def has_corresponding_torch_dtype(np_dtype):
    try:
        numpy_to_torch_dtype(np_dtype)
        return True
    except KeyError:
        return False

# 如果在 Windows 平台上，则将 numpy 的 intc 类型映射为 torch 的 int 类型
if IS_WINDOWS:
    # `np.intc` 的大小由平台定义
    numpy_to_torch_dtype_dict[np.intc] = torch.int

# PyTorch 数据类型到 NumPy 数据类型的映射字典，包括一些特殊映射关系
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}
torch_to_numpy_dtype_dict.update({
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64
})

# 装饰器函数：如果 nn 模块内联化，则跳过测试
def skipIfNNModuleInlined(
    msg="test doesn't currently work with nn module inlining",
    condition=torch._dynamo.config.inline_inbuilt_nn_modules,
):  # noqa: F821
    def decorator(fn):
        if not isinstance(fn, type):

            @wraps(fn)
            def wrapper(*args, **kwargs):
                if condition:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)

            return wrapper

        assert isinstance(fn, type)
        if condition:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn

    return decorator

# 装饰器函数：如果在 ROCm 平台上，则跳过测试
def skipIfRocm(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    def dec_fn(fn):
        reason = f"skipIfRocm: {msg}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if TEST_WITH_ROCM:  # noqa: F821
                raise unittest.SkipTest(reason)
            else:
                return fn(*args, **kwargs)
        return wrapper
    
    if func:
        return dec_fn(func)
    return dec_fn
# 在 ROCm 上运行测试时的装饰器，根据 TEST_WITH_ROCM 环境变量决定是否跳过测试
def runOnRocm(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_WITH_ROCM:  # noqa: F821
            fn(*args, **kwargs)  # 如果在 ROCm 上，则执行被装饰的函数
        else:
            raise unittest.SkipTest("test currently only works on the ROCm stack")  # 否则抛出跳过测试异常
    return wrapper

# 在 XPU 上跳过测试的装饰器工厂函数
def skipIfXpu(func=None, *, msg="test doesn't currently work on the XPU stack"):
    def dec_fn(fn):
        reason = f"skipIfXpu: {msg}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if TEST_XPU:  # 如果在 XPU 上，则跳过测试
                raise unittest.SkipTest(reason)
            else:
                return fn(*args, **kwargs)  # 否则执行被装饰的函数
        return wrapper
    if func:
        return dec_fn(func)
    return dec_fn

# 在 MPS 上跳过测试的装饰器
def skipIfMps(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_MPS:  # 如果在 MPS 上，则跳过测试
            raise unittest.SkipTest("test doesn't currently work with MPS")
        else:
            fn(*args, **kwargs)  # 否则执行被装饰的函数
    return wrapper

# 在 HPU 上跳过测试的装饰器
def skipIfHpu(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_HPU:  # 如果在 HPU 上，则跳过测试
            raise unittest.SkipTest("test doesn't currently work with HPU")
        else:
            fn(*args, **kwargs)  # 否则执行被装饰的函数
    return wrapper

# 如果 ROCm 的版本低于指定版本，则跳过测试的装饰器
def skipIfRocmVersionLessThan(version=None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if TEST_WITH_ROCM:  # noqa: F821
                rocm_version = str(torch.version.hip)
                rocm_version = rocm_version.split("-")[0]    # 忽略 git sha
                rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
                if rocm_version_tuple is None or version is None or rocm_version_tuple < tuple(version):
                    reason = f"ROCm {rocm_version_tuple} is available but {version} required"
                    raise unittest.SkipTest(reason)  # 如果 ROCm 版本不符合要求，则跳过测试
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

# 如果没有启用 MIOpen NHWC 激活，则跳过测试的装饰器
def skipIfNotMiopenSuggestNHWC(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_MIOPEN_SUGGEST_NHWC:  # 如果没有启用 MIOpen NHWC 激活，则跳过测试
            raise unittest.SkipTest("test doesn't currently work without MIOpen NHWC activation")
        else:
            fn(*args, **kwargs)  # 否则执行被装饰的函数
    return wrapper

# 将 linalg 后端重置为默认值，确保一个测试的失败不会影响其他测试
def setLinalgBackendsToDefaultFinally(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        _preferred_backend = torch.backends.cuda.preferred_linalg_library()
        try:
            fn(*args, **kwargs)
        finally:
            torch.backends.cuda.preferred_linalg_library(_preferred_backend)  # 最终将 linalg 后端重置为原来的设置
    return _fn

# 将 BLAS 后端重置为默认值，确保一个测试的失败不会影响其他测试
def setBlasBackendsToDefaultFinally(fn):
    @wraps(fn)
    # 定义一个内部函数 _fn，接受任意位置参数 *args 和任意关键字参数 **kwargs
    def _fn(*args, **kwargs):
        # 获取当前 CUDA 的优选 BLAS 库设置并保存
        _preferred_backend = torch.backends.cuda.preferred_blas_library()
        try:
            # 调用外部传入的函数 fn，并传递所有参数和关键字参数
            fn(*args, **kwargs)
        finally:
            # 恢复之前保存的 CUDA 优选 BLAS 库设置
            torch.backends.cuda.preferred_blas_library(_preferred_backend)
    # 返回内部函数 _fn 作为结果
    return _fn
# 上下文管理器，用于设置确定性标志并自动将其重置为原始值
class DeterministicGuard:
    def __init__(self, deterministic, *, warn_only=False, fill_uninitialized_memory=True):
        self.deterministic = deterministic  # 初始化确定性标志
        self.warn_only = warn_only  # 是否仅警告模式
        self.fill_uninitialized_memory = fill_uninitialized_memory  # 填充未初始化内存的设置

    def __enter__(self):
        # 保存当前的确定性算法启用状态和警告模式
        self.deterministic_restore = torch.are_deterministic_algorithms_enabled()
        self.warn_only_restore = torch.is_deterministic_algorithms_warn_only_enabled()
        # 保存当前的填充未初始化内存的设置
        self.fill_uninitialized_memory_restore = torch.utils.deterministic.fill_uninitialized_memory
        # 设置新的确定性算法和填充未初始化内存的配置
        torch.use_deterministic_algorithms(
            self.deterministic,
            warn_only=self.warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = self.fill_uninitialized_memory

    def __exit__(self, exception_type, exception_value, traceback):
        # 恢复到之前保存的确定性算法和警告模式设置
        torch.use_deterministic_algorithms(
            self.deterministic_restore,
            warn_only=self.warn_only_restore)
        # 恢复到之前保存的填充未初始化内存的设置
        torch.utils.deterministic.fill_uninitialized_memory = self.fill_uninitialized_memory_restore

# 上下文管理器，用于始终警告类型存储的移除
class AlwaysWarnTypedStorageRemoval:
    def __init__(self, always_warn):
        assert isinstance(always_warn, bool)
        self.always_warn = always_warn  # 始终警告类型存储移除的设置

    def __enter__(self):
        # 保存当前始终警告类型存储移除的设置
        self.always_warn_restore = torch.storage._get_always_warn_typed_storage_removal()
        # 设置新的始终警告类型存储移除的配置
        torch.storage._set_always_warn_typed_storage_removal(self.always_warn)

    def __exit__(self, exception_type, exception_value, traceback):
        # 恢复到之前保存的始终警告类型存储移除的设置
        torch.storage._set_always_warn_typed_storage_removal(self.always_warn_restore)

# 上下文管理器，用于设置 CUDA 同步调试模式并将其重置为原始值
class CudaSyncGuard:
    def __init__(self, sync_debug_mode):
        self.mode = sync_debug_mode  # CUDA 同步调试模式的设置

    def __enter__(self):
        # 保存当前的 CUDA 同步调试模式设置
        self.debug_mode_restore = torch.cuda.get_sync_debug_mode()
        # 设置新的 CUDA 同步调试模式
        torch.cuda.set_sync_debug_mode(self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        # 恢复到之前保存的 CUDA 同步调试模式设置
        torch.cuda.set_sync_debug_mode(self.debug_mode_restore)

# 上下文管理器，用于设置 torch.__future__.set_swap_module_params_on_conversion
# 并自动将其重置为原始值
class SwapTensorsGuard:
    def __init__(self, use_swap_tensors):
        self.use_swap_tensors = use_swap_tensors  # 设置是否使用交换张量参数

    def __enter__(self):
        # 保存当前的交换张量参数设置
        self.swap_tensors_restore = torch.__future__.get_swap_module_params_on_conversion()
        # 如果设置了使用交换张量参数，则应用新的设置
        if self.use_swap_tensors is not None:
            torch.__future__.set_swap_module_params_on_conversion(self.use_swap_tensors)

    def __exit__(self, exception_type, exception_value, traceback):
        # 恢复到之前保存的交换张量参数设置
        torch.__future__.set_swap_module_params_on_conversion(self.swap_tensors_restore)
# 定义一个装饰器函数，用于包装具有确定性算法的测试函数
def wrapDeterministicFlagAPITest(fn):
    # 定义装饰器的包装函数，接受任意参数
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 使用 DeterministicGuard 上下文管理器，保护和恢复确定性算法的状态
        with DeterministicGuard(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled()):
            # 定义 CuBLASConfigGuard 类，用于管理 CuBLAS 工作空间配置
            class CuBLASConfigGuard:
                # CuBLAS 变量名
                cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'

                # 进入上下文时执行的方法
                def __enter__(self):
                    # 检查 CUDA 是否为 10.2 或更高版本
                    self.is_cuda10_2_or_higher = (
                        (torch.version.cuda is not None)
                        and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2]))
                    # 如果是 CUDA 10.2 或更高版本，保存当前的 CuBLAS 配置，并设置新的配置
                    if self.is_cuda10_2_or_higher:
                        self.cublas_config_restore = os.environ.get(self.cublas_var_name)
                        os.environ[self.cublas_var_name] = ':4096:8'

                # 离开上下文时执行的方法
                def __exit__(self, exception_type, exception_value, traceback):
                    # 如果是 CUDA 10.2 或更高版本，恢复之前的 CuBLAS 配置
                    if self.is_cuda10_2_or_higher:
                        cur_cublas_config = os.environ.get(self.cublas_var_name)
                        if self.cublas_config_restore is None:
                            if cur_cublas_config is not None:
                                del os.environ[self.cublas_var_name]
                        else:
                            os.environ[self.cublas_var_name] = self.cublas_config_restore

            # 使用 CuBLASConfigGuard 上下文管理器
            with CuBLASConfigGuard():
                # 调用被装饰的函数，传入参数和关键字参数
                fn(*args, **kwargs)
    # 返回包装后的函数
    return wrapper

# 该装饰器用于 API 测试，安全地调用 torch.__future__.set_swap_module_params_on_conversion
# `swap` 可以设置为 True、False 或 None，其中 None 表示上下文管理器不设置标志。
# 测试完成后，会恢复之前的交换标志设置。
def wrapSwapTensorsTest(swap=None):
    ```python`
        # 定义一个装饰器函数 dec_fn，接受一个函数 fn 作为参数
        def dec_fn(fn):
            # 定义一个内部函数 wrapper，接收可变参数 *args 和关键字参数 **kwargs
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # 使用 SwapTensorsGuard 上下文管理器，传入 swap 对象，确保在代码块执行期间进行张量的交换
                with SwapTensorsGuard(swap):
                    # 调用原始函数 fn，并传递所有参数和关键字参数
                    fn(*args, **kwargs)
            # 返回 wrapper 函数
            return wrapper
        # 返回 dec_fn 函数，完成装饰器定义
        return dec_fn
# 定义一个参数化测试类，用于执行参数化测试
class swap(_TestParametrizer):
    def __init__(self, swap_values):
        super().__init__()
        self.swap_values = swap_values

    # 重写父类方法，对测试进行参数化
    def _parametrize_test(self, test, generic_cls, device_cls):
        # 遍历所有的交换值进行测试参数化
        for swap in self.swap_values:
            # 包装测试用例并应用交换值
            yield wrapSwapTensorsTest(swap)(test), f'swap_{swap}', {}, lambda _: []

# 装饰器函数，用于在没有安装 numpy 的情况下跳过测试
def skipIfCompiledWithoutNumpy(fn):
    # 检查是否支持 numpy 模块
    numpy_support = TEST_NUMPY
    if numpy_support:
        try:
            # 验证 PyTorch 是否使用了 numpy 支持
            torch.from_numpy(np.array([2, 2]))
        except RuntimeError:
            numpy_support = False

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果不支持 numpy，则跳过测试
        if not numpy_support:
            raise unittest.SkipTest("PyTorch was compiled without numpy support")
        else:
            fn(*args, **kwargs)
    return wrapper

# 辅助函数，将函数转换为测试运行函数
def _test_function(fn, device):
    def run_test_function(self):
        return fn(self, device)
    return run_test_function

# 装饰器函数，用于在没有启用 XNNPACK 的情况下跳过测试
def skipIfNoXNNPACK(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果未启用 XNNPACK，则跳过测试
        if not torch.backends.xnnpack.enabled:
            raise unittest.SkipTest('XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.')
        else:
            fn(*args, **kwargs)
    return wrapper

# 装饰器函数，用于在没有编译 Lapack 的情况下跳过测试
def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果没有编译 Lapack，则跳过测试
        if not torch._C.has_lapack:
            raise unittest.SkipTest('PyTorch compiled without Lapack')
        else:
            fn(*args, **kwargs)
    return wrapper

# 装饰器函数，用于检查操作是否在 core._REGISTERED_OPERATORS 中注册，如果没有则跳过测试
def skipIfNotRegistered(op_name, message):
    """Wraps the decorator to hide the import of the `core`.

    Args:
        op_name: Check if this op is registered in `core._REGISTERED_OPERATORS`.
        message: message to fail with.

    Usage:
        @skipIfNotRegistered('MyOp', 'MyOp is not linked!')
            This will check if 'MyOp' is in the caffe2.python.core
    """
    return unittest.skip("Pytorch is compiled without Caffe2")

# 装饰器函数，用于在没有安装 SciPy 的情况下跳过测试
def skipIfNoSciPy(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果没有安装 SciPy，则跳过测试
        if not TEST_SCIPY:
            raise unittest.SkipTest("test require SciPy, but SciPy not found")
        else:
            fn(*args, **kwargs)
    return wrapper

# 装饰器函数，用于在使用 pytest 运行测试时跳过测试
def skip_if_pytest(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # 如果在 pytest 环境下运行，则跳过测试
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise unittest.SkipTest("does not work under pytest")
        return fn(*args, **kwargs)

    return wrapped

# 装饰器函数，用于标记慢速测试，仅当 TEST_WITH_SLOW 为 True 时才运行
def slowTest(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果不允许慢速测试，则跳过测试
        if not TEST_WITH_SLOW:  # noqa: F821
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")
        else:
            fn(*args, **kwargs)
    wrapper.__dict__['slow_test'] = True
    return wrapper

# 装饰器函数，用于根据条件决定是否标记慢速测试
def slowTestIf(condition):
    # 如果条件成立，则返回 slowTest；否则返回一个匿名函数，该函数接受一个参数 fn 并返回 fn 本身。
    return slowTest if condition else lambda fn: fn
# 如果条件满足，则跳过 CUDA 内存泄漏检查的装饰器
def skipCUDAMemoryLeakCheckIf(condition):
    def dec(fn):
        # 检查函数是否已经设置了 '_do_cuda_memory_leak_check' 属性，默认为 True
        if getattr(fn, '_do_cuda_memory_leak_check', True):
            # 设置 '_do_cuda_memory_leak_check' 属性，如果条件满足，则设置为 False
            fn._do_cuda_memory_leak_check = not condition
        return fn
    return dec

# 如果条件满足，则跳过非默认 CUDA 流的装饰器
def skipCUDANonDefaultStreamIf(condition):
    def dec(fn):
        # 检查函数是否已经设置了 '_do_cuda_non_default_stream' 属性，默认为 True
        if getattr(fn, '_do_cuda_non_default_stream', True):
            # 设置 '_do_cuda_non_default_stream' 属性，如果条件满足，则设置为 False
            fn._do_cuda_non_default_stream = not condition
        return fn
    return dec

# 忽略警告信息的装饰器
def suppress_warnings(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 使用 'warnings' 模块捕获并忽略所有警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 调用被装饰的函数
            fn(*args, **kwargs)
    return wrapper

# 将对象转移到 GPU 上的函数
def to_gpu(obj, type_map=None):
    # 如果 'type_map' 未指定，则初始化为空字典
    if type_map is None:
        type_map = {}
    
    # 如果对象是 torch.Tensor 类型
    if isinstance(obj, torch.Tensor):
        # 断言张量是叶子节点
        assert obj.is_leaf
        # 获取张量对应的数据类型
        t = type_map.get(obj.dtype, obj.dtype)
        # 在 CUDA 设备上创建张量的副本
        with torch.no_grad():
            res = obj.clone().to(dtype=t, device="cuda")
            res.requires_grad = obj.requires_grad
        return res
    
    # 如果对象是 torch.Storage 类型
    elif torch.is_storage(obj):
        # 创建与输入对象相同大小的新存储，并复制数据到新存储中
        return obj.new().resize_(obj.size()).copy_(obj)
    
    # 如果对象是列表类型
    elif isinstance(obj, list):
        # 递归地将列表中的每个元素转移到 GPU 上
        return [to_gpu(o, type_map) for o in obj]
    
    # 如果对象是元组类型
    elif isinstance(obj, tuple):
        # 递归地将元组中的每个元素转移到 GPU 上
        return tuple(to_gpu(o, type_map) for o in obj)
    
    else:
        # 对象类型未被支持时，进行深拷贝
        return deepcopy(obj)

# 获取函数的参数列表
def get_function_arglist(func):
    return inspect.getfullargspec(func).args

# 设置随机数生成器的种子
def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if TEST_NUMPY:  # 假设此处存在全局变量 TEST_NUMPY
        np.random.seed(seed)

# 上下文管理器：设置默认的张量数据类型
@contextlib.contextmanager
def set_default_dtype(dtype):
    # 保存当前的默认数据类型
    saved_dtype = torch.get_default_dtype()
    # 设置新的默认数据类型
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        # 恢复原来的默认数据类型
        torch.set_default_dtype(saved_dtype)

# 上下文管理器：设置默认的张量类型
@contextlib.contextmanager
def set_default_tensor_type(tensor_type):
    # 保存当前的默认张量类型
    saved_tensor_type = torch.tensor([]).type()
    # 设置新的默认张量类型
    torch.set_default_tensor_type(tensor_type)
    try:
        yield
    finally:
        # 恢复原来的默认张量类型
        torch.set_default_tensor_type(saved_tensor_type)

# 迭代张量的索引值
def iter_indices(tensor):
    # 如果张量维度为 0，返回空范围
    if tensor.dim() == 0:
        return range(0)
    # 如果张量维度为 1，返回张量的索引范围
    if tensor.dim() == 1:
        return range(tensor.size(0))
    # 对多维张量，返回每个维度的索引范围的笛卡尔积
    return product(*(range(s) for s in tensor.size()))

# 检查对象是否可迭代
def is_iterable(obj):
    try:
        # 尝试迭代对象，如果成功则返回 True
        iter(obj)
        return True
    except TypeError:
        # 如果对象不可迭代，则返回 False
        return False

# 检查对象是否为张量的可迭代对象
def is_iterable_of_tensors(iterable, include_empty=False):
    """ Returns True if iterable is an iterable of tensors and False o.w.

        If the iterable is empty, the return value is :attr:`include_empty`
    """
    # 首先检查 iterable 是否为单个张量，张量本身也是可迭代的
    if isinstance(iterable, torch.Tensor):
        return False

    try:
        # 如果 iterable 为空，根据 include_empty 返回值决定结果
        if len(iterable) == 0:
            return include_empty

        # 遍历 iterable 中的每个元素，检查是否为张量
        for t in iter(iterable):
            if not isinstance(t, torch.Tensor):
                return False

    except TypeError as te:
        # 如果遍历失败，说明 iterable 不是可迭代的
        return False

    # 如果所有元素都是张量，则返回 True
    return True

# 用于管理非默认 CUDA 流的类
class CudaNonDefaultStream:
    # 进入上下文管理器时调用，用于设置 CUDA 测试前的流配置
    def __enter__(self):
        # 保存当前所有 CUDA 设备上的活动流，并为所有 CUDA 设备设置新的非默认流，
        # 以确保 CUDA 测试不会误用默认流。
        beforeDevice = torch.cuda.current_device()
        self.beforeStreams = []
        for d in range(torch.cuda.device_count()):
            # 记录当前设备的活动流
            self.beforeStreams.append(torch.cuda.current_stream(d))
            # 创建新的 CUDA 流并同步
            deviceStream = torch.cuda.Stream(device=d)
            self.beforeStreams[-1].synchronize()
            # 设置新创建的流为当前设备的活动流
            torch._C._cuda_setStream(stream_id=deviceStream.stream_id,
                                     device_index=deviceStream.device_index,
                                     device_type=deviceStream.device_type)
        # 恢复之前的 CUDA 设备
        torch._C._cuda_setDevice(beforeDevice)

    # 退出上下文管理器时调用，用于恢复 CUDA 测试前的流配置
    def __exit__(self, exec_type, exec_value, traceback):
        # 在完成 CUDA 测试后，加载之前保存的各 CUDA 设备上的活动流。
        beforeDevice = torch.cuda.current_device()
        for d in range(torch.cuda.device_count()):
            # 恢复之前保存的流配置
            torch._C._cuda_setStream(stream_id=self.beforeStreams[d].stream_id,
                                     device_index=self.beforeStreams[d].device_index,
                                     device_type=self.beforeStreams[d].device_type)
        # 恢复之前的 CUDA 设备
        torch._C._cuda_setDevice(beforeDevice)
class CudaMemoryLeakCheck:
    # 初始化 CUDA 内存泄漏检查器，接受一个测试用例和可选的名称作为参数
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase
        
        # 初始化上下文和随机数生成器，以防止在测试中首次初始化时产生误报
        from torch.testing._internal.common_cuda import initialize_cuda_context_rng
        initialize_cuda_context_rng()

    # 进入上下文管理器，用于存储由 PyTorch 缓存分配器和 CUDA 驱动提供的 CUDA 内存数据
    #
    # 注意：未记录的 torch.cuda.mem_get_info() 返回 GPU 上的空闲字节数和总可用字节数
    def __enter__(self):
        self.caching_allocator_befores = []
        self.driver_befores = []

        # 如果需要，执行垃圾回收（如果持有任何 CUDA 内存）
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
            # 注意：垃圾回收仅基于缓存分配器的内存，因为驱动程序总会有一些字节在使用中（上下文大小？）
            if caching_allocator_mem_allocated > 0:
                gc.collect()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                break

        # 在运行测试之前获取缓存分配器和驱动程序的统计信息
        for i in range(num_devices):
            self.caching_allocator_befores.append(torch.cuda.memory_allocated(i))
            bytes_free, bytes_total = torch.cuda.mem_get_info(i)
            driver_mem_allocated = bytes_total - bytes_free
            self.driver_befores.append(driver_mem_allocated)

@contextmanager
# 用于跳过特定类型异常的上下文管理器
def skip_exception_type(exc_type):
    try:
        yield
    except exc_type as e:
        raise unittest.SkipTest(f"not implemented: {e}") from e

@contextmanager
# 在失败时打印失败重现信息的上下文管理器
def print_repro_on_failure(repro_str):
    try:
        yield
    except unittest.SkipTest:
        raise
    except Exception as e:
        # 注意：修改异常参数是添加失败重现信息的最干净方式，而不会污染堆栈跟踪。
        if len(e.args) >= 1:
            e.args = (f"{e.args[0]}\n{repro_str}", *e.args[1:])
        raise

# "min_satisfying_examples" 设置已在 hypothesis 3.56.0 中弃用，并在 hypothesis 4.x 中移除
try:
    import hypothesis

    def settings(*args, **kwargs):
        if 'min_satisfying_examples' in kwargs and hypothesis.version.__version_info__ >= (3, 56, 0):
            kwargs.pop('min_satisfying_examples')
        return hypothesis.settings(*args, **kwargs)

    # 注册名为 "pytorch_ci" 的 hypothesis 设置配置文件
    hypothesis.settings.register_profile(
        "pytorch_ci",
        settings(
            derandomize=True,
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=50,
            verbosity=hypothesis.Verbosity.normal))
    # 注册名为 "dev" 的Hypothesis配置文件，设置健康检查忽略列表，数据库为None，最大示例数为10，日志详细程度为normal
    hypothesis.settings.register_profile(
        "dev",
        settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=10,
            verbosity=hypothesis.Verbosity.normal))
    
    # 注册名为 "debug" 的Hypothesis配置文件，设置健康检查忽略列表，数据库为None，最大示例数为1000，日志详细程度为verbose
    hypothesis.settings.register_profile(
        "debug",
        settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=1000,
            verbosity=hypothesis.Verbosity.verbose))
    
    # 加载名为 "pytorch_ci" 的Hypothesis配置文件（如果在CI环境下），否则根据环境变量'PYTORCH_HYPOTHESIS_PROFILE'加载，忽略F821错误
    hypothesis.settings.load_profile(
        "pytorch_ci" if IS_CI else os.getenv('PYTORCH_HYPOTHESIS_PROFILE', 'dev')  # noqa: F821
    )
# 如果导入失败，则打印提示信息
except ImportError:
    print('Fail to import hypothesis in common_utils, tests are not derandomized')

# 用于检查是否应禁用测试方法，通过检查@dtypes参数化时附加的设备和数据类型后缀，清理测试方法名称
# 例如，标题为"DISABLED test_bitwise_ops (__main__.TestBinaryUfuncs)"的问题应禁用所有参数化的test_bitwise_ops测试，
# 如test_bitwise_ops_cuda_int32
def remove_device_and_dtype_suffixes(test_name: str) -> str:
    # 由于可能存在循环依赖问题，此处局部导入以避免问题
    from torch.testing._internal.common_device_type import get_device_type_test_bases
    # 获取设备类型的后缀列表
    device_suffixes = [x.device_type for x in get_device_type_test_bases()]
    # 获取数据类型的后缀列表，去除"torch."前缀
    dtype_suffixes = [str(dt)[len("torch."):] for dt in get_all_dtypes()]

    # 将测试方法名称按下划线分割成块
    test_name_chunks = test_name.split("_")
    if len(test_name_chunks) > 0 and test_name_chunks[-1] in dtype_suffixes:
        if len(test_name_chunks) > 1 and test_name_chunks[-2] in device_suffixes:
            return "_".join(test_name_chunks[0:-2])  # 去除设备和数据类型后缀后重新连接
        return "_".join(test_name_chunks[0:-1])  # 只去除数据类型后缀后重新连接
    return test_name  # 如果没有匹配的后缀，则返回原始的测试方法名称

# 检查是否应启用测试方法
def check_if_enable(test: unittest.TestCase):
    # 获取测试类名
    classname = str(test.__class__).split("'")[1].split(".")[-1]
    # 清理测试方法名称，去除设备和数据类型后缀
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    # 内部函数，检查目标是否与测试方法匹配
    def matches_test(target: str):
        # 将目标测试名称按空格分割成部分
        target_test_parts = target.split()
        if len(target_test_parts) < 2:
            # 目标测试名称格式不正确
            return False
        # 目标测试方法名称和类名
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split(".")[-1]
        # 如果测试方法名称或其清理后的版本与禁用的测试方法名称完全匹配，并且允许非参数化的套件名称禁用参数化的套件
        return classname.startswith(target_classname) and (target_testname in (test._testMethodName, sanitized_testname))

    # 如果禁用的测试方法名称在慢速测试字典的键中，则标记测试方法为慢速测试
    if any(matches_test(x) for x in slow_tests_dict.keys()):
        getattr(test, test._testMethodName).__dict__['slow_test'] = True
        if not TEST_WITH_SLOW:  # noqa: F821
            # 如果不允许使用慢速测试，则抛出unittest.SkipTest异常
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")
    # 如果不在沙盒环境中（通过 IS_SANDCASTLE 标志位判断），则执行以下逻辑
    if not IS_SANDCASTLE:  # noqa: F821
        # 初始化变量，标志是否应该跳过测试和跳过消息
        should_skip = False
        skip_msg = ""

        # 遍历禁用测试字典中的每个测试及其关联的问题 URL 和平台列表
        for disabled_test, (issue_url, platforms) in disabled_tests_dict.items():
            # 如果当前禁用的测试与正在执行的测试匹配
            if matches_test(disabled_test):
                # 定义平台到条件的映射关系
                platform_to_conditional: Dict = {
                    "mac": IS_MACOS,
                    "macos": IS_MACOS,
                    "win": IS_WINDOWS,
                    "windows": IS_WINDOWS,
                    "linux": IS_LINUX,
                    "rocm": TEST_WITH_ROCM,  # noqa: F821
                    "xpu": TEST_XPU,  # noqa: F821
                    "asan": TEST_WITH_ASAN,  # noqa: F821
                    "dynamo": TEST_WITH_TORCHDYNAMO,  # noqa: F821
                    "inductor": TEST_WITH_TORCHINDUCTOR,  # noqa: F821
                    "slow": TEST_WITH_SLOW,  # noqa: F821
                }

                # 找出在映射中未定义的平台
                invalid_platforms = list(filter(lambda p: p not in platform_to_conditional, platforms))
                # 如果有未定义的平台，则输出警告信息并修正平台列表
                if len(invalid_platforms) > 0:
                    invalid_plats_str = ", ".join(invalid_platforms)
                    valid_plats = ", ".join(platform_to_conditional.keys())

                    print(f"Test {disabled_test} is disabled for some unrecognized ",
                          f"platforms: [{invalid_plats_str}]. Please edit issue {issue_url} to fix the platforms ",
                          'assigned to this flaky test, changing "Platforms: ..." to a comma separated ',
                          f"subset of the following (or leave it blank to match all platforms): {valid_plats}")

                    # 过滤出映射中定义的平台，以确保继续禁用测试
                    platforms = list(filter(lambda p: p in platform_to_conditional, platforms))

                # 如果平台列表为空，或者其中任一平台满足条件
                if platforms == [] or any(platform_to_conditional[platform] for platform in platforms):
                    # 设置应该跳过测试的标志，并设置跳过消息
                    should_skip = True
                    skip_msg = f"Test is disabled because an issue exists disabling it: {issue_url}" \
                        f" for {'all' if platforms == [] else ''}platform(s) {', '.join(platforms)}. " \
                        "If you're seeing this on your local machine and would like to enable this test, " \
                        "please make sure CI is not set and you are not using the flag --import-disabled-tests."
                    break

        # 如果应该跳过测试，并且不是在重新运行禁用测试的模式下
        if should_skip and not RERUN_DISABLED_TESTS:
            # 抛出 SkipTest 异常，跳过当前测试
            raise unittest.SkipTest(skip_msg)

        # 如果不应该跳过测试，并且是在重新运行禁用测试的模式下
        if not should_skip and RERUN_DISABLED_TESTS:
            # 设置跳过消息，说明仅运行禁用的测试
            skip_msg = "Test is enabled but --rerun-disabled-tests verification mode is set, so only" \
                " disabled tests are run"
            # 抛出 SkipTest 异常，跳过当前测试
            raise unittest.SkipTest(skip_msg)
    # 如果 TEST_SKIP_FAST 变量为真（由 noqa: F821 忽略未定义变量错误），则执行以下代码块
    if TEST_SKIP_FAST:
        # 检查 test 对象是否具有当前测试方法的属性，并且该方法未标记为 slow_test
        if hasattr(test, test._testMethodName) and not getattr(test, test._testMethodName).__dict__.get('slow_test', False):
            # 如果以上条件成立，则抛出 unittest.SkipTest 异常，指示跳过测试
            raise unittest.SkipTest("test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")
# 定义一个新的类 RelaxedBooleanPair，继承自 BooleanPair 类
class RelaxedBooleanPair(BooleanPair):
    """Pair for boolean-like inputs.

    In contrast to the builtin :class:`BooleanPair`, this class also supports one input being a number or a single
    element tensor-like.
    """

    # 获取 NumberPair 类的支持的数据类型列表，并添加到当前类的支持数据类型中
    _supported_number_types = NumberPair(0, 0)._supported_types

    # 重写 _process_inputs 方法，处理输入参数，使得一个可以是布尔值，另一个可以是布尔值、数字、单个元素张量或数组
    def _process_inputs(self, actual, expected, *, id):
        # 定义张量或数组的类型元组
        tensor_or_array_types: Tuple[Type, ...] = (torch.Tensor, np.ndarray)
        # 将当前类支持的数据类型与数字对的支持数据类型合并
        other_supported_types = (*self._supported_types, *self._supported_number_types, *tensor_or_array_types)
        
        # 检查输入参数是否满足条件，其中一个必须是当前类支持的类型，另一个可以是其他支持类型
        if not (
            (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types))
            or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))
        ):
            # 如果不满足条件，则调用 _inputs_not_supported 方法
            self._inputs_not_supported()

        # 将 actual 和 expected 转换为布尔值，并返回转换后的列表
        return [self._to_bool(input, id=id) for input in (actual, expected)]

    # 将输入参数 bool_like 转换为布尔值
    def _to_bool(self, bool_like, *, id):
        # 如果 bool_like 是 numpy 的数值类型，将其转换为布尔值
        if isinstance(bool_like, np.number):
            return bool(bool_like.item())
        # 如果 bool_like 是当前类支持的数字类型，直接转换为布尔值
        elif type(bool_like) in self._supported_number_types:
            return bool(bool_like)
        # 如果 bool_like 是 torch.Tensor 或者 np.ndarray 类型
        elif isinstance(bool_like, (torch.Tensor, np.ndarray)):
            # 获取张量或数组的元素个数
            numel = bool_like.numel() if isinstance(bool_like, torch.Tensor) else bool_like.size
            # 如果元素个数大于1，则抛出异常
            if numel > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a boolean. "
                    f"Got {numel} elements instead.",
                    id=id
                )

            # 否则将张量或数组中的单个元素转换为布尔值并返回
            return bool(bool_like.item())
        # 如果 bool_like 是其他类型，则调用父类的 _to_bool 方法进行处理
        else:
            return super()._to_bool(bool_like, id=id)


# 定义一个新的类 RelaxedNumberPair，继承自 NumberPair 类
class RelaxedNumberPair(NumberPair):
    """Pair for number-like inputs.

    In contrast to the builtin :class:`NumberPair`, this class also supports one input being a single element
    tensor-like or a :class:`enum.Enum`. (D)Type checks are disabled, meaning comparing 1 to 1.0 succeeds even when
    ``check_dtype=True`` is passed.
    """
    # 类型映射字典，将 Python 原生类型映射到对应的 Torch 张量类型
    _TYPE_TO_DTYPE = {
        int: torch.int64,
        float: torch.float32,
        complex: torch.complex64,
    }

    def __init__(
            self, actual, expected, *, rtol_override=0.0, atol_override=0.0, check_dtype=None, **other_parameters
    ) -> None:
        # 调用父类的初始化方法，禁用类型检查，允许其他参数传入
        super().__init__(actual, expected, check_dtype=False, **other_parameters)
        # 根据传入的 rtol_override 和 atol_override，更新相对容差和绝对容差
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(self, actual, expected, *, id):
        # 定义支持的张量和数组类型
        tensor_or_array_types: Tuple[Type, ...] = (torch.Tensor, np.ndarray)
        # 将其他支持的类型与默认支持的类型合并
        other_supported_types = (*self._supported_types, *tensor_or_array_types)
        # 检查输入是否满足支持的组合条件，否则调用输入不支持的异常处理方法
        if not (
                (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types))
                or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))
        ):
            self._inputs_not_supported()

        # 将输入转换为数字，确保只有一个元素
        return [self._to_number(input, id=id) for input in (actual, expected)]

    def _to_number(self, number_like, *, id):
        # 如果输入是张量或数组
        if isinstance(number_like, (torch.Tensor, np.ndarray)):
            # 获取张量或数组的元素数
            numel = number_like.numel() if isinstance(number_like, torch.Tensor) else number_like.size
            # 如果元素数大于1，抛出数值错误异常
            if numel > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a number. "
                    f"Got {numel} elements instead.",
                    id=id
                )
            # 获取张量或数组的单个元素值，并将布尔类型转换为整数
            number = number_like.item()
            if isinstance(number, bool):
                number = int(number)

            return number
        # 如果输入是枚举类型，返回其整数值
        elif isinstance(number_like, Enum):
            return int(number_like)  # type: ignore[call-overload]
        else:
            # 调用父类方法，将输入转换为数字
            return super()._to_number(number_like, id=id)
# 定义一个继承自 TensorLikePair 的类，用于处理 tensor-like 输入的对
class TensorOrArrayPair(TensorLikePair):
    """Pair for tensor-like inputs.

    On the one hand this class is stricter than the builtin :class:`TensorLikePair` since it only allows instances of
    :class:`torch.Tensor` and :class:`numpy.ndarray` rather than allowing any tensor-like than can be converted into a
    tensor. On the other hand this class is looser since it converts all inputs into tensors with no regard of their
    relationship, e.g. comparing a :class:`torch.Tensor` to :class:`numpy.ndarray` is fine.

    In addition, this class supports overriding the absolute and relative tolerance through the ``@precisionOverride``
    and ``@toleranceOverride`` decorators.
    """

    # 初始化方法，接受 actual 和 expected 作为参数，还有一些可选参数
    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        # 调用父类的初始化方法，传递 actual 和 expected 以及其他参数
        super().__init__(actual, expected, **other_parameters)
        # 使用传入的 rtol_override 和 atol_override 来覆盖默认的相对和绝对容差
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    # 处理输入的私有方法，验证输入是否为 torch.Tensor 或 np.ndarray 的实例
    def _process_inputs(self, actual, expected, *, id, allow_subclasses):
        # 调用 _check_inputs_isinstance 方法，确保 actual 和 expected 是 torch.Tensor 或 np.ndarray 的实例
        self._check_inputs_isinstance(actual, expected, cls=(torch.Tensor, np.ndarray))
        
        # 将 actual 和 expected 转换为 tensor，确保它们可以被处理
        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        
        # 对转换后的 tensor 进行支持性检查，通过 id 参数标识检查对象
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        
        # 返回处理后的 actual 和 expected
        return actual, expected


# 定义一个继承自 TensorLikePair 的类，用于处理 torch.storage.TypedStorage 输入的对
class TypedStoragePair(TensorLikePair):
    """Pair for :class:`torch.storage.TypedStorage` inputs."""

    # 初始化方法，接受 actual 和 expected 作为参数，还有一些可选参数
    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        # 使用 _check_inputs_isinstance 方法，确保 actual 和 expected 是 torch.storage.TypedStorage 的实例
        self._check_inputs_isinstance(actual, expected, cls=torch.storage.TypedStorage)
        # 调用父类的初始化方法，传递 actual 和 expected 以及其他参数
        super().__init__(actual, expected, **other_parameters)
        # 使用传入的 rtol_override 和 atol_override 来覆盖默认的相对和绝对容差
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    # 将 TypedStorage 转换为 tensor 的方法
    def _to_tensor(self, typed_storage):
        return torch.tensor(
            typed_storage._untyped_storage,  # 使用 TypedStorage 的未类型化存储来创建 tensor
            dtype={  # 根据 TypedStorage 的 dtype 来指定 tensor 的数据类型
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8
            }.get(typed_storage.dtype, typed_storage.dtype),  # 使用字典来进行 dtype 的映射
            device=typed_storage.device,  # 使用 TypedStorage 的设备信息来指定 tensor 的设备
        )


# 定义一个继承自 Pair 的类，用于处理非数值输入的对
class UnittestPair(Pair):
    """Fallback ABC pair that handles non-numeric inputs.

    To avoid recreating the mismatch messages of :meth:`unittest.TestCase.assertEqual`, this pair simply wraps it in
    order to use it with the :class:`Pair` "framework" from :func:`are_equal`.

    Define the :attr:`UnittestPair.CLS` in a subclass to indicate which class(es) of the inputs the pair should support.
    """

    # 类型变量，指定该类支持的输入类别
    CLS: Union[Type, Tuple[Type, ...]]
    # 类型名称，可选的类属性，用于标识该类处理的输入类型
    TYPE_NAME: Optional[str] = None

    # 初始化方法，接受 actual 和 expected 作为参数，还有一些可选参数
    def __init__(self, actual, expected, **other_parameters):
        # 使用 _check_inputs_isinstance 方法，确保 actual 和 expected 是指定的类实例
        self._check_inputs_isinstance(actual, expected, cls=self.CLS)
        # 调用父类的初始化方法，传递 actual 和 expected 以及其他参数
        super().__init__(actual, expected, **other_parameters)
    # 定义一个名为 compare 的方法，属于当前类的实例方法
    def compare(self):
        # 创建一个单元测试的 TestCase 对象
        test_case = unittest.TestCase()

        try:
            # 调用 TestCase 对象的 assertEqual 方法，比较实际值和期望值
            return test_case.assertEqual(self.actual, self.expected)
        except test_case.failureException as error:
            # 如果比较抛出异常，捕获错误消息
            msg = str(error)

        # 确定要显示的类型名称，如果未提供则根据 CLS 的类型或类名获取
        type_name = self.TYPE_NAME or (self.CLS if isinstance(self.CLS, type) else self.CLS[0]).__name__
        # 调用类内部的 _fail 方法，抛出 AssertionError，并包含比较失败的详细消息
        self._fail(AssertionError, f"{type_name.title()} comparison failed: {msg}")
# 定义一个自定义的 UnittestPair 类型，用于测试字符串类型的断言
class StringPair(UnittestPair):
    CLS = (str, bytes)  # 类型定义为 str 或 bytes
    TYPE_NAME = "string"  # 类型名称为 "string"


# 定义一个自定义的 UnittestPair 类型，用于测试集合类型的断言
class SetPair(UnittestPair):
    CLS = set  # 类型定义为 set


# 定义一个自定义的 UnittestPair 类型，用于测试类型对象的断言
class TypePair(UnittestPair):
    CLS = type  # 类型定义为 type


# 定义一个自定义的 UnittestPair 类型，用于测试对象类型的断言
class ObjectPair(UnittestPair):
    CLS = object  # 类型定义为 object


# 实现一个变体的 assertRaises/assertRaisesRegex 函数，用于捕获 NotImplementedError 异常，
# 如果捕获到该异常，则跳过测试而不是将其标记为失败
# 这通过继承 unittest.case 中的 _AssertRaisesContext 类来实现，稍作修改以实现新的行为。
# 注意：2021 年，此处使用了 unittest.case 中的私有实现，自 2010 年以来未发生变更，因此风险较低。
class AssertRaisesContextIgnoreNotImplementedError(unittest.case._AssertRaisesContext):
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None and issubclass(exc_type, NotImplementedError):
            self.test_case.skipTest(f"not_implemented: {exc_value}")  # 如果是 NotImplementedError 则跳过测试
        return super().__exit__(exc_type, exc_value, tb)


# 实现一个上下文管理器函数，用于设置是否始终发出警告的上下文
# 参数 new_val: 设置新的警告状态
@contextmanager
def set_warn_always_context(new_val: bool):
    old_val = torch.is_warn_always_enabled()  # 保存当前的警告状态
    torch.set_warn_always(new_val)  # 设置新的警告状态
    try:
        yield  # 执行被管理的代码块
    finally:
        torch.set_warn_always(old_val)  # 恢复原始的警告状态


# 定义一个类，用于使 pytest 不将其识别为测试类
class NoTest:
    __test__ = False  # 阻止 pytest 将此类识别为测试类


# 定义一个 TestCase 类，继承自 expecttest.TestCase
class TestCase(expecttest.TestCase):
    # 注意: "precision" 用于让类和生成的测试在比较张量时设置最小的 atol 值。
    # 例如，被 @precisionOverride 和 @toleranceOverride 使用。
    _precision: float = 0  # 设置精度初始值为 0

    # 注意: "rel_tol" 用于让类和生成的测试在比较张量时设置最小的 rtol 值。
    # 例如，被 @toleranceOverride 使用。
    _rel_tol: float = 0  # 设置相对容差初始值为 0

    # 控制是否断言 `torch.get_default_dtype()` 返回 `torch.float` 当调用 `setUp` 和 `tearDown` 时。
    _default_dtype_check_enabled: bool = False  # 是否启用默认数据类型检查的标志

    # 始终使用 difflib 在多行相等性时打印差异。
    # unittest 中的未记录特性
    _diffThreshold = sys.maxsize
    maxDiff = None  # 不限制最大差异显示

    # 检查是否应该停止整个测试套件，如果发生无法恢复的失败。
    def _should_stop_test_suite(self):
        if torch.cuda.is_initialized():
            # 如果 CUDA 初始化了，则进行 CUDA 设备端的错误检查
            # 在 torch.cuda.synchronize() 过程中捕获 RuntimeError 将导致后续测试用例失败。
            try:
                torch.cuda.synchronize()
            except RuntimeError as rte:
                print("TEST SUITE EARLY TERMINATION due to torch.cuda.synchronize() failure", file=sys.stderr)
                print(str(rte), file=sys.stderr)
                return True  # 停止整个测试套件
            return False  # 不停止测试套件
        else:
            return False  # 不停止测试套件

    @property
    def precision(self) -> float:
        return self._precision  # 返回当前的精度设置

    @precision.setter
    def precision(self, prec: float) -> None:
        self._precision = prec  # 设置新的精度值
    # 定义一个属性方法，返回对象的相对容差
    def rel_tol(self) -> float:
        return self._rel_tol

    # 定义一个属性的设置方法，设置对象的相对容差
    @rel_tol.setter
    def rel_tol(self, prec: float) -> None:
        self._rel_tol = prec

    # 定义一个类级别的变量，用于标识是否进行 CUDA 内存泄漏检查
    _do_cuda_memory_leak_check = False

    # 定义一个类级别的变量，用于标识是否使用非默认的 CUDA 流
    _do_cuda_non_default_stream = False

    # 当设置为 True 时，如果测试用例引发 NotImplementedError 异常，
    # 而不是使测试失败，跳过该测试用例。
    _ignore_not_implemented_error = False
    def __init__(self, method_name='runTest', methodName='runTest'):
        # methodName 是在 unittest 中正确的命名方式，而 testslide 使用关键字参数。
        # 因此我们需要同时使用这两种方式来：1) 不破坏向后兼容性，2) 支持 testslide。
        if methodName != "runTest":
            method_name = methodName
        # 调用父类的构造方法，传入调整后的方法名
        super().__init__(method_name)

        # 获取当前测试类中的测试方法对象
        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # 如果需要进行 CUDA 内存泄漏检查，则调整被测试方法的包装
            if TEST_CUDA_MEM_LEAK_CHECK:  # noqa: F821
                # 更新 CUDA 内存泄漏检查标志位，如果测试方法中有对应标志位，则使用其值
                self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
                # FIXME: 解决在 Windows 平台上的 -1024 反泄漏问题。参见 issue #8044
                # 如果需要进行 CUDA 内存泄漏检查且不是在 Windows 平台上，则进行方法包装
                if self._do_cuda_memory_leak_check and not IS_WINDOWS:
                    self.wrap_with_cuda_policy(method_name, self.assertLeaksNoCudaTensors)

            # 如果需要强制使用非默认 CUDA 流，则调整被测试方法的包装
            self._do_cuda_non_default_stream &= getattr(test_method, '_do_cuda_non_default_stream', True)
            # 如果需要强制使用非默认 CUDA 流且不是在 Windows 平台上，则进行方法包装
            if self._do_cuda_non_default_stream and not IS_WINDOWS:
                self.wrap_with_cuda_policy(method_name, self.enforceNonDefaultStream)

            # 如果忽略 NotImplementedError 异常，则调整被测试方法的包装
            if self._ignore_not_implemented_error:
                self.wrap_with_policy(method_name, lambda: skip_exception_type(NotImplementedError))

            # 如果在测试失败时需要打印复现信息，则设置相关环境变量前缀
            if PRINT_REPRO_ON_FAILURE:  # noqa: F821
                env_var_prefix = TestEnvironment.repro_env_var_prefix()
                try:
                    def _get_rel_test_path(abs_test_path):
                        # 尝试基于 "test" 目录获取相对路径。
                        # 在 CI 中，工作目录不一定是基础存储库目录，因此不能仅从那里计算相对路径。
                        parts = Path(abs_test_path).parts
                        for i, part in enumerate(parts):
                            if part == "test":
                                base_dir = os.path.join(*parts[:i]) if i > 0 else ''
                                return os.path.relpath(abs_test_path, start=base_dir)

                        # 无法确定包含目录；只返回测试文件名。
                        # 虽然路径不是严格正确的，但这比没有要好。
                        return os.path.split(abs_test_path)[1]

                    # 注意：在 Python 3.8 中，getfile() 调用将返回相对于工作目录的路径，因此将其转换为绝对路径。
                    abs_test_path = os.path.abspath(inspect.getfile(type(self)))
                    test_filename = _get_rel_test_path(abs_test_path)
                    class_name = type(self).__name__
                    repro_str = f"""
# 为了执行这个测试，请从基础存储库目录运行以下命令：
# {env_var_prefix} python {test_filename} -k {class_name}.{method_name}
# 
# 如果设置 PYTORCH_PRINT_REPRO_ON_FAILURE=0，则可以抑制此消息
"""

这段代码在一个测试失败时打印复现字符串。复现字符串可以通过设置 PYTORCH_PRINT_REPRO_ON_FAILURE=0 来抑制。

                    self.wrap_with_policy(
                        method_name,
                        lambda repro_str=repro_str: print_repro_on_failure(repro_str=repro_str))
                except Exception as e:
                    # 如果无法获取测试文件名，不会完全失败
                    log.info("could not print repro string", extra=str(e))

    def assertLeaksNoCudaTensors(self, name=None):
        name = self.id() if name is None else name
        return CudaMemoryLeakCheck(self, name)

    def enforceNonDefaultStream(self):
        return CudaNonDefaultStream()

    def assertExpectedInline(self, actual, expect, skip=0):
        return super().assertExpectedInline(actual if isinstance(actual, str) else str(actual), expect, skip + 1)

    # Munges exceptions that internally contain stack traces, using munge_exc
    def assertExpectedInlineMunged(
        self, exc_type, callable, expect, *, suppress_suffix=True
    ):
        try:
            callable()
        except exc_type as e:
            self.assertExpectedInline(
                munge_exc(e, suppress_suffix=suppress_suffix, skip=1), expect, skip=1
            )
            return
        self.fail(msg="Did not raise when expected to")

    def assertLogs(self, logger=None, level=None):
        if logger is None:
            logger = logging.getLogger("torch")
        return super().assertLogs(logger, level)

    def assertNoLogs(self, logger=None, level=None):
        if logger is None:
            logger = logging.getLogger("torch")
        return super().assertNoLogs(logger, level)

    def wrap_with_cuda_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        # 下面的导入可能会初始化 CUDA 上下文，因此只有在
        # self._do_cuda_memory_leak_check 或 self._do_cuda_non_default_stream
        # 为 True 时才执行。
        # TODO: 确实看起来我们在这里无条件初始化上下文 -- ezyang
        from torch.testing._internal.common_cuda import TEST_CUDA
        fullname = self.id().lower()  # class_name.method_name
        if TEST_CUDA and ('gpu' in fullname or 'cuda' in fullname):
            setattr(self, method_name, self.wrap_method_with_policy(test_method, policy))

    def wrap_with_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        setattr(self, method_name, self.wrap_method_with_policy(test_method, policy))

    # 一个策略是一个零参数函数，返回一个上下文管理器。
    # 我们不直接取上下文管理器，因为可能需要每个测试方法构建一次
    def wrap_method_with_policy(self, method, policy):
        # 定义一个方法，用于将给定的 `method` 方法按照指定的 `policy` 进行包装
        # 注意：Python 异常（例如 unittest.Skip）会使对象保持在作用域内，
        #       所以这不能在 setUp 和 tearDown 中完成，因为 tearDown 无论测试
        #       是否通过都会运行。出于同样的原因，我们不能在 try-finally 中包装
        #       `method` 调用并始终进行检查。
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # 使用给定的 `policy` 上下文执行 `method`
            with policy():
                method(*args, **kwargs)
        return types.MethodType(wrapper, self)

    def wrap_with_cuda_memory_check(self, method):
        # 使用 `wrap_method_with_policy` 方法将给定的 `method` 与 `assertLeaksNoCudaTensors` 包装
        return self.wrap_method_with_policy(method, self.assertLeaksNoCudaTensors)

    def run(self, result=None):
        # 使用上下文管理器 `ExitStack` 创建堆栈 `stack`
        with contextlib.ExitStack() as stack:
            # 如果 TEST_WITH_CROSSREF 为真，则进入交叉引用模式的上下文
            if TEST_WITH_CROSSREF:  # noqa: F821
                stack.enter_context(CrossRefMode())
            # 调用自定义方法 `_run_custom`，并传入 `result` 参数
            self._run_custom(
                result=result,
            )

    def setUp(self):
        # 检查是否启用了某些全局设置
        check_if_enable(self)
        # 设置随机数种子为预定义的 `SEED`
        set_rng_seed(SEED)

        # 保存全局稀疏张量不变性检查的状态，以便在 tearDown 中恢复
        self._check_invariants = torch.sparse.check_sparse_tensor_invariants.is_enabled()

        # 启用所有稀疏张量构造的不变性检查，包括不安全的构造方法。
        # 如果某些测试用例不需要这个检查，可以在稀疏张量构造函数中使用 `check_invariants=False`
        # 的可选参数，或者使用 `@torch.sparse.check_sparse_tensor_invariants(False)`
        # 装饰器来禁用不变性检查。
        torch.sparse.check_sparse_tensor_invariants.enable()

        if self._default_dtype_check_enabled:
            # 断言当前的默认数据类型为 `torch.float`
            assert torch.get_default_dtype() == torch.float

        # 尝试在测试结束时重置一些全局状态
        self._prev_grad_state = torch.is_grad_enabled()

    def tearDown(self):
        # 有些测试用例可能会覆盖 `TestCase.setUp` 的定义，因此不能假定 `_check_invariants`
        # 属性通常会定义。

        if hasattr(self, '_check_invariants'):
            # 恢复全局稀疏张量不变性检查的状态
            if self._check_invariants:
                torch.sparse.check_sparse_tensor_invariants.enable()
            else:
                torch.sparse.check_sparse_tensor_invariants.disable()

        if self._default_dtype_check_enabled:
            # 断言当前的默认数据类型为 `torch.float`
            assert torch.get_default_dtype() == torch.float

        # 如果 `_prev_grad_state` 属性被定义，则设置梯度计算状态
        if hasattr(self, '_prev_grad_state'):
            torch.set_grad_enabled(self._prev_grad_state)

    @staticmethod
    # 定义生成稀疏压缩张量的方法，返回稀疏压缩张量
    def genSparseCompressedTensor(self, size, nnz, *, layout, device, dtype, index_dtype, blocksize=(), dense_dims=0):
        # 导入所需的运算符模块和reduce函数
        from operator import mul
        from functools import reduce
        # 稀疏维度设为2
        sparse_dim = 2
        # 断言确保所有维度尺寸大于0或nnz为0，否则抛出异常
        assert all(size[d] > 0 for d in range(len(size))) or nnz == 0, 'invalid arguments'
        # 断言确保尺寸维度大于等于稀疏维度
        assert len(size) >= sparse_dim
        # 如果存在块尺寸，确保块尺寸长度为2，并且尺寸的倒数第3个维度可以整除blocksize[0]，倒数第2个维度可以整除blocksize[1]
        if blocksize:
            assert len(blocksize) == 2, (size, blocksize)
            assert size[-2 - dense_dims] % blocksize[0] == 0, (size, blocksize)
            assert size[-1 - dense_dims] % blocksize[1] == 0, (size, blocksize)
            blocksize0, blocksize1 = blocksize
        else:
            blocksize0 = blocksize1 = 1

        # 将size转换为元组形式
        size = tuple(size)
        # 获取密集部分的尺寸
        dense_size = size[(len(size) - dense_dims):]

        # 定义随机生成稀疏压缩张量的函数
        def random_sparse_compressed(n_compressed_dims, n_plain_dims, nnz):
            # 生成压缩索引
            compressed_indices = self._make_crow_indices(n_compressed_dims, n_plain_dims, nnz, device=device, dtype=index_dtype)
            # 创建全零的普通索引
            plain_indices = torch.zeros(nnz, dtype=index_dtype, device=device)
            # 对每个压缩维度，根据随机排列的方式生成普通索引
            for i in range(n_compressed_dims):
                count = compressed_indices[i + 1] - compressed_indices[i]
                plain_indices[compressed_indices[i]:compressed_indices[i + 1]], _ = torch.sort(
                    torch.randperm(n_plain_dims, dtype=index_dtype, device=device)[:count])
            # 根据数据类型和设备生成值张量，范围由low和high确定
            low = -1 if dtype != torch.uint8 else 0
            high = 1 if dtype != torch.uint8 else 2
            values = make_tensor((nnz,) + blocksize + dense_size, device=device, dtype=dtype, low=low, high=high)
            return values, compressed_indices, plain_indices

        # 计算批次形状的乘积，作为批次数量
        batch_shape = size[:-2 - dense_dims]
        n_batch = reduce(mul, batch_shape, 1)

        # 根据布局类型确定压缩维度和普通维度
        if layout in {torch.sparse_csr, torch.sparse_bsr}:
            n_compressed_dims, n_plain_dims = size[-2 - dense_dims] // blocksize0, size[-1 - dense_dims] // blocksize1
        else:
            n_compressed_dims, n_plain_dims = size[-1 - dense_dims] // blocksize1, size[-2 - dense_dims] // blocksize0
        # 每个块的非零元素数量
        blocknnz = nnz // (blocksize0 * blocksize1)
        # 随机生成多个稀疏压缩张量
        sparse_tensors = [random_sparse_compressed(n_compressed_dims, n_plain_dims, blocknnz) for _ in range(n_batch)]
        # 将生成的稀疏张量组成迭代器
        sparse_tensors_it = map(list, zip(*sparse_tensors))

        # 将稀疏值张量堆叠为指定形状，并重新整形为批次形状、块非零元素数量、块尺寸和密集部分尺寸
        values = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, blocknnz, *blocksize, *dense_size)
        # 压缩索引张量，重新整形为批次形状和展开的索引
        compressed_indices = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, -1)
        # 普通索引张量，重新整形为批次形状和展开的索引
        plain_indices = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, -1)
        # 返回稀疏压缩张量对象
        return torch.sparse_compressed_tensor(compressed_indices, plain_indices,
                                              values, size=size, dtype=dtype, layout=layout, device=device)
    def genSparseCSRTensor(self, size, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # 使用 genSparseCompressedTensor 方法生成稀疏 CSR 格式的张量
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_csr, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=(), dense_dims=dense_dims)

    def genSparseCSCTensor(self, size, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # 使用 genSparseCompressedTensor 方法生成稀疏 CSC 格式的张量
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_csc, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=(), dense_dims=0)

    def genSparseBSRTensor(self, size, blocksize, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # 断言 blocksize 的长度为 2
        assert len(blocksize) == 2
        # 使用 genSparseCompressedTensor 方法生成稀疏 BSR 格式的张量
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_bsr, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=blocksize, dense_dims=dense_dims)

    def genSparseBSCTensor(self, size, blocksize, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # 断言 blocksize 的长度为 2
        assert len(blocksize) == 2
        # 使用 genSparseCompressedTensor 方法生成稀疏 BSC 格式的张量
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_bsc, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=blocksize, dense_dims=dense_dims)

    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device, dtype):
        # 断言不会出现不可能的组合，即稀疏维度的尺寸大于零，或者 nnz 等于零
        assert all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0, 'invalid arguments'

        # 创建一个大小为 [nnz, size[sparse_dim:]] 的张量 v
        v_size = [nnz] + list(size[sparse_dim:])
        v = make_tensor(v_size, device=device, dtype=dtype, low=-1, high=1)
        
        # 生成随机的稀疏索引 i，乘以对应维度的大小
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        
        # 如果是非压缩的稀疏张量，则分割索引 i
        if is_uncoalesced:
            i1 = i[:, :(nnz // 2), ...]
            i2 = i[:, :((nnz + 1) // 2), ...]
            i = torch.cat([i1, i2], 1)
        
        # 使用稀疏 COO 格式创建张量 x
        x = torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype, device=device)

        # 如果不是非压缩的稀疏张量，则压缩张量 x
        if not is_uncoalesced:
            x = x.coalesce()
        else:
            # FIXME: `x` 是 `v` 的稀疏视图。目前未实现稀疏视图的重定位历史记录，因此需要这个 workaround 来处理在 `x` 上的原地操作，例如 copy_()。
            # NOTE: 我们在 detach() 之后执行 clone()，因为我们需要能够之后修改 x 的尺寸/存储。
            x = x.detach().clone()._coalesced_(False)
        
        # 返回稀疏张量 x，以及它的索引和值的克隆
        return x, x._indices().clone(), x._values().clone()
    # 将稀疏张量转换为密集张量。仅支持 COO 格式的稀疏张量。
    def safeToDense(self, t):
        # 如果输入张量 t 的布局是稀疏 COO 格式，则进行压缩操作，使之变为稠密张量
        if t.layout == torch.sparse_coo:
            t = t.coalesce()
        return t.to_dense()

    # 使用给定的样本输入对象，比较 Torch 函数与参考函数的输出值
    # 注意：此处仅比较值，不进行类型比较
    def compare_with_reference(self, torch_fn, ref_fn, sample_input, **kwargs):
        numpy_sample = sample_input.numpy()
        n_inp, n_args, n_kwargs = numpy_sample.input, numpy_sample.args, numpy_sample.kwargs
        t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs

        # 使用 Torch 函数计算实际输出
        actual = torch_fn(t_inp, *t_args, **t_kwargs)
        # 使用参考函数计算期望输出
        expected = ref_fn(n_inp, *n_args, **n_kwargs)

        # 使用 self.assertEqual 进行实际输出与期望输出的比较，允许设备不完全匹配
        self.assertEqual(actual, expected, exact_device=False, **kwargs)

    # 比较给定的 Torch 和 NumPy 函数在给定的类似张量对象上的表现
    # 注意：torch_fn 和 np_fn 应为接受单个张量（数组）作为参数的函数。
    #   如果 Torch 和/或 NumPy 函数需要额外的参数，则需使用 lambda 包装函数或传递偏函数。
    # TODO: 添加 args/kwargs 以传递给 assertEqual（例如 rtol、atol）
    def compare_with_numpy(self, torch_fn, np_fn, tensor_like,
                           device=None, dtype=None, **kwargs):
        assert TEST_NUMPY

        if isinstance(tensor_like, torch.Tensor):
            assert device is None
            assert dtype is None
            # 将张量分离并移动到 CPU 上
            t_cpu = tensor_like.detach().cpu()
            # 如果张量是 torch.bfloat16 类型，则转换为 float 类型
            if t_cpu.dtype is torch.bfloat16:
                t_cpu = t_cpu.float()
            a = t_cpu.numpy()
            t = tensor_like
        else:
            d = copy.copy(torch_to_numpy_dtype_dict)
            d[torch.bfloat16] = np.float32
            # 使用给定的 dtype 将 tensor_like 转换为 NumPy 数组 a
            a = np.array(tensor_like, dtype=d[dtype])
            # 创建 Torch 张量 t，指定设备和数据类型
            t = torch.tensor(tensor_like, device=device, dtype=dtype)

        # 使用 np_fn 计算 NumPy 结果
        np_result = np_fn(a)
        # 使用 torch_fn 计算 Torch 结果，并将其移动到 CPU
        torch_result = torch_fn(t).cpu()

        # 如果 np_result 是 ndarray，则尝试将其转换为 Torch 张量
        if isinstance(np_result, np.ndarray):
            try:
                np_result = torch.from_numpy(np_result)
            except Exception:
                # 注意：在转换之前必须复制数组，例如，当数组具有负步长时
                np_result = torch.from_numpy(np_result.copy())
            # 如果 t 和 torch_result 的数据类型都是 torch.bfloat16，而 np_result 的数据类型是 torch.float，则将 torch_result 转换为 torch.float
            if t.dtype is torch.bfloat16 and torch_result.dtype is torch.bfloat16 and np_result.dtype is torch.float:
                torch_result = torch_result.to(torch.float)

        # 使用 self.assertEqual 进行 np_result 和 torch_result 的比较
        self.assertEqual(np_result, torch_result, **kwargs)

    def assertEqualIgnoreType(self, *args, **kwargs) -> None:
        # 如果看到此函数被使用，说明测试编写错误，需要进行详细调查
        # 调用 self.assertEqual，不精确比较数据类型
        return self.assertEqual(*args, exact_dtype=False, **kwargs)
    # 断言函数，用于比较张量 x 和 y 是否相等，y 可能需要广播到 x 的形状
    def assertEqualBroadcasting(self, x, y, *args, **kwargs) -> None:
        r"""Tests if tensor x equals to y, if y to be broadcast to x.shape.
        """
        if not isinstance(y, Iterable):
            # 如果 y 不是可迭代对象，则可能是 int、float 等或者形状不同的张量，将 y 广播到与 x 相同的形状
            y = torch.ones_like(x) * y
        if not isinstance(y, torch.Tensor):
            # 如果 y 是可迭代对象但不是张量，则将其转换为张量并广播到与 x 相同的形状
            y = torch.ones_like(x) * torch.tensor(y)
        # 调用 assertEqual 方法比较 x 和 y 是否相等，传入其他参数和关键字参数
        return self.assertEqual(x, y, *args, **kwargs)

    # 断言函数，比较 x 和 y 是否相等
    def assertEqual(
            self,
            x,
            y,
            msg: Optional[Union[str, Callable[[str], str]]] = None,
            *,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            equal_nan=True,
            exact_dtype=True,
            exact_device=False,  # TODO: default this to True
            exact_layout=False,
            exact_stride=False,
            exact_is_coalesced=False
    ):
        # 使用断言检查 x 和 y 是否相等，可以设置消息、绝对误差、相对误差等参数
        pass

    # 断言函数，比较 x 和 y 是否不相等
    def assertNotEqual(self, x, y, msg: Optional[str] = None, *,                                       # type: ignore[override]
                       atol: Optional[float] = None, rtol: Optional[float] = None, **kwargs) -> None:
        # 使用 with 上下文管理器检查 x 和 y 是否相等，如果相等则抛出 AssertionError
        with self.assertRaises(AssertionError, msg=msg):
            self.assertEqual(x, y, msg, atol=atol, rtol=rtol, **kwargs)

    # 断言函数，比较两个对象 x 和 y 的设备、数据类型和稀疏性是否相同
    def assertEqualTypeString(self, x, y) -> None:
        # 这个 API 用于模拟已废弃的 x.type() == y.type() 检查
        self.assertEqual(x.device, y.device)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.is_sparse, y.is_sparse)

    # 断言函数，检查对象 obj 是否在可迭代对象 iterable 中
    def assertObjectIn(self, obj: Any, iterable: Iterable[Any]) -> None:
        for elem in iterable:
            if id(obj) == id(elem):
                return
        # 如果 obj 不在 iterable 中，则抛出 AssertionError
        raise AssertionError("object not found in iterable")

    # 重新实现的 assertRaises 方法，当 _ignore_not_implemented_error 为 True 时提供特殊行为
    def assertRaises(self, expected_exception, *args, **kwargs):
        if self._ignore_not_implemented_error:
            # 如果 _ignore_not_implemented_error 为 True，则使用特定的上下文处理异常
            context: Optional[AssertRaisesContextIgnoreNotImplementedError] = \
                AssertRaisesContextIgnoreNotImplementedError(expected_exception, self)  # type: ignore[call-arg]
            try:
                return context.handle('assertRaises', args, kwargs)  # type: ignore[union-attr]
            finally:
                # 在最后确保 context 被置为 None，避免潜在的问题
                context = None
        else:
            # 否则调用父类的 assertRaises 方法处理异常
            return super().assertRaises(expected_exception, *args, **kwargs)
    # 验证在调用中抛出预期异常类型和匹配预期正则表达式的异常。
    # 如果测试用例实例化为非本地设备类型（例如XLA），则不验证消息。

    # 检查测试是否实例化为设备类型，通过检查测试类是否定义了device_type属性，
    # 并且如果定义了，则检查实例化的设备类型是否为本地或非本地
    if hasattr(self, 'device_type') and self.device_type not in NATIVE_DEVICES and self.device_type != "mps":  # type: ignore[attr-defined]
        # 空字符串匹配任何字符串
        expected_regex = ''

    # 如果忽略未实现错误，则创建AssertRaisesContextIgnoreNotImplementedError上下文处理器，
    # 并返回其处理结果
    if self._ignore_not_implemented_error:
        context = AssertRaisesContextIgnoreNotImplementedError(  # type: ignore[call-arg]
            expected_exception, self, expected_regex)
        return context.handle('assertRaisesRegex', args, kwargs)  # type: ignore[attr-defined]
    else:
        # 否则调用父类的assertRaisesRegex方法进行验证
        return super().assertRaisesRegex(expected_exception, expected_regex, *args, **kwargs)

# 验证在调用中不会引发无法提升的异常。与普通异常不同，这些异常实际上不会传播到调用方并且会被抑制。我们必须特别测试它们。
def assertNoUnraisable(self, callable, *args, **kwargs):
    raised = None

    def record_unraisable(unraisable):
        nonlocal raised
        raised = unraisable

    # 在运行callable时禁用GC，以防止在callable内部进行不幸的GC操作导致的干扰
    prev = gc.isenabled()
    gc.disable()
    try:
        # 使用unittest.mock.patch记录未提升的钩子
        with unittest.mock.patch("sys.unraisablehook", record_unraisable):
            callable(*args, **kwargs)
    finally:
        # 恢复GC状态
        if prev:
            gc.enable()

    # 断言没有未提升的异常被引发
    self.assertIsNone(raised)

# TODO: 支持上下文管理器接口
# 注意：转发给callable的kwargs会丢失'subname'参数。
# 如果需要'subname'参数，可以手动在lambda中应用您的callable。
def assertExpectedRaises(self, exc_type, callable, *args, **kwargs):
    subname = None
    # 如果kwargs中包含'subname'参数，则将其提取出来并从kwargs中删除
    if 'subname' in kwargs:
        subname = kwargs['subname']
        del kwargs['subname']
    try:
        # 调用callable，并捕获预期的异常类型
        callable(*args, **kwargs)
    except exc_type as e:
        # 断言引发的异常的字符串表示符合预期的'subname'
        self.assertExpected(str(e), subname)
        return
    # 如果没有引发预期的异常类型，则测试失败
    self.fail(msg="Did not raise when expected to")
    # 断言函数不会引发任何警告
    def assertNotWarn(self, callable, msg=''):
        r"""
        Test if :attr:`callable` does not raise a warning.
        """
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            # 设置警告过滤器，允许引发任何警告
            warnings.simplefilter("always")
            # 使用 set_warn_always_context(True) 设置上下文以确保警告行为符合预期
            with set_warn_always_context(True):
                callable()  # 调用待测函数
            # 断言没有捕获到任何警告
            self.assertTrue(len(ws) == 0, msg)

    # 上下文管理器，用于确保代码必须始终发出警告
    def assertWarnsOnceRegex(self, category, regex=''):
        """Context manager for code that *must always* warn

        This filters expected warnings from the test and fails if
        the expected warning is not caught. It uses set_warn_always() to force
        TORCH_WARN_ONCE to behave like TORCH_WARN
        """
        pattern = re.compile(regex)  # 编译正则表达式
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            # 设置警告过滤器，允许引发任何警告
            warnings.simplefilter("always")
            # 使用 set_warn_always_context(True) 设置上下文以确保警告行为符合预期
            with set_warn_always_context(True):
                yield  # 执行被装饰的代码块
            # 如果未捕获到任何警告，测试失败
            if len(ws) == 0:
                self.fail('no warning caught')
            # 断言至少有一个警告的类型符合指定的 category
            self.assertTrue(any(type(w.message) is category for w in ws))
            # 断言至少有一个警告的消息文本匹配指定的正则表达式 pattern
            self.assertTrue(
                any(re.match(pattern, str(w.message)) for w in ws),
                f'{pattern}, {[w.message for w in ws if type(w.message) is category]}')

    # 断言函数执行结果满足预期，并移除 Torch 生成的特定字符串
    def assertExpectedStripMangled(self, s, subname=None):
        s = re.sub(r'__torch__[^ ]+', '', s)  # 移除字符串中匹配特定模式的部分
        self.assertExpected(s, subname)  # 断言处理后的字符串满足预期

    # 断言第一个值大于或几乎等于第二个值，支持设置精度或绝对差
    def assertGreaterAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """Assert that ``first`` is greater than or almost equal to ``second``.

        The equality of ``first`` and ``second`` is determined in a similar way to
        the ``assertAlmostEqual`` function of the standard library.
        """
        # 如果同时指定了 delta 和 places，抛出 TypeError
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        # 如果 first 大于等于 second，则断言通过
        if first >= second:
            return

        # 计算两个值的差异
        diff = second - first
        # 如果指定了 delta，则使用绝对差异进行比较
        if delta is not None:
            if diff <= delta:
                return
            standardMsg = f"{first} not greater than or equal to {second} within {delta} delta"
        else:
            # 如果未指定 delta，则使用指定的精度 places 进行比较
            if places is None:
                places = 7
            if round(diff, places) == 0:
                return
            standardMsg = f"{first} not greater than or equal to {second} within {places} places"

        # 格式化错误信息并引发断言失败异常
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)
    # 定义一个名为 assertAtenOp 的方法，用于在给定的 ONNX 模型中查找特定的 ATen 操作节点
    def assertAtenOp(self, onnx_model, operator, overload_name=""):
        # 从 ONNX 模型的节点列表中筛选出所有操作类型为 "ATen"，并且域为 "org.pytorch.aten" 的节点
        all_aten_nodes = [p for p in onnx_model.graph.node
                          if p.op_type == "ATen" and p.domain == "org.pytorch.aten"]
        # 断言至少存在一个符合条件的 ATen 节点
        self.assertTrue(all_aten_nodes)

        # 遍历所有的 ATen 节点
        for op in all_aten_nodes:
            # 提取节点的属性，将属性的字节字符串解码为字符串，并组成一个属性字典
            attrs = {attr.name: attr.s.decode() for attr in op.attribute}
            # 如果节点的操作符属性等于指定的 operator
            if attrs.get("operator") == operator:
                # 停止遍历，找到了符合条件的节点
                break

        # 断言找到的节点的操作符属性与指定的 operator 相等
        self.assertEqual(attrs["operator"], operator)
        # 断言找到的节点的重载名称属性（如果有的话）与指定的 overload_name 相等
        self.assertEqual(attrs.get("overload_name", ""), overload_name)
    def check_nondeterministic_alert(self, fn, caller_name, should_alert=True):
        '''Checks that an operation produces a nondeterministic alert when
        expected while `torch.use_deterministic_algorithms(True)` is set.

        Args:
          fn (callable): Function to check for a nondeterministic alert

          caller_name (str): Name of the operation that produces the
              nondeterministic alert. This name is expected to appear at the
              beginning of the error/warning message.

          should_alert (bool, optional): If True, then the check will only pass
              if calling `fn` produces a nondeterministic error/warning with the
              expected message. If False, then the check will only pass if
              calling `fn` does not produce an error. Default: `True`.
        '''

        alert_message = '^' + caller_name + ' does not have a deterministic implementation, but you set'

        # Check that errors are thrown correctly
        with DeterministicGuard(True):
            if should_alert:
                # Check that a RuntimeError with alert_message is raised
                with self.assertRaisesRegex(
                        RuntimeError,
                        alert_message,
                        msg='expected a non-deterministic error, but it was not raised'):
                    fn()
            else:
                try:
                    # Ensure no RuntimeError is raised
                    fn()
                except RuntimeError as e:
                    # Fail the test if an unexpected RuntimeError is caught
                    if 'does not have a deterministic implementation' in str(e):
                        self.fail(
                            'did not expect non-deterministic error message, '
                            + 'but got one anyway: "' + str(e) + '"')
                    # Reraise exceptions unrelated to nondeterminism
                    raise

        # Check that warnings are thrown correctly
        with DeterministicGuard(True, warn_only=True):
            if should_alert:
                # Check that a UserWarning with alert_message is issued
                with self.assertWarnsRegex(
                        UserWarning,
                        alert_message):
                    fn()
            else:
                # Check that no UserWarning with alert_message is issued
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    fn()
                    for warning in w:
                        if isinstance(warning, UserWarning):
                            # Fail the test if an unexpected UserWarning is caught
                            self.assertTrue(re.search(alert_message, str(warning)) is None)

    # run code in subprocess and capture exceptions.
    @staticmethod
    def run_process_no_exception(code, env=None):
        import subprocess

        # Start a subprocess to run the given code
        popen = subprocess.Popen(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env)
        (stdout, stderr) = popen.communicate()
        return (stdout, stderr)

    # returns captured stderr
    @staticmethod
    # 定义一个函数，用于在设置了 PYTORCH_API_USAGE_STDERR 环境变量的情况下执行给定的代码，并返回标准错误输出
    def runWithPytorchAPIUsageStderr(code):
        # 复制当前环境变量以避免修改原始环境
        env = os.environ.copy()
        # 设置 PYTORCH_API_USAGE_STDERR 环境变量为 "1"，用于记录 PyTorch API 使用情况到标准错误输出
        env["PYTORCH_API_USAGE_STDERR"] = "1"
        
        # 如果当前环境中存在 "CI" 标志，则从环境变量中删除它
        # 因为这是一个包装的测试过程，CI 标志应该只在父进程中设置
        if "CI" in env.keys():
            del env["CI"]
        
        # 调用 TestCase 类的 run_process_no_exception 方法执行指定代码，传入修改后的环境变量
        # 捕获执行过程中的标准输出和标准错误输出
        (stdout, stderr) = TestCase.run_process_no_exception(code, env=env)
        
        # 将标准错误输出解码为 ASCII 字符串，并返回
        return stderr.decode('ascii')
class TestCaseBase(TestCase):
    """
    Base class for test cases.

    Calls to super() in dynamically created classes are a bit odd.
    See https://github.com/pytorch/pytorch/pull/118586 for more info
    Subclassing this class and then calling super(TestCaseBase) will run
    TestCase's setUp, tearDown etc functions
    """
    pass


def download_file(url, binary=True):
    """
    Download a file from a URL and save it locally.

    :param url: The URL of the file to download.
    :param binary: Flag indicating if the file should be saved in binary mode (default: True).
    :return: The local path where the file is saved.
    """
    from urllib.parse import urlsplit
    from urllib import request, error

    filename = os.path.basename(urlsplit(url)[2])
    data_dir = get_writable_path(os.path.join(os.path.dirname(__file__), 'data'))
    path = os.path.join(data_dir, filename)

    if os.path.exists(path):
        return path
    try:
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb' if binary else 'w') as f:
            f.write(data)
        return path
    except error.URLError as e:
        msg = f"could not download test file '{url}'"
        warnings.warn(msg, RuntimeWarning)
        raise unittest.SkipTest(msg) from e


def find_free_port():
    """
    Finds an available port and returns that port number.

    NOTE: If this function is being used to allocate a port to Store (or
    indirectly via init_process_group or init_rpc), it should be used
    in conjunction with the `retry_on_connect_failures` decorator as there is a potential
    race condition where the allocated port may become unavailable before it can be used
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('localhost', 0))
        _, port = sock.getsockname()
        return port


# Errors that we can get in c10d initialization for which we should retry tests for.
ADDRESS_IN_USE = "Address already in use"
CONNECT_TIMEOUT = "connect() timed out."


def retry_on_connect_failures(func=None, connect_errors=(ADDRESS_IN_USE)):
    """
    Reruns a test if the test returns a RuntimeError and the exception
    contains one of the strings in connect_errors.

    :param func: The function to decorate.
    :param connect_errors: Tuple of error strings that trigger retrying the function.
    :return: Decorated function.
    """
    # This if block is executed when using this function as a decorator with arguments.
    if func is None:
        return partial(retry_on_connect_failures, connect_errors=connect_errors)

    @wraps(func)
    def wrapper(*args, **kwargs):
        n_retries = 10
        tries_remaining = n_retries
        while True:
            try:
                return func(*args, **kwargs)
            except RuntimeError as error:
                if any(connect_error in str(error) for connect_error in connect_errors):
                    tries_remaining -= 1
                    if tries_remaining == 0:
                        raise RuntimeError(f"Failing after {n_retries} retries with error: {str(error)}") from error
                    time.sleep(random.random())
                    continue
                raise

    return wrapper


# Decorator to retry upon certain Exceptions.
def retry(ExceptionToCheck, tries=3, delay=3, skip_after_retries=False):
    """
    Decorator function that retries executing a function if certain exceptions are raised.

    :param ExceptionToCheck: The exception (or tuple of exceptions) to catch and retry upon.
    :param tries: Number of retry attempts.
    :param delay: Delay in seconds between retries.
    :param skip_after_retries: Flag to skip the function call after retries are exhausted.
    """
    # 定义一个装饰器函数，用于给函数添加重试功能
    def deco_retry(f):
        # 使用 functools.wraps 装饰器来保留原始函数的元数据
        @wraps(f)
        def f_retry(*args, **kwargs):
            # 初始化重试次数和重试延迟时间
            mtries, mdelay = tries, delay
            # 当还有重试次数时继续重试
            while mtries > 1:
                try:
                    # 调用被装饰的函数，并返回其结果
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    # 如果捕获到特定异常，则打印异常信息并等待一段时间后重试
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            # 如果所有重试均失败，则进行最后一次尝试
            try:
                return f(*args, **kwargs)
            except ExceptionToCheck as e:
                # 如果仍然捕获到异常，根据设置决定是否抛出 SkipTest 异常
                raise unittest.SkipTest(f"Skipping after {tries} consecutive {str(e)}") from e if skip_after_retries else e
        
        return f_retry  # 返回经过装饰后的函数对象，即真正的装饰器
# FIXME: modernize these to be consistent with make_tensor
#   and review including them in torch.testing
# Methods for matrix generation

# 生成一个具有指定秩的随机方阵
def random_square_matrix_of_rank(l, rank, dtype=torch.double, device='cpu'):
    assert rank <= l  # 断言：秩应小于等于方阵的大小
    A = torch.randn(l, l, dtype=dtype, device=device)  # 生成一个随机的 l x l 大小的张量 A
    u, s, vh = torch.linalg.svd(A, full_matrices=False)  # 对 A 进行奇异值分解
    for i in range(l):
        if i >= rank:
            s[i] = 0  # 将秩大于等于指定秩的奇异值设为0
        elif s[i] == 0:
            s[i] = 1  # 将为0的奇异值设为1
    return (u * s.to(dtype).unsqueeze(-2)) @ vh  # 返回按指定秩修剪后的奇异值分解结果

# 生成一个条件良好的随机矩阵（或批量矩阵）
def random_well_conditioned_matrix(*shape, dtype, device, mean=1.0, sigma=0.001):
    """
    返回一个随机的矩形矩阵（或矩阵批量），其奇异值从均值为 `mean`，标准差为 `sigma` 的高斯分布中采样。
    `sigma` 越小，输出矩阵的条件数越好。
    """
    primitive_dtype = {
        torch.float: torch.float,
        torch.double: torch.double,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double
    }
    x = torch.rand(shape, dtype=dtype, device=device)  # 生成一个指定形状的随机张量 x
    m = x.size(-2)
    n = x.size(-1)
    u, _, vh = torch.linalg.svd(x, full_matrices=False)  # 对 x 进行奇异值分解
    s = (torch.randn(*(shape[:-2] + (min(m, n),)), dtype=primitive_dtype[dtype], device=device) * sigma + mean) \
        .sort(-1, descending=True).values.to(dtype)  # 生成符合要求的奇异值并排序
    return (u * s.unsqueeze(-2)) @ vh  # 返回按照指定条件生成的矩阵

# 返回一个与给定张量 t 形状和值相同但非连续的张量
def noncontiguous_like(t):
    # 如果 t 已经是非连续的，则直接返回 t
    if not t.is_contiguous():
        return t

    # 选择一个“奇怪”的值，以确保内部维度的元素间隔为零或 NaN（如果可能）
    if t.dtype.is_floating_point or t.dtype.is_complex:
        value = math.nan
    elif t.dtype == torch.bool:
        value = True
    else:
        value = 12

    result = t.new_empty(t.shape + (2,))
    result[..., 0] = value  # 第一个维度的值设为奇怪的值
    result[..., 1] = t.detach()  # 第二个维度的值设为 t 的分离版本
    result = result[..., 1]  # 返回第二个维度的值
    result.requires_grad_(t.requires_grad)  # 设置梯度信息与 t 一致
    return result  # 返回非连续的结果张量

# TODO: remove this (prefer make_symmetric_matrices below)
# 生成一个随机对称矩阵（或批量对称矩阵）
def random_symmetric_matrix(l, *batches, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)  # 生成一个随机的 l x l 大小的张量 A
    A = (A + A.mT).div_(2)  # 将 A 与其转置矩阵相加然后除以2，生成对称矩阵
    return A  # 返回生成的对称矩阵

# 创建一个对称矩阵或批量对称矩阵
# 形状必须是方阵或方阵批量
def make_symmetric_matrices(*shape, device, dtype):
    assert shape[-1] == shape[-2]  # 断言：最后两个维度应相等，即形成方阵
    t = make_tensor(shape, device=device, dtype=dtype)  # 生成一个指定形状的张量 t
    t = (t + t.mT).div_(2)  # 将 t 与其转置矩阵相加然后除以2，生成对称矩阵
    return t  # 返回生成的对称矩阵

# 生成一个随机的埃尔米特矩阵（或批量埃尔米特矩阵）
def random_hermitian_matrix(l, *batches, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)  # 生成一个随机的 l x l 大小的张量 A
    # 将张量 A 与其转置的平均值相加，然后进行原地除以2的操作
    A = (A + A.mH).div_(2)
    # 返回操作后的张量 A
    return A
def random_symmetric_psd_matrix(l, *batches, **kwargs):
    """
    Returns a batch of random symmetric positive-semi-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_symmetric_psd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    # 获取关键字参数中的数据类型（默认为 torch.double）
    dtype = kwargs.get('dtype', torch.double)
    # 获取关键字参数中的设备类型（默认为 'cpu'）
    device = kwargs.get('device', 'cpu')
    # 生成一个 l x l 大小的随机张量，数据类型为 dtype，存储设备为 device
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    # 返回 A 与其转置矩阵的乘积，保证结果是半正定矩阵
    return A @ A.mT


def random_hermitian_psd_matrix(matrix_size, *batch_dims, dtype=torch.double, device='cpu'):
    """
    Returns a batch of random Hermitian positive-semi-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_psd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    # 生成一个具有 batch_dims + (matrix_size, matrix_size) 形状的随机张量
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)), dtype=dtype, device=device)
    # 返回 A 与其共轭转置矩阵的乘积，保证结果是半正定矩阵
    return A @ A.mH


# TODO: remove this (prefer make_symmetric_pd_matrices below)
def random_symmetric_pd_matrix(matrix_size, *batch_dims, **kwargs):
    # 获取关键字参数中的数据类型（默认为 torch.double）
    dtype = kwargs.get('dtype', torch.double)
    # 获取关键字参数中的设备类型（默认为 'cpu'）
    device = kwargs.get('device', 'cpu')
    # 生成一个具有 batch_dims + (matrix_size, matrix_size) 形状的随机张量
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    # 返回 A 与其转置矩阵的乘积，再加上一个微小的单位矩阵，确保结果是正定的
    return torch.matmul(A, A.mT) \
        + torch.eye(matrix_size, dtype=dtype, device=device) * 1e-5


# Creates a symmetric positive-definite matrix or batch of
#   such matrices
def make_symmetric_pd_matrices(*shape, device, dtype):
    # 断言最后两个维度相等，以确保是方阵
    assert shape[-1] == shape[-2]
    # 生成一个指定形状的张量，并在对角线上加上一个微小的单位矩阵，确保结果是正定的
    t = make_tensor(shape, device=device, dtype=dtype)
    i = torch.eye(shape[-1], device=device, dtype=dtype) * 1e-5
    return t @ t.mT + i


def random_hermitian_pd_matrix(matrix_size, *batch_dims, dtype, device):
    """
    Returns a batch of random Hermitian positive-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_pd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    # 生成一个具有 batch_dims + (matrix_size, matrix_size) 形状的随机张量
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    # 返回 A 与其共轭转置矩阵的乘积，再加上一个微小的单位矩阵，确保结果是正定的
    return A @ A.mH + torch.eye(matrix_size, dtype=dtype, device=device)

# Creates a full rank matrix with distinct singular values or
#   a batch of such matrices
def make_fullrank_matrices_with_distinct_singular_values(*shape, device, dtype, requires_grad=False):
    # 使用 torch.no_grad() 上下文管理器，确保在此范围内的运算不会被记录梯度
    with torch.no_grad():
        # 创建一个指定形状、设备和数据类型的张量 t
        t = make_tensor(shape, device=device, dtype=dtype)
        # 对张量 t 进行奇异值分解（SVD），返回结果中的左奇异向量、奇异值和右奇异向量
        u, _, vh = torch.linalg.svd(t, full_matrices=False)
        # 确定张量 t 的实部数据类型（如果 t 是复数类型则为实部数据类型）
        real_dtype = t.real.dtype if t.dtype.is_complex else t.dtype
        # 取最小的两个维度作为奇异值个数 k
        k = min(shape[-1], shape[-2])
        # 初始化奇异值向量 s，起始值为 [2, 3, ..., k+1]
        s = torch.arange(2, k + 2, dtype=real_dtype, device=device)
        # 将 s 中的奇数索引位置的值乘以 -1，得到 [2, -3, 4, ..., (-1)^k k+1]
        s[1::2] *= -1.
        # 将 s 取倒数并加上 1，以使得奇异值位于区间 [2/3, 3/2]
        s.reciprocal_().add_(1.)
        # 计算最终的生成矩阵 x，其中 u 是左奇异向量，vh 是右奇异向量
        x = (u * s.to(u.dtype)) @ vh
    # 将生成的张量 x 设置为需要梯度计算
    x.requires_grad_(requires_grad)
    # 返回计算结果 x
    return x
# 返回一个具有指定行和列的矩阵或矩阵批次的随机矩阵。
# 支持以下关键字参数:
#   dtype - 数据类型
#   device - 设备类型
#   silent - 当为True时，如果没有lapack支持，返回全为1的矩阵
#   singular - 当为True时，生成的矩阵将是奇异的
def random_matrix(rows, columns, *batch_dims, **kwargs):
    dtype = kwargs.get('dtype', torch.double)  # 获取数据类型，默认为torch.double
    device = kwargs.get('device', 'cpu')       # 获取设备类型，默认为cpu
    silent = kwargs.get("silent", False)       # 获取silent参数，默认为False
    singular = kwargs.get("singular", False)   # 获取singular参数，默认为False
    
    if silent and not torch._C.has_lapack:  # 如果silent为True且系统没有lapack支持
        return torch.ones(rows, columns, dtype=dtype, device=device)  # 返回全为1的矩阵
    
    # 生成随机的矩阵A，维度为batch_dims + (rows, columns)，数据类型为dtype，设备类型为device
    A = torch.randn(batch_dims + (rows, columns), dtype=dtype, device=device)
    
    if A.numel() == 0:  # 如果A中元素个数为0
        return A
    
    # 对A进行奇异值分解(SVD)，返回左奇异向量u、奇异值s、右奇异向量的转置vh，限制为非全尺寸矩阵
    u, _, vh = torch.linalg.svd(A, full_matrices=False)
    
    k = min(rows, columns)  # 取rows和columns中较小的作为k值
    # 创建一个从1 / (k + 1)到1的等差序列，长度为k，数据类型为dtype，设备类型为device
    s = torch.linspace(1 / (k + 1), 1, k, dtype=dtype, device=device)
    
    if singular:
        # 将最后一个奇异值设为0，使得矩阵奇异化
        s[k - 1] = 0
        if k > 2:
            # 将第一个奇异值设为0，增加奇异性的阶数，使得LU分解中的枢轴选择变得非平凡
            s[0] = 0
    
    # 返回奇异值分解后的矩阵乘积结果
    return (u * s.unsqueeze(-2)) @ vh


# 返回具有给定秩的矩阵或矩阵批次的随机低秩矩阵
def random_lowrank_matrix(rank, rows, columns, *batch_dims, **kwargs):
    # 生成行数为rows，列数为rank的随机矩阵B
    B = random_matrix(rows, rank, *batch_dims, **kwargs)
    # 生成行数为rank，列数为columns的随机矩阵C
    C = random_matrix(rank, columns, *batch_dims, **kwargs)
    # 返回矩阵B和矩阵C的乘积结果
    return B.matmul(C)


# 返回具有给定密度的矩阵的稀疏随机矩阵
def random_sparse_matrix(rows, columns, density=0.01, **kwargs):
    dtype = kwargs.get('dtype', torch.double)  # 获取数据类型，默认为torch.double
    device = kwargs.get('device', 'cpu')       # 获取设备类型，默认为cpu
    singular = kwargs.get("singular", False)   # 获取singular参数，默认为False
    
    k = min(rows, columns)  # 取rows和columns中较小的作为k值
    # 计算非零元素的数量，最小为min(rows, columns)，最大为int(rows * columns * density)
    nonzero_elements = max(min(rows, columns), int(rows * columns * density))
    
    # 生成随机的行索引和列索引，确保每一列至少有一个非零元素
    row_indices = [i % rows for i in range(nonzero_elements)]
    column_indices = [i % columns for i in range(nonzero_elements)]
    random.shuffle(column_indices)  # 随机打乱列索引的顺序
    
    indices = [row_indices, column_indices]
    
    # 生成非零元素的值，按照对角线主导的方式进行调整
    values = torch.randn(nonzero_elements, dtype=dtype, device=device)
    values *= torch.tensor([-float(i - j)**2 for i, j in zip(*indices)], dtype=dtype, device=device).exp()
    
    indices_tensor = torch.tensor(indices)
    # 生成稀疏的COO格式的张量A
    A = torch.sparse_coo_tensor(indices_tensor, values, (rows, columns), device=device)
    # 将COO格式的张量A转换为稀疏表示并返回
    return A.coalesce()


# 返回具有给定密度的随机稀疏正定矩阵
def random_sparse_pd_matrix(matrix_size, density=0.01, **kwargs):
    # 定义矩阵的大小为matrix_size
    # 生成矩阵的特征值，范围为1到matrix_size+1
    # 返回特征值
    Such As Did Sh Offer T Gan May Be Ntsi Philosophy Additionally Prior Medicine Degree Problem She ? Its
    """
    Algorithm:
      A = diag(arange(1, matrix_size+1)/matrix_size)
      while <A density is smaller than required>:
          <choose random i, j in range(matrix_size), theta in [0, 2*pi]>
          R = <rotation matrix (i,j,theta)>
          A = R^T A R
    """

    # 导入数学库
    import math
    # 获取torch模块，若不存在则使用全局的torch模块
    torch = kwargs.get('torch', globals()['torch'])
    # 设置数据类型，默认为torch.double
    dtype = kwargs.get('dtype', torch.double)
    # 设置设备，默认为cpu
    device = kwargs.get('device', 'cpu')
    # 创建一个字典，用来存储对角线元素
    data = {(i, i): float(i + 1) / matrix_size
            for i in range(matrix_size)}

    # 定义一个函数，用于矩阵乘法操作
    def multiply(data, N, i, j, cs, sn, left=True):
        for k in range(N):
            if left:
                ik, jk = (k, i), (k, j)
            else:
                ik, jk = (i, k), (j, k)
            aik, ajk = data.get(ik, 0), data.get(jk, 0)
            aik, ajk = cs * aik + sn * ajk, -sn * aik + cs * ajk
            if aik:
                data[ik] = aik
            else:
                data.pop(ik, None)
            if ajk:
                data[jk] = ajk
            else:
                data.pop(jk, None)

    # 计算目标稀疏矩阵的非零元素个数
    target_nnz = density * matrix_size * matrix_size
    # 当data中的元素个数小于目标非零元素个数时，执行以下循环
    while len(data) < target_nnz:
        # 随机选择两个不同的索引 i 和 j
        i = random.randint(0, matrix_size - 1)
        j = random.randint(0, matrix_size - 1)
        if i != j:
            # 随机选择一个角度 theta
            theta = random.uniform(0, 2 * math.pi)
            cs = math.cos(theta)
            sn = math.sin(theta)
            # 执行两次乘法操作，left参数分别为True和False
            multiply(data, matrix_size, i, j, cs, sn, left=True)
            multiply(data, matrix_size, i, j, cs, sn, left=False)

    # 从data中提取出行坐标、列坐标和对应的值
    icoords, jcoords, values = [], [], []
    for (i, j), v in sorted(data.items()):
        icoords.append(i)
        jcoords.append(j)
        values.append(v)
    
    # 创建稀疏 COO 张量，并返回
    indices_tensor = torch.tensor([icoords, jcoords])
    return torch.sparse_coo_tensor(indices_tensor, values, (matrix_size, matrix_size), dtype=dtype, device=device)
# FIXME: remove this by updating test suites using it
# 对于每种数据类型进行测试
def do_test_dtypes(self, dtypes, layout, device):
    # 遍历给定的数据类型列表
    for dtype in dtypes:
        # 如果数据类型不是 torch.float16
        if dtype != torch.float16:
            # 创建一个全零张量，指定数据类型、布局和设备
            out = torch.zeros((2, 3), dtype=dtype, layout=layout, device=device)
            # 断言张量的数据类型与给定的数据类型一致
            self.assertIs(dtype, out.dtype)
            # 断言张量的布局与给定的布局一致
            self.assertIs(layout, out.layout)
            # 断言张量的设备与给定的设备一致
            self.assertEqual(device, out.device)

# FIXME: remove this by updating test suites using it
# 对于 torch.empty 和 torch.full 函数进行测试
def do_test_empty_full(self, dtypes, layout, device):
    # 定义张量的形状为 [2, 3]
    shape = torch.Size([2, 3])

    # 定义检查张量值的函数
    def check_value(tensor, dtype, layout, device, value, requires_grad):
        # 断言张量的形状与预期形状一致
        self.assertEqual(shape, tensor.shape)
        # 断言张量的数据类型与给定的数据类型一致
        self.assertIs(dtype, tensor.dtype)
        # 断言张量的布局与给定的布局一致
        self.assertIs(layout, tensor.layout)
        # 断言张量的 requires_grad 属性与预期一致
        self.assertEqual(tensor.requires_grad, requires_grad)
        # 如果张量在 GPU 上，并且给定了设备，断言张量的设备与给定设备一致
        if tensor.is_cuda and device is not None:
            self.assertEqual(device, tensor.device)
        # 如果给定了值，创建一个与张量形状相同并填充指定值的张量，断言两者相等
        if value is not None:
            fill = tensor.new(shape).fill_(value)
            self.assertEqual(tensor, fill)

    # 获取默认数据类型
    default_dtype = torch.get_default_dtype()
    # 测试 torch.empty 函数
    check_value(torch.empty(shape), default_dtype, torch.strided, -1, None, False)
    # 测试 torch.full 函数
    check_value(torch.full(shape, -5.), default_dtype, torch.strided, -1, None, False)
    # 对于每种数据类型执行以下操作
    for dtype in dtypes:
        # 对于每种数据类型，包括浮点数和非浮点数（False）
        for rg in {dtype.is_floating_point, False}:
            # 获取相应的 int64 数据类型
            int64_dtype = get_int64_dtype(dtype)
            
            # 创建一个空的张量 v，根据指定的形状、数据类型、设备、布局和是否需要梯度进行设置
            v = torch.empty(shape, dtype=dtype, device=device, layout=layout, requires_grad=rg)
            
            # 检查新创建的张量 v 的值是否符合预期
            check_value(v, dtype, layout, device, None, rg)
            
            # 创建一个新的张量 out，使用 v 作为原型
            out = v.new()
            
            # 使用指定的形状、设备、布局、数据类型和是否需要梯度创建一个空张量，并检查其值是否符合预期
            check_value(torch.empty(shape, out=out, device=device, layout=layout, requires_grad=rg),
                        dtype, layout, device, None, rg)
            
            # 使用 v 的形状创建一个空张量，并检查其值是否符合预期
            check_value(v.new_empty(shape), dtype, layout, device, None, False)
            
            # 使用指定的形状、数据类型、设备和是否需要梯度创建一个空张量，并检查其值是否符合预期
            check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)
            
            # 使用 v 的形状创建一个与其形状相同的空张量，并检查其值是否符合预期
            check_value(torch.empty_like(v), dtype, layout, device, None, False)
            
            # 使用指定的形状、数据类型、设备和是否需要梯度创建一个与 v 形状相同的空张量，并检查其值是否符合预期
            check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)

            # 如果数据类型不是 torch.float16 并且布局不是 torch.sparse_coo
            if dtype is not torch.float16 and layout != torch.sparse_coo:
                # 设置一个固定的值 fv = 3
                fv = 3
                
                # 创建一个填充了固定值 fv 的张量 v，并检查其值是否符合预期
                v = torch.full(shape, fv, dtype=dtype, layout=layout, device=device, requires_grad=rg)
                check_value(v, dtype, layout, device, fv, rg)
                
                # 使用 v 作为原型创建一个新的张量，并检查其值是否符合预期
                check_value(v.new_full(shape, fv + 1), dtype, layout, device, fv + 1, False)
                
                # 创建一个新的张量 out，使用 v 作为原型
                out = v.new()
                
                # 使用指定的形状、设备、布局、数据类型和是否需要梯度创建一个填充了固定值的张量，并检查其值是否符合预期
                check_value(torch.full(shape, fv + 2, out=out, device=device, layout=layout, requires_grad=rg),
                            dtype, layout, device, fv + 2, rg)
                
                # 使用指定的形状、数据类型、设备和是否需要梯度创建一个填充了固定值的张量，并检查其值是否符合预期
                check_value(v.new_full(shape, fv + 3, dtype=int64_dtype, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 3, False)
                
                # 使用 v 的形状创建一个与其形状相同的张量，并填充固定值 fv + 4，并检查其值是否符合预期
                check_value(torch.full_like(v, fv + 4), dtype, layout, device, fv + 4, False)
                
                # 使用指定的形状、数据类型、设备和是否需要梯度创建一个与 v 形状相同的张量，并填充固定值 fv + 5，并检查其值是否符合预期
                check_value(torch.full_like(v, fv + 5,
                                            dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 5, False)
# FIXME: improve load_tests() documentation here
# 全局变量，用于存储当前运行脚本的绝对路径
running_script_path = None

# 设置运行脚本的路径为全局变量 running_script_path
def set_running_script_path():
    global running_script_path
    try:
        # 获取当前运行脚本的绝对路径
        running_file = os.path.abspath(os.path.realpath(sys.argv[0]))
        # 如果运行脚本是一个 Python 脚本，则将其路径赋值给 running_script_path
        if running_file.endswith('.py'):  # skip if the running file is not a script
            running_script_path = running_file
    except Exception:
        pass

# 检查测试案例是否在当前运行的脚本中定义
def check_test_defined_in_running_script(test_case):
    if running_script_path is None:
        return
    # 获取测试案例所属类的文件路径
    test_case_class_file = os.path.abspath(os.path.realpath(inspect.getfile(test_case.__class__)))
    # 断言测试案例所属类的文件路径与当前运行脚本的路径相同
    assert test_case_class_file == running_script_path, f'Class of loaded TestCase "{test_case.id()}" ' \
        f'is not defined in the running script "{running_script_path}", but in "{test_case_class_file}". Did you ' \
        "accidentally import a unittest.TestCase from another file?"

# 加载测试用例并执行，返回测试套件
def load_tests(loader, tests, pattern):
    # 设置当前运行脚本的路径
    set_running_script_path()
    # 创建一个空的测试套件
    test_suite = unittest.TestSuite()
    # 遍历所有测试组
    for test_group in tests:
        # 如果不禁用运行脚本检查
        if not DISABLE_RUNNING_SCRIPT_CHK:  # noqa: F821
            # 遍历测试组中的每个测试案例，检查其是否定义在当前运行的脚本中
            for test in test_group:
                check_test_defined_in_running_script(test)
        # 如果测试组中有测试案例，则将其加入到测试套件中
        if test_group._tests:
            test_suite.addTest(test_group)
    # 返回构建好的测试套件
    return test_suite

# FIXME: document this and move it to test_serialization
# BytesIOContext 类，继承自 io.BytesIO，用作上下文管理器
class BytesIOContext(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

# 定义梯度检查时使用的非确定性容差值
GRADCHECK_NONDET_TOL = 1e-12

# 在测试环境中定义一个标志，用于控制是否进行慢速梯度检查
TestEnvironment.def_flag("TEST_WITH_SLOW_GRADCHECK", env_var="PYTORCH_TEST_WITH_SLOW_GRADCHECK")

# 如果设置了 TEST_WITH_SLOW_GRADCHECK 标志，则跳过梯度检查测试
skipIfSlowGradcheckEnv = unittest.skipIf(
    TEST_WITH_SLOW_GRADCHECK,  # noqa: F821
    "Tests that don't use gradcheck don't need to run on slow_gradcheck CI"
)

# 包装器函数，用于梯度检查，并启用默认的特定键
def gradcheck(fn, inputs, **kwargs):
    # 默认的键值对
    default_values = {
        "check_batched_grad": True,
        "fast_mode": True,
    }

    # 如果设置了 TEST_WITH_SLOW_GRADCHECK 标志，则禁用 fast_mode
    if TEST_WITH_SLOW_GRADCHECK:  # noqa: F821
        default_values["fast_mode"] = False

    # 遍历默认键值对，如果 kwargs 中没有指定某个键的值，则使用默认值
    for key, value in default_values.items():
        # 如果 kwargs 中已经指定了某个键的值，则不覆盖
        k = kwargs.get(key, None)
        kwargs[key] = k if k is not None else value

    # 调用 torch 的 autograd.gradcheck 函数进行梯度检查
    return torch.autograd.gradcheck(fn, inputs, **kwargs)
# 定义一个函数 gradgradcheck，用于对输入的函数进行二阶梯度检查
def gradgradcheck(fn, inputs, grad_outputs=None, **kwargs):
    # 封装在 gradgradcheck 外部以便默认启用特定键
    # 详见上文 gradcheck 的解释，说明为何需要类似这样的封装函数
    #
    # 所有进行测试的 PyTorch 开发者应该使用这个包装器，而不是直接使用 autograd.gradgradcheck
    default_values = {
        "check_batched_grad": True,  # 检查批次梯度，默认为 True
        "fast_mode": True,  # 使用快速模式，默认为 True
    }

    if TEST_WITH_SLOW_GRADCHECK:  # 如果测试需要慢速梯度检查
        default_values["fast_mode"] = False  # 则禁用快速模式

    for key, value in default_values.items():
        # 默认值覆盖显式设置为 None 的值
        k = kwargs.get(key, None)
        kwargs[key] = k if k is not None else value

    # 调用 torch.autograd.gradgradcheck 进行二阶梯度检查
    return torch.autograd.gradgradcheck(fn, inputs, grad_outputs, **kwargs)


# 定义一个辅助函数 _assertGradAndGradgradChecks，用于在测试用例中断言梯度检查和二阶梯度检查的结果
def _assertGradAndGradgradChecks(test_case, apply_fn, inputs, **kwargs):
    # 调用 assert 函数而不是直接返回布尔值，因为这样更友好
    # 如果梯度检查或二阶梯度检查失败，会在测试用例中断言失败的信息
    test_case.assertTrue(gradcheck(apply_fn, inputs, **kwargs))
    test_case.assertTrue(gradgradcheck(apply_fn, inputs, **kwargs))


# 定义一个上下文管理器 set_cwd，用于设置当前工作目录，并在退出时恢复原始工作目录
@contextmanager
def set_cwd(path: str) -> Iterator[None]:
    old_cwd = os.getcwd()  # 获取当前工作目录
    try:
        os.chdir(path)  # 设置新的工作目录
        yield  # 执行被设置的新工作目录中的代码块
    finally:
        os.chdir(old_cwd)  # 在退出时恢复原始工作目录


# FIXME: delete this
# 推荐使用 @toleranceOverride 来指定测试的具体方法
# 这些值只是在测试 test_nn 中有效的一些数值。
dtype2prec_DONTUSE = {torch.float: 1e-5,
                      torch.double: 1e-5,
                      torch.half: 1e-2,
                      torch.bfloat16: 1e-1}

# FIXME: move to test_sparse or sparse utils
# 定义一个装饰器 coalescedonoff，用于运行测试两次：一次使用 coalesced=True，另一次使用 coalesced=False，用于测试稀疏张量的合并与非合并状态
def coalescedonoff(f):
    @wraps(f)
    def wrapped(self, *args, **kwargs):
        f(self, *args, **kwargs, coalesced=True)  # 运行带有 coalesced=True 的测试
        f(self, *args, **kwargs, coalesced=False)  # 运行带有 coalesced=False 的测试
    return wrapped


# 定义一个函数 is_coalesced_indices，用于检查稀疏张量的索引是否已经合并
def is_coalesced_indices(s):
    indices = s._indices()  # 获取稀疏张量的索引
    hash_coeffs = (1,) + s.shape[s.sparse_dim() - 1:0:-1]  # 计算哈希系数
    hash_indices = torch.tensor(hash_coeffs, device=s.device).cumprod(-1).flip(-1)  # 计算哈希索引
    if s.sparse_dim() > 1:
        hash_indices.unsqueeze_(-1)
        hash_indices = (indices * hash_indices).sum(0)
    else:
        hash_indices = indices * hash_indices

    # 检查索引是否已排序
    res = torch.allclose(hash_indices, hash_indices.sort()[0])

    # 检查索引是否没有重复
    res = res and torch.allclose(hash_indices, hash_indices.unique())

    return res


# 定义一个上下文管理器 disable_gc，用于在测试期间禁用和恢复垃圾回收
@contextlib.contextmanager
def disable_gc():
    if gc.isenabled():  # 如果垃圾回收器已启用
        try:
            gc.disable()  # 禁用垃圾回收
            yield  # 执行禁用垃圾回收后的代码块
        finally:
            gc.enable()  # 在退出时重新启用垃圾回收
    else:
        yield  # 如果垃圾回收器未启用，则直接执行代码块


# 定义一个函数 find_library_location，用于查找指定库文件的位置
def find_library_location(lib_name: str) -> Path:
    # 如果存在安装文件夹中的共享库文件，则返回该文件，
    # 否则返回构建文件夹中的文件
    # 获取 torch 库安装路径的根目录
    torch_root = Path(torch.__file__).resolve().parent
    # 构建要查找的库文件的路径，该文件位于 torch 库的 'lib' 目录下
    path = torch_root / 'lib' / lib_name
    # 如果找到了指定路径的文件
    if os.path.exists(path):
        # 返回该路径
        return path
    # 如果未找到指定路径的文件，则重新定位到当前文件的父目录的父目录的父目录，作为新的 torch 根目录
    torch_root = Path(__file__).resolve().parent.parent.parent
    # 返回重新定位后的路径，该文件位于 torch 根目录的 'build' 目录下的 'lib' 目录下
    return torch_root / 'build' / 'lib' / lib_name
# 定义一个装饰器函数，用于在沙堡环境中“跳过”测试，避免因连续跳过测试而创建任务投诉
def skip_but_pass_in_sandcastle(reason):
    """
    Similar to unittest.skip, however in the sandcastle environment it just
    "passes" the test instead to avoid creating tasks complaining about tests
    skipping continuously.
    """
    def decorator(func):
        # 如果不是沙堡环境，则标记函数跳过测试，并记录跳过原因
        if not IS_SANDCASTLE:  # noqa: F821
            func.__unittest_skip__ = True
            func.__unittest_skip_why__ = reason
            return func

        # 在沙堡环境中，定义一个包装器函数，打印跳过测试的信息并直接返回
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
            return
        return wrapper

    return decorator


# 返回一个函数，该函数调用给定方法的真实实现，并将参数传递给模拟对象
def mock_wrapper(method):
    """
    Returns a function that calls the real implementation of a method
    in addition to passing args to a mock object.
    """
    mock = MagicMock()  # 创建一个 MagicMock 对象作为模拟对象

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)  # 调用模拟对象的方法，并传递参数
        return method(self, *args, **kwargs)  # 调用真实方法，并返回其结果
    wrapper.mock = mock  # 将模拟对象保存在 wrapper 函数的属性中
    return wrapper


# 返回给定参数和关键字参数中所有 Tensor 对象的集合
def get_tensors_from(args, kwargs):
    """ Returns a set of all Tensor objects in the given args and kwargs. """
    return set([arg for arg in args if isinstance(arg, Tensor)] +
               [v for v in kwargs.values() if isinstance(v, Tensor)])


# 返回一个由整数字节值列表表示的标量张量表示
def bytes_to_scalar(byte_list: List[int], dtype: torch.dtype, device: torch.device):
    """
    Returns scalar tensor representation of a list of integer byte values
    """
    # 定义不同数据类型到 ctypes 类型的映射关系
    dtype_to_ctype: Dict[torch.dtype, Any] = {
        torch.int8: ctypes.c_int8,
        torch.uint8: ctypes.c_uint8,
        torch.uint16: ctypes.c_uint16,
        torch.uint32: ctypes.c_uint32,
        torch.uint64: ctypes.c_uint64,
        torch.int16: ctypes.c_int16,
        torch.int32: ctypes.c_int32,
        torch.int64: ctypes.c_int64,
        torch.bool: ctypes.c_bool,
        torch.float32: ctypes.c_float,
        torch.complex64: ctypes.c_float,
        torch.float64: ctypes.c_double,
        torch.complex128: ctypes.c_double,
    }
    ctype = dtype_to_ctype[dtype]  # 获取给定 dtype 对应的 ctypes 类型
    num_bytes = ctypes.sizeof(ctype)  # 计算该 ctypes 类型的大小（字节数）

    # 检查字节列表中的每个字节是否在合法范围内（0~255）
    def check_bytes(byte_list):
        for byte in byte_list:
            assert 0 <= byte <= 255

    # 如果 dtype 是复数类型，则字节列表的长度应为 ctypes 类型大小的两倍
    if dtype.is_complex:
        assert len(byte_list) == (num_bytes * 2)  # 断言字节列表长度正确
        check_bytes(byte_list)  # 检查字节列表中的字节合法性
        # 将字节列表分成实部和虚部，然后将它们转换为对应的 ctypes 类型值
        real = ctype.from_buffer((ctypes.c_byte * num_bytes)(*byte_list[:num_bytes])).value
        imag = ctype.from_buffer((ctypes.c_byte * num_bytes)(*byte_list[num_bytes:])).value
        res = real + 1j * imag  # 构建复数值
    else:
        assert len(byte_list) == num_bytes  # 断言字节列表长度正确
        check_bytes(byte_list)  # 检查字节列表中的字节合法性
        # 将字节列表转换为对应的 ctypes 类型值
        res = ctype.from_buffer((ctypes.c_byte * num_bytes)(*byte_list)).value

    # 返回构建的张量，指定设备和数据类型
    return torch.tensor(res, device=device, dtype=dtype)


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    # 使用 types 模块的 FunctionType 创建一个新的函数 g，使用给定的参数来复制函数 f 的行为
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    # 使用 functools 模块的 update_wrapper 方法更新函数 g 的元数据，以便 g 看起来像函数 f
    g = functools.update_wrapper(g, f)
    # 将函数 f 的关键字参数默认值 __kwdefaults__ 复制给函数 g
    g.__kwdefaults__ = f.__kwdefaults__
    # 返回更新后的函数 g
    return g
# 根据给定的测试名列表，标记它们为预期失败。这在使用子类化通用测试类进行贫乏参数化测试时很有用。
def xfail_inherited_tests(tests):
    def deco(cls):
        # 遍历测试名列表
        for t in tests:
            # 使用 unittest.expectedFailure 装饰器标记为预期失败的测试方法，
            # 需要先复制原始函数以便进行修改
            setattr(cls, t, unittest.expectedFailure(copy_func(getattr(cls, t))))
        return cls
    return deco


# 在沙堡环境中，类似于 unittest.skipIf，但会将测试标记为“通过”，避免由于连续跳过测试而创建任务时的投诉。
def skip_but_pass_in_sandcastle_if(condition, reason):
    def decorator(func):
        if condition:
            if IS_SANDCASTLE:  # 检查是否在沙堡环境中
                @wraps(func)
                def wrapper(*args, **kwargs):
                    # 输出跳过信息到标准错误流
                    print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
                return wrapper
            else:
                # 在非沙堡环境中，设置函数属性指示跳过原因
                func.__unittest_skip__ = True
                func.__unittest_skip_why__ = reason

        return func

    return decorator


# 返回数据类型的漂亮名称（例如，torch.int64 -> int64）。
def dtype_name(dtype):
    return str(dtype).split('.')[1]


# 数据类型缩写字典
dtype_abbrs = {
    torch.bfloat16: 'bf16',
    torch.float64: 'f64',
    torch.float32: 'f32',
    torch.float16: 'f16',
    torch.complex32: 'c32',
    torch.complex64: 'c64',
    torch.complex128: 'c128',
    torch.int8: 'i8',
    torch.int16: 'i16',
    torch.int32: 'i32',
    torch.int64: 'i64',
    torch.bool: 'b8',
    torch.uint8: 'u8',
}


# 使用 functools.lru_cache 装饰器，测量并返回 torch.cuda._sleep 函数每毫秒的大致循环数。
@functools.lru_cache
def get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep
    """

    def measure() -> float:
        # 创建启用计时的 CUDA 事件
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # 调用 torch.cuda._sleep 模拟延迟
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        # 计算每毫秒的循环数
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # 获取多次测量值并排除最大和最小值后的平均值，以避免系统干扰影响结果。
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])


# OpInfo 工具函数

T = TypeVar('T')
# 返回 Iterable 中的第一个样本
def first_sample(self: unittest.TestCase, samples: Iterable[T]) -> T:
    """
    # 从一个样本的可迭代对象中返回第一个样本，例如 OpInfo 返回的对象。
    # 如果没有样本可用，则跳过测试。
    try:
        # 返回可迭代对象 samples 的下一个元素作为结果
        return next(iter(samples))
    except StopIteration as e:
        # 如果迭代器已经耗尽，抛出 unittest.SkipTest 异常，并附加原始 StopIteration 异常信息
        raise unittest.SkipTest('Skipped! Need at least 1 sample input') from e
# 定义一个辅助方法，用于递归地克隆 OpInfo 测试过的操作符的张量类型输入
def clone_input_helper(input):
    if isinstance(input, torch.Tensor):
        # 如果输入是张量，则使用 torch.clone 方法进行克隆
        return torch.clone(input)

    if isinstance(input, Sequence):
        # 如果输入是序列，则递归地对每个元素应用 clone_input_helper 方法，返回元组
        return tuple(map(clone_input_helper, input))

    # 其他情况直接返回输入
    return input

# 定义一个上下文管理器/装饰器，用于测试自定义操作符的 ONNX 导出
@contextmanager
def custom_op(opname, symbolic_fn, opset_version):
    """Context manager/decorator to test ONNX export with custom operator"""
    try:
        # 注册自定义操作符的符号函数和操作集版本号
        register_custom_op_symbolic(opname, symbolic_fn, opset_version)
        # 执行 yield 之前的代码块
        yield
    finally:
        # 在退出上下文管理器后，注销自定义操作符的符号函数
        unregister_custom_op_symbolic(opname, opset_version)

# 定义一个函数，计算给定函数对输入的输出和梯度
def outs_and_grads(fn, graph_inps, inps):
    # 调用给定函数计算输出
    outs = fn(*graph_inps)
    # 对输出中所有叶子节点进行迭代
    for out in pytree.tree_leaves(outs):
        if isinstance(out, torch.Tensor) and out.requires_grad:
            # 如果输出是张量并且需要梯度，则计算其梯度并保留计算图
            out.sum().backward(retain_graph=True)
    # 收集所有输入张量的梯度
    grads = [inp.grad for inp in pytree.tree_leaves(inps) if isinstance(inp, torch.Tensor)]
    # 将所有输入张量的梯度设为 None
    for inp in pytree.tree_leaves(inps):
        if isinstance(inp, torch.Tensor):
            inp.grad = None
    # 返回计算得到的输出和梯度
    return outs, grads

# 定义一个函数，比较两个模型在给定输入下的输出和梯度是否相等
def compare_equal_outs_and_grads(test, m1, m2, inps):
    # 分别计算两个模型在给定输入下的输出和梯度
    r1, g1 = outs_and_grads(m1, inps, inps)
    r2, g2 = outs_and_grads(m2, inps, inps)
    # 使用单元测试框架的 assertEqual 方法比较输出和梯度是否相等
    test.assertEqual(r1, r2)
    test.assertEqual(g1, g2)

# 定义一个测试类，用于测试梯度相关功能
class TestGradients(TestCase):
    exact_dtype = True

    # 获取安全的 inplace 操作函数，用于避免 inplace 修改叶子节点的梯度需求
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    # 梯度测试的辅助方法，用于检查各种梯度相关功能
    def _grad_test_helper(self, device, dtype, op, variant, *, check_forward_ad=False, check_backward_ad=True,
                          check_batched_grad=None, check_batched_forward_grad=False):
        return self._check_helper(device, dtype, op, variant, 'gradcheck', check_forward_ad=check_forward_ad,
                                  check_backward_ad=check_backward_ad, check_batched_grad=check_batched_grad,
                                  check_batched_forward_grad=check_batched_forward_grad)

    # 跳过测试的辅助方法，用于检查是否应该跳过某些测试
    def _skip_helper(self, op, device, dtype):
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Op doesn't support autograd for this dtype.")
        if not op.supports_autograd and not op.supports_forward_ad:
            self.skipTest("Skipped! autograd not supported.")

# 定义一个函数，用于延迟初始化类的方法
def make_lazy_class(cls):

    def lazy_init(self, cb):
        self._cb = cb
        self._value = None

    # 动态地将 lazy_init 方法作为 __init__ 方法绑定到类上
    cls.__init__ = lazy_init

    # 遍历一组操作名，为每个操作名生成相应的方法并绑定到类上
    for basename in [
        "add", "sub", "mul", "truediv", "floordiv", "mod", "divmod", "pow",
        "lshift", "rshift", "and", "or", "xor", "neg", "pos", "abs", "invert",
        "eq", "ne", "lt", "le", "gt", "ge", "bool", "int", "index",
    ]:
        # 构造特殊名称，以双下划线包围基础名称
        name = f"__{basename}__"

        # 定义内部装饰器函数，接受名称参数
        def inner_wrapper(name):
            # 确定是否应该使用运算符而不是对象方法
            use_operator = basename not in ("bool", "int")

            # 定义包装函数，接受 self, *args, **kwargs 参数
            def wrapped(self, *args, **kwargs):
                # 如果存在回调函数 _cb，则调用它并更新 _value，然后置 _cb 为 None
                if self._cb is not None:
                    self._value = self._cb()
                    self._cb = None
                # 如果不应该使用运算符，则调用 self._value 对象的相应名称方法
                if not use_operator:
                    return getattr(self._value, name)(*args, **kwargs)
                # 否则，使用 operator 模块中的相应名称函数来操作 self._value 和传入参数
                else:
                    return getattr(operator, name)(self._value, *args, **kwargs)
            return wrapped

        # 将内部装饰器函数 inner_wrapper 绑定到类 cls 上的特殊名称属性
        setattr(cls, name, inner_wrapper(name))

    # 返回装饰后的类对象
    return cls
# 使用装饰器将 LazyVal 类转换为延迟加载类
@make_lazy_class
class LazyVal:
    pass


# 根据传入的异常对象 e 进行异常信息处理，并返回处理后的字符串结果
def munge_exc(e, *, suppress_suffix=True, suppress_prefix=True, file=None, skip=0):
    # 如果未指定 file 参数，则获取调用栈上一层的文件名，用于过滤异常信息中的堆栈帧
    if file is None:
        file = inspect.stack()[1 + skip].filename  # skip one frame

    # 将异常对象 e 转换为字符串
    s = str(e)

    # 定义替换函数 repl_frame，用于过滤掉异常信息中指定文件之外的堆栈帧信息
    def repl_frame(m):
        if m.group(1) != file:
            return ""
        if m.group(2) == "<module>":
            return ""
        return m.group(0)

    # 使用正则表达式替换，过滤异常信息中的堆栈帧信息
    s = re.sub(r'  File "([^"]+)", line \d+, in (.+)\n    .+\n( +[~^]+ *\n)?', repl_frame, s)
    s = re.sub(r"line \d+", "line N", s)  # 将行号替换为常规标记
    s = re.sub(r".py:\d+", ".py:N", s)  # 将文件名和行号替换为常规标记
    s = re.sub(file, os.path.basename(file), s)  # 替换文件路径为文件名
    s = re.sub(os.path.join(os.path.dirname(torch.__file__), ""), "", s)  # 替换掉 Torch 库路径
    s = re.sub(r"\\", "/", s)  # 将 Windows 风格的路径分隔符替换为 Unix 风格
    if suppress_suffix:
        # 如果需要，去除异常信息中的后缀内容
        s = re.sub(r"\n*Set TORCH_LOGS.+", "", s, flags=re.DOTALL)
        s = re.sub(r"\n*You can suppress this exception.+", "", s, flags=re.DOTALL)
    if suppress_prefix:
        # 如果需要，去除异常信息中的前缀内容
        s = re.sub(r"Cannot export model.+\n\n", "", s)
    s = re.sub(r" +$", "", s, flags=re.M)  # 去除每行末尾的空白字符
    return s  # 返回处理后的异常信息字符串
```