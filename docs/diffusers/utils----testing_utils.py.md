# `.\diffusers\utils\testing_utils.py`

```py
# 导入所需的标准库和第三方库
import functools  # 提供高阶函数的工具
import importlib  # 处理动态导入模块
import inspect  # 获取对象的内部信息
import io  # 提供用于处理流的基本工具
import logging  # 提供记录日志的功能
import multiprocessing  # 处理多进程的支持
import os  # 提供与操作系统交互的功能
import random  # 生成随机数的工具
import re  # 处理正则表达式
import struct  # 处理 C 语言结构体的工具
import sys  # 处理 Python 解释器的变量和函数
import tempfile  # 创建临时文件的工具
import time  # 提供时间相关的功能
import unittest  # 提供单元测试的框架
import urllib.parse  # 处理 URL 的解析和构造
from contextlib import contextmanager  # 提供上下文管理器的功能
from io import BytesIO, StringIO  # 提供内存中的字节流和字符串流
from pathlib import Path  # 处理文件路径的工具
from typing import Callable, Dict, List, Optional, Union  # 提供类型注释的工具

import numpy as np  # 导入 NumPy 库
import PIL.Image  # 导入 PIL 图像处理库
import PIL.ImageOps  # 导入 PIL 图像操作功能
import requests  # 处理 HTTP 请求的库
from numpy.linalg import norm  # 导入计算向量范数的函数
from packaging import version  # 处理版本比较的工具

# 导入自定义工具模块中的所需功能
from .import_utils import (
    BACKENDS_MAPPING,  # 后端映射的字典
    is_compel_available,  # 检查 Compel 是否可用的函数
    is_flax_available,  # 检查 Flax 是否可用的函数
    is_note_seq_available,  # 检查 NoteSeq 是否可用的函数
    is_onnx_available,  # 检查 ONNX 是否可用的函数
    is_opencv_available,  # 检查 OpenCV 是否可用的函数
    is_peft_available,  # 检查 PEFT 是否可用的函数
    is_timm_available,  # 检查 TIMM 是否可用的函数
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_torch_version,  # 检查 PyTorch 版本的函数
    is_torchsde_available,  # 检查 TorchSDE 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)
from .logging import get_logger  # 导入自定义日志记录功能

# 创建全局随机数生成器
global_rng = random.Random()

# 获取当前模块的日志记录器
logger = get_logger(__name__)

# 检查是否满足 PEFT 的版本要求
_required_peft_version = is_peft_available() and version.parse(
    version.parse(importlib.metadata.version("peft")).base_version
) > version.parse("0.5")  # 检查 PEFT 版本是否大于 0.5

# 检查是否满足 Transformers 的版本要求
_required_transformers_version = is_transformers_available() and version.parse(
    version.parse(importlib.metadata.version("transformers")).base_version
) > version.parse("4.33")  # 检查 Transformers 版本是否大于 4.33

# 根据版本要求确定是否使用 PEFT 后端
USE_PEFT_BACKEND = _required_peft_version and _required_transformers_version

# 如果 PyTorch 可用，执行以下代码
if is_torch_available():
    import torch  # 导入 PyTorch 库

    # 设置用于自定义加速器的后端环境变量
    if "DIFFUSERS_TEST_BACKEND" in os.environ:
        backend = os.environ["DIFFUSERS_TEST_BACKEND"]  # 获取环境变量中的后端名称
        try:
            _ = importlib.import_module(backend)  # 尝试导入指定的后端模块
        except ModuleNotFoundError as e:  # 捕获模块未找到异常
            raise ModuleNotFoundError(
                f"Failed to import `DIFFUSERS_TEST_BACKEND` '{backend}'! This should be the name of an installed module \
                    to enable a specified backend.):\n{e}"  # 提示用户导入失败
            ) from e  # 抛出原始异常

    # 如果指定了设备环境变量
    if "DIFFUSERS_TEST_DEVICE" in os.environ:
        torch_device = os.environ["DIFFUSERS_TEST_DEVICE"]  # 获取指定的设备名称
        try:
            # 尝试创建设备以验证提供的设备是否有效
            _ = torch.device(torch_device)
        except RuntimeError as e:  # 捕获运行时异常
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `DIFFUSERS_TEST_DEVICE`: {torch_device}"  # 提示用户设备未知
            ) from e  # 抛出原始异常
        logger.info(f"torch_device overrode to {torch_device}")  # 记录当前使用的设备名称
    else:
        # 默认情况下根据 CUDA 可用性选择设备
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        is_torch_higher_equal_than_1_12 = version.parse(
            version.parse(torch.__version__).base_version
        ) >= version.parse("1.12")  # 检查 PyTorch 版本是否大于等于 1.12

        if is_torch_higher_equal_than_1_12:  # 如果 PyTorch 版本符合条件
            # 某些版本的 PyTorch 1.12 未注册 mps 后端，需查看相关问题
            mps_backend_registered = hasattr(torch.backends, "mps")  # 检查是否注册了 mps 后端
            # 根据 mps 后端可用性选择设备
            torch_device = "mps" if (mps_backend_registered and torch.backends.mps.is_available()) else torch_device
# 定义一个函数，用于检查两个 PyTorch 张量是否相近
def torch_all_close(a, b, *args, **kwargs):
    # 检查是否安装了 PyTorch，如果没有，则引发异常
    if not is_torch_available():
        raise ValueError("PyTorch needs to be installed to use this function.")
    # 检查两个张量是否相近，如果不相近，则断言失败，并显示最大差异
    if not torch.allclose(a, b, *args, **kwargs):
        assert False, f"Max diff is absolute {(a - b).abs().max()}. Diff tensor is {(a - b).abs()}."
    # 如果相近，返回 True
    return True


# 定义一个函数，计算两个 NumPy 数组之间的余弦相似度距离
def numpy_cosine_similarity_distance(a, b):
    # 计算两个数组的余弦相似度
    similarity = np.dot(a, b) / (norm(a) * norm(b))
    # 计算相似度的距离，距离越小越相似
    distance = 1.0 - similarity.mean()

    # 返回计算得到的距离
    return distance


# 定义一个函数，用于打印张量测试结果
def print_tensor_test(
    tensor,
    limit_to_slices=None,
    max_torch_print=None,
    filename="test_corrections.txt",
    expected_tensor_name="expected_slice",
):
    # 如果设置了最大打印数量，则更新 PyTorch 打印选项
    if max_torch_print:
        torch.set_printoptions(threshold=10_000)

    # 获取当前测试的名称
    test_name = os.environ.get("PYTEST_CURRENT_TEST")
    # 如果输入不是张量，则将其转换为张量
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)
    # 如果设置了切片限制，则限制张量的切片
    if limit_to_slices:
        tensor = tensor[0, -3:, -3:, -1]

    # 将张量转换为字符串格式，并移除换行符
    tensor_str = str(tensor.detach().cpu().flatten().to(torch.float32)).replace("\n", "")
    # 将张量字符串格式化为 NumPy 数组形式
    output_str = tensor_str.replace("tensor", f"{expected_tensor_name} = np.array")
    # 分离测试名称为文件、类和函数名
    test_file, test_class, test_fn = test_name.split("::")
    test_fn = test_fn.split()[0]
    # 以附加模式打开文件并写入测试结果
    with open(filename, "a") as f:
        print("::".join([test_file, test_class, test_fn, output_str]), file=f)


# 定义一个函数，用于获取测试目录的路径
def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path
    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.
    """
    # 获取调用该函数的文件的路径
    caller__file__ = inspect.stack()[1][1]
    # 获取调用文件所在目录的绝对路径
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    # 循环查找直到找到以 "tests" 结尾的目录
    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    # 如果提供了附加路径，则返回合并后的完整路径
    if append_path:
        return Path(tests_dir, append_path).as_posix()
    else:
        # 否则返回测试目录路径
        return tests_dir


# 从 PR 中提取的函数
# https://github.com/huggingface/accelerate/pull/1964
def str_to_bool(value) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0). True values are `y`, `yes`, `t`, `true`,
    `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    # 将输入值转换为小写以进行比较
    value = value.lower()
    # 检查输入值是否是“真”值，返回 1
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    # 检查输入值是否是“假”值，返回 0
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        # 如果输入值无效，则引发异常
        raise ValueError(f"invalid truth value {value}")


# 定义一个函数，从环境变量中解析布尔标志
def parse_flag_from_env(key, default=False):
    try:
        # 尝试从环境变量中获取值
        value = os.environ[key]
    except KeyError:
        # 如果未设置环境变量，则使用默认值
        _value = default
    else:
        # 如果没有设置 KEY，进入此分支，准备将值转换为布尔值
        # 尝试将字符串值转换为布尔值（True 或 False）
        try:
            _value = str_to_bool(value)
        except ValueError:
            # 如果转换失败，抛出更具体的错误信息，提示值必须是 'yes' 或 'no'
            raise ValueError(f"If set, {key} must be yes or no.")
    # 返回转换后的布尔值
    return _value
# 从环境变量中解析是否运行慢测试的标志，默认为 False
_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
# 从环境变量中解析是否运行夜间测试的标志，默认为 False
_run_nightly_tests = parse_flag_from_env("RUN_NIGHTLY", default=False)
# 从环境变量中解析是否运行编译测试的标志，默认为 False
_run_compile_tests = parse_flag_from_env("RUN_COMPILE", default=False)


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """创建一个随机的 float32 张量"""
    # 如果没有提供随机数生成器，则使用全局随机数生成器
    if rng is None:
        rng = global_rng

    # 初始化总维度为 1
    total_dims = 1
    # 计算张量的总元素数量
    for dim in shape:
        total_dims *= dim

    # 初始化一个空列表以存储随机值
    values = []
    # 根据总元素数量生成随机值
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    # 将生成的值转换为张量，并按照指定形状调整，返回连续的张量
    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def slow(test_case):
    """
    装饰器标记一个测试为慢测试。

    慢测试默认被跳过。设置 RUN_SLOW 环境变量为真值以运行它们。
    """
    # 如果 _run_slow_tests 为真，则返回原测试，否则跳过测试并提示信息
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def nightly(test_case):
    """
    装饰器标记一个每晚在 diffusers CI 中运行的测试。

    慢测试默认被跳过。设置 RUN_NIGHTLY 环境变量为真值以运行它们。
    """
    # 如果 _run_nightly_tests 为真，则返回原测试，否则跳过测试并提示信息
    return unittest.skipUnless(_run_nightly_tests, "test is nightly")(test_case)


def is_torch_compile(test_case):
    """
    装饰器标记一个在 diffusers CI 中运行的编译测试。

    编译测试默认被跳过。设置 RUN_COMPILE 环境变量为真值以运行它们。
    """
    # 如果 _run_compile_tests 为真，则返回原测试，否则跳过测试并提示信息
    return unittest.skipUnless(_run_compile_tests, "test is torch compile")(test_case)


def require_torch(test_case):
    """
    装饰器标记一个需要 PyTorch 的测试。未安装 PyTorch 时这些测试将被跳过。
    """
    # 如果 PyTorch 可用，则返回原测试，否则跳过测试并提示信息
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def require_torch_2(test_case):
    """
    装饰器标记一个需要 PyTorch 2 的测试。这些测试在未安装 PyTorch 2 时将被跳过。
    """
    # 检查 PyTorch 是否可用且版本大于等于 2.0.0，然后返回原测试，否则跳过测试
    return unittest.skipUnless(is_torch_available() and is_torch_version(">=", "2.0.0"), "test requires PyTorch 2")(
        test_case
    )


def require_torch_gpu(test_case):
    """装饰器标记一个需要 CUDA 和 PyTorch 的测试。"""
    # 检查 PyTorch 是否可用且当前设备为 CUDA，然后返回原测试，否则跳过测试
    return unittest.skipUnless(is_torch_available() and torch_device == "cuda", "test requires PyTorch+CUDA")(
        test_case
    )


# 这些装饰器用于特定加速器行为，而不是仅限于 GPU
def require_torch_accelerator(test_case):
    """装饰器标记一个需要加速器后端和 PyTorch 的测试。"""
    # 检查 PyTorch 是否可用且当前设备不是 CPU，然后返回原测试，否则跳过测试
    return unittest.skipUnless(is_torch_available() and torch_device != "cpu", "test requires accelerator+PyTorch")(
        test_case
    )


def require_torch_multi_gpu(test_case):
    """
    装饰器标记一个需要多 GPU 设置的测试（在 PyTorch 中）。这些测试在没有多个 GPU 的机器上将被跳过。
    若要仅运行 multi_gpu 测试，假设所有测试名称包含 multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    # 检查是否可以使用 PyTorch
        if not is_torch_available():
            # 如果不可用，则跳过测试，给出提示
            return unittest.skip("test requires PyTorch")(test_case)
    
        # 导入 PyTorch 库
        import torch
    
        # 跳过测试，除非 GPU 设备数量大于 1
        return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)
# 装饰器，标记需要支持 FP16 数据类型的加速器的测试
def require_torch_accelerator_with_fp16(test_case):
    # 如果当前设备支持 FP16，则跳过此装饰器
    return unittest.skipUnless(_is_torch_fp16_available(torch_device), "test requires accelerator with fp16 support")(
        test_case
    )


# 装饰器，标记需要支持 FP64 数据类型的加速器的测试
def require_torch_accelerator_with_fp64(test_case):
    # 如果当前设备支持 FP64，则跳过此装饰器
    return unittest.skipUnless(_is_torch_fp64_available(torch_device), "test requires accelerator with fp64 support")(
        test_case
    )


# 装饰器，标记需要支持训练的加速器的测试
def require_torch_accelerator_with_training(test_case):
    # 如果当前设备可用且支持训练，则跳过此装饰器
    return unittest.skipUnless(
        is_torch_available() and backend_supports_training(torch_device),
        "test requires accelerator with training support",
    )(test_case)


# 装饰器，标记如果 torch_device 是 'mps' 则跳过测试
def skip_mps(test_case):
    # 如果当前设备不是 'mps'，则跳过此装饰器
    return unittest.skipUnless(torch_device != "mps", "test requires non 'mps' device")(test_case)


# 装饰器，标记需要 JAX 和 Flax 的测试
def require_flax(test_case):
    # 如果 Flax 可用，则跳过此装饰器
    return unittest.skipUnless(is_flax_available(), "test requires JAX & Flax")(test_case)


# 装饰器，标记需要 compel 库的测试
def require_compel(test_case):
    # 如果 compel 可用，则跳过此装饰器
    return unittest.skipUnless(is_compel_available(), "test requires compel")(test_case)


# 装饰器，标记需要 onnxruntime 的测试
def require_onnxruntime(test_case):
    # 如果 onnxruntime 可用，则跳过此装饰器
    return unittest.skipUnless(is_onnx_available(), "test requires onnxruntime")(test_case)


# 装饰器，标记需要 note_seq 的测试
def require_note_seq(test_case):
    # 如果 note_seq 可用，则跳过此装饰器
    return unittest.skipUnless(is_note_seq_available(), "test requires note_seq")(test_case)


# 装饰器，标记需要 torchsde 的测试
def require_torchsde(test_case):
    # 如果 torchsde 可用，则跳过此装饰器
    return unittest.skipUnless(is_torchsde_available(), "test requires torchsde")(test_case)


# 装饰器，标记需要 PEFT 后端的测试
def require_peft_backend(test_case):
    # 如果需要 PEFT 后端，则跳过此装饰器
    return unittest.skipUnless(USE_PEFT_BACKEND, "test requires PEFT backend")(test_case)


# 装饰器，标记需要 timm 库的测试
def require_timm(test_case):
    # 如果 timm 可用，则跳过此装饰器
    return unittest.skipUnless(is_timm_available(), "test requires timm")(test_case)


# 装饰器，标记需要特定版本 PEFT 的测试
def require_peft_version_greater(peft_version):
    # 装饰器标记一个测试，该测试要求特定版本的 PEFT 后端，需满足特定版本
        """
    
        # 定义装饰器函数，接受一个测试用例作为参数
        def decorator(test_case):
            # 检查 PEFT 是否可用，并且当前版本是否大于指定的 PEFT 版本
            correct_peft_version = is_peft_available() and version.parse(
                version.parse(importlib.metadata.version("peft")).base_version
            ) > version.parse(peft_version)
            # 如果满足版本要求，则跳过此测试，并提供相应的提示信息
            return unittest.skipUnless(
                correct_peft_version, f"test requires PEFT backend with the version greater than {peft_version}"
            )(test_case)
    
        # 返回装饰器函数
        return decorator
# 定义一个装饰器，要求加速版本大于给定版本
def require_accelerate_version_greater(accelerate_version):
    # 装饰器内部函数，用于装饰测试用例
    def decorator(test_case):
        # 检查 PEFT 是否可用，并解析当前加速库版本
        correct_accelerate_version = is_peft_available() and version.parse(
            version.parse(importlib.metadata.version("accelerate")).base_version
        ) > version.parse(accelerate_version)
        # 根据版本判断是否跳过测试用例
        return unittest.skipUnless(
            correct_accelerate_version, f"Test requires accelerate with the version greater than {accelerate_version}."
        )(test_case)

    return decorator


# 定义一个装饰器，标记将在 PEFT 后端之后跳过的测试
def deprecate_after_peft_backend(test_case):
    """
    装饰器，标记将在 PEFT 后端之后跳过的测试
    """
    # 根据是否使用 PEFT 后端决定是否跳过测试
    return unittest.skipUnless(not USE_PEFT_BACKEND, "test skipped in favor of PEFT backend")(test_case)


# 获取当前 Python 版本
def get_python_version():
    # 获取系统的版本信息
    sys_info = sys.version_info
    # 提取主版本和次版本
    major, minor = sys_info.major, sys_info.minor
    # 返回主版本和次版本
    return major, minor


# 加载 NumPy 数组，支持 URL 和本地路径
def load_numpy(arry: Union[str, np.ndarray], local_path: Optional[str] = None) -> np.ndarray:
    # 检查 arry 是否为字符串
    if isinstance(arry, str):
        if local_path is not None:
            # local_path 用于修正测试的图像路径
            return Path(local_path, arry.split("/")[-5], arry.split("/")[-2], arry.split("/")[-1]).as_posix()
        # 检查字符串是否为 URL
        elif arry.startswith("http://") or arry.startswith("https://"):
            response = requests.get(arry)  # 发送请求获取内容
            response.raise_for_status()  # 检查请求是否成功
            arry = np.load(BytesIO(response.content))  # 从响应内容加载 NumPy 数组
        # 检查字符串是否为有效的文件路径
        elif os.path.isfile(arry):
            arry = np.load(arry)  # 从文件路径加载 NumPy 数组
        else:
            # 抛出路径或 URL 不正确的错误
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {arry} is not a valid path"
            )
    # 检查 arry 是否为 NumPy 数组
    elif isinstance(arry, np.ndarray):
        pass  # 如果是 NumPy 数组则不做任何处理
    else:
        # 抛出格式不正确的错误
        raise ValueError(
            "Incorrect format used for numpy ndarray. Should be an url linking to an image, a local path, or a"
            " ndarray."
        )

    return arry  # 返回处理后的数组


# 从给定 URL 加载 PyTorch 张量
def load_pt(url: str):
    response = requests.get(url)  # 发送请求获取内容
    response.raise_for_status()  # 检查请求是否成功
    arry = torch.load(BytesIO(response.content))  # 从响应内容加载 PyTorch 张量
    return arry  # 返回加载的张量


# 加载图像，支持字符串和 PIL.Image.Image 类型
def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    将 `image` 加载为 PIL 图像。

    参数：
        image (`str` 或 `PIL.Image.Image`):
            要转换为 PIL 图像格式的图像。
    返回：
        `PIL.Image.Image`:
            一个 PIL 图像。
    """
    # 检查 image 是否为字符串
    if isinstance(image, str):
        # 检查字符串是否为 URL
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)  # 从 URL 加载图像
        # 检查字符串是否为有效的文件路径
        elif os.path.isfile(image):
            image = PIL.Image.open(image)  # 从文件路径加载图像
        else:
            # 抛出路径或 URL 不正确的错误
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    # 检查 image 是否为 PIL 图像
    elif isinstance(image, PIL.Image.Image):
        image = image  # 如果是 PIL 图像则不做任何处理
    # 如果不是正确的格式，则引发一个值错误
    else:
        # 提供详细的错误信息，说明接受的格式要求
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    # 对图像应用 EXIF 变换，调整其方向
    image = PIL.ImageOps.exif_transpose(image)
    # 将图像转换为 RGB 模式
    image = image.convert("RGB")
    # 返回处理后的图像
    return image
# 预处理输入图像以适应特定模型需求
def preprocess_image(image: PIL.Image, batch_size: int):
    # 获取图像的宽度和高度
    w, h = image.size
    # 调整宽度和高度为8的整数倍，以便于处理
    w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
    # 根据新的宽高调整图像大小，使用高质量重采样
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # 将图像数据转换为numpy数组并归一化到[0, 1]区间
    image = np.array(image).astype(np.float32) / 255.0
    # 扩展图像维度并复制以形成batch
    image = np.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
    # 将numpy数组转换为PyTorch张量
    image = torch.from_numpy(image)
    # 将图像值从[0, 1]范围缩放到[-1, 1]范围
    return 2.0 * image - 1.0


# 将图像序列导出为GIF文件
def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None) -> str:
    # 如果未指定输出路径，则创建临时GIF文件
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    # 保存第一帧，并附加后续帧
    image[0].save(
        output_gif_path,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=100,
        loop=0,
    )
    # 返回生成的GIF文件路径
    return output_gif_path


# 上下文管理器，用于缓冲写入文件
@contextmanager
def buffered_writer(raw_f):
    # 创建一个缓冲写入对象
    f = io.BufferedWriter(raw_f)
    # 暴露缓冲写入对象给上下文
    yield f
    # 刷新缓冲区，确保所有数据写入
    f.flush()


# 导出网格为PLY文件
def export_to_ply(mesh, output_ply_path: str = None):
    """
    写入一个网格的PLY文件。
    """
    # 如果未指定输出路径，则创建临时PLY文件
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

    # 获取顶点坐标并转移到CPU和NumPy数组
    coords = mesh.verts.detach().cpu().numpy()
    # 获取面索引并转移到CPU和NumPy数组
    faces = mesh.faces.cpu().numpy()
    # 获取RGB颜色通道数据并堆叠为RGB格式
    rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)

    # 使用缓冲写入器打开PLY文件进行写入
    with buffered_writer(open(output_ply_path, "wb")) as f:
        # 写入PLY文件头
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        # 如果存在RGB数据，写入颜色属性
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        # 如果存在面数据，写入面索引
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        # 写入文件头结束标志
        f.write(b"end_header\n")

        # 如果存在RGB数据，处理顶点数据
        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            # 定义数据打包格式
            format = struct.Struct("<3f3B")
            # 写入每个顶点的数据
            for item in vertices:
                f.write(format.pack(*item))
        else:
            # 定义顶点数据打包格式
            format = struct.Struct("<3f")
            # 写入每个顶点坐标
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        # 如果存在面数据，处理面索引数据
        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))

    # 返回生成的PLY文件路径
    return output_ply_path


# 导出网格为OBJ文件
def export_to_obj(mesh, output_obj_path: str = None):
    # 如果未指定输出路径，则创建临时OBJ文件
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

    # 获取顶点坐标并转移到CPU和NumPy数组
    verts = mesh.verts.detach().cpu().numpy()
    # 获取面索引并转移到CPU和NumPy数组
    faces = mesh.faces.cpu().numpy()
    # 将 mesh 中的顶点颜色通道提取并转换为 NumPy 数组，堆叠成一个数组
    vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)
    # 将顶点坐标和颜色组合成字符串列表，格式化为 OBJ 文件所需的顶点定义
    vertices = [
        "{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())
    ]

    # 将面数据格式化为 OBJ 文件所需的面定义，面索引从 0 转为从 1 开始
    faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]

    # 合并顶点数据和面数据，形成完整的 OBJ 数据列表
    combined_data = ["v " + vertex for vertex in vertices] + faces

    # 打开指定路径的文件，以写入模式创建文件对象 f
    with open(output_obj_path, "w") as f:
        # 将合并的数据写入文件，每个条目之间用换行符分隔
        f.writelines("\n".join(combined_data))
# 导出视频帧为视频文件，返回输出视频的路径
def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None) -> str:
    # 检查 OpenCV 是否可用
    if is_opencv_available():
        # 导入 OpenCV 库
        import cv2
    else:
        # 如果不可用，抛出导入错误
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    # 如果未指定输出视频路径，则创建临时文件
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    # 设置视频编码格式为 mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 获取第一帧的高度、宽度和通道数
    h, w, c = video_frames[0].shape
    # 创建视频写入对象
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8, frameSize=(w, h))
    # 遍历每一帧视频
    for i in range(len(video_frames)):
        # 将帧从 RGB 转换为 BGR 格式
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        # 将转换后的帧写入视频
        video_writer.write(img)
    # 返回输出视频的路径
    return output_video_path


# 加载 Hugging Face 上的 NumPy 数组
def load_hf_numpy(path) -> np.ndarray:
    # 定义基础 URL
    base_url = "https://huggingface.co/datasets/fusing/diffusers-testing/resolve/main"

    # 如果路径不是以 http 或 https 开头，则拼接基础 URL
    if not path.startswith("http://") and not path.startswith("https://"):
        path = os.path.join(base_url, urllib.parse.quote(path))

    # 加载并返回 NumPy 数组
    return load_numpy(path)


# --- pytest 配置函数 --- #

# 避免在测试中多次调用，确保只调用一次
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    该函数从 `conftest.py` 调用，使用 `pytest_addoption` 包装器。

    允许同时加载两个 `conftest.py` 文件，而不会因添加相同的 pytest 选项而失败。
    """
    option = "--make-reports"
    # 检查选项是否已注册
    if option not in pytest_opt_registered:
        # 添加选项到解析器
        parser.addoption(
            option,
            action="store",
            default=False,
            help="生成报告文件。此选项的值用作报告名称的前缀",
        )
        # 标记选项已注册
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    在测试套件运行结束时生成多个报告 - 每个报告存储在当前目录的独立文件中。
    报告文件以测试套件名称为前缀。

    模拟 --duration 和 -rA pytest 参数。

    从 `conftest.py` 调用此函数。
    
    Args:
    - tr: 从 `conftest.py` 传递的 `terminalreporter`
    - id: 像 `tests` 或 `examples` 的唯一 ID，将并入最终报告文件名
    """
    from _pytest.config import create_terminal_writer

    # 如果没有提供 ID，默认为 "tests"
    if not len(id):
        id = "tests"

    # 获取配置和原始终端写入器
    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    # 保存原始报告字符设置
        orig_reportchars = tr.reportchars
    
        # 定义报告文件存放目录
        dir = "reports"
        # 创建目录，父目录可选，如果已存在则不报错
        Path(dir).mkdir(parents=True, exist_ok=True)
        # 生成报告文件的字典，包含不同报告类型的文件名
        report_files = {
            k: f"{dir}/{id}_{k}.txt"
            for k in [
                "durations",
                "errors",
                "failures_long",
                "failures_short",
                "failures_line",
                "passes",
                "stats",
                "summary_short",
                "warnings",
            ]
        }
    
        # 自定义持续时间报告
        # 注意：无需调用 pytest --durations=XX 获取此单独报告
        # 来源于 https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
        dlist = []  # 初始化持续时间列表
        # 遍历测试结果中的统计数据
        for replist in tr.stats.values():
            for rep in replist:
                # 如果报告有持续时间属性，则添加到列表中
                if hasattr(rep, "duration"):
                    dlist.append(rep)
        # 如果持续时间列表非空
        if dlist:
            # 按持续时间降序排序
            dlist.sort(key=lambda x: x.duration, reverse=True)
            # 打开文件以写入持续时间报告
            with open(report_files["durations"], "w") as f:
                durations_min = 0.05  # 最小持续时间（秒）
                # 写入标题
                f.write("slowest durations\n")
                # 遍历报告并写入文件
                for i, rep in enumerate(dlist):
                    # 如果报告的持续时间小于最小值，写入省略信息并退出循环
                    if rep.duration < durations_min:
                        f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                        break
                    # 写入每条报告的持续时间、执行时机和节点ID
                    f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")
    
        # 定义简短失败摘要的函数
        def summary_failures_short(tr):
            # 假设报告使用了长格式，截取最后一帧
            reports = tr.getreports("failed")
            # 如果没有失败报告，则直接返回
            if not reports:
                return
            # 写入分隔符和标题
            tr.write_sep("=", "FAILURES SHORT STACK")
            # 遍历失败报告
            for rep in reports:
                # 获取失败消息
                msg = tr._getfailureheadline(rep)
                # 写入分隔符和消息
                tr.write_sep("_", msg, red=True, bold=True)
                # 去掉可选的前导额外帧，只保留最后一帧
                longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
                tr._tw.line(longrepr)
                # 注意：不打印任何 rep.sections 以保持报告简短
    
        # 使用现成的报告函数，劫持文件句柄以记录到专用文件
        # 来源于 https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
        # 注意：某些 pytest 插件可能会干扰默认的 `terminalreporter`（例如，pytest-instafail）
    
        # 以行/短/长样式报告失败
        config.option.tbstyle = "auto"  # 完整的回溯
        # 打开文件以写入长失败报告
        with open(report_files["failures_long"], "w") as f:
            # 创建终端写入器，并劫持到文件
            tr._tw = create_terminal_writer(config, f)
            # 写入失败摘要
            tr.summary_failures()
    
        # config.option.tbstyle = "short" # 短回溯
        # 打开文件以写入短失败报告
        with open(report_files["failures_short"], "w") as f:
            # 创建终端写入器，并劫持到文件
            tr._tw = create_terminal_writer(config, f)
            # 写入简短失败摘要
            summary_failures_short(tr)
    
        # 将回溯样式设置为每个错误一行
        config.option.tbstyle = "line"  # 一行每个错误
        # 打开文件以写入行失败报告
        with open(report_files["failures_line"], "w") as f:
            # 创建终端写入器，并劫持到文件
            tr._tw = create_terminal_writer(config, f)
            # 写入失败摘要
            tr.summary_failures()
    # 打开指定的错误报告文件以写入模式
    with open(report_files["errors"], "w") as f:
        # 创建终端写入器并将其赋值给 tr._tw
        tr._tw = create_terminal_writer(config, f)
        # 输出错误摘要
        tr.summary_errors()

    # 打开指定的警告报告文件以写入模式
    with open(report_files["warnings"], "w") as f:
        # 创建终端写入器并将其赋值给 tr._tw
        tr._tw = create_terminal_writer(config, f)
        # 输出常规警告摘要
        tr.summary_warnings()  # normal warnings
        # 输出最终警告摘要
        tr.summary_warnings()  # final warnings

    # 设置报告字符，用于模拟 -rA 选项（在 summary_passes() 和 short_test_summary() 中使用）
    tr.reportchars = "wPpsxXEf"  
    # 打开指定的通过报告文件以写入模式
    with open(report_files["passes"], "w") as f:
        # 创建终端写入器并将其赋值给 tr._tw
        tr._tw = create_terminal_writer(config, f)
        # 输出通过的摘要
        tr.summary_passes()

    # 打开指定的短摘要报告文件以写入模式
    with open(report_files["summary_short"], "w") as f:
        # 创建终端写入器并将其赋值给 tr._tw
        tr._tw = create_terminal_writer(config, f)
        # 输出短测试摘要
        tr.short_test_summary()

    # 打开指定的统计报告文件以写入模式
    with open(report_files["stats"], "w") as f:
        # 创建终端写入器并将其赋值给 tr._tw
        tr._tw = create_terminal_writer(config, f)
        # 输出统计摘要
        tr.summary_stats()

    # 恢复原始设置：
    # 恢复终端写入器为原始写入器
    tr._tw = orig_writer
    # 恢复报告字符为原始字符
    tr.reportchars = orig_reportchars
    # 恢复配置的跟踪样式为原始样式
    config.option.tbstyle = orig_tbstyle
# 从 GitHub 复制的代码，装饰器用于处理不稳定的测试
def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None, description: Optional[str] = None):
    """
    装饰不稳定的测试，失败时会重试。

    参数：
        max_attempts (`int`, *可选*, 默认值为 5):
            重新尝试不稳定测试的最大尝试次数。
        wait_before_retry (`float`, *可选*):
            如果提供，重试测试前将等待该秒数。
        description (`str`, *可选*):
            描述情况的字符串（什么/在哪里/为什么不稳定，链接到 GitHub 问题/PR 评论、错误等）。
    """

    # 装饰器函数，接收待装饰的测试函数
    def decorator(test_func_ref):
        # 包装函数，处理重试逻辑
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1  # 初始化重试计数

            # 在最大尝试次数内循环
            while retry_count < max_attempts:
                try:
                    # 尝试调用测试函数并返回结果
                    return test_func_ref(*args, **kwargs)

                except Exception as err:  # 捕获测试函数的异常
                    # 打印错误信息和当前重试次数
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    if wait_before_retry is not None:  # 如果提供了等待时间
                        time.sleep(wait_before_retry)  # 等待指定的秒数
                    retry_count += 1  # 增加重试计数

            # 达到最大重试次数后再次尝试调用测试函数并返回结果
            return test_func_ref(*args, **kwargs)

        return wrapper  # 返回包装后的函数

    return decorator  # 返回装饰器


# 从 GitHub 复制的代码，用于在子进程中运行测试
def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    在子进程中运行测试。这可以避免 (GPU) 内存问题。

    参数：
        test_case (`unittest.TestCase`):
            将运行 `target_func` 的测试用例。
        target_func (`Callable`):
            实现实际测试逻辑的函数。
        inputs (`dict`, *可选*, 默认值为 `None`):
            将通过（输入）队列传递给 `target_func` 的输入。
        timeout (`int`, *可选*, 默认值为 `None`):
            将传递给输入和输出队列的超时（以秒为单位）。如果未指定，将检查环境变量 `PYTEST_TIMEOUT`。如果仍为 `None`，其值将设置为 `600`。
    """
    if timeout is None:
        # 获取超时设置，如果环境变量未设置则默认为 600 秒
        timeout = int(os.environ.get("PYTEST_TIMEOUT", 600))

    start_methohd = "spawn"  # 定义子进程的启动方法为 "spawn"
    ctx = multiprocessing.get_context(start_methohd)  # 获取指定上下文的 multiprocessing

    input_queue = ctx.Queue(1)  # 创建一个输入队列，最大大小为 1
    output_queue = ctx.JoinableQueue(1)  # 创建一个可加入的输出队列，最大大小为 1

    # 我们不能将 `unittest.TestCase` 发送到子进程，否则会出现关于 pickle 的问题。
    input_queue.put(inputs, timeout=timeout)  # 将输入放入队列，指定超时时间

    # 创建并启动子进程，指定目标函数和参数
    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()  # 启动子进程
    # 如果不能及时从子进程获取输出，则终止子进程：否则，悬挂的子进程会阻塞
    # 测试以正确的方式退出
        try:
            # 从输出队列中获取结果，设置超时
            results = output_queue.get(timeout=timeout)
            # 标记任务为已完成
            output_queue.task_done()
        except Exception as e:
            # 处理异常，终止进程
            process.terminate()
            # 记录测试失败的原因
            test_case.fail(e)
        # 等待进程结束，设置超时
        process.join(timeout=timeout)
    
        # 检查结果中的错误信息是否存在
        if results["error"] is not None:
            # 记录测试失败的具体错误信息
            test_case.fail(f'{results["error"]}')
# 定义一个上下文管理器类，用于捕获日志流
class CaptureLogger:
    """
    参数:
    上下文管理器，用于捕获 `logging` 流
        logger: 'logging` 日志对象
    返回:
        捕获的输出可以通过 `self.out` 获取
    示例:
    ```python
    >>> from diffusers import logging
    >>> from diffusers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.py")
    >>> with CaptureLogger(logger) as cl:
    ...     logger.info(msg)
    >>> assert cl.out, msg + "\n"
    ```py
    """

    # 初始化 CaptureLogger 类，接受一个日志对象
    def __init__(self, logger):
        self.logger = logger  # 保存日志对象
        self.io = StringIO()  # 创建一个字符串流用于捕获日志
        self.sh = logging.StreamHandler(self.io)  # 创建流处理器以写入字符串流
        self.out = ""  # 初始化捕获的输出为空字符串

    # 进入上下文时添加日志处理器
    def __enter__(self):
        self.logger.addHandler(self.sh)  # 将处理器添加到日志对象
        return self  # 返回当前实例

    # 退出上下文时移除日志处理器并获取输出
    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)  # 移除处理器
        self.out = self.io.getvalue()  # 获取捕获的日志输出

    # 返回捕获的日志字符串的表示形式
    def __repr__(self):
        return f"captured: {self.out}\n"


# 启用全确定性以保证分布式训练中的可重现性
def enable_full_determinism():
    """
    帮助函数以保证分布式训练期间的可重现行为。请参见
    - https://pytorch.org/docs/stable/notes/randomness.html 以了解 PyTorch
    """
    # 启用 PyTorch 确定性模式。可能需要设置环境变量 'CUDA_LAUNCH_BLOCKING' 或 'CUBLAS_WORKSPACE_CONFIG'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 设置 CUDA 同步执行
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 配置 CUBLAS 工作区
    torch.use_deterministic_algorithms(True)  # 启用确定性算法

    # 启用 CUDNN 确定性模式
    torch.backends.cudnn.deterministic = True  # 设置 CUDNN 为确定性
    torch.backends.cudnn.benchmark = False  # 禁用基准优化
    torch.backends.cuda.matmul.allow_tf32 = False  # 禁用 TF32 精度

# 禁用全确定性以恢复非确定性行为
def disable_full_determinism():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 关闭 CUDA 同步执行
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""  # 清除 CUBLAS 工作区配置
    torch.use_deterministic_algorithms(False)  # 禁用确定性算法


# 检查给定设备上是否支持 FP16
def _is_torch_fp16_available(device):
    if not is_torch_available():  # 如果 PyTorch 不可用，返回 False
        return False

    import torch  # 导入 PyTorch

    device = torch.device(device)  # 将设备字符串转换为设备对象

    try:
        x = torch.zeros((2, 2), dtype=torch.float16).to(device)  # 创建 FP16 张量并移动到指定设备
        _ = torch.mul(x, x)  # 进行乘法运算以检查支持情况
        return True  # 如果没有异常，返回 True

    except Exception as e:  # 捕获异常
        if device.type == "cuda":  # 如果设备类型为 cuda
            raise ValueError(
                f"You have passed a device of type 'cuda' which should work with 'fp16', but 'cuda' does not seem to be correctly installed on your machine: {e}"
            )  # 抛出错误，提示 cuda 安装问题

        return False  # 其他设备返回 False


# 检查给定设备上是否支持 FP64
def _is_torch_fp64_available(device):
    if not is_torch_available():  # 如果 PyTorch 不可用，返回 False
        return False

    import torch  # 导入 PyTorch

    device = torch.device(device)  # 将设备字符串转换为设备对象

    try:
        x = torch.zeros((2, 2), dtype=torch.float64).to(device)  # 创建 FP64 张量并移动到指定设备
        _ = torch.mul(x, x)  # 进行乘法运算以检查支持情况
        return True  # 如果没有异常，返回 True
    # 捕获异常并处理
        except Exception as e:
            # 检查设备类型是否为 'cuda'
            if device.type == "cuda":
                # 引发值错误，提示 'cuda' 应该支持 'fp64'，但似乎未正确安装
                raise ValueError(
                    f"You have passed a device of type 'cuda' which should work with 'fp64', but 'cuda' does not seem to be correctly installed on your machine: {e}"
                )
    
            # 返回 False，表示操作失败
            return False
# Guard these lookups for when Torch is not used - alternative accelerator support is for PyTorch
if is_torch_available():  # 检查 PyTorch 是否可用
    # Behaviour flags
    BACKEND_SUPPORTS_TRAINING = {"cuda": True, "cpu": True, "mps": False, "default": True}  # 定义支持训练的后端标志

    # Function definitions
    BACKEND_EMPTY_CACHE = {"cuda": torch.cuda.empty_cache, "cpu": None, "mps": None, "default": None}  # 定义后端清空缓存的函数
    BACKEND_DEVICE_COUNT = {"cuda": torch.cuda.device_count, "cpu": lambda: 0, "mps": lambda: 0, "default": 0}  # 定义获取设备数量的函数
    BACKEND_MANUAL_SEED = {"cuda": torch.cuda.manual_seed, "cpu": torch.manual_seed, "default": torch.manual_seed}  # 定义设置随机种子的函数


# This dispatches a defined function according to the accelerator from the function definitions.
def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):  # 根据设备和函数表调度相应的函数
    if device not in dispatch_table:  # 如果设备不在调度表中
        return dispatch_table["default"](*args, **kwargs)  # 返回默认函数的结果

    fn = dispatch_table[device]  # 获取对应设备的函数

    # Some device agnostic functions return values. Need to guard against 'None' instead at
    # user level
    if fn is None:  # 如果函数为 None
        return None  # 返回 None

    return fn(*args, **kwargs)  # 调用并返回函数的结果


# These are callables which automatically dispatch the function specific to the accelerator
def backend_manual_seed(device: str, seed: int):  # 设置特定设备的随机种子
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)  # 调用调度函数


def backend_empty_cache(device: str):  # 清空特定设备的缓存
    return _device_agnostic_dispatch(device, BACKEND_EMPTY_CACHE)  # 调用调度函数


def backend_device_count(device: str):  # 获取特定设备的数量
    return _device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)  # 调用调度函数


# These are callables which return boolean behaviour flags and can be used to specify some
# device agnostic alternative where the feature is unsupported.
def backend_supports_training(device: str):  # 检查特定设备是否支持训练
    if not is_torch_available():  # 如果 PyTorch 不可用
        return False  # 返回 False

    if device not in BACKEND_SUPPORTS_TRAINING:  # 如果设备不在支持训练的字典中
        device = "default"  # 使用默认设备

    return BACKEND_SUPPORTS_TRAINING[device]  # 返回该设备的支持训练标志


# Guard for when Torch is not available
if is_torch_available():  # 检查 PyTorch 是否可用
    # Update device function dict mapping
    def update_mapping_from_spec(device_fn_dict: Dict[str, Callable], attribute_name: str):  # 更新设备函数字典映射
        try:
            # Try to import the function directly
            spec_fn = getattr(device_spec_module, attribute_name)  # 尝试从模块中获取指定属性
            device_fn_dict[torch_device] = spec_fn  # 更新设备函数字典
        except AttributeError as e:  # 捕获属性错误
            # If the function doesn't exist, and there is no default, throw an error
            if "default" not in device_fn_dict:  # 如果字典中没有默认函数
                raise AttributeError(  # 抛出属性错误
                    f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found."
                ) from e  # 追踪原始错误
    # 检查环境变量中是否存在特定的设备规格
        if "DIFFUSERS_TEST_DEVICE_SPEC" in os.environ:
            # 获取设备规格文件的路径
            device_spec_path = os.environ["DIFFUSERS_TEST_DEVICE_SPEC"]
            # 检查指定的设备规格文件路径是否是一个文件
            if not Path(device_spec_path).is_file():
                # 如果文件不存在，抛出值错误并给出提示
                raise ValueError(f"Specified path to device specification file is not found. Received {device_spec_path}")
    
            try:
                # 从文件路径中提取模块名称（去掉.py后缀）
                import_name = device_spec_path[: device_spec_path.index(".py")]
            except ValueError as e:
                # 如果路径中没有找到.py，抛出值错误
                raise ValueError(f"Provided device spec file is not a Python file! Received {device_spec_path}") from e
    
            # 动态导入设备规格模块
            device_spec_module = importlib.import_module(import_name)
    
            try:
                # 从模块中获取设备名称
                device_name = device_spec_module.DEVICE_NAME
            except AttributeError:
                # 如果模块中没有DEVICE_NAME，抛出属性错误
                raise AttributeError("Device spec file did not contain `DEVICE_NAME`")
    
            # 检查环境变量中是否存在另一设备名称，并与当前设备名称进行比较
            if "DIFFUSERS_TEST_DEVICE" in os.environ and torch_device != device_name:
                # 如果不匹配，构建错误信息并抛出值错误
                msg = f"Mismatch between environment variable `DIFFUSERS_TEST_DEVICE` '{torch_device}' and device found in spec '{device_name}'\n"
                msg += "Either unset `DIFFUSERS_TEST_DEVICE` or ensure it matches device spec name."
                raise ValueError(msg)
    
            # 将torch_device设置为提取的设备名称
            torch_device = device_name
    
            # 对每个`BACKEND_*`字典添加一条条目
            update_mapping_from_spec(BACKEND_MANUAL_SEED, "MANUAL_SEED_FN")
            update_mapping_from_spec(BACKEND_EMPTY_CACHE, "EMPTY_CACHE_FN")
            update_mapping_from_spec(BACKEND_DEVICE_COUNT, "DEVICE_COUNT_FN")
            update_mapping_from_spec(BACKEND_SUPPORTS_TRAINING, "SUPPORTS_TRAINING")
```