# `.\diffusers\utils\import_utils.py`

```py
# 版权信息，标明此文件的版权所有者及相关许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 许可，使用该文件需遵循该许可
# 许可的使用条件；可以在以下地址获取许可
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，按照许可分发的软件以 "现状" 方式提供，
# 不附带任何形式的保证或条件，明示或暗示。
# 详见许可中对权限和限制的具体规定
"""
导入工具：与导入和懒初始化相关的工具函数
"""

# 导入模块，提供动态导入和模块相关功能
import importlib.util
# 导入操作符模块，方便使用比较操作符
import operator as op
# 导入操作系统模块，提供与操作系统交互的功能
import os
# 导入系统模块，提供对 Python 解释器的访问
import sys
# 从 collections 模块导入有序字典
from collections import OrderedDict
# 从 itertools 模块导入链式迭代工具
from itertools import chain
# 导入模块类型
from types import ModuleType
# 导入类型注释工具
from typing import Any, Union

# 从 huggingface_hub.utils 导入检查 Jinja 是否可用的工具
from huggingface_hub.utils import is_jinja_available  # noqa: F401
# 导入版本控制工具
from packaging import version
# 从 packaging.version 导入版本和解析函数
from packaging.version import Version, parse

# 导入当前目录下的 logging 模块
from . import logging

# 根据 Python 版本选择合适的 importlib_metadata 模块导入方式
if sys.version_info < (3, 8):
    # 如果 Python 版本低于 3.8，导入 importlib_metadata
    import importlib_metadata
else:
    # 如果 Python 版本为 3.8 或更高，导入 importlib.metadata
    import importlib.metadata as importlib_metadata

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义环境变量的真值集合
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
# 定义环境变量的真值和自动值集合
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

# 从环境变量中获取是否使用 TensorFlow，默认为 "AUTO"
USE_TF = os.environ.get("USE_TF", "AUTO").upper()
# 从环境变量中获取是否使用 PyTorch，默认为 "AUTO"
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
# 从环境变量中获取是否使用 JAX，默认为 "AUTO"
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()
# 从环境变量中获取是否使用 SafeTensors，默认为 "AUTO"
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()
# 从环境变量中获取是否进行慢导入，默认为 "FALSE"
DIFFUSERS_SLOW_IMPORT = os.environ.get("DIFFUSERS_SLOW_IMPORT", "FALSE").upper()
# 将慢导入的环境变量值转换为布尔值
DIFFUSERS_SLOW_IMPORT = DIFFUSERS_SLOW_IMPORT in ENV_VARS_TRUE_VALUES

# 定义操作符与对应函数的映射
STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

# 初始化 PyTorch 版本为 "N/A"
_torch_version = "N/A"
# 检查是否启用 PyTorch，并且 TensorFlow 未被启用
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    # 尝试查找 PyTorch 模块是否可用
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            # 获取 PyTorch 的版本信息
            _torch_version = importlib_metadata.version("torch")
            # 记录可用的 PyTorch 版本
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            # 如果 PyTorch 模块未找到，将其标记为不可用
            _torch_available = False
else:
    # 如果 USE_TORCH 被设置，则禁用 PyTorch
    logger.info("Disabling PyTorch because USE_TORCH is set")
    _torch_available = False

# 检查 PyTorch XLA 是否可用
_torch_xla_available = importlib.util.find_spec("torch_xla") is not None
if _torch_xla_available:
    try:
        # 获取 PyTorch XLA 的版本信息
        _torch_xla_version = importlib_metadata.version("torch_xla")
        # 记录可用的 PyTorch XLA 版本
        logger.info(f"PyTorch XLA version {_torch_xla_version} available.")
    except ImportError:
        # 如果 PyTorch XLA 导入失败，将其标记为不可用
        _torch_xla_available = False

# 检查 torch_npu 是否可用
_torch_npu_available = importlib.util.find_spec("torch_npu") is not None
if _torch_npu_available:
    # 尝试获取 "torch_npu" 包的版本信息
        try:
            # 使用 importlib_metadata 获取 "torch_npu" 的版本
            _torch_npu_version = importlib_metadata.version("torch_npu")
            # 记录可用的 "torch_npu" 版本信息到日志
            logger.info(f"torch_npu version {_torch_npu_version} available.")
        # 捕获导入错误，表示 "torch_npu" 包不可用
        except ImportError:
            # 设置标志，表示 "torch_npu" 不可用
            _torch_npu_available = False
# 初始化 JAX 版本为 "N/A"
_jax_version = "N/A"
# 初始化 Flax 版本为 "N/A"
_flax_version = "N/A"
# 检查 USE_JAX 是否在环境变量的真值或自动值列表中
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    # 检查 JAX 和 Flax 是否可用
    _flax_available = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
    # 如果 Flax 可用
    if _flax_available:
        try:
            # 获取 JAX 的版本
            _jax_version = importlib_metadata.version("jax")
            # 获取 Flax 的版本
            _flax_version = importlib_metadata.version("flax")
            # 记录 JAX 和 Flax 的版本信息
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        except importlib_metadata.PackageNotFoundError:
            # 如果找不到包，则设置 Flax 不可用
            _flax_available = False
else:
    # 如果 USE_JAX 不在环境变量的真值或自动值列表中，设置 Flax 不可用
    _flax_available = False

# 检查 USE_SAFETENSORS 是否在环境变量的真值或自动值列表中
if USE_SAFETENSORS in ENV_VARS_TRUE_AND_AUTO_VALUES:
    # 检查 safetensors 是否可用
    _safetensors_available = importlib.util.find_spec("safetensors") is not None
    # 如果 safetensors 可用
    if _safetensors_available:
        try:
            # 获取 safetensors 的版本
            _safetensors_version = importlib_metadata.version("safetensors")
            # 记录 safetensors 的版本信息
            logger.info(f"Safetensors version {_safetensors_version} available.")
        except importlib_metadata.PackageNotFoundError:
            # 如果找不到包，则设置 safetensors 不可用
            _safetensors_available = False
else:
    # 如果 USE_SAFETENSORS 不在环境变量的真值或自动值列表中，记录信息并设置 safetensors 不可用
    logger.info("Disabling Safetensors because USE_TF is set")
    _safetensors_available = False

# 检查 transformers 是否可用
_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    # 获取 transformers 的版本
    _transformers_version = importlib_metadata.version("transformers")
    # 记录 transformers 的版本信息
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果找不到包，则设置 transformers 不可用
    _transformers_available = False

# 检查 inflect 是否可用
_inflect_available = importlib.util.find_spec("inflect") is not None
try:
    # 获取 inflect 的版本
    _inflect_version = importlib_metadata.version("inflect")
    # 记录 inflect 的版本信息
    logger.debug(f"Successfully imported inflect version {_inflect_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果找不到包，则设置 inflect 不可用
    _inflect_available = False

# 初始化 onnxruntime 版本为 "N/A"
_onnxruntime_version = "N/A"
# 检查 onnxruntime 是否可用
_onnx_available = importlib.util.find_spec("onnxruntime") is not None
# 如果 onnxruntime 可用
if _onnx_available:
    # 可能的 onnxruntime 包候选列表
    candidates = (
        "onnxruntime",
        "onnxruntime-gpu",
        "ort_nightly_gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
        "ort_nightly_directml",
        "onnxruntime-rocm",
        "onnxruntime-training",
    )
    # 初始化 onnxruntime 版本为 None
    _onnxruntime_version = None
    # 对于元数据，我们需要查找 onnxruntime 和 onnxruntime-gpu
    for pkg in candidates:
        try:
            # 获取当前候选包的版本
            _onnxruntime_version = importlib_metadata.version(pkg)
            break  # 如果找到版本，跳出循环
        except importlib_metadata.PackageNotFoundError:
            pass  # 如果找不到包，继续下一个候选
    # 如果找到了 onnxruntime 版本，则设置可用状态
    _onnx_available = _onnxruntime_version is not None
    # 如果 onnxruntime 可用，记录其版本信息
    if _onnx_available:
        logger.debug(f"Successfully imported onnxruntime version {_onnxruntime_version}")
# (sayakpaul): importlib.util.find_spec("opencv-python") 返回 None 即使它已经被安装。
# 判断 OpenCV 是否可用，若未找到则返回 False
# _opencv_available = importlib.util.find_spec("opencv-python") is not None
try:
    # 定义可能的 OpenCV 包候选名称
    candidates = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    # 初始化 OpenCV 版本变量
    _opencv_version = None
    # 遍历候选包名称
    for pkg in candidates:
        try:
            # 尝试获取当前包的版本
            _opencv_version = importlib_metadata.version(pkg)
            # 成功获取版本后退出循环
            break
        except importlib_metadata.PackageNotFoundError:
            # 如果包未找到，则继续尝试下一个候选包
            pass
    # 检查是否成功获取到 OpenCV 版本
    _opencv_available = _opencv_version is not None
    # 如果 OpenCV 可用，记录调试信息
    if _opencv_available:
        logger.debug(f"Successfully imported cv2 version {_opencv_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果没有找到 OpenCV 包，标记为不可用
    _opencv_available = False

# 判断 SciPy 是否可用，若未找到则返回 False
_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    # 尝试获取 SciPy 包的版本
    _scipy_version = importlib_metadata.version("scipy")
    # 如果成功，记录调试信息
    logger.debug(f"Successfully imported scipy version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果包未找到，则标记为不可用
    _scipy_available = False

# 判断 librosa 是否可用，若未找到则返回 False
_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    # 尝试获取 librosa 包的版本
    _librosa_version = importlib_metadata.version("librosa")
    # 如果成功，记录调试信息
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果包未找到，则标记为不可用
    _librosa_available = False

# 判断 accelerate 是否可用，若未找到则返回 False
_accelerate_available = importlib.util.find_spec("accelerate") is not None
try:
    # 尝试获取 accelerate 包的版本
    _accelerate_version = importlib_metadata.version("accelerate")
    # 如果成功，记录调试信息
    logger.debug(f"Successfully imported accelerate version {_accelerate_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果包未找到，则标记为不可用
    _accelerate_available = False

# 判断 xformers 是否可用，若未找到则返回 False
_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    # 尝试获取 xformers 包的版本
    _xformers_version = importlib_metadata.version("xformers")
    # 如果 torch 可用，获取其版本并进行版本比较
    if _torch_available:
        _torch_version = importlib_metadata.version("torch")
        # 如果 torch 版本小于 1.12，抛出错误
        if version.Version(_torch_version) < version.Version("1.12"):
            raise ValueError("xformers is installed in your environment and requires PyTorch >= 1.12")

    # 如果成功，记录调试信息
    logger.debug(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果包未找到，则标记为不可用
    _xformers_available = False

# 判断 k_diffusion 是否可用，若未找到则返回 False
_k_diffusion_available = importlib.util.find_spec("k_diffusion") is not None
try:
    # 尝试获取 k_diffusion 包的版本
    _k_diffusion_version = importlib_metadata.version("k_diffusion")
    # 如果成功，记录调试信息
    logger.debug(f"Successfully imported k-diffusion version {_k_diffusion_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果包未找到，则标记为不可用
    _k_diffusion_available = False

# 判断 note_seq 是否可用，若未找到则返回 False
_note_seq_available = importlib.util.find_spec("note_seq") is not None
try:
    # 尝试获取 note_seq 包的版本
    _note_seq_version = importlib_metadata.version("note_seq")
    # 如果成功，记录调试信息
    logger.debug(f"Successfully imported note-seq version {_note_seq_version}")
except importlib_metadata.PackageNotFoundError:
    # 如果包未找到，则标记为不可用
    _note_seq_available = False

# 判断 wandb 是否可用，若未找到则返回 False
_wandb_available = importlib.util.find_spec("wandb") is not None
try:
    # 尝试获取 wandb 包的版本
    _wandb_version = importlib_metadata.version("wandb")
    # 记录调试信息，指示成功导入 wandb 的版本
    logger.debug(f"Successfully imported wandb version {_wandb_version }")
# 捕获导入错误，表示没有找到 'wandb' 包
except importlib_metadata.PackageNotFoundError:
    # 设置 'wandb' 不可用标志为 False
    _wandb_available = False

# 检查 'tensorboard' 包是否可用
_tensorboard_available = importlib.util.find_spec("tensorboard")
try:
    # 获取 'tensorboard' 的版本信息
    _tensorboard_version = importlib_metadata.version("tensorboard")
    # 记录成功导入 'tensorboard' 的版本
    logger.debug(f"Successfully imported tensorboard version {_tensorboard_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'tensorboard' 不可用标志为 False
    _tensorboard_available = False

# 检查 'compel' 包是否可用
_compel_available = importlib.util.find_spec("compel")
try:
    # 获取 'compel' 的版本信息
    _compel_version = importlib_metadata.version("compel")
    # 记录成功导入 'compel' 的版本
    logger.debug(f"Successfully imported compel version {_compel_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'compel' 不可用标志为 False
    _compel_available = False

# 检查 'ftfy' 包是否可用
_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    # 获取 'ftfy' 的版本信息
    _ftfy_version = importlib_metadata.version("ftfy")
    # 记录成功导入 'ftfy' 的版本
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'ftfy' 不可用标志为 False
    _ftfy_available = False

# 检查 'bs4' 包是否可用
_bs4_available = importlib.util.find_spec("bs4") is not None
try:
    # importlib metadata 以不同名称获取
    _bs4_version = importlib_metadata.version("beautifulsoup4")
    # 记录成功导入 'beautifulsoup4' 的版本
    logger.debug(f"Successfully imported ftfy version {_bs4_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'bs4' 不可用标志为 False
    _bs4_available = False

# 检查 'torchsde' 包是否可用
_torchsde_available = importlib.util.find_spec("torchsde") is not None
try:
    # 获取 'torchsde' 的版本信息
    _torchsde_version = importlib_metadata.version("torchsde")
    # 记录成功导入 'torchsde' 的版本
    logger.debug(f"Successfully imported torchsde version {_torchsde_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'torchsde' 不可用标志为 False
    _torchsde_available = False

# 检查 'imwatermark' 包是否可用
_invisible_watermark_available = importlib.util.find_spec("imwatermark") is not None
try:
    # 获取 'invisible-watermark' 的版本信息
    _invisible_watermark_version = importlib_metadata.version("invisible-watermark")
    # 记录成功导入 'invisible-watermark' 的版本
    logger.debug(f"Successfully imported invisible-watermark version {_invisible_watermark_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'invisible-watermark' 不可用标志为 False
    _invisible_watermark_available = False

# 检查 'peft' 包是否可用
_peft_available = importlib.util.find_spec("peft") is not None
try:
    # 获取 'peft' 的版本信息
    _peft_version = importlib_metadata.version("peft")
    # 记录成功导入 'peft' 的版本
    logger.debug(f"Successfully imported peft version {_peft_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'peft' 不可用标志为 False
    _peft_available = False

# 检查 'torchvision' 包是否可用
_torchvision_available = importlib.util.find_spec("torchvision") is not None
try:
    # 获取 'torchvision' 的版本信息
    _torchvision_version = importlib_metadata.version("torchvision")
    # 记录成功导入 'torchvision' 的版本
    logger.debug(f"Successfully imported torchvision version {_torchvision_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'torchvision' 不可用标志为 False
    _torchvision_available = False

# 检查 'sentencepiece' 包是否可用
_sentencepiece_available = importlib.util.find_spec("sentencepiece") is not None
try:
    # 获取 'sentencepiece' 的版本信息
    _sentencepiece_version = importlib_metadata.version("sentencepiece")
    # 记录成功导入 'sentencepiece' 的版本
    logger.info(f"Successfully imported sentencepiece version {_sentencepiece_version}")
except importlib_metadata.PackageNotFoundError:
    # 设置 'sentencepiece' 不可用标志为 False
    _sentencepiece_available = False

# 检查 'matplotlib' 包是否可用
_matplotlib_available = importlib.util.find_spec("matplotlib") is not None
try:
    # 导入 matplotlib 的版本信息
        _matplotlib_version = importlib_metadata.version("matplotlib")
        # 记录成功导入 matplotlib 的版本信息到调试日志
        logger.debug(f"Successfully imported matplotlib version {_matplotlib_version}")
# 捕获导入库时的包未找到异常
except importlib_metadata.PackageNotFoundError:
    # 如果未找到 matplotlib，设置其可用状态为 False
    _matplotlib_available = False

# 检查 "timm" 库是否可用，返回结果为布尔值
_timm_available = importlib.util.find_spec("timm") is not None
# 如果 "timm" 库可用
if _timm_available:
    try:
        # 获取 "timm" 库的版本信息
        _timm_version = importlib_metadata.version("timm")
        # 记录可用的 "timm" 版本信息
        logger.info(f"Timm version {_timm_version} available.")
    # 捕获导入库时的包未找到异常
    except importlib_metadata.PackageNotFoundError:
        # 如果未找到 "timm"，设置其可用状态为 False
        _timm_available = False

# 定义函数以返回 "timm" 库的可用状态
def is_timm_available():
    return _timm_available

# 检查 "bitsandbytes" 库是否可用，返回结果为布尔值
_bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
try:
    # 获取 "bitsandbytes" 库的版本信息
    _bitsandbytes_version = importlib_metadata.version("bitsandbytes")
    # 记录成功导入的 "bitsandbytes" 版本信息
    logger.debug(f"Successfully imported bitsandbytes version {_bitsandbytes_version}")
# 捕获导入库时的包未找到异常
except importlib_metadata.PackageNotFoundError:
    # 如果未找到 "bitsandbytes"，设置其可用状态为 False
    _bitsandbytes_available = False

# 检查当前是否在 Google Colab 环境中
_is_google_colab = "google.colab" in sys.modules or any(k.startswith("COLAB_") for k in os.environ)

# 检查 "imageio" 库是否可用，返回结果为布尔值
_imageio_available = importlib.util.find_spec("imageio") is not None
# 如果 "imageio" 库可用
if _imageio_available:
    try:
        # 获取 "imageio" 库的版本信息
        _imageio_version = importlib_metadata.version("imageio")
        # 记录成功导入的 "imageio" 版本信息
        logger.debug(f"Successfully imported imageio version {_imageio_version}")
    # 捕获导入库时的包未找到异常
    except importlib_metadata.PackageNotFoundError:
        # 如果未找到 "imageio"，设置其可用状态为 False
        _imageio_available = False

# 定义函数以返回 "torch" 库的可用状态
def is_torch_available():
    return _torch_available

# 定义函数以返回 "torch_xla" 库的可用状态
def is_torch_xla_available():
    return _torch_xla_available

# 定义函数以返回 "torch_npu" 库的可用状态
def is_torch_npu_available():
    return _torch_npu_available

# 定义函数以返回 "flax" 库的可用状态
def is_flax_available():
    return _flax_available

# 定义函数以返回 "transformers" 库的可用状态
def is_transformers_available():
    return _transformers_available

# 定义函数以返回 "inflect" 库的可用状态
def is_inflect_available():
    return _inflect_available

# 定义函数以返回 "unidecode" 库的可用状态
def is_unidecode_available():
    return _unidecode_available

# 定义函数以返回 "onnx" 库的可用状态
def is_onnx_available():
    return _onnx_available

# 定义函数以返回 "opencv" 库的可用状态
def is_opencv_available():
    return _opencv_available

# 定义函数以返回 "scipy" 库的可用状态
def is_scipy_available():
    return _scipy_available

# 定义函数以返回 "librosa" 库的可用状态
def is_librosa_available():
    return _librosa_available

# 定义函数以返回 "xformers" 库的可用状态
def is_xformers_available():
    return _xformers_available

# 定义函数以返回 "accelerate" 库的可用状态
def is_accelerate_available():
    return _accelerate_available

# 定义函数以返回 "k_diffusion" 库的可用状态
def is_k_diffusion_available():
    return _k_diffusion_available

# 定义函数以返回 "note_seq" 库的可用状态
def is_note_seq_available():
    return _note_seq_available

# 定义函数以返回 "wandb" 库的可用状态
def is_wandb_available():
    return _wandb_available

# 定义函数以返回 "tensorboard" 库的可用状态
def is_tensorboard_available():
    return _tensorboard_available

# 定义函数以返回 "compel" 库的可用状态
def is_compel_available():
    return _compel_available

# 定义函数以返回 "ftfy" 库的可用状态
def is_ftfy_available():
    return _ftfy_available

# 定义函数以返回 "bs4" 库的可用状态
def is_bs4_available():
    return _bs4_available

# 定义函数以返回 "torchsde" 库的可用状态
def is_torchsde_available():
    return _torchsde_available

# 定义函数以返回 "invisible_watermark" 库的可用状态
def is_invisible_watermark_available():
    return _invisible_watermark_available

# 定义函数以返回 "peft" 库的可用状态
def is_peft_available():
    return _peft_available

# 定义函数以返回 "torchvision" 库的可用状态
def is_torchvision_available():
    return _torchvision_available

# 定义函数以返回 "matplotlib" 库的可用状态
def is_matplotlib_available():
    return _matplotlib_available

# 定义函数以返回 "safetensors" 库的可用状态
def is_safetensors_available():
    return _safetensors_available

# 定义函数以返回 "bitsandbytes" 库的可用状态
def is_bitsandbytes_available():
    return _bitsandbytes_available

# 定义函数以返回是否在 Google Colab 环境中
def is_google_colab():
    return _is_google_colab

# 定义函数以返回 "sentencepiece" 库的可用状态
def is_sentencepiece_available():
    # 返回句子分割工具的可用性状态
        return _sentencepiece_available
# 定义一个函数来检查 imageio 库是否可用
def is_imageio_available():
    # 返回 _imageio_available 的值，表示 imageio 库的可用性
    return _imageio_available


# docstyle-ignore
# 定义一个字符串，提示用户缺少 FLAX 库并提供安装链接
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 inflect 库并提供安装命令
INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 PyTorch 库并提供安装链接
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 onnxruntime 库并提供安装命令
ONNX_IMPORT_ERROR = """
{0} requires the onnxruntime library but it was not found in your environment. You can install it with pip: `pip
install onnxruntime`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 OpenCV 库并提供安装命令
OPENCV_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with pip: `pip
install opencv-python`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 scipy 库并提供安装命令
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip: `pip install
scipy`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 librosa 库并提供安装链接
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library but it was not found in your environment.  Checkout the instructions on the
installation page: https://librosa.org/doc/latest/install.html and follow the ones that match your environment.
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 transformers 库并提供安装命令
TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 unidecode 库并提供安装命令
UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 k-diffusion 库并提供安装命令
K_DIFFUSION_IMPORT_ERROR = """
{0} requires the k-diffusion library but it was not found in your environment. You can install it with pip: `pip
install k-diffusion`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 note-seq 库并提供安装命令
NOTE_SEQ_IMPORT_ERROR = """
{0} requires the note-seq library but it was not found in your environment. You can install it with pip: `pip
install note-seq`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 wandb 库并提供安装命令
WANDB_IMPORT_ERROR = """
{0} requires the wandb library but it was not found in your environment. You can install it with pip: `pip
install wandb`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 tensorboard 库并提供安装命令
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
"""


# docstyle-ignore
# 定义一个字符串，提示用户缺少 compel 库并提供安装命令
COMPEL_IMPORT_ERROR = """
{0} requires the compel library but it was not found in your environment. You can install it with pip: `pip install compel`
"""

# docstyle-ignore
# 定义一个字符串，提示用户缺少 Beautiful Soup 库并提供安装命令
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 ftfy 库
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 torchsde 库
TORCHSDE_IMPORT_ERROR = """
{0} requires the torchsde library but it was not found in your environment. You can install it with pip: `pip install torchsde`
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 invisible-watermark 库
INVISIBLE_WATERMARK_IMPORT_ERROR = """
{0} requires the invisible-watermark library but it was not found in your environment. You can install it with pip: `pip install invisible-watermark>=0.2.0`
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 peft 库
PEFT_IMPORT_ERROR = """
{0} requires the peft library but it was not found in your environment. You can install it with pip: `pip install peft`
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 safetensors 库
SAFETENSORS_IMPORT_ERROR = """
{0} requires the safetensors library but it was not found in your environment. You can install it with pip: `pip install safetensors`
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 sentencepiece 库
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the sentencepiece library but it was not found in your environment. You can install it with pip: `pip install sentencepiece`
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 bitsandbytes 库
BITSANDBYTES_IMPORT_ERROR = """
{0} requires the bitsandbytes library but it was not found in your environment. You can install it with pip: `pip install bitsandbytes`
"""

# docstyle-ignore
# 定义一个错误消息模板，提示用户缺少 imageio 库和 ffmpeg
IMAGEIO_IMPORT_ERROR = """
{0} requires the imageio library and ffmpeg but it was not found in your environment. You can install it with pip: `pip install imageio imageio-ffmpeg`
"""

# 定义一个有序字典，用于映射后端
BACKENDS_MAPPING = OrderedDict(
    # 创建一个包含库名称及其可用性检查和导入错误的元组列表
        [
            # BS4 库的可用性检查及对应的导入错误信息
            ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
            # Flax 库的可用性检查及对应的导入错误信息
            ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
            # Inflect 库的可用性检查及对应的导入错误信息
            ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
            # ONNX 库的可用性检查及对应的导入错误信息
            ("onnx", (is_onnx_available, ONNX_IMPORT_ERROR)),
            # OpenCV 库的可用性检查及对应的导入错误信息
            ("opencv", (is_opencv_available, OPENCV_IMPORT_ERROR)),
            # SciPy 库的可用性检查及对应的导入错误信息
            ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
            # PyTorch 库的可用性检查及对应的导入错误信息
            ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
            # Transformers 库的可用性检查及对应的导入错误信息
            ("transformers", (is_transformers_available, TRANSFORMERS_IMPORT_ERROR)),
            # Unidecode 库的可用性检查及对应的导入错误信息
            ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
            # Librosa 库的可用性检查及对应的导入错误信息
            ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
            # K-Diffusion 库的可用性检查及对应的导入错误信息
            ("k_diffusion", (is_k_diffusion_available, K_DIFFUSION_IMPORT_ERROR)),
            # Note Seq 库的可用性检查及对应的导入错误信息
            ("note_seq", (is_note_seq_available, NOTE_SEQ_IMPORT_ERROR)),
            # WandB 库的可用性检查及对应的导入错误信息
            ("wandb", (is_wandb_available, WANDB_IMPORT_ERROR)),
            # TensorBoard 库的可用性检查及对应的导入错误信息
            ("tensorboard", (is_tensorboard_available, TENSORBOARD_IMPORT_ERROR)),
            # Compel 库的可用性检查及对应的导入错误信息
            ("compel", (is_compel_available, COMPEL_IMPORT_ERROR)),
            # FTFY 库的可用性检查及对应的导入错误信息
            ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
            # TorchSDE 库的可用性检查及对应的导入错误信息
            ("torchsde", (is_torchsde_available, TORCHSDE_IMPORT_ERROR)),
            # Invisible Watermark 库的可用性检查及对应的导入错误信息
            ("invisible_watermark", (is_invisible_watermark_available, INVISIBLE_WATERMARK_IMPORT_ERROR)),
            # PEFT 库的可用性检查及对应的导入错误信息
            ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
            # SafeTensors 库的可用性检查及对应的导入错误信息
            ("safetensors", (is_safetensors_available, SAFETENSORS_IMPORT_ERROR)),
            # BitsAndBytes 库的可用性检查及对应的导入错误信息
            ("bitsandbytes", (is_bitsandbytes_available, BITSANDBYTES_IMPORT_ERROR)),
            # SentencePiece 库的可用性检查及对应的导入错误信息
            ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
            # ImageIO 库的可用性检查及对应的导入错误信息
            ("imageio", (is_imageio_available, IMAGEIO_IMPORT_ERROR)),
        ]
# 定义一个装饰器函数，检查所需后端是否可用
def requires_backends(obj, backends):
    # 如果后端参数不是列表或元组，则将其转换为列表
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    # 获取对象的名称，如果对象有 __name__ 属性则使用该属性，否则使用类名
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    # 根据后端列表生成检查器元组
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    # 遍历检查器，生成未通过检查的错误信息列表
    failed = [msg.format(name) for available, msg in checks if not available()]
    # 如果有检查失败，则抛出导入错误，显示所有错误信息
    if failed:
        raise ImportError("".join(failed))

    # 如果对象名称在特定管道列表中，并且 transformers 版本小于 4.25.0，则抛出错误
    if name in [
        "VersatileDiffusionTextToImagePipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionDualGuidedPipeline",
        "StableDiffusionImageVariationPipeline",
        "UnCLIPPipeline",
    ] and is_transformers_version("<", "4.25.0"):
        raise ImportError(
            f"You need to install `transformers>=4.25` in order to use {name}: \n```\n pip install"
            " --upgrade transformers \n```py"
        )

    # 如果对象名称在另一个特定管道列表中，并且 transformers 版本小于 4.26.0，则抛出错误
    if name in ["StableDiffusionDepth2ImgPipeline", "StableDiffusionPix2PixZeroPipeline"] and is_transformers_version(
        "<", "4.26.0"
    ):
        raise ImportError(
            f"You need to install `transformers>=4.26` in order to use {name}: \n```\n pip install"
            " --upgrade transformers \n```py"
        )


# 定义一个元类，用于生成虚拟对象
class DummyObject(type):
    """
    Dummy 对象的元类。任何继承自它的类在用户尝试访问其任何方法时都会返回由
    `requires_backend` 生成的 ImportError。
    """

    # 重写 __getattr__ 方法，处理属性访问
    def __getattr__(cls, key):
        # 如果属性以 "_" 开头且不是特定的两个属性，则返回默认实现
        if key.startswith("_") and key not in ["_load_connected_pipes", "_is_onnx"]:
            return super().__getattr__(cls, key)
        # 检查类是否满足后端要求
        requires_backends(cls, cls._backends)


# 该函数用于比较库版本与要求版本的关系，来源于指定的 GitHub 链接
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    参数：
    比较库版本与某个要求之间的关系，使用给定的操作符。
        library_or_version (`str` 或 `packaging.version.Version`):
            要检查的库名或版本。
        operation (`str`):
            操作符的字符串表示，如 `">"` 或 `"<="`。
        requirement_version (`str`):
            要与库版本比较的版本
    """
    # 检查操作符是否在预定义的操作符字典中
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    # 获取对应的操作函数
    operation = STR_OPERATION_TO_FUNC[operation]
    # 如果传入的库或版本是字符串，则获取其版本信息
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    # 使用操作函数比较版本，并返回结果
    return operation(library_or_version, parse(requirement_version))


# 该函数用于检查 PyTorch 版本与给定要求的关系，来源于指定的 GitHub 链接
def is_torch_version(operation: str, version: str):
    """
    参数：
    # 比较当前 PyTorch 版本与给定参考版本及操作符
    # operation (`str`): 操作符的字符串表示，例如 `">"` 或 `"<="`
    # version (`str`): PyTorch 的字符串版本
    """
    # 返回解析后的当前 PyTorch 版本与指定版本的比较结果
    return compare_versions(parse(_torch_version), operation, version)
# 定义一个函数，用于比较当前 Transformers 版本与给定版本之间的关系
def is_transformers_version(operation: str, version: str):
    """
    Args:
    Compares the current Transformers version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    # 检查 Transformers 是否可用，如果不可用则返回 False
    if not _transformers_available:
        return False
    # 将当前 Transformers 版本与给定版本进行比较，并返回比较结果
    return compare_versions(parse(_transformers_version), operation, version)


# 定义一个函数，用于比较当前 Accelerate 版本与给定版本之间的关系
def is_accelerate_version(operation: str, version: str):
    """
    Args:
    Compares the current Accelerate version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    # 检查 Accelerate 是否可用，如果不可用则返回 False
    if not _accelerate_available:
        return False
    # 将当前 Accelerate 版本与给定版本进行比较，并返回比较结果
    return compare_versions(parse(_accelerate_version), operation, version)


# 定义一个函数，用于比较当前 PEFT 版本与给定版本之间的关系
def is_peft_version(operation: str, version: str):
    """
    Args:
    Compares the current PEFT version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    # 检查 PEFT 版本是否可用，如果不可用则返回 False
    if not _peft_version:
        return False
    # 将当前 PEFT 版本与给定版本进行比较，并返回比较结果
    return compare_versions(parse(_peft_version), operation, version)


# 定义一个函数，用于比较当前 k-diffusion 版本与给定版本之间的关系
def is_k_diffusion_version(operation: str, version: str):
    """
    Args:
    Compares the current k-diffusion version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    # 检查 k-diffusion 是否可用，如果不可用则返回 False
    if not _k_diffusion_available:
        return False
    # 将当前 k-diffusion 版本与给定版本进行比较，并返回比较结果
    return compare_versions(parse(_k_diffusion_version), operation, version)


# 定义一个函数，用于从指定模块中提取对象
def get_objects_from_module(module):
    """
    Args:
    Returns a dict of object names and values in a module, while skipping private/internal objects
        module (ModuleType):
            Module to extract the objects from.

    Returns:
        dict: Dictionary of object names and corresponding values
    """

    # 创建一个空字典，用于存储对象名称及其对应的值
    objects = {}
    # 遍历模块中的所有对象名称
    for name in dir(module):
        # 跳过以 "_" 开头的私有对象
        if name.startswith("_"):
            continue
        # 获取对象并将名称和值存入字典
        objects[name] = getattr(module, name)

    # 返回包含对象名称及值的字典
    return objects


# 定义一个异常类，用于表示可选依赖未在环境中找到的错误
class OptionalDependencyNotAvailable(BaseException):
    """An error indicating that an optional dependency of Diffusers was not found in the environment."""


# 定义一个懒加载模块类，只有在请求对象时才会执行相关导入
class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # 受到 optuna.integration._IntegrationModule 的启发
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    # 初始化方法，接收模块的名称、文件、导入结构及可选参数
        def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
            # 调用父类初始化方法
            super().__init__(name)
            # 存储导入结构中模块的集合
            self._modules = set(import_structure.keys())
            # 存储类与模块的映射字典
            self._class_to_module = {}
            # 遍历导入结构，将每个类与其对应模块关联
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # 为 IDE 的自动补全准备的属性
            self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
            # 记录模块文件的路径
            self.__file__ = module_file
            # 记录模块的规格
            self.__spec__ = module_spec
            # 存储模块路径
            self.__path__ = [os.path.dirname(module_file)]
            # 存储额外对象，如果没有则初始化为空字典
            self._objects = {} if extra_objects is None else extra_objects
            # 存储模块名称
            self._name = name
            # 存储导入结构
            self._import_structure = import_structure
    
        # 为 IDE 的自动补全准备的方法
        def __dir__(self):
            # 获取当前对象的所有属性
            result = super().__dir__()
            # 将未包含在当前属性中的__all__元素添加到结果中
            for attr in self.__all__:
                if attr not in result:
                    result.append(attr)
            # 返回包含所有属性的列表
            return result
    
        def __getattr__(self, name: str) -> Any:
            # 如果对象中包含该名称，返回对应对象
            if name in self._objects:
                return self._objects[name]
            # 如果名称在模块中，获取模块
            if name in self._modules:
                value = self._get_module(name)
            # 如果名称在类到模块的映射中，获取对应模块的属性
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            # 如果名称不存在，抛出属性错误
            else:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")
    
            # 将获取到的值设置为对象的属性
            setattr(self, name, value)
            # 返回值
            return value
    
        def _get_module(self, module_name: str):
            # 尝试导入指定模块
            try:
                return importlib.import_module("." + module_name, self.__name__)
            # 捕获异常，抛出运行时错误
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                    f" traceback):\n{e}"
                ) from e
    
        def __reduce__(self):
            # 返回对象的序列化信息
            return (self.__class__, (self._name, self.__file__, self._import_structure))
```