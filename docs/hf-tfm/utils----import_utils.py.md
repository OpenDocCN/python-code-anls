# `.\utils\import_utils.py`

```
# 导入模块：与导入和懒初始化相关的实用工具
import importlib.metadata  # 导入标准库中的 importlib.metadata 模块
import importlib.util  # 导入标准库中的 importlib.util 模块
import json  # 导入标准库中的 json 模块
import os  # 导入标准库中的 os 模块
import shutil  # 导入标准库中的 shutil 模块
import subprocess  # 导入标准库中的 subprocess 模块
import sys  # 导入标准库中的 sys 模块
import warnings  # 导入标准库中的 warnings 模块
from collections import OrderedDict  # 从标准库的 collections 模块中导入 OrderedDict 类
from functools import lru_cache  # 从标准库的 functools 模块中导入 lru_cache 装饰器
from itertools import chain  # 从标准库的 itertools 模块中导入 chain 函数
from types import ModuleType  # 从标准库的 types 模块中导入 ModuleType 类
from typing import Any, Tuple, Union  # 导入 typing 模块中的 Any、Tuple、Union 类型

from packaging import version  # 从 packaging 库中导入 version 模块

from . import logging  # 从当前包中导入 logging 模块


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
# 获取当前模块的 logger 实例，用于记录日志，名称为当前模块的名称
# pylint: disable=invalid-name 是禁止 pylint 检查器发出的无效名称警告


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    """
    检查指定的包是否可用，并返回其版本信息（如果指定）。

    Args:
        pkg_name (str): 要检查的包的名称。
        return_version (bool, optional): 是否返回包的版本信息。默认为 False。

    Returns:
        Union[Tuple[bool, str], bool]: 如果 return_version 为 True，则返回包的存在状态和版本信息的元组；
        否则，仅返回包的存在状态（布尔值）。

    Notes:
        如果包存在，则尝试获取其版本信息，如果无法获取则使用特定的后备方法。
        使用 logging 模块记录调试信息，包括检测到的包的版本信息。
    """
    # 检查包是否存在，并获取其版本信息以避免导入本地目录
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # 主要方法获取包的版本信息
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # 备用方法：仅针对 "torch" 和包含 "dev" 的版本
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # 检查版本信息中是否包含 "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # 如果无法导入包，则表示不可用
                    package_exists = False
            else:
                # 对于除了 "torch" 外的包，不尝试后备方法，直接设置为不可用
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
# 获取环境变量 USE_TF 的值，并转换为大写形式，如果未设置则默认为 "AUTO"
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

# 尝试通过设置该值为0，在安装了TorchXLA的环境中运行原生的PyTorch作业。
USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

# `transformers`需要`torch>=1.11`，但此变量对外公开，因此不能简单地删除它。
# 这是运行torch.fx特性和torch.onnx与字典输入所需的torch版本。
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")

ACCELERATE_MIN_VERSION = "0.21.0"
FSDP_MIN_VERSION = "1.12.0"

# 检查是否安装了accelerate包，并返回其是否可用及其版本号。
_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
# 检查是否安装了apex包。
_apex_available = _is_package_available("apex")
# 检查是否安装了aqlm包。
_aqlm_available = _is_package_available("aqlm")
# 检查是否安装了bitsandbytes包。
_bitsandbytes_available = _is_package_available("bitsandbytes")
# 检查是否安装了galore_torch包。
_galore_torch_available = _is_package_available("galore_torch")
# 检查是否安装了beautifulsoup4包（注意，使用的是find_spec函数，因为导入的名称与包名称不同）。
_bs4_available = importlib.util.find_spec("bs4") is not None
# 检查是否安装了coloredlogs包。
_coloredlogs_available = _is_package_available("coloredlogs")
# 检查是否安装了cv2（opencv-python-headless）包。
_cv2_available = importlib.util.find_spec("cv2") is not None
# 检查是否安装了datasets包。
_datasets_available = _is_package_available("datasets")
# 检查是否安装了decord包。
_decord_available = importlib.util.find_spec("decord") is not None
# 检查是否安装了detectron2包。
_detectron2_available = _is_package_available("detectron2")
# 检查是否安装了faiss或faiss-cpu包。
_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    # 尝试获取faiss包的版本信息。
    _faiss_version = importlib.metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib.metadata.PackageNotFoundError:
    try:
        # 如果faiss包未找到，则尝试获取faiss-cpu包的版本信息。
        _faiss_version = importlib.metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib.metadata.PackageNotFoundError:
        # 如果faiss和faiss-cpu包都未找到，则标记_faiss_available为False。
        _faiss_available = False
# 检查是否安装了ftfy包。
_ftfy_available = _is_package_available("ftfy")
# 检查是否安装了g2p_en包。
_g2p_en_available = _is_package_available("g2p_en")
# 检查是否安装了intel_extension_for_pytorch包，并返回其是否可用及其版本号。
_ipex_available, _ipex_version = _is_package_available("intel_extension_for_pytorch", return_version=True)
# 检查是否安装了jieba包。
_jieba_available = _is_package_available("jieba")
# 检查是否安装了jinja2包。
_jinja_available = _is_package_available("jinja2")
# 检查是否安装了kenlm包。
_kenlm_available = _is_package_available("kenlm")
# 检查是否安装了keras_nlp包。
_keras_nlp_available = _is_package_available("keras_nlp")
# 检查是否安装了Levenshtein包。
_levenshtein_available = _is_package_available("Levenshtein")
# 检查是否安装了librosa包。
_librosa_available = _is_package_available("librosa")
# 检查是否安装了natten包。
_natten_available = _is_package_available("natten")
# 检查是否安装了nltk包。
_nltk_available = _is_package_available("nltk")
# 检查是否安装了onnx包。
_onnx_available = _is_package_available("onnx")
# 检查是否安装了openai包。
_openai_available = _is_package_available("openai")
# 检查是否安装了optimum包。
_optimum_available = _is_package_available("optimum")
# 检查是否安装了auto_gptq包。
_auto_gptq_available = _is_package_available("auto_gptq")
# 检查是否安装了awq包。
# （注意，此处应有代码，但由于未找到正确的导入方式，省略了相关部分）
# 检查是否可以导入名为 "awq" 的模块
_auto_awq_available = importlib.util.find_spec("awq") is not None

# 检查名为 "quanto" 的包是否可用
_quanto_available = _is_package_available("quanto")

# 检查名为 "pandas" 的包是否可用
_pandas_available = _is_package_available("pandas")

# 检查名为 "peft" 的包是否可用
_peft_available = _is_package_available("peft")

# 检查名为 "phonemizer" 的包是否可用
_phonemizer_available = _is_package_available("phonemizer")

# 检查名为 "psutil" 的包是否可用
_psutil_available = _is_package_available("psutil")

# 检查名为 "py3nvml" 的包是否可用
_py3nvml_available = _is_package_available("py3nvml")

# 检查名为 "pyctcdecode" 的包是否可用
_pyctcdecode_available = _is_package_available("pyctcdecode")

# 检查名为 "pytesseract" 的包是否可用
_pytesseract_available = _is_package_available("pytesseract")

# 检查名为 "pytest" 的包是否可用
_pytest_available = _is_package_available("pytest")

# 检查名为 "pytorch_quantization" 的包是否可用
_pytorch_quantization_available = _is_package_available("pytorch_quantization")

# 检查名为 "rjieba" 的包是否可用
_rjieba_available = _is_package_available("rjieba")

# 检查名为 "sacremoses" 的包是否可用
_sacremoses_available = _is_package_available("sacremoses")

# 检查名为 "safetensors" 的包是否可用
_safetensors_available = _is_package_available("safetensors")

# 检查名为 "scipy" 的包是否可用
_scipy_available = _is_package_available("scipy")

# 检查名为 "sentencepiece" 的包是否可用
_sentencepiece_available = _is_package_available("sentencepiece")

# 检查名为 "seqio" 的包是否可用
_is_seqio_available = _is_package_available("seqio")

# 检查是否可以导入名为 "sklearn" 的模块
_sklearn_available = importlib.util.find_spec("sklearn") is not None
if _sklearn_available:
    try:
        # 尝试获取 "scikit-learn" 的版本信息
        importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        # 如果找不到 "scikit-learn" 包，将 _sklearn_available 设为 False
        _sklearn_available = False

# 检查是否可以导入名为 "smdistributed" 的模块
_smdistributed_available = importlib.util.find_spec("smdistributed") is not None

# 检查名为 "soundfile" 的包是否可用
_soundfile_available = _is_package_available("soundfile")

# 检查名为 "spacy" 的包是否可用
_spacy_available = _is_package_available("spacy")

# 检查名为 "sudachipy" 的包是否可用，并获取其版本信息
_sudachipy_available, _sudachipy_version = _is_package_available("sudachipy", return_version=True)

# 检查名为 "tensorflow_probability" 的包是否可用
_tensorflow_probability_available = _is_package_available("tensorflow_probability")

# 检查名为 "tensorflow_text" 的包是否可用
_tensorflow_text_available = _is_package_available("tensorflow_text")

# 检查名为 "tf2onnx" 的包是否可用
_tf2onnx_available = _is_package_available("tf2onnx")

# 检查名为 "timm" 的包是否可用
_timm_available = _is_package_available("timm")

# 检查名为 "tokenizers" 的包是否可用
_tokenizers_available = _is_package_available("tokenizers")

# 检查名为 "torchaudio" 的包是否可用
_torchaudio_available = _is_package_available("torchaudio")

# 检查名为 "torchdistx" 的包是否可用
_torchdistx_available = _is_package_available("torchdistx")

# 检查名为 "torchvision" 的包是否可用
_torchvision_available = _is_package_available("torchvision")

# 检查名为 "mlx" 的包是否可用
_mlx_available = _is_package_available("mlx")

# 初始化 _torch_version 变量为 "N/A"，_torch_available 变量为 False
_torch_version = "N/A"
_torch_available = False

# 如果 USE_TORCH 在 ENV_VARS_TRUE_AND_AUTO_VALUES 中且 USE_TF 不在 ENV_VARS_TRUE_VALUES 中
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    # 尝试获取 "torch" 包的版本信息，并设置 _torch_available 为 True
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
else:
    # 记录信息表明禁用 PyTorch 因为 USE_TF 已设置
    logger.info("Disabling PyTorch because USE_TF is set")
    # 设置 _torch_available 为 False
    _torch_available = False

# 初始化 _tf_version 变量为 "N/A"，_tf_available 变量为 False
_tf_version = "N/A"
_tf_available = False

# 如果 FORCE_TF_AVAILABLE 在 ENV_VARS_TRUE_VALUES 中
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    # 设置 _tf_available 为 True
    _tf_available = True
else:
    # 检查环境变量中是否启用了 TensorFlow，并且未启用 Torch
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        # 注意：_is_package_available("tensorflow") 对 tensorflow-cpu 会失败，请测试下面的代码行
        # 在使用 tensorflow-cpu 时确保它仍然有效！

        # 检查是否可以导入 tensorflow 库
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
            # 可选的 TensorFlow 包列表
            candidates = (
                "tensorflow",
                "tensorflow-cpu",
                "tensorflow-gpu",
                "tf-nightly",
                "tf-nightly-cpu",
                "tf-nightly-gpu",
                "tf-nightly-rocm",
                "intel-tensorflow",
                "intel-tensorflow-avx512",
                "tensorflow-rocm",
                "tensorflow-macos",
                "tensorflow-aarch64",
            )
            _tf_version = None
            # 在候选包列表中查找 TensorFlow 的版本信息
            for pkg in candidates:
                try:
                    _tf_version = importlib.metadata.version(pkg)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
            # 更新 _tf_available 状态为找到的 TensorFlow 版本是否非空
            _tf_available = _tf_version is not None

        if _tf_available:
            # 如果找到 TensorFlow 并且版本小于 2，则记录警告信息并将 _tf_available 置为 False
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(
                    f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum."
                )
                _tf_available = False
    else:
        # 如果 USE_TORCH 已设置，则记录禁用 TensorFlow 的信息
        logger.info("Disabling Tensorflow because USE_TORCH is set")
# 检查是否安装了 Essentia 库
_essentia_available = importlib.util.find_spec("essentia") is not None
try:
    # 获取 Essentia 库的版本信息
    _essentia_version = importlib.metadata.version("essentia")
    logger.debug(f"Successfully imported essentia version {_essentia_version}")
except importlib.metadata.PackageNotFoundError:
    # 如果 Essentia 库未找到，则标记为不可用
    _essentia_version = False


# 检查是否安装了 Pretty MIDI 库
_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None
try:
    # 获取 Pretty MIDI 库的版本信息
    _pretty_midi_version = importlib.metadata.version("pretty_midi")
    logger.debug(f"Successfully imported pretty_midi version {_pretty_midi_version}")
except importlib.metadata.PackageNotFoundError:
    # 如果 Pretty MIDI 库未找到，则标记为不可用
    _pretty_midi_available = False


# 初始化 CCL 版本信息，默认为 "N/A"，检查是否安装了 CCL 相关库
ccl_version = "N/A"
_is_ccl_available = (
    importlib.util.find_spec("torch_ccl") is not None
    or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
)
try:
    # 获取 oneccl_bind_pt 库的版本信息
    ccl_version = importlib.metadata.version("oneccl_bind_pt")
    logger.debug(f"Detected oneccl_bind_pt version {ccl_version}")
except importlib.metadata.PackageNotFoundError:
    # 如果 oneccl_bind_pt 库未找到，则标记 CCL 不可用
    _is_ccl_available = False


# 初始化 Flax 是否可用，默认为 False
_flax_available = False
# 如果使用 JAX 环境变量指定为 True
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    # 检查 Flax 包是否可用，并获取其版本信息
    _flax_available, _flax_version = _is_package_available("flax", return_version=True)
    if _flax_available:
        # 如果 Flax 可用，则检查 JAX 包是否也可用，并获取其版本信息
        _jax_available, _jax_version = _is_package_available("jax", return_version=True)
        if _jax_available:
            # 如果 JAX 可用，则记录日志显示 JAX 和 Flax 的版本信息
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        else:
            # 如果 JAX 不可用，则将 Flax 和 JAX 的可用性都设为 False，并将版本信息置为 "N/A"
            _flax_available = _jax_available = False
            _jax_version = _flax_version = "N/A"


# 初始化 Torch FX 是否可用，默认为 False
_torch_fx_available = False
# 如果 Torch 可用
if _torch_available:
    # 解析 Torch 版本信息
    torch_version = version.parse(_torch_version)
    # 检查 Torch FX 是否可用，需满足指定的最低版本要求
    _torch_fx_available = (torch_version.major, torch_version.minor) >= (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )


# 初始化 Torch XLA 是否可用，默认为 False
_torch_xla_available = False
# 如果使用 Torch XLA 环境变量指定为 True
if USE_TORCH_XLA in ENV_VARS_TRUE_VALUES:
    # 检查 Torch XLA 包是否可用，并获取其版本信息
    _torch_xla_available, _torch_xla_version = _is_package_available("torch_xla", return_version=True)
    if _torch_xla_available:
        # 如果 Torch XLA 可用，则记录日志显示 Torch XLA 的版本信息
        logger.info(f"Torch XLA version {_torch_xla_version} available.")


# 返回 KenLM 库是否可用的函数
def is_kenlm_available():
    return _kenlm_available


# 返回 OpenCV 库是否可用的函数
def is_cv2_available():
    return _cv2_available


# 返回 Torch 库是否可用的函数
def is_torch_available():
    return _torch_available


# 返回当前使用的 Torch 版本信息的函数
def get_torch_version():
    return _torch_version


# 检查是否安装了 Torch SDPA 库
def is_torch_sdpa_available():
    # 如果 Torch 不可用，则 SDPA 也不可用
    if not is_torch_available():
        return False
    # 如果 Torch 版本信息为 "N/A"，则 SDPA 也不可用
    elif _torch_version == "N/A":
        return False

    # 笔记: 我们要求 torch>=2.1（而不是torch>=2.0）以在 Transformers 中使用 SDPA 有两个原因：
    # - 允许全局使用在 https://github.com/pytorch/pytorch/pull/95259 中引入的 `scale` 参数
    # - 内存高效的注意力支持任意的 attention_mask: https://github.com/pytorch/pytorch/pull/104310
    # 笔记: 我们要求 torch>=2.1.1 以避免 SDPA 在非连续输入中出现的数值问题：https://github.com/pytorch/pytorch/issues/112577
    return version.parse(_torch_version) >= version.parse("2.1.1")


# 返回 Torch Vision 库是否可用的函数
def is_torchvision_available():
    return _torchvision_available


# 返回变量 _torchvision_available 的值作为函数的返回结果
# 检查是否 galore_torch 可用，返回对应的状态
def is_galore_torch_available():
    return _galore_torch_available


# 检查是否 pyctcdecode 可用，返回对应的状态
def is_pyctcdecode_available():
    return _pyctcdecode_available


# 检查是否 librosa 可用，返回对应的状态
def is_librosa_available():
    return _librosa_available


# 检查是否 essentia 可用，返回对应的状态
def is_essentia_available():
    return _essentia_available


# 检查是否 pretty_midi 可用，返回对应的状态
def is_pretty_midi_available():
    return _pretty_midi_available


# 检查是否 torch 可用，并且 CUDA 是否可用
def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


# 检查是否 torch 可用，并且检查是否 mamba_ssm 包可用
def is_mamba_ssm_available():
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        else:
            return _is_package_available("mamba_ssm")
    return False


# 检查是否 torch 可用，并且检查是否 causal_conv1d 包可用
def is_causal_conv1d_available():
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        return _is_package_available("causal_conv1d")
    return False


# 检查是否 torch 可用，并且检查是否 torch.backends.mps 可用
def is_torch_mps_available():
    if is_torch_available():
        import torch

        if hasattr(torch.backends, "mps"):
            return torch.backends.mps.is_available()
    return False


# 检查是否 torch 可用，并且检查是否 CUDA BF16 支持
def is_torch_bf16_gpu_available():
    if not is_torch_available():
        return False

    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# 检查是否 torch 可用，并且检查是否 CPU BF16 支持
def is_torch_bf16_cpu_available():
    if not is_torch_available():
        return False

    import torch

    try:
        _ = torch.cpu.amp.autocast
    except AttributeError:
        return False

    return True


# 检查是否 torch 可用，并且检查是否 GPU 或 CPU 上 BF16 支持
def is_torch_bf16_available():
    warnings.warn(
        "The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available "
        "or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu",
        FutureWarning,
    )
    return is_torch_bf16_gpu_available()


# 使用 lru_cache 修饰器，检查在指定设备上是否 torch 的 FP16 可用
def is_torch_fp16_available_on_device(device):
    if not is_torch_available():
        return False

    import torch

    try:
        # 创建一个小张量，并执行矩阵乘法操作以检查 FP16 支持
        x = torch.zeros(2, 2, dtype=torch.float16).to(device)
        _ = x @ x

        # 检查在设备上是否支持 LayerNorm 操作，因为许多模型使用此操作
        batch, sentence_length, embedding_dim = 3, 4, 5
        embedding = torch.randn(batch, sentence_length, embedding_dim, dtype=torch.float16, device=device)
        layer_norm = torch.nn.LayerNorm(embedding_dim, dtype=torch.float16, device=device)
        _ = layer_norm(embedding)

    except:  # 捕获所有异常，返回 False
        return False

    return True
# 使用 LRU 缓存装饰器缓存函数结果，避免重复计算
@lru_cache()
# 检查指定设备上是否可用 Torch 的 BF16 支持
def is_torch_bf16_available_on_device(device):
    # 如果 Torch 不可用，则返回 False
    if not is_torch_available():
        return False

    # 导入 Torch 库
    import torch

    # 如果设备是 "cuda"，则检查 GPU 上是否可用 BF16 支持
    if device == "cuda":
        return is_torch_bf16_gpu_available()

    # 尝试在指定设备上创建一个 bfloat16 类型的张量并执行矩阵乘法操作
    try:
        x = torch.zeros(2, 2, dtype=torch.bfloat16).to(device)
        _ = x @ x
    except:  # noqa: E722
        # 捕获所有异常，通常返回 RuntimeError，但不保证
        # TODO: 如果可能的话，进行更精确的异常匹配
        return False

    # 如果以上尝试成功，则返回 True，表示 BF16 在该设备上可用
    return True


# 检查当前环境是否支持 Torch 的 TF32 支持
def is_torch_tf32_available():
    # 如果 Torch 不可用，则返回 False
    if not is_torch_available():
        return False

    # 导入 Torch 库
    import torch

    # 如果 CUDA 不可用或者 CUDA 版本为 None，则返回 False
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    # 如果 CUDA 设备的主版本号小于 8，则返回 False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    # 如果 CUDA 版本的主版本号小于 11，则返回 False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    # 如果 Torch 版本小于 1.7，则返回 False
    if version.parse(version.parse(torch.__version__).base_version) < version.parse("1.7"):
        return False

    # 如果以上条件都满足，则返回 True，表示 TF32 在当前环境中可用
    return True


# 返回 Torch FX 是否可用的标志
def is_torch_fx_available():
    return _torch_fx_available


# 返回 PEFT 是否可用的标志
def is_peft_available():
    return _peft_available


# 返回 Beautiful Soup (bs4) 是否可用的标志
def is_bs4_available():
    return _bs4_available


# 返回 TensorFlow 是否可用的标志
def is_tf_available():
    return _tf_available


# 返回 coloredlogs 是否可用的标志
def is_coloredlogs_available():
    return _coloredlogs_available


# 返回 TF2ONNX 是否可用的标志
def is_tf2onnx_available():
    return _tf2onnx_available


# 返回 ONNX 是否可用的标志
def is_onnx_available():
    return _onnx_available


# 返回 OpenAI 的库是否可用的标志
def is_openai_available():
    return _openai_available


# 返回 Flax 是否可用的标志
def is_flax_available():
    return _flax_available


# 返回 ftfy 是否可用的标志
def is_ftfy_available():
    return _ftfy_available


# 返回 g2p_en 是否可用的标志
def is_g2p_en_available():
    return _g2p_en_available


# 使用 LRU 缓存装饰器缓存函数结果，避免重复计算
@lru_cache()
# 检查是否 Torch TPU 可用（即是否安装了 torch_xla 并且环境中存在 TPU）
def is_torch_tpu_available(check_device=True):
    # 发出警告，提示函数即将被弃用
    warnings.warn(
        "`is_torch_tpu_available` is deprecated and will be removed in 4.41.0. "
        "Please use the `is_torch_xla_available` instead.",
        FutureWarning,
    )

    # 如果 Torch 不可用，则返回 False
    if not _torch_available:
        return False
    # 如果安装了 torch_xla，则进一步检查是否存在 TPU 设备
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # 需要检查是否可以找到 `xla_device`，如果找不到将引发 RuntimeError
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


# 使用 LRU 缓存装饰器缓存函数结果，避免重复计算
@lru_cache
# 检查 Torch XLA 是否可用
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    # 断言 `check_is_tpu` 和 `check_is_gpu` 不能同时为 True
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    # 如果 Torch XLA 不可用，则返回 False
    if not _torch_xla_available:
        return False

    # 导入 torch_xla 库
    import torch_xla

    # 如果需要检查 GPU，则返回当前设备类型是否为 GPU 或 CUDA
    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    # 如果检测到是TPU设备，则返回是否为TPU
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"
    # 否则返回True
    return True
# 使用 lru_cache 装饰器，缓存函数调用结果，提升性能
@lru_cache()
# 检查是否存在 torch_neuronx 模块，若存在则调用 is_torch_xla_available 函数
def is_torch_neuroncore_available(check_device=True):
    if importlib.util.find_spec("torch_neuronx") is not None:
        return is_torch_xla_available()
    return False


# 使用 lru_cache 装饰器，缓存函数调用结果，提升性能
@lru_cache()
# 检查是否安装了 torch_npu 模块，并可选地检查环境中是否存在 NPU 设备
def is_torch_npu_available(check_device=False):
    # 如果 _torch_available 为 False 或者找不到 torch_npu 模块，则返回 False
    if not _torch_available or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # 如果没有找到 NPU 设备会抛出 RuntimeError
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    # 检查 torch 是否有 npu 属性并且 NPU 可用
    return hasattr(torch, "npu") and torch.npu.is_available()


# 检查是否存在 torch _dynamo 模块以判断是否可用
def is_torchdynamo_available():
    if not is_torch_available():  # 如果 torch 不可用，则返回 False
        return False
    try:
        import torch._dynamo as dynamo  # noqa: F401

        return True
    except Exception:
        return False


# 检查是否存在 torch.compile 属性来判断是否可用
def is_torch_compile_available():
    if not is_torch_available():  # 如果 torch 不可用，则返回 False
        return False

    import torch

    # 不进行版本检查以支持夜间版本标记为 1.14。最终需要与 2.0 版本进行检查，但暂时不处理
    return hasattr(torch, "compile")


# 检查是否在编译 torch _dynamo 模块
def is_torchdynamo_compiling():
    if not is_torch_available():  # 如果 torch 不可用，则返回 False
        return False
    try:
        import torch._dynamo as dynamo  # noqa: F401

        return dynamo.is_compiling()
    except Exception:
        return False


# 检查是否安装了 torch_tensorrt 模块，并且是否存在 torch_tensorrt.fx 子模块
def is_torch_tensorrt_fx_available():
    if importlib.util.find_spec("torch_tensorrt") is None:  # 如果找不到 torch_tensorrt 模块，则返回 False
        return False
    return importlib.util.find_spec("torch_tensorrt.fx") is not None  # 检查是否存在 torch_tensorrt.fx 子模块


# 返回 _datasets_available 变量的值
def is_datasets_available():
    return _datasets_available


# 返回 _detectron2_available 变量的值
def is_detectron2_available():
    return _detectron2_available


# 返回 _rjieba_available 变量的值
def is_rjieba_available():
    return _rjieba_available


# 返回 _psutil_available 变量的值
def is_psutil_available():
    return _psutil_available


# 返回 _py3nvml_available 变量的值
def is_py3nvml_available():
    return _py3nvml_available


# 返回 _sacremoses_available 变量的值
def is_sacremoses_available():
    return _sacremoses_available


# 返回 _apex_available 变量的值
def is_apex_available():
    return _apex_available


# 返回 _aqlm_available 变量的值
def is_aqlm_available():
    return _aqlm_available


# 检查系统是否安装了 ninja 构建系统
def is_ninja_available():
    r"""
    Code comes from *torch.utils.cpp_extension.is_ninja_available()*. Returns `True` if the
    [ninja](https://ninja-build.org/) build system is available on the system, `False` otherwise.
    """
    try:
        subprocess.check_output("ninja --version".split())  # 执行命令检查 ninja 版本
    except Exception:
        return False  # 捕获异常则返回 False
    else:
        return True  # 执行成功则返回 True


# 检查是否安装了 ipex 模块以及 torch 可用性和 _ipex_available 变量
def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    if not is_torch_available() or not _ipex_available:
        return False  # 如果 torch 不可用或者 _ipex_available 为 False，则返回 False

    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    # 检查当前安装的 PyTorch 主版本和次版本是否与 Intel Extension for PyTorch 所需版本匹配
    if torch_major_and_minor != ipex_major_and_minor:
        # 如果不匹配，记录警告信息，提示用户切换到匹配的 PyTorch 版本后重新运行
        logger.warning(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        # 返回 False 表示版本不匹配
        return False
    # 如果版本匹配，返回 True
    return True
# 使用 lru_cache 装饰器来缓存函数的结果，提升函数性能
@lru_cache
# 检查是否安装了 intel_extension_for_pytorch 并且可能存在 XPU 设备
def is_torch_xpu_available(check_device=False):
    if not is_ipex_available():  # 如果没有安装 intel_extension_for_pytorch，则返回 False
        return False

    import intel_extension_for_pytorch  # 引入 intel_extension_for_pytorch 模块，用于检查是否安装
    import torch  # 引入 torch 模块

    if check_device:
        try:
            # 尝试获取 XPU 设备的数量，如果没有 XPU 设备会抛出 RuntimeError
            _ = torch.xpu.device_count()
            # 返回当前是否有可用的 XPU 设备
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    # 检查是否存在 torch.xpu 模块，并且该模块当前是否可用
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_bitsandbytes_available():
    if not is_torch_available():  # 如果 torch 不可用，则返回 False
        return False

    # bitsandbytes 在没有 cuda 可用时会抛出错误，这里添加简单检查避免异常
    import torch  # 引入 torch 模块

    return _bitsandbytes_available and torch.cuda.is_available()  # 返回 bitsandbytes 是否可用以及当前是否有 cuda 可用


def is_flash_attn_2_available():
    if not is_torch_available():  # 如果 torch 不可用，则返回 False
        return False

    if not _is_package_available("flash_attn"):  # 如果 flash_attn 包不可用，则返回 False
        return False

    import torch  # 引入 torch 模块

    if not torch.cuda.is_available():  # 如果没有 cuda 可用，则返回 False
        return False

    if torch.version.cuda:  # 如果是 CUDA 版本
        # 检查 flash_attn 包的版本是否大于等于 2.1.0
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    elif torch.version.hip:  # 如果是 HIP 版本
        # TODO: 一旦在 https://github.com/ROCmSoftwarePlatform/flash-attention 发布，将要求将要求版本提升至 2.1.0
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    else:
        return False


def is_flash_attn_greater_or_equal_2_10():
    if not _is_package_available("flash_attn"):  # 如果 flash_attn 包不可用，则返回 False
        return False

    # 检查 flash_attn 包的版本是否大于等于 2.1.0
    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


def is_torchdistx_available():
    return _torchdistx_available  # 返回 _torchdistx_available 变量的值


def is_faiss_available():
    return _faiss_available  # 返回 _faiss_available 变量的值


def is_scipy_available():
    return _scipy_available  # 返回 _scipy_available 变量的值


def is_sklearn_available():
    return _sklearn_available  # 返回 _sklearn_available 变量的值


def is_sentencepiece_available():
    return _sentencepiece_available  # 返回 _sentencepiece_available 变量的值


def is_seqio_available():
    return _is_seqio_available  # 返回 _is_seqio_available 变量的值


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:  # 如果找不到 google 模块，则返回 False
        return False
    # 检查是否找到 google.protobuf 模块，并返回结果
    return importlib.util.find_spec("google.protobuf") is not None


def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION):
    if min_version is not None:
        # 检查 _accelerate_available 变量的值，并且检查其版本是否大于等于 min_version
        return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)
    return _accelerate_available  # 返回 _accelerate_available 变量的值


def is_fsdp_available(min_version: str = FSDP_MIN_VERSION):
    if is_torch_available():  # 如果 torch 可用
        # 检查 _torch_version 的版本是否大于等于 min_version
        return version.parse(_torch_version) >= version.parse(min_version)
    return False  # 如果 torch 不可用，则返回 False


def is_optimum_available():
    return _optimum_available  # 返回 _optimum_available 变量的值


def is_auto_awq_available():
    return _auto_awq_available  # 返回 _auto_awq_available 变量的值


def is_quanto_available():
    return _quanto_available  # 返回 _quanto_available 变量的值


def is_auto_gptq_available():
    return _auto_gptq_available  # 返回 _auto_gptq_available 变量的值


def is_levenshtein_available():
    # 此函数未实现，没有返回值
    return _levenshtein_available


    # 返回变量 _levenshtein_available 的值作为函数的返回结果
# 检查是否已经安装了 optimum.neuron 包并且 _optimum_available 变量为真
def is_optimum_neuron_available():
    return _optimum_available and _is_package_available("optimum.neuron")


# 返回 _safetensors_available 变量的值
def is_safetensors_available():
    return _safetensors_available


# 返回 _tokenizers_available 变量的值
def is_tokenizers_available():
    return _tokenizers_available


# 使用 lru_cache 装饰器缓存函数结果，检查 PIL 库是否可用
@lru_cache
def is_vision_available():
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version("Pillow")
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version("Pillow-SIMD")
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


# 返回 _pytesseract_available 变量的值
def is_pytesseract_available():
    return _pytesseract_available


# 返回 _pytest_available 变量的值
def is_pytest_available():
    return _pytest_available


# 返回 _spacy_available 变量的值
def is_spacy_available():
    return _spacy_available


# 返回 is_tf_available() 和 _tensorflow_text_available 变量的逻辑与结果
def is_tensorflow_text_available():
    return is_tf_available() and _tensorflow_text_available


# 返回 is_tensorflow_text_available() 和 _keras_nlp_available 变量的逻辑与结果
def is_keras_nlp_available():
    return is_tensorflow_text_available() and _keras_nlp_available


# 在 Notebook 环境中检查 IPython 模块的存在
def is_in_notebook():
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            raise ImportError("databricks")
        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


# 返回 _pytorch_quantization_available 变量的值
def is_pytorch_quantization_available():
    return _pytorch_quantization_available


# 返回 _tensorflow_probability_available 变量的值
def is_tensorflow_probability_available():
    return _tensorflow_probability_available


# 返回 _pandas_available 变量的值
def is_pandas_available():
    return _pandas_available


# 检查 SageMaker 是否启用了分布式数据并行 (Distributed Data Parallel, DDP)
# 通过解析环境变量 SM_FRAMEWORK_PARAMS 检查 sagemaker_distributed_dataparallel_enabled 字段
def is_sagemaker_dp_enabled():
    sagemaker_params = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        sagemaker_params = json.loads(sagemaker_params)
        if not sagemaker_params.get("sagemaker_distributed_dataparallel_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    return _smdistributed_available


# 获取 SageMaker 的 MP 参数变量 SM_HP_MP_PARAMETERS
def is_sagemaker_mp_enabled():
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # 尝试解析 smp_options 变量，并检查是否包含 "partitions" 字段，这是模型并行所需的。
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        # 解析失败或格式错误，返回 False
        return False

    # 从 mpi_options 变量中获取 SageMaker 特定的框架参数。
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # 尝试解析 mpi_options 变量，并检查是否包含 "sagemaker_mpi_enabled" 字段。
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        # 解析失败或格式错误，返回 False
        return False
    
    # 最后，检查是否存在 `smdistributed` 模块。
    return _smdistributed_available
# 检查当前运行环境是否为 SageMaker 环境，通过检查环境变量中是否存在 "SAGEMAKER_JOB_NAME"
def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


# 返回一个布尔值，指示是否安装了 soundfile 库
def is_soundfile_availble():
    return _soundfile_available


# 返回一个布尔值，指示是否安装了 timm 库
def is_timm_available():
    return _timm_available


# 返回一个布尔值，指示是否安装了 natten 库
def is_natten_available():
    return _natten_available


# 返回一个布尔值，指示是否安装了 nltk 库
def is_nltk_available():
    return _nltk_available


# 返回一个布尔值，指示是否安装了 torchaudio 库
def is_torchaudio_available():
    return _torchaudio_available


# 返回一个布尔值，指示是否安装了与语音处理相关的库，目前依赖于 torchaudio
def is_speech_available():
    return _torchaudio_available


# 返回一个布尔值，指示是否安装了 phonemizer 库
def is_phonemizer_available():
    return _phonemizer_available


# 返回一个装饰器函数，用于检查是否安装了 torch 库，如果未安装则抛出 ImportError
def torch_only_method(fn):
    def wrapper(*args, **kwargs):
        if not _torch_available:
            raise ImportError(
                "You need to install pytorch to use this method or class, "
                "or activate it with environment variables USE_TORCH=1 and USE_TF=0."
            )
        else:
            return fn(*args, **kwargs)

    return wrapper


# 返回一个布尔值，指示是否安装了 ccl 库
def is_ccl_available():
    return _is_ccl_available


# 返回一个布尔值，指示是否安装了 decord 库
def is_decord_available():
    return _decord_available


# 返回一个布尔值，指示是否安装了 sudachipy 库
def is_sudachi_available():
    return _sudachipy_available


# 返回当前 sudachipy 库的版本信息
def get_sudachi_version():
    return _sudachipy_version


# 返回一个布尔值，指示是否安装了 sudachipy 并且支持 projection 选项
def is_sudachi_projection_available():
    if not is_sudachi_available():
        return False

    # 检查 sudachipy 版本是否大于等于 0.6.8，以确定是否支持 projection 选项
    return version.parse(_sudachipy_version) >= version.parse("0.6.8")


# 返回一个布尔值，指示是否安装了 jumanpp 库
def is_jumanpp_available():
    # 使用 importlib.util.find_spec 检查 rhoknp 模块和 shutil.which 检查 jumanpp 是否存在
    return (importlib.util.find_spec("rhoknp") is not None) and (shutil.which("jumanpp") is not None)


# 返回一个布尔值，指示是否安装了 cython 库
def is_cython_available():
    return importlib.util.find_spec("pyximport") is not None


# 返回一个布尔值，指示是否安装了 jieba 库
def is_jieba_available():
    return _jieba_available


# 返回一个布尔值，指示是否安装了 jinja 库
def is_jinja_available():
    return _jinja_available


# 返回一个布尔值，指示是否安装了 mlx 库
def is_mlx_available():
    return _mlx_available


# CV2_IMPORT_ERROR 的文本内容，提醒用户需要安装 OpenCV 库才能继续执行相关操作
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:

pip install opencv-python

Please note that you may need to restart your runtime after installation.
"""


# DATASETS_IMPORT_ERROR 的文本内容，提醒用户需要安装 🤗 Datasets 库才能继续执行相关操作
DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your environment. You can install it with:

pip install datasets

In a notebook or a colab, you can install it by executing a cell with

!pip install datasets

then restarting your kernel.

Note that if you have a local folder named `datasets` or a local python file named `datasets.py` in your current
working directory, python may try to import this instead of the 🤗 Datasets library. You should rename this folder or
that python file if that's the case. Please note that you may need to restart your runtime after installation.
"""


# TOKENIZERS_IMPORT_ERROR 是空字符串，没有具体的内容或注释
TOKENIZERS_IMPORT_ERROR = """
# 格式化字符串，用于给定模块名的导入错误提示信息
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# 格式化字符串，用于给定模块名的导入错误提示信息
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# 格式化字符串，用于给定模块名的导入错误提示信息
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# 格式化字符串，用于给定模块名的导入错误提示信息
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# 格式化字符串，用于给定模块名的导入错误提示信息
TORCHVISION_IMPORT_ERROR = """
{0} requires the Torchvision library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# 格式化字符串，用于给定模块名的导入错误提示信息，同时提供了关于 TensorFlow 和 PyTorch 的信息
PYTORCH_IMPORT_ERROR_WITH_TF = """
{0} requires the PyTorch library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to our PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use PyTorch please go to
https://pytorch.org/get-started/locally/ and follow the instructions that
match your environment.
"""

# 格式化字符串，用于给定模块名的导入错误提示信息，同时提供了关于 TensorFlow 和 PyTorch 的信息
TF_IMPORT_ERROR_WITH_PYTORCH = """
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
# 定义错误消息模板，用于缺少 Beautiful Soup 库时显示
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 scikit-learn 库时显示
SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your environment. You can install it with:

pip install -U scikit-learn

In a notebook or a colab, you can install it by executing a cell with

!pip install -U scikit-learn

Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 TensorFlow 库时显示
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 detectron2 库时显示
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 FLAX 库时显示
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 ftfy 库时显示
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 python-Levenshtein 库时显示
LEVENSHTEIN_IMPORT_ERROR = """
{0} requires the python-Levenshtein library but it was not found in your environment. You can install it with pip: `pip
install python-Levenshtein`. Please note that you may need to restart your runtime after installation.
"""

# 定义错误消息模板，用于缺少 g2p-en 库时显示
G2P_EN_IMPORT_ERROR = """
{0} requires the g2p-en library but it was not found in your environment. You can install it with pip:
`pip install g2p-en`. Please note that you may need to restart your runtime after installation.
"""

# 空白的错误消息模板，用于缺少 PyTorch Quantization 库时显示
PYTORCH_QUANTIZATION_IMPORT_ERROR = """
"""
# 定义当缺少 pytorch-quantization 库时所需的错误消息模板
TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} requires the tensorflow_probability library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/tensorflow/probability. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 tensorflow_text 库时所需的错误消息模板
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} requires the tensorflow_text library but it was not found in your environment. You can install it with pip as
explained here: https://www.tensorflow.org/text/guide/tf_text_intro.
Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 pandas 库时所需的错误消息模板
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 phonemizer 库时所需的错误消息模板
PHONEMIZER_IMPORT_ERROR = """
{0} requires the phonemizer library but it was not found in your environment. You can install it with pip:
`pip install phonemizer`. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 sacremoses 库时所需的错误消息模板
SACREMOSES_IMPORT_ERROR = """
{0} requires the sacremoses library but it was not found in your environment. You can install it with pip:
`pip install sacremoses`. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 scipy 库时所需的错误消息模板
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 torchaudio 库时所需的错误消息模板
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 timm 库时所需的错误消息模板
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 natten 库时所需的错误消息模板
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""

# 定义当缺少 NLTK 库时所需的错误消息模板
NLTK_IMPORT_ERROR = """
{0} requires the NLTK library but it was not found in your environment. You can install it by referring to:
# 引入 docstyle-ignore，以下注释内容是一些导入错误消息的字符串模板
# 引入 Vision 模块时的导入错误消息模板
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

# 引入 PyTesseract 模块时的导入错误消息模板
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""

# 引入 pyctcdecode 模块时的导入错误消息模板
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""

# 引入 accelerate 模块时的导入错误消息模板
ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library >= {ACCELERATE_MIN_VERSION} it was not found in your environment.
You can install or update it with pip: `pip install --upgrade accelerate`. Please note that you may need to restart your
runtime after installation.
"""

# 引入 torch ccl 模块时的导入错误消息模板
CCL_IMPORT_ERROR = """
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
"""

# 引入 essentia 模块时的导入错误消息模板
ESSENTIA_IMPORT_ERROR = """
{0} requires essentia library. But that was not found in your environment. You can install them with pip:
`pip install essentia==2.1b6.dev1034`
Please note that you may need to restart your runtime after installation.
"""

# 引入 librosa 模块时的导入错误消息模板
LIBROSA_IMPORT_ERROR = """
{0} requires thes librosa library. But that was not found in your environment. You can install them with pip:
`pip install librosa`
Please note that you may need to restart your runtime after installation.
"""

# 引入 pretty_midi 模块时的导入错误消息模板
PRETTY_MIDI_IMPORT_ERROR = """
{0} requires thes pretty_midi library. But that was not found in your environment. You can install them with pip:
`pip install pretty_midi`
Please note that you may need to restart your runtime after installation.
"""

# 引入 decord 模块时的导入错误消息模板
DECORD_IMPORT_ERROR = """
{0} requires the decord library but it was not found in your environment. You can install it with pip: `pip install
decord`. Please note that you may need to restart your runtime after installation.
"""

# 引入 Cython 模块时的导入错误消息模板
CYTHON_IMPORT_ERROR = """
{0} requires the Cython library but it was not found in your environment. You can install it with pip: `pip install
Cython`. Please note that you may need to restart your runtime after installation.
"""

# 引入 jieba 模块时的导入错误消息模板
JIEBA_IMPORT_ERROR = """
{0} requires the jieba library but it was not found in your environment. You can install it with pip: `pip install
jieba`. Please note that you may need to restart your runtime after installation.
"""

# 引入 PEFT 模块时的注释内容为空，因此无需添加任何注释
PEFT_IMPORT_ERROR = """
# 引入 OrderedDict 类型，用于定义一个有序的映射关系
BACKENDS_MAPPING = OrderedDict(
    # 列表包含了各个库及其可用性检查函数和导入错误常量的元组
    [
        # BeautifulSoup4 库的可用性检查和导入错误常量
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        # OpenCV 库的可用性检查和导入错误常量
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        # Datasets 库的可用性检查和导入错误常量
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        # Detectron2 库的可用性检查和导入错误常量
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        # Essentia 库的可用性检查和导入错误常量
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        # Faiss 库的可用性检查和导入错误常量
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        # Flax 库的可用性检查和导入错误常量
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        # FTFY 库的可用性检查和导入错误常量
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        # g2p_en 库的可用性检查和导入错误常量
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
        # Pandas 库的可用性检查和导入错误常量
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        # Phonemizer 库的可用性检查和导入错误常量
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        # Pretty MIDI 库的可用性检查和导入错误常量
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        # Levenshtein 库的可用性检查和导入错误常量
        ("levenshtein", (is_levenshtein_available, LEVENSHTEIN_IMPORT_ERROR)),
        # Librosa 库的可用性检查和导入错误常量
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        # Protobuf 库的可用性检查和导入错误常量
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        # PyCTCDecode 库的可用性检查和导入错误常量
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        # Pytesseract 库的可用性检查和导入错误常量
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        # Sacremoses 库的可用性检查和导入错误常量
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        # PyTorch Quantization 库的可用性检查和导入错误常量
        ("pytorch_quantization", (is_pytorch_quantization_available, PYTORCH_QUANTIZATION_IMPORT_ERROR)),
        # SentencePiece 库的可用性检查和导入错误常量
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        # Scikit-learn 库的可用性检查和导入错误常量
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        # Speech 库的可用性检查和导入错误常量
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        # TensorFlow Probability 库的可用性检查和导入错误常量
        ("tensorflow_probability", (is_tensorflow_probability_available, TENSORFLOW_PROBABILITY_IMPORT_ERROR)),
        # TensorFlow 库的可用性检查和导入错误常量
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        # TensorFlow Text 库的可用性检查和导入错误常量
        ("tensorflow_text", (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR)),
        # Timm 库的可用性检查和导入错误常量
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        # Natten 库的可用性检查和导入错误常量
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        # NLTK 库的可用性检查和导入错误常量
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
        # Tokenizers 库的可用性检查和导入错误常量
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        # PyTorch 库的可用性检查和导入错误常量
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        # Torchvision 库的可用性检查和导入错误常量
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        # Vision 库的可用性检查和导入错误常量
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        # SciPy 库的可用性检查和导入错误常量
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        # Accelerate 库的可用性检查和导入错误常量
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        # OneCCL 绑定库的可用性检查和导入错误常量
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        # Decord 库的可用性检查和导入错误常量
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
        # Cython 库的可用性检查和导入错误常量
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        # 结巴分词 库的可用性检查和导入错误常量
        ("jieba", (is_jieba_available, JIEBA_IMPORT_ERROR)),
        # PEFT 库的可用性检查和导入错误常量
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        # Jinja 库的可用性检查和导入错误常量
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
    ]
    # 定义一个名为 `DummyObject` 的元类，用于创建虚拟对象类
    class DummyObject(type):
        """
        Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
        `requires_backend` each time a user tries to access any method of that class.
        """

        # 拦截对类属性和方法的访问，检查所需的后端是否可用
        def __getattribute__(cls, key):
            if key.startswith("_") and key != "_from_config":
                return super().__getattribute__(key)
            # 调用 `requires_backends` 函数，检查类 `cls` 所需的后端是否可用
            requires_backends(cls, cls._backends)


    # 判断对象 `x` 是否为 Torch FX 的代理对象
    def is_torch_fx_proxy(x):
        if is_torch_fx_available():
            import torch.fx

            return isinstance(x, torch.fx.Proxy)
        return False


    # 定义一个 `_LazyModule` 类，用于惰性加载模块
    class _LazyModule(ModuleType):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        # 构造函数，初始化惰性加载模块
        def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
            super().__init__(name)
            # 设置模块的导入结构和相关属性
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            # 为类和其模块之间的映射建立字典
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # 设置模块的 `__all__` 属性，用于 IDE 的自动补全
            self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            # 设置模块的额外对象属性
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = import_structure

        # 为了在 IDE 中进行自动补全而需要的特殊方法
    # 继承父类的 __dir__() 方法，获取默认的属性列表
    def __dir__(self):
        result = super().__dir__()
        # 检查 self.__all__ 中的元素是否是子模块，有些可能已经在属性列表中，取决于是否已被访问
        # 只添加那些尚未在属性列表中的 self.__all__ 元素
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        # 返回更新后的属性列表
        return result

    # 获取属性值的方法，支持动态获取 self._objects 中的对象或者通过模块名称获取模块中的属性
    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]  # 如果属性在 self._objects 中，直接返回其值
        if name in self._modules:
            value = self._get_module(name)  # 如果属性在 self._modules 中，调用 _get_module 获取模块对象
        elif name in self._class_to_module.keys():
            # 如果属性在 self._class_to_module 中，获取相应的模块对象，并从中获取属性值
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            # 如果属性不存在于以上三种情况，则引发 AttributeError
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)  # 将获取到的属性值设置为实例的属性，以便下次直接访问
        return value

    # 根据模块名称导入模块的方法
    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            # 如果导入失败，抛出 RuntimeError 异常
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    # 序列化对象时调用的方法，返回对象的类、名称、导入结构元组
    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
class OptionalDependencyNotAvailable(BaseException):
    """用于表示未找到可选依赖项的内部错误类。"""


def direct_transformers_import(path: str, file="__init__.py") -> ModuleType:
    """直接导入 transformers 模块

    Args:
        path (`str`): 源文件的路径
        file (`str`, optional): 要与路径拼接的文件名。默认为 "__init__.py".

    Returns:
        `ModuleType`: 导入的结果模块对象
    """
    # 设置模块名为 "transformers"
    name = "transformers"
    # 构建文件的完整路径
    location = os.path.join(path, file)
    # 创建模块的规范对象
    spec = importlib.util.spec_from_file_location(name, location, submodule_search_locations=[path])
    # 根据规范对象创建模块
    module = importlib.util.module_from_spec(spec)
    # 执行模块的代码，加载模块
    spec.loader.exec_module(module)
    # 获取已加载的模块对象
    module = sys.modules[name]
    # 返回导入的模块对象
    return module
```