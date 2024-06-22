# `.\transformers\utils\import_utils.py`

```
# 版权声明和许可证信息
# 版权声明和许可证信息，指定了代码的版权和许可证信息
# 详细信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取

"""
导入工具：与导入和懒加载相关的工具。
"""

# 导入模块
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union

# 导入 packaging 模块中的 version 类
from packaging import version

# 导入 logging 模块
from . import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 检查指定包是否可用
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # 检查是否可以找到指定包的规范
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # 尝试获取指定包的版本信息
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

# 环境变量中表示 True 的值集合
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
# 环境变量中表示 True 和 AUTO 的值集合
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

# 获取环境变量 USE_TF、USE_TORCH、USE_JAX 的值
USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

# 获取环境变量 FORCE_TF_AVAILABLE 的值
FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

# `transformers` 需要 `torch>=1.11`，但这个变量是公开的，不能简单地删除它。
# 运行 torch.fx 特性和 torch.onnx 需要的 torch 版本。
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")

# 加速库的最小版本要求
ACCELERATE_MIN_VERSION = "0.21.0"
# FSDP 的最小版本要求
FSDP_MIN_VERSION = "1.12.0"

# 检查 accelerate 包是否可用，并获取其版本信息
_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
# 检查 apex 包是否可用
_apex_available = _is_package_available("apex")
# 检查 bitsandbytes 包是否可用
_bitsandbytes_available = _is_package_available("bitsandbytes")
# 对于 bs4 包，`importlib.metadata.version` 无法使用，需要使用 `beautifulsoup4`。
# 检查是否安装了 `bs4` 模块
_bs4_available = importlib.util.find_spec("bs4") is not None
# 检查是否安装了 `coloredlogs` 模块
_coloredlogs_available = _is_package_available("coloredlogs")
# 检查是否安装了 `cv2` 模块
_cv2_available = importlib.util.find_spec("cv2") is not None
# 检查是否安装了 `datasets` 模块
_datasets_available = _is_package_available("datasets")
# 检查是否安装了 `decord` 模块
_decord_available = importlib.util.find_spec("decord") is not None
# 检查是否安装了 `detectron2` 模块
_detectron2_available = _is_package_available("detectron2")
# 检查是否安装了 `faiss` 或 `faiss-cpu` 模块
_faiss_available = importlib.util.find_spec("faiss") is not None
# 尝试获取 `faiss` 或 `faiss-cpu` 模块的版本信息
try:
    _faiss_version = importlib.metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib.metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib.metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib.metadata.PackageNotFoundError:
        _faiss_available = False
# 检查是否安装了 `ftfy` 模块
_ftfy_available = _is_package_available("ftfy")
# 检查是否安装了 `g2p_en` 模块
_g2p_en_available = _is_package_available("g2p_en")
# 检查是否安装了 `intel_extension_for_pytorch` 模块，并获取版本信息
_ipex_available, _ipex_version = _is_package_available("intel_extension_for_pytorch", return_version=True)
# 检查是否安装了 `jieba` 模块
_jieba_available = _is_package_available("jieba")
# 检查是否安装了 `jinja2` 模块
_jinja_available = _is_package_available("jinja2")
# 检查是否安装了 `kenlm` 模块
_kenlm_available = _is_package_available("kenlm")
# 检查是否安装了 `keras_nlp` 模块
_keras_nlp_available = _is_package_available("keras_nlp")
# 检查是否安装了 `Levenshtein` 模块
_levenshtein_available = _is_package_available("Levenshtein")
# 检查是否安装了 `librosa` 模块
_librosa_available = _is_package_available("librosa")
# 检查是否安装了 `natten` 模块
_natten_available = _is_package_available("natten")
# 检查是否安装了 `nltk` 模块
_nltk_available = _is_package_available("nltk")
# 检查是否安装了 `onnx` 模块
_onnx_available = _is_package_available("onnx")
# 检查是否安装了 `openai` 模块
_openai_available = _is_package_available("openai")
# 检查是否安装了 `optimum` 模块
_optimum_available = _is_package_available("optimum")
# 检查是否安装了 `auto_gptq` 模块
_auto_gptq_available = _is_package_available("auto_gptq")
# 检查是否安装了 `awq` 模块
_auto_awq_available = importlib.util.find_spec("awq") is not None
# 检查是否安装了 `pandas` 模块
_pandas_available = _is_package_available("pandas")
# 检查是否安装了 `peft` 模块
_peft_available = _is_package_available("peft")
# 检查是否安装了 `phonemizer` 模块
_phonemizer_available = _is_package_available("phonemizer")
# 检查是否安装了 `psutil` 模块
_psutil_available = _is_package_available("psutil")
# 检查是否安装了 `py3nvml` 模块
_py3nvml_available = _is_package_available("py3nvml")
# 检查是否安装了 `pyctcdecode` 模块
_pyctcdecode_available = _is_package_available("pyctcdecode")
# 检查是否安装了 `pytesseract` 模块
_pytesseract_available = _is_package_available("pytesseract")
# 检查是否安装了 `pytest` 模块
_pytest_available = _is_package_available("pytest")
# 检查是否安装了 `pytorch_quantization` 模块
_pytorch_quantization_available = _is_package_available("pytorch_quantization")
# 检查是否安装了 `rjieba` 模块
_rjieba_available = _is_package_available("rjieba")
# 检查是否安装了 `sacremoses` 模块
_sacremoses_available = _is_package_available("sacremoses")
# 检查是否安装了 `safetensors` 模块
_safetensors_available = _is_package_available("safetensors")
# 检查是否安装了 `scipy` 模块
_scipy_available = _is_package_available("scipy")
# 检查是否安装了 `sentencepiece` 模块
_sentencepiece_available = _is_package_available("sentencepiece")
# 检查是否安装了 `seqio` 模块
_is_seqio_available = _is_package_available("seqio")
# 检查是否安装了 `sklearn` 模块
_sklearn_available = importlib.util.find_spec("sklearn") is not None
# 如果安装了 `sklearn` 模块，则尝试获取 `scikit-learn` 模块的版本信息
if _sklearn_available:
    try:
        importlib.metadata.version("scikit-learn")
    # 捕获 importlib.metadata.PackageNotFoundError 异常
    except importlib.metadata.PackageNotFoundError:
        # 设置 _sklearn_available 为 False
        _sklearn_available = False
# 检查是否安装了smdistributed包，返回布尔值
_smdistributed_available = importlib.util.find_spec("smdistributed") is not None
# 检查是否安装了soundfile包，返回布尔值
_soundfile_available = _is_package_available("soundfile")
# 检查是否安装了spacy包，返回布尔值
_spacy_available = _is_package_available("spacy")
# 检查是否安装了sudachipy包，返回布尔值
_sudachipy_available = _is_package_available("sudachipy")
# 检查是否安装了tensorflow_probability包，返回布尔值
_tensorflow_probability_available = _is_package_available("tensorflow_probability")
# 检查是否安装了tensorflow_text包，返回布尔值
_tensorflow_text_available = _is_package_available("tensorflow_text")
# 检查是否安装了tf2onnx包，返回布尔值
_tf2onnx_available = _is_package_available("tf2onnx")
# 检查是否安装了timm包，返回布尔值
_timm_available = _is_package_available("timm")
# 检查是否安装了tokenizers包，返回布尔值
_tokenizers_available = _is_package_available("tokenizers")
# 检查是否安装了torchaudio包，返回布尔值
_torchaudio_available = _is_package_available("torchaudio")
# 检查是否安装了torchdistx包，返回布尔值
_torchdistx_available = _is_package_available("torchdistx")
# 检查是否安装了torchvision包，返回布尔值
_torchvision_available = _is_package_available("torchvision")

# 初始化torch版本为"N/A"，torch是否可用为False
_torch_version = "N/A"
_torch_available = False
# 如果USE_TORCH在ENV_VARS_TRUE_AND_AUTO_VALUES中且USE_TF不在ENV_VARS_TRUE_VALUES中
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    # 检查torch包是否可用，如果可用则获取版本号
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False

# 初始化tensorflow版本为"N/A"，tensorflow是否可用为False
_tf_version = "N/A"
_tf_available = False
# 如果FORCE_TF_AVAILABLE在ENV_VARS_TRUE_VALUES中
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        # 检查tensorflow包是否可用
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
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
            # 获取tensorflow的版本号
            for pkg in candidates:
                try:
                    _tf_version = importlib.metadata.version(pkg)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
            _tf_available = _tf_version is not None
        if _tf_available:
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(
                    f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum."
                )
                _tf_available = False
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")

# 检查是否安装了essentia包，返回布尔值
_essentia_available = importlib.util.find_spec("essentia") is not None
try:
    # 获取essentia的版本号
    _essentia_version = importlib.metadata.version("essentia")
    # 使用 debug 级别的日志记录成功导入的 essentia 版本信息
    logger.debug(f"Successfully imported essentia version {_essentia_version}")
# 捕获 importlib.metadata.PackageNotFoundError 异常，设置 _essentia_version 为 False
except importlib.metadata.PackageNotFoundError:
    _essentia_version = False

# 检查是否安装了 pretty_midi 模块，设置 _pretty_midi_available 为 True 或 False
_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None
try:
    # 获取 pretty_midi 模块的版本号，记录日志
    _pretty_midi_version = importlib.metadata.version("pretty_midi")
    logger.debug(f"Successfully imported pretty_midi version {_pretty_midi_version}")
except importlib.metadata.PackageNotFoundError:
    # 捕获 importlib.metadata.PackageNotFoundError 异常，设置 _pretty_midi_available 为 False
    _pretty_midi_available = False

# 初始化 ccl_version 为 "N/A"，检查是否安装了 torch_ccl 或 oneccl_bindings_for_pytorch 模块，设置 _is_ccl_available 为 True 或 False
ccl_version = "N/A"
_is_ccl_available = (
    importlib.util.find_spec("torch_ccl") is not None
    or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
)
try:
    # 获取 oneccl_bind_pt 模块的版本号，记录日志
    ccl_version = importlib.metadata.version("oneccl_bind_pt")
    logger.debug(f"Detected oneccl_bind_pt version {ccl_version}")
except importlib.metadata.PackageNotFoundError:
    # 捕获 importlib.metadata.PackageNotFoundError 异常，设置 _is_ccl_available 为 False
    _is_ccl_available = False

# 初始化 _flax_available 为 False，检查是否安装了 flax 模块，设置 _flax_available 为 True 或 False
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available, _flax_version = _is_package_available("flax", return_version=True)
    if _flax_available:
        # 如果 flax 模块可用，检查是否安装了 jax 模块，记录日志
        _jax_available, _jax_version = _is_package_available("jax", return_version=True)
        if _jax_available:
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        else:
            # 如果 jax 模块不可用，设置 _flax_available 和 _jax_available 为 False
            _flax_available = _jax_available = False
            _jax_version = _flax_version = "N/A"

# 初始化 _torch_fx_available 为 False，如果 torch 模块可用，检查 torch 版本是否符合要求，设置 _torch_fx_available 为 True 或 False
if _torch_available:
    torch_version = version.parse(_torch_version)
    _torch_fx_available = (torch_version.major, torch_version.minor) >= (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )

# 返回 kenlm 模块是否可用的布尔值
def is_kenlm_available():
    return _kenlm_available

# 返回 cv2 模块是否可用的布尔值
def is_cv2_available():
    return _cv2_available

# 返回 torch 模块是否可用的布尔值
def is_torch_available():
    return _torch_available

# 返回 torch 模块的版本号
def get_torch_version():
    return _torch_version

# 返回 torch_sdpa 模块是否可用的布尔值
def is_torch_sdpa_available():
    if not is_torch_available():
        return False
    elif _torch_version == "N/A":
        return False
    # 检查 torch 版本是否符合要求，返回布尔值
    return version.parse(_torch_version) >= version.parse("2.1.1")

# 返回 torchvision 模块是否可用的布尔值
def is_torchvision_available():
    return _torchvision_available

# 返回 pyctcdecode 模块是否可用的布尔值
def is_pyctcdecode_available():
    return _pyctcdecode_available

# 返回 librosa 模块是否可用的布尔值
def is_librosa_available():
    return _librosa_available

# 返回 essentia 模块是否可用的布尔值
def is_essentia_available():
    return _essentia_available

# 返回 pretty_midi 模块是否可用的布尔值
def is_pretty_midi_available():
    return _pretty_midi_available

# 返回 torch.cuda 模块是否可用的布尔值
def is_torch_cuda_available():
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    else:
        return False

# 返回 torch_mps 模块是否可用的布尔值
def is_torch_mps_available():
    # 检查是否安装了 torch 库
    if is_torch_available():
        # 导入 torch 库
        import torch
        # 检查 torch.backends 中是否有 "mps" 属性
        if hasattr(torch.backends, "mps"):
            # 返回 torch.backends.mps.is_available() 的结果
            return torch.backends.mps.is_available()
    # 如果未安装 torch 或者没有 "mps" 属性，则返回 False
    return False
# 检查是否存在可用的 torch 库
def is_torch_bf16_gpu_available():
    if not is_torch_available():
        return False

    import torch

    # 检查是否存在可用的 CUDA GPU 并且是否支持 bf16
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# 检查是否存在可用的 torch 库
def is_torch_bf16_cpu_available():
    if not is_torch_available():
        return False

    import torch

    try:
        # 尝试访问 torch.cpu.amp.autocast 属性，检查是否存在
        _ = torch.cpu.amp.autocast
    except AttributeError:
        return False

    return True


# 检查是否存在可用的 torch 库
def is_torch_bf16_available():
    # 原始的 bf16 检查仅适用于 GPU，但后来出现了 CPU/bf16 组合，因此此实用程序已变得模糊，因此已弃用
    warnings.warn(
        "The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available "
        "or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu",
        FutureWarning,
    )
    return is_torch_bf16_gpu_available()


# 检查在特定设备上是否存在可用的 torch fp16
@lru_cache()
def is_torch_fp16_available_on_device(device):
    if not is_torch_available():
        return False

    import torch

    try:
        x = torch.zeros(2, 2, dtype=torch.float16).to(device)
        _ = x @ x
    except:  # noqa: E722
        # TODO: 更精确的异常匹配，如果可能的话
        # 大多数后端应该返回 `RuntimeError`，但这并不是保证
        return False

    return True


# 检查在特定设备上是否存在可用的 torch bf16
@lru_cache()
def is_torch_bf16_available_on_device(device):
    if not is_torch_available():
        return False

    import torch

    if device == "cuda":
        return is_torch_bf16_gpu_available()

    try:
        x = torch.zeros(2, 2, dtype=torch.bfloat16).to(device)
        _ = x @ x
    except:  # noqa: E722
        # TODO: 更精确的异常匹配，如果可能的话
        # 大多数后端应该返回 `RuntimeError`，但这并不是保证
        return False

    return True


# 检查是否存在可用的 torch tf32
def is_torch_tf32_available():
    if not is_torch_available():
        return False

    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if version.parse(version.parse(torch.__version__).base_version) < version.parse("1.7"):
        return False

    return True


# 检查是否存在可用的 torch fx
def is_torch_fx_available():
    return _torch_fx_available


# 检查是否存在可用的 peft
def is_peft_available():
    return _peft_available


# 检查是否存在可用的 bs4
def is_bs4_available():
    return _bs4_available


# 检查是否存在可用的 tensorflow
def is_tf_available():
    return _tf_available


# 检查是否存在可用的 coloredlogs
def is_coloredlogs_available():
    return _coloredlogs_available


# 检查是否存在可用的 tf2onnx
def is_tf2onnx_available():
    return _tf2onnx_available


# 检查是否存在可用的 onnx
def is_onnx_available():
    return _onnx_available


# 检查是否存在可用的 openai
def is_openai_available():
    return _openai_available


# 检查是否存在可用的 flax
def is_flax_available():
    return _flax_available


# 检查是否存在可用的 ftfy
def is_ftfy_available():
    return _ftfy_available
# 检查是否可用 G2P 英文模型
def is_g2p_en_available():
    return _g2p_en_available


# 使用 lru_cache 装饰器缓存结果，检查是否可用 Torch TPU
def is_torch_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    if not _torch_available:
        return False
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # 检查是否能找到 `xla_device`，如果找不到会引发 RuntimeError
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


# 使用 lru_cache 装饰器缓存结果，检查是否可用 Torch NeuronCore
def is_torch_neuroncore_available(check_device=True):
    if importlib.util.find_spec("torch_neuronx") is not None:
        return is_torch_tpu_available(check_device)
    return False


# 使用 lru_cache 装饰器缓存结果，检查是否可用 Torch NPU
def is_torch_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # 如果找不到 NPU 会引发 RuntimeError
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


# 检查是否可用 Torch Dynamo
def is_torchdynamo_available():
    if not is_torch_available():
        return False
    try:
        import torch._dynamo as dynamo  # noqa: F401

        return True
    except Exception:
        return False


# 检查是否可用 Torch 编译
def is_torch_compile_available():
    if not is_torch_available():
        return False

    import torch

    # 这里不进行任何版本检查，以支持标记为 1.14 的夜间版本。最终需要与 2.0 版本进行版本检查，但暂时不做。
    return hasattr(torch, "compile")


# 检查是否正在编译 Torch Dynamo
def is_torchdynamo_compiling():
    if not is_torch_available():
        return False
    try:
        import torch._dynamo as dynamo  # noqa: F401

        return dynamo.is_compiling()
    except Exception:
        return False


# 检查是否可用 Torch TensorRT FX
def is_torch_tensorrt_fx_available():
    if importlib.util.find_spec("torch_tensorrt") is None:
        return False
    return importlib.util.find_spec("torch_tensorrt.fx") is not None


# 检查是否可用数据集
def is_datasets_available():
    return _datasets_available


# 检查是否可用 Detectron2
def is_detectron2_available():
    return _detectron2_available


# 检查是否可用 rJieba
def is_rjieba_available():
    return _rjieba_available


# 检查是否可用 psutil
def is_psutil_available():
    return _psutil_available


# 检查是否可用 py3nvml
def is_py3nvml_available():
    return _py3nvml_available


# 检查是否可用 SacreMoses
def is_sacremoses_available():
    return _sacremoses_available


# 检查是否可用 Apex
def is_apex_available():
    return _apex_available


# 检查是否可用 Ninja
def is_ninja_available():
    r"""
    Code comes from *torch.utils.cpp_extension.is_ninja_available()*. Returns `True` if the
    # 检查系统上是否安装了 ninja 构建系统，如果有则返回 True，否则返回 False
    """
    # 尝试运行命令 "ninja --version"，如果成功则说明系统上安装了 ninja
    try:
        subprocess.check_output("ninja --version".split())
    # 如果运行命令失败，则捕获异常
    except Exception:
        # 返回 False
        return False
    # 如果没有异常，则说明系统上安装了 ninja，返回 True
    else:
        return True
# 检查当前环境是否安装了 Intel Extension for PyTorch，并且是否可用
def is_ipex_available():
    # 从完整版本号中获取主版本号和次版本号
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    # 如果 Torch 不可用或者 _ipex_available 为 False，则返回 False
    if not is_torch_available() or not _ipex_available:
        return False

    # 获取 Torch 和 Intel Extension for PyTorch 的主版本号和次版本号
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    # 如果 Torch 和 Intel Extension for PyTorch 的主版本号和次版本号不一致，则输出警告信息并返回 False
    if torch_major_and_minor != ipex_major_and_minor:
        logger.warning(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    # 返回 True
    return True


# 使用缓存装饰器检查是否安装了 Torch XPU，并且可能检查环境中是否有 XPU 设备
@lru_cache
def is_torch_xpu_available(check_device=False):
    "Checks if `intel_extension_for_pytorch` is installed and potentially if a XPU is in the environment"
    # 如果 Intel Extension for PyTorch 不可用，则返回 False
    if not is_ipex_available():
        return False

    import intel_extension_for_pytorch  # noqa: F401
    import torch

    # 如果需要检查设备，则尝试获取 XPU 设备数量，如果没有找到则返回 False
    if check_device:
        try:
            # 如果没有找到 XPU 设备，则会引发 RuntimeError
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    # 检查是否有 torch.xpu 属性并且 XPU 可用
    return hasattr(torch, "xpu") and torch.xpu.is_available()


# 检查是否安装了 bitsandbytes
def is_bitsandbytes_available():
    # 如果 Torch 不可用，则返回 False
    if not is_torch_available():
        return False

    # bitsandbytes 在没有可用的 cuda 时会引发错误，通过添加简单检查来避免这种情况
    import torch

    return _bitsandbytes_available and torch.cuda.is_available()


# 检查是否安装了 flash_attn_2
def is_flash_attn_2_available():
    # 如果 Torch 不可用，则返回 False
    if not is_torch_available():
        return False

    # 如果 flash_attn 包不可用，则返回 False
    if not _is_package_available("flash_attn"):
        return False

    # 检查是否有 cuda 可用
    import torch

    if not torch.cuda.is_available():
        return False

    # 根据不同的环境版本要求，返回是否 flash_attn 版本大于等于指定版本
    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    elif torch.version.hip:
        # TODO: 一旦在 https://github.com/ROCmSoftwarePlatform/flash-attention 中发布版本，请将要求提高到 2.1.0
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    else:
        return False


# 检查是否 flash_attn 版本大于等于 2.1.0
def is_flash_attn_greater_or_equal_2_10():
    # 如果 flash_attn 包不可用，则返回 False
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


# 检查是否安装了 flash_attn
def is_flash_attn_available():
    logger.warning(
        "Using `is_flash_attn_available` is deprecated and will be removed in v4.38. "
        "Please use `is_flash_attn_2_available` instead."
    )
    return is_flash_attn_2_available()


# 检查是否安装了 torchdistx
def is_torchdistx_available():
    return _torchdistx_available


# 检查是否安装了 faiss
def is_faiss_available():
    return _faiss_available


# 检查是否安装了 scipy
def is_scipy_available():
    return _scipy_available


# 检查是否安装了 sklearn
def is_sklearn_available():
    # 返回一个变量 _sklearn_available 的值
    return _sklearn_available
# 检查是否安装了 sentencepiece 库
def is_sentencepiece_available():
    return _sentencepiece_available


# 检查是否安装了 seqio 库
def is_seqio_available():
    return _is_seqio_available


# 检查是否安装了 protobuf 库
def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


# 检查是否安装了 accelerate 库，并且版本符合要求
def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION):
    if min_version is not None:
        return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)
    return _accelerate_available


# 检查是否安装了 fsdp 库，并且版本符合要求
def is_fsdp_available(min_version: str = FSDP_MIN_VERSION):
    return is_torch_available() and version.parse(_torch_version) >= version.parse(min_version)


# 检查是否安装了 optimum 库
def is_optimum_available():
    return _optimum_available


# 检查是否安装了 auto_awq 库
def is_auto_awq_available():
    return _auto_awq_available


# 检查是否安装了 auto_gptq 库
def is_auto_gptq_available():
    return _auto_gptq_available


# 检查是否安装了 levenshtein 库
def is_levenshtein_available():
    return _levenshtein_available


# 检查是否安装了 optimum.neuron 库
def is_optimum_neuron_available():
    return _optimum_available and _is_package_available("optimum.neuron")


# 检查是否安装了 safetensors 库
def is_safetensors_available():
    return _safetensors_available


# 检查是否安装了 tokenizers 库
def is_tokenizers_available():
    return _tokenizers_available


# 检查是否安装了 vision 库
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


# 检查是否安装了 pytesseract 库
def is_pytesseract_available():
    return _pytesseract_available


# 检查是否安装了 pytest 库
def is_pytest_available():
    return _pytest_available


# 检查是否安装了 spacy 库
def is_spacy_available():
    return _spacy_available


# 检查是否安装了 tensorflow_text 库
def is_tensorflow_text_available():
    return is_tf_available() and _tensorflow_text_available


# 检查是否安装了 keras_nlp 库
def is_keras_nlp_available():
    return is_tensorflow_text_available() and _keras_nlp_available


# 检查是否在 notebook 环境中
def is_in_notebook():
    try:
        # 从 tqdm.autonotebook 中适配的测试
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            # Databricks Runtime 11.0 及以上默认使用 IPython 内核，因此应与 Jupyter notebook 兼容
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False
# 检查是否 PyTorch 量化可用
def is_pytorch_quantization_available():
    return _pytorch_quantization_available


# 检查是否 TensorFlow 概率可用
def is_tensorflow_probability_available():
    return _tensorflow_probability_available


# 检查是否 Pandas 可用
def is_pandas_available():
    return _pandas_available


# 检查是否 SageMaker 数据并行可用
def is_sagemaker_dp_enabled():
    # 获取 SageMaker 特定环境变量
    sagemaker_params = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # 解析并检查字段 "sagemaker_distributed_dataparallel_enabled"
        sagemaker_params = json.loads(sagemaker_params)
        if not sagemaker_params.get("sagemaker_distributed_dataparallel_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # 最后，检查 `smdistributed` 模块是否存在
    return _smdistributed_available


# 检查是否 SageMaker 模型并行可用
def is_sagemaker_mp_enabled():
    # 从 smp_options 变量获取 SageMaker 特定 mp 参数
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # 解析并检查字段 "partitions" 是否包含在内，这是模型并行所需的
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # 从 mpi_options 变量获取 SageMaker 特定框架参数
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # 解析并检查字段 "sagemaker_mpi_enabled"
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # 最后，检查 `smdistributed` 模块是否存在
    return _smdistributed_available


# 检查是否在 SageMaker 上运行训练
def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


# 检查是否 SoundFile 可用
def is_soundfile_availble():
    return _soundfile_available


# 检查是否 Timm 可用
def is_timm_available():
    return _timm_available


# 检查是否 Natten 可用
def is_natten_available():
    return _natten_available


# 检查是否 NLTK 可用
def is_nltk_available():
    return _nltk_available


# 检查是否 TorchAudio 可用
def is_torchaudio_available():
    return _torchaudio_available


# 检查是否 Speech 可用
def is_speech_available():
    # 目前依赖于 TorchAudio，但确切的依赖关系可能会在未来发生变化
    return _torchaudio_available


# 检查是否 Phonemizer 可用
def is_phonemizer_available():
    return _phonemizer_available


# 仅适用于 Torch 的方法
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


# 检查是否 CCL 可用
def is_ccl_available():
    return _is_ccl_available


# 检查是否 Decord 可用
def is_decord_available():
    return _decord_available


# 检查是否 Sudachi 可用
def is_sudachi_available():
    return _sudachipy_available


# 检查是否 Juman++ 可用
def is_jumanpp_available():
    # 检查是否存在名为"rhoknp"的模块，并且检查是否存在名为"jumanpp"的可执行文件，返回两者的逻辑与结果
    return (importlib.util.find_spec("rhoknp") is not None) and (shutil.which("jumanpp") is not None)
# 检查是否安装了 Cython
def is_cython_available():
    return importlib.util.find_spec("pyximport") is not None


# 检查是否安装了结巴分词库
def is_jieba_available():
    return _jieba_available


# 检查是否安装了 Jinja 模板库
def is_jinja_available():
    return _jinja_available


# 忽略文档风格检查，OpenCV 库未找到时的错误提示信息
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:

pip install opencv-python

Please note that you may need to restart your runtime after installation.
"""


# 忽略文档风格检查，Datasets 库未找到时的错误提示信息
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


# 忽略文档风格检查，Tokenizers 库未找到时的错误提示信息
TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:

pip install tokenizers

In a notebook or a colab, you can install it by executing a cell with

!pip install tokenizers

Please note that you may need to restart your runtime after installation.
"""


# 忽略文档风格检查，SentencePiece 库未找到时的错误提示信息
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# 忽略文档风格检查，Protobuf 库未找到时的错误提示信息
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# 忽略文档风格检查，Faiss 库未找到时的错误提示信息
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# 忽略文档风格检查，PyTorch 库未找到时的错误提示信息
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# 忽略文档风格检查
# 当导入某个模块时出现 Torchvision 库未找到的错误提示信息
TORCHVISION_IMPORT_ERROR = """
{0} requires the Torchvision library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
# 当导入某个模块时出现 PyTorch 库未找到的错误提示信息，但找到 TensorFlow 安装的情况下的提示信息
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

# docstyle-ignore
# 当导入某个模块时出现 TensorFlow 库未找到的错误提示信息，但找到 PyTorch 安装的情况下的提示信息
TF_IMPORT_ERROR_WITH_PYTORCH = """
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "TF", but are otherwise identically named to our TF classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use TensorFlow, please follow the instructions on the
installation page https://www.tensorflow.org/install that match your environment.
"""

# docstyle-ignore
# 当导入某个模块时出现 Beautiful Soup 库未找到的错误提示信息
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
# 当导入某个模块时出现 scikit-learn 库未找到的错误提示信息
SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your environment. You can install it with:

pip install -U scikit-learn

In a notebook or a colab, you can install it by executing a cell with

!pip install -U scikit-learn

Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
# 当导入某个模块时出现 TensorFlow 库未找到的错误提示信息
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
# 当导入某个模块时出现 detectron2 库未找到的错误提示信息
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
# 当导入某个模块时出现 FLAX 库未找到的错误提示信息
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
# 安装页面: https://github.com/google/flax 并且遵循与您的环境相匹配的指南。
# 请注意，安装后可能需要重新启动运行时。

# 如果 ftfy 库在您的环境中找不到，则抛出此异常
FTFY_IMPORT_ERROR = """
{0} 需要 ftfy 库，但在您的环境中找不到。检查安装部分的说明：
https://github.com/rspeer/python-ftfy/tree/master#installing 并遵循与您的环境匹配的指南。
请注意，安装后可能需要重新启动运行时。

# 如果 python-Levenshtein 库在您的环境中找不到，则抛出此异常
LEVENSHTEIN_IMPORT_ERROR = """
{0} 需要 python-Levenshtein 库，但在您的环境中找不到。您可以使用 pip 安装它：`pip install python-Levenshtein`。
请注意，安装后可能需要重新启动运行时。

# 如果 g2p-en 库在您的环境中找不到，则抛出此异常
G2P_EN_IMPORT_ERROR = """
{0} 需要 g2p-en 库，但在您的环境中找不到。您可以使用 pip 安装它：`pip install g2p-en`。
请注意，安装后可能需要重新启动运行时。

# 如果 pytorch-quantization 库在您的环境中找不到，则抛出此异常
PYTORCH_QUANTIZATION_IMPORT_ERROR = """
{0} 需要 pytorch-quantization 库，但在您的环境中找不到。您可以使用 pip 安装它：`pip install pytorch-quantization --extra-index-url
https://pypi.ngc.nvidia.com`
请注意，安装后可能需要重新启动运行时。

# 如果 tensorflow_probability 库在您的环境中找不到，则抛出此异常
TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} 需要 tensorflow_probability 库，但在您的环境中找不到。您可以按照这里的说明使用 pip 安装：
https://github.com/tensorflow/probability。
请注意，安装后可能需要重新启动运行时。

# 如果 tensorflow_text 库在您的环境中找不到，则抛出此异常
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} 需要 tensorflow_text 库，但在您的环境中找不到。您可以按照这里的说明使用 pip 安装：
https://www.tensorflow.org/text/guide/tf_text_intro。
请注意，安装后可能需要重新启动运行时。

# 如果 pandas 库在您的环境中找不到，则抛出此异常
PANDAS_IMPORT_ERROR = """
{0} 需要 pandas 库，但在您的环境中找不到。您可以按照这里的说明使用 pip 安装：
https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html。
请注意，安装后可能需要重新启动运行时。

# 如果 phonemizer 库在您的环境中找不到，则抛出此异常
PHONEMIZER_IMPORT_ERROR = """
{0} 需要 phonemizer 库，但在您的环境中找不到。你可以使用 pip 安装它：`pip install phonemizer`。
请注意，安装后可能需要重新启动运行时。

# 如果 sacremoses 库在您的环境中找不到，则抛出此异常
SACREMOSES_IMPORT_ERROR = """
{0} 需要 sacremoses 库，但在您的环境中找不到。您可以使用 pip 安装它：`pip install sacremoses`。
请注意，安装后可能需要重新启动运行时。

# 如果 scipy 库在您的环境中找不到，则抛出此异常
SCIPY_IMPORT_ERROR = """
{0} 需要 scipy 库，但在您的环境中找不到。您可以使用 pip 安装它：
# 定义 Speech 相关错误提示信息
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`. Please note that you may need to restart your runtime after installation.
"""

# 定义 Timm 相关错误提示信息
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`. Please note that you may need to restart your runtime after installation.
"""

# 定义 Natten 相关错误提示信息
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""

# 定义 NLTK 相关错误提示信息
NLTK_IMPORT_ERROR = """
{0} requires the NLTK library but it was not found in your environment. You can install it by referring to:
https://www.nltk.org/install.html. Please note that you may need to restart your runtime after installation.
"""

# 定义 Vision 相关错误提示信息
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

# 定义 PyTesseract 相关错误提示信息
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""

# 定义 PyCTCDecode 相关错误提示信息
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""

# 定义 Accelerate 相关错误提示信息
ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library >= {ACCELERATE_MIN_VERSION} it was not found in your environment.
You can install or update it with pip: `pip install --upgrade accelerate`. Please note that you may need to restart your
runtime after installation.
"""

# 定义 CCL 相关错误提示信息
CCL_IMPORT_ERROR = """
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
"""

# 定义 Essentia 相关错误提示信息
ESSENTIA_IMPORT_ERROR = """
{0} requires essentia library. But that was not found in your environment. You can install them with pip:
`pip install essentia==2.1b6.dev1034`
Please note that you may need to restart your runtime after installation.
"""

# 定义 Librosa 相关错误提示信息
LIBROSA_IMPORT_ERROR = """
# This block intentionally left blank
"""
# 显示缺少 librosa 库的错误信息及安装提示
{0} requires thes librosa library. But that was not found in your environment. You can install them with pip:
`pip install librosa`
Please note that you may need to restart your runtime after installation.
"""

# 显示缺少 pretty_midi 库的错误信息及安装提示
PRETTY_MIDI_IMPORT_ERROR = """
{0} requires thes pretty_midi library. But that was not found in your environment. You can install them with pip:
`pip install pretty_midi`
Please note that you may need to restart your runtime after installation.
"""

# 显示缺少 decord 库的错误信息及安装提示
DECORD_IMPORT_ERROR = """
{0} requires the decord library but it was not found in your environment. You can install it with pip: `pip install
decord`. Please note that you may need to restart your runtime after installation.
"""

# 显示缺少 Cython 库的错误信息及安装提示
CYTHON_IMPORT_ERROR = """
{0} requires the Cython library but it was not found in your environment. You can install it with pip: `pip install
Cython`. Please note that you may need to restart your runtime after installation.
"""

# 显示缺少 jieba 库的错误信息及安装提示
JIEBA_IMPORT_ERROR = """
{0} requires the jieba library but it was not found in your environment. You can install it with pip: `pip install
jieba`. Please note that you may need to restart your runtime after installation.
"""

# 显示缺少 peft 库的错误信息及安装提示
PEFT_IMPORT_ERROR = """
{0} requires the peft library but it was not found in your environment. You can install it with pip: `pip install
peft`. Please note that you may need to restart your runtime after installation.
"""

# 显示缺少 jinja 库的错误信息及安装提示
JINJA_IMPORT_ERROR = """
{0} requires the jinja library but it was not found in your environment. You can install it with pip: `pip install
jinja2`. Please note that you may need to restart your runtime after installation.
"""

# 指定不同的后端和其对应的顺序
BACKENDS_MAPPING = OrderedDict(
    [
        # 检查是否 bs4 库可用, 如不可用则引发 BS4_IMPORT_ERROR
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        # 检查是否 cv2 库可用, 如不可用则引发 CV2_IMPORT_ERROR
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        # 检查是否 datasets 库可用, 如不可用则引发 DATASETS_IMPORT_ERROR
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        # 检查是否 detectron2 库可用, 如不可用则引发 DETECTRON2_IMPORT_ERROR
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        # 检查是否 essentia 库可用, 如不可用则引发 ESSENTIA_IMPORT_ERROR
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        # 检查是否 faiss 库可用, 如不可用则引发 FAISS_IMPORT_ERROR
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        # 检查是否 flax 库可用, 如不可用则引发 FLAX_IMPORT_ERROR
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        # 检查是否 ftfy 库可用, 如不可用则引发 FTFY_IMPORT_ERROR
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        # 检查是否 g2p_en 库可用, 如不可用则引发 G2P_EN_IMPORT_ERROR
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
        # 检查是否 pandas 库可用, 如不可用则引发 PANDAS_IMPORT_ERROR
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        # 检查是否 phonemizer 库可用, 如不可用则引发 PHONEMIZER_IMPORT_ERROR
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        # 检查是否 pretty_midi 库可用, 如不可用则引发 PRETTY_MIDI_IMPORT_ERROR
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        # 检查是否 levenshtein 库可用, 如不可用则引发 LEVENSHTEIN_IMPORT_ERROR
        ("levenshtein", (is_levenshtein_available, LEVENSHTEIN_IMPORT_ERROR)),
        # 检查是否 librosa 库可用, 如不可用则引发 LIBROSA_IMPORT_ERROR
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        # 检查是否 protobuf 库可用, 如不可用则引发 PROTOBUF_IMPORT_ERROR
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        # 检查是否 pyctcdecode 库可用, 如不可用则引发 PYCTCDECODE_IMPORT_ERROR
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        # 检查是否 pytesseract 库可用, 如不可用则引发 PYTESSERACT_IMPORT_ERROR
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        # 检查是否 sacremoses 库可用, 如不可用则引发 SACREMOSES_IMPORT_ERROR
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        # 检查是否 pytorch_quantization 库可用, 如不可用则引发 PYTORCH_QUANTIZATION_IMPORT_ERROR
        ("pytorch_quantization", (is_pytorch_quantization_available, PYTORCH_QUANTIZATION_IMPORT_ERROR)),
        # 检查是否 sentencepiece 库可用, 如不可用则引发 SENTENCEPIECE_IMPORT_ERROR
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        # 检查是否 sklearn 库可用, 如不可用则引发 SKLEARN_IMPORT_ERROR
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        # 检查是否 speech 库可用, 如不可用则引发 SPEECH_IMPORT_ERROR
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        # 检查是否 tensorflow_probability 库可用, 如不可用则引发 TENSORFLOW_PROBABILITY_IMPORT_ERROR
        ("tensorflow_probability", (is_tensorflow_probability_available, TENSORFLOW_PROBABILITY_IMPORT_ERROR)),
        # 检查是否 tf 库可用, 如不可用则引发 TENSORFLOW_IMPORT_ERROR
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        # 检查是否 tensorflow_text 库可用, 如不可用则引发 TENSORFLOW_TEXT_IMPORT_ERROR
        ("tensorflow_text", (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR)),
        # 检查是否 timm 库可用, 如不可用则引发 TIMM_IMPORT_ERROR
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        # 检查是否 natten 库可用, 如不可用则引发 NATTEN_IMPORT_ERROR
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        # 检查是否 nltk 库可用, 如不可用则引发 NLTK_IMPORT_ERROR
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
        # 检查是否 tokenizers 库可用, 如不可用则引发 TOKENIZERS_IMPORT_ERROR
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        # 检查是否 torch 库可用, 如不可用则引发 PYTORCH_IMPORT_ERROR
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        # 检查是否 torchvision 库可用, 如不可用则引发 TORCHVISION_IMPORT_ERROR
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        # 检查是否 vision 库可用, 如不可用则引发 VISION_IMPORT_ERROR
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        # 检查是否 scipy 库可用, 如不可用则引发 SCIPY_IMPORT_ERROR
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        # 检查是否 accelerate 库可用, 如不可用则引发 ACCELERATE_IMPORT_ERROR
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        # 检查是否 oneccl_bind_pt 库可用, 如不可用则引发 CCL_IMPORT_ERROR
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        # 检查是否 decord 库可用, 如不可用则引发 DECORD_IMPORT_ERROR
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
        # 检查是否 cython 库可用, 如不可用则引发 CYTHON_IMPORT_ERROR
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        # 检查是否 jieba 库可用, 如不可用则引发 JIEBA_IMPORT_ERROR
        ("jieba", (is_jieba_available, JIEBA_IMPORT_ERROR)),
        # 检查是否 peft 库可用, 如不可用则引发 PEFT_IMPORT_ERROR
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        # 检查是否 jinja 库可用, 如不可用则引发 JINJA_IMPORT_ERROR
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
    ]
    ```py  
# 定义一个函数，用于检查对象所需的后端
def requires_backends(obj, backends):
    # 如果backends不是列表或元组类型，则将其转换为列表
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    # 获取对象的名称，如果有 "__name__" 属性则使用该属性，否则使用类名
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    # 对于没有 "TF" 的类进行 torch-only 的错误提示
    if "torch" in backends and "tf" not in backends and not is_torch_available() and is_tf_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))

    # 对于尝试加载 TF 类的 PyTorch 用户进行反向错误提示
    if "tf" in backends and "torch" not in backends and is_torch_available() and not is_tf_available():
        raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))

    # 对于每个后端进行检查，如果有未满足条件的则抛出 ImportError
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


# 定义一个元类，用于创建占位对象，尝试访问对象的方法时将抛出 ImportError
class DummyObject(type):
    # 重载__getattribute__方法，实现访问对象方法时抛出 ImportError 的功能
    def __getattribute__(cls, key):
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)


# 判断对象是否为 torch.fx.Proxy 对象
def is_torch_fx_proxy(x):
    # 如果存在 torch.fx 模块，则判断是否为 torch.fx.Proxy 对象
    if is_torch_fx_available():
        import torch.fx
        return isinstance(x, torch.fx.Proxy)
    return False


# 定义一个懒加载模块类，延迟导入模块中的对象直到对象被访问时
class _LazyModule(ModuleType):
    # 初始化方法，设置模块的属性和导入结构
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())  # 设置模块集合
        self._class_to_module = {}  # 设置类名到模块名的映射字典
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # 用于 IDE 的自动补全
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file  # 模块文件路径
        self.__spec__ = module_spec  # 模块规范
        self.__path__ = [os.path.dirname(module_file)]  # 模块路径
        self._objects = {} if extra_objects is None else extra_objects  # 额外的对象集合
        self._name = name  # 模块名称
        self._import_structure = import_structure  # 导入结构
    # 定义 __dir__ 方法，返回对象的属性列表
    def __dir__(self):
        # 先调用父类的 __dir__ 方法获取属性列表
        result = super().__dir__()
        # 遍历 self.__all__ 中的元素
        for attr in self.__all__:
            # 如果该元素不在 result 中
            if attr not in result:
                # 则添加到 result 中
                result.append(attr)
        # 返回最终的属性列表
        return result
    
    # 定义 __getattr__ 方法，用于获取对象的属性
    def __getattr__(self, name: str) -> Any:
        # 如果 name 在 self._objects 中
        if name in self._objects:
            # 返回对应的值
            return self._objects[name]
        # 如果 name 在 self._modules 中
        elif name in self._modules:
            # 获取对应的模块
            value = self._get_module(name)
        # 如果 name 在 self._class_to_module 的键中
        elif name in self._class_to_module.keys():
            # 获取对应的模块
            module = self._get_module(self._class_to_module[name])
            # 从模块中获取对应的属性值
            value = getattr(module, name)
        else:
            # 如果都没找到，则抛出属性错误异常
            raise AttributeError(f"module {self.__name__} has no attribute {name}")
        
        # 将获取到的属性值设置到 self 中
        setattr(self, name, value)
        # 返回属性值
        return value
    
    # 定义 _get_module 方法，用于获取模块
    def _get_module(self, module_name: str):
        try:
            # 使用 importlib.import_module 导入模块
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            # 如果导入模块出错，抛出运行时错误异常
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e
    
    # 定义 __reduce__ 方法，用于对象序列化
    def __reduce__(self):
        # 返回元组，包含类对象和构造参数
        return (self.__class__, (self._name, self.__file__, self._import_structure))
class OptionalDependencyNotAvailable(BaseException):
    """自定义异常类，用于表示未找到可选依赖。"""


def direct_transformers_import(path: str, file="__init__.py") -> ModuleType:
    """直接导入transformers模块

    Args:
        path (`str`): 源文件的路径
        file (`str`, optional): 与路径组合的文件名。默认为"__init__.py"。

    Returns:
        `ModuleType`: 导入的模块
    """
    # 模块名称
    name = "transformers"
    # 源文件的完整路径
    location = os.path.join(path, file)
    # 根据路径和文件创建模块的规范
    spec = importlib.util.spec_from_file_location(name, location, submodule_search_locations=[path])
    # 根据规范创建模块对象
    module = importlib.util.module_from_spec(spec)
    # 执行模块对象，将其载入到内存中
    spec.loader.exec_module(module)
    # 获取模块对象
    module = sys.modules[name]
    # 返回导入的模块
    return module
```