# `.\testing_utils.py`

```
# 导入必要的标准库和第三方库
import collections  # 提供额外的数据容器，如deque（双端队列）
import contextlib  # 提供用于管理上下文的工具
import doctest  # 提供用于运行文档测试的模块
import functools  # 提供函数式编程的工具，如partial函数应用
import importlib  # 提供用于动态加载模块的工具
import inspect  # 提供用于检查源代码的工具
import logging  # 提供用于记录日志消息的功能
import multiprocessing  # 提供用于多进程编程的工具
import os  # 提供与操作系统交互的功能
import re  # 提供支持正则表达式的工具
import shlex  # 提供用于解析和操作命令行字符串的工具
import shutil  # 提供高级文件操作功能的工具
import subprocess  # 提供用于创建子进程的功能
import sys  # 提供与Python解释器交互的功能
import tempfile  # 提供创建临时文件和目录的功能
import time  # 提供时间相关的功能
import unittest  # 提供用于编写和运行单元测试的工具
from collections import defaultdict  # 提供默认字典的功能
from collections.abc import Mapping  # 提供抽象基类，用于检查映射类型
from functools import wraps  # 提供用于创建装饰器的工具
from io import StringIO  # 提供内存中文本I/O的工具
from pathlib import Path  # 提供面向对象的路径操作功能
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union  # 提供类型提示支持
from unittest import mock  # 提供用于模拟测试的工具
from unittest.mock import patch  # 提供用于模拟测试的工具

import urllib3  # 提供HTTP客户端的功能

from transformers import logging as transformers_logging  # 导入transformers库中的logging模块

from .integrations import (  # 导入自定义模块中的一系列集成检查函数
    is_clearml_available,
    is_optuna_available,
    is_ray_available,
    is_sigopt_available,
    is_tensorboard_available,
    is_wandb_available,
)
from .integrations.deepspeed import is_deepspeed_available  # 导入自定义模块中的深度加速集成检查函数
from .utils import (  # 导入自定义模块中的一系列实用工具检查函数
    is_accelerate_available,
    is_apex_available,
    is_aqlm_available,
    is_auto_awq_available,
    is_auto_gptq_available,
    is_bitsandbytes_available,
    is_bs4_available,
    is_cv2_available,
    is_cython_available,
    is_decord_available,
    is_detectron2_available,
    is_essentia_available,
    is_faiss_available,
    is_flash_attn_2_available,
    is_flax_available,
    is_fsdp_available,
    is_ftfy_available,
    is_g2p_en_available,
    is_galore_torch_available,
    is_ipex_available,
    is_jieba_available,
    is_jinja_available,
    is_jumanpp_available,
    is_keras_nlp_available,
    is_levenshtein_available,
    is_librosa_available,
    is_natten_available,
    is_nltk_available,
    is_onnx_available,
    is_optimum_available,
    is_pandas_available,
    is_peft_available,
    is_phonemizer_available,
    is_pretty_midi_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytest_available,
    is_pytorch_quantization_available,
    is_quanto_available,
    is_rjieba_available,
    is_sacremoses_available,
    is_safetensors_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_seqio_available,
    is_soundfile_availble,
    is_spacy_available,
    is_sudachi_available,
    is_sudachi_projection_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
)
    # 检查当前设备是否支持 Torch 的 BF16 数据类型
    is_torch_bf16_available_on_device,
    # 检查当前 CPU 是否支持 Torch 的 BF16 数据类型
    is_torch_bf16_cpu_available,
    # 检查当前 GPU 是否支持 Torch 的 BF16 数据类型
    is_torch_bf16_gpu_available,
    # 检查当前设备是否支持 Torch 的 FP16 数据类型
    is_torch_fp16_available_on_device,
    # 检查当前设备是否支持 Torch 的 NeuronCore 加速器
    is_torch_neuroncore_available,
    # 检查当前设备是否支持 Torch 的 NPU 加速器
    is_torch_npu_available,
    # 检查当前设备是否支持 Torch 的 SDPA 加速器
    is_torch_sdpa_available,
    # 检查当前设备是否支持 Torch 的 TensorRT FX 加速器
    is_torch_tensorrt_fx_available,
    # 检查当前设备是否支持 Torch 的 TF32 数据类型
    is_torch_tf32_available,
    # 检查当前设备是否支持 Torch 的 XLA 加速器
    is_torch_xla_available,
    # 检查当前设备是否支持 Torch 的 XPU 加速器
    is_torch_xpu_available,
    # 检查当前环境是否支持 Torch Audio 库
    is_torchaudio_available,
    # 检查当前环境是否支持 TorchDynamo 库
    is_torchdynamo_available,
    # 检查当前环境是否支持 TorchVision 库
    is_torchvision_available,
    # 检查当前环境是否支持 Torch 的 Vision 扩展
    is_vision_available,
    # 将字符串转换为布尔值（支持"true", "false", "yes", "no", "1", "0"等）
    strtobool,
# 如果加速功能可用，则从 accelerate.state 中导入 AcceleratorState 和 PartialState 类
if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState


# 如果 pytest 可用，则从 _pytest.doctest 中导入以下模块
# Module: 用于表示 Python 模块的类
# _get_checker: 获取 doctest 的检查器
# _get_continue_on_failure: 获取 doctest 的继续失败选项
# _get_runner: 获取 doctest 的运行器
# _is_mocked: 检查是否模拟了对象
# _patch_unwrap_mock_aware: 解除 Mock 对象感知的补丁
# get_optionflags: 获取 doctest 的选项标志
from _pytest.doctest import (
    Module,
    _get_checker,
    _get_continue_on_failure,
    _get_runner,
    _is_mocked,
    _patch_unwrap_mock_aware,
    get_optionflags,
)

# 如果 pytest 不可用，则将 Module 和 DoctestItem 设置为 object 类型
else:
    Module = object
    DoctestItem = object


# 定义了一个小型模型的标识符字符串
SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"

# 用于测试自动检测模型类型的标识符
DUMMY_UNKNOWN_IDENTIFIER = "julien-c/dummy-unknown"
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"

# 用于测试 Hub 的用户和端点
USER = "__DUMMY_TRANSFORMERS_USER__"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

# 仅在受控的 CI 实例中可用，用于测试用的令牌
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"


# 从环境变量中解析布尔类型的标志
def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # 如果 KEY 未设置，则使用默认值 `default`
        _value = default
    else:
        # 如果 KEY 已设置，则尝试将其转换为 True 或 False
        try:
            _value = strtobool(value)
        except ValueError:
            # 如果值不是 `yes` 或 `no`，则抛出异常
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


# 从环境变量中解析整数类型的值
def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            # 如果值不是整数，则抛出异常
            raise ValueError(f"If set, {key} must be a int.")
    return _value


# 根据环境变量 `RUN_SLOW` 解析是否运行慢速测试的标志
_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
# 根据环境变量 `RUN_PT_TF_CROSS_TESTS` 解析是否运行 PyTorch 和 TensorFlow 交叉测试的标志
_run_pt_tf_cross_tests = parse_flag_from_env("RUN_PT_TF_CROSS_TESTS", default=True)
# 根据环境变量 `RUN_PT_FLAX_CROSS_TESTS` 解析是否运行 PyTorch 和 Flax 交叉测试的标志
_run_pt_flax_cross_tests = parse_flag_from_env("RUN_PT_FLAX_CROSS_TESTS", default=True)
# 根据环境变量 `RUN_CUSTOM_TOKENIZERS` 解析是否运行自定义分词器测试的标志
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
# 根据环境变量 `HUGGINGFACE_CO_STAGING` 解析是否运行在 Hugging Face CO 预发布环境中的标志
_run_staging = parse_flag_from_env("HUGGINGFACE_CO_STAGING", default=False)
# 根据环境变量 `TF_GPU_MEMORY_LIMIT` 解析 TensorFlow GPU 内存限制的值
_tf_gpu_memory_limit = parse_int_from_env("TF_GPU_MEMORY_LIMIT", default=None)
# 根据环境变量 `RUN_PIPELINE_TESTS` 解析是否运行管道测试的标志
_run_pipeline_tests = parse_flag_from_env("RUN_PIPELINE_TESTS", default=True)
# 根据环境变量 `RUN_TOOL_TESTS` 解析是否运行工具测试的标志
_run_tool_tests = parse_flag_from_env("RUN_TOOL_TESTS", default=False)
# 根据环境变量 `RUN_THIRD_PARTY_DEVICE_TESTS` 解析是否运行第三方设备测试的标志
_run_third_party_device_tests = parse_flag_from_env("RUN_THIRD_PARTY_DEVICE_TESTS", default=False)


# 函数装饰器，用于标记 PT+TF 交叉测试
def is_pt_tf_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and TensorFlow.

    PT+TF tests are skipped by default and we can run only them by setting RUN_PT_TF_CROSS_TESTS environment variable
    to a truthy value and selecting the is_pt_tf_cross_test pytest mark.

    """
    # 如果未设置环境变量 `RUN_PT_TF_CROSS_TESTS` 或者当前环境中没有安装 PyTorch 或 TensorFlow，
    # 则跳过 PT+TF 测试
    if not _run_pt_tf_cross_tests or not is_torch_available() or not is_tf_available():
        return unittest.skip("test is PT+TF test")(test_case)
    else:
        # 尝试导入 pytest 模块，避免在主库中硬编码依赖 pytest
        try:
            import pytest  
        # 如果导入失败，返回原始的 test_case
        except ImportError:
            return test_case
        # 如果导入成功，应用 pytest.mark.is_pt_tf_cross_test() 装饰器到 test_case 上
        else:
            return pytest.mark.is_pt_tf_cross_test()(test_case)
# 标记一个测试用例为控制 PyTorch 和 Flax 交互的测试的装饰器

PT+FLAX 测试默认情况下会被跳过，只有当设置了环境变量 RUN_PT_FLAX_CROSS_TESTS 为真值并且选择了 is_pt_flax_cross_test pytest 标记时才会运行。

def is_pt_flax_cross_test(test_case):
    if not _run_pt_flax_cross_tests or not is_torch_available() or not is_flax_available():
        # 如果不满足运行条件（未设置环境变量或者没有可用的 PyTorch 或 Flax），则跳过测试
        return unittest.skip("test is PT+FLAX test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中强制依赖 pytest
        except ImportError:
            return test_case
        else:
            # 使用 pytest 的 is_pt_flax_cross_test 标记来标记测试用例
            return pytest.mark.is_pt_flax_cross_test()(test_case)


# 标记一个测试用例为在 staging 环境下运行的测试的装饰器

这些测试将在 huggingface.co 的 staging 环境下运行，而不是真实的模型中心。

def is_staging_test(test_case):
    if not _run_staging:
        # 如果不运行 staging 测试，则跳过测试
        return unittest.skip("test is staging test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中强制依赖 pytest
        except ImportError:
            return test_case
        else:
            # 使用 pytest 的 is_staging_test 标记来标记测试用例
            return pytest.mark.is_staging_test()(test_case)


# 标记一个测试用例为 pipeline 测试的装饰器

如果未将 RUN_PIPELINE_TESTS 设置为真值，则这些测试将被跳过。

def is_pipeline_test(test_case):
    if not _run_pipeline_tests:
        # 如果不运行 pipeline 测试，则跳过测试
        return unittest.skip("test is pipeline test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中强制依赖 pytest
        except ImportError:
            return test_case
        else:
            # 使用 pytest 的 is_pipeline_test 标记来标记测试用例
            return pytest.mark.is_pipeline_test()(test_case)


# 标记一个测试用例为工具测试的装饰器

如果未将 RUN_TOOL_TESTS 设置为真值，则这些测试将被跳过。

def is_tool_test(test_case):
    if not _run_tool_tests:
        # 如果不运行工具测试，则跳过测试
        return unittest.skip("test is a tool test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中强制依赖 pytest
        except ImportError:
            return test_case
        else:
            # 使用 pytest 的 is_tool_test 标记来标记测试用例
            return pytest.mark.is_tool_test()(test_case)


# 标记一个测试用例为慢速测试的装饰器

慢速测试默认情况下会被跳过。设置 RUN_SLOW 环境变量为真值以运行这些测试。

def slow(test_case):
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


# 标记一个测试用例为太慢测试的装饰器

太慢的测试在修复过程中会被跳过。不应将任何测试标记为 "tooslow"，因为这些测试不会被 CI 测试。

def tooslow(test_case):
    return unittest.skip("test is too slow")(test_case)


# 标记一个测试用例为自定义分词器测试的装饰器
    """
    自定义分词器需要额外的依赖项，默认情况下会被跳过。将环境变量 RUN_CUSTOM_TOKENIZERS
    设置为真值，以便运行它们。
    """
    # 返回一个装饰器，根据 _run_custom_tokenizers 的真假决定是否跳过测试用例
    return unittest.skipUnless(_run_custom_tokenizers, "test of custom tokenizers")(test_case)
# 装饰器，用于标记需要 BeautifulSoup4 的测试用例。在未安装 BeautifulSoup4 时跳过这些测试。
def require_bs4(test_case):
    return unittest.skipUnless(is_bs4_available(), "test requires BeautifulSoup4")(test_case)


# 装饰器，用于标记需要 GaLore 的测试用例。在未安装 GaLore 时跳过这些测试。
def require_galore_torch(test_case):
    return unittest.skipUnless(is_galore_torch_available(), "test requires GaLore")(test_case)


# 装饰器，用于标记需要 OpenCV 的测试用例。在未安装 OpenCV 时跳过这些测试。
def require_cv2(test_case):
    return unittest.skipUnless(is_cv2_available(), "test requires OpenCV")(test_case)


# 装饰器，用于标记需要 Levenshtein 的测试用例。在未安装 Levenshtein 时跳过这些测试。
def require_levenshtein(test_case):
    return unittest.skipUnless(is_levenshtein_available(), "test requires Levenshtein")(test_case)


# 装饰器，用于标记需要 NLTK 的测试用例。在未安装 NLTK 时跳过这些测试。
def require_nltk(test_case):
    return unittest.skipUnless(is_nltk_available(), "test requires NLTK")(test_case)


# 装饰器，用于标记需要 accelerate 的测试用例。在未安装 accelerate 时跳过这些测试。
def require_accelerate(test_case):
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


# 装饰器，用于标记需要 fsdp 的测试用例。在未安装 fsdp 或版本不符合要求时跳过这些测试。
def require_fsdp(test_case, min_version: str = "1.12.0"):
    return unittest.skipUnless(is_fsdp_available(min_version), f"test requires torch version >= {min_version}")(test_case)


# 装饰器，用于标记需要 g2p_en 的测试用例。在未安装 SentencePiece 时跳过这些测试。
def require_g2p_en(test_case):
    return unittest.skipUnless(is_g2p_en_available(), "test requires g2p_en")(test_case)


# 装饰器，用于标记需要 safetensors 的测试用例。在未安装 safetensors 时跳过这些测试。
def require_safetensors(test_case):
    return unittest.skipUnless(is_safetensors_available(), "test requires safetensors")(test_case)


# 装饰器，用于标记需要 rjieba 的测试用例。在未安装 rjieba 时跳过这些测试。
def require_rjieba(test_case):
    return unittest.skipUnless(is_rjieba_available(), "test requires rjieba")(test_case)


# 装饰器，用于标记需要 jieba 的测试用例。在未安装 jieba 时跳过这些测试。
def require_jieba(test_case):
    return unittest.skipUnless(is_jieba_available(), "test requires jieba")(test_case)


# 装饰器，用于标记需要 jinja 的测试用例。在此处仅声明函数，实际装饰逻辑未提供。
def require_jinja(test_case):
    # Placeholder for decorator marking tests requiring Jinja
    pass
    # 使用装饰器标记一个需要 jinja 的测试用例。如果 jinja 没有安装，则跳过这些测试。
    """
    使用 unittest.skipUnless 函数来动态地装饰测试用例，只有在 jinja 可用时才运行该测试用例。
    如果 is_jinja_available() 函数返回 True，则装饰器返回一个可用于跳过测试的装饰器函数，否则返回 None。
    """
    return unittest.skipUnless(is_jinja_available(), "test requires jinja")(test_case)
# 根据条件判断是否加载 tf2onnx
def require_tf2onnx(test_case):
    return unittest.skipUnless(is_tf2onnx_available(), "test requires tf2onnx")(test_case)


# 根据条件判断是否加载 ONNX
def require_onnx(test_case):
    return unittest.skipUnless(is_onnx_available(), "test requires ONNX")(test_case)


# 根据条件判断是否加载 Timm
def require_timm(test_case):
    """
    Decorator marking a test that requires Timm.

    These tests are skipped when Timm isn't installed.
    """
    return unittest.skipUnless(is_timm_available(), "test requires Timm")(test_case)


# 根据条件判断是否加载 NATTEN
def require_natten(test_case):
    """
    Decorator marking a test that requires NATTEN.

    These tests are skipped when NATTEN isn't installed.
    """
    return unittest.skipUnless(is_natten_available(), "test requires natten")(test_case)


# 根据条件判断是否加载 PyTorch
def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.
    """
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


# 根据条件判断是否加载 Flash Attention
def require_flash_attn(test_case):
    """
    Decorator marking a test that requires Flash Attention.

    These tests are skipped when Flash Attention isn't installed.
    """
    return unittest.skipUnless(is_flash_attn_2_available(), "test requires Flash Attention")(test_case)


# 根据条件判断是否加载 PyTorch's SDPA
def require_torch_sdpa(test_case):
    """
    Decorator marking a test that requires PyTorch's SDPA.

    These tests are skipped when requirements are not met (torch version).
    """
    return unittest.skipUnless(is_torch_sdpa_available(), "test requires PyTorch SDPA")(test_case)


# 根据条件判断是否加载 HF token
def require_read_token(fn):
    """
    A decorator that loads the HF token for tests that require to load gated models.
    """
    token = os.getenv("HF_HUB_READ_TOKEN")

    @wraps(fn)
    def _inner(*args, **kwargs):
        with patch("huggingface_hub.utils._headers.get_token", return_value=token):
            return fn(*args, **kwargs)

    return _inner


# 根据条件判断是否加载 PEFT
def require_peft(test_case):
    """
    Decorator marking a test that requires PEFT.

    These tests are skipped when PEFT isn't installed.
    """
    return unittest.skipUnless(is_peft_available(), "test requires PEFT")(test_case)


# 根据条件判断是否加载 Torchvision
def require_torchvision(test_case):
    """
    Decorator marking a test that requires Torchvision.

    These tests are skipped when Torchvision isn't installed.
    """
    return unittest.skipUnless(is_torchvision_available(), "test requires Torchvision")(test_case)


# 根据条件判断是否加载 PyTorch 或 TensorFlow
def require_torch_or_tf(test_case):
    """
    Decorator marking a test that requires PyTorch or TensorFlow.

    These tests are skipped when neither PyTorch nor TensorFlow is installed.
    """
    return unittest.skipUnless(is_torch_available() or is_tf_available(), "test requires PyTorch or TensorFlow")(
        test_case
    )


# 根据条件判断是否加载 Intel Extension for PyTorch
def require_intel_extension_for_pytorch(test_case):
    """
    Decorator marking a test that requires Intel Extension for PyTorch.
    """
    # 注释部分未提供
    pass
    # 当未安装Intel Extension for PyTorch或者其版本与当前PyTorch版本不匹配时，跳过这些测试。
    """
    返回一个装饰器，用于根据条件跳过测试。
    装饰器检查是否可用Intel Extension for PyTorch（IPEX）。
    如果不可用或版本不匹配，则跳过测试，并提供相应的提示信息。
    参考链接：https://github.com/intel/intel-extension-for-pytorch
    """
    return unittest.skipUnless(
        is_ipex_available(),
        "test requires Intel Extension for PyTorch to be installed and match current PyTorch version, see"
        " https://github.com/intel/intel-extension-for-pytorch",
    )(test_case)
# 装饰器，用于标记一个测试需要 TensorFlow probability
def require_tensorflow_probability(test_case):
    # 返回一个装饰器，其功能是当 TensorFlow probability 未安装时跳过测试
    return unittest.skipUnless(is_tensorflow_probability_available(), "test requires TensorFlow probability")(
        test_case
    )


# 装饰器，用于标记一个测试需要 torchaudio
def require_torchaudio(test_case):
    # 返回一个装饰器，其功能是当 torchaudio 未安装时跳过测试
    return unittest.skipUnless(is_torchaudio_available(), "test requires torchaudio")(test_case)


# 装饰器，用于标记一个测试需要 TensorFlow
def require_tf(test_case):
    # 返回一个装饰器，其功能是当 TensorFlow 未安装时跳过测试
    return unittest.skipUnless(is_tf_available(), "test requires TensorFlow")(test_case)


# 装饰器，用于标记一个测试需要 JAX & Flax
def require_flax(test_case):
    # 返回一个装饰器，其功能是当 JAX 或 Flax 未安装时跳过测试
    return unittest.skipUnless(is_flax_available(), "test requires JAX & Flax")(test_case)


# 装饰器，用于标记一个测试需要 SentencePiece
def require_sentencepiece(test_case):
    # 返回一个装饰器，其功能是当 SentencePiece 未安装时跳过测试
    return unittest.skipUnless(is_sentencepiece_available(), "test requires SentencePiece")(test_case)


# 装饰器，用于标记一个测试需要 Sacremoses
def require_sacremoses(test_case):
    # 返回一个装饰器，其功能是当 Sacremoses 未安装时跳过测试
    return unittest.skipUnless(is_sacremoses_available(), "test requires Sacremoses")(test_case)


# 装饰器，用于标记一个测试需要 Seqio
def require_seqio(test_case):
    # 返回一个装饰器，其功能是当 Seqio 未安装时跳过测试
    return unittest.skipUnless(is_seqio_available(), "test requires Seqio")(test_case)


# 装饰器，用于标记一个测试需要 Scipy
def require_scipy(test_case):
    # 返回一个装饰器，其功能是当 Scipy 未安装时跳过测试
    return unittest.skipUnless(is_scipy_available(), "test requires Scipy")(test_case)


# 装饰器，用于标记一个测试需要 🤗 Tokenizers
def require_tokenizers(test_case):
    # 返回一个装饰器，其功能是当 🤗 Tokenizers 未安装时跳过测试
    return unittest.skipUnless(is_tokenizers_available(), "test requires tokenizers")(test_case)


# 装饰器，用于标记一个测试需要 tensorflow_text
def require_tensorflow_text(test_case):
    # 返回一个装饰器，其功能是当 tensorflow_text 未安装时跳过测试
    return unittest.skipUnless(is_tensorflow_text_available(), "test requires tensorflow_text")(test_case)


# 装饰器，用于标记一个测试需要 keras_nlp
def require_keras_nlp(test_case):
    # 返回一个装饰器，其功能是当 keras_nlp 未安装时跳过测试
    return unittest.skipUnless(is_keras_nlp_available(), "test requires keras_nlp")(test_case)


# 装饰器，用于标记一个测试需要 Pandas
def require_pandas(test_case):
    """
    Decorator marking a test that requires Pandas. These tests are skipped when Pandas isn't installed.
    """
    return unittest.skipUnless(is_pandas_available(), "test requires Pandas")(test_case)
    # 使用装饰器标记一个需要 pandas 的测试用例。当 pandas 没有安装时，这些测试将被跳过。
    """
    # 返回一个装饰器，根据 pandas 的可用性决定是否跳过测试用例
    return unittest.skipUnless(is_pandas_available(), "test requires pandas")(test_case)
# 标记一个测试需要 PyTesseract。如果 PyTesseract 没有安装，则跳过这些测试。
def require_pytesseract(test_case):
    return unittest.skipUnless(is_pytesseract_available(), "test requires PyTesseract")(test_case)


# 标记一个测试需要 PyTorch Quantization Toolkit。如果 PyTorch Quantization Toolkit 没有安装，则跳过这些测试。
def require_pytorch_quantization(test_case):
    return unittest.skipUnless(is_pytorch_quantization_available(), "test requires PyTorch Quantization Toolkit")(test_case)


# 标记一个测试需要视觉相关的依赖。如果 torchaudio 没有安装，则跳过这些测试。
def require_vision(test_case):
    return unittest.skipUnless(is_vision_available(), "test requires vision")(test_case)


# 标记一个测试需要 ftfy。如果 ftfy 没有安装，则跳过这些测试。
def require_ftfy(test_case):
    return unittest.skipUnless(is_ftfy_available(), "test requires ftfy")(test_case)


# 标记一个测试需要 SpaCy。如果 SpaCy 没有安装，则跳过这些测试。
def require_spacy(test_case):
    return unittest.skipUnless(is_spacy_available(), "test requires spacy")(test_case)


# 标记一个测试需要 decord。如果 decord 没有安装，则跳过这些测试。
def require_decord(test_case):
    return unittest.skipUnless(is_decord_available(), "test requires decord")(test_case)


# 标记一个测试需要多 GPU 设置（在 PyTorch 中）。如果没有多个 GPU，则跳过这些测试。
# 若要仅运行多 GPU 测试，请假设所有测试名称包含 multi_gpu：
# $ pytest -sv ./tests -k "multi_gpu"
def require_torch_multi_gpu(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)


# 标记一个测试需要多加速器设置（在 PyTorch 中）。如果没有多个加速器，则跳过这些测试。
# 若要仅运行多加速器测试，请假设所有测试名称包含 multi_accelerator：
# $ pytest -sv ./tests -k "multi_accelerator"
def require_torch_multi_accelerator(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    return unittest.skipUnless(backend_device_count(torch_device) > 1, "test requires multiple accelerators")(test_case)


# 标记一个测试需要 0 或 1 个 GPU 设置（在 PyTorch 中）。
def require_torch_non_multi_gpu(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch
    # 返回一个装饰器，用于条件性跳过测试
    return unittest.skipUnless(torch.cuda.device_count() < 2, "test requires 0 or 1 GPU")(test_case)
# 标记一个测试需要零或一个加速器设置（在PyTorch中）的装饰器
def require_torch_non_multi_accelerator(test_case):
    # 如果PyTorch不可用，则跳过测试
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    # 返回一个条件，该条件检查当前设备上的后端设备数量是否小于2，否则跳过测试
    return unittest.skipUnless(backend_device_count(torch_device) < 2, "test requires 0 or 1 accelerator")(test_case)


# 标记一个测试需要零、一个或两个GPU设置（在PyTorch中）的装饰器
def require_torch_up_to_2_gpus(test_case):
    # 如果PyTorch不可用，则跳过测试
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    # 返回一个条件，该条件检查当前机器上的GPU数量是否小于3，否则跳过测试
    return unittest.skipUnless(torch.cuda.device_count() < 3, "test requires 0 or 1 or 2 GPUs")(test_case)


# 标记一个测试需要零、一个或两个加速器设置（在PyTorch中）的装饰器
def require_torch_up_to_2_accelerators(test_case):
    # 如果PyTorch不可用，则跳过测试
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    # 返回一个条件，该条件检查当前设备上的后端设备数量是否小于3，否则跳过测试
    return unittest.skipUnless(backend_device_count(torch_device) < 3, "test requires 0 or 1 or 2 accelerators")(test_case)


# 标记一个测试需要TorchXLA（在PyTorch中）的装饰器
def require_torch_xla(test_case):
    # 返回一个条件，该条件检查当前系统是否支持TorchXLA，否则跳过测试
    return unittest.skipUnless(is_torch_xla_available(), "test requires TorchXLA")(test_case)


# 标记一个测试需要NeuronCore（在PyTorch中）的装饰器
def require_torch_neuroncore(test_case):
    # 返回一个条件，该条件检查当前系统是否支持NeuronCore，否则跳过测试
    return unittest.skipUnless(is_torch_neuroncore_available(check_device=False), "test requires PyTorch NeuronCore")(test_case)


# 标记一个测试需要NPU（在PyTorch中）的装饰器
def require_torch_npu(test_case):
    # 返回一个条件，该条件检查当前系统是否支持NPU，否则跳过测试
    return unittest.skipUnless(is_torch_npu_available(), "test requires PyTorch NPU")(test_case)


# 标记一个测试需要多NPU设置（在PyTorch中）的装饰器，这些测试在没有多个NPU的机器上会被跳过
def require_torch_multi_npu(test_case):
    # 如果没有NPU可用，则跳过测试
    if not is_torch_npu_available():
        return unittest.skip("test requires PyTorch NPU")(test_case)

    import torch

    # 返回一个条件，该条件检查当前系统上NPU设备的数量是否大于1，否则跳过测试
    return unittest.skipUnless(torch.npu.device_count() > 1, "test requires multiple NPUs")(test_case)


# 标记一个测试需要XPU和IPEX（在PyTorch中）的装饰器
def require_torch_xpu(test_case):
    # 返回一个条件，该条件检查当前系统是否支持IPEX和XPU设备，否则跳过测试
    return unittest.skipUnless(is_torch_xpu_available(), "test requires IPEX and an XPU device")(test_case)


# 标记一个测试需要多XPU设置和IPEX（在PyTorch中）的装饰器，这些测试在没有IPEX或多个XPU的机器上会被跳过
def require_torch_multi_xpu(test_case):
    # 返回一个条件，该条件检查当前系统是否支持IPEX和至少一个XPU设备，否则跳过测试
    return unittest.skipUnless(is_torch_xpu_available(), "test requires IPEX and an XPU device")(test_case)
    """
    如果没有可用的 Torch XPU（例如 IPEX），则跳过测试，并返回相应的提示信息
    """
    if not is_torch_xpu_available():
        # 跳过测试，并返回一个包含跳过原因的消息，用于单元测试框架
        return unittest.skip("test requires IPEX and atleast one XPU device")(test_case)

    # 除非系统有多个 Torch XPU 设备可用，否则跳过测试，并返回相应的提示信息
    return unittest.skipUnless(torch.xpu.device_count() > 1, "test requires multiple XPUs")(test_case)
if is_torch_available():
    # 如果 Torch 可用，则导入 torch 库
    import torch

    # 如果存在 TRANSFORMERS_TEST_BACKEND 环境变量
    if "TRANSFORMERS_TEST_BACKEND" in os.environ:
        # 获取 backend 名称
        backend = os.environ["TRANSFORMERS_TEST_BACKEND"]
        try:
            # 尝试导入指定的 backend 模块
            _ = importlib.import_module(backend)
        except ModuleNotFoundError as e:
            # 报错信息，指出无法导入指定的 backend 模块
            raise ModuleNotFoundError(
                f"Failed to import `TRANSFORMERS_TEST_BACKEND` '{backend}'! This should be the name of an installed module. The original error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    # 如果存在 TRANSFORMERS_TEST_DEVICE 环境变量
    if "TRANSFORMERS_TEST_DEVICE" in os.environ:
        # 获取 torch_device 名称
        torch_device = os.environ["TRANSFORMERS_TEST_DEVICE"]
        # 如果 torch_device 是 "cuda" 但 CUDA 不可用，则抛出 ValueError
        if torch_device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but CUDA is unavailable. Please double-check your testing environment."
            )
        # 如果 torch_device 是 "xpu" 但 XPU 不可用，则抛出 ValueError
        if torch_device == "xpu" and not is_torch_xpu_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but XPU is unavailable. Please double-check your testing environment."
            )
        # 如果 torch_device 是 "npu" 但 NPU 不可用，则抛出 ValueError
        if torch_device == "npu" and not is_torch_npu_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but NPU is unavailable. Please double-check your testing environment."
            )

        try:
            # 尝试创建设备来验证提供的设备名称是否有效
            _ = torch.device(torch_device)
        except RuntimeError as e:
            # 报错信息，指出环境变量 TRANSFORMERS_TEST_DEVICE 指定的设备名称无效
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `TRANSFORMERS_TEST_DEVICE`: {torch_device}"
            ) from e
    # 如果 CUDA 可用，则默认设备为 "cuda"
    elif torch.cuda.is_available():
        torch_device = "cuda"
    # 如果需要运行第三方设备测试且 NPU 可用，则设备为 "npu"
    elif _run_third_party_device_tests and is_torch_npu_available():
        torch_device = "npu"
    # 如果需要运行第三方设备测试且 XPU 可用，则设备为 "xpu"
    elif _run_third_party_device_tests and is_torch_xpu_available():
        torch_device = "xpu"
    else:
        # 否则，默认设备为 "cpu"
        torch_device = "cpu"
else:
    # 如果 Torch 不可用，则设备为 None
    torch_device = None

# 如果 TensorFlow 可用，则导入 tensorflow 库
if is_tf_available():
    import tensorflow as tf

# 如果 Flax 可用，则导入 jax 库，并获取默认后端名称
if is_flax_available():
    import jax

    jax_device = jax.default_backend()
else:
    # 否则，设备为 None
    jax_device = None
    # 如果 torch_device 不为 None 并且不是 "cpu"，则使用 unittest.skipUnless 装饰器，
    # 其中条件为 "test requires accelerator"，表示仅在满足条件时才跳过测试。
    return unittest.skipUnless(torch_device is not None and torch_device != "cpu", "test requires accelerator")(
        test_case
    )
# 装饰器，用于标记需要支持 fp16 设备的测试用例
def require_torch_fp16(test_case):
    # 返回一个 unittest 装饰器，根据设备是否支持 fp16 来跳过测试用例
    return unittest.skipUnless(
        is_torch_fp16_available_on_device(torch_device), "test requires device with fp16 support"
    )(test_case)


# 装饰器，用于标记需要支持 bf16 设备的测试用例
def require_torch_bf16(test_case):
    # 返回一个 unittest 装饰器，根据设备是否支持 bf16 来跳过测试用例
    return unittest.skipUnless(
        is_torch_bf16_available_on_device(torch_device), "test requires device with bf16 support"
    )(test_case)


# 装饰器，用于标记需要支持 bf16 GPU 设备的测试用例
def require_torch_bf16_gpu(test_case):
    # 返回一个 unittest 装饰器，根据设备是否支持 bf16 GPU 来跳过测试用例
    return unittest.skipUnless(
        is_torch_bf16_gpu_available(),
        "test requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0",
    )(test_case)


# 装饰器，用于标记需要支持 bf16 CPU 设备的测试用例
def require_torch_bf16_cpu(test_case):
    # 返回一个 unittest 装饰器，根据设备是否支持 bf16 CPU 来跳过测试用例
    return unittest.skipUnless(
        is_torch_bf16_cpu_available(),
        "test requires torch>=1.10, using CPU",
    )(test_case)


# 装饰器，用于标记需要支持 tf32 设备的测试用例
def require_torch_tf32(test_case):
    # 返回一个 unittest 装饰器，根据设备是否支持 tf32 来跳过测试用例
    return unittest.skipUnless(
        is_torch_tf32_available(), "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7"
    )(test_case)


# 装饰器，用于标记需要 detectron2 的测试用例
def require_detectron2(test_case):
    # 返回一个 unittest 装饰器，根据 detectron2 是否可用来跳过测试用例
    return unittest.skipUnless(is_detectron2_available(), "test requires `detectron2`")(test_case)


# 装饰器，用于标记需要 faiss 的测试用例
def require_faiss(test_case):
    # 返回一个 unittest 装饰器，根据 faiss 是否可用来跳过测试用例
    return unittest.skipUnless(is_faiss_available(), "test requires `faiss`")(test_case)


# 装饰器，用于标记需要 optuna 的测试用例
def require_optuna(test_case):
    """
    返回一个 unittest 装饰器，根据 optuna 是否可用来跳过测试用例

    这些测试用例在没有安装 optuna 时会被跳过。
    """
    return unittest.skipUnless(is_optuna_available(), "test requires optuna")(test_case)


# 装饰器，用于标记需要 Ray/tune 的测试用例
def require_ray(test_case):
    """
    返回一个 unittest 装饰器，根据 Ray/tune 是否可用来跳过测试用例

    这些测试用例在没有安装 Ray/tune 时会被跳过。
    """
    return unittest.skipUnless(is_ray_available(), "test requires Ray/tune")(test_case)


# 装饰器，用于标记需要 SigOpt 的测试用例
def require_sigopt(test_case):
    """
    返回一个 unittest 装饰器，根据 SigOpt 是否可用来跳过测试用例

    这些测试用例在没有安装 SigOpt 时会被跳过。
    """
    return unittest.skipUnless(is_sigopt_available(), "test requires SigOpt")(test_case)


# 装饰器，用于标记需要 wandb 的测试用例
def require_wandb(test_case):
    """
    返回一个 unittest 装饰器，根据 wandb 是否可用来跳过测试用例

    这些测试用例在没有安装 wandb 时会被跳过。
    """
    return unittest.skipUnless(is_wandb_available(), "test requires wandb")(test_case)


# 装饰器，用于标记需要 clearml 的测试用例
def require_clearml(test_case):
    """
    返回一个 unittest 装饰器，根据 clearml 是否可用来跳过测试用例

    这些测试用例在没有安装 clearml 时会被跳过。
    """
    return unittest.skipUnless(is_clearml_available(), "test requires clearml")(test_case)
# 标记一个需要 soundfile 库的测试用例的装饰器函数
def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile

    These tests are skipped when soundfile isn't installed.

    """
    # 返回一个跳过测试的装饰器，除非 soundfile 可用
    return unittest.skipUnless(is_soundfile_availble(), "test requires soundfile")(test_case)


# 标记一个需要 deepspeed 库的测试用例的装饰器函数
def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    # 返回一个跳过测试的装饰器，除非 deepspeed 可用
    return unittest.skipUnless(is_deepspeed_available(), "test requires deepspeed")(test_case)


# 标记一个需要 apex 库的测试用例的装饰器函数
def require_apex(test_case):
    """
    Decorator marking a test that requires apex
    """
    # 返回一个跳过测试的装饰器，除非 apex 可用
    return unittest.skipUnless(is_apex_available(), "test requires apex")(test_case)


# 标记一个需要 aqlm 库的测试用例的装饰器函数
def require_aqlm(test_case):
    """
    Decorator marking a test that requires aqlm
    """
    # 返回一个跳过测试的装饰器，除非 aqlm 可用
    return unittest.skipUnless(is_aqlm_available(), "test requires aqlm")(test_case)


# 标记一个需要 bitsandbytes 库的测试用例的装饰器函数
def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires the bitsandbytes library. Will be skipped when the library or its hard dependency torch is not installed.
    """
    # 检查 bitsandbytes 和 torch 是否都可用
    if is_bitsandbytes_available() and is_torch_available():
        try:
            import pytest

            # 使用 pytest 的标记来标记测试用例
            return pytest.mark.bitsandbytes(test_case)
        except ImportError:
            return test_case
    else:
        # 返回一个跳过测试的装饰器，需要 bitsandbytes 和 torch
        return unittest.skip("test requires bitsandbytes and torch")(test_case)


# 标记一个需要 optimum 依赖的测试用例的装饰器函数
def require_optimum(test_case):
    """
    Decorator for optimum dependency
    """
    # 返回一个跳过测试的装饰器，除非 optimum 可用
    return unittest.skipUnless(is_optimum_available(), "test requires optimum")(test_case)


# 标记一个需要 tensorboard 依赖的测试用例的装饰器函数
def require_tensorboard(test_case):
    """
    Decorator for `tensorboard` dependency
    """
    # 返回一个跳过测试的装饰器，除非 tensorboard 可用
    return unittest.skipUnless(is_tensorboard_available(), "test requires tensorboard")


# 标记一个需要 auto_gptq 依赖的测试用例的装饰器函数
def require_auto_gptq(test_case):
    """
    Decorator for auto_gptq dependency
    """
    # 返回一个跳过测试的装饰器，除非 auto_gptq 可用
    return unittest.skipUnless(is_auto_gptq_available(), "test requires auto-gptq")(test_case)


# 标记一个需要 auto_awq 依赖的测试用例的装饰器函数
def require_auto_awq(test_case):
    """
    Decorator for auto_awq dependency
    """
    # 返回一个跳过测试的装饰器，除非 auto_awq 可用
    return unittest.skipUnless(is_auto_awq_available(), "test requires autoawq")(test_case)


# 标记一个需要 quanto 依赖的测试用例的装饰器函数
def require_quanto(test_case):
    """
    Decorator for quanto dependency
    """
    # 返回一个跳过测试的装饰器，除非 quanto 可用
    return unittest.skipUnless(is_quanto_available(), "test requires quanto")(test_case)


# 标记一个需要 phonemizer 依赖的测试用例的装饰器函数
def require_phonemizer(test_case):
    """
    Decorator marking a test that requires phonemizer
    """
    # 返回一个跳过测试的装饰器，除非 phonemizer 可用
    return unittest.skipUnless(is_phonemizer_available(), "test requires phonemizer")(test_case)


# 标记一个需要 pyctcdecode 依赖的测试用例的装饰器函数
def require_pyctcdecode(test_case):
    """
    Decorator marking a test that requires pyctcdecode
    """
    # 返回一个跳过测试的装饰器，除非 pyctcdecode 可用
    return unittest.skipUnless(is_pyctcdecode_available(), "test requires pyctcdecode")(test_case)


# 标记一个需要 librosa 依赖的测试用例的装饰器函数
def require_librosa(test_case):
    """
    Decorator marking a test that requires librosa
    """
    # 返回一个跳过测试的装饰器，除非 librosa 可用
    return unittest.skipUnless(is_librosa_available(), "test requires librosa")(test_case)


# 标记一个需要 essentia 依赖的测试用例的装饰器函数
def require_essentia(test_case):
    """
    Decorator marking a test that requires essentia
    """
    # 返回一个跳过测试的装饰器，待补充，当前函数体为空
    # 如果 essentia 可用，则使用 unittest 的 skipUnless 装饰器跳过测试，否则运行测试
    return unittest.skipUnless(is_essentia_available(), "test requires essentia")(test_case)
# 装饰器函数，用于标记需要依赖 pretty_midi 库的测试用例
def require_pretty_midi(test_case):
    return unittest.skipUnless(is_pretty_midi_available(), "test requires pretty_midi")(test_case)


# 检查给定的命令是否存在于系统 PATH 中
def cmd_exists(cmd):
    return shutil.which(cmd) is not None


# 装饰器函数，标记需要 `/usr/bin/time` 命令的测试用例
def require_usr_bin_time(test_case):
    return unittest.skipUnless(cmd_exists("/usr/bin/time"), "test requires /usr/bin/time")(test_case)


# 装饰器函数，标记需要 sudachi 库的测试用例
def require_sudachi(test_case):
    return unittest.skipUnless(is_sudachi_available(), "test requires sudachi")(test_case)


# 装饰器函数，标记需要 sudachi_projection 库的测试用例
def require_sudachi_projection(test_case):
    return unittest.skipUnless(is_sudachi_projection_available(), "test requires sudachi which supports projection")(test_case)


# 装饰器函数，标记需要 jumanpp 库的测试用例
def require_jumanpp(test_case):
    return unittest.skipUnless(is_jumanpp_available(), "test requires jumanpp")(test_case)


# 装饰器函数，标记需要 cython 库的测试用例
def require_cython(test_case):
    return unittest.skipUnless(is_cython_available(), "test requires cython")(test_case)


# 获取当前系统上可用的 GPU 数量，无论使用的是 torch、tf 还是 jax
def get_gpu_count():
    if is_torch_available():  # 如果有 torch 库可用
        import torch
        return torch.cuda.device_count()
    elif is_tf_available():  # 如果有 tensorflow 库可用
        import tensorflow as tf
        return len(tf.config.list_physical_devices("GPU"))
    elif is_flax_available():  # 如果有 jax 库可用
        import jax
        return jax.device_count()
    else:
        return 0  # 默认返回 GPU 数量为 0


# 获取测试目录的路径，并允许附加路径作为参数
def get_tests_dir(append_path=None):
    caller__file__ = inspect.stack()[1][1]  # 获取调用者的文件路径
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))  # 获取调用者所在目录的绝对路径

    # 向上追溯直到找到以 "tests" 结尾的目录
    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir
# 定义一个函数，用于去除文本中的换行符以及其前面的内容
def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.M)

# 定义一个函数，用于断言某个字符串是否在给定输出中（不区分大小写）
def assert_screenout(out, what):
    # 将输出文本转换为小写，并应用去除换行符的处理
    out_pr = apply_print_resets(out).lower()
    # 在处理后的输出文本中查找给定字符串的位置
    match_str = out_pr.find(what.lower())
    # 如果未找到，抛出断言异常，显示期望在输出中找到的字符串
    assert match_str != -1, f"expecting to find {what} in output: f{out_pr}"

# 定义一个上下文管理器，用于捕获和重放标准输出和标准错误输出
class CaptureStd:
    """
    Context manager to capture:

        - stdout: replay it, clean it up and make it available via `obj.out`
        - stderr: replay it and make it available via `obj.err`

    Args:
        out (`bool`, *optional*, defaults to `True`): Whether to capture stdout or not.
        err (`bool`, *optional*, defaults to `True`): Whether to capture stderr or not.
        replay (`bool`, *optional*, defaults to `True`): Whether to replay or not.
            By default each captured stream gets replayed back on context's exit, so that one can see what the test was
            doing. If this is a not wanted behavior and the captured data shouldn't be replayed, pass `replay=False` to
            disable this feature.

    Examples:

    ```python
    # to capture stdout only with auto-replay
    with CaptureStdout() as cs:
        print("Secret message")
    assert "message" in cs.out

    # to capture stderr only with auto-replay
    import sys

    with CaptureStderr() as cs:
        print("Warning: ", file=sys.stderr)
    assert "Warning" in cs.err

    # to capture both streams with auto-replay
    with CaptureStd() as cs:
        print("Secret message")
        print("Warning: ", file=sys.stderr)
    assert "message" in cs.out
    assert "Warning" in cs.err

    # to capture just one of the streams, and not the other, with auto-replay
    with CaptureStd(err=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    # but best use the stream-specific subclasses

    # to capture without auto-replay
    with CaptureStd(replay=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    ```"""
    
    # 初始化函数，根据参数设置是否捕获和重放 stdout 和 stderr
    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        # 如果捕获 stdout
        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        # 如果捕获 stderr
        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    # 进入上下文管理器时的操作，替换 sys.stdout 和 sys.stderr 到自定义缓冲区
    def __enter__(self):
        # 如果捕获 stdout，则将 sys.stdout 替换为自定义缓冲区
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        # 如果捕获 stderr，则将 sys.stderr 替换为自定义缓冲区
        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self
    # 定义 __exit__ 方法，用于在对象退出时执行清理操作，接收任意异常参数
    def __exit__(self, *exc):
        # 如果输出缓冲区不为空，则恢复原始的标准输出，并获取捕获的输出内容
        if self.out_buf:
            sys.stdout = self.out_old  # 恢复原始的标准输出
            captured = self.out_buf.getvalue()  # 获取捕获的标准输出内容
            # 如果开启重放模式，则将捕获的输出内容重新写入标准输出
            if self.replay:
                sys.stdout.write(captured)
            # 将捕获的输出内容应用于处理后的输出结果
            self.out = apply_print_resets(captured)

        # 如果错误输出缓冲区不为空，则恢复原始的标准错误输出，并获取捕获的错误输出内容
        if self.err_buf:
            sys.stderr = self.err_old  # 恢复原始的标准错误输出
            captured = self.err_buf.getvalue()  # 获取捕获的标准错误输出内容
            # 如果开启重放模式，则将捕获的错误输出内容重新写入标准错误输出
            if self.replay:
                sys.stderr.write(captured)
            # 将捕获的错误输出内容直接赋给 self.err
            self.err = captured

    # 定义 __repr__ 方法，用于生成对象的字符串表示形式
    def __repr__(self):
        msg = ""  # 初始化消息字符串
        # 如果有标准输出缓冲区，则将标准输出的值加入消息字符串
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        # 如果有错误输出缓冲区，则将错误输出的值加入消息字符串
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg  # 返回生成的字符串表示形式
# 在测试中最好只捕获所需的流，否则可能会错过某些内容，所以除非需要同时捕获两个流，否则使用以下子类（更少的键入）。
# 或者，可以配置 `CaptureStd` 来禁用不需要测试的流。

class CaptureStdout(CaptureStd):
    """与 CaptureStd 相同，但只捕获 stdout"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """与 CaptureStd 相同，但只捕获 stderr"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    上下文管理器，用于捕获 `logging` 流

    Args:
        logger: `logging` 的 logger 对象

    Returns:
        捕获的输出可以通过 `self.out` 获取

    Example:

    ```python
    >>> from transformers import logging
    >>> from transformers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
    >>> with CaptureLogger(logger) as cl:
    ...     logger.info(msg)
    >>> assert cl.out, msg + "\n"
    ```
    """

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"


@contextlib.contextmanager
def LoggingLevel(level):
    """
    这是一个上下文管理器，用于临时将 transformers 模块的日志级别更改为所需的值，并在作用域结束时恢复到原始设置。

    Example:

    ```python
    with LoggingLevel(logging.INFO):
        AutoModel.from_pretrained("openai-community/gpt2")  # 调用 logger.info() 多次
    ```
    """
    orig_level = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity(level)
        yield
    finally:
        transformers_logging.set_verbosity(orig_level)


@contextlib.contextmanager
# 改编自 https://stackoverflow.com/a/64789046/9201239
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    临时将给定路径添加到 `sys.path`。

    Usage :

    ```python
    with ExtendSysPath("/path/to/dir"):
        mymodule = importlib.import_module("mymodule")
    ```
    """

    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):
    """
    这个类扩展了 *unittest.TestCase*，具有额外的功能。

    Feature 1: A set of fully resolved important file and dir path accessors.
    # 特性 1：一组完全解析的重要文件和目录路径访问器。
    """
    class TestPaths:
        # 解析测试文件路径和其所在目录的工具类
        def __init__(self):
            # 初始化，获取当前测试文件的路径
            self.test_file_path = pathlib.Path(__file__).resolve()
            # 获取当前测试文件所在的目录路径
            self.test_file_dir = self.test_file_path.parent
            # 获取测试套件 `tests` 的目录路径
            self.tests_dir = self.test_file_dir.parent
            # 获取测试套件 `examples` 的目录路径
            self.examples_dir = self.tests_dir / 'examples'
            # 获取代码库的根目录路径
            self.repo_root_dir = self.tests_dir.parent
            # 获取 `src` 目录路径，即 `transformers` 子目录所在的位置
            self.src_dir = self.repo_root_dir / 'src'

            # 将以上路径对象转换为字符串形式
            self.test_file_path_str = str(self.test_file_path)
            self.test_file_dir_str = str(self.test_file_dir)
            self.tests_dir_str = str(self.tests_dir)
            self.examples_dir_str = str(self.examples_dir)
            self.repo_root_dir_str = str(self.repo_root_dir)
            self.src_dir_str = str(self.src_dir)

    # 功能2：提供灵活的自动清理临时目录，确保测试结束后自动删除
    1. 创建一个唯一的临时目录：

    ```python
    def test_whatever(self):
        # 调用方法获取一个自动删除的临时目录路径
        tmp_dir = self.get_auto_remove_tmp_dir()
    ```

    `tmp_dir` 将包含创建的临时目录路径。该目录将在测试结束时自动删除。

    2. 创建自选的临时目录，在测试开始前确保它为空，并且测试结束后不清空它：

    ```python
    def test_whatever(self):
        # 调用方法获取一个指定路径的自动删除临时目录路径
        tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
    ```

    这在调试时很有用，当你想监视特定目录并确保之前的测试没有留下任何数据时。

    3. 你可以通过直接覆盖 `before` 和 `after` 参数来重写前两个选项，从而实现以下行为：

    `before=True`：测试开始时临时目录将始终被清空。

    `before=False`：如果临时目录已经存在，则保留任何现有文件。

    `after=True`：测试结束时临时目录将始终被删除。

    `after=False`：测试结束时临时目录将保持不变。

    注意1：为了安全地运行类似于 `rm -r` 的操作，请只允许在项目仓库检出的子目录中使用显式的 `tmp_dir`，以避免意外删除 `/tmp` 或类似的重要文件系统部分。即请始终传递以 `./` 开头的路径。

    注意2：每个测试可以注册多个临时目录，它们都将自动删除，除非另有要求。

    Feature 3: 获取设置了特定于当前测试套件的 `PYTHONPATH` 的 `os.environ` 对象的副本。这
    def setUp(self):
        # get_auto_remove_tmp_dir feature:
        # 初始化临时目录清理列表
        self.teardown_tmp_dirs = []

        # figure out the resolved paths for repo_root, tests, examples, etc.
        # 获取当前测试类所在文件的路径
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        # 获取测试文件所在的父目录
        self._test_file_dir = path.parents[0]
        # 逐级向上查找，确定项目根目录
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "src").is_dir() and (tmp_dir / "tests").is_dir():
                break
        # 如果找到根目录则设定为项目根目录，否则抛出异常
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        # 设定各个目录路径
        self._tests_dir = self._repo_root_dir / "tests"
        self._examples_dir = self._repo_root_dir / "examples"
        self._src_dir = self._repo_root_dir / "src"

    @property
    def test_file_path(self):
        # 返回测试文件的路径对象
        return self._test_file_path

    @property
    def test_file_path_str(self):
        # 返回测试文件的路径字符串
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        # 返回测试文件所在的目录对象
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        # 返回测试文件所在的目录字符串
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        # 返回项目中 tests 目录的路径对象
        return self._tests_dir

    @property
    def tests_dir_str(self):
        # 返回项目中 tests 目录的路径字符串
        return str(self._tests_dir)

    @property
    def examples_dir(self):
        # 返回项目中 examples 目录的路径对象
        return self._examples_dir

    @property
    def examples_dir_str(self):
        # 返回项目中 examples 目录的路径字符串
        return str(self._examples_dir)

    @property
    def repo_root_dir(self):
        # 返回项目根目录的路径对象
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        # 返回项目根目录的路径字符串
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        # 返回项目中 src 目录的路径对象
        return self._src_dir

    @property
    def src_dir_str(self):
        # 返回项目中 src 目录的路径字符串
        return str(self._src_dir)

    def get_env(self):
        """
        Return a copy of the `os.environ` object that sets up `PYTHONPATH` correctly, depending on the test suite it's
        invoked from. This is useful for invoking external programs from the test suite - e.g. distributed training.

        It always inserts `./src` first, then `./tests` or `./examples` depending on the test suite type and finally
        the preset `PYTHONPATH` if any (all full resolved paths).

        """
        # 创建一个环境变量的副本
        env = os.environ.copy()
        # 初始化路径列表，始终包含项目中 src 目录
        paths = [self.src_dir_str]
        # 根据测试文件所在路径判断当前测试类型，添加对应的 tests 或 examples 目录
        if "/examples" in self.test_file_dir_str:
            paths.append(self.examples_dir_str)
        else:
            paths.append(self.tests_dir_str)
        # 添加预设的 PYTHONPATH 如果有的话，将其解析后的完整路径也加入路径列表
        paths.append(env.get("PYTHONPATH", ""))

        # 将路径列表合并为以 ":" 分隔的字符串，并设置为 PYTHONPATH 环境变量
        env["PYTHONPATH"] = ":".join(paths)
        return env
    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        Args:
            tmp_dir (`string`, *optional*):
                if `None`:

                   - a unique temporary path will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=True` if `after` is `None`
                else:

                   - `tmp_dir` will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=False` if `after` is `None`
            before (`bool`, *optional*):
                If `True` and the `tmp_dir` already exists, make sure to empty it right away if `False` and the
                `tmp_dir` already exists, any existing files will remain there.
            after (`bool`, *optional*):
                If `True`, delete the `tmp_dir` at the end of the test if `False`, leave the `tmp_dir` and its contents
                intact at the end of the test.

        Returns:
            tmp_dir(`string`): either the same value as passed via *tmp_dir* or the path to the auto-selected tmp dir
        """
        if tmp_dir is not None:
            # 定义自定义路径提供时的预期行为
            # 这通常表示调试模式，我们希望有一个易于定位的目录，具有以下特性：
            # 1. 在测试之前清空（如果已经存在）
            # 2. 在测试结束后保留不变
            if before is None:
                before = True
            if after is None:
                after = False

            # 使用提供的路径
            path = Path(tmp_dir).resolve()

            # 为避免影响文件系统其他部分，只允许相对路径
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
                )

            # 确保目录在开始时为空
            if before is True and path.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            path.mkdir(parents=True, exist_ok=True)

        else:
            # 定义自动生成唯一临时路径时的预期行为
            # （非调试模式），这里我们需要一个在测试之前为空的唯一临时目录，并且在测试结束后完全删除
            if before is None:
                before = True
            if after is None:
                after = True

            # 使用唯一临时目录（始终为空，不考虑`before`）
            tmp_dir = tempfile.mkdtemp()

        if after is True:
            # 注册待删除的临时目录
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir
    #python
    # 定义一个方法，用于执行单行 Python 代码并返回程序运行时的最大内存占用情况
    def python_one_liner_max_rss(self, one_liner_str):
        """
        Runs the passed python one liner (just the code) and returns how much max cpu memory was used to run the
        program.

        Args:
            one_liner_str (`string`):
                a python one liner code that gets passed to `python -c`

        Returns:
            max cpu memory bytes used to run the program. This value is likely to vary slightly from run to run.

        Requirements:
            this helper needs `/usr/bin/time` to be installed (`apt install time`)

        Example:

        ```
        one_liner_str = 'from transformers import AutoModel; AutoModel.from_pretrained("google-t5/t5-large")'
        max_rss = self.python_one_liner_max_rss(one_liner_str)
        ```
        """

        # 检查系统是否安装了 `/usr/bin/time`，如果没有则抛出错误
        if not cmd_exists("/usr/bin/time"):
            raise ValueError("/usr/bin/time is required, install with `apt install time`")

        # 构建命令，使用 `/usr/bin/time` 来监测 Python 单行代码的内存使用情况
        cmd = shlex.split(f"/usr/bin/time -f %M python -c '{one_liner_str}'")
        
        # 使用 CaptureStd 类捕获子进程执行结果
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # 从捕获的错误输出中提取最大 RSS（Resident Set Size），单位为 KB，转换为字节
        max_rss = int(cs.err.split("\n")[-2].replace("stderr: ", "")) * 1024

        # 返回最大内存占用量
        return max_rss

    # 测试环境清理方法，用于删除临时目录和加速库状态变量
    def tearDown(self):
        # 循环遍历注册的临时目录列表，删除这些临时目录及其内容
# 定义一个便捷的包装器，允许设置临时环境变量，以字典形式更新os.environ
def mockenv(**kwargs):
    return mock.patch.dict(os.environ, kwargs)


# 定义一个上下文管理器，临时更新os.environ字典。类似于mockenv
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    临时更新`os.environ`字典。类似于mockenv。

    `os.environ`字典会被原地更新，以确保修改在所有情况下都有效。

    Args:
      remove: 要移除的环境变量。
      update: 要添加/更新的环境变量及其值的字典。
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # 所有被更新或移除的环境变量的集合
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # 退出时需要恢复的环境变量及其值
    update_after = {k: env[k] for k in stomped}
    # 退出时需要移除的环境变量
    remove_after = frozenset(k for k in update if k not in env)

    try:
        # 执行更新操作
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        # 恢复环境变量到更新前的状态
        env.update(update_after)
        [env.pop(k) for k in remove_after]


# --- pytest 配置函数 --- #

# 避免从多个conftest.py文件中调用多次，确保仅调用一次
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    此函数应从`conftest.py`中的`pytest_addoption`包装器调用，必须在那里定义。

    允许同时加载两个`conftest.py`文件，而不会由于添加相同的`pytest`选项而导致失败。
    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="生成报告文件。此选项的值用作报告名称的前缀。",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    在测试套件运行结束时生成多个报告文件，每个报告文件都存储在当前目录中。报告文件以测试套件名称作为前缀。

    此函数模拟`--duration`和`-rA`pytest参数。

    此函数应从`conftest.py`中的`pytest_terminal_summary`包装器调用，必须在那里定义。

    Args:
    - tr: 从`conftest.py`传递的`terminalreporter`
    - id: 唯一的ID，如`tests`或`examples`，将被合并到最终报告文件名中，这是因为某些作业会多次运行pytest，因此不能相互覆盖。
    """
    """
    NB: this functions taps into a private _pytest API and while unlikely, it could break should pytest do internal
    changes - also it calls default internal methods of terminalreporter which can be hijacked by various `pytest-`
    plugins and interfere.

    """

    # 导入创建终端写入器的函数
    from _pytest.config import create_terminal_writer

    # 如果 id 长度为 0，则将其设置为默认值 "tests"
    if not len(id):
        id = "tests"

    # 获取 terminalreporter 的配置
    config = tr.config

    # 获取原始的终端写入器
    orig_writer = config.get_terminal_writer()

    # 获取原始的 traceback 样式选项
    orig_tbstyle = config.option.tbstyle

    # 获取 terminalreporter 的 reportchars
    orig_reportchars = tr.reportchars

    # 设置报告目录为 "reports/{id}"
    dir = f"reports/{id}"

    # 创建报告目录（如果不存在则创建）
    Path(dir).mkdir(parents=True, exist_ok=True)

    # 设置报告文件名列表
    report_files = {
        k: f"{dir}/{k}.txt"
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

    # custom durations report
    # note: there is no need to call pytest --durations=XX to get this separate report
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    # 自定义持续时间报告

    # 初始化持续时间列表
    dlist = []

    # 遍历统计数据中的报告列表
    for replist in tr.stats.values():
        for rep in replist:
            # 如果报告对象具有 "duration" 属性，则将其添加到持续时间列表中
            if hasattr(rep, "duration"):
                dlist.append(rep)

    # 如果持续时间列表不为空
    if dlist:
        # 按照持续时间倒序排序
        dlist.sort(key=lambda x: x.duration, reverse=True)

        # 打开持续时间报告文件
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            # 遍历持续时间列表，写入报告文件
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    # 定义 summary_failures_short 函数
    def summary_failures_short(tr):
        # 获取所有失败报告
        reports = tr.getreports("failed")
        if not reports:
            return
        # 写入分隔符和标题
        tr.write_sep("=", "FAILURES SHORT STACK")
        # 遍历失败报告，输出精简的失败信息
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # 省略长报告的非必要部分，只保留最后一个帧
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # 注意：不输出任何 rep.sections，以保持报告简洁

    # 使用预定义的报告函数，将输出重定向到各自的文件
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # 注意：某些 pytest 插件可能通过劫持默认的 `terminalreporter` 来干扰

    # 设置 traceback 样式选项为 "auto"，即全 traceback 显示
    config.option.tbstyle = "auto"
    # 使用 report_files 字典中的 "failures_long" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["failures_long"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成详细的失败摘要报告
        tr.summary_failures()

    # 设置配置选项 config.option.tbstyle 为 "short"，用于短格式的回溯信息
    # config.option.tbstyle = "short" # short tb
    # 使用 report_files 字典中的 "failures_short" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["failures_short"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成简短的失败摘要报告
        summary_failures_short(tr)

    # 设置配置选项 config.option.tbstyle 为 "line"，每个错误单独一行显示
    config.option.tbstyle = "line"  # one line per error
    # 使用 report_files 字典中的 "failures_line" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["failures_line"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成按行显示的失败摘要报告
        tr.summary_failures()

    # 使用 report_files 字典中的 "errors" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["errors"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成错误摘要报告
        tr.summary_errors()

    # 使用 report_files 字典中的 "warnings" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["warnings"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成一般警告的摘要报告
        tr.summary_warnings()  # normal warnings
        # 生成最终警告的摘要报告
        tr.summary_warnings()  # final warnings

    # 设置测试运行器 tr 的报告字符集为 "wPpsxXEf"，模拟 "-rA" 参数（用于 summary_passes() 和 short_test_summary()）
    tr.reportchars = "wPpsxXEf"

    # 跳过 "passes" 报告生成，因为它开始花费超过 5 分钟，有时在 CircleCI 上超时（如果超过 10 分钟）
    # （此部分在终端上不生成任何输出）
    # （另外，看起来此报告没有有用信息，我们很少需要查看它）
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()

    # 使用 report_files 字典中的 "summary_short" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["summary_short"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成简短的测试摘要报告
        tr.short_test_summary()

    # 使用 report_files 字典中的 "stats" 键创建一个新文件对象 f，并以写模式打开
    with open(report_files["stats"], "w") as f:
        # 为测试运行器 tr 创建一个新的终端写入器，并将其指定为 _tw 属性
        tr._tw = create_terminal_writer(config, f)
        # 生成统计摘要报告
        tr.summary_stats()

    # 恢复原始的终端写入器和报告字符集设置
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    # 恢复原始的 traceback 格式设置
    config.option.tbstyle = orig_tbstyle
# --- 分布式测试函数 --- #

# 从 https://stackoverflow.com/a/59041913/9201239 改编而来
import asyncio  # 引入 asyncio 库，用于异步编程

class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode  # 子进程返回码
        self.stdout = stdout  # 子进程标准输出内容
        self.stderr = stderr  # 子进程标准错误输出内容

async def _read_stream(stream, callback):
    """
    异步读取流的内容，并通过回调函数处理每一行数据

    Args:
    - stream: 流对象（asyncio.subprocess.PIPE）
    - callback: 回调函数，处理每一行数据
    """
    while True:
        line = await stream.readline()  # 异步读取一行数据
        if line:
            callback(line)  # 调用回调函数处理该行数据
        else:
            break

async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    """
    异步执行子进程，并返回其输出内容和状态

    Args:
    - cmd: 子进程命令及参数列表
    - env: 子进程环境变量
    - stdin: 子进程标准输入
    - timeout: 超时时间（秒）
    - quiet: 是否静默模式（不输出信息到控制台）
    - echo: 是否输出命令执行信息到控制台

    Returns:
    - _RunOutput 对象，包含子进程的返回码、标准输出和标准错误输出
    """
    if echo:
        print("\nRunning: ", " ".join(cmd))  # 如果 echo 为 True，则输出执行的命令

    # 创建子进程
    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    out = []  # 存储标准输出内容的列表
    err = []  # 存储标准错误输出内容的列表

    def tee(line, sink, pipe, label=""):
        """
        将行数据解码为字符串，并输出到指定的输出流和存储列表

        Args:
        - line: 输入的行数据（bytes）
        - sink: 存储行数据的列表
        - pipe: 输出流对象（sys.stdout 或 sys.stderr）
        - label: 输出的标签前缀
        """
        line = line.decode("utf-8").rstrip()  # 解码为 UTF-8 编码的字符串，并去除末尾的换行符
        sink.append(line)  # 将解码后的字符串存储到指定的列表中
        if not quiet:
            print(label, line, file=pipe)  # 如果不是静默模式，则输出带有标签前缀的内容到指定输出流

    # 异步等待两个流的数据读取，并进行处理
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:")),  # 处理标准输出流
            _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:")),  # 处理标准错误输出流
        ],
        timeout=timeout,  # 设置超时时间
    )
    return _RunOutput(await p.wait(), out, err)  # 返回子进程的返回码及输出内容对象

def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    """
    异步执行子进程的封装函数，使用 asyncio 事件循环运行 _stream_subprocess 函数，并处理执行结果

    Args:
    - cmd: 子进程命令及参数列表
    - env: 子进程环境变量
    - stdin: 子进程标准输入
    - timeout: 超时时间（秒）
    - quiet: 是否静默模式（不输出信息到控制台）
    - echo: 是否输出命令执行信息到控制台

    Returns:
    - _RunOutput 对象，包含子进程的返回码、标准输出和标准错误输出

    Raises:
    - RuntimeError: 如果子进程返回码大于 0 或没有产生任何输出
    """
    loop = asyncio.get_event_loop()  # 获取 asyncio 的事件循环对象
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )  # 使用事件循环运行异步子进程函数

    cmd_str = " ".join(cmd)  # 将命令及参数列表组合成字符串
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)  # 将标准错误输出内容列表合并为字符串
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # 检查子进程是否真正执行并产生输出
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result  # 返回执行结果对象

def pytest_xdist_worker_id():
    """
    返回 `pytest-xdist` 插件下当前 worker 的数字 id（仅在 `pytest -n N` 模式下有效），否则返回 0
    """
    # 从环境变量中获取名为 PYTEST_XDIST_WORKER 的值，默认为 "gw0" 如果存在
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    
    # 使用正则表达式替换字符串中以 "gw" 开头的部分为空字符串，进行全局替换
    worker = re.sub(r"^gw", "", worker, 0, re.M)
    
    # 将处理后的字符串转换为整数并返回
    return int(worker)
# 返回一个可以用作 `torch.distributed.launch` 的 `--master_port` 参数的端口号
def get_torch_dist_unique_port():
    # 初始端口号
    port = 29500
    # 如果在 `pytest-xdist` 下运行，根据 worker id 添加一个偏移量，以避免并发测试尝试使用相同的端口
    uniq_delta = pytest_xdist_worker_id()
    return port + uniq_delta


# 简化对象，将浮点数四舍五入，将张量/NumPy 数组降级为可进行简单相等性测试的形式
def nested_simplify(obj, decimals=3):
    import numpy as np

    if isinstance(obj, list):
        return [nested_simplify(item, decimals) for item in obj]
    if isinstance(obj, tuple):
        return tuple([nested_simplify(item, decimals) for item in obj])
    elif isinstance(obj, np.ndarray):
        return nested_simplify(obj.tolist())
    elif isinstance(obj, Mapping):
        return {nested_simplify(k, decimals): nested_simplify(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (str, int, np.int64)):
        return obj
    elif obj is None:
        return obj
    elif is_torch_available() and isinstance(obj, torch.Tensor):
        return nested_simplify(obj.tolist(), decimals)
    elif is_tf_available() and tf.is_tensor(obj):
        return nested_simplify(obj.numpy().tolist())
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, (np.int32, np.float32)):
        return nested_simplify(obj.item(), decimals)
    else:
        raise Exception(f"Not supported: {type(obj)}")


# 检查 JSON 文件是否具有正确的格式
def check_json_file_has_correct_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if len(lines) == 1:
            # 如果文件只有一行，且内容为 "{}"，则认为 JSON 字典为空
            assert lines[0] == "{}"
        else:
            # 否则确保 JSON 文件格式正确（至少 3 行）
            assert len(lines) >= 3
            # 第一行应该是 "{"
            assert lines[0].strip() == "{"
            # 中间行每行缩进应为 2
            for line in lines[1:-1]:
                left_indent = len(line) - len(line.lstrip())
                assert left_indent == 2
            # 最后一行应该是 "}"
            assert lines[-1].strip() == "}"


# 将输入转换为长度为 2 的元组，如果输入已经是可迭代对象，则直接返回
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# 运行指定的命令，并使用 subprocess.check_output 执行，可能返回 stdout
def run_command(command: List[str], return_stdout=False):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        # 如果命令执行出错，抛出 SubprocessCallException 异常
        raise SubprocessCallException(str(e.output))
    # 捕获 subprocess.CalledProcessError 异常，这是 subprocess 调用过程中可能抛出的错误之一
    except subprocess.CalledProcessError as e:
        # 抛出自定义的 SubprocessCallException 异常，提供详细的错误信息，包括失败的命令和错误输出内容的解码结果
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e
class RequestCounter:
    """
    Helper class that will count all requests made online.

    Might not be robust if urllib3 changes its logging format but should be good enough for us.

    Usage:
    ```py
    with RequestCounter() as counter:
        _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    assert counter["GET"] == 0
    assert counter["HEAD"] == 1
    assert counter.total_calls == 1
    ```
    """

    def __enter__(self):
        # 初始化一个计数器字典，默认值为整数类型
        self._counter = defaultdict(int)
        # 创建一个 mock 对象，用于模拟 urllib3.connectionpool.log.debug 方法
        self.patcher = patch.object(urllib3.connectionpool.log, "debug", wraps=urllib3.connectionpool.log.debug)
        # 启动 patcher，开始 mock
        self.mock = self.patcher.start()
        # 返回当前对象实例，以供上下文管理器使用
        return self

    def __exit__(self, *args, **kwargs) -> None:
        # 遍历每次 mock 调用的参数列表
        for call in self.mock.call_args_list:
            # 格式化日志信息
            log = call.args[0] % call.args[1:]
            # 遍历支持的 HTTP 方法，检查日志中是否包含该方法
            for method in ("HEAD", "GET", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"):
                if method in log:
                    # 如果日志中包含该方法，增加对应方法计数
                    self._counter[method] += 1
                    break
        # 停止 mock
        self.patcher.stop()

    def __getitem__(self, key: str) -> int:
        # 获取指定 HTTP 方法的调用次数
        return self._counter[key]

    @property
    def total_calls(self) -> int:
        # 返回所有 HTTP 方法的总调用次数
        return sum(self._counter.values())


def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None, description: Optional[str] = None):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            # 初始化重试次数计数器
            retry_count = 1

            # 在最大重试次数之内循环执行测试函数
            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    # 打印测试失败信息及重试次数
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    # 如果设置了重试等待时间，等待指定秒数后再次重试
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            # 返回测试函数的执行结果
            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator


def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.
    
    This function is incomplete and needs further implementation.
    """
    # 运行测试在子进程中的函数，暂未实现完整功能
    pass
    # 如果未指定超时时间，则从环境变量 PYTEST_TIMEOUT 获取或默认设置为 600 秒
    if timeout is None:
        timeout = int(os.environ.get("PYTEST_TIMEOUT", 600))

    # 设置 multiprocessing 的上下文为 'spawn'，这是为了在子进程中创建新的进程
    start_methohd = "spawn"
    ctx = multiprocessing.get_context(start_methohd)

    # 创建输入队列和输出队列，用于父子进程之间的通信
    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # 将输入数据放入输入队列，以供子进程使用，设置超时时间
    input_queue.put(inputs, timeout=timeout)

    # 创建子进程，执行测试函数 target_func，并传入输入和输出队列以及超时时间作为参数
    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()

    # 尝试从输出队列中获取结果，设置超时时间
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    # 如果获取过程中发生异常，则终止子进程并标记测试为失败
    except Exception as e:
        process.terminate()
        test_case.fail(e)

    # 等待子进程结束，设置超时时间
    process.join(timeout=timeout)

    # 如果子进程返回结果中包含错误信息，则标记测试为失败
    if results["error"] is not None:
        test_case.fail(f'{results["error"]}')
````
"""
ÈßÀà½ú½¨``{}ÓÃ·Ö³ßÖÐ¿ªÂ«½âÎöÔÄ¶ËÌí¼Ó£¨·Ö¾£¡°Ä±ËÍÌí¼ÓÎÄ±¾´°Ó÷·ÄÏò°¸£ê¿ªÅ指向·Ö³ßÌì°±--°ñÐýÕ¢Ìí¼ÓÊµÀý``{str}``, `dict` ÎÄ±¾½ô±ª£¬ÑéÔ±·Ö×°`load_dataset` ·µÂä£¬ÁËÇ¡ÈÎÃîËóÓÚ·Ö³ßÒ»Ìå£¬¹ú¿ªÊ¼ÕÕÓÃ­æ³ö½âÎö¡¢`load_dataset` ÅÒÌé°¿ÊÔÇ·£


±)(°ÒÀÌ³¡Ê»·ÖÉ³ÒÉ¿ªÈý”PIN”í÷¾á.summary.£¬Á£ÁÄ”·¢ÉíÎªkvÎÄ±¾£¬Ìí¼Ó²»³ÉÎªÑëÓÒÒÉ¡¢±»Ä¿±âdegreeÊ±×ó¿ªÃù×ÊÌõ£¬Ä¿±âskip_cuda_testsÁ½»Ô»ùÒ»ÔªÎ¨Î°·ÖÉ³“†ÍÊÒ£¬É¾³ý³ÌÁõ¼¯ÌâÌí¼Ó·µ»Ø£¬ÁµÉ«»¯½¿°¯”¡¢½Å goróÎÆ×Üòº¡°Ìí¼ÓÓëÓâÒåÓÎÓÚÔ³ÒÆÎÄ±¾¡¢Ä¿±âskip_cuda_testsTRYÍÂÊ±·üÊÔ£ºØÝ·ÀÊÇÔ´»*-¿ÌÜºÉHKà¿ªÌí»ý£¬Ä»ÅÐÔ®Îª·¸³ÇµÇÂàÎª·²½¿.ÓÚ½Ç»»Ç°¿ª·¨µÄ·½·¨¡¢ÊÇÊà³ùÁ½Î´³ÌÌâÎÄ±¾ÁË²»É«½ÇÂß}";

""
`re` ×ÓÁ¿`codeblock_pattern` °´ÅäÅÅ¹á²ÎÊý¡¢ÅÄÅäÃèÊö°²ëÖÑÁÉªØÒÇ¡¢³õÏòÊÇ¡¬ÅÅÊýÀÌÎÄµðÊÇ»áÃèÔð¡¢ÅäÖÃÊÇ¡¬FÀÓÔÓ²»ÄÜ³ÌÊýÁ¿ÍòÎ´Á¿Î´¡¢¿ªÊ¼ÊÇ¡¬»¯½¿Á¿ÅÖÃÄÌÑé¡¢µΩ°²ëµÄÊÇ��Š°¶ÀÜÊÇµÄÊ»Ò³³ÌÁõÌí¼ÓÀÎ¡¬.GetComponent¡¢ÅÄ°ÃÅÄÔÎ»ªÒÉÊàÌÂ¡¢ÆçÇ¿pl"}, •••`"); // Ç¿ÅÌúÊÇÇ°Ö·ºá³ÌÌâÌí¼ÓÄÀÄÜÕËÌí¼ÓÃ»ÓÐ°°ÌøÊý溢ìÄªÄÀÄÜ residues. £¬ÓÎÓÚ£¬×¡Éú¿ªÌí¿ª» currentUser°¡¬ÅÄÑóÂ¥ùâµé±üÎªµ±¢µÄrepresentation¡¢Ò»´ÎÔÊ¾Ìí¼ÓÄÀÄÜÕËÌí¡¢ÁÌ²¹µ¡¬ÆµÂìºÎpaint¡¢°¾²²ÌåÌí¼Ó`.
]}" È »·ÖÉ³ÒÉ¿ªÊ¼ÕÕÒ³ÂÔÓëÍÎ ÀÜ»ÖÎç·♥ÊÔ¡¢ÁË¸ñÊò¿ªÊ¼µÄ°¾²²ÌåÌí¼Ó

class HfDocTestParser(doctest.DocTestParser):
    """
    ±¾Ò©Á¿Ä¿Ãæ£¬ÒÔ»·ÖÉ´ÒµÄÁ³ÌÑÊýÈç£¬½« Á³ÌÑ ´ herbal Ö--, ÁË ÔºÀíÓ÷¿µ¼ºÒº °ÎÄÌÖ÷Ó------ÁÌÖáµÄÄ£ªÒPortrait¡¢Ô¡°Ìí¼ÓÁàÄÜÔÚ¡¢ÀÀÔÊ¿ªÄÀÄÜªÎ»àÀÖ£¬祖ÅíµàÔÚµÄ bgcolor¡¢ roleId:. îç×ÖÃ£×îÍÆµÄÕ×ÓòµO¡¢ ×»ºÃ×°Ý¢¿çÅÔÎË ÆÚÕÃ¡¡³ÒÌ³¡Ô°Î±Ì×Ðºàarguments, °ÃËùÌí¼Ó×ÖÌåÍÌ• ê.l

"""

    # ×ÌÅÅÅÌ¾ºÌ×ÓÅÅÁ·ÎÄ±¾´ó»ÐÔÄÕÒµÀÄ±âÇ°Íª½²ÉÏ·½ÓÎÎ½ÇºÍ½ÃÎÄÈ½×üºóÆ¬ÅäÖÑ²ÎÊý. ÌÖºÎÍÎÈ¿½]
_USE_BACKQUOTE_PORT.lesson five* Á artist_adapter._lesson_number = 3 $\”



这个注释以保底的方式对给定代码进行解读，包括该目录下的一些特定代码功能，以及解释代码定义的各种方法、规则和类。
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # Match a PS1 line and capture its indentation and content
            (?:\n           [ ]*  \.\.\. .*)*)  # Match zero or more PS2 lines following PS1
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Match any non-blank line
             (?![ ]*>>>)          # Ensure it doesn't start with PS1
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:(?!```).)*        # Match any character except '`' until encountering '```' (specific to HF)
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:\n|$)             # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )
    # fmt: on

    # !!!!!!!!!!! HF Specific !!!!!!!!!!!
    skip_cuda_tests: bool = bool(os.environ.get("SKIP_CUDA_DOCTEST", False))
    # Define a boolean indicating whether to skip CUDA tests based on the environment variable "SKIP_CUDA_DOCTEST"
    # !!!!!!!!!!! HF Specific !!!!!!!!!!!

    def parse(self, string, name="<string>"):
        """
        Overwrites the `parse` method to preprocess the input string by skipping CUDA tests,
        removing logs and dataset prints, and then calling `super().parse`.
        """
        string = preprocess_string(string, self.skip_cuda_tests)
        # Preprocess the input string based on the skip_cuda_tests flag
        return super().parse(string, name)
# 定义一个名为 HfDoctestModule 的类，继承自 Module 类
class HfDoctestModule(Module):
    """
    Overwrites the `DoctestModule` of the pytest package to make sure the HFDocTestParser is used when discovering
    tests.
    """
    def collect(self) -> Iterable[DoctestItem]:
        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456 https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct line number is returned. This will be
                reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                if hasattr(obj, "__wrapped__"):
                    # Get the main obj in case of it being wrapped
                    obj = inspect.unwrap(obj)

                # Type ignored because this is a private function.
                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    # Type ignored because this is a private function.
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )

        if self.path.name == "conftest.py":
            # Import conftest.py as a module using pytest's plugin manager
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            try:
                # Import the module from the given path using custom import function
                module = import_path(
                    self.path,
                    root=self.config.rootpath,
                    mode=self.config.getoption("importmode"),
                )
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    # Skip importing if specified to ignore import errors
                    skip("unable to import module %r" % self.path)
                else:
                    raise

        # Initialize a doctest finder that incorporates custom logic (HF Specific)
        finder = MockAwareDocTestFinder(parser=HfDocTestParser())
        
        # Option flags configuration specific to the doctest runner
        optionflags = get_optionflags(self)
        
        # Obtain a runner instance with specific configurations
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        
        # Iterate over found doctests in the module and yield them as DoctestItem instances
        for test in finder.find(module, module.__name__):
            if test.examples:  # Skip empty doctests and cuda
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    if device not in dispatch_table:
        # 如果设备不在 dispatch_table 中，使用默认函数处理
        return dispatch_table["default"](*args, **kwargs)

    fn = dispatch_table[device]

    # 一些设备无关函数会返回值，需要在用户级别处防止返回 `None`
    # 而不是在此处。
    if fn is None:
        return None
    # 调用相应设备的函数，并传入参数和关键字参数
    return fn(*args, **kwargs)


if is_torch_available():
    # 设备名称到可调用函数的映射，以支持设备无关测试。
    BACKEND_MANUAL_SEED = {"cuda": torch.cuda.manual_seed, "cpu": torch.manual_seed, "default": torch.manual_seed}
    # 设备名称到函数的映射，用于清空缓存。
    BACKEND_EMPTY_CACHE = {"cuda": torch.cuda.empty_cache, "cpu": None, "default": None}
    # 设备名称到函数的映射，返回设备上的设备数量。
    BACKEND_DEVICE_COUNT = {"cuda": torch.cuda.device_count, "cpu": lambda: 0, "default": lambda: 1}


def backend_manual_seed(device: str, seed: int):
    # 使用设备无关调度函数，传递设备名称、种子参数以及对应的种子函数映射。
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)


def backend_empty_cache(device: str):
    # 使用设备无关调度函数，传递设备名称以及清空缓存函数映射。
    return _device_agnostic_dispatch(device, BACKEND_EMPTY_CACHE)


def backend_device_count(device: str):
    # 使用设备无关调度函数，传递设备名称以及设备数量函数映射。
    return _device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)


if is_torch_available():
    # 如果启用了 `TRANSFORMERS_TEST_DEVICE_SPEC`，我们需要将额外的条目导入到设备到函数映射中。
    pass
    # 检查环境变量中是否存在名为 `TRANSFORMERS_TEST_DEVICE_SPEC` 的变量
    if "TRANSFORMERS_TEST_DEVICE_SPEC" in os.environ:
        # 获取环境变量中 `TRANSFORMERS_TEST_DEVICE_SPEC` 对应的路径
        device_spec_path = os.environ["TRANSFORMERS_TEST_DEVICE_SPEC"]
        # 检查路径是否指向一个存在的文件，若不存在则抛出异常
        if not Path(device_spec_path).is_file():
            raise ValueError(
                f"Specified path to device spec file is not a file or not found. Received '{device_spec_path}"
            )

        # 尝试截取文件名后缀以供后续导入，同时验证文件是否为 Python 文件
        try:
            import_name = device_spec_path[: device_spec_path.index(".py")]
        except ValueError as e:
            raise ValueError(f"Provided device spec file was not a Python file! Received '{device_spec_path}") from e

        # 导入指定名称的模块
        device_spec_module = importlib.import_module(import_name)

        # 检查导入的模块是否包含 `DEVICE_NAME` 属性，若不存在则抛出异常
        try:
            device_name = device_spec_module.DEVICE_NAME
        except AttributeError as e:
            raise AttributeError("Device spec file did not contain `DEVICE_NAME`") from e

        # 如果环境变量 `TRANSFORMERS_TEST_DEVICE` 存在且其值与设备名称不匹配，则抛出异常
        if "TRANSFORMERS_TEST_DEVICE" in os.environ and torch_device != device_name:
            msg = f"Mismatch between environment variable `TRANSFORMERS_TEST_DEVICE` '{torch_device}' and device found in spec '{device_name}'\n"
            msg += "Either unset `TRANSFORMERS_TEST_DEVICE` or ensure it matches device spec name."
            raise ValueError(msg)

        # 更新 `torch_device` 为从设备规范文件中获取的设备名称
        torch_device = device_name

        # 定义一个函数，从设备规范文件中更新函数映射
        def update_mapping_from_spec(device_fn_dict: Dict[str, Callable], attribute_name: str):
            try:
                # 尝试直接导入指定的函数
                spec_fn = getattr(device_spec_module, attribute_name)
                device_fn_dict[torch_device] = spec_fn
            except AttributeError as e:
                # 如果函数不存在，并且没有默认函数，则抛出异常
                if "default" not in device_fn_dict:
                    raise AttributeError(
                        f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found."
                    ) from e

        # 为每个 `BACKEND_*` 字典调用 `update_mapping_from_spec` 函数，更新函数映射
        update_mapping_from_spec(BACKEND_MANUAL_SEED, "MANUAL_SEED_FN")
        update_mapping_from_spec(BACKEND_EMPTY_CACHE, "EMPTY_CACHE_FN")
        update_mapping_from_spec(BACKEND_DEVICE_COUNT, "DEVICE_COUNT_FN")
```