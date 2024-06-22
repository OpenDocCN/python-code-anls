# `.\transformers\testing_utils.py`

```py
# 导入模块和库
import collections  # 导入 collections 模块
import contextlib  # 导入 contextlib 模块
import doctest  # 导入 doctest 模块
import functools  # 导入 functools 模块
import importlib  # 导入 importlib 模块
import inspect  # 导入 inspect 模块
import logging  # 导入 logging 模块
import multiprocessing  # 导入 multiprocessing 模块
import os  # 导入 os 模块
import re  # 导入 re 模块
import shlex  # 导入 shlex 模块
import shutil  # 导入 shutil 模块
import subprocess  # 导入 subprocess 模块
import sys  # 导入 sys 模块
import tempfile  # 导入 tempfile 模块
import time  # 导入 time 模块
import unittest  # 导入 unittest 模块
from collections import defaultdict  # 从 collections 模块导入 defaultdict 类
from collections.abc import Mapping  # 从 collections.abc 模块导入 Mapping 类
from io import StringIO  # 从 io 模块导入 StringIO 类
from pathlib import Path  # 从 pathlib 模块导入 Path 类
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union  # 从 typing 模块导入若干类型
from unittest import mock  # 从 unittest 模块导入 mock 模块
from unittest.mock import patch  # 从 unittest.mock 模块导入 patch 类

import urllib3  # 导入 urllib3 模块

from transformers import logging as transformers_logging  # 从 transformers 包中导入 logging 模块
# 从本地 integrations 模块导入若干函数判断外部库是否可用
from .integrations import (
    is_clearml_available,
    is_optuna_available,
    is_ray_available,
    is_sigopt_available,
    is_tensorboard_available,
    is_wandb_available,
)
# 从本地 integrations.deepspeed 模块导入是否可用 deepspeed
from .integrations.deepspeed import is_deepspeed_available
# 从本地 utils 模块导入若干函数判断外部库是否可用
from .utils import (
    is_accelerate_available,
    is_apex_available,
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
    is_rjieba_available,
    is_safetensors_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_seqio_available,
    is_soundfile_availble,
    is_spacy_available,
    is_sudachi_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_available_on_device,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_fp16_available_on_device,
    is_torch_neuroncore_available,
)
    # 检查当前环境下是否可用 Torch 的 NPU 功能
    is_torch_npu_available,
    # 检查当前环境下是否可用 Torch 的 SDPA 功能
    is_torch_sdpa_available,
    # 检查当前环境下是否可用 Torch 的 TensorRT FX 功能
    is_torch_tensorrt_fx_available,
    # 检查当前环境下是否可用 Torch 的 TF32 功能
    is_torch_tf32_available,
    # 检查当前环境下是否可用 Torch 的 TPU 功能
    is_torch_tpu_available,
    # 检查当前环境下是否可用 Torch 的 XPU 功能
    is_torch_xpu_available,
    # 检查当前环境下是否可用 Torchaudio 库
    is_torchaudio_available,
    # 检查当前环境下是否可用 TorchDynamo 库
    is_torchdynamo_available,
    # 检查当前环境下是否可用 TorchVision 库
    is_torchvision_available,
    # 检查当前环境下是否可用 Vision 功能（可能是 TorchVision 的一部分）
    is_vision_available,
    # 将字符串转换为布尔值的函数，用于解析配置等
    strtobool,
# 检查是否可用加速器
if is_accelerate_available():
    # 如果可用，从加速器状态模块导入加速器状态和部分状态
    from accelerate.state import AcceleratorState, PartialState


# 检查是否可用 pytest
if is_pytest_available():
    # 如果可用，从 pytest 的 doctest 模块导入所需内容
    from _pytest.doctest import (
        Module,  # 导入 Module 类
        _get_checker,  # 导入 _get_checker 函数
        _get_continue_on_failure,  # 导入 _get_continue_on_failure 函数
        _get_runner,  # 导入 _get_runner 函数
        _is_mocked,  # 导入 _is_mocked 函数
        _patch_unwrap_mock_aware,  # 导入 _patch_unwrap_mock_aware 函数
        get_optionflags,  # 导入 get_optionflags 函数
        import_path,  # 导入 import_path 函数
    )
    # 从 pytest 的 outcomes 模块导入 skip 函数
    from _pytest.outcomes import skip
    # 从 pytest 导入 DoctestItem 类
    from pytest import DoctestItem
else:
    # 如果 pytest 不可用，将 Module 和 DoctestItem 设为 object 类的实例
    Module = object
    DoctestItem = object


# 定义一个小模型的标识符
SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
# 定义一个未知模型的标识符，用于测试
DUMMY_UNKNOWN_IDENTIFIER = "julien-c/dummy-unknown"
# 定义一个具有不同分词器的虚拟模型的标识符，用于测试
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"

# 用于测试 hub
# 定义用户
USER = "__DUMMY_TRANSFORMERS_USER__"
# 定义用于测试的端点
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

# 不是关键的，仅在受限的 CI 实例上可用
# 定义令牌
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"


# 从环境中解析布尔型标志
def parse_flag_from_env(key, default=False):
    try:
        # 尝试从环境中获取值
        value = os.environ[key]
    except KeyError:
        # 如果未设置键，则默认为 `default`
        _value = default
    else:
        # 如果设置了键，将其转换为 True 或 False
        try:
            _value = strtobool(value)
        except ValueError:
            # 支持更多的值，但让消息保持简单
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


# 从环境中解析整数
def parse_int_from_env(key, default=None):
    try:
        # 尝试从环境中获取值
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        # 如果设置了键，将其转换为整数
        try:
            _value = int(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be a int.")
    return _value


# 从环境中解析运行慢测试的标志
_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
# 从环境中解析运行 PyTorch + TensorFlow 交叉测试的标志
_run_pt_tf_cross_tests = parse_flag_from_env("RUN_PT_TF_CROSS_TESTS", default=True)
# 从环境中解析运行 PyTorch + Flax 交叉测试的标志
_run_pt_flax_cross_tests = parse_flag_from_env("RUN_PT_FLAX_CROSS_TESTS", default=True)
# 从环境中解析运行自定义分词器测试的标志
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
# 从环境中解析运行在 staging 上的测试的标志
_run_staging = parse_flag_from_env("HUGGINGFACE_CO_STAGING", default=False)
# 从环境中解析 TensorFlow GPU 内存限制
_tf_gpu_memory_limit = parse_int_from_env("TF_GPU_MEMORY_LIMIT", default=None)
# 从环境中解析运行管道测试的标志
_run_pipeline_tests = parse_flag_from_env("RUN_PIPELINE_TESTS", default=True)
# 从环境中解析运行工具测试的标志
_run_tool_tests = parse_flag_from_env("RUN_TOOL_TESTS", default=False)
# 从环境中解析运行第三方设备测试的标志
_run_third_party_device_tests = parse_flag_from_env("RUN_THIRD_PARTY_DEVICE_TESTS", default=False)


# 标记一个测试为控制 PyTorch 和 TensorFlow 之间交互的测试的装饰器
def is_pt_tf_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and TensorFlow.

    PT+TF tests are skipped by default and we can run only them by setting RUN_PT_TF_CROSS_TESTS environment variable
    to a truthy value and selecting the is_pt_tf_cross_test pytest mark.

    """
    # 如果不运行 PyTorch + TensorFlow 交叉测试或者 PyTorch 或 TensorFlow 不可用，则跳过测试
    if not _run_pt_tf_cross_tests or not is_torch_available() or not is_tf_available():
        return unittest.skip("test is PT+TF test")(test_case)
    else:
        # 尝试导入 pytest 模块，如果导入失败，则返回原始测试用例
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        # 如果导入成功，执行下面的代码块
        else:
            # 使用 pytest.mark.is_pt_tf_cross_test() 装饰器装饰测试用例，并返回装饰后的测试用例
            return pytest.mark.is_pt_tf_cross_test()(test_case)
# 用于装饰测试，标记测试为 PyTorch 和 Flax 之间交互的测试
def is_pt_flax_cross_test(test_case):
    # 如果不运行 PT+FLAX 测试或者 PyTorch 或 Flax 不可用，则跳过测试
    if not _run_pt_flax_cross_tests or not is_torch_available() or not is_flax_available():
        return unittest.skip("test is PT+FLAX test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中硬依赖于 pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pt_flax_cross_test()(test_case)


# 用于装饰测试，标记测试为分段测试
def is_staging_test(test_case):
    # 如果不运行分段测试，则跳过测试
    if not _run_staging:
        return unittest.skip("test is staging test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中硬依赖于 pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_staging_test()(test_case)


# 用于装饰测试，标记测试为管道测试。如果 RUN_PIPELINE_TESTS 设置为假值，则跳过测试。
def is_pipeline_test(test_case):
    if not _run_pipeline_tests:
        return unittest.skip("test is pipeline test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中硬依赖于 pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pipeline_test()(test_case)


# 用于装饰测试，标记测试为工具测试。如果 RUN_TOOL_TESTS 设置为假值，则跳过测试。
def is_tool_test(test_case):
    if not _run_tool_tests:
        return unittest.skip("test is a tool test")(test_case)
    else:
        try:
            import pytest  # 我们不需要在主库中硬依赖于 pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_tool_test()(test_case)


# 用于装饰测试，标记测试为慢速测试。慢速测试默认情况下会被跳过。设置 RUN_SLOW 环境变量为真值来运行它们。
def slow(test_case):
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


# 用于装饰测试，标记测试为太慢测试。慢速测试在被修复过程中会被跳过。没有测试应该标记为 "tooslow"，因为这些测试将不会被 CI 测试。
def tooslow(test_case):
    return unittest.skip("test is too slow")(test_case)


# 用于装饰测试，标记测试为自定义分词器测试。
def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.
    """
```  
    # 定义一个装饰器函数，用于跳过测试用例（unittest）。
    # Custom tokenizers 需要额外的依赖项，默认情况下会被跳过。将环境变量 RUN_CUSTOM_TOKENIZERS 设置为真值以运行它们。
    # 返回一个装饰器函数，如果 _run_custom_tokenizers 为真，则返回装饰过的测试用例，否则返回一个跳过测试的函数。
    return unittest.skipUnless(_run_custom_tokenizers, "test of custom tokenizers")(test_case)
# 标记需要 BeautifulSoup4 的测试用例的装饰器，当未安装 BeautifulSoup4 时跳过这些测试
def require_bs4(test_case):
    """
    Decorator marking a test that requires BeautifulSoup4. These tests are skipped when BeautifulSoup4 isn't installed.
    """
    return unittest.skipUnless(is_bs4_available(), "test requires BeautifulSoup4")(test_case)


# 标记需要 OpenCV 的测试用例的装饰器，当未安装 OpenCV 时跳过这些测试
def require_cv2(test_case):
    """
    Decorator marking a test that requires OpenCV.

    These tests are skipped when OpenCV isn't installed.

    """
    return unittest.skipUnless(is_cv2_available(), "test requires OpenCV")(test_case)


# 标记需要 Levenshtein 的测试用例的装饰器，当未安装 Levenshtein 时跳过这些测试
def require_levenshtein(test_case):
    """
    Decorator marking a test that requires Levenshtein.

    These tests are skipped when Levenshtein isn't installed.

    """
    return unittest.skipUnless(is_levenshtein_available(), "test requires Levenshtein")(test_case)


# 标记需要 NLTK 的测试用例的装饰器，当未安装 NLTK 时跳过这些测试
def require_nltk(test_case):
    """
    Decorator marking a test that requires NLTK.

    These tests are skipped when NLTK isn't installed.

    """
    return unittest.skipUnless(is_nltk_available(), "test requires NLTK")(test_case)


# 标记需要 accelerate 的测试用例的装饰器，当未安装 accelerate 时跳过这些测试
def require_accelerate(test_case):
    """
    Decorator marking a test that requires accelerate. These tests are skipped when accelerate isn't installed.
    """
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


# 标记需要 fsdp 的测试用例的装饰器，当未安装 fsdp 时跳过这些测试
def require_fsdp(test_case, min_version: str = "1.12.0"):
    """
    Decorator marking a test that requires fsdp. These tests are skipped when fsdp isn't installed.
    """
    return unittest.skipUnless(is_fsdp_available(min_version), f"test requires torch version >= {min_version}")(
        test_case
    )


# 标记需要 g2p_en 的测试用例的装饰器，当未安装 g2p_en 时跳过这些测试
def require_g2p_en(test_case):
    """
    Decorator marking a test that requires g2p_en. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_g2p_en_available(), "test requires g2p_en")(test_case)


# 标记需要 safetensors 的测试用例的装饰器，当未安装 safetensors 时跳过这些测试
def require_safetensors(test_case):
    """
    Decorator marking a test that requires safetensors. These tests are skipped when safetensors isn't installed.
    """
    return unittest.skipUnless(is_safetensors_available(), "test requires safetensors")(test_case)


# 标记需要 rjieba 的测试用例的装饰器，当未安装 rjieba 时跳过这些测试
def require_rjieba(test_case):
    """
    Decorator marking a test that requires rjieba. These tests are skipped when rjieba isn't installed.
    """
    return unittest.skipUnless(is_rjieba_available(), "test requires rjieba")(test_case)


# 标记需要 jieba 的测试用例的装饰器，当未安装 jieba 时跳过这些测试
def require_jieba(test_case):
    """
    Decorator marking a test that requires jieba. These tests are skipped when jieba isn't installed.
    """
    return unittest.skipUnless(is_jieba_available(), "test requires jieba")(test_case)


# 标记需要 jinja 的测试用例的装饰器，当未安装 jinja 时跳过这些测试
def require_jinja(test_case):
    """
    Decorator marking a test that requires jinja. These tests are skipped when jinja isn't installed.
    """
    return unittest.skipUnless(is_jinja_available(), "test requires jinja")(test_case)


# 标记需要 tf2onnx 的测试用例的装饰器，当未安装 tf2onnx 时跳过这些测试
def require_tf2onnx(test_case):
    return unittest.skipUnless(is_tf2onnx_available(), "test requires tf2onnx")(test_case)


# 标记需要 onnx 的测试用例的装饰器，当未安装 onnx 时跳过这些测试
def require_onnx(test_case):
    # 如果 ONNX 可用，则跳过测试，否则返回一个测试用例
    return unittest.skipUnless(is_onnx_available(), "test requires ONNX")(test_case)
# 标记一个需要 Timm 的测试的装饰器
def require_timm(test_case):
    """
    Decorator marking a test that requires Timm.

    These tests are skipped when Timm isn't installed.

    """
    # 返回一个装饰器函数，检查 Timm 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_timm_available(), "test requires Timm")(test_case)


# 标记一个需要 NATTEN 的测试的装饰器
def require_natten(test_case):
    """
    Decorator marking a test that requires NATTEN.

    These tests are skipped when NATTEN isn't installed.

    """
    # 返回一个装饰器函数，检查 NATTEN 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_natten_available(), "test requires natten")(test_case)


# 标记一个需要 PyTorch 的测试的装饰器
def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    # 返回一个装饰器函数，检查 PyTorch 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


# 标记一个需要 Flash Attention 的测试的装饰器
def require_flash_attn(test_case):
    """
    Decorator marking a test that requires Flash Attention.

    These tests are skipped when Flash Attention isn't installed.

    """
    # 返回一个装饰器函数，检查 Flash Attention 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_flash_attn_2_available(), "test requires Flash Attention")(test_case)


# 标记一个需要 PyTorch's SDPA 的测试的装饰器
def require_torch_sdpa(test_case):
    """
    Decorator marking a test that requires PyTorch's SDPA.

    These tests are skipped when requirements are not met (torch version).
    """
    # 返回一个装饰器函数，检查 PyTorch's SDPA 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_torch_sdpa_available(), "test requires PyTorch SDPA")(test_case)


# 标记一个需要 PEFT 的测试的装饰器
def require_peft(test_case):
    """
    Decorator marking a test that requires PEFT.

    These tests are skipped when PEFT isn't installed.

    """
    # 返回一个装饰器函数，检查 PEFT 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_peft_available(), "test requires PEFT")(test_case)


# 标记一个需要 Torchvision 的测试的装饰器
def require_torchvision(test_case):
    """
    Decorator marking a test that requires Torchvision.

    These tests are skipped when Torchvision isn't installed.

    """
    # 返回一个装饰器函数，检查 Torchvision 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_torchvision_available(), "test requires Torchvision")(test_case)


# 标记一个需要 PyTorch 或 TensorFlow 的测试的装饰器
def require_torch_or_tf(test_case):
    """
    Decorator marking a test that requires PyTorch or TensorFlow.

    These tests are skipped when neither PyTorch nor TensorFlow is installed.

    """
    # 返回一个装饰器函数，检查 PyTorch 或 TensorFlow 是否可用，若都不可用则跳过测试
    return unittest.skipUnless(is_torch_available() or is_tf_available(), "test requires PyTorch or TensorFlow")(
        test_case
    )


# 标记一个需要 Intel Extension for PyTorch 的测试的装饰器
def require_intel_extension_for_pytorch(test_case):
    """
    Decorator marking a test that requires Intel Extension for PyTorch.

    These tests are skipped when Intel Extension for PyTorch isn't installed or it does not match current PyTorch
    version.

    """
    # 返回一个装饰器函数，检查 Intel Extension for PyTorch 是否可用，若不可用则跳过测试
    return unittest.skipUnless(
        is_ipex_available(),
        "test requires Intel Extension for PyTorch to be installed and match current PyTorch version, see"
        " https://github.com/intel/intel-extension-for-pytorch",
    )(test_case)


# 标记一个需要 TensorFlow probability 的测试的装饰器
def require_tensorflow_probability(test_case):
    """
    Decorator marking a test that requires TensorFlow probability.

    These tests are skipped when TensorFlow probability isn't installed.

    """
    # 返回一个装饰器函数，检查 TensorFlow probability 是否可用，若不可用则跳过测试
    return unittest.skipUnless(is_tensorflow_probability_available(), "test requires TensorFlow probability")(
        test_case
    )
    # 返回一个装饰器，用于跳过测试，除非 TensorFlow probability 可用
    return unittest.skipUnless(is_tensorflow_probability_available(), "test requires TensorFlow probability")(
        test_case
    )
# 标记一个需要 torchaudio 的测试用例的装饰器。当 torchaudio 未安装时，跳过这些测试。
def require_torchaudio(test_case):
    """
    Decorator marking a test that requires torchaudio. These tests are skipped when torchaudio isn't installed.
    """
    return unittest.skipUnless(is_torchaudio_available(), "test requires torchaudio")(test_case)

# 标记一个需要 TensorFlow 的测试用例的装饰器。当 TensorFlow 未安装时，跳过这些测试。
def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow. These tests are skipped when TensorFlow isn't installed.
    """
    return unittest.skipUnless(is_tf_available(), "test requires TensorFlow")(test_case)

# 标记一个需要 JAX & Flax 的测试用例的装饰器。当其中一个或两者未安装时，跳过这些测试。
def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax. These tests are skipped when one / both are not installed
    """
    return unittest.skipUnless(is_flax_available(), "test requires JAX & Flax")(test_case)

# 标记一个需要 SentencePiece 的测试用例的装饰器。当 SentencePiece 未安装时，跳过这些测试。
def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_sentencepiece_available(), "test requires SentencePiece")(test_case)

# 标记一个需要 Seqio 的测试用例的装饰器。当 Seqio 未安装时，跳过这些测试。
def require_seqio(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_seqio_available(), "test requires Seqio")(test_case)

# 标记一个需要 Scipy 的测试用例的装饰器。当 Scipy 未安装时，跳过这些测试。
def require_scipy(test_case):
    """
    Decorator marking a test that requires Scipy. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_scipy_available(), "test requires Scipy")(test_case)

# 标记一个需要 🤗 Tokenizers 的测试用例的装饰器。当 🤗 Tokenizers 未安装时，跳过这些测试。
def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers. These tests are skipped when 🤗 Tokenizers isn't installed.
    """
    return unittest.skipUnless(is_tokenizers_available(), "test requires tokenizers")(test_case)

# 标记一个需要 tensorflow_text 的测试用例的装饰器。当 tensorflow_text 未安装时，跳过这些测试。
def require_tensorflow_text(test_case):
    """
    Decorator marking a test that requires tensorflow_text. These tests are skipped when tensroflow_text isn't
    installed.
    """
    return unittest.skipUnless(is_tensorflow_text_available(), "test requires tensorflow_text")(test_case)

# 标记一个需要 keras_nlp 的测试用例的装饰器。当 keras_nlp 未安装时，跳过这些测试。
def require_keras_nlp(test_case):
    """
    Decorator marking a test that requires keras_nlp. These tests are skipped when keras_nlp isn't installed.
    """
    return unittest.skipUnless(is_keras_nlp_available(), "test requires keras_nlp")(test_case)

# 标记一个需要 pandas 的测试用例的装饰器。当 pandas 未安装时，跳过这些测试。
def require_pandas(test_case):
    """
    Decorator marking a test that requires pandas. These tests are skipped when pandas isn't installed.
    """
    return unittest.skipUnless(is_pandas_available(), "test requires pandas")(test_case)

# 标记一个需要 PyTesseract 的测试用例的装饰器。当 PyTesseract 未安装时，跳过这些测试。
def require_pytesseract(test_case):
    """
    Decorator marking a test that requires PyTesseract. These tests are skipped when PyTesseract isn't installed.
    """
    return unittest.skipUnless(is_pytesseract_available(), "test requires PyTesseract")(test_case)

# 标记一个需要 PyTorch 量化功能的测试用例的装饰器。暂时缺少了这个装饰器的具体注释。
def require_pytorch_quantization(test_case):
    """
    # 装饰器标记一个需要 PyTorch 量化工具包的测试。当 PyTorch 量化工具包未安装时，这些测试将被跳过。
    """
    # 使用 unittest.skipUnless() 函数装饰测试用例，当 is_pytorch_quantization_available() 函数返回 False 时跳过测试，
    # 并提供一条消息说明测试需要 PyTorch 量化工具包
    return unittest.skipUnless(is_pytorch_quantization_available(), "test requires PyTorch Quantization Toolkit")(
        test_case
    )
# 标记一个需要视觉依赖的测试用例的装饰器，当没有安装 torchaudio 时会跳过这些测试
def require_vision(test_case):
    return unittest.skipUnless(is_vision_available(), "test requires vision")(test_case)

# 标记一个需要 ftfy 的测试用例的装饰器，当没有安装 ftfy 时会跳过这些测试
def require_ftfy(test_case):
    return unittest.skipUnless(is_ftfy_available(), "test requires ftfy")(test_case)

# 标记一个需要 SpaCy 的测试用例的装饰器，当没有安装 SpaCy 时会跳过这些测试
def require_spacy(test_case):
    return unittest.skipUnless(is_spacy_available(), "test requires spacy")(test_case)

# 标记一个需要 decord 的测试用例的装饰器，当没有安装 decord 时会跳过这些测试
def require_decord(test_case):
    return unittest.skipUnless(is_decord_available(), "test requires decord")(test_case)

# 标记一个需要多 GPU 设置（在 PyTorch 中）的测试用例的装饰器，当机器没有多个 GPU 时会跳过这些测试
def require_torch_multi_gpu(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)

# 标记一个需要多加速器设置（在 PyTorch 中）的测试用例的装饰器，当机器没有多个加速器时会跳过这些测试
def require_torch_multi_accelerator(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    return unittest.skipUnless(backend_device_count(torch_device) > 1, "test requires multiple accelerators")(test_case)

# 标记一个需要 0 或 1 个 GPU 设置（在 PyTorch 中）的测试用例的装饰器
def require_torch_non_multi_gpu(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() < 2, "test requires 0 or 1 GPU")(test_case)

# 标记一个需要 0 或 1 个加速器设置（在 PyTorch 中）的测试用例的装饰器
def require_torch_non_multi_accelerator(test_case):
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    return unittest.skipUnless(backend_device_count(torch_device) < 2, "test requires 0 or 1 accelerator")(test_case)

# 标记一个需要 0 或 1 或 2 个 GPU 设置（在 PyTorch 中）的测试用例的装饰器
def require_torch_up_to_2_gpus(test_case):
    # 检查当前环境是否可用 PyTorch 框架
    if not is_torch_available():
        # 如果不可用，则跳过测试并返回相应的消息
        return unittest.skip("test requires PyTorch")(test_case)
    
    # 导入 PyTorch 框架
    import torch
    
    # 仅在当前 CUDA 设备数量小于 3 时执行测试，否则跳过测试并返回消息
    return unittest.skipUnless(torch.cuda.device_count() < 3, "test requires 0 or 1 or 2 GPUs")(test_case)
# 装饰器，标记一个测试需要最多两个加速器（在 PyTorch 中）
def require_torch_up_to_2_accelerators(test_case):
    # 如果 PyTorch 不可用，则跳过测试
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    # 除非当前设备加速器数量小于 3，否则跳过测试
    return unittest.skipUnless(backend_device_count(torch_device) < 3, "test requires 0 or 1 or 2 accelerators")(test_case)


# 装饰器，标记一个测试需要 TPU（在 PyTorch 中）
def require_torch_tpu(test_case):
    # 除非 PyTorch TPU 可用，否则跳过测试
    return unittest.skipUnless(is_torch_tpu_available(check_device=False), "test requires PyTorch TPU")(test_case)


# 装饰器，标记一个测试需要 NeuronCore（在 PyTorch 中）
def require_torch_neuroncore(test_case):
    # 除非 PyTorch NeuronCore 可用，否则跳过测试
    return unittest.skipUnless(is_torch_neuroncore_available(check_device=False), "test requires PyTorch NeuronCore")(test_case)


# 装饰器，标记一个测试需要 NPU（在 PyTorch 中）
def require_torch_npu(test_case):
    # 除非 PyTorch NPU 可用，否则跳过测试
    return unittest.skipUnless(is_torch_npu_available(), "test requires PyTorch NPU")(test_case)


# 装饰器，标记一个测试需要多个 NPU（在 PyTorch 中）。这些测试在没有多个 NPU 的机器上跳过。
def require_torch_multi_npu(test_case):
    # 如果 PyTorch NPU 不可用，则跳过测试
    if not is_torch_npu_available():
        return unittest.skip("test requires PyTorch NPU")(test_case)

    # 除非当前设备的 NPU 数量大于 1，否则跳过测试
    return unittest.skipUnless(torch.npu.device_count() > 1, "test requires multiple NPUs")(test_case)


# 装饰器，标记一个测试需要 XPU 和 IPEX。这些测试在未安装 Intel Extension for PyTorch 或其版本不匹配当前 PyTorch 版本时跳过。
def require_torch_xpu(test_case):
    # 除非 IPEX 和 XPU 设备可用，否则跳过测试
    return unittest.skipUnless(is_torch_xpu_available(), "test requires IPEX and an XPU device")(test_case)


# 装饰器，标记一个测试需要带有 IPEX 和至少一个 XPU 设备的多个 XPU 设置。这些测试在没有 IPEX 或多个 XPU 的机器上跳过。
def require_torch_multi_xpu(test_case):
    # 如果 IPEX 和 XPU 不可用，则跳过测试
    if not is_torch_xpu_available():
        return unittest.skip("test requires IPEX and atleast one XPU device")(test_case)

    # 除非当前设备的 XPU 数量大于 1，否则跳过测试
    return unittest.skipUnless(torch.xpu.device_count() > 1, "test requires multiple XPUs")(test_case)


# 如果 PyTorch 可用，则设置环境变量 CUDA_VISIBLE_DEVICES="" 以强制使用 CPU 模式
if is_torch_available():
    import torch
    # 检查环境变量中是否设置了名为 "TRANSFORMERS_TEST_BACKEND" 的变量
    if "TRANSFORMERS_TEST_BACKEND" in os.environ:
        # 如果设置了，则获取该变量的值
        backend = os.environ["TRANSFORMERS_TEST_BACKEND"]
        try:
            # 尝试动态导入该变量指定的模块
            _ = importlib.import_module(backend)
        except ModuleNotFoundError as e:
            # 如果导入失败，则抛出模块未找到的异常，并提供详细信息
            raise ModuleNotFoundError(
                f"Failed to import `TRANSFORMERS_TEST_BACKEND` '{backend}'! This should be the name of an installed module. The original error (look up to see its"
                f" traceback):\n{e}"
            ) from e
    
    # 检查环境变量中是否设置了名为 "TRANSFORMERS_TEST_DEVICE" 的变量
    if "TRANSFORMERS_TEST_DEVICE" in os.environ:
        # 如果设置了，则获取该变量的值作为 Torch 设备
        torch_device = os.environ["TRANSFORMERS_TEST_DEVICE"]
        try:
            # 尝试创建 Torch 设备，以验证提供的设备是否有效
            _ = torch.device(torch_device)
        except RuntimeError as e:
            # 如果创建设备时发生错误，则抛出运行时错误，并提供详细信息
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `TRANSFORMERS_TEST_DEVICE`: {torch_device}"
            ) from e
    # 如果环境变量中未设置测试设备，并且 CUDA 可用，则选择 CUDA 设备
    elif torch.cuda.is_available():
        torch_device = "cuda"
    # 如果第三方设备测试启用，并且 Torch NPU 可用，则选择 NPU 设备
    elif _run_third_party_device_tests and is_torch_npu_available():
        torch_device = "npu"
    # 如果第三方设备测试启用，并且 Torch XPU 可用，则选择 XPU 设备
    elif _run_third_party_device_tests and is_torch_xpu_available():
        torch_device = "xpu"
    # 如果以上条件都不满足，则选择 CPU 设备
    else:
        torch_device = "cpu"
else:
    # 如果没有其他设备可用，则将 torch_device 设为 None
    torch_device = None

# 如果 TensorFlow 可用，则导入 TensorFlow 库
if is_tf_available():
    import tensorflow as tf

# 如果 Flax 可用，则导入 JAX 库，并设置默认设备为当前设备
if is_flax_available():
    import jax
    # 获取默认的 JAX 后端设备
    jax_device = jax.default_backend()
else:
    # 否则将 jax_device 设为 None
    jax_device = None

# 以下为一系列装饰器函数，用于标记需要特定环境支持的测试用例

# 要求 TorchDynamo，需要 TorchDynamo 可用
def require_torchdynamo(test_case):
    """Decorator marking a test that requires TorchDynamo"""
    return unittest.skipUnless(is_torchdynamo_available(), "test requires TorchDynamo")(test_case)

# 要求 Torch-TensorRT FX，需要 Torch-TensorRT FX 可用
def require_torch_tensorrt_fx(test_case):
    """Decorator marking a test that requires Torch-TensorRT FX"""
    return unittest.skipUnless(is_torch_tensorrt_fx_available(), "test requires Torch-TensorRT FX")(test_case)

# 要求 Torch GPU，需要 CUDA 和 PyTorch 可用
def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)

# 要求 Torch 加速器，需要可用的加速器和 PyTorch
def require_torch_accelerator(test_case):
    """Decorator marking a test that requires an accessible accelerator and PyTorch."""
    return unittest.skipUnless(torch_device is not None and torch_device != "cpu", "test requires accelerator")(test_case)

# 要求 Torch fp16，需要设备支持 fp16
def require_torch_fp16(test_case):
    """Decorator marking a test that requires a device that supports fp16"""
    return unittest.skipUnless(
        is_torch_fp16_available_on_device(torch_device), "test requires device with fp16 support"
    )(test_case)

# 要求 Torch bf16，需要设备支持 bf16
def require_torch_bf16(test_case):
    """Decorator marking a test that requires a device that supports bf16"""
    return unittest.skipUnless(
        is_torch_bf16_available_on_device(torch_device), "test requires device with bf16 support"
    )(test_case)

# 要求 Torch bf16 GPU，需要 torch>=1.10，并且使用 Ampere GPU 或更新的架构，或 cuda>=11.0
def require_torch_bf16_gpu(test_case):
    """Decorator marking a test that requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0"""
    return unittest.skipUnless(
        is_torch_bf16_gpu_available(),
        "test requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0",
    )(test_case)

# 要求 Torch bf16 CPU，需要 torch>=1.10，并且使用 CPU
def require_torch_bf16_cpu(test_case):
    """Decorator marking a test that requires torch>=1.10, using CPU."""
    return unittest.skipUnless(
        is_torch_bf16_cpu_available(),
        "test requires torch>=1.10, using CPU",
    )(test_case)

# 要求 Torch tf32，需要 Ampere 或更新的 GPU 架构，cuda>=11 和 torch>=1.7
def require_torch_tf32(test_case):
    """Decorator marking a test that requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7."""
    return unittest.skipUnless(
        is_torch_tf32_available(), "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7"
    )(test_case)

# 要求 Detectron2，需要 detectron2 可用
def require_detectron2(test_case):
    """Decorator marking a test that requires detectron2."""
    return unittest.skipUnless(is_detectron2_available(), "test requires `detectron2`")(test_case)

# 要求 Faiss，需要 faiss 可用
def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    return unittest.skipUnless(is_faiss_available(), "test requires `faiss`")(test_case)

# 要求 Optuna，需要 optuna 可用
def require_optuna(test_case):
    """
    Decorator marking a test that requires optuna.

    These tests are skipped when optuna isn't installed.

    """
    # 如果 optuna 可用，则跳过测试，否则提示测试需要 optuna
    return unittest.skipUnless(is_optuna_available(), "test requires optuna")(test_case)
def require_ray(test_case):
    """
    Decorator marking a test that requires Ray/tune.

    These tests are skipped when Ray/tune isn't installed.

    """
    # 返回一个装饰器，用于标记需要 Ray/tune 的测试用例
    return unittest.skipUnless(is_ray_available(), "test requires Ray/tune")(test_case)


def require_sigopt(test_case):
    """
    Decorator marking a test that requires SigOpt.

    These tests are skipped when SigOpt isn't installed.

    """
    # 返回一个装饰器，用于标记需要 SigOpt 的测试用例
    return unittest.skipUnless(is_sigopt_available(), "test requires SigOpt")(test_case)


def require_wandb(test_case):
    """
    Decorator marking a test that requires wandb.

    These tests are skipped when wandb isn't installed.

    """
    # 返回一个装饰器，用于标记需要 wandb 的测试用例
    return unittest.skipUnless(is_wandb_available(), "test requires wandb")(test_case)


def require_clearml(test_case):
    """
    Decorator marking a test requires clearml.

    These tests are skipped when clearml isn't installed.

    """
    # 返回一个装饰器，用于标记需要 clearml 的测试用例
    return unittest.skipUnless(is_clearml_available(), "test requires clearml")(test_case)


def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile

    These tests are skipped when soundfile isn't installed.

    """
    # 返回一个装饰器，用于标记需要 soundfile 的测试用例
    return unittest.skipUnless(is_soundfile_availble(), "test requires soundfile")(test_case)


def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    # 返回一个装饰器，用于标记需要 deepspeed 的测试用例
    return unittest.skipUnless(is_deepspeed_available(), "test requires deepspeed")(test_case)


def require_apex(test_case):
    """
    Decorator marking a test that requires apex
    """
    # 返回一个装饰器，用于标记需要 apex 的测试用例
    return unittest.skipUnless(is_apex_available(), "test requires apex")(test_case)


def require_bitsandbytes(test_case):
    """
    Decorator for bits and bytes (bnb) dependency
    """
    # 返回一个装饰器，用于标记需要 bnb 的测试用例
    return unittest.skipUnless(is_bitsandbytes_available(), "test requires bnb")(test_case)


def require_optimum(test_case):
    """
    Decorator for optimum dependency
    """
    # 返回一个装饰器，用于标记需要 optimum 的测试用例
    return unittest.skipUnless(is_optimum_available(), "test requires optimum")(test_case)


def require_tensorboard(test_case):
    """
    Decorator for `tensorboard` dependency
    """
    # 返回一个装饰器，用于标记需要 tensorboard 的测试用例
    return unittest.skipUnless(is_tensorboard_available(), "test requires tensorboard")


def require_auto_gptq(test_case):
    """
    Decorator for auto_gptq dependency
    """
    # 返回一个装饰器，用于标记需要 auto_gptq 的测试用例
    return unittest.skipUnless(is_auto_gptq_available(), "test requires auto-gptq")(test_case)


def require_auto_awq(test_case):
    """
    Decorator for auto_awq dependency
    """
    # 返回一个装饰器，用于标记需要 autoawq 的测试用例
    return unittest.skipUnless(is_auto_awq_available(), "test requires autoawq")(test_case)


def require_phonemizer(test_case):
    """
    Decorator marking a test that requires phonemizer
    """
    # 返回一个装饰器，用于标记需要 phonemizer 的测试用例
    return unittest.skipUnless(is_phonemizer_available(), "test requires phonemizer")(test_case)


def require_pyctcdecode(test_case):
    """
    Decorator marking a test that requires pyctcdecode
    """
    # 返回一个装饰器，用于标记需要 pyctcdecode 的测试用例
    return unittest.skipUnless(is_pyctcdecode_available(), "test requires pyctcdecode")(test_case)


def require_librosa(test_case):
    # Placeholder for require_librosa
    pass
    # 定义一个装饰器，用于标记需要使用 librosa 的测试
    """
    Decorator marking a test that requires librosa
    """
    # 返回一个装饰器，根据是否可用 librosa 决定是否跳过测试
    return unittest.skipUnless(is_librosa_available(), "test requires librosa")(test_case)
# 标记一个测试需要 essentia 的装饰器
def require_essentia(test_case):
    """
    Decorator marking a test that requires essentia
    """
    return unittest.skipUnless(is_essentia_available(), "test requires essentia")(test_case)


# 标记一个测试需要 pretty_midi 的装饰器
def require_pretty_midi(test_case):
    """
    Decorator marking a test that requires pretty_midi
    """
    return unittest.skipUnless(is_pretty_midi_available(), "test requires pretty_midi")(test_case)


# 检查给定命令是否存在
def cmd_exists(cmd):
    return shutil.which(cmd) is not None


# 标记一个测试需要 `/usr/bin/time` 的装饰器
def require_usr_bin_time(test_case):
    """
    Decorator marking a test that requires `/usr/bin/time`
    """
    return unittest.skipUnless(cmd_exists("/usr/bin/time"), "test requires /usr/bin/time")(test_case)


# 标记一个测试需要 sudachi 的装饰器
def require_sudachi(test_case):
    """
    Decorator marking a test that requires sudachi
    """
    return unittest.skipUnless(is_sudachi_available(), "test requires sudachi")(test_case)


# 标记一个测试需要 jumanpp 的装饰器
def require_jumanpp(test_case):
    """
    Decorator marking a test that requires jumanpp
    """
    return unittest.skipUnless(is_jumanpp_available(), "test requires jumanpp")(test_case)


# 标记一个测试需要 cython 的装饰器
def require_cython(test_case):
    """
    Decorator marking a test that requires jumanpp
    """
    return unittest.skipUnless(is_cython_available(), "test requires cython")(test_case)


# 返回可用 GPU 数量（不管是使用 torch、tf 还是 jax）
def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch, tf or jax is used)
    """
    if is_torch_available():
        import torch

        return torch.cuda.device_count()
    elif is_tf_available():
        import tensorflow as tf

        return len(tf.config.list_physical_devices("GPU"))
    elif is_flax_available():
        import jax

        return jax.device_count()
    else:
        return 0


# 获取测试目录路径
def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # 获取调用该函数的文件路径
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    # 循环直到找到��含 "tests" 的目录
    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    # 如果提供了 append_path，则将其连接到 "tests" 目录后面
    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


#
# 用于处理测试文本输出的辅助函数
# 原始代码来源于：
# https://github.com/fastai/fastai/blob/master/tests/utils/text.py


# 当任何函数包含 print() 调用并且被覆盖时，比如进度条，
# 需要特别注意，因为在 pytest -s 捕获的输出（capsys 或 contextlib.redirect_stdout）
# 包含任何临时打印的字符串，后面跟着 \r。这个辅助函数确保缓冲区将包含相同的输出
# 无论是否在 pytest 中使用 -s，将:
# foo bar\r tar mar\r final message
# 转换为:
# final message
# 定义一个函数，用于处理单个字符串或多行缓冲区
def apply_print_resets(buf):
    # 使用正则表达式替换掉以\r结尾的内容，返回处理后的结果
    return re.sub(r"^.*\r", "", buf, 0, re.M)

# 断言输出中包含特定内容
def assert_screenout(out, what):
    # 将输出内容转换为小写，并去除特定格式的内容
    out_pr = apply_print_resets(out).lower()
    # 在处理后的输出中查找特定内容，如果找到则继续执行，否则抛出异常
    match_str = out_pr.find(what.lower())
    assert match_str != -1, f"expecting to find {what} in output: f{out_pr}"

# 定义一个上下文管理器，用于捕获和重放 stdout 和 stderr
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

    ```py
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

    # 初始化方法，设置是否捕获 stdout 和 stderr，以及是否重放
    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        # 如���捕获 stdout，则创建一个 StringIO 对象
        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        # 如果捕获 stderr，则创建一个 StringIO 对象
        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    # 进入上下文时执行的方法
    def __enter__(self):
        # 如果捕获 stdout，则将 sys.stdout 重定向到 StringIO 对象
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        # 如果捕获 stderr，则将 sys.stderr 重定向到 StringIO 对象
        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self
    # 当退出上下文时的操作，接受任意异常信息
    def __exit__(self, *exc):
        # 如果有输出缓冲区
        if self.out_buf:
            # 恢复标准输出到先前状态
            sys.stdout = self.out_old
            # 获取输出缓冲区中的内容
            captured = self.out_buf.getvalue()
            # 如果需要重放
            if self.replay:
                # 将捕获的内容写回标准输出
                sys.stdout.write(captured)
            # 应用输出重置并更新实例变量
            self.out = apply_print_resets(captured)

        # 如果有错误输出缓冲区
        if self.err_buf:
            # 恢复标准错误输出到先前状态
            sys.stderr = self.err_old
            # 获取错误输出缓冲区中的内容
            captured = self.err_buf.getvalue()
            # 如果需要重放
            if self.replay:
                # 将捕获的内容写回标准错误输出
                sys.stderr.write(captured)
            # 更新实例变量
            self.err = captured

    # 定义对象的字符串表示形式
    def __repr__(self):
        # 初始化消息为空字符串
        msg = ""
        # 如果存在输出缓冲区
        if self.out_buf:
            # 添加标准输出的字符串表示形式到消息中
            msg += f"stdout: {self.out}\n"
        # 如果存在错误输出缓冲区
        if self.err_buf:
            # 添加标准错误输出的字符串表示形式到消息中
            msg += f"stderr: {self.err}\n"
        # 返回消息
        return msg
# 在测试中，最好只捕获所需的流，否则很容易错过一些东西，所以除非需要捕获两个流，否则使用下面的子类（输入更少）。
# 或者可以配置`CaptureStd`来禁用不需要测试的流。

class CaptureStdout(CaptureStd):
    """与CaptureStd相同，但仅捕获stdout"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """与CaptureStd相同，但仅捕获stderr"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    上下文管理器，用于捕获`logging`流

    Args:
        logger: 'logging` logger对象

    Returns:
        通过`self.out`可获得捕获的输出

    示例:

    ```py
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
    这是一个上下文管理器，用于临时更改transformers模块的日志级别为所需值，并在作用域结束时将其恢复为原始设置。

    示例:

    ```py
    with LoggingLevel(logging.INFO):
        AutoModel.from_pretrained("gpt2")  # 调用logger.info()多次
    ```
    """

    orig_level = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity(level)
        yield
    finally:
        transformers_logging.set_verbosity(orig_level)


@contextlib.contextmanager
# 改编自https://stackoverflow.com/a/64789046/9201239
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    临时将给定路径添加到`sys.path`中。

    用法:

    ```py
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
    此类扩展了*unittest.TestCase*，具有附加功能。

    特性1: 一组完全解析的重要文件和目录路径访问器。
    class TestPaths:
        """
        在测试中通常需要知道事物相对于当前测试文件的位置，这并不是一个简单的问题，因为测试可以从多个目录调用，或者可能位于具有不同深度的子目录中。该类通过整理所有基本路径来解决这个问题，并提供了易于访问的访问器：
    
        - `pathlib` 对象（全部解析）：
    
           - `test_file_path` - 当前测试文件路径（=`__file__`）
           - `test_file_dir` - 包含当前测试文件的目录
           - `tests_dir` - `tests` 测试套件的目录
           - `examples_dir` - `examples` 测试套件的目录
           - `repo_root_dir` - 仓库的目录
           - `src_dir` - `src` 的目录（即 `transformers` 子目录所在的位置）
    
        - 字符串化的路径---与上述相同，但这些返回路径作为字符串，而不是 `pathlib` 对象：
    
           - `test_file_path_str`
           - `test_file_dir_str`
           - `tests_dir_str`
           - `examples_dir_str`
           - `repo_root_dir_str`
           - `src_dir_str`
    
        功能 2: 灵活的自动可移除临时目录，保证在测试结束时被删除。
    
        1. 创建一个唯一的临时目录：
    
        ```py
        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir()
        ```
    
        `tmp_dir` 将包含创建的临时目录的路径。它将在测试结束时自动删除。
    
    
        2. 创建我选择的临时目录，在测试开始前确保它为空，并在测试结束后不清空它。
    
        ```py
        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
        ```
    
        当您希望监视特定目录并确保以前的测试未在其中留下任何数据时，这是有用的。
    
        3. 您可以通过直接覆盖 `before` 和 `after` 参数来覆盖前两个选项，导致以下行为：
    
        `before=True`：临时目录将始终在测试开始时清除。
    
        `before=False`：如果临时目录已存在，则任何现有文件将保留在其中。
    
        `after=True`：临时目录将始终在测试结束时删除。
    
        `after=False`：临时目录将始终在测试结束时保持不变。
    
        注意 1：为了安全运行等同于 `rm -r` 的操作，只允许使用显式 `tmp_dir` 的项目仓库检出的子目录，以便不会意外地清理 `/tmp` 或类似的文件系统的重要部分。即请始终传递以 `./` 开头的路径。
    
        注意 2：每个测试都可以注册多个临时目录，并且除非另有要求，否则它们都将被自动删除。
    
        功能 3: 获取设置了特定于当前测试套件的 `PYTHONPATH` 的 `os.environ` 对象的副本。这
        """
        def __init__(self):
            # 初始化测试路径对象
            self._initialize_test_paths()
    
        def _initialize_test_paths(self):
            # 初始化测试路径
            self.test_file_path = Path(__file__).resolve()
            # 当前测试文件所在的目录
            self.test_file_dir = self.test_file_path.parent
            # `tests` 测试套件的目录
            self.tests_dir = self.test_file_dir.parent
            # `examples` 测试套件的目录
            self.examples_dir = self.tests_dir / "examples"
            # 仓库的目录
            self.repo_root_dir = self.tests_dir.parent
            # `src` 的目录
            self.src_dir = self.repo_root_dir / "src"
    
            # 将路径转换为字符串
            self.test_file_path_str = str(self.test_file_path)
            self.test_file_dir_str = str(self.test_file_dir)
            self.tests_dir_str = str(self.tests_dir)
            self.examples_dir_str = str(self.examples_dir)
            self.repo_root_dir_str = str(self.repo_root_dir)
            self.src_dir_str = str(self.src_dir)
    
        def get_auto_remove_tmp_dir(self, path=None, before=True, after=True):
            # 获取自动可移除临时目录
            tmp_dir = TemporaryDirectory(prefix="tmp_", dir=path)
            # 返回临时目录路径
            return tmp_dir.name
    def test_whatever(self):
        # 获取设置好的环境变量
        env = self.get_env()
    ```py

    def setUp(self):
        # get_auto_remove_tmp_dir feature:
        # 初始化用于自动清理临时目录的列表
        self.teardown_tmp_dirs = []

        # 获取测试文件所在的绝对路径
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        # 获取测试文件的父目录
        self._test_file_dir = path.parents[0]
        # 通过迭代查找项目的根目录
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            # 判断是否找到了根目录
            if (tmp_dir / "src").is_dir() and (tmp_dir / "tests").is_dir():
                break
        # 如果找到根目录，则设置根目录路径；否则，抛出异常
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        # 设置测试、示例和源代码目录路径
        self._tests_dir = self._repo_root_dir / "tests"
        self._examples_dir = self._repo_root_dir / "examples"
        self._src_dir = self._repo_root_dir / "src"

    @property
    def test_file_path(self):
        # 返回测试文件路径
        return self._test_file_path

    @property
    def test_file_path_str(self):
        # 返回测试文件路径的字符串形式
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        # 返回测试文件所在目录
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        # 返回测试文件所在目录的字符串形式
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        # 返回测试目录路径
        return self._tests_dir

    @property
    def tests_dir_str(self):
        # 返回测试目录路径的字符串形式
        return str(self._tests_dir)

    @property
    def examples_dir(self):
        # 返回示例目录路径
        return self._examples_dir

    @property
    def examples_dir_str(self):
        # 返回示例目录路径的字符串形式
        return str(self._examples_dir)

    @property
    def repo_root_dir(self):
        # 返回项目根目录路径
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        # 返回项目根目录路径的字符串形式
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        # 返回源代码目录路径
        return self._src_dir

    @property
    def src_dir_str(self):
        # 返回源代码目录路径的字符串形式
        return str(self._src_dir)

    def get_env(self):
        """
        Return a copy of the `os.environ` object that sets up `PYTHONPATH` correctly, depending on the test suite it's
        invoked from. This is useful for invoking external programs from the test suite - e.g. distributed training.

        It always inserts `./src` first, then `./tests` or `./examples` depending on the test suite type and finally
        the preset `PYTHONPATH` if any (all full resolved paths).

        """
        # 复制当前环境变量
        env = os.environ.copy()
        # 构建正确设置了 `PYTHONPATH` 的环境变量
        paths = [self.src_dir_str]
        # 根据测试套件类型插入 `./tests` 或 `./examples`
        if "/examples" in self.test_file_dir_str:
            paths.append(self.examples_dir_str)
        else:
            paths.append(self.tests_dir_str)
        # 插入预设的 `PYTHONPATH`（如果有的话，全都是完全解析的路径）
        paths.append(env.get("PYTHONPATH", ""))

        # 将路径列表拼接成字符串，并设置到环境变量中
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
            # 定义自定义路径提供时最可能的期望行为。
            # 这很可能表示调试模式，我们想要一个易于定位的目录，具有以下特点：
            # 1. 在测试之前清除（如果已经存在）
            # 2. 在测试结束后保留
            if before is None:
                before = True
            if after is None:
                after = False

            # 使用提供的路径
            path = Path(tmp_dir).resolve()

            # 为了避免破坏文件系统的部分，只允许相对路径
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
                )

            # 确保目录起始为空
            if before is True and path.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            path.mkdir(parents=True, exist_ok=True)

        else:
            # 定义自动生成唯一临时路径时最可能的期望行为（不是调试模式）。
            # 在这种情况下，我们需要一个唯一的临时目录：
            # 1. 在测试之前为空（在这种情况下，它将始终为空）
            # 2. 在测试结束后完全删除
            if before is None:
                before = True
            if after is None:
                after = True

            # 使用唯一的临时目录（始终为空，不管 `before` 如何）
            tmp_dir = tempfile.mkdtemp()

        if after is True:
            # 注册以进行删除
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir
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
        one_liner_str = 'from transformers import AutoModel; AutoModel.from_pretrained("t5-large")'
        max_rss = self.python_one_liner_max_rss(one_liner_str)
        ```py
        """

        # 检查是否存在 /usr/bin/time 命令
        if not cmd_exists("/usr/bin/time"):
            raise ValueError("/usr/bin/time is required, install with `apt install time`")

        # 将命令字符串解析为列表
        cmd = shlex.split(f"/usr/bin/time -f %M python -c '{one_liner_str}'")
        # 使用 CaptureStd 上下文管理器捕获标准输出和标准错误
        with CaptureStd() as cs:
            # 异步执行子进程
            execute_subprocess_async(cmd, env=self.get_env())
        # 获取最大 RSS（Resident Set Size）并转换为字节
        max_rss = int(cs.err.split("\n")[-2].replace("stderr: ", "")) * 1024
        return max_rss

    def tearDown(self):
        # get_auto_remove_tmp_dir feature: remove registered temp dirs
        # 遍历需要清理的临时目录列表，删除目录
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []
        # 如果加速器可用，则重置状态
        if is_accelerate_available():
            AcceleratorState._reset_state()
            PartialState._reset_state()

            # 删除所有环境变量中包含 `ACCELERATE` 的变量
            for k in list(os.environ.keys()):
                if "ACCELERATE" in k:
                    del os.environ[k]
# 定义一个便捷的包装器，允许在测试函数中方便地设置环境变量
def mockenv(**kwargs):
    """
   this is a convenience wrapper, that allows this ::

   @mockenv(RUN_SLOW=True, USE_TF=False) def test_something():
        run_slow = os.getenv("RUN_SLOW", False) use_tf = os.getenv("USE_TF", False)

   """
    return mock.patch.dict(os.environ, kwargs)


# 临时更新 `os.environ` 字典的上下文管理器，类似于 mockenv
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    Temporarily updates the `os.environ` dictionary in-place. Similar to mockenv

    The `os.environ` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
      remove: Environment variables to remove.
      update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # 被更新或删除的环境变量列表
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # 退出时需要恢复的环境变量和值
    update_after = {k: env[k] for k in stomped}
    # 退出时需要删除的环境变量
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


# --- pytest conf functions --- #

# 避免从 tests/conftest.py 和 examples/conftest.py 多次调用 - 确保只调用一次
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.
    # 导入所需模块，注意这里使用了 _pytest 的私有 API，若 pytest 进行内部更改可能会导致该功能失效；同时，调用了 terminalreporter 的默认内部方法，可能会被各种 `pytest-` 插件劫持而产生干扰。
    from _pytest.config import create_terminal_writer
    
    # 如果 id 为空，则将其设置为 "tests"
    if not len(id):
        id = "tests"
    
    # 获取 terminalreporter 对应的配置信息
    config = tr.config
    
    # 获取原始的终端写入器
    orig_writer = config.get_terminal_writer()
    
    # 获取原始的 traceback 显示方式
    orig_tbstyle = config.option.tbstyle
    
    # 获取 terminalreporter 的原始报告字符
    orig_reportchars = tr.reportchars
    
    # 创建报告保存的文件夹路径
    dir = f"reports/{id}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    
    # 定义不同类型报告的文件名及路径
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
    
    # 自定义耗时报告
    # 注意：不需要调用 pytest --durations=XX 来获取单独的报告
    # 适配自 https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")
    
    # 定义简短失败报告
    def summary_failures_short(tr):
        # 期望报告为 --tb=long (默认) 格式，此处将其截断至最后一帧
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # 截断可选的额外前导帧，只保留最后一帧
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # 注意：不打印任何 rep.sections，以保持报告简洁
    
    # 使用预先准备好的报告函数，将日志输出到专用文件中
    # 适配自 https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # 注意：某些 pytest 插件可能会通过劫持默认的 `terminalreporter` 来产生干扰
    # 报告失败时使用 line/short/long 样式
    config.option.tbstyle = "auto"  # 全部 traceback 显示
    # 将失败长报告写入文件
    with open(report_files["failures_long"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 汇总失败
        tr.summary_failures()
    
    # 将失败短报告写入文件
    with open(report_files["failures_short"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 简短汇总失败
        summary_failures_short(tr)
    
    # 将失败行报告写入文件
    config.option.tbstyle = "line"  # 每个错误一行
    with open(report_files["failures_line"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 汇总失败
        tr.summary_failures()
    
    # 将错误报告写入文件
    with open(report_files["errors"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 汇总错误
        tr.summary_errors()
    
    # 将警告报告写入文件
    with open(report_files["warnings"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 汇总普通警告
        tr.summary_warnings()
        # 汇总最终警告
        tr.summary_warnings()
    
    # 设置报告字符以模拟 `-rA`（用于 summary_passes() 和 short_test_summary() 中）
    tr.reportchars = "wPpsxXEf"
    
    # 跳过 `passes` 报告，因为它开始花费超过 5 分钟，有时在 CircleCI 上超时，如果花费 > 10 分钟（因为此部分不在终端上生成任何输出）。
    # （而且，似乎在此报告中没有有用的信息，我们很少需要阅读它）
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()
    
    # 将简短测试摘要写入文件
    with open(report_files["summary_short"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 汇总简短测试摘要
        tr.short_test_summary()
    
    # 将统计摘要写入文件
    with open(report_files["stats"], "w") as f:
        # 创建终端写入器并将其配置为写入文件
        tr._tw = create_terminal_writer(config, f)
        # 汇总统计信息
        tr.summary_stats()
    
    # 恢复:
    # 恢复终端写入器为原始写入器
    tr._tw = orig_writer
    # 恢复报告字符为原始报告字符
    tr.reportchars = orig_reportchars
    # 恢复 traceback 格式为原始格式
    config.option.tbstyle = orig_tbstyle
# --- distributed testing functions --- #

# 导入 asyncio 模块
import asyncio  # noqa

# 定义一个用于存储子进程输出的类
class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

# 异步读取流的函数
async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break

# 异步执行子进程并处理输出流的函数
async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    if echo:
        print("\nRunning: ", " ".join(cmd))

    # 创建子进程
    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # 读取子进程的输出流
    out = []
    err = []

    # 处理输出流的回调函数
    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # 异步等待输出流的处理
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)

# 同步执行子进程的函数
def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # 检查子进程是否有输出
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result

# 返回 pytest-xdist 的工作进程编号
def pytest_xdist_worker_id():
    """
    Returns an int value of worker's numerical id under `pytest-xdist`'s concurrent workers `pytest -n N` regime, or 0
    if `-n 1` or `pytest-xdist` isn't being used.
    """
    # 获取环境变量 "PYTEST_XDIST_WORKER" 的值，若不存在则默认为 "gw0"
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    # 使用正则表达式替换字符串中以 "gw" 开头的部分为空字符串
    worker = re.sub(r"^gw", "", worker, 0, re.M)
    # 将结果转换为整数类型并返回
    return int(worker)
# 返回一个可以传递给 `torch.distributed.launch` 的 `--master_port` 参数的端口号
def get_torch_dist_unique_port():
    port = 29500
    # 如果在 `pytest-xdist` 下运行，根据 worker id 添加一个增量，以避免并发测试尝试同时使用相同的端口
    uniq_delta = pytest_xdist_worker_id()
    return port + uniq_delta


# 简化对象，将浮点数四舍五入，将张量/NumPy 数组降级以便在测试中进行简单的相等性检查
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
            # 如果长度为 1，则字典为空
            assert lines[0] == "{}"
        else:
            # 否则确保 JSON 格式正确（至少有 3 行）
            assert len(lines) >= 3
            # 每个键一行，缩进应为 2，最小长度为 3
            assert lines[0].strip() == "{"
            for line in lines[1:-1]:
                left_indent = len(lines[1]) - len(lines[1].lstrip())
                assert left_indent == 2
            assert lines[-1].strip() == "}"


# 将输入转换为二元组
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# 这些工具与确保在运行脚本时接收到正确的错误消息有关
class SubprocessCallException(Exception):
    pass


# 运行 `command`，使用 `subprocess.check_output`，可能返回 `stdout`。还将正确捕获运行 `command` 时是否发生错误
def run_command(command: List[str], return_stdout=False):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    # 捕获子进程调用时可能抛出的异常，存储在变量e中
    except subprocess.CalledProcessError as e:
        # 抛出自定义的SubprocessCallException异常，并传递错误信息
        raise SubprocessCallException(
            # 格式化字符串，包含失败的命令和错误输出
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
    ```py
    """

    def __enter__(self):
        # 初始化请求计数器字典
        self._counter = defaultdict(int)
        # 开始拦截 urllib3 的 debug 日志
        self.patcher = patch.object(urllib3.connectionpool.log, "debug", wraps=urllib3.connectionpool.log.debug)
        # 启动拦截器
        self.mock = self.patcher.start()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        # 遍历拦截到的每个日志调用
        for call in self.mock.call_args_list:
            # 提取日志内容
            log = call.args[0] % call.args[1:]
            # 遍历 HTTP 方法
            for method in ("HEAD", "GET", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"):
                # 如果日志中包含当前方法
                if method in log:
                    # 将该方法计数加一
                    self._counter[method] += 1
                    break
        # 停止拦截器
        self.patcher.stop()

    def __getitem__(self, key: str) -> int:
        # 返回指定方法的请求计数
        return self._counter[key]

    @property
    def total_calls(self) -> int:
        # 返回所有请求的总计数
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
            # 初始化重试次数
            retry_count = 1

            # 在达到最大重试次数前循环
            while retry_count < max_attempts:
                try:
                    # 调用测试函数
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    # 输出错误信息和重试次数
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    # 如果设置了重试等待时间，则等待指定时间后再重试
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    # 增加重试次数
                    retry_count += 1

            # 返回测试函数的最终结果
            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator


def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.
    """
    Args:
        test_case (`unittest.TestCase`):
            运行 `target_func` 的测试用例。
        target_func (`Callable`):
            实现实际测试逻辑的函数。
        inputs (`dict`, *可选*, 默认为 `None`):
            通过输入队列传递给 `target_func` 的输入。
        timeout (`int`, *可选*, 默认为 `None`):
            传递给输入和输出队列的超时时间（秒）。如果未指定，则检查环境变量 `PYTEST_TIMEOUT`。如果仍为 `None`，则将其值设置为 `600`。

    """
    # 如果未指定超时时间，则将其设置为环境变量 `PYTEST_TIMEOUT` 的值，若未定义，则设为 `600`。
    if timeout is None:
        timeout = int(os.environ.get("PYTEST_TIMEOUT", 600))

    # 使用 "spawn" 方法创建多进程上下文。
    start_methohd = "spawn"
    ctx = multiprocessing.get_context(start_methohd)

    # 创建容量为1的输入队列和输出队列。
    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # 无法将 `unittest.TestCase` 发送到子进程，否则会出现有关 pickle 的问题。
    # 将输入放入输入队列。
    input_queue.put(inputs, timeout=timeout)

    # 创建一个子进程，目标函数为 `target_func`，参数为输入队列、输出队列和超时时间。
    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()

    # 如果无法及时从子进程获取输出，则终止子进程以防止测试无法正常退出。
    try:
        # 获取子进程的输出结果。
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        # 如果出现异常，则终止子进程并在测试用例中标记为失败。
        process.terminate()
        test_case.fail(e)

    # 等待子进程终止。
    process.join(timeout=timeout)

    # 如果结果中存在错误，则在测试用例中标记为失败。
    if results["error"] is not None:
        test_case.fail(f'{results["error"]}')
"""
The following contains utils to run the documentation tests without having to overwrite any files.

The `preprocess_string` function adds `# doctest: +IGNORE_RESULT` markers on the fly anywhere a `load_dataset` call is
made as a print would otherwise fail the corresonding line.

To skip cuda tests, make sure to call `SKIP_CUDA_DOCTEST=1 pytest --doctest-modules <path_to_files_to_test>
"""

# 定义一个函数，用于在不覆盖任何文件的情况下运行文档测试
def preprocess_string(string, skip_cuda_tests):
    """Prepare a docstring or a `.md` file to be run by doctest.

    The argument `string` would be the whole file content if it is a `.md` file. For a python file, it would be one of
    its docstring. In each case, it may contain multiple python code examples. If `skip_cuda_tests` is `True` and a
    cuda stuff is detective (with a heuristic), this method will return an empty string so no doctest will be run for
    `string`.
    """
    # 定义代码块的正则表达式模式
    codeblock_pattern = r"(```(?:python|py)\s*\n\s*>>> )((?:.*?\n)*?.*?```py)"
    # 使用正则表达式拆分字符串，提取代码块
    codeblocks = re.split(re.compile(codeblock_pattern, flags=re.MULTILINE | re.DOTALL), string)
    # 初始化 CUDA 检测标志
    is_cuda_found = False
    # 遍历代码块列表
    for i, codeblock in enumerate(codeblocks):
        # 在代码块中发现 `load_dataset` 调用并且没有 `# doctest: +IGNORE_RESULT` 标记时，在其后添加标记
        if "load_dataset(" in codeblock and "# doctest: +IGNORE_RESULT" not in codeblock:
            codeblocks[i] = re.sub(r"(>>> .*load_dataset\(.*)", r"\1 # doctest: +IGNORE_RESULT", codeblock)
        # 如果代码块包含 CUDA 相关内容，并且需要跳过 CUDA 测试，则将 CUDA 检测标志设为 True，并退出循环
        if (
            (">>>" in codeblock or "..." in codeblock)
            and re.search(r"cuda|to\(0\)|device=0", codeblock)
            and skip_cuda_tests
        ):
            is_cuda_found = True
            break

    # 如果没有发现 CUDA 相关内容，则将修改后的代码块组合成字符串返回
    modified_string = ""
    if not is_cuda_found:
        modified_string = "".join(codeblocks)

    return modified_string


# 定义一个类，继承自 doctest.DocTestParser，用于解析黑色格式的代码块
class HfDocTestParser(doctest.DocTestParser):
    """
    Overwrites the DocTestParser from doctest to properly parse the codeblocks that are formatted with black. This
    means that there are no extra lines at the end of our snippets. The `# doctest: +IGNORE_RESULT` marker is also
    added anywhere a `load_dataset` call is made as a print would otherwise fail the corresponding line.

    Tests involving cuda are skipped base on a naive pattern that should be updated if it is not enough.
    """
    # This regular expression is used to find doctest examples in a
    # string.  It defines three groups: `source` is the source code
    # (including leading indentation and prompts); `indent` is the
    # indentation of the first (PS1) line of the source code; and
    # `want` is the expected output (including leading indentation).
    # fmt: off
    # 编译正则表达式，用于匹配源代码和期望输出
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)          # Not a line starting with PS1
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:(?!```).)*        # Match any character except '`' until a '```py' is found (this is specific to HF because black removes the last line)
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )
    
    # !!!!!!!!!!! HF Specific !!!!!!!!!!!
    # 设置是否跳过 CUDA 测试的标志，通过检查环境变量是否设置来确定
    skip_cuda_tests: bool = bool(os.environ.get("SKIP_CUDA_DOCTEST", False))
    # !!!!!!!!!!! HF Specific !!!!!!!!!!!
    
    # 重写 `parse` 方法以包含对 CUDA 测试的跳过，并在调用 `super().parse` 前移除日志和数据集打印
    def parse(self, string, name="<string>"):
        """
        Overwrites the `parse` method to incorporate a skip for CUDA tests, and remove logs and dataset prints before
        calling `super().parse`
        """
        # 预处理字符串，根据是否跳过 CUDA 测试来决定是否移除 CUDA 相关的代码
        string = preprocess_string(string, self.skip_cuda_tests)
        # 调用父类的解析方法
        return super().parse(string, name)
# 定义名为 HfDoctestModule 的类，继承自 Module 类
class HfDoctestModule(Module):
    """
    Overwrites the `DoctestModule` of the pytest package to make sure the HFDocTestParser is used when discovering
    tests.
    """
    # 重写了 pytest 包中的 DoctestModule，确保在发现测试时使用 HFDocTestParser
    # 定义一个方法，用于收集 doctest 项
    def collect(self) -> Iterable[DoctestItem]:
        # 定义一个特殊的 doctest finder，用于修复标准库中的 bug
        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456 https://bugs.python.org/issue25532
            """

            # 重写 _find_lineno 方法以修复标准库的 bug
            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct line number is returned. This will be
                reported upstream. #8796
                """
                # 如果 obj 是 property 类型，则尝试获取其 fget 属性
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                # 如果 obj 有 __wrapped__ 属性，则获取其原始对象
                if hasattr(obj, "__wrapped__"):
                    # 获取被包装的主要对象以获得正确的行号
                    obj = inspect.unwrap(obj)

                # Type ignored because this is a private function.
                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            # 重写 _find 方法以修复标准库的 bug
            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                # 如果 obj 是被模拟的，则直接返回，不执行测试
                if _is_mocked(obj):
                    return
                # 用 _patch_unwrap_mock_aware() 上下文包装器解决问题
                with _patch_unwrap_mock_aware():
                    # Type ignored because this is a private function.
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )

        # 如果路径的名称为 "conftest.py"，则从配置的根路径中导入 conftest 模块
        if self.path.name == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            # 否则，尝试导入给定路径的模块
            try:
                module = import_path(
                    self.path,
                    root=self.config.rootpath,
                    mode=self.config.getoption("importmode"),
                )
            # 如果导入失败，根据配置决定是跳过还是引发 ImportError
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    skip("unable to import module %r" % self.path)
                else:
                    raise

        # 创建 MockAwareDocTestFinder 实例，用于查找 doctest
        finder = MockAwareDocTestFinder(parser=HfDocTestParser())
        # 获取选项标志
        optionflags = get_optionflags(self)
        # 获取测试运行器
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        # 遍历找到的所有 doctest，生成相应的 DoctestItem
        for test in finder.find(module, module.__name__):
            # 如果测试中包含示例，则生成对应的 DoctestItem
            if test.examples:  # skip empty doctests and cuda
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
# 定义一个函数，根据设备类型分发执行不同的函数
def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    # 如果设备不在分发表中，则调用默认函数
    if device not in dispatch_table:
        return dispatch_table["default"](*args, **kwargs)

    # 获取对应设备的函数
    fn = dispatch_table[device]

    # 一些与设备无关的函数会返回值，需要在用户级别处对 `None` 进行处理
    if fn is None:
        return None
    return fn(*args, **kwargs)


# 如果 Torch 可用
if is_torch_available():
    # 设备名称到可调用函数的映射，用于支持设备无关测试
    BACKEND_MANUAL_SEED = {"cuda": torch.cuda.manual_seed, "cpu": torch.manual_seed, "default": torch.manual_seed}
    # 清空缓存的函数映射，对 CPU 设备和其他设备处理方式不同
    BACKEND_EMPTY_CACHE = {"cuda": torch.cuda.empty_cache, "cpu": None, "default": None}
    # 设备数量查询的函数映射，对 CPU 和其他设备的处理方式不同
    BACKEND_DEVICE_COUNT = {"cuda": torch.cuda.device_count, "cpu": lambda: 0, "default": lambda: 1}


# 设置随机种子的后端函数
def backend_manual_seed(device: str, seed: int):
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)


# 清空缓存的后端函数
def backend_empty_cache(device: str):
    return _device_agnostic_dispatch(device, BACKEND_EMPTY_CACHE)


# 查询设备数量的后端函数
def backend_device_count(device: str):
    return _device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)


# 如果 Torch 可用
if is_torch_available():
    # 如果 `TRANSFORMERS_TEST_DEVICE_SPEC` 已启用，需要导入额外的设备到函数映射项
```  
    # 检查环境变量中是否存在名为"TRANSFORMERS_TEST_DEVICE_SPEC"的键
    if "TRANSFORMERS_TEST_DEVICE_SPEC" in os.environ:
        # 如果存在，获取环境变量中指定的设备规格文件路径
        device_spec_path = os.environ["TRANSFORMERS_TEST_DEVICE_SPEC"]
        # 检查路径是否指向一个存在的文件
        if not Path(device_spec_path).is_file():
            # 如果不是文件或文件不存在，则引发值错误异常
            raise ValueError(
                f"Specified path to device spec file is not a file or not found. Received '{device_spec_path}"
            )

        # 尝试从文件路径中去除扩展名以备后续导入 - 同时验证是否导入了一个 Python 文件
        try:
            import_name = device_spec_path[: device_spec_path.index(".py")]
        except ValueError as e:
            # 如果提供的设备规格文件不是 Python 文件，则引发值错误异常
            raise ValueError(f"Provided device spec file was not a Python file! Received '{device_spec_path}") from e

        # 导入设备规格模块
        device_spec_module = importlib.import_module(import_name)

        # 导入的文件必须包含 `DEVICE_NAME`。如果没有，则提前终止。
        try:
            # 尝试从导入的模块中获取 `DEVICE_NAME`
            device_name = device_spec_module.DEVICE_NAME
        except AttributeError as e:
            # 如果模块不包含 `DEVICE_NAME`，则引发属性错误异常
            raise AttributeError("Device spec file did not contain `DEVICE_NAME`") from e

        # 如果环境变量中存在"TRANSFORMERS_TEST_DEVICE"且其值与设备名称不匹配，则引发值错误异常
        if "TRANSFORMERS_TEST_DEVICE" in os.environ and torch_device != device_name:
            msg = f"Mismatch between environment variable `TRANSFORMERS_TEST_DEVICE` '{torch_device}' and device found in spec '{device_name}'\n"
            msg += "Either unset `TRANSFORMERS_TEST_DEVICE` or ensure it matches device spec name."
            raise ValueError(msg)

        # 更新 Torch 设备名称为设备规格中的名称
        torch_device = device_name

        # 定义一个函数，用于从设备规格文件中更新指定字典的映射关系
        def update_mapping_from_spec(device_fn_dict: Dict[str, Callable], attribute_name: str):
            try:
                # 尝试直接导入函数
                spec_fn = getattr(device_spec_module, attribute_name)
                # 将函数添加到指定字典中
                device_fn_dict[torch_device] = spec_fn
            except AttributeError as e:
                # 如果函数不存在，并且字典中没有默认值，则引发属性错误异常
                if "default" not in device_fn_dict:
                    raise AttributeError(
                        f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found."
                    ) from e

        # 在此处为每个 `BACKEND_*` 字典添加一个条目，并从设备规格文件中更新映射关系
        update_mapping_from_spec(BACKEND_MANUAL_SEED, "MANUAL_SEED_FN")
        update_mapping_from_spec(BACKEND_EMPTY_CACHE, "EMPTY_CACHE_FN")
        update_mapping_from_spec(BACKEND_DEVICE_COUNT, "DEVICE_COUNT_FN")
```