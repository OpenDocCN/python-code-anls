# `.\pytorch\test\onnx\pytorch_test_common.py`

```
# Owner(s): ["module: onnx"]
from __future__ import annotations

import functools  # 导入 functools 模块，用于创建装饰器
import os  # 导入 os 模块，用于处理操作系统相关的功能
import random  # 导入 random 模块，用于生成随机数
import sys  # 导入 sys 模块，用于访问与 Python 解释器相关的变量和函数
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from enum import auto, Enum  # 导入 enum 模块，用于创建枚举类型
from typing import Optional  # 导入 typing 模块，用于类型提示

import numpy as np  # 导入 numpy 库，用于数值计算
import packaging.version  # 导入 packaging.version 模块，用于处理版本号
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import torch  # 导入 PyTorch 深度学习库
from torch.autograd import function  # 从 torch.autograd 中导入 function 模块
from torch.onnx._internal import diagnostics  # 导入 torch.onnx._internal 中的 diagnostics 模块
from torch.testing._internal import common_utils  # 导入 torch.testing._internal 中的 common_utils 模块

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# 获取当前文件所在目录的父目录的路径，赋值给 pytorch_test_dir 变量
sys.path.insert(-1, pytorch_test_dir)
# 将 pytorch_test_dir 插入到 sys.path 的倒数第二个位置，用于导入测试相关的模块

torch.set_default_dtype(torch.float)
# 设置 PyTorch 的默认数据类型为浮点型

BATCH_SIZE = 2  # 定义 BATCH_SIZE 常量为 2

RNN_BATCH_SIZE = 7  # 定义 RNN_BATCH_SIZE 常量为 7
RNN_SEQUENCE_LENGTH = 11  # 定义 RNN_SEQUENCE_LENGTH 常量为 11
RNN_INPUT_SIZE = 5  # 定义 RNN_INPUT_SIZE 常量为 5
RNN_HIDDEN_SIZE = 3  # 定义 RNN_HIDDEN_SIZE 常量为 3

class TorchModelType(Enum):  # 定义 TorchModelType 枚举类型
    TORCH_NN_MODULE = auto()  # 自动分配值给 TORCH_NN_MODULE
    TORCH_EXPORT_EXPORTEDPROGRAM = auto()  # 自动分配值给 TORCH_EXPORT_EXPORTEDPROGRAM

def _skipper(condition, reason):  # 定义 _skipper 函数装饰器
    def decorator(f):  # 内部定义的装饰器函数
        @functools.wraps(f)  # 使用 functools.wraps 保留被装饰函数的元数据
        def wrapper(*args, **kwargs):  # 装饰器包裹的函数
            if condition():  # 如果 condition 函数返回 True
                raise unittest.SkipTest(reason)  # 抛出 unittest.SkipTest 异常，跳过测试
            return f(*args, **kwargs)  # 否则执行被装饰的函数

        return wrapper  # 返回装饰后的函数

    return decorator  # 返回装饰器本身

skipIfNoCuda = _skipper(lambda: not torch.cuda.is_available(), "CUDA is not available")
# 使用 _skipper 装饰器创建 skipIfNoCuda 装饰器，判断是否有 CUDA 支持，如果没有则跳过测试

skipIfTravis = _skipper(lambda: os.getenv("TRAVIS"), "Skip In Travis")
# 使用 _skipper 装饰器创建 skipIfTravis 装饰器，如果在 Travis CI 中则跳过测试

skipIfNoBFloat16Cuda = _skipper(
    lambda: not torch.cuda.is_bf16_supported(), "BFloat16 CUDA is not available"
)
# 使用 _skipper 装饰器创建 skipIfNoBFloat16Cuda 装饰器，判断是否有 BFloat16 CUDA 支持，如果没有则跳过测试

skipIfQuantizationBackendQNNPack = _skipper(
    lambda: torch.backends.quantized.engine == "qnnpack",
    "Not compatible with QNNPack quantization backend",
)
# 使用 _skipper 装饰器创建 skipIfQuantizationBackendQNNPack 装饰器，
# 如果使用的量化后端为 QNNPack 则跳过测试

def skipIfUnsupportedMinOpsetVersion(min_opset_version):  # 定义 skipIfUnsupportedMinOpsetVersion 装饰器
    def skip_dec(func):  # 内部定义的装饰器函数
        @functools.wraps(func)  # 使用 functools.wraps 保留被装饰函数的元数据
        def wrapper(self, *args, **kwargs):  # 装饰器包裹的函数
            if self.opset_version < min_opset_version:  # 如果 opset_version 小于指定的最小版本
                raise unittest.SkipTest(  # 抛出 unittest.SkipTest 异常，跳过测试
                    f"Unsupported opset_version: {self.opset_version} < {min_opset_version}"
                )
            return func(self, *args, **kwargs)  # 否则执行被装饰的函数

        return wrapper  # 返回装饰后的函数

    return skip_dec  # 返回装饰器本身

def skipIfUnsupportedMaxOpsetVersion(max_opset_version):  # 定义 skipIfUnsupportedMaxOpsetVersion 装饰器
    def skip_dec(func):  # 内部定义的装饰器函数
        @functools.wraps(func)  # 使用 functools.wraps 保留被装饰函数的元数据
        def wrapper(self, *args, **kwargs):  # 装饰器包裹的函数
            if self.opset_version > max_opset_version:  # 如果 opset_version 大于指定的最大版本
                raise unittest.SkipTest(  # 抛出 unittest.SkipTest 异常，跳过测试
                    f"Unsupported opset_version: {self.opset_version} > {max_opset_version}"
                )
            return func(self, *args, **kwargs)  # 否则执行被装饰的函数

        return wrapper  # 返回装饰后的函数

    return skip_dec  # 返回装饰器本身

def skipForAllOpsetVersions():  # 定义 skipForAllOpsetVersions 函数
    # 定义一个装饰器函数 skip_dec，接受一个函数 func 作为参数
    def skip_dec(func):
        # 定义内部函数 wrapper，使用 functools 模块的 wraps 装饰器保留原始函数的元数据
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 如果 self.opset_version 存在且为真值（非空或非零），抛出 unittest.SkipTest 异常
            if self.opset_version:
                raise unittest.SkipTest(
                    "Skip verify test for unsupported opset_version"
                )
            # 否则调用原始函数 func，并传递参数
            return func(self, *args, **kwargs)
    
        # 返回装饰后的函数 wrapper
        return wrapper
# 定义一个装饰器函数，用于跳过 opset 版本低于 skip_before_opset_version 的测试
def skipTraceTest(skip_before_opset_version: Optional[int] = None, reason: str = ""):
    """Skip tracing test for opset version less than skip_before_opset_version.

    Args:
        skip_before_opset_version: The opset version before which to skip tracing test.
            If None, tracing test is always skipped.
        reason: The reason for skipping tracing test.

    Returns:
        A decorator for skipping tracing test.
    """

    # 实际的装饰器函数，接受一个函数作为参数
    def skip_dec(func):
        # 包装器函数，用于包裹原始测试函数
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 判断是否需要跳过当前 opset 版本的测试
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            
            # 如果需要跳过当前 opset 版本的测试，并且不是 TorchScript 测试，抛出跳过测试的异常
            if self.skip_this_opset and not self.is_script:
                raise unittest.SkipTest(f"Skip verify test for torch trace. {reason}")
            
            # 否则，继续执行原始测试函数
            return func(self, *args, **kwargs)

        # 返回包装器函数
        return wrapper

    # 返回装饰器函数
    return skip_dec


# 定义一个装饰器函数，用于跳过 opset 版本低于 skip_before_opset_version 的 TorchScript 测试
def skipScriptTest(skip_before_opset_version: Optional[int] = None, reason: str = ""):
    """Skip scripting test for opset version less than skip_before_opset_version.

    Args:
        skip_before_opset_version: The opset version before which to skip scripting test.
            If None, scripting test is always skipped.
        reason: The reason for skipping scripting test.

    Returns:
        A decorator for skipping scripting test.
    """

    # 实际的装饰器函数，接受一个函数作为参数
    def skip_dec(func):
        # 包装器函数，用于包裹原始测试函数
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 判断是否需要跳过当前 opset 版本的 TorchScript 测试
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            
            # 如果需要跳过当前 opset 版本的 TorchScript 测试，并且是 TorchScript 测试，抛出跳过测试的异常
            if self.skip_this_opset and self.is_script:
                raise unittest.SkipTest(f"Skip verify test for TorchScript. {reason}")
            
            # 否则，继续执行原始测试函数
            return func(self, *args, **kwargs)

        # 返回包装器函数
        return wrapper

    # 返回装饰器函数
    return skip_dec


# 注意: 此装饰器当前未使用，但在未来可能会用于在发布的 ORT 中不支持的更多测试中使用
def skip_min_ort_version(reason: str, version: str, dynamic_only: bool = False):
    # 实际的装饰器函数，接受一个函数作为参数
    def skip_dec(func):
        # 包装器函数，用于包裹原始测试函数
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 判断当前的 ONNX Runtime 版本是否小于指定的版本
            if (
                packaging.version.parse(self.ort_version).release
                < packaging.version.parse(version).release
            ):
                # 如果需要动态形状测试，并且当前不支持动态形状，则跳过测试
                if dynamic_only and not self.dynamic_shapes:
                    return func(self, *args, **kwargs)

                # 否则，抛出跳过测试的异常，指出 ONNX Runtime 版本太老的原因
                raise unittest.SkipTest(
                    f"ONNX Runtime version: {version} is older than required version {version}. "
                    f"Reason: {reason}."
                )
            
            # 如果 ONNX Runtime 版本符合要求，则继续执行原始测试函数
            return func(self, *args, **kwargs)

        # 返回包装器函数
        return wrapper

    # 返回装饰器函数
    return skip_dec


def xfail_dynamic_fx_test(
    error_message: str,
    model_type: Optional[TorchModelType] = None,
    reason: Optional[str] = None,


    # 定义一个名为reason的变量，类型为Optional[str]，默认值为None
# 定义一个装饰器函数，用于标记动态导出测试的预期失败
def xfail_dynamic_exporting_test(
    reason: str,
    model_type: Optional[TorchModelType] = None,
):
    """Xfail dynamic exporting test.

    Args:
        reason: 对动态导出测试预期失败的原因进行说明。
        model_type (TorchModelType): 需要预期失败的模型类型。
            如果为None，则不依据模型类型来预期失败动态测试。

    Returns:
        一个装饰器函数，用于标记动态导出测试的预期失败。
    """

    # 定义装饰器函数的实现
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查是否启用了动态形状，并且模型类型匹配或者未指定模型类型
            if self.dynamic_shapes and (
                not model_type or self.model_type == model_type
            ):
                # 如果条件成立，返回一个预期失败的测试结果
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            # 否则正常执行测试函数
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# 定义一个装饰器函数，用于标记操作级别调试测试的预期失败
def xfail_op_level_debug_test(
    error_message: str,
    model_type: Optional[TorchModelType] = None,
    reason: Optional[str] = None,
):
    """Xfail op level debug test.

    Args:
        reason: 对操作级别调试测试预期失败的原因进行说明。
        model_type (TorchModelType): 需要预期失败的模型类型。
            如果为None，则不依据模型类型来预期失败操作级别调试测试。

    Returns:
        一个装饰器函数，用于标记操作级别调试测试的预期失败。
    """

    # 定义装饰器函数的实现
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查是否启用了操作级别调试，并且模型类型匹配或者未指定模型类型
            if self.op_level_debug and (
                not model_type or self.model_type == model_type
            ):
                # 如果条件成立，返回一个预期失败的测试结果
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            # 否则正常执行测试函数
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# 定义一个装饰器函数，用于跳过动态导出测试
def skip_dynamic_fx_test(reason: str, model_type: TorchModelType = None):
    """Skip dynamic exporting test.

    Args:
        reason: 跳过动态导出测试的原因说明。
        model_type (TorchModelType): 需要跳过的模型类型。
            如果为None，则不依据模型类型来跳过动态测试。

    Returns:
        一个装饰器函数，用于跳过动态导出测试。
    """

    # 定义装饰器函数的实现
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查是否启用了动态形状，并且模型类型匹配或者未指定模型类型
            if self.dynamic_shapes and (
                not model_type or self.model_type == model_type
            ):
                # 如果条件成立，在CI环境中抛出跳过测试的异常
                raise unittest.SkipTest(
                    f"Skip verify dynamic shapes test for FX. {reason}"
                )
            # 否则正常执行测试函数
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# 定义一个装饰器函数，用于在CI环境中跳过测试
def skip_in_ci(reason: str):
    """Skip test in CI.

    Args:
        reason: 跳过在CI环境中执行测试的原因说明。

    Returns:
        一个装饰器函数，用于在CI环境中跳过测试。
    """

    # 定义装饰器函数的实现
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查是否在CI环境中
            if os.getenv("CI"):
                # 如果条件成立，在CI环境中抛出跳过测试的异常
                raise unittest.SkipTest(f"Skip test in CI. {reason}")
            # 否则正常执行测试函数
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec
# 定义一个装饰器函数 `xfail`，用于标记预期失败的测试用例
def xfail(error_message: str, reason: Optional[str] = None):
    """Expect failure.

    Args:
        error_message: 预期的错误消息。
        reason: 预期失败的原因。

    Returns:
        一个装饰器函数，用于标记预期失败的测试用例。
    """

    # 定义装饰器函数 `wrapper`，用于包装测试函数
    def wrapper(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            # 尝试执行被装饰的测试函数
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                # 捕获异常，并根据异常类型进行处理
                if isinstance(e, torch.onnx.OnnxExporterError):
                    # 如果是 ONNX 导出错误，则检查异常的原因中是否包含预期的错误消息
                    assert error_message in str(
                        e.__cause__
                    ), f"Expected error message: {error_message} NOT in {str(e.__cause__)}"
                else:
                    # 否则，检查异常对象本身中是否包含预期的错误消息
                    assert error_message in str(
                        e
                    ), f"Expected error message: {error_message} NOT in {str(e)}"
                # 标记测试为预期失败，并附带失败的原因或默认信息
                pytest.xfail(reason if reason else f"Expected failure: {error_message}")
            else:
                # 如果没有抛出异常，则标记为意外成功，测试失败
                pytest.fail("Unexpected success!")

        return inner

    return wrapper


# 装饰器函数 `skipIfUnsupportedOpsetVersion`，用于跳过不支持的 Opset 版本的测试用例
def skipIfUnsupportedOpsetVersion(unsupported_opset_versions):
    """skips tests for opset_versions listed in unsupported_opset_versions.

    Args:
        unsupported_opset_versions: 不支持的 Opset 版本列表。

    Returns:
        一个装饰器函数，用于跳过不支持的 Opset 版本的测试用例。
    """
    
    # 定义装饰器函数 `skip_dec`，用于包装测试函数
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 如果当前 Opset 版本在不支持的版本列表中，则抛出跳过测试的异常
            if self.opset_version in unsupported_opset_versions:
                raise unittest.SkipTest(
                    "Skip verify test for unsupported opset_version"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# 装饰器函数 `skipShapeChecking`，用于跳过形状检查的测试用例
def skipShapeChecking(func):
    """Skip shape checking for a test case.

    Args:
        func: 被装饰的测试函数。

    Returns:
        一个装饰器函数，用于跳过形状检查的测试用例。
    """
    
    # 定义装饰器函数 `wrapper`，用于包装测试函数
    def wrapper(self, *args, **kwargs):
        # 禁用形状检查标志
        self.check_shape = False
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器函数 `skipDtypeChecking`，用于跳过数据类型检查的测试用例
def skipDtypeChecking(func):
    """Skip dtype checking for a test case.

    Args:
        func: 被装饰的测试函数。

    Returns:
        一个装饰器函数，用于跳过数据类型检查的测试用例。
    """
    
    # 定义装饰器函数 `wrapper`，用于包装测试函数
    def wrapper(self, *args, **kwargs):
        # 禁用数据类型检查标志
        self.check_dtype = False
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器函数 `xfail_if_model_type_is_exportedprogram`，用于针对 ExportedProgram 类型模型的测试用例标记预期失败
def xfail_if_model_type_is_exportedprogram(
    error_message: str, reason: Optional[str] = None
):
    """xfail test with models using ExportedProgram as input.

    Args:
        error_message: The error message to raise when the test is xfailed.
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests based on the model type.
    """
    
    # 定义装饰器函数 `xfail_dec`，用于包装测试函数
    def xfail_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 如果模型类型为 ExportedProgram，则标记测试为预期失败
            if self.model_type == TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return xfail_dec


# 装饰器函数 `xfail_if_model_type_is_not_exportedprogram`，用于针对非 ExportedProgram 类型模型的测试用例标记预期失败
def xfail_if_model_type_is_not_exportedprogram(
    error_message: str, reason: Optional[str] = None
):
    """xfail test if model type is not ExportedProgram.

    Args:
        error_message: The error message to raise when the test is xfailed.
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests based on the model type.
    """
    
    # 这里的函数未提供完整定义，在实际使用中应补充完整以保证功能正确性
    """xfail test without models using ExportedProgram as input.

    Args:
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests.
    """

    # 定义一个装饰器函数 xfail_dec，用于标记不使用 ExportedProgram 模型的测试为 xfail
    def xfail_dec(func):
        # 定义装饰器函数的包装器 wrapper，用于检查模型类型是否为 TORCH_EXPORT_EXPORTEDPROGRAM
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 如果模型类型不是 TORCH_EXPORT_EXPORTEDPROGRAM，则执行 xfail 标记的逻辑
            if self.model_type != TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                # 返回 xfail 函数的结果，标记测试为 xfail
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            # 如果模型类型是 TORCH_EXPORT_EXPORTEDPROGRAM，则直接执行原函数
            return func(self, *args, **kwargs)

        # 返回包装器函数
        return wrapper

    # 返回装饰器函数 xfail_dec 本身
    return xfail_dec
# 定义一个函数，用于将输入的数据结构扁平化为元组
def flatten(x):
    return tuple(function._iter_filter(lambda o: isinstance(o, torch.Tensor))(x))


# 设置随机数种子，保证结果的可重复性
def set_rng_seed(seed):
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)
    # 设置Python内置随机数生成器的种子
    random.seed(seed)
    # 设置NumPy的随机数种子
    np.random.seed(seed)


# 定义一个测试用例类，用于ONNX导出的测试
class ExportTestCase(common_utils.TestCase):
    """Test case for ONNX export.

    Any test case that tests functionalities under torch.onnx should inherit from this class.
    """

    # 在每个测试运行之前执行的设置方法
    def setUp(self):
        # 调用父类的setUp方法，执行通用的测试准备工作
        super().setUp()
        # TODO(#88264): Flaky test failures after changing seed.
        # 设置随机数种子为0，用于减少测试的随机性带来的影响
        set_rng_seed(0)
        # 如果CUDA可用，设置所有CUDA设备的随机数种子为0
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        # 清除诊断工具引擎的状态，准备进行新一轮的测试
        diagnostics.engine.clear()
```