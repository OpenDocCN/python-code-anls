# `.\pytorch\test\onnx\onnx_test_common.py`

```py
# Owner(s): ["module: onnx"]

from __future__ import annotations

import contextlib

import copy
import dataclasses
import io
import logging
import os
import unittest
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

import onnxruntime
import pytest
import pytorch_test_common

import torch
from torch import export as torch_export
from torch.onnx import _constants, verification
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics
from torch.testing._internal import common_utils
from torch.testing._internal.opinfo import core as opinfo_core
from torch.types import Number

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
_InputArgsType = Optional[
    Union[torch.Tensor, int, float, bool, Sequence[Any], Mapping[str, Any]]
]
_OutputsType = Sequence[_NumericType]

onnx_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    "repos",
    "onnx",
    "onnx",
    "backend",
    "test",
    "data",
)

pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")

pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")

def run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    """Run an ONNX model test using specified options.

    Args:
        test_suite (_TestONNXRuntime): Test suite object containing ONNX model test details.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Verification result of the ONNX model test.
    """
    options = verification.VerificationOptions()

    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    if hasattr(test_suite, "check_shape"):
        options.check_shape = test_suite.check_shape
    if hasattr(test_suite, "check_dtype"):
        options.check_dtype = test_suite.check_dtype

    names = {f.name for f in dataclasses.fields(options)}
    keywords_to_pop = []
    for k, v in kwargs.items():
        if k in names:
            setattr(options, k, v)
            keywords_to_pop.append(k)
    for k in keywords_to_pop:
        kwargs.pop(k)

    return verification.verify(*args, options=options, **kwargs)


def assert_dynamic_shapes(onnx_program: torch.onnx.ONNXProgram, dynamic_shapes: bool):
    """Assert whether the exported model has dynamic shapes or not.

    Args:
        onnx_program (torch.onnx.ONNXProgram): The output of torch.onnx.dynamo_export.
        dynamic_shapes (bool): Whether the exported model has dynamic shapes or not.
            When True, raises if graph inputs don't have at least one dynamic dimension
            When False, raises if graph inputs have at least one dynamic dimension.

    Raises:
        AssertionError: If the exported model has dynamic shapes and dynamic_shapes is False and vice-versa.
    """

    if dynamic_shapes is None:
        return

    model_proto = onnx_program.model_proto
    # Process graph inputs
    dynamic_inputs = []
    # 遍历模型协议中图形的输入列表
    for inp in model_proto.graph.input:
        # 将动态输入的维度信息添加到列表 dynamic_inputs 中，
        # 条件是维度的值为 0 并且有参数化的维度名称
        dynamic_inputs += [
            dim
            for dim in inp.type.tensor_type.shape.dim
            if dim.dim_value == 0 and dim.dim_param != ""
        ]
    # 断言检查动态形状 dynamic_shapes 是否与动态输入的数量一致，
    # 如果不一致则抛出异常信息
    assert dynamic_shapes == (
        len(dynamic_inputs) > 0
    ), "Dynamic shape check failed for graph inputs"
# 定义一个函数，将类名与参数化的参数组合起来
def parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    # 根据输入的参数字典创建一个后缀，格式为 "_key1_value1_key2_value2"
    suffix = "_".join(f"{k}_{v}" for k, v in input_dicts.items())
    # 返回格式化后的类名，包含类的原始名称和参数化后的后缀
    return f"{cls.__name__}_{suffix}"


class _TestONNXRuntime(pytorch_test_common.ExportTestCase):
    # 设置ONNX默认的操作集版本号
    opset_version = _constants.ONNX_DEFAULT_OPSET
    # 将初始值保留为输入，用于IR版本3类型导出
    keep_initializers_as_inputs = True
    # 是否使用脚本模式
    is_script = False
    # 是否检查形状
    check_shape = True
    # 是否检查数据类型
    check_dtype = True

    def setUp(self):
        # 调用父类的setUp方法
        super().setUp()
        # 设置ONNX运行时的随机种子为0
        onnxruntime.set_seed(0)
        # 如果CUDA可用，设置所有CUDA设备的随机种子为0
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        # 设置环境变量"ALLOW_RELEASED_ONNX_OPSET_ONLY"为"0"
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
        # 启用脚本测试模式
        self.is_script_test_enabled = True

    # 导出的ONNX模型可能比PyTorch模型的输入少，这是由于常量折叠的结果
    # 这在单元测试中经常发生，我们广泛使用torch.size或torch.shape
    # 因此输出仅依赖于输入的形状，而不是值
    # remained_onnx_input_idx用于指示哪些PyTorch模型输入索引在ONNX模型中保留
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=torch.onnx.TrainingMode.EVAL,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        # 定义一个内部函数 `_run_test`，用于运行模型测试，并返回测试结果
        def _run_test(m, remained_onnx_input_idx, flatten=True, ignore_none=True):
            return run_model_test(
                self,
                m,
                input_args=input_args,
                input_kwargs=input_kwargs,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                additional_test_inputs=additional_test_inputs,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                remained_onnx_input_idx=remained_onnx_input_idx,
                flatten=flatten,
                ignore_none=ignore_none,
                verbose=verbose,
            )

        # 如果 `remained_onnx_input_idx` 是一个字典，则分别提取脚本化和跟踪化的输入索引
        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            # 如果 `remained_onnx_input_idx` 不是字典，则将其用作脚本化和跟踪化的输入索引
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        # 检查模型是否是脚本模型（torch.jit.ScriptModule 或 torch.jit.ScriptFunction 的实例）
        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        # 如果启用了脚本测试，并且当前模型是脚本模型，则使用脚本模型进行测试
        if self.is_script_test_enabled and self.is_script:
            # 如果模型本身已经是脚本模型，则直接使用；否则将普通模型转换为脚本模型
            script_model = model if is_model_script else torch.jit.script(model)
            # 调用内部 `_run_test` 函数进行测试，使用脚本化的输入索引，并关闭扁平化和忽略空输入
            _run_test(
                script_model,
                scripting_remained_onnx_input_idx,
                flatten=False,
                ignore_none=False,
            )
        
        # 如果模型不是脚本模型，并且当前环境不是脚本模式，则使用普通模型进行测试
        if not is_model_script and not self.is_script:
            # 调用内部 `_run_test` 函数进行测试，使用跟踪化的输入索引
            _run_test(model, tracing_remained_onnx_input_idx)

    # 使用 Beartype 检查类型的装饰器，确保函数 `run_test_with_fx_to_onnx_exporter_and_onnx_runtime` 的参数类型正确
    @_beartype.beartype
    def run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
        self,
        model: _ModelType,
        input_args: Sequence[_InputArgsType],
        *,
        input_kwargs: Optional[Mapping[str, _InputArgsType]] = None,
        rtol: Optional[float] = 1e-3,
        atol: Optional[float] = 1e-7,
        has_mutation: bool = False,
        additional_test_inputs: Optional[
            List[
                Union[
                    Tuple[Sequence[_InputArgsType], Mapping[str, _InputArgsType]],
                    Tuple[Sequence[_InputArgsType]],
                ]
            ]
        ] = None,
        skip_dynamic_shapes_check: bool = False,
@_beartype.beartype
def run_ort(
    onnx_model: Union[str, torch.onnx.ONNXProgram],
    pytorch_inputs: Sequence[_InputArgsType],
) -> _OutputsType:
    """Run ORT on the given ONNX model and inputs

    Used in test_fx_to_onnx_with_onnxruntime.py

    Args:
        onnx_model (Union[str, torch.onnx.ONNXProgram]): Converter ONNX model
        pytorch_inputs (Sequence[_InputArgsType]): The given torch inputs

    Raises:
        AssertionError: ONNX and PyTorch should have the same input sizes

    Returns:
        _OutputsType: ONNX model predictions
    """
    # 如果onnx_model是torch.onnx.ONNXProgram类型，将其保存为字节流
    if isinstance(onnx_model, torch.onnx.ONNXProgram):
        buffer = io.BytesIO()
        onnx_model.save(buffer)
        ort_model = buffer.getvalue()
    else:
        ort_model = onnx_model

    # 抑制来自ONNX Runtime的大量警告
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3  # 错误级别
    # 创建ONNX Runtime的推理会话对象
    session = onnxruntime.InferenceSession(
        ort_model, providers=["CPUExecutionProvider"], sess_options=session_options
    )
    # 获取输入的名称列表
    input_names = [ort_input.name for ort_input in session.get_inputs()]

    # 检查输入名称的数量与给定的torch输入数量是否相同
    if len(input_names) != len(pytorch_inputs):
        raise AssertionError(
            f"Expected {len(input_names)} inputs, got {len(pytorch_inputs)}"
        )

    # 准备ONNX Runtime所需的输入字典
    ort_input = {
        k: torch.Tensor.numpy(v, force=True)
        for k, v in zip(input_names, pytorch_inputs)
    }
    # 运行ONNX Runtime会话并返回结果
    return session.run(None, ort_input)


@_beartype.beartype
def _try_clone_model(model: _ModelType) -> _ModelType:
    """Used for preserving original model in case forward mutates model states."""
    try:
        # 尝试深度复制模型
        return copy.deepcopy(model)
    except Exception:
        # 如果失败，发出警告并返回原模型
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification."
        )
        return model


@_beartype.beartype
def _try_clone_inputs(input_args, input_kwargs):
    # 深度复制输入参数和关键字参数
    ref_input_args = copy.deepcopy(input_args)
    ref_input_kwargs = copy.deepcopy(input_kwargs)
    return ref_input_args, ref_input_kwargs


@_beartype.beartype
def _compare_pytorch_onnx_with_ort(
    onnx_program: torch.onnx.ONNXProgram,
    model: _ModelType,
    input_args: Sequence[_InputArgsType],
    input_kwargs: Mapping[str, _InputArgsType],
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    has_mutation: bool = False,
):
    if has_mutation:
        # 如果有变化，尝试克隆模型和输入
        ref_model = _try_clone_model(model)
        ref_input_args, ref_input_kwargs = _try_clone_inputs(input_args, input_kwargs)
    else:
        # 否则，直接使用原始模型和输入
        ref_model = model
        ref_input_args = input_args
        ref_input_kwargs = input_kwargs

    # 注意：ONNXProgram保留对原始ref_model（包括其state_dict）的引用，而不是副本。
    # 因此，必须在运行ONNXProgram()之前运行ref_model()，以防止ref_model.forward()改变state_dict。
    # 否则，ref_model可能会更改state_dict上的缓冲区，而ONNXProgram.__call__()将使用这些更改后的state_dict。
    # 使用给定的输入参数调用 ONNX 程序，得到 ONNX 的输出
    ort_outputs = onnx_program(*input_args, **input_kwargs)
    # 使用参考模型和输入参数调用参考模型，得到参考模型的输出
    ref_outputs = ref_model(*ref_input_args, **ref_input_kwargs)
    # 将参考模型的输出转换为符合 ONNX 输出要求的格式
    ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(ref_outputs)

    # 检查输出的数量是否一致，如果不一致则抛出断言错误
    if len(ref_outputs) != len(ort_outputs):
        raise AssertionError(
            f"Expected {len(ref_outputs)} outputs, got {len(ort_outputs)}"
        )

    # 逐个比较参考模型输出和 ONNX 输出，使用指定的相对和绝对容忍度进行数值比较
    for ref_output, ort_output in zip(ref_outputs, ort_outputs):
        torch.testing.assert_close(
            ref_output, torch.tensor(ort_output), rtol=rtol, atol=atol
        )
# 定义最小的 ONNX 操作集版本进行测试
MIN_ONNX_OPSET_VERSION = 9
# 定义最大的 ONNX 操作集版本进行测试，使用常量 _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
MAX_ONNX_OPSET_VERSION = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
# 定义测试涵盖的操作集范围
TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

# 定义用于 FX（FX是一种PyTorch的模块化，函数式编程库）的最小 ONNX 操作集版本进行测试
FX_MIN_ONNX_OPSET_VERSION = 18
# 定义用于 FX 的最大 ONNX 操作集版本进行测试
FX_MAX_ONNX_OPSET_VERSION = 18
# 定义 FX 测试涵盖的操作集范围
FX_TESTED_OPSETS = range(FX_MIN_ONNX_OPSET_VERSION, FX_MAX_ONNX_OPSET_VERSION + 1)

# 布尔类型数据的集合
BOOL_TYPES = (torch.bool,)

# 整数类型数据的集合
INT_TYPES = (
    # torch.int8,  # 已注释掉，未在当前代码中使用
    # torch.int16,  # 已注释掉，未在当前代码中使用
    torch.int32,
    torch.int64,
    # torch.uint8,  # 已注释掉，未在当前代码中使用
)

# 量化整数类型数据的集合
QINT_TYPES = (
    torch.qint8,
    torch.quint8,
)

# 浮点数类型数据的集合
FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    # torch.float64,  # ORT（Open Neural Network Exchange）不支持双精度浮点数
)

# 复数类型数据的集合
COMPLEX_TYPES = (
    # torch.complex32,  # torch.complex32 在 torch 中为实验性质
    torch.complex64,
    # torch.complex128,  # ORT 不支持复数类型
)

# 所有测试的数据类型的集合
TESTED_DTYPES = (
    # 布尔类型
    torch.bool,
    # 整数类型
    *INT_TYPES,
    # 浮点数类型
    *FLOAT_TYPES,
    # 复数类型
    *COMPLEX_TYPES,
)


@dataclasses.dataclass
class DecorateMeta:
    """关于跳过或标记为xfail的测试用例的信息。

    适配自 functorch: functorch/test/common_utils.py

    Attributes:
        op_name: 操作符的名称。
        variant_name: OpInfo 变体的名称。
        decorator: 应用于测试用例的装饰器。
        opsets: 应用装饰器的操作集。
        dtypes: 应用装饰器的数据类型。
        reason: 跳过的原因。
        test_behavior: 测试用例的行为（跳过或xfail）。
        matcher: 应用于测试用例的匹配器。
        enabled_if: 是否启用测试行为。通常用于控制 onnx/ort 的版本。
        model_type: torch 模型的类型。默认为 None。
    """

    op_name: str
    variant_name: str
    decorator: Callable
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]]
    dtypes: Optional[Collection[torch.dtype]]
    reason: str
    test_behavior: str
    matcher: Optional[Callable[[Any], bool]] = None
    enabled_if: bool = True
    model_type: Optional[pytorch_test_common.TorchModelType] = None

    def contains_opset(self, opset: int) -> bool:
        """检查是否包含指定的操作集版本。

        Args:
            opset: 要检查的操作集版本。

        Returns:
            bool: 如果操作集在列表中则返回 True，否则返回 False。
        """
        if self.opsets is None:
            return True
        return any(
            opset == opset_spec if isinstance(opset_spec, int) else opset_spec(opset)
            for opset_spec in self.opsets
        )


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], bool]] = None,
    enabled_if: bool = True,
    model_type: Optional[pytorch_test_common.TorchModelType] = None,
):
    """标记一个 OpInfo 测试预期失败。

    Args:
        op_name: 操作符的名称。
        variant_name: OpInfo 变体的名称。默认为空字符串。
        reason: 测试预期失败的原因。
        opsets: 应用装饰器的操作集。
        dtypes: 应用装饰器的数据类型。
        matcher: 应用于测试用例的匹配器。
        enabled_if: 是否启用测试行为。默认为 True。
        model_type: torch 模型的类型。默认为 None。
    """
    # 根据参数创建一个装饰器元数据对象，用于标记测试为预期失败的测试用例
    def xfail_test(
        op_name,            # 操作符的名称
        variant_name,       # 变体的名称
        opsets,             # 预期失败的操作集合，例如 [9, 10] 或 [opsets_before(11)]
        dtypes,             # 预期失败的数据类型
        reason,             # 失败的原因
        matcher=None,       # 一个用于匹配测试样本输入的函数，仅在 xfail 在 SKIP_XFAIL_SUBTESTS 列表中时使用
        enabled_if=True,    # 是否启用 xfail，通常用于 onnx/ort 版本控制
        model_type=None     # torch 模型的类型，默认为 None
    ):
        return DecorateMeta(
            op_name=op_name,
            variant_name=variant_name,
            decorator=unittest.expectedFailure,  # 使用 unittest 的预期失败装饰器
            opsets=opsets,
            dtypes=dtypes,
            enabled_if=enabled_if,
            matcher=matcher,
            reason=reason,
            test_behavior="xfail",  # 测试行为标记为预期失败
            model_type=model_type,
        )
def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
    model_type: Optional[pytorch_test_common.TorchModelType] = None,
):
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable skip. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_behavior="skip",
        model_type=model_type,
    )


```  
def skip_slow(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    model_type: Optional[pytorch_test_common.TorchModelType] = None,
):
    """Skips a test case in OpInfo that is too slow.

    It needs further investigation to understand why it is slow.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=common_utils.slowTest,
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=not common_utils.TEST_WITH_SLOW,
        test_behavior="skip",
        model_type=model_type,
    )


```py  
def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    opset: int,
    skip_or_xfails: Iterable[DecorateMeta],
):
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list.

    Args:
        all_opinfos: A sequence of OpInfo objects representing all operations to test.
        test_class_name: The name of the test class.
        base_test_name: The base name of the test.
        opset: The opset version being tested.
        skip_or_xfails: An iterable of DecorateMeta objects specifying decorators
            for skipped or expected-to-fail tests.
    """
    Args:
        all_opinfos: All OpInfos.
        test_class_name: The name of the test class.
        base_test_name: The name of the test method.
        opset: The opset to decorate for.
        skip_or_xfails: DecorateMeta's list containing skip or xfail configurations.
    """
    # Create a mapping from (op name, variant test name) to OpInfo objects using all_opinfos
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    
    # Iterate over each DecorateMeta object in skip_or_xfails
    for decorate_meta in skip_or_xfails:
        # Check if the current decorate_meta applies to the specified opset
        if not decorate_meta.contains_opset(opset):
            # Skip processing if decorate_meta does not apply to this opset
            continue
        
        # Retrieve OpInfo corresponding to (op_name, variant_name) from ops_mapping
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        
        # Ensure that OpInfo is found for decorate_meta; otherwise, raise an assertion error
        assert opinfo is not None, f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
        
        # Ensure that model_type is not specified for decorate_meta
        assert decorate_meta.model_type is None, (
            f"Tested op: {decorate_meta.op_name} in wrong position! "
            "If model_type needs to be specified, it should be "
            "put under SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE."
        )
        
        # Create a new DecorateInfo object using decorate_meta details
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            active_if=decorate_meta.enabled_if,
        )
        
        # Append the new_decorator to the decorators list of opinfo
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # Define a decorator function wrapped that returns its input function fn unchanged
    def wrapped(fn):
        return fn

    # Return the wrapped decorator function
    return wrapped
# 返回一个比较函数，用于判断给定的操作集是否在指定操作集之前
def opsets_before(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is before the specified."""
    
    def compare(other_opset: int):
        return other_opset < opset
    return compare


# 返回一个比较函数，用于判断给定的操作集是否在指定操作集之后
def opsets_after(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is after the specified."""
    
    def compare(other_opset: int):
        return other_opset > opset
    return compare


# 格式化原因：ONNX 脚本不支持给定的数据类型
def reason_onnx_script_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX script doesn't support the given dtypes."""
    
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX script"


# 格式化原因：ONNX Runtime 不支持给定的数据类型
def reason_onnx_runtime_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX Runtime doesn't support the given dtypes."""
    
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX Runtime"


# 格式化原因：ONNX 不支持给定的数据类型
def reason_onnx_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX doesn't support the given dtypes."""
    
    return f"{operator} on {dtypes or 'certain dtypes'} not supported by the ONNX Spec"


# 格式化原因：Dynamo 不支持给定的数据类型
def reason_dynamo_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: Dynamo doesn't support the given dtypes."""
    
    return f"{operator} on {dtypes or 'certain dtypes'} not supported by the Dynamo Spec"


# 格式化原因：JIT 追踪器错误
def reason_jit_tracer_error(info: str) -> str:
    """Formats the reason: JIT tracer errors."""
    
    return f"JIT tracer error on {info}"


# 格式化原因：测试不稳定
def reason_flaky() -> str:
    """Formats the reason: test is flaky."""
    
    return "flaky test"


# 正常情况下，处理 xfail 和 skip 测试的不同行为的上下文管理器
@contextlib.contextmanager
def normal_xfail_skip_test_behaviors(
    test_behavior: Optional[str] = None, reason: Optional[str] = None
):
    """This context manager is used to handle the different behaviors of xfail and skip.
    
    Args:
        test_behavior (optional[str]): From DecorateMeta name, can be 'skip', 'xfail', or None.
        reason (optional[str]): The reason for the failure or skip.
    
    Raises:
        e: Any exception raised by the test case if it's not an expected failure.
    """
    
    # 尽快跳过，因为 SegFault 也可能是一个情况。
    if test_behavior == "skip":
        pytest.skip(reason=reason)

    try:
        yield
    # 我们可以使用 `except (AssertionError, RuntimeError, ...) as e:`，但需要遍历所有测试用例以找到正确的异常类型。
    except Exception as e:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise e
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
```