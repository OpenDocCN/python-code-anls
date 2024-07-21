# `.\pytorch\test\onnx\test_fx_op_consistency.py`

```
# 所有权归 module: onnx 所有

"""测试 torch.onnx FX 导出操作符与相同输入的 torch 操作符的输出值一致性。

用法:

    1. 测试所有操作符:

    pytest test/onnx/test_fx_op_consistency.py

    2. 运行特定操作符的测试（例如 torch.ceil）:

    pytest test/onnx/test_fx_op_consistency.py -k ceil
    pytest test/onnx/test_fx_op_consistency.py -k nn_functional_scaled_dot_product_attention

    3. 设置 `CREATE_REPRODUCTION_REPORT=1` 来创建错误重现的 markdown 文件。例如:

    CREATE_REPRODUCTION_REPORT=1 python -m pytest test/onnx/test_fx_op_consistency.py -k div_mode_int

    注意：阅读更多关于运行和编写测试的信息：
        https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests

注意事项:

    1. 请确保安装了 pytest-subtests。否则，子测试将被忽略。

    2. 如果目标是运行所有测试，请安装 pytest-xdist 以并行运行测试。

    3. 当支持新的操作时，请向下滚动修改 EXPECTED_SKIPS_OR_FAILS_WITH_DTYPES 和
    TESTED_OPS 列表。见“修改此部分”。

"""

from __future__ import annotations

import copy  # 导入 copy 模块
import itertools  # 导入 itertools 模块
import os  # 导入 os 模块
from typing import (  # 导入类型提示
    Any,
    Callable,
    Collection,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import error_reproduction  # 导入 error_reproduction 模块

import onnx_test_common  # 导入 onnx_test_common 模块

import parameterized  # 导入 parameterized 模块
import pytest  # 导入 pytest 模块
import pytorch_test_common  # 导入 pytorch_test_common 模块
from onnx_test_common import skip, skip_slow, xfail  # 从 onnx_test_common 模块导入 skip, skip_slow, xfail

import torch  # 导入 torch 模块
from torch.onnx._internal.diagnostics import _rules  # 导入 torch.onnx._internal.diagnostics 模块中的 _rules
from torch.testing._internal import (  # 导入 torch.testing._internal 中的模块
    common_device_type,
    common_methods_invocations,
    common_utils,
)
from torch.testing._internal.opinfo import core as opinfo_core  # 导入 torch.testing._internal.opinfo.core 并重命名为 opinfo_core
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,


    # 定义一个可选的匹配器函数类型的参数matcher，默认为None，可以接受任意类型参数和返回值
    matcher: Optional[Callable[[Any], Any]] = None,
    # 定义一个布尔类型的参数enabled_if，默认为True，用于控制特定功能的开关状态
    enabled_if: bool = True,
# Prefer using xfail_torchlib_forward_compatibility over this (skip) when possible.
# Only skip when the test is not failing consistently.
return skip(
    # 操作名称
    op_name,
    # 变体名称
    variant_name=variant_name,
    # 跳过的原因，包括 GitHub 问题链接
    reason=f"{reason}. GitHub Issue: {github_issue}",
    # 操作集
    opsets=opsets,
    # 数据类型
    dtypes=dtypes,
    # 匹配器
    matcher=matcher,
    # 如果启用条件
    enabled_if=enabled_if,
)
    xfail(
        "add", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Add")
    ),
    # 标记测试为预期失败的情况，测试名称为 "add"，数据类型为布尔类型，失败原因是 ONNX 不支持 "Add" 操作

    xfail(
        "add",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "Add", "int8, int16, uint8 have type issue."
        ),
    ),
    # 标记测试为预期失败的情况，测试名称为 "add"，数据类型包括 uint8、int8 和 int16，失败原因是 ONNX 脚本不支持这些类型的 "Add" 操作，存在类型问题

    xfail(
        "addbmm",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addbmm", "complex64")
    ),
    # 标记测试为预期失败的情况，测试名称为 "addbmm"，数据类型为复杂类型，失败原因是 Dynamo 不支持 "Addbmm" 操作中的 complex64 类型

    xfail(
        "addmm", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Addmm")
    ),
    # 标记测试为预期失败的情况，测试名称为 "addmm"，数据类型为布尔类型，失败原因是 ONNX 不支持 "Addmm" 操作

    xfail(
        "addmm",
        variant_name="decomposed",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Addmm")
    ),
    # 标记测试为预期失败的情况，测试名称为 "addmm"，变体名称为 "decomposed"，数据类型包括布尔类型和整数类型，失败原因是 ONNX 不支持 "Addmm" 操作

    skip(
        "addmm", dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addmm", "complex64 (core dump)")
    ),
    # 标记测试为跳过的情况，测试名称为 "addmm"，数据类型为复杂类型，跳过原因是 Dynamo 不支持 "Addmm" 操作中的 complex64 类型（核心转储）

    skip(
        "addmm",
        variant_name="decomposed",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addmm", "complex64 (core dump)")
    ),
    # 标记测试为跳过的情况，测试名称为 "addmm"，变体名称为 "decomposed"，数据类型为复杂类型，跳过原因是 Dynamo 不支持 "Addmm" 操作中的 complex64 类型（核心转储）

    xfail(
        "addr",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "Addr", "bool"
        ),
    ),
    # 标记测试为预期失败的情况，测试名称为 "addr"，数据类型为布尔类型，失败原因是 ONNX 脚本不支持 "Addr" 操作中的 bool 类型

    xfail(
        "addr",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addr", "complex64")
    ),
    # 标记测试为预期失败的情况，测试名称为 "addr"，数据类型为复杂类型，失败原因是 Dynamo 不支持 "Addr" 操作中的 complex64 类型

    xfail(
        "alias_copy",
        reason="OnnxExporterError: Failed to export model",
    ),
    # 标记测试为预期失败的情况，测试名称为 "alias_copy"，失败原因是 "OnnxExporterError: Failed to export model"

    xfail(
        "allclose",
        reason=onnx_test_common.reason_dynamo_does_not_support("Allclose")
    ),
    # 标记测试为预期失败的情况，测试名称为 "allclose"，失败原因是 Dynamo 不支持 "Allclose" 操作

    xfail(
        "amax",
        dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES),
        reason=onnx_test_common.reason_onnx_does_not_support("ReduceMin", "bool, int16"),
    ),
    # 标记测试为预期失败的情况，测试名称为 "amax"，数据类型包括 int16 和布尔类型，失败原因是 ONNX 不支持 "ReduceMin" 操作中的 bool 和 int16 类型

    xfail(
        "amin", dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES),
        reason=onnx_test_common.reason_dynamo_does_not_support("ReduceMin", "bool, int16")
    ),
    # 标记测试为预期失败的情况，测试名称为 "amin"，数据类型包括 int16 和布尔类型，失败原因是 Dynamo 不支持 "ReduceMin" 操作中的 bool 和 int16 类型

    xfail(
        "aminmax",
        dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES),
        reason=onnx_test_common.reason_onnx_does_not_support("ReduceMin", "bool, int16"),
    ),
    # 标记测试为预期失败的情况，测试名称为 "aminmax"，数据类型包括 int16 和布尔类型，失败原因是 ONNX 不支持 "ReduceMin" 操作中的 bool 和 int16 类型

    xfail(
        "arange",
        dtypes=(torch.uint8,),
        reason=onnx_test_common.reason_onnx_script_does_not_support("Arange", "uint8, int8"),
    ),
    # 标记测试为预期失败的情况，测试名称为 "arange"，数据类型为 uint8，失败原因是 ONNX 脚本不支持 "Arange" 操作中的 uint8 和 int8 类型

    xfail(
        "arange",
        dtypes=(torch.int16, torch.int32),
        reason="AssertionError: The values for attribute 'shape' do not match",
    ),
    # 标记测试为预期失败的情况，测试名称为 "arange"，数据类型为 int16 和 int32，失败原因是断言错误："The values for attribute 'shape' do not match"

    xfail(
        "argmax",
        dtypes=(
            torch.int16,
            torch.int64,
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ArgMax", "int16, int64"
        ),
    ),
    # 标记测试为预期失败的情况，测试名称为 "argmax"，数据类型包括 int16 和 int64，失败原因是 ONNX 运行时不支持 "ArgMax" 操作中的 int16 和 int64 类型
    xfail(
        "argmin",  # 标记为测试预期失败的函数名为 "argmin"
        dtypes=(
            torch.uint8,  # 数据类型为 torch.uint8
            torch.int8,   # 数据类型为 torch.int8
            torch.int16,  # 数据类型为 torch.int16
            torch.int64,  # 数据类型为 torch.int64
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ArgMin", "uint8, int8, int16, int64"  # 不支持的原因是 ONNX 运行时不支持这些数据类型
        ),
    ),
    xfail(
        "argwhere",  # 标记为测试预期失败的函数名为 "argwhere"
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    skip(
        "as_strided",  # 跳过测试的函数名为 "as_strided"
        variant_name="partial_views",  # 测试的变体名称为 "partial_views"
        reason="ONNX doesn't have partial view for tensor; [PostInline][ORT] segfaults",  # 跳过的原因是 ONNX 不支持张量的部分视图，会导致错误
    ),
    xfail(
        "atan2",  # 标记为测试预期失败的函数名为 "atan2"
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,  # 数据类型包括布尔类型和整数类型
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    xfail(
        "baddbmm",  # 标记为测试预期失败的函数名为 "baddbmm"
        dtypes=(
            torch.uint8,  # 数据类型为 torch.uint8
            torch.int8,   # 数据类型为 torch.int8
            torch.int16,  # 数据类型为 torch.int16
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Matmul", "uint8, int8, int16"  # 不支持的原因是 ONNX 运行时不支持这些数据类型的矩阵乘法
        ),
    ),
    xfail(
        "baddbmm",  # 标记为测试预期失败的函数名为 "baddbmm"
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 数据类型包括复数类型
        reason=onnx_test_common.reason_dynamo_does_not_support("baddbmm", "complex64")  # 不支持的原因是 Dynamo 不支持 complex64 类型的 baddbmm 操作
    ),
    xfail(
        "bernoulli",  # 标记为测试预期失败的函数名为 "bernoulli"
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 不支持的原因是 Dynamo 不支持 wrapper_set_seed 操作
    ),
    xfail(
        "bfloat16",  # 标记为测试预期失败的函数名为 "bfloat16"
        reason="fixme: ORT errors with RuntimeError: No corresponding Numpy type for Tensor Type.",  # 失败的原因是 ORT 在处理 bfloat16 时出现了 RuntimeError：没有对应的 Numpy 类型
    ),
    xfail(
        "bincount",  # 标记为测试预期失败的函数名为 "bincount"
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.bincount.default"),  # 不支持的原因是 Dynamo 不支持 aten.bincount.default 操作
    ),
    xfail(
        "block_diag",  # 标记为测试预期失败的函数名为 "block_diag"
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 数据类型包括复数类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Block_diag", "complex"),  # 不支持的原因是 ONNX 运行时不支持 complex 类型的 Block_diag 操作
    ),
    xfail(
        "bmm",  # 标记为测试预期失败的函数名为 "bmm"
        dtypes=(
            torch.uint8,  # 数据类型为 torch.uint8
            torch.int8,   # 数据类型为 torch.int8
            torch.int16,  # 数据类型为 torch.int16
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Matmul", "uint8, int8, int16"  # 不支持的原因是 ONNX 运行时不支持这些数据类型的矩阵乘法
        ),
    ),
    xfail(
        "broadcast_shapes",  # 标记为测试预期失败的函数名为 "broadcast_shapes"
        reason=onnx_test_common.reason_dynamo_does_not_support("output is int"),  # 不支持的原因是 Dynamo 不支持输出为 int 类型
    ),
    xfail(
        "cauchy",  # 标记为测试预期失败的函数名为 "cauchy"
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 不支持的原因是 Dynamo 不支持 wrapper_set_seed 操作
    ),
    skip(
        "ceil",  # 跳过测试的函数名为 "ceil"
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,  # 数据类型包括布尔类型和整数类型
        reason=onnx_test_common.reason_onnx_does_not_support("Ceil", "bool and int")  # 跳过的原因是 ONNX 不支持 bool 和 int 类型的 Ceil 操作
    ),
    xfail(
        "chalf",  # 标记为测试预期失败的函数名为 "chalf"
        reason="fixme: ONNX shape type inference error: Invalid tensor data type 0."  # 失败的原因是 ONNX 的形状类型推断错误：无效的张量数据类型 0
    ),
    xfail(
        "chunk",  # 标记为测试预期失败的函数名为 "chunk"
        dtypes=(torch.uint8, torch.int8, torch.int16,),  # 数据类型为 torch.uint8, torch.int8, torch.int16
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Chunk", "uint8, int8, int16"  # 不支持的原因是 ONNX 运行时不支持这些数据类型的 Chunk 操作
        ),
    ),
    xfail(
        "clamp",  # 标记为测试预期失败的函数名为 "clamp"
        dtypes=(torch.uint8, torch.int8, torch.int16,),  # 数据类型为 torch.uint8, torch.int8, torch.int16
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Max", "uint8, int8, int16"  # 不支持的原因是 ONNX 运行时不支持这些数据类型的 Max 操作
        ),
    ),
    xfail(
        "clamp_max", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Clamp_max", "bool")
    ),
    # 标记函数 "clamp_max" 在指定数据类型为布尔类型时预期失败，
    # 原因是 ONNX 脚本不支持 "Clamp_max" 操作应用于布尔类型数据
    
    xfail(
        "clamp_max",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Max", "uint8, int8, int16"
        ),
    ),
    # 标记函数 "clamp_max" 在指定数据类型为 torch.uint8, torch.int8, torch.int16 时预期失败，
    # 原因是 ONNX 运行时不支持 "Max" 操作应用于这些数据类型
    
    xfail(
        "clamp_min",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Max", "uint8, int8, int16"
        ),
    ),
    # 标记函数 "clamp_min" 在指定数据类型为 torch.uint8, torch.int8, torch.int16 时预期失败，
    # 原因是 ONNX 运行时不支持 "Max" 操作应用于这些数据类型
    
    xfail(
        "clamp_min", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Clamp_min", "bool")
    ),
    # 标记函数 "clamp_min" 在指定数据类型为布尔类型时预期失败，
    # 原因是 ONNX 脚本不支持 "Clamp_min" 操作应用于布尔类型数据
    
    xfail(
        "constant_pad_nd",
        dtypes=(torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Constant_pad_nd", "int16"
        ),
    ),
    # 标记函数 "constant_pad_nd" 在指定数据类型为 torch.int16 时预期失败，
    # 原因是 ONNX 运行时不支持 "Constant_pad_nd" 操作应用于 int16 数据类型
    
    xfail(
        "constant_pad_nd",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "Constant_pad_nd", "complex64"
        ),
    ),
    # 标记函数 "constant_pad_nd" 在复杂数据类型集合中预期失败，
    # 原因是 Dynamo 不支持 "Constant_pad_nd" 操作应用于 complex64 数据类型
    
    xfail(
        "corrcoef",
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "aten.equal.default"
        ),
    ),
    # 标记函数 "corrcoef" 预期失败，
    # 原因是 Dynamo 不支持 "aten.equal.default" 操作
    
    xfail(
        "cov",
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "aten.equal.default"
        ),
    ),
    # 标记函数 "cov" 预期失败，
    # 原因是 Dynamo 不支持 "aten.equal.default" 操作
    
    xfail(
        "cumsum", dtypes=onnx_test_common.BOOL_TYPES + (torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("Cumsum", "bool, uint8, int8, int16")
    ),
    # 标记函数 "cumsum" 在指定数据类型为布尔类型或 torch.uint8, torch.int8, torch.int16 时预期失败，
    # 原因是 ONNX 不支持 "Cumsum" 操作应用于这些数据类型
    
    xfail(
        "combinations",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.masked.select"),
    ),
    # 标记函数 "combinations" 预期失败，
    # 原因是 Dynamo 不支持 "aten.masked.select" 操作
    
    xfail(
        "diag",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Diagonal", "bool"),
    ),
    # 标记函数 "diag" 在指定数据类型为布尔类型时预期失败，
    # 原因是 ONNX 运行时不支持 "Diagonal" 操作应用于布尔类型数据
    
    xfail(
        "diagonal_copy",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Diagonal", "bool"),
    ),
    # 标记函数 "diagonal_copy" 在指定数据类型为布尔类型时预期失败，
    # 原因是 ONNX 运行时不支持 "Diagonal" 操作应用于布尔类型数据
    
    xfail(
        "dot", dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("MatMul", "uint8, int8, int16")
    ),
    # 标记函数 "dot" 在指定数据类型为 torch.uint8, torch.int8, torch.int16 时预期失败，
    # 原因是 ONNX 不支持 "MatMul" 操作应用于这些数据类型
    
    skip(
        "dot",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Dot", "complex64(core dump)"),
    ),
    # 标记函数 "dot" 在复杂数据类型集合中跳过测试，
    # 原因是 Dynamo 不支持 "Dot" 操作应用于 complex64 数据类型 (导致核心转储)
    
    xfail(
        "empty",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: kwargs dtpye=complex64 is not supported in ONNX."
    ),
    # 标记函数 "empty" 在复杂数据类型集合中预期失败，
    # 原因是 ONNX 不支持 complex64 数据类型
    
    xfail(
        "empty_strided",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记函数 "empty_strided" 预期失败，
    # 原因是 Dynamo 不支持 "wrapper_set_seed" 操作
    
    xfail(
        "eq",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Equal", "uint8, int8, int16"),
    ),
    # 标记函数 "eq" 在指定数据类型为 torch.uint8, torch.int8, torch.int16 时预期失败，
    # 原因是 ONNX 运行时不支持 "Equal" 操作应用于这些数据类型
    xfail(
        "equal",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.equal.default")
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 "equal" 操作
    xfail(
        "exponential",
        reason=onnx_test_common.reason_dynamo_does_not_support("exponential"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 "exponential" 操作
    xfail(
        "fft.fft",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 FFT 的结果不符合预期
    xfail(
        "fft.fft2",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 FFT2 的结果不符合预期
    xfail(
        "fft.fftn",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 FFTN 的结果不符合预期
    xfail(
        "fft.ifft",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 IFFT 的结果不符合预期
    xfail(
        "fft.ifft2",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 IFFT2 的结果不符合预期
    xfail(
        "fft.ifftn",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 IFFTN 的结果不符合预期
    xfail(
        "fft.irfft",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 IRFFT 的结果不符合预期
    xfail(
        "fft.irfft2",
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记为测试失败（xfail），因为当前环境下 IRFFT2 的结果不符合预期
    xfail(
        "fft.irfftn",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 IRFFTN 操作
    xfail(
        "fft.rfft",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 RFFT 操作
    xfail(
        "fft.rfftn",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 RFFTN 操作
    xfail(
        "fft.rfft2",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 RFFT2 操作
    xfail(
        "floor",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "bool, int"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 Floor 操作对布尔型和整型的输入
    xfail(
        "floor_divide",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "bool, int"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 Floor Divide 操作对布尔型和整型的输入
    xfail(
        "full",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("full", "complex64")
    ),
    # 标记为测试失败（xfail），因为当前环境不支持创建复杂类型（complex64）的 full 张量
    xfail(
        "full_like",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("full_like", "complex64")
    ),
    # 标记为测试失败（xfail），因为当前环境不支持创建与指定张量相同形状的复杂类型（complex64）的 full 张量
    xfail(
        "gather",
        reason="GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'"
    ),
    # 标记为测试失败（xfail），因为当前环境下 Gather 操作要求数据张量和索引张量的秩（Rank）相同
    xfail(
        "geometric",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持几何操作中的 set_seed 函数包装
    xfail(
        "heaviside",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Heaviside", "bool"),
    ),
    # 标记为测试失败（xfail），因为当前环境不支持 Heaviside 操作对布尔型的输入
    xfail(
        "index_add",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterND", "int64, int32, bool"),
    ),
    # 标记为测试失败（xfail），因为当前运行时环境不支持 ScatterND 操作对 float16 类型及其它指定类型的输入
    xfail(
        "index_fill",  # 标记为测试失败，测试函数为 index_fill
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 使用复杂类型的数据类型
        reason=onnx_test_common.reason_dynamo_does_not_support("index_fill", "complex64")  # 不支持 index_fill 函数及 complex64 类型的原因
    ),
    xfail(
        "index_fill",  # 标记为测试失败，测试函数为 index_fill
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES,  # 使用整数、布尔和浮点数类型的数据类型
        reason="fixme: Constant input list has None. ONNXScript does not support None in constant list."  # 修复常量输入列表中有 None 的问题，ONNXScript 不支持常量列表中的 None
    ),
    xfail(
        "index_put",  # 标记为测试失败，测试函数为 index_put
        dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,),  # 使用布尔类型和 torch.float16 数据类型
        reason=onnx_test_common.reason_onnx_script_does_not_support("index_put", "bool"),  # ONNXScript 不支持 index_put 函数及 bool 类型的原因
    ),
    xfail(
        "index_put",  # 标记为测试失败，测试函数为 index_put
        dtypes=(torch.uint8, torch.int8, torch.int16,),  # 使用 torch.uint8, torch.int8, torch.int16 数据类型
        reason=onnx_test_common.reason_onnx_script_does_not_support("Add", "int8, int16"),  # ONNXScript 不支持 Add 函数及 int8, int16 类型的原因
    ),
    xfail(
        "index_put",  # 标记为测试失败，测试函数为 index_put
        dtypes=(torch.float16,),  # 使用 torch.float16 数据类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterND", "float16"),  # ONNX 运行时不支持 ScatterND 函数及 float16 类型的原因
    ),
    xfail(
        "isnan",  # 标记为测试失败，测试函数为 isnan
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES,  # 使用整数和布尔类型的数据类型
        reason=onnx_test_common.reason_onnx_does_not_support("IsNaN", "int, bool"),  # ONNX 不支持 IsNaN 函数及 int, bool 类型的原因
    ),
    xfail(
        "istft",  # 标记为测试失败，测试函数为 istft
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),  # Dynamo 不支持 data-dependent 的原因
    ),
    xfail(
        "item",  # 标记为测试失败，测试函数为 item
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # Dynamo 不支持 wrapper_set_seed 的原因
    ),
    xfail(
        "lerp",  # 标记为测试失败，测试函数为 lerp
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 使用复杂类型的数据类型
        reason=onnx_test_common.reason_dynamo_does_not_support("lerp", "complex64")  # Dynamo 不支持 lerp 函数及 complex64 类型的原因
    ),
    xfail(
        "linalg.lstsq",  # 标记为测试失败，测试函数为 linalg.lstsq
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.linalg_lstsq.default"),  # Dynamo 不支持 aten.linalg_lstsq.default 的原因
    ),
    xfail(
        "linalg.lstsq",  # 标记为测试失败，测试函数为 linalg.lstsq
        variant_name="grad_oriented",  # 变体名称为 grad_oriented
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.linalg_lstsq.default"),  # Dynamo 不支持 aten.linalg_lstsq.default 的原因
    ),
    xfail(
        "linalg.matrix_power",  # 标记为测试失败，测试函数为 linalg.matrix_power
        reason="fixme: The values for attribute 'shape' do not match: torch.Size([2, 2]) != torch.Size([2, 2, 2])."  # 修复属性 'shape' 值不匹配的问题
    ),
    xfail(
        "linalg.norm",  # 标记为测试失败，测试函数为 linalg.norm
        reason="fixme: Assertion error: result mismatch",  # 修复断言错误：结果不匹配
    ),
    xfail(
        "linalg.norm",  # 标记为测试失败，测试函数为 linalg.norm
        variant_name="subgradients_at_zero",  # 变体名称为 subgradients_at_zero
        reason="fixme: Assertion error: result mismatch",  # 修复断言错误：结果不匹配
    ),
    xfail(
        "linalg.vecdot",  # 标记为测试失败，测试函数为 linalg.vecdot
        reason="fixme: Assertion error: result shape mismatch",  # 修复断言错误：结果形状不匹配
    ),
    xfail(
        "linspace",  # 标记为测试失败，测试函数为 linspace
        dtypes=(torch.int64, torch.int32,),  # 使用 torch.int64 和 torch.int32 数据类型
        reason="fixme: Results do not match with PyTorch. https://github.com/microsoft/onnxscript/issues/854",  # 修复结果与 PyTorch 不匹配的问题
    ),
    xfail(
        "linspace",  # 标记为测试失败，测试函数为 linspace
        variant_name="tensor_overload",  # 变体名称为 tensor_overload
        dtypes=(torch.int64, torch.int32,),  # 使用 torch.int64 和 torch.int32 数据类型
        reason="fixme: Results do not match with PyTorch. https://github.com/microsoft/onnxscript/issues/854",  # 修复结果与 PyTorch 不匹配的问题
    ),
    xfail(
        "linspace",  # 标记为测试失败，测试函数为 linspace
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 使用复杂类型的数据类型
        reason=onnx_test_common.reason_onnx_script_does_not_support("linspace", "complex64")  # ONNXScript 不支持 linspace 函数及 complex64 类型的原因
    ),
    xfail(
        "linspace",
        variant_name="tensor_overload",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("linspace", "complex64")
    ),
    xfail(
        "log_normal",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "log_softmax",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "log_softmax",
        variant_name="with_dtype",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "logical_and",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("And", "float, int"),
    ),
    xfail(
        "logical_not",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Not", "float, int"),
    ),
    xfail(
        "logical_or",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Or", "float, int"),
    ),
    xfail(
        "logical_xor",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Xor", "float, int"),
    ),
    skip(
        "masked.logsumexp",
        reason="fixme: https://github.com/onnx/onnx/issues/4986",
    ),
    xfail(
        "masked.amax",
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "masked.amin",
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "masked.argmin",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES + (torch.int64,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "masked.argmax",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES + (torch.int64,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "masked_fill",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "masked.sum",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "masked.log_softmax",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),


注释：


# 标记为测试失败（xfail），测试的函数为 "linspace"
# 使用 "tensor_overload" 变种名，测试数据类型为 onnx_test_common.COMPLEX_TYPES
# 原因是 onnx 脚本不支持 "linspace" 函数和 "complex64" 类型的组合
xfail(
    "linspace",
    variant_name="tensor_overload",
    dtypes=onnx_test_common.COMPLEX_TYPES,
    reason=onnx_test_common.reason_onnx_script_does_not_support("linspace", "complex64")
),

# 标记为测试失败（xfail），测试的函数为 "log_normal"
# 原因是 dynamo 不支持 "wrapper_set_seed" 函数
xfail(
    "log_normal",
    reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
),

# 标记为测试失败（xfail），测试的函数为 "log_softmax"
# 测试数据类型为 torch.float16
# 原因是 ORT 优化器出现错误，详细问题描述在指定的 GitHub 问题链接中
xfail(
    "log_softmax",
    dtypes=(torch.float16,),
    reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
),

# 标记为测试失败（xfail），测试的函数为 "log_softmax"，带有变种名 "with_dtype"
# 测试数据类型为 torch.float16
# 同样的原因是 ORT 优化器出现错误，详细问题描述在指定的 GitHub 问题链接中
xfail(
    "log_softmax",
    variant_name="with_dtype",
    dtypes=(torch.float16,),
    reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
),

# 标记为测试失败（xfail），测试的函数为 "logical_and"
# 测试数据类型为 onnx_test_common.FLOAT_TYPES 和 onnx_test_common.INT_TYPES 的组合
# 原因是 onnx 脚本不支持 "And" 函数和 "float, int" 类型的组合
xfail(
    "logical_and",
    dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_script_does_not_support("And", "float, int"),
),

# 标记为测试失败（xfail），测试的函数为 "logical_not"
# 测试数据类型为 onnx_test_common.FLOAT_TYPES 和 onnx_test_common.INT_TYPES 的组合
# 原因是 onnx 脚本不支持 "Not" 函数和 "float, int" 类型的组合
xfail(
    "logical_not",
    dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_script_does_not_support("Not", "float, int"),
),

# 标记为测试失败（xfail），测试的函数为 "logical_or"
# 测试数据类型为 onnx_test_common.FLOAT_TYPES 和 onnx_test_common.INT_TYPES 的组合
# 原因是 onnx 脚本不支持 "Or" 函数和 "float, int" 类型的组合
xfail(
    "logical_or",
    dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_script_does_not_support("Or", "float, int"),
),

# 标记为测试失败（xfail），测试的函数为 "logical_xor"
# 测试数据类型为 onnx_test_common.FLOAT_TYPES 和 onnx_test_common.INT_TYPES 的组合
# 原因是 onnx 脚本不支持 "Xor" 函数和 "float, int" 类型的组合
xfail(
    "logical_xor",
    dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_script_does_not_support("Xor", "float, int"),
),

# 标记为跳过测试（skip），测试的函数为 "masked.logsumexp"
# 原因是存在待修复的问题，详细问题描述在指定的 GitHub 问题链接中
skip(
    "masked.logsumexp",
    reason="fixme: https://github.com/onnx/onnx/issues/4986",
),

# 标记为测试失败（xfail），测试的函数为 "masked.amax"
# 原因是 ORT 优化器出现错误，详细问题描述在指定的 GitHub 问题链接中
xfail(
    "masked.amax",
    reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
),

# 标记为测试失败（xfail），测试的函数为 "masked.amin"
# 原因是 ORT 优化器出现错误，详细问题描述在指定的 GitHub 问题链接中
xfail(
    "masked.amin",
    reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
),

# 标记为测试失败（xfail），测试的函数为 "masked.argmin"
# 测试数据类型为 onnx_test_common.BOOL_TYPES、onnx_test_common.FLOAT_TYPES 和 torch.int64
# 原因是断言错误，结果不匹配
xfail(
    "masked.argmin",
    dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES + (torch.int64,),
    reason="fixme: Assertion error: result mismatch",
),

# 标记为测试失败（xfail），测试的函数为 "masked.argmax"
# 测试数据类型为 onnx_test_common.BOOL_TYPES、onnx_test_common.FLOAT_TYPES 和 torch.int64
# 原因是断言错误，结果不匹
    xfail(
        "masked.mean",  # 标记此测试为预期失败，测试函数为 masked.mean
        dtypes=onnx_test_common.BOOL_TYPES,  # 测试的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_does_not_support("ReduceMean", "bool"),  # 失败的原因是 ONNX 不支持 ReduceMean 操作的布尔类型
    ),
    xfail(
        "masked.norm",  # 标记此测试为预期失败，测试函数为 masked.norm
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    xfail(
        "masked.prod",  # 标记此测试为预期失败，测试函数为 masked.prod
        dtypes=onnx_test_common.BOOL_TYPES,  # 测试的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),  # 失败的原因是 ONNX 运行时不支持布尔类型的 Where 操作
    ),
    xfail(
        "masked_select",  # 标记此测试为预期失败，测试函数为 masked_select
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.masked_select.default"),  # 失败的原因是 Dynamo 不支持 aten.masked_select.default
    ),
    xfail(
        "max",  # 标记此测试为预期失败，测试函数为 max
        variant_name="reduction_no_dim",  # 测试的变体名称为 reduction_no_dim
        dtypes=onnx_test_common.BOOL_TYPES,  # 测试的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMax", "bool"),  # 失败的原因是 ONNX 运行时不支持布尔类型的 ReduceMax 操作
    ),
    xfail(
        "max",  # 标记此测试为预期失败，测试函数为 max
        variant_name="reduction_with_dim",  # 测试的变体名称为 reduction_with_dim
        dtypes=onnx_test_common.BOOL_TYPES,  # 测试的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMax", "bool"),  # 失败的原因是 ONNX 运行时不支持布尔类型的 ReduceMax 操作
    ),
    xfail(
        "max",  # 标记此测试为预期失败，测试函数为 max
        variant_name="reduction_with_dim",  # 测试的变体名称为 reduction_with_dim
        dtypes=(torch.int64,),  # 测试的数据类型为 torch.int64
        reason="https://github.com/onnx/onnx/issues/4986",  # 失败的原因是 ONNX 存在问题：https://github.com/onnx/onnx/issues/4986
    ),
    xfail(
        "min",  # 标记此测试为预期失败，测试函数为 min
        variant_name="reduction_no_dim",  # 测试的变体名称为 reduction_no_dim
        dtypes=onnx_test_common.BOOL_TYPES,  # 测试的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMin", "bool"),  # 失败的原因是 ONNX 运行时不支持布尔类型的 ReduceMin 操作
    ),
    xfail(
        "min",  # 标记此测试为预期失败，测试函数为 min
        variant_name="reduction_with_dim",  # 测试的变体名称为 reduction_with_dim
        dtypes=onnx_test_common.BOOL_TYPES + (torch.int64,),  # 测试的数据类型包括布尔类型和 torch.int64
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMin", "bool"),  # 失败的原因是 ONNX 运行时不支持布尔类型的 ReduceMin 操作
    ),
    skip(
        "mm",  # 标记此测试为跳过，测试函数为 mm
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 测试的数据类型为复杂类型
        reason=onnx_test_common.reason_dynamo_does_not_support("MM", "complex64(core dump)"),  # 跳过的原因是 Dynamo 不支持 MM 操作的 complex64 类型（核心转储）
    ),
    xfail(
        "multinomial",  # 标记此测试为预期失败，测试函数为 multinomial
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 失败的原因是 Dynamo 不支持 wrapper_set_seed
    ),
    xfail(
        "nanquantile",  # 标记此测试为预期失败，测试函数为 nanquantile
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.equal.default")  # 失败的原因是 Dynamo 不支持 aten.equal.default
    ),
    xfail(
        "nansum",  # 标记此测试为预期失败，测试函数为 nansum
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES,  # 测试的数据类型包括整数和布尔类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("IsNaN", "int, bool"),  # 失败的原因是 ONNX 运行时不支持整数和布尔类型的 IsNaN 操作
    ),
    xfail(
        "narrow",  # 标记此测试为预期失败，测试函数为 narrow
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),  # 失败的原因是 Dynamo 不支持 data-dependent
    ),
    skip(
        "native_batch_norm",  # 标记此测试为跳过，测试函数为 native_batch_norm
        reason=onnx_test_common.reason_onnx_script_does_not_support("cpu is not supported: \
            https://github.com/microsoft/onnxscript/pull/1289")  # 跳过的原因是 ONNX 脚本不支持 CPU：https://github.com/microsoft/onnxscript/pull/1289
    ),
    xfail(
        "native_layer_norm",  # 标记此测试为预期失败，测试函数为 native_layer_norm
        dtypes=(torch.float16,),  # 测试的数据类型为 torch.float16
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",  # 失败的原因是 ORT 优化器错误：https://github.com/microsoft/onnxruntime/issues/16438
    ),
    xfail(
        "new_full",  # 标记此测试为预期失败，测试函数为 new_full
        dtypes=onnx_test_common.COMPLEX_TYPES,  # 测试的数据类型为复杂类型
        reason=onnx_test_common.reason_dynamo_does_not_support("new_full", "complex64")  # 失败的原因是 Dynamo 不支持 new_full 操作的 complex64 类型
    ),
    xfail(
        "nn.functional.adaptive_avg_pool2d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("RecursionError: \
            maximum recursion depth exceeded while calling a Python object"),
    ),
    xfail(
        "nn.functional.adaptive_avg_pool3d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._adaptive_avg_pool3d.default"),
    ),
    xfail(
        "nn.functional.alpha_dropout",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.avg_pool1d",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
    ),
    xfail(
        "nn.functional.avg_pool2d",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
    ),
    xfail(
        "nn.functional.avg_pool3d",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
    ),
    xfail(
        "nn.functional.batch_norm",
        dtypes=(torch.float16,),
        reason="fixme: https://github.com/microsoft/onnxscript/issues/1270",
    ),
    xfail(
        "nn.functional.conv_transpose1d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv1d", "int64"),
    ),
    xfail(
        "nn.functional.conv_transpose2d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv2d", "int64"),
    ),
    xfail(
        "nn.functional.conv_transpose3d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv3d", "int64"),
    ),
    skip(
        "nn.functional.conv_transpose1d",
        reason="fixme: Assertion error: result mismatch",
    ),
    skip(
        "nn.functional.conv_transpose2d",
        reason="fixme: Assertion error: result mismatch",
    ),
    skip(
        "nn.functional.conv_transpose3d",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.conv1d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv1d", "int64"),
    ),
    xfail(
        "nn.functional.conv2d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv2d", "int64"),
    ),
    xfail(
        "nn.functional.conv2d",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.conv3d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv3d", "int64"),
    ),
    xfail(
        "nn.functional.conv3d",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),



# 该代码块包含一系列的测试标记函数调用

xfail(
    "nn.functional.adaptive_avg_pool2d",
    # 使用自定义函数返回不支持的原因
    reason=onnx_test_common.reason_onnx_script_does_not_support("RecursionError: \
        maximum recursion depth exceeded while calling a Python object"),
),

xfail(
    "nn.functional.adaptive_avg_pool3d",
    # 使用自定义函数返回不支持的原因
    reason=onnx_test_common.reason_onnx_script_does_not_support("aten._adaptive_avg_pool3d.default"),
),

xfail(
    "nn.functional.alpha_dropout",
    # 使用自定义函数返回不支持的原因
    reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
),

xfail(
    "nn.functional.avg_pool1d",
    # 指定数据类型为整型，使用自定义函数返回不支持的原因
    dtypes=onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
),

xfail(
    "nn.functional.avg_pool2d",
    # 指定数据类型为整型，使用自定义函数返回不支持的原因
    dtypes=onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
),

xfail(
    "nn.functional.avg_pool3d",
    # 指定数据类型为整型，使用自定义函数返回不支持的原因
    dtypes=onnx_test_common.INT_TYPES,
    reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
),

xfail(
    "nn.functional.batch_norm",
    # 指定数据类型为 torch.float16，说明原因为某个问题的链接
    dtypes=(torch.float16,),
    reason="fixme: https://github.com/microsoft/onnxscript/issues/1270",
),

xfail(
    "nn.functional.conv_transpose1d",
    # 指定数据类型为 torch.int64，使用自定义函数返回不支持的原因
    dtypes=(torch.int64,),
    reason=onnx_test_common.reason_onnx_does_not_support("Conv1d", "int64"),
),

xfail(
    "nn.functional.conv_transpose2d",
    # 指定数据类型为 torch.int64，使用自定义函数返回不支持的原因
    dtypes=(torch.int64,),
    reason=onnx_test_common.reason_onnx_does_not_support("Conv2d", "int64"),
),

xfail(
    "nn.functional.conv_transpose3d",
    # 指定数据类型为 torch.int64，使用自定义函数返回不支持的原因
    dtypes=(torch.int64,),
    reason=onnx_test_common.reason_onnx_does_not_support("Conv3d", "int64"),
),

skip(
    "nn.functional.conv_transpose1d",
    # 跳过的原因是断言错误导致结果不匹配
    reason="fixme: Assertion error: result mismatch",
),

skip(
    "nn.functional.conv_transpose2d",
    # 跳过的原因是断言错误导致结果不匹配
    reason="fixme: Assertion error: result mismatch",
),

skip(
    "nn.functional.conv_transpose3d",
    # 跳过的原因是断言错误导致结果不匹配
    reason="fixme: Assertion error: result mismatch",
),

xfail(
    "nn.functional.conv1d",
    # 指定数据类型为 torch.int64，使用自定义函数返回不支持的原因
    dtypes=(torch.int64,),
    reason=onnx_test_common.reason_onnx_does_not_support("Conv1d", "int64"),
),

xfail(
    "nn.functional.conv2d",
    # 指定数据类型为 torch.int64，使用自定义函数返回不支持的原因
    dtypes=(torch.int64,),
    reason=onnx_test_common.reason_onnx_does_not_support("Conv2d", "int64"),
),

xfail(
    "nn.functional.conv2d",
    # 指定数据类型为复杂类型，说明原因是断言错误导致结果不匹配
    dtypes=onnx_test_common.COMPLEX_TYPES,
    reason="fixme: Assertion error: result mismatch",
),

xfail(
    "nn.functional.conv3d",
    # 指定数据类型为 torch.int64，使用自定义函数返回不支持的原因
    dtypes=(torch.int64,),
    reason=onnx_test_common.reason_onnx_does_not_support("Conv3d", "int64"),
),

xfail(
    "nn.functional.conv3d",
    # 指定数据类型为复杂类型，说明原因是断言错误导致结果不匹配
    dtypes=onnx_test_common.COMPLEX_TYPES,
    reason="fixme: Assertion error: result mismatch",
),
    # 将 "nn.functional.cosine_embedding_loss" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.cosine_embedding_loss",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("CosineEmbeddingLoss", "bool"),
    ),
    
    # 将 "nn.functional.ctc_loss" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.ctc_loss",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.ctc_loss.default"),
    ),
    
    # 将 "nn.functional.dropout" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.dropout",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.dropout2d" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.dropout2d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.dropout3d" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.dropout3d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.feature_alpha_dropout" 标记为失败，用于特定的 ONNX 测试用例（带训练）
    xfail(
        "nn.functional.feature_alpha_dropout",
        variant_name="with_train",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.feature_alpha_dropout" 标记为失败，用于特定的 ONNX 测试用例（不带训练）
    xfail(
        "nn.functional.feature_alpha_dropout",
        variant_name="without_train",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.fractional_max_pool2d" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.fractional_max_pool2d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.fractional_max_pool3d" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.fractional_max_pool3d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.gaussian_nll_loss" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.gaussian_nll_loss",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.gaussian_nll_loss"),
    ),
    
    # 将 "nn.functional.grid_sample" 标记为失败，原因是结果不匹配
    xfail(
        "nn.functional.grid_sample",
        reason="fixme: Assertion error: result mismatch",
    ),
    
    # 将 "nn.functional.group_norm" 标记为失败，用于特定的 ONNX 测试用例（float16 数据类型）
    xfail(
        "nn.functional.group_norm",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("GroupNormalization", "float16"),
    ),
    
    # 将 "nn.functional.local_response_norm" 标记为失败，用于特定的 ONNX 测试用例（int64 数据类型）
    xfail(
        "nn.functional.local_response_norm",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("avgpool", "int64"),
    ),
    
    # 将 "nn.functional.linear" 标记为失败，用于特定的 ONNX 测试用例（多种整数数据类型）
    xfail(
        "nn.functional.linear",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Gemm", "int"),
    ),
    
    # 将 "nn.functional.max_pool2d" 标记为失败，用于特定的 ONNX 测试用例（布尔和整数数据类型）
    xfail(
        "nn.functional.max_pool2d",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Max_pool2d"),
    ),
    
    # 将 "nn.functional.max_pool3d" 标记为失败，用于特定的 ONNX 测试用例（布尔和整数数据类型）
    xfail(
        "nn.functional.max_pool3d",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Max_pool3d"),
    ),
    
    # 将 "nn.functional.multi_head_attention_forward" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.multi_head_attention_forward",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    
    # 将 "nn.functional.one_hot" 标记为失败，用于特定的 ONNX 测试用例
    xfail(
        "nn.functional.one_hot",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    xfail(
        "nn.functional.pad",  # 标记这是对 nn.functional.pad 函数的测试，预期是失败的情况
        variant_name="replicate",  # 测试使用的变体名称为 "replicate"
        reason="fixme: ORT error: padding size",  # 失败的原因是 ORT 错误：填充大小问题
    ),
    xfail(
        "nn.functional.pad",
        variant_name="replicate_negative",
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    xfail(
        "nn.functional.pad",
        variant_name="reflect",
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    xfail(
        "nn.functional.pixel_shuffle",
        dtypes=(torch.int32, torch.int64) + onnx_test_common.BOOL_TYPES,
        reason="fixme: ONNX Runtime does not support int32/64 inputs",  # 失败的原因是 ONNX Runtime 不支持 int32/64 的输入类型
    ),
    xfail(
        "nn.functional.pixel_unshuffle",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten.pixel_unshuffle.default"),  # 失败的原因是 ONNX 脚本不支持 "aten.pixel_unshuffle.default"
    ),
    xfail(
        "nn.functional.poisson_nll_loss",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason="fixme: result mismatch with NaN.",  # 失败的原因是结果与 NaN 不匹配
    ),
    xfail(
        "nn.functional.rrelu",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 失败的原因是 dynamo 不支持 "wrapper_set_seed"
    ),
    xfail(
        "nn.functional.rrelu",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Relu", "int64"),  # 失败的原因是 ONNX Runtime 不支持 "Relu" 和 "int64"
    ),
    skip(
        "nn.functional.scaled_dot_product_attention",
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,  # 跳过的条件是 dropout_p 不等于 0.0
        reason="dropout is random so the results do not match",  # 跳过的原因是 dropout 是随机的，结果不匹配
    ),
    xfail(
        "nn.functional.scaled_dot_product_attention",
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",  # 失败的原因是 ORT 失败，附带问题链接
    ),
    xfail(
        "nn.functional.selu",
        reason="fixme: nn.functional.selu is not in torch._decomp.decomposition_table",  # 失败的原因是 nn.functional.selu 不在 torch._decomp.decomposition_table 中
    ),
    xfail(
        "nn.functional.soft_margin_loss",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    xfail(
        "nn.functional.tanhshrink",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",  # 失败的原因是断言错误：结果不匹配
    ),
    xfail(
        "nonzero",
        dtypes=(torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("NonZero", "int8, int16"),  # 失败的原因是 ONNX Runtime 不支持 "NonZero", "int8, int16"
    ),
    xfail(
        "normal",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 失败的原因是 dynamo 不支持 "wrapper_set_seed"
    ),
    xfail(
        "normal",
        variant_name="in_place",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 失败的原因是 dynamo 不支持 "wrapper_set_seed"
    ),
    xfail(
        "normal",
        variant_name="number_mean",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 失败的原因是 dynamo 不支持 "wrapper_set_seed"
    ),
    xfail(
        "ones",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: kwargs dtpye=complex64 is not supported in ONNX."  # 失败的原因是 kwargs 的 complex64 类型在 ONNX 中不被支持
    ),
    xfail(
        "pca_lowrank",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 失败的原因是 dynamo 不支持 "wrapper_set_seed"
    ),
    xfail(
        "quantile",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.equal.default")
    ),
    # 标记 "quantile" 函数为测试失败，原因是 ONNX 运行时不支持 "aten.equal.default" 函数

    xfail(
        "rand_like",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记 "rand_like" 函数为测试失败，原因是 ONNX 运行时不支持 "wrapper_set_seed" 函数

    xfail(
        "randint",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记 "randint" 函数为测试失败，原因是 ONNX 运行时不支持 "wrapper_set_seed" 函数

    xfail(
        "randint_like",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记 "randint_like" 函数为测试失败，原因是 ONNX 运行时不支持 "wrapper_set_seed" 函数

    xfail(
        "randn",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记 "randn" 函数为测试失败，原因是 ONNX 运行时不支持 "wrapper_set_seed" 函数

    xfail(
        "randn_like",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记 "randn_like" 函数为测试失败，原因是 ONNX 运行时不支持 "wrapper_set_seed" 函数

    xfail(
        "resize_",
        reason=onnx_test_common.reason_dynamo_does_not_support("resize_as_")
    ),
    # 标记 "resize_" 函数为测试失败，原因是 ONNX 运行时不支持 "resize_as_" 函数

    xfail(
        "resize_as_",
        reason=onnx_test_common.reason_dynamo_does_not_support("resize_as_")
    ),
    # 标记 "resize_as_" 函数为测试失败，原因是 ONNX 运行时不支持 "resize_as_" 函数

    xfail(
        "round",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Round", "int"),
    ),
    # 标记 "round" 函数为测试失败，限定数据类型为整数类型，原因是 ONNX 运行时不支持 "Round" 函数在整数上的操作

    xfail(
        "rsub",
        dtypes=(torch.uint8, torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Mul", "uint8, int8, int16"
        ),
    ),
    # 标记 "rsub" 函数为测试失败，限定数据类型为 uint8、int8 和 int16，原因是 ONNX 运行时不支持这些数据类型上的 "Mul" 函数操作

    xfail(
        "scatter_add",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=sum", "float16"),
    ),
    # 标记 "scatter_add" 函数为测试失败，限定数据类型为 float16，原因是 ONNX 运行时不支持 "ScatterElements reduction=sum" 操作在 float16 上的应用

    xfail(
        "scatter_reduce",
        variant_name="sum",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=sum", "float16"),
    ),
    # 标记 "scatter_reduce" 函数为测试失败，variant_name="sum"，限定数据类型为 float16，原因是 ONNX 运行时不支持 "ScatterElements reduction=sum" 操作在 float16 上的应用

    xfail(
        "scatter_reduce",
        variant_name="prod",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=prod", "float16"),
    ),
    # 标记 "scatter_reduce" 函数为测试失败，variant_name="prod"，限定数据类型为 float16，原因是 ONNX 运行时不支持 "ScatterElements reduction=prod" 操作在 float16 上的应用

    xfail(
        "scatter_reduce",
        variant_name="amin",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=amin", "float16"),
    ),
    # 标记 "scatter_reduce" 函数为测试失败，variant_name="amin"，限定数据类型为布尔类型和 float16，原因是 ONNX 运行时不支持 "ScatterElements reduction=amin" 操作在 float16 上的应用

    xfail(
        "scatter_reduce",
        variant_name="amax",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=amax", "float16"),
    ),
    # 标记 "scatter_reduce" 函数为测试失败，variant_name="amax"，限定数据类型为布尔类型和 float16，原因是 ONNX 运行时不支持 "ScatterElements reduction=amax" 操作在 float16 上的应用

    xfail(
        "scatter_reduce",
        variant_name="mean",
        reason="ONNX doesn't support reduce='mean' option",
    ),
    # 标记 "scatter_reduce" 函数为测试失败，variant_name="mean"，原因是 ONNX 不支持 "reduce='mean'" 选项

    xfail(
        "sgn",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Sign", "bool"),
    ),
    # 标记 "sgn" 函数为测试失败，限定数据类型为布尔类型，原因是 ONNX 脚本不支持 "Sign" 函数在布尔类型上的操作

    xfail(
        "sign",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Sign", "bool"),
    ),
    # 标记 "sign" 函数为测试失败，限定数据类型为布尔类型，原因是 ONNX 脚本不支持 "Sign" 函数在布尔类型上的操作
    xfail(
        "signal.windows.kaiser",
        reason=onnx_test_common.reason_dynamo_does_not_support("functionalization"),
    ),
    # 标记测试为失败，针对信号处理中的 Kaiser 窗口
    xfail(
        "softmax",
        dtypes=(torch.float16,),
        reason="ORT error: https://github.com/microsoft/onnxruntime/issues/16438"
    ),
    # 标记测试为失败，针对 softmax 操作，使用 torch.float16 数据类型
    xfail(
        "sparse.mm",
        variant_name="reduce",
        reason=onnx_test_common.reason_dynamo_does_not_support("InternalTorchDynamoError: Sparse CSR tensors do not have strides"),
    ),
    # 标记测试为失败，针对稀疏矩阵乘法操作，使用 reduce 变体，给出失败原因
    xfail(
        "sparse.sampled_addmm",
        reason=onnx_test_common.reason_dynamo_does_not_support("InternalTorchDynamoError: Sparse CSR tensors do not have strides"),
    ),
    # 标记测试为失败，针对稀疏采样加乘操作，给出失败原因
    xfail(
        "special.erfcx",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Erf", "int, bool"),
    ),
    # 标记测试为失败，针对特殊函数 erfcx，使用整数和布尔类型数据，给出失败原因
    xfail(
        "special.erfcx",
        dtypes=onnx_test_common.FLOAT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Erfcx"),
    ),
    # 标记测试为失败，针对特殊函数 erfcx，使用浮点数类型数据，给出失败原因
    xfail(
        "special.log_ndtr",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.FLOAT_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记测试为失败，针对特殊函数 log_ndtr，使用整数和浮点数类型数据，给出失败原因
    xfail(
        "special.ndtr",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记测试为失败，针对特殊函数 ndtr，使用 torch.float16 数据类型，给出失败原因
    xfail(
        "square",
        dtypes=(torch.int8, torch.uint8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Pow", "int8, uint8, int16"),
    ),
    # 标记测试为失败，针对平方操作，使用 int8, uint8, int16 数据类型，给出失败原因
    xfail(
        "squeeze",
        variant_name="multiple",
        reason="fixme: https://github.com/microsoft/onnxscript/issues/1264",
    ),
    # 标记测试为失败，针对 squeeze 操作的多变体，给出失败原因
    xfail(
        "svd_lowrank",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    # 标记测试为失败，针对 svd_lowrank 操作，给出失败原因
    xfail(
        "stft",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten._fft_r2c.default"),
    ),
    # 标记测试为失败，针对 stft 操作，给出失败原因
    xfail(
        "sub",
        dtypes=(torch.uint8, torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Mul", "uint8, int8, int16"
        ),
    ),
    # 标记测试为失败，针对减法操作，使用 uint8, int8, int16 数据类型，给出失败原因
    xfail(
        "take",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    # 标记测试为失败，针对 take 操作，给出失败原因
    xfail(
        "tensor_split",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    # 标记测试为失败，针对 tensor_split 操作，给出失败原因
    xfail(
        "topk",
        dtypes=(torch.int64, torch.int32),
        reason="fixme: Assertion error: result mismatch",
    ),
    # 标记测试为失败，针对 topk 操作，使用 torch.int64, torch.int32 数据类型，给出失败原因
    xfail(
        "tril",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.int32,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("trilu", "bool, int32"),
    ),
    # 标记测试为失败，针对 tril 操作，使用布尔类型和 int32 数据类型，给出失败原因
    xfail(
        "triu",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.int32,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("trilu", "bool, int32"),
    ),
    # 标记测试为失败，针对 triu 操作，使用布尔类型和 int32 数据类型，给出失败原因
    xfail(
        "trunc",  # 将 "trunc" 标记为预期失败，因为 ONNX 不支持 "Floor" 操作对整数的支持
        dtypes=onnx_test_common.INT_TYPES,  # 指定操作涉及的数据类型为整数类型
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "int"),  # 指定失败的原因
    ),
    xfail(
        "unflatten",  # 将 "unflatten" 标记为预期失败，因为 ONNX 不支持 "Unflatten" 操作
        dtypes=onnx_test_common.BOOL_TYPES,  # 指定操作涉及的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_does_not_support("Unflatten")  # 指定失败的原因
    ),
    xfail(
        "uniform",  # 将 "uniform" 标记为预期失败，因为 Dynamo 不支持 "wrapper_set_seed" 操作
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),  # 指定失败的原因
    ),
    xfail(
        "unique",  # 将 "unique" 标记为预期失败，因为 Dynamo 不支持 "aten.unique_consecutive.default" 操作
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.unique_consecutive.default"),  # 指定失败的原因
    ),
    xfail(
        "unique_consecutive",  # 将 "unique_consecutive" 标记为预期失败，因为 Dynamo 不支持 "aten.unique_consecutive.default" 操作
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.unique_consecutive.default"),  # 指定失败的原因
    ),
    xfail(
        "unravel_index",  # 将 "unravel_index" 标记为预期失败，因为 ONNX 不支持 "Floor" 操作对布尔和整数的支持
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,  # 指定操作涉及的数据类型为布尔和整数类型的组合
        reason=onnx_test_common.reason_onnx_script_does_not_support("Floor", "bool, int"),  # 指定失败的原因
    ),
    xfail(
        "where",  # 将 "where" 标记为预期失败，因为 ONNX 运行时不支持 "Where" 操作对布尔类型的支持
        dtypes=onnx_test_common.BOOL_TYPES,  # 指定操作涉及的数据类型为布尔类型
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),  # 指定失败的原因
    ),
    xfail(
        "zeros",  # 将 "zeros" 标记为预期失败，因为在 ONNX 中不支持使用 complex64 数据类型
        reason="fixme: kwargs dtpye=complex64 is not supported in ONNX."  # 指定失败的原因
    ),
    # SLOW TESTS (All are xfails if we run them)
    # TODO: https://github.com/pytorch/pytorch/issues/117118
    skip_slow(
        "cdist",  # 跳过 "cdist" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "histogram",  # 跳过 "histogram" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "histogramdd",  # 跳过 "histogramdd" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "linalg.lu_solve",  # 跳过 "linalg.lu_solve" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "linalg.solve_triangular",  # 跳过 "linalg.solve_triangular" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "linalg.svd",  # 跳过 "linalg.svd" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "logspace",  # 跳过 "logspace" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "logspace",  # 跳过 "logspace" 测试，因为测试集太大
        variant_name="tensor_overload",  # 指定测试的变体名称
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "max_pool2d_with_indices_backward",  # 跳过 "max_pool2d_with_indices_backward" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "nn.functional.interpolate",  # 跳过 "nn.functional.interpolate" 测试，因为测试集太大
        variant_name="bicubic",  # 指定测试的变体名称
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "nn.functional.max_unpool1d",  # 跳过 "nn.functional.max_unpool1d" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "nn.functional.max_unpool2d",  # 跳过 "nn.functional.max_unpool2d" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "nn.functional.max_unpool3d",  # 跳过 "nn.functional.max_unpool3d" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "nn.functional.max_pool1d",  # 跳过 "nn.functional.max_pool1d" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    skip_slow(
        "nn.functional.max_pool2d",  # 跳过 "nn.functional.max_pool2d" 测试，因为测试集太大
        reason="fixme: Test sets are too many.",  # 指定跳过的原因
    ),
    # 跳过针对 "nn.functional.max_pool3d" 的测试，原因是测试集太多
    skip_slow(
        "nn.functional.max_pool3d",
        reason="fixme: Test sets are too many.",
    ),
    # 跳过针对 "nn.functional.unfold" 的测试，原因是测试集太多
    skip_slow(
        "nn.functional.unfold",
        reason="fixme: Test sets are too many.",
    ),
    # 跳过针对 "ormqr" 的测试，原因是测试集太多
    skip_slow(
        "ormqr",
        reason="fixme: Test sets are too many.",
    ),
    # 跳过针对 "searchsorted" 的测试，原因是测试集太多
    skip_slow(
        "searchsorted",
        reason="fixme: Test sets are too many.",
    ),
    # 跳过针对 "svd" 的测试，原因是测试集太多
    skip_slow(
        "svd",
        reason="fixme: Test sets are too many.",
    ),
# fmt: on

# NOTE: 将在 `SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE` 部分中跳过带有匹配器函数或模型类型的 xfail 和 skip 测试。
SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE: tuple[
    onnx_test_common.DecorateMeta, ...
] = (
    # 使用 skip 跳过具有以下模型类型的测试：TORCH_EXPORT_EXPORTEDPROGRAM
    skip(
        "_native_batch_norm_legit",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="https://github.com/pytorch/pytorch/issues/115106",
    ),
    # 使用 skip 跳过具有以下模型类型的测试：TORCH_EXPORT_EXPORTEDPROGRAM
    skip(
        "_batch_norm_with_update",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="https://github.com/pytorch/pytorch/issues/115106",
    ),
    # TODO: 这个测试当前仅对特定输入失败，例如 shape([3, 1])。
    # 数值上，ONNX 程序是正确的，但 `save_mean` 和 `save_var` 的输出形状是 tensor(-2.1268)，
    # 而不是正确的 tensor([-2.1268])。
    skip(
        "_batch_norm_with_update",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
        reason="not supported yet",
    ),
    # 使用 xfail 跳过具有以下条件的测试：matcher 返回的样本输入数据类型为 uint8, int8, int16, int32, int64
    xfail(
        "addmm",
        matcher=lambda sample: sample.input.dtype
        in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Gemm", "uint8, int8, int16, int32, int64"
        ),
    ),
    # 使用 xfail 跳过具有以下条件的测试：matcher 返回的样本参数的元素数量为 0
    xfail(
        "addmm",
        matcher=lambda sample: sample.args[0].numel() == 0,
        reason="ONNX Runtime does not support empty tensors multiplication",
    ),
    # 使用 xfail 跳过具有以下条件的测试：matcher 返回的样本参数的元素数量为 0，变种名称为 decomposed
    xfail(
        "addmm",
        variant_name="decomposed",
        matcher=lambda sample: sample.args[0].numel() == 0,
        reason="ONNX Runtime does not support empty tensors multiplication",
    ),
    # 使用 xfail 跳过具有以下条件的测试：matcher 返回的样本输入形状长度为 0，并且 `dim` 参数不为空
    xfail(
        "amax",
        matcher=lambda sample: len(sample.input.shape) == 0
        and (sample.kwargs.get("dim") is not None and sample.kwargs.get("dim") != ()),
        reason="Op (ReduceMax) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    # 使用 xfail 跳过具有以下条件的测试：matcher 返回的样本输入形状长度为 0，并且 `dim` 参数不为空
    xfail(
        "amin",
        matcher=lambda sample: len(sample.input.shape) == 0
        and (sample.kwargs.get("dim") is not None and sample.kwargs.get("dim") != ()),
        reason="Op (ReduceMin) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    # 使用 xfail 跳过具有以下条件的测试：matcher 返回的样本输入形状长度为 0，并且 `dim` 参数不为空
    xfail(
        "aminmax",
        matcher=lambda sample: len(sample.input.shape) == 0
        and sample.kwargs.get("dim") is not None,
        reason="Op (ReduceMin) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    # 使用 skip 跳过具有以下条件的测试：matcher 返回的样本输入的第一个元素是空张量
    skip(
        "cat",
        matcher=lambda sample: sample.input[0].equal(torch.tensor([])),
        reason="core dump - cat does not support zero-dim tensors yet",
    ),
)
    xfail(
        "index_add",  # 标记测试用例为预期失败，测试函数为 index_add
        matcher=lambda sample: len(sample.input.shape) == 0,  # 匹配器函数，检查输入的形状是否为空
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ScatterND", "0-D tensor"  # 不支持的原因，使用了 ScatterND 和 0-D 张量
        ),
    ),
    xfail(
        "index_add",  # 标记测试用例为预期失败，测试函数为 index_add
        matcher=lambda sample: isinstance(sample.args[0], int) and sample.args[0] == -1,  # 匹配器函数，检查第一个参数是否为整数且为 -1
        reason="fixme: aten::index_put indices contains None when dim is -1",  # 不支持的原因，索引包含 None 时维度为 -1
    ),
    xfail(
        "index_copy",  # 标记测试用例为预期失败，测试函数为 index_copy
        matcher=lambda sample: len(sample.input.shape) == 0,  # 匹配器函数，检查输入的形状是否为空
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ScatterND", "0-D tensor"  # 不支持的原因，使用了 ScatterND 和 0-D 张量
        ),
    ),
    xfail(
        "index_copy",  # 标记测试用例为预期失败，测试函数为 index_copy
        matcher=lambda sample: isinstance(sample.args[0], int) and sample.args[0] == -1,  # 匹配器函数，检查第一个参数是否为整数且为 -1
        reason="fixme: aten::index_put indices contains None when dim is -1",  # 不支持的原因，索引包含 None 时维度为 -1
    ),
    xfail(
        "index_put",  # 标记测试用例为预期失败，测试函数为 index_put
        matcher=lambda sample: (sample.args[0][0].dtype == torch.bool)  # 匹配器函数，检查第一个参数的第一个元素是否为布尔类型
        and (sample.kwargs.get("accumulate") is False),  # 检查 accumulate 参数是否为 False
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "https://github.com/pytorch/pytorch/issues/101150"  # 不支持的原因，参考 GitHub 上的 issue
        ),
    ),
    skip(
        "linalg.multi_dot",  # 标记测试用例为跳过，测试函数为 linalg.multi_dot
        matcher=lambda sample: sum(torch.numel(input) for input in sample.input) == 0,  # 匹配器函数，检查输入的所有元素数量之和是否为 0
        reason="fixme: Undefined",  # 跳过的原因，未定义
    ),
    skip(
        "log_softmax",  # 标记测试用例为跳过，测试函数为 log_softmax
        matcher=lambda sample: len(sample.input.shape) == 0,  # 匹配器函数，检查输入的形状是否为空
        reason="fixme: LogSoftMax does not support empty tensor as input",  # 跳过的原因，LogSoftMax 不支持空张量作为输入
    ),
    skip(
        "log_softmax",  # 标记测试用例为跳过，测试函数为 log_softmax
        variant_name="with_dtype",  # 变体名称为 with_dtype
        matcher=lambda sample: len(sample.input.shape) == 0,  # 匹配器函数，检查输入的形状是否为空
        reason="fixme: LogSoftMax does not support empty tensor as input",  # 跳过的原因，LogSoftMax 不支持空张量作为输入
    ),
    skip(
        "masked.log_softmax",  # 标记测试用例为跳过，测试函数为 masked.log_softmax
        matcher=lambda sample: len(sample.input.shape) == 0,  # 匹配器函数，检查输入的形状是否为空
        reason="fixme: LogSoftMax does not support empty tensor as input",  # 跳过的原因，LogSoftMax 不支持空张量作为输入
    ),
    skip(
        "matmul",  # 标记测试用例为跳过，测试函数为 matmul
        matcher=lambda sample: torch.numel(sample.input) == 0,  # 匹配器函数，检查输入的元素数量是否为 0
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",  # 跳过的原因，[m, 0] 和 [0, n] 矩阵的乘积结果未定义
    ),
    skip(
        "mm",  # 标记测试用例为跳过，测试函数为 mm
        matcher=lambda sample: torch.numel(sample.input) == 0,  # 匹配器函数，检查输入的元素数量是否为 0
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",  # 跳过的原因，[m, 0] 和 [0, n] 矩阵的乘积结果未定义
    ),
    xfail(
        "native_batch_norm",  # 标记测试用例为预期失败，测试函数为 native_batch_norm
        matcher=lambda sample: sample.args[-3] is True  # 匹配器函数，检查倒数第三个参数是否为 True
        and any(arg is not None for arg in sample.args[2:4]),  # 检查第二到第四个参数是否有任何一个不为 None
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,  # 指定模型类型
        reason="https://github.com/pytorch/pytorch/issues/115106",  # 不支持的原因，参考 GitHub 上的 issue
    ),
    xfail(
        "nn.functional.avg_pool1d",  # 标记函数 nn.functional.avg_pool1d 为预期失败状态
        matcher=lambda sample: (sample.kwargs.get("ceil_mode") is True)  # 匹配条件：ceil_mode 参数为 True
        and (
            sample.kwargs.get("count_include_pad") is True  # 并且 count_include_pad 参数为 True
            or sample.input.shape[2]  # 或者输入的第三维度不是 0
            % (
                sample.args[0][0]  # 如果 sample.args[0] 是元组，则取第一个元素
                if isinstance(sample.args[0], tuple)
                else sample.args[0]  # 否则直接取 sample.args[0]
            )
            != 0
        ),
        reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19",  # 失败原因说明
    ),
    xfail(
        "nn.functional.avg_pool2d",  # 标记函数 nn.functional.avg_pool2d 为预期失败状态
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)  # 匹配条件：args 的长度大于 5 并且 args[5] 不为 None
        or (sample.kwargs.get("divisor_override") is not None),  # 或者 divisor_override 参数不为 None
        reason="ONNX doesn't support divisor_override argument",  # 失败原因说明
    ),
    xfail(
        "nn.functional.avg_pool3d",  # 标记函数 nn.functional.avg_pool3d 为预期失败状态
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True,  # 匹配条件：ceil_mode 参数为 True
        reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19",  # 失败原因说明
    ),
    xfail(
        "nn.functional.avg_pool3d",  # 标记函数 nn.functional.avg_pool3d 为预期失败状态
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)  # 匹配条件：args 的长度大于 5 并且 args[5] 不为 None
        or (sample.kwargs.get("divisor_override") is not None),  # 或者 divisor_override 参数不为 None
        reason="ONNX doesn't support divisor_override argument",  # 失败原因说明
    ),
    xfail(
        "nn.functional.batch_norm",  # 标记函数 nn.functional.batch_norm 为预期失败状态
        matcher=lambda sample: sample.kwargs.get("training") is True  # 匹配条件：training 参数为 True
        and any(arg is not None for arg in sample.args[2:4]),  # 并且 sample.args[2:4] 中的任意参数不为 None
        reason="Flaky failure: https://github.com/pytorch/pytorch/issues/115106",  # 失败原因说明
    ),
    xfail(
        "nn.functional.conv2d",  # 标记函数 nn.functional.conv2d 为预期失败状态
        matcher=lambda sample: sample.kwargs.get("padding") == "valid",  # 匹配条件：padding 参数为 "valid"
        reason="fixme: https://github.com/pytorch/pytorch/issues/117054",  # 失败原因说明
    ),
    xfail(
        "nn.functional.conv3d",  # 标记函数 nn.functional.conv3d 为预期失败状态
        matcher=lambda sample: sample.kwargs.get("padding") == "valid",  # 匹配条件：padding 参数为 "valid"
        reason="fixme: https://github.com/pytorch/pytorch/issues/117054",  # 失败原因说明
    ),
    skip(
        "nn.functional.cross_entropy",  # 跳过函数 nn.functional.cross_entropy 的测试
        matcher=lambda sample: not isinstance(sample.kwargs.get("weight"), int),  # 匹配条件：weight 参数不是 int 类型
        reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[weight] is int type",  # 跳过原因说明
    ),
    xfail(
        "nn.functional.embedding",  # 标记函数 nn.functional.embedding 为预期失败状态
        matcher=lambda sample: sample.kwargs.get("max_norm") is not None,  # 匹配条件：max_norm 参数不为 None
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,  # 指定模型类型
        reason="https://github.com/pytorch/pytorch/issues/115106",  # 失败原因说明
    ),
    skip_torchlib_forward_compatibility(
        "nn.functional.embedding_bag",  # 跳过函数 nn.functional.embedding_bag 的测试
        matcher=lambda sample: sample.kwargs.get("padding_idx") is not None or True,  # 匹配条件：padding_idx 参数不为 None 或者为 True
        reason=onnx_test_common.reason_onnx_script_does_not_support(  # 跳过原因说明
            "'padding_idx' overload for _embedding_bag and _embedding_bag_forward_only. "
            "'padding_idx=-1' is emitted for aten op when 'padding_idx' is not provided"
        ),
        github_issue="https://github.com/microsoft/onnxscript/issues/1056",  # 相关的 GitHub 问题链接
    ),
    xfail(
        "nn.functional.group_norm",  # 指定要跳过的测试函数为 nn.functional.group_norm
        matcher=lambda sample: torch.numel(sample.input) == 0,  # 使用 lambda 函数匹配输入是否为空张量
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Reshape", "empty tensor"  # 使用 onnx_test_common 中的函数生成不支持的原因
        ),
    ),
    xfail(
        "nn.functional.instance_norm",  # 指定要跳过的测试函数为 nn.functional.instance_norm
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,  # 设置模型类型为导出程序
        matcher=lambda sample: sample.kwargs.get("running_mean") is not None,  # 使用 lambda 函数匹配是否存在 "running_mean" 参数
        reason="fixme: KeyError: 'self___kwargs__running_mean'",  # 设置跳过的原因为 KeyError
    ),
    xfail(
        "nn.functional.max_pool3d",  # 指定要跳过的测试函数为 nn.functional.max_pool3d
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True  # 使用 lambda 函数匹配是否使用 ceil_mode=True
        and sample.kwargs.get("padding") == 1,  # 继续匹配是否使用 padding=1
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",  # 设置跳过的原因为问题解决后修复
    ),
    xfail(
        "nn.functional.pixel_shuffle",  # 指定要跳过的测试函数为 nn.functional.pixel_shuffle
        matcher=lambda sample: sample.input.numel() == 0,  # 使用 lambda 函数匹配输入是否为空张量
        reason="fixme: ORT does not support empty tensor as input",  # 设置跳过的原因为 ORT 不支持空张量作为输入
    ),
    xfail(
        "nonzero",  # 指定要跳过的测试函数为 nonzero
        matcher=lambda sample: len(sample.input.shape) == 0  # 使用 lambda 函数匹配输入张量的维度是否为 0
        and sample.kwargs.get("as_tuple", False) is False,  # 继续匹配 as_tuple 参数是否为 False
        reason="Output 'shape' do not match: torch.Size([0, 1]) != torch.Size([0, 0]).",  # 设置跳过的原因为输出形状不匹配
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,  # 设置模型类型为 TORCH_NN_MODULE
    ),
    xfail(
        "scatter_add",  # 指定要跳过的测试函数为 scatter_add
        matcher=lambda sample: len(sample.input.shape) == 0,  # 使用 lambda 函数匹配输入张量的维度是否为 0
        reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch",  # 设置跳过的原因为 ORT 处理不同分支导致的秩不同
    ),
    skip(
        "scatter_reduce",  # 指定要跳过的测试函数为 scatter_reduce
        variant_name="amax",  # 设置变体名称为 amax
        # ONNX 不包含 include_self 参数，默认 include_self=True 模式
        matcher=lambda sample: sample.kwargs.get("include_self") is False,  # 使用 lambda 函数匹配 include_self 参数是否为 False
        reason="ONNX does't support include_self=False option",  # 设置跳过的原因为 ONNX 不支持 include_self=False 选项
    ),
    skip(
        "scatter_reduce",  # 指定要跳过的测试函数为 scatter_reduce
        variant_name="amin",  # 设置变体名称为 amin
        # ONNX 不包含 include_self 参数，默认 include_self=True 模式
        matcher=lambda sample: sample.kwargs.get("include_self") is False,  # 使用 lambda 函数匹配 include_self 参数是否为 False
        reason="ONNX does't support include_self=False option",  # 设置跳过的原因为 ONNX 不支持 include_self=False 选项
    ),
    skip(
        "scatter_reduce",  # 指定要跳过的测试函数为 scatter_reduce
        variant_name="prod",  # 设置变体名称为 prod
        # ONNX 不包含 include_self 参数，默认 include_self=True 模式
        matcher=lambda sample: sample.kwargs.get("include_self") is False,  # 使用 lambda 函数匹配 include_self 参数是否为 False
        reason="ONNX does't support include_self=False option",  # 设置跳过的原因为 ONNX 不支持 include_self=False 选项
    ),
    skip(
        "scatter_reduce",  # 指定要跳过的测试函数为 scatter_reduce
        variant_name="sum",  # 设置变体名称为 sum
        # ONNX 不包含 include_self 参数，默认 include_self=True 模式
        matcher=lambda sample: sample.kwargs.get("include_self") is False,  # 使用 lambda 函数匹配 include_self 参数是否为 False
        reason="ONNX does't support include_self=False option",  # 设置跳过的原因为 ONNX 不支持 include_self=False 选项
    ),
    skip(
        "softmax",  # 指定要跳过的测试函数为 softmax
        matcher=lambda sample: len(sample.input.shape) == 0,  # 使用 lambda 函数匹配输入张量的维度是否为 0
        reason="fixme: LogSoftMax does not support empty tensor as input",  # 设置跳过的原因为 LogSoftMax 不支持空张量作为输入
    ),
    xfail(
        "unflatten",
        reason="Logic not implemented for size 0 inputs in op.Reshape",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
    ),
    # 标记测试用例 "unflatten" 为预期失败状态
    # 原因是 op.Reshape 中对于大小为 0 的输入逻辑尚未实现
    # 使用 matcher 函数检查样本，确保输入形状中的任何维度是否为 0

    skip(
        "signal.windows.hamming",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    # 跳过测试用例 "signal.windows.hamming"
    # 原因是模型类型与预期的节点名称不匹配

    skip(
        "signal.windows.general_hamming",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    # 跳过测试用例 "signal.windows.general_hamming"
    # 原因是模型类型与预期的节点名称不匹配

    skip(
        "signal.windows.blackman",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    # 跳过测试用例 "signal.windows.blackman"
    # 原因是模型类型与预期的节点名称不匹配

    skip(
        "signal.windows.general_cosine",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    # 跳过测试用例 "signal.windows.general_cosine"
    # 原因是模型类型与预期的节点名称不匹配

    skip(
        "signal.windows.hann",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    # 跳过测试用例 "signal.windows.hann"
    # 原因是模型类型与预期的节点名称不匹配

    skip(
        "signal.windows.nuttall",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    # 跳过测试用例 "signal.windows.nuttall"
    # 原因是模型类型与预期的节点名称不匹配
)

OPS_DB = copy.deepcopy(common_methods_invocations.op_db)
OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset(
    meta.op_name for meta in SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE
)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)


def _torch_size_flatten_spec(d: List[Any], spec: Any) -> List[Any]:
    # 根据给定的规范，展平大小列表 d 并返回
    return [d[i] for i in range(spec.num_children)]


torch.fx._pytree.register_pytree_flatten_spec(
    torch.Size,
    _torch_size_flatten_spec,
)


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        # 初始化方法，用于设置操作符和参数
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        # 前向传播方法，调用操作符并传递参数
        return self.operator(*args, **self.kwargs)


def _should_skip_xfail_test_sample(
    op_name: str,
    variant_test_name: str,
    sample,
    model_type: pytorch_test_common.TorchModelType,
) -> Tuple[Optional[str], Optional[str]]:
    """Check if the test sample should be skipped or xfailed.

    If the xfail/skip decorator meta is matched with its op_name and model_type,
    return the test_behavior and reason. Otherwise, return None, None. Note that
    if the matcher is None, the test is decorator_meta is meant to skip/xfail all model types.

    Args:
        op_name: The name of the op.
        sample: The test sample.
        model_type: The model type of the test.

    Returns:
        A tuple of (test_behavior, reason). test_behavior is either "skip" or "xfail".
        reason is the reason for the test_behavior.
    """

    if op_name not in OP_WITH_SKIPPED_XFAIL_SUBTESTS:
        return None, None
    for decorator_meta in SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE:
        # 线性搜索 SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE 列表，因为列表规模较小，性能可接受。
        # 注意：如果 model_type 为 None，则 decorator_meta 用于跳过/标记所有模型类型的测试。
        if (
            decorator_meta.op_name == op_name
            and decorator_meta.variant_name == variant_test_name
        ) and (
            model_type == decorator_meta.model_type or decorator_meta.model_type is None
        ):
            if decorator_meta.matcher is None and decorator_meta.model_type is None:
                raise TypeError(
                    "Either Matcher or model_type must be defined in sub xfail and skip."
                )
            if decorator_meta.matcher is not None and decorator_meta.matcher(sample):
                return decorator_meta.test_behavior, decorator_meta.reason
            elif decorator_meta.matcher is None:
                # 如果没有匹配器，则跳过/标记整个模型类型的测试
                return decorator_meta.test_behavior, decorator_meta.reason
    return None, None


def _compare_onnx_and_torch_exported_program(
    torch_exported_program,
    onnx_exported_program,
    input_args,
    input_kwargs=None,
    test_name=None,
    sample_num=None,
):
    # 比较导出的 Torch 和 ONNX 程序
    # 默认参数设置为 None 的样本关键字参数
    sample_kwargs=None,
    # 相对容差设为 0.001，用于数值比较
    rtol=1e-03,
    # 绝对容差设为 0.0000001，用于数值比较
    atol=1e-07,
    # 是否仅检查形状，默认为 False，表示同时检查数值
    only_check_shape=False,
    # 避免可变默认参数
    if input_kwargs is None:
        input_kwargs = {}

    # 注意：ONNXProgram 保持对原始 ref_model 的引用（非复制），包括其 state_dict。
    # 因此，必须先运行 ONNXProgram()，再运行 ref_model()，以防止 ref_model.forward() 更改 state_dict。
    # 否则，ref_model 可能会更改 state_dict 上的缓冲区，而这些缓冲区将被 ONNXProgram.__call__() 使用。
    onnx_outputs = onnx_exported_program(*input_args, **input_kwargs)

    # 如果 torch_exported_program 是 torch.export.ExportedProgram 类型的实例
    if isinstance(torch_exported_program, torch.export.ExportedProgram):
        # 使用 torch_exported_program.module() 调用模型
        torch_outputs = torch_exported_program.module()(*input_args, **input_kwargs)
    else:
        # 直接调用 torch_exported_program
        torch_outputs = torch_exported_program(*input_args, **input_kwargs)

    # 将 torch 输出数据格式适配到 ONNX 格式
    torch_outputs_onnx_format = onnx_exported_program.adapt_torch_outputs_to_onnx(
        torch_outputs
    )

    # 检查输出的数量是否一致
    if len(torch_outputs_onnx_format) != len(onnx_outputs):
        raise AssertionError(
            f"Expected {len(torch_outputs_onnx_format)} outputs, got {len(onnx_outputs)}"
        )

    # 遍历每个输出并进行比较
    for j, (torch_output, onnx_output) in enumerate(
        zip(torch_outputs_onnx_format, onnx_outputs)
    ):
        # 如果只检查形状
        if only_check_shape:
            assert torch_output.shape == onnx_output.shape
        else:
            try:
                # 使用 torch.testing.assert_close 检查输出数据的接近程度
                torch.testing.assert_close(
                    torch.tensor(onnx_output),
                    torch_output,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            except AssertionError as e:
                # 如果需要创建问题报告
                if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                    # 创建不匹配报告
                    error_reproduction.create_mismatch_report(
                        test_name,
                        sample_num,
                        onnx_exported_program.model_proto,
                        input_args,
                        sample_kwargs,
                        torch.tensor(onnx_output),
                        torch_output,
                        e,
                    )
                # 如果输出数量大于 1，则抛出详细的不匹配信息
                if len(torch_outputs_onnx_format) > 1:
                    raise AssertionError(f"Output {j} mismatch") from e
                # 否则，抛出原始异常
                raise

def _run_test_output_match(
    test_suite: onnx_test_common._TestONNXRuntime,
    device: str,
    dtype: torch.dtype,
    op: opinfo_core.OpInfo,
):
    # 确保只在 CPU 上运行测试
    assert device == "cpu"

    # 生成操作的输入样本
    samples = op.sample_inputs(
        device,
        dtype,
        requires_grad=False,
    )

def _parameterized_class_attrs_and_values():
    # 初始化空的输入值列表
    input_values = []
    # 扩展输入值列表，包含 opset 和 TorchModelType 的所有组合
    input_values.extend(
        itertools.product(
            (opset for opset in onnx_test_common.FX_TESTED_OPSETS),
            (
                pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
                pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
            ),
        )
    )
    # 返回一个字典对象，包含两个键值对：
    # "attrs": ["opset_version", "model_type"]，表示一组属性列表
    # "input_values": input_values，表示输入值，可能是一个变量或参数
    return {
        "attrs": ["opset_version", "model_type"],
        "input_values": input_values,
    }
# 将类名与参数化参数组合起来
def _parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    # 初始化后缀列表
    suffixes = []
    # 遍历输入的字典，生成形如 'key_value' 的后缀列表项
    for k, v in input_dicts.items():
        suffixes.append(f"{k}_{v}")
    # 返回类名与所有后缀列表项用下划线连接的字符串
    return f"{cls.__name__}_{'_'.join(suffixes)}"


# 使用参数化测试类装饰器，设置参数化类名生成函数为 _parameterize_class_name
@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(),
    class_name_func=_parameterize_class_name,
)
# 继承自 _TestONNXRuntime 类，用于测试导出的 ONNX 模型与 PyTorch eager 模式的输出一致性
class TestOnnxModelOutputConsistency(onnx_test_common._TestONNXRuntime):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    # 设置操作集版本为 -1
    opset_version = -1
    # 设置操作级别调试为 False
    op_level_debug: bool = False
    # 设置动态形状为 False
    dynamic_shapes: bool = False
    # 设置模型类型为 TorchModelType.TORCH_NN_MODULE
    model_type: pytorch_test_common.TorchModelType = (
        pytorch_test_common.TorchModelType.TORCH_NN_MODULE
    )

    # NOTE: Follow torchlib settings in ops_test_data.py
    # 仅进行形状检查的方法列表
    only_shape_check_list = [
        "empty",
        "empty_like",
        "empty_strided",
        "new_empty",
        "new_empty_strided",
    ]

    # 低精度浮点数字典，针对特定的操作名及其所允许的误差范围
    fp32_low_precision_dict = {
        "native_layer_norm": [2e-4, 7e-4],
    }
    # 定义一个字典，包含不同函数的低精度变体及其对应的阈值列表
    fp16_low_precision_dict = {
        "addbmm": [2e-1, 2e-2],  # addbmm函数的两个阈值：0.2和0.02
        "addcdiv": [3e-2, 1e-3],  # addcdiv函数的两个阈值：0.03和0.001
        "addcmul": [3e-2, 1e-3],  # addcmul函数的两个阈值：0.03和0.001
        "addmv": [5e-2, 3e-2],  # addmv函数的两个阈值：0.05和0.03
        "addr": [3e-3, 4e-3],  # addr函数的两个阈值：0.003和0.004
        "baddbmm": [3e-2, 1e-3],  # baddbmm函数的两个阈值：0.03和0.001
        "cumulative_trapezoid": [3e-2, 1e-3],  # cumulative_trapezoid函数的两个阈值：0.03和0.001
        "cross": [3e-2, 2e-2],  # cross函数的两个阈值：0.03和0.02
        "diff": [1e-2, 5e-2],  # diff函数的两个阈值：0.01和0.05
        "gradient": [3e-3, 4e-3],  # gradient函数的两个阈值：0.003和0.004
        "linalg.cross": [1e-3, 2e-2],  # linalg.cross函数的两个阈值：0.001和0.02
        "linalg.multi_dot": [3e-2, 1e-3],  # linalg.multi_dot函数的两个阈值：0.03和0.001
        "linalg.vecdot": [1e-2, 2e-2],  # linalg.vecdot函数的两个阈值：0.01和0.02
        "linspace": [2e-2, 2e-3],  # linspace函数的两个阈值：0.02和0.002
        "masked.std": [2e-2, 2e-3],  # masked.std函数的两个阈值：0.02和0.002
        "masked.var": [2e-2, 2e-2],  # masked.var函数的两个阈值：0.02和0.02
        "matmul": [2e-2, 6e-2],  # matmul函数的两个阈值：0.02和0.06
        "nn.functional.batch_norm": [3e-2, 1e-3],  # nn.functional.batch_norm函数的两个阈值：0.03和0.001
        "nn.functional.binary_cross_entropy": [3e-2, 1e-3],  # nn.functional.binary_cross_entropy函数的两个阈值：0.03和0.001
        "nn.functional.binary_cross_entropy_with_logits": [3e-2, 1e-3],  # nn.functional.binary_cross_entropy_with_logits函数的两个阈值：0.03和0.001
        "nn.functional.cosine_similarity": [3e-2, 1e-3],  # nn.functional.cosine_similarity函数的两个阈值：0.03和0.001
        "nn.functional.cosine_embedding_loss": [1e-2, 1e-3],  # nn.functional.cosine_embedding_loss函数的两个阈值：0.01和0.001
        "nn.functional.hardsigmoid": [1e-3, 5e-3],  # nn.functional.hardsigmoid函数的两个阈值：0.001和0.005
        "nn.functional.hardswish": [1e-3, 5e-3],  # nn.functional.hardswish函数的两个阈值：0.001和0.005
        "nn.functional.hinge_embedding_loss": [4e-1, 3e-3],  # nn.functional.hinge_embedding_loss函数的两个阈值：0.4和0.003
        "nn.functional.huber_loss": [1e-2, 1e-1],  # nn.functional.huber_loss函数的两个阈值：0.01和0.1
        "nn.functional.instance_norm": [1e-2, 1e-3],  # nn.functional.instance_norm函数的两个阈值：0.01和0.001
        "nn.functional.interpolate": [1e-2, 1e-3],  # nn.functional.interpolate函数的两个阈值：0.01和0.001
        "nn.functional.kl_div": [2e-3, 2e-4],  # nn.functional.kl_div函数的两个阈值：0.002和0.0002
        "nn.functional.multilabel_soft_margin_loss": [4e-2, 5e-3],  # nn.functional.multilabel_soft_margin_loss函数的两个阈值：0.04和0.005
        "nn.functional.local_response_norm": [1e-2, 5e-3],  # nn.functional.local_response_norm函数的两个阈值：0.01和0.005
        "nn.functional.poisson_nll_loss": [3e-2, 1e-3],  # nn.functional.poisson_nll_loss函数的两个阈值：0.03和0.001
        "nn.functional.nll_loss": [3e-2, 1e-3],  # nn.functional.nll_loss函数的两个阈值：0.03和0.001
        "nn.functional.triplet_margin_loss": [2e-2, 1e-2],  # nn.functional.triplet_margin_loss函数的两个阈值：0.02和0.01
        "nn.functional.triplet_margin_with_distance_loss": [3e-2, 1e-2],  # nn.functional.triplet_margin_with_distance_loss函数的两个阈值：0.03和0.01
        "native_batch_norm": [3e-2, 1e-3],  # native_batch_norm函数的两个阈值：0.03和0.001
        "norm": [1e-2, 1e-2],  # norm函数的两个阈值：0.01和0.01
        "dot": [3e-2, 1e-3],  # dot函数的两个阈值：0.03和0.001
        "logit": [3e-2, 1e-3],  # logit函数的两个阈值：0.03和0.001
        "rsub": [3e-2, 1e-3],  # rsub函数的两个阈值：0.03和0.001
        "sinc": [2e-1, 6e-4],  # sinc函数的两个阈值：0.2和0.0006
        "sub": [3e-2, 1e-3],  # sub函数的两个阈值：0.03和0.001
        "trapezoid": [1e-3, 7e-3],  # trapezoid函数的两个阈值：0.001和0.007
        "trapz": [1e-3, 7e-3],  # trapz函数的两个阈值：0.001和0.007
        "vdot": [1e-3, 1e-2],  # vdot函数的两个阈值：0.001和0.01
    }
    
    # 定义一个字典，包含特定函数及其参数组合的低精度变体及其对应的阈值列表
    fp16_low_precision_variant_dict = {
        ("nn.functional.interpolate", "trilinear"): [3e-2, 3e-3],  # nn.functional.interpolate函数的trilinear参数的两个阈值：0.03和0.003
        ("nn.functional.interpolate", "linear"): [3e-2, 3e-3],  # nn.functional.interpolate函数的linear参数的两个阈值：0.03和0.003
    }
    
    # 使用装饰器 @common_device_type.ops 对测试函数进行装饰，指定操作、数据类型和允许的操作集
    @common_device_type.ops(
        [op for op in OPS_DB if op.name
# 遍历 ONNX 测试中的所有已测试操作集
for opset in onnx_test_common.FX_TESTED_OPSETS:
    # 遍历 PyTorch 模型类型枚举
    for model_type in pytorch_test_common.TorchModelType:
        # 根据操作集和模型类型创建测试类名
        test_class_name = f"TestOnnxModelOutputConsistency_opset_version_{opset}_model_type_TorchModelType.{model_type.name}"
        
        # 向 OPS_DB 中添加装饰信息，用于测试输出的一致性
        onnx_test_common.add_decorate_info(
            OPS_DB,
            test_class_name,
            "test_output_match",
            opset=opset,
            skip_or_xfails=EXPECTED_SKIPS_OR_FAILS_WITH_DTYPES,
        )

        # 实例化特定设备类型的测试，仅限于 CPU
        common_device_type.instantiate_device_type_tests(
            globals()[test_class_name], globals(), only_for="cpu"
        )

# 如果该脚本作为主程序运行，则执行通用测试工具中的运行测试函数
if __name__ == "__main__":
    common_utils.run_tests()
```