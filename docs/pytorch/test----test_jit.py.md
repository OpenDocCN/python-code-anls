# `.\pytorch\test\test_jit.py`

```py
# Owner(s): ["oncall: jit"]

# 导入PyTorch库
import torch

# 这里包含了位于test/jit/...中的测试文件，
# 它们被包含在这里以便在调用`test_jit.py`时被调用，
# 请不要直接运行这些测试文件。

# 以下是测试模块的导入，每个模块对应不同的测试内容，通过`# noqa: F401`标记确保不出现未使用警告

from jit.test_tracer import TestTracer, TestMixTracingScripting  # noqa: F401
from jit.test_recursive_script import TestRecursiveScript  # noqa: F401
from jit.test_type_sharing import TestTypeSharing  # noqa: F401
from jit.test_logging import TestLogging  # noqa: F401
from jit.test_backends import TestBackends, TestBackendsWithCompiler  # noqa: F401
from jit.test_backend_nnapi import TestNnapiBackend  # noqa: F401
from jit.test_list_dict import TestList, TestDict, TestNamedTuple, TestScriptDict, TestScriptList  # noqa: F401
from jit.test_async import TestAsync  # noqa: F401
from jit.test_await import TestAwait  # noqa: F401
from jit.test_data_parallel import TestDataParallel  # noqa: F401
from jit.test_models import TestModels  # noqa: F401
from jit.test_modules import TestModules  # noqa: F401
from jit.test_autodiff import TestAutodiffJit  # noqa: F401
from jit.test_autodiff_subgraph_slicing import TestAutodiffSubgraphSlicing  # noqa: F401
from jit.test_custom_operators import TestCustomOperators  # noqa: F401
from jit.test_graph_rewrite_passes import TestGraphRewritePasses  # noqa: F401
from jit.test_class_type import TestClassType  # noqa: F401
from jit.test_builtins import TestBuiltins, TestTensorBuiltins  # noqa: F401
from jit.test_ignore_context_manager import TestIgnoreContextManager  # noqa: F401
from jit.test_symbolic_shape_analysis import TestSymbolicShapeAnalysis  # noqa: F401
from jit.test_op_decompositions import TestOpDecompositions  # noqa: F401
from jit.test_unsupported_ops import TestUnsupportedOps  # noqa: F401
from jit.test_freezing import TestFreezing, TestFrozenOptimizations, TestMKLDNNReinplacing  # noqa: F401
from jit.test_peephole import TestPeephole  # noqa: F401
from jit.test_alias_analysis import TestAliasAnalysis  # noqa: F401
from jit.test_save_load import TestSaveLoad, TestSaveLoadFlatbuffer  # noqa: F401
from jit.test_save_load_for_op_version import TestSaveLoadForOpVersion  # noqa: F401
from jit.test_module_containers import TestModuleContainers  # noqa: F401
from jit.test_python_bindings import TestPythonBindings  # noqa: F401
from jit.test_python_ir import TestPythonIr  # noqa: F401
from jit.test_functional_blocks import TestFunctionalBlocks  # noqa: F401
from jit.test_remove_mutation import TestRemoveMutation  # noqa: F401
from jit.test_torchbind import TestTorchbind  # noqa: F401
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from jit.test_with import TestWith  # noqa: F401
from jit.test_enum import TestEnum  # noqa: F401
from jit.test_string_formatting import TestStringFormatting  # noqa: F401
from jit.test_profiler import TestProfiler  # noqa: F401
from jit.test_slice import TestSlice  # noqa: F401
# 从 jit.test_ignorable_args 模块导入 TestIgnorableArgs 类，用于测试可忽略参数的情况
from jit.test_ignorable_args import TestIgnorableArgs  # noqa: F401
# 从 jit.test_hooks 模块导入 TestHooks 类，用于测试钩子功能
from jit.test_hooks import TestHooks  # noqa: F401
# 从 jit.test_warn 模块导入 TestWarn 类，用于测试警告功能
from jit.test_warn import TestWarn  # noqa: F401
# 从 jit.test_isinstance 模块导入 TestIsinstance 类，用于测试 isinstance 函数
from jit.test_isinstance import TestIsinstance  # noqa: F401
# 从 jit.test_cuda 模块导入 TestCUDA 类，用于测试 CUDA 相关功能
from jit.test_cuda import TestCUDA  # noqa: F401
# 从 jit.test_python_builtins 模块导入 TestPythonBuiltinOP 类，用于测试 Python 内建操作
from jit.test_python_builtins import TestPythonBuiltinOP  # noqa: F401
# 从 jit.test_typing 模块导入 TestTyping 类，用于测试类型注解功能
from jit.test_typing import TestTyping  # noqa: F401
# 从 jit.test_hash 模块导入 TestHash 类，用于测试哈希功能
from jit.test_hash import TestHash  # noqa: F401
# 从 jit.test_complex 模块导入 TestComplex 类，用于测试复杂功能
from jit.test_complex import TestComplex  # noqa: F401
# 从 jit.test_jit_utils 模块导入 TestJitUtils 类，用于测试 JIT 工具函数
from jit.test_jit_utils import TestJitUtils  # noqa: F401
# 从 jit.test_scriptmod_ann 模块导入 TestScriptModuleInstanceAttributeTypeAnnotation 类，用于测试脚本模块实例属性类型注解
from jit.test_scriptmod_ann import TestScriptModuleInstanceAttributeTypeAnnotation  # noqa: F401
# 从 jit.test_types 模块导入 TestTypesAndAnnotation 类，用于测试类型和注解
from jit.test_types import TestTypesAndAnnotation  # noqa: F401
# 从 jit.test_misc 模块导入 TestMisc 类，用于测试杂项功能
from jit.test_misc import TestMisc  # noqa: F401
# 从 jit.test_upgraders 模块导入 TestUpgraders 类，用于测试升级功能
from jit.test_upgraders import TestUpgraders  # noqa: F401
# 从 jit.test_pdt 模块导入 TestPDT 类，用于测试 PDT（Python 数据类型）相关功能
from jit.test_pdt import TestPDT  # noqa: F401
# 从 jit.test_tensor_creation_ops 模块导入 TestTensorCreationOps 类，用于测试张量创建操作
from jit.test_tensor_creation_ops import TestTensorCreationOps  # noqa: F401
# 从 jit.test_module_apis 模块导入 TestModuleAPIs 类，用于测试模块 API
from jit.test_module_apis import TestModuleAPIs  # noqa: F401
# 从 jit.test_script_profile 模块导入 TestScriptProfile 类，用于测试脚本性能分析功能
from jit.test_script_profile import TestScriptProfile  # noqa: F401
# 从 jit.test_convert_activation 模块导入 TestFunctionalToInplaceActivation, TestInplaceToFunctionalActivation 类，用于测试激活函数转换
from jit.test_convert_activation import TestFunctionalToInplaceActivation, TestInplaceToFunctionalActivation  # noqa: F401
# 从 jit.test_parametrization 模块导入 TestParametrization 类，用于测试参数化
from jit.test_parametrization import TestParametrization  # noqa: F401
# 从 jit.test_attr 模块导入 TestGetDefaultAttr 类，用于测试获取默认属性
from jit.test_attr import TestGetDefaultAttr  # noqa: F401
# 从 jit.test_aten_pow 模块导入 TestAtenPow 类，用于测试 Aten 的幂函数
from jit.test_aten_pow import TestAtenPow  # noqa: F401
# 从 jit.test_optimize_for_mobile_preserve_debug_info 模块导入 TestOptimizeForMobilePreserveDebugInfo 类，用于测试在移动设备上保留调试信息的优化
from jit.test_optimize_for_mobile_preserve_debug_info import TestOptimizeForMobilePreserveDebugInfo  # noqa: F401
# 从 jit.test_union 模块导入 TestUnion 类，用于测试 Union 类型
from jit.test_union import TestUnion  # noqa: F401
# 从 jit.test_batch_mm 模块导入 TestBatchMM 类，用于测试批量矩阵乘法
from jit.test_batch_mm import TestBatchMM  # noqa: F401
# 从 jit.test_dtype_analysis 模块导入 TestDtypeAnalysis, TestDtypeCustomRulesCPU 类，用于测试数据类型分析
from jit.test_dtype_analysis import TestDtypeAnalysis, TestDtypeCustomRulesCPU  # noqa: F401
# 从 jit.test_device_analysis 模块导入 TestDeviceAnalysis 类，用于测试设备分析
from jit.test_device_analysis import TestDeviceAnalysis  # noqa: F401
# 从 jit.test_dce 模块导入 TestDCE 类，用于测试死代码消除
from jit.test_dce import TestDCE  # noqa: F401
# 从 jit.test_sparse 模块导入 TestSparse 类，用于测试稀疏张量
from jit.test_sparse import TestSparse  # noqa: F401
# 从 jit.test_tensor_methods 模块导入 TestTensorMethods 类，用于测试张量方法
from jit.test_tensor_methods import TestTensorMethods  # noqa: F401
# 从 jit.test_dataclasses 模块导入 TestDataclasses 类，用于测试数据类
from jit.test_dataclasses import TestDataclasses  # noqa: F401
# 从 jit.test_generator 模块导入 TestGenerator 类，用于测试生成器
from jit.test_generator import TestGenerator  # noqa: F401

# 从 torch 模块导入 Tensor 类
from torch import Tensor
# 从 torch._C 模块导入 TensorType, BoolType, parse_ir, _propagate_shapes
from torch._C import TensorType, BoolType, parse_ir, _propagate_shapes
# 从 torch.autograd 模块导入 Variable 类
from torch.autograd import Variable
# 从 torch.jit.annotations 模块导入 BroadcastingList2, BroadcastingList3, Any 类
from torch.jit.annotations import BroadcastingList2, BroadcastingList3, Any  # noqa: F401
# 从 torch.nn.utils.rnn 模块导入 PackedSequence 类
from torch.nn.utils.rnn import PackedSequence
# 从 torch.testing 模块导入 FileCheck, make_tensor 函数
from torch.testing import FileCheck, make_tensor
# 从 torch.autograd.profiler 模块导入 profiler 类
import torch.autograd.profiler
# 从 torch.cuda 模块导入 cuda
import torch.cuda
# 导入 torch.jit 模块
import torch.jit
# 从 torch.jit._logging 模块导入 _logging
import torch.jit._logging
# 从 torch.jit.frontend 模块导入 frontend
import torch.jit.frontend
# 从 torch.nn 模块导入 nn
import torch.nn as nn
# 从 torch.nn.functional 模块导入 F
import torch.nn.functional as F

# 从 torch.testing._internal 模块导入 jit_utils
from torch.testing._internal import jit_utils
# 从 torch.testing._internal.common_jit 模块导入 check_against_reference
from torch.testing._internal.common_jit import check_against_reference
# 从 torch.testing._internal.common_utils 模块导入 run_tests, IS_WINDOWS, TEST_WITH_UBSAN, suppress_warnings, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, freeze_rng_state, slowTest, TemporaryFileName
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, TEST_WITH_UBSAN, \
    suppress_warnings, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, \
    freeze_rng_state, slowTest, TemporaryFileName
    # 导入测试中使用的各种辅助函数和装饰器
    enable_profiling_mode_for_profiling_tests, TEST_MKL, set_default_dtype, num_profiled_runs, \
        skipIfCrossRef, skipIfTorchDynamo
# 导入必要的模块和函数来测试 PyTorch 的 JIT 编译功能
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining, \
    _trace, do_input_map, get_execution_plan, make_global, \
    execWrapper, _inline_everything, _tmp_donotuse_dont_inline_everything, \
    RUN_CUDA
# 导入用于 JIT 元编程的辅助工具函数和类
from torch.testing._internal.jit_metaprogramming_utils import (
    get_script_args,
    create_input, unpack_variables,
    additional_module_tests, EXCLUDE_SCRIPT_MODULES,
    get_nn_module_name_from_kwargs, get_nn_mod_test_name, script_method_template)
# 导入用于测试 nn 模块的通用测试函数
from torch.testing._internal.common_nn import module_tests, new_module_tests, criterion_tests

# 导入用于在 Python 2 中测试 truediv 的模块
from torch.testing._internal.test_module.future_div import div_int_future, div_float_future
from torch.testing._internal.test_module.no_future_div import div_int_nofuture, div_float_nofuture

# 标准库导入
from collections import defaultdict, namedtuple, OrderedDict  # 导入用于创建特定类型字典、命名元组、有序字典的类和函数
from copy import deepcopy  # 导入深拷贝函数
from itertools import product  # 导入用于生成迭代器的函数
from textwrap import dedent  # 导入用于移除字符串首部空白的函数
from typing import List, Dict, NamedTuple, Optional, Tuple, Union  # 导入用于类型标注的模块
import copy  # 导入浅拷贝函数
import functools  # 导入用于高阶函数操作的函数
import inspect  # 导入用于检查对象的类型和值的函数
import io  # 导入用于处理流的核心工具
import itertools  # 导入用于创建迭代器的函数
import math  # 导入数学函数
import numpy as np  # 导入用于科学计算的核心库
import os  # 导入操作系统相关的功能
import pickle  # 导入用于序列化和反序列化 Python 对象的功能
import pickletools  # 导入用于分析和生成 pickle 数据的模块
import random  # 导入生成随机数的功能
import re  # 导入正则表达式的功能
import shutil  # 导入高级文件操作功能
import string  # 导入处理字符串的函数
import sys  # 导入系统相关的功能
import tempfile  # 导入用于创建临时文件和目录的功能
import types  # 导入操作 Python 类型的功能
import typing  # 导入类型标注相关的模块
import unittest  # 导入用于编写和运行单元测试的模块
import warnings  # 导入用于处理警告的模块
import zipfile  # 导入处理 ZIP 文件的模块
import tracemalloc  # 导入用于追踪 Python 内存分配的模块


# 定义函数 canonical，用于执行 Torch 的 JIT 编译规范化操作
def canonical(graph):
    return torch._C._jit_pass_canonicalize(graph).str(False)

# 定义函数 LSTMCellF，用于调用 LSTMCell 函数，并将输入参数适配为元组形式
def LSTMCellF(input, hx, cx, *params):
    return LSTMCell(input, (hx, cx), *params)

# 定义函数 doAutodiffCheck，用于检查是否需要禁用自动微分的子图内联
def doAutodiffCheck(testname):
    # 如果测试名称包含 "test_t_" 或者为 "test_t"，则返回 False，不执行自动微分检查
    if "test_t_" in testname or testname == "test_t":
        return False

    # 如果使用的执行模式为 ProfilingMode.SIMPLE，则返回 False
    if GRAPH_EXECUTOR == ProfilingMode.SIMPLE:
        return False

    # 如果使用的执行模式为 ProfilingMode.LEGACY，则返回 True
    if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
        return True

    # 如果测试名称在例外列表中，则返回 False，因为 BailOut 节点会干扰 Differentiable Graphs 的子图切片
    test_exceptions = [
        # functional
        'test_nn_dropout',
        'test_nn_log_softmax',
        'test_nn_relu',
        'test_nn_softmax',
        'test_nn_threshold',
        'test_nn_lp_pool2d',
        'test_nn_lp_pool1d',
        'test_nn_gumbel_softmax_hard',
        'test_nn_gumbel_softmax',
        'test_nn_multilabel_soft_margin_loss',
        'test_nn_batch_norm',
        'test_nn_max_pool2d_with_indices',
        # AutogradJitGenerated
        'test___rdiv___constant',
        'test___rdiv___scalar_constant',
        'test_split',
        'test_split_dim',
        'test_split_dim_neg0',
        'test_split_size_list',
        'test_split_size_list_dim',
        'test_split_size_list_dim_neg0',
        'test_split_with_sizes',
        'test_split_with_sizes_dim',
        'test_split_with_sizes_dim_neg0',
        'test_split_with_sizes_size_0',
        'test_nn_max_pool2d_with_indices',
    ]
    # 检查变量 testname 是否在 test_exceptions 列表中
    if testname in test_exceptions:
        # 如果在异常列表中，则返回 False
        return False
    # 如果不在异常列表中，则返回 True
    return True
# TODO: enable TE in PE when all tests are fixed
# 设置 Torch 的图执行器（GRAPH_EXECUTOR）的张量表达式融合器是否启用的配置，条件是图执行模式为 PROFILING 时启用
torch._C._jit_set_texpr_fuser_enabled(GRAPH_EXECUTOR == ProfilingMode.PROFILING)
# 设置 Torch 的图执行器（GRAPH_EXECUTOR）的性能分析执行器是否启用的配置，条件是图执行模式不为 LEGACY 时启用
torch._C._jit_set_profiling_executor(GRAPH_EXECUTOR != ProfilingMode.LEGACY)

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    # LSTM 单元的输入包括当前输入、隐藏状态和权重与偏置参数的线性组合
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    # 将线性组合后的结果分割成四部分，分别是输入门、遗忘门、细胞状态门和输出门
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    # 分别对四个门的值进行激活函数处理
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    # 根据门控机制计算新的细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


def LSTMCellC(*args, **kwargs):
    # 对 LSTMCellF 的结果进行拼接
    hy, cy = LSTMCellF(*args, **kwargs)
    return torch.cat((hy, cy))


def LSTMCellS(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    # LSTM 单元的输入包括当前输入、前一时刻的隐藏状态和权重与偏置参数的线性组合
    gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
    # 将线性组合后的结果分割成四部分，分别是输入门、遗忘门、细胞状态门和输出门
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    # 分别对四个门的值进行激活函数处理
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    # 根据门控机制计算新的细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


# Code reference: https://github.com/pytorch/translate/blob/master/pytorch_translate/rnn_cell.py#L27:44
def MiLSTMCell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    # 使用论文中描述的权重和偏置参数计算 MiLSTM 的门控信息
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())
    # 根据门控信息公式计算 MiLSTM 的门控信号
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias
    # 将门控信号分割成四部分，分别是输入门、遗忘门、细胞状态门和输出门
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    # 分别对四个门的值进行激活函数处理
    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()
    # 根据门控机制计算新的细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()
    return hy, cy


def get_lstm_inputs(device, training=False, seq_length=None):
    # 返回 LSTM 单元的输入，包括输入数据、隐藏状态、细胞状态和相关参数
    input_shape = (3, 10) if seq_length is None else (seq_length, 3, 10)
    input = torch.randn(*input_shape, dtype=torch.float, device=device, requires_grad=training)
    hx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    cx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    module = nn.LSTMCell(10, 20).to(device, torch.float)  # 为了分配正确尺寸的权重，仅用于分配
    if training:
        params = tuple(module.parameters())
    else:
        params = tuple(p.requires_grad_(False) for p in module.parameters())
    return (input, hx, cx) + params


def get_milstm_inputs(device, training=False):
    # 返回 MiLSTM 单元的输入，包括输入数据、隐藏状态和细胞状态
    minibatch = 3
    input_size = 10
    hidden_size = 20
    x = torch.randn(minibatch, input_size, device=device, dtype=torch.float)
    hx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)
    cx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)
    # 初始化输入到隐藏状态的权重矩阵，形状为 (4 * hidden_size, input_size)，使用随机生成的张量
    ih = torch.randn(4 * hidden_size, input_size, device=device, dtype=torch.float, requires_grad=training)
    
    # 初始化隐藏到隐藏状态的权重矩阵，形状为 (4 * hidden_size, hidden_size)，使用随机生成的张量
    hh = torch.randn(4 * hidden_size, hidden_size, device=device, dtype=torch.float, requires_grad=training)
    
    # 初始化门控神经元的偏置参数 alpha，形状为 (4 * hidden_size)，使用随机生成的张量
    alpha = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    
    # 初始化输入门的偏置参数 ibeta，形状为 (4 * hidden_size)，使用随机生成的张量
    ibeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    
    # 初始化隐藏状态门的偏置参数 hbeta，形状为 (4 * hidden_size)，使用随机生成的张量
    hbeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    
    # 初始化所有门的偏置参数 bias，形状为 (4 * hidden_size)，使用随机生成的张量
    bias = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    
    # 返回所有初始化的张量作为结果，包括输入 x、隐藏状态 hx、细胞状态 cx，以及前述的权重和偏置参数
    return x, hx, cx, ih, hh, alpha, ibeta, hbeta, bias
# 根据给定的文件名和脚本路径导入模块
def get_fn(file_name, script_path):
    import importlib.util
    # 创建一个模块规范
    spec = importlib.util.spec_from_file_location(file_name, script_path)
    # 根据模块规范创建一个模块
    module = importlib.util.module_from_spec(spec)
    # 使用加载器执行模块代码，将模块对象填充
    spec.loader.exec_module(module)
    # 从模块中获取名为 'fn' 的函数并返回
    fn = module.fn
    return fn

def get_grad_executor(plan_state, diff_graph_idx=None, skip_check=False):
    # 如果未提供差分图索引，则获取所有节点列表
    if diff_graph_idx is None:
        nodes = list(plan_state.graph.nodes())
        # 如果不跳过检查
        if not skip_check:
            # 过滤掉特定类型的节点，确保图是可微分的
            nodes = list(filter(lambda n : n.kind() != "prim::BailOut" and n.kind() != "prim::BailoutTemplate", nodes))
            # 检查节点数量和类型，如果符合特定条件则跳过
            if len(nodes) == 1 or (len(nodes) == 2 and nodes[1].kind() == "prim::TupleConstruct"):
                pass
            elif len(nodes) == 2 and nodes[0].kind() == "prim::RequiresGradCheck" and nodes[1].kind() == "prim::If":
                pass
            else:
                raise RuntimeError("Can't get a grad_executor for a non-differentiable graph")
    # 获取计算图的梯度执行器状态列表
    grad_executors = list(plan_state.code.grad_executor_states())
    return grad_executors[diff_graph_idx or 0]

def all_backward_graphs(script_module, diff_graph_idx=None):
    # 获取调试状态
    ge_state = script_module.get_debug_state()
    # 获取前向执行计划
    fwd_plan = get_execution_plan(ge_state)
    # 获取梯度执行器状态
    grad_executor_state = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx)
    # 获取所有反向执行计划列表
    bwd_plans = list(grad_executor_state.execution_plans.values())
    return [p.graph.copy() for p in bwd_plans]

def backward_graph(script_module, diff_graph_idx=None, skip_check=False):
    # 获取调试状态
    ge_state = script_module.get_debug_state()
    # 获取前向执行计划
    fwd_plan = get_execution_plan(ge_state)
    # 获取梯度执行器状态，并可选择跳过检查
    grad_executor_state = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx, skip_check=skip_check)
    # 获取反向执行计划
    bwd_plan = get_execution_plan(grad_executor_state)
    # 运行 JIT 传递需要拥有图（使用 shared_ptr 拥有），因此需要复制图以确保拥有权
    # 调试状态结构体并不拥有其图，因此需要复制它
    return bwd_plan.graph.copy()

# 用于获取 List[Tensor] 的总和的辅助函数
def _sum_of_list(tensorlist):
    s = 0
    for t in tensorlist:
        s += t.sum()
    return s

# 必须放在顶层，否则会导致 Pickle 报错
class FooToPickle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bar = torch.jit.ScriptModule()

class TestJitProfiler(JitTestCase):
    """
    运行需要设置一些全局状态的测试，如 torch._C._set_graph_executor_optimize，
    并在之后恢复这些值，例如 test_profiler。这是为了解决 https://github.com/pytorch/pytorch/issues/91483 中
    test_profiler 的不稳定问题，在未能恢复 torch._C._set_graph_executor_optimize 到原始值之前失败。
    这会导致所有后续测试运行时出现问题。

    在此处使用一个单独的测试类，因此不需要为 TestJit 中的所有测试运行设置和拆卸。
    """
    # 设置测试的初始化方法，调用父类的 setUp 方法
    def setUp(self):
        super().setUp()
        # 获取当前图执行优化的状态并保存
        self.graph_executor_optimize_opt = torch._C._get_graph_executor_optimize()

    # 设置测试的清理方法，调用父类的 tearDown 方法
    def tearDown(self):
        super().tearDown()
        # 恢复图执行优化的状态到之前保存的状态
        torch._C._set_graph_executor_optimize(
            self.graph_executor_optimize_opt
        )

    # 测试性能分析器功能
    def test_profiler(self):
        # 关闭图执行优化
        torch._C._set_graph_executor_optimize(False)

        # 定义一个简单的函数 other_fn，用于测试
        def other_fn(x):
            return x * 2

        # 创建一个张量 x
        x = torch.rand(3, 4)

        # 对 other_fn 进行追踪
        traced_other_fn = torch.jit.trace(other_fn, x)

        # 定义一个复杂一些的函数 fn，内部调用 traced_other_fn 进行测试
        def fn(x):
            y = traced_other_fn(x)
            # 使用 jit._fork 创建一个异步任务 fut
            fut = torch.jit._fork(traced_other_fn, x)
            # 等待 fut 的完成
            y = torch.jit._wait(fut)
            return y

        # 对 fn 进行追踪
        traced_fn = torch.jit.trace(fn, x)

        # 使用性能分析器进行性能分析
        with torch.autograd.profiler.profile() as prof:
            traced_fn(x)

        # 期望看到 other_fn 的 TS 函数调用，
        # 其中 CPU 时间 >= mul CPU 时间，并且存在一个 forked 的 other_fn

        # 创建两个 defaultdict 用于记录事件
        mul_events = defaultdict(int)
        other_fn_events = defaultdict(int)

        # 遍历性能分析器中的函数事件
        for e in prof.function_events:
            if e.name == "aten::mul":
                # 确保当前线程不在 mul_events 中
                self.assertTrue(e.thread not in mul_events)
                # 记录当前线程的时间范围
                mul_events[e.thread] = e.time_range.elapsed_us()
            elif e.name == "other_fn":
                # 确保当前线程不在 other_fn_events 中
                self.assertTrue(e.thread not in other_fn_events)
                # 记录当前线程的时间范围
                other_fn_events[e.thread] = e.time_range.elapsed_us()

        # 确保 mul_events 和 other_fn_events 中有两个线程记录
        self.assertTrue(len(mul_events) == 2)
        self.assertTrue(len(other_fn_events) == 2)

        # 遍历 mul_events 中的线程及其时间，并确保在 other_fn_events 中也有对应线程，并且时间大于等于 mul_time
        for thread, mul_time in mul_events.items():
            self.assertTrue(thread in other_fn_events)
            self.assertTrue(other_fn_events[thread] >= mul_time)
class TestJit(JitTestCase):
    # 继承自 JitTestCase 的测试类 TestJit

    @unittest.skip("Requires a lot of RAM")
    # 跳过测试，因为需要大量的内存

    def test_big(self):
        # 定义测试方法 test_big

        m = torch.jit.ScriptModule()
        # 创建一个空的 Torch 脚本模块

        gig = int(1024 * 1024 * 1024 / 4)
        # 计算 1GB 的大小（单位是 Tensor 元素数量）

        # a small tensor in the first 4GB
        m.v0 = nn.Parameter(torch.full((2,), 1, dtype=torch.float))
        # 将一个小型张量放在前 4GB 的空间内

        # a large tensor in the first 4GB that ends outside of it
        m.v1 = nn.Parameter(torch.full((5, gig), 2, dtype=torch.float))
        # 将一个大型张量放在前 4GB 的空间内，并跨越其边界

        # a small tensor in >4GB space
        m.v2 = nn.Parameter(torch.full((2,), 3, dtype=torch.float))
        # 将一个小型张量放在大于 4GB 的空间内

        # a large tensor in the > 4GB space
        m.v3 = nn.Parameter(torch.full((5, gig), 4, dtype=torch.float))
        # 将一个大型张量放在大于 4GB 的空间内

        m2 = self.getExportImportCopy(m)
        # 调用自定义方法获取脚本模块 m 的导出/导入副本

        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        # 断言两个模块的参数相同

    def test_inferred_as_tensor(self):
        # 定义测试方法 test_inferred_as_tensor

        with self.assertRaisesRegex(RuntimeError, "Inferred the value for argument 'dim' to be of type 'Tensor' "
                                                  "because it was not annotated with an explicit type"):
            # 使用断言检测是否抛出预期的 RuntimeError 异常信息

            @torch.jit.script
            # Torch 脚本修饰符
            def dot(points, query, dim):
                return (points * query).sum(dim)
                # 计算点乘的和

    def test_constants_pkl(self):
        # 定义测试方法 test_constants_pkl

        # This test asserts that the serialization archive includes a `constants.pkl`
        # file. This file is used by `torch.load` to determine whether a zip file
        # is a normal eager-mode serialization zip or a jit serialization zip. If
        # you are deleting `constants.pkl`, make sure to update `torch.serialization.load`
        # so it is still able to figure out which is which.

        @torch.jit.script
        # Torch 脚本修饰符
        def fn(x):
            return x
            # 简单的返回函数

        buf = io.BytesIO()
        # 创建一个字节流对象

        torch.jit.save(fn, buf)
        # 将函数 fn 保存到字节流中

        buf.seek(0)
        # 将流的读写位置移动到开头

        files = zipfile.ZipFile(buf).filelist
        # 获取字节流中的文件列表

        self.assertTrue(any('archive/constants.pkl' == f.filename for f in files))
        # 断言文件列表中包含 'archive/constants.pkl'

    def test_script_fn_pkl(self):
        # 定义测试方法 test_script_fn_pkl

        with self.assertRaisesRegex(pickle.PickleError, "ScriptFunction cannot be pickled"):
            # 使用断言检测是否抛出预期的 PickleError 异常信息

            @torch.jit.script
            # Torch 脚本修饰符
            def fn(x: torch.Tensor) -> torch.Tensor:
                return x
                # 简单的返回函数

            pkl_fn = pickle.dumps(fn, protocol=0)
            # 使用 pickle 序列化函数 fn

    def test_restore_device(self):
        # 定义测试方法 test_restore_device

        class M(torch.jit.ScriptModule):
            # 定义 M 类，继承自 Torch 脚本模块

            def __init__(self, cpu_device_str):
                super().__init__()
                # 调用父类初始化方法

                self.p0 = nn.Parameter(torch.tensor([0.3], dtype=torch.float,
                                                    device=cpu_device_str))
                # 创建一个带有设备信息的参数张量

                self.b0 = torch.tensor([0.9], dtype=torch.float,
                                       device=cpu_device_str)
                # 创建一个带有设备信息的张量

        # main purpose is checking map_location works
        # 主要目的是检查 map_location 的工作方式

        m = M("cpu")
        # 创建一个 M 类的实例，指定在 CPU 上运行

        m2 = self.getExportImportCopy(m)
        # 调用自定义方法获取脚本模块 m 的导出/导入副本

        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        # 断言两个模块的参数相同

        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        # 断言两个模块的缓冲区相同

        self.assertFalse(m2.p0.is_cuda)
        # 断言新模块的 p0 张量不在 CUDA 上运行

        self.assertFalse(m2.b0.is_cuda)
        # 断言新模块的 b0 张量不在 CUDA 上运行

    @unittest.skipIf(not RUN_CUDA, "restore device requires CUDA")
    # 如果不支持 CUDA，则跳过测试
    def test_restore_shared_storage_on_cuda(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个整个张量，存储在 CPU 上
                whole_tensor = torch.randn(4, 5, dtype=torch.float, device='cpu')
                # 创建一个参数，从整个张量中窄化得到
                self.p0 = nn.Parameter(whole_tensor.narrow(0, 0, 1))
                # 注册一个缓冲区，从整个张量中窄化得到
                self.register_buffer('b0', whole_tensor.narrow(0, 3, 1))

        # 创建 Foo 类的实例
        m = Foo()
        # 使用 getExportImportCopy 方法获取在指定位置的副本
        m2 = self.getExportImportCopy(m, map_location=torch.device('cuda:0'))
        # 断言两个模型的参数相等
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        # 断言两个模型的缓冲区相等
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        # 断言 m2 的参数在 CUDA 上
        self.assertTrue(m2.p0.is_cuda)
        # 断言 m2 的缓冲区在 CUDA 上
        self.assertTrue(m2.b0.is_cuda)
        # 断言 m2 的参数是共享的
        self.assertTrue(m2.p0.is_shared())
        # 断言 m2 的缓冲区是共享的
        self.assertTrue(m2.b0.is_shared())
        # 断言 m2 的缓冲区的数据指针等于 m2 的参数的数据指针
        self.assertEqual(m2.b0.storage().data_ptr(), m2.p0.storage().data_ptr())
    # 定义一个测试函数 test_repeat_interleave_script，测试脚本化的函数 fn_scripted 是否能正确执行
    def test_repeat_interleave_script(self):
        # 定义一个函数 fn，接受两个参数 input 和 repeats，都是 torch.Tensor 类型，返回一个 torch.Tensor 类型的输出
        def fn(input: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
            # 调用 input 的 repeat_interleave 方法，根据 repeats 的值重复元素，生成一个新的 Tensor
            output = input.repeat_interleave(repeats)
            return output
        # 使用 torch.jit.script 将函数 fn 脚本化，以便进行后续的脚本化执行
        fn_scripted = torch.jit.script(fn)

        # 创建输入 Tensor input，包含元素 5 和 7，数据类型为 torch.int64
        input = torch.tensor([5, 7], dtype=torch.int64)
        # 创建输入 Tensor repeats，包含元素 3 和 6，数据类型为 torch.int64
        repeats = torch.tensor([3, 6], dtype=torch.int64)

        # 调用原始的函数 fn，计算输出结果
        output = fn(input, repeats)
        # 调用脚本化后的函数 fn_scripted，计算输出结果
        output_scripted = fn_scripted(input, repeats)
        # 使用 unittest 的断言方法，检查脚本化后的输出与原始输出是否相等
        self.assertEqual(output_scripted, output)

    # 如果 GRAPH_EXECUTOR 不等于 ProfilingMode.LEGACY，则跳过测试，因为简单执行器缺少形状信息
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "Simple executor doesn't have shape information")
    def test_peephole_optimize_shape_ops(self):
        # 定义测试函数test_peephole_optimize_shape_ops
        def test_input(func, input, result):
            # 定义测试输入函数test_input，验证函数func对输入input的计算结果是否为result
            # 若结果为2，将触发一次退出并且未分析的图应返回正确结果
            self.assertEqual(func(input, profile_and_replay=True), result)
            # 获取func关联的图形表示
            gre = func.graph_for(input)
            # 使用FileCheck验证图形表示中不包含"prim::If"
            FileCheck().check_not("prim::If").run(gre)

        # 定义测试维度的函数test_dim
        def test_dim():
            @torch.jit.script
            def func(x):
                # 如果输入张量x的维度为1，返回1；否则返回2
                if x.dim() == 1:
                    return 1
                else:
                    return 2

            # 对func进行测试，输入torch.tensor([0.5])，预期输出1
            test_input(func, torch.tensor([0.5]), 1)
            # 对func进行测试，输入torch.tensor([[0.5]])，预期输出2
            test_input(func, torch.tensor([[0.5]]), 2)
        test_dim()

        # 定义测试大小索引的函数test_size_index
        def test_size_index():
            @torch.jit.script
            def func(x):
                # 如果输入张量x的第一个维度大小为1，返回1；否则返回2
                if x.size(0) == 1:
                    return 1
                else:
                    return 2

            # 对func进行测试，输入大小为[1, 2]的随机张量，预期输出1
            test_input(func, torch.rand([1, 2]), 1)
            # 对func进行测试，输入大小为[1, 3]的随机张量，预期输出1
            test_input(func, torch.rand([1, 3]), 1)

            @torch.jit.script
            def neg_index(x):
                # 如果输入张量x的倒数第二个维度大小为1，返回1；否则返回2
                if x.size(-2) == 1:
                    return 1
                else:
                    return 2

            # 对neg_index进行测试，输入大小为[1, 2]的随机张量，预期输出1
            test_input(neg_index, torch.rand([1, 2]), 1)
            # 对neg_index进行测试，输入大小为[1, 3]的随机张量，预期输出1
            test_input(neg_index, torch.rand([1, 3]), 1)

        # 若图形执行模式为ProfilingMode.PROFILING，则执行测试大小索引函数test_size_index
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            test_size_index()

        # 定义测试数据类型的函数test_dtype
        def test_dtype():
            @torch.jit.script
            def func(x):
                # 如果输入张量x的数据类型为torch.float32，返回1；否则返回2
                if x.dtype == torch.float32:
                    return 1
                else:
                    return 2

            # 对func进行测试，输入数据为torch.tensor(0.5, dtype=torch.float32)，预期输出1
            test_input(func, torch.tensor(0.5, dtype=torch.float32), 1)
            # 对func进行测试，输入数据为torch.tensor(0.5, dtype=torch.int64)，预期输出2
            test_input(func, torch.tensor(0.5, dtype=torch.int64), 2)
        test_dtype()

        # 定义测试是否为浮点数的函数test_is_floating_poiint
        def test_is_floating_poiint():
            @torch.jit.script
            def func(x):
                # 如果输入张量x为浮点数类型，返回1；否则返回2
                if x.is_floating_point():
                    return 1
                else:
                    return 2

            # 对func进行测试，输入数据为torch.tensor(0.5, dtype=torch.float32)，预期输出1
            test_input(func, torch.tensor(0.5, dtype=torch.float32), 1)
            # 对func进行测试，输入数据为torch.tensor(0.5, dtype=torch.int64)，预期输出2
            test_input(func, torch.tensor(0.5, dtype=torch.int64), 2)
        test_is_floating_poiint()

        # 定义测试设备的函数test_device
        def test_device():
            @torch.jit.script
            def func_1(x):
                # 如果输入张量x的设备为cuda:0，返回0；否则返回1
                if x.device == torch.device('cuda:0'):
                    a = 0
                else:
                    a = 1
                return a

            @torch.jit.script
            def func_2(x):
                # 如果输入张量x在CUDA设备上，返回0；否则返回1
                if x.is_cuda:
                    a = 0
                else:
                    a = 1
                return a

            # 对func_1进行测试，输入torch.tensor(0.5)，预期输出1
            test_input(func_1, torch.tensor(0.5), 1)
            # 对func_2进行测试，输入torch.tensor(0.5)，预期输出1
            test_input(func_2, torch.tensor(0.5), 1)

            # 若运行环境支持CUDA，则执行以下测试
            if RUN_CUDA:
                # 对func_1进行测试，输入torch.tensor(0.5, device="cuda:0")，预期输出0
                test_input(func_1, torch.tensor(0.5, device="cuda:0"), 0)
                # 对func_2进行测试，输入torch.tensor(0.5, device="cuda:0")，预期输出0
                test_input(func_2, torch.tensor(0.5, device="cuda:0"), 0)

        # 执行设备测试函数test_device
        test_device()
    def test_attrs(self):
        def foo(x):
            return (
                # 获取张量的设备信息
                x.device,
                # 获取张量的形状
                x.shape,
                # 检查张量是否在 CUDA 上
                x.is_cuda,
                # 检查张量是否在 MKL-DNN 上
                x.is_mkldnn,
                # 检查张量是否量化
                x.is_quantized,
                # 检查张量是否需要梯度
                x.requires_grad,
                # 返回张量的转置（transpose）视图
                x.T,
                # 返回张量的主转置视图
                x.mT,
                # 返回张量的共轭转置（Hermitian transpose）视图
                x.H,
                # 返回张量的主共轭转置视图
                x.mH
                # x.layout TODO: layout long -> instance conversion
            )

        # 对 foo 函数进行 Torch 脚本化
        scripted = torch.jit.script(foo)
        # 创建一个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        # 断言 Torch 脚本化的结果与原始函数的结果一致
        self.assertEqual(scripted(x), foo(x))

    def test_layout(self):
        @torch.jit.script
        def check(x, y):
            # 检查两个张量的布局是否相同
            return x.layout == y.layout

        # 创建两个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)

        # 断言布局相同
        self.assertTrue(check(x, y))

    def test_matrix_transpose(self):
        @torch.jit.script
        def check(x):
            # 检查矩阵的主转置是否与 transpose(-2, -1) 的结果相等
            return torch.equal(x.mT, x.transpose(-2, -1))

        # 创建一个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        # 断言检查通过
        self.assertTrue(check(x))

    def test_transpose(self):
        @torch.jit.script
        def check(x):
            # 检查张量的转置是否与 t() 的结果相等
            return torch.equal(x.T, x.t())

        # 创建一个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        # 断言检查通过
        self.assertTrue(check(x))

    def test_matrix_conj_transpose(self):
        @torch.jit.script
        def check(x):
            # 检查矩阵的主共轭转置是否与 transpose(-2, -1).conj() 的结果相等
            return torch.equal(x.mH, x.transpose(-2, -1).conj())

        # 创建一个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        # 断言检查通过
        self.assertTrue(check(x))

        # 创建一个大小为 (3, 4)、位于 CPU 上、类型为 complex64 的张量
        x = make_tensor((3, 4), device="cpu", dtype=torch.complex64)
        # 断言检查通过
        self.assertTrue(check(x))

    def test_conj_transpose(self):
        @torch.jit.script
        def check(x):
            # 检查张量的共轭转置是否与 t().conj() 的结果相等
            return torch.equal(x.H, x.t().conj())

        # 创建一个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        # 断言检查通过
        self.assertTrue(check(x))

        # 创建一个大小为 (3, 4)、位于 CPU 上、类型为 complex64 的张量
        x = make_tensor((3, 4), device="cpu", dtype=torch.complex64)
        # 断言检查通过
        self.assertTrue(check(x))

    def test_T_mT_H_mH(self):
        def T(x):
            return x.mT

        def mT(x):
            return x.mT

        def H(x):
            return x.H

        def mH(x):
            return x.mH

        # 创建一个大小为 (3, 4) 的随机张量
        x = torch.rand(3, 4)
        # 创建一个大小为 (3, 4)、位于 CPU 上、类型为 complex64 的张量
        y = make_tensor((3, 4), device="cpu", dtype=torch.complex64)

        # 分别对 T、mT、H、mH 函数进行 Torch 脚本化的检查
        self.checkScript(T, (x, ))
        self.checkScript(mT, (x, ))
        self.checkScript(H, (x, ))
        self.checkScript(mH, (x, ))
        self.checkScript(T, (y, ))
        self.checkScript(mT, (y, ))
        self.checkScript(H, (y, ))
        self.checkScript(mH, (y, ))
    def test_nn_conv(self):
        # 定义一个测试类 Mod，继承自 nn.Module
        class Mod(nn.Module):
            # 初始化方法，接受一个卷积层对象 conv
            def __init__(self, conv):
                super().__init__()
                self.conv = conv

            # 前向传播方法，传入 input，并调用 self.conv 对象进行处理
            def forward(self, input):
                return self.conv(input)

        # 定义多组测试输入
        inputs = [
            # Conv1d
            (Mod(nn.Conv1d(16, 33, 3, stride=2)), torch.randn(20, 16, 5)),
            # Conv2d
            (Mod(nn.Conv2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10)),
            # Conv3d
            (Mod(nn.Conv3d(16, 33, 3, stride=2)), torch.randn(20, 16, 3, 5, 4)),
            # ConvTranspose1d
            (Mod(nn.ConvTranspose1d(16, 33, 3, stride=2)), torch.randn(20, 16, 5)),
            # ConvTranspose2d
            (Mod(nn.ConvTranspose2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10)),
            # ConvTranspose3d
            (Mod(nn.ConvTranspose3d(16, 33, 3, stride=2)), torch.randn(20, 16, 3, 5, 4)),
        ]

        # 遍历每组输入，并调用 self.checkModule 进行测试
        for m, inp in inputs:
            self.checkModule(m, (inp,))

    # 使用条件跳过装饰器，根据 GRAPH_EXECUTOR 的值判断是否跳过测试
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, 'Not implemented for Simple or Legacy')
    def test_debug_flush_compilation_cache(self):
        # 定义一个简单的函数 foo，返回 x + 2
        def foo(x):
            return x + 2

        # 定义一个简单的 nn.Module 类 Mod，实现了一个简单的前向传播操作
        class Mod(nn.Module):
            def forward(self, t):
                return t + 2

        # 使用 torch.jit.script 方法将 Mod 类转换为 Torch 脚本
        m = torch.jit.script(Mod())
        # 创建一个随机张量 x
        x = torch.rand(1, 10)

        # 启用测试模式，调用 self.checkScript 对 foo 进行脚本化测试
        with enable_profiling_mode_for_profiling_tests():
            jitted = self.checkScript(foo, (x,))
            # 获取调试状态，不应抛出异常
            states = jitted.get_debug_state()

            # 调用 _debug_flush_compilation_cache 方法来刷新编译缓存
            jitted._debug_flush_compilation_cache()
            # 断言应该抛出 RuntimeError 异常，包含 "INTERNAL ASSERT FAILED"
            with self.assertRaisesRegex(RuntimeError, "INTERNAL ASSERT FAILED"):
                states = jitted.get_debug_state()

            # 设置运行次数 NUM_RUNS
            NUM_RUNS = 1
            with num_profiled_runs(NUM_RUNS):
                m(x)
                m(x)
                # 获取模型的调试状态
                fwd = m._c._get_method("forward")
                states = m.get_debug_state()

                # 刷新前向方法的编译缓存
                fwd._debug_flush_compilation_cache()
                # 断言应该抛出 RuntimeError 异常，包含 "INTERNAL ASSERT FAILED"
                with self.assertRaisesRegex(RuntimeError, "INTERNAL ASSERT FAILED"):
                    states = m.get_debug_state()

    def test_numel(self):
        # 定义一个 Torch 脚本函数，返回输入张量的元素个数
        @torch.jit.script
        def get_numel_script(x):
            return x.numel()

        # 创建一个随机张量 x
        x = torch.rand(3, 4)
        # 调用 get_numel_script 函数获取其元素个数
        numel = get_numel_script(x)
        # 断言获取的元素个数与 x.numel() 相等
        self.assertEqual(numel, x.numel())

    def test_element_size(self):
        # 定义一个 Torch 脚本函数，返回输入张量的每个元素占据的字节数
        @torch.jit.script
        def get_element_size_script(x):
            return x.element_size()

        # 创建一个随机张量 x
        x = torch.rand(3, 4)
        # 调用 get_element_size_script 函数获取其每个元素占据的字节数
        element_size = get_element_size_script(x)
        # 断言获取的字节数与 x.element_size() 相等
        self.assertEqual(element_size, x.element_size())
    # 定义测试函数 test_Sequential，用于测试 Sequential 模块的功能
    def test_Sequential(self):
        # 定义一个继承自 nn.Module 的子类 Seq
        class Seq(nn.Module):
            # 初始化函数，构建一个包含两个线性层的 Sequential 模块
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))

            # 使用 TorchScript 注解的前向传播方法
            @torch.jit.script_method
            def forward(self, x):
                # 遍历 Sequential 模块中的每一层，并逐层应用到输入 x 上
                for l in self.seq:
                    x = l(x)
                return x

        # 将 Seq 类实例化为 TorchScript 模块 m
        m = torch.jit.script(Seq())
        # 断言模块 m 已经成功编译成图形表示，即 JIT 编译成功
        assert m.graph  # ensure jit was able to compile

    # 定义测试函数 test_ModuleList，用于测试 ModuleList 模块的功能
    def test_ModuleList(self):
        # 定义一个继承自 nn.Module 的子类 Mod
        class Mod(nn.Module):
            # 初始化函数，构建一个包含多个线性层的 ModuleList
            def __init__(self):
                super().__init__()
                # 创建包含 10 个 nn.Linear(10, 10) 线性层的 ModuleList
                self.model = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
                # 添加一个 nn.Linear(10, 20) 线性层到 ModuleList
                self.model += (nn.Linear(10, 20),)
                # 在 ModuleList 中追加一个 nn.Linear(20, 30) 线性层
                self.model.append(nn.Linear(20, 30))
                # 在 ModuleList 中扩展两个线性层：nn.Linear(30, 40) 和 nn.Linear(40, 50)
                self.model.extend([nn.Linear(30, 40), nn.Linear(40, 50)])

            # 前向传播方法，遍历 ModuleList 中的每一层，并逐层应用到输入 v 上
            def forward(self, v):
                for m in self.model:
                    v = m(v)
                return v

        # 将 Mod 类实例化为 TorchScript 模块 m
        m = torch.jit.script(Mod())
        # 断言模块 m 已经成功编译成图形表示，即 JIT 编译成功
        assert m.graph  # ensure jit was able to compile

    # 定义测试函数 test_disabled，用于测试 TorchScript 的禁用状态
    def test_disabled(self):
        # 禁用 TorchScript 编译功能
        torch.jit._state.disable()
        try:
            # 定义一个简单的函数 f，对两个输入 x 和 y 进行加法操作
            def f(x, y):
                return x + y

            # 使用 TorchScript 对函数 f 进行跟踪，返回的应该是原始函数 f
            self.assertIs(torch.jit.trace(f, (torch.randn(2, 2), torch.randn(2, 2))), f)
            # 使用 TorchScript 对函数 f 进行脚本化，返回的应该是原始函数 f
            self.assertIs(torch.jit.script(f), f)

            # 定义一个继承自 torch.jit.ScriptModule 的 MyModule 类
            class MyModule(torch.jit.ScriptModule):
                # 使用 TorchScript 注解的方法 method，接受输入 x 并返回它本身
                @torch.jit.script_method
                def method(self, x):
                    return x

            # 断言 MyModule 的 method 方法是一个方法或函数对象
            # 因为在某些 Python 版本中，ScriptModule 不会简单地变为 Module
            self.assertTrue(inspect.ismethod(MyModule.method) or inspect.isfunction(MyModule.method))
        finally:
            # 恢复 TorchScript 的启用状态
            torch.jit._state.enable()
    def test_train_eval(self):
        # 定义一个简单的子模块，根据训练状态返回不同的输出
        class Sub(nn.Module):
            def forward(self, input):
                if self.training:
                    return input
                else:
                    return -input

        # 定义一个使用torch.jit脚本的模块，对输入施加一个变换并加一
        class MyModule(torch.jit.ScriptModule):
            def __init__(self, module):
                super().__init__()
                self.module = module

            @torch.jit.script_method
            def forward(self, input):
                return self.module(input) + 1

        # 实例化MyModule，并测试其在不同状态下的行为
        m = MyModule(Sub())
        input = torch.rand(3, 4)
        self.assertEqual(input + 1, m(input))
        m.eval()
        self.assertEqual(-input + 1, m(input))

        # 测试批量归一化和dropout在训练和评估状态下的行为
        input = torch.randn(6, 10)
        batchnorm = nn.BatchNorm1d(10)
        dropout = nn.Dropout(p=0.2)

        m_batchnorm = MyModule(batchnorm)
        self.assertEqual(batchnorm(input) + 1, m_batchnorm(input))
        batchnorm.eval()
        m_batchnorm.eval()
        self.assertEqual(batchnorm(input) + 1, m_batchnorm(input))

        m_dropout = MyModule(dropout)
        dropout.eval()
        m_dropout.eval()
        self.assertEqual(dropout(input) + 1, m_dropout(input))

    def test_nn_lp_pool2d(self):
        # 定义一个包含LPPool2d层的模块，并测试其功能
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.LPPool2d(2, 3)
                self.n = torch.nn.LPPool2d(2, (7, 1))

            def forward(self, x):
                return (self.l(x),
                        self.n(x),
                        torch.nn.functional.lp_pool2d(x, float(2), 3),
                        torch.nn.functional.lp_pool2d(x, 2, 3),
                        torch.nn.functional.lp_pool2d(x, float(2), (7, 1)))

        # 使用checkModule函数检查Mod模块的行为
        self.checkModule(Mod(), (torch.rand(1, 3, 7, 7),))

    def test_nn_lp_pool1d(self):
        # 定义一个包含LPPool1d层的模块，并测试其功能
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.LPPool1d(2, 3)
                self.n = torch.nn.LPPool1d(2, 7)

            def forward(self, x):
                return (self.l(x),
                        self.n(x),
                        torch.nn.functional.lp_pool1d(x, float(2), 3),
                        torch.nn.functional.lp_pool1d(x, 2, 3),
                        torch.nn.functional.lp_pool1d(x, float(2), 7))

        # 使用checkModule函数检查Mod模块的行为
        self.checkModule(Mod(), (torch.rand(1, 3, 7),))

    def test_nn_padding_functional(self):
        # 定义一个包含F.pad函数调用的模块，并测试其功能
        class Mod(nn.Module):
            def __init__(self, *pad):
                super().__init__()
                self.pad = pad

            def forward(self, x):
                return F.pad(x, self.pad, mode='constant', value=3.5)

        # 准备不同维度输入和对应的Mod实例，使用checkModule函数检查它们的行为
        inputs = [
            (Mod(1, 2), torch.randn(1, 3, 4)),  # 1D
            (Mod(1, 2, 3, 4), torch.randn(1, 3, 4)),  # 2D
            (Mod(1, 2, 3, 4, 5, 6), torch.randn(1, 3, 4)),  # 3D
        ]

        for m, inp in inputs:
            self.checkModule(m, (inp,))
    def test_nn_padding(self):
        # 定义一个用于测试填充操作的测试函数
        class Mod(nn.Module):
            def __init__(self, padding):
                super().__init__()
                self.padding = padding

            def forward(self, input):
                return self.padding(input)

        # 准备输入数据和相应的模型实例
        inputs = [
            (Mod(nn.ConstantPad1d(2, 3.5)), torch.randn(1, 2, 4)),  # 在1维数据的两端使用常量填充
            (Mod(nn.ConstantPad2d(2, 3.5)), torch.randn(1, 2, 2)),  # 在2维数据的四周使用常量填充
            (Mod(nn.ConstantPad3d(3, 3.5)), torch.randn(16, 3, 10, 20, 30)),  # 在3维数据的六个面使用常量填充
            (Mod(nn.ReflectionPad1d(2)), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)),  # 在1维数据的两端使用反射填充
            (Mod(nn.ReflectionPad2d(2)), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)),  # 在2维数据的四周使用反射填充
            (Mod(nn.ReflectionPad3d(3)), torch.randn(16, 3, 8, 32, 48)),  # 在3维数据的六个面使用反射填充
            (Mod(nn.ReplicationPad1d(2)), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)),  # 在1维数据的两端使用复制填充
            (Mod(nn.ReplicationPad2d(2)), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)),  # 在2维数据的四周使用复制填充
            (Mod(nn.ReplicationPad3d(3)), torch.randn(16, 3, 8, 32, 48)),  # 在3维数据的六个面使用复制填充
            (Mod(nn.ZeroPad2d(2)), torch.randn(1, 1, 3, 3))  # 在2维数据的四周使用零填充
        ]

        # 对每一个输入进行模型测试
        for m, inp in inputs:
            self.checkModule(m, (inp,))

    def test_script_autograd_grad(self):
        # 定义测试自动求导和脚本化的函数
        def test_simple_grad(x, y):
            # type: (Tensor, Tensor) -> List[Optional[Tensor]]
            # 计算简单的梯度
            z = x + 2 * y + x * y
            return torch.autograd.grad((z.sum(), ), (x, y))

        def test_simple_grad_with_grad_outputs(x, y):
            # type: (Tensor, Tensor) -> List[Optional[Tensor]]
            # 计算带有梯度输出的简单梯度
            z = x + 2 * y + x * y
            grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones((2, 2)), ])
            return torch.autograd.grad((z, ), (x, y), grad_outputs)

        def test_one_output_not_requires_grad(x, y):
            # type: (Tensor, Tensor) -> List[Optional[Tensor]]
            # 计算仅有一个输出不需要梯度的情况
            z = 2 * y + y
            return torch.autograd.grad((z.sum(),), (x, y), allow_unused=True)

        def test_retain_graph(x, y):
            # type: (Tensor, Tensor) -> None
            # 测试保留计算图的情况
            z = x + 2 * y + x * y
            torch.autograd.grad((z.sum(), ), (x, y), retain_graph=True)
            torch.autograd.grad((z.sum(), ), (x, y))

        # 创建输入张量，并要求其梯度
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        # 分别对每一个测试函数使用脚本化测试
        self.checkScript(test_simple_grad, (x, y), inputs_requires_grad=True)
        self.checkScript(test_simple_grad_with_grad_outputs, (x, y), inputs_requires_grad=True)
        self.checkScript(test_one_output_not_requires_grad, (x, y), inputs_requires_grad=True)
        self.checkScript(test_retain_graph, (x, y), inputs_requires_grad=True)
    # 定义测试方法 test_script_backward，用于测试反向传播脚本化函数的行为
    def test_script_backward(self):
        # 定义内部函数 checkBackwardScript，用于检查脚本化函数的反向传播结果是否正确
        def checkBackwardScript(fn, inputs):
            # 将原始函数 fn 脚本化
            scripted_fn = torch.jit.script(fn)
            # 使用 FileCheck 检查脚本化函数的代码是否包含 "torch.autograd.backward"
            FileCheck().check("torch.autograd.backward").run(scripted_fn.code)
            # 将输入映射为需要梯度的张量，并记录输入
            recording_inputs = do_input_map(lambda t: t.detach().requires_grad_(), inputs)

            # 调用原始函数和脚本化函数
            fn(*inputs)
            scripted_fn(*recording_inputs)

            # 检查每个输入的梯度是否相等
            for inp1, inp2 in zip(inputs, recording_inputs):
                self.assertEqual(inp1.grad, inp2.grad)

        # 定义测试单个张量的反向传播函数 test_tensor_backward
        def test_tensor_backward(input):
            # type: (Tensor) -> None
            # 计算张量的ReLU和softmax
            output = torch.relu(input)
            output = output.softmax(0)
            # 对输出进行求和，并进行反向传播
            sum_out = output.sum()
            sum_out.backward()

        # 定义使用 torch.autograd.backward 进行反向传播的测试函数 test_torch_autograd_backward
        def test_torch_autograd_backward(input):
            # type: (Tensor) -> None
            # 计算张量的ReLU和softmax
            output = torch.relu(input)
            output = output.softmax(0)
            # 对输出的和进行反向传播
            torch.autograd.backward(output.sum())

        # 定义使用 torch.autograd.backward 和梯度张量进行反向传播的测试函数 test_torch_autograd_backward_with_grad_tensors
        def test_torch_autograd_backward_with_grad_tensors(input):
            # type: (Tensor) -> None
            # 计算张量的ReLU和softmax
            output = torch.relu(input)
            output = output.softmax(0)
            # 准备梯度输出张量列表，用 torch.autograd.backward 进行反向传播
            grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones((2, 2)), ])
            torch.autograd.backward((output,), grad_outputs)

        # 创建一个随机张量作为输入，要求计算其梯度
        inp = torch.randn(2, 2, requires_grad=True)
        # 分别对三个反向传播函数进行测试
        checkBackwardScript(test_tensor_backward, (inp,))
        checkBackwardScript(test_torch_autograd_backward, (inp,))
        checkBackwardScript(test_torch_autograd_backward_with_grad_tensors, (inp,))
    # 定义一个测试函数 test_diff_subgraph_clones_constants，用于测试自动微分子图中常量的克隆情况
    def test_diff_subgraph_clones_constants():
        # 使用 torch.jit.script 装饰器将函数 f 编译为 Torch 脚本
        @torch.jit.script
        def f(x, y):
            # 返回一个复杂的表达式，其中包含多次对 x 和 y 的加法操作
            return x + x + y + x + y + x + y + x + y + x

        # 定义一个函数 count_constants，用于统计图中的常量节点数量
        def count_constants(graph):
            # 统计所有节点中类型为 'prim::Constant' 的节点数量
            return sum(node.kind() == 'prim::Constant' for node in graph.nodes())

        # 复制函数 f 的图形对象
        graph = f.graph.copy()
        # 运行常量合并优化 Pass 'cse'，在图上执行常量合并优化
        self.run_pass('cse', graph)
        # 运行创建自动微分子图的 Pass 'create_autodiff_subgraphs'
        self.run_pass('create_autodiff_subgraphs', graph)
        # 获取图中所有节点的列表
        nodes = list(graph.nodes())
        # 断言常量节点的数量为 1
        self.assertEqual(count_constants(graph), 1)
        # 断言第二个节点的子图中常量节点的数量为 1
        self.assertEqual(count_constants(nodes[1].g('Subgraph')), 1)
    def test_arg_configurations(self):
        """Different arg configurations should trigger different traces"""
        # 创建一个 4x4 的随机张量 x
        x = Variable(torch.FloatTensor(4, 4).uniform_())
        # 创建 x 的双精度版本 x_double
        x_double = Variable(x.data.double())
        # 创建带梯度的 x_grad 张量
        x_grad = Variable(x.data.clone(), requires_grad=True)
        # 创建一个形状为 (4,) 的随机张量 y
        y = Variable(torch.randn(4))

        # 不同的参数组合列表
        configurations = [
            (x,),  # 单个 x
            (x_double,),  # 单个 x_double
            (x_grad,),  # 单个 x_grad
            (y,),  # 单个 y
            ([x, x],),  # [x, x] 组合
            ([x, y],),  # [x, y] 组合
        ]
        # 如果 CUDA 可用，添加 CUDA 版本的参数组合
        if torch.cuda.is_available():
            x_cuda = Variable(x.data.cuda())
            configurations += [
                (x_cuda,),  # 单个 x_cuda
                ([x, x_cuda],),  # [x, x_cuda] 组合
                ([x_cuda, x],),  # [x_cuda, x] 组合
                ([[x_cuda, x]],),  # [[x_cuda, x]] 组合
            ]
            # 如果有多个 CUDA 设备，添加多个 CUDA 设备的参数组合
            if torch.cuda.device_count() > 1:
                x_cuda_1 = Variable(x.data.cuda(1))
                configurations += [
                    (x_cuda_1,),  # 单个 x_cuda_1
                    ([x_cuda, x_cuda_1],),  # [x_cuda, x_cuda_1] 组合
                ]

        # 编译函数 fn 以进行 Torch JIT 编译，不接受任何导数
        @torch.jit.compile(nderivs=0)
        def fn(*args):
            # 将参数展平为输入变量的列表
            in_vars, _ = torch._C._jit_flatten(args)
            # 返回输入变量的第一个元素加 1
            return in_vars[0] + 1

        # 遍历参数配置列表
        for i, config in enumerate(configurations):
            # 断言 fn 没有给定配置的追踪
            self.assertFalse(fn.has_trace_for(*config))
            # 执行 fn 函数并记录追踪
            fn(*config)
            # 断言 fn 有给定配置的追踪
            self.assertTrue(fn.has_trace_for(*config))
            # 对于后续的未知配置，断言 fn 没有追踪
            for unk_config in configurations[i + 1:]:
                self.assertFalse(fn.has_trace_for(*unk_config))
        # 断言 fn 的 hits 属性为 0
        self.assertEqual(fn.hits, 0)

    def test_torch_sum(self):
        # 定义求和函数 fn
        def fn(x):
            return torch.sum(x)

        # 定义带维度参数的求和函数 fn1
        def fn1(x, dim: int):
            return torch.sum(x, dim)

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 检查 fn 函数的 Torch 脚本
        self.checkScript(fn, (x, ))
        # 检查 fn1 函数维度为 1 的 Torch 脚本
        self.checkScript(fn1, (x, 1, ))
        # 检查 fn1 函数维度为 0 的 Torch 脚本
        self.checkScript(fn1, (x, 0, ))

    def test_cse(self):
        # 创建带梯度的张量 x 和 y
        x = torch.tensor([0.4, 0.3], requires_grad=True)
        y = torch.tensor([0.7, 0.5], requires_grad=True)

        # 定义计算函数 fn
        def fn(x, y):
            # 计算 w = (x + y)^3
            w = (x + y) * (x + y) * (x + y)
            # 计算 t = tanh(w) + tanh(w)
            t = torch.tanh(w) + torch.tanh(w)
            # 计算 z = (x + y)^3 + t
            z = (x + y) * (x + y) * (x + y) + t
            return z

        # 获取函数 fn 的追踪图形 g
        g, _ = torch.jit._get_trace_graph(fn, (x, y))
        # 运行传递优化 "cse"
        self.run_pass('cse', g)
        do_exactly = True
        # 使用 FileCheck 检查优化后的图形 g
        FileCheck().check_count("add", 1).check_count("mul", 2, do_exactly) \
            .check_count("tanh", 1, do_exactly).check_count("add", 2, do_exactly).check_next("return")  \
            .run(str(g))
        # 断言导出和导入后的图形 g 保持一致
        self.assertExportImport(g, (x, y))
    # 定义测试函数，验证常量传播与CSE（公共子表达式消除）是否正常工作
    def test_cse_not_introduce_aliasing(self):
        # 定义一个torch脚本函数tensor_alias_outputs，返回输入张量的两倍
        @torch.jit.script
        def tensor_alias_outputs(x):
            return x + x, x + x

        # 在tensor_alias_outputs的计算图上运行CSE优化
        self.run_pass('cse', tensor_alias_outputs.graph)
        # 使用FileCheck工具检查计算图中"aten::add"操作的数量是否为2
        FileCheck().check_count("aten::add", 2).run(tensor_alias_outputs.graph)

        # 定义一个torch脚本函数ints_alias_outputs，接受整数并返回整数的两倍
        @torch.jit.script
        def ints_alias_outputs(x):
            # type: (int) -> Tuple[int, int]
            return x + x, x + x

        # 对ints_alias_outputs的计算图运行CSE优化
        # 非别名类型可以进行公共子表达式消除
        self.run_pass('cse', ints_alias_outputs.graph)
        # 使用FileCheck工具确切地检查计算图中"aten::add"操作的数量是否为1
        FileCheck().check_count("aten::add", 1, exactly=True).run(ints_alias_outputs.graph)

    # 定义测试函数，验证递归CSE是否正常工作
    def test_recursive_cse(self):
        # 输入的字符串为空，因为这部分还未给出
# 定义一个名为 graph 的函数，接受三个参数：%x（Tensor 类型）、%y（Tensor 类型）、%20（整数类型）
graph(%x : Tensor,
      %y : Tensor,
      %20 : int):
  # 创建一个整数常量 %2，其值为 1
  %2 : int = prim::Constant[value=1]()
  # 计算 %x 和 %y 的和，结果存储在 %3 中
  %3 : Tensor = aten::add(%x, %y, %2)
  # 将 %2 和 %20 相加，结果存储在 %4 中
  %4 : int = aten::add(%2, %20)
  # 将 %4 转换为布尔类型，结果存储在 %5 中
  %5 : bool = aten::Bool(%4)
  # 使用条件判断 %5 进入不同的分支
  %z : int = prim::If(%5)
    # CHECK: block
    block0():
      # CHECK-NOT: aten::add
      # 如果条件为真，执行此块，计算 %2 和 %20 的和，结果存储在 %z.1 中
      %z.1 : int = aten::add(%2, %20)
      -> (%z.1)
    block1():
      # 如果条件为假，执行此块，直接返回 %2
      -> (%2)
  # 返回 %z 作为函数结果
  return (%z)
"""
# 从输入字符串解析成一个图形对象
graph = parse_ir(input_str)
# 运行名为 'cse' 的优化 pass，优化传入的图形对象 graph
self.run_pass('cse', graph)
# 使用 FileCheck 运行输入字符串与 graph 的匹配检查
FileCheck().run(input_str, graph)

def test_pattern_based_rewrite(self):
    # mul(mul(mul(mul(x,y),z),x),y) --> mul(mul(mulmul(x,y,z), x), y) -->
    # --> mulmul(mulmul(x,y,z), x, y)
    # 定义一个输入字符串 input_str，包含自定义的匹配规则
    input_str = """
graph(%x, %y, %z):
    # CHECK-NOT: aten::mul
    # CHECK: my::fused_mulmul
    # 计算 %x 和 %y 的乘积，结果存储在 %t 中
    %t = aten::mul(%x, %y)
    # 计算 %t 和 %z 的乘积，结果存储在 %p 中
    %p = aten::mul(%t, %z)
    # 使用自定义函数 my::fused_mulmul 计算 %p、%x 和 %y 的乘积，结果存储在 %u 中
    %u = aten::mul(%p, %x)
    # 计算 %u 和 %y 的乘积，结果存储在 %o 中
    %o = aten::mul(%u, %y)
    # 返回 %o 作为函数结果
    return (%o)"""
  # 从输入字符串解析成一个图形对象
  graph = parse_ir(input_str)
  # 使用自定义的图形重写规则进行优化
  torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%a, %b, %c):
  %q = aten::mul(%a, %b)
  %r = aten::mul(%q, %c)
  return (%r)""", """
graph(%a, %b, %c):
  %r = my::fused_mulmul(%a, %b, %c)
  return (%r)""", graph)
  # 使用 FileCheck 运行输入字符串与 graph 的匹配检查
  FileCheck().run(input_str, graph)

  # Check that overlapping matches are handled correctly
  # mul(mul(mul(x,y),z),x) --> mul(mulmul(x,y,z), x)
  # 定义一个输入字符串 input_str，包含自定义的匹配规则
  input_str = """
  graph(%x, %y, %z):
  # CHECK-NOT: aten::mul
  # CHECK: my::fused_mulmul
  # 计算 %x 和 %y 的乘积，结果存储在 %t 中
  %t = aten::mul(%x, %y)
  # 计算 %t 和 %z 的乘积，结果存储在 %p 中
  %p = aten::mul(%t, %z)
  # CHECK-NEXT: aten::mul
  # 使用自定义函数 my::fused_mulmul 计算 %p、%x 和 %y 的乘积，结果存储在 %u 中
  %u = aten::mul(%p, %x)
  # 返回 %u 作为函数结果
  return (%u)"""
  # 从输入字符串解析成一个图形对象
  graph = parse_ir(input_str)
  # 使用自定义的图形重写规则进行优化
  torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
  graph(%a, %b, %c):
  %q = aten::mul(%a, %b)
  %r = aten::mul(%q, %c)
  return (%r)""", """
  graph(%a, %b, %c):
  %r = my::fused_mulmul(%a, %b, %c)
  return (%r)""", graph)
  # 使用 FileCheck 运行输入字符串与 graph 的匹配检查
  FileCheck().run(input_str, graph)

  # Check add(mul(x,y),z) --> muladd(x,y,z) replacement
  # 定义一个输入字符串 input_str，包含自定义的匹配规则
  input_str = """
  graph(%x, %y, %z):
  # CHECK-NOT: aten::mul
  # CHECK-NOT: aten::add
  # 创建一个整数常量 %c，其值为 1
  %c = prim::Const[value=1]()
  # 计算 %x 和 %y 的乘积，结果存储在 %t 中
  %t = aten::mul(%x, %y)
  # 计算 %t 和 %z 的和，结果存储在 %p 中
  %p = aten::add(%t, %z, %c)
  # CHECK: my::muladd
  # CHECK-NEXT: return
  # 返回 %p 作为函数结果
  return (%p)"""
  # 从输入字符串解析成一个图形对象
  graph = parse_ir(input_str)
  # 使用自定义的图形重写规则进行优化
  torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
  graph(%a, %b, %c, %d):
  %q = aten::mul(%a, %b)
  %r = aten::add(%q, %c, %d)
  return (%r)""", """
  graph(%a, %b, %c, %d):
  %r = my::muladd(%a, %b, %c, %d)
  return (%r)""", graph)
  # 使用 FileCheck 运行输入字符串与 graph 的匹配检查
  FileCheck().run(input_str, graph)

  # Check add(mul(x,y),z) --> sub(add(x,y),z) replacement
  # 定义一个输入字符串 input_str，包含自定义的匹配规则
  input_str = """
  graph(%x, %y, %z):
  # CHECK-NOT: aten::mul
  # 创建一个整数常量 %c，其值为 1
  %c = prim::Const[value=1]()
  # CHECK: aten::add
  # 计算 %x 和 %y 的乘积，结果存储在 %t 中
  %t = aten::mul(%x, %y)
  # CHECK-NEXT: aten::sub
  # 计算 %t 和 %z 的和，结果存储在 %p 中
  %p = aten::add(%t, %z, %c)
  # CHECK-NOT: aten::add
  # CHECK-NEXT: return
  # 返回 %p 作为函数结果
  return (%p)"""
  # 从输入字符串解析成一个图形对象
  graph = parse_ir(input_str)
  # 使用自定义的图形重写规则进行优化
  torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
        # 定义名为 graph 的函数，接受参数 %a, %b, %c, %d
        def graph(%a, %b, %c, %d):
          # 计算 %q = %a * %b
          %q = aten::mul(%a, %b)
          # 计算 %r = %q + %c + %d
          %r = aten::add(%q, %c, %d)
          # 返回结果 %r
          return (%r)
    # 定义测试函数，用于测试在循环中使用unsqueeze操作的形状分析
    def test_shape_analysis_unsqueeze_in_loop(self):
        # 定义输入的 IR 字符串
        input_str = """graph(%x.1 : Tensor):
          %4 : bool = prim::Constant[value=1]()
          %1 : int = prim::Constant[value=2]()
          %7 : int = prim::Constant[value=0]()
          # CHECK: FloatTensor(requires_grad=0, device=cpu) = prim::Loop
          %x : Tensor = prim::Loop(%1, %4, %x.1)
            # CHECK: : FloatTensor(requires_grad=0, device=cpu)):
            block0(%i : int, %x.6 : Tensor):
              # CHECK: FloatTensor(requires_grad=0, device=cpu) = aten::unsqueeze
              %x.3 : Tensor = aten::unsqueeze(%x.6, %7)
              -> (%4, %x.3)
          return (%x)"""
        # 解析 IR 字符串并获取计算图
        graph = parse_ir(input_str)
        # 执行完整的形状分析过程
        torch._C._jit_pass_complete_shape_analysis(graph, (torch.zeros(2, 2, dtype=torch.float32),), False)
        # 使用 FileCheck 检查 IR 字符串和计算图是否匹配
        FileCheck().run(input_str, graph)

    # 定义测试函数，用于测试脚本化的张量类型转换
    def test_script_tensor_type(self):
        # 定义一个函数 foo，接受一个张量 x 和一个类型 t，并返回 x 的类型转换结果
        def foo(x, t: torch.dtype):
            return x.type(t)
        # 对函数 foo 进行脚本化
        scr = torch.jit.script(foo)
        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 对一系列类型进行遍历，验证脚本化后的函数和原始函数的结果是否一致
        for t in [torch.int8, torch.float64, torch.float32,
                  torch.bfloat16, torch.complex64, torch.complex128, torch.bool]:
            self.assertEqual(scr(x, t), foo(x, t))

    # 定义测试函数，用于测试脚本化的布尔字面量转换
    def test_script_bool_literal_conversion(self):
        # 定义一个函数 foo，接受一个张量 x，并返回 x 与 True 的逐元素乘积
        def foo(x):
            return torch.mul(x, True)
        # 对函数 foo 进行脚本化
        scr = torch.jit.script(foo)
        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 验证脚本化后的函数和原始函数的结果是否一致
        self.assertEqual(scr(x), foo(x))

    # 定义测试函数，用于测试形状分析中的 masked_select 操作
    def test_shape_analysis_masked_select(self):
        # 定义输入的 IR 字符串
        input_str = """graph(%0 : Float(),
          %1 : Bool()):
          # CHECK: Float(*, requires_grad=0, device=cpu) = aten::masked_select
          %2 : Tensor = aten::masked_select(%0, %1) # test/test_jit.py:15261:0
          return (%2)"""
        # 解析 IR 字符串并获取计算图
        graph = parse_ir(input_str)
        # 创建一个张量 x，其值为全 1，数据类型为 float32
        x = torch.ones(1, dtype=torch.float32)[0]
        # 根据 x 的值生成一个掩码 mask，大于等于 0.5 的位置为 True，否则为 False
        mask = x.ge(0.5)
        # 执行完整的形状分析过程
        torch._C._jit_pass_complete_shape_analysis(graph, (x, mask), False)
        # 使用 FileCheck 检查 IR 字符串和计算图是否匹配
        FileCheck().run(input_str, graph)

    # TODO: update verify to work with GraphExecutors
    @unittest.skip("verify needs to be updated to work with GraphExecutors")
    def test_verify(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        @torch.jit.compile
        def f(x, y):
            z = torch.sigmoid(x * (x + y))
            w = torch.abs(x * x * x + y) + Variable(torch.ones(1))
            return z, w

        torch.jit.verify(f, (x, y), loss_fn=lambda z, w: z * w, devices=[])

    # TODO: adapt to a GraphExecutor test
    @unittest.skip("Need to instrument GraphExecutors a bit more")
    # 定义一个测试方法，用于测试标志设置的功能
    def test_flags(self):
        # 生成两个形状为 (2, 2) 的随机张量 x 和 y
        x, y = torch.randn(2, 2)
        # 创建一个带梯度的变量 y
        y = Variable(torch.randn(2, 2))

        # 使用 Torch JIT 编译装饰器，编译下面定义的函数 fn
        @torch.jit.compile
        def fn(x, y):
            # 计算表达式 (x * x + y * y + x * y) 的和
            return (x * x + y * y + x * y).sum()

        # 初始化梯度字典
        grads = {}

        # 使用 product 函数生成 (True, True), (True, False), (False, True), (False, False) 四种组合
        for rx, ry in product((True, False), repeat=2):
            # 根据 rx 和 ry 设置 x 和 y 是否需要梯度
            x.requires_grad = rx
            y.requires_grad = ry

            # 断言 fn 是否没有记录下 x, y 的追踪
            self.assertFalse(fn.has_trace_for(x, y))
            # 计算 fn(x, y)，得到输出 out
            out = fn(x, y)

            # 断言 fn 是否没有记录下 x, y 的追踪
            self.assertFalse(fn.has_trace_for(x, y))

            # 遍历 (x, 'x', rx), (y, 'y', ry) 元组列表
            for v, name, compute in [(x, 'x', rx), (y, 'y', ry)]:
                # 如果 compute 为 False，跳过本次循环
                if not compute:
                    continue
                # 计算 out 对 v 的梯度 grad_v，同时保留计算图以便多次反向传播
                grad_v, = torch.autograd.grad(out, v, retain_graph=True)
                # 将计算得到的梯度 grad_v 存入 grads 字典，并返回旧值或默认值
                expected_grad = grads.setdefault(name, grad_v)
                # 断言计算得到的梯度 grad_v 等于预期梯度 expected_grad
                self.assertEqual(grad_v, expected_grad)

            # 断言 fn 是否记录了 x 或 y 的追踪
            self.assertEqual(fn.has_trace_for(x, y), rx or ry)

    # 定义一个测试方法，用于测试 Python IR 的功能
    def test_python_ir(self):
        # 创建两个需要梯度的张量 x 和 y，分别为 [0.4] 和 [0.7]
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        # 定义一个函数 doit，计算 torch.sigmoid(torch.tanh(x * (x + y)))
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        # 获取 doit 函数的跟踪图 g，同时获取输出 outputs 和输入 inputs
        g, _ = torch.jit._get_trace_graph(doit, (x, y))

        # 在跟踪图 g 上运行 DCE（死代码删除）优化
        self.run_pass('dce', g)
        # 在跟踪图 g 上运行 canonicalize 优化
        self.run_pass('canonicalize', g)

        # 创建一个新的 Torch 图 g2
        g2 = torch._C.Graph()
        # 创建 g 和 g2 之间的映射关系 g_to_g2
        g_to_g2 = {}
        # 遍历 g 的输入节点，将它们添加到 g2，并更新映射关系
        for node in g.inputs():
            g_to_g2[node] = g2.addInput()

        # 遍历 g 的所有节点
        for node in g.nodes():
            # 在 g2 上克隆节点 node，使用 g_to_g2 更新输入和输出节点之间的映射
            n_ = g2.createClone(node, lambda x: g_to_g2[x])
            g2.appendNode(n_)
            for o, no in zip(node.outputs(), n_.outputs()):
                g_to_g2[o] = no

        # 将 g 的输出节点注册到 g2 上
        for node in g.outputs():
            g2.registerOutput(g_to_g2[node])

        # 在 g2 上创建一个新的 "prim::TensorTest" 节点 t_node，并设置其属性为 torch.ones([2, 2])
        t_node = g2.create("prim::TensorTest").t_("a", torch.ones([2, 2]))
        # 断言 t_node 的属性名称为 ["a"]
        self.assertEqual(t_node.attributeNames(), ["a"])
        # 在 g2 的末尾追加节点 t_node
        g2.appendNode(t_node)
        # 断言 torch.ones(2, 2) 是否等于 t_node 的属性 "a"
        self.assertTrue(torch.equal(torch.ones(2, 2), t_node.t("a")))

        # 遍历 g 的所有节点
        for node in g.nodes():
            # 断言 g2 是否能找到与 node 类型相同的节点
            self.assertTrue(g2.findNode(node.kind()) is not None)

    # 定义一个测试方法，用于测试 C++ 功能
    @unittest.skipIf(IS_SANDCASTLE, "gtest runs these in sandcastle")
    @unittest.skipIf(RUN_CUDA, "covered by test_cpp_cuda")
    @unittest.skipIf(not torch._C._jit_has_cpp_tests(), "Tests were not built, use BUILD_TEST=1")
    def test_cpp(self):
        # 从 cpp.jit 导入 tests_setup 模块
        from cpp.jit import tests_setup
        # 调用 tests_setup 模块的 setup 函数
        tests_setup.setup()
        # 运行 Torch C++ 测试
        torch._C._jit_run_cpp_tests()
        # 调用 tests_setup 模块的 shutdown 函数
        tests_setup.shutdown()

    # 定义一个测试方法，用于测试 BatchNorm2d 的功能
    def test_batchnorm(self):
        # 创建一个形状为 (2, 2, 2, 2) 的张量 x，其值全为 1
        x = torch.ones(2, 2, 2, 2)
        # 获取 nn.BatchNorm2d(2) 的跟踪图 g，同时获取输出 outputs 和输入 inputs
        g, outputs, inputs = torch.jit._get_trace_graph(nn.BatchNorm2d(2), x,
                                                        _force_outplace=True, return_inputs=True)
        # 从跟踪图 g 创建一个函数 m
        m = self.createFunctionFromGraph(g)
        # 断言输出 outputs 是否等于 m 的输入 inputs
        self.assertEqual(outputs, m(*inputs))
    # 定义一个单元测试方法，测试 torch 的 dropout 函数
    def test_dropout(self):
        # 创建一个全为1的张量
        x = torch.ones(2, 2)
        # 在没有指定设备的情况下，使用随机数生成器分叉
        with torch.random.fork_rng(devices=[]):
            # 获取 dropout 模块的跟踪图及其输入输出
            g, outputs, inputs = torch.jit._get_trace_graph(nn.Dropout(0.6), x, return_inputs=True)
        # 再次分叉随机数生成器
        with torch.random.fork_rng(devices=[]):
            # 从跟踪图创建一个函数
            m = self.createFunctionFromGraph(g)
            # 断言输出与函数应用于输入的结果相等
            self.assertEqual(outputs, m(*inputs))

    # 根据条件跳过测试：如果没有运行 CUDA，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "test requires CUDA")
    # 根据条件跳过测试：如果图执行器不是性能分析模式，则跳过该测试
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "skip if profiling isn't enabled")
    def test_native_dropout_corner_case(self):
        # 禁用自动微分子图内联
        with disable_autodiff_subgraph_inlining():
            # 定义一个函数 t，它应用了 torch.dropout
            def t(x, p: float, t: bool):
                o = torch.dropout(x, p, t)
                return o

            # 对函数 t 进行脚本化
            jit_t = torch.jit.script(t)
            # 创建一个随机张量，并针对其图进行检查，确保其包含可微分图
            x = torch.randn(5).requires_grad_()
            FileCheck().check("prim::DifferentiableGraph").run(jit_t.graph_for(x, 1.0, True, profile_and_replay=True))

            # 针对不同的训练状态、概率和设备进行测试
            for train in [True, False]:
                for p in [0.0, 1.0]:
                    for device in ["cuda", "cpu"]:
                        # 创建一个随机张量并移动到指定设备，同时要求梯度
                        x = torch.randn(5).to(device=device).requires_grad_()
                        x_ref = x.detach().requires_grad_()
                        # 对使用脚本化函数计算的输出和参考输出进行比较
                        o = jit_t(x, p, train)
                        o_ref = t(x_ref, p, train)
                        # 对输出求和并进行反向传播
                        o.sum().backward()
                        o_ref.sum().backward()
                        # 断言两个张量相等
                        assert o.equal(o_ref)
                        # 断言两个张量的梯度相等
                        assert x.grad.equal(x_ref.grad)

    # 标记为慢速测试，并根据执行器模式跳过测试
    @slowTest
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'Testing differentiable graph')
    # 定义一个测试函数，用于验证 dropout 模块是否正确设置梯度要求
    def test_dropout_module_requires_grad(self):
        # 在性能测试中启用性能模式
        with enable_profiling_mode_for_profiling_tests():
            # 定义一个自定义的 PyTorch 模块 MyModule
            class MyModule(torch.nn.Module):
                # 初始化函数，设置 dropout 和线性层
                def __init__(self, M):
                    super().__init__()
                    self.dropout = torch.nn.Dropout(0.5)
                    self.linear = torch.nn.Linear(M, M)

                # 前向传播函数
                def forward(self, input):
                    # 应用 dropout 到输入数据
                    input = self.dropout(input)
                    # 线性层处理 dropout 后的输入
                    output = self.linear(input)
                    return output

            # 定义一个用于分析函数调用的函数 profile
            def profile(func, X):
                # 使用 torch.autograd.profiler.profile 进行性能分析
                with torch.autograd.profiler.profile() as prof:
                    func(X)
                # 返回性能分析结果中的函数事件名列表
                return [e.name for e in prof.function_events]

            # 设定模块参数
            M = 1000
            # 使用 torch.jit.script 对 MyModule 进行脚本化
            scripted = torch.jit.script(MyModule(M))

            # 为了减少关于预期行为的混乱:
            #   requires_grad 控制是否对 dropout 进行符号微分。
            #   training 控制符号微分中是否调用 bernoulli_。
            # * 当 requires_grad == training 时，预期行为显而易见。
            # * 当 requires_grad=True 且 training=False 时，bernoulli_ 可能仍然出现在图中。
            #   但它在未调用的分支中。这就是为什么我们有 autograd 分析器的单独检查，以确保它没有运行。
            # * 当 requires_grad=False 且 training=True 时，由于训练模式下 dropout 层的预期行为，
            #   必须运行 bernoulli_。这与图是否需要梯度无关。实际上，在这种情况下，bernoulli_ 来自 autograd 而不是 autodiff。
            for training in (True, False):
                if training:
                    # 将脚本化模块设置为训练模式
                    scripted.train()
                else:
                    # 将脚本化模块设置为评估模式
                    scripted.eval()
                for requires_grad in (True, False):
                    # 创建随机张量作为输入
                    X = torch.randn(M, M, requires_grad=requires_grad)
                    if requires_grad:
                        # 使用 FileCheck 检查预期的图中是否包含 "aten::native_dropout"
                        FileCheck().check("aten::native_dropout").run(scripted.graph_for(X, profile_and_replay=True))
                    # 使用 assertEqual 检查训练状态下是否包含 "aten::bernoulli_"，以确认预期行为
                    self.assertEqual(training, 'aten::bernoulli_' in profile(scripted, X))

    # 根据 GRAPH_EXECUTOR 的值决定是否跳过测试，用于测试可微分图
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.SIMPLE, 'Testing differentiable graph')
    @skipIfTorchDynamo("Torchdynamo cannot correctly handle profiler.profile calls")
    # 定义测试函数，验证 dropout 函数在需要梯度时的行为
    def test_dropout_func_requires_grad(self):
        # 定义训练阶段的 dropout 函数
        def dropout_training(input):
            return F.dropout(input, 0.5, training=True)

        # 定义评估阶段的 dropout 函数
        def dropout_eval(input):
            return F.dropout(input, 0.5, training=False)

        # 定义性能分析函数，使用 Torch 的自动求导分析器
        def profile(func, X):
            with torch.autograd.profiler.profile() as prof:
                func(X)
            return [e.name for e in prof.function_events]

        M = 1000
        # 将 dropout_training 和 dropout_eval 函数转换为 Torch 脚本
        scripted_training = torch.jit.script(dropout_training)
        scripted_eval = torch.jit.script(dropout_eval)

        # 在测试期间禁用自动微分子图内联
        with disable_autodiff_subgraph_inlining():
            # 对于两种需要梯度和不需要梯度的情况
            for requires_grad in (True, False):
                # 创建一个随机张量 X，指定是否需要梯度
                X = torch.randn(M, M, requires_grad=requires_grad)
                if requires_grad:
                    # 检查训练阶段的 Torch 脚本图是否包含 native_dropout 操作
                    FileCheck().check("aten::native_dropout").run(scripted_training.graph_for(X, profile_and_replay=True))
                # 检查训练阶段和评估阶段的性能分析中是否包含 bernoulli_ 操作
                self.assertIn('aten::bernoulli_', profile(scripted_training, X))
                self.assertNotIn('aten::bernoulli_', profile(scripted_eval, X))

    # 如果 CUDA 可用，测试 dropout 在 CUDA 上的行为
    @unittest.skipIf(not RUN_CUDA, "test_dropout_cuda require CUDA")
    def test_dropout_cuda(self):
        # 在 CUDA 情况下，dropout 自动微分调度到 _fused_dropout
        def _zero_rate(t):
            return torch.true_divide((t == 0).sum(), t.numel())

        # 创建一个需要梯度的 CUDA 张量 x
        x = torch.ones(1000, 1000).cuda().requires_grad_()

        # 在性能分析模式下启用 CUDA 测试
        with enable_profiling_mode_for_profiling_tests():
            # 定义一个 Torch 脚本函数 func
            @torch.jit.script
            def func(x):
                return torch.nn.functional.dropout(x)

            # 冻结随机数生成器状态
            with freeze_rng_state():
                # 计算在 dropout 应用后的输出和梯度（参考结果）
                out_ref = torch.nn.functional.dropout(x)
                grad_ref = torch.autograd.grad(out_ref.sum(), x)

            # 再次冻结随机数生成器状态
            with freeze_rng_state():
                # 使用 Torch 脚本函数计算 dropout 应用后的输出和梯度
                out = func(x)
                grad = torch.autograd.grad(out.sum(), x)

            # 检查输出和梯度的零率是否统计上相同
            self.assertEqual(_zero_rate(out), _zero_rate(out_ref), rtol=1e-3, atol=1e-4)
            self.assertEqual(_zero_rate(grad[0]), _zero_rate(grad_ref[0]), rtol=1e-3, atol=1e-4)

    # 测试 Torch 操作重载情况
    def test_torch_ops_overloaded(self):
        # 使用断言捕获预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "failed to match any schema"):
            torch.ops.aten.add("a", 1)
        # 检查字符串连接操作的正确性
        self.assertEqual("ab", torch.ops.aten.add("a", "b"))
        # 创建两个随机张量 a 和 b
        a, b = torch.rand(3, 4), torch.rand(3, 4)
        # 检查张量加法操作的正确性
        self.assertEqual(a + b, torch.ops.aten.add(a, b))
        # 检查张量和标量加法操作的正确性
        self.assertEqual(a + 1, torch.ops.aten.add(a, 1))
    # 定义一个测试函数，用于测试 torch 操作中的关键字参数
    def test_torch_ops_kwonly(self):
        # 创建两个 3x4 的随机张量 a 和 b
        a, b = torch.rand(3, 4), torch.rand(3, 4)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并检查其错误消息是否包含 "positional argument"
        with self.assertRaisesRegex(RuntimeError, "positional argument"):
            # 调用 torch.ops.aten.add 函数，尝试传递额外的位置参数 2
            torch.ops.aten.add(a, b, 2)

        # 使用 torch.ops.aten.prod 计算张量 a 沿着第一维的乘积，并使用 self.assertEqual 断言比较结果
        # Chillee 指出这是一个模棱两可的情况
        self.assertEqual(a.prod(1), torch.ops.aten.prod(a, 1))

    # 定义一个测试函数，用于测试 torch 中的复数操作
    def test_torch_complex(self):
        # 定义一个函数 fn，接受实部和虚部，返回 torch 中的复数表示
        def fn(real, img):
            return torch.complex(real, img)

        # 定义一个函数 fn_out，接受实部、虚部和输出张量 out，并将结果存储在 out 中
        def fn_out(real, img, out):
            return torch.complex(real, img, out=out)

        # 使用 self.checkScript 检查 fn 函数的脚本化版本，传入不同的参数组合
        self.checkScript(fn, (torch.rand(3, 4), torch.rand(3, 4), ))
        self.checkScript(fn, (torch.ones(5, 1, 4), torch.ones(5, 1, 4), ))
        self.checkScript(fn, (torch.zeros(1, 6), torch.ones(6, 1), ))
        self.checkScript(fn, (torch.zeros(1, 6), torch.zeros(6, 1), ))
        self.checkScript(fn, (torch.empty(3, 4), torch.empty(3, 4), ))

        # 创建 torch.tensor 张量 real 和 img，用于测试 fn_out 函数
        real = torch.tensor([1, 2], dtype=torch.float32)
        img = torch.tensor([3, 4], dtype=torch.float32)
        # 创建一个空的复数张量 out，用于存储 fn_out 的结果
        out = torch.empty([3, 4], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        # 类似地，使用不同的数据类型和形状创建输入张量和输出复数张量，测试 fn_out 函数
        real = torch.tensor([5, 2], dtype=torch.float64)
        img = torch.tensor([3, 4], dtype=torch.float64)
        out = torch.empty([5, 2], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([1, 2])
        img = torch.ones([1, 2])
        out = torch.empty([1, 2], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([3, 8, 7])
        img = torch.ones([3, 8, 7])
        out = torch.empty([3, 8, 7], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.empty([3, 2, 6])
        img = torch.empty([3, 2, 6])
        out = torch.empty([3, 2, 6], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.zeros([1, 3])
        img = torch.empty([3, 1])
        out = torch.empty([3, 3], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([2, 5])
        img = torch.empty([2, 1])
        out = torch.empty([2, 5], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([2, 5])
        img = torch.zeros([2, 1])
        out = torch.empty([2, 5], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))
    # 定义测试函数 test_einsum
    def test_einsum(self):
        # 定义内部函数 check，用于检查生成的 JIT 代码中是否包含 'aten::einsum' 操作，并比较原始函数和 JIT 函数的输出结果
        def check(fn, jitted, *args):
            self.assertGraphContains(jitted.graph, kind='aten::einsum')
            self.assertEqual(fn(*args), jitted(*args))

        # 定义使用 torch.einsum 函数生成矩阵乘法的函数 equation_format
        def equation_format(x, y):
            return torch.einsum('i,j->ij', (x, y))

        # 定义使用 torch.einsum 函数生成矩阵乘法的函数 equation_format_varargs，使用变长参数形式传递 x 和 y
        def equation_format_varargs(x, y):
            return torch.einsum('i,j->ij', x, y)

        # 定义使用 torch.einsum 函数以子列表形式生成矩阵乘法的函数 sublist_format
        def sublist_format(x, y):
            return torch.einsum(x, [0], y, [1], [0, 1])

        # 创建 CPU 上的测试张量 x 和 y
        x = make_tensor((5,), dtype=torch.float32, device="cpu")
        y = make_tensor((10,), dtype=torch.float32, device="cpu")

        # 对每个函数进行测试：使用 JIT 脚本化和追踪的方式
        for fn in [equation_format, equation_format_varargs, sublist_format]:
            check(fn, torch.jit.script(fn), x, y)
            check(fn, torch.jit.trace(fn, (x, y)), x, y)

    # 标记为需要跳过的测试函数，条件是 TorchDynamo 报错的情况
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_python_ivalue(self):
        # 测试纯 Python 对象是否可以作为 IValue，并且检查 IValue 和 PyObject 之间的转换是否正确
        # 测试 numpy 对象
        py_array = np.arange(15)
        ret_py_obj = torch._C._ivalue_debug_python_object(py_array)
        self.assertEqual(py_array, ret_py_obj)

        # 测试函数对象
        ret_py_obj = torch._C._ivalue_debug_python_object(F.relu)
        self.assertEqual(F.relu, ret_py_obj)

        # 测试内存管理：确保 IValue 正确调用增减引用计数，避免悬挂指针或内存泄漏
        def test_func_scope_helper(inp):
            # 创建一个作用域并进行转换 -> ivalue -> pyobject
            # 此函数返回一个新的 pyobject，其引用计数 +1
            inp_refcount = sys.getrefcount(inp)
            ivalue_holder = torch._C._ivalue_debug_python_object(inp)
            self.assertEqual(inp_refcount + 1, sys.getrefcount(ivalue_holder))
            return ivalue_holder + 1

        # 准备测试输入值
        test_input = 2200
        before_count = sys.getrefcount(test_input)
        test_func_scope_helper(test_input)
        after_count = sys.getrefcount(test_input)

        # 测试结束后，检查 test_input 的引用计数是否与测试前相同，以避免悬挂指针或内存泄漏
        self.assertEqual(before_count, after_count)
    def test_decompose_addmm(self):
        # 定义函数 does_decompose，用于测试 torch.jit.script 下的 addmm 函数
        def does_decompose():
            # 使用 torch.jit.script 装饰器定义 addmm 函数，实现矩阵运算并返回结果
            @torch.jit.script
            def addmm(mat, mat1, mat2):
                # 调用 mat 的 addmm 方法，计算 a
                a = mat.addmm(mat1, mat2)
                # 调用 mat 的 addmm 方法，传入额外的 alpha 和 beta 参数，计算 b
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                # 返回 a 和 b 的和
                return a + b

            # 创建随机矩阵作为测试数据
            mat = torch.randn(2, 2)
            mat1 = torch.randn(2, 4)
            mat2 = torch.randn(4, 2)

            # 调用 addmm 函数获取参考输出
            out_ref = addmm(mat, mat1, mat2)
            # 运行自定义的 'decompose_ops' 优化 pass
            self.run_pass('decompose_ops', addmm.graph)
            # 再次调用 addmm 函数获取测试输出
            out_test = addmm(mat, mat1, mat2)
            # 断言参考输出与测试输出相等
            self.assertEqual(out_ref, out_test)
            # 使用 FileCheck 检查 addmm.graph 中是否不包含 "addmm" 字符串
            FileCheck().check_not("addmm").run(str(addmm.graph))

        # 定义函数 doesnt_decompose，测试不同参数形式下的 addmm 函数
        def doesnt_decompose():
            # 使用 torch.jit.script 装饰器定义 addmm 函数，接受 alpha 和 beta 参数
            @torch.jit.script
            def addmm(mat, mat1, mat2, alpha, beta):
                # 调用 mat 的 addmm 方法，使用指定的 alpha 和 beta 参数，计算 a
                a = mat.addmm(mat1, mat2, alpha=4.20, beta=2.0)
                # 调用 mat 的 addmm 方法，将 alpha 和 beta 强制转换为整数，计算 b
                b = mat.addmm(mat1, mat2, alpha=int(alpha), beta=int(beta))

                # 返回 a 和 b 的和
                return a + b

            # 获取原始的 addmm.graph 字符串表示
            orig = str(addmm.graph)
            # 运行自定义的 'decompose_ops' 优化 pass
            self.run_pass('decompose_ops', addmm.graph)
            # 断言优化后的 addmm.graph 与原始表示相同
            self.assertTrue(orig == str(addmm.graph))

        # 分别执行函数 does_decompose 和 doesnt_decompose 进行测试
        does_decompose()
        doesnt_decompose()

    @suppress_warnings
    def test_sparse_tensors(self):
        # 定义忽略警告的测试函数 test_sparse_tensors
        @torch.jit.ignore
        def get_sparse():
            # 返回一个稀疏的二维张量，大小为 (2, 3)，数据类型为 torch.float32
            return torch.sparse_coo_tensor((2, 3), dtype=torch.float32)

        # 使用 torch.jit.script 装饰器定义 test_is_sparse 函数，判断输入张量是否稀疏
        @torch.jit.script
        def test_is_sparse(input):
            # type: (Tensor) -> bool
            return input.is_sparse

        # 调用 test_is_sparse 函数，分别测试稀疏和密集张量的输出
        script_out_is_sparse = test_is_sparse(get_sparse())
        script_out_is_dense = test_is_sparse(torch.randn(2, 3))
        # 断言稀疏张量返回 True，密集张量返回 False
        self.assertEqual(script_out_is_sparse, True)
        self.assertEqual(script_out_is_dense, False)

        # 定义测试基本稀疏张量的函数 test_basic_sparse
        def test_basic_sparse(input):
            # 调用 get_sparse 函数获取稀疏张量，并与输入张量一起返回
            output = get_sparse()
            return output, input

        # 使用 self.checkScript 检查 test_basic_sparse 函数的脚本化版本
        self.checkScript(test_basic_sparse, (get_sparse(),))
        self.checkScript(test_basic_sparse, (torch.tensor([1]),))

        # 定义测试稀疏张量求和的函数 test_sparse_sum
        def test_sparse_sum(input):
            # 对输入的稀疏张量进行求和操作
            return torch.sparse.sum(input)

        # 使用 self.checkScript 检查 test_sparse_sum 函数的脚本化版本
        self.checkScript(test_sparse_sum, (get_sparse(),))

        # 定义测试稀疏张量矩阵乘法的函数 test_sparse_mm
        def test_sparse_mm(input1, input2):
            # 对两个稀疏张量进行矩阵乘法操作
            return torch.sparse.mm(input1, input2)

        # 使用 self.checkScript 检查 test_sparse_mm 函数的脚本化版本
        self.checkScript(test_sparse_mm, (get_sparse(), torch.randn(3, 4)))

        # 定义测试稀疏张量 addmm 操作的函数 test_sparse_addmm
        def test_sparse_addmm(input, input1, input2):
            # 对稀疏张量 input 和输入张量 input1、input2 进行 addmm 操作
            return torch.sparse.addmm(input, input1, input2)

        # 定义测试带有 alpha 和 beta 参数的稀疏张量 addmm 操作的函数 test_sparse_addmm_alpha_beta
        def test_sparse_addmm_alpha_beta(input, input1, input2):
            # 对稀疏张量 input 和输入张量 input1、input2 进行带有 alpha 和 beta 参数的 addmm 操作
            return torch.sparse.addmm(input, input1, input2, alpha=1.3, beta=1.5)

        # 使用 self.checkScript 分别检查两个稀疏张量 addmm 操作函数的脚本化版本
        self.checkScript(test_sparse_addmm, (torch.randn(2, 4), get_sparse(), torch.randn(3, 4)))
        self.checkScript(test_sparse_addmm_alpha_beta, (torch.randn(2, 4), get_sparse(), torch.randn(3, 4)))
    def test_sparse_csr_tensors(self):
        @torch.jit.ignore
        # 定义忽略装饰器，用于指示 JIT 编译时忽略此函数
        def get_sparse_csr():
            return torch.randn(3, 3).to_sparse_csr()

        @torch.jit.script
        # 定义脚本装饰器，用于 JIT 编译为 TorchScript
        def test_is_sparse_csr(input):
            # type: (Tensor) -> bool
            # 检查输入是否为稀疏 CSR 张量，并返回布尔值
            return input.is_sparse_csr

        # 测试稀疏 CSR 张量的 TorchScript 函数调用结果
        script_out_is_sparse_csr = test_is_sparse_csr(get_sparse_csr())
        # 测试非稀疏 CSR 张量的 TorchScript 函数调用结果
        script_out_is_dense_csr = test_is_sparse_csr(torch.randn(3, 3))

        # 断言稀疏 CSR 张量返回 True
        self.assertEqual(script_out_is_sparse_csr, True)
        # 断言非稀疏 CSR 张量返回 False
        self.assertEqual(script_out_is_dense_csr, False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    # 如果没有 CUDA，跳过此测试
    def test_device_not_equal(self):

        def compare_device(x: torch.device):
            # 比较设备 x 是否不等于 CUDA 设备 "cuda:0"
            return x != torch.device("cuda:0")

        def compare_two_device(x: torch.device, y: torch.device):
            # 比较两个设备 x 和 y 是否不相等
            return x != y

        # 使用 self.checkScript 检查 compare_device 函数的 TorchScript
        self.checkScript(compare_device, (torch.device("cuda:0"),))
        # 使用 self.checkScript 检查 compare_two_device 函数的 TorchScript
        self.checkScript(compare_two_device, (torch.device("cuda:0"), torch.device("cuda:1"), ))

    def test_constant_prop_simple(self):
        @torch.jit.script
        # 定义脚本装饰器，用于 JIT 编译为 TorchScript
        def constant_prop(input_int):
            # type: (int) -> int
            # 计算常量表达式 a = 2 * 3
            a = 2 * 3
            # 计算常量表达式 b = a + 2
            b = a + 2
            # 返回常量表达式结果 b - input_int
            return b - input_int

        # 调用 constant_prop 函数并获取参考输出
        out_ref = constant_prop(2)
        # 运行常量传播优化 pass，并获取测试输出
        self.run_pass('constant_propagation', constant_prop.graph)
        out_test = constant_prop(2)
        # 断言参考输出与测试输出相等
        self.assertEqual(out_ref, out_test)
        # 获取常量传播优化后的图形表达字符串
        graph_str = str(constant_prop.graph)
        # 断言图形表达字符串中不包含 "aten::add" 和 "aten::mul"
        self.assertTrue("aten::add" not in graph_str and "aten::mul" not in graph_str)
        # 获取常量节点的值并断言其为 8
        const = constant_prop.graph.findNode("prim::Constant").output().toIValue()
        self.assertEqual(const, 8)

    def test_constant_prop_nested(self):
        @torch.jit.script
        # 定义脚本装饰器，用于 JIT 编译为 TorchScript
        def constant_prop(a):
            # 定义常量 b = 2 + 1
            b = 2 + 1
            # 如果 a 小于 2，则 c = b + 2，否则 c = b - 2
            if bool(a < 2):
                c = b + 2
            else:
                c = b - 2
            # 返回结果 c
            return c
        # 调用 constant_prop 函数并获取参考输出
        out_ref = constant_prop(torch.tensor(2))
        # 运行常量传播优化 pass，并获取测试输出
        self.run_pass('constant_propagation', constant_prop.graph)
        out_test = constant_prop(torch.tensor(2))
        # 断言参考输出与测试输出相等
        self.assertEqual(out_ref, out_test)
        # 查找条件语句节点 "prim::If"
        if_node = constant_prop.graph.findNode("prim::If")
        # 遍历条件语句节点的每个块和节点，并断言节点类型为 "prim::Constant"
        for block in if_node.blocks():
            for node in block.nodes():
                self.assertTrue(node.kind() == "prim::Constant")

    def test_constant_prop_print(self):
        @torch.jit.script
        # 定义脚本装饰器，用于 JIT 编译为 TorchScript
        def constant_prop(input_tensor):
            # 计算常量表达式 a = 2 * 3
            a = 2 * 3
            # 打印常量 a 的值
            print(a)
            # 计算常量表达式 b = a + 2
            b = a + 2
            # 返回常量表达式结果 b + input_tensor
            return b + input_tensor

        # 运行常量传播优化 pass，并获取图形对象
        self.run_pass('constant_propagation', constant_prop.graph)
        graph = constant_prop.graph
        # 查找打印节点 "prim::Print"
        print_node = graph.findNode("prim::Print")
        # 断言打印节点的输入值为 6
        self.assertTrue(print_node.input().toIValue() == 6)
    def test_constant_prop_rand(self):
        # 使用 torch.jit.script 装饰器将函数 constant_prop 转换为 Torch 脚本
        @torch.jit.script
        def constant_prop():
            # 创建一个形状为 [3] 的随机张量 a
            a = torch.randn([3])
            # 将张量 a 的每个元素加上常数 2，得到张量 b
            b = a + 2
            return b

        # 运行常量传播优化，传递 constant_prop 函数的计算图
        self.run_pass('constant_propagation', constant_prop.graph)
        # 断言计算图中包含 "aten::randn" 操作
        self.assertTrue("aten::randn" in str(constant_prop.graph))

    def test_constant_prop_none(self):
        # 使用 torch.jit.script 装饰器将函数 typed_none 转换为 Torch 脚本
        @torch.jit.script
        def typed_none():
            # type: () -> Optional[int]
            # 返回一个类型为 Optional[int] 的 None 值
            return None

        # 使用 torch.jit.script 装饰器将函数 constant_prop 转换为 Torch 脚本
        @torch.jit.script
        def constant_prop():
            # 调用 typed_none 函数，得到变量 a
            a = typed_none()
            # 再次调用 typed_none 函数，得到变量 b
            b = typed_none()
            # 如果 a 和 b 均为 None，则将 a 赋值为 2，否则赋值为 1
            if (a is None and b is None):
                a = 2
            else:
                a = 1
            return a

        # 运行常量传播优化，传递 constant_prop 函数的计算图
        self.run_pass('constant_propagation', constant_prop.graph)
        # 使用 FileCheck 检查是否有 "prim::Constant" 操作
        FileCheck().check("prim::Constant").run(constant_prop.graph)

    def test_constant_prop_if_inline(self):
        # 使用 torch.jit.script 装饰器将函数 constant_prop 转换为 Torch 脚本
        @torch.jit.script
        def constant_prop():
            # 定义一个条件变量 cond，并赋值为 True
            cond = True
            # 初始化变量 a，并赋值为 1
            a = 1
            # 如果 cond 为 True，则将 a 重新赋值为 1 * 2，否则为 1 // 0（异常）
            if cond:
                a = 1 * 2
            else:
                a = 1 // 0
            return a

        # 测试常量传播优化，确保不会抛出 1 // 0 异常
        self.run_pass('constant_propagation', constant_prop.graph)

    def test_constant_prop_exception(self):
        # 定义一个函数 bad_index，接受一个参数 x（类型为 bool）
        # 如果 x 为 True，则返回列表 a 中索引为 4 的元素，否则返回 0
        def bad_index(x):
            # type: (bool)
            y = 0
            if x:
                a = [1, 2, 3]
                y = a[4]
            return y

        # 检查 bad_index 函数在常量传播优化时是否会出错
        self.checkScript(bad_index, (False,))

    def test_constant_prop_aliasing_type(self):
        # 使用 torch.jit.script 装饰器将函数 foo 转换为 Torch 脚本
        @torch.jit.script
        def foo():
            # 返回一个元组，包含列表 [1] 的长度和张量 [2] 的长度
            return len([1]), len(torch.tensor([2]))

        # 使用 FileCheck 检查计算图中是否包含 "aten::tensor" 和 "aten::len" 操作
        FileCheck().check_dag("aten::tensor").check_dag("aten::len").run(foo.graph)

        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn():
            # 如果条件表达式为真，则返回 1，否则返回 2
            if 1 == 1:
                return 1
            else:
                return 2

        # 使用 FileCheck 检查计算图中是否没有 "prim::If" 操作
        FileCheck().check_not("prim::If").run(fn.graph)

    def test_unchecked_cast(self):
        # 定义一个函数 test，接受一个参数 cond（类型为 bool）
        def test(cond):
            # type: (bool)
            # 创建一个包含元素 10 的张量 a
            a = torch.tensor([10])
            # 根据条件 cond，将变量 b 赋值为 None 或者张量 a
            if cond:
                b = None
            else:
                b = a
            # 如果 b 不为 None，则将 b 的第一个元素赋值为 5
            if b is not None:
                b[0] = 5
            # 返回张量 a 的整数类型表示
            return a.int()

        # 使用 checkScript 函数检查 test 函数在给定参数 True 和 False 时的行为
        self.checkScript(test, (True,))
        self.checkScript(test, (False,))
    # 测试常量传播中的条件常量情况
    def test_constant_prop_if_constant(self):
        @torch.jit.script
        def constant_prop(a, b):
            # 初始化常量变量
            c0 = 1
            c1 = 1
            c2 = 1
            # 第一个条件语句
            if bool(a):  # -> c0, c1
                # 第二个嵌套的条件语句
                if bool(b):  # -> c0
                    # 第三个嵌套的条件语句
                    if 1 == 1:  # -> c0
                        # 更新 c0
                        c0 = c0 + 1
                        # 如果条件不成立，以下语句不执行
                        if 1 == 2:
                            c1 = c1 + 1
                            c2 = c2 + 1
            else:  # -> c0, c1
                # 更新 c1
                c1 = c1 + 1

            # 无条件执行的语句块，即使条件为常量也被内联
            if 1 == 1:  # inlined
                # 动态更新 c0
                c0 = c0 + 1  # dynamic
                # 强制设置 c2 的值为 5
                c2 = c2 + 4  # set to 5
            # 返回计算结果
            return a + c0 + c1 + c2

        # 获取函数的计算图
        graph = constant_prop.graph
        # 执行常量传播优化
        self.run_pass('constant_propagation', graph)
        # 查找所有的 If 节点
        ifs = graph.findAllNodes("prim::If", recurse=False)
        # 第二个 If 被内联优化
        snd_if_inlined = len(ifs) == 1
        self.assertTrue(snd_if_inlined)
        # 检查第一个 If 节点的输出数量为 2
        first_if = ifs[0]
        self.assertTrue(first_if.outputsSize() == 2)
        # 查找第一个 If 节点内的第二个 If 节点
        second_if = first_if.findNode("prim::If", recurse=False)
        # 第二个 If 节点的输出数量为 1
        self.assertTrue(second_if.outputsSize() == 1)
        # 确保第二个 If 节点内不再有嵌套的 If 节点
        self.assertTrue(second_if.findNode("prim::If") is None)

    # 测试常量传播中的循环常量情况
    def test_constant_prop_loop_constant(self):
        @torch.jit.script
        def constant_prop(cond, iter):
            # type: (bool, int) -> int
            b = 0
            # 无限循环，打印语句会被保留
            while True:
                print("stays")
            # 固定次数循环，打印语句会被保留
            for _ in range(2):
                print("stays")
            # 循环次数由变量控制，打印语句会被保留
            for _ in range(iter):
                print("stays")
            # 条件循环，打印语句会被保留
            while cond:
                print("stays")
            # 永假循环，以下打印语句将被移除
            while False:
                print("removed")
            # 循环次数为0，以下打印语句将被移除
            for _i in range(0):
                print("removed")
            # 循环次数为负数，以下打印语句将被移除
            for _i in range(-4):
                print("removed")
            # 返回结果
            return b

        # 执行常量传播优化
        self.run_pass('constant_propagation', constant_prop.graph)
        # 规范化计算图
        graph = canonical(constant_prop.graph)
        # 确保被移除的打印语句数量为 0
        self.assertTrue(graph.count("removed") == 0)
        # 确保保留的打印语句数量为 1
        self.assertTrue(graph.count("stays") == 1)  # constant gets pooled
        # 确保总共的打印语句数量为 4
        self.assertTrue(graph.count("prim::Print") == 4)

    # 测试常量传播中移除输出
    def test_constant_prop_remove_output(self):
        @torch.jit.script
        def constant_prop(iter):
            # type: (int) -> None
            a = 1
            b = 1
            c = 1
            # 循环，部分条件不满足，但未被移除的打印语句会保留
            for i in range(iter):
                if 1 == 2:
                    a = 10
                if i == 5:
                    b = 2
                    c = 3
            # 打印最终结果
            print(a, b, c)

        # 获取函数的计算图
        graph = constant_prop.graph
        # 执行常量传播优化
        self.run_pass('constant_propagation', graph)
        # 确保循环节点的输出数量为 2
        self.assertTrue(graph.findNode("prim::Loop").outputsSize() == 2)
    def test_cuda_export_restore(self):
        # 定义一个继承自ScriptModule的子类Sub，包含一个参数weight，形状为(3, 4)
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))

            @torch.jit.script_method
            # 定义前向传播方法，对输入参数thing和参数weight进行加法运算
            def forward(self, thing):
                return self.weight + thing

        # 定义一个继承自ScriptModule的主类M，包含一个Sub类的实例mod
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.mod = Sub()

            @torch.jit.script_method
            # 定义前向传播方法，调用mod实例的forward方法
            def forward(self, v):
                return self.mod(v)

        # 创建M类的实例m
        m = M()
        # 将模型m移到CUDA设备上
        m.cuda()
        # 使用getExportImportCopy方法获得m的一个复制m2
        m2 = self.getExportImportCopy(m)
        # 将m2移到CUDA设备上
        m2.cuda()
        # 创建一个形状为(3, 4)的随机张量，并移到CUDA设备上
        input = torch.rand(3, 4).cuda()
        # 断言m和m2在输入input上的输出是否相等
        self.assertEqual(m(input), m2(input))

    @slowTest
    def test_export_batchnorm(self):
        # 针对不同的mode和clazz进行循环测试
        for mode in ['eval', 'train']:
            for clazz in [
                    torch.nn.BatchNorm1d(100),
                    torch.nn.BatchNorm1d(100, affine=False),
                    torch.nn.BatchNorm2d(100),
                    torch.nn.BatchNorm2d(100, affine=False)]:
                # 调用clazz的eval或train方法
                getattr(clazz, mode)()
                # 根据clazz类型选择相应形状的随机输入张量
                input = torch.randn(20, 100) if isinstance(clazz, torch.nn.BatchNorm1d) else \
                    torch.randn(20, 100, 35, 45)
                # 使用torch.jit.trace对clazz进行追踪
                traced = torch.jit.trace(clazz, (input,))
                # 使用getExportImportCopy方法获得traced的一个复制imported
                imported = self.getExportImportCopy(traced)
                # 再次创建相应形状的随机输入张量x
                x = torch.randn(20, 100) if isinstance(clazz, torch.nn.BatchNorm1d) else \
                    torch.randn(20, 100, 35, 45)
                # 断言traced和imported在输入x上的输出是否相等
                self.assertEqual(traced(x), imported(x))

    def test_export_rnn(self):
        # 针对不同的clazz进行循环测试
        for clazz in [nn.RNN(10, 20, 2), nn.GRU(10, 20, 2)]:
            # 定义一个RNNTest类，包含一个clazz类型的rnn属性
            class RNNTest(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.rnn = clazz

                # 定义前向传播方法，对输入x进行RNN运算
                def forward(self, x, lengths, h0):
                    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
                    out, h = self.rnn(packed, h0)
                    padded_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
                    return padded_outs

            # 创建RNNTest类的实例test
            test = RNNTest()

            # 使用torch.jit.trace对test进行追踪
            traced = torch.jit.trace(test, (torch.randn(5, 3, 10), torch.LongTensor([3, 2, 1]), torch.randn(2, 3, 20)))
            # 使用getExportImportCopy方法获得traced的一个复制imported
            imported = self.getExportImportCopy(traced)
            # 为了确保pad_packed的参数存储正常工作，传入一个具有不同最大序列长度的批次
            x, lengths, h0 = torch.randn(7, 4, 10), torch.LongTensor([7, 3, 2, 1]), torch.randn(2, 4, 20)
            # 断言traced和imported在输入x, lengths, h0上的输出是否相等
            self.assertEqual(traced(x, lengths, h0), imported(x, lengths, h0))
    def test_export_lstm(self):
        # 定义一个测试用的 LSTM 模型
        class LSTMTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 LSTM 层，输入维度为10，隐藏层维度为20，层数为2
                self.rnn = nn.LSTM(10, 20, 2)

            def forward(self, x, lengths, hiddens):
                h0, c0 = hiddens
                # 对输入数据进行填充序列的打包
                packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
                # 将打包的序列输入到 LSTM 中，同时传入初始隐藏状态 h0 和 c0
                out, (h, c) = self.rnn(packed, (h0, c0))
                # 将 LSTM 的输出进行填充序列的解包
                padded_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
                return padded_outs

        # 创建 LSTMTest 的实例
        test = LSTMTest()

        # 对 test 模型进行 TorchScript 脚本化
        traced = torch.jit.trace(test, (torch.randn(5, 3, 10),
                                        torch.LongTensor([3, 2, 1]),
                                        (torch.randn(2, 3, 20), torch.randn(2, 3, 20))))
        # 使用自定义方法 getExportImportCopy 处理脚本化后的模型，获取导入的副本
        imported = self.getExportImportCopy(traced)

        # 创建测试数据
        x, lengths, h0, c0 = \
            torch.randn(7, 3, 10), torch.LongTensor([7, 5, 2]), torch.randn(2, 3, 20), torch.randn(2, 3, 20)
        # 比较原始模型和导入模型在给定输入下的输出是否相等
        self.assertEqual(traced(x, lengths, (h0, c0)), imported(x, lengths, (h0, c0)))

    def test_unique_state_dict(self):
        # 定义一个包含共享参数的简单模型
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                shared_param = torch.nn.Parameter(torch.ones(1))
                # 注册两个共享参数 w1 和 w2
                self.register_parameter('w1', shared_param)
                self.register_parameter('w2', shared_param)

            def forward(self, input):
                # 在前向传播中使用共享参数
                return input + self.w1 + self.w2

        # 创建 MyModule 的实例
        model = MyModule()
        # 使用 torch.jit._unique_state_dict 函数获取模型的唯一状态字典，测试不保留变量
        unittest.TestCase.assertEqual(
            self, len(torch.jit._unique_state_dict(model, keep_vars=False)), 1)
        # 再次使用 torch.jit._unique_state_dict 函数获取模型的唯一状态字典，测试保留变量
        unittest.TestCase.assertEqual(
            self, len(torch.jit._unique_state_dict(model, keep_vars=True)), 1)

    def test_export_dropout(self):
        # 定义一个 Dropout 模型并将其设置为评估模式
        test = torch.nn.Dropout()
        test.eval()

        # 对 Dropout 模型进行 TorchScript 脚本化
        traced = torch.jit.trace(test, (torch.rand(3, 4),), check_trace=False)
        # 使用自定义方法 getExportImportCopy 处理脚本化后的模型，获取导入的副本
        imported = self.getExportImportCopy(traced)

        # 创建测试数据
        x = torch.randn(3, 4)
        # 比较原始模型和导入模型在给定输入下的输出是否相等
        self.assertEqual(traced(x), imported(x))
    def test_pretty_printer(self):
        @torch.jit.script
        def if_test(a, b):
            # FIXME: use 0 instead of a.
            # c = 0
            c = a  # 将变量 c 初始化为 a 的值
            if bool(a < b):  # 如果 a 小于 b，则执行以下代码块
                c = b  # 如果条件成立，将 c 设为 b 的值
            else:  # 如果条件不成立，则执行以下代码块
                c = a  # 将 c 设为 a 的值
            return c  # 返回变量 c 的值

        @torch.jit.script
        def if_one(a, b):
            c = b  # 将变量 c 初始化为 b 的值
            if bool(a < b):  # 如果 a 小于 b，则执行以下代码块
                c = a  # 如果条件成立，将 c 设为 a 的值
            return c  # 返回变量 c 的值

        @torch.jit.script
        def while_test(a, i):
            while bool(i < 3):  # 当 i 小于 3 时执行循环
                a *= a  # 将 a 的平方赋给 a
                i += 1  # i 自增 1
            return a  # 返回变量 a 的值

        @torch.jit.script
        def while_if_test(a, b):
            c = 0  # 初始化变量 c 为 0
            while bool(a < 10):  # 当 a 小于 10 时执行循环
                a = a + 1  # a 自增 1
                b = b + 1  # b 自增 1
                if bool(a > b):  # 如果 a 大于 b，则执行以下代码块
                    c = 2  # 将 c 设为 2
                else:  # 如果条件不成立，则执行以下代码块
                    c = 3  # 将 c 设为 3
            return a + 1 + c  # 返回 a + 1 + c 的值

        @torch.jit.script
        def loop_use_test(y):
            x = y + 1  # 将 y + 1 赋给 x
            z = x + 5  # 将 x + 5 赋给 z
            while bool(y < 8):  # 当 y 小于 8 时执行循环
                y += 1  # y 自增 1
                z = x  # 将 x 的值赋给 z
            return x, z  # 返回 x 和 z 的值

        @torch.jit.ignore
        def python_fn(x):
            return x + 10  # 返回 x + 10 的值

        @torch.jit.script
        def python_op_name_test(y):
            return python_fn(y)  # 调用 python_fn 函数并返回其结果

        @torch.jit.script
        def empty_int_list_test(y):
            x = torch.jit.annotate(List[int], [])  # 声明一个空的整数列表 x
            return x[0]  # 返回列表 x 的第一个元素（空列表会导致索引错误）

        @torch.jit.script
        def empty_float_list_test(y):
            return [1.0, 2.0, 3.0]  # 返回包含三个浮点数的列表

        @torch.jit.script
        def print_weird_test(y):
            print("hi\016")  # 打印带有八进制转义字符的字符串 "hi"

        self.assertExpected(if_test.code, "if_test")  # 断言 if_test 函数的代码与预期一致
        self.assertExpected(if_one.code, "if_one")  # 断言 if_one 函数的代码与预期一致
        self.assertExpected(while_test.code, "while_test")  # 断言 while_test 函数的代码与预期一致
        self.assertExpected(while_if_test.code, "while_if_test")  # 断言 while_if_test 函数的代码与预期一致
        self.assertExpected(loop_use_test.code, "loop_use_test")  # 断言 loop_use_test 函数的代码与预期一致
        self.assertExpected(python_op_name_test.code, "python_op_name_test")  # 断言 python_op_name_test 函数的代码与预期一致
        self.assertExpected(empty_int_list_test.code, "empty_int_list_test")  # 断言 empty_int_list_test 函数的代码与预期一致
        self.assertExpected(empty_float_list_test.code, "empty_float_list_test")  # 断言 empty_float_list_test 函数的代码与预期一致
        self.assertExpected(print_weird_test.code, "print_weird_test")  # 断言 print_weird_test 函数的代码与预期一致

    def test_cu_escaped_number(self):
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                print("hi\016")  # 打印带有八进制转义字符的字符串 "hi"
        ''')
        self.assertExpected(cu.foo.code)  # 断言 cu.foo 的代码与预期一致

    def test_import_method(self):
        with torch._jit_internal._disable_emit_hooks():  # 禁用 JIT 编译时的钩子函数
            class Foo(torch.jit.ScriptModule):
                @torch.jit.script_method
                def forward(self, x, y):
                    return 2 * x + y  # 返回 2 * x + y 的值

            foo = Foo()  # 创建 Foo 类的实例
            buffer = io.BytesIO()  # 创建一个字节流缓冲区
            torch.jit.save(foo, buffer)  # 将 foo 对象保存到缓冲区中

            buffer.seek(0)  # 将缓冲区指针移动到起始位置
            foo_loaded = torch.jit.load(buffer)  # 从缓冲区加载对象 foo
            self.assertExpected(foo_loaded.forward.code)  # 断言 foo_loaded.forward 的代码与预期一致

    @unittest.skip("temporarily disable the test for fwd compatibility")  # 跳过当前单元测试，用于前向兼容性
    def test_non_ascii_string(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.a = "Over \u0e55\u0e57 57"  # 设置字符串属性，包含非 ASCII 字符

            @torch.jit.script_method
            def forward(self, x, y):
                return self.a + "hi\xA1"  # 返回字符串属性与另一个字符串的连接

        foo = Foo()  # 创建 Foo 类的实例
        buffer = io.BytesIO()  # 创建字节流对象
        torch.jit.save(foo, buffer)  # 将 Foo 实例保存到字节流中

        buffer.seek(0)  # 将字节流指针位置移到开头
        foo_loaded = torch.jit.load(buffer)  # 从字节流中加载 Foo 实例
        self.assertExpected(foo_loaded.forward.code)  # 断言加载后的实例的 forward 方法的代码与预期一致

    def test_function_default_values(self):
        outer_var = torch.tensor(20)  # 定义外部变量 outer_var
        outer_var2 = torch.tensor(30)  # 定义外部变量 outer_var2
        a = torch.tensor(0.5)  # 定义张量 a
        b = torch.tensor(10)  # 定义张量 b

        @torch.jit.script
        def simple_fn(x, a=a, b=b, c=outer_var + outer_var2):
            return x + a + b + c  # 返回 x、a、b 和 c 的总和

        self.assertEqual(
            simple_fn(torch.ones(1)),
            torch.ones(1) + 0.5 + 10 + (20 + 30))  # 断言 simple_fn 对于 torch.ones(1) 的输出值

        self.assertEqual(
            simple_fn(torch.ones(1), torch.tensor(1), torch.tensor(3), torch.tensor(4)),
            torch.ones(1) + 1 + 3 + 4)  # 断言 simple_fn 对给定参数的输出值

        outer_c = torch.tensor(9)  # 定义外部变量 outer_c
        outer_flag = torch.tensor(False)  # 定义外部变量 outer_flag

        @torch.jit.script
        def bool_fn(x, a=outer_c, flag=outer_flag):
            if bool(flag):
                result = x
            else:
                result = x + a
            return result  # 根据条件返回 x 或 x + a 的结果

        self.assertEqual(bool_fn(torch.ones(1)), torch.ones(1) + 9)  # 断言 bool_fn 对于 torch.ones(1) 的输出值
        self.assertEqual(
            bool_fn(torch.ones(1), torch.tensor(1), torch.tensor(True)),
            torch.ones(1))  # 断言 bool_fn 对给定参数的输出值

        @torch.jit.script
        def none_fn(x=None):
            # type: (Optional[int]) -> Optional[int]
            return x  # 返回可选整数类型的参数 x

        self.assertEqual(none_fn(), None)  # 断言 none_fn() 返回 None
        self.assertEqual(none_fn(1), 1)  # 断言 none_fn(1) 返回 1

        @torch.jit.script
        def hints(x, a=0.5, b=10):
            # type: (Tensor, float, int) -> Tensor
            return x + a + b  # 返回张量 x、浮点数 a 和整数 b 的总和

        self.assertEqual(hints(torch.ones(1)), torch.ones(1) + 0.5 + 10)  # 断言 hints 对于 torch.ones(1) 的输出值

        with self.assertRaisesRegex(RuntimeError, "Expected a default value"):

            @torch.jit.script
            def hints_bad_types(x, a=10, b=0.5):  # noqa: T484
                # type: (Tensor, float, int) -> Tensor
                return x + a + b  # 返回张量 x、浮点数 a 和整数 b 的总和

        with self.assertRaisesRegex(RuntimeError, "Expected a default value"):
            @torch.jit.script
            def bad_no_optional(x=None):
                # type: (Dict[str, int]) -> Dict[str, int]
                return x  # 返回字典类型的参数 x

    def test_module_default_values(self):
        four = torch.tensor(4)  # 定义张量 four

        class Test(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, other=four):
                return input + other  # 返回 input 与 other 的和

        t = Test()  # 创建 Test 类的实例
        self.assertEqual(t(torch.ones(1)), torch.ones(1) + 4)  # 断言 t 对于 torch.ones(1) 的输出值
    def test_mutable_default_values(self):
        # 检查可变默认参数是否会引发异常
        with self.assertRaisesRegex(Exception, "Mutable default parameters"):
            # 使用 torch.jit.script 装饰器定义函数 foo，其中 x 是一个元组，第二个元素是空列表作为默认值
            @torch.jit.script
            def foo(x=(1, [])):
                # type: (Tuple[int, List[Tensor]])
                return x

        class Test(torch.nn.Module):
            def forward(self, input=[]):  # noqa: B006
                # 返回输入参数 input，如果未提供则为一个空列表作为默认值
                return input

        # 检查 torch.jit.script 是否能正确处理带有可变默认参数的模块
        with self.assertRaisesRegex(Exception, "Mutable default parameters"):
            torch.jit.script(Test())

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_warnings(self):
        import warnings

        def fn(x):
            # 如果 x < 2，则发出警告信息 "x is less than 2"
            if bool(x < 2):
                warnings.warn("x is less than 2")
            return x

        class M(torch.nn.Module):
            def forward(self, x):
                # 如果 x < 2，则发出警告信息 "x is less than 2"
                if bool(x < 2):
                    warnings.warn("x is less than 2")
                return x

        # 对 M 类和 fn 函数进行 torch.jit.script 转换
        scripted_mod = torch.jit.script(M())
        scripted_fn = torch.jit.script(fn)

        # 捕获 fn 调用时产生的警告
        with warnings.catch_warnings(record=True) as warns:
            fn(torch.ones(1))

        # 捕获使用 torch.jit.script 转换后 fn 调用时产生的警告
        with warnings.catch_warnings(record=True) as script_warns:
            scripted_fn(torch.ones(1))

        # 捕获使用 torch.jit.script 转换后 M 类实例调用时产生的警告
        with warnings.catch_warnings(record=True) as script_mod_warns:
            scripted_mod(torch.ones(1))

        # 断言两种方式下的警告信息相同
        self.assertEqual(str(warns[0]), str(script_warns[0]))
        # 断言 script_mod_warns 中仅有一条警告信息
        self.assertEqual(len(script_mod_warns), 1)
        # 断言两种方式下的警告信息文本相同
        self.assertEqual(str(warns[0].message), str(script_mod_warns[0].message))

    def test_no_erroneous_warnings(self):
        import warnings

        def fn(x):
            # 如果 x > 0，则发出警告信息 'This should NOT be printed'，并且增加 x 的值
            if bool(x > 0):
                warnings.warn('This should NOT be printed')
                x += 1
            return x

        # 捕获 fn 调用时产生的所有警告
        with warnings.catch_warnings(record=True) as warns:
            # 对 fn 函数进行 torch.jit.script 转换
            fn_script = torch.jit.script(fn)
            fn_script(torch.tensor(0))
        # 确保没有警告被触发
        warns = [str(w.message) for w in warns]
        self.assertEqual(len(warns), 0)

    @unittest.skipIf(True, "TODO: re-enable with https://github.com/pytorch/pytorch/pull/29339")
    def test_torch_load_error(self):
        class J(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                # 在输入上加上 100 并返回
                return input + 100

        j = J()
        with TemporaryFileName() as fname:
            # 保存 J 类的实例到临时文件中
            j.save(fname)
            # 使用 torch.load 加载临时文件，预期会引发 RuntimeError 异常并提示 "is a zip"
            with self.assertRaisesRegex(RuntimeError, "is a zip"):
                torch.load(fname)

    def test_torch_load_zipfile_check(self):
        @torch.jit.script
        def fn(x):
            # 返回输入参数 x 加上 10
            return x + 10

        with TemporaryFileName() as fname:
            # 将 fn 函数保存到临时文件中
            fn.save(fname)
            with open(fname, 'rb') as f:
                # 检查文件 f 是否是有效的 Zip 文件
                self.assertTrue(torch.serialization._is_zipfile(f))
    # 定义测试方法，验证 LSTMCellS 是否能被 JIT 脚本化
    def test_python_bindings(self):
        # 将 LSTMCellS 脚本化
        lstm_cell = torch.jit.script(LSTMCellS)

        # 定义 LSTM 函数，接受输入 x，初始隐藏状态 hx 和细胞状态 cx，以及权重和偏置
        def lstm(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            # 遍历输入张量 x 的每个时间步
            for i in range(x.size(0)):
                # 使用脚本化的 lstm_cell 进行 LSTM 计算，更新隐藏状态 hx 和细胞状态 cx
                hx, cx = lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
            # 返回最终的隐藏状态 hx
            return hx

        # 将 lstm 函数脚本化
        slstm = torch.jit.script(lstm)

        # 获取用于测试的 LSTM 输入
        inputs = get_lstm_inputs('cpu', training=True, seq_length=10)

        # 对脚本化的 slstm 模型进行前向传播，计算损失并反向传播
        slstm(*inputs).sum().backward()

        # 获取 slstm 的前向计算图
        global fw_graph
        fw_graph = slstm.graph_for(*inputs)

        # 获取计算图中的节点列表
        nodes = list(fw_graph.nodes())

        # 标志变量，用于检测是否已测试过块
        tested_blocks = False

        # 遍历计算图中的节点
        for node in nodes:
            # 检查节点的输出
            for output in node.outputs():
                self.assertTrue(hasattr(output, 'type'))
                self.assertTrue(output.type() is not None)

            # 检查节点的输入
            for input in node.inputs():
                self.assertTrue(hasattr(input, 'type'))
                self.assertTrue(input.type() is not None)

            # 遍历节点的块
            for block in node.blocks():
                # 标记已经测试过块
                tested_blocks = True

                # 检查块的输入和输出
                self.assertTrue(hasattr(block, 'inputs'))
                self.assertTrue(hasattr(block, 'outputs'))

                # 检查块中输出节点的类型信息
                for output in block.outputs():
                    self.assertTrue(hasattr(output, 'type'))
                    self.assertTrue(output.type() is not None)

                # 检查块中输入节点的类型信息
                for input in block.inputs():
                    self.assertTrue(hasattr(input, 'type'))
                    self.assertTrue(input.type() is not None)

                # 检查块的返回节点
                self.assertTrue(hasattr(block, 'returnNode'))
                self.assertTrue(type(block.returnNode()) == torch._C.Node)

                # 检查块的参数节点
                self.assertTrue(hasattr(block, 'paramNode'))
                self.assertTrue(type(block.paramNode()) == torch._C.Node)

        # 最终确认已经测试过块
        self.assertTrue(tested_blocks)

    # 定义测试方法，验证模型的导出操作名
    def test_export_opnames(self):
        # 定义 Foo 类，继承自 torch.jit.ScriptModule
        class Foo(torch.jit.ScriptModule):
            # 定义 one 方法，接受两个张量 x 和 y，返回它们的和
            def one(self, x, y):
                # type 注释：指定参数和返回类型为 Tensor
                # 返回 x 和 y 的和
                return x + y

            # 定义 two 方法，接受一个张量 x，返回它的两倍
            def two(self, x):
                # type 注释：指定参数和返回类型为 Tensor
                # 返回 x 的两倍
                return 2 * x

            # 定义前向传播方法，接受一个张量 x，调用 two 和 one 方法，返回结果
            @torch.jit.script_method
            def forward(self, x):
                # type 注释：指定参数和返回类型为 Tensor
                # 返回 one 方法的结果
                return self.one(self.two(x), x)

        # 定义 Bar 类，继承自 torch.jit.ScriptModule
        class Bar(torch.jit.ScriptModule):
            # 初始化方法，创建 Foo 实例并赋值给 self.sub
            def __init__(self):
                super().__init__()
                self.sub = Foo()

            # 定义前向传播方法，接受一个张量 x，调用 self.sub 的 forward 方法，返回结果
            @torch.jit.script_method
            def forward(self, x):
                # type 注释：指定参数和返回类型为 Tensor
                # 调用 self.sub 的 forward 方法
                return self.sub.forward(x)

        # 创建 Bar 的实例
        bar = Bar()

        # 调用 torch.jit.export_opnames 获取 bar 模型的导出操作名
        ops = torch.jit.export_opnames(bar)

        # 期望的操作名列表
        expected = ['aten::add.Tensor', 'aten::mul.Scalar']

        # 断言期望的操作名集合是否是 ops 的子集
        self.assertTrue(set(expected).issubset(set(ops)))
    # 定义一个测试方法，用于验证在关闭 PYTORCH_JIT 环境下的行为
    def test_pytorch_jit_env_off(self):
        import subprocess  # 导入子进程模块
        env = os.environ.copy()  # 复制当前环境变量
        env['PYTORCH_JIT'] = '0'  # 设置 PYTORCH_JIT 环境变量为 '0'
        try:
            # 执行一个子进程来检查是否可以导入 torch 模块
            subprocess.check_output([sys.executable, '-c', 'import torch'], env=env)
        except subprocess.CalledProcessError as e:
            # 如果导入失败，抛出运行时异常
            raise RuntimeError("Could not 'import torch' with PYTORCH_JIT=0") from e

    # 定义一个测试方法，用于打印 torch.ops 模块
    def test_print_op_module(self):
        # 解决问题 #19351：Python 2 和 Python 3 路径不同的问题
        # Python 2 返回 '<module 'torch.ops' (built-in)>'
        # Python 3 使用 __file__ 并返回 '<module 'torch.ops' from '/scratch/ailzhang/pytorch/torch/_ops.py'>'
        s = str(torch.ops)
        self.assertRegex(s, r'ops')

    # 定义一个测试方法，用于打印 torch.classes 模块
    def test_print_classes_module(self):
        s = str(torch.classes)
        self.assertRegex(s, r'classes')

    # 定义一个测试方法，用于打印 torch._ops.ops.quantized 模块和 torch._ops.ops.atan 模块
    def test_print_torch_ops_modules(self):
        s = str(torch._ops.ops.quantized)
        self.assertRegex(s, r'torch.ops')
        s = str(torch._ops.ops.atan)
        self.assertRegex(s, r'torch.ops')

    # 定义一个测试方法，测试隐藏源代码范围的上下文管理器
    def test_hide_source_ranges_context_manager(self):
        @torch.jit.script
        def foo(x):
            return torch.add(x, x)

        graph = foo.graph  # 获取函数 foo 的图形表示
        source_range_regex = "# .*\\.py"  # 源代码范围的正则表达式
        self.assertRegex(graph.__repr__(), source_range_regex)  # 断言图形的字符串表示符合源代码范围正则表达式
        with torch.jit._hide_source_ranges():  # 使用隐藏源代码范围的上下文管理器
            self.assertNotRegex(graph.__repr__(), source_range_regex)  # 断言图形的字符串表示不再包含源代码范围
            self.assertRegex(graph.str(print_source_ranges=True), source_range_regex)  # 断言打印源代码范围后的字符串表示符合源代码范围正则表达式
        self.assertRegex(graph.__repr__(), source_range_regex)  # 断言图形的字符串表示仍然符合源代码范围正则表达式
class TestFrontend(JitTestCase):

    def test_instancing_error(self):
        # 定义一个被忽略的 Torch JIT 脚本类
        @torch.jit.ignore
        class MyScriptClass:
            # 不可脚本化的方法，返回一个字符串与整数相加，会引发错误
            def unscriptable(self):
                return "a" + 200

        # 定义一个测试模块，其 forward 方法返回 MyScriptClass 实例
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return MyScriptClass()

        # 使用 torch.jit.script 尝试将 TestModule 脚本化，期望引发 FrontendError 异常
        with self.assertRaises(torch.jit.frontend.FrontendError) as cm:
            torch.jit.script(TestModule())

        # 创建 FileCheck 对象用于检查异常信息
        checker = FileCheck()
        # 检查异常信息中包含 "Cannot instantiate class"
        checker.check("Cannot instantiate class")
        # 检查异常信息中包含 "def forward"
        checker.check("def forward")
        # 运行 FileCheck 对象来验证异常信息是否符合预期
        checker.run(str(cm.exception))

    def test_dictionary_as_example_inputs_for_jit_trace(self):
        # 定义一个测试模块，接收多个输入参数并返回它们的和
        class TestModule_v1(torch.nn.Module):
            def forward(self, key2=None, key3=None, key4=None, key5=None, key1=None, key6=None):
                return key1 + key2 + key3

        # 定义另一个测试模块，接收两个参数 x 和 y，并返回它们的和
        class TestModule_v2(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # 定义一个普通函数，接收两个参数并返回它们的和
        def test_func(x, y):
            return x + y

        # 创建 TestModule_v1 和 TestModule_v2 的实例
        model_1 = TestModule_v1()
        model_2 = TestModule_v2()

        # 创建三个张量，作为示例输入
        value1 = torch.ones(1)
        value2 = torch.ones(1)
        value3 = torch.ones(1)

        # 创建示例输入字典，用于 JIT 跟踪
        example_input_dict = {'key1': value1, 'key2': value2, 'key3': value3}
        # 创建另一个示例输入字典，用于 JIT 跟踪函数
        example_input_dict_func = {'x': value1, 'y': value2}

        # 使用 torch.jit.trace 将 model_1 脚本化，使用示例输入字典
        traced_model_1 = torch.jit.trace(model_1, example_kwarg_inputs=example_input_dict, strict=False)
        # 使用 torch.jit.trace_module 将 model_1 脚本化，传入 forward 方法的示例输入字典
        traced_model_1_m = torch.jit.trace_module(
            model_1, {'forward': example_input_dict}, example_inputs_is_kwarg=True, strict=False)
        # 使用 torch.jit.trace 将 model_2 脚本化，传入指定示例输入字典
        traced_model_2 = torch.jit.trace(model_2, example_kwarg_inputs={'x': torch.rand([2]), 'y': torch.rand([2])})
        # 使用 torch.jit.trace 将 test_func 脚本化，传入示例输入字典
        traced_func = torch.jit.trace(test_func, example_kwarg_inputs=example_input_dict_func, strict=False)

        # 调用脚本化的模型和函数，获取结果
        res_1 = traced_model_1(**example_input_dict)
        res_1_m = traced_model_1_m(**example_input_dict)
        res_func = traced_func(**example_input_dict_func)

        # 断言脚本化模型和函数的结果是否符合预期
        self.assertEqual(res_1, 3 * torch.ones(1))
        self.assertEqual(res_1_m, 3 * torch.ones(1))
        self.assertEqual(res_func, 2 * torch.ones(1))

        # 测试错误情况：缺少 'x' 参数时应引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"forward\(\) is missing value for argument 'x'."):
            res_2 = traced_model_2(**{'z': torch.rand([2]), 'y': torch.rand([2])})  # noqa: PIE804

        # 测试错误情况：缺少 'y' 参数时应引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"forward\(\) is missing value for argument 'y'."):
            res_2 = traced_model_2(**{'x': torch.rand([2]), 'z': torch.rand([2])})  # noqa: PIE804


class TestScript(JitTestCase):

    # 测试重复对函数应用 torch.jit.script 的情况
    def test_repeated_script_on_function(self):
        # 定义一个函数 fn，并重复两次应用 torch.jit.script
        @torch.jit.script
        @torch.jit.script
        def fn(x):
            return x

        # 再次应用 torch.jit.script
        torch.jit.script(torch.jit.script(fn))
    def test_pretty_print_function(self):
        # 定义一个 Torch 脚本函数 foo，用于对输入张量进行插值操作
        @torch.jit.script
        def foo(x):
            return torch.nn.functional.interpolate(x)

        # 使用 FileCheck 类检查 foo 函数的生成的代码中是否包含 "interpolate"
        FileCheck().check("interpolate").run(foo.code)

    def test_inlined_graph(self):
        """
        Check that the `inlined_graph` property correctly returns an inlined
        graph, both through function calls and method calls.
        """
        # 定义一个 Torch 脚本函数 foo，将输入张量 x 自身相加
        @torch.jit.script
        def foo(x):
            return torch.add(x, x)

        # 定义一个内嵌的 Torch 模块 MyNestedMod，包含一个 forward 方法实现张量 x 减去自身
        class MyNestedMod(torch.nn.Module):
            def forward(self, x):
                return torch.sub(x, x)

        # 定义一个 Torch 模块 MyMod，包含一个内嵌模块 MyNestedMod，并在 forward 方法中调用这些函数
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = MyNestedMod()

            def forward(self, x):
                x = self.nested(x)  # 调用 MyNestedMod 中的 sub 方法
                x = foo(x)  # 调用 foo 函数中的 add 方法
                return torch.mul(x, x)  # 返回 x 与自身相乘的结果

        # 将 MyMod 模块转换为 Torch 脚本，命名为 m
        m = torch.jit.script(MyMod())
        # 使用 FileCheck 类检查 inlined_graph 中是否包含 "aten::sub", "aten::add" 和 "aten::mul"
        FileCheck().check("aten::sub") \
                   .check("aten::add") \
                   .check("aten::mul") \
                   .run(m.inlined_graph)

    def test_static_method_on_module(self):
        """
        Check that the `@staticmethod` annotation on a function on a module works.
        """
        # 定义一个 Torch 模块 MyCell，包含一个静态方法 do_it，实现张量 x 和 h 的双曲正切和
        class MyCell(torch.nn.Module):
            @staticmethod
            def do_it(x, h):
                new_h = torch.tanh(x + h)
                return new_h, new_h

            def forward(self, x, h):
                return self.do_it(x, h)  # 调用静态方法 do_it

        # 将 MyCell 模块转换为 Torch 脚本，命名为 my_cell
        my_cell = torch.jit.script(MyCell())
        x = torch.rand(3, 4)
        h = torch.rand(3, 4)
        # 对比 Torch 脚本转换后的 my_cell 和直接调用 MyCell.do_it 的结果
        jitted_cell = my_cell(x, h)
        non_jitted_cell = MyCell().do_it(x, h)

        # 断言 Torch 脚本转换后的结果与直接调用的结果相等
        self.assertEqual(jitted_cell, non_jitted_cell)

    def test_code_with_constants(self):
        """
        Check that the `code_with_constants` property correctly returns graph CONSTANTS in the
        CONSTANTS.cN format used in the output of the `code` property.
        """
        # 定义一个 Torch 脚本函数 foo，默认参数为 torch.ones(1)
        @torch.jit.script
        def foo(x=torch.ones(1)):
            return x

        # 定义一个 Torch 模块 Moddy，包含一个 forward 方法调用 foo 函数
        class Moddy(torch.nn.Module):
            def forward(self, x):
                return foo()

        # 将 Moddy 模块转换为 Torch 脚本，命名为 m
        m = torch.jit.script(Moddy())
        # 获取 m 的代码和常量值
        src, CONSTANTS = m.code_with_constants

        # 断言 CONSTANTS.c0 的值等于 torch.ones(1)
        self.assertEqual(CONSTANTS.c0, torch.ones(1))
        # 断言 m 的代码等于源代码
        self.assertEqual(src, m.code)

    def test_code_with_constants_restore(self):
        """
        Check that the `code_with_constants` property correctly works on restoration after save() + load()
        """
        # 定义一个 Torch 脚本函数 foo，默认参数为 torch.ones(1)
        @torch.jit.script
        def foo(x=torch.ones(1)):
            return x

        # 定义一个 Torch 模块 Moddy，包含一个 forward 方法调用 foo 函数
        class Moddy(torch.nn.Module):
            def forward(self, x):
                return foo()

        # 将 Moddy 模块转换为 Torch 脚本，命名为 m
        m = torch.jit.script(Moddy())
        # 获取 m 的代码和常量值
        src, CONSTANTS = m.code_with_constants
        # 通过保存和加载操作获得的模块 eic
        eic = self.getExportImportCopy(m)

        # 获取 eic 的代码和常量值
        src_eic, CONSTANTS_eic = eic.code_with_constants

        # 断言 m 和 eic 的代码相等
        self.assertEqual(src, src_eic)
        # 断言 CONSTANTS.c0 和 CONSTANTS_eic.c0 的值相等
        self.assertEqual(CONSTANTS.c0, CONSTANTS_eic.c0)
    # 定义单行函数测试方法，使用 `fn` 函数并禁用 E704 错误提示
    def test_oneline_func(self):
        def fn(x): return x  # noqa: E704

        # 调用 `checkScript` 方法，传入参数为包含全 1 的 2x2 的 Torch 张量
        self.checkScript(fn, (torch.ones(2, 2), ))

    # 定义请求退出测试方法
    def test_request_bailout(self):
        # 启用用于性能分析测试的性能模式
        with enable_profiling_mode_for_profiling_tests():

            # 定义循环函数 `fct_loop`，接受参数 `x`
            def fct_loop(x):
                # 执行三次迭代
                for i in range(3):
                    # 在维度 0 上连接张量 `x` 的副本
                    x = torch.cat((x, x), 0)
                # 返回扩展后的张量 `x`
                return x

            # 创建全为 1 的 2x3x4 的浮点型 Torch 张量 `x`
            x = torch.ones(2, 3, 4, dtype=torch.float32)
            # 计算预期结果
            expected = fct_loop(x)
            # 对 `fct_loop` 函数进行 Torch JIT 脚本化
            jitted = torch.jit.script(fct_loop)
            # 进行性能分析
            jitted(x)
            # 进行优化
            jitted(x)
            # 获取调试状态
            dstate = jitted.get_debug_state()
            # 获取执行计划
            eplan = get_execution_plan(dstate)
            # 获取退化请求次数
            num_bailouts = eplan.code.num_bailouts()

            # 遍历所有退化请求
            for i in range(0, num_bailouts):
                # 请求执行计划中的指定退化请求 `i`
                eplan.code.request_bailout(i)
                # 断言脚本化函数 `jitted` 对 `x` 的执行结果与预期结果 `expected` 相等
                self.assertEqual(jitted(x), expected)

    # 标记测试跳过，注释指出退化功能正在被弃用
    @unittest.skip("bailouts are being deprecated")
    def test_dominated_bailout(self):
        with enable_profiling_mode_for_profiling_tests():
            # 在性能分析测试中启用性能分析模式

            # 定义一个 Torch 脚本函数 foo，用于动态地 JIT 编译
            @torch.jit.script
            def foo(x):
                # 获取张量 x 的维度
                dim = x.dim()
                # 如果维度为 0，则将 x 转换为整数类型
                if dim == 0:
                    y = int(x)
                else:
                    # 否则取最后一个维度的大小
                    y = x.size()[dim - 1]
                return y

            # 创建一个全零张量
            x = torch.zeros(2)
            # 调用 foo 函数并断言结果为 2
            self.assertEqual(foo(x), 2)
            # 再次调用 foo 函数并断言结果为 2
            self.assertEqual(foo(x), 2)
            # 获取最近一次 JIT 优化后的图形表示
            g = torch.jit.last_executed_optimized_graph()
            # 将图形表示转换为字符串
            g_s = str(g)
            # 截取到第一个 "return" 之前的部分
            g_s = g_s[0:g_s.find("return")]
            # 使用 FileCheck 工具验证出现了一次 "prim::BailOut["
            FileCheck().check_count("prim::BailOut[", 1, exactly=True).run(g_s)

            # 定义另一个 Torch 脚本函数 foo，用于动态地 JIT 编译
            @torch.jit.script
            def foo(x):
                # 获取张量 x 的维度
                dim = x.dim()
                # 将 x 中所有元素加上 3
                x.add_(3)
                # 如果维度为 0，则返回 0
                if dim == 0:
                    return 0
                else:
                    # 否则返回最后一个维度的大小
                    return x.size()[dim - 1]

            # 创建一个全零张量
            x = torch.zeros(2)
            # 调用 foo 函数并断言结果为 2
            self.assertEqual(foo(x), 2)
            # 再次调用 foo 函数并断言结果为 2
            self.assertEqual(foo(x), 2)
            # 获取最近一次 JIT 优化后的图形表示
            g = torch.jit.last_executed_optimized_graph()
            # 使用 FileCheck 工具验证出现了 "prim::BailOut[" 和 "aten::add_"，并且在 "return" 之后出现了 "prim::BailOut["
            FileCheck().check("prim::BailOut[").check("aten::add_").check_next("prim::BailOut[").check("return").run(g)

            # 在 Torch 梯度计算环境下启用梯度
            with torch.enable_grad():
                # 定义一个忽略 Torch JIT 编译的函数 disable_grad
                @torch.jit.ignore
                def disable_grad():
                    torch.set_grad_enabled(False)

                # 定义一个忽略 Torch JIT 编译的函数 enable_grad
                @torch.jit.ignore
                def enable_grad():
                    torch.set_grad_enabled(True)

                # 定义一个 Torch 脚本函数 foo，用于动态地 JIT 编译
                @torch.jit.script
                def foo(x):
                    # 将 x 中所有元素加上 1
                    x = x + 1
                    # 获取张量 x 的维度
                    dim = x.dim()
                    # 禁用梯度计算
                    disable_grad()
                    # 如果维度为 0，则将 x 转换为整数类型
                    if dim == 0:
                        y = int(x)
                    else:
                        # 否则取最后一个维度的大小
                        y = x.size()[dim - 1]
                    # 启用梯度计算
                    enable_grad()
                    return y

                # 创建一个全零张量，并标记为需要梯度计算
                x = torch.zeros(2, requires_grad=True)
                # 调用 foo 函数并断言结果为 2
                self.assertEqual(foo(x), 2)
                # 再次调用 foo 函数并断言结果为 2
                self.assertEqual(foo(x), 2)
                # 获取最近一次 JIT 优化后的图形表示
                g = torch.jit.last_executed_optimized_graph()
                # 使用 FileCheck 工具验证在 disable_grad 调用后仍然有 "prim::BailOut[" 出现
                FileCheck().check("disable_grad").check("BailOut[").check("BailoutTemplate").run(g)

    @skipIfTorchDynamo("Torchdynamo cannot correctly handle profiler.profile calls")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "skip if profiling isn't enabled")
    def test_loop_liveness(self):
        # 用于测试循环中变量的活跃性
        with enable_profiling_mode_for_profiling_tests():
            # 启用性能分析模式以进行性能测试
            @torch.jit.script
            def f(i):
                # type: (int) -> Tensor
                # 定义一个空列表
                l = []
                # 循环遍历固定的列表
                for n in [2, 1]:
                    # 向列表中添加一个全零张量
                    l.append(torch.zeros(n, i))

                # 返回列表中的第一个张量
                return l[0]

            # 调用函数f并传入参数2
            f(2)
            # 调用函数f并传入参数1
            f(1)

    def test_bailout_loop_carried_deps_name_clash(self):
        # 用于测试循环中的名称冲突和依赖关系
        with enable_profiling_mode_for_profiling_tests():
            # 启用性能分析模式以进行性能测试
            NUM_ITERATIONS = 10

            @torch.jit.script
            def fct_loop(z, size):
                # type: (int, int) -> Tuple[Tensor, List[int]]
                # 声明一个计数器列表
                counters = torch.jit.annotate(List[int], [])
                # 初始化一个变量j
                j = 0
                # 创建一个全一张量y
                y = torch.ones(2)
                # 循环执行指定次数
                for i in range(size):
                    # 将当前i加上j的值添加到计数器列表中
                    counters.append(i + j)
                    # 将y与一个全一张量拼接
                    y = torch.cat((y, torch.ones(z)), 0)
                    # 更新j的值
                    j = j + 1
                # 返回拼接后的张量y和计数器列表
                return y, counters

            # 准备多个输入值
            inputs = [1, 2, 3, 4]
            # 预期的计数器列表结果
            expected = [x * 2 for x in range(NUM_ITERATIONS)]
            # 遍历每个输入值
            for inp in inputs:
                # 调用fct_loop函数并检查计数器列表的结果是否符合预期
                results = fct_loop(inp, NUM_ITERATIONS)
                self.assertEqual(results[1], expected)
    def test_bailout_loop_counter_transition(self):
        # 启用测试时的性能分析模式
        with enable_profiling_mode_for_profiling_tests():
            # 定义循环的迭代次数
            NUM_ITERATIONS = 10

            @torch.jit.script
            def fct_loop(z, size):
                # type: (int, int) -> Tuple[Tensor, List[int]]
                # 定义一个计数器列表
                counters = torch.jit.annotate(List[int], [])
                # 创建一个包含全1张量的y
                y = torch.ones(2)
                # 循环size次
                for i in range(size):
                    # 将当前迭代次数i添加到计数器列表中
                    counters.append(i)
                    # 在y张量的末尾拼接z个全1张量
                    y = torch.cat((y, torch.ones(z)), 0)
                return y, counters

            # 输入测试用例
            inputs = [1, 2, 3, 4]
            # 期望的计数器结果
            expected = list(range(NUM_ITERATIONS))
            # 遍历每个输入，执行函数并断言计数器结果符合期望
            for inp in inputs:
                results = fct_loop(inp, NUM_ITERATIONS)
                self.assertEqual(results[1], expected)

    def test_ignored_method_binding(self):
        # 定义一个继承自torch.nn.Module的类Bar
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个整数类型的属性x为0
                self.x : int = 0

            # 定义一个导出到Torch脚本的方法setx，用于设置属性x的值
            @torch.jit.export
            def setx(self, x : int):
                self.x = x

            # 定义一个导出到Torch脚本的方法getx，用于获取属性x的值
            @torch.jit.export
            def getx(self):
                return self.x

            # 定义一个被忽略的方法ignored_getx，用于获取属性x的值
            @torch.jit.ignore
            def ignored_getx(self):
                return self.x

        # 创建Bar类的实例b
        b = Bar()
        # 设置b的属性x为123
        b.setx(123)
        # 将b转换为Torch脚本对象sb，并断言获取属性x的值为123
        sb = torch.jit.script(b)
        self.assertEqual(sb.getx(), 123)
        # 调用被忽略的方法ignored_getx，断言获取属性x的值为123
        self.assertEqual(sb.ignored_getx(), 123)

        # 设置sb的属性x为456，并断言获取属性x的值为456
        sb.setx(456)
        self.assertEqual(sb.getx(), 456)
        # 调用被忽略的方法ignored_getx，断言获取属性x的值为456
        self.assertEqual(sb.ignored_getx(), 456)

    def test_set_attribute_through_optional(self):
        # 定义一个继承自torch.nn.Module的类A
        class A(torch.nn.Module):
            __annotations__ = {"x": Optional[torch.Tensor]}

            def __init__(self):
                super().__init__()
                # 初始化属性x为None
                self.x = None

            # 定义一个被忽略的方法foo，用于设置属性x的值为张量[3]
            @torch.jit.ignore
            def foo(self):
                if self.x is None:
                    self.x = torch.tensor([3])
                return self.x

            # 前向传播函数，调用忽略方法foo，并返回x + 1
            def forward(self, x):
                a = self.foo()
                return x + 1

        # 创建A类的Torch脚本模型m
        m = torch.jit.script(A())
        # 断言属性x的初始值为None
        self.assertEqual(m.x, None)
        # 调用m的前向传播函数，断言属性x的值为张量[3]
        m(torch.rand(1))
        self.assertEqual(m.x, torch.tensor([3]))

    def test_mutate_constant(self):
        # 定义一个继承自torch.jit.ScriptModule的类M
        class M(torch.jit.ScriptModule):
            __constants__ = ["foo"]

            def __init__(self, foo):
                super().__init__()
                # 初始化一个常量属性foo
                self.foo = foo

        # 创建M类的实例m，尝试修改常量属性foo并断言抛出运行时错误
        m = M(5)
        with self.assertRaises(RuntimeError):
            m.foo = 6

    def test_class_attribute(self):
        # 定义一个继承自torch.jit.ScriptModule的类M
        class M(torch.jit.ScriptModule):
            FOO = 0

            def __init__(self):
                super().__init__()
                # 初始化实例属性foo为类属性FOO的值
                self.foo = self.FOO

        # 创建M类的实例m，断言实例属性foo的初始值为类属性FOO的值0
        m = M()
        self.assertEqual(m.foo, M.FOO)
    def test_class_attribute_in_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            # 类属性 FOO 被初始化为整数 0
            FOO = 0

            # 定义一个脚本方法 forward
            @torch.jit.script_method
            def forward(self):
                # 返回类属性 FOO 的值
                return self.FOO
        # 使用断言检查在实例化 M 类时是否会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            M()

    def test_not_initialized_err(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                # 在初始化函数中，尝试初始化实例变量 foo 为一个 2x3 的随机张量
                self.foo = torch.rand(2, 3)
        # 使用断言检查在实例化 M 类时是否会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            M()

    def test_attribute_in_init(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 在初始化函数中，创建一个名为 foo 的 jit.Attribute，初始值为 0.1，类型为 float
                self.foo = torch.jit.Attribute(0.1, float)
                # 使用断言检查 self.foo 是否能够作为一个 float 使用，即其值大于 0.0
                assert 0.0 < self.foo
        # 实例化类 M，触发初始化函数，并执行断言
        M()

    def test_scriptable_fn_as_attr(self):
        # 定义一个继承自 torch.nn.Module 的类 M，构造函数接受一个函数 fn 作为参数
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                # 将传入的函数 fn 作为类的一个属性保存
                self.fn = fn

            def forward(self, x):
                # 在前向传播函数中，调用保存的函数 fn 并返回其结果
                return self.fn(x)

        # 实例化类 M，传入 torch.sigmoid 函数作为 fn 参数
        m = M(torch.sigmoid)
        # 创建一个随机张量作为输入
        inp = torch.rand(2, 3)
        # 使用 self.checkModule 方法检查模块 m 的行为是否符合预期
        self.checkModule(m, (inp, ))

    def test_sequence_parsing(self):
        # 定义一组测试用例 tests，每个元素是一个包含表达式和预期结果的元组
        tests = [
            ("return [x, x,]", True),
            ("return [x x]", "expected ]"),
            ("return x, x,", True),
            ("return bar(x, x,)", True),
            ("return bar()", "Argument x not provided"),
            ("for a, b, in x, x,:\n        pass", "List of iterables"),
            ("a, b, = x, x,\n    return a + b", True)
        ]
        # 遍历测试用例
        for exp, result in tests:
            # 创建一个 torch.jit.CompilationUnit 的实例 cu
            cu = torch.jit.CompilationUnit()
            # 构建完整的测试代码块 full，其中包含一个带注释的字符串文本
            full = f"""
def bar(x, y):
    return x + y  # 返回两个参数的和

def foo(x):
    """
    定义一个函数 foo，接受一个参数 x
            """
    if isinstance(result, str):
        with self.assertRaisesRegex(RuntimeError, result):
            cu.define(full)
    else:
        cu.define(full)

def test_namedtuple_python(self):
    """
    测试命名元组在 Python 中的使用
    """
    global MyTuple, MyMod  # 全局变量 MyTuple 和 MyMod
    MyTuple = namedtuple('MyTuple', ['a'])  # 创建命名元组 MyTuple，包含一个字段 'a'

    @torch.jit.unused
    def fn():
        """
        定义一个未使用的 Torch 脚本函数 fn
        # type: () -> MyTuple
        返回一个 MyTuple 对象，包含值为 1 的字段 'a'
        """
        return MyTuple(1)

    # 仅检查编译情况
    @torch.jit.script
    def fn2():
        """
        Torch 脚本函数 fn2
        # type: () -> MyTuple
        返回 fn() 的执行结果
        """
        return fn()

    FileCheck().check("NamedTuple").run(fn2.graph)  # 运行文件检查器，检查 fn2 的图形

    class MyMod(torch.nn.Module):
        """
        定义一个 Torch 模块 MyMod
        """
        @torch.jit.unused
        def fn(self):
            """
            定义一个未使用的 Torch 脚本方法 fn
            # type: () -> MyTuple
            返回一个 MyTuple 对象，包含值为 1 的字段 'a'
            """
            return MyTuple(1)

        def forward(self, x):
            """
            Torch 模块的前向传播方法
            """
            if 1 == 1:
                return MyTuple(torch.rand(2, 3))  # 返回一个 MyTuple 对象，包含随机生成的 2x3 的张量
            else:
                return self.fn()  # 返回 fn 方法的执行结果

    # 不应抛出类型错误
    torch.jit.script(MyMod())

def test_unused_decorator(self):
    """
    测试未使用的装饰器的行为
    """
    class MyMod(torch.nn.Module):
        """
        定义一个 Torch 模块 MyMod
        """
        @torch.jit.unused
        @torch.no_grad()
        def fn(self, x):
            """
            定义一个未使用的 Torch 方法 fn
            # type: (Tensor) -> int
            返回输入张量的下一个元素，虽然无效，但应被忽略
            """
            return next(x)

        def forward(self, x):
            """
            Torch 模块的前向传播方法
            """
            return self.fn(x)

    torch.jit.script(MyMod())

@_inline_everything
def test_lazy_script(self):
    """
    测试惰性脚本化的行为
    """
    def untraceable(x):
        """
        定义一个不可追踪的函数 untraceable
        """
        if x.ndim > 2:
            print("hello")
        else:
            print("goodbye")
        return x + 2

    # 非工作示例
    def fn(x):
        """
        定义一个函数 fn，调用 untraceable 函数
        """
        return untraceable(x)

    with self.capture_stdout():
        traced_bad = torch.jit.trace(fn, [torch.ones(2, 2)])

    FileCheck().check_not("goodbye").check_not("hello").run(traced_bad.graph)

    # 工作示例
    untraceable = torch.jit.script_if_tracing(untraceable)

    def fn2(x):
        """
        定义一个函数 fn2，调用脚本化后的 untraceable 函数
        """
        return untraceable(x)

    with self.capture_stdout():
        traced = torch.jit.trace(fn, [torch.ones(2, 2)])

    FileCheck().check("goodbye").run(traced.graph)

    def foo(x: int):
        """
        定义一个接受整数参数 x 的函数 foo
        """
        return x + 1

    @torch.jit.script_if_tracing
    def fee(x: int = 2):
        """
        定义一个 Torch 脚本函数 fee，接受整数参数 x，默认为 2
        """
        return foo(1) + x

    # 测试直接编译函数
    fee_compiled = torch.jit.script(fee)
    self.assertEqual(fee_compiled(), fee())

    # 测试在另一个函数中编译它
    @torch.jit.script
    def hum():
        """
        定义一个 Torch 脚本函数 hum
        """
        return fee(x=3)

    self.assertEqual(hum(), 5)
    def test_big_int_literals(self):
        def ok():
            # signed 64 bit max
            a = 9223372036854775807
            return a

        def toobig():
            # Attempting to assign a value greater than signed 64 bit max
            a = 9223372036854775808
            return a

        def waytoobig():
            # Attempting to assign an excessively large integer value
            a = 99999999999999999999
            return a

        self.checkScript(ok, [])

        # Verifying script compilation failure due to out-of-range integer
        # Raises RuntimeError with message "out of range"
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            torch.jit.script(toobig)

        # Verifying script compilation failure due to out-of-range integer
        # Raises RuntimeError with message "out of range"
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            torch.jit.script(waytoobig)

    def test_hex_literals(self):
        def test1():
            return 0xaaaaaa

        def test2():
            return 0xaaaaaa

        def test3():
            return -0xaaaaaa

        self.checkScript(test1, [])
        self.checkScript(test2, [])
        self.checkScript(test3, [])

        def ok():
            # Largest signed 64 bit integer represented in hexadecimal
            a = 0x7FFFFFFFFFFFFFFF
            return a

        def toobig():
            # Attempting to assign a value greater than signed 64 bit max in hexadecimal
            a = 0xFFFFFFFFFFFFFFFF
            return a

        def waytoobig():
            # Attempting to assign an excessively large hexadecimal value
            a = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
            return a

        self.checkScript(ok, [])

        # Verifying script compilation failure due to out-of-range hexadecimal
        # Raises RuntimeError with message "out of range"
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            torch.jit.script(toobig)

        # Verifying script compilation failure due to out-of-range hexadecimal
        # Raises RuntimeError with message "out of range"
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            torch.jit.script(waytoobig)

    def test_big_float_literals(self):
        def ok():
            # Python interprets this as inf
            a = 1.2E400
            return a

        def check(fn):
            self.assertTrue(fn() == ok())

        # Check using torch.jit.script to verify script compilation
        # Note: checkScript doesn't work here since assertEqual doesn't consider
        # `inf` == `inf`
        check(torch.jit.script(ok))

        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(ok)))
        check(cu.ok)

    def _test_device_type(self, dest):
        def fn(x):
            # type: (Device) -> Tuple[str, Optional[int]]
            return x.type, x.index

        # Creating a torch device and verifying script compilation
        device = torch.ones(2).to(dest).device
        self.checkScript(fn, [device])

    def test_device_type(self):
        # Testing device type 'cpu'
        self._test_device_type('cpu')

    @unittest.skipIf(not RUN_CUDA, "Requires CUDA")
    def test_device_type_cuda(self):
        # Testing device type 'cuda' (if CUDA is available)
        self._test_device_type('cuda')

    def test_string_device_implicit_conversion(self):
        @torch.jit.script
        def fn(x: torch.device):
            return x

        # Verifying implicit conversion from string to torch.device
        self.assertEqual(fn("cpu"), torch.device("cpu"))

        # Verifying RuntimeError when an invalid device string is provided
        # Raises RuntimeError with message "Expected one of"
        with self.assertRaisesRegex(RuntimeError, "Expected one of"):
            fn("invalid_device")
    # 定义测试函数 test_eval_python，用于测试 Python 执行环境
    def test_eval_python(self):
        # 定义内部测试函数 _test，接受一个模型 m 作为参数
        def _test(m):
            # 断言模型 m 对全 1 的输入返回 True
            self.assertTrue(m(torch.ones(2, 2)))
            # 断言模型 m 当前处于训练状态
            self.assertTrue(m.training)
            # 断言通过私有属性 _c 获取的 training 值为 True

            # 将模型 m 设置为评估模式
            m.eval()

            # 断言模型 m 不再处于训练状态
            self.assertFalse(m.training)
            # 断言通过私有属性 _c 获取的 training 值为 False
            self.assertFalse(m._c.getattr('training'))
            # 断言模型 m 对全 1 的输入返回 False
            self.assertFalse(m(torch.ones(2, 2)))

            # 创建一个字节流缓冲区
            buffer = io.BytesIO()
            # 将模型 m 保存到字节流缓冲区中
            torch.jit.save(m, buffer)
            buffer.seek(0)

            # 从字节流缓冲区中加载模型
            loaded = torch.jit.load(buffer)

            # 断言加载后的模型不处于训练状态
            self.assertFalse(loaded.training)
            # 断言通过私有属性 _c 获取的 training 值为 False

        # 定义一个继承自 nn.Module 的类 M
        class M(nn.Module):
            # 实现前向传播方法，返回当前模型是否处于训练状态
            def forward(self, x):
                return self.training

        # 定义一个继承自 torch.jit.ScriptModule 的类 OldM
        class OldM(torch.jit.ScriptModule):
            # 实现前向传播方法，返回当前模型是否处于训练状态
            @torch.jit.script_method
            def forward(self, x):
                return self.training

        # 分别测试使用 torch.jit.script 方法包装 M 和 OldM 类
        _test(torch.jit.script(M()))
        _test(OldM())

    # 定义测试函数 test_inherit_method，用于测试方法的继承
    def test_inherit_method(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 A
        class A(torch.jit.ScriptModule):
            # 实现前向传播方法，返回输入 x 与 bar(x) 的和
            @torch.jit.script_method
            def forward(self, x):
                return x + self.bar(x)

        # 定义一个继承自 A 类的类 B
        class B(A):
            # 实现前向传播方法，返回输入 x 的平方
            @torch.jit.script_method
            def bar(self, x):
                return x * x

        # 断言在创建 A 类的实例时会抛出 RuntimeError，因为 bar 方法未定义
        with self.assertRaisesRegex(RuntimeError, 'attribute'):
            A()  # 不能使用，因为 bar 方法未定义

        v = torch.rand(3, 4)
        b = B()
        # 断言调用 B 类的实例 b 对输入 v 的结果与 v + v * v 相等
        self.assertEqual(b(v), v + v * v)

        # 定义一个继承自 torch.jit.ScriptModule 的类 C
        class C(torch.jit.ScriptModule):
            # 实现前向传播方法，返回输入 x
            @torch.jit.script_method
            def bar(self, x):
                return x

        # 定义一个同时继承自 C 和 B 类的类 D
        class D(C, B):
            # 构造方法，调用父类构造方法初始化
            def __init__(self):
                super().__init__()

        # 断言调用 D 类的实例，对输入 v 的结果与 v + v 相等
        self.assertEqual(D()(v), v + v)

    # 定义测试函数 test_tensor_subclasses，用于测试张量的子类
    def test_tensor_subclasses(self):
        # 定义函数 check_subclass，接受类型 x 和张量 tensor 作为参数
        def check_subclass(x, tensor):
            # 定义一个模板字符串，用于生成 TorchScript 函数
            template = dedent("""
                def func(input: {}) -> {}:
                    return torch.zeros((input.shape[0], 1), dtype=input.dtype)
                """)
            
            # 检查生成的 TorchScript 代码模板，生成函数名为 func 的 TorchScript 函数
            self._check_code(template.format(x, x), "func", [tensor])

        # 使用不同类型的张量调用 check_subclass 函数，检查生成的 TorchScript 代码
        check_subclass("torch.LongTensor", torch.LongTensor([[1, 2], [3, 4]]))
        check_subclass("torch.DoubleTensor", torch.DoubleTensor([[1.2, 2.3], [3.4, 4.5]]))
        check_subclass("torch.IntTensor", torch.IntTensor([[1, 2], [3, 4]]))
        check_subclass("torch.BoolTensor", torch.BoolTensor([[False, True], [True, False]]))

        # 定义一个函数 check_subclass_warn，接受类型为 torch.LongTensor 的输入参数，
        # 返回类型也为 torch.LongTensor
        def check_subclass_warn(input: torch.LongTensor) -> torch.LongTensor:
            return torch.zeros((input.shape[0], 1), dtype=input.dtype)

        # 捕获 TorchScript 编译时的警告信息
        with warnings.catch_warnings(record=True) as warns:
            # 对 check_subclass_warn 函数进行 TorchScript 编译
            scripted = torch.jit.script(check_subclass_warn)
        # 使用 FileCheck 检查警告信息中是否包含 "TorchScript will treat type annotations of Tensor" 字符串
        FileCheck().check("TorchScript will treat type annotations of Tensor").run(str(warns[0]))
    #`
    def test_first_class_module(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 Foo
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个形状为 (3, 4) 的参数化张量 foo
                self.foo = nn.Parameter(torch.rand(3, 4))

            @torch.jit.script_method
            def forward(self, input):
                # 将输入赋值给 foo
                self.foo = input
                return self.foo
        
        # 创建一个 Foo 类的实例 foo
        foo = Foo()
        # 创建一个形状为 (3, 4) 的随机张量 input
        input = torch.rand(3, 4)
        # 调用 foo 的 forward 方法，传入 input
        foo.forward(input)
        # 断言输入 input 等于 foo 实例中的 foo 参数
        self.assertEqual(input, foo.foo)

    @_tmp_donotuse_dont_inline_everything
    def test_first_class_calls(self):
        # 定义一个通过 torch.jit.script 装饰的类 Foo
        @torch.jit.script
        class Foo:
            def __init__(self, x):
                # 初始化属性 bar 为 x
                self.bar = x

            def stuff(self, x):
                # 返回 bar 和 x 的和
                return self.bar + x

        # 定义一个通过 torch.jit.script 装饰的函数 foo
        @torch.jit.script
        def foo(x):
            # 返回 x 的平方加上 Foo 类实例化对象的 stuff 方法的结果
            return x * x + Foo(x).stuff(2 * x)

        # 定义一个通过 torch.jit.script 装饰的函数 bar
        @torch.jit.script
        def bar(x):
            # 返回 foo(x) 的平方
            return foo(x) * foo(x)

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 断言调用 bar(x) 的结果等于 (x * x + 3 * x) 的平方
        self.assertEqual(bar(x), (x * x + 3 * x) * (x * x + 3 * x))

    def test_static_methods(self):
        # 定义一个继承自 nn.Module 的类 M
        class M(nn.Module):
            @staticmethod
            def my_method(x):
                # 返回 x + 100 的静态方法
                return x + 100

            def forward(self, x):
                # 返回 x 加上 M 类的静态方法 my_method 的结果
                return x + M.my_method(x)

        # 定义一个继承自 nn.Module 的类 N
        class N(nn.Module):
            @staticmethod
            def my_method(x):
                # 返回 x 乘以 100 的静态方法
                return x * 100

            def forward(self, x):
                # 返回 x 减去 M 类的静态方法 my_method 的结果再加上 N 类的静态方法 my_method 的结果
                return x - M.my_method(x) + N.my_method(x)

        # 使用 checkModule 方法验证 M 类的实例和输入为 torch.ones(2, 2) 的元组
        self.checkModule(M(), (torch.ones(2, 2),))

        # 使用 checkModule 方法验证 N 类的实例和输入为 torch.ones(2, 2) 的元组
        self.checkModule(N(), (torch.ones(2, 2),))

    def test_invalid_prefix_annotation(self):
        # 使用 self.assertRaisesRegex 捕获 RuntimeError 异常，并验证异常信息中包含 "annotation prefix in line"
        with self.assertRaisesRegex(RuntimeError, "annotation prefix in line"):
            # 使用 torch.jit.script 装饰定义一个函数 invalid_prefix_annotation1
            with self.capture_stdout() as captured:
                @torch.jit.script
                def invalid_prefix_annotation1(a):
                    #type: (Int) -> Int # noqa: E265
                    return a + 2

        with self.assertRaisesRegex(RuntimeError, "annotation prefix in line"):
            # 使用 torch.jit.script 装饰定义一个函数 invalid_prefix_annotation2
            with self.capture_stdout() as captured:
                @torch.jit.script
                def invalid_prefix_annotation2(a):
                    #type   : (Int) -> Int # noqa: E265
                    return a + 2

        with self.assertRaisesRegex(RuntimeError, "annotation prefix in line"):
            # 使用 torch.jit.script 装饰定义一个函数 invalid_prefix_annotation3
            with self.capture_stdout() as captured:
                @torch.jit.script
                def invalid_prefix_annotation3(a):
                    #     type: (Int) -> Int
                    return a + 2

    def test_builtin_function_attributes(self):
        # 定义一个继承自 nn.Module 的类 Add
        class Add(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 add 属性为 torch.add 函数
                self.add = torch.add

            def forward(self, input):
                # 返回调用 add 方法对输入 input 进行加法运算的结果
                return self.add(input, input)

        # 使用 checkModule 方法验证 Add 类的实例和输入为 torch.randn(2, 2) 的列表
        self.checkModule(Add(), [torch.randn(2, 2)])
    # 定义一个测试函数，用于测试 PyTorch 的脚本类型比较功能
    def test_pybind_type_comparisons(self):
        # 使用 torch.jit.script 装饰器将函数 f 编译为 TorchScript
        @torch.jit.script
        def f():
            # 返回空值
            return None
        
        # 获取图中的第一个节点
        node = list(f.graph.nodes())[0]
        # 获取该节点的输出类型
        t = node.outputsAt(0).type()
        # 断言输出类型不为空
        self.assertIsNotNone(t)

    # 如果运行环境是 Windows，则跳过该测试用例（需要修复）
    @unittest.skipIf(IS_WINDOWS, 'TODO: need to fix the test case')
    def test_unmatched_type_annotation(self):
        # 准备错误消息的正则表达式
        message1 = re.escape("Number of type annotations (2) did not match the number of function parameters (1):")
        message2 = 'def invalid2\\(a\\):\n\\s*~+\\.*\\s+<--- HERE\n\\s+# type: \\(Int, Int\\) -> Int\n\\s+return a \\+ 2'
        message3 = 'def invalid4\\(a\\):\n\\s*~+\\.*\\s+<--- HERE\n\\s+# type: \\(Int, Int\\) -> Int\n\\s+return a \\+ 2'

        # 第一个测试用例：验证函数签名与参数数量不匹配的错误消息
        with self.assertRaisesRegex(RuntimeError, message1):
            # 使用 torch.jit.script 装饰器将函数 invalid1 编译为 TorchScript
            @torch.jit.script
            def invalid1(a):
                # type: (Int, Int) -> Int
                return a + 2

        # 第二个测试用例：验证函数签名与参数数量不匹配的错误消息
        with self.assertRaisesRegex(RuntimeError, message2):
            # 使用 torch.jit.script 装饰器将函数 invalid2 编译为 TorchScript
            def invalid2(a):
                # type: (Int, Int) -> Int
                return a + 2

        # 第三个测试用例：验证函数签名与参数数量不匹配的错误消息
        with self.assertRaisesRegex(RuntimeError, message1):
            def invalid3(a):
                # type: (Int, Int) -> Int
                return a + 2
            # 编译 invalid3 函数为 TorchScript
            torch.jit.script(invalid3)

        # 第四个测试用例：验证函数签名与参数数量不匹配的错误消息
        with self.assertRaisesRegex(RuntimeError, message3):
            def invalid4(a):
                # type: (Int, Int) -> Int
                return a + 2
            # 编译 invalid4 函数为 TorchScript
            torch.jit.script(invalid4)

    # 测试函数：验证类型注解中不应包含调用的错误消息
    def test_calls_in_type_annotations(self):
        # 断言运行时错误消息中应包含指定的字符串
        with self.assertRaisesRegex(RuntimeError, "Type annotation should not contain calls"):
            # 定义函数 spooky
            def spooky(a):
                # type: print("Hello") -> Tensor # noqa: F723
                # 返回 a + 2
                return a + 2
            # 打印 torch 库的文件位置
            print(torch.__file__)
            # 获取函数 spooky 的签名信息
            torch.jit.annotations.get_signature(spooky, None, 1, True)

    # 测试函数：验证是否为可选类型
    def test_is_optional(self):
        # 定义一个联合类型注解
        ann = Union[List[int], List[float]]
        # 检查 ann 是否为可选类型
        torch._jit_internal.is_optional(ann)
    def test_interpreter_fuzz(self):
        import builtins
        # 此测试生成随机的类树状程序，用于模糊测试解释器，以确保其在堆栈操作代码中没有错误。
        # 该代码中的断言确保不会重新排序单个操作符。
        templates = [
            "torch.rand(3, 4)",  # 生成一个3x4的随机张量
            "({} + {})",  # 生成两个表达式的加法
            "-{}",  # 生成一个表达式的负数
            "({} * {})",  # 生成两个表达式的乘法
            "torch.tanh({})",  # 计算给定张量的双曲正切函数
            "VAR {}",  # 创建一个变量赋值的表达式
        ]

        def gen_code():
            src_lines = ['def f():']
            exprs = []
            n_variables = 0

            def get_expr(idx):
                elem = exprs[idx]
                exprs[idx] = exprs[-1]
                exprs.pop()
                return elem

            def select_expr_or_var():
                idx = random.randrange(0, len(exprs) + n_variables)
                if idx < len(exprs):
                    return get_expr(idx)
                else:
                    return f'v{idx - len(exprs)}'

            for i in range(50):
                n = None
                while n is None or n > len(exprs) + n_variables:
                    template = random.choice(templates)
                    n = template.count('{}')

                if 'VAR' in template:
                    src_lines.append(f'  v{n_variables} = {select_expr_or_var()}')
                    n_variables += 1
                else:
                    exprs.append(template.format(*(select_expr_or_var() for _ in range(n))))

            src_lines.append('  return ({})\n'.format(''.join(f'v{i},' for i in range(n_variables))))
            return '\n'.join(src_lines)

        for i in range(100):
            g = {'torch': torch}
            code = gen_code()
            builtins.exec(code, g, None)
            cu = torch.jit.CompilationUnit(code)
            with freeze_rng_state():
                o1 = g['f']()
            with freeze_rng_state():
                o2 = cu.f()
            self.assertEqual(o1, o2)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_parameter_order(self):
        m = nn.Module()
        for i, name in enumerate(string.ascii_letters):
            setattr(m, name, nn.Parameter(torch.tensor([float(i)])))
        ms = torch.jit.script(m)
        print(torch.cat(list(m.parameters())))
        print(torch.cat(list(ms.parameters())))
        self.assertEqual(list(m.parameters()), list(ms.parameters()))

    def test_python_op_builtins(self):
        @torch.jit.unused
        def fn(x):
            # type: (List[int]) -> int
            return sum(x)

        @torch.jit.script
        def script_fn(x):
            # type: (List[int]) -> int
            return fn(x)
    def test_submodule_twice(self):
        # 定义一个 TorchScript 函数 foo，实现对输入 x 的平方操作
        @torch.jit.script
        def foo(x):
            return x * x

        # 定义一个继承自 torch.jit.ScriptModule 的类 What
        class What(torch.jit.ScriptModule):
            # 初始化方法，接受参数 x
            def __init__(self, x):
                super().__init__()
                # 将传入的 foo 函数作为属性保存在 self.foo 中
                self.foo = x

        # 创建两个 What 类的实例 a 和 c，都使用相同的 foo 函数作为参数
        a = What(foo)
        c = What(foo)

    def test_training_param(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 What
        class What(torch.jit.ScriptModule):
            # 前向传播方法的 TorchScript 版本
            @torch.jit.script_method
            def forward(self, x):
                # type: (int) -> int
                # 如果当前处于训练模式
                if self.training:
                    r = x
                else:
                    r = x + 4
                # 再次检查是否处于训练模式，并根据情况修改 r
                if self.training:
                    r = r + 1
                return r

        # 创建 What 类的实例 w
        w = What()
        # 断言 w(3) 的输出为 4（处于训练模式时）
        self.assertEqual(4, w(3))
        # 将模型置为评估模式（非训练模式），断言 w(3) 的输出为 7
        w.train(False)
        self.assertEqual(7, w(3))
        # 断言模型状态字典中不包含 "training" 字段
        self.assertFalse("training" in w.state_dict())

    def test_class_as_attribute(self):
        # 定义一个 TorchScript 类 Foo321
        @torch.jit.script
        class Foo321:
            # 初始化方法，设置属性 self.x = 3
            def __init__(self):
                self.x = 3

        # 定义一个继承自 torch.nn.Module 的类 FooBar1234
        class FooBar1234(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 Foo321 类的实例作为属性 self.f
                self.f = Foo321()

            # 前向传播方法，对输入 x 加上 self.f.x 的值并返回
            def forward(self, x):
                return x + self.f.x

        # 使用 torch.jit.script 将 FooBar1234 类转换为 TorchScript
        scripted = torch.jit.script(FooBar1234())
        # 调用外部方法 getExportImportCopy，对 scripted 进行导出和导入的复制
        eic = self.getExportImportCopy(scripted)
        # 生成一个 3x4 的随机张量 x
        x = torch.rand(3, 4)
        # 断言 scripted(x) 和 eic(x) 的输出结果相等
        self.assertEqual(scripted(x), eic(x))

    def test_module_str(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 前向传播方法，使用 ReLU 激活函数处理输入 x 并返回
            def forward(self, x):
                return torch.relu(x)

        # 使用 torch.jit.script 将 Foo 类转换为 TorchScript
        f = torch.jit.script(Foo())

        # 获取 f._c 对象的字符串表示形式
        str_f = str(f._c)
        # 断言 str_f 以 'ScriptObject' 开头，并且包含 '__torch__.' 和 '.Foo'
        self.assertTrue(str_f.startswith('ScriptObject'))
        self.assertTrue('__torch__.' in str_f)
        self.assertTrue('.Foo' in str_f)

    def test_jitter_bug(self):
        # 定义一个 TorchScript 函数 fn2，接受输入 input 和 kernel_size
        @torch.jit.script
        def fn2(input, kernel_size):
            # type: (Tensor, List[int]) -> Tensor
            # 如果 kernel_size 的第一个元素大于 1，则设置 _stride 为 [2]
            if kernel_size[0] > 1:
                _stride = [2]
            else:
                _stride = kernel_size
            # 打印 _stride 和 kernel_size 的值，并返回 input
            print(_stride, kernel_size)
            return input

        # 定义一个 TorchScript 函数 fn，调用 fn2，并将 kernel_size 设置为 [1]
        @torch.jit.script
        def fn(input):
            # type: (Tensor) -> Tensor
            return fn2(input, [1])

    def test_parser_kwargonly(self):
        # 创建一个 torch.jit.CompilationUnit 对象 cu，包含两个 TorchScript 函数
        cu = torch.jit.CompilationUnit('''
            def foo(x, *, y) -> Tuple[Tensor, Tensor]:
                return x, x
            def bar(x):
                return foo(x, y=x)
        ''')
        # 断言 cu.foo 的 schema 中包含 '*'
        self.assertTrue('*' in str(cu.foo.schema))
        # 使用断言捕获 RuntimeError，确保在以下情况下抛出异常
        with self.assertRaisesRegex(RuntimeError, "not provided"):
            torch.jit.CompilationUnit('''
                def foo(x, *, y) -> Tuple[Tensor, Tensor]:
                    return x, x
                def bar(x):
                    return foo(x, x)
            ''')
    def test_annoying_doubles(self):
        # 创建一个名为 "temp" 的临时模块
        mod = types.ModuleType("temp")
        # 在临时模块中定义特殊的浮点数常量
        mod.inf = float("inf")     # 正无穷大
        mod.ninf = float("-inf")   # 负无穷大
        mod.nan = float("nan")     # 非数值（NaN）

        # 禁用 torch 内部的发射钩子
        with torch._jit_internal._disable_emit_hooks():
            # 定义一个继承自 torch.jit.ScriptModule 的脚本模块 Foo
            class Foo(torch.jit.ScriptModule):
                @torch.jit.script_method
                def forward(self):
                    # 返回一些常量和 mod 模块中定义的特殊浮点数
                    return math.pi, 0.1, mod.inf, mod.ninf, 2.225073858507201e-308, mod.nan

            foo = Foo()  # 创建 Foo 类的实例
            buffer = io.BytesIO()  # 创建一个字节流对象
            torch.jit.save(foo, buffer)  # 将 foo 保存到字节流中

            buffer.seek(0)  # 将字节流指针移动到开头
            foo_loaded = torch.jit.load(buffer)  # 从字节流中加载模型

            r = foo()  # 调用原始模型实例
            r2 = foo_loaded()  # 调用加载后的模型实例
            # 使用精确的断言，检查浮点数细节
            self.assertTrue(r[:-1] == r2[:-1])  # 检查前几个返回值是否相等
            self.assertTrue(math.isnan(r[-1]) and math.isnan(r2[-1]))  # 检查最后一个返回值是否都是 NaN

    def test_type_annotate(self):

        def foo(a):
            return torch.jit.annotate(torch.Tensor, a)

        self.checkScript(foo, (torch.rand(3),))

        def bar():
            a = torch.jit.annotate(List[int], [])  # 声明一个空列表，列表元素为整型
            for _ in range(10):
                a.append(4)  # 向列表中添加整数 4
            return a

        self.checkScript(bar, ())

        def baz(a):
            return torch.jit.annotate(float, a)  # 声明一个浮点数

        self.checkScript(baz, (torch.rand(()),))  # 使用随机生成的零维张量作为参数

        # 测试注解为 None 类型
        def annotate_none():
            return torch.jit.annotate(Optional[torch.Tensor], None)

        self.checkScript(annotate_none, ())

    def test_robust_op_resolution(self):
        neg = torch.add  # 用误导性的名称来确保我们通过函数名解析

        def stuff(x):
            return neg(x, x)  # 执行 torch.add 操作，对 x 加上自身

        a = (torch.rand(3),)
        self.checkScript(stuff, a)
    def test_nested_aug_assign(self):
        @torch.jit.script
        class SomeClass:
            def __init__(self):
                self.num = 99

            def __iadd__(self, x):
                # type: (int)
                # 实现就地加法操作，将传入的整数 x 添加到 self.num 上
                self.num += x
                return self

            def __eq__(self, other):
                # type: (SomeClass) -> bool
                # 比较两个 SomeClass 实例的 num 属性是否相等
                return self.num == other.num

        @torch.jit.script
        class SomeOutOfPlaceClass:
            def __init__(self):
                self.num = 99

            def __add__(self, x):
                # type: (int)
                # 返回一个新的 SomeOutOfPlaceClass 实例，其中 num 属性设置为 x
                self.num = x
                return self

            def __eq__(self, other):
                # type: (SomeClass) -> bool
                # 比较两个 SomeOutOfPlaceClass 实例的 num 属性是否相等
                return self.num == other.num

        class Child(nn.Module):
            def __init__(self):
                super().__init__()
                self.x = 2
                self.o = SomeClass()  # 创建 SomeClass 的实例
                self.oop = SomeOutOfPlaceClass()  # 创建 SomeOutOfPlaceClass 的实例
                self.list = [1, 2, 3]

        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()  # 创建 Child 的实例

            def forward(self):
                self.child.x += 1  # 增加 Child 实例的 x 属性
                self.child.o += 5  # 调用 SomeClass 实例的 __iadd__ 方法，增加 num 属性
                self.child.oop += 5  # 调用 SomeOutOfPlaceClass 实例的 __add__ 方法，设置 num 属性
                some_list = [1, 2]
                self.child.list += some_list  # 将 some_list 添加到 Child 实例的 list 属性中
                self.child.list *= 2  # 将 Child 实例的 list 属性扩展一倍
                return self.child.x, self.child.o, self.child.list, self.child.oop

        a = A()  # 创建 A 的实例
        sa = torch.jit.script(A())  # 使用 Torch 脚本创建 A 的实例
        eager_result = a()  # 调用实例 a 的 forward 方法
        script_result = sa()  # 调用 Torch 脚本实例 sa 的 forward 方法
        self.assertEqual(eager_result, script_result)  # 检查两个 forward 方法的结果是否一致
        self.assertEqual(a.child.x, sa.child.x)  # 检查两个实例的 child.x 属性是否相等
        self.assertEqual(a.child.o, sa.child.o)  # 检查两个实例的 child.o 属性是否相等
        self.assertEqual(a.child.list, sa.child.list)  # 检查两个实例的 child.list 属性是否相等

        @torch.jit.script
        class SomeNonAddableClass:
            def __init__(self):
                self.num = 99

            def __eq__(self, other):
                # type: (SomeClass) -> bool
                # 比较两个 SomeNonAddableClass 实例的 num 属性是否相等
                return self.num == other.num

        # with self.assertRaisesRegex(RuntimeError, "")
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.x = SomeNonAddableClass()  # 创建 SomeNonAddableClass 的实例

            def forward(self):
                self.x += SomeNonAddableClass()  # 尝试对 SomeNonAddableClass 实例应用 += 操作
                return self.x

        with self.assertRaisesRegex(RuntimeError, "Cannot emit inplace op"):
            torch.jit.script(A())  # 尝试将 A 类 Torch 脚本化，应该触发 RuntimeError
    # 定义一个测试函数，测试变量的增量赋值功能
    def test_var_aug_assign(self):
        # 使用 Torch Script 注解定义一个不可加操作的类
        @torch.jit.script
        class SomeNonAddableClass:
            def __init__(self):
                self.num = 99

            # 定义相等性方法，用于比较对象是否相等
            def __eq__(self, other):
                # type: (SomeNonAddableClass) -> bool
                return self.num == other.num

        # 测试捕获 RuntimeError 异常，检测是否能够生成不可原位操作的 Torch Script
        with self.assertRaisesRegex(RuntimeError, "Cannot emit inplace op"):
            @torch.jit.script
            def fn():
                # 创建 SomeNonAddableClass 的实例 a
                a = SomeNonAddableClass()
                # 尝试对 a 执行增量赋值操作
                a += SomeNonAddableClass()
                return a

        # 使用 Torch Script 注解定义一个可增量赋值的类
        @torch.jit.script
        class SomeClass:
            def __init__(self):
                self.num = 99

            # 定义增量赋值方法，将 self.num 增加 x
            def __iadd__(self, x):
                # type: (int)
                self.num += x
                return self

            # 定义相等性方法，用于比较对象是否相等
            def __eq__(self, other):
                # type: (SomeClass) -> bool
                return self.num == other.num

        # 使用 Torch Script 注解定义一个非原位操作的类
        @torch.jit.script
        class SomeOutOfPlaceClass:
            def __init__(self):
                self.num = 99

            # 定义非原位加法方法，将 self.num 设置为 x
            def __add__(self, x):
                # type: (int)
                self.num = x
                return self

            # 定义相等性方法，用于比较对象是否相等
            def __eq__(self, other):
                # type: (SomeClass) -> bool
                return self.num == other.num

        # 定义一个函数 fn2，测试不同类型对象的增量赋值行为
        def fn2():
            # 创建 SomeClass 的实例 a
            a = SomeClass()
            # 复制 a 的引用到 a_copy
            a_copy = a
            # 对 a 执行增量赋值操作，增加 20
            a += 20
            # 断言增量赋值后 a 仍然指向原来的对象
            assert a is a_copy

            # 创建 SomeOutOfPlaceClass 的实例 b
            b = SomeOutOfPlaceClass()
            # 复制 b 的引用到 b_copy
            b_copy = b
            # 对 b 执行非原位加法操作，增加 99
            b += 99
            # 断言非原位操作后 b 仍然指向原来的对象
            assert b is b_copy

            # 创建列表 c
            c = [1, 2, 3]
            # 复制 c 的引用到 c_copy
            c_copy = c
            # 对列表 c 执行增量乘法操作，重复原列表内容
            c *= 2
            # 断言增量乘法后 c 仍然指向原来的对象
            assert c is c_copy
            # 对列表 c 执行增量加法操作，添加额外的元素
            c += [4, 5, 6]

            # 创建 Tensor d，初始化为全 1
            d = torch.ones(2, 2)
            # 复制 d 的引用到 d_copy
            d_copy = d
            # 对 Tensor d 执行增量加法操作，每个元素加 1
            d += torch.ones(2, 2)
            # 断言增量加法后 d 仍然指向原来的对象
            assert d is d_copy

            # 返回测试结果
            return a, b, c, d

        # 调用自定义方法检查 Torch Script 生成的函数 fn2
        self.checkScript(fn2, [])

    # 定义一个测试函数，测试嵌套列表的构造
    def test_nested_list_construct(self):
        # 定义一个简单函数 foo，返回包含两个嵌套列表的列表
        def foo():
            return [[4]] + [[4, 5]]

        # 调用自定义方法检查 Torch Script 生成的函数 foo
        self.checkScript(foo, ())

    # 定义一个测试函数，测试在错误的文件行号上生成错误
    def test_file_line_error(self):
        # 定义一个函数 foobar，尝试调用不存在的 Torch 函数
        def foobar(xyz):
            return torch.blargh(xyz)

        # 获取函数 foobar 的源代码行号
        _, lineno = inspect.getsourcelines(foobar)
        # 测试捕获 RuntimeError 异常，检测是否能在错误的文件行号上生成错误信息
        with self.assertRaisesRegex(RuntimeError, f'test_jit.py", line {lineno + 1}'):
            # 尝试生成 Torch Script
            scripted = torch.jit.script(foobar)

    # 定义一个测试函数，测试在类定义中错误的文件行号上生成错误
    def test_file_line_error_class_defn(self):
        # 定义一个简单的类 FooBar，包含一个方法 baz，尝试调用不存在的 Torch 函数
        class FooBar:
            def baz(self, xyz):
                return torch.blargh(xyz)

        # 获取类 FooBar 的源代码行号
        _, lineno = inspect.getsourcelines(FooBar)
        # 测试捕获 RuntimeError 异常，检测是否能在错误的文件行号上生成错误信息
        with self.assertRaisesRegex(RuntimeError, f'test_jit.py", line {lineno + 2}'):
            # 尝试生成 Torch Script
            torch.jit.script(FooBar)

    # 定义一个测试函数，测试 Torch Script 生成的图中的文件行号
    def test_file_line_graph(self):
        # 定义一个函数 foobar，将输入的数值取负值
        def foobar(xyz):
            return torch.neg(xyz)

        # 生成 Torch Script
        scripted = torch.jit.script(foobar)

        # 获取函数 foobar 的源代码行号
        _, lineno = inspect.getsourcelines(foobar)
        # 创建 FileCheck 对象，检查 Torch Script 生成的图中的文件行号
        fc = FileCheck().check(f'test_jit.py:{lineno + 1}:19')
        # 在 Torch Script 生成的图上运行 FileCheck 对象
        fc.run(scripted.graph)
        # 在字符串表示的 Torch Script 生成的图上运行 FileCheck 对象
        fc.run(str(scripted.graph))
    # 定义一个测试方法，用于测试文件行保存和加载功能
    def test_file_line_save_load(self):
        # 定义一个继承自torch.jit.ScriptModule的类Scripted
        class Scripted(torch.jit.ScriptModule):
            # 定义一个脚本方法forward，接收xyz作为参数，返回其负值
            @torch.jit.script_method
            def forward(self, xyz):
                return torch.neg(xyz)

        # 创建Scripted类的实例
        scripted = Scripted()

        # 创建一个注释，说明为何不使用getExportImportCopy方法，而是使用完整的保存/加载路径
        # 因为getExportImportCopy会调用CompilationUnit._import，而不是完整的保存/加载路径
        buffer = scripted.save_to_buffer()
        bytesio = io.BytesIO(buffer)
        # 从BytesIO加载已保存的模型，并重新赋值给scripted
        scripted = torch.jit.load(bytesio)

        # 获取Scripted类定义的源码行号
        _, lineno = inspect.getsourcelines(Scripted)
        # 创建FileCheck对象，并检查scripted.graph中是否包含特定的行号
        fc = FileCheck().check(f':{lineno + 3}')
        # 在scripted.graph上运行FileCheck检查
        fc.run(scripted.graph)
        # 将scripted.graph转换为字符串，并在其上运行FileCheck检查
        fc.run(str(scripted.graph))

    # 定义一个测试方法，用于测试文件行字符串操作
    def test_file_line_string(self):
        # 创建一个torch.jit.CompilationUnit对象，初始化为空字符串
        scripted = torch.jit.CompilationUnit('''
def foo(xyz):
    return torch.neg(xyz)
        ''')

# 定义函数 foo，接受参数 xyz，并返回其相反数
fc = FileCheck().check('<string>:3:11')
# 创建 FileCheck 对象，检查字符串 '<string>:3:11'
fc.run(scripted.foo.graph)
# 运行 FileCheck 对象的检查函数，检查 scripted.foo.graph 是否符合预期
fc.run(str(scripted.foo.graph))
# 再次运行 FileCheck 对象的检查函数，检查字符串表示的 scripted.foo.graph 是否符合预期

@skipIfCrossRef
def test_file_line_trace(self):
    def foobar(xyz):
        return torch.neg(xyz)

    scripted = torch.jit.trace(foobar, (torch.rand(3, 4)))
    # 使用 torch.jit.trace 将函数 foobar 转换为 Torch 脚本

    _, lineno = inspect.getsourcelines(foobar)
    # 获取函数 foobar 的源代码行号
    fc = FileCheck().check(f'test_jit.py:{lineno + 1}:0')
    # 创建 FileCheck 对象，检查字符串 f'test_jit.py:{lineno + 1}:0'
    fc.run(scripted.graph)
    # 运行 FileCheck 对象的检查函数，检查 scripted.graph 是否符合预期
    fc.run(str(scripted.graph))
    # 再次运行 FileCheck 对象的检查函数，检查字符串表示的 scripted.graph 是否符合预期

def test_serialized_source_ranges(self):

    class FooTest(torch.jit.ScriptModule):
        @torch.jit.script_method
        def forward(self, x, w):
            return torch.mm(x, w.t())

    ft = FooTest()
    # 创建 FooTest 的实例 ft
    loaded = self.getExportImportCopy(ft)
    # 调用 self.getExportImportCopy 方法导出并导入 ft 的副本
    _, lineno = inspect.getsourcelines(FooTest)
    # 获取类 FooTest 的源代码行号

    with self.assertRaisesRegex(RuntimeError, f'test_jit.py", line {lineno + 3}'):
        loaded(torch.rand(3, 4), torch.rand(30, 40))
    # 使用 loaded 调用模型，预期抛出 RuntimeError 异常，异常信息中包含 f'test_jit.py", line {lineno + 3}'

def test_serialized_source_ranges_graph(self):

    class FooTest3(torch.jit.ScriptModule):
        @torch.jit.script_method
        def forward(self, x, w):
            return torch.mm(x, w.t())

    ft = FooTest3()
    # 创建 FooTest3 的实例 ft
    loaded = self.getExportImportCopy(ft)
    # 调用 self.getExportImportCopy 方法导出并导入 ft 的副本
    _, lineno = inspect.getsourcelines(FooTest3)
    # 获取类 FooTest3 的源代码行号

    fc = FileCheck().check(f'test_jit.py:{lineno + 3}')
    # 创建 FileCheck 对象，检查字符串 f'test_jit.py:{lineno + 3}'
    fc.run(loaded.graph)
    # 运行 FileCheck 对象的检查函数，检查 loaded.graph 是否符合预期

def test_serialized_source_ranges2(self):

    class FooTest2(torch.jit.ScriptModule):
        @torch.jit.script_method
        def forward(self):
            raise RuntimeError('foo')

    _, lineno = inspect.getsourcelines(FooTest2)
    # 获取类 FooTest2 的源代码行号

    with self.assertRaisesRegex(torch.jit.Error, f'test_jit.py", line {lineno + 3}'):
        ft = FooTest2()
        # 创建 FooTest2 的实例 ft
        loaded = self.getExportImportCopy(ft)
        # 调用 self.getExportImportCopy 方法导出并导入 ft 的副本
        loaded()
        # 调用 loaded 实例，预期抛出 torch.jit.Error 异常，异常信息中包含 f'test_jit.py", line {lineno + 3}'
    # 定义一个测试方法，验证序列化后的源范围不会抖动
    def test_serialized_source_ranges_dont_jitter(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 FooTest3
        class FooTest3(torch.jit.ScriptModule):
            # 定义一个 torch.jit.script_method，表示这是一个脚本方法
            @torch.jit.script_method
            # 定义前向传播方法，参数为 lim
            def forward(self, lim):
                # 初始化变量 first、second、i、somenum、dontmutateme 和 third
                first = 1
                second = 1
                i = 1
                somenum = 5
                dontmutateme = 3
                third = 0
                # 当 i 小于 lim 时循环执行
                while bool(i < lim):
                    # 计算 third，并更新 first 和 second
                    third = first + second
                    first = second
                    second = third
                    j = 0
                    # 内部循环，每次将 somenum 乘以 2
                    while j < 10:
                        somenum = somenum * 2
                        j = j + 1
                    # 更新 i
                    i = i + j
                    # 更新 i，加上 dontmutateme 的值
                    i = i + dontmutateme

                # 计算 st 和 fs，并返回 third、st、fs
                st = second + third
                fs = first + second
                return third, st, fs

        # 创建 FooTest3 的实例 ft3
        ft3 = FooTest3()

        # 定义一个用于调试的方法，返回从模块中获取的调试记录和缓冲区
        def debug_records_from_mod(self, mod):
            # 创建一个字节流缓冲区 buffer
            buffer = io.BytesIO()
            # 将 ft3 序列化后保存到 buffer 中
            torch.jit.save(ft3, buffer)
            # 将缓冲区指针移动到开头
            buffer.seek(0)
            # 使用 zipfile 打开缓冲区中的存档
            archive = zipfile.ZipFile(buffer)
            # 过滤出以 'archive/code/' 开头的文件名
            files = filter(lambda x: x.startswith('archive/code/'), archive.namelist())
            # 过滤出以 '.debug_pkl' 结尾的文件名，保存在 debug_files 列表中
            debug_files = list(filter(lambda f: f.endswith('.debug_pkl'), files))
            # 断言调试文件的数量为 1
            self.assertEqual(len(debug_files), 1)
            # 打开第一个调试文件
            debug_file = archive.open(debug_files[0])
            # 返回反序列化后的 pickle 数据和缓冲区对象
            return pickle.load(debug_file), buffer

        # 调用 debug_records_from_mod 方法获取 records1 和 buffer
        records1, buffer = debug_records_from_mod(self, ft3)

        # 将 buffer 指针移动到开头
        buffer.seek(0)
        # 从缓冲区中加载模型，得到 loaded 对象
        loaded = torch.jit.load(buffer)
        # 再次调用 debug_records_from_mod 方法获取 records2 和 buffer
        records2, buffer = debug_records_from_mod(self, loaded)

        # 将 buffer 指针移动到开头
        buffer.seek(0)
        # 从缓冲区中加载模型，得到 loaded2 对象
        loaded2 = torch.jit.load(buffer)
        # 再次调用 debug_records_from_mod 方法获取 records3 和忽略的对象
        records3, _ = debug_records_from_mod(self, loaded2)

        # 断言 records1 等于 records2
        self.assertEqual(records1, records2)
        # 断言 records2 等于 records3
        self.assertEqual(records2, records3)
    def test_serialized_source_ranges_no_dups(self):
        # 定义一个继承自torch.jit.ScriptModule的测试类FooTest3，用于测试脚本化模块
        class FooTest3(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义脚本方法forward，接受参数lim
            def forward(self, lim):
                # 初始化变量first, second, i, somenum, dontmutateme, third
                first = 1
                second = 1
                i = 1
                somenum = 5
                dontmutateme = 3
                third = 0
                # while循环，条件为i < lim
                while bool(i < lim):
                    # 计算third为first和second的和
                    third = first + second
                    # 更新first和second的值
                    first = second
                    second = third
                    j = 0
                    # 内部while循环，条件为j < 10
                    while j < 10:
                        # 更新somenum为原值的两倍
                        somenum = somenum * 2
                        j = j + 1
                    # 更新i为i加上j
                    i = i + j
                    # 更新i为i加上dontmutateme
                    i = i + dontmutateme

                # 计算st为second和third的和，fs为first和second的和
                st = second + third
                fs = first + second
                # 返回third, st, fs这三个值
                return third, st, fs

        # 创建FooTest3类的实例ft3
        ft3 = FooTest3()

        # 定义函数debug_records_from_mod，用于从模块中获取调试记录
        def debug_records_from_mod(mod):
            # 创建一个字节流缓冲区
            buffer = io.BytesIO()
            # 将ft3模块保存到buffer中
            torch.jit.save(ft3, buffer)
            buffer.seek(0)
            # 从buffer中创建zip文件对象archive
            archive = zipfile.ZipFile(buffer)
            # 获取以'archive/code/'开头的文件列表
            files = list(filter(lambda x: x.startswith('archive/code/'), archive.namelist()))
            # 过滤出以'.debug_pkl'结尾的调试文件列表
            debug_files = filter(lambda f: f.endswith('.debug_pkl'), files)
            # 打开每个调试文件并加载pickle数据，提取第三个元素
            debug_files = (archive.open(f) for f in debug_files)
            debug_files = (pickle.load(f) for f in debug_files)
            debug_files = (f[2] for f in debug_files)
            # 返回调试文件的列表
            return list(debug_files)

        # 调用debug_records_from_mod函数，获取ft3模块的调试记录
        debug_files = debug_records_from_mod(ft3)
        # 遍历调试文件列表中的每个调试文件
        for debug_file in debug_files:
            # 遍历每个调试文件中的元素，比较相邻元素的source_range值不相等
            for i in range(len(debug_file) - 1):
                offset, source_range_tag, source_range = debug_file[i]
                offset2, source_range_tag2, source_range2 = debug_file[i + 1]
                self.assertNotEqual(source_range, source_range2)

    # 定义测试函数test_circular_dependency，用于测试循环依赖情况
    def test_circular_dependency(self):
        """
        https://github.com/pytorch/pytorch/issues/25871
        """
        # 定义类A，继承自torch.jit.ScriptModule
        class A(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义脚本方法forward，直接返回输入x
            def forward(self, x):
                return x

        # 定义类B，继承自torch.jit.ScriptModule
        class B(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化属性foo为包含一个A实例的ModuleList
                self.foo = torch.nn.ModuleList([A()])

            @torch.jit.script_method
            # 定义脚本方法forward，循环调用self.foo中的模块处理输入x
            def forward(self, x):
                for f in self.foo:
                    x = f(x)
                return x

        # 定义类C，继承自torch.jit.ScriptModule
        class C(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化属性foo为包含一个B实例的Sequential容器
                self.foo = torch.nn.Sequential(B())

            @torch.jit.script_method
            # 定义脚本方法forward，循环调用self.foo中的模块处理输入x
            def forward(self, x):
                for f in self.foo:
                    x = f(x)
                return x

        # 调用self.getExportImportCopy(C())，测试C类的导出和导入
        self.getExportImportCopy(C())
    # 定义一个测试方法，用于测试序列化长行
    def test_serialize_long_lines(self):
        # 定义一个继承自torch.nn.Module的类OrderModuleLong
        class OrderModuleLong(torch.nn.Module):
            # 实现Module类的forward方法，接受名为long_arg_name的torch.Tensor列表
            def forward(self, long_arg_name: List[torch.Tensor]):
                # 返回一个元组列表，包含long_arg_name[1]和long_arg_name[0]的argmax结果
                return [(long_arg_name[1],), (long_arg_name[0].argmax(),)]
        # 通过torch.jit.script将OrderModuleLong类转换为脚本并获取其代码的字符串表示
        src = str(torch.jit.script(OrderModuleLong()).code)
        # 使用FileCheck检查生成的脚本代码，确保long_arg_name[1]在argmax之后没有重新排序
        FileCheck().check("long_arg_name[1]").check("argmax").run(src)

    # 定义一个测试方法，用于测试Tensor的形状
    def test_tensor_shape(self):
        # 创建一个形状为(34, 56, 78)的空Tensor
        x = torch.empty(34, 56, 78)

        # 定义一个函数f，接受一个参数x，并返回x的形状
        def f(x):
            return x.shape

        # 调用self.checkScript方法对函数f进行脚本化，并传入参数x进行验证
        self.checkScript(f, (x,))

    # 定义一个测试方法，用于测试在循环中阻止输入梯度
    def test_block_input_grad_in_loop(self):
        # 创建形状为(3, 3)的随机Tensor x，设置requires_grad为False
        x = torch.randn(3, 3, requires_grad=False)
        # 创建形状为(3, 3)的随机Tensor y，设置requires_grad为True
        y = torch.randn(3, 3, requires_grad=True)

        # 定义一个函数grad_in_loop，接受x和y作为参数，进行100次矩阵乘法运算
        def grad_in_loop(x, y):
            for i in range(100):
                x = y @ x
            return x

        # 使用torch.jit.script对grad_in_loop函数进行脚本化
        scripted = torch.jit.script(grad_in_loop)
        # 获取scripted的计算图，针对输入x和y
        outer = scripted.graph_for(x, y)
        # 查找计算图中的"prim::Loop"节点
        loop = outer.findNode("prim::Loop")
        # 获取循环块的引用
        loop_block = next(loop.blocks())
        # 获取循环块的参数节点
        param_node = loop_block.paramNode()
        # 获取参数节点的第二个输出值（在这里是x），即循环中的计算结果
        x_value = list(param_node.outputs())[1]
        # 断言x_value是否需要梯度，即是否设置了requires_grad为True
        self.assertTrue(x_value.requires_grad())

    # 定义一个测试方法，用于测试Tensor的梯度
    def test_tensor_grad(self):
        # 创建形状为(3, 4)的随机Tensor x，设置requires_grad为True
        x = torch.randn(3, 4, requires_grad=True)
        # 创建形状为(3, 4)的随机Tensor y，设置requires_grad为False
        y = torch.randn(3, 4, requires_grad=False)

        # 定义一个函数f_requires_grad，接受一个参数x，返回x是否需要梯度
        def f_requires_grad(x):
            return x.requires_grad

        # 调用self.checkScript方法对函数f_requires_grad进行脚本化，并传入参数x进行验证
        self.checkScript(f_requires_grad, (x,))
        # 调用self.checkScript方法对函数f_requires_grad进行脚本化，并传入参数y进行验证
        self.checkScript(f_requires_grad, (y,))

        # 定义一个函数f_grad，接受一个参数x，返回x的梯度
        def f_grad(x):
            return x.grad

        # 对x的所有元素求和并反向传播梯度
        x.sum().backward()
        # 调用self.checkScript方法对函数f_grad进行脚本化，并传入参数x进行验证
        self.checkScript(f_grad, (x,))
        # 调用self.checkScript方法对函数f_grad进行脚本化，并传入参数y进行验证
        self.checkScript(f_grad, (y,))

    # 使用unittest.skipIf装饰器，条件是GRAPH_EXECUTOR不等于ProfilingMode.LEGACY时跳过测试
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "shape analysis is only enabled in Legacy")
    # 定义一个测试方法，用于测试prim::grad的未定义行为
    def test_prim_grad_undefined(self):
        # 创建一个全为1的Tensor x，形状为(2,)
        x = torch.ones(2)

        # 定义一个函数f_grad，接受一个参数x，返回x的梯度
        def f_grad(x):
            return x.grad

        # 使用self.checkScript方法对函数f_grad进行脚本化，并传入参数x进行验证
        scripted = self.checkScript(f_grad, (x,))
        # 获取scripted的计算图，针对输入x
        g = scripted.graph_for(x)

        # 查找计算图中的"prim::grad"节点
        prim_grad_node = g.findNode("prim::grad")
        # 断言prim_grad_node的输出类型是否为未定义(undefined)
        self.assertTrue(next(prim_grad_node.outputs()).type().undefined() is None)

    # 定义一个测试方法，用于测试Tensor的数据属性
    def test_tensor_data(self):
        # 创建形状为(3, 4)的随机Tensor x，设置requires_grad为True
        x = torch.randn(3, 4, requires_grad=True)
        # 创建形状为(4, 5)的随机Tensor y

        y = torch.randn(4, 5)

        # 定义一个函数f_data，接受一个参数x，返回x的数据属性（data）
        def f_data(x):
            return x.data

        # 使用torch.jit.script对函数f_data进行脚本化
        scripted_f_data = torch.jit.script(f_data)

        # 对Tensor x和y分别调用scripted_f_data函数进行脚本化
        scripted_x = scripted_f_data(x)
        # 断言scripted_x与直接调用f_data(x)的结果相等
        self.assertEqual(scripted_x, f_data(x))
        # 断言scripted_x的requires_grad属性为False
        self.assertEqual(scripted_x.requires_grad, False)

        scripted_y = scripted_f_data(y)
        # 断言scripted_y与直接调用f_data(y)的结果相等
        self.assertEqual(scripted_y, f_data(y))
        # 断言scripted_x的requires_grad属性为False
        self.assertEqual(scripted_x.requires_grad, False)
    # 定义测试函数，验证张量的数据类型是否正确
    def test_tensor_dtype(self):
        # 创建一个无元素张量，数据类型为 uint8
        x_byte = torch.empty(34, 56, 78, dtype=torch.uint8)
        # 创建一个无元素张量，数据类型为 long
        x_long = torch.empty(34, 56, 78, dtype=torch.long)
        # 创建一个无元素张量，数据类型为 float32
        x_float32 = torch.empty(34, 56, 78, dtype=torch.float32)

        # 使用 Torch Script 将函数 byte 编译为脚本
        @torch.jit.script
        def byte(x):
            return x.dtype == torch.uint8

        # 使用 Torch Script 将函数 long 编译为脚本
        @torch.jit.script
        def long(x):
            return x.dtype == torch.long

        # 使用 Torch Script 将函数 float32 编译为脚本
        @torch.jit.script
        def float32(x):
            return x.dtype == torch.float32

        # 断言 byte 函数对 x_byte 返回 True
        self.assertTrue(byte(x_byte))
        # 断言 byte 函数对 x_long 返回 False
        self.assertFalse(byte(x_long))
        # 断言 byte 函数对 x_float32 返回 False
        self.assertFalse(byte(x_float32))
        
        # 断言 long 函数对 x_byte 返回 False
        self.assertFalse(long(x_byte))
        # 断言 long 函数对 x_long 返回 True
        self.assertTrue(long(x_long))
        # 断言 long 函数对 x_float32 返回 False
        self.assertFalse(long(x_float32))
        
        # 断言 float32 函数对 x_byte 返回 False
        self.assertFalse(float32(x_byte))
        # 断言 float32 函数对 x_long 返回 False
        self.assertFalse(float32(x_long))
        # 断言 float32 函数对 x_float32 返回 True
        self.assertTrue(float32(x_float32))

    # 如果没有 CUDA 环境，跳过此测试
    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    # 测试张量的设备分配
    def test_tensor_device(self):
        # 创建一个在 CPU 上的张量
        cpu = torch.empty(34, 56, 78, device='cpu')
        # 创建一个在 CUDA 上的张量
        gpu = torch.empty(34, 56, 78, device='cuda')

        # 使用 Torch Script 将函数 same_device 编译为脚本
        @torch.jit.script
        def same_device(x, y):
            return x.device == y.device

        # 断言两个 CPU 张量设备相同
        self.assertTrue(same_device(cpu, cpu))
        # 断言两个 GPU 张量设备相同
        self.assertTrue(same_device(gpu, gpu))
        # 断言一个 CPU 张量和一个 GPU 张量设备不同
        self.assertFalse(same_device(cpu, gpu))

    # 如果没有 CUDA 环境，跳过此测试
    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    # 测试张量的设备转换
    def test_tensor_to_device(self):
        # 定义一个函数，将张量转换到 CUDA，然后再转回 CPU
        def to_device(x):
            return x.to(device="cuda").to(device=torch.device("cpu"))

        # 使用 checkScript 验证 to_device 函数的 Torch Script
        self.checkScript(to_device, (torch.ones(3, 4),))

    # 测试张量转到 CPU 的函数
    def test_tensor_to_cpu(self):
        # 定义一个函数，将张量转到 CPU
        def to_cpu(x):
            return x.cpu()

        # 创建一个张量
        x = torch.ones(3, 4)
        # 使用 Torch Script 将函数 to_cpu 编译为脚本
        script_fn = torch.jit.script(to_cpu)
        # 断言直接调用和 Torch Script 版本的结果设备相同
        self.assertEqual(to_cpu(x).device, script_fn(x).device)
        # 使用 checkScript 验证 to_cpu 函数的 Torch Script
        self.checkScript(to_cpu, (x,))

    # 如果没有 CUDA 环境，跳过此测试
    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    # 测试张量转到 CUDA 的函数
    def test_tensor_to_cuda(self):
        # 定义一个函数，将张量转到 CUDA
        def to_cuda(x):
            return x.cuda()

        # 创建一个张量
        x = torch.ones(3, 4)
        # 使用 Torch Script 将函数 to_cuda 编译为脚本
        script_fn = torch.jit.script(to_cuda)
        # 断言直接调用和 Torch Script 版本的结果设备相同
        self.assertEqual(to_cuda(x).device, script_fn(x).device)
        # 使用 checkScript 验证 to_cuda 函数的 Torch Script
        self.checkScript(to_cuda, (x,))

    # 测试泛型列表错误处理
    def test_generic_list_errors(self):
        # 使用断言检查是否抛出指定异常类型和消息的错误
        with self.assertRaisesRegex(RuntimeError, "previously matched to type"):
            # 使用 Torch Script 将函数 foo 编译为脚本
            @torch.jit.script
            def foo(x):
                return [[x]] + [[1]]

    # 测试脚本编译单元
    def test_script_cu(self):
        # 创建一个 Torch 脚本编译单元
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                b = a
                return b
        ''')
        # 创建一个变量张量
        a = Variable(torch.rand(1))
        # 断言调用 cu 的 foo 方法与直接调用 foo 方法结果相同
        self.assertEqual(a, cu.foo(a))

    # 因为编译单元接受 Python 字符串作为输入
    # 要使用转义序列转义反斜杠 (\\n = \n)
    def test_string_cu(self):
        # 创建一个 Torch 脚本编译单元
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                print(a, """a\\n\tb\\n""", 2, "a\
        ''')
    def test_tuple_unsortable_element_type():
        @torch.jit.script
        # 定义一个脚本函数 foo，该函数返回一个列表，其中包含两个字典作为元组的元素
        def foo():
            tups = [({1: 2}, {2: 3})]
            # 尝试对 tups 列表进行排序操作，但由于字典类型的元素不可排序，会引发异常
            tups.sort()
            return tups

        # 使用断言检查是否会抛出 RuntimeError 异常，异常消息包含 "are not sortable"，并且高亮显示出错的地方在 tups.sort 这行
        with self.assertRaisesRegexWithHighlight(RuntimeError, "are not sortable", "tups.sort"):
            foo()

    def test_tuple_unsortable_diff_type():
        @torch.jit.script
        # 定义一个脚本函数 foo，接受一个输入列表，尝试对其进行排序
        def foo(inputs: List[Any]):
            # 尝试对 inputs 列表进行排序操作，但由于包含不同类型的元素，会引发异常
            inputs.sort()
            return inputs

        # 准备一个包含不同类型元素的输入列表
        inputs = [(1, 2), ("foo", "bar")]
        # 使用断言检查是否会抛出 RuntimeError 异常，异常消息包含 "Only values of same type can be compared"，并且高亮显示出错的地方在 inputs.sort 这行
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Only values of same type can be compared", "inputs.sort"):
            foo(inputs)
    def test_tuple_nested_sort(self):
        # 定义一个内部函数 foo，参数为一个元组列表，每个元组包含一个整数和一个元组
        def foo(inputs: List[Tuple[int, Tuple[int, str]]]):
            # 对输入的元组列表按默认顺序进行排序
            inputs.sort()
            return inputs

        # 初始化输入列表 inputs，包含三个元组
        inputs = [(1, (2, "foo")), (1, (2, "bar")), (1, (0, "bar"))]
        # 调用外部函数的检查脚本，传入 foo 函数和 inputs 作为参数
        self.checkScript(foo, (inputs,))

    def test_tuple_unsortable_nested_diff_type(self):
        # 使用 Torch 的脚本装饰器定义一个函数 foo，参数为任意类型的列表
        @torch.jit.script
        def foo(inputs: List[Any]):
            # 尝试对输入的任意类型列表进行排序
            inputs.sort()
            return inputs

        # 初始化输入列表 inputs，包含两个元组
        inputs = [(1, (2, 3)), (2, ("foo", "bar"))]
        # 用带有断言的上下文管理器来验证排序时抛出的运行时错误
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Only values of same type can be compared", "inputs.sort"):
            foo(inputs)

    def test_string_new_line(self):
        # 使用 Torch 的编译单元来创建一个测试函数，其中包含一个打印语句，输出包含换行符的字符串
        with self.assertRaisesRegex(RuntimeError, "expected a valid token*"):
            torch.jit.CompilationUnit('''
            def test_while(a):
                print("
                    a")
                return a
            ''')

    def test_string_single_escape(self):
        # 使用 Torch 的编译单元来创建一个测试函数，其中包含一个打印语句，输出包含单个反斜杠的字符串
        with self.assertRaisesRegex(RuntimeError, "expected a valid token*"):
            torch.jit.CompilationUnit('''
            def test_while(a):
                print("\\")
                return a
            ''')

    def test_script_annotation(self):
        # 使用 Torch 的脚本装饰器定义一个函数 foo，参数为一个变量 a，返回值为 a + a + a
        @torch.jit.script
        def foo(a):
            return a + a + a
        # 创建一个包含随机数的变量 s
        s = Variable(torch.rand(2))
        # 断言 s + s + s 的结果等于调用 foo 函数时传入 s 的结果
        self.assertEqual(s + s + s, foo(s))

    def test_torch_pow(self):
        # 定义一个计算幂函数 func，接受两个参数 a 和 b
        def func(a, b):
            return pow(a, b)

        # 定义一个复合幂函数 func2，接受四个参数 a、b、c、d
        def func2(a, b, c, d):
            return pow(pow(c + a, b), d)

        # 定义一个带类型注解的幂函数 func3，参数 a 是整数，参数 b 是浮点数
        def func3(a : int, b : float):
            # type: (int, float) -> float
            return pow(a, b)

        # 定义一个不带参数的常量幂函数 func4
        def func4():
            # type: () -> float
            return pow(2, -2)

        # 定义一个从张量提取数值后进行幂运算的函数 func5
        def func5(x, y):
            return pow(x.item(), y.item())

        # 定义一个带类型注解的幂函数 func6，参数 a 和 b 都是整数，返回值是浮点数
        def func6(a : int, b : int):
            # type: (int, int) -> float
            return pow(a, b)

        # 创建四个张量 a、b、c、d，用于后续函数调用
        a = torch.rand(1)
        b = torch.rand(1)
        c = torch.rand(1)
        d = torch.rand(1)
        # 分别对 func、func2、func3、func4、func6 函数进行脚本检查
        self.checkScript(func, (a, b))
        self.checkScript(func2, (a, b, c, d))
        self.checkScript(func3, (4, -0.5))
        self.checkScript(func4, ())
        self.checkScript(func6, (2, 4))

        # 创建一个输入张量列表 inputs，包含四个张量
        inputs = [torch.tensor(2), torch.tensor(-2), torch.tensor(.5), torch.tensor(.2)]
        # 对 inputs 中的每个张量进行两两组合，调用 func5 函数进行脚本检查
        for x in inputs:
            for y in inputs:
                if x < 0:
                    continue
                else:
                    self.checkScript(func5, (x, y))

    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    def test_pow_scalar_backward_cuda(self):
        # 测试标量指数与 CUDA 基础兼容性（issue编号：19253）
        with enable_profiling_mode_for_profiling_tests():
            # 遍历数据类型列表，包括 torch.float 和 torch.double
            for dtype in [torch.float, torch.double]:
                @torch.jit.script
                def func(a, b):
                    # type: (Tensor, float) -> Tensor
                    # 返回 (a * 2) 的 b 次幂
                    return (a * 2) ** b

                # 在 CUDA 设备上生成一个随机张量 a，并要求计算梯度
                a = torch.rand(1, requires_grad=True, device='cuda', dtype=dtype)
                # 调用 func 函数并进行反向传播
                func(a, 1, profile_and_replay=True).backward()

                @torch.jit.script
                def func(a, b):
                    # type: (float, Tensor) -> Tensor
                    # 返回 a 的 (b * 2 + 1) 次幂
                    return a ** (b * 2 + 1)

                # 在 CUDA 设备上生成一个随机张量 a，并要求计算梯度
                a = torch.rand(1, requires_grad=True, device='cuda', dtype=dtype)
                # 调用 func 函数并进行反向传播
                func(2, a, profile_and_replay=True).backward()

    def _check_code(self, code_str, fn_name, inputs):
        scope = {}
        # 在全局范围执行传入的代码字符串，并将结果保存在 scope 中
        exec(code_str, globals(), scope)
        cu = torch.jit.CompilationUnit(code_str)
        # 使用 CompilationUnit 创建的函数调用并检查结果
        self.assertEqual(cu.func(*inputs), scope[fn_name](*inputs))

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_scriptmodule_releases_tensors_cuda(self):
        with enable_profiling_mode_for_profiling_tests():
            @torch.jit.script
            def fn(x, y):
                # 返回 x 的 sigmoid 函数值与 y 的 tanh 函数值的乘积
                return x.sigmoid() * y.tanh()

            def test(backward=False):
                # 在 CUDA 设备上生成两个随机张量 x 和 y，并要求计算梯度
                x = torch.randn(3, 3, dtype=torch.double, device='cuda', requires_grad=True)
                y = torch.randn(3, 3, dtype=torch.double, device='cuda', requires_grad=True)
                # 调用 fn 函数并生成输出
                out = fn(x, y, profile_and_replay=True)
                if backward:
                    # 如果指定进行反向传播，则对输出求和并进行反向传播
                    out.sum().backward()

            # 使用 assertLeaksNoCudaTensors 断言，检查 CUDA 张量是否没有泄漏
            with self.assertLeaksNoCudaTensors():
                test()
                test()
                test()

            # 如果不是 SIMPLE 模式，则进一步测试带有反向传播的情况
            if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                with self.assertLeaksNoCudaTensors():
                    test(backward=True)
                    test(backward=True)
                    test(backward=True)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_module_copy_with_attributes(self):
        # 定义一个继承自 torch.jit.ScriptModule 的 Vocabulary 类
        class Vocabulary(torch.jit.ScriptModule):
            # 初始化方法，接受一个词汇列表作为参数
            def __init__(self, vocab_list):
                super().__init__()
                # 设置一个名为 _vocab 的 TorchScript 属性，类型为 List[str]
                self._vocab = torch.jit.Attribute(vocab_list, List[str])
                # 设置一个名为 some_idx 的 TorchScript 属性，类型为 int
                self.some_idx = torch.jit.Attribute(2, int)
                # 设置一个名为 idx 的 TorchScript 属性，类型为 Dict[str, int]，包含词汇列表的索引
                self.idx = torch.jit.Attribute(
                    {word: i for i, word in enumerate(vocab_list)}, Dict[str, int]
                )

            # TorchScript 方法，查找给定词汇列表 values 中每个词汇在词汇表中的索引
            @torch.jit.script_method
            def lookup_indices_1d(self, values):
                # type: (List[str]) -> List[int]
                # 初始化一个空的结果列表
                result = torch.jit.annotate(List[int], [])
                # 循环遍历 values 列表，获取每个词汇的索引或默认索引值，并添加到结果列表中
                for i in range(len(values)):
                    value = values[i]
                    result.append(self.idx.get(value, self.some_idx))
                return result

            # TorchScript 方法，将多个词汇列表转换为对应的索引列表
            @torch.jit.script_method
            def forward(self, values):
                # type: (List[List[str]]) -> List[List[int]]
                # 初始化一个空的结果列表
                result = torch.jit.annotate(List[List[int]], [])
                # 循环遍历 values 中的每个词汇列表，调用 lookup_indices_1d 方法获取索引列表，并添加到结果中
                for i in range(len(values)):
                    result.append(self.lookup_indices_1d(values[i]))
                return result

        # 创建一个 Vocabulary 类的实例 v，初始化词汇列表为 'uabcdefg'
        v = Vocabulary(list('uabcdefg'))
        # 调用对象的 __copy__ 方法
        v.__copy__()

    def test_tuple_to_opt_list(self):
        # 定义一个 TorchScript 函数 foo，接受一个 Optional[List[int]] 类型的参数并返回 int 类型
        @torch.jit.script
        def foo(x):
            # type: (Optional[List[int]]) -> int
            return 1

        # 定义一个 TorchScript 函数 tuple_call，调用 foo 函数，并将元组 (1, 2) 作为参数传递给 foo
        @torch.jit.script
        def tuple_call():
            return foo((1, 2))

    def test_keyword(self):
        # 定义一个 TorchScript 函数 func，接受一个 Tensor x，并在 dim=0 上求和后返回
        @torch.jit.script
        def func(x):
            return torch.sum(x, dim=0)

        # 创建一个形状为 (10,)、数据类型为 float 的随机 Tensor x，并设置 requires_grad=True
        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        # 调用 func 函数对 x 进行求和操作
        y = func(x)
        # 直接调用 torch.sum 对 x 在 dim=0 上进行求和
        y2 = torch.sum(x, dim=0)
        # 断言两种方式求和得到的结果 y 和 y2 相等
        self.assertEqual(y, y2)

    def test_constant_pooling_none(self):
        # 定义一个 TorchScript 函数 typed_nones，接受三个可选类型参数，返回一个元组
        @torch.jit.script
        def typed_nones(a=None, b=None, c=None):
            # type: (Optional[int], Optional[bool], Optional[Tensor]) -> Tuple[Optional[int], Optional[bool], Optional[Tensor]]
            return a, b, c

        # 定义一个 TorchScript 函数 test，接受一个布尔值 a 作为参数，根据条件打印调用 typed_nones 函数的结果
        @torch.jit.script
        def test(a):
            # type: (bool) -> None
            if a:
                print(typed_nones())
            else:
                print(typed_nones())

        # 将 test 函数的计算图转换为字符串形式，并断言字符串中 "NoneType = prim::Constant" 出现的次数为 1
        graph_str = str(test.graph)
        self.assertTrue(graph_str.count("NoneType = prim::Constant") == 1)
    def test_constant_pooling_same_identity(self):
        # 定义内部函数 foo，用于测试常量池化的行为
        def foo():
            # 创建张量 a，包含单个元素 4
            a = torch.tensor([4])
            # 创建元组 b，将张量 a 放入其中
            b = (a,)
            # 计算索引值，获取元组 b 中的元素
            index = len(a) - 1
            c = b[index]  # 获取元组中的元素
            d = b[index]  # 获取元组中的元素
            return c, d

        # 对函数 foo 进行 Torch 脚本编译
        foo_script = torch.jit.script(foo)
        # 运行常量传播优化 pass
        self.run_pass('constant_propagation', foo_script.graph)
        # 运行常量池化优化 pass
        self.run_pass('constant_pooling', foo_script.graph)
        # 尽管 c 和 d 逃逸了作用域，我们仍能将它们合并为一个常量，因为它们是相同的对象
        FileCheck().check_count("prim::Constant", 1, exactly=True).run(foo_script.graph)
        # 断言 Torch 脚本函数的执行结果与未优化版本的执行结果相同
        self.assertEqual(foo(), foo_script())

    def test_constant_pooling_introduce_aliasing(self):
        # 定义 Torch 脚本函数 foo，用于测试常量池化在引入别名时的行为
        @torch.jit.script
        def foo():
            # 创建张量 a 和 b，值均为 1
            a = torch.tensor(1)
            b = torch.tensor(1)
            return a, b

        # 运行常量传播优化 pass
        self.run_pass('constant_propagation', foo.graph)
        # 运行常量池化优化 pass
        self.run_pass('constant_pooling', foo.graph)
        # 不进行常量池化，因为这会引入可观察的别名关系变化
        a, b = foo()
        # 断言 a 和 b 是不同的对象
        self.assertIsNot(a, b)

    def test_literal(self):
        # 定义函数 func1，func2，func3 用于测试 Torch 脚本编译和优化
        def func1(a, b):
            c = a, b
            d, e = c
            return d + e

        def func2(a, b):
            c = a, (a, b)
            d, e = c
            f, g = e
            return d + f + g

        def func3(a, b):
            # type: (float, float) -> float
            c = 0., (0., 0.)
            x = True
            while x:
                x = False
                c = a, (a, b)
            d, e = c
            f, g = e
            return d + f + g

        # 创建具有梯度的随机张量 a 和 b
        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        # 使用脚本检查 func1，func2，func3 的 Torch 脚本编译和优化
        self.checkScript(func1, (a, b), optimize=True)
        self.checkScript(func2, (a, b), optimize=True)
        self.checkScript(func3, (a.item(), b.item()), optimize=True)

    def test_expand(self):
        # 定义 Torch 脚本函数 func，用于测试 Torch 脚本编译和优化
        @torch.jit.script
        def func(x, y):
            return x + y

        # 创建具有梯度的随机张量 x 和 y
        x = torch.rand(2, 3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)
        # 对 func 进行 Torch 脚本编译
        out = func(x, y)
        # 断言 Torch 脚本函数的执行结果与未优化版本的执行结果相同
        self.assertEqual(func(x, y), x + y)

        # 创建梯度张量 grad
        grad = torch.randn(2, 3, dtype=torch.float)
        # 对输出 out 执行反向传播
        out.backward(grad)
        # 断言 x 的梯度与 grad 相同
        self.assertEqual(x.grad, grad)
        # 断言 y 的梯度与 grad 按列求和后相同
        self.assertEqual(y.grad, grad.sum(dim=0))

    def test_sum(self):
        # 定义 Torch 脚本函数 func，用于测试 Torch 脚本编译和优化
        @torch.jit.script
        def func(x):
            return x.sum(dim=[4])

        @torch.jit.script
        def func2(x):
            return x.sum(dim=4)

        # 运行常量传播优化 pass
        self.run_pass('constant_propagation', func.graph)
        self.run_pass('constant_propagation', func2.graph)
        # 调用 _propagate_shapes 函数，验证 sum 函数的形状分析是否正确
        g = _propagate_shapes(func.graph, (torch.zeros(1, 1, 1, 1, 4),), False)
        g2 = _propagate_shapes(func2.graph, (torch.zeros(1, 1, 1, 1, 4),), False)
    def test_cat(self):
        # 在启用性能分析模式的测试中
        with enable_profiling_mode_for_profiling_tests():
            # 定义一个 TorchScript 函数 func，将输入张量 x 沿指定维度进行拼接
            @torch.jit.script
            def func(x):
                return torch.cat((x, x), dim=0)

            # 创建一个形状为 (10,) 的随机张量 x，类型为 float，启用梯度计算
            x = torch.rand(10, dtype=torch.float, requires_grad=True)
            # 断言 func(x, profile_and_replay=True) 的结果与 torch.cat((x, x), dim=0) 相等
            self.assertEqual(func(x, profile_and_replay=True), torch.cat((x, x), dim=0))

            # 定义一个带两个参数 x 和 y 的 TorchScript 函数 func2，将 x 沿 y 指定的维度进行拼接
            @torch.jit.script
            def func2(x, y):
                return torch.cat((x, x), y)

            # 在禁用自动微分子图内联的环境中
            with disable_autodiff_subgraph_inlining():
                # 遍历不同尺寸的输入张量 x
                for sizes in ((2, 2), (0, 2)):
                    # 创建形状为 sizes 的随机张量 x，要求梯度
                    x = torch.rand(sizes).requires_grad_()
                    # 创建张量 y，值为 1
                    y = torch.tensor(1)

                    # 调用 func2(x, y, profile_and_replay=True)，得到输出
                    output = func2(x, y, profile_and_replay=True)
                    # 创建参考输出 output_ref，通过 torch.cat((x, x), y) 得到
                    output_ref = torch.cat((x, x), y)
                    # 断言 output 和 output_ref 相等
                    self.assertEqual(output, output_ref)

                    # 如果 GRAPH_EXECUTOR 不等于 ProfilingMode.SIMPLE
                    if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                        # 断言 func2 对应的计算图包含 'aten::cat' 操作
                        self.assertAutodiffNode(func2.graph_for(x, y), True, ['aten::cat'], [])

                        # 计算 output 对 x 的梯度
                        grad = torch.autograd.grad(output.sum(), x)
                        # 计算 output_ref 对 x 的梯度作为参考
                        grad_ref = torch.autograd.grad(output_ref.sum(), x)
                        # 断言 grad 和 grad_ref 相等
                        self.assertEqual(grad, grad_ref)

    def test_cat_lifts(self):
        # 定义一个 TorchScript 函数 foo，将输入张量 x 沿指定维度进行拼接
        @torch.jit.script
        def foo(x):
            return torch.cat([x, x], dim=1)

        # 定义一个 TorchScript 函数 foo2，尝试在空列表上进行拼接
        @torch.jit.script
        def foo2(x):
            return torch.cat([], dim=1)

        # 定义一个 TorchScript 函数 foo3，将输入张量 x 沿指定维度进行拼接，使用单元素列表
        @torch.jit.script
        def foo3(x):
            return torch.cat([x], dim=1)

        # 遍历 foo, foo2, foo3 的计算图
        for g in [foo.graph, foo2.graph, foo3.graph]:
            # 使用 FileCheck 检查计算图 g 是否包含所需的模式
            FileCheck().check("int =").check("ListConstruct").check("aten::cat").run(str(g))

    def test_stack(self):
        # 在启用性能分析模式的测试中
        with enable_profiling_mode_for_profiling_tests():
            # 定义一个 TorchScript 函数 func，将输入张量 x 沿指定维度进行堆叠
            @torch.jit.script
            def func(x):
                return torch.stack((x, x), dim=1)
            # 创建形状为 (10, 10) 的随机张量 x
            x = torch.rand(10, 10)
            # 断言 func(x, profile_and_replay=True) 的结果与 torch.stack((x, x), dim=1) 相等
            self.assertEqual(func(x, profile_and_replay=True), torch.stack((x, x), dim=1))

            # 定义一个带两个参数 x 和 y 的 TorchScript 函数 func2，将输入张量 x 和 y 沿指定维度进行堆叠
            @torch.jit.script
            def func2(x, y):
                return torch.stack((x, y), dim=0)

            # 在禁用自动微分子图内联的环境中
            with disable_autodiff_subgraph_inlining():
                # 创建形状为 [2, 2] 的随机张量 x 和 y，要求梯度
                x = torch.randn([2, 2]).requires_grad_()
                y = torch.randn([2, 2]).requires_grad_()

                # 调用 func2(x, y, profile_and_replay=True)，得到输出
                output = func2(x, y, profile_and_replay=True)
                # 创建参考输出 output_ref，通过 torch.stack((x, y), 0) 得到
                output_ref = torch.stack((x, y), 0)
                # 断言 output 和 output_ref 相等
                self.assertEqual(output, output_ref)
                # 如果 GRAPH_EXECUTOR 不等于 ProfilingMode.SIMPLE
                if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                    # 断言 func2 对应的计算图包含 'aten::stack' 操作
                    self.assertAutodiffNode(func2.graph_for(x, y), True, ['aten::stack'], [])

                    # 计算 output 对 x 和 y 的梯度
                    grads = torch.autograd.grad(output.sum(), (x, y))
                    # 计算 output_ref 对 x 和 y 的梯度作为参考
                    grads_ref = torch.autograd.grad(output_ref.sum(), (x, y))
                    # 断言 grads 和 grads_ref 相等
                    self.assertEqual(grads, grads_ref)
    def test_unbind(self):
        # 在性能分析测试中启用性能模式
        with enable_profiling_mode_for_profiling_tests():
            # 使用 Torch JIT 脚本修饰符定义函数
            @torch.jit.script
            def func(x, y):
                # type: (Tensor, int) -> List[Tensor]
                # 调用 torch.unbind 函数解绑张量 x 沿着维度 y，返回张量列表
                return torch.unbind(x, y)

            # 禁用自动微分子图内联
            with disable_autodiff_subgraph_inlining():
                # 创建一个随机张量 x，并要求其梯度
                x = torch.rand([2, 2]).requires_grad_()
                y = 0
                # 调用 func 函数，使用 profile_and_replay=True 进行分析和重放
                outputs = func(x, y, profile_and_replay=True)
                # 使用 torch.unbind 函数解绑张量 x，维度为 y，并得到参考输出
                outputs_ref = torch.unbind(x, dim=y)
                # 断言 func 的输出与参考输出相等
                self.assertEqual(outputs, outputs_ref)
                # 断言 func 的计算图支持自动微分，无需传入任何梯度
                self.assertAutodiffNode(func.graph_for(x, y), True, [], [])

                # 计算 func 的输出相对于输入 x 的梯度
                grad = torch.autograd.grad(_sum_of_list(outputs), x)
                # 计算参考输出相对于输入 x 的梯度
                grad_ref = torch.autograd.grad(_sum_of_list(outputs_ref), x)
                # 断言 func 的输出梯度与参考输出梯度相等
                self.assertEqual(grad, grad_ref)


    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.PROFILING,
                     "Profiling executor fails to recognize that tensors in a list require gradients")
    def test_meshgrid(self):
        # 在性能分析测试中启用性能模式
        with enable_profiling_mode_for_profiling_tests():
            # 使用 Torch JIT 脚本修饰符定义函数
            @torch.jit.script
            def func(a):
                # type: (List[Tensor]) -> List[Tensor]
                # 调用 torch.meshgrid 函数，生成网格矩阵
                return torch.meshgrid(a)
            # 禁用自动微分子图内联
            with disable_autodiff_subgraph_inlining():
                # 创建两个张量 a 和 b，并要求其梯度
                a = torch.tensor([1.0, 2, 3]).requires_grad_()
                b = torch.tensor([1.0, 2, 3, 4]).requires_grad_()
                # 构建输入张量列表
                inputs = [a, b]

                # 使用 torch.meshgrid 函数生成参考输出
                outputs_ref = torch.meshgrid(inputs)
                # 调用 func 函数，使用 profile_and_replay=True 进行分析和重放
                outputs = func(inputs, profile_and_replay=True)
                # 断言 func 的输出与参考输出相等
                self.assertEqual(outputs, outputs_ref)

                # 如果不是简单模式，则断言 func 的计算图支持自动微分，无需传入任何梯度
                if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                    self.assertAutodiffNode(func.graph_for(inputs), True, [], [])

                    # 计算 func 的输出列表的总和相对于输入列表的梯度
                    grads = torch.autograd.grad(_sum_of_list(outputs), inputs)
                    # 计算参考输出列表的总和相对于输入列表的梯度
                    grads_ref = torch.autograd.grad(_sum_of_list(outputs_ref), inputs)
                    # 断言 func 的输出梯度与参考输出梯度相等
                    self.assertEqual(grads, grads_ref)

    def test_tensor_len(self):
        # 定义一个函数 func，返回输入张量 x 的长度
        def func(x):
            return len(x)

        # 对 func 函数进行脚本化检查，输入为包含一个张量的列表
        self.checkScript(func, [torch.ones(4, 5, 6)])

    def test_func_call(self):
        # 定义一个加法函数 add
        def add(a, b):
            return a + b

        # 定义一个乘法函数 mul
        def mul(a, x):
            return a * x

        # 定义一个复合函数 func，使用 add 和 mul 函数进行计算
        def func(alpha, beta, x, y):
            return add(mul(alpha, x), mul(beta, y))

        # 创建四个张量 alpha、beta、x 和 y，并要求其梯度
        alpha = torch.rand(1, dtype=torch.float, requires_grad=True)
        beta = torch.rand(1, dtype=torch.float, requires_grad=True)
        x = torch.rand(3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)

        # 注意：目前无法优化，因为融合器运行之前未插入广播
        # 对 func 函数进行脚本化检查，输入为 alpha、beta、x 和 y，不进行优化
        self.checkScript(func, [alpha, beta, x, y], optimize=False)

    @unittest.skip("bailouts are being deprecated")
    def test_profiling_graph_executor(self):
        @torch.jit.script
        def def_in_one_branch(x, z):
            # type: (Tensor, bool) -> float
            # 将输入的张量 x 赋值给 y
            y = x
            # 如果 z 是 False，则对 y 进行加一操作
            if z is False:
                y = x + 1

            # 返回 y 的元素和
            return y.sum()

        # 创建一个形状为 (2, 3) 的随机张量 a
        a = torch.rand(2, 3)

        # 在性能分析测试中启用性能分析模式
        with enable_profiling_mode_for_profiling_tests():
            # 检查是否插入了 "prim::profile"
            profiled_graph_str = str(def_in_one_branch.graph_for(a, True))
            FileCheck().check_count("prim::profile", 4).run(profiled_graph_str)
            
            # 对于形状为 (2, 3) 的张量 a，调用 def_in_one_branch 函数，并传入 False
            def_in_one_branch(a, False)
            
            # 将张量 a 的形状改变为 (3)，使其进入一个回退路径
            a = torch.ones(3)
            
            # 检查是否插入了 "prim::BailOut"
            bailout_graph_str = str(def_in_one_branch.graph_for(a, True))
            FileCheck().check_count("prim::BailOut", 3).run(bailout_graph_str)
            
            # 对形状为 (3) 的张量 a 调用 def_in_one_branch 函数，并传入 False，期望结果为 6.0
            self.assertEqual(def_in_one_branch(a, False), 6.0)
            
            # 对形状为 (3) 的张量 a 调用 def_in_one_branch 函数，并传入 True，期望结果为 3.0
            self.assertEqual(def_in_one_branch(a, True), 3.0)

    @unittest.skip("bailouts are being deprecated")
    def test_maxpool_guard_elimination(self):
        @torch.jit.script
        def my_maxpool(x):
            # 对输入张量 x 进行一维最大池化操作，然后加上形状为 [32, 32, 32] 的全为 1 的张量
            return F.max_pool1d(x, kernel_size=[1]) + torch.ones([32, 32, 32])

        # 创建一个形状为 (32, 32, 32) 的随机张量 a
        a = torch.rand(32, 32, 32)

        # 在性能分析测试中启用性能分析模式
        with enable_profiling_mode_for_profiling_tests():
            # 调用 my_maxpool 函数
            my_maxpool(a)
            # 检查是否插入了 "prim::BailOut"
            bailout_graph_str = str(my_maxpool.graph_for(a))
            FileCheck().check_count("prim::BailOut", 1).run(bailout_graph_str)

    @unittest.skip("bailouts are being deprecated")
    def test_slice_guard_elimination(self):
        @torch.jit.script
        def my_slice(x):
            # 返回 x 的切片 [0:16:2] 与自身切片 [0:16:2] 的和
            return x[0:16:2] + x[0:16:2]

        # 创建一个形状为 (32, 4) 的随机张量 a
        a = torch.rand(32, 4)

        # 在性能分析测试中启用性能分析模式
        with enable_profiling_mode_for_profiling_tests():
            # 调用 my_slice 函数
            my_slice(a)
            # 检查是否插入了 "prim::BailOut"
            bailout_graph_str = str(my_slice.graph_for(a))
            FileCheck().check_count("prim::BailOut", 1).run(bailout_graph_str)

    @unittest.skip("bailouts are being deprecated")
    def test_unsqueeze_guard_elimination(self):
        @torch.jit.script
        def my_unsqueeze(x):
            # 对输入张量 x 在维度 0 上进行两次 unsqueeze 操作，然后求和
            return torch.unsqueeze(x, 0) + torch.unsqueeze(x, 0)

        # 创建一个形状为 (32, 4) 的随机张量 a
        a = torch.rand(32, 4)

        # 在性能分析测试中启用性能分析模式
        with enable_profiling_mode_for_profiling_tests():
            # 调用 my_unsqueeze 函数
            my_unsqueeze(a)
            # 检查是否插入了两次 "prim::BailOut"
            bailout_graph_str = str(my_unsqueeze.graph_for(a))
            FileCheck().check_count("prim::BailOut", 2).run(bailout_graph_str)
    def test_resize_input_ops(self):
        # resize_ and resize_as resize the input tensor. because our shape analysis
        # is flow invariant, we set any Tensor that can alias a resized Tensor
        # to the base Tensor Type, without size information.

        # testing that value which is an input of a graph gets handled
        def out_op_graph_input():
            # 定义一个 Torch Script 函数 test，接受三个参数 x, y, z
            @torch.jit.script
            def test(x, y, z):
                # 使用 torch.mul 对 x 和 y 进行乘法操作，并将结果写入 z
                torch.mul(x, y, out=z)
                return z

            # 对 test 函数的计算图进行形状传播，传入三个张量作为输入
            graph = _propagate_shapes(test.graph,
                                      (torch.zeros(2, 1), torch.zeros(1, 2), torch.zeros(1, 1, 1)), False)
            # 断言图的输出类型为 TensorType 类型
            self.assertTrue(next(graph.outputs()).type() == TensorType.get())
        out_op_graph_input()

        def test_resize():
            # 定义一个 Torch Script 函数 test，接受一个参数 x
            @torch.jit.script
            def test(x):
                # 创建一个新的全零张量 after_resize_alias
                after_resize_alias = torch.zeros([2])
                # 循环5次
                for _i in range(5):
                    # 将 x 加上 1 赋给 b
                    b = x + 1
                    # 创建一个列表 f，包含元素 1
                    f = [1]
                    # 对 b 执行减法并返回结果给 before_resize_alias
                    before_resize_alias = b.sub_(1)
                    # 向列表 f 中添加元素 1
                    f.append(1)
                    # 调整张量 b 的大小为 f 的形状
                    b.resize_(f)
                    # 将 b 加上 1 赋给 after_resize_alias
                    after_resize_alias = b.add_(1)
                return after_resize_alias

            # 在 test 函数的计算图上运行常量传播优化
            self.run_pass('constant_propagation', test.graph)
            # 对 test 函数的计算图进行形状传播，传入一个全零张量作为输入
            g = _propagate_shapes(test.graph, (torch.zeros(1, 1),), False)
            # 查找计算图中的 aten::resize_ 节点
            resize_node = g.findNode("aten::resize_")
            # 断言 resize_ 节点的第一个输入和输出类型为 TensorType 类型
            self.assertTrue(next(resize_node.inputs()).type() == TensorType.get())
            self.assertTrue(next(resize_node.outputs()).type() == TensorType.get())

            # 查找计算图中的 aten::sub_ 节点
            before_resize = g.findNode("aten::sub_")
            # 断言 sub_ 节点的输出类型为 TensorType 类型
            self.assertTrue(next(before_resize.outputs()).type() == TensorType.get())

            # 查找计算图中的 aten::add_ 节点
            after_resize = g.findNode("aten::add_")
            # 断言 add_ 节点的输出类型为 TensorType 类型
            self.assertTrue(next(after_resize.outputs()).type() == TensorType.get())

        test_resize()

        def test_resize_as():
            # 定义一个 Torch Script 函数 test，接受一个参数 x
            @torch.jit.script
            def test(x):
                # 创建一个形状为 [2, 2] 的全零张量 b
                b = torch.zeros([2, 2])
                # 将张量 b 调整为与 x 相同的形状
                b.resize_as_(x)
                return b

            # 获取 test 函数的计算图
            g = test.graph
            # 在计算图上运行常量传播优化
            self.run_pass('constant_propagation', g)
            # 对 test 函数的计算图进行形状传播，传入一个全零张量作为输入
            g = _propagate_shapes(test.graph, (torch.zeros(1, 1),), False)

            # 断言输入的 x 不会被设置为基础张量类型
            self.assertTrue(next(g.inputs()).type() != TensorType.get())
            # 断言返回值被调整大小
            self.assertTrue(next(g.outputs()).type() == TensorType.get())

        test_resize_as()
    # 定义一个单元测试，测试未初始化的情况
    def test_uninitialized(self):
        # 定义一个表示图结构的字符串
        graph_str = """graph():
          %1 : int = prim::Uninitialized()  # 创建一个未初始化的整数节点 %1
          %2 : int = prim::Constant[value=1]()  # 创建一个常量整数节点 %2
          %3 : int = aten::add(%1, %2)  # 执行加法操作 %1 + %2，结果存储在 %3 中
          return (%3)  # 返回结果 %3
        """
        # 解析图结构字符串，生成图对象 g
        g = parse_ir(graph_str)
        # 根据图对象创建一个函数 m
        m = self.createFunctionFromGraph(g)
        # 获取 m 的导出和导入副本
        self.getExportImportCopy(m)
        # 断言执行 m() 时会抛出 RuntimeError，并且异常消息包含 "expected int"
        with self.assertRaisesRegex(RuntimeError, "expected int"):
            m()

    # 根据执行模式不同选择跳过测试
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.SIMPLE, "Simple Executor doesn't use requires_grad information")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.PROFILING, "Peeling is now disabled")
    def test_requires_grad_loop(self):
        # 定义一个 Torch 脚本函数 test，接受三个参数 x, y, z
        @torch.jit.script
        def test(x, y, z):
            # type: (Tensor, Tensor, int) -> Tensor
            # 循环 z 次
            for _ in range(z):
                x = y  # 将 y 赋值给 x
            return x  # 返回最终的 x

        # 定义测试输入 inps，其中第一个参数 x 需要梯度，第二个参数 y 不需要梯度
        inps = (torch.tensor(1.0, requires_grad=True), torch.tensor(1), 10)
        # 调用 test 函数，并进行分析和回放
        test(*inps, profile_and_replay=True)

        # 获取 test 函数的计算图
        graph = test.graph_for(*inps)
        # 查找计算图中的循环节点 "prim::Loop"
        loop = graph.findNode("prim::Loop")
        # 获取循环体的第一个块
        loop_body = next(loop.blocks())
        # 获取循环的输入
        loop_inputs = list(loop_body.inputs())
        # 获取循环的输出
        loop_outputs = list(loop_body.outputs())

        # 根据执行模式选择不同的断言
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            # 在 PROFILING 模式下，验证特定的优化图结构
            # 优化后的图结构中有 3 个循环
            # 原始循环被展开
            # 展开的循环也被展开
            index_of_x_in_peeled_unrolled_loop = -2
            # 断言在展开的优化循环中 x 需要梯度
            self.assertTrue(loop_inputs[index_of_x_in_peeled_unrolled_loop].requires_grad())
            # 查找所有的 "prim::BailOut" 节点，不包括外部块
            bailouts_in_outer_block = graph.findAllNodes("prim::BailOut", False)
            last_bailout_index_on_loops_output = -1
            # 断言最后一个 "prim::BailOut" 节点的输出不需要梯度
            self.assertFalse(bailouts_in_outer_block[last_bailout_index_on_loops_output].output().requires_grad())
        else:
            # 在其他模式下，普通地验证循环的输入和输出的梯度情况
            self.assertTrue(loop_inputs[1].requires_grad())
            self.assertTrue(loop.output().requires_grad())
            self.assertFalse(loop_outputs[1].requires_grad())

    # 定义一个测试视图形状属性的函数
    def test_view_shape_prop(self):
        # 创建一个 Torch 编译单元 cu，包含一个测试函数 test_view_shape_prop
        cu = torch.jit.CompilationUnit('''
        def test_view_shape_prop(a):
            return a.view(size=[-1])  # 对输入 a 进行形状重塑，变成一维数组
        ''')
        inputs = [torch.zeros(10, 10)]  # 创建一个 10x10 的全零张量作为输入
        outputs = torch.zeros(100)  # 创建一个形状为 100 的全零张量作为预期输出

        # 执行 cu 中的 test_view_shape_prop 函数，并获得实际输出 real_outs
        real_outs = cu.test_view_shape_prop(*inputs)
        # 断言实际输出与预期输出相等
        self.assertEqual(real_outs, outputs)

    # 根据 TorchDynamo 的问题跳过测试
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    # 定义测试函数，用于验证视图构造形状属性
    def test_view_listconstruct_shape_prop(self):
        # 定义内部函数 fn，接收参数 x
        def fn(x):
            # 获取张量 x 的大小信息
            B = x.size(0)
            C = x.size(1)
            T = x.size(2)
            # 调整张量 x 的视图形状为 (T, B, C)，并返回
            return x.view(T, B, C)

        # 创建一个形状为 (3, 1, 5) 的随机张量 x，要求梯度
        x = torch.randn(3, 1, 5, requires_grad=True)
        # 将函数 fn 编译为 TorchScript
        fn = torch.jit.script(fn)
        # 使用 _propagate_shapes 函数推断图形的形状，传入参数 x，返回的图形不包含多余信息
        graph = _propagate_shapes(fn.graph, (x,), False)
        # 断言图形输出的类型标量为 'Float'
        self.assertTrue(next(graph.outputs()).type().scalarType() == 'Float')

    # 定义测试函数，用于验证形状推断的提升
    def test_shape_prop_promotion(self):
        # 定义 TorchScript 函数 fn，接收参数 x 和 y
        @torch.jit.script
        def fn(x, y):
            # 返回 x 和 y 的加法结果
            return x + y

        # 创建两个张量 x 和 y，形状都为 (3, 4)，分别为 float 和 double 类型
        x, y = torch.rand(3, 4, dtype=torch.float), torch.rand(3, 4, dtype=torch.double)
        # 使用 _propagate_shapes 函数推断图形的形状，传入参数 x 和 y，返回的图形不包含多余信息
        graph = _propagate_shapes(fn.graph, (x, y), False)
        # 运行 FileCheck 检查，验证图形是否包含 'Double(*, *, device=cpu) = aten::add' 的内容
        FileCheck().check('Double(*, *, device=cpu) = aten::add').run(graph)

    # 定义测试函数，用于验证标量参数的形状推断提升
    def test_shape_prop_promote_scalar_arg(self):
        # 定义 TorchScript 函数 fn，接收参数 x
        @torch.jit.script
        def fn(x):
            # 返回数学常数 pi 与 x 的加法结果
            return math.pi + x

        # 创建一个形状为 (3, 4)，类型为 long 的零张量 x
        x = torch.zeros(3, 4, dtype=torch.long)
        # 使用 _propagate_shapes 函数推断图形的形状，传入参数 x，返回的图形不包含多余信息
        graph = _propagate_shapes(fn.graph, (x,), False)
        # 获取默认的张量类型
        default = torch.get_default_dtype()
        # 根据默认类型运行 FileCheck 检查，验证图形是否包含正确的类型和操作
        if default == torch.float:
            FileCheck().check('Float(*, *, requires_grad=0, device=cpu) = aten::add').run(graph)
        else:
            FileCheck().check('Double(*, *, requires_grad=0, device=cpu) = aten::add').run(graph)

    # 定义测试函数，用于验证整数形状推断
    def test_integral_shape_inference(self):
        # 创建 TorchScript 编译单元 cu，包含测试函数 test_integral_shape_inference(a)
        cu = torch.jit.CompilationUnit('''
        def test_integral_shape_inference(a):
            return a * a
        ''')
        # 创建一个形状为 (10, 10)，类型为 long 的全 1 张量作为输入
        inputs = [torch.ones(10, 10, dtype=torch.long)]
        # 创建一个形状为 (10, 10)，类型为 long 的全 1 张量作为期望输出
        outputs = torch.ones(10, 10, dtype=torch.long)

        # 断言调用 cu 的 test_integral_shape_inference 函数产生的结果与预期输出一致
        self.assertEqual(cu.test_integral_shape_inference(*inputs), outputs)

    # 标记测试函数，用于测试 CPU 上的批量归一化融合
    @unittest.skipIf(RUN_CUDA, 'This tests the CPU fuser')
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    @enable_cpu_fuser
    def test_batchnorm_fuser_cpu(self):
        # 定义包含 IR 代码的字符串 code
        code = '''
            graph(%3 : Tensor,
                  %7 : Tensor,
                  %12 : Float(*, *),
                  %13 : Tensor,
                  %25 : Tensor):
                %23 : int = prim::Constant[value=1]()
                %22 : float = prim::Constant[value=1e-05]()
                %26 : Tensor = aten::sqrt(%25)
                %24 : Tensor = aten::add(%26, %22, %23)
                %20 : Tensor = aten::reciprocal(%24)
                %norm_invstd : Tensor = aten::mul(%20, %23)
                %15 : Tensor = aten::sub(%12, %13, %23)
                %11 : Tensor = aten::mul(%15, %norm_invstd)
                %8 : Tensor = aten::mul(%11, %7)
                %5 : Tensor = aten::add(%8, %3, %23)
                %1 : Float(*, *) = aten::relu(%5)
                return (%1)
        '''

        # 解析 IR 代码，得到图形对象 graph
        graph = parse_ir(code)
        # 创建包含 5 个形状为 (26, 2048)，类型为 float 的随机张量作为输入
        inputs = 5 * [torch.rand(26, 2048, dtype=torch.float)]
        # 获取融合内核代码
        code = torch._C._jit_fuser_get_fused_kernel_code(graph, inputs)
        # 运行 FileCheck 检查，验证融合内核代码中是否包含 'sqrtf' 的内容
        FileCheck().check('sqrtf').run(code)

    # 标记为慢速测试，并跳过 CUDA 测试和 Sandcastle 上的测试
    @slowTest
    @unittest.skipIf(RUN_CUDA, 'This tests the CPU fuser')
    # 如果在Sandcastle环境中运行，则跳过测试，因为暂时不支持fuser
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    # 启用CPU fuser进行测试
    @enable_cpu_fuser
    # 测试双精度和单精度代码生成
    def test_fuser_double_float_codegen(self):
        # 定义需要测试的数学函数列表
        fns = ['log', 'log10', 'log1p', 'log2', 'lgamma', 'exp', 'expm1', 'erf',
               'erfc', 'cos', 'acos', 'cosh', 'sin', 'asin', 'sinh', 'tan',
               'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'round', 'trunc',
               'frac']

        # 查找C语言等效的函数
        def lookup_c_equivalent_fn(aten_fn):
            return aten_fn

        # 定义测试分发函数
        def test_dispatch(op, expects, dtype, binary=False):
            # 根据数据类型选择对应的字符串表示
            if dtype == torch.double:
                dtype_str = 'Double'
            elif dtype == torch.float:
                dtype_str = 'Float'
            else:
                raise RuntimeError('Unknown dtype')

            # 根据是否二元操作选择不同的代码块
            if binary:
                code = f'''
                    graph(%3 : Tensor, %4 : Tensor):
                        %2 : {dtype_str}(*, *) = aten::{op}(%3, %4)
                        %1 : {dtype_str}(*, *) = aten::relu(%2)
                        return (%1)
                '''
            else:
                code = f'''
                    graph(%3 : Tensor):
                        %2 : {dtype_str}(*, *) = aten::{op}(%3)
                        %1 : {dtype_str}(*, *) = aten::relu(%2)
                        return (%1)
                '''

            # 解析IR代码生成图
            graph = parse_ir(code)
            # 创建随机输入
            inputs = (2 if binary else 1) * [torch.rand(26, 2048, dtype=dtype)]
            # 获取融合内核代码
            code = torch._C._jit_fuser_get_fused_kernel_code(graph, inputs)
            # 使用FileCheck检查期望的结果
            FileCheck().check(expects).run(code)

        # 对每个函数进行测试分发
        for fn in fns:
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + '(', torch.double)
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + 'f(', torch.float)

        # 'min', 'max'曾经进行过测试，但现在用三元表达式替代了fmin()和fmax()
        # 对于二元函数列表进行测试
        binary_fns = ['pow']
        for fn in binary_fns:
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + '(', torch.double, binary=True)
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + 'f(', torch.float, binary=True)

    # 如果在CUDA环境中运行，则跳过测试，因为此测试是CPU fuser相关的
    @unittest.skipIf(RUN_CUDA, 'This tests the CPU fuser')
    # 如果在Sandcastle环境中运行，则跳过测试，因为暂时不支持fuser
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    # 启用CPU fuser进行测试
    @enable_cpu_fuser
    # 测试双精度文字精度
    def test_fuser_double_literal_precision(self):
        # 定义IR代码
        code = '''
        graph(%2 : Float(*, *)):
            %4 : int = prim::Constant[value=1]()
            %3 : float = prim::Constant[value=1.282549830161864]()
            %5 : Float(*, *) = aten::add(%2, %3, %4)
            %1 : Float(*, *) = aten::relu(%5)
            return (%1)
        '''

        # 解析IR代码生成图
        graph = parse_ir(code)
        # 获取融合内核代码
        code = torch._C._jit_fuser_get_fused_kernel_code(graph, [torch.rand(3, 4)])
        # 使用FileCheck检查特定的文字精度
        FileCheck().check('1.282549830161864').run(code)
    def test_fuser_multiple_blocks(self):
        # 使用 torch.jit.CompilationUnit 创建一个 JIT 编译单元，定义一个测试函数 test_fuser_multiple_blocks
        cu = torch.jit.CompilationUnit('''
        def test_fuser_multiple_blocks(this, that, theother, meme):
            i = 0
            while i < 20:
                # 使用 torch.cat 在指定维度上拼接张量 this 和 meme
                this = torch.cat([this, meme], dim=0)
                that = torch.cat([that, meme], dim=0)
                theother = torch.cat([theother, meme], dim=0)
                i = i + 1
            # 返回拼接后的张量 this, that, theother
            return this, that, theother
        ''')

        # 创建测试用例的输入张量列表，均为形状为 (0, 10, 10) 的张量
        inputs = [torch.ones(0, 10, 10)] * 3
        # 将形状为 (1, 10, 10) 的张量添加到输入列表
        inputs += [torch.ones(1, 10, 10)]
        # 创建期望的输出张量列表，均为形状为 (20, 10, 10) 的张量
        outputs = [torch.ones(20, 10, 10)] * 3

        # 断言调用 cu.test_fuser_multiple_blocks(*inputs) 返回期望的 outputs
        self.assertEqual(cu.test_fuser_multiple_blocks(*inputs), outputs)

    @unittest.skip("RuntimeError: VariableType::ID() not implemented")
    def test_cast(self):
        # 定义一个脚本，将输入 x 转换为整型数
        script = '''
        def to_int(x):
            return int(x)
        '''
        # 创建一个浮点类型张量 x，并标记需要梯度
        x = Variable(torch.FloatTensor([1.1, 2.3]), requires_grad=True)
        # 创建一个整型张量 out，并标记需要梯度
        out = Variable(torch.IntTensor([1, 2]), requires_grad=True)
        # 调用 self.checkScript 检查脚本执行结果，预期输出为 out
        self.checkScript(script, [x], optimize=True, outputs=[out], func='to_int')

    def test_str_cast(self):
        # 定义一个脚本函数 to_str，将整数 x 转换为字符串
        @torch.jit.script
        def to_str(x):
            # type: (int) -> str
            return str((x, x))

        # 断言调用 to_str(1) 返回字符串 "(1, 1)"
        self.assertEqual("(1, 1)", to_str(1))

    def test_int_cast(self):
        # 定义一个脚本函数 to_int，将字符串 x 转换为整型数
        @torch.jit.script
        def to_int(x):
            # type: (str) -> int
            return int(x)

        # 断言不同输入字符串下的转换结果是否符合预期
        self.assertEqual(5, to_int('5'))
        self.assertEqual(-5, to_int('-5'))
        self.assertEqual(2147483647, to_int('2147483647'))
        self.assertEqual(-2147483648, to_int('-2147483648'))

        # 断言对于不合法的字符串输入，to_int 函数是否能够正确引发异常
        with self.assertRaisesRegex(RuntimeError, "invalid literal for int()"):
            to_int('0x20')

        with self.assertRaisesRegex(RuntimeError, "invalid literal for int()"):
            to_int('0b0001')

    def test_python_frontend(self):
        # 定义一个 Python 函数 fn，包含多种操作并使用 Torch JIT 前端获取其定义
        def fn(x, y, z):
            q = None
            q = x + y - z.sigmoid()
            print(q)
            w = -z
            if not x and not y and z:
                m = x if not z else y
            while x < y > z:
                q = x
            # 使用 assert 语句检查条件，如果条件不满足则抛出 AssertionError
            assert 1 == 1, "hello"
            # 返回参数 x
            return x

        # 获取函数 fn 的 JIT 定义并断言其输出符合预期
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        self.assertExpected(str(ast))

    def test_python_frontend_source_range(self):
        # 定义一个引发异常的 Python 函数 fn，并获取其源码范围信息
        def fn():
            raise Exception("hello")  # noqa: TRY002
        # 获取函数 fn 的 JIT 定义并检查其源码范围信息是否符合预期
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        FileCheck().check("SourceRange at:") \
                   .check("def fn():") \
                   .check("~~~~~~~~~") \
                   .check('raise Exception("hello")') \
                   .check('~~~~~~~~~~~~~~~~~ <--- HERE') \
                   .run(str(ast.range()))

    def test_python_frontend_py3(self):
        # 定义一个引发异常的 Python 函数 fn，并获取其 JIT 定义，用于 Python 3
        def fn():
            raise Exception("hello")  # noqa: TRY002
        # 获取函数 fn 的 JIT 定义并断言其输出符合预期
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        self.assertExpected(str(ast))
    # 定义一个函数 `_make_scalar_vars`，用于将给定数组 `arr` 中的每个元素转换为指定数据类型 `dtype` 的 Torch 张量，并返回转换后的张量列表
    def _make_scalar_vars(self, arr, dtype):
        return [torch.tensor(val, dtype=dtype) for val in arr]


    # 测试函数 `test_string_print`
    def test_string_print(self):
        # 定义内部函数 `func`，接受参数 `a`，打印参数及其后的字符串和数字，然后返回参数 `a`
        def func(a):
            print(a, "a" 'b' '''c''' """d""", 2, 1.5)
            return a

        # 使用 `_make_scalar_vars` 函数创建输入参数 `inputs`，包含一个 Torch int64 类型的张量列表
        inputs = self._make_scalar_vars([1], torch.int64)
        # 调用对象的 `checkScript` 方法，传入函数 `func`、输入参数 `inputs`，并捕获输出
        self.checkScript(func, inputs, capture_output=True)


    # 测试函数 `test_while`
    def test_while(self):
        # 定义内部函数 `func`，接受参数 `a`、`b`、`max`
        def func(a, b, max):
            # 当 `a` 小于 `max` 时执行循环
            while bool(a < max):
                a = a + 1
                b = b + 1
            # 计算 `c`，返回 `a` 和 `b` 的和
            c = a + b
            return c

        # 使用 `_make_scalar_vars` 函数创建输入参数 `inputs`，包含一个 Torch int64 类型的张量列表
        inputs = self._make_scalar_vars([1, 1, 10], torch.int64)
        # 调用对象的 `checkScript` 方法，传入函数 `func`、输入参数 `inputs`，并启用优化
        self.checkScript(func, inputs, optimize=True)


    # 测试函数 `test_fibb`
    def test_fibb(self):
        # 定义内部函数 `func`，接受参数 `lim`
        def func(lim):
            # 初始化变量 `first`、`second`、`i`、`somenum`、`dontmutateme`、`third`
            first = 1
            second = 1
            i = 1
            somenum = 5
            dontmutateme = 3
            third = 0
            # 当 `i` 小于 `lim` 时执行外部循环
            while bool(i < lim):
                # 计算 `third`、更新 `first` 和 `second`
                third = first + second
                first = second
                second = third
                j = 0
                # 当 `j` 小于 10 时执行内部循环
                while j < 10:
                    somenum = somenum * 2
                    j = j + 1
                i = i + j
                i = i + dontmutateme

            # 计算 `st`、`fs`，返回 `third`、`st`、`fs` 的元组
            st = second + third
            fs = first + second
            return third, st, fs

        # 使用 `_make_scalar_vars` 函数创建输入参数 `inputs`，包含一个 Torch int64 类型的张量列表
        inputs = self._make_scalar_vars([10], torch.int64)
        # 调用对象的 `checkScript` 方法，传入函数 `func`、输入参数 `inputs`，并启用优化
        self.checkScript(func, inputs, optimize=True)


    # 测试函数 `test_fibb_totally_better`
    def test_fibb_totally_better(self):
        # 定义函数 `fib`，接受一个整数参数 `x`，返回 Fibonacci 数列的第 `x` 项
        def fib(x):
            # 初始化变量 `prev`、`v`
            prev = 1
            v = 1
            # 执行 `x` 次循环
            for i in range(0, x):
                save = v
                v = v + prev
                prev = save
            return v

        # 调用对象的 `checkScript` 方法，传入函数 `fib` 和参数 `(10,)`
        self.checkScript(fib, (10,))


    # 测试函数 `test_if`
    def test_if(self):
        # 定义内部函数 `func`，接受参数 `a`、`b`
        def func(a, b):
            # 初始化变量 `d`
            d = 3
            # 若 `a` 大于 10 则执行条件语句
            if bool(a > 10):
                a = 3 + d
            else:
                b = 3 + d
                d = 4
            # 计算 `c`，返回 `a` 和 `b` 的和
            c = a + b
            return c

        # 使用 `_make_scalar_vars` 函数创建输入参数 `inputs`，包含一个 Torch int64 类型的张量列表
        inputs = self._make_scalar_vars([1, -1], torch.int64)
        # 调用对象的 `checkScript` 方法，传入函数 `func`、输入参数 `inputs`，并启用优化
        self.checkScript(func, inputs, optimize=True)


    # 测试函数 `test_if_for_in_range`
    def test_if_for_in_range(self):
        # 定义内部函数 `func`，接受参数 `a`、`b`
        def func(a, b):
            # 初始化变量 `d`
            d = 3
            # 执行 20 次循环
            for _ in range(20):
                # 若 `a` 大于 10 则执行条件语句
                if bool(a > 10):
                    a = 3 + d
                else:
                    b = 3 + d
                    d = 4
                # 计算 `c`，返回 `a` 和 `b` 的和
                c = a + b
            return d

        # 使用 `_make_scalar_vars` 函数创建输入参数 `inputs`，包含一个 Torch int64 类型的张量列表
        inputs = self._make_scalar_vars([1, -1], torch.int64)
        # 调用对象的 `checkScript` 方法，传入函数 `func`、输入参数 `inputs`，并启用优化
        self.checkScript(func, inputs, optimize=True)


    # 测试函数 `test_if_noelse`
    def test_if_noelse(self):
        # 定义内部函数 `func`，接受参数 `a`、`b`
        def func(a, b):
            # 若 `a` 大于 10 则执行条件语句
            if bool(a > 10):
                a = 3 + b
            # 计算 `c`，返回 `a` 和 `b` 的和
            c = a + b
            return c

        # 使用 `_make_scalar_vars` 函数创建输入参数 `inputs`，包含一个 Torch int64 类型的张量列表
        inputs = self._make_scalar_vars([-1, 1], torch.int64)
        # 调用对象的 `checkScript` 方法，传入函数 `func`、输入参数 `inputs`，并启用优化
        self.checkScript(func, inputs, optimize=True)
    def test_conditional_casting(self):
        # 定义测试布尔类型张量转换函数
        def test_bool_cast_tensor(x):
            # 如果 x 是真值，则返回 1，否则返回 0
            if x:
                return 1
            else:
                return 0

        # 遍历两种不同的输入条件
        for make_one_dim in [True, False]:
            # 遍历不同的输入值
            for inp_val in [0.1, 0.0, -0.0, -0.1, -1, 0, 1]:
                # 如果 make_one_dim 为 True，则将输入值转换为单元素列表
                inp_val = [inp_val] if make_one_dim else inp_val
                # 调用 self.checkScript 来检查 test_bool_cast_tensor 函数的脚本化版本的行为
                self.checkScript(test_bool_cast_tensor, (torch.tensor(inp_val),))

        # 使用 self.checkScriptRaisesRegex 检查对包含多个值的张量的调用是否引发异常
        self.checkScriptRaisesRegex(test_bool_cast_tensor, (torch.tensor([1, 1]),), Exception,
                                    "Boolean value of Tensor with more than one value")

        # 定义测试逻辑非操作的函数
        def test_not_cast(x):
            # 如果 x 是假值，则返回 1，否则返回 0
            if not x:
                return 1
            else:
                return 0

        # 调用 self.checkScript 来检查 test_not_cast 函数的脚本化版本的行为
        self.checkScript(test_not_cast, (torch.tensor(1),))
        self.checkScript(test_not_cast, (torch.tensor(0),))

        # 使用 self.assertRaisesRegex 检查是否正确引发特定类型的异常
        with self.assertRaisesRegex(RuntimeError, r"Could not cast value of type Tuple\[Tensor, Tensor\]"):  # noqa: W605
            @torch.jit.script
            def test_mult(x, y):
                # 返回 x 和 y 的逻辑非结果
                return not (x, y)

        # 定义测试整数类型转换的函数
        def test_cast_int(x):
            # type: (int) -> int
            # 如果 x 是真值，则返回 1，否则返回 0
            if x:
                return 1
            else:
                return 0

        # 调用 self.checkScript 来检查 test_cast_int 函数的脚本化版本的行为
        self.checkScript(test_cast_int, (1,))
        self.checkScript(test_cast_int, (0,))
        self.checkScript(test_cast_int, (-1,))

        # 定义测试浮点数类型转换的函数
        def test_cast_float(x):
            # type: (float) -> int
            # 如果 x 是真值，则返回 1，否则返回 0
            if x:
                return 1
            else:
                return 0

        # 调用 self.checkScript 来检查 test_cast_float 函数的脚本化版本的行为
        self.checkScript(test_cast_float, (1.,))
        self.checkScript(test_cast_float, (0.,))
        self.checkScript(test_cast_float, (-1.,))

        # 使用 self.assertRaisesRegex 检查是否正确引发特定类型的异常
        with self.assertRaisesRegex(RuntimeError, r"Could not cast value of type Tuple\[int, int\] to bool"):  # noqa: W605
            @torch.jit.script
            def test_bad_conditional(x):
                # 如果 (1, 2) 是真值，则返回 None，否则返回 0
                if (1, 2):  # noqa: F634
                    return
                else:
                    return 0

    def test_while_nonexistent_value(self):
        # 使用 self.assertRaisesRegex 检查是否正确引发特定类型的异常
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            # 定义一个包含错误的 while 循环的脚本
            torch.jit.CompilationUnit('''
            def test_while(a, b):
                while bool(a < 10):
                    a = a + x
                    b = b + 1
                return a + b
            ''')

    def test_assertion_optional_refinement(self):
        # 定义带有断言的脚本函数
        @torch.jit.script
        def test(x, y):
            # type: (Optional[int], Optional[int]) -> int
            # 断言 x 和 y 非空，否则抛出异常
            assert x is not None and y is not None
            # 返回 x 和 y 的和
            return x + y

        # 调用 self.assertEqual 检查 test 函数的行为是否符合预期
        self.assertEqual(test(2, 2), 4)
        # 使用 self.assertRaisesRegex 检查是否正确引发特定类型的异常
        with self.assertRaisesRegex(Exception, ""):
            test(1, None)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "the current version of Profiler doesn't profile/specialize Optionals")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "the current version of Profiler doesn't profile/specialize Optionals")
    # 定义测试函数，测试可选的张量参数
    def test_optional_tensor(self):
        @torch.jit.script
        def fn(x, y):
            # type: (Optional[Tensor], int) -> int
            # 如果 x 是 None，则返回 y
            if x is None:
                return y
            else:
                return 0

        # 测试函数调用，期望返回 1
        res = fn(None, 1)
        self.assertEqual(res, 1)
        # 获取最后一次执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 获取第一个输入
        first_input = next(g.inputs())
        # 检查输入是否被断开连接
        self.assertEqual(first_input.type().kind(), 'OptionalType')
        self.assertEqual(first_input.uses(), [])
        
        # 创建一个张量
        t = torch.ones(1)
        # 测试函数调用，期望返回 0
        res = fn(t, 1)
        self.assertEqual(res, 0)
        # 获取最后一次执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 检查第一个输入的类型是否为张量类型
        self.assertEqual(next(g.inputs()).type().kind(), 'TensorType')

        @torch.jit.script
        def fn(x, y, b):
            # type: (Optional[Tensor], Tensor, bool) -> Tensor
            # 如果 b 为真，则返回 y；否则解开 x 的可选包装并返回
            if b:
                res = y
            else:
                res = torch.jit._unwrap_optional(x)
            return res

        # 创建另一个张量
        t2 = torch.zeros(1)
        # 测试函数调用，期望返回 t2
        res = fn(t, t2, True)
        self.assertEqual(res, t2)
        # 预期抛出异常，因为尝试解开 null 可选包装
        with self.assertRaisesRegex(RuntimeError, "Unwrapping null optional"):
            res = fn(None, t2, False)
        # 测试函数调用，期望返回 t2
        res = fn(None, t2, True)
        # 获取最后一次执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 检查输出的类型字符串是否为 "Tensor" 或 "Tensor(requires_grad=1)"
        self.assertIn(next(g.outputs()).type().str(), ("Tensor", "Tensor(requires_grad=1)"))

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "the current version of Profiler doesn't profile/specialize Optionals")
    # 定义测试函数，测试可选的列表参数
    def test_optional_list(self):
        @torch.jit.script
        def fn(x, y):
            # type: (Optional[List[int]], int) -> int
            # 如果 x 是 None，则返回 y；否则返回列表元素的总和
            if x is None:
                return y
            else:
                res = 0
                for d in x:
                    res += d
                return res

        # 测试函数调用，期望返回 1
        res = fn(None, 1)
        self.assertEqual(res, 1)
        # 获取最后一次执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 获取第一个输入
        first_input = next(g.inputs())
        # 检查输入是否被断开连接
        self.assertEqual(first_input.type().kind(), 'OptionalType')
        self.assertEqual(first_input.uses(), [])
        
        # 创建一个列表
        l = [2, 3]
        # 测试函数调用，期望返回列表元素的总和 5
        res = fn(l, 1)
        self.assertEqual(res, 5)
        # 获取最后一次执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 检查第一个输入的类型是否为列表类型
        self.assertEqual(next(g.inputs()).type().kind(), 'ListType')

        @torch.jit.script
        def fn(x, y, b):
            # type: (Optional[List[int]], List[int], bool) -> List[int]
            # 如果 b 为真，则解开 x 的可选包装；否则返回 y
            if b:
                l = torch.jit._unwrap_optional(x)
            else:
                l = y
            return l

        # 创建另一个列表
        l2 = [0, 1]
        # 测试函数调用，期望返回 l
        res = fn(l, l2, True)
        self.assertEqual(res, l)
        # 预期抛出异常，因为尝试解开 null 可选包装
        with self.assertRaisesRegex(RuntimeError, "Unwrapping null optional"):
            res = fn(None, l2, True)
        # 测试函数调用，期望返回 l2
        res = fn(None, l2, False)
        # 获取最后一次执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        # 检查输出的类型字符串是否为 "int[]"
        self.assertEqual(next(g.outputs()).type().str(), "int[]")
    def test_alias_covariant_type_containers(self):
        @torch.jit.script
        def foo(x):
            # 定义函数 foo，参数 x 的类型注释为 bool
            # 如果 x 为 True，将 a 设为包含 None 的元组
            # 如果 x 为 False，将 a 设为包含空列表的元组
            if x:
                a = (None,)
            else:
                a = ([],)
            return a

        @torch.jit.script
        def foo2(x, li):
            # 定义函数 foo2，参数 x 的类型注释为 bool，参数 li 的类型注释为 Tuple[Optional[List[Tensor]]]
            # 如果 x 为 True，将 li 设为包含 None 的元组
            # 返回参数 li
            if x:
                li = (None,)
            return li

    def test_while_write_outer_then_read(self):
        # 定义函数 func，参数 a 和 b 为整数，在 a 小于 10 的条件下执行循环
        def func(a, b):
            while bool(a < 10):
                # 在循环中，a 和 b 分别加 1
                a = a + 1
                b = a + 1
            # 返回 a + b 的结果
            return a + b

        # 生成输入变量，类型为 torch.int64，分别为 42 和 1337
        inputs = self._make_scalar_vars([42, 1337], torch.int64)
        # 调用 self.checkScript 函数对 func 进行脚本化检查，并进行优化
        self.checkScript(func, inputs, optimize=True)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_while_nest_if(self):
        # 定义函数 func，参数 a 和 b 为整数，返回值类型为整数
        def func(a, b):
            c = 0
            # 在 a 小于 10 的条件下执行循环
            while a < 10:
                # 在循环中，a 和 b 分别加 1
                a = a + 1
                b = b + 1
                # 如果 a 大于 b，则将 c 设为 -a，否则设为 -b
                if a > b:
                    c = -a
                else:
                    c = -b
            # 返回 c + 1 的结果
            return c + 1

        # 生成输入变量，类型为 torch.int64，分别为 -1234 和 4321
        inputs = self._make_scalar_vars([-1234, 4321], torch.int64)
        # 调用 self.checkScript 函数对 func 进行脚本化检查，并进行优化
        self.checkScript(func, inputs, optimize=True)
    def test_divmod(self):
        def func_int(a, b):
            # 定义一个函数，用于计算两个整数的除法和取余操作，并返回结果的元组
            # 参数类型：a为整数，b为整数
            return divmod(a, b)

        def func_float(a, b):
            # 定义一个函数，用于计算两个浮点数的除法和取余操作，并返回结果的元组
            # 参数类型：a为浮点数，b为浮点数
            return divmod(a, b)

        def func_int_float(a, b):
            # 定义一个函数，用于计算一个整数和一个浮点数的除法和取余操作，并返回结果的元组
            # 参数类型：a为整数，b为浮点数
            return divmod(a, b)

        def func_float_int(a, b):
            # 定义一个函数，用于计算一个浮点数和一个整数的除法和取余操作，并返回结果的元组
            # 参数类型：a为浮点数，b为整数
            return divmod(a, b)

        def divmod_test_iterator(func, num, den):
            # 通过迭代器测试四种不同的除法和取余函数
            for i in num:
                for j in den:
                    self.checkScript(func, (i, j), frames_up=2)

        num_int = [1024, -1024]
        den_int = [10, -10]
        num_float = [5.3, -5.3]
        den_float = [2.0, -2.0]
        # 对四种函数类型进行迭代测试
        divmod_test_iterator(func_int, num_int, den_int)
        divmod_test_iterator(func_float, num_float, den_float)
        divmod_test_iterator(func_int_float, num_int, den_float)
        divmod_test_iterator(func_float_int, num_float, den_int)

        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError: integer division or modulo by zero"):
            # 使用断言验证整数除法函数在除零情况下是否抛出正确异常
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_int)))
            cu.func_int(1024, 0)
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError: float divmod()"):
            # 使用断言验证浮点数除法函数在除零情况下是否抛出正确异常
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_float)))
            cu.func_float(5.3, 0.0)
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError: float divmod()"):
            # 使用断言验证整数与浮点数混合运算函数在除零情况下是否抛出正确异常
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_int_float)))
            cu.func_int_float(1024, 0.0)
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError: float divmod()"):
            # 使用断言验证浮点数与整数混合运算函数在除零情况下是否抛出正确异常
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_float_int)))
            cu.func_float_int(5.3, 0)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_if_nest_while(self):
        def func(a, b):
            # 定义一个函数，用于比较两个整数，并在条件满足时执行循环操作
            # 参数类型：a为整数，b为整数
            c = 0
            if a > b:
                while a > b:
                    b = b + 1
                    c = -b
            return c

        inputs = self._make_scalar_vars([4321, 1234], torch.int64)
        # 对定义的比较与循环函数进行脚本验证
        self.checkScript(func, inputs)
    # 定义一个测试函数，用于测试在参数为 None 的情况下的返回值
    def test_script_optional_none(self):
        # 定义一个函数 none_stmt，接受一个参数 x，将其赋值给 output 并返回
        def none_stmt(x):
            output = None  # 初始化 output 为 None
            output = x  # 将参数 x 赋值给 output
            return output

        # 定义一个函数 none_args，接受一个参数 x，类型为 Optional[Tensor]，返回值类型也为 Optional[Tensor]
        def none_args(x):
            return None  # 直接返回 None

        # 调用自定义函数 checkScript，分别测试 none_stmt 和 none_args 函数的 Torch 脚本化结果
        self.checkScript(none_stmt, [torch.arange(0, 2)], optimize=True)
        self.checkScript(none_args, [None], optimize=True)

        # 定义一个带有默认参数 x=None 的测试函数 test_script_optional_tensor_none
        def test_script_optional_tensor_none(x=None):
            # type: (Optional[Tensor]) -> Tensor
            res = torch.zeros(1, dtype=torch.int8)  # 创建一个dtype为torch.int8的全0张量res
            if x is None:
                res = res + 1  # 如果参数x为None，则res加1
            else:
                res = x  # 否则，将参数x赋值给res
            return res

        fn = test_script_optional_tensor_none
        scripted_fn = torch.jit.script(fn)  # 对函数进行Torch脚本化
        self.assertEqual(fn(), scripted_fn())  # 检查函数调用结果是否与脚本化函数调用结果相等
        self.assertEqual(fn(torch.zeros(1)), scripted_fn(torch.zeros(1)))  # 检查带参数调用的结果是否相等

        # 定义一个带有默认参数 x=None 的测试函数 test_script_optional_other_none
        def test_script_optional_other_none(x=None):
            # type: (Optional[float]) -> float
            res = 2.0  # 初始化res为2.0
            if x is None:
                res = res + 1.0  # 如果参数x为None，则res加1.0
            else:
                res = x  # 否则，将参数x赋值给res
            return res

        fn = test_script_optional_other_none
        scripted_fn = torch.jit.script(fn)  # 对函数进行Torch脚本化
        self.assertEqual(fn(), scripted_fn())  # 检查函数调用结果是否与脚本化函数调用结果相等
        self.assertEqual(fn(1.0), scripted_fn(1.0))  # 检查带参数调用的结果是否相等

    # 定义一个测试函数，测试 torch.clamp 函数在 min=None 和 max=None 的情况下的行为
    def test_script_clamp_none(self):
        # 定义一个测试函数，使用 torch.clamp 对输入张量 x 进行最大值截断，最大值设为 None
        def test_script_clamp_max_none(x):
            return torch.clamp(x, min=2, max=None)

        # 定义一个测试函数，使用 torch.clamp 对输入张量 x 进行最大值截断，最大值设为 2
        def test_script_clamp_max(x):
            return torch.clamp(x, max=2)

        # 定义一个测试函数，使用 torch.clamp 对输入张量 x 进行最小值截断，最小值设为 None
        def test_script_clamp_min_none(x):
            return torch.clamp(x, min=None, max=2)

        # 定义一个测试函数，使用 torch.clamp 对输入张量 x 进行最小值截断，最小值设为 2
        def test_script_clamp_min(x):
            return torch.clamp(x, min=2)

        input = [torch.arange(0, 3)]  # 创建一个包含 torch.arange(0, 3) 的输入列表
        # 分别对四个 clamp 测试函数进行 Torch 脚本化，并使用自定义函数 checkScript 进行测试
        self.checkScript(test_script_clamp_max_none, input, optimize=True)
        self.checkScript(test_script_clamp_max, input, optimize=True)
        self.checkScript(test_script_clamp_min_none, input, optimize=True)
        self.checkScript(test_script_clamp_min, input, optimize=True)

    # 定义一个测试函数，测试返回一个布尔常量的 Torch 脚本化行为
    def test_script_bool_constant(self):
        # 定义一个函数 test_script_bool_constant，设置变量 a 为 True 并返回它
        def test_script_bool_constant():
            a = True
            return a

        # 对函数 test_script_bool_constant 进行 Torch 脚本化，并使用自定义函数 checkScript 进行测试
        self.checkScript(test_script_bool_constant, [])

    # 定义一个测试函数，测试三元表达式在 Torch 脚本中的使用
    def test_ternary(self):
        # 定义一个函数 func，接受两个参数 a 和 b，将 a + b 赋值给 c 如果 a > 3，否则将 b 赋值给 c
        def func(a, b):
            c = 3
            c = a + b if bool(a > 3) else b
            return c

        inputs_true = self._make_scalar_vars([5, 2], torch.int64)  # 创建两组输入，第一组符合条件，第二组不符合条件
        inputs_false = self._make_scalar_vars([1, 0], torch.int64)  # 创建两组输入，第一组符合条件，第二组不符合条件
        # 分别对 func 函数使用自定义函数 checkScript 进行 Torch 脚本化和测试
        self.checkScript(func, inputs_true, optimize=True)
        self.checkScript(func, inputs_false, optimize=True)
    def test_ternary_module_type_hint(self):
        # 定义三个测试用的神经网络模块类，每个类都有一个前向传播函数
        class M1(torch.nn.Module):
            # 前向传播返回字符串 'out' 或空字典，取决于模型的训练状态
            def forward(self) -> Any:
                return 'out' if self.training else {}

        class M2(torch.nn.Module):
            # 前向传播返回字符串 'out' 或空字典，存储在变量 out 中，再返回
            def forward(self) -> Any:
                out: Any = 'out' if self.training else {}
                return out

        class M3(torch.nn.Module):
            # 前向传播返回 None 或整数 1，取决于模型的训练状态
            def forward(self) -> Optional[int]:
                return None if self.training else 1

        # 分别对三个模型进行训练和评估状态下的测试
        for module in [M1, M2, M3]:
            self.checkModule(module().train(), ())   # 测试训练模式
            self.checkModule(module().eval(), ())    # 测试评估模式

    def test_ternary_static_if(self):
        # 测试条件变量标注为 Final 时的 True 分支
        class M1(torch.nn.Module):
            flag: torch.jit.Final[bool]

            def __init__(self):
                super().__init__()
                self.flag = True

            # 根据 self.flag 的值返回全为 1 的张量或空字典
            def forward(self) -> torch.Tensor:
                return torch.ones(3) if self.flag else {}

        # 测试条件变量标注为 Final 时的 True 分支
        class M2(torch.nn.Module):
            flag: torch.jit.Final[bool]

            def __init__(self):
                super().__init__()
                self.flag = False

            # 根据 self.flag 的值返回空字典或全为 1 的张量
            def forward(self) -> torch.Tensor:
                return {} if self.flag else torch.ones(3)

        model1 = M1()
        model2 = M2()
        script_model_1 = torch.jit.script(model1)
        script_model_2 = torch.jit.script(model2)
        # 断言脚本化模型的前向传播结果与原始模型的前向传播结果相等
        self.assertEqual(model1.forward(), script_model_1.forward())
        self.assertEqual(model2.forward(), script_model_2.forward())

    def test_ternary_right_associative(self):
        # 定义一个根据输入 x 的不同值返回不同结果的函数
        def plus_123(x: int):
            return x + 1 if x == 1 else x + 2 if x == 2 else x + 3
        # 分别测试输入为 1、2、3 时的函数 plus_123
        self.checkScript(plus_123, (1,))
        self.checkScript(plus_123, (2,))
        self.checkScript(plus_123, (3,))

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_print(self):
        # 定义一个打印函数，用于测试打印语句在脚本化环境中的行为
        def func(x, y):
            q = (x + y).sigmoid()
            # 打印 q 的值、1、2、列表 [1, 2]、列表 [1.0, 2.0]
            print(q, 1, 2, [1, 2], [1.0, 2.0])
            w = -q
            return w * w

        x = torch.arange(4., requires_grad=True)
        y = torch.arange(0., 8, 2, requires_grad=True)
        # 检查函数 func 在优化和捕获输出的条件下的脚本化行为
        self.checkScript(func, [x, y], optimize=True, capture_output=True)

    def test_format(self):
        # 定义一个格式化字符串的函数，用于测试格式化字符串在脚本化环境中的行为
        def func(x):
            print("{}, I'm a {}".format("Hello", "test"))
            print("format blank".format())
            print("stuff before {}".format("hi"))
            print("{} stuff after".format("hi"))
            return x + 1

        x = torch.arange(4., requires_grad=True)
        # 检查函数 func 在优化和捕获输出的条件下的脚本化行为
        self.checkScript(func, [x], optimize=True, capture_output=True)
    def test_logical_short_circuit(self):
        @torch.jit.script
        def testNoThrows(t):
            c1 = 1
            # 如果 t[1] 是空的张量或者 True，设置 c1 为 0
            if (False and bool(t[1])) or (True or bool(t[1])):
                c1 = 0
            return c1

        # 检查 testNoThrows 的图中是否没有出现 "prim::If" 操作
        FileCheck().check_not("prim::If").run(testNoThrows.graph)
        # 测试 t 为空张量时的返回值是否为 0
        self.assertEqual(0, testNoThrows(torch.randn(0)))
        # 测试 t 为形状为 [2, 3] 的张量时的返回值是否为 0
        self.assertEqual(0, testNoThrows(torch.randn([2, 3])))

        @torch.jit.script
        def throwsOr(t):
            # 计算 c0，如果 t[1] 不存在则会抛出错误
            c0 = False or bool(t[1])
            print(c0)

        @torch.jit.script
        def throwsAnd(t):
            # 计算 c0，如果 t[1] 不存在则会抛出错误
            c0 = True and bool(t[1])
            print(c0)

        t = torch.randn(0)
        # 测试 throwsOr 是否在 t 为空张量时抛出 RuntimeError 错误
        with self.assertRaisesRegex(RuntimeError, "index 1 out of range for tensor of size"):
            throwsOr(t)
        # 测试 throwsAnd 是否在 t 为空张量时抛出 RuntimeError 错误
        with self.assertRaisesRegex(RuntimeError, "index 1 out of range for tensor of size"):
            throwsAnd(t)
    def test_error(self):
        @torch.jit.script
        def foo(a):
            return a.t()
        s = Variable(torch.rand(5, 5, 5))
        # XXX: this should stay quiet in stay propagation and only fail in the interpreter
        # 使用torch.jit.script装饰器将foo函数编译为Torch脚本
        with self.assertRaisesRegex(RuntimeError, "failed in the TorchScript interpreter"):
            # 调用经过装饰的foo函数，预期在TorchScript解释器中失败引发RuntimeError异常
            foo(s)

        @torch.jit.script
        def bar(c, b):
            return c + b

        with self.assertRaisesRegex(RuntimeError, "failed in the TorchScript interpreter"):
            # 使用torch.jit.script装饰器将bar函数编译为Torch脚本
            # 传入具有梯度要求的变量参数，预期在TorchScript解释器中失败引发RuntimeError异常
            bar(Variable(torch.rand(10), requires_grad=True), Variable(torch.rand(9), requires_grad=True))

    def test_error_stacktrace(self):
        @torch.jit.script
        def baz(c, b):
            return c + b

        @torch.jit.script
        def foo(c, b):
            return baz(c, b)

        @torch.jit.script
        def bar(c, b):
            return foo(c, b)

        with self.assertRaises(RuntimeError) as cm:
            # 调用经过装饰的bar函数，预期在TorchScript解释器中失败引发RuntimeError异常
            bar(torch.rand(10), torch.rand(9))
        FileCheck().check("The following operation failed in the TorchScript interpreter") \
                   .check("Traceback") \
                   .check("in foo").check("in baz").run(str(cm.exception))

    def test_error_stacktrace_interface(self):
        @torch.jit.script
        def baz(c, b):
            return c + b

        @torch.jit.script
        def foo(c, b):
            return baz(c, b)

        @torch.jit.script
        def bar(c, b):
            return foo(c, b)

        @torch.jit.script
        class Bar:
            def one(self, x, y):
                return bar(x, y)

        @torch.jit.interface
        class IFace:
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                pass

        make_global(IFace)

        @torch.jit.script
        def as_interface(x):
            # type: (IFace) -> IFace
            return x

        f = as_interface(Bar())

        with self.assertRaises(RuntimeError) as cm:
            # 调用Bar类的one方法，预期在TorchScript解释器中失败引发RuntimeError异常
            x = f.one(torch.rand(10), torch.rand(9))
            bar(torch.rand(10), torch.rand(9))
        FileCheck().check("The following operation failed in the TorchScript interpreter") \
                   .check("Traceback") \
                   .check("in foo").check("in baz").run(str(cm.exception))

    def test_operator_precedence(self):
        def double(x):
            # type: (int) -> int
            return 2 * x

        def complicated_arithmetic_operation():
            # TODO we need to test exponent operator '**' and bitwise not
            # operator '~' once they are properly supported.
            list = [0, 1, 2, 3]
            # 执行复杂的算术操作，涵盖多种运算符的优先级和结合性
            result = list[1:3][0] + double(4) + (-3 + 8) * 6 // 2 % 4 << 2 + 1 >> 1 | 23 & 16 + 3 ^ 4
            return result

        self.checkScript(complicated_arithmetic_operation, ())

    def test_in_operator_with_two_strings(self):
        def fn() -> bool:
            return "a" in "abcd"
        # 检查字符串"a"是否存在于字符串"abcd"中
        self.checkScript(fn, ())
    # 定义测试函数 test_bitwise_ops，测试位操作符的功能
    def test_bitwise_ops(self):

        # 定义整数测试函数 int_test，测试整数的位与、异或、或、左移、右移操作
        def int_test():
            return 2 & 3, 2 ^ 3, 2 | 3, 2 << 3, 2 >> 3

        # 使用 self.checkScript 方法验证 int_test 函数的输出
        self.checkScript(int_test, ())

        # 定义布尔值测试函数 bool_test，测试布尔值的位与、异或、或 操作
        def bool_test(x, y):
            # type: (bool, bool) -> Tuple[bool, bool, bool]
            return x & y, x ^ y, x | y

        # 使用 self.checkScript 方法验证 bool_test 函数在不同输入条件下的输出
        self.checkScript(bool_test, (True, False))
        self.checkScript(bool_test, (True, True))

        # 定义张量测试函数 tensor_test，测试张量的位与、异或、或 操作
        def tensor_test(x, y):
            return x & y, x ^ y, x | y

        # 定义与整数结合的张量测试函数 tensor_with_int_test，测试张量的左移、右移操作
        def tensor_with_int_test(x, y):
            # type: (Tensor, int) -> Tuple[Tensor, Tensor]
            return x << y, x >> y

        # 创建 torch.tensor 对象 x 和 y，分别赋值为 2 和 3
        x = torch.tensor(2)
        y = torch.tensor(3)

        # 使用 self.checkScript 方法验证 tensor_test 和 tensor_with_int_test 函数的输出
        self.checkScript(tensor_test, (x, y))
        self.checkScript(tensor_with_int_test, (x, 2))

        # 定义取反操作的测试函数 not_test，测试张量的取反操作
        def not_test(x):
            return ~x

        # 使用 self.checkScript 方法验证 not_test 函数的输出
        self.checkScript(not_test, (torch.tensor([2, 4]), ))

    # 定义测试函数 test_all，测试 all 函数在不同类型列表或张量上的行为
    def test_all(self):

        # 定义 torch.jit.script 下的张量测试函数 test_all_tensor，测试 all 函数在张量上的行为
        @torch.jit.script
        def test_all_tensor(x):
            return all(x)

        # 验证 test_all_tensor 函数在特定张量输入下的输出
        self.assertFalse(test_all_tensor(torch.tensor([1, 0, 3], dtype=torch.uint8)))
        self.assertTrue(test_all_tensor(torch.tensor([3.14, 3, 99], dtype=torch.uint8)))
        self.assertTrue(test_all_tensor(torch.tensor([True, True], dtype=torch.uint8)))
        self.assertFalse(test_all_tensor(torch.tensor([True, False], dtype=torch.uint8)))

        # 定义 torch.jit.script 下的布尔列表测试函数 test_all_bool_list，测试 all 函数在布尔列表上的行为
        @torch.jit.script
        def test_all_bool_list(x):
            # type: (List[bool]) -> bool
            return all(x)

        # 验证 test_all_bool_list 函数在不同布尔列表输入下的输出
        self.assertTrue(test_all_bool_list([True, True]))
        self.assertTrue(test_all_bool_list([True, 1]))
        self.assertFalse(test_all_bool_list([True, False]))
        self.assertFalse(test_all_bool_list([True, 0]))
        self.assertFalse(test_all_bool_list([False, 0]))
        self.assertTrue(test_all_bool_list([]))

        # 定义 torch.jit.script 下的整数列表测试函数 test_all_int_list，测试 all 函数在整数列表上的行为
        @torch.jit.script
        def test_all_int_list(x):
            # type: (List[int]) -> bool
            return all(x)

        # 验证 test_all_int_list 函数在不同整数列表输入下的输出
        self.assertTrue(test_all_int_list([3, 6]))
        self.assertFalse(test_all_int_list([2, 0]))

        # 定义 torch.jit.script 下的浮点数列表测试函数 test_all_float_list，测试 all 函数在浮点数列表上的行为
        @torch.jit.script
        def test_all_float_list(x):
            # type: (List[float]) -> bool
            return all(x)

        # 验证 test_all_float_list 函数在不同浮点数列表输入下的输出
        self.assertTrue(test_all_float_list([3.14, 8.1]))
        self.assertFalse(test_all_float_list([3.14, 0, 8.9]))

    # 使用 skipIfTorchDynamo 装饰器，跳过在 TorchDynamo 环境下不适合执行的测试用例
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_number_math(self):
        ops_template = dedent('''
        def func():
            return {scalar1} {op} {scalar2}
        ''')
        ops = ['+', '-', '*', '%', '<', '<=', '>', '>=', '==', '!=', '//']
        funcs_template = dedent('''
        def func():
            return {func}({scalar1}, {scalar2})
        ''')
        funcs = ['min', 'max']
        scalars = ['7', '2', '3', '-3', '3.14', '0.125', '-0.5', '2.0', '-2.0']
        scalar_pairs = [(scalar1, scalar2) for scalar1 in scalars for scalar2 in scalars]

        def run_test(code):
            # 创建一个空的作用域字典
            scope = {}
            # 在全局和局部作用域中执行给定的代码
            execWrapper(code, globals(), scope)
            # 使用 torch 的 JIT 编译给定的代码
            cu = torch.jit.CompilationUnit(code)

            # 断言 JIT 编译后的函数调用结果与作用域中函数执行结果相等
            self.assertEqual(cu.func(), scope['func']())

        # 遍历所有可能的标量对
        for scalar1, scalar2 in scalar_pairs:
            # 遍历所有运算符
            for op in ops:
                # 使用运算符模板生成代码
                code = ops_template.format(op=op, scalar1=scalar1, scalar2=scalar2)
                # 运行测试
                run_test(code)
            # 遍历所有函数
            for func in funcs:
                # 使用函数模板生成代码
                code = funcs_template.format(func=func, scalar1=scalar1, scalar2=scalar2)
                # 运行测试
                run_test(code)

        # 测试标量重载
        for scalar1, scalar2 in scalar_pairs:
            item1 = 'torch.tensor(' + scalar1 + ').item()'
            item2 = 'torch.tensor(' + scalar2 + ').item()'
            # 遍历所有运算符
            for op in ops:
                # 使用运算符模板生成代码
                code = ops_template.format(op=op, scalar1=item1, scalar2=scalar2)
                # 运行测试
                run_test(code)
                code = ops_template.format(op=op, scalar1=scalar1, scalar2=item2)
                run_test(code)
                code = ops_template.format(op=op, scalar1=item1, scalar2=item2)
                run_test(code)
            # 遍历所有函数
            for func in funcs:
                # 使用函数模板生成代码
                code = funcs_template.format(func=func, scalar1=item1, scalar2=scalar2)
                # 运行测试
                run_test(code)
                code = funcs_template.format(func=func, scalar1=scalar1, scalar2=item2)
                run_test(code)
                code = funcs_template.format(func=func, scalar1=item1, scalar2=item2)
                run_test(code)

    def test_number_abs(self):
        # 定义函数 func1，接受一个 float 参数，返回其绝对值
        def func1(x):
            # type: (float) -> float
            return abs(x)

        # 定义函数 func2，接受一个 int 参数，返回其绝对值
        def func2(x):
            # type: (int) -> int
            return abs(x)

        # 定义函数 func3，接受一个参数，返回其绝对值（类型隐含）
        def func3(x):
            return abs(x)

        # 使用 checkScript 方法测试 func1 函数的 JIT 编译版本，传入参数 -3.14 和 3.14
        self.checkScript(func1, (-3.14,))
        self.checkScript(func1, (3.14,))
        # 使用 checkScript 方法测试 func2 函数的 JIT 编译版本，传入参数 -10 和 10
        self.checkScript(func2, (-10,))
        self.checkScript(func2, (10,))
        # 使用 checkScript 方法测试 func3 函数的 JIT 编译版本，传入包含负数和正数的张量
        self.checkScript(func3, (torch.tensor([-5, -10, -20]),))
        self.checkScript(func3, (torch.tensor([5, 10, 20]),))
        self.checkScript(func3, (torch.tensor([-5, 10, -20]),))

    def test_number_div(self):
        # 断言 div_int_future 函数的返回结果与其 JIT 编译版本的返回结果相等
        self.assertEqual(div_int_future(), torch.jit.script(div_int_future)())
        # 使用 checkScript 方法测试 div_float_future 函数的 JIT 编译版本
        self.checkScript(div_float_future, ())

        # 使用 checkScript 方法测试 div_int_nofuture 和 div_float_nofuture 函数
        self.checkScript(div_int_nofuture, ())
        self.checkScript(div_float_nofuture, ())

    # Testing bitwise shorthand aug assignment
    # 定义测试函数，测试布尔型变量的按位或赋值运算
    def test_bool_augassign_bitwise_or(self):
        # 定义内部函数 func，接受两个布尔型参数，返回按位或赋值后的结果
        def func(a: bool, b: bool) -> bool:
            # 对变量 a 进行按位或赋值运算
            a |= b
            return a

        # 使用自定义的测试函数 checkScript 测试 func 函数的不同参数组合
        self.checkScript(func, (True, False), optimize=True)
        self.checkScript(func, (True, True), optimize=True)
        self.checkScript(func, (False, False), optimize=True)
        self.checkScript(func, (False, True), optimize=True)

    # 定义测试函数，测试布尔型变量的按位与赋值运算
    def test_bool_augassign_bitwise_and(self):
        # 定义内部函数 func，接受两个布尔型参数，返回按位与赋值后的结果
        def func(a: bool, b: bool) -> bool:
            # 对变量 a 进行按位与赋值运算
            a &= b
            return a

        # 使用自定义的测试函数 checkScript 测试 func 函数的不同参数组合
        self.checkScript(func, (True, False), optimize=True)
        self.checkScript(func, (True, True), optimize=True)
        self.checkScript(func, (False, False), optimize=True)
        self.checkScript(func, (False, True), optimize=True)

    # 定义测试函数，测试布尔型变量的按位异或赋值运算
    def test_bool_augassign_bitwise_xor(self):
        # 定义内部函数 func，接受两个布尔型参数，返回按位异或赋值后的结果
        def func(a: bool, b: bool) -> bool:
            # 对变量 a 进行按位异或赋值运算
            a ^= b
            return a

        # 使用自定义的测试函数 checkScript 测试 func 函数的不同参数组合
        self.checkScript(func, (True, False), optimize=True)
        self.checkScript(func, (True, True), optimize=True)
        self.checkScript(func, (False, False), optimize=True)
        self.checkScript(func, (False, True), optimize=True)

    # 定义测试函数，测试整数变量的按位左移赋值运算
    def test_number_augassign_bitwise_lshift(self):
        # 定义内部函数 func，返回整数变量进行按位左移赋值运算后的结果
        def func() -> int:
            # 初始化变量 z 为整数 8，对其进行按位左移赋值运算
            z = 8
            z <<= 2
            return z

        # 使用自定义的测试函数 checkScript 测试 func 函数
        self.checkScript(func, (), optimize=True)

    # 定义测试函数，测试整数变量的按位右移赋值运算
    def test_number_augassign_bitwise_rshift(self):
        # 定义内部函数 func，返回整数变量进行按位右移赋值运算后的结果
        def func() -> int:
            # 初始化变量 z 为整数 8，对其进行按位右移赋值运算
            z = 8
            z >>= 2
            return z

        # 使用自定义的测试函数 checkScript 测试 func 函数
        self.checkScript(func, (), optimize=True)

    # 定义测试函数，测试整数变量的按位幂赋值运算
    def test_number_augassign_bitwise_pow(self):
        # 定义内部函数 func，返回整数变量进行按位幂赋值运算后的结果
        def func() -> float:
            # 初始化变量 z 为整数 8，对其进行按位幂赋值运算
            z = 8
            z **= 2
            return z

        # 使用自定义的测试函数 checkScript 测试 func 函数
        self.checkScript(func, (), optimize=True)

    # 定义测试函数，测试整数变量的加法赋值运算
    def test_number_augassign(self):
        # 定义内部函数 func，返回整数变量进行加法赋值运算后的结果
        def func():
            # 初始化变量 z 为整数 1，对其进行加法赋值运算
            z = 1
            z += 2
            return z

        # 使用自定义的测试函数 checkScript 测试 func 函数
        self.checkScript(func, (), optimize=True)

    # 定义测试函数，测试嵌套类中的属性赋值
    def test_nested_select_assign(self):
        # 定义 SubSubModule 类，继承自 torch.nn.Module
        class SubSubModule(torch.nn.Module):
            # 初始化方法，设置属性 abc 为整数 11
            def __init__(self):
                super().__init__()
                self.abc = 11

            # 前向传播方法，返回属性 abc 的值
            def forward(self, x):
                return self.abc

        # 定义 SubModule 类，继承自 torch.nn.Module
        class SubModule(torch.nn.Module):
            # 初始化方法，设置属性 a 为整数 11，并初始化 nested 为 SubSubModule 类的实例
            def __init__(self):
                super().__init__()
                self.a = 11
                self.nested = SubSubModule()

            # 前向传播方法，返回属性 a 的值
            def forward(self, x):
                return self.a

        # 定义 TestModule 类，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            # 初始化方法，设置属性 sub 为 SubModule 类的实例，属性 hi 为整数 1
            def __init__(self):
                super().__init__()
                self.sub = SubModule()
                self.hi = 1

            # 前向传播方法，修改属性 hi 的值为整数 5，修改 sub 的属性 a 和 nested 的属性 abc 的值
            # 返回 sub.a * 20 + sub.nested.abc * 3 + self.hi 的计算结果
            def forward(self):
                self.hi = 5
                self.sub.a = 1
                self.sub.nested.abc = 5
                return self.sub.a * 20 + self.sub.nested.abc * 3 + self.hi

        # 使用自定义的测试函数 checkModule 测试 TestModule 类的实例
        self.checkModule(TestModule(), ())
    def test_number_neg(self):
        # 定义一个返回整数 -8 的函数
        def func1():
            return -8

        # 定义一个返回浮点数 -3.14 的函数
        def func2():
            return -3.14

        # 使用自定义的函数检查脚本化结果，优化开启
        self.checkScript(func1, (), optimize=True)
        self.checkScript(func2, (), optimize=True)

    def test_compare_two_bool_inputs(self):
        # 定义比较两个布尔类型参数是否相等的函数
        def compare_eq(a: bool, b: bool):
            return a == b

        # 定义比较两个布尔类型参数是否不相等的函数
        def compare_ne(a: bool, b: bool):
            return a != b

        # 对比脚本化后的函数与原始函数的执行结果是否相等
        scripted_fn_eq = torch.jit.script(compare_eq)
        scripted_fn_ne = torch.jit.script(compare_ne)
        self.assertEqual(scripted_fn_eq(True, False), compare_eq(True, False))
        self.assertEqual(scripted_fn_eq(False, True), compare_eq(False, True))
        self.assertEqual(scripted_fn_eq(True, True), compare_eq(True, True))
        self.assertEqual(scripted_fn_eq(False, False), compare_eq(False, False))

        self.assertEqual(scripted_fn_ne(True, False), compare_ne(True, False))
        self.assertEqual(scripted_fn_ne(False, True), compare_ne(False, True))
        self.assertEqual(scripted_fn_ne(True, True), compare_ne(True, True))
        self.assertEqual(scripted_fn_ne(False, False), compare_ne(False, False))
    def _test_tensor_number_math(self, device='cpu'):
        # 定义一个模板字符串，用于生成测试函数的代码
        template = dedent('''
        def func(t):
            return {lhs} {op} {rhs}
        ''')

        def test(op, tensor, const, swap_args, template=template):
            # 根据参数决定是否交换常量和张量的位置
            args = ('t', const)
            if swap_args:
                args = (const, 't')

            # 根据模板生成具体的函数代码
            code = template.format(lhs=args[0], rhs=args[1], op=op)
            scope = {}
            # 执行生成的代码，并将函数定义导入到局部作用域
            execWrapper(code, globals(), scope)
            # 使用 torch.jit.CompilationUnit 编译生成的代码
            cu = torch.jit.CompilationUnit(code)
            # 准备测试的消息
            message = f'with code `{args[0]} {op} {args[1]}` and t={tensor}'
            # 调用通过 CompilationUnit 生成的函数，并比较结果
            res1 = cu.func(tensor)
            # 直接调用局部作用域中的函数，并比较结果
            res2 = scope['func'](tensor)
            # 使用 self.assertEqual 断言两种调用的结果应该相等，并输出相关消息
            self.assertEqual(res1, res2, msg=message + "\nres1=" + str(res1) + "\nres2=" + str(res2))
            # 进一步断言两种结果的数据类型应该相等，并输出相关消息
            self.assertEqual(res1.dtype, res2.dtype, msg=message + "\nres1=" + str(res1) + "\nres2=" + str(res2))

        # 定义测试使用的整数和浮点数常量
        var_int = [2, -2]
        var_float = [1.4321, -1.2]

        # 定义支持的运算符列表
        ops = ['+', '-', '*', '%', '<', '<=', '>', '>=', '==', '!=', '/']

        # 创建不同类型的张量以及相应的常量列表
        float_tensor = torch.randn(5, 5, device=device)
        double_tensor = torch.randn(5, 5, dtype=torch.double, device=device)
        long_tensor = torch.randint(-5, 5, (5, 5), dtype=torch.long, device=device)
        long_tensor[long_tensor == 0] = 2

        tensors = [float_tensor, double_tensor, long_tensor]
        consts = var_int + var_float

        # 对每个运算符、张量、常量和是否交换参数顺序的组合进行测试
        for op, tensor, const, swap_args in product(ops, tensors, consts, [True, False]):
            # 对于除法运算，如果张量是 long_tensor，则跳过测试（因为未正确实现）
            if op == '/' and tensor.data_ptr() == long_tensor.data_ptr():
                continue

            # 对于取模运算，不支持交换参数顺序
            if op == '%' and swap_args is True:
                continue

            # 调用 test 函数执行具体的测试
            test(op, tensor, const, swap_args)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_tensor_number_math(self):
        # 调用 _test_tensor_number_math 函数执行测试
        self._test_tensor_number_math()

    def test_torch_tensor_bad_input(self):
        # 测试处理不良输入时是否引发了正确的异常信息
        with self.assertRaisesRegex(RuntimeError, "must be of ints, floats, "
                                    "or bools, got None"):
            @torch.jit.script
            def test():
                return torch.tensor([None])
            test()

        with self.assertRaisesRegex(RuntimeError, r"Empty lists default to List\[Tensor\]"):
            @torch.jit.script
            def tmp():
                return torch.tensor([])
            tmp()

        @torch.jit.script
        def foo():
            return torch.tensor([[2, 2], [1]])
        with self.assertRaisesRegex(RuntimeError, "Expected sequence of length"):
            foo()

    @suppress_warnings
    def test_torch_tensor_as_tensor_empty_list(self):
        # 定义一个测试函数，用于测试 torch.jit.annotate 的使用
        tensor_template = dedent('''
        def func():
            # 使用 torch.jit.annotate 创建一个空的整数列表
            empty_list = torch.jit.annotate(List[int], [])
            # 使用给定的操作符和输入来创建张量
            ten1 = torch.{tensor_op}({input})
            return ten1
        ''')
        # 可选的操作符和输入列表
        ops = ['tensor', 'as_tensor']
        inputs = ['empty_list', '[empty_list, empty_list]', '[[[empty_list]]]']

        # 遍历所有操作符和输入组合进行测试
        for op in ops:
            for inp in inputs:
                # 根据模板和当前的操作符和输入生成具体的代码
                code = tensor_template.format(tensor_op=op, input=inp)
                # 创建一个空的命名空间作用域
                scope = {}
                # 在全局命名空间中执行生成的代码，并将定义的变量和函数放入 scope 中
                exec(code, globals(), scope)
                # 使用 torch.jit.CompilationUnit 创建一个编译单元
                cu = torch.jit.CompilationUnit(code)
                # 调用生成的函数 func() 获取结果
                t1 = cu.func()
                t2 = scope['func']()
                # 如果输入为 'empty_list'，则验证 torchscript 返回整数张量，Python 返回浮点数张量
                if inp == 'empty_list':
                    self.assertNotEqual(t1.dtype, t2.dtype)
                # 验证两个张量的值相等（忽略精确的数据类型）
                self.assertEqual(t1, t2, exact_dtype=False)
                # 验证两个张量的设备类型相同
                self.assertEqual(t1.device, t2.device)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "Simple Executor doesn't have any shapes to propagate")
    # 定义测试函数 test_tensor_as_tensor_shape_prop，用于测试不同的张量操作和输入形状
    def test_tensor_as_tensor_shape_prop(self):
        # 定义包含缩进的字符串模板，其中包含一个 torch 张量操作和一个输入参数
        tensor_template = dedent('''
        def func():
            return torch.{tensor_op}({input})
        ''')
        # 定义要测试的张量操作和相应的输入参数
        ops = ['tensor', 'as_tensor']
        inputs = ['[1]', '[False]', '[2.5]', '0.5', '1', 'False', '[[1]]', 'torch.jit.annotate(List[List[int]], [])']
        # 定义预期的张量形状字符串列表
        expected_shape = ["Long(*, device=cpu)", "Bool(*, device=cpu)",
                          "Float(*, device=cpu)", "Float(device=cpu)",
                          "Long(device=cpu)", "Bool(device=cpu)", "Long(*, *, device=cpu)"]

        # 遍历所有的操作和输入参数组合进行测试
        for op in ops:
            for inp, expect in zip(inputs, expected_shape):
                # 使用字符串模板填充得到具体的测试代码
                code = tensor_template.format(tensor_op=op, input=inp)
                # 创建一个空的作用域字典
                scope = {}
                # 在全局作用域中执行填充后的代码，将结果存储在作用域中
                exec(code, globals(), scope)
                # 使用填充后的代码创建一个 Torch 脚本编译单元
                cu = torch.jit.CompilationUnit(code)
                # 对编译单元的函数图进行完整的形状分析
                torch._C._jit_pass_complete_shape_analysis(cu.func.graph, (), False)
                # 运行文件检查，验证预期的张量形状和操作是否存在于函数图中
                FileCheck().check(expect).check(f"aten::{op}").run(cu.func.graph)

        # 定义 Torch 脚本函数 test_dtype，接受一个 torch.dtype 类型的输入参数
        @torch.jit.script
        def test_dtype(inp_dtype: torch.dtype):
            # 创建两个张量，其中一个指定了数据类型和梯度要求
            a = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
            b = torch.tensor(1.0, dtype=inp_dtype)
            return a, b

        # 根据 GRAPH_EXECUTOR 的值选择不同的测试路径
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            # 为输入 5 创建测试图形，并进行分析和回放
            g = test_dtype.graph_for(5, profile_and_replay=True)
            # 运行文件检查，验证张量类型和 prim::BailOut 是否存在于图形中
            FileCheck().check("Tensor = aten::tensor").check("Float(device=cpu) = prim::BailOut") \
                       .check("Tensor = aten::tensor").check("Half(device=cpu) = prim::BailOut").run(g)
        else:
            # 为输入 5 创建测试图形
            g = test_dtype.graph_for(5)
            # 运行文件检查，验证第一个张量的类型设置，第二个张量的类型未设置
            FileCheck().check("Float(requires_grad=1, device=cpu) = aten::tensor") \
                       .check("Tensor(requires_grad=0) = aten::tensor").run(g)

        # 定义 Torch 脚本函数 test_as_tensor_tensor_input，接受一个输入参数 input
        @torch.jit.script
        def test_as_tensor_tensor_input(input):
            # 使用输入参数创建两个张量，分别指定相同的数据类型
            a = torch.as_tensor(input, dtype=input.dtype)
            b = torch.as_tensor(input, dtype=torch.float)
            return a, b

        # 根据 GRAPH_EXECUTOR 的值选择不同的测试路径
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            # 为输入 torch.ones(3, 4) 创建测试图形，并进行分析和回放
            g = test_as_tensor_tensor_input.graph_for(torch.ones(3, 4), profile_and_replay=True)
            # 运行文件检查，验证张量类型和 prim::BailOut 是否存在于图形中
            FileCheck().check("Tensor = aten::as_tensor").check("Float(3, 4) = prim::BailOut") \
                       .check("Tensor = aten::as_tensor").check("Float(3, 4) = prim::BailOut").run(g)
        else:
            # 为输入 torch.ones(3, 4) 创建测试图形
            g = test_as_tensor_tensor_input.graph_for(torch.ones(3, 4))
            # 运行文件检查，验证张量类型和其他参数是否存在于图形中
            FileCheck().check("Tensor = aten::as_tensor").check("Float(*, *, requires_grad=0, device=cpu) = aten::as_tensor").run(g)

    # 如果 GRAPH_EXECUTOR 不是 ProfilingMode.LEGACY，则跳过该测试
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "testing legacy behavior")
    def test_tensor_requires_grad(self):
        @torch.jit.script
        def test(b):
            # type: (bool) -> Tuple[Tensor, Tensor, Tensor]
            # 创建一个张量a，根据输入的布尔值b决定是否需要计算梯度
            a = torch.tensor(1., requires_grad=b)
            # 创建张量b，始终需要计算梯度
            b = torch.tensor(1., requires_grad=True)
            # 创建张量c，不需要计算梯度
            c = torch.tensor(1., requires_grad=False)
            return a, b, c

        # 获取test函数的计算图
        g = test.graph_for(True)
        # 获取计算图的输出节点
        out = next(g.outputs())
        # 获取输出节点的输入
        out_inp = list(out.node().inputs())

        # 断言第一个输出张量需要计算梯度
        self.assertTrue(out_inp[0].requires_grad())
        # 断言第二个输出张量需要计算梯度
        self.assertTrue(out_inp[1].requires_grad())
        # 断言第三个输出张量不需要计算梯度
        self.assertFalse(out_inp[2].requires_grad())

    def test_grad_from_script(self):
        # 定义一个函数test，创建一个需要计算梯度的张量a，计算张量b作为其两倍
        def test():
            a = torch.tensor(2.5, requires_grad=True)
            b = a * 2
            return a, b

        # 调用test函数，并进行反向传播
        a, b = test()
        b.backward()

        # 使用Torch JIT对test函数进行脚本化，得到脚本化后的张量a_script和b_script
        a_script, b_script = torch.jit.script(test)()
        # 对脚本化后的张量b_script进行反向传播
        b_script.backward()

        # 断言原始张量a和脚本化后的张量a_script的梯度相等
        self.assertEqual(a.grad, a_script.grad)
    # 定义一个测试方法，用于测试 torch.Tensor 和 torch.as_tensor 方法
    def test_torch_tensor_as_tensor(self):
        # 定义一个模板字符串，用于生成包含特定操作的函数代码
        tensor_template = dedent('''
        def func():
            # 根据给定的列表创建一个 Python 列表
            li = {list_create}
            # 使用 torch.{tensor_op} 方法将其转换为张量
            ten1 = torch.{tensor_op}(li {options})
            return ten1
        ''')

        # 列表，包含用于测试的字符串表示的列表
        lists = ["2.5", "4", "True", "False", "[2]", "[-.5]", "[False, True, False]", "[2, 2]", "(1, 1)",
                 "torch.jit.annotate(List[List[int]], [])",
                 "torch.jit.annotate(List[int], [])", "[2.5, 2.5]", "[[2], [2]]", "[[-.5], [2.2]]", "[[False], [True]]"]

        # 数据类型列表，包含各种数据类型的字符串表示
        dtypes = ["", ", dtype=torch.float", ", dtype=torch.double", ", dtype=torch.half",
                  ", dtype=torch.uint8", ", dtype=torch.int8", ", dtype=torch.short",
                  ", dtype=torch.int", ", dtype=torch.long", ", dtype=torch.cfloat",
                  ", dtype=torch.cdouble"]

        # 操作列表，包含测试的操作类型
        ops = ['tensor', 'as_tensor']
        # 设备列表，包含空字符串和用于 GPU 的字符串
        devices = ['', ", device='cpu'"]
        if RUN_CUDA:
            devices.append(", device='cuda'")

        # 可选项对列表，包含不同数据类型和设备的组合
        option_pairs = [dtype + device for dtype in dtypes for device in devices]

        # 嵌套循环，用于生成所有可能的代码变体进行测试
        for op in ops:
            for li in lists:
                for option in option_pairs:
                    # 如果列表中包含 "annotate" 并且选项中不包含 "dtype"，则跳过
                    if "annotate" in li and "dtype" not in option:
                        continue
                    # 如果 Python 版本为 3.10 且选项中包含 "torch.uint8" 且列表中包含负数，则跳过
                    if sys.version_info[:2] >= (3, 10) and "torch.uint8" in option and "-" in li:
                        continue
                    # 使用模板生成具体的代码
                    code = tensor_template.format(list_create=li, tensor_op=op, options=option)
                    # 创建一个空的命名空间字典
                    scope = {}
                    # 在全局命名空间执行生成的代码，并将结果存储在 scope 中
                    exec(code, globals(), scope)
                    # 使用 torch.jit.CompilationUnit 编译生成的代码
                    cu = torch.jit.CompilationUnit(code)
                    # 调用生成的函数并获取返回的张量
                    t1 = cu.func()
                    t2 = scope['func']()
                    # 如果张量类型为 torch.float16，则进行特殊的相等性检查（半精度张量不支持直接相等性比较）
                    if t1.dtype == torch.float16:
                        self.assertTrue(str(t1) == str(t2))
                    else:
                        self.assertEqual(t1, t2)
                    # 检查张量的数据类型和设备是否相同
                    self.assertEqual(t1.dtype, t2.dtype)
                    self.assertEqual(t1.device, t2.device)

        # 定义一个内部测试方法，测试 torch.as_tensor 方法接收张量输入的情况
        def test_as_tensor_tensor_input(input):
            # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor]
            # 使用 torch.as_tensor 方法将输入转换为指定数据类型的张量
            return torch.as_tensor(input, dtype=torch.cfloat), torch.as_tensor(input, dtype=torch.float), \
                torch.as_tensor(input, dtype=torch.int32)

        # 创建一个随机张量作为输入
        inp = torch.randn(3, 4, dtype=torch.cfloat)
        # 调用自定义的测试方法进行检查
        self.checkScript(test_as_tensor_tensor_input, (inp,))
    def test_torch_tensor_dtype(self):
        def foo(s: float):
            # 创建一个张量和一个形状相同的张量列表，元素都是输入的浮点数s
            return torch.tensor(s), torch.tensor([s, s])

        # 设置默认数据类型为双精度浮点数，以重新运行形状分析
        with set_default_dtype(torch.double):
            # 检查通过JIT脚本化的foo函数和直接调用foo函数的结果是否相等，并确保数据类型完全匹配
            self.assertEqual(torch.jit.script(foo)(1.), foo(1.), exact_dtype=True)
            # 如果图执行器为遗留模式，则运行文件检查，检查是否包含"Double"和"aten::tensor"，并运行最后优化图
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                FileCheck().check("Double").check_same("aten::tensor").run(torch.jit.last_executed_optimized_graph())

        # 设置默认数据类型为单精度浮点数
        with set_default_dtype(torch.float):
            # 清除foo函数的JIT缓存
            del torch.jit._state._jit_caching_layer[foo]
            # 检查通过JIT脚本化的foo函数和直接调用foo函数的结果是否相等，并确保数据类型完全匹配
            self.assertEqual(torch.jit.script(foo)(1.), foo(1.), exact_dtype=True)
            # 如果图执行器为遗留模式，则运行文件检查，检查是否包含"Float"和"aten::tensor"，并运行最后优化图
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                FileCheck().check("Float").check_same("aten::tensor").run(torch.jit.last_executed_optimized_graph())

        # 设置默认数据类型为半精度浮点数
        with set_default_dtype(torch.half):
            # 清除foo函数的JIT缓存
            del torch.jit._state._jit_caching_layer[foo]
            # 检查通过JIT脚本化的foo函数和直接调用foo函数的结果是否相等，并确保数据类型完全匹配
            self.assertEqual(torch.jit.script(foo)(1.), foo(1.), exact_dtype=True)
            # 如果图执行器为遗留模式，则运行文件检查，检查是否包含"Half"和"aten::tensor"，并运行最后优化图
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                FileCheck().check("Half").check_same("aten::tensor").run(torch.jit.last_executed_optimized_graph())

    def test_shape_analysis_grad_property(self):
        @torch.jit.script
        def foo(x):
            # 返回x和tanh(x)的差
            return torch.sub(x, torch.tanh(x))

        # 执行完整的形状分析
        torch._C._jit_pass_complete_shape_analysis(foo.graph, (torch.tensor([0.39]),), False)

        # 断言aten::sub节点的输出不应该意外地设置requires_grad属性
        self.assertTrue(foo.graph.findNode("aten::sub").output().requiresGrad() is None)

    def test_empty_like_memory_format_bc(self):
        def f(x):
            # 返回一个与输入张量x形状相同的全零张量，内存格式为None
            return torch.zeros_like(x, memory_format=None)

        # 对函数f进行脚本化
        scripted_f = torch.jit.script(f)
        x = torch.rand(3, 4)
        # 断言通过脚本化的函数和原始函数对相同输入x的结果是否相等
        self.assertEqual(scripted_f(x), f(x))

    def test_multiline_string_dedents(self):
        def foo() -> None:
            multiline_string_dedent_1 = """
        # 多行字符串，通过三个双引号包围，可以包含多行文本，保留了原始缩进
        multiline_string_dedent_2 = """ This is a
  string dedent """
        # 多行字符串，通过三个双引号包围，保留了原始缩进
        multiline_string_dedent_3 = """
            This is a string
dedent """
        # 多行字符串，通过三个双引号包围，保留了原始缩进
        multiline_string_dedent_4 = """ This is a string dedent """

        # 使用 torch.jit.script 将函数 foo 脚本化
        scripted_foo = torch.jit.script(foo)
        # 断言脚本化后的函数执行结果与原始函数执行结果相等
        self.assertEqual(scripted_foo(), foo())

    def test_class_with_comment_at_lower_indentation(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 实现 forward 方法
            def forward(self, x):
                # 对 x 执行 torch.neg 操作
                x = torch.neg(x)
                # 返回处理后的 x
                return x
        # 此注释位于错误的缩进级别

        # 对类 Foo 进行脚本化
        torch.jit.script(Foo())

    # 从 test_torch 中的测试进行适配
    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_tensor_number_math_cuda(self):
        # 在 CUDA 设备上执行 _test_tensor_number_math 测试
        self._test_tensor_number_math(device='cuda')

    def test_not(self):
        # 测试 Python 中的 not 操作符
        # TODO: 当布尔转换准备就绪时，添加更多测试
        def test_not_op(a):
            # 返回 a 是否大于 1 的逻辑非结果
            return not bool(a > 1)

        # 使用 checkScript 方法验证 test_not_op 函数
        self.checkScript(test_not_op, (torch.tensor(2), ), optimize=True)

    def test_is_isnot(self):
        # 测试 Python 中的 is 和 is not 操作符
        template = dedent('''
        def func():
            # type: () -> bool
            return {lhs} {op} {rhs}
        ''')

        def test(op, args):
            # 根据模板动态生成代码
            code = template.format(lhs=args[0], rhs=args[1], op=op)
            scope = {}
            # 使用 execWrapper 执行动态生成的代码
            execWrapper(code, globals(), scope)
            # 编译生成 torch.jit.CompilationUnit 对象
            cu = torch.jit.CompilationUnit(code)
            # 断言编译后的函数执行结果与动态生成的函数执行结果相等
            self.assertEqual(
                cu.func(),
                scope['func'](),
                msg=f"Failed with op: {op}, lhs: {args[0]}, rhs: {args[1]}"
            )

        # 待测试的操作符列表
        ops = ['is', 'is not']
        # 待测试的类型文本
        type_literals = [True, False, None, [1, 1], 1, 2, .5, 1.5]

        # 对 ops 和 type_literals 进行组合，尝试不同类型的组合
        for op, lhs, rhs in product(ops, type_literals, type_literals):
            test(op, [lhs, rhs])
    def test_isinstance_refinement(self):
        @torch.jit.script
        def foo(a):
            # 定义函数foo，参数a可以是可选的整数，返回一个整数
            # 检查a是否为整数类型
            if isinstance(a, int):
                return a + 3  # 如果a是整数，则返回a加3的结果
            else:
                return 4  # 如果a不是整数，则返回4

        self.assertEqual(foo(4), 7)  # 测试foo函数对整数参数的返回结果是否为7
        self.assertEqual(foo(None), 4)  # 测试foo函数对None参数的返回结果是否为4

        @torch.jit.script
        def foo2(a, b):
            # 定义函数foo2，参数a和b可以是可选的整数，返回一个整数
            # 检查a和b是否都是整数类型
            if not isinstance(a, int) or not isinstance(b, int):
                return 0  # 如果a或者b不是整数，则返回0
            else:
                return a + b  # 如果a和b都是整数，则返回它们的和

        self.assertEqual(foo2(3, 4), 7)  # 测试foo2函数对整数参数的返回结果是否为7
        self.assertEqual(foo2(None, 4), 0)  # 测试foo2函数对其中一个参数为None的返回结果是否为0
        self.assertEqual(foo2(4, None), 0)  # 测试foo2函数对其中一个参数为None的返回结果是否为0

        @torch.jit.script
        def any_refinement(a, b):
            # 定义函数any_refinement，参数a和b可以是任意类型，返回一个整数
            # 检查a和b是否都是整数类型
            if isinstance(a, int) and isinstance(b, int):
                return a + b  # 如果a和b都是整数，则返回它们的和
            return 0  # 如果a和b不都是整数，则返回0

        self.assertEqual(any_refinement(3, 4), 7)  # 测试any_refinement函数对整数参数的返回结果是否为7
        self.assertEqual(any_refinement(3, "hi"), 0)  # 测试any_refinement函数对其中一个参数为字符串的返回结果是否为0

        @torch.jit.script
        def any_refinement2(a):
            # 定义函数any_refinement2，参数a可以是任意类型，返回一个Tensor
            # 检查a是否是Tensor类型
            if isinstance(a, Tensor):
                return a  # 如果a是Tensor类型，则直接返回a
            return torch.tensor(3)  # 如果a不是Tensor类型，则返回一个值为3的Tensor

        self.assertEqual(any_refinement2(3), torch.tensor(3))  # 测试any_refinement2函数对整数参数的返回结果是否为Tensor(3)
        self.assertEqual(any_refinement2(torch.tensor(5)), torch.tensor(5))  # 测试any_refinement2函数对Tensor参数的返回结果是否为输入的Tensor本身

    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "bug persists in deprecated executor")
    def test_unspecialized_any_binding(self):
        # 对于任意绑定，如果类型推断为特定的Tensor类型 `x`，则Dict类型将无法通过isinstance检查

        @torch.jit.script
        def foo(x: Any):
            assert isinstance(x, Dict[str, torch.Tensor])  # 断言x是字典类型，其中键为字符串，值为torch.Tensor类型

        foo({"1": torch.tensor(3)})  # 调用foo函数，传入一个包含一个键为"1"、值为torch.Tensor(3)的字典
        with self.assertRaises(Exception):
            foo(2)  # 调用foo函数，传入一个整数，预期会引发异常

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_isinstance(self):
        # 测试 isinstance 操作符进行静态类型检查
        template = dedent('''
        def func(x):
            # type: ({type_hint}) -> bool
            return isinstance(x, {typ})
        ''')

        def test(inp, typ, type_hint):
            # 根据模板生成代码
            code = template.format(typ=typ, type_hint=type_hint)
            # 创建一个空的作用域字典
            scope = {}
            # 在全局环境中执行生成的代码
            execWrapper(code, globals(), scope)
            # 使用 torch.jit.CompilationUnit 编译代码
            cu = torch.jit.CompilationUnit(code)
            # 断言编译单元中的 func 函数的输出与作用域中的 func 函数的输出相同
            self.assertEqual(
                cu.func(inp),
                scope['func'](inp),
                msg=f"Failed with typ: {typ}"
            )

        # 输入数据
        inputs = [True, 1, 1.0, torch.tensor(1), [1, 2], (1.0,), [1, 2], 1]
        # 类型字面量
        type_literals = ['bool', 'int', 'float', 'torch.Tensor', 'list', 'tuple',
                         '(list, tuple)', '(int, float, bool)']
        # 类型注解
        type_annotations = ['bool', 'int', 'float', 'Tensor', 'List[int]', 'Tuple[float]',
                            'List[int]', 'int']

        # 通过 zip 函数将不同的输入、类型字面量和类型注解组合在一起进行测试
        for inp, typ, type_hint in zip(inputs, type_literals, type_annotations):
            test(inp, typ, type_hint)

        # 测试可选的 isinstance 检查
        @torch.jit.script
        def opt_func(x):
            # type: (Optional[int]) -> bool
            return isinstance(x, int)
        # 断言 opt_func 函数对于输入值 3 返回 True
        self.assertTrue(opt_func(3))
        # 断言 opt_func 函数对于输入值 None 返回 False
        self.assertFalse(opt_func(None))
    def test_dropout_eval(self):
        # 定义一个 ScriptedConv2d 类，继承自 torch.jit.ScriptModule，用于脚本化的卷积层
        class ScriptedConv2d(torch.jit.ScriptModule):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                # 创建一个卷积层，不包含偏置，根据传入参数设置
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
                # 创建一个批归一化层，设置输出通道数和小的 epsilon 值
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

            @torch.jit.script_method
            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 执行批归一化操作
                x = self.bn(x)
                # 使用 ReLU 激活函数，并在原地执行
                return F.relu(x, inplace=True)

        # 定义一个 ScriptMod 类，继承自 torch.jit.ScriptModule，包含脚本化的模型结构
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个 ScriptedConv2d 对象，输入通道为3，输出通道为32，卷积核大小为3x3，步长为2
                self.Conv2d_1a_3x3 = ScriptedConv2d(3, 32, kernel_size=3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                # 执行卷积操作
                x = self.Conv2d_1a_3x3(x)
                # 执行 dropout 操作，根据训练状态确定是否启用
                return F.dropout(x, training=self.training)

        # 定义一个 EagerConv2d 类，继承自 torch.nn.Module，用于即时执行的卷积层
        class EagerConv2d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                # 创建一个卷积层，不包含偏置，根据传入参数设置
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
                # 创建一个批归一化层，设置输出通道数和小的 epsilon 值
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 执行批归一化操作
                x = self.bn(x)
                # 使用 ReLU 激活函数，并在原地执行
                return F.relu(x, inplace=True)

        # 定义一个 EagerMod 类，继承自 torch.nn.Module，包含即时执行的模型结构
        class EagerMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 EagerConv2d 对象，输入通道为3，输出通道为32，卷积核大小为3x3，步长为2
                self.Conv2d_1a_3x3 = EagerConv2d(3, 32, kernel_size=3, stride=2)

            def forward(self, x):
                # 执行卷积操作
                x = self.Conv2d_1a_3x3(x)
                # 执行 dropout 操作，根据训练状态确定是否启用
                return F.dropout(x, training=self.training)

        # 创建一个随机输入张量作为脚本模型的输入
        script_input = torch.rand(4, 3, 299, 299)
        # 克隆脚本输入张量，作为即时模型的输入
        eager_input = script_input.clone()

        # 使用 freeze_rng_state() 函数确保随机数生成状态冻结，用于保证结果可重复性
        with freeze_rng_state():
            # 创建并初始化一个 ScriptMod 对象
            script_mod = ScriptMod()
            # 将脚本模型设为评估模式
            script_mod.eval()
            # 执行脚本模型的前向传播计算
            script_output = script_mod(script_input)

        # 使用 freeze_rng_state() 函数确保随机数生成状态冻结，用于保证结果可重复性
        with freeze_rng_state():
            # 创建并初始化一个 EagerMod 对象
            eager_mod = EagerMod()
            # 将即时模型设为评估模式
            eager_mod.eval()
            # 执行即时模型的前向传播计算
            eager_output = eager_mod(eager_input)

        # 使用断言检查脚本模型和即时模型的输出结果是否一致
        self.assertEqual(script_output, eager_output)

        # 使用 freeze_rng_state() 函数确保随机数生成状态冻结，用于保证结果可重复性
        with freeze_rng_state():
            # 创建并初始化一个 ScriptMod 对象
            script_mod = ScriptMod()
            # 将脚本模型设为训练模式
            script_mod.train()
            # 执行脚本模型的前向传播计算
            script_output = script_mod(script_input)

        # 使用 freeze_rng_state() 函数确保随机数生成状态冻结，用于保证结果可重复性
        with freeze_rng_state():
            # 创建并初始化一个 EagerMod 对象
            eager_mod = EagerMod()
            # 将即时模型设为训练模式
            eager_mod.train()
            # 执行即时模型的前向传播计算
            eager_output = eager_mod(eager_input)

        # 使用断言检查脚本模型和即时模型的输出结果是否一致
        self.assertEqual(script_output, eager_output)
    def test_nested_breaks(self):
        def no_bool_loop_outputs(g):
            # 检验“退出”转换值不是循环块输出（因此不会影响一个循环到另一个循环的影响）
            loops = g.findAllNodes("prim::Loop")
            for loop in loops:
                for out in loop.outputs():
                    self.assertTrue(out.type() != BoolType.get())

        def test(y):
            # type: (int)
            ret = 0
            tensor = torch.tensor(0)
            while int(tensor.add_(1)) < 4:
                if y == 1:
                    continue
                for i in range(y):
                    # 跳过当前迭代，不执行后续代码
                    continue
                    ret += 1
                ret += 1
            return ret, int(tensor)

        # 断言 JIT 脚本化的 test 函数与非脚本化的输出结果一致
        self.assertEqual(torch.jit.script(test)(1), test(1))
        self.assertEqual(torch.jit.script(test)(2), test(2))
        # 检验脚本化后的 test 函数图中不存在布尔类型的循环输出
        no_bool_loop_outputs(torch.jit.script(test).graph)

        def foo():
            y = torch.tensor(0)
            z = 0
            while int(y.add_(1)) < 20:
                if int(y) < 10:
                    for i in range(6):
                        if i == 3:
                            # 继续执行下一次迭代，跳过本次循环体剩余部分
                            continue
                        else:
                            if i > 3:
                                # 中断当前循环体的执行
                                break
                        z += 2
                if int(y) == 18:
                    # 中断整个循环
                    break
                if int(y) == 15:
                    # 继续下一次迭代，跳过本次循环体剩余部分
                    continue
                z += 1
            return int(y), z

        # 检验脚本化后的 foo 函数图中不存在布尔类型的循环输出
        no_bool_loop_outputs(torch.jit.script(foo).graph)
        # 检验脚本化后的 foo 函数通过自定义检查函数 checkScript 的检查
        self.checkScript(foo, ())

        def test_nested_two():
            i = 0
            k = 0
            while i < 5:
                for j in range(5):
                    k += 1
                    if j == 3:
                        # 继续执行下一次迭代，跳过本次循环体剩余部分
                        continue
                i += 1
                k += 1
                if i == 4:
                    # 中断整个循环
                    break
            return i, k

        # 检验脚本化后的 test_nested_two 函数图中不存在布尔类型的循环输出
        self.checkScript(test_nested_two, ())
        no_bool_loop_outputs(torch.jit.script(test_nested_two).graph)

    def test_break_continue_error(self):
        # 断言捕获到 RuntimeError，并检查其异常信息是否包含特定文本
        with self.assertRaisesRegex(RuntimeError, "Syntax"):
            cu = torch.jit.CompilationUnit('''
            def other_func(a):
                break
                ''')

        with self.assertRaisesRegex(RuntimeError, "Syntax"):
            cu = torch.jit.CompilationUnit('''
            def other_func(a):
                for i in range(5):
                    def foo():
                        break
                ''')

        with self.assertRaisesRegex(RuntimeError, "do not support break or continue inside"):
            @torch.jit.script
            def foo(x):
                i = 0
                for a in (1, "2", 1.5):
                    b = a
                    if x:
                        # 在脚本化函数中不支持 break 或 continue
                        break
                return b
    def test_python_call(self):
        # 定义一个简单的 Python 函数
        def pyfunc(a):
            return a * 3.0

        # 使用 torch.jit.CompilationUnit 定义一个 Torch 脚本
        cu = torch.jit.CompilationUnit('''
        # 定义另一个简单的 Python 函数
        def other_func(a):
            return a + a

        # 定义一个测试函数 test_call_python，调用了之前定义的 pyfunc 和 other_func
        def test_call_python(a):
            b = pyfunc(a)  # 调用 pyfunc 函数
            b = other_func(b)  # 调用 other_func 函数
            i = 0
            step = 1
            while i < 10:
                b = pyfunc(b)  # 再次调用 pyfunc 函数
                if bool(b > 3.0):  # 判断 b 是否大于 3.0
                    b = pyfunc(b)  # 如果大于 3.0，则再次调用 pyfunc 函数
                i = 11  # 设置 i = 11，退出 while 循环
            return b  # 返回 b
        ''')
        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([54], torch.float)

        self.assertEqual(cu.test_call_python(*inputs), outputs[0])

    def test_python_call_failure(self):
        # 使用 assertRaisesRegex 检测 RuntimeError 异常并验证消息是否包含 "undefined value pyfunc2"
        with self.assertRaisesRegex(RuntimeError, "undefined value pyfunc2"):
            # 定义一个简单的 Python 函数
            def pyfunc(a):
                return a * 3.0

            # 使用 torch.jit.CompilationUnit 定义一个 Torch 脚本
            cu = torch.jit.CompilationUnit('''
            # 定义另一个简单的 Python 函数
            def other_func(a):
                return a + a

            # 定义一个测试函数 test_call_python，调用了之前定义的 pyfunc 和 other_func
            def test_call_python(a):
                b = pyfunc(a)  # 调用 pyfunc 函数
                b = other_func(b)  # 调用 other_func 函数
                i = 0
                step = 1
                while i < 10:
                    b = pyfunc2(b)  # 引发错误，调用了未定义的函数 pyfunc2
                    if b > 3.0:
                        b = pyfunc(b)  # 如果 b 大于 3.0，则再次调用 pyfunc 函数
                    i = 11  # 设置 i = 11，退出 while 循环
                return b  # 返回 b
            ''')
            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([54], torch.float)

            self.assertEqual(cu.test_call_python(*inputs), outputs)

    def test_type_call_in_script(self):
        # 使用 torch.jit.script 装饰器定义一个 Torch 脚本
        @torch.jit.script
        def fn(x):
            return type(x)

        # 使用 assertRaisesRegex 检测 RuntimeError 异常并验证消息是否包含 "value of type _TensorMeta"
        with self.assertRaisesRegex(RuntimeError, "value of type _TensorMeta"):
            fn(torch.tensor(.5))

    def test_python_call_annotation(self):
        # 定义一个简单的 Python 函数
        def pyfunc(a):
            return a * 3.0

        # 使用 torch.jit.script 装饰器定义一个 Torch 脚本
        @torch.jit.script
        def foo(a):
            return pyfunc(a) + pyfunc(a)

        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([6], torch.float)
        self.assertEqual(foo(*inputs), outputs[0])

    def test_python_call_annoytation_failure(self):
        # 使用 assertRaisesRegex 检测 RuntimeError 异常并验证消息是否包含 "undefined value pyfunc2"
        with self.assertRaisesRegex(RuntimeError, "undefined value pyfunc2"):
            # 定义一个简单的 Python 函数
            def pyfunc(a):
                return a * 3.0

            # 使用 torch.jit.script 装饰器定义一个 Torch 脚本
            @torch.jit.script
            def foo(a):
                return pyfunc2(a) + pyfunc(a)  # 引发错误，调用了未定义的函数 pyfunc2

            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([6], torch.float)

            self.assertEqual(foo(*inputs), outputs[0])

    def test_desugar_module(self):
        # 导入 torch.nn.functional 模块
        import torch.nn.functional as F

        # 定义一个函数 fn，调用了 torch.nn.functional.prelu 和 F.prelu 函数
        def fn(x, slope):
            a = torch.abs(x)
            b = torch.nn.functional.prelu(x, slope)
            c = F.prelu(x, slope)
            return a, b, c

        x = torch.arange(-3., 4)
        slope = torch.tensor([0.5])
        # 使用 self.checkScript 检查 fn 函数的 Torch 脚本化
        self.checkScript(fn, [x, slope], optimize=True)
    # 定义一个测试方法 test_script_docstring
    def test_script_docstring(self):
        # 使用 torch.jit.script 装饰器将函数 with_docstring 转换为 Torch 脚本
        @torch.jit.script
        def with_docstring(x):
            """test str"""
            # 将参数 x 赋值给变量 y
            y = x
            """y is the same as x"""
            # 返回变量 y
            return y
        # 断言 with_docstring 函数的文档字符串为 'test str'
        self.assertEqual(with_docstring.__doc__, 'test str')

    # 定义一个测试方法 test_script_method_docstring
    def test_script_method_docstring(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 A
        class A(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器将方法 with_docstring 转换为 Torch 脚本方法
            @torch.jit.script_method
            def with_docstring(self, x):
                """test str"""
                # 将参数 x 赋值给变量 y
                y = x
                """y is the same as x"""
                # 返回变量 y
                return y
        # 创建类 A 的实例 a
        a = A()
        # 断言 a.with_docstring 方法的文档字符串为 'test str'
        self.assertEqual(a.with_docstring.__doc__, 'test str')
    def test_script_module(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M1
        class M1(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化一个参数 weight，形状为 (2,)，其值为随机生成的张量
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            # 定义一个 torch.jit.script_method 的方法 forward，用于前向计算
            def forward(self, thing):
                # 返回 self.weight 与输入张量 thing 的加法结果
                return self.weight + thing

        # 定义一个继承自 nn.Module 的类 PModule
        class PModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个参数 a，形状为 (2, 3)，其值为随机生成的张量
                self.a = nn.Parameter(torch.randn(2, 3))

            def forward(self, a):
                # 返回 self.a 与输入张量 a 的矩阵乘法结果
                return self.a.mm(a)

        # 定义一个继承自 torch.jit.ScriptModule 的类 M2
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 实例化类 M1，并赋值给属性 sub，用于测试子模块
                self.sub = M1()
                # 实例化类 PModule，并赋值给属性 sub2，用于测试子模块
                self.sub2 = PModule()
                # 初始化一个参数 weight，形状为 (2, 3)，其值为随机生成的张量，用于测试参数
                self.weight = nn.Parameter(torch.randn(2, 3))
                # 初始化一个参数 bias，形状为 (2,)，其值为随机生成的张量，用于测试参数
                self.bias = nn.Parameter(torch.randn(2))
                # 使用字符串定义一个方法 hi，用于测试从字符串定义方法
                self.define("""
                    def hi(self, a):
                        return self.weight.mm(a)
                """)

            @torch.jit.script_method
            # 定义一个 torch.jit.script_method 的方法 doit，用于前向计算
            def doit(self, input):
                # 返回 self.weight 与输入张量 input 的矩阵乘法结果，用于测试参数的使用
                return self.weight.mm(input)

            @torch.jit.script_method
            # 定义一个 torch.jit.script_method 的方法 doit2，用于前向计算
            def doit2(self, input):
                # 返回 self.weight 与输入张量 input 的矩阵乘法结果，用于测试参数的使用
                return self.weight.mm(input)

            @torch.jit.script_method
            # 定义一个 torch.jit.script_method 的方法 forward，用于前向计算
            def forward(self, input):
                # 调用 doit 方法计算结果 a
                a = self.doit(input)
                # 调用 doit2 方法计算结果 b
                b = self.doit2(input)
                # 调用 hi 方法计算结果 c
                c = self.hi(input)
                # 调用 sub2 的 forward 方法计算结果 d
                d = self.sub2(input)
                # 返回 a + b + self.bias + self.sub(a) + c + d 的结果，用于测试整体前向计算
                return a + b + self.bias + self.sub(a) + c + d

        # 关闭 torch.jit 的优化执行
        with torch.jit.optimized_execution(False):
            # 实例化类 M2
            m2 = M2()
            # 生成一个形状为 (3, 2) 的随机输入张量 input
            input = torch.randn(3, 2)
            # 计算 m2.weight 与 input 的矩阵乘法结果 a
            a = m2.weight.mm(input)
            # 计算 m2.weight 与 input 的矩阵乘法结果 b
            b = m2.weight.mm(input)
            # 计算 m2.weight 与 input 的矩阵乘法结果 c
            c = m2.weight.mm(input)
            # 计算 m2.sub2.a 与 input 的矩阵乘法结果 d
            d = m2.sub2.a.mm(input)
            # 计算参考结果 ref，用于断言
            ref = a + b + m2.bias + m2.sub.weight + a + c + d
            # 断言 m2.forward(input) 的结果与 ref 相等
            self.assertEqual(ref, m2.forward(input))
            # 将 m2.weight、m2.bias、m2.sub.weight 和 m2.sub2.a 的值分别置零
            m2.weight = nn.Parameter(torch.zeros_like(m2.weight))
            m2.bias = nn.Parameter(torch.zeros_like(m2.bias))
            m2.sub.weight = nn.Parameter(torch.zeros_like(m2.sub.weight))
            m2.sub2.a.data.zero_()
            # 断言 m2.forward(torch.randn(3, 2)) 的结果为形状为 (2, 2) 的全零张量
            self.assertEqual(torch.zeros(2, 2), m2.forward(torch.randn(3, 2)))

    def test_irparser(self):
        # 定义一个字符串 graph_str，表示一个计算图
        graph_str = """graph(%0 : Double(5, 5)):
          # CHECK: aten::relu
          %1 : Double(5, 5) = aten::relu(%0)
          return (%1)
        """
        # 使用 FileCheck() 检查 graph_str 的解析结果，调用 parse_ir 函数解析 graph_str
        FileCheck().run(graph_str, parse_ir(graph_str))
    # 定义测试函数，用于解析张量常量
    def test_parse_tensor_constants(self):
        # 定义内部函数 foo，返回一个 4x4 的全零张量
        def foo():
            return torch.zeros([4, 4])

        # 对函数 foo 进行 Torch 脚本化，生成脚本化的函数对象 foo_s
        foo_s = torch.jit.script(foo)
        # 对 foo_s 的图进行常量传播优化
        torch._C._jit_pass_constant_propagation(foo_s.graph)

        # 将 foo_s 的图转换为字符串表示形式
        g = str(foo_s.graph)
        # 解析 IR 字符串 g，同时解析张量常量
        g_parsed = parse_ir(g, parse_tensor_constants=True)
        # 断言解析后的规范形式与原始图的规范形式相等
        self.assertEqual(str(canonical(g_parsed)), str(canonical(foo_s.graph)))

        # 从解析后的 IR 创建 Torch 函数对象 func
        func = torch._C._create_function_from_graph("forward", g_parsed)

        # 使用 func 运行得到的结果 out_parsed
        out_parsed = func()
        # 直接调用 foo 函数得到的结果 out_func
        out_func = foo()

        # 不检查数据，只比较数据类型、大小等信息
        out_parsed[:] = 0
        out_func[:] = 0
        # 断言两者结果相等
        self.assertEqual(out_func, out_parsed)

        # 检查当不解析张量常量时，是否会抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            parse_ir(g, parse_tensor_constants=False)

    # 定义测试函数，用于解析嵌套命名
    def test_parse_nested_names(self):
        # 定义一个 IR 字符串表示的计算图 g_str
        g_str = """
    graph(%x.1 : Tensor):
        %3 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %hi.submod.value.5 : Tensor = aten::add(%x.1, %2, %3)
        return (%hi.submod.value.5)
        """
        # 解析 IR 字符串 g_str，得到计算图 g
        g = parse_ir(g_str)
        # 对解析后的计算图 g 再次进行解析，以确保序列化后的一致性
        round_trip_g = parse_ir(str(g))
        # 断言原始图 g 和回转后的图 round_trip_g 的规范形式是否相等
        self.assertEqual(canonical(g), canonical(round_trip_g))

        # 从计算图 g 和 round_trip_g 分别创建 Torch 函数对象 func1 和 func2
        func1 = torch._C._create_function_from_graph("forward", g)
        func2 = torch._C._create_function_from_graph("forward", round_trip_g)

        # 使用 func1 和 func2 分别计算张量输入为全 1 的结果，并进行比较
        self.assertEqual(func1(torch.ones([2])), func2(torch.ones([2])))
    def test_is_after_use(self):
        # 定义一个内部函数，用于按照使用顺序排序给定图中的输入使用节点
        def sorted_input_use(g):
            # 获取图的下一个输入使用节点列表，并将其转换为列表
            uses = list(next(g.inputs()).uses())
            # 使用 functools.cmp_to_key 将 uses[0] 的类型的 isAfter 方法作为排序关键字，进行排序
            return sorted(uses, key=functools.cmp_to_key(type(uses[0]).isAfter))

        # 定义一个 Torch 脚本函数 foo，计算 x + 1 并返回元组 (x, x, a)
        @torch.jit.script
        def foo(x):
            a = x + 1
            return (x, x, a)

        # 对 foo 的使用节点按照 isAfter 方法排序
        uses_sorted = sorted_input_use(foo.graph)
        # 断言第一个使用节点不在第二个使用节点之后
        self.assertFalse(uses_sorted[0].isAfter(uses_sorted[1]))
        # 断言第一个使用节点的类型为 "aten::add"
        self.assertTrue(uses_sorted[0].user.kind() == "aten::add")
        # 断言第二个使用节点的偏移量为 0
        self.assertEqual(uses_sorted[1].offset, 0)

        # 定义一个 Torch 脚本函数 foo，根据条件 cond 计算 x + 3 或 x - 3 并返回结果
        @torch.jit.script
        def foo(x, cond: bool):
            if cond:
                return x + 3
            else:
                return x - 3

        # 对 foo 的使用节点按照 isAfter 方法排序
        uses_sorted = sorted_input_use(foo.graph)
        # 断言第一个使用节点的类型为 "aten::add"
        self.assertTrue(uses_sorted[0].user.kind() == "aten::add")
        # 断言第二个使用节点的类型为 "aten::sub"
        self.assertTrue(uses_sorted[1].user.kind() == "aten::sub")

        # 定义一个 Torch 脚本函数 foo，根据多个条件 cond 和 cond2 计算 x + 3、x - 3 或 x / 3 并返回结果
        @torch.jit.script
        def foo(x, cond: bool, cond2: bool):
            if cond:
                return x + 3
            elif cond2:
                return x - 3
            else:
                return x / 3

        graph1 = foo.graph

        # 定义一个 Torch 脚本函数 foo，根据多个条件 cond 和 cond2 计算 x + 3、x - 3 或 x / 3 并返回结果
        @torch.jit.script
        def foo(x, cond: bool, cond2: bool):
            if cond:
                return x + 3
            else:
                if cond2:
                    return x - 3
                return x / 3

        graph2 = foo.graph

        # 对两个图的使用节点按照 isAfter 方法排序，并分别进行断言检查
        for graph in [graph1, graph2]:
            uses_sorted = sorted_input_use(graph)
            # 断言第一个使用节点的类型为 "aten::add"
            self.assertTrue(uses_sorted[0].user.kind() == "aten::add")
            # 断言第二个使用节点的类型为 "aten::sub"
            self.assertTrue(uses_sorted[1].user.kind() == "aten::sub")
            # 断言第三个使用节点的类型为 "aten::div"
            self.assertTrue(uses_sorted[2].user.kind() == "aten::div")
   `
    def test_canonicalize_control_outputs(self):
        def test_all_outputs(g):
            # 查找图中所有的 "prim::If" 节点
            ifs = g.findAllNodes("prim::If")
            # 查找图中所有的 "prim::Loop" 节点
            loops = g.findAllNodes("prim::Loop")

            # 定义一个函数，计算一个节点包含的 "prim::If" 和 "prim::Loop" 的数量
            def contained_blocks(node):
                return len(node.findAllNodes("prim::If")) * 2 + len(node.findAllNodes("prim::Loop"))
            # 遍历所有的 "prim::If" 和 "prim::Loop" 节点
            for node in ifs + loops:
                # 获取节点的所有输出
                outs = list(node.outputs())
                # 获取所有输出的名字
                out_name = [x.debugName() for x in outs]
                # 如果没有输出，则跳过
                if len(out_name) == 0:
                    continue
                # 创建 FileCheck 对象
                fc = FileCheck()
                # 查找最后一个输出及其所有后续使用
                fc.check(out_name[-1] + " : ")
                # 跳过节点体中的内容
                for i in range(contained_blocks(node)):
                    fc.check("->")
                # 根据节点类型进行不同的校验
                if (node.kind() == "prim::If"):
                    fc.check("->").check("->").check("\n")
                else:
                    fc.check("->").check("\n")
                # 确保文件的规范顺序与第一个使用的顺序一致
                for name in out_name:
                    fc.check(name)
                # 运行 FileCheck 校验
                fc.run(g)

        @torch.jit.script
        def test(x):
            # 定义一个函数，输入类型为 bool，返回值类型为 Tuple[int, int]
            # 初始化变量
            b = 2
            a = 1
            # 根据 x 的值，修改变量 a 和 b 的值
            if x:
                a = 1
                b = 2
                x = False
            if x:
                b = a
            else:
                a = b

            # 返回变量 a 和 b 的值
            return a, b
        # 调用 test_all_outputs 函数，传入 test 的图
        test_all_outputs(test.graph)

        @torch.jit.script
        def test2(x):
            # 定义一个函数，输入类型为 bool，返回值类型为 Tuple[int, int]
            # 初始化变量
            b = 2
            a = 1
            # 根据 x 的值，修改变量 a 和 b 的值
            if x:
                a = 1
                b = 2
                x = False
            if x:
                print(a)
            else:
                if x:
                    print(b)

            # 返回变量 a 和 b 的值
            return a, b
        # 调用 test_all_outputs 函数，传入 test2 的图
        test_all_outputs(test2.graph)

        @torch.jit.script
        def test_loop(x, iter):
            # 定义一个函数，输入类型为 bool 和 int，返回值类型为 None
            # 初始化变量
            a = 1
            b = 2
            c = 3
            # 循环 iter 次
            for i in range(iter):
                a = 4
                b = 5
                c = 6
                x = True
            # 输出变量 c 的值
            print(c)
            # 根据 x 的值，输出变量 a 和 b 的值
            if x:
                print(a, b)
        # 调用 test_all_outputs 函数，传入 test_loop 的图
        test_all_outputs(test_loop.graph)

        @torch.jit.script
        def loop_unused(iter):
            # 定义一个函数，输入类型为 int，返回值类型为 None
            # 初始化变量
            a = 1
            b = 2
            c = 3
            # 循环 iter 次
            for i in range(iter):
                c = c + 1
                b = b + 1
                a = a + 1
                print(a, b)
            # 输出变量 c 的值
            print(c)

        # 校验变量 c 是使用过的，然后未使用的变量按字母顺序排序
        FileCheck().check(r"%c : int, %a : int, %b : int").run(loop_unused.graph)
    def _dtype_to_jit_name(self, dtype):
        # 根据给定的 PyTorch 数据类型，返回对应的 JIT 名称
        if dtype == torch.float32:
            return "Float"
        if dtype == torch.float64:
            return "Double"
        if dtype == torch.int64:
            return "Long"
        if dtype == torch.int32:
            return "Int"
        if dtype == torch.bool:
            return "Bool"
        # 如果传入的 dtype 没有匹配的处理方式，则抛出运行时错误
        raise RuntimeError('dtype not handled')

    def _dtype_to_expect(self, dtype, dim=0):
        # 根据数据类型和维度生成预期的 JIT 类型字符串
        param = ', '.join(['*'] * dim + ['device=cpu'])
        param = '(' + param + ')'
        jit_type = self._dtype_to_jit_name(dtype)
        if dim >= 0:
            return jit_type + param
        # 对于特殊情况，表示包装的数字
        else:
            return jit_type.lower()

    def _test_dtype_op_shape(self, ops, args, input_dims=1):
        if input_dims < 1:
            # 如果输入维度小于1，则抛出运行时错误
            raise RuntimeError("input dims must be at least 1")
        dtypes = [torch.float32, torch.float64, torch.int64, torch.int32]
        str_args = ', '.join([str(arg) for arg in args]) + (', ' if len(args) else '')
        tensor_data = ('[' * input_dims) + '1, 2, 3' + (input_dims * ']')
        template = dedent('''
        def func():
            return {return_line}
        ''')

        for op in ops:
            for dtype in (dtypes + [None]):
                for tensor_type in dtypes:
                    # 对于非浮点类型，跳过一些操作
                    if not tensor_type.is_floating_point or (dtype is not None and not dtype.is_floating_point):
                        if op in ['mean', 'softmax', 'log_softmax']:
                            continue
                    return_line = f"torch.tensor({tensor_data}, dtype={tensor_type}).{op}({str_args}dtype={dtype})"
                    # 如果需要调试失败的测试，取消下面一行的注释：
                    # print("testing {}".format(return_line))
                    code = template.format(return_line=return_line)
                    scope = {}
                    exec(code, globals(), scope)
                    cu = torch.jit.CompilationUnit(code)
                    graph = cu.func.graph
                    torch._C._jit_pass_complete_shape_analysis(graph, (), False)
                    input_array = [1, 2, 3]
                    for _ in range(1, input_dims):
                        input_array = [input_array]
                    t = torch.tensor(input_array, dtype=tensor_type)
                    attr = getattr(t, op)
                    kwargs = {'dtype': dtype}
                    result = attr(*args, **kwargs)
                    expect = self._dtype_to_expect(result.dtype, result.dim())
                    FileCheck().check("aten::tensor").check(expect).run(graph)

    def test_dtype_op_shape(self):
        ops = ['prod']
        self._test_dtype_op_shape(ops, args=[])
        self._test_dtype_op_shape(ops, args=[0, False])
        self._test_dtype_op_shape(ops, args=[0, False])
        self._test_dtype_op_shape(ops, args=[0, True])
    # 定义测试方法，测试多种运算类型的函数在指定参数下的行为
    def test_dtype_op_shape2(self):
        # 指定测试的运算类型列表
        ops = ['cumprod', 'cumsum', 'softmax', 'log_softmax']
        # 测试以输入参数0调用 _test_dtype_op_shape 方法
        self._test_dtype_op_shape(ops, args=[0])

        # 测试以输入参数1和4维输入调用 _test_dtype_op_shape 方法
        self._test_dtype_op_shape(ops, args=[1], input_dims=4)


    # 定义测试方法，测试二元运算的形状
    def _test_binary_op_shape(self, ops, input_dims=1):

        # 定义数据类型列表
        dtypes = [torch.float32, torch.float64, torch.int64, torch.int32, torch.bool]

        # 根据输入的维度数确定形状
        if input_dims == 0:
            shape = '1'
        else:
            shape = '[' + ('1,' * 4) + ']'
            for _ in range(1, input_dims):
                shape = '[' + ",".join([shape] * 4) + ']'

        # 定义模板字符串，用于生成测试函数的代码
        template = dedent('''
        def func():
            arg1 = {}
            arg2 = {}
            return torch.{}(arg1, arg2)
        ''')

        # 初始化参数列表
        args = []
        for dtype in dtypes:
            args = args + [f"torch.tensor({shape}, dtype={dtype})"]
        args = args + [1, 1.5]

        # 检查是否为布尔类型
        def isBool(arg):
            return type(arg) == bool or (type(arg) == str and "torch.bool" in arg)

        # 遍历运算类型和参数组合，生成测试代码并执行
        for op in ops:
            for first_arg in args:
                for second_arg in args:
                    # 对于不支持布尔类型的减法和除法运算，跳过
                    if (op == 'sub' or op == 'div') and (isBool(first_arg) or isBool(second_arg)):
                        continue
                    # 对于混合类型或整数参数的除法运算，跳过
                    if (op == 'div' and (type(first_arg) != type(second_arg) or
                       isinstance(first_arg, int) or
                       (isinstance(first_arg, str) and 'int' in first_arg))):
                        continue
                    # 生成运算函数的返回字符串
                    return_line = f"torch.{op}({first_arg}, {second_arg})"
                    # 执行模板代码，创建局部作用域，运行生成的函数并获取结果
                    code = template.format(first_arg, second_arg, op)
                    scope = {}
                    exec(code, globals(), scope)
                    non_jit_result = scope['func']()

                    # 使用 JIT 编译单元创建图形并进行形状分析
                    cu = torch.jit.CompilationUnit(code)
                    graph = cu.func.graph
                    torch._C._jit_pass_complete_shape_analysis(graph, (), False)
                    # 使用 dim=-1 表示一个 Python/JIT 标量
                    dim = -1 if type(first_arg) != str and type(second_arg) != str else non_jit_result.dim()
                    dtype = non_jit_result.dtype
                    # JIT 只支持整数/浮点数标量
                    if dim < 0:
                        if dtype == torch.int64:
                            dtype = torch.int32
                        if dtype == torch.float64:
                            dtype = torch.float32
                    # 将期望结果转换为具体的预期值
                    expect = self._dtype_to_expect(dtype, dim)
                    jit_output = next(graph.outputs())

                    # 运行 FileCheck 来验证预期输出
                    check = FileCheck()
                    check.check(expect).run(str(jit_output))
    def test_filecheck_parse(self):
        # 定义测试函数，用于检查文件内容是否符合预期
        def test_check():
            # 第一个测试文件内容
            file = """
                # CHECK: 2
                # CHECK: 3
                # CHECK: 2
                232
                """
            # 使用 FileCheck 类运行文件检查，验证是否包含预期的内容
            FileCheck().run(checks_file=file, test_file=file)

            # 第二个测试文件内容
            file = """
                # CHECK: 232
                232
                """
            # 再次使用 FileCheck 类运行文件检查，验证是否包含预期的内容
            FileCheck().run(file, "232")

            # 使用断言验证运行时错误是否会抛出，期望找到 "232"
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "232"'):
                FileCheck().run(file, "22")

            # 使用断言验证运行时错误是否会抛出，期望找到 "22"
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "22"'):
                FileCheck().run("# CHECK: 22", "23")

        # 执行文件内容检查的测试函数
        test_check()

        # 定义检查计数的测试函数
        def test_check_count():
            # 测试文件内容为 "22222"
            file = "22222"
            # 使用 FileCheck 类运行检查，期望找到 5 次 "2"
            FileCheck().run("# CHECK-COUNT-5: 2", file)
            # 使用 FileCheck 类运行检查，期望找到恰好 5 次 "2"
            FileCheck().run("# CHECK-COUNT-EXACTLY-5: 2", file)
            # 使用 FileCheck 类运行检查，期望找到 2 次 "22"
            FileCheck().run("# CHECK-COUNT-2: 22", file)
            # 使用 FileCheck 类运行检查，期望找到 1 次 "222"
            FileCheck().run("# CHECK-COUNT-1: 222", file)

            # 使用断言验证运行时错误是否会抛出，期望未找到
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find'):
                FileCheck().run("# CHECK-COUNT-EXACTLY-2: 2", file)

        # 执行检查计数的测试函数
        test_check_count()

        # 定义检查相同内容的测试函数
        def test_check_same():
            # 测试文件内容为 "22\n33"
            file = "22\n33"
            # 使用 FileCheck 类运行检查，期望找到 "22"
            FileCheck().run("# CHECK-SAME: 22", file)

            # 使用断言验证运行时错误是否会抛出，期望未找到 "33"
            with self.assertRaisesRegex(RuntimeError, "Expected to not find"):
                FileCheck().run("# CHECK-SAME: 33", file)

            # 文件内容为 "22  1  3"
            file = "22  1  3"

            # 使用 FileCheck 类运行检查，期望找到 "2" 并且 "3" 在相同行
            FileCheck().run("# CHECK: 2\n # CHECK-SAME: 3", file)
            # 使用 FileCheck 类运行检查，期望找到 2 次 "2" 并且 "3" 在相同行
            FileCheck().run("# CHECK-COUNT-2: 2\n # CHECK-SAME: 3", file)

        # 执行检查相同内容的测试函数
        test_check_same()

        # 定义测试不良输入的测试函数
        def test_bad_input():
            # 使用断言验证运行时错误是否会抛出，期望找到 "Check for bad input"
            with self.assertRaisesRegex(RuntimeError, "Check for bad input"):
                FileCheck().run("", "1")

            # 使用断言验证运行时错误是否会抛出，期望找到 "Could not parse check"
            with self.assertRaisesRegex(RuntimeError, "Could not parse check"):
                FileCheck().run("# CHECK1", "")

        # 执行测试不良输入的测试函数
        test_bad_input()
    def test_module_apis(self):
        class Sub(torch.nn.Module):
            def forward(self, thing):
                return thing - 2

        class Double(torch.nn.Module):
            def forward(self, thing):
                return thing * 2

        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = (Sub())  # 创建一个 Sub 模块实例，并将其赋值给 self.mod
                self.mod2 = (Sub())  # 创建另一个 Sub 模块实例，并将其赋值给 self.mod2
                self.mod3 = nn.Sequential(nn.Sequential(Sub()))  # 创建一个包含一个 Sub 模块的 Sequential 容器，并赋值给 self.mod3
                self.mod4 = nn.Sequential(Sub(), Double())  # 创建一个包含 Sub 和 Double 模块的 Sequential 容器，并赋值给 self.mod4

            @torch.jit.export
            def method(self, x, x1, y, y1):
                mod_names = ""
                for name, mod in self.named_modules():  # 遍历所有子模块（包括自己），拼接子模块名称到 mod_names
                    mod_names = mod_names + " " + name
                    x = mod(x)  # 逐个对输入 x 应用子模块的 forward 方法

                children_names = ""
                for name, mod in self.named_children():  # 遍历所有子模块，拼接子模块名称到 children_names
                    children_names = children_names + " " + name
                    x1 = mod(x1)  # 逐个对输入 x1 应用子模块的 forward 方法

                for mod in self.modules():  # 遍历所有模块（包括自己），对输入 y 应用每个模块的 forward 方法
                    y = mod(y)

                for mod in self.children():  # 遍历所有子模块，对输入 y1 应用每个子模块的 forward 方法
                    y1 = mod(y1)

                return mod_names, children_names, x, x1, y, y1  # 返回拼接的模块名称字符串和处理后的输入结果

            def forward(self, x):
                return x + 2  # 对输入 x 执行加法操作，并返回结果

        mod = torch.jit.script(MyMod())  # 使用 Torch JIT 将 MyMod 模块转换为脚本模块
        inps = tuple([torch.tensor(i) for i in range(1, 5)])  # 创建输入张量元组，包含数值 1 到 4 的张量
        self.assertEqual(mod.method(*inps), MyMod().method(*inps))  # 断言脚本模块和原始模块使用相同输入时的 method 方法输出相等
    # 定义一个测试方法，用于测试包含常量的脚本模块的行为
    def test_script_module_const(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 M
        class M(torch.jit.ScriptModule):
            # 定义类级别的常量列表
            __constants__ = ['b', 'i', 'c', 's']

            # 初始化方法
            def __init__(self):
                super().__init__()
                # 设置四个常量属性
                self.b = False
                self.i = 1
                self.c = 3.5
                self.s = ["hello"]

            # torch.jit.script_method 装饰器，声明 forward 方法为脚本方法
            @torch.jit.script_method
            def forward(self):
                # 返回三个常量属性的值
                return self.b, self.i, self.c

        # 关闭优化执行上下文
        with torch.jit.optimized_execution(False):
            # 创建 M 类的实例
            m = M()
            # 调用实例，获取返回值并解包
            o0, o1, o2 = m()
        # 断言返回值符合预期
        self.assertEqual(o0, 0)
        self.assertEqual(o1, 1)
        self.assertEqual(o2, 3.5)

    # 测试脚本模块的异常情况：访问不存在的属性
    def test_script_module_fail_exist(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 M
        class M(torch.jit.ScriptModule):
            # torch.jit.script_method 装饰器，声明 forward 方法为脚本方法
            @torch.jit.script_method
            def forward(self, x):
                # 尝试访问不存在的属性 self.whatisgoingon
                return x + self.whatisgoingon
        # 断言抛出 RuntimeError 异常，包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "Module 'M' has no attribute"):
            M()

    # 跳过当前测试：可选属性的 NoneType 精化不起作用
    @unittest.skip("[module dedupe] currently NoneType refinement on optional attributes doesn't work.")
    def test_script_module_none_exist_fail(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 M
        class M(torch.jit.ScriptModule):
            # 初始化方法，接受一个可选参数 my_optional
            def __init__(self, my_optional):
                super().__init__()
                # 设置实例属性 my_optional
                self.my_optional = my_optional

            # torch.jit.script_method 装饰器，声明 forward 方法为脚本方法
            @torch.jit.script_method
            def forward(self, x):
                # 如果 my_optional 不为 None，则执行计算，否则只取反
                if self.my_optional is not None:
                    return torch.neg(x) + self.my_optional
                return torch.neg(x)
        # 断言抛出 RuntimeError 异常，包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "has no attribute 'my_optional'"):
            x = torch.rand(3, 4)
            fb = M(None)
            fb(x)

    # 测试脚本模块的异常情况：定义无效的常量属性
    def test_script_module_invalid_consts(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 Foo
        class Foo(torch.jit.ScriptModule):
            # 定义一个包含无效常量属性 'invalid' 的类级别常量列表
            __constants__ = ['invalid']

            def __init__(self):
                super().__init__()
                # 尝试将 nn.Linear(3, 4) 对象赋给无效常量属性
                self.invalid = [nn.Linear(3, 4)]

        # 断言抛出 TypeError 异常，包含指定错误信息
        with self.assertRaisesRegex(
                TypeError,
                "Linear' object in attribute 'Foo.invalid' is not a valid constant"):
            Foo()

        # 定义一个继承自 torch.jit.ScriptModule 的内部类 Foo2
        class Foo2(torch.jit.ScriptModule):
            # 定义一个包含无效常量属性 'invalid' 的类级别常量列表
            __constants__ = ['invalid']

            def __init__(self):
                super().__init__()
                # 尝试将 int 类型对象赋给无效常量属性
                self.invalid = int

        # 断言抛出 TypeError 异常，包含指定错误信息
        with self.assertRaisesRegex(TypeError, "not a valid constant"):
            Foo2()

        # 定义一个继承自 torch.jit.ScriptModule 的内部类 Foo3
        class Foo3(torch.jit.ScriptModule):
            # 定义一个包含无效常量属性 'invalid' 的类级别常量列表
            __constants__ = ['invalid']

            def __init__(self):
                super().__init__()
                # 尝试将元组对象赋给无效常量属性
                self.invalid = (3, 4, {})

        # 断言抛出 TypeError 异常，包含指定错误信息
        with self.assertRaisesRegex(TypeError, "not a valid constant"):
            Foo3()

        # 定义一个继承自 torch.jit.ScriptModule 的内部类 Foo4
        class Foo4(torch.jit.ScriptModule):
            # 定义一个包含无效常量属性 'invalid' 的类级别常量列表
            __constants__ = ['invalid']

            def __init__(self):
                super().__init__()
                # 尝试将 np.int64(5) 对象赋给无效常量属性
                self.invalid = np.int64(5)

        # 断言抛出 TypeError 异常，包含指定错误信息
        # 验证异常信息是否包含可读性强的类名 'numpy.int64'
        with self.assertRaisesRegex(TypeError, "numpy.int64"):
            Foo4()
    # 定义一个测试方法，用于测试脚本模块参数缓冲区变异的情况
    def test_script_module_param_buffer_mutation(self):
        # TODO: add param mutation test case after JIT support it
        # 定义一个继承自 torch.jit.ScriptModule 的类 ModuleBufferMutate
        class ModuleBufferMutate(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 注册一个名为 'running_var' 的缓冲区，初始值为0，数据类型为 torch.long
                self.register_buffer('running_var', torch.tensor(0, dtype=torch.long))

            @torch.jit.script_method
            # 定义一个前向方法
            def forward(self):
                # 如果处于训练模式，对 running_var 进行自增操作
                if self.training:
                    self.running_var += 1
                return self.running_var

        # 禁用优化执行
        with torch.jit.optimized_execution(False):
            # 创建 ModuleBufferMutate 类的实例 m
            m = ModuleBufferMutate()
            # 断言调用 m() 后返回值为 1
            self.assertEqual(m(), 1)
            # 将 m 设为评估模式
            m.eval()
            # 再次断言调用 m() 后返回值为 1
            self.assertEqual(m(), 1)

    # 定义一个测试方法，用于测试脚本模块中的 for 循环
    def test_script_module_for(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            __constants__ = ['b']

            def __init__(self):
                super().__init__()
                # 初始化属性 b 为一个包含 [1, 2, 3, 4] 的列表
                self.b = [1, 2, 3, 4]

            @torch.jit.script_method
            # 定义一个前向方法
            def forward(self):
                # 初始化 sum 为 0
                sum = 0
                # 遍历 self.b 中的元素，将它们累加到 sum 中
                for i in self.b:
                    sum += i
                return sum

        # 禁用优化执行
        with torch.jit.optimized_execution(False):
            # 创建类 M 的实例 m
            m = M()
            # 断言调用 m() 后返回值为 10
            self.assertEqual(m(), 10)

    # 定义一个测试方法，用于测试魔术方法的重写
    def test_override_magic(self):
        # 定义一个继承自 nn.Module 的类 OverrideMagic
        class OverrideMagic(nn.Module):
            @torch.jit.export
            # 重写魔术方法 __len__，返回固定值 10
            def __len__(self):
                return 10

        # 创建 OverrideMagic 类的实例 mod
        mod = OverrideMagic()
        # 断言 len(mod) 等于 len(torch.jit.script(mod))
        self.assertEqual(len(mod), len(torch.jit.script(mod)))

        # 定义一个继承自 nn.Sequential 的类 OverrideMagicSeq
        class OverrideMagicSeq(nn.Sequential):
            @torch.jit.export
            # 重写魔术方法 __len__，返回固定值 10
            def __len__(self):
                return 10

        # 创建 OverrideMagicSeq 类的实例 mod
        mod = OverrideMagicSeq()
        # 断言 len(mod) 等于 len(torch.jit.script(mod))
        self.assertEqual(len(mod), len(torch.jit.script(mod)))
        # 断言 torch.jit.script(mod) 返回 True
        self.assertTrue(torch.jit.script(mod))

    # 定义一个测试方法，用于测试嵌套脚本模块中的 for 循环
    def test_script_module_for2(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 Sub
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化一个名为 weight 的参数，其值为形状为 (2,) 的随机张量
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            # 定义一个前向方法，接受参数 thing
            def forward(self, thing):
                # 返回 self.weight 与参数 thing 的和
                return self.weight + thing

        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化一个包含 10 个 Sub 实例的模块列表 mods
                self.mods = nn.ModuleList([Sub() for i in range(10)])

            @torch.jit.script_method
            # 定义一个前向方法，接受参数 v
            def forward(self, v):
                # 遍历 self.mods 中的每个子模块 m
                for m in self.mods:
                    # 对参数 v 执行 m 的前向操作，更新 v
                    v = m(v)
                return v

        # 禁用优化执行
        with torch.jit.optimized_execution(False):
            # 创建一个形状为 (2,) 的空张量 i
            i = torch.empty(2)
            # 创建类 M 的实例 m
            m = M()
            # 对 m 执行前向方法，传入参数 i，将结果赋给 o
            o = m(i)
            # 将 i 复制给 v
            v = i
            # 遍历 m.mods 中的每个子模块 sub，对 v 执行 sub 的前向操作
            for sub in m.mods:
                v = sub(v)
            # 断言 o 等于 v
            self.assertEqual(o, v)
            # 断言在迭代 m 时会抛出异常，异常消息为 "object is not iterable"
            with self.assertRaisesRegex(Exception, "object is not iterable"):
                print(list(m))
    # 定义一个测试函数，用于测试属性和脚本化
    def test_attr_qscheme_script(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 设置对象的属性 qscheme 为 torch.per_tensor_affine
                self.qscheme = torch.per_tensor_affine

            @torch.jit.script_method
            def forward(self):
                # 如果对象的 qscheme 属性为 torch.per_tensor_symmetric，则返回 3
                if self.qscheme == torch.per_tensor_symmetric:
                    return 3
                else:
                    # 否则返回 4
                    return 4

        # 创建类 Foo 的实例 f
        f = Foo()
        # 对类 Foo 进行脚本化
        scripted = torch.jit.script(f)
        # 断言调用 f() 和 scripted() 返回相同结果
        self.assertEqual(f(), scripted())

    # 定义一个测试函数，用于测试脚本化模块中的子模块常量
    def test_script_module_const_submodule_fail(self):
        # 定义继承自 torch.jit.ScriptModule 的子类 Sub
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化子模块的权重参数为随机张量
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                # 返回子模块的权重参数加上输入的 thing
                return self.weight + thing

        # 定义继承自 torch.jit.ScriptModule 的主类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建包含 10 个 Sub 类实例的列表 mods
                self.mods = [Sub() for _ in range(10)]

            @torch.jit.script_method
            def forward(self):
                # 对列表 self.mods 进行迭代，打印数字 1
                for _ in self.mods:
                    print(1)
                # 返回常数值 4
                return 4

        # 使用断言检查创建 M 类实例时是否会抛出 RuntimeError 异常，异常信息为 "has no attribute 'mods'"
        with self.assertRaisesRegex(RuntimeError, "has no attribute 'mods'"):
            M()

    # 定义一个继承自 torch.jit.ScriptModule 的派生状态模块类 DerivedStateModule
    class DerivedStateModule(torch.jit.ScriptModule):
        def __init__(self):
            super(TestScript.DerivedStateModule, self).__init__()
            # 初始化模块的参数 param 为全为 1 的张量
            self.param = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float))
            # 注册缓冲区 derived，其值为 param 取负后并分离的克隆
            self.register_buffer('derived', torch.neg(self.param).detach().clone())

            # 用于测试 pack 方法是否被调用的标志
            self.register_buffer('pack_called', torch.zeros(1, dtype=torch.long))
            # 用于测试 unpack 方法是否被调用的标志
            self.register_buffer('unpack_called', torch.zeros(1, dtype=torch.long))

        @torch.jit.script_method
        def _pack(self):
            # 设置 pack_called 标志为 1
            self.pack_called.set_(torch.ones(1, dtype=torch.long))
            # 更新 derived 缓冲区的值为随机数并分离

        @torch.jit.script_method
        def _unpack(self):
            # 设置 unpack_called 标志为 1
            self.unpack_called.set_(torch.ones(1, dtype=torch.long))
            # 更新 derived 缓冲区的值为 param 取负后并分离

        @torch.jit.script_method
        def forward(self, x):
            # 返回输入 x 加上 derived 缓冲区的值
            return x + self.derived
    # 定义测试方法，验证状态模块的打包与解包功能
    def test_pack_unpack_state(self):
        # 创建 DerivedStateModule 的实例
        sm = TestScript.DerivedStateModule()
        # 生成一个随机的 3x4 的张量 x
        x = torch.rand(3, 4)
        # 断言 sm(x) 的结果与 x + torch.neg(torch.ones(3, 4, dtype=torch.float)) 接近
        torch.testing.assert_close(sm(x), x + torch.neg(torch.ones(3, 4, dtype=torch.float)))

        # 测试保存路径
        # 断言 pack_called 属性为 False
        self.assertFalse(sm.pack_called.item())
        # 断言 unpack_called 属性为 False
        self.assertFalse(sm.unpack_called.item())
        # 使用打包功能获取导出并导入的拷贝 imported
        imported = self.getExportImportCopyWithPacking(sm)
        # 确保序列化之前已调用打包功能
        self.assertTrue(sm.pack_called.item())
        # 确保序列化之后已调用解包功能，以确保模块处于初始化状态
        self.assertTrue(sm.unpack_called.item())

        # 断言 derived 属性与 torch.neg(sm.param) 的结果接近
        torch.testing.assert_close(sm.derived, torch.neg(sm.param))

        # 测试加载路径
        # 断言导入的对象已调用解包功能
        self.assertTrue(imported.unpack_called.item())
        # 断言 imported(x) 的结果与 x + torch.neg(torch.ones(3, 4, dtype=torch.float)) 接近
        torch.testing.assert_close(imported(x), x + torch.neg(torch.ones(3, 4, dtype=torch.float)))

    @unittest.skipIf(not TEST_MKL, "PyTorch is built without MKL support")
    @unittest.skipIf(True, "Skipping while landing PR stack")
    def test_torch_functional(self):
        # 定义一个用于计算短时傅里叶变换（STFT）的函数，返回复数形式的结果张量
        def stft(input, n_fft):
            # type: (Tensor, int) -> Tensor
            return torch.stft(input, n_fft, return_complex=True)

        # 生成测试输入和参数
        inps = (torch.randn(10), 7)
        # 断言STFT函数和其脚本化版本在给定输入下的输出相等
        self.assertEqual(stft(*inps), torch.jit.script(stft)(*inps))

        # 定义一个用于计算逆短时傅里叶变换（ISTFT）的函数
        def istft(input, n_fft):
            # type: (Tensor, int) -> Tensor
            return torch.istft(input, n_fft)

        # 生成ISTFT函数的输入参数
        inps2 = (stft(*inps), inps[1])
        # 断言ISTFT函数和其脚本化版本在给定输入下的输出相等
        self.assertEqual(istft(*inps2), torch.jit.script(istft)(*inps2))

        # 定义一个用于进行LU分解的函数，返回LU分解后的张量
        def lu_unpack(x):
            # 使用torch.linalg.lu_factor函数计算LU分解及其置换向量
            A_LU, pivots = torch.linalg.lu_factor(x)
            # 返回通过LU分解和置换向量解包的结果
            return torch.lu_unpack(A_LU, pivots)

        # 针对不同形状的张量进行LU分解函数的脚本化检查
        for shape in ((3, 3), (5, 3, 3), (7, 3, 5, 5), (7, 5, 3, 3, 3)):
            a = torch.randn(*shape)
            self.checkScript(lu_unpack, (a,))

        # 定义一个计算两个集合之间距离的函数
        def cdist_fn():
            # 定义两个张量作为输入数据
            a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
            b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
            # 返回a和b之间的距离矩阵，使用矩阵乘法计算欧几里得距离
            return torch.cdist(a, b, compute_mode="use_mm_for_euclid_dist")

        # 检查cdist_fn函数的脚本化版本
        self.checkScript(cdist_fn, ())

        # 定义一个计算张量范数的函数，包括弗罗贝尼乌斯范数、核范数和不同阶数的范数
        def norm():
            c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
            # 返回不同种类的范数计算结果
            return torch.norm(c, p="fro"), torch.norm(c, p="nuc"), torch.norm(c), torch.norm(c, p=.5)

        # 检查norm函数的脚本化版本
        self.checkScript(norm, ())

        # 定义一个查找张量唯一值的函数
        def torch_unique(dim: Optional[int]):
            # 创建一个张量并找出其唯一值
            ten = torch.unique(torch.tensor([[1, 3], [2, 3]], dtype=torch.long))
            # 分别返回不同选项下的唯一值、计数、反向索引结果
            a = torch.unique(ten, dim=dim)
            b = torch.unique(ten, return_counts=True, dim=dim)
            c = torch.unique(ten, return_inverse=True, dim=dim)
            d = torch.unique(ten, return_counts=True, return_inverse=True, dim=dim)
            return a, b, c, d

        # 检查torch_unique函数的脚本化版本，包括未指定和指定维度的情况
        self.checkScript(torch_unique, (None,))
        self.checkScript(torch_unique, (0,))

        # 定义一个连续唯一值查找函数
        def torch_unique_consecutive(dim: Optional[int]):
            # 创建一个张量并找出其连续唯一值
            ten = torch.unique(torch.tensor([[1, 3], [3, 2], [3, 2], [2, 3]], dtype=torch.long))
            # 分别返回不同选项下的连续唯一值、计数、反向索引结果
            a = torch.unique_consecutive(ten, dim=dim)
            b = torch.unique_consecutive(ten, return_counts=True, dim=dim)
            c = torch.unique_consecutive(ten, return_inverse=True, dim=dim)
            d = torch.unique_consecutive(ten, return_counts=True, return_inverse=True, dim=dim)
            return a, b, c, d

        # 检查torch_unique_consecutive函数的脚本化版本，包括未指定和指定维度的情况
        self.checkScript(torch_unique_consecutive, (None,))
        self.checkScript(torch_unique_consecutive, (0,))
    # 定义测试函数 test_torch_functional_tensordot_int，用于测试整数 dims 的 tensordot 操作
    def test_torch_functional_tensordot_int(self):
        # 定义内部函数 tensordot_dims_int，接受两个 torch.Tensor 对象和一个整数 dims，返回 tensordot 的结果
        def tensordot_dims_int(a: torch.Tensor, b: torch.Tensor, dims: int):
            return torch.tensordot(a, b, dims=dims)

        # 创建两个 Tensor 对象 a 和 b，并reshape成不同形状
        a = torch.arange(120.).reshape(2, 3, 4, 5)
        b = torch.arange(840.).reshape(4, 5, 6, 7)
        # 设定 dims 为整数 2，使用 self.checkScript 检查脚本化后的函数结果
        dims = 2
        self.checkScript(tensordot_dims_int, (a, b, dims))

        # 遍历 dims 的两个特定情况：-1 和 5
        for dims in [-1, 5]:
            try:
                # 调用 tensordot_dims_int 函数，捕获 RuntimeError 异常
                tensordot_dims_int(a, b, dims)
            except RuntimeError as error:
                # 根据 dims 的值不同，检查异常信息是否符合预期
                if dims < 0:
                    self.assertEqual(str(error), "tensordot expects dims >= 0, but got dims=" + str(dims))
                if dims > min(a.dim(), b.dim()):
                    self.assertEqual(str(error), "tensordot expects dims < ndim_a or ndim_b, but got dims=" + str(dims))

    # 定义测试函数 test_torch_functional_tensordot_tensor，用于测试 Tensor dims 的 tensordot 操作
    def test_torch_functional_tensordot_tensor(self):
        # 定义内部函数 tensordot_dims_tensor，接受两个 torch.Tensor 对象和一个 Tensor dims，返回 tensordot 的结果
        def tensordot_dims_tensor(a: torch.Tensor, b: torch.Tensor, dims: torch.Tensor):
            return torch.tensordot(a, b, dims=dims)

        # 创建两个 Tensor 对象 a 和 b，并reshape成不同形状
        a = torch.arange(120.).reshape(2, 3, 4, 5)
        b = torch.arange(840.).reshape(4, 5, 6, 7)
        # 设定 dims 为包含单个整数 2 的 Tensor，使用 self.checkScript 检查脚本化后的函数结果
        dims = torch.tensor([2])
        self.checkScript(tensordot_dims_tensor, (a, b, dims))

        # 创建两个不同形状的 Tensor 对象 a 和 b
        a = torch.arange(60.).reshape(3, 4, 5)
        b = torch.arange(24.).reshape(4, 3, 2)
        # 设定 dims 为包含多个整数的 Tensor，使用 self.checkScript 检查脚本化后的函数结果
        dims = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        self.checkScript(tensordot_dims_tensor, (a, b, dims))

    # 定义测试函数 test_torch_functional_tensordot_list，用于测试列表 dims 的 tensordot 操作
    def test_torch_functional_tensordot_list(self):
        # 定义内部函数 tensordot_dims_list，接受两个 torch.Tensor 对象和一个列表 dims，返回 tensordot 的结果
        def tensordot_dims_list(a: torch.Tensor, b: torch.Tensor, dims: List[List[int]]):
            return torch.tensordot(a, b, dims=dims)

        # 创建两个不同形状的 Tensor 对象 a 和 b
        a = torch.arange(60.).reshape(3, 4, 5)
        b = torch.arange(24.).reshape(4, 3, 2)
        # 设定 dims 为包含多个列表的列表，使用 self.checkScript 检查脚本化后的函数结果
        dims = [[1, 0], [0, 1]]
        self.checkScript(tensordot_dims_list, (a, b, dims))

    # 定义测试函数 test_torch_functional_tensordot_tuple，用于测试元组 dims 的 tensordot 操作
    def test_torch_functional_tensordot_tuple(self):
        # 定义内部函数 tensordot_dims_tuple，接受两个 torch.Tensor 对象和一个元组 dims，返回 tensordot 的结果
        def tensordot_dims_tuple(a: torch.Tensor, b: torch.Tensor, dims: Tuple[List[int], List[int]]):
            return torch.tensordot(a, b, dims=dims)

        # 创建两个不同形状的 Tensor 对象 a 和 b
        a = torch.arange(60.).reshape(3, 4, 5)
        b = torch.arange(24.).reshape(4, 3, 2)
        # 设定 dims 为包含两个列表的元组，使用 self.checkScript 检查脚本化后的函数结果
        dims = ([1, 0], [0, 1])
        self.checkScript(tensordot_dims_tuple, (a, b, dims))

    # 定义测试函数 test_missing_getstate，用于测试缺失 __getstate__ 方法时的异常情况
    def test_missing_getstate(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo，其中未定义 __getstate__ 方法
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = 1

            def forward(self, x):
                return x * self.x

            @torch.jit.export
            # 定义 __setstate__ 方法，接受 state，并设置对象的两个属性
            def __setstate__(self, state):
                self.x = state[0]
                self.training = state[1]

        # 使用 torch.jit.script 尝试对 Foo 类进行脚本化，期望捕获 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "getstate"):
            scripted = torch.jit.script(Foo())
    def test_pack_unpack_nested(self):
        class SubSubMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 注册一个形状为 (3, 4) 的缓冲区，填充为全 3 的张量
                self.register_buffer('buf', torch.ones(3, 4) * 3)

            @torch.jit.script_method
            def _pack(self):
                # 将缓冲区设为全 0 的张量
                self.buf.set_(torch.zeros(1))

            @torch.jit.script_method
            def _unpack(self):
                # 将缓冲区设为全 3 的张量
                self.buf.set_(torch.ones(3, 4) * 3)

            @torch.jit.script_method
            def forward(self, x):
                # 返回输入 x 和缓冲区 buf 相加的结果
                return x + self.buf

        class SubMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 注册一个形状为 (3, 4) 的缓冲区，填充为全 2 的张量
                self.register_buffer('buf', torch.ones(3, 4) * 2)
                # 创建 SubSubMod 实例
                self.ssm = SubSubMod()

            @torch.jit.script_method
            def _pack(self):
                # 将缓冲区设为全 0 的张量
                self.buf.set_(torch.zeros(1))

            @torch.jit.script_method
            def _unpack(self):
                # 将缓冲区设为全 2 的张量
                self.buf.set_(torch.ones(3, 4) * 2)

            @torch.jit.script_method
            def forward(self, x):
                # 返回输入 x 加上缓冲区 buf 和 SubSubMod 实例的输出的结果
                return self.ssm(x + self.buf)

        class Mod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建 SubMod 实例
                self.submod = SubMod()
                # 注册一个形状为 (3, 4) 的缓冲区，填充为全 1 的张量
                self.register_buffer('buf', torch.ones(3, 4) * 1)

            @torch.jit.script_method
            def _pack(self):
                # 将缓冲区设为全 0 的张量
                self.buf.set_(torch.zeros(1))

            @torch.jit.script_method
            def _unpack(self):
                # 将缓冲区设为全 1 的张量
                self.buf.set_(torch.ones(3, 4))

            @torch.jit.script_method
            def forward(self, x):
                # 返回输入 x 加上 SubMod 实例的输出的结果
                return self.submod(x + self.buf)

        # 创建 Mod 实例
        m = Mod()
        # 断言 Mod 实例对输入全 0 的张量输出的结果与形状为 (3, 4) 的全 6 张量相近
        torch.testing.assert_close(m(torch.zeros(3, 4)), torch.ones(3, 4) * 6)
        # 对 Mod 实例应用 _pack 方法
        m.apply(lambda s: s._pack())
        # 断言 Mod 实例对输入全 0 的张量输出的结果与全 0 的张量相近
        torch.testing.assert_close(m(torch.zeros(3, 4)), torch.zeros(3, 4))
        # 对 Mod 实例应用 _unpack 方法
        m.apply(lambda s: s._unpack())
        # 断言 Mod 实例对输入全 0 的张量输出的结果与形状为 (3, 4) 的全 6 张量相近
        torch.testing.assert_close(m(torch.zeros(3, 4)), torch.ones(3, 4) * 6)
    def test_torch_any(self):
        # 定义一个函数 fn，用于计算输入张量 x 是否存在任何非零元素
        def fn(x):
            return torch.any(x)

        # 定义一个函数 fn1，用于计算输入张量 x 沿指定维度 dim 是否存在任何非零元素
        def fn1(x, dim: int):
            return torch.any(x, dim)

        # 对函数 fn 进行脚本化检查，验证其在给定参数下的行为
        self.checkScript(fn, (torch.randn(3, 4), ))
        self.checkScript(fn, (torch.empty(3), ))
        self.checkScript(fn, (torch.empty(1), ))
        self.checkScript(fn, (torch.ones(3, 4),))
        self.checkScript(fn, (torch.zeros(5, 7, 1),))
        # 对函数 fn1 进行脚本化检查，验证其在给定参数下的行为
        self.checkScript(fn1, (torch.empty(3, 4), -2))
        self.checkScript(fn1, (torch.randn(3, 8), 1))
        self.checkScript(fn1, (torch.zeros(3, 6, 9), -3))
        self.checkScript(fn1, (torch.empty(5), 0))

    def test_any(self):
        # 定义一个函数 fn，用于判断整数列表 x 中是否存在任何非零元素
        def fn(x: List[int]):
            return any(x)

        # 定义一个函数 fn1，用于判断浮点数列表 x 中是否存在任何非零元素
        def fn1(x: List[float]):
            return any(x)

        # 定义一个函数 fn2，用于判断布尔值列表 x 中是否存在任何非零元素
        def fn2(x: List[bool]):
            return any(x)

        # 定义一个函数 fn3，用于判断字符串列表 x 中是否存在任何非空字符串
        def fn3(x: List[str]):
            return any(x)

        # 对函数 fn 进行脚本化检查，验证其在给定参数下的行为
        self.checkScript(fn, ([0, 0, 0, 0], ))
        self.checkScript(fn, ([0, 3, 0], ))
        self.checkScript(fn, ([], ))
        # 对函数 fn1 进行脚本化检查，验证其在给定参数下的行为
        self.checkScript(fn1, ([1.0, 2.0, 3.0], ))
        self.checkScript(fn1, ([0.0, 0.0, 0.0], ))
        self.checkScript(fn1, ([0, 0, 0], ))
        self.checkScript(fn1, ([], ))
        # 对函数 fn2 进行脚本化检查，验证其在给定参数下的行为
        self.checkScript(fn2, ([True, False, False], ))
        self.checkScript(fn2, ([False, False, False], ))
        self.checkScript(fn2, ([True, True, True, True], ))
        self.checkScript(fn2, ([], ))
        # 对函数 fn3 进行脚本化检查，验证其在给定参数下的行为
        self.checkScript(fn3, (["", "", ""], ))
        self.checkScript(fn3, (["", "", "", "-1"], ))
        self.checkScript(fn3, ([], ))

    def test_script_module_not_tuple(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M，含有常量 mods，但其类型为整数
        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super().__init__()
                self.mods = 1

            @torch.jit.script_method
            def forward(self, v):
                # 迭代 self.mods，但由于其为整数类型，无法迭代，故抛出异常
                for m in self.mods:
                    print(m)
                return v
        # 验证在初始化 M 类时是否会抛出特定的运行时错误信息
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):
            M()

    def test_attr_module_constants(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M2，含有 mods 属性，其类型为 mod_list
        class M2(torch.jit.ScriptModule):
            def __init__(self, mod_list):
                super().__init__()
                self.mods = mod_list

            @torch.jit.script_method
            def forward(self, x):
                return self.mods.forward(x)

        # 禁用优化执行模式，创建 M2 类的实例 m，并验证其在导出和导入过程中的行为
        with torch.jit.optimized_execution(False):
            m = M2(nn.Sequential(nn.ReLU()))
            self.assertExportImportModule(m, (torch.randn(2, 2),))
    def test_script_sequential_for(self):
        # 定义一个继承自 torch.jit.ScriptModule 的子类 Sub
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个形状为 (2,) 的随机张量参数
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                # 返回权重张量和输入张量的和
                return self.weight + thing

        # 定义一个继承自 torch.jit.ScriptModule 的主类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个包含三个 Sub 实例的序列模块
                self.mods = nn.Sequential(Sub(), Sub(), Sub())

            @torch.jit.script_method
            def forward(self, v):
                # 依次对 self.mods 中的每个子模块进行前向传播
                for m in self.mods:
                    v = m(v)
                return v

            @torch.jit.script_method
            def forward2(self, v):
                # 调用 self.mods 对象的前向传播方法
                return self.mods(v)

        # 关闭优化执行
        with torch.jit.optimized_execution(False):
            # 创建一个形状为 (2,) 的空张量 i
            i = torch.empty(2)
            # 创建 M 类的实例 m
            m = M()
            # 对输入 i 进行前向传播，得到输出 o
            o = m(i)
            # 复制输入张量 i 给 v
            v = i
            # 对 self.mods 中每个子模块进行遍历并进行前向传播
            for sub in m.mods._modules.values():
                v = sub(v)
            # 断言 o 与 v 相等
            self.assertEqual(o, v)

            # 对象 m 调用 forward2 方法进行前向传播，得到输出 o2
            o2 = m.forward2(i)
            # 断言 o2 与 v 相等
            self.assertEqual(o2, v)

    def test_script_sequential_sliced_iteration(self):
        # 定义一个继承自 nn.Module 的 seq_mod 类
        class seq_mod(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含三个 nn.ReLU() 实例的列表，并将其转换为 nn.Sequential 对象
                self.layers = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
                self.layers = nn.Sequential(*self.layers)

            def forward(self, input):
                # 对 self.layers 中的第一个元素进行前向传播
                x = self.layers[0].forward(input)
                # 对 self.layers 中第 1 至 2 个元素进行遍历并进行前向传播
                for layer in self.layers[1:3]:
                    x = layer.forward(x)
                # 对 self.layers 中第 2 至末尾的元素进行遍历并进行前向传播
                for layer in self.layers[2:]:
                    x = layer.forward(x)
                return x

        # 创建 seq_mod 类的实例 seq
        seq = seq_mod()
        # 调用 checkModule 方法，检查 seq 模块的输出是否符合预期
        self.checkModule(seq, [torch.tensor([-2, 1, -1, 2])])

    def test_script_sequential_orderdict(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个包含有序字典的序列模块，包括一个卷积层和一个 ReLU 激活层
                self.mods = nn.Sequential(OrderedDict([
                    ("conv", nn.Conv2d(1, 20, 5)),
                    ("relu", nn.ReLU())
                ]))

            @torch.jit.script_method
            def forward(self, input):
                # 对输入 input 进行前向传播
                return self.mods(input)

        # 创建 M 类的实例 m
        m = M()
        # 断言 mods.conv.weight 是否在模型的状态字典中的键中
        self.assertTrue('mods.conv.weight' in m.state_dict().keys())
    @_tmp_donotuse_dont_inline_everything
    def test_script_sequential_in_mod_list(self):
        # 定义一个名为 Sub 的 Torch 脚本模块，包含一个参数化的权重
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            # Torch 脚本方法，实现模块的前向传播
            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        # 定义一个名为 M 的 Torch 脚本模块
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个模块列表 mods，其中包含一个 Sub 实例和一个嵌套的 Sequential 实例
                self.mods = nn.ModuleList([Sub(), nn.Sequential(Sub(), nn.Sequential(Sub(), Sub()), Sub())])

            # Torch 脚本方法，实现模块的前向传播
            @torch.jit.script_method
            def forward(self, v):
                # 遍历模块列表 mods 中的每个模块，并依次应用到输入 v 上
                for mod in self.mods:
                    v = mod(v)
                return v

        # 创建 M 类的实例 m
        m = M()
        # 将模型的计算图转换为字符串形式
        graph = str(m.graph)
        # 断言模型计算图中的 prim::CallMethod 调用次数为 2
        self.assertTrue(graph.count("prim::CallMethod") == 2)
        # 断言模型计算图中不包含 "python" 字符串
        self.assertTrue("python" not in graph)



    @_tmp_donotuse_dont_inline_everything
    def test_script_nested_mod_list(self):
        # 定义一个名为 Sub 的 Torch 脚本模块，包含一个参数化的权重
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            # Torch 脚本方法，实现模块的前向传播
            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        # 定义一个名为 M 的 Torch 脚本模块
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个模块列表 mods，其中包含一个嵌套的 ModuleList，一个 Sequential 实例，和另一个嵌套的 ModuleList
                self.mods = nn.ModuleList([nn.ModuleList([Sub()]), nn.Sequential(Sub()), nn.ModuleList([Sub(), Sub()])])

            # Torch 脚本方法，实现模块的前向传播
            @torch.jit.script_method
            def forward(self, v):
                # 遍历模块列表 mods 中的每个模块，再次遍历内部的模块，并依次应用到输入 v 上
                for mod in self.mods:
                    for m in mod:
                        v = m(v)
                return v

        # 创建 M 类的实例 m
        m = M()
        # 将模型的计算图转换为字符串形式
        graph = str(m.graph)
        # 断言模型计算图中的 prim::CallMethod 调用次数为 4
        self.assertTrue(graph.count("prim::CallMethod") == 4)
        # 断言模型计算图中不包含 "python" 字符串
        self.assertTrue("python" not in graph)
    def test_constant_as_attr(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            # 定义常量属性 __constants__，包含维度 'dim'
            __constants__ = ['dim']

            # 初始化方法
            def __init__(self):
                super().__init__()
                # 设置实例属性 dim 为 1
                self.dim = 1

            # 定义前向传播方法，使用 torch.jit.script_method 装饰器
            @torch.jit.script_method
            def forward(self, v):
                # 返回在指定维度 dim 上连接三个 v 张量的结果
                return torch.cat([v, v, v], dim=self.dim)
        
        # 创建一个形状为 (1, 1) 的零张量 v
        v = torch.zeros(1, 1)
        
        # 关闭优化执行
        with torch.jit.optimized_execution(False):
            # 断言 torch.cat 在维度 1 上连接三个 v 张量的结果等于 M 类实例的前向传播结果
            self.assertEqual(torch.cat([v, v, v], dim=1), M()(v))

    class StarTestSumStarred(torch.nn.Module):
        # 定义继承自 torch.nn.Module 的类 StarTestSumStarred
        def __init__(self):
            super(TestScript.StarTestSumStarred, self).__init__()

        # 定义前向传播方法，接收可变数量的输入参数
        def forward(self, *inputs):
            # 初始化输出为第一个输入
            output = inputs[0]
            # 循环遍历其余输入，累加到输出中
            for i in range(1, len(inputs)):
                output += inputs[i]
            # 返回累加后的输出
            return output

    class StarTestReturnThree(torch.nn.Module):
        # 定义继承自 torch.nn.Module 的类 StarTestReturnThree
        def __init__(self):
            super(TestScript.StarTestReturnThree, self).__init__()

        # 定义前向传播方法，接收一个参数 rep
        def forward(self, rep):
            # 返回三个相同的 rep 参数
            return rep, rep, rep

    def test_script_star_expr(self):

        # 定义一个继承自 torch.jit.ScriptModule 的类 M2
        class M2(torch.jit.ScriptModule):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 通过 torch.jit.trace 方法追踪 StarTestSumStarred 类的实例
                self.m = torch.jit.trace(TestScript.StarTestSumStarred(),
                                         (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)))
                # 通过 torch.jit.trace 方法追踪 StarTestReturnThree 类的实例
                self.g = torch.jit.trace(TestScript.StarTestReturnThree(), torch.ones(4, 3))

            # 定义前向传播方法，使用 torch.jit.script_method 装饰器
            @torch.jit.script_method
            def forward(self, rep):
                # 调用 self.g 方法获取返回的元组 tup
                tup = self.g(rep)
                # 调用 self.m 方法，传入 tup 元组解包后的参数，并返回结果
                return self.m(*tup)

        # 创建 M2 类的实例 m
        m = M2()
        # 断言调用 m 实例传入零张量的结果等于 3 倍的零张量
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_star_expr_string(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M2
        class M2(torch.jit.ScriptModule):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 通过 torch.jit.trace 方法追踪 StarTestSumStarred 类的实例
                self.m = torch.jit.trace(TestScript.StarTestSumStarred(),
                                         (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)))
                # 通过 torch.jit.trace 方法追踪 StarTestReturnThree 类的实例
                self.g = torch.jit.trace(TestScript.StarTestReturnThree(), torch.ones(4, 3))

                # 定义前向传播方法字符串形式
                self.define('''
            def forward(self, rep):
                tup = self.g(rep)
                return self.m(*tup)
                ''')

        # 创建 M2 类的实例 m
        m = M2()
        # 断言调用 m 实例传入零张量的结果等于 3 倍的零张量
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    class StarTestSumAndReturnThree(torch.nn.Module):
        # 定义继承自 torch.nn.Module 的类 StarTestSumAndReturnThree
        def __init__(self):
            super(TestScript.StarTestSumAndReturnThree, self).__init__()

        # 定义前向传播方法，接收可变数量的输入参数
        def forward(self, *inputs):
            # 初始化输出为第一个输入
            output = inputs[0]
            # 循环遍历其余输入，累加到输出中
            for i in range(1, len(inputs)):
                output += inputs[i]
            # 返回三个相同的累加后的输出
            return output, output, output
    def test_script_star_assign(self):
        # 定义一个继承自torch.jit.ScriptModule的类M2
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 使用torch.jit.trace对TestScript.StarTestSumAndReturnThree进行跟踪，传入参数为torch.ones(4, 3)
                self.g = torch.jit.trace(TestScript.StarTestSumAndReturnThree(), torch.ones(4, 3))
                # 定义forward方法，接受参数rep，从self.g(rep)中解包出head作为第一个元素，其余的元素作为tail
                self.define('''
            def forward(self, rep):
                head, *tail = self.g(rep)
                return head
                ''')

        m = M2()
        # 断言m(torch.zeros(4, 3))的结果与3乘以torch.zeros(4, 3)相等
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_module_star_assign2(self):
        # 定义一个继承自torch.jit.ScriptModule的类M2
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 使用torch.jit.trace对TestScript.StarTestSumAndReturnThree进行跟踪，
                # 传入参数为(torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3))，强制使用outplace模式
                self.g = torch.jit.trace(
                    TestScript.StarTestSumAndReturnThree(),
                    (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)),
                    _force_outplace=True)
                # 定义forward方法，接受参数rep，解包出self.g(rep, rep, rep)的结果中除了最后一个元素外的所有元素为head，最后一个元素为tail
                self.define('''
            def forward(self, rep):
                *head, tail = self.g(rep, rep, rep)
                return tail
                ''')

        m = M2()
        # 断言m(torch.ones(4, 3))的结果与3乘以torch.ones(4, 3)相等
        self.assertEqual(m(torch.ones(4, 3)), 3 * torch.ones(4, 3))

    def test_script_module_star_assign2_inplace(self):
        # 定义一个继承自torch.jit.ScriptModule的类M2
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 使用torch.jit.trace对TestScript.StarTestSumAndReturnThree进行跟踪，
                # 传入参数为(torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3))，强制使用inplace模式
                self.g = torch.jit.trace(
                    TestScript.StarTestSumAndReturnThree(),
                    (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)),
                    _force_outplace=False)
                # 定义forward方法，接受参数rep，解包出self.g(rep, rep, rep)的结果中除了最后一个元素外的所有元素为head，最后一个元素为tail
                self.define('''
            def forward(self, rep):
                *head, tail = self.g(rep, rep, rep)
                return tail
                ''')

        m = M2()
        # 因为forward()在将输入的rep传递给StarTestSumAndReturnThree()之前创建了三个别名，所以inplace行为将不同于上面的outplace。
        self.assertEqual(m(torch.ones(4, 3)), 4 * torch.ones(4, 3))

    def test_script_module_star_assign_fail_pythonop(self):

        # 使用self.assertRaisesRegex断言在运行时捕获到的RuntimeError中包含"cannot be used as a tuple"
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            # 定义一个继承自torch.jit.ScriptModule的类M2
            class M2(torch.jit.ScriptModule):
                def __init__(self):
                    super().__init__()
                    # 定义一个被@torch.jit.ignore修饰的函数myfunc，返回torch.zeros(1, 2, 3)和torch.zeros(1, 2, 3)
                    @torch.jit.ignore
                    def myfunc():
                        return torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)

                    # 定义forward方法，接受参数rep，将myfunc()的结果解包为a和*作为b，返回a
                    self.define('''
                def forward(self, rep):
                    a, *b = myfunc()
                    return a
                    ''')

            m = M2()
            m(torch.zeros(4, 3))
    # 定义一个测试用例，验证在使用脚本模块时，星号赋值操作无法作为元组使用时抛出 RuntimeError 异常
    def test_script_module_star_assign_fail_builtin(self):
        # 使用断言检查是否抛出指定异常信息
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            # 定义一个继承自 torch.jit.ScriptModule 的脚本模块类 M2
            class M2(torch.jit.ScriptModule):
                def __init__(self):
                    super().__init__()

                    # 定义 forward 方法
                    self.define('''
                def forward(self, rep):
                    # 对输入张量 rep 中的元素取负值，并尝试进行星号赋值操作
                    a, *b = torch.neg(rep)
                    # 返回取负值后的结果张量 a
                    return a
                    ''')

            # 创建 M2 类的实例
            m = M2()
            # 调用该实例的 forward 方法，并传入一个形状为 (4, 3) 的零张量
            m(torch.zeros(4, 3))

    # 定义测试函数 test_script_pack_padded_sequence
    def test_script_pack_padded_sequence(self):
        # 导入 pack_padded_sequence 和 pad_packed_sequence 函数
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        # 定义内部函数 pack_padded_pad_packed_script，接收 x 和 seq_lens 两个参数
        def pack_padded_pad_packed_script(x, seq_lens):
            # 对输入数据 x 进行 pack_padded_sequence 操作
            x = pack_padded_sequence(x, seq_lens)
            # 对 pack_padded_sequence 后的结果进行 pad_packed_sequence 操作
            x, lengths = pad_packed_sequence(x)
            # 返回处理后的结果张量 x 和长度列表 lengths
            return x, lengths

        # 设置 T（时间步长）、B（批次大小）、C（通道数）的值分别为 3、5、7
        T, B, C = 3, 5, 7
        # 创建一个形状为 (T, B, C) 的全 1 张量 x
        x = torch.ones((T, B, C))
        # 创建一个包含每个序列长度的张量 seq_lens
        seq_lens = torch.tensor([3, 3, 2, 2, 1])
        # 如果某个序列长度小于 T，则将该序列之后的部分置为 0
        for b in range(B):
            if seq_lens[b] < T:
                x[seq_lens[b]:, b, :] = 0

        # 调用 pack_padded_pad_packed_script 函数处理输入张量 x 和序列长度 seq_lens
        eager_seq, eager_lengths = pack_padded_pad_packed_script(x, seq_lens)

        # 使用 torch.jit.script 将 pack_padded_pad_packed_script 函数转换为脚本形式
        with torch._jit_internal._disable_emit_hooks():
            scripted_pack_padded_seq = torch.jit.script(pack_padded_pad_packed_script)

        # 使用脚本化的 pack_padded_pad_packed_script 处理输入张量 x 和序列长度 seq_lens
        script_seq, script_lengths = scripted_pack_padded_seq(x, seq_lens)

        # 使用断言检查两种方式处理得到的结果是否一致
        self.assertEqual(eager_seq, script_seq)
        self.assertEqual(eager_lengths, script_lengths)

        # 定义一个 ExperimentalLSTM 类，继承自 torch.nn.Module
        class ExperimentalLSTM(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()

            def forward(self, input):
                # type: (Tensor)
                # 对输入张量 input 执行 pack_padded_sequence 操作，长度为 [1, 2]，不要求排序
                packed = pack_padded_sequence(
                    input=input, lengths=torch.tensor([1, 2]), enforce_sorted=False
                )
                # 对 packed 序列进行 pad_packed_sequence 操作，总长度为 2
                output, lengths = pad_packed_sequence(
                    sequence=packed, total_length=2
                )
                # 返回输出张量 output 的第一个元素
                return output[0]

        # 创建 ExperimentalLSTM 类的实例 lstm
        lstm = ExperimentalLSTM(input_dim=2, hidden_dim=2)

        # 禁用 torch.jit 内部的钩子函数，以确保正确性
        with torch._jit_internal._disable_emit_hooks():
            # 使用自定义的函数 self.checkModule 检查 lstm 模块的行为
            self.checkModule(lstm, [torch.ones(2, 2)])
    def test_script_pad_sequence_pack_sequence(self):
        from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence

        def pad_sequence_func(tensor_list, batch_first=False, padding_value=0.0):
            # 定义一个函数，用于对列表中的张量进行填充操作，返回一个填充后的张量
            return pad_sequence(tensor_list, batch_first, padding_value)

        def pack_sequence_func(tensor_list, enforce_sorted=True):
            # 定义一个函数，用于对列表中的张量进行打包操作，返回打包后的张量
            return pad_packed_sequence(pack_sequence(tensor_list, enforce_sorted))[0]

        ones3 = torch.ones(3, 5)
        ones4 = torch.ones(4, 5)
        ones5 = torch.ones(5, 5)
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5])
        tensor3 = torch.tensor([6])
        with torch._jit_internal._disable_emit_hooks():
            # 检查 pad_sequence_func 函数的脚本化
            self.checkScript(pad_sequence_func,
                             ([ones3, ones4, ones5],))
            # 检查 pad_sequence_func 函数的脚本化，设置 batch_first 参数为 True
            self.checkScript(pad_sequence_func,
                             ([ones3, ones4, ones5], True))
            # 检查 pad_sequence_func 函数的脚本化，设置 batch_first 参数为 True，padding_value 参数为 2.5
            self.checkScript(pad_sequence_func,
                             ([ones3, ones4, ones5], True, 2.5))
            # 检查 pack_sequence_func 函数的脚本化
            self.checkScript(pack_sequence_func,
                             ([tensor1, tensor2, tensor3],))
            # 检查 pack_sequence_func 函数的脚本化，设置 enforce_sorted 参数为 False
            self.checkScript(pack_sequence_func,
                             ([tensor1, tensor2, tensor3], False))

    def test_script_get_tracing_state(self):
        def test_if_tracing(x):
            # 检查是否处于跟踪状态，根据状态返回不同的计算结果
            if torch._C._get_tracing_state():
                return x + 1
            else:
                return x - 1

        inp = torch.randn(3, 3)
        self.checkScript(test_if_tracing, (inp,))

    def test_script_is_tracing(self):
        def test_is_tracing(x):
            # 检查当前是否正在进行模型脚本化，根据结果返回不同的计算结果
            if torch.jit.is_tracing():
                return x + 1
            else:
                return x - 1

        inp = torch.randn(3, 3)
        self.checkScript(test_is_tracing, (inp,))

    def test_is_scripting(self):
        def foo():
            # 返回当前是否正在进行模型脚本化的状态
            return torch.jit.is_scripting()

        self.assertFalse(foo())
        # 将函数 foo 进行脚本化，并验证其返回值
        scripted = torch.jit.script(foo)
        self.assertTrue(scripted())

    def test_comment_ignore_indent(self):
        class Model(torch.nn.Module):
            def __init__(self):
                # 初始化父类的构造函数
                super().__init__()

            def forward(self):
                # 模型的前向传播函数，返回固定值 5
                return 5

        # 创建 Model 类的实例，并验证其能够正确编译
        self.checkModule(Model(), ())
    def test_script_outputs(self):
        # 使用 assertRaisesRegex 检查 RuntimeError，确保错误消息包含 "cannot be used as a tuple"
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            # 将函数 foo 转换为 Torch 脚本
            @torch.jit.script
            def foo(a):
                # 尝试将 a + a 的结果解包到变量 c 和 d 中
                c, d = a + a
                return c + d

        # 定义一个返回元组 (1, 2, 3) 的 Torch 脚本函数
        @torch.jit.script
        def return3():
            return 1, 2, 3

        # 使用 assertRaisesRegex 检查 RuntimeError，确保错误消息包含 "too many values to unpack"
        with self.assertRaisesRegex(RuntimeError, "too many values to unpack"):
            # 将函数 bind2 转换为 Torch 脚本
            @torch.jit.script
            def bind2():
                # 尝试将 return3() 返回的值解包到变量 a 和 b 中
                a, b = return3()
                print(a)
                print(b)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_script_get_device_cuda(self):
        # 定义一个返回输入张量在 CUDA 设备上的索引的 Torch 脚本函数
        @torch.jit.script
        def foo(a):
            return a.get_device()

        # 创建一个在 CUDA 设备上的随机张量
        v = torch.randn(1, device='cuda')
        # 使用断言验证 foo 函数返回的设备索引是否为 0
        self.assertEqual(foo(v), 0)

    def test_script_chunk(self):
        # 定义一个 Torch 脚本函数，使用 torch.chunk 在指定维度上将输入张量分块
        @torch.jit.script
        def foo(a):
            b, c = torch.chunk(a, dim=0, chunks=2)
            return b
        # 创建一个形状为 (10, 3) 的随机张量
        v = torch.rand(10, 3)
        # 使用断言验证 foo 函数和 torch.chunk 函数的结果是否相等
        self.assertEqual(torch.chunk(v, dim=0, chunks=2)[0], foo(v))

    def test_script_copy(self):
        # 定义一个继承自 torch.nn.Module 的类 M，用于 Torch 脚本
        class M(torch.nn.Module):
            __annotations__ = {
                "val": Optional[torch.Tensor]
            }

            def __init__(self):
                super().__init__()
                self.val = None

            def some_method(self):
                return 3

            def forward(self, x):
                # type: (Tensor) -> Tensor
                # 在 forward 方法中更新 self.val，将输入张量 x 加上某个值
                self.val = x + self.some_method()
                return x

        # 使用 torch.jit.script 将 M 类转换为 Torch 脚本
        m = torch.jit.script(M())
        # 测试 copy 操作是否正常运行
        copy.copy(m)
        copy.deepcopy(m)

    def test_script_forward_method_replacement(self):
        # 定义一个低级别的 torch.nn.Module，其 forward 方法执行一个简单的操作
        class LowLevelModule(torch.nn.Module):
            def forward(self, input: torch.Tensor):
                # Generic forward dispatch
                return self.forward_pytorch(input) * 2

        # 定义一个 TestModule 类，继承自 LowLevelModule
        class TestModule(LowLevelModule):
            def __init__(self):
                super().__init__()
                # 替换 TestModule 的 forward 方法为 LowLevelModule 的 forward 方法
                self.forward = types.MethodType(LowLevelModule.forward, self)

            def forward_pytorch(self, input: torch.Tensor):
                return torch.tensor(123)

            def forward(self, input: torch.Tensor):
                # 不应使用此 forward 方法，因此引发 AssertionError
                raise AssertionError("This method should not be used")
                return self.forward_pytorch(input)

        # 创建 TestModule 实例
        m = TestModule()
        # 使用断言验证 m(torch.tensor(1)) 的输出是否为 torch.tensor(246)
        self.assertEqual(m(torch.tensor(1)), torch.tensor(246))

        # 使用 torch.jit.script 将 TestModule 类转换为 Torch 脚本
        m_scripted = torch.jit.script(m)
        # 使用断言验证 m_scripted(torch.tensor(1)) 的输出是否为 torch.tensor(246)
        self.assertEqual(m_scripted(torch.tensor(1)), torch.tensor(246))
    # 定义一个测试函数，用于验证在调用时传入非张量类型时的行为
    def test_python_call_non_tensor(self):
        # 定义内部函数 foo，接受一个张量、一个整数和一个元组作为参数，并返回一个整数和一个张量的元组
        def foo(a, b, c):
            # type 注释说明了 foo 函数的参数类型和返回类型
            # a 是一个张量（Tensor），b 是一个整数，c 是一个包含张量和整数的元组
            # 返回一个整数和一个张量
            d, e = c  # 解包元组 c 中的值
            return b + e, a + d  # 返回计算结果

        # 使用 torch.jit.script 装饰器将 bar 函数编译为 TorchScript
        @torch.jit.script
        def bar():
            x = torch.ones(3, 4)  # 创建一个形状为 (3, 4) 的全一张量
            a, b = foo(x, 3, (x, 3))  # 调用 foo 函数
            return a, b  # 返回 foo 函数的结果

        # 断言 bar 函数的返回值与期望值相等
        self.assertEqual((6, torch.ones(3, 4) + 1), bar())

    # 测试当调用一个返回类型不是张量的函数时，是否会引发错误
    def test_python_call_non_tensor_wrong(self):
        # 使用 self.assertRaisesRegex 检查是否会抛出特定类型的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"but instead got value of type tuple"):
            # 使用 torch.jit.ignore 装饰器标记 foo 函数，使其在 TorchScript 编译时被忽略
            @torch.jit.ignore
            def foo():
                # type 注释指明 foo 函数的返回类型是张量
                return ((3, 4),)  # noqa: T484

            # 使用 torch.jit.script 装饰器将 bar 函数编译为 TorchScript
            @torch.jit.script
            def bar():
                return foo()  # 调用 foo 函数

            bar()  # 调用 bar 函数

    # 测试当条件分支中出现不同类型时是否会引发错误
    def test_if_different_type(self):
        # 使用 self.assertRaisesRegex 检查是否会抛出特定类型的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "c0 is set to type "
                                    "int in the true branch and type "
                                    "float in the false branch"):
            # 使用 torch.jit.script 装饰器将 diff_type_used 函数编译为 TorchScript
            @torch.jit.script
            def diff_type_used():
                if 1 == 2:
                    c0 = 1  # 在真分支中 c0 被设置为整数类型
                else:
                    c0 = 1.0  # 在假分支中 c0 被设置为浮点数类型
                return c0  # 返回 c0

        # 使用 self.assertRaisesRegex 检查是否会抛出特定类型的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Variable 'c0' previously had type float"):
            # 使用 torch.jit.script 装饰器将 diff_existing_type 函数编译为 TorchScript
            @torch.jit.script
            def diff_existing_type(x):
                c0 = 1.0  # 初始化 c0 为浮点数类型
                if 1 == 2:
                    c0 = 1  # 在真分支中将 c0 重新赋值为整数类型
                    print(x)
                return x  # 返回 x

        # 使用 torch.jit.script 装饰器将 diff_type_unused 函数编译为 TorchScript
        @torch.jit.script
        def diff_type_unused():
            if 1 == 1:
                c0 = 1  # 在真分支中 c0 被设置为整数类型
                print(c0)
            else:
                c0 = 1.0  # 在假分支中 c0 被设置为浮点数类型
                print(c0)
            return 1  # 返回整数 1

    # 测试在条件分支中未定义变量时是否会引发错误
    def test_if_not_defined_error(self):
        # 使用 self.assertRaisesRegex 检查是否会抛出特定类型的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "c0 is not defined in the false branch"):
            # 使用 torch.jit.script 装饰器将 test 函数编译为 TorchScript
            @torch.jit.script
            def test():
                if 1 == 1:
                    c0 = 1  # 在真分支中定义并赋值变量 c0
                return c0  # 返回变量 c0

        # 使用 self.assertRaisesRegex 检查是否会抛出特定类型的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "c0 is not defined in the true branch"):
            # 使用 torch.jit.script 装饰器将 test2 函数编译为 TorchScript
            @torch.jit.script
            def test2():
                if 1 == 1:
                    pass
                else:
                    c0 = 1  # 在假分支中定义并赋值变量 c0
                return c0  # 返回变量 c0

    # 测试在 TorchScript 中使用列表拼接时不会抛出错误
    def test_if_list_cat(self):
        # 使用 torch.jit.script 装饰器将 test_list 函数编译为 TorchScript
        @torch.jit.script
        def test_list(x):
            if bool(x.sum() < 1):
                c = [x, x]  # 如果条件成立，创建一个包含两个张量 x 的列表
            else:
                c = [x, x, x]  # 如果条件不成立，创建一个包含三个张量 x 的列表
            return torch.cat(c)  # 拼接列表 c 中的张量并返回结果

        b = torch.zeros(2, 4)  # 创建一个形状为 (2, 4) 的全零张量 b
        _propagate_shapes(test_list.graph, (b,), False)  # 调用函数 _propagate_shapes 进行形状传播
    def test_if_supertype(self):
        @torch.jit.script
        def tensor_unifying(x, y, z):
            # 检测动态类型是否适当设置为 y 和 z
            if bool(x):
                # 如果 x 是真值，执行以下赋值操作
                x, y, z = x + 1, y, z
            else:
                # 如果 x 是假值，执行以下赋值操作
                x, y, z = x + 1, x, y

            # 返回更新后的 x, y, z 值
            return x, y, z

        # 创建三个全零张量，分别为 float32、int64 和 float32 类型
        a = torch.zeros(2, 2, dtype=torch.float)
        b = torch.zeros(2, 4, dtype=torch.long)
        c = torch.zeros(2, 4, dtype=torch.float)

        # 传入张量和函数图到 _propagate_shapes 函数中，返回新的图
        graph = _propagate_shapes(tensor_unifying.graph, (a, b, c), False)
        # 找到 if 语句节点并获取其输出
        if_outputs = list(graph.findNode("prim::If").outputs())
        # 断言 if 语句的输出类型符合预期
        self.assertTrue(if_outputs[0].type().str() == "Float(*, *, requires_grad=0, device=cpu)")
        self.assertTrue(if_outputs[1].type().str() == "Tensor(*, *, requires_grad=0, device=cpu)")
        self.assertTrue(if_outputs[2].type().str() == "Tensor(*, *, requires_grad=0, device=cpu)")

    def test_list_unify(self):
        # 允许一个统一的 int?[] 可能会导致运行时错误，因为
        # 索引操作期望 int?[] 是一个泛型列表，
        # 但在真分支中 IValue 将是一个 int 列表
        with self.assertRaisesRegex(RuntimeError, "int[] in the true branch and type None[]"):
            @torch.jit.script
            def list_optional_fails(x):
                # type: (bool) -> Optional[int]
                if x:
                    # 如果 x 为真，分配值 [1]
                    y = [1]
                else:
                    # 如果 x 为假，分配值 [None]
                    y = [None]  # noqa: T484
                return y[0]

        @torch.jit.script
        def list_tensors(x):
            # type: (bool) -> Tuple[Tensor, List[Tensor]]
            if x:
                # 如果 x 为真，创建形状为 [1, 1] 的全零张量 a，并将其放入列表 y 中
                a = torch.zeros([1, 1])
                y = [a]
            else:
                # 如果 x 为假，创建形状为 [1, 2] 的全零张量 a，并将其放入列表 y 中
                a = torch.zeros([1, 2])
                y = [a]
            return a, y

        # 对 list_tensors 函数的图进行常量传播优化
        self.run_pass('constant_propagation', list_tensors.graph)
        # 从优化后的图创建函数对象 m
        m = self.createFunctionFromGraph(list_tensors.graph)
        # 测试列表的张量类型是否已统一
        self.getExportImportCopy(m)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @_inline_everything
    # 定义一个测试方法，用于验证常量未特化时的行为
    def test_import_constants_not_specialized(self):
        # 定义一个继承自 torch.nn.Module 的模块类 Mod
        class Mod(torch.nn.Module):
            # 定义模块的前向传播方法，将输入张量 x 连接成两倍长度的张量
            def forward(self, x):
                return torch.cat(2 * [x], dim=0)

        # 定义一个继承自 torch.jit.ScriptModule 的脚本模块类 ScriptMod
        class ScriptMod(torch.jit.ScriptModule):
            # 初始化方法，接受一个模块实例 mod
            def __init__(self, mod):
                super().__init__()
                # 创建一个形状为 (1, 3) 的全零张量 x
                x = torch.zeros(1, 3)
                # 定义一个 lambda 函数 mod_fn，调用 mod 对象对 x 进行前向计算
                mod_fn = lambda: mod(x)  # noqa: E731
                # 使用 torch.jit.trace 方法对 mod_fn 进行追踪，形成脚本模块的 mod
                self.mod = torch.jit.trace(mod_fn, tuple())

            # 定义一个 torch.jit.script_method 装饰的方法 forward
            @torch.jit.script_method
            def forward(self):
                # 返回模块 self.mod 的结果
                return self.mod()

        # 创建一个 ScriptMod 实例 cm，使用 Mod 类实例化
        cm = ScriptMod(Mod())
        # 在图中检查是否存在特化后的张量
        FileCheck().check("Float(1, 3, strides=[3, 1], requires_grad=0, device=cpu)").run(cm.forward.graph)
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将 cm 模块保存到字节流缓冲区中
        torch.jit.save(cm, buffer)
        # 将缓冲区指针移到开头
        buffer.seek(0)
        # 从字节流缓冲区加载模块 cm_load
        cm_load = torch.jit.load(buffer)
        # 检查加载后的图中是否存在未特化的 Float(1, 3) 张量
        FileCheck().check_not("Float(1, 3)").run(cm_load.forward.graph)

    # 跳过 TorchDynamo 测试失败的情况
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    # 定义一个测试方法，验证重复列表类型注解的行为
    def test_type_annotations_repeated_list(self):
        # 定义一个 torch.jit.script 装饰的函数 float_fn，接受一个 float 和 BroadcastingList3[float] 类型参数，返回 List[float]
        @torch.jit.script
        def float_fn(x, y):
            # 返回参数 y
            return y
        # 断言 float_fn(2.0, 1.0) 和 float_fn(2.0, [1.0, 1.0, 1.0]) 的结果相等
        self.assertEqual(float_fn(2.0, 1.0), float_fn(2.0, [1.0, 1.0, 1.0]))
        # 断言 float_fn(2.0, 1.0) 和 float_fn(2.0, (1.0, 1.0, 1.0)) 的结果相等
        self.assertEqual(float_fn(2.0, 1.0), float_fn(2.0, (1.0, 1.0, 1.0)))

        # 定义一个 torch.jit.script 装饰的函数 float_fn_call
        @torch.jit.script
        def float_fn_call():
            # 打印 float_fn 的两种不同调用结果
            print(float_fn(1.0, 1.0))
            print(float_fn(1.0, (1.0, 1.0, 1.0)))

        # 定义一个 torch.jit.script 装饰的函数 int_fn，接受一个 BroadcastingList3[int] 类型参数，返回 List[int]
        @torch.jit.script
        def int_fn(x):
            # 返回参数 x
            return x
        # 断言 int_fn(1) 和 int_fn([1, 1, 1]) 的结果相等
        self.assertEqual(int_fn(1), int_fn([1, 1, 1]))
        # 断言 int_fn(1) 和 int_fn((1, 1, 1)) 的结果相等
        self.assertEqual(int_fn(1), int_fn((1, 1, 1)))

        # 定义一个 torch.jit.script 装饰的函数 int_fn_call
        @torch.jit.script
        def int_fn_call():
            # 打印 int_fn 的两种不同调用结果
            print(int_fn(1))
            print(int_fn((1, 1, 1)))

        # 使用 self.assertRaisesRegex 检查 RuntimeError 异常，验证不合法的类型构造器错误信息
        with self.assertRaisesRegex(RuntimeError, "must be a positive integer:"):
            @torch.jit.script  # noqa: T484
            # 定义一个 torch.jit.script 装饰的函数 fn，接受 BroadcastingListx[int] 类型参数，返回 List[int]
            def fn(x):
                # 返回参数 x
                return x

        # 使用 self.assertRaisesRegex 检查 RuntimeError 异常，验证未知类型构造器错误信息
        # 创建一个 torch.jit.CompilationUnit 对象 cu，定义一个包含错误类型构造器的函数 nested
        with self.assertRaisesRegex(RuntimeError, "Unknown type constructor"):
            cu = torch.jit.CompilationUnit('''
                def nested(x, y):
                    # type: (int, Tuple[int, int[2]]) -> List[int]
                    return x  # noqa: T484
            ''')

        # 定义一个 torch.jit.script 装饰的函数 f，接受 BroadcastingList2[int] 类型参数，返回该参数
        @torch.jit.script
        def f(x: BroadcastingList2[int]):
            return x

        # 调用函数 f，传入参数 1，验证返回值的类型为 List[int]
        out = f(1)
        # 断言输出列表的第一个元素是整数类型
        self.assertTrue(isinstance(out[0], int))
        # 断言输出结果为 [1, 1]
        self.assertEqual(out, [1, 1])
    def test_ntuple_builtins(self):
        from torch.nn.modules.utils import _single, _pair, _triple, _quadruple

        def test_ints():
            # 调用 _single 函数，返回单个值 (1,)
            return _single(1), _pair(2), _triple(3), _quadruple(4)

        def test_floats():
            # 调用 _pair 函数，返回两个值 (1, 1)
            # 调用 _triple 函数，返回三个值 (3.1, 3.1, 3.1)
            # 调用 _quadruple 函数，返回四个值 (4.1, 4.1, 4.1, 4.1)
            return _single(1), _pair(2.1), _triple(3.1), _quadruple(4.1)

        # 检查 test_ints 函数的脚本化版本
        self.checkScript(test_ints, ())
        # 检查 test_floats 函数的脚本化版本
        self.checkScript(test_floats, ())

    def test_embedding_renorm_grad_error(self):
        # 测试 embedding_renorm_ 函数在调用 .backward() 后是否正确抛出错误

        def embedding_norm(input, embedding_matrix, max_norm):
            # 使用 F.embedding 函数对 input 进行嵌入，并限制 max_norm 为 0.01
            F.embedding(input, embedding_matrix, max_norm=0.01)

        @torch.jit.script
        def embedding_norm_script(input, embedding_matrix, max_norm):
            # type: (Tensor, Tensor, float) -> None
            # 使用 F.embedding 函数对 input 进行脚本化嵌入，并限制 max_norm 为 0.01
            F.embedding(input, embedding_matrix, max_norm=0.01)

        # 对 embedding_norm 和 embedding_norm_script 函数进行迭代测试
        for _ in [embedding_norm, embedding_norm_script]:
            input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
            embedding_matrix = torch.randn(10, 3)

            var1 = torch.randn(10, 3, requires_grad=True)
            var2 = var1.detach().requires_grad_()
            output1 = var1 * embedding_matrix
            output2 = var2 * embedding_matrix

            # 对 output1 进行求和并反向传播梯度
            output1.sum().backward()

            # 调用 F.embedding 函数，并期望在对 output2 求和并调用 .backward() 时抛出 RuntimeError 错误
            ignore = F.embedding(input, embedding_matrix, max_norm=0.01)
            with self.assertRaisesRegex(RuntimeError, "modified"):
                output2.sum().backward()

    def test_type_annotations(self):
        def fn(x, y):
            # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
            # 返回值为 x, x * 2, x * 3 的元组
            return x, x * 2, x * 3

        # 期望在脚本化版本中调用 fn 函数时抛出 RuntimeError 错误，因为返回的值少于需要的 4 个
        with self.assertRaisesRegex(RuntimeError, r"need 4 values .* found only 3"):
            @torch.jit.script
            def script_fn(x):
                x, y, z, w = fn(x, x)

        # 期望在脚本化版本中调用 fn 函数时抛出 RuntimeError 错误，因为返回的值多于需要的 2 个
        with self.assertRaisesRegex(RuntimeError, r"too many values .* need 2 but found 3"):
            @torch.jit.script
            def script_fn2(x):
                x, y = fn(x, x)

        def fn_unpack(x):
            # 调用 fn 函数并解包结果，只返回第一个值
            y, z, w = fn(x, x)
            return y

        def fn_index(x):
            # 直接返回 fn 函数的结果
            q = fn(x, x)
            return x

        def fn_string(str, strpair):
            # type: (str, Tuple[str, str]) -> Tuple[str, int, str, str]
            str1, str2 = strpair
            # 返回一个元组，包含 str, 2, str1, str2
            return str, 2, str1, str2

        x = torch.ones(2, 2)
        # 检查 fn_unpack 函数的脚本化版本
        self.checkScript(fn_unpack, (x,), optimize=True)
        # 检查 fn_index 函数的脚本化版本
        self.checkScript(fn_index, (x,), optimize=True)
        # 检查 fn_string 函数的脚本化版本
        self.checkScript(fn_string, ("1", ("3", "4")), optimize=True)
    # 定义测试函数，用于测试类型注解和可变参数的情况
    def test_type_annotations_varargs(self):
        
        # 使用 torch.jit.ignore 装饰的函数，忽略其在 TorchScript 中的编译
        @torch.jit.ignore
        def fn_varargs(x, *args):
            # 如果有可变参数 args，则返回 args 的第一个元素；否则返回 x
            return args[0] if args else x

        # 定义函数 fn1，接收参数 x, y, z，调用 fn_varargs 函数并返回结果
        def fn1(x, y, z):
            return fn_varargs(x)

        # 定义函数 fn2，接收参数 x, y, z，调用 fn_varargs 函数并返回结果
        def fn2(x, y, z):
            return fn_varargs(x, y)

        # 定义函数 fn3，接收参数 x, y, z，调用 fn_varargs 函数并返回结果
        def fn3(x, y, z):
            return fn_varargs(x, y, z)

        # 生成随机的 Tensor x, y, z
        x, y, z = (torch.randn(2, 2) for _ in range(3))
        
        # 分别对 fn1, fn2, fn3 进行 TorchScript 的编译和类型检查
        self.checkScript(fn1, (x, y, z), optimize=True)
        self.checkScript(fn2, (x, y, z), optimize=True)
        self.checkScript(fn3, (x, y, z), optimize=True)

    # 定义测试函数，用于测试 Python 3 中的类型注解
    def test_type_annotation_py3(self):
        # 定义代码字符串，导入必要的模块和类型注解
        code = dedent("""
        import torch
        from torch import Tensor
        from typing import Tuple

        def fn(x : torch.Tensor, y : Tensor, z) -> Tuple[Tensor, Tensor, Tensor]:
            return (x, y + z, z)
        """)

        # 使用临时目录 tmp_dir 创建一个临时文件
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 定义脚本文件的路径
            script_path = os.path.join(tmp_dir, 'script.py')
            with open(script_path, 'w') as f:
                # 将代码写入临时脚本文件
                f.write(code)
            
            # 调用 get_fn 函数获取脚本文件中的函数 fn
            fn = get_fn('test_type_annotation_py3', script_path)
            # 忽略 fn 函数在 TorchScript 中的编译
            fn = torch.jit.ignore(fn)

            # 使用 torch.jit.script 装饰的函数 bad_fn，测试错误情况
            with self.assertRaisesRegex(RuntimeError, r"Expected a value of type 'Tensor' for argument"
                                                      r" 'x' but instead found type 'Tuple\[Tensor,"):
                @torch.jit.script
                def bad_fn(x):
                    # 调用 fn 函数，传入错误的参数类型
                    x, y = fn((x, x), x, x)
                    return y

            # 使用 torch.jit.script 装饰的函数 bad_fn2，测试错误情况
            with self.assertRaisesRegex(RuntimeError, r"too many values .* need 2 but found 3"):
                @torch.jit.script
                def bad_fn2(x):
                    # 调用 fn 函数，传入错误的参数数量
                    x, y = fn(x, x, x)
                    return y

            # 使用 torch.jit.script 装饰的函数 bad_fn3，测试错误情况
            with self.assertRaisesRegex(RuntimeError, r"need 4 values .* found only 3"):
                @torch.jit.script
                def bad_fn3(x):
                    # 调用 fn 函数，传入错误的参数数量
                    x, y, z, w = fn(x, x, x)
                    return y

            # 定义函数 good_fn，测试正确的参数传递情况
            def good_fn(x):
                # 调用 fn 函数，正确接收返回的值
                y, z, w = fn(x, x, x)
                return y, z, w

            # 对 good_fn 函数进行 TorchScript 的编译和类型检查
            self.checkScript(good_fn, (torch.ones(2, 2),), optimize=True)
    def test_type_annotation_module(self):
        class BaseModule(torch.jit.ScriptModule):
            @torch.jit.ignore
            def foo(self, x):
                # type: (Tensor) -> Tensor
                return x + 1

            @torch.jit.ignore
            def bar(self, x, y):
                # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
                return x + y, y

            @torch.jit.ignore
            def baz(self, x, y):
                # 没有类型注释，类型推断将决定参数和返回类型
                return x

        class ModuleTooMany(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                return self.foo(x, x)

        class ModuleTooFew(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                # 由于缺少参数，预期抛出运行时错误
                return self.bar(x)

        class ModuleTooManyAssign(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                # 解包时期望获得三个值，但只有两个值，预期抛出运行时错误
                y, z, w = self.bar(x, x)
                return x

        class ModuleDefault(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                # 缺少参数 'y'，预期抛出运行时错误
                y = self.baz(x)
                return x

        with self.assertRaisesRegex(RuntimeError, "Expected at most 2 arguments but found 3"):
            ModuleTooMany()
        with self.assertRaisesRegex(RuntimeError, "Argument y not provided"):
            ModuleTooFew()
        with self.assertRaisesRegex(RuntimeError, "need 3 values .* found only 2"):
            ModuleTooManyAssign()
        with self.assertRaisesRegex(RuntimeError, "Argument y not provided."):
            ModuleDefault()

    def test_type_inferred_from_empty_annotation(self):
        """
        Test that the type inferred from an empty or missing annotation is Torch.Tensor wtih `inferred=true`
        """
        @torch.jit.script
        def fn(x):
            return x

        graph = fn.graph
        n = next(graph.inputs())
        # 检查推断的类型是否为 Torch.Tensor
        self.assertTrue(n.type() == torch._C.TensorType.getInferred())

        with self.assertRaisesRegex(RuntimeError, "Inferred 'x' to be of type 'Tensor"):
            # 传递字符串而不是预期的张量，预期抛出运行时错误
            fn("1")

    def test_script_define_order(self):
        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                return input + 1

        m = M()
        # 测试方法定义顺序，确保调用正确
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    def test_script_define_order_recursive_fail(self):
        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                # 尝试递归调用自身方法，预期抛出运行时错误
                self.call_foo(input)

        with self.assertRaisesRegex(RuntimeError, 'called recursively'):
            M()
    # 定义一个测试类的方法，用于测试带有关键字参数的脚本函数调用
    def test_script_kwargs_fn_call(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 M
        class M(torch.jit.ScriptModule):

            # 定义一个脚本方法 call_foo，接受一个输入参数 input
            @torch.jit.script_method
            def call_foo(self, input):
                # 调用类中的 foo 方法，传递 input 参数和 bar=1 的关键字参数
                return self.foo(input=input, bar=1)

            # 定义另一个脚本方法 foo，接受参数 bar 和 input
            @torch.jit.script_method
            def foo(self, bar, input):
                # 指定 foo 方法的类型签名，表明其接受一个整数和一个 Tensor，并返回一个 Tensor
                # 计算并返回 input 和 bar 相加的结果
                return input + bar

        # 实例化类 M
        m = M()
        # 断言调用 m.call_foo 方法返回的结果为 2
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    # 定义一个测试函数，用于测试带有条件语句的脚本函数
    def test_if_define(self):

        # 定义一个脚本函数 foo，接受参数 a
        @torch.jit.script
        def foo(a):
            # 如果 a 等于 0，设置变量 b 为 1，否则设置为 0
            if bool(a == 0):
                b = 1
            else:
                b = 0
            # 返回 b 加 1 的结果
            return b + 1

        # 定义另一个脚本函数 foo2，接受参数 a
        @torch.jit.script
        def foo2(a):
            # 初始化变量 b 为 0
            b = 0
            # 如果 a 等于 0，设置变量 b 为 1
            if bool(a == 0):
                b = 1
            # 返回 b 加 1 的结果
            return b + 1

        # 定义另一个脚本函数 foo3，接受参数 a
        @torch.jit.script
        def foo3(a):
            # 初始化变量 b 为 1
            b = 1
            # 如果 a 等于 0，设置变量 c 为 4，否则设置变量 b 为 0
            if bool(a == 0):
                c = 4
            else:
                b = 0
            # 返回 b 加 1 的结果
            return b + 1

        # 初始化变量 a 和 b，分别为 torch.long 类型的全 1 和全 0 Tensor
        a = torch.ones(1, dtype=torch.long)
        b = torch.zeros(1, dtype=torch.long)
        
        # 断言调用 foo 函数，传递参数 a 和 b，返回值分别为 1 和 2
        self.assertEqual(1, foo(a))
        self.assertEqual(2, foo(b))
        
        # 断言调用 foo2 函数，传递参数 a 和 b，返回值分别为 1 和 2
        self.assertEqual(1, foo2(a))
        self.assertEqual(2, foo2(b))
        
        # 断言调用 foo3 函数，传递参数 a 和 b，返回值分别为 1 和 2
        self.assertEqual(1, foo3(a))
        self.assertEqual(2, foo3(b))
    # 定义测试方法，用于测试导出子模块的脚本模块
    def test_script_module_export_submodule(self):
        # 定义 M1 类，继承自 torch.jit.ScriptModule
        class M1(torch.jit.ScriptModule):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义模块的权重参数为一个包含随机数的张量
                self.weight = nn.Parameter(torch.randn(2))

            # 使用 torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, thing):
                # 返回权重和输入的和
                return self.weight + thing

        # 定义 M2 类，继承自 torch.jit.ScriptModule
        class M2(torch.jit.ScriptModule):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 M1 的实例作为子模块
                self.sub = M1()
                # 定义模块的权重参数为一个 2x3 的张量
                self.weight = nn.Parameter(torch.randn(2, 3))
                # 定义模块的偏置参数为一个包含随机数的张量
                self.bias = nn.Parameter(torch.randn(2))
                # 使用 define 方法定义字符串形式的函数
                self.define("""
                    def hi(self, a):
                        return self.weight.mm(a)
                """)

            # 使用 torch.jit.script_method 装饰器定义方法，进行矩阵乘法运算
            @torch.jit.script_method
            def doit(self, input):
                return self.weight.mm(input)

            # 使用 torch.jit.script_method 装饰器定义方法，进行矩阵乘法运算
            @torch.jit.script_method
            def doit2(self, input):
                return self.weight.mm(input)

            # 使用 torch.jit.script_method 装饰器定义方法，将输入加一
            @torch.jit.script_method
            def doit3(self, input):
                return input + torch.ones([1], dtype=torch.double)

            # 使用 torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, input):
                # 调用 doit、doit2 和 hi 方法，并将它们的结果相加，再加上偏置参数
                a = self.doit(input)
                b = self.doit2(input)
                c = self.hi(input)
                return a + b + self.bias + c

        # 关闭优化执行设置
        with torch.jit.optimized_execution(False):
            # 创建 M2 的原始实例和导出/导入后的实例
            m_orig = M2()
            m_import = self.getExportImportCopy(m_orig)

            # 创建随机输入张量
            input = torch.randn(3, 2)
            # 断言导出/导入的实例执行 doit 方法得到的结果与原始实例相同
            self.assertEqual(m_orig.doit(input), m_import.doit(input))
            # 断言导出/导入的实例执行 hi 方法得到的结果与原始实例相同
            self.assertEqual(m_orig.hi(input), m_import.hi(input))
            # 断言导出/导入的实例执行 doit3 方法得到的结果与原始实例相同
            self.assertEqual(m_orig.doit3(input), m_import.doit3(input))
            # 断言导出/导入的实例执行前向传播方法得到的结果与原始实例相同
            self.assertEqual(m_orig.forward(input), m_import.forward(input))

    # 使用 slowTest 装饰器定义测试编译带常量的模块方法
    @slowTest
    def test_compile_module_with_constant(self):
        # 定义 Double 类，继承自 nn.Module
        class Double(nn.Module):
            # 初始化方法
            def __init__(self, downsample=None):
                super().__init__()

            # 定义前向传播方法，将输入乘以2
            def forward(self, input):
                return input * 2

        # 定义 Mod 类，继承自 nn.Module
        class Mod(nn.Module):
            # 声明常量列表 '__constants__'
            __constants__ = ['downsample']

            # 初始化方法
            def __init__(self, downsample=None):
                super().__init__()
                # 设置 downsample 属性
                self.downsample = downsample

            # 定义前向传播方法
            def forward(self, input):
                # 如果 downsample 不为 None，则调用 downsample 方法
                if self.downsample is not None:
                    return self.downsample(input)
                return input

        # 使用 torch.jit.script 将 Mod 类实例化为脚本模块
        none_mod = torch.jit.script(Mod(None))
        # 使用 torch.jit.script 将 Mod 类实例化为脚本模块，传入 Double 类的实例作为参数
        double_mod = torch.jit.script(Mod(Double()))
        # 断言对 none_mod 的调用结果与预期相同
        self.assertEqual(none_mod(torch.tensor(1)), torch.tensor(1))
        # 断言对 double_mod 的调用结果与预期相同
        self.assertEqual(double_mod(torch.tensor(1)), torch.tensor(1) * 2)

    # 定义测试设备关键字参数的方法
    def test_device_kwarg(self):
        # 导入 torch 的 device 函数
        from torch import device

        # 定义函数 f
        def f():
            # 返回 CUDA 类型的设备和 CPU 类型的设备
            return device(type='cuda'), torch.device(type='cpu')
        # 调用 checkScript 方法验证函数 f 的脚本化结果
        self.checkScript(f, ())
    def test_script_module_export_tensor_type(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M，接受一个类型参数
        class M(torch.jit.ScriptModule):
            def __init__(self, type):
                super().__init__()
                # 创建一个形状为 (5, 5) 的零张量，并根据给定类型随机初始化
                self.param = torch.nn.Parameter(torch.zeros((5, 5), dtype=type).random_())

            @torch.jit.script_method
            def foo(self):
                # 返回 param 属性的值
                return self.param

        # 关闭优化执行
        with torch.jit.optimized_execution(False):
            # 对于每种类型，创建原始模型和导入的模型副本
            for type in [torch.float, torch.double]:
                m_orig = M(type)
                m_import = self.getExportImportCopy(m_orig)
                # 检查存储是否没有调整大小
                self.assertTrue(m_orig.param.storage().size() == 25)
                # 检查原始模型和导入模型的 foo 方法返回值是否相等
                self.assertEqual(m_orig.foo(), m_import.foo())
                # 检查原始模型和导入模型的 foo 方法返回值的数据类型是否相等
                self.assertTrue(m_orig.foo().dtype == m_import.foo().dtype)

    @unittest.skipIf(not RUN_CUDA, "testing cuda tensors require CUDA")
    def test_script_module_export_tensor_cuda(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):

            def __init__(self):
                super().__init__()
                # 在 CUDA 设备上创建一个形状为 (5, 5) 的零张量，并随机初始化
                self.param = torch.nn.Parameter(torch.zeros((5, 5), device='cuda:0').random_())

            @torch.jit.script_method
            def foo(self):
                # 返回 param 属性的值
                return self.param

        # 创建原始模型和导入的模型副本
        m_orig = M()
        m_import = self.getExportImportCopy(m_orig)
        # 检查存储是否没有调整大小
        self.assertTrue(m_orig.param.storage().size() == 25)
        # 检查原始模型和导入模型的 foo 方法返回值的设备是否为 CUDA 设备
        self.assertTrue(m_import.foo().device == torch.device('cuda:0'))
        # 检查原始模型和导入模型的 foo 方法返回值是否相等
        self.assertEqual(m_orig.foo(), m_import.foo())
        # 检查原始模型和导入模型的 foo 方法返回值的数据类型是否相等
        self.assertTrue(m_orig.foo().dtype == m_import.foo().dtype)

    def test_script_module_export_blocks(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M，接受两个参数 n 和 m
        class M(torch.jit.ScriptModule):
            def __init__(self, n, m):
                super().__init__()
                # 创建一个形状为 (n, m) 的随机初始化的权重张量
                self.weight = torch.nn.Parameter(torch.rand(n, m))

            @torch.jit.script_method
            def forward(self, input):
                # 如果输入的和大于 0，则执行矩阵向量乘法操作
                if bool(input.sum() > 0):
                    output = self.weight.mv(input)
                else:
                    # 否则，执行张量的加法操作
                    output = self.weight + input
                return output

        # 创建原始模型和导入的模型副本
        m_orig = M(200, 200)
        m_import = self.getExportImportCopy(m_orig)

        # 创建一个形状为 (200,) 的随机输入张量，并分别对比原始模型和导入模型的输出结果
        t = torch.rand(200)
        self.assertEqual(m_orig(t), m_import(t))
    def test_script_module_export_shared_storage(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):

            def __init__(self):
                super().__init__()
                # 添加名为 param1 的参数，大小为 5x5，数据为随机生成的张量
                self.param1 = torch.nn.Parameter(torch.rand(5, 5))
                # 添加名为 param2 的参数，值为 param1 的第四行（索引为 3）的元素，作为参数的一个部分
                self.param2 = torch.nn.Parameter(self.param1[3])
                # 添加名为 param3 的参数，大小为 5x5，数据为随机生成的张量
                self.param3 = torch.nn.Parameter(torch.rand(5, 5))
                # 添加名为 param4 的参数，大小为 11x5，数据为随机生成的张量的部分切片（索引从 1 到 6）
                self.param4 = torch.nn.Parameter(torch.rand(11, 5)[1:6])

            @torch.jit.script_method
            # 定义一个脚本方法 foo，返回 param1、param2、param3 和 param4 的和
            def foo(self):
                return self.param1 + self.param2 + self.param3 + self.param4

        # 关闭优化执行，创建 M 类的原始实例 m_orig 和导入实例 m_import
        with torch.jit.optimized_execution(False):
            m_orig = M()
            m_import = self.getExportImportCopy(m_orig)

            # 断言 m_orig 和 m_import 的 foo 方法返回值相等
            self.assertEqual(m_orig.foo(), m_import.foo())

            # 断言 m_import 的 param1 和 param2 共享存储的数据指针相同
            self.assertTrue(m_import.param1.storage().data_ptr() == m_import.param2.storage().data_ptr())
            # 断言 m_import 的 param1 和 param3 共享存储的数据指针不同
            self.assertTrue(m_import.param1.storage().data_ptr() != m_import.param3.storage().data_ptr())

    def test_sequential_intermediary_types(self):
        # 定义一个继承自 torch.nn.Module 的类 A，实现一个前向传播函数，对输入 x 加上 3
        class A(torch.nn.Module):
            def forward(self, x):
                return x + 3

        # 定义一个继承自 torch.nn.Module 的类 B，实现一个前向传播函数，将输入 x 包装在字典中
        class B(torch.nn.Module):
            def forward(self, x):
                return {"1": x}

        # 定义一个继承自 torch.nn.Module 的类 C，构造函数初始化一个名为 foo 的 Sequential 容器，包含类 A 和 B 的实例
        class C(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Sequential(A(), B())

            def forward(self, x):
                # 前向传播函数返回 foo 对输入 x 的应用结果
                return self.foo(x)

        # 使用 checkModule 方法检查类 C 的实例对输入 torch.tensor(1) 的运算结果
        self.checkModule(C(), (torch.tensor(1),))

    def test_ellipsis_const_mid(self):
        # 定义函数 ellipsize，接受一个类型为 Tensor 的参数 x，返回部分切片的大小
        def ellipsize(x):
            # 返回 x 的指定切片 [2, ..., 0:4, 4:8] 的大小
            return x[2, Ellipsis, 0:4, 4:8].size()

        # 创建一个全零的大小为 (8, 8, 8, 8, 8) 的张量 dummy
        dummy = torch.zeros(8, 8, 8, 8, 8)
        # 使用 checkScript 方法检查函数 ellipsize 对 dummy 的运算结果，开启优化
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_const_mid_select(self):
        # 定义函数 ellipsize，接受一个类型为 Tensor 的参数 x，返回部分切片的大小
        def ellipsize(x):
            # 返回 x 的指定切片 [2, ..., 4, 4, 4:8, 2] 的大小
            return x[2, Ellipsis, 4, 4, 4:8, 2].size()

        # 创建一个全零的大小为 (8, 8, 8, 8, 8, 8, 8) 的张量 dummy
        dummy = torch.zeros(8, 8, 8, 8, 8, 8, 8)
        # 使用 checkScript 方法检查函数 ellipsize 对 dummy 的运算结果，开启优化
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_const_start(self):
        # 定义函数 ellipsize，接受一个类型为 Tensor 的参数 x，返回部分切片的大小
        def ellipsize(x):
            # 返回 x 的指定切片 [Ellipsis, 0:4, 4:8] 的大小
            return x[Ellipsis, 0:4, 4:8].size()
        
        # 创建一个全零的大小为 (8, 8, 8, 8, 8) 的张量 dummy
        dummy = torch.zeros(8, 8, 8, 8, 8)
        # 使用 checkScript 方法检查函数 ellipsize 对 dummy 的运算结果，开启优化
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_const_end(self):
        # 定义函数 ellipsize，接受一个类型为 Tensor 的参数 x，返回部分切片的大小
        def ellipsize(x):
            # 返回 x 的指定切片 [0:4, 2, Ellipsis] 的大小
            return x[0:4, 2, Ellipsis].size()

        # 创建一个全零的大小为 (8, 8, 8, 8, 8) 的张量 dummy
        dummy = torch.zeros(8, 8, 8, 8, 8)
        # 使用 checkScript 方法检查函数 ellipsize 对 dummy 的运算结果，开启优化
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_mid(self):
        # 定义函数 ellipsize，接受一个类型为 Tensor 的参数 x，返回部分切片的大小
        def ellipsize(x):
            # 返回 x 的指定切片 [2, ..., 0:4, 4:8] 的大小
            return x[2, ..., 0:4, 4:8].size()

        # 创建一个全零的大小为 (8, 8, 8, 8, 8) 的张量 dummy
        dummy = torch.zeros(8, 8, 8, 8, 8)
        # 使用 checkScript 方法检查函数 ellipsize 对 dummy 的运算结果，开启优化
        self.checkScript(ellipsize, (dummy,), optimize=True)
    def test_ellipsis_mid_select(self):
        # 定义一个函数 ellipsize，接受一个张量参数 x，返回一个整数列表
        def ellipsize(x):
            # type: (Tensor) -> List[int]
            # 返回从张量 x 中选择的子张量的尺寸大小
            return x[2, ..., 4, 4, 4:8, 2].size()

        # 创建一个全零张量作为测试数据
        dummy = torch.zeros(8, 8, 8, 8, 8, 8, 8)
        # 对 ellipsize 函数进行脚本化，检查优化结果
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_start(self):
        # 定义一个函数 ellipsize，接受一个张量参数 x，返回一个整数列表
        def ellipsize(x):
            # type: (Tensor) -> List[int]
            # 返回从张量 x 中选择的子张量的尺寸大小
            return x[..., 0:4, 4:8].size()
        
        # 创建一个全零张量作为测试数据
        dummy = torch.zeros(8, 8, 8, 8, 8)
        # 对 ellipsize 函数进行脚本化，检查优化结果
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_end(self):
        # 定义一个函数 ellipsize，接受一个张量参数 x，返回一个整数列表
        def ellipsize(x):
            # type: (Tensor) -> List[int]
            # 返回从张量 x 中选择的子张量的尺寸大小
            return x[0:4, 2, ...].size()
        
        # 创建一个全零张量作为测试数据
        dummy = torch.zeros(8, 8, 8, 8, 8)
        # 对 ellipsize 函数进行脚本化，检查优化结果
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_torch_manual_seed(self):
        # 使用 torch 的冻结随机数生成器状态上下文管理器
        with freeze_rng_state():
            # 定义一个测试函数
            def test():
                # 设定随机种子为 2
                torch.manual_seed(2)
                # 返回一个随机张量
                return torch.rand(1)

            # 将 test 函数脚本化
            script = torch.jit.script(test)
            # 断言原始函数和脚本化函数的输出相等
            self.assertEqual(test(), script())
            # 获取脚本化图形对象
            graph = script.graph_for()
            # 运行 FileCheck 检查图形中是否包含 "aten::manual_seed"
            FileCheck().check("aten::manual_seed").run(graph)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_index_select_shape_prop(self):

        @torch.jit.script
        def foo(x, y):
            # 使用 torch 的索引选择函数，选择张量 x 的维度 1 上的索引 y
            return torch.index_select(x, index=y, dim=1)

        # 创建两个全零张量作为测试数据
        a = torch.zeros(2, 2)
        b = torch.zeros(4, dtype=torch.long)
        # 执行完整的形状分析过程
        torch._C._jit_pass_complete_shape_analysis(foo.graph, (a, b), False)
        # 运行 FileCheck 检查图形的输出格式
        FileCheck().check("Float(2, 4, strides=[4, 1], requires_grad=0, device=cpu)").run(str(foo.graph))

    def test_shape_analysis_loop(self):
        # 定义一个函数 foo，接受三个参数 a, b, x，返回一个张量
        def foo(a, b, x):
            # 将 a 赋值给 c
            c = a
            # 在循环中执行一系列操作
            for _ in range(2):
                # 更新 a 为 c + b
                a = c + b
                # 更新 c 为 x
                c = x
                # 更新 b 为 x
                b = x
            # 返回最终结果张量 a
            return a

        # 对 foo 函数进行脚本化，不进行优化
        self.checkScript(foo, (torch.zeros(1), torch.zeros(4), torch.zeros(5)), optimize=False)

    def test_intlist_args(self):
        # 定义一个函数 func_1，接受一个张量参数 x，返回一个适应性平均池化的结果
        def func_1(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, 1)

        # 定义一个函数 func_2，接受一个张量参数 x，返回一个适应性平均池化的结果
        def func_2(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=1)

        # 定义一个函数 func_3，接受一个张量参数 x，返回一个适应性平均池化的结果
        def func_3(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=[1])

        # 创建一个随机张量作为测试数据
        x = torch.randn(8, 8, 8)
        # 对 func_1 函数进行脚本化，检查优化结果
        self.checkScript(func_1, [x], optimize=True)
        # 对 func_2 函数进行脚本化，检查优化结果
        self.checkScript(func_2, [x], optimize=True)
        # 对 func_3 函数进行脚本化，检查优化结果
        self.checkScript(func_3, [x], optimize=True)
    def test_wrong_implicit_expand(self):
        # 定义一个测试函数，验证在未显式扩展参数时的行为

        @_trace(torch.zeros(3), torch.zeros(1))
        # 使用装饰器进行跟踪，传入两个张量作为参数
        def foo(a, b):
            return a + b

        # 创建两个随机张量
        a = torch.rand(4)
        b = torch.rand(4)
        # 断言两种计算方式得到的结果相等
        self.assertEqual(a + b, foo(a, b))

    def test_builtin_args_fails(self):
        # 测试在使用内置函数参数时的失败情况

        with self.assertRaisesRegex(RuntimeError, 'Argument self not provided'):
            @torch.jit.script
            # 使用 Torch 脚本装饰器
            def f1(a):
                torch.sum(foo=4)

        with self.assertRaisesRegex(RuntimeError, 'specified twice'):
            @torch.jit.script
            def f2(a):
                torch.sum(a, self=a)

        with self.assertRaisesRegex(RuntimeError, 'not provided'):
            @torch.jit.script
            def f3(a):
                torch.sum(dim=4)

        with self.assertRaisesRegex(RuntimeError, 'for argument \'tensors\' but instead found type \'Tensor'):
            @torch.jit.script
            def f4(a):
                torch.cat(a)

        with self.assertRaisesRegex(RuntimeError, r'argument \'tensors\' but instead found type \'List\[int\]'):
            @torch.jit.script
            def f5(a):
                torch.cat([3])

        with self.assertRaisesRegex(RuntimeError, r'Expected a value of'
                                    r' type \'List\[int\]\' for argument'
                                    r' \'size\' but instead found type '
                                    r'\'List\[Union\[List\[int\], int\]\]'):
            @torch.jit.script
            def f6(a):
                a.expand(size=[3, [4]])

    def test_builtin_args(self):
        # 测试使用内置函数参数的情况

        def t0(a):
            # default arg dim
            # 默认参数 dim
            return torch.cat([a, a])

        self.checkScript(t0, (torch.zeros(1, 1),))

        def t1(a):
            # keywords out of order
            # 关键字顺序错误
            return torch.cat(dim=1, tensors=[a, a])

        self.checkScript(t1, (torch.zeros(1, 1, 2),))

        def t2(a):
            # mix const/non-const attributes
            # 混合常量和非常量属性
            if 1 == 1:
                b = 1
            else:
                b = 0
            return torch.sum(a, dim=b, keepdim=False)

        self.checkScript(t2, (torch.zeros(1, 1, 2),))

    def test_parser_type_annotations(self):
        # 测试解析器中的类型注解

        cu = torch.jit.CompilationUnit('''
            def foo(x : Tensor, y : Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
                return x, x
        ''')

        self.assertExpected(str(cu.foo.schema))

    def test_parser_type_annotations_comment(self):
        # 测试带有类型注释的解析器

        cu = torch.jit.CompilationUnit('''
            def foo(x, y):
                # type: (Tensor, Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tuple[Tensor, Tensor]
                return x, x
        ''')

        self.assertExpected(str(cu.foo.schema))
    # 测试解析器在遇到未知类型时的行为，期望抛出 RuntimeError 异常，并包含指定错误信息
    def test_parser_type_annotations_unknown_type(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown type name 'Foo'"):
            # 使用 torch.jit.CompilationUnit 编译一个包含类型注解的函数
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tuple[Tuple[Foo, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    # 测试解析器在遇到非标识符类型注解的情况下的行为，期望抛出 RuntimeError 异常，并包含指定错误信息
    def test_parser_type_annotations_subscript_non_ident(self):
        with self.assertRaisesRegex(RuntimeError, r'Subscripted type must be a type identifier'):
            # 使用 torch.jit.CompilationUnit 编译一个包含类型注解的函数
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tuple[Tensor, Tensor][Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    # 测试解析器在遇到张量类型的子类型注解时的行为，期望抛出 RuntimeError 异常，并包含指定错误信息
    def test_parser_type_annotations_subscript_tensor(self):
        with self.assertRaisesRegex(RuntimeError, r'Unknown type constructor Tensor'):
            # 使用 torch.jit.CompilationUnit 编译一个包含类型注解的函数
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tensor[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    # 测试解析器在遇到类型表达式中不兼容的表达式时的行为，期望抛出 RuntimeError 异常，并包含指定错误信息
    def test_parser_type_annotations_incompatible_expression(self):
        with self.assertRaisesRegex(RuntimeError, r'Expression of type \+ cannot be used in a type expression'):
            # 使用 torch.jit.CompilationUnit 编译一个包含类型注解的函数
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tuple[3 + 4, Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    # 测试 gather_dynamic_index 函数
    def test_gather_dynamic_index(self):
        def t(x):
            # 取第一个索引的值
            gather1 = x[0]
            # 动态计算索引值
            idx = 0 + 1
            # 根据动态计算的索引取值
            gather2 = x[idx]
            # 返回两个 gather 值的和
            return gather1 + gather2

        # 使用 self.checkScript 检查 t 函数的脚本化版本
        self.checkScript(t, (torch.zeros(3, 2, 3),))

    # 测试 Torch 中 @torch.jit.ignore 的使用，验证方法被忽略后是否影响模型脚本化
    def test_torch_ignore_conversion_to_none(self):
        # 定义模型类 A
        class A(torch.nn.Module):
            # 使用 @torch.jit.ignore 标记的方法，参数为 int 类型，返回 None
            @torch.jit.ignore
            def ignored(self, a: int) -> None:
                # 计算一个列表中满足条件的元素个数
                l: int = len([2 for i in range(a) if i > 2])
                return

            # 正向传播方法，返回类型为 int
            def forward(self) -> int:
                a: int = 4
                b: int = 5
                # 调用被忽略的方法
                self.ignored(a)
                # 返回 a 和 b 的和
                return a + b

        # 将模型 A 脚本化
        modelA = torch.jit.script(A())
        # 断言模型 A 执行后的结果为 9
        self.assertEqual(modelA(), 9)

        # 定义模型类 B
        class B(torch.nn.Module):
            # 使用 @torch.jit.ignore 标记的方法，参数为 int 类型，返回值忽略
            @torch.jit.ignore
            def ignored(self, a: int):
                # 计算一个列表中满足条件的元素个数
                l: int = len([2 for i in range(a) if i > 2])
                return

            # 正向传播方法，返回类型为 int
            def forward(self) -> int:
                a: int = 4
                b: int = 5
                # 调用被忽略的方法
                self.ignored(a)
                # 返回 a 和 b 的和
                return a + b

        # 将模型 B 脚本化
        modelB = torch.jit.script(B())
        # 断言模型 B 执行后的结果为 9
        self.assertEqual(modelB(), 9)
    # 定义测试函数 test_addmm_grad，用于测试以下几个条件：
    # 1. 在操作偏置项的 addmm 操作之前插入了一个扩展节点。
    # 2. 执行的最终图中包含了 addmm 的融合形式。
    # 3. 为沿着偏置项的第0维累积梯度发出了一个求和操作。
    # 4. 对 mm 操作的反向传播的正确符号表示（x.t() -> mm）被发出。

    @torch.jit.script
    # 定义一个 torch script 函数 addmm_grad_test，接受参数 b, x, w，返回 torch.addmm(b, x, w) 的结果
    def addmm_grad_test(b, x, w):
        return torch.addmm(b, x, w)

    # 初始化参数和输入值
    w_init = torch.rand(2, 5)  # 创建一个形状为 (2, 5) 的随机张量 w_init
    b_init = torch.rand(5)      # 创建一个形状为 (5,) 的随机张量 b_init
    x = torch.rand(3, 2)         # 创建一个形状为 (3, 2) 的随机张量 x

    # 克隆可训练参数
    b = b_init.clone()          # 克隆 b_init 到 b
    b.requires_grad_()          # 设置 b 的 requires_grad 属性为 True，使其可以计算梯度
    w = w_init.clone()          # 克隆 w_init 到 w
    w.requires_grad_()          # 设置 w 的 requires_grad 属性为 True，使其可以计算梯度

    # 测试符号微分
    y = addmm_grad_test(b, x, w)  # 调用 addmm_grad_test 函数计算结果赋给 y
    y.sum().backward()           # 对 y 的和进行反向传播计算梯度

    # 为自动求导参考克隆参数
    b_ref = b_init.clone()       # 克隆 b_init 到 b_ref
    b_ref.requires_grad_()       # 设置 b_ref 的 requires_grad 属性为 True，使其可以计算梯度
    w_ref = w_init.clone()       # 克隆 w_init 到 w_ref
    w_ref.requires_grad_()       # 设置 w_ref 的 requires_grad 属性为 True，使其可以计算梯度
    y_ref = torch.addmm(b_ref, x, w_ref)  # 使用克隆的参数计算 torch.addmm(b_ref, x, w_ref) 的结果赋给 y_ref
    y_ref.sum().backward()       # 对 y_ref 的和进行反向传播计算梯度

    # 断言梯度的正确性
    self.assertEqual(w.grad, w_ref.grad)  # 断言 w 的梯度与 w_ref 的梯度相等
    self.assertEqual(b.grad, b_ref.grad)  # 断言 b 的梯度与 b_ref 的梯度相等

@unittest.skipIf(not RUN_CUDA, "running tests on cuda to verify cudnn fix")
    def test_batch_norm_inference_backward_cuda(self):
        with enable_profiling_mode_for_profiling_tests():
            # 定义一个自定义的批归一化模块
            class MyBatchNorm(torch.nn.Module):
                def __init__(self, num_features, affine, track_running_stats):
                    super().__init__()
                    # 初始化 BatchNorm2d 层
                    self.bn = torch.nn.BatchNorm2d(
                        num_features, 1e-5, affine=affine, track_running_stats=track_running_stats).float()

                def forward(self, x: torch.Tensor):
                    # 前向传播函数
                    o = self.bn(x)  # 执行批归一化
                    o = torch.nn.functional.relu(o)  # ReLU 激活函数
                    return o

            batch = 4  # 批大小
            c = 2  # 通道数
            hw = 3  # 图像高宽
            # 初始化参数和输入值
            x_init = torch.randn(batch, c, hw, hw, dtype=torch.float).cuda()  # 初始化输入张量
            grad = torch.randn(batch, c, hw, hw, dtype=torch.float).cuda()  # 初始化梯度张量

            training = False  # 是否训练模式
            affine = True  # 是否使用仿射变换
            track_running_stats = True  # 是否跟踪运行时统计信息

            module = torch.jit.script(MyBatchNorm(c, affine, track_running_stats)).cuda()  # 将自定义模块脚本化并移到 GPU
            ref_module = MyBatchNorm(c, affine, track_running_stats).cuda()  # 创建参考模块实例并移到 GPU
            module.eval()  # 将模块设置为评估模式
            ref_module.eval()  # 将参考模块设置为评估模式

            jit_module = torch.jit.script(module)  # 对模块进行脚本化

            # 加载状态字典到参考模块
            ref_module.load_state_dict(module.state_dict())

            x = x_init.detach().clone()  # 深拷贝初始化输入张量
            x.requires_grad_()  # 设置输入张量需要梯度
            x_ref = x_init.detach().clone()  # 深拷贝初始化输入张量作为参考
            x_ref.requires_grad_()  # 设置参考输入张量需要梯度

            # 测试符号微分
            # 运行三次前向传播和反向传播以触发自动微分图
            for i in range(0, 3):
                y = jit_module(x)  # 对脚本化模块执行前向传播
                y.backward(grad)  # 执行反向传播计算梯度
            x.grad.zero_()  # 梯度置零

            # 重置脚本化模块的运行均值和方差
            module.bn.running_mean.zero_()
            module.bn.running_var.fill_(1.0)
            # 重置参考模块的运行均值和方差
            ref_module.bn.running_mean.zero_()
            ref_module.bn.running_var.fill_(1.0)

            # 运行脚本化模块
            y = jit_module(x)
            y.backward(grad)
            # 运行参考计算
            y_ref = ref_module(x_ref)
            y_ref.backward(grad)

            # 断言输出结果一致
            self.assertEqual(y_ref, y)
            # 断言输入梯度一致
            self.assertEqual(x.grad, x_ref.grad)
            # 断言批归一化模块的运行均值一致
            self.assertEqual(module.bn.running_mean, ref_module.bn.running_mean)
            # 断言批归一化模块的运行方差一致
            self.assertEqual(module.bn.running_var, ref_module.bn.running_var)

    def test_zeros(self):
        # 定义一个继承自 ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                super().__init__()
                self.d = torch.device('cpu')  # 初始化设备为 CPU

            @torch.jit.script_method
            def create(self):
                # 创建一个全零张量
                return torch.zeros([1, 1, 2], dtype=torch.float, device=self.d, layout=torch.strided)

        r = M().create()  # 创建 M 类的实例并调用 create 方法
        # 断言返回的张量类型为 torch.float
        self.assertEqual(r.dtype, torch.float)
        # 断言返回的张量与预期的全零张量一致
        self.assertEqual(torch.zeros([1, 1, 2], dtype=torch.float), r)

        def fn():
            return torch.zeros((1, 2, 3))  # 创建一个形状为 (1, 2, 3) 的全零张量

        self.checkScript(fn, ())  # 检查 fn 函数是否可以脚本化
    def test_vararg_zeros(self):
        # 定义内部函数 foo，返回一个形状为 (3, 4, 5) 的零张量，数据类型为 torch.int
        def foo():
            return torch.zeros(3, 4, 5, dtype=torch.int)

        # 调用自定义的测试方法 checkScript，测试 foo 函数在无参数输入时的脚本化结果
        self.checkScript(foo, ())

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "the original version of test_rand")
    def test_rand(self):
        # 定义内部函数 test_rand，生成一个形状为 [3, 4] 的随机张量 a，并返回 a + 1.0 - a
        def test_rand():
            a = torch.rand([3, 4])
            return a + 1.0 - a

        # 调用自定义的测试方法 checkScript，测试 test_rand 函数在无参数输入时的脚本化结果
        self.checkScript(test_rand, ())
        # 将 test_rand 函数转换为脚本函数 fn
        fn = torch.jit.script(test_rand)
        # 执行脚本函数 fn
        out = fn()
        # 断言输出张量的数据类型为默认数据类型
        self.assertEqual(out.dtype, torch.get_default_dtype())
        # 获取脚本函数的计算图
        g = fn.graph_for()
        # 如果不是简单模式，通过文件检查工具验证形状分析是否正确设置类型
        if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
            FileCheck().check("Double(*, *, requires_grad=0, device=cpu)") \
                       .check_not("Float(*, *, requires_grad=0, device=cpu)").run(g)

        # 定义脚本函数 randint，生成一个形状为 [1, 2]，取值范围为 [0, 5) 的随机整数张量
        @torch.jit.script
        def randint():
            return torch.randint(0, 5, [1, 2])
        # 执行脚本函数 randint
        out = randint()
        # 断言输出张量的数据类型为 torch.int64
        self.assertEqual(out.dtype, torch.int64)
        # 如果不是简单模式，通过文件检查工具验证形状分析是否正确设置类型
        if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
            FileCheck().check("Long(*, *, requires_grad=0, device=cpu)") \
                       .check_not("Float(*, *, requires_grad=0, device=cpu)") \
                       .check_not("Double(*, *, requires_grad=0, device=cpu)") \
                       .run(randint.graph_for())

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "skip if profiling isn't enabled")
    def test_autodiff_complex(self):
        # 定义函数 foo，接受三个参数 x、y、W，分别为 torch.Tensor 类型，返回 torch.complex(x, y) 与 W.cfloat() 的矩阵乘积的指数
        def foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
            return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

        # 定义脚本函数 jitted_foo，与 foo 函数相同的功能
        @torch.jit.script
        def jitted_foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
            return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

        # 生成在 CUDA 设备上的随机张量 x、y、W
        x = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
        y = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
        W = torch.randn(16, 1, dtype=torch.float32, device='cuda:0', requires_grad=True)
        W.data /= 4

        # 启用用于性能分析测试的性能模式
        with enable_profiling_mode_for_profiling_tests():
            # 循环执行四次
            for i in range(4):
                # 断言 foo 和 jitted_foo 函数的梯度函数属性是否同时为 None
                self.assertTrue((foo(x, y, W).grad_fn is None) == (jitted_foo(x, y, W).grad_fn is None))
    def test_linear_grad(self):
        with enable_profiling_mode_for_profiling_tests():
            # 定义一个函数 t，用于执行 torch.nn.functional.linear 操作
            def t(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]):
                return torch.nn.functional.linear(x, w, b)

            # 初始化输入 x, w, b 和梯度 grad
            x_init = torch.randn(4, 2)
            w_init = torch.randn(3, 2)
            b_init = torch.randn(3)
            grad = torch.randn(4, 3)

            with disable_autodiff_subgraph_inlining():
                # 使用 torch.jit.script 将函数 t 转换为脚本模块 jit_t
                jit_t = torch.jit.script(t)

                # 分别对输入 x, w, b 进行处理，使其可以计算梯度
                x = x_init.detach().requires_grad_()
                w = w_init.detach().requires_grad_()
                b = b_init.detach().requires_grad_()
                x_ref = x_init.detach().requires_grad_()
                w_ref = w_init.detach().requires_grad_()
                b_ref = b_init.detach().requires_grad_()

                # 运行脚本模块 jit_t，并进行反向传播
                jit_o = jit_t(x, w, b)
                jit_o.backward(grad)
                jit_o = jit_t(x, w, b)
                jit_o.backward(grad)

                # 清空梯度
                x.grad.zero_()
                w.grad.zero_()
                b.grad.zero_()

                # 再次运行脚本模块 jit_t，并进行反向传播
                jit_o = jit_t(x, w, b)
                jit_o.backward(grad)

                # 使用非脚本模块执行函数 t，并进行反向传播
                o = t(x_ref, w_ref, b_ref)
                o.backward(grad)

                # 断言两种方式得到的输出结果相等
                self.assertEqual(jit_o, o)
                # 断言梯度计算的结果相等
                self.assertEqual(x.grad, x_ref.grad)
                self.assertEqual(w.grad, w_ref.grad)
                self.assertEqual(b.grad, b_ref.grad)

                # 清空梯度
                x.grad.zero_()
                w.grad.zero_()
                x_ref.grad.zero_()
                w_ref.grad.zero_()

                # 使用脚本模块 jit_t，并将 b 参数设为 None，进行反向传播
                jit_o = jit_t(x, w, None)
                jit_o.backward(grad)

                # 使用非脚本模块执行函数 t，并将 b 参数设为 None，进行反向传播
                o = t(x_ref, w_ref, None)
                o.backward(grad)

                # 断言两种方式得到的输出结果相等
                self.assertEqual(jit_o, o)
                # 断言梯度计算的结果相等
                self.assertEqual(x.grad, x_ref.grad)
                self.assertEqual(w.grad, w_ref.grad)

    @skipIfTorchDynamo("TorchDynamo doesn't support profile")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "the profiling version of test_rand")
    def test_rand_profiling(self):
        def test_rand():
            # 生成一个形状为 [3, 4] 的随机张量
            a = torch.rand([3, 4])
            # 对该张量执行加法和减法操作
            return a + 1.0 - a

        # 用于测试形状分析是否正确设置类型
        with enable_profiling_mode_for_profiling_tests():
            with num_profiled_runs(1):
                # 对 test_rand 函数进行脚本化编译
                fn = torch.jit.script(test_rand)
                # 执行脚本化的函数
                out = fn()
                # 获取最近一次优化执行的图形表示
                graph_str = torch.jit.last_executed_optimized_graph()
                # 断言输出张量的数据类型为 torch.float
                self.assertEqual(out.dtype, torch.float)
                # 运行 FileCheck 工具来验证优化后的图形表示中的特定内容
                FileCheck().check("Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)") \
                           .check_not("Double(3, 4, strides=[4, 1], requires_grad=0, device=cpu)").run(graph_str)

            # 以下代码被注释掉，不参与当前测试

        @torch.jit.script
        def randint():
            # 生成一个形状为 [1, 2] 的随机整数张量
            return torch.randint(0, 5, [1, 2])

        with enable_profiling_mode_for_profiling_tests():
            with num_profiled_runs(1):
                # 调用 randint 函数生成随机整数张量
                out = randint()
                # 获取最近一次优化执行的图形表示
                graph_str = torch.jit.last_executed_optimized_graph()
                # 断言输出张量的数据类型为 torch.int64
                self.assertEqual(out.dtype, torch.int64)
                # 运行 FileCheck 工具来验证优化后的图形表示中的特定内容
                FileCheck().check("profiled_type=Long(1, 2, strides=[2, 1], requires_grad=0, device=cpu)").run(graph_str)


    def test_erase_number_types(self):
        def func(a):
            # 创建一个整数常量表达式 b = 7 + 1 + 3
            b = 7 + 1 + 3
            # 计算 c = a + b
            c = a + b
            # 进行 c += b 的操作
            c += b
            # 返回计算结果 c
            return c

        # 获取 func 函数的图形表示
        graph = torch.jit.script(func).graph
        # 使用 FileCheck 工具来验证图形中的特定内容
        FileCheck().check("int = prim::Constant").check("aten::add_").run(str(graph))
        # 运行 erase_number_types 优化传递
        self.run_pass("erase_number_types", graph)
        # 验证图形中不再存在整数常量的表示
        FileCheck().check_not("int = prim::Constant").run(str(graph))


    def test_refine_tuple_types(self):
        # 注释：TupleConstruct 输出类型在此处不正确。
        graph_str = """
        graph(%a : Float(123), %b : Float(4, 5, 6)):
          %c : (Tensor, Tensor) = prim::TupleConstruct(%a, %b)
          return (%c)
        """
        # 解析给定的图形表示字符串
        graph = parse_ir(graph_str)
        # 执行 Torch 中的 Tuple 类型细化传递
        torch._C._jit_pass_refine_tuple_types(graph)

        # 注释：经过此传递后，输出类型应该已经更新。
        self.assertTrue('(Float(123), Float(4, 5, 6))' in str(graph.findNode('prim::TupleConstruct').output()))

    # TODO(henrytu): Add test for RefineTypes for NamedTuple when it's supported by IR parser.
    # 定义测试函数，用于测试在移除 Dropout 操作时的行为
    def test_remove_dropout(self):
        # 定义权重形状和输入形状
        weight_0_shape = (20, 5)
        weight_1_shape = (20, 20)
        input_shape = (10, 5)

        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个权重参数
                self.weight_0 = torch.nn.Parameter(torch.rand(weight_0_shape))
                self.weight_1 = torch.nn.Parameter(torch.rand(weight_1_shape))

            def forward(self, x):
                # 执行第一个线性变换
                o = F.linear(x, self.weight_0)
                # 在训练模式下对 o 执行 Dropout 操作
                o = F.dropout(o, training=self.training)
                # 执行第二个线性变换
                o = F.linear(o, self.weight_1)
                return o

        # 创建随机输入数据
        data = torch.rand(input_shape)
        # 实例化神经网络模型
        m = M()
        # 将模型转换为 Torch Script 形式
        m = torch.jit.script(m)
        
        # 测试是否会抛出异常，要求在训练模式下移除 Dropout 操作不被支持
        with self.assertRaisesRegex(RuntimeError, r'Dropout removal module in training mode is not yet supported'):
            torch._C._jit_pass_remove_dropout(m._c)
        
        # 将模型切换为评估模式
        m.eval()
        # 记录模型在给定输入数据上的输出
        ref_res = m(data)
        
        # 内联 Torch Script 模块，以避免 Function 实例化问题
        from torch.jit._recursive import wrap_cpp_module
        m = wrap_cpp_module(torch._C._freeze_module(m._c))
        # 移除模型中的 Dropout 操作
        torch._C._jit_pass_remove_dropout(m._c)
        # 计算移除 Dropout 操作后的输出结果
        res = m(data)
        
        # 检查模型的图中是否不包含 'aten::dropout' 操作
        FileCheck().check_not("aten::dropout").run(str(m.graph))
        # 断言移除 Dropout 操作前后的输出结果近似相等
        torch.testing.assert_close(ref_res, res, rtol=1e-2, atol=1e-3)

    # 定义测试函数，用于测试在 unfold 操作中处理零维的情况
    def test_unfold_zero_dim(self):
        # 定义一个函数，使用 unfold 操作展开输入张量
        def fn(x):
            return x.unfold(0, 1, 1)

        # 将函数 fn 转换为 Torch Script 形式，并获取其计算图
        graph = torch.jit.script(fn).graph
        # 执行完整的形状分析，为图添加形状信息
        torch._C._jit_pass_complete_shape_analysis(graph, (torch.tensor(0.39),), False)
        # 获取使用 fn 函数处理特定输入后的输出维度
        out_dims = fn(torch.tensor(0.3923)).ndim
        # 断言 unfold 操作节点的输出维度与预期相符
        self.assertEqual(graph.findNode("aten::unfold").output().type().dim(), out_dims)

    # 定义测试函数，用于测试批处理中的矩阵乘法操作
    def test_mm_batching(self):
        # 在性能分析测试中启用性能分析模式
        with enable_profiling_mode_for_profiling_tests():
            # 将 LSTMCellS 类转换为 Torch Script 形式
            lstm_cell = torch.jit.script(LSTMCellS)

            # 定义一个 LSTM 前向计算函数
            def lstm(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
                # 对输入序列执行 LSTM 单元操作
                for i in range(x.size(0)):
                    hx, cx = lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
                return hx

            # 将 lstm 函数转换为 Torch Script 形式
            slstm = torch.jit.script(lstm)

            # 获取用于 LSTM 测试的输入数据
            inputs = get_lstm_inputs('cpu', training=True, seq_length=10)
            # 运行 slstm 模型，启用性能分析和回放模式，并对其结果进行梯度反向传播
            slstm(*inputs, profile_and_replay=True).sum().backward(retain_graph=True)
            
            # 根据 GRAPH_EXECUTOR 类型执行不同的后向图形操作
            if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
                slstm(*inputs, profile_and_replay=True).sum().backward()

            # 获取 slstm 模型的前向计算图
            fw_graph = slstm.graph_for(*inputs)
            # 根据 GRAPH_EXECUTOR 类型获取反向计算图
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                bw_graph = backward_graph(slstm, diff_graph_idx=0)
                # 断言前向图中包含批量矩阵乘法操作
                self.assertTrue('prim::MMBatchSide' in str(fw_graph))
                # 断言反向图中包含树形矩阵减少操作
                self.assertTrue('prim::MMTreeReduce' in str(bw_graph))

            # 使用 slstm 模型计算输出结果
            sout = slstm(*inputs)
            # 使用原始 lstm 函数计算输出结果
            out = lstm(*inputs)
            # 断言使用 Torch 自动求导计算的 slstm 输出结果与原始输出结果相等
            self.assertEqual(sout, out)
            # 断言使用 Torch 自动求导计算的 slstm 输出结果的梯度与原始输出结果的梯度相等
            self.assertEqual(torch.autograd.grad(sout.sum(), inputs),
                             torch.autograd.grad(out.sum(), inputs))
    # 定义一个测试函数，用于测试循环展开
    def test_loop_unrolling(self):
        # 定义一个简单的函数 fn，计算从 0 到 x-1 的整数和的负数
        def fn(x):
            y = 0
            for i in range(int(x)):
                y -= i
            return y

        # 使用 Torch 的 JIT 编译 fn 函数，获取其计算图
        graph = torch.jit.script(fn).graph
        # 运行名为 'loop_unrolling' 的优化 pass
        self.run_pass('loop_unrolling', graph)
        # 设定循环展开的因子为 8
        unroll_factor = 8
        # 使用 FileCheck 检查计算图中的循环结构和子操作的数量
        FileCheck().check("prim::Loop").check_count("aten::sub", unroll_factor) \
            .check("prim::Loop").check("aten::sub").run(str(graph))
        # 检查 JIT 编译后的 fn 函数在输入为 10 时的执行结果
        self.checkScript(fn, (torch.tensor(10),))

    # 定义一个测试函数，用于测试带有常量循环的循环展开
    def test_loop_unrolling_const(self):
        # 定义一个简单的函数 fn，计算一个固定次数的减法操作
        def fn():
            y = 0
            for _ in range(10):
                y -= 1
            return y
        
        # 定义另一个函数 fn2，计算从 0 到 9 的整数和的负数
        def fn2():
            y = 0
            for i in range(10):
                y -= i
            return y
        
        # 定义一个检查函数，用于检查传入的函数 fn 在 JIT 编译后是否完全展开
        def check(fn, name):
            graph = torch.jit.script(fn).graph
            self.run_pass('loop_unrolling', graph)
            # 使用 FileCheck 检查是否完全展开循环
            FileCheck().check_not("prim::Loop'").run(str(graph))
            # 检查 JIT 编译后的 fn 函数在没有输入参数的情况下的执行结果
            self.checkScript(fn, ())

        # 对 fn 和 fn2 分别进行检查
        check(fn, 'add_const')
        check(fn2, 'add_iter')

    # 定义一个测试函数，用于测试嵌套循环的循环展开
    def test_loop_unrolling_nested(self):
        # 定义一个函数 fn，计算一个嵌套循环的负数和
        def fn(x):
            y = 0
            for _ in range(10):
                for j in range(int(x)):
                    y -= j
            return y
        
        # 使用 Torch 的 JIT 编译 fn 函数，获取其计算图
        graph = torch.jit.script(fn).graph
        # 运行名为 'loop_unrolling' 的优化 pass
        self.run_pass('loop_unrolling', graph)
        # 设定内部循环的展开因子为 8
        unroll_factor = 8
        # 使用 FileCheck 检查计算图中的循环结构和子操作的数量
        FileCheck().check("prim::Loop").check("prim::Loop").check_count('aten::sub', unroll_factor) \
            .check("prim::Loop").check("aten::sub").run(str(graph))
        # 检查 JIT 编译后的 fn 函数在输入为 10 时的执行结果
        self.checkScript(fn, (torch.tensor(10),))

    # 定义一个测试函数，用于测试循环展开在负数输入情况下的表现
    def test_loop_unroll_unused_counter(self):
        # 定义一个函数 fn，计算一个带有未使用计数器的减法操作
        def fn(x):
            y = 0
            for _ in range(int(x)):
                y -= 1
            return y
        
        # 使用 Torch 的 JIT 编译 fn 函数，获取其计算图
        graph = torch.jit.script(fn).graph
        # 运行名为 'loop_unrolling' 的优化 pass
        self.run_pass('loop_unrolling', graph)
        # 使用 FileCheck 检查计算图中是否存在循环和不必要的加法操作
        FileCheck().check("prim::Loop").check_not("aten::add").check("return") \
            .run(str(graph))

    # 定义一个测试函数，用于测试循环展开在负数输入情况下的表现
    def test_loop_unroll_negative(self):
        # 定义一个函数 fn，计算一个正数或负数的加法操作
        def fn(x):
            y = 0
            for _ in range(int(x)):
                y += 1
            return y
        
        # 对一系列输入值进行测试
        self.checkScript(fn, (torch.tensor(-20),))
        self.checkScript(fn, (torch.tensor(-2),))
        self.checkScript(fn, (torch.tensor(-1),))
        self.checkScript(fn, (torch.tensor(0),))
        self.checkScript(fn, (torch.tensor(1),))
        self.checkScript(fn, (torch.tensor(2),))

    # 定义一个测试函数，用于测试 torch.where 的使用
    def test_where(self):
        # 定义一个函数 fn，使用 torch.where 实现条件选择
        def fn(x, y):
            return torch.where(x > 0.0, x, y)
        
        # 检查 JIT 编译后的 fn 函数在给定输入时的执行结果
        self.checkScript(fn, (torch.randn(3, 2, dtype=torch.float), torch.ones(3, 2, dtype=torch.float)))

    # 定义一个测试函数，用于测试 torch.Tensor.where 方法的使用
    def test_where_method(self):
        # 定义一个函数 fn，使用 torch.Tensor.where 方法实现条件选择
        def fn(x, y):
            return x.where(x > 0.0, y)
        
        # 检查 JIT 编译后的 fn 函数在给定输入时的执行结果
        self.checkScript(fn, (torch.randn(3, 2, dtype=torch.float), torch.ones(3, 2, dtype=torch.float)))
    def test_union_to_number(self):
        # 定义一个 Torch Script 函数 fn，接受两个参数 x 和 y，类型可以是 int、complex 或 float 中的任意一种
        @torch.jit.script
        def fn(x: Union[int, complex, float], y: Union[int, complex, float]):
            # 返回 x 和 y 的和
            return x + y
        # 使用 FileCheck 检查 fn 的图形表示中是否包含指定的字符串": Scalar):"
        FileCheck().check(": Scalar):").run(fn.graph)

    def test_reassign_module_lhs(self):
        # 使用断言检查是否抛出 RuntimeError，且错误信息包含 'Cannot re-assign \'self\'' 字符串
        with self.assertRaisesRegex(RuntimeError, 'Cannot re-assign \'self\''):
            # 定义一个 Torch Script 模块 ReassignSelfLHS
            class ReassignSelfLHS(torch.jit.ScriptModule):
                # 定义 forward 方法
                @torch.jit.script_method
                def forward(self, x):
                    # 循环20次，尝试将 self 重新赋值为 x
                    for _ in range(20):
                        self = x
                    # 返回最后的 self
                    return self

            # 实例化 ReassignSelfLHS 类
            ReassignSelfLHS()

    def test_reassign_module_rhs(self):
        # 使用断言检查是否抛出 RuntimeError，且错误信息包含 'Cannot re-assign \'x\' to a value of type module' 字符串
        with self.assertRaisesRegex(RuntimeError, 'Cannot re-assign \'x\' to a value of type module'):
            # 定义一个 Torch Script 模块 ReassignSelfRHS
            class ReassignSelfRHS(torch.jit.ScriptModule):
                # 定义 forward 方法
                @torch.jit.script_method
                def forward(self, x):
                    # 循环20次，尝试将 x 重新赋值为 self
                    for _ in range(20):
                        x = self
                    # 返回 self
                    return self

            # 实例化 ReassignSelfRHS 类
            ReassignSelfRHS()

    def test_unknown_builtin(self):
        # 使用断言检查是否抛出 RuntimeError，且错误信息包含 'object has no attribute or method' 字符串
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute or method'):
            # 定义一个 Torch Script 函数 unknown_builtin
            @torch.jit.script
            def unknown_builtin(x):
                # 尝试调用 x 的不存在的方法 splork(3)
                return x.splork(3)

    def test_return_tuple(self):
        # 定义一个普通 Python 函数 return_tuple
        def return_tuple(x):
            # 创建一个元组 a，包含两个 x
            a = (x, x)
            # 返回元组 a 和 x 的元组
            return a, x
        # 使用 self.checkScript 方法检查 return_tuple 函数的 Torch Script 版本
        self.checkScript(return_tuple, (torch.rand(4),))

    def test_add_tuple_optional(self):
        # 定义一个函数 foo，参数为一个包含 torch.Tensor 和可选 torch.Tensor 的元组，并返回一个可选的 torch.Tensor
        def foo(input: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            # 对第一个元素 input[0] 加 1
            changed_input = input[0] + 1
            # 构建一个新的元组 value，将 changed_input 放在首位
            value: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]] = (changed_input,) + input[1:]
            # 返回 value 的第三个元素
            return value[2]
        # 定义输入 inp，包含一个 torch.Tensor 和两个 None
        inp: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]] = (torch.rand(4), None, None)
        # 使用 self.checkScript 方法检查 foo 函数的 Torch Script 版本
        self.checkScript(foo, (inp,))

    def test_add_tuple_non_optional(self):
        # 定义一个函数 foo，参数为一个包含三个 torch.Tensor 的元组，并返回一个 torch.Tensor
        def foo(input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
            # 对第一个元素 input[0] 加 1
            changed_input = input[0] + 1
            # 构建一个新的元组 value，将 changed_input 放在首位
            value: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (changed_input,) + input[1:]
            # 返回 value 的第三个元素的总和加 4
            return torch.sum(value[2]) + 4
        # 定义输入 inp，包含三个随机生成的 torch.Tensor
        inp: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (torch.rand(4), torch.rand(4), torch.rand(4))
        # 使用 self.checkScript 方法检查 foo 函数的 Torch Script 版本
        self.checkScript(foo, (inp,))

    def test_add_tuple_different_types(self):
        # 定义一个函数 foo，参数为两个不同类型的元组 a 和 b，返回一个整数
        def foo(a: Tuple[int, float], b: Tuple[int]) -> int:
            # 将两个元组 a 和 b 连接成新的元组 c
            c: Tuple[int, float, int] = a + b
            # 将新的元组 c 和元组 b 连接成 d
            d: Tuple[int, float, int, int] = c + b
            # 返回 d 的第四个元素加 1
            return d[3] + 1
        # 定义输入 a 和 b，分别为 (1, 2.0) 和 (3,)
        a = (1, 2.0)
        b = (3,)
        # 使用 self.checkScript 方法检查 foo 函数的 Torch Script 版本
        self.checkScript(foo, (a, b))
    def test_add_tuple_same_types(self):
        # 定义一个函数 foo，接受两个参数 a 和 b，都是元组类型 (int, int) 和 (int, int, int)，返回值是 int 类型
        def foo(a: Tuple[int, int], b: Tuple[int, int, int]) -> int:
            # 将 a 和 b 相加得到 c，类型为 Tuple[int, int, int, int, int]
            c: Tuple[int, int, int, int, int] = a + b
            # 将 c 和 b 再次相加得到 d，类型为 Tuple[int, int, int, int, int, int, int, int]
            d: Tuple[int, int, int, int, int, int, int, int] = c + b
            # 返回 d 中第 7 个元素减去 2
            return d[6] - 2
        # 定义元组 a 和 b
        a = (1, 2)
        b = (3, 4, 5)
        # 调用 self.checkScript 方法，将 foo 函数和元组 (a, b) 作为参数传递
        self.checkScript(foo, (a, b))

    def test_method_no_self(self):
        # 使用 self.assertRaisesRegex 检查是否会抛出 RuntimeError 异常，异常信息包含 'methods must have a self argument'
        with self.assertRaisesRegex(RuntimeError, 'methods must have a self argument'):
            # 定义一个类 MethodNoSelf，继承自 torch.jit.ScriptModule
            class MethodNoSelf(torch.jit.ScriptModule):
                # 使用 @torch.jit.script_method 装饰器标记 forward 方法，忽略 B902 类型的错误
                @torch.jit.script_method  # noqa: B902
                # 定义 forward 方法，返回一个 3x4 的全零张量
                def forward():  # noqa: B902
                    return torch.zeros(3, 4)

            # 实例化 MethodNoSelf 类，期望抛出 RuntimeError 异常
            MethodNoSelf()

    def test_return_stmt_not_at_end(self):
        # 定义一个函数 return_stmt，接受一个参数 x，根据 x 的值进行条件判断并返回不同的值
        def return_stmt(x):
            if bool(x > 3):
                return x + 3
            else:
                return x
        # 调用 self.checkScript 方法，将 return_stmt 函数和 torch.rand(1) 的结果作为参数传递
        self.checkScript(return_stmt, (torch.rand(1),))

    def test_for_in_range(self):
        # 定义一个函数 fn，计算从 0 到 99 的所有整数之和
        def fn():
            c = 0
            for i in range(100):
                c += i
            return c
        # 调用 self.checkScript 方法，将 fn 函数作为参数传递
        self.checkScript(fn, ())

    def test_for_in_range_dynamic(self):
        # 定义一个函数 fn，计算累加和，其中每次累加的次数由外层循环的索引决定
        def fn():
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c
        # 调用 self.checkScript 方法，将 fn 函数和空元组作为参数传递，并禁用优化选项
        self.checkScript(fn, (), optimize=False)

    def test_for_in_range_ast(self):
        # 定义一个函数 test_script_for_in_range_ast，计算嵌套循环累加和
        def test_script_for_in_range_ast():
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c

        # 调用 self.checkScript 方法，将 test_script_for_in_range_ast 函数作为参数传递
        self.checkScript(test_script_for_in_range_ast, ())

    def test_for_in_range_if_ast(self):
        # 使用 @torch.jit.script 装饰器定义一个静态图函数 test_script_for_in_range_if_ast，接受一个参数 x
        @torch.jit.script
        def test_script_for_in_range_if_ast(x):
            output = x
            for i in range(20):
                if i == 0:
                    output = x.unsqueeze(0)
                else:
                    output = torch.cat((output, x.unsqueeze(0)), dim=0)
            return output
        # 生成一个 torch.int64 类型的输入变量列表
        inputs = self._make_scalar_vars([0], torch.int64)

        # 断言调用 test_script_for_in_range_if_ast 函数后的输出张量 shape 的第一个维度为 20
        self.assertEqual(test_script_for_in_range_if_ast(*inputs).shape[0], 20)

    def test_for_in_range_start_end(self):
        # 定义一个函数 fn，计算从 7 到 99 的所有整数之和
        def fn():
            x = 0
            for i in range(7, 100):
                x += i
            return x
        # 调用 self.checkScript 方法，将 fn 函数作为参数传递
        self.checkScript(fn, ())

    def test_for_in_range_start_end_step(self):
        # 定义一个函数 fn，根据参数 start、end 和 step 计算整数之和
        def fn(start, end, step):
            # type: (int, int, int) -> int
            x = 0
            for i in range(start, end, step):
                x += i
            return x

        # 分别调用多组参数测试 self.checkScript 方法
        self.checkScript(fn, (7, 100, 7))
        self.checkScript(fn, (7, 100, -7))
        self.checkScript(fn, (2, -11, -3))
        self.checkScript(fn, (2, -11, 3))
        self.checkScript(fn, (2, 10, 3))
        self.checkScript(fn, (-2, -10, -10))
    def test_for_in_range_zero_step(self):
        @torch.jit.script
        def fn():
            # 初始化变量 x 为 0
            x = 0
            # 使用 range 函数进行循环，从 2 开始到 -11（不包括），步长为 0，这会引发错误
            for i in range(2, -11, 0):
                # 对 x 进行累加操作
                x += i
            return x

        # 验证在运行时是否捕获到“must not be zero”错误
        with self.assertRaisesRegex(RuntimeError, "must not be zero"):
            fn()

    def test_range_args(self):
        # 验证在使用 range 函数时没有传递任何参数的错误
        with self.assertRaisesRegex(RuntimeError, r'range expected at least 1 arguments, got 0'):
            @torch.jit.script
            def range_no_arg(x):
                # 尝试在循环中使用空的 range 函数
                for _ in range():
                    x += 1
                return x
        # 验证在使用 range 函数时传递浮点数参数的错误
        with self.assertRaisesRegex(RuntimeError, r'found float'):
            @torch.jit.script
            def range_non_float():
                # 尝试在循环中使用浮点数作为 range 的参数
                for i in range(.5):
                    print(i)

    def test_parse_empty_tuple_annotation(self):
        # 使用 Torch 的 CompilationUnit 来定义一个函数，该函数使用空元组作为参数和返回类型
        cu = torch.jit.CompilationUnit('''
            def foo(x : Tuple[()]) -> Tuple[()]:
                return x
        ''')

        # 获取生成的函数 foo 的代码，并使用 FileCheck 验证其是否正确包含了 "Tuple[()]" 注释
        foo_code = cu.find_function('foo').code
        FileCheck().check("Tuple[()]").check("Tuple[()]").run(foo_code)

    def test_parse_empty_tuple_annotation_element_error(self):
        # 验证在元组类型注释中不应该包含任何元素的错误
        with self.assertRaisesRegex(
                RuntimeError, 'Tuple literal in Tuple type annotation must not have any elements'):
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tuple[(int,)]) -> Tuple[(int,)]:
                    return x
            ''')

    def test_parse_none_type_annotation(self):
        # 使用 Torch 的 CompilationUnit 定义一个函数，该函数使用 NoneType 作为参数和返回类型
        cu = torch.jit.CompilationUnit('''
            def foo(x : NoneType) -> NoneType:
                return x
        ''')

        # 获取生成的函数 foo 的代码，并使用 FileCheck 验证其是否正确包含了 ": NoneType" 和 "-> NoneType" 注释
        foo_code = cu.find_function('foo').code
        FileCheck().check(": NoneType").check("-> NoneType").run(foo_code)

    def test_empty_tuple_str(self):
        # 创建一个空元组类型对象，并验证其 annotation_str 属性是否正确返回 "Tuple[()]"
        empty_tuple_type = torch._C.TupleType([])
        g = {'Tuple' : typing.Tuple}
        python_type = eval(empty_tuple_type.annotation_str, g)
        assert python_type is typing.Tuple[()]

    def test_tuple_str(self):
        # 创建包含一个和两个字符串类型的元组类型对象，并验证其 annotation_str 属性的返回值
        tuple1_type = torch._C.TupleType([torch._C.StringType.get()])
        self.assertEqual(tuple1_type.annotation_str, "Tuple[str]")
        tuple2_type = torch._C.TupleType([torch._C.StringType.get(), torch._C.StringType.get()])
        self.assertEqual(tuple2_type.annotation_str, "Tuple[str, str]")

    def test_dict_str(self):
        # 创建一个字典类型对象，并验证其 annotation_str 属性的返回值
        dict_type = torch._C.DictType(torch._C.StringType.get(), torch._C.StringType.get())
        self.assertEqual(dict_type.annotation_str, "Dict[str, str]")

    def test_none_type_str(self):
        # 创建一个 NoneType 类型对象，并验证其 annotation_str 属性的返回值
        none_type = torch._C.NoneType.get()
        g = {'NoneType' : type(None)}
        python_type = eval(none_type.annotation_str, g)
        assert python_type is type(None)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_zip_enumerate_modulelist(self):
        # 定义一个内部子类 Sub，继承自 torch.nn.Module，重写 forward 方法
        class Sub(torch.nn.Module):
            def forward(self, thing):
                return thing - 2

        # 定义一个内部子类 Double，继承自 torch.nn.Module，重写 forward 方法
        class Double(torch.nn.Module):
            def forward(self, thing):
                return thing * 2

        # 定义一个内部类 ZipModLists，继承自 torch.nn.Module
        # mods 和 mods2 是输入的 ModuleList 列表
        class ZipModLists(torch.nn.Module):
            def __init__(self, mods, mods2):
                super().__init__()
                self.mods = mods
                self.mods2 = mods2

            # 实现 forward 方法
            def forward(self, x):
                iter = 0
                # 使用 zip 迭代 mods 和 mods2 的元素
                for mod1, mod2 in zip(self.mods, self.mods2):
                    # 先将输入 x 经过 mod1 处理，再经过 mod2 处理
                    x = mod2(mod1(x))
                    iter += 1
                return x, iter

        # 定义一个内部类 ZipWithValues，继承自 torch.nn.Module
        # mods 和 mods2 是输入的 ModuleList 列表
        class ZipWithValues(torch.nn.Module):
            __constants__ = ['tup_larger', 'tup_smaller']

            def __init__(self, mods, mods2):
                super().__init__()
                self.mods = mods
                self.mods2 = mods2
                # tup_larger 是一个列表，包含 mods2 的长度加一的整数
                self.tup_larger = list(range(len(mods2) + 1))
                # tup_smaller 是一个列表，包含 mods2 的长度加一和 1 中的较大值的整数
                self.tup_smaller = list(range(max(len(mods2) + 1, 1)))

            # 实现 forward 方法
            def forward(self, x):
                iter = 0
                x2 = x
                # 使用 zip 迭代 tup_larger、mods 和 mods2 的元素
                for val, mod1, mod2 in zip(self.tup_larger, self.mods, self.mods2):
                    # 先将输入 x 经过 mod1 处理，再经过 mod2 处理，然后加上 val
                    x = mod2(mod1(x)) + val
                    iter += 1
                # 使用 zip 迭代 tup_smaller、mods 和 mods2 的元素
                for val, mod1, mod2 in zip(self.tup_smaller, self.mods, self.mods2):
                    # 先将输入 x2 经过 mod1 处理，再经过 mod2 处理，然后加上 val
                    x2 = mod2(mod1(x2)) + val
                    iter += 1
                return x, iter

        # mods 是一个元组，包含三个 ModuleList 实例，每个实例包含不同数量的 Double 和 Sub 类
        mods = nn.ModuleList([Double()]), nn.ModuleList([Double(), Sub(), Sub()]), nn.ModuleList([Sub(), Double()])
        # 使用嵌套循环遍历 mods 中的每个元素
        for i in range(len(mods)):
            for j in range(len(mods)):
                # 创建 ZipModLists 的实例 mod，使用 mods[i] 和 mods[j] 作为参数
                mod = ZipModLists(mods[i], mods[j])
                # 调用外部函数 checkModule 来验证 mod 的输出
                self.checkModule(mod, (torch.tensor(.5),))
                # 创建 ZipWithValues 的实例 mod2，使用 mods[i] 和 mods[j] 作为参数
                mod2 = ZipWithValues(mods[i], mods[j])
                # 调用外部函数 checkModule 来验证 mod2 的输出
                self.checkModule(mod2, (torch.tensor(.5),))
    # 定义一个测试方法，用于测试枚举和修改模块列表的功能
    def test_enumerate_modlist_range(self):
        # 定义一个简单的双倍计算模块
        class Double(torch.nn.Module):
            # 前向传播函数，对输入进行双倍计算
            def forward(self, thing):
                return thing * 2

        # 定义一个包含两个双倍计算模块的修改模块
        class Mod(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 使用ModuleList创建一个包含两个Double模块的列表
                self.mods = nn.ModuleList([Double(), Double()])

            # 前向传播函数
            def forward(self, x):
                # 复制输入张量x到x2
                x2 = x
                # 初始化迭代变量iter为0
                iter = 0
                # 遍历self.mods列表中的模块，同时枚举它们的索引和值
                for val, mod in enumerate(self.mods):
                    # 对x2应用当前模块mod，并乘以其索引val
                    x2 = mod(x2) * val
                    # 迭代器加1
                    iter += 1
                # 返回迭代次数iter，输入x和处理后的x2
                return iter, x, x2

        # 使用自定义的检查函数检查Mod模块
        self.checkModule(Mod(), (torch.tensor(.5),))

        # 变长输入和模块列表
        # 定义一个继承自Mod的类Mod2
        class Mod2(Mod):
            # 重载前向传播函数
            def forward(self, x):
                # 使用zip函数将输入x的范围和self.mods列表进行组合
                for val, mod in zip(range(int(x)), self.mods):
                    # 对x应用当前模块mod，并乘以其索引val
                    x = mod(x) * val
                # 返回处理后的x
                return x

        # 使用torch.jit.script尝试将Mod2模块转换为脚本时，期望抛出异常
        with self.assertRaisesRegex(Exception, "that does not have a statically determinable length"):
            torch.jit.script(Mod2())

        # 模块列表和变长输入
        # 定义一个继承自Mod的类Mod3
        class Mod3(Mod):
            # 重载前向传播函数
            def forward(self, x):
                # 使用zip函数将self.mods列表和输入x的范围进行组合
                for val, mod in zip(self.mods, range(int(x))):
                    # 对x应用当前模块mod，并乘以其索引val
                    x = mod(x) * val
                # 返回处理后的x
                return x

        # 使用torch.jit.script尝试将Mod3模块转换为脚本时，期望抛出异常
        with self.assertRaisesRegex(Exception, "that does not have a statically determinable length"):
            torch.jit.script(Mod3())
    def test_for_in_enumerate(self):
        def fn(x):
            # 定义一个函数 fn，接受一个列表参数 x，返回一个整数
            sum = 0
            # 初始化 sum 为 0
            for (i, v) in enumerate(x):
                # 使用 enumerate 遍历列表 x，i 是索引，v 是对应的元素值
                sum += i * v
                # 计算 sum，每次加上索引 i 乘以元素值 v 的乘积

            return sum
            # 返回计算结果 sum

        self.checkScript(fn, ([1, 2, 3, 4, 5],))
        # 使用 self.checkScript 方法测试函数 fn，传入参数为包含一个列表 [1, 2, 3, 4, 5] 的元组

        def fn_enumerate_start_arg(x):
            # 定义一个函数 fn_enumerate_start_arg，接受一个列表参数 x，返回一个整数
            sum = 0
            # 初始化 sum 为 0
            for (i, v) in enumerate(x, 1):
                # 使用 enumerate 遍历列表 x，从索引 1 开始，i 是索引，v 是对应的元素值
                sum += i * v
                # 计算 sum，每次加上索引 i 乘以元素值 v 的乘积

            return sum
            # 返回计算结果 sum

        self.checkScript(fn_enumerate_start_arg, ([1, 2, 3, 4, 5],))
        # 使用 self.checkScript 方法测试函数 fn_enumerate_start_arg，传入参数为包含一个列表 [1, 2, 3, 4, 5] 的元组

        def fn_enumerate_start_kwarg(x):
            # 定义一个函数 fn_enumerate_start_kwarg，接受一个列表参数 x，返回一个整数
            sum = 0
            # 初始化 sum 为 0
            for (i, v) in enumerate(x, start=1):
                # 使用 enumerate 遍历列表 x，从索引 1 开始，i 是索引，v 是对应的元素值
                sum += i * v
                # 计算 sum，每次加上索引 i 乘以元素值 v 的乘积

            return sum
            # 返回计算结果 sum

        self.checkScript(fn_enumerate_start_kwarg, ([1, 2, 3, 4, 5],))
        # 使用 self.checkScript 方法测试函数 fn_enumerate_start_kwarg，传入参数为包含一个列表 [1, 2, 3, 4, 5] 的元组

        def fn_nested_enumerate(x):
            # 定义一个函数 fn_nested_enumerate，接受一个列表参数 x，返回一个整数
            sum = 0
            # 初始化 sum 为 0
            for (i, (j, v)) in enumerate(enumerate(x)):
                # 使用 enumerate 遍历列表 x，并且再次使用 enumerate 遍历内部元素，i 是外部索引，(j, v) 是内部元组
                sum += i * j * v
                # 计算 sum，每次加上外部索引 i 乘以内部索引 j 乘以元素值 v 的乘积

            return sum
            # 返回计算结果 sum

        self.checkScript(fn_nested_enumerate, ([1, 2, 3, 4, 5],))
        # 使用 self.checkScript 方法测试函数 fn_nested_enumerate，传入参数为包含一个列表 [1, 2, 3, 4, 5] 的元组

        with self.assertRaisesRegex(RuntimeError, r'enumerate expected at least 1 arguments, got 0'):
            # 使用 self.assertRaisesRegex 捕获 RuntimeError 异常，确保异常消息包含 'enumerate expected at least 1 arguments, got 0'
            @torch.jit.script
            def enumerate_no_arg(x):
                # type: (List[int]) -> int
                sum = 0
                # 初始化 sum 为 0
                for _ in enumerate():
                    # 使用 enumerate 遍历空对象，将会抛出异常
                    sum += 1

                return sum
                # 返回计算结果 sum

        with self.assertRaisesRegex(RuntimeError, r'enumerate expected at most 2 arguments, got 3'):
            # 使用 self.assertRaisesRegex 捕获 RuntimeError 异常，确保异常消息包含 'enumerate expected at most 2 arguments, got 3'
            @torch.jit.script
            def enumerate_too_many_args(x):
                # type: (List[int]) -> int
                sum = 0
                # 初始化 sum 为 0
                for _ in enumerate(x, x, x):
                    # 使用 enumerate 同时传入三个参数，将会抛出异常
                    sum += 1

                return sum
                # 返回计算结果 sum
    def test_list_comprehension_modulelist(self):
        # 定义一个内部的 PyTorch 模块类 Inner，用于在 M 类中使用
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        # 定义 M 类，接受一个模块列表 mod_list 作为参数
        class M(torch.nn.Module):
            def __init__(self, mod_list):
                super().__init__()
                self.module_list = mod_list

            def forward(self, x):
                # 使用列表推导式生成一个列表 out，其中每个元素是调用 self.module_list 中模块的结果
                out = torch.jit.annotate(List[Tensor], [mod(x) for mod in self.module_list])
                return out

        # 创建 M 类的实例 mod，传入包含两个 Inner 模块的 ModuleList
        mod = M(nn.ModuleList([Inner(), Inner()]))
        # 调用 self.checkModule 函数验证模块的行为
        self.checkModule(mod, (torch.tensor(3),))

        # 创建一个空的 ModuleList 实例 mod
        mod = M(nn.ModuleList([]))
        # 对模型进行 Torch 脚本转换
        torch.jit.script(mod)

        # 定义 M2 类，继承自 M 类，接受一个模块列表 mod_list 作为参数
        class M2(M):
            def __init__(self, mod_list):
                super().__init__(mod_list)

            def forward(self, x):
                # 使用列表推导式生成一个列表 out，其中每个元素是调用 self.module_list 中模块的结果
                out = [mod(x) for mod in self.module_list]
                return out

        # 创建 M2 类的实例 mod，传入包含两个 Inner 模块的 ModuleList
        mod = M2(nn.ModuleList([Inner(), Inner()]))
        # 调用 self.checkModule 函数验证模块的行为
        self.checkModule(mod, (torch.tensor(3),))

        # 创建一个空的 ModuleList 实例 mod
        # 对模型进行 Torch 脚本转换，并检查返回结果是否为空列表
        self.assertEqual(torch.jit.script(mod)(torch.tensor(.5)), [])

        # 定义一个函数 bad_type_annotation，使用列表推导式尝试为 out 进行类型注解，期望得到异常
        def bad_type_annotation():
            out = torch.jit.annotate(int, [x for x in [1, 2, 3]])  # noqa: C416
            return out

        # 使用 assertRaisesRegex 验证 bad_type_annotation 函数调用时是否抛出预期的异常信息
        with self.assertRaisesRegex(Exception, "Expected an annotation"
                                    " of type List"):
            torch.jit.script(bad_type_annotation)

    def test_list_comprehension_variable_write(self):
        # 定义一个函数 foo，在其中使用列表推导式生成一个列表 x，同时不影响函数作用域中的变量 i
        def foo():
            i = 1
            x = [i if i != 5 else 3 for i in range(7)]  # noqa: C416
            return i, x

        # 调用 foo 函数，验证其返回结果与 Torch 脚本版本的 foo 函数返回结果一致
        self.assertEqual(foo(), torch.jit.script(foo)())
    def test_for_in_zip(self):
        def fn(x, y):
            # type: (List[int], List[int]) -> int
            # 初始化变量 sum 为 0
            sum = 0
            # 使用 zip 函数并行迭代 x 和 y 列表
            for (i, j) in zip(x, y):
                # 计算累加乘积 i * j
                sum += i * j

            # 返回累加结果
            return sum

        # 使用 self.checkScript 函数检查 fn 函数的执行结果
        self.checkScript(fn, ([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))

        def fn_multi_inputs(x, y, z):
            # type: (List[int], List[int], List[int]) -> int
            # 初始化变量 sum 为 0
            sum = 0
            # 使用 zip 函数并行迭代 x, y, z 三个列表
            for (i, j, k) in zip(x, y, z):
                # 计算累加乘积 i * j * k
                sum += i * j * k

            # 返回累加结果
            return sum

        # 使用 self.checkScript 函数检查 fn_multi_inputs 函数的执行结果
        self.checkScript(fn_multi_inputs, ([1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]))

        def fn_nested_zip(x, y, z):
            # type: (List[int], List[int], List[int]) -> int
            # 初始化变量 sum 为 0
            sum = 0
            # 使用 zip 函数并行迭代 y 和 z 列表，然后再与 x 列表并行迭代
            for (i, (j, k)) in zip(x, zip(y, z)):
                # 计算累加乘积 i * j * k
                sum += i * j * k

            # 返回累加结果
            return sum

        # 使用 self.checkScript 函数检查 fn_multi_inputs 函数的执行结果
        self.checkScript(fn_multi_inputs, ([1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]))

        # 使用 with 语句捕获 RuntimeError 异常，并验证其消息
        with self.assertRaisesRegex(RuntimeError, r'zip expected at least 1 arguments, got 0'):
            @torch.jit.script
            def zip_no_arg(x):
                # type: (List[int]) -> int
                # 初始化变量 sum 为 0
                sum = 0
                # 使用 zip 函数迭代空的迭代器，应该抛出异常
                for _ in zip():
                    sum += 1

                # 返回累加结果
                return sum

        # 使用 with 语句捕获 RuntimeError 异常，并验证其消息
        with self.assertRaisesRegex(RuntimeError, r'too many values to unpack: need 2 but found 3'):
            @torch.jit.script
            def fn_nested_zip_wrong_target_assign(x, y, z):
                # type: (List[int], List[int], List[int]) -> int
                # 初始化变量 sum 为 0
                sum = 0
                # 尝试在 zip 函数中使用三个参数，应该抛出异常
                for (i, (j, k)) in zip(x, y, z):
                    sum += i * j * k

                # 返回累加结果
                return sum

    def test_for_in_zip_enumerate(self):
        def fn_zip_enumerate(x, y):
            # type: (List[int], List[int]) -> int
            # 初始化变量 sum 为 0
            sum = 0
            # 使用 zip 函数并行迭代 x 和 y 列表，并与 range(0, 100) 并行迭代
            for (i, (j, v), k) in zip(x, enumerate(y), range(0, 100)):
                # 计算累加乘积 i * j * v * k
                sum += i * j * v * k

            # 返回累加结果
            return sum

        # 使用 self.checkScript 函数检查 fn_zip_enumerate 函数的执行结果
        self.checkScript(fn_zip_enumerate, ([1, 2, 3, 4], [2, 3, 4, 5]))

        def fn_enumerate_zip(x, y):
            # type: (List[int], List[int]) -> int
            # 初始化变量 sum 为 0
            sum = 0
            # 使用 enumerate 函数并行迭代 zip(x, y) 的结果
            for (i, (j, v)) in enumerate(zip(x, y)):
                # 计算累加乘积 i * j * v
                sum += i * j * v

            # 返回累加结果
            return sum

        # 使用 self.checkScript 函数检查 fn_enumerate_zip 函数的执行结果
        self.checkScript(fn_enumerate_zip, ([1, 2, 3, 4], [2, 3, 4, 5]))

    def test_for_in_tensors(self):
        def test_sizes(x):
            # 初始化变量 sumz 为 0
            sumz = 0
            # 使用 for 循环迭代 tensor x 中的元素
            for s in x:
                # 累加计数器
                sumz += 1
            return sumz

        # 使用 self.checkScript 函数检查 test_sizes 函数的执行结果
        self.checkScript(test_sizes, (torch.rand(5, 4, 3, 2, 1),))
        self.checkScript(test_sizes, (torch.rand(777),))
        self.checkScript(test_sizes, (torch.rand(0),))

    def test_for_in_tensors_rank0(self):
        # 使用 with 语句捕获 RuntimeError 异常，并验证其消息
        with self.assertRaisesRegex(RuntimeError, "of a 0-d tensor"):
            @torch.jit.script
            def test_sizes(x):
                # 初始化变量 sumz 为 0
                sumz = 0
                # 尝试使用 for 循环迭代标量 tensor x，应该抛出异常
                for s in x:
                    sumz += 1
                return sumz

            # 调用 test_sizes 函数，并传入标量 tensor，期望抛出异常
            test_sizes(torch.tensor(1))
    def test_for_in_tensors_fail_scalar(self):
        # 使用断言确保在运行时捕获特定的异常信息
        with self.assertRaisesRegex(RuntimeError, "'float' object is not iterable"):
            # 使用 Torch Script 注解来定义静态类型
            @torch.jit.script
            def test_sizes(x):
                # type: (float) -> int
                sumz = 0
                # 遍历输入的标量 x，这里会触发异常
                for s in x:
                    sumz += 1
                return sumz

            # 调用 test_sizes 函数，传入标量值 0.0
            test_sizes(0.0)

    def test_for_in_tensors_nested(self):
        # 定义一个函数 test_sizes，接受一个输入参数 x
        def test_sizes(x):
            sumz = 0
            # 嵌套循环，遍历 x 中的每个元素 n
            for n in x:
                # 对于每个 n，再次遍历其内部的元素 t
                for t in n:
                    sumz += 1
            return sumz

        # 使用 self.checkScript 方法验证 test_sizes 函数
        # 传入一个 5x4x3x2x1 的随机张量作为参数
        self.checkScript(test_sizes, (torch.rand(5, 4, 3, 2, 1),))

    # to avoid defining sum_list in multiple tests
    def get_sum_list_fn(self):
        # 定义一个内部函数 sum_list，接受一个列表参数 a
        def sum_list(a):
            # type: (List[int]) -> int
            sum = 0
            # 遍历列表 a 中的每个元素 i，累加到 sum 中
            for i in a:
                sum += i

            return sum

        # 返回内部函数 sum_list 作为结果
        return sum_list

    def test_sum_list_diff_elms(self):
        # 使用 self.checkScript 方法验证 self.get_sum_list_fn() 的输出
        # 传入包含 [1, 2, 3, 4, 5] 的元组作为参数
        self.checkScript(self.get_sum_list_fn(), ([1, 2, 3, 4, 5],))

    def test_sum_list_empty(self):
        # 使用 self.checkScript 方法验证 self.get_sum_list_fn() 的输出
        # 传入一个空列表作为参数
        self.checkScript(self.get_sum_list_fn(), ([],))

    def test_sum_list_one(self):
        # 使用 self.checkScript 方法验证 self.get_sum_list_fn() 的输出
        # 传入包含一个元素 [1] 的元组作为参数
        self.checkScript(self.get_sum_list_fn(), ([1],))

    def test_sum_list_literal(self):

        def sum_list():
            # type: () -> int
            sum = 0
            # 遍历字面量列表 [1, 2, 3, 4, 5] 中的每个元素 i，累加到 sum 中
            for i in [1, 2, 3, 4, 5]:
                sum += i

            return sum

        # 使用 self.checkScript 方法验证 sum_list 函数
        # 不传入任何参数
        self.checkScript(sum_list, ())

    def test_sum_list_wrong_type(self):

        # 使用断言确保在运行时捕获特定的异常信息
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):
            # 使用 Torch Script 注解来定义静态类型
            @torch.jit.script
            def sum_list(a):
                # type: (int) -> int
                sum = 0
                # 遍历输入的标量 a，这里会触发异常
                for i in a:  # noqa: T484
                    sum += i

                return sum

            # 调用 sum_list 函数，传入标量值 1
            sum_list(1)

    def test_list_iterables(self):
        # 使用断言确保在运行时捕获特定的异常信息
        with self.assertRaisesRegex(RuntimeError, 'List of iterables is not supported currently'):
            # 创建一个 Torch JIT CompilationUnit 对象 cu
            cu = torch.jit.CompilationUnit('''
            def list_iterables(x):
                # 循环遍历一个包含两个列表的列表 [2, 3, 4], [5, 6, 7]
                for i, j in [2, 3, 4], [5, 6, 7]:
                    x += i
                    x += j
                return x
            ''')

    def test_for_in_string(self):
        # 定义一个函数 test_strings，接受一个输入参数 x
        def test_strings(x):
            # type: (str) -> str
            reverse = ""
            # 遍历输入字符串 x 中的每个字符 c，倒序拼接到 reverse 变量中
            for c in x:
                reverse = c + reverse
            return reverse

        # 使用 self.checkScript 方法验证 test_strings 函数
        # 传入字符串 "hello" 作为参数
        self.checkScript(test_strings, ("hello",))
        # 再次使用 self.checkScript 方法验证 test_strings 函数
        # 传入空字符串作为参数
        self.checkScript(test_strings, ("",))

        def test_list_strings(x):
            # type: (List[str]) -> str
            result = ""
            # 遍历输入列表 x 中的每个子字符串 sub_str，依次拼接到 result 变量中
            for sub_str in x:
                result += sub_str
            return result

        # 使用 self.checkScript 方法验证 test_list_strings 函数
        # 传入包含两个字符串 "hello" 和 "world" 的列表作为参数
        self.checkScript(test_list_strings, (["hello", "world"],))
        # 再次使用 self.checkScript 方法验证 test_list_strings 函数
        # 传入包含多个字符串的列表，包括空字符串作为参数
        self.checkScript(test_list_strings, (["hello", " ", "world", ""],))
    def test_for_in_dict(self):
        def test_dicts(x):
            # type: (Dict[str, int]) -> int
            # 初始化求和变量
            sum = 0
            # 遍历字典中的键
            for key in x:
                # 将字典中每个键对应的值累加到求和变量
                sum += x[key]
            return sum

        # 使用给定的测试脚本检查 test_dicts 函数
        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

        def test_dict_keys_values(x):
            # type: (Dict[str, int]) -> Tuple[str, int]
            # 初始化空字符串来存储字典的所有键
            key_str = ""
            # 初始化求和变量
            sum = 0
            # 遍历字典中的键
            for key in x.keys():
                # 将字典中每个键连接成一个字符串
                key_str += key
            # 遍历字典中的值
            for val in x.values():
                # 将字典中每个值累加到求和变量
                sum += val
            return key_str, sum

        # 使用给定的测试脚本检查 test_dict_keys_values 函数
        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

    def test_for_tuple_unpack(self):
        def for_tuple_unpack(x, y):
            # 将列表中的每个子列表的两个元素分别赋值给 i 和 j
            for i, j in [[3, 4], [5, 6], [7, 8]]:
                x += i
                y += j
            return x, y

        # 使用给定的测试脚本检查 for_tuple_unpack 函数
        self.checkScript(for_tuple_unpack, (torch.tensor(3), torch.tensor(5)))

        def nested_tuple_unpack(x, y):
            # type: (List[int], List[int]) -> int
            # 初始化求和变量
            sum = 0
            # 遍历两个列表中的元素，同时使用 zip 函数获取 x 的索引和值，并将 y 的值分配给 v
            for i, (j, k), v in zip(x, enumerate(x), y):
                # 将所有元素的和累加到求和变量
                sum += i + j + k + v
            return sum

        # 使用给定的测试脚本检查 nested_tuple_unpack 函数
        self.checkScript(nested_tuple_unpack, ([1, 3, 5], [2, 4, 6]))

    def test_for_tuple_assign(self):
        def test_simple_assign(x):
            # type: (Tuple[int, float]) -> float
            # 初始化求和变量
            sum = 0.0
            # 遍历元组中的每个元素，并将其转换为 float 类型后累加到求和变量
            for a in x:
                sum += float(a)
            return sum

        # 使用给定的测试脚本检查 test_simple_assign 函数
        self.checkScript(test_simple_assign, ((1, 2.5),))

        def test_tuple_assign(x):
            # type: (Tuple[Tuple[int, int], Tuple[int, int]]) -> int
            # 初始化求和变量
            sum = 0
            # 遍历元组中的每个元组，将每个元组中的两个整数相加并累加到求和变量
            for a in x:
                sum += a[0]
                sum += a[1]
            return sum

        # 使用给定的测试脚本检查 test_tuple_assign 函数
        self.checkScript(test_tuple_assign, (((1, 2), (4, 7)), ))

    def test_single_starred_lhs(self):
        # 在非 star 表达式的存在下，将 starred 表达式只出现在 lhs 上的情况下抛出运行时错误
        with self.assertRaisesRegex(RuntimeError, 'A Starred expression may only appear on the lhs within the presence'
                                                  ' of another non-starred expression'):
            cu = torch.jit.CompilationUnit('''
            def single_starred_lhs(x):
                a = (x, x, x)
                *b, = a
                return b
            ''')

    def test_singleton_tuple_unpack(self):
        def foo(a):
            # 将单个元素的元组解包并赋值给 b
            b, = (a,)
            return b + 1
        # 使用给定的测试脚本检查 foo 函数
        self.checkScript(foo, (torch.rand(3),))
    def test_tuple_assignments(self):
        def var_tuple_assign(x, y):
            # type: (Tuple[Tensor, Tensor], Tensor) -> Tensor
            # 解构元组 x 为 (a, b) 和变量 c
            (a, b), c = x, y
            return a + b + c

        tuple_inputs = (torch.randn(1, 4), torch.randn(3, 4))
        # 调用自定义函数 var_tuple_assign 进行脚本化检查
        self.checkScript(var_tuple_assign, (tuple_inputs, torch.randn(3, 4)))

        def nested_tuple_assign(x, y, z):
            # type: (int, Tuple[int, Tuple[int, int]], Tuple[int, int]) -> int
            # 解构 x 为 a，y 为 (b, (c, d))，z 为 (e, f)
            a, (b, (c, d)), (e, f) = x, y, z
            return a + b + c + d + e + f

        # 调用自定义函数 nested_tuple_assign 进行脚本化检查
        self.checkScript(nested_tuple_assign, ((1, (2, (3, 4)), (5, 6))))

        def subscript_tuple_assign(a, x, i):
            # type: (List[int], Tensor, int) -> Tuple[int, Tensor, int]
            # 修改列表 a 的第 i 个元素为 1，解构元组 (x[i], b)
            a[i], (x[i], b) = 1, (2, 3)
            return a[i] + 1, x + 5, b

        # 调用自定义函数 subscript_tuple_assign 进行脚本化检查
        self.checkScript(subscript_tuple_assign, ([12, 7, 9, 11], torch.tensor((3, 13, 17)), 0))

        def star_tuple_assign():
            # type: () -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]
            # 解构 a 为 1，(b, *c) 为 (2, 3, 4)，*d 为 (5, 6)
            a, (b, *c), *d = 1, (2, 3, 4), 5, 6
            return a, b, c, d

        # 调用自定义函数 star_tuple_assign 进行脚本化检查
        self.checkScript(star_tuple_assign, ())

        def subscript_tuple_augmented_assign(a):
            # type: (Tuple[int, int]) -> Tuple[int, int]
            # 抛出运行时异常，不支持增强赋值
            a[0] += 1
            return a

        # 使用 torch.jit.script 尝试脚本化 subscript_tuple_augmented_assign 函数，期望抛出异常
        with self.assertRaisesRegex(RuntimeError, 'does not support augmented assign'):
            scripted_aug_assign = torch.jit.script(subscript_tuple_augmented_assign)

        class AttrTupleAssignmentTestClass:
            def __init__(self, a: int, b: int):
                self.a = a
                self.b = b

            def set_ab(self, a: int, b: int):
                # 对象属性同时赋值为 (a, b)
                self.a, self.b = (a, b)

            def get(self) -> Tuple[int, int]:
                return (self.a, self.b)

        make_global(AttrTupleAssignmentTestClass)

        @torch.jit.script
        def attr_tuple_assignment(o: AttrTupleAssignmentTestClass, a: int, b: int):
            # 调用对象方法 set_ab 设置属性值为 (a, b)，并返回对象 o
            o.set_ab(a, b)
            return o

        o = AttrTupleAssignmentTestClass(1, 2)
        # 调用 attr_tuple_assignment 函数，期望返回设置后的属性值 (3, 4)
        self.assertEqual(attr_tuple_assignment(o, 3, 4).get(), (3, 4))

    def test_multiple_assign(self):
        def test():
            # 多重赋值
            a = b, c = d, f = (1, 1)

            # 副作用
            ten = torch.tensor(1)
            ten1 = ten2 = ten.add_(1)

            # 顺序赋值
            x = 1
            y = 3
            x, y = y, x + y

            return a, b, c, d, f, ten, ten1, ten2, x, y

        # 调用 test 函数进行脚本化检查
        self.checkScript(test, ())

    def test_multi_reduction(self):
        # 使用 torch.jit.CompilationUnit 尝试编译包含错误的多重赋值的脚本
        with self.assertRaisesRegex(
                RuntimeError,
                'augmented assignment can only have one LHS expression'):
            cu = torch.jit.CompilationUnit('''
            def multi_reduction(x):
                a, b += x
                return a, b
            ''')
    def test_invalid_call_arguments(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'but instead found type '):
            # 使用 Torch Script 装饰器将函数标记为 Torch 脚本
            @torch.jit.script
            def invalid_call_arguments(x):
                # 尝试调用 torch.unsqueeze 函数，传入多个参数
                return torch.unsqueeze(3, 4, 5, 6, 7, 8)

    def test_invalid_lhs_assignment(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'unexpected expression'):
            # 创建一个 Torch 脚本编译单元，包含一个无效的赋值语句
            cu = torch.jit.CompilationUnit('''
            def invalid_lhs_assignment(x):
                # 尝试将 x + 1 赋值给 x，这是不合法的
                x + 1 = x
                return x
            ''')

    def test_multi_starred_expr_lhs(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'Only one starred expression is allowed on the lhs'):
            # 创建一个 Torch 脚本编译单元，包含多个星号表达式作为左值
            cu = torch.jit.CompilationUnit('''
            def multi_starred_expr_lhs():
                # 尝试在赋值时使用多个星号表达式，这是不允许的
                a, *b, *c = [1, 2, 3, 4, 5, 6]
                return a
            ''')

    def test_pack_tuple_into_non_var(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'Cannot pack a tuple into a non-variable'):
            # 创建一个 Torch 脚本编译单元，包含尝试将元组打包进非变量的赋值
            cu = torch.jit.CompilationUnit('''
            def pack_tuple_into_non_var(x):
                # 尝试使用 *1 将元组 (3, 4, 5) 打包进非变量
                a, *1 = (3, 4, 5)
                return x
            ''')

    def test_print_kwargs(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'print doesn\'t accept any keyword arguments'):
            # 创建一个 Torch 脚本编译单元，包含尝试在 print 函数中使用关键字参数的调用
            cu = torch.jit.CompilationUnit('''
            def print_kwargs(x):
                # 尝试在 print 函数调用中使用 flush=True 的关键字参数
                print(x, flush=True)
                return x
            ''')

    def test_builtin_use_as_value(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'builtin cannot be used as a value'):
            # 使用 Torch Script 装饰器将函数标记为 Torch 脚本
            @torch.jit.script
            def builtin_use_as_value(x):
                # 尝试使用内置函数 unsqueeze 作为一个值返回
                return x.unsqueeze

    def test_wrong_use_as_tuple(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a tuple'):
            # 定义一个普通 Python 函数 test_fn
            def test_fn():
                return 3

            # 使用 Torch Script 装饰器将函数标记为 Torch 脚本
            @torch.jit.script
            def wrong_use_as_tuple(self):
                # 尝试将 test_fn 当作元组来使用
                a, b = test_fn
                return a

    def test_wrong_attr_lookup(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'attribute lookup is not defined on builtin'):
            # 使用 Torch Script 装饰器将函数标记为 Torch 脚本
            @torch.jit.script
            def wrong_attr_lookup(self, x):
                # 尝试访问 x.unsqueeze.myattr，这是不允许的操作
                a = x.unsqueeze.myattr
                return a

    def test_wrong_use_as_callable(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'cannot call a value'):
            # 使用 Torch Script 装饰器将函数标记为 Torch 脚本
            @torch.jit.script
            def wrong_use_as_callable(x):
                # 尝试调用一个非可调用对象 x
                return x(3, 4, 5)

    def test_python_val_doesnt_have_attr(self):
        # 断言捕获运行时错误，期望错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute abcd'):
            # 使用 Torch Script 装饰器将函数标记为 Torch 脚本
            @torch.jit.script
            def python_val_doesnt_have_attr():
                # 尝试访问 shutil.abcd 属性，但 shutil 不是一个模块
                # 因此无法进行属性查找
                return shutil.abcd
    # 测试错误的模块属性查找
    def test_wrong_module_attr_lookup(self):
        # 断言在运行时抛出异常，异常信息包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'python value of type \'type\' cannot be used as a value'):
            import io  # 导入io模块

            @torch.jit.script
            def wrong_module_attr_lookup():
                return io.BytesIO  # 返回io.BytesIO对象

    # 测试错误的方法调用参数
    def test_wrong_method_call_inputs(self):
        # 断言在运行时抛出异常，异常信息包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'Argument y not provided'):
            # 定义一个继承自torch.jit.ScriptModule的类SomeModule
            class SomeModule(torch.jit.ScriptModule):

                @torch.jit.script_method
                def foo(self, x, y):
                    return x

                @torch.jit.script_method
                def forward(self, x, y):
                    return self.foo(x)  # 调用self.foo(x)，但未提供参数y
            SomeModule()  # 实例化SomeModule类

    # 测试使用单星号表达式的for循环
    def test_single_starred_expr_for_loop(self):
        # 断言在运行时抛出异常，异常信息包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'A Starred expression may only appear'):
            cu = torch.jit.CompilationUnit('''
            def test():
                x = 0
                for *a in [1, 2, 3]:  # 使用单星号表达式，但语法错误
                    x = x + 1
                return x
            ''')

    # 测试调用函数时传入过多的参数
    def test_call_ge(self):
        # 断言在运行时抛出异常，异常信息包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'Expected at most 1 arguments but found 3'):
            @_trace(torch.zeros(1, 2, 3))  # 使用_trace装饰器
            def foo(x):
                return x

            @torch.jit.script
            def test_fn():
                return foo(torch.full([1], 1), torch.full([1], 2), torch.full([1], 3))  # 传入了三个参数，但foo函数定义只接受一个参数

    # 测试错误的返回类型
    def test_wrong_return_type(self):
        # 断言在运行时抛出异常，异常信息包含特定字符串
        with self.assertRaisesRegex(RuntimeError, 'but instead got value of type tuple'):
            @torch.jit.ignore
            def somefunc():
                # type: () -> Tuple[Tuple[Tensor, Tensor]]
                return torch.zeros(3, 4), torch.zeros(4, 5)  # 返回类型与声明不符

            @torch.jit.script
            def wrong_return_type():
                return somefunc()  # 调用somefunc返回的类型与预期不符
            wrong_return_type()

    # 测试从跟踪函数中调用Python函数
    def test_call_python_fn_from_tracing_fn(self):
        # 定义一个Python函数python_fn
        def python_fn(x):
            return torch.neg(x)  # 调用torch.neg函数对x取负值

        @_trace(torch.rand(3, 4))  # 使用_trace装饰器
        def traced_fn(x):
            return python_fn(x) + 1  # 调用python_fn并加1

        # 断言在生成的图中包含neg操作，表明python_fn中的neg操作已正确内联到图中
        FileCheck().check("aten::neg").run(str(traced_fn.graph))
    def test_call_python_mod_from_tracing_fn(self):
        # 定义一个简单的 Python 模块类
        class PythonMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个不需要梯度的随机参数矩阵
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            # 前向传播函数，执行矩阵乘法操作
            def forward(self, x):
                return torch.mm(x, self.param)

        # 创建 PythonMod 的实例
        pm = PythonMod()

        # 使用 @_trace 装饰器对函数进行追踪
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # 调用 PythonMod 的实例 pm 进行前向传播，并加上常数 1.0
            return pm(x) + 1.0

        # 断言追踪函数的输入参数数量为 1
        self.assertTrue(len(list(traced_fn.graph.inputs())) == 1)
        # 使用 FileCheck 验证图中是否包含特定的运算节点（矩阵乘法和加法）
        FileCheck().check("aten::mm").check("aten::add").run(str(traced_fn.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_tracing_fn(self):
        # 使用 @_trace 装饰器对函数进行追踪
        @_trace(torch.rand(3, 4))
        def traced_fn1(x):
            # 返回输入张量的负值
            return torch.neg(x)

        # 使用 @_trace 装饰器对函数进行追踪
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # 调用 traced_fn1 函数对输入张量进行操作，并加上常数 1
            return traced_fn1(x) + 1

        # 使用 FileCheck 验证图中是否包含特定的运算节点（函数调用和加法）
        FileCheck().check("traced_fn").check("prim::CallFunction").check("aten::add") \
            .run(str(traced_fn.graph))

    @unittest.skip("error in first class mode")
    def test_call_traced_mod_from_tracing_fn(self):
        # 定义一个追踪过的 PyTorch 模块类
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个不需要梯度的随机参数矩阵
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            # 前向传播函数，执行矩阵乘法操作
            def forward(self, x):
                return torch.mm(x, self.param)

        # 对 TracedModule 进行追踪
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # 断言在运行时抛出特定异常，指示必须将其注册为子模块
        with self.assertRaisesRegex(RuntimeError, "must be registered as submodules"):
            @_trace(torch.rand(3, 4))
            def traced_fn(x):
                # 调用追踪过的模块 tm 进行前向传播，并加上常数 1.0
                return tm(x) + 1.0

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_tracing_fn(self):
        # 使用 TorchScript 将函数 script_fn 转换为 Torch 脚本
        @torch.jit.script
        def script_fn(x):
            # 返回输入张量的负值
            return torch.neg(x)

        # 使用 @_trace 装饰器对函数进行追踪
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # 调用 Torch 脚本函数 script_fn 对输入张量进行操作，并加上常数 1
            return script_fn(x) + 1

        # 使用 FileCheck 验证图中是否包含特定的运算节点（函数调用和加法）
        FileCheck().check("prim::CallFunction").check("aten::add").run(str(traced_fn.graph))

    @unittest.skip("error in first class mode")
    def test_call_script_mod_from_tracing_fn(self):
        # 断言在运行时抛出特定异常，指示必须将其注册为子模块
        with self.assertRaisesRegex(RuntimeError, "must be registered as submodules"):
            # 定义一个 TorchScript 的脚本模块类
            class ScriptMod(torch.jit.ScriptModule):
                def __init__(self):
                    super().__init__()
                    # 初始化一个不需要梯度的随机参数矩阵
                    self.param = torch.nn.Parameter(torch.rand(3, 4), requires_grad=False)

                # 使用 TorchScript 方法定义的前向传播函数
                @torch.jit.script_method
                def forward(self, x):
                    for _i in range(4):
                        x += self.param
                    return x

            # 创建 ScriptMod 的实例
            sm = ScriptMod()

            # 使用 @_trace 装饰器对函数进行追踪
            @_trace(torch.rand(3, 4))
            def traced_fn(x):
                # 调用 TorchScript 模块 sm 进行前向传播，并加上常数 1.0
                return sm(x) + 1.0
    def test_call_python_fn_from_traced_module(self):
        def python_fn(x):
            return torch.neg(x)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                # 调用定义的 python_fn 函数对输入进行负值操作，然后与 self.param 矩阵相乘
                return torch.mm(python_fn(x), self.param)

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # Note: parameter self.param from the traced module should appear as
        # an input to the graph and the neg op from the Python function should
        # be properly inlined
        # 断言：验证图中的输入参数数量为2（包括 self.param 和输入 x）
        self.assertTrue(len(list(tm.graph.inputs())) == 2)
        # 使用 FileCheck 工具验证图中是否包含 "aten::neg" 和 "aten::mm" 操作
        FileCheck().check("aten::neg").check("aten::mm").run(str(tm.graph))

    def test_call_python_mod_from_traced_module(self):
        class PythonModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                # 使用 torch.mm 计算输入 x 与 self.param 矩阵的乘积
                return torch.mm(x, self.param)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = PythonModule()

            def forward(self, x):
                # 调用 PythonModule 实例的 forward 方法，计算结果与 1.0 相加
                return self.mod(torch.mm(x, self.param)) + 1.0

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # 使用 FileCheck 工具验证图中不包含 "value=<Tensor>"，并检查是否包含 "aten::mm"、
        # 'prim::CallMethod[name="forward"]' 和 "aten::add" 操作
        FileCheck().check_not("value=<Tensor>").check("aten::mm")\
            .check('prim::CallMethod[name="forward"]').check("aten::add") \
            .run(str(tm.graph))
        # 使用 FileCheck 工具验证 PythonModule 实例的图中是否包含 "aten::mm" 操作
        FileCheck().check("aten::mm").run(str(tm.mod.graph))

    def test_op_dtype(self):

        def check_equal_and_dtype(a, b):
            # 断言：验证 a 和 b 的值相等
            self.assertEqual(a, b)
            # 断言：验证 a 和 b 的数据类型相等
            self.assertEqual(a.dtype, b.dtype)

        def fn():
            # 创建多个张量 a, b, c, d, e, f，使用不同的参数类型和步长
            a = torch.arange(10)
            b = torch.arange(10, dtype=torch.float)
            c = torch.arange(1, 10, 2)
            d = torch.arange(1, 10, 2, dtype=torch.float)
            e = torch.arange(1, 10., 2)
            f = torch.arange(1, 10., 2, dtype=torch.float)
            return a, b, c, d, e, f

        # 使用 torch.jit.script 对函数 fn 进行脚本化
        scripted_fn = torch.jit.script(fn)
        # 分别获取 eager 模式和脚本化模式下函数 fn 的输出
        eager_out = fn()
        script_out = scripted_fn()
        # 遍历输出结果，并对每对结果调用 check_equal_and_dtype 进行比较
        for a, b in zip(eager_out, script_out):
            check_equal_and_dtype(a, b)

    def test_floor_div(self):
        @torch.jit.script
        def foo(a, b):
            # type: (int, int) -> int
            # 返回整数除法结果 a // b
            return a // b
        # 遍历多个整数对 (i, j)，验证除法操作 foo(i, j) 是否正确
        for i in range(-8, 8):
            for j in range(-8, 8):
                if j != 0:
                    # 断言：验证 foo(i, j) 的计算结果是否等于 i // j
                    self.assertEqual(foo(i, j), i // j)
    # 定义测试方法：测试 torch.floor_divide 函数
    def test_floordiv(self):
        # 定义包含代码模板的字符串，用于生成测试函数
        funcs_template = dedent('''
        def fn():
            # 创建张量 ten，根据给定构造方法 {a_construct}
            ten = {a_construct}
            # 创建张量或标量 ten_or_scalar，根据给定构造方法 {b_construct}
            ten_or_scalar = {b_construct}
            # 返回 ten 与 ten_or_scalar 的整除结果及使用 torch.floor_divide 函数的结果
            return ten // ten_or_scalar, torch.floor_divide(ten, ten_or_scalar)
        ''')

        # 左操作数的不同构造方法
        lhs = ["torch.tensor([5.5, 3.2])", "torch.tensor([2, 2])", "torch.tensor([3, 2])"]
        # 右操作数的不同构造方法，包括标量和左操作数的各种构造方法
        rhs = ["1.5", "2", "4", "1.1"] + lhs
        # 对每个左操作数和右操作数组合进行测试
        for tensor in lhs:
            for tensor_or_scalar in rhs:
                # 使用 funcs_template 生成具体的测试函数代码字符串
                funcs_str = funcs_template.format(a_construct=tensor, b_construct=tensor_or_scalar)
                # 创建一个空的作用域
                scope = {}
                # 执行包装后的 funcs_str 代码，在全局变量空间 globals() 中定义
                execWrapper(funcs_str, globals(), scope)
                # 使用 funcs_str 创建 torch.jit.CompilationUnit 对象
                cu = torch.jit.CompilationUnit(funcs_str)
                # 从 cu 中获取名为 fn 的函数
                f_script = cu.fn
                # 从作用域中获取名为 'fn' 的函数
                f = scope['fn']
                # 断言编译后的脚本函数和动态执行得到的函数结果相等
                self.assertEqual(f_script(), f())

    # 定义测试方法：测试从脚本函数调用 Python 函数
    def test_call_python_fn_from_script_fn(self):
        # 定义一个被 torch.jit.ignore 忽略的 Python 函数
        @torch.jit.ignore
        def python_fn(x):
            return torch.neg(x)

        # 定义一个使用 torch.jit.script 装饰的脚本函数
        def script_fn(x):
            return python_fn(x) + 1

        # 脚本函数的调用，预期结果是 torch.tensor(0)
        a = torch.tensor(1)
        self.assertEqual(script_fn(a), torch.tensor(0))
        # 使用 FileCheck 检查脚本函数的计算图，查找 "python_fn" 字符串
        FileCheck().check("python_fn").run(str(script_fn.graph))

    # 定义测试方法：测试从脚本函数调用 Python 模块
    def test_call_python_mod_from_script_fn(self):
        # 定义一个继承自 torch.nn.Module 的 Python 类
        class PythonModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                return torch.mm(x, self.param)

        # 创建 PythonModule 类的实例 pm
        pm = PythonModule()

        # 定义一个使用 torch.jit.script 装饰的脚本函数
        def script_fn(x):
            return pm(x) + 1

        # 使用 FileCheck 检查脚本函数的计算图，查找 "python_value" 和 "aten::add" 字符串
        FileCheck().check("python_value").check("aten::add").run(str(script_fn.graph))

    # 定义测试方法：测试从脚本函数调用另一个脚本函数
    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_script_fn(self):
        # 定义一个使用 torch.jit.script 装饰的脚本函数
        def script_fn1(x):
            return torch.neg(x)

        # 定义一个使用 torch.jit.script 装饰的脚本函数
        def script_fn(x):
            return script_fn1(x) + 1

        # 使用 FileCheck 检查脚本函数的计算图，查找 "prim::CallFunction" 字符串
        FileCheck().check("prim::CallFunction").run(str(script_fn.graph))

    # 定义测试方法：测试从脚本函数调用脚本模块
    def test_call_script_mod_from_script_fn(self):
        # 使用 self.assertRaisesRegex 断言特定异常消息
        with self.assertRaisesRegex(RuntimeError, "Cannot call a ScriptModule that is not a submodule of the caller"):
            # 定义一个继承自 torch.jit.ScriptModule 的脚本模块类
            class ScriptMod(torch.jit.ScriptModule):
                @torch.jit.script_method
                def forward(self, x):
                    return torch.mm(x, torch.zeros([4, 3]))

            # 创建 ScriptMod 类的实例 sm
            sm = ScriptMod()

            # 定义一个使用 torch.jit.script 装饰的脚本函数
            def script_fn(x):
                return sm(x) + 1
    def test_call_python_fn_from_script_module(self):
        @torch.jit.ignore
        def python_fn(x):
            return torch.neg(x)
        # 定义一个被 Torch JIT 忽略的 Python 函数，对输入 x 取负

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                # 定义一个形状为 (4, 3) 的参数张量

            @torch.jit.script_method
            def forward(self, x):
                return python_fn(torch.mm(x, self.param))
                # 调用 python_fn 函数，对输入 x 和 self.param 进行矩阵乘法运算后取负

        sm = ScriptMod()
        # 创建 ScriptMod 实例

        FileCheck().check("aten::mm").check("python_fn") \
            .run(str(sm.forward.graph))
        # 使用 FileCheck 验证 sm.forward 的计算图中是否包含 "aten::mm" 和 "python_fn"

    def test_call_python_mod_from_script_module(self):
        class PythonMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))
                # 定义一个形状为 (3, 5) 的参数张量

            @torch.jit.ignore
            def forward(self, x):
                return torch.mm(x, self.param)
                # Torch JIT 忽略的 Python 模块的 forward 方法，对输入 x 和 self.param 进行矩阵乘法运算

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                # 定义一个形状为 (4, 3) 的参数张量
                self.pm = PythonMod()
                # 创建 PythonMod 实例

            @torch.jit.script_method
            def forward(self, x):
                return self.pm(torch.mm(x, self.param))
                # 调用 self.pm 的 forward 方法，对输入 x 和 self.param 进行矩阵乘法运算

        sm = ScriptMod()
        # 创建 ScriptMod 实例

        # 注意：调用 PythonMod 的 forward 方法显示为 ^forward()，参数不会内联展开
        FileCheck().check("aten::mm").check("forward").run(str(sm.graph))
        # 使用 FileCheck 验证 sm.forward 的计算图中是否包含 "aten::mm" 和 "forward"

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_script_module(self):
        @torch.jit.script
        def script_fn(x):
            return torch.neg(x)
            # 定义一个 Torch JIT 脚本函数，对输入 x 取负

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                # 定义一个形状为 (4, 3) 的参数张量

            @torch.jit.script_method
            def forward(self, x):
                return script_fn(torch.mm(x, self.param))
                # 调用 script_fn 函数，对输入 x 和 self.param 进行矩阵乘法运算后取负

        sm = ScriptMod()
        # 创建 ScriptMod 实例

        graph = (sm.forward.graph)
        # 获取 sm.forward 的计算图

        FileCheck().check("aten::mm").check("prim::CallFunction").run(str(graph))
        # 使用 FileCheck 验证 sm.forward 的计算图中是否包含 "aten::mm" 和 "prim::CallFunction"
    # 定义一个测试函数，用于测试调用一个继承自torch.jit.ScriptModule的子类ScriptMod1
    def test_call_script_mod_from_script_module(self):
        # 定义一个继承自torch.jit.ScriptModule的类ScriptMod1
        class ScriptMod1(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个3x5的随机权重参数
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            # 定义前向传播方法，实现矩阵乘法操作
            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.param)

        # 定义一个继承自torch.jit.ScriptModule的类ScriptMod
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个4x3的随机权重参数
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                # 创建ScriptMod1类的实例tm
                self.tm = ScriptMod1()

            # 定义前向传播方法，调用ScriptMod1实例tm的前向传播方法
            @torch.jit.script_method
            def forward(self, x):
                return self.tm(torch.mm(x, self.param))

        # 创建ScriptMod类的实例sm
        sm = ScriptMod()
        # 注意事项: 两个模块的参数应出现在图的扁平化输入列表中。ScriptMod1的mm操作应被正确地内联。
        # 在图的输入列表中应有3%的值，体内应有两个mm操作。
        FileCheck().check_count('%', 3).check(":").check_count("mm", 1).check("prim::CallMethod").run(str(sm.graph))

    # 定义一个测试函数，测试包含参数的模块的调用失败情况
    def test_module_with_params_called_fails(self):
        # 使用断言检查抛出的异常信息是否包含指定文本
        with self.assertRaisesRegex(RuntimeError, "Cannot call a ScriptModule that is not a submodule of the caller"):
            # 定义一个继承自torch.jit.ScriptModule的类ScriptMod
            class ScriptMod(torch.jit.ScriptModule):
                def __init__(self):
                    super().__init__()
                    # 创建一个3x3的随机权重参数
                    self.param = torch.nn.Parameter(torch.rand(3, 3))

                # 定义前向传播方法，实现矩阵乘法操作
                @torch.jit.script_method
                def forward(self, x):
                    return torch.mm(x, self.param)

            # 创建ScriptMod类的实例sm
            sm = ScriptMod()

            # 定义一个使用ScriptMod实例sm的函数some_func
            @torch.jit.script
            def some_func(x):
                return sm(x)

    # 定义一个测试函数，测试对元组索引到列表的支持情况
    def test_tuple_index_to_list(self):
        # 定义一个接受布尔类型输入并返回整数的函数test_non_constant_input
        def test_non_constant_input(a):
            # type: (bool) -> int
            if a:
                b = 1
            else:
                b = 0
            # 创建一个元组c
            c = (0, 1)
            # 返回元组c中索引为b的值
            return c[b]

        # 使用self.checkScript测试函数test_non_constant_input，输入为(True)
        self.checkScript(test_non_constant_input, (True,))
        # 使用self.checkScript测试函数test_non_constant_input，输入为(False)
        self.checkScript(test_non_constant_input, (False,))

        # 使用断言检查抛出的异常信息是否包含指定文本
        with self.assertRaisesRegex(RuntimeError, "because we cannot resolve the output type"):
            # 定义一个使用torch.jit.script装饰的函数test_non_constant_input
            @torch.jit.script
            def test_non_constant_input(a):
                # type: (bool) -> None
                if a:
                    b = 1
                else:
                    b = 0
                # 创建一个元组c，包含一个浮点数1.1
                c = (0, 1.1)
                # 打印元组c中索引为b的值（这里会引发类型解析失败的异常）
                print(c[b])
    # 定义一个测试函数，测试元组的索引操作
    def test_tuple_indexing(self):
        # 定义内部函数 tuple_index，根据参数 a 返回元组 b 的部分元素
        def tuple_index(a):
            # 如果 a 转换为布尔值为真，则创建元组 b 包含 (1, 2)
            if bool(a):
                b = (1, 2)
            else:
                # 否则创建元组 b 包含 (0, 2)
                b = (0, 2)
            # 返回元组 b 的倒数第二个元素和第二个元素
            return b[-2], b[1]

        # 使用 self.checkScript 方法测试 tuple_index 函数的脚本化版本
        self.checkScript(tuple_index, (torch.tensor([0]),))
        self.checkScript(tuple_index, (torch.tensor([1]),))
        self.checkScript(tuple_index, (torch.tensor([1]),), optimize=True)
        # 将 tuple_index 函数脚本化，赋值给 tuple_comp
        tuple_comp = torch.jit.script(tuple_index)
        # 使用 FileCheck 检查 tuple_comp 的图中 "TupleIndex" 出现的次数是否为 2
        FileCheck().check_count("TupleIndex", 2, exactly=True).run(str(tuple_comp.graph))

        # 使用 self.assertRaisesRegex 捕获 RuntimeError 异常，验证对浮点数索引的错误处理
        with self.assertRaisesRegex(RuntimeError, "index must be an integer"):
            @torch.jit.script
            def test_indexing_float():
                # 创建元组 c 包含 (1, 2)
                c = (1, 2)
                # 返回 c 的浮点数索引，预期会抛出异常
                return c[0.1]

        # 定义测试函数 test_indexing_out_of_bounds_pos，测试正索引越界情况
        def test_indexing_out_of_bounds_pos():
            # 创建元组 c 包含 (1, 2)
            c = (1, 2)
            # 尝试访问 c 中不存在的索引 2，预期抛出异常 "out of range"
            return c[2]

        # 使用 self.checkScriptRaisesRegex 捕获 Exception 异常，验证正索引越界的处理
        self.checkScriptRaisesRegex(test_indexing_out_of_bounds_pos, (), Exception,
                                    "out of range")

        # 定义测试函数 test_indexing_out_of_bounds_neg，测试负索引越界情况
        def test_indexing_out_of_bounds_neg():
            # 创建元组 c 包含 (1, 2)
            c = (1, 2)
            # 尝试访问 c 中不存在的负索引 -3，预期抛出异常 "out of range"
            return c[-3]

        # 使用 self.checkScriptRaisesRegex 捕获 Exception 异常，验证负索引越界的处理
        self.checkScriptRaisesRegex(test_indexing_out_of_bounds_pos, (), Exception,
                                    "out of range")

        # 定义 negative_index 函数，测试负索引访问最后一个元素
        def negative_index():
            # 创建元组 tup 包含 (1, 2, 3, 4)
            tup = (1, 2, 3, 4)
            # 返回 tup 的倒数第一个元素，即 4
            return tup[-1]

        # 使用 self.checkScript 方法测试 negative_index 函数的脚本化版本
        self.checkScript(negative_index, [])

        # 定义 really_negative_index 函数，测试超出负索引范围的情况
        def really_negative_index():
            # 创建元组 tup 包含 (1, 2, 3, 4)
            tup = (1, 2, 3, 4)
            # 尝试访问 tup 中不存在的超负索引 -100，预期抛出异常 "index out of range"
            return tup[-100]

        # 使用 self.checkScriptRaisesRegex 捕获 Exception 异常，验证超负索引的处理
        self.checkScriptRaisesRegex(really_negative_index, [], Exception, "index out of range")

        # 定义 negative_slice 函数，测试负索引切片操作
        def negative_slice():
            # 创建元组 tup 包含 (1, 2, 3, 4)
            tup = (1, 2, 3, 4)
            # 返回 tup 的负索引切片 [-3:4]，即 (2, 3, 4)
            return tup[-3:4]

        # 使用 self.checkScript 方法测试 negative_slice 函数的脚本化版本
        self.checkScript(negative_slice, [])

        # 定义 really_slice_out_of_bounds 函数，测试超出负索引切片范围的情况
        def really_slice_out_of_bounds():
            # 创建元组 tup 包含 (1, 2, 3, 4)
            tup = (1, 2, 3, 4)
            # 尝试访问 tup 中不存在的超负索引切片范围 [-300:4000]，预期返回空元组 ()
            return tup[-300:4000]

        # 使用 self.checkScript 方法测试 really_slice_out_of_bounds 函数的脚本化版本
        self.checkScript(really_slice_out_of_bounds, [])

    # 定义测试函数，测试命名元组的属性访问
    def test_namedtuple_attr(self):
        # 定义函数 f，对输入张量 x 进行操作，返回两个张量的最大索引相加
        def f(x):
            return x.max(dim=1).indices + torch.max(x, dim=1).indices

        # 使用 self.checkScript 方法测试函数 f 的脚本化版本
        self.checkScript(f, (torch.rand(20, 20, 20),), optimize=True)

        # 使用 self.assertRaisesRegex 捕获 RuntimeError 异常，验证访问未知属性或方法的错误处理
        with self.assertRaisesRegex(RuntimeError, "object has no attribute or method"):
            @torch.jit.script
            def g1(x):
                # 尝试访问 x 的未知属性 unknown_symbol，预期抛出异常
                return x.max(dim=1).unknown_symbol

        # 使用 self.assertRaisesRegex 捕获 RuntimeError 异常，验证访问未知属性或方法的错误处理
        with self.assertRaisesRegex(RuntimeError, "object has no attribute or method"):
            @torch.jit.script
            def g2(x):
                # 打印元组 (x, x, x) 的文档字符串，预期抛出异常
                print((x, x, x).__doc__)
                return x

    # 定义测试函数，测试元组的长度获取操作
    def test_tuple_len(self):
        # 使用 torch.jit.script 脚本化函数 foo，返回元组 (1, "str", None) 的长度
        @torch.jit.script
        def foo():
            return len((1, "str", None))

        # 使用 self.assertEqual 断言 foo 函数的返回值为 3
        self.assertEqual(foo(), 3)

        # 定义测试函数 test_indexing_end_out_of_bounds，测试超出索引范围的切片操作
        @torch.jit.script
        def test_indexing_end_out_of_bounds():
            # 创建元组 c 包含 (1, 2)
            c = (1, 2)
            # 尝试访问 c 中索引范围 [2:10] 的切片，预期返回空元组 ()
            return c[2:10]

        # 使用 self.assertEqual 断言 test_indexing_end_out_of_bounds 函数的返回值为空元组 ()
        self.assertEqual(test_indexing_end_out_of_bounds(), ())
    def test_lower_nested_tuples(self):
        @torch.jit.script
        def test():
            return ((1, 2), 3)

        self.run_pass('constant_propagation', test.graph)
        # 检查图中是否存在常量传播的优化，同时不应存在TupleConstruct
        FileCheck().check("prim::Constant").check_not("TupleConstruct").run(test.graph)
        # 如果无法降级元组，则测试失败
        self.run_pass('lower_all_tuples', test.graph)

    def test_unwrap_optional_builtin(self):
        def test(x):
            # type: (Optional[int]) -> int
            # 解包可选类型的值x
            x = torch.jit._unwrap_optional(x)
            x = x + x  # noqa: T484
            return x

        self.checkScript(test, (3,))
        
        # 断言解包None时抛出AssertionError异常
        with self.assertRaisesRegex(AssertionError, "Unwrapping null optional"):
            test(None)

        test_script = torch.jit.script(test)
        # 断言解包None时抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "Unwrapping null optional"):
            test_script(None)

        @torch.jit.script
        def test_test():
            return torch.jit._unwrap_optional(1)
        
        # 断言传递None时抛出RuntimeError异常，因为类型无法从实际类型None推断出来
        with self.assertRaisesRegex(RuntimeError, r"could not be inferred from actual type None"):
            @torch.jit.script
            def test_no_type():
                # type: () -> int
                return torch.jit._unwrap_optional(None)

    def test_indexing_error(self):
        # 断言当尝试对int类型的对象进行索引时抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "'int' object is not subscriptable"):
            @torch.jit.script
            def test_wrong_type():
                a = 8
                return a[0]

    def test_unsupported_builtin_error(self):
        # 断言调用不支持的Python内置函数（如math.hypot）时抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError,
                                    "Python builtin <built-in function hypot> is currently"):
            @torch.jit.script
            def test_unsupported(a):
                return math.hypot(a, 2.0)

    def test_annotated_script_fn(self):
        @torch.jit.script
        def foo(x, y, z):
            # type: (Tensor, Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tuple[Tensor, Tensor]]) -> Tensor
            return x
        
        # 断言foo函数的模式字符串符合预期
        self.assertExpected(str(foo.schema))

    def test_annotated_script_method(self):
        class SM(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                # type: (Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor, Tensor]
                return y, y, y
        
        sm = SM()
        
        # 断言sm.forward方法的剥离缠绕后的模式字符串符合预期
        self.assertExpectedStripMangled(str(sm.forward.schema))

    def test_annotated_script_fn_return_mismatch(self):
        # 断言当返回类型与注释中的类型不匹配时抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "but is actually of type"):
            @torch.jit.script
            def return_tup(x):
                # type: (Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]
                return x, x  # noqa: T484
    # 定义一个测试方法，用于验证带有 Torch 脚本装饰器的函数，当参数不匹配时是否会引发 RuntimeError 异常，并且异常信息应包含 "Arguments for call are not valid"
    def test_annotated_script_fn_arg_mismatch(self):
        # 使用 assertRaisesRegex 断言验证是否会抛出指定异常类型和包含特定信息的异常
        with self.assertRaisesRegex(RuntimeError, r"Arguments for call are not valid"):
            # 使用 Torch 的 jit.script 装饰器定义一个函数 tuple_arg，该函数接收一个参数 x
            @torch.jit.script
            def tuple_arg(x):
                # type: (Tuple[Tensor, Tensor]) -> Tensor
                # 函数期望返回一个 Tensor 类型的值，但实际上返回了 x + 1，通过 noqa: T484 标记表示忽略静态类型检查错误
                return x + 1  # noqa: T484

    # 定义一个测试方法，用于验证带有 Torch 脚本装饰器的函数 fn，接收一个 Tensor 类型的参数 x 和一个 float 类型的参数 y，并返回一个 float 类型的值
    def test_script_non_tensor_args_outputs(self):
        # 使用 Torch 的 jit.script 装饰器定义函数 fn
        @torch.jit.script
        def fn(x, y):
            # type: (Tensor, float) -> float
            # 返回值为 (x + y) 的总和，并转换为 float 类型
            return float((x + y).sum())

        # 创建一个 2x2 的全一张量 x
        x = torch.ones(2, 2)
        # 调用函数 fn，并将结果赋给 z
        z = fn(x, 1)
        # 断言 z 的类型为 float
        self.assertIsInstance(z, float)
        # 断言 z 的值等于 8.0
        self.assertEqual(z, 8.)

    # 跳过当前测试用例，注释指向相关 issue 的网址
    @unittest.skip('https://github.com/pytorch/pytorch/issues/9595')
    # 定义一个测试方法，用于验证带有 Torch 脚本装饰器的函数 some_func，接收一个参数 x，并调用函数 to_inline
    def test_inline_and_run_annotated_script_fn(self):
        # 使用 Torch 的 jit.script 装饰器定义函数 to_inline，该函数接收一个 Tuple 类型的参数 (x, x) 和一个 Tensor 类型的参数 x，并返回一个 Tensor 类型的值
        @torch.jit.script
        def to_inline(x, y):
            # type: (Tuple[Tensor, Tensor], Tensor) -> Tensor
            # 直接返回参数 y
            return y

        # 使用 Torch 的 jit.script 装饰器定义函数 some_func，该函数接收一个参数 x，并调用函数 to_inline
        @torch.jit.script
        def some_func(x):
            # 返回调用 to_inline 函数的结果
            return to_inline((x, x), x)

        # 创建一个形状为 3x4 的随机张量 x
        x = torch.rand(3, 4)
        # 断言调用 some_func(x) 的结果等于 x
        self.assertEqual(some_func(x), x)

    # 定义一个辅助方法，用于生成一个测试文件，返回生成的文件名、缓冲区列表和序列化后的偏移量
    def _make_filereader_test_file(self):
        # 创建一个临时文件名
        filename = tempfile.mktemp()
        # 使用 Torch 的 _C.PyTorchFileWriter 类创建一个文件写入对象 writer
        writer = torch._C.PyTorchFileWriter(filename)
        # 创建大小在 1 到 100 之间随机选择的缓冲区列表
        buffers = [os.urandom(size) for size in [random.randint(1, 100) for i in range(20)]]
        # 创建一个偏移量列表
        offsets = []
        # 遍历缓冲区列表，写入每个缓冲区的数据，并记录偏移量
        for i, buf in enumerate(buffers):
            writer.write_record(str(i), buf, len(buf))
            offsets.append(i)
        # 序列化偏移量列表
        serialized_offsets = pickle.dumps(offsets)
        # 将序列化后的偏移量写入文件
        writer.write_record("meta", serialized_offsets, len(serialized_offsets))
        # 写入文件结束标志
        writer.write_end_of_file()
        # 返回生成的文件名、缓冲区列表和序列化后的偏移量
        return filename, buffers, serialized_offsets

    # 定义一个测试方法，用于验证文件格式的序列化和反序列化功能
    def test_file_format_serialization(self):
        # 调用辅助方法生成测试文件，并获取返回值
        filename, buffers, serialized_offsets = self._make_filereader_test_file()

        # 使用 Torch 的 _C.PyTorchFileReader 类创建一个文件读取对象 reader
        reader = torch._C.PyTorchFileReader(filename)
        # 获取 "meta" 记录的序列化偏移量数据
        serialized_offsets_read = reader.get_record("meta")
        # 反序列化得到偏移量列表
        parsed_serialized_offsets = pickle.loads(serialized_offsets)

        # 遍历偏移量列表和缓冲区列表，逐个比较读取的数据与原始数据是否一致
        for i, offset in enumerate(parsed_serialized_offsets):
            # 获取当前偏移量对应的记录数据
            data = reader.get_record(str(offset))
            # 断言读取的数据与原始数据相等
            assert data == buffers[i]

    # 定义一个测试方法，用于验证文件读取类在大量迭代加载时不会出现内存泄漏
    def test_file_reader_no_memory_leak(self):
        # 定义迭代次数
        num_iters = 10000
        # 调用辅助方法生成测试文件，并获取文件名
        filename, _, _ = self._make_filereader_test_file()

        # 从文件名加载数据
        tracemalloc.start()
        for i in range(num_iters):
            torch._C.PyTorchFileReader(filename)
        # 获取追踪到的内存使用情况
        _, peak_from_string = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 从文件流加载数据
        tracemalloc.start()
        with open(filename, 'rb') as f:
            for i in range(num_iters):
                f.seek(0)
                torch._C.PyTorchFileReader(f)
        # 获取追踪到的内存使用情况
        _, peak_from_file = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 断言文件流加载时的内存使用峰值不超过字符串加载时的内存使用峰值的 500 倍
        self.assertLess(peak_from_file, peak_from_string * 500)

    # 对于每种类型，输入类型注释和相应的返回类型注释
    # 返回输入和输出类型的列表，每个元素是一个元组，表示输入类型和对应的输出类型
    def type_input_return_pairs(self):
        return [
            ('Tensor', 'Tensor'),  # 输入和输出都是 Tensor 类型
            ('torch.Tensor', 'Tensor'),  # 输入是 torch.Tensor，输出是 Tensor 类型
            ('str', 'str'),  # 输入和输出都是 str 类型
            ('int', 'int'),  # 输入和输出都是 int 类型
            ('bool', 'bool'),  # 输入和输出都是 bool 类型
            ('BroadcastingList3[float]', 'List[float]'),  # 输入是 BroadcastingList3[float]，输出是 List[float] 类型
            ('BroadcastingList2[int]', 'List[int]'),  # 输入是 BroadcastingList2[int]，输出是 List[int] 类型
            ('List[int]', 'List[int]'),  # 输入和输出都是 List[int] 类型
            ('Optional[int]', 'Optional[int]'),  # 输入和输出都是 Optional[int] 类型
        ]

    # 替换代码中的输入和返回类型的占位符
    def format_code(self, code, pair):
        return code.format(input=pair[0], output=pair[1])

    # ***** Type annotation tests ****

    # 测试不同组合:
    # {String frontend, Python AST Frontend}
    # {Python 3-style type annotations, MyPy-style type comments}
    # {Script method, Script function}

    # 测试组合: String frontend , Python 3-style type annotations , Script function
    def test_annot_string_py3_fn(self):
        code = '''
            def foo(x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                return x, x
        '''
        test_str = []
        for pair in self.type_input_return_pairs():
            # 使用编译单元来编译格式化后的代码，并获取 foo 函数的 schema 信息
            cu = torch.jit.CompilationUnit(self.format_code(code, pair))
            test_str.append(str(cu.foo.schema))
        # 断言测试结果符合预期
        self.assertExpected("\n".join(test_str) + "\n")

    # 测试组合: String frontend , Python 3-style type annotations , Script method
    def test_annot_string_py3_method(self):
        # 定义一个继承自 torch.jit.ScriptModule 的测试模块
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()

        code = '''
            def foo(self, x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                return x, x
        '''
        test_str = []
        for pair in self.type_input_return_pairs():
            # 清除类注册表，因为我们将定义多个 foo 函数
            jit_utils.clear_class_registry()
            tm = TestModule()
            # 在 TestModule 中定义格式化后的代码作为 foo 函数
            tm.define(self.format_code(code, pair))
            test_str.append(str(tm.foo.schema))
        # 断言测试结果符合预期，并删除掉名称混淆
        self.assertExpectedStripMangled("\n".join(test_str) + "\n")

    # 测试组合: String frontend , MyPy-style type comments , Script function
    def test_annot_string_mypy_fn(self):
        code = '''
            def foo(x, y):
                # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
                return x, x
        '''
        test_str = []
        for pair in self.type_input_return_pairs():
            # 使用编译单元来编译格式化后的代码，并获取 foo 函数的 schema 信息
            cu = torch.jit.CompilationUnit(self.format_code(code, pair))
            test_str.append(str(cu.foo.schema))
        # 断言测试结果符合预期，并删除掉名称混淆
        self.assertExpectedStripMangled("\n".join(test_str) + "\n")

    # 测试组合: String frontend , MyPy-style type comments , Script method
    def test_annot_string_mypy_method(self):
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()

        code = '''
        def foo(self, x, y):
            # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
            return x, x
        '''

        test_str = []
        for pair in self.type_input_return_pairs():
            # 清空类注册表，因为我们将多次定义 foo 函数
            jit_utils.clear_class_registry()
            # 创建 TestModule 实例
            tm = TestModule()
            # 使用给定的代码格式化函数，并定义到 TestModule 中
            tm.define(self.format_code(code, pair))
            # 将 foo 函数的 schema 字符串添加到 test_str 列表中
            test_str.append(str(tm.foo.schema))
        # 断言期望的结果字符串
        self.assertExpectedStripMangled("\n".join(test_str) + "\n")

    # Python AST 前端，Python 3 风格的类型注解，Script 函数
    def test_annot_ast_py3_fn(self):
        code = dedent('''
            from typing import Tuple, List, Optional
            from torch import Tensor
            from torch.jit.annotations import BroadcastingList2, BroadcastingList3
            import torch
            @torch.jit.script
            def foo(x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                return x, x
        ''')
        test_str = []
        for pair in self.type_input_return_pairs():
            # 获取经过 Python 3 编码的代码
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'foo')
            # 将 foo 函数的 schema 字符串添加到 test_str 列表中
            test_str.append(str(fn.schema))
        # 断言期望的结果字符串
        self.assertExpectedStripMangled("\n".join(test_str) + "\n")

    def test_multiline_annot_ast_py3_fn(self):
        code = dedent('''
            from typing import Tuple, List, Optional
            from torch import Tensor
            from torch.jit.annotations import BroadcastingList2, BroadcastingList3
            import torch
            @torch.jit.script
            def foo(x,  # type: {input}
                    y   # type: Tuple[Tensor, Tensor]
                    ):
                # type: (...) -> Tuple[{output}, {output}]
                return x, x
        ''')
        test_str = []

        for pair in self.type_input_return_pairs():
            # 获取经过 Python 3 编码的代码
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'foo')
            # 获取函数的参数和返回值
            args = fn.schema.arguments
            returns = fn.schema.returns
            # 断言第一个参数的类型与给定的输入类型匹配
            self.assertEqual(str(args[0].type), pair[1])
            # 断言第二个参数的类型是 Tuple[Tensor, Tensor]
            self.assertEqual(str(args[1].type), "Tuple[Tensor, Tensor]")
            # 断言返回值的类型是 Tuple[input, input]
            self.assertEqual(str(returns[0].type), f"Tuple[{pair[1]}, {pair[1]}]")
    # 定义一个测试方法，用于测试多行注释格式错误的情况
    def test_bad_multiline_annotations(self):
        # 断言捕获运行时错误，并检查错误消息中是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "Return type line"):
            # 使用 torch.jit.script 装饰器，定义一个函数 bad_type_line
            @torch.jit.script
            def bad_type_line(a,  # type: Tensor
                              b,  # type: Tensor
                              c   # type: Tensor
                              ):
                # 注释: 指定参数类型为 (int, int, int)，返回类型为 Tensor
                # 注释: 错误的类型行  # noqa: F723
                return a + b + c

        with self.assertRaisesRegex(RuntimeError, "Return type line"):
            # 使用 torch.jit.script 装饰器，定义一个函数 bad_return_line
            @torch.jit.script
            def bad_return_line(a,  # type: Tensor
                                b,
                                c   # type: Tensor
                                ):
                # 注释: 指定参数类型为 (int, int, int)，返回类型为 Tensor
                return a + b + c

        # TODO: 应该支持这种形式，但解析起来很困难
        with self.assertRaisesRegex(RuntimeError, "Number of type annotations"):
            # 使用 torch.jit.script 装饰器，定义一个函数 missing_type
            @torch.jit.script
            def missing_type(a,  # type: Tensor
                             b,
                             c   # type: Tensor
                             ):
                # 注释: 指定参数为任意类型，返回类型为 Tensor
                return a + b + c

    # Python AST 前端，Python 3 风格的类型注解，Script 方法
    def test_annot_ast_py3_method(self):
        # 定义一个包含 Python 代码的字符串
        code = dedent('''
            from typing import Tuple, List, Optional
            from torch import Tensor
            from torch.jit.annotations import BroadcastingList2, \\
                BroadcastingList3
            import torch
            class FooModule(torch.jit.ScriptModule):
                @torch.jit.script_method
                def foo(self, x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                    return x, x
            instance = FooModule()
        ''')

        test_str = []
        # 对于每一个输入输出类型对，执行以下操作
        for pair in self.type_input_return_pairs():
            # 获取 Python 3 风格的代码，并使用 instance 对象进行格式化
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'instance')
            # 将函数的 schema 转换为字符串并添加到测试字符串列表中
            test_str.append(str(fn.foo.schema))
        # 断言预期的结果与剥离后的名称匹配
        self.assertExpectedStripMangled("\n".join(test_str) + "\n")

    # Python AST 前端，MyPy 风格的类型注释，Script 函数
    def test_annot_ast_mypy_fn(self):
        # 定义一个包含 Python 代码的字符串
        code = dedent('''
            import torch
            @torch.jit.script
            def foo(x, y):
                # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
                return x, x
        ''')

        test_str = []
        # 对于每一个输入输出类型对，执行以下操作
        for pair in self.type_input_return_pairs():
            # 获取 Python 3 风格的代码，并使用 foo 函数进行格式化
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'foo')
            # 将函数的 schema 转换为字符串并添加到测试字符串列表中
            test_str.append(str(fn.schema))
        # 断言预期的结果与剥离后的名称匹配
        self.assertExpected("\n".join(test_str) + "\n")

    # Python AST 前端，MyPy 风格的类型注释，Script 方法
    # 定义一个测试方法，用于测试 AST（抽象语法树）注解和 Mypy 类型检查
    def test_annot_ast_mypy_method(self):
        # 生成测试用例的代码段
        code = dedent('''
            import torch
            class FooModule(torch.jit.ScriptModule):
                @torch.jit.script_method
                def foo(self, x, y):
                    # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
                    return x, x
            instance = FooModule()
        ''')

        # 初始化空列表，用于存储每个测试用例的输出结果
        test_str = []
        # 对于每一对输入和返回类型，执行以下操作
        for pair in self.type_input_return_pairs():
            # 格式化代码并获取 Python3 代码对象
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'instance')
            # 获取 foo 方法的 schema 并将其转换为字符串，添加到结果列表中
            test_str.append(str(fn.foo.schema))
        # 断言预期的剥离混淆后的结果
        self.assertExpectedStripMangled("\n".join(test_str) + "\n")

    # 测试确保 "# type: ignore[*]" 在类型行中被支持并正确忽略
    def test_mypy_type_ignore(self):
        # 定义一个被 Torch JIT 脚本化的函数 foo，其中类型声明被忽略
        @torch.jit.script
        def foo(x):  # type: ignore
            return x

        # 定义另一个被 Torch JIT 脚本化的函数 bar，其中指定了忽略的重新定义规则
        @torch.jit.script
        def bar(x):  # type: ignore[no-redef]
            return x

    # 测试方法：测试方法转换为脚本时的类型转换
    def test_method_casts_script(self):
        # 定义需要进行的类型转换列表
        cast_types = [
            'byte', 'char', 'double', 'float', 'int', 'long', 'short'
        ]

        # 对于每种类型转换执行以下操作
        for cast_type in cast_types:
            # 使用 Torch JIT 编译单元定义一个字符串，该字符串包含类型转换函数
            cu = torch.jit.CompilationUnit(f'''
            def cast_to(x):
                return x.{cast_type}()
            ''')

            # 创建一个 3x4x5 的随机张量 x
            x = torch.rand(3, 4, 5) * 128
            # 对 cu 中定义的 cast_to 函数进行调用
            cu_result = cu.cast_to(x)
            # 获取 x 对象中相应类型转换方法的引用作为参考结果
            reference = getattr(x, cast_type)()
            # 断言 cu_result 和 reference 相等
            self.assertEqual(cu_result, reference)

    # 测试字符串前端中的 elif 语句
    def test_string_frontend_elif(self):
        # 定义一个 func 函数，该函数接受一个参数 niter，返回计算结果 rv
        code = '''
            def func(niter):
                # type: (int)
                rv = 0
                for i in range(niter):
                    if i % 3 == 0 and i % 5 == 0:
                        rv += 35
                    elif i % 3 == 0:
                        rv += 3
                    elif i % 5 == 0:
                        rv += 5
                    else:
                        rv += i
                return rv
        '''

        # 使用 checkScript 方法验证代码的脚本化结果，传入参数 (101,)
        self.checkScript(dedent(code), (101,))
    def test_module_parameters_and_buffers(self):
        # 创建一个大小为10x10的随机权重张量
        weights = torch.randn(10, 10)
        # 创建一个大小为10的随机偏置张量
        bias = torch.randn(10)
        # 创建另一个大小为10x10的随机权重张量
        weights2 = torch.randn(10, 10)
        # 创建另一个大小为10的随机偏置张量
        bias2 = torch.randn(10)

        # 定义一个测试用的线性模块
        class TestLinear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                # 使用空的参数张量初始化权重
                self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
                # 使用空的参数张量初始化偏置
                self.bias = torch.nn.Parameter(torch.empty(out_features))
                # 注册一个缓冲区 `counter`，初始化为全1的张量
                self.register_buffer('counter', torch.ones(out_features))
                # 重置模块的参数
                self.reset_parameters()

            def reset_parameters(self):
                # 使用 Kaiming 均匀分布初始化权重
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                # 如果偏置不为空，则使用均匀分布初始化偏置
                if self.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(self.bias, -bound, bound)

            def forward(self, input):
                # 执行线性变换，加上 `counter` 缓冲区的值
                return F.linear(input, self.weight, self.bias) + self.counter

        # 初始化一个脚本模块，多次使用上面定义的弱模块
        class Strong(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建两个 TestLinear 实例作为模块的成员变量
                self.fc1 = TestLinear(10, 10)
                self.fc1.weight = torch.nn.Parameter(weights)
                self.fc1.bias = torch.nn.Parameter(bias)
                self.fc2 = TestLinear(10, 10)
                self.fc2.weight = torch.nn.Parameter(weights2)
                self.fc2.bias = torch.nn.Parameter(bias2)

            @torch.jit.script_method
            def forward(self, x):
                # 执行模块的前向传播，计算结果为输入加上两个 fc1 和一个 fc2 的输出
                return x + self.fc1(x) + self.fc1(x) + self.fc2(x)

        # 创建 Strong 类的实例
        strong_mod = Strong()

        # 运行与模块相同的计算
        inp = torch.ones(10)
        # 创建一个线性层实例，使用上面定义的权重和偏置
        lin = torch.nn.Linear(10, 10)
        lin.weight = torch.nn.Parameter(weights)
        lin.bias = torch.nn.Parameter(bias)
        # 创建另一个线性层实例，使用上面定义的权重和偏置
        lin2 = torch.nn.Linear(10, 10)
        lin2.weight = torch.nn.Parameter(weights2)
        lin2.bias = torch.nn.Parameter(bias2)
        # 期望的计算结果
        expected_result = inp + (lin(inp) + torch.ones(10)) * 2 + lin2(inp) + torch.ones(10)

        # 断言 Strong 模块的输出与期望的结果相等
        self.assertEqual(strong_mod(inp), expected_result)
        # 断言模块的导出和导入
        self.assertExportImportModule(strong_mod, (inp,))
    def test_module_copying(self):
        # 定义一个简单的子模块，用于模拟神经网络模块的结构
        class Submodule(torch.nn.Module):
            # 前向传播函数，简单地将输入加上 100
            def forward(self, x):
                return x + 100

        # 定义一个较弱的神经网络模块，包含权重、偏置、缓冲区和一个子模块
        class Weak(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                # 初始化权重和偏置，以及注册一个缓冲区和子模块
                self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.ones(out_features))
                self.register_buffer("buffer", torch.ones(out_features))
                self.submodule = Submodule()

            # 前向传播函数，使用 F.linear 进行线性变换，同时加上缓冲区和子模块的输出
            def forward(self, x):
                return F.linear(x, self.weight, self.bias) \
                    + self.buffer + self.submodule(x)

        # 定义一个较强的神经网络模块，继承自 ScriptModule
        class Strong(torch.jit.ScriptModule):
            def __init__(self, weak):
                super().__init__()
                self.weak = weak

            # 使用 torch.jit.script_method 装饰器定义前向传播函数
            @torch.jit.script_method
            def forward(self, x):
                return self.weak(x)

        # 创建一个输入张量
        inp = torch.ones(5, 5) * 5
        # 实例化一个较弱的模块
        weak_mod = Weak(5, 5)
        # 实例化一个较强的模块，将较弱的模块作为参数传入
        strong_mod = Strong(weak_mod)

        # 断言较强模块的 weak 属性是否为 ScriptModule 类型
        self.assertTrue(isinstance(strong_mod.weak, torch.jit.ScriptModule))
        # 断言较弱模块本身不是 ScriptModule 类型
        self.assertFalse(isinstance(weak_mod, torch.jit.ScriptModule))

        # 断言较强模块中的权重和较弱模块中的权重是同一个对象
        self.assertIs(strong_mod.weak.weight, weak_mod.weight)
        # 断言较强模块中的缓冲区和较弱模块中的缓冲区是同一个对象
        self.assertIs(strong_mod.weak.buffer, weak_mod.buffer)
        # 断言较强模块中的子模块已经递归地转换为脚本形式
        # （即不是同一个 Python 对象，而是已经被转换为 Torch 脚本对象）
        self.assertIsNot(strong_mod.weak.submodule, weak_mod.submodule)

        # 修改较弱模块的权重数据，增加一定值
        weak_mod.weight.data += torch.ones(5, 5) * 100
        # 断言较强模块的输出与修改后的较弱模块的输出在数值上相近
        self.assertTrue(strong_mod(inp).allclose(weak_mod(inp)))

        # 尝试直接重新赋值权重参数，这种情况下不会被 Torch 脚本追踪到
        weak_mod.weight = torch.nn.Parameter(torch.ones(5, 5) * 100)
        # 断言较强模块的输出与修改后的较弱模块的输出在数值上不相近
        self.assertFalse(strong_mod(inp).allclose(weak_mod(inp)))

    def test_backend_cudnn_enabled(self):
        # 定义一个简单的 Torch 脚本函数，根据 cudnn 是否启用来选择不同的计算
        @torch.jit.script
        def fn(x):
            if torch.backends.cudnn.enabled:
                x = x + 2
            else:
                x = x + 3
            return x

    def test_inplace_add(self):
        # 定义一个 Python 函数，演示张量的原地加法操作
        def foo(a, b):
            # 计算 a 和 b 的和，并赋值给 c
            c = a + b
            # 对 c 进行原地加法操作
            c.add_(b)
            return c
        # 使用 self.checkScript 方法检查 foo 函数是否能够成功转换为 Torch 脚本
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_add_out(self):
        # 定义一个 Python 函数，演示使用 torch.add 函数进行张量加法，并将结果输出到预先分配的张量 e 中
        def foo(a, b):
            # 计算 a 和 b 的和，并赋值给 c
            c = a + b
            # 计算 2 * a，并将结果输出到 e 中
            e = 2 * a
            torch.add(c, b, out=e)
            return e
        # 使用 self.checkScript 方法检查 foo 函数是否能够成功转换为 Torch 脚本
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_tuple_error_msg(self):
        # 定义一个 Torch 脚本函数，接受一个任意类型的参数 t，并检查是否是元组类型，如果是则解构为 a 和 b，然后返回它们的和
        def fn(t: Any):
            if isinstance(t, tuple):
                a, b = t
            return a + b
        # 使用 self.assertRaisesRegexWithHighlight 方法断言转换过程中是否会抛出指定的异常和消息
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Provided tuple is not fully defined/refined", "t"):
            s = torch.jit.script(fn)

    def test_augmented_assign(self):
        # 定义一个 Python 函数，演示增强赋值操作符（+=, -=, /=, *=）对张量的操作
        def foo(a, b):
            # 对 a 执行增强赋值操作
            a += b
            # 对 a 执行减法并赋值
            a -= b
            # 对 a 执行除法并赋值
            a /= b
            # 对 a 执行乘法并赋值
            a *= b
            return a, b
        # 使用 self.checkScript 方法检查 foo 函数是否能够成功转换为 Torch 脚本
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))
    def test_ignored_props(self):
        class A(nn.Module):
            # 定义被忽略的属性列表，这些属性不会被 TorchScript 脚本化
            __jit_ignored_attributes__ = ["ignored", "ignored_return_val"]

            @property
            def ignored(self):
                # 如果被调用，抛出数值错误异常，不应该被调用
                raise ValueError("shouldn't be called")

            @property
            def ignored_return_val(self):
                # 返回固定值 1 的属性
                return 1

            @torch.jit.ignore
            def call(self):
                # 调用被忽略的属性 `ignored_return_val` 并返回其值
                return self.ignored_return_val

        # 将类 A 脚本化
        f = torch.jit.script(A())
        # 使用简单的方法检查是否没有错误
        self.assertTrue(isinstance(f, torch.jit.ScriptModule))
        # 使用 `call` 方法，验证返回值是否为属性 `ignored_return_val` 的值
        self.assertTrue(isinstance(f.call(), property))


    def test_pass(self):
        def foo(x):
            # type: (bool) -> int
            # 循环 3 次，什么也不做
            for _i in range(3):
                pass
            # 如果 x 为真，什么也不做
            if x:
                pass
            else:
                pass
            # 返回整数 3
            return 3

        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (True,))

    def test_lhs_indexing(self):
        def foo(a, b):
            # 克隆张量 a
            a = a.clone()
            # 修改张量 a 的第一个元素为 b
            a[0] = b
            return a
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_advanced_indexing_assignment(self):
        def foo(x, y):
            # 计算张量 x 的指数
            a = torch.exp(x)
            # 创建布尔张量 b，指示 x 中值为 1 的位置
            b = x == 1
            # 将张量 a 中 b 为真的位置赋值为张量 y 对应位置的值
            a[b] = y[b]
            return a
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.ones(4, 3), torch.ones(4, 3)))

    def test_lhs_advanced_indexing_augmented_assignment(self):
        def foo(x, y):
            # 计算张量 x 的指数
            a = torch.exp(x)
            # 创建布尔张量 b，指示 x 中值为 1 的位置
            b = x == 1
            # 将张量 a 中 b 为真的位置的值增加张量 y 对应位置的值
            a[b] += y[b]
            return a
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.ones(4, 3), torch.ones(4, 3)))

    def test_lhs_indexing_list(self):
        def foo(a, b):
            # 创建包含张量 a 的列表 ls
            ls = [a]
            # 修改列表 ls 的第一个元素为 b
            ls[0] = b
            return ls
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_inplace_copy_script(self):
        def foo(x):
            # 创建一个形状为 (3, 4) 的随机张量 a
            a = torch.rand(3, 4)
            # 将张量 a 复制为 x 的值，使用 in-place 操作
            a.copy_(x)
            return a
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.rand(3, 4),))

    def test_lhs_indexing_increment(self):
        def foo(a, b):
            # 将张量 a 的第一个元素加上 b 的值
            a[0] += b
            return a
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_indexing_increment_list(self):
        def foo(a, b):
            # 克隆张量 a
            a = a.clone()
            # 创建包含张量 a 和张量 b 的列表 ls
            ls = [a, b]
            # 将列表 ls 的第一个元素的值加上 b 的值
            ls[0] += b
            return ls
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_indexing_increment_list_prim(self):
        def foo():
            # 创建一个整数列表 ls
            ls = [1, 2, 3]
            # 将列表 ls 的第一个元素增加 5
            ls[0] += 5
            return ls
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, ())

    def test_lhs_indexing_multi(self):
        def foo(a, b):
            # 克隆张量 a
            a = a.clone()
            # 分别将 foo、a 的第一个元素和 bar 设置为 (1, b, 3)
            foo, a[0], bar = (1, b, 3)
            return foo, a, bar
        # 使用 `checkScript` 方法验证函数 `foo` 的 TorchScript 脚本化
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))
    def test_bool_dispatch(self):
        # 在 Torch 的脚本化环境下，禁用代码生成钩子以便测试
        with torch._jit_internal._disable_emit_hooks():  
            # 定义一个函数，接受一个 Tensor 参数，返回不带索引的最大池化结果
            def kwarg_false(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1, return_indices=False)
            # 使用脚本化环境检查上述函数
            self.checkScript(kwarg_false, (torch.randn(3, 3, 3),))

            # 定义一个函数，接受一个 Tensor 参数，返回带索引的最大池化结果
            def kwarg_true(x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                return F.max_pool1d(x, 1, 1, return_indices=True)
            # 使用脚本化环境检查上述函数
            self.checkScript(kwarg_true, (torch.randn(3, 3, 3),))

            # 定义一个函数，接受一个 Tensor 参数，返回不带索引的最大池化结果，并指定参数
            def full_kwarg_false(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1, ceil_mode=False, return_indices=False)
            # 使用脚本化环境检查上述函数
            self.checkScript(full_kwarg_false, (torch.randn(3, 3, 3),))

            # 定义一个函数，接受一个 Tensor 参数，返回带索引的最大池化结果，并指定参数
            def full_kwarg_true(x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                return F.max_pool1d(x, 1, 1, ceil_mode=False, return_indices=True)
            # 使用脚本化环境检查上述函数
            self.checkScript(full_kwarg_true, (torch.randn(3, 3, 3),))

            # 定义一个函数，接受一个 Tensor 参数，使用默认参数返回最大池化结果
            def use_default(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1)
            # 使用脚本化环境检查上述函数
            self.checkScript(use_default, (torch.randn(3, 3, 3),))

            # 定义一个函数，接受一个 Tensor 参数，返回不带索引的最大池化结果，并使用额外参数
            def arg_false(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1, 0, 1, False, False)
            # 使用脚本化环境检查上述函数
            self.checkScript(arg_false, (torch.randn(3, 3, 3),))

            # 定义一个函数，接受一个 Tensor 参数，返回带索引的最大池化结果，并使用额外参数
            def arg_true(x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                return F.max_pool1d(x, 1, 1, 0, 1, False, True)
            # 使用脚本化环境检查上述函数
            self.checkScript(arg_true, (torch.randn(3, 3, 3),))

    def test_infer_size(self):
        # 导入 Torch 中的 _infer_size 函数
        from torch._C import _infer_size

        # 定义一个函数，接受两个 Tensor 参数，返回它们的尺寸列表
        def fn(x, y):
            # type: (Tensor, Tensor) -> List[int]
            return _infer_size(x.size(), y.size())

        # 使用脚本化环境检查上述函数
        self.checkScript(fn, (torch.ones(2, 4, 2), torch.ones(2, 4, 2)))

    def test_hash(self):
        # 定义一个函数，接受一个函数和输入列表，测试这些输入的哈希结果是否符合预期
        def tester(fn, inputs):
            for x in inputs:
                for y in inputs:
                    if x == y:
                        self.assertEqual(fn(x), fn(y))
                    else:
                        self.assertNotEqual(fn(x), fn(y))

        # 定义一个脚本化函数，接受一个整数参数，返回其哈希值
        @torch.jit.script
        def int_hash(x):
            # type: (int) -> int
            return hash(x)

        # 定义一个脚本化函数，接受一个浮点数参数，返回其哈希值
        @torch.jit.script
        def float_hash(x):
            # type: (float) -> int
            return hash(x)

        # 定义一个脚本化函数，接受一个字符串参数，返回其哈希值
        @torch.jit.script
        def str_hash(x):
            # type: (str) -> int
            return hash(x)

        # 分别测试整数、浮点数和字符串的哈希值
        tester(int_hash, (20, 21, 22))
        tester(float_hash, (20.0, 21.00001, 22.443))
        tester(str_hash, ("", "hello", "a"))
    def test_id(self):
        # 断言运行时错误中包含特定文本 "Expected a value"
        with self.assertRaisesRegex(RuntimeError, "Expected a value"):
            # 使用 torch.jit.script 装饰器将 test_id_scalars 函数转换为 Torch 脚本
            @torch.jit.script
            def test_id_scalars():
                # 检查 id(2) 是否等于 id(None)，并返回结果
                return id(2) == id(None)

        # 使用 torch.jit.script 装饰器将 FooTest 类转换为 Torch 脚本
        @torch.jit.script
        class FooTest:
            def __init__(self, x):
                # 初始化对象的属性 foo
                self.foo = x

            def getFooTest(self):
                # 返回对象的属性 foo
                return self.foo

        # 使用 torch.jit.script 装饰器将 test_id_class_types 函数转换为 Torch 脚本
        @torch.jit.script
        def test_id_class_types():
            # 创建两个 FooTest 类型的对象，使用不同的张量作为参数
            obj1 = FooTest(torch.tensor(3))
            obj2 = FooTest(torch.tensor(2))
            # 断言两个对象不是同一个对象（不同的引用）
            assert obj1 is not obj2
            # 断言两个对象的 id 不相同
            assert id(obj1) != id(obj2)
            # 断言对象的 id 不等于 None 的 id
            assert id(obj1) != id(None)
            # 返回 True
            return True

        # 断言 test_id_class_types 函数返回 True
        self.assertTrue(test_id_class_types())

    def test_mutable_dce(self):
        # 使用 torch.jit.script 装饰器将 foo 函数转换为 Torch 脚本
        @torch.jit.script
        def foo():
            # 创建一个张量 a，并加上一个随机张量
            a = torch.rand(2, 3)
            a += torch.rand(2, 3)
            # 创建一个张量 b，并加上一个随机张量
            b = torch.rand(2, 3)
            b += torch.rand(2, 3)
            # 返回张量 a
            # b 应该被清除，但 a 不应该
            return a

        # 使用 FileCheck 检查 foo 函数的图形中 "aten::rand" 出现的次数为 2 次，"aten::add" 出现的次数为 1 次
        FileCheck().check_count("aten::rand", 2, exactly=True) \
            .check_count("aten::add", 1, exactly=True).run(str(foo.graph))

    def test_mutable_dce_block(self):
        # 使用 torch.jit.script 装饰器将 foo 函数转换为 Torch 脚本
        @torch.jit.script
        def foo():
            # 创建一个张量 a，并加上一个随机张量
            a = torch.rand(2, 3)
            a += torch.rand(2, 3)
            # 创建一个张量 b，并加上一个随机张量
            b = torch.rand(2, 3)
            if bool(a > torch.zeros(2, 3)):
                # 如果条件成立，则 b 加上一个随机张量，a 加上一个随机张量
                b += torch.rand(2, 3)
                a += torch.rand(2, 3)
            # 返回张量 b
            # a 应该被清除，但 b 不应该
            return b

        # 使用 FileCheck 检查 foo 函数的图形中包含 "prim::If"，"aten::rand" 出现的次数为 1 次
        FileCheck().check("prim::If").check_count("aten::rand", 1, exactly=True) \
            .run(str(foo.graph))

    def test_mutable_dce_graph_input(self):
        # 使用 torch.jit.script 装饰器将 foo 函数转换为 Torch 脚本
        @torch.jit.script
        def foo(a):
            # 张量 a 加上一个随机张量
            a += torch.rand(2, 3)
            # 尽管在输出中没有使用，但不应清除 `a`
        
        # 使用 FileCheck 检查 foo 函数的图形中包含 "aten::rand" 和 "aten::add"
        FileCheck().check("aten::rand").check("aten::add").run(str(foo.graph))

    def test_mutable_dce_list(self):
        # 使用 torch.jit.script 装饰器将 foo 函数转换为 Torch 脚本
        @torch.jit.script
        def foo(a):
            # 创建一个空列表 l，并将参数 a 添加到列表中
            l = []
            l.append(a)
            # 从列表中获取第一个元素，并赋值给变量 c
            c = l[0]
            # 创建一个张量 b，并加上一个随机张量
            b = torch.rand(2, 3)
            c += torch.rand(2, 3)
            # 返回张量 b
            # c 不会被清除，因为有通配符 + 的变异
            return b

        # 使用 FileCheck 检查 foo 函数的图形中 "aten::rand" 出现的次数为 2 次
        FileCheck().check_count("aten::rand", 2, exactly=True).run(str(foo.graph))

    def test_mutable_dce_loop(self):
        # 使用 torch.jit.script 装饰器将 foo 函数转换为 Torch 脚本
        @torch.jit.script
        def foo(a):
            # 创建一个空列表 l，并将参数 a 添加到列表中
            l = []
            l.append(a)
            i = 0
            # 创建一个张量 b
            b = torch.rand(2, 3)
            while i < 1:
                # 创建一个随机张量 dead
                dead = torch.rand(2, 3)
                # 从列表中获取第一个元素，并赋值给变量 c
                c = l[0]
                c += torch.rand(2, 3)
                i += 1
            # 返回张量 b
            # c 应该被清除，但是没有随机张量
            return b

        # 使用 FileCheck 检查 foo 函数的图形中包含 "prim::Loop"，"aten::rand" 出现的次数为 1 次
        FileCheck().check("prim::Loop").check_not("aten::rand").check("aten::__getitem__") \
            .check_count("aten::rand", 1, exactly=True).run(str(foo.graph))
    def test_mutable_dce_indirect_wildcards(self):
        # 定义测试函数，测试可变删除元素的间接通配符
        def fn():
            # 创建一个2x3的全1张量
            x = torch.ones(2, 3)
            # 将x展平为一维张量x_1
            x_1 = x.view(-1)
            # 创建一个空列表l，并将x_1添加到列表末尾
            l = []
            l.append(x_1)
            # 获取列表l中的第一个元素，并赋值给x_view
            x_view = l[0]
            # 对原始张量x执行原地加法操作，向每个元素添加1
            x.add_(torch.ones(2, 3))
            # 返回x_view，即原始张量x的视图
            return x_view
        # 调用测试函数，并使用self.checkScript进行检查
        self.checkScript(fn, ())

    def test_mutable_dce_indirect_wildcard_write(self):
        # 定义测试函数，测试可变删除元素的间接通配符写入操作
        def fn():
            # 创建一个空列表indexes，用于存储张量
            indexes = torch.jit.annotate(List[Tensor], [])
            # 创建一个全零的长度为10的整数类型张量word_ids
            word_ids = torch.zeros(10, dtype=torch.int32)
            # 将索引为1的位置设置为1
            word_ids[1] = 1
            # 将word_ids张量添加到indexes列表末尾
            indexes.append(word_ids)

            # 返回word_ids张量
            return word_ids
        # 调用测试函数，并使用self.checkScript进行检查
        self.checkScript(fn, ())

    def test_mutable_dce_wildcards(self):
        # 定义测试函数，测试可变删除元素的通配符
        def fn():
            # 创建一个2x3的全1张量x
            x = torch.ones(2, 3)
            # 创建一个空列表l，并将张量x添加到列表末尾
            l = []
            l.append(x)
            # 获取列表l中的第一个元素，并赋值给x_view
            x_view = l[0]
            # 对原始张量x执行原地加法操作，向每个元素添加1
            x.add_(torch.ones(2, 3))
            # 返回x_view，即原始张量x的视图
            return x_view

        # 调用测试函数，并使用self.checkScript进行检查，使用简单的性能分析模式
        self.checkScript(fn, (), profiling=ProfilingMode.SIMPLE)

    def test_cpp_function_tensor_str(self):
        # 创建一个2x2的随机张量x
        x = torch.randn(2, 2)
        # 创建一个2x2的随机张量scale，并设置requires_grad为True
        scale = torch.randn(2, 2, requires_grad=True)
        # 创建一个2x2的随机张量shift，并设置requires_grad为True
        shift = torch.randn(2, 2, requires_grad=True)

        # 定义一个使用torch.jit.script装饰的脚本函数fn，对输入进行线性变换
        @torch.jit.script
        def fn(x, scale, shift):
            return scale * x + shift

        # 使用self.capture_stdout()捕获标准输出，并打印fn函数的结果
        with self.capture_stdout() as captured:
            print(fn(x, scale, shift))

    def test_string_index(self):
        # 定义一个函数fn，接受一个字符串x作为参数，返回字符串的第三个字符和倒数第一个字符
        def fn(x):
            # type: (str)
            return x[2], x[-1]

        # 调用测试函数，并使用self.checkScript进行检查，传入参数"abcde"
        self.checkScript(fn, ("abcde",))

    def test_ord(self):
        # 定义一个函数fn，接受一个字符串x作为参数，返回其ASCII码值
        def fn(x):
            # type: (str) -> int
            return ord(x)

        # 使用self.checkScript进行检查，传入参数"h"和"y"
        self.checkScript(fn, ("h"))
        self.checkScript(fn, ("y"))

        # 定义一个函数index_str_to_tensor，接受一个字符串s作为参数，返回其ASCII码值对应的张量
        def index_str_to_tensor(s):
            # type: (str) -> Tensor
            return torch.tensor(ord(s))  # noqa: T484

        # 获取字符串'\u00a3'的编码的前一个字节，并使用self.checkScript进行检查
        s = '\u00a3'.encode()[:1]
        self.checkScript(index_str_to_tensor, (s,))

    def test_chr(self):
        # 定义一个函数fn，接受一个整数x作为参数，返回其对应的字符
        def fn(x):
            # type: (int) -> str
            return chr(x)

        # 使用self.checkScript进行检查，传入参数1和97
        self.checkScript(fn, (1,))
        self.checkScript(fn, (97,))

    def test_round(self):
        # 定义一个函数round_float，接受一个浮点数x作为参数，返回其四舍五入的结果
        def round_float(x):
            # type: (float) -> float
            return round(x)

        # 定义一个函数round_int，接受一个整数x作为参数，返回其四舍五入的结果
        def round_int(x):
            # type: (int) -> float
            return round(x)

        # 使用self.checkScript进行检查，传入参数1.5和2
        self.checkScript(round_float, (1.5,))
        self.checkScript(round_int, (2,))

    def test_convert_base(self):
        # 定义三个函数，将整数x转换为二进制、八进制和十六进制字符串
        def test_hex(x):
            # type: (int) -> str
            return hex(x)

        def test_oct(x):
            # type: (int) -> str
            return oct(x)

        def test_bin(x):
            # type: (int) -> str
            return bin(x)

        # 定义一个整数列表numbers，包含多个测试数字
        numbers = [-1000, -10, 0, 1, 10, 2343]
        # 遍历numbers列表，并使用self.checkScript对每个数字执行二进制、八进制和十六进制转换函数的检查
        for n in numbers:
            self.checkScript(test_bin, (n,))
            self.checkScript(test_oct, (n,))
            self.checkScript(test_hex, (n,))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: TemporaryFileName support for Windows or Sandcastle")
    def test_string_slicing(self):
        # 定义一个函数 fn1，接收一个字符串参数 x，返回 x 的子串 x[1:3]
        def fn1(x):
            # type: (str) -> str
            return x[1:3]

        # 定义一个函数 fn2，接收一个字符串参数 x，返回 x 的子串 x[-1:3]
        def fn2(x):
            # type: (str) -> str
            return x[-1:3]

        # 定义一个函数 fn3，接收一个字符串参数 x，返回 x 的子串 x[3:1]
        def fn3(x):
            # type: (str) -> str
            return x[3:1]

        # 定义一个函数 fn4，接收一个字符串参数 x，返回 x 的子串 x[3:100]
        def fn4(x):
            # type: (str) -> str
            return x[3:100]

        # 使用 checkScript 方法测试 fn1 函数，传入参数 ("abcdefghi",)
        self.checkScript(fn1, ("abcdefghi",))
        # 使用 checkScript 方法测试 fn2 函数，传入参数 ("abcdefghi",)
        self.checkScript(fn2, ("abcdefghi",))
        # 使用 checkScript 方法测试 fn3 函数，传入参数 ("abcdefghi",)
        self.checkScript(fn3, ("abcdefghi",))
        # 使用 checkScript 方法测试 fn4 函数，传入参数 ("abcdefghi",)
        self.checkScript(fn4, ("abcdefghi",))

    def test_early_return_closure(self):
        # 定义一个包含嵌套函数的字符串代码块 code
        code = dedent('''
            def tanh(self):
                output = torch.tanh(self)
                # 定义一个嵌套函数 backward(grad_output)，暂时不实现具体逻辑
                def backward(grad_output):
                    pass
                # 返回 output 和 backward 函数
                return output, backward
        ''')
        # 使用 CompilationUnit 类编译字符串代码块
        cu = torch.jit.CompilationUnit(code)
        # 获取编译单元 cu 中 tanh 函数的计算图 g
        g = cu.tanh.graph
        # 使用 FileCheck 对象检查计算图 g，验证闭包数量为 2，存在 NoneType = prim::Constant，下一个指令为 return
        FileCheck().check_count("prim::Closure_0", 2).check("NoneType = prim::Constant") \
                   .check_next("return").run(g)

        # 定义另一个包含嵌套函数的字符串代码块 code
        code = dedent('''
            def tanh(self):
                output = torch.tanh(self)
                # 定义一个嵌套函数 backward(grad_output)
                def backward(grad_output):
                    a = 1
                    # 如果 output 存在，则返回 1
                    if output:
                        return 1
                    else:
                        a = 2
                    # 否则返回 a 的值
                    return a
                # 返回 output 和 backward 函数
                return output, backward
        ''')
        # 使用 CompilationUnit 类编译字符串代码块
        cu = torch.jit.CompilationUnit(code)
        # 获取编译单元 cu 中 tanh 函数的计算图 g
        g = cu.tanh.graph
        # 使用 FileCheck 对象检查计算图 g，验证闭包数量为 2，存在 int = prim::If
        FileCheck().check_count("prim::Closure_0", 2).check("int = prim::If") \
                   .run(g)

        # 定义另一个包含嵌套函数的字符串代码块 code
        code = dedent('''
            def loop_in_closure(self):
                output = torch.tanh(self)
                # 定义一个嵌套函数 backward(grad_output)
                def backward(grad_output):
                    # 循环 3 次，每次返回 1
                    for i in range(3):
                        return 1
                    # 循环结束后返回 4
                    return 4
                # 返回 output 和 backward 函数
                return output, backward
        ''')
        # 使用 CompilationUnit 类编译字符串代码块
        cu = torch.jit.CompilationUnit(code)
        # 使用 FileCheck 对象检查编译单元 cu 中的计算图，验证存在 prim::Closure、prim::Loop 和 2 次 prim::If
        fc = FileCheck()
        fc.check("prim::Closure").check("(Tensor, NoneType) = prim::TupleConstruct")
        fc.check("prim::Closure").check("prim::Loop").check_count("prim::If", 2)
        fc.run(cu.loop_in_closure.graph)

        # 定义另一个包含嵌套函数的字符串代码块 code
        code = dedent('''
            def tanh(self):
                output = torch.tanh(self)
                # 定义一个嵌套函数 backward(grad_output)
                def backward(grad_output):
                    # 如果 1 等于 1，则返回 1
                    if 1 == 1:
                        return 1
                    else:
                        return 1.
                # 返回 output 和 backward 函数
                return output, backward
        ''')
        # 使用 self.assertRaisesRegex 检查 RuntimeError 异常，确保代码块返回 int 类型值但实际返回了 float 类型值
        with self.assertRaisesRegex(RuntimeError, "returned a value of type int but"):
            cu = torch.jit.CompilationUnit(code)

    @_inline_everything
    def test_early_return_fork_join(self):
        # 定义一个 TorchScript 函数 foo，根据输入张量 x 的维度返回不同的结果
        @torch.jit.script
        def foo(x):
            # 如果 x 是二维张量，则返回其取负和原始张量
            if x.dim() == 2:
                return torch.neg(x), x
            else:
                # 否则返回其取负和加一的结果
                return torch.neg(x), x + 1

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)

        # 定义另一个 TorchScript 函数 wait_script，其中包含 fork 和 wait 操作
        @torch.jit.script
        def wait_script(x):
            # 使用 torch.jit._fork 在新的线程中执行 foo 函数
            fut = torch.jit._fork(foo, x)
            # 直接调用 foo 函数，得到 y_hat
            y_hat = foo(x)
            # 等待并获取 fork 操作的结果
            y = torch.jit._wait(fut)
            # 返回两次 foo 调用的结果
            return y, y_hat

        # 在 wait_script 的图中运行 FileCheck，验证包含特定操作的图结构
        FileCheck().check("with prim::fork").check("prim::If").check("return")\
                   .run(wait_script.graph)

    def test_early_return_type_refinement(self):
        # 定义一个 TorchScript 函数 test，根据输入的可选整数 x 的值进行处理
        @torch.jit.script
        def test(x):
            # type: (Optional[int]) -> int
            # 如果 x 是 None，则返回 1
            if x is None:
                return 1
            else:
                # 否则返回 x 的值
                return x

        # 验证当输入为 None 时返回 1，当输入为 2 时返回 2
        self.assertEqual(test(None), 1)
        self.assertEqual(test(2), 2)

    def test_exception_exits_closure(self):
        # 定义一个字符串代码块，包含一个 no_return_func 函数和一个嵌套的 backward 函数
        code = dedent('''
            def no_return_func(self):
                # type: (Tensor) -> Tensor
                # 对输入张量执行 tanh 操作
                output = torch.tanh(self)
                # 定义 backward 函数，用于处理梯度输出，但会抛出 RuntimeError 异常
                def backward(grad_output):
                    raise RuntimeError("Hi")
        ''')
        # 验证在 CompilationUnit 中编译时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "does not return along all"):
            cu = torch.jit.CompilationUnit(code)

        # 定义另一个字符串代码块，包含一个 test_exit_pair_reset 函数
        code = dedent('''
            def test_exit_pair_reset(x):
                # type: (int) -> int
                # 如果 x 大于 0，则进行条件分支
                if x > 0:
                    # 初始化变量 a 为 0
                    a = 0
                    # 定义嵌套函数 backward，用于处理梯度输出，但会抛出 RuntimeError 异常
                    def backward(grad_output):
                        raise RuntimeError("Hi")
                    # 增加变量 a 的值
                    a = a + 1
                else:
                    # 如果 x 不大于 0，则直接返回 x
                    return x
                # 返回 a + 1 的结果
                return a + 1
        ''')
        # 编译代码块并获取其中的 test_exit_pair_reset 函数
        func = torch.jit.CompilationUnit(code).test_exit_pair_reset
        # 验证当输入为 1 时返回 2，当输入为 -1 时返回 -1
        self.assertEqual(func(1,), 2)
        self.assertEqual(func(-1,), -1)
        # 验证函数图中有且仅有一个 prim::If 操作
        FileCheck().check_count("prim::If", 1, exactly=True).run(func.graph)
    def test_non_final_return(self):
        # 测试函数：检查非最终返回的情况

        def simple(x):
            # 简单函数：根据条件返回不同的值
            if bool(x > 3):
                return x + 1
            else:
                return x + 2
            # 永远不会执行到这里的异常抛出
            raise RuntimeError("nope")

        def nest(x):
            # 嵌套函数：多层条件判断和返回
            x = x + 1
            if bool(x > 3):
                if bool(x > 4):
                    x += 1
                return x + 1
            else:
                return x + 2

        def early_ret(x):
            # 提前返回函数：在条件判断后直接返回
            x = x + 1
            if bool(x > 3):
                return x + 1
            x = x + 1
            return x + 2

        def nest_early_ret(x):
            # 嵌套提前返回函数：多层条件判断后直接返回
            x = x + 1
            if bool(x > 3):
                if bool(x > 4):
                    return x + 2
                return x + 1
            x = x + 1
            return x + 2

        def not_early_ret(x):
            # 非提前返回函数：根据条件逐步构建返回值
            s = ""
            if bool(x > 3):
                if bool(x > 4):
                    return 1, s
                s += "foo"
            else:
                s += "5"
            s += "hi"
            return 7, s

        def not_total_ret(x):
            # 非完全返回函数：根据条件逐步构建返回值，可能不完全返回
            s = ""
            if bool(x > 3):
                if bool(x > 4):
                    return 1, s
                else:
                    return 2, s
            else:
                s += "5"
            return 7, s

        # 使用 self.checkScript 检查多个函数的脚本化结果
        for i in range(3):
            for func in [simple, nest, early_ret, nest_early_ret, not_early_ret,
                         not_total_ret]:
                self.checkScript(func, (torch.tensor(2.5 + i),))

        def vars_used_after_ret(x):
            # type: (int) -> int
            # 在返回后使用变量：根据输入值返回或使用其他变量
            if x == 0:
                return x
            else:
                y = 2
                z = 3
            return x + y * z

        # 使用 self.checkScript 检查使用了返回后变量的函数
        self.checkScript(vars_used_after_ret, (1,))
        self.checkScript(vars_used_after_ret, (0,))

        def complicated(x):
            # type: (int) -> int
            # 复杂条件函数：根据多层条件返回不同的值
            if x:
                if x == 2:
                    return 1
                    assert 1 == 2
                else:
                    if x == 3:
                        return 2
                        assert 1 == 2
                    else:
                        a = 2
                        b = 3
            else:
                a = 4
                b = 1
            return a + b
            assert 1 == 2

        # 使用 self.checkScript 检查复杂条件函数的脚本化结果
        for i in range(4):
            self.checkScript(complicated, (i,))
    def test_partial_returns(self):
        # 检查未完全返回的情况是否会引发 RuntimeError 异常，并包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, "does not return along all"):
            @torch.jit.script
            def no_ret():
                # type: () -> int
                pass

        # 检查部分条件下返回的情况是否会引发 RuntimeError 异常，并包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, "does not return along all"):
            @torch.jit.script
            def partial(x):
                # type: (Tensor) -> int
                if x:
                    return 1

        # 检查带有 Optional 类型注释但没有返回的情况是否会引发 RuntimeError 异常，并包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, "does not return along all"):
            @torch.jit.script
            def typed_none():
                # type: () -> Optional[int]
                pass

        # 没有返回语句的函数定义，期望返回 None
        @torch.jit.script
        def none_ret():
            pass

        # 断言调用 none_ret() 后返回 None
        self.assertIs(none_ret(), None)
        
        # 使用 FileCheck 检查 none_ret 的计算图，期望包含 ": None"
        FileCheck().check(": None").run(none_ret.graph)

    def test_early_returns_loops(self):
        # 嵌套的 while 循环和条件返回的示例函数
        def nest_while_ret(x):
            # type: (int) -> int
            y = 4
            while x < 4:
                if x < 3:
                    return y
                else:
                    y = y + 1
                    break
                y = y + 2
            y = y + 1
            return y

        # 使用 checkScript 方法检查 nest_while_ret 函数的脚本化版本
        self.checkScript(nest_while_ret, (2,))
        self.checkScript(nest_while_ret, (3,))
        self.checkScript(nest_while_ret, (4,))

        # 带有 for 循环和条件返回的示例函数
        def loop_ret(x, y):
            # type: (int, int) -> (int)
            i = 0
            for i in range(x):
                if x == y:
                    return x + y
                i = i + y
            i = i - 1
            return i

        # 使用 checkScript 方法检查 loop_ret 函数的脚本化版本
        self.checkScript(loop_ret, (3, 3))
        self.checkScript(loop_ret, (2, 3))
        self.checkScript(loop_ret, (3, 1))

        # 带有 for 循环和一定条件下返回的示例函数
        def test_will_ret(y):
            # type: (int) -> int
            for i in range(y):
                return 2
            return 1

        # 使用 checkScript 方法检查 test_will_ret 函数的脚本化版本
        self.checkScript(test_will_ret, (0,))
        self.checkScript(test_will_ret, (1,))

        # 嵌套的 for 循环和条件返回的示例函数
        def test_loop_nest_ret(y):
            # type: (int) -> int
            for i in range(y):
                for i in range(y - 2):
                    return 10
                return 5
            return 0

        # 使用 checkScript 方法检查 test_loop_nest_ret 函数的脚本化版本
        self.checkScript(test_loop_nest_ret, (0,))
        self.checkScript(test_loop_nest_ret, (1,))
        self.checkScript(test_loop_nest_ret, (2,))
    def test_nn_init(self):
        # 定义测试用例元组列表，每个元组包含初始化函数名、参数生成函数和类型描述字符串
        tests = (
            ('constant_', (lambda: (torch.ones(2, 2), 2.5)), "Tensor, float"),
            ('ones_', (lambda: (torch.ones(2, 2),)), "Tensor"),
            ('zeros_', (lambda: (torch.ones(2, 2),)), "Tensor"),
            ('uniform_', (lambda: (torch.ones(2, 2),)), "Tensor"),
            ('normal_', (lambda: (torch.ones(2, 2),)), "Tensor"),
            ('xavier_normal_', (lambda: (torch.ones(2, 2),)), "Tensor"),
            ('xavier_uniform_', (lambda: (torch.ones(2, 2),)), "Tensor"),
        )

        # 遍历测试用例
        for name, args_fn, type_str in tests:
            # 构建测试函数的参数字符串
            arg_str = ', '.join([chr(i + ord('a')) for i in range(len(args_fn()))])

            # 构建测试函数的代码块
            code = dedent('''
                def test({arg_str}):
                    # type: ({type_str})
                    return torch.nn.init.{name}({arg_str})
            ''').format(arg_str=arg_str, type_str=type_str, name=name)
            cu = torch.jit.CompilationUnit(code)

            # 比较函数的输出结果
            init_fn = getattr(torch.nn.init, name)
            script_out = self.runAndSaveRNG(cu.test, args_fn())
            eager_out = self.runAndSaveRNG(init_fn, args_fn())
            self.assertEqual(script_out, eager_out)

            # 检查生成的图是否没有 PythonOp
            FileCheck().check_not("prim::PythonOp").run(cu.test.graph)

    def test_nn_init_generator(self):
        # 定义初始化函数名列表
        init_fns = (
            'uniform_', 'normal_', 'xavier_normal_', 'xavier_uniform_',
        )

        # 遍历初始化函数名列表
        for name in init_fns:
            # 构建测试函数的代码块
            code = dedent('''
                def test(tensor, generator):
                    # type: (Tensor, Generator)
                    return torch.nn.init.{name}(tensor, generator=generator)
            ''').format(name=name)
            cu = torch.jit.CompilationUnit(code)

            # 比较函数的输出结果
            init_fn = getattr(torch.nn.init, name)

            torch.manual_seed(1)

            g = torch.Generator()
            g.manual_seed(2023)
            script_out = cu.test(torch.ones(2, 2), g)

            # 更改默认生成器的种子以确保使用提供的生成器
            torch.manual_seed(2)

            g = torch.Generator()
            g.manual_seed(2023)
            eager_out = init_fn(torch.ones(2, 2), generator=g)

            self.assertEqual(script_out, eager_out)

            # 检查生成的图是否没有 PythonOp
            FileCheck().check_not("prim::PythonOp").run(cu.test.graph)
    # 定义一个测试函数 test_early_return_rewrite
    def test_early_return_rewrite(self):
        # 定义内部函数 test_foo，参数 x 是一个布尔值
        def test_foo(x: bool):
            # 如果 x 为 True，则返回 1
            if x:
                return 1
            # 否则返回 2
            return 2
        
        # 调用 self.checkScript 方法测试 test_foo 函数，传入参数 (True)
        self.checkScript(test_foo, (True,))
        # 再次调用 self.checkScript 方法测试 test_foo 函数，传入参数 (False)
        self.checkScript(test_foo, (False,))
        # 使用 FileCheck().check_count 方法验证 torch.jit.script(test_foo).graph 中 prim::If 的数量为 1
        FileCheck().check_count("prim::If", 1, exactly=True).run(torch.jit.script(test_foo).graph)

        # 定义内部函数 test_multiple，参数 x 是一个整数
        def test_multiple(x: int):
            # 如果 x 等于 5，则返回 x*x
            if x == 5:
                return x * x
            else:
                # 否则 y 是 2*x
                y = 2 * x
            
            # z 是 y 的两倍
            z = y * 2
            # 如果 z 等于 8，则返回 1
            if z == 8:
                return 1
            
            # 如果 z 不等于 16
            if z != 16:
                # z 减去 2
                z = z - 2
                # 定义 abc 为 4
                abc = 4
            else:
                # 否则返回 3
                return 3
            
            # z 等于 z 乘以 abc
            z = z * abc
            # 返回 z 的立方
            return z * z * z
        
        # 分别使用 self.checkScript 方法测试 test_multiple 函数，传入不同参数
        self.checkScript(test_multiple, (5,))
        self.checkScript(test_multiple, (2,))
        self.checkScript(test_multiple, (4,))
        self.checkScript(test_multiple, (3,))
        self.checkScript(test_multiple, (10,))
        
        # 获取 torch.jit.script(test_multiple) 的图形表示
        graph = torch.jit.script(test_multiple).graph
        # 使用 FileCheck().check_count 方法验证图中 prim::If 的数量为 3
        FileCheck().check_count("prim::If", 3, exactly=True).run(graph)

    # 定义一个测试函数 test_is_scripting_metacompile
    def test_is_scripting_metacompile(self):
        # 使用 torch.jit.script 装饰器定义函数 foo
        @torch.jit.script
        def foo():
            # 如果 torch.jit.is_scripting() 返回 True，则返回 1
            if torch.jit.is_scripting():
                return 1
            else:
                # 否则打印 "hello" + 2，这部分不会被编译
                print("hello") + 2  # will not be compiled
        
        # 断言调用 foo() 返回结果为 1
        self.assertEqual(foo(), 1)

    # 定义一个测试函数 test_boolean_literal_constant_metacompile
    def test_boolean_literal_constant_metacompile(self):
        # 定义一个继承自 torch.nn.Module 的模块 Mod
        class Mod(torch.nn.Module):
            # 指定 __constants__ 列表包含 'val'
            __constants__ = ['val']

            # 初始化方法，接收参数 val
            def __init__(self, val):
                super().__init__()
                self.val = val

            # 前向传播方法
            def forward(self):
                # 如果 self.val 为 True，则返回 1
                if self.val:
                    return 1
                else:
                    # 否则返回字符串 "2"
                    return "2"
        
        # 分别使用 self.checkModule 方法测试 Mod 类的实例，传入不同参数
        self.checkModule(Mod(True), ())
        self.checkModule(Mod(False), ())

        # 使用 torch.jit.script 装饰器定义函数 foo
        @torch.jit.script
        def foo():
            # 如果条件 True 恒为真，则返回 1
            if True:
                return 1
            else:
                # 否则返回字符串 "2"
                return "2"
        
        # 断言调用 foo() 返回结果为 1
        self.assertEqual(foo(), 1)

    # 定义一个测试函数 test_assert_is_scripting_metacompile
    def test_assert_is_scripting_metacompile(self):
        # 定义函数 foo
        def foo():
            # 断言 torch.jit.is_scripting() 返回 False，否则抛出 "TestErrorMsg" 异常
            assert not torch.jit.is_scripting(), "TestErrorMsg"
            # 打印 "hello" + 2，这部分不会被编译
            print("hello") + 2  # will not be compiled
        
        # 使用 torch.jit.script 将 foo 函数转换为脚本
        f = torch.jit.script(foo)
        # 使用 assertRaisesRegex 验证调用 f() 抛出 torch.jit.Error 异常，并包含 "TestErrorMsg" 字符串
        with self.assertRaisesRegex(torch.jit.Error, "TestErrorMsg"):
            f()
    def test_isinstance_metacompile(self):
        @torch.jit.script
        def test_primitive_type(x):
            # 声明函数签名，接受一个整数参数 x，返回一个整数
            # 如果 x 是整数，返回 x + 1
            if isinstance(x, int):
                return x + 1
            else:
                return x - 1

        # 测试函数 test_primitive_type，期望输入 1 返回 2
        self.assertEqual(test_primitive_type(1), 2)
        # 使用 assertRaisesRegex 测试异常情况，期望捕获到异常并匹配指定字符串
        with self.assertRaisesRegex(Exception, "Expected a value of type"):
            test_primitive_type(1.5)

        _MyNamedTuple = namedtuple('_MyNamedTuple', ['value'])

        @torch.jit.script
        def test_non_primitive_types(x):
            # 声明函数签名，接受一个 _MyNamedTuple 类型的参数 x，返回一个 Tensor
            # 这里检查了 1 是否是 _MyNamedTuple 类型（这里可能是一个错误）
            if isinstance(1, _MyNamedTuple):
                return 10

            # 检查 x 是否是 _MyNamedTuple 类型，如果是，返回 x.value + 1
            # 否则返回 1
            if isinstance(x, _MyNamedTuple):
                return x.value + 1
            else:
                return 1

        # 测试函数 test_non_primitive_types，期望输入 _MyNamedTuple(value=torch.tensor(5.0)) 返回 torch.tensor(6.0)
        out = test_non_primitive_types(_MyNamedTuple(value=torch.tensor(5.0)))
        self.assertEqual(out, torch.tensor(6.0))

    def test_namedtuple_type_inference(self):
        _AnnotatedNamedTuple = NamedTuple('_NamedTupleAnnotated', [('value', int)])  # noqa: UP014
        _UnannotatedNamedTuple = namedtuple('_NamedTupleUnAnnotated', ['value'])

        def test_check_named_tuple_value():
            # 创建一个 _AnnotatedNamedTuple 对象，value 为 1
            named_tuple = _AnnotatedNamedTuple(1)
            return named_tuple.value

        # 使用 checkScript 检查函数 test_check_named_tuple_value 的脚本化版本
        self.checkScript(test_check_named_tuple_value, ())

        def test_error():
            # 创建一个 _UnannotatedNamedTuple 对象，value 为 1，预期会抛出异常
            return _UnannotatedNamedTuple(1)

        # 使用 assertRaisesRegex 测试异常情况，期望捕获到特定异常并匹配指定字符串
        with self.assertRaisesRegex(RuntimeError, r"Expected a value of type \'Tensor \(inferred\)\' "
                                                  r"for argument \'value\' but instead found type \'int\'."):
            # 对 test_error 函数进行脚本化，预期会抛出异常
            torch.jit.script(test_error)

    def test_namedtuple_default_values_simple_type(self):

        class Point(NamedTuple):
            x: Optional[int] = None
            y: int = 2

        # 将 Point 类全局化
        make_global(Point)

        class M(torch.nn.Module):
            def forward(self, point: Point):
                return point

        # 创建一个 Point 对象 p，x=3，y=2
        p = Point(x=3, y=2)

        # 使用 checkModule 测试 M() 模块的正常和默认参数点 p 的情况
        self.checkModule(M(), (p,))
        self.checkModule(M(), (Point(),))

        # 对 M 类进行脚本化，获取其图形表示，并使用 FileCheck 检查其中的命名元组默认值
        m = torch.jit.script(M())

        FileCheck().check(r"NamedTuple(x : int? = None, y : int = 2))")   \
                   .run(m.graph)

    def test_namedtuple_default_values_missing(self):

        class Point(NamedTuple):
            x: Optional[int]
            y: int
            z: int = 3

        # 将 Point 类全局化
        make_global(Point)

        class M(torch.nn.Module):
            def forward(self, point: Point):
                return point

        # 创建两个 Point 对象，p1: x=3, y=2；p2: x=3, y=2, z=1
        p1 = Point(x=3, y=2)
        p2 = Point(x=3, y=2, z=1)

        # 使用 checkModule 测试 M() 模块的两种参数点 p1 和 p2 的情况
        self.checkModule(M(), (p1,))
        self.checkModule(M(), (p2,))

        # 对 M 类进行脚本化，获取其图形表示，并使用 FileCheck 检查其中的命名元组默认值
        m = torch.jit.script(M())

        FileCheck().check(r"NamedTuple(x : int?, y : int, z : int = 3))")   \
                   .run(m.graph)
    # 定义一个测试方法，测试具有命名字段的命名元组的默认值，容器类型版本
    def test_namedtuple_default_values_container_type(self):

        # 定义一个名为 Point 的命名元组类，包含三个字段
        class Point(NamedTuple):
            x: Optional[List[int]] = None  # 可选的整数列表，默认为 None
            y: List[int] = [1, 2, 3]  # 整数列表，默认为 [1, 2, 3]
            z: Optional[Dict[str, int]] = {"a": 1}  # 可选的字符串到整数字典，默认为 {"a": 1}

        # 将 Point 类注册为全局类型
        make_global(Point)

        # 定义一个继承自 torch.nn.Module 的模块类 M
        class M(torch.nn.Module):
            # 定义模块的前向传播方法，接受一个 Point 类型的参数 point
            def forward(self, point: Point):
                return point

        # 创建一个 Point 实例 p，指定其字段 x、y 和 z 的值
        p = Point(x=[4, 5, 6], y=[3, 2, 1], z={"b": 2})

        # 使用 checkModule 方法检查模块 M 的运行结果是否符合预期
        self.checkModule(M(), (p,))

        # 创建一个默认值的 Point 实例，并使用 checkModule 方法检查其运行结果
        self.checkModule(M(), (Point(),))

        # 将模块 M 转换为 TorchScript，得到转换后的模块 m
        m = torch.jit.script(M())

        # 定义字符串 first_line 作为预期的命名元组类型的字符串表示
        first_line = r"NamedTuple(x : int[]? = None, y : int[] = "    \
                     r"[1, 2, 3], z : Dict(str, int)? = {a: 1}))"

        # 使用 FileCheck 检查 TorchScript 图形中的 first_line 字符串
        FileCheck().check(first_line)   \
                   .run(m.graph)

    # 定义一个测试方法，测试具有命名字段的命名元组的默认值，使用 Tensor 类型版本
    def test_namedtuple_default_values_Tensor_type(self):

        # 定义一个名为 Point 的命名元组类，包含一个字段 x，其默认值为一个指定大小的随机 Tensor
        class Point(NamedTuple):
            x: torch.Tensor = torch.rand(2, 3)

        # 将 Point 类注册为全局类型
        make_global(Point)

        # 定义一个继承自 torch.nn.Module 的模块类 M
        class M(torch.nn.Module):
            # 定义模块的前向传播方法，接受一个 Point 类型的参数 point
            def forward(self, point: Point):
                return point

        # 创建一个指定 x 字段值的 Point 实例 p
        p = Point(x=torch.rand(2, 3))

        # 使用 self.assertRaisesRegex 检查尝试将模块 M 转换为 TorchScript 时的运行时错误
        with self.assertRaisesRegex(RuntimeError, "Tensors are not "
                                    "supported as default NamedTuple "
                                    "fields"):
            m = torch.jit.script(M())
            m(p)

    # 定义一个测试方法，测试使用工厂构造函数创建具有默认值的命名元组
    def test_namedtuple_default_values_using_factory_constructor(self):
        
        # 使用 namedtuple 工厂构造函数创建一个 Pair 命名元组类，包含两个字段 x 和 y，并指定它们的默认值
        Pair = namedtuple("Pair", ["x", "y"], defaults=(1, 2))

        # 将 Pair 类注册为全局类型
        make_global(Pair)

        # 定义一个 TorchScript 函数 fn，接受一个 Pair 类型的参数 x，返回值也是 Pair 类型
        @torch.jit.script
        def fn(x: Pair) -> Pair:
            return x

        # TODO: 由于 TorchScript 不支持使用命名元组工厂构造函数，所以无法使用 checkScript 方法进行检查
        # 使用 FileCheck 检查 TorchScript 图形中的命名元组类型的字符串表示
        FileCheck().check(r"NamedTuple(x : Tensor = 1, y : Tensor = 2))")   \
                   .check_next(r"return (%x.1)")    \
                   .run(fn.graph)

    # 定义一个测试方法，测试动态类型检查 isinstance 在 TorchScript 中的使用
    def test_isinstance_dynamic(self):
        # 定义一个 TorchScript 函数 foo，接受一个参数 a，可以是 Optional[List[int]] 类型，返回一个整数
        @torch.jit.script
        def foo(a):
            # type: (Optional[List[int]]) -> int
            b = 0
            # 如果 a 是 int、float、list 或 str 类型之一，则 b 加 1
            if isinstance(a, (int, (float,), list, str)):
                b += 1
            # 如果 a 是 int 或 str 类型之一，则 b 再加 1
            if isinstance(a, (int, str)):
                b += 1
            # 如果 a 是 List[int] 类型，则 b 再加 1
            if isinstance(a, List[int]):
                b += 1
            return b

        # 使用 self.assertEqual 检查 foo 函数对输入参数 [3, 4] 的返回值是否为 2
        self.assertEqual(foo([3, 4]), 2)
        # 使用 self.assertEqual 检查 foo 函数对输入参数 None 的返回值是否为 0
        self.assertEqual(foo(None), 0)
    # 测试函数重载误用情况，使用断言检查是否引发了预期的 RuntimeError 异常，并验证错误消息内容
    def test_function_overload_misuse(self):
        # 使用 @torch.jit._overload 装饰器声明一个错误的函数体
        with self.assertRaisesRegex(RuntimeError, "Only `pass` statement or `...` can be the body"):
            @torch.jit._overload
            def wrong_decl_body(x: str) -> str:
                return x + "0"

        # 使用 @torch.jit._overload_method 装饰器声明一个类方法的错误实现
        with self.assertRaisesRegex(RuntimeError, "Only `pass` statement or `...` can be the body"):
            class MyClass:
                @torch.jit._overload_method
                def method(self):
                    return 0

        # 声明一个空的函数体重载，以省略号 `...` 结尾，用于测试
        @torch.jit._overload
        def null_overload(x: int) -> int: ...
        
        # 声明一个字符串参数的函数体重载，只包含 `pass` 语句，用于测试
        @torch.jit._overload  # noqa: F811
        def null_overload(x: str) -> str:  # noqa: F811
            pass

        # 定义一个调用空函数体重载的驱动函数
        def null_overload_driver():
            return null_overload(0)

        # 使用 torch.jit.script 尝试编译驱动函数，期望引发函数实现缺失的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'Implementation for the function ".+" is missing.'):
            torch.jit.script(null_overload_driver)

        # 定义一个继承自 torch.nn.Module 的类，用于测试方法重载误用情况
        class OverloadMisuse(torch.nn.Module):
            # 使用 @torch.jit._overload_method 装饰器声明一个整数参数的方法重载，只包含 `pass` 语句
            @torch.jit._overload_method
            def forward(self, x: int):
                pass

            # 使用 @torch.jit._overload_method 装饰器声明一个 Tensor 参数的方法重载，只包含 `pass` 语句
            @torch.jit._overload_method  # noqa: F811
            def forward(self, x: Tensor):  # noqa: F811
                pass

        # 使用 torch.jit.script 尝试编译 OverloadMisuse 类，期望引发方法实现缺失的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'Implementation for the method ".+" is missing.'):
            m = torch.jit.script(OverloadMisuse())


    # 测试使用 torch.jit.script 方法对自定义模块进行脚本化，并验证脚本化模块的行为
    def test_script_method_torch_function_overload(self):
        # 定义一个继承自 torch.Tensor 的自定义张量类
        class MyCustomTensor(torch.Tensor):
            pass

        # 定义一个继承自 torch.nn.Module 的自定义模块类，包含一个简单的 forward 方法
        class MyCustomModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        # 使用 torch.jit.script 方法将 MyCustomModule 类实例化并脚本化
        scripted_mod = torch.jit.script(MyCustomModule())
        t = torch.tensor([3.0])
        # 测试脚本化模块在普通张量输入下的输出
        ref_out = scripted_mod(t)

        t_custom = MyCustomTensor([3.0])
        # 测试脚本化模块在自定义张量输入下的输出
        out1 = scripted_mod(t_custom)
        self.assertEqual(out1, ref_out)

        # 测试通过显式调用 forward 方法获得的输出与直接调用脚本化模块的输出是否一致
        out2 = scripted_mod.forward(t_custom)
        self.assertEqual(out2, ref_out)


    # 测试函数重载的 isinstance 使用情况
    def test_function_overloading_isinstance(self):
        # 声明一个接受 float 和 str 参数的函数体重载，只包含注释
        @torch.jit._overload  # noqa: F811
        def my_conv(x, y):  # noqa: F811
            # type: (float, str) -> (float)
            pass

        # 声明一个接受两个 float 参数的函数体重载，只包含注释
        @torch.jit._overload  # noqa: F811
        def my_conv(x, y):  # noqa: F811
            # type: (float, float) -> (float)
            pass

        # 定义一个真正的函数 my_conv，根据 y 的类型返回不同的结果
        def my_conv(x, y=2.0):  # noqa: F811
            if isinstance(y, str):
                if y == "hi":
                    return 4.0 - x
                else:
                    return 5.0 - x
            else:
                return 2.0 + x

        # 定义一个测试函数，调用 my_conv 函数的不同形式进行测试
        def test_uses():
            return my_conv(1.5), my_conv(1.5, "hi"), my_conv(1.5, 5.0)

        # 使用 self.checkScript 方法验证 test_uses 函数是否能被成功脚本化
        self.checkScript(test_uses, ())


    # 测试函数 foo 中的 narrow_copy 方法调用
    def test_narrow_copy(self):
        # 定义一个函数 foo，接受一个参数 a，并返回其调用 narrow_copy 方法后的结果
        def foo(a):
            return a.narrow_copy(0, 0, 5)

        # 使用 self.checkScript 方法验证 foo 函数是否能被成功脚本化，并传入一个 tensor 参数
        self.checkScript(foo, [torch.rand(10)])
    def test_select_after_chunk(self):
        # 定义内部函数 foo，接受参数 x
        def foo(x):
            # 使用 torch.chunk 将 x 分成 1 份
            chunked = torch.chunk(x, 1)
            # 取出 chunked 中的第一个元素，并命名为 foo
            foo = chunked[0]
            # 在 foo 上执行原地加法操作，增加 5
            foo.add_(5)
            # 返回原始参数 x，未被修改
            return x

        # 调用自定义函数 checkScript，传入 foo 函数和参数列表 [torch.rand(2, 3)]
        self.checkScript(foo, [torch.rand(2, 3)])

    def test_nn_LSTM_with_layers(self):
        # 定义继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化一个具有 2 个输入通道，3 个输出通道，2 层的 LSTM 模型，无 dropout
                self.rnn = nn.LSTM(2, 3, 2, dropout=0)

            @torch.jit.script_method
            # 定义 forward 方法，接受参数 x, lengths, h0, c0
            def forward(self, x, lengths, h0, c0):
                # 调用 LSTM 模型 self.rnn，返回输出序列中的第一个元素
                return self.rnn(x, (h0, c0))[0]

        # 定义继承自 torch.nn.Module 的类 Eager
        class Eager(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个具有 2 个输入通道，3 个输出通道，2 层的 LSTM 模型，无 dropout
                self.rnn = nn.LSTM(2, 3, 2, dropout=0)

            def forward(self, x, lengths, h0, c0):
                # 调用 LSTM 模型 self.rnn，返回输出序列中的第一个元素
                return self.rnn(x, (h0, c0))[0]

        # 定义输入参数元组 inputs
        inputs = (torch.randn(1, 1, 2), torch.LongTensor([7]), torch.randn(2, 1, 3), torch.randn(2, 1, 3))
        # 创建 Eager 类的实例并调用其 forward 方法，保存输出结果的第一个元素到 eager_out
        eager_out = self.runAndSaveRNG(lambda: Eager()(*inputs), ())[0]
        # 创建 M 类的实例并调用其 forward 方法，保存输出结果的第一个元素到 script_out
        script_out = self.runAndSaveRNG(lambda: M()(*inputs), ())[0]

        # 断言 eager_out 等于 script_out
        self.assertEqual(eager_out, script_out)

    def test_nn_LSTM(self):
        # 创建输入数据，使用 torch.nn.utils.rnn.pack_sequence 将随机张量列表打包成 PackedSequence 对象
        input = torch.nn.utils.rnn.pack_sequence([torch.randn(5, 5)])

        # 定义继承自 torch.jit.ScriptModule 的类 S
        class S(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化一个具有 5 个输入通道，5 个输出通道的 LSTM 模型
                self.x = torch.nn.LSTM(5, 5)

            @torch.jit.script_method
            # 定义 forward 方法，接受 PackedSequence 类型的 input，返回元组类型的结果
            def forward(self, input: PackedSequence) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                # 调用 LSTM 模型 self.x，返回输出结果
                return self.x(input)

        # 使用 Eager 模式计算 LSTM 输出结果，保存到 eager_out
        eager_out = self.runAndSaveRNG(lambda x: torch.nn.LSTM(5, 5)(x), (input,))[0]
        # 使用 Script 模式计算 LSTM 输出结果，保存到 script_out
        script_out = self.runAndSaveRNG(lambda x: S()(x), (input,))[0]

        # 断言 eager_out 等于 script_out
        self.assertEqual(eager_out, script_out)
    # 定义一个名为 test_nn_GRU 的测试方法
    def test_nn_GRU(self):
        # 创建一个序列输入，并对其进行打包
        seq_input = torch.nn.utils.rnn.pack_sequence([torch.randn(5, 5)])
        # 创建一个普通的张量输入
        tensor_input = torch.randn(5, 5, 5)

        # 定义一个名为 SeqLengthGRU 的 Torch 脚本模块
        class SeqLengthGRU(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.GRU(5, 5)

            # Torch 脚本方法，定义了模块的前向传播逻辑
            @torch.jit.script_method
            def forward(self, input: PackedSequence) -> Tuple[PackedSequence, torch.Tensor]:
                return self.x(input)

        # 定义一个名为 TensorGRU 的 Torch 脚本模块
        class TensorGRU(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.GRU(5, 5)

            # Torch 脚本方法，定义了模块的前向传播逻辑
            @torch.jit.script_method
            def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.x(input)

        # 在非 Torch 脚本环境下执行 GRU 模型的前向传播，并获取结果
        seq_eager_out = self.runAndSaveRNG(lambda x: torch.nn.GRU(5, 5)(x), (seq_input,))[0]
        # 在 Torch 脚本模块 SeqLengthGRU 下执行 GRU 模型的前向传播，并获取结果
        seq_script_out = self.runAndSaveRNG(lambda x: SeqLengthGRU()(x), (seq_input,))[0]
        # 在非 Torch 脚本环境下执行 GRU 模型的前向传播，并获取结果
        tensor_eager_out = self.runAndSaveRNG(lambda x: torch.nn.GRU(5, 5)(x), (tensor_input,))[0]
        # 在 Torch 脚本模块 TensorGRU 下执行 GRU 模型的前向传播，并获取结果
        tensor_script_out = self.runAndSaveRNG(lambda x: TensorGRU()(x), (tensor_input,))[0]

        # 断言两种执行方式下的输出结果应当相等
        self.assertEqual(seq_eager_out, seq_script_out)
        self.assertEqual(tensor_eager_out, tensor_script_out)

    # 定义一个名为 test_torchscript_memoryformat 的测试方法
    def test_torchscript_memoryformat(self):
        # 定义一个 Torch 脚本函数，用于确保张量 x 是连续的，并指定了内存格式为 channels_last
        @torch.jit.script
        def fn(x):
            return x.contiguous(memory_format=torch.channels_last)
        # 创建一个随机张量 x
        x = torch.randn(4, 3, 6, 6)
        # 调用 Torch 脚本函数 fn 对 x 进行处理，得到结果 y
        y = fn(x)
        # 断言结果 y 在 channels_last 内存格式下是连续的
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))

    # 定义一个名为 test_torchscript_multi_head_attn_fast_path 的测试方法
    def test_torchscript_multi_head_attn_fast_path(self):
        # 定义一些变量，设置注意力机制的参数
        src_l = 3
        bsz = 5
        embed_size = 8
        nhead = 2
        # 创建一个具有多头注意力机制的 nn.MultiheadAttention 实例，并将其设置为评估模式
        multi_head_attn = torch.nn.MultiheadAttention(embed_size, nhead, batch_first=True)
        multi_head_attn = multi_head_attn.eval()

        # 创建查询（query）、键（key）和值（value）张量，形状为 (bsz, src_l, embed_size)
        query = key = value = torch.rand((bsz, src_l, embed_size))

        # 在无梯度环境下，使用 nn.MultiheadAttention 处理查询、键和值，得到 py_out 结果
        with torch.no_grad():
            py_out = multi_head_attn(query, key, value)
            # 使用 Torch 脚本对 multi_head_attn 进行脚本化处理，得到 mha 对象
            mha = torch.jit.script(multi_head_attn)
            # 使用 Torch 脚本处理后的多头注意力机制对象 mha 对查询、键和值进行处理，得到 jit_out 结果
            jit_out = mha(query, key, value)
        # 断言 Torch 脚本处理后的输出 jit_out 与原始处理 py_out 结果相等
        torch.testing.assert_close(jit_out, py_out)
    def test_scriptmodule_multi_head_attn_cuda(self):
        # 定义一个测试函数，用于测试 CUDA 下的多头注意力模块

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                # 生成一个形状为 (3, 2, embed_dim) 的随机张量作为样本查询和键值对
                sample_q = torch.randn(3, 2, embed_dim)
                sample_kv = torch.randn(3, 2, embed_dim)
                # 创建一个多头注意力层，设为评估模式
                attention = nn.MultiheadAttention(embed_dim, num_heads)
                attention.eval()

                # 使用 torch.jit.trace 将注意力层跟踪为 TorchScript 模块
                self.mod = torch.jit.trace(attention,
                                           (sample_q, sample_kv, sample_kv))

            @torch.jit.script_method
            def forward(self, q, k, v):
                # 前向传播函数，调用跟踪的 TorchScript 模块进行计算
                return self.mod(q, k, v)

        embed_dim = 8
        num_heads = 2
        sl = 3  # 序列长度为 3
        bs = 2  # 批大小为 2

        # 创建 MyModule 实例并移动到 CUDA 设备
        model = MyModule(embed_dim, num_heads).cuda()
        
        # 生成形状为 (sl, bs, embed_dim) 的随机查询和键值对张量，并移动到 CUDA 设备
        q = torch.randn(sl, bs, embed_dim, device="cuda")
        kv = torch.randn(sl, bs, embed_dim, device="cuda")

        # 使用 JIT 模块进行前向传播计算
        jit_out = model(q, kv, kv)[0]
        
        # 使用 PyTorch 函数进行多头注意力的前向传播计算
        py_out = torch.nn.functional.multi_head_attention_forward(q, kv, kv,
                                                                  embed_dim, num_heads,
                                                                  model.mod.in_proj_weight,
                                                                  model.mod.in_proj_bias,
                                                                  None, None, None, 0.0,
                                                                  model.mod.out_proj.weight,
                                                                  model.mod.out_proj.bias)[0]
        
        # 断言 JIT 模块的输出与 PyTorch 函数的输出在一定误差范围内相等
        self.assertEqual(jit_out, py_out, atol=5e-4, rtol=1e-4)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_transformer_cuda(self):
        # 定义一个测试函数，用于测试带有 CUDA 支持的 ScriptModule 的行为
        class MyModule(torch.jit.ScriptModule):
            def __init__(self, transformer, sample_q, sample_kv):
                super().__init__()
                transformer.eval()  # 将传入的 transformer 模型设为评估模式

                # 使用 torch.jit.trace 将 transformer 模型转换为 TorchScript 模块
                self.mod = torch.jit.trace(transformer, (sample_q, sample_kv))

            @torch.jit.script_method
            def forward(self, q, k):
                return self.mod(q, k)  # 调用 TorchScript 模块的前向传播方法

        d_model = 8
        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 16
        bsz = 2
        seq_length = 5
        tgt_length = 3

        with torch.no_grad():
            # 生成随机的输入数据 src 和 tgt
            src = torch.randn(seq_length, bsz, d_model)
            tgt = torch.randn(tgt_length, bsz, d_model)

            # 创建 nn.Transformer 模型实例 transformer
            transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                         num_decoder_layers, dim_feedforward, dropout=0.0)

            # 使用 MyModule 类创建 TorchScript 模块 model
            model = MyModule(transformer, tgt, src)

            # 再次生成随机的输入数据 src 和 tgt
            src = torch.randn(seq_length, bsz, d_model)
            tgt = torch.randn(tgt_length, bsz, d_model)

            # 使用 TorchScript 模块进行前向传播得到 jit_out
            jit_out = model(tgt, src)

            # 直接使用 nn.Transformer 模型进行前向传播得到 py_out
            py_out = transformer(tgt, src)

            # 断言 TorchScript 模块输出与 nn.Transformer 模型输出一致
        self.assertEqual(jit_out, py_out, atol=5e-4, rtol=1e-4)

    def test_list_python_op(self):
        # 定义一个测试函数，用于测试处理 Python 列表操作的函数 fn
        def python_list_op(lst):
            # type: (List[Tensor]) -> Tensor
            return lst[0]  # 返回列表 lst 的第一个元素

        def fn(lst):
            # type: (List[Tensor]) -> Tensor
            return python_list_op(lst)  # 调用 python_list_op 函数

        self.checkScript(fn, ([torch.ones(2) + 2, torch.ones(2)],))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_weak_cuda(self):
        # 定义一个测试函数，用于测试带有 CUDA 支持的 TorchScript 模块 M 的行为
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(5, 5)
                self.lstm.cuda()  # 将 LSTM 模型移动到 CUDA 设备上

            @torch.jit.script_method
            def forward(self, x):
                return self.lstm(x)  # 调用 LSTM 模型的前向传播方法

        m = M()
        m.cuda()  # 将整个 TorchScript 模块 M 移动到 CUDA 设备上
        out = m(torch.ones(5, 5, 5).cuda())  # 使用 CUDA 输入进行模型推断
        self.assertTrue(out[0].is_cuda)  # 断言输出张量位于 CUDA 设备上
    def test_ignore_decorator(self):
        # 使用 warnings 模块捕获所有警告
        with warnings.catch_warnings(record=True) as warns:
            # 定义一个继承自 torch.jit.ScriptModule 的模块类 M
            class M(torch.jit.ScriptModule):
                def __init__(self):
                    super().__init__()
                    # 创建一个不需要梯度的全零张量，并注册为模块的缓冲区 'some_state'
                    tensor = torch.zeros(1, requires_grad=False)
                    self.register_buffer('some_state', torch.nn.Parameter(tensor))

                @torch.jit.script_method
                def forward(self, x):
                    # 在 forward 方法中调用被忽略的方法 ignored_code
                    self.ignored_code(x)
                    return x

                @torch.jit.ignore(drop_on_export=True)
                def ignored_code(self, x):
                    # 将 self.some_state 设置为 (100,) 的张量
                    self.some_state = torch.tensor((100,))

        # 使用 FileCheck 检查警告消息，确保 TorchScript 将会丢弃该函数
        FileCheck().check("TorchScript will now drop the function").run(str(warns[0]))

        # 断言被忽略的代码确实被执行
        m = M()

        # 获取模块 m 的导出/导入副本 m2
        m2 = self.getExportImportCopy(m)
        pp = str(m2.forward.code)
        # 断言在导出的 forward 方法代码中不包含 'ignored_code'
        self.assertNotIn('ignored_code', pp)

        # 使用断言检查在运行被忽略方法时是否抛出 TorchScript 错误
        with self.assertRaisesRegex(torch.jit.Error, "annotated to be ignored and cannot be run"):
            m2.forward(torch.ones(1))

    def test_ignored_as_value(self):
        # 定义一个继承自 nn.Module 的模块类 Model
        class Model(nn.Module):
            @torch.jit.unused
            def tuple_ignored(self, x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                # 忽略方法，返回输入张量 x 的元组形式
                return x, x

            @torch.jit.unused
            def single_val_ignored(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                # 忽略方法，返回输入张量 x
                return x

            def forward(self, x, use_ignore_path):
                # type: (Tensor, bool) -> Tuple[Tensor, Tensor]
                # 如果条件为假，返回 tuple_ignored 方法的结果
                if 1 == 2:
                    return self.tuple_ignored(x)
                # 如果 use_ignore_path 为真，返回 single_val_ignored 方法的结果的元组形式
                if use_ignore_path:
                    return self.single_val_ignored(x, x), self.single_val_ignored(x, x)
                # 否则返回输入张量 x 的元组形式
                return x, x

        # 创建一个原始的 Model 实例
        original = Model()
        # 对 Model 进行 TorchScript 脚本化
        scripted = torch.jit.script(original)
        # 使用断言验证脚本化后的模块对输入张量 0.5 的处理结果是否符合预期
        self.assertEqual(scripted(torch.tensor(.5), False), (torch.tensor(.5), torch.tensor(.5)))

        # 创建一个字节流缓冲区，并保存脚本化后的模块
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        # 加载保存的模块
        loaded = torch.jit.load(buffer)

        # 使用断言检查在运行被忽略方法时是否抛出 TorchScript 错误
        with self.assertRaisesRegex(torch.jit.Error, "annotated to be ignored and cannot be run"):
            loaded(torch.tensor(.5), True)

    def test_module_error(self):
        # 定义一个继承自 nn.Module 的模块类 MyModule
        class MyModule(torch.nn.Module):
            def forward(self, foo):
                # 返回输入 foo
                return foo

        # 使用断言检查是否抛出 RuntimeError，因为 MyModule 继承自 nn.Module，不能被编译
        with self.assertRaisesRegex(RuntimeError, "cannot be compiled since it inherits from nn.Module"):
            torch.jit.script(MyModule)

    def test_view_write(self):
        # 定义一个函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 创建一个空列表 l，并将 x 添加到列表中
            l = []
            l.append(x)
            # 从列表中取出第一个元素，赋值给 x_view
            x_view = l[0]
            # 计算 a = x + x
            a = x + x
            # 在 x_view 上执行原地加法 x_view.add_(y)
            x_view.add_(y)
            # 计算 b = x + x
            b = x + x
            # 返回 a 是否等于 b 的布尔值
            return a == b
        # 使用 self.checkScript 检查 fn 函数的 TorchScript 版本是否通过测试
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))
    def test_module_attrs(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            # 初始化方法，接受一个 table 参数
            def __init__(self, table):
                super().__init__()
                # 设置一个名为 table 的属性，类型为 Dict[str, torch.Tensor]
                self.table = torch.jit.Attribute(table, Dict[str, torch.Tensor])
                # 设置一个名为 x 的参数，初始值为 torch.tensor([100.0])
                self.x = torch.nn.Parameter(torch.tensor([100.0]))

            # 前向传播方法，接受一个 key 参数，返回一个 Tensor 类型的数据
            @torch.jit.script_method
            def forward(self, key):
                # type: (str) -> Tensor
                # 返回 self.table 中 key 对应的值加上 self.x
                return self.table[key] + self.x

        # 禁用 emit 钩子以防止 Python 打印属性时出现错误
        with torch._jit_internal._disable_emit_hooks():
            # 创建 M 类的实例 m，使用一个字典作为 table 参数
            m = M({char : torch.ones(1) + ord(char) - ord("a") for char in "abcdefg"})
            # 断言 m("c") 的返回值等于 torch.tensor([103.])
            self.assertEqual(m("c"), torch.tensor([103.]))

    def test_module_none_attrs(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 MyMod
        class MyMod(torch.jit.ScriptModule):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 设置一个可选值的属性 optional_value，初始值为 None
                self.optional_value = None

            # 前向传播方法，返回 self.optional_value
            @torch.jit.script_method
            def forward(self):
                return self.optional_value

        # 获取 MyMod 实例的前向传播图
        graph = MyMod().forward.graph
        # 运行 FileCheck 检查是否有 prim::GetAttr 操作
        FileCheck().check("prim::GetAttr").run(graph)
        # 运行 peephole 优化 pass
        self.run_pass('peephole', graph)
        # 再次运行 FileCheck 检查是否没有 prim::GetAttr 操作
        FileCheck().check_not("prim::GetAttr").run(graph)

    def test_tensor_import_export(self):
        # 定义一个装饰为 torch.jit.script 的函数 foo
        @torch.jit.script
        def foo(x):
            # 创建一个值为 1 的 Tensor a
            a = torch.tensor(1)
            # 创建一个包含 [1, 2] 的 Tensor b
            b = torch.tensor([1, 2])
            # 创建一个包含 a 和 b 的列表 c
            c = [a, b]
            return c

        # 运行常量传播 pass
        self.run_pass('constant_propagation', foo.graph)
        # 从图形创建函数 m
        m = self.createFunctionFromGraph(foo.graph)
        # 获取 m 的导出/导入副本
        self.getExportImportCopy(m)

    def get_pickle_values(self):
        # 返回多个 pickle 值的元组
        return (('dict', {"I": "am", "a test": "test"}, Dict[str, str]),
                ('float', 2.3, float),
                ('int', 99, int),
                ('bool', False, bool),
                ('tuple', (1, 2, 3, 4), Tuple[int, int, int, int]),
                ('list', [(1, 2), (3, 4)], List[Tuple[int, int]]),
                ('tensor', torch.randn(2, 2), torch.Tensor),
                ('int_list', [1, 2, 3, 4], List[int]),
                ('tensor_list', [torch.ones(2, 2) + i for i in range(4)], List[torch.Tensor]),
                ('bool_list', [True, True, False, True], List[bool]),
                ('float_list', [1., 2., 3., 4.], List[float]),
                ('str_list', ['hello', 'bye'], List[str]),
                ('none', None, Optional[int]),
                ('a_device', torch.device('cpu'), torch.device),
                ('another_device', torch.device('cuda:1'), torch.device))
    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: TemporaryFileName support for Windows or Sandcastle")
    def test_attribute_unpickling(self):
        tensor = torch.randn(2, 2)
        tester = self

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 遍历 tester 对象的 pickle 值元组，并为每个属性设置 torch.jit.Attribute 属性
                for name, value, the_type in tester.get_pickle_values():
                    setattr(self, "_" + name, torch.jit.Attribute(value, the_type))

            @torch.jit.script_method
            def forward(self):
                # 返回脚本化方法的元组，包含所有带下划线前缀的属性
                return (self._dict, self._float, self._int, self._bool, self._tuple,
                        self._list, self._int_list, self._tensor_list, self._bool_list,
                        self._float_list, self._str_list, self._none)

        # 使用临时文件名创建 M 类的实例，并保存到临时文件中
        with TemporaryFileName() as fname:
            M().save(fname)
            # 加载保存的模型
            loaded = torch.jit.load(fname)

            # 检查值是否为张量，如果是，则跳过检查
            def is_tensor_value(item):
                if isinstance(item, torch.Tensor):
                    return True
                if isinstance(item, list):
                    return is_tensor_value(item[0])
                return False
            
            # 遍历 tester 对象的 pickle 值元组，并逐一检查加载后的属性值是否相等
            for name, value, the_type in self.get_pickle_values():
                if is_tensor_value(value):
                    continue
                self.assertEqual(value, getattr(loaded, "_" + name))
    def test_submodule_attribute_serialization(self):
        class S(torch.jit.ScriptModule):
            def __init__(self, list_data):
                super().__init__()
                # 定义名为 table 的属性，类型为 Dict[str, str]，包含键值对
                self.table = torch.jit.Attribute({"I": "am", "a test": "test"}, Dict[str, str])
                # 定义名为 list 的属性，类型为 List[Tuple[int, int]]，初始化为 list_data
                self.list = torch.jit.Attribute(list_data, List[Tuple[int, int]])

            @torch.jit.script_method
            def forward(self):
                # 返回当前对象的 table 和 list 属性
                return (self.table, self.list)

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义名为 table 的属性，类型为 Dict[str, str]，包含不同的键值对
                self.table = torch.jit.Attribute({"this": "is", "a different": "dict"}, Dict[str, str])
                # 定义名为 tensor 的属性，类型为 torch.Tensor，初始化为随机生成的 2x2 张量
                self.tensor = torch.jit.Attribute(torch.randn(2, 2), torch.Tensor)
                # 定义名为 s1 和 s2 的属性，类型为 S 类的实例，分别初始化为包含元组的列表
                self.s1 = S([(1, 2)])
                self.s2 = S([(4, 5)])

            @torch.jit.script_method
            def forward(self):
                # 返回当前对象的 table, tensor, s1.table, s2.list 和 s1.list 属性
                return (self.table, self.tensor, self.s1.table, self.s2.list, self.s1.list)

        m = M()
        # 通过自定义的函数 getExportImportCopy 复制并导入模块 m
        imported_m = self.getExportImportCopy(m)
        # 断言调用 m 和 imported_m 的结果相等
        self.assertEqual(m(), imported_m())

    def test_serialization_big_ints(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义名为 int32_max 的属性，类型为 int，值为 2**31 - 1
                self.int32_max = torch.jit.Attribute(2**31 - 1, int)
                # 定义名为 int32_min 的属性，类型为 int，值为 -2**31
                self.int32_min = torch.jit.Attribute(-2**31, int)
                # 定义名为 uint32_max 的属性，类型为 int，值为 2**32
                self.uint32_max = torch.jit.Attribute(2**32, int)

                # 定义名为 int64_max 的属性，类型为 int，值为 2**63 - 1
                self.int64_max = torch.jit.Attribute(2**63 - 1, int)
                # 定义名为 int64_min 的属性，类型为 int，值为 -2**63
                self.int64_min = torch.jit.Attribute(-2**63, int)

                # 定义名为 tensor 的属性，类型为 torch.nn.Parameter，初始化为全为 1 的 2x2 张量
                self.tensor = torch.nn.Parameter(torch.ones(2, 2))

            @torch.jit.script_method
            def forward(self, x):
                # type: (int) -> (int)
                # 返回 x 加上 int32_max 和 int32_min 的和，再加上 int64_max 和 int64_min 的和
                return x + (self.int32_max + self.int32_min) + (self.int64_max + self.int64_min)

        m = M()
        # 复制并导入模块 m
        imported = self.getExportImportCopy(m)
        # 断言调用 m 和 imported 的结果相等，且各属性值也相等
        self.assertEqual(m(10), imported(10))
        self.assertEqual(m.int32_max, imported.int32_max)
        self.assertEqual(m.int32_min, imported.int32_min)
        self.assertEqual(m.uint32_max, imported.uint32_max)
        self.assertEqual(m.int64_max, imported.int64_max)
        self.assertEqual(m.int64_min, imported.int64_min)

    def test_script_scope(self):
        # 将 torch.nn.functional.triplet_margin_loss 函数转换为 TorchScript
        scripted = torch.jit.script(torch.nn.functional.triplet_margin_loss)

    @unittest.skipIf(IS_WINDOWS, "NYI: TemporaryFileName on Windows")
    # 定义一个测试方法，用于测试对象序列化和共享的功能
    def test_serialization_sharing(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 初始化一个属性 list，类型为 List[str]
                self.list = torch.jit.Attribute([], List[str])

            @torch.jit.script_method
            # 定义一个前向方法 forward，接受一个参数 key（字符串），返回一个字符串列表
            def forward(self, key):
                # type: (str) -> List[str]
                # 将 key 添加三次到 self.list 中
                self.list.append(key)
                self.list.append(key)
                self.list.append(key)
                return self.list

        # 创建 M 类的实例 m
        m = M()
        # 定义两个字符串变量 s1 和 s2
        s1 = "a long string"
        s2 = "a different, even longer string"
        # 断言调用 m(s1) 返回包含 s1 三次的列表
        self.assertEqual(m(s1), [s1] * 3)
        # 断言调用 m(s2) 返回包含 s1 三次和 s2 三次的列表
        self.assertEqual(m(s2), [s1] * 3 + [s2] * 3)
        
        # 使用临时文件名创建一个临时文件
        with TemporaryFileName() as fname:
            # 将模型 m 保存到临时文件中
            m.save(fname)
            # 获取保存后的文件名
            archive_name = os.path.basename(os.path.normpath(fname))
            # 打开保存的 ZIP 归档文件
            archive = zipfile.ZipFile(fname, 'r')
            # 从 ZIP 归档文件中读取名为 'data.pkl' 的数据
            pickled_data = archive.read(os.path.join(archive_name, 'data.pkl'))

            # 创建一个 StringIO 对象 out
            out = io.StringIO()
            # 使用 pickletools.dis 方法解析 pickled_data 的内容，输出到 out 中
            pickletools.dis(pickled_data, out=out)
            # 获取 disassembled 的字符串值
            disassembled = out.getvalue()

            # 使用 FileCheck 检查 disassembled 中 s1 出现的次数为一次
            FileCheck().check_count(s1, 1, exactly=True) \
                # 检查 "BINGET" 出现的次数为两次
                .check_count("BINGET", 2, exactly=True) \
                # 检查 disassembled 中 s2 出现的次数为一次
                .check_count(s2, 1, exactly=True) \
                # 再次检查 "BINGET" 出现的次数为两次
                .check_count("BINGET", 2, exactly=True).run(out.getvalue())

    # 定义一个测试方法，用于测试 sys.stdout 被重定向的情况
    def test_sys_stdout_override(self):
        # 定义一个装饰为 torch.jit.script 的函数 foo
        @torch.jit.script
        def foo():
            print('foo')

        # 定义一个重定向类 Redirect
        class Redirect:
            def __init__(self):
                self.s = ''

            # 重写 write 方法，将输入的字符串 s 追加到 self.s 中
            def write(self, s):
                self.s += s

        # 保存当前的标准输出对象到 old_stdout
        old_stdout = sys.stdout
        # 创建 Redirect 类的实例 redirect
        redirect = Redirect()
        try:
            # 将 sys.stdout 重定向到 redirect
            sys.stdout = redirect
            # 调用函数 foo，其输出将被重定向到 redirect.s 中
            foo()
        finally:
            # 恢复原来的 sys.stdout
            sys.stdout = old_stdout

        # 使用 FileCheck 检查 redirect.s 中是否包含字符串 'foo'
        FileCheck().check('foo').run(redirect.s)

    # 定义一个测试方法，用于测试 dtype 属性的脚本化
    def test_dtype_attr(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个属性 dtype，值为 torch.zeros([]) 的数据类型
                self.dtype = torch.zeros([]).dtype

            # 定义一个前向方法 forward，返回一个数据类型为 self.dtype 的全零张量
            def forward(self):
                return torch.zeros(3, 4, dtype=self.dtype)

        # 创建 Foo 类的实例 f
        f = Foo()
        # 对 f 进行脚本化
        torch.jit.script(f)
    def test_named_buffers_are_iterable(self):
        # 定义一个测试用例，验证模型中的命名缓冲是否可迭代
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块
                self.mod = (torch.nn.ReLU())
                self.mod2 = (torch.nn.ReLU())
                # 嵌套的 Sequential 模块
                self.mod3 = torch.nn.Sequential(torch.nn.Sequential(torch.nn.ReLU()))
                # 注册缓冲区 'x' 和 'y'，以及普通属性 'z'
                self.register_buffer('x', torch.zeros(3))
                self.register_buffer('y', torch.zeros(3))
                self.z = torch.zeros(3)

            def bleh(self):
                # 返回 self.z 加 4 的结果
                return self.z + 4

            @torch.jit.export
            def method(self):
                # 获取所有命名缓冲区的名称和值，对每个缓冲区的值加 2，并返回结果
                names = [""]
                vals = []
                for name, buffer in self.named_buffers():
                    names.append(name)
                    vals.append(buffer + 2)

                return names, vals

            def forward(self, x):
                # 前向传播函数，简单地返回输入 x
                return x

        model = MyMod()
        # 对模型进行脚本化
        x = torch.jit.script(model)
        # 获得模型的导出和导入拷贝
        z = self.getExportImportCopy(x)

        # 断言两个模型的 method 方法的输出相等
        self.assertEqual(z.method(), x.method())
        self.assertEqual(z.method(), model.method())
        self.assertEqual(x.method(), model.method())

        # 获取 x 模型的 method 方法的返回结果中的所有名称
        names = x.method()
        # 对每个名称进行断言，确保不等于字符串 'z'
        for name in names:
            self.assertNotEqual('z', name)
    def test_static_if_prop(self):
        # 定义一个测试方法，用于测试静态属性的情况

        class MaybeHasAttr(torch.nn.Module):
            def __init__(self, add_attr):
                super().__init__()
                # 如果 add_attr 为真，则添加一个 maybe_attr 属性
                if add_attr:
                    self.maybe_attr = 1

            def forward(self):
                # 如果 self 中有 maybe_attr 属性且条件为真，则返回该属性的值
                if hasattr(self, "maybe_attr") and True:
                    return self.maybe_attr
                else:
                    return 0

        class MaybeHasAttr2(torch.nn.Module):
            def __init__(self, add_attr):
                super().__init__()
                # 如果 add_attr 为真，则添加一个 maybe_attr 属性
                if add_attr:
                    self.maybe_attr = 1

            def forward(self):
                # 如果 self 中没有 maybe_attr 属性或条件为假，则返回 0
                if not hasattr(self, "maybe_attr") or False:
                    return 0
                else:
                    return self.maybe_attr

        # 使用 torch.jit.script 将类转换为 Torch 脚本以进行 JIT 编译
        torch.jit.script(MaybeHasAttr(True))
        torch.jit.script(MaybeHasAttr(False))
        torch.jit.script(MaybeHasAttr2(True))
        torch.jit.script(MaybeHasAttr2(False))

        class MyMod(torch.nn.Module):
            def forward(self):
                # 如果 self 中有 foo 属性，则返回 1，否则返回 0
                if hasattr(self, "foo"):
                    return 1
                else:
                    return 0

            @torch.jit.export
            def fee(self):
                return 1

        # 调用自定义的 checkModule 方法，验证 MyMod 类的行为
        self.checkModule(MyMod(), ())

        class HasAttrMod(torch.nn.Module):
            __constants__ = ["fee"]

            def __init__(self):
                super().__init__()
                self.fee = 3

            def forward(self):
                # 检查 self 中的各个属性是否存在，并返回它们的存在情况
                a = hasattr(self, "fee")
                b = hasattr(self, "foo")
                c = hasattr(self, "hi")
                d = hasattr(self, "nonexistant")
                return (a, b, c, d)

            def foo(self):
                return 1

            @torch.jit._overload_method
            def hi(self, x: Tensor): ...  # noqa: E704

            def hi(self, x):  # noqa: F811
                return 2

        # 调用自定义的 checkModule 方法，验证 HasAttrMod 类的行为
        self.checkModule(HasAttrMod(), ())

        @torch.jit.script
        class FooTest:
            def __init__(self):
                self.x = 1

            def foo(self, y):
                return self.x + y

        def foo():
            # 创建一个 FooTest 实例并检查其属性的存在情况
            a = FooTest()
            val1 = hasattr(a, "foo"), hasattr(a, "x"), hasattr(a, "bla")
            val2 = hasattr(FooTest, "foo"), hasattr(FooTest, "a")
            return val1, val2

        # 使用 assertEqual 方法验证 foo 函数的输出与 torch.jit.script(foo) 的输出是否一致
        self.assertEqual(foo(), torch.jit.script(foo)())
    # 测试使用 pickle 保存和加载模型的功能，将模型参数保存到临时文件中
    def _test_pickle_checkpoint(self, device):
        # 使用临时文件名上下文管理器
        with TemporaryFileName() as fname:
            # 定义一个继承自 torch.jit.ScriptModule 的模型类 M
            class M(torch.jit.ScriptModule):
                __constants__ = ['fname']

                # 模型初始化函数，接受一个 tensor 参数
                def __init__(self, tensor):
                    super().__init__()
                    # 设置模型的 fname 和 tensor 属性
                    self.fname = fname
                    self.tensor = torch.nn.Parameter(tensor)

                # 前向传播方法的 Torch 脚本版本
                @torch.jit.script_method
                def forward(self, x):
                    # 计算输出 y
                    y = self.tensor + x
                    # 将 y 保存到文件 fname 中
                    torch.save(y, self.fname)
                    return y

            # 创建一个 tensor 作为模型的参数
            param = torch.randn(2, 2).to(device)
            # 创建一个输入 tensor
            input = torch.randn(2, 2).to(device)
            # 实例化模型 M
            m = M(param)
            # 进行一次前向传播
            m(input)
            # 打开 fname 对应的文件，加载保存的 tensor
            with open(fname, "rb") as handle:
                # 加载并比较加载的 tensor 是否与预期的 input + param 相等
                loaded_tensor = torch.load(fname)
                self.assertEqual(loaded_tensor, input + param)

    # 测试使用 pickle 保存和加载模型中视图（view）的功能，将多个 tensor 保存到临时文件中
    def _test_pickle_checkpoint_views(self, device):
        # 使用临时文件名上下文管理器
        with TemporaryFileName() as fname:
            # 定义一个继承自 torch.jit.ScriptModule 的模型类 M
            class M(torch.jit.ScriptModule):
                __constants__ = ['fname']

                # 模型初始化函数，接受一个 tensor 参数
                def __init__(self, tensor):
                    super().__init__()
                    # 设置模型的 fname 和 tensor 属性
                    self.fname = fname
                    self.tensor = torch.nn.Parameter(tensor)

                # 前向传播方法的 Torch 脚本版本
                @torch.jit.script_method
                def forward(self, x):
                    # 计算输出 y
                    y = self.tensor + x
                    # 创建 y 的视图 y_view，并将 y、y_view 和 y 一起保存到文件 fname 中
                    y_view = y.view(4)
                    torch.save((y, y_view, y), self.fname)
                    return y

            # 创建一个 tensor 作为模型的参数
            param = torch.randn(2, 2).to(device)
            # 创建一个输入 tensor
            input = torch.randn(2, 2).to(device)
            # 实例化模型 M
            m = M(param)
            # 进行一次前向传播
            m(input)
            # 打开 fname 对应的文件，加载保存的 tensors
            with open(fname, "rb") as handle:
                # 加载并比较加载的 tensors 是否与预期的 input + param 相等
                loaded_y, loaded_y_view, loaded_y_2 = torch.load(fname)
                self.assertEqual(loaded_y, input + param)
                # 使用 torch.no_grad() 确保不进行梯度计算
                with torch.no_grad():
                    # 修改 loaded_y_view 中的一个元素
                    loaded_y_view[1] += 20
                    # 断言 loaded_y 的视图与 loaded_y_view 相等，以验证 loaded_y 是否也被修改
                    self.assertEqual(loaded_y.view(4), loaded_y_view)
                    self.assertEqual(loaded_y_2.view(4), loaded_y_view)

    # 在 CUDA 设备上测试使用 pickle 保存和加载模型的功能
    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_pickle_checkpoint_cuda(self):
        # 测试 _test_pickle_checkpoint 和 _test_pickle_checkpoint_views 方法在 CUDA 设备上的运行情况
        self._test_pickle_checkpoint('cuda')
        self._test_pickle_checkpoint_views('cuda')

    # 在 CPU 设备上测试使用 pickle 保存和加载模型的功能
    def test_pickle_checkpoint(self):
        # 测试 _test_pickle_checkpoint 和 _test_pickle_checkpoint_views 方法在 CPU 设备上的运行情况
        self._test_pickle_checkpoint('cpu')
        self._test_pickle_checkpoint_views('cpu')

    # 测试使用 pickle 保存和加载简单数据的功能
    def test_pickle_checkpoint_tup(self):
        # 定义一个 Torch 脚本函数 foo，接受一个字符串作为参数 fname，将 (3, 4) 保存到文件 fname 中
        @torch.jit.script
        def foo(fname):
            # type: (str) -> None
            torch.save((3, 4), fname)
        # 使用临时文件名上下文管理器
        with TemporaryFileName() as name:
            # 调用 foo 函数，将 (3, 4) 保存到临时文件 name 中
            foo(name)
            # 断言从 name 文件加载的数据与预期的 (3, 4) 相等
            self.assertEqual(torch.load(name), (3, 4))

    # 测试字符串转列表的功能
    def test_string_list(self):
        # 定义一个函数 fn，接受一个字符串参数 string，返回其字符组成的列表
        def fn(string):
            # type: (str) -> List[str]
            return list(string)

        # 调用 self.checkScript 方法验证 fn 函数在输入 "abcdefgh" 上的 Torch 脚本执行情况
        self.checkScript(fn, ("abcdefgh",))
    def test_unicode_comments(self):
        @torch.jit.script
        def test(self, a):
            # 'shrug' 注释：简单的注释，表示不确定或者无所谓的意思
            return torch.nn.functional.relu(a)

    def test_get_set_state_with_tensors(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tensor = torch.randn(2, 2)

            @torch.jit.export
            def __getstate__(self):
                # 返回对象状态 注释：定义获取对象状态的方法，返回包含张量和训练状态的元组
                return (self.tensor, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                # 设置对象状态 注释：定义设置对象状态的方法，根据传入的状态元组恢复张量和训练状态
                self.tensor = state[0]
                self.training = state[1]

            def forward(self, x):
                # 模型前向传播 注释：定义模型的前向传播过程
                return x + self.tensor

        with TemporaryFileName() as fname:
            m = torch.jit.script(M())
            m.save(fname)
            loaded = torch.jit.load(fname)
            self.assertEqual(loaded.tensor, m.tensor)

    def test_in_for_and_comp_expr(self):
        def fn(d):
            # type: (Dict[str, int]) -> List[int] 注释：函数声明及参数注释
            out = [1]
            for i in range(d["hi"] if "hi" in d else 6):
                out.append(i)  # noqa: PERF402 注释：在列表末尾添加当前循环变量的值

            return out

        self.checkScript(fn, ({'hi': 2, 'bye': 3},)) 注释：使用torch.jit.script检查函数fn的脚本化版本
        self.checkScript(fn, ({'bye': 3},)) 注释：使用torch.jit.script检查函数fn的脚本化版本

    def test_for_else(self):
        def fn():
            c = 0
            for i in range(4):
                c += 10
            else:
                print("In else block of for...else") 注释：在for循环完成后执行的else块，但在PyTorch脚本中不支持

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, "else branches of for loops aren't supported"):
            torch.jit.script(fn) 注释：使用torch.jit.script尝试脚本化函数fn，预期会抛出else分支不支持的异常

    def test_split(self):
        def split_two(tensor):
            a, b, c = torch.split(tensor, 2, dim=1)
            return a, b, c 注释：使用torch.split函数将输入张量在指定维度上分割成三部分并返回

        x = torch.randn(3, 6)
        y = torch.randn(3, 6)
        self.checkScript(split_two, [(x + y)]) 注释：使用torch.jit.script检查函数split_two的脚本化版本

    def test_conv_error(self):
        @torch.jit.script
        def fn(x, y):
            return F.conv2d(x, y)

        try:
            fn(torch.ones(2, 2), torch.ones(4, 4))
        except RuntimeError as e:
            self.assertFalse('frame' in str(e)) 注释：捕获运行时异常，并验证异常消息中不包含'frame'

    def test_python_op_name(self):
        import random

        with self.assertRaisesRegex(RuntimeError, "randint"): 注释：使用with语句验证运行时抛出的异常消息包含'randint'
            @torch.jit.script
            def fn():
                return random.randint()

    def test_dir(self):
        class M(torch.jit.ScriptModule):
            def forward(self, t):
                return t

        self.assertTrue('forward' in dir(M())) 注释：验证M类的实例在其dir中包含'forward'方法

    def test_kwarg_expansion_error(self):
        @torch.jit.ignore
        def something_else(h, i):
            pass

        def fn(x):
            something_else(**x) 注释：调用被@torch.jit.ignore装饰的函数something_else，并展开关键字参数x

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, "keyword-arg expansion is not supported"): 注释：使用with语句验证torch.jit.script尝试脚本化函数fn时抛出关键字参数展开不支持的异常
            torch.jit.script(fn)
    # 定义一个测试方法，测试关键字参数错误消息处理
    def test_kwargs_error_msg(self):
        # 定义一个内部函数 other，接收任意关键字参数并打印
        def other(**kwargs):
            print(kwargs)

        # 定义一个函数 fn，调用 other 函数但未传入任何参数
        def fn():
            return other()

        # 使用 assertRaisesRegex 断言捕获 torch.jit.frontend.NotSupportedError 异常，并验证错误消息包含 'variable number'
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'variable number'):
            torch.jit.script(fn)

        # 定义另一个内部函数 another_other，接收任意位置参数并打印
        def another_other(*args):
            print(args)

        # 定义函数 another_fn，调用 another_other 函数但未传入任何参数
        def another_fn():
            return another_other()

        # 使用 assertRaisesRegex 断言捕获 torch.jit.frontend.NotSupportedError 异常，并验证错误消息包含 'variable number'
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'variable number'):
            torch.jit.script(another_fn)

    # 定义一个测试方法，测试在类型推断失败时的错误消息输出
    def test_inferred_error_msg(self):
        """
        Test that when we get a type mismatch on a function where we inferred
        the type to be tensor, a good error message is given.
        """
        # 使用 torch.jit.script 装饰器定义一个函数 foo，返回输入参数 a
        @torch.jit.script
        def foo(a):
            return a

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误消息包含类型不匹配的信息
        with self.assertRaisesRegex(RuntimeError, (r"Expected a value of type \'Tensor \(inferred\)\'"
                                                   r"[\S\s]*Inferred \'a\' to be of type \'Tensor\'")):
            foo("1")

    # 定义一个测试方法，测试在函数体内的类型注释
    def test_type_comments_in_body(self):
        # 使用 torch.jit.script 装饰器定义一个函数 foo，接收两个参数 a 和 b，返回它们的和
        @torch.jit.script
        def foo(a,  # type: int
                b,  # type: int
                ):
            # type: (...) -> int
            # type: int
            return a + b

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            def __init__(self,
                         a,  # type: int
                         b   # type: int
                         ):
                # type: (...) -> None
                super().__init__()
                self.a = a  # type: int
                self.b = b  # type: int

        # 使用 torch.jit.script 将类 M 实例化，并传入参数 2 和 3 进行测试
        torch.jit.script(M(2, 3))

    # 定义一个测试方法，测试在模块方法中使用 input 关键字的情况
    def test_input_keyword_in_schema(self):
        # 定义一个函数 f，接收一个参数 x，并调用 torch.ceil 函数，使用 input=x 形式传入参数
        def f(x):
            return torch.ceil(input=x)

        # 创建一个输入张量 inp，其形状为 (10,)
        inp = torch.randn(10)
        # 调用 self.checkScript 方法检查函数 f 的脚本化版本，并传入输入张量 inp 进行测试
        self.checkScript(f, (inp, ))

    # 定义一个测试方法，测试模块方法重新赋值的情况
    def test_module_method_reassignment(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def _forward(self, x):
                return x

            # 将 _forward 方法重新赋值给 forward 方法
            forward = _forward

        # 使用 torch.jit.script 将类 Foo 实例化为 sm
        sm = torch.jit.script(Foo())
        # 创建一个形状为 (2, 2) 的输入张量 input
        input = torch.ones(2, 2)
        # 使用 self.assertEqual 验证输入张量 input 经过 sm 处理后与原始输入相等
        self.assertEqual(input, sm(input))

    # 测试在脚本模块中使用 torch.Tensor 的子类（如 Parameter）作为参数的情况
    # 输入张量 input 为形状为 (2, 2) 的全 1 张量
    def test_script_module_tensor_subclass_argument(self):
        # 使用 torch.jit.script 装饰器定义一个函数 parameter_script，接收一个 torch.nn.Parameter 类型的参数 x，并返回 x
        @torch.jit.script
        def parameter_script(x: torch.nn.Parameter):
            return x

        # 创建一个形状为 (2, 2) 的全 1 张量 input
        input = torch.ones(2, 2)
        # 使用 self.assertEqual 验证输入张量 input 经过 parameter_script 处理后与原始输入相等
        self.assertEqual(input, parameter_script(input))
    def test_save_load_attr_error(self):
        # 定义一个内部的 PyTorch 模块 `Inner`
        class Inner(nn.Module):
            def forward(self, x):
                return x

        # 定义一个包装器 `Wrapper`，接受一个内部模块作为参数
        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                # 尝试访问 `Inner` 类中不存在的属性 `b`
                return self.inner.b(x)

        # 通过 Torch 的 JIT 脚本化功能将 `Inner` 类实例化为一个脚本模块
        inner_module = torch.jit.script(Inner())
        # 使用辅助函数获取导出和导入的副本
        inner_module = self.getExportImportCopy(inner_module)
        # 使用 `Wrapper` 类封装脚本化的 `Inner` 模块
        wrapped = Wrapper(inner_module)
        # 这里应该正确地报告 `self.inner` 没有 `b` 属性的错误
        with self.assertRaisesRegex(RuntimeError, 'has no attribute'):
            torch.jit.script(wrapped)

    def test_rescripting_loaded_modules(self):
        # 定义一个内部子模块 `InnerSubmod`
        class InnerSubmod(nn.Module):
            __constants__ = ['my_constant']

            def __init__(self):
                super().__init__()
                # 注册缓冲区 `foo`
                self.register_buffer("foo", torch.ones(1))
                # 注册参数 `bar`
                self.register_parameter("bar", torch.nn.Parameter(torch.ones(1)))
                # 定义普通属性 `baz`
                self.baz = torch.ones(1)
                # 定义常量 `my_constant`
                self.my_constant = 1

            def forward(self, x):
                return x + x

        # 定义一个包含 `InnerSubmod` 的内部模块 `Inner`
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = InnerSubmod()

            def forward(self, x):
                return self.submod(x)

        # 定义一个包装器 `Wrapper`，接受一个内部模块作为参数
        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                # 访问内部模块中的元素
                ret = self.inner.submod(x) + self.inner.submod.foo + self.inner.submod.bar + self.inner.submod.baz
                ret = ret + self.inner.submod.my_constant
                return ret

        # 通过 Torch 的 JIT 脚本化功能将 `Inner` 类实例化为一个脚本模块
        inner_module = torch.jit.script(Inner())
        # 使用 `Wrapper` 类封装脚本化的 `Inner` 模块
        wrapped = Wrapper(inner_module)
        # 使用自定义函数检查模块
        self.checkModule(wrapped, torch.ones(1))

        # 获取重新导出和导入的 `Inner` 模块副本
        inner_module_loaded = self.getExportImportCopy(inner_module)
        # 使用重新导入的 `Inner` 模块创建包装器 `Wrapper` 的副本
        wrapped_loaded = Wrapper(inner_module_loaded)
        # 断言脚本化后的 `wrapped` 和重新导入后的 `wrapped_loaded` 表现相同
        self.assertEqual(wrapped(torch.ones(1)), wrapped_loaded(torch.ones(1)))

    def test_interpret_graph(self):
        # 定义一个简单的函数 `fn`，对输入的 Tensor 执行 unfold 操作
        def fn(x):
            return x.unfold(0, 1, 1)

        # 定义一个表示计算图的字符串
        graph_str = """
        graph(%a : Tensor, %b : Tensor):
          %c : Tensor = aten::mul(%a, %b)
          return (%c)
        """
        # 解析计算图字符串
        graph = parse_ir(graph_str)
        # 创建两个随机的 Tensor `a` 和 `b`
        a = torch.rand(10)
        b = torch.rand(10)
        # 使用 Torch 的 JIT 解释图函数 `fn`，传入 `a` 和 `b`
        test = torch._C._jit_interpret_graph(graph, (a, b))
        # 计算参考结果 `a * b`
        ref = a * b
        # 断言解释后的结果 `test` 等于参考结果 `ref`
        self.assertEqual(test, ref)

    def test_signed_float_zero(self):

        # 定义一个简单的 PyTorch 模块 `MyModule`
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 使用 Torch 的 `div` 函数，将输入 `x` 除以负零
                return torch.div(x, -0.)

        # 创建一个输入张量 `inp`，值为 1 的 Tensor
        inp = torch.ones(1)
        # 使用自定义函数检查模块 `MyModule` 的行为
        self.checkModule(MyModule(), inp)
    def test_index_with_tuple(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 重写 forward 方法，接受一个参数 x，并返回 x 中索引为 (1,) 的元素
            def forward(self, x):
                return x[(1,)]

        # 使用自定义的 MyModule 类作为参数调用 checkModule 方法，传入一个元组 (torch.ones(2, 3),)
        self.checkModule(MyModule(), (torch.ones(2, 3),))

    def test_context_manager(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 重写 forward 方法，接受两个参数 x 和 y，计算 p = x + y，q = p + 2.0，然后返回 q
            def forward(self, x, y):
                p = x + y
                q = p + 2.0
                return q

        # 创建两个形状为 (3, 2) 的随机张量 x 和 y，数据类型为 torch.float
        x = torch.randn(3, 2, dtype=torch.float)
        y = torch.randn(3, 2, dtype=torch.float)
        # 遍历 ['fuser0', 'fuser1', 'none']，分别使用不同的 fuser 名称进行上下文管理
        for fuser_name in ['fuser0', 'fuser1', 'none']:
            with torch.jit.fuser(fuser_name):
                # 使用自定义的 MyModule 类作为参数调用 checkModule 方法，传入 x 和 y 作为参数
                self.checkModule(MyModule(), (x, y))

    def test_zero_dimension_tensor_trace(self):
        # 定义一个函数 f，接受一个参数 x，返回 x 中大于 0 的元素
        def f(x):
            return x[x > 0]
        # 使用 torch.jit.trace 将函数 f 进行追踪，追踪的输入是一个包含单个元素 2. 的张量，设备为 "cpu"
        jf = torch.jit.trace(f, torch.tensor(2., device="cpu"))
# known to be failing in tracer
EXCLUDE_TRACED = {
    # The following fail due to #12024.
    # A prim::ListConstruct is involved and the indices get traced as TensorType,
    # which always require_grad. This causes a crash in autodiff.
    'test___getitem___adv_index',
    'test___getitem___adv_index_beg',
    'test___getitem___adv_index_comb',
    'test___getitem___adv_index_dup',
    'test___getitem___adv_index_sub',
    'test___getitem___adv_index_sub_2',
    'test___getitem___adv_index_sub_3',
    'test___getitem___adv_index_var',

    # jit doesn't support sparse tensors.
    'test_to_sparse',
    'test_to_sparse_dim',
}

EXCLUDE_TYPE_CHECK = {
    # slogdet tests use itemgetter to select its only differentiable output,
    # but this happens outside of the graph we handle, so there are fewer
    # reference outputs than graph outputs.
    'test_slogdet_1x1_neg_det',
    'test_slogdet_1x1_pos_det',
    'test_slogdet_distinct_singular_values',
    'test_slogdet_neg_det',
    'test_slogdet_pos_det',
    'test_slogdet_symmetric',
    'test_slogdet_symmetric_pd',
    'test_slogdet_batched_1x1_neg_det',
    'test_slogdet_batched_pos_det',
    'test_slogdet_batched_symmetric',
    'test_slogdet_batched_symmetric_pd',
    'test_slogdet_batched_distinct_singular_values'
}

# chunk returns a list in scripting and we don't unpack the list,
# Thus it won't be replaced by ConstantChunk and run AD.
# It's explicitly checked in test_chunk_constant_script_ad
# Similary for split, it's replaced by split_with_sizes in tracing,
# but we don't have AD formula for aten::split(Tensor, int[], int),
# an op registered in JIT so AD is not triggered in scripting.
EXCLUDE_SCRIPT_AD_CHECK = {
    'test_chunk',
    'test_chunk_dim',
    'test_chunk_dim_neg0',
    'test_split_size_list',
    'test_split_size_list_dim',
    'test_split_size_list_dim_neg0',
    'test_tensor_indices_sections',
    'test_tensor_indices_sections_dim',
    'test_tensor_indices_sections_dim_neg0',
    'test_tensor_split_sections',
    'test_tensor_split_sections_dim',
    'test_tensor_split_sections_dim_neg0'
}

EXCLUDE_PYTHON_PRINT = {
    # no support for BroadcastingList in python printer
    'test_nn_max_unpool1d',
    'test_nn_max_unpool2d',
    'test_nn_max_unpool3d',
    'test_nn_max_pool1d',
    'test_nn_max_pool2d',
    'test_nn_max_pool3d',
    'test_nn_max_pool1d_with_indices',
}

EXCLUDE_ALIAS = {
    # aliases, which may appear in method_tests but are tested elsewhere
    'true_divide',

    # Disable tests for lu from common_methods_invocations.py
    # TODO(@nikitaved) Enable jit tests once autograd.Function does support scripting
    'lu'
}

# These classes are placeholders for JIT-generated module and functional tests
class TestJitGeneratedModule(JitTestCase):
    pass

class TestJitGeneratedFunctional(JitTestCase):
    pass

# UBSAN per-function exclusions don't seem to work with OpenMP pragmas,
# and we have to disable the failing tests here instead.
UBSAN_DISABLED_TESTS = [
    "test___rdiv___constant",
    "test___rdiv___scalar_constant",
]
    "test_addcdiv",
    "test_addcdiv_broadcast_all",
    "test_addcdiv_broadcast_rhs",
    "test_addcdiv_scalar",
    "test_addcdiv_scalar_broadcast_lhs",
    "test_addcdiv_scalar_broadcast_rhs",
    "test_addcdiv_scalar_scale",
    "test_addcdiv_scalar_scale_broadcast_lhs",
    "test_addcdiv_scalar_scale_broadcast_rhs",
    "test_addcdiv_scale",
    "test_addcdiv_scale_broadcast_all",
    "test_addcdiv_scale_broadcast_rhs",
    "test_add_broadcast_all",
    "test_add_broadcast_lhs",
    "test_add_broadcast_rhs",
    "test_add_constant",
    "test_add_scalar",
    "test_add_scalar_broadcast_lhs",
    "test_add_scalar_broadcast_rhs",
    "test_div",
    "test_div_broadcast_all",
    "test_div_broadcast_lhs",
    "test_div_broadcast_rhs",
    "test_div_scalar",
    "test_div_scalar_broadcast_lhs",
    "test_div_scalar_broadcast_rhs",
    "test_rsqrt",
    "test_rsqrt_scalar",
    "test_add",
    "test_reciprocal",
    "test_reciprocal_scalar",



    # 测试函数：test_addcdiv
    # 测试函数：test_addcdiv_broadcast_all
    # 测试函数：test_addcdiv_broadcast_rhs
    # 测试函数：test_addcdiv_scalar
    # 测试函数：test_addcdiv_scalar_broadcast_lhs
    # 测试函数：test_addcdiv_scalar_broadcast_rhs
    # 测试函数：test_addcdiv_scalar_scale
    # 测试函数：test_addcdiv_scalar_scale_broadcast_lhs
    # 测试函数：test_addcdiv_scalar_scale_broadcast_rhs
    # 测试函数：test_addcdiv_scale
    # 测试函数：test_addcdiv_scale_broadcast_all
    # 测试函数：test_addcdiv_scale_broadcast_rhs
    # 测试函数：test_add_broadcast_all
    # 测试函数：test_add_broadcast_lhs
    # 测试函数：test_add_broadcast_rhs
    # 测试函数：test_add_constant
    # 测试函数：test_add_scalar
    # 测试函数：test_add_scalar_broadcast_lhs
    # 测试函数：test_add_scalar_broadcast_rhs
    # 测试函数：test_div
    # 测试函数：test_div_broadcast_all
    # 测试函数：test_div_broadcast_lhs
    # 测试函数：test_div_broadcast_rhs
    # 测试函数：test_div_scalar
    # 测试函数：test_div_scalar_broadcast_lhs
    # 测试函数：test_div_scalar_broadcast_rhs
    # 测试函数：test_rsqrt
    # 测试函数：test_rsqrt_scalar
    # 测试函数：test_add
    # 测试函数：test_reciprocal
    # 测试函数：test_reciprocal_scalar
]

# 定义常量 L，表示某个长度的值为 20
L = 20
# 定义常量 M，表示某个长度的值为 10
M = 10
# 定义常量 S，表示某个长度的值为 5
S = 5

# 定义函数 add_nn_module_test，接受任意位置参数和关键字参数
def add_nn_module_test(*args, **kwargs):
    # 检查 kwargs 中是否包含 'no_grad' 键，若不包含则设置 no_grad 为 False
    no_grad = False if 'no_grad' not in kwargs else kwargs['no_grad']

    # 如果 kwargs 中包含 'desc' 键并且其值包含 'eval'，则跳过这些测试
    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        return

    # 通过 get_nn_mod_test_name 函数获取测试名称
    test_name = get_nn_mod_test_name(**kwargs)

    # 使用 @suppress_warnings 装饰器来禁止警告
    @suppress_warnings
    # 如果 kwargs 中包含 'slowTest' 键，则将 do_test 标记为慢速测试
    if 'slowTest' in kwargs:
        do_test = slowTest(do_test)

    # 将测试添加到测试类 TestJitGeneratedModule 中
    post_add_test(test_name, (), do_test, TestJitGeneratedModule)


# 定义函数 post_add_test，用于将测试添加到指定的测试类中
def post_add_test(test_name, skipTestIf, do_test, test_class):
    # 断言确保测试类中没有重名的测试名称
    assert not hasattr(test_class, test_name), 'Two tests have the same name: ' + test_name

    # 对 skipTestIf 中的每个 skip 函数，应用到 do_test 上
    for skip in skipTestIf:
        do_test = skip(do_test)

    # 如果不是 UBSAN 测试或者当前测试名称不在 UBSAN_DISABLED_TESTS 中，则将 do_test 设置为测试类的属性
    if not (TEST_WITH_UBSAN and test_name in UBSAN_DISABLED_TESTS):
        setattr(test_class, test_name, do_test)


# 定义函数 normalize_check_ad，用于标准化 check_ad 的格式
def normalize_check_ad(check_ad, name):
    # 如果 check_ad 长度为 0，则设定默认值为 [False, ['aten::' + name], []]
    if len(check_ad) == 0:
        check_ad = [False, ['aten::' + name], []]
    # 如果 check_ad 长度为 1，则设定默认值为 [check_ad[0], ['aten::' + name], []]
    elif len(check_ad) == 1:
        check_ad = [check_ad[0], ['aten::' + name], []]
    # 如果 check_ad 长度为 2，则保持不变
    elif len(check_ad) == 2:
        check_ad = [check_ad[0], check_ad[1], []]
    # 如果 check_ad 长度为 3，则将其转换为列表
    elif len(check_ad) == 3:
        check_ad = list(check_ad)
    else:
        # 抛出异常，指示 check_ad 格式无效
        raise Exception('Invalid check_ad, requires (bool, str|List[str], str|List[str])')  # noqa: TRY002

    # 对 check_ad 中的每个元素，如果是字符串则转换为列表的形式
    check_ad = [[t] if isinstance(t, str) else t for t in check_ad]

    # 返回标准化后的 check_ad
    return check_ad


# 定义测试类 TestProducerVersion，继承自 TestCase
class TestProducerVersion(TestCase):

    # 定义测试方法 test_version
    def test_version(self):
        # 发布的 GitHub 问题 gh-32561
        self.assertTrue(torch.__version__.startswith(torch.onnx.producer_version))


# 遍历 module_tests、new_module_tests 和 additional_module_tests 中的每个测试，并添加到 add_nn_module_test 函数中执行
for test in module_tests + new_module_tests + additional_module_tests:
    add_nn_module_test(**test)

# 对 criterion_tests 中的每个测试，设置 'no_grad' 为 True，并添加到 add_nn_module_test 函数中执行
for test in criterion_tests:
    test['no_grad'] = True
    add_nn_module_test(**test)

# 如果当前脚本作为主程序运行，则启用默认数据类型检查，并执行所有测试
if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
    import jit.test_module_interface
    # 查找 jit.test_module_interface 中的测试用例
    suite = unittest.findTestCases(jit.test_module_interface)
    # 运行测试套件并输出结果
    unittest.TextTestRunner().run(suite)
```