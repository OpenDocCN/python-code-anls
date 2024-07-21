# `.\pytorch\torch\_dynamo\trace_rules.py`

```py
# 引入需要的模块和类型定义
# mypy: allow-untyped-defs 允许未类型化的定义
import _collections_abc  # 导入 _collections_abc 模块
import _weakrefset  # 导入 _weakrefset 模块
import abc  # 导入 abc 模块
import builtins  # 导入 builtins 模块
import collections  # 导入 collections 模块
import contextlib  # 导入 contextlib 模块
import copy  # 导入 copy 模块
import copyreg  # 导入 copyreg 模块
import dataclasses  # 导入 dataclasses 模块
import enum  # 导入 enum 模块
import functools  # 导入 functools 模块
import importlib  # 导入 importlib 模块
import inspect  # 导入 inspect 模块
import itertools  # 导入 itertools 模块
import linecache  # 导入 linecache 模块
import logging  # 导入 logging 模块
import multiprocessing  # 导入 multiprocessing 模块
import operator  # 导入 operator 模块
import os  # 导入 os 模块
import posixpath  # 导入 posixpath 模块
import random  # 导入 random 模块
import re  # 导入 re 模块
import selectors  # 导入 selectors 模块
import signal  # 导入 signal 模块
import sys  # 导入 sys 模块
import tempfile  # 导入 tempfile 模块
import threading  # 导入 threading 模块
import tokenize  # 导入 tokenize 模块
import traceback  # 导入 traceback 模块
import types  # 导入 types 模块
import typing  # 导入 typing 模块
import unittest  # 导入 unittest 模块
import weakref  # 导入 weakref 模块
from collections import defaultdict  # 从 collections 模块中导入 defaultdict 类
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union  # 从 typing 模块中导入各种类型

np: Optional[types.ModuleType] = None  # 初始化 np 变量为 Optional 类型的 types.ModuleType，初始值为 None
try:
    import numpy as np  # 尝试导入 numpy 模块，将其赋值给 np 变量
except ModuleNotFoundError:
    pass  # 如果导入失败，什么都不做

import torch  # 导入 torch 模块
import torch._inductor.test_operators  # 导入 torch._inductor.test_operators 模块
import torch.distributed  # 导入 torch.distributed 模块
import torch.utils._content_store  # 导入 torch.utils._content_store 模块
from ..utils import _config_module  # 从上级目录中的 utils 模块导入 _config_module
from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX  # 从当前目录的 resume_execution 模块中导入 TORCH_DYNAMO_RESUME_IN_PREFIX
from .utils import getfile, hashable, NP_SUPPORTED_MODULES, unwrap_if_wrapper  # 从当前目录的 utils 模块中导入 getfile, hashable, NP_SUPPORTED_MODULES, unwrap_if_wrapper

from .variables import (  # 从当前目录的 variables 模块中导入以下类和变量
    BuiltinVariable,
    FunctorchHigherOrderVariable,
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    TorchInGraphFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)

if typing.TYPE_CHECKING:
    from .variables.base import VariableTracker  # 如果是类型检查阶段，从当前目录的 variables/base 模块导入 VariableTracker 类

"""
关于跳过/内联规则的说明:

Dynamo 会查询此文件以确定是否应将函数内联或跳过。

跳过应用于帧边界，这意味着 Dynamo 在帧的开始处触发图中断或尝试跟踪/内联整个帧。
当跳过帧时，仍会跟踪递归调用的帧，除非它们也被跳过。

跳过文件（在文件级别而不是函数级别跳过）仍然适用于 Dynamo 跟踪时的帧边界，但适用于该文件中的所有函数。

@skip 是一个辅助装饰器，可以应用于您的函数，使其包含在此处。

Dynamo 的跳过/内联规则和优先级定义如下:
* 内联是默认行为，除非显式跳过。
* Dynamo 有两个 SKIPLIST: BUILTIN_SKIPLIST 和 THIRDPARTY_SKIPLIST。
    * BUILTIN_SKIPLIST 包含内置的 Python 模块，如 abc、collections 等。
    * THIRDPARTY_SKIPLIST 包含常见的第三方库，如 numpy、pandas 等。
* 这两个 SKIPLIST 中的函数始终被跳过，除非:
    * 它们在 `manual_torch_name_rule_map` 中有显式定义的规则;
    * 相应的 Python 模块已放入 MOD_INLINELIST。
* PyTorch(torch) 默认在 BUILTIN_SKIPLIST 中，但有许多情况下我们希望内联 torch 命名空间下的函数。
    我们应该指定 `manual_torch_name_rule_map` 中函数的内联，或将相应的 Python 模块放入 MOD_INLINELIST 以使 Dynamo 内联它们。
* 如果调用被跳过的模块/文件中的函数，Dynamo 将包装这些函数
"""
    as SkipFunctionVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFunctionVariable.call_function.
"""
Overall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)

To figure out what the behavior is, check the following list in order:
* `manual_torch_name_rule_map` (Inline if YES)
* MOD_INLINELIST (Inline if YES)
* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)
* Inline by default

In general, if you want to force inline a function or module, please consider adding
the function's python module to MOD_INLINELIST first.
Use the `manual_torch_name_rule_map` only when there are other functions under the same module that
you don't want to inline them.
"""

"""
Map of function objects to their tracing rules (Dynamo variables).
* TorchInGraphFunctionVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* SkipFunctionVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map_updated. To fix the failures:
If you are adding a new torch level API or Dynamo implementation:
* Add the name with the corresponding tracing rule to this map
  if you are adding a new in graph function or Dynamo implementation for an existing function.
* Remove the object name from test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names if it's there.

If you are removing an existing torch level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names
  depends on where it is.


"""
# 手动定义的 Torch 函数名称到追踪规则的映射
manual_torch_name_rule_map = {
    "torch.onnx.is_in_onnx_export": TorchInGraphFunctionVariable,
    "torch.onnx.operators.shape_as_tensor": TorchInGraphFunctionVariable,
    "torch.overrides.is_tensor_like": TorchInGraphFunctionVariable,
    "torch.jit.is_scripting": TorchInGraphFunctionVariable,
    "torch.jit.is_tracing": TorchInGraphFunctionVariable,
    "torch.jit.annotate": TorchInGraphFunctionVariable,
    "torch.distributed.is_available": TorchInGraphFunctionVariable,
    "torch.distributed.is_initialized": TorchInGraphFunctionVariable,
    "torch.distributed.get_rank": TorchInGraphFunctionVariable,
    "torch.distributed.get_world_size": TorchInGraphFunctionVariable,
    "torch.distributed._tensor.api.DTensor#from_local": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._get_group_size_by_name": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._resolve_group_name_by_ranks_and_tag": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._get_group_tag": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d.get_process_group_ranks": TorchInGraphFunctionVariable,
    "torch._utils.is_compiling": TorchInGraphFunctionVariable,
}
    # 定义一系列键值对，用于标识需要在Torch JIT图中处理的函数或变量，以及如何处理它们
    {
        # 在Torch JIT图中跟踪的符号操作，表示这些函数或变量是Torch中的图函数变量
        "torch.fx._symbolic_trace.is_fx_tracing": TorchInGraphFunctionVariable,
        "torch._dynamo.external_utils.is_compiling": TorchInGraphFunctionVariable,
        "torch.compiler.is_compiling": TorchInGraphFunctionVariable,
        "torch.compiler.is_dynamo_compiling": TorchInGraphFunctionVariable,
        # 在自动微分和性能分析中启用的变量，由于不在图中使用，因此跳过处理
        "torch.autograd._profiler_enabled": SkipFunctionVariable,
        "torch._C._to_dlpack": SkipFunctionVariable,
        "torch.to_dlpack": SkipFunctionVariable,
        # 不在AOT调度器中处理的RNG状态设置或获取函数，因为它们不是aten操作
        # 导致AOT图不包含这些setter或getter函数，生成了错误的RNG状态图
        "torch.default_generator#get_state": SkipFunctionVariable,
        "torch._C.Generator#get_state": SkipFunctionVariable,
        "torch.get_rng_state": SkipFunctionVariable,
        "torch.cuda.get_rng_state": SkipFunctionVariable,
        "torch.default_generator#set_state": SkipFunctionVariable,
        "torch._C.Generator#set_state": SkipFunctionVariable,
        "torch.set_rng_state": SkipFunctionVariable,
        "torch.cuda.set_rng_state": SkipFunctionVariable,
        # 解决的问题链接：https://github.com/pytorch/pytorch/issues/107187
        "torch.manual_seed": SkipFunctionVariable,
        # 解决的问题链接：https://github.com/pytorch/pytorch/issues/93501
        "torch.nn.utils.rnn.pack_padded_sequence": SkipFunctionVariable,
        # 在图函数中处理的参数和张量
        "torch.nn.Parameter": TorchInGraphFunctionVariable,
        # 不在图中处理的其他特定函数或变量
        "torch._nested_tensor_from_mask": SkipFunctionVariable,
        "torch._nested_from_padded": SkipFunctionVariable,
        "torch.nested.nested_tensor_from_jagged": UserFunctionVariable,
        # 在Torch JIT图中处理的符号运算符实现
        "torch.sym_not": TorchInGraphFunctionVariable,
        "torch.sym_float": TorchInGraphFunctionVariable,
        "torch.sym_int": TorchInGraphFunctionVariable,
        "torch.sym_max": TorchInGraphFunctionVariable,
        "torch.sym_min": TorchInGraphFunctionVariable,
        "torch.sym_sqrt": TorchInGraphFunctionVariable,
        "torch.sym_ite": TorchInGraphFunctionVariable,
        # 不在图中处理的张量方法和构造函数
        "torch.Tensor#_make_wrapper_subclass": SkipFunctionVariable,
        "torch.Tensor#__init__": SkipFunctionVariable,
        # 不在图中处理的CUDA相关函数
        "torch.cuda.set_device": SkipFunctionVariable,
        "torch.cuda.current_device": SkipFunctionVariable,
        # 不在图中处理的自动转换相关函数
        "torch._C.autocast_decrement_nesting": SkipFunctionVariable,
        "torch._C.autocast_increment_nesting": SkipFunctionVariable,
        # 不在图中处理的自动微分和梯度计算函数
        "torch.autograd.grad": SkipFunctionVariable,
        "torch.autograd.backward": SkipFunctionVariable,
        "torch._C.clear_autocast_cache": SkipFunctionVariable,
        # 不在图中处理的分布相关函数
        "torch.distributions.constraints.is_dependent": SkipFunctionVariable,
        # 不在图中处理的类型检查函数
        "torch.jit.isinstance": SkipFunctionVariable,
        # 不在图中处理的异常和自动转换缓存相关函数
        "torch._C.set_anomaly_enabled": SkipFunctionVariable,
        "torch._C.set_autocast_cache_enabled": SkipFunctionVariable,
        "torch._C.set_autocast_cpu_dtype": SkipFunctionVariable,
    }
    # 定义一系列的映射关系，将函数名映射到相应的变量类型
    "torch._C.set_autocast_cpu_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_gpu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_ipu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_ipu_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_xla_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_xla_enabled": SkipFunctionVariable,
    "torch.resize_as_": SkipFunctionVariable,
    "torch.resize_as_sparse_": SkipFunctionVariable,
    "torch.get_default_device": TorchInGraphFunctionVariable,
    
    # functorch/vmap 模块下的用户自定义函数映射
    "torch._functorch.vmap._check_int_or_none": UserFunctionVariable,
    "torch._functorch.vmap._check_out_dims_is_int_or_int_pytree": UserFunctionVariable,
    "torch._functorch.vmap._check_randomness_arg": UserFunctionVariable,
    "torch._functorch.vmap._chunked_vmap": UserFunctionVariable,
    "torch._functorch.vmap._concat_chunked_outputs": UserFunctionVariable,
    "torch._functorch.vmap._create_batched_inputs": UserFunctionVariable,
    "torch._functorch.vmap._flat_vmap": UserFunctionVariable,
    "torch._functorch.vmap._flatten_chunks_output": UserFunctionVariable,
    "torch._functorch.vmap._get_chunked_inputs": UserFunctionVariable,
    "torch._functorch.vmap._get_name": UserFunctionVariable,
    "torch._functorch.vmap._maybe_remove_batch_dim": UserFunctionVariable,
    "torch._functorch.vmap._num_outputs": UserFunctionVariable,
    "torch._functorch.vmap._process_batched_inputs": UserFunctionVariable,
    "torch._functorch.vmap._unwrap_batched": UserFunctionVariable,
    "torch._functorch.vmap._validate_and_get_batch_size": UserFunctionVariable,
    "torch._functorch.vmap.doesnt_support_saved_tensors_hooks": UserFunctionVariable,
    "torch._functorch.vmap.get_chunk_sizes": UserFunctionVariable,
    "torch._functorch.vmap.restore_vmap": UserFunctionVariable,
    "torch._functorch.apis.vmap": UserFunctionVariable,
    "torch._functorch.vmap.unwrap_batched": UserFunctionVariable,
    "torch._functorch.vmap.vmap_impl": FunctorchHigherOrderVariable,
    "torch._functorch.vmap.wrap_batched": UserFunctionVariable,
    
    # functorch/grad 模块下的函数映射
    "torch._functorch.eager_transforms.grad_impl": FunctorchHigherOrderVariable,
    "torch._functorch.apis.grad_and_value": UserFunctionVariable,
    "torch._functorch.eager_transforms._as_tuple": UserFunctionVariable,
    "torch._functorch.eager_transforms._check_unique_non_empty": UserFunctionVariable,
    "torch._functorch.eager_transforms._create_differentiable": UserFunctionVariable,
    "torch._functorch.eager_transforms._slice_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms._undo_create_differentiable": UserFunctionVariable,
    "torch._functorch.eager_transforms._validate_and_wrap_argnum": UserFunctionVariable,
    
    # functorch/vmap 模块下的一条注释，指出当前 Dynamo 不支持的功能
    # "torch._functorch.vmap.lazy_load_decompositions": UserFunctionVariable,
    # 定义一系列变量，用于不同的 Torch 函数和模块
    "torch._functorch.eager_transforms._validate_and_wrap_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms._wrap_all_tensors": UserFunctionVariable,
    "torch._functorch.eager_transforms._wrap_tensor_for_grad": UserFunctionVariable,

    # functorch/jacrev 模块中的变量
    "torch._functorch.eager_transforms.jacrev": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms.error_if_complex": UserFunctionVariable,
    "torch._functorch.eager_transforms._chunked_standard_basis_for_": UserFunctionVariable,
    "torch._functorch.eager_transforms._safe_zero_index": UserFunctionVariable,

    # functorch/vjp 模块中的变量
    "torch._functorch.eager_transforms.vjp": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._vjp_with_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_non_empty_tensor_output": UserFunctionVariable,

    # functorch/jvp 模块中的变量
    "torch._functorch.eager_transforms._jvp_with_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms.jvp": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._replace_args": UserFunctionVariable,
    "torch._functorch.eager_transforms.safe_unpack_dual": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_non_empty_list_of_tensors": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_output_is_tensor_or_tensors": UserFunctionVariable,

    # torch.autograd.forward_ad 模块中的变量
    "torch.autograd.forward_ad.enter_dual_level": UserFunctionVariable,
    "torch.autograd.forward_ad.exit_dual_level": UserFunctionVariable,
    "torch.autograd.forward_ad.make_dual": UserFunctionVariable,
    "torch.autograd.forward_ad.unpack_dual": UserFunctionVariable,

    # functorch/linearize 模块中的变量
    "torch._functorch.eager_transforms.linearize": FunctorchHigherOrderVariable,

    # functorch/jacfwd 模块中的变量
    "torch._functorch.eager_transforms.jacfwd": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._construct_standard_basis_for": UserFunctionVariable,
    "torch._functorch.eager_transforms.safe_unflatten": UserFunctionVariable,

    # functorch/hessian 模块中的变量
    "torch._functorch.eager_transforms.hessian": FunctorchHigherOrderVariable,

    # functorch/deprecated 模块中的变量
    "torch._functorch.deprecated.jvp": UserFunctionVariable,
    "torch._functorch.deprecated.hessian": UserFunctionVariable,
    "torch._functorch.deprecated.jacfwd": UserFunctionVariable,
    "torch._functorch.deprecated.jacrev": UserFunctionVariable,
    "torch._functorch.deprecated.grad": UserFunctionVariable,
    "torch._functorch.deprecated.grad_and_value": UserFunctionVariable,
    "torch._functorch.deprecated.vjp": UserFunctionVariable,

    # 其他 Torch 函数和模块中的变量
    "torch._constrain_as_size": UserFunctionVariable,
    "torch._tensor._convert": UserFunctionVariable,
    "torch.jit._unwrap_optional": UserFunctionVariable,
    "torch.backends.mha.get_fastpath_enabled": UserFunctionVariable,
    "torch._C._functorch._add_batch_dim": TorchInGraphFunctionVariable,
    # 定义了一系列变量名和对应的变量类型，用于标识不同的功能或模块
    
    "torch._C._functorch._remove_batch_dim": TorchInGraphFunctionVariable,
    # 表示 torch._C._functorch._remove_batch_dim 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch._C._functorch._wrap_for_grad": TorchInGraphFunctionVariable,
    # 表示 torch._C._functorch._wrap_for_grad 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch._C._functorch._unwrap_for_grad": TorchInGraphFunctionVariable,
    # 表示 torch._C._functorch._unwrap_for_grad 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch._C._functorch.maybe_current_level": TorchInGraphFunctionVariable,
    # 表示 torch._C._functorch.maybe_current_level 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch._C._functorch.is_batchedtensor": TorchInGraphFunctionVariable,
    # 表示 torch._C._functorch.is_batchedtensor 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch._dynamo.mark_static": UserFunctionVariable,
    # 表示 torch._dynamo.mark_static 是一个 UserFunctionVariable 类型的变量
    
    "torch.fx.experimental.symbolic_shapes.guard_size_oblivious": TorchInGraphFunctionVariable,
    # 表示 torch.fx.experimental.symbolic_shapes.guard_size_oblivious 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch.cuda._get_device_properties": TorchInGraphFunctionVariable,
    # 表示 torch.cuda._get_device_properties 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch.utils.hooks.BackwardHook": TorchInGraphFunctionVariable,
    # 表示 torch.utils.hooks.BackwardHook 是一个 TorchInGraphFunctionVariable 类型的变量
    
    "torch.sparse_bsc_tensor": SkipFunctionVariable,
    # 表示 torch.sparse_bsc_tensor 是一个 SkipFunctionVariable 类型的变量
    
    "torch.sparse_bsr_tensor": SkipFunctionVariable,
    # 表示 torch.sparse_bsr_tensor 是一个 SkipFunctionVariable 类型的变量
    
    "torch.sparse_csc_tensor": SkipFunctionVariable,
    # 表示 torch.sparse_csc_tensor 是一个 SkipFunctionVariable 类型的变量
    
    "torch.sparse_csr_tensor": SkipFunctionVariable,
    # 表示 torch.sparse_csr_tensor 是一个 SkipFunctionVariable 类型的变量
    
    "torch.sparse_compressed_tensor": SkipFunctionVariable,
    # 表示 torch.sparse_compressed_tensor 是一个 SkipFunctionVariable 类型的变量
    
    "torch._C._autograd._unsafe_set_version_counter": TorchInGraphFunctionVariable,
    # 表示 torch._C._autograd._unsafe_set_version_counter 是一个 TorchInGraphFunctionVariable 类型的变量
    
    # 避免在分布式单元测试中跳过用户定义的模块
    "torch/testing/_internal/common_fsdp.py#forward": UserFunctionVariable,
    # 表示 torch/testing/_internal/common_fsdp.py#forward 是一个 UserFunctionVariable 类型的变量
    
    f"torch/testing/_internal/common_fsdp.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    # 表示 torch/testing/_internal/common_fsdp.py#{TORCH_DYNAMO_RESUME_IN_PREFIX} 是一个 UserFunctionVariable 类型的变量
    
    "torch/testing/_internal/distributed/_tensor/common_dtensor.py#forward": UserFunctionVariable,
    # 表示 torch/testing/_internal/distributed/_tensor/common_dtensor.py#forward 是一个 UserFunctionVariable 类型的变量
    
    f"torch/testing/_internal/distributed/_tensor/common_dtensor.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    # 表示 torch/testing/_internal/distributed/_tensor/common_dtensor.py#{TORCH_DYNAMO_RESUME_IN_PREFIX} 是一个 UserFunctionVariable 类型的变量
    
    "torch/testing/_internal/common_distributed.py#forward": UserFunctionVariable,
    # 表示 torch/testing/_internal/common_distributed.py#forward 是一个 UserFunctionVariable 类型的变量
    
    f"torch/testing/_internal/common_distributed.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    # 表示 torch/testing/_internal/common_distributed.py#{TORCH_DYNAMO_RESUME_IN_PREFIX} 是一个 UserFunctionVariable 类型的变量
}


# 在图函数中（包括常数折叠）中的C绑定中的torch函数
torch_c_binding_in_graph_functions = dict.fromkeys(
    # 空字典，用于存储TorchInGraphFunctionVariable类型的对象
    ],
    TorchInGraphFunctionVariable,
)


if sys.version_info >= (3, 9):
    # 如果Python版本大于等于3.9，将"math.lcm"映射到TorchInGraphFunctionVariable
    torch_c_binding_in_graph_functions["math.lcm"] = TorchInGraphFunctionVariable
if sys.version_info >= (3, 11):
    # 如果Python版本大于等于3.11，将"math.exp2"和"math.cbrt"映射到TorchInGraphFunctionVariable
    torch_c_binding_in_graph_functions["math.exp2"] = TorchInGraphFunctionVariable
    torch_c_binding_in_graph_functions["math.cbrt"] = TorchInGraphFunctionVariable


# 不在C绑定中的图函数（包括常数折叠）
torch_non_c_binding_in_graph_functions = dict.fromkeys(
    # 空字典，用于存储TorchInGraphFunctionVariable类型的对象
    ],
    TorchInGraphFunctionVariable,
)


torch_name_rule_map = [
    manual_torch_name_rule_map,  # 手动定义的torch名称规则映射
    torch_c_binding_in_graph_functions,  # 在图函数中的C绑定中的torch函数映射
    torch_non_c_binding_in_graph_functions,  # 不在C绑定中的图函数中的torch函数映射
]


"""
生成torch对象 - Dynamo追踪规则（包装变量）映射。
"""


@functools.lru_cache(None)
def get_torch_obj_rule_map():
    d: Dict[Any, VariableTracker] = dict()
    for m in torch_name_rule_map:
        for k, v in m.items():  # type: ignore[attr-defined]
            if ".py#" not in k:
                obj = load_object(k)  # 加载对象k
            else:
                obj = _module_dir(torch) + k[len("torch/") :]
            if obj is not None:
                if obj in d and d[obj] != v:
                    raise AssertionError(
                        f"Duplicate torch object {obj} with different rules: {v}, {d[obj]}"
                    )
                else:
                    d[obj] = v
    return d


def _load_obj_from_str(fully_qualified_name):
    # 从字符串中加载对象
    module, obj_name = fully_qualified_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), obj_name)


"""
加载字符串表示的torch对象。
"""


def load_object(name):
    try:
        x = name.split("#")
        if len(x) == 2:
            obj = _load_obj_from_str(x[0])
            val = getattr(obj, x[1])  # 获取对象obj的属性x[1]
        else:
            assert len(x) == 1, f"Invalid obj name {name}"
            val = _load_obj_from_str(x[0])  # 加载对象x[0]
        val = unwrap_if_wrapper(val)  # 如果是包装器，则解包
    except (AttributeError, ImportError):
        val = None
    return val


"""
获取所有允许用于图函数的torch.Tensor方法。
"""


@functools.lru_cache(None)
def get_tensor_method():
    s = set()
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(
            method, (types.MethodDescriptorType, types.WrapperDescriptorType)
        ):
            s.add(method)
    return frozenset(s)


"""
返回torch对象是否为ATen操作或torch.Tensor方法。
"""


def is_aten_op_or_tensor_method(obj):
    return obj in get_tensor_method() or isinstance(
        obj,
        (torch._ops.OpOverloadPacket, torch._ops.OpOverload),
    )


class FunctionIdSet:
    """
    跟踪允许或不允许进入生成的FX图的对象的`id()`集合。用于测试torch.*、numpy.*、builtins.*等。
    """
    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """

    function_ids: Optional[Set[int]] = None  # 定义可选的整数集合变量 function_ids，初始为 None
    function_names: Optional[Dict[int, str]] = None  # 定义可选的整数到字符串字典变量 function_names，初始为 None

    def __init__(self, lazy_initializer: Callable[[], Union[Dict[int, str], Set[int]]]):
        self.lazy_initializer = lazy_initializer  # 初始化 lazy_initializer，它是一个可调用对象

    def __call__(self):
        if self.function_ids is None:
            value = self.lazy_initializer()  # 调用 lazy_initializer 初始化数据
            if isinstance(value, dict):
                self.function_ids = set(value.keys())  # 如果返回值是字典，则将键转换为集合存入 function_ids
                self.function_names = value  # 同时将整个字典存入 function_names
            else:
                assert isinstance(value, set)
                self.function_ids = value  # 如果返回值是集合，则直接存入 function_ids
        return self.function_ids  # 返回当前的 function_ids

    def get_name(self, idx: int, default: str):
        self()  # 调用 lazy 初始化函数
        assert self.function_names is not None  # 断言确保 function_names 不为 None
        return self.function_names.get(idx, default)  # 返回 function_names 中 idx 对应的名称或默认值

    def add(self, idx: int):
        function_ids = self()  # 调用 lazy 初始化函数
        function_ids.add(idx)  # 向 function_ids 集合中添加指定的 idx

    def remove(self, idx: int):
        function_ids = self()  # 调用 lazy 初始化函数
        if idx in function_ids:
            function_ids.remove(idx)  # 如果 idx 存在于 function_ids 中，则移除它

    def __contains__(self, idx: int):
        return idx in self()  # 判断 idx 是否在 lazy 初始化后的 function_ids 中
# 注册函数ID集合装饰器，返回空字典，用于存储允许的可调用函数ID和名称的映射关系
@FunctionIdSet
def _allowed_callable_ids() -> Dict[int, str]:
    rv: Dict[int, str] = {}
    return rv

# 注册函数ID集合装饰器，返回空字典，用于存储不允许的可调用函数ID和名称的映射关系
@FunctionIdSet
def _disallowed_callable_ids() -> Dict[int, str]:
    rv: Dict[int, str] = {}
    return rv

# 注册函数ID集合装饰器，返回包含内置函数ID和名称映射的字典
@FunctionIdSet
def _builtin_function_ids() -> Dict[int, str]:
    # 初始化空字典
    rv = {}
    # 遍历内置模块 builtins 中的所有项，如果项不以下划线开头且为可调用对象，则添加到字典中
    rv.update(
        {
            id(v): f"builtins.{k}"
            for k, v in builtins.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    # 遍历操作符模块 operator 中的所有项，如果项不以下划线开头且为可调用对象，则添加到字典中
    rv.update(
        {
            id(v): f"operator.{k}"
            for k, v in operator.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    # 添加 itertools 模块中 itertools.chain 和 itertools.islice 函数的ID和名称到字典中
    rv.update(
        {id(v): f"functools.{v.__name__}" for v in (itertools.chain, itertools.islice)}
    )
    # 添加 typing 模块中 typing.cast 函数的ID和名称，以及 functools 模块中 functools.reduce 和 copy 模块中 copy.deepcopy 函数的ID和名称到字典中
    rv.update(
        {
            id(cast): "typing.cast",
            id(functools.reduce): "functools.reduce",
            id(copy.deepcopy): "copy.deepcopy",
        }
    )
    return rv

# 注册函数ID集合装饰器，返回包含 NumPy 模块中函数ID和名称映射的字典
@FunctionIdSet
def _numpy_function_ids() -> Dict[int, str]:
    # 初始化空字典
    rv = dict()
    # 遍历 NP_SUPPORTED_MODULES 中的模块，将每个模块中的可调用对象的ID和名称添加到字典中
    for mod in NP_SUPPORTED_MODULES:
        rv.update(
            {
                id(v): f"{mod.__name__}.{k}"
                for k, v in mod.__dict__.items()
                if callable(v)
                and (getattr(v, "__module__", None) or mod.__name__) == mod.__name__
            }
        )
    return rv

# 注册函数ID集合装饰器，返回包含内置常量的ID和名称映射的字典
@FunctionIdSet
def _builtin_constant_ids() -> Dict[int, str]:
    """
    Collects constant builtins by eliminating callable items.
    """
    # 初始化空字典
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }
    return rv

# 初始化一个默认字典，用于延迟初始化模块函数
_lazy_module_init: Dict[str, List[Callable[[], None]]] = defaultdict(list)

def add_module_init_func(name: str, init_func: Callable[[], None]) -> None:
    """注册一个模块的初始化函数，避免过早地导入它"""
    # 如果模块已经导入，则立即运行初始化函数
    assert "." not in name, f"Expected a root module name, but got {name}"
    assert name not in _lazy_module_init
    _lazy_module_init[name].append(init_func)

def _maybe_init_lazy_module(obj: object) -> None:
    """可能初始化延迟加载的模块"""
    # 获取对象所属的模块名称
    module = getattr(obj, "__module__", None)
    if module is None:
        return

    # 提取基础模块名称，并从 _lazy_module_init 中弹出对应的初始化函数列表
    base_module = module.split(".")[0]
    init_funcs = _lazy_module_init.pop(base_module, None)
    if init_funcs is not None:
        # 依次执行每个初始化函数
        for fn in init_funcs:
            fn()

def is_callable_allowed(obj) -> bool:
    """检查对象是否在允许的可调用函数集合中"""
    _maybe_init_lazy_module(obj)
    return id(obj) in _allowed_callable_ids

def is_callable_disallowed(obj) -> bool:
    """检查对象是否在不允许的可调用函数集合中"""
    _maybe_init_lazy_module(obj)
    return id(obj) in _disallowed_callable_ids

def is_forbidden(obj) -> bool:
    """检查对象是否标记为禁止访问"""
    _maybe_init_lazy_module(obj)
    return inspect.getattr_static(obj, "_dynamo_forbidden", False)

def is_builtin_callable(obj) -> bool:
    """检查对象是否为内置可调用函数"""
    return id(obj) in _builtin_function_ids

def is_builtin_constant(obj) -> bool:
    """检查对象是否为内置常量"""
    return id(obj) in _builtin_constant_ids

def is_numpy(obj) -> bool:
    """检查对象是否属于 NumPy"""
    if np is None:
        return False
    # 检查对象是否是 NumPy 数组（np.ndarray）或者 NumPy 通用类型（np.generic）
    return isinstance(obj, (np.ndarray, np.generic)) or id(obj) in _numpy_function_ids
# 检查对象是否为 NumPy 的数据类型
def is_numpy_dtype(obj) -> bool:
    # 如果 NumPy 未导入，返回 False
    if np is None:
        return False
    # 判断对象是否为 NumPy 的数据类型
    return isinstance(obj, np.dtype)


# 检查对象是否为 NumPy 的类型信息（如 np.finfo 或 np.iinfo）
def is_numpy_type_info(obj) -> bool:
    # 如果 NumPy 未导入，返回 False
    if np is None:
        return False
    # 判断对象是否为 NumPy 的类型信息之一
    return isinstance(obj, (np.finfo, np.iinfo))


# 内置模块的跳过列表，用于懒加载和跳过特定的 Python 内置模块
BUILTIN_SKIPLIST = (
    abc,
    collections,
    contextlib,
    copy,
    copyreg,
    dataclasses,
    enum,
    functools,
    importlib,
    inspect,
    linecache,
    logging,
    multiprocessing,
    operator,
    os,
    posixpath,
    random,
    re,
    selectors,
    signal,
    tempfile,
    threading,
    tokenize,
    torch,  # torch/* is skipped by default unless specified in FUNC_INLINELIST or MOD_INLINELIST
    traceback,
    types,
    typing,
    unittest,
    weakref,
    _collections_abc,
    _weakrefset,
)

# 第三方库的跳过列表，通过字符串标识第三方库的名称，用于懒加载和跳过特定第三方库
THIRDPARTY_SKIPLIST = (
    "fx2trt_oss",
    "hypothesis",
    "networkx",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tabulate",
    "tensorflow",
    "tensorrt",
    "torch2trt",
    "tqdm",
    "tree",
    "tvm",
    "xarray",
)


# 去除文件名结尾的 '__init__.py' 后缀，返回修改后的文件名
def _strip_init_py(s):
    # TODO: Once we require py3.9 use removesuffix instead.
    suffix = "__init__.py"
    # 如果文件名以 '__init__.py' 结尾，移除该后缀
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


# 获取模块的目录路径，排除不包含 '__file__' 属性的情况，例如冻结模块
def _module_dir(m: types.ModuleType):
    # 如果模块没有导出 '__file__' 属性，返回 None
    file = getattr(m, "__file__", None)
    # 如果文件属性存在，返回去除 '__init__.py' 后缀后的文件名
    return file and _strip_init_py(file)


# 这些是旧版解决方案，不要向此列表添加新模块。请使用 MOD_INLINELIST 来强制内联特定模块下的函数。
LEGACY_MOD_INLINELIST = {
    "torch._dynamo.external_utils",
    "torch._export.db.examples",
    "torch._export.wrappers",
    "torch._functorch.apis",
    "torch._functorch.deprecated",
    "torch._higher_order_ops.cond",
    "torch.ao.quantization.pt2e.export_utils",
    "torch.ao.quantization.pt2e.qat_utils",
    "torch.ao.quantization.pt2e.representation.rewrite",
    "torch.ao.quantization.pt2e.utils",
    "torch.ao.quantization.quantizer.xnnpack_quantizer",
    "torch.optim",
}

# 如果 torch.distributed 可用，则添加额外的模块到 LEGACY_MOD_INLINELIST
if torch.distributed.is_available():
    LEGACY_MOD_INLINELIST |= {
        "torch.distributed._tensor.api",
        "torch.distributed._tensor.device_mesh",
        "torch.distributed.device_mesh",
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
        "torch.distributed.tensor.parallel._data_parallel_utils",
        "torch.distributed.tensor.parallel._utils",
        "torch.distributed.tensor.parallel.style",
        # 为了确保前向钩子不被忽略，我们必须将 'replicate' 添加到 LEGACY_MOD_INLINELIST 中。
        "torch.distributed._composable.replicate",
    }
# 我们使用 Python 模块的名称而不是文件或目录对象，以避免循环依赖。
# 请保持此处按字母顺序排序。
MOD_INLINELIST = {
    "torch.utils._python_dispatch",
    "torch._refs",
    "torch._prims",
    "torch._decomp",
    "torch._dynamo._trace_wrapped_higher_order_op",
    "torch._dynamo.comptime",
    "torch._dynamo.polyfill",
    "torch._functorch.vmap",
    "torch._functorch.autograd_function",
    "torch._library.custom_ops",
    "torch._functorch.eager_transforms",
    "torch._inductor.test_operators",
    "torch.amp.autocast_mode",
    "torch.ao.nn",
    "torch.autograd.function",
    "torch.backends.cuda",
    "torch.cuda.amp.autocast_mode",
    "torch.distributions",
    "torch.fx._pytree",
    "torch.fx.passes.shape_prop",
    "torch.nn",
    "torch.overrides",
    "torch.random",
    "torch.sparse",
    "torch.testing",
    "torch.testing._internal.hypothesis_utils",
    "torch.utils._content_store",
    "torch.utils._contextlib",
    "torch.utils._foreach_utils",
    "torch.utils._pytree",
    "torch.utils.hooks",
    "torch._tensor",
    "torch._higher_order_ops.strict_mode",
    "torch._higher_order_ops.while_loop",
    "torch._higher_order_ops.associative_scan",
}

# 如果支持分布式，则添加相关模块到 MOD_INLINELIST 中
if torch.distributed.is_available():
    MOD_INLINELIST.add("torch.distributed")
    MOD_INLINELIST.add("torch.distributed._functional_collectives")
    MOD_INLINELIST.add("torch.distributed._composable.replicate")

# 使用 functools.lru_cache(None) 修饰器，定义一个函数，返回一个遗留模块的内联列表
@functools.lru_cache(None)
def get_legacy_mod_inlinelist():
    # 遍历 LEGACY_MOD_INLINELIST 列表，生成模块的路径，并返回路径集合
    inlinelist = {
        _module_dir(torch) + m[len("torch.") :].replace(".", "/")
        for m in LEGACY_MOD_INLINELIST
    }
    return inlinelist

# 使用 functools.lru_cache(None) 修饰器，定义一个函数，返回一个当前模块的内联列表
@functools.lru_cache(None)
def get_mod_inlinelist():
    # 遍历 MOD_INLINELIST 集合，生成模块的路径，并返回路径集合
    inlinelist = {
        _module_dir(torch) + m[len("torch.") :].replace(".", "/")
        for m in MOD_INLINELIST
    }
    return inlinelist

# 跳过一些标准的 Python 内置库的目录
SKIP_DIRS = [
    "<frozen importlib",  # 内置模块 importlib 的路径
    "<__array_function__ internals>",  # 内置模块 __array_function__ internals 的路径
    _config_module.__file__,  # 给定的 _config_module 的文件路径
    "triton/backends",  # 包含 "triton/backends" 的目录
]
# 使用 BUILTIN_SKIPLIST 生成的目录，过滤 None 值后添加到 SKIP_DIRS
SKIP_DIRS.extend(filter(None, (_module_dir(m) for m in BUILTIN_SKIPLIST)))

# 创建正则表达式对象，用于匹配不匹配任何内容的字符串
SKIP_DIRS_RE = re.compile(r"match nothing^")

# 导入 torch._inductor.config 模块中的 is_fbcode() 函数，并赋值给 is_fbcode 变量
is_fbcode = importlib.import_module("torch._inductor.config").is_fbcode()

# 创建正则表达式对象，用于匹配 FBCODE_SKIP_DIRS 中任何字符串
FBCODE_SKIP_DIRS = {
    "torchrec/distributed",
    "torchrec/fb/distributed",
    "caffe2/torch/fb/sparsenn/pooled_embeddings_modules.py",
}
FBCODE_SKIP_DIRS_RE = re.compile(f".*({'|'.join(map(re.escape, FBCODE_SKIP_DIRS))})")

# TODO(yanboliang, anijain2305) - 我们应该解决一些问题：
# 1) 审核是否在 FBCODE_SKIPS_DIR 中真的需要 torchrec/distributed
# 2) 要内联某个目录中的一个文件但跳过其他文件，我们可以使用 manual_torch_name_rule_map，
#    但这很困难，因为 FBCODE 可能会添加不寻常的名称，如 torch_package。
#    因此，这是一个临时解决方案。
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS = {
    "torchrec/distributed/types.py",


# 定义一个字符串，表示文件路径 "torchrec/distributed/types.py"
}

# 正则表达式用于匹配需要跳过的文件名列表中的文件
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE = re.compile(
    f".*({'|'.join(map(re.escape, FBCODE_INLINE_FILES_IN_SKIPPED_DIRS))})"
)

# 强制跳过的文件集合，优先级高于其他规则
FORCE_SKIP_FILES = {f"{_module_dir(torch)}optim/lr_scheduler.py"}


def _recompile_re():
    # 重新编译跳过目录的正则表达式
    global SKIP_DIRS_RE
    SKIP_DIRS_RE = re.compile(rf"^[^\s<]*({'|'.join(map(re.escape, SKIP_DIRS))})")


def add(import_name: str):
    # 将指定模块名添加到跳过目录列表，并重新编译相关的正则表达式
    if isinstance(import_name, types.ModuleType):
        return add(import_name.__name__)
    assert isinstance(import_name, str)
    from importlib.util import find_spec

    module_spec = find_spec(import_name)
    if not module_spec:
        return
    origin = module_spec.origin
    if origin is None:
        return
    SKIP_DIRS.append(_strip_init_py(origin))
    _recompile_re()


@dataclasses.dataclass
class SkipResult:
    skipped: bool
    reason: Optional[str]


def check_file(filename, is_inlined_call=False):
    """判断是否应该跳过此文件？"""
    if filename is None:
        return SkipResult(True, "filename is None")
    if filename in FORCE_SKIP_FILES:
        return SkipResult(True, "FORCE_SKIP_FILES")
    if any(filename.startswith(d) for d in get_legacy_mod_inlinelist()):
        return SkipResult(
            False,
            "LEGACY_MOD_INLINELIST",
        )
    if is_inlined_call and is_torch_inline_allowed(filename):
        return SkipResult(
            False,
            "MOD_INLINELIST",
        )
    if (
        is_fbcode
        and bool(FBCODE_SKIP_DIRS_RE.match(filename))
        and not bool(FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE.match(filename))
    ):
        return SkipResult(
            True,
            "FBCODE_SKIP_DIRS",
        )
    if bool(SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, "SKIP_DIRS")
    else:
        return SkipResult(False, "inlined by default")


@dataclasses.dataclass
class FunctionInfo:
    py_obj: Optional[object]
    name: Optional[str]
    filename: str
    code: Optional[types.CodeType]


"""
这是确定对象（函数）是否应该内联或跳过的主要入口点。
让我们用一个例子说明这个逻辑：
    @torch.compile
    def f1(x, y):
        ......
        f2(x, y)
        ......

    def f2(x, y):
        ......
        f3(x, y)
        ......

    def f3(x, y):
        ......

主要有三个调用 check/check_verbose 的调用点：
* 编译区域入口（如函数 f1），对应的代码位于 eval_frame.py 中。
* 当追踪递归调用的函数时（如函数 f2 和 f3）。
    * Dynamo 每次遇到新的递归函数调用时决定内联/跳过，并且调用点在 symbolic_convert.py 的 InliningInstructionTranslator.check_inlineable 中。
"""
    # 如果 Dynamo 在评估 f3 的帧时跳过了 f2，那么需要在 catch_errors_wrapper.catch_errors 函数中再次进行内联/跳过检查。
    # 调用站点位于 convert_frame.py 文件中。
# 定义函数 check_verbose，用于详细检查对象并返回跟踪规则的结果
def check_verbose(obj, is_inlined_call=False):
    # 如果 obj 是用户定义的函数或方法变量，获取其函数对象信息
    if isinstance(
        obj, (UserFunctionVariable, UserMethodVariable, NestedUserFunctionVariable)
    ):
        try:
            py_obj = obj.get_function()  # 获取函数对象
        except NotImplementedError:
            py_obj = None
        # 创建 FunctionInfo 对象，包含函数的名称、文件名、代码等信息
        fi = FunctionInfo(py_obj, obj.get_name(), obj.get_filename(), obj.get_code())
    # 如果 obj 是代码对象，创建 FunctionInfo 对象
    elif isinstance(obj, types.CodeType):
        fi = FunctionInfo(None, obj.co_name, obj.co_filename, obj)
    # 如果 obj 是函数或方法对象，创建 FunctionInfo 对象
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        fi = FunctionInfo(
            obj, obj.__name__, getfile(obj), obj.__code__  # type: ignore[union-attr] # FIXME Add MethodType.__code__ to typeshed
        )
    # 否则，创建 FunctionInfo 对象，包含对象和文件名信息
    else:
        fi = FunctionInfo(obj, None, getfile(obj), None)

    # 根据 torch._dynamo.trace_rules 中定义的规则查找跟踪规则
    reasons: Set[str] = set()
    rule = torch._dynamo.trace_rules.lookup_inner(
        fi.py_obj, fi.name, fi.filename, is_inlined_call, reasons
    )
    # 如果规则为 UserFunctionVariable 或 FunctorchHigherOrderVariable，则返回不跳过
    if rule in [UserFunctionVariable, FunctorchHigherOrderVariable]:
        return SkipResult(
            False,
            f"inlined according trace_rules.lookup {reasons.pop()}",
        )
    else:
        # 否则，断言规则为 SkipFunctionVariable，并返回跳过结果
        assert rule == SkipFunctionVariable, rule
        return SkipResult(
            True,
            f"skipped according trace_rules.lookup {reasons.pop()}",
        )


# 定义函数 check，简化检查对象的操作，返回跳过结果
def check(obj, is_inlined_call=False):
    return check_verbose(obj, is_inlined_call).skipped


# 将 THIRDPARTY_SKIPLIST 中的名称添加到全局跳过列表中
for _name in THIRDPARTY_SKIPLIST:
    add(_name)

# 重新编译正则表达式
_recompile_re()


# 判断是否允许对 filename 中的 Torch 模块进行内联
def is_torch_inline_allowed(filename):
    return any(filename.startswith(d) for d in get_mod_inlinelist())


# 使用 functools.lru_cache 缓存结果，返回 torch._dynamo 模块的目录
@functools.lru_cache(None)
def dynamo_dir():
    import torch._dynamo
    return _module_dir(torch._dynamo)


# 判断 filename 是否属于 Torch 模块
def is_torch(filename):
    # 如果 filename 开始于 dynamo_dir() 返回的目录，返回 False
    if filename.startswith(dynamo_dir()):
        return False
    # 否则，判断 filename 是否属于 torch 模块
    return filename.startswith(_module_dir(torch))


"""
给定可调用对象，主要入口点查找跟踪规则（Dynamo 变量）。
"""
def lookup_callable(obj):
    # 如果 obj 不可散列，返回 None
    if not hashable(obj):
        return None
    # 如果 obj 在图中被禁止调用，返回 SkipFunctionVariable
    if is_callable_disallowed(obj):
        return SkipFunctionVariable
    # 如果 obj 在图中被允许调用，返回 TorchInGraphFunctionVariable
    if is_callable_allowed(obj):
        return TorchInGraphFunctionVariable
    # 如果 obj 是内置可调用对象，返回 BuiltinVariable
    if is_builtin_callable(obj):
        return BuiltinVariable
"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given function object.
E.g, the lookup result of `torch.sin` is `TorchInGraphFunctionVariable`.
"""

# 定义函数 lookup，用于查找给定函数对象的追踪规则
def lookup(obj):
    return lookup_inner(obj)

# 定义函数 lookup_inner，用于实际查找函数对象的追踪规则
def lookup_inner(
    obj,
    name=None,
    filename=None,
    is_direct_call=True,
    reasons: Union[None, Set[str]] = None,
):
    # Step 1: lookup obj's tracing rule in `torch_name_rule_map`.
    # The rules defined in `torch_name_rule_map` mainly includes two parts:
    # - Manually defined rules for any functions.
    # - The list of torch in graph functions.
    
    # 尝试检测对象是否可哈希
    try:
        can_hash = hashable(obj)
    except Exception:
        can_hash = False
    # 如果对象不可哈希，则返回 None，并记录 reasons 如果可用
    if not can_hash:
        if reasons is not None:
            reasons.add("obj is not hashable")
        return None
    
    # 如果对象不为空且是 torch 的操作或张量方法，则返回 TorchInGraphFunctionVariable
    if obj is not None:
        if is_aten_op_or_tensor_method(obj):
            return TorchInGraphFunctionVariable
        
        # 从 torch 对象规则映射中获取规则
        rule = get_torch_obj_rule_map().get(obj, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    
    # 如果函数名和文件名均不为空且不是直接调用，则根据名称和文件名获取规则
    elif name is not None and filename is not None and not is_direct_call:
        if name.startswith(TORCH_DYNAMO_RESUME_IN_PREFIX):
            rule = get_torch_obj_rule_map().get(
                filename + "#" + TORCH_DYNAMO_RESUME_IN_PREFIX, None
            )
        else:
            rule = get_torch_obj_rule_map().get(filename + "#" + name, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    
    # Step 2: lookup obj's tracing rule by function name.
    # 如果是直接调用，并且函数名为 patched_init，则返回 SkipFunctionVariable
    if is_direct_call:
        if name == "patched_init":
            if reasons is not None:
                reasons.add("func name is patched_init")
            return SkipFunctionVariable
        # 如果函数名为 __torch_function__，则返回 UserFunctionVariable
        elif name == "__torch_function__":
            if reasons is not None:
                reasons.add("func name is __torch_function__")
            return UserFunctionVariable
    # 如果不是直接调用（即 is_direct_call 为 False）
    if not is_direct_call:
        # 如果函数名为 "__getattr__"
        if name == "__getattr__":
            # 当 is_direct_call = False 时，表示这是顶层帧的追踪（即不是内联化且不是从 InliningInstructionTranslator 调用的）。
            # 在顶层追踪 __getattr__ 是不太可能的，因为我们会为 UserDefinedObjectVariable 内联它。
            # 这种情况仅发生在 UnspecializedNNModuleVariable 上，Dynamo 在跟踪时直接调用 __getattr__，生成 LOAD_ATTR 字节码，
            # 而不通过底层的 __getattr__ 数据结构。当执行此优化的字节码时，会再次触发 Dynamo 对 __getattr__ 的调用。
            # 因此，在这种情况下，我们跳过 Dynamo 的追踪。
            if reasons is not None:
                reasons.add(
                    "Tracing __getattr__ as the top level frame, unsuitable for tracing."
                )
            return SkipFunctionVariable

    # Step 3: 根据对象的文件名查找其追踪规则。
    if filename is None:
        filename = getfile(obj)

    # 检查文件是否应该跳过追踪
    skip_result = check_file(filename, is_direct_call)
    if reasons is not None:
        reasons.add(skip_result.reason)
    
    # 如果应该跳过追踪，则返回 SkipFunctionVariable
    if skip_result.skipped:
        return SkipFunctionVariable
    else:
        # 否则返回 UserFunctionVariable
        return UserFunctionVariable
# 清空LRU缓存，以确保下一次调用时重新计算结果
def clear_lru_cache():
    # 清空torch._dynamo.trace_rules.get_torch_obj_rule_map的缓存
    torch._dynamo.trace_rules.get_torch_obj_rule_map.cache_clear()
    # 清空torch._dynamo.trace_rules.get_tensor_method的缓存
    torch._dynamo.trace_rules.get_tensor_method.cache_clear()
    # 清空torch._dynamo.trace_rules.get_legacy_mod_inlinelist的缓存
    torch._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()
    # 清空torch._dynamo.trace_rules.get_mod_inlinelist的缓存
    torch._dynamo.trace_rules.get_mod_inlinelist.cache_clear()
    # 清空torch._dynamo.trace_rules.dynamo_dir的缓存
    torch._dynamo.trace_rules.dynamo_dir.cache_clear()
```