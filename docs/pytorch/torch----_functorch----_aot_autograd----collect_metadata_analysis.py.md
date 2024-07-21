# `.\pytorch\torch\_functorch\_aot_autograd\collect_metadata_analysis.py`

```
# mypy: allow-untyped-defs
"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the analysis here constructs view and mutation metadata from running
a functionalized version of the graph under compilation.
"""

import collections  # 引入collections模块，用于处理集合数据类型
import logging  # 引入logging模块，用于记录日志信息
from functools import wraps  # 从functools模块导入wraps装饰器，用于保留被装饰函数的元信息
from typing import Callable, DefaultDict, Dict, List  # 引入类型提示相关的类和装饰器

import torch  # 引入torch模块，用于深度学习计算
import torch.utils._pytree as pytree  # 导入torch的_pytree模块，具体作用需要进一步分析
from torch import Tensor  # 从torch模块导入Tensor类，用于操作张量
from torch._guards import detect_fake_mode  # 从torch._guards模块导入detect_fake_mode函数
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode  # 导入功能性张量相关的子类和模式
from torch._subclasses.meta_utils import safe_is_leaf  # 从torch._subclasses.meta_utils导入safe_is_leaf函数
from torch.fx.experimental.symbolic_shapes import is_concrete_int  # 从torch.fx.experimental.symbolic_shapes导入is_concrete_int函数
from torch.multiprocessing.reductions import StorageWeakRef  # 从torch.multiprocessing.reductions导入StorageWeakRef类
from torch.utils._python_dispatch import (  # 从torch.utils._python_dispatch导入多个函数
    is_traceable_wrapper_subclass,
    transform_subclass,
)
from .functional_utils import (  # 从当前包的functional_utils模块导入多个函数
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    from_fun,
    has_data_mutation,
    has_metadata_mutation,
    has_same_metadata,
    to_fun,
    was_inductor_storage_resized,
)
from .schemas import (  # 从当前包的schemas模块导入多个类和常量
    FunctionalTensorMetadataEq,
    InputAliasInfo,
    MutationType,
    OutputAliasInfo,
    OutputType,
    ViewAndMutationMeta,
)
from .subclass_utils import create_subclass_meta  # 从当前包的subclass_utils模块导入create_subclass_meta函数

from .utils import _get_autocast_states, KNOWN_TYPES, strict_zip  # 从当前包的utils模块导入多个函数和常量

zip = strict_zip  # 将strict_zip函数赋值给zip变量

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


# Note [Tangents must be contiguous]
# We force tangents to be contiguous today.
# The idea is that we are technically making a guess about the strides of our tangents,
# while we trace out the joint.
# Today, we force this guess to be correct by additioanlly calling contiguous()
# on all tangents at runtime.
# In the future, you could imagine lifting this restriction, since these contiguous()
# calls can have noticeable perf overhead depending on the model.
def coerce_tangent(x):
    if not isinstance(x, Tensor):
        return x
    out = x.detach().contiguous()  # 将张量x进行分离并保证其是连续的
    # Note [Tangents must be contiguous, Part 2]
    # In the same way that "what strides do we assigns to our tangents" is a question
    # that we can not answer (and therefore have to guess) as we trace the backward ahead-of-time,
    # The same applies to any tensor subclass metadata, when we have tangents that are subclasses.
    # To handle this situation, we have two new methods that a tensor subclass can implement:
    # (1) __coerce_tangent_metadata__(self)
    #     Given a subclass with "non-standard" metadata, turn it into a new subclass with "normal" metadata.
    #     The main example here is a DTensor with the "_Partial" placement.
    #     If we have a forward output with a _Partial placement, and corresponding tangent
    #     with a Replicate/Shard placement, we have no way to convert the tangent "back" to a _Partial placement.
    #     This method lets us avoid the problem entirely by allowing subclasses to ensure that we can never
    #     have a tangent with "problematic" metadata, that we cannot convert to.
    # (1) __coerce_same_metadata_as_tangent__(self, metadata)
    #     Given a subclass, and a target differing metadata,
    #     convert self to have the same metadata as the target.
    #     With DTensor being the main example, we can use this to convert a DTensor with a Replicate()
    #     placement into one with a Shard() placement, in the case that we "guessed wrong",
    #     and traced tangents with a Shard() placement at compile time.
    #
    # Check if `out` is a subclass of a traceable wrapper and has the method __coerce_tangent_metadata__
    if is_traceable_wrapper_subclass(out) and hasattr(
        out, "__coerce_tangent_metadata__"
    ):
        # Call __coerce_tangent_metadata__ to potentially adjust metadata of `out`
        out = out.__coerce_tangent_metadata__()
    
    # It's possible to have a subclass that advertises as contiguous,
    # but has noncontiguous inner tensors.
    # Force these to be contiguous as well
    if is_traceable_wrapper_subclass(out):
        # Iterate over attributes returned by __tensor_flatten__()[0] and check tensor contiguity
        for attr in out.__tensor_flatten__()[0]:  # type: ignore[attr-defined]
            elem = getattr(out, attr)
            # If element `elem` is not contiguous, make it contiguous and set it back to `out`
            if not elem.is_contiguous():
                elem_contig = elem.contiguous()
                setattr(out, attr, elem_contig)
    
    # Return the modified or unmodified `out` object
    return out
# 这是为 AOTAutograd 使用案例专门设计的功能化版本。
# 不像 functorch 的变体，这里没有使用 functorch 的层级系统，
# 而是直接使用 PyTorch 的传统调度程序来触发功能化关键操作。
# 这意味着 FunctionalTensorWrapper 可以直接存储自动求导数据。
#
# 在典型的 AOTAutograd 使用中，调度键的顺序如下：
#
#   Autograd - Functionalization ~~~~> Proxy Mode - Fake Tensor
#       外部张量                         内部张量
#
# 返回：
# - ViewAndMutationMeta，提供关于输入和输出的元数据，
#   以及前向传播的输出列表，但**仅**需要作为切线传递给反向传播的输出。
#   特别是，前向传播的别名输出会被重新生成，并且不参与编译后的反向传播函数。
def run_functionalized_fw_and_collect_metadata(
    f,
    *,
    keep_input_mutations: bool,
    # TODO: refactor to kill this flag
    is_train: bool = False,
    pre_dispatch: bool = False,
) -> Callable[..., ViewAndMutationMeta]:
    # 用于存储张量及其对应功能化版本的字典
    memo: Dict[Tensor, Tensor] = {}

    def _to_fun(t):
        # 如果是张量，则尝试从 memo 中获取功能化版本，或者创建并存储功能化版本
        if isinstance(t, Tensor):
            if t in memo:
                return memo[t]
            r = to_fun(t)
            memo[t] = r
            return r
        else:
            return t

    # 将函数 f 的装饰器应用于内部函数 inner，并返回 inner 函数的包装器
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner
```