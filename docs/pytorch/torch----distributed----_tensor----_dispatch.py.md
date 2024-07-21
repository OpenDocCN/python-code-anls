# `.\pytorch\torch\distributed\_tensor\_dispatch.py`

```
# 版权声明以及导入必要的模块和类型声明
import contextlib  # 提供用于创建上下文管理器的实用工具
import functools  # 提供用于函数式编程的实用工具
import operator  # 提供用于操作符的函数工具
import warnings  # 提供警告管理工具
from typing import cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING  # 引入类型提示

import torch  # PyTorch主模块
import torch.distributed as dist  # 分布式模块
import torch.distributed._tensor.api as dtensor  # 分布式张量API
import torch.distributed._tensor.random as random  # 分布式张量随机数生成
from torch.distributed._tensor._op_schema import (
    _is_inplace_op,  # 判断是否为原地操作的辅助函数
    _is_out_variant_op,  # 判断是否为输出变种操作的辅助函数
    OpInfo,  # 操作信息类
    OpSchema,  # 操作模式类
    OutputSpecType,  # 输出规范类型
)
from torch.distributed._tensor._redistribute import redistribute_local_tensor  # 重新分配本地张量
from torch.distributed._tensor._sharding_prop import ShardingPropagator  # 分片传播器
from torch.distributed._tensor._tp_conv import (
    convolution_backward_handler,  # 卷积反向处理器
    convolution_handler,  # 卷积处理器
)
from torch.distributed._tensor._utils import try_find_mesh_from_args  # 尝试从参数中查找网格
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate, TensorMeta  # 张量规格、复制、张量元信息
from torch.distributed._tensor.random import is_rng_supported_mesh  # 判断随机数生成是否支持网格

# 如果是类型检查模式，则导入DeviceMesh类
if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.utils import _cxx_pytree as pytree  # 尝试导入C++ PyTree模块
except ImportError:
    from torch.utils import _pytree as pytree  # 如果导入失败，则导入Python PyTree模块（忽略重新定义）

aten = torch.ops.aten  # 获取torch.ops.aten操作模块


def decompose_handler(
    op_call: torch._ops.OpOverload,  # 操作重载对象
    args: Tuple[object, ...],  # 参数元组
    kwargs: Dict[str, object],  # 关键字参数字典
) -> object:
    """
    将操作分解为核心的 ATen 操作，此处理器主要用于推断模式下的使用，其中操作不是核心的 ATen 操作。
    """
    r = op_call.decompose(*args, **kwargs)  # 调用操作对象的分解方法
    if r is not NotImplemented:  # 如果成功分解操作
        return r  # 返回分解结果
    else:
        raise RuntimeError("Decomposition failed")  # 否则抛出运行时错误，分解失败


def is_same_size_handler(
    op_call: torch._ops.OpOverload,  # 操作重载对象
    args: Tuple[object, ...],  # 参数元组
    kwargs: Dict[str, object],  # 关键字参数字典
) -> bool:
    lhs = cast(torch.Tensor, args[0])  # 将第一个参数转换为 torch.Tensor 类型
    rhs = cast(torch.Tensor, args[1])  # 将第二个参数转换为 torch.Tensor 类型
    return lhs.shape == rhs.shape  # 返回左右两个张量的形状是否相同的布尔值


class OpDispatcher:
    """
    操作调度类实例，用于处理参数/关键字参数的预处理（解包），分片传播，重新分配本地参数，
    本地计算，并进行后处理（重新封装）。同时，如果需要，还处理任何特定于操作的逻辑。
    """
    def __init__(self) -> None:
        # 初始化分片传播器
        self.sharding_propagator = ShardingPropagator()
        # 初始化随机操作集合
        self._random_ops = {
            aten.native_dropout.default,
            aten.normal_.default,
            aten.rand_like.default,
            aten.randn_like.default,
            aten.randint_like.default,
            aten.randint_like.low_dtype,
            aten.randint_like.low_dtype_out,
            aten.uniform_.default,
            aten.bernoulli.default,
            aten.bernoulli_.float,
        }
        # 初始化自定义操作处理器字典
        self._custom_op_handlers = {
            aten.linear.default: decompose_handler,
            aten.is_same_size.default: is_same_size_handler,
            aten.convolution.default: convolution_handler,
            aten.convolution_backward.default: convolution_backward_handler,
        }

        # 内部标志，用于控制是否将 torch.Tensor(non-DTensor) 隐式复制，或者向用户抛出错误
        # 注意：默认情况下将此标志设为 False 非常不安全，所以我们故意保持其默认状态为 False
        self._allow_implicit_replication = False

    def dispatch(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ):
        # 这里是一个方法定义，没有具体的实现内容，需要在后续完善

    @staticmethod
    def redistribute_local_args(
        op_info: OpInfo,
        suggested_input_schema: OpSchema,
    ) -> None:
        # 注意：很少有情况下我们需要重新分配 kwargs，因此我们有意跳过它

        # TODO: 可能需要保持操作模式模式的平铺状态，以避免进行这种树形展开
        # 在执行此操作之前，需要修复所有操作
        if op_info.args_tree_spec is not None:
            flatten_args_schema_to_reshard = tuple(
                pytree.tree_leaves(suggested_input_schema.args_schema)
            )
        else:
            flatten_args_schema_to_reshard = suggested_input_schema.args_schema

        new_local_args: List[object] = []
        for i, arg_spec in enumerate(op_info.flat_args_schema):
            reshard_arg_spec = flatten_args_schema_to_reshard[i]
            if isinstance(arg_spec, DTensorSpec):
                local_tensor = cast(torch.Tensor, op_info.local_args[i])
                if arg_spec != reshard_arg_spec:
                    resharded_local_tensor = redistribute_local_tensor(
                        local_tensor, arg_spec, reshard_arg_spec
                    )
                    new_local_args.append(resharded_local_tensor)
                else:
                    new_local_args.append(local_tensor)
            else:
                new_local_args.append(reshard_arg_spec)

        op_info.local_args = tuple(new_local_args)

    def unwrap_to_op_info(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ):
        # 这里是一个方法定义，没有具体的实现内容，需要在后续完善
    # 定义一个函数 wrap，用于根据输入的输出 res 和输出规范 spec 进行包装处理
    def wrap(res: object, spec: OutputSpecType) -> object:
        # 如果 res 是 torch.Tensor 类型
        if isinstance(res, torch.Tensor):
            # 如果 spec 不为 None，则要求 spec 是 DTensorSpec 类型
            if spec is not None:
                assert isinstance(
                    spec, DTensorSpec
                ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
                # 返回一个经过 DTensor 包装过的 DTensor 对象，保留梯度信息
                return dtensor.DTensor(res, spec, requires_grad=res.requires_grad)
            else:
                # 如果没有 spec，由于特定操作，输出应为标量张量
                assert res.ndim == 0, "output tensor should be scalar!"
                # 直接返回 res
                return res
        # 如果 res 是 list 或者 tuple 类型
        elif isinstance(res, (list, tuple)):
            # 要求 spec 不为 None，并且是 list 或者 tuple 类型
            assert spec is not None and isinstance(
                spec, (list, tuple)
            ), f"output spec does not match with output! Expected list/tuple, got {spec}."
            # 生成一个空列表 res_list，用于存放处理后的结果
            res_list = []
            # 对于 res 和 spec 中的每一个元素 e 和 s，调用 OpDispatcher.wrap 进行包装处理
            for e, s in zip(res, spec):
                res_list.append(OpDispatcher.wrap(e, s))

            # 如果输入 res 是 tuple，则返回一个元组，否则返回一个列表
            return tuple(res_list) if isinstance(res, tuple) else res_list
        else:
            # 如果 res 中只包含非张量值（如 int/float/None），直接返回 res，无需重新包装成 DTensor
            return res
```