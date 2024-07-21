# `.\pytorch\torch\onnx\_internal\fx\decomposition_table.py`

```
# mypy: allow-untyped-defs
"""Dispatcher for AtenLib functions from onnx-script."""

# 引入未来版本的注释支持
from __future__ import annotations

# 引入类型提示相关模块
from typing import Callable, Dict, Set, Union

# 引入PyTorch库
import torch
import torch._ops
import torch.fx

# 引入内部ONNX相关模块
from torch.onnx._internal import _beartype

# 引入ONNX注册相关模块
from torch.onnx._internal.fx import registration


# NOTE: OnnxRegistry annotation: beartype is a runtime type checker for python3,
# so it doesn't work with TYPE_CHECKING
# 使用beartype进行类型检查
@_beartype.beartype
# 创建一个函数用于生成支持ONNX的操作重载表
def _create_onnx_supports_op_overload_table(
    registry,
) -> Set[Union[torch._ops.OperatorBase, Callable]]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry (OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    # 初始化一个空集合用于存放OperatorBase和Callable对象
    table: Set[Union[torch._ops.OperatorBase, Callable]] = set()

    # Some ops in `torch.ops.aten` are not discoverable through `dir(torch.ops.aten)`,
    # but retrievable via explicit lookup.
    # https://github.com/pytorch/pytorch/issues/99681
    # This is a workaround to make sure we register ONNX symbolic functions for these.
    # 获取所有以"aten::"开头的注册操作，并提取操作名称部分作为查找表项
    onnx_supported_aten_lookup_table = [
        k.split("::")[1].split(".")[0]
        for k in registry._all_registered_ops()
        if k.startswith("aten::")
    ]
    # 遍历 torch.ops.aten 和 torch.ops.prims 中的命名空间
    for op_namespace in (torch.ops.aten, torch.ops.prims):
        # 获取命名空间中的所有属性名
        attr_names = dir(op_namespace)
        
        # 如果当前命名空间是 torch.ops.aten，则添加额外的属性名到列表中
        if op_namespace is torch.ops.aten:
            attr_names += onnx_supported_aten_lookup_table
        
        # 遍历当前命名空间下的每个属性名
        for attr_name in attr_names:
            # 如果当前命名空间中不包含该属性名，则跳过
            if not hasattr(op_namespace, attr_name):
                # torchlib 拥有一些不是 aten 操作的属性。
                continue
            
            # 获取当前属性名对应的操作重载包
            op_overload_packet = getattr(op_namespace, attr_name)
            
            # 如果操作重载包不是 torch._ops.OpOverloadPacket 类型，则跳过
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue
            
            # 遍历操作重载包中的每个重载名称
            for overload_name in op_overload_packet.overloads():
                # 获取当前重载名称对应的操作重载
                op_overload = getattr(op_overload_packet, overload_name)
                
                # 根据操作重载的完全限定名称创建内部操作名称对象
                internal_op_name = registration.OpName.from_qualified_name(
                    qualified_name=op_overload.name()
                )
                
                # 如果注册表中注册了该操作的特定重载或默认重载，则将其添加到表中
                if registry.is_registered_op(
                    namespace=internal_op_name.namespace,
                    op_name=internal_op_name.op_name,
                    overload=internal_op_name.overload,
                ) or registry.is_registered_op(
                    namespace=internal_op_name.namespace,
                    op_name=internal_op_name.op_name,
                    overload=None,
                ):
                    # 这一行将 torch.ops.aten.add.Tensor、torch.ops.aten.add.Scalar、torch.ops.aten.add.out 等映射到 "aten::add"。
                    # 这意味着 "aten::add" 的导出器将用于所有 "aten::add" 的重载。
                    # 这适用于 torch.ops.aten 下的所有操作。
                    table.add(op_overload)
    
    # 返回填充后的表格
    return table
# 使用装饰器 @_beartype.beartype 对下面的函数进行类型检查
@_beartype.beartype
# 创建一个函数，用于生成一个字典，其中包含没有 ONNX 符号函数的操作符重载及其分解函数
def create_onnx_friendly_decomposition_table(
    registry,  # 参数 registry 是一个 torch.onnx.OnnxRegistry 实例，用于 PyTorch 的 ONNX 注册表
) -> Dict[torch._ops.OperatorBase, Callable]:  # 函数返回一个字典，将操作符重载映射到它们对应的分解函数
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        registry (torch.onnx.OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    # 创建空字典 decomposition_table，用于存储操作符重载到分解函数的映射关系
    decomposition_table: Dict[torch._ops.OperatorBase, Callable] = {}
    # 调用 _create_onnx_supports_op_overload_table(registry) 函数，返回一个字典 _ONNX_SUPPORT_OP_OVERLOADS，
    # 用于存储支持的操作符重载到导出器查找键的映射关系
    _ONNX_SUPPORT_OP_OVERLOADS = _create_onnx_supports_op_overload_table(registry)

    # 循环遍历 torch._decomp.decomposition_table.items() 中的每个元素，元素是操作符重载及其对应的分解函数
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():  # type: ignore[attr-defined]
        # 如果分解函数来自 torch._refs 模块，或者 op_overload 在 _ONNX_SUPPORT_OP_OVERLOADS 字典中已存在，
        # 则跳过当前循环，不将其加入 decomposition_table 中
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload in _ONNX_SUPPORT_OP_OVERLOADS
        ):
            continue
        # 将当前操作符重载 op_overload 及其对应的分解函数 decomp_fn 加入 decomposition_table 字典中
        decomposition_table[op_overload] = decomp_fn

    # 再次循环遍历 torch._decomp.core_aten_decompositions() 返回的字典中的每个元素，
    # 元素是核心 ATen 中的操作符重载及其对应的分解函数
    for op_overload, decomp_fn in torch._decomp.core_aten_decompositions().items():
        # 如果 op_overload 在 _ONNX_SUPPORT_OP_OVERLOADS 字典中已存在，则跳过当前循环，不将其加入 decomposition_table 中
        if op_overload in _ONNX_SUPPORT_OP_OVERLOADS:
            continue
        # 将当前操作符重载 op_overload 及其对应的分解函数 decomp_fn 加入 decomposition_table 字典中
        decomposition_table[op_overload] = decomp_fn

    # 返回最终生成的 decomposition_table 字典
    return decomposition_table
```