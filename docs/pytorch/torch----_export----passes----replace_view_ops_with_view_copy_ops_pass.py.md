# `.\pytorch\torch\_export\passes\replace_view_ops_with_view_copy_ops_pass.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型引用
from typing import Dict, Optional, Set

# 导入 torch 库
import torch
# 导入相关的操作重载类
from torch._ops import OpOverload, OpOverloadPacket, HigherOrderOperator
# 导入内部错误异常类
from torch._export.error import InternalError
# 导入底层导出基类（不建议使用）
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse

# 指定本模块对外公开的类名
__all__ = ["ReplaceViewOpsWithViewCopyOpsPass"]

# 定义非功能性操作到功能性操作的映射字典
_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: Dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
}

# 定义黑名单操作集合，用于存放不可用于处理的操作包
_BLACK_LISTED_OPS: Set[OpOverloadPacket] = {
    torch.ops.aten.sym_size,
    torch.ops.aten.sym_stride,
    torch.ops.aten.sym_numel,
}

# 判断给定的函数架构是否为视图操作
def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    # 如果参数列表为空，则返回 False
    if len(schema.arguments) == 0:
        return False
    # 获取第一个参数的别名信息
    alias_info = schema.arguments[0].alias_info
    # 返回是否为视图操作的判断结果
    return (alias_info is not None) and (not alias_info.is_write)

# 获取视图操作对应的视图拷贝操作
def get_view_copy_of_view_op(schema: torch._C.FunctionSchema) -> Optional[OpOverload]:
    # 如果是视图操作并且名称以 "aten::" 开头
    if is_view_op(schema) and schema.name.startswith("aten::"):
        # 提取视图操作的名称
        view_op_name = schema.name.split("::")[1]
        # 获取视图操作的重载名称
        view_op_overload = (
            schema.overload_name
            if schema.overload_name != ""
            else "default"
        )
        # 构建视图拷贝操作的名称
        view_copy_op_name = view_op_name + "_copy"
        # 如果在 torch.ops.aten 模块中找不到对应的视图拷贝操作，则抛出内部错误异常
        if not hasattr(torch.ops.aten, view_copy_op_name):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        # 获取视图拷贝操作的重载包
        view_copy_op_overload_packet = getattr(torch.ops.aten, view_copy_op_name)

        # 如果视图拷贝操作的重载包中找不到对应的重载版本，则抛出内部错误异常
        if not hasattr(view_copy_op_overload_packet, view_op_overload):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        # 返回对应的视图拷贝操作对象
        return getattr(view_copy_op_overload_packet, view_op_overload)

    # 如果不是视图操作，则返回 None
    return None

# 定义一个类，用于替换视图操作为视图拷贝操作的导出通行证（不建议使用）
class ReplaceViewOpsWithViewCopyOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    """
    # 重写父类方法，用于调用操作符
    def call_operator(self, op, args, kwargs, meta):
        # 如果操作符在非功能性操作到功能性操作的映射字典中
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            # 调用父类方法，使用映射后的功能性操作
            return super().call_operator(
                (_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op]), args, kwargs, meta
            )

        # 如果操作符在黑名单操作集合中或者是高阶操作符，则调用父类方法
        if op in _BLACK_LISTED_OPS or isinstance(op, HigherOrderOperator):
            return super().call_operator(op, args, kwargs, meta)

        # 获取视图操作对应的视图拷贝操作
        if view_copy_op := get_view_copy_of_view_op(op._schema):
            # 调用父类方法，使用视图拷贝操作
            return super().call_operator(view_copy_op, args, kwargs, meta)

        # 否则，调用父类方法，使用原始操作符
        return super().call_operator(op, args, kwargs, meta)
```