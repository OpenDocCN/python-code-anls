# `.\pytorch\torch\onnx\_internal\fx\registration.py`

```py
# 模块用于处理 ATen 到 ONNX 函数的注册

from __future__ import annotations  # 允许在类型检查时使用注解

import dataclasses  # 导入用于创建不可变数据类的模块
import types  # 导入用于处理类型的模块
from typing import Optional, TYPE_CHECKING, Union  # 导入类型提示工具

import torch._ops  # 导入 PyTorch 内部操作的模块
from torch.onnx._internal import _beartype  # 导入内部的类型检查工具

# 当在类型检查环境中时，只能在此模块中导入 onnx，以确保即使没有安装 'onnx'，'import torch.onnx' 仍然可用。
if TYPE_CHECKING:
    import onnxscript  # type: ignore[import]  # 导入 onnxscript 模块，忽略类型检查


@dataclasses.dataclass(frozen=True, eq=True)
class ONNXFunction:
    """包装 onnx-script 函数的类。

    onnx_function: torchlib 中的 onnx-script 函数。
    op_full_name: 函数的完全限定名称，格式为 '<namespace>::<op_name>.<overload>'。
    is_custom: 是否为自定义函数。
    is_complex: 是否为处理复杂值输入的函数。
    """
    onnx_function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]
    op_full_name: str
    is_custom: bool = False
    is_complex: bool = False


@dataclasses.dataclass(frozen=True, eq=True)
class OpName:
    """内部 ONNX 转换器中操作符名称的类。

    namespace: 操作符所属的命名空间。
    op_name: 操作符的名称。
    overload: 操作符的重载版本。

    """
    namespace: str
    op_name: str
    overload: str

    @classmethod
    @_beartype.beartype
    def from_name_parts(
        cls, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> OpName:
        """从名称部分创建 OpName 实例。

        如果未提供 overload 或者为空，则表示默认的重载版本。
        """
        if overload is None or overload == "":
            overload = "default"
        return cls(namespace, op_name, overload)

    @classmethod
    @_beartype.beartype
    def from_qualified_name(cls, qualified_name: str) -> OpName:
        """从完全限定名称创建 OpName 实例。

        名称格式为 <namespace>::<op_name>[.<overload>]。
        """
        namespace, opname_overload = qualified_name.split("::")
        op_name, *overload = opname_overload.split(".", 1)
        overload = overload[0] if overload else "default"
        return cls(namespace, op_name, overload)

    @classmethod
    @_beartype.beartype
    def from_op_overload(cls, op_overload: torch._ops.OpOverload) -> OpName:
        """从 torch._ops.OpOverload 实例创建 OpName 实例。"""
        return cls.from_qualified_name(op_overload.name())

    @classmethod
    @_beartype.beartype
    def from_builtin_function(
        cls, builtin_function: types.BuiltinFunctionType
    ) -> OpName:
        """从内置函数（例如 operator.add、math.ceil 等）获取操作名称。

        FX 图使用内置函数来计算 sympy 表达式。此函数用于从内置函数获取操作名称。

        Args:
            builtin_function (types.BuiltinFunctionType): operator.add、math.ceil 等。

        Returns:
            OpName: 操作名称
        """
        op = builtin_function.__name__  # 获取内置函数的名称，如 add、sub 等
        module = builtin_function.__module__  # 获取内置函数所在的模块，如 _operators 或 math
        return cls.from_qualified_name(module + "::" + op)

    @_beartype.beartype
    def qualified_name(self) -> str:
        """返回完全限定的名称，包括命名空间、操作名称和重载版本。

        Returns:
            str: 完全限定的名称，格式为 "{self.namespace}::{self.op_name}.{self.overload}"
        """
        return f"{self.namespace}::{self.op_name}.{self.overload}"
```