# `.\pytorch\torch\onnx\_internal\fx\decomposition_skip.py`

```py
# mypy: allow-untyped-defs
"""A context manager that disables the decomposition of certain ops during dynamo tracing.

The approach is to temporarily hijack the operator callable with PT2 custom operator.
The custom operator will not be decomposed and will show up as a single node to be exported to ONNX.

For the time being the decomposition of these ops is otherwise unavoidable.

https://github.com/pytorch/pytorch/issues/116684
https://github.com/pytorch/pytorch/issues/115883

This solution will no longer be required once the issue is resolved.
"""

# 引入未来的注释类型，支持函数签名的类型声明
from __future__ import annotations

# 引入抽象基类模块和上下文管理器模块
import abc
import contextlib

# 引入类型提示相关模块
from typing import Callable, Sequence, Type

# 引入自定义操作的相关模块，若找不到则忽略类型检查
from onnxscript.function_libs.torch_lib.ops import (
    core as torchlib_core,
    nn as torchlib_nn,
)

# 引入 PyTorch 模块
import torch
# 引入 Torch 内部的分解模块
from torch._decomp import decompositions

# 自定义操作的命名空间
_NEW_OP_NAMESPACE: str = "onnx_export"
"""The namespace for the custom operator."""


# 抽象基类，用于跳过分解的操作
class DecompSkip(abc.ABC):
    op_callable: Callable
    """The original operator callable to skip decomposition."""
    onnxscript_function: Callable
    """The ONNXScript function to be registered for exporting the custom operator."""

    new_op_name: str
    """The name for the custom operator."""
    new_op_schema: str
    """The schema for the custom operator. This should match with the signature of the original operator."""

    @classmethod
    @abc.abstractmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        """Registers the custom operator and overrides the original operator.

        It should do the following steps in order:

        1. Register the custom operator.
        2. Override the original operator with the replacement callable.
        3. Register the ONNXScript function for exporting the custom operator.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def unregister(cls):
        """Restores the original operator callable."""
        ...

    @classmethod
    @abc.abstractmethod
    def abstract(cls, *args, **kwargs):
        """An abstract impl (meta kernel) for the operator."""
        ...

    @classmethod
    def register_custom_op(cls):
        """Registers the custom operator."""
        # 定义新操作的限定名称，并注册
        new_op_qualname = f"{_NEW_OP_NAMESPACE}::{cls.new_op_name}"
        torch.library.define(new_op_qualname, cls.new_op_schema)
        # 使用默认实现注册新操作
        torch.library.impl(new_op_qualname, "default", cls.replacement)
        # 注册伪造的新操作
        torch.library.register_fake(new_op_qualname, cls.abstract)

    @classmethod
    def replacement(cls, *args, **kwargs):
        """A replacement callable for the operator to be hijacked.

        This has the same signature and eager behavior as the original operator.
        """
        # 替换操作的调用方式，维持和原操作相同的签名和行为
        return cls.op_callable(*args, **kwargs)


# 继承抽象基类，用于跳过双线性上采样操作的分解
class UpsampleBilinear2DDecompSkip(DecompSkip):
    op_callable = torch._C._nn.upsample_bilinear2d  # type: ignore[attr-defined]
    onnxscript_function = torchlib_nn.aten_upsample_bilinear2d_vec  # type: ignore[attr-defined]
    # 设置新操作的名称为 "upsample_bilinear2d"
    new_op_name = "upsample_bilinear2d"
    # 定义新操作的签名字符串
    new_op_schema = "(Tensor self, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)"

    @classmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        # 如果 torch.ops 中不存在指定的命名空间或者 onnx_export 中不存在新操作的名称
        if not hasattr(torch.ops, _NEW_OP_NAMESPACE) or not hasattr(
            torch.ops.onnx_export, cls.new_op_name
        ):
            # 注册自定义操作
            cls.register_custom_op()
        # 将 torch._C._nn.upsample_bilinear2d 设置为 torch.ops.onnx_export.upsample_bilinear2d
        torch._C._nn.upsample_bilinear2d = torch.ops.onnx_export.upsample_bilinear2d  # type: ignore[attr-defined]
        # 如果 export_options 中的 onnx_registry 为 None
        if export_options.onnx_registry is None:
            # 创建一个新的 OnnxRegistry 对象
            export_options.onnx_registry = torch.onnx.OnnxRegistry()
        # 获取 export_options 中的 onnx_registry
        registry = export_options.onnx_registry
        # 在注册表中注册操作函数
        registry.register_op(
            function=cls.onnxscript_function,
            namespace=_NEW_OP_NAMESPACE,
            op_name=cls.new_op_name,
        )

    @classmethod
    def unregister(cls):
        # 将 torch._C._nn.upsample_bilinear2d 设置为 cls.op_callable
        torch._C._nn.upsample_bilinear2d = cls.op_callable  # type: ignore[attr-defined]

    @classmethod
    def abstract(cls, input, output_size, align_corners, scale_factors):
        # 计算输出大小
        osize = decompositions.upsample_compute_output_size(
            input.size(), output_size, scale_factors
        )
        # 创建一个空的 Tensor，具有指定的形状、数据类型和设备
        return torch.empty(
            (input.size(0), input.size(1), *osize),
            dtype=input.dtype,
            device=input.device,
        )
class UpsampleTrilinear3DDecompSkip(DecompSkip):
    # 设置静态变量，指向 Torch 中的 trilinear 3D 上采样函数
    op_callable = torch._C._nn.upsample_trilinear3d  # type: ignore[attr-defined]
    # 设置 ONNX 脚本函数，用于导出 ONNX 格式的 trilinear 3D 上采样
    onnxscript_function = torchlib_nn.aten_upsample_trilinear3d_vec  # type: ignore[attr-defined]
    # 新操作的名称
    new_op_name = "upsample_trilinear3d"
    # 新操作的 ONNX 模式定义
    new_op_schema = "(Tensor self, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)"

    @classmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        # 如果新操作的命名空间或 ONNX 导出中没有相应的操作，注册自定义操作
        if not hasattr(torch.ops, _NEW_OP_NAMESPACE) or not hasattr(
            torch.ops.onnx_export, cls.new_op_name
        ):
            cls.register_custom_op()
        # 将 Torch 中的 trilinear 3D 上采样函数替换为对应的 ONNX 导出函数
        torch._C._nn.upsample_trilinear3d = torch.ops.onnx_export.upsample_trilinear3d  # type: ignore[attr-defined]
        # 如果导出选项中没有 ONNX 注册表，创建一个新的 ONNX 注册表
        if export_options.onnx_registry is None:
            export_options.onnx_registry = torch.onnx.OnnxRegistry()
        registry = export_options.onnx_registry
        # 在 ONNX 注册表中注册新操作
        registry.register_op(
            function=cls.onnxscript_function,
            namespace=_NEW_OP_NAMESPACE,
            op_name=cls.new_op_name,
        )

    @classmethod
    def unregister(cls):
        # 恢复 Torch 中的 trilinear 3D 上采样函数为初始的可调用对象
        torch._C._nn.upsample_trilinear3d = cls.op_callable  # type: ignore[attr-defined]

    @classmethod
    def abstract(cls, input, output_size, align_corners, scale_factors):
        # 计算上采样后的输出尺寸
        osize = decompositions.upsample_compute_output_size(
            input.size(), output_size, scale_factors
        )
        # 创建一个空的张量作为抽象描述
        return torch.empty(
            (input.size(0), input.size(1), input.size(2), *osize),
            dtype=input.dtype,
            device=input.device,
        )


class InstanceNormDecompSkip(DecompSkip):
    # 设置静态变量，指向 Torch 中的实例归一化函数
    op_callable = torch.instance_norm  # type: ignore[attr-defined]
    # 设置 ONNX 脚本函数，用于导出 ONNX 格式的实例归一化
    onnxscript_function = torchlib_core.aten_instance_norm  # type: ignore[attr-defined]
    # 新操作的名称
    new_op_name = "instance_norm"
    # 新操作的 ONNX 模式定义
    new_op_schema = (
        "(Tensor input, Tensor? weight, Tensor? bias, "
        "Tensor? running_mean, Tensor? running_var, "
        "bool use_input_stats, float momentum, float eps, "
        "bool cudnn_enabled) -> Tensor"
    )

    @classmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        # 如果新操作的命名空间或 ONNX 导出中没有相应的操作，注册自定义操作
        if not hasattr(torch.ops, _NEW_OP_NAMESPACE) or not hasattr(
            torch.ops.onnx_export, cls.new_op_name
        ):
            cls.register_custom_op()

        # 将 Torch 中的实例归一化函数替换为对应的 ONNX 导出函数
        torch.instance_norm = torch.ops.onnx_export.instance_norm  # type: ignore[attr-defined]
        # 如果导出选项中没有 ONNX 注册表，创建一个新的 ONNX 注册表
        if export_options.onnx_registry is None:
            export_options.onnx_registry = torch.onnx.OnnxRegistry()
        registry = export_options.onnx_registry
        # 在 ONNX 注册表中注册新操作
        registry.register_op(
            function=cls.onnxscript_function,
            namespace=_NEW_OP_NAMESPACE,
            op_name=cls.new_op_name,
        )

    @classmethod
    def unregister(cls):
        # 恢复 Torch 中的实例归一化函数为初始的可调用对象
        torch.instance_norm = cls.op_callable  # type: ignore[attr-defined]

    @classmethod
    # 定义一个静态方法 abstract，用于生成一个空的 torch 张量
    def abstract(
        cls,  # 参数 cls：类本身，通常用于类方法的第一个参数，表示类的类型
        input,  # 参数 input：输入张量，用于指定返回张量的形状
        weight,  # 参数 weight：权重张量，可能用于后续操作中
        bias,  # 参数 bias：偏置张量，可能用于后续操作中
        running_mean,  # 参数 running_mean：运行时均值张量，通常在归一化操作中使用
        running_var,  # 参数 running_var：运行时方差张量，通常在归一化操作中使用
        use_input_stats: bool,  # 参数 use_input_stats：布尔值，指定是否使用输入统计数据
        momentum: float,  # 参数 momentum：动量值，通常在归一化操作中用于平滑均值和方差
        eps: float,  # 参数 eps：小数值，用于避免除零错误，通常在归一化操作中使用
        cudnn_enabled: bool,  # 参数 cudnn_enabled：布尔值，指定是否启用 cuDNN 加速
    ):
        # 返回一个空的 torch 张量，形状与输入张量相同，数据类型和设备与输入张量一致
        return torch.empty(
            input.size(),
            dtype=input.dtype,
            device=input.device,
        )
# 默认的跳过列表，包含需要跳过分解的操作类型
_DEFAULT_SKIP_LIST = [
    UpsampleBilinear2DDecompSkip,   # 双线性2D上采样操作的跳过类
    InstanceNormDecompSkip,         # 实例归一化操作的跳过类
    UpsampleTrilinear3DDecompSkip,  # 三线性3D上采样操作的跳过类
]


@contextlib.contextmanager
def enable_decomposition_skips(
    export_options: torch.onnx.ExportOptions,
    skips: Sequence[Type[DecompSkip]] = _DEFAULT_SKIP_LIST,
):
    """一个上下文管理器，用于启用分解跳过。

    原始的操作调用将被替换为自定义操作符进行分解。
    ONNXScript 函数用于将自定义操作符导出并添加到 export_options 中的 ONNX 注册表中。
    """
    try:
        # 注册每个跳过类的导出选项
        for skip in skips:
            skip.register(export_options)
        # 进入上下文管理器的主体部分
        yield
    finally:
        # 在上下文管理器结束时取消注册每个跳过类
        for skip in skips:
            skip.unregister()
```