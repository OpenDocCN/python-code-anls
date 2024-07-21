# `.\pytorch\torch\distributed\tensor\parallel\style.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的模块和类
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,                # 导入 DeviceMesh 类，用于定义设备网格
    distribute_module,         # 导入 distribute_module 函数，用于分布式模块的处理
    distribute_tensor,         # 导入 distribute_tensor 函数，用于分布式张量的处理
    DTensor,                   # 导入 DTensor 类，用于分布式张量的表示
    Placement,                 # 导入 Placement 枚举，用于指定张量的布局位置
    Replicate,                 # 导入 Replicate 枚举，用于指定张量的复制策略
    Shard,                     # 导入 Shard 枚举，用于指定张量的分片策略
)

# 暴露给外部的模块列表
__all__ = [
    "ParallelStyle",           # 并行风格的抽象基类
    "RowwiseParallel",         # 行并行处理类
    "SequenceParallel",        # 序列并行处理类
    "ColwiseParallel",         # 列并行处理类
    "PrepareModuleInput",      # 模块输入预处理类
    "PrepareModuleOutput",     # 模块输出预处理类
]


class ParallelStyle(ABC):
    """
    并行风格的抽象基类，定义了模块或子模块的并行化方式。

    只定义了 ``apply`` 方法，供 ``parallelize_module`` 使用，这允许最大灵活性地实现不同类型的风格。
    """

    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        ...


class ColwiseParallel(ParallelStyle):
    """
    按列划分兼容的 nn.Module。目前支持 nn.Linear 和 nn.Embedding。
    用户可以与 RowwiseParallel 组合，以实现更复杂模块的分片（如 MLP、Attention）。

    关键字参数:
        input_layouts (Placement, optional):
            nn.Module 输入张量的 DTensor 布局，用于标注输入张量以成为 DTensor。如果未指定，假定输入张量被复制。
        output_layouts (Placement, optional):
            nn.Module 输出的 DTensor 布局，用于确保输出与用户期望的布局兼容。如果未指定，输出张量在最后一个维度上分片。
        use_local_output (bool, optional):
            是否使用本地的 :class:`torch.Tensor` 而不是 :class:`DTensor` 作为模块的输出，默认为 True。

    返回:
        表示 nn.Module 列分片的 :class:`ParallelStyle` 对象。

    示例::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m 是包含 "w1" nn.Linear 子模块的 nn.Module
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # 默认情况下，"w1" Linear 的输入将被转换为 Replicated DTensor
        >>> # "w1" 的输出将返回在最后一个维度上分片的 :class:`torch.Tensor`。
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel()})
        >>> ...
    """
    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there're operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    """

    # 定义一个类 ColwiseParallel，用于按列并行处理
    class ColwiseParallel(nn.Module):
        
        # 初始化方法，设置输入布局、输出布局以及是否使用本地输出
        def __init__(
            self,
            *,
            input_layouts: Optional[Placement] = None,
            output_layouts: Optional[Placement] = None,
            use_local_output: bool = True,
        ):
            super().__init__()
            # 设置输入布局，默认为 Replicate()
            self.input_layouts = (input_layouts or Replicate(),)
            # 设置输出布局，默认为在最后一个维度上进行分片
            self.output_layouts = (output_layouts or Shard(-1),)
            # colwise linear runtime sharding (desired sharding):
            # 1. requires replicate input
            # 2. shard output on last dim
            # 设置期望的输入布局，需要使用 Replicate() 作为输入
            self.desired_input_layouts = (Replicate(),)
            # 是否使用本地输出
            self.use_local_output = use_local_output

        @staticmethod
        # 准备输入的静态方法
        def _prepare_input_fn(
            input_layouts, desired_input_layouts, mod, inputs, device_mesh
        ):
            # TODO: figure out dynamo support for instance method and switch this to instance method

            # 使用 input_layouts 标注模块输入的布局/分片
            input_tensor = inputs[0]
            # 如果输入不是 DTensor 类型，则将其转换为 DTensor 类型
            if not isinstance(input_tensor, DTensor):
                input_tensor = DTensor.from_local(
                    input_tensor, device_mesh, input_layouts, run_check=False
                )

            # 将输入布局转换为 ColwiseParallel 的期望输入布局
            if input_layouts != desired_input_layouts:
                input_tensor = input_tensor.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            return input_tensor

        # 分片线性函数
        def _partition_linear_fn(self, name, module, device_mesh):
            # 将权重/偏置进行分片，权重使用 Shard(0) 分片
            # 对于 Colwise 作为线性运算，其输入为 input * weight^T + bias，
            # 其中权重会变成 Shard(1)
            for name, param in module.named_parameters():
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
                module.register_parameter(name, dist_param)

        # 分片嵌入函数
        def _partition_embedding_fn(self, name, module, device_mesh):
            # 直接将嵌入的权重 embedding.weight 进行 Shard(1) 分片
            for name, param in module.named_parameters():
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(1)]))
                module.register_parameter(name, dist_param)

        @staticmethod
        # 准备输出的静态方法
        def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
            # 输出在最后一个维度上是 Shard(-1) 分片的 DTensor
            if outputs.placements != output_layouts:
                outputs = outputs.redistribute(placements=output_layouts, async_op=True)
            # 转换回本地张量
            return outputs.to_local() if use_local_output else outputs
    # 定义私有方法 `_apply`，用于将给定的神经网络模块按照设备网格进行分布式部署，并返回部署后的模块
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # 如果模块是线性层（nn.Linear），选择线性层的分布式划分函数
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        # 如果模块是嵌入层（nn.Embedding），选择嵌入层的分布式划分函数
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            # 如果模块不是支持的类型，则抛出未实现错误
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        # 调用 distribute_module 函数，将模块分布到设备网格上
        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w2" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w2" Linear will be converted to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w2": RowwiseParallel()}),
        >>> ...

    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        # 设置输入数据的布局，默认为在最后一个维度上分片
        self.input_layouts = (input_layouts or Shard(-1),)
        # 设置输出数据的布局，默认为复制输出
        self.output_layouts = (output_layouts or Replicate(),)
        # 是否使用本地的 torch.Tensor 作为模块的输出，默认为 True
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # 获取输入张量
        input_tensor = inputs[0]
        # 如果输入张量不是 DTensor 类型，则将其从本地转换为 DTensor
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        # 如果输入数据布局与期望的输入布局不一致，则重新分配输入张量的布局
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor
    def _partition_linear_fn(self, name, module, device_mesh):
        # 定义线性层参数分区函数
        # 将权重分区为Shard(1)，偏置为Replicate()，权重为Shard(1)
        # 这意味着对于nn.Linear，计算结果为 input * weight^T + bias，
        # 其中权重会变成Shard(0)
        
        # 注册权重参数，并使用distribute_tensor函数在device_mesh上进行分发
        module.register_parameter(
            "weight",
            nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)])),
        )
        
        # 如果存在偏置，同样进行注册和分发处理
        if module.bias is not None:
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(module.bias, device_mesh, [Replicate()])
                ),
            )

    def _partition_embedding_fn(self, name, module, device_mesh):
        # 定义嵌入层参数分区函数
        # 将嵌入层的权重参数分区为Shard(0)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # 准备输出函数
        # 根据输出布局进行行级分片，可能是：
        # 1. 复制 -> 全部汇总
        # 2. 分片 -> 局部汇总
        if outputs.placements != output_layouts:
            # 如果输出布局与期望的不同，则重新分发输出
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        
        # 如果use_local_output为True，则将输出转换回本地张量
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # 应用函数，将模块在device_mesh上进行分布式处理
        if isinstance(module, nn.Linear):
            # 如果模块是线性层，则使用线性层的分区函数
            partition_fn = self._partition_linear_fn
            # 线性层的输入布局期望为Shard(-1)
            self.desired_input_layouts: Tuple[Placement, ...] = (Shard(-1),)
        elif isinstance(module, nn.Embedding):
            # 如果模块是嵌入层，则使用嵌入层的分区函数
            partition_fn = self._partition_embedding_fn
            # 嵌入层的输入布局期望为Replicate()
            self.desired_input_layouts = (Replicate(),)
        else:
            # 如果模块类型不支持，则抛出未实现的错误
            raise NotImplementedError(
                "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        # 将模块分发到device_mesh上，并应用相应的分区、输入和输出处理函数
        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
# SequenceParallel 类继承自 ParallelStyle，用于处理兼容的 nn.Module 参数的复制，并在序列维度上执行分片计算。
class SequenceParallel(ParallelStyle):
    """
    SequenceParallel replicates a compatible ``nn.Module`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__

    Both the input and output of the ``nn.Module`` will be sharded on the sequence dimension.

    Keyword Args:
        sequence_dim (int, optional):
            The sequence dimension of the input tensor for the ``nn.Module``, this is used to annotate the input tensor to
            become a DTensor that is sharded on the sequence dimension, default: 1.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: False.
    Returns:
        A :class:`ParallelStyle` object that represents Sequence Parallel of the ``nn.Module``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "norm" nn.LayerNorm submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "norm" will be converted to DTensor that shards on the sequence dim
        >>> # and the output of "norm" will return a sharded on sequence dimension :class:`DTensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"norm": SequenceParallel()}),
        >>> ...

    .. note:: SequenceParallel style assumes ones initialization if there are weights in the nn.Module (i.e.
        ``nn.LayerNorm`` or ``RMSNorm``, and they by default have ones initialization). If you have custom
        inits for the weights on those modules, you need to broadcast the weights before/after parallelizing
        to ensure that they are replicated.
    """

    # 初始化方法，设置序列维度和是否使用本地输出
    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False):
        super().__init__()  # 调用父类 ParallelStyle 的初始化方法
        self.sequence_dim = sequence_dim  # 设置序列维度
        self.use_local_output = use_local_output  # 设置是否使用本地输出

    # 私有方法，用于复制模块函数
    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        # 遍历模块中所有的命名参数
        for p_name, param in module.named_parameters():
            # 使用从LayerNorm/RMSNorm初始化的简单复制参数，允许我们直接使用from_local方法
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            # 将复制后的参数注册到模块中
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(sequence_dim, mod, inputs, device_mesh):
        # 获取输入张量
        input_tensor = inputs[0]
        # 如果输入是DTensor类型，则直接返回输入
        if isinstance(input_tensor, DTensor):
            return inputs
        # 如果输入是torch.Tensor类型，则将其转换为DTensor类型
        elif isinstance(input_tensor, torch.Tensor):
            return DTensor.from_local(
                input_tensor, device_mesh, [Shard(sequence_dim)], run_check=False
            )
        # 如果输入类型不符合预期，则引发数值错误异常
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        # 根据use_local_output的值，决定是否将输出转换为本地输出
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # 分布式应用模块
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,  # 复制模块函数
            partial(self._prepare_input_fn, self.sequence_dim),  # 准备输入函数的部分应用
            partial(self._prepare_output_fn, self.use_local_output),  # 准备输出函数的部分应用
        )
class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Module's inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.

    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    """


注释：
    # 初始化函数，接受多个命名参数，设置对象的初始状态
    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, Tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_output: bool = False,
    ):
        # 设置对象的 input_layouts 属性，如果 input_layouts 是 Placement 类型，则包装成元组
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        # 设置对象的 desired_input_layouts 属性，如果 desired_input_layouts 是 Placement 类型，则包装成元组
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        # 设置对象的 use_local_output 属性，默认为 False
        self.use_local_output = use_local_output
        # 如果 input_layouts 不为 None，则检查 desired_input_layouts 不能为 None
        if self.input_layouts is not None:
            assert (
                self.desired_input_layouts is not None
            ), "desired module inputs should not be None!"
            # 断言 input_layouts 和 desired_input_layouts 长度需相同
            assert len(self.input_layouts) == len(
                self.desired_input_layouts
            ), "input_layouts and desired_input_layouts should have same length!"
        # 判断是否有输入关键字参数
        self.with_kwargs = input_kwarg_layouts is not None
        # 设置对象的 input_kwarg_layouts 属性，如果为 None 则设为空字典
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        # 设置对象的 desired_input_kwarg_layouts 属性，如果为 None 则设为空字典
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        # 如果有输入关键字参数，则断言 input_kwarg_layouts 和 desired_input_kwarg_layouts 长度需相同
        if self.with_kwargs:
            assert len(self.input_kwarg_layouts) == len(
                self.desired_input_kwarg_layouts
            ), "input_kwarg_layouts and desired_input_kwarg_layouts should have same length!"

    # 准备输入参数的私有方法，根据布局需求对输入进行准备
    def _prepare_input_arg(
        self,
        input: Any,
        mesh: DeviceMesh,
        input_layout: Optional[Placement],
        desired_layout: Optional[Placement],
    ):
        # 如果指定了 input_layout
        if input_layout is not None:
            # 如果输入是 DTensor 类型
            if isinstance(input, DTensor):
                # TODO: 一旦修复编译路径，重新启用检查
                # 断言输入的布局与 input_layout 相符
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                # 断言输入为 torch.Tensor 类型
                assert isinstance(
                    input, torch.Tensor
                ), "expecting input to be a torch.Tensor!"
                # 从本地创建 DTensor 对象，使用给定的 mesh 和 input_layout，不运行检查
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            # 如果指定了 desired_layout，并且 input_layout 与 desired_layout 不同，则重新分配布局
            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            # 如果 use_local_output 为 True，则返回本地化的 dt_inp，否则返回 dt_inp
            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            # 如果未指定 input_layout，则直接返回输入 input
            return input
    # 准备输入函数，根据输入和设备网格准备输入数据
    def _prepare_input_fn(self, inputs, device_mesh):
        # 如果输入布局未定义，直接返回输入数据
        if self.input_layouts is None:
            return inputs
        # 准备输入列表
        prepared_inputs = []
        # 如果输入不是元组，则转换为元组
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        # 检查输入和输入布局的长度是否一致，若不一致则引发 ValueError 异常
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        # 断言期望的输入布局不为 None
        assert (
            self.desired_input_layouts is not None
        ), "desired module inputs should not be None!"
        # 遍历输入、输入布局和期望布局，准备输入参数
        for inp, input_layout, desired_layout in zip(
            inputs, self.input_layouts, self.desired_input_layouts
        ):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, input_layout, desired_layout)
            )
        # 返回准备好的输入数据的元组
        return tuple(prepared_inputs)

    # 准备关键字参数输入函数，根据输入和关键字输入准备数据
    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        # 准备位置参数输入
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        # 准备关键字参数输入的字典
        prepared_kwarg_inputs = {}
        # 遍历关键字输入的键
        for kwarg_key in kwarg_inputs.keys():
            kwarg_val = kwarg_inputs[kwarg_key]
            # 获取关键字参数的输入布局和期望布局
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)

            # 准备关键字参数的输入数据
            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )

        # 返回位置参数输入和准备好的关键字参数输入的元组
        return (prepared_arg_inputs, prepared_kwarg_inputs)

    # 应用函数，注册前处理钩子函数，根据是否有关键字参数决定注册哪种前处理钩子
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # 如果设置了关键字参数标志
        if self.with_kwargs:
            # 注册带关键字参数的前处理钩子函数
            module.register_forward_pre_hook(
                lambda _, inputs, kwargs: self._prepare_input_kwarg_fn(
                    inputs, kwargs, device_mesh
                ),
                with_kwargs=True,
            )  # type: ignore[misc]
        else:
            # 注册普通的前处理钩子函数
            module.register_forward_pre_hook(lambda _, inputs: self._prepare_input_fn(inputs, device_mesh))  # type: ignore[misc, call-arg]
        # 返回注册了前处理钩子的模块
        return module
    """
    Configure the nn.Module's outputs to convert the output tensors of the nn.Module to DTensors at runtime according to
    ``output_layouts``, and perform layout redistribution according to the ``desired_output_layouts``.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    """

    def __init__(
        self,
        *,
        output_layouts: Union[Placement, Tuple[Placement]],
        desired_output_layouts: Union[Placement, Tuple[Placement]],
        use_local_output: bool = True,
    ):
        # Initialize the PrepareModuleOutput object with provided arguments
        self.output_layouts = (
            (output_layouts,)  # Wrap `output_layouts` in a tuple if it's a single Placement object
            if isinstance(output_layouts, Placement)
            else output_layouts  # Otherwise, assume it's already a tuple of Placements
        )
        self.desired_output_layouts = (
            (desired_output_layouts,)  # Wrap `desired_output_layouts` in a tuple if it's a single Placement object
            if isinstance(desired_output_layouts, Placement)
            else desired_output_layouts  # Otherwise, assume it's already a tuple of Placements
        )
        self.use_local_output = use_local_output  # Flag indicating whether to use local tensors or DTensors
        # Assert that the length of `output_layouts` matches `desired_output_layouts`
        assert len(self.output_layouts) == len(
            self.desired_output_layouts
        ), "output_layouts and desired_output_layouts should have same length!"
    def _prepare_out_fn(self, outputs, device_mesh):
        prepared_outputs = []  # 准备存储处理后的输出结果列表
        if not isinstance(outputs, tuple):
            outputs = (outputs,)  # 如果输出不是元组，转换为元组
        if len(outputs) != len(self.output_layouts):
            raise ValueError(
                "module outputs and output_layouts should have same length!"
            )  # 如果输出数量与指定布局数量不匹配，抛出数值错误异常

        for out, out_layout, desired_out_layout in zip(
            outputs, self.output_layouts, self.desired_output_layouts
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    # TODO: 一旦我们修复编译路径，重新启用此检查
                    # assert out.placements[0] == out_layout
                    dt_out = out  # 如果输出是 DTensor 类型，直接使用
                else:
                    dt_out = DTensor.from_local(
                        out, device_mesh, (out_layout,), run_check=False
                    )  # 否则，从本地创建 DTensor 对象

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(placements=(desired_out_layout,))
                    # 如果输出布局与期望布局不同，重新分配张量的位置

                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )  # 将处理后的输出加入到准备好的输出列表中
            else:
                prepared_outputs.append(out)  # 如果输出布局为 None，直接添加原始输出到列表中

        if len(prepared_outputs) == 1:
            return prepared_outputs[0]  # 如果只有一个输出，返回单个输出
        else:
            return tuple(prepared_outputs)  # 如果有多个输出，返回元组形式的输出列表

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_forward_hook(lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh))  # 注册前向钩子函数，将处理函数与模块绑定
        return module  # 返回经过处理后的模块对象
```