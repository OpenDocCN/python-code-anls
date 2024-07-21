# `.\pytorch\torch\distributed\_spmd\parallel_mode.py`

```py
# 从abc模块导入ABC（抽象基类）和abstractmethod（抽象方法装饰器）
from abc import ABC, abstractmethod
# 从typing模块导入Any（表示任意类型）、Callable（可调用对象类型）、Dict（字典类型）、List（列表类型）、Optional（可选类型）、Tuple（元组类型）
from typing import Any, Callable, Dict, List, Optional, Tuple

# 导入PyTorch库
import torch
# 导入torch.distributed模块并命名为dist
import torch.distributed as dist
# 导入torch.utils._pytree模块并命名为pytree
import torch.utils._pytree as pytree
# 从torch._subclasses模块导入FakeTensorMode（虚拟张量模式）
from torch._subclasses import FakeTensorMode
# 从torch.distributed._spmd.data_parallel模块导入DataParallelStyle（数据并行风格）、partition_data_parallel（数据并行分区函数）
from torch.distributed._spmd.data_parallel import (
    DataParallelStyle,
    partition_data_parallel,
)
# 从torch.distributed._spmd.distribute模块导入_convert_to_distributed（转换为分布式函数）、Schema（模式）
from torch.distributed._spmd.distribute import _convert_to_distributed, Schema
# 从torch.distributed._tensor模块导入DeviceMesh（设备网格）、Placement（放置）、Replicate（复制）、Shard（碎片）
from torch.distributed._tensor import DeviceMesh, Placement, Replicate, Shard
# 从torch.fx模块导入GraphModule（图模块）
from torch.fx import GraphModule

# 定义ParallelMode类，继承自ABC（抽象基类）
class ParallelMode(ABC):
    """
    Basic Parallel Mode interface. Each parallelism pattern should implement
    this interface to describe how to partition and compile the graph in the
    spmd compiler.
    """

    # 定义partition方法，声明为抽象方法
    @abstractmethod
    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        """
        Partition a single device graph to a distributed graph.

        TODO(@wanchaol): some of these arguments are not necessary for
        partitioning, remove the unnecessary ones later.
        """
        raise NotImplementedError

    # 定义transform_and_compile方法，声明为抽象方法
    @abstractmethod
    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform and compile a distributed graph with a set of graph
        transformation and optimization passes for each parallel mode.

        The returned result should be a compiled executable graph in
        the distributed environment.
        """
        # TODO: add more necessary arguments to this interface.
        raise NotImplementedError

# 定义DataParallel类，继承自ParallelMode类，实现数据并行模式
class DataParallel(ParallelMode):
    """Data Parallelism mode."""

    # 初始化方法，定义数据并行模式的参数
    def __init__(
        self,
        parallel_style: str = "replicate",
        *,
        input_batch_dim: int = 0,
        custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None,
    ):
        """
        DataParallel Mode that partition the model and graph to data parallel style
        parallelism (i.e. DDP/FSDP/ZERO-3). It currently supports three different
        parallel styles: "replicate", "fully_shard", and "default". See
        :class:`DataParallelStyle` for more details.

        Args:
            parallel_style (str): parallel style to use. Currently supports
                "replicate", "fully_shard", and "default".

        Keyword args:
            input_batch_dim (int): the batch dimension of the input tensor.
                 default: 0
            custom_passes (Callable[[GraphModule], GraphModule], optional):
                A custom callable that overrides the default graph transformation
                and optimization passes.
        """
        # 根据用户传入的 parallel_style 设置并行风格
        if parallel_style == "replicate":
            self.parallel_style = DataParallelStyle.REPLICATE
        elif parallel_style == "fully_shard":
            self.parallel_style = DataParallelStyle.FULLY_SHARD
        elif parallel_style == "default":
            self.parallel_style = DataParallelStyle.DEFAULT
        else:
            # 如果传入的 parallel_style 无法识别，则抛出运行时异常
            raise RuntimeError(f"Unknown parallel style: {parallel_style}")

        # TODO: 如果用户传入了不正确的 `input_batch_dim`，我们应该如何检测并进行适当的错误处理？
        # 设置输入张量的批处理维度
        self.input_batch_dim = input_batch_dim

        # 如果用户传入了自定义的 passes 函数，则使用该函数作为优化传递函数，否则使用默认的 lambda 函数
        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            # TODO: 在此处添加一些默认的 passes
            self._gm_passes = lambda gm: gm

    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        # TODO: 找出一种方法来避免显式的 "cuda" mesh。
        # 创建一个 "cuda" 设备网格
        mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()))

        # 使用 partition_data_parallel 函数对 gm 进行数据并行分区
        gm = partition_data_parallel(
            gm,
            model,
            optimizer,
            params_and_buffers,
            named_states,
            args,
            kwargs,
            mesh,
            self.parallel_style,
            self.input_batch_dim,
        )
        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """使用一组优化 passes 来优化分布式图"""
        # TODO: 向这个接口添加更多必要的参数。
        # 调用 _gm_passes 方法来对 gm 进行变换和编译
        return self._gm_passes(gm)
# 定义一个名为 DTensorExpandMode 的类，继承自 ParallelMode 类
class DTensorExpandMode(ParallelMode):
    """
    DTensor Expand 模式。它复制参数并分片输入，以模拟 DDP 的行为，
    目前是在我们转向新的数据并行扩展之前的一个临时模式。
    """

    # 初始化方法，接受一个可选的自定义传递函数 custom_passes
    def __init__(
        self, custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None
    ):
        # _placements_override 属性，用于覆盖的放置方式字典，初始化为空字典
        self._placements_override: Dict[int, List[Placement]] = {}
        # 如果传入了 custom_passes 函数，则使用传入的函数作为 _gm_passes 属性
        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            # 否则使用 lambda 表达式将 gm 自身作为参数返回，作为默认的 _gm_passes 函数
            # TODO: 在这里添加一些默认的传递函数。
            self._gm_passes = lambda gm: gm

    # partition 方法，用于分片处理图模块 gm
    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        # 将参数 args 和 kwargs 展平为一个列表 flat_args
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)

        # 创建一个 DeviceMesh 对象，使用 "cuda" 设备和从 dist.get_world_size() 返回的张量
        mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()).cuda())
        # 创建分片模式 Schema，包含一个 mesh 和一个 Shard(0) 的放置方式
        shard_schema: Schema = Schema(mesh=mesh, placements=[Shard(0)])
        # FIXME: 允许其他分片模式的模式
        # 创建复制模式 Schema，包含一个 mesh 和一个 Replicate() 的放置方式
        replicate_schema: Schema = Schema(mesh=mesh, placements=[Replicate()])

        # 初始化空列表 inps 和 schemas
        inps, schemas = [], []

        # 遍历 params_and_buffers 的所有叶子节点，将每个 Tensor 添加到 inps 中，并使用 replicate_schema 添加到 schemas 中
        for p in pytree.tree_leaves(params_and_buffers):
            assert isinstance(p, torch.Tensor), f"expecting Tensor but got {type(p)}"
            inps.append(p)
            schemas.append(replicate_schema)

        # 遍历 named_states 的所有叶子节点，如果是 Tensor，则添加到 inps 中并使用 replicate_schema 添加到 schemas 中；否则添加一个空 Tensor
        for o in pytree.tree_leaves(named_states):
            if isinstance(o, torch.Tensor):
                inps.append(o)
                schemas.append(replicate_schema)
            else:
                inps.append(torch.empty(0))
                schemas.append(replicate_schema)

        # 遍历 flat_args 中的所有元素，如果是 Tensor，则添加到 inps 中，并根据 _placements_override 中的设置选择合适的 Schema，否则添加一个空 Tensor 和 shard_schema
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                inps.append(a)
                if id(a) in self._placements_override:
                    schemas.append(
                        Schema(mesh=mesh, placements=self._placements_override[id(a)])
                    )
                else:
                    schemas.append(shard_schema)
            else:
                # 对于非 Tensor 输入，创建一个虚拟的 Tensor 和 shard_schema 用于 dtensor 扩展的目的
                inps.append(torch.empty(0))
                schemas.append(shard_schema)

        # 使用 FakeTensorMode 上下文管理器，允许非虚拟输入
        with FakeTensorMode(allow_non_fake_inputs=True):
            # 创建 fake_inps 列表，包含与 inps 中每个元素相同形状的空 Tensor
            fake_inps = [torch.empty_like(inp) for inp in inps]

        # 调用 _convert_to_distributed 函数，将 gm、fake_inps、schemas 转换为分布式表示，返回结果的第一个元素
        return _convert_to_distributed(
            gm, fake_inps, schemas, default_mesh=mesh, _allow_partial=False
        )[0]
    # 定义一个方法，用于转换和编译一个分布式图形（GraphModule）
    # gm: GraphModule类型参数，表示输入的图形模块
    # 返回类型为GraphModule，表示经过转换和优化后的图形模块
    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform and compile a distributed graph with a set of graph transformation
        and optimization passes for the dtensor fallback parallel mode.
        """
        # TODO: 将传递给此函数的转换移动到其他地方
        # 调用私有方法_gm_passes，传入GraphModule对象gm进行处理
        return self._gm_passes(gm)
```