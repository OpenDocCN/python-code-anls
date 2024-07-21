# `.\pytorch\torch\testing\_internal\common_fsdp.py`

```py
# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import contextlib                  # 上下文管理模块，用于管理代码块的上下文
import os                          # 操作系统接口模块，提供了与操作系统交互的函数
import re                          # 正则表达式模块，用于处理字符串匹配和操作
import sys                         # 系统相关的参数和功能的模块
import warnings                    # 警告处理模块，用于控制警告信息的显示

from abc import ABC, abstractmethod  # 抽象基类模块，定义抽象基类和抽象方法
from contextlib import nullcontext  # 上下文管理模块的一个特殊上下文，不执行任何操作
from copy import deepcopy          # 深拷贝模块，用于创建对象的深层副本
from enum import auto, Enum        # 枚举模块，用于定义枚举类型和自动编号
from functools import wraps        # 装饰器模块，用于封装函数，增加额外功能
from typing import (               # 类型提示模块，用于静态类型检查和提示
    Any,                          # 任意类型
    Callable,                     # 可调用对象类型
    Dict,                         # 字典类型
    List,                         # 列表类型
    no_type_check,                # 标记类型不进行类型检查
    Optional,                     # 可选类型
    Tuple,                        # 元组类型
    Type,                         # 类型对象
    Union,                        # 联合类型
)
from unittest import mock         # 单元测试模块的 mock 对象，用于模拟测试对象

import torch                      # PyTorch 深度学习库
import torch.distributed as dist  # PyTorch 分布式通信模块
import torch.nn as nn             # PyTorch 神经网络模块
import torch.nn.functional as F   # PyTorch 神经网络函数模块
from torch.distributed._composable import checkpoint  # 分布式通信模块的检查点功能
from torch.distributed._composable.fsdp import fully_shard  # 分布式通信模块的完全分片功能
from torch.distributed._composable.fsdp._fsdp_param_group import (
    FSDPParamGroup,              # FSDP 参数组模块
    RegisterPostBackwardFunction,  # 注册后向函数模块
)
from torch.distributed._tensor import distribute_tensor, DTensor, Shard  # 分布式通信模块的张量分发和分片
from torch.distributed.device_mesh import DeviceMesh  # 设备网格模块
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP  # FSDP 模块和完全分片数据并行模块
from torch.distributed.fsdp._common_utils import TrainingState  # FSDP 公共工具模块的训练状态
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES  # FSDP 初始化工具模块的策略常量
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,            # 后向预取模块
    MixedPrecision,              # 混合精度模块
    ShardingStrategy,            # 分片策略模块
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler  # 分片梯度缩放器模块
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap  # FSDP 包装策略模块
from torch.distributed.tensor.parallel import (
    ColwiseParallel,             # 列并行模块
    parallelize_module,          # 并行化模块
    RowwiseParallel,             # 行并行模块
    SequenceParallel,            # 序列并行模块
)
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # Transformer 编解码器层模块
from torch.nn.parallel.distributed import DistributedDataParallel as DDP  # 分布式数据并行模块
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,        # 多进程测试用例模块
    MultiThreadedTestCase,       # 多线程测试用例模块
    run_subtests,                # 运行子测试函数
    TEST_SKIPS,                  # 测试跳过策略常量
)
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms  # 内部测试的文件模式和获取每毫秒周期数函数
from torch.utils._triton import has_triton  # Triton 加速库的检查函数


class FSDPInitMode(Enum):
    # FSDP 初始化模式枚举
    # 不进行 FSDP 包装
    NO_FSDP = auto()
    # FSDP 递归包装
    RECURSIVE = auto()
    # TODO: FSDP 非递归包装
    # NONRECURSIVE = auto()


class CUDAInitMode(Enum):
    # CUDA 初始化模式枚举
    # 在传递给 FSDP 构造函数之前将模型移动到 CUDA
    CUDA_BEFORE = auto()
    # 在传递给 FSDP 构造函数之后将模型移动到 CUDA
    CUDA_AFTER = auto()
    # 保持在 CPU 上
    CUDA_NEVER = auto()


class FSDPTestModel(nn.Module, ABC):
    """This defines the interface expected from all models used commonly for
    FSDP unit tests."""

    @abstractmethod
    def get_input(self, device) -> Tuple[torch.Tensor, ...]:
        """Returns an input for the model as as tuple."""
        ...

    @abstractmethod
    def get_loss(self, input, output) -> torch.Tensor:
        """Returns the loss given the input and output."""
        ...

    @abstractmethod
    def run_backward(self, loss) -> None:
        """Runs the backward pass (e.g. including ``loss.backward()``)."""
        ...
    # 使用 @staticmethod 装饰器将 init 方法定义为静态方法，使得可以在不创建实例的情况下调用
    # 使用 @abstractmethod 装饰器声明 init 方法为抽象方法，要求任何子类都必须实现这个方法
    def init(*args: Any, **kwargs: Any) -> nn.Module:
        """Initializes an instance of this model."""
        # 这是一个抽象方法的定义，子类需要实现该方法以初始化模型实例
        ...
def _assert_module_states(
    model: nn.Module,
    process_group: dist.ProcessGroup,
    assert_fn: Callable,
):
    """
    All-gathers module states across ranks and calls ``assert_fn`` on each pair
    of corresponding states from rank 0 and a nonzero rank. For example, if
    ``assert_fn`` is ``self.assertEqual()``, then this checks that all module
    states are equal across ranks.
    """
    # 获取模型参数和缓冲区的名称及其对应的数据，用于调试方便
    named_module_states = [
        (param_name, param.detach().cpu())
        for param_name, param in model.named_parameters()
    ]
    named_module_states += [
        (buffer_name, buffer.detach().cpu())
        for buffer_name, buffer in model.named_buffers()
    ]
    # 获取进程组中的总进程数
    world_size = dist.get_world_size(process_group)
    # 创建一个列表，用于存储从各个进程收集到的模型状态
    olist = [None for _ in range(world_size)]
    # 在进程组中所有进程之间收集模型状态
    dist.all_gather_object(olist, named_module_states, group=process_group)
    # 获取主进程（rank 0）的模型状态
    rank0_states = olist[0]
    assert rank0_states is not None  # mypy
    # 逐一比较其他进程的模型状态与主进程的状态，并调用指定的断言函数进行比较
    for state in olist[1:]:
        assert state is not None  # mypy
        for (_, p1), (_, p2) in zip(rank0_states, state):
            assert_fn(p1, p2)


def _zero_model(
    model: nn.Module,
    zero_buffers: bool = False,
    summon_full=True,
):
    """Zeros the parameters and optionally buffers of ``model`` in place."""
    # 根据 summon_full 参数选择是否完全召唤模型参数
    ctx = FSDP.summon_full_params(model) if summon_full else nullcontext()
    with ctx:
        # 将模型的参数全部置零
        for param in model.parameters():
            with torch.no_grad():
                param.zero_()
        # 如果 zero_buffers 为 True，则将模型的缓冲区也置零
        if zero_buffers:
            for buffer in model.buffers():
                with torch.no_grad():
                    buffer.zero_()


def _get_state_dict(model, cpu_offload=False, half=False):
    # 根据 cpu_offload 参数决定是否将模型移动到 GPU 上
    if not cpu_offload:
        model = model.cuda()
    # 如果 half 参数为 True，则将模型的数据类型设置为半精度浮点数
    if half:
        model.half()

    return model.state_dict()


def subtest_name(test_name_mapping, *args):
    # 根据传入的参数和映射表生成子测试名称
    return "_".join(
        [test_name_mapping[str(s)] if s is not None else "none" for s in args]
    )


def _broadcast_state_dict(rank, state_dict):
    # 对于非 FSDP 根进程，在 rank 0 上的部分模型状态可能不在 CPU 上，所以将所有参数移动到 CPU 上以避免问题
    for param_name, param in state_dict.items():
        if param.device != torch.device("cpu"):
            state_dict[param_name] = param.cpu()

    # 创建一个包含状态字典的列表，仅在 rank 0 上广播
    olist = [state_dict if rank == 0 else None]
    # 使用分布式工具包广播对象列表
    dist.broadcast_object_list(olist)
    # 获取广播后的状态字典
    state_dict = olist[0]
    # 确保状态字典中的参数在 CUDA 上
    for param_name in state_dict.keys():
        state_dict[param_name] = state_dict[param_name].cuda()
    return state_dict


def get_full_params(model: nn.Module, recurse: bool = True):
    """
    Returns the full unsharded parameters of ``model``. Any FSDP-managed
    parameters offloaded to CPU are moved to GPU in the returned list.
    """
    # 返回模型的完整参数状态字典，根据 recurse 参数决定是否递归获取子模块的参数
    Args:
        recurse (bool): 如果为 ``False``，仅对 ``model`` 直接参数进行反分片；如果为 ``True``，则递归遍历从 ``model`` 根节点开始的整个模块层次结构。
    """
    使用 FSDP.summon_full_params 方法对模型进行全参数调用，根据 recurse 参数确定是否递归处理。
    返回模型参数的深度拷贝列表。
# 定义一个函数 _maybe_cuda，用于将模型移动到 CUDA 设备上，如果 move_to_cuda 为 True，则移动，否则不移动
def _maybe_cuda(model: nn.Module, move_to_cuda: bool):
    return model.cuda() if move_to_cuda else model

# 定义一个函数 _maybe_wrap_fsdp，用于可能地包装模型到 FSDP（Fully Sharded Data Parallelism）中，如果 wrap_fsdp 为 True，则包装模型，否则返回原模型
def _maybe_wrap_fsdp(model: nn.Module, wrap_fsdp: bool, *args, **kwargs):
    return model if not wrap_fsdp else FSDP(model, *args, **kwargs)

# 定义 DummyProcessGroup 类，模拟分布式训练的进程组
class DummyProcessGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank  # 初始化进程组的排名
        self._size = size  # 初始化进程组的大小

    # 返回进程组的排名
    def rank(self) -> int:
        return self._rank

    # 返回进程组的大小
    def size(self) -> int:
        return self._size

    # 模拟执行分布式的 allreduce 操作，返回一个模拟的 Future 对象
    def allreduce(self, *args, **kwargs):
        dist_wait = mock.Mock()

        # 定义获取 Future 对象的方法，设置结果为 1 的 Future
        def get_future():
            future: torch.futures.Future = torch.futures.Future()
            future.set_result(1)
            return future

        dist_wait.get_future = get_future
        return dist_wait

# 定义 TransformerWithSharedParams 类，继承自 FSDPTestModel，包含共享参数的 Transformer 模型
class TransformerWithSharedParams(FSDPTestModel):
    def __init__(
        self,
        group: dist.ProcessGroup,
        cuda_init_mode: CUDAInitMode,
        add_bn: bool,
        deterministic: bool,
    ):
        super().__init__()  # 调用父类构造函数

        self.rank = group.rank()  # 获取进程组的排名
        self.world_size = group.size()  # 获取进程组的大小

        if deterministic:
            torch.manual_seed(0)  # 如果需要确定性，设置随机种子为 0

        d_vocab = 23  # 词汇大小
        d_model = 16  # 模型维度

        # 定义词嵌入层
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        # 定义 Transformer 模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=8,
            dropout=0.1,
        )
        # 定义输出投影层
        self.output_proj = nn.Linear(d_model, d_vocab)

        # 共享词嵌入和输出投影的权重
        self.output_proj.weight = self.embed_tokens.weight
        # 注册一个缓冲区，用全为 1 的张量初始化，与词嵌入维度一致
        self.register_buffer(
            "vocab_bias", self.embed_tokens.weight.new_ones((d_model,))
        )
        # 注册一个长期使用的缓冲区，初始化为全零，与 vocab_bias 的数据类型一致
        self.register_buffer(
            "long_buffer",
            torch.zeros_like(self.vocab_bias, dtype=torch.long),
        )  # type: ignore[arg-type]

        self.bs = 2  # 批量大小
        # 如果 add_bn 为 True，则使用 BatchNorm1d，否则使用恒等映射
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()

        # 如果 cuda_init_mode 是 CUDA_BEFORE，则将模型移动到 CUDA 设备上
        if cuda_init_mode == CUDAInitMode.CUDA_BEFORE:
            self = self.cuda()
        
        # 如果需要确定性，则设置模型为评估模式
        if deterministic:
            self.eval()

    # 获取输入数据，设备由参数 device 指定
    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # 保持所有操作的确定性
        src = torch.arange(12, device=device).view(6, self.bs)  # T x B
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)  # T x B
        return (src, tgt)

    # 前向传播方法，接收源序列和目标序列的索引作为输入，返回预测的输出
    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)  # type: ignore[operator]
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    # 计算损失函数，接收输入和模型输出，返回交叉熵损失
    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum"
        )
    # 定义一个方法用于反向传播计算梯度
    def run_backward(self, loss):
        loss.backward()

    # 静态方法，用于初始化操作
    @staticmethod
    # 参数包括分布式进程组、FSDP（Fully Sharded Data Parallelism）初始化模式、CUDA 初始化模式
    # 还可以选择提供其他FSDP参数的字典、是否确定性操作、是否添加批归一化
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        add_bn: bool = True,
        ):
        ) -> Union[nn.Module, FSDP]:
        """
        Initializes a :class:`TransformerWithSharedParams` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps with
                top-level FSDP. By default, the top-level FSDP uses the
                ``ModuleWrapPolicy`` for encoder and decoder layers, but a
                different auto wrap policy may be specified via
                ``fsdp_kwargs``.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
            add_bn (bool): Whether to include batch norm in the model.
        """

        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        # 检查是否禁用了 FSDP
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            # 如果 group 是 tuple，则选择第一个元素作为 pg，否则直接使用 group
            if isinstance(group, tuple):
                pg = group[0]
            else:
                pg = group
            # 返回一个未包装 FSDP 的 TransformerWithSharedParams 实例
            return TransformerWithSharedParams(
                pg, cuda_init_mode, add_bn, deterministic
            )
        # 检查是否使用了递归初始化 FSDP
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # 默认使用 ModuleWrapPolicy
            if "auto_wrap_policy" not in fsdp_kwargs:
                auto_wrap_policy = ModuleWrapPolicy(
                    {
                        TransformerEncoderLayer,
                        TransformerDecoderLayer,
                    }
                )
            else:
                # 如果指定了 auto_wrap_policy，则使用给定的策略
                auto_wrap_policy = fsdp_kwargs.pop("auto_wrap_policy")

            # 如果指定了 sharding_strategy 且属于 HYBRID_SHARD 或 _HYBRID_SHARD_ZERO2，并且 group 不是 tuple
            if (
                "sharding_strategy" in fsdp_kwargs
                and fsdp_kwargs["sharding_strategy"]
                in {ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2}
                and not isinstance(group, tuple)
            ):
                fsdp_pg = None
            else:
                fsdp_pg = group

            # 根据 group 是否为 tuple 决定使用的 pg
            if isinstance(group, tuple):
                tformer_pg = group[0]
            else:
                tformer_pg = group

            # 创建 TransformerWithSharedParams 实例
            m = TransformerWithSharedParams(
                tformer_pg, cuda_init_mode, add_bn, deterministic
            )
            # 使用 FSDP 封装模型
            fsdp_model = FSDP(
                m,
                fsdp_pg,
                auto_wrap_policy=auto_wrap_policy,
                **fsdp_kwargs,
            )
            # 根据 CUDA 初始化模式，将模型移到 GPU 上
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        # 抛出不支持的 FSDP 初始化模式异常
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")

    def get_ignored_modules(self):
        # 返回忽略的模块列表，这里返回了 self.transformer
        return [self.transformer]
class NestedWrappedModule(FSDPTestModel):
    # 定义一个嵌套包装的模块，继承自FSDPTestModel
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        cuda_init_mode: CUDAInitMode,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 获取当前进程组的排名和大小
        self.rank = group.rank()
        self.world_size = group.size()
        # 根据CUDA初始化模式决定是否将模型移动到GPU
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        # 定义一个内部函数，根据wrap_fsdp的值可能会对层进行FSDP包装
        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer

        # 如果需要确定性操作，则设置随机数种子为0
        if deterministic:
            torch.manual_seed(0)
        
        # 定义模型为一个Sequential容器，包含多个层
        self.module = nn.Sequential(
            _maybe_cuda(nn.Linear(8, 4), move_to_cuda),  # 可能移动到CUDA的线性层
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)),  # 可能移动到CUDA的线性层
                    _maybe_cuda(nn.Linear(16, 16), move_to_cuda),  # 可能移动到CUDA的线性层
                ),
            ),
            _maybe_wrap(_maybe_cuda(nn.Linear(16, 4), move_to_cuda)),  # 可能移动到CUDA的线性层
            _maybe_cuda(nn.Linear(4, 8), move_to_cuda),  # 可能移动到CUDA的线性层
        )

    # 返回模型输入，保持一致性地确定性
    def get_input(self, device):
        torch.manual_seed(1 + self.rank)
        return (torch.rand(4, 8, device=device),)

    # 前向传播函数
    def forward(self, x):
        return self.module(x)

    # 计算损失函数
    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    # 执行反向传播
    def run_backward(self, loss):
        loss.backward()

    # 静态方法：初始化模型
    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    def initialize_nested_module(
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> nn.Module:
        """
        Initializes a :class:`NestedWrappedModule` instance.
    
        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps some nested
                modules with FSDP but not the top-level module. The model may
                later be wrapped with a top-level FSDP external to this method
                if desired.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
        """
    
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
    
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            # Return an instance of NestedWrappedModule without FSDP wrapping
            return NestedWrappedModule(
                group,
                wrap_fsdp=False,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
            )
    
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # Initialize NestedWrappedModule with FSDP wrapping for nested modules
            fsdp_model = NestedWrappedModule(
                group,
                wrap_fsdp=True,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()  # Move model to CUDA if specified
            return fsdp_model
    
        # Raise an error if an unsupported FSDP init mode is provided
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")
class NonUniformReqGradNWM(NestedWrappedModule):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        cuda_init_mode: CUDAInitMode,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super(NestedWrappedModule, self).__init__()
        # This `__init__` only differs from `NestedWrappedModule.__init__` in that
        # the last two `nn.Linear` layers are FSDP wrapped in a `nn.Sequential`
        # container. This arrangement results in all elements of the last two parameters
        # residing on a single rank. Freezing all parameters except those two allows us
        # to verify that `ShardedGradScaler` accommodates situations where some ranks
        # have no (non-zero sized) parameter shards.
        
        # 获取当前进程组的秩和大小
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE
        
        # 定义一个函数，根据需求可能对层进行 FSDP 封装
        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer
        
        # 如果设置了确定性，则设置随机种子为0
        if deterministic:
            torch.manual_seed(0)
        
        # 使用 nn.Sequential 封装神经网络模块
        self.module = nn.Sequential(
            # 对第一个线性层应用 CUDA 移动和可能的 FSDP 封装
            _maybe_cuda(nn.Linear(8, 4), move_to_cuda),
            # 对第二个 nn.Sequential 容器应用可能的 FSDP 封装
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)),
                    _maybe_cuda(nn.Linear(16, 16), move_to_cuda),
                ),
            ),
            # 对第三个 nn.Sequential 容器应用可能的 FSDP 封装
            _maybe_wrap(
                nn.Sequential(
                    _maybe_cuda(nn.Linear(16, 4), move_to_cuda),
                    _maybe_cuda(nn.Linear(4, 8), move_to_cuda),
                ),
            ),
        )
    def _set_nonuniform_req_grad(model, req_grad_mask) -> None:
        # 遍历模型的所有参数
        for n, p in model.named_parameters():
            # 如果参数名不符合 req_grad_mask 所指定的模式
            if not re.match(req_grad_mask, n):
                # 将该参数的 requires_grad 属性设置为 False
                p.requires_grad_(False)

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ):
        """
        Initializes a :class:`NestedWrappedModule` instance, but unlike
        :meth:`NestedWrappedModule.init`, it wraps a second :class:`torch.nn.Sequential`
        container to enable the desired non-uniform ``requires_grad``
        ``use_orig_params=True`` tests. For both ``RECURSIVE`` and ``NO_FSDP``
        init modes, freezes all parameters except the last two to validate
        ``ShardedGradScaler`` support for ranks with no (non-zero sized) local shards in
        FSDP ``use_orig_params=True`` mode.
        """
        # 定义一个正则表达式模式，用于匹配需要保持梯度非冻结的参数名
        req_grad_pattern = re.compile(r"module\.2.*\.1.*")
        
        # 根据 fsdp_init_mode 参数的不同选择初始化模式
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            # 创建一个 NonUniformReqGradNWM 实例，禁用 FSDP 包装
            ddp_model = NonUniformReqGradNWM(
                group,
                wrap_fsdp=False,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
            )
            # 调用 _set_nonuniform_req_grad 方法，冻结不需要保持梯度的参数
            NonUniformReqGradNWM._set_nonuniform_req_grad(ddp_model, req_grad_pattern)
            # 返回初始化后的模型实例
            return ddp_model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # 如果 fsdp_kwargs 为 None，则设置为空字典
            if fsdp_kwargs is None:
                fsdp_kwargs = {}
            # 创建一个 NonUniformReqGradNWM 实例，启用 FSDP 包装
            fsdp_model = NonUniformReqGradNWM(
                group,
                wrap_fsdp=True,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            # 如果 cuda_init_mode 为 CUDAInitMode.CUDA_AFTER，则将模型移动到 CUDA 设备
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            # 调用 _set_nonuniform_req_grad 方法，冻结不需要保持梯度的参数
            NonUniformReqGradNWM._set_nonuniform_req_grad(fsdp_model, req_grad_pattern)
            # 返回初始化后的模型实例
            return fsdp_model
        # 如果 fsdp_init_mode 不是预期的值，则抛出 ValueError 异常
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")
class ModuleWithDelay(FSDPTestModel):
    """This class wraps a :class:`FSDPTestModel` to optionally add a delay
    after computing the loss and/or before the gradient reduction."""

    def __init__(
        self,
        module: nn.Module,
        delay_after_loss_ms: int,
        delay_before_reduction_ms: int,
    ):
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms  # 设置延迟毫秒数，用于在计算损失后/优化步骤前添加延迟
        self.delay_before_reduction_ms = delay_before_reduction_ms  # 设置延迟毫秒数，用于在减少散射梯度之前添加延迟
        self.module = module  # 初始化要包装的模块对象

    def get_input(self, device):
        return self.module.get_input(device)  # 调用包装模块的获取输入方法

    def forward(self, x):
        return self.module(x)  # 调用包装模块的前向传播方法

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)  # 调用包装模块的获取损失方法
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))  # 如果延迟时间大于零，则在CUDA上休眠指定的毫秒数
        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed.reduce_scatter_tensor

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(
                    int(self.delay_before_reduction_ms * get_cycles_per_ms())
                )  # 如果延迟时间大于零，则在CUDA上休眠指定的毫秒数
            return orig_reduce_scatter(*args, **kwargs)

        with mock.patch(
            "torch.distributed.reduce_scatter_tensor", _delayed_reduce_scatter
        ):
            self.module.run_backward(loss)  # 使用包装模块的方法运行反向传播

    @staticmethod
    def init(
        module_class: Type[FSDPTestModel],
        *model_args: Any,
        delay_after_loss_ms: int,
        delay_before_reduction_ms: int,
        **model_kwargs: Any,
    ):
        """
        Args:
            module_class (Type[FSDPTestModel]): Wrapped module class to which
                to add delays.
            model_args: Positional arguments forwarded to the ``module_class``
                ``init()``.
            delay_after_loss_ms (int): Delay after computing the loss/before
                the optimizer step (in ms).
            delay_before_reduction_ms (int): Delay before reduce-scattering
                gradients (in ms).
            model_kwargs: Keyword arguments forwarded to the ``module_class``
                ``init()``.
        """
        return ModuleWithDelay(
            module_class.init(*model_args, **model_kwargs),  # 初始化包装模块
            delay_after_loss_ms,
            delay_before_reduction_ms,
        )


class NestedWrappedModuleWithDelay(ModuleWithDelay):
    @staticmethod
    def init(  # type: ignore[override]
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode = CUDAInitMode.CUDA_AFTER,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        delay_after_loss_ms: int = 0,
        delay_before_reduction_ms: int = 0,
    ):
        # 该方法用于初始化嵌套的包装模块，继承自ModuleWithDelay类
        ...
    ):
        # 如果需要延迟初始化模块，返回延迟初始化后的模块实例
        return ModuleWithDelay.init(
            NestedWrappedModule,  # 使用 NestedWrappedModule 初始化延迟加载的模块
            group=group,  # 设置模块组
            fsdp_init_mode=fsdp_init_mode,  # 设置模块的初始化模式
            cuda_init_mode=cuda_init_mode,  # 设置 CUDA 初始化模式
            fsdp_kwargs=fsdp_kwargs,  # FSDP 参数
            deterministic=deterministic,  # 是否确定性初始化
            delay_after_loss_ms=delay_after_loss_ms,  # 损失计算后延迟时间（毫秒）
            delay_before_reduction_ms=delay_before_reduction_ms,  # 汇总前延迟时间（毫秒）
        )
class DummyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module  # 初始化时保存传入的模块对象引用

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)  # 前向传播调用保存的模块对象


class MixtureOfExperts(NestedWrappedModule):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        cuda_init_mode: CUDAInitMode,
        delay_before_free_ms: int,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super().__init__(
            group=group,
            wrap_fsdp=wrap_fsdp,
            cuda_init_mode=cuda_init_mode,
            deterministic=deterministic,
        )
        self.group = group  # 保存传入的进程组对象引用
        self.delay_before_free_ms = delay_before_free_ms  # 保存延迟释放的毫秒数
        self.wrap_fsdp = wrap_fsdp  # 是否包装为FSDP对象的标志
        self.move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE  # 是否在初始化时移动到CUDA设备的标志

        if deterministic:
            # 设置随机种子，确保每个进程有不同的专家参数
            torch.manual_seed(42 + self.rank)

        d_expert = 23  # 专家模型的输入维度
        d_shared = 12  # 共享模型的输出维度
        d_input = 8  # 输入数据的维度

        expert = _maybe_cuda(nn.Linear(d_expert, d_shared), self.move_to_cuda)  # 创建可能位于CUDA上的专家模型

        # 统计专家模型参数的总数，并为每个参数设置“expert”属性
        self.num_expert_params = sum(p.numel() for p in expert.parameters())
        for p in expert.parameters():
            p.expert = True  # 标记为专家模型参数

        if deterministic:
            # 设置随机种子，确保所有进程共享相同的其他参数
            torch.manual_seed(0)

        shared = _maybe_cuda(nn.Linear(d_shared, d_expert), self.move_to_cuda)  # 创建可能位于CUDA上的共享模型

        if wrap_fsdp:
            # 如果需要包装为FSDP对象，则创建大小为1的进程组用于专家模型参数
            expert_group = torch.distributed.new_group(
                [group.rank()]
            )  # 当前进程创建大小为1的进程组
            expert = FSDP(expert, expert_group, **fsdp_kwargs)  # 包装专家模型为FSDP对象
            shared = FSDP(shared, group, **fsdp_kwargs)  # 包装共享模型为FSDP对象

        # 构建模型的前向传播顺序
        self.module = nn.Sequential(
            _maybe_cuda(nn.Linear(d_input, d_shared), self.move_to_cuda),  # 输入到共享层的线性变换
            shared,  # 共享模型层
            expert,  # 专家模型层
            _maybe_cuda(nn.Linear(d_shared, d_input), self.move_to_cuda),  # 共享层到输出的线性变换
        )

    def forward(self, x):
        if self.delay_before_free_ms > 0:
            expert = self.module[2]  # 获取模型中的专家模型层
            if isinstance(expert, FSDP):
                orig_reshard = torch.distributed.fsdp._runtime_utils._reshard

                def _delayed_reshard(*args, **kwargs):
                    torch.cuda._sleep(
                        int(self.delay_before_free_ms * get_cycles_per_ms())
                    )  # 延迟释放CUDA内存
                    return orig_reshard(*args, **kwargs)

                # 使用mock.patch延迟重分片操作，确保在模型计算前进行延迟
                with mock.patch(
                    "torch.distributed.fsdp._runtime_utils._reshard", _delayed_reshard
                ):
                    return self.module(x)

        return self.module(x)  # 正常执行模型的前向传播
    def run_backward(self, loss):
        # 对损失函数进行反向传播
        loss.backward()
        
        # 如果未包装在FullyShardedDataParallel中，则手动减少梯度
        if not self.wrap_fsdp:
            # 使用torch.no_grad()上下文管理器，确保不计算梯度
            with torch.no_grad():
                # 遍历模型的所有参数
                for p in self.parameters():
                    # 如果参数具有"expert"属性，则跳过，这些参数不需要梯度减少
                    if hasattr(p, "expert"):
                        continue
                    # 如果参数的梯度不为None，则将梯度除以进程数
                    if p.grad is not None:
                        p.grad.div_(self.world_size)
                        # 全局同步所有进程的梯度
                        torch.distributed.all_reduce(p.grad, group=self.group)

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        delay_before_free_ms: int = 0,
    ):
        """
        Initializes a :class:`MixtureOfExperts` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps some nested
                modules with FSDP, including the expert and shared layers, but
                not the top-level module. The model may later be wrapped with a
                top-level FSDP external to this method if desired.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
            delay_before_free_ms (int): Delay before resharding expert
                parameters in the forward pass (in ms).
        """
        # 如果未提供fsdp_kwargs，则设为空字典
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        
        # 根据fsdp_init_mode的不同选项进行初始化不同的MixtureOfExperts实例
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return MixtureOfExperts(
                group,
                wrap_fsdp=False,
                cuda_init_mode=cuda_init_mode,
                delay_before_free_ms=delay_before_free_ms,
                deterministic=deterministic,
            )
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # 创建包含FSDP的MixtureOfExperts实例
            fsdp_model = MixtureOfExperts(
                group,
                wrap_fsdp=True,
                cuda_init_mode=cuda_init_mode,
                delay_before_free_ms=delay_before_free_ms,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            # 如果cuda_init_mode为CUDA_AFTER，则将模型移到GPU
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        # 如果fsdp_init_mode不是支持的选项，则引发异常
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")
class MLP(nn.Module):
    # 定义一个多层感知机（MLP）的神经网络模型
    def __init__(
        self,
        dim: int,
        device: Optional[torch.device] = None,
        *,
        bias: bool = True,
        with_buffer: bool = False,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        # 输入层到隐藏层的线性变换，输入维度为dim，输出维度为dim_multiplier * dim
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        # 隐藏层到输出层的线性变换，输入维度为dim_multiplier * dim，输出维度为dim
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)
        # 如果设置了with_buffer为True，则注册一个缓冲区，否则缓冲区为None
        if with_buffer:
            self.register_buffer("buffer", torch.randn((dim,), device=device))
        else:
            self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x经过输入层到隐藏层的线性变换
        z = self.in_proj(x)
        # 使用ReLU激活函数处理隐藏层输出
        z = F.relu(z)
        # 经过隐藏层到输出层的线性变换
        z = self.out_proj(z)
        # 再次使用ReLU激活函数处理输出层输出
        z = F.relu(z)
        # 如果缓冲区不为None，则将缓冲区的值加到输出中
        if self.buffer is not None:
            z = z + self.buffer
        # 返回处理后的输出张量
        return z

    def reset_parameters(self):
        # 如果缓冲区不为None，则重新初始化缓冲区中的值
        if self.buffer is not None:
            torch.nn.init.normal_(self.buffer)


class MLPStack(nn.Sequential):
    # 定义多个MLP模型堆叠的序列容器
    def __init__(self, mlp_dim: int, *, with_seq_parallel: bool = False):
        modules: List[nn.Module] = [
            # 使用3倍的维度乘数创建一个MLP模型
            MLP(mlp_dim, dim_multiplier=3),
            # 创建一个标准的MLP模型
            MLP(mlp_dim),
            # 再次使用3倍的维度乘数创建另一个MLP模型
            MLP(mlp_dim, dim_multiplier=3),
        ]
        # 如果指定了使用序列并行处理
        if with_seq_parallel:
            # 将一个LayerNorm层添加到模块列表中
            modules.append(nn.LayerNorm(mlp_dim, bias=False))
        super().__init__(*modules)
        # 记录是否使用序列并行处理
        self.with_seq_parallel = with_seq_parallel

    def parallelize(
        self,
        tp_mesh: DeviceMesh,
        dp_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        **fsdp_kwargs,
    ) -> "MLPStack":
        # 定义并行化计划，为模型中的不同部分分配不同的并行处理策略
        parallelize_plan = {
            "0.in_proj": ColwiseParallel(use_local_output=False),
            "0.out_proj": RowwiseParallel(use_local_output=False),
            "1.in_proj": ColwiseParallel(use_local_output=False),
            "1.out_proj": RowwiseParallel(use_local_output=False),
            "2.in_proj": ColwiseParallel(use_local_output=False),
            "2.out_proj": RowwiseParallel(output_layouts=Shard(1))
            if self.with_seq_parallel
            else RowwiseParallel(),
        }
        # 如果使用了序列并行处理
        if self.with_seq_parallel:
            # 添加一个SequenceParallel层到并行化计划中
            parallelize_plan["3"] = SequenceParallel(sequence_dim=1)
        # 调用parallelize_module函数，将模型并行化处理
        parallelize_module(self, device_mesh=tp_mesh, parallelize_plan=parallelize_plan)
        # 遍历模型中的每个子模块
        for module in self:
            # 如果是LayerNorm层，则跳过
            if isinstance(module, nn.LayerNorm):
                continue
            # 如果使用了激活检查点技术，则对当前模块应用检查点
            if use_activation_checkpointing:
                checkpoint(module)
            # 对当前模块进行全面分片分布处理
            fully_shard(module, mesh=dp_mesh, **fsdp_kwargs)
        # 对整个模型进行全面分片分布处理
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)
        # 返回并行化后的模型
        return self


class DoubleLinear(nn.Module):
    """
    This can be used for returning multiple outputs from a module
    (``use_second_linear=True``) or for having an unused module (``False``).
    """
    # 双线性模块，用于从模块中返回多个输出（use_second_linear=True）或存在未使用的模块（False）
    # 初始化函数，用于初始化神经网络模型的各个组件
    def __init__(self, dim: int, use_second_linear: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入和输出维度均为dim，用于进行线性变换
        self.lin1 = nn.Linear(dim, dim)
        # 创建另一个线性层，同样输入和输出维度为dim，用于第二次线性变换
        self.lin2 = nn.Linear(dim, dim)
        # 创建ReLU激活函数的实例，用于非线性变换
        self.relu = nn.ReLU()
        # 是否使用第二个线性层的标志
        self.use_second_linear = use_second_linear

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # 如果设置使用第二个线性层
        if self.use_second_linear:
            # 对输入数据x先分别经过lin1和lin2线性层，再经过ReLU激活函数，返回两个张量的元组
            return self.relu(self.lin1(x)), self.relu(self.lin2(x))
        # 否则只使用lin1线性层和ReLU激活函数，返回单个张量
        return self.relu(self.lin1(x))
# NOTE: For these patch methods, if we want safety under multi-threading (e.g.
# when using multi-threaded process group), then we want:
# (1) a barrier immediately after reading the original value to ensure that all
# threads see the same original value
# (2) a barrier immediately before restoring the original value to ensure that
# all threads use the patched value inside the context

@contextlib.contextmanager
def patch_all_gather(new_all_gather_into_tensor: Callable):
    # 保存原始的 dist.all_gather_into_tensor 函数
    orig_all_gather = dist.all_gather_into_tensor
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 dist.all_gather_into_tensor 替换为新的函数 new_all_gather_into_tensor
    dist.all_gather_into_tensor = new_all_gather_into_tensor
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 dist.all_gather_into_tensor 函数
        dist.all_gather_into_tensor = orig_all_gather


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter_tensor: Callable):
    # 保存原始的 dist.reduce_scatter_tensor 函数
    orig_reduce_scatter = dist.reduce_scatter_tensor
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 dist.reduce_scatter_tensor 替换为新的函数 new_reduce_scatter_tensor
    dist.reduce_scatter_tensor = new_reduce_scatter_tensor
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 dist.reduce_scatter_tensor 函数
        dist.reduce_scatter_tensor = orig_reduce_scatter


@contextlib.contextmanager
def patch_all_reduce(new_all_reduce: Callable):
    # 保存原始的 dist.all_reduce 函数
    orig_all_reduce = dist.all_reduce
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 dist.all_reduce 替换为新的函数 new_all_reduce
    dist.all_reduce = new_all_reduce
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 dist.all_reduce 函数
        dist.all_reduce = orig_all_reduce


@no_type_check
@contextlib.contextmanager
def patch_unshard(new_unshard: Callable):
    # 保存原始的 FSDPParamGroup.unshard 函数
    orig_unshard = FSDPParamGroup.unshard
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 FSDPParamGroup.unshard 替换为新的函数 new_unshard
    FSDPParamGroup.unshard = new_unshard
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 FSDPParamGroup.unshard 函数
        FSDPParamGroup.unshard = orig_unshard


@no_type_check
@contextlib.contextmanager
def patch_reshard(new_reshard: Callable):
    # 保存原始的 FSDPParamGroup.reshard 函数
    orig_reshard = FSDPParamGroup.reshard
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 FSDPParamGroup.reshard 替换为新的函数 new_reshard
    FSDPParamGroup.reshard = new_reshard
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 FSDPParamGroup.reshard 函数
        FSDPParamGroup.reshard = orig_reshard


@no_type_check
@contextlib.contextmanager
def patch_post_backward(new_post_backward: Callable):
    # 保存原始的 FSDPParamGroup.post_backward 函数
    orig_post_backward = FSDPParamGroup.post_backward
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 FSDPParamGroup.post_backward 替换为新的函数 new_post_backward
    FSDPParamGroup.post_backward = new_post_backward
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 FSDPParamGroup.post_backward 函数
        FSDPParamGroup.post_backward = orig_post_backward


@no_type_check
@contextlib.contextmanager
def patch_register_post_backward_hook_backward(new_backward: Callable):
    # 保存原始的 RegisterPostBackwardFunction.backward 函数
    orig_backward = RegisterPostBackwardFunction.backward
    # 插入 barrier，确保所有线程在此处看到相同的原始值
    dist.barrier()
    # 将 RegisterPostBackwardFunction.backward 替换为新的函数 new_backward
    RegisterPostBackwardFunction.backward = new_backward
    try:
        yield
    finally:
        # 再次插入 barrier，确保所有线程在此处使用上下文内的修补值
        dist.barrier()
        # 恢复原始的 RegisterPostBackwardFunction.backward 函数
        RegisterPostBackwardFunction.backward = orig_backward


def reduce_scatter_with_assert(
    cls,
    orig_reduce_scatter: Callable,
    assert_fn: Callable,  # `assert_fn(output: Tensor)`
    *args: Any,
    **kwargs: Any,
):
    if len(args) > 0:
        output = args[0]
    elif "output" in kwargs:
        output = kwargs["output"]
    else:
        # 如果不满足条件，则抛出断言错误，显示无法获取reduce-scatter的输出，同时列出传入的args和kwargs参数
        raise AssertionError(
            f"Cannot get reduce-scatter output from\nargs: {args}\nkwargs: {kwargs}"
        )
    # 对输出进行断言检查，确保输出符合预期
    assert_fn(output)
    # 返回原始reduce_scatter函数的结果，传入原始的args和kwargs参数
    return orig_reduce_scatter(*args, **kwargs)
def check_sharded_parity(
    cls,  # 单元测试类
    replicated_module: nn.Module,  # 复制的模块
    sharded_module: nn.Module,  # 分片的模块
    prefixes_to_ignore: Tuple[str, ...] = (),  # 忽略的前缀列表，默认为空元组
):
    for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
        replicated_module.named_parameters(), sharded_module.named_parameters()
    ):
        clean_sharded_name = sharded_name
        # 移除指定前缀，生成干净的分片模块参数名
        for prefix in prefixes_to_ignore:
            clean_sharded_name = clean_sharded_name.replace(prefix, "")
        # 断言复制模块参数名与干净的分片模块参数名相等
        cls.assertEqual(replicated_name, clean_sharded_name)
        # 断言分片模块参数是 DTensor 类型
        cls.assertIsInstance(sharded_param, DTensor)
        assert isinstance(sharded_param, DTensor)  # mypy
        # 获取分片模块参数的设备网格和放置信息
        mesh, placements = sharded_param.device_mesh, sharded_param.placements
        # 如果放置信息为 (Shard(0), Shard(0))，抛出断言错误
        if tuple(placements) == (Shard(0), Shard(0)):
            raise AssertionError(
                "FSDP's (Shard(0), Shard(0)) layout differs from distribute_tensor(), "
                "so we cannot check for equality using it"
            )
        # 使用 distribute_tensor 函数对复制模块参数进行分片参考参数的分布式转移
        sharded_ref_param = distribute_tensor(replicated_param, mesh, placements)
        # 断言分片模块参数的本地化结果与分片参考参数的本地化结果相等
        cls.assertEqual(sharded_param.to_local(), sharded_ref_param.to_local())
        # 如果复制模块参数的梯度为 None，断言分片模块参数的梯度也为 None，并继续下一次循环
        if replicated_param.grad is None:
            cls.assertIsNone(sharded_param.grad)
            continue
        # 断言分片模块参数的梯度不为 None
        cls.assertIsNotNone(sharded_param.grad)
        # 使用 distribute_tensor 函数对复制模块参数梯度进行分片参考梯度的分布式转移
        sharded_ref_grad = distribute_tensor(replicated_param.grad, mesh, placements)
        # 断言分片模块参数的梯度是 DTensor 类型
        cls.assertIsInstance(sharded_param.grad, DTensor)
        assert isinstance(sharded_param.grad, DTensor)  # mypy
        # 断言分片模块参数的梯度的本地化结果与分片参考梯度的本地化结果相等


class FSDPTestMultiThread(MultiThreadedTestCase):
    @property
    def world_size(self):
        return torch.cuda.device_count() if torch.cuda.is_available() else 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    def perThreadSetUp(self):
        torch._dynamo.reset()

    def perThreadTearDown(self):
        torch._dynamo.reset()


class FSDPTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # 设置 TORCH_NCCL_DESYNC_DEBUG=0 以禁用 NCCL 的 `workCleanupLoop()`，
        # 这可以避免单元测试的不稳定性：
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "0"
        self._spawn_processes()

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8) if torch.cuda.is_available() else 4

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def init_method(self):
        return f"{FILE_SCHEMA}{self.file_name}"

    def _check_cpu_offload(self, fsdp_model, cpu_offload):
        self.assertEqual(cpu_offload, fsdp_model.cpu_offload)
    # 验证传入的 backward_prefetch 参数是否与 fsdp_model 中的相同
    def _check_backward_prefetch(self, fsdp_model, backward_prefetch):
        self.assertEqual(backward_prefetch, fsdp_model.backward_prefetch)

    # 验证传入的 forward_prefetch 参数是否与 fsdp_model 中的相同
    def _check_forward_prefetch(self, fsdp_model, forward_prefetch):
        self.assertEqual(forward_prefetch, fsdp_model.forward_prefetch)

    # 执行子测试，调用 run_subtests 函数并返回结果
    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    # 类方法，用于执行测试
    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        # 创建一个类实例
        self = cls(test_name)
        # 设置实例的 rank 属性
        self.rank = rank
        # 设置实例的 file_name 属性
        self.file_name = file_name

        # 打印初始化信息，包括进程 rank 和 world size
        print(f"dist init r={self.rank}, world={self.world_size}")

        # 根据 CUDA 是否可用选择使用的后端，确保 init_process_group() 调用成功
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        try:
            # 初始化进程组，指定初始化方法、后端、world size 和 rank
            dist.init_process_group(
                init_method=self.init_method,
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            # 处理运行时错误，如果错误信息中包含 "recompile"，则退出并指定测试跳过
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        device_ids = None
        if torch.cuda.is_available() and torch.cuda.device_count():
            # 如果 CUDA 可用且有 GPU，则计算设备 ID 并设置当前设备
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            device_ids = [device_id]

        # 执行 barrier 操作，确保每个进程初始化完成，避免因为跳过测试导致不一致性
        dist.barrier(device_ids=device_ids)

        # 重置 Torch._dynamo 状态，运行测试函数
        torch._dynamo.reset()
        self.run_test(test_name, pipe)
        torch._dynamo.reset()

        # 再次执行 barrier 操作，确保每个进程完成测试后同步
        dist.barrier(device_ids=device_ids)

        # 销毁进程组
        dist.destroy_process_group()

    # 训练模型多个步骤的方法，接受多个参数来配置训练过程
    def _train_for_several_steps(
        self,
        model: nn.Module,
        num_steps: int,
        autocast: bool,
        lr: float = 0.01,
        fsdp_cpu_offload: Optional[CPUOffload] = None,
        save_model: bool = False,
        mixed_precision: Optional[MixedPrecision] = None,
        enable_sharded_grad_scaler: bool = False,
        use_pure_fp16: bool = False,
        sharded_grad_scaler_kwargs: Optional[Dict[str, Any]] = None,
    # 定义私有方法 _test_fsdp_parity，用于测试 Flexible Model-Parallelism（FSDP）的一致性
    # 这个方法接受多个参数来配置测试环境和模型行为
    # model_class: 要测试的模型类别，必须是 FSDPTestModel 的子类
    # fsdp_init_mode: FSDP 初始化模式，控制 FSDP 的初始化方式
    # cuda_init_mode: CUDA 初始化模式，控制 CUDA 的初始化方式
    # ref_init_fn: 可选参数，参考的初始化函数，用于设置模型的初始状态
    # num_iters: 执行测试的迭代次数
    # save_model: 是否保存模型状态
    # cpu_offload: 控制是否使用 CPU 卸载
    # backward_prefetch: 可选参数，反向传播预取策略
    # sharding_strategy: 可选参数，分片策略
    # mixed_precision: 可选参数，混合精度配置
    # forward_prefetch: 是否启用前向传播预取
    # use_orig_params: 是否使用原始参数
    # enable_sharded_grad_scaler: 是否启用分片梯度缩放器
    # use_pure_fp16: 是否使用纯粹的 FP16
    # init_kwargs: 可选参数，用于额外的初始化参数
    # sharded_grad_scaler_kwargs: 可选参数，分片梯度缩放器的额外参数
    # **fsdp_kwargs: 其余 FSDP 相关的参数，以字典形式传入
def test_compiled_fsdp(compile_compute_on_module: Optional[type] = None):
    # 定义一个测试函数，用于测试带有编译计算的FSDP（Fully Sharded Data Parallelism）
    def fully_shard_with_compiled_compute(*args, **kwargs):
        # 调用torch.distributed._composable.fsdp.fully_shard函数，类型忽略操作符
        torch.distributed._composable.fsdp.fully_shard(*args, **kwargs)  # type: ignore[operator]
        # 如果指定了编译计算的模块或者args的第一个参数是指定类型的实例，则编译这个模块
        if compile_compute_on_module is None or isinstance(
            args[0], compile_compute_on_module
        ):
            args[0].compile()

    class FullyShardMode(Enum):
        EAGER = auto()
        COMPILED_COMPUTE = auto()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 保存原始的fully_shard函数引用
            original_fully_shard = torch.distributed._composable.fsdp.fully_shard
            # 遍历FullyShardMode枚举
            for mode in FullyShardMode:
                # 如果不是EAGER模式且没有Triton，则发出警告并继续下一个模式的处理
                if mode != FullyShardMode.EAGER and not has_triton():
                    warnings.warn("Inductor on GPU needs Triton and recent GPU arch")
                    continue
                # barrier以确保线程读取相同的值
                original_skip_fsdp_hooks = torch._dynamo.config.skip_fsdp_hooks
                original_compile_threads = torch._inductor.config.compile_threads
                torch.distributed.barrier()

                if mode == FullyShardMode.EAGER:
                    # 如果是EAGER模式，则使用原始的fully_shard函数
                    fully_shard_patch = original_fully_shard
                elif mode == FullyShardMode.COMPILED_COMPUTE:
                    # 如果是COMPILED_COMPUTE模式，则设置相关的配置，使用带有编译计算的fully_shard函数
                    torch._dynamo.config.skip_fsdp_hooks = True
                    torch._inductor.config.compile_threads = 1
                    fully_shard_patch = fully_shard_with_compiled_compute  # type: ignore[assignment]
                else:
                    # 如果是其他模式，则抛出未实现错误
                    raise NotImplementedError(
                        f"Need to implement FullyShardMode={mode}"
                    )

                # 将patch后的fully_shard函数作为全局变量导入到func的全局命名空间中
                func.__globals__[original_fully_shard.__name__] = fully_shard_patch
                # 调用func函数，传递其参数和关键字参数
                func(*args, **kwargs)
                # 其他线程在这个线程恢复之前使用patch过的func函数
                torch.distributed.barrier()
                # 恢复原始的fully_shard函数引用
                func.__globals__[original_fully_shard.__name__] = original_fully_shard
                torch._dynamo.config.skip_fsdp_hooks = original_skip_fsdp_hooks
                torch._inductor.config.compile_threads = original_compile_threads

        return wrapper

    return decorator


class SkipModule(nn.Module):
    # 定义一个简单的神经网络模块SkipModule
    def __init__(self):
        super().__init__()
        # 初始化一个线性层，输入和输出维度均为10，无偏置
        self.lin = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        # 前向传播函数，返回线性层的输出
        return self.lin(x)


class NestedLinear(nn.Module):
    # 定义一个嵌套的神经网络模块NestedLinear
    def __init__(self, fsdp_wrap):
        super().__init__()
        # 根据fsdp_wrap参数决定是否对线性层进行FSDP封装
        if fsdp_wrap:
            self.nested_linear = wrap(nn.Linear(10, 10, bias=False).cuda())
        else:
            self.nested_linear = nn.Linear(10, 10, bias=False).cuda()

    def forward(self, x):
        # 前向传播函数，直接返回嵌套线性层的输出
        return self.nested_linear(x)


class SkipModel(nn.Module):
    # 定义一个简单的神经网络模块SkipModel
    # 定义类的初始化方法，接受一个名为 double_nest 的参数
    def __init__(self, double_nest):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入大小为10，输出大小为10，不使用偏置，并将其放在 GPU 上
        self.linear = nn.Linear(10, 10, bias=False).cuda()
        # 创建一个 SkipModule 实例，并将其放在 GPU 上
        self.linear_skip = SkipModule().cuda()
        # 调用 wrap 函数来包装一个 NestedLinear 实例，传入参数 double_nest
        self.nested_linear = wrap(NestedLinear(fsdp_wrap=double_nest))

    # 定义前向传播方法
    def forward(self, x):
        # 将输入 x 传入 self.linear，得到线性层的输出
        x = self.linear(x)
        # 将线性层的输出 x 传入 self.linear_skip，得到 SkipModule 的输出
        x = self.linear_skip(x)
        # 将 SkipModule 的输出 x 传入 self.nested_linear，得到 NestedLinear 的输出
        x = self.nested_linear(x)
        # 返回最终的输出 x
        return x
```