# `.\pytorch\torch\_decomp\decompositions_for_rng.py`

```
# 设置类型检查允许未标记的函数定义
# 导入 functools 库，用于高阶函数
# 从 collections 库导入 defaultdict，创建默认字典
# 从 typing 库导入 Callable 和 Dict，用于类型提示

import functools
from collections import defaultdict
from typing import Callable, Dict

# 导入 torch 库
import torch
# 导入 torch._decomp 模块，并将其重命名为 decomp
import torch._decomp as decomp
# 从 torch._decomp 模块导入 get_decompositions 函数
from torch._decomp import get_decompositions
# 从 torch._ops 模块导入 OpOverload 类
from torch._ops import OpOverload

# 使用 torch.ops.aten 别名 aten 来引用 torch 操作的 aten 命名空间

aten = torch.ops.aten

# 创建默认字典 rng_decompositions，其键是字符串，值是 OpOverload 到 Callable 的字典
rng_decompositions: Dict[str, Dict[OpOverload, Callable]] = defaultdict(dict)


# 定义函数 register_rng_decomposition，接受一个参数 aten_op
def register_rng_decomposition(aten_op):
    # 调用 decomp.register_decomposition 函数，将 aten_op 注册到 rng_decompositions 字典中
    return decomp.register_decomposition(aten_op, rng_decompositions)


# 定义函数 throw_on_non_cuda，接受一个参数 device
def throw_on_non_cuda(device):
    # 抛出 RuntimeError 异常，说明不支持在非 cuda 设备上的 RNG 操作
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


# 注册 RNG 操作 aten.rand 的分解函数
# TODO - We have to register many more distributions here, and also higher level
# ops like dropout which have fused implementation and can hide the rand inside.
@register_rng_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    # 如果设备存在且不是 "cuda" 类型，调用 throw_on_non_cuda 函数抛出异常
    if device and device.type != "cuda":
        throw_on_non_cuda(device)
    # 获取 PhiloxStateTracker 的状态作为种子和偏移量
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    # 设置 dtype 为 torch.float32，如果未指定的话
    dtype = dtype or torch.float32
    # 调用 torch.ops.rngprims.philox_rand 函数进行随机数生成
    out, offset_jump = torch.ops.rngprims.philox_rand(
        shape, seed, offset, None, device, dtype
    )
    # 更新偏移量
    PhiloxStateTracker.advance_offset(offset_jump)
    # 返回生成的随机数
    return out


# 注册 RNG 操作 aten.rand_like 的分解函数
@register_rng_decomposition(aten.rand_like)
def rand_like(
    x: torch.Tensor,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=torch.preserve_format,
):
    # 如果 device 不存在，则使用 x 的设备
    device = device or x.device
    # 如果设备类型不是 "cuda"，调用 throw_on_non_cuda 函数抛出异常
    if device.type != "cuda":
        throw_on_non_cuda(device)
    # 设置 dtype 为 x 的数据类型，如果未指定的话
    dtype = dtype or x.dtype
    # 获取 PhiloxStateTracker 的状态作为种子和偏移量
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    # 调用 torch.ops.rngprims.philox_rand 函数进行随机数生成
    out, offset_jump = torch.ops.rngprims.philox_rand(
        x.shape, seed, offset, None, device, dtype
    )
    # 更新偏移量
    PhiloxStateTracker.advance_offset(offset_jump)
    # 返回生成的随机数
    return out


# 定义类 PhiloxState，表示 PhiloxRngState 的状态，包括种子、基础偏移量和相对偏移量
class PhiloxState:
    """
    Represents a PhiloxRngState - (seed, offset) where offset = base_offset +
    relative_offset. seed and base_offset basically point to the rng state just
    before tracing starts. relative offset tracks the totally consumed offset at
    trace time.
    """

    # 定义初始化方法，初始化种子、基础偏移量、相对偏移量等状态
    def __init__(self):
        self.reset()

    # 定义重置方法，重置种子、基础偏移量、相对偏移量等状态
    def reset(self):
        self.seed = torch.tensor(())
        self.base_offset = torch.tensor(())
        self.relative_offset = 0
        self.offset_advanced_alteast_once = False

    # 定义验证状态方法，确保种子和基础偏移量不为空
    def validate_state(self):
        assert self.seed.numel() != 0 and self.base_offset.numel() != 0

    # 定义更新偏移量方法，根据传入的消耗偏移量更新相对偏移量
    def advance_offset(self, consumed_offset):
        self.offset_advanced_alteast_once = True
        self.relative_offset = self.relative_offset + consumed_offset

    # 定义设置状态方法，设置种子、基础偏移量和相对偏移量
    def set_state(self, seed, base_offset, relative_offset=0):
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset
    # 返回当前状态的元组表示，包括种子和相对偏移量
    def get_state_as_tuple(self):
        # 确保状态有效性
        self.validate_state()
        return (self.seed, self.base_offset + self.relative_offset)

    # 返回当前状态的张量表示，用于覆盖 get_rng_state 方法
    def get_state_as_tensor(self):
        # 只在覆盖 get_rng_state 方法时需要
        self.validate_state()
        return torch.stack([self.seed, self.base_offset + self.relative_offset])

    # 从给定的张量状态设置种子和基础偏移量，用于覆盖 set_rng_state 方法
    def set_state_from_tensor(self, state):
        # 只在覆盖 set_rng_state 方法时需要
        self.seed, self.base_offset = torch.unbind(state)
        self.relative_offset = 0
class PhiloxStateTracker:
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. For dynamic shapes, these
    offsets are a function of input shapes. Therefore, the AOT generated graphs
    have additional outputs that compute total consumed forward and backward
    offsets.
    """

    running_state: PhiloxState  # 跟踪当前运行的 Philox 状态
    fwd_state: PhiloxState      # 前向运行的 Philox 状态
    bwd_state: PhiloxState      # 反向运行的 Philox 状态

    def __enter__(self):
        PhiloxStateTracker.reset()  # 进入上下文时重置 PhiloxStateTracker 的状态
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb):
        PhiloxStateTracker.reset()  # 退出上下文时同样重置 PhiloxStateTracker 的状态

    @classmethod
    def reset(cls):
        cls.running_state = PhiloxState()  # 重置运行状态为新的 PhiloxState 实例
        cls.fwd_state = PhiloxState()      # 重置前向状态为新的 PhiloxState 实例
        cls.bwd_state = PhiloxState()      # 重置后向状态为新的 PhiloxState 实例

    @classmethod
    def mark_beginning_of_forward(cls):
        # 告知追踪器使用前向状态作为当前运行状态
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        # 告知追踪器使用后向状态作为当前运行状态
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        # 记录种子和偏移张量。这些张量用于调用 philox_rand 功能原语。
        if mode == "forward":
            cls.fwd_state.set_state(seed, offset)  # 设置前向状态的种子和偏移
            cls.mark_beginning_of_forward()       # 标记开始进行前向操作
        else:
            assert mode == "backward"
            cls.bwd_state.set_state(seed, offset)  # 设置后向状态的种子和偏移

    @classmethod
    def get_state_as_tensor(cls):
        # 该方法存在的唯一原因是在跟踪期间覆盖了 get_rng_state 和 set_rng_state。
        # get_rng_state 期望张量输出，因此返回 (seed, offset) 元组以适应程序的其他部分。

        # 坏处是如果用户保存和恢复 RNG 状态，生成的代码会有些混乱，
        # 首先将 (seed, offset) 连接为一个张量以供 get_rng_state 使用，
        # 然后再分割回 (seed, offset) 元组以供 set_rng_state 使用。

        # TODO: 调查是否有更好的方法来包装元组成为一个假的 Tensor 对象，然后在稍后解开它。
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def set_state_from_tensor(cls, x):
        # 这仅在我们覆盖了 set_rng_state 时才需要。查看 get_state_from_tensor 方法中的注释。
        cls.running_state.set_state_from_tensor(x)

    @classmethod
    def advance_offset(cls, consumed_offset):
        cls.running_state.advance_offset(consumed_offset)
    # 返回当前类的运行状态的相对偏移量
    def get_current_relative_offset(cls):
        return cls.running_state.relative_offset

    # 静态方法：确保偏移量是4的倍数，适用于 torch CUDA RNG 状态偏移量
    # 当我们累加所有元素时，结果可能不是4的倍数，这个方法确保它是4的倍数。
    def multiple_of_4(offset):
        return (offset + 3) // 4 * 4

    # 类方法：获取更新后的前向传播偏移量
    def get_updated_fwd_offset(cls):
        # 如果没有观察到随机操作，则快速返回基础偏移量
        if not cls.fwd_state.offset_advanced_alteast_once:
            return cls.fwd_state.base_offset
        # 计算更新后的前向传播偏移量，确保是4的倍数
        return cls.multiple_of_4(
            cls.fwd_state.base_offset + cls.fwd_state.relative_offset
        )

    # 类方法：获取更新后的反向传播偏移量
    def get_updated_bwd_offset(cls):
        # 如果没有观察到随机操作，则快速返回基础偏移量
        if not cls.bwd_state.offset_advanced_alteast_once:
            return cls.bwd_state.base_offset
        # 计算更新后的反向传播偏移量，确保是4的倍数
        return cls.multiple_of_4(
            cls.bwd_state.base_offset + cls.bwd_state.relative_offset
        )
# 增加更多的分解操作，这些操作最终会在分解器中使用 rand_like。
# 将这些操作添加到 rng_decompositions 中，确保 rand_like 在这些分解操作中被功能化。
# 这个列表是从感应器代码库复制的，它用于类似的目的。
#
# 注意 - 这些分解操作的精度不同于 eager 模式。然而，我们不能仅仅通过配置标志（如 fallback_random）禁用它们，
# 因为为了功能化 rng 操作，我们必须对这些操作进行分解。
extra_random_decomps = get_decompositions(
    [
        aten.cauchy,                # 柯西分布
        aten.cauchy_,               # 柯西分布（原地操作）
        aten.exponential,           # 指数分布
        aten.exponential_,          # 指数分布（原地操作）
        aten.geometric,             # 几何分布
        aten.geometric_,            # 几何分布（原地操作）
        aten.native_dropout,        # 本地丢弃
        aten.normal,                # 正态分布
        aten.normal_,               # 正态分布（原地操作）
        aten.normal_functional,     # 正态分布（函数式操作）
        aten.log_normal,            # 对数正态分布
        aten.log_normal_,           # 对数正态分布（原地操作）
        aten.rrelu_with_noise,      # 带噪声的随机修正线性单元
        aten.rrelu_with_noise_,     # 带噪声的随机修正线性单元（原地操作）
        aten.uniform_,              # 均匀分布（原地操作）
    ]
)
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)

# 注册额外的随机数分解操作，特别针对 bernoulli_ 方法
@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    if self.device == torch.device("cpu"):
        return NotImplemented
    return self.copy_(torch.rand_like(self, dtype=torch.float32) < p)

# 注册额外的随机数分解操作，特别针对 bernoulli_p 方法
@register_extra_random_decomp([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    if self.device == torch.device("cpu"):
        return NotImplemented
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < p

# 更新 rng_decompositions 集合，将额外的随机数分解操作合并进来
rng_decompositions.update(extra_random_decomps)  # type: ignore[arg-type]
```