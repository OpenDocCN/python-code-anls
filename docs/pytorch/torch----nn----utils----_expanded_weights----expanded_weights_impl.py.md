# `.\pytorch\torch\nn\utils\_expanded_weights\expanded_weights_impl.py`

```py
# mypy: allow-untyped-defs
# 引入 functools 模块，用于高阶函数的支持
import functools
# 引入 contextmanager 用于创建上下文管理器
from contextlib import contextmanager
# 引入 Callable 和 Dict 用于类型提示
from typing import Callable, Dict

# 引入 torch 模块
import torch
# 从 torch._decomp 模块引入 decomposition_table
from torch._decomp import decomposition_table
# 从 torch.utils._pytree 模块引入 tree_map_only
from torch.utils._pytree import tree_map_only

# 定义全局变量 HANDLED_FUNCTIONS，类型为字典，用于映射函数到 torch.autograd.Function 对象
HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function] = {}

# 设置 aten 为 torch._ops.ops.aten，用于访问 torch 原始操作
aten = torch._ops.ops.aten

# expanded_weights_rnn_decomps 定义了不同 RNN 类型的扩展权重的分解方式
expanded_weights_rnn_decomps = {
    # torch.rnn_relu 的分解方式
    torch.rnn_relu: (
        decomposition_table[aten.rnn_relu.input],
        decomposition_table[aten.rnn_relu.data],
    ),
    # torch.rnn_tanh 的分解方式
    torch.rnn_tanh: (
        decomposition_table[aten.rnn_tanh.input],
        decomposition_table[aten.rnn_tanh.data],
    ),
    # torch.lstm 的分解方式
    torch.lstm: (
        decomposition_table[aten.lstm.input],
        decomposition_table[aten.lstm.data],
    ),
    # torch.gru 的分解方式
    torch.gru: (
        decomposition_table[aten.gru.input],
        decomposition_table[aten.gru.data],
    ),
}


# batch_second 上下文管理器用于确保 RNN 操作中批次维度次序为第二维
@contextmanager
def batch_second(args, kwargs):
    # 设置 ExpandedWeight 对象的 batch_first 属性为 False
    def set_batch_second(ew):
        ew.set_batch_first(False)

    # 将 ExpandedWeight 对象的 batch_first 属性重置为 True
    def reset_batch_first(ew):
        ew.set_batch_first(True)

    # 对 args 中的 ExpandedWeight 对象应用 set_batch_second 函数
    tree_map_only(ExpandedWeight, set_batch_second, args)
    # 对 kwargs 中的 ExpandedWeight 对象应用 set_batch_second 函数
    tree_map_only(ExpandedWeight, set_batch_second, kwargs)
    try:
        yield
    finally:
        # 对 args 中的 ExpandedWeight 对象应用 reset_batch_first 函数
        tree_map_only(ExpandedWeight, reset_batch_first, args)
        # 对 kwargs 中的 ExpandedWeight 对象应用 reset_batch_first 函数
        tree_map_only(ExpandedWeight, reset_batch_first, kwargs)


# allow_smaller_batches 上下文管理器用于支持 packed sequences，允许更小的批次
@contextmanager
def allow_smaller_batches(args, kwargs):
    # 设置 ExpandedWeight 对象的 allow_smaller_batches 属性为 True
    def allow(ew):
        ew.set_allow_smaller_batches(True)

    # 将 ExpandedWeight 对象的 allow_smaller_batches 属性重置为 False
    def reset(ew):
        ew.set_allow_smaller_batches(False)

    # 对 args 中的 ExpandedWeight 对象应用 allow 函数
    tree_map_only(ExpandedWeight, allow, args)
    # 对 kwargs 中的 ExpandedWeight 对象应用 allow 函数
    tree_map_only(ExpandedWeight, allow, kwargs)
    try:
        yield
    finally:
        # 对 args 中的 ExpandedWeight 对象应用 reset 函数
        tree_map_only(ExpandedWeight, reset, args)
        # 对 kwargs 中的 ExpandedWeight 对象应用 reset 函数
        tree_map_only(ExpandedWeight, reset, kwargs)


# setup_rnn 函数用于设置 RNN 操作的上下文环境
@contextmanager
def setup_rnn(use_input_variant, args, kwargs):
    # 如果 use_input_variant 为 True，则使用 batch_second 上下文管理器，否则使用 allow_smaller_batches 上下文管理器
    with batch_second(args, kwargs) if use_input_variant else allow_smaller_batches(
        args, kwargs
    ):
        yield


# implements_per_sample_grads 函数用于装饰支持 __torch_function__ 的函数
def implements_per_sample_grads(torch_function):
    @functools.wraps(torch_function)
    def decorator(autograd_func):
        # 将 torch_function 映射到 autograd_func，存储在 HANDLED_FUNCTIONS 字典中
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func

    return decorator


# ExpandedWeight 表示具有扩展批次维度的权重张量
# ExpandedWeight 的操作与没有扩展批次维度的张量完全相同，但调用 .backward() 方法时，会将每个样本的梯度填充到原始张量的 grad_sample 字段中
#
# ExpandedWeight 有一个始终失败的后备，因为我们无法知道批次的大小
# dimension of the input tensor is and therefore cannot know if this is a valid call
#
# This is a __torch_function__ object but it could have also been a Tensor Extension
# with a dispatch key.
#
# Needs to be a tensor subclass to allow reparameterization
#
# 定义了一个名为 ExpandedWeight 的新类，继承自 torch.Tensor
class ExpandedWeight(torch.Tensor):
    def __init__(self, orig_weight, batch_size, loss_reduction):
        # 初始化方法，设置对象的初始属性
        self.batch_size = batch_size
        self.batch_first = True
        self.allow_smaller_batches = False
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction

    # 定义类属性 handled_functions，用于处理的函数集合
    handled_functions = HANDLED_FUNCTIONS

    # 定义类方法 __new__，用于创建新的 ExpandedWeight 实例
    @classmethod
    def __new__(cls, orig_weight, batch_size, loss_reduction):
        # 如果 orig_weight 不是 torch.Tensor 类型，则抛出异常
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(
                f"Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}"
            )
        # 如果 orig_weight 不需要梯度，则抛出异常
        if not orig_weight.requires_grad:
            raise RuntimeError(
                "Can only build ExpandedWeights objects of tensors that require_grad"
            )
        # 调用父类方法 _make_subclass 创建子类实例
        ret = torch.Tensor._make_subclass(cls, orig_weight, True)
        return ret

    # 定义类方法 __torch_function__，处理 Torch 函数的调用
    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # 如果 func 是 expanded_weights_rnn_decomps 中的函数
        if func in expanded_weights_rnn_decomps:
            # 确定使用的变体是输入还是数据变体
            decomp_opts = expanded_weights_rnn_decomps[func]
            use_input_variant = isinstance(
                args[2], list
            )  # 数据变体在这里使用列表
            decomp = decomp_opts[0] if use_input_variant else decomp_opts[1]

            # 如果存在可用的分解函数，则在设置好 RNN 后调用它
            if decomp is not None:
                with setup_rnn(use_input_variant, args, kwargs):
                    return decomp(*args, **kwargs)
        # 如果 func 是 torch._cudnn_rnn_flatten_weight，则返回空
        if func == torch._cudnn_rnn_flatten_weight:
            # 因为我们不使用融合的 CUDA 内核进行 RNN，因此不执行任何操作
            return
        # 如果 func 在 handled_functions 中，则调用其对应的处理函数
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(
                tuple(kwargs.keys()), func, *(args + tuple(kwargs.values()))
            )
        # 对于任何常规张量输入，我们无法确定批处理维度，因此无法使用回退方案
        raise RuntimeError(
            f"Expanded Weights encountered but cannot handle function {func.__name__}"
        )

    # 定义属性 dtype，返回原始张量的数据类型
    @property
    def dtype(self):
        return self.orig_weight.dtype

    # 定义属性 data，返回原始张量的数据
    @property
    def data(self):
        return self.orig_weight.data

    # 定义属性 shape，返回原始张量的形状
    @property
    def shape(self):
        return self.orig_weight.shape

    # 定义属性 device，返回原始张量所在的设备
    @property
    def device(self):
        return self.orig_weight.device

    # 定义属性 is_cuda，返回原始张量是否在 CUDA 上
    @property
    def is_cuda(self):
        return self.orig_weight.is_cuda

    # 定义方法 data_ptr，返回原始张量数据的指针
    def data_ptr(self):
        return self.orig_weight.data_ptr()

    # 定义方法 get_device，返回原始张量所在的设备索引
    def get_device(self):
        return self.orig_weight.get_device()
    # 设置是否允许使用更小的批次大小的方法
    def set_allow_smaller_batches(self, is_allow_smaller_batches):
        # 将输入的是否允许更小批次大小的标志设置给对象的属性
        self.allow_smaller_batches = is_allow_smaller_batches

    # 设置是否将批次数据的第一个维度作为批次大小的方法，默认为True
    def set_batch_first(self, is_batch_first=True):
        # 将输入的是否将批次数据的第一个维度作为批次大小的标志设置给对象的属性
        self.batch_first = is_batch_first
```