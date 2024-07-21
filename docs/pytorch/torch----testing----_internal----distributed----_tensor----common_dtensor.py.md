# `.\pytorch\torch\testing\_internal\distributed\_tensor\common_dtensor.py`

```
# mypy: ignore-errors  # 忽略类型检查错误
# Copyright (c) Meta Platforms, Inc. and affiliates  # 版权声明

import itertools  # 导入 itertools 库，用于迭代操作
import sys  # 导入 sys 库，提供系统相关功能
from dataclasses import dataclass  # 导入 dataclass 用于创建数据类
from functools import wraps  # 导入 wraps 用于装饰器相关功能
from typing import Any, Callable, cast, Dict, Iterator, List, Sequence, Tuple, TypeVar  # 导入类型提示相关库

import torch  # 导入 PyTorch 深度学习库
import torch.distributed as dist  # 导入分布式训练相关模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络中的常用函数

from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard  # 导入分布式张量相关模块
from torch.distributed._tensor.placement_types import Placement  # 导入分布式张量的放置类型
from torch.distributed.tensor.parallel import (  # 导入分布式张量并行计算相关模块
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import (  # 导入分布式测试相关模块
    MultiProcessTestCase,
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
    run_subtests,
    TEST_SKIPS,
)

from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec  # 导入 PyTree 相关函数

DEVICE_TYPE = (  # 设置设备类型，如果有 CUDA 设备且数量大于 1，则为 "cuda"，否则为 "cpu"
    "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
)

NUM_DEVICES = 4  # 指定设备数量为 4

# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # 当存在多个 GPU 时，将 NUM_DEVICES 限制在实际 GPU 数量内
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

T = TypeVar("T")  # 定义泛型变量 T

# simple RMSNorm layer for testing
class RMSNormPython(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 初始化 epsilon
        self.weight = torch.nn.Parameter(torch.ones(dim))  # 初始化权重参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算 RMSNorm

    def forward(self, x):
        output = self._norm(x)  # 前向传播计算 RMSNorm
        return output * self.weight  # 返回加权后的输出


class MLPModule(nn.Module):
    def __init__(self, device, bias: bool = True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)  # 第一个线性层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)  # 第二个线性层

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))  # 神经网络前向传播

    def reset_parameters(self):
        self.net1.reset_parameters()  # 重置第一个线性层参数
        self.net2.reset_parameters()  # 重置第二个线性层参数


class MLPStacked(nn.Module):
    def __init__(self, device, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([MLPModule(device) for i in range(n_layers)])  # 创建多层 MLP 模型

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # 逐层进行前向传播
        return x


@dataclass
class ModelArgs:
    n_layers: int = 2  # 默认层数为 2
    vocab_size: int = 8  # 词汇量大小为 8
    max_seq_len: int = 16  # 最大序列长度为 16
    dim: int = 16  # 维度大小为 16
    n_heads: int = 4  # 头部数量为 4
    dropout_p: float = 0.1  # Dropout 概率为 0.1
    use_attn_mask: bool = True  # 是否使用注意力掩码，默认为 True
    weight_tying: bool = True  # 是否进行权重绑定，默认为 True
    checkpoint_activations: bool = False  # 是否记录激活状态，默认为 False


class Attention(nn.Module):
    # 空白，未完成的注意力模块
    pass
    # 初始化函数，接受一个参数 args，其类型为 ModelArgs
    def __init__(self, args: ModelArgs):
        # 调用父类的初始化方法
        super().__init__()
        # 断言确保 args.dim 能被 args.n_heads 整除
        assert args.dim % args.n_heads == 0
        # 计算每个头部的维度
        self.head_dim = args.dim // args.n_heads
        # 设置头部的数量
        self.n_heads = args.n_heads
        # 设置 dropout 概率
        self.dropout_p = args.dropout_p
        # 创建一个 dropout 层，用于残差连接
        self.resid_dropout = nn.Dropout(args.dropout_p)
        # 根据参数决定是否使用注意力掩码
        self.use_attn_mask = args.use_attn_mask

        # 创建线性层，用于查询（query）变换
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        # 创建线性层，用于键（key）变换
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        # 创建线性层，用于值（value）变换
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        # 创建线性层，用于最终输出变换
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 获取输入张量的批大小（batch size）、序列长度和特征维度
        bsz, seq_len, _ = x.size()
        # 使用查询、键、值线性层对输入 x 进行变换
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        # 将变换后的查询、键、值张量重塑为多头注意力的形状
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        # 将重塑后的查询、键、值张量进行维度交换，以便进行多头注意力计算
        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        # 调用 scaled_dot_product_attention 函数进行注意力计算
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,  # 不使用注意力掩码
            self.dropout_p if self.training else 0,  # 根据是否训练决定是否应用 dropout
            self.use_attn_mask,  # 是否使用注意力掩码
        )
        # 将多头注意力计算的输出进行维度交换和重塑，以便进行最终线性变换
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        # 对最终输出应用残差连接中的 dropout
        return self.resid_dropout(self.wo(output))
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        # 定义第一层全连接层，输入维度为dim，输出维度为hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim)
        # GELU 激活函数
        self.gelu = nn.GELU()
        # 定义第二层全连接层，输入维度为hidden_dim，输出维度为dim
        self.w2 = nn.Linear(hidden_dim, dim)
        # 使用dropout进行正则化，丢弃概率为dropout_p
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # 前向传播函数，先通过第一层全连接层和GELU激活函数，再应用第二层全连接层和dropout
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Layer Normalization，标准化输入特征args.dim
        self.attention_norm = nn.LayerNorm(args.dim)
        # 注意力机制
        self.attention = Attention(args)
        # Layer Normalization，标准化输入特征args.dim
        self.ffn_norm = nn.LayerNorm(args.dim)
        # 前馈神经网络模块
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        # TransformerBlock的前向传播，包括注意力机制和前馈神经网络
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 断言确保参数的完整性
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        # token的embedding层，将词汇表大小args.vocab_size映射到维度为args.dim的向量
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # positional embedding层，将序列长度args.max_seq_len映射到维度为args.dim的向量
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        # dropout层，丢弃概率为args.dropout_p
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        # 创建args.n_layers个TransformerBlock层
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        # Layer Normalization，标准化输入特征args.dim
        self.norm = nn.LayerNorm(args.dim)
        # 输出层，将维度为args.dim的向量映射到词汇表大小args.vocab_size的向量
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            # 权重绑定，将输出层的权重与token的embedding层的权重绑定
            self.output.weight = self.tok_embeddings.weight
        self.checkpoint_activations = args.checkpoint_activations

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        # 断言确保序列长度不超过self.max_seq_len
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            if self.checkpoint_activations:
                # 如果启用了激活检查点，则使用checkpoint机制执行layer
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @staticmethod
    def parallelize(
        module: "Transformer", device_mesh: DeviceMesh, use_seq_parallel: bool
    ):
        """
        并行化方法，根据设备网格和是否使用序列并行化来并行化Transformer模块
        """

def skip_unless_torch_gpu(method: T) -> T:
    """
    测试装饰器，除非torch有可用GPU，否则跳过测试

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_gpu
    >>> def test_some_method(self) -> None:
    >>>   ...
    """
    # The builtin @skip_if_no_gpu relies on os.environ['WORLD_SIZE'] being set.
    # 返回一个经过类型转换的值，该值是使用 skip_if_lt_x_gpu(NUM_DEVICES) 函数对 method 参数进行处理后得到的结果。
    return cast(T, skip_if_lt_x_gpu(NUM_DEVICES)(method))
class DTensorTestBase(MultiProcessTestCase):
    # 定义一个测试基类，继承自多进程测试用例基类

    @property
    def world_size(self) -> int:
        # 返回设备数量作为世界大小
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        # 根据设备类型确定后端，CUDA 设备使用 NCCL，其他设备使用 Gloo
        backend = "nccl" if self.device_type == "cuda" else "gloo"
        return backend

    def build_device_mesh(self) -> DeviceMesh:
        # 构建设备网格
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def init_pg(self) -> None:
        # 初始化进程组
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            # 如果是 NCCL 后端并且 CUDA 设备数量小于世界大小，则退出测试
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in ["nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl"]:
            # 如果后端不在支持的列表中，则抛出运行时错误
            raise RuntimeError(f"Backend {self.backend} not supported!")

        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
        )

        # 设置用于 NCCL 进程组的设备以进行集合操作
        if "nccl" in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # 等待所有进程在此处达到，然后开始关闭进程组
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        # 设置测试环境
        super().setUp()
        self._spawn_processes()

    # pyre-ignore[2]:
    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        # 测试操作方法，验证转换器的成功性和输出是否正确
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            # pyre can't find assertTrue anymore?
            self.assertEqual(dtc.successful(), True)
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    def run_subtests(self, *args, **kwargs):
        # 运行子测试
        return run_subtests(self, *args, **kwargs)


TestFunc = Callable[[object], object]


# wrapper to initialize comms (processgroup)
def with_comms(func: TestFunc) -> TestFunc:
    # 初始化通信（进程组）的装饰器函数
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        # 如果没有可用的 CUDA 设备或 CUDA 设备数量小于世界大小，则使用 CPU
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.world_size:
            self.device_type = "cpu"
        else:
            self.device_type = DEVICE_TYPE

        self.init_pg()  # 初始化进程组
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()  # 关闭进程组

    return wrapper
# 这是一个基础的测试类，用于多线程测试
class DTensorOpTestBase(MultiThreadedTestCase):
    
    @property
    def world_size(self) -> int:
        # 返回设备数量，使用常量 NUM_DEVICES
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        # 返回设备类型，使用常量 DEVICE_TYPE
        return DEVICE_TYPE

    def build_device_mesh(self):
        # 构建设备网格对象，使用设备类型和从 0 到设备数量的列表
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def setUp(self) -> None:
        # 调用父类的 setUp 方法
        super().setUp()
        # 启动测试中的多线程
        self._spawn_threads()


# 这是一个用于将操作的参数/关键字参数转换为分布式参数/关键字参数的类
class DTensorConverter:
    def __init__(
        self,
        mesh: DeviceMesh,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        # 初始化命中和未命中计数器
        self.hit = 0
        self.miss = 0
        # 设备网格对象
        self.mesh = mesh
        # 参数元组
        self.args = args
        # 关键字参数字典
        self.kwargs = kwargs
        # 扁平化的参数列表和结构说明
        flatten_args, flatten_args_spec = tree_flatten(args)
        # 扁平化的关键字参数列表和结构说明
        flatten_kwargs, flatten_kwargs_spec = tree_flatten(kwargs)

        # 扁平化后的参数列表
        self.flatten_args: List[object] = flatten_args
        # 扁平化后的参数结构说明
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        # 扁平化后的关键字参数列表
        self.flatten_kwargs: List[object] = flatten_kwargs
        # 扁平化后的关键字参数结构说明
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec

        # 为参数中的 torch.Tensor 类型生成分片选择
        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        # 生成所有参数可能的分片组合的迭代器
        self.sharding_combs: Iterator[Sequence[Placement]] = iter(
            itertools.product(*choices_for_args)
        )

    def successful(self) -> bool:
        # 返回命中次数大于 0 且未命中次数为 0
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        # TODO: 分布式张量需要支持量化和稀疏张量，
        # 量化张量可能相对简单，但稀疏张量具有特殊的布局，我们可能需要处理，
        # 在我们明确这些内容之前，我们不正式支持它们。
        return not any(
            [
                t.is_sparse_csr,        # 是否为 CSR 格式的稀疏张量
                t.is_sparse,            # 是否为稀疏张量
                t.is_mkldnn,            # 是否为 MKL-DNN 张量
                t.is_quantized,         # 是否为量化张量
                t.is_nested,            # 是否为嵌套张量
                torch._is_functional_tensor(t),  # 是否为函数张量
                t.is_neg(),             # 是否为负张量
                t.is_conj(),            # 是否为共轭张量
                t.device.type in ("lazy", "meta"),  # 是否为惰性或元数据设备类型
                # 我们需要一种方法来测试张量是否批处理，但目前没有官方的 API 支持
                # torch._C._is_batched(t),
            ]
        )
    # 为给定的 torch.Tensor 参数生成分片选择的选项
    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        # 获取当前的 Mesh 大小
        mesh_size = self.mesh.size()
        # 默认的分片选择为 Replicate()，即不分片
        sharding_choices: List[Placement] = [Replicate()]
        
        # 如果参数的数据类型不是 bool，则进行以下操作
        # c10d 集体操作不支持布尔张量，对于布尔张量我们视为复制操作
        if arg.dtype != torch.bool:
            # 只生成以下选择：复制（Replicate），或者在可以进行分片的维度上进行分片
            sharding_choices = sharding_choices + [
                Shard(i)
                for i, s in enumerate(arg.shape)
                if s > 1 and s % mesh_size == 0
            ]
        
        # TODO: 添加多网格选择
        
        # 返回生成的分片选择列表
        return sharding_choices

    # 实现迭代器协议方法，返回自身作为迭代器
    def __iter__(self) -> "DTensorConverter":
        return self

    # 实现迭代器协议方法，返回下一个元素
    def __next__(self) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        try:
            # 获取下一个分片组合
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            # 准备存储新参数的列表
            new_args: List[object] = []
            # 遍历扁平化的参数列表
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    # 如果参数是 torch.Tensor，则转换为分布式张量
                    new_args.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_args.append(arg)

            # 准备存储新关键字参数的列表
            new_kwargs: List[object] = []
            # 遍历扁平化的关键字参数列表
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    # 如果参数是 torch.Tensor，则转换为分布式张量
                    new_kwargs.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_kwargs.append(arg)

            # 返回重新构造的原始参数和关键字参数的元组
            return (
                tree_unflatten(new_args, self.flatten_args_spec),
                tree_unflatten(new_kwargs, self.flatten_kwargs_spec),
            )
        except StopIteration as e:
            # 捕获 StopIteration 异常并重新抛出
            raise StopIteration from e

    # 将给定的 torch.Tensor 转换为分布式张量的方法
    def to_dist_tensor(
        self, t: torch.Tensor, mesh: DeviceMesh, placements: List[Placement]
    ) -> torch.Tensor:
        # 如果输入参数 t 是 torch.Tensor 或者 nn.Parameter 类型
        if type(t) is torch.Tensor or type(t) is nn.Parameter:
            # 检查该类型的张量是否受支持
            if self.is_supported_tensor(t):
                # 增加 hit 计数器，表示成功处理的张量数加一
                self.hit += 1
                # 如果张量是零维的标量张量，默认会复制到所有的处理单元上
                if t.ndim == 0:
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    # 对于非标量张量，根据 placements 分布到 mesh 上
                    r = distribute_tensor(t, mesh, placements)
                # 如果输入是 nn.Parameter 类型，将分布后的张量再封装成 nn.Parameter
                if type(t) is nn.Parameter:
                    r = nn.Parameter(  # type: ignore[assignment]
                        r, requires_grad=r.requires_grad
                    )
                return r
            else:
                # 如果输入张量不受支持，增加 miss 计数器，表示未能处理的张量数加一
                self.miss += 1
                # 直接返回原始输入张量
                return t
        elif torch.overrides.is_tensor_like(t):
            # 对于类似张量的对象，强制转换为分布式张量可能导致问题，因此暂时禁止这种转换
            self.miss += 1
            return t
        else:
            # 如果输入既不是 torch.Tensor 也不是类似张量对象，抛出错误
            raise RuntimeError(f"Trying to convert to DTensor, but got {type(t)}")
```