# `.\pytorch\torch\testing\_internal\distributed\distributed_test.py`

```
# 忽略类型检查错误，针对 mypy 工具的设置
# 导入标准库模块
import copy  # 导入深拷贝操作
import json  # 导入 JSON 处理模块
import itertools  # 导入迭代工具模块
import math  # 导入数学函数模块
import os  # 导入操作系统接口模块
import random  # 导入随机数生成模块
import sys  # 导入系统相关模块
import tempfile  # 导入临时文件和目录创建模块
import time  # 导入时间处理模块

# 导入标准库中的数据结构和上下文管理器
from collections import namedtuple, OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass  # 导入数据类支持
from datetime import timedelta  # 导入时间间隔处理类
from functools import reduce  # 导入函数工具模块中的 reduce 函数
from typing import Union, NamedTuple, Callable, Any  # 导入类型提示相关

# 导入单元测试框架
import unittest

# 导入科学计算库和深度学习框架
import numpy as np  # 导入 NumPy 数学计算库
import torch  # 导入 PyTorch 深度学习框架
import torch.cuda  # 导入 PyTorch CUDA 模块
import torch.distributed as dist  # 导入 PyTorch 分布式模块
import torch.distributed.algorithms.model_averaging.averagers as averagers  # 导入模型平均算法实现
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD  # 导入分层模型平均算法
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils  # 导入模型平均工具函数
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 神经网络函数模块

# 导入 PyTorch 内部工具
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR  # 导入测试主地址
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT  # 导入测试主端口
from torch.utils._python_dispatch import TorchDispatchMode  # 导入分发模式定义
from torch.autograd import DeviceType  # 导入设备类型管理
from torch.cuda.amp import GradScaler, autocast  # 导入梯度缩放和自动混合精度支持

# 导入分布式通信钩子和优化器应用函数
from torch.distributed.ddp_comm_hooks import (
    post_localSGD_hook as post_localSGD,
    powerSGD_hook as powerSGD,
    default_hooks as default,
    quantization as quantization_hooks,
)

# 导入分布式 C10D 模块相关函数
from torch.distributed.distributed_c10d import (
    get_world_size,
    _get_default_group,
    _get_pg_config,
)

# 导入分布式工具函数
from torch.distributed.utils import (
    _verify_param_shape_across_processes,
    _sync_module_states,
)

# 导入性能分析相关模块
from torch.profiler import (
    ExecutionTraceObserver,
    ProfilerActivity,
)

# 导入分布式数据并行支持模块
from torch.nn.parallel import DistributedDataParallel

# 导入分布式深度学习相关函数和环境变量工具
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision

# 导入分布式测试相关工具和测试用例
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    init_multigpu_helper,
    initialize_temp_directories,
    cleanup_temp_dir,
    simple_sparse_reduce_tests,
    skip_if_rocm,
    skip_if_small_worldsize,
    skip_if_odd_worldsize,
    skip_if_lt_x_gpu,
    nccl_skip_if_lt_x_gpu,
    skip_if_no_gpu,
    require_n_gpus_for_nccl_backend,
    requires_nccl_version,
    captured_output,
    with_nccl_blocking_wait,
    with_dist_debug_levels,
    verify_ddp_error_logged,
    DistTestCases,
)

# 导入通用测试工具函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_WINDOWS,
    FILE_SCHEMA,
    IS_FBCODE,
    NO_MULTIPROCESSING_SPAWN,
    IS_SANDCASTLE,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
)

# 导入分布式优化器
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer

# 导入分布式数据集采样器
from torch.utils.data.distributed import DistributedSampler

# 导入运算符模块
import operator

# 尝试导入 torchvision 库，标记是否可用
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 根据操作系统平台导入相应模块
if sys.platform == "win32":
    import msvcrt  # 导入 Windows 平台专用模块
else:
    import fcntl  # 导入非 Windows 平台专用模块

# 定义神经网络模块类 NetWithBuffers
class NetWithBuffers(nn.Module):
    # 初始化函数，用于创建对象实例时的初始化操作
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为10，输出维度为10，不使用偏置
        self.a = nn.Linear(10, 10, bias=False)
        # 创建另一个线性层，输入维度为10，输出维度为1，不使用偏置
        self.b = nn.Linear(10, 1, bias=False)
        # 注册一个缓冲区，其中存储一个1x2的张量，内容为随机数
        self.register_buffer("buffer", torch.randn(1, 2))

    # 前向传播函数，定义了模型的数据流向
    def forward(self, x):
        # 在缓冲区的内容上加1，使用了in-place操作
        self.buffer.add_(1)
        # 对输入x先经过self.a线性层，再经过self.b线性层，得到输出
        return self.b(self.a(x))
class Foo:
    # 定义一个名为 Foo 的类
    def __init__(self, x):
        # 初始化方法，接受一个参数 x，可以是张量或整数
        self.x = x

    def __eq__(self, other):
        # 自定义相等性比较方法，用于比较两个 Foo 实例是否相等
        def eq(value, other):
            # 内部函数：判断两个值是否相等，如果是张量则使用 torch.equal 进行比较
            if isinstance(value, torch.Tensor):
                return torch.equal(value, other)
            return value == other

        # 遍历当前实例和另一个实例的所有属性
        for attr, value in self.__dict__.items():
            other_value = other.__dict__[attr]
            # 如果任何属性值不相等，则返回 False
            if not eq(value, other_value):
                return False
        # 如果所有属性值相等，则返回 True
        return True


# 创建一个 Foo 类的实例 f，初始化参数为 10
f = Foo(10)
# 向实例 f 动态添加一个名为 bar 的属性，并赋值为 1
f.bar = 1

# 创建一个张量类型的 Foo 类的实例 foo_cpu_tensor
foo_cpu_tensor = Foo(torch.randn(3, 3))

# 定义一个包含多种对象的列表
COLLECTIVES_OBJECT_TEST_LIST = [
    {"key1": 3, "key2": 4, "key3": {"nested": True}},
    f,
    foo_cpu_tensor,
    "foo",
    [1, 2, True, "string", [4, 5, "nested"]],
]

# 定义支持性能分析的分布式后端白名单
PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.NCCL,
    dist.Backend.GLOO,
    dist.Backend.MPI,
    dist.Backend.UCC,
]

# 定义在 use_cuda=True 情况下支持性能分析的分布式后端白名单
CUDA_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.GLOO,
    dist.Backend.MPI,
    dist.Backend.NCCL,
    dist.Backend.UCC,
]

# 定义支持点对点操作性能分析的分布式后端白名单
SEND_RECV_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.MPI,
    dist.Backend.GLOO,
    dist.Backend.NCCL,
    dist.Backend.UCC,
]

# 定义用于测试 DDP 对 NamedTuple 类型支持的虚拟 NamedTuple 数据结构
EXPECTED_FIELDS = ("a", "b")
TestNamedTupleInput_0 = namedtuple("NamedTuple", EXPECTED_FIELDS)


class TestNamedTupleInput_1(NamedTuple):
    # 定义一个 NamedTuple 类型 TestNamedTupleInput_1，包含两个属性 a 和 b，类型为张量
    a: torch.tensor
    b: torch.tensor


# 根据是否有 torchvision，决定是否跳过测试
skipIfNoTorchVision = skip_but_pass_in_sandcastle_if(
    not HAS_TORCHVISION, "no torchvision"
)

# 从环境变量获取分布式训练的后端类型
BACKEND = os.environ["BACKEND"]
# 从环境变量获取初始化方法，默认为 "env://"
INIT_METHOD = os.getenv("INIT_METHOD", "env://")

# 默认的超时时间为 300 秒
DEFAULT_TIMEOUT = 300
# 自定义的超时时间字典，针对不同的测试用例设置不同的超时时间
CUSTOMIZED_TIMEOUT = {"test_DistributedDataParallel": 500}


def get_profiling_event(event_name, profiler, dedup_gpu_user_annotation=False):
    # 获取指定事件名称的性能分析事件列表
    event_list = (
        profiler.events()
        if isinstance(profiler, torch.profiler.profile)
        else profiler.function_events
    )
    return [
        event for event in event_list
        if (
            (event.name.endswith(event_name) or event.name.startswith(event_name))
            and (not dedup_gpu_user_annotation or event.device_type != DeviceType.CUDA)
        )
    ]


def get_profiler_nccl_meta(prof):
    """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
    We will need to test metadata obtained from profiler here"""
    # 获取 Torch 分析器中插入的名为 "record_param_comms" 的 nccl 元数据
    tf = tempfile.NamedTemporaryFile(
        mode="w+t", suffix=".json", delete=False
    )
    tf.close()
    trace_file = tf.name

    # 导出分析结果到 Chrome 追踪文件中
    prof.export_chrome_trace(trace_file)
    with open(trace_file) as f:
        events = json.load(f)["traceEvents"]
    print(f"Trace saved to {trace_file}")

    # 删除临时文件，用于调试
    os.remove(trace_file)

    return [e for e in events if e.get("name") == "record_param_comms"]
# 在未完成的缩减中基本错误消息子字符串
ddp_prev_reduction_unfinished_str = (
    "Expected to have finished reduction in the prior iteration"
)
# 当未传递 find_unused_parameters=True 关键字参数时的错误消息子字符串
ddp_recommend_find_unused_params_str = (
    "passing the keyword argument `find_unused_parameters=True`"
)
# 当启用 find_unused_parameters=True 时的错误消息子字符串
ddp_find_unused_params_enabled_str = "Since `find_unused_parameters=True` is enabled"
# 指示可能没有所有模型输出用于损失计算的错误消息子字符串
ddp_outputs_not_used_in_loss_str = (
    "`forward` function outputs participate in calculating loss"
)
# 建议使用 TORCH_DISTRIBUTED_DEBUG 的错误消息子字符串
ddp_suggest_debug_mode_str = (
    "set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL"
)

# 定义名为 DDPUnevenTestInput 的命名元组类，包含多个字段用于测试不均匀输入
class DDPUnevenTestInput(NamedTuple):
    name: str                    # 输入的名称
    model: nn.Module             # 模型对象
    inp: Union[torch.tensor, tuple]  # 输入数据，可以是张量或元组
    sync_interval: int           # 同步间隔
    throw_on_early_termination: bool = False  # 是否在早期终止时抛出异常，默认为 False
    hook: Callable = None        # 回调函数，可选
    state: Any = None            # 状态信息，可选

# 定义名为 _FC2 的神经网络模块类，继承自 nn.Module
class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)   # 定义具有偏置的线性层
        self.fc.bias.requires_grad = False       # 设置偏置参数不可训练

    def forward(self, x):
        x = self.fc(x)                          # 前向传播：线性层计算
        return x

# 定义名为 Net 的神经网络模块类，继承自 nn.Module
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)  # 定义无偏置的线性层
        self.fc2 = _FC2()                       # 使用 _FC2 类的实例作为一个模块
        self.fc3 = nn.Linear(50, 4, bias=False) # 定义无偏置的线性层
        self.relu = nn.ReLU()                   # 定义 ReLU 激活函数
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        )                                      # 定义不可训练的参数

    def forward(self, x):
        x = self.relu(self.fc1(x))              # 前向传播：第一个线性层后接 ReLU 激活函数
        x = self.relu(self.fc2(x))              # 前向传播：第二个线性层后接 ReLU 激活函数
        x = self.fc3(x)                         # 前向传播：第三个线性层
        return F.softmax(x, dim=1)              # 对最后一层输出进行 softmax 处理

# 定义名为 LargeNet 的神经网络模块类，继承自 nn.Module
class LargeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 2000, bias=False)  # 定义大规模线性层1
        self.fc2 = nn.Linear(2000, 500, bias=False)   # 定义大规模线性层2

    def forward(self, x):
        x = self.fc1(x)                     # 前向传播：第一个大规模线性层
        x = self.fc2(x)                     # 前向传播：第二个大规模线性层
        return x

# 定义名为 Task 的神经网络模块类，继承自 nn.Module
class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))   # 定义可训练参数 p

    def forward(self, x):
        return self.p + x                       # 前向传播：返回 p 与输入 x 的和

# 定义名为 BatchNormNet 的神经网络模块类，继承自 nn.Module
class BatchNormNet(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.fc1 = nn.Linear(2, 40, bias=False)  # 定义无偏置的线性层
        self.bn = nn.BatchNorm1d(4, affine=affine)  # 定义批归一化层
        self.fc2 = nn.Linear(40, 4, bias=False)  # 定义无偏置的线性层

    def forward(self, x):
        x = torch.reshape(self.fc1(x), (-1, 4, 10))  # 前向传播：线性层后重塑张量形状
        x = self.bn(x)                          # 前向传播：批归一化层
        x = torch.reshape(x, (-1, 40))           # 再次重塑张量形状
        x = self.fc2(x)                         # 前向传播：第二个线性层
        return F.softmax(x, dim=1)              # 对最后一层输出进行 softmax 处理

# 定义名为 UnusedParamTwoLinLayerNet 的神经网络模块类，继承自 nn.Module
class UnusedParamTwoLinLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)  # 定义无偏置的线性层 a
        self.b = nn.Linear(10, 10, bias=False)  # 定义无偏置的线性层 b
        self.c = nn.Linear(5, 5, bias=False)    # 定义无偏置的线性层 c
    # 定义一个前向传播方法，接受输入张量 x
    def forward(self, x):
        # 调用类中的函数 a 对输入 x 进行处理，得到结果 a
        a = self.a(x)
        # 调用类中的函数 b 对输入 x 进行处理，得到结果 b
        b = self.b(x)
        # 返回元组 (a, b)，包含处理后的结果 a 和 b
        return (a, b)
class DictOutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 实例化一个 UnusedParamTwoLinLayerNet 类的对象，并赋值给 self.module
        self.module = UnusedParamTwoLinLayerNet()

    def forward(self, x):
        # 将输入 x 传递给 self.module，得到预测值 predictions
        predictions = self.module(x)
        # 计算两个预测值的和，并求和得到损失值 loss
        loss = (predictions[0] + predictions[1]).sum()
        # 返回预测值和损失值的字典
        return {
            "predictions": predictions,
            "loss": loss,
        }


class TwoLinLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个线性层，输入维度和输出维度均为 10，没有偏置
        self.a = nn.Linear(10, 10, bias=False)
        # 创建另一个线性层，输入维度为 10，输出维度为 1，没有偏置
        self.b = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        # 将输入 x 分别传递给两个线性层，得到输出 a 和 b
        a = self.a(x)
        b = self.b(x)
        # 返回两个输出作为元组
        return (a, b)


class EmbeddingNetDifferentParams(nn.Module):
    """
    A module containing an embedding with different dimension or different # of
    parameters depending on the rank.
    """

    def __init__(self, rank, diff_num_params=False):
        super().__init__()
        # 根据参数 rank 和 diff_num_params 决定 embedding 的维度
        embedding_dim = 500 if diff_num_params or rank == 0 else 50
        # 创建一个嵌入层，嵌入维度根据上一步的结果确定
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        # 创建一个线性层，输入维度为 embedding_dim，输出维度为 1
        self.lin = nn.Linear(embedding_dim, 1)
        if diff_num_params:
            # 如果 diff_num_params 为 True，创建另一个线性层，输入和输出维度均为 1，没有偏置
            self.lin2 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        # 将输入 x 传递给嵌入层，得到输出
        x = self.embedding(x)
        # 将嵌入后的结果传递给线性层，得到最终输出
        return self.lin(x)


class ControlFlowToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个线性层，输入和输出维度均为 10，没有偏置
        self.lin1 = nn.Linear(10, 10, bias=False)
        self.lin2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        # 根据输入 x 决定是否使用第二个线性层
        use_second_layer = torch.equal(x, torch.ones(20, 10, device=x.device))
        if use_second_layer:
            # 如果 use_second_layer 为 True，先传递 x 给第一个线性层，再传递给第二个线性层，并应用 ReLU 激活函数
            return self.lin2(F.relu(self.lin1(x)))
        else:
            # 否则，只传递 x 给第一个线性层，并应用 ReLU 激活函数
            return F.relu(self.lin1(x))


DDP_NET = Net()  # 未提供 Net 类的定义，无法注释该行
BN_NET = BatchNormNet()  # 未提供 BatchNormNet 类的定义，无法注释该行
BN_NET_NO_AFFINE = BatchNormNet(affine=False)  # 未提供 BatchNormNet 类的定义，无法注释该行
ONLY_SBN_NET = nn.SyncBatchNorm(2, momentum=0.99)  # 创建一个具有同步批量归一化的对象


def get_timeout(test_id):
    # 从 test_id 中提取测试名
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        # 如果测试名在 CUSTOMIZED_TIMEOUT 中，返回其对应的超时时间
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        # 否则，返回默认超时时间
        return DEFAULT_TIMEOUT


default_pg_timeout = 60

CUSTOM_PG_TIMEOUT = {
    # "test_ddp_uneven_inputs" 测试的自定义超时时间为 300 秒
    "test_ddp_uneven_inputs": 300,
    # "test_ddp_model_diff_across_ranks" 和 "test_ddp_has_finalized" 测试的自定义超时时间均为 5 秒
    "test_ddp_model_diff_across_ranks": 5,
    "test_ddp_has_finalized": 5,
}

def require_backend_is_available(backends):
    # 根据 backends 参数确定所需的后端是否可用
    # 定义一个函数用于检查指定的分布式后端是否可用
    def check(backend):
        # 检查后端是否为 GLOO，返回 GLOO 是否可用的布尔值
        if backend == dist.Backend.GLOO:
            return dist.is_gloo_available()
        # 检查后端是否为 NCCL，返回 NCCL 是否可用的布尔值
        if backend == dist.Backend.NCCL:
            return dist.is_nccl_available()
        # 检查后端是否为 MPI，返回 MPI 是否可用的布尔值
        if backend == dist.Backend.MPI:
            return dist.is_mpi_available()
        # 检查后端是否为 UCC，返回 UCC 是否可用的布尔值
        if backend == dist.Backend.UCC:
            return dist.is_ucc_available()
        # 检查后端是否在 DistTestCases.backend_feature["plugin"] 中，返回 True
        if backend in DistTestCases.backend_feature["plugin"]:
            return True
        # 若都不是以上情况，则返回 False
        return False

    # 如果指定的 BACKEND 不在允许的后端列表 backends 中，则跳过测试但在 sandcastle 中标记通过
    if BACKEND not in backends:
        return skip_but_pass_in_sandcastle(
            f"Test requires backend {BACKEND} to be one of {backends}"
        )

    # 检查指定的 BACKEND 是否可用，若不可用，则跳过测试但在 sandcastle 中标记通过
    if not check(dist.Backend(BACKEND)):
        return skip_but_pass_in_sandcastle(
            f"Test requires backend {BACKEND} to be available"
        )
    
    # 如果指定的 BACKEND 可用，则返回一个 lambda 函数，该函数参数为要执行的测试函数
    return lambda func: func
# 要求设定全局大小，检查环境变量中的 WORLD_SIZE 是否小于所需大小
def require_world_size(world_size):
    if int(os.environ["WORLD_SIZE"]) < world_size:
        # 如果条件不满足，调用特定函数并返回，用于在特定环境中跳过测试
        return skip_but_pass_in_sandcastle(
            "Test requires world size of %d" % world_size
        )
    # 条件满足时返回一个 lambda 函数，用于修饰测试函数
    return lambda func: func


# 上下文管理器，用于管理临时文件锁
@contextmanager
def _lock():
    # 从环境变量中获取临时目录路径
    TEMP_DIR = os.environ["TEMP_DIR"]
    # 构建锁文件路径
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    # 打开锁文件
    with open(lockfile, "w") as lf:
        try:
            # 根据操作系统类型执行相应的文件锁定操作
            if sys.platform == "win32":
                msvcrt.locking(lf.fileno(), msvcrt.LK_RLCK, 1)
                yield
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            # 最终释放文件锁
            if sys.platform == "win32":
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


# 上下文管理器，用于处理临时文件名的广播和清理
@contextmanager
def _rank_temp_file():
    # 如果当前进程的分布式计算中的排名为 0
    if dist.get_rank() == 0:
        # 创建临时文件并获取文件名
        fd, name = tempfile.mkstemp()
        os.close(fd)
    else:
        # 否则设置文件名为 None
        name = None
    # 构建对象列表，包含临时文件名
    object_list = [name]
    # 在分布式环境中广播对象列表
    dist.broadcast_object_list(object_list)
    # 获取广播后的文件名
    name = object_list[0]
    try:
        # 返回临时文件名，供使用，使用完毕后清理文件
        yield name
    finally:
        # 如果当前进程的排名为 0，则移除临时文件
        if dist.get_rank() == 0:
            os.remove(name)


# 创建一个指定大小和数值的张量，可以选择性地指定数据类型和计算设备
def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        # 在 CPU 上创建张量并填充指定值
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        # 在指定 GPU 设备上创建张量并填充指定值
        return torch.empty(size, size, size, dtype=dtype).fill_(value).cuda(device_id)


# 创建一个多维张量，可以选择性地指定维数、维度大小、数值和数据类型
def _build_multidim_tensor(dim, dim_size, value=None, dtype=torch.float):
    if value is None:
        value = dim
    # 创建指定维度大小和数据类型的多维张量，并填充指定值
    return torch.empty(size=[dim_size for _ in range(dim)], dtype=dtype).fill_(value)


# 创建一个自动微分分析器，用于分析 PyTorch 自动微分操作
def _create_autograd_profiler():
    return torch.autograd.profiler.profile(record_shapes=True)


# 创建一个 Torch 性能分析器，用于分析指定活动的性能
def _create_torch_profiler():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
    )


# 类：屏障，用于同步多个进程的执行
class Barrier:
    barrier_id = 0

    @classmethod
    def init(cls):
        # 初始化屏障，清理临时目录中的所有文件
        cls.barrier_id = 0
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))
    # 同步方法，用于多进程间的同步操作
    def sync(cls, wait_for=None, timeout=10):
        # 如果未指定等待数，使用当前进程组的进程数
        if wait_for is None:
            wait_for = dist.get_world_size()
        # 增加屏障ID以标识新的同步点
        cls.barrier_id += 1
        # 设置屏障文件夹路径
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        # 获取当前进程的PID，并构建对应的屏障文件路径
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        # 使用线程锁确保写入操作的原子性
        with _lock():
            # 在屏障文件中写入当前的屏障ID
            with open(barrier_file, "w") as f:
                f.write(str(cls.barrier_id))

        # 记录同步开始时间
        start_time = time.time()
        while True:
            arrived = 0
            # 使用线程锁确保读取操作的原子性
            with _lock():
                # 遍历屏障文件夹中的文件
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name)) as f:
                        data = f.read()
                        # 如果文件中的屏障ID大于等于当前屏障ID，表示进程已到达同步点
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            # 如果到达同步点的进程数达到指定的等待数，则退出循环
            if arrived == wait_for:
                break

            # 如果超过了设定的超时时间，抛出超时异常
            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            # 等待一段时间后再次检查
            time.sleep(0.1)
# 定义一个名为 TestDistBackend 的测试类，继承自 MultiProcessTestCase 类
class TestDistBackend(MultiProcessTestCase):
    
    @classmethod
    def setUpClass(cls):
        # 设置环境变量 MASTER_ADDR 为指定的 MASTER_ADDR 值（假设在全局已定义）
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        # 不设置 MASTER_PORT，自动获取一个空闲端口
        super().setUpClass()

    def setUp(self):
        # 调用父类的 setUp 方法进行初始化
        super().setUp()
        # 初始化临时目录
        initialize_temp_directories()
        # 初始化 Barrier
        Barrier.init()
        # 设置跳过返回码检查的测试列表，因为它们可能由于 TORCH_NCCL_ASYNC_ERROR_HANDLING 而导致进程崩溃
        self.skip_return_code_checks = [self.test_ddp_has_finalized.__wrapped__]

    def tearDown(self):
        # 清理临时目录
        cleanup_temp_dir()
        # 调用父类的 tearDown 方法进行清理
        super().tearDown()

    @property
    def init_method(self):
        # 返回格式化的文件初始化方法，使用 FILE_SCHEMA 和当前实例的 file_name 属性
        return f"{FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        # 如果使用的后端是 "nccl" 且 CUDA 可用性为假，则退出进程
        if BACKEND == "nccl" and not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        # 创建当前类的实例，初始化测试名称、进程排名和文件名
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        # 如果 CUDA 可用且 CUDA 设备数小于当前实例的 world_size 属性值，则退出进程
        if torch.cuda.is_available() and torch.cuda.device_count() < int(
            self.world_size
        ):
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        
        try:
            # 根据测试名称获取自定义的进程组超时时间，如果不存在则使用默认值
            pg_timeout_seconds = CUSTOM_PG_TIMEOUT.get(test_name, default_pg_timeout)
            timeout = timedelta(seconds=pg_timeout_seconds)
            # 初始化进程组
            dist.init_process_group(
                init_method=self.init_method,
                backend=BACKEND,
                world_size=int(self.world_size),
                rank=self.rank,
                timeout=timeout,
            )
        except RuntimeError as e:
            # 捕获 RuntimeError 异常，如果异常信息中包含 "recompile" 字符串，则退出进程
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise
        
        # 执行 Barrier 操作，确保每个进程都完成初始化，以避免后续测试因跳过而导致不稳定
        self._barrier()
        
        # 运行指定的测试方法，并将输出写入管道 pipe
        self.run_test(test_name, pipe)
        
        # 再次执行 Barrier 操作，确保所有进程都完成测试运行
        self._barrier()
        
        # 销毁进程组
        dist.destroy_process_group()
        
        # 正常退出进程
        sys.exit(0)

    # MultiProcessTestCase 假定 world_size 为 4，但我们可能在不同的 world_size 下运行这些测试，因此需要重新定义该属性
    @property
    def world_size(self):
        # 返回环境变量 WORLD_SIZE 的值
        return os.environ["WORLD_SIZE"]
```