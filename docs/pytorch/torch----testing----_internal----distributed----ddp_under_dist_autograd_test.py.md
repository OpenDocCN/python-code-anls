# `.\pytorch\torch\testing\_internal\distributed\ddp_under_dist_autograd_test.py`

```py
# 忽略类型检查错误，这通常用于类型检查工具（如mypy）忽略该文件中的类型错误
# 导入标准库模块
import contextlib           # 提供常见的上下文管理器
import enum                 # 枚举类型支持
import logging              # 日志记录模块
import os                   # 系统操作模块
import threading            # 线程支持模块
from typing import NamedTuple  # 提供命名元组支持

import torch                # PyTorch主库
import torch.distributed as dist  # PyTorch分布式通信库
import torch.distributed.autograd as dist_autograd  # PyTorch分布式自动求导模块
import torch.nn as nn       # PyTorch神经网络模块
from torch.distributed import rpc  # PyTorch分布式RPC支持
from torch.distributed.nn import RemoteModule  # 远程模块支持
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行支持
from torch.testing._internal.common_distributed import (  # 导入测试相关的分布式函数和装饰器
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init  # 测试内部分布式初始化函数
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (  # RPC代理测试固件
    RpcAgentTestFixture,
)


NUM_EM_ROW = 2  # 嵌入矩阵行数
D_SPARSE = 3     # 稀疏特征维度
D_DENSE = 2      # 密集特征维度
D_HID = 3         # 隐藏层维度
D_OUT = 1         # 输出层维度
NUM_TRAINERS = 4  # 训练节点数
WORLD_SIZE = NUM_TRAINERS + 2  # 总的分布式训练节点数
TRAINER_RANKS = list(range(NUM_TRAINERS))  # 训练节点的排名列表
REMOTE_WORKER_RANK = TRAINER_RANKS[-1] + 1  # 远程工作节点的排名
MASTER_RANK = REMOTE_WORKER_RANK + 1  # 主节点的排名


class DdpMode(enum.Enum):
    # 不应用分布式数据并行
    NONE = enum.auto()
    # 对顶层nn.Module应用分布式数据并行
    OUTSIDE = enum.auto()
    # 在顶层nn.Module内部嵌入分布式数据并行
    INSIDE = enum.auto()


def init_logger():
    logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
    level = logging.DEBUG if "debug" in os.environ else logging.INFO  # 根据环境变量设置日志级别
    logger.setLevel(level)  # 设置日志记录器的级别
    console = logging.StreamHandler()  # 创建一个输出到控制台的日志处理器
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )  # 设置日志消息格式
    console.setFormatter(formatter)  # 将格式应用到控制台处理器
    console.setLevel(level)  # 设置控制台处理器的级别
    logger.addHandler(console)  # 将控制台处理器添加到日志记录器
    logger.propagate = False  # 防止日志消息重复输出
    return logger  # 返回配置好的日志记录器


gLogger = init_logger()  # 初始化全局日志记录器


class FeatureSet(NamedTuple):
    """特征集包含两种类型的特征"""

    dense_features: torch.Tensor       # 密集特征张量
    sparse_features: torch.LongTensor  # 稀疏特征张量
    values: torch.Tensor               # 值张量


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)  # 调用指定的方法并传递参数


def _remote_method(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))  # 创建参数元组
    return rpc.rpc_sync(rref.owner(), _call_method, args=args_tup, kwargs=kwargs)  # 同步远程调用RPC方法


def _remote_method_async(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))  # 创建参数元组
    return rpc.rpc_async(rref.owner(), _call_method, args=args_tup, kwargs=kwargs)  # 异步远程调用RPC方法


class RemoteEM(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        gLogger.info("Initing RemoteEM with %s %s", num_embeddings, embedding_dim)  # 记录初始化信息
        super().__init__()  # 调用父类构造函数
        init_em = [0.5] * embedding_dim  # 初始化嵌入向量的初始值
        self.em = nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            _weight=torch.tensor([init_em] * num_embeddings),  # 使用初始嵌入向量初始化权重
        )

    def forward(self, input: torch.Tensor):
        gLogger.debug("Running RemoteEM.forward() on: %s", input)  # 调试级别日志记录输入数据
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))  # 调用嵌入层的前向传播方法
# 返回具有预定义参数的线性模块。
def getLinear(d_in, d_out):
    # 创建一个不带偏置的线性层，输入维度为 d_in，输出维度为 d_out
    l = nn.Linear(d_in, d_out, bias=False)
    # 创建一个全为1的权重张量，形状为 (d_out, d_in)
    w = torch.ones((d_out, d_in))
    # 修改权重张量的第一个元素为 -1，并标记为需要梯度计算
    w[0][0] = -1
    w.requires_grad_()
    # 将修改后的权重张量赋值给线性层的权重数据
    l.weight.data = w
    # 返回构建好的线性层
    return l


class RemoteNet(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        # 输出初始化信息到日志，说明 RemoteNet 的输入和输出维度
        gLogger.info("Initing RemoteNet with %s %s", d_in, d_out)
        super().__init__()
        # 使用 getLinear 函数创建 RemoteNet 的线性层 fc
        self.fc = getLinear(d_in, d_out)
        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        # 输出调试信息到日志，说明在 RemoteNet 上运行 forward 方法
        gLogger.debug("Running RemoteNet.forward() on: %s", input)
        # 返回经过 ReLU 激活的线性层输出结果
        return self.relu(self.fc(input))


class HybridModel(nn.Module):
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        process_group_for_ddp: dist.ProcessGroup = None,
    ):
        super().__init__()
        # 初始化 HybridModel，接收两个远程引用对象
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        # 创建两个本地的线性层：fc1 和 fc2
        self.fc1 = getLinear(D_DENSE, D_DENSE)
        self.fc2 = getLinear(D_HID, D_OUT)

        # 将 fc1 和 fc2 的参数分别存储到 non_ddp_params 和 ddp_params 中
        self.non_ddp_params = tuple(self.fc1.parameters()) + tuple(
            self.fc2.parameters()
        )
        self.ddp_params = ()

        # 如果指定了 DDP 的过程组，将 fc2 应用于 DistributedDataParallel
        if process_group_for_ddp is not None:
            self.non_ddp_params, self.ddp_params = (
                tuple(self.fc1.parameters()),
                tuple(self.fc2.parameters()),
            )
            gLogger.info("Use DDP for the second local net.")
            self.fc2 = DistributedDataParallel(
                self.fc2, check_reduction=True, process_group=process_group_for_ddp
            )

        # 输出初始化信息到日志，说明 HybridModel 中包含的参数组数
        gLogger.info(
            "HybridModel has %s groups of parameters.", len(list(self.parameters()))
        )

    def forward(self, input: FeatureSet):
        # 输出调试信息到日志，说明在 HybridModel 上运行 forward 方法
        gLogger.debug("Running HybridModel.forward on %s", input)
        # 调用 _remote_method 执行远程计算，获取稀疏特征的结果
        sparse = _remote_method(
            RemoteEM.forward, self.remote_em_rref, input.sparse_features
        )
        # 断言稀疏特征和稠密特征的批次大小相同
        assert sparse.shape[0] == input.dense_features.shape[0]
        # 将稠密特征输入到 fc1 中
        dense = self.fc1(input.dense_features)
        # 将稠密特征和稀疏特征拼接在一起
        x = torch.cat((dense, sparse), 1)
        # 输出调试信息到日志，说明拼接后的特征向量
        gLogger.debug("Concatenated feature: %s", x)
        # 调用 _remote_method 执行远程计算，获取 RemoteNet 的输出
        x = _remote_method(RemoteNet.forward, self.remote_net_rref, x)
        # 将输出结果输入到 fc2 中
        return self.fc2(x)


class Trainer:
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        ddp_mode: DdpMode,
        rank: int,
    ):
        # 初始化函数，设置对象的初始属性
        self.rank = rank
        # 根据不同的分布式训练模式，创建训练组，或者设置为 None
        self.trainer_group = (
            dist.new_group(TRAINER_RANKS)
            if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE)
            else None
        )
        # 设置远程推理模型的远程引用
        self.remote_em_rref = remote_em_rref
        # 设置远程网络模型的远程引用
        self.remote_net_rref = remote_net_rref
        # 创建混合模型对象，传入远程推理模型引用、远程网络模型引用和训练组
        self.hybrid_module = HybridModel(
            self.remote_em_rref,
            self.remote_net_rref,
            self.trainer_group if ddp_mode in (DdpMode.INSIDE,) else None,
        )
        # 将混合模型对象的 DDP 参数和非 DDP 参数分配给对象的相应属性
        self.ddp_params, self.non_ddp_params = (
            self.hybrid_module.ddp_params,
            self.hybrid_module.non_ddp_params,
        )
        # 如果 DDP 模式为 OUTSIDE，则将整个混合模型包装成 DDP
        if ddp_mode == DdpMode.OUTSIDE:
            gLogger.info("Wrapping the whole hybrid module into DDP.")
            self.ddp_params += self.non_ddp_params
            self.non_ddp_params = ()
            self.hybrid_module = DistributedDataParallel(
                self.hybrid_module,
                check_reduction=True,
                process_group=self.trainer_group,
            )
        # 记录成功创建 HybridModel 实例的信息，包括 DDP 参数和本地参数的数量
        gLogger.info(
            "Succeeded in creating a HybridModel instance with "
            "%s ddp params and %s other local params.",
            len(self.ddp_params), len(self.non_ddp_params)
        )

    def destroy_pg(self):
        # 销毁训练组的函数
        if self.trainer_group:
            dist.destroy_process_group(self.trainer_group)

    def train_batch(
        self,
        mini_batch: FeatureSet,
        trainer_has_less_inputs: bool,
        simulate_uneven_inputs: bool,
    ):
        grads_dict = None  # 初始化梯度字典为 None

        if not simulate_uneven_inputs:
            input_batches = [mini_batch]  # 如果不模拟不均匀输入，则使用单一的 mini_batch
        else:
            # 拆分为微批次，并修剪以模拟不均匀输入
            dense_features = mini_batch.dense_features
            sparse_features = mini_batch.sparse_features
            values = mini_batch.values

            dense_microbatch = torch.split(dense_features, 2)  # 按照指定维度大小拆分 dense_features
            sparse_microbatch = torch.split(sparse_features, 2)  # 按照指定维度大小拆分 sparse_features
            values_microbatch = torch.split(values, 2)  # 按照指定维度大小拆分 values
            batches = []
            for d, s, v in zip(dense_microbatch, sparse_microbatch, values_microbatch):
                feature_set = FeatureSet(dense_features=d, sparse_features=s, values=v)  # 创建特征集对象
                batches.append(feature_set)  # 将特征集对象添加到 batches 中

            if trainer_has_less_inputs:
                input_batches = batches[: len(batches) // 2]  # 如果训练器输入较少，则只使用前一半的 batches
                gLogger.info(
                    "Trainer reduced input patches from %s "
                    "to %s to simulate uneven inputs.",
                    len(batches), len(input_batches)
                )
            else:
                input_batches = batches  # 否则使用全部的 batches

        # 根据 simulate_uneven_inputs 的条件选择是否加入上下文管理器
        with self.hybrid_module.join() if simulate_uneven_inputs else contextlib.nullcontext():
            for b in input_batches:  # 遍历输入的 batches
                with dist_autograd.context() as context_id:  # 创建分布式自动求导上下文
                    output = self.hybrid_module.forward(b)  # 调用模型的前向传播
                    loss = (output * mini_batch.values).sum()  # 计算损失
                    dist_autograd.backward(context_id, [loss])  # 反向传播计算梯度
                    grads_dict = dist_autograd.get_gradients(context_id)  # 获取梯度字典
                    gLogger.info(
                        "Loss is %s for mini batch: %s. "
                        "Grads dict has %s entries: %s", loss, mini_batch, len(grads_dict), grads_dict
                    )  # 记录日志，显示损失值和梯度字典的信息
        return (
            tuple(grads_dict[param] for param in self.ddp_params),  # 返回分布式数据并行参数的梯度元组
            tuple(grads_dict[param] for param in self.non_ddp_params),  # 返回非分布式数据并行参数的梯度元组
        )
# 定义函数用于获取训练样本集
def get_training_examples():
    # 设置每个训练样本的数量
    n = 16
    # 创建特征集对象，包括稠密特征、稀疏特征和对应值
    training_examples = FeatureSet(
        dense_features=torch.zeros((n, D_DENSE)),  # 创建全零稠密特征张量
        sparse_features=torch.zeros(n, dtype=torch.long),  # 创建全零稀疏特征张量
        values=torch.zeros(n),  # 创建全零值张量
    )
    idx = 0
    # 每个样本有一个完全相同的特征，但值相反的另一个样本，因此它们在全局归约中的梯度会相互抵消
    for value in (-1, 1):
        for x in (-1.0 * value, 1.0 * value):
            for y in (1.0 * value, -1.0 * value):
                for z in (0, 1):
                    # 填充训练样本集的特征和值
                    training_examples.dense_features[idx, :] = torch.tensor((x, y))  # 设置稠密特征
                    training_examples.sparse_features[idx] = z  # 设置稀疏特征
                    training_examples.values[idx] = value  # 设置值
                    idx += 1

    # 将样本分配给 NUM_TRAINERS 个训练器
    assert 0 == (n % NUM_TRAINERS)  # 断言确保样本数能被训练器数量整除
    examples_per_trainer = int(n / NUM_TRAINERS)  # 每个训练器分配的样本数
    return [
        FeatureSet(
            dense_features=training_examples.dense_features[
                start : start + examples_per_trainer, :
            ],  # 切片获取每个训练器对应的稠密特征
            sparse_features=training_examples.sparse_features[
                start : start + examples_per_trainer
            ],  # 切片获取每个训练器对应的稀疏特征
            values=training_examples.values[start : start + examples_per_trainer],  # 切片获取每个训练器对应的值
        )
        for start in range(0, n, examples_per_trainer)  # 遍历每个训练器的起始位置
    ]


# 创建一个线程条件变量用于关闭信号通知
shutdown_signal = threading.Condition()


# 定义设置关闭信号的函数
def set_shutdown_signal():
    global shutdown_signal
    with shutdown_signal:
        shutdown_signal.notify()


# 定义一个测试类，继承自 RpcAgentTestFixture
class DdpUnderDistAutogradTest(RpcAgentTestFixture):
    @property
    def world_size(self) -> int:
        return WORLD_SIZE  # 返回全局变量 WORLD_SIZE 的值

    def remote_worker_name(self) -> str:
        # 返回远程工作进程的名称，与 'dist_init' 装饰器中的名称一致
        return f"worker{REMOTE_WORKER_RANK}"

    def trainer_name(self, rank):
        # 返回训练器的名称，与 'dist_init' 装饰器中的名称一致
        return f"worker{rank}"

    def _remote_worker_process(self, ddp_mode):
        gLogger.info("The remote worker is running.")  # 记录信息：远程工作进程正在运行
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # 使用文件名格式化初始化方法模板
            world_size=self.world_size,  # 设置分布式进程组的全局大小
            rank=self.rank,  # 设置当前进程的排名
        )

        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            # 如果 DDP 模式为 INSIDE 或 OUTSIDE，则需要在训练器排名上调用 new_group
            dist.new_group(TRAINER_RANKS)

        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()  # 等待关闭信号
        gLogger.info("Exiting remote worker.")  # 记录信息：退出远程工作进程
        dist.destroy_process_group()  # 销毁分布式进程组
    # 定义一个方法 `_trainer_process`，用于处理训练器的逻辑，接受一个整数参数 `rank` 作为进程的标识
    def _trainer_process(self, rank: int):
        # 在日志中记录当前训练器的运行信息，包括其标识符 `rank`
        gLogger.info("Running the trainer #%s...", rank)
        # 在日志中记录初始化训练器进程组的信息，包括当前训练器的 `rank` 和所有训练器的排名列表 `TRAINER_RANKS`
        gLogger.info(
            "Initing trainer process group by trainer #%s with ranks %s", rank, TRAINER_RANKS
        )
        # 初始化分布式进程组，使用 'gloo' 后端，指定初始化方法的模板化文件名，以及当前进程的 `world_size` 和 `rank`
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # 在日志中记录等待训练器关闭信号的信息，包括当前训练器的 `rank`
        gLogger.info("Waiting for shutdown signal on trainer #%s...", rank)

        # 声明全局变量 `shutdown_signal` 并使用 `with` 语句获取其锁，等待关闭信号
        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        # 在日志中记录训练器退出的信息，包括当前训练器的 `rank`
        gLogger.info("Exiting the trainer #%s...", rank)
        # 销毁当前进程组的分布式进程组
        dist.destroy_process_group()

    # 定义一个方法 `_master_process`，用于处理主进程的逻辑，接受 `ddp_mode` 和 `simulate_uneven_inputs` 作为参数
    def _master_process(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool):
        # 在日志中记录主进程的运行信息
        gLogger.info("Running the master process...")
        # 初始化分布式进程组，使用 'gloo' 后端，指定初始化方法的模板化文件名，以及当前进程的 `world_size` 和 `rank`
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # 在远程工作节点上创建远程 `RRef` 对象 `remote_em_rref`，表示远程的 EM（期望最大化）计算对象
        remote_em_rref = rpc.remote(
            self.remote_worker_name(), RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
        )
        # 在远程工作节点上创建远程 `RRef` 对象 `remote_net_rref`，表示远程的神经网络计算对象
        remote_net_rref = rpc.remote(
            self.remote_worker_name(), RemoteNet, args=(D_DENSE + D_SPARSE, D_HID)
        )
        # 在日志中记录在主进程上创建远程 `RRef` 对象的信息
        gLogger.info("Created remote rrefs on master")

        # 在主进程上执行测试逻辑，传递 `ddp_mode`、`simulate_uneven_inputs`、远程 `RRef` 对象 `remote_em_rref` 和 `remote_net_rref`
        self.do_test_on_master(
            ddp_mode, simulate_uneven_inputs, remote_em_rref, remote_net_rref
        )

    # 定义一个方法 `do_test_on_master`，用于在主进程上执行测试逻辑
    def do_test_on_master(
        self,
        ddp_mode: DdpMode,
        simulate_uneven_inputs: bool,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
    ):
        # 该方法未提供具体的实现代码，在这里仅为方法定义提供了注释
        pass
        ):
            # 如果设置了模拟不均匀输入，记录日志以指示正在运行带有不均匀输入模拟的DDP + RPC测试。
            gLogger.info(
                "Running DDP + RPC test with simulating uneven inputs across trainers."
            )

        trainer_rrefs = []
        for rank in TRAINER_RANKS:
            # 为每个训练器创建远程引用，并添加到trainer_rrefs列表中
            trainer = self.trainer_name(rank)
            trainer_rrefs.append(
                rpc.remote(
                    trainer,
                    Trainer,
                    args=(remote_em_rref, remote_net_rref, ddp_mode, rank),
                )
            )

        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            # 在TRAINER_RANKS上调用dist.new_group以创建新的进程组
            dist.new_group(TRAINER_RANKS)

        # 获取训练示例数据
        training_examples = get_training_examples()
        for _ in range(3):
            futures = []
            num_trainers = len(trainer_rrefs)
            for idx, trainer_rref in enumerate(trainer_rrefs):
                # 判断是否模拟不均匀输入，并确定当前训练器是否有较少的输入数据
                trainer_has_less_inputs = (
                    simulate_uneven_inputs and idx < num_trainers // 2
                )
                # 异步调用远程方法Trainer.train_batch，并将结果future添加到futures列表中
                futures.append(
                    _remote_method_async(
                        Trainer.train_batch,
                        trainer_rref,
                        training_examples[idx],
                        trainer_has_less_inputs,
                        simulate_uneven_inputs,
                    )
                )

            for future in futures:
                # 等待future完成并获取返回的梯度
                ddp_grads, non_ddp_grads = future.wait()
                # 当没有模拟不均匀输入时，检查DDP参数的梯度是否全部为零
                if not simulate_uneven_inputs:
                    for grad in ddp_grads:
                        self.assertEqual(
                            grad,
                            torch.zeros_like(grad),
                            msg=f"The grad for any ddp parameter should be zeros, because "
                            "the training examples' grads cancel each other. Received "
                            f"gradient {grad}",
                        )
                # 检查非DDP参数的梯度是否不全为零
                for grad in non_ddp_grads:
                    self.assertNotEqual(
                        grad,
                        torch.zeros_like(grad),
                        msg="The grad for any non-ddp parameter shouldn't be zeros",
                    )

        # 销毁进程组
        for idx, trainer_rref in enumerate(trainer_rrefs):
            # 异步调用远程方法Trainer.destroy_pg来销毁进程组，并等待其完成
            _remote_method_async(Trainer.destroy_pg, trainer_rref).wait()

        # 发送关闭信号
        for rank in TRAINER_RANKS:
            # 获取每个训练器的名称，并使用RPC同步方式发送关闭信号
            trainer = self.trainer_name(rank)
            rpc.rpc_sync(trainer, set_shutdown_signal, args=())

        # 向远程工作器发送关闭信号
        rpc.rpc_sync(self.remote_worker_name(), set_shutdown_signal, args=())
    # 定义一个方法来执行测试，根据当前进程的角色进行不同的处理
    def _do_test(self, ddp_mode, simulate_uneven_inputs=False):
        # 如果当前进程是主进程
        if self.rank == MASTER_RANK:
            # 调用主进程处理方法，传入分布式数据并行模式和是否模拟不均匀输入
            self._master_process(ddp_mode, simulate_uneven_inputs)
        # 如果当前进程是远程工作进程
        elif self.rank == REMOTE_WORKER_RANK:
            # 调用远程工作进程处理方法，传入分布式数据并行模式
            self._remote_worker_process(ddp_mode)
        # 如果当前进程的角色在训练器进程列表中
        elif self.rank in TRAINER_RANKS:
            # 调用训练器进程处理方法，传入当前进程的角色
            self._trainer_process(self.rank)
        else:
            # 抛出运行时异常，表示未知的进程角色
            raise RuntimeError(f"Unknown process rank: {self.rank}")
    
    # 使用Gloo后端进行初始化，声明这是一个分布式测试方法
    @requires_gloo()
    @dist_init
    # 定义一个测试方法，用于测试在没有使用分布式数据并行的情况下的反向传播
    def test_backward_no_ddp(self):
        # 调用测试方法，执行没有使用分布式数据并行的反向传播测试
        self._do_test(DdpMode.NONE)
    
    # 使用Gloo后端进行初始化，声明这是一个分布式测试方法
    @requires_gloo()
    @dist_init
    # 定义一个测试方法，用于测试在分布式数据并行模式为外部环境的情况下的反向传播
    def test_backward_ddp_outside(self):
        # 调用测试方法，执行分布式数据并行模式为外部环境的反向传播测试
        self._do_test(DdpMode.OUTSIDE)
    
    # 使用Gloo后端进行初始化，声明这是一个分布式测试方法
    @requires_gloo()
    @dist_init
    # 定义一个测试方法，用于测试在分布式数据并行模式为外部环境且模拟不均匀输入的情况下的反向传播
    def test_backward_ddp_outside_uneven_inputs(self):
        # 调用测试方法，执行分布式数据并行模式为外部环境且模拟不均匀输入的反向传播测试
        self._do_test(DdpMode.OUTSIDE, simulate_uneven_inputs=True)
    
    # 使用Gloo后端进行初始化，声明这是一个分布式测试方法
    @requires_gloo()
    @dist_init
    # 定义一个测试方法，用于测试在分布式数据并行模式为内部环境的情况下的反向传播
    def test_backward_ddp_inside(self):
        # 调用测试方法，执行分布式数据并行模式为内部环境的反向传播测试
        self._do_test(DdpMode.INSIDE)
# 定义一个测试类，用于 CPU 和 CUDA 测试套件的通用工具
class CommonDdpComparisonTest(RpcAgentTestFixture):
    
    @property
    def world_size(self) -> int:
        # 返回训练节点的数量，这个值由 NUM_TRAINERS 决定
        return NUM_TRAINERS

    def trainer_name(self, rank):
        # 返回训练节点的名称，名称需与 'dist_init' 装饰器中保持一致
        return f"worker{rank}"

    @staticmethod
    def get_remote_grads(rref, context_id):
        # 使用 rref 和 context_id 获取远程梯度
        return dist_autograd.get_gradients(context_id)[rref.local_value().weight]


# 创建一个子类，继承自 CommonDdpComparisonTest 类
class DdpComparisonTest(CommonDdpComparisonTest):
    # 定义一个测试方法，用于比较分布式数据并行（DDP）的效果，支持可选的不均匀输入模拟
    def _run_test_ddp_comparision(self, simulate_uneven_inputs=False):
        # 在日志中记录当前训练器的排名
        gLogger.info("Running trainer rank: %s", self.rank)
        
        # 每个训练器使用不同的随机种子，以确保初始模型参数、输入和梯度都不相同，
        # 否则在DDP的全局归约之前和之后梯度可能相同。
        torch.manual_seed(self.rank)
        
        # 初始化进程组，使用gloo后端，初始化方法包括文件名后缀"_pg"，因为文件名也会被RPC代理使用
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=f"{self.file_name}_pg"),
            world_size=self.world_size,
            rank=self.rank,
        )
        
        # 创建一个简单的线性神经网络模型，输入维度为2，输出维度为3
        net = nn.Linear(2, 3)
        
        # 使用DDP对网络进行包装，使其支持分布式数据并行训练
        ddp_net = DistributedDataParallel(net)

        # 如果simulate_uneven_inputs为True，则模拟不均匀的输入
        num_inputs = 1
        if simulate_uneven_inputs:
            if self.rank % 2 == 0:
                num_inputs += 2
        inputs_list = [torch.rand((3, 2)) for _ in range(num_inputs)]

        # 如果simulate_uneven_inputs为True，则记录日志，说明当前排名的训练器使用了多少个输入
        if simulate_uneven_inputs:
            gLogger.info("Rank %s training with %s inputs.", self.rank, len(inputs_list))

        # 使用分布式自动求导，梯度将存储在RPC上下文映射中
        grads_dict = {}
        with ddp_net.join(simulate_uneven_inputs):
            for i, inputs in enumerate(inputs_list):
                with dist_autograd.context() as context_id:
                    # 计算损失并进行反向传播
                    loss = ddp_net(inputs).norm()
                    dist_autograd.backward(context_id, [loss])
                    grads_dict = dist_autograd.get_gradients(context_id)
                gLogger.info("Trainer #%s got grad dict: %s", self.rank, grads_dict)

                # 使用本地自动求导，梯度将存储在每个变量的'.grad'属性中
                ddp_net.zero_grad()
                loss = ddp_net(inputs).norm()
                loss.backward()

                # 检查本地自动求导和分布式自动求导得到的梯度是否一致
                for param in net.parameters():
                    self.assertTrue(
                        param in grads_dict,
                        msg=f"Param {param} is not in dist_auto grad dict {grads_dict} for iteration {i}",
                    )
                    self.assertEqual(
                        grads_dict[param],
                        param.grad,
                        msg=f"The grads for param {param} are different under local "
                        f"and dist autograd: {param.grad} \n---\n {grads_dict[param]} for iteration {i}",
                    )
        
        # 销毁进程组
        dist.destroy_process_group()

    @requires_gloo()
    @dist_init
    def test_ddp_comparison(self):
        # 执行DDP效果比较的测试
        self._run_test_ddp_comparision()

    @requires_gloo()
    @dist_init
    def test_ddp_comparison_uneven_inputs(self):
        # 执行模拟不均匀输入的DDP效果比较的测试
        self._run_test_ddp_comparision(simulate_uneven_inputs=True)

    # 注解结束
    def test_ddp_dist_autograd_sparse_grads(self):
        # 每个训练器使用不同的随机种子。否则，它们将具有完全相同的初始模型参数、输入和梯度，
        # 这意味着在 DDP 的全局归约之前和之后，梯度将是相同的。
        torch.manual_seed(self.rank)
        
        # 初始化分布式进程组，使用 Gloo 后端
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # 创建一个稀疏的 EmbeddingBag 模型
        model = nn.EmbeddingBag(10, 3, sparse=True)
        # 使用 DistributedDataParallel 封装模型
        ddp_model = DistributedDataParallel(model)

        # 为每个输入创建不同的数据
        input = torch.LongTensor(10).random_(0, 10)
        offsets = torch.LongTensor([0, 4])

        # 在本地运行模型
        loss = ddp_model(input, offsets).sum()
        # 计算梯度
        loss.backward()

        # 使用 dist_autograd 上下文管理器
        with dist_autograd.context() as context_id:
            # 再次运行模型
            loss = ddp_model(input, offsets).sum()
            # 调用 dist_autograd 进行反向传播
            dist_autograd.backward(context_id, [loss])
            # 获取梯度字典
            grads_dict = dist_autograd.get_gradients(context_id)
            # 断言梯度字典中的模型权重梯度
            self.assertEqual(1, len(grads_dict))
            self.assertEqual(model.weight.grad, grads_dict[model.weight])

    @requires_gloo()
    @dist_init
    def test_ddp_dist_autograd_local_vs_remote(self):
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        # 设置随机种子，确保每个训练器使用不同的种子，避免初始模型参数、输入及梯度完全相同，
        # 这意味着在进行DDP全局归约前后梯度应该不同。
        torch.manual_seed(self.rank)
        
        # 初始化进程组，使用Gloo后端，根据给定的初始化方法和参数进行初始化
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # 使用两种不同的远程设备输入字符串，一种带有默认设备字符串"cpu"，一种不带
        for remote_device in ["worker0/cpu", "worker0"]:
            # 创建远程模块对象，指定远程设备和模块类，初始化参数为(输入维度, 输出维度, 是否包含偏置)
            remote_layer1 = RemoteModule(
                remote_device=remote_device, module_cls=nn.Linear, args=(10, 5, False)
            )
            # 创建本地模块对象，与远程模块初始参数相同
            layer1 = nn.Linear(10, 5, False)
            layer1.weight = remote_layer1.module_rref.to_here().weight  # 使用远程模块的权重作为本地模块的初始权重

            # 运行本地情况
            layer2 = nn.Linear(5, 1)
            inputs = torch.rand((10, 10))
            ddp_model = DistributedDataParallel(layer2)
            loss = ddp_model(layer1(inputs)).sum()  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 运行远程情况
            with dist_autograd.context() as context_id:
                loss = ddp_model(remote_layer1(inputs)).sum()  # 计算远程模块的损失
                dist_autograd.backward(context_id, [loss])  # 使用分布式自动求导进行反向传播
                grads_dict = dist_autograd.get_gradients(context_id)  # 获取梯度字典
                dist.barrier()  # 等待所有进程完成

                # 检查本地模块的权重梯度与远程模块的权重梯度是否相等
                self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])
                
                # 使用RPC同步调用获取远程模块的权重梯度，并检查与本地模块的权重梯度是否相等
                self.assertEqual(
                    layer1.weight.grad,
                    rpc.rpc_sync(
                        "worker0",
                        CommonDdpComparisonTest.get_remote_grads,
                        args=(remote_layer1.module_rref, context_id),
                    ),
                )
class CudaDdpComparisonTest(CommonDdpComparisonTest):
    @skip_if_lt_x_gpu(NUM_TRAINERS)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_ddp_dist_autograd_local_vs_remote_gpu(self):
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)  # 设置随机种子，每个 trainer 使用不同的种子

        # 初始化进程组，使用 gloo 后端，初始化方法使用指定的模板格式
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # 创建远程模块，运行在 "worker0/cpu" 上的 nn.Linear 模块，输入参数为 (10, 7, False)
        remote_layer1 = RemoteModule(
            remote_device="worker0/cpu", module_cls=nn.Linear, args=(10, 7, False)
        )

        # 创建本地的 nn.Linear 模块，输入参数为 (10, 7, False)
        layer1 = nn.Linear(10, 7, False)

        # 将远程模块的权重复制到本地模块中
        layer1.weight = remote_layer1.module_rref.to_here().weight

        # 在当前设备上创建 nn.Linear 模块 (7, 5)，并将其放置在 self.rank 对应的 GPU 上
        layer2 = nn.Linear(7, 5).cuda(self.rank)

        # 使用 DistributedDataParallel 封装 layer2，指定设备为 self.rank 对应的 GPU
        ddp_layer2 = DistributedDataParallel(layer2, device_ids=[self.rank])

        # 创建远程模块，运行在 "worker0/cpu" 上的 nn.Linear 模块，输入参数为 (5, 3, False)
        remote_layer3 = RemoteModule(
            remote_device="worker0/cpu", module_cls=nn.Linear, args=(5, 3, False)
        )

        # 创建本地的 nn.Linear 模块，输入参数为 (5, 3, False)
        layer3 = nn.Linear(5, 3, False)

        # 将远程模块的权重复制到本地模块中
        layer3.weight = remote_layer3.module_rref.to_here().weight

        # 在当前设备上创建 nn.Linear 模块 (3, 1)，并将其放置在 self.rank 对应的 GPU 上
        layer4 = nn.Linear(3, 1).cuda(self.rank)

        # 使用 DistributedDataParallel 封装 layer4，指定设备为 self.rank 对应的 GPU
        ddp_layer4 = DistributedDataParallel(layer4, device_ids=[self.rank])

        # 运行本地情况下的计算
        inputs = torch.rand((10, 10))
        loss = ddp_layer4(
            layer3(ddp_layer2(layer1(inputs).cuda(self.rank)).cpu()).cuda(self.rank)
        ).sum()
        loss.backward()

        # 运行远程情况下的计算
        with dist_autograd.context() as context_id:
            loss = ddp_layer4(
                remote_layer3(
                    ddp_layer2(remote_layer1(inputs).cuda(self.rank)).cpu()
                ).cuda(self.rank)
            ).sum()
            dist_autograd.backward(context_id, [loss])
            grads_dict = dist_autograd.get_gradients(context_id)
            dist.barrier()

            # 断言每个层的梯度与远程梯度一致
            self.assertEqual(
                layer1.weight.grad,
                rpc.rpc_sync(
                    "worker0",
                    CommonDdpComparisonTest.get_remote_grads,
                    args=(remote_layer1.module_rref, context_id),
                ),
            )
            self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])
            self.assertEqual(
                layer3.weight.grad,
                rpc.rpc_sync(
                    "worker0",
                    CommonDdpComparisonTest.get_remote_grads,
                    args=(remote_layer3.module_rref, context_id),
                ),
            )
            self.assertEqual(layer4.weight.grad, grads_dict[layer4.weight])
```