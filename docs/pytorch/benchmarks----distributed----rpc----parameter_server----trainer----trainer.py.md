# `.\pytorch\benchmarks\distributed\rpc\parameter_server\trainer\trainer.py`

```
import functools  # 导入 functools 模块，用于高阶函数和函数操作
import time  # 导入 time 模块，用于时间相关操作
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 类和 abstractmethod 装饰器

from metrics.MetricsLogger import MetricsLogger  # 从自定义模块 metrics.MetricsLogger 导入 MetricsLogger 类

import torch  # 导入 PyTorch 库

class TrainerBase(ABC):
    BATCH_LEVEL_METRIC = "batch_level_metric"  # 类常量，表示批次级别的度量
    BATCH_ALL = "batch_all"  # 类常量，表示所有批次的度量
    FORWARD_METRIC = "forward_metric"  # 类常量，表示前向传播的度量
    FORWARD_PASS = "forward_pass"  # 类常量，表示前向传播的事件
    BACKWARD_METRIC = "backward_metric"  # 类常量，表示反向传播的度量
    BACKWARD = "backward"  # 类常量，表示反向传播的事件

    def __init__(self, rank):
        r"""
        Inits TrainerBase class.
        Args:
            rank (int): worker rank
        """
        self.__metrics_logger = MetricsLogger(rank)  # 初始化 MetricsLogger 实例，传入工作器的排名

    @abstractmethod
    def train(self):
        r"""
        A method to be implemented by child class that will train a neural network.
        """
        return  # 抽象方法，子类必须实现，用于训练神经网络

    def record_start(self, type, key, name, cuda=True):
        r"""
        A method that records the start event for a metric.
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
            name (str): description of the metric
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(type, key, name, cuda)  # 调用 MetricsLogger 实例的记录开始事件的方法

    def record_end(self, type, key):
        r"""
        A method that records the end event for a metric.
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(type, key)  # 调用 MetricsLogger 实例的记录结束事件的方法

    def record_batch_start(self, key, cuda=True):
        r"""
        A helper method that records a batch metric for the
        given key. A user should call this at the start of an
        iteration step during training.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.BATCH_LEVEL_METRIC, key, self.BATCH_ALL, cuda
        )  # 调用 MetricsLogger 实例的记录批次开始事件的方法

    def record_batch_end(self, key):
        r"""
        A helper method that records a batch metric for the
        given key. A user should call this at the end of an
        iteration step during training.
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(self.BATCH_LEVEL_METRIC, key)  # 调用 MetricsLogger 实例的记录批次结束事件的方法

    def record_forward_start(self, key, cuda=True):
        r"""
        A helper method that records a forward metric
        for the given key. A user should call this before
        their neural network forward.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.FORWARD_METRIC, key, self.FORWARD_PASS, cuda
        )  # 调用 MetricsLogger 实例的记录前向传播开始事件的方法
    # 记录前向传播结束时的指标数据，用户应在神经网络前向传播后调用此方法
    def record_forward_end(self, key):
        self.__metrics_logger.record_end(self.FORWARD_METRIC, key)

    # 记录反向传播开始时的指标数据，用户应在 .backward() 调用前调用此方法
    def record_backward_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.BACKWARD_METRIC, key, self.BACKWARD, cuda
        )

    # 记录反向传播结束时的指标数据，用户应在 .backward() 调用后调用此方法
    def record_backward_end(self, key):
        self.__metrics_logger.record_end(self.BACKWARD_METRIC, key)

    @staticmethod
    # 一个装饰器，用于记录装饰方法的指标数据
    def methodmetric(name, type="method_metric", cuda=True):
        def decorator(function):
            @functools.wraps(function)
            def wrapper(self, *args):
                key = time.time()
                # 记录方法执行开始时的指标数据
                self.__metrics_logger.record_start(type, key, name, cuda)
                result = function(self, *args)
                # 记录方法执行结束时的指标数据
                self.__metrics_logger.record_end(type, key)
                return result

            return wrapper

        return decorator

    # 返回由 __metrics_logger 捕获的指标数据的方法
    def get_metrics(self):
        return self.__metrics_logger.get_processed_metrics()

    # 清除 __metrics_logger 记录的指标数据的方法
    def clear_metrics(self):
        return self.__metrics_logger.clear_metrics()
# 定义一个名为 DdpTrainer 的类，继承自 TrainerBase 类
class DdpTrainer(TrainerBase):
    # 初始化方法，接收多个参数来配置 DDP 训练器
    def __init__(
        self,
        process_group,  # 分布式进程组
        use_cuda_rpc,   # 是否使用 CUDA RPC
        server_rref,    # 服务器的远程引用
        backend,        # 分布式通信后端
        epochs,         # 训练的总轮数
        preprocess_data,    # 数据预处理函数，用于训练前处理数据
        create_criterion,   # 创建损失函数的函数
        create_ddp_model,   # 创建 DDP 模型的函数
        hook_state_class,   # 用于在训练期间跟踪状态的类
        hook,           # DDP 通信钩子函数
        iteration_step  # 执行训练的一步操作的函数
    ):
        # 调用父类 TrainerBase 的初始化方法，传入当前进程的排名
        super().__init__(process_group.rank())
        
        # 设置各个参数作为实例变量
        self.process_group = process_group
        self.use_cuda_rpc = use_cuda_rpc
        self.server_rref = server_rref
        self.backend = backend
        self.epochs = epochs
        self.preprocess_data = preprocess_data
        self.create_criterion = create_criterion
        self.create_ddp_model = create_ddp_model
        self.hook_state_class = hook_state_class
        self.hook = hook
        self.iteration_step = iteration_step
        
        # 获取当前进程的排名和进程组的总数
        self.rank = process_group.rank()
        self.trainer_count = process_group.size()

    # 定义一个方法 epoch_key，返回编码的键，代表当前的 epoch 和迭代索引
    def epoch_key(self, epoch, index):
        """
        Args:
            epoch (int): 当前的 epoch 索引
            index (int): 当前的迭代索引
        """
        return f"{epoch},{index}"
    def train(self, model, data):
        r"""
        A method that implements the training algorithm.
        Args:
            model (nn.Module): neural network model
            data (list): training examples
        """
        # 将模型移到指定的 GPU 设备上进行训练
        model = model.cuda(self.rank)
        # 对数据进行预处理，可能依赖于当前进程的排名
        data = self.preprocess_data(self.rank, data)
        # 创建用于计算损失的标准
        criterion = self.create_criterion(self.rank)
        # 创建分布式数据并行模型及其钩子状态
        ddp_model, hook_state = self.create_ddp_model(
            self, self.rank, model, self.process_group, self.hook_state_class, self.hook
        )
        # 使用随机梯度下降算法初始化优化器
        optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)

        # 开始训练循环，迭代每个 epoch
        for epoch in range(self.epochs):
            # 每隔五个 epoch 打印训练状态信息（仅限排名为 0 的进程）
            if epoch % 5 == 0 and self.rank == 0:
                print(f"train epoch={epoch}")
            # 遍历数据集的每个 batch 进行训练
            for index, batch in enumerate(data):
                # 执行训练的迭代步骤，传递当前的训练状态和参数
                self.iteration_step(
                    self,
                    ddp_model,
                    criterion,
                    optimizer,
                    hook_state,
                    epoch,
                    index,
                    batch,
                )
        # 等待当前 GPU 设备上的所有操作完成
        torch.cuda.synchronize(self.rank)
```