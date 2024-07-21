# `.\pytorch\benchmarks\distributed\rpc\parameter_server\server\server.py`

```
import functools  # 导入 functools 模块，用于高阶函数的支持
import threading  # 导入 threading 模块，用于多线程编程的支持
import time  # 导入 time 模块，用于时间相关操作的支持
from abc import ABC, abstractmethod  # 从 abc 模块中导入 ABC 和 abstractmethod 装饰器

from metrics.MetricsLogger import MetricsLogger  # 从 metrics.MetricsLogger 模块中导入 MetricsLogger 类
from utils import sparse_rpc_format_to_tensor, sparse_tensor_to_rpc_format  # 从 utils 模块中导入两个函数

import torch  # 导入 PyTorch 库
import torch.distributed.rpc as rpc  # 导入 PyTorch 的分布式 RPC 模块


class ParameterServerBase(ABC):
    PARAMETER_SERVER_BATCH_METRIC = "parameter_server_batch_metric"  # 定义批次相关的参数服务器指标常量
    PARAMETER_SERVER_STRAGGLER_METRIC = "parameter_server_straggler_metric"  # 定义滞后相关的参数服务器指标常量
    PARAM_INDEX_STRAGGLER = "param_index_straggler"  # 定义滞后参数索引常量
    PARAM_INDEX_BATCH = "param_index_batch"  # 定义批次参数索引常量

    def __init__(self, rank):
        r"""
        Inits ParameterServerBase class.
        Args:
            rank (int): worker rank
        """
        self.__metrics_logger = MetricsLogger(rank)  # 使用指定的 rank 创建 MetricsLogger 实例

    @abstractmethod
    def process_gradient(self):
        r"""
        A method to be implemented by child class that will process a
        gradient received by a server.
        """
        return  # 抽象方法，子类需要实现，用于处理服务器接收到的梯度

    @staticmethod
    @abstractmethod
    def average_gradient():
        r"""
        A method to be implemented by child class that will average
        gradients.
        """
        return  # 抽象静态方法，子类需要实现，用于计算平均梯度

    @staticmethod
    @abstractmethod
    def reset_state():
        r"""
        A method to be implemented by child class that will reset
        the server state.
        """
        return  # 抽象静态方法，子类需要实现，用于重置服务器状态

    def record_start(self, type, key, name, cuda=True):
        r"""
        A method that records the start event for a metric.
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
            name (str): description of the metric
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(type, key, name, cuda)  # 调用 MetricsLogger 实例的记录起始事件的方法

    def record_end(self, type, key):
        r"""
        A method that records the end event for a metric
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(type, key)  # 调用 MetricsLogger 实例的记录结束事件的方法

    def record_straggler_start(self, key, cuda=True):
        r"""
        A helper method that records a straggler metric
        for the given key. A user should call this when
        the first gradient for the param location is received.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.PARAMETER_SERVER_STRAGGLER_METRIC,  # 使用参数服务器滞后指标
            key,  # 使用指定的 key
            self.PARAM_INDEX_STRAGGLER,  # 使用滞后参数索引
            cuda,  # 使用给定的 CUDA 指示符
        )
    # 记录参数服务器中落单任务结束时的指标
    def record_straggler_end(self, key):
        self.__metrics_logger.record_end(self.PARAMETER_SERVER_STRAGGLER_METRIC, key)

    # 记录参数服务器中批处理任务开始时的指标
    def record_batch_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.PARAMETER_SERVER_BATCH_METRIC, key, self.PARAM_INDEX_BATCH, cuda
        )

    # 记录参数服务器中批处理任务结束时的指标
    def record_batch_end(self, key):
        self.__metrics_logger.record_end(self.PARAMETER_SERVER_BATCH_METRIC, key)

    # 静态方法：用于装饰记录方法指标的装饰器
    @staticmethod
    def record_method(name, type="method_metric", cuda=True):
        def decorator(function):
            @functools.wraps(function)
            def wrapper(self, *args):
                key = time.time()
                # 记录方法开始时的指标
                self.__metrics_logger.record_start(type, key, name, cuda)
                result = function(self, *args)
                # 记录方法结束时的指标
                self.__metrics_logger.record_end(type, key)
                return result

            return wrapper

        return decorator

    # 静态方法：返回由 __metrics_logger 捕获的指标
    @staticmethod
    def get_metrics(server_rref):
        self = server_rref.local_value()
        return self.__metrics_logger.get_processed_metrics()

    # 清除 __metrics_logger 记录的指标的方法
    def clear_metrics(self):
        return self.__metrics_logger.clear_metrics()
class AverageParameterServer(ParameterServerBase):
    def __init__(self, rank, trainer_count, use_cuda_rpc):
        r"""
        A parameter server that averages the gradients
        from trainers for each training iteration step.
        Gradients are added as they are received from trainers.
        When all gradients have been received, the sum is
        divided by the number of trainers.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainers sending
                gradients to the server
            use_cuda_rpc (bool): indicator for CUDA RPC
        """
        super().__init__(rank)

        # 初始化锁对象，用于多线程同步
        self.lock = threading.Lock()
        # 设置参数服务器的排名
        self.rank = rank
        # 设置训练器的数量
        self.trainer_count = trainer_count
        # 设置是否使用 CUDA RPC
        self.use_cuda_rpc = use_cuda_rpc

        # 初始化批次号为 0
        self.batch_number = 0
        # 初始化 futures 字典，用于存储未来的异步任务
        self.futures = {}
        # 初始化梯度字典，用于存储从训练器接收到的梯度
        self.gradient_dict = {}

    @staticmethod
    def reset_state(server_rref):
        r"""
        A method that clears the state of the server.
        Args:
            server_rref (RRef): remote reference to the server
        """
        # 从 server_rref 中获取本地对象的引用
        self = server_rref.local_value()
        # 重置批次号为 0
        self.batch_number = 0
        # 清空未来的异步任务字典
        self.futures.clear()
        # 清空梯度字典
        self.gradient_dict.clear()
        # 清空服务器的度量指标
        self.clear_metrics()

    def param_key(self, param_loc):
        r"""
        A method that returns an encoded key that represents
        the current batch and param location.
        Args:
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        # 返回编码的键，代表当前批次和参数位置
        return f"{self.batch_number},{param_loc}"

    def clear_batch_state(self):
        r"""
        Clears the current server batch state.
        """
        # 清空当前服务器批次状态的未来异步任务字典
        self.futures.clear()
        # 清空当前服务器批次状态的梯度字典
        self.gradient_dict.clear()

    def process_gradient(self, gradient, param_loc):
        r"""
        Stores the gradient if param_loc is not in gradient_dict.
        Adds the gradient to param_loc if it is in gradient_dict.
        Args:
            gradient (torch.Tensor): tensor sent from trainer
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        # 如果 param_loc 不在梯度字典中，记录慢处理器的开始和批次的开始，并存储梯度
        if param_loc not in self.gradient_dict:
            self.record_straggler_start(self.param_key(param_loc))
            self.record_batch_start(self.param_key(param_loc))
            self.gradient_dict[param_loc] = gradient
        else:
            # 如果 param_loc 已经在梯度字典中，则将梯度累加到现有的梯度上
            self.gradient_dict[param_loc] += gradient

    @ParameterServerBase.record_method(name="average computation")
    def average(self, param_loc):
        r"""
        Obtains the tensor at the param_loc in the gradient_dict
        and then divides by number of trainers.
        Args:
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        # 获取梯度字典中 param_loc 处的张量，然后除以训练器的数量
        param_loc_avg = self.gradient_dict[param_loc]
        param_loc_avg / (1.0 * self.trainer_count)
        return param_loc_avg
    # 使用 @rpc.functions.async_execution 装饰器定义一个异步函数，用于处理平均梯度的计算和分发
    @rpc.functions.async_execution
    # 定义函数 average_gradient，用于接收并处理来自训练器的梯度数据
    def average_gradient(server_rref, received_batch_number, param_loc, gradient):
        r"""
        An asynchronous function that will average gradients
        sent from trainers.
        Args:
            server_rref (RRef): remote reference to the server
                服务器的远程引用
            received_batch_number (int): batch number sent by
                the trainer
                训练器发送的批次号
            param_loc (int): bucket location sent by the trainer
                containing the gradient
                包含梯度的桶位置，由训练器发送
            gradient (torch.Tensor or list): tensor sent by the trainer
                训练器发送的张量或列表
        """
        # 获取 server_rref 的本地值
        self = server_rref.local_value()
        # 如果梯度是列表形式，则转换为稀疏张量格式
        if type(gradient) is list:
            gradient = sparse_rpc_format_to_tensor(gradient)
        # 将梯度移动到与服务器 rank 对应的 CUDA 设备上
        gradient = gradient.cuda(self.rank)
        # 创建一个 Torch Future 对象
        fut = torch.futures.Future()
        # 使用 self.lock 进行同步
        with self.lock:
            # 如果当前批次号比接收到的批次号小，则更新批次号并清除批次状态
            if self.batch_number < received_batch_number:
                self.batch_number = received_batch_number
                self.clear_batch_state()
            # 处理梯度更新到服务器
            self.process_gradient(gradient, param_loc)
            # 如果 param_loc 不在 self.futures 中，则将其添加进去
            if param_loc not in self.futures:
                self.futures[param_loc] = []
            # 将当前 Future 对象添加到相应的 param_loc 的 Future 列表中
            self.futures[param_loc].append(fut)
            # 如果某个 param_loc 的 Future 数量等于训练器的数量，则进行以下操作
            if len(self.futures[param_loc]) == self.trainer_count:
                # 记录 straggler 的结束时间，并计算该 param_loc 的平均值
                self.record_straggler_end(self.param_key(param_loc))
                param_loc_avg = self.average(param_loc)
                # 如果不使用 CUDA RPC，则将 param_loc_avg 转移到 CPU 上
                if not self.use_cuda_rpc:
                    param_loc_avg = param_loc_avg.cpu()
                # 如果 param_loc_avg 是稀疏张量，则转换为 RPC 格式
                if param_loc_avg.is_sparse:
                    param_loc_avg = sparse_tensor_to_rpc_format(param_loc_avg)
                # 将 param_loc_avg 设置为每个 Future 的结果
                for cur_fut in self.futures[param_loc]:
                    cur_fut.set_result(param_loc_avg)
                # 记录该批次的结束时间
                self.record_batch_end(self.param_key(param_loc))
        # 返回 Future 对象
        return fut
# 继承自 AverageParameterServer 的 AverageBatchParameterServer 类，用于实现批处理平均参数服务器功能

def __init__(self, rank, trainer_count, use_cuda_rpc):
    r"""
    初始化方法，用于设置参数服务器的基本属性和状态
    Args:
        rank (int): 工作节点的排名
        trainer_count (int): 发送梯度到服务器的训练节点数量
        use_cuda_rpc (bool): 指示是否使用 CUDA RPC
    """
    super().__init__(rank, trainer_count, use_cuda_rpc)

def process_gradient(self, gradient, param_loc):
    r"""
    处理梯度的方法，将梯度添加到指定的参数位置的梯度字典中
    Args:
        gradient (torch.Tensor): 从训练节点发送的张量梯度
        param_loc (int): 训练节点发送的包含梯度的桶位置
    """
    if param_loc not in self.gradient_dict:
        # 记录梯度字典中新桶的处理开始时间
        self.record_straggler_start(self.param_key(param_loc))
        # 记录梯度字典中新桶的批处理开始时间
        self.record_batch_start(self.param_key(param_loc))
        # 初始化一个空的梯度列表作为该位置的梯度存储
        self.gradient_dict[param_loc] = []
    # 将梯度添加到指定参数位置的梯度列表中
    self.gradient_dict[param_loc].append(gradient)

@ParameterServerBase.record_method(name="average computation")
def average(self, param_loc):
    r"""
    计算平均值的方法，对指定位置的梯度进行求和然后除以训练节点数量得到平均值
    Args:
        param_loc (int): 训练节点发送的包含梯度的桶位置
    """
    # 获取指定位置梯度列表的第一个梯度作为初始平均值
    param_loc_avg = self.gradient_dict[param_loc][0]
    # 依次加上剩余梯度，求得总和
    for gradient in self.gradient_dict[param_loc][1:]:
        param_loc_avg += gradient
    # 计算平均值，但此处没有对 param_loc_avg 进行更新
    param_loc_avg / (1.0 * self.trainer_count)
    return param_loc_avg
```