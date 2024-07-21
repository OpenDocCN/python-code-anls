# `.\pytorch\torch\testing\_internal\distributed\rpc\examples\parameter_server_test.py`

```
# mypy: ignore-errors

# 如果需要修改此文件以使测试通过，请同样地修改以下文件：
# https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py
# 和 https://pytorch.org/tutorials/intermediate/rpc_async_execution.html#batch-updating-parameter-server

import threading  # 导入线程模块
from datetime import datetime  # 导入日期时间模块中的日期时间类
from time import perf_counter  # 导入性能计数器模块中的性能计数器函数

import torch  # 导入PyTorch模块
import torch.distributed.rpc as rpc  # 导入PyTorch分布式RPC模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import optim  # 从PyTorch中导入优化器模块

from torch.testing._internal.dist_utils import (  # 导入内部测试工具中的分布式工具
    dist_init,  # 分布式初始化函数
    worker_name,  # 工作节点名称函数
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture  # 导入RPC代理测试固件类

batch_size = 20  # 批处理大小
in_features = 100  # 输入特征数
out_features = 30  # 输出特征数
num_batches = 4  # 批次数量


def timed_log(text):
    # 打印当前时间和给定文本的日志信息
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer:

    def __init__(self, batch_update_size):
        self.model = nn.Linear(in_features, out_features)  # 创建线性模型
        self.lock = threading.Lock()  # 创建线程锁对象
        self.future_model = torch.futures.Future()  # 创建用于异步操作的Future对象
        self.batch_update_size = batch_update_size  # 批量更新大小
        self.curr_update_size = 0  # 当前更新计数器
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)  # 创建SGD优化器
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)  # 将每个参数的梯度初始化为零向量

    def get_model(self):
        # 返回当前模型
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        # 静态方法：更新模型并获取模型的异步RPC函数装饰器
        self = ps_rref.local_value()  # 获取本地值
        for p, g in zip(self.model.parameters(), grads):
            if p.grad is None:
                p.grad = g  # 如果梯度为空，则设置为当前梯度
            else:
                p.grad += g  # 否则累加梯度
        with self.lock:
            timed_log(f"PS got {self.curr_update_size}/{self.batch_update_size} updates")  # 记录日志：参数服务器接收到的更新计数
            self.curr_update_size += 1  # 更新计数器加1
            fut = self.future_model  # 获取当前Future对象

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size  # 计算平均梯度
                self.curr_update_size = 0  # 重置更新计数器
                self.optimizer.step()  # 执行优化步骤
                self.optimizer.zero_grad()  # 清零梯度
                fut.set_result(self.model)  # 设置Future的结果为当前模型
                timed_log("PS updated model")  # 记录日志：参数服务器更新了模型
                self.future_model = torch.futures.Future()  # 创建新的Future对象

        return fut  # 返回Future对象


class Trainer:

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref  # 参数服务器的远程引用
        self.loss_fn = nn.L1Loss()  # L1损失函数

    def get_next_batch(self):
        # 生成器函数：生成下一个批次的输入和标签
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, in_features)  # 生成随机输入数据
            labels = torch.zeros(batch_size, out_features)  # 创建全零标签数据
            yield inputs, labels  # 生成器的返回值
    # 获取当前工作进程的名称
    name = rpc.get_worker_info().name
    # 从参数服务器同步获取模型
    m = self.ps_rref.rpc_sync().get_model()
    # 对每个获取到的输入数据和标签执行训练批处理
    for inputs, labels in self.get_next_batch():
        # 记录日志，显示当前工作进程正在处理一个批次
        timed_log(f"{name} processing one batch")
        # 使用当前模型对输入进行预测并计算损失，然后反向传播损失
        self.loss_fn(m(inputs), labels).backward()
        # 记录日志，显示当前工作进程正在报告梯度
        timed_log(f"{name} reporting grads")
        # 将模型参数梯度传输至参数服务器，并获取更新后的模型
        m = rpc.rpc_sync(
            self.ps_rref.owner(),
            BatchUpdateParameterServer.update_and_fetch_model,
            args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
        )
        # 记录日志，显示当前工作进程已获取更新后的模型
        timed_log(f"{name} got updated model")
# 定义运行训练器的函数，接受参数服务器的远程引用作为输入
def run_trainer(ps_rref):
    # 创建训练器对象
    trainer = Trainer(ps_rref)
    # 调用训练方法
    trainer.train()

# 运行参数服务器的训练过程
def run_ps(trainers):
    # 记录开始训练的时间点
    timed_log("Start training")
    start = perf_counter()
    # 创建参数服务器的远程引用，并用其初始化 RPC
    ps_rref = rpc.RRef(BatchUpdateParameterServer(len(trainers)))
    # 初始化异步 RPC 调用的结果列表
    futs = []
    # 对每个训练器发起异步 RPC 调用，运行训练函数并传入参数服务器的远程引用
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    # 等待所有异步调用完成
    torch.futures.wait_all(futs)
    # 记录结束训练的时间点
    stop = perf_counter()
    # 记录训练总时长
    timed_log("Finish training")
    timed_log(f"Time spent training: {stop-start}s")

# 参数服务器测试类，继承自 RpcAgentTestFixture
class ParameterServerTest(RpcAgentTestFixture):

    @dist_init(setup_rpc=False)
    def test_batch_updating_parameter_server(self):
        # 如果当前进程不是主进程
        if self.rank != 0:
            # 初始化 RPC，作为一个 worker 节点
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        else:
            # 初始化 RPC，作为主节点
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            # 运行参数服务器的训练过程，传入除主节点外的所有 worker 节点名称列表
            run_ps([f"{worker_name(r)}" for r in range(1, self.world_size)])

        # 关闭 RPC
        rpc.shutdown()
```