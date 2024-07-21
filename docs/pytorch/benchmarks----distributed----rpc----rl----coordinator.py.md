# `.\pytorch\benchmarks\distributed\rpc\rl\coordinator.py`

```
import time  # 导入时间模块，用于处理时间相关操作

import numpy as np  # 导入NumPy库，用于数值计算

from agent import AgentBase  # 导入AgentBase类，用于定义Agent的基本功能
from observer import ObserverBase  # 导入ObserverBase类，用于定义Observer的基本功能

import torch  # 导入PyTorch深度学习库
import torch.distributed.rpc as rpc  # 导入PyTorch分布式RPC模块，用于远程过程调用

COORDINATOR_NAME = "coordinator"  # 定义协调器的名称常量
AGENT_NAME = "agent"  # 定义Agent的名称常量
OBSERVER_NAME = "observer{}"  # 定义Observer的名称模板常量

EPISODE_STEPS = 100  # 定义每个Episode的步数常量


class CoordinatorBase:
    def __init__(self, batch_size, batch, state_size, nlayers, out_features):
        r"""
        Coordinator object to run on worker.  Only one coordinator exists.  Responsible
        for facilitating communication between agent and observers and recording benchmark
        throughput and latency data.
        Args:
            batch_size (int): Number of observer requests to process in a batch
            batch (bool): Whether to process and respond to observer requests as a batch or 1 at a time
            state_size (list): List of ints dictating the dimensions of the state
            nlayers (int): Number of layers in the model
            out_features (int): Number of out features in the model
        """
        self.batch_size = batch_size  # 初始化批处理大小属性
        self.batch = batch  # 初始化批处理模式属性

        self.agent_rref = None  # Agent的远程引用，初始值为None
        self.ob_rrefs = []  # Observer的远程引用列表，初始为空列表

        agent_info = rpc.get_worker_info(AGENT_NAME)  # 获取Agent的工作信息
        self.agent_rref = rpc.remote(agent_info, AgentBase)  # 在远程创建Agent对象的引用

        for rank in range(batch_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank + 2))  # 获取Observer的工作信息
            ob_ref = rpc.remote(ob_info, ObserverBase)  # 在远程创建Observer对象的引用
            self.ob_rrefs.append(ob_ref)  # 将创建的Observer远程引用添加到列表中

            ob_ref.rpc_sync().set_state(state_size, batch)  # 调用Observer的同步RPC方法设置其状态信息

        self.agent_rref.rpc_sync().set_world(
            batch_size, state_size, nlayers, out_features, self.batch
        )  # 调用Agent的同步RPC方法设置整个系统的参数信息
```