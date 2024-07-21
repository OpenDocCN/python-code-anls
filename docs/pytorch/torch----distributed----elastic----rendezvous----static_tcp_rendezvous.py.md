# `.\pytorch\torch\distributed\elastic\rendezvous\static_tcp_rendezvous.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime  # 导入datetime模块，用于处理日期和时间
import logging  # 导入logging模块，用于日志记录
from typing import cast, Optional  # 导入类型提示模块中的cast和Optional类型

from torch.distributed import PrefixStore, Store, TCPStore  # 导入Torch分布式相关模块
from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
)  # 导入Rendezvous相关类和函数
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint  # 导入解析Rendezvous端点的工具函数


__all__ = ["StaticTCPRendezvous", "create_rdzv_handler"]  # 定义可导出的模块成员名称列表

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

_default_timeout_seconds = 600  # 设置默认超时时间为600秒


class StaticTCPRendezvous(RendezvousHandler):
    """
    Static rendezvous that is a wrapper around the TCPStore.

    Creates TCPStore based on the input parameters with the
    listener on the agent with group_rank=0
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        run_id: str,
        timeout: int,
    ):
        self.master_addr = master_addr  # 设置主节点地址
        self.master_port = master_port  # 设置主节点端口号
        self.rank = rank  # 设置当前节点的排名
        self.world_size = world_size  # 设置总节点数
        self.run_id = run_id  # 设置运行时ID
        self.timeout = datetime.timedelta(seconds=timeout)  # 设置超时时间间隔为指定秒数的时间增量
        self._store: Optional[Store] = None  # 初始化存储对象为None

    def get_backend(self) -> str:
        return "static"  # 返回后端类型为"static"

    @property
    def use_agent_store(self) -> bool:
        return True  # 始终返回True，表示使用代理存储

    def next_rendezvous(self) -> RendezvousInfo:
        logger.info("Creating TCPStore as the c10d::Store implementation")
        is_master = self.rank == 0  # 判断当前节点是否为主节点
        if not self._store:
            self._store = TCPStore(  # 创建TCPStore对象
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
            )
        store = PrefixStore(self.run_id, self._store)  # 使用前缀存储封装TCPStore
        # TCPStore服务器实例被训练器代码使用
        bootstrap_store_info = RendezvousStoreInfo(self.master_addr, self.master_port)
        return RendezvousInfo(
            store,
            self.rank,
            self.world_size,
            bootstrap_store_info,
        )  # 返回RendezvousInfo对象，包含存储、当前节点排名、总节点数和引导存储信息

    def is_closed(self):
        return False  # 始终返回False，表示未关闭状态

    def set_closed(self):
        pass  # 空操作，未实现关闭操作

    def num_nodes_waiting(self):
        return 0  # 返回等待的节点数为0，表示没有节点在等待

    def get_run_id(self) -> str:
        return self.run_id  # 返回当前运行ID

    def shutdown(self) -> bool:
        return True  # 返回True，表示成功关闭


def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    if "rank" not in params.config:
        raise ValueError(
            "rank is absent in RendezvousParameters."
            "Try add --node-rank to the cmd request"
        )  # 如果参数中没有包含"rank"键，则抛出值错误异常
    endpoint = params.endpoint.strip()  # 去除参数中端点字符串两侧的空白字符
    # 如果未提供 endpoint 参数，则抛出 ValueError 异常，指示 "RendezvousParameters" 中缺少 endpoint
    if not endpoint:
        raise ValueError(
            "endpoint is absent in RendezvousParameters"
            "Try add --master-port and --master-addr to the cmd request"
        )
    
    # 解析 endpoint 获取 master_addr 和 master_port
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, -1)
    
    # 如果解析出的 master_port 为 -1，则抛出 ValueError 异常，指示在 endpoint 中缺少端口信息
    if master_port == -1:
        raise ValueError(
            f"Port is absent in endpoint: {endpoint}. Try launching with --master-port"
        )
    
    # 获取并设置 world_size 为 params.max_nodes
    world_size = params.max_nodes
    
    # 从 params.config 中获取 "rank" 键对应的值，并将其转换为整数类型，赋值给 rank
    rank = cast(int, params.config.get("rank"))
    
    # 从 params 中获取 run_id
    run_id = params.run_id
    
    # 如果 params.config 中存在 "timeout" 键，则将其转换为整数类型并赋给 timeout；否则使用默认的 _default_timeout_seconds
    if "timeout" in params.config:
        timeout = int(params.config["timeout"])
    else:
        timeout = _default_timeout_seconds
    
    # 创建并返回 StaticTCPRendezvous 对象，传入 master_addr, master_port, rank, world_size, run_id, timeout 作为参数
    return StaticTCPRendezvous(
        master_addr, master_port, rank, world_size, run_id, timeout
    )
```