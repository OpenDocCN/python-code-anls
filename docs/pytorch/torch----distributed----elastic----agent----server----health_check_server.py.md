# `.\pytorch\torch\distributed\elastic\agent\server\health_check_server.py`

```
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch.distributed.elastic.utils.logging import get_logger

# 获取 logger 对象
log = get_logger(__name__)

__all__ = ["HealthCheckServer", "create_healthcheck_server"]


class HealthCheckServer:
    """
    Interface for health check monitoring server, which can be extended
    by starting tcp/http server on the specified port.

    Args:

        alive_callback: Callable[[], int], callback to last progress time of agent

        port: int, port number to start tcp/http server

        timeout: int, timeout seconds to decide agent is alive/dead
    """

    _alive_callback: Callable[[], int]
    _port: int
    _timeout: int

    def __init__(
        self, alive_callback: Callable[[], int], port: int, timeout: int
    ) -> None:
        # 初始化 HealthCheckServer 实例的属性
        self._alive_callback = alive_callback
        self._port = port
        self._timeout = timeout

    def start(self) -> None:
        """
        Unsupported functionality for Pytorch, doesn't start any health check server
        """
        # 启动函数，此处仅记录警告信息，未启动健康检查服务器
        log.warning("No health check server started")

    def stop(self) -> None:
        """
        Function to stop health check server
        """
        # 停止函数，记录停止信息
        log.info("Stopping noop health check server.")


def create_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
) -> HealthCheckServer:
    """
    creates health check server object
    """
    # 创建并返回 HealthCheckServer 实例
    return HealthCheckServer(alive_callback, port, timeout)
```