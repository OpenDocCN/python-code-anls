# `kubehunter\kube_hunter\modules\discovery\ports.py`

```py
# 导入 logging 模块
import logging
# 从 socket 模块中导入 socket 类
from socket import socket

# 从 kube_hunter.core.types 模块中导入 Discovery 类
from kube_hunter.core.types import Discovery
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 NewHostEvent, OpenPortEvent 类
from kube_hunter.core.events.types import NewHostEvent, OpenPortEvent

# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)
# 定义默认的端口列表
default_ports = [8001, 8080, 10250, 10255, 30000, 443, 6443, 2379]

# 订阅 NewHostEvent 事件
@handler.subscribe(NewHostEvent)
class PortDiscovery(Discovery):
    """Port Scanning
    Scans Kubernetes known ports to determine open endpoints for discovery
    """

    # 初始化方法，接收事件对象
    def __init__(self, event):
        self.event = event
        self.host = event.host
        self.port = event.port

    # 执行方法
    def execute(self):
        # 记录调试信息
        logger.debug(f"host {self.host} try ports: {default_ports}")
        # 遍历默认端口列表
        for single_port in default_ports:
            # 调用 test_connection 方法测试连接
            if self.test_connection(self.host, single_port):
                # 如果连接成功，记录可达端口信息，并发布 OpenPortEvent 事件
                logger.debug(f"Reachable port found: {single_port}")
                self.publish_event(OpenPortEvent(port=single_port))

    # 静态方法，用于测试连接
    @staticmethod
    def test_connection(host, port):
        # 创建 socket 对象
        s = socket()
        # 设置超时时间
        s.settimeout(1.5)
        try:
            # 记录调试信息
            logger.debug(f"Scanning {host}:{port}")
            # 尝试连接主机和端口
            success = s.connect_ex((str(host), port))
            # 如果连接成功，返回 True
            if success == 0:
                return True
        except Exception:
            # 如果连接失败，记录调试信息
            logger.debug(f"Failed to probe {host}:{port}")
        finally:
            # 关闭 socket 连接
            s.close()
        # 返回 False
        return False
```