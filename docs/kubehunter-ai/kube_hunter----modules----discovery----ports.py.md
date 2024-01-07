# `.\kubehunter\kube_hunter\modules\discovery\ports.py`

```

# 导入日志模块和套接字模块
import logging
from socket import socket

# 导入 kube_hunter 中的类型和事件相关模块
from kube_hunter.core.types import Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import NewHostEvent, OpenPortEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)
# 默认需要扫描的端口列表
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

    # 执行端口扫描
    def execute(self):
        logger.debug(f"host {self.host} try ports: {default_ports}")
        # 遍历默认端口列表
        for single_port in default_ports:
            # 测试连接是否成功
            if self.test_connection(self.host, single_port):
                logger.debug(f"Reachable port found: {single_port}")
                # 发布 OpenPortEvent 事件
                self.publish_event(OpenPortEvent(port=single_port))

    # 静态方法，用于测试主机和端口的连接
    @staticmethod
    def test_connection(host, port):
        s = socket()
        s.settimeout(1.5)
        try:
            logger.debug(f"Scanning {host}:{port}")
            success = s.connect_ex((str(host), port))
            if success == 0:
                return True
        except Exception:
            logger.debug(f"Failed to probe {host}:{port}")
        finally:
            s.close()
        return False

```