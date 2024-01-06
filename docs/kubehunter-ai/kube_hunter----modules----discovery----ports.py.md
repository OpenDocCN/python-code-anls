# `kubehunter\kube_hunter\modules\discovery\ports.py`

```
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

# 获取 logger 对象
logger = logging.getLogger(__name__)
# 定义默认端口列表
default_ports = [8001, 8080, 10250, 10255, 30000, 443, 6443, 2379]

# 订阅 NewHostEvent 事件
@handler.subscribe(NewHostEvent)
class PortDiscovery(Discovery):
    """Port Scanning
    Scans Kubernetes known ports to determine open endpoints for discovery
    """

    # 初始化函数，接收事件对象
    def __init__(self, event):
        self.event = event
        self.host = event.host
# 设置对象的端口属性为事件的端口
self.port = event.port

# 执行方法，尝试连接默认端口
def execute(self):
    # 记录日志，显示主机尝试的端口
    logger.debug(f"host {self.host} try ports: {default_ports}")
    # 遍历默认端口列表
    for single_port in default_ports:
        # 如果连接成功
        if self.test_connection(self.host, single_port):
            # 记录日志，显示可达的端口
            logger.debug(f"Reachable port found: {single_port}")
            # 发布开放端口事件
            self.publish_event(OpenPortEvent(port=single_port))

# 静态方法，用于测试主机和端口的连接
@staticmethod
def test_connection(host, port):
    # 创建套接字对象
    s = socket()
    # 设置超时时间为1.5秒
    s.settimeout(1.5)
    try:
        # 记录日志，显示正在扫描的主机和端口
        logger.debug(f"Scanning {host}:{port}")
        # 尝试连接主机和端口
        success = s.connect_ex((str(host), port))
        # 如果连接成功
        if success == 0:
            return True
    # 捕获异常
    except Exception:
        # 记录日志，显示探测失败的主机和端口
        logger.debug(f"Failed to probe {host}:{port}")

# 最终执行的代码块，无论是否发生异常都会执行，关闭文件或者网络连接
        finally:
            s.close()
        # 返回 False 作为函数的结果
```