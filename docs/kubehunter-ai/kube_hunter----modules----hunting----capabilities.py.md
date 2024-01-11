# `kubehunter\kube_hunter\modules\hunting\capabilities.py`

```
# 导入 socket 模块
import socket
# 导入 logging 模块
import logging

# 从 kube_hunter.modules.discovery.hosts 模块中导入 RunningAsPodEvent 类
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event, Vulnerability 类
from kube_hunter.core.events.types import Event, Vulnerability
# 从 kube_hunter.core.types 模块中导入 Hunter, AccessRisk, KubernetesCluster 类
from kube_hunter.core.types import Hunter, AccessRisk, KubernetesCluster

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)


# 定义 CapNetRawEnabled 类，继承自 Event, Vulnerability 类
class CapNetRawEnabled(Event, Vulnerability):
    """CAP_NET_RAW is enabled by default for pods.
    If an attacker manages to compromise a pod,
    they could potentially take advantage of this capability to perform network
    attacks on other pods running on the same node"""

    # 初始化方法
    def __init__(self):
        # 调用 Vulnerability 类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="CAP_NET_RAW Enabled", category=AccessRisk,
        )


# 使用 handler.subscribe 装饰器，订阅 RunningAsPodEvent 事件
@handler.subscribe(RunningAsPodEvent)
# 定义 PodCapabilitiesHunter 类，继承自 Hunter 类
class PodCapabilitiesHunter(Hunter):
    """Pod Capabilities Hunter
    Checks for default enabled capabilities in a pod
    """

    # 初始化方法
    def __init__(self, event):
        # 保存传入的 event 参数
        self.event = event

    # 检查是否具有 CAP_NET_RAW 权限的方法
    def check_net_raw(self):
        # 记录调试信息
        logger.debug("Passive hunter's trying to open a RAW socket")
        try:
            # 尝试打开一个原始套接字，如果没有 CAP_NET_RAW 权限会引发 PermissionsError
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)
            s.close()
            # 记录调试信息
            logger.debug("Passive hunter's closing RAW socket")
            return True
        except PermissionError:
            # 记录调试信息
            logger.debug("CAP_NET_RAW not enabled")

    # 执行方法
    def execute(self):
        # 如果具有 CAP_NET_RAW 权限，则发布 CapNetRawEnabled 事件
        if self.check_net_raw():
            self.publish_event(CapNetRawEnabled())
```