# `kubehunter\kube_hunter\modules\hunting\capabilities.py`

```
# 导入所需的模块
import socket  # 用于网络通信
import logging  # 用于记录日志

# 导入所需的类和函数
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Event, Vulnerability
from kube_hunter.core.types import Hunter, AccessRisk, KubernetesCluster

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义一个事件和漏洞类，表示 CAP_NET_RAW 功能默认启用
class CapNetRawEnabled(Event, Vulnerability):
    """CAP_NET_RAW is enabled by default for pods.
    If an attacker manages to compromise a pod,
    they could potentially take advantage of this capability to perform network
    attacks on other pods running on the same node"""

    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="CAP_NET_RAW Enabled", category=AccessRisk,
# 订阅 RunningAsPodEvent 事件，表示该类是一个事件处理器，用于检查 Pod 的能力
@handler.subscribe(RunningAsPodEvent)
class PodCapabilitiesHunter(Hunter):
    """Pod Capabilities Hunter
    Checks for default enabled capabilities in a pod
    """

    def __init__(self, event):
        self.event = event

    # 检查是否具有 net_raw 能力
    def check_net_raw(self):
        logger.debug("Passive hunter's trying to open a RAW socket")
        try:
            # 尝试打开一个原始套接字，如果没有 CAP_NET_RAW 权限会引发 PermissionsError
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)
            s.close()
            logger.debug("Passive hunter's closing RAW socket")
            return True
# 捕获权限错误异常
        except PermissionError:
            # 如果捕获到权限错误异常，则记录调试信息
            logger.debug("CAP_NET_RAW not enabled")

    # 执行函数
    def execute(self):
        # 检查是否具有原始网络访问权限
        if self.check_net_raw():
            # 如果具有原始网络访问权限，则发布 CapNetRawEnabled 事件
            self.publish_event(CapNetRawEnabled())
```