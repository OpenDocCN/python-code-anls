# `.\kubehunter\kube_hunter\modules\hunting\capabilities.py`

```

# 导入所需的模块
import socket  # 用于网络通信
import logging  # 用于记录日志

# 导入自定义模块
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent  # 用于发现运行在 Pod 中的主机
from kube_hunter.core.events import handler  # 用于处理事件
from kube_hunter.core.events.types import Event, Vulnerability  # 用于定义事件和漏洞类型
from kube_hunter.core.types import Hunter, AccessRisk, KubernetesCluster  # 用于定义 Hunter 和 Kubernetes 集群

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义一个事件和漏洞类型，表示 CAP_NET_RAW 默认启用
class CapNetRawEnabled(Event, Vulnerability):
    """CAP_NET_RAW is enabled by default for pods.
    If an attacker manages to compromise a pod,
    they could potentially take advantage of this capability to perform network
    attacks on other pods running on the same node"""

    def __init__(self):
        Vulnerability.__init__(
            self, KubernetesCluster, name="CAP_NET_RAW Enabled", category=AccessRisk,
        )

# 订阅 RunningAsPodEvent 事件的处理函数
@handler.subscribe(RunningAsPodEvent)
class PodCapabilitiesHunter(Hunter):
    """Pod Capabilities Hunter
    Checks for default enabled capabilities in a pod
    """

    def __init__(self, event):
        self.event = event

    # 检查是否启用了 CAP_NET_RAW
    def check_net_raw(self):
        logger.debug("Passive hunter's trying to open a RAW socket")
        try:
            # 尝试打开一个原始套接字，如果没有 CAP_NET_RAW 权限会引发 PermissionsError
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)
            s.close()
            logger.debug("Passive hunter's closing RAW socket")
            return True
        except PermissionError:
            logger.debug("CAP_NET_RAW not enabled")

    # 执行检查
    def execute(self):
        if self.check_net_raw():
            self.publish_event(CapNetRawEnabled())

```