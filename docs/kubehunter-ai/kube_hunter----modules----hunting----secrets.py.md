# `kubehunter\kube_hunter\modules\hunting\secrets.py`

```py
# 导入 logging 模块
import logging
# 导入 os 模块
import os
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability, Event
from kube_hunter.core.events.types import Vulnerability, Event
# 从 kube_hunter.core.types 模块中导入 Hunter, KubernetesCluster, AccessRisk
from kube_hunter.core.types import Hunter, KubernetesCluster, AccessRisk
# 从 kube_hunter.modules.discovery.hosts 模块中导入 RunningAsPodEvent
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 ServiceAccountTokenAccess 类，继承自 Vulnerability, Event
class ServiceAccountTokenAccess(Vulnerability, Event):
    """ Accessing the pod service account token gives an attacker the option to use the server API """

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Read access to pod's service account token",
            category=AccessRisk,
            vid="KHV050",
        )
        # 设置 evidence 属性
        self.evidence = evidence

# 定义 SecretsAccess 类，继承自 Vulnerability, Event
class SecretsAccess(Vulnerability, Event):
    """ Accessing the pod's secrets within a compromised pod might disclose valuable data to a potential attacker"""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, component=KubernetesCluster, name="Access to pod's secrets", category=AccessRisk,
        )
        # 设置 evidence 属性
        self.evidence = evidence

# Passive Hunter
# 订阅 RunningAsPodEvent 事件
@handler.subscribe(RunningAsPodEvent)
# 定义 AccessSecrets 类，继承自 Hunter
class AccessSecrets(Hunter):
    """Access Secrets
    Accessing the secrets accessible to the pod"""

    # 初始化方法
    def __init__(self, event):
        # 设置 event 属性
        self.event = event
        # 初始化 secrets_evidence 属性
        self.secrets_evidence = ""

    # 获取服务方法
    def get_services(self):
        # 记录 debug 日志
        logger.debug("Trying to access pod's secrets directory")
        # 获取所有文件和子目录文件
        self.secrets_evidence = []
        for dirname, _, files in os.walk("/var/run/secrets/"):
            for f in files:
                self.secrets_evidence.append(os.path.join(dirname, f))
        # 如果 secrets_evidence 长度大于 0，则返回 True，否则返回 False
        return True if (len(self.secrets_evidence) > 0) else False
    # 执行方法，用于执行一系列操作
    def execute(self):
        # 如果事件的认证令牌不为空
        if self.event.auth_token is not None:
            # 发布事件，使用认证令牌访问服务账号令牌
            self.publish_event(ServiceAccountTokenAccess(self.event.auth_token))
        # 如果获取到服务
        if self.get_services():
            # 发布事件，使用秘密证据访问秘密
            self.publish_event(SecretsAccess(self.secrets_evidence))
```