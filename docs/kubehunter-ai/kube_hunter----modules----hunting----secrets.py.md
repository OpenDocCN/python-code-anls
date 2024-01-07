# `.\kubehunter\kube_hunter\modules\hunting\secrets.py`

```

# 导入日志和操作系统模块
import logging
import os

# 导入事件处理和事件类型模块
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Vulnerability, Event
from kube_hunter.core.types import Hunter, KubernetesCluster, AccessRisk
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent

# 获取日志记录器
logger = logging.getLogger(__name__)


# 定义访问服务账户令牌的漏洞类
class ServiceAccountTokenAccess(Vulnerability, Event):
    """ Accessing the pod service account token gives an attacker the option to use the server API """

    def __init__(self, evidence):
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Read access to pod's service account token",
            category=AccessRisk,
            vid="KHV050",
        )
        self.evidence = evidence


# 定义访问秘密信息的漏洞类
class SecretsAccess(Vulnerability, Event):
    """ Accessing the pod's secrets within a compromised pod might disclose valuable data to a potential attacker"""

    def __init__(self, evidence):
        Vulnerability.__init__(
            self, component=KubernetesCluster, name="Access to pod's secrets", category=AccessRisk,
        )
        self.evidence = evidence


# 被动猎手
@handler.subscribe(RunningAsPodEvent)
class AccessSecrets(Hunter):
    """Access Secrets
    Accessing the secrets accessible to the pod"""

    def __init__(self, event):
        self.event = event
        self.secrets_evidence = ""

    # 获取服务
    def get_services(self):
        logger.debug("Trying to access pod's secrets directory")
        # 获取所有文件和子目录文件
        self.secrets_evidence = []
        for dirname, _, files in os.walk("/var/run/secrets/"):
            for f in files:
                self.secrets_evidence.append(os.path.join(dirname, f))
        return True if (len(self.secrets_evidence) > 0) else False

    # 执行操作
    def execute(self):
        if self.event.auth_token is not None:
            self.publish_event(ServiceAccountTokenAccess(self.event.auth_token))
        if self.get_services():
            self.publish_event(SecretsAccess(self.secrets_evidence))

```