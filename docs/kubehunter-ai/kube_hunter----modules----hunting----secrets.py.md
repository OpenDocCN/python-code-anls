# `kubehunter\kube_hunter\modules\hunting\secrets.py`

```
# 导入日志和操作系统模块
import logging
import os

# 导入事件处理器和事件类型
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Vulnerability, Event
# 导入类型和风险类型
from kube_hunter.core.types import Hunter, KubernetesCluster, AccessRisk
# 导入运行为 Pod 事件模块
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义 ServiceAccountTokenAccess 类，继承自 Vulnerability 和 Event 类
class ServiceAccountTokenAccess(Vulnerability, Event):
    """ Accessing the pod service account token gives an attacker the option to use the server API """

    # 初始化方法，接受 evidence 参数
    def __init__(self, evidence):
        # 调用 Vulnerability 类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Read access to pod's service account token",
            category=AccessRisk,
# 定义一个名为SecretsAccess的类，继承自Vulnerability和Event类，表示在受损的pod中访问pod的秘密可能会向潜在攻击者披露有价值的数据
class SecretsAccess(Vulnerability, Event):
    # 初始化方法，接受evidence作为参数
    def __init__(self, evidence):
        # 调用Vulnerability类的初始化方法，设置组件为KubernetesCluster，名称为Access to pod's secrets，类别为AccessRisk
        Vulnerability.__init__(
            self, component=KubernetesCluster, name="Access to pod's secrets", category=AccessRisk,
        )
        # 设置evidence属性
        self.evidence = evidence


# Passive Hunter
# 订阅RunningAsPodEvent事件的处理程序
@handler.subscribe(RunningAsPodEvent)
class AccessSecrets(Hunter):
    # 访问秘密
    """Access Secrets
    Accessing the secrets accessible to the pod"""
```

# 初始化方法，接收一个事件对象作为参数
def __init__(self, event):
    # 将事件对象保存在实例变量中
    self.event = event
    # 初始化一个空字符串，用于保存秘密证据
    self.secrets_evidence = ""

# 获取服务方法
def get_services(self):
    # 记录调试信息
    logger.debug("Trying to access pod's secrets directory")
    # 获取指定目录下的所有文件和子目录
    self.secrets_evidence = []
    for dirname, _, files in os.walk("/var/run/secrets/"):
        for f in files:
            # 将文件的完整路径添加到秘密证据列表中
            self.secrets_evidence.append(os.path.join(dirname, f))
    # 如果秘密证据列表不为空，则返回True，否则返回False
    return True if (len(self.secrets_evidence) > 0) else False

# 执行方法
def execute(self):
    # 如果事件的认证令牌不为空，则发布ServiceAccountTokenAccess事件
    if self.event.auth_token is not None:
        self.publish_event(ServiceAccountTokenAccess(self.event.auth_token))
    # 如果成功获取服务，则发布SecretsAccess事件，并将秘密证据作为参数
    if self.get_services():
        self.publish_event(SecretsAccess(self.secrets_evidence))
```