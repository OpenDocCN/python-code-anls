# `kubehunter\kube_hunter\modules\hunting\etcd.py`

```py
# 导入 logging 模块
import logging
# 导入 requests 模块
import requests
# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability, Event, OpenPortEvent 类
from kube_hunter.core.events.types import Vulnerability, Event, OpenPortEvent
# 从 kube_hunter.core.types 模块中导入 ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure, RemoteCodeExec, UnauthenticatedAccess, AccessRisk 类
from kube_hunter.core.types import (
    ActiveHunter,
    Hunter,
    KubernetesCluster,
    InformationDisclosure,
    RemoteCodeExec,
    UnauthenticatedAccess,
    AccessRisk,
)
# 获取 logger 对象
logger = logging.getLogger(__name__)
# 设置 ETCD_PORT 常量为 2379
ETCD_PORT = 2379

""" Vulnerabilities """

# 定义 EtcdRemoteWriteAccessEvent 类，继承自 Vulnerability 和 Event 类
class EtcdRemoteWriteAccessEvent(Vulnerability, Event):
    """Remote write access might grant an attacker full control over the kubernetes cluster"""

    # 初始化方法
    def __init__(self, write_res):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="Etcd Remote Write Access Event", category=RemoteCodeExec, vid="KHV031",
        )
        # 设置 evidence 属性
        self.evidence = write_res

# 定义 EtcdRemoteReadAccessEvent 类，继承自 Vulnerability 和 Event 类
class EtcdRemoteReadAccessEvent(Vulnerability, Event):
    """Remote read access might expose to an attacker cluster's possible exploits, secrets and more."""

    # 初始化方法
    def __init__(self, keys):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="Etcd Remote Read Access Event", category=AccessRisk, vid="KHV032",
        )
        # 设置 evidence 属性
        self.evidence = keys

# 定义 EtcdRemoteVersionDisclosureEvent 类，继承自 Vulnerability 和 Event 类
class EtcdRemoteVersionDisclosureEvent(Vulnerability, Event):
    """Remote version disclosure might give an attacker a valuable data to attack a cluster"""

    # 初始化方法
    def __init__(self, version):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Etcd Remote version disclosure",
            category=InformationDisclosure,
            vid="KHV033",
        )
        # 设置 evidence 属性
        self.evidence = version

# 定义 EtcdAccessEnabledWithoutAuthEvent 类，继承自 Vulnerability 和 Event 类
class EtcdAccessEnabledWithoutAuthEvent(Vulnerability, Event):
    """Etcd is accessible using HTTP (without authorization and authentication),
    it would allow a potential attacker to
     gain access to the etcd"""
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, version):
        # 调用父类Vulnerability的初始化函数，传入当前对象实例、KubernetesCluster类、漏洞名称、漏洞类别、漏洞ID
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Etcd is accessible using insecure connection (HTTP)",
            category=UnauthenticatedAccess,
            vid="KHV034",
        )
        # 将传入的版本信息赋给当前对象的evidence属性
        self.evidence = version
# Active Hunter
# 订阅 OpenPortEvent 事件，当端口为 ETCD_PORT 时触发，执行 EtcdRemoteAccessActive 类
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == ETCD_PORT)
class EtcdRemoteAccessActive(ActiveHunter):
    """Etcd Remote Access
    Checks for remote write access to etcd, will attempt to add a new key to the etcd DB"""

    def __init__(self, event):
        self.event = event
        self.write_evidence = ""

    # 检查是否具有对数据库键的写入权限
    def db_keys_write_access(self):
        logger.debug(f"Trying to write keys remotely on host {self.event.host}")
        data = {"value": "remotely written data"}
        try:
            # 尝试向 etcd 数据库添加新的键
            r = requests.post(
                f"{self.protocol}://{self.event.host}:{ETCD_PORT}/v2/keys/message",
                data=data,
                timeout=config.network_timeout,
            )
            # 如果请求成功并且返回内容不为空，则将内容存储在 write_evidence 中
            self.write_evidence = r.content if r.status_code == 200 and r.content else False
            return self.write_evidence
        except requests.exceptions.ConnectionError:
            return False

    # 执行函数
    def execute(self):
        # 如果具有对数据库键的写入权限，则发布 EtcdRemoteWriteAccessEvent 事件
        if self.db_keys_write_access():
            self.publish_event(EtcdRemoteWriteAccessEvent(self.write_evidence))


# Passive Hunter
# 订阅 OpenPortEvent 事件，当端口为 ETCD_PORT 时触发，执行 EtcdRemoteAccess 类
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == ETCD_PORT)
class EtcdRemoteAccess(Hunter):
    """Etcd Remote Access
    Checks for remote availability of etcd, its version, and read access to the DB
    """

    def __init__(self, event):
        self.event = event
        self.version_evidence = ""
        self.keys_evidence = ""
        self.protocol = "https"
    # 暴露数据库键信息的方法
    def db_keys_disclosure(self):
        # 记录调试信息，Passive hunter 尝试远程读取 etcd 键
        logger.debug(f"{self.event.host} Passive hunter is attempting to read etcd keys remotely")
        try:
            # 发送 GET 请求，获取 etcd 键的内容
            r = requests.get(
                f"{self.protocol}://{self.eventhost}:{ETCD_PORT}/v2/keys", verify=False, timeout=config.network_timeout,
            )
            # 如果状态码为 200 并且内容不为空，则将内容赋值给 self.keys_evidence，否则赋值为 False
            self.keys_evidence = r.content if r.status_code == 200 and r.content != "" else False
            return self.keys_evidence
        except requests.exceptions.ConnectionError:
            return False

    # 暴露版本信息的方法
    def version_disclosure(self):
        # 记录调试信息，尝试远程检查 etcd 版本
        logger.debug(f"Trying to check etcd version remotely at {self.event.host}")
        try:
            # 发送 GET 请求，获取 etcd 版本信息
            r = requests.get(
                f"{self.protocol}://{self.event.host}:{ETCD_PORT}/version",
                verify=False,
                timeout=config.network_timeout,
            )
            # 如果状态码为 200 并且内容不为空，则将内容赋值给 self.version_evidence，否则赋值为 False
            self.version_evidence = r.content if r.status_code == 200 and r.content else False
            return self.version_evidence
        except requests.exceptions.ConnectionError:
            return False

    # 不安全访问的方法
    def insecure_access(self):
        # 记录调试信息，尝试不安全访问 etcd
        logger.debug(f"Trying to access etcd insecurely at {self.event.host}")
        try:
            # 发送不安全的 GET 请求，获取 etcd 版本信息
            r = requests.get(
                f"http://{self.event.host}:{ETCD_PORT}/version", verify=False, timeout=config.network_timeout,
            )
            # 如果状态码为 200 并且内容不为空，则返回内容，否则返回 False
            return r.content if r.status_code == 200 and r.content else False
        except requests.exceptions.ConnectionError:
            return False
    # 执行函数，用于执行一系列安全检查
    def execute(self):
        # 如果存在不安全的访问方式，则使用http协议
        if self.insecure_access():  # make a decision between http and https protocol
            self.protocol = "http"
        # 如果存在版本泄露，则发布相应的事件
        if self.version_disclosure():
            self.publish_event(EtcdRemoteVersionDisclosureEvent(self.version_evidence))
            # 如果使用http协议，则发布未经授权访问事件
            if self.protocol == "http":
                self.publish_event(EtcdAccessEnabledWithoutAuthEvent(self.version_evidence))
            # 如果存在数据库键值泄露，则发布远程读取访问事件
            if self.db_keys_disclosure():
                self.publish_event(EtcdRemoteReadAccessEvent(self.keys_evidence))
```