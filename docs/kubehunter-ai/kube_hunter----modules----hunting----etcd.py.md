# `.\kubehunter\kube_hunter\modules\hunting\etcd.py`

```

# 导入 logging 和 requests 模块
import logging
import requests

# 从 kube_hunter.conf 模块中导入 config
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability, Event, OpenPortEvent
from kube_hunter.core.events.types import Vulnerability, Event, OpenPortEvent
# 从 kube_hunter.core.types 模块中导入 ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure, RemoteCodeExec, UnauthenticatedAccess, AccessRisk
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

# 定义不同的漏洞类

# EtcdRemoteWriteAccessEvent 类，继承自 Vulnerability 和 Event
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

# EtcdRemoteReadAccessEvent 类，继承自 Vulnerability 和 Event
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

# EtcdRemoteVersionDisclosureEvent 类，继承自 Vulnerability 和 Event
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

# EtcdAccessEnabledWithoutAuthEvent 类，继承自 Vulnerability 和 Event
class EtcdAccessEnabledWithoutAuthEvent(Vulnerability, Event):
    """Etcd is accessible using HTTP (without authorization and authentication),
    it would allow a potential attacker to
     gain access to the etcd"""

    # 初始化方法
    def __init__(self, version):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Etcd is accessible using insecure connection (HTTP)",
            category=UnauthenticatedAccess,
            vid="KHV034",
        )
        # 设置 evidence 属性
        self.evidence = version

# ActiveHunter
# 订阅 OpenPortEvent 事件，当端口为 ETCD_PORT 时触发
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == ETCD_PORT)
class EtcdRemoteAccessActive(ActiveHunter):
    """Etcd Remote Access
    Checks for remote write access to etcd, will attempt to add a new key to the etcd DB"""

    # 初始化方法
    def __init__(self, event):
        self.event = event
        self.write_evidence = ""

    # 写入数据库键的访问方法
    def db_keys_write_access(self):
        logger.debug(f"Trying to write keys remotely on host {self.event.host}")
        data = {"value": "remotely written data"}
        try:
            r = requests.post(
                f"{self.protocol}://{self.event.host}:{ETCD_PORT}/v2/keys/message",
                data=data,
                timeout=config.network_timeout,
            )
            self.write_evidence = r.content if r.status_code == 200 and r.content else False
            return self.write_evidence
        except requests.exceptions.ConnectionError:
            return False

    # 执行方法
    def execute(self):
        if self.db_keys_write_access():
            self.publish_event(EtcdRemoteWriteAccessEvent(self.write_evidence))

# PassiveHunter
# 订阅 OpenPortEvent 事件，当端口为 ETCD_PORT 时触发
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == ETCD_PORT)
class EtcdRemoteAccess(Hunter):
    """Etcd Remote Access
    Checks for remote availability of etcd, its version, and read access to the DB
    """

    # 初始化方法
    def __init__(self, event):
        self.event = event
        self.version_evidence = ""
        self.keys_evidence = ""
        self.protocol = "https"

    # 读取数据库键的方法
    def db_keys_disclosure(self):
        logger.debug(f"{self.event.host} Passive hunter is attempting to read etcd keys remotely")
        try:
            r = requests.get(
                f"{self.protocol}://{self.eventhost}:{ETCD_PORT}/v2/keys", verify=False, timeout=config.network_timeout,
            )
            self.keys_evidence = r.content if r.status_code == 200 and r.content != "" else False
            return self.keys_evidence
        except requests.exceptions.ConnectionError:
            return False

    # 版本泄露的方法
    def version_disclosure(self):
        logger.debug(f"Trying to check etcd version remotely at {self.event.host}")
        try:
            r = requests.get(
                f"{self.protocol}://{self.event.host}:{ETCD_PORT}/version",
                verify=False,
                timeout=config.network_timeout,
            )
            self.version_evidence = r.content if r.status_code == 200 and r.content else False
            return self.version_evidence
        except requests.exceptions.ConnectionError:
            return False

    # 不安全访问的方法
    def insecure_access(self):
        logger.debug(f"Trying to access etcd insecurely at {self.event.host}")
        try:
            r = requests.get(
                f"http://{self.event.host}:{ETCD_PORT}/version", verify=False, timeout=config.network_timeout,
            )
            return r.content if r.status_code == 200 and r.content else False
        except requests.exceptions.ConnectionError:
            return False

    # 执行方法
    def execute(self):
        if self.insecure_access():  # make a decision between http and https protocol
            self.protocol = "http"
        if self.version_disclosure():
            self.publish_event(EtcdRemoteVersionDisclosureEvent(self.version_evidence))
            if self.protocol == "http":
                self.publish_event(EtcdAccessEnabledWithoutAuthEvent(self.version_evidence))
            if self.db_keys_disclosure():
                self.publish_event(EtcdRemoteReadAccessEvent(self.keys_evidence))

```