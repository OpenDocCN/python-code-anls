# `kubehunter\kube_hunter\modules\hunting\etcd.py`

```
# 导入 logging 模块，用于记录日志
import logging
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability、Event、OpenPortEvent 类
from kube_hunter.core.events.types import Vulnerability, Event, OpenPortEvent
# 从 kube_hunter.core.types 模块中导入各种类型的类和对象
from kube_hunter.core.types import (
    ActiveHunter,
    Hunter,
    KubernetesCluster,
    InformationDisclosure,
    RemoteCodeExec,
    UnauthenticatedAccess,
    AccessRisk,
)
# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)
# 定义 ETCD_PORT 常量，赋值为 2379
ETCD_PORT = 2379
# 定义一个名为EtcdRemoteWriteAccessEvent的类，该类继承自Vulnerability和Event类
# 该类表示远程写访问可能会授予攻击者对kubernetes集群的完全控制
class EtcdRemoteWriteAccessEvent(Vulnerability, Event):
    # 类的文档字符串，描述了远程写访问可能会授予攻击者对kubernetes集群的完全控制
    """Remote write access might grant an attacker full control over the kubernetes cluster"""

    # 类的初始化方法，接受write_res参数
    def __init__(self, write_res):
        # 调用Vulnerability类的初始化方法，传入KubernetesCluster、name、category和vid参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Etcd Remote Write Access Event", category=RemoteCodeExec, vid="KHV031",
        )
        # 设置evidence属性为write_res
        self.evidence = write_res


# 定义一个名为EtcdRemoteReadAccessEvent的类，该类继承自Vulnerability和Event类
# 该类表示远程读访问可能会暴露给攻击者集群的可能漏洞、秘密等
class EtcdRemoteReadAccessEvent(Vulnerability, Event):
    # 类的文档字符串，描述了远程读访问可能会暴露给攻击者集群的可能漏洞、秘密等
    """Remote read access might expose to an attacker cluster's possible exploits, secrets and more."""

    # 类的初始化方法，接受keys参数
    def __init__(self, keys):
        # 调用Vulnerability类的初始化方法，传入KubernetesCluster、name、category和vid参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Etcd Remote Read Access Event", category=AccessRisk, vid="KHV032",
        )
# 定义一个类，表示Etcd远程版本泄露事件，可能会给攻击者提供攻击集群的有价值数据
class EtcdRemoteVersionDisclosureEvent(Vulnerability, Event):
    # 初始化函数，接受版本参数
    def __init__(self, version):
        # 调用父类的初始化函数，设置漏洞相关信息
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Etcd Remote version disclosure",
            category=InformationDisclosure,
            vid="KHV033",
        )
        # 设置evidence为版本参数
        self.evidence = version


# 定义一个类，表示Etcd访问未经授权的事件
class EtcdAccessEnabledWithoutAuthEvent(Vulnerability, Event):
    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数，设置漏洞相关信息
        Vulnerability.__init__(
            self,
            Etcd,
            name="Etcd Access Enabled Without Authorization",
            category=AccessControl,
            vid="KHV034",
        )
        # 设置evidence为空
        self.evidence = None
# 定义一个名为EtcdRemoteAccessActive的类，继承自ActiveHunter类，用于检查远程对etcd的访问权限
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == ETCD_PORT)
# 注册一个事件处理器，当OpenPortEvent事件发生且端口为ETCD_PORT时触发
class EtcdRemoteAccessActive(ActiveHunter):
    """Etcd Remote Access
    Checks for remote write access to etcd, will attempt to add a new key to the etcd DB"""
    # 检查远程对etcd的写访问权限，尝试向etcd数据库添加新的键
# 初始化方法，接受一个事件对象作为参数
def __init__(self, event):
    # 将事件对象赋值给实例变量
    self.event = event
    # 初始化写入证据为空字符串
    self.write_evidence = ""

# 检查数据库是否有写入权限的方法
def db_keys_write_access(self):
    # 记录调试信息，尝试在主机上远程写入密钥
    logger.debug(f"Trying to write keys remotely on host {self.event.host}")
    # 准备要发送的数据
    data = {"value": "remotely written data"}
    try:
        # 发送 POST 请求，尝试远程写入数据
        r = requests.post(
            f"{self.protocol}://{self.event.host}:{ETCD_PORT}/v2/keys/message",
            data=data,
            timeout=config.network_timeout,
        )
        # 如果请求成功并且有返回内容，将返回内容赋值给写入证据，否则赋值为 False
        self.write_evidence = r.content if r.status_code == 200 and r.content else False
        return self.write_evidence
    # 处理连接错误异常
    except requests.exceptions.ConnectionError:
        return False

# 执行方法
def execute(self):
    # 调用检查数据库写入权限的方法
    if self.db_keys_write_access():
# 发布 EtcdRemoteWriteAccessEvent 事件，传递写入证据
self.publish_event(EtcdRemoteWriteAccessEvent(self.write_evidence))

# Passive Hunter
# 订阅 OpenPortEvent 事件，端口为 ETCD_PORT 的事件
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == ETCD_PORT)
class EtcdRemoteAccess(Hunter):
    """Etcd Remote Access
    检查 etcd 的远程可用性、版本以及对数据库的读取权限
    """

    def __init__(self, event):
        self.event = event
        self.version_evidence = ""
        self.keys_evidence = ""
        self.protocol = "https"

    # 获取数据库键的泄露
    def db_keys_disclosure(self):
        logger.debug(f"{self.event.host} Passive hunter is attempting to read etcd keys remotely")
        try:
            # 发送 GET 请求，尝试读取 etcd 键
            r = requests.get(
# 发起 HTTP GET 请求，获取 etcd keys 的信息
def get_etcd_keys(self):
    # 尝试连接 etcd 服务器，获取 keys 的信息
    try:
        # 发起 HTTP GET 请求，获取 etcd keys 的信息
        r = requests.get(
            f"{self.protocol}://{self.eventhost}:{ETCD_PORT}/v2/keys", verify=False, timeout=config.network_timeout,
        )
        # 如果请求成功并且返回内容不为空，则将内容赋值给 self.keys_evidence，否则赋值为 False
        self.keys_evidence = r.content if r.status_code == 200 and r.content != "" else False
        # 返回获取到的 etcd keys 信息
        return self.keys_evidence
    # 如果连接失败，则返回 False
    except requests.exceptions.ConnectionError:
        return False

# 获取远程 etcd 版本信息
def version_disclosure(self):
    # 在日志中记录尝试远程检查 etcd 版本的操作
    logger.debug(f"Trying to check etcd version remotely at {self.event.host}")
    # 尝试连接 etcd 服务器，获取版本信息
    try:
        # 发起 HTTP GET 请求，获取 etcd 版本信息
        r = requests.get(
            f"{self.protocol}://{self.event.host}:{ETCD_PORT}/version",
            verify=False,
            timeout=config.network_timeout,
        )
        # 如果请求成功并且返回内容不为空，则将内容赋值给 self.version_evidence，否则赋值为 False
        self.version_evidence = r.content if r.status_code == 200 and r.content else False
        # 返回获取到的 etcd 版本信息
        return self.version_evidence
    # 如果连接失败，则返回 False
    except requests.exceptions.ConnectionError:
        return False
# 定义一个方法，用于不安全地访问 etcd
def insecure_access(self):
    # 记录调试信息，尝试不安全地访问 etcd
    logger.debug(f"Trying to access etcd insecurely at {self.event.host}")
    try:
        # 发起不安全的 HTTP 请求，获取 etcd 的版本信息
        r = requests.get(
            f"http://{self.event.host}:{ETCD_PORT}/version", verify=False, timeout=config.network_timeout,
        )
        # 如果请求成功且有返回内容，则返回内容，否则返回 False
        return r.content if r.status_code == 200 and r.content else False
    except requests.exceptions.ConnectionError:
        # 如果发生连接错误，则返回 False
        return False

# 定义一个方法，用于执行一系列操作
def execute(self):
    # 判断是否可以不安全地访问 etcd，决定使用 http 还是 https 协议
    if self.insecure_access():
        self.protocol = "http"
    # 获取 etcd 的版本信息
    if self.version_disclosure():
        # 发布 etcd 远程版本泄露事件
        self.publish_event(EtcdRemoteVersionDisclosureEvent(self.version_evidence))
        # 如果使用的是 http 协议，则发布 etcd 未经授权访问事件
        if self.protocol == "http":
            self.publish_event(EtcdAccessEnabledWithoutAuthEvent(self.version_evidence))
        # 获取 etcd 数据库的键信息
        if self.db_keys_disclosure():
            # 发布 etcd 远程读取访问事件
            self.publish_event(EtcdRemoteReadAccessEvent(self.keys_evidence))
```