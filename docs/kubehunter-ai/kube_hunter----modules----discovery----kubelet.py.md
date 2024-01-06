# `kubehunter\kube_hunter\modules\discovery\kubelet.py`

```
# 导入日志、请求、urllib3、枚举模块
import logging
import requests
import urllib3
from enum import Enum

# 从kube_hunter.conf模块导入config
from kube_hunter.conf import config
# 从kube_hunter.core.types模块导入Discovery
from kube_hunter.core.types import Discovery
# 从kube_hunter.core.events模块导入handler
from kube_hunter.core.events import handler
# 从kube_hunter.core.events.types模块导入OpenPortEvent, Event, Service
from kube_hunter.core.events.types import OpenPortEvent, Event, Service

# 禁用urllib3的不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

""" Services """

# 定义ReadOnlyKubeletEvent类，继承自Service和Event类
class ReadOnlyKubeletEvent(Service, Event):
    """The read-only port on the kubelet serves health probing endpoints,
    and is relied upon by many kubernetes components"""
```
以上代码中，我们导入了一些模块，并定义了一个类。同时，我们禁用了urllib3的不安全请求警告。
# 初始化函数，设置服务名称为“Kubelet API (readonly)”
def __init__(self):
    Service.__init__(self, name="Kubelet API (readonly)")

# 安全的Kubelet事件类，继承自Service和Event类
class SecureKubeletEvent(Service, Event):
    """The Kubelet is the main component in every Node, all pod operations goes through the kubelet"""

    # 初始化函数，设置证书、令牌、匿名认证等属性
    def __init__(self, cert=False, token=False, anonymous_auth=True, **kwargs):
        self.cert = cert
        self.token = token
        self.anonymous_auth = anonymous_auth
        # 调用父类Service的初始化函数，设置服务名称为“Kubelet API”
        Service.__init__(self, name="Kubelet API", **kwargs)

# Kubelet端口枚举类
class KubeletPorts(Enum):
    SECURED = 10250
    READ_ONLY = 10255

# 订阅OpenPortEvent事件，当端口为10250或10255时触发
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port in [10250, 10255])
# 定义一个 KubeletDiscovery 类，继承自 Discovery 类
# 用于检查 Kubelet 服务的存在以及其开放的端口
class KubeletDiscovery(Discovery):
    """Kubelet Discovery
    Checks for the existence of a Kubelet service, and its open ports
    """

    # 初始化方法，接收一个 event 参数
    def __init__(self, event):
        self.event = event

    # 获取只读访问权限的方法
    def get_read_only_access(self):
        # 构建访问 kubelet 的端点地址
        endpoint = f"http://{self.event.host}:{self.event.port}/pods"
        # 记录调试日志
        logger.debug(f"Trying to get kubelet read access at {endpoint}")
        # 发起 GET 请求，获取只读访问权限
        r = requests.get(endpoint, timeout=config.network_timeout)
        # 如果返回状态码为 200，则发布只读 kubelet 事件
        if r.status_code == 200:
            self.publish_event(ReadOnlyKubeletEvent())

    # 获取安全访问权限的方法
    def get_secure_access(self):
        # 记录调试日志
        logger.debug("Attempting to get kubelet secure access")
        # 检查 kubelet 的 ping 状态
        ping_status = self.ping_kubelet()
        # 如果 ping 状态为 200，则发布安全 kubelet 事件
        if ping_status == 200:
            self.publish_event(SecureKubeletEvent(secure=False))
# 如果 kubelet 返回状态码为 403，则发布一个安全 kubelet 事件
        elif ping_status == 403:
            self.publish_event(SecureKubeletEvent(secure=True))
# 如果 kubelet 返回状态码为 401，则发布一个安全 kubelet 事件，匿名认证为 false
        elif ping_status == 401:
            self.publish_event(SecureKubeletEvent(secure=True, anonymous_auth=False)

# 发送请求到 kubelet 的端点，获取 pod 信息
    def ping_kubelet(self):
        # 构建 kubelet 端点的 URL
        endpoint = f"https://{self.event.host}:{self.event.port}/pods"
        logger.debug("Attempting to get pods info from kubelet")
        try:
            # 发送 GET 请求到 kubelet 端点，关闭 SSL 验证，设置超时时间
            return requests.get(endpoint, verify=False, timeout=config.network_timeout).status_code
        except Exception:
            # 如果请求失败，记录错误日志
            logger.debug(f"Failed pinging https port on {endpoint}", exc_info=True)

# 根据事件的端口号执行相应的操作
    def execute(self):
        # 如果事件的端口号为安全端口，则获取安全访问
        if self.event.port == KubeletPorts.SECURED.value:
            self.get_secure_access()
        # 如果事件的端口号为只读端口，则获取只读访问
        elif self.event.port == KubeletPorts.READ_ONLY.value:
            self.get_read_only_access()
```