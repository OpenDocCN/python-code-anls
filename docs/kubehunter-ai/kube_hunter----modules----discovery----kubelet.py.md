# `kubehunter\kube_hunter\modules\discovery\kubelet.py`

```
# 导入日志、请求和禁用警告的模块
import logging
import requests
import urllib3
from enum import Enum

# 导入自定义配置、发现类型和事件处理器
from kube_hunter.conf import config
from kube_hunter.core.types import Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import OpenPortEvent, Event, Service

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 获取日志记录器
logger = logging.getLogger(__name__)

""" Services """

# 定义只读的 Kubelet 事件
class ReadOnlyKubeletEvent(Service, Event):
    """The read-only port on the kubelet serves health probing endpoints,
    and is relied upon by many kubernetes components"""

    def __init__(self):
        Service.__init__(self, name="Kubelet API (readonly)")

# 定义安全的 Kubelet 事件
class SecureKubeletEvent(Service, Event):
    """The Kubelet is the main component in every Node, all pod operations goes through the kubelet"""

    def __init__(self, cert=False, token=False, anonymous_auth=True, **kwargs):
        self.cert = cert
        self.token = token
        self.anonymous_auth = anonymous_auth
        Service.__init__(self, name="Kubelet API", **kwargs)

# 定义 Kubelet 端口的枚举
class KubeletPorts(Enum):
    SECURED = 10250
    READ_ONLY = 10255

# 订阅端口打开事件，检查是否存在 Kubelet 服务和其开放的端口
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port in [10250, 10255])
class KubeletDiscovery(Discovery):
    """Kubelet Discovery
    Checks for the existence of a Kubelet service, and its open ports
    """

    def __init__(self, event):
        self.event = event

    # 获取只读访问权限
    def get_read_only_access(self):
        endpoint = f"http://{self.event.host}:{self.event.port}/pods"
        logger.debug(f"Trying to get kubelet read access at {endpoint}")
        r = requests.get(endpoint, timeout=config.network_timeout)
        if r.status_code == 200:
            self.publish_event(ReadOnlyKubeletEvent())
    # 获取安全访问权限的方法
    def get_secure_access(self):
        # 记录调试信息
        logger.debug("Attempting to get kubelet secure access")
        # 调用 ping_kubelet 方法获取响应状态码
        ping_status = self.ping_kubelet()
        # 根据响应状态码发布不同的事件
        if ping_status == 200:
            self.publish_event(SecureKubeletEvent(secure=False))
        elif ping_status == 403:
            self.publish_event(SecureKubeletEvent(secure=True))
        elif ping_status == 401:
            self.publish_event(SecureKubeletEvent(secure=True, anonymous_auth=False))

    # 发送请求到 kubelet 获取响应状态码的方法
    def ping_kubelet(self):
        # 构建请求的端点
        endpoint = f"https://{self.event.host}:{self.event.port}/pods"
        # 记录调试信息
        logger.debug("Attempting to get pods info from kubelet")
        try:
            # 发送请求并返回响应状态码
            return requests.get(endpoint, verify=False, timeout=config.network_timeout).status_code
        except Exception:
            # 记录异常信息
            logger.debug(f"Failed pinging https port on {endpoint}", exc_info=True)

    # 执行方法
    def execute(self):
        # 根据端口号调用不同的访问方法
        if self.event.port == KubeletPorts.SECURED.value:
            self.get_secure_access()
        elif self.event.port == KubeletPorts.READ_ONLY.value:
            self.get_read_only_access()
```