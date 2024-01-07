# `.\kubehunter\kube_hunter\modules\discovery\kubelet.py`

```

# 导入日志、请求、urllib3模块以及枚举类型
import logging
import requests
import urllib3
from enum import Enum

# 导入kube_hunter包中的配置、发现类型和事件处理器
from kube_hunter.conf import config
from kube_hunter.core.types import Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import OpenPortEvent, Event, Service

# 禁用urllib3的不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 获取logger对象
logger = logging.getLogger(__name__)

""" Services """

# 定义只读kubelet事件类，继承自Service和Event类
class ReadOnlyKubeletEvent(Service, Event):
    """The read-only port on the kubelet serves health probing endpoints,
    and is relied upon by many kubernetes components"""

    def __init__(self):
        # 初始化只读kubelet事件对象
        Service.__init__(self, name="Kubelet API (readonly)")

# 定义安全kubelet事件类，继承自Service和Event类
class SecureKubeletEvent(Service, Event):
    """The Kubelet is the main component in every Node, all pod operations goes through the kubelet"""

    def __init__(self, cert=False, token=False, anonymous_auth=True, **kwargs):
        # 初始化安全kubelet事件对象
        self.cert = cert
        self.token = token
        self.anonymous_auth = anonymous_auth
        Service.__init__(self, name="Kubelet API", **kwargs)

# 定义kubelet端口的枚举类型
class KubeletPorts(Enum):
    SECURED = 10250
    READ_ONLY = 10255

# 订阅OpenPortEvent事件，当端口为10250或10255时触发KubeletDiscovery事件
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port in [10250, 10255])
class KubeletDiscovery(Discovery):
    """Kubelet Discovery
    Checks for the existence of a Kubelet service, and its open ports
    """

    def __init__(self, event):
        # 初始化KubeletDiscovery对象
        self.event = event

    # 获取只读访问权限
    def get_read_only_access(self):
        endpoint = f"http://{self.event.host}:{self.event.port}/pods"
        logger.debug(f"Trying to get kubelet read access at {endpoint}")
        r = requests.get(endpoint, timeout=config.network_timeout)
        if r.status_code == 200:
            self.publish_event(ReadOnlyKubeletEvent())

    # 获取安全访问权限
    def get_secure_access(self):
        logger.debug("Attempting to get kubelet secure access")
        ping_status = self.ping_kubelet()
        if ping_status == 200:
            self.publish_event(SecureKubeletEvent(secure=False))
        elif ping_status == 403:
            self.publish_event(SecureKubeletEvent(secure=True))
        elif ping_status == 401:
            self.publish_event(SecureKubeletEvent(secure=True, anonymous_auth=False))

    # ping kubelet服务
    def ping_kubelet(self):
        endpoint = f"https://{self.event.host}:{self.event.port}/pods"
        logger.debug("Attempting to get pods info from kubelet")
        try:
            return requests.get(endpoint, verify=False, timeout=config.network_timeout).status_code
        except Exception:
            logger.debug(f"Failed pinging https port on {endpoint}", exc_info=True)

    # 执行事件处理
    def execute(self):
        if self.event.port == KubeletPorts.SECURED.value:
            self.get_secure_access()
        elif self.event.port == KubeletPorts.READ_ONLY.value:
            self.get_read_only_access()

```