# `kubehunter\kube_hunter\modules\discovery\proxy.py`

```py
# 导入 logging 模块
import logging
# 导入 requests 模块
import requests

# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入 Discovery 类
from kube_hunter.core.types import Discovery
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Service、Event、OpenPortEvent 类
from kube_hunter.core.events.types import Service, Event, OpenPortEvent

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)


# 定义 KubeProxyEvent 类，继承自 Event 和 Service 类
class KubeProxyEvent(Event, Service):
    """proxies from a localhost address to the Kubernetes apiserver"""
    # 初始化方法
    def __init__(self):
        # 调用父类 Service 的初始化方法，设置服务名称为 "Kubernetes Proxy"
        Service.__init__(self, name="Kubernetes Proxy")


# 使用 handler.subscribe 装饰器，订阅 OpenPortEvent 事件，当端口号为 8001 时触发
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port == 8001)
# 定义 KubeProxy 类，继承自 Discovery 类
class KubeProxy(Discovery):
    """Proxy Discovery
    Checks for the existence of a an open Proxy service
    """
    # 初始化方法
    def __init__(self, event):
        # 设置事件对象
        self.event = event
        # 设置主机地址
        self.host = event.host
        # 设置端口号，默认为 8001
        self.port = event.port or 8001

    # 定义 accesible 属性
    @property
    def accesible(self):
        # 构建请求的端点地址
        endpoint = f"http://{self.host}:{self.port}/api/v1"
        # 记录调试日志
        logger.debug("Attempting to discover a proxy service")
        try:
            # 发起 GET 请求
            r = requests.get(endpoint, timeout=config.network_timeout)
            # 如果响应状态码为 200 并且响应文本中包含 "APIResourceList" 字符串
            if r.status_code == 200 and "APIResourceList" in r.text:
                # 返回 True
                return True
        # 处理请求超时异常
        except requests.Timeout:
            # 记录调试日志，包括异常信息
            logger.debug(f"failed to get {endpoint}", exc_info=True)
        # 返回 False
        return False

    # 定义 execute 方法
    def execute(self):
        # 如果服务可访问
        if self.accesible:
            # 发布 KubeProxyEvent 事件
            self.publish_event(KubeProxyEvent())
```