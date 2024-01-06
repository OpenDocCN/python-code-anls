# `kubehunter\kube_hunter\modules\discovery\proxy.py`

```
# 导入 logging 和 requests 模块
import logging
import requests

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入 Discovery 类和 Service 类
from kube_hunter.core.types import Discovery
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Service 类、Event 类和 OpenPortEvent 类
from kube_hunter.core.events.types import Service, Event, OpenPortEvent

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)

# 定义 KubeProxyEvent 类，继承自 Event 类和 Service 类
class KubeProxyEvent(Event, Service):
    """proxies from a localhost address to the Kubernetes apiserver"""

    # 初始化方法
    def __init__(self):
        # 调用父类 Service 的初始化方法，设置 name 属性为 "Kubernetes Proxy"
        Service.__init__(self, name="Kubernetes Proxy")

# 使用 handler.subscribe 装饰器注册事件处理函数，当 OpenPortEvent 事件的端口号为 8001 时触发
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port == 8001)
# 定义 KubeProxy 类，继承自 Discovery 类
class KubeProxy(Discovery):
# 代理发现
# 检查是否存在开放的代理服务

# 初始化方法，接收一个事件对象作为参数
def __init__(self, event):
    # 将事件对象保存为实例属性
    self.event = event
    # 获取事件对象的主机名
    self.host = event.host
    # 如果端口不存在，则默认使用8001端口
    self.port = event.port or 8001

# 装饰器，将方法转换为属性
@property
# 检查代理服务是否可访问
def accesible(self):
    # 构建代理服务的地址
    endpoint = f"http://{self.host}:{self.port}/api/v1"
    # 记录调试信息
    logger.debug("Attempting to discover a proxy service")
    try:
        # 发起GET请求，设置超时时间为配置文件中的网络超时时间
        r = requests.get(endpoint, timeout=config.network_timeout)
        # 如果响应状态码为200且响应文本中包含"APIResourceList"，则代理服务可访问
        if r.status_code == 200 and "APIResourceList" in r.text:
            return True
    # 处理请求超时异常
    except requests.Timeout:
        logger.debug(f"failed to get {endpoint}", exc_info=True)
    # 代理服务不可访问，返回False
    return False
# 执行函数，用于执行特定操作
def execute(self):
    # 检查是否可访问
    if self.accesible:
        # 如果可访问，则发布 KubeProxyEvent 事件
        self.publish_event(KubeProxyEvent())
```