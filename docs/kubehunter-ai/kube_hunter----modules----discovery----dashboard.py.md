# `kubehunter\kube_hunter\modules\discovery\dashboard.py`

```
# 导入所需的模块
import json
import logging
import requests

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event, OpenPortEvent, Service 类
from kube_hunter.core.events.types import Event, OpenPortEvent, Service
# 从 kube_hunter.core.types 模块中导入 Discovery 类
from kube_hunter.core.types import Discovery

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 KubeDashboardEvent 类，继承自 Service 和 Event 类
class KubeDashboardEvent(Service, Event):
    """A web-based Kubernetes user interface allows easy usage with operations on the cluster"""
    # 初始化方法
    def __init__(self, **kargs):
        # 调用父类的初始化方法，设置服务名称为 "Kubernetes Dashboard"
        Service.__init__(self, name="Kubernetes Dashboard", **kargs)

# 使用 handler.subscribe 装饰器，订阅 OpenPortEvent 事件，端口号为 30000
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port == 30000)
# 定义 KubeDashboard 类，继承自 Discovery 类
class KubeDashboard(Discovery):
    """K8s Dashboard Discovery
    Checks for the existence of a Dashboard
    """
    # 初始化方法
    def __init__(self, event):
        # 设置事件属性
        self.event = event

    # 定义 secure 属性
    @property
    def secure(self):
        # 构建请求的端点地址
        endpoint = f"http://{self.event.host}:{self.event.port}/api/v1/service/default"
        # 记录调试信息
        logger.debug("Attempting to discover an Api server to access dashboard")
        try:
            # 发起 GET 请求
            r = requests.get(endpoint, timeout=config.network_timeout)
            # 检查响应内容，判断是否安全
            if "listMeta" in r.text and len(json.loads(r.text)["errors"]) == 0:
                return False
        # 处理请求超时异常
        except requests.Timeout:
            logger.debug(f"failed getting {endpoint}", exc_info=True)
        # 默认返回安全
        return True

    # 定义 execute 方法
    def execute(self):
        # 如果不安全，则发布 KubeDashboardEvent 事件
        if not self.secure:
            self.publish_event(KubeDashboardEvent())
```