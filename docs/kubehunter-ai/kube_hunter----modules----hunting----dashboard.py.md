# `kubehunter\kube_hunter\modules\hunting\dashboard.py`

```
# 导入日志、JSON 和请求模块
import logging
import json
import requests

# 从 kube_hunter.conf 模块中导入配置
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入 Hunter、RemoteCodeExec 和 KubernetesCluster 类
from kube_hunter.core.types import Hunter, RemoteCodeExec, KubernetesCluster
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability 和 Event 类
from kube_hunter.core.events.types import Vulnerability, Event
# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardEvent
from kube_hunter.modules.discovery.dashboard import KubeDashboardEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)


# 定义 DashboardExposed 类，继承自 Vulnerability 和 Event 类
class DashboardExposed(Vulnerability, Event):
    """All operations on the cluster are exposed"""

    # 初始化方法
    def __init__(self, nodes):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Dashboard Exposed", category=RemoteCodeExec, vid="KHV029",
        )
        # 设置 evidence 属性
        self.evidence = "nodes: {}".format(" ".join(nodes)) if nodes else None


# 使用 handler.subscribe 装饰器订阅 KubeDashboardEvent 事件
@handler.subscribe(KubeDashboardEvent)
# 定义 KubeDashboard 类，继承自 Hunter 类
class KubeDashboard(Hunter):
    """Dashboard Hunting
    Hunts open Dashboards, gets the type of nodes in the cluster
    """

    # 初始化方法
    def __init__(self, event):
        # 设置 event 属性
        self.event = event

    # 获取节点类型的方法
    def get_nodes(self):
        # 记录调试日志
        logger.debug("Passive hunter is attempting to get nodes types of the cluster")
        # 发送 GET 请求获取节点类型
        r = requests.get(f"http://{self.event.host}:{self.event.port}/api/v1/node", timeout=config.network_timeout)
        # 如果响应状态码为 200 并且响应文本中包含 "nodes"
        if r.status_code == 200 and "nodes" in r.text:
            # 返回节点名称列表
            return [node["objectMeta"]["name"] for node in json.loads(r.text)["nodes"]

    # 执行方法
    def execute(self):
        # 发布 DashboardExposed 事件
        self.publish_event(DashboardExposed(nodes=self.get_nodes()))
```