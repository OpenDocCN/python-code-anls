# `kubehunter\kube_hunter\modules\hunting\dashboard.py`

```
# 导入日志、JSON 和请求模块
import logging
import json
import requests

# 从 kube_hunter.conf 模块中导入配置
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入 Hunter、RemoteCodeExec 和 KubernetesCluster 类型
from kube_hunter.core.types import Hunter, RemoteCodeExec, KubernetesCluster
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability 和 Event 类型
from kube_hunter.core.events.types import Vulnerability, Event
# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardEvent 类
from kube_hunter.modules.discovery.dashboard import KubeDashboardEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 DashboardExposed 类，继承自 Vulnerability 和 Event 类
class DashboardExposed(Vulnerability, Event):
    """All operations on the cluster are exposed"""

    # 初始化方法，接受 nodes 参数
    def __init__(self, nodes):
        # 调用 Vulnerability 类的初始化方法，传入 KubernetesCluster 类型、"Dashboard Exposed" 字符串、category=RemoteCodeExec 和 vid="KHV029" 参数
        Vulnerability.__init__(
            self, KubernetesCluster, "Dashboard Exposed", category=RemoteCodeExec, vid="KHV029",
        )
# 设置 evidence 属性为 "nodes: " + nodes 列表的字符串形式，如果 nodes 不为空；否则设置为 None
self.evidence = "nodes: {}".format(" ".join(nodes)) if nodes else None

# 订阅 KubeDashboardEvent 事件，并定义 KubeDashboard 类
@handler.subscribe(KubeDashboardEvent)
class KubeDashboard(Hunter):
    """Dashboard Hunting
    Hunts open Dashboards, gets the type of nodes in the cluster
    """

    # 初始化方法，接收 event 参数
    def __init__(self, event):
        self.event = event

    # 获取集群中节点的类型
    def get_nodes(self):
        logger.debug("Passive hunter is attempting to get nodes types of the cluster")
        # 发送 GET 请求获取节点信息
        r = requests.get(f"http://{self.event.host}:{self.event.port}/api/v1/node", timeout=config.network_timeout)
        # 如果请求成功并且返回的文本中包含 "nodes"，则返回节点名称列表
        if r.status_code == 200 and "nodes" in r.text:
            return [node["objectMeta"]["name"] for node in json.loads(r.text)["nodes"]]

    # 执行方法，发布 DashboardExposed 事件
    def execute(self):
        self.publish_event(DashboardExposed(nodes=self.get_nodes()))
由于提供的代码为空，无法为其添加注释。
```