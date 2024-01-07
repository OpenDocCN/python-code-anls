# `.\kubehunter\kube_hunter\modules\hunting\dashboard.py`

```

# 导入所需的模块
import logging
import json
import requests

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入 Hunter, RemoteCodeExec, KubernetesCluster 类
from kube_hunter.core.types import Hunter, RemoteCodeExec, KubernetesCluster
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability, Event 类
from kube_hunter.core.events.types import Vulnerability, Event
# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardEvent 类
from kube_hunter.modules.discovery.dashboard import KubeDashboardEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 DashboardExposed 类，继承自 Vulnerability 和 Event 类
class DashboardExposed(Vulnerability, Event):
    """All operations on the cluster are exposed"""

    def __init__(self, nodes):
        # 调用父类的构造函数
        Vulnerability.__init__(
            self, KubernetesCluster, "Dashboard Exposed", category=RemoteCodeExec, vid="KHV029",
        )
        # 设置 evidence 属性
        self.evidence = "nodes: {}".format(" ".join(nodes)) if nodes else None

# 使用 handler.subscribe 装饰器注册 KubeDashboardEvent 事件的处理函数
@handler.subscribe(KubeDashboardEvent)
class KubeDashboard(Hunter):
    """Dashboard Hunting
    Hunts open Dashboards, gets the type of nodes in the cluster
    """

    def __init__(self, event):
        self.event = event

    # 获取集群节点信息的方法
    def get_nodes(self):
        logger.debug("Passive hunter is attempting to get nodes types of the cluster")
        # 发起 HTTP GET 请求获取节点信息
        r = requests.get(f"http://{self.event.host}:{self.event.port}/api/v1/node", timeout=config.network_timeout)
        # 如果请求成功并且返回的文本中包含 "nodes" 字符串，则解析返回的 JSON 数据并提取节点名称
        if r.status_code == 200 and "nodes" in r.text:
            return [node["objectMeta"]["name"] for node in json.loads(r.text)["nodes"]]

    # 执行方法，发布 DashboardExposed 事件
    def execute(self):
        self.publish_event(DashboardExposed(nodes=self.get_nodes()))

```