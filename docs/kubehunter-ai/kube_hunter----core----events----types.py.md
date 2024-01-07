# `.\kubehunter\kube_hunter\core\events\types.py`

```

# 导入所需的模块
import threading
import requests
import logging

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入各种类型
from kube_hunter.core.types import (
    InformationDisclosure,
    DenialOfService,
    RemoteCodeExec,
    IdentityTheft,
    PrivilegeEscalation,
    AccessRisk,
    UnauthenticatedAccess,
    KubernetesCluster,
)

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义事件过滤器基类
class EventFilterBase(object):
    def __init__(self, event):
        self.event = event

    # 默认返回 self.event
    # 如果有更改，应返回已更改的新事件
    # 返回 None 表示应丢弃事件
    def execute(self):
        return self.event

# 定义事件类
class Event(object):
    def __init__(self):
        self.previous = None
        self.hunter = None

    # 获取属性的方法
    def __getattr__(self, name):
        if name == "previous":
            return None
        for event in self.history:
            if name in event.__dict__:
                return event.__dict__[name]

    # 事件的逻辑位置，主要用于报告
    def location(self):
        location = None
        if self.previous:
            location = self.previous.location()

        return location

    # 返回按时间顺序排列的事件历史
    @property
    def history(self):
        previous, history = self.previous, list()
        while previous:
            history.append(previous)
            previous = previous.previous
        return history

# 定义服务类
class Service(object):
    def __init__(self, name, path="", secure=True):
        self.name = name
        self.secure = secure
        self.path = path
        self.role = "Node"

    def get_name(self):
        return self.name

    def get_path(self):
        return "/" + self.path if self.path else ""

    def explain(self):
        return self.__doc__

# 定义漏洞类
class Vulnerability(object):
    severity = dict(
        {
            InformationDisclosure: "medium",
            DenialOfService: "medium",
            RemoteCodeExec: "high",
            IdentityTheft: "high",
            PrivilegeEscalation: "high",
            AccessRisk: "low",
            UnauthenticatedAccess: "low",
        }
    )

    # 初始化漏洞对象
    def __init__(self, component, name, category=None, vid="None"):
        self.vid = vid
        self.component = component
        self.category = category
        self.name = name
        self.evidence = ""
        self.role = "Node"

    def get_vid(self):
        return self.vid

    def get_category(self):
        if self.category:
            return self.category.name

    def get_name(self):
        return self.name

    def explain(self):
        return self.__doc__

    def get_severity(self):
        return self.severity.get(self.category, "low")

# 定义全局变量和锁
global event_id_count_lock
event_id_count_lock = threading.Lock()
event_id_count = 0

# 定义新主机事件类
class NewHostEvent(Event):
    def __init__(self, host, cloud=None):
        global event_id_count
        self.host = host
        self.cloud_type = cloud

        with event_id_count_lock:
            self.event_id = event_id_count
            event_id_count += 1

    @property
    def cloud(self):
        if not self.cloud_type:
            self.cloud_type = self.get_cloud()
        return self.cloud_type

    def get_cloud(self):
        try:
            logger.debug("Checking whether the cluster is deployed on azure's cloud")
            # 利用第三方工具 https://github.com/blrchen/AzureSpeed 进行 Azure 云 IP 检测
            result = requests.get(
                f"https://api.azurespeed.com/api/region?ipOrUrl={self.host}", timeout=config.network_timeout,
            ).json()
            return result["cloud"] or "NoCloud"
        except requests.ConnectionError:
            logger.info(f"Failed to connect cloud type service", exc_info=True)
        except Exception:
            logger.warning(f"Unable to check cloud of {self.host}", exc_info=True)
        return "NoCloud"

    def __str__(self):
        return str(self.host)

    # 事件的逻辑位置，主要用于报告
    def location(self):
        return str(self.host)

# 定义开放端口事件类
class OpenPortEvent(Event):
    def __init__(self, port):
        self.port = port

    def __str__(self):
        return str(self.port)

    # 事件的逻辑位置，主要用于报告
    def location(self):
        if self.host:
            location = str(self.host) + ":" + str(self.port)
        else:
            location = str(self.port)
        return location

# 定义猎人完成事件类
class HuntFinished(Event):
    pass

# 定义猎人开始事件类
class HuntStarted(Event):
    pass

# 定义报告分发事件类
class ReportDispatched(Event):
    pass

# 定义 Kubernetes 版本泄露漏洞类
class K8sVersionDisclosure(Vulnerability, Event):
    """The kubernetes version could be obtained from the {} endpoint """

    def __init__(self, version, from_endpoint, extra_info=""):
        Vulnerability.__init__(
            self, KubernetesCluster, "K8s Version Disclosure", category=InformationDisclosure, vid="KHV002",
        )
        self.version = version
        self.from_endpoint = from_endpoint
        self.extra_info = extra_info
        self.evidence = version

    def explain(self):
        return self.__doc__.format(self.from_endpoint) + self.extra_info

```