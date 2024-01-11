# `kubehunter\kube_hunter\core\events\types.py`

```
# 导入 threading 模块，用于支持多线程编程
import threading
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 logging 模块，用于记录日志
import logging

# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入各种事件类型
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

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)


# 定义 EventFilterBase 类
class EventFilterBase(object):
    # 初始化方法，接受一个 event 参数
    def __init__(self, event):
        self.event = event

    # 执行方法，默认返回 self.event
    # 如果有修改，应返回已更改的事件
    # 返回 None 表示应丢弃事件
    def execute(self):
        return self.event


# 定义 Event 类
class Event(object):
    # 初始化方法
    def __init__(self):
        self.previous = None
        self.hunter = None

    # 获取属性的方法
    # 如果属性名为 "previous"，返回 None
    # 否则遍历历史事件，返回匹配属性名的属性值
    def __getattr__(self, name):
        if name == "previous":
            return None
        for event in self.history:
            if name in event.__dict__:
                return event.__dict__[name]

    # 事件的逻辑位置，主要用于报告
    # 如果事件没有实现该方法，则检查前一个事件
    # 这是因为事件是组合的（前一个 -> 前一个 ...），而不是继承的
    def location(self):
        location = None
        if self.previous:
            location = self.previous.location()

        return location

    # 返回按时间顺序从新到旧排序的事件历史
    @property
    def history(self):
        previous, history = self.previous, list()
        while previous:
            history.append(previous)
            previous = previous.previous
        return history


# 定义 Service 类
class Service(object):
    # 初始化方法，接受 name、path 和 secure 参数
    def __init__(self, name, path="", secure=True):
        self.name = name
        self.secure = secure
        self.path = path
        self.role = "Node"

    # 获取服务名称的方法
    def get_name(self):
        return self.name

    # 获取服务路径的方法
    def get_path(self):
        return "/" + self.path if self.path else ""
    # 定义一个方法，用于返回对象的文档字符串
    def explain(self):
        # 返回对象的文档字符串
        return self.__doc__
# 定义漏洞类
class Vulnerability(object):
    # 定义漏洞严重程度字典
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
    # TODO: 迁移完成后使 vid 成为必填项
    def __init__(self, component, name, category=None, vid="None"):
        self.vid = vid
        self.component = component
        self.category = category
        self.name = name
        self.evidence = ""
        self.role = "Node"

    # 获取漏洞的 vid
    def get_vid(self):
        return self.vid

    # 获取漏洞的类别
    def get_category(self):
        if self.category:
            return self.category.name

    # 获取漏洞的名称
    def get_name(self):
        return self.name

    # 解释漏洞
    def explain(self):
        return self.__doc__

    # 获取漏洞的严重程度
    def get_severity(self):
        return self.severity.get(self.category, "low")


# 定义全局事件 ID 计数锁
global event_id_count_lock
event_id_count_lock = threading.Lock()
event_id_count = 0


# 定义新主机事件类
class NewHostEvent(Event):
    def __init__(self, host, cloud=None):
        global event_id_count
        self.host = host
        self.cloud_type = cloud

        # 使用事件 ID 计数锁来保证事件 ID 的唯一性
        with event_id_count_lock:
            self.event_id = event_id_count
            event_id_count += 1

    # 获取云类型
    @property
    def cloud(self):
        if not self.cloud_type:
            self.cloud_type = self.get_cloud()
        return self.cloud_type
    # 获取云平台信息的方法
    def get_cloud(self):
        try:
            # 记录调试信息，检查集群是否部署在 Azure 云上
            logger.debug("Checking whether the cluster is deployed on azure's cloud")
            # 利用第三方工具 https://github.com/blrchen/AzureSpeed 进行 Azure 云 IP 检测
            result = requests.get(
                f"https://api.azurespeed.com/api/region?ipOrUrl={self.host}", timeout=config.network_timeout,
            ).json()
            # 返回云平台信息，如果没有则返回 "NoCloud"
            return result["cloud"] or "NoCloud"
        except requests.ConnectionError:
            # 记录连接错误信息
            logger.info(f"Failed to connect cloud type service", exc_info=True)
        except Exception:
            # 记录异常信息
            logger.warning(f"Unable to check cloud of {self.host}", exc_info=True)
        # 如果出现异常或连接错误，返回 "NoCloud"
        return "NoCloud"
    
    # 返回主机的字符串表示形式
    def __str__(self):
        return str(self.host)
    
    # 返回事件的逻辑位置，主要用于报告
    def location(self):
        return str(self.host)
# 定义一个 OpenPortEvent 类，继承自 Event 类
class OpenPortEvent(Event):
    # 初始化方法，接收一个端口号参数
    def __init__(self, port):
        self.port = port

    # 返回端口号的字符串表示
    def __str__(self):
        return str(self.port)

    # 返回事件的逻辑位置，主要用于报告
    def location(self):
        # 如果存在主机信息，则位置为主机名:端口号，否则位置为端口号
        if self.host:
            location = str(self.host) + ":" + str(self.port)
        else:
            location = str(self.port)
        return location


# 定义一个 HuntFinished 类，继承自 Event 类
class HuntFinished(Event):
    pass


# 定义一个 HuntStarted 类，继承自 Event 类
class HuntStarted(Event):
    pass


# 定义一个 ReportDispatched 类，继承自 Event 类
class ReportDispatched(Event):
    pass


# 定义一个 K8sVersionDisclosure 类，继承自 Vulnerability 和 Event 类
class K8sVersionDisclosure(Vulnerability, Event):
    """The kubernetes version could be obtained from the {} endpoint """

    # 初始化方法，接收版本号、来源端点和额外信息参数
    def __init__(self, version, from_endpoint, extra_info=""):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "K8s Version Disclosure", category=InformationDisclosure, vid="KHV002",
        )
        self.version = version
        self.from_endpoint = from_endpoint
        self.extra_info = extra_info
        self.evidence = version

    # 返回漏洞的解释说明
    def explain(self):
        return self.__doc__.format(self.from_endpoint) + self.extra_info
```