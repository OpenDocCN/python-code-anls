# `kubehunter\kube_hunter\core\events\types.py`

```
# 导入所需的模块
import threading  # 导入线程模块
import requests  # 导入请求模块
import logging  # 导入日志模块

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

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义一个事件过滤器基类
class EventFilterBase(object):
# 初始化方法，将传入的事件对象赋值给实例变量
def __init__(self, event):
    self.event = event

# 执行方法，返回默认的事件对象
# 如果有修改，应返回已更改的新事件
# 返回None表示应丢弃事件
def execute(self):
    return self.event

# 事件类，初始化方法设置previous和hunter为None
class Event(object):
    def __init__(self):
        self.previous = None
        self.hunter = None

# 当获取属性时，如果是"previous"，返回None
# 否则遍历历史事件
def __getattr__(self, name):
    if name == "previous":
        return None
    for event in self.history:
# 如果事件对象的属性字典中包含指定的属性名，则返回该属性的值
if name in event.__dict__:
    return event.__dict__[name]

# 事件的逻辑位置，主要用于报告
# 如果事件没有实现位置属性，则检查前一个事件
# 这是因为事件是由前一个事件组成的（前一个 -> 前一个 ...），而不是继承的
def location(self):
    location = None
    if self.previous:
        location = self.previous.location()
    return location

# 返回按照从新到旧顺序排列的事件历史记录
@property
def history(self):
    previous, history = self.previous, list()
    while previous:
        history.append(previous)
# 定义一个类 Service，表示一个服务
class Service(object):
    # 初始化方法，设置服务的名称、路径和安全性，默认安全
    def __init__(self, name, path="", secure=True):
        self.name = name
        self.secure = secure
        self.path = path
        self.role = "Node"  # 设置默认角色为 "Node"

    # 获取服务的名称
    def get_name(self):
        return self.name

    # 获取服务的路径，如果没有路径则返回根路径
    def get_path(self):
        return "/" + self.path if self.path else ""

    # 返回服务的说明文档
    def explain(self):
        return self.__doc__
# 定义一个 Vulnerability 类
class Vulnerability(object):
    # 定义严重程度的字典，将不同类型的漏洞映射到对应的严重程度
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

    # 初始化方法，初始化漏洞对象的组件、名称、类别和漏洞ID
    # TODO: 迁移完成后，使 vid 成为必填项
    def __init__(self, component, name, category=None, vid="None"):
        self.vid = vid  # 漏洞ID
        self.component = component  # 漏洞所属组件
        self.category = category  # 漏洞类别
        self.name = name  # 漏洞名称
# 初始化 evidence 为空字符串
self.evidence = ""
# 初始化 role 为 "Node"
self.role = "Node"

# 获取 vid 属性的值
def get_vid(self):
    return self.vid

# 获取 category 属性的值，如果存在则返回其名称
def get_category(self):
    if self.category:
        return self.category.name

# 获取 name 属性的值
def get_name(self):
    return self.name

# 返回对象的文档字符串
def explain(self):
    return self.__doc__

# 获取 severity 属性的值，如果存在则返回对应的严重程度，否则返回 "low"
def get_severity(self):
    return self.severity.get(self.category, "low")
# 定义全局变量 event_id_count_lock，用于线程同步
global event_id_count_lock
# 初始化事件计数器锁
event_id_count_lock = threading.Lock()
# 初始化事件计数器
event_id_count = 0

# 定义 NewHostEvent 类，继承自 Event 类
class NewHostEvent(Event):
    # 初始化方法，接收主机和云类型作为参数
    def __init__(self, host, cloud=None):
        # 引用全局变量 event_id_count
        global event_id_count
        # 设置主机属性
        self.host = host
        # 设置云类型属性
        self.cloud_type = cloud

        # 使用事件计数器锁，确保事件 ID 的唯一性
        with event_id_count_lock:
            # 设置事件 ID
            self.event_id = event_id_count
            # 递增事件计数器
            event_id_count += 1

    # 定义 cloud 属性的 getter 方法
    @property
    def cloud(self):
        # 如果云类型为空，则调用 get_cloud 方法获取云类型
        if not self.cloud_type:
            self.cloud_type = self.get_cloud()
        # 返回云类型
        return self.cloud_type
    # 获取云平台信息
    def get_cloud(self):
        try:
            # 打印日志，检查集群是否部署在 Azure 云上
            logger.debug("Checking whether the cluster is deployed on azure's cloud")
            # 使用第三方工具 https://github.com/blrchen/AzureSpeed 进行 Azure 云 IP 检测
            result = requests.get(
                f"https://api.azurespeed.com/api/region?ipOrUrl={self.host}", timeout=config.network_timeout,
            ).json()
            # 返回云平台信息，如果没有则返回 "NoCloud"
            return result["cloud"] or "NoCloud"
        except requests.ConnectionError:
            # 打印日志，连接云类型服务失败
            logger.info(f"Failed to connect cloud type service", exc_info=True)
        except Exception:
            # 打印日志，无法检查主机的云平台信息
            logger.warning(f"Unable to check cloud of {self.host}", exc_info=True)
        # 如果出现异常，返回 "NoCloud"
        return "NoCloud"

    # 返回主机的字符串表示形式
    def __str__(self):
        return str(self.host)

    # 事件的逻辑位置，主要用于报告
    def location(self):
# 返回主机的字符串表示形式
        return str(self.host)


# 表示端口打开事件的类
class OpenPortEvent(Event):
    # 初始化方法，接受端口参数
    def __init__(self, port):
        self.port = port

    # 返回事件的字符串表示形式
    def __str__(self):
        return str(self.port)

    # 用于报告的事件逻辑位置
    def location(self):
        # 如果有主机信息，则返回主机和端口的组合字符串
        if self.host:
            location = str(self.host) + ":" + str(self.port)
        # 否则只返回端口字符串
        else:
            location = str(self.port)
        return location


# 表示搜索结束事件的类
class HuntFinished(Event):
```

# pass 语句用于占位，表示暂时不做任何操作

# 定义 HuntStarted 类，继承自 Event 类
class HuntStarted(Event):
    pass

# 定义 ReportDispatched 类，继承自 Event 类
class ReportDispatched(Event):
    pass

# 定义 K8sVersionDisclosure 类，继承自 Vulnerability 和 Event 类
# 该类表示 Kubernetes 版本可以从特定的端点获取
class K8sVersionDisclosure(Vulnerability, Event):
    """The kubernetes version could be obtained from the {} endpoint """

    # 初始化方法，接受版本号、端点和额外信息作为参数
    def __init__(self, version, from_endpoint, extra_info=""):
        # 调用父类的初始化方法，设置漏洞类型、名称、类别和 ID
        Vulnerability.__init__(
            self, KubernetesCluster, "K8s Version Disclosure", category=InformationDisclosure, vid="KHV002",
        )
        # 设置版本号和端点
        self.version = version
        self.from_endpoint = from_endpoint
# 设置对象的额外信息为传入的额外信息
self.extra_info = extra_info
# 设置对象的证据为传入的版本信息
self.evidence = version

# 解释对象的功能，使用对象的文档字符串格式化传入的端点信息，并加上额外信息
def explain(self):
    return self.__doc__.format(self.from_endpoint) + self.extra_info
```