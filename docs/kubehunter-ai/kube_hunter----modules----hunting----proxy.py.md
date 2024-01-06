# `kubehunter\kube_hunter\modules\hunting\proxy.py`

```
# 导入 logging 模块，用于记录日志
import logging
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 Enum 枚举类型，用于定义枚举类型
from enum import Enum
# 从 kube_hunter.conf 模块中导入 config 配置
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 事件处理器
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event、Vulnerability、K8sVersionDisclosure 类型
from kube_hunter.core.events.types import Event, Vulnerability, K8sVersionDisclosure
# 从 kube_hunter.core.types 模块中导入 ActiveHunter、Hunter、KubernetesCluster、InformationDisclosure 类型
from kube_hunter.core.types import ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure
# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardEvent 类型
from kube_hunter.modules.discovery.dashboard import KubeDashboardEvent
# 从 kube_hunter.modules.discovery.proxy 模块中导入 KubeProxyEvent 类型
from kube_hunter.modules.discovery.proxy import KubeProxyEvent
# 获取 logger 对象，用于记录日志
logger = logging.getLogger(__name__)
# 定义一个名为KubeProxyExposed的类，该类继承自Vulnerability和Event类
class KubeProxyExposed(Vulnerability, Event):
    """All operations on the cluster are exposed"""
    # 初始化方法，设置漏洞类型为"Proxy Exposed"，类别为InformationDisclosure，漏洞ID为"KHV049"
    def __init__(self):
        Vulnerability.__init__(
            self, KubernetesCluster, "Proxy Exposed", category=InformationDisclosure, vid="KHV049",
        )

# 定义一个枚举类型Service，包含一个名为DASHBOARD的枚举值
class Service(Enum):
    DASHBOARD = "kubernetes-dashboard"

# 订阅KubeProxyEvent事件的处理器
@handler.subscribe(KubeProxyEvent)
# 定义一个名为KubeProxy的类，该类继承自Hunter类
class KubeProxy(Hunter):
    """Proxy Hunting
    Hunts for a dashboard behind the proxy
    """
    # 初始化方法，接收一个事件参数
    def __init__(self, event):
# 将传入的 event 参数赋值给对象的 event 属性
self.event = event
# 根据 event 的 host 和 port 属性拼接出 API 的 URL，并赋值给对象的 api_url 属性
self.api_url = f"http://{self.event.host}:{self.event.port}/api/v1"

# 执行方法
def execute(self):
    # 发布 KubeProxyExposed 事件
    self.publish_event(KubeProxyExposed())
    # 遍历 self.services 中的 namespace 和 services
    for namespace, services in self.services.items():
        # 遍历 services
        for service in services:
            # 如果 service 是 DASHBOARD 服务
            if service == Service.DASHBOARD.value:
                # 记录日志
                logger.debug(f"Found a dashboard service '{service}'")
                # TODO: 检查 /proxy 是否是其他服务的约定
                # 拼接当前路径
                curr_path = f"api/v1/namespaces/{namespace}/services/{service}/proxy"
                # 发布 KubeDashboardEvent 事件
                self.publish_event(KubeDashboardEvent(path=curr_path, secure=False))

# 获取命名空间属性
@property
def namespaces(self):
    # 发送 GET 请求获取命名空间的 JSON 数据
    resource_json = requests.get(f"{self.api_url}/namespaces", timeout=config.network_timeout).json()
    # 提取命名空间名称并返回
    return self.extract_names(resource_json)

# 获取服务属性
@property
def services(self):
# 创建一个空的字典，用于存储命名空间和服务名称的映射关系
services = dict()
# 遍历命名空间列表
for namespace in self.namespaces:
    # 构建资源路径
    resource_path = f"{self.api_url}/namespaces/{namespace}/services"
    # 发起 GET 请求，获取资源的 JSON 数据
    resource_json = requests.get(resource_path, timeout=config.network_timeout).json()
    # 提取资源中的服务名称，并存储到映射关系中
    services[namespace] = self.extract_names(resource_json)
# 记录日志，列举已枚举的服务
logger.debug(f"Enumerated services [{' '.join(services)}]")
# 返回存储了命名空间和服务名称映射关系的字典
return services

# 从资源的 JSON 数据中提取服务名称
@staticmethod
def extract_names(resource_json):
    names = list()
    # 遍历资源中的每个项目，提取其中的服务名称，并存储到列表中
    for item in resource_json["items"]:
        names.append(item["metadata"]["name"])
    # 返回存储了服务名称的列表
    return names

# 订阅 KubeProxyExposed 事件，并定义 ProveProxyExposed 类
@handler.subscribe(KubeProxyExposed)
class ProveProxyExposed(ActiveHunter):
    """Build Date Hunter
# 当代理暴露时，提取kubernetes的构建日期
class K8sVersionDisclosureProve(ActiveHunter):
    # 初始化方法，接收事件参数
    def __init__(self, event):
        self.event = event

    # 执行方法，发送请求获取kubernetes版本元数据，提取构建日期
    def execute(self):
        version_metadata = requests.get(
            f"http://{self.event.host}:{self.event.port}/version", verify=False, timeout=config.network_timeout,
        ).json()
        # 如果版本元数据中包含构建日期，则设置事件的evidence属性为构建日期
        if "buildDate" in version_metadata:
            self.event.evidence = "build date: {}".format(version_metadata["buildDate"])

# 订阅KubeProxyExposed事件的处理器
@handler.subscribe(KubeProxyExposed)
class K8sVersionDisclosureProve(ActiveHunter):
    """K8s Version Hunter
    当代理暴露时，提取kubernetes的版本
    """
# 初始化方法，接受一个事件对象作为参数
def __init__(self, event):
    # 将事件对象保存在实例变量中
    self.event = event

# 执行方法，发送请求获取版本元数据，并根据条件发布事件
def execute(self):
    # 发送 GET 请求获取版本元数据，使用事件对象中的主机和端口信息拼接请求地址
    version_metadata = requests.get(
        f"http://{self.event.host}:{self.event.port}/version", verify=False, timeout=config.network_timeout,
    ).json()
    # 如果版本元数据中包含 gitVersion 字段
    if "gitVersion" in version_metadata:
        # 发布 K8sVersionDisclosure 事件，包含版本信息、请求地址和额外信息
        self.publish_event(
            K8sVersionDisclosure(
                version=version_metadata["gitVersion"], from_endpoint="/version", extra_info="on kube-proxy",
            )
        )
```