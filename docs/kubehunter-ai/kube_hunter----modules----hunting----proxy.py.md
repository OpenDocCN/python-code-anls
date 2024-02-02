# `kubehunter\kube_hunter\modules\hunting\proxy.py`

```py
# 导入 logging 模块
import logging
# 导入 requests 模块
import requests

# 导入 Enum 类
from enum import Enum

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event, Vulnerability, K8sVersionDisclosure 类
from kube_hunter.core.events.types import Event, Vulnerability, K8sVersionDisclosure
# 从 kube_hunter.core.types 模块中导入 ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure 类
from kube_hunter.core.types import ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure
# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardEvent 类
from kube_hunter.modules.discovery.dashboard import KubeDashboardEvent
# 从 kube_hunter.modules.discovery.proxy 模块中导入 KubeProxyEvent 类
from kube_hunter.modules.discovery.proxy import KubeProxyEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 KubeProxyExposed 类，继承自 Vulnerability 和 Event 类
class KubeProxyExposed(Vulnerability, Event):
    """All operations on the cluster are exposed"""

    # 初始化方法
    def __init__(self):
        # 调用 Vulnerability 类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Proxy Exposed", category=InformationDisclosure, vid="KHV049",
        )

# 定义 Service 枚举类
class Service(Enum):
    DASHBOARD = "kubernetes-dashboard"

# 订阅 KubeProxyEvent 事件
@handler.subscribe(KubeProxyEvent)
# 定义 KubeProxy 类，继承自 Hunter 类
class KubeProxy(Hunter):
    """Proxy Hunting
    Hunts for a dashboard behind the proxy
    """

    # 初始化方法
    def __init__(self, event):
        # 保存事件对象
        self.event = event
        # 构建 API 地址
        self.api_url = f"http://{self.event.host}:{self.event.port}/api/v1"

    # 执行方法
    def execute(self):
        # 发布 KubeProxyExposed 事件
        self.publish_event(KubeProxyExposed())
        # 遍历服务列表
        for namespace, services in self.services.items():
            for service in services:
                if service == Service.DASHBOARD.value:
                    # 记录日志
                    logger.debug(f"Found a dashboard service '{service}'")
                    # TODO: check if /proxy is a convention on other services
                    # 构建当前路径
                    curr_path = f"api/v1/namespaces/{namespace}/services/{service}/proxy"
                    # 发布 KubeDashboardEvent 事件
                    self.publish_event(KubeDashboardEvent(path=curr_path, secure=False))

    # 获取命名空间属性
    @property
    def namespaces(self):
        # 发送请求获取资源信息
        resource_json = requests.get(f"{self.api_url}/namespaces", timeout=config.network_timeout).json()
        # 提取命名空间名称
        return self.extract_names(resource_json)

    # 定义另一个属性
    @property
    # 定义一个方法用于获取服务信息
    def services(self):
        # 创建一个空的字典，用于存储命名空间和服务名称的映射关系
        services = dict()
        # 遍历命名空间列表
        for namespace in self.namespaces:
            # 构建资源路径
            resource_path = f"{self.api_url}/namespaces/{namespace}/services"
            # 发起 GET 请求获取资源的 JSON 数据
            resource_json = requests.get(resource_path, timeout=config.network_timeout).json()
            # 提取服务名称并存储到映射关系中
            services[namespace] = self.extract_names(resource_json)
        # 记录调试信息，列举已枚举的服务
        logger.debug(f"Enumerated services [{' '.join(services)}]")
        # 返回存储了命名空间和服务名称映射关系的字典
        return services

    # 定义一个静态方法用于提取服务名称
    @staticmethod
    def extract_names(resource_json):
        # 创建一个空列表，用于存储服务名称
        names = list()
        # 遍历资源 JSON 数据中的每个项目
        for item in resource_json["items"]:
            # 提取每个项目的元数据中的名称，并存储到列表中
            names.append(item["metadata"]["name"])
        # 返回存储了服务名称的列表
        return names
# 当 KubeProxyExposed 事件发生时，执行 ProveProxyExposed 类
@handler.subscribe(KubeProxyExposed)
class ProveProxyExposed(ActiveHunter):
    """Build Date Hunter
    Hunts when proxy is exposed, extracts the build date of kubernetes
    """

    # 初始化函数，接收事件参数
    def __init__(self, event):
        self.event = event

    # 执行函数
    def execute(self):
        # 发送请求获取版本元数据
        version_metadata = requests.get(
            f"http://{self.event.host}:{self.event.port}/version", verify=False, timeout=config.network_timeout,
        ).json()
        # 如果版本元数据中包含 buildDate，则将其作为证据保存在事件中
        if "buildDate" in version_metadata:
            self.event.evidence = "build date: {}".format(version_metadata["buildDate"])


# 当 KubeProxyExposed 事件发生时，执行 K8sVersionDisclosureProve 类
@handler.subscribe(KubeProxyExposed)
class K8sVersionDisclosureProve(ActiveHunter):
    """K8s Version Hunter
    Hunts Proxy when exposed, extracts the version
    """

    # 初始化函数，接收事件参数
    def __init__(self, event):
        self.event = event

    # 执行函数
    def execute(self):
        # 发送请求获取版本元数据
        version_metadata = requests.get(
            f"http://{self.event.host}:{self.event.port}/version", verify=False, timeout=config.network_timeout,
        ).json()
        # 如果版本元数据中包含 gitVersion，则发布 K8sVersionDisclosure 事件
        if "gitVersion" in version_metadata:
            self.publish_event(
                K8sVersionDisclosure(
                    version=version_metadata["gitVersion"], from_endpoint="/version", extra_info="on kube-proxy",
                )
            )
```