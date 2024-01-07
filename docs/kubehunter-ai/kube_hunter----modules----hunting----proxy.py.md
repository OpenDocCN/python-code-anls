# `.\kubehunter\kube_hunter\modules\hunting\proxy.py`

```

# 导入 logging 和 requests 模块
import logging
import requests

# 导入 Enum 枚举类型
from enum import Enum

# 从 kube_hunter.conf 模块中导入 config 配置
from kube_hunter.conf import config

# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler

# 从 kube_hunter.core.events.types 模块中导入 Event, Vulnerability, K8sVersionDisclosure
from kube_hunter.core.events.types import Event, Vulnerability, K8sVersionDisclosure

# 从 kube_hunter.core.types 模块中导入 ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure
from kube_hunter.core.types import ActiveHunter, Hunter, KubernetesCluster, InformationDisclosure

# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardEvent
from kube_hunter.modules.discovery.dashboard import KubeDashboardEvent

# 从 kube_hunter.modules.discovery.proxy 模块中导入 KubeProxyEvent
from kube_hunter.modules.discovery.proxy import KubeProxyEvent

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义一个 Vulnerability 类，继承自 Event 类
class KubeProxyExposed(Vulnerability, Event):
    """All operations on the cluster are exposed"""

    # 初始化方法
    def __init__(self):
        # 调用 Vulnerability 类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Proxy Exposed", category=InformationDisclosure, vid="KHV049",
        )

# 定义一个枚举类型 Service
class Service(Enum):
    DASHBOARD = "kubernetes-dashboard"

# 订阅 KubeProxyEvent 事件
@handler.subscribe(KubeProxyEvent)
class KubeProxy(Hunter):
    """Proxy Hunting
    Hunts for a dashboard behind the proxy
    """

    # 初始化方法
    def __init__(self, event):
        self.event = event
        self.api_url = f"http://{self.event.host}:{self.event.port}/api/v1"

    # 执行方法
    def execute(self):
        # 发布 KubeProxyExposed 事件
        self.publish_event(KubeProxyExposed())
        # 遍历服务的命名空间和服务名
        for namespace, services in self.services.items():
            for service in services:
                if service == Service.DASHBOARD.value:
                    logger.debug(f"Found a dashboard service '{service}'")
                    # TODO: check if /proxy is a convention on other services
                    curr_path = f"api/v1/namespaces/{namespace}/services/{service}/proxy"
                    self.publish_event(KubeDashboardEvent(path=curr_path, secure=False))

    # 命名空间属性
    @property
    def namespaces(self):
        resource_json = requests.get(f"{self.api_url}/namespaces", timeout=config.network_timeout).json()
        return self.extract_names(resource_json)

    # 服务属性
    @property
    def services(self):
        # 命名空间和服务名的映射关系
        services = dict()
        for namespace in self.namespaces:
            resource_path = f"{self.api_url}/namespaces/{namespace}/services"
            resource_json = requests.get(resource_path, timeout=config.network_timeout).json()
            services[namespace] = self.extract_names(resource_json)
        logger.debug(f"Enumerated services [{' '.join(services)}]")
        return services

    # 提取名称的静态方法
    @staticmethod
    def extract_names(resource_json):
        names = list()
        for item in resource_json["items"]:
            names.append(item["metadata"]["name"])
        return names

# 订阅 KubeProxyExposed 事件
@handler.subscribe(KubeProxyExposed)
class ProveProxyExposed(ActiveHunter):
    """Build Date Hunter
    Hunts when proxy is exposed, extracts the build date of kubernetes
    """

    # 初始化方法
    def __init__(self, event):
        self.event = event

    # 执行方法
    def execute(self):
        version_metadata = requests.get(
            f"http://{self.event.host}:{self.event.port}/version", verify=False, timeout=config.network_timeout,
        ).json()
        if "buildDate" in version_metadata:
            self.event.evidence = "build date: {}".format(version_metadata["buildDate"])

# 订阅 KubeProxyExposed 事件
@handler.subscribe(KubeProxyExposed)
class K8sVersionDisclosureProve(ActiveHunter):
    """K8s Version Hunter
    Hunts Proxy when exposed, extracts the version
    """

    # 初始化方法
    def __init__(self, event):
        self.event = event

    # 执行方法
    def execute(self):
        version_metadata = requests.get(
            f"http://{self.event.host}:{self.event.port}/version", verify=False, timeout=config.network_timeout,
        ).json()
        if "gitVersion" in version_metadata:
            self.publish_event(
                K8sVersionDisclosure(
                    version=version_metadata["gitVersion"], from_endpoint="/version", extra_info="on kube-proxy",
                )
            )

```