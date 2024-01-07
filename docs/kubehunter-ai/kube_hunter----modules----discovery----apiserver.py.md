# `.\kubehunter\kube_hunter\modules\discovery\apiserver.py`

```

# 导入requests和logging模块
import requests
import logging

# 导入kube_hunter中的Discovery、handler、OpenPortEvent、Service、Event、EventFilterBase等类和config配置
from kube_hunter.core.types import Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import OpenPortEvent, Service, Event, EventFilterBase
from kube_hunter.conf import config

# 已知的Kubernetes API端口
KNOWN_API_PORTS = [443, 6443, 8080]

# 获取logger对象
logger = logging.getLogger(__name__)

# 定义K8sApiService类，继承自Service和Event类
class K8sApiService(Service, Event):
    """A Kubernetes API service"""

    def __init__(self, protocol="https"):
        Service.__init__(self, name="Unrecognized K8s API")
        self.protocol = protocol

# 定义ApiServer类，继承自Service和Event类
class ApiServer(Service, Event):
    """The API server is in charge of all operations on the cluster."""

    def __init__(self):
        Service.__init__(self, name="API Server")
        self.protocol = "https"

# 定义MetricsServer类，继承自Service和Event类
class MetricsServer(Service, Event):
    """The Metrics server is in charge of providing resource usage metrics for pods and nodes to the API server"""

    def __init__(self):
        Service.__init__(self, name="Metrics Server")
        self.protocol = "https"

# 订阅OpenPortEvent事件，用于发现Kubernetes API服务
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port in KNOWN_API_PORTS)
class ApiServiceDiscovery(Discovery):
    """API Service Discovery
    Checks for the existence of K8s API Services
    """

    def __init__(self, event):
        self.event = event
        self.session = requests.Session()
        self.session.verify = False

    def execute(self):
        logger.debug(f"Attempting to discover an API service on {self.event.host}:{self.event.port}")
        protocols = ["http", "https"]
        for protocol in protocols:
            if self.has_api_behaviour(protocol):
                self.publish_event(K8sApiService(protocol))

    def has_api_behaviour(self, protocol):
        try:
            r = self.session.get(f"{protocol}://{self.event.host}:{self.event.port}", timeout=config.network_timeout)
            if ("k8s" in r.text) or ('"code"' in r.text and r.status_code != 200):
                return True
        except requests.exceptions.SSLError:
            logger.debug(f"{[protocol]} protocol not accepted on {self.event.host}:{self.event.port}")
        except Exception:
            logger.debug(f"Failed probing {self.event.host}:{self.event.port}", exc_info=True)

# 订阅K8sApiService事件，用于对API服务进行分类
@handler.subscribe(K8sApiService)
class ApiServiceClassify(EventFilterBase):
    """API Service Classifier
    Classifies an API service
    """

    def __init__(self, event):
        self.event = event
        self.classified = False
        self.session = requests.Session()
        self.session.verify = False
        # 如果存在认证令牌，则使用该令牌进行认证
        if self.event.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.event.auth_token}"})

    def classify_using_version_endpoint(self):
        """Tries to classify by accessing /version. if could not access succeded, returns"""
        try:
            endpoint = f"{self.event.protocol}://{self.event.host}:{self.event.port}/version"
            versions = self.session.get(endpoint, timeout=config.network_timeout).json()
            if "major" in versions:
                if versions.get("major") == "":
                    self.event = MetricsServer()
                else:
                    self.event = ApiServer()
        except Exception:
            logging.warning("Could not access /version on API service", exc_info=True)

    def execute(self):
        discovered_protocol = self.event.protocol
        # 如果作为pod运行
        if self.event.kubeservicehost:
            # 如果主机是API服务器的IP，则知道它是Api Server
            if self.event.kubeservicehost == str(self.event.host):
                self.event = ApiServer()
            else:
                self.event = MetricsServer()
        # 如果不作为pod运行
        else:
            self.classify_using_version_endpoint()

        # 确保链接到先前发现的协议
        self.event.protocol = discovered_protocol
        # 如果某个检查分类了服务，则事件将被替换
        return self.event

```