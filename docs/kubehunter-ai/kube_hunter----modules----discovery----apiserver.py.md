# `kubehunter\kube_hunter\modules\discovery\apiserver.py`

```py
# 导入requests模块，用于发送HTTP请求
import requests
# 导入logging模块，用于记录日志
import logging

# 导入kube_hunter.core.types模块中的Discovery类
from kube_hunter.core.types import Discovery
# 导入kube_hunter.core.events模块中的handler函数
from kube_hunter.core.events import handler
# 导入kube_hunter.core.events.types模块中的OpenPortEvent、Service、Event、EventFilterBase类
from kube_hunter.core.events.types import OpenPortEvent, Service, Event, EventFilterBase
# 导入kube_hunter.conf模块中的config对象
from kube_hunter.conf import config

# 已知的Kubernetes API端口列表
KNOWN_API_PORTS = [443, 6443, 8080]

# 获取logger对象
logger = logging.getLogger(__name__)


# 定义K8sApiService类，继承自Service和Event类
class K8sApiService(Service, Event):
    """A Kubernetes API service"""

    def __init__(self, protocol="https"):
        # 调用父类Service的初始化方法，设置服务名称
        Service.__init__(self, name="Unrecognized K8s API")
        # 设置协议
        self.protocol = protocol


# 定义ApiServer类，继承自Service和Event类
class ApiServer(Service, Event):
    """The API server is in charge of all operations on the cluster."""

    def __init__(self):
        # 调用父类Service的初始化方法，设置服务名称
        Service.__init__(self, name="API Server")
        # 设置协议
        self.protocol = "https"


# 定义MetricsServer类，继承自Service和Event类
class MetricsServer(Service, Event):
    """The Metrics server is in charge of providing resource usage metrics for pods and nodes to the API server"""

    def __init__(self):
        # 调用父类Service的初始化方法，设置服务名称
        Service.__init__(self, name="Metrics Server")
        # 设置协议
        self.protocol = "https"


# 使用handler.subscribe装饰器，订阅OpenPortEvent事件，通过predicate参数过滤端口
@handler.subscribe(OpenPortEvent, predicate=lambda x: x.port in KNOWN_API_PORTS)
class ApiServiceDiscovery(Discovery):
    """API Service Discovery
    Checks for the existence of K8s API Services
    """

    def __init__(self, event):
        # 初始化方法，接收事件对象
        self.event = event
        # 创建会话对象
        self.session = requests.Session()
        # 禁用SSL证书验证
        self.session.verify = False

    def execute(self):
        # 记录调试日志，尝试在指定主机和端口上发现API服务
        logger.debug(f"Attempting to discover an API service on {self.event.host}:{self.event.port}")
        # 定义协议列表
        protocols = ["http", "https"]
        # 遍历协议列表
        for protocol in protocols:
            # 如果具有API行为
            if self.has_api_behaviour(protocol):
                # 发布K8sApiService事件
                self.publish_event(K8sApiService(protocol))
    # 检查对象是否具有 API 行为
    def has_api_behaviour(self, protocol):
        # 尝试发送 GET 请求到指定的协议、主机和端口，设置超时时间为配置文件中的网络超时时间
        try:
            r = self.session.get(f"{protocol}://{self.event.host}:{self.event.port}", timeout=config.network_timeout)
            # 如果响应文本中包含"k8s"，或者包含'"code"'并且状态码不是200，则返回True
            if ("k8s" in r.text) or ('"code"' in r.text and r.status_code != 200):
                return True
        # 捕获 SSL 错误异常，记录日志并提示该协议不被接受
        except requests.exceptions.SSLError:
            logger.debug(f"{[protocol]} protocol not accepted on {self.event.host}:{self.event.port}")
        # 捕获其他异常，记录日志并提示探测失败
        except Exception:
            logger.debug(f"Failed probing {self.event.host}:{self.event.port}", exc_info=True)
# 作为服务的过滤器，如果我们可以对 API 进行分类，我们就会将过滤后的事件与新的对应服务进行交换，以便下一步发布
# 分类可以根据执行的上下文进行，目前我们分类：Metrics 服务器和 Api 服务器
# 如果作为一个 pod 运行：
# 我们知道 Api 服务器的 IP，所以可以很容易地进行分类
# 如果不是：
# 我们通过访问服务上的 /version 来确定
# Api 服务器将包含一个主要版本字段，而 Metrics 不会
@handler.subscribe(K8sApiService)
class ApiServiceClassify(EventFilterBase):
    """API 服务分类器
    对 API 服务进行分类
    """

    def __init__(self, event):
        self.event = event
        self.classified = False
        self.session = requests.Session()
        self.session.verify = False
        # 如果可以，使用认证令牌，以防我们的检查需要进行认证
        if self.event.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.event.auth_token}"})

    def classify_using_version_endpoint(self):
        """尝试通过访问 /version 进行分类，如果无法成功访问，则返回"""
        try:
            endpoint = f"{self.event.protocol}://{self.event.host}:{self.event.port}/version"
            versions = self.session.get(endpoint, timeout=config.network_timeout).json()
            if "major" in versions:
                if versions.get("major") == "":
                    self.event = MetricsServer()
                else:
                    self.event = ApiServer()
        except Exception:
            logging.warning("无法访问 API 服务上的 /version", exc_info=True)
    # 执行函数，处理事件
    def execute(self):
        # 获取事件的协议
        discovered_protocol = self.event.protocol
        # 如果作为 pod 运行
        if self.event.kubeservicehost:
            # 如果主机是 API 服务器的 IP，我们知道它是 API 服务器
            if self.event.kubeservicehost == str(self.event.host):
                # 将事件替换为 ApiServer 对象
                self.event = ApiServer()
            else:
                # 将事件替换为 MetricsServer 对象
                self.event = MetricsServer()
        # 如果不作为 pod 运行
        else:
            # 使用版本端点进行分类
            self.classify_using_version_endpoint()

        # 无论如何，确保链接到先前发现的协议
        self.event.protocol = discovered_protocol
        # 如果某个检查分类了服务，事件将被替换
        return self.event
```