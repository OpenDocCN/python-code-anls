# `kubehunter\kube_hunter\modules\discovery\apiserver.py`

```
# 导入requests和logging模块
import requests
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

    # 初始化方法，设置协议默认值为https
    def __init__(self, protocol="https"):
        # 调用父类Service的初始化方法，设置服务名称为"Unrecognized K8s API"
        Service.__init__(self, name="Unrecognized K8s API")
        # 设置协议属性
        self.protocol = protocol
# 创建一个名为 ApiServer 的类，继承自 Service 和 Event 类
# 该类负责集群上的所有操作
class ApiServer(Service, Event):
    """The API server is in charge of all operations on the cluster."""

    # 初始化方法，设置服务名称为 "API Server"，协议为 https
    def __init__(self):
        Service.__init__(self, name="API Server")
        self.protocol = "https"


# 创建一个名为 MetricsServer 的类，继承自 Service 和 Event 类
# 该类负责为 API 服务器提供有关 pod 和节点资源使用情况的指标
class MetricsServer(Service, Event):
    """The Metrics server is in charge of providing resource usage metrics for pods and nodes to the API server"""

    # 初始化方法，设置服务名称为 "Metrics Server"，协议为 https
    def __init__(self):
        Service.__init__(self, name="Metrics Server")
        self.protocol = "https"


# 其他设备可能会打开这个端口，但我们可以检查它是否看起来像一个 Kubernetes API
# 一个 Kubernetes API 服务将以包含 HTTP 状态码的 "code" 字段的 JSON 消息作出响应
# 使用 handler.subscribe 装饰器订阅 OpenPortEvent 事件，并使用 lambda 表达式作为筛选条件，判断端口是否在已知的 API 端口列表中
# ApiServiceDiscovery 类用于 API 服务的发现，检查 K8s API 服务的存在
class ApiServiceDiscovery(Discovery):
    """API Service Discovery
    Checks for the existence of K8s API Services
    """

    # 初始化方法，接收事件参数，并初始化会话和会话验证
    def __init__(self, event):
        self.event = event
        self.session = requests.Session()
        self.session.verify = False

    # 执行方法，尝试在指定主机和端口上发现 API 服务
    def execute(self):
        logger.debug(f"Attempting to discover an API service on {self.event.host}:{self.event.port}")
        protocols = ["http", "https"]
        # 遍历协议列表
        for protocol in protocols:
            # 如果具有指定协议的 API 行为，则发布 K8sApiService 事件
            if self.has_api_behaviour(protocol):
                self.publish_event(K8sApiService(protocol))

    # 判断是否具有指定协议的 API 行为
    def has_api_behaviour(self, protocol):
        try:
# 发起一个 GET 请求，获取指定主机和端口的内容，设置超时时间为配置文件中的网络超时时间
r = self.session.get(f"{protocol}://{self.event.host}:{self.event.port}", timeout=config.network_timeout)
# 如果返回的内容包含"k8s"，或者包含'"code"'并且状态码不是200，则返回True
if ("k8s" in r.text) or ('"code"' in r.text and r.status_code != 200):
    return True
# 如果发生 SSL 错误，记录日志
except requests.exceptions.SSLError:
    logger.debug(f"{[protocol]} protocol not accepted on {self.event.host}:{self.event.port}")
# 如果发生其他异常，记录日志
except Exception:
    logger.debug(f"Failed probing {self.event.host}:{self.event.port}", exc_info=True)

# 作为服务的过滤器，如果可以对 API 进行分类，则将过滤后的事件替换为新的对应服务以便下一步发布
# 分类可以根据执行上下文进行，目前我们分类：Metrics Server 和 Api Server
# 如果作为一个 Pod 运行：
# 我们知道 Api 服务器的 IP，所以可以很容易地进行分类
# 如果不是：
# 我们通过访问服务上的 /version 来确定
# Api 服务器将包含一个主要版本字段，而 Metrics 不会
@handler.subscribe(K8sApiService)
class ApiServiceClassify(EventFilterBase):
    """API Service Classifier
    Classifies an API service
    """

    # 初始化方法，接受一个事件对象作为参数
    def __init__(self, event):
        # 将事件对象保存到实例变量中
        self.event = event
        # 初始化分类标志为 False
        self.classified = False
        # 创建一个会话对象
        self.session = requests.Session()
        # 禁用 SSL 证书验证
        self.session.verify = False
        # 如果存在认证令牌，将其添加到请求头中
        if self.event.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.event.auth_token}"})

    # 使用版本端点进行分类
    def classify_using_version_endpoint(self):
        """Tries to classify by accessing /version. if could not access succeded, returns"""
        # 尝试访问 /version 端点进行分类
        try:
            # 构建版本端点的 URL
            endpoint = f"{self.event.protocol}://{self.event.host}:{self.event.port}/version"
            # 发起 GET 请求获取版本信息，设置超时时间为配置文件中的网络超时时间
            versions = self.session.get(endpoint, timeout=config.network_timeout).json()
            # 如果返回的版本信息中包含 "major" 字段
            if "major" in versions:
                # 如果 "major" 字段的值为空
                if versions.get("major") == "":
# 如果条件成立，创建一个MetricsServer对象并赋值给self.event；否则创建一个ApiServer对象并赋值给self.event
if running as pod
    # 如果主机是API服务器的IP，则将self.event设置为ApiServer对象
    if self.event.kubeservicehost == str(self.event.host):
        self.event = ApiServer()
    # 否则将self.event设置为MetricsServer对象
    else:
        self.event = MetricsServer()
# 如果不是作为pod运行
else:
    # 调用classify_using_version_endpoint方法
    self.classify_using_version_endpoint()

# 确保链接到先前发现的协议
        # 将发现的协议赋值给事件的协议属性
        self.event.protocol = discovered_protocol
        # 如果某个检查分类了该服务，
        # 则事件将被替换。
        # 返回事件对象
        return self.event
```