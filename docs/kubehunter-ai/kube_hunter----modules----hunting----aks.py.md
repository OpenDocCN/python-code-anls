# `kubehunter\kube_hunter\modules\hunting\aks.py`

```
# 导入所需的模块
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import requests  # 导入发送 HTTP 请求的模块

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.modules.hunting.kubelet 模块中导入 ExposedRunHandler 类
from kube_hunter.modules.hunting.kubelet import ExposedRunHandler
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event 和 Vulnerability 类
from kube_hunter.core.events.types import Event, Vulnerability
# 从 kube_hunter.core.types 模块中导入 Hunter, ActiveHunter, IdentityTheft, Azure 类
from kube_hunter.core.types import Hunter, ActiveHunter, IdentityTheft, Azure

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)

# 定义 AzureSpnExposure 类，继承自 Vulnerability 和 Event 类
class AzureSpnExposure(Vulnerability, Event):
    """The SPN is exposed, potentially allowing an attacker to gain access to the Azure subscription"""

    # 初始化方法，接受 container 参数
    def __init__(self, container):
        # 调用 Vulnerability 类的初始化方法
        Vulnerability.__init__(
            self, Azure, "Azure SPN Exposure", category=IdentityTheft, vid="KHV004",
        )
```

# 设置类的属性为传入的容器对象
        self.container = container

# 订阅ExposedRunHandler事件，当cloud为"Azure"时执行
@handler.subscribe(ExposedRunHandler, predicate=lambda x: x.cloud == "Azure")
class AzureSpnHunter(Hunter):
    """AKS Hunting
    Hunting Azure cluster deployments using specific known configurations
    """

    # 初始化方法，接收event参数，并设置base_url属性为特定格式的URL
    def __init__(self, event):
        self.event = event
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 获取具有访问azure.json文件权限的容器
    def get_key_container(self):
        # 设置endpoint为特定格式的URL
        endpoint = f"{self.base_url}/pods"
        # 记录调试信息
        logger.debug("Trying to find container with access to azure.json file")
        # 尝试发送GET请求获取容器信息
        try:
            r = requests.get(endpoint, verify=False, timeout=config.network_timeout)
        # 处理请求超时异常
        except requests.Timeout:
# 记录调试信息，表示获取 pod 信息失败
logger.debug("failed getting pod info")
# 如果成功获取到 pod 数据
else:
    # 从返回的 JSON 数据中获取 pod 列表
    pods_data = r.json().get("items", [])
    # 遍历每个 pod 数据
    for pod_data in pods_data:
        # 遍历每个 pod 中的容器
        for container in pod_data["spec"]["containers"]:
            # 遍历每个容器中的挂载点
            for mount in container["volumeMounts"]:
                # 获取挂载路径
                path = mount["mountPath"]
                # 如果挂载路径是以 "/etc/kubernetes/azure.json" 开头
                if "/etc/kubernetes/azure.json".startswith(path):
                    # 返回包含容器名称、pod 名称和命名空间的字典
                    return {
                        "name": container["name"],
                        "pod": pod_data["metadata"]["name"],
                        "namespace": pod_data["metadata"]["namespace"],
                    }

# 执行方法
def execute(self):
    # 获取关键容器信息
    container = self.get_key_container()
    # 如果获取到容器信息
    if container:
        # 发布 AzureSpnExposure 事件，包含容器信息
        self.publish_event(AzureSpnExposure(container=container))
# 使用装饰器订阅 AzureSpnExposure 事件
@handler.subscribe(AzureSpnExposure)
class ProveAzureSpnExposure(ActiveHunter):
    """Azure SPN Hunter
    Gets the azure subscription file on the host by executing inside a container
    """

    # 初始化方法，接收事件对象并设置基本 URL
    def __init__(self, event):
        self.event = event
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 执行方法，接收命令和容器参数，发送 POST 请求执行命令
    def run(self, command, container):
        run_url = "/".join(self.base_url, "run", container["namespace"], container["pod"], container["name"])
        return requests.post(run_url, verify=False, params={"cmd": command}, timeout=config.network_timeout)

    # 执行方法，尝试运行命令获取 Azure 订阅信息
    def execute(self):
        try:
            subscription = self.run("cat /etc/kubernetes/azure.json", container=self.event.container).json()
        except requests.Timeout:
            logger.debug("failed to run command in container", exc_info=True)
        except json.decoder.JSONDecodeError:
            # JSON 解析错误处理
# 如果解析 SPN 失败，则记录警告信息
logger.warning("failed to parse SPN")
# 如果解析成功，则将订阅信息中的相关字段赋值给事件对象的对应属性
if "subscriptionId" in subscription:
    self.event.subscriptionId = subscription["subscriptionId"]
    self.event.aadClientId = subscription["aadClientId"]
    self.event.aadClientSecret = subscription["aadClientSecret"]
    self.event.tenantId = subscription["tenantId"]
    # 生成事件的证据信息
    self.event.evidence = f"subscription: {self.event.subscriptionId}"
```