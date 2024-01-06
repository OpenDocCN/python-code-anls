# `kubehunter\kube_hunter\modules\hunting\kubelet.py`

```
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 logging 模块，用于记录日志
import logging
# 导入 Enum 枚举类型，用于定义枚举类型
from enum import Enum

# 导入 re 模块，用于正则表达式操作
import re
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 urllib3 模块，用于处理 HTTP 请求
import urllib3

# 从 kube_hunter.conf 模块中导入 config 配置
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 事件处理器
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability、Event、K8sVersionDisclosure 类型
from kube_hunter.core.events.types import Vulnerability, Event, K8sVersionDisclosure
# 从 kube_hunter.core.types 模块中导入 Hunter、ActiveHunter、KubernetesCluster、Kubelet、InformationDisclosure、RemoteCodeExec、AccessRisk 类型
from kube_hunter.core.types import (
    Hunter,
    ActiveHunter,
    KubernetesCluster,
    Kubelet,
    InformationDisclosure,
    RemoteCodeExec,
    AccessRisk,
)
# 导入需要的模块
from kube_hunter.modules.discovery.kubelet import (
    ReadOnlyKubeletEvent,  # 导入ReadOnlyKubeletEvent模块
    SecureKubeletEvent,   # 导入SecureKubeletEvent模块
)

# 导入日志模块
logger = logging.getLogger(__name__)
# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 定义漏洞类
class ExposedPodsHandler(Vulnerability, Event):
    """An attacker could view sensitive information about pods that are
    bound to a Node using the /pods endpoint"""
    # 初始化方法
    def __init__(self, pods):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Pods", category=InformationDisclosure,
        )
# 将传入的 pods 参数赋值给当前对象的 pods 属性
self.pods = pods
# 根据 pods 的数量生成一条描述信息，并赋值给当前对象的 evidence 属性
self.evidence = f"count: {len(self.pods)}"

# 创建一个名为 AnonymousAuthEnabled 的类，继承自 Vulnerability 和 Event 类
# 该类表示 kubelet 配置错误，可能允许对 kubelet 上的所有请求进行安全访问，无需进行身份验证
def __init__(self):
    # 调用 Vulnerability 类的初始化方法，设置组件、名称、类别和 vid 属性
    Vulnerability.__init__(
        self, component=Kubelet, name="Anonymous Authentication", category=RemoteCodeExec, vid="KHV036",
    )

# 创建一个名为 ExposedContainerLogsHandler 的类，继承自 Vulnerability 和 Event 类
# 该类表示正在使用暴露的 /containerLogs 端点输出运行容器的日志
def __init__(self):
    # 调用 Vulnerability 类的初始化方法，设置组件、名称、类别和 vid 属性
    Vulnerability.__init__(
        self, component=Kubelet, name="Exposed Container Logs", category=InformationDisclosure, vid="KHV037",
# 定义一个名为ExposedRunningPodsHandler的类，该类继承自Vulnerability和Event类
# 该类用于输出当前正在运行的pod的列表以及它们的一些元数据，可能会泄露敏感信息
class ExposedRunningPodsHandler(Vulnerability, Event):

    # 初始化ExposedRunningPodsHandler类
    def __init__(self, count):
        # 调用Vulnerability类的初始化方法
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Running Pods", category=InformationDisclosure, vid="KHV038",
        )
        # 设置当前正在运行的pod的数量
        self.count = count
        # 设置evidence属性，用于记录当前正在运行的pod的数量
        self.evidence = "{} running pods".format(self.count)


# 定义一个名为ExposedExecHandler的类，该类继承自Vulnerability和Event类
# 该类用于描述攻击者可能在容器上运行任意命令的漏洞
class ExposedExecHandler(Vulnerability, Event):

    # 初始化ExposedExecHandler类
    def __init__(self):
        # 调用Vulnerability类的初始化方法
        Vulnerability.__init__(
# 定义一个名为ExposedExecHandler的类，继承自Vulnerability和Event类，表示暴露在容器上的执行操作
class ExposedExecHandler(Vulnerability, Event):
    """An attacker could execute commands inside a container"""

    # 初始化ExposedExecHandler类
    def __init__(self):
        # 调用Vulnerability类的初始化方法，设置组件为Kubelet，名称为Exposed Exec On Container，类别为RemoteCodeExec，漏洞ID为KHV039
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Exec On Container", category=RemoteCodeExec, vid="KHV039",
        )

# 定义一个名为ExposedRunHandler的类，继承自Vulnerability和Event类，表示暴露在容器内部运行命令的操作
class ExposedRunHandler(Vulnerability, Event):
    """An attacker could run an arbitrary command inside a container"""

    # 初始化ExposedRunHandler类
    def __init__(self):
        # 调用Vulnerability类的初始化方法，设置组件为Kubelet，名称为Exposed Run Inside Container，类别为RemoteCodeExec，漏洞ID为KHV040
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Run Inside Container", category=RemoteCodeExec, vid="KHV040",
        )

# 定义一个名为ExposedPortForwardHandler的类，继承自Vulnerability和Event类，表示暴露在容器上设置端口转发规则的操作
class ExposedPortForwardHandler(Vulnerability, Event):
    """An attacker could set port forwarding rule on a pod"""

    # 初始化ExposedPortForwardHandler类
    def __init__(self):
        # 调用Vulnerability类的初始化方法，设置组件为Kubelet，名称为Exposed Port Forward，类别为RemoteCodeExec，漏洞ID为KHV041
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Port Forward", category=RemoteCodeExec, vid="KHV041",
        )
# 定义一个类 ExposedAttachHandler，继承自 Vulnerability 和 Event 类
# 该类打开一个 WebSocket，可能使攻击者能够附加到运行中的容器
class ExposedAttachHandler(Vulnerability, Event):
    """Opens a websocket that could enable an attacker
    to attach to a running container"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self):
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Attaching To Container", category=RemoteCodeExec, vid="KHV042",
        )


# 定义一个类 ExposedHealthzHandler，继承自 Vulnerability 和 Event 类
# 通过访问开放的 /healthz 处理程序，攻击者可以在不进行身份验证的情况下获取集群的健康状态
class ExposedHealthzHandler(Vulnerability, Event):
    """By accessing the open /healthz handler,
    an attacker could get the cluster health state without authenticating"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self, status):
        Vulnerability.__init__(
            self, component=Kubelet, name="Cluster Health Disclosure", category=InformationDisclosure, vid="KHV043",
        )
# 设置对象的状态属性
self.status = status
# 设置对象的证据属性，包含状态信息
self.evidence = f"status: {self.status}"

# 特权容器存在于节点上，可能会暴露节点/集群给不需要的 root 操作
class PrivilegedContainers(Vulnerability, Event):
    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self, containers):
        Vulnerability.__init__(
            self, component=KubernetesCluster, name="Privileged Container", category=AccessRisk, vid="KHV044",
        )
        # 设置容器属性
        self.containers = containers
        # 设置证据属性，包含容器和数量信息
        self.evidence = f"pod: {containers[0][0]}, " f"container: {containers[0][1]}, " f"count: {len(containers)}"

# 系统日志从 kubelet 的 /logs 端点暴露出来
class ExposedSystemLogs(Vulnerability, Event):
    # 初始化方法
    def __init__(self):
# 初始化Vulnerability对象，设置组件、名称、类别和漏洞ID
Vulnerability.__init__(
    self, component=Kubelet, name="Exposed System Logs", category=InformationDisclosure, vid="KHV045",
)

# 定义ExposedKubeletCmdline类，继承自Vulnerability和Event类
class ExposedKubeletCmdline(Vulnerability, Event):
    """Commandline flags that were passed to the kubelet can be obtained from the pprof endpoints"""

    # 初始化ExposedKubeletCmdline对象，设置组件、名称、类别和漏洞ID
    def __init__(self, cmdline):
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Kubelet Cmdline", category=InformationDisclosure, vid="KHV046",
        )
        # 设置cmdline属性
        self.cmdline = cmdline
        # 设置evidence属性，包含cmdline信息
        self.evidence = f"cmdline: {self.cmdline}"

# 定义KubeletHandlers枚举类，包含PODS属性
class KubeletHandlers(Enum):
    # GET
    PODS = "pods"
    # GET
    CONTAINERLOGS = "containerLogs/{pod_namespace}/{pod_id}/{container_name}"
    # 定义一个字符串常量，用于构建容器日志的路径

    RUNNINGPODS = "runningpods"
    # 定义一个字符串常量，用于获取运行中的 pod 列表

    EXEC = "exec/{pod_namespace}/{pod_id}/{container_name}?command={cmd}&input=1&output=1&tty=1"
    # 定义一个字符串常量，用于执行在容器内部执行命令的路径，包括命令、输入输出和终端设置

    RUN = "run/{pod_namespace}/{pod_id}/{container_name}?cmd={cmd}"
    # 定义一个字符串常量，用于在容器内部运行命令的路径，包括命令参数

    PORTFORWARD = "portForward/{pod_namespace}/{pod_id}?port={port}"
    # 定义一个字符串常量，用于将本地端口转发到 pod 内部的指定端口

    ATTACH = "attach/{pod_namespace}/{pod_id}/{container_name}?command={cmd}&input=1&output=1&tty=1"
    # 定义一个字符串常量，用于将当前终端附加到容器内部的路径，包括命令、输入输出和终端设置

    LOGS = "logs/{path}"
    # 定义一个字符串常量，用于获取指定路径的日志

    PPROF_CMDLINE = "debug/pprof/cmdline"
    # 定义一个字符串常量，用于获取 pprof 的命令行信息

@handler.subscribe(ReadOnlyKubeletEvent)
class ReadOnlyKubeletPortHunter(Hunter):
    """Kubelet Readonly Ports Hunter
    # 定义一个类，用于订阅 ReadOnlyKubeletEvent 事件，实现 Kubelet 只读端口的查找
# 在只读的 Kubelet 服务器上寻找特定端口上的特定端点
class PassiveHunter:
    # 初始化方法，接收一个事件对象作为参数
    def __init__(self, event):
        # 将事件对象保存为实例属性
        self.event = event
        # 根据事件的主机和端口构建访问路径
        self.path = f"http://{self.event.host}:{self.event.port}"
        # 初始化 pods_endpoint_data 属性为空字符串
        self.pods_endpoint_data = ""

    # 获取 Kubernetes 版本信息
    def get_k8s_version(self):
        # 记录调试日志
        logger.debug("Passive hunter is attempting to find kubernetes version")
        # 发起请求获取 metrics 数据
        metrics = requests.get(f"{self.path}/metrics", timeout=config.network_timeout).text
        # 遍历 metrics 数据的每一行
        for line in metrics.split("\n"):
            # 如果行以 "kubernetes_build_info" 开头
            if line.startswith("kubernetes_build_info"):
                # 遍历括号内的信息
                for info in line[line.find("{") + 1 : line.find("}")].split(","):
                    # 将键值对分割出来
                    k, v = info.split("=")
                    # 如果键是 "gitVersion"
                    if k == "gitVersion":
                        # 返回版本号
                        return v.strip('"')

    # 返回特权容器及其所在的 Pod 的元组列表
    def find_privileged_containers(self):
# 记录调试信息，尝试查找特权容器及其所属的 pods
logger.debug("Trying to find privileged containers and their pods")
# 初始化特权容器列表
privileged_containers = []
# 如果存在 pods 的端点数据
if self.pods_endpoint_data:
    # 遍历每个 pod
    for pod in self.pods_endpoint_data["items"]:
        # 遍历每个 pod 中的容器
        for container in pod["spec"]["containers"]:
            # 如果容器具有特权权限
            if container.get("securityContext", {}).get("privileged"):
                # 将特权容器的信息加入特权容器列表
                privileged_containers.append((pod["metadata"]["name"], container["name"]))
# 如果特权容器列表不为空，则返回列表，否则返回 None
return privileged_containers if len(privileged_containers) > 0 else None

# 获取 pods 的端点数据
def get_pods_endpoint(self):
    # 记录调试信息，尝试查找 pods 的端点
    logger.debug("Attempting to find pods endpoints")
    # 发送 GET 请求获取 pods 的端点数据
    response = requests.get(f"{self.path}/pods", timeout=config.network_timeout)
    # 如果响应中包含 "items" 字段，则返回 JSON 格式的数据
    if "items" in response.text:
        return response.json()

# 检查健康检查的端点
def check_healthz_endpoint(self):
    # 发送 GET 请求检查健康检查的端点
    r = requests.get(f"{self.path}/healthz", verify=False, timeout=config.network_timeout)
    # 如果响应状态码为 200，则返回响应文本，否则返回 False
    return r.text if r.status_code == 200 else False

# 执行主程序
def execute(self):
# 获取 pods 的端点数据
self.pods_endpoint_data = self.get_pods_endpoint()
# 获取 Kubernetes 版本信息
k8s_version = self.get_k8s_version()
# 查找特权容器
privileged_containers = self.find_privileged_containers()
# 检查健康检查端点
healthz = self.check_healthz_endpoint()
# 如果获取到 Kubernetes 版本信息，则发布 K8sVersionDisclosure 事件
if k8s_version:
    self.publish_event(
        K8sVersionDisclosure(version=k8s_version, from_endpoint="/metrics", extra_info="on Kubelet")
    )
# 如果存在特权容器，则发布 PrivilegedContainers 事件
if privileged_containers:
    self.publish_event(PrivilegedContainers(containers=privileged_containers))
# 如果存在健康检查端点，则发布 ExposedHealthzHandler 事件
if healthz:
    self.publish_event(ExposedHealthzHandler(status=healthz))
# 如果存在 pods 的端点数据，则发布 ExposedPodsHandler 事件
if self.pods_endpoint_data:
    self.publish_event(ExposedPodsHandler(pods=self.pods_endpoint_data["items"]))

# 订阅 SecureKubeletEvent 事件，并定义 SecureKubeletPortHunter 类
@handler.subscribe(SecureKubeletEvent)
class SecureKubeletPortHunter(Hunter):
    """Kubelet Secure Ports Hunter
    Hunts specific endpoints on an open secured Kubelet
    """

    # 定义一个调试处理器类
    class DebugHandlers(object):
        """ 所有方法如果成功将返回处理器名称 """

        # 初始化方法，接受路径、pod 和会话参数
        def __init__(self, path, pod, session=None):
            self.path = path
            self.session = session if session else requests.Session()
            self.pod = pod

        # 输出特定容器的日志
        def test_container_logs(self):
            # 构建获取日志的 URL
            logs_url = self.path + KubeletHandlers.CONTAINERLOGS.value.format(
                pod_namespace=self.pod["namespace"], pod_id=self.pod["name"], container_name=self.pod["container"],
            )
            # 发起 GET 请求，验证状态码是否为 200
            return self.session.get(logs_url, verify=False, timeout=config.network_timeout).status_code == 200

        # 需要进一步研究 WebSockets 协议以进行进一步实现
        def test_exec_container(self):
            # 打开一个流以使用 WebSockets 连接
# 设置请求头，指定使用的流协议版本
headers = {"X-Stream-Protocol-Version": "v2.channel.k8s.io"}

# 构建执行命令的 URL，包括命名空间、Pod ID、容器名称和命令
exec_url = self.path + KubeletHandlers.EXEC.value.format(
    pod_namespace=self.pod["namespace"],
    pod_id=self.pod["name"],
    container_name=self.pod["container"],
    cmd="",
)

# 发起 GET 请求，检查是否支持在指定 URL 上进行远程执行命令
return (
    "/cri/exec/"
    in self.session.get(
        exec_url, headers=headers, allow_redirects=False, verify=False, timeout=config.network_timeout,
    ).text
)

# 需要进一步研究 WebSockets 协议以进行进一步的实现
def test_port_forward(self):
    # 设置请求头，指定升级为 WebSockets 协议
    headers = {
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-Websocket-Key": "s",
        # 设置 WebSocket 请求的头部信息
        headers = {
            "Sec-Websocket-Version": "13",
            "Sec-Websocket-Protocol": "SPDY",
        }
        # 构建端口转发的 URL
        pf_url = self.path + KubeletHandlers.PORTFORWARD.value.format(
            pod_namespace=self.pod["namespace"], pod_id=self.pod["name"], port=80,
        )
        # 发起 GET 请求，获取端口转发的状态码
        self.session.get(
            pf_url, headers=headers, verify=False, stream=True, timeout=config.network_timeout,
        ).status_code == 200
        # TODO: what to return?  // 待确定返回什么值？

    # 执行一个命令并返回输出
    def test_run_container(self):
        # 构建执行容器命令的 URL
        run_url = self.path + KubeletHandlers.RUN.value.format(
            pod_namespace="test", pod_id="test", container_name="test", cmd="",
        )
        # 如果收到 Method Not Allowed 状态码，表示通过了身份验证和授权
        return self.session.get(run_url, verify=False, timeout=config.network_timeout).status_code == 405

    # 返回当前运行的 Pod 列表
# 测试正在运行的容器
def test_running_pods(self):
    # 构建获取正在运行的容器信息的 URL
    pods_url = self.path + KubeletHandlers.RUNNINGPODS.value
    # 发起 GET 请求，获取响应
    r = self.session.get(pods_url, verify=False, timeout=config.network_timeout)
    # 如果响应状态码为 200，则返回 JSON 格式的数据，否则返回 False
    return r.json() if r.status_code == 200 else False

# 需要进一步研究 attach 和 exec 之间的区别
def test_attach_container(self):
    # 构建 attach 容器的 URL
    attach_url = self.path + KubeletHandlers.ATTACH.value.format(
        pod_namespace=self.pod["namespace"],
        pod_id=self.pod["name"],
        container_name=self.pod["container"],
        cmd="",
    )
    # 发起 GET 请求，检查响应文本中是否包含 "/cri/attach/"，不允许重定向
    return (
        "/cri/attach/"
        in self.session.get(
            attach_url, allow_redirects=False, verify=False, timeout=config.network_timeout,
        ).text
    )
# 检查对日志端点的访问权限
def test_logs_endpoint(self):
    # 获取日志端点的 URL
    logs_url = self.session.get(
        self.path + KubeletHandlers.LOGS.value.format(path=""), timeout=config.network_timeout,
    ).text
    # 检查返回的文本中是否包含 "<pre>" 标签
    return "<pre>" in logs_url

# 返回用于运行 kubelet 的命令行
def test_pprof_cmdline(self):
    # 获取运行 kubelet 的命令行
    cmd = self.session.get(
        self.path + KubeletHandlers.PPROF_CMDLINE.value, verify=False, timeout=config.network_timeout,
    )
    # 如果返回状态码为 200，则返回命令行文本，否则返回 None
    return cmd.text if cmd.status_code == 200 else None

# 初始化方法，接收一个事件对象作为参数
def __init__(self, event):
    # 将事件对象保存到实例变量中
    self.event = event
    # 创建一个会话对象
    self.session = requests.Session()
    # 如果事件对象的 secure 属性为 True，则添加 Authorization 头部到会话对象中
    if self.event.secure:
        self.session.headers.update({"Authorization": f"Bearer {self.event.auth_token}"})
# 将当前会话的证书设置为事件的客户端证书
# 复制会话到事件
self.event.session = self.session
# 设置路径为指定主机的HTTPS地址
self.path = "https://{self.event.host}:10250"
# 设置kube-hunter_pod字典
self.kubehunter_pod = {
    "name": "kube-hunter",
    "namespace": "default",
    "container": "kube-hunter",
}
# 初始化pods_endpoint_data为空字符串
self.pods_endpoint_data = ""

# 获取pods的终结点
def get_pods_endpoint(self):
    # 发送GET请求获取pods的终结点数据
    response = self.session.get(f"{self.path}/pods", verify=False, timeout=config.network_timeout)
    # 如果响应文本中包含"items"，则返回JSON格式的响应数据
    if "items" in response.text:
        return response.json()

# 检查healthz终结点
def check_healthz_endpoint(self):
    # 发送GET请求检查healthz终结点
    r = requests.get(f"{self.path}/healthz", verify=False, timeout=config.network_timeout)
    # 如果响应状态码为200，则返回响应文本，否则返回False
    return r.text if r.status_code == 200 else False
# 执行函数，用于执行一系列操作
def execute(self):
    # 如果事件使用匿名身份验证，发布匿名身份验证已启用的事件
    if self.event.anonymous_auth:
        self.publish_event(AnonymousAuthEnabled())

    # 获取 pods 的端点数据
    self.pods_endpoint_data = self.get_pods_endpoint()
    # 检查健康检查端点
    healthz = self.check_healthz_endpoint()
    # 如果存在 pods 的端点数据，发布暴露的 pods 处理程序事件
    if self.pods_endpoint_data:
        self.publish_event(ExposedPodsHandler(pods=self.pods_endpoint_data["items"]))
    # 如果存在健康检查端点，发布暴露的健康检查处理程序事件
    if healthz:
        self.publish_event(ExposedHealthzHandler(status=healthz))
    # 测试处理程序
    self.test_handlers()

# 测试处理程序
def test_handlers(self):
    # 如果 kube-hunter 在一个 pod 中运行，使用 kube-hunter 的 pod 进行测试
    pod = self.kubehunter_pod if config.pod else self.get_random_pod()
    # 如果存在 pod，创建调试处理程序对象
    if pod:
        debug_handlers = self.DebugHandlers(self.path, pod, self.session)
        try:
            # TODO: 使用 Python 3.8 中引入的命名表达式
            # 测试运行中的 pods
            running_pods = debug_handlers.test_running_pods()
# 如果有正在运行的容器，则发布事件通知暴露运行中的容器处理程序
if running_pods:
    self.publish_event(ExposedRunningPodsHandler(count=len(running_pods["items"])))
# 获取调试处理程序的命令行
cmdline = debug_handlers.test_pprof_cmdline()
# 如果有命令行，则发布事件通知暴露 Kubelet 命令行处理程序
if cmdline:
    self.publish_event(ExposedKubeletCmdline(cmdline=cmdline))
# 如果测试容器日志成功，则发布事件通知暴露容器日志处理程序
if debug_handlers.test_container_logs():
    self.publish_event(ExposedContainerLogsHandler())
# 如果测试执行容器成功，则发布事件通知暴露执行处理程序
if debug_handlers.test_exec_container():
    self.publish_event(ExposedExecHandler())
# 如果测试运行容器成功，则发布事件通知暴露运行处理程序
if debug_handlers.test_run_container():
    self.publish_event(ExposedRunHandler())
# 如果测试端口转发成功，则发布事件通知暴露端口转发处理程序（未实现）
if debug_handlers.test_port_forward():
    self.publish_event(ExposedPortForwardHandler())  # 未实现
# 如果测试附加容器成功，则发布事件通知暴露附加处理程序
if debug_handlers.test_attach_container():
    self.publish_event(ExposedAttachHandler())
# 如果测试日志端点成功，则发布事件通知暴露系统日志处理程序
if debug_handlers.test_logs_endpoint():
    self.publish_event(ExposedSystemLogs())
# 捕获任何异常并记录调试处理程序测试失败
except Exception:
    logger.debug("Failed testing debug handlers", exc_info=True)
    # 从默认命名空间获取一个 pod，如果不存在，则获取一个 kube-system 的 pod
    def get_random_pod(self):
        # 如果存在 pods_endpoint_data，则获取其中的 items
        if self.pods_endpoint_data:
            pods_data = self.pods_endpoint_data["items"]

            # 定义判断是否为默认命名空间的 pod 的函数
            def is_default_pod(pod):
                return pod["metadata"]["namespace"] == "default" and pod["status"]["phase"] == "Running"

            # 定义判断是否为 kube-system 命名空间的 pod 的函数
            def is_kubesystem_pod(pod):
                return pod["metadata"]["namespace"] == "kube-system" and pod["status"]["phase"] == "Running"

            # 从 pods_data 中筛选出符合条件的默认命名空间的 pod
            pod_data = next(filter(is_default_pod, pods_data), None)
            # 如果不存在符合条件的默认命名空间的 pod，则从 pods_data 中筛选出符合条件的 kube-system 命名空间的 pod
            if not pod_data:
                pod_data = next(filter(is_kubesystem_pod, pods_data), None)

            # 如果存在符合条件的 pod，则获取其容器数据
            if pod_data:
                container_data = next(pod_data["spec"]["containers"], None)
                # 如果存在容器数据，则返回包含 pod 名称和容器数据的字典
                if container_data:
                    return {
                        "name": pod_data["metadata"]["name"],
# 定义一个包含容器名称和命名空间的字典
{
    "container": container_data["name"],  # 容器名称
    "namespace": pod_data["metadata"]["namespace"],  # 命名空间
}

# 订阅ExposedRunHandler事件，并定义ProveRunHandler类作为其处理程序
@handler.subscribe(ExposedRunHandler)
class ProveRunHandler(ActiveHunter):
    """Kubelet Run Hunter
    Executes uname inside of a random container
    """

    # 初始化方法，接收事件参数并设置基本路径
    def __init__(self, event):
        self.event = event  # 事件
        self.base_path = f"https://{self.event.host}:{self.event.port}"  # 基本路径

    # 执行方法，接收命令和容器参数
    def run(self, command, container):
        # 构建运行URL，包含容器的命名空间、ID和名称
        run_url = KubeletHandlers.RUN.value.format(
            pod_namespace=container["namespace"],  # 容器的命名空间
            pod_id=container["pod"],  # 容器的ID
            container_name=container["name"],  # 容器的名称
# 执行给定的命令，并返回执行结果
def execute(self):
    # 发送 GET 请求获取所有的 POD 数据
    r = self.event.session.get(
        self.base_path + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
    )
    # 检查返回的文本中是否包含"items"字段
    if "items" in r.text:
        # 解析返回的 JSON 数据，获取所有的 POD 数据
        pods_data = r.json()["items"]
        # 遍历每个 POD 数据
        for pod_data in pods_data:
            # 获取该 POD 中的容器数据
            container_data = next(pod_data["spec"]["containers"])
            # 如果存在容器数据
            if container_data:
                # 执行给定的命令，并返回执行结果
                output = self.run(
                    "uname -a",
                    container={
                        "namespace": pod_data["metadata"]["namespace"],
                        "pod": pod_data["metadata"]["name"],
# 订阅 ExposedContainerLogsHandler 事件的处理程序
@handler.subscribe(ExposedContainerLogsHandler)
class ProveContainerLogsHandler(ActiveHunter):
    """Kubelet Container Logs Hunter
    Retrieves logs from a random container
    """

    # 初始化方法，接收事件对象
    def __init__(self, event):
        self.event = event
        # 根据端口号确定协议
        protocol = "https" if self.event.port == 10250 else "http"
        # 构建基础 URL
        self.base_url = f"{protocol}://{self.event.host}:{self.event.port}/"

    # 执行方法
    def execute(self):
        # 获取容器数据中的名称
        container_name = container_data["name"]
        # 调用外部命令执行器执行命令
        output = execute_command(
            "kubectl logs {container_name}",
            {
                "name": container_data["name"],
            },
        )
        # 如果输出不为空且不包含 "exited with" 字符串，则更新事件证据
        if output and "exited with" not in output:
            self.event.evidence = "uname -a: " + output
            # 跳出循环
            break
# 从服务器获取原始的Pod数据
pods_raw = self.event.session.get(
    self.base_url + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
).text
# 检查原始数据中是否包含"items"字段
if "items" in pods_raw:
    # 解析原始数据，获取Pod数据列表
    pods_data = json.loads(pods_raw)["items"]
    # 遍历每个Pod数据
    for pod_data in pods_data:
        # 获取Pod中的容器数据
        container_data = next(pod_data["spec"]["containers"])
        # 如果存在容器数据
        if container_data:
            # 获取容器名称
            container_name = container_data["name"]
            # 从服务器获取容器日志数据
            output = requests.get(
                f"{self.base_url}/"
                + KubeletHandlers.CONTAINERLOGS.value.format(
                    pod_namespace=pod_data["metadata"]["namespace"],
                    pod_id=pod_data["metadata"]["name"],
                    container_name=container_name,
                ),
                verify=False,
                timeout=config.network_timeout,
            )
            # 检查获取日志数据的状态码和内容
            if output.status_code == 200 and output.text:
# 设置事件的证据为容器名称和输出文本的组合
self.event.evidence = f"{container_name}: {output.text}"
# 返回结果
return

# 订阅ExposedSystemLogs事件的处理程序
@handler.subscribe(ExposedSystemLogs)
class ProveSystemLogs(ActiveHunter):
    """Kubelet System Logs Hunter
    Retrieves commands from host's system audit
    """

    # 初始化函数，接受事件参数
    def __init__(self, event):
        self.event = event
        # 设置基本URL
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 执行函数
    def execute(self):
        # 获取系统审计日志
        audit_logs = self.event.session.get(
            f"{self.base_url}/" + KubeletHandlers.LOGS.value.format(path="audit/audit.log"),
            verify=False,
            timeout=config.network_timeout,
        ).text
# 使用调试日志记录主机的审计日志的前10条内容
logger.debug(f"Audit log of host {self.event.host}: {audit_logs[:10]}")
# 遍历 proctitles 并将它们转换为可读的字符串
proctitles = []
# 使用正则表达式找到 proctitle 并将其转换为可读的字符串，然后添加到 proctitles 列表中
for proctitle in re.findall(r"proctitle=(\w+)", audit_logs):
    proctitles.append(bytes.fromhex(proctitle).decode("utf-8").replace("\x00", " "))
# 将处理后的 proctitles 赋值给事件对象的 proctitles 属性
self.event.proctitles = proctitles
# 将 proctitles 作为证据添加到事件对象的 evidence 属性中
self.event.evidence = f"audit log: {proctitles}"
```