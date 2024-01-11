# `kubehunter\kube_hunter\modules\hunting\kubelet.py`

```
# 导入所需的模块
import json
import logging
from enum import Enum
import re
import requests
import urllib3
# 导入自定义模块
from kube_hunter.conf import config
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Vulnerability, Event, K8sVersionDisclosure
from kube_hunter.core.types import (
    Hunter,
    ActiveHunter,
    KubernetesCluster,
    Kubelet,
    InformationDisclosure,
    RemoteCodeExec,
    AccessRisk,
)
from kube_hunter.modules.discovery.kubelet import (
    ReadOnlyKubeletEvent,
    SecureKubeletEvent,
)

# 获取日志记录器
logger = logging.getLogger(__name__)
# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

""" Vulnerabilities """

# 定义暴露的 Pods 漏洞类
class ExposedPodsHandler(Vulnerability, Event):
    """An attacker could view sensitive information about pods that are
    bound to a Node using the /pods endpoint"""

    def __init__(self, pods):
        # 初始化漏洞信息
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Pods", category=InformationDisclosure,
        )
        self.pods = pods
        self.evidence = f"count: {len(self.pods)}"

# 定义匿名认证启用漏洞类
class AnonymousAuthEnabled(Vulnerability, Event):
    """The kubelet is misconfigured, potentially allowing secure access to all requests on the kubelet,
    without the need to authenticate"""

    def __init__(self):
        # 初始化漏洞信息
        Vulnerability.__init__(
            self, component=Kubelet, name="Anonymous Authentication", category=RemoteCodeExec, vid="KHV036",
        )

# 定义暴露的容器日志漏洞类
class ExposedContainerLogsHandler(Vulnerability, Event):
    """Output logs from a running container are using the exposed /containerLogs endpoint"""

    def __init__(self):
        # 初始化漏洞信息
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Container Logs", category=InformationDisclosure, vid="KHV037",
        )

# 定义暴露的运行中 Pods 漏洞类
class ExposedRunningPodsHandler(Vulnerability, Event):
    """Outputs a list of currently running pods,
    and some of their metadata, which can reveal sensitive information"""
    # 初始化方法，接受一个参数 count
    def __init__(self, count):
        # 调用父类Vulnerability的初始化方法，设置组件、名称、类别和漏洞ID
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Running Pods", category=InformationDisclosure, vid="KHV038",
        )
        # 设置当前对象的 count 属性
        self.count = count
        # 根据 count 属性生成 evidence 属性，表示暴露的运行中的 pods 数量
        self.evidence = "{} running pods".format(self.count)
# 定义一个暴露执行处理程序类，继承自漏洞和事件类
class ExposedExecHandler(Vulnerability, Event):
    """An attacker could run arbitrary commands on a container"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self):
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Exec On Container", category=RemoteCodeExec, vid="KHV039",
        )


# 定义一个暴露运行处理程序类，继承自漏洞和事件类
class ExposedRunHandler(Vulnerability, Event):
    """An attacker could run an arbitrary command inside a container"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self):
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Run Inside Container", category=RemoteCodeExec, vid="KHV040",
        )


# 定义一个暴露端口转发处理程序类，继承自漏洞和事件类
class ExposedPortForwardHandler(Vulnerability, Event):
    """An attacker could set port forwarding rule on a pod"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self):
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Port Forward", category=RemoteCodeExec, vid="KHV041",
        )


# 定义一个暴露附加处理程序类，继承自漏洞和事件类
class ExposedAttachHandler(Vulnerability, Event):
    """Opens a websocket that could enable an attacker
    to attach to a running container"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID
    def __init__(self):
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Attaching To Container", category=RemoteCodeExec, vid="KHV042",
        )


# 定义一个暴露健康检查处理程序类，继承自漏洞和事件类
class ExposedHealthzHandler(Vulnerability, Event):
    """By accessing the open /healthz handler,
    an attacker could get the cluster health state without authenticating"""

    # 初始化方法，设置组件、名称、类别和漏洞 ID，并初始化状态和证据
    def __init__(self, status):
        Vulnerability.__init__(
            self, component=Kubelet, name="Cluster Health Disclosure", category=InformationDisclosure, vid="KHV043",
        )
        self.status = status
        self.evidence = f"status: {self.status}"


# 定义一个特权容器类，继承自漏洞和事件类
class PrivilegedContainers(Vulnerability, Event):
    """A Privileged container exist on a node
    could expose the node/cluster to unwanted root operations"""
    # 初始化方法，用于初始化对象
    def __init__(self, containers):
        # 调用父类Vulnerability的初始化方法，设置组件、名称、类别和漏洞ID
        Vulnerability.__init__(
            self, component=KubernetesCluster, name="Privileged Container", category=AccessRisk, vid="KHV044",
        )
        # 设置对象的containers属性为传入的containers参数
        self.containers = containers
        # 设置对象的evidence属性为包含容器信息的字符串
        self.evidence = f"pod: {containers[0][0]}, " f"container: {containers[0][1]}, " f"count: {len(containers)}"
# 定义一个名为ExposedSystemLogs的类，继承自Vulnerability和Event类
class ExposedSystemLogs(Vulnerability, Event):
    """System logs are exposed from the /logs endpoint on the kubelet"""

    # 初始化ExposedSystemLogs类
    def __init__(self):
        # 调用Vulnerability类的初始化方法，设置组件、名称、类别和vid
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed System Logs", category=InformationDisclosure, vid="KHV045",
        )


# 定义一个名为ExposedKubeletCmdline的类，继承自Vulnerability和Event类
class ExposedKubeletCmdline(Vulnerability, Event):
    """Commandline flags that were passed to the kubelet can be obtained from the pprof endpoints"""

    # 初始化ExposedKubeletCmdline类，接受cmdline参数
    def __init__(self, cmdline):
        # 调用Vulnerability类的初始化方法，设置组件、名称、类别和vid
        Vulnerability.__init__(
            self, component=Kubelet, name="Exposed Kubelet Cmdline", category=InformationDisclosure, vid="KHV046",
        )
        # 设置cmdline属性
        self.cmdline = cmdline
        # 设置evidence属性，包含cmdline信息
        self.evidence = f"cmdline: {self.cmdline}"


# 定义一个名为KubeletHandlers的枚举类
class KubeletHandlers(Enum):
    # GET请求，获取pods信息
    PODS = "pods"
    # GET请求，获取指定容器的日志
    CONTAINERLOGS = "containerLogs/{pod_namespace}/{pod_id}/{container_name}"
    # GET请求，获取正在运行的pods信息
    RUNNINGPODS = "runningpods"
    # GET请求，通过WebSocket执行命令
    EXEC = "exec/{pod_namespace}/{pod_id}/{container_name}?command={cmd}&input=1&output=1&tty=1"
    # POST请求，由于历史原因，使用不同的查询参数
    RUN = "run/{pod_namespace}/{pod_id}/{container_name}?cmd={cmd}"
    # GET/POST请求，端口转发
    PORTFORWARD = "portForward/{pod_namespace}/{pod_id}?port={port}"
    # GET请求，通过WebSocket附加到容器
    ATTACH = "attach/{pod_namespace}/{pod_id}/{container_name}?command={cmd}&input=1&output=1&tty=1"
    # GET请求，获取日志信息
    LOGS = "logs/{path}"
    # GET请求，获取pprof的cmdline信息
    PPROF_CMDLINE = "debug/pprof/cmdline"


# 订阅ReadOnlyKubeletEvent事件的处理器
@handler.subscribe(ReadOnlyKubeletEvent)
class ReadOnlyKubeletPortHunter(Hunter):
    """Kubelet Readonly Ports Hunter
    Hunts specific endpoints on open ports in the readonly Kubelet server
    """

    # 初始化ReadOnlyKubeletPortHunter类，接受event参数
    def __init__(self, event):
        # 设置event属性
        self.event = event
        # 构建路径信息，包含主机和端口
        self.path = f"http://{self.event.host}:{self.event.port}"
        # 初始化pods_endpoint_data属性为空字符串
        self.pods_endpoint_data = ""
    # 获取 Kubernetes 版本信息
    def get_k8s_version(self):
        # 记录调试信息
        logger.debug("Passive hunter is attempting to find kubernetes version")
        # 发起 GET 请求获取 metrics 数据
        metrics = requests.get(f"{self.path}/metrics", timeout=config.network_timeout).text
        # 遍历 metrics 数据的每一行
        for line in metrics.split("\n"):
            # 寻找包含 kubernetes_build_info 的行
            if line.startswith("kubernetes_build_info"):
                # 从该行中提取 kubernetes 版本信息
                for info in line[line.find("{") + 1 : line.find("}")].split(","):
                    k, v = info.split("=")
                    if k == "gitVersion":
                        return v.strip('"')

    # 返回特权容器及其所在的 pod 的元组列表
    def find_privileged_containers(self):
        # 记录调试信息
        logger.debug("Trying to find privileged containers and their pods")
        # 初始化特权容器列表
        privileged_containers = []
        # 如果存在 pods_endpoint_data
        if self.pods_endpoint_data:
            # 遍历每个 pod
            for pod in self.pods_endpoint_data["items"]:
                # 遍历每个 pod 的容器
                for container in pod["spec"]["containers"]:
                    # 如果容器有特权权限，将其信息加入特权容器列表
                    if container.get("securityContext", {}).get("privileged"):
                        privileged_containers.append((pod["metadata"]["name"], container["name"]))
        # 如果特权容器列表不为空，则返回列表，否则返回 None
        return privileged_containers if len(privileged_containers) > 0 else None

    # 获取 pods endpoint 数据
    def get_pods_endpoint(self):
        # 记录调试信息
        logger.debug("Attempting to find pods endpoints")
        # 发起 GET 请求获取 pods 数据
        response = requests.get(f"{self.path}/pods", timeout=config.network_timeout)
        # 如果 response 中包含 items 字段，则返回其 JSON 格式数据
        if "items" in response.text:
            return response.json()

    # 检查健康检查 endpoint
    def check_healthz_endpoint(self):
        # 发起 GET 请求检查健康检查 endpoint
        r = requests.get(f"{self.path}/healthz", verify=False, timeout=config.network_timeout)
        # 如果返回状态码为 200，则返回响应文本，否则返回 False
        return r.text if r.status_code == 200 else False
    # 执行函数，用于执行一系列操作
    def execute(self):
        # 获取 pods 的 endpoint 数据
        self.pods_endpoint_data = self.get_pods_endpoint()
        # 获取 Kubernetes 版本信息
        k8s_version = self.get_k8s_version()
        # 查找特权容器
        privileged_containers = self.find_privileged_containers()
        # 检查健康检查端点
        healthz = self.check_healthz_endpoint()
        # 如果获取到 Kubernetes 版本信息
        if k8s_version:
            # 发布 K8sVersionDisclosure 事件
            self.publish_event(
                K8sVersionDisclosure(version=k8s_version, from_endpoint="/metrics", extra_info="on Kubelet")
            )
        # 如果存在特权容器
        if privileged_containers:
            # 发布 PrivilegedContainers 事件
            self.publish_event(PrivilegedContainers(containers=privileged_containers))
        # 如果存在健康检查端点
        if healthz:
            # 发布 ExposedHealthzHandler 事件
            self.publish_event(ExposedHealthzHandler(status=healthz))
        # 如果存在 pods 的 endpoint 数据
        if self.pods_endpoint_data:
            # 发布 ExposedPodsHandler 事件，传入 pods 数据
            self.publish_event(ExposedPodsHandler(pods=self.pods_endpoint_data["items"]))
# 订阅 SecureKubeletEvent 事件的处理程序
@handler.subscribe(SecureKubeletEvent)
class SecureKubeletPortHunter(Hunter):
    """Kubelet Secure Ports Hunter
    Hunts specific endpoints on an open secured Kubelet
    """

    # 初始化方法，接收事件对象作为参数
    def __init__(self, event):
        # 保存事件对象
        self.event = event
        # 创建一个会话对象
        self.session = requests.Session()
        # 如果事件对象是安全的
        if self.event.secure:
            # 更新会话对象的请求头，添加认证信息
            self.session.headers.update({"Authorization": f"Bearer {self.event.auth_token}"})
            # 设置会话对象的证书
            # self.session.cert = self.event.client_cert
        # 将会话对象复制给事件对象
        self.event.session = self.session
        # 设置请求路径
        self.path = "https://{self.event.host}:10250"
        # 定义 kube-hunter pod 的信息
        self.kubehunter_pod = {
            "name": "kube-hunter",
            "namespace": "default",
            "container": "kube-hunter",
        }
        # 初始化 pods_endpoint_data
        self.pods_endpoint_data = ""

    # 获取 pods 的端点数据
    def get_pods_endpoint(self):
        # 发送 GET 请求获取 pods 数据
        response = self.session.get(f"{self.path}/pods", verify=False, timeout=config.network_timeout)
        # 如果响应中包含 "items" 字段，则返回 JSON 格式的数据
        if "items" in response.text:
            return response.json()

    # 检查 healthz 端点
    def check_healthz_endpoint(self):
        # 发送 GET 请求检查 healthz 端点
        r = requests.get(f"{self.path}/healthz", verify=False, timeout=config.network_timeout)
        # 如果状态码为 200，则返回响应文本，否则返回 False
        return r.text if r.status_code == 200 else False

    # 执行方法
    def execute(self):
        # 如果事件对象使用匿名认证
        if self.event.anonymous_auth:
            # 发布事件，表示匿名认证已启用
            self.publish_event(AnonymousAuthEnabled())

        # 获取 pods 端点数据
        self.pods_endpoint_data = self.get_pods_endpoint()
        # 检查 healthz 端点
        healthz = self.check_healthz_endpoint()
        # 如果存在 pods 端点数据，则发布事件，表示暴露的 pods
        if self.pods_endpoint_data:
            self.publish_event(ExposedPodsHandler(pods=self.pods_endpoint_data["items"]))
        # 如果存在 healthz 数据，则发布事件，表示暴露的 healthz
        if healthz:
            self.publish_event(ExposedHealthzHandler(status=healthz))
        # 测试处理程序
        self.test_handlers()
    # 测试处理程序的方法
    def test_handlers(self):
        # 如果 kube-hunter 在一个 pod 中运行，就使用 kube-hunter 的 pod 进行测试
        pod = self.kubehunter_pod if config.pod else self.get_random_pod()
        # 如果存在 pod
        if pod:
            # 创建 DebugHandlers 对象
            debug_handlers = self.DebugHandlers(self.path, pod, self.session)
            try:
                # TODO: 使用 Python 3.8 中引入的命名表达式
                # 测试运行中的 pods
                running_pods = debug_handlers.test_running_pods()
                # 如果存在运行中的 pods，发布事件
                if running_pods:
                    self.publish_event(ExposedRunningPodsHandler(count=len(running_pods["items"])))
                # 测试 pprof 命令行
                cmdline = debug_handlers.test_pprof_cmdline()
                # 如果存在 cmdline，发布事件
                if cmdline:
                    self.publish_event(ExposedKubeletCmdline(cmdline=cmdline))
                # 测试容器日志
                if debug_handlers.test_container_logs():
                    self.publish_event(ExposedContainerLogsHandler())
                # 测试执行容器
                if debug_handlers.test_exec_container():
                    self.publish_event(ExposedExecHandler())
                # 测试运行容器
                if debug_handlers.test_run_container():
                    self.publish_event(ExposedRunHandler())
                # 测试端口转发
                if debug_handlers.test_port_forward():
                    self.publish_event(ExposedPortForwardHandler())  # 未实现
                # 测试附加容器
                if debug_handlers.test_attach_container():
                    self.publish_event(ExposedAttachHandler())
                # 测试日志端点
                if debug_handlers.test_logs_endpoint():
                    self.publish_event(ExposedSystemLogs())
            except Exception:
                # 记录异常信息
                logger.debug("Failed testing debug handlers", exc_info=True)

    # 尝试从默认命名空间获取一个 pod，如果不存在，则获取一个 kube-system 的 pod
    # 获取一个随机的 Pod 数据
    def get_random_pod(self):
        # 如果存在 pods_endpoint_data
        if self.pods_endpoint_data:
            # 获取所有的 pods 数据
            pods_data = self.pods_endpoint_data["items"]

            # 定义判断是否为默认 Pod 的函数
            def is_default_pod(pod):
                return pod["metadata"]["namespace"] == "default" and pod["status"]["phase"] == "Running"

            # 定义判断是否为 kube-system Pod 的函数
            def is_kubesystem_pod(pod):
                return pod["metadata"]["namespace"] == "kube-system" and pod["status"]["phase"] == "Running"

            # 从 pods_data 中找到第一个符合条件的默认 Pod
            pod_data = next(filter(is_default_pod, pods_data), None)
            # 如果没有找到默认 Pod，则从 pods_data 中找到第一个符合条件的 kube-system Pod
            if not pod_data:
                pod_data = next(filter(is_kubesystem_pod, pods_data), None)

            # 如果存在符合条件的 Pod 数据
            if pod_data:
                # 获取 Pod 中的容器数据
                container_data = next(pod_data["spec"]["containers"], None)
                # 如果存在容器数据，则返回包含 Pod 名称、容器名称和命名空间的字典
                if container_data:
                    return {
                        "name": pod_data["metadata"]["name"],
                        "container": container_data["name"],
                        "namespace": pod_data["metadata"]["namespace"],
                    }
# 订阅 ExposedRunHandler 事件，并创建 ProveRunHandler 类
class ProveRunHandler(ActiveHunter):
    """Kubelet Run Hunter
    Executes uname inside of a random container
    """

    # 初始化方法，接收 event 参数
    def __init__(self, event):
        # 将 event 参数赋值给实例变量 event
        self.event = event
        # 构建 base_path，使用 event 的 host 和 port 构成 URL
        self.base_path = f"https://{self.event.host}:{self.event.port}"

    # 定义 run 方法，接收 command 和 container 参数
    def run(self, command, container):
        # 构建 run_url，使用 KubeletHandlers.RUN.value 的格式化字符串
        run_url = KubeletHandlers.RUN.value.format(
            pod_namespace=container["namespace"],
            pod_id=container["pod"],
            container_name=container["name"],
            cmd=command,
        )
        # 发起 POST 请求，执行命令，并返回结果
        return self.event.session.post(
            f"{self.base_path}/{run_url}", verify=False, timeout=config.network_timeout,
        ).text

    # 定义 execute 方法
    def execute(self):
        # 发起 GET 请求，获取所有 pods 的数据
        r = self.event.session.get(
            self.base_path + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
        )
        # 检查返回的文本中是否包含 "items"
        if "items" in r.text:
            # 解析返回的 JSON 数据，获取所有 pods 的数据
            pods_data = r.json()["items"]
            # 遍历每个 pod 的数据
            for pod_data in pods_data:
                # 获取第一个容器的数据
                container_data = next(pod_data["spec"]["containers"])
                # 如果容器数据存在
                if container_data:
                    # 执行 run 方法，获取容器的 uname -a 输出
                    output = self.run(
                        "uname -a",
                        container={
                            "namespace": pod_data["metadata"]["namespace"],
                            "pod": pod_data["metadata"]["name"],
                            "name": container_data["name"],
                        },
                    )
                    # 如果输出存在且不包含 "exited with"
                    if output and "exited with" not in output:
                        # 设置 event 的 evidence 为 "uname -a: " + output，并结束循环
                        self.event.evidence = "uname -a: " + output
                        break


# 订阅 ExposedContainerLogsHandler 事件，并创建 ProveContainerLogsHandler 类
@handler.subscribe(ExposedContainerLogsHandler)
class ProveContainerLogsHandler(ActiveHunter):
    """Kubelet Container Logs Hunter
    Retrieves logs from a random container
    """

    # 初始化方法，接收 event 参数
    def __init__(self, event):
        # 将 event 参数赋值给实例变量 event
        self.event = event
        # 根据 event 的 port 确定协议，构建 base_url
        protocol = "https" if self.event.port == 10250 else "http"
        self.base_url = f"{protocol}://{self.event.host}:{self.event.port}/"
    # 定义一个方法，用于执行一些操作
    def execute(self):
        # 发送请求获取所有的 POD 数据
        pods_raw = self.event.session.get(
            self.base_url + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
        ).text
        # 检查返回的数据中是否包含"items"字段
        if "items" in pods_raw:
            # 解析获取到的 JSON 数据，提取出"items"字段对应的值
            pods_data = json.loads(pods_raw)["items"]
            # 遍历每个 POD 数据
            for pod_data in pods_data:
                # 获取当前 POD 中的容器数据
                container_data = next(pod_data["spec"]["containers"])
                # 如果存在容器数据
                if container_data:
                    # 获取容器的名称
                    container_name = container_data["name"]
                    # 发送请求获取容器日志
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
                    # 如果请求成功并且返回了日志数据
                    if output.status_code == 200 and output.text:
                        # 将容器名称和日志内容作为事件的证据
                        self.event.evidence = f"{container_name}: {output.text}"
                        # 结束方法的执行
                        return
# 订阅ExposedSystemLogs事件的处理程序类
@handler.subscribe(ExposedSystemLogs)
class ProveSystemLogs(ActiveHunter):
    """Kubelet System Logs Hunter
    Retrieves commands from host's system audit
    """

    # 初始化方法，接收事件对象作为参数
    def __init__(self, event):
        self.event = event
        self.base_url = f"https://{self.event.host}:{self.event.port}"

    # 执行方法
    def execute(self):
        # 获取主机系统审计日志
        audit_logs = self.event.session.get(
            f"{self.base_url}/" + KubeletHandlers.LOGS.value.format(path="audit/audit.log"),
            verify=False,
            timeout=config.network_timeout,
        ).text
        # 记录主机审计日志的前10个字符
        logger.debug(f"Audit log of host {self.event.host}: {audit_logs[:10]}")
        # 遍历proctitles并将它们转换为可读字符串
        proctitles = []
        for proctitle in re.findall(r"proctitle=(\w+)", audit_logs):
            proctitles.append(bytes.fromhex(proctitle).decode("utf-8").replace("\x00", " "))
        # 将处理后的proctitles赋值给事件对象
        self.event.proctitles = proctitles
        # 将审计日志作为证据赋值给事件对象
        self.event.evidence = f"audit log: {proctitles}"
```