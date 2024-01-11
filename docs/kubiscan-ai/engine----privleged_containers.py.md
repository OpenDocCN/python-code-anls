# `KubiScan\engine\privleged_containers.py`

```
# 导入所需的模块
import engine.capabilities.capabilities as caps
from api import api_client

# 定义函数，列出所有命名空间的 pod 或者指定命名空间的 pod
def list_pods_for_all_namespaces_or_one_namspace(namespace=None):
    # 如果未指定命名空间，则列出所有命名空间的 pod
    if namespace is None:
        pods = api_client.CoreV1Api.list_pod_for_all_namespaces(watch=False)
    # 否则，列出指定命名空间的 pod
    else:
        pods = api_client.CoreV1Api.list_namespaced_pod(namespace)
    return pods

# 定义函数，列出指定命名空间的 pod
def list_pods(namespace=None):
    return list_pods_for_all_namespaces_or_one_namspace(namespace)

# 定义函数，判断容器是否具有特权
def is_privileged(security_context, is_container=False):
    is_privileged = False
    # 如果存在安全上下文
    if security_context:
        # 如果运行用户是 root 用户
        if security_context.run_as_user == 0:
            is_privileged = True
        # 如果是容器，并且具有特权
        elif is_container:
            if security_context.privileged:
                is_privileged = True
            # 如果允许特权升级
            elif security_context.allow_privilege_escalation:
                is_privileged = True
            # 如果具有危险的系统能力
            elif security_context.capabilities:
                if security_context.capabilities.add:
                    for cap in security_context.capabilities.add:
                        if cap in caps.dangerous_caps:
                            is_privileged = True
                            break
    return is_privileged

# 定义函数，获取具有特权的容器
def get_privileged_containers(namespace=None):
    privileged_pods = []
    # 获取指定命名空间的 pod
    pods = list_pods_for_all_namespaces_or_one_namspace(namespace)
    # 遍历所有的 pods
    for pod in pods.items:
        # 初始化特权容器列表
        privileged_containers = []
        # 检查是否为特权容器，如果是则直接将所有容器添加到特权容器列表中
        if pod.spec.host_ipc or pod.spec.host_pid or pod.spec.host_network or is_privileged(pod.spec.security_context, is_container=False):
            privileged_containers = pod.spec.containers
        else:
            # 遍历每个容器
            for container in pod.spec.containers:
                # 初始化特权容器标志
                found_privileged_container = False
                # 检查容器是否为特权容器，如果是则添加到特权容器列表中
                if is_privileged(container.security_context, is_container=True):
                    privileged_containers.append(container)
                # 如果容器有端口，则检查端口是否为主机端口，如果是则添加到特权容器列表中
                elif container.ports:
                    for ports in container.ports:
                        if ports.host_port:
                            privileged_containers.append(container)
                            found_privileged_container = True
                            break
                # 如果容器不是特权容器，且没有主机端口，则检查容器的卷是否为主机路径，如果是则添加到特权容器列表中
                if not found_privileged_container:
                    if pod.spec.volumes is not None:
                        for volume in pod.spec.volumes:
                            if found_privileged_container:
                                break
                            if volume.host_path:
                                for volume_mount in container.volume_mounts:
                                    if volume_mount.name == volume.name:
                                        privileged_containers.append(container)
                                        found_privileged_container = True
                                        break
        # 如果存在特权容器，则将特权容器列表赋值给 pod 的容器列表，并添加到特权 pods 列表中
        if privileged_containers:
            pod.spec.containers = privileged_containers
            privileged_pods.append(pod)

    # 返回特权 pods 列表
    return privileged_pods
```