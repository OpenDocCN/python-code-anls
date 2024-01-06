# `KubiScan\engine\privleged_containers.py`

```
# 导入所需的模块
import engine.capabilities.capabilities as caps
from api import api_client

# 获取所有命名空间或特定命名空间的 pod 列表
def list_pods_for_all_namespaces_or_one_namspace(namespace=None):
    # 如果未指定命名空间，则获取所有命名空间的 pod 列表
    if namespace is None:
        pods = api_client.CoreV1Api.list_pod_for_all_namespaces(watch=False)
    # 如果指定了命名空间，则获取该命名空间的 pod 列表
    else:
        pods = api_client.CoreV1Api.list_namespaced_pod(namespace)
    return pods

# 获取指定命名空间的 pod 列表
def list_pods(namespace=None):
    return list_pods_for_all_namespaces_or_one_namspace(namespace)

# 检查安全上下文是否具有特权
def is_privileged(security_context, is_container=False):
    is_privileged = False
    # 如果安全上下文存在
    if security_context:
        # 如果运行用户是 root 用户
        if security_context.run_as_user == 0:
            is_privileged = True
        # 如果是容器，并且具有其他特权设置
        elif is_container:
# 如果安全上下文具有特权，则设置 is_privileged 为 True
if security_context.privileged:
    is_privileged = True
# 如果允许特权升级，则设置 is_privileged 为 True
elif security_context.allow_privilege_escalation:
    is_privileged = True
# 如果具有特权容器的能力，则设置 is_privileged 为 True
elif security_context.capabilities:
    # 如果具有添加能力，则遍历添加的能力列表
    if security_context.capabilities.add:
        for cap in security_context.capabilities.add:
            # 如果添加的能力在危险能力列表中，则设置 is_privileged 为 True，并跳出循环
            if cap in caps.dangerous_caps:
                is_privileged = True
                break
# 返回是否具有特权
return is_privileged

# 获取具有特权容器的列表
def get_privileged_containers(namespace=None):
    privileged_pods = []
    # 获取指定命名空间或所有命名空间的 Pod 列表
    pods = list_pods_for_all_namespaces_or_one_namspace(namespace)
    # 遍历 Pod 列表
    for pod in pods.items:
        privileged_containers = []
        # 如果 Pod 具有特权或者容器具有特权，则将容器添加到特权容器列表中
        if pod.spec.host_ipc or pod.spec.host_pid or pod.spec.host_network or is_privileged(pod.spec.security_context, is_container=False):
            privileged_containers = pod.spec.containers
        else:
# 遍历 Pod 中的容器
for container in pod.spec.containers:
    # 初始化标志位 found_privileged_container
    found_privileged_container = False
    # 检查容器是否具有特权权限，如果是则加入特权容器列表
    if is_privileged(container.security_context, is_container=True):
        privileged_containers.append(container)
    # 如果容器有端口
    elif container.ports:
        # 遍历容器的端口
        for ports in container.ports:
            # 如果端口有主机端口，则将容器加入特权容器列表，并设置标志位为 True，然后跳出循环
            if ports.host_port:
                privileged_containers.append(container)
                found_privileged_container = True
                break
    # 如果没有找到特权容器
    if not found_privileged_container:
        # 如果 Pod 中有卷
        if pod.spec.volumes is not None:
            # 遍历 Pod 中的卷
            for volume in pod.spec.volumes:
                # 如果已经找到特权容器，则跳出循环
                if found_privileged_container:
                    break
                # 如果卷是主机路径类型
                if volume.host_path:
                    # 遍历容器的卷挂载
                    for volume_mount in container.volume_mounts:
                        # 如果卷挂载的名称与卷的名称相同，则将容器加入特权容器列表，并设置标志位为 True
                        if volume_mount.name == volume.name:
                            privileged_containers.append(container)
                            found_privileged_container = True
# 如果存在特权容器列表
if privileged_containers:
    # 将特权容器列表赋值给 pod 对象的 containers 属性
    pod.spec.containers = privileged_containers
    # 将包含特权容器的 pod 对象添加到特权 pod 列表中
    privileged_pods.append(pod)

# 返回特权 pod 列表
return privileged_pods
```