# `KubiScan\engine\utils.py`

```
# 导入 requests 模块，用于发送 HTTP 请求
import requests

# 从 engine.role 模块中导入 Role 类
from engine.role import Role
# 从 engine.priority 模块中导入 Priority 类
from engine.priority import Priority
# 从 static_risky_roles 模块中导入 STATIC_RISKY_ROLES 列表
from static_risky_roles import STATIC_RISKY_ROLES
# 从 engine.role_binding 模块中导入 RoleBinding 类
from engine.role_binding import RoleBinding
# 从 kubernetes.stream 模块中导入 stream 函数
from kubernetes.stream import stream
# 从 engine.pod 模块中导入 Pod 类
from engine.pod import Pod
# 从 engine.container 模块中导入 Container 类
from engine.container import Container
# 导入 json 模块
import json
# 从 api 模块中导入 api_client
from api import api_client
# 从 engine.subject 模块中导入 Subject 类
from engine.subject import Subject
# 从 misc.constants 模块中导入一些常量
from misc.constants import *
# 从 kubernetes.client.rest 模块中导入 ApiException 类
from kubernetes.client.rest import ApiException
# 导入 urllib3 模块，用于禁用 SSL 警告
import urllib3
# 禁用 SSL 请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 定义一个区域 - Roles and ClusteRoles
# 创建一个空列表，用于存储服务账户
list_of_service_accounts = []

# 检查是否存在风险资源名称
def is_risky_resource_name_exist(source_rolename, source_resourcenames):
    # 初始化风险标志为False
    is_risky = False
    # 遍历资源名称列表
    for resource_name in source_resourcenames:
        # 防止循环
        if resource_name != source_rolename:
            # TODO: 需要考虑命名空间，允许对 'roles' 资源名称进行检查
            # 通过名称和类型获取角色
            role = get_role_by_name_and_kind(resource_name, CLUSTER_ROLE_KIND)
            # 如果角色存在
            if role is not None:
                # 检查角色是否存在风险
                is_risky, priority = is_risky_role(role)
                # 如果存在风险，跳出循环
                if is_risky:
                    break
    # 返回是否存在风险
    return is_risky

# 检查规则是否包含风险规则
def is_rule_contains_risky_rule(source_role_name, source_rule, risky_rule):
    # 初始化包含标志为True
    is_contains = True
    # 初始化绑定动作标志为False
    is_bind_verb_found = False
    # 初始化角色资源标志为False
    is_role_resource_found = False
# 可选：取消注释并将所有内容移到'return'之前以添加任何具有"*"的动词或资源的规则。
# 目前它在risky_roles.yaml中部分处理
# 如果（source_rule.verbs不为空且source_rule.verbs中没有"*"）并且（source_rule.resources不为空且source_rule.resources中没有"*"）
for verb in risky_rule.verbs:
    # 如果risky_rule中的动词不在source_rule中的动词列表中，则is_contains为False
    if verb not in source_rule.verbs:
        is_contains = False
        break

    # 如果动词为"bind"，则is_bind_verb_found为True
    if verb.lower() == "bind":
        is_bind_verb_found = True

# 如果is_contains为True且source_rule.resources不为空
if is_contains and source_rule.resources is not None:
    for resource in risky_rule.resources:
        # 如果risky_rule中的资源不在source_rule中的资源列表中，则is_contains为False
        if resource not in source_rule.resources:
            is_contains = False
            break
        # 如果资源为"roles"或"clusterroles"，则is_role_resource_found为True
        if resource.lower() == "roles" or resource.lower() == "clusterroles":
            is_role_resource_found = True
# 如果 is_contains 为真且 risky_rule.resource_names 不为空，则将 is_contains 设为假
# 如果 is_bind_verb_found 为真且 is_role_resource_found 为真，则检查是否存在风险资源名，如果是，则将 is_contains 设为真
# 否则将 is_contains 设为假
def is_contains_risky_resource(is_contains, risky_rule, source_role_name, is_bind_verb_found, is_role_resource_found):
    if is_contains and risky_rule.resource_names is not None:
        is_contains = False
        if is_bind_verb_found and is_role_resource_found:
            is_risky = is_risky_resource_name_exist(source_role_name, source_rule.resource_names)
            if is_risky:
                is_contains = True
    else:
        is_contains = False
    return is_contains

# 获取当前版本号
def get_current_version(certificate_authority_file=None, client_certificate_file=None, client_key_file=None, host=None):
    # 如果 host 为空，则从 api_client 获取版本号并返回
    if host is None:
        version = api_client.api_version.get_code().git_version
        return version.replace('v', "")
    # 否则，如果证书文件和密钥文件都为空，则向指定 host 发送请求获取版本号
    else:
        if certificate_authority_file is None and client_certificate_file is None and client_key_file is None:
            response = requests.get(host + '/version', verify=False)
            # 如果响应状态码不是 200，则...
# 打印响应内容
print(response.text)
# 返回空值
return None
# 如果响应状态码不是 200，则打印响应内容并返回空值
else:
    return response.json()["gitVersion"].replace('v', "")
# 如果证书机构文件、客户端证书文件和客户端密钥文件都不为空
if certificate_authority_file is not None and client_certificate_file is not None and client_key_file is not None:
    # 发送带证书的 GET 请求
    response = requests.get(host + '/version', cert=(client_certificate_file, client_key_file),
                            verify=certificate_authority_file)
    # 如果响应状态码不是 200，则打印响应内容并返回空值
    if response.status_code != 200:
        print(response.text)
        return None
    # 否则，返回响应中的 git 版本号
    else:
        return response.json()["gitVersion"].replace('v', "")
# 如果证书机构文件、客户端证书文件或客户端密钥文件为空，或者主机地址为空
if certificate_authority_file is None or client_certificate_file is None or client_key_file is None or host is None:
    # 打印提示信息
    print("Please provide certificate authority file path, client certificate file path,"
          " client key file path and host address")
    # 返回空值
    return None
# 发送带证书的 GET 请求
response = requests.get(host + '/version', cert=(client_certificate_file, client_key_file),
                        verify=certificate_authority_file)
# 如果响应状态码不是 200，则打印响应内容
if response.status_code != 200:
    print(response.text)
# 如果条件不满足，返回 None
return None
# 如果条件满足，返回 response.json()["gitVersion"] 中去掉 'v' 后的字符串
else:
    return response.json()["gitVersion"].replace('v', "")

# 根据名称和类型获取角色
def get_role_by_name_and_kind(name, kind, namespace=None):
    requested_role = None
    # 获取指定类型的所有角色
    roles = get_roles_by_kind(kind)
    # 遍历所有角色
    for role in roles.items:
        # 如果角色的名称与指定名称相同，则将该角色赋值给 requested_role，并跳出循环
        if role.metadata.name == name:
            requested_role = role
            break
    # 返回获取到的角色
    return requested_role

# 判断规则是否包含其他规则
def are_rules_contain_other_rules(source_role_name, source_rules, target_rules):
    is_contains = False
    matched_rules = 0
# 如果目标规则和源规则都为空，则返回 is_contains
if not (target_rules and source_rules):
    return is_contains

# 遍历目标规则列表
for target_rule in target_rules:
    # 如果源规则不为空
    if source_rules is not None:
        # 遍历源规则列表
        for source_rule in source_rules:
            # 如果规则包含风险规则
            if is_rule_contains_risky_rule(source_role_name, source_rule, target_rule):
                # 匹配的规则数加一
                matched_rules += 1
                # 如果匹配的规则数等于目标规则的总数
                if matched_rules == len(target_rules):
                    # 设置 is_contains 为 True，并返回
                    is_contains = True
                    return is_contains

# 返回 is_contains
return is_contains

# 判断角色是否具有风险
def is_risky_role(role):
    # 初始化 is_risky 为 False
    is_risky = False
    # 初始化优先级为 LOW
    priority = Priority.LOW
    # 遍历静态风险角色列表
    for risky_role in STATIC_RISKY_ROLES:
        # 如果角色的规则包含其他规则
        if are_rules_contain_other_rules(role.metadata.name, role.rules, risky_role.rules):
            # 设置 is_risky 为 True
            is_risky = True
# 获取风险角色的优先级
priority = risky_role.priority
# 结束循环
break

# 返回是否为风险角色和优先级
return is_risky, priority


# 查找风险角色
def find_risky_roles(roles, kind):
    # 存储风险角色的列表
    risky_roles = []
    # 遍历所有角色
    for role in roles:
        # 判断角色是否为风险角色，并获取其优先级
        is_risky, priority = is_risky_role(role)
        # 如果是风险角色，则将其添加到风险角色列表中
        if is_risky:
            risky_roles.append(
                Role(role.metadata.name, priority, rules=role.rules, namespace=role.metadata.namespace, kind=kind,
                     time=role.metadata.creation_timestamp))
    # 返回风险角色列表
    return risky_roles


# 根据种类获取角色
def get_roles_by_kind(kind):
    # 存储所有角色的列表
    all_roles = []
# 如果 kind 等于 ROLE_KIND，则调用 api_client.RbacAuthorizationV1Api.list_role_for_all_namespaces() 获取所有命名空间的角色
# 否则，调用 api_client.api_temp.list_cluster_role() 获取集群角色
# 返回获取到的所有角色

# 根据 kind 获取所有角色，并找出其中的风险角色
def get_risky_role_by_kind(kind):
    # 初始化风险角色列表
    risky_roles = []
    # 获取所有角色
    all_roles = get_roles_by_kind(kind)
    # 如果获取到了角色
    if all_roles is not None:
        # 找出其中的风险角色
        risky_roles = find_risky_roles(all_roles.items, kind)
    # 返回风险角色列表
    return risky_roles

# 获取风险角色和集群角色
def get_risky_roles_and_clusterroles():
# 获取有风险的角色和集群角色
risky_roles = get_risky_roles()
risky_clusterroles = get_risky_clusterroles()

# 将有风险的角色和集群角色合并成一个列表
all_risky_roles = risky_roles + risky_clusterroles
return all_risky_roles

# 获取有风险的角色
def get_risky_roles():
    return get_risky_role_by_kind('Role')

# 获取有风险的集群角色
def get_risky_clusterroles():
    return get_risky_role_by_kind('ClusterRole')
# 判断给定的角色绑定是否属于风险角色绑定
def is_risky_rolebinding(risky_roles, rolebinding):
    # 初始化风险标志和优先级
    is_risky = False
    priority = Priority.LOW
    # 遍历风险角色列表
    for risky_role in risky_roles:
        # 如果角色绑定的角色名称与风险角色名称相同
        if rolebinding.role_ref.name == risky_role.name:
            # 设置为风险，更新优先级，并跳出循环
            is_risky = True
            priority = risky_role.priority
            break
    # 返回是否为风险和优先级
    return is_risky, priority

# 查找风险角色绑定或集群角色绑定
def find_risky_rolebindings_or_clusterrolebindings(risky_roles, rolebindings, kind):
    risky_rolebindings = []
    # 遍历角色绑定列表
    for rolebinding in rolebindings:
        # 判断角色绑定是否为风险，获取优先级
        is_risky, priority = is_risky_rolebinding(risky_roles, rolebinding)
        # 如果是风险角色绑定，则添加到风险角色绑定列表中
        if is_risky:
            risky_rolebindings.append(RoleBinding(rolebinding.metadata.name, 
# 根据角色绑定的类型和命名空间获取具有高风险的角色绑定
def get_risky_roles_and_clusterroles():
    # 获取所有具有高风险的角色绑定
    risky_rolebindings = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces().items
    # 遍历所有角色绑定，找出具有高风险的角色绑定
    for rolebinding in risky_rolebindings:
        # 检查角色绑定的优先级，命名空间，类型，主体，创建时间等信息，确定是否为高风险角色绑定
        if is_risky(rolebinding.priority, namespace=rolebinding.metadata.namespace, kind=kind, subjects=rolebinding.subjects, time=rolebinding.metadata.creation_timestamp):
            # 将具有高风险的角色绑定添加到结果列表中
            risky_rolebindings.append(rolebinding)
    # 返回具有高风险的角色绑定列表
    return risky_rolebindings

# 根据角色绑定的类型获取所有命名空间中的角色绑定
def get_rolebinding_by_kind_all_namespaces(kind):
    all_roles = []
    # 如果角色绑定的类型是 ROLE_BINDING_KIND
    if kind == ROLE_BINDING_KIND:
        # 获取所有命名空间中的角色绑定
        all_roles = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()
    # 返回所有角色绑定
    return all_roles

# 获取所有具有高风险的角色绑定
def get_all_risky_rolebinding():
    # 获取所有具有高风险的角色和集群角色
    all_risky_roles = get_risky_roles_and_clusterroles()
# 获取所有危险角色绑定
risky_rolebindings = get_risky_rolebindings(all_risky_roles)
# 获取所有危险集群角色绑定
risky_clusterrolebindings = get_risky_clusterrolebindings(all_risky_roles)

# 合并危险角色绑定和危险集群角色绑定
risky_rolebindings_and_clusterrolebindings = risky_clusterrolebindings + risky_rolebindings
# 返回合并后的结果
return risky_rolebindings_and_clusterrolebindings


# 获取危险角色绑定
def get_risky_rolebindings(all_risky_roles=None):
    # 如果未提供危险角色，则获取所有危险角色和集群角色
    if all_risky_roles is None:
        all_risky_roles = get_risky_roles_and_clusterroles()
    # 获取所有角色绑定
    all_rolebindings = get_rolebinding_by_kind_all_namespaces(ROLE_BINDING_KIND)
    # 查找危险角色绑定
    risky_rolebindings = find_risky_rolebindings_or_clusterrolebindings(all_risky_roles, all_rolebindings.items,
                                                                        "RoleBinding")

    return risky_rolebindings


# 获取危险集群角色绑定
def get_risky_clusterrolebindings(all_risky_roles=None):
    # 如果未提供危险角色，则获取所有危险角色和集群角色
    if all_risky_roles is None:
# 获取所有风险角色和集群角色
all_risky_roles = get_risky_roles_and_clusterroles()

# 由于集群不起作用，暂时注释掉
# https://github.com/kubernetes-client/python/issues/577 - 当问题解决后，可以删除注释
# all_clusterrolebindings = api_client.RbacAuthorizationV1Api.list_cluster_role_binding()
all_clusterrolebindings = api_client.api_temp.list_cluster_role_binding()

# 查找风险的集群角色绑定
risky_clusterrolebindings = find_risky_rolebindings_or_clusterrolebindings(all_risky_roles, all_clusterrolebindings, "ClusterRoleBinding")
return risky_clusterrolebindings

# endregion - RoleBindings and ClusterRoleBindings

# region- Risky Users

# 获取所有风险主体
def get_all_risky_subjects():
    all_risky_users = []
    all_risky_rolebindings = get_all_risky_rolebinding()
    passed_users = {}
    for risky_rolebinding in all_risky_rolebindings:
# 如果'risky_rolebinding.subjects'为'None'，则'or []'将防止出现异常。
for user in risky_rolebinding.subjects or []:
    # 删除重复的用户
    if ''.join((user.kind, user.name, str(user.namespace))) not in passed_users:
        passed_users[''.join((user.kind, user.name, str(user.namespace)))] = True
        if user.namespace == None and (user.kind).lower() == "serviceaccount":
            user.namespace = risky_rolebinding.namespace
        all_risky_users.append(Subject(user, risky_rolebinding.priority))

return all_risky_users

# endregion - Risky Users

# region- Risky Pods

'''
# JWT token解码示例
{
# 定义一个函数，用于在容器内执行命令并读取结果
def pod_exec_read_token(pod, container_name, path):
    # 构建读取文件内容的命令
    cat_command = 'cat ' + path
    # 构建执行命令的参数列表
    exec_command = ['/bin/sh',
                    '-c',
                    cat_command]
    # 初始化响应结果
    resp = ''
    # 尝试在容器内执行命令并获取结果
    try:
        # 调用 Kubernetes API 执行命令，并获取结果
        resp = stream(api_client.CoreV1Api.connect_post_namespaced_pod_exec, pod.metadata.name, pod.metadata.namespace,
                      command=exec_command, container=container_name,
                      stderr=False, stdin=False,
# 调用 Kubernetes API 以执行在容器内部的命令，并返回执行结果
def pod_exec_read_token(pod, container_name, exec_command):
    # 创建 API 客户端
    api_client = client.CoreV1Api()
    try:
        # 调用 API 客户端的方法执行在容器内部的命令
        resp = api_client.connect_post_namespaced_pod_exec(pod.metadata.name, pod.metadata.namespace, command=exec_command, container=container_name, stderr=True, stdin=False, stdout=True, tty=False)
    except ApiException as e:
        # 捕获异常并打印错误信息
        print("Exception when calling api_client.CoreV1Api->connect_post_namespaced_pod_exec: %s\n" % e)
        print('{0}, {1}'.format(pod.metadata.name, pod.metadata.namespace))

    # 返回执行结果
    return resp


# 从两个路径中读取容器内的 JWT token
def pod_exec_read_token_two_paths(pod, container_name):
    # 从第一个路径中读取 token
    result = pod_exec_read_token(pod, container_name, '/run/secrets/kubernetes.io/serviceaccount/token')
    # 如果第一个路径中没有找到 token，则从第二个路径中读取
    if result == '':
        result = pod_exec_read_token(pod, container_name, '/var/run/secrets/kubernetes.io/serviceaccount/token')

    # 返回读取到的 token
    return result


# 从容器内获取 JWT token
def get_jwt_token_from_container(pod, container_name):
    # 调用函数从两个路径中读取 token
    resp = pod_exec_read_token_two_paths(pod, container_name)

    # 初始化 token_body 变量
    token_body = ''
# 如果响应不为空且不以'OCI'开头，则解码JWT令牌数据并将其转换为JSON格式
if resp != '' and not resp.startswith('OCI'):
    from engine.jwt_token import decode_jwt_token_data
    decoded_data = decode_jwt_token_data(resp)
    token_body = json.loads(decoded_data)

# 返回解码后的JWT令牌数据和原始响应
return token_body, resp


# 检查两个用户是否相同
def is_same_user(a_username, a_namespace, b_username, b_namespace):
    return (a_username == b_username and a_namespace == b_namespace)


# 从容器中获取风险用户
def get_risky_user_from_container(jwt_body, risky_users):
    risky_user_in_container = None
    for risky_user in risky_users:
        # 如果风险用户的用户信息类型为'ServiceAccount'，并且与JWT令牌中的命名空间和用户名匹配，则将其设置为容器中的风险用户
        if risky_user.user_info.kind == 'ServiceAccount':
            if is_same_user(jwt_body['kubernetes.io/serviceaccount/service-account.name'],
                            jwt_body['kubernetes.io/serviceaccount/namespace'],
                            risky_user.user_info.name, risky_user.user_info.namespace):
                risky_user_in_container = risky_user
# 从容器中获取风险用户信息
def get_risky_containers(pod, risky_users, read_token_from_container=False):
    # 存储风险容器的列表
    risky_containers = []
    # 如果需要从容器中读取令牌
    if read_token_from_container:
        # 跳过已终止和驱逐的 pod
        # 这将仅在状态为“就绪”的容器上运行
        if pod.status.container_statuses:
            # 遍历容器状态
            for container in pod.status.container_statuses:
                # 如果容器就绪
                if container.ready:
                    # 从容器中获取 JWT 令牌的主体和签名
                    jwt_body, _ = get_jwt_token_from_container(pod, container.name)
                    # 如果存在 JWT 主体
                    if jwt_body:
                        # 从容器中获取风险用户信息
                        risky_user = get_risky_user_from_container(jwt_body, risky_users)
                        # 如果存在风险用户
                        if risky_user:
                            # 将容器信息添加到风险容器列表中
                            risky_containers.append(
                                Container(container.name, risky_user.user_info.name, risky_user.user_info.namespace,
                                          risky_user.priority))
    else:
        # 为卷创建一个字典
        volumes_dict = {}
        # 遍历 pod.spec.volumes 列表，将卷的名称作为键，卷对象作为值，存入字典
        for volume in pod.spec.volumes or []:
            volumes_dict[volume.name] = volume
        # 遍历 pod.spec.containers 列表
        for container in pod.spec.containers:
            # 从容器中获取风险用户集合
            risky_users_set = get_risky_users_from_container(container, risky_users, pod, volumes_dict)
            # 如果容器不在风险容器列表中
            if not container_exists_in_risky_containers(risky_containers, container.name,
                                                        risky_users_set):
                # 如果风险用户集合不为空
                if len(risky_users_set) > 0:
                    # 获取最高优先级的风险用户
                    priority = get_highest_priority(risky_users_set)
                    # 将容器信息添加到风险容器列表中
                    risky_containers.append(
                        Container(container.name, None, pod.metadata.namespace, risky_users_set,
                                  priority))
    # 返回风险容器列表
    return risky_containers


# 获取列表中优先级最高的用户
def get_highest_priority(risky_users_list):
# 初始化最高优先级为 NONE
highest_priority = Priority.NONE
# 遍历风险用户列表，找到最高优先级的用户
for user in risky_users_list:
    # 如果当前用户的优先级高于最高优先级，则更新最高优先级
    if user.priority.value > highest_priority.value:
        highest_priority = user.priority
# 返回最高优先级
return highest_priority

# 从容器中获取风险用户
def get_risky_users_from_container(container, risky_users, pod, volumes_dict):
    risky_users_set = set()
    # 如果容器的卷挂载不为空，则遍历卷挂载
    for volume_mount in container.volume_mounts or []:
        # 如果卷挂载的名称在卷字典中
        if volume_mount.name in volumes_dict:
            # 如果卷字典中的投影不为空，则遍历投影源
            if volumes_dict[volume_mount.name].projected is not None:
                for source in volumes_dict[volume_mount.name].projected.sources or []:
                    # 如果源的服务账户令牌不为空
                    if source.service_account_token is not None:
                        # 判断当前服务账户是否为风险用户，是则加入风险用户集合
                        risky_user = is_user_risky(risky_users, pod.spec.service_account, pod.metadata.namespace)
                        if risky_user is not None:
                            risky_users_set.add(risky_user)
            # 如果卷字典中的秘钥不为空
            elif volumes_dict[volume_mount.name].secret is not None:
                # 获取 JWT 并解码，判断是否为风险用户
                risky_user = get_jwt_and_decode(pod, risky_users, volumes_dict[volume_mount.name])
# 如果存在风险用户，则将其添加到风险用户集合中
if risky_user is not None:
    risky_users_set.add(risky_user)
# 返回风险用户集合
return risky_users_set

# 检查风险容器列表中是否存在指定容器，并将风险用户列表中的用户添加到该容器的服务账户名中
def container_exists_in_risky_containers(risky_containers, container_name, risky_users_list):
    for risky_container in risky_containers:
        if risky_container.name == container_name:
            for user_name in risky_users_list:
                risky_container.service_account_name.append(user_name)
            return True
    return False

# 检查卷挂载列表中是否存在默认路径 "/var/run/secrets/kubernetes.io/serviceaccount"
def default_path_exists(volume_mounts):
    for volume_mount in volume_mounts:
        if volume_mount.mount_path == "/var/run/secrets/kubernetes.io/serviceaccount":
            return True
    return False
# 检查用户是否属于风险用户列表，如果是则返回该用户信息，否则返回 None
def is_user_risky(risky_users, service_account, namespace):
    # 遍历风险用户列表
    for risky_user in risky_users:
        # 检查用户信息是否匹配给定的服务账号和命名空间
        if risky_user.user_info.name == service_account and risky_user.user_info.namespace == namespace:
            # 如果匹配，则返回该风险用户信息
            return risky_user
    # 如果没有匹配的用户信息，则返回 None
    return None


# 获取 JWT 并解码
def get_jwt_and_decode(pod, risky_users, volume):
    # 导入 JWT 解码函数
    from engine.jwt_token import decode_base64_jwt_token
    try:
        # 读取指定命名空间下的 Secret
        secret = api_client.CoreV1Api.read_namespaced_secret(name=volume.secret.secret_name,
                                                             namespace=pod.metadata.namespace)
    except Exception:
        # 如果出现异常，则将 secret 设为 None
        secret = None
    try:
        # 如果 secret 不为 None 且包含数据
        if secret is not None and secret.data is not None:
            # 如果数据中包含 'token'
            if 'token' in secret.data:
                # 解码 token 数据
                decoded_data = decode_base64_jwt_token(secret.data['token'])
                # 将解码后的数据转换为 JSON 格式
                token_body = json.loads(decoded_data)
# 如果 token_body 不为空，则从容器中获取风险用户并返回
if token_body:
    risky_user = get_risky_user_from_container(token_body, risky_users)
    return risky_user
# 如果 token_body 为空，则抛出异常
raise Exception()
# 捕获异常
except Exception:
    # 如果 secret 不为空，则调用 get_risky_user_from_container_secret 函数
    if secret is not None:
        return get_risky_user_from_container_secret(secret, risky_users)

# 从 secret 中获取风险用户
def get_risky_user_from_container_secret(secret, risky_users):
    # 如果 secret 不为空
    if secret is not None:
        # 获取全局变量 list_of_service_accounts
        global list_of_service_accounts
        # 如果 list_of_service_accounts 为空，则获取所有命名空间的服务账户列表
        if not list_of_service_accounts:
            list_of_service_accounts = api_client.CoreV1Api.list_service_account_for_all_namespaces()
        # 遍历服务账户列表
        for sa in list_of_service_accounts.items:
            # 遍历服务账户的密钥列表
            for service_account_secret in sa.secrets or []:
                # 如果 secret 的名称与服务账户的密钥名称相同
                if secret.metadata.name == service_account_secret.name:
                    # 遍历风险用户列表
                    for risky_user in risky_users:
                        # 如果风险用户的名称与服务账户的名称相同，则返回该风险用户
                        if risky_user.user_info.name == sa.metadata.name:
                            return risky_user
# 获取风险的 Pod 列表
def get_risky_pods(namespace=None, deep_analysis=False):
    # 初始化风险的 Pod 列表
    risky_pods = []
    # 获取指定命名空间或所有命名空间的 Pod 列表
    pods = list_pods_for_all_namespaces_or_one_namspace(namespace)
    # 获取所有风险主体
    risky_users = get_all_risky_subjects()
    # 遍历每个 Pod
    for pod in pods.items:
        # 获取该 Pod 中的风险容器
        risky_containers = get_risky_containers(pod, risky_users, deep_analysis)
        # 如果存在风险容器，则将该 Pod 加入风险 Pod 列表
        if len(risky_containers) > 0:
            risky_pods.append(Pod(pod.metadata.name, pod.metadata.namespace, risky_containers))
    # 返回风险 Pod 列表
    return risky_pods


# 获取所有命名空间的 RoleBindings 和 ClusterRoleBindings
def get_rolebindings_all_namespaces_and_clusterrolebindings():
    # 获取所有命名空间的 RoleBindings
    namespaced_rolebindings = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()

    # TODO: check when this bug will be fixed
    # 获取集群级的 ClusterRoleBindings
    cluster_rolebindings = api_client.api_temp.list_cluster_role_binding()
# 返回命名空间角色绑定和集群角色绑定
def get_rolebindings_and_clusterrolebindings_associated_to_subject(subject_name, kind, namespace):
    # 获取所有命名空间的角色绑定和集群角色绑定
    rolebindings_all_namespaces, cluster_rolebindings = get_rolebindings_all_namespaces_and_clusterrolebindings()
    # 与主体相关的角色绑定
    associated_rolebindings = []

    # 遍历所有命名空间的角色绑定
    for rolebinding in rolebindings_all_namespaces.items:
        # 如果 'rolebinding.subjects' 为 'None'，'or []' 将防止出现异常
        for subject in rolebinding.subjects or []:
            # 如果主体名称和类型与给定的主体名称和类型匹配
            if subject.name.lower() == subject_name.lower() and subject.kind.lower() == kind.lower():
                # 如果类型为服务账户
                if kind == SERVICEACCOUNT_KIND:
                    # 如果主体的命名空间与给定的命名空间匹配
                    if subject.namespace.lower() == namespace.lower():
                        # 将角色绑定添加到相关角色绑定列表中
                        associated_rolebindings.append(rolebinding)
                else:
                    # 将角色绑定添加到相关角色绑定列表中
                    associated_rolebindings.append(rolebinding)

    # 相关的集群角色绑定
    associated_clusterrolebindings = []
    for clusterrolebinding in cluster_rolebindings:
        # 在这里继续处理集群角色绑定
# 如果'clusterrolebinding.subjects'为'None'，'or []'将防止出现异常。
for subject in clusterrolebinding.subjects or []:
    # 如果主体名称和类型与给定的名称和类型匹配，则将集群角色绑定添加到关联的集群角色绑定列表中
    if subject.name == subject_name.lower() and subject.kind.lower() == kind.lower():
        # 如果类型为SERVICEACCOUNT_KIND，则检查命名空间是否匹配，然后将集群角色绑定添加到关联的集群角色绑定列表中
        if kind == SERVICEACCOUNT_KIND:
            if subject.namespace.lower() == namespace.lower():
                associated_clusterrolebindings.append(clusterrolebinding)
        else:
            associated_clusterrolebindings.append(clusterrolebinding)

# 返回关联的角色绑定和关联的集群角色绑定
return associated_rolebindings, associated_clusterrolebindings


# 角色只能存在于角色绑定中
def get_rolebindings_associated_to_role(role_name, namespace):
    # 获取所有命名空间中的角色绑定
    rolebindings_all_namespaces = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()
    associated_rolebindings = []

    # 遍历所有角色绑定
    for rolebinding in rolebindings_all_namespaces.items:
        # 如果角色绑定的角色名称、类型和命名空间与给定的名称、类型和命名空间匹配，则将角色绑定添加到关联的角色绑定列表中
        if rolebinding.role_ref.name.lower() == role_name.lower() and rolebinding.role_ref.kind == ROLE_KIND and rolebinding.metadata.namespace.lower() == namespace.lower():
            associated_rolebindings.append(rolebinding)
# 返回与集群角色相关联的角色绑定
def get_rolebindings_and_clusterrolebindings_associated_to_clusterrole(role_name):
    # 获取所有命名空间的角色绑定和集群角色绑定
    rolebindings_all_namespaces, cluster_rolebindings = get_rolebindings_all_namespaces_and_clusterrolebindings()

    # 与角色相关联的角色绑定列表
    associated_rolebindings = []

    # 遍历所有命名空间的角色绑定
    for rolebinding in rolebindings_all_namespaces.items:
        # 如果角色绑定的角色名称与给定的角色名称相同，并且角色绑定的类型是集群角色，则将其添加到相关联的角色绑定列表中
        if rolebinding.role_ref.name.lower() == role_name.lower() and rolebinding.role_ref.kind == CLUSTER_ROLE_KIND:
            associated_rolebindings.append(rolebinding)

    # 与角色相关联的集群角色绑定列表
    associated_clusterrolebindings = []

    # 遍历集群角色绑定
    for clusterrolebinding in cluster_rolebindings:
        # 如果集群角色绑定的角色名称与给定的角色名称相同，并且集群角色绑定的类型是集群角色，则将其添加到相关联的角色绑定列表中
        if clusterrolebinding.role_ref.name.lower() == role_name.lower() and clusterrolebinding.role_ref.kind == CLUSTER_ROLE_KIND:
            associated_rolebindings.append(clusterrolebinding)
# 返回关联的角色绑定和关联的集群角色绑定
def dump_containers_tokens_by_pod(pod_name, namespace, read_token_from_container=False):
    # 初始化容器列表
    containers_with_tokens = []
    try:
        # 读取指定命名空间中的 Pod 对象
        pod = api_client.CoreV1Api.read_namespaced_pod(name=pod_name, namespace=namespace)
    except ApiException:
        # 如果出现异常，打印错误信息并返回空值
        print(pod_name + " was not found in " + namespace + " namespace")
        return None
    # 如果需要从容器中读取令牌
    if read_token_from_container:
        # 遍历 Pod 中的容器状态
        if pod.status.container_statuses:
            for container in pod.status.container_statuses:
                # 如果容器就绪
                if container.ready:
                    # 从容器中获取 JWT 令牌
                    jwt_body, raw_jwt_token = get_jwt_token_from_container(pod, container.name)
                    # 如果获取到 JWT 令牌，则将容器信息添加到容器列表中
                    if jwt_body:
                        containers_with_tokens.append(
                            Container(container.name, token=jwt_body, raw_jwt_token=raw_jwt_token))
    # 如果不需要从容器中读取令牌
    else:
# 将容器和令牌列表填充到容器列表中
def fill_container_with_tokens_list(containers_with_tokens, pod):
    # 导入解码 base64 JWT 令牌的函数
    from engine.jwt_token import decode_base64_jwt_token
    # 遍历 Pod 中的容器
    for container in pod.spec.containers:
        # 遍历容器的挂载卷
        for volume_mount in container.volume_mounts or []:
            # 遍历 Pod 的卷
            for volume in pod.spec.volumes or []:
                # 如果卷的名称与挂载卷的名称相同，并且卷是一个密钥
                if volume.name == volume_mount.name and volume.secret:
                    # 尝试读取密钥
                    try:
                        secret = api_client.CoreV1Api.read_namespaced_secret(volume.secret.secret_name,
                                                                             pod.metadata.namespace)
                        # 如果密钥存在且包含数据和令牌
                        if secret and secret.data and secret.data['token']:
                            # 解码令牌数据
                            decoded_data = decode_base64_jwt_token(secret.data['token'])
                            token_body = json.loads(decoded_data)
                            # 将容器和令牌添加到容器列表中
                            containers_with_tokens.append(Container(container.name, token=token_body,
                                                                    raw_jwt_token=None))
                    # 捕获异常
                    except ApiException:
                        print("No secret found.")
# 如果未指定命名空间，则获取所有命名空间的所有 Pod 列表，否则获取指定命名空间的 Pod 列表
def dump_all_pods_tokens_or_by_namespace(namespace=None, read_token_from_container=False):
    # 存储具有令牌的 Pod 的列表
    pods_with_tokens = []
    # 获取指定命名空间的所有 Pod 列表
    pods = list_pods_for_all_namespaces_or_one_namspace(namespace)
    # 遍历每个 Pod
    for pod in pods.items:
        # 获取 Pod 中容器的令牌信息
        containers = dump_containers_tokens_by_pod(pod.metadata.name, pod.metadata.namespace, read_token_from_container)
        # 如果容器信息不为空，则将 Pod 加入到具有令牌的 Pod 列表中
        if containers is not None:
            pods_with_tokens.append(Pod(pod.metadata.name, pod.metadata.namespace, containers))
    # 返回具有令牌的 Pod 列表
    return pods_with_tokens

# 获取指定 Pod 的令牌信息
def dump_pod_tokens(name, namespace, read_token_from_container=False):
    # 存储具有令牌的 Pod 的列表
    pod_with_tokens = []
    # 获取指定 Pod 中容器的令牌信息
    containers = dump_containers_tokens_by_pod(name, namespace, read_token_from_container)
    # 将 Pod 和其容器的令牌信息加入到列表中
    pod_with_tokens.append(Pod(name, namespace, containers))
    # 返回具有令牌的 Pod 列表
    return pod_with_tokens
# 根据给定的 kind 在 subjects 中搜索匹配的主题，并返回匹配的主题列表
def search_subject_in_subjects_by_kind(subjects, kind):
    # 初始化一个空列表用于存放匹配的主题
    subjects_found = []
    # 遍历所有主题
    for subject in subjects:
        # 如果主题的类型（kind）与给定的类型相同（不区分大小写）
        if subject.kind.lower() == kind.lower():
            # 将匹配的主题添加到结果列表中
            subjects_found.append(subject)
    # 返回匹配的主题列表
    return subjects_found

# 获取所有角色绑定的指定类型的主题
def get_subjects_by_kind(kind):
    # 初始化一个空列表用于存放匹配的主题
    subjects_found = []
    # 获取所有命名空间的角色绑定
    rolebindings = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()
    # 获取集群角色绑定
    clusterrolebindings = api_client.api_temp.list_cluster_role_binding()
    # 遍历所有角色绑定
    for rolebinding in rolebindings.items:
        # 如果角色绑定中包含主题
        if rolebinding.subjects is not None:
            # 调用 search_subject_in_subjects_by_kind 函数，将匹配的主题添加到结果列表中
            subjects_found += search_subject_in_subjects_by_kind(rolebinding.subjects, kind)

    # 遍历所有集群角色绑定
    for clusterrolebinding in clusterrolebindings:
        # 如果集群角色绑定中包含主题
        if clusterrolebinding.subjects is not None:
            # 这里应该有一些代码，用于将匹配的主题添加到结果列表中，但是缺失了
# 将搜索到的主体按照类型添加到已找到的主体列表中
def search_subject_in_subjects_by_kind(subjects, kind):
    subjects_found = []
    for s in subjects:
        if s.kind == kind:
            subjects_found.append(s)
    return subjects_found

# 去除重复的主体
def remove_duplicated_subjects(subjects):
    seen_subjects = set()  # 创建一个空集合用于存储已经出现过的主体
    new_subjects = []  # 创建一个空列表用于存储去重后的主体
    for s1 in subjects:  # 遍历传入的主体列表
        if s1.namespace == None:  # 如果主体的命名空间为空
            s1_unique_name = ''.join([s1.name, s1.kind])  # 将主体的名称和类型拼接成唯一的名称
        else:
            s1_unique_name = ''.join([s1.name, s1.namespace, s1.kind])  # 将主体的名称、命名空间和类型拼接成唯一的名称
        if s1_unique_name not in seen_subjects:  # 如果该唯一名称不在已经出现过的主体集合中
            new_subjects.append(s1)  # 将该主体添加到去重后的主体列表中
            seen_subjects.add(s1_unique_name)  # 将该唯一名称添加到已经出现过的主体集合中
    return new_subjects  # 返回去重后的主体列表
# 根据角色绑定名称和命名空间获取角色绑定的角色
def get_rolebinding_role(rolebinding_name, namespace):
    rolebinding = None  # 初始化角色绑定对象
    role = None  # 初始化角色对象
    try:
        # 通过 API 客户端读取指定命名空间中的角色绑定
        rolebinding = api_client.RbacAuthorizationV1Api.read_namespaced_role_binding(rolebinding_name, namespace)
        # 如果角色绑定的角色引用类型为 ROLE_KIND
        if rolebinding.role_ref.kind == ROLE_KIND:
            # 通过 API 客户端读取指定命名空间中的角色
            role = api_client.RbacAuthorizationV1Api.read_namespaced_role(rolebinding.role_ref.name,
                                                                          rolebinding.metadata.namespace)
        else:
            # 通过 API 客户端读取集群中的角色
            role = api_client.RbacAuthorizationV1Api.read_cluster_role(rolebinding.role_ref.name)

        return role  # 返回获取到的角色
    except ApiException:
        if rolebinding is None:
            print("Could not find " + rolebinding_name + " rolebinding in " + namespace + " namespace")
        elif role is None:
            print(
                "Could not find " + rolebinding.role_ref.name + " role in " + rolebinding.role_ref.name + " rolebinding")
        return None  # 返回空值
# 根据集群角色绑定名称获取集群角色
def get_clusterrolebinding_role(cluster_rolebinding_name):
    cluster_role = ''
    try:
        # 通过 API 客户端读取集群角色绑定
        cluster_rolebinding = api_client.RbacAuthorizationV1Api.read_cluster_role_binding(cluster_rolebinding_name)
        # 通过 API 客户端读取集群角色
        cluster_role = api_client.RbacAuthorizationV1Api.read_cluster_role(cluster_rolebinding.role_ref.name)
    except ApiException as e:
        # 捕获异常并打印错误信息
        print(e)
        # 退出程序
        exit()
    # 返回获取的集群角色
    return cluster_role


# 获取与主体关联的角色
def get_roles_associated_to_subject(subject_name, kind, namespace):
    # 获取与主体关联的角色绑定和集群角色绑定
    associated_rolebindings, associated_clusterrolebindings = get_rolebindings_and_clusterrolebindings_associated_to_subject(
        subject_name, kind, namespace)

    # 初始化关联的角色列表
    associated_roles = []
    # 遍历关联的角色绑定
    for rolebind in associated_rolebindings:
        try:
# 获取角色绑定的角色，并添加到关联角色列表中
role = get_rolebinding_role(rolebind.metadata.name, rolebind.metadata.namespace)
associated_roles.append(role)
# 如果发生异常，即资源未找到，继续循环
except ApiException as e:
    # 404 not found
    continue

# 遍历关联的集群角色绑定，获取角色并添加到关联角色列表中
for clusterrolebinding in associated_clusterrolebindings:
    role = get_clusterrolebinding_role(clusterrolebinding.metadata.name)
    associated_roles.append(role)

# 返回关联的角色列表
return associated_roles

# 根据命名空间列出所有或指定命名空间的 Pod
def list_pods_for_all_namespaces_or_one_namspace(namespace=None):
    try:
        # 如果未指定命名空间，则列出所有命名空间的 Pod
        if namespace is None:
            pods = api_client.CoreV1Api.list_pod_for_all_namespaces(watch=False)
        # 否则，列出指定命名空间的 Pod
        else:
            pods = api_client.CoreV1Api.list_namespaced_pod(namespace)
        # 返回获取到的 Pod 列表
        return pods
    # 捕获 ApiException 异常并返回 None
    except ApiException:
        return None


# 获取解码后的引导令牌列表
def list_boostrap_tokens_decoded():
    # 初始化令牌列表
    tokens = []
    # 获取 kube-system 命名空间下的 secrets 列表，筛选出类型为 bootstrap.kubernetes.io/token 的 secret
    secrets = api_client.CoreV1Api.list_namespaced_secret(namespace='kube-system',
                                                          field_selector='type=bootstrap.kubernetes.io/token')
    # 导入 base64 模块

    # 遍历 secrets 列表，解码 token-id 和 token-secret 字段的值，并将其拼接后添加到 tokens 列表中
    for secret in secrets.items:
        tokens.append('.'.join((base64.b64decode(secret.data['token-id']).decode('utf-8'),
                                base64.b64decode(secret.data['token-secret']).decode('utf-8'))))

    # 返回解码后的令牌列表
    return tokens
```