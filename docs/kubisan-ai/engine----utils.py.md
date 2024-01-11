# `KubiScan\engine\utils.py`

```
# 导入 requests 模块
import requests

# 从 engine.role 模块导入 Role 类
from engine.role import Role
# 从 engine.priority 模块导入 Priority 类
from engine.priority import Priority
# 导入 STATIC_RISKY_ROLES
from static_risky_roles import STATIC_RISKY_ROLES
# 从 engine.role_binding 模块导入 RoleBinding 类
from engine.role_binding import RoleBinding
# 从 kubernetes.stream 模块导入 stream 函数
from kubernetes.stream import stream
# 从 engine.pod 模块导入 Pod 类
from engine.pod import Pod
# 从 engine.container 模块导入 Container 类
from engine.container import Container
# 导入 json 模块
import json
# 从 api 模块导入 api_client
from api import api_client
# 从 engine.subject 模块导入 Subject 类
from engine.subject import Subject
# 从 misc.constants 模块导入所有内容
from misc.constants import *
# 从 kubernetes.client.rest 模块导入 ApiException 类
from kubernetes.client.rest import ApiException
# 导入 urllib3 模块
import urllib3

# 禁用 urllib3 的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# endregion - Roles and ClusteRoles

# 初始化空列表 list_of_service_accounts
list_of_service_accounts = []

# 判断是否存在风险资源名称
def is_risky_resource_name_exist(source_rolename, source_resourcenames):
    # 初始化 is_risky 为 False
    is_risky = False
    # 遍历 source_resourcenames 中的资源名称
    for resource_name in source_resourcenames:
        # 防止循环
        if resource_name != source_rolename:
            # TODO: 需要允许此检查也适用于 'roles' 资源名称，应考虑命名空间...
            # 根据资源名称和 CLUSTER_ROLE_KIND 获取角色
            role = get_role_by_name_and_kind(resource_name, CLUSTER_ROLE_KIND)
            # 如果角色不为空
            if role is not None:
                # 判断角色是否为风险角色
                is_risky, priority = is_risky_role(role)
                # 如果是风险角色，则跳出循环
                if is_risky:
                    break
    # 返回是否存在风险资源名称
    return is_risky

# 判断规则是否包含风险规则
def is_rule_contains_risky_rule(source_role_name, source_rule, risky_rule):
    # 初始化 is_contains 为 True
    is_contains = True
    # 初始化 is_bind_verb_found 为 False
    is_bind_verb_found = False
    # 初始化 is_role_resource_found 为 False

    # 可选: 取消注释并将下面的所有内容移至 'return' 以添加具有 "*" 的动词或资源的任何规则。
    # 目前它在 risky_roles.yaml 中部分处理
    # if (source_rule.verbs is not None and "*" not in source_rule.verbs) and (source_rule.resources is not None and "*" not in source_rule.resources):
    # 遍历风险规则中的动词
    for verb in risky_rule.verbs:
        # 如果动词不在源规则的动词中
        if verb not in source_rule.verbs:
            # 设置 is_contains 为 False，并跳出循环
            is_contains = False
            break
        # 如果动词为 "bind"
        if verb.lower() == "bind":
            # 设置 is_bind_verb_found 为 True
            is_bind_verb_found = True
    # 检查是否包含资源，并且源规则的资源不为空
    if is_contains and source_rule.resources is not None:
        # 遍历风险规则的资源
        for resource in risky_rule.resources:
            # 如果资源不在源规则的资源中，则将 is_contains 设为 False，并跳出循环
            if resource not in source_rule.resources:
                is_contains = False
                break
            # 如果资源为 "roles" 或 "clusterroles"，则将 is_role_resource_found 设为 True
            if resource.lower() == "roles" or resource.lower() == "clusterroles":
                is_role_resource_found = True

        # 如果 is_contains 为 True 并且风险规则的资源名称不为空
        if is_contains and risky_rule.resource_names is not None:
            # 将 is_contains 设为 False
            is_contains = False
            # 如果 is_bind_verb_found 和 is_role_resource_found 都为 True
            if is_bind_verb_found and is_role_resource_found:
                # 检查是否存在风险资源名称，如果存在则将 is_contains 设为 True
                is_risky = is_risky_resource_name_exist(source_role_name, source_rule.resource_names)
                if is_risky:
                    is_contains = True
    else:
        # 如果不包含资源或者源规则的资源为空，则将 is_contains 设为 False
        is_contains = False

    # 返回最终的 is_contains 值
    return is_contains
# 获取当前版本信息
def get_current_version(certificate_authority_file=None, client_certificate_file=None, client_key_file=None, host=None):
    # 如果未提供主机地址，则从 API 客户端获取版本信息
    if host is None:
        version = api_client.api_version.get_code().git_version
        return version.replace('v', "")
    else:
        # 如果未提供证书文件，则使用不安全的方式发送请求
        if certificate_authority_file is None and client_certificate_file is None and client_key_file is None:
            response = requests.get(host + '/version', verify=False)
            # 如果请求失败，则打印错误信息并返回 None
            if response.status_code != 200:
                print(response.text)
                return None
            else:
                # 如果请求成功，则返回响应中的 git 版本信息
                return response.json()["gitVersion"].replace('v', "")
        # 如果提供了证书文件，则使用安全的方式发送请求
        if certificate_authority_file is not None and client_certificate_file is not None and client_key_file is not None:
            response = requests.get(host + '/version', cert=(client_certificate_file, client_key_file),
                                    verify=certificate_authority_file)
            # 如果请求失败，则打印错误信息并返回 None
            if response.status_code != 200:
                print(response.text)
                return None
            else:
                # 如果请求成功，则返回响应中的 git 版本信息
                return response.json()["gitVersion"].replace('v', "")
        # 如果未提供完整的证书信息或主机地址，则打印错误信息并返回 None
        if certificate_authority_file is None or client_certificate_file is None or client_key_file is None or host is None:
            print("Please provide certificate authority file path, client certificate file path,"
                  " client key file path and host address")
            return None
        # 使用提供的证书信息发送请求
        response = requests.get(host + '/version', cert=(client_certificate_file, client_key_file),
                                verify=certificate_authority_file)
        # 如果请求失败，则打印错误信息并返回 None
        if response.status_code != 200:
            print(response.text)
            return None
        else:
            # 如果请求成功，则返回响应中的 git 版本信息
            return response.json()["gitVersion"].replace('v', "")

# 根据名称和类型获取角色信息
def get_role_by_name_and_kind(name, kind, namespace=None):
    requested_role = None
    # 根据类型获取所有角色信息
    roles = get_roles_by_kind(kind)
    # 遍历 roles 字典中的每个元素
    for role in roles.items:
        # 如果当前角色的名称与指定名称相同
        if role.metadata.name == name:
            # 将当前角色赋值给 requested_role
            requested_role = role
            # 结束循环
            break
    # 返回 requested_role
    return requested_role
# 检查源角色的规则是否包含在目标规则中
def are_rules_contain_other_rules(source_role_name, source_rules, target_rules):
    # 初始化变量，用于标记是否包含和匹配的规则数量
    is_contains = False
    matched_rules = 0
    # 如果目标规则或源规则为空，则直接返回 False
    if not (target_rules and source_rules):
        return is_contains
    # 遍历目标规则
    for target_rule in target_rules:
        # 如果源规则不为空，则遍历源规则
        if source_rules is not None:
            for source_rule in source_rules:
                # 调用函数判断源规则是否包含在目标规则中
                if is_rule_contains_risky_rule(source_role_name, source_rule, target_rule):
                    # 如果匹配到规则，则匹配的规则数量加一
                    matched_rules += 1
                    # 如果匹配的规则数量等于目标规则的数量，则标记为包含并返回
                    if matched_rules == len(target_rules):
                        is_contains = True
                        return is_contains
    # 返回是否包含的标记
    return is_contains


# 判断角色是否属于风险角色
def is_risky_role(role):
    # 初始化变量，用于标记是否为风险角色和优先级
    is_risky = False
    priority = Priority.LOW
    # 遍历静态风险角色列表
    for risky_role in STATIC_RISKY_ROLES:
        # 调用函数判断角色的规则是否包含在风险角色的规则中
        if are_rules_contain_other_rules(role.metadata.name, role.rules, risky_role.rules):
            # 如果是风险角色，则标记为风险角色，并更新优先级
            is_risky = True
            priority = risky_role.priority
            break
    # 返回是否为风险角色和优先级
    return is_risky, priority


# 查找风险角色
def find_risky_roles(roles, kind):
    risky_roles = []
    # 遍历角色列表
    for role in roles:
        # 调用函数判断角色是否为风险角色
        is_risky, priority = is_risky_role(role)
        # 如果是风险角色，则添加到风险角色列表中
        if is_risky:
            risky_roles.append(
                Role(role.metadata.name, priority, rules=role.rules, namespace=role.metadata.namespace, kind=kind,
                     time=role.metadata.creation_timestamp))
    # 返回风险角色列表
    return risky_roles


# 根据类型获取角色列表
def get_roles_by_kind(kind):
    all_roles = []
    # 如果类型为 ROLE_KIND，则获取所有命名空间的角色列表
    if kind == ROLE_KIND:
        all_roles = api_client.RbacAuthorizationV1Api.list_role_for_all_namespaces()
    else:
        # 否则获取集群角色列表
        all_roles = api_client.api_temp.list_cluster_role()
    # 返回角色列表
    return all_roles


# 根据类型获取风险角色列表
def get_risky_role_by_kind(kind):
    risky_roles = []
    # 获取指定类型的角色列表
    all_roles = get_roles_by_kind(kind)
    # 如果角色列表不为空，则查找风险角色
    if all_roles is not None:
        risky_roles = find_risky_roles(all_roles.items, kind)
    # 返回风险角色列表
    return risky_roles


# 获取风险角色和集群角色
def get_risky_roles_and_clusterroles():
    risky_roles = get_risky_roles()
    # 获取风险的集群角色
    risky_clusterroles = get_risky_clusterroles()
    
    # 将风险的角色和风险的集群角色合并成一个列表
    all_risky_roles = risky_roles + risky_clusterroles
    # 返回所有风险的角色列表
    return all_risky_roles
# 获取所有风险角色
def get_risky_roles():
    return get_risky_role_by_kind('Role')


# 获取所有风险集群角色
def get_risky_clusterroles():
    return get_risky_role_by_kind('ClusterRole')


# 判断角色绑定是否属于风险角色
def is_risky_rolebinding(risky_roles, rolebinding):
    is_risky = False
    priority = Priority.LOW
    for risky_role in risky_roles:
        # 如果角色绑定的角色名与风险角色名相同，则标记为风险
        if rolebinding.role_ref.name == risky_role.name:
            is_risky = True
            priority = risky_role.priority
            break
    return is_risky, priority


# 查找所有风险角色绑定或集群角色绑定
def find_risky_rolebindings_or_clusterrolebindings(risky_roles, rolebindings, kind):
    risky_rolebindings = []
    for rolebinding in rolebindings:
        is_risky, priority = is_risky_rolebinding(risky_roles, rolebinding)
        if is_risky:
            # 将风险角色绑定添加到列表中
            risky_rolebindings.append(RoleBinding(rolebinding.metadata.name,
                                                  priority,
                                                  namespace=rolebinding.metadata.namespace,
                                                  kind=kind, subjects=rolebinding.subjects,
                                                  time=rolebinding.metadata.creation_timestamp))
    return risky_rolebindings


# 根据类型获取所有命名空间的角色绑定
def get_rolebinding_by_kind_all_namespaces(kind):
    all_roles = []
    if kind == ROLE_BINDING_KIND:
        # 获取所有命名空间的角色绑定
        all_roles = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()
    # else:
    # TODO: check if it was fixed
    # all_roles = api_client.RbacAuthorizationV1Api.list_cluster_role_binding()

    return all_roles


# 获取所有风险角色绑定
def get_all_risky_rolebinding():
    all_risky_roles = get_risky_roles_and_clusterroles()

    risky_rolebindings = get_risky_rolebindings(all_risky_roles)
    risky_clusterrolebindings = get_risky_clusterrolebindings(all_risky_roles)

    # 将风险角色绑定和集群角色绑定合并成一个列表
    risky_rolebindings_and_clusterrolebindings = risky_clusterrolebindings + risky_rolebindings
    # 返回变量 risky_rolebindings_and_clusterrolebindings 的值
    return risky_rolebindings_and_clusterrolebindings
# 获取所有风险角色绑定
def get_risky_rolebindings(all_risky_roles=None):
    # 如果未提供所有风险角色，则获取所有风险角色和集群角色
    if all_risky_roles is None:
        all_risky_roles = get_risky_roles_and_clusterroles()
    # 获取所有角色绑定
    all_rolebindings = get_rolebinding_by_kind_all_namespaces(ROLE_BINDING_KIND)
    # 查找风险角色绑定
    risky_rolebindings = find_risky_rolebindings_or_clusterrolebindings(all_risky_roles, all_rolebindings.items,
                                                                        "RoleBinding")
    # 返回风险角色绑定
    return risky_rolebindings


# 获取所有风险集群角色绑定
def get_risky_clusterrolebindings(all_risky_roles=None):
    # 如果未提供所有风险角色，则获取所有风险角色和集群角色
    if all_risky_roles is None:
        all_risky_roles = get_risky_roles_and_clusterroles()
    # 获取所有集群角色绑定
    # 集群不起作用。
    # https://github.com/kubernetes-client/python/issues/577 - 当问题解决后，可以删除注释
    # all_clusterrolebindings = api_client.RbacAuthorizationV1Api.list_cluster_role_binding()
    all_clusterrolebindings = api_client.api_temp.list_cluster_role_binding()
    # 查找风险集群角色绑定
    risky_clusterrolebindings = find_risky_rolebindings_or_clusterrolebindings(all_risky_roles, all_clusterrolebindings,
                                                                               "ClusterRoleBinding")
    # 返回风险集群角色绑定
    return risky_clusterrolebindings


# 区域 - 角色绑定和集群角色绑定

# 区域 - 风险用户

# 获取所有风险主体
def get_all_risky_subjects():
    all_risky_users = []
    all_risky_rolebindings = get_all_risky_rolebinding()
    passed_users = {}
    # 遍历所有存在风险的角色绑定
    for risky_rolebinding in all_risky_rolebindings:

        # 如果'risky_rolebinding.subjects'为'None'，则使用'or []'来避免出现异常
        for user in risky_rolebinding.subjects or []:
            # 移除重复的用户
            if ''.join((user.kind, user.name, str(user.namespace))) not in passed_users:
                passed_users[''.join((user.kind, user.name, str(user.namespace)))] = True
                # 如果用户的命名空间为None，并且用户类型为"serviceaccount"，则将用户的命名空间设置为risky_rolebinding的命名空间
                if user.namespace == None and (user.kind).lower() == "serviceaccount":
                    user.namespace = risky_rolebinding.namespace
                # 将用户和对应的风险角色绑定的优先级添加到所有存在风险的用户列表中
                all_risky_users.append(Subject(user, risky_rolebinding.priority))

    # 返回所有存在风险的用户列表
    return all_risky_users
# endregion - Risky Users

# region- Risky Pods

'''
Example of JWT token decoded:
{
    'kubernetes.io/serviceaccount/service-account.uid': '11a8e2a1-6f07-11e8-8d52-000c2904e34b',
     'iss': 'kubernetes/serviceaccount',
     'sub': 'system:serviceaccount:default:myservice',
     'kubernetes.io/serviceaccount/namespace': 'default',
     'kubernetes.io/serviceaccount/secret.name': 'myservice-token-btwvr',
     'kubernetes.io/serviceaccount/service-account.name': 'myservice'
 }
'''

# 从容器中读取指定路径的文件内容
def pod_exec_read_token(pod, container_name, path):
    cat_command = 'cat ' + path
    exec_command = ['/bin/sh',
                    '-c',
                    cat_command]
    resp = ''
    try:
        # 执行命令获取文件内容
        resp = stream(api_client.CoreV1Api.connect_post_namespaced_pod_exec, pod.metadata.name, pod.metadata.namespace,
                      command=exec_command, container=container_name,
                      stderr=False, stdin=False,
                      stdout=True, tty=False)
    except ApiException as e:
        # 捕获异常并打印错误信息
        print("Exception when calling api_client.CoreV1Api->connect_post_namespaced_pod_exec: %s\n" % e)
        print('{0}, {1}'.format(pod.metadata.name, pod.metadata.namespace))

    return resp

# 从容器中读取两个指定路径的文件内容
def pod_exec_read_token_two_paths(pod, container_name):
    result = pod_exec_read_token(pod, container_name, '/run/secrets/kubernetes.io/serviceaccount/token')
    if result == '':
        result = pod_exec_read_token(pod, container_name, '/var/run/secrets/kubernetes.io/serviceaccount/token')

    return result

# 从容器中获取 JWT token，并解码
def get_jwt_token_from_container(pod, container_name):
    resp = pod_exec_read_token_two_paths(pod, container_name)

    token_body = ''
    if resp != '' and not resp.startswith('OCI'):
        from engine.jwt_token import decode_jwt_token_data
        decoded_data = decode_jwt_token_data(resp)
        token_body = json.loads(decoded_data)

    return token_body, resp

# 检查两个用户是否相同
def is_same_user(a_username, a_namespace, b_username, b_namespace):
    # 检查两个用户名和命名空间是否相等，如果相等则返回True，否则返回False
    return (a_username == b_username and a_namespace == b_namespace)
# 从 JWT 身份验证令牌中获取容器中的风险用户
def get_risky_user_from_container(jwt_body, risky_users):
    # 初始化容器中的风险用户为 None
    risky_user_in_container = None
    # 遍历所有风险用户
    for risky_user in risky_users:
        # 如果风险用户的用户信息类型为 'ServiceAccount'
        if risky_user.user_info.kind == 'ServiceAccount':
            # 如果当前用户与风险用户匹配，则将风险用户设置为容器中的风险用户，并结束循环
            if is_same_user(jwt_body['kubernetes.io/serviceaccount/service-account.name'],
                            jwt_body['kubernetes.io/serviceaccount/namespace'],
                            risky_user.user_info.name, risky_user.user_info.namespace):
                risky_user_in_container = risky_user
                break
    # 返回容器中的风险用户
    return risky_user_in_container


# 获取风险容器
def get_risky_containers(pod, risky_users, read_token_from_container=False):
    # 初始化风险容器列表
    risky_containers = []
    # 如果需要从容器中读取令牌
    if read_token_from_container:
        # 跳过已终止和驱逐的 pod
        # 这将仅在具有 "ready" 状态的容器上运行
        if pod.status.container_statuses:
            # 遍历所有容器状态
            for container in pod.status.container_statuses:
                # 如果容器处于就绪状态
                if container.ready:
                    # 从容器中获取 JWT 令牌
                    jwt_body, _ = get_jwt_token_from_container(pod, container.name)
                    # 如果成功获取到 JWT 令牌
                    if jwt_body:
                        # 获取容器中的风险用户
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
        # 遍历 pod.spec.volumes 列表，将卷的名称和卷对象存入字典
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
                    # 获取最高优先级
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
    # 遍历风险用户列表
    for user in risky_users_list:
        # 如果用户的优先级高于当前最高优先级，则更新最高优先级
        if user.priority.value > highest_priority.value:
            highest_priority = user.priority
    # 返回最高优先级
    return highest_priority


# 从容器中获取风险用户
def get_risky_users_from_container(container, risky_users, pod, volumes_dict):
    risky_users_set = set()
    # 如果容器的 volume_mounts 为空，则使用 '[]' 进行检查
    for volume_mount in container.volume_mounts or []:
        # 如果 volume_mount 的名称在 volumes_dict 中
        if volume_mount.name in volumes_dict:
            # 如果 volume_mount 对应的 projected 不为空
            if volumes_dict[volume_mount.name].projected is not None:
                # 遍历 projected 的 sources
                for source in volumes_dict[volume_mount.name].projected.sources or []:
                    # 如果 source 的 service_account_token 不为空
                    if source.service_account_token is not None:
                        # 判断当前服务账户是否为风险用户
                        risky_user = is_user_risky(risky_users, pod.spec.service_account, pod.metadata.namespace)
                        # 如果是风险用户，则加入风险用户集合
                        if risky_user is not None:
                            risky_users_set.add(risky_user)
            # 如果 volume_mount 对应的 secret 不为空
            elif volumes_dict[volume_mount.name].secret is not None:
                # 获取 JWT 并解码，判断是否为风险用户
                risky_user = get_jwt_and_decode(pod, risky_users, volumes_dict[volume_mount.name])
                # 如果是风险用户，则加入风险用户集合
                if risky_user is not None:
                    risky_users_set.add(risky_user)
    # 返回风险用户集合
    return risky_users_set


# 判断容器是否存在于风险容器中
def container_exists_in_risky_containers(risky_containers, container_name, risky_users_list):
    # 遍历风险容器列表
    for risky_container in risky_containers:
        # 如果容器名称与指定容器名称相同
        if risky_container.name == container_name:
            # 将风险用户列表中的用户名称添加到风险容器的 service_account_name 中
            for user_name in risky_users_list:
                risky_container.service_account_name.append(user_name)
            # 返回 True，表示容器存在于风险容器中
            return True
    # 如果未找到指定容器，则返回 False
    return False


# 判断默认路径是否存在于 volume_mounts 中
def default_path_exists(volume_mounts):
    # 遍历 volume_mounts
    for volume_mount in volume_mounts:
        # 如果 mount_path 为 "/var/run/secrets/kubernetes.io/serviceaccount"，则返回 True
        if volume_mount.mount_path == "/var/run/secrets/kubernetes.io/serviceaccount":
            return True
    # 如果未找到默认路径，则返回 False
    return False


# 判断用户是否为风险用户
def is_user_risky(risky_users, service_account, namespace):
    # 在此处添加判断用户是否为风险用户的逻辑
    # ...
    # 遍历风险用户列表
    for risky_user in risky_users:
        # 检查风险用户的用户名和命名空间是否与服务账号和命名空间匹配
        if risky_user.user_info.name == service_account and risky_user.user_info.namespace == namespace:
            # 如果匹配，则返回该风险用户
            return risky_user
    # 如果没有匹配的风险用户，则返回 None
    return None
# 获取 JWT 并解码，根据风险用户和卷返回风险用户
def get_jwt_and_decode(pod, risky_users, volume):
    # 导入解码 base64 JWT 令牌的函数
    from engine.jwt_token import decode_base64_jwt_token
    try:
        # 读取指定命名空间中的密钥
        secret = api_client.CoreV1Api.read_namespaced_secret(name=volume.secret.secret_name,
                                                             namespace=pod.metadata.namespace)
    except Exception:
        secret = None
    try:
        if secret is not None and secret.data is not None:
            if 'token' in secret.data:
                # 解码 base64 JWT 令牌
                decoded_data = decode_base64_jwt_token(secret.data['token'])
                # 将解码后的数据转换为 JSON 格式
                token_body = json.loads(decoded_data)
                if token_body:
                    # 从容器中获取风险用户
                    risky_user = get_risky_user_from_container(token_body, risky_users)
                    return risky_user
        # 抛出异常
        raise Exception()
    except Exception:
        if secret is not None:
            # 从密钥中获取风险用户
            return get_risky_user_from_container_secret(secret, risky_users)

# 从密钥中获取风险用户
def get_risky_user_from_container_secret(secret, risky_users):
    if secret is not None:
        # 全局变量，存储服务账户列表
        global list_of_service_accounts
        if not list_of_service_accounts:
            # 获取所有命名空间的服务账户列表
            list_of_service_accounts = api_client.CoreV1Api.list_service_account_for_all_namespaces()
        for sa in list_of_service_accounts.items:
            for service_account_secret in sa.secrets or []:
                if secret.metadata.name == service_account_secret.name:
                    for risky_user in risky_users:
                        if risky_user.user_info.name == sa.metadata.name:
                            return risky_user

# 获取风险 Pod
def get_risky_pods(namespace=None, deep_analysis=False):
    risky_pods = []
    # 获取所有 Pod 列表
    pods = list_pods_for_all_namespaces_or_one_namspace(namespace)
    # 获取所有风险主体
    risky_users = get_all_risky_subjects()
    for pod in pods.items:
        # 获取 Pod 中的风险容器
        risky_containers = get_risky_containers(pod, risky_users, deep_analysis)
        if len(risky_containers) > 0:
            # 将风险 Pod 添加到列表中
            risky_pods.append(Pod(pod.metadata.name, pod.metadata.namespace, risky_containers))

    return risky_pods
# endregion- Risky Pods

# 获取所有命名空间的角色绑定和集群角色绑定
def get_rolebindings_all_namespaces_and_clusterrolebindings():
    # 获取所有命名空间的角色绑定
    namespaced_rolebindings = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()

    # 获取集群角色绑定
    cluster_rolebindings = api_client.api_temp.list_cluster_role_binding()
    return namespaced_rolebindings, cluster_rolebindings


# 获取与主体关联的角色绑定和集群角色绑定
def get_rolebindings_and_clusterrolebindings_associated_to_subject(subject_name, kind, namespace):
    # 获取所有命名空间的角色绑定和集群角色绑定
    rolebindings_all_namespaces, cluster_rolebindings = get_rolebindings_all_namespaces_and_clusterrolebindings()
    associated_rolebindings = []

    # 遍历所有命名空间的角色绑定
    for rolebinding in rolebindings_all_namespaces.items:
        # 如果 'rolebinding.subjects' 为 'None'，'or []' 将防止出现异常
        for subject in rolebinding.subjects or []:
            if subject.name.lower() == subject_name.lower() and subject.kind.lower() == kind.lower():
                if kind == SERVICEACCOUNT_KIND:
                    if subject.namespace.lower() == namespace.lower():
                        associated_rolebindings.append(rolebinding)
                else:
                    associated_rolebindings.append(rolebinding)

    associated_clusterrolebindings = []
    # 遍历集群角色绑定
    for clusterrolebinding in cluster_rolebindings:
        # 如果 'clusterrolebinding.subjects' 为 'None'，'or []' 将防止出现异常
        for subject in clusterrolebinding.subjects or []:
            if subject.name == subject_name.lower() and subject.kind.lower() == kind.lower():
                if kind == SERVICEACCOUNT_KIND:
                    if subject.namespace.lower() == namespace.lower():
                        associated_clusterrolebindings.append(clusterrolebinding)
                else:
                    associated_clusterrolebindings.append(clusterrolebinding)

    return associated_rolebindings, associated_clusterrolebindings
# 获取与指定角色相关联的角色绑定
def get_rolebindings_associated_to_role(role_name, namespace):
    # 获取所有命名空间中的角色绑定
    rolebindings_all_namespaces = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()
    associated_rolebindings = []

    # 遍历所有角色绑定
    for rolebinding in rolebindings_all_namespaces.items:
        # 如果角色绑定的角色名称与指定的角色名称相同，并且命名空间也相同，则将其添加到相关角色绑定列表中
        if rolebinding.role_ref.name.lower() == role_name.lower() and rolebinding.role_ref.kind == ROLE_KIND and rolebinding.metadata.namespace.lower() == namespace.lower():
            associated_rolebindings.append(rolebinding)

    return associated_rolebindings


# 获取与集群角色相关联的角色绑定和集群角色绑定
def get_rolebindings_and_clusterrolebindings_associated_to_clusterrole(role_name):
    # 获取所有命名空间中的角色绑定和集群角色绑定
    rolebindings_all_namespaces, cluster_rolebindings = get_rolebindings_all_namespaces_and_clusterrolebindings()

    associated_rolebindings = []

    # 遍历所有角色绑定
    for rolebinding in rolebindings_all_namespaces.items:
        # 如果角色绑定的角色名称与指定的集群角色名称相同，则将其添加到相关角色绑定列表中
        if rolebinding.role_ref.name.lower() == role_name.lower() and rolebinding.role_ref.kind == CLUSTER_ROLE_KIND:
            associated_rolebindings.append(rolebinding)

    associated_clusterrolebindings = []

    # 遍历所有集群角色绑定
    for clusterrolebinding in cluster_rolebindings:
        # 如果集群角色绑定的角色名称与指定的集群角色名称相同，则将其添加到相关角色绑定列表中
        if clusterrolebinding.role_ref.name.lower() == role_name.lower() and clusterrolebinding.role_ref.kind == CLUSTER_ROLE_KIND:
            associated_rolebindings.append(clusterrolebinding)

    return associated_rolebindings, associated_clusterrolebindings


# 根据 Pod 名称和命名空间获取容器中的令牌
def dump_containers_tokens_by_pod(pod_name, namespace, read_token_from_container=False):
    containers_with_tokens = []
    try:
        # 读取指定命名空间中的 Pod
        pod = api_client.CoreV1Api.read_namespaced_pod(name=pod_name, namespace=namespace)
    except ApiException:
        # 如果出现异常，打印错误信息并返回 None
        print(pod_name + " was not found in " + namespace + " namespace")
        return None
    # 如果从容器中读取令牌
    if read_token_from_container:
        # 如果 Pod 的状态包含容器状态
        if pod.status.container_statuses:
            # 遍历 Pod 的容器状态
            for container in pod.status.container_statuses:
                # 如果容器已准备就绪
                if container.ready:
                    # 从容器中获取 JWT 令牌的主体和原始 JWT 令牌
                    jwt_body, raw_jwt_token = get_jwt_token_from_container(pod, container.name)
                    # 如果存在 JWT 令牌主体
                    if jwt_body:
                        # 将容器名称、令牌主体和原始 JWT 令牌添加到容器令牌列表中
                        containers_with_tokens.append(
                            Container(container.name, token=jwt_body, raw_jwt_token=raw_jwt_token))
    # 如果不从容器中读取令牌
    else:
        # 使用 Pod 填充容器令牌列表
        fill_container_with_tokens_list(containers_with_tokens, pod)
    # 返回容器令牌列表
    return containers_with_tokens
# 用 tokens 列表填充容器
def fill_container_with_tokens_list(containers_with_tokens, pod):
    # 导入解码 base64 JWT token 的函数
    from engine.jwt_token import decode_base64_jwt_token
    # 遍历 Pod 中的容器
    for container in pod.spec.containers:
        # 遍历容器的挂载卷
        for volume_mount in container.volume_mounts or []:
            # 遍历 Pod 的卷
            for volume in pod.spec.volumes or []:
                # 如果卷的名称与挂载卷的名称相同，并且卷是一个 secret 类型的卷
                if volume.name == volume_mount.name and volume.secret:
                    try:
                        # 读取 secret 对象
                        secret = api_client.CoreV1Api.read_namespaced_secret(volume.secret.secret_name,
                                                                             pod.metadata.namespace)
                        # 如果 secret 存在并且包含 token 数据
                        if secret and secret.data and secret.data['token']:
                            # 解码 base64 JWT token
                            decoded_data = decode_base64_jwt_token(secret.data['token'])
                            # 将解码后的数据转换为 JSON 格式
                            token_body = json.loads(decoded_data)
                            # 将容器名称和 token 添加到容器列表中
                            containers_with_tokens.append(Container(container.name, token=token_body,
                                                                    raw_jwt_token=None))
                    except ApiException:
                        # 捕获异常，打印错误信息
                        print("No secret found.")


# 获取所有 Pod 的 token 或者根据命名空间获取
def dump_all_pods_tokens_or_by_namespace(namespace=None, read_token_from_container=False):
    # 存储带有 token 的 Pod 列表
    pods_with_tokens = []
    # 获取所有 Pod 或者特定命名空间的 Pod 列表
    pods = list_pods_for_all_namespaces_or_one_namspace(namespace)
    # 遍历 Pod 列表
    for pod in pods.items:
        # 获取 Pod 中的容器列表的 token
        containers = dump_containers_tokens_by_pod(pod.metadata.name, pod.metadata.namespace, read_token_from_container)
        # 如果容器列表不为空
        if containers is not None:
            # 将 Pod 名称、命名空间和容器列表添加到带有 token 的 Pod 列表中
            pods_with_tokens.append(Pod(pod.metadata.name, pod.metadata.namespace, containers))

    return pods_with_tokens


# 获取 Pod 的 token
def dump_pod_tokens(name, namespace, read_token_from_container=False):
    # 存储带有 token 的 Pod 列表
    pod_with_tokens = []
    # 获取 Pod 中的容器列表的 token
    containers = dump_containers_tokens_by_pod(name, namespace, read_token_from_container)
    # 将 Pod 名称、命名空间和容器列表添加到带有 token 的 Pod 列表中
    pod_with_tokens.append(Pod(name, namespace, containers))

    return pod_with_tokens


# 根据类型在 subjects 中搜索主体
def search_subject_in_subjects_by_kind(subjects, kind):
    # 存储找到的主体列表
    subjects_found = []
    # 遍历给定的主题列表
    for subject in subjects:
        # 检查主题的类型是否与给定类型相同（不区分大小写）
        if subject.kind.lower() == kind.lower():
            # 如果相同，将主题添加到已找到的主题列表中
            subjects_found.append(subject)
    # 返回已找到的主题列表
    return subjects_found
# 根据角色类型获取所有角色绑定的主体
def get_subjects_by_kind(kind):
    # 初始化主体列表
    subjects_found = []
    # 获取所有命名空间的角色绑定
    rolebindings = api_client.RbacAuthorizationV1Api.list_role_binding_for_all_namespaces()
    # 获取集群角色绑定
    clusterrolebindings = api_client.api_temp.list_cluster_role_binding()
    # 遍历角色绑定列表
    for rolebinding in rolebindings.items:
        # 如果角色绑定的主体不为空
        if rolebinding.subjects is not None:
            # 在主体列表中搜索指定类型的主体
            subjects_found += search_subject_in_subjects_by_kind(rolebinding.subjects, kind)

    # 遍历集群角色绑定列表
    for clusterrolebinding in clusterrolebindings:
        # 如果集群角色绑定的主体不为空
        if clusterrolebinding.subjects is not None:
            # 在主体列表中搜索指定类型的主体
            subjects_found += search_subject_in_subjects_by_kind(clusterrolebinding.subjects, kind)

    # 返回去重后的主体列表
    return remove_duplicated_subjects(subjects_found)


# 去除重复的主体
def remove_duplicated_subjects(subjects):
    # 初始化集合用于存储已经出现过的主体
    seen_subjects = set()
    # 初始化新的主体列表
    new_subjects = []
    # 遍历主体列表
    for s1 in subjects:
        # 根据主体的名称、命名空间和类型生成唯一标识
        if s1.namespace == None:
            s1_unique_name = ''.join([s1.name, s1.kind])
        else:
            s1_unique_name = ''.join([s1.name, s1.namespace, s1.kind])
        # 如果主体的唯一标识不在已经出现过的主体集合中
        if s1_unique_name not in seen_subjects:
            # 将主体添加到新的主体列表中
            new_subjects.append(s1)
            # 将主体的唯一标识添加到已经出现过的主体集合中
            seen_subjects.add(s1_unique_name)

    # 返回去重后的主体列表
    return new_subjects


# 获取角色绑定的角色
def get_rolebinding_role(rolebinding_name, namespace):
    rolebinding = None
    role = None
    try:
        # 读取指定命名空间的角色绑定
        rolebinding = api_client.RbacAuthorizationV1Api.read_namespaced_role_binding(rolebinding_name, namespace)
        # 如果角色绑定的角色类型是指定的角色类型
        if rolebinding.role_ref.kind == ROLE_KIND:
            # 读取指定命名空间的角色
            role = api_client.RbacAuthorizationV1Api.read_namespaced_role(rolebinding.role_ref.name,
                                                                          rolebinding.metadata.namespace)
        else:
            # 读取集群角色
            role = api_client.RbacAuthorizationV1Api.read_cluster_role(rolebinding.role_ref.name)

        # 返回角色
        return role
    # 捕获特定的 ApiException 异常
    except ApiException:
        # 如果 rolebinding 为空，打印未找到指定角色绑定的信息
        if rolebinding is None:
            print("Could not find " + rolebinding_name + " rolebinding in " + namespace + " namespace")
        # 如果 role 为空，打印未找到指定角色绑定的角色信息
        elif role is None:
            print(
                "Could not find " + rolebinding.role_ref.name + " role in " + rolebinding.role_ref.name + " rolebinding")
        # 返回空值
        return None
# 根据集群角色绑定名称获取集群角色
def get_clusterrolebinding_role(cluster_rolebinding_name):
    cluster_role = ''
    try:
        # 通过 API 客户端读取集群角色绑定对象
        cluster_rolebinding = api_client.RbacAuthorizationV1Api.read_cluster_role_binding(cluster_rolebinding_name)
        # 通过 API 客户端读取集群角色对象
        cluster_role = api_client.RbacAuthorizationV1Api.read_cluster_role(cluster_rolebinding.role_ref.name)
    except ApiException as e:
        # 打印异常信息并退出程序
        print(e)
        exit()
    # 返回集群角色对象
    return cluster_role


# 获取与主体关联的角色
def get_roles_associated_to_subject(subject_name, kind, namespace):
    # 获取与主体关联的角色绑定和集群角色绑定
    associated_rolebindings, associated_clusterrolebindings = get_rolebindings_and_clusterrolebindings_associated_to_subject(
        subject_name, kind, namespace)

    associated_roles = []
    # 遍历关联的角色绑定
    for rolebind in associated_rolebindings:
        try:
            # 获取角色绑定的角色
            role = get_rolebinding_role(rolebind.metadata.name, rolebind.metadata.namespace)
            associated_roles.append(role)
        except ApiException as e:
            # 404 not found
            # 如果发生异常，继续下一个角色绑定
            continue

    # 遍历关联的集群角色绑定
    for clusterrolebinding in associated_clusterrolebindings:
        # 获取集群角色绑定的角色
        role = get_clusterrolebinding_role(clusterrolebinding.metadata.name)
        associated_roles.append(role)

    # 返回关联的角色列表
    return associated_roles


# 列出所有命名空间的 Pod 或指定命名空间的 Pod
def list_pods_for_all_namespaces_or_one_namspace(namespace=None):
    try:
        if namespace is None:
            # 如果未指定命名空间，则列出所有命名空间的 Pod
            pods = api_client.CoreV1Api.list_pod_for_all_namespaces(watch=False)
        else:
            # 如果指定了命名空间，则列出该命名空间的 Pod
            pods = api_client.CoreV1Api.list_namespaced_pod(namespace)
        return pods
    except ApiException:
        # 如果发生异常，返回空值
        return None


# 列出解码后的引导令牌
def list_boostrap_tokens_decoded():
    tokens = []
    # 列出 kube-system 命名空间中类型为 bootstrap.kubernetes.io/token 的 Secret
    secrets = api_client.CoreV1Api.list_namespaced_secret(namespace='kube-system',
                                                          field_selector='type=bootstrap.kubernetes.io/token')
    import base64
    # 遍历秘钥字典中的每个秘钥
    for secret in secrets.items:
        # 将 token-id 和 token-secret 解码为 UTF-8 格式，并拼接成字符串，添加到 tokens 列表中
        tokens.append('.'.join((base64.b64decode(secret.data['token-id']).decode('utf-8'),
                                base64.b64decode(secret.data['token-secret']).decode('utf-8'))))
    # 返回 tokens 列表
    return tokens
```