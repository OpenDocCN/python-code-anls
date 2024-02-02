# `KubiScan\KubiScan.py`

```py
# 导入所需的模块
import json
import logging
import os
import re
import sys
from argparse import ArgumentParser
import engine.utils
import engine.privleged_containers
from prettytable import PrettyTable, ALL
from engine.priority import Priority
from misc.colours import *
from misc import constants
import datetime
from api.api_client import api_init, running_in_container

# 初始化变量
json_filename = ""
output_file = ""
no_color = False
curr_header = ""

# 根据优先级获取颜色
def get_color_by_priority(priority):
    color = WHITE
    if priority == Priority.CRITICAL:
        color = RED
    elif priority == Priority.HIGH:
        color = LIGHTYELLOW
    return color

# 根据天数过滤对象
def filter_objects_less_than_days(days, objects):
    filtered_objects= []
    current_datetime = datetime.datetime.now()
    for object in objects:
        if object.time:
            if (current_datetime.date() - object.time.date()).days < days:
                filtered_objects.append(object)
    objects = filtered_objects
    return objects

# 根据优先级过滤对象
def filter_objects_by_priority(priority, objects):
    filtered_objects= []
    for object in objects:
        if object.priority.name == priority.upper():
            filtered_objects.append(object)
    objects = filtered_objects
    return objects

# 获取从现在到指定日期的天数差
def get_delta_days_from_now(date):
    current_datetime = datetime.datetime.now()
    return (current_datetime.date() - date.date()).days

# 打印所有风险角色
def print_all_risky_roles(show_rules=False, days=None, priority=None, namespace=None):
    # 获取所有风险角色
    risky_any_roles = engine.utils.get_risky_roles_and_clusterroles()
    # 如果指定了命名空间，则发出警告
    if namespace is not None:
        logging.warning("'-rar' switch does not expect namespace ('-ns')\n")
    # 如果指定了天数，则根据天数过滤
    if days:
        risky_any_roles = filter_objects_less_than_days(int(days), risky_any_roles)
    # 如果指定了优先级，则根据优先级过滤
    if priority:
        risky_any_roles = filter_objects_by_priority(priority, risky_any_roles)
    # 打印所有风险角色
    generic_print('|Risky Roles and ClusterRoles|', risky_any_roles, show_rules)

# 打印风险角色
def print_risky_roles(show_rules=False, days=None, priority=None, namespace=None):
    # 获取风险角色列表
    risky_roles = engine.utils.get_risky_roles()
    
    # 如果指定了天数，则筛选出指定天数内创建的风险角色
    if days:
        risky_roles = filter_objects_less_than_days(int(days), risky_roles)
    
    # 如果指定了优先级，则筛选出指定优先级的风险角色
    if priority:
        risky_roles = filter_objects_by_priority(priority, risky_roles)
    
    # 初始化一个空的筛选后的风险角色列表
    filtered_risky_roles = []
    
    # 如果未指定命名空间，则打印所有风险角色
    if namespace is None:
        generic_print('|Risky Roles |', risky_roles, show_rules)
    # 如果指定了命名空间，则筛选出指定命名空间的风险角色并打印
    else:
        for risky_role in risky_roles:
            # 如果风险角色的命名空间与指定的命名空间相符，则加入筛选后的列表
            if risky_role.namespace == namespace:
                filtered_risky_roles.append(risky_role)
        # 打印筛选后的风险角色列表
        generic_print('|Risky Roles |', filtered_risky_roles, show_rules)
# 打印与当前 Kubernetes 版本相关的 CVE（通用漏洞和暴露）信息
def print_cve(certificate_authority_file=None, client_certificate_file=None, client_key_file=None, host=None):
    # 获取当前 Kubernetes 版本
    current_k8s_version = engine.utils.get_current_version(certificate_authority_file, client_certificate_file, client_key_file, host)
    # 如果当前 Kubernetes 版本为空，则返回
    if current_k8s_version is None:
        return
    # 获取受当前 Kubernetes 版本影响的所有 CVE 表格
    cve_table = get_all_affecting_cves_table_by_version(current_k8s_version)
    # 打印对齐左边的 CVE 表格
    print_table_aligned_left(cve_table)

# 根据版本获取受影响的所有 CVE 表格
def get_all_affecting_cves_table_by_version(current_k8s_version):
    # 创建包含列名的漂亮表格
    cve_table = PrettyTable(['Severity', 'CVE Grade', 'CVE', 'Description', 'FixedVersions'])
    # 设置表格的水平线规则
    cve_table.hrules = ALL
    # 打开 CVE.json 文件并加载数据
    with open('CVE.json', 'r') as f:
        data = json.load(f)
    # 获取所有 CVE
    cves = data['CVES']
    # 遍历每个 CVE
    for cve in cves:
        # 如果当前版本受影响
        if curr_version_is_affected(cve, current_k8s_version):
            # 分割 CVE 描述
            cve_description = split_cve_description(cve['Description'])
            # 获取 CVE 的修复版本列表
            fixed_version_list = get_fixed_versions_of_cve(cve["FixedVersions"])
            # 获取 CVE 的颜色
            cve_color = get_cve_color(cve['Severity'])
            # 向表格添加一行数据
            cve_table.add_row([cve_color + cve['Severity'] + WHITE, cve['Grade'], cve['CVENumber'], cve_description,
                       fixed_version_list])
    # 按照 CVE 等级排序表格
    cve_table.sortby = "CVE Grade"
    # 反向排序表格
    cve_table.reversesort = True
    # 返回 CVE 表格
    return cve_table

# 根据 CVE 严重程度获取颜色
def get_cve_color(cve_severity):
    if cve_severity == "Low":
        return WHITE
    elif cve_severity == "Medium":
        return LIGHTYELLOW
    elif cve_severity == "High" or cve_severity == "Critical":
        return RED

# 获取 CVE 的修复版本列表
def get_fixed_versions_of_cve(cve_fixed_versions):
    fixed_version_list = ""
    # 遍历每个修复版本
    for fixed_version in cve_fixed_versions:
        fixed_version_list += fixed_version["Raw"] + "\n"
    # 返回修复版本列表（去除最后一个换行符）
    return fixed_version_list[:-1]

# 分割 CVE 描述
def split_cve_description(cve_description):
    words = cve_description.split()
    res_description = ""
    words_in_row = 10
    # 遍历每个单词
    for i, word in enumerate(words):
        # 每行 10 个单词
        if i % words_in_row == 0 and i != 0:
            res_description += "\n"
        res_description += word + " "
    # 返回描述（去除最后一个空格）
    return res_description[:-1]
# 判断当前的 Kubernetes 版本是否受到 CVE 影响
def curr_version_is_affected(cve, current_k8s_version):
    # 获取 CVE 的最大修复版本号
    max_fixed_version = find_max_fixed_version(cve)
    # 获取 CVE 的最小修复版本号
    min_fixed_version = find_min_fixed_version(cve)
    # 如果当前 Kubernetes 版本小于最小修复版本号，则受影响
    if compare_versions(current_k8s_version, min_fixed_version) == -1:
        return True
    # 如果当前 Kubernetes 版本大于等于最大修复版本号，则不受影响
    if compare_versions(current_k8s_version, max_fixed_version) >= 0:
        return False
    # 遍历 CVE 的修复版本列表，判断当前版本是否在修复版本之间
    for fixed_version in cve['FixedVersions']:
        if is_vulnerable_in_middle(current_k8s_version, fixed_version['Raw']):
            return True
    return False


# 判断当前 Kubernetes 版本是否在修复版本之间存在漏洞
def is_vulnerable_in_middle(current_k8s_version, cve_fixed_version):
    # 将版本号字符串转换为数字列表，例如：1.15.2 -> [1, 15, 2]
    current_k8s_version_nums = [int(num) for num in current_k8s_version.split('.')]
    fixed_version_nums = [int(num) for num in cve_fixed_version.split('.')]
    # 如果主版本号和次版本号相同，但修订版本号大于当前版本，则存在漏洞
    if fixed_version_nums[0] == current_k8s_version_nums[0] and fixed_version_nums[1] == current_k8s_version_nums[1]:
        if fixed_version_nums[2] > current_k8s_version_nums[2]:
            return True
    return False


# 比较两个版本号的大小
# 如果 version1 > version2 返回 1
# 如果 version1 < version2 返回 -1
# 如果 version1 = version2 返回 0
def compare_versions(version1, version2):
    version1_nums = [int(num) for num in version1.split('.')]
    version2_nums = [int(num) for num in version2.split('.')]
    for i in range(len(version2_nums)):
        if version2_nums[i] > version1_nums[i]:
            return -1
        elif version2_nums[i] < version1_nums[i]:
            return 1
    else:
        return 0


# 获取 CVE 的最大修复版本号
def find_max_fixed_version(cve):
    versions = []
    for fixed_version in cve['FixedVersions']:
        versions.append(fixed_version['Raw'])
    # 返回修复版本号列表中的最大版本号
    max_version = max(versions, key=lambda x: [int(num) for num in x.split('.')])
    return max_version

# 获取 CVE 的最小修复版本号
def find_min_fixed_version(cve):
    versions = []
    for fixed_version in cve['FixedVersions']:
        versions.append(fixed_version['Raw'])
    # 这里应该有返回值，但是代码中缺少了返回语句
    # 从版本列表中找到最小的版本号，使用lambda函数将版本号转换为数字列表，然后找到最小的版本号
    min_version = min(versions, key=lambda x: [int(num) for num in x.split('.')])
    # 返回最小的版本号
    return min_version
# 打印风险的集群角色，可以选择是否显示规则，以及按天数、优先级和命名空间进行过滤
def print_risky_clusterroles(show_rules=False, days=None, priority=None, namespace=None):
    # 如果命名空间不为空，则发出警告
    if namespace is not None:
        logging.warning("'-rcr' switch does not expect namespace ('-ns')\n")
    # 获取风险的集群角色
    risky_clusterroles = engine.utils.get_risky_clusterroles()
    # 如果指定了天数，则按天数过滤
    if days:
        risky_clusterroles = filter_objects_less_than_days(int(days), risky_clusterroles)
    # 如果指定了优先级，则按优先级过滤
    if priority:
        risky_clusterroles = filter_objects_by_priority(priority, risky_clusterroles)
    # 打印风险的集群角色
    generic_print('|Risky ClusterRoles |', risky_clusterroles, show_rules)

# 打印所有风险的角色绑定，可以按天数和优先级进行过滤
def print_all_risky_rolebindings(days=None, priority=None, namespace=None):
    # 如果命名空间不为空，则发出警告
    if namespace is not None:
        logging.warning("'-rab' switch does not expect namespace ('-ns')\n")
    # 获取所有风险的角色绑定
    risky_any_rolebindings = engine.utils.get_all_risky_rolebinding()
    # 如果指定了天数，则按天数过滤
    if days:
        risky_any_rolebindings = filter_objects_less_than_days(int(days), risky_any_rolebindings)
    # 如果指定了优先级，则按优先级过滤
    if priority:
        risky_any_rolebindings = filter_objects_by_priority(priority, risky_any_rolebindings)
    # 打印所有风险的角色绑定
    generic_print('|Risky RoleBindings and ClusterRoleBindings|', risky_any_rolebindings)

# 打印风险的角色绑定，可以按天数、优先级和命名空间进行过滤
def print_risky_rolebindings(days=None, priority=None, namespace=None):
    # 获取风险的角色绑定
    risky_rolebindings = engine.utils.get_risky_rolebindings()
    # 如果指定了天数，则按天数过滤
    if days:
        risky_rolebindings = filter_objects_less_than_days(int(days), risky_rolebindings)
    # 如果指定了优先级，则按优先级过滤
    if priority:
        risky_rolebindings = filter_objects_by_priority(priority, risky_rolebindings)
    # 如果命名空间为空，则打印风险的角色绑定
    if namespace is None:
        generic_print('|Risky RoleBindings|', risky_rolebindings)
    else:
        # 否则，按命名空间过滤后打印风险的角色绑定
        filtered_risky_rolebindings = []
        for risky_rolebinding in risky_rolebindings:
            if risky_rolebinding.namespace == namespace:
                filtered_risky_rolebindings.append(risky_rolebinding)
        generic_print('|Risky RoleBindings|', filtered_risky_rolebindings)

# 打印风险的集群角色绑定，可以按天数、优先级和命名空间进行过滤
def print_risky_clusterrolebindings(days=None, priority=None, namespace=None):
    # 如果命名空间不为空，则记录警告信息
    if namespace is not None:
        logging.warning("'-rcb' switch does not expect namespace ('-ns')\n")
    # 获取存在风险的集群角色绑定
    risky_clusterrolebindings = engine.utils.get_risky_clusterrolebindings()
    # 如果指定了天数，则筛选出指定天数内的集群角色绑定
    if days:
        risky_clusterrolebindings = filter_objects_less_than_days(int(days), risky_clusterrolebindings)
    # 如果指定了优先级，则根据优先级筛选集群角色绑定
    if priority:
        risky_clusterrolebindings = filter_objects_by_priority(priority, risky_clusterrolebindings)
    # 打印存在风险的集群角色绑定
    generic_print('|Risky ClusterRoleBindings|', risky_clusterrolebindings)
# 定义一个通用的打印函数，用于打印对象列表，并可选择是否显示规则
def generic_print(header, objects, show_rules=False):
    # 生成顶部边框
    roof = '+' + ('-' * (len(header)-2)) + '+'
    # 声明全局变量 curr_header
    global curr_header
    # 将当前标题赋值给 curr_header
    curr_header = header
    # 打印顶部边框
    print(roof)
    # 打印标题
    print(header)
    # 如果需要显示规则
    if show_rules:
        # 创建一个 PrettyTable 对象，包含优先级、类型、命名空间、名称、创建时间和规则
        t = PrettyTable(['Priority', 'Kind', 'Namespace', 'Name', 'Creation Time', 'Rules'])
        # 遍历对象列表
        for o in objects:
            # 如果时间为空
            if o.time is None:
                # 添加一行数据到 PrettyTable 对象
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, 'No creation time', get_pretty_rules(o.rules)])
            else:
                # 添加一行数据到 PrettyTable 对象
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, o.time.ctime() + " (" + str(get_delta_days_from_now(o.time)) + " days)", get_pretty_rules(o.rules)])
    # 如果不需要显示规则
    else:
        # 创建一个 PrettyTable 对象，包含优先级、类型、命名空间、名称和创建时间
        t = PrettyTable(['Priority', 'Kind', 'Namespace', 'Name', 'Creation Time'])
        # 遍历对象列表
        for o in objects:
            # 如果时间为空
            if o.time is None:
                # 添加一行数据到 PrettyTable 对象
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, 'No creation time'])
            else:
                # 添加一行数据到 PrettyTable 对象
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, o.time.ctime() + " (" + str(get_delta_days_from_now(o.time)) + " days)"])
    # 打印格式化对齐的 PrettyTable 对象
    print_table_aligned_left(t)

# 打印所有风险容器的信息
def print_all_risky_containers(priority=None, namespace=None, read_token_from_container=False):
    # 获取所有风险的 Pod 列表
    pods = engine.utils.get_risky_pods(namespace, read_token_from_container)
    # 声明全局变量 curr_header
    global curr_header
    # 将当前标题赋值给 curr_header
    curr_header = "|Risky Containers|"
    # 打印顶部边框
    print("+----------------+")
    # 打印标题
    print("|Risky Containers|")
    # 创建一个 PrettyTable 对象，包含优先级、Pod 名称、命名空间、容器名称、服务账户命名空间和服务账户名称
    t = PrettyTable(['Priority', 'PodName', 'Namespace', 'ContainerName', 'ServiceAccountNamespace', 'ServiceAccountName'])
    # 遍历 pods 列表中的每个 pod
    for pod in pods:
        # 如果存在优先级，则根据优先级过滤 pod 中的容器对象
        if priority:
            pod.containers = filter_objects_by_priority(priority, pod.containers)
        # 遍历每个 pod 中的容器对象
        for container in pod.containers:
            # 初始化所有服务账户为空字符串
            all_service_account = ''
            # 遍历容器对象中的服务账户名称集合
            for service_account in container.service_accounts_name_set:
                # 将服务账户名称拼接到所有服务账户字符串中
                all_service_account += service_account.user_info.name + ", "
            # 去除最后的逗号和空格
            all_service_account = all_service_account[:-2]
            # 向表格中添加一行数据，包括容器优先级、pod名称、命名空间、容器名称、服务账户命名空间和所有服务账户
            t.add_row([get_color_by_priority(container.priority)+container.priority.name+WHITE, pod.name, pod.namespace, container.name, container.service_account_namespace, all_service_account])

    # 打印左对齐的表格
    print_table_aligned_left(t)
# 打印所有风险主体的信息
def print_all_risky_subjects(priority=None, namespace=None):
    # 获取所有风险主体
    subjects = engine.utils.get_all_risky_subjects()
    # 如果有优先级参数，则根据优先级筛选主体
    if priority:
        subjects = filter_objects_by_priority(priority, subjects)
    # 设置全局变量 curr_header
    global curr_header
    curr_header = "|Risky Users|"
    # 打印表头
    print("+-----------+")
    print("|Risky Users|")
    # 创建 PrettyTable 对象
    t = PrettyTable(['Priority', 'Kind', 'Namespace', 'Name'])
    # 遍历主体列表
    for subject in subjects:
        # 如果主体的命名空间与指定的命名空间相同，或者未指定命名空间
        if subject.user_info.namespace == namespace or namespace is None:
            # 将主体信息添加到 PrettyTable 中
            t.add_row([get_color_by_priority(subject.priority)+subject.priority.name+WHITE, subject.user_info.kind, subject.user_info.namespace, subject.user_info.name])
    # 打印格式化对齐的表格
    print_table_aligned_left(t)

# 打印所有相关信息
def print_all(days=None, priority=None, read_token_from_container=False):
    # 打印所有风险角色
    print_all_risky_roles(days=days, priority=priority)
    # 打印所有风险角色绑定
    print_all_risky_rolebindings(days=days, priority=priority)
    # 打印所有风险主体
    print_all_risky_subjects(priority=priority)
    # 打印所有风险容器
    print_all_risky_containers(priority=priority, read_token_from_container=False)

# 打印与指定角色相关的角色绑定
def print_associated_rolebindings_to_role(role_name, namespace=None):
    # 获取与指定角色相关的角色绑定
    associated_rolebindings = engine.utils.get_rolebindings_associated_to_role(role_name=role_name, namespace=namespace)
    # 打印相关角色绑定的表头
    print("Associated Rolebindings to Role \"{0}\":".format(role_name))
    # 创建 PrettyTable 对象
    t = PrettyTable(['Kind', 'Name', 'Namespace'])
    # 遍历相关角色绑定列表
    for rolebinding in associated_rolebindings:
        # 将角色绑定信息添加到 PrettyTable 中
        t.add_row(['RoleBinding', rolebinding.metadata.name, rolebinding.metadata.namespace])
    # 打印格式化对齐的表格
    print_table_aligned_left(t)

# 打印与指定集群角色相关的任何角色绑定
def print_associated_any_rolebindings_to_clusterrole(clusterrole_name):
    # 获取与指定集群角色相关的角色绑定和集群角色绑定
    associated_rolebindings, associated_clusterrolebindings = engine.utils.get_rolebindings_and_clusterrolebindings_associated_to_clusterrole(role_name=clusterrole_name)
    # 打印相关角色绑定的表头
    print("Associated Rolebindings\ClusterRoleBinding to ClusterRole \"{0}\":".format(clusterrole_name))
    # 创建 PrettyTable 对象
    t = PrettyTable(['Kind', 'Name', 'Namespace'])
    # 遍历关联的角色绑定列表，将每个角色绑定的类型、名称和命名空间添加到表格中
    for rolebinding in associated_rolebindings:
        t.add_row(['RoleBinding', rolebinding.metadata.name, rolebinding.metadata.namespace])

    # 遍历关联的集群角色绑定列表，将每个集群角色绑定的类型、名称和命名空间添加到表格中
    for clusterrolebinding in associated_clusterrolebindings:
        t.add_row(['ClusterRoleBinding', clusterrolebinding.metadata.name, clusterrolebinding.metadata.namespace])

    # 打印格式对齐的表格
    print_table_aligned_left(t)
# 打印与指定主体相关的角色绑定和集群角色绑定
def print_associated_rolebindings_and_clusterrolebindings_to_subject(subject_name, kind, namespace=None):
    # 获取与主体相关的角色绑定和集群角色绑定
    associated_rolebindings, associated_clusterrolebindings = engine.utils.get_rolebindings_and_clusterrolebindings_associated_to_subject(subject_name, kind, namespace)

    # 打印与主体相关的角色绑定和集群角色绑定的标题
    print("Associated Rolebindings\ClusterRoleBindings to subject \"{0}\":".format(subject_name))
    # 创建一个表格
    t = PrettyTable(['Kind', 'Name', 'Namespace'])

    # 遍历关联的角色绑定，添加到表格中
    for rolebinding in associated_rolebindings:
        t.add_row(['RoleBinding', rolebinding.metadata.name, rolebinding.metadata.namespace])

    # 遍历关联的集群角色绑定，添加到表格中
    for clusterrolebinding in associated_clusterrolebindings:
        t.add_row(['ClusterRoleBinding', clusterrolebinding.metadata.name, clusterrolebinding.metadata.namespace])

    # 打印表格
    print_table_aligned_left(t)

# 反序列化令牌
def desrialize_token(token):
    desirialized_token = ''
    # 遍历令牌的键值对，拼接成字符串
    for key in token.keys():
        desirialized_token += key + ': ' + token[key]
        desirialized_token += '\n'
    return desirialized_token

# 从 Pod 中获取令牌并打印
def dump_tokens_from_pods(pod_name=None, namespace=None, read_token_from_container=False):
    # 如果指定了 Pod 名称，则获取该 Pod 的令牌
    if pod_name is not None:
        pods_with_tokens = engine.utils.dump_pod_tokens(pod_name, namespace, read_token_from_container)
    # 否则获取所有 Pod 的令牌或指定命名空间的 Pod 的令牌
    else:
        pods_with_tokens = engine.utils.dump_all_pods_tokens_or_by_namespace(namespace, read_token_from_container)

    # 创建一个表格
    t = PrettyTable(['PodName',  'Namespace', 'ContainerName', 'Decoded Token'])
    # 遍历包含令牌的 Pod
    for pod in pods_with_tokens:
        # 遍历 Pod 中的容器
        for container in pod.containers:
            # 反序列化令牌并添加到表格中
            new_token = desrialize_token(container.token)
            t.add_row([pod.name, pod.namespace, container.name, new_token])

    # 打印表格
    print_table_aligned_left(t)

# 根据类型打印主体
def print_subjects_by_kind(kind):
    # 获取指定类型的主体
    subjects = engine.utils.get_subjects_by_kind(kind)
    # 打印所有角色绑定中指定类型的主体
    print('Subjects (kind: {0}) from all rolebindings:'.format(kind))
    # 创建一个表格
    t = PrettyTable(['Kind', 'Namespace', 'Name'])
    # 遍历主体并添加到表格中
    for subject in subjects:
        t.add_row([subject.kind, subject.namespace, subject.name])

    # 打印表格
    print_table_aligned_left(t)
    # 打印输出主题数量的信息，使用字符串格式化输出
    print('Total number: %s' % len(subjects))
# 获取格式化的权限规则
def get_pretty_rules(rules):
    pretty = ''
    if rules is not None:
        for rule in rules:
            verbs_string = '('
            for verb in rule.verbs:
                verbs_string += verb + ','
            verbs_string = verbs_string[:-1]
            verbs_string += ')->'

            resources_string = '('
            if rule.resources is None:
                resources_string += 'None'
            else:
                for resource in rule.resources:
                    resources_string += resource + ','

                resources_string = resources_string[:-1]
            resources_string += ')\n'
            pretty += verbs_string + resources_string
    return pretty

# 打印角色绑定的规则
def print_rolebinding_rules(rolebinding_name, namespace):
    role = engine.utils.get_rolebinding_role(rolebinding_name, namespace)
    print("RoleBinding '{0}\{1}' rules:".format(namespace, rolebinding_name))
    t = PrettyTable(['Kind', 'Namespace', 'Name', 'Rules'])
    t.add_row([role.kind, role.metadata.namespace, role.metadata.name, get_pretty_rules(role.rules)])

    print_table_aligned_left(t)

# 打印集群角色绑定的规则
def print_clusterrolebinding_rules(cluster_rolebinding_name):
    cluster_role = engine.utils.get_clusterrolebinding_role(cluster_rolebinding_name)
    print("ClusterRoleBinding '{0}' rules:".format(cluster_rolebinding_name))
    t = PrettyTable(['Kind', 'Namespace', 'Name', 'Rules'])
    t.add_row([cluster_role.kind, cluster_role.metadata.namespace, cluster_role.metadata.name, get_pretty_rules(cluster_role.rules)])

    print_table_aligned_left(t)

# 打印与主体关联的规则
def print_rules_associated_to_subject(name, kind, namespace=None):
    roles = engine.utils.get_roles_associated_to_subject(name, kind, namespace)
    print("Roles associated to Subject '{0}':".format(name))
    t = PrettyTable(['Kind', 'Namespace', 'Name', 'Rules'])
    for role in roles:
        t.add_row([role.kind, role.metadata.namespace, role.metadata.name, get_pretty_rules(role.rules)])

    print_table_aligned_left(t)
# 根据给定的命名空间列出所有或指定命名空间的 Pod
def print_pods_with_access_secret_via_volumes(namespace=None):
    # 调用工具函数列出指定命名空间的所有 Pod
    pods = engine.utils.list_pods_for_all_namespaces_or_one_namspace(namespace)

    # 打印具有通过卷访问秘密数据的 Pod
    print("Pods with access to secret data through volumes:")
    # 创建一个表格对象，包含 Pod 名称、命名空间、容器名称和挂载的秘密数据
    t = PrettyTable(['Pod Name', 'Namespace', 'Container Name', 'Volume Mounted Secrets'])
    # 遍历每个 Pod
    for pod in pods.items:
        # 遍历每个容器
        for container in pod.spec.containers:
            # 初始化挂载信息和秘密数量
            mount_info = ''
            secrets_num = 1
            # 如果容器有挂载卷
            if container.volume_mounts is not None:
                # 遍历每个挂载的卷
                for volume_mount in container.volume_mounts:
                    # 遍历每个卷
                    for volume in pod.spec.volumes:
                        # 如果卷是秘密卷并且名称匹配
                        if volume.secret is not None and volume.name == volume_mount.name:
                            # 将挂载信息添加到 mount_info 中
                            mount_info += '{2}. Mounted path: {0}\n   Secret name: {1}\n'.format(volume_mount.mount_path, volume.secret.secret_name, secrets_num)
                            secrets_num += 1
                # 如果挂载信息不为空，则将 Pod 信息添加到表格中
                if mount_info != '':
                    t.add_row([pod.metadata.name, pod.metadata.namespace, container.name, mount_info])

    # 打印格式化对齐的表格
    print_table_aligned_left(t)

# 根据给定的命名空间列出所有或指定命名空间的 Pod
def print_pods_with_access_secret_via_environment(namespace=None):
    # 调用工具函数列出指定命名空间的所有 Pod
    pods = engine.utils.list_pods_for_all_namespaces_or_one_namspace(namespace)

    # 打印具有通过环境变量访问秘密数据的 Pod
    print("Pods with access to secret data through environment:")
    # 创建一个表格对象，包含 Pod 名称、命名空间、容器名称和环境变量中的秘密数据
    t = PrettyTable(['Pod Name', 'Namespace', 'Container Name', 'Environment Mounted Secrets'])
    # 遍历所有的 pods 对象
    for pod in pods.items:
        # 遍历每个 pod 中的容器对象
        for container in pod.spec.containers:
            # 初始化挂载信息和秘钥数量
            mount_info = ''
            secrets_num = 1
            # 如果容器中有环境变量
            if container.env is not None:
                # 遍历容器中的环境变量
                for env in container.env:
                    # 如果环境变量来源于秘钥引用
                    if env.value_from is not None and env.value_from.secret_key_ref is not None:
                        # 将挂载信息添加到字符串中，并递增秘钥数量
                        mount_info += '{2}. Environemnt variable name: {0}\n   Secret name: {1}\n'.format(env.name, env.value_from.secret_key_ref.name, secrets_num)
                        secrets_num += 1
                # 如果挂载信息不为空
                if mount_info != '':
                    # 将 pod 名称、命名空间、容器名称和挂载信息添加到表格中
                    t.add_row([pod.metadata.name, pod.metadata.namespace, container.name, mount_info])

    # 打印左对齐的表格
    print_table_aligned_left(t)
# 解析安全上下文，返回格式化后的上下文信息
def parse_security_context(security_context):
    # 初始化标志位，用于标记是否已经设置了头部信息
    is_header_set = False
    # 初始化上下文字符串
    context = ''
    # 如果安全上下文存在
    if security_context:
        # 将安全上下文转换为字典
        dict =  security_context.to_dict()
        # 遍历字典的键
        for key in dict.keys():
            # 如果键对应的值不为空
            if dict[key] is not None:
                # 如果头部信息未设置，则添加头部信息
                if not is_header_set:
                    context += "SecurityContext:\n"
                    is_header_set = True
                # 格式化键值对并添加到上下文字符串中
                context += '  {0}: {1}\n'.format(key, dict[key])
    # 返回格式化后的上下文信息
    return context

# 解析容器规范，返回格式化后的规范信息
def parse_container_spec(container_spec):
    # 初始化规范字符串
    spec = ''
    # 将容器规范转换为字典
    dict =  container_spec.to_dict()
    # 初始化标志位，用于标记是否已经设置了端口头部信息
    is_ports_header_set = False
    # 遍历字典的键
    for key in dict.keys():
        # 如果键对应的值不为空
        if dict[key] is not None:
            # 如果键为'ports'
            if key == 'ports':
                # 如果端口头部信息未设置，则添加端口头部信息
                if not is_ports_header_set:
                    spec += "Ports:\n"
                    is_ports_header_set = True
                # 遍历端口对象列表
                for port_obj in dict[key]:
                    # 如果端口对象中包含'host_port'
                    if 'host_port' in port_obj:
                        # 格式化并添加容器端口和主机端口信息到规范字符串中
                        spec += '  {0}: {1}\n'.format('container_port', port_obj['container_port'])
                        spec += '  {0}: {1}\n'.format('host_port', port_obj['host_port'])
                        break
    # 将容器的安全上下文信息添加到规范字符串中
    spec += parse_security_context(container_spec.security_context)
    # 返回格式化后的规范信息
    return spec

# 解析 Pod 规范，返回格式化后的规范信息
def parse_pod_spec(pod_spec, container):
    # 初始化规范字符串
    spec = ''
    # 将 Pod 规范转换为字典
    dict =  pod_spec.to_dict()
    # 初始化标志位，用于标记是否已经设置了卷头部信息
    is_volumes_header_set = False
    # 遍历字典的键
    for key in dict.keys():
        # 如果字典的值不为空
        if dict[key] is not None:
            # 如果键是 'host_pid'、'host_ipc' 或者 'host_network'
            if key == 'host_pid' or key == 'host_ipc' or key == 'host_network':
                # 将键值对格式化后添加到 spec 变量中
                spec += '{0}: {1}\n'.format(key, dict[key])

            # 如果键是 'volumes' 并且容器的卷挂载不为空
            if key == 'volumes' and container.volume_mounts is not None:
                # 遍历字典中 'volumes' 对应的值
                for volume_obj in dict[key]:
                    # 如果 volume_obj 中包含 'host_path'
                    if 'host_path' in volume_obj:
                        # 如果 'host_path' 不为空
                        if volume_obj['host_path']:
                            # 遍历容器的卷挂载
                            for volume_mount in container.volume_mounts:
                                # 如果 volume_obj 的 'name' 与 volume_mount 的 name 相等
                                if volume_obj['name'] == volume_mount.name:
                                    # 如果卷头部还没有设置
                                    if not is_volumes_header_set:
                                        # 设置卷头部
                                        spec += "Volumes:\n"
                                        is_volumes_header_set = True
                                    # 添加卷的名称到 spec 变量中
                                    spec += '  -{0}: {1}\n'.format('name', volume_obj['name'])
                                    # 添加 host_path 的信息到 spec 变量中
                                    spec += '   host_path:\n'
                                    spec += '     {0}: {1}\n'.format('path', volume_obj['host_path']['path'])
                                    spec += '     {0}: {1}\n'.format('type', volume_obj['host_path']['type'])
                                    spec += '     {0}: {1}\n'.format('container_path', volume_mount.mount_path)

    # 将安全上下文的信息解析后添加到 spec 变量中
    spec += parse_security_context(pod_spec.security_context)
    # 返回 spec 变量
    return spec
# 打印特权容器信息
def print_privileged_containers(namespace=None):
    # 声明全局变量 curr_header
    global curr_header
    # 设置 curr_header 的值为 "|Privileged Containers|"
    curr_header = "|Privileged Containers|"
    # 打印分隔线
    print("+---------------------+")
    # 打印标题
    print("|Privileged Containers|")
    # 创建 PrettyTable 对象 t，设置表头
    t = PrettyTable(['Pod', 'Namespace', 'Pod Spec', 'Container', 'Container info'])
    # 获取特权容器列表
    pods = engine.privleged_containers.get_privileged_containers(namespace)
    # 遍历特权容器列表
    for pod in pods:
        # 遍历容器列表
        for container in pod.spec.containers:
            # 向表中添加行数据
            t.add_row([pod.metadata.name, pod.metadata.namespace, parse_pod_spec(pod.spec, container), container.name, parse_container_spec(container)])
    # 打印格式化对齐的表格
    print_table_aligned_left(t)

# 打印加入集群的令牌信息
def print_join_token():
    # 导入必要的模块
    import os
    from kubernetes.client import Configuration
    # 获取主节点 IP 和端口
    master_ip = Configuration().host.split(':')[1][2:]
    master_port = Configuration().host.split(':')[2]
    # 设置 CA 证书路径
    ca_cert = '/etc/kubernetes/pki/ca.crt'
    # 如果 CA 证书不存在，则使用默认路径
    if not os.path.exists(ca_cert):
        ca_cert = '/etc/kubernetes/ca.crt'
    # 如果在容器中运行，则使用环境变量中的 CA 证书路径
    if running_in_container():
        ca_cert = os.getenv('KUBISCAN_VOLUME_PATH', '/tmp') + ca_cert
    # 设置加入集群的令牌脚本路径
    join_token_path = os.path.dirname(os.path.realpath(__file__)) + '/engine/join_token.sh'
    # 获取解码后的令牌列表
    tokens = engine.utils.list_boostrap_tokens_decoded()
    # 如果没有令牌存在，则打印提示信息
    if not tokens:
        print("No bootstrap tokens exist")
    else:
        # 遍历令牌列表
        for token in tokens:
            # 构建执行命令
            command = 'sh ' + join_token_path + ' ' + ' '.join([master_ip, master_port, ca_cert, token])
            # 打印执行命令
            print('\nExecute: %s' % command)
            # 执行命令
            os.system(command)

# 打印 Logo
def print_logo():
    # 定义 Logo 字符串
    logo = '''
                   `-/osso/-`                    
                `-/osssssssssso/-`                
            .:+ssssssssssssssssssss+:.            
        .:+ssssssssssssssssssssssssssss+:.        
     :osssssssssssssssssssssssssssssssssso:     
    '''
    # 这是一个艺术字的字符串，没有实际的代码含义
    # 可能是用来作为程序的装饰或者标识
    # 但在代码中并没有实际作用
-sss:`//..-`` .`-`-//`.----. //-`-`. ``-..//.:sss-
osss:.::`...`- ..`.:/`+ssss+`/:``.. -`...`::.:ssso
+ssso`:/:`--`:`--`/:-`ssssss`-//`--`:`--`:/:`osss+
 :sss+`-//.`...`-//..osssssso..//-`...`.//-`+sss: 
  `+sss/...::/::..-+ssssssssss+-..::/::.../sss+`  
    -ossss+/:::/+ssssssssssssssss+/:::/+sssso-    
      :ssssssssssssssssssssssssssssssssssss/      
       `+ssssssssssssssssssssssssssssssss+`       
         -osssssssssssssssssssssssssssss-         
          `/ssssssssssssssssssssssssss/`       
    
               KubiScan version 1.7
               Author: Eviatar Gerzi
    '''
    # 打印 KubiScan 版本和作者信息
    print(logo)

def print_examples():
    # 导入 os 模块
    import os
    # 打开 examples 文件夹下的 examples.txt 文件，并打印其中的内容
    with open(os.path.dirname(os.path.realpath(__file__)) + '/examples/examples.txt', 'r') as f:
        print(f.read())

def main():
    # 创建 ArgumentParser 对象，用于解析命令行参数
    opt = ArgumentParser(description='KubiScan.py - script used to get information on risky permissions on Kubernetes', usage="""KubiScan.py [options...]

This tool can get information about risky roles\clusterroles, rolebindings\clusterrolebindings, users and pods.
Use "KubiScan.py -h" for help or "KubiScan.py -e" to see examples.
Requirements:
    - Python 3
    - Kubernetes python client (https://github.com/kubernetes-client/python) 
      Can be installed:
            From source:
                git clone --recursive https://github.com/kubernetes-client/python.git
                cd python
                python setup.py install
            From PyPi directly:
                pip3 install kubernetes
    - Prettytable
        pip3 install PTable
    """)
    # 添加命令行参数选项
    opt.add_argument('-rr', '--risky-roles', action='store_true', help='Get all risky Roles (can be used with -r to view rules)', required=False)
    opt.add_argument('-rcr', '--risky-clusterroles', action='store_true', help='Get all risky ClusterRoles (can be used with -r to view rules)',required=False)
    opt.add_argument('-rar', '--risky-any-roles', action='store_true', help='Get all risky Roles and ClusterRoles', required=False)
    # 添加命令行参数 -rb/--risky-rolebindings，设置为True时获取所有风险的RoleBindings
    opt.add_argument('-rb', '--risky-rolebindings', action='store_true', help='Get all risky RoleBindings', required=False)
    # 添加命令行参数 -rcb/--risky-clusterrolebindings，设置为True时获取所有风险的ClusterRoleBindings
    opt.add_argument('-rcb', '--risky-clusterrolebindings', action='store_true',help='Get all risky ClusterRoleBindings', required=False)
    # 添加命令行参数 -rab/--risky-any-rolebindings，设置为True时获取所有风险的RoleBindings和ClusterRoleBindings
    opt.add_argument('-rab', '--risky-any-rolebindings', action='store_true', help='Get all risky RoleBindings and ClusterRoleBindings', required=False)
    
    # 添加命令行参数 -rs/--risky-subjects，设置为True时获取所有风险的Subjects（用户、组或服务账户）
    opt.add_argument('-rs', '--risky-subjects', action='store_true',help='Get all risky Subjects (Users, Groups or Service Accounts)', required=False)
    # 添加命令行参数 -rp/--risky-pods，设置为True时获取所有风险的Pods/Containers
    # 使用 -d/--deep 开关从当前运行的容器中读取令牌
    opt.add_argument('-rp', '--risky-pods', action='store_true', help='Get all risky Pods\Containers.\n'
                                                                          'Use the -d\--deep switch to read the tokens from the current running containers', required=False)
    # 添加命令行参数 -d/--deep，设置为True时仅与 -rp/--risky-pods 开关一起使用，如果指定了此参数，将执行每个Pod以获取其令牌
    # 如果没有指定此参数，将从ETCD中读取挂载的服务账户密钥，这种方法不太可靠但速度更快
    opt.add_argument('-d', '--deep', action='store_true', help='Works only with -rp\--risky-pods switch. If this is specified, it will execute each pod to get its token.\n'
                                                                   'Without it, it will read the pod mounted service account secret from the ETCD, it less reliable but much faster.', required=False)
    # 添加命令行参数 -pp/--privleged-pods，设置为True时获取所有特权的Pods/Containers
    opt.add_argument('-pp', '--privleged-pods', action='store_true', help='Get all privileged Pods\Containers.',  required=False)
    # 添加命令行参数 -a/--all，设置为True时获取所有风险的Roles/ClusterRoles、RoleBindings/ClusterRoleBindings、用户和Pods/Containers
    opt.add_argument('-a', '--all', action='store_true',help='Get all risky Roles\ClusterRoles, RoleBindings\ClusterRoleBindings, users and pods\containers', required=False)
    # 添加命令行参数 -cve/--cve，设置为True时进行CVE扫描
    opt.add_argument('-cve', '--cve', action='store_true', help=f"Scan of CVE's", required=False)
    # 添加命令行参数 -jt/--join-token，设置为True时获取集群的加入令牌。必须安装OpenSsl和kubeadm
    opt.add_argument('-jt', '--join-token', action='store_true', help='Get join token for the cluster. OpenSsl must be installed + kubeadm', required=False)
    # 添加命令行参数 -psv/--pods-secrets-volume，设置为True时显示所有具有通过Volume访问秘密数据的Pods
    opt.add_argument('-psv', '--pods-secrets-volume', action='store_true', help='Show all pods with access to secret data throught a Volume', required=False)
    # 添加命令行参数，用于显示所有具有通过环境变量访问秘密数据的 pod
    opt.add_argument('-pse', '--pods-secrets-env', action='store_true', help='Show all pods with access to secret data throught a environment variables', required=False)
    # 添加命令行参数，用于指定要运行的上下文。如果没有指定，则在当前上下文中运行
    opt.add_argument('-ctx', '--context', action='store', help='Context to run. If none, it will run in the current context.', required=False)
    # 添加命令行参数，用于按优先级过滤（CRITICAL\HIGH\LOW）
    opt.add_argument('-p', '--priority', action='store', help='Filter by priority (CRITICAL\HIGH\LOW)', required=False)

    # 创建一个辅助开关组
    helper_switches = opt.add_argument_group('Helper switches')
    # 添加命令行参数，用于过滤存在时间少于 X 天的对象
    helper_switches.add_argument('-lt', '--less-than', action='store', metavar='NUMBER', help='Used to filter object exist less than X days.\nSupported on Roles\ClusterRoles and RoleBindings\ClusterRoleBindings.'
                                                                                              'IMPORTANT: If object does not have creation time (usually in ClusterRoleBindings), omit this switch to see it.', required=False)
    # 添加命令行参数，用于指定要使用的命名空间范围
    helper_switches.add_argument('-ns', '--namespace', action='store', help='If present, the namespace scope that will be used', required=False)
    # 添加命令行参数，用于指定对象的类型
    helper_switches.add_argument('-k', '--kind', action='store', help='Kind of the object', required=False)
    # 添加命令行参数，用于显示规则。仅在打印风险的 Roles\ClusterRoles 上支持
    helper_switches.add_argument('-r', '--rules', action='store_true', help='Show rules. Supported only on pinrting risky Roles\ClusterRoles.', required=False)
    # 添加命令行参数，用于显示示例
    helper_switches.add_argument('-e', '--examples', action='store_true', help='Show examples.', required=False)
    # 添加命令行参数，用于指定名称
    helper_switches.add_argument('-n', '--name', action='store', help='Name', required=False)
    # 创建一个用于转储令牌的参数组，描述为“使用开关：名称（-n\--name）或命名空间（-ns\ --namespace）”
    dumping_tokens = opt.add_argument_group('Dumping tokens', description='Use the switches: name (-n\--name) or namespace (-ns\ --namespace)')
    # 添加命令行参数 -dt/--dump-tokens，设置为True表示需要dump tokens，带有帮助信息
    dumping_tokens.add_argument('-dt', '--dump-tokens', action='store_true', help='Dump tokens from pod\pods\n'
                                                                                  'Example: -dt OR -dt -ns \"kube-system\"\n'
                                                                                  '-dt -n \"nginx1\" -ns \"default\"', required=False)

    # 创建一个命令行参数组，用于存放远程开关相关的参数
    helper_switches = opt.add_argument_group('Remote switches')
    # 添加命令行参数 -ho/--host，用于指定包含主节点IP和端口的主机，带有帮助信息
    helper_switches.add_argument('-ho', '--host', action='store', metavar='<MASTER_IP>:<PORT>', help='Host contain the master ip and port.\n'
                                                                                                     'For example: 10.0.0.1:6443', required=False)
    # 添加命令行参数 -c/--cert-filename，用于指定证书授权路径，带有帮助信息
    helper_switches.add_argument('-c', '--cert-filename', action='store', metavar='CA_FILENAME', help='Certificate authority path (\'/../ca.crt\'). If not specified it will try without SSL verification.\n'
                                                                            'Inside Pods the default location is \'/var/run/secrets/kubernetes.io/serviceaccount/ca.crt\''
                                                                            'Or \'/run/secrets/kubernetes.io/serviceaccount/ca.crt\'.', required=False)
    # 添加命令行参数 -cc/--client-certificate，用于指定客户端密钥文件路径
    helper_switches.add_argument('-cc', '--client-certificate', action='store', metavar='CA_FILENAME',
                                 help='Path to client key file', required=False)
    # 添加命令行参数 -ck/--client-key，用于指定客户端证书文件路径
    helper_switches.add_argument('-ck', '--client-key', action='store', metavar='CA_FILENAME',
                                 help='Path to client certificate file', required=False)
    # 添加命令行参数 -co/--kube-config，用于指定kube配置文件路径，带有帮助信息
    helper_switches.add_argument('-co', '--kube-config', action='store', metavar='KUBE_CONFIG_FILENAME',
                                 help='The kube config file.\n'
                                      'For example: ~/.kube/config', required=False)
    # 添加命令行参数 -t 或 --token-filename，用于指定令牌文件名，存储令牌信息
    # 包含帮助信息，描述令牌的作用和所需的最小权限
    helper_switches.add_argument('-t', '--token-filename', action='store', metavar='TOKEN_FILENAME',
                                 help='A bearer token. If this token does not have the required permissions for this application,'
                                      'the application will faill to get some of the information.\n'
                                      'Minimum required permissions:\n'
                                      '- resources: [\"roles\", \"clusterroles\", \"rolebindings\", \"clusterrolebindings\", \"pods\", \"secrets\"]\n'
                                      '  verbs: [\"get\", \"list\"]\n'
                                      '- resources: [\"pods/exec\"]\n'
                                      '  verbs: [\"create\"]')
    # 添加命令行参数 -o 或 --output-file，用于指定输出文件的路径
    helper_switches.add_argument('-o', '--output-file', metavar='OUTPUT_FILENAME', help='Path to output file')
    # 添加命令行参数 -q 或 --quiet，用于隐藏横幅信息
    helper_switches.add_argument('-q', '--quiet', action='store_true', help='Hide the banner')
    # 添加命令行参数 -j 或 --json，用于导出到 JSON 文件
    helper_switches.add_argument('-j', '--json', metavar='JSON_FILENAME', help='Export to json')
    # 添加命令行参数 -nc 或 --no-color，用于在打印时不显示颜色
    helper_switches.add_argument('-nc', '--no-color', action='store_true', help='Print without color')
    # 创建一个参数组 associated_rb_crb_to_role，用于关联 RoleBindings\ClusterRoleBindings 到 Role
    associated_rb_crb_to_role = opt.add_argument_group('Associated RoleBindings\ClusterRoleBindings to Role', description='Use the switch: namespace (-ns\--namespace).')
    # 在参数组 associated_rb_crb_to_role 中添加命令行参数 -aarbr，用于获取关联到特定角色的 RoleBindings\ClusterRoleBindings
    associated_rb_crb_to_role.add_argument('-aarbr', '--associated-any-rolebindings-role', action='store', metavar='ROLE_NAME',
                                           help='Get associated RoleBindings\ClusterRoleBindings to a specific role\n'
                                                'Example: -aarbr \"read-secrets-role\" -ns \"default\"', required=False)
    # 创建一个参数组 associated_rb_crb_to_clusterrole，用于关联 RoleBindings\ClusterRoleBindings 到 ClusterRole
    associated_rb_crb_to_clusterrole = opt.add_argument_group('Associated RoleBindings\ClusterRoleBindings to ClusterRole')
    # 添加参数 -aarbcr/--associated-any-rolebindings-clusterrole，设置动作为存储，设置参数的元变量为CLUSTERROLE_NAME
    # 帮助信息为获取与特定角色关联的RoleBindings\ClusterRoleBindings
    # 例如：-aarbcr "read-secrets-clusterrole"
    # 参数为可选
    associated_rb_crb_to_clusterrole.add_argument('-aarbcr', '--associated-any-rolebindings-clusterrole', action='store', metavar='CLUSTERROLE_NAME',
                                                  help='Get associated RoleBindings\ClusterRoleBindings to a specific role\n'
                                                       'Example:  -aarbcr \"read-secrets-clusterrole\"', required=False)

    # 创建关联到主体（用户、组或服务账户）的RoleBindings\ClusterRoleBindings的参数组
    # 描述信息为使用开关：namespace (-ns\--namespace) 和 kind (-k\--kind)
    associated_rb_crb_to_subject = opt.add_argument_group('Associated RoleBindings\ClusterRoleBindings to Subject (user, group or service account)',
                                                           description='Use the switches: namespace (-ns\--namespace) and kind (-k\--kind).\n')
    # 添加参数 -aarbs/--associated-any-rolebindings-subject，设置动作为存储，设置参数的元变量为SUBJECT_NAME
    # 帮助信息为获取与特定主体（用户、组或服务账户）关联的Rolebindings\ClusterRoleBindings
    # 例如：-aarbs "system:masters" -k "Group"
    # 参数为可选
    associated_rb_crb_to_subject.add_argument('-aarbs', '--associated-any-rolebindings-subject', action='store', metavar='SUBJECT_NAME',
                                              help='Get associated Rolebindings\ClusterRoleBindings to a specific Subject (user, group or service account)\n'
                                                   'Example: -aarbs \"system:masters\" -k \"Group\"', required=False)

    # 创建关联到主体（用户、组或服务账户）的Roles\ClusterRoles的参数组
    # 描述信息为使用开关：namespace (-ns\--namespace) 和 kind (-k\--kind)
    associated_rb_crb_to_subject = opt.add_argument_group('Associated Roles\ClusterRoles to Subject (user, group or service account)',
                                                           description='Use the switches: namespace (-ns\--namespace) and kind (-k\--kind).\n')
    # 添加参数 -aars/--associated-any-roles-subject，设置动作为存储，设置参数的元变量为SUBJECT_NAME
    # 帮助信息为获取与特定主体（用户、组或服务账户）关联的Roles\ClusterRoles
    # 例如：-aars "generic-garbage-collector" -k "ServiceAccount" -ns "kube-system"
    # 参数为可选
    associated_rb_crb_to_subject.add_argument('-aars', '--associated-any-roles-subject', action='store', metavar='SUBJECT_NAME',
                                              help='Get associated Roles\ClusterRoles to a specific Subject (user, group or service account)\n'
                                                   'Example: -aars \"generic-garbage-collector\" -k \"ServiceAccount\" -ns \"kube-system\"', required=False)

    # 创建列出主体的参数组
    list_subjects = opt.add_argument_group('List Subjects')
    # 添加命令行参数，用于获取带有 User 类型的 Subjects
    list_subjects.add_argument('-su', '--subject-users', action='store_true', help='Get Subjects with User kind', required=False)
    # 添加命令行参数，用于获取带有 Group 类型的 Subjects
    list_subjects.add_argument('-sg', '--subject-groups', action='store_true', help='Get Subjects with Group kind', required=False)
    # 添加命令行参数，用于获取带有 ServiceAccount 类型的 Subjects
    list_subjects.add_argument('-ss', '--subject-serviceaccounts', action='store_true', help='Get Subjects with ServiceAccount kind', required=False)

    # 创建一个参数组，用于列出 RoleBinding\ClusterRoleBinding 的规则
    list_rules = opt.add_argument_group('List rules of RoleBinding\ClusterRoleBinding')
    # 添加命令行参数，用于获取 RoleBinding 的规则
    list_rules.add_argument('-rru', '--rolebinding-rules', action='store', metavar='ROLEBINDING_NAME', help='Get rules of RoleBinding', required=False)
    # 添加命令行参数，用于获取 ClusterRoleBinding 的规则
    list_rules.add_argument('-crru', '--clusterrolebinding-rules', action='store', metavar='CLUSTERROLEBINDING_NAME',  help='Get rules of ClusterRoleBinding',required=False)

    # 解析命令行参数
    args = opt.parse_args()
    # 如果设置了 no_color 参数，则将全局变量 no_color 设置为 True
    if args.no_color:
        global no_color
        no_color = True
    # 如果设置了 json 参数，则将全局变量 json_filename 设置为参数值
    if args.json:
        global json_filename
        json_filename = args.json
    # 如果设置了 output_file 参数，则打开文件并将标准输出重定向到该文件
    if args.output_file:
        f = open(args.output_file, 'w')
        sys.stdout = f
    # 如果没有设置 quiet 参数，则打印 logo
    if not args.quiet:
        print_logo()

    # 如果设置了 examples 参数，则打印示例并退出程序
    if args.examples:
        print_examples()
        exit()

    # 初始化 API 连接
    api_init(kube_config_file=args.kube_config, host=args.host, token_filename=args.token_filename, cert_filename=args.cert_filename, context=args.context)

    # 如果设置了 cve 参数，则打印 CVE 信息
    if args.cve:
        print_cve(args.cert_filename, args.client_certificate, args.client_key, args.host)
    # 如果设置了 risky_roles 参数，则打印风险角色信息
    if args.risky_roles:
        print_risky_roles(show_rules=args.rules, days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果设置了 risky_clusterroles 参数，则打印风险集群角色信息
    if args.risky_clusterroles:
        print_risky_clusterroles(show_rules=args.rules, days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果设置了 risky_any_roles 参数，则打印所有风险角色信息
    if args.risky_any_roles:
        print_all_risky_roles(show_rules=args.rules, days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果参数中包含需要打印风险角色绑定的选项，则执行相应的函数并传入参数
    if args.risky_rolebindings:
        print_risky_rolebindings(days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果参数中包含需要打印风险集群角色绑定的选项，则执行相应的函数并传入参数
    if args.risky_clusterrolebindings:
        print_risky_clusterrolebindings(days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果参数中包含需要打印所有风险角色绑定的选项，则执行相应的函数并传入参数
    if args.risky_any_rolebindings:
        print_all_risky_rolebindings(days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果参数中包含需要打印所有风险主体的选项，则执行相应的函数并传入参数
    if args.risky_subjects:
        print_all_risky_subjects(priority=args.priority, namespace=args.namespace)
    # 如果参数中包含需要打印所有风险容器的选项，则执行相应的函数并传入参数
    if args.risky_pods:
        print_all_risky_containers(priority=args.priority, namespace=args.namespace, read_token_from_container=args.deep)
    # 如果参数中包含需要打印所有信息的选项，则执行相应的函数并传入参数
    if args.all:
        print_all(days=args.less_than, priority=args.priority, read_token_from_container=args.deep)
    # 如果参数中包含需要打印特权容器的选项，则执行相应的函数并传入参数
    elif args.privleged_pods:
        print_privileged_containers(namespace=args.namespace)
    # 如果参数中包含需要打印加入令牌的选项，则执行相应的函数
    elif args.join_token:
        print_join_token()
    # 如果参数中包含需要打印使用卷的容器的选项，则执行相应的函数并传入参数
    elif args.pods_secrets_volume:
        # 如果参数中包含命名空间，则执行相应的函数并传入参数
        if args.namespace:
            print_pods_with_access_secret_via_volumes(namespace=args.namespace)
        # 否则执行相应的函数
        else:
            print_pods_with_access_secret_via_volumes()
    # 如果参数中包含需要打印使用环境变量的容器的选项，则执行相应的函数并传入参数
    elif args.pods_secrets_env:
        # 如果参数中包含命名空间，则执行相应的函数并传入参数
        if args.namespace:
            print_pods_with_access_secret_via_environment(namespace=args.namespace)
        # 否则执行相应的函数
        else:
            print_pods_with_access_secret_via_environment()
    # 如果参数中包含需要打印与任何角色绑定相关的角色的选项，则执行相应的函数并传入参数
    elif args.associated_any_rolebindings_role:
        # 如果参数中包含命名空间，则执行相应的函数并传入参数
        if args.namespace:
            print_associated_rolebindings_to_role(args.associated_any_rolebindings_role, args.namespace)
    # 如果参数中包含需要打印与任何角色绑定相关的集群角色的选项，则执行相应的函数并传入参数
    elif args.associated_any_rolebindings_clusterrole:
        print_associated_any_rolebindings_to_clusterrole(args.associated_any_rolebindings_clusterrole)
    # 如果参数中包含了与任何角色绑定的主体
    elif args.associated_any_rolebindings_subject:
        # 如果参数中包含了资源类型
        if args.kind:
            # 如果资源类型是服务账户
            if args.kind == constants.SERVICEACCOUNT_KIND:
                # 如果参数中包含了命名空间
                if args.namespace:
                    # 打印与主体相关的角色绑定和集群角色绑定
                    print_associated_rolebindings_and_clusterrolebindings_to_subject(args.associated_any_rolebindings_subject, args.kind, args.namespace)
                else:
                    # 提示需要指定服务账户类型的命名空间
                    print('For ServiceAccount kind specify namespace (-ns, --namespace)')
            else:
                # 打印与主体相关的角色绑定和集群角色绑定
                print_associated_rolebindings_and_clusterrolebindings_to_subject(args.associated_any_rolebindings_subject, args.kind)
        else:
            # 提示未指定主体命名空间或类型
            print('Subject namespace (-ns, --namespace) or kind (-k, --kind) was not specificed')
    # 如果参数中包含了与任何角色相关的主体
    elif args.associated_any_roles_subject:
        # 如果参数中包含了资源类型
        if args.kind:
            # 如果资源类型是服务账户
            if args.kind == constants.SERVICEACCOUNT_KIND:
                # 如果参数中包含了命名空间
                if args.namespace:
                    # 打印与主体相关的规则
                    print_rules_associated_to_subject(args.associated_any_roles_subject, args.kind, args.namespace)
                else:
                    # 提示需要指定服务账户类型的命名空间
                    print('For ServiceAccount kind specify namespace (-ns, --namespace)')
            else:
                # 打印与主体相关的规则
                print_rules_associated_to_subject(args.associated_any_roles_subject, args.kind)
        else:
            # 提示需要指定类型
            print("Please specify kind (-k, --kind).")
    # 如果参数中包含了转储令牌
    elif args.dump_tokens:
        # 如果参数中包含了名称
        if args.name:
            # 如果参数中包含了命名空间
            if args.namespace:
                # 从 Pod 中转储令牌
                dump_tokens_from_pods(pod_name=args.name, namespace=args.namespace, read_token_from_container=args.deep)
            else:
                # 提示在指定 Pod 名称时，需要同时指定命名空间
                print('When specificing Pod name, need also namespace')
        # 如果参数中包含了命名空间
        elif args.namespace:
            # 从 Pod 中转储令牌
            dump_tokens_from_pods(namespace=args.namespace, read_token_from_container=args.deep)
        else:
            # 从 Pod 中转储令牌
            dump_tokens_from_pods(read_token_from_container=args.deep)
    # 如果参数中包含了主体用户
    elif args.subject_users:
        # 打印指定类型的主体
        print_subjects_by_kind(constants.USER_KIND)
    # 如果参数中包含了主体组
    elif args.subject_groups:
        # 打印指定类型的主体
        print_subjects_by_kind(constants.GROUP_KIND)
    # 如果参数中包含 subject_serviceaccounts，则打印指定类型的主体
    elif args.subject_serviceaccounts:
        print_subjects_by_kind(constants.SERVICEACCOUNT_KIND)
    # 如果参数中包含 rolebinding_rules，则根据命名空间打印角色绑定规则
    elif args.rolebinding_rules:
        # 如果参数中包含命名空间，则打印指定命名空间的角色绑定规则
        if args.namespace:
            print_rolebinding_rules(args.rolebinding_rules, args.namespace)
        # 如果参数中没有指定命名空间，则打印错误信息
        else:
            print("Namespace was not specified")
    # 如果参数中包含 clusterrolebinding_rules，则打印集群角色绑定规则
    elif args.clusterrolebinding_rules:
        print_clusterrolebinding_rules(args.clusterrolebinding_rules)
# 定义一个函数，用于将表格左对齐打印出来
def print_table_aligned_left(table):
    # 声明全局变量 json_filename
    global json_filename
    # 如果 json_filename 不为空，则将表格导出为 JSON 文件
    if json_filename != "":
        export_to_json(table, json_filename)
    # 声明全局变量 output_file
    global output_file
    # 如果设置了 no_color 标志，则去除表格中的 ANSI 颜色码
    if no_color:
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        for row in table._rows:
            row[0] = ansi_escape.sub('', row[0])

    # 设置表格对齐方式为左对齐
    table.align = 'l'
    # 打印表格
    print(table)
    print('\n')


# 定义一个函数，用于将表格导出为 JSON 文件
def export_to_json(table, json_filename):
    # 声明全局变量 curr_header
    global curr_header
    # 去除表格中的 ANSI 颜色码
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    # 获取表格的列名
    headers = table.field_names
    # 去除 curr_header 中的竖线字符
    curr_header = curr_header.replace("|", "")
    # 创建一个空的数据字典
    data = {curr_header: []}
    try:
        # 尝试打开 JSON 文件，读取其内容
        with open(json_filename, "r") as json_file:
            json_file_content = json_file.read()
    except:
        # 如果文件不存在，则将 json_file_content 设置为空字符串
        json_file_content = ""

    # 如果 JSON 文件内容为空，则将 res 设置为一个空列表，否则解析 JSON 文件内容
    res = [] if json_file_content == "" else json.loads(json_file_content)
    # 以读写方式打开 JSON 文件
    json_file = open(json_filename, "w+")
    # 遍历表格的每一行
    for row_index, row in enumerate(table._rows):
        curr_item = {}
        # 遍历每一行的每个单元格
        for i, entity in enumerate(row):
            # 如果列名为 'Priority'，则去除 ANSI 颜色码
            entity_without_color = ansi_escape.sub('', entity) if headers[i] == 'Priority' else entity
            # 将单元格数据添加到 curr_item 中
            curr_item[headers[i]] = entity_without_color

        # 将 curr_item 添加到数据字典中
        data[curr_header].append(curr_item)
    # 将数据字典添加到 res 中
    res.append(data)
    # 将 res 转换为 JSON 格式并写入 JSON 文件
    json_file.write(json.dumps(res, indent=2))
    json_file.flush()
    json_file.close()


# 如果当前脚本为主程序，则执行 main() 函数
if __name__ == '__main__':
    main()
```