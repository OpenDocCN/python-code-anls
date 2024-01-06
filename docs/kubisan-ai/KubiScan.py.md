# `KubiScan\KubiScan.py`

```
# 导入所需的模块
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关的模块
from argparse import ArgumentParser  # 从 argparse 模块中导入 ArgumentParser 类
import engine.utils  # 导入 engine.utils 模块
import engine.privleged_containers  # 导入 engine.privleged_containers 模块
from prettytable import PrettyTable, ALL  # 从 prettytable 模块中导入 PrettyTable 类和 ALL 常量
from engine.priority import Priority  # 从 engine.priority 模块中导入 Priority 类
from misc.colours import *  # 从 misc.colours 模块中导入所有颜色
from misc import constants  # 导入 misc 模块中的 constants
import datetime  # 导入处理日期和时间的模块
from api.api_client import api_init, running_in_container  # 从 api.api_client 模块中导入 api_init 和 running_in_container 函数

# 初始化变量
json_filename = ""  # JSON 文件名
output_file = ""  # 输出文件名
no_color = False  # 是否禁用颜色

# 根据优先级获取对应的颜色
def get_color_by_priority(priority):
    color = WHITE  # 默认颜色为白色
    if priority == Priority.CRITICAL:  # 如果优先级为 CRITICAL
    # 如果优先级是低，则颜色为红色
    color = RED
    # 如果优先级是高，则颜色为浅黄色
    elif priority == Priority.HIGH:
        color = LIGHTYELLOW
    # 返回颜色
    return color

# 根据给定天数过滤对象
def filter_objects_less_than_days(days, objects):
    # 创建一个空的过滤对象列表
    filtered_objects= []
    # 获取当前日期时间
    current_datetime = datetime.datetime.now()
    # 遍历对象列表
    for object in objects:
        # 如果对象有时间属性
        if object.time:
            # 如果当前日期减去对象时间的日期小于给定天数
            if (current_datetime.date() - object.time.date()).days < days:
                # 将对象添加到过滤对象列表中
                filtered_objects.append(object)
    # 将原对象列表替换为过滤后的对象列表
    objects = filtered_objects
    # 返回过滤后的对象列表
    return objects

# 根据优先级过滤对象
def filter_objects_by_priority(priority, objects):
    # 创建一个空的过滤对象列表
    filtered_objects= []
    # 遍历对象列表
    for object in objects:
# 如果对象的优先级与给定的优先级相同，则将对象添加到过滤后的对象列表中
if object.priority.name == priority.upper():
    filtered_objects.append(object)
# 将过滤后的对象列表赋值给对象列表变量
objects = filtered_objects
# 返回过滤后的对象列表
return objects

# 计算给定日期距离当前日期的天数差
def get_delta_days_from_now(date):
    # 获取当前日期时间
    current_datetime = datetime.datetime.now()
    # 返回日期差的天数
    return (current_datetime.date() - date.date()).days

# 打印所有风险角色
def print_all_risky_roles(show_rules=False, days=None, priority=None, namespace=None):
    # 获取所有风险角色和集群角色
    risky_any_roles = engine.utils.get_risky_roles_and_clusterroles()
    # 如果指定了命名空间，则发出警告
    if namespace is not None:
        logging.warning("'-rar' switch does not expect namespace ('-ns')\n")
    # 如果指定了天数，则过滤出指定天数内的对象
    if days:
        risky_any_roles = filter_objects_less_than_days(int(days), risky_any_roles)
    # 如果指定了优先级，则根据优先级过滤对象
    if priority:
        risky_any_roles = filter_objects_by_priority(priority, risky_any_roles)
    # 打印所有风险角色和集群角色
    generic_print('|Risky Roles and ClusterRoles|', risky_any_roles, show_rules)
# 打印风险角色信息
def print_risky_roles(show_rules=False, days=None, priority=None, namespace=None):
    # 获取所有风险角色
    risky_roles = engine.utils.get_risky_roles()

    # 如果指定了天数，则筛选出指定天数内的风险角色
    if days:
        risky_roles = filter_objects_less_than_days(int(days), risky_roles)
    # 如果指定了优先级，则筛选出指定优先级的风险角色
    if priority:
        risky_roles = filter_objects_by_priority(priority, risky_roles)

    # 存储筛选后的风险角色
    filtered_risky_roles = []
    # 如果未指定命名空间，则打印所有风险角色信息
    if namespace is None:
        generic_print('|Risky Roles |', risky_roles, show_rules)
    # 如果指定了命名空间，则筛选出该命名空间下的风险角色并打印信息
    else:
        for risky_role in risky_roles:
            if risky_role.namespace == namespace:
                filtered_risky_roles.append(risky_role)
        generic_print('|Risky Roles |', filtered_risky_roles, show_rules)

# 打印 CVE 信息
def print_cve(certificate_authority_file=None, client_certificate_file=None, client_key_file=None, host=None):
    # 获取当前 Kubernetes 版本信息
    current_k8s_version = engine.utils.get_current_version(certificate_authority_file, client_certificate_file, client_key_file, host)
# 如果当前的 Kubernetes 版本为空，则直接返回
if current_k8s_version is None:
    return
# 根据当前的 Kubernetes 版本获取受影响的 CVE 表格
cve_table = get_all_affecting_cves_table_by_version(current_k8s_version)
# 打印对齐左边的 CVE 表格
print_table_aligned_left(cve_table)

# 根据当前的 Kubernetes 版本获取受影响的所有 CVE 表格
def get_all_affecting_cves_table_by_version(current_k8s_version):
    # 创建一个包含列名的漂亮表格
    cve_table = PrettyTable(['Severity', 'CVE Grade', 'CVE', 'Description', 'FixedVersions'])
    # 设置表格的水平线规则
    cve_table.hrules = ALL
    # 打开 CVE.json 文件并读取其中的数据
    with open('CVE.json', 'r') as f:
        data = json.load(f)
    # 获取所有的 CVE
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
            # 向表格中添加一行数据
            cve_table.add_row([cve_color + cve['Severity'] + WHITE, cve['Grade'], cve['CVENumber'], cve_description,
                       fixed_version_list])
    # 按照 CVE 等级排序表格
    cve_table.sortby = "CVE Grade"
# 设置 cve_table 的排序方式为倒序
cve_table.reversesort = True
# 返回 cve_table
return cve_table

# 根据 cve_severity 返回对应的颜色
def get_cve_color(cve_severity):
    if cve_severity == "Low":
        return WHITE
    elif cve_severity == "Medium":
        return LIGHTYELLOW
    elif cve_severity == "High" or cve_severity == "Critical":
        return RED

# 获取 cve_fixed_versions 的修复版本信息
def get_fixed_versions_of_cve(cve_fixed_versions):
    fixed_version_list = ""
    for fixed_version in cve_fixed_versions:
        fixed_version_list += fixed_version["Raw"] + "\n"
    return fixed_version_list[:-1]

# 将 cve_description 拆分成单词列表
def split_cve_description(cve_description):
    words = cve_description.split()
# 初始化结果描述为空字符串
res_description = ""
# 每行显示的单词数量
words_in_row = 10
# 遍历单词列表
for i, word in enumerate(words):
    # 当 i 能被 words_in_row 整除且不为 0 时，添加换行符
    if i % words_in_row == 0 and i != 0:
        res_description += "\n"
    # 添加单词和空格到结果描述
    res_description += word + " "
# 返回结果描述，去掉最后一个空格
return res_description[:-1]

# 判断当前 Kubernetes 版本是否受影响
def curr_version_is_affected(cve, current_k8s_version):
    # 获取 CVE 的最大修复版本和最小修复版本
    max_fixed_version = find_max_fixed_version(cve)
    min_fixed_version = find_min_fixed_version(cve)
    # 如果当前版本小于最小修复版本，则受影响
    if compare_versions(current_k8s_version, min_fixed_version) == -1:
        return True
    # 如果当前版本大于等于最大修复版本，则不受影响
    if compare_versions(current_k8s_version, max_fixed_version) >= 0:
        return False
    # 遍历 CVE 的修复版本列表
    for fixed_version in cve['FixedVersions']:
        # 判断当前版本是否在修复版本的中间
        if is_vulnerable_in_middle(current_k8s_version, fixed_version['Raw']):
            # 如果是，则受影响
            return True
# 如果当前 Kubernetes 版本在修复版本之间，则返回 True，否则返回 False
def is_vulnerable_in_middle(current_k8s_version, cve_fixed_version):
    # 将当前 Kubernetes 版本和修复版本分割成数字列表
    current_k8s_version_nums = [int(num) for num in current_k8s_version.split('.')]
    fixed_version_nums = [int(num) for num in cve_fixed_version.split('.')]
    # 检查修复版本的主要版本号和次要版本号是否与当前版本相同，如果是，则比较修复版本的修订号和当前版本的修订号
    if fixed_version_nums[0] == current_k8s_version_nums[0] and fixed_version_nums[1] == current_k8s_version_nums[1]:
        if fixed_version_nums[2] > current_k8s_version_nums[2]:
            return True
    return False


# 比较两个版本号的大小
# 如果 version1 > version2 返回 1
# 如果 version1 < version2 返回 -1
# 如果 version1 = version2 返回 0
def compare_versions(version1, version2):
    # 将版本号分割成数字列表
    version1_nums = [int(num) for num in version1.split('.')]
    version2_nums = [int(num) for num in version2.split('.')]
# 遍历 version2_nums 列表，比较对应位置的数字大小，返回-1、1或0
for i in range(len(version2_nums)):
    if version2_nums[i] > version1_nums[i]:
        return -1
    elif version2_nums[i] < version1_nums[i]:
        return 1
else:
    return 0

# 查找 cve 中的最大修复版本号
def find_max_fixed_version(cve):
    versions = []
    # 遍历 cve 中的 FixedVersions 列表，将 Raw 字段的值添加到 versions 列表中
    for fixed_version in cve['FixedVersions']:
        versions.append(fixed_version['Raw'])
    # 使用 max 函数找到 versions 列表中的最大值，以 . 分割后转换为整数进行比较
    max_version = max(versions, key=lambda x: [int(num) for num in x.split('.')])
    return max_version

# 查找 cve 中的最小修复版本号
def find_min_fixed_version(cve):
    versions = []
    # 遍历 cve 中的 FixedVersions 列表，将 Raw 字段的值添加到 versions 列表中
    for fixed_version in cve['FixedVersions']:
        versions.append(fixed_version['Raw'])
# 找到版本列表中最小的版本号
min_version = min(versions, key=lambda x: [int(num) for num in x.split('.')])
# 返回最小的版本号
return min_version

# 打印有风险的集群角色
def print_risky_clusterroles(show_rules=False, days=None, priority=None, namespace=None):
    # 如果指定了命名空间，则发出警告
    if namespace is not None:
        logging.warning("'-rcr' switch does not expect namespace ('-ns')\n")
    # 获取有风险的集群角色
    risky_clusterroles = engine.utils.get_risky_clusterroles()
    # 如果指定了天数，则过滤出指定天数内的角色
    if days:
        risky_clusterroles = filter_objects_less_than_days(int(days), risky_clusterroles)
    # 如果指定了优先级，则根据优先级过滤角色
    if priority:
        risky_clusterroles = filter_objects_by_priority(priority, risky_clusterroles)
    # 打印有风险的集群角色
    generic_print('|Risky ClusterRoles |', risky_clusterroles, show_rules)

# 打印所有有风险的角色绑定
def print_all_risky_rolebindings(days=None, priority=None, namespace=None):
    # 如果指定了命名空间，则发出警告
    if namespace is not None:
        logging.warning("'-rab' switch does not expect namespace ('-ns')\n")
    # 获取所有有风险的角色绑定
    risky_any_rolebindings = engine.utils.get_all_risky_rolebinding()
    # 如果指定了天数，则过滤出指定天数内的角色绑定
    if days:
        risky_any_rolebindings = filter_objects_less_than_days(int(days), risky_any_rolebindings)
    # 如果指定了优先级，则根据优先级过滤角色绑定
    if priority:
# 根据优先级过滤风险角色绑定对象
risky_any_rolebindings = filter_objects_by_priority(priority, risky_any_rolebindings)
# 打印风险角色绑定和集群角色绑定
generic_print('|Risky RoleBindings and ClusterRoleBindings|', risky_any_rolebindings)

# 打印风险角色绑定
def print_risky_rolebindings(days=None, priority=None, namespace=None):
    # 获取风险角色绑定对象
    risky_rolebindings = engine.utils.get_risky_rolebindings()

    # 如果指定了天数，则根据天数过滤风险角色绑定对象
    if days:
        risky_rolebindings = filter_objects_less_than_days(int(days), risky_rolebindings)
    # 如果指定了优先级，则根据优先级过滤风险角色绑定对象
    if priority:
        risky_rolebindings = filter_objects_by_priority(priority, risky_rolebindings)

    # 如果未指定命名空间，则打印所有风险角色绑定
    if namespace is None:
        generic_print('|Risky RoleBindings|', risky_rolebindings)
    # 如果指定了命名空间，则根据命名空间过滤风险角色绑定并打印
    else:
        filtered_risky_rolebindings = []
        for risky_rolebinding in risky_rolebindings:
            if risky_rolebinding.namespace == namespace:
                filtered_risky_rolebindings.append(risky_rolebinding)
        generic_print('|Risky RoleBindings|', filtered_risky_rolebindings)
# 打印风险的 ClusterRoleBindings
def print_risky_clusterrolebindings(days=None, priority=None, namespace=None):
    # 如果指定了 namespace，则发出警告
    if namespace is not None:
        logging.warning("'-rcb' switch does not expect namespace ('-ns')\n")
    # 获取风险的 ClusterRoleBindings
    risky_clusterrolebindings = engine.utils.get_risky_clusterrolebindings()
    # 如果指定了 days，则根据 days 过滤 ClusterRoleBindings
    if days:
        risky_clusterrolebindings = filter_objects_less_than_days(int(days), risky_clusterrolebindings)
    # 如果指定了 priority，则根据 priority 过滤 ClusterRoleBindings
    if priority:
        risky_clusterrolebindings = filter_objects_by_priority(priority, risky_clusterrolebindings)
    # 调用 generic_print 函数打印结果
    generic_print('|Risky ClusterRoleBindings|', risky_clusterrolebindings)

# 通用打印函数，打印表头和对象
def generic_print(header, objects, show_rules=False):
    # 打印表头的装饰线
    roof = '+' + ('-' * (len(header)-2)) + '+'
    print(roof)
    # 打印表头
    print(header)
    # 如果需要显示规则
    if show_rules:
        # 创建 PrettyTable 对象
        t = PrettyTable(['Priority', 'Kind', 'Namespace', 'Name', 'Creation Time', 'Rules'])
        # 遍历对象列表，添加到 PrettyTable 中
        for o in objects:
            if o.time is None:
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, 'No creation time', get_pretty_rules(o.rules)])
            else:
    # 如果条件不成立，则创建一个表格对象
    else:
        t = PrettyTable(['Priority', 'Kind', 'Namespace', 'Name', 'Creation Time'])
        # 遍历对象列表，将数据添加到表格中
        for o in objects:
            # 如果对象的时间为空，则添加一行数据到表格中
            if o.time is None:
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, 'No creation time'])
            # 如果对象的时间不为空，则添加带有时间信息的数据到表格中
            else:
                t.add_row([get_color_by_priority(o.priority) + o.priority.name + WHITE, o.kind, o.namespace, o.name, o.time.ctime() + " (" + str(get_delta_days_from_now(o.time)) + " days)"])

    # 调用函数打印左对齐的表格
    print_table_aligned_left(t)

# 打印所有风险容器的信息
def print_all_risky_containers(priority=None, namespace=None, read_token_from_container=False):
    # 获取所有风险的 Pod 对象
    pods = engine.utils.get_risky_pods(namespace, read_token_from_container)

    # 打印标题
    print("+----------------+")
    print("|Risky Containers|")
    # 创建一个表格对象
    t = PrettyTable(['Priority', 'PodName', 'Namespace', 'ContainerName', 'ServiceAccountNamespace', 'ServiceAccountName'])
    # 遍历 Pod 列表，将数据添加到表格中
    for pod in pods:
        # 如果有指定优先级，则根据优先级过滤容器对象
        if priority:
            pod.containers = filter_objects_by_priority(priority, pod.containers)
# 遍历每个容器
for container in pod.containers:
    # 初始化所有服务账户的字符串
    all_service_account = ''
    # 遍历容器的服务账户集合，将服务账户的用户名拼接成一个字符串
    for service_account in container.service_accounts_name_set:
        all_service_account += service_account.user_info.name + ", "
    # 去除最后的逗号和空格
    all_service_account = all_service_account[:-2]
    # 将容器的优先级、Pod名称、命名空间、容器名称、服务账户命名空间和所有服务账户添加到表格中
    t.add_row([get_color_by_priority(container.priority)+container.priority.name+WHITE, pod.name, pod.namespace, container.name, container.service_account_namespace, all_service_account])

# 打印左对齐的表格
print_table_aligned_left(t)

# 打印所有风险主体
def print_all_risky_subjects(priority=None, namespace=None):
    # 获取所有风险主体
    subjects = engine.utils.get_all_risky_subjects()
    # 如果有指定优先级，则根据优先级筛选主体
    if priority:
        subjects = filter_objects_by_priority(priority, subjects)
    # 打印表头
    print("+-----------+")
    print("|Risky Users|")
    # 创建表格
    t = PrettyTable(['Priority', 'Kind', 'Namespace', 'Name'])
    # 遍历所有主体
    for subject in subjects:
        # 如果主体的命名空间与指定的命名空间相同，或者未指定命名空间
        if subject.user_info.namespace == namespace or namespace is None:
           # 将主体的优先级、类型、命名空间和名称添加到表格中
           t.add_row([get_color_by_priority(subject.priority)+subject.priority.name+WHITE, subject.user_info.kind, subject.user_info.namespace, subject.user_info.name])
# 调用函数打印左对齐的表格
print_table_aligned_left(t)

# 打印所有风险角色、角色绑定、主体和容器
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

    # 打印与指定角色相关的角色绑定的信息
    print("Associated Rolebindings to Role \"{0}\":".format(role_name))
    t = PrettyTable(['Kind', 'Name', 'Namespace'])

    # 遍历相关的角色绑定并添加到表格中
    for rolebinding in associated_rolebindings:
        t.add_row(['RoleBinding', rolebinding.metadata.name, rolebinding.metadata.namespace])

    # 调用函数打印左对齐的表格
    print_table_aligned_left(t)
# 打印与指定集群角色相关的角色绑定和集群角色绑定
def print_associated_any_rolebindings_to_clusterrole(clusterrole_name):
    # 获取与指定集群角色相关的角色绑定和集群角色绑定
    associated_rolebindings, associated_clusterrolebindings = engine.utils.get_rolebindings_and_clusterrolebindings_associated_to_clusterrole(role_name=clusterrole_name)

    # 打印与指定集群角色相关的角色绑定和集群角色绑定的标题
    print("Associated Rolebindings\ClusterRoleBinding to ClusterRole \"{0}\":".format(clusterrole_name))
    # 创建一个表格对象
    t = PrettyTable(['Kind', 'Name', 'Namespace'])

    # 遍历关联的角色绑定，将其信息添加到表格中
    for rolebinding in associated_rolebindings:
        t.add_row(['RoleBinding', rolebinding.metadata.name, rolebinding.metadata.namespace])

    # 遍历关联的集群角色绑定，将其信息添加到表格中
    for clusterrolebinding in associated_clusterrolebindings:
        t.add_row(['ClusterRoleBinding', clusterrolebinding.metadata.name, clusterrolebinding.metadata.namespace])

    # 打印表格
    print_table_aligned_left(t)

# 打印与指定主体相关的角色绑定和集群角色绑定
def print_associated_rolebindings_and_clusterrolebindings_to_subject(subject_name, kind, namespace=None):
    # 获取与指定主体相关的角色绑定和集群角色绑定
    associated_rolebindings, associated_clusterrolebindings = engine.utils.get_rolebindings_and_clusterrolebindings_associated_to_subject(subject_name, kind, namespace)

    # 打印与指定主体相关的角色绑定和集群角色绑定的标题
    print("Associated Rolebindings\ClusterRoleBindings to subject \"{0}\":".format(subject_name))
    # 创建一个表格对象
    t = PrettyTable(['Kind', 'Name', 'Namespace'])
# 遍历关联的角色绑定列表，将角色绑定的信息添加到表格中
for rolebinding in associated_rolebindings:
    t.add_row(['RoleBinding', rolebinding.metadata.name, rolebinding.metadata.namespace])

# 遍历关联的集群角色绑定列表，将集群角色绑定的信息添加到表格中
for clusterrolebinding in associated_clusterrolebindings:
    t.add_row(['ClusterRoleBinding', clusterrolebinding.metadata.name, clusterrolebinding.metadata.namespace])

# 打印左对齐的表格
print_table_aligned_left(t)

# 反序列化令牌，将令牌对象转换为字符串形式
def desrialize_token(token):
    desirialized_token = ''
    for key in token.keys():
        desirialized_token += key + ': ' + token[key]
        desirialized_token += '\n'
    return desirialized_token

# 从容器中获取令牌并转储，如果指定了 pod_name 和 namespace，则获取指定 pod 的令牌信息
def dump_tokens_from_pods(pod_name=None, namespace=None, read_token_from_container=False):
    if pod_name is not None:
        # 获取指定 pod 的令牌信息
        pods_with_tokens = engine.utils.dump_pod_tokens(pod_name, namespace, read_token_from_container)
    else:
# 使用引擎工具类的方法获取指定命名空间中所有Pod的令牌信息
pods_with_tokens = engine.utils.dump_all_pods_tokens_or_by_namespace(namespace, read_token_from_container)

# 创建一个表格对象，设置表头
t = PrettyTable(['PodName',  'Namespace', 'ContainerName', 'Decoded Token'])
# 遍历每个Pod，获取其容器的令牌信息并添加到表格中
for pod in pods_with_tokens:
    for container in pod.containers:
        new_token = desrialize_token(container.token)
        t.add_row([pod.name, pod.namespace, container.name, new_token])

# 打印格式对齐的表格
print_table_aligned_left(t)


# 根据指定的资源类型获取所有相关的主体信息
def print_subjects_by_kind(kind):
    subjects = engine.utils.get_subjects_by_kind(kind)
    # 打印主体信息的表头
    print('Subjects (kind: {0}) from all rolebindings:'.format(kind))
    t = PrettyTable(['Kind', 'Namespace', 'Name'])
    # 遍历每个主体，将其信息添加到表格中
    for subject in subjects:
        t.add_row([subject.kind, subject.namespace, subject.name])

    # 打印格式对齐的表格
    print_table_aligned_left(t)
    # 打印主体信息的总数
    print('Total number: %s' % len(subjects))
# 根据规则列表生成可读的规则字符串
def get_pretty_rules(rules):
    # 初始化可读规则字符串
    pretty = ''
    # 如果规则列表不为空
    if rules is not None:
        # 遍历规则列表
        for rule in rules:
            # 初始化动词字符串
            verbs_string = '('
            # 遍历规则中的动词列表
            for verb in rule.verbs:
                verbs_string += verb + ','
            verbs_string = verbs_string[:-1]
            verbs_string += ')->'

            # 初始化资源字符串
            resources_string = '('
            # 如果规则中的资源列表为空
            if rule.resources is None:
                resources_string += 'None'
            else:
                # 遍历规则中的资源列表
                for resource in rule.resources:
                    resources_string += resource + ','
                resources_string = resources_string[:-1]
            resources_string += ')\n'
# 将动词字符串和资源字符串拼接起来
pretty += verbs_string + resources_string
# 返回拼接后的字符串
return pretty

# 打印角色绑定的规则
def print_rolebinding_rules(rolebinding_name, namespace):
    # 获取角色绑定的角色
    role = engine.utils.get_rolebinding_role(rolebinding_name, namespace)
    # 打印角色绑定的名称和规则
    print("RoleBinding '{0}\{1}' rules:".format(namespace, rolebinding_name))
    # 创建一个表格对象
    t = PrettyTable(['Kind', 'Namespace', 'Name', 'Rules'])
    # 添加一行数据到表格中
    t.add_row([role.kind, role.metadata.namespace, role.metadata.name, get_pretty_rules(role.rules)])
    # 打印左对齐的表格
    print_table_aligned_left(t)

# 打印与主体相关的规则
def print_rules_associated_to_subject(name, kind, namespace=None):
# 获取与指定主体关联的角色
roles = engine.utils.get_roles_associated_to_subject(name, kind, namespace)
# 打印与主体 '{0}' 关联的角色
print("Roles associated to Subject '{0}':".format(name))
# 创建一个表格对象
t = PrettyTable(['Kind', 'Namespace', 'Name', 'Rules'])
# 遍历角色列表，将角色信息添加到表格中
for role in roles:
    t.add_row([role.kind, role.metadata.namespace, role.metadata.name, get_pretty_rules(role.rules)])
# 打印格式化对齐的表格
print_table_aligned_left(t)

# 创建一个打印具有通过卷访问秘密数据的 Pod 的函数
# 获取所有命名空间或指定命名空间的 Pod 列表
def print_pods_with_access_secret_via_volumes(namespace=None):
    pods = engine.utils.list_pods_for_all_namespaces_or_one_namspace(namespace)

    # 打印具有通过卷访问秘密数据的 Pod
    print("Pods with access to secret data through volumes:")
    # 创建一个表格对象
    t = PrettyTable(['Pod Name', 'Namespace', 'Container Name', 'Volume Mounted Secrets'])
    # 遍历 Pod 列表，将相关信息添加到表格中
    for pod in pods.items:
        for container in pod.spec.containers:
            mount_info = ''
            secrets_num = 1
            if container.volume_mounts is not None:
                for volume_mount in container.volume_mounts:
# 遍历 Pod 的存储卷，检查是否有秘钥卷，并且名称匹配
for volume in pod.spec.volumes:
    if volume.secret is not None and volume.name == volume_mount.name:
        # 如果匹配，将挂载路径、秘钥名称和卷名称添加到挂载信息中
        mount_info += '{2}. Mounted path: {0}\n   Secret name: {1}\n'.format(volume_mount.mount_path, volume.secret.secret_name, secrets_num)
        secrets_num += 1
# 如果挂载信息不为空，将 Pod 名称、命名空间、容器名称和挂载信息添加到表格中
if mount_info != '':
    t.add_row([pod.metadata.name, pod.metadata.namespace, container.name, mount_info])

# 打印格式化对齐的表格
print_table_aligned_left(t)

# 获取所有 Pod，并遍历每个 Pod
def print_pods_with_access_secret_via_environment(namespace=None):
    pods = engine.utils.list_pods_for_all_namespaces_or_one_namspace(namespace)

    # 打印标题
    print("Pods with access to secret data through environment:")
    # 创建表格
    t = PrettyTable(['Pod Name', 'Namespace', 'Container Name', 'Environment Mounted Secrets'])
    for pod in pods.items:
        # 遍历每个容器
        for container in pod.spec.containers:
            # 初始化挂载信息和秘钥编号
            mount_info = ''
            secrets_num = 1
# 如果容器的环境变量不为空，则遍历环境变量列表
if container.env is not None:
    for env in container.env:
        # 如果环境变量的值来自于密钥引用
        if env.value_from is not None and env.value_from.secret_key_ref is not None:
            # 将环境变量名和密钥引用的密钥名添加到挂载信息中
            mount_info += '{2}. Environemnt variable name: {0}\n   Secret name: {1}\n'.format(env.name, env.value_from.secret_key_ref.name, secrets_num)
            secrets_num += 1
    # 如果挂载信息不为空，则将容器、Pod和挂载信息添加到表格中
    if mount_info != '':
        t.add_row([pod.metadata.name, pod.metadata.namespace, container.name, mount_info])

# 打印左对齐的表格
print_table_aligned_left(t)

# 解析安全上下文
def parse_security_context(security_context):
    is_header_set = False
    context = ''
    # 如果安全上下文不为空
    if security_context:
        # 将安全上下文转换为字典
        dict =  security_context.to_dict()
        # 遍历字典的键
        for key in dict.keys():
            # 如果键对应的值不为空
            if dict[key] is not None:
                # 如果标题未设置，则添加安全上下文的标题
                if not is_header_set:
                    context += "SecurityContext:\n"
                    is_header_set = True
# 将容器规范解析为字符串
def parse_container_spec(container_spec):
    # 初始化规范字符串
    spec = ''
    # 将容器规范对象转换为字典
    dict =  container_spec.to_dict()
    # 初始化端口头部是否已设置的标志
    is_ports_header_set = False
    # 遍历容器规范字典的键
    for key in dict.keys():
        # 如果值不为空
        if dict[key] is not None:
            # 如果键为'ports'
            if key == 'ports':
                # 如果端口头部未设置，则设置端口头部
                if not is_ports_header_set:
                    spec += "Ports:\n"
                    is_ports_header_set = True
                # 遍历端口对象列表
                for port_obj in dict[key]:
                    # 如果端口对象中包含'host_port'键
                    if 'host_port' in port_obj:
                        # 添加容器端口和主机端口到规范字符串
                        spec += '  {0}: {1}\n'.format('container_port', port_obj['container_port'])
                        spec += '  {0}: {1}\n'.format('host_port', port_obj['host_port'])
                        break
    # 将安全上下文解析结果添加到规范字符串
    spec += parse_security_context(container_spec.security_context)
    # 返回规范字符串
    return spec
# 解析 Pod 规范和容器信息，生成规范字符串
def parse_pod_spec(pod_spec, container):
    # 初始化规范字符串
    spec = ''
    # 将 Pod 规范转换为字典
    dict =  pod_spec.to_dict()
    # 初始化标记，用于标识是否已经设置了卷头部
    is_volumes_header_set = False
    # 遍历字典的键
    for key in dict.keys():
        # 如果值不为空
        if dict[key] is not None:
            # 如果键是 'host_pid'、'host_ipc' 或 'host_network'，将其添加到规范字符串中
            if key == 'host_pid' or key == 'host_ipc' or key == 'host_network':
                spec += '{0}: {1}\n'.format(key, dict[key])

            # 如果键是 'volumes' 并且容器的卷挂载不为空
            if key == 'volumes' and container.volume_mounts is not None:
                # 遍历卷对象列表
                for volume_obj in dict[key]:
                    # 如果卷对象中包含 'host_path'
                    if 'host_path' in volume_obj:
                        # 如果 'host_path' 不为空
                        if volume_obj['host_path']:
                            # 遍历容器的卷挂载
                            for volume_mount in container.volume_mounts:
                                # 如果卷对象的名称与卷挂载的名称相匹配
                                if volume_obj['name'] == volume_mount.name:
                                    # 如果卷头部尚未设置，添加卷头部
                                    if not is_volumes_header_set:
                                        spec += "Volumes:\n"
                                        is_volumes_header_set = True
                                    # 将卷信息添加到规范字符串中
                                    spec += '  -{0}: {1}\n'.format('name', volume_obj['name'])
# 拼接 spec 字符串，包含 host_path 的信息
spec += '   host_path:\n'
spec += '     {0}: {1}\n'.format('path', volume_obj['host_path']['path'])
spec += '     {0}: {1}\n'.format('type', volume_obj['host_path']['type'])
spec += '     {0}: {1}\n'.format('container_path', volume_mount.mount_path)

# 解析安全上下文并拼接到 spec 字符串中
spec += parse_security_context(pod_spec.security_context)
# 返回拼接好的 spec 字符串
return spec

# 打印特权容器的信息
def print_privileged_containers(namespace=None):
    # 打印表头
    print("+---------------------+")
    print("|Privileged Containers|")
    # 创建一个表格对象
    t = PrettyTable(['Pod', 'Namespace', 'Pod Spec', 'Container', 'Container info'])
    # 获取特权容器的信息
    pods = engine.privleged_containers.get_privileged_containers(namespace)
    # 遍历每个特权容器，将信息添加到表格中
    for pod in pods:
        for container in pod.spec.containers:
            t.add_row([pod.metadata.name, pod.metadata.namespace, parse_pod_spec(pod.spec, container), container.name, parse_container_spec(container)])
    # 打印表格
    print_table_aligned_left(t)

# 打印加入令牌的信息
def print_join_token():
    # 导入操作系统模块
    import os
    # 从kubernetes客户端模块中导入配置
    from kubernetes.client import Configuration
    # 从配置中获取主节点IP地址
    master_ip = Configuration().host.split(':')[1][2:]
    # 从配置中获取主节点端口号
    master_port = Configuration().host.split(':')[2]

    # 设置CA证书路径
    ca_cert = '/etc/kubernetes/pki/ca.crt'
    # 如果CA证书路径不存在，则使用默认路径
    if not os.path.exists(ca_cert):
        ca_cert = '/etc/kubernetes/ca.crt'

    # 如果在容器中运行，则使用环境变量中的路径
    if running_in_container():
        ca_cert = os.getenv('KUBISCAN_VOLUME_PATH', '/tmp') + ca_cert

    # 设置加入令牌路径
    join_token_path = os.path.dirname(os.path.realpath(__file__)) + '/engine/join_token.sh'
    # 获取解码后的引导令牌列表
    tokens = engine.utils.list_boostrap_tokens_decoded()

    # 如果没有引导令牌，则打印消息
    if not tokens:
        print("No bootstrap tokens exist")
    # 否则，遍历引导令牌列表
    else:
        for token in tokens:
# 执行 shell 命令，拼接命令字符串，打印并执行
command = 'sh ' + join_token_path + ' ' + ' '.join([master_ip, master_port, ca_cert, token])
print('\nExecute: %s' % command)
os.system(command)

# 打印程序的 logo
def print_logo():
    logo = '''
                   `-/osso/-`                    
                `-/osssssssssso/-`                
            .:+ssssssssssssssssssss+:.            
        .:+ssssssssssssssssssssssssssss+:.        
     :osssssssssssssssssssssssssssssssssso:     
    /sssssssssssss+::osssssso::+sssssssssssss+    
   `sssssssssso:--..-`+ssss+ -..--:ossssssssss`   
   /sssssssss:.+ssss/ /ssss/ /ssss+.:sssssssss/   
  `ssssssss:.+sssssss./ssss/`sssssss+.:ssssssss`  
  :ssssss/`-///+oss+/`-////-`/+sso+///-`/ssssss/  
  sssss+.`.-:-:-..:/`-++++++-`/:..-:-:-.`.+sssss` 
 :ssso..://:-`:://:.. osssso ..://::`-://:..osss: 
 osss`-/-.`-- :.`.-/. /ssss/ ./-.`-: --`.-/-`osso 
-sss:`//..-`` .`-`-//`.----. //-`-`. ``-..//.:sss-
```
# 打印 KubiScan 版本和作者信息
def print_logo():
    logo = '''
osss:.::`...`- ..`.:/`+ssss+`/:``.. -`...`::.:ssso
+ssso`:/:`--`:`--`/:-`ssssss`-//`--`:`--`:/:`osss+
 :sss+`-//.`...`-//..osssssso..//-`...`.//-`+sss: 
  `+sss/...::/::..-+ssssssssss+-..::/::.../sss+`  
    -ossss+/:::/+ssssssssssssssss+/:::/+sssso-    
      :ssssssssssssssssssssssssssssssssssss/      
       `+ssssssssssssssssssssssssssssssss+`       
         -osssssssssssssssssssssssssssss-         
          `/ssssssssssssssssssssssssss/`       
    
               KubiScan version 1.6
               Author: Eviatar Gerzi
    '''
    print(logo)

# 打印示例
def print_examples():
    import os
    # 打开示例文件并打印内容
    with open(os.path.dirname(os.path.realpath(__file__)) + '/examples/examples.txt', 'r') as f:
        print(f.read())
# 主函数，用于执行程序
def main():
    # 创建参数解析器对象，描述和用法
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
    
    # 添加命令行参数，获取所有风险角色
    opt.add_argument('-rr', '--risky-roles', action='store_true', help='Get all risky Roles (can be used with -r to view rules)', required=False)
# 添加命令行参数 -rcr/--risky-clusterroles，设置为True时获取所有危险的ClusterRoles（可以与-r一起使用以查看规则），不是必需的
opt.add_argument('-rcr', '--risky-clusterroles', action='store_true', help='Get all risky ClusterRoles (can be used with -r to view rules)',required=False)

# 添加命令行参数 -rar/--risky-any-roles，设置为True时获取所有危险的Roles和ClusterRoles，不是必需的
opt.add_argument('-rar', '--risky-any-roles', action='store_true', help='Get all risky Roles and ClusterRoles', required=False)

# 添加命令行参数 -rb/--risky-rolebindings，设置为True时获取所有危险的RoleBindings，不是必需的
opt.add_argument('-rb', '--risky-rolebindings', action='store_true', help='Get all risky RoleBindings', required=False)

# 添加命令行参数 -rcb/--risky-clusterrolebindings，设置为True时获取所有危险的ClusterRoleBindings，不是必需的
opt.add_argument('-rcb', '--risky-clusterrolebindings', action='store_true',help='Get all risky ClusterRoleBindings', required=False)

# 添加命令行参数 -rab/--risky-any-rolebindings，设置为True时获取所有危险的RoleBindings和ClusterRoleBindings，不是必需的
opt.add_argument('-rab', '--risky-any-rolebindings', action='store_true', help='Get all risky RoleBindings and ClusterRoleBindings', required=False)

# 添加命令行参数 -rs/--risky-subjects，设置为True时获取所有危险的Subjects（用户、组或服务账户），不是必需的
opt.add_argument('-rs', '--risky-subjects', action='store_true',help='Get all risky Subjects (Users, Groups or Service Accounts)', required=False)

# 添加命令行参数 -rp/--risky-pods，设置为True时获取所有危险的Pods/Containers，不是必需的
# 使用-d/--deep开关从当前运行的容器中读取令牌
opt.add_argument('-rp', '--risky-pods', action='store_true', help='Get all risky Pods\Containers.\n'
                                                                      'Use the -d\--deep switch to read the tokens from the current running containers', required=False)

# 添加命令行参数 -d/--deep，设置为True时仅与-rp/--risky-pods开关一起使用。如果指定了此参数，它将执行每个Pod以获取其令牌。
# 没有指定时，它将从ETCD中读取挂载的服务账户密钥，这种方法不太可靠但速度更快。
opt.add_argument('-d', '--deep', action='store_true', help='Works only with -rp\--risky-pods switch. If this is specified, it will execute each pod to get its token.\n'
                                                               'Without it, it will read the pod mounted service account secret from the ETCD, it less reliable but much faster.', required=False)

# 添加命令行参数 -pp/--privleged-pods，设置为True时获取所有特权的Pods/Containers，不是必需的
opt.add_argument('-pp', '--privleged-pods', action='store_true', help='Get all privileged Pods\Containers.',  required=False)

# 添加命令行参数 -a/--all，设置为True时获取所有危险的Roles/ClusterRoles、RoleBindings/ClusterRoleBindings、用户和Pods/Containers，不是必需的
opt.add_argument('-a', '--all', action='store_true',help='Get all risky Roles\ClusterRoles, RoleBindings\ClusterRoleBindings, users and pods\containers', required=False)

# 添加命令行参数 -cve/--cve，设置为True时进行CVE扫描，不是必需的
opt.add_argument('-cve', '--cve', action='store_true', help=f"Scan of CVE's", required=False)

# 添加命令行参数 -jt/--join-token，设置为True时获取集群的加入令牌。必须安装OpenSsl和kubeadm，不是必需的
opt.add_argument('-jt', '--join-token', action='store_true', help='Get join token for the cluster. OpenSsl must be installed + kubeadm', required=False)

# 添加命令行参数 -psv/--pods-secrets-volume，设置为True时显示所有具有通过卷访问秘密数据的Pods，不是必需的
opt.add_argument('-psv', '--pods-secrets-volume', action='store_true', help='Show all pods with access to secret data throught a Volume', required=False)

# 添加命令行参数 -pse/--pods-secrets-env，设置为True时显示所有具有通过环境变量访问秘密数据的Pods，不是必需的
opt.add_argument('-pse', '--pods-secrets-env', action='store_true', help='Show all pods with access to secret data throught a environment variables', required=False)

# 添加命令行参数 -ctx/--context，设置为True时指定要运行的上下文。如果没有指定，它将在当前上下文中运行，不是必需的
opt.add_argument('-ctx', '--context', action='store', help='Context to run. If none, it will run in the current context.', required=False)

# 添加命令行参数 -p/--priority，设置为True时按优先级过滤（CRITICAL\HIGH\LOW），不是必需的
opt.add_argument('-p', '--priority', action='store', help='Filter by priority (CRITICAL\HIGH\LOW)', required=False)
# 创建一个参数组，用于存放辅助开关
helper_switches = opt.add_argument_group('Helper switches')

# 向辅助开关参数组中添加参数，用于指定命名空间范围
helper_switches.add_argument('-ns', '--namespace', action='store', help='If present, the namespace scope that will be used', required=False)

# 向辅助开关参数组中添加参数，用于指定对象的类型
helper_switches.add_argument('-k', '--kind', action='store', help='Kind of the object', required=False)

# 向辅助开关参数组中添加参数，用于显示规则。仅在打印风险角色/集群角色时支持
helper_switches.add_argument('-r', '--rules', action='store_true', help='Show rules. Supported only on pinrting risky Roles\ClusterRoles.', required=False)

# 向辅助开关参数组中添加参数，用于显示示例
helper_switches.add_argument('-e', '--examples', action='store_true', help='Show examples.', required=False)

# 向辅助开关参数组中添加参数，用于指定名称
helper_switches.add_argument('-n', '--name', action='store', help='Name', required=False)

# 创建一个参数组，用于存放令牌转储相关开关
dumping_tokens = opt.add_argument_group('Dumping tokens', description='Use the switches: name (-n\--name) or namespace (-ns\ --namespace)')

# 向令牌转储参数组中添加参数，用于执行令牌转储操作
dumping_tokens.add_argument('-dt', '--dump-tokens', action='store_true', help='Dump tokens from pod\pods\n'
                                                                              'Example: -dt OR -dt -ns \"kube-system\"\n'
                                                                              '-dt -n \"nginx1\" -ns \"default\"', required=False)

# 创建一个参数组，用于存放远程相关开关
helper_switches = opt.add_argument_group('Remote switches')

# 向远程开关参数组中添加参数，用于指定主机地址和端口
helper_switches.add_argument('-ho', '--host', action='store', metavar='<MASTER_IP>:<PORT>', help='Host contain the master ip and port.\n'
                                                                                                 'For example: 10.0.0.1:6443', required=False)
                                                                            'Inside Pods the default location is \'/var/run/secrets/kubernetes.io/serviceaccount/ca.crt\''
                                                                            'Or \'/run/secrets/kubernetes.io/serviceaccount/ca.crt\'.', required=False)

# 向远程开关参数组中添加参数，用于指定客户端证书文件名
helper_switches.add_argument('-cc', '--client-certificate', action='store', metavar='CA_FILENAME',
```
# 添加命令行参数，用于指定客户端证书文件的路径
helper_switches.add_argument('-cc', '--client-cert', action='store', metavar='CLIENT_CERT_FILENAME',
                             help='Path to client certificate file', required=False)

# 添加命令行参数，用于指定客户端密钥文件的路径
helper_switches.add_argument('-ck', '--client-key', action='store', metavar='CA_FILENAME',
                             help='Path to client key file', required=False)

# 添加命令行参数，用于指定 kube 配置文件的路径
helper_switches.add_argument('-co', '--kube-config', action='store', metavar='KUBE_CONFIG_FILENAME',
                             help='The kube config file.\n'
                                  'For example: ~/.kube/config', required=False)

# 添加命令行参数，用于指定令牌文件的路径
helper_switches.add_argument('-t', '--token-filename', action='store', metavar='TOKEN_FILENAME',
                             help='A bearer token. If this token does not have the required permissions for this application,'
                                  'the application will fail to get some of the information.\n'
                                  'Minimum required permissions:\n'
                                  '- resources: [\"roles\", \"clusterroles\", \"rolebindings\", \"clusterrolebindings\", \"pods\", \"secrets\"]\n'
                                  '  verbs: [\"get\", \"list\"]\n'
                                  '- resources: [\"pods/exec\"]\n'
                                  '  verbs: [\"create\"]')

# 添加命令行参数，用于指定输出文件的路径
helper_switches.add_argument('-o', '--output-file', metavar='OUTPUT_FILENAME', help='Path to output file')

# 添加命令行参数，用于隐藏横幅
helper_switches.add_argument('-q', '--quiet', action='store_true', help='Hide the banner')

# 添加命令行参数，用于导出为 json 文件
helper_switches.add_argument('-j', '--json', metavar='JSON_FILENAME', help='Export to json')

# 添加命令行参数，用于打印时不使用颜色
helper_switches.add_argument('-nc', '--no-color', action='store_true', help='Print without color')
# 创建一个参数组，用于关联 RoleBindings\ClusterRoleBindings 到 Role，提供描述信息
associated_rb_crb_to_role = opt.add_argument_group('Associated RoleBindings\ClusterRoleBindings to Role', description='Use the switch: namespace (-ns\--namespace).')
# 添加一个参数到参数组，用于获取关联到特定角色的 RoleBindings\ClusterRoleBindings
associated_rb_crb_to_role.add_argument('-aarbr', '--associated-any-rolebindings-role', action='store', metavar='ROLE_NAME',
                                       help='Get associated RoleBindings\ClusterRoleBindings to a specific role\n'
                                            'Example: -aarbr \"read-secrets-role\" -ns \"default\"', required=False)

# 创建一个参数组，用于关联 RoleBindings\ClusterRoleBindings 到 ClusterRole
associated_rb_crb_to_clusterrole = opt.add_argument_group('Associated RoleBindings\ClusterRoleBindings to ClusterRole')
# 添加一个参数到参数组，用于获取关联到特定 ClusterRole 的 RoleBindings\ClusterRoleBindings
associated_rb_crb_to_clusterrole.add_argument('-aarbcr', '--associated-any-rolebindings-clusterrole', action='store', metavar='CLUSTERROLE_NAME',
                                              help='Get associated RoleBindings\ClusterRoleBindings to a specific role\n'
                                                   'Example:  -aarbcr \"read-secrets-clusterrole\"', required=False)

# 创建一个参数组，用于关联 RoleBindings\ClusterRoleBindings 到 Subject（用户、组或服务账户），提供描述信息
associated_rb_crb_to_subject = opt.add_argument_group('Associated RoleBindings\ClusterRoleBindings to Subject (user, group or service account)',
                                                       description='Use the switches: namespace (-ns\--namespace) and kind (-k\--kind).\n')
# 添加一个参数到参数组，用于获取关联到特定 Subject 的 RoleBindings\ClusterRoleBindings
associated_rb_crb_to_subject.add_argument('-aarbs', '--associated-any-rolebindings-subject', action='store', metavar='SUBJECT_NAME',
                                          help='Get associated Rolebindings\ClusterRoleBindings to a specific Subject (user, group or service account)\n'
                                               'Example: -aarbs \"system:masters\" -k \"Group\"', required=False)

# 创建一个参数组，用于关联 Roles\ClusterRoles 到 Subject（用户、组或服务账户），提供描述信息
associated_rb_crb_to_subject = opt.add_argument_group('Associated Roles\ClusterRoles to Subject (user, group or service account)',
                                                       description='Use the switches: namespace (-ns\--namespace) and kind (-k\--kind).\n')
# 添加一个参数到参数组，用于获取关联到特定 Subject 的 Roles\ClusterRoles
associated_rb_crb_to_subject.add_argument('-aars', '--associated-any-roles-subject', action='store', metavar='SUBJECT_NAME',
# 添加关于获取与特定主体（用户、组或服务账户）相关的角色\集群角色的帮助信息
opt.add_argument('-aars', '--associated-roles', action='store', metavar='SUBJECT_NAME', 
                                                  help='Get associated Roles\ClusterRoles to a specific Subject (user, group or service account)\n'
                                                       'Example: -aars \"generic-garbage-collector\" -k \"ServiceAccount\" -ns \"kube-system\"', required=False)

# 添加关于列出主体的帮助信息
list_subjects = opt.add_argument_group('List Subjects')
list_subjects.add_argument('-su', '--subject-users', action='store_true', help='Get Subjects with User kind', required=False)
list_subjects.add_argument('-sg', '--subject-groups', action='store_true', help='Get Subjects with Group kind', required=False)
list_subjects.add_argument('-ss', '--subject-serviceaccounts', action='store_true', help='Get Subjects with ServiceAccount kind', required=False)

# 添加关于列出角色绑定\集群角色绑定规则的帮助信息
list_rules = opt.add_argument_group('List rules of RoleBinding\ClusterRoleBinding')
list_rules.add_argument('-rru', '--rolebinding-rules', action='store', metavar='ROLEBINDING_NAME', help='Get rules of RoleBinding', required=False)
list_rules.add_argument('-crru', '--clusterrolebinding-rules', action='store', metavar='CLUSTERROLEBINDING_NAME',  help='Get rules of ClusterRoleBinding',required=False)

# 解析命令行参数
args = opt.parse_args()
# 如果设置了不显示颜色的选项，则设置全局变量 no_color 为 True
if args.no_color:
    global no_color
    no_color = True
# 如果设置了输出为 JSON 格式的选项，则设置全局变量 json_filename 为对应的文件名
if args.json:
    global json_filename
    json_filename = args.json
    # 如果指定了输出文件，则打开该文件以便写入，并将标准输出重定向到该文件
    if args.output_file:
        f = open(args.output_file, 'w')
        sys.stdout = f
    # 如果没有指定安静模式，则打印logo
    if not args.quiet:
        print_logo()

    # 如果指定了示例参数，则打印示例并退出
    if args.examples:
        print_examples()
        exit()

    # 初始化 API，传入相应的参数
    api_init(kube_config_file=args.kube_config, host=args.host, token_filename=args.token_filename, cert_filename=args.cert_filename, context=args.context)

    # 如果指定了 cve 参数，则打印 cve 信息
    if args.cve:
        print_cve(args.cert_filename, args.client_certificate, args.client_key, args.host)
    # 如果指定了 risky_roles 参数，则打印风险角色信息
    if args.risky_roles:
        print_risky_roles(show_rules=args.rules, days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果指定了 risky_clusterroles 参数，则打印风险集群角色信息
    if args.risky_clusterroles:
        print_risky_clusterroles(show_rules=args.rules, days=args.less_than, priority=args.priority, namespace=args.namespace)
    # 如果指定了 risky_any_roles 参数，则...
    if args.risky_any_roles:
# 如果参数中包含 show_rules，则打印所有风险角色
print_all_risky_roles(show_rules=args.rules, days=args.less_than, priority=args.priority, namespace=args.namespace)
# 如果参数中包含 risky_rolebindings，则打印风险角色绑定
if args.risky_rolebindings:
    print_risky_rolebindings(days=args.less_than, priority=args.priority, namespace=args.namespace)
# 如果参数中包含 risky_clusterrolebindings，则打印风险集群角色绑定
if args.risky_clusterrolebindings:
    print_risky_clusterrolebindings(days=args.less_than, priority=args.priority, namespace=args.namespace)
# 如果参数中包含 risky_any_rolebindings，则打印所有风险角色绑定
if args.risky_any_rolebindings:
    print_all_risky_rolebindings(days=args.less_than, priority=args.priority, namespace=args.namespace)
# 如果参数中包含 risky_subjects，则打印所有风险主体
if args.risky_subjects:
    print_all_risky_subjects(priority=args.priority, namespace=args.namespace)
# 如果参数中包含 risky_pods，则打印所有风险容器
if args.risky_pods:
    print_all_risky_containers(priority=args.priority, namespace=args.namespace, read_token_from_container=args.deep)
# 如果参数中包含 all，则打印所有信息
if args.all:
    print_all(days=args.less_than, priority=args.priority, read_token_from_container=args.deep)
# 如果参数中包含 privleged_pods，则打印特权容器
elif args.privleged_pods:
    print_privileged_containers(namespace=args.namespace)
# 如果参数中包含 join_token，则打印加入令牌
elif args.join_token:
    print_join_token()
# 如果参数中包含 pods_secrets_volume，则打印通过卷访问秘密的容器
elif args.pods_secrets_volume:
    # 如果参数中包含 namespace，则打印指定命名空间中通过卷访问秘密的容器
    if args.namespace:
        print_pods_with_access_secret_via_volumes(namespace=args.namespace)
    # 如果没有指定参数，则打印具有访问秘钥的卷的Pods
    else:
        print_pods_with_access_secret_via_volumes()
    # 如果指定了参数 pods_secrets_env，则根据命名空间打印具有通过环境变量访问秘钥的Pods
    elif args.pods_secrets_env:
        if args.namespace:
            print_pods_with_access_secret_via_environment(namespace=args.namespace)
        else:
            print_pods_with_access_secret_via_environment()
    # 如果指定了参数 associated_any_rolebindings_role，则根据命名空间打印与指定角色相关的角色绑定
    elif args.associated_any_rolebindings_role:
        if args.namespace:
            print_associated_rolebindings_to_role(args.associated_any_rolebindings_role, args.namespace)
    # 如果指定了参数 associated_any_rolebindings_clusterrole，则打印与指定集群角色相关的任意角色绑定
    elif args.associated_any_rolebindings_clusterrole:
        print_associated_any_rolebindings_to_clusterrole(args.associated_any_rolebindings_clusterrole)
    # 如果指定了参数 associated_any_rolebindings_subject，则根据指定的类型和命名空间打印与主体相关的角色绑定和集群角色绑定
    elif args.associated_any_rolebindings_subject:
        if args.kind:
            if args.kind == constants.SERVICEACCOUNT_KIND:
                if args.namespace:
                    print_associated_rolebindings_and_clusterrolebindings_to_subject(args.associated_any_rolebindings_subject, args.kind, args.namespace)
                else:
                    print('For ServiceAccount kind specify namespace (-ns, --namespace)')
            else:
                # 其他情况
# 如果参数中指定了关联任何角色绑定的主体，则打印关联的角色绑定和集群角色绑定
if args.associated_any_rolebindings_subject:
    # 如果指定了主体的类型
    if args.kind:
        # 如果指定了服务账号类型
        if args.kind == constants.SERVICEACCOUNT_KIND:
            # 如果指定了命名空间
            if args.namespace:
                # 打印与主体关联的规则
                print_rules_associated_to_subject(args.associated_any_rolebindings_subject, args.kind, args.namespace)
            else:
                # 提示需要指定命名空间
                print('For ServiceAccount kind specify namespace (-ns, --namespace)')
        else:
            # 打印与主体关联的规则
            print_rules_associated_to_subject(args.associated_any_rolebindings_subject, args.kind)
    else:
        # 提示需要指定类型
        print("Please specify kind (-k, --kind).")
# 如果参数中指定了dump_tokens，则执行以下操作
elif args.dump_tokens:
    # 如果指定了名称
    if args.name:
        # 如果指定了命名空间
        if args.namespace:
            # 从Pod中转储令牌
            dump_tokens_from_pods(pod_name=args.name, namespace=args.namespace, read_token_from_container=args.deep)
        else:
            # 提示需要同时指定Pod名称和命名空间
            print('When specificing Pod name, need also namespace')
# 如果参数中包含 namespace，则从指定的 namespace 中获取 pod 的 token，并根据 deep 参数读取容器中的 token
elif args.namespace:
    dump_tokens_from_pods(namespace=args.namespace, read_token_from_container=args.deep)
# 如果参数中包含 subject_users，则打印用户的信息
elif args.subject_users:
    print_subjects_by_kind(constants.USER_KIND)
# 如果参数中包含 subject_groups，则打印用户组的信息
elif args.subject_groups:
    print_subjects_by_kind(constants.GROUP_KIND)
# 如果参数中包含 subject_serviceaccounts，则打印服务账号的信息
elif args.subject_serviceaccounts:
    print_subjects_by_kind(constants.SERVICEACCOUNT_KIND)
# 如果参数中包含 rolebinding_rules，则根据参数中的 namespace 打印角色绑定规则
elif args.rolebinding_rules:
    if args.namespace:
        print_rolebinding_rules(args.rolebinding_rules, args.namespace)
    else:
        print("Namespace was not specified")
# 如果参数中包含 clusterrolebinding_rules，则打印集群角色绑定规则
elif args.clusterrolebinding_rules:
    print_clusterrolebinding_rules(args.clusterrolebinding_rules)

# 打印左对齐的表格
def print_table_aligned_left(table):
    global json_filename
    # 如果给定的 JSON 文件名不为空，则将表格导出为 JSON 文件
    if json_filename != "":
        export_to_json(table, json_filename)
    # 设置全局变量 output_file
    global output_file
    # 如果设置了 no_color 标志，则移除表格中的颜色信息
    if no_color:
        # 使用正则表达式移除 ANSI 转义码
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        for row in table._rows:
            row[0] = ansi_escape.sub('', row[0])

    # 设置表格对齐方式为左对齐
    table.align = 'l'
    # 打印表格
    print(table)
    # 打印换行符
    print('\n')


def export_to_json(table, json_filename):
    # 移除字符串中的颜色信息
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    # 获取表格的列名
    headers = table.field_names
    # 创建空的数据字典
    data = {}
    # 初始化索引变量
    i = 0
    # 初始化结果字符串
    res = "["
# 遍历表格的行
for row in table._rows:
    # 遍历行中的实体
    for entity in row:
        # 创建一个不带颜色的实体副本
        entity_without_color = entity
        # 如果表头是'Priority'，则从字符串中移除颜色
        if headers[i] == 'Priority':
            entity_without_color = ansi_escape.sub('', entity)
        # 更新数据字典，将表头和实体对应起来
        data.update({headers[i]: entity_without_color})
        i += 1
    # 将数据转换为 JSON 格式，并添加到结果字符串中
    res += json.dumps(data, indent=4) + ','
    i = 0
# 移除最后一个逗号，并添加右括号，形成完整的 JSON 数组
res = res[:-1] + ']'
# 打开 JSON 文件，写入结果字符串，并关闭文件
json_file = open(json_filename, 'w')
json_file.write(res)
json_file.close()

# 如果作为主程序运行，则调用 main 函数
if __name__ == '__main__':
    main()
```