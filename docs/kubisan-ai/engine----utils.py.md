# `KubiScan\engine\utils.py`

```

# 导入requests库
import requests

# 从engine.role中导入Role类
from engine.role import Role
# 从engine.priority中导入Priority类
from engine.priority import Priority
# 从static_risky_roles中导入STATIC_RISKY_ROLES列表
from static_risky_roles import STATIC_RISKY_ROLES
# 从engine.role_binding中导入RoleBinding类
from engine.role_binding import RoleBinding
# 从kubernetes.stream中导入stream函数
from kubernetes.stream import stream
# 从engine.pod中导入Pod类
from engine.pod import Pod
# 从engine.container中导入Container类
from engine.container import Container
# 导入json库
import json
# 从api中导入api_client
from api import api_client
# 从engine.subject中导入Subject类
from engine.subject import Subject
# 从misc.constants中导入常量
from misc.constants import *
# 从kubernetes.client.rest中导入ApiException类
from kubernetes.client.rest import ApiException
# 导入urllib3库
import urllib3
# 禁用urllib3的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# region - Roles and ClusteRoles

# 创建一个空列表list_of_service_accounts
list_of_service_accounts = []

# 定义函数is_risky_resource_name_exist，接受source_rolename和source_resourcenames两个参数
def is_risky_resource_name_exist(source_rolename, source_resourcenames):
    # 初始化is_risky为False
    is_risky = False
    # 遍历source_resourcenames列表
    for resource_name in source_resourcenames:
        # 防止循环
        if resource_name != source_rolename:
            # TODO: 需要允许此检查也适用于'roles'资源名称，应考虑命名空间...
            # 调用get_role_by_name_and_kind函数获取role对象
            role = get_role_by_name_and_kind(resource_name, CLUSTER_ROLE_KIND)
            # 如果role对象不为空
            if role is not None:
                # 调用is_risky_role函数判断role是否为高风险
                is_risky, priority = is_risky_role(role)
                # 如果role为高风险
                if is_risky:
                    # 结束循环
                    break
    # 返回is_risky
    return is_risky

# 定义函数is_rule_contains_risky_rule，接受source_role_name、source_rule和risky_rule三个参数
def is_rule_contains_risky_rule(source_role_name, source_rule, risky_rule):
    # 初始化is_contains为True
    is_contains = True
    # 初始化is_bind_verb_found为False
    is_bind_verb_found = False
    # 初始化is_role_resource_found为False
    is_role_resource_found = False

    # 可选：取消注释并将所有内容移至'return'之前以添加具有"*"的动词或资源的任何规则。
    # 目前它在risky_roles.yaml中部分处理
    # 如果source_rule.verbs不为空且"*"不在source_rule.verbs中，且source_rule.resources不为空且"*"不在source_rule.resources中
    # for循环遍历risky_rule.verbs列表
    for verb in risky_rule.verbs:
        # 如果verb不在source_rule.verbs中
        if verb not in source_rule.verbs:
            # 将is_contains设置为False
            is_contains = False
            # 结束循环
            break
        # 如果verb转换为小写后等于"bind"
        if verb.lower() == "bind":
            # 将is_bind_verb_found设置为True
            is_bind_verb_found = True

    # 如果is_contains为True且source_rule.resources不为空
    if is_contains and source_rule.resources is not None:
        # for循环遍历risky_rule.resources列表
        for resource in risky_rule.resources:
            # 如果resource不在source_rule.resources中
            if resource not in source_rule.resources:
                # 将is_contains设置为False
                is_contains = False
                # 结束循环
                break
            # 如果resource转换为小写后等于"roles"或"clusterroles"
            if resource.lower() == "roles" or resource.lower() == "clusterroles":
                # 将is_role_resource_found设置为True
                is_role_resource_found = True

        # 如果is_contains为True且risky_rule.resource_names不为空
        if is_contains and risky_rule.resource_names is not None:
            # 将is_contains设置为False
            is_contains = False
            # 如果is_bind_verb_found为True且is_role_resource_found为True
            if is_bind_verb_found and is_role_resource_found:
                # 调用is_risky_resource_name_exist函数判断source_role_name和source_rule.resource_names是否为高风险
                is_risky = is_risky_resource_name_exist(source_role_name, source_rule.resource_names)
                # 如果is_risky为True
                if is_risky:
                    # 将is_contains设置为True
                    is_contains = True
    else:
        # 将is_contains设置为False
        is_contains = False

    # 返回is_contains
    return is_contains

# ... (后续代码略)

```