# `KubiScan\static_risky_roles.py`

```
# 从 engine.role 模块中导入 Role 类
from engine.role import Role
# 从 engine.rule 模块中导入 Rule 类
from engine.rule import Rule
# 从 engine.priority 模块中导入 get_priority_by_name 函数
from engine.priority import get_priority_by_name
# 从 misc.constants 模块中导入所有常量
from misc.constants import *
# 导入 yaml 模块
import yaml
# 导入 os 模块

# 创建一个空列表，用于存储静态风险角色
STATIC_RISKY_ROLES = []

# 从 YAML 文件中读取角色信息，并设置为静态风险角色
def set_risky_roles_from_yaml(items):
    # 遍历角色列表
    for role in items:
        rules = []
        # 遍历每个角色的规则列表
        for rule in role['rules']:
            # 创建 Rule 对象并添加到规则列表中
            rule_obj = Rule(resources=rule['resources'], verbs=rule['verbs'])
            rules.append(rule_obj)

        # 创建 Role 对象并添加到静态风险角色列表中
        STATIC_RISKY_ROLES.append(Role(role['metadata']['name'],
                                       get_priority_by_name(role['metadata']['priority']),
                                       rules,
                                       namespace=RISKY_NAMESPACE)
# 打开指定路径下的 risky_roles.yaml 文件，以只读模式
with open(os.path.dirname(os.path.realpath(__file__)) + '/risky_roles.yaml', 'r') as stream:
    # 尝试安全加载 YAML 文件内容
    try:
        # 将 YAML 文件内容加载到 loaded_yaml 变量中
        loaded_yaml = yaml.safe_load(stream)
        # 从加载的 YAML 中设置风险角色
        set_risky_roles_from_yaml(loaded_yaml['items'])
    # 如果出现 YAML 错误，则打印错误信息
    except yaml.YAMLError as exc:
        print(exc)
```