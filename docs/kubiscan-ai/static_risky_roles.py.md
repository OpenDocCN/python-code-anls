# `KubiScan\static_risky_roles.py`

```py
# 从 engine.role 模块中导入 Role 类
from engine.role import Role
# 从 engine.rule 模块中导入 Rule 类
from engine.rule import Rule
# 从 engine.priority 模块中导入 get_priority_by_name 函数
from engine.priority import get_priority_by_name
# 从 misc.constants 模块中导入所有内容
from misc.constants import *
# 导入 yaml 模块
import yaml
# 导入 os 模块

# 定义一个空列表 STATIC_RISKY_ROLES
STATIC_RISKY_ROLES = []

# 定义一个函数，从 YAML 文件中设置风险角色
def set_risky_roles_from_yaml(items):
    # 遍历传入的角色列表
    for role in items:
        # 定义一个空列表 rules
        rules = []
        # 遍历每个角色的规则
        for rule in role['rules']:
            # 创建 Rule 对象并添加到 rules 列表中
            rule_obj = Rule(resources=rule['resources'], verbs=rule['verbs'])
            rules.append(rule_obj)

            # 创建 Role 对象并添加到 STATIC_RISKY_ROLES 列表中
            STATIC_RISKY_ROLES.append(Role(role['metadata']['name'],
                                           get_priority_by_name(role['metadata']['priority']),
                                           rules,
                                           namespace=RISKY_NAMESPACE)
                                      )

# 打开 risky_roles.yaml 文件
with open(os.path.dirname(os.path.realpath(__file__)) + '/risky_roles.yaml', 'r') as stream:
    try:
        # 加载 YAML 文件内容
        loaded_yaml = yaml.safe_load(stream)
        # 调用 set_risky_roles_from_yaml 函数，传入加载的 YAML 内容中的 items
        set_risky_roles_from_yaml(loaded_yaml['items'])
    # 捕获 YAML 解析错误并打印异常信息
    except yaml.YAMLError as exc:
        print(exc)
```