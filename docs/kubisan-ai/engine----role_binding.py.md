# `KubiScan\engine\role_binding.py`

```

# 这个类也是用于 ClusterRoleBinding
class RoleBinding:
    # 初始化方法，设置角色绑定的属性
    def __init__(self, name, priority, namespace=None, kind=None, subjects=None, time=None):
        # 设置角色绑定的名称
        self.name = name
        # 设置角色绑定的优先级
        self.priority = priority
        # 设置角色绑定的命名空间
        self.namespace = namespace
        # 设置角色绑定的类型
        self.kind = kind
        # 设置角色绑定的主体
        self.subjects = subjects
        # 设置角色绑定的时间
        self.time = time

```