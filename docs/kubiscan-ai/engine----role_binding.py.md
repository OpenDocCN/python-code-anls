# `KubiScan\engine\role_binding.py`

```py
# 这个类也是用于 ClusterRoleBinding
class RoleBinding:
    # 初始化方法，设置角色绑定的属性
    def __init__(self, name, priority, namespace=None, kind=None, subjects=None, time=None):
        # 设置角色绑定的名称
        self.name = name
        # 设置角色绑定的优先级
        self.priority = priority
        # 设置角色绑定的命名空间，可选参数
        self.namespace = namespace
        # 设置角色绑定的类型，可选参数
        self.kind = kind
        # 设置角色绑定的主体，可选参数
        self.subjects = subjects
        # 设置角色绑定的时间，可选参数
        self.time = time
```