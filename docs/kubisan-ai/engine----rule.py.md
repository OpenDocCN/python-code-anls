# `KubiScan\engine\rule.py`

```

# 创建一个名为Rule的类，用于表示权限规则
class Rule:
    # 初始化方法，接受动词、资源和资源名称作为参数
    def __init__(self, verbs, resources, resource_names=None):
        # 将传入的动词赋值给实例变量verbs
        self.verbs = verbs
        # 将传入的资源赋值给实例变量resources
        self.resources = resources
        # 将传入的资源名称赋值给实例变量resource_names，如果没有传入资源名称则默认为None
        self.resource_names = resource_names

```