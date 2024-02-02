# `KubiScan\engine\rule.py`

```py
# 定义一个类 Rule
class Rule:
    # 初始化方法，接受动词、资源和资源名称作为参数
    def __init__(self, verbs, resources, resource_names=None):
        # 将传入的动词赋值给实例变量
        self.verbs = verbs
        # 将传入的资源赋值给实例变量
        self.resources = resources
        # 如果传入了资源名称，则赋值给实例变量，否则为 None
        self.resource_names = resource_names
```