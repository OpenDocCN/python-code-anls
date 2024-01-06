# `KubiScan\engine\pod.py`

```
# 创建一个名为Pod的类，用于表示一个容器组
# 初始化方法，接受名称、命名空间和容器列表作为参数
class Pod:
    def __init__(self, name, namespace, containers):
        # 设置名称属性
        self.name = name
        # 设置命名空间属性
        self.namespace = namespace
        # 设置容器列表属性
        self.containers = containers
```