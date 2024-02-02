# `KubiScan\engine\pod.py`

```py
# TODO: 添加一个优先级字段，该字段将从容器中具有最高优先级
class Pod:
    def __init__(self, name, namespace, containers):
        # 初始化 Pod 对象的名称
        self.name = name
        # 初始化 Pod 对象的命名空间
        self.namespace = namespace
        # 初始化 Pod 对象的容器列表
        self.containers = containers
```