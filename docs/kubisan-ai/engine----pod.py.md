# `KubiScan\engine\pod.py`

```

# 创建一个名为Pod的类
class Pod:
    # 初始化方法，接受name、namespace和containers三个参数
    def __init__(self, name, namespace, containers):
        # 将传入的name赋值给实例变量self.name
        self.name = name
        # 将传入的namespace赋值给实例变量self.namespace
        self.namespace = namespace
        # 将传入的containers赋值给实例变量self.containers
        self.containers = containers

```