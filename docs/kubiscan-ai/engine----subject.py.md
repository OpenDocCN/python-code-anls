# `KubiScan\engine\subject.py`

```
# 定义一个主题类，用于表示三种类型的主题：用户、组和服务账户
# 参考链接：https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Subject.md
class Subject:
    # 初始化方法，接受原始信息和优先级作为参数
    def __init__(self, raw_info, priority):
        # 将原始信息存储在用户信息属性中
        self.user_info = raw_info
        # 将优先级存储在优先级属性中
        self.priority = priority
```