# `KubiScan\engine\role.py`

```

# 导入kubernetes模块中的client和config
from kubernetes import client, config

# 定义一个Role类，用于表示ClusterRole
class Role:
    # 初始化方法，设置角色的名称、优先级、规则、命名空间、类型和时间
    def __init__(self, name, priority, rules=None, namespace=None, kind=None, time=None):
        self.name = name
        self.priority = priority
        self.rules = rules
        self.namespace = namespace
        self.kind = kind
        self.time = time

    # 获取角色的规则
    def get_rules(self):
        # 加载Kubernetes配置
        config.load_kube_config()
        # 创建RbacAuthorizationV1Api对象
        v1 = client.RbacAuthorizationV1Api()
        # 如果角色类型是"role"，则返回指定命名空间中的角色规则
        if self.kind.lower() == "role":
            return (v1.read_namespaced_role(self.name, self.namespace)).rules
        else: # 如果角色类型是"clusterrole"，则返回集群中的角色规则
            return (v1.read_cluster_role(self.name)).rules

```