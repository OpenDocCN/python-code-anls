# `kubehunter\kube_hunter\modules\hunting\__init__.py`

```
# 从当前目录中导入指定模块
from . import (
    aks,  # 导入名为 aks 的模块
    apiserver,  # 导入名为 apiserver 的模块
    arp,  # 导入名为 arp 的模块
    capabilities,  # 导入名为 capabilities 的模块
    certificates,  # 导入名为 certificates 的模块
    cves,  # 导入名为 cves 的模块
    dashboard,  # 导入名为 dashboard 的模块
    dns,  # 导入名为 dns 的模块
    etcd,  # 导入名为 etcd 的模块
    kubelet,  # 导入名为 kubelet 的模块
    mounts,  # 导入名为 mounts 的模块
    proxy,  # 导入名为 proxy 的模块
    secrets,  # 导入名为 secrets 的模块
)

# 将所有导入的模块放入 __all__ 列表中
__all__ = [
    aks,  # 将 aks 模块添加到 __all__ 列表中
    apiserver,  # 将 apiserver 模块添加到 __all__ 列表中
    arp,  # 将 arp 模块添加到 __all__ 列表中
    capabilities,  # 变量，可能用于存储系统功能的信息
    certificates,  # 变量，可能用于存储证书信息
    cves,  # 变量，可能用于存储漏洞信息
    dashboard,  # 变量，可能用于存储仪表板信息
    dns,  # 变量，可能用于存储 DNS 相关信息
    etcd,  # 变量，可能用于存储 etcd 相关信息
    kubelet,  # 变量，可能用于存储 kubelet 相关信息
    mounts,  # 变量，可能用于存储挂载信息
    proxy,  # 变量，可能用于存储代理信息
    secrets,  # 变量，可能用于存储密钥信息
]
```