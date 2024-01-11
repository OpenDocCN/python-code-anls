# `kubehunter\kube_hunter\modules\discovery\__init__.py`

```
# 从当前目录中导入以下模块
from . import (
    apiserver,  # 导入 apiserver 模块
    dashboard,  # 导入 dashboard 模块
    etcd,       # 导入 etcd 模块
    hosts,      # 导入 hosts 模块
    kubectl,    # 导入 kubectl 模块
    kubelet,    # 导入 kubelet 模块
    ports,      # 导入 ports 模块
    proxy,      # 导入 proxy 模块
)

# 将所有导入的模块放入 __all__ 列表中
__all__ = [
    apiserver,  # 将 apiserver 模块加入 __all__ 列表
    dashboard,  # 将 dashboard 模块加入 __all__ 列表
    etcd,       # 将 etcd 模块加入 __all__ 列表
    hosts,      # 将 hosts 模块加入 __all__ 列表
    kubectl,    # 将 kubectl 模块加入 __all__ 列表
    kubelet,    # 将 kubelet 模块加入 __all__ 列表
    ports,      # 将 ports 模块加入 __all__ 列表
    proxy,      # 将 proxy 模块加入 __all__ 列表
]
```