# `.\kubehunter\kube_hunter\modules\discovery\__init__.py`

```

# 从当前目录中导入以下模块
from . import (
    apiserver,
    dashboard,
    etcd,
    hosts,
    kubectl,
    kubelet,
    ports,
    proxy,
)

# 将以下模块添加到 __all__ 列表中，以便在使用 import * 时被导入
__all__ = [
    apiserver,
    dashboard,
    etcd,
    hosts,
    kubectl,
    kubelet,
    ports,
    proxy,
]

```