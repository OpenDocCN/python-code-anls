# `kubehunter\kube_hunter\modules\__init__.py`

```
# 从当前目录中导入 report、discovery 和 hunting 模块
from . import report
from . import discovery
from . import hunting
# 将 report、discovery 和 hunting 模块添加到 __all__ 列表中，以便在使用 import * 时被导入
__all__ = [report, discovery, hunting]
```