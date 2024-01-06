# `kubehunter\kube_hunter\__init__.py`

```
# 从当前目录中导入 core 和 modules 模块
from . import core
from . import modules
# 将 core 和 modules 模块添加到 __all__ 列表中，使其在被导入时可以被访问
__all__ = [core, modules]
```