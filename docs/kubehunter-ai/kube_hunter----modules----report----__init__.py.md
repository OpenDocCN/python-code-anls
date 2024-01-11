# `kubehunter\kube_hunter\modules\report\__init__.py`

```
# 从 kube_hunter.modules.report.factory 模块中导入 get_reporter 和 get_dispatcher 函数
from kube_hunter.modules.report.factory import get_reporter, get_dispatcher
# 将 get_reporter 和 get_dispatcher 函数添加到 __all__ 列表中，表示它们可以被外部导入
__all__ = [get_reporter, get_dispatcher]
```