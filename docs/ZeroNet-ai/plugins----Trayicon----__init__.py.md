# `ZeroNet\plugins\Trayicon\__init__.py`

```py
# 导入 sys 模块
import sys

# 如果操作系统是 Windows，则导入 TrayiconPlugin 模块
if sys.platform == 'win32':
    from . import TrayiconPlugin
```