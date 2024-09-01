# `.\flux\src\flux\__init__.py`

```py
# 尝试从当前包的 `_version` 模块导入 `version` 和 `version_tuple`
try:
    from ._version import version as __version__  # type: ignore  # type: ignore 用于忽略类型检查器的警告
    from ._version import version_tuple
# 如果导入失败（模块不存在），则设置默认的版本信息
except ImportError:
    __version__ = "unknown (no version information available)"  # 设置版本号为未知
    version_tuple = (0, 0, "unknown", "noinfo")  # 设置版本元组为未知

# 导入 Path 类以便处理文件路径
from pathlib import Path

# 设置包的名称，将包名中的下划线替换为短横线
PACKAGE = __package__.replace("_", "-")
# 获取当前文件所在目录的路径
PACKAGE_ROOT = Path(__file__).parent
```