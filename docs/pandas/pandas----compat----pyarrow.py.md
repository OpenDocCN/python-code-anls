# `D:\src\scipysrc\pandas\pandas\compat\pyarrow.py`

```
"""support pyarrow compatibility across versions"""

# 引入将来版本的特性，确保在当前版本中也能兼容
from __future__ import annotations

# 从 pandas.util.version 中引入 Version 类
from pandas.util.version import Version

# 尝试导入 pyarrow 库
try:
    import pyarrow as pa

    # 从 pyarrow 版本字符串创建 Version 对象，并提取其基本版本信息
    _palv = Version(Version(pa.__version__).base_version)

    # 检查 pyarrow 版本是否小于特定版本，生成对应的布尔值
    pa_version_under10p1 = _palv < Version("10.0.1")
    pa_version_under11p0 = _palv < Version("11.0.0")
    pa_version_under12p0 = _palv < Version("12.0.0")
    pa_version_under13p0 = _palv < Version("13.0.0")
    pa_version_under14p0 = _palv < Version("14.0.0")
    pa_version_under14p1 = _palv < Version("14.0.1")
    pa_version_under15p0 = _palv < Version("15.0.0")
    pa_version_under16p0 = _palv < Version("16.0.0")
    pa_version_under17p0 = _palv < Version("17.0.0")

# 如果导入失败，将所有的 pyarrow 版本标记为低于指定版本
except ImportError:
    pa_version_under10p1 = True
    pa_version_under11p0 = True
    pa_version_under12p0 = True
    pa_version_under13p0 = True
    pa_version_under14p0 = True
    pa_version_under14p1 = True
    pa_version_under15p0 = True
    pa_version_under16p0 = True
    pa_version_under17p0 = True
```