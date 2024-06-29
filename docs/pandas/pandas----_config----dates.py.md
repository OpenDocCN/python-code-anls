# `D:\src\scipysrc\pandas\pandas\_config\dates.py`

```
"""
config for datetime formatting
"""

# 从未来导入注解以支持类型提示
from __future__ import annotations

# 导入 pandas 库的配置对象作为 cf
from pandas._config import config as cf

# 日期格式化中日期优先选项的文档字符串
pc_date_dayfirst_doc = """
: boolean
    When True, prints and parses dates with the day first, eg 20/01/2005
"""

# 日期格式化中年份优先选项的文档字符串
pc_date_yearfirst_doc = """
: boolean
    When True, prints and parses dates with the year first, eg 2005/01/20
"""

# 在配置对象的"display"前缀下注册日期优先选项，初始值为 False，
# 文档字符串为 pc_date_dayfirst_doc，验证器为 cf.is_bool
with cf.config_prefix("display"):
    cf.register_option(
        "date_dayfirst", False, pc_date_dayfirst_doc, validator=cf.is_bool
    )
    # 在配置对象的"display"前缀下注册年份优先选项，初始值为 False，
    # 文档字符串为 pc_date_yearfirst_doc，验证器为 cf.is_bool
    cf.register_option(
        "date_yearfirst", False, pc_date_yearfirst_doc, validator=cf.is_bool
    )
```