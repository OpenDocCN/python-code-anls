# `.\DB-GPT-src\dbgpt\util\pd_utils.py`

```py
import math  # 导入数学库

import pandas as pd  # 导入 pandas 库，并用 pd 作为别名

# 定义函数 csv_colunm_foramt，用于格式化 CSV 列中的值
def csv_colunm_foramt(val):
    try:
        # 检查是否为缺失值，如果是则返回 NaN
        if pd.isna(val):
            return math.nan
        # 如果值中包含 "$"，则移除 "$" 和 "," 后转换为浮点数返回
        if str(val).find("$") >= 0:
            return float(val.replace("$", "").replace(",", ""))
        # 如果值中包含 "¥"，则移除 "¥" 和 "," 后转换为浮点数返回
        if str(val).find("¥") >= 0:
            return float(val.replace("¥", "").replace(",", ""))
        # 若以上条件均不满足，则返回原始值
        return val
    except ValueError:
        # 捕获值转换为浮点数时可能出现的 ValueError 异常，返回原始值
        return val
```