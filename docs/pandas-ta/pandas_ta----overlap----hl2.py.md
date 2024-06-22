# `.\pandas-ta\pandas_ta\overlap\hl2.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 hl2，计算 HL2 指标
def hl2(high, low, offset=None, **kwargs):
    """Indicator: HL2 """
    # 验证参数
    # 确保 high 和 low 是有效的序列数据
    high = verify_series(high)
    low = verify_series(low)
    # 获取偏移量
    offset = get_offset(offset)

    # 计算结果
    # HL2 指标的计算公式为 (high + low) / 2
    hl2 = 0.5 * (high + low)

    # 偏移
    # 如果偏移量不为 0，则对 hl2 进行偏移
    if offset != 0:
        hl2 = hl2.shift(offset)

    # 名称和类别
    # 设置 hl2 的名称为 "HL2"，类别为 "overlap"
    hl2.name = "HL2"
    hl2.category = "overlap"

    return hl2
```