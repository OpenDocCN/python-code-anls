# `.\graphrag\graphrag\index\verbs\unzip.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""包含解压方法定义的模块。"""

# 引入必要的模块和类型提示
from typing import cast

import pandas as pd
from datashaper import TableContainer, VerbInput, verb


# TODO: 检查这是否已经存在
# 接受 1|(x,y)|b 形式的输入
# 并转换为
# 1|x|y|b
@verb(name="unzip")
def unzip(
    input: VerbInput, column: str, to: list[str], **_kwargs: dict
) -> TableContainer:
    """解压包含元组的列到多个列中。"""
    # 将输入转换为 pandas DataFrame
    table = cast(pd.DataFrame, input.get_input())

    # 将包含元组的列拆分为多个列，并添加到数据框中
    table[to] = pd.DataFrame(table[column].tolist(), index=table.index)

    # 返回处理后的数据表
    return TableContainer(table=table)
```