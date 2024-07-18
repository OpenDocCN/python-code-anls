# `.\graphrag\graphrag\index\verbs\graph\report\prepare_community_reports_nodes.py`

```py
# 导入所需的模块和函数
from typing import cast  # 导入类型提示工具 cast

import pandas as pd  # 导入 pandas 库
from datashaper import TableContainer, VerbInput, verb  # 导入 datashaper 中的 TableContainer, VerbInput, verb 函数

from graphrag.index.graph.extractors.community_reports.schemas import (  # 从指定路径导入一系列变量
    NODE_DEGREE,  # 导入 NODE_DEGREE 变量
    NODE_DESCRIPTION,  # 导入 NODE_DESCRIPTION 变量
    NODE_DETAILS,  # 导入 NODE_DETAILS 变量
    NODE_ID,  # 导入 NODE_ID 变量
    NODE_NAME,  # 导入 NODE_NAME 变量
)

_MISSING_DESCRIPTION = "No Description"  # 定义一个缺失描述的默认字符串


@verb(name="prepare_community_reports_nodes")
def prepare_community_reports_nodes(
    input: VerbInput,  # 输入参数 input，类型为 VerbInput
    to: str = NODE_DETAILS,  # 参数 to，默认值为 NODE_DETAILS，目标列名
    id_column: str = NODE_ID,  # 参数 id_column，默认值为 NODE_ID，节点 ID 列名
    name_column: str = NODE_NAME,  # 参数 name_column，默认值为 NODE_NAME，节点名称列名
    description_column: str = NODE_DESCRIPTION,  # 参数 description_column，默认值为 NODE_DESCRIPTION，节点描述列名
    degree_column: str = NODE_DEGREE,  # 参数 degree_column，默认值为 NODE_DEGREE，节点度数列名
    **_kwargs,  # 其他未命名参数，存储在 _kwargs 中
) -> TableContainer:
    """Merge edge details into an object."""
    node_df = cast(pd.DataFrame, input.get_input())  # 将输入转换为 pandas DataFrame

    node_df = node_df.fillna(value={description_column: _MISSING_DESCRIPTION})  # 填充缺失值，使用默认描述字符串

    # 将四个列的值合并成一个字典，存放在目标列 to 中
    node_df[to] = node_df.apply(
        lambda x: {
            id_column: x[id_column],
            name_column: x[name_column],
            description_column: x[description_column],
            degree_column: x[degree_column],
        },
        axis=1,  # 按行应用 lambda 函数
    )

    return TableContainer(table=node_df)  # 返回 TableContainer 对象，其中包含更新后的 DataFrame
```