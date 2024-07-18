# `.\graphrag\graphrag\index\verbs\__init__.py`

```py
# 版权声明，标明版权归 Microsoft Corporation 所有，使用 MIT 许可证授权
# 导入依赖的模块和函数
from .covariates import extract_covariates  # 导入从 covariates 模块中的 extract_covariates 函数
from .entities import entity_extract, summarize_descriptions  # 导入从 entities 模块中的 entity_extract 和 summarize_descriptions 函数
from .genid import genid  # 导入从 genid 模块中的 genid 函数
from .graph import (  # 导入 graph 模块中的多个函数
    cluster_graph,
    create_community_reports,
    create_graph,
    embed_graph,
    layout_graph,
    merge_graphs,
    unpack_graph,
)
from .overrides import aggregate, concat, merge  # 导入从 overrides 模块中的 aggregate, concat, merge 函数
from .snapshot import snapshot  # 导入从 snapshot 模块中的 snapshot 函数
from .snapshot_rows import snapshot_rows  # 导入从 snapshot_rows 模块中的 snapshot_rows 函数
from .spread_json import spread_json  # 导入从 spread_json 模块中的 spread_json 函数
from .text import (  # 导入 text 模块中的多个函数
    chunk,
    text_embed,
    text_split,
    text_translate,
)
from .unzip import unzip  # 导入从 unzip 模块中的 unzip 函数
from .zip import zip_verb  # 导入从 zip 模块中的 zip_verb 函数

# 定义导出的模块成员列表，包括了这些函数和模块
__all__ = [
    "aggregate",
    "chunk",
    "cluster_graph",
    "concat",
    "create_community_reports",
    "create_graph",
    "embed_graph",
    "entity_extract",
    "extract_covariates",
    "genid",
    "layout_graph",
    "merge",
    "merge_graphs",
    "snapshot",
    "snapshot_rows",
    "spread_json",
    "summarize_descriptions",
    "text_embed",
    "text_split",
    "text_translate",
    "unpack_graph",
    "unzip",
    "zip_verb",
]
```