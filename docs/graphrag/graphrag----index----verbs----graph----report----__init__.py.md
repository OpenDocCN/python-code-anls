# `.\graphrag\graphrag\index\verbs\graph\report\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 当前文件为索引引擎图形报告包的根目录

# 从当前目录导入以下模块和函数
from .create_community_reports import (
    CreateCommunityReportsStrategyType,  # 导入创建社区报告策略类型
    create_community_reports,  # 导入创建社区报告函数
)
from .prepare_community_reports import prepare_community_reports  # 导入准备社区报告函数
from .prepare_community_reports_claims import prepare_community_reports_claims  # 导入准备社区声明报告函数
from .prepare_community_reports_edges import prepare_community_reports_edges  # 导入准备社区边缘报告函数
from .prepare_community_reports_nodes import prepare_community_reports_nodes  # 导入准备社区节点报告函数
from .restore_community_hierarchy import restore_community_hierarchy  # 导入恢复社区层次结构函数

# 定义了可以通过 'from package import *' 导入的所有公共接口
__all__ = [
    "CreateCommunityReportsStrategyType",  # 创建社区报告策略类型
    "create_community_reports",  # 创建社区报告函数
    "create_community_reports",  # 创建社区报告函数（此处重复了一次）
    "prepare_community_reports",  # 准备社区报告函数
    "prepare_community_reports_claims",  # 准备社区声明报告函数
    "prepare_community_reports_edges",  # 准备社区边缘报告函数
    "prepare_community_reports_nodes",  # 准备社区节点报告函数
    "restore_community_hierarchy",  # 恢复社区层次结构函数
]
```