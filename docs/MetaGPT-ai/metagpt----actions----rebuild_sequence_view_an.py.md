# `MetaGPT\metagpt\actions\rebuild_sequence_view_an.py`

```

#!/usr/bin/env python
# 指定解释器为 python

# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8

"""
@Time    : 2024/1/4
@Author  : mashenquan
@File    : rebuild_sequence_view_an.py
"""
# 文件注释，包括时间、作者、文件名等信息

# 导入 ActionNode 类
from metagpt.actions.action_node import ActionNode
# 导入 MMC2 类
from metagpt.utils.mermaid import MMC2

# 创建 ActionNode 对象，用于将代码转换为 Mermaid 序列图
CODE_2_MERMAID_SEQUENCE_DIAGRAM = ActionNode(
    key="Program call flow",
    expected_type=str,
    instruction='Translate the "context" content into "format example" format.',
    example=MMC2,
)

```