# `MetaGPT\metagpt\actions\design_api_an.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/12 22:24
@Author  : alexanderwu
@File    : design_api_an.py
"""
# 导入所需的模块
from typing import List
from metagpt.actions.action_node import ActionNode
from metagpt.utils.mermaid import MMC1, MMC2

# 创建一个表示实现方法的 ActionNode 对象
IMPLEMENTATION_APPROACH = ActionNode(
    key="Implementation approach",
    expected_type=str,
    instruction="Analyze the difficult points of the requirements, select the appropriate open-source framework",
    example="We will ...",
)

# 创建一个表示项目名称的 ActionNode 对象
PROJECT_NAME = ActionNode(
    key="Project name", expected_type=str, instruction="The project name with underline", example="game_2048"
)

# 创建一个表示文件列表的 ActionNode 对象
FILE_LIST = ActionNode(
    key="File list",
    expected_type=List[str],
    instruction="Only need relative paths. ALWAYS write a main.py or app.py here",
    example=["main.py", "game.py"],
)

# 创建一个表示数据结构和接口的 ActionNode 对象
DATA_STRUCTURES_AND_INTERFACES = ActionNode(
    key="Data structures and interfaces",
    expected_type=str,
    instruction="Use mermaid classDiagram code syntax, including classes, method(__init__ etc.) and functions with type"
    " annotations, CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. "
    "The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design.",
    example=MMC1,
)

# 创建一个表示程序调用流程的 ActionNode 对象
PROGRAM_CALL_FLOW = ActionNode(
    key="Program call flow",
    expected_type=str,
    instruction="Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE "
    "accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.",
    example=MMC2,
)

# 创建一个表示不清楚的事项的 ActionNode 对象
ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",
    expected_type=str,
    instruction="Mention unclear project aspects, then try to clarify it.",
    example="Clarification needed on third-party API integration, ...",
)

# 将所有的 ActionNode 对象组成一个列表
NODES = [
    IMPLEMENTATION_APPROACH,
    # PROJECT_NAME,
    FILE_LIST,
    DATA_STRUCTURES_AND_INTERFACES,
    PROGRAM_CALL_FLOW,
    ANYTHING_UNCLEAR,
]

# 创建一个表示设计 API 的 ActionNode 对象
DESIGN_API_NODE = ActionNode.from_children("DesignAPI", NODES)

```