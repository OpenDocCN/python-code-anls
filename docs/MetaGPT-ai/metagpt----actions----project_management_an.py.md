# `MetaGPT\metagpt\actions\project_management_an.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/14 15:28
@Author  : alexanderwu
@File    : project_management_an.py
"""
# 导入所需的模块
from typing import List
from metagpt.actions.action_node import ActionNode
from metagpt.logs import logger

# 定义必需的 Python 包
REQUIRED_PYTHON_PACKAGES = ActionNode(
    key="Required Python packages",
    expected_type=List[str],
    instruction="Provide required Python packages in requirements.txt format.",
    example=["flask==1.1.2", "bcrypt==3.2.0"],
)

# 定义其他语言的第三方包
REQUIRED_OTHER_LANGUAGE_PACKAGES = ActionNode(
    key="Required Other language third-party packages",
    expected_type=List[str],
    instruction="List down the required packages for languages other than Python.",
    example=["No third-party dependencies required"],
)

# 定义逻辑分析
LOGIC_ANALYSIS = ActionNode(
    key="Logic Analysis",
    expected_type=List[List[str]],
    instruction="Provide a list of files with the classes/methods/functions to be implemented, "
    "including dependency analysis and imports.",
    example=[
        ["game.py", "Contains Game class and ... functions"],
        ["main.py", "Contains main function, from game import Game"],
    ],
)

# 定义任务列表
TASK_LIST = ActionNode(
    key="Task list",
    expected_type=List[str],
    instruction="Break down the tasks into a list of filenames, prioritized by dependency order.",
    example=["game.py", "main.py"],
)

# 定义完整的 API 规范
FULL_API_SPEC = ActionNode(
    key="Full API spec",
    expected_type=str,
    instruction="Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end "
    "and back-end communication is not required, leave it blank.",
    example="openapi: 3.0.0 ...",
)

# 定义共享知识
SHARED_KNOWLEDGE = ActionNode(
    key="Shared Knowledge",
    expected_type=str,
    instruction="Detail any shared knowledge, like common utility functions or configuration variables.",
    example="'game.py' contains functions shared across the project.",
)

# 定义项目管理中不清楚的事项
ANYTHING_UNCLEAR_PM = ActionNode(
    key="Anything UNCLEAR",
    expected_type=str,
    instruction="Mention any unclear aspects in the project management context and try to clarify them.",
    example="Clarification needed on how to start and initialize third-party libraries.",
)

# 将所有定义的节点组成一个列表
NODES = [
    REQUIRED_PYTHON_PACKAGES,
    REQUIRED_OTHER_LANGUAGE_PACKAGES,
    LOGIC_ANALYSIS,
    TASK_LIST,
    FULL_API_SPEC,
    SHARED_KNOWLEDGE,
    ANYTHING_UNCLEAR_PM,
]

# 将节点列表组合成一个父节点
PM_NODE = ActionNode.from_children("PM_NODE", NODES)

# 主函数
def main():
    # 编译父节点的内容
    prompt = PM_NODE.compile(context="")
    # 输出日志
    logger.info(prompt)

# 程序入口
if __name__ == "__main__":
    main()

```