# `MetaGPT\metagpt\actions\write_prd_an.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/14 11:40
@Author  : alexanderwu
@File    : write_prd_an.py
"""
# 导入所需的模块
from typing import List
# 导入自定义模块
from metagpt.actions.action_node import ActionNode
from metagpt.logs import logger

# 定义各种节点，包括关键信息、预期类型、指导和示例
LANGUAGE = ActionNode(
    key="Language",
    expected_type=str,
    instruction="Provide the language used in the project, typically matching the user's requirement language.",
    example="en_us",
)

# ... 其他节点的定义

# 将所有节点组成一个列表
NODES = [
    LANGUAGE,
    PROGRAMMING_LANGUAGE,
    # ... 其他节点
]

# 将节点列表组成一个新的节点
WRITE_PRD_NODE = ActionNode.from_children("WritePRD", NODES)
WP_ISSUE_TYPE_NODE = ActionNode.from_children("WP_ISSUE_TYPE", [ISSUE_TYPE, REASON])
WP_IS_RELATIVE_NODE = ActionNode.from_children("WP_IS_RELATIVE", [IS_RELATIVE, REASON])

# 主函数
def main():
    # 编译节点，生成提示信息
    prompt = WRITE_PRD_NODE.compile(context="")
    logger.info(prompt)

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()

```