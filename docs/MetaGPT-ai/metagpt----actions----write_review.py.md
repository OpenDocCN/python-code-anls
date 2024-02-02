# `MetaGPT\metagpt\actions\write_review.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : alexanderwu
@File    : write_review.py
"""
# 导入需要的模块
from typing import List
from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode

# 定义REVIEW节点，用于给出评论
REVIEW = ActionNode(
    key="Review",
    expected_type=List[str],
    instruction="Act as an experienced Reviewer and review the given output. Ask a series of critical questions, "
    "concisely and clearly, to help the writer improve their work.",
    example=[
        "This is a good PRD, but I think it can be improved by adding more details.",
    ],
)

# 定义LGTM节点，用于判断输出是否符合要求
LGTM = ActionNode(
    key="LGTM",
    expected_type=str,
    instruction="LGTM/LBTM. If the output is good enough, give a LGTM (Looks Good To Me) to the writer, "
    "else LBTM (Looks Bad To Me).",
    example="LGTM",
)

# 定义WRITE_REVIEW_NODE节点，包含REVIEW和LGTM节点
WRITE_REVIEW_NODE = ActionNode.from_children("WRITE_REVIEW_NODE", [REVIEW, LGTM])

# 定义WriteReview类，用于写评论
class WriteReview(Action):
    """Write a review for the given context."""

    name: str = "WriteReview"

    # 异步运行函数，填充WRITE_REVIEW_NODE节点
    async def run(self, context):
        return await WRITE_REVIEW_NODE.fill(context=context, llm=self.llm, schema="json")

```