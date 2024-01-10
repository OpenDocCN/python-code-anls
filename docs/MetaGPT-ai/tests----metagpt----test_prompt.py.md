# `MetaGPT\tests\metagpt\test_prompt.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:45
@Author  : alexanderwu
@File    : test_llm.py
"""

import pytest

from metagpt.llm import LLM

# 定义一个常量，包含一个JavaScript代码片段，用于测试
CODE_REVIEW_SMALLEST_CONTEXT = """
## game.js

// game.js
class Game {
    constructor() {
        this.board = this.createEmptyBoard();
        this.score = 0;
        this.bestScore = 0;
    }
    ...
}


"""

# 定义另一个常量，包含另一个JavaScript代码片段，用于测试
MOVE_FUNCTION = """
## move function implementation


move(direction) {
    let moved = false;
    switch (direction) {
        case 'up':
            ...
            break;
        case 'down':
            ...
            break;
        case 'left':
            ...
            break;
        case 'right':
            ...
            break;
    }

    if (moved) {
        this.addRandomTile();
    }
}

"""

# 定义一个常量，包含一个Python类的代码片段，用于测试
FUNCTION_TO_MERMAID_CLASS = """
## context

class UIDesign(Action):
    ...
}

-----
## format example
[CONTENT]
{
    "ClassView": "classDiagram\n        class A {\n        -int x\n        +int y\n        -int speed\n        -int direction\n        +__init__(x: int, y: int, speed: int, direction: int)\n        +change_direction(new_direction: int) None\n        +move() None\n    }\n    "
}
[/CONTENT]
## nodes: "<node>: <type>  # <comment>"
- ClassView: <class 'str'>  # Generate the mermaid class diagram corresponding to source code in "context."
## constraint
- Language: Please use the same language as the user input.
- Format: output wrapped inside [CONTENT][/CONTENT] as format example, nothing else.
## action
Fill in the above nodes(ClassView) based on the format example.
"""

# 定义一个pytest的fixture，用于测试LLM类
@pytest.fixture()
def llm():
    return LLM()

# 定义一个异步测试函数，用于测试LLM类的代码审查功能
@pytest.mark.asyncio
async def test_llm_code_review(llm):
    choices = [
        "Please review the move function code above. Should it be refactor?",
        "Please implement the move function",
        "Please write a draft for the move function in order to implement it",
    ]
    # prompt = CODE_REVIEW_SMALLEST_CONTEXT+ "\n\n" + MOVE_DRAFT + "\n\n" + choices[1]
    # rsp = await llm.aask(prompt)

    prompt = CODE_REVIEW_SMALLEST_CONTEXT + "\n\n" + MOVE_FUNCTION + "\n\n" + choices[0]
    prompt = FUNCTION_TO_MERMAID_CLASS

    _ = await llm.aask(prompt)

# if __name__ == "__main__":
#     pytest.main([__file__, "-s"])

```