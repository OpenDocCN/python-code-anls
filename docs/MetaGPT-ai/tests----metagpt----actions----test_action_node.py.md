# `MetaGPT\tests\metagpt\actions\test_action_node.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/23 15:49
@Author  : alexanderwu
@File    : test_action_node.py
"""
# 导入所需的模块
from typing import List, Tuple
import pytest
from pydantic import ValidationError
from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode
from metagpt.environment import Environment
from metagpt.llm import LLM
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

# 测试函数，测试两个角色的辩论
@pytest.mark.asyncio
async def test_debate_two_roles():
    # 创建两个动作对象
    action1 = Action(name="AlexSay", instruction="Express your opinion with emotion and don't repeat it")
    action2 = Action(name="BobSay", instruction="Express your opinion with emotion and don't repeat it")
    # 创建两个角色对象
    biden = Role(
        name="Alex", profile="Democratic candidate", goal="Win the election", actions=[action1], watch=[action2]
    )
    trump = Role(
        name="Bob", profile="Republican candidate", goal="Win the election", actions=[action2], watch=[action1]
    )
    # 创建环境对象
    env = Environment(desc="US election live broadcast")
    # 创建团队对象
    team = Team(investment=10.0, env=env, roles=[biden, trump])

    # 运行团队的辩论，返回历史记录
    history = await team.run(idea="Topic: climate change. Under 80 words per message.", send_to="Alex", n_round=3)
    # 断言检查历史记录中是否包含 "Alex"
    assert "Alex" in history

# 其他测试函数省略...

# 创建一个字典对象，包含各种数据类型
t_dict = {
    "Required Python third-party packages": '"""\nflask==1.1.2\npygame==2.0.1\n"""\n',
    "Required Other language third-party packages": '"""\nNo third-party packages required for other languages.\n"""\n',
    "Full API spec": '"""\nopenapi: 3.0.0\ninfo:\n  title: Web Snake Game API\n  version: 1.0.0\npaths:\n  /game:\n    get:\n      summary: Get the current game state\n      responses:\n        \'200\':\n          description: A JSON object of the game state\n    post:\n      summary: Send a command to the game\n      requestBody:\n        required: true\n        content:\n          application/json:\n            schema:\n              type: object\n              properties:\n                command:\n                  type: string\n      responses:\n        \'200\':\n          description: A JSON object of the updated game state\n"""\n',
    "Logic Analysis": [
        ["app.py", "Main entry point for the Flask application. Handles HTTP requests and responses."],
        ["game.py", "Contains the Game and Snake classes. Handles the game logic."],
        ["static/js/script.js", "Handles user interactions and updates the game UI."],
        ["static/css/styles.css", "Defines the styles for the game UI."],
        ["templates/index.html", "The main page of the web application. Displays the game UI."],
    ],
    "Task list": ["game.py", "app.py", "static/css/styles.css", "static/js/script.js", "templates/index.html"],
    "Shared Knowledge": "\"\"\"\n'game.py' contains the Game and Snake classes which are responsible for the game logic. The Game class uses an instance of the Snake class.\n\n'app.py' is the main entry point for the Flask application. It creates an instance of the Game class and handles HTTP requests and responses.\n\n'static/js/script.js' is responsible for handling user interactions and updating the game UI based on the game state returned by 'app.py'.\n\n'static/css/styles.css' defines the styles for the game UI.\n\n'templates/index.html' is the main page of the web application. It displays the game UI and loads 'static/js/script.js' and 'static/css/styles.css'.\n\"\"\"\n",
    "Anything UNCLEAR": "We need clarification on how the high score should be stored. Should it persist across sessions (stored in a database or a file) or should it reset every time the game is restarted? Also, should the game speed increase as the snake grows, or should it remain constant throughout the game?",
}

# 创建一个字典对象，包含部分数据
t_dict_min = {
    "Required Python third-party packages": '"""\nflask==1.1.2\npygame==2.0.1\n"""\n',
}

# 创建一个字典对象，用于验证数据类型
WRITE_TASKS_OUTPUT_MAPPING = {
    "Required Python third-party packages": (str, ...),
    "Required Other language third-party packages": (str, ...),
    "Full API spec": (str, ...),
    "Logic Analysis": (List[Tuple[str, str]], ...),
    "Task list": (List[str], ...),
    "Shared Knowledge": (str, ...),
    "Anything UNCLEAR": (str, ...),
}

# 创建一个字典对象，包含部分数据，用于验证数据类型
WRITE_TASKS_OUTPUT_MAPPING_MISSING = {
    "Required Python third-party packages": (str, ...),
}

# 测试函数，创建模型类
def test_create_model_class():
    # 创建一个模型类
    test_class = ActionNode.create_model_class("test_class", WRITE_TASKS_OUTPUT_MAPPING)
    # 断言检查模型类的名称
    assert test_class.__name__ == "test_class"
    # 使用字典数据创建模型对象
    output = test_class(**t_dict)
    # 打印模型对象的模式
    print(output.schema())
    # 断言检查模型对象的模式
    assert output.schema()["title"] == "test_class"
    assert output.schema()["type"] == "object"
    assert output.schema()["properties"]["Full API spec"]

# 其他测试函数省略...

# 如果是主程序，则执行测试函数
if __name__ == "__main__":
    test_create_model_class()
    test_create_model_class_with_mapping()

```