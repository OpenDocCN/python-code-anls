# `MetaGPT\examples\build_customized_agent.py`

```

"""
Filename: MetaGPT/examples/build_customized_agent.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
"""
# 导入必要的模块
import asyncio
import re
import subprocess

import fire

# 导入自定义模块
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

# 定义一个动作类，用于编写代码
class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    name: str = "SimpleWriteCode"

    # 异步执行动作
    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    # 解析代码
    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text

# 定义一个动作类，用于运行代码
class SimpleRunCode(Action):
    name: str = "SimpleRunCode"

    # 异步执行动作
    async def run(self, code_text: str):
        result = subprocess.run(["python3", "-c", code_text], capture_output=True, text=True)
        code_result = result.stdout
        logger.info(f"{code_result=}")
        return code_result

# 定义一个角色类，用于编写代码
class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([SimpleWriteCode])

    # 执行动作
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # todo will be SimpleWriteCode()

        msg = self.get_memories(k=1)[0]  # find the most recent messages
        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg

# 定义一个角色类，用于运行代码
class RunnableCoder(Role):
    name: str = "Alice"
    profile: str = "RunnableCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([SimpleWriteCode, SimpleRunCode])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)

    # 执行动作
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        # By choosing the Action by order under the hood
        # todo will be first SimpleWriteCode() then SimpleRunCode()
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        result = await todo.run(msg.content)

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg

# 主函数
def main(msg="write a function that calculates the product of a list and run it"):
    # role = SimpleCoder()
    role = RunnableCoder()
    logger.info(msg)
    result = asyncio.run(role.run(msg))
    logger.info(result)

# 程序入口
if __name__ == "__main__":
    fire.Fire(main)

```