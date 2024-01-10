# `MetaGPT\examples\agent_creator.py`

```

"""
Filename: MetaGPT/examples/agent_creator.py
Created Date: Tuesday, September 12th 2023, 3:28:37 pm
Author: garylin2099
"""
# 导入 re 模块
import re
# 导入 metagpt 中的相关模块和类
from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.const import METAGPT_ROOT
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

# 读取示例代码文件
EXAMPLE_CODE_FILE = METAGPT_ROOT / "examples/build_customized_agent.py"
MULTI_ACTION_AGENT_CODE_EXAMPLE = EXAMPLE_CODE_FILE.read_text()

# 创建一个名为 CreateAgent 的类，继承自 Action 类
class CreateAgent(Action):
    # 定义一个字符串模板
    PROMPT_TEMPLATE: str = """
    ### BACKGROUND
    You are using an agent framework called metagpt to write agents capable of different actions,
    the usage of metagpt can be illustrated by the following example:
    ### EXAMPLE STARTS AT THIS LINE
    {example}
    ### EXAMPLE ENDS AT THIS LINE
    ### TASK
    Now you should create an agent with appropriate actions based on the instruction, consider carefully about
    the PROMPT_TEMPLATE of all actions and when to call self._aask()
    ### INSTRUCTION
    {instruction}
    ### YOUR CODE
    Return ```python your_code_here ``` with NO other texts, your code:
    """
    
    # 异步方法，运行创建代理的操作
    async def run(self, example: str, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(example=example, instruction=instruction)
        # 调用 _aask 方法，等待用户输入
        rsp = await self._aask(prompt)
        # 解析用户输入的代码
        code_text = CreateAgent.parse_code(rsp)
        return code_text

    # 静态方法，解析用户输入的代码
    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else ""
        CONFIG.workspace_path.mkdir(parents=True, exist_ok=True)
        new_file = CONFIG.workspace_path / "agent_created_agent.py"
        new_file.write_text(code_text)
        return code_text

# 创建一个名为 AgentCreator 的类，继承自 Role 类
class AgentCreator(Role):
    name: str = "Matrix"
    profile: str = "AgentCreator"
    agent_template: str = MULTI_ACTION_AGENT_CODE_EXAMPLE

    # 初始化方法
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([CreateAgent])

    # 异步方法，执行操作
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        msg = self.rc.memory.get()[-1]
        instruction = msg.content
        code_text = await CreateAgent().run(example=self.agent_template, instruction=instruction)
        msg = Message(content=code_text, role=self.profile, cause_by=todo)
        return msg

# 主程序入口
if __name__ == "__main__":
    import asyncio

    # 异步方法，执行主程序
    async def main():
        agent_template = MULTI_ACTION_AGENT_CODE_EXAMPLE
        creator = AgentCreator(agent_template=agent_template)
        msg = """
        Write an agent called SimpleTester that will take any code snippet (str) and do the following:
        1. write a testing code (str) for testing the given code snippet, save the testing code as a .py file in the current working directory;
        2. run the testing code.
        You can use pytest as the testing framework.
        """
        await creator.run(msg)

    asyncio.run(main())

```