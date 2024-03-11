# `.\Langchain-Chatchat\server\agent\tools\shell.py`

```
# 导入必要的模块和类
from pydantic import BaseModel, Field
from langchain.tools import ShellTool

# 定义一个函数，用于在 LangChain 中执行 Shell 命令
def shell(query: str):
    # 创建一个 ShellTool 实例
    tool = ShellTool()
    # 调用 ShellTool 实例的 run 方法，传入要执行的 Shell 命令，并返回执行结果
    return tool.run(tool_input=query)

# 定义一个数据模型，用于验证输入的 Shell 命令是否符合要求
class ShellInput(BaseModel):
    # 定义一个字段 query，类型为 str，描述为“一个能在Linux命令行运行的Shell命令”
    query: str = Field(description="一个能在Linux命令行运行的Shell命令")
```