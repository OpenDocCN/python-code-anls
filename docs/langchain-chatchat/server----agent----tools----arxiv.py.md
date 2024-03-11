# `.\Langchain-Chatchat\server\agent\tools\arxiv.py`

```
# 导入 LangChain 的 ArxivQueryRun 工具
from pydantic import BaseModel, Field
from langchain.tools.arxiv.tool import ArxivQueryRun

# 定义一个函数 arxiv，接受一个查询字符串作为参数
def arxiv(query: str):
    # 创建一个 ArxivQueryRun 实例
    tool = ArxivQueryRun()
    # 调用 ArxivQueryRun 实例的 run 方法，传入查询字符串作为参数，并返回结果
    return tool.run(tool_input=query)

# 定义一个 Pydantic BaseModel 类 ArxivInput，包含一个名为 query 的字段，描述为"The search query title"
class ArxivInput(BaseModel):
    query: str = Field(description="The search query title")
```