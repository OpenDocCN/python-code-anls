# `.\Langchain-Chatchat\server\agent\tools\search_youtube.py`

```
# 导入 Langchain 自带的 YouTube 搜索工具
from langchain.tools import YouTubeSearchTool
# 导入 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 定义一个函数用于搜索 YouTube 视频，接受一个查询字符串作为参数
def search_youtube(query: str):
    # 创建一个 YouTubeSearchTool 实例
    tool = YouTubeSearchTool()
    # 调用 YouTubeSearchTool 实例的 run 方法，传入查询字符串作为参数，并返回结果
    return tool.run(tool_input=query)

# 定义一个名为 YoutubeInput 的数据模型，继承自 BaseModel
class YoutubeInput(BaseModel):
    # 定义一个名为 location 的字段，类型为 str，描述为"Query for Videos search"
    location: str = Field(description="Query for Videos search")
```