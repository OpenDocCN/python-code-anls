# `.\Langchain-Chatchat\server\agent\tools\search_internet.py`

```
# 导入所需的模块
import json
from server.chat.search_engine_chat import search_engine_chat
from configs import VECTOR_SEARCH_TOP_K, MAX_TOKENS
import asyncio
from server.agent import model_container
from pydantic import BaseModel, Field

# 异步函数，用于与搜索引擎进行交互
async def search_engine_iter(query: str):
    # 调用搜索引擎聊天功能，传入参数
    response = await search_engine_chat(query=query,
                                         search_engine_name="bing", # 切换搜索引擎为 Bing
                                         model_name=model_container.MODEL.model_name,
                                         temperature=0.01, # 设置温度为0.01
                                         history=[],
                                         top_k = VECTOR_SEARCH_TOP_K,
                                         max_tokens= MAX_TOKENS,
                                         prompt_name = "default",
                                         stream=False)

    contents = ""

    # 异步迭代器，遍历搜索引擎返回的数据流
    async for data in response.body_iterator: # data 是一个 JSON 字符串
        # 将 JSON 字符串解析为 Python 对象
        data = json.loads(data)
        # 获取回答内容
        contents = data["answer"]
        # 获取相关文档
        docs = data["docs"]

    return contents

# 同步函数，用于调用异步函数并返回结果
def search_internet(query: str):
    return asyncio.run(search_engine_iter(query))

# 定义输入模型，用于接收查询参数
class SearchInternetInput(BaseModel):
    location: str = Field(description="Query for Internet search")

# 主程序入口
if __name__ == "__main__":
    # 调用搜索函数并打印结果
    result = search_internet("今天星期几")
    print("答案:",result)
```