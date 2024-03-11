# `.\Langchain-Chatchat\server\agent\tools\search_knowledgebase_simple.py`

```py
# 从指定路径导入knowledge_base_chat函数
from server.chat.knowledge_base_chat import knowledge_base_chat
# 从configs模块导入VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, MAX_TOKENS
from configs import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, MAX_TOKENS
# 导入json模块
import json
# 导入asyncio模块
import asyncio
# 从server.agent模块导入model_container
from server.agent import model_container

# 异步函数，用于搜索知识库并迭代返回结果
async def search_knowledge_base_iter(database: str, query: str) -> str:
    # 调用knowledge_base_chat函数，传入参数并获取响应
    response = await knowledge_base_chat(query=query,
                                         knowledge_base_name=database,
                                         model_name=model_container.MODEL.model_name,
                                         temperature=0.01,
                                         history=[],
                                         top_k=VECTOR_SEARCH_TOP_K,
                                         max_tokens=MAX_TOKENS,
                                         prompt_name="knowledge_base_chat",
                                         score_threshold=SCORE_THRESHOLD,
                                         stream=False)

    # 初始化contents为空字符串
    contents = ""
    # 异步迭代response的body_iterator
    async for data in response.body_iterator: # 这里的data是一个json字符串
        # 将data解析为json格式
        data = json.loads(data)
        # 获取data中的"answer"字段赋值给contents
        contents = data["answer"]
        # 获取data中的"docs"字段赋值给docs
        docs = data["docs"]
    # 返回contents
    return contents

# 简化搜索知识库的函数，调用search_knowledge_base_iter函数
def search_knowledgebase_simple(query: str):
    return asyncio.run(search_knowledge_base_iter(query))

# 当作为主程序运行时，执行以下代码
if __name__ == "__main__":
    # 调用search_knowledgebase_simple函数，传入参数"大数据男女比例"，并将结果赋值给result
    result = search_knowledgebase_simple("大数据男女比例")
    # 打印结果
    print("答案:",result)
```