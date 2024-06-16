# `.\agent\GenAINewsAgent\server\agent.py`

```
# 从 llms.groq 模块中导入 GroqLLMStream 类
from llms.groq import GroqLLMStream
# 从 configs 模块中导入 GROQ_API_KEY 和 GROQ_MODEL_NAME 变量
from configs import GROQ_API_KEY, GROQ_MODEL_NAME
# 从 news 模块中导入 getNews 函数
from news import getNews
# 从 prompts 模块中导入 SYSTEM_PROMPT 变量
from prompts import SYSTEM_PROMPT
# 从 brave_search 模块中导入 BraveSearch 类
from brave_search import BraveSearch
# 从 time 模块中导入 time 函数
from time import time

# 创建 GroqLLMStream 类的实例，传入 GROQ_API_KEY 作为参数
llm = GroqLLMStream(GROQ_API_KEY)

# 创建 BraveSearch 类的实例
bs = BraveSearch()

# 定义异步函数 newsAgent，接受一个字符串参数 query
async def newsAgent(query: str):
    # 记录函数开始时间
    st_time = time()
    # 调用 BraveSearch 类的方法，传入查询参数 query，异步获取新闻信息
    retrieved_news_items = await bs(query)
    # 记录函数结束时间
    en_time = time()
    # 打印搜索所用时间
    print(f'Search Time: {en_time - st_time}s')
    # 如果未检索到任何新闻信息
    if not retrieved_news_items:
        # 返回一条无法获取相关新闻的提示信息
        yield "\n_Cannot fetch any relevant news related to the search query._"
        return
    # 创建包含查询和检索到的新闻信息的消息列表
    messages = [{
        "role":
        "user",
        "content":
        f"Query: {query}\n\nNews Items: {retrieved_news_items}"
    }]
    # 通过 GroqLLMStream 实例 llm 进行模型推理
    # 传入模型名称、消息列表、系统提示、最大生成令牌数、温度参数
    async for chunk in llm(GROQ_MODEL_NAME,
                           messages,
                           system=SYSTEM_PROMPT,
                           max_tokens=1024,
                           temperature=0.2):
        # 返回推理结果的每个 chunk
        yield chunk
```