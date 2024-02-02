# `MetaGPT\metagpt\learn\google_search.py`

```py

# 从 metagpt.tools.search_engine 模块中导入 SearchEngine 类
from metagpt.tools.search_engine import SearchEngine

# 定义一个异步函数，用于执行谷歌搜索并获取搜索结果
async def google_search(query: str, max_results: int = 6, **kwargs):
    """Perform a web search and retrieve search results.

    :param query: The search query. # 搜索查询字符串
    :param max_results: The number of search results to retrieve # 要获取的搜索结果数量
    :return: The web search results in markdown format. # 以 markdown 格式返回网页搜索结果
    """
    # 调用 SearchEngine 类的 run 方法执行搜索，并获取结果
    results = await SearchEngine().run(query, max_results=max_results, as_string=False)
    # 将搜索结果格式化为 markdown 格式的字符串
    return "\n".join(f"{i}. [{j['title']}]({j['link']}): {j['snippet']}" for i, j in enumerate(results, 1))

```