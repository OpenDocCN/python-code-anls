# `.\AutoGPT\autogpts\forge\forge\actions\web\web_search.py`

```py
from __future__ import annotations


# 导入未来版本的特性 annotations

import json
import time
from itertools import islice

from duckduckgo_search import DDGS

from ..registry import action

DUCKDUCKGO_MAX_ATTEMPTS = 3

@action(
    name="web_search",
    description="Searches the web",
    parameters=[
        {
            "name": "query",
            "description": "The search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)


# 定义一个名为 web_search 的动作，描述为 "Searches the web"，参数为一个包含查询信息的字典，输出类型为字符串列表

async def web_search(agent, task_id: str, query: str) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """


# 异步函数，返回 Google 搜索结果
# 参数 query 为搜索查询字符串
# 返回搜索结果的字符串

    search_results = []
    attempts = 0
    num_results = 8

    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)

        results = DDGS().text(query)
        search_results = list(islice(results, num_results))

        if search_results:
            break

        time.sleep(1)
        attempts += 1

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)

def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """


# 定义一个函数 safe_google_results，将 Google 搜索结果以安全格式返回
# 参数 results 可以是字符串或列表，返回结果为字符串

    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message
```