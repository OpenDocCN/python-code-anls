# `.\AutoGPT\autogpts\autogpt\autogpt\commands\web_search.py`

```py
# 导入所需的模块和库
from __future__ import annotations
import json
import time
from itertools import islice
from duckduckgo_search import DDGS
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import ConfigurationError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

# 定义命令的类别和标题
COMMAND_CATEGORY = "web_search"
COMMAND_CATEGORY_TITLE = "Web Search"

# 设置 DuckDuckGo 最大尝试次数
DUCKDUCKGO_MAX_ATTEMPTS = 3

# 定义一个命令函数，用于进行网页搜索
@command(
    "web_search",
    "Searches the web",
    {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The search query",
            required=True,
        )
    },
    aliases=["search"],
)
def web_search(query: str, agent: Agent, num_results: int = 8) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    # 初始化搜索结果列表和尝试次数
    search_results = []
    attempts = 0

    # 在尝试次数小于最大尝试次数时循环
    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        # 如果查询为空，则返回空的搜索结果
        if not query:
            return json.dumps(search_results)

        # 使用 DuckDuckGo 搜索引擎进行搜索
        results = DDGS().text(query)
        # 从搜索结果中获取指定数量的结果
        search_results = list(islice(results, num_results))

        # 如果搜索结果不为空，则跳出循环
        if search_results:
            break

        # 等待一秒钟后继续尝试
        time.sleep(1)
        attempts += 1

    # 对搜索结果进行处理，提取标题、URL和摘要信息
    search_results = [
        {
            "title": r["title"],
            "url": r["href"],
            **({"exerpt": r["body"]} if r.get("body") else {}),
        }
        for r in search_results
    ]

    # 返回搜索结果的标题
    results = (
        "## Search results\n"
        # "Read these results carefully."
        # " Extract the information you need for your task from the list of results"
        # " if possible. Otherwise, choose a webpage from the list to read entirely."
        # "\n\n"
    # 将搜索结果中每个结果的标题、URL和摘录格式化为字符串，并用换行符连接起来
    ) + "\n\n".join(
        # 格式化每个搜索结果的标题和URL
        f"### \"{r['title']}\"\n"
        f"**URL:** {r['url']}  \n"
        # 检查是否存在摘录，如果存在则格式化为字符串，否则显示"N/A"
        "**Excerpt:** " + (f'"{exerpt}"' if (exerpt := r.get("exerpt")) else "N/A")
        # 遍历搜索结果列表中的每个结果
        for r in search_results
    )
    # 返回格式化后的安全搜索结果
    return safe_google_results(results)
# 定义一个名为 google 的命令，用于进行 Google 搜索
@command(
    "google",  # 命令名称为 "google"
    "Google Search",  # 命令描述为 "Google Search"
    {
        "query": JSONSchema(  # 参数 "query" 的 JSONSchema 描述
            type=JSONSchema.Type.STRING,  # 参数类型为字符串
            description="The search query",  # 参数描述为 "The search query"
            required=True,  # 参数为必填项
        )
    },
    lambda config: bool(config.google_api_key)  # 配置中存在 google_api_key
    and bool(config.google_custom_search_engine_id),  # 配置中存在 google_custom_search_engine_id
    "Configure google_api_key and custom_search_engine_id.",  # 配置缺少 google_api_key 和 custom_search_engine_id 时的提示信息
    aliases=["search"],  # 命令的别名为 "search"
)
def google(query: str, agent: Agent, num_results: int = 8) -> str | list[str]:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.  # 参数 query 为搜索关键词
        num_results (int): The number of results to return.  # 参数 num_results 为返回结果的数量

    Returns:
        str: The results of the search.  # 返回搜索结果的字符串列表
    """

    from googleapiclient.discovery import build  # 导入 build 函数用于构建 Google API 服务
    from googleapiclient.errors import HttpError  # 导入 HttpError 用于处理 Google API 错误

    try:
        # 从配置文件中获取 Google API key 和 Custom Search Engine ID
        api_key = agent.legacy_config.google_api_key
        custom_search_engine_id = agent.legacy_config.google_custom_search_engine_id

        # 初始化 Custom Search API 服务
        service = build("customsearch", "v1", developerKey=api_key)

        # 发送搜索查询并获取结果
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=num_results)
            .execute()
        )

        # 从响应中提取搜索结果项
        search_results = result.get("items", [])

        # 创建仅包含搜索结果链接的列表
        search_results_links = [item["link"] for item in search_results]
    # 捕获 HttpError 异常
    except HttpError as e:
        # 将异常内容解码为 JSON 格式
        error_details = json.loads(e.content.decode())

        # 检查错误是否与无效或缺失的 API 密钥有关
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            # 抛出 ConfigurationError 异常，提示 API 密钥无效或缺失
            raise ConfigurationError(
                "The provided Google API key is invalid or missing."
            )
        # 重新抛出异常
        raise
    # google_result 可能是一个列表或一个字符串，取决于搜索结果

    # 返回搜索结果 URL 列表
    return safe_google_results(search_results_links)
# 将谷歌搜索结果以安全格式返回
def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    # 如果结果是列表类型
    if isinstance(results, list):
        # 将结果列表中的每个结果编码为 UTF-8 格式，忽略无法编码的字符，然后解码为 UTF-8 格式，最后转换为 JSON 字符串
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        # 将结果编码为 UTF-8 格式，忽略无法编码的字符，然后解码为 UTF-8 格式
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    # 返回安全格式的结果
    return safe_message
```