# `.\Langchain-Chatchat\server\agent\tools_select.py`

```py
# 从指定模块中导入 Tool 类
from langchain.tools import Tool
# 从 server.agent.tools 模块中导入所有内容
from server.agent.tools import *

# 创建一个包含多个 Tool 对象的列表
tools = [
    # 创建一个 Tool 对象，使用 calculate 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=calculate,
        name="calculate",
        description="Useful for when you need to answer questions about simple calculations",
        args_schema=CalculatorInput,
    ),
    # 创建一个 Tool 对象，使用 arxiv 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=arxiv,
        name="arxiv",
        description="A wrapper around Arxiv.org for searching and retrieving scientific articles in various fields.",
        args_schema=ArxivInput,
    ),
    # 创建一个 Tool 对象，使用 weathercheck 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=weathercheck,
        name="weather_check",
        description="",
        args_schema=WeatherInput,
    ),
    # 创建一个 Tool 对象，使用 shell 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=shell,
        name="shell",
        description="Use Shell to execute Linux commands",
        args_schema=ShellInput,
    ),
    # 创建一个 Tool 对象，使用 search_knowledgebase_complex 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=search_knowledgebase_complex,
        name="search_knowledgebase_complex",
        description="Use Use this tool to search local knowledgebase and get information",
        args_schema=KnowledgeSearchInput,
    ),
    # 创建一个 Tool 对象，使用 search_internet 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=search_internet,
        name="search_internet",
        description="Use this tool to use bing search engine to search the internet",
        args_schema=SearchInternetInput,
    ),
    # 创建一个 Tool 对象，使用 wolfram 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=wolfram,
        name="Wolfram",
        description="Useful for when you need to calculate difficult formulas",
        args_schema=WolframInput,
    ),
    # 创建一个 Tool 对象，使用 search_youtube 函数作为功能，指定名称、描述、参数模式
    Tool.from_function(
        func=search_youtube,
        name="search_youtube",
        description="use this tools to search youtube videos",
        args_schema=YoutubeInput,
    ),
]

# 从 tools 列表中提取每个 Tool 对象的名称，组成一个新的列表
tool_names = [tool.name for tool in tools]
```