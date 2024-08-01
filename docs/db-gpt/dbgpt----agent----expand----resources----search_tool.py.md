# `.\DB-GPT-src\dbgpt\agent\expand\resources\search_tool.py`

```py
# 导入正则表达式模块
import re

# 导入 Annotated 和 Doc 类型扩展
from typing_extensions import Annotated, Doc

# 从资源工具基础模块导入工具函数装饰器
from ...resource.tool.base import tool

# 使用工具函数装饰器，描述百度搜索并返回结果的 Markdown 字符串
@tool(
    description="Baidu search and return the results as a markdown string. Please set "
    "number of results not less than 8 for rich search results.",
)
# 定义百度搜索函数，接受查询字符串和返回结果数，默认返回 8 个结果
def baidu_search(
    query: Annotated[str, Doc("The search query.")],  # 查询字符串参数
    num_results: Annotated[int, Doc("The number of search results to return.")] = 8,  # 返回结果数参数，默认为 8
) -> str:  # 返回字符串类型
    """Baidu search and return the results as a markdown string.

    Please set number of results not less than 8 for rich search results.
    """
    try:
        import requests  # 尝试导入 requests 库
    except ImportError:
        raise ImportError(
            "`requests` is required for baidu_search tool, please run "
            "`pip install requests` to install it."
        )
    try:
        from bs4 import BeautifulSoup  # 尝试导入 BeautifulSoup 库
    except ImportError:
        raise ImportError(
            "`beautifulsoup4` is required for baidu_search tool, please run "
            "`pip install beautifulsoup4` to install it."
        )

    # 设置请求头部信息，模拟 Firefox 浏览器的请求
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:112.0) "
        "Gecko/20100101 Firefox/112.0"
    }

    # 确保 num_results 至少为 8
    num_results = int(num_results)
    if num_results < 8:
        num_results = 8

    # 构建百度搜索的 URL
    url = f"https://www.baidu.com/s?wd={query}&rn={num_results}"
    # 发送 GET 请求到百度搜索页面
    response = requests.get(url, headers=headers)
    # 设置响应内容编码为 UTF-8
    response.encoding = "utf-8"
    # 使用 BeautifulSoup 解析响应文本
    soup = BeautifulSoup(response.text, "html.parser")

    # 存储搜索结果的列表
    search_results = []
    # 遍历所有搜索结果的 div 元素
    for result in soup.find_all("div", class_=re.compile("^result c-container ")):
        # 获取搜索结果标题
        title = result.find("h3", class_="t").get_text()
        # 获取搜索结果链接
        link = result.find("a", href=True)["href"]
        # 获取搜索结果摘要信息
        snippet = result.find("span", class_=re.compile("^content-right_"))
        if snippet:
            snippet = snippet.get_text()
        else:
            snippet = ""
        # 将结果以字典形式存储到 search_results 中
        search_results.append({"title": title, "href": link, "snippet": snippet})

    # 转换搜索结果列表为 Markdown 格式的视图字符串并返回
    return _search_to_view(search_results)


# 定义内部函数 _search_to_view，将搜索结果列表转换为 Markdown 视图字符串
def _search_to_view(results) -> str:
    # 存储视图结果的列表
    view_results = []
    # 遍历结果列表中的每一项
    for item in results:
        # 将每一项格式化为 Markdown 格式的字符串并添加到视图结果列表中
        view_results.append(
            f"### [{item['title']}]({item['href']})\n{item['snippet']}\n"
        )
    # 将所有视图结果连接为一个字符串并返回
    return "\n".join(view_results)
```