# `.\AutoGPT\autogpts\autogpt\autogpt\processing\html.py`

```py
"""HTML processing functions"""
# 导入必要的模块和类
from __future__ import annotations

from bs4 import BeautifulSoup
from requests.compat import urljoin

# 从 BeautifulSoup 对象中提取超链接
def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    # 遍历 BeautifulSoup 对象中的所有 <a> 标签，提取文本和链接，组成元组列表
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]

# 格式化超链接以便显示给用户
def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    # 格式化超链接文本和链接，组成字符串列表
    return [f"{link_text.strip()} ({link_url})" for link_text, link_url in hyperlinks]
```