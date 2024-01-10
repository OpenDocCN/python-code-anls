# `MetaGPT\metagpt\utils\parse_html.py`

```

#!/usr/bin/env python
# 指定脚本的解释器为 Python

from __future__ import annotations
# 导入未来版本的注解特性

from typing import Generator, Optional
# 导入类型提示相关的模块

from urllib.parse import urljoin, urlparse
# 导入处理 URL 相关的模块

from bs4 import BeautifulSoup
# 导入用于解析 HTML 的模块

from pydantic import BaseModel, PrivateAttr
# 导入用于数据验证和设置私有属性的模块

class WebPage(BaseModel):
    # 定义 WebPage 类，继承自 BaseModel
    inner_text: str
    # 定义 inner_text 属性，表示页面的文本内容
    html: str
    # 定义 html 属性，表示页面的 HTML 内容
    url: str
    # 定义 url 属性，表示页面的 URL

    _soup: Optional[BeautifulSoup] = PrivateAttr(default=None)
    # 定义私有属性 _soup，用于存储页面的 BeautifulSoup 对象，默认值为 None
    _title: Optional[str] = PrivateAttr(default=None)
    # 定义私有属性 _title，用于存储页面的标题，默认值为 None

    @property
    def soup(self) -> BeautifulSoup:
        # 定义 soup 属性，用于获取页面的 BeautifulSoup 对象
        if self._soup is None:
            self._soup = BeautifulSoup(self.html, "html.parser")
        return self._soup
        # 如果 _soup 为空，则使用 BeautifulSoup 解析 html，并返回 _soup

    @property
    def title(self):
        # 定义 title 属性，用于获取页面的标题
        if self._title is None:
            title_tag = self.soup.find("title")
            self._title = title_tag.text.strip() if title_tag is not None else ""
        return self._title
        # 如果 _title 为空，则从 _soup 中查找 title 标签，并返回标题文本

    def get_links(self) -> Generator[str, None, None]:
        # 定义 get_links 方法，用于获取页面中的链接
        for i in self.soup.find_all("a", href=True):
            url = i["href"]
            result = urlparse(url)
            if not result.scheme and result.path:
                yield urljoin(self.url, url)
            elif url.startswith(("http://", "https://")):
                yield urljoin(self.url, url)
        # 遍历页面中的所有 a 标签，获取链接并返回

def get_html_content(page: str, base: str):
    # 定义 get_html_content 函数，用于获取页面的文本内容
    soup = _get_soup(page)
    return soup.get_text(strip=True)
    # 使用 _get_soup 函数解析页面，然后获取页面的文本内容并返回

def _get_soup(page: str):
    # 定义私有函数 _get_soup，用于解析页面并去除不需要的内容
    soup = BeautifulSoup(page, "html.parser")
    for s in soup(["style", "script", "[document]", "head", "title"]):
        s.extract()
    return soup
    # 使用 BeautifulSoup 解析页面，并去除不需要的内容后返回

```