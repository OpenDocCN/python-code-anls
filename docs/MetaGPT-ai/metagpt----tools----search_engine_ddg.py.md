# `MetaGPT\metagpt\tools\search_engine_ddg.py`

```

#!/usr/bin/env python
# 指定使用的 Python 解释器

from __future__ import annotations
# 导入未来版本的注解特性

import asyncio
# 异步 I/O 库
import json
# JSON 编解码器
from concurrent import futures
# 并发执行库
from typing import Literal, overload
# 类型提示

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError(
        "To use this module, you should have the `duckduckgo_search` Python package installed. "
        "You can install it by running the command: `pip install -e.[search-ddg]`"
    )
# 尝试导入 duckduckgo_search 模块，如果导入失败则抛出 ImportError 异常

from metagpt.config import CONFIG
# 从 metagpt.config 模块导入 CONFIG 配置

class DDGAPIWrapper:
    """Wrapper around duckduckgo_search API.

    To use this module, you should have the `duckduckgo_search` Python package installed.
    """
    # DDGAPIWrapper 类，封装了 duckduckgo_search API

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: futures.Executor | None = None,
    ):
        # 初始化方法，接受 asyncio 事件循环和执行器作为参数
        kwargs = {}
        if CONFIG.global_proxy:
            kwargs["proxies"] = CONFIG.global_proxy
        self.loop = loop
        self.executor = executor
        self.ddgs = DDGS(**kwargs)
        # 根据配置创建 DDGS 对象

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
        focus: list[str] | None = None,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
        focus: list[str] | None = None,
    ) -> list[dict[str, str]]:
        ...

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
    ) -> str | list[dict]:
        """Return the results of a Google search using the official Google API

        Args:
            query: The search query.
            max_results: The number of results to return.
            as_string: A boolean flag to determine the return type of the results. If True, the function will
                return a formatted string with the search results. If False, it will return a list of dictionaries
                containing detailed information about each search result.

        Returns:
            The results of the search.
        """
        loop = self.loop or asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._search_from_ddgs,
            query,
            max_results,
        )
        search_results = await future

        # Return the list of search result URLs
        if as_string:
            return json.dumps(search_results, ensure_ascii=False)
        return search_results
        # 返回搜索结果

    def _search_from_ddgs(self, query: str, max_results: int):
        return [
            {"link": i["href"], "snippet": i["body"], "title": i["title"]}
            for (_, i) in zip(range(max_results), self.ddgs.text(query))
        ]
        # 从 duckduckgo_search 中获取搜索结果

if __name__ == "__main__":
    import fire

    fire.Fire(DDGAPIWrapper().run)
    # 使用 fire 库将 DDGAPIWrapper 类的 run 方法暴露为命令行接口

```