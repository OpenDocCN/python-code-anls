# `MetaGPT\metagpt\tools\web_browser_engine.py`

```py

#!/usr/bin/env python
"""
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

# 导入未来的注解特性
from __future__ import annotations

# 导入模块
import importlib
from typing import Any, Callable, Coroutine, overload

# 从metagpt.config模块中导入CONFIG
from metagpt.config import CONFIG
# 从metagpt.tools模块中导入WebBrowserEngineType
from metagpt.tools import WebBrowserEngineType
# 从metagpt.utils.parse_html模块中导入WebPage
from metagpt.utils.parse_html import WebPage

# 定义WebBrowserEngine类
class WebBrowserEngine:
    # 初始化方法
    def __init__(
        self,
        engine: WebBrowserEngineType | None = None,
        run_func: Callable[..., Coroutine[Any, Any, WebPage | list[WebPage]]] | None = None,
    ):
        # 如果engine为None，则使用CONFIG中的web_browser_engine
        engine = engine or CONFIG.web_browser_engine
        # 如果engine为None，则抛出NotImplementedError
        if engine is None:
            raise NotImplementedError

        # 根据不同的引擎类型，选择对应的模块和运行函数
        if WebBrowserEngineType(engine) is WebBrowserEngineType.PLAYWRIGHT:
            module = "metagpt.tools.web_browser_engine_playwright"
            run_func = importlib.import_module(module).PlaywrightWrapper().run
        elif WebBrowserEngineType(engine) is WebBrowserEngineType.SELENIUM:
            module = "metagpt.tools.web_browser_engine_selenium"
            run_func = importlib.import_module(module).SeleniumWrapper().run
        elif WebBrowserEngineType(engine) is WebBrowserEngineType.CUSTOM:
            run_func = run_func
        else:
            raise NotImplementedError
        # 设置运行函数和引擎类型
        self.run_func = run_func
        self.engine = engine

    # 定义异步运行方法
    @overload
    async def run(self, url: str) -> WebPage:
        ...

    # 定义异步运行方法
    @overload
    async def run(self, url: str, *urls: str) -> list[WebPage]:
        ...

    # 定义异步运行方法
    async def run(self, url: str, *urls: str) -> WebPage | list[WebPage]:
        # 调用运行函数并返回结果
        return await self.run_func(url, *urls)

```