# `MetaGPT\metagpt\tools\search_engine.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 20:15
@Author  : alexanderwu
@File    : search_engine.py
"""
# 导入模块
import importlib
from typing import Callable, Coroutine, Literal, Optional, Union, overload

# 导入自定义模块
from semantic_kernel.skill_definition import sk_function
# 导入配置
from metagpt.config import CONFIG
# 导入工具类
from metagpt.tools import SearchEngineType

# 定义搜索引擎类
class SkSearchEngine:
    def __init__(self):
        self.search_engine = SearchEngine()

    # 异步搜索方法
    @sk_function(
        description="searches results from Google. Useful when you need to find short "
        "and succinct answers about a specific topic. Input should be a search query.",
        name="searchAsync",
        input_description="search",
    )
    async def run(self, query: str) -> str:
        result = await self.search_engine.run(query)
        return result

# 定义搜索引擎类
class SearchEngine:
    """Class representing a search engine.

    Args:
        engine: The search engine type. Defaults to the search engine specified in the config.
        run_func: The function to run the search. Defaults to None.

    Attributes:
        run_func: The function to run the search.
        engine: The search engine type.
    """

    def __init__(
        self,
        engine: Optional[SearchEngineType] = None,
        run_func: Callable[[str, int, bool], Coroutine[None, None, Union[str, list[str]]]] = None,
    ):
        engine = engine or CONFIG.search_engine
        # 根据不同的搜索引擎类型选择对应的模块和运行函数
        if engine == SearchEngineType.SERPAPI_GOOGLE:
            module = "metagpt.tools.search_engine_serpapi"
            run_func = importlib.import_module(module).SerpAPIWrapper().run
        elif engine == SearchEngineType.SERPER_GOOGLE:
            module = "metagpt.tools.search_engine_serper"
            run_func = importlib.import_module(module).SerperWrapper().run
        elif engine == SearchEngineType.DIRECT_GOOGLE:
            module = "metagpt.tools.search_engine_googleapi"
            run_func = importlib.import_module(module).GoogleAPIWrapper().run
        elif engine == SearchEngineType.DUCK_DUCK_GO:
            module = "metagpt.tools.search_engine_ddg"
            run_func = importlib.import_module(module).DDGAPIWrapper().run
        elif engine == SearchEngineType.CUSTOM_ENGINE:
            pass  # run_func = run_func
        else:
            raise NotImplementedError
        self.engine = engine
        self.run_func = run_func

    # 重载运行方法，根据参数不同返回不同类型的结果
    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
    ) -> list[dict[str, str]]:
        ...

    # 异步运行搜索方法
    async def run(self, query: str, max_results: int = 8, as_string: bool = True) -> Union[str, list[dict[str, str]]]:
        """Run a search query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return. Defaults to 8.
            as_string: Whether to return the results as a string or a list of dictionaries. Defaults to True.

        Returns:
            The search results as a string or a list of dictionaries.
        """
        return await self.run_func(query, max_results=max_results, as_string=as_string)

```