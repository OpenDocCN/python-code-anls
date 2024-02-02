# `MetaGPT\tests\metagpt\tools\test_search_engine.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 17:46
@Author  : alexanderwu
@File    : test_search_engine.py
"""
# 导入必要的模块
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
import pytest
import tests.data.search
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.tools import SearchEngineType
from metagpt.tools.search_engine import SearchEngine

# 设置搜索缓存路径
search_cache_path = Path(tests.data.search.__path__[0])

# 创建模拟的搜索引擎类
class MockSearchEnine:
    async def run(self, query: str, max_results: int = 8, as_string: bool = True) -> str | list[dict[str, str]]:
        rets = [
            {"url": "https://metagpt.com/mock/{i}", "title": query, "snippet": query * i} for i in range(max_results)
        ]
        return "\n".join(rets) if as_string else rets

# 使用pytest标记异步测试
@pytest.mark.asyncio
# 参数化测试用例
@pytest.mark.parametrize(
    ("search_engine_type", "run_func", "max_results", "as_string"),
    [
        (SearchEngineType.SERPAPI_GOOGLE, None, 8, True),
        (SearchEngineType.SERPAPI_GOOGLE, None, 4, False),
        (SearchEngineType.DIRECT_GOOGLE, None, 8, True),
        (SearchEngineType.DIRECT_GOOGLE, None, 6, False),
        (SearchEngineType.SERPER_GOOGLE, None, 8, True),
        (SearchEngineType.SERPER_GOOGLE, None, 6, False),
        (SearchEngineType.DUCK_DUCK_GO, None, 8, True),
        (SearchEngineType.DUCK_DUCK_GO, None, 6, False),
        (SearchEngineType.CUSTOM_ENGINE, MockSearchEnine().run, 8, False),
        (SearchEngineType.CUSTOM_ENGINE, MockSearchEnine().run, 6, False),
    ],
)
# 定义测试函数
async def test_search_engine(search_engine_type, run_func: Callable, max_results: int, as_string: bool, aiohttp_mocker):
    # 设置缓存JSON路径
    cache_json_path = None
    # 根据搜索引擎类型进行不同的断言和设置缓存路径
    if search_engine_type is SearchEngineType.SERPAPI_GOOGLE:
        assert CONFIG.SERPAPI_API_KEY and CONFIG.SERPAPI_API_KEY != "YOUR_API_KEY"
        cache_json_path = search_cache_path / f"serpapi-metagpt-{max_results}.json"
    elif search_engine_type is SearchEngineType.DIRECT_GOOGLE:
        assert CONFIG.GOOGLE_API_KEY and CONFIG.GOOGLE_API_KEY != "YOUR_API_KEY"
        assert CONFIG.GOOGLE_CSE_ID and CONFIG.GOOGLE_CSE_ID != "YOUR_CSE_ID"
    elif search_engine_type is SearchEngineType.SERPER_GOOGLE:
        assert CONFIG.SERPER_API_KEY and CONFIG.SERPER_API_KEY != "YOUR_API_KEY"
        cache_json_path = search_cache_path / f"serper-metagpt-{max_results}.json"

    # 如果存在缓存路径，则读取缓存数据并设置为模拟的HTTP响应
    if cache_json_path:
        with open(cache_json_path) as f:
            data = json.load(f)
            aiohttp_mocker.set_json(data)
    # 创建搜索引擎对象并执行搜索
    search_engine = SearchEngine(search_engine_type, run_func)
    rsp = await search_engine.run("metagpt", max_results, as_string)
    logger.info(rsp)
    # 根据返回类型进行断言
    if as_string:
        assert isinstance(rsp, str)
    else:
        assert isinstance(rsp, list)
        assert len(rsp) <= max_results

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```