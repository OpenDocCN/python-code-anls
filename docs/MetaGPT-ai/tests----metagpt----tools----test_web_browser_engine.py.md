# `MetaGPT\tests\metagpt\tools\test_web_browser_engine.py`

```
"""
# 添加修改信息，标明修改人和日期，移除全局配置`CONFIG`，启用业务隔离的配置支持

# 导入 pytest 模块
import pytest

# 从 metagpt.tools 模块中导入 WebBrowserEngineType 和 web_browser_engine
from metagpt.tools import WebBrowserEngineType, web_browser_engine
# 从 metagpt.utils.parse_html 模块中导入 WebPage
from metagpt.utils.parse_html import WebPage

# 标记异步测试
@pytest.mark.asyncio
# 参数化测试，传入参数 browser_type, url, urls
@pytest.mark.parametrize(
    "browser_type, url, urls",
    [
        (WebBrowserEngineType.PLAYWRIGHT, "https://deepwisdom.ai", ("https://deepwisdom.ai",)),
        (WebBrowserEngineType.SELENIUM, "https://deepwisdom.ai", ("https://deepwisdom.ai",)),
    ],
    # 为参数化测试指定 ids
    ids=["playwright", "selenium"],
)
# 异步测试函数
async def test_scrape_web_page(browser_type, url, urls):
    # 创建 WebBrowserEngine 对象
    browser = web_browser_engine.WebBrowserEngine(engine=browser_type)
    # 运行浏览器，获取结果
    result = await browser.run(url)
    # 断言结果类型为 WebPage
    assert isinstance(result, WebPage)
    # 断言结果内包含"MetaGPT"
    assert "MetaGPT" in result.inner_text

    # 如果有 urls
    if urls:
        # 运行浏览器，获取多个结果
        results = await browser.run(url, *urls)
        # 断言结果类型为列表
        assert isinstance(results, list)
        # 断言结果长度符合预期
        assert len(results) == len(urls) + 1
        # 断言所有结果内包含"MetaGPT"
        assert all(("MetaGPT" in i.inner_text) for i in results)

# 如果当前文件被直接执行
if __name__ == "__main__":
    # 运行 pytest
    pytest.main([__file__, "-s"])
"""
```