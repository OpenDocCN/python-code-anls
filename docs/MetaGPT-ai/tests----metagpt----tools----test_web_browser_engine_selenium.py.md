# `MetaGPT\tests\metagpt\tools\test_web_browser_engine_selenium.py`

```
"""
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

# 导入所需的模块
import pytest
# 从metagpt.config模块中导入CONFIG
from metagpt.config import CONFIG
# 从metagpt.tools中导入web_browser_engine_selenium
from metagpt.tools import web_browser_engine_selenium
# 从metagpt.utils.parse_html中导入WebPage
from metagpt.utils.parse_html import WebPage

# 标记为异步测试
@pytest.mark.asyncio
# 参数化测试
@pytest.mark.parametrize(
    "browser_type, use_proxy, url, urls",
    [
        ("chrome", True, "https://deepwisdom.ai", ("https://deepwisdom.ai",)),
        ("firefox", False, "https://deepwisdom.ai", ("https://deepwisdom.ai",)),
        ("edge", False, "https://deepwisdom.ai", ("https://deepwisdom.ai",)),
    ],
    ids=["chrome-normal", "firefox-normal", "edge-normal"],
)
# 异步测试函数
async def test_scrape_web_page(browser_type, use_proxy, url, urls, proxy, capfd):
    # Prerequisites
    # firefox, chrome, Microsoft Edge

    # 保存全局代理配置
    global_proxy = CONFIG.global_proxy
    try:
        # 如果使用代理，则设置代理
        if use_proxy:
            server, proxy = await proxy
            CONFIG.global_proxy = proxy
        # 使用SeleniumWrapper创建浏览器对象
        browser = web_browser_engine_selenium.SeleniumWrapper(browser_type=browser_type)
        # 运行浏览器访问url
        result = await browser.run(url)
        # 断言结果为WebPage类型，并且包含"MetaGPT"
        assert isinstance(result, WebPage)
        assert "MetaGPT" in result.inner_text

        # 如果有urls
        if urls:
            # 分别运行浏览器访问urls
            results = await browser.run(url, *urls)
            # 断言结果为列表类型，并且长度符合预期
            assert isinstance(results, list)
            assert len(results) == len(urls) + 1
            # 断言所有结果都包含"MetaGPT"
            assert all(("MetaGPT" in i.inner_text) for i in results)
        # 如果使用代理，则关闭代理
        if use_proxy:
            server.close()
            # 断言输出中包含"Proxy:"
            assert "Proxy:" in capfd.readouterr().out
    finally:
        # 恢复全局代理配置
        CONFIG.global_proxy = global_proxy

# 当作为主程序运行时执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])
"""
```