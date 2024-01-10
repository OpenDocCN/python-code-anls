# `MetaGPT\tests\metagpt\tools\test_web_browser_engine_playwright.py`

```

"""
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

# 导入所需的模块
import pytest
# 导入自定义的模块
from metagpt.config import CONFIG
from metagpt.tools import web_browser_engine_playwright
from metagpt.utils.parse_html import WebPage

# 异步测试标记
@pytest.mark.asyncio
# 参数化测试
@pytest.mark.parametrize(
    "browser_type, use_proxy, kwagrs, url, urls",
    [
        ("chromium", {"proxy": True}, {}, "https://www.deepwisdom.ai", ("https://www.deepwisdom.ai",)),
        ("firefox", {}, {"ignore_https_errors": True}, "https://www.deepwisdom.ai", ("https://www.deepwisdom.ai",)),
        ("webkit", {}, {"ignore_https_errors": True}, "https://www.deepwisdom.ai", ("https://www.deepwisdom.ai",)),
    ],
    ids=["chromium-normal", "firefox-normal", "webkit-normal"],
)
# 异步测试函数
async def test_scrape_web_page(browser_type, use_proxy, kwagrs, url, urls, proxy, capfd):
    # 保存全局代理配置
    global_proxy = CONFIG.global_proxy
    try:
        # 如果使用代理，则获取代理服务器和代理配置
        if use_proxy:
            server, proxy = await proxy
            CONFIG.global_proxy = proxy
        # 使用 PlaywrightWrapper 类创建浏览器对象
        browser = web_browser_engine_playwright.PlaywrightWrapper(browser_type=browser_type, **kwagrs)
        # 运行浏览器并获取结果
        result = await browser.run(url)
        # 断言结果类型和内容
        assert isinstance(result, WebPage)
        assert "MetaGPT" in result.inner_text

        # 如果有多个 URL，则依次运行浏览器并获取结果
        if urls:
            results = await browser.run(url, *urls)
            # 断言结果类型和内容
            assert isinstance(results, list)
            assert len(results) == len(urls) + 1
            assert all(("MetaGPT" in i.inner_text) for i in results)
        # 如果使用代理，则关闭代理服务器并断言输出中包含代理信息
        if use_proxy:
            server.close()
            assert "Proxy:" in capfd.readouterr().out
    finally:
        # 恢复全局代理配置
        CONFIG.global_proxy = global_proxy

# 程序入口
if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-s"])

```