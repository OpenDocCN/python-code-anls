# `MetaGPT\metagpt\tools\web_browser_engine_selenium.py`

```py

#!/usr/bin/env python
"""
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

from __future__ import annotations

import asyncio  # 引入异步编程库
import importlib  # 动态导入模块
from concurrent import futures  # 并发编程库
from copy import deepcopy  # 复制对象
from typing import Literal  # 类型提示

from selenium.webdriver.common.by import By  # 导入 Selenium 的定位方式
from selenium.webdriver.support import expected_conditions as EC  # 导入 Selenium 的预期条件
from selenium.webdriver.support.wait import WebDriverWait  # 导入 Selenium 的等待
from webdriver_manager.core.download_manager import WDMDownloadManager  # 导入 WebDriver 管理器
from webdriver_manager.core.http import WDMHttpClient  # 导入 WebDriver 管理器的 HTTP 客户端

from metagpt.config import CONFIG  # 导入配置
from metagpt.utils.parse_html import WebPage  # 导入 WebPage 类


class SeleniumWrapper:
    """Wrapper around Selenium.

    To use this module, you should check the following:

    1. Run the following command: pip install metagpt[selenium].
    2. Make sure you have a compatible web browser installed and the appropriate WebDriver set up
       for that browser before running. For example, if you have Mozilla Firefox installed on your
       computer, you can set the configuration SELENIUM_BROWSER_TYPE to firefox. After that, you
       can scrape web pages using the Selenium WebBrowserEngine.
    """

    def __init__(
        self,
        browser_type: Literal["chrome", "firefox", "edge", "ie"] | None = None,  # 初始化函数，指定浏览器类型
        launch_kwargs: dict | None = None,  # 初始化函数，指定启动参数
        *,
        loop: asyncio.AbstractEventLoop | None = None,  # 初始化函数，指定事件循环
        executor: futures.Executor | None = None,  # 初始化函数，指定执行器
    ) -> None:
        if browser_type is None:
            browser_type = CONFIG.selenium_browser_type  # 如果未指定浏览器类型，则使用配置中的默认值
        self.browser_type = browser_type
        launch_kwargs = launch_kwargs or {}  # 如果未指定启动参数，则使用空字典
        if CONFIG.global_proxy and "proxy-server" not in launch_kwargs:  # 如果全局代理开启且启动参数中未包含代理服务器
            launch_kwargs["proxy-server"] = CONFIG.global_proxy  # 则将全局代理设置为启动参数中的代理服务器

        self.executable_path = launch_kwargs.pop("executable_path", None)  # 弹出启动参数中的可执行路径
        self.launch_args = [f"--{k}={v}" for k, v in launch_kwargs.items()]  # 将启动参数转换为列表
        self._has_run_precheck = False  # 标记是否已经运行了预检查
        self._get_driver = None  # 获取驱动程序
        self.loop = loop  # 事件循环
        self.executor = executor  # 执行器

    async def run(self, url: str, *urls: str) -> WebPage | list[WebPage]:  # 异步运行函数，接收 URL 和多个 URL
        await self._run_precheck()  # 等待运行预检查

        _scrape = lambda url: self.loop.run_in_executor(self.executor, self._scrape_website, url)  # 定义内部函数 _scrape

        if urls:  # 如果有多个 URL
            return await asyncio.gather(_scrape(url), *(_scrape(i) for i in urls))  # 则并发执行多个 URL 的抓取
        return await _scrape(url)  # 否则只抓取单个 URL

    async def _run_precheck(self):  # 异步运行预检查函数
        if self._has_run_precheck:  # 如果已经运行了预检查
            return  # 则直接返回
        self.loop = self.loop or asyncio.get_event_loop()  # 如果未指定事件循环，则使用默认事件循环
        self._get_driver = await self.loop.run_in_executor(  # 异步获取驱动程序
            self.executor,
            lambda: _gen_get_driver_func(self.browser_type, *self.launch_args, executable_path=self.executable_path),
        )
        self._has_run_precheck = True  # 标记已经运行了预检查

    def _scrape_website(self, url):  # 抓取网站函数
        with self._get_driver() as driver:  # 使用获取的驱动程序
            try:
                driver.get(url)  # 打开指定 URL
                WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))  # 等待页面加载完成
                inner_text = driver.execute_script("return document.body.innerText;")  # 获取页面文本
                html = driver.page_source  # 获取页面源代码
            except Exception as e:
                inner_text = f"Fail to load page content for {e}"  # 如果抓取失败，则返回错误信息
                html = ""  # 置空 HTML 内容
            return WebPage(inner_text=inner_text, html=html, url=url)  # 返回 WebPage 对象


_webdriver_manager_types = {
    "chrome": ("webdriver_manager.chrome", "ChromeDriverManager"),  # Chrome 驱动程序类型
    "firefox": ("webdriver_manager.firefox", "GeckoDriverManager"),  # Firefox 驱动程序类型
    "edge": ("webdriver_manager.microsoft", "EdgeChromiumDriverManager"),  # Edge 驱动程序类型
    "ie": ("webdriver_manager.microsoft", "IEDriverManager"),  # IE 驱动程序类型
}


class WDMHttpProxyClient(WDMHttpClient):  # WDMHttpProxyClient 类，继承自 WDMHttpClient
    def get(self, url, **kwargs):  # 重写 get 方法
        if "proxies" not in kwargs and CONFIG.global_proxy:  # 如果未指定代理且全局代理开启
            kwargs["proxies"] = {"all_proxy": CONFIG.global_proxy}  # 则设置代理
        return super().get(url, **kwargs)  # 调用父类的 get 方法


def _gen_get_driver_func(browser_type, *args, executable_path=None):  # 生成获取驱动程序的函数
    WebDriver = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.webdriver"), "WebDriver")  # 获取 WebDriver 类
    Service = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.service"), "Service")  # 获取 Service 类
    Options = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.options"), "Options")  # 获取 Options 类

    if not executable_path:  # 如果未指定可执行路径
        module_name, type_name = _webdriver_manager_types[browser_type]  # 获取驱动程序类型
        DriverManager = getattr(importlib.import_module(module_name), type_name)  # 获取驱动程序管理器
        driver_manager = DriverManager(download_manager=WDMDownloadManager(http_client=WDMHttpProxyClient()))  # 创建驱动程序管理器
        executable_path = driver_manager.install()  # 安装驱动程序

    def _get_driver():  # 获取驱动程序的函数
        options = Options()  # 创建选项对象
        options.add_argument("--headless")  # 添加参数
        options.add_argument("--enable-javascript")  # 添加参数
        if browser_type == "chrome":  # 如果是 Chrome 浏览器
            options.add_argument("--disable-gpu")  # 添加参数
            options.add_argument("--disable-dev-shm-usage")  # 添加参数
            options.add_argument("--no-sandbox")  # 添加参数
        for i in args:  # 遍历参数
            options.add_argument(i)  # 添加参数
        return WebDriver(options=deepcopy(options), service=Service(executable_path=executable_path))  # 返回 WebDriver 对象

    return _get_driver  # 返回获取驱动程序的函数

```