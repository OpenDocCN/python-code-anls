# MetaGPT源码解析 10

# `metagpt/tools/web_browser_engine.py`

该代码是一个Python脚本，实现了WebBrowserEngine的初始化和运行方法。WebBrowserEngine是一个用于从指定URL加载网页的类，可以实现对Selenium和Playwright浏览器的支持。具体来说，该脚本实现了以下功能：1. 通过环境变量或CONFIG配置从Platform指定工具链加载WebBrowser引擎；2. 加载指定工具链中的WebBrowser引擎，包括对PLAYWRIGHT、SELENIUM和CUSTOM engine的支持；3. WebBrowserEngine的run函数接受一个URL参数，用于运行WebBrowser从该URL加载网页；4. WebBrowserEngine可以实现run参数传递给函数内部使用，也可以实现多个URL参数传递给函数内部的不同Web页面；5. WebBrowserEngine可以与overload结合使用，实现asyncio协程下的异步运行。


```py
#!/usr/bin/env python

from __future__ import annotations

import importlib
from typing import Any, Callable, Coroutine, Literal, overload

from metagpt.config import CONFIG
from metagpt.tools import WebBrowserEngineType
from metagpt.utils.parse_html import WebPage


class WebBrowserEngine:
    def __init__(
        self,
        engine: WebBrowserEngineType | None = None,
        run_func: Callable[..., Coroutine[Any, Any, WebPage | list[WebPage]]] | None = None,
    ):
        engine = engine or CONFIG.web_browser_engine

        if engine == WebBrowserEngineType.PLAYWRIGHT:
            module = "metagpt.tools.web_browser_engine_playwright"
            run_func = importlib.import_module(module).PlaywrightWrapper().run
        elif engine == WebBrowserEngineType.SELENIUM:
            module = "metagpt.tools.web_browser_engine_selenium"
            run_func = importlib.import_module(module).SeleniumWrapper().run
        elif engine == WebBrowserEngineType.CUSTOM:
            run_func = run_func
        else:
            raise NotImplementedError
        self.run_func = run_func
        self.engine = engine

    @overload
    async def run(self, url: str) -> WebPage:
        ...

    @overload
    async def run(self, url: str, *urls: str) -> list[WebPage]:
        ...

    async def run(self, url: str, *urls: str) -> WebPage | list[WebPage]:
        return await self.run_func(url, *urls)


```

这段代码是一个Python脚本，主要作用是定义了一个名为“main”的函数，该函数接受一个URL和一个或多个URL参数，然后使用WebBrowserEngine类（来自playwright库，如果使用的是selenium库，则应该是from selenium import WebBrowserEngine）在这些URL上运行指定的任务。

具体来说，这段代码执行以下操作：

1. 定义了名为“main”的函数，该函数有一个参数“url”，代表要访问的URL。
2. 定义了一个名为“**kwargs”的参数，用于传递给WebBrowserEngine类的多个参数。
3. 在函数内部，使用Literal（constant）类型定义了引擎类型为"playwright"或"selenium"。根据实际使用情况，可能需要使用环境变量的值来确定这一点。
4. 创建了一个名为“WebBrowserEngine”的类，该类实现了从Web浏览器中启动URL的任务。
5. 使用“fire”模块的“Fire”函数来运行“main”函数，即将定义的“main”函数作为fire.Fire函数的参数传入。


```py
if __name__ == "__main__":
    import fire

    async def main(url: str, *urls: str, engine_type: Literal["playwright", "selenium"] = "playwright", **kwargs):
        return await WebBrowserEngine(WebBrowserEngineType(engine_type), **kwargs).run(url, *urls)

    fire.Fire(main)

```

# `metagpt/tools/web_browser_engine_playwright.py`

这段代码是一个Python脚本，使用了Python 3.7+的虚拟环境，并在文件末尾以 #!/usr/bin/env python 的形式配置了环境。接下来是主要的功能和模块导入：

1. 从标准库中导入了一个未来时注释类型：从Python 3.6开始引入的。
2. 导入asyncio，这是Python 3.7引入的异步编程库，用于编写与网络和服务器交互的代码，以及事件循环的代码。
3. 导入sys，这是Python 2.7引入的用于编写操作系统功能的标准库，提供访问操作系统函数和模块的接口。
4. 从pathlib中导入Path，这是Python标准库中的pathlib模块，用于处理文件和目录操作。
5. 从typing中导入Literal，这是用于定义单个变量的类型注释。
6. 从playwright中导入async_playwright，这是Google Chrome浏览器的Python API，用于开发与浏览器交互的代码。
7. 从metagpt中导入config、logger和utils.parse_html，这是metagpt的Python API，用于处理元数据解析、日志记录和HTML解析等功能。
8. 导入了函数和类：playwright中的几个函数和一个类。

总结：这段代码是一个用于处理浏览器与服务器交互操作的Python脚本，通过导入一些Python标准库、第三方库和函数式编程模型的机制来实现。


```py
#!/usr/bin/env python
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

from playwright.async_api import async_playwright

from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.utils.parse_html import WebPage


```



This is a class called `ScrapePlayer` which appears to be a tool for scraping content from a webpage using the Playwright library.

The `ScrapePlayer` class has two methods: `_scrape` and `_run_precheck`.

The `_scrape` method takes two arguments: `browser` and `url`. It opens a new browser context and a new page, and then navigates to the specified URL and executes some JavaScript code to scrape the HTML content of the page.

The `_run_precheck` method is a utility method that checks whether `_scrape` should be runnning. It sets the `executable_path` attribute of the `browser` object to the path of the Playwright executable, and also sets the `ALL_PROXY` environment variable to `CONFIG.global_proxy` if it is set.

It appears that `ScrapePlayer` also has a `_install_browsers` method which installs the necessary browsers for `ScrapePlayer` to work properly.

Note that `ScrapePlayer` is intended for internal use and should not be used directly in production.


```py
class PlaywrightWrapper:
    """Wrapper around Playwright.

    To use this module, you should have the `playwright` Python package installed and ensure that
    the required browsers are also installed. You can install playwright by running the command
    `pip install metagpt[playwright]` and download the necessary browser binaries by running the
    command `playwright install` for the first time.
    """

    def __init__(
        self,
        browser_type: Literal["chromium", "firefox", "webkit"] | None = None,
        launch_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        if browser_type is None:
            browser_type = CONFIG.playwright_browser_type
        self.browser_type = browser_type
        launch_kwargs = launch_kwargs or {}
        if CONFIG.global_proxy and "proxy" not in launch_kwargs:
            args = launch_kwargs.get("args", [])
            if not any(str.startswith(i, "--proxy-server=") for i in args):
                launch_kwargs["proxy"] = {"server": CONFIG.global_proxy}
        self.launch_kwargs = launch_kwargs
        context_kwargs = {}
        if "ignore_https_errors" in kwargs:
            context_kwargs["ignore_https_errors"] = kwargs["ignore_https_errors"]
        self._context_kwargs = context_kwargs
        self._has_run_precheck = False

    async def run(self, url: str, *urls: str) -> WebPage | list[WebPage]:
        async with async_playwright() as ap:
            browser_type = getattr(ap, self.browser_type)
            await self._run_precheck(browser_type)
            browser = await browser_type.launch(**self.launch_kwargs)
            _scrape = self._scrape

            if urls:
                return await asyncio.gather(_scrape(browser, url), *(_scrape(browser, i) for i in urls))
            return await _scrape(browser, url)

    async def _scrape(self, browser, url):
        context = await browser.new_context(**self._context_kwargs)
        page = await context.new_page()
        async with page:
            try:
                await page.goto(url)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                html = await page.content()
                inner_text = await page.evaluate("() => document.body.innerText")
            except Exception as e:
                inner_text = f"Fail to load page content for {e}"
                html = ""
            return WebPage(inner_text=inner_text, html=html, url=url)

    async def _run_precheck(self, browser_type):
        if self._has_run_precheck:
            return

        executable_path = Path(browser_type.executable_path)
        if not executable_path.exists() and "executable_path" not in self.launch_kwargs:
            kwargs = {}
            if CONFIG.global_proxy:
                kwargs["env"] = {"ALL_PROXY": CONFIG.global_proxy}
            await _install_browsers(self.browser_type, **kwargs)

            if self._has_run_precheck:
                return

            if not executable_path.exists():
                parts = executable_path.parts
                available_paths = list(Path(*parts[:-3]).glob(f"{self.browser_type}-*"))
                if available_paths:
                    logger.warning(
                        "It seems that your OS is not officially supported by Playwright. "
                        "Try to set executable_path to the fallback build version."
                    )
                    executable_path = available_paths[0].joinpath(*parts[-2:])
                    self.launch_kwargs["executable_path"] = str(executable_path)
        self._has_run_precheck = True


```

这段代码定义了一个名为 `_install_lock` 的全局变量，以及一个名为 `_install_browsers` 的异步函数。

`_get_install_lock` 函数用于获取一个全局锁，用于确保在多个进程之间对 `_install_cache` 的访问是线程安全的。

`_install_browsers` 函数用于安装指定列表中的浏览器。它使用 `asyncio` 包中的 `create_subprocess_exec` 函数创建一个新进程来运行 `playwright` 命令，指定要安装的浏览器，并将它们的依赖项添加到 `playwright` 配置中。然后，它使用 `asyncio` 包中的 `gather` 函数等待新进程的输出并将其传递给 `logger.info` 和 `logger.warning` 函数。如果 `install` 成功，它将向 `logger.info` 函数发送一条消息，否则将向 `logger.warning` 函数发送一条消息，并将 `_install_cache` 更新为安装的浏览器列表。


```py
def _get_install_lock():
    global _install_lock
    if _install_lock is None:
        _install_lock = asyncio.Lock()
    return _install_lock


async def _install_browsers(*browsers, **kwargs) -> None:
    async with _get_install_lock():
        browsers = [i for i in browsers if i not in _install_cache]
        if not browsers:
            return
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "playwright",
            "install",
            *browsers,
            # "--with-deps",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )

        await asyncio.gather(_log_stream(process.stdout, logger.info), _log_stream(process.stderr, logger.warning))

        if await process.wait() == 0:
            logger.info("Install browser for playwright successfully.")
        else:
            logger.warning("Fail to install browser for playwright.")
        _install_cache.update(browsers)


```

这段代码定义了一个名为 `_log_stream` 的异步函数，它使用 `asyncio` 库的 `AbstractItem` 类来读取 `url` 参数并获取 `browser_type` 参数。

该函数的作用是在一个无限循环中读取 `url` 参数，并执行 `log_func` 函数。`log_func` 函数接收一个 `line` 参数，它是一个字符串，通过 `decode()` 方法将其转换为字节字符串，并使用 `strip()` 方法去除头和尾的空格。然后，将 `line` 参数传递给 `log_func` 函数，它将接收一个字符串，并将其作为 `browser_type` 的参数，以便在日志中记录下来。

该代码还定义了一个名为 `_install_lock` 的同步锁，用于在两个不同的 `asyncio` 事件循环之间同步 `_install_cache` 集合。`_install_cache` 集合用于存储 `browser_type` 的映射，以便在多个 `asyncio` 事件循环之间共享。

最后，该代码使用 `fire` 库的 `Fire` 函数来运行 `main` 函数。`main` 函数是一个异步函数，它接受一个 `url` 参数，一个或多个 `url` 参数和一个 `browser_type` 参数。它使用 `PlaywrightWrapper` 类来加载 `browser_type` 的 `asyncio` 库，并使用 `run` 方法将 `url` 和 `browser_type` 参数传递给 `PlaywrightWrapper` 类。


```py
async def _log_stream(sr, log_func):
    while True:
        line = await sr.readline()
        if not line:
            return
        log_func(f"[playwright install browser]: {line.decode().strip()}")


_install_lock: asyncio.Lock = None
_install_cache = set()


if __name__ == "__main__":
    import fire

    async def main(url: str, *urls: str, browser_type: str = "chromium", **kwargs):
        return await PlaywrightWrapper(browser_type, **kwargs).run(url, *urls)

    fire.Fire(main)

```

# `metagpt/tools/web_browser_engine_selenium.py`

这段代码是一个Python脚本，使用了Python未来的特性（即[[训练时间]]）。它通过导入异步io、异步psutil库，创建了一个事件循环（asyncio loop），并使用asgi技术将Python的上下文切换为异步模式。

接下来，它通过导入selenium库和metagpt库，使用selenium库从浏览器中获取页面元素，使用metagpt库解析页面内容。

然后，代码定义了一个Copy类，该类使用deepcopy库对复制对象进行深拷贝。

最后，代码导入了未来函数，这意味着该代码在Python 3.6及更高版本中可能会有所改动。


```py
#!/usr/bin/env python
from __future__ import annotations

import asyncio
import importlib
from concurrent import futures
from copy import deepcopy
from typing import Literal

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from metagpt.config import CONFIG
from metagpt.utils.parse_html import WebPage


```

这是一个基于 launchd 网页组件的 Python 实现。这个组件接收一个 `executable_path` 和一个 `launch_args` 参数。`executable_path` 是要运行的 JavaScript 脚本文件的路径，`launch_args` 是用于指定在运行脚本时需要传递给脚本的参数。`run` 方法用于运行脚本，它接收一个 `url` 和一些 `urls` 参数。如果 `urls` 是空括号，那么 `run` 将返回一个空列表。`_scrape_website` 方法用于从 `executable_path` 中加载 `scrape_website` 函数，这个函数将收到的 `url` 参数渲染成网页内容并返回。

注意：这个组件没有做任何错误处理。在实际应用中，您需要添加一些错误处理以应对可能出现的错误情况。


```py
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
        browser_type: Literal["chrome", "firefox", "edge", "ie"] | None = None,
        launch_kwargs: dict | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: futures.Executor | None = None,
    ) -> None:
        if browser_type is None:
            browser_type = CONFIG.selenium_browser_type
        self.browser_type = browser_type
        launch_kwargs = launch_kwargs or {}
        if CONFIG.global_proxy and "proxy-server" not in launch_kwargs:
            launch_kwargs["proxy-server"] = CONFIG.global_proxy

        self.executable_path = launch_kwargs.pop("executable_path", None)
        self.launch_args = [f"--{k}={v}" for k, v in launch_kwargs.items()]
        self._has_run_precheck = False
        self._get_driver = None
        self.loop = loop
        self.executor = executor

    async def run(self, url: str, *urls: str) -> WebPage | list[WebPage]:
        await self._run_precheck()

        _scrape = lambda url: self.loop.run_in_executor(self.executor, self._scrape_website, url)

        if urls:
            return await asyncio.gather(_scrape(url), *(_scrape(i) for i in urls))
        return await _scrape(url)

    async def _run_precheck(self):
        if self._has_run_precheck:
            return
        self.loop = self.loop or asyncio.get_event_loop()
        self._get_driver = await self.loop.run_in_executor(
            self.executor,
            lambda: _gen_get_driver_func(self.browser_type, *self.launch_args, executable_path=self.executable_path),
        )
        self._has_run_precheck = True

    def _scrape_website(self, url):
        with self._get_driver() as driver:
            try:
                driver.get(url)
                WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                inner_text = driver.execute_script("return document.body.innerText;")
                html = driver.page_source
            except Exception as e:
                inner_text = f"Fail to load page content for {e}"
                html = ""
            return WebPage(inner_text=inner_text, html=html, url=url)


```

这段代码定义了一个名为 `_webdriver_manager_types` 的字典，它包含了五种不同浏览器的 WebDriver 管理器。每个管理器都包含一个 `webdriver_manager` 和一个 `Service` 对象，这些对象用于管理特定浏览器的 WebDriver。

代码中定义了一个名为 `_gen_get_driver_func` 的函数，它接收两个参数：`browser_type` 和 `*args`，它根据 `browser_type` 加载相应的 WebDriver 管理器，并允许用户指定 `executable_path` 参数。函数首先根据 `browser_type` 从 `_webdriver_manager_types` 字典中获取相应的 WebDriver 管理器，然后创建一个 `WebDriver` 对象、一个 `Service` 对象，并将它们组合在一起，最后允许用户加载特定 executable_path。


```py
_webdriver_manager_types = {
    "chrome": ("webdriver_manager.chrome", "ChromeDriverManager"),
    "firefox": ("webdriver_manager.firefox", "GeckoDriverManager"),
    "edge": ("webdriver_manager.microsoft", "EdgeChromiumDriverManager"),
    "ie": ("webdriver_manager.microsoft", "IEDriverManager"),
}


def _gen_get_driver_func(browser_type, *args, executable_path=None):
    WebDriver = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.webdriver"), "WebDriver")
    Service = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.service"), "Service")
    Options = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.options"), "Options")

    if not executable_path:
        module_name, type_name = _webdriver_manager_types[browser_type]
        DriverManager = getattr(importlib.import_module(module_name), type_name)
        driver_manager = DriverManager()
        # driver_manager.driver_cache.find_driver(driver_manager.driver))
        executable_path = driver_manager.install()

    def _get_driver():
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--enable-javascript")
        if browser_type == "chrome":
            options.add_argument("--no-sandbox")
        for i in args:
            options.add_argument(i)
        return WebDriver(options=deepcopy(options), service=Service(executable_path=executable_path))

    return _get_driver


```

这段代码使用了Python的异步编程库——asyncio和Python的Selenium库，实现了自动运行一个URL指定的测试套件的自动测试脚本。

具体来说，这段代码定义了一个名为`main`的函数，该函数使用了异步编程中的`async`关键字，说明该函数是一个异步函数。该函数接收一个URL参数`url`，以及一个或多个URL参数`urls`，和一个浏览器类型参数`browser_type`，同时接受一个或多个关键字参数`kwargs`。

该函数内部使用`seleniumWrapper`函数从异步上下文中获取一个 Selenium 版本，然后使用该版本的 `run` 方法运行一个URL指定的测试套件，并返回该测试套件的运行结果。这里使用 `asyncio` 库中的 `await` 关键字，说明该函数是使用异步编程的方式执行测试。


```py
if __name__ == "__main__":
    import fire

    async def main(url: str, *urls: str, browser_type: str = "chrome", **kwargs):
        return await SeleniumWrapper(browser_type, **kwargs).run(url, *urls)

    fire.Fire(main)

```

# `metagpt/tools/__init__.py`

这段代码定义了一个名为`SearchEngineType`的枚举类型，它有五种不同的枚举值，分别为`SERPAPI_GOOGLE`,`SERPER_GOOGLE`,`DIRECT_GOOGLE`,`DUCK_DUCK_GO`和`CUSTOM_ENGINE`。

`SearchEngineType`枚举类型定义了一种枚举类型，用于指定搜索引擎。这个枚举类型提供了用于从给定选项中选择一个搜索引擎的方法，可以用于程序中的搜索功能。

具体来说，`SearchEngineType`枚举类型定义了以下几种搜索引擎：

- `SERPAPI_GOOGLE`：谷歌搜索引擎
- `SERPER_GOOGLE`：谷歌搜索引擎的替代品，可能是由于一些原因无法使用谷歌搜索引擎
- `DIRECT_GOOGLE`：直接使用谷歌搜索引擎
- `DUCK_DUCK_GO`：来自韩国 search.io 的搜索引擎，具体信息和使用情况不明
- `CUSTOM_ENGINE`：自定义搜索引擎，可以通过 `SearchEngineType` 函数进行指定。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:35
@Author  : alexanderwu
@File    : __init__.py
"""


from enum import Enum


class SearchEngineType(Enum):
    SERPAPI_GOOGLE = "serpapi"
    SERPER_GOOGLE = "serper"
    DIRECT_GOOGLE = "google"
    DUCK_DUCK_GO = "ddg"
    CUSTOM_ENGINE = "custom"


```

这段代码定义了一个名为 "WebBrowserEngineType" 的枚举类型，它继承了 "Enum" 类型。

具体来说，这个枚举类型定义了三种可能的浏览器引擎类型：PLAYWRIGHT(playwright)、SELENIUM(selenium) 和 CUSTOM(custom)。

每种枚举类型的值都有一个对应的枚举类型成员，例如 PLAYWRIGHT 的成员可以是 "playwright"、"chrome" 或 "firefox"，而 SELENIUM 的成员可以是 "webdriver" 或 "chrome driver"。

枚举类型通常用于代码中，以便在不同的上下文中使用不同的枚举值。在这个例子中，它可以帮助作者在代码中选择使用哪种浏览器引擎，而不需要显式地指定浏览器驱动程序的名称。


```py
class WebBrowserEngineType(Enum):
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    CUSTOM = "custom"

```

# `metagpt/utils/common.py`

该代码是一个Python脚本，用于处理文本文件中的语法错误。具体来说，它实现了以下功能：

1. 导入需要用到的库：ast、contextlib、inspect、os、platform和re。
2. 使用inspect库的getsource方法读取文本文件的内容。
3. 使用platform库的 listdir 函数获取文本文件所在的目录。
4. 使用os库的 mkdir 函数创建一个新目录，并将当前工作目录移动到该目录。
5. 使用re库的 findall 函数提取文本文件中的所有正则表达式匹配的行。
6. 对于每个匹配的行，使用inspect库的 getsource 方法获取该行的源代码。
7. 使用ast库的 parse 函数解析源代码中的语法错误。
8. 如果解析成功，使用inspect库的 exec 方法运行源代码中的 code。
9. 如果没有解析成功，打印错误信息并退出程序。

总之，该程序的作用是读取文本文件中的语法错误，并尝试自动修复它们。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:07
@Author  : alexanderwu
@File    : common.py
"""
import ast
import contextlib
import inspect
import os
import platform
import re
from typing import List, Tuple, Union

```

这段代码是一个Python函数，名为`check_cmd_exists`，它接受一个参数`command`，用于检查命令是否存在于系统。函数内部使用`os.system`方法来运行一个命令，并通过获取命令运行结果来判断命令是否存在。具体来说，如果命令是在Windows系统上运行的，那么函数会首先在命令提示符（Windows的命令提示符是一个窗口，类似于Linux的终端）中搜索该命令；否则，函数会运行一个复杂的命令，该命令会尝试在命令行中查找该命令，并输出结果。如果命令存在，函数返回0；否则，函数返回一个非0（表示错误）。


```py
from metagpt.logs import logger


def check_cmd_exists(command) -> int:
    """检查命令是否存在
    :param command: 待检查的命令
    :return: 如果命令存在，返回0，如果不存在，返回非0
    """
    if platform.system().lower() == "windows":
        check_command = "where " + command
    else:
        check_command = "command -v " + command + ' >/dev/null 2>&1 || { echo >&2 "no mermaid"; exit 1; }'
    result = os.system(check_command)
    return result


```

This is a Python function called `OutputParser.extract_struct` that takes in a string `text` and extracts the structure of the text using the specified data type (list or dict).

The function has two parameters: `text` and `data_type`. The `text` parameter is the input text, and the `data_type` parameter specifies the data type for the extracted structure.

The function first finds the first "[" or "{" and the last "]" or "}" character in the text by code that is similar to the following:
```pylua
if data_type is list:
   start_index = text.find("[" if data_type is list else "{")
   end_index = text.rfind("]" if data_type is list else "}")
```
If the `data_type` is `list`, the code will look like this:
```pypython
if data_type is list:
   start_index = text.find("[" if data_type is list else "{")
   end_index = text.rfind("]")
```
If the `data_type` is `dict`, the code will look like this:
```pypython
if data_type is dict:
   start_index = text.find("{")
   end_index = text.rfind("}")
```
The function then attempts to convert the text to the specified data type using `ast.literal_eval` method. If the text can be successfully converted, the function will return the result.
```pypython
try:
   result = ast.literal_eval(structure_text)

   # Ensure the result matches the specified data type
   if isinstance(result, list) or isinstance(result, dict):
       return result

   raise ValueError(f"The extracted structure is not a {data_type}.")
```
If the `ast.literal_eval` method fails, the function will raise an exception with a message related to the `data_type`.

If the structure is not found in the text, the function will raise an exception.
```pyjava
except (ValueError, SyntaxError) as e:
   raise Exception(f"Error while extracting and parsing the {data_type}: {e}")
```
Finally, if the `data_type` is `list`, the function will return an empty list.
```pysql
return [] if data_type is list else {}
```
If the `data_type` is `dict`, the function will return the dictionary.
```pyvbnet
return result_dict if data_type is dict else {}
```
Overall, this function is useful for extracting the structure of text in a specific data type.


```py
class OutputParser:
    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split("##")

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
                # LLM可能出错，在这里做一下修正
                if block_title[-1] == ":":
                    block_title = block_title[:-1]
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, text: str, lang: str = "") -> str:
        pattern = rf"```{lang}.*?\s+(.*?)```py"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise Exception
        return code

    @classmethod
    def parse_str(cls, text: str):
        text = text.split("=")[-1]
        text = text.strip().strip("'").strip('"')
        return text

    @classmethod
    def parse_file_list(cls, text: str) -> list[str]:
        # Regular expression pattern to find the tasks list.
        pattern = r"\s*(.*=.*)?(\[.*\])"

        # Extract tasks list string using regex.
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            tasks = text.split("\n")
        return tasks

    @staticmethod
    def parse_python_code(text: str) -> str:
        for pattern in (
            r"(.*?```python.*?\s+)?(?P<code>.*)(```py.*?)",
            r"(.*?```python.*?\s+)?(?P<code>.*)",
        ):
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            code = match.group("code")
            if not code:
                continue
            with contextlib.suppress(Exception):
                ast.parse(code)
                return code
        raise ValueError("Invalid python code")

    @classmethod
    def parse_data(cls, data):
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                pass

            # 尝试解析list
            try:
                content = cls.parse_file_list(text=content)
            except Exception:
                pass
            parsed_data[block] = content
        return parsed_data

    @classmethod
    def parse_data_with_mapping(cls, data, mapping):
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                pass
            typing_define = mapping.get(block, None)
            if isinstance(typing_define, tuple):
                typing = typing_define[0]
            else:
                typing = typing_define
            if typing == List[str] or typing == List[Tuple[str, str]] or typing == List[List[str]]:
                # 尝试解析list
                try:
                    content = cls.parse_file_list(text=content)
                except Exception:
                    pass
            # TODO: 多余的引号去除有风险，后期再解决
            # elif typing == str:
            #     # 尝试去除多余的引号
            #     try:
            #         content = cls.parse_str(text=content)
            #     except Exception:
            #         pass
            parsed_data[block] = content
        return parsed_data

    @classmethod
    def extract_struct(cls, text: str, data_type: Union[type(list), type(dict)]) -> Union[list, dict]:
        """Extracts and parses a specified type of structure (dictionary or list) from the given text.
        The text only contains a list or dictionary, which may have nested structures.

        Args:
            text: The text containing the structure (dictionary or list).
            data_type: The data type to extract, can be "list" or "dict".

        Returns:
            - If extraction and parsing are successful, it returns the corresponding data structure (list or dictionary).
            - If extraction fails or parsing encounters an error, it throw an exception.

        Examples:
            >>> text = 'xxx [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}] xxx'
            >>> result_list = OutputParser.extract_struct(text, "list")
            >>> print(result_list)
            >>> # Output: [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}]

            >>> text = 'xxx {"x": 1, "y": {"a": 2, "b": {"c": 3}}} xxx'
            >>> result_dict = OutputParser.extract_struct(text, "dict")
            >>> print(result_dict)
            >>> # Output: {"x": 1, "y": {"a": 2, "b": {"c": 3}}}
        """
        # Find the first "[" or "{" and the last "]" or "}"
        start_index = text.find("[" if data_type is list else "{")
        end_index = text.rfind("]" if data_type is list else "}")

        if start_index != -1 and end_index != -1:
            # Extract the structure part
            structure_text = text[start_index : end_index + 1]

            try:
                # Attempt to convert the text to a Python data type using ast.literal_eval
                result = ast.literal_eval(structure_text)

                # Ensure the result matches the specified data type
                if isinstance(result, list) or isinstance(result, dict):
                    return result

                raise ValueError(f"The extracted structure is not a {data_type}.")

            except (ValueError, SyntaxError) as e:
                raise Exception(f"Error while extracting and parsing the {data_type}: {e}")
        else:
            logger.error(f"No {data_type} found in the text.")
            return [] if data_type is list else {}


```py

以上代码定义了一个 `Block` 类，用于解析 Git 提交中的代码块。`Block` 类包含了以下方法：

* `parse_code(block, text, lang)`: 将给定的代码块解析成一个可读的格式。这个方法将 `text` 参数中的内容与给定的 `lang` 参数一起作为参数传递，以便解析语言特定的代码块。这个方法返回解析后的代码。
* `parse_str(block, text, lang)`: 将给定的代码块解析成一个字符串，其中的代码块用单引号或双引号括起来的字符串。这个方法将 `text` 参数中的内容与给定的 `lang` 参数一起作为参数传递，以便解析语言特定的代码块。这个方法返回解析后的字符串。
* `parse_file_list(block, text, lang)`: 将给定的代码文件中的代码块列表提取出来，并将它们打印出来。这个方法将 `text` 参数中的内容与给定的 `lang` 参数一起作为参数传递，以便解析语言特定的代码块。这个方法返回一个包含代码块列表的字符串列表。

`Block` 类的实例可以用来提取代码块中的内容，例如：
```makefile
import re

class Block:
   @classmethod
   def parse_code(cls, block, text, lang):
       if block.strip() != "":
           # 将block的标题和内容分开，并分别去掉前后的空白字符
           block_title, block_content = block.split("\n", 1)
           block_dict = {}
           for title in block_title.strip().split(": "):
               key = title.strip()
               value = block_content.strip()
               block_dict[key] = value
           return block_dict
       return block

block_dict = Block.parse_code("[![走 AxCkY4[[小萱出汗反抗
```py  # just add the text to the block_dict

```
print(block_dict)
# Output: {'@open_source': '[![走 AxCkY4[[小萱出汗反抗'}}
```py

```python
import re

class Block:
   @classmethod
   def parse_str(cls, block, text, lang):
       code = cls.parse_code(block, text, lang)
       code = code.split("=")[-1]
       code = code.strip().strip("'")
       return code

block_str = "`@open_source'
print(block_str)
# Output: '@open_source'
```py

```python
class Block:
   @classmethod
   def parse_file_list(cls, block, text, lang):
       # Regular expression pattern to find the tasks list.
       code = cls.parse_str(block, text, lang)
       # print(code)
       pattern = r"\s*(.*=.*)?(\[.*\])"

       # Extract tasks list string using regex.
       match = re.search(pattern, code, re.DOTALL)
       if match:
           tasks_list_str = match.group(2)

           # Convert string representation of list to a Python list using ast.literal_eval.
           tasks = ast.literal_eval(tasks_list_str)
       else:
           raise Exception
       return tasks

block_list = "[[1]|[2]|[3]"
print(block_list)
# Output: [1, 2, 3]
```py


```
class CodeParser:
    @classmethod
    def parse_block(cls, block: str, text: str) -> str:
        blocks = cls.parse_blocks(text)
        for k, v in blocks.items():
            if block in k:
                return v
        return ""

    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split("##")

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, block: str, text: str, lang: str = "") -> str:
        if block:
            text = cls.parse_block(block, text)
        pattern = rf"```py{lang}.*?\s+(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            logger.error(f"{pattern} not match following text:")
            logger.error(text)
            # raise Exception
            return text  # just assume original text is code
        return code

    @classmethod
    def parse_str(cls, block: str, text: str, lang: str = ""):
        code = cls.parse_code(block, text, lang)
        code = code.split("=")[-1]
        code = code.strip().strip("'").strip('"')
        return code

    @classmethod
    def parse_file_list(cls, block: str, text: str, lang: str = "") -> list[str]:
        # Regular expression pattern to find the tasks list.
        code = cls.parse_code(block, text, lang)
        # print(code)
        pattern = r"\s*(.*=.*)?(\[.*\])"

        # Extract tasks list string using regex.
        match = re.search(pattern, code, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            raise Exception
        return tasks


```py

这段代码定义了一个名为 `NoMoneyException` 的异常类。当一个操作无法完成时，这是因为缺少足够的资金。

在这个异常类中，有一个 `__init__` 方法，用于初始化发生异常时的金额和消息。当这个异常类继承自 `Exception` 类时，`__init__` 方法将调用父类的 `__init__` 方法，从而提供基本的功能。

另外，这个异常类有一个名为 `__str__` 的方法，当这个对象被打印时，将返回一个字符串，其中包含消息和金额，以及发生这个异常的类名。

最后，该异常类中还有一个名为 `print_members` 的函数，用于打印出当前模块中所有类的信息，包括类的名称、成员变量和方法。


```
class NoMoneyException(Exception):
    """Raised when the operation cannot be completed due to insufficient funds"""

    def __init__(self, amount, message="Insufficient funds"):
        self.amount = amount
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} -> Amount required: {self.amount}"


def print_members(module, indent=0):
    """
    https://stackoverflow.com/questions/1796180/how-can-i-get-a-list-of-all-classes-within-current-module-in-python
    :param module:
    :param indent:
    :return:
    """
    prefix = " " * indent
    for name, obj in inspect.getmembers(module):
        print(name, obj)
        if inspect.isclass(obj):
            print(f"{prefix}Class: {name}")
            # print the methods within the class
            if name in ["__class__", "__base__"]:
                continue
            print_members(obj, indent + 2)
        elif inspect.isfunction(obj):
            print(f"{prefix}Function: {name}")
        elif inspect.ismethod(obj):
            print(f"{prefix}Method: {name}")


```py

这段代码定义了一个名为 `parse_recipient` 的函数，它接受一个字符串参数 `text`。函数的作用是解析字符串中的收件人信息，如果成功找到收件人，则返回该收件人的姓名，否则返回一个空字符串。

函数的核心部分是使用正则表达式模式匹配收件人信息。正则表达式模式为 `## Send To:\s*([A-Za-z]+)\s*?`，其中 `##` 表示字符串中的这部分是整个匹配的部分，需要被匹配的整个字符串不能被分割成多个部分；`Send To:` 和 `?` 分别表示匹配收件人名称和收件人名称的位置；`([A-Za-z]+)` 表示匹配一个或多个字母、数字或下划线的字符；`\s*?` 表示匹配后面可以有空字符，但整个字符串必须为空字符串。

函数的接收者参数 `text` 是一个字符串类型，因此函数可以安全地使用其中的字符串操作方法，如搜索、提取等。


```
def parse_recipient(text):
    pattern = r"## Send To:\s*([A-Za-z]+)\s*?"  # hard code for now
    recipient = re.search(pattern, text)
    return recipient.group(1) if recipient else ""

```py

# `metagpt/utils/custom_decoder.py`

This appears to be a Python implementation of the `parse_string` function that takes a string and an index into that string. It appears to handle various cases for extracting value from the string, such as finding the index of a指定 character, finding the index of the nextchar, finding the nextchar, finding the index of the nextchar, finding the nextchar继续处，handling the case if the nextchar is '}' and so on. It also appears to support various nextchar options such as `continue`, `delimiter`, `strict`, `name`, `ignore`, `normalize`, `负面`, `raise_error`, `cache`, `save`, `timeout`, `schema`, `weights`, `typecode`, `附带选项`, `recover`, `initialize`, `calculate]。`

Please let me know if this implementation is accurate or if there's anything else you need.


```
import json
import re
from json import JSONDecodeError
from json.decoder import _decode_uXXXX

NUMBER_RE = re.compile(r"(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?", (re.VERBOSE | re.MULTILINE | re.DOTALL))


def py_make_scanner(context):
    parse_object = context.parse_object
    parse_array = context.parse_array
    parse_string = context.parse_string
    match_number = NUMBER_RE.match
    strict = context.strict
    parse_float = context.parse_float
    parse_int = context.parse_int
    parse_constant = context.parse_constant
    object_hook = context.object_hook
    object_pairs_hook = context.object_pairs_hook
    memo = context.memo

    def _scan_once(string, idx):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar == '"' or nextchar == "'":
            if idx + 2 < len(string) and string[idx + 1] == nextchar and string[idx + 2] == nextchar:
                # Handle the case where the next two characters are the same as nextchar
                return parse_string(string, idx + 3, strict, delimiter=nextchar * 3)  # triple quote
            else:
                # Handle the case where the next two characters are not the same as nextchar
                return parse_string(string, idx + 1, strict, delimiter=nextchar)
        elif nextchar == "{":
            return parse_object((string, idx + 1), strict, _scan_once, object_hook, object_pairs_hook, memo)
        elif nextchar == "[":
            return parse_array((string, idx + 1), _scan_once)
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            return None, idx + 4
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            return True, idx + 4
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            return False, idx + 5

        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return res, m.end()
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return parse_constant("NaN"), idx + 3
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return parse_constant("Infinity"), idx + 8
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return parse_constant("-Infinity"), idx + 9
        else:
            raise StopIteration(idx)

    def scan_once(string, idx):
        try:
            return _scan_once(string, idx)
        finally:
            memo.clear()

    return scan_once


```py

这段代码定义了一些正则表达式的 Flags，用于匹配输入文本中的特定字符或子串。

首先，定义了一个名为 FLAGS 的变量，它使用了 re.VERBOSE、re.MULTILINE 和 re.DOTALL 这三个正则表达式，这三个正则表达式用于配置正则表达式的行为。

接着，定义了一个名为 STRINGCHUNK 的正则表达式，它使用了 re.compile 方法来创建一个表达式，这个表达式匹配以某个字符或其下属标识符开头，后面可以跟一个或多个非转移符包含的字符或其下属标识符，并且可以不换行。

然后，定义了三个名为 STRINGCHUNK_SINGLEQUOTE、STRINGCHUNK_TRIPLE_DOUBLE_QUOTE 和 STRINGCHUNK_TRIPLE_SINGLEQUOTE 的正则表达式，它们也使用了 re.compile 方法来创建，但分别匹配以 double quote 或 single quote 开头的字符串，这个字符串中可以包含一个或多个 double quote 或 single quote，并且可以不换行。

接着，定义了一个名为 BACKSLASH 的字典，它定义了多个键值对，每个键值对对应一个 backslash 字符，这个字典用于将 backslash 转义为双引号。

最后，这些正则表达式被用于匹配输入文本，如果匹配成功，将返回匹配的元组（匹配到的字符串）作为 output 返回。


```
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
STRINGCHUNK = re.compile(r'(.*?)(["\\\x00-\x1f])', FLAGS)
STRINGCHUNK_SINGLEQUOTE = re.compile(r"(.*?)([\'\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_DOUBLE_QUOTE = re.compile(r"(.*?)(\"\"\"|[\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_SINGLEQUOTE = re.compile(r"(.*?)('''|[\\\x00-\x1f])", FLAGS)
BACKSLASH = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
```py

This is a JavaScript function that takes a string `s` and returns a list of key-value pairs that were extracted from the string.

It works by first scanning the string for the nextchar character, which is either a double quote or a closing curly brace, and then looking for the nextchar in each pair. If the nextchar is a closing curly brace, it opens thebracket and reads the property name.

If the nextchar is a double quote, it assumes the property name is enclosed in double quotes and reads the property name. If the nextchar is not a closing curly brace or double quote, the function raises an error.

The function also includes a try-except block to handle errors, such as if the string is not a valid JSON string.

The function uses the `scanstring`, `scan_once`, and `memo_get` functions to extract the property name and value from the string. The `scanstring` function scans the string for the nextchar character, while `scan_once` function reads the property name from the string. The `memo_get` function returns the cached value for the given key.

The function also includes a try-except block to handle errors, such as if the string is not a valid JSON string.


```
WHITESPACE = re.compile(r"[ \t\n\r]*", FLAGS)
WHITESPACE_STR = " \t\n\r"


def JSONObject(
    s_and_end, strict, scan_once, object_hook, object_pairs_hook, memo=None, _w=WHITESPACE.match, _ws=WHITESPACE_STR
):
    """Parse a JSON object from a string and return the parsed object.

    Args:
        s_and_end (tuple): A tuple containing the input string to parse and the current index within the string.
        strict (bool): If `True`, enforces strict JSON string decoding rules.
            If `False`, allows literal control characters in the string. Defaults to `True`.
        scan_once (callable): A function to scan and parse JSON values from the input string.
        object_hook (callable): A function that, if specified, will be called with the parsed object as a dictionary.
        object_pairs_hook (callable): A function that, if specified, will be called with the parsed object as a list of pairs.
        memo (dict, optional): A dictionary used to memoize string keys for optimization. Defaults to None.
        _w (function): A regular expression matching function for whitespace. Defaults to WHITESPACE.match.
        _ws (str): A string containing whitespace characters. Defaults to WHITESPACE_STR.

    Returns:
        tuple or dict: A tuple containing the parsed object and the index of the character in the input string
        after the end of the object.
    """

    s, end = s_and_end
    pairs = []
    pairs_append = pairs.append
    # Backwards compatibility
    if memo is None:
        memo = {}
    memo_get = memo.setdefault
    # Use a slice to prevent IndexError from being raised, the following
    # check will raise a more specific ValueError if the string is empty
    nextchar = s[end : end + 1]
    # Normally we expect nextchar == '"'
    if nextchar != '"' and nextchar != "'":
        if nextchar in _ws:
            end = _w(s, end).end()
            nextchar = s[end : end + 1]
        # Trivial empty object
        if nextchar == "}":
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                return result, end + 1
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return pairs, end + 1
        elif nextchar != '"':
            raise JSONDecodeError("Expecting property name enclosed in double quotes", s, end)
    end += 1
    while True:
        if end + 1 < len(s) and s[end] == nextchar and s[end + 1] == nextchar:
            # Handle the case where the next two characters are the same as nextchar
            key, end = scanstring(s, end + 2, strict, delimiter=nextchar * 3)
        else:
            # Handle the case where the next two characters are not the same as nextchar
            key, end = scanstring(s, end, strict, delimiter=nextchar)
        key = memo_get(key, key)
        # To skip some function call overhead we optimize the fast paths where
        # the JSON key separator is ": " or just ":".
        if s[end : end + 1] != ":":
            end = _w(s, end).end()
            if s[end : end + 1] != ":":
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        end += 1

        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass

        try:
            value, end = scan_once(s, end)
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        pairs_append((key, value))
        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _w(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ""
        end += 1

        if nextchar == "}":
            break
        elif nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        end = _w(s, end).end()
        nextchar = s[end : end + 1]
        end += 1
        if nextchar != '"':
            raise JSONDecodeError("Expecting property name enclosed in double quotes", s, end - 1)
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return result, end
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return pairs, end


```py

This appears to be a Python implementation of the `json.decode()` function from the `json` module. It takes a string `s` that contains a JSON string and an optional `end` index for the start of the expected end of the string, and returns a tuple of the decoded string and the end index.

The function first checks if the `s` string is already being consumed by the decoder, and if not, it attempts to parse a JSON string from the input string. If the decoder cannot find the expected end of the string, it raises an `JSONDecodeError`.

If the input string contains a valid JSON string, the function groups the string contents into a list and then decodes them using the `json.decode()` function. This function attempts to find the first unescaped string character (`uXXXX` format) or a literal escape sequence in the input string, and if one is found, returns the decoded string and the end index.

If the input string does not contain a valid JSON string, the function raises an `JSONDecodeError`.

The function also includes checks for certain issues that might cause problems with the input string, such as an empty string. If the input string is empty or contains only a single character, the function returns an empty string and the end index.


```
def py_scanstring(s, end, strict=True, _b=BACKSLASH, _m=STRINGCHUNK.match, delimiter='"'):
    """Scan the string s for a JSON string.

    Args:
        s (str): The input string to be scanned for a JSON string.
        end (int): The index of the character in `s` after the quote that started the JSON string.
        strict (bool): If `True`, enforces strict JSON string decoding rules.
            If `False`, allows literal control characters in the string. Defaults to `True`.
        _b (dict): A dictionary containing escape sequence mappings.
        _m (function): A regular expression matching function for string chunks.
        delimiter (str): The string delimiter used to define the start and end of the JSON string.
            Can be one of: '"', "'", '\"""', or "'''". Defaults to '"'.

    Returns:
        tuple: A tuple containing the decoded string and the index of the character in `s`
        after the end quote.
    """

    chunks = []
    _append = chunks.append
    begin = end - 1
    if delimiter == '"':
        _m = STRINGCHUNK.match
    elif delimiter == "'":
        _m = STRINGCHUNK_SINGLEQUOTE.match
    elif delimiter == '"""':
        _m = STRINGCHUNK_TRIPLE_DOUBLE_QUOTE.match
    else:
        _m = STRINGCHUNK_TRIPLE_SINGLEQUOTE.match
    while 1:
        chunk = _m(s, end)
        if chunk is None:
            raise JSONDecodeError("Unterminated string starting at", s, begin)
        end = chunk.end()
        content, terminator = chunk.groups()
        # Content is contains zero or more unescaped string characters
        if content:
            _append(content)
        # Terminator is the end of string, a literal control character,
        # or a backslash denoting that an escape sequence follows
        if terminator == delimiter:
            break
        elif terminator != "\\":
            if strict:
                # msg = "Invalid control character %r at" % (terminator,)
                msg = "Invalid control character {0!r} at".format(terminator)
                raise JSONDecodeError(msg, s, end)
            else:
                _append(terminator)
                continue
        try:
            esc = s[end]
        except IndexError:
            raise JSONDecodeError("Unterminated string starting at", s, begin) from None
        # If not a unicode escape sequence, must be in the lookup table
        if esc != "u":
            try:
                char = _b[esc]
            except KeyError:
                msg = "Invalid \\escape: {0!r}".format(esc)
                raise JSONDecodeError(msg, s, end)
            end += 1
        else:
            uni = _decode_uXXXX(s, end)
            end += 5
            if 0xD800 <= uni <= 0xDBFF and s[end : end + 2] == "\\u":
                uni2 = _decode_uXXXX(s, end + 1)
                if 0xDC00 <= uni2 <= 0xDFFF:
                    uni = 0x10000 + (((uni - 0xD800) << 10) | (uni2 - 0xDC00))
                    end += 6
            char = chr(uni)
        _append(char)
    return "".join(chunks), end


```py

这段代码定义了一个名为 CustomDecoder 的类，该类继承自 JSONDecoder 类，用于将 JSON 数据中的数据按照一定的规则进行解析。

在 CustomDecoder 的初始化函数中，传入了 6 个参数，包括 object_hook、parse_float、parse_int、parse_constant、strict 和 object_pairs_hook。这些参数都用于指定 JSONDecoder 类中的函数，用于指定 JSON 数据中的数据如何被解析。

CustomDecoder 的 `decode` 函数与 JSONDecoder 的 `decode` 函数具有相同的签名，但是它返回的结果是一个 Python 对象，而不是 JSON 数据。这个 Python 对象是由 `self.parse_object` 和 `self.parse_string` 函数计算出来的。其中，`self.parse_object` 函数将 JSON 数据中的数据解析为 Python 对象，而 `self.parse_string` 函数则是一个实现了 `py_scanstring` 函数的扫描器，用于将 Python 代码中的字符串转换为 JSON 数据。

因此，这段代码的作用是将一个 JSON 数据对象解析为 Python 对象，并实现了将 Python 代码中的字符串转换为 JSON 数据的功能。


```
scanstring = py_scanstring


class CustomDecoder(json.JSONDecoder):
    def __init__(
        self,
        *,
        object_hook=None,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        strict=True,
        object_pairs_hook=None
    ):
        super().__init__(
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )
        self.parse_object = JSONObject
        self.parse_string = py_scanstring
        self.scan_once = py_make_scanner(self)

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        return super().decode(s)

```py

# `metagpt/utils/file.py`

This is a simple class that provides file reading and writing functionality using the `aiofiles` library. The class includes two methods, `read` and `write`, which are implemented asynchronously using the `asyncio` framework.

The `write` method takes a file path, the binary content of the file, and a `chunk_size` parameter (default is 64KB). It creates the directory path if it does not exist, then writes the file's content to the specified path using an `asyncio` writer. The method returns the full filename of the file.

The `read` method takes a file path, reads the file's content in chunks of size `chunk_size`, and returns the binary content of the file. It opens the file in binary mode and reads it chunk by chunk. If the file is read successfully, it returns the content. If an error occurs, it logs the error and raises it.

The `read` method uses a `try`-`except` block to handle any exceptions that may occur during the file reading process. It also uses the `return` statement to return only the file's content, rather than the chunks. This is because the `read` method does not modify the original file, it only returns its content.


```
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : file.py
@Describe : General file operations.
"""
import aiofiles
from pathlib import Path

from metagpt.logs import logger


class File:
    """A general util for file operations."""

    CHUNK_SIZE = 64 * 1024

    @classmethod
    async def write(cls, root_path: Path, filename: str, content: bytes) -> Path:
        """Write the file content to the local specified path.

        Args:
            root_path: The root path of file, such as "/data".
            filename: The name of file, such as "test.txt".
            content: The binary content of file.

        Returns:
            The full filename of file, such as "/data/test.txt".

        Raises:
            Exception: If an unexpected error occurs during the file writing process.
        """
        try:
            root_path.mkdir(parents=True, exist_ok=True)
            full_path = root_path / filename
            async with aiofiles.open(full_path, mode="wb") as writer:
                await writer.write(content)
                logger.debug(f"Successfully write file: {full_path}")
                return full_path
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            raise e

    @classmethod
    async def read(cls, file_path: Path, chunk_size: int = None) -> bytes:
        """Partitioning read the file content from the local specified path.

        Args:
            file_path: The full file name of file, such as "/data/test.txt".
            chunk_size: The size of each chunk in bytes (default is 64kb).

        Returns:
            The binary content of file.

        Raises:
            Exception: If an unexpected error occurs during the file reading process.
        """
        try:
            chunk_size = chunk_size or cls.CHUNK_SIZE
            async with aiofiles.open(file_path, mode="rb") as reader:
                chunks = list()
                while True:
                    chunk = await reader.read(chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                content = b''.join(chunks)
                logger.debug(f"Successfully read file, the path of file: {file_path}")
                return content
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise e


```py

# `metagpt/utils/get_template.py`

这段代码定义了一个名为 `get_template` 的函数，它接受一个名为 `templates` 的字典，并从其中检索出一个指定的模板。如果模板不存在，则抛出一个 `ValueError`。

具体来说，函数首先从 `CONFIG` 类中获取一个名为 `prompt_format` 的配置参数，然后使用 `templates.get` 方法获取指定格式的模板。如果 `template` 参数为 `None`，则抛出一个 `ValueError`。如果模板存在，则返回模板和模板的格式示例。如果模板不存在，则抛出一个 `ValueError`。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/19 20:39
@Author  : femto Zheng
@File    : get_template.py
"""
from metagpt.config import CONFIG


def get_template(templates, format=CONFIG.prompt_format):
    selected_templates = templates.get(format)
    if selected_templates is None:
        raise ValueError(f"Can't find {format} in passed in templates")

    # Extract the selected templates
    prompt_template = selected_templates["PROMPT_TEMPLATE"]
    format_example = selected_templates["FORMAT_EXAMPLE"]

    return prompt_template, format_example

```py

# `metagpt/utils/highlight.py`

这段代码定义了一个名为 `highlight` 的函数，它接受一个代码字符串（`code`）和一个编程语言（`language`）作为参数，并返回一段代码片段高亮后的结果。

在函数内部，首先根据传入的语言选择相应的语法解析器（`PythonLexer` 和 `SqlLexer`）。然后，如果语言是 Python，就从 Python 解析器中选择一个语法格式化器（`TerminalFormatter` 和 `HtmlFormatter`）。如果语言是 SQL，则从 SQL 解析器中选择一个语法格式化器。如果语法格式化器选择不支持所选编程语言，函数将抛出一个 `ValueError`。

接下来，使用 Pygments 库中的 `highlight_` 函数提取要高亮的代码片段。 Pygments 是一个 Python 语法格式化库，提供了多种语法格式，包括简洁的语法格式和漂亮的语法格式。通过调用 `highlight_` 函数，可以将代码片段高亮为不同格式的格式，包括文本格式化和 HTML 格式化。最后，将生成的格式化后的代码返回。


```
# 添加代码语法高亮显示
from pygments import highlight as highlight_
from pygments.lexers import PythonLexer, SqlLexer
from pygments.formatters import TerminalFormatter, HtmlFormatter


def highlight(code: str, language: str = 'python', formatter: str = 'terminal'):
    # 指定要高亮的语言
    if language.lower() == 'python':
        lexer = PythonLexer()
    elif language.lower() == 'sql':
        lexer = SqlLexer()
    else:
        raise ValueError(f"Unsupported language: {language}")

    # 指定输出格式
    if formatter.lower() == 'terminal':
        formatter = TerminalFormatter()
    elif formatter.lower() == 'html':
        formatter = HtmlFormatter()
    else:
        raise ValueError(f"Unsupported formatter: {formatter}")

    # 使用 Pygments 高亮代码片段
    return highlight_(code, lexer, formatter)

```