# `.\AutoGPT\autogpts\forge\forge\actions\web\web_selenium.py`

```py
# 导入所需的模块和库
"""Commands for browsing a website"""

# 导入未来版本的注解特性
from __future__ import annotations

# 定义命令类别和标题
COMMAND_CATEGORY = "web_browse"
COMMAND_CATEGORY_TITLE = "Web Browsing"

# 导入必要的模块和库
import functools
import logging
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Type
from urllib.parse import urljoin, urlparse

# 导入 BeautifulSoup 模块
from bs4 import BeautifulSoup
# 导入 requests.compat 模块中的 urljoin 函数
from requests.compat import urljoin
# 导入 Selenium 异常模块
from selenium.common.exceptions import WebDriverException
# 导入 Chrome 浏览器选项模块
from selenium.webdriver.chrome.options import Options as ChromeOptions
# 导入 Chrome 驱动服务模块
from selenium.webdriver.chrome.service import Service as ChromeDriverService
# 导入 Chrome 驱动模块
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
# 导入 Selenium 定位元素模块
from selenium.webdriver.common.by import By
# 导入 Selenium 浏览器选项模块
from selenium.webdriver.common.options import ArgOptions as BrowserOptions
# 导入 Edge 浏览器选项模块
from selenium.webdriver.edge.options import Options as EdgeOptions
# 导入 Edge 驱动服务模块
from selenium.webdriver.edge.service import Service as EdgeDriverService
# 导入 Edge 驱动模块
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
# 导入 Firefox 浏览器选项模块
from selenium.webdriver.firefox.options import Options as FirefoxOptions
# 导入 Gecko 驱动服务模块
from selenium.webdriver.firefox.service import Service as GeckoDriverService
# 导入 Firefox 驱动模块
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
# 导入 Selenium 远程驱动模块
from selenium.webdriver.remote.webdriver import WebDriver
# 导入 Safari 浏览器选项模块
from selenium.webdriver.safari.options import Options as SafariOptions
# 导入 Safari 驱动模块
from selenium.webdriver.safari.webdriver import WebDriver as SafariDriver
# 导入 Selenium 条件模块
from selenium.webdriver.support import expected_conditions as EC
# 导入 Selenium 等待模块
from selenium.webdriver.support.wait import WebDriverWait
# 导入 Chrome 驱动管理模块
from webdriver_manager.chrome import ChromeDriverManager
# 导入 Firefox 驱动管理模块
from webdriver_manager.firefox import GeckoDriverManager
# 导入 Edge 驱动管理模块
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager

# 导入自定义错误模块
from forge.sdk.errors import CommandExecutionError

# 导入动作注册模块
from ..registry import action

# 定义函数，从 BeautifulSoup 对象中提取超链接
def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    # 从 BeautifulSoup 对象中提取超链接
    # 参数:
    #   - soup (BeautifulSoup): BeautifulSoup 对象
    #   - base_url (str): 基础 URL
    # 返回值:
    #   - List[Tuple[str, str]]: 提取的超链接列表，每个元素是一个元组，包含文本和完整链接
    
    return [
        # 遍历 BeautifulSoup 对象中所有包含 href 属性的 a 标签
        # 提取文本和链接，使用 urljoin 函数将相对链接转换为绝对链接
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]
# 格式化超链接以便显示给用户
def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]


# 验证 URL 的装饰器，用于验证任何需要 URL 作为参数的命令
def validate_url(func: Callable[..., Any]) -> Any:
    """The method decorator validate_url is used to validate urls for any command that requires
    a url as an argument"""

    @functools.wraps(func)
    def wrapper(url: str, *args, **kwargs) -> Any:
        """Check if the URL is valid using a basic check, urllib check, and local file check

        Args:
            url (str): The URL to check

        Returns:
            the result of the wrapped function

        Raises:
            ValueError if the url fails any of the validation tests
        """
        # 最基本的检查 URL 是否有效：
        if not re.match(r"^https?://", url):
            raise ValueError("Invalid URL format")
        if not is_valid_url(url):
            raise ValueError("Missing Scheme or Network location")
        # 限制对本地文件的访问
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")
        # 检查 URL 长度
        if len(url) > 2000:
            raise ValueError("URL is too long")

        return func(sanitize_url(url), *args, **kwargs)

    return wrapper


# 检查 URL 是否有效
def is_valid_url(url: str) -> bool:
    """Check if the URL is valid

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# 清理 URL
def sanitize_url(url: str) -> str:
    """Sanitize the URL

    Args:
        url (str): The URL to sanitize

    Returns:
        str: The sanitized URL
    """
    # 解析给定的 URL，将其拆分成各个部分
    parsed_url = urlparse(url)
    # 重新构建 URL，包括路径、参数和查询部分
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    # 将重新构建的 URL 与原始 URL 进行合并，返回最终的 URL
    return urljoin(url, reconstructed_url)
# 检查 URL 是否为本地文件
def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    # 定义本地文件的前缀列表
    local_prefixes = [
        "file:///",
        "file://localhost/",
        "file://localhost",
        "http://localhost",
        "http://localhost/",
        "https://localhost",
        "https://localhost/",
        "http://2130706433",
        "http://2130706433/",
        "https://2130706433",
        "https://2130706433/",
        "http://127.0.0.1/",
        "http://127.0.0.1",
        "https://127.0.0.1/",
        "https://127.0.0.1",
        "https://0.0.0.0/",
        "https://0.0.0.0",
        "http://0.0.0.0/",
        "http://0.0.0.0",
        "http://0000",
        "http://0000/",
        "https://0000",
        "https://0000/",
    ]
    # 检查 URL 是否以本地文件前缀开头
    return any(url.startswith(prefix) for prefix in local_prefixes)


# 获取日志记录器
logger = logging.getLogger(__name__)

# 获取当前文件的父目录
FILE_DIR = Path(__file__).parent.parent
# 触发摘要的令牌数
TOKENS_TO_TRIGGER_SUMMARY = 50
# 返回的链接数
LINKS_TO_RETURN = 20


# 定义一个自定义异常类，表示在浏览页面时发生的错误
class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


# 定义一个动作装饰器，用于读取网页内容并提取特定信息
@action(
    name="read_webpage",
    description="Read a webpage, and extract specific information from it if a question is specified. If you are looking to extract specific information from the webpage, you should specify a question.",
    parameters=[
        {
            "name": "url",
            "description": "The URL to visit",
            "type": "string",
            "required": True,
        },
        {
            "name": "question",
            "description": "A question that you want to answer using the content of the webpage.",
            "type": "string",
            "required": False,
        },
    ],
    output_type="string",
)
# 对 URL 进行验证
@validate_url
# 异步函数，读取网页内容并返回结果
async def read_webpage(
    agent, task_id: str, url: str, question: str = ""
) -> Tuple(str, List[str]):
    """浏览网站并返回用户的答案和链接

    Args:
        url (str): 要浏览的网站的 URL
        question (str): 使用网页内容回答的问题

    Returns:
        str: 用户和 webdriver 的答案和链接
    """
    # 初始化 driver 变量
    driver = None
    try:
        # 在浏览器中打开页面
        driver = open_page_in_browser(url)

        # 使用 Selenium 抓取文本内容
        text = scrape_text_with_selenium(driver)
        # 使用 Selenium 抓取链接
        links = scrape_links_with_selenium(driver, url)

        # 如果没有文本内容，则返回链接
        if not text:
            return f"Website did not contain any text.\n\nLinks: {links}"

        # 限制链接数量为 LINKS_TO_RETURN
        if len(links) > LINKS_TO_RETURN:
            links = links[:LINKS_TO_RETURN]
        return (text, links)

    except WebDriverException as e:
        # 这些错误通常很长，包含很多上下文信息。只获取第一行。
        msg = e.msg.split("\n")[0]
        if "net::" in msg:
            raise BrowsingError(
                f"A networking error occurred while trying to load the page: "
                + re.sub(r"^unknown error: ", "", msg)
            )
        raise CommandExecutionError(msg)
    finally:
        # 如果 driver 存在，则关闭浏览器
        if driver:
            close_browser(driver)
def scrape_text_with_selenium(driver: WebDriver) -> str:
    """Scrape text from a browser window using selenium

    Args:
        driver (WebDriver): A driver object representing the browser window to scrape

    Returns:
        str: the text scraped from the website
    """

    # 从浏览器的 DOM 直接获取 HTML 内容
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    # 从 BeautifulSoup 对象中移除 script 和 style 标签
    for script in soup(["script", "style"]):
        script.extract()

    # 获取纯文本内容
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


def scrape_links_with_selenium(driver: WebDriver, base_url: str) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): A driver object representing the browser window to scrape
        base_url (str): The base URL to use for resolving relative links

    Returns:
        List[str]: The links scraped from the website
    """
    # 获取页面源码
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    # 从 BeautifulSoup 对象中移除 script 和 style 标签
    for script in soup(["script", "style"]):
        script.extract()

    # 提取超链接
    hyperlinks = extract_hyperlinks(soup, base_url)

    return format_hyperlinks(hyperlinks)


def open_page_in_browser(url: str) -> WebDriver:
    """Open a browser window and load a web page using Selenium

    Params:
        url (str): The URL of the page to load

    Returns:
        driver (WebDriver): A driver object representing the browser window to scrape
    """
    # 设置 Selenium 日志级别
    logging.getLogger("selenium").setLevel(logging.CRITICAL)
    selenium_web_browser = "chrome"
    selenium_headless = True
    # 定义一个字典，将浏览器选项名称映射到对应的浏览器选项类
    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    # 根据传入的 selenium_web_browser 参数选择对应的浏览器选项类，并创建选项对象
    options: BrowserOptions = options_available[selenium_web_browser]()
    # 添加自定义的用户代理信息到选项中
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    # 根据不同的浏览器类型进行不同的处理
    if selenium_web_browser == "firefox":
        # 如果是 Firefox 浏览器，根据是否 headless 设置选项
        if selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        # 创建 Firefox 驱动对象
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif selenium_web_browser == "edge":
        # 创建 Edge 驱动对象
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif selenium_web_browser == "safari":
        # 需要在用户端进行更多的设置
        # 参考链接：https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = SafariDriver(options=options)
    else:
        # 对于其他浏览器类型，根据平台设置一些选项
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        # 设置 Chromium 驱动路径
        chromium_driver_path = Path("/usr/bin/chromedriver")

        # 创建 Chrome 驱动对象
        driver = ChromeDriver(
            service=ChromeDriverService(str(chromium_driver_path))
            if chromium_driver_path.exists()
            else ChromeDriverService(ChromeDriverManager().install()),
            options=options,
        )
    # 打开指定的 URL
    driver.get(url)

    # 等待页面加载完成
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # 返回驱动对象
    return driver
# 关闭浏览器的函数
def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    # 退出浏览器驱动
    driver.quit()
```