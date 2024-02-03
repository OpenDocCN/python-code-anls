# `.\AutoGPT\autogpts\autogpt\autogpt\commands\web_selenium.py`

```py
# 导入必要的库和模块
from __future__ import annotations
import asyncio
import logging
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Optional, Type
from urllib.request import urlretrieve

# 导入第三方库
from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.options import ArgOptions as BrowserOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeDriverService
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as GeckoDriverService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.safari.webdriver import WebDriver as SafariDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager

# 导入自定义异常和装饰器
from autogpt.agents.utils.exceptions import CommandExecutionError, TooMuchOutputError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.processing.text import extract_information, summarize_text
# 导入必要的模块
from autogpt.url_utils.validators import validate_url

# 定义命令类别和标题
COMMAND_CATEGORY = "web_browse"
COMMAND_CATEGORY_TITLE = "Web Browsing"

# 如果类型检查开启，导入必要的模块
if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.config import Config

# 导入日志模块
logger = logging.getLogger(__name__)

# 获取当前文件的父目录
FILE_DIR = Path(__file__).parent.parent
# 定义最大原始内容长度
MAX_RAW_CONTENT_LENGTH = 500
# 定义要返回的链接数量
LINKS_TO_RETURN = 20

# 定义浏览错误类，用于处理浏览页面时出现的错误
class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""

# 定义命令装饰器，用于读取网页内容
@command(
    "read_webpage",
    (
        "Read a webpage, and extract specific information from it."
        " You must specify either topics_of_interest, a question, or get_raw_content."
    ),
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The URL to visit",
            required=True,
        ),
        "topics_of_interest": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(type=JSONSchema.Type.STRING),
            description=(
                "A list of topics about which you want to extract information "
                "from the page."
            ),
            required=False,
        ),
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "A question that you want to answer using the content of the webpage."
            ),
            required=False,
        ),
        "get_raw_content": JSONSchema(
            type=JSONSchema.Type.BOOLEAN,
            description=(
                "If true, the unprocessed content of the webpage will be returned. "
                "This consumes a lot of tokens, so use it with caution."
            ),
            required=False,
        ),
    },
)
# 验证 URL 格式
@validate_url
# 异步函数，读取网页内容
async def read_webpage(
    url: str,
    agent: Agent,
    *,
    topics_of_interest: list[str] = [],
    get_raw_content: bool = False,
    question: str = "",
) -> str:
    """Browse a website and return the answer and links to the user
    Args:
        url (str): 要浏览的网站的 URL
        question (str): 使用网页内容回答的问题

    Returns:
        str: 答案和链接给用户和 webdriver
    """
    # 初始化 driver 变量
    driver = None
    try:
        # 在浏览器中打开网页
        driver = await open_page_in_browser(url, agent.legacy_config)

        # 使用 Selenium 抓取文本内容
        text = scrape_text_with_selenium(driver)
        # 使用 Selenium 抓取链接
        links = scrape_links_with_selenium(driver, url)

        # 初始化返回纯文本内容和是否已总结的标志
        return_literal_content = True
        summarized = False
        # 如果没有文本内容，则返回包含链接的消息
        if not text:
            return f"Website did not contain any text.\n\nLinks: {links}"
        # 如果需要原始内容
        elif get_raw_content:
            # 检查原始内容长度是否超过限制
            if (
                output_tokens := agent.llm_provider.count_tokens(text, agent.llm.name)
            ) > MAX_RAW_CONTENT_LENGTH:
                oversize_factor = round(output_tokens / MAX_RAW_CONTENT_LENGTH, 1)
                raise TooMuchOutputError(
                    f"Page content is {oversize_factor}x the allowed length "
                    "for `get_raw_content=true`"
                )
            return text + (f"\n\nLinks: {links}" if links else "")
        else:
            # 对网页内容进行总结
            text = await summarize_memorize_webpage(
                url, text, question or None, topics_of_interest, agent, driver
            )
            return_literal_content = bool(question)
            summarized = True

        # 限制链接数量为 LINKS_TO_RETURN
        if len(links) > LINKS_TO_RETURN:
            links = links[:LINKS_TO_RETURN]

        # 格式化文本内容和链接
        text_fmt = f"'''{text}'''" if "\n" in text else f"'{text}'"
        links_fmt = "\n".join(f"- {link}" for link in links)
        # 返回包含文本内容和链接的消息
        return (
            f"Page content{' (summary)' if summarized else ''}:"
            if return_literal_content
            else "Answer gathered from webpage:"
        ) + f" {text_fmt}\n\nLinks:\n{links_fmt}"
    # 捕获 WebDriverException 异常
    except WebDriverException as e:
        # 这些错误通常很长，包含很多上下文信息
        # 只获取第一行错误信息
        msg = e.msg.split("\n")[0]
        # 如果错误信息中包含"net::"
        if "net::" in msg:
            # 抛出 BrowsingError 异常，提示网络错误
            raise BrowsingError(
                "A networking error occurred while trying to load the page: %s"
                % re.sub(r"^unknown error: ", "", msg)
            )
        # 抛出 CommandExecutionError 异常
        raise CommandExecutionError(msg)
    finally:
        # 最终执行，确保关闭浏览器
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
    # 获取页面源代码
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    # 从 BeautifulSoup 对象中移除 script 和 style 标签
    for script in soup(["script", "style"]):
        script.extract()

    # 提取超链接
    hyperlinks = extract_hyperlinks(soup, base_url)

    return format_hyperlinks(hyperlinks)


async def open_page_in_browser(url: str, config: Config) -> WebDriver:
    """Open a browser window and load a web page using Selenium

    Params:
        url (str): The URL of the page to load
        config (Config): The applicable application configuration

    Returns:
        driver (WebDriver): A driver object representing the browser window to scrape
    """
    # 设置 Selenium 日志级别为 CRITICAL
    logging.getLogger("selenium").setLevel(logging.CRITICAL)
    # 定义一个字典，将浏览器选项名称映射到对应的浏览器选项类
    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    # 根据配置文件中的浏览器选项，选择对应的浏览器选项类实例化对象
    options: BrowserOptions = options_available[config.selenium_web_browser]()
    # 添加用户代理到浏览器选项中
    options.add_argument(f"user-agent={config.user_agent}")

    # 根据不同的浏览器选项类进行不同的处理
    if isinstance(options, FirefoxOptions):
        # 如果配置为无头模式，设置浏览器选项为无头模式，并添加参数
        if config.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        # 实例化 Firefox 浏览器驱动对象
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif isinstance(options, EdgeOptions):
        # 实例化 Edge 浏览器驱动对象
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif isinstance(options, SafariOptions):
        # 需要用户端进行更多的设置
        # 参考链接：https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = SafariDriver(options=options)
    elif isinstance(options, ChromeOptions):
        # 如果运行在 Linux 系统上，添加特定参数
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        # 如果配置为无头模式，设置浏览器选项为无头模式，并添加参数
        if config.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        # 加载 Chrome 扩展程序
        _sideload_chrome_extensions(options, config.app_data_dir / "assets" / "crx")

        # 设置 Chromium 驱动程序路径
        chromium_driver_path = Path("/usr/bin/chromedriver")

        # 实例化 Chrome 浏览器驱动对象
        driver = ChromeDriver(
            service=ChromeDriverService(str(chromium_driver_path))
            if chromium_driver_path.exists()
            else ChromeDriverService(ChromeDriverManager().install()),
            options=options,
        )
    # 打开指定 URL
    driver.get(url)

    # 等待页面加载完成，休眠 2 秒，再次等待页面加载完成
    # 等待页面加载完成，直到body标签出现，给予cookiewall squasher时间去处理cookie墙
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    # 异步等待2秒
    await asyncio.sleep(2)
    # 再次等待页面加载完成，直到body标签出现
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # 返回driver对象
    return driver
# 从 Chrome 网上应用商店下载 Chrome 扩展并加载到 ChromeOptions 中
def _sideload_chrome_extensions(options: ChromeOptions, dl_folder: Path) -> None:
    # Chrome 扩展下载链接模板
    crx_download_url_template = "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=49.0&acceptformat=crx3&x=id%3D{crx_id}%26installsource%3Dondemand%26uc"  # noqa
    # CookieWall Squasher 扩展的 ID
    cookiewall_squasher_crx_id = "edibdbjcniadpccecjdfdjjppcpchdlm"
    # 广告拦截器扩展的 ID
    adblocker_crx_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"

    # 确保目标文件夹存在
    dl_folder.mkdir(parents=True, exist_ok=True)

    # 遍历需要下载的扩展 ID
    for crx_id in (cookiewall_squasher_crx_id, adblocker_crx_id):
        # 构建扩展文件路径
        crx_path = dl_folder / f"{crx_id}.crx"
        # 如果文件不存在，则下载并添加到 ChromeOptions 中
        if not crx_path.exists():
            logger.debug(f"Downloading CRX {crx_id}...")
            crx_download_url = crx_download_url_template.format(crx_id=crx_id)
            urlretrieve(crx_download_url, crx_path)
            logger.debug(f"Downloaded {crx_path.name}")
        options.add_extension(str(crx_path))


# 关闭浏览器
def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


# 使用 OpenAI API 对文本进行摘要
async def summarize_memorize_webpage(
    url: str,
    text: str,
    question: str | None,
    topics_of_interest: list[str],
    agent: Agent,
    driver: Optional[WebDriver] = None,
) -> str:
    """Summarize text using the OpenAI API

    Args:
        url (str): The url of the text
        text (str): The text to summarize
        question (str): The question to ask the model
        driver (WebDriver): The webdriver to use to scroll the page

    Returns:
        str: The summary of the text
    """
    # 如果没有文本，则抛出异常
    if not text:
        raise ValueError("No text to summarize")

    # 计算文本长度
    text_length = len(text)
    logger.debug(f"Web page content length: {text_length} characters")

    # 获取记忆内容
    # memory = get_memory(agent.legacy_config)

    # 创建新的记忆项，从网页内容中提取
    # new_memory = MemoryItem.from_webpage(
    #     content=text,
    #     url=url,
    #     config=agent.legacy_config,
    #     question=question,
    # 如果存在感兴趣的主题
    if topics_of_interest:
        # 调用提取信息的异步函数，传入文本、感兴趣的主题、LLM 提供者和配置信息
        information = await extract_information(
            text,
            topics_of_interest=topics_of_interest,
            llm_provider=agent.llm_provider,
            config=agent.legacy_config,
        )
        # 返回信息列表中每个元素前加上"* "后的字符串拼接结果
        return "\n".join(f"* {i}" for i in information)
    else:
        # 调用摘要文本的异步函数，传入文本、问题、LLM 提供者和配置信息
        result, _ = await summarize_text(
            text,
            question=question,
            llm_provider=agent.llm_provider,
            config=agent.legacy_config,
        )
        # 返回摘要结果
        return result
```