# `.\yolov8\ultralytics\hub\utils.py`

```py
# 导入所需的库
import os
import platform
import random
import threading
import time
from pathlib import Path

# 导入第三方库 requests
import requests

# 导入 ultralytics.utils 下的多个模块和函数
from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    IS_COLAB,
    IS_GIT_DIR,
    IS_PIP_PACKAGE,
    LOGGER,
    ONLINE,
    RANK,
    SETTINGS,
    TESTS_RUNNING,
    TQDM,
    TryExcept,
    __version__,
    colorstr,
    get_git_origin_url,
)
# 导入 ultralytics.utils.downloads 模块中的 GITHUB_ASSETS_NAMES
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

# 设置 HUB_API_ROOT 和 HUB_WEB_ROOT 变量，若环境变量 ULTRALYTICS_HUB_API 或 ULTRALYTICS_HUB_WEB 未定义，则使用默认值
HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")

# 使用 colorstr 函数创建 PREFIX 变量，用于打印带颜色的文本前缀
PREFIX = colorstr("Ultralytics HUB: ")
# 定义帮助信息字符串
HELP_MSG = "If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance."


def request_with_credentials(url: str) -> any:
    """
    在 Google Colab 环境中发送带有附加 cookies 的 AJAX 请求。

    Args:
        url (str): 要发送请求的 URL。

    Returns:
        (any): AJAX 请求的响应数据。

    Raises:
        OSError: 如果函数不在 Google Colab 环境中运行。
    """
    # 如果不在 Colab 环境中，则抛出 OSError 异常
    if not IS_COLAB:
        raise OSError("request_with_credentials() must run in a Colab environment")
    
    # 导入必要的 Colab 相关库
    from google.colab import output  # noqa
    from IPython import display  # noqa

    # 使用 display.Javascript 创建一个 AJAX 请求，并附加 cookies
    display.display(
        display.Javascript(
            """
            window._hub_tmp = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("%s", {
                    method: 'POST',
                    credentials: 'include'
                })
                    .then((response) => resolve(response.json()))
                    .then((json) => {
                    clearTimeout(timeout);
                    }).catch((err) => {
                    clearTimeout(timeout);
                    reject(err);
                });
            });
            """
            % url
        )
    )
    # 返回输出的结果
    return output.eval_js("_hub_tmp")


def requests_with_progress(method, url, **kwargs):
    """
    使用指定的方法和 URL 发送 HTTP 请求，支持可选的进度条显示。

    Args:
        method (str): 要使用的 HTTP 方法 (例如 'GET'、'POST')。
        url (str): 要发送请求的 URL。
        **kwargs (any): 传递给底层 `requests.request` 函数的其他关键字参数。

    Returns:
        (requests.Response): HTTP 请求的响应对象。

    Note:
        - 如果 'progress' 设置为 True，则进度条将显示已知内容长度的下载进度。
        - 如果 'progress' 是一个数字，则进度条将显示假设内容长度为 'progress' 的下载进度。
    """
    # 弹出 kwargs 中的 progress 参数，默认为 False
    progress = kwargs.pop("progress", False)
    # 如果 progress 为 False，则直接发送请求
    if not progress:
        return requests.request(method, url, **kwargs)
    # 发起 HTTP 请求并获取响应
    response = requests.request(method, url, stream=True, **kwargs)
    # 从响应头中获取内容长度信息，如果 progress 参数是布尔值则返回内容长度，否则返回 progress 参数的值作为总大小
    total = int(response.headers.get("content-length", 0) if isinstance(progress, bool) else progress)  # total size
    try:
        # 初始化进度条对象，显示总大小并按照适当的单位进行缩放
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        # 逐块迭代响应数据流，每次更新进度条
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        # 关闭进度条
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:  # 避免出现 'Connection broken: IncompleteRead' 的警告
        # 关闭响应以处理异常
        response.close()
    # 返回完整的 HTTP 响应对象
    return response
    """
    Makes an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        method (str): The HTTP method to use for the request. Choices are 'post' and 'get'.
        url (str): The URL to make the request to.
        retry (int, optional): Number of retries to attempt before giving up. Default is 3.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying. Default is 30.
        thread (bool, optional): Whether to execute the request in a separate daemon thread. Default is True.
        code (int, optional): An identifier for the request, used for logging purposes. Default is -1.
        verbose (bool, optional): A flag to determine whether to print out to console or not. Default is True.
        progress (bool, optional): Whether to show a progress bar during the request. Default is False.
        **kwargs (any): Keyword arguments to be passed to the requests function specified in method.

    Returns:
        (requests.Response): The HTTP response object. If the request is executed in a separate thread, returns None.
    """
    retry_codes = (408, 500)  # retry only these codes

    # Decorator to handle exceptions and log messages
    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        """Make HTTP requests with retries and timeouts, with optional progress tracking."""
        r = None  # response object
        t0 = time.time()  # start time for timeout
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            # Perform HTTP request with progress tracking if enabled
            r = requests_with_progress(func_method, func_url, **func_kwargs)
            # Check if response status code indicates success
            if r.status_code < 300:
                break
            try:
                m = r.json().get("message", "No JSON message.")
            except AttributeError:
                m = "Unable to read JSON."
            # Handle retry logic based on response status code
            if i == 0:
                if r.status_code in retry_codes:
                    m += f" Retrying {retry}x for {timeout}s." if retry else ""
                elif r.status_code == 429:  # rate limit exceeded
                    h = r.headers  # response headers
                    m = (
                        f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). "
                        f"Please retry after {h['Retry-After']}s."
                    )
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                # Return response if no need to retry
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2**i)  # exponential backoff wait
        return r

    # Prepare arguments and pass progress flag to kwargs
    args = method, url
    kwargs["progress"] = progress
    # 如果 thread 参数为真，则创建一个新线程并启动，运行 func 函数
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    # 如果 thread 参数为假，则直接调用 func 函数并返回其结果
    else:
        return func(*args, **kwargs)
class Events:
    """
    A class for collecting anonymous event analytics. Event analytics are enabled when sync=True in settings and
    disabled when sync=False. Run 'yolo settings' to see and update settings YAML file.

    Attributes:
        url (str): The URL to send anonymous events.
        rate_limit (float): The rate limit in seconds for sending events.
        metadata (dict): A dictionary containing metadata about the environment.
        enabled (bool): A flag to enable or disable Events based on certain conditions.
    """

    # 设置 Google Analytics 收集匿名事件的 URL
    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self):
        """Initializes the Events object with default values for events, rate_limit, and metadata."""
        # 初始化事件列表
        self.events = []  # events list
        # 设置事件发送的速率限制（单位：秒）
        self.rate_limit = 30.0  # rate limit (seconds)
        # 初始化事件发送的计时器（单位：秒）
        self.t = 0.0  # rate limit timer (seconds)
        # 设置环境的元数据
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",  # 检查命令行是否为 'yolo'
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",  # 检查安装方式是 git 还是 pip 或其他
            "python": ".".join(platform.python_version_tuple()[:2]),  # Python 版本号，例如 3.10
            "version": __version__,  # 从模块中获取版本号
            "env": ENVIRONMENT,  # 获取环境变量
            "session_id": round(random.random() * 1e15),  # 创建随机会话 ID
            "engagement_time_msec": 1000,  # 设置参与时间（毫秒）
        }
        # 根据设置和其他条件，确定是否启用事件收集
        self.enabled = (
            SETTINGS["sync"]  # 检查是否设置为同步
            and RANK in {-1, 0}  # 检查当前排名是否为 -1 或 0
            and not TESTS_RUNNING  # 确保没有正在运行的测试
            and ONLINE  # 确保在线状态
            and (IS_PIP_PACKAGE or get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git")  # 检查安装来源是否为指定的 GitHub 仓库
        )
    # 定义一个特殊方法 __call__()，使实例可以像函数一样被调用
    def __call__(self, cfg):
        """
        Attempts to add a new event to the events list and send events if the rate limit is reached.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
        """
        # 如果事件功能未启用，直接返回，不执行任何操作
        if not self.enabled:
            # Events disabled, do nothing
            return

        # 尝试添加事件到事件列表
        if len(self.events) < 25:  # 事件列表最多包含 25 个事件，超过部分将被丢弃
            # 构建事件参数字典，包括元数据和配置的任务和模型信息
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
            }
            # 如果配置模式为 "export"，则添加格式信息到参数字典中
            if cfg.mode == "export":
                params["format"] = cfg.format
            # 将新事件以字典形式添加到事件列表中
            self.events.append({"name": cfg.mode, "params": params})

        # 检查发送速率限制
        t = time.time()
        if (t - self.t) < self.rate_limit:
            # 如果发送时间间隔未超过限制，等待发送
            return

        # 如果时间间隔超过限制，立即发送事件数据
        data = {"client_id": SETTINGS["uuid"], "events": self.events}  # 使用 SHA-256 匿名化的 UUID 哈希和事件列表

        # 发送 POST 请求，相当于 requests.post(self.url, json=data)，不进行重试和输出详细信息
        smart_request("post", self.url, json=data, retry=0, verbose=False)

        # 重置事件列表和发送时间计时器
        self.events = []
        self.t = t
# 在 hub/utils 初始化中运行以下代码
events = Events()
```