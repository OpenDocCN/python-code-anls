# `.\DB-GPT-src\dbgpt\util\api_utils.py`

```py
# 引入异步I/O模块
import asyncio
# 引入日志记录模块
import logging
# 引入线程模块
import threading
# 引入时间模块
import time
# 引入抽象基类模块
from abc import ABC
# 引入并发执行器模块
from concurrent.futures import Executor, ThreadPoolExecutor
# 引入数据类模块中的asdict和is_dataclass函数
from dataclasses import asdict, is_dataclass
# 引入日期时间模块中的datetime和timedelta类
from datetime import datetime, timedelta
# 引入函数签名检查模块中的signature函数
from inspect import signature
# 引入类型提示模块中的List, Literal, Optional, Tuple, Type, TypeVar, Union和get_type_hints函数
from typing import List, Literal, Optional, Tuple, Type, TypeVar, Union, get_type_hints

# 定义类型变量T
T = TypeVar("T")

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


# 定义一个API混合类，继承自抽象基类ABC
class APIMixin(ABC):
    """API mixin class."""

    # 初始化方法
    def __init__(
        self,
        urls: Union[str, List[str]],  # 接收一个字符串或字符串列表作为URL参数
        health_check_path: str,  # 健康检查路径字符串参数
        health_check_interval_secs: int = 5,  # 健康检查间隔时间，默认5秒
        health_check_timeout_secs: int = 30,  # 健康检查超时时间，默认30秒
        check_health: bool = True,  # 是否进行健康检查，默认为True
        choice_type: Literal["latest_first", "random"] = "latest_first",  # 选择类型，字面值类型，可选latest_first或random，默认latest_first
        executor: Optional[Executor] = None,  # 并发执行器对象，可选参数
    ):
        # 如果urls是字符串，则按逗号分隔成列表
        if isinstance(urls, str):
            urls = urls.split(",")
        # 去除每个URL前后的空格，并存储在_remote_urls属性中
        urls = [url.strip() for url in urls]
        self._remote_urls = urls  # 存储远程URL列表
        self._health_check_path = health_check_path  # 存储健康检查路径
        self._health_urls = []  # 初始化健康URL列表为空
        self._health_check_interval_secs = health_check_interval_secs  # 存储健康检查间隔时间
        self._health_check_timeout_secs = health_check_timeout_secs  # 存储健康检查超时时间
        self._heartbeat_map = {}  # 初始化心跳映射字典为空
        # 存储选择类型（latest_first或random）
        self._choice_type = choice_type
        # 创建一个线程用于执行心跳检查方法
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_checker)
        # 如果未提供executor，则创建一个最大工作线程数为3的线程池执行器
        self._heartbeat_executor = executor or ThreadPoolExecutor(max_workers=3)
        self._heartbeat_stop_event = threading.Event()  # 初始化一个线程事件对象

        # 如果需要进行健康检查，则将心跳检查线程设置为守护线程并启动
        if check_health:
            self._heartbeat_thread.daemon = True
            self._heartbeat_thread.start()

    # 心跳检查方法
    def _heartbeat_checker(self):
        logger.debug("Running health check")  # 记录调试日志：运行健康检查
        # 当线程事件未设置时循环执行健康检查
        while not self._heartbeat_stop_event.is_set():
            try:
                # 调用检查和更新健康URL的私有方法，并获取健康的URL列表
                healthy_urls = self._check_and_update_health()
                logger.debug(f"Healthy urls: {healthy_urls}")  # 记录调试日志：健康的URL列表
            except Exception as e:
                logger.warning(f"Health check failed, error: {e}")  # 记录警告日志：健康检查失败，记录错误信息
            time.sleep(self._health_check_interval_secs)  # 线程休眠，等待下一次健康检查间隔时间

    # 对象销毁时调用的方法
    def __del__(self):
        self._heartbeat_stop_event.set()  # 设置线程事件，通知心跳检查线程停止

    # 检查单个URL的健康状态的私有方法
    def _check_health(self, url: str) -> Tuple[bool, str]:
        try:
            import requests  # 导入requests模块

            logger.debug(f"Checking health for {url}")  # 记录调试日志：检查URL的健康状态
            req_url = url + self._health_check_path  # 构建完整的健康检查URL
            # 发送GET请求，设置超时时间为10秒，获取响应对象
            response = requests.get(req_url, timeout=10)
            # 返回是否健康和URL
            return response.status_code == 200, url
        except Exception as e:
            logger.warning(f"Health check failed for {url}, error: {e}")  # 记录警告日志：健康检查失败，记录错误信息
            # 返回健康状态为False和URL
            return False, url
    # 检查所有远程 URL 的健康状态并更新健康 URL 列表
    def _check_and_update_health(self) -> List[str]:
        """Check health of all remote urls and update the health urls list."""
        # 创建异步任务列表，用于检查每个 URL 的健康状态
        check_tasks = []
        # 存储检查结果的列表
        check_results = []
        # 遍历所有远程 URL
        for url in self._remote_urls:
            # 提交健康检查任务到线程池中，并将返回的 Future 对象添加到任务列表中
            check_tasks.append(self._heartbeat_executor.submit(self._check_health, url))
        # 遍历所有任务，获取每个任务的结果并存储到结果列表中
        for task in check_tasks:
            check_results.append(task.result())
        # 获取当前时间
        now = datetime.now()
        # 更新健康 URL 映射表，记录最新的心跳时间
        for is_healthy, url in check_results:
            if is_healthy:
                self._heartbeat_map[url] = now
        # 从健康 URL 映射表中筛选出在健康检查间隔内仍然健康的 URL 列表
        healthy_urls = []
        for url, last_heartbeat in self._heartbeat_map.items():
            if now - last_heartbeat < timedelta(
                seconds=self._health_check_interval_secs
            ):
                healthy_urls.append((url, last_heartbeat))
        # 按照最后心跳时间排序健康 URL 列表，最新的排在最前面
        healthy_urls.sort(key=lambda x: x[1], reverse=True)

        # 更新实例变量 _health_urls，只保留 URL，不需要心跳时间
        self._health_urls = [url for url, _ in healthy_urls]
        # 返回当前的健康 URL 列表
        return self._health_urls

    # 异步方法：选择一个健康的 URL 发送请求
    async def select_url(self, max_wait_health_timeout_secs: int = 2) -> str:
        """Select a healthy url to send request.

        If no healthy urls found, select randomly.
        """
        import random

        # 内部函数：根据选择类型选择 URL
        def _select(urls: List[str]):
            if self._choice_type == "latest_first":
                return urls[0]
            elif self._choice_type == "random":
                return random.choice(urls)
            else:
                raise ValueError(f"Invalid choice type: {self._choice_type}")

        # 如果存在健康的 URL，则根据选择类型返回一个 URL
        if self._health_urls:
            return _select(self._health_urls)
        # 如果等待超时时间大于 0，则在超时时间内等待健康 URL 出现
        elif max_wait_health_timeout_secs > 0:
            start_time = datetime.now()
            while datetime.now() - start_time < timedelta(
                seconds=max_wait_health_timeout_secs
            ):
                if self._health_urls:
                    return _select(self._health_urls)
                await asyncio.sleep(0.1)
        # 如果未找到健康的 URL，记录警告并随机选择一个远程 URL 返回
        logger.warning("No healthy urls found, selecting randomly")
        return _select(self._remote_urls)
    def sync_select_url(self, max_wait_health_timeout_secs: int = 2) -> str:
        """Synchronous version of select_url."""
        import random  # 导入随机模块
        import time  # 导入时间模块

        def _select(urls: List[str]):
            # 根据选择类型返回一个 URL
            if self._choice_type == "latest_first":
                return urls[0]
            elif self._choice_type == "random":
                return random.choice(urls)
            else:
                raise ValueError(f"Invalid choice type: {self._choice_type}")

        # 如果有健康的 URL 列表，则直接返回选择的 URL
        if self._health_urls:
            return _select(self._health_urls)
        # 如果允许等待时间大于 0，则尝试等待一段时间寻找健康的 URL
        elif max_wait_health_timeout_secs > 0:
            start_time = datetime.now()  # 记录开始时间
            while datetime.now() - start_time < timedelta(
                seconds=max_wait_health_timeout_secs
            ):
                if self._health_urls:  # 如果找到健康的 URL，则返回选择的 URL
                    return _select(self._health_urls)
                time.sleep(0.1)  # 等待一小段时间
        logger.warning("No healthy urls found, selecting randomly")  # 如果未找到健康的 URL，记录警告日志
        return _select(self._remote_urls)  # 返回随机选择的远程 URL
# 从泛型类型提示中提取实际的数据类，例如 List[dataclass], Optional[dataclass] 等
def _extract_dataclass_from_generic(type_hint: Type[T]) -> Union[Type[T], None]:
    import typing_inspect

    # 判断是否为泛型类型，并且有类型参数
    if typing_inspect.is_generic_type(type_hint) and typing_inspect.get_args(type_hint):
        # 返回泛型类型的第一个参数，即实际的数据类
        return typing_inspect.get_args(type_hint)[0]
    return None


# 构建请求的函数
def _build_request(self, base_url, func, path, method, *args, **kwargs):
    # 获取函数返回类型的类型提示
    return_type = get_type_hints(func).get("return")
    if return_type is None:
        # 如果返回类型没有被注解，则抛出类型错误异常
        raise TypeError("Return type must be annotated in the decorated function.")

    # 从泛型类型提示中提取实际的数据类
    actual_dataclass = _extract_dataclass_from_generic(return_type)

    # 记录调试信息，输出返回类型和实际数据类
    logger.debug(f"return_type: {return_type}, actual_dataclass: {actual_dataclass}")

    # 如果未提取到实际的数据类，则默认使用返回类型本身
    if not actual_dataclass:
        actual_dataclass = return_type

    # 获取函数的参数签名
    sig = signature(func)

    # 绑定参数，包括 self 和其他位置参数以及关键字参数
    bound = sig.bind(self, *args, **kwargs)
    bound.apply_defaults()

    # 格式化 URL，将路径中的参数用实际的参数值替换
    formatted_url = base_url + path.format(**bound.arguments)

    # 从函数签名中提取参数名列表，排除 "self"
    arg_names = list(sig.parameters.keys())[1:]

    # 将位置参数和关键字参数合并成一个字典
    combined_args = dict(zip(arg_names, args))
    combined_args.update(kwargs)

    # 初始化请求数据字典
    request_data = {}

    # 遍历合并后的参数字典
    for key, value in combined_args.items():
        if is_dataclass(value):
            # 如果值是数据类，则将 request_data 直接设置为其字典表示形式
            request_data = asdict(value)
        else:
            # 否则，将参数名和对应的值添加到 request_data 中
            request_data[key] = value

    # 初始化请求参数字典，包括方法和 URL
    request_params = {"method": method, "url": formatted_url}

    # 根据请求方法设置请求数据
    if method in ["POST", "PUT", "PATCH"]:
        request_params["json"] = request_data
    else:  # 对于 GET, DELETE 等方法
        request_params["params"] = request_data

    # 记录调试信息，输出请求参数、位置参数和关键字参数
    logger.debug(f"request_params: {request_params}, args: {args}, kwargs: {kwargs}")

    # 返回函数的返回类型、实际数据类和请求参数字典
    return return_type, actual_dataclass, request_params


# 远程 API 函数装饰器，定义路径、方法和最大等待健康超时秒数
def _api_remote(path: str, method: str = "GET", max_wait_health_timeout_secs: int = 2):
    # 定义一个装饰器函数，接受一个函数作为参数
    def decorator(func):
        # 定义一个异步函数作为装饰器的包装器，接受self和任意位置和关键字参数
        async def wrapper(self, *args, **kwargs):
            # 导入httpx模块，用于发送HTTP请求
            import httpx

            # 检查当前类是否继承自APIMixin，若没有则抛出类型错误
            if not isinstance(self, APIMixin):
                raise TypeError(
                    "The class must inherit from APIMixin to use the @_api_remote "
                    "decorator."
                )
            # 在可用的健康URL中选择一个发送请求的基础URL
            base_url = await self.select_url(
                max_wait_health_timeout_secs=max_wait_health_timeout_secs
            )
            # 构建请求的类型、实际数据类和请求参数
            return_type, actual_dataclass, request_params = _build_request(
                self, base_url, func, path, method, *args, **kwargs
            )
            # 使用异步HTTP客户端发送请求
            async with httpx.AsyncClient() as client:
                # 发起请求并等待响应
                response = await client.request(**request_params)
                # 如果响应状态码为200，则解析并返回响应内容
                if response.status_code == 200:
                    return _parse_response(
                        response.json(), return_type, actual_dataclass
                    )
                else:
                    # 如果响应状态码不为200，则抛出异常并包含错误信息
                    error_msg = f"Remote request error, error code: {response.status_code}, error msg: {response.text}"
                    raise Exception(error_msg)

        # 返回装饰器的包装器函数
        return wrapper

    # 返回装饰器函数本身
    return decorator
# 定义一个装饰器函数 `_sync_api_remote`，接受远程 API 的路径、HTTP 方法和最大等待健康状态超时秒数作为参数
def _sync_api_remote(
    path: str, method: str = "GET", max_wait_health_timeout_secs: int = 2
):
    # 定义装饰器函数 `decorator`，用于装饰实际执行 API 调用的函数
    def decorator(func):
        # 定义包装函数 `wrapper`，它将替代原始的函数调用
        def wrapper(self, *args, **kwargs):
            import requests  # 导入 requests 库，用于发送 HTTP 请求

            # 检查 self 是否为 APIMixin 的子类，否则抛出 TypeError
            if not isinstance(self, APIMixin):
                raise TypeError(
                    "The class must inherit from APIMixin to use the @_sync_api_remote "
                    "decorator."
                )
            # 调用 APIMixin 类的方法获取实际请求的基础 URL
            base_url = self.sync_select_url(
                max_wait_health_timeout_secs=max_wait_health_timeout_secs
            )

            # 调用 _build_request 函数构建请求参数
            return_type, actual_dataclass, request_params = _build_request(
                self, base_url, func, path, method, *args, **kwargs
            )

            # 发送 HTTP 请求并获取响应对象
            response = requests.request(**request_params)

            # 如果响应状态码为 200，则解析 JSON 格式的响应数据并返回
            if response.status_code == 200:
                return _parse_response(response.json(), return_type, actual_dataclass)
            else:
                # 若响应状态码不为 200，则抛出异常，包含错误信息
                error_msg = f"Remote request error, error code: {response.status_code}, error msg: {response.text}"
                raise Exception(error_msg)

        return wrapper  # 返回包装函数

    return decorator  # 返回装饰器函数
```