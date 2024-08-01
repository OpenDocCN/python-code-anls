# `.\DB-GPT-src\dbgpt\util\global_helper.py`

```py
# 导入必要的模块和库
import asyncio  # 异步编程支持
import os  # 操作系统接口
import random  # 随机数生成
import sys  # 系统相关功能
import time  # 时间操作
import traceback  # 异常处理跟踪
import uuid  # UUID生成
from contextlib import contextmanager  # 上下文管理器
from dataclasses import dataclass  # 数据类支持
from functools import partial, wraps  # 函数工具
from itertools import islice  # 迭代器工具
from pathlib import Path  # 文件路径操作
from typing import (  # 类型提示
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    Union,
    cast,
)


class GlobalsHelper:
    """全局变量助手类，用于获取全局变量并缓存."""

    _tokenizer: Optional[Callable[[str], List]] = None
    _stopwords: Optional[List[str]] = None

    @property
    def tokenizer(self) -> Callable[[str], List]:
        """获取分词器."""
        if self._tokenizer is None:
            tiktoken_import_err = (
                "`tiktoken` package not found, please run `pip install tiktoken`"
            )
            try:
                import tiktoken  # 尝试导入tiktoken模块
            except ImportError:
                raise ImportError(tiktoken_import_err)
            enc = tiktoken.get_encoding("gpt2")
            self._tokenizer = cast(Callable[[str], List], enc.encode)
            self._tokenizer = partial(self._tokenizer, allowed_special="all")
        return self._tokenizer  # type: ignore

    @property
    def stopwords(self) -> List[str]:
        """获取停用词列表."""
        if self._stopwords is None:
            try:
                import nltk  # 尝试导入nltk模块
                from nltk.corpus import stopwords
            except ImportError:
                raise ImportError(
                    "`nltk` package not found, please run `pip install nltk`"
                )

            from llama_index.utils import get_cache_dir  # 导入get_cache_dir函数

            cache_dir = get_cache_dir()
            nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

            # 更新nltk的数据路径以便找到数据
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(nltk_data_dir)

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", download_dir=nltk_data_dir)
            self._stopwords = stopwords.words("english")
        return self._stopwords


globals_helper = GlobalsHelper()  # 创建全局变量助手实例


def get_new_id(d: Set) -> str:
    """获取一个新的字符串类型的ID."""
    while True:
        new_id = str(uuid.uuid4())  # 生成UUID作为新ID
        if new_id not in d:  # 确保ID在集合d中唯一
            break
    return new_id


def get_new_int_id(d: Set) -> int:
    """获取一个新的整数类型的ID."""
    while True:
        new_id = random.randint(0, sys.maxsize)  # 生成一个随机整数作为新ID
        if new_id not in d:  # 确保ID在集合d中唯一
            break
    return new_id


@contextmanager
def temp_set_attrs(obj: Any, **kwargs: Any) -> Generator:
    """临时属性设置器.

    用于在类上设置临时属性的实用程序类.
    参考链接: https://tinyurl.com/2p89xymh

    """
    # 保存对象中指定属性的原始数值，以字典形式存储
    prev_values = {k: getattr(obj, k) for k in kwargs}
    # 遍历关键字参数，将对象的属性设置为对应的值
    for k, v in kwargs.items():
        setattr(obj, k, v)
    # 执行生成器函数
    try:
        yield
    # 无论生成器函数是否正常执行完成，都会执行以下代码块
    finally:
        # 恢复对象中指定属性的原始数值
        for k, v in prev_values.items():
            setattr(obj, k, v)
@dataclass
class ErrorToRetry:
    """Exception types that should be retried.

    Args:
        exception_cls (Type[Exception]): Class of exception.
        check_fn (Optional[Callable[[Any]], bool]]):
            A function that takes an exception instance as input and returns
            whether to retry.

    """

    exception_cls: Type[Exception]
    check_fn: Optional[Callable[[Any], bool]] = None


def retry_on_exceptions_with_backoff(
    lambda_fn: Callable,
    errors_to_retry: List[ErrorToRetry],
    max_tries: int = 10,
    min_backoff_secs: float = 0.5,
    max_backoff_secs: float = 60.0,
) -> Any:
    """Execute lambda function with retries and exponential backoff.

    Args:
        lambda_fn (Callable): Function to be called and output we want.
        errors_to_retry (List[ErrorToRetry]): List of errors to retry.
            At least one needs to be provided.
        max_tries (int): Maximum number of tries, including the first. Defaults to 10.
        min_backoff_secs (float): Minimum amount of backoff time between attempts.
            Defaults to 0.5.
        max_backoff_secs (float): Maximum amount of backoff time between attempts.
            Defaults to 60.

    """
    # 如果没有提供任何错误重试类型，则抛出 ValueError 异常
    if not errors_to_retry:
        raise ValueError("At least one error to retry needs to be provided")

    # 创建异常类与检查函数的字典
    error_checks = {
        error_to_retry.exception_cls: error_to_retry.check_fn
        for error_to_retry in errors_to_retry
    }
    # 获取异常类的元组
    exception_class_tuples = tuple(error_checks.keys())

    # 初始化指数退避时间和尝试次数
    backoff_secs = min_backoff_secs
    tries = 0

    # 开始执行重试逻辑
    while True:
        try:
            # 调用 lambda_fn 函数执行
            return lambda_fn()
        except exception_class_tuples as e:
            traceback.print_exc()  # 打印异常的堆栈跟踪信息
            tries += 1
            # 如果尝试次数超过最大尝试次数，则抛出异常
            if tries >= max_tries:
                raise
            # 获取当前异常类对应的检查函数
            check_fn = error_checks.get(e.__class__)
            # 如果存在检查函数且检查函数返回 False，则抛出异常
            if check_fn and not check_fn(e):
                raise
            # 等待指数退避时间
            time.sleep(backoff_secs)
            # 更新指数退避时间
            backoff_secs = min(backoff_secs * 2, max_backoff_secs)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    # 如果文本长度小于等于最大长度，则直接返回文本
    if len(text) <= max_length:
        return text
    # 否则返回截断后的文本加上省略号
    return text[: max_length - 3] + "..."


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """Iterate over an iterable in batches.

    >>> list(iter_batch([1, 2, 3, 4, 5], 3))
    [[1, 2, 3], [4, 5]]
    """
    # 创建源可迭代对象的迭代器
    source_iter = iter(iterable)
    # 循环迭代
    while source_iter:
        # 从源迭代器中获取指定大小的批次
        b = list(islice(source_iter, size))
        # 如果批次长度为 0，则退出循环
        if len(b) == 0:
            break
        # 返回当前批次
        yield b


def concat_dirs(dirname: str, basename: str) -> str:
    """
    Append basename to dirname, avoiding backslashes when running on windows.

    os.path.join(dirname, basename) will add a backslash before dirname if
    basename does not end with a slash, so we make sure it does.
    """
    # 如果 dirname 的最后一个字符不是斜杠，则添加斜杠
    dirname += "/" if dirname[-1] != "/" else ""
    # 使用 os.path.join 连接 dirname 和 basename，确保在 Windows 上避免添加反斜杠
    return os.path.join(dirname, basename)
# 定义函数：根据是否展示进度条，获取一个 tqdm 迭代器或者原始的可迭代对象
def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    """
    Optionally get a tqdm iterable. Ensures tqdm.auto is used.
    """
    # 将 items 赋值给 _iterator，作为默认返回的迭代对象
    _iterator = items
    # 如果需要展示进度条
    if show_progress:
        try:
            # 尝试导入 tqdm.auto 库
            from tqdm.auto import tqdm
            # 返回一个 tqdm 迭代器，带有描述信息
            return tqdm(items, desc=desc)
        except ImportError:
            pass
    # 如果无法导入 tqdm 或者不需要展示进度条，则返回原始的迭代对象 _iterator
    return _iterator


# 定义函数：计算文本中的 token 数量
def count_tokens(text: str) -> int:
    # 使用全局辅助函数 globals_helper.tokenizer 对文本进行分词处理，返回 token 列表
    tokens = globals_helper.tokenizer(text)
    # 返回 token 列表的长度，即 token 数量
    return len(tokens)


# 定义函数：根据模型名获取相应的 tokenizer 函数
def get_transformer_tokenizer_fn(model_name: str) -> Callable[[str], List[str]]:
    """
    Args:
        model_name(str): the model name of the tokenizer.
                        For instance, fxmarty/tiny-llama-fast-tokenizer.
    """
    try:
        # 尝试导入 transformers 库中的 AutoTokenizer 类
        from transformers import AutoTokenizer
    except ImportError:
        # 如果导入失败，抛出 ValueError 异常
        raise ValueError(
            "`transformers` package not found, please run `pip install transformers`"
        )
    # 使用给定的 model_name 创建一个 AutoTokenizer 对象
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 返回该 tokenizer 对象的 tokenize 方法作为结果
    return tokenizer.tokenize


# 定义函数：获取一个平台适配的 llama_index 缓存目录，并在必要时创建它
def get_cache_dir() -> str:
    """Locate a platform-appropriate cache directory for llama_index,
    and create it if it doesn't yet exist.
    """
    # 用户自定义缓存目录的优先级设置
    if "LLAMA_INDEX_CACHE_DIR" in os.environ:
        path = Path(os.environ["LLAMA_INDEX_CACHE_DIR"])

    # Linux, Unix, AIX 等 POSIX 系统的默认缓存目录
    elif os.name == "posix" and sys.platform != "darwin":
        path = Path("/tmp/llama_index")

    # macOS 系统的默认缓存目录
    elif sys.platform == "darwin":
        path = Path(os.path.expanduser("~"), "Library/Caches/llama_index")

    # Windows 系统的默认缓存目录
    else:
        local = os.environ.get("LOCALAPPDATA", None) or os.path.expanduser(
            "~\\AppData\\Local"
        )
        path = Path(local, "llama_index")

    # 如果该目录不存在，则创建它，确保路径存在，避免潜在的问题
    if not os.path.exists(path):
        os.makedirs(
            path, exist_ok=True
        )  # 避免 https://github.com/jerryjliu/llama_index/issues/7362
    # 返回缓存目录的路径字符串表示
    return str(path)


# 定义函数：为一个异步函数添加同步版本的装饰器
def add_sync_version(func: Any) -> Any:
    """Decorator for adding sync version of an async function. The sync version
    is added as a function attribute to the original function, func.

    Args:
        func(Any): the async function for which a sync variant will be built.
    """
    # 确保 func 是一个异步函数
    assert asyncio.iscoroutinefunction(func)

    # 定义装饰器函数 _wrapper
    @wraps(func)
    def _wrapper(*args: Any, **kwds: Any) -> Any:
        # 在当前事件循环中运行异步函数，并返回其结果
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwds))

    # 将同步版本的函数 _wrapper 添加为原始函数 func 的 sync 属性
    func.sync = _wrapper
    # 返回原始的异步函数 func，此时已经添加了同步版本的属性
    return func


# llama_index 自述文件中的示例文本
SAMPLE_TEXT = """
Context
LLMs are a phenomenal piece of technology for knowledge generation and reasoning.
They are pre-trained on large amounts of publicly available data.
How do we best augment LLMs with our own private data?
We need a comprehensive toolkit to help perform this data augmentation for LLMs.

Proposed Solution
That's where LlamaIndex comes in. LlamaIndex is a "data framework" to help
you build LLM  apps. It provides the following tools:
"""
# 声明一个字典，用于存储 LlamaIndex 的颜色对应关系
_LLAMA_INDEX_COLORS = {
    "llama_pink": "38;2;237;90;200",
    "llama_blue": "38;2;90;149;237",
    "llama_turquoise": "38;2;11;159;203",
    "llama_lavender": "38;2;155;135;227",
}

# 声明一个字典，用于存储 ANSI 颜色对应关系
_ANSI_COLORS = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "pink": "38;5;200",
}

def get_color_mapping(
    items: List[str], use_llama_index_colors: bool = True
) -> Dict[str, str]:
    """
    根据给定的条目列表获取颜色映射。

    Args:
        items (List[str]): 要映射到颜色的条目列表。
        use_llama_index_colors (bool, optional): 指示是否使用 LlamaIndex 颜色或 ANSI 颜色的标志。
            默认为 True。

    Returns:
        Dict[str, str]: 条目到颜色的映射。
    """
    # 根据 use_llama_index_colors 参数选择颜色调色板
    if use_llama_index_colors:
        color_palette = _LLAMA_INDEX_COLORS
    else:
        color_palette = _ANSI_COLORS

    # 获取颜色列表
    colors = list(color_palette.keys())
    # 构建条目到颜色的映射字典，循环使用颜色列表中的颜色
    return {item: colors[i % len(colors)] for i, item in enumerate(items)}


def _get_colored_text(text: str, color: str) -> str:
    """
    获取输入文本的带颜色版本。

    Args:
        text (str): 输入文本。
        color (str): 要应用于文本的颜色。

    Returns:
        str: 输入文本的带颜色版本。
    """
    # 合并 LlamaIndex 和 ANSI 颜色字典
    all_colors = {**_LLAMA_INDEX_COLORS, **_ANSI_COLORS}

    # 如果指定的颜色不在字典中，返回粗体和斜体化的文本
    if color not in all_colors:
        return f"\033[1;3m{text}\033[0m"  # just bolded and italicized

    # 获取指定颜色的 ANSI 控制码
    color = all_colors[color]

    # 返回带指定颜色的文本
    return f"\033[1;3;{color}m{text}\033[0m"


def print_text(text: str, color: Optional[str] = None, end: str = "") -> None:
    """
    打印带指定颜色的文本。

    Args:
        text (str): 要打印的文本。
        color (str, optional): 要应用于文本的颜色。支持的颜色有：
            llama_pink, llama_blue, llama_turquoise, llama_lavender,
            red, green, yellow, blue, magenta, cyan, pink。
        end (str, optional): 添加到文本末尾的字符串。

    Returns:
        None
    """
    # 如果指定了颜色，则获取带颜色的文本；否则使用原始文本
    text_to_print = _get_colored_text(text, color) if color is not None else text
    # 打印文本内容到标准输出，使用指定的结尾字符（如果有）
    print(text_to_print, end=end)
# 推断当前系统的 Torch 设备
def infer_torch_device() -> str:
    """Infer the input to torch.device."""
    # 尝试检查是否可以使用 CUDA
    try:
        has_cuda = torch.cuda.is_available()
    except NameError:
        # 如果出现 NameError，则导入 torch 库并重新检查 CUDA 可用性
        import torch
        has_cuda = torch.cuda.is_available()
    
    # 如果 CUDA 可用，则返回 "cuda"
    if has_cuda:
        return "cuda"
    
    # 如果 MPS 可用（多进程服务），则返回 "mps"
    if torch.backends.mps.is_available():
        return "mps"
    
    # 否则返回 "cpu"
    return "cpu"


# 生成器函数，返回一个包含单个元素的生成器
def unit_generator(x: Any) -> Generator[Any, None, None]:
    """A function that returns a generator of a single element.

    Args:
        x (Any): the element to build yield

    Yields:
        Any: the single element
    """
    # 使用 yield 关键字生成单个元素 x
    yield x


# 异步生成器函数，返回一个包含单个元素的异步生成器
async def async_unit_generator(x: Any) -> AsyncGenerator[Any, None]:
    """A function that returns a generator of a single element.

    Args:
        x (Any): the element to build yield

    Yields:
        Any: the single element
    """
    # 使用 yield 关键字生成单个元素 x，异步生成器函数需要使用 async def 声明
    yield x
```