# `.\AutoGPT\autogpts\autogpt\autogpt\url_utils\validators.py`

```py
# 导入必要的库
import functools
import re
from typing import Any, Callable, ParamSpec, TypeVar
from urllib.parse import urljoin, urlparse

# 定义参数规范和类型变量
P = ParamSpec("P")
T = TypeVar("T")

# 定义装饰器函数，用于验证 URL
def validate_url(func: Callable[P, T]) -> Callable[P, T]:
    """
    The method decorator validate_url is used to validate urls for any command that
    requires a url as an argument.
    """

    @functools.wraps(func)
    def wrapper(url: str, *args, **kwargs) -> Any:
        """Check if the URL is valid and not a local file accessor.

        Args:
            url (str): The URL to check

        Returns:
            the result of the wrapped function

        Raises:
            ValueError if the url fails any of the validation tests
        """

        # 最基本的检查，验证 URL 格式是否有效
        if not re.match(r"^https?://", url):
            raise ValueError("Invalid URL format")
        # 检查 URL 是否包含 scheme 和 network location
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

# 对 URL 进行清理
def sanitize_url(url: str) -> str:
    """Sanitize the URL

    Args:
        url (str): The URL to sanitize

    Returns:
        str: The sanitized URL
    """
    parsed_url = urlparse(url)
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    return urljoin(url, reconstructed_url)

# 检查是否有访问本地文件的权限
def check_local_file_access(url: str) -> bool:
    # 检查 URL 是否为本地文件
    
    Args:
        url (str): 要检查的 URL
    
    Returns:
        bool: 如果 URL 是本地文件则返回 True，否则返回 False
    """
    # 本地文件的前缀列表
    local_file_prefixes = [
        "file:///",
        "file://localhost",
    ]
    
    # 检查 URL 是否以本地文件的前缀开头，如果有任何一个前缀匹配则返回 True
    return any(url.startswith(prefix) for prefix in local_file_prefixes)
```