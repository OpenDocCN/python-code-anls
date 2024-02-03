# `.\AutoGPT\autogpts\autogpt\tests\vcr\vcr_filter.py`

```py
# 导入必要的库
import contextlib
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

from vcr.request import Request

# 从环境变量中获取代理信息
PROXY = os.environ.get("PROXY")

# 定义需要替换的内容列表
REPLACEMENTS: List[Dict[str, str]] = [
    {
        "regex": r"\w{3} \w{3} {1,2}\d{1,2} \d{2}:\d{2}:\d{2} \d{4}",
        "replacement": "Tue Jan  1 00:00:00 2000",
    },
    {
        "regex": r"<selenium.webdriver.chrome.webdriver.WebDriver[^>]*>",
        "replacement": "",
    },
]

# 定义允许访问的主机名列表
ALLOWED_HOSTNAMES: List[str] = [
    "api.openai.com",
    "localhost:50337",
    "duckduckgo.com",
]

# 如果存在代理信息，则添加到允许访问的主机名列表中
if PROXY:
    ALLOWED_HOSTNAMES.append(PROXY)
    ORIGINAL_URL = PROXY
else:
    ORIGINAL_URL = "no_ci"

NEW_URL = "api.openai.com"

# 替换消息内容中的动态项
def replace_message_content(content: str, replacements: List[Dict[str, str]]) -> str:
    for replacement in replacements:
        pattern = re.compile(replacement["regex"])
        content = pattern.sub(replacement["replacement"], content)

    return content

# 冻结请求体中的动态项
def freeze_request_body(body: dict) -> bytes:
    """Remove any dynamic items from the request body"""

    if "messages" not in body:
        return json.dumps(body, sort_keys=True).encode()

    if "max_tokens" in body:
        del body["max_tokens"]

    for message in body["messages"]:
        if "content" in message and "role" in message:
            if message["role"] == "system":
                message["content"] = replace_message_content(
                    message["content"], REPLACEMENTS
                )

    return json.dumps(body, sort_keys=True).encode()

# 冻结请求
def freeze_request(request: Request) -> Request:
    if not request or not request.body:
        return request

    with contextlib.suppress(ValueError):
        request.body = freeze_request_body(
            json.loads(
                request.body.getvalue()
                if isinstance(request.body, BytesIO)
                else request.body
            )
        )

    return request
# 在记录响应之前对响应进行处理，删除响应头中的"Transfer-Encoding"字段
def before_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    if "Transfer-Encoding" in response["headers"]:
        del response["headers"]["Transfer-Encoding"]
    return response


# 在记录请求之前对请求进行处理，替换请求中的原始主机名为新的主机名
def before_record_request(request: Request) -> Request | None:
    # 替换请求中的主机名
    request = replace_request_hostname(request, ORIGINAL_URL, NEW_URL)

    # 过滤请求中的主机名
    filtered_request = filter_hostnames(request)
    if not filtered_request:
        return None

    # 冻结请求中的动态数据
    filtered_request_without_dynamic_data = freeze_request(filtered_request)
    return filtered_request_without_dynamic_data


# 替换请求中的主机名
def replace_request_hostname(
    request: Request, original_url: str, new_hostname: str
) -> Request:
    # 解析请求的 URI
    parsed_url = urlparse(request.uri)

    # 如果请求的主机名在原始 URL 中，则替换主机名和路径
    if parsed_url.hostname in original_url:
        new_path = parsed_url.path.replace("/proxy_function", "")
        request.uri = urlunparse(
            parsed_url._replace(netloc=new_hostname, path=new_path, scheme="https")
        )

    return request


# 过滤请求中的主机名
def filter_hostnames(request: Request) -> Request | None:
    # 在这里添加过滤主机名的实现
    if any(hostname in request.url for hostname in ALLOWED_HOSTNAMES):
        return request
    else:
        return None
```