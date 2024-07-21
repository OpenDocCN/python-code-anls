# `.\pytorch\tools\github\github_utils.py`

```
"""GitHub Utilities"""

# 导入所需模块和类型
from __future__ import annotations
import json
import os
from typing import Any, Callable, cast, Dict
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

# 定义函数 gh_fetch_url_and_headers，用于获取 URL 和其响应头信息
def gh_fetch_url_and_headers(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> tuple[Any, Any]:
    # 如果 headers 为 None，则初始化为空字典
    if headers is None:
        headers = {}
    # 从环境变量中获取 GitHub Token
    token = os.environ.get("GITHUB_TOKEN")
    # 如果存在 Token 并且 URL 是 GitHub API 的 URL，则添加 Authorization 头部
    if token is not None and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"token {token}"
    # 将 data 转换为 JSON 格式的字节流，如果 data 不为 None
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        # 使用 urlopen 发起请求，获取响应连接对象 conn
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            # 返回连接对象的 headers 和通过 reader 函数读取的数据
            return conn.headers, reader(conn)
    except HTTPError as err:
        # 处理 HTTP 错误，如果是 403 错误且包含必要的限速信息，则打印限速信息
        if err.code == 403 and all(
            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]
        ):
            print(
                f"""Rate limit exceeded:
                Used: {err.headers['X-RateLimit-Used']}
                Limit: {err.headers['X-RateLimit-Limit']}
                Remaining: {err.headers['X-RateLimit-Remaining']}
                Resets at: {err.headers['x-RateLimit-Reset']}"""
            )
        # 抛出处理后的 HTTPError 异常
        raise

# 定义函数 gh_fetch_url，简化只获取 URL 返回数据的函数
def gh_fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Any:
    # 调用 gh_fetch_url_and_headers 函数，返回其返回结果的第二个元素
    return gh_fetch_url_and_headers(
        url, headers=headers, data=data, reader=json.load, method=method
    )[1]

# 定义内部函数 _gh_fetch_json_any，用于获取 JSON 格式的任意数据
def _gh_fetch_json_any(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    # 初始化 headers，指定接受 GitHub API v3 的 JSON 格式
    headers = {"Accept": "application/vnd.github.v3+json"}
    # 如果存在 params，则将其拼接到 URL 中作为查询参数
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    # 调用 gh_fetch_url 函数，获取 URL 返回的 JSON 数据
    return gh_fetch_url(url, headers=headers, data=data, reader=json.load)

# 定义函数 gh_fetch_json_dict，用于获取 JSON 格式数据并强制转换为字典类型
def gh_fetch_json_dict(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 调用 _gh_fetch_json_any 函数，获取 JSON 数据并使用 cast 强制转换为字典类型
    return cast(Dict[str, Any], _gh_fetch_json_any(url, params, data))

# 定义函数 gh_fetch_commit，用于获取指定仓库的提交信息
def gh_fetch_commit(org: str, repo: str, sha: str) -> dict[str, Any]:
    # 构建获取提交信息的 URL，调用 gh_fetch_json_dict 函数获取提交信息并返回
    return gh_fetch_json_dict(
        f"https://api.github.com/repos/{org}/{repo}/commits/{sha}"
    )
```