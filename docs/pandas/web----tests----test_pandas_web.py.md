# `D:\src\scipysrc\pandas\web\tests\test_pandas_web.py`

```
from unittest.mock import (  # noqa: TID251
    mock_open,
    patch,
)

import pytest
import requests

from web.pandas_web import Preprocessors


class MockResponse:
    def __init__(self, status_code: int, response: dict) -> None:
        self.status_code = status_code
        self._resp = response

    def json(self):
        return self._resp

    @staticmethod
    def raise_for_status() -> None:
        return


@pytest.fixture
def context() -> dict:
    return {
        "main": {"github_repo_url": "pandas-dev/pandas"},
        "target_path": "test_target_path",
    }


@pytest.fixture
def mock_response(monkeypatch, request) -> None:
    # 定义用于模拟响应的函数
    def mocked_resp(*args, **kwargs):
        status_code, response = request.param
        return MockResponse(status_code, response)

    # 使用 monkeypatch 设置 requests.get 方法为模拟函数 mocked_resp
    monkeypatch.setattr(requests, "get", mocked_resp)


_releases_list = [
    {
        "prerelease": False,
        "published_at": "2024-01-19T03:34:05Z",
        "tag_name": "v1.5.6",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2023-11-10T19:07:37Z",
        "tag_name": "v2.1.3",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2023-08-30T13:24:32Z",
        "tag_name": "v2.1.0",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2023-04-30T13:24:32Z",
        "tag_name": "v2.0.0",
        "assets": None,
    },
    {
        "prerelease": True,
        "published_at": "2023-01-19T03:34:05Z",
        "tag_name": "v1.5.3xd",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2027-01-19T03:34:05Z",
        "tag_name": "v10.0.1",
        "assets": None,
    },
]


@pytest.mark.parametrize("mock_response", [(200, _releases_list)], indirect=True)
def test_web_preprocessor_creates_releases(mock_response, context) -> None:
    # 使用 mock_open 创建一个 mock 文件对象 m
    m = mock_open()
    # 使用 patch 替换内置函数 open 为 mock_open 对象 m
    with patch("builtins.open", m):
        # 调用预处理器方法 Preprocessors.home_add_releases 处理上下文
        context = Preprocessors.home_add_releases(context)
        # 获取处理后的 release 版本列表
        release_versions = [release["name"] for release in context["releases"]]
        # 断言处理后的 release 版本列表符合预期
        assert release_versions == ["10.0.1", "2.1.3", "2.0.0", "1.5.6"]
```