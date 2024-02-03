# `.\AutoGPT\autogpts\autogpt\tests\unit\test_web_search.py`

```py
# 导入所需的模块
import json

# 导入 pytest 模块
import pytest
# 导入 HttpError 异常类
from googleapiclient.errors import HttpError

# 导入 Agent 类
from autogpt.agents.agent import Agent
# 导入 ConfigurationError 异常类
from autogpt.agents.utils.exceptions import ConfigurationError
# 导入 web_search 函数
from autogpt.commands.web_search import google, safe_google_results, web_search

# 参数化测试用例，测试 safe_google_results 函数
@pytest.mark.parametrize(
    "query, expected_output",
    [("test", "test"), (["test1", "test2"], '["test1", "test2')],
)
def test_safe_google_results(query, expected_output):
    # 调用 safe_google_results 函数
    result = safe_google_results(query)
    # 断言返回结果为字符串类型
    assert isinstance(result, str)
    # 断言返回结果与预期输出相等
    assert result == expected_output

# 测试 safe_google_results 函数对于无效输入的情况
def test_safe_google_results_invalid_input():
    # 使用 pytest 检查是否抛出 AttributeError 异常
    with pytest.raises(AttributeError):
        safe_google_results(123)

# 参数化测试用例，测试 web_search 函数
@pytest.mark.parametrize(
    "query, num_results, expected_output_parts, return_value",
    [
        (
            "test",
            1,
            ("Result 1", "https://example.com/result1"),
            [{"title": "Result 1", "href": "https://example.com/result1"}],
        ),
        ("", 1, (), []),
        ("no results", 1, (), []),
    ],
)
def test_google_search(
    query, num_results, expected_output_parts, return_value, mocker, agent: Agent
):
    # 创建 Mock 对象
    mock_ddg = mocker.Mock()
    mock_ddg.return_value = return_value

    # 使用 mocker.patch 替换 DDGS.text 方法
    mocker.patch("autogpt.commands.web_search.DDGS.text", mock_ddg)
    # 调用 web_search 函数
    actual_output = web_search(query, agent=agent, num_results=num_results)
    # 断言预期输出部分在实际输出中
    for o in expected_output_parts:
        assert o in actual_output

# 定义 mock_googleapiclient fixture
@pytest.fixture
def mock_googleapiclient(mocker):
    # 创建 Mock 对象
    mock_build = mocker.patch("googleapiclient.discovery.build")
    mock_service = mocker.Mock()
    mock_build.return_value = mock_service
    # 返回 mock_service.cse().list().execute().get 方法
    return mock_service.cse().list().execute().get

# 参数化测试用例，测试 Google API 客户端
@pytest.mark.parametrize(
    "query, num_results, search_results, expected_output",
    [
        (
            "test",  # 第一个元组的第一个元素，表示测试名称
            3,  # 第一个元组的第二个元素，表示数量
            [  # 第一个元组的第三个元素，包含三个字典，每个字典有一个链接键值对
                {"link": "http://example.com/result1"},
                {"link": "http://example.com/result2"},
                {"link": "http://example.com/result3"},
            ],
            [  # 第一个元组的第四个元素，包含三个链接字符串
                "http://example.com/result1",
                "http://example.com/result2",
                "http://example.com/result3",
            ],
        ),
        (  # 第二个元组
            "",  # 第二个元组的第一个元素，表示空字符串
            3,  # 第二个元组的第二个元素，表示数量
            [],  # 第二个元组的第三个元素，表示空列表
            [],  # 第二个元组的第四个元素，表示空列表
        ),
    ],
# 定义一个测试函数，用于测试 Google 官方搜索功能
def test_google_official_search(
    query,  # 搜索关键词
    num_results,  # 期望的搜索结果数量
    expected_output,  # 期望的搜索结果
    search_results,  # 模拟的搜索结果
    mock_googleapiclient,  # 模拟的 Google API 客户端
    agent: Agent,  # 代理对象
):
    # 设置模拟的 Google API 客户端返回的搜索结果
    mock_googleapiclient.return_value = search_results
    # 调用 google 函数进行搜索
    actual_output = google(query, agent=agent, num_results=num_results)
    # 断言实际搜索结果与期望搜索结果相同
    assert actual_output == safe_google_results(expected_output)


# 使用 pytest 的参数化装饰器，定义测试 Google 官方搜索功能中的错误情况
@pytest.mark.parametrize(
    "query, num_results, expected_error_type, http_code, error_msg",
    [
        (
            "invalid query",  # 无效的搜索关键词
            3,  # 期望的搜索结果数量
            HttpError,  # 期望的错误类型
            400,  # HTTP 状态码
            "Invalid Value",  # 错误消息
        ),
        (
            "invalid API key",  # 无效的 API 密钥
            3,  # 期望的搜索结果数量
            ConfigurationError,  # 期望的错误类型
            403,  # HTTP 状态码
            "invalid API key",  # 错误消息
        ),
    ],
)
# 定义测试 Google 官方搜索功能中的错误情况的函数
def test_google_official_search_errors(
    query,  # 搜索关键词
    num_results,  # 期望的搜索结果数量
    expected_error_type,  # 期望的错误类型
    mock_googleapiclient,  # 模拟的 Google API 客户端
    http_code,  # HTTP 状态码
    error_msg,  # 错误消息
    agent: Agent,  # 代理对象
):
    # 定义一个模拟的 HTTP 响应类
    class resp:
        def __init__(self, _status, _reason):
            self.status = _status
            self.reason = _reason

    # 构造模拟的错误响应内容
    response_content = {
        "error": {"code": http_code, "message": error_msg, "reason": "backendError"}
    }
    # 创建一个 HttpError 对象，模拟 HTTP 请求返回的错误
    error = HttpError(
        resp=resp(http_code, error_msg),
        content=str.encode(json.dumps(response_content)),
        uri="https://www.googleapis.com/customsearch/v1?q=invalid+query&cx",
    )

    # 设置模拟的 Google API 客户端抛出的异常
    mock_googleapiclient.side_effect = error
    # 使用 pytest 的断言检查是否抛出了期望的错误类型
    with pytest.raises(expected_error_type):
        google(query, agent=agent, num_results=num_results)
```