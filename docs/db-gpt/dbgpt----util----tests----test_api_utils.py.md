# `.\DB-GPT-src\dbgpt\util\tests\test_api_utils.py`

```py
# 导入必要的模块
import time  # 时间模块，用于处理时间相关的操作
from concurrent.futures import ThreadPoolExecutor  # 线程池模块，用于并发执行任务
from datetime import datetime, timedelta  # 日期时间模块，用于处理日期时间相关的操作
from unittest.mock import MagicMock, patch  # 单元测试模块的模拟对象和模拟补丁功能

import pytest  # 测试框架 pytest

from ..api_utils import APIMixin  # 导入自定义的 APIMixin 类


# Mock requests.get
@pytest.fixture
def mock_requests_get():
    """为 requests.get 进行模拟的测试夹具"""
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def apimixin():
    """创建 APIMixin 实例的测试夹具"""
    urls = "http://example.com,http://example2.com"
    health_check_path = "/health"
    apimixin = APIMixin(urls, health_check_path)
    yield apimixin
    # 确保在测试结束后适当关闭执行器
    apimixin._heartbeat_executor.shutdown(wait=False)


def test_apimixin_initialization(apimixin):
    """测试 APIMixin 的初始化，验证各种参数设置"""
    assert apimixin._remote_urls == ["http://example.com", "http://example2.com"]
    assert apimixin._health_check_path == "/health"
    assert apimixin._health_check_interval_secs == 5
    assert apimixin._health_check_timeout_secs == 30
    assert apimixin._choice_type == "latest_first"
    assert isinstance(apimixin._heartbeat_executor, ThreadPoolExecutor)


def test_health_check(apimixin, mock_requests_get):
    """测试 _check_health 方法"""
    url = "http://example.com"

    # 模拟成功响应
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    is_healthy, checked_url = apimixin._check_health(url)
    assert is_healthy
    assert checked_url == url

    # 模拟失败响应
    mock_requests_get.side_effect = Exception("Connection error")
    is_healthy, checked_url = apimixin._check_health(url)
    assert not is_healthy
    assert checked_url == url


def test_check_and_update_health(apimixin, mock_requests_get):
    """测试 _check_and_update_health 方法"""
    apimixin._heartbeat_map = {
        "http://example.com": datetime.now() - timedelta(seconds=3),
        "http://example2.com": datetime.now() - timedelta(seconds=10),
    }

    # 模拟响应
    def side_effect(url, timeout):
        mock_response = MagicMock()
        if url == "http://example.com/health":
            mock_response.status_code = 200
        elif url == "http://example2.com/health":
            mock_response.status_code = 500
        return mock_response

    mock_requests_get.side_effect = side_effect

    health_urls = apimixin._check_and_update_health()
    assert "http://example.com" in health_urls
    assert "http://example2.com" not in health_urls


@pytest.mark.asyncio
async def test_select_url(apimixin, mock_requests_get):
    """测试异步方法 select_url"""
    apimixin._health_urls = ["http://example.com"]

    selected_url = await apimixin.select_url()
    assert selected_url == "http://example.com"

    # 测试没有健康 URL 的情况
    apimixin._health_urls = []
    selected_url = await apimixin.select_url(max_wait_health_timeout_secs=1)
    assert selected_url in ["http://example.com", "http://example2.com"]
# 定义测试同步方法 sync_select_url 的测试函数
def test_sync_select_url(apimixin, mock_requests_get):
    """Test the synchronous sync_select_url method."""
    
    # 设置 apimixin 的健康 URL 列表，这里包含一个示例 URL
    apimixin._health_urls = ["http://example.com"]

    # 调用 sync_select_url 方法，选择一个 URL
    selected_url = apimixin.sync_select_url()
    
    # 断言选择的 URL 应该是 "http://example.com"
    assert selected_url == "http://example.com"

    # 测试没有健康 URL 的情况
    apimixin._health_urls = []
    
    # 调用 sync_select_url 方法，设置最大等待健康超时为 1 秒
    selected_url = apimixin.sync_select_url(max_wait_health_timeout_secs=1)
    
    # 断言选择的 URL 应该是 "http://example.com" 或 "http://example2.com"
    assert selected_url in ["http://example.com", "http://example2.com"]
```