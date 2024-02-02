# `MetaGPT\tests\metagpt\provider\test_openai.py`

```py

# 从 unittest.mock 模块中导入 Mock 类
from unittest.mock import Mock
# 导入 pytest 模块
import pytest
# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.provider.openai_api 模块中导入 OpenAILLM 类
from metagpt.provider.openai_api import OpenAILLM
# 从 metagpt.schema 模块中导入 UserMessage 类
from metagpt.schema import UserMessage
# 将 CONFIG.openai_proxy 设置为 None

# 定义异步测试函数 test_aask_code
@pytest.mark.asyncio
async def test_aask_code():
    # 创建 OpenAILLM 实例
    llm = OpenAILLM()
    # 定义消息内容
    msg = [{"role": "user", "content": "Write a python hello world code."}]
    # 调用 aask_code 方法并获取返回结果
    rsp = await llm.aask_code(msg)  # -> {'language': 'python', 'code': "print('Hello, World!')"}
    # 断言返回结果中包含 "language" 和 "code" 字段
    assert "language" in rsp
    assert "code" in rsp
    # 断言返回结果中的 "code" 字段长度大于 0
    assert len(rsp["code"]) > 0

# 定义异步测试函数 test_aask_code_str
@pytest.mark.asyncio
async def test_aask_code_str():
    # 创建 OpenAILLM 实例
    llm = OpenAILLM()
    # 定义消息内容
    msg = "Write a python hello world code."
    # 调用 aask_code 方法并获取返回结果
    rsp = await llm.aask_code(msg)  # -> {'language': 'python', 'code': "print('Hello, World!')"}
    # 断言返回结果中包含 "language" 和 "code" 字段
    assert "language" in rsp
    assert "code" in rsp
    # 断言返回结果中的 "code" 字段长度大于 0
    assert len(rsp["code"]) > 0

# 定义异步测试函数 test_aask_code_Message
@pytest.mark.asyncio
async def test_aask_code_Message():
    # 创建 OpenAILLM 实例
    llm = OpenAILLM()
    # 创建 UserMessage 实例
    msg = UserMessage("Write a python hello world code.")
    # 调用 aask_code 方法并获取返回结果
    rsp = await llm.aask_code(msg)  # -> {'language': 'python', 'code': "print('Hello, World!')"}
    # 断言返回结果中包含 "language" 和 "code" 字段
    assert "language" in rsp
    assert "code" in rsp
    # 断言返回结果中的 "code" 字段长度大于 0
    assert len(rsp["code"]) > 0

# 定义 TestOpenAI 类
class TestOpenAI:
    # 定义 config fixture
    @pytest.fixture
    def config(self):
        return Mock(
            openai_api_key="test_key",
            OPENAI_API_KEY="test_key",
            openai_base_url="test_url",
            OPENAI_BASE_URL="test_url",
            openai_proxy=None,
            openai_api_type="other",
        )

    # 定义 config_azure fixture
    @pytest.fixture
    def config_azure(self):
        return Mock(
            openai_api_key="test_key",
            OPENAI_API_KEY="test_key",
            openai_api_version="test_version",
            openai_base_url="test_url",
            OPENAI_BASE_URL="test_url",
            openai_proxy=None,
            openai_api_type="azure",
        )

    # 定义 config_proxy fixture
    @pytest.fixture
    def config_proxy(self):
        return Mock(
            openai_api_key="test_key",
            OPENAI_API_KEY="test_key",
            openai_base_url="test_url",
            OPENAI_BASE_URL="test_url",
            openai_proxy="http://proxy.com",
            openai_api_type="other",
        )

    # 定义 config_azure_proxy fixture
    @pytest.fixture
    def config_azure_proxy(self):
        return Mock(
            openai_api_key="test_key",
            OPENAI_API_KEY="test_key",
            openai_api_version="test_version",
            openai_base_url="test_url",
            OPENAI_BASE_URL="test_url",
            openai_proxy="http://proxy.com",
            openai_api_type="azure",
        )

    # 定义测试函数 test_make_client_kwargs_without_proxy
    def test_make_client_kwargs_without_proxy(self, config):
        # 创建 OpenAILLM 实例
        instance = OpenAILLM()
        # 将 config 赋值给实例的 config 属性
        instance.config = config
        # 调用 _make_client_kwargs 方法并获取返回结果
        kwargs = instance._make_client_kwargs()
        # 断言返回结果符合预期
        assert kwargs == {"api_key": "test_key", "base_url": "test_url"}
        assert "http_client" not in kwargs

    # 定义测试函数 test_make_client_kwargs_without_proxy_azure
    def test_make_client_kwargs_without_proxy_azure(self, config_azure):
        # 创建 OpenAILLM 实例
        instance = OpenAILLM()
        # 将 config_azure 赋值给实例的 config 属性
        instance.config = config_azure
        # 调用 _make_client_kwargs 方法并获取返回结果
        kwargs = instance._make_client_kwargs()
        # 断言返回结果符合预期
        assert kwargs == {"api_key": "test_key", "base_url": "test_url"}
        assert "http_client" not in kwargs

    # 定义测试函数 test_make_client_kwargs_with_proxy
    def test_make_client_kwargs_with_proxy(self, config_proxy):
        # 创建 OpenAILLM 实例
        instance = OpenAILLM()
        # 将 config_proxy 赋值给实例的 config 属性
        instance.config = config_proxy
        # 调用 _make_client_kwargs 方法并获取返回结果
        kwargs = instance._make_client_kwargs()
        # 断言返回结果中包含 "http_client"
        assert "http_client" in kwargs

    # 定义测试函数 test_make_client_kwargs_with_proxy_azure
    def test_make_client_kwargs_with_proxy_azure(self, config_azure_proxy):
        # 创建 OpenAILLM 实例
        instance = OpenAILLM()
        # 将 config_azure_proxy 赋值给实例的 config 属性
        instance.config = config_azure_proxy
        # 调用 _make_client_kwargs 方法并获取返回结果
        kwargs = instance._make_client_kwargs()
        # 断言返回结果中包含 "http_client"
        assert "http_client" in kwargs

```