# `.\AutoGPT\autogpts\autogpt\tests\unit\test_api_manager.py`

```py
# 导入 pytest 模块
import pytest
# 从 pytest_mock 模块中导入 MockerFixture 类
from pytest_mock import MockerFixture

# 从 autogpt.core.resource.model_providers 模块中导入 OPEN_AI_CHAT_MODELS 和 OPEN_AI_EMBEDDING_MODELS 常量
from autogpt.core.resource.model_providers import (
    OPEN_AI_CHAT_MODELS,
    OPEN_AI_EMBEDDING_MODELS,
)
# 从 autogpt.llm.api_manager 模块中导入 ApiManager 类
from autogpt.llm.api_manager import ApiManager

# 创建 ApiManager 类的实例对象
api_manager = ApiManager()

# 重置 ApiManager 实例对象的状态
@pytest.fixture(autouse=True)
def reset_api_manager():
    api_manager.reset()
    yield

# 使用 mocker 来模拟成本
@pytest.fixture(autouse=True)
def mock_costs(mocker: MockerFixture):
    # 对 OPEN_AI_CHAT_MODELS 中的 "gpt-3.5-turbo" 模型进行成本模拟
    mocker.patch.multiple(
        OPEN_AI_CHAT_MODELS["gpt-3.5-turbo"],
        prompt_token_cost=0.0013,
        completion_token_cost=0.0025,
    )
    # 对 OPEN_AI_EMBEDDING_MODELS 中的 "text-embedding-ada-002" 模型进行成本模拟
    mocker.patch.multiple(
        OPEN_AI_EMBEDDING_MODELS["text-embedding-ada-002"],
        prompt_token_cost=0.0004,
    )
    yield

# 定义测试类 TestApiManager
class TestApiManager:
    # 测试获取方法
    def test_getter_methods(self):
        """Test the getter methods for total tokens, cost, and budget."""
        # 更新成本信息
        api_manager.update_cost(600, 1200, "gpt-3.5-turbo")
        # 设置总预算
        api_manager.set_total_budget(10.0)
        # 断言获取总提示令牌数
        assert api_manager.get_total_prompt_tokens() == 600
        # 断言获取总完成令牌数
        assert api_manager.get_total_completion_tokens() == 1200
        # 断言获取总成本
        assert api_manager.get_total_cost() == (600 * 0.0013 + 1200 * 0.0025) / 1000
        # 断言获取总预算
        assert api_manager.get_total_budget() == 10.0

    # 静态方法，测试设置总预算
    @staticmethod
    def test_set_total_budget():
        """Test if setting the total budget works correctly."""
        # 设置总预算
        total_budget = 10.0
        api_manager.set_total_budget(total_budget)

        # 断言获取总预算
        assert api_manager.get_total_budget() == total_budget

    @staticmethod
    # 定义一个测试函数，用于测试更新成本模型是否正常工作
    def test_update_cost_completion_model():
        """Test if updating the cost works correctly."""
        # 设置提示令牌数量和完成令牌数量
        prompt_tokens = 50
        completion_tokens = 100
        # 设置模型名称
        model = "gpt-3.5-turbo"

        # 调用 API 管理器的更新成本方法
        api_manager.update_cost(prompt_tokens, completion_tokens, model)

        # 断言总提示令牌数量是否等于设定的提示令牌数量
        assert api_manager.get_total_prompt_tokens() == prompt_tokens
        # 断言总完成令牌数量是否等于设定的完成令牌数量
        assert api_manager.get_total_completion_tokens() == completion_tokens
        # 断言总成本是否等于提示令牌成本和完成令牌成本的总和
        assert (
            api_manager.get_total_cost()
            == (prompt_tokens * 0.0013 + completion_tokens * 0.0025) / 1000
        )

    # 静态方法，用于测试更新成本模型是否正常工作
    @staticmethod
    def test_update_cost_embedding_model():
        """Test if updating the cost works correctly."""
        # 设置提示令牌数量和模型名称
        prompt_tokens = 1337
        model = "text-embedding-ada-002"

        # 调用 API 管理器的更新成本方法
        api_manager.update_cost(prompt_tokens, 0, model)

        # 断言总提示令牌数量是否等于设定的提示令牌数量
        assert api_manager.get_total_prompt_tokens() == prompt_tokens
        # 断言总完成令牌数量是否为0
        assert api_manager.get_total_completion_tokens() == 0
        # 断言总成本是否等于提示令牌成本
        assert api_manager.get_total_cost() == (prompt_tokens * 0.0004) / 1000
```