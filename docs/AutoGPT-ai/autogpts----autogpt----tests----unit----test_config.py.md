# `.\AutoGPT\autogpts\autogpt\tests\unit\test_config.py`

```py
"""
Test cases for the config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
# 导入必要的模块
import os
from typing import Any
from unittest import mock
from unittest.mock import patch

import pytest
from openai.pagination import SyncPage
from openai.types import Model
from pydantic import SecretStr

# 导入需要测试的类和方法
from autogpt.app.configurator import GPT_3_MODEL, GPT_4_MODEL, apply_overrides_to_config
from autogpt.config import Config, ConfigBuilder

# 测试初始值是否正确
def test_initial_values(config: Config) -> None:
    """
    Test if the initial values of the config class attributes are set correctly.
    """
    assert config.continuous_mode is False
    assert config.tts_config.speak_mode is False
    assert config.fast_llm == "gpt-3.5-turbo-16k"
    assert config.smart_llm.startswith("gpt-4")

# 测试设置连续模式是否正确
def test_set_continuous_mode(config: Config) -> None:
    """
    Test if the set_continuous_mode() method updates the continuous_mode attribute.
    """
    # 存储连续模式以便在测试后重置
    continuous_mode = config.continuous_mode

    config.continuous_mode = True
    assert config.continuous_mode is True

    # 重置连续模式
    config.continuous_mode = continuous_mode

# 测试设置说话模式是否正确
def test_set_speak_mode(config: Config) -> None:
    """
    Test if the set_speak_mode() method updates the speak_mode attribute.
    """
    # 存储说话模式以便在测试后重置
    speak_mode = config.tts_config.speak_mode

    config.tts_config.speak_mode = True
    assert config.tts_config.speak_mode is True

    # 重置说话模式
    config.tts_config.speak_mode = speak_mode

# 测试设置快速LLM是否正确
def test_set_fast_llm(config: Config) -> None:
    """
    Test if the set_fast_llm() method updates the fast_llm attribute.
    """
    # 存储模型名称以便在测试后重置
    fast_llm = config.fast_llm

    config.fast_llm = "gpt-3.5-turbo-test"
    assert config.fast_llm == "gpt-3.5-turbo-test"

    # 重置模型名称
    config.fast_llm = fast_llm
# 测试 set_smart_llm() 方法是否更新 smart_llm 属性
def test_set_smart_llm(config: Config) -> None:
    # 存储模型名称以便在测试后重置
    smart_llm = config.smart_llm

    # 设置 config.smart_llm 为 "gpt-4-test"
    config.smart_llm = "gpt-4-test"
    # 断言 config.smart_llm 是否为 "gpt-4-test"
    assert config.smart_llm == "gpt-4-test"

    # 重置模型名称
    config.smart_llm = smart_llm


# 使用 patch 装饰器模拟 mock_list_models 方法
@patch("openai.resources.models.Models.list")
# 测试如果 gpt-4 不可用，则模型更新为 gpt-3.5-turbo
def test_fallback_to_gpt3_if_gpt4_not_available(
    mock_list_models: Any, config: Config
) -> None:
    # 存储 fast_llm 和 smart_llm 以便在测试后重置
    fast_llm = config.fast_llm
    smart_llm = config.smart_llm

    # 设置 config.fast_llm 和 config.smart_llm 为 "gpt-4"
    config.fast_llm = "gpt-4"
    config.smart_llm = "gpt-4"

    # 模拟 mock_list_models 返回值
    mock_list_models.return_value = SyncPage(
        data=[Model(id=GPT_3_MODEL, created=0, object="model", owned_by="AutoGPT")],
        object="Models",  # 不清楚这应该是什么，但不相关
    )

    # 应用配置覆盖
    apply_overrides_to_config(
        config=config,
        gpt3only=False,
        gpt4only=False,
    )

    # 断言 config.fast_llm 和 config.smart_llm 是否为 "gpt-3.5-turbo"
    assert config.fast_llm == "gpt-3.5-turbo"
    assert config.smart_llm == "gpt-3.5-turbo"

    # 重置配置
    config.fast_llm = fast_llm
    config.smart_llm = smart_llm


# 测试缺少 azure 配置
def test_missing_azure_config(config: Config) -> None:
    # 断言 config.openai_credentials 不为 None
    assert config.openai_credentials is not None

    # 设置配置文件路径
    config_file = config.app_data_dir / "azure_config.yaml"
    # 断言抛出 FileNotFoundError 异常
    with pytest.raises(FileNotFoundError):
        config.openai_credentials.load_azure_config(config_file)

    # 写入空内容到配置文件
    config_file.write_text("")
    # 断言抛出 ValueError 异常
    with pytest.raises(ValueError):
        config.openai_credentials.load_azure_config(config_file)

    # 断言 config.openai_credentials.api_type 不为 "azure"
    assert config.openai_credentials.api_type != "azure"
    # 断言 config.openai_credentials.api_version 为 ""
    assert config.openai_credentials.api_version == ""
    # 断言 config.openai_credentials.azure_model_to_deploy_id_map 为 None


# fixture 为 config_with_azure
@pytest.fixture
def config_with_azure(config: Config):
    # 设置配置文件路径
    config_file = config.app_data_dir / "azure_config.yaml"
    # 写入 azure_api_type: azure 到配置文件
    config_file.write_text(
        """
        azure_api_type: azure
        """
    )
# 设置 Azure API 版本
azure_api_version: 2023-06-01-preview
# 设置 Azure 终端点
azure_endpoint: https://dummy.openai.azure.com
# 设置 Azure 模型映射关系
azure_model_map:
    {config.fast_llm}: FAST-LLM_ID
    {config.smart_llm}: SMART-LLM_ID
    {config.embedding_model}: embedding-deployment-id-for-azure
"""
    )
    # 设置环境变量 USE_AZURE 为 True
    os.environ["USE_AZURE"] = "True"
    # 设置环境变量 AZURE_CONFIG_FILE 为配置文件路径
    os.environ["AZURE_CONFIG_FILE"] = str(config_file)
    # 从环境变量构建配置对象
    config_with_azure = ConfigBuilder.build_config_from_env(
        project_root=config.project_root
    )
    # 返回配置对象
    yield config_with_azure
    # 删除环境变量 USE_AZURE
    del os.environ["USE_AZURE"]
    # 删除环境变量 AZURE_CONFIG_FILE

# 测试 Azure 配置
def test_azure_config(config_with_azure: Config) -> None:
    # 断言是否存在 Azure 凭据
    assert (credentials := config_with_azure.openai_credentials) is not None
    # 断言 API 类型为 azure
    assert credentials.api_type == "azure"
    # 断言 API 版本为指定值
    assert credentials.api_version == "2023-06-01-preview"
    # 断言 Azure 终端点为指定值
    assert credentials.azure_endpoint == SecretStr("https://dummy.openai.azure.com")
    # 断言 Azure 模型映射关系为指定值
    assert credentials.azure_model_to_deploy_id_map == {
        config_with_azure.fast_llm: "FAST-LLM_ID",
        config_with_azure.smart_llm: "SMART-LLM_ID",
        config_with_azure.embedding_model: "embedding-deployment-id-for-azure",
    }

    # 获取 fast_llm 和 smart_llm
    fast_llm = config_with_azure.fast_llm
    smart_llm = config_with_azure.smart_llm
    # 断言获取 fast_llm 的模型访问参数中的模型为 FAST-LLM_ID
    assert (
        credentials.get_model_access_kwargs(config_with_azure.fast_llm)["model"]
        == "FAST-LLM_ID"
    )
    # 断言获取 smart_llm 的模型访问参数中的模型为 SMART-LLM_ID

    # 模拟 --gpt4only
    config_with_azure.fast_llm = smart_llm
    # 断言获取 fast_llm 的模型访问参数中的模型为 SMART-LLM_ID
    assert (
        credentials.get_model_access_kwargs(config_with_azure.fast_llm)["model"]
        == "SMART-LLM_ID"
    )
    # 断言获取 smart_llm 的模型访问参数中的模型为 SMART-LLM_ID

    # 模拟 --gpt3only
    config_with_azure.fast_llm = config_with_azure.smart_llm = fast_llm
    # 断言获取 fast_llm 的模型访问参数中的模型为 FAST-LLM_ID
    assert (
        credentials.get_model_access_kwargs(config_with_azure.fast_llm)["model"]
        == "FAST-LLM_ID"
    )  # 结束 assert 语句的括号
    assert (  # 断言：验证条件是否为真
        credentials.get_model_access_kwargs(config_with_azure.smart_llm)["model"]  # 从配置中获取 Azure 智能 LLM 模型的访问参数中的模型信息
        == "FAST-LLM_ID"  # 判断获取的模型信息是否为 "FAST-LLM_ID"
    )  # 结束 assert 语句的条件判断
# 测试创建仅支持 GPT-4 的配置
def test_create_config_gpt4only(config: Config) -> None:
    # 使用 mock.patch 创建一个模拟对象，模拟 ApiManager.get_models 方法
    with mock.patch("autogpt.llm.api_manager.ApiManager.get_models") as mock_get_models:
        # 设置模拟对象的返回值为包含 GPT-4 模型信息的列表
        mock_get_models.return_value = [
            Model(id=GPT_4_MODEL, created=0, object="model", owned_by="AutoGPT")
        ]
        # 应用配置覆盖，设置为仅支持 GPT-4
        apply_overrides_to_config(
            config=config,
            gpt4only=True,
        )
        # 断言配置中的 fast_llm 和 smart_llm 值为 GPT-4 模型
        assert config.fast_llm == GPT_4_MODEL
        assert config.smart_llm == GPT_4_MODEL


# 测试创建仅支持 GPT-3 的配置
def test_create_config_gpt3only(config: Config) -> None:
    # 使用 mock.patch 创建一个模拟对象，模拟 ApiManager.get_models 方法
    with mock.patch("autogpt.llm.api_manager.ApiManager.get_models") as mock_get_models:
        # 设置模拟对象的返回值为包含 GPT-3 模型信息的字典
        mock_get_models.return_value = [{"id": GPT_3_MODEL}]
        # 应用配置覆盖，设置为仅支持 GPT-3
        apply_overrides_to_config(
            config=config,
            gpt3only=True,
        )
        # 断言配置中的 fast_llm 和 smart_llm 值为 GPT-3 模型
        assert config.fast_llm == GPT_3_MODEL
        assert config.smart_llm == GPT_3_MODEL
```