# `.\AutoGPT\autogpts\autogpt\tests\integration\memory\utils.py`

```py
# 导入所需的库
import numpy
import pytest
from pytest_mock import MockerFixture

# 导入需要测试的模块和类
import autogpt.memory.vector.memory_item as vector_memory_item
import autogpt.memory.vector.providers.base as memory_provider_base
from autogpt.config.config import Config
from autogpt.core.resource.model_providers import OPEN_AI_EMBEDDING_MODELS
from autogpt.memory.vector import get_memory
from autogpt.memory.vector.utils import Embedding

# 定义一个 fixture，返回嵌入维度
@pytest.fixture
def embedding_dimension(config: Config):
    return OPEN_AI_EMBEDDING_MODELS[config.embedding_model].embedding_dimensions

# 定义一个 fixture，返回模拟的嵌入
@pytest.fixture
def mock_embedding(embedding_dimension: int) -> Embedding:
    return numpy.full((1, embedding_dimension), 0.0255, numpy.float32)[0]

# 定义一个 fixture，用于模拟获取嵌入
@pytest.fixture
def mock_get_embedding(mocker: MockerFixture, mock_embedding: Embedding):
    mocker.patch.object(
        vector_memory_item,
        "get_embedding",
        return_value=mock_embedding,
    )
    mocker.patch.object(
        memory_provider_base,
        "get_embedding",
        return_value=mock_embedding,
    )

# 定义一个 fixture，用于测试内存为 None 的情况
@pytest.fixture
def memory_none(agent_test_config: Config, mock_get_embedding):
    # 保存原始的内存后端配置
    was_memory_backend = agent_test_config.memory_backend

    # 将内存后端配置设置为 "no_memory"
    agent_test_config.memory_backend = "no_memory"
    # 返回一个内存对象
    yield get_memory(agent_test_config)

    # 恢复原始的内存后端配置
    agent_test_config.memory_backend = was_memory_backend
```