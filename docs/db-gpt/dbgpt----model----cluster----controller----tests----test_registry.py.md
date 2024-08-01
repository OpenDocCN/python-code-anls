# `.\DB-GPT-src\dbgpt\model\cluster\controller\tests\test_registry.py`

```py
# 引入 asyncio 库，支持异步编程
import asyncio
# 引入 datetime 和 timedelta 类，用于处理日期和时间
from datetime import datetime, timedelta

# 引入 pytest 库，用于编写和运行测试
import pytest

# 从 dbgpt.model.base 模块中引入 ModelInstance 类
from dbgpt.model.base import ModelInstance
# 从 dbgpt.model.cluster.registry 模块中引入 EmbeddedModelRegistry 类
from dbgpt.model.cluster.registry import EmbeddedModelRegistry

# 定义一个 pytest fixture，返回一个 EmbeddedModelRegistry 的实例
@pytest.fixture
def model_registry():
    return EmbeddedModelRegistry()

# 定义一个 pytest fixture，返回一个 ModelInstance 实例
@pytest.fixture
def model_instance():
    return ModelInstance(
        model_name="test_model",
        host="192.168.1.1",
        port=5000,
    )

# Async function to test the registry
# 用于测试注册表的异步函数

# 标记为 asyncio 任务
@pytest.mark.asyncio
async def test_register_instance(model_registry, model_instance):
    """
    Test if an instance can be registered correctly
    """
    # 确保实例能够正确注册
    assert await model_registry.register_instance(model_instance) == True
    # 确保注册后注册表中相应模型名称的实例数量为 1
    assert len(model_registry.registry[model_instance.model_name]) == 1

# Async function to test deregistering an instance
# 用于测试注销实例的异步函数

# 标记为 asyncio 任务
@pytest.mark.asyncio
async def test_deregister_instance(model_registry, model_instance):
    """
    Test if an instance can be deregistered correctly
    """
    # 注册实例
    await model_registry.register_instance(model_instance)
    # 确保实例能够正确注销
    assert await model_registry.deregister_instance(model_instance) == True
    # 确保注销后注册表中相应模型名称的实例列表中的第一个实例状态为非健康状态
    assert not model_registry.registry[model_instance.model_name][0].healthy

# Async function to test fetching all instances
# 用于测试获取所有实例的异步函数

# 标记为 asyncio 任务
@pytest.mark.asyncio
async def test_get_all_instances(model_registry, model_instance):
    """
    Test if all instances can be retrieved, with and without the healthy_only filter
    """
    # 注册实例
    await model_registry.register_instance(model_instance)
    # 确保能够获取到所有指定模型名称的实例数量为 1
    assert len(await model_registry.get_all_instances(model_instance.model_name)) == 1
    # 确保能够获取到健康状态的指定模型名称的实例数量为 1
    assert (
        len(
            await model_registry.get_all_instances(
                model_instance.model_name, healthy_only=True
            )
        )
        == 1
    )
    # 将实例标记为非健康状态
    model_instance.healthy = False
    # 确保获取到健康状态的指定模型名称的实例数量为 0
    assert (
        len(
            await model_registry.get_all_instances(
                model_instance.model_name, healthy_only=True
            )
        )
        == 0
    )

# Async function to test selecting a healthy instance
# 用于测试选择健康实例的异步函数

# 标记为 asyncio 任务
@pytest.mark.asyncio
async def test_select_one_health_instance(model_registry, model_instance):
    """
    Test if a single healthy instance can be selected
    """
    # 注册实例
    await model_registry.register_instance(model_instance)
    # 选择一个健康的实例
    selected_instance = await model_registry.select_one_health_instance(
        model_instance.model_name
    )
    # 确保选择的实例不为 None
    assert selected_instance is not None
    # 确保选择的实例为健康状态
    assert selected_instance.healthy
    # 确保选择的实例为启用状态
    assert selected_instance.enabled

# Async function to test sending a heartbeat
# 用于测试发送心跳的异步函数

# 标记为 asyncio 任务
@pytest.mark.asyncio
async def test_send_heartbeat(model_registry, model_instance):
    """
    Test if a heartbeat can be sent and that it correctly updates the last_heartbeat timestamp
    """
    # 注册实例
    await model_registry.register_instance(model_instance)
    # 设置上次心跳时间为 10 秒钟前
    last_heartbeat = datetime.now() - timedelta(seconds=10)
    model_instance.last_heartbeat = last_heartbeat
    # 确保能够成功发送心跳
    assert await model_registry.send_heartbeat(model_instance) == True
    # 确保发送心跳后注册表中相应模型名称的第一个实例的最后心跳时间晚于上次设置的心跳时间
    assert (
        model_registry.registry[model_instance.model_name][0].last_heartbeat
        > last_heartbeat
    )
    # 确保发送心跳后注册表中相应模型名称的第一个实例为健康状态
    assert model_registry.registry[model_instance.model_name][0].healthy == True

# 标记为 asyncio 任务
@pytest.mark.asyncio
async def test_heartbeat_timeout(model_registry, model_instance):
    """
    Test if an instance is marked as unhealthy when the heartbeat is not sent within the timeout
    """
    # 创建一个新的嵌入式模型注册表实例
    model_registry = EmbeddedModelRegistry(1, 1)
    # 注册模型实例到注册表中
    await model_registry.register_instance(model_instance)
    # 将模型实例的最后心跳时间设置为超过超时时间1秒的时间前
    model_registry.registry[model_instance.model_name][0].last_heartbeat = datetime.now() - timedelta(
        seconds=model_registry.heartbeat_timeout_secs + 1
    )
    # 等待超过心跳间隔时间1秒
    await asyncio.sleep(model_registry.heartbeat_interval_secs + 1)
    # 断言模型实例的健康状态为非健康状态
    assert not model_registry.registry[model_instance.model_name][0].healthy


@pytest.mark.asyncio
async def test_multiple_instances(model_registry, model_instance):
    """
    Test if multiple instances of the same model are handled correctly
    """
    # 创建第二个模型实例
    model_instance2 = ModelInstance(
        model_name="test_model",
        host="192.168.1.2",
        port=5000,
    )
    # 注册两个模型实例到注册表中
    await model_registry.register_instance(model_instance)
    await model_registry.register_instance(model_instance2)
    # 断言获取相同模型名的所有实例数量为2
    assert len(await model_registry.get_all_instances(model_instance.model_name)) == 2


@pytest.mark.asyncio
async def test_same_model_name_different_ip_port(model_registry):
    """
    Test if instances with the same model name but different IP and port are handled correctly
    """
    # 创建两个具有相同模型名称但不同IP和端口的模型实例
    instance1 = ModelInstance(model_name="test_model", host="192.168.1.1", port=5000)
    instance2 = ModelInstance(model_name="test_model", host="192.168.1.2", port=6000)
    # 注册这两个模型实例到注册表中
    await model_registry.register_instance(instance1)
    await model_registry.register_instance(instance2)
    # 获取所有具有模型名称 "test_model" 的实例列表
    instances = await model_registry.get_all_instances("test_model")
    # 断言实例列表长度为2
    assert len(instances) == 2
    # 断言两个实例的主机地址不相同
    assert instances[0].host != instances[1].host
    # 断言两个实例的端口号不相同
    assert instances[0].port != instances[1].port
```