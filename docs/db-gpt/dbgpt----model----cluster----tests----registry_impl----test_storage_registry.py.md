# `.\DB-GPT-src\dbgpt\model\cluster\tests\registry_impl\test_storage_registry.py`

```py
# 引入 asyncio 库，用于异步编程
import asyncio
# 从 concurrent.futures 模块中引入 ThreadPoolExecutor 类，用于线程池管理
from concurrent.futures import ThreadPoolExecutor
# 从 datetime 模块中引入 datetime 和 timedelta 类，用于处理日期时间
from datetime import datetime, timedelta
# 从 unittest.mock 模块中引入 AsyncMock 和 MagicMock 类，用于生成模拟对象
from unittest.mock import AsyncMock, MagicMock

# 引入 pytest 测试框架
import pytest

# 从 dbgpt.core.interface.storage 中引入 InMemoryStorage 和 QuerySpec 类
from dbgpt.core.interface.storage import InMemoryStorage, QuerySpec
# 从 dbgpt.util.serialization.json_serialization 中引入 JsonSerializer 类
from dbgpt.util.serialization.json_serialization import JsonSerializer

# 从 ...registry_impl.storage 中引入 ModelInstance, ModelInstanceStorageItem, StorageModelRegistry 类
from ...registry_impl.storage import (
    ModelInstance,
    ModelInstanceStorageItem,
    StorageModelRegistry,
)


@pytest.fixture
def in_memory_storage():
    # 返回一个 InMemoryStorage 的实例，使用 JsonSerializer 进行序列化
    return InMemoryStorage(serializer=JsonSerializer())


@pytest.fixture
def thread_pool_executor():
    # 返回一个 ThreadPoolExecutor 的实例，最大工作线程数为 2
    return ThreadPoolExecutor(max_workers=2)


@pytest.fixture
def registry(in_memory_storage, thread_pool_executor):
    # 返回一个 StorageModelRegistry 的实例，使用给定的 storage 和 executor，
    # 并设置心跳间隔为 1 秒，心跳超时为 2 秒
    return StorageModelRegistry(
        storage=in_memory_storage,
        executor=thread_pool_executor,
        heartbeat_interval_secs=1,
        heartbeat_timeout_secs=2,
    )


@pytest.fixture
def model_instance():
    # 返回一个 ModelInstance 的实例，表示一个模型实例，包括模型名称、主机、端口等信息，
    # 并设置最后心跳时间为当前时间
    return ModelInstance(
        model_name="test_model",
        host="localhost",
        port=8080,
        weight=1.0,
        check_healthy=True,
        healthy=True,
        enabled=True,
        prompt_template=None,
        last_heartbeat=datetime.now(),
    )


@pytest.fixture
def model_instance_2():
    # 返回另一个 ModelInstance 的实例，具体设置与 model_instance 相似，但端口不同
    return ModelInstance(
        model_name="test_model",
        host="localhost",
        port=8081,
        weight=1.0,
        check_healthy=True,
        healthy=True,
        enabled=True,
        prompt_template=None,
        last_heartbeat=datetime.now(),
    )


@pytest.fixture
def model_instance_3():
    # 返回第三个 ModelInstance 的实例，表示另一个模型实例，模型名称不同于之前的两个
    return ModelInstance(
        model_name="test_model_2",
        host="localhost",
        port=8082,
        weight=1.0,
        check_healthy=True,
        healthy=True,
        enabled=True,
        prompt_template=None,
        last_heartbeat=datetime.now(),
    )


@pytest.fixture
def model_instance_storage_item(model_instance):
    # 根据给定的 model_instance 创建一个 ModelInstanceStorageItem 的实例
    return ModelInstanceStorageItem.from_model_instance(model_instance)


@pytest.mark.asyncio
async def test_register_instance_new(registry, model_instance):
    """Test registering a new model instance."""
    # 测试注册一个新的模型实例，并验证注册是否成功
    result = await registry.register_instance(model_instance)

    assert result is True
    # 获取特定模型名称下的所有实例，验证实例列表长度为 1
    instances = await registry.get_all_instances(model_instance.model_name)
    assert len(instances) == 1
    saved_instance = instances[0]
    # 验证保存的实例与原始实例具有相同的模型名称、主机、端口等信息，并且健康状态为 True，最后心跳时间不为空
    assert saved_instance.model_name == model_instance.model_name
    assert saved_instance.host == model_instance.host
    assert saved_instance.port == model_instance.port
    assert saved_instance.healthy is True
    assert saved_instance.last_heartbeat is not None


@pytest.mark.asyncio
async def test_register_instance_existing(
    registry, model_instance, model_instance_storage_item
):
    """Test registering an existing model instance and updating it."""
    # 注册一个已存在的模型实例，并更新其心跳时间
    await registry.register_instance(model_instance)

    # 再次注册相同实例，期望返回 True
    result = await registry.register_instance(model_instance)

    assert result is True
    # 从注册表中获取特定模型实例的所有实例
    instances = await registry.get_all_instances(model_instance.model_name)
    
    # 确保获取到的实例列表长度为1，即确保只有一个匹配的实例
    assert len(instances) == 1
    
    # 从列表中取出更新后的实例对象
    updated_instance = instances[0]
    
    # 确保更新后的实例的模型名称与原实例的模型名称相同
    assert updated_instance.model_name == model_instance.model_name
    
    # 确保更新后的实例的主机地址与原实例的主机地址相同
    assert updated_instance.host == model_instance.host
    
    # 确保更新后的实例的端口号与原实例的端口号相同
    assert updated_instance.port == model_instance.port
    
    # 确保更新后的实例的健康状态为真
    assert updated_instance.healthy is True
    
    # 确保更新后的实例的最后心跳时间不为None，即已经有心跳信息更新
    assert updated_instance.last_heartbeat is not None
# 使用 pytest 框架进行异步测试标记
@pytest.mark.asyncio
# 定义异步测试函数，测试从注册表中注销模型实例的操作
async def test_deregister_instance(registry, model_instance):
    # 注册模型实例到注册表中
    await registry.register_instance(model_instance)

    # 调用注销模型实例的方法，并获取操作结果
    result = await registry.deregister_instance(model_instance)

    # 断言注销操作成功
    assert result is True
    # 获取所有指定模型名的实例列表
    instances = await registry.get_all_instances(model_instance.model_name)
    # 断言实例列表长度为1（即只有一个实例）
    assert len(instances) == 1
    # 获取被注销的实例
    deregistered_instance = instances[0]
    # 断言被注销的实例状态为不健康（healthy is False）
    assert deregistered_instance.healthy is False


# 使用 pytest 框架进行异步测试标记
@pytest.mark.asyncio
# 定义异步测试函数，测试获取所有模型实例的操作
async def test_get_all_instances(registry, model_instance):
    # 注册模型实例到注册表中
    await registry.register_instance(model_instance)

    # 调用获取所有模型实例的方法，并传入健康状态筛选条件
    result = await registry.get_all_instances(
        model_instance.model_name, healthy_only=True
    )

    # 断言返回的实例列表长度为1
    assert len(result) == 1
    # 断言返回的实例模型名称与预期一致
    assert result[0].model_name == model_instance.model_name


# 定义同步测试函数，测试同步获取所有模型实例的操作
def test_sync_get_all_instances(registry, model_instance):
    # 调用同步获取所有模型实例的方法，并传入健康状态筛选条件
    registry.sync_get_all_instances(model_instance.model_name, healthy_only=True)
    # 保存模型实例到存储
    registry._storage.save(ModelInstanceStorageItem.from_model_instance(model_instance))

    # 再次调用同步获取所有模型实例的方法，并传入健康状态筛选条件
    result = registry.sync_get_all_instances(
        model_instance.model_name, healthy_only=True
    )

    # 断言返回的实例列表长度为1
    assert len(result) == 1
    # 断言返回的实例模型名称与预期一致
    assert result[0].model_name == model_instance.model_name


# 使用 pytest 框架进行异步测试标记
@pytest.mark.asyncio
# 定义异步测试函数，测试发送新实例的心跳操作
async def test_send_heartbeat_new_instance(registry, model_instance):
    # 发送模型实例的心跳，并获取操作结果
    result = await registry.send_heartbeat(model_instance)

    # 断言发送心跳操作成功
    assert result is True
    # 获取所有指定模型名的实例列表
    instances = await registry.get_all_instances(model_instance.model_name)
    # 断言实例列表长度为1（即只有一个实例）
    assert len(instances) == 1
    # 获取保存的实例
    saved_instance = instances[0]
    # 断言保存的实例模型名称与预期一致
    assert saved_instance.model_name == model_instance.model_name


# 使用 pytest 框架进行异步测试标记
@pytest.mark.asyncio
# 定义异步测试函数，测试发送现有实例的心跳操作
async def test_send_heartbeat_existing_instance(registry, model_instance):
    # 注册模型实例到注册表中
    await registry.register_instance(model_instance)

    # 发送模型实例的心跳，并获取操作结果
    result = await registry.send_heartbeat(model_instance)

    # 断言发送心跳操作成功
    assert result is True
    # 获取所有指定模型名的实例列表
    instances = await registry.get_all_instances(model_instance.model_name)
    # 断言实例列表长度为1（即只有一个实例）
    assert len(instances) == 1
    # 获取更新后的实例
    updated_instance = instances[0]
    # 断言更新后的实例的最后心跳时间晚于原实例的最后心跳时间
    assert updated_instance.last_heartbeat > model_instance.last_heartbeat


# 使用 pytest 框架进行异步测试标记
@pytest.mark.asyncio
# 定义异步测试函数，测试心跳检测器机制
async def test_heartbeat_checker(
    in_memory_storage, thread_pool_executor, model_instance
):
    # 设置心跳超时秒数
    heartbeat_timeout_secs = 1
    # 创建带有指定配置参数的存储模型注册表
    registry = StorageModelRegistry(
        storage=in_memory_storage,
        executor=thread_pool_executor,
        heartbeat_interval_secs=0.1,
        heartbeat_timeout_secs=heartbeat_timeout_secs,
    )
    # 异步函数，用于检查模型实例的心跳状态
    async def check_heartbeat(model_name: str, expected_healthy: bool):
        # 从注册表获取指定模型名的所有实例
        instances = await registry.get_all_instances(model_name)
        # 断言实例列表长度为1，确保只有一个实例
        assert len(instances) == 1
        # 获取更新后的实例对象
        updated_instance = instances[0]
        # 断言实例的健康状态是否与预期一致
        assert updated_instance.healthy == expected_healthy

    # 注册模型实例到注册表
    await registry.register_instance(model_instance)
    # 第一次心跳应该成功
    await check_heartbeat(model_instance.model_name, True)
    # 等待心跳超时时间加0.5秒
    await asyncio.sleep(heartbeat_timeout_secs + 0.5)
    # 再次检查心跳，预期为失败状态
    await check_heartbeat(model_instance.model_name, False)

    # 发送心跳信号
    await registry.send_heartbeat(model_instance)
    # 应该恢复为健康状态
    await check_heartbeat(model_instance.model_name, True)
```