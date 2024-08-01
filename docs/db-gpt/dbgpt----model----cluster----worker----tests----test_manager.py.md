# `.\DB-GPT-src\dbgpt\model\cluster\worker\tests\test_manager.py`

```py
# 引入需要的数据类序列化函数
from dataclasses import asdict
# 引入类型提示
from typing import Dict, Iterator, List, Tuple
# 使用 unittest 模拟对象
from unittest.mock import AsyncMock, patch

# 引入 pytest 测试框架
import pytest

# 引入基础模块
from dbgpt.model.base import ModelInstance, WorkerApplyType
# 引入集群基础模块
from dbgpt.model.cluster.base import WorkerApplyRequest, WorkerStartupRequest
# 引入集群运行数据
from dbgpt.model.cluster.manager_base import WorkerRunData
# 引入集群测试的配置
from dbgpt.model.cluster.tests.conftest import (
    MockModelWorker,
    _create_workers,
    _new_worker_params,
    _start_worker_manager,
    manager_2_embedding_workers,
    manager_2_workers,
    manager_with_2_workers,
)
# 引入工作管理器相关函数
from dbgpt.model.cluster.worker.manager import (
    ApplyFunction,
    DeregisterFunc,
    LocalWorkerManager,
    RegisterFunc,
    SendHeartbeatFunc,
)
# 引入模型工作者基础类
from dbgpt.model.cluster.worker_base import ModelWorker
# 引入模型参数
from dbgpt.model.parameter import ModelParameters, ModelWorkerParameters, WorkerType

# 测试用模型名称和路径
_TEST_MODEL_NAME = "vicuna-13b-v1.5"
_TEST_MODEL_PATH = "/app/models/vicuna-13b-v1.5"

# pytest fixture：创建一个模拟的工作者
@pytest.fixture
def worker():
    mock_worker = _create_workers(1)
    yield mock_worker[0][0]

# pytest fixture：创建一个新的工作者参数对象
@pytest.fixture
def worker_param():
    return _new_worker_params()

# pytest fixture：创建一个本地工作者管理器
@pytest.fixture
def manager(request):
    # 如果没有请求或者请求没有参数，初始化为 None 或空列表
    if not request or not hasattr(request, "param"):
        register_func = None
        deregister_func = None
        send_heartbeat_func = None
        model_registry = None
        workers = []
    else:
        # 从请求参数中获取注册、注销、心跳发送函数及模型注册信息
        register_func = request.param.get("register_func")
        deregister_func = request.param.get("deregister_func")
        send_heartbeat_func = request.param.get("send_heartbeat_func")
        model_registry = request.param.get("model_registry")
        workers = request.param.get("model_registry")

    # 创建本地工作者管理器对象
    worker_manager = LocalWorkerManager(
        register_func=register_func,
        deregister_func=deregister_func,
        send_heartbeat_func=send_heartbeat_func,
        model_registry=model_registry,
    )
    yield worker_manager

# 异步测试函数：测试运行阻塞函数的功能
@pytest.mark.asyncio
async def test_run_blocking_func(manager: LocalWorkerManager):
    # 定义一个简单的阻塞函数，返回整数
    def f1() -> int:
        return 0

    # 定义一个接受两个整数参数并返回它们和的函数
    def f2(a: int, b: int) -> int:
        return a + b

    # 定义一个错误的异步函数，它应该返回 None，但实际上返回了整数
    async def error_f3() -> None:
        return 0

    # 测试运行阻塞函数 f1，并断言返回值为 0
    assert await manager.run_blocking_func(f1) == 0
    # 测试运行阻塞函数 f2，并传入参数 1 和 2，断言返回值为 3
    assert await manager.run_blocking_func(f2, 1, 2) == 3
    # 测试运行错误的异步函数 error_f3，断言会抛出 ValueError 异常
    with pytest.raises(ValueError):
        await manager.run_blocking_func(error_f3)

# 异步测试函数：测试添加工作者到管理器的功能
@pytest.mark.asyncio
async def test_add_worker(
    manager: LocalWorkerManager,
    worker: ModelWorker,
    worker_param: ModelWorkerParameters,
):
    # TODO test with register function
    # 断言成功添加工作者到管理器
    assert manager.add_worker(worker, worker_param)
    # 再次尝试添加相同的工作者和参数，断言添加失败（返回 False）
    assert manager.add_worker(worker, worker_param) == False
    # 根据工作者类型和模型名称生成管理器内部的工作者键
    key = manager._worker_key(worker_param.worker_type, worker_param.model_name)
    # 断言管理器中的工作者列表长度为 1
    assert len(manager.workers) == 1
    # 断言指定键的工作者列表长度为 1
    assert len(manager.workers[key]) == 1
    # 断言管理器中的工作者对象与预期相符
    assert manager.workers[key][0].worker == worker
    # 添加一个新的工作人员到管理器中，并进行断言检查是否成功添加
    assert manager.add_worker(
        worker,
        _new_worker_params(
            model_name="chatglm2-6b", model_path="/app/models/chatglm2-6b"
        ),
    )
    
    # 尝试再次添加相同的工作人员到管理器中，并进行断言检查是否返回 False，表示未成功添加
    assert (
        manager.add_worker(
            worker,
            _new_worker_params(
                model_name="chatglm2-6b", model_path="/app/models/chatglm2-6b"
            ),
        )
        == False
    )
    
    # 断言管理器中的工作人员数量是否为 2
    assert len(manager.workers) == 2
@pytest.mark.asyncio
async def test__apply_worker(manager_2_workers: LocalWorkerManager):
    manager = manager_2_workers

    async def f1(wr: WorkerRunData) -> int:
        return 0

    # 对所有 worker 应用函数 f1，并断言返回值为 [0, 0]
    assert await manager._apply_worker(None, apply_func=f1) == [0, 0]

    workers = _create_workers(4)
    async with _start_worker_manager(workers=workers) as manager:
        # 对单个模型应用函数 f1
        req = WorkerApplyRequest(
            model=workers[0][1].model_name,
            apply_type=WorkerApplyType.START,
            worker_type=WorkerType.LLM,
        )
        # 断言返回值为 [0]
        assert await manager._apply_worker(req, apply_func=f1) == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_2_workers", [{"start": False}], indirect=True)
async def test__start_all_worker(manager_2_workers: LocalWorkerManager):
    manager = manager_2_workers
    # 启动所有 worker，并断言启动成功
    out = await manager._start_all_worker(None)
    assert out.success
    assert len(manager.workers) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_workers, is_error_worker",
    [
        ({"start": False, "error_worker": False}, False),
        ({"start": False, "error_worker": True}, True),
    ],
    indirect=["manager_2_workers"],
)
async def test_start_worker_manager(
    manager_2_workers: LocalWorkerManager, is_error_worker: bool
):
    manager = manager_2_workers
    if is_error_worker:
        # 当出现错误 worker 时，预期抛出异常
        with pytest.raises(Exception):
            await manager.start()
    else:
        # 启动 manager
        await manager.start()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_workers, is_stop_error",
    [
        ({"stop": False, "stop_error": False}, False),
        ({"stop": False, "stop_error": True}, True),
    ],
    indirect=["manager_2_workers"],
)
async def test__stop_all_worker(
    manager_2_workers: LocalWorkerManager, is_stop_error: bool
):
    manager = manager_2_workers
    # 停止所有 worker，并根据 is_stop_error 断言是否成功
    out = await manager._stop_all_worker(None)
    if is_stop_error:
        assert not out.success
    else:
        assert out.success


@pytest.mark.asyncio
async def test__restart_all_worker(manager_2_workers: LocalWorkerManager):
    manager = manager_2_workers
    # 重启所有 worker，并断言重启成功
    out = await manager._restart_all_worker(None)
    assert out.success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_workers, is_stop_error",
    [
        ({"stop": False, "stop_error": False}, False),
        ({"stop": False, "stop_error": True}, True),
    ],
    indirect=["manager_2_workers"],
)
async def test_stop_worker_manager(
    manager_2_workers: LocalWorkerManager, is_stop_error: bool
):
    manager = manager_2_workers
    if is_stop_error:
        # 当停止 worker 出现错误时，预期抛出异常
        with pytest.raises(Exception):
            await manager.stop()
    else:
        # 停止 manager
        await manager.stop()


@pytest.mark.asyncio
async def test__remove_worker():
    workers = _create_workers(3)
    # 使用异步上下文管理器启动工作管理器，并将工作者数量设置为预期的值
    async with _start_worker_manager(workers=workers, stop=False) as manager:
        # 断言当前工作管理器中的工作者数量为3
        assert len(manager.workers) == 3
        # 遍历传入的工作者列表，逐个移除对应的工作者
        for _, worker_params, _ in workers:
            manager._remove_worker(worker_params)
        # 创建一个不存在的工作者参数对象，这里是模拟一个不存在的工作者
        not_exist_parmas = _new_worker_params(
            model_name="this is a not exist worker params"
        )
        # 移除刚刚创建的不存在的工作者
        manager._remove_worker(not_exist_parmas)
# 使用 pytest 的 asyncio 标记，标记此函数为异步测试函数
@pytest.mark.asyncio
# 使用 patch 装饰器，模拟替换指定路径下的函数 _build_worker
@patch("dbgpt.model.cluster.worker.manager._build_worker")
# 异步测试函数：测试模型启动过程
async def test_model_startup(mock_build_worker):
    # 异步上下文管理器：启动 worker manager
    async with _start_worker_manager() as manager:
        # 创建一个 worker 列表，只包含一个 worker
        workers = _create_workers(1)
        # 解构赋值：取出第一个 worker 的相关信息
        worker, worker_params, model_instance = workers[0]
        # mock_build_worker 返回设定的 worker
        mock_build_worker.return_value = worker

        # 构造 worker 启动请求对象
        req = WorkerStartupRequest(
            host="127.0.0.1",
            port=8001,
            model=worker_params.model_name,
            worker_type=WorkerType.LLM,
            params=asdict(worker_params),
        )
        # 调用 manager 的模型启动方法，等待完成
        await manager.model_startup(req)
        # 断言：使用 pytest 检查是否抛出异常
        with pytest.raises(Exception):
            await manager.model_startup(req)

    # 再次异步上下文管理器：启动另一个 worker manager
    async with _start_worker_manager() as manager:
        # 创建一个 worker 列表，包含一个 error_worker
        workers = _create_workers(1, error_worker=True)
        # 解构赋值：取出第一个 worker 的相关信息
        worker, worker_params, model_instance = workers[0]
        # mock_build_worker 返回设定的 worker
        mock_build_worker.return_value = worker
        # 构造 worker 启动请求对象
        req = WorkerStartupRequest(
            host="127.0.0.1",
            port=8001,
            model=worker_params.model_name,
            worker_type=WorkerType.LLM,
            params=asdict(worker_params),
        )
        # 断言：使用 pytest 检查是否抛出异常
        with pytest.raises(Exception):
            await manager.model_startup(req)


# 使用 pytest 的 asyncio 标记，标记此函数为异步测试函数
@pytest.mark.asyncio
# 使用 patch 装饰器，模拟替换指定路径下的函数 _build_worker
@patch("dbgpt.model.cluster.worker.manager._build_worker")
# 异步测试函数：测试模型关闭过程
async def test_model_shutdown(mock_build_worker):
    # 异步上下文管理器：启动 worker manager，但禁止启动和停止动作
    async with _start_worker_manager(start=False, stop=False) as manager:
        # 创建一个 worker 列表，只包含一个 worker
        workers = _create_workers(1)
        # 解构赋值：取出第一个 worker 的相关信息
        worker, worker_params, model_instance = workers[0]
        # mock_build_worker 返回设定的 worker
        mock_build_worker.return_value = worker

        # 构造 worker 启动请求对象
        req = WorkerStartupRequest(
            host="127.0.0.1",
            port=8001,
            model=worker_params.model_name,
            worker_type=WorkerType.LLM,
            params=asdict(worker_params),
        )
        # 调用 manager 的模型启动方法，等待完成
        await manager.model_startup(req)
        # 调用 manager 的模型关闭方法，等待完成
        await manager.model_shutdown(req)


# 使用 pytest 的 asyncio 标记，标记此函数为异步测试函数
@pytest.mark.asyncio
# 异步测试函数：测试支持的模型列表
async def test_supported_models(manager_2_workers: LocalWorkerManager):
    # 将传入的 manager 参数赋值给 manager 变量
    manager = manager_2_workers
    # 调用 manager 的支持的模型列表方法，等待完成
    models = await manager.supported_models()
    # 断言：检查返回的模型列表长度是否为 1
    assert len(models) == 1
    # 取出 models 列表中的第一个元素的 models 属性
    models = models[0].models
    # 断言：检查模型数量是否大于 10
    assert len(models) > 10


# 使用 pytest 的 asyncio 和 parametrize 标记，标记此函数为异步测试函数，且参数化测试
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async",
    [
        True,
        False,
    ],
)
# 异步测试函数：测试获取模型实例
async def test_get_model_instances(is_async):
    # 创建包含 3 个 worker 的列表
    workers = _create_workers(3)
    # 使用异步上下文管理器启动工作管理器，确保工作管理器正常运行
    async with _start_worker_manager(workers=workers, stop=False) as manager:
        # 断言工作管理器中的工作线程数为3
        assert len(manager.workers) == 3
        # 遍历每个工作参数
        for _, worker_params, _ in workers:
            # 获取模型名称和工作类型
            model_name = worker_params.model_name
            worker_type = worker_params.worker_type
            # 如果是异步模式
            if is_async:
                # 断言获取特定工作类型和模型名称的实例数为1
                assert (
                    len(await manager.get_model_instances(worker_type, model_name)) == 1
                )
            else:  # 如果是同步模式
                # 断言同步获取特定工作类型和模型名称的实例数为1
                assert (
                    len(manager.sync_get_model_instances(worker_type, model_name)) == 1
                )
        # 如果是异步模式
        if is_async:
            # 断言获取不存在的模型实例时返回空列表
            assert not await manager.get_model_instances(
                worker_type, "this is not exist model instances"
            )
        else:  # 如果是同步模式
            # 断言同步获取不存在的模型实例时返回空列表
            assert not manager.sync_get_model_instances(
                worker_type, "this is not exist model instances"
            )
@pytest.mark.asyncio
async def test__simple_select(
    manager_with_2_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ]
):
    # 解包测试参数，获取管理器和工作者列表
    manager, workers = manager_with_2_workers
    # 遍历工作者列表
    for _, worker_params, _ in workers:
        # 获取模型名称和工作者类型
        model_name = worker_params.model_name
        worker_type = worker_params.worker_type
        # 异步获取模型实例列表
        instances = await manager.get_model_instances(worker_type, model_name)
        # 断言至少有一个实例
        assert instances
        # 使用简单选择算法选择一个实例
        inst = manager._simple_select(worker_params.worker_type, model_name, instances)
        # 断言选择的实例不为空
        assert inst is not None
        # 断言选择的实例参数与原参数相同
        assert inst.worker_params == worker_params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async",
    [
        True,
        False,
    ],
)
async def test_select_one_instance(
    is_async: bool,
    manager_with_2_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
):
    # 解包测试参数，获取管理器和工作者列表
    manager, workers = manager_with_2_workers
    # 遍历工作者列表
    for _, worker_params, _ in workers:
        # 获取模型名称和工作者类型
        model_name = worker_params.model_name
        worker_type = worker_params.worker_type
        # 根据是否异步选择获取模型实例
        if is_async:
            inst = await manager.select_one_instance(worker_type, model_name)
        else:
            inst = manager.sync_select_one_instance(worker_type, model_name)
        # 断言获取的实例不为空
        assert inst is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async",
    [
        True,
        False,
    ],
)
async def test__get_model(
    is_async: bool,
    manager_with_2_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
):
    # 解包测试参数，获取管理器和工作者列表
    manager, workers = manager_with_2_workers
    # 遍历工作者列表
    for _, worker_params, _ in workers:
        # 获取模型名称和工作者类型
        model_name = worker_params.model_name
        worker_type = worker_params.worker_type
        # 准备参数字典
        params = {"model": model_name}
        # 根据是否异步获取模型资源
        if is_async:
            wr = await manager._get_model(params, worker_type=worker_type)
        else:
            wr = manager._sync_get_model(params, worker_type=worker_type)
        # 断言获取的模型资源不为空
        assert wr is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_with_2_workers, expected_messages",
    [
        ({"stream_messags": ["Hello", " world."]}, "Hello world."),
        ({"stream_messags": ["你好，我是", "张三。"]}, "你好，我是张三。"),
    ],
    indirect=["manager_with_2_workers"],
)
async def test_generate_stream(
    manager_with_2_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
    expected_messages: str,
):
    # 解包测试参数，获取管理器和工作者列表
    manager, workers = manager_with_2_workers
    # 遍历工作者列表
    for _, worker_params, _ in workers:
        # 获取模型名称和工作者类型
        model_name = worker_params.model_name
        worker_type = worker_params.worker_type
        # 准备参数字典
        params = {"model": model_name}
        # 初始化文本为空字符串
        text = ""
        # 异步迭代生成文本流
        async for out in manager.generate_stream(params):
            text = out.text
        # 断言生成的文本与预期相符
        assert text == expected_messages
    [
        # 第一个元组：包含一个字典和一个期望的输出字符串
        ({"stream_messags": ["Hello", " world."]}, "Hello world."),
        # 第二个元组：包含一个字典和一个期望的输出字符串，使用了中文字符
        ({"stream_messags": ["你好，我是", "张三。"]}, "你好，我是张三。"),
    ],
    # 使用 pytest 的 indirect 参数，指定使用自定义的 fixture "manager_with_2_workers" 来间接传递测试数据
    indirect=["manager_with_2_workers"],
# 引入pytest库，用于编写和运行测试用例
@pytest.mark.asyncio
# 测试函数：测试生成文本功能
async def test_generate(
    # 参数1: 包含两个工作进程的管理器和工作进程列表
    manager_with_2_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
    # 参数2: 预期的生成文本结果
    expected_messages: str,
):
    # 解构赋值，获取管理器和工作进程列表
    manager, workers = manager_with_2_workers
    # 遍历工作进程列表
    for _, worker_params, _ in workers:
        # 获取模型名称
        model_name = worker_params.model_name
        # 获取工作进程类型
        worker_type = worker_params.worker_type
        # 构造参数字典，指定模型名称
        params = {"model": model_name}
        # 调用管理器的生成文本方法，异步等待结果
        out = await manager.generate(params)
        # 断言生成的文本与预期结果相等
        assert out.text == expected_messages


@pytest.mark.asyncio
# 参数化测试函数：测试嵌入向量功能
@pytest.mark.parametrize(
    # 参数1: 包含两个嵌入向量工作进程的管理器、预期嵌入向量、是否异步标志
    "manager_2_embedding_workers, expected_embedding, is_async",
    [
        # 参数化数据集1
        ({"embeddings": [[1, 2, 3], [4, 5, 6]]}, [[1, 2, 3], [4, 5, 6]], True),
        # 参数化数据集2
        ({"embeddings": [[0, 0, 0], [1, 1, 1]]}, [[0, 0, 0], [1, 1, 1]], False),
    ],
    # 指定"manager_2_embedding_workers"参数为间接参数化
    indirect=["manager_2_embedding_workers"],
)
# 测试函数：测试嵌入向量返回
async def test_embeddings(
    # 参数1: 包含两个嵌入向量工作进程的管理器和工作进程列表
    manager_2_embedding_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
    # 参数2: 预期的嵌入向量结果
    expected_embedding: List[List[int]],
    # 参数3: 是否异步标志
    is_async: bool,
):
    # 解构赋值，获取管理器和工作进程列表
    manager, workers = manager_2_embedding_workers
    # 遍历工作进程列表
    for _, worker_params, _ in workers:
        # 获取模型名称
        model_name = worker_params.model_name
        # 获取工作进程类型
        worker_type = worker_params.worker_type
        # 构造参数字典，指定模型名称和输入文本列表
        params = {"model": model_name, "input": ["hello", "world"]}
        # 根据异步标志选择调用异步嵌入向量方法或同步嵌入向量方法
        if is_async:
            out = await manager.embeddings(params)
        else:
            out = manager.sync_embeddings(params)
        # 断言返回的嵌入向量与预期结果相等
        assert out == expected_embedding


@pytest.mark.asyncio
# 测试函数：测试参数描述信息获取
async def test_parameter_descriptions(
    # 参数1: 包含两个工作进程的管理器和工作进程列表
    manager_with_2_workers: Tuple[
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ]
):
    # 解构赋值，获取管理器和工作进程列表
    manager, workers = manager_with_2_workers
    # 遍历工作进程列表
    for _, worker_params, _ in workers:
        # 获取模型名称
        model_name = worker_params.model_name
        # 获取工作进程类型
        worker_type = worker_params.worker_type
        # 调用管理器的参数描述信息获取方法，获取参数描述
        params = await manager.parameter_descriptions(worker_type, model_name)
        # 断言参数描述信息不为None，并且参数数量大于5
        assert params is not None
        assert len(params) > 5


@pytest.mark.asyncio
# 测试函数：更新所有工作进程参数（待实现）
async def test__update_all_worker_params():
    # TODO
    # 目前此函数仅作为占位符，尚未实现具体功能
    pass
```