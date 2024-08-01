# `.\DB-GPT-src\dbgpt\core\awel\tests\test_run_dag.py`

```py
# 引入必要的模块和库函数来进行测试
from typing import List

import pytest  # 引入 pytest 测试框架

# 从上级目录导入所需的类和函数
from .. import (
    DAG,
    BranchOperator,
    DAGContext,
    InputOperator,
    JoinOperator,
    MapOperator,
    ReduceStreamOperator,
    SimpleInputSource,
    TaskState,
    WorkflowRunner,
)
# 从当前目录的 conftest.py 文件中导入测试辅助函数和数据
from .conftest import (
    _is_async_iterator,
    input_node,
    input_nodes,
    runner,
    stream_input_node,
    stream_input_nodes,
)

# 使用 pytest 框架进行异步测试
@pytest.mark.asyncio
async def test_input_node(runner: WorkflowRunner):
    # 创建一个输入节点，使用字符串 "hello" 作为输入源，并指定任务 ID
    input_node = InputOperator(SimpleInputSource("hello"), task_id="112232")
    # 执行工作流，并获取执行结果
    res: DAGContext[str] = await runner.execute_workflow(input_node)
    # 断言当前任务状态为成功
    assert res.current_task_context.current_state == TaskState.SUCCESS
    # 断言任务输出为 "hello"
    assert res.current_task_context.task_output.output == "hello"

    # 定义一个生成器函数，生成从 0 到 n-1 的异步迭代器
    async def new_steam_iter(n: int):
        for i in range(n):
            yield i

    num_iter = 10
    # 创建另一个输入节点，使用 new_steam_iter 生成的异步迭代器作为输入源
    steam_input_node = InputOperator(
        SimpleInputSource(new_steam_iter(num_iter)), task_id="112232"
    )
    # 再次执行工作流，并获取执行结果
    res: DAGContext[str] = await runner.execute_workflow(steam_input_node)
    # 断言当前任务状态为成功
    assert res.current_task_context.current_state == TaskState.SUCCESS
    # 获取任务输出流
    output_steam = res.current_task_context.task_output.output_stream
    # 断言输出流不为空
    assert output_steam
    # 断言输出流是一个异步迭代器
    assert _is_async_iterator(output_steam)
    i = 0
    # 遍历异步输出流，断言其值依次为 0 到 9
    async for x in output_steam:
        assert x == i
        i += 1


@pytest.mark.asyncio
async def test_map_node(runner: WorkflowRunner, stream_input_node: InputOperator):
    # 在名为 "test_map" 的 DAG 上下文中执行以下代码块
    with DAG("test_map") as dag:
        # 创建一个映射操作符，对输入值乘以 2
        map_node = MapOperator(lambda x: x * 2)
        # 将输入节点连接到映射操作符
        stream_input_node >> map_node
        # 执行工作流，并获取执行结果
        res: DAGContext[int] = await runner.execute_workflow(map_node)
        # 获取任务输出流
        output_steam = res.current_task_context.task_output.output_stream
        # 断言输出流不为空
        assert output_steam
        i = 0
        # 遍历异步输出流，断言其值依次为 0 到 18（由于输入流中有 10 个数）
        async for x in output_steam:
            assert x == i * 2
            i += 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream_input_node, expect_sum",
    [
        ({"output_streams": [[0, 1, 2, 3]]}, 6),  # 参数化测试，输入为 [0, 1, 2, 3]，预期和为 6
        ({"output_streams": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]}, 55),  # 输入为 [0, 1, ..., 10]，预期和为 55
    ],
    indirect=["stream_input_node"],  # 使用间接参数化，即从 stream_input_node 参数中获取输入
)
async def test_reduce_node(
    runner: WorkflowRunner, stream_input_node: InputOperator, expect_sum: int
):
    # 在名为 "test_reduce_node" 的 DAG 上下文中执行以下代码块
    with DAG("test_reduce_node") as dag:
        # 创建一个流式归约操作符，对流中的元素求和
        reduce_node = ReduceStreamOperator(lambda x, y: x + y)
        # 将输入节点连接到归约操作符
        stream_input_node >> reduce_node
        # 执行工作流，并获取执行结果
        res: DAGContext[int] = await runner.execute_workflow(reduce_node)
        # 断言当前任务状态为成功
        assert res.current_task_context.current_state == TaskState.SUCCESS
        # 断言任务输出不是流式输出
        assert not res.current_task_context.task_output.is_stream
        # 断言任务输出结果等于预期和
        assert res.current_task_context.task_output.output == expect_sum


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_nodes",
    [
        ({"outputs": [0, 1, 2]}),  # 参数化测试，输入为 [0, 1, 2]
    ],
    indirect=["input_nodes"],  # 使用间接参数化，即从 input_nodes 参数中获取输入
)
async def test_join_node(runner: WorkflowRunner, input_nodes: List[InputOperator]):
    # 定义一个联接函数，对输入参数求和
    def join_func(p1, p2, p3) -> int:
        return p1 + p2 + p3
    # 创建名为 "test_join_node" 的有向无环图（DAG）
    with DAG("test_join_node") as dag:
        # 创建一个使用给定函数 join_func 的 JoinOperator 实例 join_node
        join_node = JoinOperator(join_func)
        # 遍历输入节点列表 input_nodes
        for input_node in input_nodes:
            # 将每个输入节点连接到 join_node
            input_node >> join_node
        # 使用 runner 执行 join_node 所代表的工作流，并将结果赋给 res
        res: DAGContext[int] = await runner.execute_workflow(join_node)
        # 断言执行结果的当前任务上下文状态为 SUCCESS
        assert res.current_task_context.current_state == TaskState.SUCCESS
        # 断言任务输出不是流式的
        assert not res.current_task_context.task_output.is_stream
        # 断言任务输出的结果等于 3
        assert res.current_task_context.task_output.output == 3
# 使用 pytest 的 asyncio 标记，表示这是一个异步测试函数
@pytest.mark.asyncio
# 参数化测试，测试不同的输入节点和是否奇数的情况
@pytest.mark.parametrize(
    "input_node, is_odd",
    [
        ({"outputs": [0]}, False),  # 输入节点的输出是 [0]，期望结果为假
        ({"outputs": [1]}, True),   # 输入节点的输出是 [1]，期望结果为真
    ],
    # 声明参数化的参数是 input_node，间接依赖
    indirect=["input_node"],
)
# 异步测试函数，测试分支节点的功能
async def test_branch_node(
    runner: WorkflowRunner,  # 测试运行器实例
    input_node: InputOperator,  # 输入节点操作符实例
    is_odd: bool  # 是否为奇数的布尔值
):
    # 定义一个用于连接输出的函数，返回逻辑或结果
    def join_func(o1, o2) -> int:
        print(f"join func result, o1: {o1}, o2: {o2}")
        return o1 or o2

    # 使用 DAG("test_join_node") 创建一个测试用的有向无环图（DAG）实例
    with DAG("test_join_node") as dag:
        # 创建一个映射操作符 odd_node，每个输入映射到 999
        odd_node = MapOperator(
            lambda x: 999, task_id="odd_node", task_name="odd_node_name"
        )
        # 创建一个映射操作符 even_node，每个输入映射到 888
        even_node = MapOperator(
            lambda x: 888, task_id="even_node", task_name="even_node_name"
        )
        # 创建一个连接操作符 join_node，使用定义的 join_func，不能在分支中跳过
        join_node = JoinOperator(join_func, can_skip_in_branch=False)
        # 创建一个分支操作符 branch_node，根据输入值奇偶性选择对应的映射操作符
        branch_node = BranchOperator(
            {lambda x: x % 2 == 1: odd_node, lambda x: x % 2 == 0: even_node}
        )
        # 配置数据流：分支节点的结果输入到奇数节点，奇数节点的输出输入到连接节点
        branch_node >> odd_node >> join_node
        # 配置数据流：分支节点的结果输入到偶数节点，偶数节点的输出输入到连接节点
        branch_node >> even_node >> join_node

        # 配置数据流：输入节点的输出输入到分支节点
        input_node >> branch_node

        # 运行工作流并获取结果
        res: DAGContext[int] = await runner.execute_workflow(join_node)
        # 断言最后一个任务的状态为成功
        assert res.current_task_context.current_state == TaskState.SUCCESS
        # 根据 is_odd 的值确定期望的任务输出结果
        expect_res = 999 if is_odd else 888
        assert res.current_task_context.task_output.output == expect_res
```