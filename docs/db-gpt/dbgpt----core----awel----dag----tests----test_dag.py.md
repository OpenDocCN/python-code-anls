# `.\DB-GPT-src\dbgpt\core\awel\dag\tests\test_dag.py`

```py
# 引入 asyncio 库，用于支持异步编程
import asyncio
# 引入 threading 库，用于支持多线程编程
import threading
# 引入 pytest 库，用于编写和运行测试用例
import pytest
# 从上级目录中的 base 模块中导入 DAG 和 DAGVar 类
from ..base import DAG, DAGVar


# 定义同步环境下的 DAG 上下文测试函数
def test_dag_context_sync():
    # 创建两个 DAG 实例
    dag1 = DAG("dag1")
    dag2 = DAG("dag2")

    # 在 dag1 的上下文中进行测试
    with dag1:
        # 断言当前 DAGVar 中的当前 DAG 是 dag1
        assert DAGVar.get_current_dag() == dag1
        # 在 dag2 的上下文中进行测试
        with dag2:
            # 断言当前 DAGVar 中的当前 DAG 是 dag2
            assert DAGVar.get_current_dag() == dag2
        # 断言当前 DAGVar 中的当前 DAG 仍然是 dag1
        assert DAGVar.get_current_dag() == dag1
    # 断言当前 DAGVar 中的当前 DAG 已经为 None
    assert DAGVar.get_current_dag() is None


# 定义多线程环境下的 DAG 上下文测试函数
def test_dag_context_threading():
    # 定义线程执行的函数
    def thread_function(dag):
        # 进入指定的 DAG 环境
        DAGVar.enter_dag(dag)
        # 断言当前 DAGVar 中的当前 DAG 是 dag
        assert DAGVar.get_current_dag() == dag
        # 退出 DAG 环境
        DAGVar.exit_dag()

    # 创建两个 DAG 实例
    dag1 = DAG("dag1")
    dag2 = DAG("dag2")

    # 创建两个线程，分别执行 thread_function 函数
    thread1 = threading.Thread(target=thread_function, args=(dag1,))
    thread2 = threading.Thread(target=thread_function, args=(dag2,))

    # 启动并等待两个线程执行完毕
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # 断言当前 DAGVar 中的当前 DAG 已经为 None
    assert DAGVar.get_current_dag() is None


# 定义异步环境下的 DAG 上下文测试函数
@pytest.mark.asyncio
async def test_dag_context_async():
    # 定义异步执行的函数
    async def async_function(dag):
        # 进入指定的 DAG 环境
        DAGVar.enter_dag(dag)
        # 断言当前 DAGVar 中的当前 DAG 是 dag
        assert DAGVar.get_current_dag() == dag
        # 退出 DAG 环境
        DAGVar.exit_dag()

    # 创建两个 DAG 实例
    dag1 = DAG("dag1")
    dag2 = DAG("dag2")

    # 使用 asyncio.gather 并发执行两个异步函数
    await asyncio.gather(async_function(dag1), async_function(dag2))

    # 断言当前 DAGVar 中的当前 DAG 已经为 None
    assert DAGVar.get_current_dag() is None
```