# `MetaGPT\tests\metagpt\test_subscription.py`

```

# 导入必要的模块
import asyncio
import pytest
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.subscription import SubscriptionRunner

# 标记为异步测试
@pytest.mark.asyncio
async def test_subscription_run():
    callback_done = 0

    # 定义触发器，每隔一天生成一条关于OpenAI的消息
    async def trigger():
        while True:
            yield Message(content="the latest news about OpenAI")
            await asyncio.sleep(3600 * 24)

    # 定义一个模拟角色
    class MockRole(Role):
        async def run(self, message=None):
            return Message(content="")

    # 定义回调函数，统计回调次数
    async def callback(message):
        nonlocal callback_done
        callback_done += 1

    # 创建订阅运行器
    runner = SubscriptionRunner()

    roles = []
    # 订阅两个模拟角色
    for _ in range(2):
        role = MockRole()
        roles.append(role)
        await runner.subscribe(role, trigger(), callback)

    # 创建并运行订阅任务
    task = asyncio.get_running_loop().create_task(runner.run())

    # 等待回调完成
    for _ in range(10):
        if callback_done == 2:
            break
        await asyncio.sleep(0)
    else:
        raise TimeoutError("callback not call")

    # 取消订阅
    role = roles[0]
    assert role in runner.tasks
    await runner.unsubscribe(roles[0])

    # 等待取消订阅完成
    for _ in range(10):
        if role not in runner.tasks:
            break
        await asyncio.sleep(0)
    else:
        raise TimeoutError("callback not call")

    # 取消任务
    task.cancel()
    for i in runner.tasks.values():
        i.cancel()

# 标记为异步测试
@pytest.mark.asyncio
async def test_subscription_run_error(loguru_caplog):
    # 定义触发器1，每隔一天生成一条关于OpenAI的消息
    async def trigger1():
        while True:
            yield Message(content="the latest news about OpenAI")
            await asyncio.sleep(3600 * 24)

    # 定义触发器2，生成一条关于OpenAI的消息
    async def trigger2():
        yield Message(content="the latest news about OpenAI")

    # 定义模拟角色1，抛出运行时错误
    class MockRole1(Role):
        async def run(self, message=None):
            raise RuntimeError

    # 定义模拟角色2，返回一条消息
    class MockRole2(Role):
        async def run(self, message=None):
            return Message(content="")

    # 定义回调函数，打印消息
    async def callback(msg: Message):
        print(msg)

    # 创建订阅运行器
    runner = SubscriptionRunner()

    # 订阅模拟角色1，并捕获运行时错误
    await runner.subscribe(MockRole1(), trigger1(), callback)
    with pytest.raises(RuntimeError):
        await runner.run()

    # 订阅模拟角色2
    await runner.subscribe(MockRole2(), trigger2(), callback)
    task = asyncio.get_running_loop().create_task(runner.run(False))

    # 等待任务完成
    for _ in range(10):
        if not runner.tasks:
            break
        await asyncio.sleep(0)
    else:
        raise TimeoutError("wait runner tasks empty timeout")

    # 取消任务
    task.cancel()
    for i in runner.tasks.values():
        i.cancel()
    # 断言日志记录
    assert len(loguru_caplog.records) >= 2
    logs = "".join(loguru_caplog.messages)
    assert "run error" in logs
    assert "has completed" in logs

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```