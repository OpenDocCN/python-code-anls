# `.\pytorch\test\lazy\test_step_closures.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需模块和类
from threading import Event
from time import sleep

# 导入特定的 Torch 模块和函数
import torch._lazy
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase

# 初始化 Torch 的时序后端
torch._lazy.ts_backend.init()

# 定义测试类 ClosuresTest，继承自 TestCase
class ClosuresTest(TestCase):
    
    # 测试同步闭包函数
    def test_synchronous(self):
        flag = Event()  # 创建一个事件对象
        assert not flag.is_set()  # 断言事件对象未设置

        # 定义一个闭包函数 closure
        def closure():
            sleep(1)  # 等待 1 秒钟
            assert not flag.is_set()  # 断言事件对象未设置
            flag.set()  # 设置事件对象

        # 添加步骤闭包函数到 Torch 的步骤管理中
        torch._lazy.add_step_closure(closure)
        torch._lazy.mark_step()  # 标记一个步骤

        # 在闭包函数完成运行之前不应该执行到这一部分
        assert flag.is_set()  # 断言事件对象已设置

    # 测试异步闭包函数
    def test_asynchronous(self):
        flag = Event()  # 创建一个事件对象
        assert not flag.is_set()  # 断言事件对象未设置

        # 定义一个闭包函数 closure
        def closure():
            sleep(1)  # 等待 1 秒钟
            assert flag.is_set()  # 断言事件对象已设置

        # 添加步骤闭包函数到 Torch 的步骤管理中，并指定异步运行
        torch._lazy.add_step_closure(closure, run_async=True)
        torch._lazy.mark_step()  # 标记一个步骤

        # 在闭包函数完成运行之前应该执行到这一部分，并在闭包函数完成前完成
        assert not flag.is_set()  # 断言事件对象未设置
        flag.set()  # 设置事件对象

    # 测试同步闭包函数中的异常情况
    def test_synchronous_exception(self):
        flag = Event()  # 创建一个事件对象
        assert not flag.is_set()  # 断言事件对象未设置

        try:
            # 定义一个闭包函数 closure，设置事件对象并抛出异常
            def closure():
                flag.set()
                raise RuntimeError("Simulating exception in closure")

            # 添加步骤闭包函数到 Torch 的步骤管理中
            torch._lazy.add_step_closure(closure)
            torch._lazy.mark_step()  # 标记一个步骤

            raise AssertionError  # 不应该执行到这里的异常抛出
        except RuntimeError as e:
            # 断言已捕获到来自闭包函数的异常
            assert flag.is_set(), "Should have caught exception from closure"

    # 测试异步闭包函数中的异常情况
    def test_asynchronous_exception(self):
        flag = Event()  # 创建一个事件对象
        assert not flag.is_set()  # 断言事件对象未设置

        # 定义一个闭包函数 closure1，设置事件对象并抛出异常
        def closure1():
            flag.set()
            raise RuntimeError("Simulating exception in closure1")

        # 添加步骤闭包函数到 Torch 的步骤管理中，并指定异步运行
        torch._lazy.add_step_closure(closure1, run_async=True)
        torch._lazy.mark_step()  # 标记一个步骤

        flag.wait(timeout=5)  # 等待事件对象设置，最多等待 5 秒钟

        try:
            # 定义一个闭包函数 closure2，不应该执行到这里
            def closure2():
                flag.clear()

            # 添加步骤闭包函数到 Torch 的步骤管理中，并指定异步运行
            torch._lazy.add_step_closure(closure2, run_async=True)
            torch._lazy.mark_step()  # 标记一个步骤

            raise AssertionError  # 不应该执行到这里的异常抛出
        except RuntimeError as e:
            # 断言已捕获到来自 closure1 的异常
            pass

        assert flag.is_set()  # 断言事件对象已设置


if __name__ == "__main__":
    run_tests()  # 运行测试用例
```