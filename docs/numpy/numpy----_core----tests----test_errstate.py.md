# `.\numpy\numpy\_core\tests\test_errstate.py`

```
# 导入 pytest 模块，用于编写和运行测试用例
import pytest
# 导入 sysconfig 模块，用于访问 Python 系统配置信息
import sysconfig

# 导入 numpy 库，并将其命名为 np
import numpy as np
# 从 numpy.testing 模块中导入断言函数 assert_、assert_raises 和 IS_WASM
from numpy.testing import assert_, assert_raises, IS_WASM

# 检测是否为 ARM EABI 系统上的软浮点模拟，若是则标记为 True
hosttype = sysconfig.get_config_var('HOST_GNU_TYPE')
arm_softfloat = False if hosttype is None else hosttype.endswith('gnueabi')

# 定义测试类 TestErrstate
class TestErrstate:
    # 标记为 pytest 的测试用例，并在 WASM 环境下跳过此测试
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 在 ARM 软浮点问题存在时跳过此测试，并说明原因
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-413,-15562)')
    # 测试无效操作错误处理
    def test_invalid(self):
        # 进入上下文管理器，设置所有错误都抛出，下溢时忽略
        with np.errstate(all='raise', under='ignore'):
            # 创建一个负数数组
            a = -np.arange(3)
            # 在无效操作（如负数的平方根）时忽略错误
            with np.errstate(invalid='ignore'):
                np.sqrt(a)
            # 预期下面的操作会引发浮点数错误
            with assert_raises(FloatingPointError):
                np.sqrt(a)

    # 标记为 pytest 的测试用例，并在 WASM 环境下跳过此测试
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 在 ARM 软浮点问题存在时跳过此测试，并说明原因
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-15562)')
    # 测试除零错误处理
    def test_divide(self):
        # 进入上下文管理器，设置所有错误都抛出，下溢时忽略
        with np.errstate(all='raise', under='ignore'):
            # 创建一个负数数组
            a = -np.arange(3)
            # 在除零操作时忽略错误
            with np.errstate(divide='ignore'):
                a // 0
            # 预期下面的操作会引发浮点数错误
            with assert_raises(FloatingPointError):
                a // 0
            # 预期下面的操作会引发浮点数错误，参见 gh-15562
            with assert_raises(FloatingPointError):
                a // a

    # 测试错误回调函数
    def test_errcall(self):
        # 定义一个计数器函数 foo
        count = 0
        def foo(*args):
            nonlocal count
            count += 1

        # 获取当前的错误回调函数
        olderrcall = np.geterrcall()
        # 进入上下文管理器，设置错误回调函数为 foo
        with np.errstate(call=foo):
            # 验证当前错误回调函数为 foo
            assert np.geterrcall() is foo
            # 进入嵌套上下文管理器，设置错误回调函数为 None
            with np.errstate(call=None):
                # 验证当前错误回调函数为 None
                assert np.geterrcall() is None
        # 验证退出上下文管理器后恢复原来的错误回调函数
        assert np.geterrcall() is olderrcall
        # 验证计数器值为 0
        assert count == 0

        # 进入上下文管理器，设置错误回调函数为 foo，并处理无效操作
        with np.errstate(call=foo, invalid="call"):
            np.array(np.inf) - np.array(np.inf)

        # 验证计数器值增加到 1
        assert count == 1

    # 测试错误状态的装饰器功能
    def test_errstate_decorator(self):
        # 使用 errstate 装饰器，设置所有错误都忽略
        @np.errstate(all='ignore')
        def foo():
            # 创建一个负数数组
            a = -np.arange(3)
            # 在除零操作时忽略错误
            a // 0
            
        # 调用 foo 函数
        foo()

    # 测试错误状态进入一次性
    def test_errstate_enter_once(self):
        # 创建一个错误状态对象，设置无效操作时警告
        errstate = np.errstate(invalid="warn")
        # 进入上下文管理器
        with errstate:
            pass

        # 试图第二次进入错误状态上下文管理器时，预期会引发类型错误异常
        with pytest.raises(TypeError,
                match="Cannot enter `np.errstate` twice"):
            with errstate:
                pass

    # 标记为 pytest 的测试用例，并在 WASM 环境下跳过此测试
    @pytest.mark.skipif(IS_WASM, reason="wasm doesn't support asyncio")
    def test_asyncio_safe(self):
        # 引入 asyncio 库，如果缺失则跳过测试
        # Pyodide/wasm 不支持 asyncio。如果测试出现问题，应该大量跳过或者以不同方式运行。
        asyncio = pytest.importorskip("asyncio")

        @np.errstate(invalid="ignore")
        def decorated():
            # 被装饰的非异步函数（异步函数不能被装饰）
            assert np.geterr()["invalid"] == "ignore"

        async def func1():
            decorated()
            await asyncio.sleep(0.1)
            decorated()

        async def func2():
            with np.errstate(invalid="raise"):
                assert np.geterr()["invalid"] == "raise"
                await asyncio.sleep(0.125)
                assert np.geterr()["invalid"] == "raise"

        # 又一个有趣的实验：第三个函数使用不同的错误状态
        async def func3():
            with np.errstate(invalid="print"):
                assert np.geterr()["invalid"] == "print"
                await asyncio.sleep(0.11)
                assert np.geterr()["invalid"] == "print"

        async def main():
            # 同时运行所有三个函数多次：
            await asyncio.gather(
                    func1(), func2(), func3(), func1(), func2(), func3(),
                    func1(), func2(), func3(), func1(), func2(), func3())

        # 创建一个新的事件循环
        loop = asyncio.new_event_loop()
        with np.errstate(invalid="warn"):
            # 运行主函数
            asyncio.run(main())
            assert np.geterr()["invalid"] == "warn"

        # 最终确认错误状态应该为默认的 "warn"
        assert np.geterr()["invalid"] == "warn"
        # 关闭事件循环
        loop.close()
```