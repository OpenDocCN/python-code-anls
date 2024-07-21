# `.\pytorch\test\test_futures.py`

```py
# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]

# 导入所需模块和类
import threading  # 导入线程模块
import time  # 导入时间模块
import torch  # 导入PyTorch库
import unittest  # 导入单元测试模块
from torch.futures import Future  # 导入PyTorch中的Future类
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase, TemporaryFileName, run_tests  # 导入测试相关的实用函数和类
from typing import TypeVar  # 导入TypeVar，用于泛型类型注解

T = TypeVar("T")  # 定义泛型类型T


def add_one(fut):
    return fut.wait() + 1  # 返回Future对象的结果加1


class TestFuture(TestCase):
    def test_set_exception(self) -> None:
        # This test is to ensure errors can propagate across futures.
        # 测试确保异常可以在Future之间传播。
        error_msg = "Intentional Value Error"
        value_error = ValueError(error_msg)

        f = Future[T]()  # 创建一个泛型类型为T的Future对象
        # 设置异常
        f.set_exception(value_error)
        # 等待时应抛出异常
        with self.assertRaisesRegex(ValueError, "Intentional"):
            f.wait()

        # 也应该在访问值时抛出异常
        f = Future[T]()  # 创建另一个泛型类型为T的Future对象
        f.set_exception(value_error)
        with self.assertRaisesRegex(ValueError, "Intentional"):
            f.value()

        def cb(fut):
            fut.value()

        f = Future[T]()  # 创建另一个泛型类型为T的Future对象
        f.set_exception(value_error)

        # 在回调函数中等待时应该抛出异常
        with self.assertRaisesRegex(RuntimeError, "Got the following error"):
            cb_fut = f.then(cb)
            cb_fut.wait()

    def test_set_exception_multithreading(self) -> None:
        # Ensure errors can propagate when one thread waits on future result
        # and the other sets it with an error.
        # 确保一个线程等待Future结果时，另一个线程设置了异常。

        error_msg = "Intentional Value Error"
        value_error = ValueError(error_msg)

        def wait_future(f):
            with self.assertRaisesRegex(ValueError, "Intentional"):
                f.wait()

        f = Future[T]()  # 创建一个泛型类型为T的Future对象
        t = threading.Thread(target=wait_future, args=(f, ))
        t.start()
        f.set_exception(value_error)
        t.join()

        def cb(fut):
            fut.value()

        def then_future(f):
            fut = f.then(cb)
            with self.assertRaisesRegex(RuntimeError, "Got the following error"):
                fut.wait()

        f = Future[T]()  # 创建另一个泛型类型为T的Future对象
        t = threading.Thread(target=then_future, args=(f, ))
        t.start()
        f.set_exception(value_error)
        t.join()

    def test_done(self) -> None:
        f = Future[torch.Tensor]()  # 创建一个泛型类型为torch.Tensor的Future对象
        self.assertFalse(f.done())  # 确保Future对象未完成

        f.set_result(torch.ones(2, 2))  # 设置Future对象的结果为一个2x2的全1张量
        self.assertTrue(f.done())  # 确保Future对象已完成

    def test_done_exception(self) -> None:
        err_msg = "Intentional Value Error"

        def raise_exception(unused_future):
            raise RuntimeError(err_msg)

        f1 = Future[torch.Tensor]()  # 创建一个泛型类型为torch.Tensor的Future对象
        self.assertFalse(f1.done())  # 确保Future对象未完成
        f1.set_result(torch.ones(2, 2))  # 设置Future对象的结果为一个2x2的全1张量
        self.assertTrue(f1.done())  # 确保Future对象已完成

        f2 = f1.then(raise_exception)  # 对f1应用回调函数，返回新的Future对象f2
        self.assertTrue(f2.done())  # 确保新的Future对象f2已完成
        with self.assertRaisesRegex(RuntimeError, err_msg):
            f2.wait()  # 等待新的Future对象f2，捕获预期的运行时异常
    # 定义一个测试方法，验证在设置结果后等待 Future 返回结果是否正确
    def test_wait(self) -> None:
        # 创建一个 Future 对象，用于保存 Tensor 类型的结果
        f = Future[torch.Tensor]()
        # 设置 Future 的结果为一个 2x2 的全 1 张量
        f.set_result(torch.ones(2, 2))

        # 断言等待 Future 返回的结果是否与预期的全 1 张量相同
        self.assertEqual(f.wait(), torch.ones(2, 2))

    # 定义一个测试方法，验证多线程情况下等待 Future 返回结果的正确性
    def test_wait_multi_thread(self) -> None:

        # 定义一个函数，用于延迟设置 Future 的结果
        def slow_set_future(fut, value):
            time.sleep(0.5)  # 延迟 0.5 秒
            fut.set_result(value)

        # 创建一个 Future 对象，用于保存 Tensor 类型的结果
        f = Future[torch.Tensor]()

        # 创建一个线程，在其中调用 slow_set_future 函数设置 Future 的结果
        t = threading.Thread(target=slow_set_future, args=(f, torch.ones(2, 2)))
        t.start()

        # 断言等待 Future 返回的结果是否与预期的全 1 张量相同
        self.assertEqual(f.wait(), torch.ones(2, 2))
        t.join()  # 等待线程执行完毕

    # 定义一个测试方法，验证尝试两次标记 Future 完成时会引发 RuntimeError 异常
    def test_mark_future_twice(self) -> None:
        # 创建一个 Future 对象，用于保存 int 类型的结果
        fut = Future[int]()
        fut.set_result(1)  # 设置 Future 的结果为 1

        # 使用断言上下文检查尝试再次设置 Future 结果是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Future can only be marked completed once"
        ):
            fut.set_result(1)

    # 定义一个测试方法，验证尝试序列化 Future 对象时会引发 RuntimeError 异常
    def test_pickle_future(self):
        # 创建一个 Future 对象，用于保存 int 类型的结果
        fut = Future[int]()
        errMsg = "Can not pickle torch.futures.Future"

        # 使用临时文件名作为保存路径，检查尝试序列化 Future 对象是否会引发 RuntimeError 异常
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                torch.save(fut, fname)

    # 定义一个测试方法，验证 Future.then 方法的基本用法
    def test_then(self):
        # 创建一个 Future 对象，用于保存 Tensor 类型的结果
        fut = Future[torch.Tensor]()
        # 调用 Future.then 方法，在结果就绪后执行加 1 操作
        then_fut = fut.then(lambda x: x.wait() + 1)

        # 设置 Future 的结果为一个 2x2 的全 1 张量
        fut.set_result(torch.ones(2, 2))
        # 断言等待 Future 返回的结果是否与预期的全 1 张量相同
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # 断言等待 then_fut 返回的结果是否为预期结果加 1
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)

    # 定义一个测试方法，验证 Future.then 方法的链式调用
    def test_chained_then(self):
        # 创建一个 Future 对象，用于保存 Tensor 类型的结果
        fut = Future[torch.Tensor]()
        futs = []
        last_fut = fut
        # 进行多次链式调用 Future.then 方法
        for _ in range(20):
            last_fut = last_fut.then(add_one)
            futs.append(last_fut)

        # 设置 Future 的结果为一个 2x2 的全 1 张量
        fut.set_result(torch.ones(2, 2))

        # 逐一断言每个链式调用后返回的结果是否符合预期
        for i in range(len(futs)):
            self.assertEqual(futs[i].wait(), torch.ones(2, 2) + i + 1)

    # 定义一个测试方法，验证 Future.then 方法在错误参数下的异常情况
    def _test_then_error(self, cb, errMsg):
        # 创建一个 Future 对象，用于保存 int 类型的结果
        fut = Future[int]()
        # 调用 Future.then 方法，传入可能引发异常的回调函数
        then_fut = fut.then(cb)

        # 设置 Future 的结果为 5
        fut.set_result(5)
        # 断言等待 Future 返回的结果是否为预期的 5
        self.assertEqual(5, fut.wait())
        # 使用断言上下文检查等待 then_fut 返回结果时是否会引发预期的异常消息
        with self.assertRaisesRegex(RuntimeError, errMsg):
            then_fut.wait()

    # 定义一个测试方法，验证 Future.then 方法在错误类型的参数下引发异常
    def test_then_wrong_arg(self):

        # 定义一个错误的回调函数，试图将 Future 的 int 类型结果加 1
        def wrong_arg(tensor):
            return tensor + 1

        # 调用 _test_then_error 方法，传入错误的回调函数和预期的异常消息
        self._test_then_error(wrong_arg, "unsupported operand type.*Future.*int")

    # 定义一个测试方法，验证 Future.then 方法在没有参数情况下引发异常
    def test_then_no_arg(self):

        # 定义一个不带参数的回调函数
        def no_arg():
            return True

        # 调用 _test_then_error 方法，传入不带参数的回调函数和预期的异常消息
        self._test_then_error(no_arg, "takes 0 positional arguments but 1 was given")

    # 定义一个测试方法，验证 Future.then 方法在回调函数内部引发异常
    def test_then_raise(self):

        # 定义一个回调函数，内部引发 ValueError 异常
        def raise_value_error(fut):
            raise ValueError("Expected error")

        # 调用 _test_then_error 方法，传入引发异常的回调函数和预期的异常消息
        self._test_then_error(raise_value_error, "Expected error")

    # 定义一个测试方法，验证添加 Future 完成回调函数的基本功能
    def test_add_done_callback_simple(self):
        # 初始化回调结果为 False
        callback_result = False

        # 定义一个回调函数，设置回调结果为 True
        def callback(fut):
            nonlocal callback_result
            fut.wait()
            callback_result = True

        # 创建一个 Future 对象，用于保存 Tensor 类型的结果
        fut = Future[torch.Tensor]()
        # 添加 Future 完成时的回调函数
        fut.add_done_callback(callback)

        # 断言初始回调结果为 False
        self.assertFalse(callback_result)
        # 设置 Future 的结果为一个 2x2 的全 1 张量
        fut.set_result(torch.ones(2, 2))
        # 断言等待 Future 返回的结果是否与预期的全 1 张量相同
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # 断言回调结果为 True
        self.assertTrue(callback_result)
    # 测试确保 add_done_callback 保持回调顺序
    def test_add_done_callback_maintains_callback_order(self):
        # 初始化回调结果为 0
        callback_result = 0

        # 定义第一个设置回调函数
        def callback_set1(fut):
            nonlocal callback_result
            # 等待 Future 完成
            fut.wait()
            # 设置回调结果为 1
            callback_result = 1

        # 定义第二个设置回调函数
        def callback_set2(fut):
            nonlocal callback_result
            # 等待 Future 完成
            fut.wait()
            # 设置回调结果为 2
            callback_result = 2

        # 创建一个 Future 对象，其结果为 torch.Tensor 类型
        fut = Future[torch.Tensor]()
        # 向 Future 对象添加第一个回调函数
        fut.add_done_callback(callback_set1)
        # 向 Future 对象添加第二个回调函数
        fut.add_done_callback(callback_set2)

        # 设置 Future 的结果为 2x2 的全 1 矩阵
        fut.set_result(torch.ones(2, 2))
        # 等待 Future 完成，并断言结果为全 1 矩阵
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # 断言 callback_result 的值为 2，表示最后一个设置的回调函数被调用
        self.assertEqual(callback_result, 2)

    # 测试确保错误在 add_done_callback 中被忽略
    def _test_add_done_callback_error_ignored(self, cb):
        # 创建一个 Future 对象，其结果为 int 类型
        fut = Future[int]()
        # 向 Future 对象添加回调函数 cb
        fut.add_done_callback(cb)

        # 设置 Future 的结果为 5
        fut.set_result(5)
        # 断言 Future 的等待结果为 5
        self.assertEqual(5, fut.wait())

    # 测试确保错误在 add_done_callback 中被忽略
    def test_add_done_callback_error_is_ignored(self):

        # 抛出 ValueError 的回调函数
        def raise_value_error(fut):
            raise ValueError("Expected error")

        # 调用 _test_add_done_callback_error_ignored 方法，传入抛出 ValueError 的回调函数
        self._test_add_done_callback_error_ignored(raise_value_error)

    # 测试确保错误在 add_done_callback 中被忽略（无参数版本）
    def test_add_done_callback_no_arg_error_is_ignored(self):

        # 无参数的回调函数，返回 True
        def no_arg():
            return True

        # 添加额外的函数层级，以避免 mypy 对 no_arg 的类型不兼容导致 CI 失败
        # 调用 _test_add_done_callback_error_ignored 方法，传入无参数的回调函数
        self._test_add_done_callback_error_ignored(no_arg)

    # 测试确保交错使用 then 和 add_done_callback 保持回调顺序
    def test_interleaving_then_and_add_done_callback_maintains_callback_order(self):
        # 初始化回调结果为 0
        callback_result = 0

        # 定义第一个设置回调函数
        def callback_set1(fut):
            nonlocal callback_result
            # 等待 Future 完成
            fut.wait()
            # 设置回调结果为 1
            callback_result = 1

        # 定义第二个设置回调函数
        def callback_set2(fut):
            nonlocal callback_result
            # 等待 Future 完成
            fut.wait()
            # 设置回调结果为 2
            callback_result = 2

        # 定义 then 的回调函数
        def callback_then(fut):
            nonlocal callback_result
            # 等待 Future 完成，并返回结果加上 callback_result
            return fut.wait() + callback_result

        # 创建一个 Future 对象，其结果为 torch.Tensor 类型
        fut = Future[torch.Tensor]()
        # 向 Future 对象添加第一个回调函数
        fut.add_done_callback(callback_set1)
        # 调用 then 方法创建 then_fut 对象，并传入 then 的回调函数
        then_fut = fut.then(callback_then)
        # 向 Future 对象添加第二个回调函数
        fut.add_done_callback(callback_set2)

        # 设置 Future 的结果为 2x2 的全 1 矩阵
        fut.set_result(torch.ones(2, 2))
        # 等待 Future 完成，并断言结果为全 1 矩阵
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # 断言 then_fut 的等待结果为全 1 矩阵加上 1，表示 then 的回调函数被调用
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)
        # 断言 callback_result 的值为 2，表示最后一个设置的回调函数被调用
        self.assertEqual(callback_result, 2)

    # 测试确保交错使用 then 和 add_done_callback 传播错误
    def test_interleaving_then_and_add_done_callback_propagates_error(self):
        # 抛出 ValueError 的回调函数
        def raise_value_error(fut):
            raise ValueError("Expected error")

        # 创建一个 Future 对象，其结果为 torch.Tensor 类型
        fut = Future[torch.Tensor]()
        # 调用 then 方法创建 then_fut 对象，并传入抛出 ValueError 的回调函数
        then_fut = fut.then(raise_value_error)
        # 向 Future 对象添加抛出 ValueError 的回调函数
        fut.add_done_callback(raise_value_error)
        # 设置 Future 的结果为 2x2 的全 1 矩阵
        fut.set_result(torch.ones(2, 2))

        # 错误来自 add_done_callback 的回调函数被吞噬
        # 错误来自 then 的回调函数不会被吞噬，会抛出 RuntimeError 异常，异常信息为 "Expected error"
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            then_fut.wait()
    def test_collect_all(self):
        # 创建两个 Future 对象，用于存储异步操作的结果
        fut1 = Future[int]()
        fut2 = Future[int]()
        # 使用 torch.futures.collect_all() 方法收集所有 Future 对象的结果
        fut_all = torch.futures.collect_all([fut1, fut2])

        # 定义一个在线程中执行的函数，模拟耗时操作
        def slow_in_thread(fut, value):
            time.sleep(0.1)
            fut.set_result(value)

        # 创建一个线程，执行 slow_in_thread 函数
        t = threading.Thread(target=slow_in_thread, args=(fut1, 1))
        # 设置 fut2 的结果为 2
        fut2.set_result(2)
        # 启动线程
        t.start()

        # 等待所有 Future 对象的结果
        res = fut_all.wait()
        # 断言 fut1 的结果为 1
        self.assertEqual(res[0].wait(), 1)
        # 断言 fut2 的结果为 2
        self.assertEqual(res[1].wait(), 2)
        # 等待线程结束
        t.join()

    @unittest.skipIf(IS_WINDOWS, "TODO: need to fix this testcase for Windows")
    def test_wait_all(self):
        fut1 = Future[int]()
        fut2 = Future[int]()

        # 设置 fut1 和 fut2 的结果，无异常版本
        fut1.set_result(1)
        fut2.set_result(2)
        # 等待所有 Future 对象的结果
        res = torch.futures.wait_all([fut1, fut2])
        print(res)
        # 断言结果为 [1, 2]
        self.assertEqual(res, [1, 2])

        # 包含异常的版本
        def raise_in_fut(fut):
            raise ValueError("Expected error")
        # 对 fut1 应用 then 方法，引发异常
        fut3 = fut1.then(raise_in_fut)
        # 断言在 wait_all 中抛出 RuntimeError 异常，异常信息包含 "Expected error"
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            torch.futures.wait_all([fut3, fut2])

    def test_wait_none(self):
        fut1 = Future[int]()
        # 断言在 wait 方法中抛出 RuntimeError 异常，异常信息包含 "Future can't be None"
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            torch.jit.wait(None)
        # 断言在 wait_all 方法中抛出 RuntimeError 异常，异常信息包含 "Future can't be None"
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            torch.futures.wait_all((None,))  # type: ignore[arg-type]
        # 断言在 collect_all 方法中抛出 RuntimeError 异常，异常信息包含 "Future can't be None"
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            torch.futures.collect_all((fut1, None,))  # type: ignore[arg-type]
# 如果这个脚本是作为主程序运行（而不是被导入到其他脚本中），则执行 run_tests() 函数
if __name__ == '__main__':
    run_tests()
```