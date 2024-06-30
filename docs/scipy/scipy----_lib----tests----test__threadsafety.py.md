# `D:\src\scipysrc\scipy\scipy\_lib\tests\test__threadsafety.py`

```
import threading  # 导入线程模块
import time  # 导入时间模块
import traceback  # 导入异常追溯模块

from numpy.testing import assert_  # 从 numpy.testing 模块导入 assert_ 函数
from pytest import raises as assert_raises  # 从 pytest 模块导入 raises 函数并重命名为 assert_raises
from scipy._lib._threadsafety import ReentrancyLock, non_reentrant, ReentrancyError  # 导入线程安全相关的类和异常


def test_parallel_threads():
    # 检查 ReentrancyLock 在并行线程中序列化工作。
    #
    # 此测试不完全确定，如果时间安排不当，可能会导致虚假成功。

    lock = ReentrancyLock("failure")  # 创建一个 ReentrancyLock 对象，用于线程安全

    failflag = [False]  # 创建一个标志位列表，用于标记是否发生失败
    exceptions_raised = []  # 创建一个异常列表，用于存储被捕获的异常信息

    def worker(k):
        try:
            with lock:  # 使用 ReentrancyLock 锁定代码块，确保线程安全
                assert_(not failflag[0])  # 断言标志位应为 False
                failflag[0] = True  # 将标志位设为 True
                time.sleep(0.1 * k)  # 线程休眠一定时间，模拟工作
                assert_(failflag[0])  # 断言标志位应为 True
                failflag[0] = False  # 将标志位重新设为 False，表示工作完成
        except Exception:
            exceptions_raised.append(traceback.format_exc(2))  # 捕获异常并记录到异常列表中

    threads = [threading.Thread(target=lambda k=k: worker(k))  # 创建多个线程，每个线程运行 worker 函数
               for k in range(3)]
    for t in threads:
        t.start()  # 启动所有线程
    for t in threads:
        t.join()  # 等待所有线程执行完毕

    exceptions_raised = "\n".join(exceptions_raised)  # 将异常信息列表转换为字符串
    assert_(not exceptions_raised, exceptions_raised)  # 断言异常信息应为空


def test_reentering():
    # 检查 ReentrancyLock 阻止同一线程重新进入的情况。

    @non_reentrant()  # 使用 non_reentrant 装饰器标记 func 函数为不可重入
    def func(x):
        return func(x)  # 在 func 函数中递归调用自身

    assert_raises(ReentrancyError, func, 0)  # 断言调用 func(0) 会抛出 ReentrancyError 异常
```