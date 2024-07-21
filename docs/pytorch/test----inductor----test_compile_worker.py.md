# `.\pytorch\test\inductor\test_compile_worker.py`

```py
# Owner(s): ["module: inductor"]
# 导入操作符模块和操作系统模块
import operator
import os

# 从torch._inductor.compile_worker.subproc_pool导入相关异常处理和子进程池类
from torch._inductor.compile_worker.subproc_pool import (
    raise_testexc,
    SubprocException,
    SubprocPool,
)

# 从torch._inductor.test_case导入测试用例基类
from torch._inductor.test_case import TestCase
# 从torch.testing._internal.inductor_utils导入HAS_CPU标志
from torch.testing._internal.inductor_utils import HAS_CPU


class TestCompileWorker(TestCase):
    def test_basic_jobs(self):
        # 创建具有2个工作进程的子进程池
        pool = SubprocPool(2)
        try:
            # 向子进程池提交加法运算任务
            a = pool.submit(operator.add, 100, 1)
            # 向子进程池提交减法运算任务
            b = pool.submit(operator.sub, 100, 1)
            # 断言加法任务的结果为101
            self.assertEqual(a.result(), 101)
            # 断言减法任务的结果为99
            self.assertEqual(b.result(), 99)
        finally:
            # 关闭子进程池
            pool.shutdown()

    def test_exception(self):
        # 创建具有2个工作进程的子进程池
        pool = SubprocPool(2)
        try:
            # 向子进程池提交引发异常的任务
            a = pool.submit(raise_testexc)
            # 断言任务引发的异常符合预期的子进程异常类型
            with self.assertRaisesRegex(
                SubprocException,
                "torch._inductor.compile_worker.subproc_pool.TestException",
            ):
                a.result()
        finally:
            # 关闭子进程池
            pool.shutdown()

    def test_crash(self):
        # 创建具有2个工作进程的子进程池
        pool = SubprocPool(2)
        try:
            # 在子进程池中提交调用os._exit(1)的任务，并断言引发异常
            with self.assertRaises(Exception):
                a = pool.submit(os._exit, 1)
                a.result()

            # 即使发生了崩溃，子进程池仍然可用
            # 向子进程池提交加法任务
            b = pool.submit(operator.add, 100, 1)
            # 向子进程池提交减法任务
            c = pool.submit(operator.sub, 100, 1)
            # 断言加法任务的结果为101
            self.assertEqual(b.result(), 101)
            # 断言减法任务的结果为99
            self.assertEqual(c.result(), 99)
        finally:
            # 关闭子进程池
            pool.shutdown()


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # 如果系统有CPU，则运行测试用例
    if HAS_CPU:
        run_tests()
```