# `.\pytorch\test\distributed\test_collective_utils.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入需要的模块和类
from unittest import mock

# 导入 PyTorch 分布式模块
import torch.distributed as c10d
from torch.distributed.collective_utils import all_gather, broadcast
from torch.testing._internal.common_distributed import MultiProcessTestCase

# 定义一个测试类，继承自 MultiProcessTestCase
class TestCollectiveUtils(MultiProcessTestCase):
    
    # 在每个测试方法执行前调用，准备测试环境
    def setUp(self):
        super().setUp()
        self._spawn_processes()  # 启动多个进程进行测试

    # 在每个测试方法执行后调用，清理测试环境
    def tearDown(self) -> None:
        super().tearDown()

    # 定义一个返回 ProcessGroupGloo._Options 对象的方法
    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 50.0  # 设置超时时间为 50 秒
        opts._threads = threads  # 设置线程数，默认为 2
        return opts

    # 测试 broadcast 方法的基本功能
    def test_broadcast_result(self) -> None:
        """
        Basic unit test for broadcast using a process group of default world size.
        """
        # 创建一个文件存储，并初始化进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        pg = c10d.new_group(pg_options=self.opts())  # 创建一个新的进程组

        func = mock.MagicMock()
        func.return_value = pg.rank()  # 设置模拟函数返回当前进程的 rank

        res = broadcast(data_or_fn=func, rank=0, pg=pg)  # 调用 broadcast 方法进行广播
        assert res == 0, f"Expect res to be 0 (got {res})"  # 断言广播结果为 0

        if pg.rank() == 0:
            func.assert_called_once()  # 如果当前进程的 rank 是 0，断言模拟函数被调用一次
        else:
            func.assert_not_called()  # 如果当前进程的 rank 不是 0，断言模拟函数未被调用

        func.reset_mock()  # 重置模拟函数的调用记录

        res = broadcast(data_or_fn=func, rank=1, pg=pg)  # 再次调用 broadcast 方法进行广播
        assert res == 1, f"Expect res to be 1 (got {res})"  # 断言广播结果为 1

        if pg.rank() == 1:
            func.assert_called_once()  # 如果当前进程的 rank 是 1，断言模拟函数被调用一次
        else:
            func.assert_not_called()  # 如果当前进程的 rank 不是 1，断言模拟函数未被调用

    # 测试 broadcast 方法在没有进程组情况下的行为
    def test_broadcast_result_no_pg(self) -> None:
        """
        Ensure broadcast has no dependency on torch.distributed when run in single process.
        """
        func = mock.MagicMock()
        res = broadcast(data_or_fn=func, rank=0)  # 在单进程运行时调用 broadcast 方法
        func.assert_called_once()  # 断言模拟函数被调用一次

    # 测试 broadcast 方法在函数抛出异常时的行为
    def test_broadcast_result_raises_exceptions_from_func(self) -> None:
        """
        Ensure broadcast exception is propagated properly.
        """
        # 模拟一个函数抛出异常的情况
        func = mock.MagicMock()
        exc = Exception("test exception")
        func.side_effect = exc
        expected_exception = "test exception"
        with self.assertRaisesRegex(Exception, expected_exception):
            broadcast(data_or_fn=func, rank=0)  # 调用 broadcast 方法，断言异常被正确传播

    # 测试 all_gather 方法的基本功能
    def test_all_gather_result(self) -> None:
        """
        Basic unit test for all_gather using a process group of default world size.
        """
        # 创建一个文件存储，并初始化进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        pg = c10d.new_group(pg_options=self.opts())  # 创建一个新的进程组

        func = mock.MagicMock()
        func.return_value = pg.rank()  # 设置模拟函数返回当前进程的 rank

        res = all_gather(data_or_fn=func, pg=pg)  # 调用 all_gather 方法进行聚合
        func.assert_called_once()  # 断言模拟函数被调用一次
        assert res == list(range(self.world_size)), \
            f"Expect res to be list of 0 through {self.world_size} (got {res})"  # 断言聚合结果正确
    # 定义单元测试函数 test_all_gather_result_no_pg，测试 all_gather 在单进程环境下不依赖于 torch.distributed。
    def test_all_gather_result_no_pg(self) -> None:
        """
        Ensure all_gather has no dependency on torch.distributed when run in single process.
        """
        # 创建一个 MagicMock 对象 func
        func = mock.MagicMock()
        # 调用 all_gather 函数，并将结果存储在 res 中
        res = all_gather(data_or_fn=func)
        # 断言 func 被调用了一次
        func.assert_called_once()

    # 定义单元测试函数 test_all_gather_result_raises_exceptions_from_func，测试 all_gather 在函数抛出异常时是否正确传播异常。
    def test_all_gather_result_raises_exceptions_from_func(
        self,
    ) -> None:
        """
        Ensure all_gather exception is propagated properly.
        """
        # 没有进程组
        # 创建一个 MagicMock 对象 func
        func = mock.MagicMock()
        # 创建一个异常对象
        exc = Exception("test exception")
        # 设置 func 调用时抛出异常
        func.side_effect = exc
        # 预期的异常消息
        expected_exception = "test exception"
        # 使用 assertRaisesRegex 断言函数调用时抛出特定异常
        with self.assertRaisesRegex(Exception, expected_exception):
            # 调用 all_gather 函数，并传入 func 作为参数
            all_gather(data_or_fn=func)
```