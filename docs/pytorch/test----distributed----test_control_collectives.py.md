# `.\pytorch\test\distributed\test_control_collectives.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入所需模块和类
from datetime import timedelta
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


# 定义一个简单的用户函数，使用控制集合体类对象执行多个不同的集合操作
def simple_user_func(collectives: dist._ControlCollectives, rank: int) -> int:
    # 设置超时时间为10秒
    timeout = timedelta(seconds=10)
    # 执行一个屏障操作
    collectives.barrier("1", timeout, True)
    # 然后执行一个全局求和操作
    out = collectives.all_sum("2", rank, timeout)
    return out


class TestCollectives(TestCase):
    # 测试屏障操作
    def test_barrier(self) -> None:
        store = dist.HashStore()

        world_size = 2

        def f(rank: int) -> None:
            # 创建存储集合体对象
            collectives = dist._StoreCollectives(store, rank, world_size)
            # 执行一个屏障操作
            collectives.barrier("foo", timedelta(seconds=10), True)

        # 使用线程池并行执行测试函数
        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    # 测试广播操作
    def test_broadcast(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            # 创建存储集合体对象
            collectives = dist._StoreCollectives(store, rank, world_size)
            # 如果是特定的进程，执行广播发送
            if rank == 2:
                collectives.broadcast_send("foo", b"data", timeout)
            else:
                # 否则执行广播接收，并断言接收到的数据
                out = collectives.broadcast_recv("foo", timeout)
                self.assertEqual(out, b"data")

        # 使用线程池并行执行测试函数
        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    # 测试聚集操作
    def test_gather(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            # 创建存储集合体对象
            collectives = dist._StoreCollectives(store, rank, world_size)
            # 如果是特定的进程，执行聚集接收，并断言接收到的数据
            if rank == 2:
                out = collectives.gather_recv("foo", str(rank), timeout)
                self.assertEqual(out, [b"0", b"1", b"2", b"3"])
            else:
                # 否则执行聚集发送
                collectives.gather_send("foo", str(rank), timeout)

        # 使用线程池并行执行测试函数
        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    # 测试分散操作
    def test_scatter(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            # 创建存储集合体对象
            collectives = dist._StoreCollectives(store, rank, world_size)
            # 如果是特定的进程，执行分散发送
            if rank == 2:
                out = collectives.scatter_send(
                    "foo", [str(i) for i in range(world_size)], timeout
                )
            else:
                # 否则执行分散接收，并断言接收到的数据
                out = collectives.scatter_recv("foo", timeout)
            self.assertEqual(out, str(rank).encode())

        # 使用线程池并行执行测试函数
        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))
    # 测试所有进程的求和操作，使用了 HashStore 存储对象
    def test_all_sum(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 10 秒钟
        timeout = timedelta(seconds=10)

        # 定义内部函数 f，参数为进程的排名
        def f(rank: int) -> None:
            # 创建 StoreCollectives 对象，用于集体操作
            collectives = dist._StoreCollectives(store, rank, world_size)
            # 执行 all_sum 操作，对键 "foo" 进行求和，带有排名和超时时间参数
            out = collectives.all_sum("foo", rank, timeout)
            # 断言输出结果等于所有进程排名的总和
            self.assertEqual(out, sum(range(world_size)))

        # 使用线程池创建多线程环境，每个线程调用函数 f 处理不同的进程排名
        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    # 测试广播接收操作的超时情况
    def test_broadcast_timeout(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 1 毫秒
        timeout = timedelta(milliseconds=1)
        # 创建 StoreCollectives 对象，用于集体操作，rank 为 1
        collectives = dist._StoreCollectives(store, 1, world_size)
        # 使用断言捕获异常，验证广播接收是否会在超时时抛出 "Wait timeout" 异常
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.broadcast_recv("foo", timeout)

    # 测试聚集接收操作的超时情况
    def test_gather_timeout(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 1 毫秒
        timeout = timedelta(milliseconds=1)
        # 创建 StoreCollectives 对象，用于集体操作，rank 为 1
        collectives = dist._StoreCollectives(store, 1, world_size)
        # 使用断言捕获异常，验证聚集接收是否会在超时时抛出 "gather failed -- missing ranks: 0, 2, 3" 异常
        with self.assertRaisesRegex(
            Exception, "gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.gather_recv("foo", "data", timeout)

    # 测试分散接收操作的超时情况
    def test_scatter_timeout(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 1 毫秒
        timeout = timedelta(milliseconds=1)
        # 创建 StoreCollectives 对象，用于集体操作，rank 为 1
        collectives = dist._StoreCollectives(store, 1, world_size)
        # 使用断言捕获异常，验证分散接收是否会在超时时抛出 "Wait timeout" 异常
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.scatter_recv("foo", timeout)

    # 测试全体聚集操作的超时情况
    def test_all_gather_timeout(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 1 毫秒
        timeout = timedelta(milliseconds=1)
        # 创建 StoreCollectives 对象，用于集体操作，rank 为 1
        collectives = dist._StoreCollectives(store, 1, world_size)
        # 使用断言捕获异常，验证全体聚集操作是否会在超时时抛出 "all_gather failed -- missing ranks: 0, 2, 3" 异常
        with self.assertRaisesRegex(
            Exception, "all_gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_gather("foo", "data", timeout)

    # 测试屏障同步操作的超时情况
    def test_barrier_timeout(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 1 毫秒
        timeout = timedelta(milliseconds=1)
        # 创建 StoreCollectives 对象，用于集体操作，rank 为 1
        collectives = dist._StoreCollectives(store, 1, world_size)
        # 使用断言捕获异常，验证屏障同步操作是否会在超时时抛出 "barrier failed -- missing ranks: 0, 2, 3" 异常
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.barrier("foo", timeout, True)

    # 测试全体求和操作的超时情况
    def test_all_sum_timeout(self) -> None:
        store = dist.HashStore()

        # 定义世界大小为 4
        world_size = 4
        # 定义超时时间为 1 毫秒
        timeout = timedelta(milliseconds=1)
        # 创建 StoreCollectives 对象，用于集体操作，rank 为 1
        collectives = dist._StoreCollectives(store, 1, world_size)
        # 使用断言捕获异常，验证全体求和操作是否会在超时时抛出 "barrier failed -- missing ranks: 0, 2, 3" 异常
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_sum("foo", 1, timeout)
    # 定义一个测试方法，用于测试 HashStore 类的功能
    def test_unique(self) -> None:
        # 创建一个 HashStore 对象作为存储
        store = dist.HashStore()

        # 使用 HashStore 创建一个 _StoreCollectives 实例，rank 和 world_size 分别为 1 和 1
        collectives = dist._StoreCollectives(store, 1, 1)
        # 对 collectives 执行广播发送操作，将键 "foo" 和值 "bar" 广播给所有节点
        collectives.broadcast_send("foo", "bar")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 再次尝试使用相同的键 "foo" 进行广播发送，预期会抛出异常
            collectives.broadcast_send("foo", "bar")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行广播接收，预期会抛出异常
            collectives.broadcast_recv("foo")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行 gather 发送，预期会抛出异常
            collectives.gather_send("foo", "bar")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行 gather 接收，预期会抛出异常
            collectives.gather_recv("foo", "asdf")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行 scatter 发送，预期会抛出异常
            collectives.scatter_send("foo", ["asdf"])

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行 scatter 接收，预期会抛出异常
            collectives.scatter_recv("foo")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行 all_gather 操作，预期会抛出异常
            collectives.all_gather("foo", "bar")

        # 使用断言检测是否抛出指定异常，异常信息为 "Key foo has already been used"
        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            # 尝试使用已经存在的键 "foo" 进行 all_sum 操作，预期会抛出异常
            collectives.all_sum("foo", 2)

    # 定义一个测试简单用户函数的方法
    def test_simple_user_func(self) -> None:
        # 创建一个 HashStore 对象作为存储
        store = dist.HashStore()
        # 定义世界大小为 4
        world_size = 4

        # 定义一个函数 f，接收一个参数 rank，返回 None
        def f(rank: int) -> None:
            # 用户需要创建子 collectives
            # 但是对于不同的子 collectives，simple_user_func 不需要改变
            # 使用 HashStore 创建一个 _StoreCollectives 实例，使用 rank 和 world_size
            store_collectives = dist._StoreCollectives(store, rank, world_size)
            # 调用 simple_user_func 函数，并传入 store_collectives 和 rank 作为参数
            out = simple_user_func(store_collectives, rank)
            # 使用断言检查 out 是否等于 0 到 (world_size-1) 的和
            self.assertEqual(out, sum(range(world_size)))

        # 使用 ThreadPool 创建一个线程池，线程数为 world_size
        with ThreadPool(world_size) as pool:
            # 使用线程池并发地执行函数 f，参数为 range(world_size)，即 [0, 1, 2, 3]
            pool.map(f, range(world_size))
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中执行），则执行以下代码块
if __name__ == "__main__":
    # 使用断言确保 CUDA 上下文在主进程中未初始化，以避免影响测试的分布式执行
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    # 运行测试函数
    run_tests()
```