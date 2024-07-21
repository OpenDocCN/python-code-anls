# `.\pytorch\test\distributed\test_multi_threaded_pg.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import operator
import os
import sys
import threading
from functools import reduce
from unittest import skip, SkipTest

import torch
import torch.autograd
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

# 如果分布式环境不可用，输出错误信息并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试所需的其他模块和函数
from torch.testing._internal.common_distributed import (
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
    spawn_threads_and_init_comms,
)
from torch.testing._internal.common_utils import IS_SANDCASTLE, run_tests, TestCase

# 默认的世界大小设定为4
DEFAULT_WORLD_SIZE = 4

# 定义测试类 TestCollectivesWithWrapper，继承自 TestCase 类
class TestCollectivesWithWrapper(TestCase):
    
    # 使用 spawn_threads_and_init_comms 装饰器创建多线程并初始化通信
    @spawn_threads_and_init_comms(world_size=4)
    # 定义测试方法 test_broadcast_object_list
    def test_broadcast_object_list(self):
        # 如果当前进程的排名为0，则 val 为 99，否则为 None
        val = 99 if dist.get_rank() == 0 else None
        # 创建对象列表，长度为世界大小，初始化为 [99, None, None, None]
        object_list = [val] * dist.get_world_size()

        # 执行广播操作，将 object_list 广播到所有进程
        dist.broadcast_object_list(object_list=object_list)
        # 断言 object_list 的第一个元素为 99
        self.assertEqual(99, object_list[0])

    # 定义测试方法 test_collective_error_on_rank_zero
    def test_collective_error_on_rank_zero(self):
        # 使用 spawn_threads_and_init_comms 装饰器创建多线程并初始化通信
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            # 创建输入张量，其值为当前进程的排名
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            # 创建输出张量列表，长度为世界大小，每个元素为与 input_tensor 相同形状的空张量
            output_tensors = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            # 执行全局收集操作，将 input_tensor 收集到 output_tensors 中
            dist.all_gather(output_tensors, input_tensor)

            # 如果当前进程的排名为0，抛出 AssertionError
            if dist.get_rank() == 0:
                raise AssertionError("Mimic real test failure.")  # fail on rank 0

            # 再次执行全局收集操作，将 input_tensor 再次收集到 output_tensors 中
            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        # 断言运行时错误，确保捕获到 AssertionError
        with self.assertRaises(RuntimeError):
            _test_method(self)

    # 定义测试方法 test_collective_error_on_rank_non_zero
    def test_collective_error_on_rank_non_zero(self):
        # 使用 spawn_threads_and_init_comms 装饰器创建多线程并初始化通信
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            # 创建输入张量，其值为当前进程的排名
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            # 创建输出张量列表，长度为世界大小，每个元素为与 input_tensor 相同形状的空张量
            output_tensors = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            # 执行全局收集操作，将 input_tensor 收集到 output_tensors 中
            dist.all_gather(output_tensors, input_tensor)

            # 如果当前进程的排名为1，抛出 AssertionError
            if dist.get_rank() == 1:
                raise AssertionError("Mimic real test failure.")  # fail on rank 1

            # 再次执行全局收集操作，将 input_tensor 再次收集到 output_tensors 中
            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        # 断言运行时错误，确保捕获到 AssertionError
        with self.assertRaises(RuntimeError):
            _test_method(self)
    # 定义一个测试函数，用于测试在非零排名时的集体错误处理
    def test_collective_error_on_rank_non_zero_all(self):
        # 使用装饰器启动线程并初始化通信，设置世界大小为4
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            # 创建一个填充值为当前进程排名的 3x3 的张量，用于第一次全体聚集操作
            input_tensor = torch.ones(3, 3) * dist.get_rank()
            # 创建一个空张量列表，每个张量与进程数相同
            output_tensors = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            # 执行全体聚集操作，将 input_tensor 的副本聚集到 output_tensors 中
            dist.all_gather(output_tensors, input_tensor)

            # 如果进程排名大于0，则抛出断言错误，模拟真实测试失败
            if dist.get_rank() > 0:
                raise AssertionError(
                    "Mimic real test failure."
                )
                # 在所有非零排名上失败

            # 再次执行全体聚集操作，将 input_tensor 的副本聚集到 output_tensors 中
            dist.all_gather(output_tensors, input_tensor)

        # 使用断言检查是否抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            _test_method(self)

    # 定义一个测试函数，用于测试跳过功能是否可以正确捕获异常
    def test_skip(self):
        # 使用装饰器启动线程并初始化通信，设置世界大小为4
        @spawn_threads_and_init_comms(world_size=4)
        @skip("check if skip exception can be captured correctly.")
        def _test_method(self):
            pass

        # 如果不是在 SANCASTLE 环境下
        if not IS_SANDCASTLE:
            # 使用断言检查是否抛出 SkipTest 异常
            with self.assertRaises(SkipTest):
                _test_method(self)

    # 使用装饰器启动线程并初始化通信，设置世界大小为4
    @spawn_threads_and_init_comms(world_size=4)
    # 定义一个测试函数，用于测试 all_to_all_single 操作处理单个张量的情况
    def test_all_to_all_single_tensor(self):
        # 获取当前进程的排名
        rank = dist.get_rank()
        # 获取通信世界的大小
        world_size = dist.get_world_size()
        # 创建一个填充值为当前排名的张量，形状为 (world_size, 2)
        send = torch.full((world_size, 2), rank)
        # 创建一个包含全1的大小张量列表，长度为通信世界的大小
        sizes = torch.ones(world_size, dtype=torch.int64)

        # 创建一个与 send 形状相同的零张量 out
        out = torch.zeros(world_size, 2, dtype=send.dtype)
        # 执行全体到全体单个张量的通信操作，使用 send 和 sizes 数组
        dist.all_to_all_single(out, send, sizes, sizes)
        # 使用断言检查 out 是否与预期值相等
        self.assertEqual(out.tolist(), list(zip(range(world_size), range(world_size))))

    # 使用装饰器启动线程并初始化通信，设置世界大小为4
    @spawn_threads_and_init_comms(world_size=4)
    # 定义一个测试函数，用于测试 all_to_all_single 操作处理列表的情况
    def test_all_to_all_single_list(self):
        # 获取当前进程的排名
        rank = dist.get_rank()
        # 获取通信世界的大小
        world_size = dist.get_world_size()
        # 创建一个填充值为当前排名的张量，形状为 (world_size, 2)
        send = torch.full((world_size, 2), rank)
        # 创建一个全为1的大小列表，长度为通信世界的大小
        sizes = [1] * world_size

        # 创建一个与 send 形状相同的零张量 out
        out = torch.zeros(world_size, 2, dtype=send.dtype)
        # 执行全体到全体单个张量的通信操作，使用 send 和 sizes 列表
        dist.all_to_all_single(out, send, sizes, sizes)
        # 使用断言检查 out 是否与预期值相等
        self.assertEqual(out.tolist(), list(zip(range(world_size), range(world_size))))

    # 使用装饰器启动线程并初始化通信，设置世界大小为4
    @spawn_threads_and_init_comms(world_size=4)
    # 定义一个测试函数，用于测试 all_to_all_single 操作处理缺少 sizes 参数的情况
    def test_all_to_all_single_none(self):
        # 获取当前进程的排名
        rank = dist.get_rank()
        # 获取通信世界的大小
        world_size = dist.get_world_size()
        # 创建一个填充值为当前排名的张量，形状为 (world_size, 2)
        send = torch.full((world_size, 2), rank)

        # 创建一个与 send 形状相同的零张量 out
        out = torch.zeros(world_size, 2, dtype=send.dtype)
        # 执行全体到全体单个张量的通信操作，使用默认的 sizes 参数
        dist.all_to_all_single(out, send)
        # 使用断言检查 out 是否与预期值相等
        self.assertEqual(out.tolist(), list(zip(range(world_size), range(world_size))))
# 定义一个测试类，继承自 MultiThreadedTestCase，用于测试并发多线程场景
class TestCollectivesWithBaseClass(MultiThreadedTestCase):

    # 定义属性 world_size，返回并发测试的进程数为 4
    @property
    def world_size(self):
        return 4

    # 设置测试环境的准备方法
    def setUp(self):
        # 设置环境变量 TORCH_DIST_INIT_BARRIER 为 "1"，用于分布式初始化屏障
        os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
        # 调用父类的 setUp 方法，准备测试环境
        super().setUp()
        # 启动多线程
        self._spawn_threads()

    # 测试环境的清理方法
    def tearDown(self):
        # 调用父类的 tearDown 方法，清理测试环境
        super().tearDown()
        # 将环境变量 TORCH_DIST_INIT_BARRIER 设置为 "0"，表示结束分布式初始化
        os.environ["TORCH_DIST_INIT_BARRIER"] = "0"

    # 测试 all_gather 方法
    def test_allgather(self):
        # 创建一个 3x3 的张量，元素为当前进程的 rank
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        # 创建一个空的张量列表，长度为并发进程数
        output_tensors = [
            torch.empty_like(input_tensor) for _ in range(self.world_size)
        ]
        # 执行 all_gather 操作，收集所有进程的 input_tensor 到 output_tensors
        dist.all_gather(output_tensors, input_tensor)
        # 遍历 output_tensors，验证每个张量是否与对应 rank 的张量相等
        for rank, out_tensor in enumerate(output_tensors):
            self.assertEqual(out_tensor, torch.ones(3, 3) * rank)

    # 测试 broadcast 方法
    def test_broadcast(self):
        # 创建一个 3x3 的张量，元素为当前进程的 rank
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        # 遍历所有进程，对 input_tensor 进行 broadcast 操作
        for rank in range(self.world_size):
            cloned_input = input_tensor.clone()
            dist.broadcast(cloned_input, src=rank)
            # 验证广播后 cloned_input 是否与当前 rank 对应的张量相等
            self.assertEqual(cloned_input, torch.ones(3, 3) * rank)

    # 测试 scatter 方法
    def test_scatter(self):
        # 如果当前进程的 rank 是 0
        if dist.get_rank() == 0:
            # 创建一个列表，包含多个 3x3 的张量，元素为对应的 rank
            scatter_list = [torch.ones(3, 3) * rank for rank in range(self.world_size)]
        else:
            # 如果当前进程的 rank 不是 0，则 scatter_list 为 None
            scatter_list = None
        # 创建一个空的 3x3 张量
        output_tensor = torch.empty(3, 3)
        # 执行 scatter 操作，将 scatter_list 分散到 output_tensor
        dist.scatter(output_tensor, scatter_list)
        # 验证 output_tensor 是否等于当前进程的 rank 的张量
        self.assertEqual(output_tensor, torch.ones(3, 3) * dist.get_rank())

    # 测试 reduce_scatter 方法
    def test_reduce_scatter(self):
        # 创建一个列表，包含多个 3x3 的张量，元素为对应的 rank
        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(self.world_size)]
        # 创建一个空的 3x3 张量
        output_tensor = torch.empty(3, 3)
        # 执行 reduce_scatter 操作，将 to_reduce_scatter 进行归约散步到 output_tensor
        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        # 计算预期的张量，每个元素为当前进程的 rank 乘以并发进程数
        expected_tensor = torch.ones(3, 3) * dist.get_rank() * self.world_size
        # 验证 output_tensor 是否等于预期张量
        self.assertEqual(output_tensor, expected_tensor)

        # 再次创建一个空的 3x3 张量
        output_tensor = torch.empty(3, 3)
        # 执行 reduce_scatter 操作，使用 AVG 操作进行归约散步到 output_tensor
        dist.reduce_scatter(output_tensor, to_reduce_scatter, op=dist.ReduceOp.AVG)
        # 计算预期的张量，每个元素为当前进程的 rank
        expected_tensor = torch.ones(3, 3) * dist.get_rank()
        # 验证 output_tensor 是否等于预期张量
        self.assertEqual(output_tensor, expected_tensor)

    # 测试 broadcast_object_list 方法
    def test_broadcast_object_list(self):
        # 如果当前进程的 rank 是 0，val 为 99，否则为 None
        val = 99 if dist.get_rank() == 0 else None
        # 创建一个对象列表，长度为并发进程数，所有元素都是 val
        object_list = [val] * dist.get_world_size()
        # 输出当前进程的 rank 和并发进程数
        print(f"{dist.get_rank()} -> {dist.get_world_size()}")
        # 执行 broadcast_object_list 操作，广播 object_list
        dist.broadcast_object_list(object_list=object_list)
        # 验证 object_list 的第一个元素是否等于 99
        self.assertEqual(99, object_list[0])

    # 测试 all_reduce 方法
    def test_all_reduce(self):
        # 创建一个 3x3 的张量，元素为当前进程的 rank
        output = torch.ones(3, 3) * dist.get_rank()
        # 执行 all_reduce 操作，对 output 进行全局归约操作
        dist.all_reduce(output)
        # 计算预期的张量，每个元素为所有进程 rank 的和
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        # 验证 output 是否等于预期张量
        self.assertEqual(output, torch.ones(3, 3) * res_num)
    def test_all_to_all(self):
        # 获取当前进程的排名和总进程数
        rank = self.rank
        world_size = self.world_size
        # 创建输入张量列表，每个张量大小为3x3，数值为 rank * world_size 到 (rank + 1) * world_size - 1
        input_tensor_list = [
            torch.ones(3, 3) * x
            for x in range(rank * world_size, (rank + 1) * world_size)
        ]
        # 创建一个与输入张量列表相同大小的空张量列表
        output_tensor_list = [torch.empty_like(tensor) for tensor in input_tensor_list]
        # 执行分布式全局 all-to-all 通信操作
        dist.all_to_all(output_tensor_list, input_tensor_list)
        # 创建预期输出张量列表，每个张量大小为3x3，数值为 rank 到 world_size * world_size - 1，步长为 world_size
        expected_tensor_list = [
            torch.ones(3, 3) * x
            for x in range(rank, world_size * world_size, world_size)
        ]
        # 使用断言检查输出张量列表是否等于预期输出张量列表
        self.assertEqual(expected_tensor_list, output_tensor_list)

    def test_all_reduce_ops(self):
        # 创建包含当前进程排名加一的张量
        tensor = torch.tensor([dist.get_rank() + 1])
        # 执行全局 all-reduce 操作，使用 PRODUCT 运算
        dist.all_reduce(tensor, op=ReduceOp.PRODUCT)
        # 计算预期结果，即从1到总进程数的乘积
        expected = reduce(operator.mul, range(1, self.world_size + 1))
        # 使用断言检查结果张量是否等于预期值
        self.assertEqual(expected, tensor.item())

        # 以下为类似操作，依次执行 MIN、MAX、BAND、BOR、BXOR 运算的全局 all-reduce 操作，并进行断言检查结果是否符合预期

    def test_assert_equal_on_rank(self):
        # 自定义张量并克隆为 rank 0 张量
        self_tensor = torch.rand(3, 3)
        rank_0_tensor = self_tensor.clone()
        # 将 rank 0 张量广播到所有进程
        dist.broadcast(rank_0_tensor, src=0)
        # 使用自定义方法进行断言，仅在 rank 0 执行
        self.assertEqualOnRank(rank_0_tensor, self_tensor, rank=0)
        # 使用自定义方法进行断言，仅在 rank 1 执行
        self.assertNotEqualOnRank(rank_0_tensor, self_tensor, rank=1)

    def test_subpg(self):
        # 创建两个子进程组 subpg0 和 subpg1
        subpg0 = dist.new_group([0, 1])
        subpg1 = dist.new_group([2, 3])
        # 获取当前进程的排名
        current_rank = dist.get_rank()
        # 创建大小为3x3，数值为当前排名的张量
        output = torch.ones(3, 3) * current_rank

        # 如果当前排名在 [0, 1] 中，则在子进程组 subpg0 上执行全局 all-reduce 操作，否则在 subpg1 上执行
        if current_rank in [0, 1]:
            dist.all_reduce(output, group=subpg0)
        else:
            dist.all_reduce(output, group=subpg1)

        # 如果当前排名在 [0, 1] 中，则使用断言检查输出张量是否等于全1张量乘以1，否则乘以5
        if current_rank in [0, 1]:
            self.assertEqual(output, torch.ones(3, 3) * 1)
        else:
            self.assertEqual(output, torch.ones(3, 3) * 5)
    # 测试在另一个线程中使用分布式进程组进行操作
    def test_using_pg_from_another_thread(self):
        # 定义在另一个线程中执行的函数，创建一个具有梯度的张量
        def stuff_in_other_thread(pg):
            x = torch.rand(4, requires_grad=True)
            # 对张量 x 在指定的分组 pg 上进行全局归约操作

        # 创建一个线程对象，用于执行 stuff_in_other_thread 函数
        t = threading.Thread(target=stuff_in_other_thread, args=(dist.group.WORLD,))
        # 启动线程
        t.start()
        # 等待线程执行结束
        t.join()

    # 测试 gather 函数
    def test_gather(self):
        # 如果当前进程的 rank 是 0
        if dist.get_rank() == 0:
            # 创建一个空的张量列表，用于接收来自所有进程的数据
            gather_list = [torch.empty(3, 3) for _ in range(self.world_size)]
        else:
            # 否则，gather_list 设置为 None
            gather_list = None
        # 创建一个输入张量，其值为当前进程的 rank
        input_tensor = torch.ones(3, 3) * dist.get_rank()

        # 在指定的分组上进行 gather 操作，将输入张量的数据收集到 gather_list 中
        dist.gather(input_tensor, gather_list)
        # 如果当前进程的 rank 是 0
        if dist.get_rank() == 0:
            # 验证 gather 操作的结果是否符合预期
            for i in range(self.world_size):
                self.assertEqual(gather_list[i], torch.ones(3, 3) * i)

    # 测试 all_reduce_coalesced 函数
    def test_all_reduce_coalesced(self):
        # 创建一个张量 t0，其值为当前进程的 rank
        t0 = torch.ones(3, 3) * dist.get_rank()
        # 创建一个张量 t1，其值为当前进程的 rank 乘以 2
        t1 = torch.ones(3, 3) * dist.get_rank() * 2
        # 对输入的张量列表进行全局归约操作
        dist.all_reduce_coalesced([t0, t1])
        # 计算预期结果的数值
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        # 验证 t0 张量的值是否符合预期
        self.assertEqual(t0, torch.ones(3, 3) * res_num)
        # 验证 t1 张量的值是否符合预期
        self.assertEqual(t1, torch.ones(3, 3) * (res_num * 2))

    # 装饰器函数，如果 GPU 数量小于 1 则跳过该测试
    @skip_if_lt_x_gpu(1)
    # 测试反向传播是否能看到前向传播中的进程组信息
    def test_bwd_sees_fwd_pg(self):
        # 获取当前线程的标识符
        fwd_tid = threading.current_thread().ident

        # 定义一个自定义的 Torch 自动求导函数
        class MyFunc(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，接收一个 rank 参数
            def forward(ctx, rank):
                result = rank * 2

                # 保存计算过程中的结果和 rank 参数
                ctx.save_for_backward(result, rank)
                # 断言当前 rank 参数的值与当前进程的 rank 是否相等
                assert int(rank.item()) == dist.get_rank()
                return result

            @staticmethod
            # 反向传播函数，接收梯度输出 grad_output
            def backward(ctx, grad_output):
                # 恢复保存的结果和 rank 参数
                result, rank = ctx.saved_tensors
                # 获取当前线程的标识符
                bwd_tid = threading.current_thread().ident

                # 验证反向传播是否在相同的线程上运行
                self.assertEqual(
                    fwd_tid,
                    bwd_tid,
                    f"bwd not running in the same thread a fwd for rank {rank.item()}",
                )
                # 断言分布式环境已经初始化
                self.assertTrue(dist.is_initialized())
                # 断言当前 rank 参数的值与当前进程的 rank 是否相等
                self.assertEqual(int(rank.item()), dist.get_rank())
                # 对结果进行全局归约操作
                dist.all_reduce(result)
                # 验证结果的数值是否符合预期
                self.assertEqual(int(result.item()), 12)  # (0 + 1 + 2 + 3) * 2

                return grad_output * result

        # 创建一个张量 x，其值为当前进程的 rank，使用 GPU 进行计算，并要求梯度计算
        x = torch.tensor(
            [dist.get_rank()], dtype=torch.float, device="cuda", requires_grad=True
        )
        # 应用自定义的函数 MyFunc 对张量 x 进行计算
        x = MyFunc.apply(x)
        # 对计算结果进行求和并进行反向传播
        x.sum().backward()
# 如果当前脚本被作为主程序执行（而不是被导入到其他模块），则执行下面的代码
if __name__ == "__main__":
    # 调用函数运行测试函数
    run_tests()
```