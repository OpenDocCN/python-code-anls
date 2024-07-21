# `.\pytorch\test\test_cuda_sanitizer.py`

```
# Owner(s): ["module: cuda"]

# 导入必要的模块和类
import sys
import textwrap
import traceback
from typing import List

# 导入 PyTorch 相关模块
import torch
import torch.cuda._sanitizer as csan
from torch.cuda._sanitizer import DataPtr, EventId, StreamId
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase

# 如果未开启 CUDA 测试，则输出提示信息并将 TestCase 设置为 NoTest
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

# 定义测试类 TestArgumentHandler，继承自 TestCase
class TestArgumentHandler(TestCase):
    
    # 定义测试函数 test_add
    def test_add(self):
        # 获取 torch.add 操作的函数句柄
        add_func = torch.ops.aten.add.Tensor
        # 创建两个在 CUDA 设备上的 Tensor 对象
        a = torch.ones(5, 3, device="cuda")
        b = torch.randn(5, 3, device="cuda")
        
        # 创建 ArgumentHandler 对象，用于处理函数参数
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数并记录数据指针
        argument_handler.parse_inputs(add_func._schema, (a, b), {})
        # 执行 torch.add 操作
        c = torch.add(a, b)
        # 解析输出参数并记录数据指针
        argument_handler.parse_outputs(c)
        
        # 断言读取的数据指针集合是否正确
        self.assertEqual({a.data_ptr(), b.data_ptr()}, argument_handler.dataptrs_read)
        # 断言写入的数据指针集合是否正确
        self.assertEqual({c.data_ptr()}, argument_handler.dataptrs_written)
    
    # 定义测试函数 test_cat
    def test_cat(self):
        # 获取 torch.cat 操作的函数句柄
        cat_func = torch.ops.aten.cat.default
        # 创建三个在 CUDA 设备上的 Tensor 对象
        a = torch.ones(2, 4, 5, device="cuda")
        b = torch.zeros(2, 1, 5, device="cuda")
        c = torch.rand(2, 7, 5, device="cuda")
        
        # 创建 ArgumentHandler 对象，用于处理函数参数
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数并记录数据指针
        argument_handler.parse_inputs(cat_func._schema, ([a, b, c], 1), {})
        # 执行 torch.cat 操作
        d = torch.cat((a, b, c), dim=1)
        # 解析输出参数并记录数据指针
        argument_handler.parse_outputs(d)
        
        # 断言读取的数据指针集合是否正确
        self.assertEqual(
            {a.data_ptr(), b.data_ptr(), c.data_ptr()}, argument_handler.dataptrs_read
        )
        # 断言写入的数据指针集合是否正确
        self.assertEqual({d.data_ptr()}, argument_handler.dataptrs_written)
    
    # 定义测试函数 test_split
    def test_split(self):
        # 获取 torch.split 操作的函数句柄
        split_func = torch.ops.aten.split.Tensor
        # 创建一个在 CUDA 设备上的 Tensor 对象
        a = torch.arange(10, device="cuda").reshape(5, 2)
        
        # 创建 ArgumentHandler 对象，用于处理函数参数
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数并记录数据指针
        argument_handler.parse_inputs(split_func._schema, (a, 2), {})
        # 执行 torch.split 操作
        out = torch.split(a, 2)
        # 解析输出参数并记录数据指针
        argument_handler.parse_outputs(out)
        
        # 计算输出 Tensor 的数据指针集合
        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        # 断言读取的数据指针集合是否正确
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        # 断言写入的数据指针集合是否正确
        self.assertEqual(
            outputs,
            argument_handler.dataptrs_written,
        )
    
    # 定义测试函数 test_inplace
    def test_inplace(self):
        # 获取 torch.add_ 操作的函数句柄
        add_inplace_func = torch.ops.aten.add_.Tensor
        # 创建一个在 CUDA 设备上的 Tensor 对象
        a = torch.rand(4, 2, device="cuda")
        
        # 创建 ArgumentHandler 对象，用于处理函数参数
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数并记录数据指针
        argument_handler.parse_inputs(add_inplace_func._schema, (a, 5), {})
        # 执行 torch.add_ 操作（原地加法）
        a.add_(5)
        # 解析输出参数并记录数据指针
        argument_handler.parse_outputs(a)
        
        # 断言读取的数据指针集合是否为空
        self.assertEqual(set(), argument_handler.dataptrs_read)
        # 断言写入的数据指针集合是否正确
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_written)
    # 定义测试方法 `test_out`，用于测试 torch 操作的输出参数功能
    def test_out(self):
        # 获取 torch 的 `mul.out` 操作函数
        mul_out_func = torch.ops.aten.mul.out
        # 在 GPU 上创建一个包含 0 到 7 的张量 `a`
        a = torch.arange(8, device="cuda")
        # 在 GPU 上创建一个空张量 `b`，用于存储输出结果
        b = torch.empty(8, device="cuda")

        # 创建参数处理器实例
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数，使用 `mul_out_func` 的模式和参数 `(a, 3)`，输出写入到 `b` 中
        argument_handler.parse_inputs(mul_out_func._schema, (a, 3), {"out": b})
        # 执行 torch 的 `mul` 操作，将 `a` 乘以 3，结果存储到 `b`
        torch.mul(a, 3, out=b)
        # 解析输出结果，记录输出数据指针
        argument_handler.parse_outputs(b)

        # 断言数据读取的指针集合与 `a` 的数据指针一致
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        # 断言数据写入的指针集合与 `b` 的数据指针一致
        self.assertEqual({b.data_ptr()}, argument_handler.dataptrs_written)

    # 定义测试方法 `test_nonzero`，用于测试 torch 的 `nonzero` 操作
    def test_nonzero(self):
        # 获取 torch 的 `aten.nonzero.default` 操作函数
        nonzero_func = torch.ops.aten.nonzero.default
        # 在 GPU 上创建一个全为 1 的张量 `a`
        a = torch.ones(5, 3, 2, device="cuda")

        # 创建参数处理器实例
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数，使用 `nonzero_func` 的模式和参数 `(a,)`，选择输出为元组形式
        argument_handler.parse_inputs(nonzero_func._schema, (a,), {"as_tuple": True})
        # 执行 torch 的 `nonzero` 操作，找到张量 `a` 中非零元素的索引，输出为元组 `out`
        out = torch.nonzero(a, as_tuple=True)
        # 解析输出结果，记录输出数据指针
        argument_handler.parse_outputs(out)

        # 构建输出数据指针集合
        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        # 断言数据读取的指针集合与 `a` 的数据指针一致
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        # 断言数据写入的指针集合与 `out` 中元素的数据指针一致
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    # 定义测试方法 `test_tensor_names`，用于测试 torch 的 `addr` 操作及参数处理
    def test_tensor_names(self):
        # 获取 torch 的 `aten.addr.default` 操作函数
        addr_func = torch.ops.aten.addr.default
        # 在 GPU 上创建一个向量 `vec`，包含元素 1 到 3
        vec = torch.arange(1, 4, device="cuda")
        # 在 GPU 上创建一个全零的 3x3 矩阵 `M`
        M = torch.zeros(3, 3, device="cuda")

        # 创建参数处理器实例
        argument_handler = csan.ArgumentHandler()
        # 解析输入参数，使用 `addr_func` 的模式和参数 `(M, vec, vec)`，无额外选项
        argument_handler.parse_inputs(addr_func._schema, (M, vec, vec), {})
        # 执行 torch 的 `addr` 操作，计算 `M + vec * vec^T` 的结果 `out`
        out = torch.addr(M, vec, vec)
        # 解析输出结果，记录输出数据指针
        argument_handler.parse_outputs(out)

        # 断言输出结果中的张量别名字典与预期的匹配
        self.assertEqual(
            argument_handler.tensor_aliases,
            {
                M.data_ptr(): ["self"],
                vec.data_ptr(): ["vec1", "vec2"],
                out.data_ptr(): [],
            },
        )
        # 断言数据写入的指针集合与 `out` 的数据指针一致
        self.assertEqual({out.data_ptr()}, argument_handler.outputs)
# 定义一个函数，将整数转换为数据指针类型并返回
def tensor_id(i: int) -> DataPtr:
    return i

# 定义一个函数，将整数转换为流ID类型并返回
def stream_id(i: int) -> StreamId:
    return 1000 + i

# 定义一个函数，将整数转换为事件ID类型并返回
def event_id(i: int) -> EventId:
    return 2000 + i

# 定义一个测试类 TestEventHandler，继承自 TestCase
class TestEventHandler(TestCase):

    # 在每个测试方法运行之前执行的初始化方法
    def setUp(self):
        self.handler = csan.EventHandler()

    # 封装了处理内核启动的方法
    def kernel_launch(
        self,
        stream: StreamId,
        read_only: List[DataPtr] = None,
        read_write: List[DataPtr] = None,
    ) -> List[csan.SynchronizationError]:
        # 如果 read_only 参数为 None，则设为一个空列表
        if read_only is None:
            read_only = []
        # 如果 read_write 参数为 None，则设为一个空列表
        if read_write is None:
            read_write = []
        # 调用 EventHandler 对象的 _handle_kernel_launch 方法处理内核启动，并返回结果
        return self.handler._handle_kernel_launch(
            stream,
            read_only,
            read_write,
            {},
            "",
            {k: [""] for k in read_only + read_write},
        )

    # 断言内核启动操作正常执行
    def assert_good_kernel_launch(
        self,
        stream: StreamId,
        read_only: List[DataPtr] = None,
        read_write: List[DataPtr] = None,
    ) -> None:
        # 断言调用 kernel_launch 方法返回空列表，表示无错误
        self.assertEqual(self.kernel_launch(stream, read_only, read_write), [])

    # 断言内核启动操作存在指定数量的错误
    def assert_bad_kernel_launch(
        self,
        number_of_errors: int,
        stream: StreamId,
        read_only: List[DataPtr] = None,
        read_write: List[DataPtr] = None,
    ) -> None:
        # 调用 kernel_launch 方法获取错误列表
        errors = self.kernel_launch(stream, read_only, read_write)
        # 断言错误列表的长度等于指定的错误数量
        self.assertEqual(len(errors), number_of_errors)

    # 测试空的内核启动操作
    def test_empty_kernel_launch(self):
        self.assert_good_kernel_launch(stream_id(0))

    # 测试简单通过的内核启动操作
    def test_simple_passing(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

    # 测试简单错误的内核启动操作
    def test_simple_error(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    # 测试简单同步的内核启动操作
    def test_simple_sync(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        # 处理事件记录和等待
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    # 测试读取操作检查最后写入的内核启动操作
    def test_reads_check_last_write(self):
        # 断言第一个内核启动操作正常执行
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 处理事件记录和等待
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        # 断言第二个内核启动操作正常执行
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        # 断言第三个内核启动操作存在一个错误
        self.assert_bad_kernel_launch(1, stream_id(3), read_only=[tensor_id(1)])
    def test_branch_sync(self):
        # 测试两个流在等待第三个流后能够读取，但在进一步同步之前不能写入。

        # 在流1上进行一个好的内核启动，读写tensor1
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 处理事件记录，事件0，流1
        self.handler._handle_event_record(event_id(0), stream_id(1))
        # 处理事件等待，事件0，流2
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        # 处理事件等待，事件0，流3
        self.handler._handle_event_wait(event_id(0), stream_id(3))
        # 在流2上进行一个好的内核启动，只读tensor1
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        # 在流3上进行一个好的内核启动，只读tensor1
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

        # 在流2上进行一个坏的内核启动，第1个，读写tensor1
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_chain_sync(self):
        # 迭代次数为10

        # 在流0上进行一个好的内核启动，只读tensor1
        self.assert_good_kernel_launch(stream_id(0), read_only=[tensor_id(1)])
        # 循环迭代
        for i in range(iterations):
            # 处理事件记录，事件i，流i
            self.handler._handle_event_record(event_id(i), stream_id(i))
            # 处理事件等待，事件i，流i+1
            self.handler._handle_event_wait(event_id(i), stream_id(i + 1))
        # 在流iterations上进行一个好的内核启动，读写tensor1
        self.assert_good_kernel_launch(stream_id(iterations), read_write=[tensor_id(1)])

    def test_expired_record(self):
        # 在流1上进行一个好的内核启动，只读tensor1
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        # 处理事件记录，事件0，流1
        self.handler._handle_event_record(event_id(0), stream_id(1))
        # 在流1上进行一个好的内核启动，只读tensor1
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        # 处理事件等待，事件0，流2
        self.handler._handle_event_wait(event_id(0), stream_id(2))

        # 在流2上进行一个坏的内核启动，第1个，读写tensor1
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_deleted_record(self):
        # 对于应该删除、应该创建的情况组合进行循环测试
        for should_delete, should_create in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            # 设置测试环境
            self.setUp()
            # 使用子测试，应该删除和应该创建
            with self.subTest(should_delete=should_delete, should_create=should_create):
                # 在流1上进行一个好的内核启动，只读tensor1
                self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
                # 处理事件记录，事件0，流1
                self.handler._handle_event_record(event_id(0), stream_id(1))

                # 如果应该删除，则处理事件删除，事件0
                if should_delete:
                    self.handler._handle_event_deletion(event_id(0))
                # 如果应该创建，则处理事件创建，事件0
                if should_create:
                    self.handler._handle_event_creation(event_id(0))

                # 处理事件等待，事件0，流2
                self.handler._handle_event_wait(event_id(0), stream_id(2))
                # 在流2上进行一个坏的内核启动，第1个，读写tensor1
                self.assert_bad_kernel_launch(
                    1, stream_id(2), read_write=[tensor_id(1)]
                )
    # 测试所有读取操作是否都被检查失败
    def test_all_reads_checked_failing(self):
        # 迭代次数
        iterations = 10
        # 循环进行操作
        for i in range(1, iterations):
            # 断言一个好的内核启动，指定流ID和只读张量ID
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            # 处理事件记录
            self.handler._handle_event_record(event_id(i), stream_id(i))

        # 循环进行操作
        for i in range(1, iterations):
            # 处理事件等待
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        # 断言一个好的内核启动，指定流ID和读写张量ID
        self.assert_good_kernel_launch(stream_id(iterations), read_only=[tensor_id(1)])
        # 处理事件记录
        self.handler._handle_event_record(event_id(iterations), stream_id(i))

        # 不与最后一个读取同步
        self.assert_bad_kernel_launch(1, stream_id(0), read_write=[tensor_id(1)])

    # 测试所有读取操作是否都被检查通过
    def test_all_reads_checked_passing(self):
        # 迭代次数
        iterations = 10
        # 循环进行操作
        for i in range(1, iterations):
            # 断言一个好的内核启动，指定流ID和只读张量ID
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            # 处理事件记录
            self.handler._handle_event_record(event_id(i), stream_id(i))

        # 循环进行操作
        for i in range(1, iterations):
            # 处理事件等待
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        # 断言一个好的内核启动，指定流ID和读写张量ID
        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])

    # 测试多个错误
    def test_multiple_errors(self):
        # 迭代次数
        iterations = 10
        # 断言一个好的内核启动，指定流ID和读写张量ID列表
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(iterations)]
        )
        # 断言一个坏的内核启动，指定迭代次数、流ID和读写张量ID列表
        self.assert_bad_kernel_launch(
            iterations,
            stream_id(1),
            read_write=[tensor_id(i) for i in range(iterations)],
        )

    # 测试正确的状态合并
    def test_correct_state_merging(self):
        # 测试等待事件后，流的状态是否确实设置为其旧状态和记录状态的逐点最大值

        # 断言一个好的内核启动，指定流ID和读写张量ID
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 断言一个好的内核启动，指定流ID和读写张量ID
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        # 处理事件记录
        self.handler._handle_event_record(event_id(1), stream_id(1))
        # 处理事件记录
        self.handler._handle_event_record(event_id(2), stream_id(2))

        # 断言一个好的内核启动，指定流ID和读写张量ID
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 断言一个好的内核启动，指定流ID和读写张量ID
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        # 处理事件等待
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        # 处理事件等待
        self.handler._handle_event_wait(event_id(2), stream_id(1))

        # 处理事件记录
        self.handler._handle_event_record(event_id(3), stream_id(2))
        # 处理事件等待
        self.handler._handle_event_wait(event_id(3), stream_id(1))
        # 断言一个好的内核启动，指定流ID和读写张量ID列表
        self.assert_good_kernel_launch(
            stream_id(1), read_write=[tensor_id(1), tensor_id(2)]
        )
    def test_record_override(self):
        # 测试重写记录功能

        # 测试第一个流ID的好内核启动，只读访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        # 测试第二个流ID的好内核启动，只读访问张量ID为2的数据
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(2)])
        # 处理事件ID为1的记录事件，关联到第一个流ID
        self.handler._handle_event_record(event_id(1), stream_id(1))
        # 处理事件ID为1的记录事件，关联到第二个流ID
        self.handler._handle_event_record(event_id(1), stream_id(2))

        # 处理事件ID为1的等待事件，关联到第三个流ID
        self.handler._handle_event_wait(event_id(1), stream_id(3))
        # 测试第三个流ID的坏内核启动，读写访问张量ID为1的数据
        self.assert_bad_kernel_launch(1, stream_id(3), read_write=[tensor_id(1)])

    def test_multiple_wait(self):
        # 测试多个等待操作

        # 测试第一个流ID的好内核启动，读写访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 处理事件ID为1的记录事件，关联到第一个流ID
        self.handler._handle_event_record(event_id(1), stream_id(1))
        # 处理事件ID为1的等待事件，关联到第二个流ID
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        # 处理事件ID为1的等待事件，关联到第三个流ID
        self.handler._handle_event_wait(event_id(1), stream_id(3))

        # 测试第二个流ID的好内核启动，只读访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        # 测试第三个流ID的好内核启动，只读访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

    def test_device_synchronize(self):
        # 测试设备同步功能

        iterations = 10
        # 迭代进行多次内核启动，每次读写访问张量ID为当前迭代数的数据
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_write=[tensor_id(i)])

        # 处理设备同步事件
        self.handler._handle_device_synchronization()
        # 测试第0号流ID的好内核启动，读写访问张量ID为1到迭代数的数据
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(1, iterations)]
        )

    def test_device_synchronization_expired(self):
        # 测试设备同步功能的一次性特性

        # 测试第一个流ID的好内核启动，读写访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 处理设备同步事件
        self.handler._handle_device_synchronization()
        # 测试第一个流ID的好内核启动，读写访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])

        # 测试第二个流ID的坏内核启动，读写访问张量ID为1的数据
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_new_stream_is_synchronized(self):
        # 测试在与主机同步操作后，新创建的流ID也会与之同步

        # 测试第一个流ID的好内核启动，读写访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 处理设备同步事件
        self.handler._handle_device_synchronization()
        # 处理创建流ID为2的操作
        self.handler._handle_stream_creation(stream_id(2))
        # 测试第二个流ID的好内核启动，读写访问张量ID为1的数据
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])
    def test_stream_synchronize(self):
        # Tests that a stream synchronization does correctly cause all streams to wait
        # for one specific stream, but does not synchronize all streams with each other.

        # 调用断言方法，验证在指定的流上进行同步操作，其他流会等待该流完成
        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])
        # 继续调用断言方法，验证在另一个流上进行操作，与第一个流独立，不会相互同步
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])
        # 调用处理器对象的流同步处理方法，指定流ID为0进行同步操作
        self.handler._handle_stream_synchronization(stream_id(0))

        # 在新的流上进行操作，验证这些操作会等待共享的张量1的读取，但不影响其他操作
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        # 继续验证另一个新流上的操作，也等待共享的张量1的读取，但不同步张量2的操作
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])
        # 调用失败的内核启动方法，验证在第4个流上，尝试读取张量2将会失败
        self.assert_bad_kernel_launch(1, stream_id(4), read_only=[tensor_id(2)])

    def test_event_synchronize(self):
        # Tests that an event synchronization does correctly cause all streams to wait
        # for a recorded event, but does not guarantee synchronization with the current
        # state of the stream that recorded the event.

        # 验证在流ID为1的流上进行操作，读写共享张量1
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        # 调用处理器对象的事件记录处理方法，记录事件ID为1，发生在流ID为1上
        self.handler._handle_event_record(event_id(1), stream_id(1))
        # 继续验证在同一流上的另一个操作，读写共享张量2
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])

        # 调用处理器对象的事件同步处理方法，同步事件ID为1
        self.handler._handle_event_synchronization(event_id(1))
        # 验证在新的流ID为2上进行操作，读写共享张量1，等待事件ID为1完成
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        # 调用失败的内核启动方法，验证在第2个流上，尝试读写张量2将会失败
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(2)])
class TestMessages(TestCase):
    # 设置测试的初始化方法，在每个测试方法执行前创建一个 EventHandler 实例
    def setUp(self):
        self.handler = csan.EventHandler()

    # 测试确保存在的情况
    def test_ensure_exists(self):
        # 定义一个测试参数
        ARG = 0
        # 遍历两个函数及其期望输出
        for func, out in [
            (
                self.handler._handle_event_deletion,
                f"Found Event with id: {ARG}, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_memory_deallocation,
                f"Found tensor with pointer: {ARG}, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
        ]:
            # 使用子测试来执行每个函数，并验证日志输出与期望的一致性
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    # 测试确保不存在的情况
    def test_ensure_does_not_exist(self):
        # 定义一个测试参数
        ARG = 0
        # 先调用两个函数来创建相关的 trace 记录
        self.handler._handle_event_creation(ARG)
        self.handler._handle_stream_creation(ARG)
        # 再次遍历两个函数及其期望输出
        for func, out in [
            (
                self.handler._handle_event_creation,
                "Found duplicate event creation in the trace for event with "
                f"id: {ARG}. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_stream_creation,
                "Found duplicate Stream creation in the trace for Stream with "
                f"id: {ARG}. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
            ),
        ]:
            # 使用子测试来执行每个函数，并验证日志输出与期望的一致性
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)
    def test_error_message(self):
        # 创建当前访问对象，指定访问类型为写入，序列号为1，流标识为stream_id(1)，操作者为"schema"，别名为["b"]，输出标志为True，包含堆栈跟踪信息
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=1,
            stream=stream_id(1),
            operator="schema",
            aliases=["b"],
            is_output=True,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace a")]
            ),
        )
        # 创建先前访问对象，指定访问类型为读取，序列号为2，流标识为stream_id(0)，操作者为"schema"，别名为["a"]，输出标志为False，包含堆栈跟踪信息
        previous_access = csan.Access(
            type=csan.AccessType.READ,
            seq_num=2,
            stream=stream_id(0),
            operator="schema",
            aliases=["a"],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace b")]
            ),
        )
        # 创建未同步访问错误对象，指定数据指针为tensor_id(1)，分配堆栈跟踪信息为包含堆栈跟踪信息，当前访问对象为current_access，先前访问对象为previous_access
        error = csan.UnsynchronizedAccessError(
            data_ptr=tensor_id(1),
            allocation_stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "alloc")]
            ),
            current_access=current_access,
            previous_access=previous_access,
        )
        # 断言错误消息字符串，使用多行字符串表示的格式化消息
        self.assertEqual(
            str(error),
            textwrap.dedent(
                """\
                ============================
                CSAN detected a possible data race on tensor with data pointer 1
                Access by stream 1001 during kernel:
                schema
                writing to argument(s) b, and to the output
                With stack trace:
                  File "file", line 0, in name
                    trace a

                Previous access by stream 1000 during kernel:
                schema
                reading from argument(s) a
                With stack trace:
                  File "file", line 0, in name
                    trace b

                Tensor was allocated with stack trace:
                  File "file", line 0, in name
                    alloc
                """
            ),
        )
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试用例
    run_tests()
```