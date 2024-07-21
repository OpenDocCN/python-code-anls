# `.\pytorch\test\distributed\test_c10d_functional_native.py`

```py
# Owner(s): ["module: c10d"]
# 导入必要的库和模块
import threading
import unittest
from typing import List

import torch

# 导入分布式相关的模块和函数
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._C import FileCheck
from torch._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
from torch.distributed._functional_collectives import (
    all_gather_into_tensor_coalesced,
    all_gather_tensor,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    AsyncCollectiveTensor,
    reduce_scatter_tensor,
    reduce_scatter_tensor_coalesced,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._triton import has_triton

# 加载测试模块的函数，用于动态导入测试代码
def load_test_module(name):
    import sys
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from unittest import mock

    # 确定测试文件的目录
    testdir = Path(__file__).absolute().parent.parent
    # 使用 mock.patch 临时修改 sys.path，加入测试文件目录
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        # 动态加载指定的测试模块
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()

# 动态加载 AOTI 运行工具模块
AOTIRunnerUtil = load_test_module("inductor.test_aot_inductor_utils").AOTIRunnerUtil

import sys

# 如果分布式包不可用，则打印警告信息并退出程序
if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 标记需要使用 NCCL 的测试类
@requires_nccl()
class TestWithNCCL(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()  # 创建多个进程用于测试

    @property
    def world_size(self) -> int:
        return 2

    @property
    def ranks(self) -> List[int]:
        return list(range(self.world_size))

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process_group(self) -> None:
        # 设置 torch._inductor 的 Triton 配置选项
        torch._inductor.config.triton.store_cubin = True
        torch._inductor.config.debug = True

        # 设置当前 CUDA 设备
        torch.cuda.set_device(self.device)
        # 使用文件存储初始化进程组
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 将当前进程组注册为默认进程组
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    @skip_if_lt_x_gpu(2)
    @skip_if_lt_x_gpu(2)
    # 如果当前环境下 GPU 少于两个，则跳过测试
    def test_all_reduce_single(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建一个大小为 10x10 的张量，每个元素的值为当前进程的排名，位于指定的设备上
        input = torch.full((10, 10), float(self.rank), device=self.device)
        
        # 执行全局平均值的reduce操作，使用"default"通信方式
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        
        # 等待异步张量操作完成
        output = torch.ops._c10d_functional.wait_tensor(output)
        
        # 断言输出张量和输入张量不是同一个对象
        assert id(output) != id(input)
        
        # 计算预期的值，这里是所有进程排名的平均值
        expect = sum(self.ranks) / self.world_size
        
        # 断言输出张量中的所有元素是否与预期值相等
        assert output.eq(expect).all()

        # 测试 Python API 和 AsyncCollectiveTensor
        output = all_reduce(
            input,
            "avg",
            "default",
        )
        
        # 断言输出是AsyncCollectiveTensor类型
        assert isinstance(output, AsyncCollectiveTensor)
        
        # 断言输出尚未完成
        assert not output.completed
        
        # 再次断言输出张量中的所有元素是否与预期值相等
        assert output.eq(expect).all()
        
        # 最后断言输出已经完成
        assert output.completed

    @skip_if_lt_x_gpu(2)
    # 如果当前环境下 GPU 少于两个，则跳过测试
    def test_all_reduce_single_(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建一个大小为 10x10 的张量，每个元素的值为当前进程的排名，位于指定的设备上
        input = torch.full((10, 10), float(self.rank), device=self.device)
        
        # 执行全局平均值的inplace reduce操作，使用"default"通信方式
        output = torch.ops._c10d_functional.all_reduce_(
            input,
            "avg",
            "default",
        )
        
        # 等待异步张量操作完成
        output = torch.ops._c10d_functional.wait_tensor(output)
        
        # 断言输出张量和输入张量是同一个对象
        assert id(output) == id(input)
        
        # 计算预期的值，这里是所有进程排名的平均值
        expect = sum(self.ranks) / self.world_size
        
        # 断言输出张量中的所有元素是否与预期值相等
        assert output.eq(expect).all()

    @skip_if_lt_x_gpu(2)
    # 如果当前环境下 GPU 少于两个，则跳过测试
    def test_all_reduce_coalesced(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建多个不同大小的张量列表，每个张量的值由当前进程的排名乘以其索引决定，位于指定的设备上
        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        
        # 执行多个张量的全局平均值reduce操作，使用"default"通信方式
        outputs = torch.ops._c10d_functional.all_reduce_coalesced(
            inputs,
            "avg",
            "default",
        )
        
        # 遍历每对输入输出张量
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            # 等待异步张量操作完成
            output = torch.ops._c10d_functional.wait_tensor(output)
            
            # 断言输出张量和输入张量不是同一个对象
            assert id(output) != id(input)
            
            # 计算预期的值，这里是所有进程排名的平均值乘以当前索引
            expect = sum(self.ranks) / self.world_size * i
            
            # 断言输出张量中的所有元素是否与预期值相等
            assert output.eq(expect).all()

        # 测试 Python API 和 AsyncCollectiveTensor
        outputs = all_reduce_coalesced(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            # 断言输出尚未完成
            assert not output.completed
            
            # 再次断言输出张量中的所有元素是否与预期值相等
            assert output.eq(sum(self.ranks) / self.world_size * i).all()
            
            # 最后断言输出已经完成
            assert output.completed

    @skip_if_lt_x_gpu(2)
    # 如果当前环境下 GPU 少于两个，则跳过测试
    def test_all_reduce_coalesced_(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建多个不同大小的张量列表，每个张量的值由当前进程的排名乘以其索引决定，位于指定的设备上
        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        
        # 执行多个张量的全局平均值的inplace reduce操作，使用"default"通信方式
        outputs = torch.ops._c10d_functional.all_reduce_coalesced_(
            inputs,
            "avg",
            "default",
        )
        
        # 遍历每对输入输出张量
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            # 等待异步张量操作完成
            output = torch.ops._c10d_functional.wait_tensor(output)
            
            # 断言输出张量和输入张量是同一个对象
            assert id(output) == id(input)
            
            # 计算预期的值，这里是所有进程排名的平均值乘以当前索引
            expect = sum(self.ranks) / self.world_size * i
            
            # 断言输出张量中的所有元素是否与预期值相等
            assert output.eq(expect).all()
    # 定义测试函数，用于测试all_gather_into_tensor_single功能
    def test_all_gather_into_tensor_single(self) -> None:
        # 初始化进程组，准备进行测试
        self._init_process_group()

        # 创建一个填充值为当前进程编号的10x10张量
        input = torch.full((10, 10), float(self.rank), device=self.device)
        
        # 调用底层C10D函数进行张量收集操作，并返回收集到的张量
        output = torch.ops._c10d_functional.all_gather_into_tensor(
            input,
            self.world_size,
            "default",
        )
        
        # 等待异步张量操作完成
        output = torch.ops._c10d_functional.wait_tensor(output)
        
        # 构建预期的输出张量，使用每个进程的编号填充
        expect = torch.cat(
            [
                torch.full((10, 10), float(rank), device=self.device)
                for rank in self.ranks
            ]
        )
        
        # 使用torch.allclose检查输出张量和预期张量是否近似相等
        assert torch.allclose(output, expect)
        
        # 使用torch.eq检查输出张量和预期张量的每个元素是否完全相等
        assert output.eq(expect).all()

        # 测试all_gather_into_tensor的out-variant版本
        output = torch.empty(expect.shape, device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor_out(
            input,
            self.world_size,
            "default",
            out=output,
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        
        # 使用torch.allclose检查输出张量和预期张量是否近似相等
        assert torch.allclose(output, expect)
        
        # 使用torch.eq检查输出张量和预期张量的每个元素是否完全相等
        assert output.eq(expect).all()

        # 测试Python API和AsyncCollectiveTensor的功能
        output = all_gather_tensor(
            input,
            0,
            "default",
        )
        
        # 检查输出是否为AsyncCollectiveTensor对象
        assert isinstance(output, AsyncCollectiveTensor)
        
        # 检查异步操作是否完成
        assert not output.completed
        
        # 使用torch.eq检查输出张量和预期张量的每个元素是否完全相等
        assert output.eq(expect).all()
        
        # 再次检查异步操作是否已完成
        assert output.completed

    @skip_if_lt_x_gpu(2)
    # 定义测试函数，用于测试all_gather_into_tensor_coalesced功能
    def test_all_gather_into_tensor_coalesced(self) -> None:
        # 初始化进程组，准备进行测试
        self._init_process_group()

        # 创建包含不同填充值的10个10x10张量作为输入
        inputs = [
            torch.full((10, 10), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        
        # 调用底层C10D函数进行集合操作，并返回收集到的张量列表
        outputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
            inputs,
            self.world_size,
            "default",
        )
        
        # 构建预期的输出张量列表，使用每个进程的编号和输入张量的索引来填充
        expect = [
            torch.cat(
                [
                    torch.full((10, 10), float(rank) * i, device=self.device)
                    for rank in self.ranks
                ]
            )
            for i in range(10)
        ]
        
        # 遍历输出张量列表，等待每个张量操作完成并检查结果
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            
            # 使用torch.eq检查输出张量和预期张量是否完全相等
            assert output.eq(expect[i]).all()

        # 测试Python API和AsyncCollectiveTensor的功能
        outputs = all_gather_into_tensor_coalesced(
            inputs,
            "default",
        )
        
        # 遍历输出张量列表，检查每个AsyncCollectiveTensor对象的状态和结果
        for i, output in enumerate(outputs):
            assert not output.completed
            assert output.eq(expect[i]).all()
            assert output.completed
    def test_reduce_scatter_tensor_single(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建张量，指定设备，包含进程 ranks 的信息
        input = torch.tensor(self.ranks, device=self.device)

        # 调用 C++ 扩展函数 reduce_scatter_tensor，进行平均值的 reduce-scatter
        output = torch.ops._c10d_functional.reduce_scatter_tensor(
            input,
            "avg",
            self.world_size,
            "default",
        )

        # 等待张量操作完成
        output = torch.ops._c10d_functional.wait_tensor(output)

        # 断言输出张量与当前进程的 rank 相等
        assert output.eq(self.rank).all()

        # 测试 Python API 和 AsyncCollectiveTensor
        output = reduce_scatter_tensor(
            input,
            "avg",
            0,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(self.rank).all()
        assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建多个张量，每个张量包含进程 ranks 的信息，并乘以对应的索引 i
        inputs = [torch.tensor(self.ranks, device=self.device) * i for i in range(10)]

        # 调用 C++ 扩展函数 reduce_scatter_tensor_coalesced，进行平均值的 coalesced reduce-scatter
        outputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
            inputs,
            "avg",
            self.world_size,
            "default",
        )

        # 遍历输出张量列表
        for i, output in enumerate(outputs):
            # 等待张量操作完成
            output = torch.ops._c10d_functional.wait_tensor(output)
            # 断言输出张量与当前进程 rank 乘以索引 i 相等
            assert output.eq(self.rank * i).all()

        # 测试 Python API 和 AsyncCollectiveTensor
        outputs = reduce_scatter_tensor_coalesced(
            inputs,
            "avg",
            [0] * 10,
            "default",
        )

        # 遍历输出张量列表
        for i, output in enumerate(outputs):
            # 断言输出张量未完成
            assert not output.completed
            # 断言输出张量与当前进程 rank 乘以索引 i 相等
            assert output.eq(self.rank * i).all()
            # 断言输出张量已完成
            assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self) -> None:
        # 初始化进程组
        self._init_process_group()
        
        # 设置 CUDA 设备
        torch.cuda.set_device(self.device)

        # 设置随机种子
        torch.manual_seed(42)
        
        # 创建发送大小矩阵
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        # 获取当前进程的输入分片大小和输出分片大小列表
        input_split_sizes = send_sz_matrix[self.rank].tolist()
        output_split_sizes = send_sz_matrix[:, self.rank].tolist()

        # 创建全为当前进程 rank 浮点数的输入张量，放置在 CUDA 设备上
        input = torch.full((sum(input_split_sizes),), float(self.rank)).cuda()

        # 调用 C++ 扩展函数 all_to_all_single，进行全对全的单个张量传输
        output = torch.ops._c10d_functional.all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            "default",
        )

        # 等待张量操作完成
        output = torch.ops._c10d_functional.wait_tensor(output)

        # 创建预期的输出张量
        expect = torch.cat(
            [
                torch.full((sz,), float(rank)).cuda()
                for rank, sz in enumerate(output_split_sizes)
            ]
        )

        # 断言输出张量与预期张量相等
        assert output.eq(expect).all()

        # 测试 Python API 和 AsyncCollectiveTensor
        output = all_to_all_single(
            input, output_split_sizes, input_split_sizes, "default"
        )
        # 断言输出张量未完成
        assert not output.completed
        # 断言输出张量与预期张量相等
        assert output.eq(expect).all()
        # 断言输出张量已完成
        assert output.completed
    # 定义测试方法，用于测试广播功能
    def test_broadcast(self) -> None:
        # 初始化进程组
        self._init_process_group()

        # 创建一个大小为 (10, 10) 的张量，每个元素值为当前进程的排名，存储在指定设备上
        input = torch.full((10, 10), float(self.rank), device=self.device)
        
        # 调用 C++ 扩展的广播功能来广播输入张量，rank=1，使用默认的通信后端
        output = torch.ops._c10d_functional.broadcast(
            input,
            1,
            "default",
        )
        
        # 等待异步张量操作完成
        output = torch.ops._c10d_functional.wait_tensor(output)
        
        # 断言输出张量与输入张量不是同一个对象
        assert id(output) != id(input)
        
        # 期望的输出值为1，断言输出张量所有元素是否都等于期望值
        expect = 1
        assert output.eq(expect).all()

        # 测试 Python API 和 AsyncCollectiveTensor
        output = funcol.broadcast(
            input,
            1,
            "default",
        )
        
        # 断言输出是 AsyncCollectiveTensor 类型
        assert isinstance(output, AsyncCollectiveTensor)
        
        # 断言输出异步操作尚未完成
        assert not output.completed
        
        # 断言输出张量所有元素是否都等于期望值
        assert output.eq(expect).all()
        
        # 断言输出异步操作已完成
        assert output.completed

    # 如果 GPU 数量小于 2，则跳过此测试方法
    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # 验证即使存在未等待的张量，进程也能正常终止
        self._init_process_group()

        # 创建一个大小为 (10, 10) 的张量，每个元素值为当前进程的排名，存储在指定设备上
        input = torch.full((10, 10), float(self.rank), device=self.device)
        
        # 调用 C++ 扩展的全局归约功能，将输入张量的数据进行平均归约，使用默认的通信后端
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )

    # 如果没有 Triton 或 GPU 架构不符合要求，则跳过此测试方法
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @fresh_inductor_cache()
    def test_threading(self):
        # 初始化进程组
        self._init_process_group()
        
        # 根据当前进程的排名，创建 CUDA 设备对象
        device = torch.device(f"cuda:{self.rank}")

        # 定义一个函数，对输入张量进行操作并返回处理结果
        def func(arg: torch.Tensor) -> torch.Tensor:
            # 对输入张量的每个元素加 42
            buf0 = arg + 42
            
            # 对加 42 后的张量进行全局平均归约，使用 '0' 组进行通信
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            
            # 等待异步张量操作完成
            ar0 = funcol.wait_tensor(ar0)
            
            # 对结果张量的每个元素加 1，并返回
            return ar0 + 1

        # 创建一个随机初始化的大小为 (4, 4) 的张量，存储在 CUDA 设备上
        arg = torch.rand(4, 4, device=device)
        
        # 调用 func 函数处理输入张量
        func(arg)

        # 编译 func 函数并获取其 Triton 代码
        compiled = torch.compile(func, fullgraph=True)
        code = run_and_get_triton_code(compiled, arg)
        
        # 使用 FileCheck 工具验证 Triton 代码中是否包含特定字符串
        FileCheck().check("all_reduce_.default(buf0, 'avg', '0')").run(code)

        # 除非显式指定（例如在自定义运行时中），否则进程组注册表在进程中的所有线程之间共享。
        # 这里验证主线程注册的进程组可以在不同线程中解析。
        class TestThread(threading.Thread):
            def run(self):
                self.exc = None
                try:
                    # 在当前线程中调用 func 函数和 compiled 函数
                    func(arg)
                    compiled(arg)
                except BaseException as exc:
                    self.exc = exc

            def join(self):
                threading.Thread.join(self)
                if self.exc:
                    raise self.exc

        # 创建 TestThread 实例并启动线程
        t = TestThread()
        t.start()
        t.join()
# 定义一个测试类 CompileTest，继承自 TestCase，用于编写测试用例
class CompileTest(TestCase):

    # 在每个测试方法执行前调用的方法
    def setUp(self):
        # 允许在 torch.compile 后测试 aoti
        torch._inductor.config.triton.store_cubin = True
        # 设置调试模式为 True
        torch._inductor.config.debug = True

        # 设置当前进程的 rank 为 0
        self.rank = 0
        # 设置世界中进程的数量为 2
        self.world_size = 2
        # 设置当前 CUDA 设备为 "cuda:0"
        torch.cuda.set_device("cuda:0")

        # 创建一个 FakeStore 对象
        store = FakeStore()
        # 初始化进程组，使用 fake 后端
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    # 在每个测试方法执行后调用的方法
    def tearDown(self):
        # 销毁进程组
        dist.destroy_process_group()

    # 标记为装饰器，如果没有 Triton 或者 GPU 架构过旧则跳过执行
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    # 测试函数，测试 Inductor 的全局归约功能
    def test_inductor_all_reduce_single(self):
        # 定义一个函数 func，接受一个 torch.Tensor 参数，返回一个 torch.Tensor
        def func(arg: torch.Tensor) -> torch.Tensor:
            # 将 arg 加上 42，赋值给 buf0
            buf0 = arg + 42
            # 预期使用 Inductor 分配的 buf 进行原地操作的全局归约
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            # 等待 ar0 张量操作完成
            ar0 = funcol.wait_tensor(ar0)
            # 预期对于图输入，不会进行原地操作的全局归约
            ar1 = funcol.all_reduce(arg, "avg", "0")
            # 等待 ar1 张量操作完成
            ar1 = funcol.wait_tensor(ar1)
            # 返回 ar0 和 ar1
            return ar0, ar1

        # 创建一个随机初始化的大小为 4x4 的 CUDA 张量 arg
        arg = torch.rand(4, 4, device="cuda")
        # 编译 func 函数
        compiled = torch.compile(func)

        # 运行并获取 Triton 生成的代码
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("buf0 = empty")
            .check("buf7 = empty")
            # 预期使用 Inductor 分配的 buf 进行原地操作的全局归约
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # 预期对于图输入不会进行原地操作的全局归约 (buf5 是一个克隆)
            .check("torch.ops._c10d_functional.all_reduce_.default(buf7")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf7")
            # 预期返回时不会有额外的拷贝
            .check("return (buf0, buf7, )")
            .run(code)
        )
        # 断言在生成的代码中不会出现 "= torch.ops._c10d_functional.wait_tensor.default"
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # 测试 aoti
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    # 标记为装饰器，如果没有 Triton 或者 GPU 架构过旧则跳过执行
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_all_reduce_coalesced(self):
        # 定义一个嵌套函数 func，接受一个参数列表 args（类型为 torch.Tensor 列表），返回两个 torch.Tensor 对象
        def func(args: List[torch.Tensor]) -> torch.Tensor:
            # 对参数列表中的每个 Tensor 执行加法操作（每个 Tensor 加 42），并存储在 bufs 列表中
            bufs = [arg + 42 for arg in args]
            # 调用 funcol.all_reduce_coalesced 函数对 bufs 进行聚合操作，平均值聚合，使用"0"作为分组
            ar0 = funcol.all_reduce_coalesced(bufs, "avg", "0")
            # 对 ar0 中的每个输出 Tensor 执行等待操作，将结果存储在列表 ar0 中
            ar0 = [funcol.wait_tensor(out) for out in ar0]
            # 调用 funcol.all_reduce_coalesced 函数对 args 进行聚合操作，平均值聚合，使用"0"作为分组
            ar1 = funcol.all_reduce_coalesced(args, "avg", "0")
            # 对 ar1 中的每个输出 Tensor 执行等待操作，将结果存储在列表 ar1 中
            ar1 = [funcol.wait_tensor(out) for out in ar1]
            # 返回聚合结果 ar0 和 ar1
            return ar0, ar1

        # 创建一个包含两个 shape 为 (4, 4) 的随机 Tensor 的列表 args，设备为 CUDA
        args = [torch.rand(4, 4, device="cuda") for _ in range(2)]
        # 编译函数 func
        compiled = torch.compile(func)
        # 运行编译后的 func，并获取 Triton 代码
        code = run_and_get_triton_code(compiled, args)
        (
            # 使用 FileCheck 检查代码中的字符串
            FileCheck()
            # 检查是否存在 "buf0 = empty"
            .check("buf0 = empty")
            .check("buf5 = empty")
            .check("buf1 = empty")
            .check("buf6 = empty")
            # 期望使用 inductor 分配的 buf 进行原地操作
            .check(
                "torch.ops._c10d_functional.all_reduce_coalesced_"
                ".default([buf0, buf1]"
            )
            # 期望对图形输入不进行原地操作（buf5, buf6 是克隆）
            .check(
                "torch.ops._c10d_functional.all_reduce_coalesced_"
                ".default([buf5, buf6]"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf5")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf6")
            # 期望返回时不产生额外的拷贝
            .check("return (buf0, buf1, buf5, buf6, )")
            .run(code)  # 运行 Triton 代码
        )
        # 确保代码中没有包含 "= torch.ops._c10d_functional.wait_tensor.default"
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # 测试 AOTI
        out = AOTIRunnerUtil.run("cuda", func, (args,))
        torch.cuda.synchronize()

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_inplace_op_on_view(self):
        # 定义一个函数 func，接受一个参数 arg（类型为 torch.Tensor），返回一个 torch.Tensor 对象
        def func(arg: torch.Tensor) -> torch.Tensor:
            # 对参数进行加法操作（加 10），然后取前两行
            buf0 = (arg + 10)[:2]
            # 对 buf0 执行聚合操作，平均值聚合，使用"0"作为分组
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            # 对 ar0 中的输出 Tensor 执行等待操作
            ar0 = funcol.wait_tensor(ar0)
            # 返回聚合结果 ar0
            return ar0

        # 创建一个 shape 为 (4, 4) 的随机 Tensor，设备为 CUDA
        arg = torch.rand(4, 4, device="cuda")
        # 编译函数 func
        compiled = torch.compile(func)

        # 运行编译后的 func，并获取 Triton 代码
        code = run_and_get_triton_code(compiled, arg)
        (
            # 使用 FileCheck 检查代码中的字符串
            FileCheck()
            # 检查是否存在 "buf0 = empty"
            .check("buf0 = empty")
            # 确保 all_reduce_ 输入是一个视图
            .check(
                "torch.ops._c10d_functional.all_reduce_.default(reinterpret_tensor(buf0"
            )
            .check(
                "torch.ops._c10d_functional.wait_tensor.default(reinterpret_tensor(buf0"
            )
            .check("return (reinterpret_tensor(buf0")
            .run(code)  # 运行 Triton 代码
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_reuse_buffer_after_inplace_collective(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            # Expect allocation
            # 对输入张量进行加法操作，返回新的张量 buf0
            buf0 = arg + 42
            # 调用 funcol.all_reduce 对 buf0 执行平均值归约操作
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            # 等待张量 ar0 的完成
            ar0 = funcol.wait_tensor(ar0)
            # Expect allocation
            # 对输入张量 arg 和 ar0 进行矩阵乘法操作，返回新的张量 buf1
            buf1 = torch.mm(arg, ar0)
            # Expect buf0 to be reused
            # 对输入张量 arg 和 buf1 进行矩阵乘法操作，期望 buf0 被重用
            buf2 = torch.mm(arg, buf1)
            return buf1, buf2

        arg = torch.rand(4, 4, device="cuda")
        # 编译 func 函数以便在 Triton 上运行
        compiled = torch.compile(func)
        # 运行编译后的函数并获取 Triton 代码
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            # Expect allocation
            # 检查是否存在对 buf0 的空间分配操作
            .check("buf0 = empty")
            # 调用 all_reduce 操作，对 buf0 执行平均值归约
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            # 等待 buf0 张量的完成
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect allocation
            # 检查是否存在对 buf7 的空间分配操作
            .check("buf7 = empty")
            # 调用外部的矩阵乘法内核，将 arg0_1 和 buf0 作为输入，buf7 作为输出
            .check("extern_kernels.mm(arg0_1, buf0, out=buf7")
            # Expect buf0 to be reused
            # 将 buf0 赋值给 buf8，同时删除 buf0（重用操作）
            .check("buf8 = buf0; del buf0  # reuse")
            # 调用外部的矩阵乘法内核，将 arg0_1 和 buf7 作为输入，buf8 作为输出
            .check("extern_kernels.mm(arg0_1, buf7, out=buf8")
            # Expect no extra copy on return
            # 检查返回值，不期望有额外的拷贝操作
            .check("return (buf7, buf8, )")
            .run(code)
        )
        # 断言在生成的 Triton 代码中不会出现 "= torch.ops._c10d_functional.wait_tensor.default"
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_all_gather_into_tensor_single(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            # 调用 funcol.all_gather_tensor 将 arg 收集到所有设备上的张量，返回张量 ag0
            ag0 = funcol.all_gather_tensor(arg, 0, "0")
            # 等待张量 ag0 的完成
            ag0 = funcol.wait_tensor(ag0)
            return ag0

        arg = torch.rand(4, 4, device="cuda")
        # 编译 func 函数以便在 Triton 上运行
        compiled = torch.compile(func)
        # 运行编译后的函数并获取 Triton 代码
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            # 检查是否调用了 torch.ops._c10d_functional.all_gather_into_tensor.default 对 arg0_1 进行收集
            .check("buf0 = torch.ops._c10d_functional.all_gather_into_tensor.default(arg0_1")
            # 等待 buf0 张量的完成
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no extra copy on return
            # 检查返回值，不期望有额外的拷贝操作
            .check("return (buf0, )")
            .run(code)
        )
        # 断言在生成的 Triton 代码中不会出现 "= torch.ops._c10d_functional.wait_tensor.default"
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # Test aoti
        # 运行 AOTIRunnerUtil 的辅助函数，验证其运行结果
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    # 定义测试函数：对单个张量进行 reduce scatter 操作的测试
    def test_inductor_reduce_scatter_tensor_single(self):
        # 定义内部函数：接收一个 torch.Tensor 参数，返回一个 torch.Tensor 结果
        def func(arg: torch.Tensor) -> torch.Tensor:
            # 调用 funcol.reduce_scatter_tensor 函数，对输入张量进行 reduce scatter 操作，使用 "avg" 策略，指定设备为 "0"
            rs0 = funcol.reduce_scatter_tensor(arg, "avg", 0, "0")
            # 等待 rs0 张量的操作完成
            rs0 = funcol.wait_tensor(rs0)
            # 返回操作后的结果张量 rs0
            return rs0

        # 创建一个随机初始化的张量列表，每个张量形状为 (4, 4)，设备为 "cuda"
        arg = torch.rand(4, 4, device="cuda")
        # 编译内部函数 func
        compiled = torch.compile(func)
        # 运行并获取 Triton 生成的代码
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            # 检查 Triton 生成的代码中 reduce scatter 的部分
            .check("buf0 = torch.ops._c10d_functional.reduce_scatter_tensor.default(arg0_1")
            # 检查等待张量操作的部分
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # 检查返回结果，期望无额外的复制操作
            .check("return (buf0, )")
            .run(code)  # 运行 Triton 生成的代码
        )

        # 运行 AOTI 测试
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        # 同步 CUDA 设备
        torch.cuda.synchronize()
    # 定义一个测试函数，用于测试 reduce_scatter_tensor_coalesced 函数的功能
    def test_inductor_reduce_scatter_tensor_coalesced(self):
        # 定义一个内部函数 func，接受一个 torch.Tensor 类型的列表作为参数，返回一个 torch.Tensor 对象
        def func(args: List[torch.Tensor]) -> torch.Tensor:
            # 调用 funcol.reduce_scatter_tensor_coalesced 函数，将 args 列表中的张量进行 reduce_scatter 操作，使用 "avg" 策略
            rs0 = funcol.reduce_scatter_tensor_coalesced(
                args, "avg", [0] * len(args), "0"
            )
            # 等待 rs0 中每个张量的计算完成
            rs0 = [funcol.wait_tensor(out) for out in rs0]
            # 返回处理后的 rs0 结果
            return rs0

        # 生成一个包含四个随机张量的列表，每个张量的形状为 (4, 4)，存储在 CUDA 设备上
        args = [torch.rand(4, 4, device="cuda") for _ in range(4)]
        # 编译 func 函数
        compiled = torch.compile(func)
        # 运行并获取 Triton 代码
        code = run_and_get_triton_code(compiled, args)

        # 使用 FileCheck 检查 Triton 生成的代码，验证其中是否包含特定字符串
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced"
                ".default([arg0_1, arg1_1, arg2_1, arg3_1]"
            )
            .check("buf1 = buf0[0]")
            .check("buf2 = buf0[1]")
            .check("buf3 = buf0[2]")
            .check("buf4 = buf0[3]")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf4")
            # 验证返回时不会有额外的复制操作
            .check("return (buf1, buf2, buf3, buf4, )")
            .run(code)
        )

        # 运行 AOTIRunnerUtil 的 run 方法，使用 CUDA 设备执行 func 函数，并等待计算完成
        AOTIRunnerUtil.run("cuda", func, (args,))
        # 同步 CUDA 设备，确保所有操作完成
        torch.cuda.synchronize()

    # 如果系统没有安装 Triton 或 GPU 架构过旧，则跳过该测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # 在运行测试之前刷新 Inductor 缓存，确保获取最新的编译结果
    @fresh_inductor_cache()
    def test_inductor_all_to_all_single(self):
        # 定义内部函数，将张量转换为列表，并检查每个元素是否符合尺寸要求
        def _tolist_with_constrain_as_size(tensor):
            lst = tensor.tolist()
            for elem in lst:
                torch._check_is_size(elem)
            return lst

        # 定义主测试函数，执行 all_to_all_single 操作
        def func(
            input: torch.Tensor,
            output_split_sizes: torch.Tensor,
            input_split_sizes: torch.Tensor,
        ) -> torch.Tensor:
            # 调用 funcol.all_to_all_single 函数进行通信操作
            output = funcol.all_to_all_single(
                input,
                _tolist_with_constrain_as_size(output_split_sizes),
                _tolist_with_constrain_as_size(input_split_sizes),
                "0",
            )
            # 等待输出张量完成
            return funcol.wait_tensor(output)

        # 设置随机种子
        torch.manual_seed(42)
        # 生成发送尺寸矩阵，范围在 [0, 20) 内的随机整数
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        # 从发送尺寸矩阵中获取当前进程的输入分割尺寸
        input_split_sizes = send_sz_matrix[self.rank]
        # 获取发送尺寸矩阵中对当前进程的输出分割尺寸，并保持连续性
        output_split_sizes = send_sz_matrix[:, self.rank].contiguous()
        # 创建输入张量，其大小为输入分割尺寸之和，并转移到 GPU
        input = torch.full((input_split_sizes.sum().item(),), float(self.rank)).cuda()

        # 使用 torch._dynamo.config.patch 上下文管理器设置动态计算相关的配置
        with torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            # 编译函数 func，启用动态计算
            compiled = torch.compile(func, dynamic=True)
            # 运行编译后的函数，并获取 Triton 代码
            code = run_and_get_triton_code(
                compiled, input, output_split_sizes, input_split_sizes
            )
        # 使用 FileCheck 验证 Triton 生成的代码
        (
            FileCheck()
            .check_regex(
                "torch.ops._c10d_functional.all_to_all_single.default\\("
                "arg\\d+_\\d+, \\[u\\d+, u\\d+\\], \\[u\\d+, u\\d+\\]"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(")
            .run(code)
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_broadcast(self):
        # 定义内部函数func，接受torch.Tensor类型参数arg，返回torch.Tensor类型结果
        def func(arg: torch.Tensor) -> torch.Tensor:
            # buf0为arg加上常数42的结果
            buf0 = arg + 42
            # 期望使用内部分配的缓冲区进行原位操作
            br0 = funcol.broadcast(buf0, 1, "0")
            # 等待br0操作的完成
            br0 = funcol.wait_tensor(br0)
            # 期望使用图输入，不进行原位操作
            br1 = funcol.broadcast(arg, 0, "0")
            # 等待br1操作的完成
            br1 = funcol.wait_tensor(br1)
            # 返回br0和br1的结果
            return br0, br1

        # 生成一个CUDA设备上的随机4x4张量arg
        arg = torch.rand(4, 4, device="cuda")
        # 编译func函数
        compiled = torch.compile(func)

        # 运行func编译后的代码并获取Triton代码
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("buf0 = empty")
            .check("buf7 = empty")
            # 期望使用内部分配的缓冲区进行原位操作
            .check("torch.ops._c10d_functional.broadcast_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # 期望使用图输入，不进行原位操作（buf7是一个克隆）
            .check("torch.ops._c10d_functional.broadcast_.default(buf7")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf7")
            # 返回时不会有额外的复制
            .check("return (buf0, buf7, )")
            .run(code)
        )

        # 测试AOTI
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_ranks_and_tag(self):
        # 定义内部函数func，接受torch.Tensor类型参数arg，返回torch.Tensor类型结果
        def func(arg: torch.Tensor) -> torch.Tensor:
            # buf0为arg加上常数42的结果
            buf0 = arg + 42
            # 期望使用内部分配的缓冲区进行原位操作
            ar0 = funcol.all_reduce(buf0, "avg", [0, 1], "")
            # 等待ar0操作的完成
            ar0 = funcol.wait_tensor(ar0)
            # 期望使用图输入，不进行原位操作
            ar1 = funcol.all_reduce(arg, "avg", [0, 1], "")
            # 等待ar1操作的完成
            ar1 = funcol.wait_tensor(ar1)
            # 返回ar0和ar1的结果
            return ar0, ar1

        # 生成一个CUDA设备上的随机4x4张量arg
        arg = torch.rand(4, 4, device="cuda")
        # 编译func函数，使用完整图形
        compiled = torch.compile(func, fullgraph=True)

        # 运行func编译后的代码并获取Triton代码，检查是否有all_reduce的默认操作
        (FileCheck().check("all_reduce_.default(buf0, 'avg', '0')").run(code))
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码或者功能测试
    run_tests()
```