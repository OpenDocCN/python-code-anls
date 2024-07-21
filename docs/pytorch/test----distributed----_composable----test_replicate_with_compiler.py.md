# `.\pytorch\test\distributed\_composable\test_replicate_with_compiler.py`

```
# Owner(s): ["oncall: distributed"]

# 导入所需的库和模块
import contextlib  # 上下文管理工具
import functools  # 函数工具
import os  # 系统操作模块
import unittest  # 单元测试框架
from copy import deepcopy  # 深拷贝函数
from typing import Callable, Optional  # 类型提示：函数和可选参数

import torch  # PyTorch 主库
import torch.distributed as dist  # 分布式 PyTorch 模块
from torch import _inductor as inductor, nn  # 内部编译和神经网络模块
from torch._C import FileCheck  # Torch C++ 扩展中的文件检查工具
from torch._dynamo import compiled_autograd  # 编译自动求导工具
from torch._dynamo.utils import counters  # 编译工具的计数器
from torch._inductor.utils import run_and_get_triton_code  # 编译工具和 Triton 代码的运行函数
from torch.distributed._composable.replicate import replicate  # 分布式计算模块中的复制函数
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,  # DDP 通信钩子的默认实现
)
from torch.distributed.device_mesh import init_device_mesh  # 初始化设备网格
from torch.distributed.tensor.parallel import (
    ColwiseParallel,  # 张量列并行操作
    parallelize_module,  # 并行化模块
    RowwiseParallel,  # 张量行并行操作
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP  # 分布式数据并行
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,  # 多进程测试用例基类
    skip_if_lt_x_gpu,  # 如果 GPU 少于 x 个则跳过测试
    skip_if_rocm,  # 如果是 ROCm 平台则跳过测试
)
from torch.testing._internal.common_utils import run_tests  # 运行测试工具
from torch.utils._triton import has_triton  # Triton 模块是否可用
from torch.utils.checkpoint import checkpoint  # 检查点函数

DIM = 2000  # 维度设置为 2000


class Net(nn.Module):
    def __init__(self, checkpoint=False):
        super().__init__()
        # 定义四个线性层
        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, DIM)
        self.fc4 = nn.Linear(DIM, DIM)
        self.use_checkpoint = checkpoint  # 是否使用检查点

    def forward(self, x):
        if self.use_checkpoint:
            _fc1 = checkpoint(self.fc1, x, use_reentrant=False)  # 如果使用检查点，则对第一个全连接层进行检查点操作
        else:
            _fc1 = self.fc1(x)  # 否则直接进行前向传播计算
        return self.fc4(self.fc3(self.fc2(_fc1)))  # 返回经过四个全连接层计算后的结果


def compiler_fn(no_inductor=False):
    def _compiler_fn(gm):
        def inner_compiler(gm_, example_inputs_):
            if no_inductor:
                return gm_
            else:
                return inductor.compile(gm_, example_inputs_)  # 使用编译器编译图模块

        gm = torch.compile(gm, fullgraph=True, backend=inner_compiler)  # 编译整个图模块
        return gm

    return _compiler_fn


class ReplicateTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())  # 返回当前环境下的 GPU 数量的最小值作为世界大小

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()  # 启动多进程测试

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)  # 在测试结束时尝试删除文件
        except OSError:
            pass

    def _test_compile(
        self,
        *,
        use_gpu: bool,
        no_sync: bool,
        setup_func: Optional[Callable] = None,
        no_inductor: bool = False,
        no_compile_forward: bool = False,
    ):
        # 测试使用 CPU 进行编译
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",  # 使用融合 DDP 通信操作
            "schedule_comm_wait",  # 安排通信等待
        ]
        self._test_compile(use_gpu=False, no_sync=False)  # 调用测试编译函数


    def test_compile_cpu(self):
        # 测试在 CPU 上进行编译
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",  # 使用融合 DDP 通信操作
            "schedule_comm_wait",  # 安排通信等待
        ]
        self._test_compile(use_gpu=False, no_sync=False)  # 调用测试编译函数
    # 定义测试函数，测试在 CPU 上使用 coalesced_op。
    def test_compile_cpu_no_sync(self):
        # 设置 Inductor 配置以支持 DDP 通信优化的 passes 列表
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
        # 调用 _test_compile 方法进行测试，使用 CPU，不同步
        self._test_compile(use_gpu=False, no_sync=True)

    # 如果没有 Triton 或 GPU 架构较老，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_gpu(self):
        # 调用 _test_compile 方法进行 GPU 编译测试，支持同步
        self._test_compile(use_gpu=True, no_sync=False)

    # 如果没有 Triton 或 GPU 架构较老，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_bf16(self):
        # 定义 setup 函数，为 BF16 压缩挂钩注册通信钩子
        def setup(model, compiled_replicate_model, compiled_ddp_model) -> None:
            model.register_comm_hook(None, ddp_default_hooks.bf16_compress_hook)
            compiled_m = compiled_replicate_model._orig_mod
            compiled_m.register_comm_hook(None, ddp_default_hooks.bf16_compress_hook)
            compiled_ddp_model.register_comm_hook(
                None, ddp_default_hooks.bf16_compress_hook
            )

        # 调用 _test_compile 方法进行 BF16 编译测试，支持同步，使用 setup 函数配置
        self._test_compile(use_gpu=True, no_sync=False, setup_func=setup)

    # 如果没有 Triton 或 GPU 架构较老，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_fp16(self):
        # 定义 setup 函数，为 FP16 压缩挂钩注册通信钩子
        def setup(model, compiled_replicate_model, compiled_ddp_model) -> None:
            model.register_comm_hook(None, ddp_default_hooks.fp16_compress_hook)
            compiled_m = compiled_replicate_model._orig_mod
            compiled_m.register_comm_hook(None, ddp_default_hooks.fp16_compress_hook)
            compiled_ddp_model.register_comm_hook(
                None, ddp_default_hooks.fp16_compress_hook
            )

        # TODO: 解决为什么需要禁用 Inductor 以避免测试错误
        # 调用 _test_compile 方法进行 FP16 编译测试，支持同步，使用 setup 函数配置，禁用 Inductor
        self._test_compile(
            use_gpu=True, no_sync=False, setup_func=setup, no_inductor=True
        )

    # 如果没有 Triton 或 GPU 架构较老，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_backward_only(self):
        # 调用 _test_compile 方法进行仅反向传播的编译测试，使用 GPU，支持同步，不编译前向传播
        self._test_compile(use_gpu=True, no_sync=False, no_compile_forward=True)
    # 定义一个测试函数，用于测试桶分配功能，可以选择是否初始化进程组和运行循环次数
    def _test_bucketing(self, init_process_group=True, loop=1):
        # 如果需要初始化进程组，则使用gloo后端初始化进程组
        if init_process_group:
            dist.init_process_group(
                backend="gloo",
                rank=self.rank,
                world_size=self.world_size,
                store=dist.FileStore(self.file_name, self.world_size),
            )
        # 创建一个新的神经网络模型
        model = Net()
        # 创建一个随机输入张量
        input = torch.randn([1, DIM])
        # 设置优化DDP（分布式数据并行）的配置为python_reducer
        torch._dynamo.config.optimize_ddp = "python_reducer"
        # 编译复制后的模型，不包括完整图形信息
        compiled_replicate_model = torch.compile(
            replicate(deepcopy(model)), fullgraph=False
        )

        # 定义反向传播函数，使用编译的自动求导功能
        def bwd(loss):
            with compiled_autograd.enable(compiler_fn()):
                loss.backward()

        # 执行指定次数的循环
        for i in range(loop):
            # 计算编译后的模型在输入上的输出，并求和作为损失
            loss = compiled_replicate_model(input).sum()
            # 如果不是最后一次迭代，则执行反向传播
            if i != loop - 1:
                # 最后一次反向传播由run_and_get_triton_code处理
                bwd(loss)

        # 获取运行后的triton代码
        code = run_and_get_triton_code(functools.partial(bwd, loss=loss))

        # 断言检查结果中的ddp桶数量是否为3
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        # 返回生成的代码
        return code

    # 用于测试使用coalesced操作的桶分配功能
    @torch._inductor.config.patch(
        _fuse_ddp_communication_passes=[
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
    )
    # todo: 这个pass会影响到Inductor，因为它认为这是推断而可以应用这个pass。
    # 编译后的自动求导应该关闭这些pass。
    @torch._inductor.config.patch(reorder_for_locality=False)
    def test_bucketing_coalesced_op(self):
        # 梯度为None
        code = self._test_bucketing()
        # 断言检查结果中的ddp桶数量是否为3
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        # 创建FileCheck对象
        fc = FileCheck()
        # 检查cpp_fused_和all_reduce_coalesced_.default操作的存在
        for i in range(3):
            fc.check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_coalesced_.default("
            )
        # 检查wait_tensor.default操作的存在
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")

        # 运行FileCheck对象来验证生成的代码
        fc.run(code)

        # 梯度为None
        code = self._test_bucketing(init_process_group=False, loop=2)
        # 断言检查结果中的ddp桶数量是否为3
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        # 创建FileCheck对象
        fc = FileCheck()
        # 检查cpp_fused_和all_reduce_coalesced_.default操作的存在
        for i in range(3):
            fc.check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_coalesced_.default("
            )
        # 检查wait_tensor.default操作的存在
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")

        # 运行FileCheck对象来验证生成的代码

        fc.run(code)

    # 用于测试使用concat操作的桶分配功能
    @torch._inductor.config.patch(
        _fuse_ddp_communication_passes=[
            "fuse_ddp_with_concat_op",
            "schedule_comm_wait",
        ]
    )
    # todo: 这个pass会影响到Inductor，因为它认为这是推断而可以应用这个pass。
    # 编译后的自动求导应该关闭这些pass。
    @torch._inductor.config.patch(reorder_for_locality=False)
    # 定义测试函数，测试桶装置连接操作
    def test_bucketing_concat_op(self):
        # 断言梯度为None
        code = self._test_bucketing()
        # 断言计数器中"inductor"的"ddp_buckets"键对应的值为3
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        # 创建FileCheck对象
        fc = FileCheck()
        # 循环3次，检查字符串"aten.flatten.using_ints("
        for i in range(3):
            fc.check("aten.flatten.using_ints(").check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_.default("
            )
        # 循环3次，检查字符串"torch.ops._c10d_functional.wait_tensor.default"
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")
        # 运行FileCheck对象对代码进行检查
        fc.run(code)

        # 断言梯度不为None
        code = self._test_bucketing(init_process_group=False, loop=2)
        # 再次断言计数器中"inductor"的"ddp_buckets"键对应的值为3
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        # 重新创建FileCheck对象
        fc = FileCheck()
        # 循环3次，检查字符串"aten.flatten.using_ints("
        for i in range(3):
            fc.check("aten.flatten.using_ints(").check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_.default("
            )
        # 循环3次，检查字符串"torch.ops._c10d_functional.wait_tensor.default"
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")
        # 再次运行FileCheck对象对代码进行检查
        fc.run(code)
class DDP_TP_Test(MultiProcessTestCase):
    # DDP_TP_Test 类，继承自 MultiProcessTestCase，用于测试分布式数据并行（DDP）和时间并行（TP）。

    @property
    def world_size(self) -> int:
        # 返回当前 CUDA 设备数和最大支持的 4 中的较小值作为世界大小
        return min(4, torch.cuda.device_count())

    def setUp(self) -> None:
        # 设置测试环境，在每个测试方法执行前调用
        super().setUp()
        self._spawn_processes()  # 启动测试进程

    def tearDown(self):
        # 清理测试环境，在每个测试方法执行后调用
        super().tearDown()
        try:
            os.remove(self.file_name)  # 尝试删除测试过程中创建的文件
        except OSError:
            pass  # 如果文件不存在则忽略

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(4)
    def test_ddp_tp(self):
        # 跳过测试条件：如果没有 Triton 或 GPU 架构不符合要求，则跳过测试
        torch.cuda.set_device(f"cuda:{self.rank}")  # 设置当前 CUDA 设备
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )  # 初始化分布式进程组

        model = Net().cuda()  # 创建一个在 CUDA 上运行的神经网络模型
        compiled_replicate_model = deepcopy(model)  # 深拷贝模型用于编译后的复制

        mesh_2d = init_device_mesh(
            "cuda", (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )  # 初始化设备网格，用于数据并行和时间并行

        tp_mesh = mesh_2d["tp"]  # 时间并行的设备网格
        dp_mesh = mesh_2d["dp"]  # 数据并行的设备网格

        parallelize_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
            "fc3": ColwiseParallel(),
            "fc4": RowwiseParallel(),
        }  # 并行化计划，指定模型中不同部分的并行策略

        model = parallelize_module(model, tp_mesh, parallelize_plan)  # 对模型进行数据并行和时间并行的并行化
        model = replicate(model, device_mesh=dp_mesh)  # 复制模型到指定的设备网格

        compiled_replicate_model = parallelize_module(
            compiled_replicate_model, tp_mesh, parallelize_plan
        )  # 编译后复制模型进行数据并行和时间并行的并行化
        compiled_replicate_model = replicate(
            compiled_replicate_model, device_mesh=dp_mesh
        )  # 编译后复制模型复制到指定的设备网格

        compiled_replicate_model = torch.compile(compiled_replicate_model)  # 编译模型，准备进行优化

        data = torch.randn([1, DIM]).cuda()  # 生成在 CUDA 上运行的随机数据
        with compiled_autograd.enable(compiler_fn()):
            # 启用编译后的自动求导，用编译后的模型计算损失并进行反向传播
            loss = compiled_replicate_model(data).sum()
            loss.backward()

        # 使用原始模型计算损失并进行反向传播
        loss = model(data).sum()
        loss.backward()

        # 检查编译后的模型和原始模型的参数梯度是否一致
        for p1, p2 in zip(model.parameters(), compiled_replicate_model.parameters()):
            self.assertEqual(p1.grad, p2.grad)


if __name__ == "__main__":
    run_tests()
```