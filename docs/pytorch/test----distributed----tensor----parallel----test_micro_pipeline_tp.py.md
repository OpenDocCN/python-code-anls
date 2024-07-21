# `.\pytorch\test\distributed\tensor\parallel\test_micro_pipeline_tp.py`

```py
# 导入必要的库和模块
import unittest

import torch
import torch.distributed as dist
from torch._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
from torch.distributed._functional_collectives import (
    all_gather_tensor,
    reduce_scatter_tensor,
)
from torch.distributed._symmetric_memory import _test_mode
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._triton import has_triton

# 实例化参数化测试的装饰器
@instantiate_parametrized_tests
class MicroPipelineTPTest(TestCase):
    # 设置测试环境
    def setUp(self):
        torch._inductor.config._micro_pipeline_tp = True  # 设置微流水线 TP 为 True

        self.rank = 0  # 当前进程的排名
        self.world_size = 2  # 总共的进程数
        torch.cuda.set_device("cuda:0")  # 设置当前 CUDA 设备为 cuda:0

        store = FakeStore()  # 创建一个 FakeStore 对象
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )  # 初始化进程组，使用 fake 后端，总进程数和当前进程排名

    # 测试结束后的清理工作
    def tearDown(self):
        dist.destroy_process_group()  # 销毁进程组

    # 如果没有 Triton 或者 GPU 架构不支持，跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("A_dims", [2, 3])
    @parametrize("gather_dim", [0, 1, 2])
    @fresh_inductor_cache()
    def test_fuse_all_gather_matmul(self, A_dims, gather_dim):
        if gather_dim >= A_dims:
            return  # 如果 gather_dim 大于等于 A_dims，则直接返回

        group = dist.group.WORLD  # 获取 WORLD 分组对象

        def func(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            A = all_gather_tensor(A_shard, gather_dim=gather_dim, group=group)  # 执行 all_gather 操作
            return A @ B  # 返回矩阵乘法结果

        # 根据 A_dims 设置 A_shard 的形状
        if A_dims == 2:
            A_shard_shape = [64, 32]
        elif A_dims == 3:
            A_shard_shape = [2, 64, 32]
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")  # 如果 A_dims 不是 2 或 3，抛出异常

        A_shard_shape[gather_dim] //= self.world_size  # 调整 A_shard_shape 的大小
        A_shard = torch.rand(*A_shard_shape, device="cuda")  # 在 CUDA 设备上生成随机数据 A_shard
        B = torch.rand(32, 16, device="cuda")  # 在 CUDA 设备上生成随机数据 B

        with _test_mode():  # 使用测试模式
            compiled = torch.compile(func)  # 编译 func 函数
            code = run_and_get_triton_code(compiled, A_shard, B)  # 运行 Triton 并获取代码

        if gather_dim == A_dims - 1:
            assert "fused_all_gather_matmul" not in code  # 如果 gather_dim 等于 A_dims - 1，断言不包含 "fused_all_gather_matmul"
            assert "all_gather_into_tensor" in code  # 断言包含 "all_gather_into_tensor"
        else:
            assert "fused_all_gather_matmul" in code  # 否则，断言包含 "fused_all_gather_matmul"
            assert "all_gather_into_tensor" not in code  # 断言不包含 "all_gather_into_tensor"

    # 如果没有 Triton 或者 GPU 架构不支持，跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("A_dims", [2, 3])
    @parametrize("scatter_dim", [0, 1, 2])
    @fresh_inductor_cache()
    # 定义测试方法，用于测试融合矩阵乘法和reduce_scatter操作
    def test_fuse_matmul_reduce_scatter(self, A_dims, scatter_dim):
        # 如果scatter_dim大于等于A_dims，则直接返回，不进行后续操作
        if scatter_dim >= A_dims:
            return

        # 设置通信组为WORLD
        group = dist.group.WORLD

        # 定义函数func，接收两个Tensor A和B，返回A @ B经过reduce_scatter操作后的结果
        def func(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return reduce_scatter_tensor(A @ B, "avg", scatter_dim, group)

        # 根据A_dims的不同情况，生成不同形状的随机Tensor A
        if A_dims == 2:
            A = torch.rand(64, 32, device="cuda")
        elif A_dims == 3:
            A = torch.rand(2, 64, 32, device="cuda")
        else:
            # 如果A_dims既不是2也不是3，则抛出AssertionError异常
            raise AssertionError(f"Invalid A_dims: {A_dims}")
        
        # 生成形状为(32, 16)的随机Tensor B
        B = torch.rand(32, 16, device="cuda")

        # 进入测试模式
        with _test_mode():
            # 编译func函数
            compiled = torch.compile(func)
            # 运行编译后的函数，获取其在Triton上的代码
            code = run_and_get_triton_code(compiled, A, B)

        # 断言Triton代码中包含"fused_matmul_reduce_scatter"，且不包含"reduce_scatter_tensor"
        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)

    # 标记为unittest.skipIf，如果没有Triton或者GPU架构较老，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # 参数化测试，测试不同的shard_dim值
    @parametrize("shard_dim", [0, 1])
    # 使用fresh_inductor_cache装饰器，确保每次测试都使用新的inductor缓存
    @fresh_inductor_cache()
    # 定义测试方法，测试分布式张量操作中的序列并行化
    def test_dtensor_seq_par(self, shard_dim: int):
        # 创建MLP模型，设备为cuda，无偏置项
        model = MLPModule(device="cuda", bias=False)
        # 创建DeviceMesh对象，设备为cuda，包含从0到self.world_size的设备列表
        device_mesh = DeviceMesh(
            "cuda",
            torch.arange(0, self.world_size),
        )
        # 定义并行化计划，将"net1"和"net2"分别设置为列并行和行并行，输入布局使用shard_dim分片
        parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(shard_dim)),
            "net2": RowwiseParallel(output_layouts=Shard(shard_dim)),
        }
        # 对模型进行并行化处理，使用device_mesh和parallelize_plan
        model = parallelize_module(model, device_mesh, parallelize_plan)
        
        # 根据shard_dim的不同情况，生成不同形状的随机Tensor inp
        if shard_dim == 0:
            inp = torch.rand(8, 10, device="cuda")
        elif shard_dim == 1:
            inp = torch.rand(2, 8, 10, device="cuda")
        else:
            # 如果shard_dim既不是0也不是1，则抛出AssertionError异常
            raise AssertionError("Invalid shard_dim")

        # 进入测试模式
        with _test_mode():
            # 编译模型
            compiled = torch.compile(model)
            # 运行编译后的模型，获取其在Triton上的代码
            code = run_and_get_triton_code(compiled, inp)

        # 断言Triton代码中包含"fused_all_gather_matmul"，不包含"all_gather_into_tensor"和"reduce_scatter_tensor"
        self.assertIn("fused_all_gather_matmul", code)
        self.assertNotIn("all_gather_into_tensor", code)
        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)
# 如果当前模块被直接运行而非被导入作为一个模块，执行下面的代码块
if __name__ == "__main__":
    # 调用 run_tests 函数，通常用于执行单元测试或验证模块功能
    run_tests()
```