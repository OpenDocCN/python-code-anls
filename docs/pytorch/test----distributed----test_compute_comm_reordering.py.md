# `.\pytorch\test\distributed\test_compute_comm_reordering.py`

```py
# Owner(s): ["module: inductor"]

# 引入单元测试模块
import unittest
# 引入 patch 函数，用于模拟对象的装饰器
from unittest.mock import patch

# 引入 PyTorch 库及其私有模块
import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case

# 导入 torch.distributed._functional_collectives 时须注意，导入顺序必须在 dynamo 之后，以避免集合处理错误
import torch.distributed._functional_collectives as _functional_collectives
# 导入 FileCheck 类
from torch._C import FileCheck
# 导入 torch._dynamo.utils 模块中的 same 函数
from torch._dynamo.utils import same
# 导入 torch._inductor 模块中的 ir 模块
from torch._inductor import ir
# 导入 torch._inductor.comm_analysis 中的多个符号
from torch._inductor.comm_analysis import (
    baseLat,
    hwLat,
    llMaxBws,
    NCCL_ALGO,
    NCCL_HW,
    NCCL_PROTO,
    NVIDIA_GPU_TYPE,
)
# 导入 torch._inductor.utils 模块中的 run_and_get_triton_code 函数
from torch._inductor.utils import run_and_get_triton_code
# 导入 torch.testing._internal.common_distributed 模块中的多个符号
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
# 导入 torch.utils._triton 中的 has_triton 函数
from torch.utils._triton import has_triton


def get_snode_runtime_for_reorder_compute_test(snode):
    # NOTE: custom cost model to show that the compute reordering algorithm is working
    # 自定义成本模型以展示计算重排序算法的工作方式
    # Collective kernels
    if isinstance(snode.node, ir._CollectiveKernel):
        return 100
    # Wait kernels
    elif isinstance(snode.node, ir._WaitKernel):
        return 0
    # High-arithmetic-intensity compute kernels
    elif isinstance(snode.node, ir.ExternKernel):
        return 5
    # All other kernels
    # 所有其他类型的内核
    return 1


@requires_nccl()
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    在多进程运行器中运行正确性检查，标记要在最少 GPU 数量下运行的测试
    """

    def get_world_trs(self):
        # 返回世界传输对象的描述
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        # 无论有 2、3 或 4 个 GPU，都只在 2 个 GPU 上运行，以解决 skipif<2 和 GPU 数量不确定的工作进程问题
        return 2

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    # TODO: 不知何故，导致感应器后台编译线程在分布式工作析构时出现挂起
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
        ],
    )
    def test_sink_waits(self):
        # 定义一个内部函数 `func`，它接受参数 `a`，`tag`，`ranks`，`group_size`
        def func(a, *, tag, ranks, group_size):
            # 使用 `_functional_collectives.all_reduce` 函数对 `a` 执行全局归约操作，使用 "sum" 操作，指定 `ranks` 和 `tag`
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            # 对 `a` 应用 ReLU 激活函数，结果保存在 `c` 中
            c = torch.relu(a)
            # 计算矩阵乘法 `c` 和 `c`，结果保存在 `d` 中
            d = torch.matmul(c, c)
            # 将 `d` 和全局归约结果 `ar` 相加，结果保存在 `e` 中
            e = d + ar
            # 返回元组 `(e,)`
            return (e,)

        # 在分布式环境下初始化当前进程的分布式设置
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建一个大小为 (4, 4)、类型为 `torch.float`，位于 CUDA 设备上的张量 `inputs`，并添加当前进程的秩
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            # 编译函数 `func`，返回一个编译后的函数对象 `compiled`
            compiled = torch.compile(func)
            # 运行编译后的函数 `compiled`，并获取其在 Triton 中生成的代码，使用 `inputs` 和其他分布式参数
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: 注意 `_wait_tensor` 直到第一次使用之前都是延迟的
            # 在生成的 Triton 代码中检查 "dist.all_reduce("，"triton_poi_fused_relu"，"_wait_tensor(" 的存在
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "_wait_tensor("
            ).run(code)
            # 用 `inputs` 和其他分布式参数运行 `compiled` 函数，并将结果保存在 `out` 中
            out = compiled(inputs, **self.get_world_trs())
            # 使用 `func` 函数和其他分布式参数计算正确的输出，保存在 `correct` 中
            correct = func(inputs, **self.get_world_trs())
            # 断言 `out` 和 `correct` 相同
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: 某种方式，inductor 后台编译线程在分布式工作析构时会导致 hang
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "raise_comms",
        ],
    )
    def test_raise_comms(self):
        # 定义一个内部函数 `func`，它接受参数 `a`，`tag`，`ranks`，`group_size`
        def func(a, *, tag, ranks, group_size):
            # 对 `a` 应用 ReLU 激活函数，结果保存在 `c` 中
            c = torch.relu(a)
            # 计算矩阵乘法 `c` 和 `c`，结果保存在 `d` 中
            d = torch.matmul(c, c)
            # 使用 `_functional_collectives.all_reduce` 函数对 `a` 执行全局归约操作，使用 "sum" 操作，指定 `ranks` 和 `tag`，结果保存在 `ar` 中
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            # 将 `d` 和全局归约结果 `ar` 相加，结果保存在 `e` 中
            e = d + ar
            # 返回元组 `(e,)`
            return (e,)

        # 在分布式环境下初始化当前进程的分布式设置
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建一个大小为 (4, 4)、类型为 `torch.float`，位于 CUDA 设备上的张量 `inputs`，并添加当前进程的秩
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            # 编译函数 `func`，返回一个编译后的函数对象 `compiled`
            compiled = torch.compile(func)
            # 运行编译后的函数 `compiled`，并获取其在 Triton 中生成的代码，使用 `inputs` 和其他分布式参数
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: 注意 `dist.all_reduce` 被提升到 relu 和 matmul 之上
            # 在生成的 Triton 代码中检查 "dist.all_reduce("，"_wait_tensor("，"triton_poi_fused_relu"，"extern_kernels.addmm(" 的存在
            FileCheck().check("dist.all_reduce(").check("_wait_tensor(").check(
                "triton_poi_fused_relu"
            ).check("extern_kernels.addmm(").run(code)
            # 用 `inputs` 和其他分布式参数运行 `compiled` 函数，并将结果保存在 `out` 中
            out = compiled(inputs, **self.get_world_trs())
            # 使用 `func` 函数和其他分布式参数计算正确的输出，保存在 `correct` 中
            correct = func(inputs, **self.get_world_trs())
            # 断言 `out` 和 `correct` 相同
            self.assertTrue(same(out, correct))
    # 使用 `patch.object` 装饰器，模拟对 `torch._inductor.config` 的属性进行打补丁
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
            "raise_comms",
        ],
    )
    # 定义测试方法 `test_sink_waits_raise_comms`
    def test_sink_waits_raise_comms(self):
        # 定义内部函数 `func`，接受参数 `a`，`tag`，`ranks`，`group_size`
        def func(a, *, tag, ranks, group_size):
            # 计算 `a` 的 ReLU
            c = torch.relu(a)
            # 计算 `c` 的矩阵乘法
            d = torch.matmul(c, c)
            # 执行 all_reduce 操作，对 `a` 进行求和，结果存入 `ar`
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            # 计算 `d` 加上 `ar` 的结果，并赋给 `e`
            e = d + ar
            # 返回元组 `(e,)`
            return (e,)

        # 使用 `_dynamo_dist_per_rank_init` 上下文管理器，初始化分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建一个大小为 4x4 的 CUDA 张量 `inputs`，并初始化为全 1，加上 `self.rank`
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            # 编译函数 `func`，生成编译后的版本 `compiled`
            compiled = torch.compile(func)
            # 运行 `compiled` 函数，获取 Triton 代码，传入 `inputs` 和其他世界变换器
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # 使用 `FileCheck()` 对 `code` 进行检查，确保包含指定字符串，提示其关注点
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "_wait_tensor("
            ).check("extern_kernels.addmm(").run(code)
            # 执行 `compiled` 函数，将结果存入 `out`
            out = compiled(inputs, **self.get_world_trs())
            # 调用 `func` 函数，将结果存入 `correct`
            correct = func(inputs, **self.get_world_trs())
            # 断言 `out` 与 `correct` 结果一致
            self.assertTrue(same(out, correct))

    # 如果没有 Triton 或 GPU 架构过旧，则跳过该测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # 对 `torch._inductor.config` 的 `allow_buffer_reuse` 属性打补丁，设置为 `True`
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # 通过打补丁设置 `torch._inductor.config` 的 `compile_threads` 属性为 `1`
    @patch.object(torch._inductor.config, "compile_threads", 1)
    # 设置 `torch._inductor.config` 的 `reorder_for_compute_comm_overlap` 属性为 `True`
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    # 对 `torch._inductor.config` 的 `reorder_for_compute_comm_overlap_passes` 属性打补丁，
    # 设置其值为 `["reorder_compute_for_overlap"]`
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "reorder_compute_for_overlap",
        ],
    )
    def test_reorder_compute_for_overlap(self):
        # 定义内部函数func，接受参数a，tag，ranks，group_size
        def func(a, *, tag, ranks, group_size):
            # 执行全局归约操作，将a进行"sum"操作，并指定ranks和tag
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            # 计算矩阵乘积a和a，并赋值给g
            g = torch.matmul(a, a)
            # 计算a的ReLU激活，并赋值给c
            c = torch.relu(a)
            # 计算矩阵乘积c和c，并赋值给d
            d = torch.matmul(c, c)
            # 计算f，包括d、c、ar的乘积
            f = d * c * ar
            # 对f进行全局归约操作，将其进行"sum"操作，并指定ranks和tag，结果赋值给fr
            fr = _functional_collectives.all_reduce(f, "sum", ranks, tag)
            # 计算e，包括d、ar、fr和g的乘积和
            e = torch.matmul(d + ar + fr, g)
            return (e,)

        # 初始化_dynamo_dist_per_rank_init，并使用self.rank和self.world_size
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建输入张量inputs，尺寸为4x4，数据类型为torch.float，设备为"cuda"，并加上self.rank的值
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            # 编译func函数
            compiled = torch.compile(func)
            # 运行编译后的函数，生成triton代码，并传入inputs和self.get_world_trs()的参数
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: 在调度第一个全局归约之后：
            # 1. 首先调度不依赖于第一个全局归约但是对第二个全局归约有要求的操作(c和d)。
            # 2. 然后调度既不依赖于第一个全局归约也不对第二个全局归约有要求的操作(g)。
            # 3. 接着调度既依赖于第一个全局归约又对第二个全局归约有要求的操作(f)。
            # 最后，调度第二个全局归约，并调度所有依赖于第二个全局归约的操作。
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "extern_kernels.mm("
            ).check("extern_kernels.mm(").check("_wait_tensor(").check(
                "triton_poi_fused_mul"
            ).check(
                "dist.all_reduce("
            ).check(
                "_wait_tensor("
            ).check(
                "triton_poi_fused_add"
            ).check(
                "extern_kernels.mm("
            ).run(
                code
            )
            # 执行编译后的函数，传入inputs和self.get_world_trs()的参数，结果赋值给out
            out = compiled(inputs, **self.get_world_trs())
            # 执行func函数，传入inputs和self.get_world_trs()的参数，结果赋值给correct
            correct = func(inputs, **self.get_world_trs())
            # 断言out与correct相等
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "reorder_compute_for_overlap",
        ],
    )
    @patch.object(
        torch._inductor.config,
        "estimate_op_runtime",
        get_snode_runtime_for_reorder_compute_test,
    )
    def test_reorder_compute_for_overlap_custom_runtime_estimation(self):
        # 定义一个测试函数，用于评估重叠计算的自定义运行时估计
        def func(a, *, tag, ranks, group_size):
            # 执行全局归约操作，将张量 a 在 ranks 上求和，使用指定的标签
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            # 计算矩阵乘积 g = a @ a
            g = torch.matmul(a, a)
            # 计算张量 a 的 ReLU 激活函数
            c = torch.relu(a)
            # 计算矩阵乘积 d = c @ c
            d = torch.matmul(c, c)
            # 计算混合运算结果 f = d * c * ar
            f = d * c * ar
            # 再次执行全局归约操作，将混合运算结果 f 在 ranks 上求和，使用指定的标签
            fr = _functional_collectives.all_reduce(f, "sum", ranks, tag)
            # 计算复合运算 e = (d + ar + fr) @ g
            e = torch.matmul(d + ar + fr, g)
            # 返回结果元组
            return (e,)

        # 使用 _dynamo_dist_per_rank_init 上下文初始化分布环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建输入张量，元素均为 1，并根据当前进程的 rank 调整元素值，数据类型为 float，设备为 cuda
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            # 编译函数 func，生成优化后的代码
            compiled = torch.compile(func)
            # 运行并获取 Triton 代码
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: 调度第一个全局归约后：
            # 1. 首先调度不依赖于第一个全局归约但是对第二个全局归约有依赖的操作 (c 和 d)。
            # 2. 然后，调度不依赖于第一个全局归约且对第二个全局归约无依赖的操作 (g)。
            # 3. 接着，调度既依赖于第一个全局归约又对第二个全局归约有依赖的操作 (f)。
            # 最后，调度第二个全局归约，并调度所有依赖于第二个全局归约的操作。
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "extern_kernels.mm("
            ).check("extern_kernels.mm(").check("_wait_tensor(").check(
                "triton_poi_fused_mul"
            ).check(
                "dist.all_reduce("
            ).check(
                "_wait_tensor("
            ).check(
                "triton_poi_fused_add"
            ).check(
                "extern_kernels.mm("
            ).run(
                code
            )
            # 执行编译后的函数，并获取输出结果
            out = compiled(inputs, **self.get_world_trs())
            # 计算函数 func 的正确输出结果
            correct = func(inputs, **self.get_world_trs())
            # 断言输出结果与正确结果相同
            self.assertTrue(same(out, correct))

    def test_nccl_heuristics(self):
        # 断言 baseLat 的长度与 NCCL_ALGO 的长度相同
        assert len(baseLat) == len(NCCL_ALGO)
        # 断言 baseLat 中每个元素的长度与 NCCL_PROTO 中的元素长度相同
        assert all(len(x) == len(NCCL_PROTO) for x in baseLat)

        # 断言 hwLat 的长度与 NCCL_HW 的长度相同
        assert len(hwLat) == len(NCCL_HW)
        # 断言 hwLat 中每个元素的长度与 NCCL_ALGO 中的元素长度相同
        assert all(len(x) == len(NCCL_ALGO) for x in hwLat)
        # 断言 hwLat 中每个子列表的每个元素的长度与 NCCL_PROTO 中的元素长度相同
        assert all(len(y) == len(NCCL_PROTO) for x in hwLat for y in x)

        # 断言 llMaxBws 的长度与 NVIDIA_GPU_TYPE 的长度相同
        assert len(llMaxBws) == len(NVIDIA_GPU_TYPE)
# 如果当前脚本被直接执行（而不是作为模块被导入），则执行以下代码块
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的run_tests函数，用于执行测试用例
    run_tests()
```