# `.\pytorch\test\distributed\test_inductor_collectives.py`

```py
# 模块所有者信息，指出这段代码的归属
import functools  # 导入 functools 模块，用于高阶函数操作
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from unittest.mock import patch  # 从 unittest.mock 模块导入 patch 函数，用于模拟对象

import torch  # 导入 PyTorch 库
import torch._dynamo  # 导入 torch._dynamo 模块
import torch._dynamo.logging  # 导入 torch._dynamo.logging 模块
import torch._dynamo.test_case  # 导入 torch._dynamo.test_case 模块

# 由于某些原因，在导入 dynamo 后导入 functional collectives 会导致集合处理出现问题！
import torch.distributed._functional_collectives as _functional_collectives  # 导入 torch.distributed._functional_collectives 模块，用于分布式功能集合处理
from torch._C import FileCheck  # 从 torch._C 模块导入 FileCheck 类
from torch._dynamo.testing import CompileCounter  # 导入 torch._dynamo.testing 模块的 CompileCounter 类
from torch._dynamo.utils import same  # 导入 torch._dynamo.utils 模块的 same 函数
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx  # 导入 torch._inductor.compile_fx 模块的 compile_fx 函数，并重命名为 inductor_compile_fx
from torch._inductor.utils import run_and_get_triton_code  # 导入 torch._inductor.utils 模块的 run_and_get_triton_code 函数
from torch.distributed.distributed_c10d import GroupMember  # 导入 torch.distributed.distributed_c10d 模块的 GroupMember 类
from torch.fx.experimental.proxy_tensor import make_fx  # 导入 torch.fx.experimental 模块的 make_fx 函数
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,  # 导入 torch.testing._internal.common_distributed 模块的 _dynamo_dist_per_rank_init 函数
    DynamoDistributedMultiProcTestCase,  # 导入 torch.testing._internal.common_distributed 模块的 DynamoDistributedMultiProcTestCase 类
    DynamoDistributedSingleProcTestCase,  # 导入 torch.testing._internal.common_distributed 模块的 DynamoDistributedSingleProcTestCase 类
    requires_nccl,  # 导入 torch.testing._internal.common_distributed 模块的 requires_nccl 装饰器函数
    skip_if_lt_x_gpu,  # 导入 torch.testing._internal.common_distributed 模块的 skip_if_lt_x_gpu 装饰器函数
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入 torch.testing._internal.common_utils 模块的 instantiate_parametrized_tests 函数
    parametrize,  # 导入 torch.testing._internal.common_utils 模块的 parametrize 装饰器函数
    requires_cuda,  # 导入 torch.testing._internal.common_utils 模块的 requires_cuda 装饰器函数
)
from torch.utils._triton import has_triton  # 导入 torch.utils._triton 模块的 has_triton 函数，检查是否有 Triton 支持


def _tolist_with_constrain_as_size(tensor):
    lst = tensor.tolist()  # 将张量转换为 Python 列表
    for elem in lst:
        torch._check_is_size(elem)  # 对列表中的每个元素检查其是否为有效大小
    return lst  # 返回处理后的列表


@requires_nccl()  # 使用 requires_nccl 装饰器，要求 NCCL 支持
class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    """
    在多进程运行器中运行正确性检查，标记为最少需要运行的 GPU 数量
    """

    def get_world_trs(self):
        return {
            "tag": "",  # 返回空标签
            "ranks": list(range(self.world_size)),  # 返回当前世界大小的排名列表
            "group_size": self.world_size,  # 返回当前世界大小作为组大小
        }

    @property
    def world_size(self) -> int:
        # hack: 无论我们有 2 或 3 或 4 个 GPU，都只在 2 个上运行
        # 解决了跳过小于 2 且工作人员具有不可预测数量 GPU 的问题
        return 2

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allreduce_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        # 定义一个函数，执行矩阵乘法、拼接和全局归约操作
        def matmul_cat_col(a, b, c, d, e, f, *, tag, ranks, group_size):
            # 计算矩阵 a 和 b 的乘积
            x = torch.matmul(a, b)
            # 计算矩阵 c 和 d 的乘积
            y = torch.matmul(c, d)
            # 将 x 和 y 拼接在一起
            z = torch.cat((x, y))
            # 对拼接后的张量 z 执行全局归约操作，使用 "sum" 运算
            ar = torch.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            # 计算矩阵 e 和 f 的乘积
            g = torch.matmul(e, f)
            # 等待全局归约操作的结果
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # 将 g 重复拼接到 ar 上，并返回结果
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        # 定义一个函数，编译给定的函数并返回编译后的结果
        def compile(func, example_inputs):
            # 使用输入示例生成函数的 FX 图
            graph = make_fx(func)(*example_inputs)
            # 使用 Inductor 编译 FX 图，并返回编译后的函数
            return inductor_compile_fx(graph, example_inputs)

        # 使用动态分布初始化，根据当前进程和总进程数初始化环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 使用 functools.partial 绑定 matmul_cat_col 函数的部分参数
            matmul_cat_col = functools.partial(
                matmul_cat_col,
                **self.get_world_trs(),
            )
            # 创建输入张量，如果当前进程是第一个进程，则使用随机张量 t，否则使用零张量
            t = torch.randn(4, 4, device="cuda")
            inputs = (t if self.rank == 0 else torch.zeros(4, 4, device="cuda"),) * 6
            # 在 eager 模式下执行 matmul_cat_col 函数
            eager_out = matmul_cat_col(*inputs)
            # 编译 matmul_cat_col 函数并执行
            compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
            # 在 Inductor 模式下执行编译后的函数
            inductor_out = compiled_matmul_cat_col(*inputs)
            # 断言 eager 模式和 Inductor 模式下的输出结果相同，允许误差为 0.001
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))
    def test_allreduce_inductor_cudagraph_trees(self):
        """
        Tests whether cudagraph trees support all_reduce from nccl
        """
        import torch.distributed as dist

        # dist.all_reduce is an inplace op in eager mode but a functionalized op in compiled mode.
        # so we define eager_func and func separately for the same semantic.
        def eager_func(x):
            # Calculate y = x * x
            y = x * x
            # Perform inplace all_reduce operation on y using dist.ReduceOp.SUM
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            # Apply torch's silu (Sigmoid Linear Unit) activation function on x
            x = torch.nn.functional.silu(x)
            # Return the result of x * y
            return x * y

        def func(x):
            # Calculate y = x * x
            y = x * x
            # Perform functionalized all_reduce operation on y using dist.ReduceOp.SUM
            y = dist.all_reduce(y, op=dist.ReduceOp.SUM)
            # Apply torch's silu (Sigmoid Linear Unit) activation function on x
            x = torch.nn.functional.silu(x)
            # Return the result of x * y
            return x * y

        options = {
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": True,
        }

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Compile the function 'func' using Torch's compiler with specific options
            compiled_func = torch.compile(
                func, backend="inductor", fullgraph=True, options=options, dynamic=None
            )

            # Iterate over different values of nelem
            for nelem in [1024, 2048, 4096]:
                # Generate random tensor x on GPU
                x = torch.randn(nelem, device="cuda", dtype=torch.bfloat16)
                # Compute the golden output using eager_func
                golden_out = eager_func(x)

                # Repeat the following comparison three times
                for _ in range(3):
                    # Compute output of compiled_func using input x
                    compiled_out = compiled_func(x)
                    # Assert that the compiled output matches the golden output
                    self.assertEqual(golden_out, compiled_out)

    def test_c10d_functional_tagged_pt2_compliant(self):
        # Access the default all_reduce operation from _c10d_functional and assert Tag.pt2_compliant_tag
        op = torch.ops._c10d_functional.all_reduce.default
        self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)
        # Access the default all_reduce operation from c10d_functional and assert Tag.pt2_compliant_tag
        op = torch.ops.c10d_functional.all_reduce.default
        self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # 定义测试函数 test_eager_allreduce_inductor_wait
    def test_eager_allreduce_inductor_wait(self):
        
        # 定义内部函数 eager_func，接受参数 a, b, c, d 以及关键字参数 tag, ranks, group_size
        def eager_func(a, b, c, d, *, tag, ranks, group_size):
            # 计算矩阵乘积 a 和 b，结果存储在 x 中
            x = torch.matmul(a, b)
            # 计算矩阵乘积 c 和 d，结果存储在 y 中
            y = torch.matmul(c, d)
            # 将 x 和 y 拼接成新的张量 z
            z = torch.cat((x, y))
            # 执行全局归约操作，将 z 中所有元素求和，结果存储在 ar 中
            ar = torch.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            return ar
        
        # 定义内部函数 inductor_func，接受参数 ar, e, f
        def inductor_func(ar, e, f):
            # 计算矩阵乘积 e 和 f，结果存储在 g 中
            g = torch.matmul(e, f)
            # 等待张量 ar 就绪
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # 将 g 重复扩展成与 ar 相同的形状，然后与 ar 相加，结果存储在 out 中
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)
        
        # 定义编译函数 compile，接受参数 func 和 example_inputs
        def compile(func, example_inputs):
            # 使用 make_fx 创建 func 的计算图 graph
            graph = make_fx(func)(*example_inputs)
            # 对 graph 和 example_inputs 进行编译，返回编译后的函数
            return inductor_compile_fx(graph, example_inputs)
        
        # 使用 _dynamo_dist_per_rank_init 初始化分布式环境，使用 self.rank 和 self.world_size
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 使用 functools.partial 部分应用 eager_func，设置关键字参数为 self.get_world_trs() 返回值
            eager_func = functools.partial(
                eager_func,
                **self.get_world_trs(),
            )
            # 创建 eager_inputs，包含多个 torch.ones 矩阵，设备为 "cuda"，并加上 self.rank
            eager_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 4
            # 创建 inductor_inputs，包含多个 torch.ones 矩阵，设备为 "cuda"，并加上 self.rank，数量为 2
            inductor_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2
            
            # 调用 inductor_func，传入 eager_func(*eager_inputs) 和 inductor_inputs，存储结果在 eager_out 中
            eager_out = inductor_func(eager_func(*eager_inputs), *inductor_inputs)
            
            # 编译 inductor_func，传入 [eager_func(*eager_inputs)] + list(inductor_inputs)，存储结果在 compiled_inductor_func 中
            compiled_inductor_func = compile(
                inductor_func, [eager_func(*eager_inputs)] + list(inductor_inputs)
            )
            # 调用编译后的 inductor_func，传入 eager_func(*eager_inputs) 和 inductor_inputs，存储结果在 inductor_out 中
            inductor_out = compiled_inductor_func(
                eager_func(*eager_inputs), *inductor_inputs
            )
            # 打印 eager_out 和 inductor_out 的值
            print(f"eager_out, {eager_out}")
            print(f"inductor_out, {inductor_out}")
            # 使用 self.assertTrue 检查 eager_out 和 inductor_out 是否近似相等，容差为 0.001
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    # 如果没有 Triton 或 GPU 架构较旧，则跳过此测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # 定义测试函数 test_inductor_allreduce_eager_wait，用于测试异步计算的处理方式
    def test_inductor_allreduce_eager_wait(self):
        # 定义一个在 Inductor 模式下执行的函数 inductor_func，接收多个参数和关键字参数
        def inductor_func(a, b, c, d, *, tag, ranks, group_size):
            # 计算矩阵乘积 a*b，并将结果存储在 x 中
            x = torch.matmul(a, b)
            # 计算矩阵乘积 c*d，并将结果存储在 y 中
            y = torch.matmul(c, d)
            # 将 x 和 y 拼接在一起，形成新的张量 z
            z = torch.cat((x, y))
            # 调用 C10D 库中的 all_reduce 函数，对张量 z 进行全局求和操作，返回结果 ar
            ar = torch.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            # 返回 all_reduce 操作的结果 ar
            return ar

        # 定义一个在 Eager 模式下执行的函数 eager_func，接收 ar 和两个张量 e、f 作为参数
        def eager_func(ar, e, f):
            # 计算矩阵乘积 e*f，并将结果存储在 g 中
            g = torch.matmul(e, f)
            # 等待异步操作 ar 完成
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # 将张量 g 重复两次，然后加到 ar 上，得到结果 out
            out = torch.add(ar, g.repeat(2, 1))
            # 返回包含 out 的元组
            return (out,)

        # 定义一个编译函数 compile，用于生成计算图并进行静态编译
        def compile(func, example_inputs):
            # 使用输入示例 example_inputs 构建函数 func 的计算图
            graph = make_fx(func)(*example_inputs)
            # 对构建的计算图进行静态编译，返回编译后的函数
            return inductor_compile_fx(graph, example_inputs)

        # 在分布式环境下初始化每个进程的状态
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 使用 functools.partial 将 inductor_func 与 self.get_world_trs() 部分绑定，生成新的函数 inductor_func
            inductor_func = functools.partial(
                inductor_func,
                **self.get_world_trs(),
            )
            # 构建 inductor_func 的输入参数元组 inductor_inputs
            inductor_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 4
            # 构建 eager_func 的输入参数元组 eager_inputs
            eager_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            # 执行 eager_func(inductor_func(*inductor_inputs), *eager_inputs)，得到 eager 模式下的输出 eager_out
            eager_out = eager_func(inductor_func(*inductor_inputs), *eager_inputs)
            # 对 inductor_func 进行静态编译，得到编译后的函数 compiled_inductor_func
            compiled_inductor_func = compile(inductor_func, inductor_inputs)
            # 执行 eager_func(compiled_inductor_func(*inductor_inputs), *eager_inputs)，得到静态编译模式下的输出 inductor_out
            inductor_out = eager_func(
                compiled_inductor_func(*inductor_inputs), *eager_inputs
            )
            # 断言 eager_out 与 inductor_out 相同，误差容限为 0.001
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    # 使用 unittest.skipIf 装饰器，若没有 Triton 或 GPU 架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # 使用 patch.object 修改 torch._inductor.config 中的 allow_buffer_reuse 属性为 True
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # 定义测试函数 test_allreduce_input_buffer_reuse，用于测试输入缓冲区的重用
    def test_allreduce_input_buffer_reuse(self):
        # 定义函数 func，接收张量 a 和若干关键字参数
        def func(a, *, tag, ranks, group_size):
            # 对输入张量 a 执行全局求和操作，结果存储在 ar 中
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            # 计算张量 a 的 ReLU，并存储在 c 中
            c = torch.relu(a)
            # 计算矩阵乘积 c*c，并存储在 d 中
            d = torch.matmul(c, c)
            # 将 ar 加到 d 上，并存储在 e 中
            e = d + ar
            # 返回包含 e 的元组
            return (e,)

        # 在分布式环境下初始化每个进程的状态
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 构建输入张量 inputs，所有元素为 1，并在 GPU 上分配存储空间
            inputs = torch.ones(4, 4, device="cuda") + self.rank
            # 编译函数 func，返回编译后的函数对象 compiled
            compiled = torch.compile(func)
            # 使用 compiled 执行 inputs 和 self.get_world_trs()，得到执行结果 out
            out = compiled(inputs, **self.get_world_trs())
            # 直接调用 func 执行 inputs 和 self.get_world_trs()，得到正确结果 correct
            correct = func(inputs, **self.get_world_trs())
            # 断言 out 与 correct 相同
            self.assertTrue(same(out, correct))

    # 使用 unittest.skipIf 装饰器，若没有 Triton 或 GPU 架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_permute_tensor(self):
        def func(tensor, src_dst_pairs, *, tag, ranks, group_size):
            # 调用 _functional_collectives.permute_tensor 函数进行张量排列操作
            return _functional_collectives.permute_tensor(
                tensor, src_dst_pairs, ranks, tag
            )

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 设置输入数据
            inputs = (
                # rank0: [0., 1.], rank1: [2., 3.]
                torch.arange(2, dtype=torch.float32, device="cuda") + 2 * self.rank,
                [1, 0],
            )
            # 编译函数 func
            compiled = torch.compile(func)
            # 调用编译后的函数计算输出
            out = compiled(*inputs, **self.get_world_trs())
            # 直接调用原始函数 func 计算正确结果
            correct = func(*inputs, **self.get_world_trs())
            # 断言输出与正确结果相同
            self.assertTrue(same(out, correct))

            # 计算预期结果
            # rank0: [2., 3.], rank1: [0., 1.]
            expected = torch.arange(2, dtype=torch.float32, device="cuda") + 2 * (
                (self.rank - 1 + self.world_size) % self.world_size
            )
            # 断言输出与预期结果相同
            self.assertEqual(out, expected)
            self.assertEqual(correct, expected)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    def test_allgather_output_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                # 初始化模型，包括一个嵌入层
                self.emb = torch.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                # 模型前向传播
                y = self.emb(x)
                last_dim = y.dim() - 1
                # 执行全局收集操作并拼接结果
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = torch.cat(torch.chunk(res, world_size, dim=0), dim=last_dim)
                return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建模型实例并移动到 GPU
            model = Model().cuda()
            # 编译模型
            model_compiled = torch.compile(model)
            # 准备输入数据
            inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device="cuda")
            # 使用编译后的模型计算输出
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            # 直接调用原始模型计算正确结果
            correct = model(inp, self.world_size, **self.get_world_trs())
            # 断言输出与正确结果相同
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allgather_contiguous_input(self):
        class Model(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                # 调用父类初始化方法
                super().__init__(*args, **kwargs)
                # 创建一个大小为4的Embedding层，每个embedding向量大小为4
                self.emb = torch.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                # 对输入进行embedding操作
                y = self.emb(x)
                # 获取张量的最后一个维度，用于后续操作
                last_dim = y.dim() - 1
                # 转置张量，使得0维度和最后一个维度交换位置，并保证内存连续性
                y = y.transpose_(0, last_dim).contiguous()
                # 调用all_gather_tensor函数，进行张量的全局收集操作
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                # 再次转置张量，恢复原始形状，并保证内存连续性
                out = y.transpose_(0, last_dim).contiguous()
                # 返回输出张量
                return out

        # 使用_dynamo_dist_per_rank_init方法初始化分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 创建一个Model实例并移动到GPU上
            model = Model().cuda()
            # 编译模型
            model_compiled = torch.compile(model)
            # 创建输入张量inp，指定数据类型和设备
            inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device="cuda")
            # 调用编译后的模型进行推理
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            # 获取模型的预期输出
            correct = model(inp, self.world_size, **self.get_world_trs())
            # 使用assertTrue断言判断out和correct是否相同
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allgather_into_tensor_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        def example(a, b, *, tag, ranks, group_size):
            # 计算张量a和b的矩阵乘法
            c = torch.matmul(a, b)
            # 使用torch.ops.c10d_functional.all_gather_into_tensor进行张量的全局收集
            ag = torch.ops.c10d_functional.all_gather_into_tensor(
                c, tag, ranks, group_size
            )
            # 等待张量的操作完成
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            # 返回全局收集后的结果
            return (ag,)

        def compile(func, example_inputs):
            # 使用make_fx创建函数的FX图
            graph = make_fx(func)(*example_inputs)
            # 使用inductor_compile_fx编译FX图
            return inductor_compile_fx(graph, example_inputs)

        # 使用_dynamo_dist_per_rank_init方法初始化分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 使用functools.partial创建example函数的偏函数
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            # 创建输入张量inputs
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            # 使用例子函数进行计算
            eager_out = example(*inputs)
            # 编译matmul/cat/collect的计算过程
            compiled_matmul_cat_col = compile(example, inputs)
            # 使用编译后的计算过程进行计算
            inductor_out = compiled_matmul_cat_col(*inputs)
            # 使用assertTrue断言判断eager_out和inductor_out是否相同，允许的误差为0.001
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试 reduce_scatter_tensor_inductor 函数
    def test_reduce_scatter_tensor_inductor(self):
        # 定义一个示例函数 example，接受参数 a, b 和关键字参数 tag, ranks, group_size
        def example(a, b, *, tag, ranks, group_size):
            # 计算矩阵 a 和 b 的乘积
            c = torch.matmul(a, b)
            # 调用 C10D 库中的 reduce_scatter_tensor 函数，对张量 c 执行求和操作
            ag = torch.ops.c10d_functional.reduce_scatter_tensor(
                c, "sum", tag, ranks, group_size
            )
            # 等待张量计算完成
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            # 返回一个包含 ag 的元组
            return (ag,)

        # 定义一个编译函数 compile，接受函数 func 和示例输入 example_inputs
        def compile(func, example_inputs):
            # 使用 Torch 的 FX 模块创建函数 func 的计算图
            graph = make_fx(func)(*example_inputs)
            # 使用 inductor_compile_fx 函数编译计算图并返回编译后的函数
            return inductor_compile_fx(graph, example_inputs)

        # 使用 _dynamo_dist_per_rank_init 方法初始化分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 使用 functools.partial 对 example 函数进行部分参数绑定
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            # 准备输入数据 inputs，包括两个相同的张量（4x4 的全 1 张量加上当前进程的排名）
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            # 在 eager 模式下运行 example 函数并记录结果
            eager_out = example(*inputs)
            # 编译 example 函数并获取编译后函数的输出
            compiled_fn = compile(example, inputs)
            inductor_out = compiled_fn(*inputs)
            # 使用 assertTrue 检查 eager 模式和编译模式下的输出是否一致（误差容限为 0.001）
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))
    
    # 使用 unittest.skipIf 装饰器，如果没有安装 Triton 或 GPU 架构较老则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    # 定义一个嵌套函数 example，用于执行 All-to-All 操作
    def test_all_to_all_single_inductor(self):
        def example(
            inp,
            input_split_sizes_tensor,
            output_split_sizes_tensor,
            *,
            tag,
            ranks,
            group_size,
        ):
            # 将输入的 input_split_sizes_tensor 转换为列表
            input_split_sizes = _tolist_with_constrain_as_size(input_split_sizes_tensor)
            # 将输出的 output_split_sizes_tensor 转换为列表
            output_split_sizes = _tolist_with_constrain_as_size(
                output_split_sizes_tensor
            )
            # 执行 Torch 的 C10D 函数库中的 All-to-All 单一操作
            a2a = torch.ops.c10d_functional.all_to_all_single(
                inp,
                output_split_sizes,
                input_split_sizes,
                tag,
                ranks,
                group_size,
            )
            # 等待异步张量操作的完成
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            # 对结果进行归一化处理
            out = a2a / a2a.sum(dim=0)
            return out

        # 使用 _dynamo_dist_per_rank_init 初始化每个排名的 Dynamo 分布
        # 使用 torch._dynamo.config.patch 设置动态形状和其他选项
        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size
        ), torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            # 计算行数作为输入数据的基数
            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            # 创建 input_split_sizes_tensor 张量，用于指定输入的分割大小
            input_split_sizes_tensor = torch.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=torch.int64,
            )
            # 创建 output_split_sizes_tensor 张量，用于指定输出的分割大小
            output_split_sizes_tensor = torch.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=torch.int64,
            )
            # 创建输入数据元组，包括输入张量、输入分割大小张量和输出分割大小张量
            inputs = (
                torch.ones(int(row), 5, device="cuda") * (self.rank + 1),
                input_split_sizes_tensor,
                output_split_sizes_tensor,
            )
            # 获取世界范围的 trs（可能是某种参数或配置信息）
            trs = self.get_world_trs()

            # 编译 example 函数，并生成完整图形和动态执行
            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            # 运行编译后的函数并获取 Triton 代码
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            (
                FileCheck()
                .check_regex(
                    "torch.ops._c10d_functional.all_to_all_single.default\\("
                    "arg\\d+_\\d+, "
                    "\\[u\\d+, u\\d+\\], "
                    "\\[u\\d+, u\\d+\\]"
                )
                .run(code)
            )

            # 直接执行 example 函数并获取 eager_out 结果
            eager_out = example(*inputs, **trs)
            # 执行编译后的函数并获取 inductor_out 结果
            inductor_out = compiled_fn(*inputs, **trs)
            # 断言 eager_out 和 inductor_out 结果相似，允许的误差为 0.001
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    # 根据系统是否安装了 Triton 和是否支持最新的 GPU 架构来跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # 定义测试方法，用于测试 all_to_all_single_inductor_split_sizes_none 函数
    def test_all_to_all_single_inductor_split_sizes_none(self):
        # 定义示例函数 example，接收输入 inp 和关键字参数 tag、ranks、group_size
        def example(inp, *, tag, ranks, group_size):
            # 调用 torch 的自定义操作 all_to_all_single，执行所有到所有的通信操作
            a2a = torch.ops.c10d_functional.all_to_all_single(
                inp,
                None,  # 不指定输出缓冲区
                None,  # 不指定输入缓冲区
                tag,  # 通信标签
                ranks,  # 参与通信的 ranks
                group_size,  # 分组大小
            )
            # 等待张量通信操作完成
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            # 对 a2a 进行归一化处理
            out = a2a / a2a.sum(dim=0)
            return out

        # 使用 _dynamo_dist_per_rank_init 方法初始化环境，设置当前 rank 和 world_size
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 生成输入数据 inputs，每个元素为在 GPU 上的 rank + 1 的张量
            inputs = (
                torch.ones(self.world_size, self.world_size, device="cuda")
                * (self.rank + 1),
            )
            # 获取 world_trs，可能是某种通信相关的配置
            trs = self.get_world_trs()

            # 编译 example 函数，生成动态图的编译版本
            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            # 运行编译后的函数，并获取 Triton 代码
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            (
                FileCheck()
                # 使用正则表达式检查 Triton 代码中的特定字符串，验证函数调用形式
                .check_regex(
                    "torch.ops._c10d_functional.all_to_all_single.default\\("
                    "arg\\d+_\\d+, "
                    "\\[\\(s\\d+ // \\d\\), \\(s\\d+ // \\d\\)\\], "
                    "\\[\\(s\\d+ // \\d\\), \\(s\\d+ // \\d\\)\\]"
                )
                .run(code)
            )

            # 直接调用 example 函数，获取其返回结果 eager_out
            eager_out = example(*inputs, **trs)
            # 调用编译后的函数，获取其返回结果 inductor_out
            inductor_out = compiled_fn(*inputs, **trs)
            # 使用 self.assertTrue 断言 eager_out 与 inductor_out 结果相同，容忍度为 0.001
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))
@instantiate_parametrized_tests
@requires_nccl()
@requires_cuda
class TestCollectivesInductor(DynamoDistributedSingleProcTestCase):
    """
    Prefer single-proc test runner for basic tests as it is easier to work with.
    """

    def get_world_trs(self, world_size=1):
        # 返回一个包含特定配置的测试环境字典
        return {
            "tag": "",
            "ranks": list(range(world_size)),
            "group_size": world_size,
        }

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(debug=True)
    def test_inductor_single_op(self):
        def func(inp, *, tag, ranks, group_size):
            # 执行分布式操作，将输入张量在指定 ranks 中的进程上求和
            ar = torch.ops.c10d_functional.all_reduce(
                inp, "sum", tag, ranks, group_size
            )
            # 等待张量操作完成
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            return ar

        inputs = torch.ones(4, 4, device="cuda")

        # 编译函数 func 以优化执行性能
        compiled = torch.compile(func)
        # 使用编译后的函数进行计算
        out = compiled(inputs, **self.get_world_trs())
        # 获取 Triton 编译后的代码
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unneccessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check(".run(arg0_1, buf0, 16")
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("return (buf0")
            .run(code)
        )
        # 对比计算结果和预期结果
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(debug=True)
    def test_inductor_steal_buffer(self):
        """
        it's ok and optimal if inductor allreduce mutates the buffer of an intermediate
        that isn't going to be used again
        """

        # 定义测试函数，验证在缓冲区不再使用的情况下，感应器全局归约可以修改中间缓冲区的行为是否正常和最优

        def func(inp, *, tag, ranks, group_size):
            # 对输入张量增加1
            x = inp + 1
            # 进行全局归约操作
            ar = torch.ops.c10d_functional.all_reduce(x, "sum", tag, ranks, group_size)
            # 等待张量操作完成
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # 确保 other 没有错误地引用 ar 的缓冲区
            other = torch.ones_like(inp) + 22
            return ar, other

        # 创建输入张量
        inputs = torch.ones(4, 4, device="cuda")

        # 编译函数 func
        compiled = torch.compile(func)
        # 运行并获取 Triton 代码
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check(".run(arg0_1, buf0")
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("buf5 = empty_strided")
            .check(".run(buf5, 16")
            .check("return (buf0, buf5")
            .run(code)
        )
        # 执行编译后的函数，与预期结果比较
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        # 断言输出与预期结果相同
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_doesnt_mutate_shared(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        # 确保将要重用的中间值不被修改，除非被复制

        def func(inp, *, tag, ranks, group_size):
            # 对输入张量增加1
            x = inp + 1
            # 进行全局归约操作
            ar = torch.ops.c10d_functional.all_reduce(x, "sum", tag, ranks, group_size)
            # 对 x 增加2，生成新的张量 y
            y = x + 2
            # 等待张量操作完成
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # 确保 other 没有错误地引用 ar 的缓冲区
            other = torch.ones_like(inp) + 22
            return ar, y, other

        # 创建输入张量
        inputs = torch.ones(4, 4, device="cuda")

        # 编译函数 func
        compiled = torch.compile(func)
        # 运行并获取 Triton 代码
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unneccessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf5 = empty_strided")
            .check(".run(arg0_1, buf0, buf5, 16")
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("buf6 = empty_strided")
            .check(".run(buf6, 16")
            .check("return (buf0, buf5, buf6")
            .run(code)
        )
        # 执行编译后的函数，与预期结果比较
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        # 断言输出与预期结果相同
        self.assertTrue(same(out, correct))
    # 定义测试方法：测试动态追踪全局归约操作
    def test_dynamo_trace_allreduce(self):
        # 定义内部函数 func，接收输入 inp，并对其执行全局归约操作
        def func(inp):
            ar = _functional_collectives.all_reduce(inp, "sum", "0")
            return ar
        
        # 创建大小为 4x4 的全一张量，放置在 CUDA 设备上作为输入
        inputs = torch.ones(4, 4, device="cuda")
        # 创建编译计数器对象
        counter = CompileCounter()
        # 使用编译器对 func 进行编译，同时使用 counter 作为后端
        compiled = torch.compile(func, backend=counter)
        # 执行编译后的函数并记录输出
        out = compiled(inputs)
        # 调用未编译的 func 函数计算正确的输出
        correct = func(inputs)
        # 断言帧计数为 1
        self.assertEqual(counter.frame_count, 1)

        # 断言操作计数为 2，代表了全局归约操作和等待操作
        self.assertEqual(counter.op_count, 2)
        # 检查输出是否与正确输出相同
        self.assertTrue(same(out, correct))

    # 定义测试方法：测试动态追踪全局聚集张量操作
    def test_dynamo_trace_all_gather_tensor(self):
        # 定义内部函数 func，接收输入 inp，并对其执行全局聚集张量操作
        def func(inp):
            ar = _functional_collectives.all_gather_tensor(inp, 0, "0")
            return ar
        
        # 创建大小为 4x4 的全一张量，放置在 CUDA 设备上作为输入
        inputs = torch.ones(4, 4, device="cuda")
        # 创建编译计数器对象
        counter = CompileCounter()
        # 使用编译器对 func 进行编译，同时使用 counter 作为后端
        compiled = torch.compile(func, backend=counter)
        # 执行编译后的函数并记录输出
        out = compiled(inputs)
        # 调用未编译的 func 函数计算正确的输出
        correct = func(inputs)
        # 断言帧计数为 1
        self.assertEqual(counter.frame_count, 1)

        # 断言操作计数为 2，代表了全局聚集张量操作和等待操作
        self.assertEqual(counter.op_count, 2)
        # 检查输出是否与正确输出相同
        self.assertTrue(same(out, correct))

    # 定义测试方法：测试动态追踪带分组的全局聚集张量操作
    def test_dynamo_trace_all_gather_tensor_pg(self):
        # 定义内部函数 func，接收输入 inp 和分组参数 pg，并对其执行全局聚集张量操作
        def func(inp, *, pg):
            ar = _functional_collectives.all_gather_tensor(inp, 0, pg)
            return ar
        
        # 创建大小为 4x4 的全一张量，放置在指定设备上作为输入
        inputs = torch.ones(4, 4, device=self.device)
        # 创建编译计数器对象
        counter = CompileCounter()
        # 使用编译器对 func 进行编译，同时使用 counter 作为后端，并设置 fullgraph 为 True
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        # 执行编译后的函数并记录输出，使用指定的分组成员 WORLD
        out = compiled(inputs, pg=GroupMember.WORLD)
        # 调用未编译的 func 函数计算正确的输出，使用指定的分组成员 WORLD
        correct = func(inputs, pg=GroupMember.WORLD)
        # 断言帧计数为 1
        self.assertEqual(counter.frame_count, 1)

        # 断言操作计数为 2，代表了全局聚集张量操作和等待操作
        self.assertEqual(counter.op_count, 2)
        # 检查输出是否与正确输出相同
        self.assertTrue(same(out, correct))

    # 定义测试方法：测试动态重写分布式全局聚集张量操作
    def test_dynamo_rewrite_dist_all_gather(self):
        # 定义函数 func，接收输入 inp、输出 out 和分组参数 pg，并执行分布式全局聚集张量操作
        def func(inp, out, *, pg):
            torch.distributed.all_gather_into_tensor(
                out,
                inp,
                pg,
            )

        # 定义本地大小为 [4, 4] 的列表作为输入大小
        local_size = [4, 4]
        # 单处理器测试，全局大小与本地大小相同
        global_size = local_size

        # 创建大小为 local_size 的全一张量，放置在指定设备上作为输入
        inputs = torch.ones(local_size, device=self.device)
        # 创建大小为 global_size 的空张量，放置在指定设备上作为输出
        outputs = torch.empty(global_size, device=self.device)
        # 创建大小为 global_size 的空张量，作为正确输出
        correct_outputs = torch.empty(global_size, device=self.device)
        # 创建编译计数器对象
        counter = CompileCounter()
        # 使用编译器对 func 进行编译，同时使用 counter 作为后端，并设置 fullgraph 为 True
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        # 执行编译后的函数，记录输出，使用指定的分组成员 WORLD
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        # 调用未编译的 func 函数执行操作，计算正确的输出，使用指定的分组成员 WORLD
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        # 断言帧计数为 1
        assert counter.frame_count == 1

        # 断言操作计数为 3，代表了全局聚集张量操作、等待操作和拷贝操作
        assert counter.op_count == 3
        # 断言输出张量与正确输出张量相同
        assert same(outputs, correct_outputs)
    def test_dynamo_rewrite_dist_all_gather_list(self):
        # 定义测试函数，测试 torch.distributed.all_gather 的功能
        def func(inp, out, *, pg):
            # 调用 torch.distributed.all_gather 进行数据聚集操作
            torch.distributed.all_gather(
                out,
                inp,
                pg,
            )

        local_size = [4, 4]
        # 在单进程环境下进行测试，全局大小与本地大小相同
        global_size = local_size

        # 创建输入张量，所有元素初始化为1，使用设备 self.device
        inputs = torch.ones(local_size, device=self.device)
        # 创建输出列表，包含一个全局大小的空张量，使用设备 self.device
        outputs = [torch.empty(global_size, device=self.device)]
        # 创建正确输出列表，包含一个全局大小的空张量，使用设备 self.device
        correct_outputs = [torch.empty(global_size, device=self.device)]
        # 编译测试函数 func，记录编译过程中的操作
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        # 使用编译后的函数进行调用，传入输入、输出以及分组成员 WORLD
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        # 直接调用原始函数 func 进行比较
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        # 断言编译过程中帧数为1
        assert counter.frame_count == 1
        # 断言输出与正确输出相同
        assert same(outputs, correct_outputs)

    def test_dynamo_rewrite_dist_all_gather_args_match(self):
        # 复制 test_dynamo_rewrite_dist_all_gather 的大部分结构
        # 使用 kwargs 确保重写具有匹配的参数名称
        def func(inp, out, *, pg):
            # 调用 torch.distributed.all_gather_into_tensor 进行数据聚集操作
            torch.distributed.all_gather_into_tensor(
                output_tensor=out,
                input_tensor=inp,
                group=pg,
                async_op=False,
            )

        local_size = [4, 4]
        # 在单进程环境下进行测试，全局大小与本地大小相同
        global_size = local_size

        # 创建输入张量，所有元素初始化为1，使用设备 self.device
        inputs = torch.ones(local_size, device=self.device)
        # 创建输出张量，全局大小的空张量，使用设备 self.device
        outputs = torch.empty(global_size, device=self.device)
        # 创建正确输出张量，全局大小的空张量，使用设备 self.device
        correct_outputs = torch.empty(global_size, device=self.device)
        # 编译测试函数 func，记录编译过程中的操作
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        # 使用编译后的函数进行调用，传入输入、输出以及分组成员 WORLD
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        # 直接调用原始函数 func 进行比较
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        # 断言编译过程中帧数为1
        assert counter.frame_count == 1

        # 应当更精确地测试，但是数字 3 预期应为 (all_gather, wait, copy_)
        assert counter.op_count == 3
        # 断言输出与正确输出相同
        assert same(outputs, correct_outputs)

    def test_dynamo_rewrite_dist_reduce_scatter(self):
        # 定义测试函数，测试 torch.distributed.reduce_scatter_tensor 的功能
        def func(inp, out, *, pg):
            # 调用 torch.distributed.reduce_scatter_tensor 进行张量的分散-聚集操作
            torch.distributed.reduce_scatter_tensor(
                out,
                inp,
                group=pg,
            )

        local_size = [4, 4]
        # 在单进程环境下进行测试，全局大小与本地大小相同
        global_size = local_size

        # 创建输入张量，所有元素初始化为1，使用设备 self.device
        inputs = torch.ones(local_size, device=self.device)
        # 创建输出张量，全局大小的空张量，使用设备 self.device
        outputs = torch.empty(global_size, device=self.device)
        # 创建正确输出张量，全局大小的空张量，使用设备 self.device
        correct_outputs = torch.empty(global_size, device=self.device)
        # 编译测试函数 func，记录编译过程中的操作
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        # 使用编译后的函数进行调用，传入输入、输出以及分组成员 WORLD
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        # 直接调用原始函数 func 进行比较
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        # 断言编译过程中帧数为1
        assert counter.frame_count == 1

        # 应当更精确地测试，但是数字 3 预期应为 (reduce_scatter, wait, copy_)
        assert counter.op_count == 3
        # 断言输出与正确输出相同
        assert same(outputs, correct_outputs)
    @parametrize(
        "pg_mode",
        [
            "positional",
            "positional_none",
            "kwargs",
            "kwargs_none",
            "unspecified",
        ],
    )
    # 使用 parametrize 装饰器为 test_dynamo_rewrite_dist_allreduce 函数提供多组参数化测试
    def test_dynamo_rewrite_dist_allreduce(self, pg_mode):
        # 定义内部函数 func，用于调用 torch.distributed.all_reduce
        def func(tensor, *args, **kwargs):
            torch.distributed.all_reduce(
                tensor,
                *args,
                **kwargs,
            )

        # 创建编译计数器对象
        counter = CompileCounter()
        # 编译 func 函数，并使用编译后的版本
        compiled = torch.compile(func, backend=counter, fullgraph=True)

        # 初始化空的 args 和 kwargs 列表和字典
        args = []
        kwargs = {}

        # 根据 pg_mode 参数的不同值，设置 args 和 kwargs 的不同组合
        if pg_mode == "positional":
            args.append(torch.distributed.ReduceOp.MAX)
            args.append(GroupMember.WORLD)
        elif pg_mode == "positional_none":
            args.append(torch.distributed.ReduceOp.MAX)
            args.append(None)
        elif pg_mode == "kwargs":
            kwargs["group"] = GroupMember.WORLD
        elif pg_mode == "kwargs_none":
            kwargs["group"] = None
        else:
            assert pg_mode == "unspecified"

        # 创建输入张量，用于编译版本和即时版本的函数调用
        inputs_compiled = torch.ones(2, device=self.device)
        inputs_eager = torch.ones(2, device=self.device)

        # 分别调用编译版本和即时版本的 func 函数
        compiled(inputs_compiled, *args, **kwargs)
        func(inputs_eager, *args, **kwargs)

        # 断言编译计数器的帧数为1
        assert counter.frame_count == 1
        # 断言操作数为3（预期为 all_reduce, wait, copy_）
        assert counter.op_count == 3
        # 断言编译版本和即时版本的输入张量相同
        assert same(inputs_compiled, inputs_eager)

    # 定义测试函数 test_dynamo_rewrite_dist_all_to_all_single
    def test_dynamo_rewrite_dist_all_to_all_single(self):
        # 定义内部函数 func，用于调用 torch.distributed.all_to_all_single
        def func(output, input, pg):
            torch.distributed.all_to_all_single(output, input, group=pg)

        # 创建编译计数器对象
        counter = CompileCounter()
        # 编译 func 函数，并使用编译后的版本
        compiled = torch.compile(func, backend=counter, fullgraph=True)

        # 初始化输入和输出张量
        input_compiled = torch.ones(2, device=self.device)
        input_eager = torch.ones(2, device=self.device)
        output_compiled = torch.empty(2, device=self.device)
        output_eager = torch.empty(2, device=self.device)

        # 分别调用编译版本和即时版本的 func 函数
        compiled(output_compiled, input_compiled, GroupMember.WORLD)
        func(output_eager, input_eager, GroupMember.WORLD)

        # 断言编译计数器的帧数为1
        assert counter.frame_count == 1
        # 断言编译版本和即时版本的输出张量相同
        assert same(output_compiled, output_eager)

    @parametrize(
        "reduce_op",
        [
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.AVG,
            torch.distributed.ReduceOp.PRODUCT,
            torch.distributed.ReduceOp.MIN,
            torch.distributed.ReduceOp.MAX,
        ],
    )
    # 定义一个测试方法，用于测试分布式环境下的 reduce 操作重写
    def test_dynamo_rewrite_dist_allreduce_reduce_op(self, reduce_op):
        # 导入 reduce 操作类型到字符串的映射
        from torch.distributed._functional_collectives import REDUCE_OP_TO_STR
        
        # 定义验证重写函数，接受图模型 gm 和参数 _
        def verify_rewrite(gm, _):
            # 初始化空列表，用于存储所有的 all_reduce 节点
            ar_nodes = []
            # 遍历图中的所有节点
            for node in gm.graph.nodes:
                # 如果节点的目标是指定的 all_reduce 函数
                if node.target in [
                    torch.ops.c10d_functional.all_reduce,
                    torch.ops._c10d_functional.all_reduce,
                ]:
                    # 将该节点添加到 ar_nodes 列表中
                    ar_nodes.append(node)
            # 断言 ar_nodes 列表长度为 1
            self.assertEqual(len(ar_nodes), 1)
            # 获取 all_reduce 节点的第二个参数（reduce 操作类型）的字符串表示
            reduce_op_str = ar_nodes[0].args[1]
            # 断言 REDUCE_OP_TO_STR 中的 reduce_op 和 reduce_op_str 相等
            self.assertEqual(REDUCE_OP_TO_STR[reduce_op], reduce_op_str)
            # 返回图模型 gm
            return gm

        # 使用 torch.compile 编译分布式的 all_reduce 函数，指定 backend 为 verify_rewrite，启用 fullgraph
        compiled = torch.compile(
            torch.distributed.all_reduce,
            backend=verify_rewrite,
            fullgraph=True,
        )
        # 定义输入参数元组
        inputs = (
            torch.ones(2, device=self.device),
            reduce_op,
            GroupMember.WORLD,
        )
        # 执行编译后的 all_reduce 函数
        compiled(*inputs)

    # 使用 @parametrize 装饰器定义多个测试用例
    @parametrize(
        "source",
        [
            "GroupMember.WORLD",
            "group.WORLD",
            "_get_default_group",
        ],
    )
    # 定义测试获取世界组的方法
    def test_dynamo_get_world_group(self, source):
        # 定义内部函数 func，接受一个 tensor 参数
        def func(tensor):
            # 根据 source 的不同取值确定 group 变量
            if source == "GroupMember.WORLD":
                group = torch.distributed.GroupMember.WORLD
            elif source == "group.WORLD":
                group = torch.distributed.group.WORLD
            else:
                assert source == "_get_default_group"
                group = torch.distributed.distributed_c10d._get_default_group()

            # 调用 torch.distributed.all_reduce 函数，传入 tensor 和确定的 group
            torch.distributed.all_reduce(
                tensor,
                group=group,
            )

        # 定义验证函数 verify，接受图模型 gm 和参数 _
        def verify(gm, _):
            # 初始化空列表，用于存储所有的 all_reduce 节点
            ar_nodes = []
            # 遍历图中的所有节点
            for node in gm.graph.nodes:
                # 如果节点的目标是指定的 all_reduce 函数
                if node.target in [
                    torch.ops.c10d_functional.all_reduce,
                    torch.ops._c10d_functional.all_reduce,
                ]:
                    # 将该节点添加到 ar_nodes 列表中
                    ar_nodes.append(node)
            # 断言 ar_nodes 列表长度为 1
            self.assertEqual(len(ar_nodes), 1)
            # 返回图模型 gm
            return gm

        # 使用 torch.compile 编译 func 函数，指定 backend 为 verify，启用 fullgraph
        compiled = torch.compile(func, backend=verify, fullgraph=True)
        # 定义输入参数
        input = torch.ones(2, device=self.device)
        # 执行编译后的 func 函数
        compiled(input)
    # 测试在异步操作为 False 的情况下 Dynamo 支持集合操作
    def test_dynamo_support_collective_op_with_async_op_False(self):
        # 定义一个函数，接收输入和输出张量，以及参数组 pg
        def func(inp, out, *, pg):
            # 用户显式将属性 `async_op` 设置为 False，不应该导致图断裂
            torch.distributed.reduce_scatter_tensor(out, inp, group=pg, async_op=False)

        local_size = [4, 4]
        # 单进程测试
        global_size = local_size

        # 创建输入张量，填充为全 1，使用指定设备
        inputs = torch.ones(local_size, device=self.device)
        # 创建空的输出张量，使用指定设备
        outputs = torch.empty(global_size, device=self.device)
        # 创建正确的输出张量，使用指定设备
        correct_outputs = torch.empty(global_size, device=self.device)
        # 创建计数器对象
        counter = CompileCounter()
        # 编译函数，使用计数器作为后端
        compiled = torch.compile(func, backend=counter)
        # 调用编译后的函数
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        # 调用原始函数
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        # 断言帧数为 1
        assert counter.frame_count == 1
        # 断言操作数为 3
        assert counter.op_count == 3
        # 断言输出与正确输出相同
        assert same(outputs, correct_outputs)

    # 测试 Dynamo 不支持异步操作时的图断裂
    def test_dynamo_graphbreaks_unsupported_async_op(self):
        # 定义一个函数，接收输入和输出张量，以及参数组 pg
        def func(inp, out, *, pg):
            # 执行异步操作，等待操作完成
            work = torch.distributed.reduce_scatter_tensor(
                out, inp, group=pg, async_op=True
            )
            work.wait()

        local_size = [4, 4]
        # 单进程测试
        global_size = local_size

        # 创建输入张量，填充为全 1，使用指定设备
        inputs = torch.ones(local_size, device=self.device)
        # 创建空的输出张量，使用指定设备
        outputs = torch.empty(global_size, device=self.device)
        # 创建正确的输出张量，使用指定设备
        correct_outputs = torch.empty(global_size, device=self.device)
        # 创建计数器对象
        counter = CompileCounter()
        # 编译函数，使用计数器作为后端
        compiled = torch.compile(func, backend=counter)
        # 调用编译后的函数
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        # 调用原始函数
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        # 断言帧数为 0
        assert counter.frame_count == 0
        # 断言操作数为 0
        assert counter.op_count == 0
        # 断言输出与正确输出相同
        assert same(outputs, correct_outputs)

    # 测试 Dynamo 参数组变量
    def test_dynamo_pg_var(self):
        # 定义一个函数，接收输入张量和参数组 pg
        def func(inp, *, pg):
            # 计算 x，为参数组中的排名加 1 取模排名总数
            x = pg.rank() + 1 % pg.size()
            return inp + x

        local_size = [4, 4]
        # 创建输入张量，填充为全 1，使用指定设备
        inputs = torch.ones(local_size, device=self.device)
        # 创建正确的输出张量，使用指定设备
        correct_outputs = torch.empty(local_size, device=self.device)
        # 创建计数器对象
        counter = CompileCounter()
        # 编译函数，使用计数器作为后端，完整图
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        # 调用编译后的函数
        outputs = compiled(inputs, pg=GroupMember.WORLD)
        # 调用原始函数
        correct_outputs = func(inputs, pg=GroupMember.WORLD)
        # 断言帧数为 1
        assert counter.frame_count == 1
        # 断言操作数为 1
        assert counter.op_count == 1
        # 断言输出与正确输出相同
        assert same(outputs, correct_outputs)
    def test_dynamo_trace_reduce_scatter_tensor(self):
        # 定义测试函数，用于测试 reduce_scatter_tensor 功能
        def func(inp):
            # 调用 reduce_scatter_tensor 函数进行张量的 reduce_scatter 操作
            ar = _functional_collectives.reduce_scatter_tensor(inp, "sum", 0, "0")
            return ar

        # 创建输入张量，全为1，在CUDA设备上
        inputs = torch.ones(4, 4, device="cuda")
        # 创建编译计数器
        counter = CompileCounter()
        # 使用计数器作为后端编译函数 func
        compiled = torch.compile(func, backend=counter)
        # 对输入进行编译后执行
        out = compiled(inputs)
        # 直接调用 func 获取正确的输出
        correct = func(inputs)
        # 断言编译的帧数为1
        self.assertEqual(counter.frame_count, 1)

        # 断言操作计数为2，应该包括 reduce_scatter 和等待操作
        self.assertEqual(counter.op_count, 2)
        # 断言编译输出与直接调用 func 的输出相同
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allgather_coalesced(self):
        # 定义测试函数，用于测试 allgather_coalesced 功能
        def func(inp, *, tag, ranks, group_size):
            # 调用 all_gather_into_tensor_coalesced 函数进行张量的 allgather 操作
            ar = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(
                inp, tag, ranks, group_size
            )
            return ar

        # 创建输入张量列表，在CUDA设备上
        inputs = [torch.ones(4, 4, device="cuda"), torch.ones(6, 6, device="cuda")]
        # 创建编译计数器
        counter = CompileCounter()
        # 使用计数器作为后端编译函数 func
        compiled = torch.compile(func, backend=counter)
        # 对输入进行编译后执行，传入世界传输参数
        out = compiled(inputs, **self.get_world_trs())
        # 直接调用 func 获取正确的输出，传入世界传输参数
        correct = func(inputs, **self.get_world_trs())
        # 断言编译的帧数为1
        assert counter.frame_count == 1
        # 断言操作计数为3，应该包括 all_gather 和 unpack 数组的操作
        assert counter.op_count == 3
        # 断言编译输出与直接调用 func 的输出相同
        assert same(out, correct)

    def test_backwards(self):
        """
        It's probably not that common to need backwards support for collectives.

        However, I wanted to at least see if it was possible to support it as a design goal.
        """

        # 定义测试函数，用于测试 all_reduce 的反向传播支持
        def func(inp):
            # 调用 all_reduce 函数进行张量的 all_reduce 操作
            ar = _functional_collectives.all_reduce(inp, "sum", "0")
            return ar

        # 创建输入张量，全为1，在CUDA设备上，并且需要梯度计算
        input = torch.ones(4, 4, device="cuda", requires_grad=True)
        # 抛出异常，验证是否能正确处理无梯度的张量情况
        with self.assertRaisesRegex(
            RuntimeError,
            "element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            # 使用 aot_eager 后端进行函数编译
            compiled = torch.compile(
                func, backend="aot_eager"
            )  # inductor bug with single-op allreduce graph
            # 执行编译后的函数，传入输入张量
            out = compiled(input)
            # 对输出结果求和，并进行反向传播
            out.sum().backward()

            # 创建需要梯度计算的正确输入副本
            correct_input = input.clone().detach().requires_grad_()
            # 直接调用 func 获取正确的输出
            correct = func(correct_input)
            # 对正确输出结果进行反向传播
            correct.sum().backward()
            # 断言编译输出与直接调用 func 的输出相同
            self.assertTrue(same(out, correct))
            # 断言输入张量的梯度与正确输入张量的梯度相同
            self.assertTrue(same(input.grad, correct_input.grad))

    def test_meta(self):
        # 创建元设备上的随机张量 x
        x = torch.rand((2, 3, 4), device="meta")
        # 使用 all_reduce 函数在世界传输参数下进行张量的全局归约
        out = torch.ops.c10d_functional.all_reduce(x, "sum", **self.get_world_trs())
        # 断言输入张量与输出张量的形状相同
        self.assertEqual(x.size(), out.size())

    # 如果没有 Triton 并且 GPU 架构不是最新的，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # 设置 Torch Inductor 的配置，包括调试模式和 Triton 描述性名称选项
    @torch._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_all_gather_coalesced(self):
        """
        确保中间值在重复使用时不会被意外修改
        """

        # 定义内部函数 func，接受输入 inp 和其他参数，返回多个计算结果
        def func(inp, *, tag, ranks, group_size):
            # 将 inp 加 1，保存到 x 中
            x = inp + 1
            # 调用 Torch 自定义操作 all_gather_into_tensor_coalesced，将 x 和 inp 作为输入
            tensor_list = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(
                [x, inp], tag, ranks, group_size
            )
            # 将 x 再加 2，保存到 y 中
            y = x + 2
            # 等待第一个张量的操作完成，保存结果到 ar0
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            # 等待第二个张量的操作完成，保存结果到 ar1
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            # 确保 other 不会错误地引用 ar 的缓冲区
            other = torch.ones_like(inp) + 22
            # 返回结果 ar0, y, other, ar1
            return ar0, y, other, ar1

        # 创建一个尺寸为 (4, 4) 的张量 inputs，使用 CUDA 设备
        inputs = torch.ones(4, 4, device="cuda")

        # 编译 func 函数以便在 Triton 中运行
        compiled = torch.compile(func)
        # 运行编译后的代码并获取 Triton 代码，使用指定的输入和环境参数
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        
        # 使用 FileCheck 检查 Triton 代码，确保在返回图中正确处理 wait_tensors 的输出
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf6 = empty_strided")
            .check(".run(arg0_1, buf0, buf6, 16")
            .check(
                "buf1 = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default([buf0, arg0_1]"
            )
            .check("buf2 = buf1[0]")
            .check("buf3 = buf1[1]")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("buf7 = buf0; del buf0  # reuse")
            .check(".run(buf7, 16")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("return (buf2, buf6, buf7, buf3")
            .run(code)
        )

        # 使用编译后的 func 函数执行输入数据，获取输出
        out = compiled(inputs, **self.get_world_trs())
        # 调用 func 函数获取正确的预期输出
        correct = func(inputs, **self.get_world_trs())
        # 断言输出与正确结果相同，否则输出错误信息
        assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_reduce_scatter_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            # 将输入张量 inp 的每个元素加 1，并赋值给 x
            x = inp + 1
            # 调用 C10D 库中的 reduce_scatter_tensor_coalesced 函数进行张量减少分散操作
            tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced(
                [x, inp], "sum", tag, ranks, group_size
            )
            # 将 x 再加 2，并赋值给 y
            y = x + 2
            # 等待第一个张量的操作完成，将结果赋值给 ar0
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            # 等待第二个张量的操作完成，将结果赋值给 ar1
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            # 确保 other 张量不会错误地别名化 ar 的缓冲区
            # 创建一个与 inp 张量形状相同且每个元素为 1 的张量，加 22 后赋值给 other
            other = torch.ones_like(inp) + 22
            # 返回 ar0, y, other, ar1 四个张量
            return ar0, y, other, ar1

        # 在 CUDA 设备上创建一个 4x4 全 1 的张量，作为输入
        inputs = torch.ones(4, 4, device="cuda")

        # 编译 func 函数
        compiled = torch.compile(func)
        # 运行编译后的函数，获取其 Triton 代码
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        
        # 使用 FileCheck 检查 Triton 代码的输出
        (
            FileCheck()
            .check("buf0 = empty_strided")  # 检查 buf0 是否为空
            .check("buf6 = empty_strided")  # 检查 buf6 是否为空
            .check(".run(arg0_1, buf0, buf6, 16")  # 检查是否以 arg0_1 为参数运行，并输出到 buf0 和 buf6
            .check(
                "buf1 = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default([buf0, arg0_1]"  # 检查是否正确调用 reduce_scatter_tensor_coalesced
            )
            .check("buf2 = buf1[0]")  # 检查是否正确获取 buf1 的第一个元素并赋给 buf2
            .check("buf3 = buf1[1]")  # 检查是否正确获取 buf1 的第二个元素并赋给 buf3
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")  # 检查是否正确等待 buf2 的操作完成
            .check("buf7 = buf0; del buf0  # reuse")  # 检查是否正确复用 buf0 并删除 buf0
            .check(".run(buf7, 16")  # 检查是否以 buf7 为参数再次运行，并输出到 16
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")  # 检查是否正确等待 buf3 的操作完成
            .check("return (buf2, buf6, buf7, buf3")  # 检查返回的四个张量是否正确
            .run(code)  # 运行 Triton 代码
        )

        # 使用编译后的函数计算输出
        out = compiled(inputs, **self.get_world_trs())
        # 计算正确的输出结果
        correct = func(inputs, **self.get_world_trs())
        # 断言输出结果与正确结果相同
        assert same(out, correct), f"{out} va {correct}"
# 如果当前脚本作为主程序执行（而不是被导入作为模块），则执行以下代码块
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests
    
    # 调用导入的 run_tests 函数，用于执行测试用例
    run_tests()
```