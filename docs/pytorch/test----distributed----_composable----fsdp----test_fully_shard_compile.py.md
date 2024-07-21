# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_compile.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和库
import contextlib  # 上下文管理工具库
import copy  # 复制操作相关库
import unittest  # 单元测试框架

import torch  # PyTorch 主库
import torch._dynamo.testing  # PyTorch 内部测试相关
import torch.distributed._composable.fsdp._fsdp_param  # PyTorch 分布式训练相关库
from torch import nn  # PyTorch 神经网络模块
from torch._dynamo import compiled_autograd  # PyTorch 动态编译自动求导模块

from torch.distributed._composable.fsdp import fully_shard  # 导入完全分片相关函数
from torch.distributed._composable.fsdp._fsdp_common import TrainingState  # 训练状态相关定义
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup  # FSDP 参数组定义
from torch.distributed._tensor import init_device_mesh  # 初始化设备网格
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 根据 GPU 数量跳过测试
from torch.testing._internal.common_fsdp import FSDPTest, MLP  # FSDP 测试相关和 MLP 模型
from torch.testing._internal.common_utils import run_tests, skipIfRocm  # 运行测试和根据 ROCm 跳过测试
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 分布式张量测试相关
    ModelArgs,
    Transformer,
)
from torch.utils._triton import has_triton  # 检查是否安装了 Triton

# 定义测试类 TestFullyShardCompileCompute，继承自 FSDPTest 类
class TestFullyShardCompileCompute(FSDPTest):
    # 装饰器，如果没有 Triton 或 GPU 架构太旧则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量少于 2，则跳过测试
    def test_disable_compiling_hooks(self):
        # 运行子测试，传入参数字典和测试函数
        self.run_subtests(
            {
                "skip_fsdp_hooks": [False, True],  # 参数字典，用于测试
            },
            self._test_disable_compiling_hooks,  # 测试函数名称
        )

    # 测试函数 _test_disable_compiling_hooks
    def _test_disable_compiling_hooks(
        self,
        skip_fsdp_hooks: bool,  # 是否跳过 FSDP 钩子的布尔值参数
    ):
        torch._dynamo.reset()  # 重置 PyTorch 动态配置
        trace_rules_check_count = 0  # 跟踪规则检查计数器初始化
        HOOKS_FILE_NAME = "torch/distributed/_composable/fsdp/_fsdp_state.py"  # 钩子文件名
        HOOK_WRAPPER_NAME = "fsdp_hook_wrapper"  # 钩子包装器名称

        # 定义补丁函数，用于替换 trace_rules.check 函数
        def patched_trace_rules_check(*args, **kwargs):
            nonlocal trace_rules_check_count  # 使用非本地变量 trace_rules_check_count
            f_code = args[0]  # 获取函数对象的代码对象
            if (
                hasattr(f_code, "co_filename")  # 如果代码对象具有文件名属性
                and f_code.co_filename.endswith(HOOKS_FILE_NAME)  # 并且文件名以指定字符串结尾
                and f_code.co_name != HOOK_WRAPPER_NAME  # 并且函数名不是钩子包装器名称
            ):
                trace_rules_check_count += 1  # 计数器加一
            return orig_trace_rules_check(*args, **kwargs)  # 调用原始的 trace_rules.check 函数

        original_skip_fsdp_hooks = torch._dynamo.config.skip_fsdp_hooks  # 原始跳过 FSDP 钩子配置
        orig_trace_rules_check = torch._dynamo.trace_rules.check  # 原始 trace_rules.check 函数
        torch.distributed.barrier()  # 分布式同步点
        torch._dynamo.config.skip_fsdp_hooks = skip_fsdp_hooks  # 设置跳过 FSDP 钩子配置
        torch._dynamo.trace_rules.check = patched_trace_rules_check  # 设置补丁后的 trace_rules.check 函数
        model = MLP(4)  # 创建一个具有 4 层的 MLP 模型
        fully_shard(model)  # 对模型进行完全分片
        model.compile()  # 编译模型
        model(torch.randn((4, 4), device="cuda"))  # 在 CUDA 设备上运行模型
        torch.distributed.barrier()  # 分布式同步点
        torch._dynamo.config.skip_fsdp_hooks = original_skip_fsdp_hooks  # 恢复原始的跳过 FSDP 钩子配置
        torch._dynamo.trace_rules.check = orig_trace_rules_check  # 恢复原始的 trace_rules.check 函数
        if skip_fsdp_hooks:
            self.assertEqual(trace_rules_check_count, 0)  # 如果跳过 FSDP 钩子，则检查计数器应为 0
        else:
            self.assertTrue(trace_rules_check_count > 0)  # 否则，检查计数器应大于 0

# 定义测试类 TestFullyShardCompile，继承自 FSDPTest 类
class TestFullyShardCompile(FSDPTest):
    # 属性方法，返回 CUDA 设备数和 2 的较小值
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())
    def test_dynamo_trace_use_training_state(self):
        torch._dynamo.reset()
        # 重置 Torch Dynamo 状态，准备进行测试

        # 构造一个虚拟的 FSDPParamGroup 对象，用于测试 `use_training_state` 上下文管理器
        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            torch.nn.Linear(1, 1),  # module: nn.Module,
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            None,  # device: torch.device,
            None,  # mp_policy: MixedPrecisionPolicy,
            None,  # offload_policy: OffloadPolicy,
        )

        def f(x):
            # 设置 param_group 的 `_training_state` 属性为 IDLE
            param_group._training_state = TrainingState.IDLE
            # 使用 `use_training_state` 上下文管理器将 `_training_state` 设置为 FORWARD
            with param_group.use_training_state(TrainingState.FORWARD):
                # 检查 `_training_state` 是否为 FORWARD
                if param_group._training_state == TrainingState.FORWARD:
                    return x + 1  # 如果为 FORWARD，返回 x + 1
                else:
                    return x  # 否则返回 x

        inp = torch.zeros(1)
        # 断言 param_group 的 `_training_state` 属性为 IDLE
        self.assertEqual(param_group._training_state, TrainingState.IDLE)

        eager_out = f(inp)
        # 断言 param_group 的 `_training_state` 属性为 IDLE
        self.assertEqual(param_group._training_state, TrainingState.IDLE)
        # 断言 eager_out 的值为 inp + 1
        self.assertEqual(eager_out, inp + 1)

        # 创建一个 CompileCounterWithBackend 对象
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 使用 Torch 编译函数 `torch.compile` 编译函数 f，指定 backend 为 cnt，并使用完整图形 (fullgraph=True)
        compiled_out = torch.compile(f, backend=cnt, fullgraph=True)(inp)
        # 断言 param_group 的 `_training_state` 属性为 IDLE
        self.assertEqual(param_group._training_state, TrainingState.IDLE)
        # 断言 eager_out 的值与 compiled_out 的值相等
        self.assertEqual(eager_out, compiled_out)
        # 断言 cnt 的 frame_count 为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 cnt 的 op_count 为 1
        self.assertEqual(cnt.op_count, 1)
        # 断言 cnt 的 graphs 长度为 1
        self.assertEqual(len(cnt.graphs), 1)

    def test_trace_fsdp_set_(self):
        # 定义自定义 Torch 操作 `add_one_out`，标记其对 `out` 参数进行修改
        @torch.library.custom_op("mylib::add_one_out", mutates_args={"out"})
        def add_one_out(x: torch.Tensor, out: torch.Tensor) -> None:
            torch.add(x, 1, out=out)

        def f(x):
            # 创建一个长度为 2 的全零 Tensor 对象 buf
            buf = torch.zeros(2)
            # 将 buf 视图转换为一维数组 buf_view
            buf_view = buf.view(-1)
            # 调用自定义 Torch 操作 `mylib::add_one_out`，将 x 加 1，结果存入 buf_view
            torch.ops.mylib.add_one_out(x, out=buf_view)
            # 再次将 buf 视图转换为一维数组 buf_view2
            buf_view2 = buf.view(-1)
            # 使用 Torch 操作 `fsdp.set_` 将 buf_view2 的值设置回 x
            torch.ops.fsdp.set_(x, buf_view2)

        ref_x = torch.zeros(2)
        x = copy.deepcopy(ref_x)
        f(ref_x)
        # 使用 Torch 编译函数 `torch.compile` 编译函数 f，指定 backend 为 "aot_eager"
        torch.compile(f, backend="aot_eager")(x)
        # 断言 x 的值与 ref_x 的值相等
        self.assertEqual(x, ref_x)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @torch._functorch.config.patch(recompute_views=True)
    @torch._functorch.config.patch(cse=False)
    def _test_traceable_fsdp(
        self, model_init_fn, input_creation_fn, backend, fullgraph
    ):
        # 定义一个内部函数，用于返回编译后自动微分后端的编译器函数
        def compiler_fn(compiled_autograd_backend):
            # 定义内部函数 _fn，用于编译计算图
            # fullgraph=True 是因为 Traceable FSDP2 尚不支持编译自动微分后端的图中断
            # （主要困难来自于当后向计算图中有图中断时 queue_callback 的工作不良）。
            return torch.compile(
                gm, backend=compiled_autograd_backend, fullgraph=True
            )

            return _fn

        # 定义一个运行多次迭代的函数，用于模型训练
        def run_iters(model, optim, n_iter=10, compiled_autograd_backend=None):
            # 设定随机种子
            torch.manual_seed(42)
            losses = []
            for i in range(n_iter):
                # 创建输入
                inp = input_creation_fn()
                if compiled_autograd_backend is not None:
                    # 启用编译自动微分上下文
                    maybe_compiled_autograd_ctx = compiled_autograd.enable(
                        compiler_fn(compiled_autograd_backend)
                    )
                else:
                    # 否则使用空上下文
                    maybe_compiled_autograd_ctx = contextlib.nullcontext()
                with maybe_compiled_autograd_ctx:
                    # 模型前向传播
                    out = model(inp)
                    # 计算损失
                    loss = out.sum()
                    losses.append(loss.item())
                    # 反向传播
                    loss.backward()
                # 优化器执行优化步骤
                optim.step()
                # 清空梯度
                optim.zero_grad(set_to_none=True)
            return losses

        # 定义测试编译后模型性能的函数
        def test_compiled():
            model, optim = model_init_fn()
            # FSDP2 使用第一次运行进行延迟初始化，因此运行一次以使用急切模式进行初始化
            run_iters(model, optim, n_iter=1)

            # 编译模型
            model_compiled = torch.compile(model, backend=backend, fullgraph=True)
            # 运行编译后模型的迭代训练
            res = run_iters(model_compiled, optim, compiled_autograd_backend=backend)
            return res

        # 定义测试急切模式下模型性能的函数
        def test_eager():
            model, optim = model_init_fn()
            # FSDP2 使用第一次运行进行延迟初始化，因此运行一次以使用急切模式进行初始化
            run_iters(model, optim, n_iter=1)

            # 运行急切模式下模型的迭代训练
            res = run_iters(model, optim)
            return res

        # 执行编译模式和急切模式下的测试，并对比损失值
        losses_compiled = test_compiled()
        losses_eager = test_eager()
        for loss_compiled, loss_eager in zip(losses_compiled, losses_eager):
            # 断言两种模式下的损失值是否在一定误差范围内接近
            self.assertTrue(
                torch.allclose(
                    torch.tensor(loss_compiled),
                    torch.tensor(loss_eager),
                    rtol=1e-5,
                    atol=1e-8,
                ),
                f"{loss_compiled} vs {loss_eager}",
            )
    def _create_simple_mlp_factory_fns(self):
        # 定义 MLP 隐藏层的维度
        hidden_dim = 16

        # 定义模型初始化函数
        def model_init_fn():
            # 设置随机种子为 self.rank
            torch.manual_seed(self.rank)
            # FSDP 配置为空字典
            fsdp_config = {}
            # 创建包含多个线性层和激活函数的序列模型
            model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, device="cuda"),  # 第一线性层
                nn.ReLU(),  # 第一个ReLU激活函数
                nn.Linear(hidden_dim, hidden_dim, device="cuda"),  # 第二线性层
                nn.ReLU(),  # 第二个ReLU激活函数
                nn.Linear(hidden_dim, hidden_dim, device="cuda"),  # 第三线性层
            )
            # 对模型进行全分片，每个层后执行前后的重分片
            fully_shard(model, reshard_after_forward=True, **fsdp_config)
            # 使用 SGD 优化器，学习率为 1e-4
            optim = torch.optim.SGD(model.parameters(), lr=1e-4)
            return model, optim  # 返回模型和优化器实例

        # 定义输入数据创建函数
        def input_creation_fn():
            # 设置随机种子为 self.rank
            torch.manual_seed(self.rank)
            # 生成一个随机正态分布的输入张量，形状为 (2, hidden_dim)，在 CUDA 设备上，不需要梯度
            inp = torch.randn((2, hidden_dim), device="cuda", requires_grad=False)
            return inp  # 返回输入张量

        return model_init_fn, input_creation_fn  # 返回模型初始化函数和输入创建函数的元组

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_simple_mlp_fullgraph_backend_aot_eager(self):
        # 调用 _test_traceable_fsdp 方法，传入简单 MLP 的模型初始化和输入创建函数，"aot_eager" 模式，启用全图模式
        self._test_traceable_fsdp(
            *self._create_simple_mlp_factory_fns(), "aot_eager", fullgraph=True
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_simple_mlp_fullgraph_backend_aot_eager_decomp_partition(self):
        # 调用 _test_traceable_fsdp 方法，传入简单 MLP 的模型初始化和输入创建函数，"aot_eager_decomp_partition" 模式，启用全图模式
        self._test_traceable_fsdp(
            *self._create_simple_mlp_factory_fns(),
            "aot_eager_decomp_partition",
            fullgraph=True,
        )

    @skipIfRocm
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_simple_mlp_fullgraph_backend_inductor(self):
        # 调用 _test_traceable_fsdp 方法，传入简单 MLP 的模型初始化和输入创建函数，"inductor" 模式，启用全图模式
        self._test_traceable_fsdp(
            *self._create_simple_mlp_factory_fns(), "inductor", fullgraph=True
        )

    def _create_transformer_factory_fns(self):
        # 定义 Transformer 模型的序列长度和词汇表大小
        seq_len = 16
        vocab_size = 8

        # 定义模型初始化函数
        def model_init_fn():
            # 设置随机种子为 self.rank
            torch.manual_seed(self.rank)
            # FSDP 配置为空字典
            fsdp_config = {}
            # 初始化设备网格为 CUDA，形状为 (self.world_size,)
            mesh = init_device_mesh("cuda", (self.world_size,))
            # 使用给定的词汇表大小创建模型参数
            model_args = ModelArgs(vocab_size=vocab_size)
            # 创建 Transformer 模型实例
            model = Transformer(model_args)
            # 对每个层进行全分片，每个模块后执行前后的重分片
            for layer_id, mod in enumerate(model.layers):
                fully_shard(mod, mesh=mesh, reshard_after_forward=True, **fsdp_config)
            model = fully_shard(
                model, mesh=mesh, reshard_after_forward=True, **fsdp_config
            )
            # 使用 SGD 优化器，学习率为 1e-4
            optim = torch.optim.SGD(model.parameters(), lr=1e-4)
            return model, optim  # 返回模型和优化器实例

        # 定义输入数据创建函数
        def input_creation_fn():
            # 设置随机种子为 self.rank
            torch.manual_seed(self.rank)
            # 生成一个随机整数张量，值在 [0, vocab_size) 范围内，形状为 (2, seq_len)，在 CUDA 设备上，不需要梯度
            inp = torch.randint(
                0, vocab_size, (2, seq_len), device="cuda", requires_grad=False
            )
            return inp  # 返回输入张量

        return model_init_fn, input_creation_fn  # 返回模型初始化函数和输入创建函数的元组

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_transformer_fullgraph_backend_aot_eager(self):
        # 调用 _test_traceable_fsdp 方法，传入 Transformer 模型的模型初始化和输入创建函数，"aot_eager" 模式，启用全图模式
        self._test_traceable_fsdp(
            *self._create_transformer_factory_fns(), "aot_eager", fullgraph=True
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    # TODO: native_dropout has worse accuracy after decomp, need to figure out why
    @torch._inductor.config.patch(fallback_random=True)
    def test_transformer_fullgraph_backend_aot_eager_decomp_partition(self):
        # 调用 _test_traceable_fsdp 方法，测试 Transformer 使用 fullgraph 和 aot_eager_decomp_partition 后的行为
        self._test_traceable_fsdp(
            *self._create_transformer_factory_fns(),
            "aot_eager_decomp_partition",
            fullgraph=True,
        )
    
    @skipIfRocm
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: native_dropout causes CUDA IMA error, need to figure out why
    @torch._inductor.config.patch(fallback_random=True)
    def test_transformer_fullgraph_backend_inductor(self):
        # 调用 _test_traceable_fsdp 方法，测试 Transformer 使用 fullgraph 和 inductor 后的行为
        self._test_traceable_fsdp(
            *self._create_transformer_factory_fns(), "inductor", fullgraph=True
        )
# 如果当前脚本被直接执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```