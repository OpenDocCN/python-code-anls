# `.\pytorch\test\distributed\fsdp\test_checkpoint_wrapper.py`

```
# Owner(s): ["oncall: distributed"]

# 引入上下文管理、单元测试模块，以及深拷贝函数和偏函数
import contextlib
import unittest
from copy import deepcopy
from functools import partial

# 引入 PyTorch 核心模块
import torch
import torch.nn as nn

# 引入分布式训练相关的模块和类
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
    CheckpointWrapper,
    offload_wrapper,
    OffloadWrapper,
)
# 引入分布式自定义模块
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# 引入内部测试工具相关模块和类
from torch.testing._internal.common_utils import run_tests, TestCase

# 引入 PyTorch 的检查点相关模块
from torch.utils.checkpoint import checkpoint

# 全局变量：保存模型参数前缀
_SAVED_PREFIX = "_saved_"

# 全局常量：梯度函数的下一步骤函数名称
GRAD_FN_NEXT_FUNCTIONS = "next_functions"


class CheckpointWrapperTest(TestCase):
    def test_load_activation_checkpointed_module(self):
        # 创建一个线性模型
        lin = nn.Linear(10, 10, bias=False)
        
        # 对线性模型应用检查点包装
        lin = checkpoint_wrapper(
            lin,
            checkpoint_fn=checkpoint,
            # 检查点参数
            use_reentrant=True,
            preserve_rng_state=False,
        )
        
        # 深拷贝模型的状态字典
        state_dict = deepcopy(lin.state_dict())
        
        # 创建一个新的未检查点包装的线性模型
        lin_new = nn.Linear(10, 10, bias=False)
        
        # 加载状态字典到新模型中
        lin_new.load_state_dict(state_dict)
        
        # 检查两个模型的参数是否相等
        for p1, p2 in zip(lin.parameters(), lin_new.parameters()):
            self.assertEqual(p1, p2)
            self.assertTrue(torch.allclose(p1, p2))

        # 将未检查点包装的模型加载到检查点包装的模型中
        # 改变参数值以确保不同
        for p in lin_new.parameters():
            with torch.no_grad():
                p.add_(0.5)

        # 再次深拷贝新模型的状态字典
        state_dict = deepcopy(lin_new.state_dict())
        
        # 验证检查点包装的线性模型是否能够加载未包装的线性模型
        lin.load_state_dict(state_dict)
        
        # 检查两个模型的参数是否相等
        for p1, p2 in zip(lin.parameters(), lin_new.parameters()):
            self.assertEqual(p1, p2)
    def test_checkpoint_wrapper_kwarg_support(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(10, 10)

            def forward(self, a, b, c=None, d=None, **kwargs):
                # 定义模型的前向传播逻辑，接受输入参数 a, b, c, d 和任意额外的关键字参数
                return (self.lin(a), self.lin(b), self.lin(c), self.lin(d))

        for wrapper in [
            partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT),
            partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            offload_wrapper,
        ]:
            with self.subTest(wrapper=wrapper):
                model = wrapper(MyModel())
                if wrapper == offload_wrapper:
                    # 根据不同的 wrapper 类型，验证模型类型是否正确
                    self.assertTrue(isinstance(model, OffloadWrapper))
                else:
                    self.assertTrue(isinstance(model, CheckpointWrapper))
                # 验证关键字参数能够被正确传递
                inp = torch.ones(4, 10, requires_grad=True)
                out = model(inp, inp, c=inp, d=inp, e=inp, f=inp)
                self.assertTrue(isinstance(out, tuple))
                self.assertEqual(4, len(out))
                # 没有传递关键字参数的情况下，验证梯度要求是否等价
                out_no_kwarg = model(inp, inp, inp, inp)
                for t1, t2 in zip(out_no_kwarg, out):
                    self.assertEqual(t1, t2)
                    self.assertEqual(t1.requires_grad, t2.requires_grad)

        # 测试强制要求关键字输入的模型
        class ModelEnforceKwarg(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(10, 10)

            def forward(self, *, a=None, b=None):
                # 定义模型的前向传播逻辑，强制要求只通过关键字参数 a 和 b 输入
                return (self.lin(a), self.lin(b))

        model = checkpoint_wrapper(
            ModelEnforceKwarg(), checkpoint_impl=CheckpointImpl.REENTRANT
        )

        inp = torch.ones(4, 10, requires_grad=True)
        out = model(a=inp, b=inp)
        self.assertEqual(2, len(out))

    def test_checkpoint_wrapper_args_kwargs(self):
        """
        Tests that checkpoint_wrapper can pass down args / kwargs to configure
        torch.utils.checkpoint.
        """
        count = 0

        @contextlib.contextmanager
        def ctx_manager():
            nonlocal count
            count += 1
            yield

        def get_ctx_mgrs():
            return (ctx_manager(), ctx_manager())

        # kwargs 测试
        torch_utils_checkpoint = torch.utils.checkpoint.checkpoint
        m = checkpoint_wrapper(
            torch.nn.Linear(1, 1),
            checkpoint_fn=torch_utils_checkpoint,
            use_reentrant=False,
            context_fn=get_ctx_mgrs,
        )
        m(torch.randn(2, 1)).sum().backward()
        self.assertEqual(2, count)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_wrapper_parity(self):
        """
        Tests that using checkpoint_wrapper or the functional
        torch.utils.checkpoint (with the same reentrant config)
        results in the same maximum memory usage, i.e. they are
        equivalent memory usage wise.
        """

        # 定义一个模型类，继承自 nn.Module
        class Model(nn.Module):
            def __init__(
                self,
                n: int,
                use_cp: bool,
                use_wrapper: bool = False,
                use_reentrant: bool = True,
            ):
                super().__init__()
                self.layers = nn.ModuleList()  # 初始化一个空的模块列表
                self.n = n  # 设置层数
                self.use_cp = use_cp  # 是否使用 checkpointing
                self.use_wrapper = use_wrapper  # 是否使用包装器
                self.use_reentrant = use_reentrant  # 是否可重入

                # 定义一个部分函数应用，用于包装 checkpoint 函数
                wrp = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.REENTRANT
                    if use_reentrant
                    else CheckpointImpl.NO_REENTRANT,
                )

                # 循环创建 n 层线性模型，并加入到 self.layers 中
                for i in range(self.n):
                    l = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    use_checkpoint_wrapper = self.use_wrapper
                    if use_checkpoint_wrapper:
                        l = wrp(l)  # 使用包装器包装当前层模块
                    self.layers.append(l)

            # 前向传播函数定义
            def forward(self, x):
                for i in range(self.n):
                    if self.use_wrapper or not self.use_cp:
                        x = self.layers[i](x)  # 使用当前层模块处理输入 x
                    else:
                        x = checkpoint(
                            self.layers[i], x, use_reentrant=self.use_reentrant
                        )  # 使用 checkpoint 函数处理当前层模块和输入 x
                return x

        # 定义一个测试函数，测试不同配置下的内存使用情况
        def test(use_checkpointing, use_wrapper, use_reentrant):
            a = Model(
                8,
                use_checkpointing,
                use_wrapper=use_wrapper,
                use_reentrant=use_reentrant,
            ).cuda()  # 创建一个 Model 类的实例，并放在 GPU 上
            x = torch.randn(10000, 256, requires_grad=True).cuda()  # 创建一个张量 x
            torch.cuda.reset_peak_memory_stats()  # 重置 CUDA 设备上的内存统计信息
            loss = a(x).sum()  # 计算模型输出的损失并求和
            loss.backward()  # 反向传播计算梯度
            return torch.cuda.max_memory_allocated()  # 返回 CUDA 设备上的最大内存使用量

        # 分别测试不同配置下的内存使用情况，并断言两种方式的内存使用应相等
        functional_no_reentrant = test(
            use_checkpointing=True, use_wrapper=False, use_reentrant=False
        )
        wrapper_no_reentrant = test(
            use_checkpointing=False, use_wrapper=True, use_reentrant=False
        )
        self.assertEqual(functional_no_reentrant, wrapper_no_reentrant)

        functional_reentrant = test(
            use_checkpointing=True, use_wrapper=False, use_reentrant=True
        )
        wrapper_reentrant = test(
            use_checkpointing=False, use_wrapper=True, use_reentrant=True
        )
        self.assertEqual(functional_reentrant, wrapper_reentrant)
    # 测试函数：验证 CheckpointWrapper 对象能够正确地转发索引操作
    def test_forward_missing_attributes(self):
        # 创建一个包含单个线性层的序列模型
        lin = nn.Linear(1, 1)
        m = nn.Sequential(lin, lin)
        # 使用 CheckpointWrapper 封装序列模型
        wrapped = CheckpointWrapper(m)
        # 验证索引操作被正确转发到内部的线性层
        self.assertEqual(wrapped[0], lin)
        
        # 测试：验证丢失的属性被正确转发
        m._foo = "bar"
        self.assertEqual(wrapped._foo, "bar")

    # 测试函数：验证 checkpoint_wrapper 函数的全限定名称（FQN）功能
    def test_fqn(self):
        # 创建一个具有10个输入和10个输出的线性层，并使用 checkpoint_wrapper 封装
        lin = nn.Linear(10, 10, bias=False)
        lin = checkpoint_wrapper(lin)
        # 获取封装后模型的状态字典
        state_dict = lin.state_dict()
        # 遍历封装后模型的命名参数，并确保每个参数名在状态字典中
        for fqn, _ in lin.named_parameters():
            self.assertTrue(fqn in state_dict, msg=f"{fqn} not in state_dict.")

    # 跳过 CUDA 不可用情况的测试函数
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_wrapper_cpu_offload(self):
        # 创建一个包含三个线性层的序列模型，并将其放置在 CUDA 设备上
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        ).cuda()

        # 重写 saved_tensors_hooks 的初始化方法，以便测试期间将张量保留在 CPU 上
        def patched_init(saved_tensor_hook_obj, pack_hook, _):
            saved_tensor_hook_obj.pack_hook = pack_hook

            def testing_cpu_offload_unpack_hook(packed):
                _, tensor = packed
                return tensor

            saved_tensor_hook_obj.unpack_hook = testing_cpu_offload_unpack_hook

        # 保存原始的初始化方法，并用新的方法替换
        orig_init = torch.autograd.graph.saved_tensors_hooks.__init__
        torch.autograd.graph.saved_tensors_hooks.__init__ = patched_init

        # 使用 offload_wrapper 函数包装模型，以便在 CPU 上执行
        model = offload_wrapper(model)

        # 创建一个在 CUDA 设备上随机初始化的输入张量
        inp = torch.randn(3, 10, device="cuda")
        # 计算模型在输入上的损失
        loss = model(inp).sum()

        # 验证所有 autograd 保存的张量都已经被转移到 CPU 上
        offload_verified = False

        # 深度优先搜索函数，用于遍历损失函数的梯度图
        def dfs(grad_fn):
            for e in dir(grad_fn):
                if not e.startswith(_SAVED_PREFIX):
                    continue

                saved = getattr(grad_fn, e)
                if isinstance(saved, torch.Tensor):
                    # 断言保存的张量在 CPU 上
                    self.assertEqual(torch.device("cpu"), saved.device)
                    nonlocal offload_verified
                    offload_verified = True

            if hasattr(grad_fn, GRAD_FN_NEXT_FUNCTIONS):
                for next_grad_fn, _ in grad_fn.next_functions:
                    dfs(next_grad_fn)

        # 执行深度优先搜索，检查梯度函数中的所有保存的张量
        dfs(loss.grad_fn)

        # 断言所有 autograd 保存的张量都已成功转移到 CPU 上
        self.assertTrue(offload_verified)

        # 恢复原始的 saved_tensors_hooks 初始化方法
        torch.autograd.graph.saved_tensors_hooks.__init__ = orig_init
# 如果当前模块被直接运行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests() 的函数来执行测试代码
    run_tests()
```