# `.\pytorch\test\inductor\test_distributed_patterns.py`

```py
# Owner(s): ["oncall: pt2"]

# 导入必要的库
import dataclasses  # 导入 dataclasses 库，用于数据类的支持
import functools    # 导入 functools 库，用于高阶函数（Higher-order functions）

import torch        # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch._dynamo import compiled_autograd  # 导入编译自动微分模块
from torch._dynamo.test_case import run_tests, TestCase  # 导入测试相关模块
from torch._dynamo.testing import CompileCounter  # 导入编译计数器
from torch.testing._internal.common_utils import IS_MACOS, skipIfRocm, skipIfXpu  # 导入测试工具函数
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, requires_gpu  # 导入测试工具函数

# Fake distributed
WORLD_SIZE = 2  # 设置虚拟的分布式环境，节点数量为 2


def init_fake_distributed(device="cpu"):
    @torch.no_grad
    def all_gather(t):
        # 在虚拟的分布式环境中，模拟 all_gather 操作
        return torch.cat([t] * WORLD_SIZE, 0)

    @torch.no_grad
    def reduce_scatter(t):
        # 在虚拟的分布式环境中，模拟 reduce_scatter 操作
        # 使用 clone() 是因为 reduce_scatter 的输入和输出不应该是别名（alias）
        return t.narrow(0, 0, t.size(0) // WORLD_SIZE).clone()

    def fw_pre_hook(mod, inp):
        if not compiled_autograd.compiled_autograd_enabled:
            # 如果没有启用编译自动微分，则使用较慢的复制路径而不是 torch.ops.fsdp.set_()
            mod.unsharded_weight.untyped_storage().resize_(
                mod.unsharded_weight.nelement() * mod.unsharded_weight.element_size()
            )
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                mod.unsharded_weight
            ):
                mod.unsharded_weight.copy_(all_gather(mod.sharded_weight))
        else:
            # 如果启用了编译自动微分，则使用 torch.ops.fsdp.set_()
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                mod.unsharded_weight
            ):
                torch.ops.fsdp.set_(
                    mod.unsharded_weight, all_gather(mod.sharded_weight)
                )
        mod.weight = mod.unsharded_weight

    # Forward:
    #   mod.sharded_weight = local_shard (always)
    #   Before:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    #   After:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather

    def fw_post_hook(mod, inp, out):
        # 在前向传播后的钩子中，将模型的权重设为分片权重
        mod.weight = mod.sharded_weight
        mod.unsharded_weight.untyped_storage().resize_(0)

    def bw_pre_hook(mod, gO):
        if not compiled_autograd.compiled_autograd_enabled:
            # 如果没有启用编译自动微分，则使用较慢的复制路径而不是 torch.ops.fsdp.set_()
            mod.unsharded_weight.untyped_storage().resize_(
                mod.unsharded_weight.nelement() * mod.unsharded_weight.element_size()
            )
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                mod.unsharded_weight
            ):
                mod.unsharded_weight.copy_(all_gather(mod.sharded_weight))
        else:
            # 如果启用了编译自动微分，则使用 torch.ops.fsdp.set_()
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                mod.unsharded_weight
            ):
                torch.ops.fsdp.set_(
                    mod.unsharded_weight, all_gather(mod.sharded_weight)
                )
        mod.weight = mod.unsharded_weight

    # Backward:
    #   mod.sharded_weight = local_shard (always)
    #   Before:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    #   After:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    
    # 定义一个后向传播的钩子函数，用于处理模型权重的分片和非分片的操作
    def bw_post_hook(mod, gI, gO):
        # 获取模型权重的梯度
        grad = mod.weight.grad
        # 对梯度进行reduce_scatter操作，减少数据量并合并梯度信息
        new_grad = reduce_scatter(grad)
        # 将模型的权重设置为分片后的权重
        mod.weight = mod.sharded_weight
        # 更新模型分片后的权重的梯度为reduce_scatter操作后的新梯度
        mod.weight.grad = new_grad
        # 将模型的非分片权重的存储空间大小调整为0
        mod.unsharded_weight.untyped_storage().resize_(0)
    
    # 设置随机种子为1234
    torch.manual_seed(1234)
    # 创建一个线性模型，输入维度为20，输出维度为10，无偏置，设备为指定的device
    m = nn.Linear(20, 10, bias=False, device=device)
    
    # 模拟 eager 模式的第一次迭代
    # 使用reduce_scatter函数计算并设置模型的分片权重
    m.sharded_weight = nn.Parameter(reduce_scatter(m.weight))
    # 使用all_gather函数收集模型的分片权重到非分片权重
    m.unsharded_weight = nn.Parameter(all_gather(m.sharded_weight))
    # 调整非分片权重的存储空间大小为0
    m.unsharded_weight.untyped_storage().resize_(0)
    # 删除原始的模型权重
    del m.weight
    
    # 注册前向传播的预处理钩子和后处理钩子
    m.register_full_backward_pre_hook(bw_pre_hook)
    m.register_full_backward_hook(bw_post_hook)
    m.register_forward_pre_hook(fw_pre_hook)
    m.register_forward_hook(fw_post_hook)
    
    # 返回注册了钩子函数的模型m和一个随机生成的tensor，用于演示
    return m, torch.rand(2, 20, requires_grad=True, device=device)
# 定义一个初始化模块的函数，带有反向传播钩子
def init_module_bw_hooks(allow_eager):
    # 定义前向传播前钩子函数
    def bw_pre_hook(mod, gO):
        # 如果不允许急切执行，则要求 torch._dynamo.is_compiling() 返回 True
        assert allow_eager or torch._dynamo.is_compiling()
        # 断言模块的权重大小为 (10, 10)
        assert mod.weight.size() == (10, 10)
        # 增加前向传播前钩子计数
        mod.hook_count_pre.add_(1)
        # 返回修改后的梯度作为元组的形式
        return (torch.sin(gO[0] + 1.2),)

    # 定义前向传播后钩子函数
    def bw_post_hook(mod, gI, gO):
        # 如果不允许急切执行，则要求 torch._dynamo.is_compiling() 返回 True
        assert allow_eager or torch._dynamo.is_compiling()
        # 断言模块的权重大小为 (10, 10)
        assert mod.weight.size() == (10, 10)
        # 增加前向传播后钩子计数
        mod.hook_count_post.add_(1)
        # 返回修改后的梯度作为元组的形式
        return (torch.sin(gI[0] + 3.4),)

    # 设置随机种子为 1234
    torch.manual_seed(1234)
    # 创建一个线性层模块，输入和输出维度均为 10
    m = nn.Linear(10, 10)
    # 初始化前向传播前和后钩子的计数
    m.hook_count_pre = torch.tensor(0)
    m.hook_count_post = torch.tensor(0)
    # 注册前向传播前完整钩子函数
    m.register_full_backward_pre_hook(bw_pre_hook)
    # 注册前向传播后完整钩子函数
    m.register_full_backward_hook(bw_post_hook)
    # 返回模块 m 和一个随机生成的输入张量
    return m, torch.rand(2, 10, requires_grad=True)


# 执行多步训练过程
def steps(m, inp):
    # 执行四次循环
    for _ in range(4):
        # 模块 m 对输入 inp 进行前向传播
        out = m(inp)
        # 对输出 out 的所有元素求和并执行反向传播
        out.sum().backward()
    # 返回最终的输出张量 out
    return out


# 分布式模式测试类
class DistributedPatternTests(TestCase):
    # 测试使用闭包的中间钩子
    def test_intermediate_hook_with_closure(self):
        # 定义一个自定义的数据类 CustomObj
        @dataclasses.dataclass
        class CustomObj:
            val: torch.Tensor

        # 定义一个函数 fn，接受输入 x 和对象 obj
        def fn(x, obj):
            # 计算 x 的正弦值并赋给 y
            y = x.sin()
            # 计算闭包变量 closure_var
            closure_var = y + 1
            # 注册一个钩子函数，修改梯度 grad
            y.register_hook(lambda grad: grad + obj.val + closure_var)
            # 计算 y 的正弦值并返回结果 z
            z = y.sin()
            return z

        # 使用 torch.compile 编译函数 fn，并开启完整图模式
        opt = torch.compile(fn, fullgraph=True)

        # 创建两个 CustomObj 对象 obj1 和 obj2
        obj1 = CustomObj(torch.tensor(88))
        obj2 = CustomObj(torch.tensor(99))
        # 创建四个张量 x0, x1, x2, x3，要求计算梯度
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        # 分别对 x0 和 x1 调用 fn 函数进行计算和反向传播
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        # 使用编译的 autograd 功能对 x2 和 x3 调用 opt 函数进行计算和反向传播
        with compiled_autograd.enable(functools.partial(torch.compile, fullgraph=True)):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()

        # 断言 x0 和 x2 的梯度相等
        self.assertEqual(x0.grad, x2.grad)
        # 断言 x1 和 x3 的梯度相等
        self.assertEqual(x1.grad, x3.grad)

    # 在无梯度计算下测试存储器大小调整为零的情况
    @torch.no_grad()
    def _test_storage_resize_zero(self, device):
        # 使用 torch.compile 编译函数 fn，并开启完整图模式
        @torch.compile(fullgraph=True)
        def fn(x):
            # 计算 x 的正弦值并赋给 y
            y = torch.sin(x)
            # 调整 x 的未类型化存储为零大小
            x.untyped_storage().resize_(0)
            # 返回 y 的余弦值
            return torch.cos(y)

        # 创建一个张量 x，使用指定的设备
        x = torch.randn(10, device=device)
        # 计算预期结果，为 x 的正弦值的余弦值
        expected = torch.cos(torch.sin(x))
        # 执行 fn 函数计算并赋给 y
        y = fn(x)
        # 断言计算结果 y 与预期结果 expected 相等
        self.assertEqual(y, expected)
        # 断言 x 的未类型化存储大小为零
        self.assertEqual(x.untyped_storage().size(), 0)

    # 在 CPU 上测试存储器大小调整为零的情况
    def test_storage_resize_zero_cpu(self):
        self._test_storage_resize_zero("cpu")

    # 在 GPU 上测试存储器大小调整为零的情况
    @skipIfRocm
    @requires_gpu()
    def test_storage_resize_zero_gpu(self):
        self._test_storage_resize_zero(GPU_TYPE)

    # 标记为无梯度计算
    @torch.no_grad()
    # 定义测试方法，测试在指定设备上的存储大小调整非零情况
    def _test_storage_resize_nonzero(self, device):
        # 声明编译函数，全图模式
        @torch.compile(fullgraph=True)
        def fn(x, out):
            # 计算 x 的正弦值
            y = torch.sin(x)
            # 断言输出张量的未命名存储空间大小为零
            assert out.untyped_storage().size() == 0
            # 调整输出张量的未命名存储空间大小为 x 的未命名存储空间大小
            out.untyped_storage().resize_(x.untyped_storage().size())
            # 将 y 的余弦值复制到 out 中
            out.copy_(y.cos())

        # 生成一个随机张量 x
        x = torch.randn(10, device=device)
        # 生成一个随机张量 out
        out = torch.randn(10, device=device)
        # 生成期望的结果，即 x 的正弦值的余弦值
        expected = torch.cos(torch.sin(x))
        # 调整 out 的未命名存储空间大小为零
        out.untyped_storage().resize_(0)
        # 调用 fn 函数进行计算
        fn(x, out)
        # 断言输出张量的未命名存储空间大小等于 x 的未命名存储空间大小
        self.assertEqual(out.untyped_storage().size(), x.untyped_storage().size())
        # 断言 out 的值等于期望的结果
        self.assertEqual(out, expected)

    # 测试在 CPU 上的存储大小调整非零情况
    def test_storage_resize_nonzero_cpu(self):
        self._test_storage_resize_nonzero("cpu")

    # 根据条件跳过 ROCm 平台测试，在 GPU 上的存储大小调整非零情况
    @skipIfRocm
    @requires_gpu()
    def test_storage_resize_nonzero_gpu(self):
        self._test_storage_resize_nonzero(GPU_TYPE)

    # 无梯度计算上下文环境中测试不安全的版本计数设置方法1
    @torch.no_grad()
    def test_unsafe_set_version_counter1(self):
        # 创建编译计数器对象
        cnt = CompileCounter()

        # 声明编译函数，使用 cnt 作为后端，全图模式
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(w, x):
            # 计算 x 的正弦值
            x = x.sin()
            # 获取 w 的版本号
            v = w._version
            # 将 x 加 1 后赋值给 w
            w.copy_(x + 1)
            # 不安全地设置 w 的版本号为 v
            torch._C._autograd._unsafe_set_version_counter(w, v)
            return w, v

        # 针对版本号 v 进行循环测试
        for v in (3, 0, 1):
            # 生成一个随机张量 w1
            w1 = torch.randn(16)
            # 逐步递增 w1 的版本号，直到达到 v
            for i in range(v):
                w1.fill_(i)
            # 断言 w1 的版本号为 v
            self.assertEqual(w1._version, v)
            # 生成一个随机张量 x1
            x1 = torch.randn(16)
            # 调用 fn 函数计算结果
            w2, v2 = fn(w1, x1)

            # 断言 w2 是 w1 的同一实例
            self.assertIs(w1, w2)
            # 断言 w1 等于 x1 的正弦值加 1
            self.assertEqual(w1, x1.sin() + 1)
            # 断言 v2 等于 v
            self.assertEqual(v2, v)
            # 断言 w1 的版本号为 v
            self.assertEqual(w1._version, v)
            # 断言编译计数器的帧计数为 1
            self.assertEqual(cnt.frame_count, 1)

    # 测试不安全的版本计数设置方法2
    def test_unsafe_set_version_counter2(self):
        # 声明编译函数，使用 "inductor" 作为后端，全图模式
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(w, x):
            # 计算 w 的正弦值
            r = w.sin()
            # 在无梯度计算上下文中
            with torch.no_grad():
                # 获取 w 的版本号
                v = w._version
                # 将 x 复制给 w
                w.copy_(x)
                # 不安全地设置 w 的版本号为 v
                torch._C._autograd._unsafe_set_version_counter(w, v)
            return r

        # 生成一个随机张量 w1，要求计算梯度
        w1 = torch.randn(1, requires_grad=True)
        # 生成一个随机张量 x1
        x1 = torch.randn(1)
        # 期望的结果是 w1 的去除梯度后的正弦值
        expected_r1 = w1.detach().sin()

        # 调用 fn 函数计算结果
        r1 = fn(w1, x1)
        # 对 r1 进行反向传播
        r1.backward()
        # 断言 r1 等于期望的结果
        self.assertEqual(r1, expected_r1)
        # 断言 w1 等于 x1
        self.assertEqual(w1, x1)
        # 断言 w1 的梯度等于 x1 的余弦值
        self.assertEqual(w1.grad, x1.cos())

    # 在无梯度计算上下文环境中测试不安全的版本计数保留方法1
    @torch.no_grad()
    def test_unsafe_preserve_version_counter1(self):
        # 声明编译函数，使用 "eager" 作为后端，全图模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(w, x):
            # 计算 x 的正弦值
            x = x.sin()
            # 在自动求导中不安全地保留 w 的版本号
            with torch.autograd._unsafe_preserve_version_counter(w):
                # 将 x 加 1 后赋值给 w
                w.copy_(x + 1)
            return w

        # 生成一个随机张量 w1，并先填充 0，再填充 1
        w1 = torch.randn(16).fill_(0).fill_(1)
        # 生成一个随机张量 x1
        x1 = torch.randn(16)
        # 记录 w1 的版本号
        v1 = w1._version
        # 调用 fn 函数计算结果
        w2 = fn(w1, x1)
        # 获取 w1 的版本号
        v2 = w1._version

        # 断言 w2 是 w1 的同一实例
        self.assertIs(w1, w2)
        # 断言 w1 等于 x1 的正弦值加 1
        self.assertEqual(w1, x1.sin() + 1)
        # 断言 w1 的版本号不变
        self.assertEqual(v1, v2)
    def test_unsafe_preserve_version_counter2(self):
        # 定义一个测试函数，用于测试在特定条件下的版本计数器保留行为
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(w, x):
            # 计算 sin(w)，并保存结果为 r
            r = w.sin()
            # 在没有梯度计算的上下文中，使用 _unsafe_preserve_version_counter 保留版本计数器
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(w):
                # 将 x 的值复制给 w
                w.copy_(x)
            # 返回 sin(w) 的结果 r
            return r

        # 创建一个需要梯度的随机张量 w1 和一个随机张量 x1
        w1 = torch.randn(1, requires_grad=True)
        x1 = torch.randn(1)
        # 计算预期的结果 r1，即 sin(w1) 的值
        expected_r1 = w1.detach().sin()

        # 调用 fn 函数计算结果 r1
        r1 = fn(w1, x1)
        # 对 r1 进行反向传播
        r1.backward()
        # 断言 r1 的值与预期的结果 expected_r1 相等
        self.assertEqual(r1, expected_r1)
        # 断言 w1 的值与 x1 相等
        self.assertEqual(w1, x1)
        # 断言 w1 的梯度值为 x1 的余弦值
        self.assertEqual(w1.grad, x1.cos())

    def test_module_backward_hooks_eager(self):
        # 初始化带有反向钩子的模块 m1 和输入 inp1
        m1, inp1 = init_module_bw_hooks(True)
        # 对模块 m1 进行计算并得到输出 out1
        out1 = steps(m1, inp1)

        # 初始化带有反向钩子的模块 m2 和输入 inp2
        m2, inp2 = init_module_bw_hooks(False)
        # 使用编译器计数器进行自动微分，将 m2 编译为全图计算的模式
        fw_cnt = CompileCounter()
        bw_cnt = CompileCounter()
        with compiled_autograd.enable(torch.compile(backend=bw_cnt, fullgraph=True)):
            m2 = torch.compile(m2, backend=fw_cnt, fullgraph=True)
            # 对模块 m2 进行计算并得到输出 out2
            out2 = steps(m2, inp2)

        # 断言 m1 的前钩子计数与 m2 相等
        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        # 断言 m1 的后钩子计数与 m2 相等
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        # 断言 out1 与 out2 的值相等
        self.assertEqual(out1, out2)
        # 断言 inp1 和 inp2 的梯度值相等
        self.assertEqual(inp1.grad, inp2.grad)
        # 断言 m1 的权重梯度与 m2 相等
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        # 断言 m1 的偏置梯度与 m2 相等
        self.assertEqual(m1.bias.grad, m2.bias.grad)

        # 断言前向计数器的帧数为 1
        self.assertEqual(fw_cnt.frame_count, 1)
        # 断言前向计数器的操作数为 5
        self.assertEqual(fw_cnt.op_count, 5)
        # 断言后向计数器的帧数为 2，分别表示梯度为 None 和梯度不为 None 的情况
        self.assertEqual(bw_cnt.frame_count, 2)
        # 断言后向计数器的操作数为 48
        self.assertEqual(bw_cnt.op_count, 48)

    def test_module_backward_hooks_aot(self):
        # 初始化带有反向钩子的模块 m1 和输入 inp1
        m1, inp1 = init_module_bw_hooks(True)
        # 对模块 m1 进行计算并得到输出 out1
        out1 = steps(m1, inp1)

        # 初始化带有反向钩子的模块 m2 和输入 inp2
        m2, inp2 = init_module_bw_hooks(True)
        # 使用 AOT（Ahead-of-Time）模式对 m2 进行编译
        m2 = torch.compile(m2, backend="aot_eager", fullgraph=True)
        with compiled_autograd.enable(lambda gm: gm):
            # 对模块 m2 进行计算并得到输出 out2
            out2 = steps(m2, inp2)

        # 断言 m1 的前钩子计数与 m2 相等
        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        # 断言 m1 的后钩子计数与 m2 相等
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        # 断言 out1 与 out2 的值相等
        self.assertEqual(out1, out2)
        # 断言 inp1 和 inp2 的梯度值相等
        self.assertEqual(inp1.grad, inp2.grad)
        # 断言 m1 的权重梯度与 m2 相等
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        # 断言 m1 的偏置梯度与 m2 相等
        self.assertEqual(m1.bias.grad, m2.bias.grad)

    def test_module_backward_hooks_inductor(self):
        # 初始化带有反向钩子的模块 m1 和输入 inp1
        m1, inp1 = init_module_bw_hooks(True)
        # 对模块 m1 进行计算并得到输出 out1
        out1 = steps(m1, inp1)

        # 初始化带有反向钩子的模块 m2 和输入 inp2
        m2, inp2 = init_module_bw_hooks(False)
        # 使用编译器将 m2 编译为全图计算的模式
        m2 = torch.compile(m2, fullgraph=True)
        with compiled_autograd.enable(torch.compile(fullgraph=True)):
            # 对模块 m2 进行计算并得到输出 out2
            out2 = steps(m2, inp2)

        # 断言 m1 的前钩子计数与 m2 相等
        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        # 断言 m1 的后钩子计数与 m2 相等
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        # 断言 out1 与 out2 的值相等
        self.assertEqual(out1, out2)
        # 断言 inp1 和 inp2 的梯度值相等
        self.assertEqual(inp1.grad, inp2.grad)
        # 断言 m1 的权重梯度与 m2 相等
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        # 断言 m1 的偏置梯度与 m2 相等
        self.assertEqual(m1.bias.grad, m2.bias.grad)
    # 测试多层模块反向钩子的功能
    def test_module_backward_hooks_multi_layers(self):
        # 初始化具有反向钩子的模块和输入
        a1, inp1 = init_module_bw_hooks(True)
        b1, _ = init_module_bw_hooks(True)
        # 在序列模块中应用模块a1和b1，并执行步骤
        out1 = steps(torch.nn.Sequential(a1, b1), inp1)

        # 初始化不具有反向钩子的模块和输入
        a2, inp2 = init_module_bw_hooks(False)
        b2, _ = init_module_bw_hooks(False)
        # 使用编译加速器开启完整图模式，将模块a2和b2编译为图模式，并执行步骤
        with compiled_autograd.enable(torch.compile(fullgraph=True)):
            out2 = steps(
                torch.compile(torch.nn.Sequential(a2, b2), fullgraph=True), inp2
            )

        # 断言钩子计数相等
        self.assertEqual(a1.hook_count_pre, a2.hook_count_pre)
        self.assertEqual(a1.hook_count_post, a2.hook_count_post)
        self.assertEqual(b1.hook_count_pre, b2.hook_count_pre)
        self.assertEqual(b1.hook_count_post, b2.hook_count_post)
        # 断言输出相等
        self.assertEqual(out1, out2)
        # 断言梯度相等
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(a1.weight.grad, a2.weight.grad)
        self.assertEqual(a1.bias.grad, a2.bias.grad)
        self.assertEqual(b1.weight.grad, b2.weight.grad)
        self.assertEqual(b1.bias.grad, b2.bias.grad)

    # TODO(jansel): support bw hooks with graph break

    # 断言参数和梯度相同的辅助方法
    def _assert_same_grad(self, a, b):
        self.assertEqual(type(a), type(b))
        self.assertEqual(a, b)
        self.assertEqual(a.grad, b.grad)
        self.assertEqual(a.requires_grad, b.requires_grad)

    # 测试带有返回参数的函数1
    def test_nn_param_return1(self):
        def fn(x):
            # 创建具有梯度的参数p
            p = torch.nn.Parameter(x)
            return p, p.sin()

        # 使用编译加速器编译函数fn，并开启完整图模式
        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        # 执行函数fn，计算梯度并进行反向传播
        p1, r1 = fn(x1)
        r1.sum().backward()
        # 使用编译后的优化器opt执行函数fn，计算梯度并进行反向传播
        p2, r2 = opt(x2)
        r2.sum().backward()
        # 断言参数和梯度相同
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    # 测试带有返回参数的函数2
    def test_nn_param_return2(self):
        def fn(x):
            # 创建不需要梯度的参数p
            p = torch.nn.Parameter(x, requires_grad=False)
            return p, x + 1

        # 使用编译加速器编译函数fn，并开启完整图模式
        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        # 执行函数fn
        p1, r1 = fn(x1)
        # 使用编译后的优化器opt执行函数fn
        p2, r2 = opt(x2)
        # 断言参数和梯度相同
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    # 测试带有返回参数的函数3
    def test_nn_param_return3(self):
        def fn(x):
            # 创建具有偏置的参数p
            p = torch.nn.Parameter(x + 123)
            return p, p.sin()

        # 使用编译加速器编译函数fn，并开启完整图模式
        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        # 执行函数fn，计算梯度并进行反向传播
        p1, r1 = fn(x1)
        r1.sum().backward()
        # 使用编译后的优化器opt执行函数fn，计算梯度并进行反向传播
        p2, r2 = opt(x2)
        r2.sum().backward()
        # 断言参数和梯度相同
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    # 测试带有返回参数的函数4
    def test_nn_param_return4(self):
        def fn(x):
            # 创建不需要梯度的参数p
            p = torch.nn.Parameter(x + 123, requires_grad=False)
            return p, x + 1

        # 使用编译加速器编译函数fn，并开启完整图模式
        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        # 执行函数fn
        p1, r1 = fn(x1)
        # 使用编译后的优化器opt执行函数fn
        p2, r2 = opt(x2)
        # 断言参数和梯度相同
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    # 通过functorch配置，支持重新计算视图
    @torch._functorch.config.patch(recompute_views=True)
    # 定义测试函数，用于测试假分布式模型在AOT Eager模式下的行为
    def test_fake_distributed_aot_eager(self):
        # 初始化第一个假分布式模型和输入
        m1, inp1 = init_fake_distributed()
        # 对第一个模型执行计算步骤，得到输出
        out1 = steps(m1, inp1)

        # 初始化第二个假分布式模型和输入
        m2, inp2 = init_fake_distributed()
        # 使用AOT Eager后端编译第二个模型的全图
        m2 = torch.compile(m2, backend="aot_eager", fullgraph=True)
        # 创建一个编译计数器
        bw_cnt = CompileCounter()
        # 启用编译自动求导上下文管理器，使用编译后端计数器
        with compiled_autograd.enable(torch.compile(backend=bw_cnt, fullgraph=True)):
            # 对第二个模型执行计算步骤，得到输出
            out2 = steps(m2, inp2)

        # 断言第一个和第二个模型的权重梯度相同
        self._assert_same_grad(m1.weight, m2.weight)
        # 断言第一个和第二个输入的梯度相同
        self._assert_same_grad(inp1, inp2)
        # 断言第一个和第二个输出的梯度相同
        self._assert_same_grad(out1, out2)
        # 断言编译计数器中的帧计数为2
        self.assertEqual(bw_cnt.frame_count, 2)

    # 跳过在ROCm环境下的测试
    # 跳过在XPU环境下的测试
    # 要求GPU环境的测试
    # 在Functorch配置中打补丁，使得重新计算视图为True
    @skipIfRocm
    @skipIfXpu
    @requires_gpu()
    @torch._functorch.config.patch(recompute_views=True)
    # 定义测试函数，用于测试假分布式感应器的行为
    def test_fake_distributed_inductor(self):
        # TODO: fix .set_ lowering in CPU inductor, and enable the CPU test.
        # 初始化第一个假分布式模型和输入，使用指定的GPU类型
        m1, inp1 = init_fake_distributed(GPU_TYPE)
        # 对第一个模型执行计算步骤，得到输出
        out1 = steps(m1, inp1)

        # 初始化第二个假分布式模型和输入，使用指定的GPU类型
        m2, inp2 = init_fake_distributed(GPU_TYPE)
        # 编译第二个模型的全图
        m2 = torch.compile(m2, fullgraph=True)
        # 启用编译自动求导上下文管理器，使用编译后端
        with compiled_autograd.enable(torch.compile(fullgraph=True)):
            # 对第二个模型执行计算步骤，得到输出
            out2 = steps(m2, inp2)

        # 断言第一个和第二个模型的权重梯度相同
        self._assert_same_grad(m1.weight, m2.weight)
        # 断言第一个和第二个输入的梯度相同
        self._assert_same_grad(inp1, inp2)
        # 断言第一个和第二个输出的梯度相同
        self._assert_same_grad(out1, out2)
# 如果当前脚本是作为主程序运行（而不是被导入其他模块），则执行以下代码块
if __name__ == "__main__":
    # 如果系统具有 CPU 资源且不是 macOS 系统
    if HAS_CPU and not IS_MACOS:
        # 运行测试，要求其中包含 "filelock" 的测试
        run_tests(needs="filelock")
```