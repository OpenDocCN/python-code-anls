# `.\pytorch\test\distributed\test_data_parallel.py`

```
# Owner(s): ["oncall: distributed"]

# 引入上下文管理、函数工具、IO操作、有序字典和深拷贝等必要库
import contextlib
import functools
import io
from collections import OrderedDict
from copy import deepcopy
from itertools import product

# 引入 PyTorch 相关模块
import torch
import torch.nn.functional as F
import torch.nn.parallel as dp
from torch import nn
from torch.cuda.amp import autocast
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_utils import (
    _assertGradAndGradgradChecks,
    dtype2prec_DONTUSE,
    gradcheck,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)

# 检查是否存在 NCCL 进程组，如果不存在则设置标志
NO_NCCL = not hasattr(torch.distributed, "ProcessGroupNCCL")

# 禁用批次梯度检查
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
_assertGradAndGradgradChecks = functools.partial(
    _assertGradAndGradgradChecks, check_batched_grad=False
)

# 定义测试用例类
class TestDataParallel(TestCase):
    # 如果不支持多GPU，则跳过测试
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_buffers_requiring_grad(self):
        # 定义测试模块
        class TestModule(nn.Module):
            def __init__(self, t):
                super().__init__()
                # 注册需要梯度的缓冲区和不需要梯度的缓冲区
                self.register_buffer("t_rg", t)
                self.register_buffer("t_not_rg", t.clone().detach())

            # 前向传播函数
            def forward(self, x):
                return x * self.t_rg + self.t_not_rg

        # 创建测试模块实例，传入需要梯度的张量
        m = TestModule(
            torch.randn(100, device="cuda", requires_grad=True, dtype=torch.double)
        )
        # 断言需要梯度的缓冲区确实需要梯度
        self.assertTrue(m.t_rg.requires_grad)

        # 使用 DataParallel 封装模块，指定使用的 GPU 设备
        dpm = nn.DataParallel(m, [0, 1])
        # 创建输入张量
        inp = torch.randn(2, 100, device="cuda", dtype=torch.double)

        # 定义函数，用于梯度检查
        def fn(t):
            return dpm(inp)

        # 执行梯度检查，传入需要梯度的缓冲区
        gradcheck(fn, (m.t_rg,))

    # 如果不支持多GPU，则跳过测试
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_rnn(self):
        # 定义测试用的神经网络模块
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 LSTM 模型，输入维度300，隐藏层维度1024，单层，批处理优先，双向
                self.rnn = torch.nn.LSTM(
                    300, 1024, 1, batch_first=True, bidirectional=True
                )

            def forward(self, x):
                # 展开 LSTM 参数以提高效率
                self.rnn.flatten_parameters()
                return self.rnn(x)

        # 定义执行单步优化的函数
        def step(model):
            # 使用 SGD 优化器，学习率为10
            opt = torch.optim.SGD(model.parameters(), lr=10)
            # 创建输入张量，全1，大小为(4, 4, 300)，发送到GPU设备0
            input = torch.ones(4, 4, 300).to(0)
            # 模型前向传播
            output = model(input)
            # 计算输出与全零张量之间的均方误差损失
            loss = F.mse_loss(output[0], torch.zeros_like(output[0]))
            # 反向传播损失
            loss.backward()
            # 执行优化步骤
            opt.step()

        # 禁用梯度计算上下文
        with torch.no_grad():
            # 创建测试模型并将其移动到GPU设备0
            model = TestModule().to(0)
            # 使用DataParallel复制模型，并放置在GPU上
            model_dp = torch.nn.DataParallel(deepcopy(model))

            # 确保在禁用梯度时DataParallel不会崩溃
            # 参见GitHub问题#21108
            model_dp(torch.rand(2, 4, 300).to(0))

        # 分别对原始模型和DataParallel模型执行优化步骤
        step(model)
        step(model_dp)

        # 检查每个参数对是否在数值上接近
        for p1, p2 in zip(model.parameters(), model_dp.parameters()):
            self.assertTrue(p1.allclose(p2))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_lazy_linear(self):
        # 使用断言检查是否抛出预期的值错误信息
        with self.assertRaisesRegex(
            ValueError, "Attempted to use an uninitialized parameter"
        ):
            # 使用DataParallel并将其放置在GPU 0上的LazyLinear模块
            model_dp = torch.nn.DataParallel(torch.nn.LazyLinear(10).to(0))
            model_dp(torch.rand(10, 10).to(0))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parallel_apply(self):
        # 创建两个线性层，并分别放置在cuda:0和cuda:1上
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        # 创建输入张量i1和i2，分别放置在cuda:0和cuda:1上
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        # 预期的输出expected1和expected2分别是l1(i1)和l2(i2)
        expected1 = l1(i1)
        expected2 = l2(i2)
        # modules包含l1和l2，expected_outputs包含expected1和expected2
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # 对于每种输入方式，执行parallel_apply函数，并比较输出是否符合预期
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            outputs = dp.parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)
    def test_parallel_apply_autocast(self):
        # 创建一个线性层 l1，其输入大小为10，输出大小为5，放置在 "cuda:0" 设备上，数据类型为 torch.float
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        # 创建另一个线性层 l2，其输入大小为10，输出大小为5，放置在 "cuda:1" 设备上，数据类型为 torch.float
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        # 生成一个形状为 (2, 10) 的随机张量 i1，放置在 "cuda:0" 设备上，数据类型为 torch.float
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        # 生成一个形状为 (2, 10) 的随机张量 i2，放置在 "cuda:1" 设备上，数据类型为 torch.float
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        
        # 使用自动混合精度装饰器 autocast
        with autocast():
            # 对输入 i1 应用线性层 l1，得到预期输出 expected1
            expected1 = l1(i1)
            # 对输入 i2 应用线性层 l2，得到预期输出 expected2
            expected2 = l2(i2)
        
        # 定义模块列表 modules 包括 l1 和 l2
        modules = (l1, l2)
        # 预期输出列表 expected_outputs 包括 expected1 和 expected2
        expected_outputs = (expected1, expected2)

        # 对于每个输入 inputs 的集合，可以是位置参数的集合
        # 或者表示单个参数的对象
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            # 使用自动混合精度装饰器 autocast
            with autocast():
                # 对模块列表 modules 和输入 inputs 执行并行应用
                outputs = dp.parallel_apply(modules, inputs, None)
            # 对每个输出 out 和预期输出 expected 进行比较断言
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)

    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "CUDA unavailable")
    def test_parallel_apply_passes_exception(self):
        # 定义一个会抛出 KeyError 的测试模块
        class TestModule(nn.Module):
            def forward(self, *args):
                return {}["wonderful"]

        # 创建 TestModule 实例 l1，放置在 "cuda" 设备上，数据类型为 torch.float
        l1 = TestModule().to("cuda", torch.float)
        # 并检查 parallel_apply 是否传递了异常
        # （可以在此测试中两次使用同一个设备）
        with self.assertRaisesRegex(
            KeyError,
            "Caught KeyError in replica \\d "
            "on device 0.\nOriginal Traceback"
            "[\\s\\S]+wonderful",
        ):
            # 对模块列表 (l1, l1) 和输入 (None, None) 执行并行应用
            dp.parallel_apply(modules=(l1, l1), inputs=(None, None))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_multiple_input(self):
        # 定义一个测试用的神经网络模块
        class TestModule(nn.Module):
            # 定义模块的前向传播方法，接受多个输入变量和一个浮点数，可选一个额外变量
            def forward(self, var1, var2, float1, var3=None):
                # 如果 var3 为 None，则返回 float1 乘以 var1 和 var2 的乘积
                if var3 is None:
                    return float1 * (var1 * var2)
                else:
                    # 否则返回 float1 乘以 var1、var2 和 var3 的和
                    return float1 * (var1 * var2 + var3)

        # 创建 TestModule 类的实例
        m = TestModule()
        # 创建需要用于计算的张量变量，包括需要计算梯度的 var1 和 var2
        var1 = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
        var2 = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
        # var3 是一个不需要计算梯度的张量
        var3 = torch.randn(5, 5, dtype=torch.float, requires_grad=False)

        # 创建一个随机的浮点数，作为模型计算时的参数
        float1 = torch.randn(1).item()

        # 计算预期输出
        expected = m(var1, var2, float1)
        # 计算损失函数，这里使用损失函数对预期输出求和
        loss = expected.sum()
        # 反向传播计算梯度
        loss.backward()
        # 克隆计算得到的 var1 和 var2 的梯度
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        # 定义一个局部测试函数，验证输出和梯度计算的正确性
        def local_test(out):
            with torch.no_grad():
                # 将 var1 和 var2 的梯度填充为零
                var1.grad.fill_(0.0)
                var2.grad.fill_(0.0)
            # 计算损失函数
            loss = out.sum()
            # 再次进行反向传播
            loss.backward()
            # 使用断言验证输出和梯度是否一致
            self.assertEqual(out, expected)
            self.assertEqual(gvar1_exp, var1.grad)
            self.assertEqual(gvar2_exp, var2.grad)

        # 使用数据并行处理模块 dp 对输入进行并行计算，设备 (0, 1) 上进行计算
        out = dp.data_parallel(m, (var1, var2, float1), (0, 1))
        # 调用局部测试函数验证输出和梯度
        local_test(out)

        # 交换设备顺序 (1, 0) 进行并行计算
        out = dp.data_parallel(m, (var1, var2, float1), (1, 0))
        # 调用局部测试函数验证输出和梯度
        local_test(out)

        # 在设备 (0,) 上进行并行计算
        out = dp.data_parallel(m, (var1, var2, float1), (0,))
        # 调用局部测试函数验证输出和梯度
        local_test(out)

        # 清零 var1 和 var2 的梯度
        with torch.no_grad():
            var1.grad.fill_(0.0)
            var2.grad.fill_(0.0)
        # 使用额外的 var3 进行模型计算
        expected = m(var1, var2, float1, var3=var3)
        # 计算损失函数
        loss = expected.sum()
        # 反向传播计算梯度
        loss.backward()
        # 克隆计算得到的 var1 和 var2 的梯度
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        # 使用 nn.DataParallel 将 TestModule 进行数据并行处理
        dpm = nn.DataParallel(TestModule())
        # 在多 GPU 上计算模型输出
        out = dpm(var1, var2, float1, var3=var3)
        # 调用局部测试函数验证输出和梯度
        local_test(out)

        # 使用指定设备 (0) 进行数据并行处理
        dpm = nn.DataParallel(TestModule(), device_ids=[0])
        # 在指定设备上计算模型输出
        out = dpm(var1, var2, float1, var3=var3)
        # 调用局部测试函数验证输出和梯度
        local_test(out)

        # 将参数作为关键字参数传递给 dp.data_parallel 函数
        kwarg_wrap = {"var3": var3}
        out = dp.data_parallel(
            m, (var1, var2, float1), (0, 1), module_kwargs=kwarg_wrap
        )
        # 调用局部测试函数验证输出和梯度
        local_test(out)

        # 在设备 (0,) 上并行计算，使用关键字参数 var3
        out = dp.data_parallel(m, (var1, var2, float1), (0,), module_kwargs=kwarg_wrap)
        # 调用局部测试函数验证输出和梯度
        local_test(out)
    def test_data_parallel_model_device(self):
        r"""Test device[0] check at forward time."""
        # 创建一个线性层，输入维度为2，输出维度为2
        l = nn.Linear(2, 2)
        # 生成一个2x2的随机张量作为输入
        inp = torch.randn(2, 2)
        # 将输入数据移动到CUDA设备0上
        inp_cuda0 = inp.cuda(0)
        # 将输入数据移动到CUDA设备1上
        inp_cuda1 = inp.cuda(1)

        # 错误信息模板，用于检查模块参数和缓冲是否在指定设备上
        error_msg = "module must have its parameters and buffers on device {}"

        @contextlib.contextmanager
        def dummy_ctx_manager():
            yield

        # 定义测试函数，用于测试DataParallel模块和函数
        def test(inner_m, dp_device, inp, device_ids, should_fail):
            # 如果设备ID为None，则使用所有可用CUDA设备
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))

            # 确定预期设备
            if isinstance(device_ids[0], torch.device):
                expect_device = device_ids[0]
            else:
                expect_device = torch.device(f"cuda:{device_ids[0]}")

            # 如果应该失败
            if should_fail:
                # 定义断言函数，用于捕获RuntimeError并匹配错误消息
                def assert_correct():
                    return self.assertRaisesRegex(
                        RuntimeError, error_msg.format(expect_device)
                    )

            else:
                # 否则，使用空的上下文管理器
                assert_correct = dummy_ctx_manager

            # 测试DataParallel模块
            dpm = nn.DataParallel(inner_m, device_ids)
            if dp_device is not None:
                dpm = dpm.to(dp_device)

            # 使用断言函数上下文进行测试
            with assert_correct():
                dpm(inp)

            # 测试函数式接口
            with assert_correct():
                nn.parallel.data_parallel(inner_m.to(dp_device), inp, device_ids)

        # 不同测试用例
        test(l.to("cpu"), None, inp, None, should_fail=True)
        test(l.cuda(1), None, inp_cuda0, None, should_fail=True)
        test(l.cuda(), None, inp_cuda0, [1, 0], should_fail=True)

        test(l.cuda(), None, inp_cuda0, None, should_fail=False)
        test(l.cpu(), "cuda", inp_cuda0, None, should_fail=False)
        test(l.cuda(1), None, inp_cuda1, [1, 0], should_fail=False)
        test(l.cpu(), "cuda:1", inp_cuda1, [1, 0], should_fail=False)

        s = nn.Sequential(l.cpu())
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(deepcopy(l).cpu(), l.cuda())
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(l.cuda(), deepcopy(l).cuda(1))
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(l.cuda(), deepcopy(l).cuda())
        test(s, None, inp, None, should_fail=False)
        test(s, None, inp, [0, 1], should_fail=False)
        test(s, None, inp, [1, 0], should_fail=True)
        test(s.cpu(), None, inp, [1, 0], should_fail=True)
        test(s.cuda(1), None, inp, [1, 0], should_fail=False)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_model_no_refcycles(self):
        # Python 2.7 will create reference cycles with the following
        # Module on multiple GPUs, but Python 3 shouldn't unless
        # there are refcycles on the PyTorch side (or the defined module)
        import gc  # 导入垃圾回收模块

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        gc.collect()  # 手动触发一次垃圾回收
        model = nn.DataParallel(Model().cuda())  # 创建一个多GPU数据并行的模型
        data = torch.randn(1, device="cuda")  # 生成一个在GPU上的随机数据
        model(data)  # 对模型进行前向传播

        refcycles = gc.collect()  # 再次触发垃圾回收，并获取回收的对象数
        self.assertEqual(refcycles, 0)  # 断言此次回收的对象数为0，即无循环引用

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_no_grad(self):
        test = self

        class Layer(nn.Module):
            def forward(self, x):
                test.assertFalse(torch.is_grad_enabled())  # 断言梯度计算未开启
                return x

        l = Layer()  # 创建一个测试用的层
        i = torch.randn(20, 10, dtype=torch.float, device="cuda")  # 在GPU上生成随机数据
        with torch.no_grad():
            dp.data_parallel(l, i, (0, 1))  # 在多GPU上执行数据并行计算
        self.assertRaises(AssertionError, lambda: dp.data_parallel(l, i, (0, 1)))  # 断言此处梯度计算会抛出错误

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel(self):
        l = nn.Linear(10, 5).float().cuda()  # 创建一个在GPU上的线性层
        i = torch.randn(20, 10, dtype=torch.float, device="cuda:1")  # 在第二个GPU上生成随机数据
        l.cuda(1)  # 将线性层移动到第二个GPU
        expected_out = l(i)  # 对输入数据进行线性计算得到期望输出
        loss = expected_out.sum()  # 计算输出的和作为损失
        loss.backward()  # 反向传播计算梯度
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())  # 复制每个参数的梯度作为期望值
        dev_ids_list = [(0, 1), (1, 0)]  # 多GPU设备ID列表
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()  # 将模型移到指定GPU
                l.zero_grad()  # 清空梯度
                out = dp.data_parallel(l, i, dev_id)  # 在指定GPU上执行数据并行计算
                loss = out.sum()  # 计算输出的和作为损失
                loss.backward()  # 反向传播计算梯度
                self.assertEqual(out.get_device(), dev_id[0])  # 断言输出在指定的GPU上
                self.assertEqual(out, expected_out)  # 断言输出与期望输出一致
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad, expected)  # 断言每个参数的梯度与期望值一致

        # Check for None device_ids
        l = l.cuda()  # 将模型移到默认GPU
        out = dp.data_parallel(l, i)  # 在默认GPU上执行数据并行计算
    # 定义一个测试函数，用于测试稀疏模式下的数据并行处理
    def test_data_parallel_sparse(self):
        # 创建一个稀疏的 Embedding 层，大小为10x5，使用 sparse=True，将其移到 "cuda:1" 设备
        l = nn.Embedding(10, 5, sparse=True).to("cuda:1")
        # 在 "cuda:1" 设备上生成一个大小为 (20, 5) 的长整型张量 i，其值为从0到9的随机整数
        i = torch.randint(10, (20, 5), device="cuda:1", dtype=torch.long)
        # 通过 Embedding 层计算输入 i 的预期输出
        expected_out = l(i)
        # 计算输出的总和作为损失
        loss = expected_out.sum()
        # 反向传播损失
        loss.backward()
        
        # 初始化一个空列表，用于存储预期的梯度
        expected_grads = []
        # 遍历 Embedding 层的参数，克隆每个参数的梯度并添加到 expected_grads 中
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        
        # 定义设备 ID 的列表
        dev_ids_list = [(0, 1), (1, 0)]
        # 遍历设备 ID 列表
        for dev_id in dev_ids_list:
            # 使用指定的设备 ID 切换到相应的 CUDA 设备上
            with torch.cuda.device(dev_id[0]):
                # 将 Embedding 层移动到当前 CUDA 设备
                l.cuda()
                # 清除 Embedding 层的梯度
                l.zero_grad()
                # 在指定的设备上进行数据并行处理，计算输出
                out = dp.data_parallel(l, i, dev_id)
                # 计算输出的总和作为损失
                loss = out.sum()
                # 反向传播损失
                loss.backward()
                # 断言输出张量的设备 ID 与当前设备 ID 相符
                self.assertEqual(out.get_device(), dev_id[0])
                # 断言输出与预期输出相等
                self.assertEqual(out, expected_out)
                # 遍历预期梯度和 Embedding 层的参数，断言它们的稀疏表示是否相等
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.coalesce(), expected.coalesce())

        # 在 "cuda:1" 设备上重新分配 Embedding 层
        l = l.cuda()
        # 使用数据并行处理在多个 GPU 上计算输出
        out = dp.data_parallel(l, i)

    # 跳过但在 Sandcastle 中通过如果不支持多 GPU，否则测试多层嵌套输出的数据并行处理
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_output(self):
        # 定义一个函数 fn，接受一个输入并返回嵌套的输出
        def fn(input):
            return [
                input,
                (input.sin(), input.cos(), [input.add(1)]),
                input,
                OrderedDict(a=input, b=[input.sin()]),
            ]
        
        # 定义一个继承自 nn.Module 的网络类 Net，重写 forward 方法，调用 fn 函数
        class Net(nn.Module):
            def forward(self, input):
                return fn(input)
        
        # 在 "cuda:1" 设备上生成一个形状为 (2, 2) 的浮点数张量 i
        i = torch.randn(2, 2).float().cuda(1)
        # 获取所有可用的 GPU 设备 ID
        gpus = range(torch.cuda.device_count())
        # 使用数据并行处理在多个 GPU 上计算网络输出
        output = dp.data_parallel(Net(), i, gpus)
        # 断言输出与函数 fn 在相同输入上的输出相等
        self.assertEqual(output, fn(i))
        # 断言输出的类型为 torch.Tensor
        self.assertIsInstance(output[0], torch.Tensor)
        # 断言输出的第二个元素为元组
        self.assertIsInstance(output[1], tuple)
        # 断言元组的第一个和第二个元素为 torch.Tensor
        self.assertIsInstance(output[1][0], torch.Tensor)
        self.assertIsInstance(output[1][1], torch.Tensor)
        # 断言元组的第三个元素为列表
        self.assertIsInstance(output[1][2], list)
        # 断言列表的第一个元素为 torch.Tensor
        self.assertIsInstance(output[1][2][0], torch.Tensor)
        # 断言输出的第三个元素为 torch.Tensor
        self.assertIsInstance(output[2], torch.Tensor)
        # 断言输出的第四个元素为 OrderedDict
        self.assertIsInstance(output[3], dict)
        # 断言 OrderedDict 中包含两个键
        self.assertEqual(len(output[3]), 2)
        # 断言 OrderedDict 中包含键 "a" 和 "b"
        self.assertIn("a", output[3])
        self.assertIn("b", output[3])
        # 断言 OrderedDict 的值为 torch.Tensor 或列表
        self.assertIsInstance(output[3]["a"], torch.Tensor)
        self.assertIsInstance(output[3]["b"], list)
        # 断言列表中的第一个元素为 torch.Tensor
        self.assertIsInstance(output[3]["b"][0], torch.Tensor)

    # 跳过但在 Sandcastle 中通过如果不支持多 GPU，否则测试多层嵌套输入的数据并行处理
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_input(self):
        # 定义一个函数 fn，接受一个输入并返回嵌套输入的结果
        def fn(input):
            return input[1][0]
        
        # 定义一个继承自 nn.Module 的网络类 Net，重写 forward 方法，调用 fn 函数
        class Net(nn.Module):
            def forward(self, *input):
                return fn(input)
        
        # 在 "cuda:1" 设备上生成一个形状为 (20, 3) 的浮点数张量 i
        i = torch.randn(20, 3, dtype=torch.float, device="cuda:1")
        # 创建一个嵌套的输入结构
        input = (i.cos(), (i.sin(), i), i.sin())
        # 获取所有可用的 GPU 设备 ID
        gpus = range(torch.cuda.device_count())
        # 使用数据并行处理在多个 GPU 上计算网络输出
        output = dp.data_parallel(Net(), input, gpus)
        # 断言输出与函数 fn 在相同输入上的输出相等
        self.assertEqual(output, fn(input))
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_module_zero_inputs(self):
        # 在沙堡中跳过测试，如果不支持多GPU，则测试不执行
        class TestModule(nn.Module):
            def forward(self):
                # 创建一个在cuda:0设备上的2x3单位矩阵张量
                t = torch.eye(2, 3, device="cuda:0")
                return t + (1 - t)

        def test_helper(output, expected):
            # 断言输出张量的设备为cuda:0
            self.assertEqual(output.get_device(), 0)
            # 断言输出张量与预期张量相等
            self.assertEqual(output, expected)

        expected = torch.ones(2, 3, device="cuda:0")
        model = TestModule()

        # 测试DataParallel对单个GPU的情况
        test_helper(nn.DataParallel(model, [0])(), expected)
        # 测试DataParallel对多个GPU的情况
        test_helper(nn.DataParallel(model, [0, 1])(), expected)
        # 测试dp.data_parallel函数对单个GPU的情况
        test_helper(dp.data_parallel(model, None, [0]), expected)
        # 测试dp.data_parallel函数对多个GPU的情况
        test_helper(dp.data_parallel(model, (), [0, 1]), expected)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_device_args(self):
        cuda0 = torch.device("cuda:0")
        cuda1 = torch.device("cuda:1")

        # 测试output_device参数
        l = nn.Linear(10, 5).to(cuda0, torch.float)
        i = torch.randn(20, 10, dtype=torch.float, device=cuda0, requires_grad=True)
        out = dp.data_parallel(l, i, device_ids=(0, 1), output_device=cuda0)
        self.assertEqual(out, l(i))

        # 测试device_ids参数
        l = nn.Linear(10, 5).to(cuda0, torch.float)
        i = torch.randn(20, 10, dtype=torch.float, device=cuda0, requires_grad=True)
        out = dp.data_parallel(l, i, device_ids=(cuda0, cuda1), output_device=cuda0)
        self.assertEqual(out, l(i))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_function_deletion(self):
        # 这个测试案例源自于issue #16532
        def gradient_penalty(net, x):
            output = net(x)
            # 计算梯度惩罚项
            loss = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=x.new_ones(output.size()),
                create_graph=True,
                retain_graph=True,
            )[0].mean()
            return loss

        net = nn.Linear(4, 1).cuda()
        dpn = nn.DataParallel(net, [0, 1])
        x = torch.ones(2, 4, requires_grad=True).cuda()

        dpn.zero_grad()
        loss = gradient_penalty(dpn, x)
        loss.backward()
        grads = [p.grad for p in net.parameters()]
        # 断言梯度张量的数量为2
        self.assertEqual(2, len(grads))
        # 断言第一个参数的梯度张量与预期值相等
        self.assertEqual(
            torch.tensor([[0.25, 0.25, 0.25, 0.25]], device="cuda:0"), grads[0]
        )
        # 断言第二个参数的梯度张量与预期值相等
        self.assertEqual(torch.tensor([0.0], device="cuda:0"), grads[1])
    # 定义一个测试方法，用于测试 scatter 方法在给定张量上的操作
    def _test_scatter(self, tensor):
        # 将输入张量去除梯度并标记为需要梯度
        x = tensor.detach().requires_grad_()
        # 在多设备上分散张量，期望结果为两个部分
        result = dp.scatter(x, (0, 1))
        # 断言结果列表长度为 2
        self.assertEqual(len(result), 2)
        # 断言第一个结果部分与输入张量的前两个元素相等
        self.assertEqual(result[0], x[:2])
        # 断言第一个结果部分位于第 0 设备上
        self.assertEqual(result[0].get_device(), 0)
        # 断言第二个结果部分与输入张量的后两个元素相等
        self.assertEqual(result[1], x[2:])
        # 断言第二个结果部分位于第 1 设备上
        self.assertEqual(result[1].get_device(), 1)
        # 创建一个梯度张量，用于反向传播
        grad = result[0].detach().clone().fill_(2)
        # 对第一个结果部分执行反向传播
        result[0].backward(grad)
        # 断言输入张量前两个元素的梯度与预期的梯度相等
        self.assertEqual(x.grad[:2], grad)
        # 断言输入张量后两个元素的梯度为零
        self.assertEqual(x.grad[2:], grad.clone().zero_())
        # 使用 _assertGradAndGradgradChecks 方法验证梯度和二阶梯度的检查
        _assertGradAndGradgradChecks(self, lambda y: dp.scatter(y, (0, 1)), (x,))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    # 标记为测试方法，在单 GPU 上测试 scatter 方法
    def test_scatter_cpu(self):
        self._test_scatter(torch.randn((4, 4), dtype=torch.double))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    # 标记为测试方法，在 GPU 上测试 scatter 方法
    def test_scatter_gpu(self):
        self._test_scatter(torch.randn((4, 4), dtype=torch.double).cuda())

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @skip_but_pass_in_sandcastle_if(NO_NCCL, "NCCL needed")
    # 复杂数据并行测试方法，期望复数参数通过 view_as_real 广播到实数空间
    def test_data_parallel_complex(self):
        # 定义一个复数模块
        class Cplx(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个复数参数，存储在 CUDA 设备上
                self.cplx = torch.nn.Parameter(
                    torch.zeros(1, 10, dtype=torch.cfloat).cuda()
                )

            def forward(self, x):
                # 返回输入加上复数参数的结果
                return x + self.cplx

        # 使用 DataParallel 封装复数模块，使其在多 GPU 上并行运行
        cplx = torch.nn.DataParallel(Cplx().cuda())
        # 创建一个随机输入张量，存储在 CUDA 设备上
        input = torch.rand(1, 10, dtype=torch.cfloat).cuda()
        # 对输入张量应用复数模块
        result = cplx(input)
        # 断言结果张量的大小为 [1, 10, 2]，表示复数转换为实部和虚部的形式
        self.assertEqual(result.size(), torch.Size([1, 10, 2]))
        # 断言结果张量等于输入张量的 view_as_real 结果
        self.assertEqual(result, torch.view_as_real(input))
    # 定义测试函数 _test_gather，用于测试数据并搜集结果到指定设备
    def _test_gather(self, output_device):
        # 创建两个随机张量作为输入，分别位于不同的 CUDA 设备上
        inputs = (
            torch.randn(2, 4, device="cuda:0", requires_grad=True, dtype=torch.double),
            torch.randn(2, 4, device="cuda:1", requires_grad=True, dtype=torch.double),
        )
        # 调用 dp.gather 函数对输入进行搜集操作，结果存储在 result 中
        result = dp.gather(inputs, output_device)
        # 断言结果张量的大小为 [4, 4]
        self.assertEqual(result.size(), torch.Size([4, 4]))
        # 断言结果张量的前两行等于第一个输入张量
        self.assertEqual(result[:2], inputs[0])
        # 断言结果张量的后两行等于第二个输入张量
        self.assertEqual(result[2:], inputs[1])
        # 如果输出设备不为 -1，则断言结果张量的设备编号与 output_device 相同
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            # 否则断言结果张量不在 CUDA 设备上
            self.assertFalse(result.is_cuda)
        # 创建一个随机梯度张量
        grad = torch.randn((4, 4), dtype=torch.double)
        # 如果输出设备不为 -1，则将梯度张量移动到对应的 CUDA 设备上
        if output_device != -1:
            grad = grad.cuda(output_device)
        # 对结果张量进行反向传播，使用 grad 作为梯度
        result.backward(grad)
        # 断言第一个输入张量的梯度与 grad 的前两行相等
        self.assertEqual(inputs[0].grad, grad[:2])
        # 断言第二个输入张量的梯度与 grad 的后两行相等
        self.assertEqual(inputs[1].grad, grad[2:])
        # 调用 _assertGradAndGradgradChecks 函数验证梯度和二阶梯度检查
        _assertGradAndGradgradChecks(
            self, lambda x, y: dp.gather((x, y), output_device), inputs
        )

        # 测试标量输入的情况，这种情况下应当堆叠成一个向量
        inputs = (
            torch.randn((), device="cuda:0", requires_grad=True, dtype=torch.double),
            torch.randn((), device="cuda:1", requires_grad=True, dtype=torch.double),
        )
        # 调用 dp.gather 函数对标量输入进行搜集操作，结果存储在 result 中
        result = dp.gather(inputs, output_device)
        # 断言结果张量的大小为 [2]
        self.assertEqual(result.size(), torch.Size([2]))
        # 断言结果张量的第一个元素等于第一个输入张量
        self.assertEqual(result[0], inputs[0])
        # 断言结果张量的第二个元素等于第二个输入张量
        self.assertEqual(result[1], inputs[1])
        # 如果输出设备不为 -1，则断言结果张量的设备编号与 output_device 相同
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            # 否则断言结果张量不在 CUDA 设备上
            self.assertFalse(result.is_cuda)
        # 创建一个随机梯度向量
        grad = torch.randn(2, dtype=torch.double)
        # 如果输出设备不为 -1，则将梯度向量移动到对应的 CUDA 设备上
        if output_device != -1:
            grad = grad.cuda(output_device)
        # 对结果张量进行反向传播，使用 grad 作为梯度
        result.backward(grad)
        # 断言第一个输入张量的梯度与 grad 的第一个元素相等
        self.assertEqual(inputs[0].grad, grad[0])
        # 断言第二个输入张量的梯度与 grad 的第二个元素相等
        self.assertEqual(inputs[1].grad, grad[1])
        # 调用 _assertGradAndGradgradChecks 函数验证梯度和二阶梯度检查
        _assertGradAndGradgradChecks(
            self, lambda x, y: dp.gather((x, y), output_device), inputs
        )

    # 标记为跳过测试，除非 TEST_MULTIGPU 为真且支持多 GPU
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_cpu(self):
        # 调用 _test_gather 函数进行 CPU 环境下的搜集测试
        self._test_gather(-1)

    # 标记为跳过测试，除非 TEST_MULTIGPU 为真且支持多 GPU
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_gpu(self):
        # 调用 _test_gather 函数进行 GPU 环境下的搜集测试（使用第一个 GPU 设备）
        self._test_gather(0)

    # 标记为跳过测试，除非 TEST_MULTIGPU 为真且支持多 GPU
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_different_len_dicts(self):
        # 定义包含不同长度字典的输入
        inputs = (
            {"a": torch.randn(1, 2, requires_grad=True, device="cuda:0")},
            {
                "b": torch.randn(1, 2, requires_grad=True, device="cuda:1"),
                "a": torch.randn(1, 2, requires_grad=True, device="cuda:1"),
            },
        )
        # 使用 self.assertRaises 断言抛出 ValueError 异常
        with self.assertRaises(ValueError):
            # 调用 dp.gather 函数搜集输入，指定目标设备为 0
            _ = dp.gather(inputs, target_device=0)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    # 如果 TEST_MULTIGPU 为 False，则跳过测试，否则继续执行
    def test_replicate(self):
        # 创建一个包含10个输入神经元和5个输出神经元的线性模型，转换为单精度浮点型，并放在 CUDA 设备上
        module = nn.Linear(10, 5).float().cuda()
        # 生成一个在 CUDA 设备上的随机张量，形状为(2, 10)，数据类型为单精度浮点型
        input = torch.randn(2, 10, dtype=torch.float, device="cuda")
        # 计算期望输出
        expected_output = module(input)
        # 对于每个设备列表 [(0, 1), [0, 1]]，分别进行模型复制操作
        for devices in [(0, 1), [0, 1]]:
            # 使用 dp.replicate 方法在指定设备上复制模型
            replicas = dp.replicate(module, devices)
            # 对每个复制品进行迭代
            for i, replica in enumerate(replicas):
                # 检查每个参数的设备是否与索引 i 匹配
                for p in replica.parameters():
                    self.assertEqual(p.get_device(), i)
                # 将输入数据移到索引为 i 的 CUDA 设备上，并检查 replica 的输出是否与 expected_output 相同
                replica_input = input.cuda(i)
                self.assertEqual(replica(replica_input), expected_output)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    # 如果 TEST_MULTIGPU 为 False，则跳过测试，否则继续执行
    def test_replicate_buffers(self):
        # 创建一个空的神经网络模型
        net = nn.Module()
        # 添加一个二维批量归一化层，设定输入通道数为 10，并将模型放在 CUDA 设备上
        net.bn = nn.BatchNorm2d(10)
        net.cuda()
        # 对于每个设备列表 [(0, 1), [0, 1]]，分别进行模型复制操作
        for devices in [(0, 1), [0, 1]]:
            # 使用 dp.replicate 方法在指定设备上复制模型
            replicas = dp.replicate(net, devices)
            # 对每个复制品进行迭代
            for i, replica in enumerate(replicas):
                # 检查每个批量归一化层的 running_mean 参数是否在索引 i 的设备上
                self.assertEqual(
                    replica.bn.running_mean.get_device(),
                    i,
                    msg="buffer on wrong device",
                )
                # 检查每个批量归一化层的 running_var 参数是否在索引 i 的设备上
                self.assertEqual(
                    replica.bn.running_var.get_device(), i, msg="buffer on wrong device"
                )
                # 检查每个批量归一化层的 num_batches_tracked 参数是否在索引 i 的设备上
                self.assertEqual(
                    replica.bn.num_batches_tracked.get_device(),
                    i,
                    msg="buffer on wrong device",
                )

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    # 如果 TEST_MULTIGPU 为 False，则跳过测试，否则继续执行
    def test_zero_grad(self):
        # zero_grad 应当警告在前向传播中使用梯度

        # 定义一个网络模型类 Net，继承自 torch.nn.Module
        class Net(torch.nn.Module):
            def __init__(self, testcase):
                super().__init__()
                self._testcase = testcase

            def forward(self, x):
                # 使用 self._testcase.assertWarnsRegex 断言捕获 UserWarning 类型的警告信息
                with self._testcase.assertWarnsRegex(
                    UserWarning,
                    r"Calling \.zero_grad\(\) from a module created with nn\.DataParallel\(\) has no effect.",
                ):
                    self.zero_grad()
                return x

        # 创建 Net 类的实例，放在 CUDA 设备上
        module = Net(self).cuda()
        # 使用 dp.DataParallel 包装模型
        dpm = dp.DataParallel(module)
        # 将随机生成的形状为 (4, 3, 6, 5) 的张量输入到模型中
        dpm(torch.rand(4, 3, 6, 5))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    # 如果 TEST_MULTIGPU 为 False，则跳过测试，否则继续执行
    def test_autocast(self):
        # 定义一个继承自 torch.nn.Linear 的模型类 Model
        class Model(torch.nn.Linear):
            def __init__(self):
                super().__init__(8, 8)

            @torch.cuda.amp.autocast()
            # 使用自动混合精度装饰器
            def forward(self, input):
                return super().forward(input)

        # 创建 Model 类的 DataParallel 包装实例，并放在 CUDA 设备上，数据类型为单精度浮点型
        model = dp.DataParallel(Model().cuda().to(dtype=torch.float32))
        # 创建形状为 (8, 8) 的随机张量，放在 CUDA 设备上，数据类型为单精度浮点型
        input = torch.randn((8, 8), dtype=torch.float32, device="cuda")
        # 断言模型的输出数据类型为 torch.float16
        self.assertTrue(model(input).dtype is torch.float16)
    def test_save_replica_module(self):
        # DataParallel replicas can be saved (gh-37182)
        # 创建一个在 GPU 上的线性模型
        module = torch.nn.Linear(8, 8).cuda()
        # 复制模型到指定设备上，并保持梯度连接
        dpm = torch.nn.parallel.replicate(module, devices=[0, 1], detach=False)
        # 创建一个字节流对象
        data = io.BytesIO()
        # 将复制的模型保存到字节流中
        torch.save(dpm, data)
        # 再次复制模型到指定设备上，但这次断开梯度连接
        dpm = torch.nn.parallel.replicate(module, devices=[0, 1], detach=True)
        # 将断开连接后的模型保存到同一字节流中
        torch.save(dpm, data)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parameter_list_dict_replica(self):
        class MyMod(torch.nn.Module):
            def __init__(self, data, check_fn):
                super().__init__()
                self.data = data
                self.check_fn = check_fn

            def forward(self, inp):
                # 调用检查函数
                self.check_fn(self)
                return inp

        # 创建两个参数
        p1 = torch.nn.Parameter(torch.rand(10))
        p2 = torch.nn.Parameter(torch.rand(10))
        key0 = 0
        key1 = 1

        def check_fn(self_):
            # 断言参数与模型数据对应
            self.assertEqual(p1, self_.data[key0])
            self.assertEqual(p2, self_.data[key1])
            # 确保参数需要梯度
            self.assertTrue(self_.data[key0].requires_grad)
            self.assertTrue(self_.data[key1].requires_grad)
            # 确保参数有梯度函数
            self.assertIsNotNone(self_.data[key0].grad_fn)
            self.assertIsNotNone(self_.data[key1].grad_fn)

        # 使用 ParameterList 创建模型
        module = MyMod(torch.nn.ParameterList([p1, p2]), check_fn).cuda()
        # 使用 DataParallel 封装模型
        model = dp.DataParallel(module)
        # 创建输入张量，并将其移动到 GPU
        input = torch.randn((8, 8), device="cuda")

        # 运行模型，并触发检查函数
        model(input)

        key0 = "0"
        key1 = "1"
        # 使用 ParameterDict 创建模型
        module = MyMod(torch.nn.ParameterDict({"0": p1, "1": p2}), check_fn).cuda()
        # 使用 DataParallel 封装模型
        model = dp.DataParallel(module)
        # 创建输入张量，并将其移动到 GPU
        input = torch.randn((8, 8), device="cuda")

        # 运行模型，并触发检查函数
        model(input)
class TestDataParallelDeviceType(TestCase):
    # 仅在CUDA环境下运行此测试
    @onlyCUDA
    # 跳过元数据的测试
    @skipMeta
    # 测试不同数据类型的数据并行模块
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module(self, device, dtype):
        # 创建一个线性层，将其移到指定设备和数据类型
        l = nn.Linear(10, 5).to(device, dtype)
        # 生成一个指定设备和数据类型的随机张量
        i = torch.randn(20, 10, device=device, dtype=dtype)
        # 计算预期输出
        expected_out = l(i)
        # 将线性层包装成数据并行模块
        net = nn.DataParallel(l)
        # 使用数据并行模块进行前向传播
        out = net(i)
        # 断言输出张量所在的设备为0号设备
        self.assertEqual(out.get_device(), 0)
        # 断言模型输出与预期输出一致，使用指定的误差容限
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only(self, device, dtype):
        # 定义一个简单的神经网络模块
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 将全局变量l赋给当前实例的self.l
                self.l = l

            def forward(self, input):
                return self.l(input)

        # 创建一个线性层，将其移到指定设备和数据类型
        l = nn.Linear(10, 5).to(device, dtype)
        # 生成一个指定设备和数据类型的随机张量
        i = torch.randn(20, 10, device=device, dtype=dtype)
        # 计算预期输出
        expected_out = l(i)
        # 将神经网络模块包装成数据并行模块
        n = nn.DataParallel(Net())
        # 使用数据并行模块进行前向传播，传递额外的输入参数i
        out = n(input=i)
        # 断言输出张量所在的设备为0号设备
        self.assertEqual(out.get_device(), 0)
        # 断言模型输出与预期输出一致，使用指定的误差容限
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_list(self, device, dtype):
        # 定义一个简单的神经网络模块
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 将全局变量l赋给当前实例的self.l
                self.l = l

            def forward(self, input):
                return self.l(input["data"])

        # 创建一个线性层，将其移到指定设备和数据类型
        l = nn.Linear(10, 5).to(device, dtype)
        # 生成一个指定设备和数据类型的随机张量
        i = torch.randn(20, 10, device=device, dtype=dtype)
        # 计算预期输出
        expected_out = l(i)
        # 将神经网络模块包装成数据并行模块
        n = nn.DataParallel(Net())
        # 使用数据并行模块进行前向传播，传递包含"data"键的输入字典，但"unused"键对应一个空列表
        out = n(input={"data": i, "unused": []})
        # 断言输出张量所在的设备为0号设备
        self.assertEqual(out.get_device(), 0)
        # 断言模型输出与预期输出一致，使用指定的误差容限
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_dict(self, device, dtype):
        # 定义一个简单的神经网络模块
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 将全局变量l赋给当前实例的self.l
                self.l = l

            def forward(self, input):
                return self.l(input["data"])

        # 创建一个线性层，将其移到指定设备和数据类型
        l = nn.Linear(10, 5).to(device, dtype)
        # 生成一个指定设备和数据类型的随机张量
        i = torch.randn(20, 10, device=device, dtype=dtype)
        # 计算预期输出
        expected_out = l(i)
        # 将神经网络模块包装成数据并行模块
        n = nn.DataParallel(Net())
        # 使用数据并行模块进行前向传播，传递包含"data"键的输入字典，但"unused"键对应一个空字典
        out = n(input={"data": i, "unused": {}})
        # 断言输出张量所在的设备为0号设备
        self.assertEqual(out.get_device(), 0)
        # 断言模型输出与预期输出一致，使用指定的误差容限
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)
    # 定义一个测试方法，用于测试数据并行处理模块，只接受空元组作为额外参数
    def test_data_parallel_module_kwargs_only_empty_tuple(self, device, dtype):
        # 定义一个神经网络模型类 Net
        class Net(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义模型的一个成员变量 self.l，是一个线性层对象
                self.l = l

            # 前向传播方法
            def forward(self, input):
                # 使用模型中的 self.l 对输入数据中键为 "data" 的值进行前向传播
                return self.l(input["data"])

        # 创建一个大小为 (10, 5) 的线性层对象 l，并将其移到指定设备上，指定数据类型
        l = nn.Linear(10, 5).to(device, dtype)
        # 创建一个形状为 (20, 10) 的随机张量 i，放在指定设备上，指定数据类型
        i = torch.randn(20, 10, device=device, dtype=dtype)
        # 计算预期输出，即 l 对输入 i 的计算结果
        expected_out = l(i)
        # 使用 nn.DataParallel 对 Net 类实例进行并行处理
        n = nn.DataParallel(Net())
        # 调用并行处理模块 n 的 forward 方法，传入包含 "data" 和空元组 "unused" 的字典作为输入
        out = n(input={"data": i, "unused": ()})
        # 断言输出张量 out 的设备编号为 0
        self.assertEqual(out.get_device(), 0)
        # 断言 out 的值等于预期的输出 expected_out，使用指定的数值容差和相对容差
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)
# 实例化设备类型测试，使用 TestDataParallelDeviceType 进行测试，全局环境中定义的测试
instantiate_device_type_tests(TestDataParallelDeviceType, globals())

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 启用 TestCase 的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行测试套件
    run_tests()
```