# `.\pytorch\test\inductor\test_cuda_repro.py`

```py
# Owner(s): ["module: inductor"]
# 引入必要的模块和库
import gc  # Python 的垃圾回收模块
import math  # Python 的数学函数模块
import sys  # Python 的系统相关模块
import unittest  # Python 的单元测试框架

import torch  # PyTorch 深度学习框架
import torch._dynamo.config as dynamo_config  # 导入 dynamo_config 模块
import torch.backends.cuda  # PyTorch CUDA 后端
import torch.nn.functional as F  # PyTorch 中的函数模块
from torch import nn  # PyTorch 中的神经网络模块
from torch._dynamo.debug_utils import same_two_models  # 导入调试工具中的函数
from torch._dynamo.testing import rand_strided  # 导入测试工具中的函数
from torch._dynamo.utils import same  # 导入工具中的函数
from torch._inductor import config  # 导入 inductor 模块的配置
from torch._inductor.compile_fx import compile_fx_inner  # 导入编译 FX 的函数
from torch._inductor.runtime.hints import DeviceProperties  # 导入设备属性提示
from torch._inductor.utils import run_and_get_code  # 导入运行和获取代码的函数
from torch.fx.experimental.proxy_tensor import make_fx  # 导入代理张量的函数
from torch.testing import FileCheck  # 导入文件检查工具
from torch.testing._internal.common_cuda import (  # 导入 CUDA 相关的通用工具
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM80OrLater,
)
from torch.testing._internal.common_utils import (  # 导入通用工具
    DeterministicGuard,
    freeze_rng_state,
    IS_FBCODE,
    skipIfRocm,
    TEST_WITH_ASAN,
)

from torch.testing._internal.inductor_utils import skipCUDAIf  # 导入 CUDA 相关的测试工具

try:
    try:
        import triton  # 导入 Triton 库
        from triton import language as tl  # 导入 Triton 的语言模块
    except ImportError:
        raise unittest.SkipTest("requires triton")  # 若导入失败，跳过测试并抛出异常 # noqa: B904

    try:
        from . import test_torchinductor  # 尝试从当前目录导入测试模块
    except ImportError:
        import test_torchinductor  # 若导入失败，则从全局路径导入测试模块
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)  # 若在主程序中运行，则退出
    raise


TestCase = test_torchinductor.TestCase  # 设置测试用例类
ToTuple = test_torchinductor.ToTuple  # 设置转换元组的函数
check_model_cuda = test_torchinductor.check_model_cuda  # 设置检查 CUDA 模型的函数
aten = torch.ops.aten  # 设置 PyTorch aten 操作的命名空间


class CudaReproTests(TestCase):  # CUDA 重现测试类，继承自 TestCase

    common = check_model_cuda  # 设置公共属性为检查 CUDA 模型的函数

    def test_index_put_issue(self):  # 测试索引放置问题的方法

        def forward(  # 正向传播函数，接受多个参数
            self,
            arg76_1,
            expand_default,
            full_like_default,
            _to_copy_default_67,
            zeros,
        ):
            sum_sym_int_19 = torch.ops.aten.sum(_to_copy_default_67, [0], True)  # 计算 _to_copy_default_67 的和
            view_default_57 = torch.ops.aten.view.default(sum_sym_int_19, [512, 768])  # 对 sum_sym_int_19 进行视图变换
            where_self = torch.ops.aten.where.self(  # 获取条件为 self 的索引
                expand_default, view_default_57, full_like_default
            )
            clone_default_12 = torch.ops.aten.clone.default(zeros)  # 克隆 zeros 张量
            index_put__default = torch.ops.aten.index_put_.default(  # 对 clone_default_12 执行索引放置操作
                clone_default_12, [arg76_1], where_self, True
            )
            return (index_put__default,)  # 返回结果元组

        inps = [
            (torch.Size([512]), torch.int64),
            (torch.Size([512, 768]), torch.bool),
            (torch.Size([512, 768]), torch.float16),
            (torch.Size([4, 512, 768]), torch.float16),
            (torch.Size([512, 768]), torch.float16),
        ]
        inps = [torch.zeros(())] + [
            torch.ones(shape, dtype=dtype, device="cuda") for (shape, dtype) in inps
        ]  # 准备输入数据，包括在 CUDA 设备上的各种张量
        mod = make_fx(forward)(*inps)  # 使用 forward 函数创建 FX 模块
        compiled = compile_fx_inner(mod, inps)  # 编译 FX 模块
        compiled(inps)  # 运行编译后的模块

    @skipIfRocm  # 如果是在 ROCm 平台下，则跳过该测试
    # 定义一个测试函数，用于验证输入通道在最后的情况下的处理
    def test_input_channels_last(self):
        # 创建一个包含序列模块，其中包括一个3通道到3通道的卷积层和一个ToTuple实例，并将其部署到GPU上
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),  # 创建一个3通道到3通道的卷积层
            ToTuple(),  # 将输出转换为元组形式
        ).cuda()
        # 创建一个形状为[2, 3, 16, 16]的随机张量输入inp，并在GPU上使用通道优先的内存格式
        inp = torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last).cuda()

        # 调用self.common方法，对模型m和输入inp进行通用测试，关闭低精度检查
        self.common(
            m,
            (inp,),
            check_lowp=False,
        )

        # 定义一个使用torch._dynamo.optimize()装饰器优化的函数foo，调用模型m对输入inp进行处理，并断言输出是否为通道优先的内存格式
        @torch._dynamo.optimize()
        def foo(m, inp):
            return m(inp)

        self.assertTrue(foo(m, inp)[0].is_contiguous(memory_format=torch.channels_last))

    # https://github.com/pytorch/torchdynamo/issues/1681#issuecomment-1283433527
    # 定义一个测试函数，用于验证未指定输入情况下的互操作性
    def test_unspec_inputs_interop(self):
        # 定义一个Repro类，继承自torch.nn.Module，重写forward方法，对输入x和y进行一系列操作，并返回处理后的结果
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                # 使用torch.ops.aten.unsqueeze.default对张量x进行增加维度的操作
                unsqueeze = torch.ops.aten.unsqueeze.default(x, 4)
                # 使用torch.ops.aten.permute.default对增加维度后的张量unsqueeze进行维度置换操作
                permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3])
                # 使用torch.ops.aten.add.Tensor对张量y加1
                add = torch.ops.aten.add.Tensor(y, 1)
                return [permute, add]

        # 定义inps列表，包含两个元素，一个是形状为(12, 3, 512, 64)的随机张量，另一个是形状为空的torch.int64类型张量
        inps = [
            rand_strided((12, 3, 512, 64), (64, 196608, 768, 1), torch.float32, "cuda"),
            rand_strided((), (), torch.int64, "cpu"),
        ]
        # 创建Repro类的实例mod，并使用make_fx函数将其转换为FX模块，并在cuda设备上部署
        mod = make_fx(Repro().to(device="cuda"))(*inps)
        # 编译FX模块mod的内部表示
        compiled = compile_fx_inner(mod, inps)
        # 运行编译后的模块并传入inps作为输入
        compiled(inps)

    # 使用unittest.skipIf装饰器，根据IS_FBCODE的值判断是否跳过测试
    @unittest.skipIf(
        IS_FBCODE, "RuntimeError: Triton Error [CUDA]: invalid device context"
    )
    # 定义一个测试函数，用于验证反向传播时的上下文环境
    def test_backward_context(self):
        # 定义一个简单的函数fn，对输入x进行乘以3的操作并返回结果
        def fn(x):
            return x * 3

        # 创建一个形状为(4,)的随机张量x，并在cuda设备上启用梯度计算
        x = torch.randn(4, device="cuda", requires_grad=True)
        # 创建一个与x形状相同的随机张量gO
        gO = torch.rand_like(x)
        # 使用torch.compile编译函数fn
        opt_fn = torch.compile(fn)
        # 对输入x调用编译后的函数opt_fn，并获取输出out
        out = opt_fn(x)
        # 对输出out进行反向传播，计算梯度
        out.backward(gO)

    # 使用config.patch装饰器设置配置参数fallback_random=True
    @config.patch(fallback_random=True)
    # 定义一个测试函数，用于验证dtype_factory_issue
    def test_dtype_factory_issue(self):
        # 定义一个forward函数，生成一个形状为[12, 64, 1, 64]的随机张量randn，并在cuda设备上指定dtype、device和pin_memory参数
        def forward():
            randn = torch.ops.aten.randn.default(
                [12, 64, 1, 64],
                dtype=torch.float32,
                device=torch.device(type="cuda", index=0),
                pin_memory=False,
            )
            # 使用torch.ops.aten.unsqueeze.default对randn进行维度增加操作
            unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(randn, -1)
            return (unsqueeze_default_2,)

        # 使用make_fx函数将forward函数转换为FX模块mod
        mod = make_fx(forward)()
        # 编译FX模块mod的内部表示
        compiled = compile_fx_inner(mod, ())
        # 断言编译后模块的输出第一个元素的设备类型为cuda
        assert compiled([])[0].device.type == "cuda"

    # 使用config.patch装饰器设置配置参数{"triton.cudagraphs": True}
    # 使用dynamo_config.patch装饰器设置配置参数automatic_dynamic_shapes=True
    # 定义一个测试方法，用于验证不涉及设备索引复制 cudagraphs 的行为
    def test_no_device_idx_repro_cudagraphs(self):
        # 定义一个简单的神经网络模型 Repro
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                # 创建一个指定设备（cuda:0）的全1张量，默认布局和浮点数数据类型
                full = torch.ops.aten.full.default(
                    [8, 512],
                    1,
                    dtype=torch.float32,
                    layout=torch.strided,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                # 创建一个指定设备（cuda:0）的全0张量，默认布局和64位整型数据类型
                full_1 = torch.ops.aten.full.default(
                    [8, 512],
                    0,
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                return (full_1, full)

        # 调用公共方法，验证 Repro 模型行为是否符合预期
        self.common(Repro(), ())

    # 使用配置修补装饰器，开启 Triton 的 cudagraphs 功能
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    # 定义测试扩展输入的 cudagraphs 行为
    def test_expanded_inputs_cudagraphs(self):
        # 定义一个优化函数 fn，标记为 "inductor"，接受两个参数 x 和 y
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x + y
        
        # 创建两个随机张量输入，使用 cuda 设备
        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        # 断言优化后的 fn 函数对输入的行为与两个张量相加的结果相同
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    # 使用配置修补装饰器，配置 cudagraphs 的动态到静态行为
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=False,
    )
    # 测试动态形状到静态形状的 cudagraphs 行为
    def test_dynamic_to_static_cudagraphs(self):
        # 遍历两种 cudagraph_trees 的配置
        for b in [False, True]:
            # 根据配置修补 cudagraph_trees
            with config.patch({"triton.cudagraph_trees": b}):
                # 定义一个优化函数 fn，标记为 "inductor"，接受两个参数 x 和 y
                @torch._dynamo.optimize("inductor")
                def fn(x, y):
                    # 计算 x 和 y 的和，返回结果和第一个张量的大小
                    r = x + y
                    return r, r.size(0)
                
                # 创建两个不同大小的随机张量输入，使用 cuda 设备
                inputs = (
                    torch.randn((5, 5), device="cuda"),
                    torch.randn((5, 5), device="cuda"),
                )
                # 断言优化后的 fn 函数对输入的行为与两个张量相加的结果和大小相同
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 5)))

                inputs = (
                    torch.randn((6, 6), device="cuda"),
                    torch.randn((6, 6), device="cuda"),
                )
                # 断言优化后的 fn 函数对输入的行为与两个张量相加的结果和大小相同
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 6)))

    # TODO: Abstract this out, test more extensively
    # 测试动态形状的行为
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_shapes(self):
        # 重置 dynamo 设置为 "inductor"
        torch._dynamo.reset()

        # 定义函数 f，计算输入张量的余弦，并将结果视图转换为输入张量的形状再计算正弦
        def f(x):
            return x.cos().view(x.shape).sin()

        # 创建一个带有后端编译计数器的 "inductor" 编译优化函数
        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        # 优化函数 f，使用编译计数器
        f2 = torch._dynamo.optimize(cnts)(f)

        # 对随机张量进行优化函数 f2 的计算
        f2(torch.randn(32))

        # 创建一个输入张量，计算真实输出和编译输出
        inp = torch.randn(16)
        real_out = f(inp)
        compiled_out = f2(inp)

        # 断言编译帧数为1，且真实输出等于编译输出
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(real_out, compiled_out)
        
        # 重置 dynamo 设置
        torch._dynamo.reset()
    # 使用 @config.patch 装饰器设置 triton.cudagraphs 和 size_asserts 配置项
    # 为 test_expanded_inputs_cudagraphs_no_size_asserts 方法添加装饰器，配置动态形状优化和禁用大小断言
    @config.patch({"triton.cudagraphs": True, "size_asserts": False})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs_no_size_asserts(self):
        # 使用 @torch._dynamo.optimize 装饰器对 fn 函数进行动态图优化，命名为 "inductor"
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x + y
    
        # 生成两个随机张量作为输入，通过 rand_strided 生成，使用 CUDA 设备
        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        # 断言 fn 函数对 inputs 的输出与 inputs[0] + inputs[1] 相同
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))
    
    # 使用 @config.patch 装饰器分别设置 triton.cudagraph_trees 和 triton.cudagraphs 配置项
    # 为 test_inplace_updates_cudagraphs 方法添加装饰器，配置动态形状优化
    @config.patch({"triton.cudagraph_trees": False})
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_inplace_updates_cudagraphs(self):
        # 定义一个简单的神经网络模型类 Repro
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含随机初始化权重的可训练参数 weight1
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 20, requires_grad=True)
                )
    
            def forward(self, x):
                # 模型的前向传播，执行矩阵乘法操作
                x = torch.matmul(x, self.weight1)
                return x
    
        # 导入深拷贝函数 deepcopy
        from copy import deepcopy
    
        # 创建 Repro 类的 CUDA 版本实例 model，并生成其深拷贝 model_ref
        model = Repro().cuda()
        model_ref = deepcopy(model)
        # 对 model 应用动态图优化，并命名为 model_opt
        model_opt = torch._dynamo.optimize("inductor")(model)
    
        # 创建一个随机初始化的输入张量 input，使用 CUDA 设备，要求梯度计算
        input = torch.randn(10, 10, device="cuda", requires_grad=True)
    
        # 执行两次循环
        for i in range(2):
            # 计算 model_ref 对 input 的输出
            output_ref = model_ref(input)
            # 计算 model_opt 对 input 的输出
            output_res = model_opt(input)
            # 对 output_ref 和 output_res 的结果求和并执行反向传播
            output_ref.sum().backward()
            output_res.sum().backward()
            # 比较 model_ref 和 model_opt 的每个参数的梯度，并断言它们相等
            for p_ref, p_res in zip(model_ref.parameters(), model_opt.parameters()):
                self.assertEqual(p_ref.grad, p_res.grad)
            # 使用 torch.no_grad() 上下文管理器，逐个参数对 model_ref 执行加法操作
            with torch.no_grad():
                for param in model_ref.parameters():
                    param.add_(1.0)
                # 使用 torch.no_grad() 上下文管理器，逐个参数对 model_opt 执行加法操作
                for param in model_opt.parameters():
                    param.add_(1.0)
    
    # 定义一个函数 test_inductor_output_aliases_intermediate，用于测试函数 foo 在输出别名中间结果时的行为
    # 函数内部定义了 foo 函数和对其应用动态图优化的 foo_opt 函数
    def test_inductor_output_aliases_intermediate(self):
        def foo(x):
            # 计算输入张量 x 的加法操作结果，并返回其转置
            out = x + x
            return out.t()
    
        # 对 foo 函数应用动态图优化，命名为 foo_opt
        foo_opt = torch._dynamo.optimize("inductor")(foo)
    
        # 创建一个随机初始化的输入张量 inpt，使用 CUDA 设备，要求梯度计算
        inpt = torch.randn(10, 10, device="cuda", requires_grad=True)
        # 执行 foo 函数并在其结果上执行加法操作
        out_ref = foo(inpt)
        out_ref.add_(2)
        # 暂时注释下面的代码，因为此处有 bug 需要修复
        # self.assertEqual(out_ref, out)
    def test_accuracy_issue1(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Repro
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个线性层，输入维度为768，输出维度为2，包含偏置
                self.linear = torch.nn.Linear(
                    in_features=768, out_features=2, bias=True
                )

            # 模型前向传播函数
            def forward(self, start_positions: torch.Tensor, x: torch.Tensor):
                # 对输入进行线性变换
                linear = self.linear(x)
                # 在最后一个维度上拆分张量
                split = linear.split(1, dim=-1)
                # 获取拆分后的第一个张量
                getitem = split[0]
                # 在最后一个维度上压缩张量
                squeeze = getitem.squeeze(-1)
                # 对起始位置张量进行限幅操作，将其限制在0到128之间
                clamp = start_positions.clamp(0, 128)
                # 计算交叉熵损失
                cross_entropy = torch.nn.functional.cross_entropy(
                    squeeze, clamp, None, None, 128, None, "mean", 0.0
                )
                return cross_entropy

        # 创建 Repro 类的实例，并将其移动到 CUDA 设备上
        mod = Repro().cuda()
        # 使用 torch._dynamo.optimize 函数对模型进行优化
        opt_mod = torch._dynamo.optimize("inductor")(mod)
        # 将模型设置为评估模式
        mod.eval()
        opt_mod.eval()

        # 定义一组输入参数列表
        args = [
            ((1,), (1,), torch.int64, "cuda", False),
            ((1, 128, 768), (98304, 768, 1), torch.float32, "cuda", True),
        ]
        # 根据参数列表生成张量，并设置是否需要梯度
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        # 禁用 CUDA 混合精度自动转换上下文管理器
        with torch.cuda.amp.autocast(enabled=False):
            # 断言两个模型在给定参数下相同，否则输出错误信息 "Dynamo failed"
            assert same_two_models(mod, opt_mod, args), "Dynamo failed"

    @config.patch(allow_buffer_reuse=False)
    def test_issue103461(self):
        # 定义一个前向传播函数 forward
        def forward(add_1):
            # 使用 torch.ops.aten.var_mean.correction 计算加权方差的修正值
            var_mean = torch.ops.aten.var_mean.correction(
                add_1, [2], correction=0, keepdim=True
            )
            # 获取修正后张量的第二个元素
            getitem_1 = var_mean[1]
            return getitem_1

        # 在 CUDA 设备上生成一个随机张量
        x = torch.randn(1, 8, 768, device="cuda")
        # 使用 torch.compile 函数编译 forward 函数的计算图，并计算其输出
        actual = torch.compile(forward, fullgraph=True)(x)
        # 调用 forward 函数获取正确的输出
        correct = forward(x)
        # 断言编译后的结果与正确的结果相等
        self.assertEqual(actual, correct)

    def test_full_copy(self):
        # 定义一个前向传播函数 forward
        def forward(x):
            # 使用 torch.ops.aten.full.default 创建一个指定形状、dtype 和设备的全零张量
            full_10 = torch.ops.aten.full.default(
                [204, 204, 28],
                0,
                dtype=torch.float64,
                layout=torch.strided,
                device="cuda",
                pin_memory=False,
            )
            # 将输入张量 x 移动到 CPU 设备，并与全零张量相加
            return x + full_10.to("cpu")

        # 生成一个形状为 [204, 204, 28] 的随机张量 o，数据类型为 torch.float64
        o = torch.randn([204, 204, 28], dtype=torch.float64)
        # 使用 torch.compile 函数编译 forward 函数的计算图，并计算其输出
        actual = torch.compile(forward, fullgraph=True)(o)
        # 调用 forward 函数获取正确的输出
        correct = forward(o)
        # 断言编译后的结果与正确的结果相等
        self.assertEqual(actual, correct)
    def test_autotune_inplace_kernel(self):
        """
        This UT tests autotune on an inplace kernel. The autotune should not contaminate
        the input buffers when tuning with multiple configs. For more details, refer to
        https://github.com/openai/triton/issues/781
        https://github.com/pytorch/torchdynamo/issues/1670
        """
        # 导入需要的模块和函数
        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
        from torch._inductor.runtime.hints import HeuristicType, instance_descriptor
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner, grid
        
        # 定义自动调优函数
        def autotune(configs, meta):
            def decorator(fn):
                # 使用缓存自动调优器来包装函数
                return CachingAutotuner(
                    fn,
                    triton_meta=meta,
                    configs=configs,
                    save_cache_hook=False,  # 禁用缓存保存钩子以强制自动调优
                    mutated_arg_names=["in_out_ptr0"],  # 标记会被修改的参数名列表
                    heuristic_type=HeuristicType.POINTWISE,  # 使用点对点操作的启发式方法
                )
            
            return decorator
        
        # 使用autotune装饰器来调优kernel函数
        @autotune(
            configs=[
                triton.Config({"XBLOCK": 1}),  # 配置1
                triton.Config({"XBLOCK": 2}),  # 配置2
            ],
            meta={
                "signature": {0: "*fp32", 1: "*fp32", 2: "i32"},  # 函数签名描述
                "device": DeviceProperties.create(torch.device("cuda")),  # 创建CUDA设备属性
                "configs": [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())],  # 实例描述
                "constants": {},  # 常量为空字典
            },
        )
        @triton.jit
        # 定义kernel函数
        def kernel(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
            pid = tl.program_id(0)  # 获取程序ID
            block_start = pid * XBLOCK  # 计算块的起始位置
            offsets = block_start + tl.arange(0, XBLOCK)  # 计算偏移量
            mask = offsets < xnumel  # 创建掩码以检测偏移是否超出界限
            x = tl.load(in_out_ptr0 + offsets, mask=mask, other=0.0)  # 加载输入输出指针处的数据
            y = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)  # 加载输入指针处的数据
            output = x + y  # 计算输出
            tl.store(in_out_ptr0 + offsets, output, mask=mask)  # 存储结果到输入输出指针处
        
        xnumel = 384  # 设置数据元素数目
        in0 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)  # 生成CUDA上的随机数据
        inout1 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)  # 生成CUDA上的随机数据
        inout2 = inout1.clone()  # 克隆inout1数据
        
        stream0 = get_cuda_stream(0)  # 获取CUDA流
        kernel.run(inout1, in0, xnumel, grid=grid(xnumel), stream=stream0)  # 运行kernel函数，使用inout1和in0
        kernel.run(inout2, in0, xnumel, grid=grid(xnumel), stream=stream0)  # 再次运行kernel函数，使用inout2和in0
        
        assert same(
            inout1, inout2, tol=0.001, equal_nan=True
        ), "failed autotune with inplace kernel"  # 断言，确保inplace kernel的自动调优成功
    # 定义测试函数，用于验证排序步骤的问题
    def test_sort_stride_issue(self):
        # 这是一个来自 detectron2_maskrcnn_r_50_fpn 的简化测试用例
        # 我们的 size_assert 代码曾出现误报
        @torch._dynamo.optimize(nopython=True)
        # 定义前向函数，接受一个名为 pred_objectness_logits_3_ 的 Torch 张量作为参数
        def forward(pred_objectness_logits_3_: torch.Tensor):
            # 对 pred_objectness_logits_3_ 张量按照第一维降序排序
            sort_3 = pred_objectness_logits_3_.sort(descending=True, dim=1)
            # 获取排序后的结果中的第一个元素
            getitem_12 = sort_3[0]
            return getitem_12

        # 设置测试参数列表
        args = [((1, 100), (0, 1), torch.float16, "cuda", False)]
        # 根据参数列表生成张量，其中 rand_strided 是一个生成随机数据的函数
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        # 执行前向函数，传入生成的参数 args，获取结果
        result = forward(*args)
        # 断言函数 same 检查前向函数结果与直接排序结果的一致性
        assert same(result, torch.sort(args[0], descending=True, dim=1)[0])

    # 定义测试函数，验证使用标量进行 Triton 索引的问题
    def test_scalar_triton_index(self):
        # 以下的间接索引方式曾导致 Triton 代码编译时段错误，导致 Triton 崩溃
        # 详见 https://github.com/pytorch/torchdynamo/issues/1515
        def fn(a):
            # 创建一个与 a 设备和数据类型相匹配的全零张量
            zero = torch.zeros((16,), device=a.device, dtype=torch.int64)
            # 返回 a 张量按 zero 中的索引取值后的元组
            return (a[zero],)

        # 在 CUDA 设备上生成一个随机的浮点数张量 a
        a = torch.randn((8,), dtype=torch.float32, device="cuda")

        # 使用 torch._dynamo.optimize 进行函数 fn 的优化
        fn_optimized = torch._dynamo.optimize("inductor")(fn)
        # 断言优化后的函数 fn_optimized(a) 与原函数 fn(a) 结果相同
        assert same(fn(a), fn_optimized(a))

    # 定义测试函数，验证间接索引中的稠密掩码操作
    def test_indirect_indexing_dense_mask(self):
        def fn(x, y):
            # 使用 Torch 运算符计算 x 是否不等于标量 1 的张量
            ne = torch.ops.aten.ne.Scalar(x, 1)
            # 对 ne 张量在第一维度上求和
            sum_1 = torch.ops.aten.sum.dim_IntList(ne, [1])
            # 用 1 减去 sum_1 张量
            sub = torch.ops.aten.sub.Tensor(sum_1, 1)
            # 对 sub 结果在最后一个维度上添加一个维度
            unsqueeze = torch.ops.aten.unsqueeze.default(sub, -1)
            # 使用 x 张量和 unsqueeze 张量进行 gather 操作
            gather = torch.ops.aten.gather.default(x, 1, unsqueeze)
            # 对 gather 结果进行压缩，去除维度为 1 的维度
            squeeze = torch.ops.aten.squeeze.default(gather)
            # 返回 y 与 squeeze 结果相乘后的元组
            out = torch.ops.aten.multiply(y, squeeze)
            return (out,)

        # 在 CUDA 设备上生成两个形状为 (1, 128) 的全零整数张量 a 和 b
        a = torch.zeros((1, 128), dtype=torch.int64, device="cuda")
        b = torch.zeros((1, 128), dtype=torch.int64, device="cuda")

        # 使用 torch._dynamo.optimize 进行函数 fn 的优化
        fn_optimized = torch._dynamo.optimize("inductor")(fn)
        # 断言优化后的函数 fn_optimized(a, b) 与原函数 fn(a, b) 结果相同
        assert same(fn(a, b), fn_optimized(a, b))

    # 定义测试函数，验证简化维度操作
    def test_simplify_dims(self):
        def fn(a):
            # 返回 a 张量加 1 后的元组
            return (a + 1,)

        # 使用 common 方法对 fn 函数进行测试，传入指定的张量参数
        self.common(fn, (torch.randn(2, 3, 10, 5, 6, device="cuda")[:, :, 2::2, :, :],))

    # 应用配置修补程序以进行排列融合
    @config.patch(permute_fusion=True)
    def test_permute_fusion(self):
        # 定义一个继承自torch.nn.Module的模块Repro，用于测试排列和融合操作
        class Repro(torch.nn.Module):
            def forward(self, view, reshape_2):
                # 执行视图的置换操作，将维度0和2交换，维度1不变
                permute = view.permute(0, 2, 1)
                # 将视图设置为None
                view = None
                # 将置换后的视图重塑为形状(-1, 642)
                reshape = torch.reshape(permute, (-1, 642))
                # 使用批矩阵乘法计算permute和reshape_2的乘积
                bmm = torch.bmm(permute, reshape_2)
                # 返回结果作为元组的形式
                return (bmm,)

        # 定义测试用例参数
        args = [
            ((1024, 642, 160), (102720, 160, 1), torch.float32, "cuda", True),
            ((1024, 642, 20), (12840, 20, 1), torch.float32, "cuda", True),
        ]
        # 使用rand_strided函数生成具有指定参数的张量，并设置requires_grad为True
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        # 创建Repro模块的实例
        mod = Repro()
        # 对模块进行优化，使用torch._dynamo.optimize("inductor")进行优化
        opt_mod = torch._dynamo.optimize("inductor")(mod)

        # 计算未优化模块的结果
        ref = mod(*args)
        # 计算优化后模块的结果
        res = opt_mod(*args)
        # 断言优化前后结果是否相同
        self.assertTrue(same(ref, res))

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_add_alpha_autotune(self):
        # 定义一个函数fn，对输入的x和y进行原地加法操作，alpha参数为0.55
        def fn(x, y):
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        # 创建多个形状相同的零张量，并指定在cuda上进行计算
        x1 = torch.zeros(2, 3, 4, 10, device="cuda")
        x2 = torch.zeros(2, 3, 4, 10, device="cuda")
        x3 = torch.zeros(2, 3, 4, 10, device="cuda")
        # 生成一个在channels_last内存格式下的随机张量
        y = torch.randn(2, 3, 4, 10, device="cuda").to(
            memory_format=torch.channels_last
        )
        # 使用make_fx和compile_fx_inner对函数fn进行编译和运行
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        # 在不同的输入上分别调用原始和编译后的函数fn，并断言它们的输出结果相同
        fn(x2, y)
        fn_compiled([x3, y])
        assert same(x2, x3)

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_buffer_autotune(self):
        # 定义一个函数foo，对输入的x和y执行矩阵乘法，然后与z进行加法操作
        def foo(x, y, z):
            a = x @ y
            return a.unsqueeze(0).unsqueeze(0) + z

        # 创建两个形状相同的零张量，并指定在cuda上进行计算
        x = torch.zeros(5, 5, device="cuda")
        y = torch.zeros(5, 5, device="cuda")
        # 生成一个在channels_last内存格式下的零张量
        z = torch.zeros(1, 1, 5, 5, device="cuda").to(memory_format=torch.channels_last)
        # 使用common函数测试foo函数，设置check_lowp参数为False
        self.common(
            foo,
            (x, y, z),
            check_lowp=False,
        )

    def test_memory_history_inductor(self):
        # 定义一个函数called_inside_compile，对输入的x、w和b执行矩阵乘法和加法操作，然后将结果经sigmoid函数处理
        def called_inside_compile(x, w, b):
            a = x @ w + b
            return torch.sigmoid(a)

        # 定义一个使用torch.compile修饰的函数fn，调用called_inside_compile函数两次
        @torch.compile
        def fn(x, w, b):
            x = called_inside_compile(x, w, b)
            return called_inside_compile(x, w, b)

        # 创建形状为(3, 3)的随机张量w和b，并指定在cuda上进行计算
        w = torch.rand(3, 3, device="cuda")
        b = torch.rand(3, device="cuda")
        # 创建形状为(3,)的随机张量x，并指定在cuda上进行计算
        x = torch.rand(3, device="cuda")
        try:
            # 清空CUDA缓存并启用CUDA内存历史记录
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history(True)
            # 调用fn函数，获取其返回值
            r = fn(x, w, b)
        finally:
            # 禁用CUDA内存历史记录
            torch.cuda.memory._record_memory_history(False)
        # 将CUDA内存快照转换为字符串，并断言"called_inside_compile"是否包含在其中
        snapshot = str(torch.cuda.memory._snapshot())
        self.assertTrue("called_inside_compile" in snapshot)
    def test_negative_arange_dynamic_shapes(self):
        # Repro from alibi relative encodings
        # 定义一个函数，用于返回 x 的符号信息
        def sign(x):
            return (x > 0) - (x < 0)

        # 定义一个继承自 torch.nn.Module 的类 Repro
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 设置变量 nheads 为 16
                nheads = 16
                # 计算 start 为 0.5 的对数
                start = math.log2(0.5)
                # 计算 end 为 1 / (2**8) 的对数
                end = math.log2(1 / (2**8))

                # 注册一个名为 "scales" 的 buffer
                self.register_buffer(
                    "scales",
                    # 使用 torch.arange 生成等比数列，转换为 tensor，视图为 (1, nheads, 1, 1)
                    2
                    ** torch.arange(
                        start,
                        end + 1e-6 * sign(end - start),  # 调整末端值，使其与起始值连续
                        (end - start) / (nheads - 1),    # 计算步长
                    ).view(1, nheads, 1, 1),
                )
                # 创建一个嵌入层，输入维度为 1024，输出维度为 256
                self.emb = nn.Embedding(1024, 256)
                # 创建一个 Transformer 解码器层，输入维度为 256，输出维度为 1024，头数为 16
                self.dec_layer = nn.TransformerDecoderLayer(
                    256, 16, 512, batch_first=True, norm_first=True
                )
                # 创建一个线性层，输入维度为 256，输出维度为 1024
                self.head = nn.Linear(256, 1024)

            # 前向传播函数，接受两个参数：enc_out 和 dec_in
            def forward(self, enc_out: torch.Tensor, dec_in: torch.Tensor):
                # 创建一个 mask，用于指示 dec_in 中的 pad 位置
                padmask = dec_in == 0
                # 创建一个与 dec_in 相关的 mask
                dec_mask = padmask.unsqueeze(-1) == padmask.unsqueeze(-2)
                # 将 mask 转换为 float32 类型
                dec_mask = dec_mask.to(dtype=torch.float32)
                # 创建一个下三角形状的 mask，并移动到 GPU 上
                dec_mask = dec_mask.tril(diagonal=0).cuda()

                # 创建一个与 dec_in 大小相同的位置编码 tensor，放置在 GPU 上
                q_pos = torch.arange(dec_in.size(1), dtype=torch.long, device="cuda")
                k_pos = torch.arange(dec_in.size(1), dtype=torch.long, device="cuda")
                # 计算 k_pos 和 q_pos 的相对位置
                rel_pos = k_pos[None, :] - q_pos[:, None]
                # 计算相对位置的绝对值，取负，并扩展维度
                values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)
                # 计算解码偏置，与之前计算的 scales 相乘
                dec_bias = values * self.scales
                # 将解码偏置下三角化
                dec_bias.tril_(diagonal=0)

                # 将解码 mask 与解码偏置相加
                dec_mask = dec_mask + dec_bias[0]
                # 对输入进行嵌入
                out = self.emb(dec_in)
                # 使用 Transformer 解码器层处理输出
                out = self.dec_layer(out, enc_out, tgt_mask=dec_mask)
                # 返回线性层的输出
                return self.head(out)

        # 创建 Repro 类的实例，并移动到 GPU 上
        mod = Repro().cuda()
        # 使用 torch._dynamo.optimize 函数优化模型结构，设置 dynamic=True
        opt_mod = torch._dynamo.optimize("inductor", dynamic=True)(mod)
        # 设置模型为评估模式
        mod.eval()
        opt_mod.eval()

        # 创建一个大小为 (1, 512, 256) 的随机 tensor，放置在 GPU 上
        enc_out = torch.rand(1, 512, 256).cuda()
        # 创建包含不同大小的 dec_in tensor 列表，每个 tensor 中的元素是在 [0, 512) 范围内的随机整数，放置在 GPU 上
        dec_inputs = [
            torch.randint(0, 512, (1, i + 1), dtype=torch.long).cuda() for i in range(8)
        ]

        # 遍历 dec_inputs 列表中的每个 dec_in tensor
        for dec_inp in dec_inputs:
            # 断言两个模型在给定 enc_out 和 dec_inp 下的前向传播结果相同，只检查前向传播
            assert same_two_models(
                mod, opt_mod, [enc_out, dec_inp], only_fwd=True
            ), "Inductor with dynamic shapes failed"
    def test_issue97695_1input(self):
        def fn(arg3_1, relu, permute_1):
            # 调用 Torch 提供的 addmm 操作，对 arg3_1, relu, permute_1 进行矩阵相乘并相加
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            # 使用 Torch 提供的 cat 操作，沿着第一个维度将 addmm_1 拼接成一个张量
            cat_2 = torch.ops.aten.cat.default([addmm_1], 1)
            return (cat_2,)

        # 定义多组参数用于测试
        args = [
            ((96,), (1,), torch.float32, "cuda"),
            ((10, 256), (256, 1), torch.float32, "cuda"),
            ((256, 96), (1, 256), torch.float32, "cuda"),
        ]
        # 使用 rand_strided 函数生成参数的随机化版本
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        # 计算正确结果
        correct = fn(*args)

        # 使用 make_fx 函数创建模型并进行追踪
        mod = make_fx(fn, tracing_mode="real")(*args)
        # 编译追踪后的模型
        compiled = compile_fx_inner(mod, args)
        # 计算编译后模型的结果
        ref = compiled(list(args))
        # 断言编译后结果与正确结果相同
        assert same(ref, correct)

        # 使用 torch.compile 函数对原函数进行编译
        ref = torch.compile(fn, fullgraph=True)(*args)
        # 断言编译后结果与正确结果相同
        assert same(ref, correct)

    def test_issue_103924(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = 1
                self.layer = torch.nn.Softmax(dim=1)

            def forward(self, x):
                n_samples, _ = x.shape
                y = 1.0 * torch.ones(n_samples, dtype=x.dtype, device=x.device)
                inp = x / y[..., None]
                return self.layer(inp)

        # 生成随机输入张量
        x = torch.rand([4, 4], device="cuda")
        # 实例化自定义模块
        m = MyModule()
        # 使用 torch.compile 函数优化 MyModule 模块
        opt_m = torch.compile(backend="inductor")(m)
        # 断言优化后模块与原模块在输入 x 下的输出相同
        self.assertEqual(opt_m(x), m(x))

    def test_issue97695_2input(self):
        def fn(arg3_1, arg3_2, relu, permute_1):
            # 分别对 arg3_1, arg3_2 进行 Torch 的 addmm 操作
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            addmm_2 = torch.ops.aten.addmm.default(arg3_2, relu, permute_1)
            # 使用 Torch 提供的 cat 操作，沿着第一个维度将 addmm_1 和 addmm_2 拼接成一个张量
            cat_2 = torch.ops.aten.cat.default([addmm_1, addmm_2], 1)
            return (cat_2,)

        # 定义多组参数用于测试
        args = [
            ((96,), (1,), torch.float32, "cuda"),
            ((96,), (1,), torch.float32, "cuda"),
            ((10, 256), (256, 1), torch.float32, "cuda"),
            ((256, 96), (1, 256), torch.float32, "cuda"),
        ]
        # 使用 rand_strided 函数生成参数的随机化版本
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        # 计算正确结果
        correct = fn(*args)

        # 使用 torch.compile 函数对原函数进行编译
        ref = torch.compile(fn, fullgraph=True)(*args)
        # 断言编译后结果与正确结果相同
        assert same(ref, correct)
    # 测试嵌入变量的均值计算
    def test_embedding_var_mean(self):
        # 定义前向计算函数
        def forward(arg0_1):
            # 创建一个大小为 [1, 2048]，元素全为 1 的张量，数据类型为 torch.float32，布局为 torch.strided
            # 放置在 CUDA 设备上，不使用固定内存
            full = torch.ops.aten.full.default(
                [1, 2048],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=torch.device(type="cuda", index=0),
                pin_memory=False,
            )
            # 将 full 张量转换为 torch.int64 类型
            convert_element_type_1 = torch.ops.prims.convert_element_type.default(
                full, torch.int64
            )
            # 对 convert_element_type_1 沿着维度 1 进行累加求和
            cumsum = torch.ops.aten.cumsum.default(convert_element_type_1, 1)
            # 对 cumsum 张量和 convert_element_type_1 张量对应元素相乘
            mul = torch.ops.aten.mul.Tensor(cumsum, convert_element_type_1)
            # 对 mul 张量中的每个元素减去 1
            sub_1 = torch.ops.aten.sub.Tensor(mul, 1)
            # 对 sub_1 张量进行切片操作，从维度 0 的索引 0 开始，取到最大可能的长度
            slice_5 = torch.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807)
            # 对 slice_5 张量进行切片操作，从维度 1 的索引 0 开始，取到最大可能的长度
            slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807)
            # 在 slice_6 张量的每个元素上加 2
            add_2 = torch.ops.aten.add.Tensor(slice_6, 2)
            # 使用 arg0_1 对象进行嵌入操作，嵌入表是 add_2 张量
            embedding_1 = torch.ops.aten.embedding.default(arg0_1, add_2)
            # 对 embedding_1 张量进行方差和均值计算，对第 2 维进行修正
            var_mean = torch.ops.aten.var_mean.correction(
                embedding_1, [2], correction=0, keepdim=True
            )
            # 返回计算结果列表，包含 var_mean[0], var_mean[1], add_2
            return [var_mean[0], var_mean[1], add_2]

        # 创建一个大小为 [2050, 768] 的 CUDA 张量 emb，用于测试
        emb = torch.randn([2050, 768], device="cuda")
        # 使用 make_fx 函数将 forward 函数编译成图模式
        gm = make_fx(forward)(emb)
        # 编译图 gm，并传入 emb 作为参数
        opt = torch._inductor.compile_fx.compile_fx_inner(gm, [emb])
        # 执行优化后的计算图，传入 emb 作为参数
        opt([emb])
        # 同步 CUDA 设备上的所有流
        torch.cuda.synchronize()

    # 测试确定性算法
    def test_deterministic_algorithms(self):
        # 定义一个函数 fn，用于在 CUDA 设备上进行计算
        N = 10000
        @torch.compile
        def fn(idx, values):
            # 创建一个大小为 [1] 的 CUDA 张量 x，所有元素为 0
            x = torch.zeros(1, device="cuda")
            # 将 values 添加到 x 张量的 idx 索引处
            x[idx] += values
            return x

        # 创建一个大小为 N 的零张量 idx，数据类型为 torch.int64，存储在 CUDA 设备上
        idx = torch.zeros(N, dtype=torch.int64, device="cuda")
        # 创建一个大小为 N 的随机张量 values，存储在 CUDA 设备上
        values = torch.randn(N, device="cuda")

        # 使用 fn 函数计算 r0 结果
        r0 = fn(idx, values)
        # 开启确定性计算环境
        with DeterministicGuard(True):
            # 使用 fn 函数计算 r1 结果
            r1 = fn(idx, values)
            # 迭代 10 次，使用 fn 函数计算 rn 结果，并断言 r1 与 rn 相等
            for _ in range(10):
                rn = fn(idx, values)
                self.assertEqual(r1, rn, atol=0, rtol=0)

    # 测试线性层的 CPU 输入
    # https://github.com/pytorch/pytorch/issues/96406
    def test_linear_cpu_input(self):
        # 定义一个模型类 Model，继承自 nn.Module
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为 4，输出维度为 4
                self.linear = nn.Linear(4, 4)

            def forward(self, data):
                # 将数据转移到 CUDA 设备上，然后传入线性层计算
                data = data.to("cuda")
                return self.linear(data)

        # 创建一个 Model 类的实例 mod，并将其移动到 CUDA 设备上，并设为评估模式
        mod = Model().cuda().eval()
        # 在无梯度计算下执行以下代码块
        with torch.no_grad():
            # 调用 common 方法，传入 mod 和一个随机张量作为参数
            self.common(mod, (torch.randn(4, 4),))

    # 配置补丁，设置 fallback_random 为 True，triton.cudagraphs 为 True
    @config.patch({"fallback_random": True, "triton.cudagraphs": True})
    def test_xlnet_lm_stride_repro(self):
        # 定义一个名为 Repro 的内部类，继承自 nn.Module
        class Repro(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 dropout 层，丢弃概率为 0.1
                self.dropout = nn.Dropout(p=0.1, inplace=False)

            # 定义前向传播方法
            def forward(self, x):
                # 使用 PyTorch 的 gelu 激活函数
                y = torch._C._nn.gelu(x)
                # 对 y 应用 dropout 操作
                return self.dropout(y)

        # 实例化 Repro 类为 mod 对象
        mod = Repro()
        # 生成一个随机张量 x，大小为 (512, 1, 4096)，需要梯度，放置在 GPU 上
        x = torch.randn((512, 1, 4096), requires_grad=True, device="cuda")
        # 调用 mod 对象的 forward 方法，计算输出 y
        y = torch.compile(mod)(x)
        # 对 y 求和并反向传播梯度
        # 指示器声称 gelu 的保存变量用于反向传播的输出布局将是 (4096, 4096, 1)，
        # 实际上是 (4096, 2097152, 1)。幸运的是，在实践中这并不重要。
        y.sum().backward()

    def test_lookup_seed_backward(self):
        # 定义一个带有 fullgraph=True 修饰器的函数 forward
        @torch.compile(fullgraph=True)
        def forward(inductor_seeds, mul_4, view_15):
            # 使用 torch.ops.prims.inductor_lookup_seed.default 函数获取种子值
            inductor_lookup_seed_2 = torch.ops.prims.inductor_lookup_seed.default(
                inductor_seeds, 2
            )
            # 使用 torch.ops.prims.inductor_random.default 函数生成随机数张量
            inductor_random_2 = torch.ops.prims.inductor_random.default(
                [2, 512, 768], inductor_lookup_seed_2, "rand"
            )
            # 对 inductor_random_2 应用大于操作
            gt_2 = torch.ops.aten.gt.Scalar(inductor_random_2, 0.1)
            # 对 gt_2 和 view_15 应用乘法操作
            mul_7 = torch.ops.aten.mul.Tensor(gt_2, view_15)
            # 对 mul_7 和常量 1.1111111111111112 应用乘法操作
            mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112)
            # 对 mul_8 和 mul_4 应用加法操作
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_4)
            # 对 add_5 应用 torch.ops.aten.var_mean.correction 函数，计算方差和均值
            var_mean_1 = torch.ops.aten.var_mean.correction(
                add_5, [2], correction=0, keepdim=True
            )
            # 从 var_mean_1 中获取索引为 1 的元素
            getitem_3 = var_mean_1[1]
            # 对 add_5 和 getitem_3 应用减法操作
            sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3)
            # 返回结果的元组形式
            return (sub_3,)

        # 初始化三个在 GPU 上的零张量
        buf0 = torch.zeros((37,), dtype=torch.int64, device="cuda")
        buf1 = torch.zeros((2, 512, 768), device="cuda")
        buf2 = torch.zeros((2, 512, 768), device="cuda")
        # 调用 forward 函数，传入初始化的三个张量
        forward(buf0, buf1, buf2)

    def test_issue100806(self):
        # 定义一个名为 Model 的类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个线性层
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 30)
                # 初始化一个 ReLU 激活函数
                self.relu = torch.nn.ReLU()

            # 定义前向传播方法
            def forward(self, x):
                # 应用第一个线性层
                x = self.linear1(x)
                # 应用第二个线性层
                x = self.linear2(x)
                # 沿着第二维度连接 x 和 x
                x = torch.cat((x, x), dim=1)
                # 将 x 重新视图为指定形状
                x = x.view(-1, 2, 30)
                # 选择 x 的第一列
                x = x[:, 1, :]
                # 应用 ReLU 激活函数
                x = self.relu(x)
                # 返回结果 x
                return x

        # 指定设备为 GPU
        device = "cuda"
        batch_size = 2
        # 生成一个随机张量 x，大小为 (batch_size, 10)，放置在 GPU 上
        x = torch.randn(batch_size, 10).to(device)
        # 实例化 Model 类为 func 对象，并放置在 GPU 上
        func = Model().to(device)

        # 进入无梯度计算的上下文管理器
        with torch.no_grad():
            # 设置 func 为训练模式的 False
            func.train(False)
            # 编译 func 对象
            jit_func = torch.compile(func)

            # 分别计算使用 func 和 jit_func 的结果 res1 和 res2
            res1 = func(x)
            res2 = jit_func(x)
            # 断言两个结果相等
            self.assertEqual(res1, res2)
    def test_issue103481(self):
        def fn(x, y):
            # NOTE: 6 dimensions is important! does not fail for 5 dimensions
            # 计算 x 在第 2 到第 5 维度上的平均值，并保持维度
            mean = torch.mean(x, [2, 3, 4, 5], keepdim=True)
            # 将平均值张量和 y 相加
            add = mean + y
            return add

        # 创建两个随机张量，指定在 CUDA 上运行
        x = torch.rand(4, 4, 4, 4, 4, 4, device="cuda")
        y = torch.rand((), device="cuda")
        # 计算预期的结果
        expect = fn(x, y)

        # 编译优化 fn 函数
        opt_fn = torch.compile(fn)
        # 获取优化后的实际结果
        actual = opt_fn(x, y)

        # 断言预期结果和实际结果相等
        self.assertEqual(expect, actual)

    @config.patch({"triton.dense_indexing": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_bucketize_dynamic_dense(self):
        """
        Make sure that ops.bucketize() can handle dense_indexing, which previously
        caused issues due to incorrect handling of the size of offsets.
        """

        def fn(values, offsets):
            # 使用 torch.bucketize 对 values 应用 offsets 进行分桶操作
            return torch.bucketize(values, offsets)

        # 创建 CUDA 上的随机张量 values 和 offsets
        values = torch.rand((64, 64), device="cuda")
        offsets = torch.tensor([0.05, 0.1, 0.5, 0.8, 0.85, 0.95], device="cuda")

        # 计算预期结果
        expect = fn(values, offsets)

        # 动态编译优化 fn 函数
        opt_fn = torch.compile(fn, dynamic=True)
        # 获取优化后的实际结果
        actual = opt_fn(values, offsets)

        # 断言预期结果和实际结果相等
        self.assertEqual(expect, actual)

    def test_float64_constants(self):
        def fn():
            # NOTE: tensors of all the same value are constant folded, so we
            # need a tensor with two distinct values
            # 创建一个包含两个不同值的 float64 类型的张量，并在 CUDA 上运行
            a = torch.tensor([1 / 10, 2 / 10], dtype=torch.float64, device="cuda")
            # 返回张量乘以 2e50 的结果
            return a * 2e50

        # 编译优化 fn 函数
        cfn = torch.compile(fn)
        # 计算预期结果
        expect = fn()
        # 获取优化后的实际结果
        actual = cfn()
        # 断言预期结果和实际结果相等，允许零绝对和相对误差
        self.assertEqual(expect, actual, atol=0, rtol=0)

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_inplace_cudagraph(self):
        def fn(x, y, z):
            # 创建一个与 x 相同形状的零张量，并在 CUDA 上运行
            x = torch.zeros_like(x)
            # 在 x 上使用索引 y，将 z 插入，使用原地操作
            return x.index_put_([y], z, True)

        # 创建两个 CUDA 上的零张量 x 和 y，以及一个全为 True 的张量 z
        x = torch.zeros((512, 512), device="cuda", dtype=torch.bool)
        y = torch.zeros((512,), device="cuda", dtype=torch.int64)
        z = torch.ones((512, 512), device="cuda", dtype=torch.bool)

        # 使用动态编译优化 fn 函数
        opt_fn = torch._dynamo.optimize("inductor")(fn)

        # 计算预期结果
        ref = fn(x, y, z)

        # 运行两次以测试 CUDA 图问题
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        # 断言预期结果和最终结果相等
        self.assertEqual(ref, res)

    @config.patch({"triton.cudagraphs": True})
    @config.patch({"fx_graph_cache": True})
    def test_index_put_cudagraph(self):
        # 执行两次测试
        for _ in range(2):

            def fn(x, y, z):
                # 创建一个与 x 相同形状的全零张量
                x = torch.zeros_like(x)
                # 使用索引 y 将张量 x 中指定位置更新为张量 z 的值
                return x.index_put([y], z, True)

            # 在 CUDA 设备上创建全零布尔张量 x
            x = torch.zeros((512, 512), device="cuda", dtype=torch.bool)
            # 在 CUDA 设备上创建全零长整型张量 y
            y = torch.zeros((512,), device="cuda", dtype=torch.int64)
            # 在 CUDA 设备上创建全一布尔张量 z
            z = torch.ones((512, 512), device="cuda", dtype=torch.bool)

            # 使用 torch._dynamo.optimize("inductor") 对 fn 函数进行优化
            opt_fn = torch._dynamo.optimize("inductor")(fn)

            # 计算参考结果
            ref = fn(x, y, z)

            # 两次运行以测试 CUDA 图问题
            res = opt_fn(x, y, z)
            res = opt_fn(x, y, z)

            # 断言结果相等
            self.assertEqual(ref, res)
            # 重置动态编译环境
            torch._dynamo.reset()
            # 手动触发垃圾回收
            gc.collect()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported"
    )
    def test_flash_attention_dynamic(self):
        # 定义一个模型类
        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                # 定义线性层 q, k, v
                self.q = nn.Linear(1024, 1024)
                self.k = nn.Linear(1024, 1024)
                self.v = nn.Linear(1024, 1024)

            def forward(self, x):
                # 获取输入张量 x 的批量大小、序列长度和特征维度
                batch_size, seq_len, _ = x.size()

                # 将 q, k, v 的输出重塑为适合注意力计算的形状
                queries = self.q(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                keys = self.k(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                values = self.v(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)

                # 执行缩放点积注意力机制
                attn = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                )

                return attn

        # 使用 torch._dynamo.testing.CompileCounterWithBackend 对象计数
        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        # 创建模型实例并移至 CUDA 设备，使用半精度浮点数
        model = Model().cuda().half()
        # 使用指定后端编译模型
        model = torch.compile(model, backend=cnts, dynamic=True)

        # 使用具有不同形状的输入进行测试
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            input1 = torch.rand(5, 512, 1024, device="cuda", dtype=torch.float16)
            input2 = torch.rand(5, 513, 1024, device="cuda", dtype=torch.float16)
            input3 = torch.rand(5, 514, 1024, device="cuda", dtype=torch.float16)

            # 执行模型推理
            out1 = model(input1)
            out2 = model(input2)
            out3 = model(input3)

        # 断言编译帧计数为 1
        self.assertEqual(cnts.frame_count, 1)

    @config.patch({"triton.cudagraphs": True})
    # 定义测试函数，测试在没有回退的情况下使用 CUDA 图的索引放置功能
    def test_index_put_no_fallback_cudagraph(self):
        # 定义内部函数 fn，接受 x, y, z 作为参数
        def fn(x, y, z):
            # 创建一个与 x 同样大小的零张量
            x = torch.zeros_like(x)
            # 使用索引放置功能，将张量 z 放置在索引 y 处，允许原位操作
            return x.index_put([y], z, True)

        # 创建一个大小为 (512, 512) 的零张量，存储在 CUDA 设备上，数据类型为 int32
        x = torch.zeros((512, 512), device="cuda", dtype=torch.int32)
        # 创建一个大小为 (512,) 的零张量，存储在 CUDA 设备上，数据类型为 int64
        y = torch.zeros((512,), device="cuda", dtype=torch.int64)
        # 创建一个大小为 (512, 512) 的全为 1 的张量，存储在 CUDA 设备上，数据类型为 int32
        z = torch.ones((512, 512), device="cuda", dtype=torch.int32)

        # 对 fn 进行优化，使用 torch._dynamo.optimize("inductor") 进行优化
        opt_fn = torch._dynamo.optimize("inductor")(fn)

        # 计算参考结果
        ref = fn(x, y, z)

        # 运行两次以测试 CUDA 图的问题
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        # 断言参考结果与优化后结果相等
        self.assertEqual(ref, res)

    # https://github.com/pytorch/pytorch/issues/104937
    # 测试当输入特征大小为 0 时的线性层行为
    def test_linear_with_zero_infeature_size(self):
        # 创建一个输入特征和输出特征大小均为 0 的线性层，带有偏置，存储在 CUDA 设备上
        m = nn.Linear(in_features=0, out_features=0, bias=True).to("cuda")
        # 创建一个形状为 (1, 1, 0) 的随机张量，存储在 CUDA 设备上
        x = torch.rand(1, 1, 0, device="cuda")
        # 计算期望的输出
        expect = m(x)
        # 编译线性层 m
        opt_fn = torch.compile(m)
        # 计算实际输出
        actual = opt_fn(x)
        # 断言期望输出与实际输出相等
        self.assertEqual(expect, actual)

    # 用于测试多输出布局回退的功能
    @config.patch(fallback_random=True)
    def test_multi_output_layout_fallback(self):
        # 创建一个具有指定下限、上限和就地替换属性的 RReLU 模块
        mod = nn.RReLU(lower=3.2350976, upper=8.4220314, inplace=True)
        # 创建一个形状为 [4, 4] 的随机张量，存储在 CUDA 设备上
        inp = torch.rand([4, 4]).cuda()
        # 编译 RReLU 模块
        m = torch.compile(mod)

        # 冻结随机数生成状态
        with freeze_rng_state():
            # 计算 m 的输出
            o1 = m(inp.clone())

        # 计算 mod 的输出
        o2 = mod(inp.clone())

        # 断言编译后的输出与原始模块的输出相等
        self.assertEqual(o1, o2)

    # 用于测试 int8 数据类型的张量在一个内核下进行拼接的功能
    def test_cat_int8_one_kernel(self):
        # 定义一个 cat 函数，接受 inps 作为参数
        @torch.compile()
        def cat(inps):
            # 对输入张量列表进行拼接，并在结果上加 1
            return torch.cat(inps) + 1

        # 遍历数据类型列表，分别为 uint8 和 int8
        for dtype in [torch.uint8, torch.int8]:
            # 创建一个包含四个形状为 [256, 256] 的空张量列表，数据类型为 dtype，存储在 CUDA 设备上
            inps = [
                torch.empty([256, 256], dtype=dtype, device="cuda") for _ in range(4)
            ]

            # 运行 cat 函数，并获取运行时的输出和代码
            out, code = run_and_get_code(cat, inps)
            # 断言拼接后的张量加 1 的结果与 cat 函数的输出相等
            self.assertEqual(torch.cat(inps) + 1, out)
            # 使用 FileCheck 验证代码中未包含默认的 aten.cat.default，且仅运行了一个 ".run("
            FileCheck().check_not("aten.cat.default(").check_count(
                ".run(", 1, exactly=True
            ).run(code[0])

    # 配置使用块指针的测试补丁
    @config.patch("triton.use_block_ptr", True)
    def test_selecsls42b_misaligned_address(self):
        # 测试用例名称：test_selecsls42b_misaligned_address
        # GitHub issue链接：https://github.com/openai/triton/issues/2836

        @torch.compile(fullgraph=True)
        # 定义编译函数，并设置完整图标志为True
        def fn(arg207_1, arg208_1, convert_element_type_40, expand, full, mul_3):
            # 计算除法，将expand除以16
            div = torch.ops.aten.div.Scalar(expand, 16)
            # 使用torch.ops.aten.where.self函数，执行条件判断
            where = torch.ops.aten.where.self(arg207_1, full, div)
            # 转换元素类型为float32
            convert_element_type_43 = torch.ops.prims.convert_element_type.default(
                where, torch.float32
            )
            # 按指定维度列表[0, 2, 3]对tensor进行求和
            sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_43, [0, 2, 3])
            # 对tensor进行减法操作
            sub = torch.ops.aten.sub.Tensor(convert_element_type_40, arg208_1)
            # 对tensor进行乘法操作
            mul = torch.ops.aten.mul.Tensor(convert_element_type_43, sub)
            # 按指定维度列表[0, 2, 3]对tensor进行求和
            sum_3 = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3])
            # 对tensor进行乘法操作
            mul_1 = torch.ops.aten.mul.Tensor(sum_2, 0.0078125)
            # 在指定维度上增加尺寸为1的维度
            unsqueeze = torch.ops.aten.unsqueeze.default(mul_1, 0)
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3)
            # 对tensor进行乘法操作
            mul_2 = torch.ops.aten.mul.Tensor(sum_3, 0.0078125)
            mul_4 = torch.ops.aten.mul.Tensor(mul_2, mul_3)
            unsqueeze_3 = torch.ops.aten.unsqueeze.default(mul_4, 0)
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2)
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            # 对tensor进行乘法操作
            mul_6 = torch.ops.aten.mul.Tensor(sub, unsqueeze_5)
            sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_43, mul_6)
            sub_2 = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2)
            # 返回一个包含sub_2的元组
            return (sub_2,)

        # 准备函数调用所需的参数
        args = [
            torch.randn((8, 1024, 4, 4), device="cuda") > 0,  # torch.bool tensor
            torch.randn((1, 1024, 1, 1), device="cuda"),
            torch.randn((8, 1024, 4, 4), device="cuda"),
            torch.randn((8, 1024, 1, 1), dtype=torch.float16, device="cuda").expand(
                (8, 1024, 4, 4)
            ),
            torch.randn((), device="cuda"),
            torch.randn((1024,), device="cuda"),
        ]
        # 调用函数fn，传入args作为参数
        fn(*args)
        # 同步CUDA设备，以解决Triton错误 [CUDA]: misaligned address
        torch.cuda.synchronize()
    def test_non_commutative_scan_op(self):
        # 导入必要的模块和函数
        from torch._higher_order_ops.associative_scan import associative_scan
        
        # 在 CUDA 设备上生成随机张量 a 和 b
        a = torch.randn(1024, 8192, dtype=torch.float64, device="cuda")
        b = torch.randn(1024, 8192, dtype=torch.float64, device="cuda")

        # 定义基准函数 baseline
        def baseline(v, u):
            A = []
            A.append(b[:, 0])
            # 对于每列数据，按照公式计算结果并存储在列表 A 中
            for i in range(1, v.shape[1]):
                A.append(a[:, i] * A[i - 1] + b[:, i])
            return torch.stack(A, dim=1)

        # 定义结合函数 combine_fn
        def combine_fn(i, j):
            ia, ib = i
            ja, jb = j
            return ia * ja, ib * ja + jb

        # 编译 scan 操作的函数
        @torch.compile
        def compiled_scan(a, b):
            return associative_scan(combine_fn, (a, b), dim=-1)[1]

        # 调用基准函数和编译后的函数，检验结果是否一致
        out1 = baseline(a, b)
        out2 = compiled_scan(a, b)
        self.assertEqual(out1, out2)

    def test_dynamic_persistent_reductions(self):
        # 定义动态编译函数 inner_reduce
        @torch.compile(dynamic=True)
        def inner_reduce(x):
            assert x.shape[1] <= 1024
            return x.sum(1)

        # 在 CUDA 设备上生成随机张量 a
        a = torch.randn(50, 600, device="cuda")
        
        # 运行 inner_reduce 函数并获取其输出和生成的代码
        out, code = run_and_get_code(inner_reduce, a)
        self.assertEqual(inner_reduce(a), out)
        self.assertTrue("for roffset" not in code)

        # 定义动态编译函数 outer_reduce
        @torch.compile(dynamic=True)
        def outer_reduce(x):
            assert x.shape[0] <= 64
            return x.sum(0)

        # 运行 outer_reduce 函数并获取其输出和生成的代码
        out, code = run_and_get_code(outer_reduce, a)
        self.assertEqual(outer_reduce(a), out)
        self.assertTrue("for roffset" not in code)

    def test_epilogue_fusion_with_view(self):
        # 定义一个简单的神经网络模型 ToyModel
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.linear = torch.nn.Linear(262144, 100)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 定义模型的前向传播过程
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.relu(self.linear(x))

        # 在 CUDA 设备上创建 ToyModel 实例 m 和输入张量 input_tensor
        m = ToyModel().to(device="cuda:0")
        input_tensor = torch.randn(32, 3, 64, 64).to(device="cuda:0")

        # 导入必要的模块和函数
        from torch._inductor.utils import fresh_inductor_cache
        
        # 使用 fresh_inductor_cache 上下文，编译模型 m
        with fresh_inductor_cache():
            cm = torch.compile(m, mode="max-autotune")
            # 运行编译后的模型和原始模型，并比较结果
            out = cm(input_tensor)
            out2 = m(input_tensor)
            self.assertEqual(out, out2, atol=1e-3, rtol=1e-3)
    def test_reflection_pad_loop_order(self):
        # 定义一个函数 fn，接收两个张量 x 和 y
        def fn(x, y):
            # 对张量 x 和 y 进行反射填充，每个维度填充 5 个像素
            a = torch.nn.functional.pad(x, (5, 5, 5, 5), mode="reflect")
            b = torch.nn.functional.pad(y, (5, 5, 5, 5), mode="reflect")
            # 返回填充后的张量 a 和 b 的元素求和结果
            return a + b

        # 编译函数 fn，生成可运行的加速代码 cfn
        cfn = torch.compile(fn)
        # 创建两个大小为 (10, 10, 10) 的随机张量 a 和 b，存储在 CUDA 设备上
        a = torch.rand((10, 10, 10), device="cuda")
        b = torch.rand((10, 10, 10), device="cuda")
        # 计算预期结果，即调用 fn 函数得到的结果
        expect = fn(a, b)
        # 运行加速后的函数 cfn，并获取其结果 actual 和生成的代码 code
        actual, code = run_and_get_code(cfn, a, b)
        # 断言预期结果与实际结果相等
        self.assertEqual(expect, actual)

        # 断言代码在迭代过程中是按照连续的顺序进行的，并且没有进行分块处理
        kernel_code = "\n".join(code[0].split("\n")[50:64])
        self.assertExpectedInline(
            kernel_code,
            """\
# 使用 @triton.jit 装饰器，将下面的函数编译为 Triton 内核
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # 将 xnumel 设为 4000
    xnumel = 4000
    # 计算当前程序块的偏移量
    xoffset = tl.program_id(0) * XBLOCK
    # 计算当前索引范围
    xindex = xoffset + tl.arange(0, XBLOCK)[:]

    # 创建掩码，用于检查当前索引是否小于 xnumel
    xmask = xindex < xnumel

    # 计算 x0, x1, x2, x3
    x0 = xindex % 20
    x1 = (xindex // 20) % 20
    x2 = (xindex // 400)
    x3 = xindex

    # 从 in_ptr0 和 in_ptr1 加载数据到 tmp0 和 tmp1，根据掩码 xmask 以 'evict_last' 策略驱逐缓存
    tmp0 = tl.load(in_ptr0 + (99 + ((-1)*(tl_math.abs((-9) + (tl_math.abs((-5) + x0))))) + ((-10)*(tl_math.abs((-9) + (tl_math.abs((-5) + x1))))) + (100*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (99 + ((-1)*(tl_math.abs((-9) + (tl_math.abs((-5) + x0))))) + ((-10)*(tl_math.abs((-9) + (tl_math.abs((-5) + x1))))) + (100*x2)), xmask, eviction_policy='evict_last')

    # 计算 tmp0 和 tmp1 的和，存储到 tmp2
    tmp2 = tmp0 + tmp1

    # 将 tmp2 存储到 out_ptr0 的 x3 处，根据 xmask 控制写入
    tl.store(out_ptr0 + (x3), tmp2, xmask)

@skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
# 定义测试函数 test_int64_index_intermediate
def test_int64_index_intermediate(self):
    # 定义内部函数 foo，接受输入 inp
    def foo(inp):
        # 使用 torch.ops.aten.view.default 对 inp 进行形状重塑为 [-1, 8192, 8192]
        view_23 = torch.ops.aten.view.default(inp, [-1, 8192, 8192])
        # 使用 torch.ops.aten.split.Tensor 对 view_23 进行分割成长度为 1024 的张量列表 split_1
        split_1 = torch.ops.aten.split.Tensor(view_23, 1024, 1)
        # 清空 view_23
        view_23 = None

        # 依次获取 split_1 中的各个张量
        getitem_17 = split_1[0]
        getitem_18 = split_1[1]
        getitem_19 = split_1[2]
        getitem_20 = split_1[3]
        getitem_21 = split_1[4]
        getitem_22 = split_1[5]
        getitem_23 = split_1[6]
        getitem_24 = split_1[7]

        # 清空 split_1
        split_1 = None

        # 使用 torch.ops.aten.cat.default 进行张量的拼接
        cat_1 = torch.ops.aten.cat.default(
            [
                getitem_17,
                getitem_18,
                getitem_19,
                getitem_20,
                getitem_21,
                getitem_22,
                getitem_23,
                getitem_24,
            ]
        )

        # 清空 getitem_17 到 getitem_24
        getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = None

        # 返回拼接后的张量 cat_1
        return cat_1

    # 遍历 mark_dynamic 标志列表
    for mark_dynamic in [False, True]:
        # 使用 torch.rand 生成随机张量 inp，数据类型为 torch.bfloat16，在 GPU 上运行
        inp = torch.rand((65536, 8192), dtype=torch.bfloat16, device="cuda")

        # 如果 mark_dynamic 为 True，则使用 torch._dynamo.mark_dynamic 标记 inp
        if mark_dynamic:
            torch._dynamo.mark_dynamic(inp, 0)

        # 编译 foo 函数为 foo_c
        foo_c = torch.compile(foo)

        # 使用 torch.testing.assert_allclose 检验 foo(inp) 和 foo_c(inp) 的结果是否接近
        torch.testing.assert_allclose(foo(inp), foo_c(inp))

# 如果当前脚本为主模块，则导入测试函数并执行
if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CUDA

    # 如果有 CUDA 并且不使用 ASAN，则运行测试函数 run_tests，需要 "filelock" 支持
    if HAS_CUDA and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
```