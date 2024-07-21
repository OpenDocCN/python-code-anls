# `.\pytorch\test\inductor\test_unbacked_symints.py`

```
# Owner(s): ["module: inductor"]

# 导入必要的模块和函数
import functools
import unittest

import torch
from torch._dynamo import config as dynamo_config  # 导入动态配置模块
from torch._inductor import config as inductor_config  # 导入感应器配置模块
from torch._inductor.test_case import TestCase as InductorTestCase  # 导入感应器测试用例基类
from torch._inductor.utils import is_big_gpu  # 导入判断是否是大型 GPU 的工具函数
from torch.testing import make_tensor  # 导入生成测试张量的函数
from torch.testing._internal.common_device_type import instantiate_device_type_tests  # 导入实例化设备类型测试的函数
from torch.testing._internal.common_utils import IS_LINUX, parametrize  # 导入 Linux 系统标识和参数化装饰器
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA, skipCUDAIf  # 导入 GPU 类型、是否有 CUDA 和跳过 CUDA 测试的工具函数

# 定义测试类，继承自感应器测试用例基类
class TestUnbackedSymints(InductorTestCase):
    
    # 装饰器：如果没有 CUDA，则跳过测试并显示 "requires cuda"
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    # 装饰器：配置动态输出形状操作捕获为真
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    # 测试函数：测试张量展开操作
    def test_expand(self, device):
        # 定义函数：接受两个张量 x 和 y
        def fn(x, y):
            # 获取 x 中非零元素的索引
            nz = torch.nonzero(x)
            # 在 nz.size 中存在未备份的符号整数
            x_exp = nz.expand([-1, 128])
            # 在目标大小中存在未备份的符号整数
            y_exp = y.expand([-1, nz.size(0)])
            return x_exp, y_exp
        
        # 示例输入数据
        example_inputs = (
            torch.randn((32), device=device),  # 生成设备上的随机张量
            torch.randn((32, 1), device=device),  # 生成设备上的随机张量
        )

        # 使用编译后的张量执行函数 fn
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 预期结果为直接调用函数 fn
        expected = fn(*example_inputs)

        # 断言编译后的结果与预期结果的接近程度
        torch.testing.assert_close(actual, expected)

    # 装饰器：如果没有 CUDA，则跳过测试并显示 "requires cuda"
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    # 装饰器：配置动态输出形状操作捕获为真
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    # 测试函数：测试在运行时使用断言的张量展开操作
    def test_expand_ok_with_runtime_assert(self, device):
        # 定义函数：接受一个张量 x
        def fn(x):
            # 获取 x 中非零元素的索引
            nz = x.nonzero()
            # 使用 torch._check 断言 nz.size(0) 等于 128
            torch._check(nz.size(0) == 128)
            # 对 nz 执行展开操作，目标大小为 [128, -1, 2]
            return nz.expand([128, -1, 2])

        # 生成指定设备上的随机张量 x
        x = make_tensor(32, 4, device=device, dtype=torch.float32, exclude_zero=True)
        # 使用编译后的张量执行函数 fn
        actual = torch.compile(fn, fullgraph=True)(x)

    # 装饰器：如果没有 CUDA，则跳过测试并显示 "requires cuda"
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    # 装饰器：配置动态输出形状操作捕获为真
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    # 测试函数：测试张量广播操作
    def test_broadcast_tensors(self, device):
        # 定义函数：接受一个张量 x
        def fn(x):
            # 获取 x 中非零元素的索引
            nz = x.nonzero()
            # 创建一个全零张量 a，大小为 [nz.size(0), 512]
            a = torch.zeros([nz.size(0), 512])
            # 创建一个全一张量 b，大小为 [nz.size(0), 1]
            b = torch.ones([nz.size(0), 1])
            # 返回 a 与 b 的乘积
            return a * b

        # 生成指定设备上的随机张量 x
        x = torch.randn(32, 4, device=device)
        # 使用编译后的张量执行函数 fn
        actual = torch.compile(fn, fullgraph=True)(x)
        # 预期结果为直接调用函数 fn
        expected = fn(x)
        # 断言编译后的结果与预期结果的接近程度
        torch.testing.assert_close(actual, expected)
    # 定义测试函数 test_autotuning，接受一个设备参数 device
    def test_autotuning(self, device):
        # 定义内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 找出 x 中非零元素的索引
            nz = torch.nonzero(x)
            # 使用 x 的设备创建全为1的张量，维度为 [非零元素个数, y 的行数]
            a = x.new_ones([nz.size(0), y.size(0)])
            # 返回 a 和 y 的矩阵乘积
            return a @ y

        # 准备示例输入数据
        example_inputs = (
            torch.randn((64), device=device),  # 随机张量，设备为指定设备
            torch.randn((32, 16), device=device),  # 随机张量，设备为指定设备
        )

        # 使用 inductor_config.patch 修改配置，设置 "max_autotune_gemm" 为 True
        with inductor_config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            # 编译函数 fn，并使用示例输入调用，获取编译后的结果
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            # 获取预期结果
            expected = fn(*example_inputs)

        # 使用 torch.testing.assert_close 检查 actual 和 expected 的近似程度
        torch.testing.assert_close(actual, expected)

    # 根据是否有 CUDA 跳过测试
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    # 定义测试函数 test_split_with_sizes，接受一个设备参数 device
    def test_split_with_sizes(self, device):
        # 定义内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 将 y 转换为列表
            l = y.tolist()
            # 使用列表 l 对 x 进行分割
            s = torch.split(x, l)
            # 计算列表 l 的元素和
            d = l[0] + l[1] + l[2]
            # 返回分割后的第一个张量的和以及列表 l 的元素和
            return s[0].sum(), d

        # 准备示例输入数据
        example_inputs = (torch.randn((32), device=device), torch.tensor((7, 16, 9)))

        # 编译函数 fn，并使用示例输入调用，获取编译后的结果
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 获取预期结果
        expected = fn(*example_inputs)

        # 使用 torch.testing.assert_close 检查 actual 和 expected 的近似程度
        torch.testing.assert_close(actual, expected)

    # 根据是否有 CUDA 跳过测试
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    # 定义测试函数 test_view_of_slice，接受一个设备参数 device
    def test_view_of_slice(self, device):
        # 定义函数 fn，接受一个参数 x
        # 测试 View.create(slice, size_with_unbacked_symint)
        def fn(x):
            # 找出 x 中非零元素的索引
            nz = torch.nonzero(x)  # 引入未支持的符号整数
            # 对非零索引进行平方
            squared = nz * nz  # 在降低 Slice 时避免 ReinterpretView
            # 对 squared 进行切片操作，从第1维开始，从倒数第2个元素到末尾
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            # 在 sliced 的第0维上增加一个维度
            view = sliced.unsqueeze(dim=0)
            # 确保输出的步幅中没有未支持的符号整数
            return view.squeeze(
                dim=0
            )  # 确保输出的步幅中没有未支持的符号整数

        # 准备示例输入数据
        example_inputs = (torch.randn(1, 1, 1, 1, device=device),)

        # 编译函数 fn，并使用示例输入调用，获取编译后的结果
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 获取预期结果
        expected = fn(*example_inputs)

        # 使用 torch.testing.assert_close 检查 actual 和 expected 的近似程度
        torch.testing.assert_close(actual, expected)

    # 根据是否有 CUDA 跳过测试，同时设置配置 "capture_scalar_outputs" 为 True
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    @inductor_config.patch({"abi_compatible": True})
    # 定义测试函数 test_triton_kernel_grid，接受一个设备参数 device
    def test_triton_kernel_grid(self, device):
        # 如果设备是 "cpu"，则跳过测试，因为 Triton 内核需要 GPU
        if device == "cpu":
            raise unittest.SkipTest("Triton kernel requires GPU")

        # 导入添加 Triton 内核的工具函数
        from torch.testing._internal.triton_utils import add_kernel

        # 定义函数 fn，接受一个参数 x
        def fn(x):
            # 计算 x 中的最大值，并与 512 比较取最大值
            maxlen = max(x.item(), 512)
            # 创建一个全为1的张量 a，长度为 maxlen，设备为指定设备
            a = torch.ones(maxlen, device=device)
            # 创建一个全为1的张量 b，长度为 maxlen，设备为指定设备
            b = torch.ones(maxlen, device=device)
            # 创建一个和 a 同样大小的零张量 out
            out = torch.zeros_like(a)
            # 在网格中使用未支持的符号整数
            add_kernel[(1, 1, maxlen)](a, b, out, maxlen, 32)
            # 返回 out
            return out

        # 准备示例输入数据
        example_inputs = (torch.randint(high=1024, size=(1,), device=device),)

        # 编译函数 fn，并使用示例输入调用，获取编译后的结果
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 获取预期结果
        expected = fn(*example_inputs)

        # 使用 torch.testing.assert_close 检查 actual 和 expected 的近似程度
        torch.testing.assert_close(actual, expected)
    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    # 跳过 CUDA 测试，如果没有 CUDA 的话，并设置动态输出形状操作的捕获为真
    def test_nonzero_in_inference_mode(self, device):
        # 定义一个函数 fn，返回输入张量 x 中非零元素的索引
        def fn(x):
            return torch.nonzero(x)

        # 准备一个示例输入，包括一个在指定设备上生成的随机整数张量
        example_inputs = (torch.randint(0, 2, (128,), device=device),)

        # 进入推断模式
        with torch.inference_mode():
            # 编译函数 fn 的计算图，返回实际输出
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            # 直接调用函数 fn，得到预期输出
            expected = fn(*example_inputs)

        # 使用测试工具检验 actual 和 expected 是否相近
        torch.testing.assert_close(actual, expected)

    @inductor_config.patch({"max_autotune": True})
    @dynamo_config.patch({"capture_scalar_outputs": True})
    # 设置最大自动调整为真，并捕获标量输出为真
    def test_equivalent_backed_unbacked(self, device):
        # 测试在有两个等效的支持和不支持的符号整数（symint）时的场景，
        # 当我们查找不支持的符号整数上的大小提示时，我们会无意中使用默认的回退提示。

        def fn(x, w, a, b):
            # 创建张量，其中第一个维度是支持和不支持的。
            u0, s0 = a.item(), b.size(0)
            unbacked = x.expand(u0, *x.shape)
            backed = x.expand(s0, *x.shape)

            # cat 函数统一支持和不支持的维度 -- 即 u0 == s0。
            cat = torch.cat([backed, unbacked, unbacked], dim=1)  # [s0, 30, 16]
            mat1 = torch.permute(cat, [0, 2, 1])  # [s0, 16, 30]
            mat2 = w.expand(u0, *w.shape)  # [u0, 30, 32]
            bmm = torch.ops.aten.bmm(mat1, mat2)
            return bmm

        # 准备示例输入，包括在指定设备上生成的随机浮点数张量和一个符号整数 backed。
        example_inputs = (
            torch.randn((10, 16), dtype=torch.float32, device=device),
            torch.randn((30, 32), dtype=torch.float32, device=device),
            torch.tensor(7, device=device),
            backed := torch.randn((7,), device=device),
        )
        torch._dynamo.mark_dynamic(backed, 0)  # 创建支持的符号整数 backed

        # 编译函数 fn 的计算图，返回实际输出
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 直接调用函数 fn，得到预期输出
        expected = fn(*example_inputs)
        # 使用测试工具检验 actual 和 expected 是否相近
        torch.testing.assert_close(actual, expected)

    @skipCUDAIf(not HAS_CUDA, "requires cuda")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    # 跳过 CUDA 测试，如果没有 CUDA 的话，并捕获标量输出为真
    def test_vertical_pointwise_reduction_fusion(self, device):
        # 测试融合点积和减少操作，并使用未支持的元素数/减少数。
        def fn(x, y, repeats):
            # 获取重复次数的值
            u0 = repeats.item()
            # 将 y 扩展为指定次数，形状为 [u0, 1, 16]
            unbacked = y.expand(u0, *y.shape)

            # 注意：我们将 x 添加到点积和减少操作中。否则，调度器将拒绝融合仅具有未支持的符号整数的操作。
            # 执行点积操作
            pointwise = unbacked + x
            # 执行减少操作，计算点积操作结果加上 x 的总和
            reduction = torch.sum(pointwise + x)
            return pointwise, reduction

        # 定义示例输入
        example_inputs = (
            torch.randn(32, 16).cuda(),
            torch.randn(1, 16).cuda(),
            torch.tensor(32).cuda(),
        )

        # 编译并执行函数 fn，获取实际输出
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 调用原始函数 fn，获取预期输出
        expected = fn(*example_inputs)
        # 断言实际输出与预期输出相近
        torch.testing.assert_close(actual, expected)
        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    @parametrize(
        "torch_fn", [torch.mm, torch.bmm, torch.addmm], name_fn=lambda fn: fn.__name__
    )
    @parametrize("coordinate_descent_tuning", [True, False], name_fn=str)
    def test_mm_and_friends(self, device, torch_fn, coordinate_descent_tuning):
        # 如果 torch_fn 是 torch.addmm，则将其部分应用到 torch.ones 上
        if torch_fn == torch.addmm:
            torch_fn = functools.partial(torch_fn, torch.ones(1, device=device))

        def fn(x, w, repeats, is_bmm):
            # 获取重复次数的值
            u0 = repeats.item()
            # 检查 u0 的尺寸
            torch._check_is_size(u0)

            # 将 x 和 w 分别扩展为指定尺寸
            x_unbacked = x.expand(u0, 32)
            w_unbacked = w.expand(32, u0)

            # 如果是 bmm 操作，则确保输入是批处理的
            if is_bmm:
                x_unbacked = x_unbacked.expand(10, *x_unbacked.shape)
                w_unbacked = w_unbacked.expand(10, *w_unbacked.shape)

            return torch_fn(x_unbacked, w_unbacked)

        # 定义示例输入
        example_inputs = (
            torch.randn(1, 32, device=device),
            torch.randn(32, 1, device=device),
            torch.tensor(100, device=device),
            torch_fn == torch.bmm,
        )

        with inductor_config.patch(
            {
                # coordinate_descent_tuning 在分解期间有其自己的路径
                "coordinate_descent_tuning": coordinate_descent_tuning,
            }
        ):
            # 编译并执行函数 fn，获取实际输出
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        # 调用原始函数 fn，获取预期输出
        expected = fn(*example_inputs)
        # 断言实际输出与预期输出相近
        torch.testing.assert_close(actual, expected)
# 调用函数 instantiate_device_type_tests，用于实例化设备类型测试对象 TestUnbackedSymints
# 这里的 globals() 函数将全局命名空间传递给 instantiate_device_type_tests
# only_for 参数指定了测试仅适用于 GPU_TYPE 或 "cpu" 类型的设备
instantiate_device_type_tests(
    TestUnbackedSymints, globals(), only_for=(GPU_TYPE, "cpu")
)

# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果操作系统为 Linux，并且具有 CUDA 支持，并且第一个 GPU 是大型 GPU
    if IS_LINUX and HAS_CUDA and is_big_gpu(0):
        # 运行测试
        run_tests()
```