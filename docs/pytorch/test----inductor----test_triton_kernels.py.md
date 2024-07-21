# `.\pytorch\test\inductor\test_triton_kernels.py`

```
# Owner(s): ["module: inductor"]
# flake8: noqa: E731
# Skip do not assign a lambda expression, use a def
# 从单元测试模块导入 patch 函数，用于模拟对象的方法调用
from unittest.mock import patch

# 导入 PyTorch 库及其子模块
import torch
import torch._dynamo.testing

# 导入 PyTorch 的内部测试相关模块
import torch._inductor.test_case

# 导入 PyTorch 高阶操作相关模块中的函数
from torch._higher_order_ops.triton_kernel_wrap import (
    generate_ttir,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)

# 导入 PyTorch 灌电模块中的指标模块
from torch._inductor import metrics

# 导入 PyTorch 灌电模块中的工具函数
from torch._inductor.utils import run_and_get_code

# 导入 PyTorch 内部测试的通用工具模块
from torch.testing._internal import common_utils

# 导入 PyTorch 内部测试的条件检查函数和标记
from torch.testing._internal.common_utils import skipIfRocm, skipIfXpu, TEST_WITH_ROCM

# 导入 torch.testing._internal.triton_utils 下的所有内容，忽略 F403 错误
from torch.testing._internal.triton_utils import *

# 导入 torch.testing._internal.inductor_utils 中定义的 GPU_TYPE, HAS_CUDA, HAS_GPU, HAS_XPU 变量
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA, HAS_GPU, HAS_XPU

# 如果有 GPU，导入 triton 模块及其 language 子模块
if HAS_GPU:
    import triton
    from triton import language as tl

    # 如果不是 ROCm 测试环境
    if not TEST_WITH_ROCM:
        # 如果有 CUDA，导入 CUDA 特定的 libdevice 函数
        if HAS_CUDA:
            from triton.language.extra.cuda.libdevice import (
                fast_dividef,
                fast_dividef as my_fast_dividef,
            )
        # 如果有 XPU，导入 Intel XPU 特定的 libdevice 函数
        elif HAS_XPU:
            from triton.language.extra.intel.libdevice import (
                fast_dividef,
                fast_dividef as my_fast_dividef,
            )

    # 在此定义共享的 triton 常量
    CONSTANT_C: tl.constexpr = 4
    STRING_CONSTANT_C: tl.constexpr = "CONSTANT_C"
    BOOL_CONSTANT_C: tl.constexpr = True

# 定义 KernelTests 类，继承自 torch._inductor.test_case.TestCase
class KernelTests(torch._inductor.test_case.TestCase):
    # 标记需要 GPU 的测试方法
    @requires_gpu
    def test_triton_kernel_with_kernel_param(self):
        # 定义一个接受 kernel 参数的 triton.jit 装饰的函数
        @triton.jit
        def pass_kernel(kernel):
            pass

        # 使用 torch.compile 编译函数 f，指定后端为 "eager"
        @torch.compile(backend="eager")
        def f(x):
            # 定义一个单元格 grid
            grid = (x.numel(),)
            # 调用 pass_kernel 函数，传入参数 kernel=x，使用 grid 进行调度
            pass_kernel[grid](kernel=x)

        # 创建一个在 GPU_TYPE 设备上的随机张量 t1
        t1 = torch.rand(5, device=GPU_TYPE)
        # 调用函数 f，传入 t1 作为参数
        f(t1)
        # 不需要做任何断言，目的是确保 dynamo 不会崩溃
        # No need to assert anything, the goal is to make sure dynamo does
        # not crash

    # 标记需要 GPU 的测试方法
    @requires_gpu
    # 定义一个测试函数，用于测试 Triton 内核的高阶函数功能
    def test_triton_kernel_higher_order_func(self):
        # 导入 Triton 内核包中的核心表
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        
        # 向核心表中添加一个名为 add_kernel 的内核
        add_kernel_id = kernel_side_table.add_kernel(add_kernel)

        # 创建在 GPU 上的随机张量 t1 和 t2
        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)

        # 使用 Torch 提供的张量加法运算
        torch_add = t1 + t2

        # 测试带有变异的高阶函数
        output = torch.zeros_like(t1)
        # 获取输出张量的元素数量
        n_elements = output.numel()
        # 向核心表中添加常量参数，如元素数量和块大小
        constant_args_idx = kernel_side_table.add_constant_args(
            {"n_elements": n_elements, "BLOCK_SIZE": 16}
        )
        # 定义执行内核的网格函数
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        # 调用 Triton 内核包装器以执行变异操作
        triton_kernel_wrapper_mutation(
            kernel_idx=add_kernel_id,
            constant_args_idx=constant_args_idx,
            grid=[grid],
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
            },
        )
        # 断言输出与 Torch 加法的结果相等
        self.assertEqual(output, torch_add)
        # 确保输出张量已被修改
        self.assertNotEqual(output, torch.zeros_like(t1))

        # 测试不带变异的高阶函数
        output = torch.zeros_like(t1)
        # 调用 Triton 内核包装器以执行非变异操作
        out_dict = triton_kernel_wrapper_functional(
            kernel_idx=add_kernel_id,
            constant_args_idx=constant_args_idx,
            grid=[grid],
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
            },
            tensors_to_clone=["in_ptr0", "in_ptr1", "out_ptr"],
        )
        # 断言输出字典中的 out_ptr 与 Torch 加法的结果相等
        self.assertEqual(out_dict["out_ptr"], torch_add)
        # 确保输出张量未被修改
        self.assertEqual(output, torch.zeros_like(t1))

    @requires_gpu
        def test_triton_kernel_functionalize(self):
            # 导入必要的库和模块
            from functorch import make_fx
            from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
            from torch._subclasses.functional_tensor import (
                CppFunctionalizeAPI,
                FunctionalTensorMode,
                PythonFunctionalizeAPI,
            )

            # 重置内核侧表
            kernel_side_table.reset_table()

            # 定义函数 f，调用 triton_kernel_wrapper_functional 进行内核包装
            def f(x, output):
                # 调用 triton_kernel_wrapper_functional 函数
                out = triton_kernel_wrapper_functional(
                    # 添加内核到侧表，并指定内核索引为 mul2_kernel
                    kernel_idx=kernel_side_table.add_kernel(mul2_kernel),
                    # 添加常量参数到侧表
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": output.numel(), "BLOCK_SIZE": 16}
                    ),
                    # 指定计算的网格大小
                    grid=[(x.numel(),)],
                    # 传递参数及其命名的指针
                    kwargs={
                        "in_ptr0": x,
                        "out_ptr": output,
                    },
                    # 需要克隆的张量列表
                    tensors_to_clone=["in_ptr0", "out_ptr"],
                )
                # 返回输出指针
                return out["out_ptr"]

            # 创建两个随机张量 t1 和 t2，设备为 GPU_TYPE
            t1 = torch.rand(5, device=GPU_TYPE)
            t2 = torch.rand(5, device=GPU_TYPE)

            # 使用 FunctionalTensorMode 上下文环境
            with FunctionalTensorMode():
                # 使用 PythonFunctionalizeAPI 对 f 进行功能化处理，并应用于 t1 和 t2
                gm = make_fx(PythonFunctionalizeAPI().functionalize(f))(t1, t2)
            # 确保 t2 没有被修改
            self.assertNotEqual(gm(t1, t2), t2)

            # 使用 CppFunctionalizeAPI 对 f 进行功能化处理，并应用于 t1 和 t2
            gm = make_fx(CppFunctionalizeAPI().functionalize(f))(t1, t2)
            # 确保 t2 没有被修改
            self.assertNotEqual(gm(t1, t2), t2)

            # 使用 torch.func.functionalize 对 f 进行功能化处理，并应用于 t1 和 t2
            gm = make_fx(torch.func.functionalize(f))(t1, t2)
            # 确保 t2 没有被修改
            self.assertNotEqual(gm(t1, t2), t2)

            # 使用追踪模式 "fake" 对 f 进行功能化处理，并应用于 t1 和 t2
            gm = make_fx(f, tracing_mode="fake")(t1, t2)
            # 断言内联生成的代码与预期的空白字符串相等
            self.assertExpectedInline(
                gm.code.strip(),
                """\
# 定义一个方法 `forward`，接受三个参数：`self`, `x_1`, `output_1`。
def forward(self, x_1, output_1):
    # 调用 Triton 内核封装功能代理函数，生成代理对象 `triton_kernel_wrapper_functional_proxy`，
    # 传递参数包括内核索引、常量参数索引、计算网格和关键字参数。
    triton_kernel_wrapper_functional_proxy = torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional(
        kernel_idx=0,
        constant_args_idx=3,
        grid=[(5,)],
        kwargs={'in_ptr0': x_1, 'out_ptr': output_1},
        tensors_to_clone=['in_ptr0', 'out_ptr']
    )
    # 清空 `x_1` 和 `output_1` 的引用，释放资源
    x_1 = output_1 = None
    # 从代理对象中获取名为 `in_ptr0` 的项目
    getitem = triton_kernel_wrapper_functional_proxy['in_ptr0']
    # 从代理对象中获取名为 `out_ptr` 的项目
    getitem_1 = triton_kernel_wrapper_functional_proxy['out_ptr']
    # 清空 Triton 内核封装功能代理对象的引用，释放资源
    triton_kernel_wrapper_functional_proxy = None
    # 返回 `out_ptr` 的值作为方法的结果
    return getitem_1
    # 使用 common_utils.parametrize 装饰器为 test_triton_kernel_with_views 方法添加参数化测试的支持，dynamic 参数取值为 False 和 True
    @common_utils.parametrize("dynamic", [False, True])
    # 使用 common_utils.parametrize 装饰器为 test_triton_kernel_with_views 方法添加参数化测试的支持，backend 参数取值为 "eager", "aot_eager", "inductor"
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    # 定义 test_triton_kernel_with_views 测试方法，接受 dynamic 和 backend 两个参数
    def test_triton_kernel_with_views(self, dynamic, backend):
        
        # 定义内部函数 call_triton_take_view，接受一个类型为 torch.Tensor 的参数 x
        def call_triton_take_view(x: torch.Tensor):
            # 创建一个与 x 同样大小的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 中元素的总数
            n_elements = output.numel()
            # 定义一个 lambda 函数 grid，用于计算执行的网格大小，使用 triton.cdiv 函数计算块大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用 mul2_kernel[grid] 执行核函数，对 x 和 output 进行操作，BLOCK_SIZE 设置为 16
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            # 返回处理后的 output 张量
            return output

        # 定义内部函数 call_triton_return_view，接受一个类型为 torch.Tensor 的参数 x
        def call_triton_return_view(x: torch.Tensor):
            # 创建一个与 x 同样大小的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 中元素的总数
            n_elements = output.numel()
            # 定义一个 lambda 函数 grid，用于计算执行的网格大小，使用 triton.cdiv 函数计算块大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用 mul2_kernel[grid] 执行核函数，对 x 和 output 进行操作，BLOCK_SIZE 设置为 16
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            # 返回对 output 执行 view 操作后的结果，将其转换为 4x4 的张量
            return output.view(4, 4)

        # 创建一个形状为 4x4 的随机张量 t，并指定设备为 GPU_TYPE
        t = torch.rand(4, 4, device=GPU_TYPE)
        # 将 t 扁平化为长度为 16 的张量 t_view
        t_view = t.view(16)

        # 使用 torch.compile 编译 call_triton_take_view 函数，指定 backend、dynamic 和 fullgraph 参数，并执行编译后的函数进行断言比较
        compiled_func = torch.compile(
            call_triton_take_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        # 断言编译后的函数对 t_view 的操作结果与 2 * t_view 相等
        self.assertEqual(2 * t_view, compiled_func(t_view))
        # 断言编译后的函数对 t_view 的操作结果 view 成 4x4 后与 2 * t 相等
        self.assertEqual(2 * t, compiled_func(t_view).view(4, 4))

        # 使用 torch.compile 编译 call_triton_return_view 函数，指定 backend、dynamic 和 fullgraph 参数，并执行编译后的函数进行断言比较
        compiled_func = torch.compile(
            call_triton_return_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        # 断言编译后的函数对 t 的操作结果 view 成长度为 16 的张量与 2 * t_view 相等
        self.assertEqual(2 * t_view, compiled_func(t).view(16))
        # 断言编译后的函数对 t 的操作结果与 2 * t 相等
        self.assertEqual(2 * t, compiled_func(t))

    # 使用 requires_gpu 装饰器标记测试方法，表明该方法需要 GPU 支持
    @requires_gpu
    # 使用 common_utils.parametrize 装饰器为 test_triton_kernel_with_grad_option 方法添加参数化测试的支持，grad_fn 参数取值为 torch.no_grad 和 torch.enable_grad
    @common_utils.parametrize("grad_fn", [torch.no_grad, torch.enable_grad])
    # 使用 common_utils.parametrize 装饰器为 test_triton_kernel_with_grad_option 方法添加参数化测试的支持，backend 参数取值为 "eager", "aot_eager", "inductor"
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    # 定义 test_triton_kernel_with_grad_option 测试方法，接受 grad_fn 和 backend 两个参数
    def test_triton_kernel_with_grad_option(self, grad_fn, backend):
        
        # 定义内部函数 call_triton，接受一个类型为 torch.Tensor 的参数 x
        def call_triton(x: torch.Tensor):
            # 根据 grad_fn 的值选择是否开启梯度计算上下文
            with grad_fn():
                # 创建一个与 x 同样大小的全零张量 output
                output = torch.zeros_like(x)
                # 计算 output 中元素的总数
                n_elements = output.numel()
                # 定义一个 lambda 函数 grid，用于计算执行的网格大小，使用 triton.cdiv 函数计算块大小
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                # 调用 mul2_kernel[grid] 执行核函数，对 x 和 output 进行操作，BLOCK_SIZE 设置为 16
                mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
                # 返回处理后的 output 张量
                return output

        # 创建一个形状为 5 的随机张量 t，并指定设备为 GPU_TYPE
        t = torch.rand(5, device=GPU_TYPE)
        # 使用 torch.compile 编译 call_triton 函数，指定 backend 和 fullgraph 参数，并执行编译后的函数进行断言比较
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        # 断言编译后的函数对 t 的操作结果与 2 * t 相等
        self.assertEqual(2 * t, compiled_func(t))

    # 使用 requires_gpu 装饰器标记测试方法，表明该方法需要 GPU 支持
    @requires_gpu
    # 使用 common_utils.parametrize 装饰器为 test_triton_kernel_with_grad_option 方法添加参数化测试的支持，backend 参数取值为 "eager", "aot_eager", "inductor"
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_inner_triton_function(self, backend):
        # 定义内部函数f，接受一个torch.Tensor作为输入
        def f(x: torch.Tensor):
            # 定义用triton.jit装饰的内核函数pow2_kernel
            @triton.jit
            def pow2_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                # 获取当前程序实例的ID
                pid = tl.program_id(axis=0)
                # 计算当前块的起始位置
                block_start = pid * BLOCK_SIZE
                # 计算偏移量数组
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                # 创建一个掩码，标记有效的元素
                mask = offsets < n_elements
                # 从内存中加载数据到x中，使用掩码过滤无效元素
                x = tl.load(in_ptr0 + offsets, mask=mask)
                # 计算平方并存储结果到output中，使用掩码过滤无效元素
                output = x * x
                # 将结果写入到内存中，使用掩码过滤无效元素
                tl.store(out_ptr + offsets, output, mask=mask)

            # 创建一个与x形状相同的全零Tensor作为输出
            output = torch.zeros_like(x)
            # 获取输出Tensor的元素数量
            n_elements = output.numel()
            # 定义网格函数，用于指定计算的网格大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用内核函数pow2_kernel来处理输入x和输出output
            pow2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            # 返回处理后的输出
            return output

        # 创建一个在GPU上随机初始化的Tensor
        t = torch.rand(5, device=GPU_TYPE)

        # 编译函数f，使用指定的后端，生成完整的计算图
        compiled_func = torch.compile(f, backend=backend, fullgraph=True)
        # TODO(oulgen): NYI - Support this
        # 断言编译后的函数对输入t的处理结果与t*t相同

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @patch.object(torch._inductor.config, "implicit_fallbacks", False)
    # 测试不使用克隆的triton内核
    def test_triton_kernel_no_clones(self, grad, dynamic):
        # 导入运行并获取代码的函数
        from torch._inductor.utils import run_and_get_code

        # 定义调用triton的函数，接受三个Tensor作为输入
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            # 获取输出Tensor的元素数量
            n_elements = output.numel()

            # 对x加1，存储到tmp中
            tmp = torch.add(x, 1)
            # 定义网格大小
            grid = (x.numel(),)
            # 运行add_kernel，处理输入x、y、output等参数
            add_kernel.run(
                x, y, output, n_elements, warmup=False, grid=grid, BLOCK_SIZE=16
            )

            # 返回处理后的输出Tensor和tmp
            return output, tmp

        # 创建两个在GPU上随机初始化的Tensor，其中o1是全零Tensor
        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        # 调用call_triton函数，处理输入t1、t2和o1
        torch_add = call_triton(t1, t2, o1)
        # 重置性能指标
        metrics.reset()
        # 创建一个与t1形状相同的全零Tensor，用于测试
        o2 = torch.zeros_like(t1, requires_grad=grad)
        # 运行并获取编译后的代码
        test, codes = run_and_get_code(
            torch.compile(call_triton, dynamic=dynamic), t1, t2, o2
        )
        # 如果不需要梯度
        if not grad:
            # 断言生成的内核数量为1
            self.assertEqual(metrics.generated_kernel_count, 1)
        # 断言torch_add与test相等
        self.assertEqual(torch_add, test)
        # 这两个断言不是最优的，因为需要原始的aten在元数据中，所以可能会有误报
        # 确保编译后的代码中没有"aten.copy"
        self.assertTrue("aten.copy" not in codes[0])
        # 确保编译后的代码中没有"aten.clone"
        self.assertTrue("aten.clone" not in codes[0])
        # 下面检查编译后的代码中只有输出Tensor在返回值中
        if dynamic and grad:
            # 确保编译后的代码中包含"return (buf0, s0, )"
            self.assertTrue("return (buf0, s0, )" in codes[0])
        else:
            # 确保编译后的代码中包含"return (buf0, )"
            self.assertTrue("return (buf0, )" in codes[0])

    @requires_gpu
    def test_triton_kernel_caching_duplicate(self):
        from torch._inductor.utils import run_and_get_code  # 导入运行和获取代码的工具函数

        class C:
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)  # 获取程序在指定轴上的ID
                block_start = pid * BLOCK_SIZE  # 计算块的起始位置
                offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 生成偏移量数组
                mask = offsets < n_elements  # 创建掩码以处理边界情况
                x = tl.load(in_ptr0 + offsets, mask=mask)  # 从输入指针处加载数据
                tl.store(out_ptr + offsets, x, mask=mask)  # 将数据存储到输出指针处

        class D:
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)  # 获取程序在指定轴上的ID
                block_start = pid * BLOCK_SIZE  # 计算块的起始位置
                offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 生成偏移量数组
                mask = offsets < n_elements  # 创建掩码以处理边界情况
                x = tl.load(in_ptr0 + offsets, mask=mask)  # 从输入指针处加载数据
                tl.store(out_ptr + offsets, x, mask=mask)  # 将数据存储到输出指针处

        def call_triton(x: torch.Tensor):
            output1 = torch.zeros_like(x)  # 创建与输入张量相同形状的零张量
            output2 = torch.zeros_like(x)  # 创建与输入张量相同形状的零张量
            n_elements = output1.numel()  # 获取输出张量中元素的总数
            grid = (n_elements,)  # 创建一个包含一个元素的元组作为网格参数
            C.pass_kernel[grid](x, output1, n_elements, BLOCK_SIZE=16)  # 调用类 C 的 JIT 编译的内核函数
            D.pass_kernel[grid](x, output2, n_elements, BLOCK_SIZE=16)  # 调用类 D 的 JIT 编译的内核函数
            return output1 + output2  # 返回两个输出张量的和

        t = torch.ones(5, device=GPU_TYPE)  # 创建一个包含五个元素的张量，并指定在 GPU 上运行
        test, (code,) = run_and_get_code(torch.compile(call_triton), t)  # 编译并获取调用 call_triton 后生成的代码
        # 确保这里发出了两个内核
        self.assertTrue("pass_kernel_0.run" in code)  # 检查第一个内核是否在生成的代码中
        self.assertTrue("pass_kernel_1.run" in code)  # 检查第二个内核是否在生成的代码中
    def test_triton_kernel_various_args(self):
        # 使用 triton.autotune 自动调优，设置 BLOCK_SIZE 为 128 的配置
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=[],
        )
        # 使用 triton.jit 对下面的 pass_kernel 函数进行 JIT 编译优化
        @triton.jit
        def pass_kernel(
            out_ptr,
            n_elements,
            dummy_None,
            dummy_empty,
            dummy_float,
            BLOCK_SIZE: "tl.constexpr",
            RANDOM_SIZE: "tl.constexpr",
        ):
            pass

        # 使用 torch.compile 对 call_triton 函数进行编译优化
        @torch.compile
        def call_triton(output):
            # 计算输出张量的元素数量
            n_elements = output.numel()
            # 定义执行核函数的网格
            grid = (n_elements,)
            # 调用 pass_kernel 函数执行计算
            pass_kernel[grid](
                output,
                n_elements,
                None,
                torch.empty_like(output),
                3.1415926,
                RANDOM_SIZE=0,
            )
            return output

        # 创建一个随机数张量在 GPU 上
        output = torch.randn(5, device=GPU_TYPE)
        # 确保调用不会崩溃
        call_triton(output)

    @requires_gpu
    @skipIfRocm
    def test_triton_kernel_dependancies(self):
        # 定义一个调用 triton 核函数的函数
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            # 创建一个与 x 形状相同的全零张量
            output = torch.zeros_like(x)
            # 计算输出张量的元素数量
            n_elements = output.numel()
            # 定义执行核函数的网格，使用 triton.cdiv 计算块的数量
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用 add_kernel_autotuned 执行计算
            add_kernel_autotuned[grid](x, y, output, n_elements)
            # 创建一个与 output 形状相同的全零张量 output2
            output2 = torch.zeros_like(output)
            # 再次调用 add_kernel_autotuned 执行计算
            add_kernel_autotuned[grid](output, y, output2, n_elements)
            # 对 output2 中的每个元素加 1，生成 output3
            output3 = torch.add(output2, 1)
            return output3

        # 创建两个随机数张量在 GPU 上
        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        # 分别调用未编译和编译后的 call_triton 函数，并比较结果
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_gpu
    def test_triton_kernel_reinplace_inplaceable_pass(self):
        # 定义一个调用 triton 核函数的函数
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            # 创建一个与 x 形状相同的全零张量
            output = torch.zeros_like(x)
            # 计算输出张量的元素数量
            n_elements = output.numel()
            # 定义执行核函数的网格，使用 triton.cdiv 计算块的数量
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用 add_kernel_autotuned 执行计算
            add_kernel_autotuned[grid](x, y, output, n_elements)
            # 再次调用 add_kernel_autotuned，使用输出作为输入的一部分
            add_kernel_autotuned[grid](output, x, output, n_elements)
            return output

        # 创建两个随机数张量在 GPU 上
        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        # 分别调用未编译和编译后的 call_triton 函数，并比较结果
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    # 定义测试函数，用于测试 Triton 内核的多内核功能
    def test_triton_kernel_multi_kernel(self, grad):
        # 定义 Triton 的 JIT 编译函数，用于执行乘以2并相加以及零化负数的内核操作
        @triton.jit
        def mul2_and_add_and_zero_negatives_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            ACTIVATION: "tl.constexpr",
        ):
            # 获取当前线程的程序 ID
            pid = tl.program_id(axis=0)
            # 计算当前线程块的起始索引
            block_start = pid * BLOCK_SIZE
            # 生成当前线程块的偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码以确保偏移量在有效范围内
            mask = offsets < n_elements
            # 调用间接内核函数，对 in_ptr0 进行操作
            indirection_kernel(
                in_ptr0,
                in_ptr0,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            # 调用间接内核函数，对 in_ptr1 进行操作
            indirection_kernel(
                in_ptr1,
                in_ptr1,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            # 加载 in_ptr0 中的数据
            x = tl.load(in_ptr0 + offsets, mask=mask)
            # 加载 in_ptr1 中的数据
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算输出结果
            output = x + y
            # 如果 ACTIVATION 等于 "zero_negs"，则执行零化负数操作
            if ACTIVATION == "zero_negs":
                output = zero_negs(output)
            # 存储输出数据到 out_ptr 中
            tl.store(out_ptr + offsets, output, mask=mask)

        # 定义 Torch 的编译函数，调用 Triton 内核执行计算
        @torch.compile
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
            xi: torch.Tensor,
            yi: torch.Tensor,
            output: torch.Tensor,
            outputi: torch.Tensor,
        ):
            # 获取输出张量的元素个数
            n_elements = output.numel()

            # 定义网格大小为 x 张量的元素个数
            grid = (x.numel(),)
            # 调用 Triton 内核进行浮点数计算
            mul2_and_add_and_zero_negatives_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=16, ACTIVATION="zero_negs"
            )
            # 调用 Triton 内核进行整数计算
            mul2_and_add_and_zero_negatives_kernel[grid](
                xi, yi, outputi, n_elements, BLOCK_SIZE=16, ACTIVATION=None
            )

            # 返回计算结果元组
            return (output, outputi)

        # 创建浮点数张量 t1 和 t2，设备为 GPU_TYPE，并标记是否需要梯度
        t1 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device=GPU_TYPE, requires_grad=grad
        )
        # 创建整数张量 t1i 和 t2i，设备为 GPU_TYPE
        t2 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device=GPU_TYPE, requires_grad=grad
        )
        # 计算浮点数结果，乘以2并确保大于等于0
        float_result = 2 * t1 + 2 * t2
        float_result = float_result.where(float_result >= 0, 0.0)

        # 创建整数随机张量 t1i 和 t2i，设备为 GPU_TYPE
        t1i = torch.randint(-2, 2, (5,), device=GPU_TYPE)
        t2i = torch.randint(-2, 2, (5,), device=GPU_TYPE)
        # 创建与 t1 形状相同的全零张量 o，并标记是否需要梯度
        o = torch.zeros_like(t1, requires_grad=grad)
        # 创建与 t1i 形状相同的全零张量 oi
        oi = torch.zeros_like(t1i)
        # 计算整数结果，乘以2
        int_result = 2 * t1i + 2 * t2i

        # 调用 Triton 计算框架进行计算，获取浮点数和整数结果
        (result, resulti) = call_triton(t1, t2, t1i, t2i, o, oi)
        # 断言浮点数结果与预期的 float_result 相等
        self.assertEqual(float_result, result)
        # 断言整数结果与预期的 int_result 相等
        self.assertEqual(int_result, resulti)

    # 标记测试需要 GPU 支持
    @requires_gpu
    # 标记测试在 XPU 环境下跳过
    @skipIfXpu
    # 标记测试在 ROCm 环境下跳过
    @skipIfRocm
    def test_triton_kernel_constants(self):
        # 定义一个测试函数，用于测试 Triton 内核常量的行为
        @triton.jit
        def mulC_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            CONSTANT_NAME: "tl.constexpr",
        ):
            # 获取当前线程的程序 ID
            pid = tl.program_id(axis=0)
            # 计算当前块的起始位置
            block_start = pid * BLOCK_SIZE
            # 计算所有元素的偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，用于标记有效数据
            mask = offsets < n_elements
            # 从输入指针读取数据到 x
            x = tl.load(in_ptr0 + offsets, mask=mask)
            # 根据常量名选择操作
            if CONSTANT_NAME == STRING_CONSTANT_C:
                output = CONSTANT_C * x
            # 如果布尔常量为真，再乘以常量值
            if BOOL_CONSTANT_C:
                output *= CONSTANT_C
            # 将计算结果存储到输出指针
            tl.store(out_ptr + offsets, output, mask=mask)

        # 调用 Triton 内核函数
        def call_triton(
            x: torch.Tensor,
        ):
            # 初始化输出张量
            output = torch.zeros_like(x)
            # 获取输出张量的元素数量
            n_elements = output.numel()

            # 定义计算的网格大小
            grid = (x.numel(),)
            # 调用 Triton 内核函数
            mulC_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=16, CONSTANT_NAME="CONSTANT_C"
            )
            return output

        # Triton 内核在解析时捕获全局常量的值，而不是在运行时
        # 如果 Triton 内核行为发生变化，此测试将失败
        global CONSTANT_C
        prev_c = CONSTANT_C
        # 更新常量值
        CONSTANT_C = 10
        # 断言新旧常量值不相等
        assert CONSTANT_C != prev_c

        # 创建一个随机张量
        t = torch.randn(5, device=GPU_TYPE)
        # 调用 Triton 函数并获取 Torch 的结果
        torch_result = call_triton(t)
        # 编译 Triton 函数并获取编译后的结果
        compiled_result = torch.compile(call_triton)(t)

        # 断言 Torch 和编译结果相等
        self.assertEqual(torch_result, compiled_result)

        # 恢复常量值
        CONSTANT_C = prev_c

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    def test_triton_kernel_autotune(self, grad, dynamic, backend, grid_type):
        # 定义一个测试函数，用于测试 Triton 内核的自动调整行为
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            # 获取输出张量的元素数量
            n_elements = output.numel()

            # 定义一个根据元数据返回网格大小的函数
            def grid_fn(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            # 根据不同的网格类型选择不同的网格定义方式
            if grid_type == 1:
                grid = (n_elements,)
            elif grid_type == 2:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            elif grid_type == 3:
                grid = grid_fn

            # 调用自动调整的 Triton 内核函数
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        # 创建两个随机张量和一个输出张量
        t1 = torch.rand(256, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(256, device=GPU_TYPE, requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        # 调用 Triton 函数并获取 Torch 的结果
        torch_add = call_triton(t1, t2, output)
        # 编译 Triton 函数并获取编译后的结果
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )

        # 创建另一个输出张量
        output2 = torch.zeros_like(t1, requires_grad=grad)
        # 断言编译后的结果和 Torch 结果相等
        self.assertEqual(compiled_func(t1, t2, output2), torch_add)

    @requires_gpu
    # 使用参数化装饰器设置多个参数组合，用于测试 Triton 内核的二维自动调优
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    # 定义测试函数，测试 Triton 内核的二维自动调优
    def test_triton_kernel_2d_autotune(self, grad, dynamic, backend, grid_type):
        # 定义调用 Triton 内核的函数
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            # 获取输出张量的尺寸
            x_elements = output.size()[0]
            y_elements = output.size()[1]

            # 定义一个返回网格维度的函数
            def grid_fn(meta):
                return (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )

            # 根据 grid_type 不同选择网格设置方式
            if grid_type == 1:
                grid = (x_elements, y_elements)
            elif grid_type == 2:
                grid = lambda meta: (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )
            elif grid_type == 3:
                grid = grid_fn

            # 调用自动调优后的二维加法内核
            add_kernel_2d_autotuned[grid](x, y, output, x_elements, y_elements)
            return output

        # 创建两个随机张量 t1 和 t2，并设置在 GPU 上计算，支持梯度计算
        t1 = torch.rand((512, 256), device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand((512, 256), device=GPU_TYPE, requires_grad=grad)
        # 创建一个和 t1 相同尺寸的全零张量 output，并设置支持梯度计算
        output = torch.zeros_like(t1, requires_grad=grad)

        # 调用 call_triton 函数处理 t1 和 t2，并得到 torch_result
        torch_result = call_triton(t1, t2, output)
        
        # 编译 call_triton 函数，根据给定的 backend、fullgraph 和 dynamic 参数进行编译
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )
        
        # 创建一个和 t1 相同尺寸的全零张量 output2，并设置支持梯度计算
        output2 = torch.zeros_like(t1, requires_grad=grad)
        
        # 断言编译后的函数调用结果与未编译的结果一致
        self.assertEqual(compiled_func(t1, t2, output2), torch_result)

    # 使用装饰器要求测试需要 GPU 支持
    @requires_gpu
    # 使用参数化装饰器设置多个参数组合
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    # 用 patch.object 修改 torch._inductor.config 的 implicit_fallbacks 属性为 False
    @patch.object(torch._inductor.config, "implicit_fallbacks", False)
    def test_triton_kernel_native(self, grad, dynamic, backend):
        # 定义一个内部函数 call_triton_add，用于调用 Triton 加法内核
        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            output: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
        ):
            # 计算输出张量的元素总数
            n_elements = output.numel()

            # 定义一个内部函数 grid_fn，用于生成网格参数
            def grid_fn(meta):
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            # 根据 grid_type 设置不同的网格参数
            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                # 当 grid_type 为 1 时，使用 lambda 函数计算网格参数
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            else:
                grid = grid_fn

            # 根据 positional 参数调用 Triton 加法内核
            if positional:
                add_kernel[grid](x, y, output, n_elements, 16)
            else:
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            return output

        # 创建随机张量 t1、t2，并使用 GPU_TYPE 设备
        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        # 创建一个与 t1 相同形状的全零张量 o1
        o1 = torch.zeros_like(t1, requires_grad=grad)

        # 计算标准的 PyTorch 加法结果
        torch_add = t1 + t2

        # 测试调用 Triton 加法内核是否得到预期结果，无 Dynamo 模式，使用 BLOCK_SIZE=16
        self.assertEqual(call_triton_add(t1, t2, o1, 1), torch_add)
        # 测试调用 Triton 加法内核是否得到预期结果，无 Dynamo 模式，使用 positional 参数和 BLOCK_SIZE=16
        o2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(call_triton_add(t1, t2, o2, 1, True), torch_add)

        # 使用 Dynamo 编译函数 call_triton_add，指定后端和动态图模式
        compiled_func = torch.compile(
            call_triton_add, backend=backend, fullgraph=True, dynamic=dynamic
        )
        # 测试编译函数后是否得到预期结果，使用简单的内核模式，无 positional 参数
        o3 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o3, 0), torch_add)
        # 测试编译函数后是否得到预期结果，使用 lambda 函数内核模式，无 positional 参数
        o4 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o4, 1), torch_add)
        # 测试编译函数后是否得到预期结果，使用 lambda 函数内核模式，带有 positional 参数和 BLOCK_SIZE=16
        o5 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o5, 1, 1, True), torch_add)
        # 测试编译函数后是否得到预期结果，使用用户定义函数内核模式，带有不同的参数
        o6 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o6, 2, 200), torch_add)

    @requires_gpu
    # 测试 Triton 内核变异时不标记为脏数据的情况
    def test_triton_kernel_mutation_not_mark_dirty(self):
        @torch.compile
        # 定义一个编译函数 f，用于调用 Triton 加法内核，并不标记输出张量为脏数据
        def f(x):
            n_elements = x.numel()
            add_kernel[(n_elements,)](x, x, x, n_elements, 16)
            return x

        # 创建一个在 GPU_TYPE 设备上的随机张量 x，并设置 requires_grad=True
        x = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        # 克隆张量 x，并对其执行 sin 操作
        x_cloned = x.clone()
        out = x_cloned.sin()
        # 调用函数 f 处理 x_cloned，并计算梯度
        f(x_cloned)
        out.sum().backward()

    @requires_cuda
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    def test_triton_kernel_inputs_buffer_reuse(self):
        # 定义一个内部函数 `_mul2`，用于将输入张量 `x` 复制到新的张量 `y` 中，并调用 `mul2_kernel` 函数处理 `x` 的数据
        def _mul2(x):
            y = torch.empty_like(x)
            mul2_kernel[(10,)](
                in_ptr0=x,
                out_ptr=y,
                n_elements=x.numel(),
                BLOCK_SIZE=1,
            )
            return y

        # 定义一个装饰器函数 `f`，接收一个输入张量 `x`，循环4次调用 `_mul2` 函数，并返回处理后的张量加上1
        @torch.compile
        def f(x):
            for _ in range(4):
                # 一个内核的输出是下一个内核的输入，但在某些时候应该重用缓冲区而不是分配新的
                x = _mul2(x)
            return x + 1

        # 生成一个随机张量 `x`，在 CUDA 设备上，数据类型为浮点数
        x = torch.randn(10, device="cuda", dtype=torch.float32)
        # 调用函数 `f`，记录执行结果到 `eager_out`
        eager_out = f(x)
        # 编译函数 `f`，并运行获取编译后的代码和结果
        compiled_out, (code,) = run_and_get_code(torch.compile(f), x)
        # 使用断言检查编译结果与直接执行结果是否相等
        self.assertEqual(compiled_out, eager_out)

        # 检查是否分配了最少数量的缓冲区
        num_bufs_allocated = code.count(
            "empty_strided_cuda((10, ), (1, ), torch.float32)"
        )
        self.assertEqual(num_bufs_allocated, 2)

        # 检查是否在不分配时重用了缓冲区
        num_bufs_reused = code.count("# reuse")
        self.assertEqual(num_bufs_reused, 3)

    @requires_gpu
    def test_triton_kernel_matmul_tracking(self):
        # 定义一个装饰器函数 `f`，接收一个输入张量 `x`，在 CUDA 设备上执行矩阵乘法和加法
        @torch.compile
        def f(x):
            out = torch.zeros_like(x)
            ones_kernel[(4,)](out, 16, BLOCK_SIZE=16)
            return torch.mm(out, x) + 10

        # 生成一个随机张量 `x`，形状为 (4, 4)，在 GPU 类型设备上
        x = torch.randn(4, 4, device=GPU_TYPE)
        # 调用函数 `f`，记录执行结果到 `torch_out`
        torch_out = f(x)
        # 生成全1张量与 `x` 的乘积，并加上10，记录结果到 `python_out`
        python_out = torch.mm(torch.ones(4, 4, device=GPU_TYPE), x) + 10
        # 使用断言检查 `torch_out` 是否等于 `python_out`
        self.assertEqual(torch_out, python_out)

    @requires_gpu
    def test_triton_kernel_strided_input(self):
        # 定义一个函数 `f`，接收一个输入张量 `inp`，对其进行切分和处理，并返回处理后的结果张量
        def f(inp):
            # 将输入张量 `inp` 按照指定维度切分为 `left` 和 `right` 两部分
            left, right = torch.split(inp, [128, 128], dim=1)
            # 根据 `left` 的形状创建一个空张量 `out`
            out = torch.empty_like(left)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            # 根据给定的参数调用 `double_strided_kernel` 函数处理 `left` 和 `out` 的数据
            grid = (left.size(1) // X_BLOCK_SIZE, left.size(0) // Y_BLOCK_SIZE)
            double_strided_kernel[grid](
                in_ptr=left,
                out_ptr=out,
                in_y_stride=left.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out

        # 生成一个随机张量 `inp`，形状为 (64, 256)，在 GPU 类型设备上
        inp = torch.randn(64, 256, device=GPU_TYPE)
        # 调用函数 `f`，记录执行结果到 `eager_out`
        eager_out = f(inp)
        # 编译函数 `f`，并运行获取编译后的结果
        compiled_out = torch.compile(f)(inp)
        # 使用断言检查编译结果与直接执行结果是否相等
        self.assertEqual(compiled_out, eager_out)
    def test_triton_kernel_strided_input_nonzero_offset(self):
        def f(inp):
            # 切分输入张量 inp，使得 left 的 strides 为 [256, 1]，存储偏移为 128
            left, right = torch.split(inp, [128, 128], dim=1)
            # 创建一个与 right 张量同样大小的空张量 out
            out = torch.empty_like(right)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            # 计算网格的大小，以便于并行化处理
            grid = (right.size(1) // X_BLOCK_SIZE, right.size(0) // Y_BLOCK_SIZE)
            # 调用 double_strided_kernel 函数处理 right 张量的数据
            double_strided_kernel[grid](
                in_ptr=right,
                out_ptr=out,
                in_y_stride=right.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out

        inp = torch.randn(64, 256, device=GPU_TYPE)

        # 使用函数 f 处理输入 inp，得到 eager_out 结果
        eager_out = f(inp)
        # 编译并使用函数 f 处理输入 inp，得到 compiled_out 结果
        compiled_out = torch.compile(f)(inp)
        # 断言编译结果与非编译结果相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_slice_and_view_input(self):
        def f(inp):
            # 从 inp 中切分得到 left 张量，其 strides 为 [256, 1]
            left = inp[:, :128]
            # 将 left 重塑为大小为 [64, 4, 32] 的张量
            left = left.view(64, 4, 32)
            # 创建一个与 left 张量同样大小的空张量 out
            out = torch.empty_like(left)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            # 计算网格的大小，以便于并行化处理
            grid = (
                (left.size(1) * left.size(2)) // X_BLOCK_SIZE,
                left.size(0) // Y_BLOCK_SIZE,
            )
            # 调用 double_strided_kernel 函数处理 left 张量的数据
            double_strided_kernel[grid](
                in_ptr=left,
                out_ptr=out,
                in_y_stride=left.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out + left

        inp = torch.randn(64, 256, device=GPU_TYPE)

        # 使用函数 f 处理输入 inp，得到 eager_out 结果
        eager_out = f(inp)
        # 编译并使用函数 f 处理输入 inp，得到 compiled_out 结果
        compiled_out = torch.compile(f)(inp)
        # 断言编译结果与非编译结果相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_fallback(self):
        def f(x, y):
            # 创建与 x 相同大小的全零张量 out 和 out2
            out = torch.zeros_like(x)
            out2 = torch.zeros_like(x)
            # 使用 add_kernel 处理 x 和 torch.mm(x, y) 的结果，并将结果存储到 out 中
            add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
            # 使用 add_kernel 处理 x 和 torch.sort(y).values 的结果，并将结果存储到 out 中
            add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
            return out, out2

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        # 使用函数 f 处理输入 x 和 y，得到 eager_out 结果
        eager_out = f(x, y)
        # 编译并使用函数 f 处理输入 x 和 y，得到 compiled_out 结果
        compiled_out = torch.compile(f)(x, y)
        # 断言编译结果与非编译结果相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_out_of_order(self):
        # 定义一个测试函数，用于测试 Triton 内核的乱序执行情况
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            BLOCK_SIZE: "tl.constexpr",
            out_ptr,
            n_elements,
        ):
            # 获取当前程序实例的 ID，设定轴向为0
            pid = tl.program_id(axis=0)
            # 计算当前块的起始位置
            block_start = pid * BLOCK_SIZE
            # 计算所有偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码以过滤超出有效元素范围的偏移量
            mask = offsets < n_elements
            # 从输入指针0处加载数据，应用掩码
            x = tl.load(in_ptr0 + offsets, mask=mask)
            # 从输入指针1处加载数据，应用掩码
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算输出结果
            output = x + y
            # 将计算结果存储到输出指针处，应用掩码
            tl.store(out_ptr + offsets, output, mask=mask)

        # 定义一个辅助函数，执行元素级别的加法操作
        def f(x, y):
            # 创建与输入张量x相同大小的零张量
            out = torch.zeros_like(x)
            # 获取张量x的元素数量
            n_elements = x.numel()
            # 调用 Triton 内核，进行元素级别的加法操作
            add_kernel[(n_elements,)](x, y, 4, out, n_elements)
            # 返回输出结果
            return out

        # 在 GPU 上生成随机张量x和y
        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        # 使用eager模式执行函数f，获取输出结果
        eager_out = f(x, y)
        # 使用编译后的方式执行函数f，获取输出结果
        compiled_out = torch.compile(f)(x, y)
        # 断言编译后的输出结果与eager模式的输出结果相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_unbacked_shape_tensor(self, backend):
        # 定义一个测试函数，用于测试 Triton 内核与无后端形状张量的交互
        @triton.jit
        def square(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前程序实例的 ID，设定轴向为0
            pid = tl.program_id(axis=0)
            # 计算当前块的起始位置
            block_start = pid * BLOCK_SIZE
            # 计算所有偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码以过滤超出有效元素范围的偏移量
            mask = offsets < n_elements
            # 从输入指针处加载数据，应用掩码
            x = tl.load(in_ptr + offsets, mask=mask)
            # 计算平方的输出
            output = x * x
            # 将计算结果存储到输出指针处，应用掩码
            tl.store(out_ptr + offsets, output, mask=mask)

        # 定义一个辅助函数，计算输入张量大于2的元素的平方
        def f(x):
            # 选取张量中大于2的元素
            x = x[x > 2]
            # 获取张量x的元素数量
            n_elements = x.numel()
            # 创建与张量x相同大小的零张量作为输出
            output = torch.zeros_like(x)
            # 定义一个函数，计算并存储元素的平方
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            square[grid](x, output, n_elements, BLOCK_SIZE=16)
            # 返回输出结果
            return output

        # 在 GPU 上生成一个随机张量x
        x = torch.randn(4, device=GPU_TYPE)
        # 使用eager模式执行函数f，获取输出结果
        eager_out = f(x)
        # 使用编译后的方式执行函数f，获取输出结果
        compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
        # 断言编译后的输出结果与eager模式的输出结果相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_arg(self, dynamic):
        # 定义一个使用 Triton 编译的内核函数，计算两个输入张量的和
        @triton.jit
        def add_kernel_half_n_elements(
            in_ptr0,
            in_ptr1,
            out_ptr,
            half_n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前线程块的程序 ID
            pid = tl.program_id(axis=0)
            # 计算当前线程块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成当前线程块内的偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，用于处理超出半个元素数量的数据
            mask = offsets < half_n_elements * 2
            # 从输入指针中加载数据到张量 x 和 y
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算 x 和 y 的和，存储到 output 中
            output = x + y
            # 将结果存储到输出指针中
            tl.store(out_ptr + offsets, output, mask=mask)

        # 定义一个函数 f，调用 Triton 内核函数，并返回结果张量
        def f(x, y):
            # 根据输入张量 x 创建一个同样类型的空张量 out
            out = torch.empty_like(x)
            # 计算半个元素数量
            half_n_elements = x.numel() // 2
            # 调用 Triton 内核函数处理输入张量 x 和 y，并将结果存储到 out 中
            add_kernel_half_n_elements[(half_n_elements,)](
                x, y, out, half_n_elements, BLOCK_SIZE=16
            )
            return out

        # 创建两个随机张量 x 和 y，位于 GPU 上
        x = torch.randn(2, device=GPU_TYPE)
        y = torch.randn(2, device=GPU_TYPE)
        # 使用函数 f 获取结果 eager_out
        eager_out = f(x, y)
        # 编译函数 f 并获取编译后的结果 compiled_out 和相关源代码 sources
        compiled_out, sources = run_and_get_code(
            torch.compile(f, dynamic=dynamic), x, y
        )

        # 如果 dynamic 为 True，检查源代码中是否存在特定字符串，否则检查另一个字符串
        if dynamic:
            self.assertTrue("equal_to_1=()" in sources[0])
        else:
            self.assertTrue("equal_to_1=(3,)" in sources[0])
        # 断言编译后的结果与 eager_out 相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_float_arg(self, dynamic):
        # 定义一个函数 f，调用 Triton 内核函数，对输入张量进行操作，并返回结果张量
        def f(x, y):
            # 根据输入张量 x 创建一个同类型的空张量 out
            out = torch.empty_like(x)
            # 获取输入张量的元素数量
            n_elements = x.numel()
            # 计算缩放因子
            scaling_factor = (n_elements**0) / 1.0
            # 调用 Triton 内核函数处理输入张量 x 和 y，应用缩放因子，并将结果存储到 out 中
            add_kernel_with_scaling[(n_elements,)](
                x,
                y,
                out,
                n_elements,
                scaling_factor,
                BLOCK_SIZE=16,
            )
            return out

        # 创建两个随机张量 x 和 y，位于 GPU 上
        x = torch.randn(2, device=GPU_TYPE)
        y = torch.randn(2, device=GPU_TYPE)
        # 使用函数 f 获取结果 eager_out
        eager_out = f(x, y)
        # 编译函数 f 并获取编译后的结果 compiled_out 和相关源代码 sources
        compiled_out, sources = run_and_get_code(
            torch.compile(f, dynamic=dynamic), x, y
        )

        # 检查源代码中是否存在特定字符串，表明等于 1 的特例化不应添加 1.0
        self.assertTrue("equal_to_1=()" in sources[0])
        # 断言编译后的结果与 eager_out 相等
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @skipIfRocm
    # 定义一个测试函数，用于测试 Triton 内核与导入的符号一起使用的情况
    def test_triton_kernel_with_imported_symbol(self):
        # 使用 Triton 的 jit 装饰器定义一个内核函数，该函数实现了一个加法内核，涉及导入的符号
        @triton.jit
        def add_kernel_with_imported_symbol(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前线程的程序 ID
            pid = tl.program_id(axis=0)
            # 计算当前线程的块起始位置
            block_start = pid * BLOCK_SIZE
            # 创建一个偏移量数组，表示当前线程需要处理的数据块范围
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，指示哪些偏移量是有效的（小于 n_elements）
            mask = offsets < n_elements
            # 从输入指针处加载数据到 x 变量中
            x = tl.load(in_ptr + offsets, mask=mask)
            # 调用 fast_dividef 函数对 x 中的数据进行处理
            output = fast_dividef(x, 3.14)
            # 将处理后的数据存储到输出指针处
            tl.store(out_ptr + offsets, output, mask=mask)

        # 定义一个辅助函数 f，接受一个输入 x，创建一个与 x 类型相同的空输出 out
        def f(x):
            out = torch.empty_like(x)
            # 获取 x 中元素的数量作为 n_elements
            n_elements = x.numel()
            # 调用 Triton 内核函数 add_kernel_with_imported_symbol，处理输入 x，并将结果存储到 out 中
            add_kernel_with_imported_symbol[(n_elements,)](
                x, out, n_elements, BLOCK_SIZE=16
            )
            # 返回处理后的输出 out
            return out

        # 创建一个随机张量 x，并指定在 GPU 上进行计算
        x = torch.randn(4, device=GPU_TYPE)
        # 分别使用普通模式（eager mode）和编译后模式（compiled mode）运行函数 f，并比较结果
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)

        # 断言编译后模式的输出与普通模式的输出相等
        self.assertEqual(compiled_out, eager_out)

    # 使用装饰器定义一个测试函数，测试 Triton 内核与导入的符号以及自定义名称的情况
    @requires_gpu
    @skipIfRocm
    def test_triton_kernel_with_imported_symbol_with_custom_name(self):
        # 使用 Triton 的 jit 装饰器定义一个内核函数，该函数实现了一个加法内核，涉及导入的符号和自定义函数名
        @triton.jit
        def add_kernel_with_imported_symbol(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前线程的程序 ID
            pid = tl.program_id(axis=0)
            # 计算当前线程的块起始位置
            block_start = pid * BLOCK_SIZE
            # 创建一个偏移量数组，表示当前线程需要处理的数据块范围
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，指示哪些偏移量是有效的（小于 n_elements）
            mask = offsets < n_elements
            # 从输入指针处加载数据到 x 变量中
            x = tl.load(in_ptr + offsets, mask=mask)
            # 调用自定义的 fast_dividef 函数对 x 中的数据进行处理
            output = my_fast_dividef(x, 3.14)
            # 将处理后的数据存储到输出指针处
            tl.store(out_ptr + offsets, output, mask=mask)

        # 定义一个辅助函数 f，接受一个输入 x，创建一个与 x 类型相同的空输出 out
        def f(x):
            out = torch.empty_like(x)
            # 获取 x 中元素的数量作为 n_elements
            n_elements = x.numel()
            # 调用 Triton 内核函数 add_kernel_with_imported_symbol，处理输入 x，并将结果存储到 out 中
            add_kernel_with_imported_symbol[(n_elements,)](
                x, out, n_elements, BLOCK_SIZE=16
            )
            # 返回处理后的输出 out
            return out

        # 创建一个随机张量 x，并指定在 GPU 上进行计算
        x = torch.randn(4, device=GPU_TYPE)
        # 分别使用普通模式（eager mode）和编译后模式（compiled mode）运行函数 f，并比较结果
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)

        # 断言编译后模式的输出与普通模式的输出相等
        self.assertEqual(compiled_out, eager_out)

    # 使用装饰器定义一个测试函数，测试 Triton 内核与导入的符号和自定义名称，同时参数化测试不同的大小和动态设置
    @requires_gpu
    @common_utils.parametrize("size", [4, 16])
    @common_utils.parametrize("dynamic", [False, True])
    # 定义一个测试函数，用于测试 Triton 内核在不同形状下的行为
    def test_triton_kernel_different_shapes(self, size, dynamic):
        # 导入运行和获取代码的工具函数
        from torch._inductor.utils import run_and_get_code
        
        # 定义一个内部函数 f，接收四个张量参数，并返回两个结果张量
        def f(x, y, xx, yy):
            # 计算张量 x 中的元素总数
            n_elements = x.numel()
            # 根据张量 x 的形状创建一个全零张量 output_1
            output_1 = torch.zeros_like(x)
            # 定义一个网格函数 lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用名为 add_kernel 的内核，并在 grid 中设置网格大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output_1, n_elements, BLOCK_SIZE=4)
    
            # 计算张量 xx 中的元素总数
            n_elements = xx.numel()
            # 根据张量 xx 的形状创建一个全零张量 output_2
            output_2 = torch.zeros_like(xx)
            # 定义一个网格函数 lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用名为 add_kernel 的内核，并在 grid 中设置网格大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](xx, yy, output_2, n_elements, BLOCK_SIZE=4)
    
            # 返回两个计算结果张量 output_1 和 output_2
            return output_1, output_2
    
        # 使用 torch.rand 在 GPU 上创建指定大小的随机张量 x, y, xx, yy
        x = torch.rand(size, device=GPU_TYPE)
        y = torch.rand(size, device=GPU_TYPE)
        xx = torch.rand(size, size, device=GPU_TYPE)
        yy = torch.rand(size, size, device=GPU_TYPE)
        # 将这些张量放入列表 args 中
        args = [x, y, xx, yy]
    
        # 调用函数 f，获取其返回值 eager_out
        eager_out = f(*args)
        # 使用 torch.compile 编译函数 f，并运行获取编译后的代码和其他信息
        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, fullgraph=True, dynamic=dynamic, backend="inductor"), *args
        )
        # 根据 size 和 dynamic 的值进行断言测试
        if size == 4 and not dynamic:
            # 如果 size 等于 4 且 dynamic 为 False，则预期生成两个内核
            self.assertTrue("add_kernel_0.run" in code)
            self.assertTrue("add_kernel_1.run" in code)
        else:
            # 如果 size 等于 16 或 dynamic 为 True，则预期只生成一个内核
            self.assertTrue("add_kernel_0.run" in code)
            self.assertTrue("add_kernel_1.run" not in code)
    
        # 断言编译后的结果与直接运行的结果相等
        self.assertEqual(compiled_out, eager_out)
    
    # 带 GPU 装饰器的测试函数，测试 Triton 内核在重置为零时的行为
    @requires_gpu
    def test_triton_kernel_reset_to_zero(self):
        # 定义自动调优函数的装饰器，并指定多个配置
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
            ],
            key=["n_elements"],
            reset_to_zero=["out_ptr"],
        )
        # 使用 Triton 的即时编译装饰器定义函数 add_kernel_autotuned_reset
        @triton.jit
        def add_kernel_autotuned_reset(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前程序 ID
            pid = tl.program_id(axis=0)
            # 计算每个块的起始位置
            block_start = pid * BLOCK_SIZE
            # 计算偏移量范围
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码以确保在有效范围内
            mask = offsets < n_elements
            # 从输入指针加载数据到 x 和 y，使用 mask 进行条件加载
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算输出并存储到输出指针中，使用 mask 进行条件存储
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
    
        # 使用 torch.compile 定义函数 f，编译时生成全图并优化
        @torch.compile(fullgraph=True)
        def f(x, y):
            # 根据输入张量 x 的形状创建一个全零张量 output
            output = torch.zeros_like(x)
            # 计算输出张量中的元素总数
            n_elements = output.numel()
            # 定义一个网格函数 lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用名为 add_kernel_autotuned_reset 的自动调优内核，并在 grid 中设置网格大小
            add_kernel_autotuned_reset[grid](x, y, output, n_elements)
            # 返回计算结果张量 output
            return output
    
        # 使用 torch.randn 在 GPU 上创建形状为 (4,) 的随机张量 x
        x = torch.randn(4, device=GPU_TYPE)
        # 设置错误消息字符串
        msg = "Only configs and keys are supported for triton.autotune"
        # 使用 self.assertRaisesRegex 断言捕获到异常，且错误消息符合预期
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            # 调用函数 f，传入参数 x, x，触发异常捕获
            f(x, x)
    # 要求使用 GPU 运行此测试
    @requires_gpu
    # 参数化装饰器，为 "dynamic" 参数分别传入 False 和 True
    @common_utils.parametrize("dynamic", [False, True])
    # 参数化装饰器，为 "backend" 参数分别传入 "eager", "aot_eager", "inductor"
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    # 定义 Triton 核心和 Triton 数据类型的测试方法
    def test_triton_kernel_triton_dtype(self, dynamic, backend):
        # Triton JIT 编译的函数，接受多个参数：
        @triton.jit
        def add_kernel_with_dtype(
            in_ptr0,
            in_ptr1,
            out_ptr,
            dtype: "tl.constexpr",  # 数据类型标注为 Triton 的常量表达式
            n_elements,  # 元素个数
            BLOCK_SIZE: "tl.constexpr",  # 块大小，标注为 Triton 的常量表达式
        ):
            # 获取当前程序实例的 ID（程序在第一个轴上的位置）
            pid = tl.program_id(axis=0)
            # 计算当前块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成当前块内的偏移量数组
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码，确保偏移量不超过元素总数
            mask = offsets < n_elements
            # 从输入地址加载数据，转换为指定的数据类型 dtype
            x = tl.load(in_ptr0 + offsets, mask=mask).to(dtype)
            y = tl.load(in_ptr1 + offsets, mask=mask).to(dtype)
            # 执行加法操作
            output = x + y
            # 将结果存储到输出地址中，使用与输入相同的掩码
            tl.store(out_ptr + offsets, output, mask=mask)

        # 辅助函数 f，接受两种数据类型作为参数，返回输出张量
        def f(x, y, dtype_torch, dtype_triton):
            # 创建与 x 相同形状的零张量，并转换为 dtype_torch 指定的数据类型
            output = torch.zeros_like(x).to(dtype=dtype_torch)
            # 获取输出张量的元素总数
            n_elements = output.numel()
            # 定义计算网格函数，返回计算网格大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用 Triton JIT 编译的核函数，使用 grid 计算网格大小
            add_kernel_with_dtype[grid](
                x, y, output, dtype_triton, n_elements, BLOCK_SIZE=4
            )
            # 返回输出张量
            return output

        # 创建一个随机张量 x 和 y，使用 GPU_TYPE 指定的设备
        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        # 参数列表，每个参数包含 x, y, torch 数据类型和 Triton 数据类型
        args_list = (
            [x, y, torch.float32, tl.float32],
            [x, y, torch.bfloat16, tl.bfloat16],
        )
        # 对参数列表进行迭代
        for args in args_list:
            # 调用函数 f，获取 eager_out
            eager_out = f(*args)
            # 编译函数 f，使用指定的 backend 和 dynamic 参数
            compiled_out = torch.compile(
                f, fullgraph=True, backend=backend, dynamic=dynamic
            )(*args)
            # 断言编译后的输出与 eager_out 相等
            self.assertEqual(compiled_out, eager_out)

    # 要求使用 GPU 运行此测试
    @requires_gpu
    # 参数化装饰器，为 "backend" 参数分别传入 "eager", "aot_eager", "inductor"
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])


    # 要求 GPU 环境，并使用 common_utils 的参数化装饰器，测试不同的后端
    def test_triton_kernel_num_ctas(self, backend):


        @triton.jit
        def kernel(X):
            return


        # 使用 Triton 的 JIT 编译装饰器定义一个简单的核函数
        @torch.compile(backend=backend)


        # 定义一个函数，使用 Torch 的编译装饰器，根据指定后端编译
        def f(x):


            # 调用 Triton 编译的核函数，指定一个线程块，不进行预热
            kernel[(1,)](x, num_ctas=1)


            # 运行 Triton 编译的核函数，指定线程块数为 1，不进行预热
            kernel.run(x, num_ctas=1, grid=(1,), warmup=False)


            # 返回输入的张量 x
            return x


        x = torch.randn(4, device=GPU_TYPE)


        # 调用函数 f，传入随机生成的张量 x
        f(x)
    # 定义一个测试函数，用于测试没有自动调优的 Triton 内核特殊关键字参数
    def test_triton_kernel_special_kwargs_without_autotune(self, backend):
        # 定义一个 Triton JIT 编译的内核函数 add_kernel
        @triton.jit
        def add_kernel(
            in_ptr0,            # 输入指针0，用于加载数据
            in_ptr1,            # 输入指针1，用于加载数据
            out_ptr,            # 输出指针，用于存储结果数据
            n_elements,         # 元素总数，表示操作的总元素个数
            BLOCK_SIZE: "tl.constexpr",  # 块大小，使用 Triton 的常量表达式注解
        ):
            # 获取当前程序的 ID，沿着第0轴
            pid = tl.program_id(axis=0)
            # 计算当前块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成偏移量数组，范围从 block_start 到 block_start + BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，用于保护超出 n_elements 范围的偏移量
            mask = offsets < n_elements
            # 从 in_ptr0 加载数据到 x，仅在掩码内操作
            x = tl.load(in_ptr0 + offsets, mask=mask)
            # 从 in_ptr1 加载数据到 y，仅在掩码内操作
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算 x 和 y 的和，结果存储在 output 中
            output = x + y
            # 将 output 存储到 out_ptr 中，仅在掩码内操作
            tl.store(out_ptr + offsets, output, mask=mask)

        # 使用 Torch 的编译器装饰器，生成一个编译函数 f
        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            # 创建一个与 x 相同大小的全零张量 output
            output = torch.zeros_like(x)
            # 获取 output 的元素总数
            n_elements = output.numel()
            # 定义一个网格函数 grid，用于指定 Triton 的网格大小
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用 Triton 内核函数 add_kernel，并传入相关参数
            add_kernel[grid](
                x,              # 输入张量 x
                y,              # 输入张量 y
                output,         # 输出张量 output
                n_elements,     # 元素总数
                BLOCK_SIZE=128, # 块大小设置为 128
                num_warps=8,    # Triton 内核特殊关键字参数：线程束数
                num_stages=3,   # Triton 内核特殊关键字参数：阶段数
            )
            # 返回计算后的输出张量 output
            return output

        # 生成一个随机数张量 x，使用指定的 GPU 类型 GPU_TYPE
        x = torch.randn(4, device=GPU_TYPE)
        # 调用函数 f，传入 x 和相同的输入 y，计算并返回结果
        f(x, x)
# 定义一个装饰器函数，用于创建基于 GPU 的变异测试函数
def make_mutation_test(fn):
    # 实际生成的测试函数，使用装饰器 @requires_gpu 包装
    @requires_gpu
    def test_fn(self):
        # 导入识别变异张量的函数
        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors
        
        # 调用给定函数 fn，获取返回的 kernel, inputs, outputs
        kernel, inputs, outputs = fn()
        # 断言识别出的变异张量与期望的输出一致
        self.assertListEqual(
            identify_mutated_tensors(kernel, inputs),
            outputs,
        )
    
    # 返回生成的测试函数
    return test_fn


# Triton 代码生成器存在作用域问题。
# 在此定义辅助函数
if HAS_GPU:
    
    # 定义 triton.jit 装饰的 helper_id 函数，用于返回参数 p
    @triton.jit
    def helper_id(p):
        return p

    # 定义 triton.jit 装饰的 helper_add_and_out 函数，用于计算 x + y 并返回结果和 out_ptr
    @triton.jit
    def helper_add_and_out(x, y, out_ptr):
        return x + y, out_ptr


# 定义 MutationTests 类，继承自 torch._inductor.test_case.TestCase
class MutationTests(torch._inductor.test_case.TestCase):
    # 下面是注入的测试函数

    # 使用 make_mutation_test 装饰器生成测试函数 test_out_of_order_kernel
    @make_mutation_test
    def test_out_of_order_kernel():
        # 定义 triton.jit 装饰的 add_kernel_out_of_order 函数
        @triton.jit
        def add_kernel_out_of_order(
            in_ptr0,
            n_elements,
            in_ptr1,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取程序块的 ID
            pid = tl.program_id(axis=0)
            # 计算程序块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成偏移量数组
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码以处理有效偏移量
            mask = offsets < n_elements
            # 从 in_ptr0 和 in_ptr1 加载数据，使用掩码
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算输出数据
            output = x + y
            # 将结果存储到 out_ptr 中，使用掩码
            tl.store(out_ptr + offsets, output, mask=mask)

        # 生成一个大小为 4 的随机张量 t
        t = torch.randn(4)
        # 返回生成的函数 add_kernel_out_of_order，输入参数字典，和期望的输出列表
        return (
            add_kernel_out_of_order,
            {
                "in_ptr0": t,
                "n_elements": 4,
                "in_ptr1": t,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    # 使用 make_mutation_test 装饰器生成测试函数 test_out_of_order_kernel_call
    @make_mutation_test
    def test_out_of_order_kernel_call():
        # 定义 triton.jit 装饰的 add_kernel_out_of_order_fn1 函数
        @triton.jit
        def add_kernel_out_of_order_fn1(
            in_ptr0,
            n_elements,
            in_ptr1,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取程序块的 ID
            pid = tl.program_id(axis=0)
            # 计算程序块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成偏移量数组
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码以处理有效偏移量
            mask = offsets < n_elements
            # 调用 add_kernel_out_of_order_fn2 函数
            add_kernel_out_of_order_fn2(
                in_ptr0, in_ptr1, n_elements, out_ptr, BLOCK_SIZE=BLOCK_SIZE
            )

        # 生成一个大小为 4 的随机张量 t
        t = torch.randn(4)
        # 返回生成的函数 add_kernel_out_of_order_fn1，输入参数字典，和期望的输出列表
        return (
            add_kernel_out_of_order_fn1,
            {
                "in_ptr0": t,
                "n_elements": 4,
                "in_ptr1": t,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )
    # 定义一个测试函数，用于测试 reduce_sum_kernel 函数
    def test_reduce_sum():
        # 使用 Triton 的 jit 装饰器，将下面的函数编译为 Triton 可执行的代码
        @triton.jit
        def reduce_sum_kernel(a_ptr, c_ptr, stride_am, stride_an):
            # 创建一个包含 [0, 1, 2, 3] 的张量作为偏移 am
            offs_am = tl.arange(0, 4)
            # 创建一个包含 [0, 1, 2, 3] 的张量作为偏移 an
            offs_an = tl.arange(0, 4)
            # 计算每个元素在内存中的位置并存储在 a_ptrs 中
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_an[None, :] * stride_an
            )
            # 从内存地址 a_ptrs 中加载数据到张量 a
            a = tl.load(a_ptrs)
            # 计算张量 a 沿着 axis=1 的和并存储在张量 m 中
            m = tl.sum(a, axis=1)
            # 将张量 m 的值存储回内存地址 c_ptr + [0, 1, 2, 3] 中
            tl.store(c_ptr + tl.arange(0, 4), m)

        # 创建一个包含四个随机数的张量 t
        t = torch.randn(4)
        # 将 reduce_sum_kernel 函数赋值给 kernel
        kernel = reduce_sum_kernel
        # 定义函数 reduce_sum_kernel 的参数 kwargs
        kwargs = {
            "a_ptr": t,
            "c_ptr": t,
            "stride_am": 4,
            "stride_an": 4,
        }

        # TODO(aakhundov): tt.reduce is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["c_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        # 生成 Triton IR 模块 ttir_module 和相关信息
        ttir_module, _ = generate_ttir(kernel, kwargs)
        if hasattr(ttir_module, "walk"):
            # 当具有 MLIR-based Triton analysis pass 时
            expected = ["c_ptr"]
        else:
            # 当使用 TTIR string parsing-based Triton analysis pass 时
            expected = ["a_ptr", "c_ptr"]

        # 返回 kernel 函数、kwargs 参数、以及预期的输出 expected
        return (
            kernel,
            kwargs,
            expected,
        )

    # 使用 make_mutation_test 装饰器定义测试函数 test_argmax
    @make_mutation_test
    def test_argmax():
        # 使用 Triton 的 jit 装饰器，将下面的函数编译为 Triton 可执行的代码
        @triton.jit
        def argmax_kernel(a_ptr, c_ptr, stride_am, stride_an):
            # 创建一个包含 [0, 1, 2, 3] 的张量作为偏移 am
            offs_am = tl.arange(0, 4)
            # 创建一个包含 [0, 1, 2, 3] 的张量作为偏移 an
            offs_an = tl.arange(0, 4)
            # 计算每个元素在内存中的位置并存储在 a_ptrs 中
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_an[None, :] * stride_an
            )
            # 从内存地址 a_ptrs 中加载数据到张量 a
            a = tl.load(a_ptrs)
            # 计算张量 a 沿着 axis=1 的最大值索引并存储在张量 m 中
            m = tl.argmax(a, axis=1)
            # 将张量 m 的值存储回内存地址 c_ptr + [0, 1, 2, 3] 中
            tl.store(c_ptr + tl.arange(0, 4), m)

        # 创建一个包含四个随机数的张量 t
        t = torch.randn(4)
        # 将 argmax_kernel 函数赋值给 kernel
        kernel = argmax_kernel
        # 定义函数 argmax_kernel 的参数 kwargs
        kwargs = {
            "a_ptr": t,
            "c_ptr": t,
            "stride_am": 4,
            "stride_an": 4,
        }

        # TODO(aakhundov): tt.reduce is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["c_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        # 生成 Triton IR 模块 ttir_module 和相关信息
        ttir_module, _ = generate_ttir(kernel, kwargs)
        if hasattr(ttir_module, "walk"):
            # 当具有 MLIR-based Triton analysis pass 时
            expected = ["c_ptr"]
        else:
            # 当使用 TTIR string parsing-based Triton analysis pass 时
            expected = ["a_ptr", "c_ptr"]

        # 返回 kernel 函数、kwargs 参数、以及预期的输出 expected
        return (
            kernel,
            kwargs,
            expected,
        )

    # 要求 CUDA 支持，并且不跳过 ROCm 平台的测试
    def test_triton_kernel_inference_mode(self):
        # 定义内部函数f，用于执行核函数的调用和验证输出
        def f(x, y, out):
            # 计算输入张量的元素数量
            n_elements = x.numel()
            # 定义网格函数grid，用于计算调用核函数时的网格尺寸
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            # 调用add_kernel核函数，传入参数x, y, out以及n_elements，并指定BLOCK_SIZE=4
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=4)

        # 进入Triton推断模式上下文
        with torch.inference_mode():
            # 创建在CUDA设备上全为1的张量x和y
            x = torch.ones(32, device="cuda")
            y = torch.ones(32, device="cuda")
            # 创建一个与x形状相同的全零张量out_ref和out_test
            out_ref = torch.zeros_like(x)
            out_test = torch.zeros_like(x)
            # 使用函数f计算out_ref
            f(x, y, out_ref)
            # 编译函数f，并用x, y, out_test作为参数调用
            torch.compile(f)(x, y, out_test)
            # 断言out_ref与out_test张量相等
            self.assertEqual(out_ref, out_test)

    @make_mutation_test
    def test_cumsum():
        # 定义内部函数cumsum_kernel，用于计算累积和的核函数
        @triton.jit
        def cumsum_kernel(in_ptr, out_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
            # 创建一个在0到RBLOCK范围内的行索引张量rindex
            rindex = tl.arange(0, RBLOCK)[None, :]
            # 创建一个在0到XBLOCK范围内的列索引张量xindex
            xindex = tl.arange(0, XBLOCK)[:, None]
            # 从in_ptr中加载数据，形成data张量
            data = tl.load(in_ptr + rindex)
            # 对data进行按行累积和操作，形成scan张量
            scan = tl.cumsum(data, 1)
            # 计算data每行元素的累积和，形成expected_max张量
            expected_max = tl.sum(data, 1)
            # 断言scan张量中的值小于等于expected_max中对应位置的值
            tl.device_assert(scan <= expected_max)
            # 将scan张量的内容存储到out_ptr指向的内存中，按照xindex和rindex的规定偏移量存储
            tl.store(out_ptr + xindex * RBLOCK + rindex, scan)

        # 创建一个形状为(4,)的随机张量t
        t = torch.randn(4)
        # 将cumsum_kernel函数赋值给kernel变量
        kernel = cumsum_kernel
        # 定义kwargs字典，包含了cumsum_kernel函数的所有参数
        kwargs = {
            "in_ptr": t,
            "out_ptr": t,
            "XBLOCK": 4,
            "RBLOCK": 16,
        }

        # 生成TTIR模块和预期结果列表expected
        ttir_module, _ = generate_ttir(kernel, kwargs)
        if hasattr(ttir_module, "walk"):
            # 当具有MLIR-based Triton分析通过时，expected为["out_ptr"]
            expected = ["out_ptr"]
        else:
            # 当采用TTIR字符串解析的Triton分析通过时，expected为["in_ptr", "out_ptr"]
            expected = ["in_ptr", "out_ptr"]

        return (
            kernel,
            kwargs,
            expected,
        )

    @make_mutation_test
    def test_fn_call_one_return():
        # 定义内部函数add_kernel_with_fn_call，执行带有函数调用的加法核函数
        @triton.jit
        def add_kernel_with_fn_call(
            in_ptr0,
            in_ptr1,
            n_elements,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前程序块的ID
            pid = tl.program_id(axis=0)
            # 计算当前块的起始偏移量
            block_start = pid * BLOCK_SIZE
            # 创建偏移量张量，指定块内的具体偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，用于标记偏移量是否小于n_elements
            mask = offsets < n_elements
            # 从in_ptr0和in_ptr1中加载数据，仅在mask为True时加载
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算x和y的和，形成output张量
            output = x + y
            # 调用helper_id函数，获取out_ptr的指针
            out = helper_id(out_ptr)
            # 将output存储到out指向的内存中，按照offsets和mask规定的位置存储
            tl.store(out + offsets, output, mask=mask)

        # 创建一个形状为(4,)的随机张量t
        t = torch.randn(4)
        # 返回add_kernel_with_fn_call函数、kwargs字典和预期结果列表["out_ptr"]
        return (
            add_kernel_with_fn_call,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "n_elements": 4,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )
    @make_mutation_test
    # 声明一个装饰器，用于指示测试框架对下面的函数进行变异测试
    def test_fn_call_multi_return():
        # 定义一个测试函数，测试函数会返回一个包含以下内容的元组：
        @triton.jit
        # 使用 Triton 编译器进行 JIT 编译，加速函数运行
        def add_kernel_with_fn_call(
            in_ptr0,
            in_ptr1,
            n_elements,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前程序实例的 ID（在指定轴上的）
            pid = tl.program_id(axis=0)
            # 计算当前程序实例处理的数据块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成当前数据块内的偏移量列表
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，标识有效的偏移量范围
            mask = offsets < n_elements
            # 从输入指针中加载数据到 x 和 y 中，仅使用掩码标记的数据
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 调用 helper_add_and_out 函数，计算输出和输出指针，并将结果保存到 output 和 out 中
            output, out = helper_add_and_out(x, y, out_ptr)
            # 将计算结果 output 存储到输出指针位置偏移量的位置上，仅使用掩码标记的位置
            tl.store(out + offsets, output, mask=mask)

        # 创建一个测试用的输入张量 t，包含随机生成的数据
        t = torch.randn(4)
        # 返回一个元组，包含 JIT 编译后的函数对象、输入参数字典和期望的输出结果列表
        return (
            add_kernel_with_fn_call,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "n_elements": 4,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    # 声明一个装饰器，用于指示测试框架对下面的函数进行变异测试
    def test_nested_cond_op_kernel():
        # 定义一个测试函数，测试函数会返回一个包含以下内容的元组：
        @triton.jit
        # 使用 Triton 编译器进行 JIT 编译，加速函数运行
        def nested_cond_op_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前程序实例的 ID（在指定轴上的）
            pid = tl.program_id(axis=0)
            # 计算当前程序实例处理的数据块的起始位置
            block_start = pid * BLOCK_SIZE
            # 生成当前数据块内的偏移量列表
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，标识有效的偏移量范围
            mask = offsets < n_elements
            # 从输入指针中加载数据到 x 和 y 中，仅使用掩码标记的数据
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 如果当前程序实例的第一个轴的 ID 为 0
            if tl.program_id(0) == 0:
                # 如果当前程序实例的第二个轴的 ID 为 0
                if tl.program_id(1) == 0:
                    # 计算 x 和 y 的和，保存到 output 中
                    output = x + y
                    # 将计算结果 output 存储到输出指针位置偏移量的位置上，仅使用掩码标记的位置
                    tl.store(out_ptr + offsets, output, mask=mask)
            else:
                # 如果不满足上述条件，则什么也不做
                pass

        # 创建一个测试用的输入张量 t，包含随机生成的数据
        t = torch.randn(4)
        # 返回一个元组，包含 JIT 编译后的函数对象、输入参数字典和期望的输出结果列表
        return (
            nested_cond_op_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )
    @make_mutation_test
    # 使用装饰器将函数标记为变异测试的目标函数
    def test_add_for_loop2():
        @triton.jit
        # 使用 Triton 编译器将函数编译为 JIT 内核
        def add_1_time_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前线程的程序 ID
            pid = tl.program_id(axis=0)
            # 计算当前块的起始位置
            block_start = pid * BLOCK_SIZE
            # 创建一个数组，包含从 block_start 到 block_start + BLOCK_SIZE 的偏移量
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建一个掩码，标记有效偏移量范围
            mask = offsets < n_elements
            # 从内存中加载数据到 x 和 y，仅在 mask 为 True 时加载
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 循环遍历块大小，这里没有实际操作，因为 i = tl.multiple_of(i, 1) 相当于无操作
            for i in range(0, BLOCK_SIZE):
                i = tl.multiple_of(i, 1)
            # 计算输出值，这里是 x 和 y 的加和
            output = x + y
            # 将计算结果存储回内存，仅在 mask 为 True 时存储
            tl.store(out_ptr + offsets, output, mask=mask)

        # 创建一个测试用的随机张量 t
        t = torch.randn(4)
        # 返回 JIT 内核函数、参数字典和输出标签列表
        return (
            add_1_time_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    # 使用装饰器将函数标记为变异测试的目标函数
    @make_mutation_test
    # 使用装饰器标记函数为变异测试的目标函数
    def test_add_nested_for_loop_multi_return():
        # 定义嵌套的 triton.jit 装饰的内核函数，用于向量化操作
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取当前程序线程的 ID
            pid = tl.program_id(axis=0)
            # 计算每个线程块的起始位置
            block_start = pid * BLOCK_SIZE
            # 计算线程块内的偏移量数组
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码以确保只处理有效元素
            mask = offsets < n_elements
            # 从内存中加载数据到 x 和 y 向量
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 创建两个全零数组作为输出的初始值
            output1 = tl.zeros((n_elements,), dtype=tl.float32)
            output2 = tl.zeros((n_elements,), dtype=tl.float32)
            # 嵌套循环，每个元素进行两次加法操作
            for i in range(2):
                for j in range(2):
                    output1 += y
                    output2 += x
            # 将两个输出数组相加得到最终结果
            output = output1 + output2
            # 将结果存储回内存中的 out_ptr 中，根据掩码进行存储
            tl.store(out_ptr + offsets, output, mask=mask)

        # 生成测试数据 t，这里用 torch 随机生成一个长度为 4 的张量
        t = torch.randn(4)
        # 返回内核函数、输入参数字典和输出列表
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )
    def test_labels():
        # 定义带标签的内核函数
        @triton.jit
        def kernel_with_label(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            # 获取程序的ID（在axis=0上）
            pid = tl.program_id(axis=0)
            # 如果程序ID大于1，则直接返回，不执行后续操作
            if pid > 1:
                return
            # 计算块的起始位置
            block_start = pid * BLOCK_SIZE
            # 计算偏移量数组
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # 创建掩码，用于处理超出元素数量的情况
            mask = offsets < n_elements
            # 从输入指针中加载数据x和y，根据掩码进行过滤
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            # 计算输出数据
            output = x + y
            # 将输出数据存储到指定的输出指针位置，根据掩码进行过滤
            tl.store(out_ptr + offsets, output, mask=mask)

        # 创建一个随机张量t，用作内核函数的输入
        t = torch.randn(4)
        # 返回定义的内核函数、输入参数字典和输出指针列表
        return (
            kernel_with_label,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_for_loop_arg():
        # 定义前向传播的内核函数
        @triton.jit
        def fwd_kernel(
            X_ptr,
            W1_ptr,
            b1_ptr,
            O_ptr,
            M: tl.constexpr,
            C1: tl.constexpr,
            C2: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_C2: tl.constexpr,
        ):
            # 获取程序ID（在axis=0上）
            pid_m = tl.program_id(0)

            # 计算偏移量
            offs_c1 = tl.arange(0, C1)
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

            # 加载输入数据
            x_block_ptr = X_ptr + offs_m[:, None] * C1 + offs_c1[None, :]
            x = tl.load(x_block_ptr)

            # 计算门控
            for c2 in range(0, tl.cdiv(C2, BLOCK_SIZE_C2)):
                # 计算块指针
                offs_c2 = c2 * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
                o_block_ptr = O_ptr + offs_m[:, None] * C2 + offs_c2[None, :]
                w1_block_ptr = W1_ptr + offs_c1[:, None] * C2 + offs_c2[None, :]
                b1_block_ptr = b1_ptr + offs_c2

                # 计算输出
                w = tl.load(w1_block_ptr)
                b = tl.load(b1_block_ptr)
                o = tl.dot(x, w, allow_tf32=False)
                o += b[None, :]

                # 存储输出
                tl.store(o_block_ptr, o)

        # 创建一个64维的随机张量t，用作内核函数的输入
        t = torch.randn(64)
        # 返回定义的前向传播内核函数、输入参数字典和输出指针列表
        return (
            fwd_kernel,
            {
                "X_ptr": t,
                "W1_ptr": t,
                "b1_ptr": t,
                "O_ptr": t,
                "M": 64,
                "C1": 64,
                "C2": 64,
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_C2": 64,
            },
            ["O_ptr"],
        )

    @make_mutation_test
    def test_for_loop_arg_2():
        @triton.jit
        def fwd_kernel(
            x_ptr,
            o_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_B: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # 获取程序 ID
            pid_m = tl.program_id(0)
            # 创建输入数据块的指针，用于访问输入数据
            X_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            # 创建输出数据块的指针，用于访问输出数据
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            # 使用 BLOCK_B 参数定义的块数量进行循环
            for _ in range(BLOCK_B):
                # 从输入数据块指针中加载数据 x
                x = tl.load(X_block_ptr)
                # 将数据 x 存储到输出数据块指针中
                tl.store(O_block_ptr, x)

                # 更新输入和输出数据块指针，以便处理下一个数据块
                X_block_ptr = tl.advance(X_block_ptr, (BLOCK_M, 0))
                O_block_ptr = tl.advance(O_block_ptr, (BLOCK_M, 0))

        # 创建一个随机张量 t
        t = torch.randn((32, 64, 128))
        # 创建一个与 t 具有相同形状的空张量 o
        o = torch.empty_like(t)
        # 获取张量 t 的维度信息
        B, M, N = t.shape
        # 返回函数 fwd_kernel、其参数字典和一个输出列表
        return (
            fwd_kernel,
            {
                "x_ptr": t,
                "o_ptr": o,
                "M": M,
                "N": N,
                "stride_m": N,
                "stride_n": 1,
                "BLOCK_B": B,
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            ["o_ptr"],
        )

    @make_mutation_test
    def test_while_loop():
        @triton.jit
        def fwd_kernel(
            x_ptr,
            o_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_B: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # 定义 Triton JIT 编译的函数 fwd_kernel，用于执行计算

            # 获取程序块的 ID
            pid_m = tl.program_id(0)

            # 创建输入数据的块指针 X_block_ptr
            X_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            # 创建输出数据的块指针 O_block_ptr
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            # 初始化循环计数器 i
            i = 0
            while i < BLOCK_B:
                # 从 X_block_ptr 中加载数据 x
                x = tl.load(X_block_ptr)
                # 将数据 x 存储到 O_block_ptr 中
                tl.store(O_block_ptr, x)

                # 更新输入块指针 X_block_ptr 和输出块指针 O_block_ptr
                X_block_ptr = tl.advance(X_block_ptr, (BLOCK_M, 0))
                O_block_ptr = tl.advance(O_block_ptr, (BLOCK_M, 0))
                # 增加循环计数器 i
                i += 1

        # 生成测试数据 t 和输出数据 o
        t = torch.randn((32, 64, 128))
        o = torch.empty_like(t)
        # 获取 t 的维度信息
        B, M, N = t.shape
        # 返回 fwd_kernel 函数、参数字典和输出列表
        return (
            fwd_kernel,
            {
                "x_ptr": t,
                "o_ptr": o,
                "M": M,
                "N": N,
                "stride_m": N,
                "stride_n": 1,
                "BLOCK_B": B,
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            ["o_ptr"],
        )
# 如果系统支持 GPU，则执行以下代码块
if HAS_GPU:
    # 生成一个包含4个随机数的张量
    t = torch.randn(4)
    # 生成一个包含4行1列随机数的张量
    tt = torch.randn(4, 1)
    # 对于 tests 中的每个元组，分别将其解包到 kernel、inputs、outputs 变量中
    for kernel, inputs, outputs in tests:
        # 创建一个测试函数，用于测试变异
        fn = make_mutation_test(
            # 添加默认参数以避免 Python lambda 捕获陷阱
            # 这样强制在 lambda 创建时进行捕获
            lambda kernel=kernel, inputs=inputs, outputs=outputs: (
                kernel,
                inputs,
                outputs,
            )
        )
        # 构造测试函数的名称，确保名称在 MutationTests 类中唯一
        name = f"test_mutations_{kernel.fn.__name__}"
        while name in MutationTests.__dict__:
            name += "1"

        # 将生成的测试函数设置为 MutationTests 类的属性，以便后续执行测试
        setattr(MutationTests, name, fn)

# 调用通用工具函数，实例化参数化测试用例
common_utils.instantiate_parametrized_tests(KernelTests)

# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 运行所有的测试用例
    run_tests()
```