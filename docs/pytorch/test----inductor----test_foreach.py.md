# `.\pytorch\test\inductor\test_foreach.py`

```py
# Owner(s): ["module: inductor"]

# 导入系统相关模块
import sys
# 导入单元测试框架模块
import unittest

# 导入PyTorch库
import torch

# 导入PyTorch的内部模块
import torch._inductor

# 导入自定义的测试用例类
from torch._inductor.test_case import TestCase
# 导入通用测试工具函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)

# 导入PyTorch的内部测试工具模块
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
# 导入需要CUDA支持的测试装饰器
from torch.testing._internal.triton_utils import requires_cuda

# 获取torch.ops.aten别名
aten = torch.ops.aten

# 尝试导入测试模块，处理可能的跳过或导入错误异常
try:
    try:
        from .test_torchinductor import check_model, check_model_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda
except (unittest.SkipTest, ImportError) as e:
    # 如果导入失败，输出错误信息并在主程序中退出
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise

# 待测试的原地二元操作函数列表
inplace_bin_ops_under_test = [
    torch._foreach_add_,
    torch._foreach_mul_,
    torch._foreach_sub_,
    torch._foreach_div_,
]

# 待测试的非原地二元操作函数列表
bin_ops_under_test = [
    torch._foreach_add,
    torch._foreach_mul,
    torch._foreach_sub,
    torch._foreach_div,
    torch._foreach_maximum,
    torch._foreach_minimum,
    torch._foreach_clamp_max,
    torch._foreach_clamp_min,
    aten._foreach_copy,
]

# 待测试的一元操作函数列表
un_ops_under_test = [
    torch._foreach_reciprocal,
    torch._foreach_neg,
    torch._foreach_sign,
    torch._foreach_abs,
    torch._foreach_sqrt,
]

# 合成操作函数列表
compose_ops = [torch._foreach_addcdiv, torch._foreach_addcmul]

# 组合所有需要测试的操作函数
all_ops = parametrize(
    "op", bin_ops_under_test + un_ops_under_test, name_fn=lambda f: f.__name__
)

# 组合需要测试的二元操作函数
bin_ops = parametrize("op", bin_ops_under_test, name_fn=lambda f: f.__name__)

# 组合需要测试的原地二元操作函数
inplace_bin_ops = parametrize(
    "op", inplace_bin_ops_under_test, name_fn=lambda f: f.__name__
)

# 组合需要测试的标量二元操作函数
scalar_bin_ops = parametrize("op", bin_ops_under_test[:4], name_fn=lambda f: f.__name__)

# 组合需要测试的标量张量二元操作函数
scalar_tensor_bin_ops = parametrize(
    "op", bin_ops_under_test[:2], name_fn=lambda f: f.__name__
)

# 组合需要测试的合成操作函数
decomp_ops = parametrize("op", compose_ops, name_fn=lambda f: f.__name__)


# 根据操作函数返回生成参数组合的函数
def gen_args(op):
    if op in un_ops_under_test:
        # 如果操作函数是一元操作，则返回两个随机张量作为参数
        return (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )
    else:
        # 否则返回四个随机张量作为参数
        return (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )


# 实例化参数化测试类
@instantiate_parametrized_tests
class ForeachTests(TestCase):
    # 设置CUDA环境下的模型检查函数
    check_model_cuda = check_model_cuda
    # 设置CPU环境下的模型检查函数
    check_model_cpu = check_model
    # 检查核心计数开关
    check_kernel_count = True

    # 设置测试前的初始化操作
    def setUp(self):
        super().setUp()
        # 重置PyTorch内部诊断指标
        torch._inductor.metrics.reset()

    # 设置测试后的清理操作
    def tearDown(self):
        super().tearDown()
        # 重置PyTorch内部诊断指标
        torch._inductor.metrics.reset()

    # 测试单个操作函数对列表参数的处理
    def _test_single_list(self, op):
        if op in un_ops_under_test:
            # 如果操作函数是一元操作，则定义接受两个参数的操作函数
            def fn(a0, a1):
                return op([a0, a1])
        else:
            # 否则定义接受四个参数的操作函数
            def fn(a0, a1, b0, b1):
                return op([a0, a1], [b0, b1])

        # 调用检查CUDA环境下模型的函数，传入操作函数和其生成的参数
        self.check_model_cuda(
            fn,
            gen_args(op),
        )
    # 定义一个测试函数，测试对单个标量进行操作的函数
    def _test_single_scalar(self, op):
        # 定义一个内部函数，将操作函数应用于两个参数和标量值3.3
        def fn(a0, a1):
            return op([a0, a1], 3.3)

        # 调用带有 CUDA 支持的模型检查函数，传入两个随机生成的张量作为参数
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    # 定义一个测试函数，测试对单个标量张量进行操作的函数
    def _test_single_scalar_tensor(self, op):
        # 定义一个内部函数，将操作函数应用于两个参数和包含标量值3.3的张量
        def fn(a0, a1):
            return op([a0, a1], torch.tensor(3.3, device="cuda:0"))

        # 调用带有 CUDA 支持的模型检查函数，传入两个随机生成的张量作为参数
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    # 在 test_cuda_cpp_wrapper.py 中调用，测试 CUDA 环境下的 C++ 封装迭代函数
    @requires_cuda
    def test_foreach_cpp_wrapper_cuda(self):
        self._test_single_list(op=torch._foreach_add)

    # 使用所有操作函数进行测试
    @requires_cuda
    @all_ops
    def test_single_list(self, op):
        self._test_single_list(op)
        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 使用标量二元操作函数进行测试
    @requires_cuda
    @scalar_bin_ops
    def test_single_scalar(self, op):
        self._test_single_scalar(op)
        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 使用标量张量二元操作函数进行测试
    @requires_cuda
    @scalar_tensor_bin_ops
    def test_single_scalar_tensor(self, op):
        self._test_single_scalar_tensor(op)
        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 使用所有操作函数进行测试，测试调度器在列表输入上的融合
    @requires_cuda
    @all_ops
    def test_scheduler_fusion_list(self, op):
        if op in un_ops_under_test:
            # 如果是测试的单目操作，则定义一个操作函数，对两个输入进行操作并开根号
            def fn(a0, a1):
                c = op([a0, a1])
                return torch._foreach_sqrt(c)
        else:
            # 否则，定义一个操作函数，对两个输入进行操作并将结果与一个标量加起来，同时对另一组输入进行加法操作
            def fn(a0, a1, b0, b1):
                c = op([a0, a1], [b0, b1])
                return c, torch._foreach_add([a0, a1], c)

        # 调用带有 CUDA 支持的模型检查函数，根据操作函数生成参数进行测试
        self.check_model_cuda(
            fn,
            gen_args(op),
        )

        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 使用标量二元操作函数进行测试，测试调度器在标量上的融合
    @requires_cuda
    @scalar_bin_ops
    def test_scheduler_fusion_scalar(self, op):
        # 定义一个操作函数，对两个输入进行操作并加上标量3.4，同时对另一组输入进行加法操作
        def fn(a0, a1):
            c = op([a0, a1], 3.4)
            return c, torch._foreach_add([a0, a1], c)

        # 调用带有 CUDA 支持的模型检查函数，传入两个随机生成的张量作为参数
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 使用标量二元操作函数进行测试，测试广播功能
    @requires_cuda
    @scalar_bin_ops
    def test_broadcasting(self, op):
        # 定义一个操作函数，对四个输入进行操作
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        # 对操作函数进行优化
        fn_opt = torch._dynamo.optimize()(fn)

        # 定义输入参数
        inputs = (
            torch.rand(10, 1, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(1, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        # 计算优化后的结果
        actual = fn_opt(*inputs)
        # 计算期望结果
        expected = fn(*inputs)
        
        # 断言优化后的结果与期望结果相等
        self.assertEqual(actual, expected)
        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
    # 将当前函数标记为在所有操作上的测试函数装饰器
    @all_ops
    def test_singleton_lists(self, op):
        # 如果操作在被测试的单目操作列表中
        if op in un_ops_under_test:

            # 定义一个仅接受一个参数的函数，应用操作并返回结果
            def fn(a0):
                return op([a0])

            # 准备单一参数作为测试输入
            args = (torch.rand(10, 10, device="cuda:0"),)
        else:
            
            # 定义一个接受两个参数的函数，应用操作并返回结果
            def fn(a0, b0):
                return op([a0], [b0])

            # 准备两个参数作为测试输入
            args = (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
            )

        # 检查模型在 CUDA 上的运行情况
        self.check_model_cuda(
            fn,
            args,
        )

        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 标记测试函数依赖 CUDA，并且在二元操作上进行装饰
    @requires_cuda
    @bin_ops
    def test_type_promotion(self, op):
        # 定义一个函数，接受四个参数，应用操作并返回结果
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        # 优化函数 fn 并得到优化后的版本
        fn_opt = torch._dynamo.optimize()(fn)

        # 设置不同数据类型的最大值
        max32 = torch.iinfo(torch.int32).max
        max64 = torch.iinfo(torch.int64).max
        # 准备输入数据
        inputs = (
            torch.randint(max32, (10, 10), device="cuda:0", dtype=torch.int32),
            torch.randint(max32, (20, 20), device="cuda:0", dtype=torch.int32),
            torch.randint(max32, (10, 10), device="cuda:0", dtype=torch.int32),
            torch.randint(max64, (20, 20), device="cuda:0", dtype=torch.int64),
        )
        # 对优化后的函数进行测试
        actual = fn_opt(*inputs)
        # 对原始函数进行测试
        expected = fn(*inputs)
        # 断言优化后的结果与原始结果一致
        self.assertEqual(actual, expected)
        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 标记测试函数依赖 CUDA，并且在标量二元操作上进行装饰
    @requires_cuda
    @scalar_bin_ops
    def test_kernel_split_arg_limit_list(self, op):
        # NB: foeach_copy 不会通过此测试，因为它会删除一个缓冲区

        # 定义一个函数，接受两个参数并应用操作返回结果
        def fn(a, b):
            return op(a, b)

        # 优化函数 fn 并得到优化后的版本
        fn_opt = torch._dynamo.optimize()(fn)

        # 设置最大参数数量和每个列表的最大长度
        max_args = 370
        max_list_len = (max_args // 3) + 1
        # 准备输入数据，每个列表包含多个随机生成的张量
        inputs = (
            [torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],
            [torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],
        )

        # 对优化后的函数进行测试
        actual = fn_opt(*inputs)
        # 对原始函数进行测试
        expected = fn(*inputs)
        # 断言优化后的结果与原始结果一致
        self.assertEqual(actual, expected)
        # 断言生成的内核数量为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    # 标记测试函数依赖 CUDA，并且在标量二元操作上进行装饰，同时跳过测试
    @requires_cuda
    @scalar_bin_ops
    @unittest.skip(
        "Triton recursion depth exceeded: https://github.com/openai/triton/issues/1763"
    )
    def test_kernel_split_arg_limit_scalar(self, op):
        # 定义一个函数，接受一个参数和一个标量，并应用操作返回结果
        def fn(a):
            return op(a, 3.3)

        # 优化函数 fn 并得到优化后的版本
        fn_opt = torch._dynamo.optimize()(fn)

        # 设置最大参数数量和每个列表的最大长度
        max_args = 370
        max_list_len = (max_args // 2) + 1
        # 准备输入数据，每个列表包含多个随机生成的张量
        inputs = ([torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],)

        # 对优化后的函数进行测试
        actual = fn_opt(*inputs)
        # 对原始函数进行测试
        expected = fn(*inputs)
        # 断言优化后的结果与原始结果一致
        self.assertEqual(actual, expected)
        # 断言生成的内核数量为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)
    # 定义一个测试函数，用于测试操作op对重复缓冲列表的融合情况
    def test_fusion_duplicate_buffer_list(self, op):
        # 内部函数定义，接受四个参数a0, a1, b0, b1，并使用操作op进行计算
        def fn(a0, a1, b0, b1):
            # 对[a0, a1]和[b0, b1]进行操作op，将结果保存在变量c中
            c = op([a0, a1], [b0, b1])
            # 对[a0, b0]进行操作op，然后将结果的第一个元素复制给c的两个元素
            return op([a0, b0], [c[0], c[0]])

        # 使用CUDA进行模型检查，检查函数fn在CUDA设备上的执行情况
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),  # 生成CUDA上的随机张量，形状为10x10
                torch.rand(20, 20, device="cuda:0"),  # 生成CUDA上的随机张量，形状为20x20
                torch.rand(10, 10, device="cuda:0"),  # 生成CUDA上的随机张量，形状为10x10
                torch.rand(20, 20, device="cuda:0"),  # 生成CUDA上的随机张量，形状为20x20
            ),
            reference_in_float=False,  # 不在浮点数中引用
            check_lowp=False,  # 不检查低精度
        )

        # 断言生成的内核计数为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 标记测试需要CUDA支持，并且适用于所有操作
    @requires_cuda
    @all_ops
    def test_non_foreach_consumer_list(self, op):
        # 如果操作op在un_ops_under_test中

            # 内部函数定义，接受两个参数a0, a1，并使用操作op进行计算
            def fn(a0, a1):
                # 对[a0, a1]进行操作op，将结果保存在变量c中
                c = op([a0, a1])
                # 返回操作op中第一个元素c[0]与a0的乘积
                return torch.mul(c[0], a0)

        # 否则
        else:

            # 内部函数定义，接受四个参数a0, a1, b0, b1，并使用操作op进行计算
            def fn(a0, a1, b0, b1):
                # 对[a0, a1]和[b0, b1]进行操作op，将结果保存在变量c中
                c = op([a0, a1], [b0, b1])
                # 返回操作op中第一个元素c[0]与a0的乘积
                return torch.mul(c[0], a0)

        # 使用CUDA进行模型检查，检查函数fn在CUDA设备上的执行情况，生成参数通过gen_args(op)生成
        self.check_model_cuda(
            fn,
            gen_args(op),
        )

        # 断言生成的内核计数为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 标记测试需要CUDA支持，并且适用于标量二元操作
    @requires_cuda
    @scalar_bin_ops
    def test_non_foreach_consumer_scalar(self, op):
        # 内部函数定义，接受两个参数a0, a1，并使用操作op进行计算
        def fn(a0, a1):
            # 对[a0, a1]和标量值4.7进行操作op，将结果保存在变量c中
            c = op([a0, a1], 4.7)
            # 返回操作op中第一个元素c[0]与a0的乘积
            return torch.mul(c[0], a0)

        # 使用CUDA进行模型检查，检查函数fn在CUDA设备上的执行情况
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),  # 生成CUDA上的随机张量，形状为10x10
                torch.rand(20, 20, device="cuda:0"),  # 生成CUDA上的随机张量，形状为20x20
            ),
        )

        # 断言生成的内核计数为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 标记测试需要CUDA支持，并且适用于所有操作
    @requires_cuda
    @all_ops
    def test_non_foreach_producer_list(self, op):
        # 如果操作op在un_ops_under_test中

            # 内部函数定义，接受两个参数a0, a1，并使用torch.add进行计算
            def fn(a0, a1):
                # 对a0和a1各自进行加法操作，将结果保存在变量c0和c1中
                c0 = torch.add(a0, a0)
                c1 = torch.add(a1, a1)
                # 返回操作op对[c0, c1]进行操作的结果
                return op([c0, c1])

        # 否则
        else:

            # 内部函数定义，接受四个参数a0, a1, b0, b1，并使用torch.add进行计算
            def fn(a0, a1, b0, b1):
                # 对a0和b0进行加法操作，将结果保存在变量c0中
                c0 = torch.add(a0, b0)
                # 对a1和b1进行加法操作，将结果保存在变量c1中
                c1 = torch.add(a1, b1)
                # 返回操作op对[a0, a1]和[c0, c1]进行操作的结果
                return op([a0, a1], [c0, c1])

        # 使用CUDA进行模型检查，检查函数fn在CUDA设备上的执行情况，生成参数通过gen_args(op)生成
        self.check_model_cuda(
            fn, gen_args(op), reference_in_float=False, check_lowp=False
        )

        # 断言生成的内核计数为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 标记测试需要CUDA支持，并且适用于标量二元操作
    @requires_cuda
    @scalar_bin_ops
    def test_non_foreach_producer_scalar(self, op):
        # 内部函数定义，接受四个参数a0, a1, b0, b1，并使用torch.mul进行计算
        def fn(a0, a1, b0, b1):
            # 对a0和b0进行乘法操作，将结果保存在变量c0中
            c0 = torch.mul(a0, b0)
            # 对a1和b1进行乘法操作，将结果保存在变量c1中
            c1 = torch.mul(a1, b1)
            # 返回操作op对[c0, c1]和标量值5.6进行操作的结果
            return op([c0, c1], 5.6)

        # 使用CUDA进行模型检查，检查函数fn在CUDA设备上的执行情况
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),  # 生成CUDA上的随机张量，形状为10x10
                torch.rand(20, 20, device="cuda:0"),  # 生成CUDA上的随机张量，形状为20x20
                torch.rand(10, 10, device="cuda:0"),  # 生成CUDA上的随机张量，形状为10x10
                torch.rand(20, 20, device="cuda:0"),  # 生成CUDA上的随机张量，形状为20x20
            ),
        )

        # 断言生成的内核计数为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
    def test_non_foreach_consumer_producer_list(self, op):
        # 如果操作符在未测试的一元操作列表中
        if op in un_ops_under_test:

            # 定义函数 fn，接受两个参数 a0 和 a1
            def fn(a0, a1):
                # 计算 c0 = a0 + a0
                c0 = torch.add(a0, a0)
                # 计算 c1 = a1 * a1
                c1 = torch.mul(a1, a1)
                # 对 c0 和 c1 执行操作 op，返回结果存储在 d 中
                d = op([c0, c1])
                # 计算 e0 = d[0] * a0
                e0 = torch.mul(d[0], a0)
                # 计算 e1 = d[1] * a1
                e1 = torch.mul(d[1], a1)
                # 返回结果列表 [e0, e1]
                return [e0, e1]

        else:
            # 否则，定义函数 fn，接受四个参数 a0, a1, b0, b1
            def fn(a0, a1, b0, b1):
                # 计算 c0 = a0 + b0
                c0 = torch.add(a0, b0)
                # 计算 c1 = a1 + b1
                c1 = torch.add(a1, b1)
                # 对 [a0, a1] 和 [c0, c1] 执行操作 op，返回结果存储在 d 中
                d = op([a0, a1], [c0, c1])
                # 计算 e0 = d[0] * a0
                e0 = torch.mul(d[0], a0)
                # 计算 e1 = d[1] * a1
                e1 = torch.mul(d[1], a1)
                # 返回结果列表 [e0, e1]
                return [e0, e1]

        # 调用 self.check_model_cuda 方法，验证函数 fn 在 CUDA 设备上的执行情况
        self.check_model_cuda(
            fn,
            gen_args(op),  # 生成函数参数列表
            reference_in_float=False,  # 不使用浮点数参考值
            check_lowp=False,  # 不检查低精度
        )

        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_non_foreach_consumer_producer_scalar(self, op):
        # 定义函数 fn，接受四个参数 a0, a1, b0, b1
        def fn(a0, a1, b0, b1):
            # 计算 c0 = a0 + b0
            c0 = torch.add(a0, b0)
            # 计算 c1 = a1 + b1
            c1 = torch.add(a1, b1)
            # 对 [c0, c1] 和标量值 5.8 执行操作 op，返回结果存储在 d 中
            d = op([c0, c1], 5.8)
            # 计算 e0 = d[0] * a0
            e0 = torch.mul(d[0], a0)
            # 计算 e1 = d[1] * a1
            e1 = torch.mul(d[1], a1)
            # 返回结果列表 [e0, e1]
            return [e0, e1]

        # 调用 self.check_model_cuda 方法，验证函数 fn 在 CUDA 设备上的执行情况
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),  # 随机生成张量，CUDA 设备
                torch.rand(20, 20, device="cuda:0"),  # 随机生成张量，CUDA 设备
                torch.rand(10, 10, device="cuda:0"),  # 随机生成张量，CUDA 设备
                torch.rand(20, 20, device="cuda:0"),  # 随机生成张量，CUDA 设备
            ),
            reference_in_float=False,  # 不使用浮点数参考值
            check_lowp=False,  # 不检查低精度
        )

        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @bin_ops
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_fallback(self, op):
        # 定义函数 fn，接受四个参数 a0, a1, b0, b1
        def fn(a0, a1, b0, b1):
            # 对 [a0, a1] 和 [b0, b1] 执行操作 op，返回结果
            return op([a0, a1], [b0, b1])

        # 定义输入数据
        inputs = (
            torch.rand(10, 10, device="cuda:0"),  # 随机生成张量，CUDA 设备
            torch.rand(20, 20, device="cuda:0"),  # 随机生成张量，CUDA 设备
            torch.rand(10, 10, device="cuda:0"),  # 随机生成张量，CUDA 设备
            torch.rand(20, 20, device="cuda:0"),  # 随机生成张量，CUDA 设备
        )

        # 调用 self.check_model_cuda 方法，验证函数 fn 在 CUDA 设备上的执行情况
        self.check_model_cuda(fn, inputs)

        # 断言生成的内核数量为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @unittest.skipIf(IS_FBCODE, "cpp compile not supported in fbcode")
    @bin_ops
    def test_cpu_cpp_fallback(self, op):
        # 定义函数 fn，接受四个参数 a0, a1, b0, b1
        def fn(a0, a1, b0, b1):
            # 对 [a0, a1] 和 [b0, b1] 执行操作 op，返回结果
            return op([a0, a1], [b0, b1])

        # 定义输入数据
        inputs = (
            torch.rand(10, 10, device="cpu"),  # 随机生成张量，CPU 设备
            torch.rand(20, 20, device="cpu"),  # 随机生成张量，CPU 设备
            torch.rand(10, 10, device="cpu"),  # 随机生成张量，CPU 设备
            torch.rand(20, 20, device="cpu"),  # 随机生成张量，CPU 设备
        )

        # 调用 self.check_model_cpu 方法，验证函数 fn 在 CPU 设备上的执行情况
        self.check_model_cpu(fn, inputs)

        # 断言生成的内核数量为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @decomp_ops
    # 定义一个测试函数 test_decomp，接受一个操作函数 op 作为参数
    def test_decomp(self, op):
        # 定义一个内部函数 fn，接受六个参数并使用 op 函数进行计算
        def fn(a0, a1, b0, b1, c0, c1):
            return op([a0, a1], [b0, b1], [c0, c1], value=0.5)

        # 调用 self.check_model_cuda 方法，验证 fn 函数在 CUDA 设备上执行的结果
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),  # 随机生成一个 10x10 的张量在 CUDA 设备上
                torch.rand(20, 20, device="cuda:0"),  # 随机生成一个 20x20 的张量在 CUDA 设备上
                torch.rand(10, 10, device="cuda:0"),  # 随机生成一个 10x10 的张量在 CUDA 设备上
                torch.rand(20, 20, device="cuda:0"),  # 随机生成一个 20x20 的张量在 CUDA 设备上
                torch.rand(10, 10, device="cuda:0"),  # 随机生成一个 10x10 的张量在 CUDA 设备上
                torch.rand(20, 20, device="cuda:0"),  # 随机生成一个 20x20 的张量在 CUDA 设备上
            ),
        )

        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 带有 requires_cuda 装饰器的测试函数，测试在 CUDA 上合并张量操作
    @requires_cuda
    def test_fuse_concat(self):
        # 定义一个内部函数 fn，合并 x1, x2, x3 张量并进行与 w1, w2, w3 的矩阵乘法
        def fn(x1, x2, x3, w1, w2, w3):
            x = torch.stack([x1, x2, x3])  # 将 x1, x2, x3 合并成一个张量 x
            w = torch.stack([w1, w2, w3])  # 将 w1, w2, w3 合并成一个张量 w

            y = torch.bmm(x, w)  # 对 x 和 w 进行批矩阵乘法

            return y

        # 在 CUDA 设备上创建张量 x1, x2, x3, w1, w2, w3
        x1 = torch.randn(5, 4).cuda()  # 随机生成一个大小为 5x4 的张量在 CUDA 设备上
        x2 = x1 + 1  # x2 是 x1 加上 1
        x3 = x1 + 2  # x3 是 x1 加上 2
        w1 = torch.randn(4, 3).cuda()  # 随机生成一个大小为 4x3 的张量在 CUDA 设备上
        w2 = w1 + 1  # w2 是 w1 加上 1
        w3 = w1 + 2  # w3 是 w1 加上 2

        args = (x1, x2, x3, w1, w2, w3)  # 将这些张量打包成一个元组

        # 调用 self.check_model_cuda 方法，验证 fn 函数在 CUDA 设备上执行的结果
        self.check_model_cuda(fn, args)

        # 断言生成的内核数量为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    # 带有 requires_cuda 装饰器的测试函数，测试处理零元素张量的操作
    @requires_cuda
    def test_zero_elems(self):
        # 定义一个内部函数 fn，对输入的 a0, a1, b0, b1 张量进行逐元素加法
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        # 调用 self.check_model_cuda 方法，验证 fn 函数在 CUDA 设备上执行的结果
        self.check_model_cuda(
            fn,
            (
                torch.rand(0, device="cuda:0"),  # 在 CUDA 设备上生成一个大小为 0 的随机张量
                torch.rand(10, 10, device="cuda:0"),  # 在 CUDA 设备上生成一个 10x10 的随机张量
                torch.rand(0, device="cuda:0"),  # 在 CUDA 设备上生成一个大小为 0 的随机张量
                torch.rand(10, 10, device="cuda:0"),  # 在 CUDA 设备上生成一个 10x10 的随机张量
            ),
        )

        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 带有 requires_cuda 和 bin_ops 装饰器的测试函数，测试在 CUDA 上的二维阻塞操作
    @requires_cuda
    @bin_ops
    def test_2d_blocking(self, op):
        # 定义一个内部函数 fn，对输入的 a0, a1, b0, b1 张量执行 op 函数
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        # 调用 self.check_model_cuda 方法，验证 fn 函数在 CUDA 设备上执行的结果
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 40, device="cuda:0"),  # 在 CUDA 设备上生成一个 10x40 的随机张量
                torch.rand(10, 30, device="cuda:0"),  # 在 CUDA 设备上生成一个 10x30 的随机张量
                torch.rand(40, 10, device="cuda:0").t(),  # 在 CUDA 设备上生成一个 40x10 的随机张量，并转置
                torch.rand(30, 10, device="cuda:0").t(),  # 在 CUDA 设备上生成一个 30x10 的随机张量，并转置
            ),
        )

        # 断言生成的内核数量为 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    # 带有 requires_cuda 和 bin_ops 装饰器的测试函数，测试在 CUDA 上的二维阻塞分区操作
    @requires_cuda
    @bin_ops
    def test_2d_blocking_partitioning(self, op):
        # 定义一个内部函数 fn，对输入的 a0, a1, b0, b1 张量执行 op 函数
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        # 调用 self.check_model_cuda 方法，验证 fn 函数在 CUDA 设备上执行的结果
        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda:0"),  # 在 CUDA 设备上生成一个 30x20 的随机张量
                torch.rand(40, 30, device="cuda:0"),  # 在 CUDA 设备上生成一个 40x30 的随机张量
                torch.rand(30, 20, device="cuda:0"),  # 在 CUDA 设备上生成一个 30x20 的随机张量
                torch.rand(30, 40, device="cuda:0").t(),  # 在 CUDA 设备上生成一个 30x40 的随机张量，并转置
            ),
        )

        # 断言生成的内核数量为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)
    @requires_cuda
    @inplace_bin_ops
    def test_2d_blocking_partitioning_elems(self, op):
        """测试2D阻塞分区应按yelems数分组"""

        def fn(a0, a1, a2, b0, b1, b2):
            # 调用操作函数op，传入两个输入列表a和b
            return op([a0, a1, a2], [b0, b1, b2])

        # 在CUDA设备上检查模型行为
        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 20, device="cuda:0"),
                torch.rand(30, 20, device="cuda:0"),
                torch.rand(10, 30, device="cuda:0"),
                torch.rand(20, 10, device="cuda:0").t(),
                torch.rand(20, 30, device="cuda:0").t(),
                torch.rand(30, 10, device="cuda:0").t(),
            ),
        )

        # 断言生成的内核数量为2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @inplace_bin_ops
    def test_reinplacing(self, op):
        """测试重新放置操作"""

        def fn(a0, a1, b0, b1):
            # 调用操作函数op，传入两个输入列表a和b
            op([a0, a1], [b0, b1])
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        # 在CUDA设备上检查模型行为，不检查低精度
        self.check_model_cuda(fn, inputs, check_lowp=False)

        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @inplace_bin_ops
    def test_reinplacing_mut_before(self, op):
        """测试先前的重新放置操作"""

        def fn(a0, a1, b0, b1):
            # 在a0上执行原位操作，然后调用操作函数op，传入两个输入列表a和b
            a0.add_(torch.ones(10, 10, device="cuda:0"))
            op([a0, a1], [b0, b1])
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        # 在CUDA设备上检查模型行为，不检查低精度
        self.check_model_cuda(fn, inputs, check_lowp=False)

        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @inplace_bin_ops
    def test_reinplacing_mut_after(self, op):
        """测试后续的重新放置操作"""

        def fn(a0, a1, b0, b1):
            # 调用操作函数op，传入两个输入列表a和b，然后在a0上执行原位操作
            op([a0, a1], [b0, b1])
            a0.add_(torch.ones(10, 10, device="cuda:0"))
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        # 在CUDA设备上检查模型行为，不检查低精度
        self.check_model_cuda(fn, inputs, check_lowp=False)

        # 断言生成的内核数量为1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
    # 定义一个测试方法，用于测试多设备下的操作
    def test_multi_device(self):
        # 定义一个内部函数，用于将两组张量逐元素相加
        def test_foreach_add(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        # 创建包含不同设备上张量的列表
        inps = [
            torch.ones(10, 10, device="cuda"),   # 在 CUDA 设备上创建全 1 张量
            torch.ones(20, 20, device="cpu"),    # 在 CPU 上创建全 1 张量
            torch.zeros(10, 10, device="cuda"),  # 在 CUDA 设备上创建全 0 张量
            torch.zeros(20, 20, device="cpu"),   # 在 CPU 上创建全 0 张量
        ]

        # 对输入的张量进行逐元素相加的测试，以计算出在非编译模式下的输出
        out_eager = test_foreach_add(*inps)
        # 使用编译模式对输入的张量进行逐元素相加的测试，以计算出在编译模式下的输出
        out_compiled = torch.compile(test_foreach_add)(*inps)

        # 断言两种模式下的输出结果应相等
        self.assertEqual(out_eager, out_compiled)
        # 断言生成的 CUDA 核心数量应为 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    # 标记需要 CUDA 支持的测试方法
    @requires_cuda
    def test_aliasing(self):
        # 定义一个内部函数，用于将三组张量逐元素相加
        def test_foreach_add(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add_([a0, a1, a2], [b0, b1, b2])

        # 在 CUDA 设备上创建全 1 张量
        input = torch.ones(10, 10, device="cuda")
        # 在 CUDA 设备上创建另一个全 1 张量
        input2 = torch.ones(10, 10, device="cuda")
        # 创建包含多个张量及其视图的列表，用于测试内存别名问题
        inps = [
            input,                          # 第一个全 1 张量
            input.view(10, 10),             # 第一个张量的视图
            input.view(10, 10),             # 第一个张量的另一个视图
            input2,                         # 第二个全 1 张量
            input2.view(10, 10),            # 第二个张量的视图
            input2.view(10, 10),            # 第二个张量的另一个视图
        ]

        # 对输入的张量进行逐元素相加的测试，以计算出在非编译模式下的输出
        out_eager = test_foreach_add(*inps)
        # 使用编译模式对输入的张量进行逐元素相加的测试，以计算出在编译模式下的输出
        out_compiled = torch.compile(test_foreach_add)(*inps)

        # 断言两种模式下的输出结果应相等
        self.assertEqual(out_eager, out_compiled)
        # 断言生成的 CUDA 核心数量应为 4
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 从torch._inductor.test_case模块导入run_tests函数
    from torch._inductor.test_case import run_tests
    
    # 如果已经安装了CPU或CUDA支持
    if HAS_CPU or HAS_CUDA:
        # 运行测试，确保需要filelock模块
        run_tests(needs="filelock")
```