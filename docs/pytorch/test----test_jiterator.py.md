# `.\pytorch\test\test_jiterator.py`

```
# Owner(s): ["module: cuda"]

# 导入PyTorch库
import torch
# 导入JIT函数相关的模块
from torch.cuda.jiterator import _create_jit_fn as create_jit_fn
from torch.cuda.jiterator import _create_multi_output_jit_fn as create_multi_output_jit_fn
# 导入系统相关模块
import sys
# 导入迭代工具模块
from itertools import product
# 导入测试相关模块和函数
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, TEST_CUDA, NoTest
# 导入数据类型相关模块
from torch.testing._internal.common_dtype import all_types_and_complex_and
# 导入设备类型相关模块和函数
from torch.testing._internal.common_device_type import (
    skipCUDAIfVersionLessThan, instantiate_device_type_tests, dtypes, toleranceOverride, tol)

# 如果CUDA不可用，则输出错误信息并设置 TestCase 为 NoTest 类型
if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811

# 定义一个 C++ 代码字符串
code_string = "template <typename T> T my_fused_kernel(T x, T y, T alpha, T beta) { return alpha * x + beta * y; }"
# 使用给定的 C++ 代码字符串和参数 alpha=1, beta=1 创建一个 JIT 函数
jitted_fn = create_jit_fn(code_string, alpha=1, beta=1)

# 定义一个参考函数 ref_fn，实现简单的数学运算
def ref_fn(x, y, alpha=1, beta=1):
    return alpha * x + beta * y

# 定义测试类 TestPythonJiterator，继承自 TestCase
class TestPythonJiterator(TestCase):
    # 参数化测试函数 test_all_dtype_contiguous，测试不同的数据类型和内存布局情况
    @parametrize("shape_strides", [
        (([3, 3], [3, 1]), ([3, 3], [3, 1])),  # contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_contiguous(self, device, dtypes, shape_strides):
        # 创建随机数据张量，并设置其类型
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        # 使用 as_strided 方法创建指定形状和步幅的张量
        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        # 计算参考函数的期望结果和 JIT 函数的计算结果
        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        # 断言两者相等
        self.assertEqual(expected, result)

    # 针对非连续内存布局的测试情况，添加跳过 CUDA 版本小于 11.6 的装饰器
    # 详情见 https://github.com/pytorch/pytorch/pull/76394#issuecomment-1118018287
    @skipCUDAIfVersionLessThan((11, 6))
    @parametrize("shape_strides", [
        (([3, 3], [1, 3]), ([3, 1], [1, 3])),  # non-contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_noncontiguous(self, device, dtypes, shape_strides):
        # 创建随机数据张量，并设置其类型
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        # 使用 as_strided 方法创建指定形状和步幅的张量
        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        # 计算参考函数的期望结果和 JIT 函数的计算结果
        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        # 断言两者相等
        self.assertEqual(expected, result)

    # 参数化测试 alpha 和 beta 的不同取值，同时设置 float16 类型的容差
    @dtypes(torch.float, torch.double, torch.float16, torch.bfloat16)
    @parametrize("alpha", [-1, 2.0, None])
    @parametrize("beta", [3, -4.2, None])
    @toleranceOverride({torch.float16 : tol(atol=1e-2, rtol=1e-3)})
    # 测试函数，用于测试带有额外参数的情况
    def test_extra_args(self, device, dtype, alpha, beta):
        # 创建随机张量 a 和 b，设备为指定设备，类型为指定类型，并乘以 10
        a = torch.rand(3, device=device).mul(10).type(dtype)
        b = torch.rand(3, device=device).mul(10).type(dtype)

        # 初始化额外参数字典
        extra_args = {}
        # 如果 alpha 不为空，则将其添加到额外参数字典中
        if alpha is not None:
            extra_args["alpha"] = alpha
        # 如果 beta 不为空，则将其添加到额外参数字典中
        if beta is not None:
            extra_args["beta"] = beta

        # 使用参考函数 ref_fn 计算期望值
        expected = ref_fn(a, b, **extra_args)
        # 使用编译优化函数 jitted_fn 计算结果
        result = jitted_fn(a, b, **extra_args)

        # 断言期望值与结果是否相等
        self.assertEqual(expected, result)

    # 使用 parametrize 装饰器测试布尔类型的额外参数情况
    @parametrize("is_train", [True, False])
    def test_bool_extra_args(self, device, is_train):
        # 定义 C++ 代码字符串，用于创建编译优化函数 jitted_fn
        code_string = "template <typename T> T conditional(T x, T mask, bool is_train) { return is_train ? x * mask : x; }"
        # 根据代码字符串和 is_train 参数创建编译优化函数 jitted_fn
        jitted_fn = create_jit_fn(code_string, is_train=False)

        # 定义参考函数 ref_fn，用于对比编译优化函数 jitted_fn 的结果
        def ref_fn(x, mask, is_train):
            return x * mask if is_train else x

        # 创建随机张量 a 和 b，设备为指定设备
        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)

        # 计算参考函数的期望值
        expected = ref_fn(a, b, is_train=is_train)
        # 使用编译优化函数计算结果
        result = jitted_fn(a, b, is_train=is_train)

        # 断言期望值与结果是否相等
        self.assertEqual(expected, result)

    # 测试多个函数符的情况
    def test_multiple_functors(self, device):
        # 定义多个 C++ 函数模板的字符串形式
        code_string = '''
        template <typename T> T fn(T x, T mask) { return x * mask; }
        template <typename T> T main_fn(T x, T mask, T y) { return fn(x, mask) + y; }
        '''
        # 根据代码字符串创建编译优化函数 jitted_fn
        jitted_fn = create_jit_fn(code_string)

        # 定义参考函数 ref_fn，用于对比编译优化函数 jitted_fn 的结果
        def ref_fn(x, mask, y):
            return x * mask + y

        # 创建随机张量 a、b、c，设备为指定设备
        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)
        c = torch.rand(3, device=device)

        # 计算参考函数的期望值
        expected = ref_fn(a, b, c)
        # 使用编译优化函数计算结果
        result = jitted_fn(a, b, c)

        # 断言期望值与结果是否相等
        self.assertEqual(expected, result)

    # 使用 parametrize 装饰器测试不同数量输入的情况
    @parametrize("num_inputs", [1, 5, 8])
    def test_various_num_inputs(self, num_inputs):
        # 创建 num_inputs 个随机张量，设备为 'cuda'，并乘以 10
        inputs = []
        for i in range(num_inputs):
            inputs.append(torch.rand(3, device='cuda').mul(10))

        # 构造输入的字符串形式
        input_string = ",".join([f"T i{i}" for i in range(num_inputs)])
        # 构造函数主体的字符串形式
        function_body = "+".join([f"i{i}" for i in range(num_inputs)])
        # 构造完整的 C++ 代码字符串形式
        code_string = f"template <typename T> T my_kernel({input_string}) {{ return {function_body}; }}"
        # 根据代码字符串创建编译优化函数 jitted_fn
        jitted_fn = create_jit_fn(code_string)

        # 定义参考函数 ref_fn，用于对比编译优化函数 jitted_fn 的结果
        def ref_fn(*inputs):
            return torch.sum(torch.stack(inputs), dim=0)

        # 计算参考函数的期望值
        expected = ref_fn(*inputs)
        # 使用编译优化函数计算结果
        result = jitted_fn(*inputs)

        # 断言期望值与结果是否相等
        self.assertEqual(expected, result)
    # 定义测试函数，测试不同输出数量的情况
    def test_various_num_outputs(self, num_outputs):
        # 生成一个在 CUDA 设备上的随机张量作为输入
        input = torch.rand(3, device='cuda')

        # 构建输出字符串，形如 "T& out0, T& out1, ..."
        output_string = ", ".join([f"T& out{i}" for i in range(num_outputs)])
        
        # 初始化函数体字符串
        function_body = ""
        for i in range(num_outputs):
            # 每次迭代生成一行代码，形如 "out0 = input + 0;"
            function_body += f"out{i} = input + {i};\n"
        
        # 构建完整的 C++ 代码字符串模板
        code_string = f"template <typename T> void my_kernel(T input, {output_string}) {{ {function_body} }}"
        
        # 调用函数创建多输出的 JIT 编译函数
        jitted_fn = create_multi_output_jit_fn(code_string, num_outputs)

        # 定义参考函数，用于对比验证结果
        def ref_fn(input):
            outputs = []
            for i in range(num_outputs):
                outputs.append(input + i)

            # 如果只有一个输出，返回单个值，否则返回元组
            if num_outputs == 1:
                return outputs[0]
            return tuple(outputs)

        # 计算预期输出
        expected = ref_fn(input)
        
        # 调用 JIT 编译的函数获取实际输出
        result = jitted_fn(input)

        # 断言每个输出是否与预期相符
        for i in range(num_outputs):
            self.assertEqual(expected[i], result[i])

    @parametrize("code_string", [
        # 测试无效的函数名情况，这些示例中函数名包含空格或缺失空格
        "template <typename T> T my _kernel(T x) { return x; }",
        "template <typename T> Tmy_kernel(T x) { return x; }",
    ])
    # 测试无效函数名的测试用例
    def test_invalid_function_name(self, code_string):
        # 确保在创建 JIT 函数时抛出异常
        with self.assertRaises(Exception):
            jitted_fn = create_jit_fn(code_string)
# 调用函数 instantiate_device_type_tests，用于创建测试示例，针对 TestPythonJiterator 类进行测试
# globals() 函数返回当前全局符号表，确保测试函数在全局范围内可用
instantiate_device_type_tests(TestPythonJiterator, globals(), only_for="cuda")

# 检查当前脚本是否作为主程序运行
if __name__ == '__main__':
    # 运行测试函数，执行测试代码
    run_tests()
```