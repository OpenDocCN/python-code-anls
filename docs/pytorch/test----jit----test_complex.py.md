# `.\pytorch\test\jit\test_complex.py`

```
# Owner(s): ["oncall: jit"]

import cmath  # 导入复数数学运算模块
import os  # 导入操作系统功能模块
import sys  # 导入系统相关模块
from itertools import product  # 导入迭代工具函数
from textwrap import dedent  # 导入文本处理工具函数
from typing import Dict, List  # 导入类型提示相关模块

import torch  # 导入PyTorch深度学习库
from torch.testing._internal.common_utils import IS_MACOS  # 导入PyTorch内部测试工具
from torch.testing._internal.jit_utils import execWrapper, JitTestCase  # 导入PyTorch JIT测试相关工具

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # 获取当前脚本所在目录的父目录路径
sys.path.append(pytorch_test_dir)  # 将父目录路径添加到系统路径中，使得其中的辅助文件可以被导入

class TestComplex(JitTestCase):
    def test_script(self):
        def fn(a: complex):
            return a

        self.checkScript(fn, (3 + 5j,))  # 使用复数3+5j来测试函数fn的脚本化版本

    def test_complexlist(self):
        def fn(a: List[complex], idx: int):
            return a[idx]

        input = [1j, 2, 3 + 4j, -5, -7j]
        self.checkScript(fn, (input, 2))  # 使用复数列表和索引2来测试函数fn的脚本化版本

    def test_complexdict(self):
        def fn(a: Dict[complex, complex], key: complex) -> complex:
            return a[key]

        input = {2 + 3j: 2 - 3j, -4.3 - 2j: 3j}
        self.checkScript(fn, (input, -4.3 - 2j))  # 使用复数字典和复数键来测试函数fn的脚本化版本

    def test_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.a = 3 + 5j
                self.b = [2 + 3j, 3 + 4j, 0 - 3j, -4 + 0j]
                self.c = {2 + 3j: 2 - 3j, -4.3 - 2j: 3j}

            @torch.jit.script_method
            def forward(self, b: int):
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)  # 验证加载的模块的属性a是否与预期的复数3+5j相等
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])  # 验证加载的模块的属性b是否与预期的复数列表相等
        self.assertEqual(loaded.c, {2 + 3j: 2 - 3j, -4.3 - 2j: 3j})  # 验证加载的模块的属性c是否与预期的复数字典相等
        self.assertEqual(loaded(2), 2 + 2j)  # 验证加载的模块在输入2时的前向计算结果是否与预期的复数相等

    def test_complex_parse(self):
        def fn(a: int, b: torch.Tensor, dim: int):
            # verifies `emitValueToTensor()` 's behavior
            b[dim] = 2.4 + 0.5j  # 验证在特定维度上将复数值2.4+0.5j赋给张量b的行为
            return (3 * 2j) + a + 5j - 7.4j - 4  # 返回复数计算结果

        t1 = torch.tensor(1)
        t2 = torch.tensor([0.4, 1.4j, 2.35])

        self.checkScript(fn, (t1, t2, 2))  # 使用整数1、张量t2和索引2来测试函数fn的脚本化版本

    def test_infj_nanj_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.a = 3 + 5j

            @torch.jit.script_method
            def forward(self, infj: int, nanj: int):
                if infj == 2:
                    return infj + cmath.infj
                else:
                    return nanj + cmath.nanj

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded(2, 3), 2 + cmath.infj)  # 验证加载的模块在输入为2和3时返回的结果是否与预期的复数infj相等
        self.assertEqual(loaded(3, 4), 4 + cmath.nanj)  # 验证加载的模块在输入为3和4时返回的结果是否与预期的复数nanj相等
    # 定义测试函数，测试使用张量作为构造函数参数的复杂数
    def test_torch_complex_constructor_with_tensor(self):
        # 创建包含不同类型张量的列表
        tensors = [torch.rand(1), torch.randint(-5, 5, (1,)), torch.tensor([False])]

        # 定义将张量和浮点数转换为复数的函数
        def fn_tensor_float(real, img: float):
            return complex(real, img)

        # 定义将张量和整数转换为复数的函数
        def fn_tensor_int(real, img: int):
            return complex(real, img)

        # 定义将张量和布尔值转换为复数的函数
        def fn_tensor_bool(real, img: bool):
            return complex(real, img)

        # 定义将浮点数和张量转换为复数的函数
        def fn_float_tensor(real: float, img):
            return complex(real, img)

        # 定义将整数和张量转换为复数的函数
        def fn_int_tensor(real: int, img):
            return complex(real, img)

        # 定义将布尔值和张量转换为复数的函数
        def fn_bool_tensor(real: bool, img):
            return complex(real, img)

        # 遍历张量列表，并测试不同转换函数的脚本化效果
        for tensor in tensors:
            self.checkScript(fn_tensor_float, (tensor, 1.2))
            self.checkScript(fn_tensor_int, (tensor, 3))
            self.checkScript(fn_tensor_bool, (tensor, True))

            self.checkScript(fn_float_tensor, (1.2, tensor))
            self.checkScript(fn_int_tensor, (3, tensor))
            self.checkScript(fn_bool_tensor, (True, tensor))

        # 定义接受两个张量参数并返回其复数形式相加的函数
        def fn_tensor_tensor(real, img):
            return complex(real, img) + complex(2)

        # 使用张量列表的笛卡尔积，测试复数相加的脚本化效果
        for x, y in product(tensors, tensors):
            self.checkScript(
                fn_tensor_tensor,
                (
                    x,
                    y,
                ),
            )

    # 测试复数的比较运算符
    def test_comparison_ops(self):
        # 定义比较两个复数是否相等的函数
        def fn1(a: complex, b: complex):
            return a == b

        # 定义比较两个复数是否不相等的函数
        def fn2(a: complex, b: complex):
            return a != b

        # 定义比较复数与浮点数是否相等的函数
        def fn3(a: complex, b: float):
            return a == b

        # 定义比较复数与浮点数是否不相等的函数
        def fn4(a: complex, b: float):
            return a != b

        # 定义复数变量 x, y
        x, y = 2 - 3j, 4j
        # 测试复数相等比较的脚本化效果
        self.checkScript(fn1, (x, x))
        self.checkScript(fn1, (x, y))
        # 测试复数不等比较的脚本化效果
        self.checkScript(fn2, (x, x))
        self.checkScript(fn2, (x, y))

        # 定义复数变量 x1, y1
        x1, y1 = 1 + 0j, 1.0
        # 测试复数与浮点数相等比较的脚本化效果
        self.checkScript(fn3, (x1, y1))
        # 测试复数与浮点数不等比较的脚本化效果
        self.checkScript(fn4, (x1, y1))

    # 测试复数的除法运算
    def test_div(self):
        # 定义两个复数相除的函数
        def fn1(a: complex, b: complex):
            return a / b

        # 定义复数变量 x, y
        x, y = 2 - 3j, 4j
        # 测试复数除法运算的脚本化效果
        self.checkScript(fn1, (x, y))

    # 测试复数列表的求和运算
    def test_complex_list_sum(self):
        # 定义接受复数列表并返回其求和结果的函数
        def fn(x: List[complex]):
            return sum(x)

        # 使用 torch 生成随机复数张量，将其转换为列表，并测试求和函数的脚本化效果
        self.checkScript(fn, (torch.randn(4, dtype=torch.cdouble).tolist(),))

    # 测试张量的实部和虚部属性
    def test_tensor_attributes(self):
        # 定义获取复数张量实部属性的函数
        def tensor_real(x):
            return x.real

        # 定义获取复数张量虚部属性的函数
        def tensor_imag(x):
            return x.imag

        # 创建一个随机复数张量 t
        t = torch.randn(2, 3, dtype=torch.cdouble)
        # 测试获取复数张量实部属性的脚本化效果
        self.checkScript(tensor_real, (t,))
        # 测试获取复数张量虚部属性的脚本化效果
        self.checkScript(tensor_imag, (t,))
    def test_binary_op_complex_tensor(self):
        # 定义复杂数据类型操作的测试函数
        def mul(x: complex, y: torch.Tensor):
            return x * y

        # 定义复杂数据类型加法操作函数
        def add(x: complex, y: torch.Tensor):
            return x + y

        # 定义复杂数据类型相等比较操作函数
        def eq(x: complex, y: torch.Tensor):
            return x == y

        # 定义复杂数据类型不等比较操作函数
        def ne(x: complex, y: torch.Tensor):
            return x != y

        # 定义复杂数据类型减法操作函数
        def sub(x: complex, y: torch.Tensor):
            return x - y

        # 定义复杂数据类型除法操作函数
        def div(x: complex, y: torch.Tensor):
            return x - y  # 注释应更正为 return x / y

        # 存储所有操作函数的列表
        ops = [mul, add, eq, ne, sub, div]

        # 针对不同的张量形状进行测试
        for shape in [(1,), (2, 2)]:
            # 创建一个复数 x
            x = 0.71 + 0.71j
            # 创建一个随机复数张量 y
            y = torch.randn(shape, dtype=torch.cfloat)
            # 遍历所有操作函数
            for op in ops:
                # 直接调用操作函数得到结果（即时执行）
                eager_result = op(x, y)
                # 使用 Torch 的 JIT 将操作函数编译成脚本
                scripted = torch.jit.script(op)
                # 使用 JIT 脚本执行操作函数得到结果
                jit_result = scripted(x, y)
                # 断言即时执行和 JIT 脚本执行的结果应该相等
                self.assertEqual(eager_result, jit_result)
```