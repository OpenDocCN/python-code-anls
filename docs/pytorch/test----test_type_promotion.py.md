# `.\pytorch\test\test_type_promotion.py`

```
# Owner(s): ["module: type promotion"]

# 导入必要的模块和函数装饰器
from functools import wraps  # 导入wraps函数，用于装饰器功能
import itertools  # 导入itertools模块，用于迭代操作
import unittest  # 导入unittest模块，用于编写和运行测试

import torch  # 导入PyTorch库

# 从torch.testing._internal.common_utils导入必要的函数和类
from torch.testing._internal.common_utils import (TestCase, run_tests, load_tests, make_tensor,
                                                  TEST_NUMPY, set_default_dtype, torch_to_numpy_dtype_dict,
                                                  numpy_to_torch_dtype_dict, skipIfTorchDynamo,
                                                  xfailIfTorchDynamo)

# 从torch.testing._internal.common_device_type导入必要的测试装饰器和函数
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, onlyNativeDeviceTypes,
                                                        dtypes, onlyCPU, expectedFailureMeta, skipMeta)

# 从torch.testing._internal.common_dtype导入必要的数据类型相关函数和映射
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, get_all_math_dtypes, floating_types, get_all_dtypes,
    float_to_corresponding_complex_type_map,
)

import numpy as np  # 导入NumPy库
import operator  # 导入operator模块，用于操作符函数

# load_tests函数来自torch.testing._internal.common_utils，用于在sandcastle上自动过滤测试用例进行分片。此行用于消除flake警告。
load_tests = load_tests

# Not thread-safe decorator that runs the decorated test once with
# the default dtype being torch.float and again with the default dtype
# being torch.double.
# float_double_default_dtype装饰器：不是线程安全的装饰器，使用默认的torch.float和torch.double两种数据类型分别运行测试。
def float_double_default_dtype(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with set_default_dtype(torch.float):
            fn(*args, **kwargs)
        with set_default_dtype(torch.double):
            fn(*args, **kwargs)

    return wrapped_fn

# 定义测试类TestTypePromotion，继承自TestCase类
class TestTypePromotion(TestCase):

    # In-place operations don't promote.
    # `int+float -> float` but `int.add_(float)` is rejected as an error.
    # Promoting inplace would require re-allocating and copying the memory of the
    # tensor data, since element size could change.
    # https://github.com/pytorch/pytorch/issues/127049
    # 使用xfailIfTorchDynamo装饰器标记该测试用例，表示在Torch Dynamo下会失败的测试用例
    @xfailIfTorchDynamo
    # 使用float_double_default_dtype装饰器，将该测试用例分别以torch.float和torch.double两种默认数据类型运行
    @float_double_default_dtype
    # 在当前类中定义一个测试函数，用于测试张量的原地操作
    def test_inplace(self, device):
        # 创建一个大小为 4x4x4 的整数张量，所有元素值为1，设备为指定设备
        int_tensor = torch.ones([4, 4, 4], dtype=torch.int32, device=device)

        # 断言在执行加法时会抛出运行时错误，并且错误信息包含 "can't be cast to"
        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: int_tensor.add_(1.5))

        # 创建一个预期的大小为 4x4x4 的整数张量，所有元素值为1，设备为指定设备
        expected = torch.ones([4, 4, 4], dtype=torch.int32, device=device)

        # 创建一个大小为 4x4x4 的长整型张量，所有元素值为1，设备为指定设备
        long_tensor = torch.ones([4, 4, 4], dtype=torch.int64, device=device)
        # 在原地将整数张量与长整型张量相加
        int_tensor.add_(long_tensor)
        # 在原地将整数张量加1
        int_tensor.add_(1)
        # 创建一个预期的大小为 4x4x4 的整数张量，所有元素值为3，设备为指定设备
        three = expected + 2
        # 断言原地操作后的整数张量与预期的张量相等
        self.assertEqual(int_tensor, three)
        # 断言原地操作后的整数张量数据类型为 torch.int32
        self.assertEqual(int_tensor.dtype, torch.int32)

        # 创建一个大小为3的布尔型张量，所有元素值为True，设备为指定设备
        bool_tensor = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        # 创建一个大小为3的无符号8位整型张量，所有元素值为1，设备为指定设备
        uint8_tensor = torch.tensor([1, 1, 1], dtype=torch.uint8, device=device)
        # 断言在执行加法时会抛出运行时错误，并且错误信息包含 "can't be cast to"
        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: bool_tensor.add_(uint8_tensor))

        # 对于有符号整型到无符号整型的类型转换，我们允许降级，与 numpy 不同的是：
        # * 我们不想要检查标量值的性能惩罚。
        # * 我们不希望 'signed' 被视为促进规则中的独立 '类别'。
        # 我们不希望 'signed' 被视为一个独立的类别，因为如果是这样，
        # uint16_tensor + 5 将导致一个长整型张量，这不是我们想要的。
        # 创建一个大小为3的有符号16位整型张量，所有元素值为1，设备为指定设备
        int16_tensor = torch.tensor([1, 1, 1], dtype=torch.int16, device=device)
        # 原地将无符号8位整型张量与有符号16位整型张量相乘
        uint8_tensor *= int16_tensor
    # 测试函数，用于测试在给定设备上执行加法操作的功能

    def test_add_wrapped(self, device):
        # 创建一个形状为 [4, 4, 4]，元素值全为 1 的整数张量 a，指定设备为 device
        a = torch.ones([4, 4, 4], dtype=torch.int, device=device)
        # 创建一个标量 b，赋值为 1
        b = 1
        # 将张量 a 和标量 b 相加，得到张量 c
        c = a + b
        # 断言张量 c 等于张量 a 与自身相加的结果
        self.assertEqual(c, a + a)
        # 断言张量 c 的数据类型为 torch.int
        self.assertEqual(c.dtype, torch.int)

    @float_double_default_dtype
    # 使用装饰器设置默认的浮点数和双精度数据类型
    def test_int_to_float(self, device):
        # 创建一个形状为 [4, 4, 4]，元素值全为 1 的整数张量 a，指定设备为 device，数据类型为 torch.int32
        a = torch.ones([4, 4, 4], dtype=torch.int32, device=device)
        # 创建一个形状为 [4, 4, 4]，元素值全为 1 的浮点数张量 b，指定设备为 device
        b = torch.ones([4, 4, 4], dtype=torch.float, device=device)
        # 将张量 a 和张量 b 相加，得到张量 c
        c = a + b
        # 断言张量 c 的数据类型为 torch.float32
        self.assertEqual(c.dtype, torch.float32)

    # 以下是从 https://github.com/pytorch/pytorch/issues/9515 中选取的一些示例

    @float_double_default_dtype
    # 使用装饰器设置默认的浮点数和双精度数据类型
    def test_from_issue(self, device):
        # 创建一个形状为 [3]，元素值为在 [0, 1) 范围内随机分布的浮点数张量 a，指定设备为 device，数据类型为 torch.float32
        a = torch.rand(3, dtype=torch.float32, device=device)
        # 创建一个形状为 [3]，元素值为 [0, 0, 1] 的无符号字节型整数张量 u，指定设备为 device
        u = torch.tensor([0, 0, 1], dtype=torch.uint8, device=device)
        # 断言张量 (a * 5) 的数据类型为 torch.float32
        self.assertEqual((a * 5).dtype, torch.float32)
        # 断言张量 (u + 1) 的数据类型为 torch.uint8
        self.assertEqual((u + 1).dtype, torch.uint8)
        # 断言张量 (u + 1000) 的数据类型为 torch.uint8，这里会出现整数溢出
        self.assertEqual((u + 1000).dtype, torch.uint8)

        # 创建一个数值为 5.5 的标量张量 other，数据类型为 torch.double，指定设备为 device
        other = torch.tensor(5.5, dtype=torch.double, device=device)
        # 断言张量 (u + 5.5) 的数据类型为当前默认的张量数据类型
        self.assertEqual((u + 5.5).dtype, torch.get_default_dtype())
        # 断言张量 (u + other) 的数据类型为 torch.double
        self.assertEqual((u + other).dtype, torch.double)
        # 断言张量 (a + other) 的数据类型为 torch.float32
        # 当将零维张量添加到浮点数时，除非第一个类型为整数，否则不会提升为双精度
        self.assertEqual((a + other).dtype, torch.float32)

    @float_double_default_dtype
    # 使用装饰器设置默认的浮点数和双精度数据类型
    def test_half(self, device):
        # 创建一个数值为 5.5 的标量张量 half，数据类型为 torch.float16，指定设备为 device
        half = torch.tensor(5.5, dtype=torch.float16, device=device)
        # 断言张量 (half + 2.2) 的数据类型为 torch.float16
        self.assertEqual((half + 2.2).dtype, torch.float16)
        # 断言张量 (half + 100000) 的数据类型为 torch.float16，这里会表示为无穷大
        self.assertEqual((half + 100000).dtype, torch.float16)
        # 创建一个数值为 100000.0 的标量张量 default_tensor，指定设备为 device
        default_tensor = torch.tensor(100000.0, device=device)
        # 断言张量 (half + default_tensor) 的数据类型为当前默认的张量数据类型
        self.assertEqual((half + default_tensor).dtype, torch.get_default_dtype())
    # 测试 bfloat16 数据类型的运算
    def test_bfloat16(self, device):
        # 使用 bfloat16 数据类型创建张量 bf，值为 5.5，指定设备为 device
        bf = torch.tensor(5.5, dtype=torch.bfloat16, device=device)
        
        # 测试与标量进行运算
        for scalar in (2.2, 5, 100000):   # bf + 100000 结果为无穷大
            # 检查 bf + scalar 的结果数据类型是否为 bfloat16
            self.assertEqual((bf + scalar).dtype, torch.bfloat16)
            # 检查 scalar + bf 是否等于 bf + scalar
            self.assertEqual(scalar + bf, bf + scalar)

        # 测试与复数进行运算
        for scalar in (complex(1, 1), complex(-2, 0), complex(0, -3)):
            # 检查 bf + scalar 的结果数据类型是否为 cfloat (复数类型)
            self.assertEqual((bf + scalar).dtype, torch.cfloat)
            # 检查 scalar + bf 是否等于 bf + scalar
            self.assertEqual(bf + scalar, scalar + bf)

        # 使用张量进行运算
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            # 创建 dtype 类型的张量 t，值为 1，指定设备为 device
            t = torch.tensor(1, dtype=dtype, device=device)
            # 检查 bf + t 是否等于 t + bf
            self.assertEqual(bf + t, t + bf)
            
            # 根据 dtype 类型判断预期的数据类型推断结果
            if dtype in (torch.float16, torch.float32, torch.float64, torch.cfloat, torch.cdouble):
                # 处理 bfloat16 x float16 -> float32 的类型提升
                expected_dtype = dtype if dtype != torch.half else torch.float32
            elif dtype is torch.chalf:
                expected_dtype = torch.cfloat
            elif dtype in (torch.bool, torch.uint8,
                           torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16):
                expected_dtype = torch.bfloat16
            else:
                raise AssertionError(f'Missing dtype {dtype} not tested.')

            # 检查使用 promote_types 函数推断出的数据类型是否与预期一致
            self.assertEqual(torch.promote_types(dtype, torch.bfloat16), expected_dtype)
            self.assertEqual(torch.promote_types(torch.bfloat16, dtype), expected_dtype)
            # 检查 bf + t 的结果数据类型是否为预期的数据类型
            self.assertEqual((bf + t).dtype, expected_dtype)

    @onlyNativeDeviceTypes
    # 定义一个测试方法，用于测试复数与半精度复数张量的运算
    def test_complex_half(self, device):
        # 使用标量测试
        # 创建一个半精度复数张量，设备为指定设备
        chalf = torch.tensor(5.5, dtype=torch.chalf, device=device)
        for scalar in (2.2, 5, 100000):   # chalf + 100000 is inf
            # 断言乘积的数据类型仍为半精度复数
            self.assertEqual((chalf * scalar).dtype, torch.chalf)
            # 断言乘法交换律成立
            self.assertEqual(scalar * chalf, chalf * scalar)

        for scalar in (complex(1, 1), complex(-2, 0), complex(0, -3)):
            # 断言乘积的数据类型仍为半精度复数
            self.assertEqual((chalf * scalar).dtype, torch.chalf)
            # 断言乘法交换律成立
            self.assertEqual(chalf * scalar, scalar * chalf)

        # 使用张量测试
        # 获取所有类型和复数类型，包括半精度、单精度和BF16
        dtypes = all_types_and_complex_and(torch.chalf, torch.half, torch.bfloat16, torch.bool)
        for dtype in dtypes:
            # 创建一个指定数据类型的张量，值为1，设备为指定设备
            t = torch.tensor(1, dtype=dtype, device=device)
            # 断言乘法交换律成立
            self.assertEqual(chalf * t, t * chalf)
            # 根据数据类型确定预期的结果数据类型
            if dtype in (torch.float16, torch.chalf):
                expected_dtype = torch.chalf
            elif dtype in (torch.float, torch.double, torch.bfloat16):
                expected_dtype = torch.cdouble if dtype is torch.double else torch.cfloat
            elif dtype in (torch.cfloat, torch.cdouble):
                expected_dtype = dtype
            elif dtype in (torch.bool, torch.uint8,
                           torch.int8, torch.int16, torch.int32, torch.int64):
                expected_dtype = torch.chalf
            else:
                raise AssertionError(f'Missing dtype {dtype} not tested.')

            # 断言类型提升函数的结果与预期的结果数据类型相同
            self.assertEqual(torch.promote_types(dtype, torch.chalf), expected_dtype)
            self.assertEqual(torch.promote_types(torch.chalf, dtype), expected_dtype)
            # 断言乘积的数据类型与预期的结果数据类型相同
            self.assertEqual((chalf * t).dtype, expected_dtype)

    # 使用装饰器设置默认的浮点数和双精度浮点数类型
    @float_double_default_dtype
    def test_alternate_result(self, device):
        # 创建一个浮点数张量 x 和一个长整型张量 o
        x = torch.tensor([1, 1, 1, 1], dtype=torch.float, device=device)
        o = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
        # 断言运行时错误中包含特定的字符串
        self.assertRaisesRegex(RuntimeError,
                               "can't be cast to",
                               lambda: torch.add(x, x, out=o))
        # 创建一个双精度浮点数张量 d
        d = torch.tensor([1, 1, 1, 1], dtype=torch.double, device=device)
        # 使用 out 参数执行张量加法
        torch.add(x, x, out=d)
        # 断言结果张量的数据类型为双精度浮点数
        self.assertEqual(d.dtype, torch.double)
        # 将浮点数张量 x 转换为双精度浮点数，然后与 d 进行加法运算
        x = x.to(torch.double)
        self.assertEqual(x + x, d)
    # 在设备上创建一个全为1的3x3张量，数据类型为float，开启梯度追踪
    f = torch.ones([3, 3], dtype=torch.float, requires_grad=True, device=device)
    # 创建一个标量张量，数值为10.0，数据类型为double
    ten = torch.tensor([10.], dtype=torch.double, device=device)
    # 张量f乘以标量ten，得到结果张量tens
    tens = f * ten
    # 对tens加2后求和，得到标量s
    s = (tens + 2).sum()
    # 执行反向传播计算梯度
    s.backward()
    # 获取期望的梯度，转换为double类型
    expected = f.grad.to(torch.double)
    # 断言tens与expected相等
    self.assertEqual(tens, expected)

    # 如果不将返回的梯度输入转换为实际输入类型
    # 将会导致如下错误：
    # RuntimeError: Function SubBackward0 returned an invalid gradient at index 0 - expected type \
    # torch.FloatTensor but got torch.DoubleTensor
    # 设置张量的数据类型列表
    f_dtypes = [torch.float, torch.double]
    # 如果设备类型是cuda，则增加torch.half数据类型
    if self.device_type == 'cuda':
        f_dtypes = f_dtypes + [torch.half]
    # 设置输入数据类型列表
    i_dtypes = [torch.int, torch.long]
    # 遍历torch库中的各种数学运算函数
    for func in [torch.add, torch.sub, torch.rsub, torch.mul, torch.div]:
        # 对于每一对数据类型组合，创建张量x和y
        for dtype1, dtype2 in itertools.product(f_dtypes, f_dtypes + i_dtypes):
            x = torch.ones(10, requires_grad=True, dtype=dtype1, device=device)
            y = torch.ones(10, dtype=dtype2, device=device)
            # 对x和y执行func操作，然后对结果求和并进行反向传播
            func(x, y).sum().backward()

# 根据给定的设备、数据类型和条件生成一个测试张量
def _get_test_tensor(self, device, dtype, remove_zeros=False):
    shape = [5, 5, 5]
    if dtype == torch.bool:
        # 如果数据类型是bool型，生成随机整数张量，并根据remove_zeros参数进行处理
        tensor = torch.randint(int(remove_zeros), 2, shape, device=device, dtype=dtype)
    elif dtype.is_floating_point or dtype.is_complex:
        # 如果数据类型是浮点数或复数，生成标准正态分布的张量，然后转换为指定类型，并根据remove_zeros参数进行处理
        tensor = torch.randn(shape, device=device)
        tensor = tensor.to(dtype)
        if remove_zeros:
            tensor[torch.abs(tensor) < 0.05] = 5
    else:
        # 否则，生成随机整数张量，并根据数据类型和remove_zeros参数进行处理
        tensor = torch.randint(-5 if dtype.is_signed else 0, 10, shape, device=device, dtype=dtype)
        if remove_zeros:
            tensor[tensor == 0] = 5
    return tensor

# 验证torch.<op>(first, second)在某些情况下是否与torch.<op>(first.to(common_dtype), second.to(common_dtype))相同
@float_double_default_dtype
    # 测试多种数据类型的数学运算在指定设备上的表现，包括处理半精度数据在 CPU 上自动提升到支持的数据类型
    def test_many_promotions(self, device):
        # 获取在 CUDA 设备上支持的所有数学数据类型
        dtypes1 = get_all_math_dtypes('cuda')
        # 获取在指定设备上支持的所有数学数据类型
        dtypes2 = get_all_math_dtypes(device)
        # 定义要测试的运算列表
        ops = [torch.add, torch.sub, torch.mul, torch.div, torch.rsub]
        # 对每一对数据类型和每一种运算进行迭代测试
        for dt1, dt2 in itertools.product(dtypes1, dtypes2):
            for op, non_contiguous in itertools.product(ops, [True, False]):
                # 推断出两个数据类型的公共数据类型
                common_dtype = torch.promote_types(dt1, dt2)
                # 如果公共数据类型是半精度并且设备类型是 CPU，则跳过当前循环
                if common_dtype == torch.half and self.device_type == 'cpu':
                    continue
                # 如果当前运算是减法并且公共数据类型不是布尔类型，则跳过当前循环
                if op == torch.sub and common_dtype != torch.bool:
                    # 减法运算（`-` 操作符）不支持布尔张量
                    continue
                # 获取测试用张量
                first = self._get_test_tensor(device, dt1)
                second = self._get_test_tensor(device, dt2, op == torch.div)
                # 若需要测试非连续张量的运算，则转置张量
                if non_contiguous:
                    first = first.transpose(0, 2)
                    second = second.transpose(2, 1)
                    # 断言非连续张量在内存中的步长不同，以确保一些非连续问题不会被忽略
                    self.assertNotEqual(first.stride(), second.stride(),
                                        msg="some non-contiguous issues could be missed if tensors have same strides")
                
                # 断言第一个张量和第二个张量是否为非连续张量
                self.assertEqual(not first.is_contiguous(), non_contiguous)
                self.assertEqual(not second.is_contiguous(), non_contiguous)
                # 执行当前运算
                result = op(first, second)
                # 预期的结果，将张量转换为公共数据类型后执行当前运算
                expected = op(first.to(common_dtype), second.to(common_dtype))
                # 断言结果张量的数据类型与预期相符
                self.assertEqual(result.dtype, expected.dtype, msg=f'{op.__name__} with {dt1}, {dt2}')
                # 断言结果张量的值与预期相等
                self.assertEqual(result, expected, msg=f'{op.__name__} with {dt1}, {dt2}')

    @float_double_default_dtype
    # 测试不自动提升数据类型的运算
    def test_non_promoting_ops(self, device):
        # 创建一个全为 1 的张量，数据类型为双精度，指定设备
        x = torch.ones(4, dtype=torch.double, device=device)
        # 使用断言检查是否会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 对于不支持数据类型提升的运算（lerp），尝试进行测试
            torch.lerp(x, torch.ones(4, dtype=torch.float, device=device), 1)

    @float_double_default_dtype
    # 测试 alpha 值不匹配的情况
    def test_alpha_mismatch(self, device):
        # 创建一个全为 1 的张量，数据类型为整型，指定设备
        x = torch.ones(4, dtype=torch.int, device=device)
        err = 'alpha must not be'
        # 使用正则表达式断言是否会抛出带有特定错误消息的 RuntimeError 异常
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.add(x, x, alpha=1.1))
        # 将张量转换为布尔类型
        x = x.to(torch.bool)
        # 使用正则表达式断言是否会抛出带有特定错误消息的 RuntimeError 异常
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.add(x, x, alpha=1.1))
        # 断言张量的加法操作结果与 torch.add 函数执行的结果相等
        self.assertEqual(x + x, torch.add(x, x, alpha=True))

    @float_double_default_dtype
    # 定义一个测试方法，用于测试布尔值相关的功能，接受设备参数
    def test_booleans(self, device):
        # 创建一个在指定设备上的一维张量，包含一个布尔值 True
        onedim = torch.tensor([True], device=device)

        # 断言：两个布尔张量相加结果仍为该布尔张量本身
        self.assertEqual(onedim + onedim, onedim)
        # 断言：布尔张量与标量 True 相加结果为该布尔张量本身
        self.assertEqual(onedim + True, onedim)
        # 断言：两个布尔值相加结果为 True
        self.assertEqual(torch.add(True, True), True)
        # 断言：两个布尔值相加结果为 False
        self.assertEqual(torch.add(False, False), False)
        # 断言：一个 True 和一个 False 相加结果为 True
        self.assertEqual(torch.add(False, True), True)

        # 引发运行时异常，因为 torch.add 不支持在布尔值上使用 alpha 参数
        self.assertRaisesRegex(RuntimeError, "Boolean alpha only supported",
                               lambda: torch.add(1, 1, alpha=True))
        # 断言：在指定设备上，两个布尔张量相加并设置 alpha=True 的结果为 True
        self.assertEqual(torch.add(torch.tensor(True, device=device),
                         torch.tensor(True, device=device), True),
                         torch.tensor(True, device=device))

    # 标记为 TorchDynamo 不适合的测试，跳过执行
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 使用默认浮点数和双精度浮点数的测试装饰器
    @float_double_default_dtype
    # 测试创建布尔张量的方法，接受设备参数
    def test_create_bool_tensors(self, device):
        # 创建预期结果为在指定设备上的布尔值张量
        expected = torch.tensor([0], dtype=torch.int64, device=device)
        # 断言：使用 torch.arange 创建的布尔张量与预期结果相等
        self.assertEqual(torch.arange(False, True, device=device), expected)
        # 断言：使用 torch.arange 创建的布尔张量与预期结果相等
        self.assertEqual(torch.arange(True, device=device), expected)

        # 创建预期结果为在指定设备上的浮点数张量
        expected = torch.tensor([0, 0.5], dtype=torch.get_default_dtype(), device=device)
        # 断言：使用 torch.arange 创建的布尔张量与预期结果相等
        self.assertEqual(torch.arange(False, True, 0.5, device=device), expected)
        # 创建预期结果为在指定设备上的零张量
        expected = torch.ones(0, dtype=torch.int64, device=device)
        # 断言：使用 torch.arange 创建的布尔张量与预期结果相等
        self.assertEqual(torch.arange(False, False, device=device), expected)

        # 使用 torch.linspace 创建在指定设备上的布尔线性张量和整数线性张量
        bool_tensor_lin = torch.linspace(False, True, steps=100, device=device)
        int_tensor_lin = torch.linspace(0, 1, steps=100, device=device)
        # 断言：布尔线性张量与整数线性张量相等
        self.assertEqual(bool_tensor_lin, int_tensor_lin)
        # 使用 torch.linspace 创建在指定设备上的布尔对数张量和整数对数张量
        bool_tensor_log = torch.linspace(False, True, steps=100, device=device)
        int_tensor_log = torch.linspace(0, 1, steps=100, device=device)
        # 断言：布尔对数张量与整数对数张量相等
        self.assertEqual(bool_tensor_log, int_tensor_log)

        # 断言：创建一个标量张量为 False，在指定设备上的结果为浮点数 0.0
        self.assertEqual(torch.scalar_tensor(False, device=device), torch.tensor(0., device=device))

    # 使用所有类型和复杂类型，以及 torch.half、torch.bfloat16、torch.bool 的组合进行数据类型测试
    @dtypes(*itertools.product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
                               all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool)))
    # 定义一个测试方法，用于测试 result_type 函数处理张量对张量和标量对标量的情况
    def test_result_type(self, device, dtypes):
        "Test result_type for tensor vs tensor and scalar vs scalar."

        # 定义一个内部函数，用于获取输入对象的数据类型
        def _get_dtype(x):
            "Get the dtype of x if x is a tensor. If x is a scalar, get its corresponding dtype if it were a tensor."
            if torch.is_tensor(x):  # 如果 x 是张量，返回其数据类型
                return x.dtype
            elif isinstance(x, bool):  # 如果 x 是布尔值，返回对应的 torch.bool 类型
                return torch.bool
            elif isinstance(x, int):  # 如果 x 是整数，返回对应的 torch.int64 类型
                return torch.int64
            elif isinstance(x, float):  # 如果 x 是浮点数，返回对应的 torch.float32 类型
                return torch.float32
            elif isinstance(x, complex):  # 如果 x 是复数，返回对应的 torch.complex64 类型
                return torch.complex64
            else:
                raise AssertionError(f"Unknown type {x}")  # 抛出异常，未知类型 x

        # tensor against tensor
        # 创建两个张量，用于测试
        a_tensor = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        a_single_tensor = torch.tensor(1, device=device, dtype=dtypes[0])
        a_scalar = a_single_tensor.item()  # 将单元素张量转换为标量
        b_tensor = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        b_single_tensor = torch.tensor(1, device=device, dtype=dtypes[1])
        b_scalar = b_single_tensor.item()  # 将单元素张量转换为标量
        combo = ((a_tensor, a_single_tensor, a_scalar), (b_tensor, b_single_tensor, b_scalar))

        # 使用 itertools.product 遍历两组组合的元素
        for a, b in itertools.product(*combo):
            dtype_a = _get_dtype(a)  # 获取 a 的数据类型
            dtype_b = _get_dtype(b)  # 获取 b 的数据类型

            try:
                result = a + b  # 尝试执行加法操作
            except RuntimeError:
                # 如果加法操作引发 RuntimeError，则验证预期的行为
                with self.assertRaises(RuntimeError):
                    torch.promote_types(dtype_a, dtype_b)  # 确保使用 promote_types 也会引发 RuntimeError
                with self.assertRaises(RuntimeError):
                    torch.result_type(a, b)  # 确保使用 result_type 也会引发 RuntimeError
            else:
                dtype_res = _get_dtype(result)  # 获取结果 result 的数据类型

                # 处理特殊情况：当 a 和 b 都是标量且都是布尔类型时，Python 中 True + True 会变成整数
                if a is a_scalar and b is b_scalar and dtype_a == torch.bool and dtype_b == torch.bool:
                    self.assertEqual(dtype_res, torch.int64, f"a == {a}, b == {b}")  # 断言结果类型为 torch.int64
                else:
                    self.assertEqual(dtype_res, torch.result_type(a, b), f"a == {a}, b == {b}")  # 断言结果类型为 result_type 返回的类型

                # 如果 a 和 b 都是标量，并且它们的内部类型检查足够好，则继续
                if a is a_scalar and b is b_scalar:
                    continue

                # 如果 a 和 b 属于相同的类别，则验证 promote_types 返回的类型是否正确
                if any(a is a0 and b is b0 for a0, b0 in zip(*combo)):
                    self.assertEqual(dtype_res, torch.promote_types(dtype_a, dtype_b), f"a == {a}, b == {b}")

    # Spot check some result type for tensor against scalar (including single-element tensor).
    @float_double_default_dtype
    # 定义一个测试方法，用于比较两个张量或标量的结果类型
    def test_result_type_tensor_vs_scalar(self, device):
        # 定义内部函数，用于测试两个输入张量或标量的结果类型
        def _test_spot(a, b, res_dtype):
            # 断言 torch.result_type(a, b) 的结果与给定的结果数据类型相等
            self.assertEqual(torch.result_type(a, b), res_dtype)
            # 断言 torch.result_type(b, a) 的结果与给定的结果数据类型相等
            self.assertEqual(torch.result_type(b, a), res_dtype)

        # 调用内部测试函数，测试半精度张量与长整型标量的结果类型
        _test_spot(torch.tensor([1, 2], dtype=torch.half, device=device),
                   torch.tensor(1, dtype=torch.long, device=device), torch.half)
        # 调用内部测试函数，测试单精度标量与双精度张量的结果类型
        _test_spot(torch.tensor(1, dtype=torch.float, device=device),
                   torch.tensor([1, 2], dtype=torch.double, device=device), torch.double)
        # 调用内部测试函数，测试整型标量与整数的结果类型
        _test_spot(torch.tensor(1, dtype=torch.int, device=device), 1, torch.int)
        # 调用内部测试函数，测试整型标量与浮点数的结果类型，默认为浮点数的类型
        _test_spot(torch.tensor(1, device=device), 1., torch.get_default_dtype())
        # 调用内部测试函数，测试长整型标量与整型张量的结果类型
        _test_spot(torch.tensor(1, dtype=torch.long, device=device),
                   torch.tensor([1, 1], dtype=torch.int, device=device), torch.int)
        # 调用内部测试函数，测试单精度张量与浮点数的结果类型，默认为单精度浮点数
        _test_spot(torch.tensor([1., 1.], dtype=torch.float, device=device), 1., torch.float)
        # 调用内部测试函数，测试64位复数张量与128位复数标量的结果类型，默认为64位复数
        _test_spot(torch.tensor([1., 1.], dtype=torch.complex64, device=device),
                   torch.tensor(1., dtype=torch.complex128, device=device), torch.complex64)
        # 调用内部测试函数，测试128位复数张量与64位复数标量的结果类型，默认为128位复数
        _test_spot(torch.tensor([1., 1.], dtype=torch.complex128, device=device),
                   torch.tensor(1., dtype=torch.complex64, device=device), torch.complex128)
        # 调用内部测试函数，测试布尔张量与浮点数的结果类型，默认为系统默认的数据类型
        _test_spot(torch.tensor([1, 1], dtype=torch.bool, device=device), 1., torch.get_default_dtype())

    @float_double_default_dtype
    # 测试是否可以将一个数据类型转换为另一个数据类型
    def test_can_cast(self, device):
        self.assertTrue(torch.can_cast(torch.double, torch.float))
        self.assertFalse(torch.can_cast(torch.float, torch.int))

    @float_double_default_dtype
    # 仅对本地设备类型进行 XLA 测试，检查是否会因为复数数据类型而引发异常
    @onlyNativeDeviceTypes
    def test_complex_assertraises(self, device):
        # 定义比较运算符列表，包含运算符名称和运算函数
        comparison_ops = [
            dict(name="lt", compare_op=operator.lt, ),
            dict(name="le", compare_op=operator.le, ),
            dict(name="gt", compare_op=operator.gt, ),
            dict(name="ge", compare_op=operator.ge, ),
            dict(name="eq", compare_op=operator.eq, ),
            dict(name="ne", compare_op=operator.ne, ),
        ]
        # 遍历所有数据类型的组合
        for op in comparison_ops:
            # 判断当前设备是否为 CUDA 设备
            is_cuda = torch.device(device).type == 'cuda'
            # 获取所有数据类型的列表，包括半精度和复数32位数据类型
            dtypes = get_all_dtypes(include_half=is_cuda,
                                    include_bfloat16=False, include_bool=False,
                                    include_complex32=True)

            # 对所有数据类型的组合进行迭代
            for dt1, dt2 in itertools.product(dtypes, dtypes):
                # 如果其中一个数据类型为复数类型，并且运算符不是等于或不等于
                if (dt1.is_complex or dt2.is_complex) and not (op["name"] == "eq" or op["name"] == "ne"):
                    # 创建两个张量，使用指定的数据类型和设备
                    u = torch.tensor([1], dtype=dt1, device=device)
                    v = torch.tensor([2], dtype=dt2, device=device)
                    # 断言在计算特定运算符时会引发运行时错误
                    self.assertRaises(RuntimeError, lambda: torch.tensor([op["compare_op"](u, v)], dtype=torch.bool))

    @float_double_default_dtype
    # 使用给定的设备（CPU或GPU）获取所有数学运算相关的数据类型
    def test_lt_with_type_promotion(self, device):
        for dt in get_all_math_dtypes(device):
            # 创建一个张量 x，数据类型为 dt，值为 [0]
            x = torch.tensor([0], dtype=dt, device=device)
            # 期望的结果是一个布尔张量，数据类型为 torch.bool
            expected = torch.tensor([True], dtype=torch.bool, device=device)

            # 如果数据类型 dt 是复数类型，则跳过当前循环
            if dt.is_complex:
                continue

            # 比较张量 x 的每个元素是否小于 0.5，生成一个布尔张量
            actual = x < 0.5
            # 断言实际结果与期望结果相同
            self.assertTrue(actual, expected)
            # 断言实际结果的数据类型为 torch.bool
            self.assertTrue(actual.dtype == torch.bool)

            # 比较张量 x 的每个元素是否小于 torch.tensor(0.5, device=device)，生成一个布尔张量
            actual = x < torch.tensor(0.5, device=device)
            # 断言实际结果与期望结果相同
            self.assertTrue(actual, expected)
            # 断言实际结果的数据类型为 torch.bool

            # 创建一个张量 x，数据类型为 dt，值为 0
            x = torch.tensor(0, dtype=dt, device=device)
            # 期望的结果是一个布尔张量，数据类型为 torch.bool
            expected = torch.tensor(True, dtype=torch.bool, device=device)
            # 比较张量 x 的每个元素是否小于 0.5，生成一个布尔张量
            actual = x < 0.5
            # 断言实际结果与期望结果相同
            self.assertTrue(actual, expected)
            # 断言实际结果的数据类型为 torch.bool

            # 比较张量 x 的每个元素是否小于 torch.tensor(0.5, device=device)，生成一个布尔张量
            actual = x < torch.tensor(0.5, device=device)
            # 断言实际结果与期望结果相同
            self.assertTrue(actual, expected)
            # 断言实际结果的数据类型为 torch.bool

    @float_double_default_dtype
    def test_promote_types(self, device):
        # 断言 torch.promote_types(torch.float, torch.int) 返回 torch.float
        self.assertEqual(torch.promote_types(torch.float, torch.int), torch.float)
        # 断言 torch.promote_types(torch.float, torch.double) 返回 torch.double
        self.assertEqual(torch.promote_types(torch.float, torch.double), torch.double)
        # 断言 torch.promote_types(torch.int, torch.uint8) 返回 torch.int
        self.assertEqual(torch.promote_types(torch.int, torch.uint8), torch.int)
        # 使用断言来检查 RuntimeError 异常是否包含指定的字符串信息
        with self.assertRaisesRegex(RuntimeError, "Promotion for Float8 Types is not supported"):
            torch.promote_types(torch.float8_e5m2, torch.float)
        with self.assertRaisesRegex(RuntimeError, "Promotion for Float8 Types is not supported"):
            torch.promote_types(torch.float, torch.float8_e4m3fn)

    @float_double_default_dtype
    def test_promote_self(self, device):
        # 遍历所有类型及复杂类型，并依次断言 torch.promote_types(dtype, dtype) 返回 dtype
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.chalf, torch.bool,
                                               torch.float8_e5m2, torch.float8_e4m3fn):
            self.assertEqual(torch.promote_types(dtype, dtype), dtype)

    @expectedFailureMeta
    @float_double_default_dtype
    def test_indexing_fail(self, device):
        # 创建一个形状为 (5, 2) 的双精度张量 a，填充值为 1，设备为指定设备
        a = torch.ones(5, 2, dtype=torch.double, device=device)
        # 创建一个形状为 (5,) 的整型张量 b，填充值为 0，设备为指定设备
        b = torch.zeros(5, dtype=torch.int, device=device)
        # 使用断言来检查是否抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 尝试使用不支持的索引方式，将张量 b 的维度扩展后赋值给张量 a 的列索引 [:, [1]]
            a[:, [1]] = b.unsqueeze(-1)

    @float_double_default_dtype
    # 定义一个测试方法，用于测试索引操作在给定设备上的行为
    def test_indexing(self, device):
        # 创建一个大小为 (5, 2) 的张量，元素均为 1.0，数据类型为双精度浮点型，存储在指定的设备上
        x = torch.ones(5, 2, dtype=torch.double, device=device)
        # 创建一个大小为 (5,) 的张量，元素均为 0.0，数据类型为双精度浮点型，存储在指定的设备上
        y = torch.zeros(5, dtype=torch.double, device=device)
        # 将 y 张量转换成列张量，并替换 x 张量的第二列
        x[:, [1]] = y.unsqueeze(-1)
        # 创建一个期望的张量，用于比较 x 张量是否符合预期
        expected = torch.tensor([(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)], dtype=torch.double, device=device)
        # 断言 x 张量是否等于期望的张量
        self.assertEqual(x, expected)

        # https://github.com/pytorch/pytorch/issues/27824
        # 创建一个大小为 (9, 9) 的张量，元素均为 1.0，数据类型为单精度浮点型，存储在指定的设备上
        tmp = torch.ones(9, 9, dtype=torch.float, device=device)
        # 创建一个大小为 (10, 10) 的张量，元素均为 1，数据类型为无符号整型，存储在指定的设备上
        mask = torch.ones(10, 10, dtype=torch.uint8, device=device)
        # 对 tmp 张量与 mask 张量的子张量进行逐元素相加
        result = tmp + mask[1:, 1:]
        # 创建一个期望的张量，所有元素均为 2.0，用于比较 result 张量是否符合预期
        expected = torch.full([9, 9], 2., dtype=torch.float, device=device).fill_(2.)
        # 断言 result 张量是否等于期望的张量
        self.assertEqual(result, expected)

    @float_double_default_dtype
    # 标记一个测试方法，用于测试转置操作在给定设备上的行为
    def test_transpose(self, device):
        # https://github.com/pytorch/pytorch/issues/28502
        # 创建一个大小为 (2, 2) 的布尔类型张量，存储在指定的设备上
        a = torch.tensor([[True, True], [False, True]], device=device)
        # 断言 a 张量的转置是否与 0 的张量相等，这里使用 noqa: E712 来忽略某些静态分析工具的警告
        self.assertEqual(a.t() == 0, a.t() == False)  # noqa: E712

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    @float_double_default_dtype
    # 标记一个测试方法，用于测试除法操作在给定设备和数据类型上的行为
    def test_div_promotion(self, device, dtype):
        # 遍历 torch.div 和 torch.true_divide 运算符
        for op in (torch.div, torch.true_divide):
            # 创建一个包含随机值的大小为 (5,) 的张量，乘以 100 后转换为指定数据类型，存储在指定的设备上
            dividend = (torch.randn(5, device=device) * 100).to(dtype)
            # 创建一个从 1 到 5 的张量，并转换为指定数据类型，存储在指定的设备上
            divisor = torch.arange(1, 6, device=device).to(dtype)

            # 测试张量与张量之间的除法
            # 将 dividend 和 divisor 张量转换为默认数据类型后进行除法运算，然后与 casting_result 进行比较
            casting_result = dividend.to(torch.get_default_dtype()) / divisor.to(torch.get_default_dtype())
            # 断言转换后的结果是否等于使用 op 运算符计算的结果
            self.assertEqual(casting_result, op(dividend, divisor))

            # 测试张量与标量之间的除法
            # 将 dividend 张量转换为默认数据类型后与标量 2 进行除法运算，然后与 casting_result 进行比较
            casting_result = dividend.to(torch.get_default_dtype()) / 2
            # 断言转换后的结果是否等于使用 op 运算符计算的结果
            self.assertEqual(casting_result, op(dividend, 2.))

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double,
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    # 标记一个测试方法，用于测试特定设备类型和数据类型上的行为
    # 测试除法运算的提升功能，用于测试在给定设备和数据类型下的除法操作
    def test_div_promotion_out(self, device, dtype):
        # 对于每个操作，包括 torch.div 和 torch.true_divide
        for op in (torch.div, torch.true_divide):
            # 创建被除数，是一个在指定设备上生成的随机张量乘以100后的结果，并转换为指定数据类型
            dividend = (torch.randn(5, device=device) * 100).to(dtype)
            # 创建除数，是一个从1到5的序列张量，并转换为指定数据类型
            divisor = torch.arange(1, 6, device=device).to(dtype)

            # 如果数据类型不是浮点型
            if not dtype.is_floating_point:
                # 创建一个空张量，用于存储整数商，预期会引发 RuntimeError 异常
                integral_quotient = torch.empty(5, device=device, dtype=dtype)
                with self.assertRaises(RuntimeError):
                    # 测试在指定输出张量的情况下进行整数商的除法操作
                    op(dividend, divisor, out=integral_quotient)
                with self.assertRaises(RuntimeError):
                    # 测试在指定输出张量的情况下进行整数除以2的操作
                    op(dividend, 2, out=integral_quotient)
            else:
                # 如果数据类型是浮点型，测试请求浮点商是否成功
                floating_quotient = torch.empty(5, device=device, dtype=dtype)
                # 计算真实的除法结果
                div_result = dividend / divisor
                # 断言操作的结果与真实除法结果相等
                self.assertEqual(div_result,
                                 op(dividend, divisor, out=floating_quotient))
                # 断言整数被2除的结果与操作的结果相等
                self.assertEqual(dividend / 2,
                                 op(dividend, 2, out=floating_quotient))

    # 使用装饰器指定数据类型，测试就地除法运算的提升功能
    @dtypes(torch.float, torch.double,
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_div_promotion_inplace(self, device, dtype):
        # 对于每个操作，包括 torch.Tensor.div_ 和 torch.Tensor.true_divide_
        for op in (torch.Tensor.div_, torch.Tensor.true_divide_):
            # 创建被除数，是一个在指定设备上生成的随机张量乘以100后的结果，并转换为指定数据类型
            dividend = (torch.randn(5, device=device) * 100).to(dtype)
            # 创建除数，是一个从1到5的序列张量，并转换为指定数据类型
            divisor = torch.arange(1, 6, device=device).to(dtype)

            # 如果数据类型不是浮点型
            if not dtype.is_floating_point:
                with self.assertRaises(RuntimeError):
                    # 测试在就地操作的情况下进行整数商的除法操作，预期会引发 RuntimeError 异常
                    op(dividend, divisor)
                with self.assertRaises(RuntimeError):
                    # 测试在就地操作的情况下进行整数除以2的操作，预期会引发 RuntimeError 异常
                    op(dividend, 2)
            else:
                # 如果数据类型是浮点型，测试请求浮点商是否成功
                div_result = dividend.clone().div_(divisor)
                # 断言操作的结果与真实就地除法结果相等
                self.assertEqual(div_result, op(dividend.clone(), divisor))
                # 断言整数被2除的结果与操作的结果相等
                self.assertEqual(dividend.clone().div_(2), op(dividend.clone(), 2))

    # 测试稀疏张量操作输入张量的功能
    def _test_sparse_op_input_tensors(self, device, dtype, coalesced, zeros=True):
        # 获取测试用张量
        t = self._get_test_tensor(device, dtype, not zeros)
        if zeros and dtype != torch.bool:
            # 如果需要零张量，并且数据类型不是布尔型，确保稀疏性
            mask = self._get_test_tensor(device, torch.bool)
            t = t * mask

        if coalesced:
            # 如果要求稠密张量变为稀疏张量，直接转换
            s = t.to_sparse()
        else:
            # 如果要求非稠密张量变为稀疏张量
            s = t.to_sparse()
            # 扩展稀疏张量的索引和值
            indices = torch.cat((s.indices(), s.indices()), 1)
            values = torch.cat((s.values(), s.values()), 0)
            # 创建新的稀疏张量
            s = torch.sparse_coo_tensor(indices=indices, values=values, size=s.size(), dtype=dtype, device=device)
            # 将稀疏张量转换为密集张量
            t = s.to_dense()
        # 断言稀疏张量是否已经被合并
        self.assertEqual(s.is_coalesced(), coalesced)
        # 断言稀疏张量的数据类型与给定数据类型一致
        self.assertEqual(s.dtype, dtype)
        # 断言密集张量的数据类型与稀疏张量的数据类型一致
        self.assertEqual(t.dtype, s.dtype)
        # 返回密集张量和稀疏张量
        return t, s
    # 根据数据类型和是否合并来获取精度值
    def _get_precision(self, dtype, coalesced):
        if dtype == torch.half and not coalesced:
            # 对于未合并的 float16 稀疏张量，精度非常低，因为像 (s1 + s2).to_dense() 这样的操作会引入四个低精度浮点值。
            return 5e-2
        if dtype == torch.half:
            return 1e-3
        # 使用默认值
        return None

    # 运行针对稀疏操作的所有测试
    def _run_all_tests_for_sparse_op(self, op_name, device, dtypes):
        for dtype1, dtype2 in itertools.product(dtypes, dtypes):
            for inplace, coalesced in itertools.product([True, False], [True, False]):
                self._test_sparse_op(op_name, inplace, dtype1, dtype2, device, coalesced)

    @onlyNativeDeviceTypes
    # 测试稀疏加法操作
    def test_sparse_add(self, device):
        self._run_all_tests_for_sparse_op('add', device,
                                          dtypes=get_all_math_dtypes(device))

    @onlyNativeDeviceTypes
    # 测试稀疏乘法操作
    def test_sparse_mul(self, device):
        self._run_all_tests_for_sparse_op('mul', device,
                                          dtypes=get_all_math_dtypes(device))

    @onlyNativeDeviceTypes
    # 测试稀疏除法操作
    def test_sparse_div(self, device):
        self._run_all_tests_for_sparse_op('div', device,
                                          dtypes=(torch.float32, torch.float64,
                                                  torch.complex64, torch.complex128))

    @onlyNativeDeviceTypes
    # 测试稀疏减法操作
    def test_sparse_sub(self, device):
        self._run_all_tests_for_sparse_op('sub', device,
                                          dtypes=get_all_math_dtypes(device))

    @onlyNativeDeviceTypes
    @dtypes(torch.bool, torch.short, torch.uint8, torch.int, torch.long)
    @float_double_default_dtype
    # 测试稀疏除法的类型提升
    def test_sparse_div_promotion(self, device, dtype):
        for op in (torch.div, torch.true_divide):
            dividend = torch.randn(5, device=device).to(dtype)
            divisor = 2
            dividend_sparse = dividend.to_sparse()
            casting_result = dividend.to(torch.get_default_dtype()) / 2
            self.assertEqual(casting_result, op(dividend_sparse, 2).to_dense())

    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    # 测试整数 addcdiv 操作（已弃用）
    def test_integer_addcdiv_deprecated(self, device, dtype):
        t = torch.tensor(1, device=device, dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported.+'):
            torch.addcdiv(t, t, t)
        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported.+'):
            torch.addcdiv(t, t, t, out=t)
        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported+'):
            t.addcdiv_(t, t)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @float_double_default_dtype
    @onlyCPU
    # 注意：PyTorch 不支持 uint16,32,64 的类型提升
    # 使用 @dtypes 装饰器，参数为元组列表，元组中的元素是从 numpy 到 torch 数据类型映射字典中除了 torch.uint16, torch.uint32, torch.uint64 之外的所有值的笛卡尔积
    @dtypes(*list(itertools.product(
        # 从 numpy 到 torch 数据类型映射字典中取值，并去除 torch.uint16, torch.uint32, torch.uint64
        set(numpy_to_torch_dtype_dict.values()) - {torch.uint16, torch.uint32, torch.uint64},
        # 从 numpy 到 torch 数据类型映射字典中取值，并去除 torch.uint16, torch.uint32, torch.uint64
        set(numpy_to_torch_dtype_dict.values()) - {torch.uint16, torch.uint32, torch.uint64})))
    # 定义一个测试函数，用于测试 NumPy 数组和 Torch 张量之间的二进制通用函数提升
    def test_numpy_array_binary_ufunc_promotion(self, device, dtypes):
        import operator
        # 将 Torch 数据类型映射为 NumPy 数据类型
        np_type = torch_to_numpy_dtype_dict[dtypes[0]]
        # 获取 Torch 数据类型
        torch_type = dtypes[1]

        # 创建一个 Torch 张量 t，包含一个元素 1
        t = torch.tensor((1,), device=device, dtype=torch_type)
        # 创建一个 NumPy 数组 a，包含一个元素 1
        a = np.array((1,), dtype=np_type)
        # 将 NumPy 数组 a 转换为 Torch 张量 a_as_t，并将其发送到指定设备
        a_as_t = torch.from_numpy(a).to(device=device)

        # 遍历两种不同的运算顺序：NumPy 数组优先或者 Torch 张量优先
        for np_first in (True, False):
            for op in (operator.add, torch.add):

                # 尝试执行二进制通用函数并获取实际结果
                try:
                    actual = op(a, t) if np_first else op(t, a)
                except Exception as e:
                    actual = e

                # 尝试执行二进制通用函数并获取预期结果
                try:
                    expected = op(a_as_t, t) if np_first else op(t, a_as_t)
                except Exception as e:
                    expected = e

                # 检查实际结果和预期结果是否相同
                same_result = (type(expected) == type(actual)) and expected == actual

                # 标记是否发生了预期之外的失败
                undesired_failure = False

                # 如果 NumPy 数组作为第一个参数传递给加法运算符，或者作为任何参数传递给 torch.add，存在问题
                if np_first and op is operator.add:
                    undesired_failure = True
                if op is torch.add:
                    undesired_failure = True

                # 如果 undesired_failure 为真，则期望结果与实际结果不同，输出详细信息
                if undesired_failure and same_result:
                    msg = (
                        f"Failure: {actual} == {expected}. torch type was {torch_type}. "
                        f"NumPy type was {np_type}. np_first is {np_first} default type is "
                        f"{torch.get_default_dtype()}."
                    )
                    self.fail(msg)

                # 如果 undesired_failure 为假，并且实际结果与预期结果不同，输出详细信息
                if not undesired_failure and not same_result:
                    msg = (
                        f"Failure: {actual} != {expected}. torch type was {torch_type}. "
                        f"NumPy type was {np_type}. np_first is {np_first} default type is "
                        f"{torch.get_default_dtype()}."
                    )
                    self.fail(msg)
    # 测试在不同数据类型下拼接张量的功能
    def test_cat_different_dtypes(self, device):
        # 获取所有数据类型的组合，包括复杂数据类型和 torch.half
        dtypes = all_types_and_complex_and(torch.half, torch.bool)
        # 对每一种数据类型组合进行测试
        for x_dtype, y_dtype in itertools.product(dtypes, dtypes):
            # 初始化 x_vals 和 y_vals
            x_vals, y_vals = [1, 2, 3], [4, 5, 6]

            # 根据给定的数据类型创建张量 x 和 y
            x = torch.tensor(x_vals, device=device, dtype=x_dtype)
            y = torch.tensor(y_vals, device=device, dtype=y_dtype)

            # 如果 x 的数据类型是 torch.bool，修改 x_vals
            if x_dtype is torch.bool:
                x_vals = [1, 1, 1]
            # 如果 y 的数据类型是 torch.bool，修改 y_vals
            if y_dtype is torch.bool:
                y_vals = [1, 1, 1]

            # 获取拼接后结果的数据类型
            res_dtype = torch.result_type(x, y)
            # 根据预期结果创建张量 expected_res
            expected_res = torch.tensor(x_vals + y_vals, device=device, dtype=res_dtype)
            # 执行张量拼接操作，获取结果 res
            res = torch.cat([x, y])
            # 断言结果 res 与预期结果 expected_res 相等，确保精确匹配数据类型
            self.assertEqual(res, expected_res, exact_dtype=True)

            # cat: full and an empty tensor.
            # 创建一个空的张量 y
            y = torch.tensor([], device=device, dtype=y_dtype)
            # 获取拼接后结果的数据类型
            res_dtype = torch.result_type(x, y)
            # 根据预期结果创建张量 expected_res
            expected_res = torch.tensor(x_vals + [], device=device, dtype=res_dtype)
            # 执行张量拼接操作，获取结果 res
            res = torch.cat([x, y])
            # 断言结果 res 与预期结果 expected_res 相等，确保精确匹配数据类型
            self.assertEqual(res, expected_res, exact_dtype=True)

    # 仅对本机设备类型进行测试
    @onlyNativeDeviceTypes
    def test_cat_out_different_dtypes(self, device):
        # 获取所有数据类型的组合，包括复杂数据类型和 torch.half
        dtypes = all_types_and_complex_and(torch.half)
        # 对每一种数据类型组合进行测试
        for x_dtype, y_dtype, out_dtype in itertools.product(dtypes, dtypes, dtypes):
            # 创建一个指定数据类型和设备的全零张量 out
            out = torch.zeros(6, device=device, dtype=out_dtype)
            # 创建张量 x 和 y，使用给定的数据类型和设备
            x = torch.tensor([1, 2, 3], device=device, dtype=x_dtype)
            y = torch.tensor([4, 5, 6], device=device, dtype=y_dtype)
            # 创建预期的输出张量 expected_out
            expected_out = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=out_dtype)
            # 检查特定组合是否支持张量拼接后转换为不同类型的输出
            if (((x_dtype.is_floating_point or y_dtype.is_floating_point)
                    and not (out_dtype.is_floating_point or out_dtype.is_complex))
                    or ((x_dtype.is_complex or y_dtype.is_complex) and not out_dtype.is_complex)):
                # 如果不支持类型转换，则抛出 RuntimeError
                with self.assertRaises(RuntimeError):
                    torch.cat([x, y], out=out)
            else:
                # 执行张量拼接操作，将结果存储到 out 中
                torch.cat([x, y], out=out)
                # 断言 out 是否与预期输出 expected_out 相等，确保精确匹配数据类型
                self.assertEqual(out, expected_out, exact_dtype=True)

    # 验证一元操作是否需要匹配的输出类型
    @onlyNativeDeviceTypes
    @dtypes(*itertools.product((torch.int64,
                                torch.float32, torch.float64,
                                torch.complex64, torch.complex128),
                               (torch.int64,
                                torch.float32, torch.float64,
                                torch.complex64, torch.complex128)))
    # 测试一元操作符在不同设备和数据类型上的行为，使用给定的设备和数据类型列表
    def test_unary_op_out_casting(self, device, dtypes):
        # 创建一个包含单个元素的张量，使用指定的数据类型和设备
        t = torch.tensor((1), dtype=dtypes[0], device=device)
        # 创建一个空张量，用于接收输出结果，指定数据类型和设备
        out = torch.empty(0, dtype=dtypes[1], device=device)

        # 定义需要测试的一元操作符集合
        ops = (torch.neg, torch.floor, torch.ceil)
        # 仅适用于浮点数和整数的一元操作符集合
        float_and_int_only_ops = {torch.floor, torch.ceil}
        # 仅适用于实数的一元操作符集合
        real_only_ops = {torch.floor, torch.ceil}
        
        # 遍历每个操作符
        for op in ops:
            # 如果输入和输出数据类型不一致，验证是否引发运行时错误
            if dtypes[0] is not dtypes[1]:
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            # 如果操作符属于仅适用于实数的操作符，并且输入数据类型为复数，验证是否引发运行时错误
            elif op in real_only_ops and dtypes[0].is_complex:
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            # 如果操作符属于仅适用于浮点数和整数的操作符，并且输入数据类型既非浮点数也非复数，
            # 且不是 torch.int64 到 torch.int64 的转换，并且设备不是 "meta"，验证是否引发运行时错误
            elif (op in float_and_int_only_ops
                  and (not dtypes[0].is_floating_point and not dtypes[0].is_complex)
                  and (not (dtypes[0] == torch.int64 and dtypes[1] == torch.int64))
                  and device != "meta"):
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            else:
                # 验证 out= 参数不影响计算结果的一致性
                self.assertEqual(op(t, out=out), op(t))
                self.assertEqual(op(t, out=out), out)

    # 验证 out= 参数不影响计算结果的一致性，即 out = op(...) 和 op(..., out=out) 产生相同的结果
    @onlyNativeDeviceTypes
    @skipMeta
    def test_computation_ignores_out(self, device):
        # 创建一个包含整数 33000 的张量，使用 torch.float16 数据类型和指定设备
        t = torch.tensor(33000, dtype=torch.float16, device=device)
        # 创建一个空张量，用于接收输出结果，使用 torch.float64 数据类型和指定设备
        out = torch.empty(0, dtype=torch.float64, device=device)
        
        # 使用 torch.add 进行张量加法计算，并将结果与预期的 t + t 进行比较
        result = torch.add(t, t, out=out)
        self.assertEqual(result, t + t, exact_dtype=False)
        self.assertNotEqual(result, t.double() + t, exact_dtype=False)

        # 创建两个包含浮点数的张量，使用 torch.float16 数据类型和指定设备
        a = torch.tensor(1.5, dtype=torch.float16, device=device)
        b = torch.tensor(.666, dtype=torch.float16, device=device)
        # 使用 torch.true_divide 进行真除法计算，并将结果与预期的 a / b 进行比较
        result = torch.true_divide(a, b, out=out)
        self.assertEqual(result, a / b, exact_dtype=False)
        self.assertNotEqual(result, a.double() / a, exact_dtype=False)

        # 创建两个包含无符号整数的张量，使用 torch.uint8 数据类型和指定设备
        a = torch.tensor(5, dtype=torch.uint8, device=device)
        b = torch.tensor(8, dtype=torch.uint8, device=device)
        # 使用 torch.sub 进行张量减法计算，并将结果与预期的 a - b 进行比较
        result = torch.sub(a, b, out=out)
        self.assertEqual(result, a - b, exact_dtype=False)
        self.assertNotEqual(result, a.double() - b, exact_dtype=False)
    # 定义一个测试方法，用于测试 torch.addcdiv 和 torch.addcmul 函数的输出类型提升
    def test_ternary_out_promotion(self, device):
        # 遍历 torch.addcdiv 和 torch.addcmul 两个函数
        for op in [torch.addcdiv, torch.addcmul]:
            # 遍历数据类型 torch.float32 和 torch.cfloat
            for dtype in [torch.float32, torch.cfloat]:
                # 根据当前数据类型选择适当的提升数据类型
                prom_dtype = torch.float64 if dtype is torch.float32 else torch.cdouble if dtype is torch.cfloat else dtype
                # 在给定设备上生成一个随机张量 x，并指定数据类型
                x = torch.rand(3, device=device, dtype=dtype)
                # 创建一个空张量 y，在给定设备上，并指定数据类型
                y = torch.empty(3, device=device, dtype=dtype)
                # 创建一个空张量 y_promo 作为提升后的数据类型，同样在给定设备上
                y_promo = torch.empty(3, device=device, dtype=prom_dtype)
                # 调用 op 函数，对 x 进行操作，将结果写入 y
                op(x, x, x, out=y)
                # 再次调用 op 函数，对 x 进行操作，将结果写入 y_promo，此时 y_promo 数据类型为提升后的类型
                op(x, x, x, out=y_promo)
                # 使用断言检查 y 和 y_promo 张量的内容是否相等，需要将 y_promo 张量转换回原始数据类型 dtype
                self.assertEqual(y, y_promo.to(dtype=dtype))
# 调用函数 instantiate_device_type_tests，用于实例化设备类型的测试，传入 TestTypePromotion 和全局变量字典 globals()
instantiate_device_type_tests(TestTypePromotion, globals())

# 检查当前脚本是否作为主程序执行
if __name__ == '__main__':
    # 如果是主程序，则执行测试函数 run_tests()
    run_tests()
```