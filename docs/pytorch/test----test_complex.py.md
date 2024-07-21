# `.\pytorch\test\test_complex.py`

```py
# mypy: allow-untyped-defs
# Owner(s): ["module: complex"]

# 导入PyTorch库
import torch
# 导入测试框架相关的设备类型、测试实例化函数、CPU限定装饰器
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
)
# 导入复杂数据类型
from torch.testing._internal.common_dtype import complex_types
# 导入运行测试的工具函数、设置默认数据类型的函数、测试案例类
from torch.testing._internal.common_utils import run_tests, set_default_dtype, TestCase

# 定义设备列表为CPU和第一个CUDA设备
devices = (torch.device("cpu"), torch.device("cuda:0"))


# 定义测试类TestComplexTensor，继承于TestCase类
class TestComplexTensor(TestCase):
    # 测试装饰器，测试复杂数据类型的所有操作
    @dtypes(*complex_types())
    def test_to_list(self, device, dtype):
        # 测试复数浮点张量的tolist()方法
        # 断言复数浮点张量的tolist()方法结果符合预期，且没有垃圾值
        self.assertEqual(
            torch.zeros((2, 2), device=device, dtype=dtype).tolist(),
            [[0j, 0j], [0j, 0j]],
        )

    # 测试装饰器，测试不同浮点数据类型的dtype推断
    @dtypes(torch.float32, torch.float64, torch.float16)
    def test_dtype_inference(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/36834
        # 使用设置默认数据类型的函数设置当前数据类型
        with set_default_dtype(dtype):
            # 创建张量x，包含复数和实数组成
            x = torch.tensor([3.0, 3.0 + 5.0j], device=device)
        # 根据数据类型判断断言张量x的数据类型是否符合预期
        if dtype == torch.float16:
            self.assertEqual(x.dtype, torch.chalf)
        elif dtype == torch.float32:
            self.assertEqual(x.dtype, torch.cfloat)
        else:
            self.assertEqual(x.dtype, torch.cdouble)

    # 测试装饰器，测试共轭和复制操作
    @dtypes(*complex_types())
    def test_conj_copy(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/106051
        # 创建包含复数的张量x1
        x1 = torch.tensor([5 + 1j, 2 + 2j], device=device, dtype=dtype)
        # 计算x1的共轭并赋值给xc1
        xc1 = torch.conj(x1)
        # 复制xc1的值到x1
        x1.copy_(xc1)
        # 断言x1的值是否符合预期
        self.assertEqual(x1, torch.tensor([5 - 1j, 2 - 2j], device=device, dtype=dtype))

    # 测试装饰器，测试所有操作
    @dtypes(*complex_types())
    def test_all(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/120875
        # 创建包含复数的张量x
        x = torch.tensor([1 + 2j, 3 - 4j, 5j, 6], device=device, dtype=dtype)
        # 断言张量x是否所有元素都为真
        self.assertTrue(torch.all(x))

    # 测试装饰器，测试任意操作
    @dtypes(*complex_types())
    def test_any(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/120875
        # 创建包含复数和零值的张量x
        x = torch.tensor(
            [0, 0j, -0 + 0j, -0 - 0j, 0 + 0j, 0 - 0j], device=device, dtype=dtype
        )
        # 断言张量x是否有任意一个元素为真
        self.assertFalse(torch.any(x))

    # CPU限定装饰器，测试所有复杂数据类型
    @onlyCPU
    @dtypes(*complex_types())
    # CPU限定装饰器，测试所有复杂数据类型
    @onlyCPU
    @dtypes(*complex_types())
# 使用测试类实例化设备类型测试
instantiate_device_type_tests(TestComplexTensor, globals())

# 如果当前文件作为主程序执行
if __name__ == "__main__":
    # 启用默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行所有测试
    run_tests()
```