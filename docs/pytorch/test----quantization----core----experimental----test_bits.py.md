# `.\pytorch\test\quantization\core\experimental\test_bits.py`

```
# Owner(s): ["oncall: quantization"]

# 导入所需的模块和函数
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase, skipIfRocm
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map

import itertools

# 定义一个名为Int16Tensor的新类，继承自torch.Tensor
class Int16Tensor(torch.Tensor):
    # 类的构造函数，接受一个名为elem的参数
    def __new__(cls, elem):
        # 断言elem的数据类型是torch.bits16
        assert elem.dtype == torch.bits16
        # 使用父类的特定方法创建子类实例，设置elem为其数据，elem.requires_grad作为是否需要梯度的标志
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    # 类的初始化函数，调用父类的初始化函数
    def __init__(self, elem):
        super().__init__()

    # 类方法，用于处理torch的分发逻辑
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 定义一个内部函数unwrap，用于处理参数t
        def unwrap(t):
            # 如果t是torch.Tensor类型，则在no_dispatch上下文中，将其视图变换为torch.int16
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.int16)
            return t
        # 对args和kwargs应用unwrap函数
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        # 在no_dispatch上下文中执行func函数，并将结果赋值给out
        with no_dispatch():
            out = func(*args, **kwargs)

        # 定义一个内部函数wrap，用于处理返回值t
        def wrap(t):
            # 如果t是torch.Tensor类型，则在no_dispatch上下文中，将其视图变换为torch.bits16
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.bits16)
            return t
        # 对out应用wrap函数
        out = tree_map(wrap, out)
        return out

    # 方法覆盖，用于处理torch函数的分发逻辑
    # 在这里是简单地调用父类的同名方法
    # 这段代码可能需要删除以便使用被禁用的实现方式，但在Dynamo中，这种情况下测试会失败。
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        # 在no_dispatch上下文中，将self视图变换为torch.int16，并返回格式化的字符串
        with no_dispatch():
            t16 = self.view(torch.int16)
            return f"TensorSubclassDemo{self.view(torch.int16)}"


# 定义一个测试类TestBits，继承自TestCase类
class TestBits(TestCase):
    # 装饰器，用于跳过在ROCm环境下的测试
    @skipIfRocm
    # 测试函数，接受device参数
    def test_types(self, device):
        # 定义bits_types列表，包含不同的torch位数类型
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        # 遍历bits_types列表
        for bits_type in bits_types:
            # 创建一个20x20的全零张量，数据类型为torch.int32，设备为device，然后视图变换为bits_type类型
            _ = torch.zeros(20, dtype=torch.int32, device=device).view(bits_type)
            # 创建一个空的20长度张量，数据类型为bits_type，设备为device
            _ = torch.empty(20, dtype=bits_type, device=device)
            # 创建一个20x20的随机整数张量，数据类型为torch.int8，设备为device，然后视图变换为bits_type类型
            x = torch.randint(100, (20, 20), dtype=torch.int8, device=device).view(bits_type)
            # 对x进行转置并保持连续性，赋值给y
            y = x.t().contiguous()
            # 如果x的元素大小为1，则view_type为torch.int8，否则为torch.int16
            view_type = torch.int8 if x.element_size() == 1 else torch.int16
            # 断言x转置后的视图类型和y的视图类型相等
            self.assertEqual(x.t().view(view_type), y.view(view_type))
            # 对x进行转置并克隆，赋值给y
            y = x.t().clone()
            # 断言x转置后的视图类型和y的视图类型相等
            self.assertEqual(x.t().view(view_type), y.view(view_type))
    # 定义一个测试方法，用于测试torch库中不同位数类型的向量拼接操作
    def test_cat(self, device):
        # 定义各种位数类型的向量数据类型
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        # 遍历每种位数类型
        for bits_type in bits_types:
            # 根据当前位数类型确定视图的数据类型，如果每个元素占用一个字节，则数据类型为torch.int8，否则为torch.int16
            view_type = torch.int8 if bits_type.itemsize == 1 else torch.int16
            # 生成随机整数张量x_int和y_int，形状为(512, 512)，数据类型为view_type，在指定设备上进行计算
            x_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            # 将x_int按照bits_type的类型进行视图变换，得到x
            x = x_int.view(bits_type)
            # 生成随机整数张量y_int，形状为(512, 512)，数据类型为view_type，在指定设备上进行计算
            y_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            # 将y_int按照bits_type的类型进行视图变换，得到y
            y = y_int.view(bits_type)
            # 遍历x_int的维度和转置选项，形成笛卡尔积
            for dim, transpose in itertools.product(range(x_int.ndim), (True, False)):
                # 根据transpose确定y_ref为y_int的转置或原始张量
                y_ref = y_int.t() if transpose else y_int
                # 根据transpose确定y_b为y的转置或原始张量
                y_b = y.t() if transpose else y
                # 使用torch.cat在维度dim上将x_int和y_ref拼接，得到z_ref
                z_ref = torch.cat([x_int, y_ref], dim=dim)
                # 使用torch.cat在维度dim上将x和y_b拼接，得到z
                z = torch.cat([x, y_b], dim=dim)
                # 断言z_ref与z.view(view_type)相等
                self.assertEqual(z_ref, z.view(view_type))

    # 定义一个测试方法，用于测试自定义的Int16Tensor类
    def test_subclass(self):
        # 创建一个全零张量t，形状为(20)，数据类型为torch.int16，然后将其按照torch.bits16的类型进行视图变换，得到t
        t = torch.zeros(20, dtype=torch.int16).view(torch.bits16)
        # 使用Int16Tensor类对t进行封装，得到s
        s = Int16Tensor(t)
        # 对s进行加1减1的操作
        s = s + 1 - 1
        # 断言s与全零张量，形状为(20)，数据类型为torch.bits16，相等
        self.assertTrue(torch.allclose(s, torch.zeros(20, dtype=torch.bits16)))
# 调用函数 instantiate_device_type_tests，并传入 TestBits 类型和全局变量
instantiate_device_type_tests(TestBits, globals())

# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 执行测试函数 run_tests()
    run_tests()
```