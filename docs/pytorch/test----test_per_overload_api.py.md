# `.\pytorch\test\test_per_overload_api.py`

```
# Owner(s): ["module: unknown"]
import copy  # 导入 copy 模块，用于对象的深拷贝操作

import torch  # 导入 PyTorch 模块
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关的工具函数和类


class TestPerOverloadAPI(TestCase):
    def test_basics_opoverloadpacket(self):
        # 获取 torch.ops.aten.add 操作符作为例子
        add_packet = torch.ops.aten.add

        # 测试类属性
        self.assertEqual(add_packet.__name__, "add")  # 检查操作符名称
        self.assertEqual(str(add_packet), "aten.add")  # 检查操作符的字符串表示形式

        # 调用操作符函数
        self.assertEqual(add_packet(torch.tensor(2), torch.tensor(3)), torch.tensor(5))  # 检查操作符的计算结果

        # 检查正确的模块
        self.assertEqual(add_packet.__module__, add_packet.op.__module__)  # 检查操作符所在的模块

        # 缓存检查
        another_add_packet = torch.ops.aten.add
        self.assertEqual(id(add_packet), id(another_add_packet))  # 检查操作符对象的唯一性

        # 深拷贝应为无操作
        self.assertEqual(id(add_packet), id(copy.deepcopy(add_packet)))  # 检查深拷贝后的对象是否相同

        # 漂亮打印
        self.assertEqual(repr(add_packet), "<OpOverloadPacket(op='aten.add')>")  # 检查操作符的漂亮打印形式

        self.assertRaises(AttributeError, lambda: add_packet.foo)  # 检查操作符不存在属性 'foo'

    def test_basics_opoverload(self):
        add_packet = torch.ops.aten.add
        add_tensoroverload = add_packet.Tensor

        # 测试类属性
        self.assertEqual(str(add_tensoroverload), "aten.add.Tensor")  # 检查重载的类名字符串表示形式
        self.assertEqual(add_tensoroverload.__name__, "add.Tensor")  # 检查重载类的名称
        self.assertEqual(add_tensoroverload.overloadpacket, add_packet)  # 检查重载类关联的操作包

        # 深拷贝应为无操作
        self.assertEqual(id(add_tensoroverload), id(copy.deepcopy(add_tensoroverload)))  # 检查深拷贝后的对象是否相同

        # 缓存检查
        another_add_tensoroverload = torch.ops.aten.add.Tensor
        self.assertEqual(id(add_tensoroverload), id(another_add_tensoroverload))  # 检查重载类对象的唯一性

        # 漂亮打印
        self.assertEqual(
            repr(add_tensoroverload), "<OpOverload(op='aten.add', overload='Tensor')>"
        )  # 检查重载类的漂亮打印形式

        # 调用重载类函数
        self.assertEqual(
            add_tensoroverload(torch.tensor(2), torch.tensor(3)), torch.tensor(5)
        )  # 检查重载类的计算结果

        a = torch.tensor(2)
        b = torch.tensor(0)
        torch.ops.aten.add.out(a, a, out=b)
        self.assertEqual(b, torch.tensor(4))  # 检查使用 out 参数进行的操作

        self.assertRaises(RuntimeError, lambda: add_tensoroverload(a, a, out=b))  # 检查运行时错误的引发情况

    def test_decompose(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        self.assertEqual(
            torch.ops.aten.linear.default.decompose(x, y),
            torch.ops.aten.linear.default(x, y),
        )  # 检查线性操作的默认行为与分解后的结果是否一致


if __name__ == "__main__":
    run_tests()  # 运行所有的测试用例
```