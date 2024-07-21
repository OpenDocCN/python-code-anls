# `.\pytorch\test\torch_np\numpy_tests\core\test_scalarinherit.py`

```
# Owner(s): ["module: dynamo"]

""" Test printing of scalar types.

"""
# 导入 functools 模块
import functools

# 从 unittest 模块中导入 skipIf 别名为 skipif
from unittest import skipIf as skipif

# 导入 pytest 模块
import pytest

# 导入 torch._numpy 模块，并将其别名为 np
import torch._numpy as np
# 从 torch._numpy.testing 模块导入 assert_ 函数
from torch._numpy.testing import assert_
# 从 torch.testing._internal.common_utils 模块导入 run_tests, TestCase 类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义 skip 函数作为 skipif 的偏函数，总是返回 True
skip = functools.partial(skipif, True)

# 定义类 A，表示一个空的基类
class A:
    pass

# 定义类 B，继承自 A 和 np.float64
class B(A, np.float64):
    pass

# 定义类 C，继承自 B
class C(B):
    pass

# 定义类 D，继承自 C 和 B
class D(C, B):
    pass

# 定义类 B0，继承自 np.float64 和 A
class B0(np.float64, A):
    pass

# 定义类 C0，继承自 B0
class C0(B0):
    pass

# 定义类 HasNew
class HasNew:
    # 实现 __new__ 方法，返回类名 cls，传入参数 args 和关键字参数 kwargs
    def __new__(cls, *args, **kwargs):
        return cls, args, kwargs

# 定义类 B1，继承自 np.float64 和 HasNew
class B1(np.float64, HasNew):
    pass

# 使用 skip 装饰器跳过当前测试类的执行，理由是 "scalar repr: numpy plans to make it more explicit"
@skip(reason="scalar repr: numpy plans to make it more explicit")
# 定义测试类 TestInherit，继承自 TestCase 类
class TestInherit(TestCase):
    # 定义测试方法 test_init
    def test_init(self):
        # 创建 B 类的实例 x，传入参数 1.0
        x = B(1.0)
        # 断言 x 的字符串表示为 "1.0"
        assert_(str(x) == "1.0")
        # 创建 C 类的实例 y，传入参数 2.0
        y = C(2.0)
        # 断言 y 的字符串表示为 "2.0"
        assert_(str(y) == "2.0")
        # 创建 D 类的实例 z，传入参数 3.0
        z = D(3.0)
        # 断言 z 的字符串表示为 "3.0"
        assert_(str(z) == "3.0")

    # 定义测试方法 test_init2
    def test_init2(self):
        # 创建 B0 类的实例 x，传入参数 1.0
        x = B0(1.0)
        # 断言 x 的字符串表示为 "1.0"
        assert_(str(x) == "1.0")
        # 创建 C0 类的实例 y，传入参数 2.0
        y = C0(2.0)
        # 断言 y 的字符串表示为 "2.0"
        assert_(str(y) == "2.0")

    # 定义测试方法 test_gh_15395
    def test_gh_15395(self):
        # 创建 B1 类的实例 x，传入参数 1.0
        x = B1(1.0)
        # 断言 x 的字符串表示为 "1.0"
        assert_(str(x) == "1.0")

        # 使用 pytest 的 raises 断言，期望抛出 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 B1 类的构造函数，传入参数 1.0 和 2.0
            B1(1.0, 2.0)

# 如果当前脚本作为主程序运行，则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```