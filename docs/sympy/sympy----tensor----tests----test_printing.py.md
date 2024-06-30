# `D:\src\scipysrc\sympy\sympy\tensor\tests\test_printing.py`

```
# 导入必要的类和函数来定义张量相关的对象
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
# 导入复数单位 I
from sympy import I

# 定义一个测试函数，用于测试张量乘积的打印输出
def test_printing_TensMul():
    # 创建一个张量索引类型 'R3'，维度为 3
    R3 = TensorIndexType('R3', dim=3)
    # 创建两个张量索引对象 p 和 q，它们属于索引类型 R3
    p, q = tensor_indices("p q", R3)
    # 创建一个张量头部对象 'K'，其索引类型为 R3
    K = TensorHead("K", [R3])

    # 断言表达式的打印输出与预期结果相符
    assert repr(2*K(p)) == "2*K(p)"
    assert repr(-K(p)) == "-K(p)"
    assert repr(-2*K(p)*K(q)) == "-2*K(p)*K(q)"
    assert repr(-I*K(p)) == "-I*K(p)"
    assert repr(I*K(p)) == "I*K(p)"
```