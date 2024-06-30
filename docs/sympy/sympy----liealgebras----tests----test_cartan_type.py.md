# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_cartan_type.py`

```
# 导入需要的模块中的类和函数
from sympy.liealgebras.cartan_type import CartanType, Standard_Cartan

# 定义测试函数，用于测试 Standard_Cartan 类的功能
def test_Standard_Cartan():
    # 创建一个 CartanType 对象，代表 A4 型李代数类型
    c = CartanType("A4")
    # 断言其秩为 4
    assert c.rank() == 4
    # 断言其系列为 "A"
    assert c.series == "A"
    
    # 创建一个 Standard_Cartan 对象，代表 A 型李代数的标准 Cartan 子代数，秩为 2
    m = Standard_Cartan("A", 2)
    # 断言其秩为 2
    assert m.rank() == 2
    # 断言其系列为 "A"
    assert m.series == "A"
    
    # 创建一个 CartanType 对象，代表 B12 型李代数类型
    b = CartanType("B12")
    # 断言其秩为 12
    assert b.rank() == 12
    # 断言其系列为 "B"
    assert b.series == "B"
```