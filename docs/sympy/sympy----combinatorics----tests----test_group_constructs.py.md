# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_group_constructs.py`

```
# 从 sympy.combinatorics.group_constructs 模块中导入 DirectProduct 类
# 从 sympy.combinatorics.named_groups 模块中导入 CyclicGroup 和 DihedralGroup 类
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.named_groups import CyclicGroup, DihedralGroup


# 定义测试函数 test_direct_product_n
def test_direct_product_n():
    # 创建一个阶数为 4 的循环群 C
    C = CyclicGroup(4)
    # 创建一个阶数为 4 的二面角群 D
    D = DihedralGroup(4)
    # 创建三个循环群 C 的直积，生成群 G
    G = DirectProduct(C, C, C)
    # 断言群 G 的阶数为 64
    assert G.order() == 64
    # 断言群 G 的度为 12
    assert G.degree == 12
    # 断言群 G 的轨道数为 3
    assert len(G.orbits()) == 3
    # 断言群 G 是可交换的（阿贝尔群）
    assert G.is_abelian is True
    # 创建二面角群 D 和循环群 C 的直积，生成群 H
    H = DirectProduct(D, C)
    # 断言群 H 的阶数为 32
    assert H.order() == 32
    # 断言群 H 不是可交换的
    assert H.is_abelian is False
```