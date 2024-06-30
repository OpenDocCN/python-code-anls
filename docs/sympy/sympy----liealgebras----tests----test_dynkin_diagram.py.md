# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_dynkin_diagram.py`

```
# 从 sympy 库中的 liealgebras 模块导入 DynkinDiagram 类
from sympy.liealgebras.dynkin_diagram import DynkinDiagram

# 定义一个测试函数 test_DynkinDiagram
def test_DynkinDiagram():
    # 创建一个 DynkinDiagram 对象 c，表示 "A3" 的 Dynkin 图
    c = DynkinDiagram("A3")
    # 定义预期的 Dynkin 图 diag，用于断言检查
    diag = "0---0---0\n1   2   3"
    # 断言 c 对象的字符串表示与预期的 diag 相等
    assert c == diag
    
    # 创建一个 DynkinDiagram 对象 ct，表示 "B3" 的 Dynkin 图
    ct = DynkinDiagram(["B", 3])
    # 定义另一个预期的 Dynkin 图 diag2，用于断言检查
    diag2 = "0---0=>=0\n1   2   3"
    # 断言 ct 对象的字符串表示与预期的 diag2 相等
    assert ct == diag2
```