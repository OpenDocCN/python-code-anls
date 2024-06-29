# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_textpath.py`

```
# 导入 copy 模块，用于对象的复制操作
import copy

# 从 matplotlib.textpath 模块导入 TextPath 类
from matplotlib.textpath import TextPath

# 定义测试函数 test_copy
def test_copy():
    # 创建 TextPath 对象 tp，位置为 (0, 0)，内容为单个点 "."
    tp = TextPath((0, 0), ".")
    
    # 使用 deepcopy 复制 tp 对象，断言新对象的 vertices 属性与原对象的 vertices 属性不同
    assert copy.deepcopy(tp).vertices is not tp.vertices
    
    # 使用 deepcopy 复制 tp 对象，断言新对象的 vertices 属性与原对象的 vertices 属性值完全相同
    assert (copy.deepcopy(tp).vertices == tp.vertices).all()
    
    # 使用 copy 复制 tp 对象，断言新对象的 vertices 属性与原对象的 vertices 属性是同一个对象
    assert copy.copy(tp).vertices is tp.vertices
```