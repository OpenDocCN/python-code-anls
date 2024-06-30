# `D:\src\scipysrc\seaborn\tests\test_objects.py`

```
# 导入 seaborn 库中的对象模块
import seaborn.objects
# 从 seaborn._core.plot 模块中导入 Plot 类
from seaborn._core.plot import Plot
# 从 seaborn._core.moves 模块中导入 Move 类
from seaborn._core.moves import Move
# 从 seaborn._core.scales 模块中导入 Scale 类
from seaborn._core.scales import Scale
# 从 seaborn._marks.base 模块中导入 Mark 类
from seaborn._marks.base import Mark
# 从 seaborn._stats.base 模块中导入 Stat 类
from seaborn._stats.base import Stat

# 定义一个测试函数，用于验证 seaborn.objects 模块中的各个对象是否是 Plot、Mark、Stat、Move、Scale 类的子类
def test_objects_namespace():
    # 遍历 seaborn.objects 模块中的所有属性名
    for name in dir(seaborn.objects):
        # 排除双下划线开头的特殊属性名
        if not name.startswith("__"):
            # 获取属性名对应的对象
            obj = getattr(seaborn.objects, name)
            # 断言该对象是 Plot、Mark、Stat、Move、Scale 类的子类之一
            assert issubclass(obj, (Plot, Mark, Stat, Move, Scale))
```