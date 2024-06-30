# `D:\src\scipysrc\scipy\scipy\io\_harwell_boeing\__init__.py`

```
# 从当前包中导入hb_read和hb_write函数
from .hb import hb_read, hb_write

# 将hb_read和hb_write添加到__all__列表中，表示它们是模块的公开接口
__all__ = ["hb_read", "hb_write"]

# 从scipy._lib._testutils模块中导入PytestTester类
from scipy._lib._testutils import PytestTester

# 创建一个PytestTester对象test，传入当前模块的名称作为参数
test = PytestTester(__name__)

# 删除PytestTester类的引用，这样在后续代码中就无法通过PytestTester访问该类
del PytestTester
```