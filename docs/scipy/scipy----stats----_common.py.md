# `D:\src\scipysrc\scipy\scipy\stats\_common.py`

```
# 导入 namedtuple 类型从 collections 模块
from collections import namedtuple

# 创建一个名为 ConfidenceInterval 的命名元组(namedtuple)，其中包含 low 和 high 两个字段
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])

# 为 ConfidenceInterval 类型添加文档字符串，描述其作用为处理置信区间的类
ConfidenceInterval. __doc__ = "Class for confidence intervals."
```