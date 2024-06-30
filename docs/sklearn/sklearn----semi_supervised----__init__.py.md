# `D:\src\scipysrc\scikit-learn\sklearn\semi_supervised\__init__.py`

```
"""
半监督学习算法模块。

这些算法利用少量标记数据和大量未标记数据进行分类任务。
"""

# 从 _label_propagation 模块导入 LabelPropagation 和 LabelSpreading 类
from ._label_propagation import LabelPropagation, LabelSpreading
# 从 _self_training 模块导入 SelfTrainingClassifier 类
from ._self_training import SelfTrainingClassifier

# 导出所有公开的类名，方便模块外部使用
__all__ = ["SelfTrainingClassifier", "LabelPropagation", "LabelSpreading"]
```