# `D:\src\scipysrc\scipy\scipy\io\arff\__init__.py`

```
"""
Module to read ARFF files
=========================
ARFF is the standard data format for WEKA.
It is a text file format which support numerical, string and data values.
The format can also represent missing data and sparse data.

Notes
-----
The ARFF support in ``scipy.io`` provides file reading functionality only.
For more extensive ARFF functionality, see `liac-arff
<https://github.com/renatopp/liac-arff>`_.

See the `WEKA website <http://weka.wikispaces.com/ARFF>`_
for more details about the ARFF format and available datasets.

"""
# 从 _arffread 模块中导入所有内容
from ._arffread import *

# 从当前包导入 _arffread 模块
from . import _arffread

# 弃用的命名空间，在 v2.0.0 版本中将被移除
# 从 arffread 中导入所有内容
from .import arffread

# 将 _arffread 模块的所有公共符号添加到当前模块的公共符号中
__all__ = _arffread.__all__ + ['arffread']

# 从 scipy._lib._testutils 模块中导入 PytestTester 类
from scipy._lib._testutils import PytestTester

# 创建一个 PytestTester 对象，名称为当前模块的名称
test = PytestTester(__name__)

# 删除 PytestTester 对象，清理命名空间
del PytestTester
```