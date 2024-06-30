# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\__init__.py`

```
""" FFT backend using pypocketfft """

# 导入 FFT 相关模块
from .basic import *
from .realtransforms import *
from .helper import *

# 导入用于测试的 PytestTester 类
from scipy._lib._testutils import PytestTester
# 创建一个 PytestTester 对象并命名为 test，用于当前模块的测试
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，使其不再在当前命名空间中可用
del PytestTester
```