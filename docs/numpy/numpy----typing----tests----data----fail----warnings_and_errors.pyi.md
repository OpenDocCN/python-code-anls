# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\warnings_and_errors.pyi`

```
import numpy.exceptions as ex
创建一个别名为 ex 的模块，用于导入 numpy 库中的异常类

ex.AxisError(1.0)  # 创建一个 AxisError 异常的实例，传递参数 1.0 给构造函数

ex.AxisError(1, ndim=2.0)  # 创建一个 AxisError 异常的实例，传递参数 1 给构造函数，并且指定 ndim 参数为 2.0

ex.AxisError(2, msg_prefix=404)  # 创建一个 AxisError 异常的实例，传递参数 2 给构造函数，并且指定 msg_prefix 参数为 404
```