# `.\numpy\numpy\typing\tests\data\pass\warnings_and_errors.py`

```
import numpy.exceptions as ex
创建了一个别名 ex，用于引用 numpy.exceptions 模块

ex.AxisError("test")
创建一个 AxisError 的异常对象，参数为字符串 "test"，用于标识异常

ex.AxisError(1, ndim=2)
创建一个 AxisError 的异常对象，其中第一个参数为整数 1，ndim 参数为 2，指定异常的维度信息

ex.AxisError(1, ndim=2, msg_prefix="error")
创建一个 AxisError 的异常对象，除了指定异常的维度信息外，还传入了 msg_prefix 参数，用于定制异常消息前缀

ex.AxisError(1, ndim=2, msg_prefix=None)
创建一个 AxisError 的异常对象，参数与上例相同，但是 msg_prefix 参数被设置为 None，可能会影响异常消息的生成
```