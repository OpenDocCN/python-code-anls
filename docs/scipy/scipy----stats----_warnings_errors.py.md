# `D:\src\scipysrc\scipy\scipy\stats\_warnings_errors.py`

```
# 定义一个自定义警告类，用于指示数据退化且结果可能不可靠的情况
class DegenerateDataWarning(RuntimeWarning):
    """Warns when data is degenerate and results may not be reliable."""
    def __init__(self, msg=None):
        # 如果未提供警告信息，则使用默认消息
        if msg is None:
            msg = ("Degenerate data encountered; results may not be reliable.")
        # 将消息存储在 self.args 中，以便后续使用
        self.args = (msg,)


# 继承自 DegenerateDataWarning 的警告类，用于指示数据中所有值完全相等的情况
class ConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are exactly equal."""
    def __init__(self, msg=None):
        # 如果未提供警告信息，则使用默认消息
        if msg is None:
            msg = ("All values in data are exactly equal; "
                   "results may not be reliable.")
        # 将消息存储在 self.args 中，以便后续使用
        self.args = (msg,)


# 继承自 DegenerateDataWarning 的警告类，用于指示数据中所有值几乎相等的情况
class NearConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are nearly equal."""
    def __init__(self, msg=None):
        # 如果未提供警告信息，则使用默认消息
        if msg is None:
            msg = ("All values in data are nearly equal; "
                   "results may not be reliable.")
        # 将消息存储在 self.args 中，以便后续使用
        self.args = (msg,)


# 定义一个表示在拟合数据分布时出现错误的异常类
class FitError(RuntimeError):
    """Represents an error condition when fitting a distribution to data."""
    def __init__(self, msg=None):
        # 如果未提供错误信息，则使用默认消息
        if msg is None:
            msg = ("An error occurred when fitting a distribution to data.")
        # 将消息存储在 self.args 中，以便后续使用
        self.args = (msg,)
```