# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\jpl_units\UnitDblFormatter.py`

```
"""UnitDblFormatter module containing class UnitDblFormatter."""

# 导入 matplotlib 的 ticker 模块，用于格式化标签
import matplotlib.ticker as ticker

# 定义本模块中公开的类名
__all__ = ['UnitDblFormatter']

# 定义一个新的格式化类，继承自 ScalarFormatter 类
class UnitDblFormatter(ticker.ScalarFormatter):
    """
    The formatter for UnitDbl data types.

    This allows for formatting with the unit string.
    """

    # 覆盖父类的 __call__ 方法，用于格式化数据
    def __call__(self, x, pos=None):
        # 如果没有数据位置信息，则返回空字符串
        if len(self.locs) == 0:
            return ''
        else:
            # 使用格式化字符串将 x 格式化为 12 位精度的字符串
            return f'{x:.12}'

    # 定义一个额外的方法 format_data_short，用于格式化数据为短字符串
    def format_data_short(self, value):
        # 返回 12 位精度的字符串表示
        return f'{value:.12}'

    # 定义另一个方法 format_data，用于格式化数据
    def format_data(self, value):
        # 返回 12 位精度的字符串表示
        return f'{value:.12}'
```