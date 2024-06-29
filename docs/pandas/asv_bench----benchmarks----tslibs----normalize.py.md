# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\normalize.py`

```
# 尝试导入优化后的日期时间库函数
try:
    from pandas._libs.tslibs import (
        is_date_array_normalized,
        normalize_i8_timestamps,
    )
except ImportError:
    # 如果导入失败，回退到普通的日期时间库函数
    from pandas._libs.tslibs.conversion import (
        normalize_i8_timestamps,
        is_date_array_normalized,
    )

# 导入 pandas 库并重命名为 pd
import pandas as pd

# 从当前包中导入特定模块
from .tslib import (
    _sizes,
    _tzs,
    tzlocal_obj,
)

# 定义一个 Normalize 类
class Normalize:
    # 定义类级别的参数 params 和 param_names
    params = [
        _sizes,
        _tzs,
    ]
    param_names = ["size", "tz"]

    # 初始化设置方法
    def setup(self, size, tz):
        # 创建一个日期时间索引对象 dti，确保 is_date_array_normalized 返回 True
        dti = pd.date_range("2016-01-01", periods=10, tz=tz).repeat(size // 10)
        # 将日期时间索引转换为整数表示形式并保存在实例变量 self.i8data 中
        self.i8data = dti.asi8

        # 如果 size 为 10^6 并且 tz 为 tzlocal_obj，抛出 NotImplementedError
        if size == 10**6 and tz is tzlocal_obj:
            raise NotImplementedError

    # 时间化函数 normalize_i8_timestamps 方法
    def time_normalize_i8_timestamps(self, size, tz):
        # 使用 normalize_i8_timestamps 函数将 self.i8data 数组归一化
        # 10 代表 NPY_FR_ns 时间单位
        normalize_i8_timestamps(self.i8data, tz, 10)

    # 时间化函数 is_date_array_normalized 方法
    def time_is_date_array_normalized(self, size, tz):
        # TODO: 处理不同级别的 short-circuiting 情况
        # 使用 is_date_array_normalized 函数检查 self.i8data 数组是否归一化
        # 10 代表 NPY_FR_ns 时间单位
        is_date_array_normalized(self.i8data, tz, 10)
```