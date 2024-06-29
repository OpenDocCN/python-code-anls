# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\tz_convert.py`

```
# 导入日期时间处理所需的库
from datetime import timezone

# 导入科学计算库 numpy
import numpy as np

# 导入 pandas 日期时间本地化相关函数
from pandas._libs.tslibs.tzconversion import tz_localize_to_utc

# 导入自定义的时区库 tslib 中的相关内容
from .tslib import (
    _sizes,
    _tzs,
    tzlocal_obj,
)

# 尝试导入旧版的时区转换函数，如果失败则导入新版的
try:
    old_sig = False
    from pandas._libs.tslibs import tz_convert_from_utc
except ImportError:
    try:
        old_sig = False
        from pandas._libs.tslibs.tzconversion import tz_convert_from_utc
    except ImportError:
        old_sig = True
        from pandas._libs.tslibs.tzconversion import tz_convert as tz_convert_from_utc

# 定义一个类 TimeTZConvert
class TimeTZConvert:
    # 定义类属性 params 和 param_names
    params = [
        _sizes,
        [x for x in _tzs if x is not None],
    ]
    param_names = ["size", "tz"]

    # 设置函数，在每次执行基准测试前调用
    def setup(self, size, tz):
        # 如果 size 为 10^6 并且 tz 是 tzlocal_obj，则抛出 NotImplementedError
        if size == 10**6 and tz is tzlocal_obj:
            raise NotImplementedError
        
        # 创建一个包含随机整数的 numpy 数组，数据类型为 int64
        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr

    # 计算 tz_convert_from_utc 函数的执行时间
    def time_tz_convert_from_utc(self, size, tz):
        # 有效地执行以下操作：
        #  dti = DatetimeIndex(self.i8data, tz=tz)
        #  dti.tz_localize(None)
        if old_sig:
            tz_convert_from_utc(self.i8data, timezone.utc, tz)
        else:
            tz_convert_from_utc(self.i8data, tz)

    # 计算 tz_localize_to_utc 函数的执行时间
    def time_tz_localize_to_utc(self, size, tz):
        # 有效地执行以下操作：
        #  dti = DatetimeIndex(self.i8data)
        #  dti.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        tz_localize_to_utc(self.i8data, tz, ambiguous="NaT", nonexistent="NaT")
```