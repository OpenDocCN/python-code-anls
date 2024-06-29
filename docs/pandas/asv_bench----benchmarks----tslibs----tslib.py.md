# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\tslib.py`

```
"""
ipython analogue:

tr = TimeIntsToPydatetime()
mi = pd.MultiIndex.from_product(
    tr.params[:-1] + ([str(x) for x in tr.params[-1]],)
)
df = pd.DataFrame(np.nan, index=mi, columns=["mean", "stdev"])
for box in tr.params[0]:
    for size in tr.params[1]:
        for tz in tr.params[2]:
            tr.setup(box, size, tz)
            key = (box, size, str(tz))
            print(key)
            val = %timeit -o tr.time_ints_to_pydatetime(box, size, tz)
            df.loc[key] = (val.average, val.stdev)
"""

# 导入所需的模块和库
from datetime import (
    timedelta,
    timezone,
)
import zoneinfo

from dateutil.tz import (
    gettz,
    tzlocal,
)
import numpy as np

try:
    from pandas._libs.tslibs import ints_to_pydatetime
except ImportError:
    from pandas._libs.tslib import ints_to_pydatetime

# 获取本地时区对象
tzlocal_obj = tzlocal()
# 定义不同的时区和数据大小
_tzs = [
    None,
    timezone.utc,
    timezone(timedelta(minutes=60)),
    zoneinfo.ZoneInfo("US/Pacific"),
    gettz("Asia/Tokyo"),
    tzlocal_obj,
]
_sizes = [0, 1, 100, 10**4, 10**6]

# 定义一个类来处理时间转换
class TimeIntsToPydatetime:
    params = (
        ["time", "date", "datetime", "timestamp"],
        _sizes,
        _tzs,
    )
    param_names = ["box", "size", "tz"]
    # TODO: fold?

    # 设置函数，根据不同的参数设置数据
    def setup(self, box, size, tz):
        if box == "date" and tz is not None:
            # tz is ignored, so avoid running redundant benchmarks
            raise NotImplementedError  # skip benchmark
        if size == 10**6 and tz is _tzs[-1]:
            # This is cumbersomely-slow, so skip to trim runtime
            raise NotImplementedError  # skip benchmark

        # 生成随机整数数组
        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr

    # 计算时间转换的函数
    def time_ints_to_pydatetime(self, box, size, tz):
        ints_to_pydatetime(self.i8data, tz, box=box)
```