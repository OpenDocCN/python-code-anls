# `.\numpy\benchmarks\benchmarks\bench_clip.py`

```py
# 从.common模块中导入Benchmark类
from .common import Benchmark
# 导入NumPy库并使用别名np
import numpy as np

# 定义ClipFloat类，继承Benchmark类
class ClipFloat(Benchmark):
    # 参数名列表，包括dtype和size
    param_names = ["dtype", "size"]
    # 参数的可能取值，dtype包括np.float32, np.float64, np.longdouble；size包括100和100000
    params = [
        [np.float32, np.float64, np.longdouble],
        [100, 100_000]
    ]

    # 设置方法，在每次运行之前调用，初始化数据
    def setup(self, dtype, size):
        # 使用随机种子创建随机状态对象rnd
        rnd = np.random.RandomState(994584855)
        # 创建指定dtype和size的随机数组，转换为指定dtype
        self.array = rnd.random(size=size).astype(dtype)
        # 创建与self.array相同形状和dtype的全0.5数组，并赋值给self.dataout
        self.dataout = np.full_like(self.array, 0.5)

    # 时间测量方法，用于测量np.clip方法的运行时间
    def time_clip(self, dtype, size):
        # 对self.array数组中的值进行裁剪，裁剪范围是0.125到0.875，并将结果存入self.dataout中
        np.clip(self.array, 0.125, 0.875, self.dataout)


# 定义ClipInteger类，同样继承自Benchmark类
class ClipInteger(Benchmark):
    # 参数名列表，同样包括dtype和size
    param_names = ["dtype", "size"]
    # 参数的可能取值，dtype包括np.int32和np.int64；size包括100和100000
    params = [
        [np.int32, np.int64],
        [100, 100_000]
    ]

    # 设置方法，初始化数据
    def setup(self, dtype, size):
        # 使用随机种子创建随机状态对象rnd
        rnd = np.random.RandomState(1301109903)
        # 创建指定dtype和size的随机整数数组，数值范围在0到255之间
        self.array = rnd.randint(256, size=size, dtype=dtype)
        # 创建与self.array相同形状和dtype的全128数组，并赋值给self.dataout
        self.dataout = np.full_like(self.array, 128)

    # 时间测量方法，用于测量np.clip方法的运行时间
    def time_clip(self, dtype, size):
        # 对self.array数组中的值进行裁剪，裁剪范围是32到224，并将结果存入self.dataout中
        np.clip(self.array, 32, 224, self.dataout)
```