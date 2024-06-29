# `D:\src\scipysrc\pandas\asv_bench\benchmarks\finalize.py`

```
import pandas as pd  # 导入 pandas 库

class Finalize:  # 定义名为 Finalize 的类
    param_names = ["series", "frame"]  # 类属性：参数名称列表，包括 "series" 和 "frame"
    params = [pd.Series, pd.DataFrame]  # 类属性：参数类型列表，分别是 pd.Series 和 pd.DataFrame

    def setup(self, param):  # 定义实例方法 setup，接受一个参数 param
        N = 1000  # 设定常量 N 为 1000
        obj = param(dtype=float)  # 使用传入的 param 类型创建一个对象 obj，数据类型为 float
        for i in range(N):  # 循环 N 次，从 0 到 999
            obj.attrs[i] = i  # 给对象 obj 添加属性 attrs，属性名为 i，属性值为 i
        self.obj = obj  # 将创建的对象 obj 存储在实例属性 self.obj 中

    def time_finalize_micro(self, param):  # 定义实例方法 time_finalize_micro，接受一个参数 param
        self.obj.__finalize__(self.obj, method="__finalize__")  # 调用对象 obj 的 __finalize__ 方法，传入对象本身和方法参数 "__finalize__"
```