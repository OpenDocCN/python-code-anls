# `.\numpy\benchmarks\benchmarks\bench_strings.py`

```
# 导入Benchmark类，用于性能基准测试
from .common import Benchmark

# 导入numpy库，并将其命名为np，用于数值计算
import numpy as np

# 导入operator模块，用于快速访问比较运算符的函数
import operator

# 定义一个字典_OPERATORS，包含了常见比较运算符和对应的函数映射关系
_OPERATORS = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

# 定义一个继承自Benchmark的类StringComparisons，用于进行字符串比较的性能测试
class StringComparisons(Benchmark):
    # 定义params参数，包含了多个不同的测试参数组合
    params = [
        [100, 10000, (1000, 20)],  # 形状参数
        ['U', 'S'],                # 数据类型参数
        [True, False],             # 连续性参数
        ['==', '!=', '<', '<=', '>', '>=']  # 操作符参数
    ]
    # 定义param_names参数，指定了各个params参数的名称
    param_names = ['shape', 'dtype', 'contig', 'operator']

    # 定义一个int64属性，表示np.int64类型的数据类型
    int64 = np.dtype(np.int64)

    # setup方法用于初始化测试所需的数据和状态
    def setup(self, shape, dtype, contig, operator):
        # 创建一个按照给定形状和数据类型的数组arr
        self.arr = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
        # 创建一个与arr相同的数组arr_identical
        self.arr_identical = self.arr.copy()
        # 创建一个与arr相反顺序的数组arr_different
        self.arr_different = self.arr[::-1].copy()

        # 如果contig为False，对数组进行间隔取值操作
        if not contig:
            self.arr = self.arr[..., ::2]
            self.arr_identical = self.arr_identical[..., ::2]
            self.arr_different = self.arr_different[..., ::2]

        # 根据给定的operator参数，选择对应的比较函数并保存到self.operator属性中
        self.operator = _OPERATORS[operator]

    # time_compare_identical方法用于测试相同数据情况下的比较性能
    def time_compare_identical(self, shape, dtype, contig, operator):
        self.operator(self.arr, self.arr_identical)

    # time_compare_different方法用于测试不同数据情况下的比较性能
    def time_compare_different(self, shape, dtype, contig, operator):
        self.operator(self.arr, self.arr_different)
```