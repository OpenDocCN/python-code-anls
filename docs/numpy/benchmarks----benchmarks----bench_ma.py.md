# `.\numpy\benchmarks\benchmarks\bench_ma.py`

```py
# 导入Benchmark类从common模块中
from .common import Benchmark

# 导入NumPy库并用np作为别名
import numpy as np

# 定义MA类，继承Benchmark类
class MA(Benchmark):
    
    # 设置方法，初始化self.l100为0到99的整数范围
    def setup(self):
        self.l100 = range(100)
        # 初始化self.t100为包含100个True的列表
        self.t100 = ([True] * 100)

    # 定义time_masked_array方法
    def time_masked_array(self):
        # 调用NumPy的masked_array函数，未指定参数

    # 定义time_masked_array_l100方法
    def time_masked_array_l100(self):
        # 调用NumPy的masked_array函数，使用self.l100作为数据参数

    # 定义time_masked_array_l100_t100方法
    def time_masked_array_l100_t100(self):
        # 调用NumPy的masked_array函数，使用self.l100作为数据参数，self.t100作为掩码参数

# 定义MACreation类，继承Benchmark类
class MACreation(Benchmark):
    
    # 参数名列表为'data'和'mask'
    param_names = ['data', 'mask']
    # 参数为[[10, 100, 1000], [True, False, None]]
    params = [[10, 100, 1000],
              [True, False, None]]

    # 定义time_ma_creations方法，接受'data'和'mask'作为参数
    def time_ma_creations(self, data, mask):
        # 调用NumPy的ma.array函数，data参数为由0填充的整数数组，mask参数为传入的掩码值

# 定义Indexing类，继承Benchmark类
class Indexing(Benchmark):
    
    # 参数名列表为'masked'、'ndim'、'size'
    param_names = ['masked', 'ndim', 'size']
    # 参数为[[True, False], [1, 2], [10, 100, 1000]]
    params = [[True, False],
              [1, 2],
              [10, 100, 1000]]
    
    # 设置方法，根据参数设置数据self.m和索引self.idx_scalar、self.idx_0d、self.idx_1d
    def setup(self, masked, ndim, size):
        # 创建ndim维大小为size的数组x，根据masked标志创建掩码数组或者普通数组
        x = np.arange(size**ndim).reshape(ndim * (size,))

        # 根据masked标志，创建掩码或者非掩码的ma.array对象self.m
        if masked:
            self.m = np.ma.array(x, mask=x % 2 == 0)
        else:
            self.m = np.ma.array(x)

        # 根据ndim和size设置不同的索引方式
        self.idx_scalar = (size // 2,) * ndim
        self.idx_0d = (size // 2,) * ndim + (Ellipsis,)
        self.idx_1d = (size // 2,) * (ndim - 1)

    # 定义time_scalar方法，接受'masked'、'ndim'、'size'作为参数
    def time_scalar(self, masked, ndim, size):
        # 使用self.idx_scalar索引self.m

    # 定义time_0d方法，接受'masked'、'ndim'、'size'作为参数
    def time_0d(self, masked, ndim, size):
        # 使用self.idx_0d索引self.m

    # 定义time_1d方法，接受'masked'、'ndim'、'size'作为参数
    def time_1d(self, masked, ndim, size):
        # 使用self.idx_1d索引self.m

# 定义UFunc类，继承Benchmark类
class UFunc(Benchmark):
    
    # 参数名列表为'a_masked'、'b_masked'、'size'
    param_names = ['a_masked', 'b_masked', 'size']
    # 参数为[[True, False], [True, False], [10, 100, 1000]]
    params = [[True, False],
              [True, False],
              [10, 100, 1000]]

    # 设置方法，根据参数设置数据self.a_scalar、self.b_scalar、self.a_1d、self.b_1d、self.a_2d、self.b_2d
    def setup(self, a_masked, b_masked, size):
        # 创建大小为size的无符号整数数组x
        x = np.arange(size).astype(np.uint8)

        # 根据a_masked和b_masked创建对应的标量掩码或者数值
        self.a_scalar = np.ma.masked if a_masked else 5
        self.b_scalar = np.ma.masked if b_masked else 3

        # 根据a_masked和b_masked创建1维掩码或者非掩码数组self.a_1d、self.b_1d
        self.a_1d = np.ma.array(x, mask=x % 2 == 0 if a_masked else np.ma.nomask)
        self.b_1d = np.ma.array(x, mask=x % 3 == 0 if b_masked else np.ma.nomask)

        # 根据a_1d和b_1d创建2维数组self.a_2d、self.b_2d
        self.a_2d = self.a_1d.reshape(1, -1)
        self.b_2d = self.a_1d.reshape(-1, 1)

    # 定义time_scalar方法，接受'a_masked'、'b_masked'、'size'作为参数
    def time_scalar(self, a_masked, b_masked, size):
        # 调用NumPy的ma.add函数，对self.a_scalar和self.b_scalar进行运算

    # 定义time_scalar_1d方法，接受'a_masked'、'b_masked'、'size'作为参数
    def time_scalar_1d(self, a_masked, b_masked, size):
        # 调用NumPy的ma.add函数，对self.a_scalar和self.b_1d进行运算

    # 定义time_1d方法，接受'a_masked'、'b_masked'、'size'作为参数
    def time_1d(self, a_masked, b_masked, size):
        # 调用NumPy的ma.add函数，对self.a_1d和self.b_1d进行运算

    # 定义time_2d方法，接受'a_masked'、'b_masked'、'size'作为参数
    def time_2d(self, a_masked, b_masked, size):
        # 调用NumPy的ma.add函数，对self.a_2d和self.b_2d进行运算

# 定义Concatenate类，继承Benchmark类
class Concatenate(Benchmark):
    
    # 参数名列表为'mode'、'n'
    param_names = ['mode', 'n']
    # 参数为多个元组，包括多种模式和不同的n值
    params = [
        ['ndarray', 'unmasked',
         'ndarray+masked', 'unmasked+masked',
         'masked'],
        [2, 100, 2000]
    ]
    # 定义设置方法，初始化测试模式和尺寸
    def setup(self, mode, n):
        # 避免 np.zeros 的延迟分配，这可能在基准测试期间导致页面错误。
        # np.full 会导致设置过程中的页面错误发生。
        # 创建一个 n x n 大小的全零数组，数据类型为整数
        normal = np.full((n, n), 0, int)
        # 创建一个 n x n 大小的未掩码的 Masked Array，数据类型为整数
        unmasked = np.ma.zeros((n, n), int)
        # 使用 normal 数组创建一个掩码为 True 的 Masked Array
        masked = np.ma.array(normal, mask=True)

        # 拆分模式字符串成多个部分
        mode_parts = mode.split('+')
        # 获取基础模式
        base = mode_parts[0]
        # 确定是否提升到 masked 数组
        promote = 'masked' in mode_parts[1:]

        # 根据基础模式选择相应的参数数组
        if base == 'ndarray':
            args = 10 * (normal,)
        elif base == 'unmasked':
            args = 10 * (unmasked,)
        else:
            args = 10 * (masked,)

        # 如果需要提升，则用 masked 替换最后一个参数数组
        if promote:
            args = args[:-1] + (masked,)

        # 将参数数组赋给实例变量 self.args
        self.args = args

    # 定义计时方法，执行 np.ma.concatenate 操作
    def time_it(self, mode, n):
        # 连接 self.args 中的所有 Masked Array
        np.ma.concatenate(self.args)
class MAFunctions1v(Benchmark):
    # 定义一个继承自Benchmark类的MAFunctions1v类，用于性能基准测试
    param_names = ['mtype', 'func', 'msize']
    # 参数名称列表，包括mtype（类型）、func（函数）、msize（大小）
    params = [['np', 'np.ma'],
              ['sin', 'log', 'sqrt'],
              ['small', 'big']]
    # 参数值列表，包括不同的类型、函数和大小的组合

    def setup(self, mtype, func, msize):
        # 初始化设置方法，接受mtype（类型）、func（函数）、msize（大小）作为参数
        xs = 2.0 + np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 创建一个2x3的数组xs，元素为2.0加上从-1到1均匀分布的随机数
        m1 = [[True, False, False], [False, False, True]]
        # 创建一个掩码数组m1，用于遮盖xs的部分数据
        xl = 2.0 + np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建一个100x100的数组xl，元素为2.0加上从-1到1均匀分布的随机数
        maskx = xl > 2.8
        # 创建一个掩码数组maskx，标记xl中大于2.8的元素
        self.nmxs = np.ma.array(xs, mask=m1)
        # 创建一个掩码数组self.nmxs，使用xs和m1初始化，用于处理缺失值数据
        self.nmxl = np.ma.array(xl, mask=maskx)
        # 创建一个掩码数组self.nmxl，使用xl和maskx初始化，用于处理缺失值数据

    def time_functions_1v(self, mtype, func, msize):
        # 性能基准测试方法，接受mtype（类型）、func（函数）、msize（大小）作为参数
        fun = eval(f"{mtype}.{func}")
        # 根据mtype和func拼接字符串，并使用eval函数执行，获取对应的函数对象
        if msize == 'small':
            fun(self.nmxs)
            # 如果msize为'small'，则对self.nmxs应用fun函数
        elif msize == 'big':
            fun(self.nmxl)
            # 如果msize为'big'，则对self.nmxl应用fun函数


class MAMethod0v(Benchmark):
    # 定义一个继承自Benchmark类的MAMethod0v类，用于性能基准测试
    param_names = ['method', 'msize']
    # 参数名称列表，包括method（方法）、msize（大小）
    params = [['ravel', 'transpose', 'compressed', 'conjugate'],
              ['small', 'big']]
    # 参数值列表，包括不同的方法和大小的组合

    def setup(self, method, msize):
        # 初始化设置方法，接受method（方法）、msize（大小）作为参数
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 创建一个2x3的数组xs，元素为从-1到1均匀分布的随机数
        m1 = [[True, False, False], [False, False, True]]
        # 创建一个掩码数组m1，用于遮盖xs的部分数据
        xl = np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建一个100x100的数组xl，元素为从-1到1均匀分布的随机数
        maskx = xl > 0.8
        # 创建一个掩码数组maskx，标记xl中大于0.8的元素
        self.nmxs = np.ma.array(xs, mask=m1)
        # 创建一个掩码数组self.nmxs，使用xs和m1初始化，用于处理缺失值数据
        self.nmxl = np.ma.array(xl, mask=maskx)
        # 创建一个掩码数组self.nmxl，使用xl和maskx初始化，用于处理缺失值数据

    def time_methods_0v(self, method, msize):
        # 性能基准测试方法，接受method（方法）、msize（大小）作为参数
        if msize == 'small':
            mdat = self.nmxs
            # 如果msize为'small'，则使用self.nmxs作为测试数据
        elif msize == 'big':
            mdat = self.nmxl
            # 如果msize为'big'，则使用self.nmxl作为测试数据
        getattr(mdat, method)()
        # 使用getattr函数根据method名称调用mdat对象的相应方法


class MAFunctions2v(Benchmark):
    # 定义一个继承自Benchmark类的MAFunctions2v类，用于性能基准测试
    param_names = ['mtype', 'func', 'msize']
    # 参数名称列表，包括mtype（类型）、func（函数）、msize（大小）
    params = [['np', 'np.ma'],
              ['multiply', 'divide', 'power'],
              ['small', 'big']]
    # 参数值列表，包括不同的类型、函数和大小的组合

    def setup(self, mtype, func, msize):
        # 初始化设置方法，接受mtype（类型）、func（函数）、msize（大小）作为参数
        # Small arrays
        xs = 2.0 + np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 创建一个2x3的数组xs，元素为2.0加上从-1到1均匀分布的随机数
        ys = 2.0 + np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 创建一个2x3的数组ys，元素为2.0加上从-1到1均匀分布的随机数
        m1 = [[True, False, False], [False, False, True]]
        # 创建一个掩码数组m1，用于遮盖xs的部分数据
        m2 = [[True, False, True], [False, False, True]]
        # 创建一个掩码数组m2，用于遮盖ys的部分数据
        self.nmxs = np.ma.array(xs, mask=m1)
        # 创建一个掩码数组self.nmxs，使用xs和m1初始化，用于处理缺失值数据
        self.nmys = np.ma.array(ys, mask=m2)
        # 创建一个掩码数组self.nmys，使用ys和m2初始化，用于处理缺失值数据
        # Big arrays
        xl = 2.0 + np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建一个100x100的数组xl，元素为2.0加上从-1到1均匀分布的随机数
        yl = 2.0 + np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建一个100x100的数组yl，元素为2.0加上从-1到1均匀分布的随机数
        maskx = xl > 2.8
        # 创建一个掩码数组maskx，标记xl中大于2.8的元素
        masky = yl < 1.8
        # 创建一个掩码数组masky，标记yl中小于1.8的元素
        self.nmxl = np.ma.array(xl, mask=maskx)
        # 创建一个掩码数组self.nmxl，使用xl和maskx初始化，用于处理缺失值数据
        self.nmyl = np.ma.array(yl, mask=masky)
        # 创建一个掩码数组self.nmyl，使用yl和masky初始化，用于处理缺失值数据

    def time_functions_2v(self, mtype, func, msize):
        # 性能基准测试方法，接受mtype（类型）、func（函数）、msize（大小）作为参数
        fun = eval(f"{mtype}.{func}")
        # 根据mtype和func拼接字符串，并使用eval函数执行，获取对应的函数对象
        if msize == 'small':
            fun(self.nmxs, self.nmys)
    # 定义设置方法，初始化变量xs、m1、xl和maskx，将结果存储在对象的属性中
    def setup(self, margs, msize):
        # 生成一个 2x3 的随机数数组，范围在[-1, 1)之间
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 定义一个2x3的布尔类型数组m1
        m1 = [[True, False, False], [False, False, True]]
        # 生成一个100x100的随机数数组，范围在[-1, 1)之间
        xl = np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建一个布尔掩码数组，标记xl中大于0.8的元素
        maskx = xl > 0.8
        # 使用m1数组创建带掩码的MaskedArray对象，存储在self.nmxs中
        self.nmxs = np.ma.array(xs, mask=m1)
        # 使用maskx数组创建带掩码的MaskedArray对象，存储在self.nmxl中
        self.nmxl = np.ma.array(xl, mask=maskx)

    # 定义测试方法，根据参数msize选择合适的数据集，调用对象的__getitem__方法
    def time_methods_getitem(self, margs, msize):
        # 根据参数msize选择数据集
        if msize == 'small':
            mdat = self.nmxs
        elif msize == 'big':
            mdat = self.nmxl
        # 动态调用对象mdat的__getitem__方法，并传递参数margs
        getattr(mdat, '__getitem__')(margs)
class MAMethodSetItem(Benchmark):
    # 参数名列表，用于记录测试中使用的参数名称
    param_names = ['margs', 'mset', 'msize']
    # 参数字典，每个键值对代表一组参数值
    params = [[0, (0, 0), (-1, 0)],
              [17, np.ma.masked],
              ['small', 'big']]

    def setup(self, margs, mset, msize):
        # 创建一个 2x3 的随机数组 xs，元素值在 [-1, 1) 之间
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 定义一个掩码数组 m1，用于遮蔽特定位置的值
        m1 = [[True, False, False], [False, False, True]]
        # 创建一个 100x100 的随机数组 xl，元素值在 [-1, 1) 之间
        xl = np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建一个根据阈值生成的掩码数组 maskx
        maskx = xl > 0.8
        # 使用 xs 和 xl 创建分别带有掩码的数组 nmxs 和 nmxl
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmxl = np.ma.array(xl, mask=maskx)

    def time_methods_setitem(self, margs, mset, msize):
        # 根据 msize 的值选择要操作的数组 mdat
        if msize == 'small':
            mdat = self.nmxs
        elif msize == 'big':
            mdat = self.nmxl
        # 动态调用 mdat 对象的 '__setitem__' 方法，传入 margs 和 mset 作为参数
        getattr(mdat, '__setitem__')(margs, mset)


class Where(Benchmark):
    # 参数名列表，用于记录测试中使用的参数名称
    param_names = ['mtype', 'msize']
    # 参数字典，每个键值对代表一组参数值
    params = [['np', 'np.ma'],
              ['small', 'big']]

    def setup(self, mtype, msize):
        # 创建两个 2x3 的随机数组 xs 和 ys，元素值在 [-1, 1) 之间
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        ys = np.random.uniform(-1, 1, 6).reshape(2, 3)
        # 定义两个掩码数组 m1 和 m2
        m1 = [[True, False, False], [False, False, True]]
        m2 = [[True, False, True], [False, False, True]]
        # 创建两个带有掩码的数组 nmxs 和 nmys
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmys = np.ma.array(ys, mask=m2)
        # 创建一个 100x100 的随机数组 xl 和 yl，元素值在 [-1, 1) 之间
        xl = np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        yl = np.random.uniform(-1, 1, 100*100).reshape(100, 100)
        # 创建两个根据阈值生成的掩码数组 maskx 和 masky
        maskx = xl > 0.8
        masky = yl < -0.8
        # 使用 xl、yl 和对应的掩码数组创建带有掩码的数组 nmxl 和 nmyl
        self.nmxl = np.ma.array(xl, mask=maskx)
        self.nmyl = np.ma.array(yl, mask=masky)

    def time_where(self, mtype, msize):
        # 根据 mtype 动态获取函数对象 fun
        fun = eval(f"{mtype}.where")
        # 根据 msize 的值选择要操作的数组，并调用 fun 函数
        if msize == 'small':
            fun(self.nmxs > 2, self.nmxs, self.nmys)
        elif msize == 'big':
            fun(self.nmxl > 2, self.nmxl, self.nmyl)


class Cov(Benchmark):
    # 参数名列表，用于记录测试中使用的参数名称
    param_names = ["size"]
    # 参数字典，每个键值对代表一组参数值
    params = [["small", "large"]]

    def setup(self, size):
        # 设置掩码值的比例
        prop_mask = 0.2
        # 创建一个大小为 (10, 10) 的随机浮点数数组 data
        rng = np.random.default_rng()
        data = rng.random((10, 10), dtype=np.float32)
        # 使用 data 和阈值 prop_mask 创建带有掩码的数组 small
        self.small = np.ma.array(data, mask=(data <= prop_mask))
        # 创建一个大小为 (100, 100) 的随机浮点数数组 data
        data = rng.random((100, 100), dtype=np.float32)
        # 使用 data 和阈值 prop_mask 创建带有掩码的数组 large
        self.large = np.ma.array(data, mask=(data <= prop_mask))

    def time_cov(self, size):
        # 根据 size 的值选择要操作的数组，并调用 np.ma.cov 函数
        if size == "small":
            np.ma.cov(self.small)
        if size == "large":
            np.ma.cov(self.large)


class Corrcoef(Benchmark):
    # 参数名列表，用于记录测试中使用的参数名称
    param_names = ["size"]
    # 参数字典，每个键值对代表一组参数值
    params = [["small", "large"]]
    # 设置函数 `setup`，用于初始化数据
    def setup(self, size):
        # 设置被遮蔽数值的比例为 0.2
        prop_mask = 0.2
        # 使用 NumPy 提供的随机数生成器创建实例
        rng = np.random.default_rng()
        # 生成一个大小为 (10, 10) 的随机浮点数数组作为数据
        data = rng.random((10, 10), dtype=np.float32)
        # 创建一个带有遮蔽的小数组，根据设定的比例遮蔽部分数据
        self.small = np.ma.array(data, mask=(data <= prop_mask))
        # 再次生成一个大小为 (100, 100) 的随机浮点数数组作为数据
        data = rng.random((100, 100), dtype=np.float32)
        # 创建一个带有遮蔽的大数组，根据设定的比例遮蔽部分数据
        self.large = np.ma.array(data, mask=(data <= prop_mask))

    # 设置函数 `time_corrcoef`，用于计算相关系数
    def time_corrcoef(self, size):
        # 如果参数 `size` 为 "small"，则计算 `self.small` 的相关系数
        if size == "small":
            np.ma.corrcoef(self.small)
        # 如果参数 `size` 为 "large"，则计算 `self.large` 的相关系数
        if size == "large":
            np.ma.corrcoef(self.large)
```