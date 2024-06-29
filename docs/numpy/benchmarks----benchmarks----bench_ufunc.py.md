# `.\numpy\benchmarks\benchmarks\bench_ufunc.py`

```
# 导入从 common 模块中的 Benchmark、get_squares_、TYPES1 和 DLPACK_TYPES
# 注意：'.' 表示当前目录

from .common import Benchmark, get_squares_, TYPES1, DLPACK_TYPES

# 导入 numpy 库，并将其命名为 np
import numpy as np

# 导入 itertools 库，用于生成迭代器的函数
import itertools

# 导入 version 模块，用于处理版本号的类和函数
from packaging import version

# 导入 operator 模块，用于函数操作符的函数
import operator

# 定义一个包含数学函数名的列表
ufuncs = ['abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
          'arctan', 'arctan2', 'arctanh', 'bitwise_and', 'bitwise_count', 'bitwise_not',
          'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj', 'conjugate',
          'copysign', 'cos', 'cosh', 'deg2rad', 'degrees', 'divide', 'divmod',
          'equal', 'exp', 'exp2', 'expm1', 'fabs', 'float_power', 'floor',
          'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'gcd', 'greater',
          'greater_equal', 'heaviside', 'hypot', 'invert', 'isfinite',
          'isinf', 'isnan', 'isnat', 'lcm', 'ldexp', 'left_shift', 'less',
          'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp',
          'logaddexp2', 'logical_and', 'logical_not', 'logical_or',
          'logical_xor', 'matmul', 'maximum', 'minimum', 'mod', 'modf',
          'multiply', 'negative', 'nextafter', 'not_equal', 'positive',
          'power', 'rad2deg', 'radians', 'reciprocal', 'remainder',
          'right_shift', 'rint', 'sign', 'signbit', 'sin',
          'sinh', 'spacing', 'sqrt', 'square', 'subtract', 'tan', 'tanh',
          'true_divide', 'trunc']
          
# 定义一个包含数组函数名的列表
arrayfuncdisp = ['real', 'round']

# 遍历 numpy 模块中的所有属性名
for name in dir(np):
    # 检查属性是否是 numpy 的通用函数（ufunc）并且不在 ufuncs 列表中
    if isinstance(getattr(np, name, None), np.ufunc) and name not in ufuncs:
        # 如果找到未包含的通用函数，则打印警告信息
        print("Missing ufunc %r" % (name,))

# 定义 ArrayFunctionDispatcher 类，继承自 Benchmark 类
class ArrayFunctionDispatcher(Benchmark):
    # 参数化设定为 arrayfuncdisp 列表
    params = [arrayfuncdisp]
    # 参数名称为 'func'
    param_names = ['func']
    # 设置超时时间为 10 秒
    timeout = 10
    
    # 初始化方法，设置 ufuncname 属性
    def setup(self, ufuncname):
        # 忽略所有的运行时错误
        np.seterr(all='ignore')
        try:
            # 获取 numpy 模块中的指定名称的函数对象
            self.afdn = getattr(np, ufuncname)
        except AttributeError:
            # 如果未找到指定名称的函数，则抛出未实现错误
            raise NotImplementedError()
        # 初始化 self.args 列表为空
        self.args = []
        # 遍历 get_squares_ 函数返回的字典中的每对键值对
        for _, aarg in get_squares_().items():
            # 构造参数列表，将 aarg 重复一次作为参数
            arg = (aarg,) * 1  # no nin
            try:
                # 调用 self.afdn 函数，并将 arg 作为参数传入
                self.afdn(*arg)
            except TypeError:
                # 如果调用时遇到类型错误，则跳过该参数
                continue
            # 将合法参数 arg 添加到 self.args 列表中
            self.args.append(arg)

    # 定义 time_afdn_types 方法，用于执行 self.afdn 函数的性能测试
    def time_afdn_types(self, ufuncname):
        # 遍历 self.args 列表，对每个参数列表调用 self.afdn 函数
        [self.afdn(*arg) for arg in self.args]

# 定义 Broadcast 类，继承自 Benchmark 类
class Broadcast(Benchmark):
    # 初始化方法，创建两个全为 1 的数组对象
    def setup(self):
        self.d = np.ones((50000, 100), dtype=np.float64)
        self.e = np.ones((100,), dtype=np.float64)

    # 定义 time_broadcast 方法，用于执行数组广播操作的性能测试
    def time_broadcast(self):
        # 执行数组广播操作 self.d - self.e
        self.d - self.e

# 定义 At 类，继承自 Benchmark 类
class At(Benchmark):
    # 初始化方法，使用随机数生成器创建大数组和索引数组
    def setup(self):
        rng = np.random.default_rng(1)
        self.vals = rng.random(10_000_000, dtype=np.float64)
        self.idx = rng.integers(1000, size=10_000_000).astype(np.intp)
        self.res = np.zeros(1000, dtype=self.vals.dtype)

    # 定义 time_sum_at 方法，用于执行 np.add.at 函数的性能测试
    def time_sum_at(self):
        # 对 self.res 执行 np.add.at 操作，将 self.vals 按照 self.idx 进行求和
        np.add.at(self.res, self.idx, self.vals)

    # 定义 time_maximum_at 方法，用于执行 np.maximum.at 函数的性能测试
    def time_maximum_at(self):
        # 对 self.res 执行 np.maximum.at 操作，将 self.vals 按照 self.idx 进行最大值计算
        np.maximum.at(self.res, self.idx, self.vals)

# 定义 UFunc 类，继承自 Benchmark 类
class UFunc(Benchmark):
    # 参数化设定为 ufuncs 列表
    params = [ufuncs]
    # 参数名称为 'ufunc'
    param_names = ['ufunc']
    # 设置超时时间为 10 秒
    timeout = 10
    # 设置函数，用于初始化和配置对象
    def setup(self, ufuncname):
        # 忽略所有的 NumPy 错误
        np.seterr(all='ignore')
        # 尝试从 NumPy 模块中获取指定名称的函数对象
        try:
            self.ufn = getattr(np, ufuncname)
        # 如果指定的函数名称不存在于 NumPy 中，则抛出未实现错误
        except AttributeError:
            raise NotImplementedError()
        # 初始化参数列表
        self.args = []
        # 遍历通过 get_squares_() 函数获取的项目字典中的每一个键值对
        for _, aarg in get_squares_().items():
            # 根据函数的输入参数个数，复制参数，创建参数元组
            arg = (aarg,) * self.ufn.nin
            # 尝试调用函数 self.ufn，如果参数不匹配则捕获 TypeError 异常并继续
            try:
                self.ufn(*arg)
            except TypeError:
                continue
            # 将有效参数添加到参数列表中
            self.args.append(arg)

    # 测试函数执行时间的方法
    def time_ufunc_types(self, ufuncname):
        # 对 self.args 列表中的每个参数元组调用 self.ufn 函数，记录执行时间
        [self.ufn(*arg) for arg in self.args]
class MethodsV0(Benchmark):
    """ Benchmark for the methods which do not take any arguments
    """
    params = [['__abs__', '__neg__', '__pos__'], TYPES1]  # 参数列表，包括方法名称和数据类型
    param_names = ['methods', 'npdtypes']  # 参数名称列表
    timeout = 10  # 设置超时时间为10秒

    def setup(self, methname, npdtypes):
        values = get_squares_()  # 调用获取平方数的函数，返回一个字典
        self.xarg = values.get(npdtypes)[0]  # 获取指定数据类型的平方数列表中的第一个值作为参数

    def time_ndarray_meth(self, methname, npdtypes):
        getattr(operator, methname)(self.xarg)  # 调用 operator 模块的指定方法，作用于 self.xarg


class NDArrayLRShifts(Benchmark):
    """ Benchmark for the shift methods
    """
    params = [['__lshift__', '__rshift__'],  # 参数列表，包括位移方法名称
              ['intp', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']]  # 数据类型列表
    param_names = ['methods', 'npdtypes']  # 参数名称列表
    timeout = 10  # 设置超时时间为10秒

    def setup(self, methname, npdtypes):
        self.vals = np.ones(1000, dtype=getattr(np, npdtypes)) * \  # 创建一个包含1000个元素的数组，数据类型为指定类型
                    np.random.randint(9)  # 数组元素为随机整数乘以1

    def time_ndarray_meth(self, methname, npdtypes):
        getattr(operator, methname)(*[self.vals, 2])  # 调用 operator 模块的指定位移方法，作用于 self.vals 和 2


class Methods0DBoolComplex(Benchmark):
    """Zero dimension array methods
    """
    params = [['__bool__', '__complex__'],  # 参数列表，包括布尔和复数方法名称
              TYPES1]  # 数据类型列表
    param_names = ['methods', 'npdtypes']  # 参数名称列表
    timeout = 10  # 设置超时时间为10秒

    def setup(self, methname, npdtypes):
        self.xarg = np.array(3, dtype=npdtypes)  # 创建一个零维数组，数据类型为指定类型

    def time_ndarray__0d__(self, methname, npdtypes):
        meth = getattr(self.xarg, methname)  # 获取零维数组的指定方法对象
        meth()  # 调用该方法


class Methods0DFloatInt(Benchmark):
    """Zero dimension array methods
    """
    params = [['__int__', '__float__'],  # 参数列表，包括整数和浮点数方法名称
              [dt for dt in TYPES1 if not dt.startswith('complex')]]  # 数据类型列表，排除复数类型
    param_names = ['methods', 'npdtypes']  # 参数名称列表
    timeout = 10  # 设置超时时间为10秒

    def setup(self, methname, npdtypes):
        self.xarg = np.array(3, dtype=npdtypes)  # 创建一个零维数组，数据类型为指定类型

    def time_ndarray__0d__(self, methname, npdtypes):
        meth = getattr(self.xarg, methname)  # 获取零维数组的指定方法对象
        meth()  # 调用该方法


class Methods0DInvert(Benchmark):
    """Zero dimension array methods
    """
    params = ['int16', 'int32', 'int64']  # 数据类型列表
    param_names = ['npdtypes']  # 参数名称列表
    timeout = 10  # 设置超时时间为10秒

    def setup(self, npdtypes):
        self.xarg = np.array(3, dtype=npdtypes)  # 创建一个零维数组，数据类型为指定类型

    def time_ndarray__0d__(self, npdtypes):
        self.xarg.__invert__()  # 调用数组的按位取反方法


class MethodsV1(Benchmark):
    """ Benchmark for the methods which take an argument
    """
    params = [['__add__', '__eq__', '__ge__', '__gt__', '__le__',  # 参数列表，包括需要传入参数的方法名称
               '__lt__', '__matmul__', '__mul__', '__ne__', '__pow__',
               '__sub__', '__truediv__'],
              TYPES1]  # 数据类型列表
    param_names = ['methods', 'npdtypes']  # 参数名称列表
    timeout = 10  # 设置超时时间为10秒

    def setup(self, methname, npdtypes):
        values = get_squares_().get(npdtypes)  # 调用获取平方数的函数，返回一个字典
        self.xargs = [values[0], values[1]]  # 获取平方数列表中的前两个值作为参数
        if np.issubdtype(npdtypes, np.inexact):
            # 对于低精度数据类型，避免在 __pow__/__matmul__ 中溢出
            self.xargs[1] *= 0.01  # 如果数据类型是浮点数，将第二个参数乘以0.01，避免溢出

    def time_ndarray_meth(self, methname, npdtypes):
        getattr(operator, methname)(*self.xargs)  # 调用 operator 模块的指定方法，作用于 self.xargs
class MethodsV1IntOnly(Benchmark):
    """ Benchmark for the methods which take an argument
    """
    # 参数化：方法名和数据类型
    params = [['__and__', '__or__', '__xor__'],
              ['int16', 'int32', 'int64']]
    # 参数名称：方法名和数据类型
    param_names = ['methods', 'npdtypes']
    # 超时时间：10秒
    timeout = 10

    def setup(self, methname, npdtypes):
        # 获取指定数据类型的平方值
        values = get_squares_().get(npdtypes)
        # 设置方法参数列表
        self.xargs = [values[0], values[1]]

    def time_ndarray_meth(self, methname, npdtypes):
        # 调用 operator 模块中的指定方法
        getattr(operator, methname)(*self.xargs)


class MethodsV1NoComplex(Benchmark):
    """ Benchmark for the methods which take an argument
    """
    # 参数化：方法名和不包含复数类型的数据类型
    params = [['__floordiv__', '__mod__'],
              [dt for dt in TYPES1 if not dt.startswith('complex')]]
    # 参数名称：方法名和数据类型
    param_names = ['methods', 'npdtypes']
    # 超时时间：10秒
    timeout = 10

    def setup(self, methname, npdtypes):
        # 获取指定数据类型的平方值
        values = get_squares_().get(npdtypes)
        # 设置方法参数列表
        self.xargs = [values[0], values[1]]

    def time_ndarray_meth(self, methname, npdtypes):
        # 调用 operator 模块中的指定方法
        getattr(operator, methname)(*self.xargs)


class NDArrayGetItem(Benchmark):
    param_names = ['margs', 'msize']
    # 参数化：访问方式和数据大小
    params = [[0, (0, 0), (-1, 0), [0, -1]],
              ['small', 'big']]

    def setup(self, margs, msize):
        # 创建小或大尺寸的随机数组
        self.xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        self.xl = np.random.uniform(-1, 1, 50*50).reshape(50, 50)

    def time_methods_getitem(self, margs, msize):
        # 根据数据大小选择数组，并调用其 __getitem__ 方法
        if msize == 'small':
            mdat = self.xs
        elif msize == 'big':
            mdat = self.xl
        getattr(mdat, '__getitem__')(margs)


class NDArraySetItem(Benchmark):
    param_names = ['margs', 'msize']
    # 参数化：修改方式和数据大小
    params = [[0, (0, 0), (-1, 0), [0, -1]],
              ['small', 'big']]

    def setup(self, margs, msize):
        # 创建小或大尺寸的随机数组
        self.xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        self.xl = np.random.uniform(-1, 1, 100*100).reshape(100, 100)

    def time_methods_setitem(self, margs, msize):
        # 根据数据大小选择数组，并修改其指定位置的值
        if msize == 'small':
            mdat = self.xs
        elif msize == 'big':
            mdat = self.xl
            mdat[margs] = 17


class DLPMethods(Benchmark):
    """ Benchmark for DLPACK helpers
    """
    # 参数化：DLPACK 相关方法和数据类型
    params = [['__dlpack__', '__dlpack_device__'], DLPACK_TYPES]
    # 参数名称：方法名和数据类型
    param_names = ['methods', 'npdtypes']
    # 超时时间：10秒
    timeout = 10

    def setup(self, methname, npdtypes):
        # 获取指定数据类型的平方值
        values = get_squares_()
        # 根据数据类型选择相应的值作为参数
        if npdtypes == 'bool':
            if version.parse(np.__version__) > version.parse("1.25"):
                self.xarg = values.get('int16')[0].astype('bool')
            else:
                raise NotImplementedError("Not supported before v1.25")
        else:
            self.xarg = values.get('int16')[0]

    def time_ndarray_dlp(self, methname, npdtypes):
        # 调用指定对象的 DLPACK 相关方法
        meth = getattr(self.xarg, methname)
        meth()


class NDArrayAsType(Benchmark):
    """ Benchmark for type conversion
    """
    # 参数化：数据类型组合
    params = [list(itertools.combinations(TYPES1, 2))]
    # 参数名称：类型转换
    param_names = ['typeconv']
    # 超时时间：10秒
    timeout = 10
    # 设置测试的初始化方法，接受一个类型转换器参数 typeconv
    def setup(self, typeconv):
        # 检查 typeconv 的第一个元素是否等于第二个元素，如果相等抛出未实现错误
        if typeconv[0] == typeconv[1]:
            raise NotImplementedError(
                    "Skipping test for converting to the same dtype")
        # 获取 typeconv[0] 对应的 get_squares_() 函数的结果，并将其赋值给 self.xarg
        self.xarg = get_squares_().get(typeconv[0])

    # 用于测试类型转换的方法，接受一个类型转换器参数 typeconv
    def time_astype(self, typeconv):
        # 调用 self.xarg 的 astype 方法，将其转换为 typeconv 的第二个元素指定的数据类型
        self.xarg.astype(typeconv[1])
# 创建一个继承自Benchmark的类，用于对小数组和标量上的一些ufunc进行基准测试
class UFuncSmall(Benchmark):
    """Benchmark for a selection of ufuncs on a small arrays and scalars
    
    Since the arrays and scalars are small, we are benchmarking the overhead
    of the numpy ufunc functionality
    """
    
    # 参数列表，包含需要测试的ufunc名称
    params = ['abs', 'sqrt', 'cos']
    
    # 参数名称，只有一个参数名为'ufunc'
    param_names = ['ufunc']
    
    # 设置超时时间为10秒
    timeout = 10

    # 初始化函数，在每个测试之前被调用，设置函数对象self.f为对应的numpy ufunc对象
    def setup(self, ufuncname):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        
        # 创建不同类型的测试数组和标量
        self.array_5 = np.array([1., 2., 10., 3., 4.])
        self.array_int_3 = np.array([1, 2, 3])
        self.float64 = np.float64(1.1)
        self.python_float = 1.1

    # 测试用例，测试对小数组应用ufunc的运行时间
    def time_ufunc_small_array(self, ufuncname):
        self.f(self.array_5)

    # 测试用例，测试对小数组应用ufunc并将结果保存到原数组中的运行时间
    def time_ufunc_small_array_inplace(self, ufuncname):
        self.f(self.array_5, out=self.array_5)

    # 测试用例，测试对小整数数组应用ufunc的运行时间
    def time_ufunc_small_int_array(self, ufuncname):
        self.f(self.array_int_3)

    # 测试用例，测试对numpy标量应用ufunc的运行时间
    def time_ufunc_numpy_scalar(self, ufuncname):
        self.f(self.float64)

    # 测试用例，测试对Python标量应用ufunc的运行时间
    def time_ufunc_python_float(self, ufuncname):
        self.f(self.python_float)


# 自定义的基准测试类
class Custom(Benchmark):
    # 初始化函数，在每个测试之前被调用，创建两种不同大小的布尔类型数组
    def setup(self):
        self.b = np.ones(20000, dtype=bool)
        self.b_small = np.ones(3, dtype=bool)

    # 测试用例，测试对大布尔数组应用np.nonzero()的运行时间
    def time_nonzero(self):
        np.nonzero(self.b)

    # 测试用例，测试对大布尔数组应用逻辑非操作的运行时间
    def time_not_bool(self):
        (~self.b)

    # 测试用例，测试对大布尔数组应用逻辑与操作的运行时间
    def time_and_bool(self):
        (self.b & self.b)

    # 测试用例，测试对大布尔数组应用逻辑或操作的运行时间
    def time_or_bool(self):
        (self.b | self.b)

    # 测试用例，测试对小布尔数组应用逻辑与操作的运行时间
    def time_and_bool_small(self):
        (self.b_small & self.b_small)


# 自定义的基准测试类
class CustomInplace(Benchmark):
    # 初始化函数，在每个测试之前被调用，创建不同大小和类型的数组，并对某些数组进行操作
    def setup(self):
        self.c = np.ones(500000, dtype=np.int8)
        self.i = np.ones(150000, dtype=np.int32)
        self.f = np.zeros(150000, dtype=np.float32)
        self.d = np.zeros(75000, dtype=np.float64)
        # 对某些数组进行赋值操作，可能为提高性能
        self.f *= 1.
        self.d *= 1.

    # 测试用例，测试对大int8数组应用按位或操作的运行时间
    def time_char_or(self):
        np.bitwise_or(self.c, 0, out=self.c)
        np.bitwise_or(0, self.c, out=self.c)

    # 测试用例，测试对大int8数组应用按位或操作的运行时间（使用临时对象）
    def time_char_or_temp(self):
        0 | self.c | 0

    # 测试用例，测试对大int32数组应用按位或操作的运行时间
    def time_int_or(self):
        np.bitwise_or(self.i, 0, out=self.i)
        np.bitwise_or(0, self.i, out=self.i)

    # 测试用例，测试对大int32数组应用按位或操作的运行时间（使用临时对象）
    def time_int_or_temp(self):
        0 | self.i | 0

    # 测试用例，测试对大float32数组应用加法操作的运行时间
    def time_float_add(self):
        np.add(self.f, 1., out=self.f)
        np.add(1., self.f, out=self.f)

    # 测试用例，测试对大float32数组应用加法操作的运行时间（使用临时对象）
    def time_float_add_temp(self):
        1. + self.f + 1.

    # 测试用例，测试对大float64数组应用加法操作的运行时间
    def time_double_add(self):
        np.add(self.d, 1., out=self.d)
        np.add(1., self.d, out=self.d)

    # 测试用例，测试对大float64数组应用加法操作的运行时间（使用临时对象）
    def time_double_add_temp(self):
        1. + self.d + 1.


# 自定义的基准测试类
class CustomScalar(Benchmark):
    # 参数列表，包含测试的数据类型
    params = [np.float32, np.float64]
    
    # 参数名称，只有一个参数名为'dtype'
    param_names = ['dtype']

    # 初始化函数，在每个测试之前被调用，创建不同数据类型的数组
    def setup(self, dtype):
        self.d = np.ones(20000, dtype=dtype)

    # 测试用例，测试对数组应用加法操作的运行时间
    def time_add_scalar2(self, dtype):
        np.add(self.d, 1)

    # 测试用例，测试对数组应用除法操作的运行时间
    def time_divide_scalar2(self, dtype):
        np.divide(self.d, 1)

    # 测试用例，测试对数组应用除法操作并将结果保存到原数组中的运行时间
    def time_divide_scalar2_inplace(self, dtype):
        np.divide(self.d, 1, out=self.d)


class CustomComparison(Benchmark):
    # 定义参数列表，包括多种数据类型，如 int8, int16, 等等
    params = (np.int8,  np.int16,  np.int32,  np.int64, np.uint8, np.uint16,
              np.uint32, np.uint64, np.float32, np.float64, np.bool)
    # 参数名称列表，这里只有一个参数 'dtype'
    param_names = ['dtype']

    # 设置函数，初始化三个属性 x, y, s，均为指定数据类型的全1数组
    def setup(self, dtype):
        self.x = np.ones(50000, dtype=dtype)
        self.y = np.ones(50000, dtype=dtype)
        self.s = np.ones(1, dtype=dtype)

    # 测试函数：比较 self.x 和 self.y 数组中元素的大小关系
    def time_less_than_binary(self, dtype):
        (self.x < self.y)

    # 测试函数：比较 self.s（标量）和 self.x 数组中元素的大小关系
    def time_less_than_scalar1(self, dtype):
        (self.s < self.x)

    # 测试函数：比较 self.x 数组中元素和 self.s（标量）的大小关系
    def time_less_than_scalar2(self, dtype):
        (self.x < self.s)
# 定义一个继承自 Benchmark 的自定义类 CustomScalarFloorDivideInt，用于执行整数类型的除法运算基准测试
class CustomScalarFloorDivideInt(Benchmark):
    # 定义参数，包括整数类型的所有种类和除数列表
    params = (np._core.sctypes['int'],
              [8, -8, 43, -43])
    # 参数名称，分别为数据类型和除数
    param_names = ['dtype', 'divisors']

    # 设置函数，在每次测试前准备数据
    def setup(self, dtype, divisor):
        # 获取指定数据类型的信息
        iinfo = np.iinfo(dtype)
        # 生成指定范围内随机整数数组，作为除数
        self.x = np.random.randint(
                    iinfo.min, iinfo.max, size=10000, dtype=dtype)

    # 定义执行整数除法的基准测试函数
    def time_floor_divide_int(self, dtype, divisor):
        # 执行整数除法操作
        self.x // divisor


# 定义一个继承自 Benchmark 的自定义类 CustomScalarFloorDivideUInt，用于执行无符号整数类型的除法运算基准测试
class CustomScalarFloorDivideUInt(Benchmark):
    # 定义参数，包括无符号整数类型的种类和除数列表
    params = (np._core.sctypes['uint'],
              [8, 43])
    # 参数名称，分别为数据类型和除数
    param_names = ['dtype', 'divisors']

    # 设置函数，在每次测试前准备数据
    def setup(self, dtype, divisor):
        # 获取指定数据类型的信息
        iinfo = np.iinfo(dtype)
        # 生成指定范围内随机无符号整数数组，作为除数
        self.x = np.random.randint(
                    iinfo.min, iinfo.max, size=10000, dtype=dtype)

    # 定义执行无符号整数除法的基准测试函数
    def time_floor_divide_uint(self, dtype, divisor):
        # 执行无符号整数除法操作
        self.x // divisor


# 定义一个继承自 Benchmark 的自定义类 CustomArrayFloorDivideInt，用于执行数组类型的整数除法运算基准测试
class CustomArrayFloorDivideInt(Benchmark):
    # 定义参数，包括整数和无符号整数类型的种类以及不同尺寸的数组大小
    params = (np._core.sctypes['int'] + np._core.sctypes['uint'],
              [100, 10000, 1000000])
    # 参数名称，分别为数据类型和数组大小
    param_names = ['dtype', 'size']

    # 设置函数，在每次测试前准备数据
    def setup(self, dtype, size):
        # 获取指定数据类型的信息
        iinfo = np.iinfo(dtype)
        # 生成指定范围内随机整数数组和随机整数数组（作为除数）
        self.x = np.random.randint(
                    iinfo.min, iinfo.max, size=size, dtype=dtype)
        self.y = np.random.randint(2, 32, size=size, dtype=dtype)

    # 定义执行数组整数除法的基准测试函数
    def time_floor_divide_int(self, dtype, size):
        # 执行数组整数除法操作
        self.x // self.y


# 定义一个继承自 Benchmark 的自定义类 Scalar，用于执行标量操作的基准测试
class Scalar(Benchmark):
    # 设置函数，在每次测试前准备数据
    def setup(self):
        # 初始化标量数组和复数
        self.x = np.asarray(1.0)
        self.y = np.asarray((1.0 + 1j))
        self.z = complex(1.0, 1.0)

    # 定义执行标量加法操作的基准测试函数
    def time_add_scalar(self):
        # 执行标量加法操作
        (self.x + self.x)

    # 定义执行标量加法操作（类型转换为标量）的基准测试函数
    def time_add_scalar_conv(self):
        # 执行标量加法操作（类型转换为标量）
        (self.x + 1.0)

    # 定义执行复数与标量加法操作的基准测试函数
    def time_add_scalar_conv_complex(self):
        # 执行复数与标量加法操作
        (self.y + self.z)


# 定义一个类 ArgPack，用于封装函数参数和关键字参数
class ArgPack:
    __slots__ = ['args', 'kwargs']

    # 初始化函数，接收任意数量的参数和关键字参数
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    # 返回对象的字符串表示
    def __repr__(self):
        # 格式化输出参数和关键字参数
        return '({})'.format(', '.join(
            [repr(a) for a in self.args] +
            ['{}={}'.format(k, repr(v)) for k, v in self.kwargs.items()]
        ))


# 定义一个继承自 Benchmark 的自定义类 ArgParsing，用于执行参数解析速度的基准测试
class ArgParsing(Benchmark):
    # 设置测试参数，包括各种参数和关键字参数的组合
    x = np.array(1.)
    y = np.array(2.)
    out = np.array(3.)
    param_names = ['arg_kwarg']
    params = [[
        ArgPack(x, y),
        ArgPack(x, y, out),
        ArgPack(x, y, out=out),
        ArgPack(x, y, out=(out,)),
        ArgPack(x, y, out=out, subok=True, where=True),
        ArgPack(x, y, subok=True),
        ArgPack(x, y, subok=True, where=True),
        ArgPack(x, y, out, subok=True, where=True)
    ]]

    # 定义执行参数解析速度测试的基准测试函数
    def time_add_arg_parsing(self, arg_pack):
        # 执行参数解析速度测试
        np.add(*arg_pack.args, **arg_pack.kwargs)


# 定义一个继承自 Benchmark 的自定义类 ArgParsingReduce，用于执行参数解析速度的基准测试
class ArgParsingReduce(Benchmark):
    # 设置测试参数，包括各种参数和关键字参数的组合
    # 创建包含浮点数 0 和 1 的 NumPy 数组，范围是 [0, 1)
    a = np.arange(2.)
    # 创建一个包含单个浮点数 0 的 NumPy 数组
    out = np.array(0.)
    # 参数名列表，包含一个元素 'arg_kwarg'
    param_names = ['arg_kwarg']
    # 参数列表，包含一个元素，每个元素都是一个列表，内部包含 ArgPack 对象
    params = [[
        # 使用 ArgPack 类创建对象，传入数组 a 作为参数
        ArgPack(a,),
        # 使用 ArgPack 类创建对象，传入数组 a 和整数 0 作为参数
        ArgPack(a, 0),
        # 使用 ArgPack 类创建对象，传入数组 a 和轴数 0 作为参数
        ArgPack(a, axis=0),
        # 使用 ArgPack 类创建对象，传入数组 a、整数 0 和 None 作为参数
        ArgPack(a, 0, None),
        # 使用 ArgPack 类创建对象，传入数组 a、轴数 0 和 None 作为参数
        ArgPack(a, axis=0, dtype=None),
        # 使用 ArgPack 类创建对象，传入数组 a、整数 0、None 和 out 数组作为参数
        ArgPack(a, 0, None, out),
        # 使用 ArgPack 类创建对象，传入数组 a、轴数 0、None、dtype=None 和 out 数组作为参数
        ArgPack(a, axis=0, dtype=None, out=out),
        # 使用 ArgPack 类创建对象，传入数组 a 和 out 数组作为参数
        ArgPack(a, out=out)
    ]]

    # 定义一个方法 time_add_reduce_arg_parsing，接收参数 arg_pack
    def time_add_reduce_arg_parsing(self, arg_pack):
        # 调用 NumPy 的 add.reduce 方法，传入 arg_pack 的 args 和 kwargs 作为参数
        np.add.reduce(*arg_pack.args, **arg_pack.kwargs)
# 定义一个继承自 Benchmark 类的二进制操作性能基准类 BinaryBench
class BinaryBench(Benchmark):
    # 参数列表，包含两种数据类型 np.float32 和 np.float64
    params = [np.float32, np.float64]
    # 参数名称列表，对应参数列表中的数据类型
    param_names = ['dtype']

    # 初始化方法，在每个测试函数执行前调用，根据指定的数据类型生成随机数组
    def setup(self, dtype):
        # 数组长度设定为 1000000
        N = 1000000
        # 生成随机浮点数数组 self.a 和 self.b，并将它们转换为指定的数据类型 dtype
        self.a = np.random.rand(N).astype(dtype)
        self.b = np.random.rand(N).astype(dtype)

    # 计算 np.power(self.a, self.b) 的执行时间的测试函数
    def time_pow(self, dtype):
        np.power(self.a, self.b)

    # 计算 np.power(self.a, 2.0) 的执行时间的测试函数
    def time_pow_2(self, dtype):
        np.power(self.a, 2.0)

    # 计算 np.power(self.a, 0.5) 的执行时间的测试函数
    def time_pow_half(self, dtype):
        np.power(self.a, 0.5)

    # 计算 np.arctan2(self.a, self.b) 的执行时间的测试函数
    def time_atan2(self, dtype):
        np.arctan2(self.a, self.b)

# 定义一个继承自 Benchmark 类的整数操作性能基准类 BinaryBenchInteger
class BinaryBenchInteger(Benchmark):
    # 参数列表，包含两种数据类型 np.int32 和 np.int64
    params = [np.int32, np.int64]
    # 参数名称列表，对应参数列表中的数据类型
    param_names = ['dtype']

    # 初始化方法，在每个测试函数执行前调用，根据指定的数据类型生成随机整数数组
    def setup(self, dtype):
        # 数组长度设定为 1000000
        N = 1000000
        # 生成指定范围内的随机整数数组 self.a 和 self.b，并将它们转换为指定的数据类型 dtype
        self.a = np.random.randint(20, size=N).astype(dtype)
        self.b = np.random.randint(4, size=N).astype(dtype)

    # 计算 np.power(self.a, self.b) 的执行时间的测试函数
    def time_pow(self, dtype):
        np.power(self.a, self.b)

    # 计算 np.power(self.a, 2) 的执行时间的测试函数
    def time_pow_two(self, dtype):
        np.power(self.a, 2)

    # 计算 np.power(self.a, 5) 的执行时间的测试函数
    def time_pow_five(self, dtype):
        np.power(self.a, 5)
```