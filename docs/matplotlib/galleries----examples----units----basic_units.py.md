# `D:\src\scipysrc\matplotlib\galleries\examples\units\basic_units.py`

```py
"""
.. _basic_units:

===========
Basic Units
===========

"""

# 导入数学库
import math

# 导入版本解析工具
from packaging.version import parse as parse_version

# 导入 NumPy 库，并用 np 别名表示
import numpy as np

# 导入 matplotlib 的 ticker 和 units 模块
import matplotlib.ticker as ticker
import matplotlib.units as units


# 代理委托类，用于委托方法调用
class ProxyDelegate:
    def __init__(self, fn_name, proxy_type):
        self.proxy_type = proxy_type
        self.fn_name = fn_name

    def __get__(self, obj, objtype=None):
        # 返回代理类型的实例，传入函数名和对象
        return self.proxy_type(self.fn_name, obj)


# 带标记值元类，用于动态生成代理方法
class TaggedValueMeta(type):
    def __init__(self, name, bases, dict):
        # 遍历代理字典，如果未定义相关函数，则动态设置代理方法
        for fn_name in self._proxies:
            if not hasattr(self, fn_name):
                setattr(self, fn_name,
                        ProxyDelegate(fn_name, self._proxies[fn_name]))


# 透传代理类，用于将方法调用透传给目标对象
class PassThroughProxy:
    def __init__(self, fn_name, obj):
        self.fn_name = fn_name
        self.target = obj.proxy_target

    def __call__(self, *args):
        # 获取目标对象的函数并调用，返回结果
        fn = getattr(self.target, self.fn_name)
        ret = fn(*args)
        return ret


# 参数转换代理类，继承自透传代理类，用于参数单位转换
class ConvertArgsProxy(PassThroughProxy):
    def __init__(self, fn_name, obj):
        super().__init__(fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        # 尝试对参数进行单位转换，如果失败则标记为带单位值
        converted_args = []
        for a in args:
            try:
                converted_args.append(a.convert_to(self.unit))
            except AttributeError:
                converted_args.append(TaggedValue(a, self.unit))
        # 获取转换后参数的数值，并调用目标对象的函数返回结果
        converted_args = tuple([c.get_value() for c in converted_args])
        return super().__call__(*converted_args)


# 返回值转换代理类，继承自透传代理类，用于返回值的单位转换
class ConvertReturnProxy(PassThroughProxy):
    def __init__(self, fn_name, obj):
        super().__init__(fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        # 调用父类方法获取结果，如果结果为 NotImplemented 则返回未实现
        ret = super().__call__(*args)
        return (NotImplemented if ret is NotImplemented
                else TaggedValue(ret, self.unit))


# 全部转换代理类，继承自透传代理类，用于参数和返回值的单位转换
class ConvertAllProxy(PassThroughProxy):
    def __init__(self, fn_name, obj):
        super().__init__(fn_name, obj)
        self.unit = obj.unit
    # 定义一个特殊方法 __call__，用于使对象可调用
    def __call__(self, *args):
        # 初始化一个空列表来存储转换后的参数
        converted_args = []
        # 初始化一个包含 self.unit 的列表，self.unit 是对象的单位属性
        arg_units = [self.unit]
        # 遍历传入的参数列表 args
        for a in args:
            # 如果参数 a 拥有 'get_unit' 方法但没有 'convert_to' 方法，
            # 表明此类参数不支持单位转换，返回 NotImplemented
            if hasattr(a, 'get_unit') and not hasattr(a, 'convert_to'):
                return NotImplemented

            # 如果参数 a 拥有 'convert_to' 方法
            if hasattr(a, 'convert_to'):
                try:
                    # 尝试将参数 a 转换到当前对象的单位 self.unit
                    a = a.convert_to(self.unit)
                except Exception:
                    pass
                # 将转换后的参数单位加入 arg_units 列表
                arg_units.append(a.get_unit())
                # 将转换后的参数值加入 converted_args 列表
                converted_args.append(a.get_value())
            else:
                # 如果参数 a 没有 'convert_to' 方法，则直接加入 converted_args 列表
                converted_args.append(a)
                # 如果参数 a 拥有 'get_unit' 方法，则将其单位加入 arg_units 列表
                if hasattr(a, 'get_unit'):
                    arg_units.append(a.get_unit())
                else:
                    # 否则加入 None 到 arg_units 列表
                    arg_units.append(None)
        
        # 转换后的参数列表转换为元组
        converted_args = tuple(converted_args)
        # 调用父类的 __call__ 方法，并传入转换后的参数
        ret = super().__call__(*converted_args)
        # 如果父类返回 NotImplemented，则返回 NotImplemented
        if ret is NotImplemented:
            return NotImplemented
        # 使用 unit_resolver 函数解析返回值的单位，并返回解析后的单位
        ret_unit = unit_resolver(self.fn_name, arg_units)
        # 如果解析后的单位为 NotImplemented，则返回 NotImplemented
        if ret_unit is NotImplemented:
            return NotImplemented
        # 返回一个带有标记单位的 TaggedValue 对象，包括计算结果 ret 和其单位 ret_unit
        return TaggedValue(ret, ret_unit)
class TaggedValue(metaclass=TaggedValueMeta):
    # 类属性，定义用于转换操作的代理方法
    _proxies = {'__add__': ConvertAllProxy,
                '__sub__': ConvertAllProxy,
                '__mul__': ConvertAllProxy,
                '__rmul__': ConvertAllProxy,
                '__cmp__': ConvertAllProxy,
                '__lt__': ConvertAllProxy,
                '__gt__': ConvertAllProxy,
                '__len__': PassThroughProxy}

    def __new__(cls, value, unit):
        # 创建一个新的子类，以处理特定类型的值
        value_class = type(value)
        try:
            subcls = type(f'TaggedValue_of_{value_class.__name__}',
                          (cls, value_class), {})
            return object.__new__(subcls)
        except TypeError:
            return object.__new__(cls)

    def __init__(self, value, unit):
        # 初始化 TaggedValue 实例
        self.value = value  # 存储值
        self.unit = unit    # 存储单位
        self.proxy_target = self.value  # 代理目标设为值本身

    def __copy__(self):
        # 返回 TaggedValue 的浅拷贝
        return TaggedValue(self.value, self.unit)

    def __getattribute__(self, name):
        # 获取属性的特殊方法，支持代理和自定义行为
        if name.startswith('__'):
            return object.__getattribute__(self, name)
        variable = object.__getattribute__(self, 'value')
        if hasattr(variable, name) and name not in self.__class__.__dict__:
            return getattr(variable, name)
        return object.__getattribute__(self, name)

    def __array__(self, dtype=object, copy=False):
        # 转换为 NumPy 数组
        return np.asarray(self.value, dtype)

    def __array_wrap__(self, array, context=None, return_scalar=False):
        # 将 NumPy 数组包装为 TaggedValue 实例
        return TaggedValue(array, self.unit)

    def __repr__(self):
        # 返回对象的详细表示形式
        return f'TaggedValue({self.value!r}, {self.unit!r})'

    def __str__(self):
        # 返回对象的字符串表示形式
        return f"{self.value} in {self.unit}"

    def __len__(self):
        # 返回值的长度
        return len(self.value)

    if parse_version(np.__version__) >= parse_version('1.20'):
        def __getitem__(self, key):
            # 如果 NumPy 版本允许，返回特定项的 TaggedValue
            return TaggedValue(self.value[key], self.unit)

    def __iter__(self):
        # 返回值的迭代器，使用生成器表达式以避免使用 yield 导致的 TypeError
        return (TaggedValue(inner, self.unit) for inner in self.value)

    def get_compressed_copy(self, mask):
        # 返回一个根据掩码压缩的 TaggedValue 实例
        new_value = np.ma.masked_array(self.value, mask=mask).compressed()
        return TaggedValue(new_value, self.unit)

    def convert_to(self, unit):
        # 将值转换为指定单位
        if unit == self.unit or not unit:
            return self
        try:
            new_value = self.unit.convert_value_to(self.value, unit)
        except AttributeError:
            new_value = self
        return TaggedValue(new_value, unit)

    def get_value(self):
        # 获取存储的值
        return self.value

    def get_unit(self):
        # 获取存储的单位
        return self.unit


class BasicUnit:
    def __init__(self, name, fullname=None):
        self.name = name  # 单位名称
        if fullname is None:
            fullname = name
        self.fullname = fullname  # 完整单位名称，默认为名称本身
        self.conversions = dict()  # 存储单位转换方法的字典

    def __repr__(self):
        # 返回对象的详细表示形式
        return f'BasicUnit({self.name})'
    # 返回对象的字符串表示，这里返回对象的fullname属性
    def __str__(self):
        return self.fullname

    # 调用对象时返回一个TaggedValue对象，将value与当前对象(self)关联
    def __call__(self, value):
        return TaggedValue(value, self)

    # 实现乘法运算，根据rhs的类型和属性进行单位转换或乘法操作，并返回TaggedValue对象
    def __mul__(self, rhs):
        # 默认情况下，value等于rhs，unit等于self
        value = rhs
        unit = self
        # 如果rhs具有'get_unit'属性，说明rhs可能是一个对象，获取其值和单位
        if hasattr(rhs, 'get_unit'):
            value = rhs.get_value()  # 获取rhs的值
            unit = rhs.get_unit()    # 获取rhs的单位
            # 使用unit_resolver解析乘法操作中的单位转换，返回新的单位
            unit = unit_resolver('__mul__', (self, unit))
        # 如果单位是NotImplemented，则返回NotImplemented
        if unit is NotImplemented:
            return NotImplemented
        # 返回一个TaggedValue对象，表示乘法运算后的结果
        return TaggedValue(value, unit)

    # 右乘法的特殊情况，等同于self * lhs
    def __rmul__(self, lhs):
        return self * lhs

    # 将对象包装为一个TaggedValue对象，用于处理numpy数组
    def __array_wrap__(self, array, context=None, return_scalar=False):
        return TaggedValue(array, self)

    # 返回一个numpy数组，如果指定了类型t，则返回类型转换后的数组
    def __array__(self, t=None, context=None, copy=False):
        ret = np.array(1)  # 创建一个包含单个元素1的numpy数组
        if t is not None:
            return ret.astype(t)  # 如果指定了类型t，则返回类型转换后的数组
        else:
            return ret  # 否则返回原始的numpy数组

    # 添加单位到转换函数的映射关系，unit是单位，factor是转换因子
    def add_conversion_factor(self, unit, factor):
        # 定义一个转换函数convert(x)，将x乘以factor，然后将其存储在self.conversions[unit]中
        def convert(x):
            return x * factor
        self.conversions[unit] = convert

    # 添加单位到自定义转换函数的映射关系，unit是单位，fn是自定义转换函数
    def add_conversion_fn(self, unit, fn):
        # 将自定义转换函数fn存储在self.conversions[unit]中
        self.conversions[unit] = fn

    # 获取指定单位的转换函数
    def get_conversion_fn(self, unit):
        return self.conversions[unit]

    # 将给定的值转换为指定单位，使用存储在self.conversions[unit]中的转换函数
    def convert_value_to(self, value, unit):
        conversion_fn = self.conversions[unit]  # 获取指定单位的转换函数
        ret = conversion_fn(value)  # 使用转换函数将value转换为新单位下的值
        return ret

    # 获取对象的单位，这里直接返回self
    def get_unit(self):
        return self
class UnitResolver:
    # 定义用于处理单位操作的规则类
    
    def addition_rule(self, units):
        # 实现加法规则：检查单位列表中相邻单位是否相同，若不同返回NotImplemented
        for unit_1, unit_2 in zip(units[:-1], units[1:]):
            if unit_1 != unit_2:
                return NotImplemented
        # 若所有单位相同，返回第一个单位
        return units[0]

    def multiplication_rule(self, units):
        # 实现乘法规则：检查非空单位列表中是否多于一个单位，若是返回NotImplemented
        non_null = [u for u in units if u]
        if len(non_null) > 1:
            return NotImplemented
        # 若只有一个非空单位，返回该单位
        return non_null[0]

    op_dict = {
        '__mul__': multiplication_rule,  # 乘法操作对应的处理函数
        '__rmul__': multiplication_rule,  # 右乘操作对应的处理函数
        '__add__': addition_rule,  # 加法操作对应的处理函数
        '__radd__': addition_rule,  # 右加操作对应的处理函数
        '__sub__': addition_rule,  # 减法操作对应的处理函数
        '__rsub__': addition_rule}  # 右减操作对应的处理函数

    def __call__(self, operation, units):
        # 实现调用实例来执行操作的方法
        
        if operation not in self.op_dict:
            return NotImplemented
        
        # 根据操作符调用对应的处理函数
        return self.op_dict[operation](self, units)


unit_resolver = UnitResolver()

cm = BasicUnit('cm', 'centimeters')  # 创建单位对象：厘米
inch = BasicUnit('inch', 'inches')  # 创建单位对象：英寸
inch.add_conversion_factor(cm, 2.54)  # 添加英寸到厘米的转换因子
cm.add_conversion_factor(inch, 1/2.54)  # 添加厘米到英寸的转换因子

radians = BasicUnit('rad', 'radians')  # 创建单位对象：弧度
degrees = BasicUnit('deg', 'degrees')  # 创建单位对象：角度
radians.add_conversion_factor(degrees, 180.0/np.pi)  # 添加弧度到角度的转换因子
degrees.add_conversion_factor(radians, np.pi/180.0)  # 添加角度到弧度的转换因子

secs = BasicUnit('s', 'seconds')  # 创建单位对象：秒
hertz = BasicUnit('Hz', 'Hertz')  # 创建单位对象：赫兹
minutes = BasicUnit('min', 'minutes')  # 创建单位对象：分钟

secs.add_conversion_fn(hertz, lambda x: 1./x)  # 添加秒到赫兹的转换函数
secs.add_conversion_factor(minutes, 1/60.0)  # 添加秒到分钟的转换因子


# radians formatting
def rad_fn(x, pos=None):
    # 定义用于格式化弧度单位的函数

    if x >= 0:
        n = int((x / np.pi) * 2.0 + 0.25)
    else:
        n = int((x / np.pi) * 2.0 - 0.25)

    if n == 0:
        return '0'
    elif n == 1:
        return r'$\pi/2$'
    elif n == 2:
        return r'$\pi$'
    elif n == -1:
        return r'$-\pi/2$'
    elif n == -2:
        return r'$-\pi$'
    elif n % 2 == 0:
        return fr'${n//2}\pi$'
    else:
        return fr'${n}\pi/2$'


class BasicUnitConverter(units.ConversionInterface):
    @staticmethod
    def axisinfo(unit, axis):
        """Return AxisInfo instance for x and unit."""
        
        if unit == radians:
            # 若单位为弧度，返回特定的AxisInfo对象：主要刻度为π/2的倍数，格式化由rad_fn函数完成
            return units.AxisInfo(
                majloc=ticker.MultipleLocator(base=np.pi/2),
                majfmt=ticker.FuncFormatter(rad_fn),
                label=unit.fullname,
            )
        elif unit == degrees:
            # 若单位为角度，返回特定的AxisInfo对象：使用AutoLocator自动刻度，格式化为度数
            return units.AxisInfo(
                majloc=ticker.AutoLocator(),
                majfmt=ticker.FormatStrFormatter(r'$%i^\circ$'),
                label=unit.fullname,
            )
        elif unit is not None:
            if hasattr(unit, 'fullname'):
                # 若单位对象具有fullname属性，返回一个包含该属性的AxisInfo对象
                return units.AxisInfo(label=unit.fullname)
            elif hasattr(unit, 'unit'):
                # 若单位对象具有unit属性，返回一个包含unit属性的AxisInfo对象的label
                return units.AxisInfo(label=unit.unit.fullname)
        # 若单位为None或未定义情况，返回None
        return None

    @staticmethod
    # 定义一个静态方法，用于将值转换为指定单位
    def convert(val, unit, axis):
        # 检查val是否可迭代
        if np.iterable(val):
            # 如果val是MaskedArray对象，则将其转换为浮点数数组，并填充NaN值
            if isinstance(val, np.ma.MaskedArray):
                val = val.astype(float).filled(np.nan)
            # 创建一个空数组，用于存储转换后的值
            out = np.empty(len(val))
            # 遍历val数组
            for i, thisval in enumerate(val):
                # 如果当前值是Masked值，则将输出数组对应位置设为NaN
                if np.ma.is_masked(thisval):
                    out[i] = np.nan
                else:
                    try:
                        # 尝试将当前值转换为指定单位，并获取其数值
                        out[i] = thisval.convert_to(unit).get_value()
                    except AttributeError:
                        # 如果转换失败，将当前值直接赋给输出数组
                        out[i] = thisval
            return out
        # 如果val是Masked值，则返回NaN
        if np.ma.is_masked(val):
            return np.nan
        else:
            # 否则尝试将val转换为指定单位，并获取其数值
            return val.convert_to(unit).get_value()

    @staticmethod
    def default_units(x, axis):
        """Return the default unit for x or None."""
        # 检查x是否可迭代
        if np.iterable(x):
            # 遍历x中的元素，返回第一个元素的单位
            for thisx in x:
                return thisx.unit
        # 返回x的单位
        return x.unit
# 定义一个名为 cos 的函数，计算给定角度的余弦值
def cos(x):
    # 如果 x 是可迭代对象（例如列表或数组），则对每个元素计算余弦值，并转换为弧度
    if np.iterable(x):
        return [math.cos(val.convert_to(radians).get_value()) for val in x]
    # 如果 x 不是可迭代对象，直接计算 x 的余弦值，并将 x 转换为弧度
    else:
        return math.cos(x.convert_to(radians).get_value())

# 将 BasicUnit 类型和 TaggedValue 类型注册到 BasicUnitConverter() 上下文中
units.registry[BasicUnit] = units.registry[TaggedValue] = BasicUnitConverter()
```