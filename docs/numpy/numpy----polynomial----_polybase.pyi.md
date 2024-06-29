# `D:\src\scipysrc\numpy\numpy\polynomial\_polybase.pyi`

```py
# 导入 abc 模块，用于定义抽象基类
import abc
# 导入 Any 和 ClassVar 类型提示
from typing import Any, ClassVar

# 定义模块级别变量 __all__，用于控制模块导出的内容
__all__: list[str]

# 定义抽象基类 ABCPolyBase，继承自 abc.ABC
class ABCPolyBase(abc.ABC):
    # 类变量 __hash__，标识为不可赋值类型，用于禁止赋值类型检查
    __hash__: ClassVar[None]  # type: ignore[assignment]
    # 类变量 __array_ufunc__
    __array_ufunc__: ClassVar[None]
    # 类变量 maxpower，表示最大的幂次
    maxpower: ClassVar[int]
    # 实例变量 coef，表示多项式的系数
    coef: Any

    # 属性方法 symbol，返回多项式的符号表示
    @property
    def symbol(self) -> str: ...

    # 抽象属性方法 domain，表示多项式的定义域
    @property
    @abc.abstractmethod
    def domain(self): ...

    # 抽象属性方法 window，表示多项式的窗口
    @property
    @abc.abstractmethod
    def window(self): ...

    # 抽象属性方法 basis_name，表示多项式的基函数名称
    @property
    @abc.abstractmethod
    def basis_name(self): ...

    # 实例方法 has_samecoef，用于比较两个多项式是否具有相同的系数
    def has_samecoef(self, other): ...

    # 实例方法 has_samedomain，用于比较两个多项式是否具有相同的定义域
    def has_samedomain(self, other): ...

    # 实例方法 has_samewindow，用于比较两个多项式是否具有相同的窗口
    def has_samewindow(self, other): ...

    # 实例方法 has_sametype，用于比较两个多项式是否具有相同的类型
    def has_sametype(self, other): ...

    # 初始化方法，接受系数 coef，可选参数 domain、window、symbol
    def __init__(self, coef, domain=..., window=..., symbol: str = ...) -> None: ...

    # 格式化方法，定义多项式的格式化输出方式
    def __format__(self, fmt_str): ...

    # 调用方法，使多项式实例能够像函数一样被调用
    def __call__(self, arg): ...

    # 迭代方法，使多项式实例能够被迭代
    def __iter__(self): ...

    # 返回多项式的长度，通常指其维度或长度
    def __len__(self): ...

    # 负号运算方法，定义多项式的负号运算
    def __neg__(self): ...

    # 正号运算方法，定义多项式的正号运算
    def __pos__(self): ...

    # 加法运算方法，定义多项式的加法运算
    def __add__(self, other): ...

    # 减法运算方法，定义多项式的减法运算
    def __sub__(self, other): ...

    # 乘法运算方法，定义多项式的乘法运算
    def __mul__(self, other): ...

    # 真除法运算方法，定义多项式的真除法运算
    def __truediv__(self, other): ...

    # 向下整除运算方法，定义多项式的向下整除运算
    def __floordiv__(self, other): ...

    # 取模运算方法，定义多项式的取模运算
    def __mod__(self, other): ...

    # 除法和取模运算的组合方法，定义多项式的除法和取模运算
    def __divmod__(self, other): ...

    # 幂运算方法，定义多项式的幂运算
    def __pow__(self, other): ...

    # 右加法运算方法，定义多项式的右加法运算
    def __radd__(self, other): ...

    # 右减法运算方法，定义多项式的右减法运算
    def __rsub__(self, other): ...

    # 右乘法运算方法，定义多项式的右乘法运算
    def __rmul__(self, other): ...

    # 右除法运算方法，定义多项式的右除法运算
    def __rdiv__(self, other): ...

    # 右真除法运算方法，定义多项式的右真除法运算
    def __rtruediv__(self, other): ...

    # 右向下整除运算方法，定义多项式的右向下整除运算
    def __rfloordiv__(self, other): ...

    # 右取模运算方法，定义多项式的右取模运算
    def __rmod__(self, other): ...

    # 右除法和取模运算的组合方法，定义多项式的右除法和取模运算
    def __rdivmod__(self, other): ...

    # 等于运算方法，定义多项式的等于比较运算
    def __eq__(self, other): ...

    # 不等于运算方法，定义多项式的不等于比较运算
    def __ne__(self, other): ...

    # 复制方法，返回多项式的副本
    def copy(self): ...

    # 返回多项式的次数（或阶数）
    def degree(self): ...

    # 截取多项式的次数，返回次数不超过给定值的多项式
    def cutdeg(self, deg): ...

    # 修剪多项式，移除系数绝对值小于给定容差的项
    def trim(self, tol=...): ...

    # 截断多项式，返回前定长的多项式
    def truncate(self, size): ...

    # 转换多项式的定义域、类型和窗口
    def convert(self, domain=..., kind=..., window=...): ...

    # 映射多项式的参数
    def mapparms(self): ...

    # 积分方法，计算多项式的积分
    def integ(self, m=..., k = ..., lbnd=...): ...

    # 导数方法，计算多项式的导数
    def deriv(self, m=...): ...

    # 计算多项式的根
    def roots(self): ...

    # 在给定域内生成多项式的线性间隔
    def linspace(self, n=..., domain=...): ...

    # 类方法，拟合方法，用于拟合多项式到给定数据
    @classmethod
    def fit(cls, x, y, deg, domain=..., rcond=..., full=..., w=..., window=...): ...

    # 类方法，根据根生成多项式
    @classmethod
    def fromroots(cls, roots, domain = ..., window=...): ...

    # 类方法，返回单位多项式
    @classmethod
    def identity(cls, domain=..., window=...): ...

    # 类方法，返回给定阶数的基函数多项式
    @classmethod
    def basis(cls, deg, domain=..., window=...): ...

    # 类方法，将序列转换为多项式
    @classmethod
    def cast(cls, series, domain=..., window=...): ...
```