# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\interval_arithmetic.py`

```
# 该模块提供了用于绘图的区间算术功能，但并未精确实现区间算术，因此不能用于除绘图外的其他目的。
# 如果需要使用区间算术，请使用 mpmath 库。

# Q: 为什么使用 numpy？为什么不直接使用 mpmath 的区间算术？
# A: mpmath 的区间算术模拟了浮点单位，因此速度较慢，而 numpy 的计算速度则快上许多个数量级。

# Q: 为什么要为区间创建一个单独的类？为什么不使用 SymPy 的区间集？
# A: 绘图所需的功能与区间集实现的功能有很大不同。

# Q: 为什么没有处理根据 IEEE754 进行四舍五入？
# A: 在 numpy 和 python 中都无法实现这一点。必须使用外部库来处理，这违背了速度的整体目标。
# 此外，这些库仅为很少数的函数处理四舍五入。

# Q: 我的绘图会受到影响吗？
# A: 大多数情况下不会受到影响。基于区间算术模块遇到的问题与浮点算术相同。

from sympy.core.numbers import int_valued
from sympy.core.logic import fuzzy_and
from sympy.simplify.simplify import nsimplify

from .interval_membership import intervalMembership

class interval:
    """
    表示包含浮点数起始点和结束点的区间。
    is_valid 变量跟踪函数结果作为区间是否在定义域内并且是连续的。
    - True: 表示函数结果的区间是连续的并且在函数的定义域内。
    - False: 函数参数区间不在函数的定义域内，因此函数结果区间的 is_valid 是 False。
    - None: 函数在区间上不连续或函数的参数区间部分在函数的定义域内。

    区间与实数或两个区间之间的比较可能返回 intervalMembership 的两个三值逻辑值。
    """
    # 初始化函数，用于创建一个区间对象
    def __init__(self, *args, is_valid=True, **kwargs):
        # 设置是否有效的标志位
        self.is_valid = is_valid
        # 根据传入的参数个数进行处理
        if len(args) == 1:
            # 如果只有一个参数，并且是 interval 类型的对象
            if isinstance(args[0], interval):
                # 从给定的 interval 对象中获取起始和结束值
                self.start, self.end = args[0].start, args[0].end
            else:
                # 如果参数是单个浮点数，则将起始和结束值设置为相同
                self.start = float(args[0])
                self.end = float(args[0])
        elif len(args) == 2:
            # 如果有两个参数，则根据大小确定起始和结束值
            if args[0] < args[1]:
                self.start = float(args[0])
                self.end = float(args[1])
            else:
                self.start = float(args[1])
                self.end = float(args[0])
        else:
            # 如果参数个数不是 1 或 2，则抛出异常
            raise ValueError("interval takes a maximum of two float values "
                             "as arguments")

    @property
    # 计算并返回区间的中点值
    def mid(self):
        return (self.start + self.end) / 2.0

    @property
    # 计算并返回区间的宽度
    def width(self):
        return self.end - self.start

    # 返回区间对象的字符串表示形式
    def __repr__(self):
        return "interval(%f, %f)" % (self.start, self.end)

    # 返回区间对象的字符串表示形式
    def __str__(self):
        return "[%f, %f]" % (self.start, self.end)

    # 实现区间对象与整数或浮点数的小于比较操作
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            # 如果 other 是整数或浮点数，根据起始和结束值与 other 的关系返回结果
            if self.end < other:
                return intervalMembership(True, self.is_valid)
            elif self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            # 如果 other 是区间对象，则进行模糊与运算后比较起始和结束值
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end < other.start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            # 如果类型不匹配，则返回 NotImplemented
            return NotImplemented

    # 实现区间对象与整数或浮点数的大于比较操作
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            # 如果 other 是整数或浮点数，根据起始和结束值与 other 的关系返回结果
            if self.start > other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            # 如果 other 是区间对象，则调用其 __lt__ 方法反向比较
            return other.__lt__(self)
        else:
            # 如果类型不匹配，则返回 NotImplemented
            return NotImplemented
    # 定义对象相等运算符重载方法
    def __eq__(self, other):
        # 如果 other 是 int 或 float 类型
        if isinstance(other, (int, float)):
            # 如果对象的起始和结束值都等于 other
            if self.start == other and self.end == other:
                # 返回一个指示相等的 intervalMembership 实例
                return intervalMembership(True, self.is_valid)
            # 如果 other 在当前对象表示的区间内
            if other in self:
                # 返回一个指示部分相等的 intervalMembership 实例
                return intervalMembership(None, self.is_valid)
            else:
                # 返回一个指示不相等的 intervalMembership 实例
                return intervalMembership(False, self.is_valid)

        # 如果 other 是 interval 类型
        if isinstance(other, interval):
            # 计算当前对象和 other 的有效性的模糊与运算结果
            valid = fuzzy_and([self.is_valid, other.is_valid])
            # 如果起始和结束值都相等
            if self.start == other.start and self.end == other.end:
                # 返回一个指示相等的 intervalMembership 实例
                return intervalMembership(True, valid)
            # 否则，如果当前对象小于 other
            elif self.__lt__(other)[0] is not None:
                # 返回一个指示不相等的 intervalMembership 实例
                return intervalMembership(False, valid)
            else:
                # 返回一个指示部分相等的 intervalMembership 实例
                return intervalMembership(None, valid)
        else:
            # 如果 other 不是支持的类型，返回 NotImplemented
            return NotImplemented

    # 定义对象不等运算符重载方法
    def __ne__(self, other):
        # 如果 other 是 int 或 float 类型
        if isinstance(other, (int, float)):
            # 如果对象的起始和结束值都等于 other
            if self.start == other and self.end == other:
                # 返回一个指示不相等的 intervalMembership 实例
                return intervalMembership(False, self.is_valid)
            # 如果 other 在当前对象表示的区间内
            if other in self:
                # 返回一个指示部分相等的 intervalMembership 实例
                return intervalMembership(None, self.is_valid)
            else:
                # 返回一个指示相等的 intervalMembership 实例
                return intervalMembership(True, self.is_valid)

        # 如果 other 是 interval 类型
        if isinstance(other, interval):
            # 计算当前对象和 other 的有效性的模糊与运算结果
            valid = fuzzy_and([self.is_valid, other.is_valid])
            # 如果起始和结束值都相等
            if self.start == other.start and self.end == other.end:
                # 返回一个指示不相等的 intervalMembership 实例
                return intervalMembership(False, valid)
            # 否则，如果当前对象不小于 other
            if not self.__lt__(other)[0] is None:
                # 返回一个指示相等的 intervalMembership 实例
                return intervalMembership(True, valid)
            # 返回一个指示部分相等的 intervalMembership 实例
            return intervalMembership(None, valid)
        else:
            # 如果 other 不是支持的类型，返回 NotImplemented
            return NotImplemented

    # 定义对象小于等于运算符重载方法
    def __le__(self, other):
        # 如果 other 是 int 或 float 类型
        if isinstance(other, (int, float)):
            # 如果对象的结束值小于等于 other
            if self.end <= other:
                # 返回一个指示小于等于的 intervalMembership 实例
                return intervalMembership(True, self.is_valid)
            # 如果对象的起始值大于 other
            if self.start > other:
                # 返回一个指示大于的 intervalMembership 实例
                return intervalMembership(False, self.is_valid)
            # 返回一个指示部分相等的 intervalMembership 实例
            else:
                return intervalMembership(None, self.is_valid)

        # 如果 other 是 interval 类型
        if isinstance(other, interval):
            # 计算当前对象和 other 的有效性的模糊与运算结果
            valid = fuzzy_and([self.is_valid, other.is_valid])
            # 如果对象的结束值小于等于 other 的起始值
            if self.end <= other.start:
                # 返回一个指示小于等于的 intervalMembership 实例
                return intervalMembership(True, valid)
            # 如果对象的起始值大于 other 的结束值
            if self.start > other.end:
                # 返回一个指示大于的 intervalMembership 实例
                return intervalMembership(False, valid)
            # 返回一个指示部分相等的 intervalMembership 实例
            return intervalMembership(None, valid)
        else:
            # 如果 other 不是支持的类型，返回 NotImplemented
            return NotImplemented

    # 定义对象大于等于运算符重载方法
    def __ge__(self, other):
        # 如果 other 是 int 或 float 类型
        if isinstance(other, (int, float)):
            # 如果对象的起始值大于等于 other
            if self.start >= other:
                # 返回一个指示大于等于的 intervalMembership 实例
                return intervalMembership(True, self.is_valid)
            # 否则，如果对象的结束值小于 other
            elif self.end < other:
                # 返回一个指示小于的 intervalMembership 实例
                return intervalMembership(False, self.is_valid)
            # 返回一个指示部分相等的 intervalMembership 实例
            else:
                return intervalMembership(None, self.is_valid)
        # 如果 other 是 interval 类型
        elif isinstance(other, interval):
            # 调用 other 的小于等于运算符与当前对象比较
            return other.__le__(self)
    # 定义特殊方法 __add__，实现对象与数字或另一个 interval 对象的加法操作
    def __add__(self, other):
        # 如果 other 是整数或浮点数
        if isinstance(other, (int, float)):
            # 如果当前 interval 对象有效
            if self.is_valid:
                # 返回一个新的 interval 对象，起始和结束值增加了给定的数字
                return interval(self.start + other, self.end + other)
            else:
                # 如果当前 interval 对象无效，分别计算新的起始和结束值，并返回一个新的 interval 对象
                start = self.start + other
                end = self.end + other
                return interval(start, end, is_valid=self.is_valid)

        # 如果 other 是 interval 对象
        elif isinstance(other, interval):
            # 分别计算新的起始和结束值
            start = self.start + other.start
            end = self.end + other.end
            # 使用模糊与运算确定新的有效性状态
            valid = fuzzy_and([self.is_valid, other.is_valid])
            # 返回一个新的 interval 对象，包含计算后的起始和结束值以及有效性状态
            return interval(start, end, is_valid=valid)
        else:
            # 如果不支持当前操作，返回 NotImplemented
            return NotImplemented

    # 定义特殊方法 __radd__，使对象右侧的加法操作与 __add__ 方法一致
    __radd__ = __add__

    # 定义特殊方法 __sub__，实现对象与数字或另一个 interval 对象的减法操作
    def __sub__(self, other):
        # 如果 other 是整数或浮点数
        if isinstance(other, (int, float)):
            # 分别计算新的起始和结束值，并返回一个新的 interval 对象
            start = self.start - other
            end = self.end - other
            return interval(start, end, is_valid=self.is_valid)

        # 如果 other 是 interval 对象
        elif isinstance(other, interval):
            # 分别计算新的起始和结束值
            start = self.start - other.end
            end = self.end - other.start
            # 使用模糊与运算确定新的有效性状态
            valid = fuzzy_and([self.is_valid, other.is_valid])
            # 返回一个新的 interval 对象，包含计算后的起始和结束值以及有效性状态
            return interval(start, end, is_valid=valid)
        else:
            # 如果不支持当前操作，返回 NotImplemented
            return NotImplemented

    # 定义特殊方法 __rsub__，使对象右侧的减法操作与 __sub__ 方法一致
    def __rsub__(self, other):
        # 如果 other 是整数或浮点数
        if isinstance(other, (int, float)):
            # 分别计算新的起始和结束值，并返回一个新的 interval 对象
            start = other - self.end
            end = other - self.start
            return interval(start, end, is_valid=self.is_valid)
        # 如果 other 是 interval 对象
        elif isinstance(other, interval):
            # 调用 other 对象的 __sub__ 方法，并返回其结果
            return other.__sub__(self)
        else:
            # 如果不支持当前操作，返回 NotImplemented
            return NotImplemented

    # 定义特殊方法 __neg__，实现对象的取负操作
    def __neg__(self):
        # 如果当前 interval 对象有效
        if self.is_valid:
            # 返回一个新的 interval 对象，其起始和结束值均为当前对象的相反数
            return interval(-self.end, -self.start)
        else:
            # 返回一个新的 interval 对象，其起始和结束值均为当前对象的相反数，并保持无效性状态
            return interval(-self.end, -self.start, is_valid=self.is_valid)

    # 定义特殊方法 __mul__，实现对象与数字或另一个 interval 对象的乘法操作
    def __mul__(self, other):
        # 如果 other 是 interval 对象
        if isinstance(other, interval):
            # 如果任意一个 interval 对象无效，返回一个无效的 interval 对象
            if self.is_valid is False or other.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            # 如果任意一个 interval 对象未定义，返回一个未定义的 interval 对象
            elif self.is_valid is None or other.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
                # 计算所有可能的交集点，并确定新的起始和结束值
                inters = []
                inters.append(self.start * other.start)
                inters.append(self.end * other.start)
                inters.append(self.start * other.end)
                inters.append(self.end * other.end)
                start = min(inters)
                end = max(inters)
                # 返回一个新的 interval 对象，包含计算后的起始和结束值
                return interval(start, end)
        # 如果 other 是整数或浮点数
        elif isinstance(other, (int, float)):
            # 返回一个新的 interval 对象，起始和结束值分别为当前对象的起始和结束值乘以给定的数字
            return interval(self.start*other, self.end*other, is_valid=self.is_valid)
        else:
            # 如果不支持当前操作，返回 NotImplemented
            return NotImplemented

    # 定义特殊方法 __rmul__，使对象右侧的乘法操作与 __mul__ 方法一致
    __rmul__ = __mul__

    # 定义特殊方法 __contains__，实现 in 运算符的功能
    def __contains__(self, other):
        # 如果 other 是整数或浮点数
        if isinstance(other, (int, float)):
            # 检查给定的数值是否在当前 interval 对象的范围内
            return self.start <= other and self.end >= other
        else:
            # 如果 other 是 interval 对象，检查其范围是否完全包含在当前 interval 对象的范围内
            return self.start <= other.start and other.end <= self.end
    def __rtruediv__(self, other):
        # 如果 other 是 int 或 float 类型，则转换为 interval 对象并调用其 __truediv__ 方法
        if isinstance(other, (int, float)):
            other = interval(other)
            return other.__truediv__(self)
        elif isinstance(other, interval):
            # 如果 other 是 interval 对象，则调用其 __truediv__ 方法
            return other.__truediv__(self)
        else:
            # 如果 other 类型不符合预期，则返回 NotImplemented
            return NotImplemented

    def __truediv__(self, other):
        # 检查当前 interval 是否有效
        if not self.is_valid:
            # 如果当前 interval 无效，则返回一个范围为 (-inf, inf) 的 interval，is_valid 根据 self.is_valid 确定
            return interval(-float('inf'), float('inf'), is_valid=self.is_valid)
        if isinstance(other, (int, float)):
            if other == 0:
                # 如果除数是 0，则返回一个无效的 interval，范围为 (-inf, inf)，is_valid 设置为 False
                return interval(-float('inf'), float('inf'), is_valid=False)
            else:
                # 如果除数是数值类型且不为 0，则返回除法运算结果的 interval
                return interval(self.start / other, self.end / other)

        elif isinstance(other, interval):
            # 如果除数是 interval 类型
            if other.is_valid is False or self.is_valid is False:
                # 如果任一 interval 无效，则返回一个无效的 interval，范围为 (-inf, inf)，is_valid 设置为 False
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif other.is_valid is None or self.is_valid is None:
                # 如果任一 interval 的有效性未确定，则返回一个无效的 interval，范围为 (-inf, inf)，is_valid 设置为 None
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
                # 计算两个 interval 的除法结果
                if 0 in other:
                    # 如果分母包含 0，则返回整个实数轴的 interval，is_valid 设置为 None
                    return interval(-float('inf'), float('inf'), is_valid=None)

                # 处理分母为负数的情况
                this = self
                if other.end < 0:
                    this = -this
                    other = -other

                # 处理分母为正数的情况，计算交集结果的起始和结束值
                inters = []
                inters.append(this.start / other.start)
                inters.append(this.end / other.start)
                inters.append(this.start / other.end)
                inters.append(this.end / other.end)
                start = max(inters)
                end = min(inters)
                return interval(start, end)
        else:
            # 如果除数类型不符合预期，则返回 NotImplemented
            return NotImplemented

    def __pow__(self, other):
        # 只实现整数指数的幂运算
        from .lib_interval import exp, log
        if not self.is_valid:
            # 如果当前 interval 无效，则直接返回自身
            return self
        if isinstance(other, interval):
            # 如果指数 other 是 interval 类型，则计算 exp(other * log(self)) 的结果
            return exp(other * log(self))
        elif isinstance(other, (float, int)):
            if other < 0:
                # 如果指数是负数，则返回 self 的倒数的 abs(other) 次幂
                return 1 / self.__pow__(abs(other))
            else:
                if int_valued(other):
                    # 如果指数是整数，则调用 _pow_int 方法计算 self 的整数次幂
                    return _pow_int(self, other)
                else:
                    # 如果指数是浮点数，则调用 _pow_float 方法计算 self 的浮点数次幂
                    return _pow_float(self, other)
        else:
            # 如果指数类型不符合预期，则返回 NotImplemented
            return NotImplemented
    # 定义自定义类的右幂运算符方法，用于处理右边为整数或浮点数的情况
    def __rpow__(self, other):
        # 检查 other 是否为整数或浮点数
        if isinstance(other, (float, int)):
            # 如果当前对象无效，直接返回自身
            if not self.is_valid:
                # 不执行任何操作，直接返回当前对象
                return self
            # 如果 other 小于 0
            elif other < 0:
                # 如果当前区间的宽度大于 0
                if self.width > 0:
                    # 返回一个无效的区间，表示负无穷到正无穷
                    return interval(-float('inf'), float('inf'), is_valid=False)
                else:
                    # 将 self.start 简化为最简有理数
                    power_rational = nsimplify(self.start)
                    num, denom = power_rational.as_numer_denom()
                    # 如果分母为偶数，返回一个无效区间，表示负无穷到正无穷
                    if denom % 2 == 0:
                        return interval(-float('inf'), float('inf'), is_valid=False)
                    else:
                        # 计算幂运算结果并返回一个区间对象
                        start = -abs(other) ** self.start
                        end = start
                        return interval(start, end)
            else:
                # 返回 other 的 self.start 和 self.end 次幂所组成的区间对象
                return interval(other ** self.start, other ** self.end)
        # 如果 other 是 interval 类型的对象，则调用其 __pow__ 方法进行处理
        elif isinstance(other, interval):
            return other.__pow__(self)
        else:
            # 如果 other 不是支持的类型，则返回 NotImplemented
            return NotImplemented

    # 定义自定义类的哈希方法，返回对象的哈希值
    def __hash__(self):
        return hash((self.is_valid, self.start, self.end))
# 计算一个区间的浮点幂
def _pow_float(inter, power):
    # 将幂转换为最简有理数形式
    power_rational = nsimplify(power)
    # 将幂表示为分子和分母
    num, denom = power_rational.as_numer_denom()
    
    # 如果幂是偶数
    if num % 2 == 0:
        # 计算区间起始和结束点的幂值
        start = abs(inter.start)**power
        end = abs(inter.end)**power
        # 如果起始值小于零，返回一个区间从零到最大值
        if start < 0:
            ret = interval(0, max(start, end))
        else:
            ret = interval(start, end)
        return ret
    
    # 如果分母是偶数
    elif denom % 2 == 0:
        # 如果区间的结束点小于零，返回一个无效的区间
        if inter.end < 0:
            return interval(-float('inf'), float('inf'), is_valid=False)
        # 如果区间的起始点小于零，返回一个区间从零到结束点的幂值
        elif inter.start < 0:
            return interval(0, inter.end**power, is_valid=None)
        else:
            return interval(inter.start**power, inter.end**power)
    
    # 如果幂不是偶数也不是分母是偶数的情况
    else:
        # 计算区间起始和结束点的幂值，考虑起始点为负数的情况
        if inter.start < 0:
            start = -abs(inter.start)**power
        else:
            start = inter.start**power
        
        if inter.end < 0:
            end = -abs(inter.end)**power
        else:
            end = inter.end**power
        
        # 返回计算后的区间，指定其是否有效
        return interval(start, end, is_valid=inter.is_valid)


# 计算一个区间的整数幂
def _pow_int(inter, power):
    # 将幂转换为整数
    power = int(power)
    
    # 如果幂为奇数
    if power & 1:
        # 返回区间起始和结束点的幂值
        return interval(inter.start**power, inter.end**power)
    
    # 如果幂为偶数
    else:
        # 如果区间的起始点小于零且结束点大于零，返回从零到最大幂值的区间
        if inter.start < 0 and inter.end > 0:
            start = 0
            end = max(inter.start**power, inter.end**power)
            return interval(start, end)
        else:
            # 返回区间起始和结束点的幂值
            return interval(inter.start**power, inter.end**power)
```