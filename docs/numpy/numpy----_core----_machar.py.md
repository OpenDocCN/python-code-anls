# `.\numpy\numpy\_core\_machar.py`

```py
"""
Machine arithmetic - determine the parameters of the
floating-point arithmetic system

Author: Pearu Peterson, September 2003

"""
# 定义一个列表，包含在 fromnumeric 模块中导出的任何函数或类
__all__ = ['MachAr']

# 从 fromnumeric 模块导入 any 函数
from .fromnumeric import any
# 从 _ufunc_config 模块导入 errstate 函数
from ._ufunc_config import errstate
# 从 .._utils 模块导入 set_module 函数
from .._utils import set_module

# 需要加快速度...特别是对于 longdouble

# 2021-10-20 弃用，NumPy 1.22
class MachAr:
    """
    诊断机器参数。

    Attributes
    ----------
    ibeta : int
        表示数字的基数。
    it : int
        浮点数尾数 M 中基数 `ibeta` 的位数。
    machep : int
        最小（最负）的 `ibeta` 的幂的指数，加到 1.0 上会得到与 1.0 不同的结果。
    eps : float
        浮点数 ``beta**machep``（浮点精度）。
    negep : int
        最小的 `ibeta` 的幂的指数，从 1.0 减去会得到与 1.0 不同的结果。
    epsneg : float
        浮点数 ``beta**negep``。
    iexp : int
        指数中的位数（包括其符号和偏置）。
    minexp : int
        与尾数中没有前导零一致的最小（最负）的 `ibeta` 的幂。
    xmin : float
        浮点数 ``beta**minexp``（具有完整精度的最小正浮点数）。
    maxexp : int
        导致溢出的最小（正）的 `ibeta` 的幂。
    xmax : float
        ``(1-epsneg) * beta**maxexp``（可用的最大（按数量级）浮点值）。
    irnd : int
        在 ``range(6)`` 中，有关舍入方式和如何处理下溢的信息。
    ngrd : int
        在截断两个尾数以适应表示时使用的“保护位”数量。
    epsilon : float
        与 `eps` 相同。
    tiny : float
        `smallest_normal` 的别名，保持向后兼容性。
    huge : float
        与 `xmax` 相同。
    precision : float
        ``- int(-log10(eps))``
    resolution : float
        ``- 10**(-precision)``
    smallest_normal : float
        遵循 IEEE-754 标准，在尾数中有 1 作为首位的最小正浮点数。与 `xmin` 相同。
    smallest_subnormal : float
        遵循 IEEE-754 标准，在尾数中有 0 作为首位的最小正浮点数。

    Parameters
    ----------
    float_conv : function, optional
        将整数或整数数组转换为浮点数或浮点数数组的函数。默认为 `float`。
    int_conv : function, optional
        将浮点数或浮点数数组转换为整数或整数数组的函数。默认为 `int`。
    float_to_float : function, optional
        将浮点数数组转换为浮点数的函数。默认为 `float`。
        注意，在当前实现中，这似乎没有任何有用的作用。
    def __init__(self, float_conv=float, int_conv=int,
                 float_to_float=float,
                 float_to_str=lambda v:'%24.16e' % v,
                 title='Python floating point number'):
        """
        初始化函数，用于创建一个 MachAr 对象。

        Parameters:
        ----------
        float_conv : function, optional
            用于将整数转换为浮点数的函数（数组）。默认为 `float`。
        int_conv : function, optional
            用于将浮点数（数组）转换为整数的函数。默认为 `int`。
        float_to_float : function, optional
            用于将浮点数数组转换为浮点数的函数。默认为 `float`。
        float_to_str : function, optional
            用于将浮点数数组转换为字符串的函数。默认为 lambda 函数 `lambda v:'%24.16e' % v`。
        title : str, optional
            MachAr 对象的标题，将在其字符串表示中打印出来。默认为 'Python floating point number'。
        """
        # 在此处忽略所有错误，因为我们有意触发下溢以检测运行架构的特性。
        with errstate(under='ignore'):
            # 调用内部方法 `_do_init` 进行初始化设置
            self._do_init(float_conv, int_conv, float_to_float, float_to_str, title)

    def __str__(self):
        """
        返回 MachAr 对象的字符串表示形式。

        Returns:
        -------
        str
            MachAr 对象的字符串表示形式，包含各种机器参数的详细信息。
        """
        # 定义格式化字符串模板
        fmt = (
           'Machine parameters for %(title)s\n'
           '---------------------------------------------------------------------\n'
           'ibeta=%(ibeta)s it=%(it)s iexp=%(iexp)s ngrd=%(ngrd)s irnd=%(irnd)s\n'
           'machep=%(machep)s     eps=%(_str_eps)s (beta**machep == epsilon)\n'
           'negep =%(negep)s  epsneg=%(_str_epsneg)s (beta**epsneg)\n'
           'minexp=%(minexp)s   xmin=%(_str_xmin)s (beta**minexp == tiny)\n'
           'maxexp=%(maxexp)s    xmax=%(_str_xmax)s ((1-epsneg)*beta**maxexp == huge)\n'
           'smallest_normal=%(smallest_normal)s    '
           'smallest_subnormal=%(smallest_subnormal)s\n'
           '---------------------------------------------------------------------\n'
           )
        # 使用格式化字符串和对象的属性字典返回字符串表示形式
        return fmt % self.__dict__
# 如果当前脚本作为主程序执行（而不是被导入到其他模块），则执行以下代码块
if __name__ == '__main__':
    # 打印调用 MachAr() 函数的结果
    print(MachAr())
```