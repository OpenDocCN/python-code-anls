# `D:\src\scipysrc\sympy\sympy\core\__init__.py`

```
# 核心模块。提供 sympy 中所需的基本操作。

from .sympify import sympify, SympifyError  # 导入 sympify 函数和异常 SympifyError
from .cache import cacheit  # 导入缓存函数 cacheit
from .assumptions import assumptions, check_assumptions, failing_assumptions, common_assumptions  # 导入与假设相关的函数和常量
from .basic import Basic, Atom  # 导入基本类 Basic 和原子类 Atom
from .singleton import S  # 导入单例模块 S
from .expr import Expr, AtomicExpr, UnevaluatedExpr  # 导入表达式相关类
from .symbol import Symbol, Wild, Dummy, symbols, var  # 导入符号类相关和创建符号的函数
from .numbers import Number, Float, Rational, Integer, NumberSymbol, \
    RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo, \
    AlgebraicNumber, comp, mod_inverse  # 导入数字相关和数学常数
from .power import Pow  # 导入幂运算 Pow 类
from .intfunc import integer_nthroot, integer_log, num_digits, trailing  # 导入整数函数
from .mul import Mul, prod  # 导入乘法相关类和函数
from .add import Add  # 导入加法类
from .mod import Mod  # 导入取模类
from .relational import ( Rel, Eq, Ne, Lt, Le, Gt, Ge,  # 导入关系运算符类和函数
    Equality, GreaterThan, LessThan, Unequality, StrictGreaterThan,
    StrictLessThan )
from .multidimensional import vectorize  # 导入向量化函数
from .function import Lambda, WildFunction, Derivative, diff, FunctionClass, \
    Function, Subs, expand, PoleError, count_ops, \
    expand_mul, expand_log, expand_func, \
    expand_trig, expand_complex, expand_multinomial, nfloat, \
    expand_power_base, expand_power_exp, arity  # 导入函数相关类和函数
from .evalf import PrecisionExhausted, N  # 导入数值计算相关类和函数
from .containers import Tuple, Dict  # 导入容器类 Tuple 和 Dict
from .exprtools import gcd_terms, factor_terms, factor_nc  # 导入表达式工具函数
from .parameters import evaluate  # 导入参数评估函数
from .kind import UndefinedKind, NumberKind, BooleanKind  # 导入类型定义
from .traversal import preorder_traversal, bottom_up, use, postorder_traversal  # 导入树遍历函数
from .sorting import default_sort_key, ordered  # 导入排序相关函数

# 暴露单例常数
Catalan = S.Catalan
EulerGamma = S.EulerGamma
GoldenRatio = S.GoldenRatio
TribonacciConstant = S.TribonacciConstant

__all__ = [
    'sympify', 'SympifyError',  # 导出的函数和异常

    'cacheit',  # 导出的函数

    'assumptions', 'check_assumptions', 'failing_assumptions',
    'common_assumptions',  # 导出的假设相关

    'Basic', 'Atom',  # 导出的基本类

    'S',  # 导出的单例

    'Expr', 'AtomicExpr', 'UnevaluatedExpr',  # 导出的表达式类

    'Symbol', 'Wild', 'Dummy', 'symbols', 'var',  # 导出的符号相关

    'Number', 'Float', 'Rational', 'Integer', 'NumberSymbol', 'RealNumber',
    'igcd', 'ilcm', 'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo',
    'AlgebraicNumber', 'comp', 'mod_inverse',  # 导出的数字相关和数学常数

    'Pow',  # 导出的幂运算类

    'integer_nthroot', 'integer_log', 'num_digits', 'trailing',  # 导出的整数函数

    'Mul', 'prod',  # 导出的乘法相关

    'Add',  # 导出的加法类

    'Mod',  # 导出的取模类

    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'GreaterThan',
    'LessThan', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',  # 导出的关系运算符类

    'vectorize',  # 导出的向量化函数

    'Lambda', 'WildFunction', 'Derivative', 'diff', 'FunctionClass',
    'Function', 'Subs', 'expand', 'PoleError', 'count_ops', 'expand_mul',
    'expand_log', 'expand_func', 'expand_trig', 'expand_complex',
    'expand_multinomial', 'nfloat', 'expand_power_base', 'expand_power_exp',
    'arity',  # 导出的函数相关类和函数

    'PrecisionExhausted', 'N',  # 导出的数值计算相关类和函数

    'evalf',  # 导出的模块？（注释部分可能有误）

    'Tuple', 'Dict',  # 导出的容器类

    'gcd_terms', 'factor_terms', 'factor_nc',  # 导出的表达式工具函数

    'evaluate',  # 导出的参数评估函数

    'Catalan',
    'EulerGamma',
    'GoldenRatio',
    'TribonacciConstant',  # 导出的常数
]
    # 下面列出了一些字符串，似乎是某种类别或类型的名称
    'UndefinedKind', 'NumberKind', 'BooleanKind',
    
    # 下面列出了一些函数或方法的名称，可能是程序中定义的操作或功能
    'preorder_traversal', 'bottom_up', 'use', 'postorder_traversal',
    
    # 下面列出了一些变量或属性的名称，可能是用于排序或其他数据处理中的标记
    'default_sort_key', 'ordered',
]
`
# 这里`
```