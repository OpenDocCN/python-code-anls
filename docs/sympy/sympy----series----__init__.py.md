# `D:\src\scipysrc\sympy\sympy\series\__init__.py`

```
"""A module that handles series: find a limit, order the series etc.
"""
# 导入所需模块和类

from .order import Order  # 从当前包中导入 Order 类
from .limits import limit, Limit  # 从当前包中导入 limit 和 Limit 函数或类
from .gruntz import gruntz  # 从当前包中导入 gruntz 函数
from .series import series  # 从当前包中导入 series 函数
from .approximants import approximants  # 从当前包中导入 approximants 函数
from .residues import residue  # 从当前包中导入 residue 函数
from .sequences import (SeqPer, SeqFormula, sequence, SeqAdd, SeqMul)  # 从当前包中导入多个类或函数
from .fourier import fourier_series  # 从当前包中导入 fourier_series 函数
from .formal import fps  # 从当前包中导入 fps 函数
from .limitseq import difference_delta, limit_seq  # 从当前包中导入 difference_delta 和 limit_seq 函数

from sympy.core.singleton import S  # 导入 sympy 库中的 S 单例对象
EmptySequence = S.EmptySequence  # 将 S 库中的 EmptySequence 赋值给当前变量 EmptySequence

O = Order  # 将 Order 类赋值给变量 O

__all__ = ['Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants',
        'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence',
        'SeqAdd', 'SeqMul', 'fourier_series', 'fps', 'difference_delta',
        'limit_seq'
        ]
# 将所有公开的名称放入 __all__ 列表，用于模块的导入控制
```