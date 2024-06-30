# `D:\src\scipysrc\sympy\sympy\physics\quantum\__init__.py`

```
# 定义模块中公开的所有符号列表
__all__ = [
    'AntiCommutator',                    # 符号：反对易子
    'qapply',                            # 函数：对量子操作符应用
    'Commutator',                        # 符号：对易子
    'Dagger',                            # 类：共轭转置算符

    'HilbertSpaceError', 'HilbertSpace', 'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace', 'DirectSumHilbertSpace', 'ComplexSpace', 'L2',
    'FockSpace',                          # 类：希尔伯特空间相关定义

    'InnerProduct',                       # 函数：内积计算

    'Operator', 'HermitianOperator', 'UnitaryOperator', 'IdentityOperator',
    'OuterProduct', 'DifferentialOperator', # 类：不同类型的算符

    'represent', 'rep_innerproduct', 'rep_expectation', 'integrate_result',
    'get_basis', 'enumerate_states',      # 函数：表示、内积、期望值等操作

    'KetBase', 'BraBase', 'StateBase', 'State', 'Ket', 'Bra', 'TimeDepState',
    'TimeDepBra', 'TimeDepKet', 'OrthogonalKet', 'OrthogonalBra',
    'OrthogonalState', 'Wavefunction',    # 类：量子态相关定义

    'TensorProduct', 'tensor_product_simp', # 函数：张量积及简化操作

    'hbar', 'HBar',                       # 常数：哈密顿量标准常数
]
from .anticommutator import AntiCommutator         # 导入：反对易子
from .qapply import qapply                         # 导入：量子操作应用函数
from .commutator import Commutator                 # 导入：对易子
from .dagger import Dagger                         # 导入：共轭转置算符

from .hilbert import (HilbertSpaceError, HilbertSpace,
        TensorProductHilbertSpace, TensorPowerHilbertSpace,
        DirectSumHilbertSpace, ComplexSpace, L2, FockSpace)   # 导入：希尔伯特空间相关定义

from .innerproduct import InnerProduct             # 导入：内积计算函数

from .operator import (Operator, HermitianOperator, UnitaryOperator,
        IdentityOperator, OuterProduct, DifferentialOperator)  # 导入：不同类型的算符

from .represent import (represent, rep_innerproduct, rep_expectation,
        integrate_result, get_basis, enumerate_states)         # 导入：表示、内积、期望值等操作函数

from .state import (KetBase, BraBase, StateBase, State, Ket, Bra,
        TimeDepState, TimeDepBra, TimeDepKet, OrthogonalKet,
        OrthogonalBra, OrthogonalState, Wavefunction)         # 导入：量子态相关定义

from .tensorproduct import TensorProduct, tensor_product_simp   # 导入：张量积及简化操作函数

from .constants import hbar, HBar                   # 导入：哈密顿量标准常数
```