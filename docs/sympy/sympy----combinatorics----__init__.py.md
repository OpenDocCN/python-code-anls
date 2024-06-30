# `D:\src\scipysrc\sympy\sympy\combinatorics\__init__.py`

```
# 导入 SymPy 中的排列组合模块中的特定类和函数

from sympy.combinatorics.permutations import Permutation, Cycle  # 导入排列和循环的类
from sympy.combinatorics.prufer import Prufer  # 导入 Prüfer 序列相关的函数
from sympy.combinatorics.generators import cyclic, alternating, symmetric, dihedral  # 导入生成器函数
from sympy.combinatorics.subsets import Subset  # 导入子集相关的函数
from sympy.combinatorics.partitions import (Partition, IntegerPartition,  # 导入分区相关的类和函数
    RGS_rank, RGS_unrank, RGS_enum)
from sympy.combinatorics.polyhedron import (Polyhedron, tetrahedron, cube,  # 导入多面体相关的类和函数
    octahedron, dodecahedron, icosahedron)
from sympy.combinatorics.perm_groups import PermutationGroup, Coset, SymmetricPermutationGroup  # 导入置换群相关的类
from sympy.combinatorics.group_constructs import DirectProduct  # 导入直积构造函数
from sympy.combinatorics.graycode import GrayCode  # 导入格雷码相关的类和函数
from sympy.combinatorics.named_groups import (SymmetricGroup, DihedralGroup,  # 导入命名群体相关的类
    CyclicGroup, AlternatingGroup, AbelianGroup, RubikGroup)
from sympy.combinatorics.pc_groups import PolycyclicGroup, Collector  # 导入多项循环群相关的类
from sympy.combinatorics.free_groups import free_group  # 导入自由群相关的函数

__all__ = [
    'Permutation', 'Cycle',  # 将 Permutation 和 Cycle 添加到 __all__ 列表中

    'Prufer',  # 将 Prufer 添加到 __all__ 列表中

    'cyclic', 'alternating', 'symmetric', 'dihedral',  # 将生成器函数添加到 __all__ 列表中

    'Subset',  # 将 Subset 添加到 __all__ 列表中

    'Partition', 'IntegerPartition', 'RGS_rank', 'RGS_unrank', 'RGS_enum',  # 将分区相关函数添加到 __all__ 列表中

    'Polyhedron', 'tetrahedron', 'cube', 'octahedron', 'dodecahedron',  # 将多面体相关函数添加到 __all__ 列表中
    'icosahedron',

    'PermutationGroup', 'Coset', 'SymmetricPermutationGroup',  # 将置换群相关类添加到 __all__ 列表中

    'DirectProduct',  # 将 DirectProduct 添加到 __all__ 列表中

    'GrayCode',  # 将 GrayCode 添加到 __all__ 列表中

    'SymmetricGroup', 'DihedralGroup', 'CyclicGroup', 'AlternatingGroup',  # 将命名群体相关类添加到 __all__ 列表中
    'AbelianGroup', 'RubikGroup',

    'PolycyclicGroup', 'Collector',  # 将多项循环群相关类添加到 __all__ 列表中

    'free_group',  # 将 free_group 添加到 __all__ 列表中
]
```