# `D:\src\scipysrc\scipy\scipy\stats\distributions.py`

```
#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
# NOTE: To look at history using `git blame`, use `git blame -M -C -C`
#       instead of `git blame -Lxxx,+x`.
#

# 从 _distn_infrastructure 模块中导入 rv_discrete, rv_continuous, rv_frozen 类
# noqa: F401 表示忽略 F401 错误（未使用的导入）

from ._distn_infrastructure import (rv_discrete, rv_continuous, rv_frozen)  # noqa: F401

# 从当前包中导入 _continuous_distns 和 _discrete_distns 模块
from . import _continuous_distns
from . import _discrete_distns

# 从 _continuous_distns 模块导入所有内容（除了 F403 错误，禁止导入的警告）
from ._continuous_distns import *  # noqa: F403

# 导入 levy_stable 函数
from ._levy_stable import levy_stable

# 从 _discrete_distns 模块导入所有内容（除了 F403 错误，禁止导入的警告）
from ._discrete_distns import *  # noqa: F403

# 从 _entropy 模块中导入 entropy 函数

from ._entropy import entropy

# 为了向后兼容性（例如 pymc 需要 distributions.__all__），定义 __all__ 列表包含以下标识符
# noqa: F405 表示忽略 F405 错误（__all__ 未定义的警告）

__all__ = ['rv_discrete', 'rv_continuous', 'rv_histogram', 'entropy']  # noqa: F405

# 向 __all__ 列表添加 _continuous_distns 模块中的分布名字（不包括 *_gen 名字）
__all__ += _continuous_distns._distn_names

# 向 __all__ 列表添加 levy_stable 标识符
__all__ += ['levy_stable']

# 向 __all__ 列表添加 _discrete_distns 模块中的分布名字
__all__ += _discrete_distns._distn_names
```