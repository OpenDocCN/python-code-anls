# `.\pytorch\benchmarks\instruction_counts\definitions\setup.py`

```py
"""Define some common setup blocks which benchmarks can reuse."""

import enum  # 导入枚举类型模块

from core.api import GroupedSetup  # 从core.api模块导入GroupedSetup类
from core.utils import parse_stmts  # 从core.utils模块导入parse_stmts函数


_TRIVIAL_2D = GroupedSetup(r"x = torch.ones((4, 4))", r"auto x = torch::ones({4, 4});")
# 定义一个包含两个语句的GroupedSetup对象，用于初始化一个2维张量x


_TRIVIAL_3D = GroupedSetup(
    r"x = torch.ones((4, 4, 4))", r"auto x = torch::ones({4, 4, 4});"
)
# 定义一个包含两个语句的GroupedSetup对象，用于初始化一个3维张量x


_TRIVIAL_4D = GroupedSetup(
    r"x = torch.ones((4, 4, 4, 4))", r"auto x = torch::ones({4, 4, 4, 4});"
)
# 定义一个包含两个语句的GroupedSetup对象，用于初始化一个4维张量x


_TRAINING = GroupedSetup(
    *parse_stmts(
        r"""
        Python                                   | C++
        ---------------------------------------- | ----------------------------------------
        # Inputs                                 | // Inputs
        x = torch.ones((1,))                     | auto x = torch::ones({1});
        y = torch.ones((1,))                     | auto y = torch::ones({1});
                                                 |
        # Weights                                | // Weights
        w0 = torch.ones(                         | auto w0 = torch::ones({1});
            (1,), requires_grad=True)            | w0.set_requires_grad(true);
        w1 = torch.ones(                         | auto w1 = torch::ones({1});
            (1,), requires_grad=True)            | w1.set_requires_grad(true);
        w2 = torch.ones(                         | auto w2 = torch::ones({2});
            (2,), requires_grad=True)            | w2.set_requires_grad(true);
    """
    )
)
# 定义一个包含多个语句的GroupedSetup对象，用于初始化训练过程中的变量和权重设置


class Setup(enum.Enum):
    TRIVIAL_2D = _TRIVIAL_2D  # 枚举类型，表示一个简单的2维设置
    TRIVIAL_3D = _TRIVIAL_3D  # 枚举类型，表示一个简单的3维设置
    TRIVIAL_4D = _TRIVIAL_4D  # 枚举类型，表示一个简单的4维设置
    TRAINING = _TRAINING  # 枚举类型，表示一个训练设置
```