# `.\pytorch\torch\ao\pruning\__init__.py`

```py
# 导入动态稀疏量化映射函数
from ._mappings import get_dynamic_sparse_quantized_mapping
# 导入静态稀疏量化映射函数
from ._mappings import get_static_sparse_quantized_mapping

# 导入基础稀疏化器
from .sparsifier.base_sparsifier import BaseSparsifier
# 导入权重规范化稀疏化器
from .sparsifier.weight_norm_sparsifier import WeightNormSparsifier
# 导入近对角线稀疏化器
from .sparsifier.nearly_diagonal_sparsifier import NearlyDiagonalSparsifier

# 导入基础调度器
from .scheduler.base_scheduler import BaseScheduler
# 导入Lambda调度器
from .scheduler.lambda_scheduler import LambdaSL
# 导入立方调度器
from .scheduler.cubic_scheduler import CubicSL

# 导入稀疏化工具函数 - 伪稀疏度
from .sparsifier.utils import FakeSparsity
# 导入稀疏化工具函数 - 模块到全限定名
from .sparsifier.utils import module_to_fqn
# 导入稀疏化工具函数 - 全限定名到模块
from .sparsifier.utils import fqn_to_module
# 导入稀疏化工具函数 - 从张量全限定名获取参数信息
from .sparsifier.utils import get_arg_info_from_tensor_fqn
```