# `.\pytorch\test\test_ao_sparsity.py`

```
# 导入测试内核模块
from ao.sparsity.test_kernels import (  # noqa: F401  # noqa: F401
    TestQuantizedSparseKernels,  # 导入测试量化稀疏内核类
    TestQuantizedSparseLayers,   # 导入测试量化稀疏层类
)

# 导入参数化测试模块
from ao.sparsity.test_parametrization import TestFakeSparsity  # noqa: F401

# 导入调度器测试模块
from ao.sparsity.test_scheduler import (  # noqa: F401  # noqa: F401
    TestCubicScheduler,   # 导入测试立方调度器类
    TestScheduler,        # 导入测试调度器类
)

# 导入稀疏化器测试模块
from ao.sparsity.test_sparsifier import (  # noqa: F401  # noqa: F401  # noqa: F401
    TestBaseSparsifier,           # 导入测试基础稀疏化器类
    TestNearlyDiagonalSparsifier, # 导入测试近对角线稀疏化器类
    TestWeightNormSparsifier,     # 导入测试权重归一化稀疏化器类
)

# 导入结构化修剪测试模块
from ao.sparsity.test_structured_sparsifier import (  # noqa: F401  # noqa: F401  # noqa: F401
    TestBaseStructuredSparsifier,  # 导入测试基础结构化稀疏化器类
    TestFPGMPruner,                # 导入测试FPGM修剪器类
    TestSaliencyPruner,            # 导入测试显著性修剪器类
)

# 导入torch测试内部常用工具
from torch.testing._internal.common_utils import IS_ARM64, run_tests

# 如果不是ARM64架构，导入可组合性测试模块
if not IS_ARM64:
    from ao.sparsity.test_composability import (  # noqa: F401  # noqa: F401
        TestComposability,   # 导入测试可组合性类
        TestFxComposability, # 导入测试Fx可组合性类
    )

# 导入激活稀疏化器测试模块
from ao.sparsity.test_activation_sparsifier import (  # noqa: F401
    TestActivationSparsifier,  # 导入测试激活稀疏化器类
)

# 导入数据调度器测试模块
from ao.sparsity.test_data_scheduler import TestBaseDataScheduler  # noqa: F401

# 导入数据稀疏化器测试模块
from ao.sparsity.test_data_sparsifier import (  # noqa: F401  # noqa: F401  # noqa: F401
    TestBaseDataSparsifier,    # 导入测试基础数据稀疏化器类
    TestNormDataSparsifiers,   # 导入测试规范数据稀疏化器类
    TestQuantizationUtils,     # 导入测试量化工具类
)

# 导入稀疏化工具函数测试模块
from ao.sparsity.test_sparsity_utils import TestSparsityUtilFunctions  # noqa: F401

# 如果是主模块，运行测试
if __name__ == "__main__":
    run_tests()
```