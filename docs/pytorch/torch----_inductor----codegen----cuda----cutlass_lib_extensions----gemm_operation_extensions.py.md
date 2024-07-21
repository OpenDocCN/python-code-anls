# `.\pytorch\torch\_inductor\codegen\cuda\cutlass_lib_extensions\gemm_operation_extensions.py`

```
# 引入 `try_import_cutlass` 函数从 `..cutlass_utils` 模块中
# 如果导入成功，则继续执行以下代码
from ..cutlass_utils import try_import_cutlass

# 检查是否成功导入了 Cutlass 库
if try_import_cutlass():
    # 导入 enum 模块，用于支持枚举类型
    import enum

    # 从 cutlass_library.library 中导入所有内容，但忽略 F401 和 F403 错误
    from cutlass_library.library import *  # noqa: F401, F403
    # 从 cutlass_library.gemm_operation 中导入所有内容，但忽略 F401 和 F403 错误
    from cutlass_library.gemm_operation import *  # noqa: F401, F403

    # 以下代码段是从原始代码中复制/修改的
    # 来自 https://github.com/NVIDIA/cutlass/blob/8783c41851cd3582490e04e69e0cd756a8c1db7f/tools/library/scripts/gemm_operation.py#L658
    # 以支持类似于 https://github.com/NVIDIA/cutlass/blob/8783c41851cd3582490e04e69e0cd756a8c1db7f/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu#L315C69-L315C69 的 EVT
    # noqa: B950 是为了忽略 Flake8 B950 错误
```