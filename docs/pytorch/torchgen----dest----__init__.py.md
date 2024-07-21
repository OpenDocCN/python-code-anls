# `.\pytorch\torchgen\dest\__init__.py`

```
# 导入生成非本地惰性 IR 节点的函数和类
from torchgen.dest.lazy_ir import (
    generate_non_native_lazy_ir_nodes as generate_non_native_lazy_ir_nodes,
    GenLazyIR as GenLazyIR,
    GenLazyNativeFuncDefinition as GenLazyNativeFuncDefinition,
    GenLazyShapeInferenceDefinition as GenLazyShapeInferenceDefinition,
)

# 导入计算本地函数声明的函数
from torchgen.dest.native_functions import (
    compute_native_function_declaration as compute_native_function_declaration,
)

# 导入生成注册调度键的头文件和辅助函数，以及注册调度键的类
from torchgen.dest.register_dispatch_key import (
    gen_registration_headers as gen_registration_headers,
    gen_registration_helpers as gen_registration_helpers,
    RegisterDispatchKey as RegisterDispatchKey,
)

# 导入计算 CPU 并行通用函数的函数，以及计算 CPU 并行通用函数内核的函数
from torchgen.dest.ufunc import (
    compute_ufunc_cpu as compute_ufunc_cpu,
    compute_ufunc_cpu_kernel as compute_ufunc_cpu_kernel,
    compute_ufunc_cuda as compute_ufunc_cuda,
)
```