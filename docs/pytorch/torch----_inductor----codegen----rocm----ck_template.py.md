# `.\pytorch\torch\_inductor\codegen\rocm\ck_template.py`

```
# 导入 torch 库
import torch
# 从 torch._inductor.codegen.rocm.rocm_template 中导入 ROCmTemplate 类
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate
# 从 torch._inductor.ir 中导入 IRNode 类
from torch._inductor.ir import IRNode
# 从 torch._inductor.utils 中导入 IndentedBuffer 类
from torch._inductor.utils import IndentedBuffer

# 定义 CKTemplate 类，继承自 ROCmTemplate 类
class CKTemplate(ROCmTemplate):
    """
    生成 CK 模板的基类，包含通用（非特定于 gemm 的）代码生成逻辑
    """

    # 定义字典 _TORCH_DTYPE_TO_CK，将 torch 的数据类型映射到 CK 的数据类型
    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",
        torch.float8_e5m2fnuz: "BF8",
    }

    # 重写父类的 header 方法，返回一个 IndentedBuffer 对象
    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // HIP headers

                #include <hip/hip_bfloat16.h>

                // CK headers

                #ifdef DEBUG_LOG
                #define DEBUG_LOG_TMP DEBUG_LOG
                #undef DEBUG_LOG
                #else
                #define DEBUG_LOG_TMP 0
                #endif
                #include "ck/ck.hpp"
                #undef DEBUG_LOG
                #define DEBUG_LOG DEBUG_LOG_TMP

                #include "ck/utility/data_type.hpp"
                #include "ck/library/utility/check_err.hpp"
                #include "ck/library/utility/device_memory.hpp"
                #include "ck/library/utility/fill.hpp"
                #include "ck/library/utility/host_tensor.hpp"
                #include "ck/library/utility/host_tensor_generator.hpp"
                #include "ck/library/utility/literals.hpp"
            """
        )
        return res

    # 重写父类的 globals 方法，返回一个 IndentedBuffer 对象
    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK globals

                template <ck::index_t... Is>
                using S = ck::Sequence<Is...>;

                using PassThrough = ck::tensor_operation::element_wise::PassThrough;

                // see "composable_kernel/include/ck/utility/data_type.hpp"
                using F8  = ck::f8_t;
                using BF8 = ck::bf8_t;
                using F16 = ck::half_t;
                using F32 = float;
                // using F64 = double;
                using BF16 = ck::bhalf_t;
                // using I32 = int32_t;
                // using I8 = int8_t;
                // using I4 = ck::int4_t;

                #if DEBUG_LOG
                static constexpr auto kDEBUG_LOG = 1;
                #else
                static constexpr auto kDEBUG_LOG = 0;
                #endif
            """
        )
        return res

    # 定义方法 torch_type_to_ck，将 torch 的数据类型映射为 CK 的数据类型
    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"
```