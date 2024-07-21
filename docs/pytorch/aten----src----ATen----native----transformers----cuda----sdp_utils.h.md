# `.\pytorch\aten\src\ATen\native\transformers\cuda\sdp_utils.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/Context.h>
// 包含 ATen 库的上下文头文件

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义头文件

#include <ATen/native/transformers/sdp_utils_cpp.h>
// 包含 ATen 库的 native 模块中 transformers 相关的 SDP 实用工具头文件

#include <c10/macros/Export.h>
// 包含 c10 库的导出宏定义头文件

namespace sdp {
// 命名空间 sdp，用于封装以下功能

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug);
// 检查是否有长度为 1 的嵌套张量，参数为 sdp_params 和调试标志

SDPBackend select_sdp_backend(sdp_params const& kernel_params);
// 选择 SDP 后端，参数为 sdp_params 结构体

C10_EXPORT bool can_use_flash_attention(sdp_params const& params, bool debug);
// 导出函数，判断是否可以使用 Flash 注意力机制，参数为 sdp_params 和调试标志

C10_EXPORT bool can_use_mem_efficient_attention(sdp_params const& params, bool debug);
// 导出函数，判断是否可以使用内存高效的注意力机制，参数为 sdp_params 和调试标志

C10_EXPORT bool can_use_cudnn_attention(sdp_params const& params, bool debug);
// 导出函数，判断是否可以使用 CuDNN 的注意力机制，参数为 sdp_params 和调试标志

} // namespace sdp
// 结束 sdp 命名空间
```