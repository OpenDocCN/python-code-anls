# `.\pytorch\aten\src\ATen\cuda\PeerToPeerAccess.h`

```
#include <c10/macros/Macros.h>
// 包含了 c10 库的宏定义和宏函数，用于 CUDA 相关操作

#include <cstdint>
// 包含了标准整数类型的定义，例如 int64_t

namespace at::cuda {
// 进入 at::cuda 命名空间

namespace detail {
// 进入 detail 命名空间，该命名空间定义了 CUDA 相关的详细实现细节

// 声明了一个函数 init_p2p_access_cache，用于初始化点对点（P2P）访问缓存
void init_p2p_access_cache(int64_t num_devices);
}

// 使用 TORCH_CUDA_CPP_API 宏定义了一个函数 get_p2p_access，用于获取设备之间的 P2P 访问能力
TORCH_CUDA_CPP_API bool get_p2p_access(int source_dev, int dest_dev);

}  // namespace at::cuda
// 结束 at::cuda 命名空间
```