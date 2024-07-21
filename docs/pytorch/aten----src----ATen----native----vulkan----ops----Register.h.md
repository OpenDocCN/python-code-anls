# `.\pytorch\aten\src\ATen\native\vulkan\ops\Register.h`

```py
#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则进入条件编译

namespace at {
namespace native {
namespace vulkan {
namespace ops {
// 进入 ATen（PyTorch C++ 前端库）的 Vulkan 后端操作命名空间

// 注册 Vulkan 后端的二维卷积操作上下文
int register_vulkan_conv2d_packed_context();

// 注册 Vulkan 后端的一维卷积操作上下文
int register_vulkan_conv1d_packed_context();

// 注册 Vulkan 后端的线性操作上下文
int register_vulkan_linear_packed_context();

// 注册 Vulkan 后端的层归一化操作上下文
int register_vulkan_layernorm_packed_context();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
// 结束 ATen 的 Vulkan 后端操作命名空间

#endif /* USE_VULKAN_API */
// 结束 USE_VULKAN_API 宏的条件编译
```