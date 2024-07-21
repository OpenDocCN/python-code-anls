# `.\pytorch\aten\src\ATen\native\vulkan\api\ShaderRegistry.h`

```
#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Shader.h>

#include <string>
#include <unordered_map>

// 定义宏 VK_KERNEL，用于获取 shader_name 对应的 ShaderInfo
#define VK_KERNEL(shader_name) \
  ::at::native::vulkan::api::shader_registry().get_shader_info(#shader_name)

// 定义宏 VK_KERNEL_FROM_STR，用于获取 shader_name_str 对应的 ShaderInfo
#define VK_KERNEL_FROM_STR(shader_name_str) \
  ::at::native::vulkan::api::shader_registry().get_shader_info(shader_name_str)

namespace at {
namespace native {
namespace vulkan {
namespace api {

// 枚举类型 DispatchKey，表示 Vulkan 的调度键
enum class DispatchKey : int8_t {
  CATCHALL,  // 默认调度键
  ADRENO,    // 针对 Adreno GPU 的调度键
  MALI,      // 针对 Mali GPU 的调度键
  OVERRIDE,  // 覆盖默认调度键的自定义调度键
};

// ShaderRegistry 类，管理 Vulkan Shader 的注册和查询
class ShaderRegistry final {
  // 使用 unordered_map 存储 ShaderInfo，键为 shader 名称
  using ShaderListing = std::unordered_map<std::string, ShaderInfo>;
  // 使用 unordered_map 存储 DispatchKey 和对应的 shader 名称
  using Dispatcher = std::unordered_map<DispatchKey, std::string>;
  // 使用 unordered_map 存储 op 名称和其对应的 Dispatcher
  using Registry = std::unordered_map<std::string, Dispatcher>;

  ShaderListing listings_;  // 存储所有已注册的 ShaderInfo
  Dispatcher dispatcher_;   // 存储所有注册的调度信息
  Registry registry_;       // 存储所有注册的 op 和其调度信息

 public:
  /*
   * 检查是否在注册表中有以给定名称注册的着色器
   */
  bool has_shader(const std::string& shader_name);

  /*
   * 检查注册表中是否有以给定名称注册的调度信息
   */
  bool has_dispatch(const std::string& op_name);

  /*
   * 将 ShaderInfo 注册到指定的 shader 名称
   */
  void register_shader(ShaderInfo&& shader_info);

  /*
   * 注册一个调度条目到给定的 op 名称
   */
  void register_op_dispatch(
      const std::string& op_name,
      const DispatchKey key,
      const std::string& shader_name);

  /*
   * 给定一个 shader 名称，返回包含 SPIRV 二进制的 ShaderInfo
   */
  const ShaderInfo& get_shader_info(const std::string& shader_name);
};

// ShaderRegisterInit 类，用于初始化 ShaderRegistry
class ShaderRegisterInit final {
  using InitFn = void();

 public:
  // 构造函数，调用初始化函数指针以初始化注册表
  ShaderRegisterInit(InitFn* init_fn) {
    init_fn();
  };
};

// 全局函数，返回全局的 ShaderRegistry 对象
ShaderRegistry& shader_registry();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```