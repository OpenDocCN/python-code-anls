# `.\pytorch\torch\csrc\profiler\orchestration\vulkan.cpp`

```py
#include <torch/csrc/profiler/orchestration/vulkan.h>

#include <utility>

namespace torch {
namespace profiler {
namespace impl {
namespace vulkan {
namespace {

// 声明静态变量，用于保存获取着色器名称和持续时间的函数指针
GetShaderNameAndDurationNsFn get_shader_name_and_duration_ns_fn;

} // namespace

// 注册获取着色器名称和持续时间的函数
void registerGetShaderNameAndDurationNs(
    GetShaderNameAndDurationNsFn get_shader_name_and_duration_ns) {
  // 将传入的函数指针移动给静态变量
  get_shader_name_and_duration_ns_fn =
      std::move(get_shader_name_and_duration_ns);
}

// 取消注册获取着色器名称和持续时间的函数
void deregisterGetShaderNameAndDurationNs() {
  // 将静态变量置空
  get_shader_name_and_duration_ns_fn = nullptr;
}

// 获取着色器名称和持续时间的函数
std::tuple<std::string, uint64_t> getShaderNameAndDurationNs(
    const vulkan_id_t& vulkan_id) {
  /*
    当前这里不需要担心与 deregisterGetShaderNameAndDurationNs 的竞争条件，
    因为 deregisterGetShaderNameAndDurationNs 只会在 QueryPool 的析构函数内部调用，
    而该函数只会在 getShaderNameAndDurationNs 调用完成后才会被调用
  */
  // 检查函数指针是否为 null
  TORCH_CHECK(
      get_shader_name_and_duration_ns_fn != nullptr,
      "Attempting to get shader duration in ",
      "torch::profiler::impl::vulkan::getShaderNameAndDurationNs, but "
      "get_shader_duration_fn is unregistered. Use "
      "torch::profiler::impl::vulkan::registerGetShaderNameAndDurationNs to register "
      "it first");
  // 调用注册的函数指针获取着色器名称和持续时间
  return get_shader_name_and_duration_ns_fn(vulkan_id.value_of());
}

} // namespace vulkan
} // namespace impl
} // namespace profiler
} // namespace torch
```