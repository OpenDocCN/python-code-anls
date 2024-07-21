# `.\pytorch\aten\src\ATen\CachedTensorUtils.cpp`

```
// 引入 ATen 库的头文件
#include <ATen/ATen.h>
// 引入 ATen 库的 CachedTensorUtils 头文件
#include <ATen/CachedTensorUtils.h>

// 引入 flat_hash_map 类型，位于 c10/util/flat_hash_map.h 中
#include <c10/util/flat_hash_map.h>

// 定义命名空间 at::caching
namespace at::caching {

// 使用 weakref_type 别名表示 c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;

// 声明全局变量 cached_tensorimpls_enabled 并初始化为 false
bool cached_tensorimpls_enabled = false;

// flat_hash_map 类型，将 TensorImpl* 映射到 weakref_type
ska::flat_hash_map<TensorImpl*, weakref_type> cached_tensorimpls;

// 互斥量，用于保护 cached_tensorimpls 操作的线程安全
std::mutex cached_tensorimpl_mutex;

// 检查给定的 Tensor 是否在缓存中
bool is_cached_tensor(const at::Tensor& t) {
  // 如果 cached_tensorimpls_enabled 为 false，则直接返回 false
  if (!cached_tensorimpls_enabled) {
    return false;
  }
  // 使用互斥量保护对 cached_tensorimpls 的访问
  const std::lock_guard<std::mutex> lock(cached_tensorimpl_mutex);
  // 检查 cached_tensorimpls 中是否存在给定 Tensor 的 TensorImpl*
  return cached_tensorimpls.count(t.unsafeGetTensorImpl());
}

// 将给定的 Tensor 添加到缓存中
void add_cached_tensor(const at::Tensor& t) {
  // 断言 cached_tensorimpls_enabled 为 true
  TORCH_INTERNAL_ASSERT(cached_tensorimpls_enabled);
  // 使用互斥量保护对 cached_tensorimpls 的访问
  const std::lock_guard<std::mutex> lock(cached_tensorimpl_mutex);
  // 将给定 Tensor 的 TensorImpl* 与 weakref_type 添加到 cached_tensorimpls 中
  cached_tensorimpls.emplace(t.unsafeGetTensorImpl(), weakref_type(t.getIntrusivePtr()));
}

// 从缓存中移除给定的 Tensor
void remove_cached_tensor(const at::Tensor& t) {
  // 断言 cached_tensorimpls_enabled 为 true
  TORCH_INTERNAL_ASSERT(cached_tensorimpls_enabled);
  // 使用互斥量保护对 cached_tensorimpls 的访问
  const std::lock_guard<std::mutex> lock(cached_tensorimpl_mutex);
  // 移除 cached_tensorimpls 中给定 Tensor 的 TensorImpl*
  cached_tensorimpls.erase(t.unsafeGetTensorImpl());
}

// 设置 cached_tensorimpls_enabled 的状态（启用或禁用缓存）
void set_cached_tensors_enabled(bool enabled) {
  cached_tensorimpls_enabled = enabled;
}

// 返回给定 Tensor 的调整后的引用计数
size_t adjusted_use_count(const at::Tensor& t) {
  // 返回 Tensor 的 use_count 减去是否在缓存中的条件（如果在则减去1）
  return t.use_count() - (is_cached_tensor(t) ? 1 : 0);
}

} // namespace at::caching
```