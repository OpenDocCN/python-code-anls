# `.\pytorch\aten\src\ATen\core\TorchDispatchUtils.cpp`

```
#include <ATen/core/TorchDispatchUtils.h>

// 命名空间 at::impl 中的函数定义开始

namespace at::impl {

// 检查给定的张量是否具有 Python 或 PythonTLSSnapshot 调度键
bool tensor_has_dispatch(const at::Tensor& t) {
  // 创建包含 Python 和 PythonTLSSnapshot 调度键的集合
  DispatchKeySet key_set({DispatchKey::Python, DispatchKey::PythonTLSSnapshot});
  // 检查张量 t 的调度键集合是否包含任何指定的调度键
  return t.key_set().has_any(key_set);
}

// 检查张量列表中是否存在具有 Python 或 PythonTLSSnapshot 调度键的张量
bool tensorlist_has_dispatch(at::ITensorListRef li) {
  // 遍历张量列表 li 中的每个张量 t
  for (const auto& t : li) {
    // 如果当前张量 t 具有指定的调度键，则返回 true
    if (tensor_has_dispatch(t)) {
      return true;
    }
  }
  // 如果列表中所有张量均不具有指定的调度键，则返回 false
  return false;
}

// 检查张量可选列表中是否存在具有 Python 或 PythonTLSSnapshot 调度键的张量
bool tensorlist_has_dispatch(const c10::List<std::optional<at::Tensor>>& li) {
  // 遍历可选张量列表 li 中的每个元素 i
  for (auto i : c10::irange(li.size())) {
    // 获取第 i 个元素对应的张量 t
    auto t = li.get(i);
    // 如果 t 存在且具有指定的调度键，则返回 true
    if (t && tensor_has_dispatch(*t)) {
      return true;
    }
  }
  // 如果列表中所有可选张量均不具有指定的调度键，则返回 false
  return false;
}

} // namespace at::impl
// 命名空间 at::impl 中的函数定义结束
```