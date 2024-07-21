# `.\pytorch\c10\core\Storage.cpp`

```py
// 包含 C10 库中的引用计数删除器和存储类头文件
#include <c10/core/RefcountedDeleter.h>
#include <c10/core/Storage.h>

// 进入 c10 命名空间
namespace c10 {

// 检查两个存储是否共享相同的存储空间
bool isSharedStorageAlias(const Storage& storage0, const Storage& storage1) {
  // 期望的删除器函数指针是 refcounted_deleter
  c10::DeleterFnPtr deleter_expected = &c10::refcounted_deleter;
  // 获取存储0的实际删除器函数指针
  c10::DeleterFnPtr deleter0 = storage0.data_ptr().get_deleter();
  // 获取存储1的实际删除器函数指针
  c10::DeleterFnPtr deleter1 = storage1.data_ptr().get_deleter();

  // 如果任意一个存储的删除器不是期望的 refcounted_deleter，则返回 false
  if ((deleter0 != deleter_expected) || (deleter1 != deleter_expected)) {
    return false;
  }

  // 返回两个存储是否共享相同的上下文
  return storage0.data_ptr().get_context() == storage1.data_ptr().get_context();
}

} // namespace c10
```