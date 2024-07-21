# `.\pytorch\torch\lib\libshm_windows\core.cpp`

```
// 包含必要的头文件：cstring（C风格字符串操作）、string（C++字符串操作）、unordered_map（无序映射容器）
#include <cstring>
#include <string>
#include <unordered_map>

// 包含 libshm 库的头文件
#include <libshm_windows/libshm.h>

// 初始化 libshm，但函数体为空，即不执行任何操作
void libshm_init(const char* manager_exec_path) {}

// 静态函数，用于释放 THManagedMapAllocator 实例的内存
static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

// 创建并返回一个 DataPtr 对象，该对象包含了 THManagedMapAllocator 实例的指针、数据指针和释放函数
at::DataPtr THManagedMapAllocator::makeDataPtr(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size) {
  // 创建一个 THManagedMapAllocator 实例，传入参数 manager_handle、filename、flags 和 size
  auto* context =
      new THManagedMapAllocator(manager_handle, filename, flags, size);
  // 返回一个 DataPtr 对象，包含了 THManagedMapAllocator 实例的数据指针、实例指针及其释放函数，指定在 CPU 上进行操作
  return {context->data(), context, &deleteTHManagedMapAllocator, at::kCPU};
}

// 从 DataPtr 对象中获取 THManagedMapAllocator 实例的指针，并进行类型转换
THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(
    const at::DataPtr& dptr) {
  // 将 DataPtr 对象转换为 THManagedMapAllocator*，并指定其释放函数为 deleteTHManagedMapAllocator
  return dptr.cast_context<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}
```