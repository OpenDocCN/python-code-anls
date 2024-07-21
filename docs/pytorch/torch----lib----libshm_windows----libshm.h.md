# `.\pytorch\torch\lib\libshm_windows\libshm.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/MapAllocator.h>
// 包含 ATen 库的 MapAllocator 头文件

#ifdef __cplusplus
// 如果是 C++ 环境

#ifdef SHM_EXPORTS
#define SHM_API __declspec(dllexport)
// 如果 SHM_EXPORTS 被定义，定义 SHM_API 为导出符号
#else
#define SHM_API __declspec(dllimport)
// 否则定义 SHM_API 为导入符号
#endif

SHM_API void libshm_init(const char* manager_exec_path);
// 声明 SHM_API 修饰的 libshm_init 函数，接受一个 manager_exec_path 参数

class SHM_API THManagedMapAllocator : public at::RefcountedMapAllocator {
// 声明 SHM_API 修饰的 THManagedMapAllocator 类，继承自 at::RefcountedMapAllocator
 public:
  THManagedMapAllocator(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size)
      : at::RefcountedMapAllocator(filename, flags, size) {}
  // THManagedMapAllocator 类的构造函数，调用基类 at::RefcountedMapAllocator 的构造函数初始化

  static at::DataPtr makeDataPtr(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size);
  // 声明静态函数 makeDataPtr，返回类型为 at::DataPtr，接受 manager_handle、filename、flags、size 参数

  static THManagedMapAllocator* fromDataPtr(const at::DataPtr&);
  // 声明静态函数 fromDataPtr，返回类型为 THManagedMapAllocator*，接受 at::DataPtr 参数

  const char* manager_handle() const {
    return "no_manager";
  }
  // 实现 manager_handle 函数，返回字符串 "no_manager"
};

#endif
// 结束 C++ 环境的条件编译
```