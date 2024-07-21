# `.\pytorch\torch\lib\libshm\libshm.h`

```
#pragma once
// 使用 #pragma once 来确保头文件只被编译一次，避免重复定义

#include <ATen/MapAllocator.h>
// 包含 ATen 库的 MapAllocator 头文件

#ifdef __cplusplus
// 如果是 C++ 编译环境，则编译以下内容

void libshm_init(const char* manager_exec_path);
// 函数声明：初始化共享内存管理器，传入管理器执行路径

// THManagedMapAllocatorInit 类的作用是在 at::RefcountedMapAllocator 之前运行构造函数的超类
class THManagedMapAllocatorInit {
 protected:
  THManagedMapAllocatorInit(const char* manager_handle, const char* filename);
  // 构造函数声明：传入管理器句柄和文件名，用于初始化

  std::string manager_handle_;
  // 管理器句柄的字符串成员变量
};

// THManagedMapAllocator 类继承自 THManagedMapAllocatorInit 类，同时也是 at::RefcountedMapAllocator 的子类
class THManagedMapAllocator : private THManagedMapAllocatorInit,
                              public at::RefcountedMapAllocator {
 public:
  THManagedMapAllocator(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size);
  // 构造函数声明：传入管理器句柄、文件名、标志和大小，用于初始化

  void close() override;
  // 覆盖父类的 close 方法，用于关闭资源

  ~THManagedMapAllocator() override {
    close();
  }
  // 析构函数：在对象销毁时调用 close 方法，释放资源

  static at::DataPtr makeDataPtr(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size);
  // 静态方法：创建一个数据指针，传入管理器句柄、文件名、标志和大小

  static THManagedMapAllocator* fromDataPtr(const at::DataPtr&);
  // 静态方法：根据数据指针创建 THManagedMapAllocator 对象

  const char* manager_handle() const {
    return manager_handle_.c_str();
  }
  // 成员方法：返回管理器句柄的 C 字符串形式
};

#endif
```