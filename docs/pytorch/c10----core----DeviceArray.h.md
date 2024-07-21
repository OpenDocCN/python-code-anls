# `.\pytorch\c10\core\DeviceArray.h`

```
// 包含 C10 核心模块中的 Allocator 头文件
#include <c10/core/Allocator.h>
// 包含 C10 实用工具中的 Exception 头文件
#include <c10/util/Exception.h>
// 包含标准库中的头文件，用于定义各种大小和整数类型
#include <cstddef>
#include <cstdint>
// 包含标准库中的头文件，用于类型特性的支持
#include <type_traits>

// 定义 C10 命名空间
namespace c10 {

// 定义模板类 DeviceArray，用于管理设备上的数组
template <typename T>
class DeviceArray {
 public:
  // 构造函数，接受 Allocator 和数组大小作为参数
  DeviceArray(c10::Allocator& allocator, size_t size)
      // 初始化 data_ptr_，通过 Allocator 分配 size * sizeof(T) 字节的内存
      : data_ptr_(allocator.allocate(size * sizeof(T))) {
    // 静态断言，确保 T 是平凡类型（trivial type）
    static_assert(std::is_trivial<T>::value, "T must be a trivial type");
    // 内部 Torch 断言，验证分配的内存地址是否按 T 的对齐要求对齐
    TORCH_INTERNAL_ASSERT(
        0 == (reinterpret_cast<intptr_t>(data_ptr_.get()) % alignof(T)),
        "c10::DeviceArray: Allocated memory is not aligned for this data type");
  }

  // 返回指向设备数组数据的指针
  T* get() {
    return static_cast<T*>(data_ptr_.get());
  }

 private:
  // 数据指针，使用 c10::DataPtr 类型进行管理
  c10::DataPtr data_ptr_;
};

} // namespace c10
```