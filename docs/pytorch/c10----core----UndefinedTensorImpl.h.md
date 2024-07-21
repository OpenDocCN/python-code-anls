# `.\pytorch\c10\core\UndefinedTensorImpl.h`

```py
```cpp`
#pragma once
// 使用 #pragma once 确保头文件只被编译一次，避免重复包含

#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <cstdint>

namespace c10 {

struct C10_API UndefinedTensorImpl final : public TensorImpl {
  // 定义一个继承自 TensorImpl 的结构体 UndefinedTensorImpl

 public:
  // 定义静态函数 singleton，返回一个 TensorImpl 指针
  // 在 Windows 下为静态内联函数，其他系统下为 constexpr 静态内联函数
  #ifdef _WIN32
  static inline TensorImpl* singleton() {
  #else
  static constexpr inline TensorImpl* singleton() {
  #endif
    return &_singleton;
    // 返回 _singleton 的地址
  }

  // 如果定义了 DEBUG 宏，重载 has_storage 函数
  #ifdef DEBUG
  bool has_storage() const override;
  #endif

  // 重载 set_storage_offset 函数，设置存储偏移量
  void set_storage_offset(int64_t offset) override;

 protected:
  // 重载 is_contiguous_custom 函数，检查是否按指定格式连续
  bool is_contiguous_custom(MemoryFormat format) const override;

  // 重载 strides_custom 函数，返回自定义的步长数组引用
  IntArrayRef strides_custom() const override;

  // 重载 sym_strides_custom 函数，返回自定义的对称步长数组引用
  SymIntArrayRef sym_strides_custom() const override;

 private:
  // 私有构造函数 UndefinedTensorImpl
  UndefinedTensorImpl();

  // 定义静态成员 _singleton，类型为 UndefinedTensorImpl
  static UndefinedTensorImpl _singleton;

  // 重载 tensorimpl_type_name 函数，返回张量实现类型名称的字符串
  const char* tensorimpl_type_name() const override;
};

} // namespace c10
```