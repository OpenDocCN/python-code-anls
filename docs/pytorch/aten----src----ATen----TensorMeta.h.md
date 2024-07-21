# `.\pytorch\aten\src\ATen\TensorMeta.h`

```
#pragma once
// 预处理指令，确保本文件仅被编译一次

#include <ATen/DimVector.h>
// 包含 DimVector 类的头文件

#include <ATen/core/Dimname.h>
// 包含 Dimname 类的头文件

#include <c10/core/TensorOptions.h>
// 包含 TensorOptions 类的头文件

#include <c10/util/strides.h>
// 包含 strides.h 头文件，提供有关张量步幅的工具函数

namespace at {
// 命名空间 at 的开始

class Tensor;
// 声明 Tensor 类

namespace impl {
// 命名空间 impl 的开始，用于实现细节

// 用于定义元函数原型的宏。有两个版本，一个接受一个参数（操作符名称），另一个是 FUNC2 变体，接受两个参数（操作符名称和重载名称）。
//
// 示例用法：
//
//    TORCH_META_FUNC2(add, Tensor) (
//      const Tensor& self, const Tensor& other
//    ) {
//      ... 计算大小和选项 ...
//      设置输出(sizes, options);
//    }
//
#define TORCH_META_FUNC(name) void structured_##name::meta
// 定义元函数原型的宏，接受一个参数

#define TORCH_META_FUNC2(name, overload) \
  void structured_##name##_##overload::meta
// 定义元函数原型的宏，接受两个参数

// 这些是包含 precompute_out 结构作为返回值的 TORCH_META_FUNC(2) 的版本。当本地函数定义了预计算值，并且相应的实现应返回该结构的实例时使用。
#define TORCH_PRECOMPUTE_META_FUNC(name) \
  structured_##name::meta_return_ty structured_##name::meta
// 定义具有预计算结构的元函数原型的宏，接受一个参数

#define TORCH_PRECOMPUTE_META_FUNC2(name, overload) \
  structured_##name##_##overload::meta_return_ty    \
      structured_##name##_##overload::meta
// 定义具有预计算结构的元函数原型的宏，接受两个参数

// 用于在元函数中创建预计算结构的宏。
#define TORCH_PRECOMPUTE_STRUCT(name) structured_##name::precompute_out<>
// 定义在元函数中创建预计算结构的宏，接受一个参数

#define TORCH_PRECOMPUTE_STRUCT2(name, overload) \
  structured_##name##_##overload::precompute_out<>
// 定义在元函数中创建预计算结构的宏，接受两个参数

// 用于定义实现原型的宏。该宏只接受一个参数，即你要实现的调度键条目的名称。
//
// 示例用法：
//
//    TORCH_IMPL_FUNC(add_cpu) (
//      Tensor& result, const Tensor& self, const Tensor& other
//    ) {
//      ... 执行实际的实现 ...
//    }
//
#define TORCH_IMPL_FUNC(name) void structured_##name::impl
// 定义实现原型的宏，接受一个参数

// 所有结构化内核类的基类。set_output 虚方法根据操作符是函数式的、输出/内部的不同而变化，也可以为 CPU/CUDA 等专门化（尽管目前并不是这样）。
//
// 这个接口的一个显著子类是 TensorIteratorBase。
//
struct TORCH_API MetaBase {
  // 默认构造函数
  MetaBase() = default;
  // 拷贝构造函数
  MetaBase(const MetaBase&) = default;
  // 拷贝赋值运算符重载
  MetaBase& operator=(const MetaBase&) = default;
  // 移动构造函数
  MetaBase(MetaBase&&) noexcept = default;
  // 移动赋值运算符重载
  MetaBase& operator=(MetaBase&&) noexcept = default;
  // 纯虚函数，子类必须实现，返回指定索引处的输出 Tensor 引用
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;

  // Note: [set_output_*]
  // See: https://github.com/pytorch/pytorch/issues/69813
  // 当在结构化内核的 META 函数中定义输出属性时（通常使用 `set_output`），请使用以下三种变体之一。
  // 要决定使用哪个变体，请检查以下决策树：
  //
  // - 您要实现的内核是否支持具有任意步幅的输出张量？
  //     |
  //     -- 是：`set_output_raw_strided`
  //     |
  //     -- 否：输出张量步幅是否应连续？
  //         |
  //         -- 是：`set_output_contiguous`
  //         |
  //         -- 否：`set_output_strided`
  //
  // 当内核需要特定步幅的输出时，请使用此函数。如果 `strides` 与给定的输出步幅不匹配，则会创建代理输出并传递给 IMPL 函数。
  virtual void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // 当内核知道如何处理任意步幅的输出时，请使用此函数。此函数与旧的 `set_output` 具有相同的行为：仅在给定输出被调整大小时才会重新调整步幅。
  virtual void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides_hint,
      TensorOptions options,
      DimnameList names = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // 当内核需要连续步幅时，请使用此函数。这是 `set_output_strided` 的别名，但步幅是连续的。
  void set_output_contiguous(
      int64_t output_idx,
      IntArrayRef sizes,
      TensorOptions options,
      DimnameList names = {}) {
    auto strides = c10::contiguous_strides(sizes);
    set_output_strided(output_idx, sizes, strides, options, names);
  }

  // 如果没有预置输出，则返回对未定义张量的引用
  const Tensor& maybe_get_output() {
    return maybe_get_output(0);
  }
  // 默认析构函数
  virtual ~MetaBase() = default;
};
```