# `.\pytorch\aten\src\ATen\core\type_factory.h`

```
#pragma once
// 预处理指令：确保此头文件只被包含一次

#include <type_traits>
// 引入类型特性库，用于模板元编程中的类型特性判断

#include <unordered_map>
// 引入无序映射容器，用于存储键值对，实现快速查找

#include <ATen/core/dynamic_type.h>
// 引入 ATen 库中的动态类型定义

#include <ATen/core/jit_type_base.h>
// 引入 ATen 库中的 JIT 类型基础定义

#include <c10/macros/Macros.h>
// 引入 c10 库中的宏定义

namespace c10 {

template <typename T>
struct TORCH_API TypeFactoryBase {};
// 通用模板类 TypeFactoryBase，用于派生具体类型的工厂类

template <>
struct TORCH_API TypeFactoryBase<c10::DynamicType> {
  // 针对 c10::DynamicType 特化的 TypeFactoryBase

  template <typename T, typename... Args>
  static c10::DynamicTypePtr create(TypePtr ty, Args&&... args) {
    // 创建动态类型对象，参数包括类型指针和可变参数列表
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(), // 使用类型 T 的标签值
        c10::DynamicType::Arguments(c10::ArrayRef<c10::TypePtr>(
            {std::move(ty), std::forward<Args>(args)...}))); // 构造参数列表
  }

  template <typename T>
  static c10::DynamicTypePtr create(const std::vector<c10::TypePtr>& types) {
    // 创建动态类型对象，参数是类型指针的向量
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(), // 使用类型 T 的标签值
        c10::DynamicType::Arguments(types)); // 构造参数列表
  }

  static c10::DynamicTypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<c10::TypePtr>& types) {
    // 创建命名元组动态类型对象
    return std::make_shared<c10::DynamicType>(
        c10::DynamicType::Tag::Tuple, // 使用元组标签
        name, // 元组类型的名称
        c10::DynamicType::Arguments(fields, types)); // 元组的字段和类型参数列表
  }

  template <typename T>
  C10_ERASE static c10::DynamicTypePtr createNamed(const std::string& name) {
    // 创建命名动态类型对象
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(), // 使用类型 T 的标签值
        name, // 动态类型的名称
        c10::DynamicType::Arguments{}); // 空的参数列表
  }

  template <typename T>
  C10_ERASE static c10::DynamicTypePtr get() {
    // 获取类型 T 的基础动态类型指针
    return DynamicTypeTrait<T>::getBaseType();
  }

  static const std::unordered_map<std::string, c10::TypePtr>& basePythonTypes();
  // 返回基础 Python 类型映射的引用
};

using DynamicTypeFactory = TypeFactoryBase<c10::DynamicType>;
// 使用 c10::DynamicType 实例化的 TypeFactoryBase 作为 DynamicTypeFactory

// 助手函数：内联构造动态类型的辅助函数
template <
    typename T,
    std::enable_if_t<DynamicTypeTrait<T>::isBaseType, int> = 0>
C10_ERASE DynamicTypePtr dynT() {
  // 如果 T 是基础类型，则返回对应的动态类型指针
  return DynamicTypeFactory::get<T>();
}

template <
    typename T,
    typename... Args,
    std::enable_if_t<!DynamicTypeTrait<T>::isBaseType, int> = 0>
C10_ERASE DynamicTypePtr dynT(Args&&... args) {
  // 如果 T 不是基础类型，则创建并返回对应的动态类型指针
  return DynamicTypeFactory::create<T>(std::forward<Args>(args)...);
}

template <>
struct TORCH_API TypeFactoryBase<c10::Type> {
  // 针对 c10::Type 特化的 TypeFactoryBase

  template <typename T, typename... Args>
  static c10::TypePtr create(TypePtr ty, Args&&... args) {
    // 创建类型对象，参数包括类型指针和可变参数列表
    return T::create(std::move(ty), std::forward<Args>(args)...);
  }

  template <typename T>
  static c10::TypePtr create(std::vector<c10::TypePtr> types) {
    // 创建类型对象，参数是类型指针的向量
    return T::create(std::move(types));
  }

  static c10::TypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<c10::TypePtr>& types);

  template <typename T>
  C10_ERASE static c10::TypePtr createNamed(const std::string& name) {
    // 创建命名类型对象
    // （这里的实现在截断处，下文未提供完整内容）
    // 返回通过静态成员函数 create(name) 创建的类型 T 的实例
    return T::create(name);
  }
  // 返回基本 Python 类型的无序映射表
  static const std::unordered_map<std::string, c10::TypePtr>& basePythonTypes();
  // 模板函数，返回类型 T 的 TypePtr
  template <typename T>
  C10_ERASE static c10::TypePtr get() {
    return T::get();
  }
};

// 使用别名 DefaultTypeFactory 表示 TypeFactoryBase<c10::Type> 类型
using DefaultTypeFactory = TypeFactoryBase<c10::Type>;

// 定义一个 PlatformType 别名，根据编译时的宏 C10_MOBILE 决定具体类型
using PlatformType =
#ifdef C10_MOBILE
    c10::DynamicType
#else
    c10::Type
#endif
    ;

// 使用 PlatformType 作为模板参数，定义 TypeFactory 别名
using TypeFactory = TypeFactoryBase<PlatformType>;

// 命名空间结束注释，命名空间 c10
} // namespace c10
```