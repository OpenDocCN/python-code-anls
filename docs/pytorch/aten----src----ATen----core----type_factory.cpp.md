# `.\pytorch\aten\src\ATen\core\type_factory.cpp`

```py
// 包含 ATen 库中的类型工厂相关头文件
#include <ATen/core/type_factory.h>

// 包含 ATen 库中的 JIT 类型头文件
#include <ATen/core/jit_type.h>

// 定义 c10 命名空间，用于存放 ATen 的核心功能
namespace c10 {

// Dtype 约束在编译中不是强制约束。因此，我们将所有具有不同 dtype 的张量子类映射到同一个基础 Tensor 上。
// 但是，每当用户使用任何张量子类（如 LongTensor）时，我们会警告可能的 dtype 更改。
//
// 技术上，“number” 不是一个 Python 类型，但在解析使用隐式转换到 Scalar 的序列化方法时需要它。
#define FORALL_BASE_PYTHON_TYPES(_) \
  _(Tensor, TensorType)             \
  _(LongTensor, TensorType)         \
  _(DoubleTensor, TensorType)       \
  _(FloatTensor, TensorType)        \
  _(IntTensor, TensorType)          \
  _(ShortTensor, TensorType)        \
  _(HalfTensor, TensorType)         \
  _(CharTensor, TensorType)         \
  _(ByteTensor, TensorType)         \
  _(BoolTensor, TensorType)         \
  _(int, IntType)                   \
  _(float, FloatType)               \
  _(bool, BoolType)                 \
  _(complex, ComplexType)           \
  _(str, StringType)                \
  _(Device, DeviceObjType)          \
  _(Generator, GeneratorType)       \
  _(Stream, StreamObjType)          \
  _(number, NumberType)             \
  _(None, NoneType)                 \
  _(NoneType, NoneType)             \
  _(Any, AnyType)                   \
  _(Capsule, CapsuleType)           \
  _(list, AnyListType)              \
  _(tuple, AnyTupleType)

// 返回一个静态的未排序映射，将字符串类型名称映射到相应的 ATen 类型指针
const std::unordered_map<std::string, c10::TypePtr>& DynamicTypeFactory::
    basePythonTypes() {
  static const std::unordered_map<std::string, c10::TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) \
  {#NAME, c10::DynamicTypeTrait<c10::TYPE>::getBaseType()},
      FORALL_BASE_PYTHON_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}

// 返回一个静态的未排序映射，将字符串类型名称映射到相应的 ATen 类型指针
const std::unordered_map<std::string, c10::TypePtr>& DefaultTypeFactory::
    basePythonTypes() {
  static const std::unordered_map<std::string, c10::TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) {#NAME, c10::TYPE::get()},
      FORALL_BASE_PYTHON_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}

// 创建一个具名元组类型，指定名称、字段和类型
c10::TypePtr DefaultTypeFactory::createNamedTuple(
    const std::string& name,
    const std::vector<c10::string_view>& fields,
    const std::vector<c10::TypePtr>& types) {
  return c10::TupleType::createNamed(name, fields, types);
}

} // namespace c10
```