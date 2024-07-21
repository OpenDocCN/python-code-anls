# `.\pytorch\aten\src\ATen\templates\TensorMethods.cpp`

```
// 包含 C++ 的头文件，用于定义标量和张量的基本操作
#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

// 包含 c10 库的字符串视图工具
#include <c10/util/string_view.h>

// ATen 命名空间，包含了张量操作的实现
namespace at {

// 匿名命名空间，用于定义内部使用的辅助函数和常量
namespace {

// 检查张量的类型是否与请求的类型相同
void check_type(const TensorBase& tensor, ScalarType type, c10::string_view type_name) {
  // 使用 TORCH_CHECK 宏来验证条件，若不符合则抛出异常
  TORCH_CHECK(
      tensor.scalar_type() == type
      || (isQIntType(tensor.scalar_type())
          && toUnderlying(tensor.scalar_type()) == type),
      "expected scalar type ", type_name, " but found ", tensor.scalar_type());
}

} // namespace

// 定义一个宏，用于实现不同数据类型的类型转换操作
#define DEFINE_CAST(T, name)                                         \
   template <>                                                       \
   // 实现 const_data_ptr() 方法模板特化，返回指向常量数据的指针                                              
   TORCH_API const T* TensorBase::const_data_ptr() const {           \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
   }                                                                 \
                                                                     \
   template <>                                                       \
   // 实现 const_data_ptr() 方法模板特化，返回指向常量数据的指针（带 const T 类型模板参数）                                             
   TORCH_API const T* TensorBase::const_data_ptr<const T>() const {  \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->data_ptr_impl<std::remove_const_t<T>>(); \
   }                                                                 \
                                                                     \
   template <>                                                       \
   // 实现 mutable_data_ptr() 方法模板特化，返回指向可变数据的指针                                              
   TORCH_API T* TensorBase::mutable_data_ptr() const {               \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->mutable_data_ptr_impl<T>(); \
   }                                                                 \
                                                                     \
   template <>                                                       \
   // 实现 data_ptr() 方法模板特化，返回指向数据的指针                                              
   TORCH_API T* TensorBase::data_ptr() const {                       \
     return mutable_data_ptr<T>();                                   \
   }                                                                 \

// 使用宏扩展定义上面每种标量类型的类型转换操作
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
AT_FORALL_QINT_TYPES(DEFINE_CAST)
DEFINE_CAST(uint16_t, UInt16)
DEFINE_CAST(uint32_t, UInt32)
DEFINE_CAST(uint64_t, UInt64)
#undef DEFINE_CAST

// 定义一个宏，用于实现不同数据类型的 item() 方法模板特化，返回特定类型的标量值
#define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

// 使用宏扩展定义上面每种标量类型的 item() 方法模板特化
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
#undef DEFINE_ITEM

} //namespace at
```