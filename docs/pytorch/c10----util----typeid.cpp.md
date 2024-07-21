# `.\pytorch\c10\util\typeid.cpp`

```
cpp
// 包含异常处理和类型标识头文件
#include <c10/util/Exception.h>
#include <c10/util/typeid.h>

// 包含算法和原子操作头文件
#include <algorithm>
#include <atomic>

// Caffe2命名空间
namespace caffe2 {

// Caffe2内部细节命名空间
namespace detail {

// 抛出运行时类型逻辑错误的函数定义
C10_EXPORT void _ThrowRuntimeTypeLogicError(const std::string& msg) {
  // 在早期版本中曾使用std::abort()，但对于库而言可能过于激烈了
  TORCH_CHECK(false, msg);
}

} // namespace detail

// 标记函数不返回的类型信息错误函数定义
[[noreturn]] void TypeMeta::error_unsupported_typemeta(caffe2::TypeMeta dtype) {
  // 检查条件，如果为真，则抛出错误消息
  TORCH_CHECK(
      false,
      "Unsupported TypeMeta in ATen: ",
      dtype,
      " (please report this error)");
}

// 获取类型元数据锁的静态函数
std::mutex& TypeMeta::getTypeMetaDatasLock() {
  // 静态局部变量锁
  static std::mutex lock;
  return lock;
}

// 获取下一个类型索引的静态函数声明
uint16_t TypeMeta::nextTypeIndex(NumScalarTypes);

// 获取固定长度的TypeMetaData实例数组的函数定义
detail::TypeMetaData* TypeMeta::typeMetaDatas() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static detail::TypeMetaData instances[MaxTypeIndex + 1] = {
#define SCALAR_TYPE_META(T, name)        \
  /* ScalarType::name */                 \
  detail::TypeMetaData(                  \
      sizeof(T),                         \
      detail::_PickNew<T>(),             \
      detail::_PickPlacementNew<T>(),    \
      detail::_PickCopy<T>(),            \
      detail::_PickPlacementDelete<T>(), \
      detail::_PickDelete<T>(),          \
      TypeIdentifier::Get<T>(),          \
      c10::util::get_fully_qualified_type_name<T>()),
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
#undef SCALAR_TYPE_META
      // 数组的剩余部分用TypeMetaData空白填充。
      // 第一个条目用于ScalarType::Undefined。
      // 其余的条目由CAFFE_KNOWN_TYPE条目占用。
  };
  return instances;
}

// 返回给定类型标识符对应的现有元数据索引的函数定义
uint16_t TypeMeta::existingMetaDataIndexForType(TypeIdentifier identifier) {
  auto* metaDatas = typeMetaDatas();
  const auto end = metaDatas + nextTypeIndex;
  // MaxTypeIndex不是很大，线性搜索是可以接受的。
  auto it = std::find_if(metaDatas, end, [identifier](const auto& metaData) {
    return metaData.id_ == identifier;
  });
  if (it == end) {
    return MaxTypeIndex;
  }
  return static_cast<uint16_t>(it - metaDatas);
}

// 定义已知类型的宏定义示例
CAFFE_DEFINE_KNOWN_TYPE(std::string, std_string)
CAFFE_DEFINE_KNOWN_TYPE(uint16_t, uint16_t)
CAFFE_DEFINE_KNOWN_TYPE(char, char)
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::mutex>, std_unique_ptr_std_mutex)
CAFFE_DEFINE_KNOWN_TYPE(
    std::unique_ptr<std::atomic<bool>>,
    std_unique_ptr_std_atomic_bool)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int32_t>, std_vector_int32_t)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int64_t>, std_vector_int64_t)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<unsigned long>, std_vector_unsigned_long)
CAFFE_DEFINE_KNOWN_TYPE(bool*, bool_ptr)
CAFFE_DEFINE_KNOWN_TYPE(char*, char_ptr)
CAFFE_DEFINE_KNOWN_TYPE(int*, int_ptr)

CAFFE_DEFINE_KNOWN_TYPE(
    detail::_guard_long_unique<long>,
    detail_guard_long_unique_long);

// 结束caffe2命名空间
} // namespace caffe2
    detail::_guard_long_unique<std::vector<long>>,  # 实例化 detail 命名空间中的 _guard_long_unique 模板，模板参数为 std::vector<long>
    detail_guard_long_unique_std_vector_long)       # 传递 detail_guard_long_unique_std_vector_long 作为参数
# 定义 CAFFE 的已知类型映射，将 'float*' 映射为 'float_ptr'
CAFFE_DEFINE_KNOWN_TYPE(float*, float_ptr)
# 定义 CAFFE 的已知类型映射，将 'at::Half*' 映射为 'at_Half'
CAFFE_DEFINE_KNOWN_TYPE(at::Half*, at_Half)

# 结束命名空间 'caffe2'
} // namespace caffe2
```