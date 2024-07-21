# `.\pytorch\aten\src\ATen\native\ComparisonUtils.cpp`

```py
// 引入 ATen 库中的头文件，用于张量操作
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <algorithm>  // 引入算法库，提供通用算法操作
#include <c10/util/OptionalArrayRef.h>  // 引入可选数组引用，用于处理可选的数组参数

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_assert_tensor_metadata_native.h>  // 如果定义了 AT_PER_OPERATOR_HEADERS，引入张量元数据断言操作
#endif

namespace at {  // 进入 ATen 命名空间

class Tensor;  // 前置声明张量类

namespace native {  // 进入 native 命名空间

// 比较函数模板，用于比较两个对象是否相等，如果不相等则抛出异常
template<typename O, typename C>
void _assert_match(const O& original, const C& compared, const std::string& name) {
  // 如果 compared 有值
  if (compared) {
    // 检查 original 和 compared 的值是否相等
    bool equal = (original == compared.value());
    // 如果不相等
    if (!equal) {
      std::stringstream msg;  // 创建字符串流
      msg << "Tensor " << name << " mismatch!";  // 构造错误消息
      AT_ASSERT(equal, msg.str());  // 断言失败，抛出异常，显示错误消息
    }
  }
}

// 断言张量元数据的函数，检查张量的尺寸、步长和数据类型是否与给定值匹配
void _assert_tensor_metadata(at::Tensor const& tensor, at::OptionalIntArrayRef sizes, at::OptionalIntArrayRef strides, std::optional<c10::ScalarType> dtype) {
  _assert_match(tensor.sizes(), sizes, "sizes");  // 检查张量的尺寸是否匹配
  _assert_match(tensor.strides(), strides, "strides");  // 检查张量的步长是否匹配
  _assert_match(tensor.dtype(), dtype, "dtype");  // 检查张量的数据类型是否匹配
}

}
}  // 结束 at::native 命名空间
```