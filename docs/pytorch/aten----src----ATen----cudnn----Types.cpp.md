# `.\pytorch\aten\src\ATen\cudnn\Types.cpp`

```
// 包含 ATen 库中与 cudnn 相关的数据类型定义
#include <ATen/cudnn/Types.h>

// 包含 ATen 核心库
#include <ATen/ATen.h>

// at 命名空间下的 native 子命名空间
namespace at { namespace native {

// 根据标量类型获取对应的 cudnn 数据类型
cudnnDataType_t getCudnnDataTypeFromScalarType(const at::ScalarType dtype) {
  // 检查标量类型是否为 kQInt8
  if (dtype == c10::kQInt8) {
    return CUDNN_DATA_INT8;
  } else if (dtype == at::kFloat) {  // 检查标量类型是否为 kFloat
    return CUDNN_DATA_FLOAT;
  } else if (dtype == at::kDouble) {  // 检查标量类型是否为 kDouble
    return CUDNN_DATA_DOUBLE;
  } else if (dtype == at::kHalf) {    // 检查标量类型是否为 kHalf
    return CUDNN_DATA_HALF;
  }
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
  else if (dtype == at::kBFloat16) {  // 检查标量类型是否为 kBFloat16
    return CUDNN_DATA_BFLOAT16;
  } else if (dtype == at::kInt) {     // 检查标量类型是否为 kInt
    return CUDNN_DATA_INT32;
  } else if (dtype == at::kByte) {    // 检查标量类型是否为 kByte
    return CUDNN_DATA_UINT8;
  } else if (dtype == at::kChar) {    // 检查标量类型是否为 kChar
    return CUDNN_DATA_INT8;
  }
#endif
  // 若未匹配到任何 cudnn 数据类型，则抛出运行时错误
  std::string msg("getCudnnDataTypeFromScalarType() not supported for ");
  msg += toString(dtype);  // 获取标量类型名称并添加到错误消息中
  throw std::runtime_error(msg);  // 抛出运行时异常
}

// 获取给定张量的 cudnn 数据类型
cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  return getCudnnDataTypeFromScalarType(tensor.scalar_type());  // 调用函数获取对应的 cudnn 数据类型
}

// 返回当前 cudnn 版本号
int64_t cudnn_version() {
  return CUDNN_VERSION;  // 返回预编译的 cudnn 版本号
}

}}  // namespace at::cudnn  // 结束 at 命名空间和 cudnn 子命名空间
```