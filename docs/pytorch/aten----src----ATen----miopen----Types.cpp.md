# `.\pytorch\aten\src\ATen\miopen\Types.cpp`

```py
#include <ATen/miopen/Types.h>  // 引入 miopen 的数据类型定义

#include <ATen/ATen.h>  // 引入 PyTorch ATen 库
#include <miopen/version.h>  // 引入 miopen 的版本信息

namespace at { namespace native {

miopenDataType_t getMiopenDataType(const at::Tensor& tensor) {
  // 根据 PyTorch Tensor 的标量类型返回对应的 miopen 数据类型
  if (tensor.scalar_type() == at::kFloat) {
    return miopenFloat;
  } else if (tensor.scalar_type() == at::kHalf) {
    return miopenHalf;
  }  else if (tensor.scalar_type() == at::kBFloat16) {
    return miopenBFloat16;
  }
  // 若无对应的 miopen 数据类型，则抛出运行时错误
  std::string msg("getMiopenDataType() not supported for ");
  msg += toString(tensor.scalar_type());
  throw std::runtime_error(msg);
}

int64_t miopen_version() {
  // 返回 miopen 的版本号，以整数形式表示
  return (MIOPEN_VERSION_MAJOR<<8) + (MIOPEN_VERSION_MINOR<<4) + MIOPEN_VERSION_PATCH;
}

}}  // namespace at::miopen
```