# `.\pytorch\torch\csrc\onnx\back_compat.h`

```
#pragma once

#include <onnx/onnx_pb.h>

namespace torch::onnx {

// 在此定义以下常量，以避免破坏 Meta 对 ONNX 的内部使用，因为 Meta 的使用早于 ONNX 1.14，不支持 FLOAT8：
// 参考：https://github.com/pytorch/pytorch/pull/106379#issuecomment-1675189340
// -abock, 2023-08-25

// 定义 ONNX 的新数据类型常量，用于 FLOAT8E4M3FN
constexpr auto TensorProto_DataType_FLOAT8E4M3FN =
    static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(17);

// 定义 ONNX 的新数据类型常量，用于 FLOAT8E4M3FNUZ
constexpr auto TensorProto_DataType_FLOAT8E4M3FNUZ =
    static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(18);

// 定义 ONNX 的新数据类型常量，用于 FLOAT8E5M2
constexpr auto TensorProto_DataType_FLOAT8E5M2 =
    static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(19);

// 定义 ONNX 的新数据类型常量，用于 FLOAT8E5M2FNUZ
constexpr auto TensorProto_DataType_FLOAT8E5M2FNUZ =
    static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(20);

} // namespace torch::onnx
```