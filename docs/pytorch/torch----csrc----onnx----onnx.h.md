# `.\pytorch\torch\csrc\onnx\onnx.h`

```
#pragma once
// 声明了一个命名空间 torch::onnx

namespace torch::onnx {

// 定义了一个枚举类型 OperatorExportTypes，用于指定导出类型
enum class OperatorExportTypes {
  ONNX, // 严格的 ONNX 导出
  ONNX_ATEN, // 在所有地方使用 ATen 操作的 ONNX 导出
  ONNX_ATEN_FALLBACK, // 带有 ATen 回退的 ONNX 导出
  ONNX_FALLTHROUGH, // 导出支持的 ONNX 操作，对不支持的操作进行传递
};

// 定义了一个枚举类型 TrainingMode，用于指定训练模式
enum class TrainingMode {
  EVAL, // 推断模式
  PRESERVE, // 保持模型状态（推断/训练）
  TRAINING, // 训练模式
};

// 声明了一个常量字符串 kOnnxNodeNameAttribute，用于表示 ONNX 节点名称的属性
constexpr char kOnnxNodeNameAttribute[] = "onnx_name";

} // namespace torch::onnx
```