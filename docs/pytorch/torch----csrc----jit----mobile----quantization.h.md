# `.\pytorch\torch\csrc\jit\mobile\quantization.h`

```py
#pragma once

#pragma once 指令：确保头文件只被编译一次，避免重复包含。


#include <c10/macros/Export.h>
#include <string>

包含头文件 `<c10/macros/Export.h>` 和 `<string>`。


namespace torch {
namespace jit {
namespace mobile {

命名空间声明：定义了嵌套的命名空间 torch::jit::mobile。


class Module;
namespace quantization {

声明类 Module 和命名空间 quantization。


/*
 * Device side PTQ API.
 * Once the model has been prepared for quantization on server side, such model
 * is sent to device. On device side the model is further trained. At the end of
 * the training, before the model is readied for inference, we need to quantize
 * the model.
 * Usage of this API is as follows.
 * PTQQuanizationHelper ptq_helper;
 * ptq_helper.quantize_dynamic(m, "forward");
 * Args:
 * m: Captured by reference, an instance of mobile::Module. This module will be
 * mutated in place to replace its <method_name> method with quantized
 * equivalent. method:name: Name of the method to be quantized. AOT preparation
 * for quantization must also have been done for this method. Returns: In place
 * mutated `m` whose size should be smaller due to weight quantization and whose
 * <method_name> method should use quantized ops
 */

注释部分：描述了 Device side PTQ API 的功能和用法。


class TORCH_API PTQQuanizationHelper {
 public:
  PTQQuanizationHelper() = default;
  void quantize_dynamic(
      torch::jit::mobile::Module& m,
      const std::string& method_name);
};

声明 PTQQuanizationHelper 类，其中包含 quantize_dynamic 方法用于动态量化模型。


} // namespace quantization
} // namespace mobile
} // namespace jit
} // namespace torch

命名空间的结尾。
```