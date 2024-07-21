# `.\pytorch\aten\src\ATen\core\QuantizerBase.h`

```
#pragma once
// 预处理指令：#pragma once，确保头文件只被编译一次

#include <c10/core/ScalarType.h>
// 包含标量类型定义头文件

#include <c10/core/QScheme.h>
// 包含量化方案枚举头文件

#include <c10/util/intrusive_ptr.h>
// 包含intrusive_ptr工具头文件，用于管理智能指针

namespace at {
// 命名空间开始

class Tensor;
// 前置声明Tensor类

struct QTensorImpl;
// 前置声明QTensorImpl结构体

struct Quantizer;
// 声明Quantizer结构体

using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;
// 定义常量引用类型ConstQuantizerPtr，指向Quantizer的intrusive_ptr

using QuantizerPtr = c10::intrusive_ptr<Quantizer>;
// 定义QuantizerPtr类型，指向Quantizer的intrusive_ptr

/**
 * Quantizer is the class for storing all the information
 * that's necessary to perform quantize and dequantize
 * operation.
 *
 * We might have different types of quantization schemes and this is
 * the base class for all quantizers.
 *
 * QTensorImpl will hold a pointer to Quantizer so that we can support
 * different quantization schemes on Tensor.
 *
 * For example, the most common quantization scheme, Affine Quantization,
 * requires scale and zero_point as parameters, we'll store scale and zero_point
 * inside the instance and we can use it to quantize a float Tensor or
 * dequantize a quantized Tensor.
 *
 * When you add new types of leaf Quantizer class, please also
 * make sure to add a corresponding QScheme enum since
 * they should have one to one mapping.
 *
 * Note about intrusive_ptr:
 * Quantized Tensor holds an intrusive_ptr to Quantizer, and multiple Tensor can
 * share the same Quantizer. Quantizer should be immutable.
 */
struct TORCH_API Quantizer : public c10::intrusive_ptr_target {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ScalarType scalar_type_;
  // 常量成员变量scalar_type_，存储标量类型

  explicit Quantizer(ScalarType scalar_type) : scalar_type_(scalar_type) {}
  // 显式构造函数，初始化scalar_type_

  ~Quantizer() override;
  // 虚析构函数，用于释放资源

  // Copied from torch/csrc/jit/ir/scope.h
  QuantizerPtr intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this);
    // 增加引用计数，用于从this指针创建新的intrusive_ptr

    return c10::intrusive_ptr<Quantizer>::reclaim(this);
    // 通过原始指针this创建intrusive_ptr，重新获取所有权
  }

  /**
   * Each concrete Quantizer type should have a unique QScheme type.
   */
  virtual QScheme qscheme() const = 0;
  // 纯虚函数，子类需实现，返回量化方案类型QScheme

  ScalarType scalar_type() const {
    return scalar_type_;
    // 返回标量类型scalar_type_
  }

  /**
   * quantize a float Tensor into a quantized Tensor.
   */
  virtual Tensor quantize(const Tensor& t) = 0;
  // 纯虚函数，子类需实现，将浮点Tensor量化为量化Tensor

  /**
   * dequantize a quantized Tensor into a float Tensor.
   */
  virtual Tensor dequantize(const Tensor& t) = 0;
  // 纯虚函数，子类需实现，将量化Tensor反量化为浮点Tensor

  /**
   * dequantize a quantized Tensor into a float Tensor, out= variant
   */
  virtual Tensor& dequantize_out(Tensor& out, const Tensor& t) = 0;
  // 纯虚函数，子类需实现，将量化Tensor反量化为浮点Tensor，结果写入out参数

  /**
   * Compare against `other` for equality.
   */
  virtual bool equalTo(QuantizerPtr other) const = 0;
  // 纯虚函数，子类需实现，比较两个Quantizer对象是否相等
};

} // namespace at
// 命名空间结束
```