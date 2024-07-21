# `.\pytorch\aten\src\ATen\native\Activation.h`

```
#pragma once`
#pragma once

#include <ATen/native/DispatchStub.h>  // 包含 ATen 库的分发存根头文件
#include <c10/util/Exception.h>       // 包含 c10 库的异常处理头文件
#include <c10/util/string_view.h>     // 包含 c10 库的字符串视图头文件

namespace c10 {
class Scalar;  // 定义 c10 命名空间中的标量类
}

namespace at {
struct TensorIterator;       // 定义 at 命名空间中的张量迭代器结构体
struct TensorIteratorBase;   // 定义 at 命名空间中的基础张量迭代器结构体
class TensorBase;            // 定义 at 命名空间中的张量基类
}

namespace at::native {

// 这些常量控制 gelu 函数的近似行为。
enum class GeluType {
  None,             // 基准 Gelu
  Tanh,             // Tanh Gelu 近似
  END
};

// 根据字符串视图返回 GeluType 枚举值
inline GeluType get_gelutype_enum(const c10::string_view approximate) {
  if (approximate == "none") {
    return GeluType::None;  // 如果字符串为 "none"，返回 None 类型
  } else if (approximate == "tanh") {
    return GeluType::Tanh;  // 如果字符串为 "tanh"，返回 Tanh 类型
  } else {
    TORCH_CHECK(false, "approximate argument must be either none or tanh.");  // 否则抛出异常
  }
}

// 将 GeluType 转换为字符串
inline std::string gelutype_to_string(const GeluType type) {
  switch(type) {
    case GeluType::None: return "none";   // 如果类型为 None，返回字符串 "none"
    case GeluType::Tanh: return "tanh";   // 如果类型为 Tanh，返回字符串 "tanh"
    default: TORCH_CHECK(false, "unknown GELU type: ", static_cast<int>(type));  // 否则抛出未知类型异常
  }
}

// 定义结构化激活函数指针类型
using structured_activation_fn = void (*)(TensorIteratorBase&);
// 定义结构化激活函数反向传播指针类型
using structured_activation_backward_fn = void (*)(TensorIteratorBase&);

// 定义激活函数指针类型
using activation_fn = void (*)(TensorIterator&);
// 定义激活函数反向传播指针类型
using activation_backward_fn = void (*)(TensorIterator&);
// 定义软加函数指针类型
using softplus_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
// 定义软加函数反向传播指针类型
using softplus_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
// 定义阈值函数指针类型
using threshold_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
// 定义硬切函数反向传播指针类型
using hardtanh_backward_fn = void (*)(TensorIterator&, const c10::Scalar&, const c10::Scalar&);
// 定义硬 Sigmoid 函数指针类型
using hardsigmoid_fn = void (*)(TensorIteratorBase&);
// 定义硬 Sigmoid 函数反向传播指针类型
using hardsigmoid_backward_fn = void (*)(TensorIteratorBase&);
// 定义硬 Swish 函数指针类型
using hardswish_fn = void (*)(TensorIterator&);
// 定义硬 Swish 函数反向传播指针类型
using hardswish_backward_fn = void (*)(TensorIterator&);
// 定义收缩函数指针类型
using shrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
// 定义软收缩函数指针类型
using softshrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
// 定义收缩函数反向传播指针类型
using shrink_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
// 定义 ELU 函数指针类型
using elu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&);
// 定义 ELU 函数反向传播指针类型
using elu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&, bool);
// 定义 Leaky ReLU 函数指针类型
using leaky_relu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
// 定义 Leaky ReLU 函数反向传播指针类型
using leaky_relu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
// 定义 log sigmoid CPU 函数指针类型
using log_sigmoid_cpu_fn = void (*)(TensorBase&, TensorBase&, const TensorBase&);
// 定义 Gelu 函数指针类型
using gelu_fn = void (*)(TensorIteratorBase&, GeluType);
// 定义 Gelu 函数反向传播指针类型
using gelu_backward_fn = void (*)(TensorIteratorBase&, GeluType);
// 定义 GLU JVP 函数指针类型
using glu_jvp_fn = void (*)(TensorIteratorBase&);

// 声明 ELU 函数分发存根
DECLARE_DISPATCH(elu_fn, elu_stub);
// 声明 ELU 函数反向传播分发存根
DECLARE_DISPATCH(elu_backward_fn, elu_backward_stub);
// 声明并定义一系列的函数指针，用于分派不同的激活函数的反向传播计算
DECLARE_DISPATCH(activation_backward_fn, log_sigmoid_backward_stub);
DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(gelu_fn, GeluKernel);
DECLARE_DISPATCH(gelu_backward_fn, GeluBackwardKernel);
DECLARE_DISPATCH(hardtanh_backward_fn, hardtanh_backward_stub);
DECLARE_DISPATCH(hardsigmoid_fn, hardsigmoid_stub);
DECLARE_DISPATCH(hardsigmoid_backward_fn, hardsigmoid_backward_stub);
DECLARE_DISPATCH(hardswish_fn, hardswish_stub);
DECLARE_DISPATCH(hardswish_backward_fn, hardswish_backward_stub);
DECLARE_DISPATCH(shrink_fn, hardshrink_stub);
DECLARE_DISPATCH(softshrink_fn, softshrink_stub);
DECLARE_DISPATCH(shrink_backward_fn, shrink_backward_stub);
DECLARE_DISPATCH(leaky_relu_fn, leaky_relu_stub);
DECLARE_DISPATCH(leaky_relu_backward_fn, leaky_relu_backward_stub);
DECLARE_DISPATCH(structured_activation_fn, glu_stub);
DECLARE_DISPATCH(activation_backward_fn, glu_backward_stub);
DECLARE_DISPATCH(glu_jvp_fn, glu_jvp_stub);
DECLARE_DISPATCH(structured_activation_fn, silu_stub);
DECLARE_DISPATCH(structured_activation_backward_fn, silu_backward_stub);
DECLARE_DISPATCH(structured_activation_fn, mish_stub);
DECLARE_DISPATCH(activation_backward_fn, mish_backward_stub);
DECLARE_DISPATCH(activation_fn, prelu_stub);

} // namespace at::native
```