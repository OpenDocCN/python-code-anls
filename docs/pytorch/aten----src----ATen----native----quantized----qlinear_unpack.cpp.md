# `.\pytorch\aten\src\ATen\native\quantized\qlinear_unpack.cpp`

```
/*
这段代码是针对 fbgemm、qnnpack 和 cudnn 后端的分发注册。
在运行时通过 packed_weight 指针的多态性确定正确的解包后端函数。
packed_weight 是 intrusive_ptr<LinearPackedParamsBase> 类型的指针，
在运行时可以指向 PackedLinearWeightsQnnp、PackedLinearWeights（Fbgemm）或 PackedLinearWeightsCudnn，
这些类都继承自 LinearPackedParamsBase。
解包函数的实现可以在 /cpu/LinearUnpackImpl.cpp（用于 fbgemm 和 qnnpack）和
/cudnn/linear_unpack_impl.cpp（用于 cudnn）中找到。
*/

#include <ATen/ATen.h>  // 包含 ATen 库
#include <ATen/native/quantized/cpu/fbgemm_utils.h>  // 包含 fbgemm 的相关工具函数
#include <ATen/native/quantized/PackedParams.h>  // 包含 PackedParams 相关定义
#include <ATen/native/quantized/cpu/QnnpackUtils.h>  // 包含 qnnpack 的相关工具函数
#include <torch/custom_class.h>  // 包含自定义类相关的头文件
#include <torch/library.h>  // 包含 Torch 库

int register_linear_params();  // 声明注册线性参数的函数

namespace at {  // 进入 at 命名空间
namespace native {  // 进入 native 命名空间
namespace {  // 匿名命名空间，用于定义内部类和实现

class QLinearUnpackWeightInt8 final {  // 定义 QLinearUnpackWeightInt8 类
 public:  // 公共部分开始
  static std::tuple<at::Tensor, std::optional<Tensor>> run(  // 静态方法 run，返回解包后的 Tensor 和可选的 Tensor
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {  // 参数为 intrusive_ptr<LinearPackedParamsBase> 类型的 packed_weight 指针
    return packed_weight->unpack();  // 调用 packed_weight 的 unpack 方法并返回结果
  }
};

class QLinearUnpackWeightFp16 final {  // 定义 QLinearUnpackWeightFp16 类
 public:  // 公共部分开始
  static std::tuple<at::Tensor, std::optional<Tensor>> run(  // 静态方法 run，返回解包后的 Tensor 和可选的 Tensor
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {  // 参数为 intrusive_ptr<LinearPackedParamsBase> 类型的 packed_weight 指针
    auto& ctx = at::globalContext();  // 获取全局上下文 ctx

    TORCH_CHECK(  // 断言检查
        ctx.qEngine() != at::QEngine::QNNPACK,  // 如果 qEngine 不是 QNNPACK
        "quantized::linear_unpack_fp16 is currently "  // 报错信息
        "not supported by QNNPACK");

    return packed_weight->unpack();  // 调用 packed_weight 的 unpack 方法并返回结果
  }
};

class QLinearUnpackWeightInt8Legacy final {  // 定义 QLinearUnpackWeightInt8Legacy 类
 public:  // 公共部分开始
  static std::tuple<at::Tensor, std::optional<Tensor>> run(  // 静态方法 run，返回解包后的 Tensor 和可选的 Tensor
      const at::Tensor& packed_weight) {  // 参数为 Tensor 类型的 packed_weight
    TORCH_CHECK(false,  // 报错断言，始终为假
        "quantized.linear_unpack(Tensor) is unsupported! Please "  // 报错信息
        "upgrade your model to use the newer quantized.linear_"  // 续行信息
        "unpack(LinearPackedParamsBase) overload");
  }
};

class QLinearUnpackWeightFp16Legacy final {  // 定义 QLinearUnpackWeightFp16Legacy 类
 public:  // 公共部分开始
  static std::tuple<at::Tensor, std::optional<Tensor>> run(  // 静态方法 run，返回解包后的 Tensor 和可选的 Tensor
      const at::Tensor& packed_weight) {  // 参数为 Tensor 类型的 packed_weight
    TORCH_CHECK(false,  // 报错断言，始终为假
        "quantized.linear_unpack(Tensor) is unsupported! Please "  // 报错信息
        "upgrade your model to use the newer quantized.linear_"  // 续行信息
        "unpack(LinearPackedParamsBase) overload");
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {  // Torch 库的 quantized 命名空间实现
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack.legacy"), TORCH_FN(QLinearUnpackWeightInt8Legacy::run));  // 实现 quantized::linear_unpack.legacy，调用 QLinearUnpackWeightInt8Legacy::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16.legacy"), TORCH_FN(QLinearUnpackWeightFp16Legacy::run));  // 实现 quantized::linear_unpack_fp16.legacy，调用 QLinearUnpackWeightFp16Legacy::run
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {  // Torch 库的 quantized 命名空间 CatchAll 实现
  register_linear_params();  // 注册线性参数
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack"), TORCH_FN(QLinearUnpackWeightInt8::run));  // 实现 quantized::linear_unpack，调用 QLinearUnpackWeightInt8::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16"), TORCH_FN(QLinearUnpackWeightFp16::run));  // 实现 quantized::linear_unpack_fp16，调用 QLinearUnpackWeightFp16::run
}

} // namespace
} // namespace native
} // namespace at
```