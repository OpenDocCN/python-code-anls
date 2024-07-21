# `.\pytorch\aten\src\ATen\native\vulkan\ops\Register.cpp`

```
#ifdef USE_VULKAN_API
// 包含 Vulkan 相关的头文件，用于 Vulkan API 的操作
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/vulkan/ops/Batchnorm.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Lstm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Register.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 注册 Vulkan 下 Conv2dPackedContext 类的函数，用于 Torch 库的选择性注册
int register_vulkan_conv2d_packed_context() {
  // 创建静态变量来注册 Conv2dPackedContext 类型，标记为 Vulkan 类型
  static auto register_vulkan_conv2d_context =
      torch::selective_class_<Conv2dPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("Conv2dPackedContext"))
          // 定义该类的序列化和反序列化方法
          .def_pickle(
              // __getstate__ 方法定义，将 packed 状态的 context 解包
              [](const c10::intrusive_ptr<Conv2dPackedContext>& context) {
                return context->unpack();
              },
              // __setstate__ 方法定义，使用 unpacked 状态的 state 创建新的 context
              [](c10::impl::GenericList state) {
                return c10::make_intrusive<Conv2dPackedContext>(
                    Conv2dPackedContext::pack(state));
              });
  // 返回注册成功的状态
  return 0;
}

// 注册 Vulkan 下 Conv1dPackedContext 类的函数，用于 Torch 库的选择性注册
int register_vulkan_conv1d_packed_context() {
  // 创建静态变量来注册 Conv1dPackedContext 类型，标记为 Vulkan 类型
  static auto register_vulkan_conv1d_context =
      torch::selective_class_<Conv1dPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("Conv1dPackedContext"))
          // 定义该类的序列化和反序列化方法
          .def_pickle(
              // __getstate__ 方法定义，将 packed 状态的 context 解包
              [](const c10::intrusive_ptr<Conv1dPackedContext>& context) {
                return context->unpack();
              },
              // __setstate__ 方法定义，使用 unpacked 状态的 state 创建新的 context
              [](c10::impl::GenericList state) {
                return c10::make_intrusive<Conv1dPackedContext>(
                    Conv1dPackedContext::pack(state));
              });
  // 返回注册成功的状态
  return 0;
}

// 注册 Vulkan 下 LinearPackedContext 类的函数，用于 Torch 库的选择性注册
int register_vulkan_linear_packed_context() {
  // 创建静态变量来注册 LinearPackedContext 类型，标记为 Vulkan 类型
  static auto register_vulkan_linear_context =
      torch::selective_class_<LinearPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("LinearPackedContext"))
          // 定义该类的序列化和反序列化方法
          .def_pickle(
              // __getstate__ 方法定义，将 packed 状态的 context 解包
              [](const c10::intrusive_ptr<LinearPackedContext>& context) {
                return context->unpack();
              },
              // __setstate__ 方法定义，使用 unpacked 状态的 state 创建新的 context
              [](c10::impl::GenericList state) {
                return c10::make_intrusive<LinearPackedContext>(
                    LinearPackedContext::pack(state));
              });
  // 返回注册成功的状态
  return 0;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
#endif
// 注册 Vulkan Layernorm Packed Context 的函数
int register_vulkan_layernorm_packed_context() {
  // 使用静态变量注册 Vulkan Layernorm Context 类型，命名为 "vulkan"，选择性地指定类名 "LayernormPackedContext"
  static auto register_vulkan_layernorm_context =
      torch::selective_class_<LayernormPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("LayernormPackedContext"))
          // 定义该类的序列化和反序列化方法
          .def_pickle(
              // __getstate__ 方法的实现，接收 LayernormPackedContext 的指针 context
              [](const c10::intrusive_ptr<LayernormPackedContext>& context) {
                // 返回 context 中打包的数据，说明 context 是被打包的
                return context->unpack();
              },
              // __setstate__ 方法的实现，接收 c10::impl::GenericList 类型的 state
              [](c10::impl::GenericList state) {
                // 返回使用 unpacked 的 state 创建 LayernormPackedContext 对象
                return c10::make_intrusive<LayernormPackedContext>(
                    LayernormPackedContext::pack(state));
              });
  // 返回成功注册的标志
  return 0;
}

// 匿名命名空间开始
namespace {
TORCH_LIBRARY(vulkan, m) {
  // 定义 BatchNormPackedContext 类
  m.class_<BatchNormPackedContext>("BatchNormPackedContext")
      .def_pickle(
          // 序列化 BatchNormPackedContext 对象的状态
          [](const c10::intrusive_ptr<BatchNormPackedContext>& context) {
            // 将已打包的 context 解包
            return context->unpack();
          },
          // 反序列化 BatchNormPackedContext 对象的状态
          [](c10::impl::GenericList state) {
            // state 是已解包的状态
            return c10::make_intrusive<BatchNormPackedContext>(
                BatchNormPackedContext::pack(state));
          });
  // 定义 GruPackedContext 类
  m.class_<GruPackedContext>("GruPackedContext")
      .def_pickle(
          // 序列化 GruPackedContext 对象的状态
          [](const c10::intrusive_ptr<GruPackedContext>& context) {
            // 将已打包的 context 解包
            return context->unpack();
          },
          // 反序列化 GruPackedContext 对象的状态
          [](c10::impl::GenericList state) {
            // state 是已解包的状态
            return c10::make_intrusive<GruPackedContext>(
                GruPackedContext::pack(state));
          });
  // 定义 LstmPackedContext 类
  m.class_<LstmPackedContext>("LstmPackedContext")
      .def_pickle(
          // 序列化 LstmPackedContext 对象的状态
          [](const c10::intrusive_ptr<LstmPackedContext>& context) {
            // 将已打包的 context 解包
            return context->unpack();
          },
          // 反序列化 LstmPackedContext 对象的状态
          [](c10::impl::GenericList state) {
            // state 是已解包的状态
            return c10::make_intrusive<LstmPackedContext>(
                LstmPackedContext::pack(state));
          });
  // 注册 Vulkan 的卷积操作上下文
  register_vulkan_conv2d_packed_context();
  // 注册 Vulkan 的一维卷积操作上下文
  register_vulkan_conv1d_packed_context();
  // 注册 Vulkan 的线性层操作上下文
  register_vulkan_linear_packed_context();
  // 注册 Vulkan 的层归一化操作上下文
  register_vulkan_layernorm_packed_context();
  // 为了保持向后兼容性
  // 定义 Conv2dOpContext 类
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          // 序列化 Conv2dOpContext 对象的状态
          [](const c10::intrusive_ptr<Conv2dOpContext>& context) {
            // 将已打包的 context 解包
            return context->unpack();
          },
          // 反序列化 Conv2dOpContext 对象的状态
          [](Conv2dOpContext::State state) {
            // 使用预打包的参数创建 Conv2dOpContext 对象
            return conv2d_clamp_prepack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::get<5>(state),
                std::get<6>(state),
                std::get<7>(state));
          });
}
# 定义 TORCH_LIBRARY_IMPL 宏，将 Vulkan 预打包库的 CPU 实现添加到模块 m 中
TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
    # 实现创建 Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_conv2d_context"),
        TORCH_FN(create_conv2d_context));
    # 实现 Conv2D Clamp 预打包函数的映射（向后兼容）
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"),
        TORCH_FN(conv2d_clamp_prepack)); // Backwards compatibility
    # 实现创建 Transposed Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_tconv2d_context"),
        TORCH_FN(create_tconv2d_context));
    # 实现创建 Conv1D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_conv1d_context"),
        TORCH_FN(create_conv1d_context));
    # 实现创建 Linear 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context"),
        TORCH_FN(create_linear_context));
    # 实现创建 Layernorm 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_layernorm_context"),
        TORCH_FN(create_layernorm_context));
    # 实现创建 GRU 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_gru_context"),
        TORCH_FN(create_gru_context));
    # 实现创建 LSTM 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_lstm_context"),
        TORCH_FN(create_lstm_context));
    # 实现创建 Batchnorm 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_batchnorm_context"),
        TORCH_FN(create_batchnorm_context));
}

# 定义 TORCH_LIBRARY_IMPL 宏，将 Vulkan 预打包库的 QuantizedCPU 实现添加到模块 m 中
TORCH_LIBRARY_IMPL(vulkan_prepack, QuantizedCPU, m) {
    # 实现创建 Quantized Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_qconv2d_context"),
        TORCH_FN(create_qconv2d_context));
    # 实现创建 Quantized Transposed Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::create_qtconv2d_context"),
        TORCH_FN(create_qtconv2d_context));
}

# 定义 TORCH_LIBRARY_IMPL 宏，将 Vulkan 预打包库的 Vulkan 实现添加到模块 m 中
TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
    # 实现运行 Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_conv2d_context"),
        TORCH_FN(run_conv2d_context));
    # 实现运行 Conv2D Clamp 函数的映射（向后兼容）
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"),
        TORCH_FN(conv2d_clamp_run)); // Backwards compatibility
    # 实现运行 Transposed Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_tconv2d_context"),
        TORCH_FN(run_tconv2d_context));
    # 实现运行 Quantized Conv2D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_qconv2d_context"),
        TORCH_FN(run_qconv2d_context));
    # 实现运行 Conv1D 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_conv1d_context"),
        TORCH_FN(run_conv1d_context));
    # 实现运行 Linear 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_linear_context"),
        TORCH_FN(run_linear_context));
    # 实现运行 Layernorm 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_layernorm_context"),
        TORCH_FN(run_layernorm_context));
    # 实现运行 Quantized Linear 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_qlinear_context"),
        TORCH_FN(run_qlinear_context));
    # 实现运行 GRU 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_gru_context"),
        TORCH_FN(run_gru_context));
    # 实现运行 LSTM 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_lstm_context"),
        TORCH_FN(run_lstm_context));
    # 实现运行 Batchnorm 上下文的函数映射
    m.impl(
        TORCH_SELECTIVE_NAME("vulkan_prepack::run_batchnorm_context"),
        TORCH_FN(run_batchnorm_context));
}
// 定义了 Vulkan 加速的量化操作的 Torch 库
TORCH_LIBRARY(vulkan_quantized, m) {
  // 定义 Vulkan 加速的量化加法操作的 Torch 模块方法
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::add(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point) -> Tensor qc"));
  // 定义 Vulkan 加速的量化减法操作的 Torch 模块方法
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::sub(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
  // 定义 Vulkan 加速的量化乘法操作的 Torch 模块方法
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::mul(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
  // 定义 Vulkan 加速的量化除法操作的 Torch 模块方法
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::div(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
}

// 实现了 Vulkan 加速的量化操作的 Torch 库
TORCH_LIBRARY_IMPL(vulkan_quantized, Vulkan, m) {
  // 实现 Vulkan 加速的量化加法操作的 Torch 模块方法
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::add"), TORCH_FN(quantized_add));
  // 实现 Vulkan 加速的量化减法操作的 Torch 模块方法
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::sub"), TORCH_FN(quantized_sub));
  // 实现 Vulkan 加速的量化乘法操作的 Torch 模块方法
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::mul"), TORCH_FN(quantized_mul));
  // 实现 Vulkan 加速的量化除法操作的 Torch 模块方法
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::div"), TORCH_FN(quantized_div));
}

// 结束 Vulkan 量化操作的命名空间声明
} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

// 结束条件编译指令，指示使用 Vulkan API
#endif /* USE_VULKAN_API */
```