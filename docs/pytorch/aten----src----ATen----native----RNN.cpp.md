# `.\pytorch\aten\src\ATen\native\RNN.cpp`

```
// 定义宏，用于仅包含方法运算符的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库的 RNN 相关头文件
#include <ATen/native/RNN.h>

// 引入 ATen 核心张量和列表类头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
// 引入 ATen 上下文和张量操作头文件
#include <ATen/Context.h>
#include <ATen/TensorOperators.h>
// 引入 ATen MPS 设备头文件
#include <ATen/mps/MPSDevice.h>
// 引入 ATen 量化相关头文件
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
// 引入 C10 梯度模式和宏定义头文件
#include <c10/core/GradMode.h>
#include <c10/macros/Macros.h>
// 引入 C10 范围工具头文件
#include <c10/util/irange.h>
// 引入 Torch 自定义类和库头文件
#include <torch/custom_class.h>
#include <torch/library.h>
// 引入 ATen 配置头文件
#include <ATen/Config.h>

// 如果未定义每个操作符的 AT 头文件，引入普通 AT 操作和本地 AT 操作
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 引入特定操作的 AT 头文件，如 LSTM、GRU 相关实现和优化的头文件
#include <ATen/ops/_lstm_mps.h>
#include <ATen/ops/_thnn_differentiable_gru_cell_backward_native.h>
#include <ATen/ops/_thnn_differentiable_lstm_cell_backward_native.h>
#include <ATen/ops/_thnn_fused_gru_cell.h>
#include <ATen/ops/_thnn_fused_lstm_cell.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward_impl.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward_native.h>
#include <ATen/ops/_use_cudnn_rnn_flatten_weight_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cudnn_is_acceptable.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/fbgemm_linear_int8_weight_fp32_activation.h>
#include <ATen/ops/fbgemm_linear_quantize_weight_native.h>
#include <ATen/ops/fbgemm_pack_quantized_matrix_native.h>
#include <ATen/ops/gru_cell_native.h>
#include <ATen/ops/gru_native.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/lstm_cell_native.h>
#include <ATen/ops/lstm_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/quantized_gru_cell_native.h>
#include <ATen/ops/quantized_lstm_cell_native.h>
#include <ATen/ops/quantized_rnn_relu_cell_native.h>
#include <ATen/ops/quantized_rnn_tanh_cell_native.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/rnn_relu_cell_native.h>
#include <ATen/ops/rnn_relu_native.h>
#include <ATen/ops/rnn_tanh_cell_native.h>
#include <ATen/ops/rnn_tanh_native.h>
#include <ATen/ops/sigmoid_backward.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh_backward.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_like_ops.h>
#include <utility>
#endif

// 声明注册线性参数的函数
int register_linear_params();

// 命名空间开始：ATen 本地实现命名空间
namespace at::native {

// 匿名命名空间内定义：检查是否使用 MIOpen
bool use_miopen(const at::Tensor& input, const double dropout_state) {
    // 检查条件：输入张量类型为 kFloat 或 kHalf，编译支持 MIOpen，张量在 CUDA 设备上，且 CuDNN 已启用
    bool is_miopen_acceptable = ((input.scalar_type() == at::kFloat) || (input.scalar_type() == at::kHalf)) &&
                                (detail::getCUDAHooks().compiledWithMIOpen()) &&
                                (input.is_cuda()) &&
                                (at::globalContext().userEnabledCuDNN());
    // MIOpen 函数在空张量上返回 miopenStatusBadParm。虽然某些函数支持空张量，
    // 但本地内核应该不会因输出可能为空而慢太多。
    # 如果输入的符号数量为零，则返回 false
    if (input.sym_numel() == 0) return false;

    # 返回 miopen 是否可接受的标志
    return is_miopen_acceptable;
}

// 检查是否可以使用 MKL-DNN（Math Kernel Library for Deep Neural Networks）
bool use_mkldnn(const Tensor& input, TensorList params, TensorList hx) {
#if AT_MKLDNN_ENABLED()
  // 检查全局上下文中是否启用了 MKL-DNN
  if (!at::globalContext().userEnabledMkldnn()) {
    return false;
  }
  // 检查输入张量及其参数和隐藏状态张量是否都使用 CPU 后端
  auto is_cpu_backend = [&](const TensorList tensors) {
    bool backend_cpu = true;
    for (const auto& t : tensors) {
      if (!(t.options().backend() == at::Backend::CPU)) {
        backend_cpu = false;
        break;
      }
    }
    return backend_cpu;
  };
  // 返回是否满足使用 MKL-DNN 的条件：输入张量使用 CPU 后端，参数和隐藏状态张量也都使用 CPU 后端，
  // 且输入张量的数据类型为 float 或 bfloat16，并且元素数量不为 0
  return input.options().backend() == at::Backend::CPU &&
      is_cpu_backend(params) && is_cpu_backend(hx) &&
      (input.scalar_type() == kFloat || input.scalar_type() == kBFloat16) &&
      input.numel() != 0;
#endif
  return false;
}

template<typename T>
using pair_of = std::pair<T, T>;

template<typename T>
using tpair_of = std::tuple<T, T>;

// 这些本可以是函数指针，但 MSVC 不支持将函数指针用作模板参数
// 定义函数对象 tanh_f，用于计算张量的双曲正切函数
struct tanh_f {
  Tensor operator()(const Tensor& t) const { return at::tanh(t); }
};

// 定义函数对象 relu_f，用于计算张量的 ReLU 函数
struct relu_f {
  Tensor operator()(const Tensor& t) const { return at::relu(t); }
};

// 表示压缩序列的简单类型
// data 是数据张量，batch_sizes 是批次大小张量
struct PackedSequence {
  PackedSequence() = default;
  PackedSequence(Tensor _data, Tensor _batch_sizes)
    : data(std::move(_data)), batch_sizes(std::move(_batch_sizes)) {}

  Tensor data;
  Tensor batch_sizes;
};

// 简单的类型，用于 __getstate__/__setstate__ 序列化
//
// 元素 0 是字符串键，表示 CellParam 的类型，应为 cell_params_deserializers 的有效键之一
// 元素 1 是包含在 CellParams 实例中的张量
// 元素 2 是包含在 CellParams 实例中的双精度浮点数（如果有的话）
// 元素 3 是包含在 CellParams 实例中的长整型数（如果有的话）
// 元素 4 是包含在 CellParams 实例中的 LinearPackedParamsBase 的智能指针向量
using CellParamsSerializationType = std::tuple<
    std::string,
    std::vector<at::Tensor>,
    std::vector<double>,
    std::vector<int64_t>,
    std::vector<c10::intrusive_ptr<LinearPackedParamsBase>>>;

// 基类，使我们可以多态地处理这些类型
struct CellParamsBase : torch::CustomClassHolder {
  // 纯虚函数，需要在具体的 CellParams 类中实现
  virtual Tensor matmul_ih(const Tensor& input) const = 0;
  virtual Tensor matmul_hh(const Tensor& h) const = 0;
  // 默认不做任何操作。CellParams 将会重写这个函数，以定义带有投影的 LSTM 的正确行为。
  // 这个函数不是纯虚函数，因为为不支持投影的所有 cell params 提供默认实现是有用的（例如 QuantizedCellParams 的变体）
  virtual Tensor matmul_hr(const Tensor& h) const {
    return h;
  }
  // 纯虚函数，需要在具体的 CellParams 类中实现
  virtual Tensor linear_ih(const Tensor& input_ih) const = 0;
  virtual Tensor linear_hh(const Tensor& input_hh) const = 0;

  // 纯虚函数，需要在具体的 CellParams 类中实现
  virtual const Tensor& b_ih() const = 0;
  virtual const Tensor& b_hh() const = 0;

  // 纯虚函数，需要在具体的 CellParams 类中实现
  virtual CellParamsSerializationType __getstate__() const = 0;
};

// 几乎所有我们支持的 cell 类都使用相同的参数集，但手动传递这些 4 个参数非常麻烦。它们的生命周期由外部管理，因此我们只需
struct CellParams : public CellParamsBase {
  // 定义一个结构体 CellParams，继承自 CellParamsBase
  CellParams(
      const Tensor& _w_ih,
      const Tensor& _w_hh,
      const Tensor& _b_ih,
      const Tensor& _b_hh,
      const Tensor& _w_hr)
      : w_ih(_w_ih), w_hh(_w_hh), b_ih_(_b_ih), b_hh_(_b_hh), w_hr(_w_hr) {};
      // CellParams 的构造函数，接受五个参数分别初始化 w_ih, w_hh, b_ih_, b_hh_, w_hr

  const Tensor& w_ih;   // 声明一个常量引用 Tensor 类型的成员变量 w_ih
  const Tensor& w_hh;   // 声明一个常量引用 Tensor 类型的成员变量 w_hh
  const Tensor& b_ih_;  // 声明一个常量引用 Tensor 类型的成员变量 b_ih_
  const Tensor& b_hh_;  // 声明一个常量引用 Tensor 类型的成员变量 b_hh_
  const Tensor& w_hr;   // 声明一个常量引用 Tensor 类型的成员变量 w_hr

  Tensor matmul_ih(const Tensor& input) const override {
    return at::matmul(input, w_ih.t());
    // 实现 matmul_ih 函数，返回输入 input 与 w_ih 转置矩阵相乘的结果
  }
  Tensor matmul_hh(const Tensor& h) const override {
    return at::matmul(h, w_hh.t());
    // 实现 matmul_hh 函数，返回输入 h 与 w_hh 转置矩阵相乘的结果
  }
  Tensor matmul_hr(const Tensor& h) const override {
    if (w_hr.defined()) {
      return at::matmul(h, w_hr.t());
      // 实现 matmul_hr 函数，如果 w_hr 已定义，则返回输入 h 与 w_hr 转置矩阵相乘的结果
    }
    return h;
    // 如果 w_hr 未定义，则直接返回输入 h
  }
  Tensor linear_ih(const Tensor& input) const override {
    return at::linear(input, w_ih, b_ih_);
    // 实现 linear_ih 函数，返回输入 input 与 w_ih 和 b_ih_ 线性组合的结果
  }
  Tensor linear_hh(const Tensor& h) const override {
    return at::linear(h, w_hh, b_hh_);
    // 实现 linear_hh 函数，返回输入 h 与 w_hh 和 b_hh_ 线性组合的结果
  }
  const Tensor& b_ih() const override {
    return b_ih_;
    // 实现 b_ih 函数，返回成员变量 b_ih_ 的常量引用
  }
  const Tensor& b_hh() const override {
    return b_hh_;
    // 实现 b_hh 函数，返回成员变量 b_hh_ 的常量引用
  }
  CellParamsSerializationType __getstate__() const override {
    TORCH_INTERNAL_ASSERT(false, "Not yet implemented");
    // 实现 __getstate__ 函数，暂时抛出错误信息
  }
  static c10::intrusive_ptr<CellParamsBase> __setstate__(
      CellParamsSerializationType state) {
    TORCH_INTERNAL_ASSERT(false, "Not yet implemented");
    // 实现静态 __setstate__ 函数，暂时抛出错误信息
  }
};

c10::intrusive_ptr<CellParamsBase> make_quantized_cell_params(
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    at::Tensor bias_ih,
    at::Tensor bias_hh);

struct QuantizedCellParams : public CellParamsBase {
  // 定义一个结构体 QuantizedCellParams，继承自 CellParamsBase
  QuantizedCellParams(
      Tensor _w_ih,
      Tensor _w_hh,
      Tensor _b_ih,
      Tensor _b_hh,
      Tensor _packed_ih,
      Tensor _packed_hh,
      Tensor _col_offsets_ih,
      Tensor _col_offsets_hh,
      Scalar _scale_ih,
      Scalar _scale_hh,
      Scalar _zero_point_ih,
      Scalar _zero_point_hh)
      : w_ih(std::move(_w_ih)),
        w_hh(std::move(_w_hh)),
        b_ih_(std::move(_b_ih)),
        b_hh_(std::move(_b_hh)),
        packed_ih(std::move(_packed_ih)),
        packed_hh(std::move(_packed_hh)),
        col_offsets_ih(std::move(_col_offsets_ih)),
        col_offsets_hh(std::move(_col_offsets_hh)),
        scale_ih(std::move(_scale_ih)),
        scale_hh(std::move(_scale_hh)),
        zero_point_ih(std::move(_zero_point_ih)),
        zero_point_hh(std::move(_zero_point_hh)) {}
        // QuantizedCellParams 的构造函数，接受多个参数进行初始化

  const Tensor w_ih;          // 声明一个常量 Tensor 类型的成员变量 w_ih
  const Tensor w_hh;          // 声明一个常量 Tensor 类型的成员变量 w_hh
  const Tensor b_ih_;         // 声明一个常量 Tensor 类型的成员变量 b_ih_
  const Tensor b_hh_;         // 声明一个常量 Tensor 类型的成员变量 b_hh_
  const Tensor packed_ih;     // 声明一个常量 Tensor 类型的成员变量 packed_ih
  const Tensor packed_hh;     // 声明一个常量 Tensor 类型的成员变量 packed_hh
  const Tensor col_offsets_ih;  // 声明一个常量 Tensor 类型的成员变量 col_offsets_ih
  const Tensor col_offsets_hh;  // 声明一个常量 Tensor 类型的成员变量 col_offsets_hh
  const Scalar scale_ih;      // 声明一个常量 Scalar 类型的成员变量 scale_ih
  const Scalar scale_hh;      // 声明一个常量 Scalar 类型的成员变量 scale_hh
  const Scalar zero_point_ih; // 声明一个常量 Scalar 类型的成员变量 zero_point_ih
  const Scalar zero_point_hh; // 声明一个常量 Scalar 类型的成员变量 zero_point_hh

  Tensor matmul_ih(const Tensor& input) const override {
  TORCH_CHECK(false, "matmul is not supported with quantized cell params");
  # 使用 TORCH_CHECK 来确保条件为 false，否则抛出错误，指出 matmul 操作不支持量化的单元参数

  Tensor matmul_hh(const Tensor& h) const override {
    TORCH_CHECK(false, "matmul is not supported with quantized cell params");
    # 使用 TORCH_CHECK 来确保条件为 false，否则抛出错误，指出 matmul 操作不支持量化的单元参数
  }

  Tensor linear_ih(const Tensor& input) const override {
    return at::fbgemm_linear_int8_weight_fp32_activation(
        input, w_ih, packed_ih, col_offsets_ih, scale_ih, zero_point_ih, b_ih_);
    # 调用 fbgemm_linear_int8_weight_fp32_activation 函数进行整数权重、浮点数激活的线性运算，返回结果张量
  }

  Tensor linear_hh(const Tensor& h) const override {
    return at::fbgemm_linear_int8_weight_fp32_activation(
        h, w_hh, packed_hh, col_offsets_hh, scale_hh, zero_point_hh, b_hh_);
    # 调用 fbgemm_linear_int8_weight_fp32_activation 函数进行整数权重、浮点数激活的线性运算，返回结果张量
  }

  const Tensor& b_ih() const override {
    return b_ih_;
    # 返回成员变量 b_ih_，作为实现接口中 b_ih() 方法的返回值
  }

  const Tensor& b_hh() const override {
    return b_hh_;
    # 返回成员变量 b_hh_，作为实现接口中 b_hh() 方法的返回值
  }

  CellParamsSerializationType __getstate__() const override {
    # 获取当前对象的状态，用于序列化
    std::vector<at::Tensor> tensors_to_serialize = {
        w_ih, w_hh, b_ih_, b_hh_, col_offsets_ih, col_offsets_hh};
    std::vector<double> doubles_to_serialize = {scale_ih.toDouble(),
                                                scale_hh.toDouble()};
    std::vector<int64_t> longs_to_serialize = {zero_point_ih.toLong(),
                                               zero_point_hh.toLong()};
    # 返回对象的序列化类型 CellParamsSerializationType
    return CellParamsSerializationType(
        "quantized",
        std::move(tensors_to_serialize),
        std::move(doubles_to_serialize),
        std::move(longs_to_serialize),
        {});
  }

  static c10::intrusive_ptr<CellParamsBase> __setstate__(
      CellParamsSerializationType state) {
    # 静态方法，用于从序列化状态中恢复对象的状态
    auto [_, tensors, doubles, longs, __] =
        std::move(state);
    TORCH_INTERNAL_ASSERT(tensors.size() == 6);
    TORCH_INTERNAL_ASSERT(doubles.size() == 2);
    TORCH_INTERNAL_ASSERT(longs.size() == 2);

    # 解包各种类型的数据
    at::Tensor qw_ih = std::move(tensors[0]), qw_hh = std::move(tensors[1]),
               b_ih = std::move(tensors[2]), b_hh = std::move(tensors[3]),
               col_offsets_ih = std::move(tensors[4]),
               col_offsets_hh = std::move(tensors[5]);
    double scale_ih = doubles[0], scale_hh = doubles[1];
    int64_t zero_point_ih = longs[0], zero_point_hh = longs[1];

    # 使用 fbgemm_pack_quantized_matrix 函数对权重进行打包
    at::Tensor packed_ih = at::native::fbgemm_pack_quantized_matrix(qw_ih);
    at::Tensor packed_hh = at::native::fbgemm_pack_quantized_matrix(qw_hh);

    # 返回新创建的 QuantizedCellParams 对象，用于从序列化状态中恢复
    return c10::make_intrusive<QuantizedCellParams>(
        /*w_ih=*/std::move(qw_ih),
        /*w_hh=*/std::move(qw_hh),
        /*b_ih_=*/std::move(b_ih),
        /*b_hh_=*/std::move(b_hh),
        /*packed_ih=*/std::move(packed_ih),
        /*packed_hh=*/std::move(packed_hh),
        /*col_offsets_ih=*/std::move(col_offsets_ih),
        /*col_offsets_hh=*/std::move(col_offsets_hh),
        /*scale_ih=*/scale_ih,
        /*scale_hh=*/scale_hh,
        /*zero_point_ih=*/zero_point_ih,
        /*zero_point_hh=*/zero_point_hh);
  }
};

// 创建量化的循环神经网络（RNN）单元参数，使用硬件量化（fbgemm）
c10::intrusive_ptr<CellParamsBase> make_quantized_cell_params(
    const at::Tensor& w_ih,               // 输入到隐藏层权重张量
    const at::Tensor& w_hh,               // 隐藏层到隐藏层权重张量
    at::Tensor b_ih,                      // 输入到隐藏层偏置张量
    at::Tensor b_hh) {                    // 隐藏层到隐藏层偏置张量
  auto make_vals = [&](const at::Tensor& W) {
    auto params = at::native::fbgemm_linear_quantize_weight(W);
    at::Tensor packed_weight =
        at::native::fbgemm_pack_quantized_matrix(std::get<0>(params));
    return std::tuple_cat(
        std::make_tuple(std::move(packed_weight)), std::move(params));
  };

  // 获取输入到隐藏层和隐藏层到隐藏层的量化参数
  auto [packed_ih, qw_ih, col_offsets_ih, scale_ih, zero_point_ih] =
      make_vals(w_ih);
  auto [packed_hh, qw_hh, col_offsets_hh, scale_hh, zero_point_hh] =
      make_vals(w_hh);

  // 创建并返回量化的RNN单元参数对象
  return c10::make_intrusive<QuantizedCellParams>(
      /*qw_ih=*/std::move(qw_ih),         // 输入到隐藏层权重量化参数
      /*qw_hh=*/std::move(qw_hh),         // 隐藏层到隐藏层权重量化参数
      /*b_ih=*/std::move(b_ih),           // 输入到隐藏层偏置
      /*b_hh=*/std::move(b_hh),           // 隐藏层到隐藏层偏置
      /*packed_ih=*/std::move(packed_ih), // 输入到隐藏层打包的量化权重
      /*packed_hh=*/std::move(packed_hh), // 隐藏层到隐藏层打包的量化权重
      /*col_offsets_ih=*/std::move(col_offsets_ih),   // 输入到隐藏层列偏移
      /*col_offsets_hh=*/std::move(col_offsets_hh),   // 隐藏层到隐藏层列偏移
      /*scale_ih=*/std::move(scale_ih),   // 输入到隐藏层缩放因子
      /*scale_hh=*/std::move(scale_hh),   // 隐藏层到隐藏层缩放因子
      /*zero_point_ih=*/std::move(zero_point_ih),     // 输入到隐藏层零点
      /*zero_point_hh=*/std::move(zero_point_hh));    // 隐藏层到隐藏层零点
}

// QuantizedCellParamsDynamic 与 QuantizedCellParams 的比较
//
// QuantizedCellParams 使用传统的 fbgemm_linear_int8_weight_fp32_activation
// API，需要权重的显式缩放和零点参数。QuantizedCellParamsDynamic 使用新的
// fbgemm_linear_dynamic API，不需要显式的缩放和零点参数。这些量化参数
// 封装在 aten/src/ATen/native/quantized/cpu/fbgemm_utils.h 中的
// `PackedLinearWeight` 结构体中。

// 创建动态量化的循环神经网络（RNN）单元参数
c10::intrusive_ptr<CellParamsBase> make_quantized_cell_params_dynamic(
    c10::intrusive_ptr<LinearPackedParamsBase> w_ih_packed,   // 输入到隐藏层打包的线性参数基类
    c10::intrusive_ptr<LinearPackedParamsBase> w_hh_packed,   // 隐藏层到隐藏层打包的线性参数基类
    at::Tensor bias_ih,               // 输入到隐藏层偏置张量
    at::Tensor bias_hh,               // 隐藏层到隐藏层偏置张量
    bool reduce_range);               // 是否减少范围的标志位

// 表示动态量化的RNN单元参数对象，继承自CellParamsBase类
struct QuantizedCellParamsDynamic : public CellParamsBase {
  QuantizedCellParamsDynamic(
      c10::intrusive_ptr<LinearPackedParamsBase>
          _packed_w_ih,               // 预打包的输入到隐藏层权重张量
      c10::intrusive_ptr<LinearPackedParamsBase>
          _packed_w_hh,               // 预打包的隐藏层到隐藏层权重张量
      Tensor _b_ih,                   // 浮点类型的输入到隐藏层偏置张量
      Tensor _b_hh,                   // 浮点类型的隐藏层到隐藏层偏置张量
      bool _reduce_range = false      // 是否对激活张量使用减少范围的设置，默认为否
      )
      : packed_w_ih(std::move(_packed_w_ih)),   // 移动赋值输入到隐藏层打包的权重
        packed_w_hh(std::move(_packed_w_hh)),   // 移动赋值隐藏层到隐藏层打包的权重
        b_ih_(std::move(_b_ih)),                // 移动赋值输入到隐藏层偏置
        b_hh_(std::move(_b_hh)),                // 移动赋值隐藏层到隐藏层偏置
        reduce_range_(_reduce_range) {}        // 初始化是否减少范围的设置

  c10::intrusive_ptr<LinearPackedParamsBase> packed_w_ih;   // 输入到隐藏层打包的线性参数基类
  c10::intrusive_ptr<LinearPackedParamsBase> packed_w_hh;   // 隐藏层到隐藏层打包的线性参数基类
  const Tensor b_ih_;               // 输入到隐藏层的偏置张量
  const Tensor b_hh_;               // 隐藏层到隐藏层的偏置张量
  bool reduce_range_;               // 是否减少范围的标志位

  // 执行输入与隐藏层权重矩阵乘积的函数
  Tensor matmul_ih(const Tensor& input) const override {
    // 检查条件，如果为 false，抛出错误信息，指示 matmul 不支持量化的单元参数
    TORCH_CHECK(false, "matmul is not supported with quantized cell params");
  }
  // 返回经过覆盖的 matmul_hh 函数，同样抛出错误信息，指示不支持量化的单元参数
  Tensor matmul_hh(const Tensor& h) const override {
    TORCH_CHECK(false, "matmul is not supported with quantized cell params");
  }

  // 返回线性变换的结果，应用于输入的 input_ih
  Tensor linear_ih(const Tensor& input_ih) const override {
    return packed_w_ih->apply_dynamic(input_ih, reduce_range_);
  }
  // 返回线性变换的结果，应用于输入的 input_hh
  Tensor linear_hh(const Tensor& input_hh) const override {
    return packed_w_hh->apply_dynamic(input_hh, reduce_range_);
  }

  // 返回输入的偏置 b_ih_
  const Tensor& b_ih() const override {
    return b_ih_;
  }
  // 返回隐藏状态的偏置 b_hh_
  const Tensor& b_hh() const override {
    return b_hh_;
  }

  // 返回对象的状态以进行序列化，包括偏置张量 b_ih_ 和 b_hh_，但不包括其他字段
  CellParamsSerializationType __getstate__() const override {
    std::vector<at::Tensor> tensors_to_serialize{
        /*b_ih=*/b_ih_,
        /*b_hh=*/b_hh_,
    };

    std::vector<c10::intrusive_ptr<LinearPackedParamsBase>>
        packed_params_to_serialize{packed_w_ih, packed_w_hh};

    // reduce_range 参数与整数字段值一起进行序列化
    return CellParamsSerializationType(
        "quantized_dynamic",
        std::move(tensors_to_serialize),
        {},  // 空的字符串列表，因为不需要额外的字符串字段
        {reduce_range_},  // 序列化 reduce_range_ 参数
        std::move(packed_params_to_serialize));
  }

  // 从给定状态中恢复对象的状态
  static c10::intrusive_ptr<CellParamsBase> __setstate__(
      CellParamsSerializationType state) {
    auto [_, tensors, __, serialized_ints, packed_params] =
        std::move(state);
    TORCH_INTERNAL_ASSERT(tensors.size() == 2);
    TORCH_INTERNAL_ASSERT(packed_params.size() == 2);

    // 检查 serialized_ints 是否为空，若为空则 reduce_range 设为 false
    bool reduce_range = serialized_ints.empty() ? false : serialized_ints[0];

    // 使用给定的参数重新创建一个 quantized_dynamic 的单元参数对象
    return make_quantized_cell_params_dynamic(
        /*w_ih_packed=*/std::move(packed_params[0]),
        /*w_hh_packed=*/std::move(packed_params[1]),
        /*bias_ih=*/std::move(tensors[0]),
        /*bias_hh=*/std::move(tensors[1]),
        /*reduce_range=*/reduce_range);
  }
};

// 创建一个动态量化的单元参数对象，使用给定的权重和偏置参数，以及是否减少范围的标志
c10::intrusive_ptr<CellParamsBase> make_quantized_cell_params_dynamic(
    c10::intrusive_ptr<LinearPackedParamsBase> w_ih_packed,
    c10::intrusive_ptr<LinearPackedParamsBase> w_hh_packed,
    at::Tensor bias_ih,
    at::Tensor bias_hh,
    bool reduce_range) {

  // 使用给定的参数创建一个QuantizedCellParamsDynamic对象并返回
  return c10::make_intrusive<QuantizedCellParamsDynamic>(
      /*_packed_w_ih=*/std::move(w_ih_packed),
      /*_packed_w_hh=*/std::move(w_hh_packed),
      /*_b_ih=*/std::move(bias_ih),
      /*_b_hh=*/std::move(bias_hh),
      /*_reduce_range=*/reduce_range);
}

// 创建一个半精度浮点数动态量化的单元参数对象，使用给定的权重参数
c10::intrusive_ptr<CellParamsBase> make_quantized_cell_params_fp16(
    c10::intrusive_ptr<LinearPackedParamsBase> w_ih_packed,
    c10::intrusive_ptr<LinearPackedParamsBase> w_hh_packed);

// 表示半精度浮点数动态量化的单元参数对象
struct QuantizedCellParamsFP16 : public CellParamsBase {
  QuantizedCellParamsFP16(
      c10::intrusive_ptr<LinearPackedParamsBase> _packed_ih,
      c10::intrusive_ptr<LinearPackedParamsBase> _packed_hh)
      : packed_ih(std::move(_packed_ih)), packed_hh(std::move(_packed_hh)) {}

  c10::intrusive_ptr<LinearPackedParamsBase> packed_ih;
  c10::intrusive_ptr<LinearPackedParamsBase> packed_hh;
  const Tensor b_ih_;
  const Tensor b_hh_;

  // 覆盖父类方法，执行输入矩阵乘法，抛出异常，表示不支持此操作
  Tensor matmul_ih(const Tensor& /* unused */) const override {
    TORCH_CHECK(false, "matmul is not supported with quantized cell params");
  }
  // 覆盖父类方法，执行隐藏状态矩阵乘法，抛出异常，表示不支持此操作
  Tensor matmul_hh(const Tensor& /* unused */) const override {
    TORCH_CHECK(false, "matmul is not supported with quantized cell params");
  }
  // 覆盖父类方法，执行输入的线性变换操作
  Tensor linear_ih(const Tensor& input) const override {
    return packed_ih->apply_dynamic(input);
  }
  // 覆盖父类方法，执行隐藏状态的线性变换操作
  Tensor linear_hh(const Tensor& h) const override {
    return packed_hh->apply_dynamic(h);
  }

  // 覆盖父类方法，返回输入偏置值
  const Tensor& b_ih() const override {
    return b_ih_;
  }
  // 覆盖父类方法，返回隐藏状态偏置值
  const Tensor& b_hh() const override {
    return b_hh_;
  }

  // 覆盖父类方法，序列化单元参数对象的状态
  CellParamsSerializationType __getstate__() const override {
    std::vector<c10::intrusive_ptr<LinearPackedParamsBase>>
        packed_params_to_serialize{packed_ih, packed_hh};

    return CellParamsSerializationType(
        "quantized_fp16", {}, {}, {}, std::move(packed_params_to_serialize));
  }

  // 静态方法，从序列化状态中恢复对象，返回半精度浮点数动态量化的单元参数对象
  static c10::intrusive_ptr<CellParamsBase> __setstate__(
      CellParamsSerializationType state) {
    auto packed_params = std::get<4>(std::move(state));
    TORCH_INTERNAL_ASSERT(packed_params.size() == 2);
    return make_quantized_cell_params_fp16(
        /*w_ih_packed=*/std::move(packed_params[0]),
        /*w_hh_packed=*/std::move(packed_params[1]));
  }
};

// 创建一个半精度浮点数动态量化的单元参数对象，使用给定的权重参数
c10::intrusive_ptr<CellParamsBase> make_quantized_cell_params_fp16(
    c10::intrusive_ptr<LinearPackedParamsBase> w_ih_packed,
    c10::intrusive_ptr<LinearPackedParamsBase> w_hh_packed) {
  // 返回一个QuantizedCellParamsFP16对象
  return c10::make_intrusive<QuantizedCellParamsFP16>(
      std::move(w_ih_packed), std::move(w_hh_packed));
}

// 静态映射表，将字符串映射到对应的创建单元参数对象的函数指针
static std::unordered_map<
    std::string,
    c10::intrusive_ptr<CellParamsBase> (*)(CellParamsSerializationType)>
    # 定义一个字典，用于存储不同类型的单元参数反序列化函数
    cell_params_deserializers = {
        # 键："quantized"，值：指向 QuantizedCellParams 类的 __setstate__ 方法的引用
        {"quantized", &QuantizedCellParams::__setstate__},
        # 键："quantized_dynamic"，值：指向 QuantizedCellParamsDynamic 类的 __setstate__ 方法的引用
        {"quantized_dynamic", &QuantizedCellParamsDynamic::__setstate__},
        # 键："quantized_fp16"，值：指向 QuantizedCellParamsFP16 类的 __setstate__ 方法的引用
        {"quantized_fp16", &QuantizedCellParamsFP16::__setstate__}
    };
// Stupid wrapper to convert from -> to .
// 定义一个简单的包装器，用于将参数转换为一个结构体
struct QRNNCellParamsWrapper {
  // 构造函数，接受一个 CellParamsBase 类型的智能指针作为参数并保存
  QRNNCellParamsWrapper(c10::intrusive_ptr<CellParamsBase> param)
      : param_(std::move(param)) {}

  // 调用 param_ 的 matmul_ih 方法，执行输入张量 input 的矩阵乘法
  Tensor matmul_ih(const Tensor& input) const {
    return param_->matmul_ih(input);
  }
  // 调用 param_ 的 matmul_hh 方法，执行隐藏状态张量 h 的矩阵乘法
  Tensor matmul_hh(const Tensor& h) const {
    return param_->matmul_hh(h);
  }
  // 调用 param_ 的 matmul_hr 方法，执行隐藏状态张量 h 的矩阵乘法
  Tensor matmul_hr(const Tensor& h) const {
    return param_->matmul_hr(h);
  }
  // 调用 param_ 的 linear_ih 方法，执行输入张量 input 的线性变换
  Tensor linear_ih(const Tensor& input) const {
    return param_->linear_ih(input);
  }
  // 调用 param_ 的 linear_hh 方法，执行隐藏状态张量 h 的线性变换
  Tensor linear_hh(const Tensor& h) const {
    return param_->linear_hh(h);
  }
  // 返回 param_ 的 b_ih 成员变量，即输入张量的偏置
  const Tensor& b_ih() const {
    return param_->b_ih();
  }
  // 返回 param_ 的 b_hh 成员变量，即隐藏状态张量的偏置
  const Tensor& b_hh() const {
    return param_->b_hh();
  }

  // 成员变量，保存 CellParamsBase 类型的智能指针
  c10::intrusive_ptr<CellParamsBase> param_;
};

// Gathers every two elements of a vector in a vector of pairs
// 将一个元素类型为 T 的向量中的每两个元素组成一对，存入一个 pair_of<T> 类型的向量中
template<typename T>
static std::vector<pair_of<T>> pair_vec(const std::vector<T>& vals) {
  // 检查输入向量 vals 的大小是否为偶数，否则抛出异常
  TORCH_CHECK(vals.size() % 2 == 0, "Odd number of params or hiddens given to a bidirectional RNN");
  // 创建用于存放结果的 vector
  std::vector<pair_of<T>> result;
  // 预留足够的空间，避免动态扩展
  result.reserve(vals.size() / 2);
  // 遍历 vals，每两个元素组成一对，加入 result
  for (size_t i = 0; i < vals.size(); i += 2) {
    result.emplace_back(vals[i], vals[i + 1]);
  }
  // 返回结果向量 result
  return result;
}

// Flattens a vector of pairs
// 将一组成对的元素向量展平为单个元素向量
template<typename T>
static std::vector<T> unpair_vec(std::vector<pair_of<T>>&& vals) {
  // 创建用于存放结果的 vector
  std::vector<T> result;
  // 预留足够的空间，避免动态扩展
  result.reserve(vals.size() * 2);
  // 遍历 vals，将每一对元素展开为单个元素，加入 result
  for (const auto i : c10::irange(vals.size())) {
    result.push_back(std::move(vals[i].first));
    result.push_back(std::move(vals[i].second));
  }
  // 返回结果向量 result
  return result;
}

// Parses a flat list of parameter tensors into a list of CellParams
// 将一组扁平的参数张量解析为一组 CellParams 对象
static std::vector<CellParams> gather_params(TensorList params, bool has_biases, bool has_projections = false) {
  // 定义一个静态的未定义 Tensor
  static at::Tensor undefined;
  // 创建用于存放结果的 vector
  std::vector<CellParams> result;
  // 如果存在偏置
  if (has_biases) {
    // 如果存在投影
    if (has_projections) {
      // 检查 params 的大小是否为 5 的倍数，否则抛出异常
      TORCH_CHECK(params.size() % 5 == 0, "got an incorrect number of RNN parameters");
      // 遍历 params，每五个元素构成一个 CellParams 对象，加入 result
      for (size_t i = 0; i < params.size(); i += 5) {
        result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], params[i + 4]);
      }
    } else {
      // 检查 params 的大小是否为 4 的倍数，否则抛出异常
      TORCH_CHECK(params.size() % 4 == 0, "got an incorrect number of RNN parameters");
      // 遍历 params，每四个元素构成一个 CellParams 对象，加入 result
      for (size_t i = 0; i < params.size(); i += 4) {
        result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], undefined);
      }
    }
  } else {
    // 如果不存在偏置
    if (has_projections) {
      // 检查 params 的大小是否为 3 的倍数，否则抛出异常
      TORCH_CHECK(params.size() % 3 == 0, "got an incorrect number of RNN parameters");
      // 遍历 params，每三个元素构成一个 CellParams 对象，加入 result
      for (size_t i = 0; i < params.size(); i += 3) {
        result.emplace_back(params[i], params[i + 1], undefined, undefined, params[i + 2]);
      }
    } else {
      // 检查 params 的大小是否为 2 的倍数，否则抛出异常
      TORCH_CHECK(params.size() % 2 == 0, "got an incorrect number of RNN parameters");
      // 遍历 params，每两个元素构成一个 CellParams 对象，加入 result
      for (size_t i = 0; i < params.size(); i += 2) {
        result.emplace_back(params[i], params[i + 1], undefined, undefined, undefined);
      }
    }
  }
  // 返回结果向量 result
  return result;
}
////////////////////////////////////////////////////////////////////////////////
// HIDDEN STATE FUNCTIONS
//
// Functions implemented below are implemented as templates based on hidden type,
// because they need to work both with simple RNNs and GRU (which use a single Tensor),
// as well as with LSTM (or possibly more complicated architectures in the future).
// Still, there are some operations that need to be performed on the hidden states
// alone, and for this purpose we provide an overloaded set of functions below.

// 将 Tensor 直接作为输出返回
Tensor hidden_as_output(const Tensor& t) { return t; }

// 从 tpair_of<Tensor> 中提取第一个 Tensor 作为输出返回
Tensor hidden_as_output(const tpair_of<Tensor>& t) { return std::get<0>(t); }

// 以索引 index 作为模板参数，从 tuples 中的每个 tpair_of<Tensor> 中提取第 index 个 Tensor，
// 组成一个 Tensor 数组返回
template<size_t index>
std::vector<Tensor> project(at::ArrayRef<tpair_of<Tensor>> tuples) {
  std::vector<Tensor> result;
  result.reserve(tuples.size());
  for (auto & t : tuples) {
    result.push_back(std::get<index>(t));
  }
  return result;
}

// 将一个 Tensor 数组 hiddens 沿着维度 0 进行拼接，返回拼接后的 Tensor
Tensor hidden_concat(at::ArrayRef<Tensor> hiddens) { return at::cat(hiddens, 0); }

// 将一个 tpair_of<Tensor> 数组 hiddens 中的每个 tpair_of<Tensor> 中的第 0 个和第 1 个 Tensor 分别拼接，
// 返回拼接后的 tpair_of<Tensor>
tpair_of<Tensor> hidden_concat(at::ArrayRef<tpair_of<Tensor>> hiddens) {
  return std::make_tuple(hidden_concat(project<0>(hiddens)), hidden_concat(project<1>(hiddens)));
}

// 对给定的 Tensor t 进行切片操作，从 start 开始，到 end 结束（不包含 end），返回切片后的 Tensor
Tensor hidden_slice(const Tensor& t, int64_t start, int64_t end) {
  return t.narrow(0, start, end - start);
}

// 对给定的 tpair_of<Tensor> t 进行切片操作，分别对第 0 个和第 1 个 Tensor 进行切片，
// 返回切片后的 tpair_of<Tensor>
tpair_of<Tensor> hidden_slice(const tpair_of<Tensor>& t, int64_t start, int64_t end) {
  return std::make_tuple(hidden_slice(std::get<0>(t), start, end),
                         hidden_slice(std::get<1>(t), start, end));
}

////////////////////////////////////////////////////////////////////////////////
// CELL IMPLEMENTATIONS
//
// Cell is a basic component of an RNN, representing a single application of the
// recurrent function. You can think of it as a function of signature
//
// (Tensor input, hidden_type hidden, CellParams) -> hidden_type
//
// which means that it consumes an input tensor, and updates the previous hidden state.
// It's a struct only because functional programming in C++ is a pain, and it's easier
// to pass around "vtable pointers" than actual function pointers.

// 检查 RNN cell 前向传播的输入，确保输入张量的符号大小与指定的输入大小一致
void check_rnn_cell_forward_input(const Tensor& input, c10::SymInt input_size) {
  TORCH_CHECK(
    input.sym_size(1) == input_size,
    "input has inconsistent input_size: got ", input.sym_size(1), " expected ", input_size);
}

// 检查 RNN cell 前向传播的隐藏状态，确保输入张量的批量大小与隐藏状态的批量大小一致，
// 并确保隐藏状态的大小与指定的隐藏大小一致
void check_rnn_cell_forward_hidden(const Tensor& input, const Tensor& hx, c10::SymInt hidden_size, c10::SymInt hidden_label) {
  TORCH_CHECK(
    input.sym_size(0) == hx.sym_size(0),
    "Input batch size ", input.sym_size(0), " doesn't match hidden", hidden_label, " batch size ", hx.sym_size(0));

  TORCH_CHECK(
    hx.sym_size(1) == hidden_size,
    "hidden", hidden_label, " has inconsistent hidden_size: got ", hx.sym_size(1), ", expected ", hidden_size);
}
// 定义一个结构体 Cell
struct Cell {
  // 使用别名 hidden_type 和 cell_params_tmpl 的模板参数
  using hidden_type = hidden_type_tmpl;
  using cell_params = cell_params_tmpl;

  // 默认虚析构函数，解决 -Wnon-virtual-dtor 问题
  virtual ~Cell() = default; // This is really dumb, but enables projects with
                             // -Wnon-virtual-dtor to compile...

  // 纯虚函数，子类必须实现，定义了 Cell 的操作符
  virtual hidden_type operator()(
      const Tensor& input,
      const hidden_type& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const = 0;
};

// 模板结构体 SimpleCell，继承自 Cell<Tensor, cell_params>
template<typename nonlinearity, typename cell_params>
struct SimpleCell : Cell<Tensor, cell_params> {
  // 使用别名 hidden_type 为 Tensor
  using hidden_type = Tensor;
  
  // 实现 Cell 的操作符
  Tensor operator()(
      const Tensor& input,
      const Tensor& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const override {
    // 返回非线性操作的结果，用于 SimpleCell
    return nonlinearity{}(params.linear_hh(hidden).add_(
        pre_compute_input ? input : params.linear_ih(input)));
  }
};

// 模板结构体 LSTMCell，继承自 Cell<std::tuple<Tensor, Tensor>, cell_params>
template <typename cell_params>
struct LSTMCell : Cell<std::tuple<Tensor, Tensor>, cell_params> {
  // 使用别名 hidden_type 为 std::tuple<Tensor, Tensor>
  using hidden_type = std::tuple<Tensor, Tensor>;

  // 实现 Cell 的操作符
  hidden_type operator()(
      const Tensor& input,
      const hidden_type& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const override {
    // 解构 hidden 为 hx 和 cx
    const auto& hx = std::get<0>(hidden);
    const auto& cx = std::get<1>(hidden);

    // 如果 input 在 CUDA 上或者是私有使用
    if (input.is_cuda() || input.is_privateuseone()) {
      // 检查不应使用预计算输入
      TORCH_CHECK(!pre_compute_input);
      // 计算输入门和隐藏门
      auto igates = params.matmul_ih(input);
      auto hgates = params.matmul_hh(hx);
      // 调用 fused LSTM 单元计算
      auto result = at::_thnn_fused_lstm_cell(
          igates, hgates, cx, params.b_ih(), params.b_hh());
      // 如果定义了 w_hr，应用投影
      auto hy = params.matmul_hr(std::get<0>(result));
      // 切掉工作空间参数（仅用于 AD）
      return std::make_tuple(std::move(hy), std::move(std::get<1>(result)));
    }

    // 计算 LSTM 单元的四个门控信号
    const auto gates = params.linear_hh(hx).add_(
        pre_compute_input ? input : params.linear_ih(input));
    auto chunked_gates = gates.unsafe_chunk(4, 1);
    auto ingate = chunked_gates[0].sigmoid_();
    auto forgetgate = chunked_gates[1].sigmoid_();
    auto cellgate = chunked_gates[2].tanh_();
    auto outgate = chunked_gates[3].sigmoid_();
    // 更新记忆单元 cy 和隐藏状态 hy
    auto cy = (forgetgate * cx).add_(ingate * cellgate);
    auto hy = outgate * cy.tanh();
    // 应用投影变换
    hy = params.matmul_hr(hy);
    return std::make_tuple(std::move(hy), std::move(cy));
  }
};

// 模板结构体 GRUCell，继承自 Cell<Tensor, cell_params>
template <typename cell_params>
struct GRUCell : Cell<Tensor, cell_params> {
  // 使用别名 hidden_type 为 Tensor
  using hidden_type = Tensor;

  // 实现 Cell 的操作符
  hidden_type operator()(
      const Tensor& input,
      const hidden_type& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const override {
    // 检查输入是否在 CUDA、XPU 或私有使用环境中，至少满足其中一种条件
    if (input.is_cuda() || input.is_xpu() || input.is_privateuseone()) {
      // 断言不应预先计算输入
      TORCH_CHECK(!pre_compute_input);
      // 计算输入门 igates 和隐藏状态门 hgates
      auto igates = params.matmul_ih(input);
      auto hgates = params.matmul_hh(hidden);
      // 调用 THNN 库中的融合 GRU 单元计算
      auto result = at::_thnn_fused_gru_cell(
          igates, hgates, hidden, params.b_ih(), params.b_hh());
      // 返回计算结果的第一个元素，移动语义传递所有权
      // 切除用于自动微分的工作空间参数
      return std::move(std::get<0>(result));
    }
    // 如果不是 CUDA、XPU 或私有使用环境，则继续执行以下逻辑
    // 根据预计算输入的设置，切分输入数据为三个块
    const auto chunked_igates = pre_compute_input
        ? input.unsafe_chunk(3, 1)
        : params.linear_ih(input).unsafe_chunk(3, 1);
    // 使用线性变换得到隐藏状态的门的块
    auto chunked_hgates = params.linear_hh(hidden).unsafe_chunk(3, 1);
    // 计算重置门
    const auto reset_gate =
        chunked_hgates[0].add_(chunked_igates[0]).sigmoid_();
    // 计算输入门
    const auto input_gate =
        chunked_hgates[1].add_(chunked_igates[1]).sigmoid_();
    // 计算新的门
    const auto new_gate =
        chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate)).tanh_();
    // 返回更新后的隐藏状态
    return (hidden - new_gate).mul_(input_gate).add_(new_gate);
  }
////////////////////////////////////////////////////////////////////////////////
// LAYER IMPLEMENTATIONS
//
// Layers are scan-like higher-order functions, which take in cells, and
// transform them to functions of signature
//
// (io_type input, hidden_type hidden, param_type params) -> (io_type, hidden_type)
//
// which can apply the cell over a sequence of inputs, and produce both a new set
// of hidden states, as well as a concatenated output of each step.

template<typename output_type, typename hidden_type>
struct LayerOutput {
  output_type outputs;      // 输出的类型，通常是一个输出张量或者向量
  hidden_type final_hidden; // 最终的隐藏状态，通常是最后一个时间步的隐藏状态
};

template<typename io_type, typename hidden_type, typename param_type>
struct Layer {
  using output_type = LayerOutput<io_type, hidden_type>;

  virtual ~Layer() = default; // 虚析构函数，用于确保派生类正确释放资源，避免编译警告 -Wnon-virtual-dtor
  virtual output_type operator()(
      const io_type& input,
      const hidden_type& input_hidden,
      const param_type& params) const = 0; // 纯虚函数，派生类必须实现的函数，将输入数据、隐藏状态和参数转换为输出和新的隐藏状态
};

template<typename hidden_type, typename cell_params>
struct FullLayer : Layer<Tensor, hidden_type, cell_params> {
  using output_type =
      typename Layer<Tensor, hidden_type, cell_params>::output_type;
  using unstacked_output_type = LayerOutput<std::vector<Tensor>, hidden_type>;

  FullLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {}; // 构造函数，初始化 FullLayer 的成员变量 cell_

  unstacked_output_type operator()(
      const std::vector<Tensor>& step_inputs,
      const hidden_type& input_hidden,
      const cell_params& params,
      bool pre_compute_input = false) const {
    std::vector<Tensor> step_outputs;
    auto hidden = input_hidden;
    for (const auto& input : step_inputs) {
      hidden = cell_(input, hidden, params, pre_compute_input); // 应用 cell_ 对每一个输入进行处理，更新隐藏状态
      step_outputs.emplace_back(hidden_as_output(hidden)); // 将处理后的隐藏状态转换为输出张量，存储在 step_outputs 中
    }
    return {step_outputs, hidden}; // 返回包含所有步骤输出和最终隐藏状态的对象
  }

  output_type operator()(
      const Tensor& inputs,
      const hidden_type& input_hidden,
      const cell_params& params) const override {
    if (inputs.device().is_cpu()) {
      const auto inputs_w = params.linear_ih(inputs); // 根据输入张量计算线性变换后的结果
      auto unstacked_output =
          (*this)(inputs_w.unbind(0), input_hidden, params, true); // 对每个时间步应用 FullLayer，传入线性变换后的输入
      TORCH_CHECK(unstacked_output.outputs.size()>0, "Expected sequence length to be larger than 0 in RNN");
      return {at::stack(unstacked_output.outputs, 0), // 将输出张量序列堆叠为一个张量
              unstacked_output.final_hidden}; // 返回堆叠后的输出张量和最终隐藏状态
    }
    auto unstacked_output = (*this)(inputs.unbind(0), input_hidden, params); // 对每个时间步应用 FullLayer，传入输入的每个时间步
    TORCH_CHECK(unstacked_output.outputs.size()>0, "Expected sequence length to be larger than 0 in RNN");
    return {at::stack(unstacked_output.outputs, 0), // 将输出张量序列堆叠为一个张量
            unstacked_output.final_hidden}; // 返回堆叠后的输出张量和最终隐藏状态
  }

  Cell<hidden_type, cell_params>& cell_; // 引用类型的 cell 对象，用于执行每个时间步的处理
};

template <typename dir_hidden_type, typename cell_params>
struct FullBidirectionalLayer
    : Layer<Tensor, pair_of<dir_hidden_type>, pair_of<cell_params>> {
```  
# 定义继承自 `Layer` 的模板类 `FullBidirectionalLayer`，使用模板参数 `Tensor`、`pair_of<dir_hidden_type>` 和 `pair_of<cell_params>`。

  using hidden_type = pair_of<dir_hidden_type>;
```  
# 定义类型别名 `hidden_type`，表示 `pair_of<dir_hidden_type>`。

  using param_type = pair_of<cell_params>;
```  
# 定义类型别名 `param_type`，表示 `pair_of<cell_params>`。

  using output_type = typename Layer<Tensor, hidden_type, param_type>::output_type;
```  
# 定义类型别名 `output_type`，表示 `Layer<Tensor, hidden_type, param_type>` 类的 `output_type`。

  FullBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
    : layer_(cell) {};
```  
# FullBidirectionalLayer 类的构造函数，接受类型为 `Cell<dir_hidden_type, cell_params>` 的参数 `cell`，初始化成员变量 `layer_`。

  output_type operator()(
      const Tensor& input,
      const hidden_type& input_hidden,
      const param_type& params) const override {
```  
# 重载 `operator()` 函数，接受参数 `input`（类型为 `Tensor`）、`input_hidden`（类型为 `hidden_type`）、`params`（类型为 `param_type`），并返回 `output_type`。

    std::vector<Tensor> step_inputs;
```  
# 声明 `std::vector<Tensor>` 类型的变量 `step_inputs`。

    if (input.device().is_cpu()) {
```  
# 如果 `input` 的设备类型是 CPU。

      auto input_w = params.first.linear_ih(input);
```  
# 使用 `params.first` 中的 `linear_ih` 函数对 `input` 进行线性变换，结果存储在 `input_w` 中。

      step_inputs = input_w.unbind(0);
```  
# 将 `input_w` 沿着维度 0 解绑，并赋值给 `step_inputs`。

      auto fw_result = layer_(
          step_inputs, input_hidden.first, params.first, true);
```  
# 调用 `layer_` 对象的函数调用运算符，传递 `step_inputs`、`input_hidden.first`、`params.first` 和 `true` 作为参数，存储结果在 `fw_result` 中。

      TORCH_CHECK(fw_result.outputs.size() > 0, "Expected sequence length to be larger than 0 in RNN");
```  
# 使用 `TORCH_CHECK` 断言 `fw_result.outputs.size()` 大于 0，如果不满足则抛出错误信息 "Expected sequence length to be larger than 0 in RNN"。

      auto fw_output = at::stack(fw_result.outputs, 0);
```  
# 使用 `at::stack` 将 `fw_result.outputs` 沿着维度 0 堆叠，结果存储在 `fw_output` 中。

      input_w = params.second.linear_ih(input);
```  
# 使用 `params.second` 中的 `linear_ih` 函数对 `input` 进行线性变换，结果存储在 `input_w` 中。

      step_inputs = input_w.unbind(0);
```  
# 将 `input_w` 沿着维度 0 解绑，并赋值给 `step_inputs`。

      auto rev_step_inputs = reverse(std::move(step_inputs));
```  
# 调用 `reverse` 函数对 `step_inputs` 进行反转操作，并将结果移动给 `rev_step_inputs`。

      auto rev_result =
          layer_(rev_step_inputs, input_hidden.second, params.second, true);
```  
# 调用 `layer_` 对象的函数调用运算符，传递 `rev_step_inputs`、`input_hidden.second`、`params.second` 和 `true` 作为参数，存储结果在 `rev_result` 中。

      std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
```  
# 反转 `rev_result.outputs` 容器中的元素。

      auto rev_output = at::stack(rev_result.outputs, 0);
```  
# 使用 `at::stack` 将 `rev_result.outputs` 沿着维度 0 堆叠，结果存储在 `rev_output` 中。

      return {at::cat({fw_output, rev_output}, fw_output.dim() - 1),
              std::make_pair(fw_result.final_hidden, rev_result.final_hidden)};
```  
# 返回一个 `std::pair`，包含两个元素：1. 使用 `at::cat` 沿着 `fw_output.dim() - 1` 维度连接 `fw_output` 和 `rev_output`；2. 一个 `std::pair` 包含 `fw_result.final_hidden` 和 `rev_result.final_hidden`。

    }

    step_inputs = input.unbind(0);
```  
# 如果 `input` 设备类型不是 CPU，则将 `input` 沿着维度 0 解绑，并赋值给 `step_inputs`。

    auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
```  
# 调用 `layer_` 对象的函数调用运算符，传递 `step_inputs`、`input_hidden.first` 和 `params.first` 作为参数，存储结果在 `fw_result` 中。

    TORCH_CHECK(fw_result.outputs.size() > 0, "Expected sequence length to be larger than 0 in RNN");
```  
# 使用 `TORCH_CHECK` 断言 `fw_result.outputs.size()` 大于 0，如果不满足则抛出错误信息 "Expected sequence length to be larger than 0 in RNN"。

    auto fw_output = at::stack(fw_result.outputs, 0);
```  
# 使用 `at::stack` 将 `fw_result.outputs` 沿着维度 0 堆叠，结果存储在 `fw_output` 中。

    auto rev_step_inputs = reverse(std::move(step_inputs));
```  
# 调用 `reverse` 函数对 `step_inputs` 进行反转操作，并将结果移动给 `rev_step_inputs`。

    auto rev_result =
        layer_(rev_step_inputs, input_hidden.second, params.second);
```  
# 调用 `layer_` 对象的函数调用运算符，传递 `rev_step_inputs`、`input_hidden.second` 和 `params.second` 作为参数，存储结果在 `rev_result` 中。

    std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
```  
# 反转 `rev_result.outputs` 容器中的元素。

    auto rev_output = at::stack(rev_result.outputs, 0);
```  
# 使用 `at::stack` 将 `rev_result.outputs` 沿着维度 0 堆叠，结果存储在 `rev_output` 中。

    return {at::cat({fw_output, rev_output}, fw_output.dim() - 1),
            std::make_pair(fw_result.final_hidden, rev_result.final_hidden)};
```  
# 返回一个 `std::pair`，包含两个元素：1. 使用 `at::cat` 沿着 `fw_output.dim() - 1` 维度连接 `fw_output` 和 `rev_output`；2. 一个 `std::pair` 包含 `fw_result.final_hidden` 和 `rev_result.final_hidden`。

  }

  std::vector<Tensor> reverse(std::vector<Tensor>&& x) const {
```  
# 定义 `reverse` 函数，接受右值引用 `std::vector<Tensor>&& x`，返回 `std::vector<Tensor>`。

    std::reverse(x.begin(), x.end());
```  
# 反转 `x` 中元素的顺序。

    return std::move(x);
```  
# 返回移动后的 `x`。

  }

  FullLayer<dir_hidden_type, cell_params> layer_;
```  
# 声明 `FullLayer<dir_hidden_type, cell_params>` 类型的成员变量 `layer_`。
};

// 模板定义：PackedLayer，继承自Layer类，处理PackedSequence、hidden_type和cell_params类型
template<typename hidden_type, typename cell_params>
struct PackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
  // 定义输出类型为Layer的output_type
  using output_type =
      typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

  // 构造函数，初始化cell_
  PackedLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {};

  // 重载函数调用运算符，处理PackedSequence输入和hidden_type隐藏状态
  output_type operator()(
      const PackedSequence& input,
      const hidden_type& input_hidden,
      const cell_params& params) const override {

    // 存储每个步骤的输出和隐藏状态
    std::vector<at::Tensor> step_outputs;
    std::vector<hidden_type> hiddens;
    int64_t input_offset = 0;
    int64_t num_steps = input.batch_sizes.size(0);
    int64_t* batch_sizes = input.batch_sizes.data_ptr<int64_t>();
    int64_t last_batch_size = batch_sizes[0];

    const Tensor* input_ptr = &input.data;
    bool pre_compute_input = false;
    Tensor input_w;
    
    // 如果输入数据在CPU上，通过线性函数params.linear_ih预计算input_w
    if (input.data.device().is_cpu()) {
      input_w = params.linear_ih(input.data);
      input_ptr = &input_w;
      pre_compute_input = true;
    }

    // 批次大小batch_sizes是递减长度的序列，表示每个步骤中要处理的元素数目
    // 在每个步骤中，我们从输入中切片出batch_size个元素，并可能调整隐藏状态的大小
    // 因为有些序列已经完成。切片的部分也被保存下来，因为我们需要返回最终隐藏状态的张量。
    auto hidden = input_hidden;
    for (const auto i : c10::irange(num_steps)) {
      const int64_t batch_size = batch_sizes[i];
      auto step_input = input_ptr->narrow(0, input_offset, batch_size);
      input_offset += batch_size;
      const int64_t dec = last_batch_size - batch_size;
      
      // 如果有减少的batch_size，切片隐藏状态并保存
      if (dec > 0) {
        hiddens.emplace_back(
            hidden_slice(hidden, last_batch_size - dec, last_batch_size));
        hidden = hidden_slice(hidden, 0, last_batch_size - dec);
      }

      last_batch_size = batch_size;
      // 计算下一个隐藏状态
      hidden = cell_(step_input, hidden, params, pre_compute_input);
      // 将计算结果作为步骤输出保存
      step_outputs.push_back(hidden_as_output(hidden));
    }
    // 最后一个隐藏状态也要保存
    hiddens.emplace_back(hidden);
    // 将隐藏状态反转，因为它们是从后向前计算的
    std::reverse(hiddens.begin(), hiddens.end());

    // 返回输出，包括处理后的PackedSequence和连接后的隐藏状态
    return {PackedSequence{at::cat(step_outputs, 0), input.batch_sizes},
            hidden_concat(hiddens)};
  }

  // 成员变量：保存对Cell的引用
  Cell<hidden_type, cell_params>& cell_;
};

// 模板定义：ReversedPackedLayer，继承自Layer类，处理PackedSequence、hidden_type和cell_params类型
template<typename hidden_type, typename cell_params>
struct ReversedPackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
  // 定义输出类型为Layer的output_type
  using output_type =
      typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

  // 构造函数，初始化cell_
  ReversedPackedLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {};

  // 重载函数调用运算符，处理PackedSequence输入和hidden_type隐藏状态
  output_type operator()(
      const PackedSequence& input,
      const hidden_type& input_hidden,
      const cell_params& params) const override {
    // 存储每个步骤的输出
    std::vector<at::Tensor> step_outputs;
    int64_t input_offset = input.data.size(0);
    int64_t num_steps = input.batch_sizes.size(0);
    int64_t* batch_sizes = input.batch_sizes.data_ptr<int64_t>();
    // 获取最后一个步骤的批次大小
    int64_t last_batch_size = batch_sizes[num_steps - 1];

    // 指向输入数据的指针
    const Tensor* input_ptr = &input.data;
    // 是否预先计算输入
    bool pre_compute_input = false;
    // 输入权重张量
    Tensor input_w;
    // 如果输入数据位于 CPU 上
    if (input.data.device().is_cpu()) {
      // 使用线性层计算输入权重
      input_w = params.linear_ih(input.data);
      // 更新输入指针为计算后的权重
      input_ptr = &input_w;
      // 标记已经预先计算输入
      pre_compute_input = true;
    }

    // 这里的情况类似于上面，不同之处在于我们从最小的批次大小开始（以及我们实际使用的少量隐藏状态），
    // 并且随着我们向后移动覆盖 1D 输入列表，逐步扩展隐藏状态。
    auto hidden = hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1]);
    // 从最后一个步骤开始向前遍历
    for (int64_t i = num_steps - 1; i >= 0; --i) {
      // 当前步骤的批次大小
      const int64_t batch_size = batch_sizes[i];
      // 增加的批次大小
      const int64_t inc = batch_size - last_batch_size;
      // 如果增加的批次大小大于0，则扩展隐藏状态
      if (inc > 0) {
        hidden = hidden_concat(ArrayRef<hidden_type>{
            hidden, hidden_slice(input_hidden, last_batch_size, batch_size)});
      }
      // 获取当前步骤的输入
      auto step_input =
          input_ptr->narrow(0, input_offset - batch_size, batch_size);
      // 更新输入偏移量
      input_offset -= batch_size;
      // 更新最后一个批次大小
      last_batch_size = batch_size;
      // 计算当前步骤的隐藏状态
      hidden = cell_(step_input, hidden, params, pre_compute_input);
      // 将隐藏状态作为步骤输出的一部分存储起来
      step_outputs.emplace_back(hidden_as_output(hidden));
    }
    // 将步骤输出逆序，使其按照时间步长正序排列
    std::reverse(step_outputs.begin(), step_outputs.end());
    // 返回打包后的序列和最终的隐藏状态
    return {PackedSequence{at::cat(step_outputs, 0), input.batch_sizes},
            hidden};
  }
  
  // 存储对单元格的引用
  Cell<hidden_type, cell_params>& cell_;
};

template <typename dir_hidden_type, typename cell_params>
// 定义 PackedBidirectionalLayer 结构体，继承自 Layer<PackedSequence, pair_of<dir_hidden_type>, pair_of<cell_params>>
struct PackedBidirectionalLayer
    : Layer<PackedSequence, pair_of<dir_hidden_type>, pair_of<cell_params>> {
  // 使用别名简化类型定义
  using hidden_type = pair_of<dir_hidden_type>;
  using param_type = pair_of<cell_params>;
  using output_type =
      typename Layer<PackedSequence, hidden_type, param_type>::output_type;

  // 构造函数，接受一个 cell 对象并初始化两个内部层对象
  PackedBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
    : layer_(cell), rev_layer_(cell) {};

  // 重载 () 操作符，执行双向层的前向计算
  output_type operator()(
      const PackedSequence& input,
      const hidden_type& input_hidden,
      const param_type& params) const override {
    // 执行正向层计算
    auto fw_result = layer_(input, input_hidden.first, params.first);
    // 执行反向层计算
    auto rev_result = rev_layer_(input, input_hidden.second, params.second);
    // 合并正向和反向层的输出数据，并封装成 PackedSequence
    PackedSequence output{
        at::cat({fw_result.outputs.data, rev_result.outputs.data}, -1),
        input.batch_sizes};
    // 返回合并后的输出数据和最终的隐藏状态
    return {output,
            std::make_pair(fw_result.final_hidden, rev_result.final_hidden)};
  }

  // 正向层对象
  PackedLayer<dir_hidden_type, cell_params> layer_;
  // 反向层对象
  ReversedPackedLayer<dir_hidden_type, cell_params> rev_layer_;
};

////////////////////////////////////////////////////////////////////////////////
// apply_layer_stack
//
// layers are convenient, but in reality we often want to stack them. this little
// helper manages slicing of all inputs and parameters, and repeatedly feeds them
// into the given layer. returns the last layer's outputs, and a vector of final
// hidden states produced at each level.

// 对输入张量应用 dropout 操作，返回 dropout 后的结果
Tensor dropout(const Tensor& input, double p) {
  return at::dropout(input, p, /*train=*/true);
}

// 对输入 PackedSequence 应用 dropout 操作，返回 dropout 后的 PackedSequence
PackedSequence dropout(const PackedSequence& input, double p) {
  return {at::dropout(input.data, p, /*train=*/true), input.batch_sizes};
}

// 应用堆叠的层到输入数据上，并返回最后一层的输出和每一层产生的最终隐藏状态向量
template<typename io_type, typename hidden_type, typename weight_type>
LayerOutput<io_type, std::vector<hidden_type>>
apply_layer_stack(const Layer<io_type, hidden_type, weight_type>& layer, const io_type& input,
                  const std::vector<hidden_type>& hiddens, const std::vector<weight_type>& weights,
                  int64_t num_layers, double dropout_p, bool train) {
  // 检查隐藏状态和权重向量的大小是否与层数一致
  TORCH_CHECK(num_layers == (int64_t)hiddens.size(), "Expected more hidden states in stacked_rnn");
  TORCH_CHECK(num_layers == (int64_t)weights.size(), "Expected more weights in stacked_rnn");

  // 初始化层输入为输入数据
  auto layer_input = input;
  auto hidden_it = hiddens.begin();
  auto weight_it = weights.begin();
  std::vector<hidden_type> final_hiddens;
  // 遍历每一层
  for (const auto l : c10::irange(num_layers)) {
    // 调用层的操作符重载，进行前向计算
    auto layer_output = layer(layer_input, *(hidden_it++), *(weight_it++));
    // 记录当前层的最终隐藏状态
    final_hiddens.push_back(layer_output.final_hidden);
    // 更新下一层的输入为当前层的输出
    layer_input = layer_output.outputs;

    // 如果需要进行 dropout，并且处于训练状态，并且不是最后一层，则应用 dropout
    if (dropout_p != 0 && train && l < num_layers - 1) {
      layer_input = dropout(layer_input, dropout_p);
    }
  }

  // 返回最后一层的输出和所有层产生的最终隐藏状态向量
  return {layer_input, final_hiddens};
}

////////////////////////////////////////////////////////////////////////////////
// HELPERS SIMPLIFYING DISPATCH TO FUNCTIONS ABOVE
////////////////////////////////////////////////////////////////////////////////
// 定义模板函数 _rnn_impl，用于实现单向或双向 RNN 的计算
template<typename CellType, template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
LayerOutput<io_type, std::vector<typename CellType::hidden_type>> _rnn_impl(
      const io_type& input,                                       // 输入数据
      const std::vector<cell_params>& params,                     // RNN 单元参数列表
      const std::vector<typename CellType::hidden_type>& hiddens, // RNN 隐藏状态列表
      int64_t num_layers,                                         // RNN 层数
      double dropout_p,                                           // Dropout 概率
      bool train,                                                 // 是否训练模式
      bool bidirectional) {                                       // 是否双向 RNN
  using hidden_type = typename CellType::hidden_type;
  CellType cell;                                                  // 创建 RNN 单元对象

  // 如果是双向 RNN
  if (bidirectional) {
    using BidirLayer = BidirLayerT<hidden_type, cell_params>;
    // 应用双向层堆栈，计算输出和最终隐藏状态
    auto bidir_result = apply_layer_stack(BidirLayer{cell}, input, pair_vec(hiddens), pair_vec(params), num_layers, dropout_p, train);
    return {bidir_result.outputs, unpair_vec(std::move(bidir_result.final_hidden))};
  } else {
    // 应用单向层堆栈，计算输出和最终隐藏状态
    return apply_layer_stack(LayerT<hidden_type,cell_params>{cell}, input, hiddens, params, num_layers, dropout_p, train);
  }
}

////////////////////////////////////////////////////////////////////////////////
// 定义模板函数 _rnn_impl_with_concat，用于实现带连接操作的 RNN 计算
template<typename CellType, template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
std::tuple<io_type, Tensor> _rnn_impl_with_concat(
      const io_type& input,                                       // 输入数据
      const std::vector<cell_params>& params,                     // RNN 单元参数列表
      const std::vector<typename CellType::hidden_type>& hiddens, // RNN 隐藏状态列表
      int64_t num_layers,                                         // RNN 层数
      double dropout_p,                                           // Dropout 概率
      bool train,                                                 // 是否训练模式
      bool bidirectional) {                                       // 是否双向 RNN
  auto result = _rnn_impl<CellType, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);
  // 返回 RNN 计算的输出和隐藏状态的 Tensor 堆叠结果
  return std::make_tuple(std::move(result.outputs), at::stack(result.final_hidden, 0));
}

////////////////////////////////////////////////////////////////////////////////
// 定义模板函数 _lstm_impl，用于实现 LSTM 的计算
template<template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
std::tuple<io_type, Tensor, Tensor> _lstm_impl(
      const io_type& input,                                       // 输入数据
      const std::vector<cell_params>& params,                     // LSTM 单元参数列表
      const Tensor& hx, const Tensor& cx,                         // 初始隐藏状态和细胞状态
      int64_t num_layers,                                         // LSTM 层数
      double dropout_p,                                           // Dropout 概率
      bool train,                                                 // 是否训练模式
      bool bidirectional) {                                       // 是否双向 LSTM
  // 我们更方便处理每层的 hx 和 cx 对组成的列表，因此需要对这些 Tensor 进行转置
  auto layer_hx = hx.unbind(0);
  auto layer_cx = cx.unbind(0);
  int64_t total_layers = layer_hx.size();
  std::vector<typename LSTMCell<cell_params>::hidden_type> hiddens;
  hiddens.reserve(total_layers);
  // 将每层的 hx 和 cx 对组装成隐藏状态列表
  for (const auto i : c10::irange(total_layers)) {
    hiddens.emplace_back(std::move(layer_hx[i]), std::move(layer_cx[i]));
  }

  // 调用 _rnn_impl 函数计算 LSTM 的输出和最终隐藏状态
  auto result = _rnn_impl<LSTMCell<cell_params>, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);

  // 现在，我们需要反转之前进行的转置操作，得到每层的隐藏状态 Tensor 列表
  std::vector<Tensor> hy, cy;
  hy.reserve(total_layers); cy.reserve(total_layers);
  for (auto & hidden : result.final_hidden) {
    hy.push_back(std::move(std::get<0>(hidden)));
    cy.push_back(std::move(std::get<1>(hidden)));
  }
    # 将 hidden 中第二个元素的值（使用移动语义）添加到 cy 后面
    cy.push_back(std::move(std::get<1>(hidden)));
  }
  
  # 使用移动语义将 result.outputs、hy 的堆叠结果、cy 的堆叠结果封装成元组返回
  return std::make_tuple(std::move(result.outputs), at::stack(hy, 0), at::stack(cy, 0));
} // anonymous namespace
    // 检查输入张量是否适用于 cuDNN 加速，如果是则调用相应的 cuDNN 版本的函数进行处理
    if (at::cudnn_is_acceptable(_input)) {                                  
      Tensor output, hy;                                                    
      // 调用对应的 cuDNN 版本的函数，处理输入张量并生成输出张量和隐藏状态张量
      NAME##_cudnn_stub(                                                    
          _input.device().type(),                                           
          output,                                                           
          hy,                                                               
          _input,                                                           
          hx,                                                               
          _params,                                                          
          has_biases,                                                       
          num_layers,                                                       
          dropout_p,                                                        
          train,                                                            
          bidirectional,                                                    
          batch_first);                                                     
      // 返回处理后的输出张量和隐藏状态张量的元组
      return std::make_tuple(std::move(output), std::move(hy));             
    }                                                                       
    // 如果不适用 cuDNN 加速，则检查是否适用于 MIOpen 加速
    if (use_miopen(_input, dropout_p)) {                                     
      Tensor output, hy;                                                    
      // 调用对应的 MIOpen 版本的函数，处理输入张量并生成输出张量和隐藏状态张量
      NAME##_miopen_stub(                                                   
          _input.device().type(),                                           
          output,                                                           
          hy,                                                               
          _input,                                                           
          hx,                                                               
          _params,                                                          
          has_biases,                                                       
          num_layers,                                                       
          dropout_p,                                                        
          train,                                                            
          bidirectional,                                                    
          batch_first);                                                     
      // 返回处理后的输出张量和隐藏状态张量的元组
      return std::make_tuple(std::move(output), std::move(hy));             
    }                                                                       
    // 如果既不适用 cuDNN 也不适用 MIOpen 加速，则检查输入张量和参数的属性是否满足要求
    check_attributes(_input, _params, hx);                                   
    // 根据是否 batch_first 标志位选择是否进行转置操作，生成新的输入张量
    auto input = batch_first ? _input.transpose(0, 1) : _input;              
    // 根据是否具有偏置参数，获取相应的参数并返回
    auto params = gather_params(_params, has_biases);                        
    # 调用一个带有连接功能的循环神经网络实现，具体实现由模板参数指定
    auto results =                                                          \
        _rnn_impl_with_concat<CELL, FullLayer, FullBidirectionalLayer>(     \
            input,                                                          \
            params,                                                         \
            hx.unbind(0),                                                   \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional);                                                 \
    # 如果 batch_first 为真，则交换结果张量的第一和第二维度
    if (batch_first) {                                                      \
      std::get<0>(results).transpose_(0, 1);                                \
    }                                                                       \
    # 返回 RNN 模型的结果，可能包括输出张量和最终状态张量的元组
    return results;                                                         \
  }                                                                         \
                                                                            \
  # 定义一个函数模板 NAME，接受多个参数来执行某种 RNN 操作
  std::tuple<Tensor, Tensor> NAME(                                          \
      const Tensor& data,                                                   \
      const Tensor& batch_sizes,                                            \
      const Tensor& hx,                                                     \
      TensorList _params,                                                   \
      bool has_biases,                                                      \
      int64_t num_layers,                                                   \
      double dropout_p,                                                     \
      bool train,                                                           \
      bool bidirectional) {
    // 检查是否可以使用 cuDNN 来加速计算
    if (at::cudnn_is_acceptable(data)) {                                    \
      Tensor output, hy;                                                    \
      // 调用相应的 cuDNN 加速函数
      NAME##_packed_cudnn_stub(                                             \
          data.device().type(),                                             \
          output,                                                           \
          hy,                                                               \
          data,                                                             \
          batch_sizes,                                                      \
          hx,                                                               \
          _params,                                                          \
          has_biases,                                                       \
          num_layers,                                                       \
          dropout_p,                                                        \
          train,                                                            \
          bidirectional);                                                   \
      // 返回计算结果的输出张量和最终隐藏状态
      return std::make_tuple(std::move(output), std::move(hy));             \
    }                                                                       \
    // 检查是否可以使用 MIOpen 来加速计算
    if (use_miopen(data, dropout_p)) {                                      \
      Tensor output, hy;                                                    \
      // 调用相应的 MIOpen 加速函数
      NAME##_packed_miopen_stub(                                            \
          data.device().type(),                                             \
          output,                                                           \
          hy,                                                               \
          data,                                                             \
          batch_sizes,                                                      \
          hx,                                                               \
          _params,                                                          \
          has_biases,                                                       \
          num_layers,                                                       \
          dropout_p,                                                        \
          train,                                                            \
          bidirectional);                                                   \
      // 返回计算结果的输出张量和最终隐藏状态
      return std::make_tuple(std::move(output), std::move(hy));             \
    }                                                                       \
    // 将输入数据和批次大小封装成 PackedSequence
    PackedSequence input{data, batch_sizes};                                \
    // 调用 gather_params 函数获取所有参数
    auto params = gather_params(_params, has_biases);                       \
    // 调用 RNN 实现函数 _rnn_impl_with_concat，使用模板参数 CELL, PackedLayer, PackedBidirectionalLayer
    // 传入参数包括：
    // - input: 输入数据
    // - params: RNN 参数
    // - hx.unbind(0): RNN 的初始隐藏状态
    // - num_layers: RNN 层数
    // - dropout_p: Dropout 概率
    // - train: 是否在训练模式下
    // - bidirectional: 是否使用双向 RNN
    auto result =                                                           \
        _rnn_impl_with_concat<CELL, PackedLayer, PackedBidirectionalLayer>( \
            input,                                                          \
            params,                                                         \
            hx.unbind(0),                                                   \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional);                                                 \
    // 获取 result 的第一个元素作为 packed_output 引用
    auto& packed_output = std::get<0>(result);
    // 返回一个包含两个元素的 tuple：
    // - packed_output.data 的移动语义
    // - result 的第二个元素的移动语义
    return std::make_tuple(                                                 \
        std::move(packed_output.data), std::move(std::get<1>(result)));     \
  }
#define ONE_HIDDEN_QRNN(NAME, CELL)                                         \
  static std::tuple<Tensor, Tensor> NAME##_input(                           \
      const Tensor& _input,                                                 \
      const Tensor& hx,                                                     \
      c10::List<c10::intrusive_ptr<CellParamsBase>> _params,                \
      bool has_biases,                                                      \
      int64_t num_layers,                                                   \
      double dropout_p,                                                     \
      bool train,                                                           \
      bool bidirectional,                                                   \
      bool batch_first) {                                                   \
    // 创建一个空的参数向量，用于存储转换后的参数
    std::vector<QRNNCellParamsWrapper> params;                              \
    // 将输入的参数列表 _params 转换为标准的参数向量 params
    for (c10::intrusive_ptr<CellParamsBase> x : _params) {                  \
      params.emplace_back(std::move(x));                                    \
    }                                                                       \
    // 根据 batch_first 参数决定是否转置输入张量
    auto input = batch_first ? _input.transpose(0, 1) : _input;             \
    // 调用 _rnn_impl_with_concat 函数进行 QRNN 的实现
    auto results =                                                          \
        _rnn_impl_with_concat<CELL, FullLayer, FullBidirectionalLayer>(     \
            input,                                                          \
            params,                                                         \
            hx.unbind(0),                                                   \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional);                                                 \
    // 如果 batch_first 为 true，则需要再次转置输出结果
    if (batch_first) {                                                      \
      std::get<0>(results).transpose_(0, 1);                                \
    }                                                                       \
    return results;                                                         \
  }                                                                         \
                                                                            \
  static std::tuple<Tensor, Tensor> NAME##_data(                            \  // 定义静态函数 NAME##_data，返回类型为 std::tuple<Tensor, Tensor>
      const Tensor& data,                                                   \  // 第一个参数是常引用的 Tensor 对象 data，表示输入数据
      const Tensor& batch_sizes,                                            \  // 第二个参数是常引用的 Tensor 对象 batch_sizes，表示批次大小信息
      const Tensor& hx,                                                     \  // 第三个参数是常引用的 Tensor 对象 hx，表示隐藏状态
      c10::List<c10::intrusive_ptr<CellParamsBase>> _params,                \  // 第四个参数是 CellParamsBase 指针列表 _params，表示RNN单元参数
      bool has_biases,                                                      \  // 布尔值参数，指示是否包含偏置
      int64_t num_layers,                                                   \  // 整数参数，表示层数
      double dropout_p,                                                     \  // 双精度浮点数参数，表示dropout概率
      bool train,                                                           \  // 布尔值参数，指示是否训练模式
      bool bidirectional) {                                                 \  // 布尔值参数，指示是否双向RNN
    std::vector<QRNNCellParamsWrapper> params;                              \  // 创建 QRNNCellParamsWrapper 对象的向量 params，用于存储 RNN 单元参数
    for (c10::intrusive_ptr<CellParamsBase> x : _params) {                  \  // 遍历输入的 RNN 单元参数列表 _params
      params.emplace_back(std::move(x));                                    \  // 将每个参数转移并加入 params 向量中
    }                                                                       \
    PackedSequence input{data, batch_sizes};                                \  // 创建 PackedSequence 对象 input，用给定的 data 和 batch_sizes 初始化
    auto result =                                                           \
        _rnn_impl_with_concat<CELL, PackedLayer, PackedBidirectionalLayer>( \  // 调用 _rnn_impl_with_concat 函数模板实例化，返回结果存储在 result 中
            input,                                                          \  // 输入参数：PackedSequence 对象 input
            params,                                                         \  // 输入参数：RNN 单元参数向量 params
            hx.unbind(0),                                                   \  // 输入参数：hx 的第一个维度解绑后的 Tensor
            num_layers,                                                     \  // 输入参数：层数 num_layers
            dropout_p,                                                      \  // 输入参数：dropout 概率 dropout_p
            train,                                                          \  // 输入参数：训练模式标志 train
            bidirectional);                                                 \  // 输入参数：双向标志 bidirectional
    auto& packed_output = std::get<0>(result);                              \  // 获取 result 的第一个元素作为 packed_output 引用
    return std::make_tuple(                                                 \  // 返回一个 std::tuple 对象，包含以下两个元素
        std::move(packed_output.data), std::move(std::get<1>(result)));     \  // 第一个元素是 packed_output 的 data 成员，第二个元素是 result 的第二个元素
  }
// 定义一个使用 GRU 单元格和参数类型为 CellParams 的单隐藏层 RNN 模型
ONE_HIDDEN_RNN(gru, GRUCell<CellParams>)

// 定义一个使用 quantized_gru 单元格和参数类型为 QRNNCellParamsWrapper 的单隐藏层 QRNN 模型
ONE_HIDDEN_QRNN(quantized_gru, GRUCell<QRNNCellParamsWrapper>)

// quantized_gru 的 BC 包装器

// 用于处理使用 List[Tensor] 作为参数的 quantized_gru 输入的旧版本兼容方法
static std::tuple<Tensor, Tensor> quantized_gru_input_legacy(
    const Tensor& _input,
    const Tensor& hx,
    c10::List<at::Tensor> _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_CHECK(
      false,
      "torch.quantized_gru with List[Tensor] for parameters is "
      "no longer supported. Please re-export your model "
      "using the newer definitions in torch.jit.quantized");
}

// 用于处理使用 List[Tensor] 作为参数的 quantized_gru 数据输入的旧版本兼容方法
static std::tuple<Tensor, Tensor> quantized_gru_data_legacy(
    const Tensor& data,
    const Tensor& batch_sizes,
    const Tensor& hx,
    c10::List<at::Tensor> _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  TORCH_CHECK(
      false,
      "torch.quantized_gru with List[Tensor] for parameters is "
      "no longer supported. Please re-export your model "
      "using the newer definitions in torch.jit.quantized");
}

// 定义 tanh 单元格类型为 SimpleCell<tanh_f, CellParams> 的单隐藏层 RNN 模型
using tanf_cell_type = SimpleCell<tanh_f, CellParams>;
ONE_HIDDEN_RNN(rnn_tanh, tanf_cell_type)

// 定义 relu 单元格类型为 SimpleCell<relu_f, CellParams> 的单隐藏层 RNN 模型
using relu_cell_type = SimpleCell<relu_f, CellParams>;
ONE_HIDDEN_RNN(rnn_relu, relu_cell_type);

// 定义一些 LSTM 使用的分发函数
DEFINE_DISPATCH(lstm_cudnn_stub);
DEFINE_DISPATCH(lstm_packed_cudnn_stub);
DEFINE_DISPATCH(lstm_miopen_stub);
DEFINE_DISPATCH(lstm_packed_miopen_stub);
DEFINE_DISPATCH(lstm_mkldnn_stub);

// 在 CPU 上注册不支持的 LSTM 分发函数
REGISTER_NO_CPU_DISPATCH(lstm_cudnn_stub);
REGISTER_NO_CPU_DISPATCH(lstm_packed_cudnn_stub);
REGISTER_NO_CPU_DISPATCH(lstm_miopen_stub);
REGISTER_NO_CPU_DISPATCH(lstm_packed_miopen_stub);

// LSTM 函数定义，处理输入 _input，初始状态 hx，参数列表 _params 等，并返回输出、最终隐藏状态和细胞状态的元组
std::tuple<Tensor, Tensor, Tensor> lstm(
      const Tensor& _input, TensorList hx,
      TensorList _params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  // 检查隐藏状态列表的大小是否为 2
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");

  // 如果输入 _input 可以使用 cuDNN 加速
  if (at::cudnn_is_acceptable(_input)) {
    Tensor output, hy, cy;
    // 调用 cuDNN 加速的 LSTM 实现
    lstm_cudnn_stub(_input.device().type(), output, hy, cy, _input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
    return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
  }

  // 如果使用 MPS 并且满足特定条件
#ifdef USE_MPS
  if (_input.is_mps() && (mps::is_macos_13_or_newer() || num_layers == 1)) {
    // 调用 MPS 的多层 LSTM 实现
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> output = at::_lstm_mps(_input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
    std::tuple<Tensor, Tensor, Tensor> return_values = std::make_tuple(std::get<0>(output), std::get<1>(output), std::get<2>(output));
    return return_values;
  } else if (_input.is_mps()) {
    // 如果使用 MPS 但不满足条件，给出警告并回退到单层 LSTMCell 迭代
    TORCH_WARN_ONCE("Native multi-layer LSTM support in MPS available only on MacOS 13 onwards.",
                    " Falling back to LSTMCell iteration.",
                    " This may have performance implications.");
  }
#endif
}
#endif
  // 如果单元格大小不同，表示使用了投影
  bool has_projections = (hx[0].size(2) != hx[1].size(2));
  // 如果使用 MIOpen 加速
  if (use_miopen(data, dropout_p)) {
    // 如果没有投影
    if (!has_projections) {
      // 定义输出张量和隐藏状态张量
      Tensor output, hy, cy;
      // 调用 MIOpen 版本的 LSTM 实现
      lstm_packed_miopen_stub(data.device().type(), output, hy, cy, data, batch_sizes, hx,
              _params, has_biases, num_layers, dropout_p, train, bidirectional);
      // 返回结果元组
      return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
    } else {
      // 如果有投影，发出警告信息
      TORCH_WARN_ONCE(
          "LSTM with projections is not supported with MIOpen. Using default implementation.");
    }
  }

  // 如果使用 MKL-DNN 加速
  if (use_mkldnn(data, _params, hx)) {
    // 如果没有投影
    if (!has_projections) {
      // 如果第一个隐藏状态张量具有符号大小和步长
      if (hx[0].unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
        // 发出警告信息，因为 MKL-DNN 不支持带符号大小和步长的 LSTM
        TORCH_WARN_ONCE(
          "LSTM with symbolic sizes and strides is not supported with oneDNN. Using default implementation.");
      } else {
        // 定义输出张量和隐藏状态张量
        Tensor output, hy, cy;
        // 调用 MKL-DNN 版本的 LSTM 实现
        lstm_mkldnn_stub(data.device().type(), output, hy, cy, data, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional);
        // 返回结果元组
        return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
      }
    } else {
      // 如果有投影，发出警告信息
      TORCH_WARN_ONCE(
          "LSTM with projections is not supported with oneDNN. Using default implementation.");
    }
  }

  // 检查输入数据、参数和隐藏状态的属性
  check_attributes(data, _params, hx);
  // 如果 batch_first 为 true，则将输入数据进行转置
  auto input = batch_first ? data.transpose(0, 1) : data;
  // 根据是否有投影，收集相关参数
  auto params = gather_params(_params, has_biases, has_projections);
  // 调用 LSTM 的默认实现
  auto results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  // 如果 batch_first 为 true，则将输出数据进行转置
  if (batch_first) {
    std::get<0>(results) = std::get<0>(results).transpose(0, 1);
  }
  // 返回 LSTM 结果元组
  return results;
}
  }
  // 创建一个 PackedSequence 对象，使用给定的数据和批次大小
  PackedSequence input { data, batch_sizes };
  // 调用 gather_params 函数，从 _params 中收集参数，考虑是否有偏置和投影
  auto params = gather_params(_params, has_biases, has_projections);
  // 调用 _lstm_impl 函数执行 LSTM 操作，返回结果
  auto result = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  // 获取结果中的 packed_output 引用
  auto & packed_output = std::get<0>(result);
  // 返回一个包含 packed_output.data, result 的第二个元素，以及 result 的第三个元素的 tuple
  return std::make_tuple(std::move(packed_output.data),
                         std::move(std::get<1>(result)),
                         std::move(std::get<2>(result)));
}

std::tuple<Tensor, Tensor> lstm_cell(
    const Tensor& input, TensorList hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // 从可选的张量中获取偏置 b_ih，确保存在有效的引用
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  const Tensor& b_ih = *b_ih_maybe_owned;
  // 从可选的张量中获取偏置 b_hh，如果不存在，则返回一个空张量
  const Tensor& b_hh = c10::value_or_else(b_hh_opt, [] {return Tensor();});

  // 检查隐藏状态列表的大小是否为 2，如果不是则抛出异常
  TORCH_CHECK(hx.size() == 2, "lstm_cell expects two hidden states");
  // 检查输入张量的形状是否符合预期的输入权重 w_ih 的要求
  check_rnn_cell_forward_input(input, w_ih.sym_size(1));
  // 获取隐藏状态的大小（等于 w_hh 的第二个维度的大小）
  auto hidden_size = w_hh.sym_size(1);
  // 检查第一个隐藏状态的形状是否符合预期
  check_rnn_cell_forward_hidden(input, hx[0], hidden_size, 0);
  // 检查第二个隐藏状态的形状是否符合预期
  check_rnn_cell_forward_hidden(input, hx[1], std::move(hidden_size), 1);
  // 创建一个未定义的静态张量
  static at::Tensor undefined;
  // 调用 LSTMCell<CellParams> 的 operator() 方法，执行 LSTM 单元的前向计算，并返回结果元组
  return LSTMCell<CellParams>{}(input, std::make_tuple(hx[0], hx[1]), CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_thnn_differentiable_lstm_cell_backward( const std::optional<Tensor>& grad_hy_opt, const std::optional<Tensor>& grad_cy_opt,
    const Tensor& input_gates,
    const Tensor& hidden_gates, const std::optional<Tensor>& input_bias_opt, const std::optional<Tensor>& hidden_bias_opt,
    const Tensor& cx,
    const Tensor& cy) {
  // 从可选的张量中获取梯度 grad_hy，确保存在有效的引用
  c10::MaybeOwned<Tensor> grad_hy_maybe_owned = at::borrow_from_optional_tensor(grad_hy_opt);
  const Tensor& grad_hy = *grad_hy_maybe_owned;
  // 从可选的张量中获取梯度 grad_cy，如果不存在，则返回一个空张量
  const Tensor& grad_cy = c10::value_or_else(grad_cy_opt, [] {return Tensor();});
  // 从可选的张量中获取输入偏置 input_bias，如果不存在，则返回一个空张量
  const Tensor& input_bias = c10::value_or_else(input_bias_opt, [] {return Tensor();});
  // 从可选的张量中获取隐藏偏置 hidden_bias，如果不存在，则返回一个空张量
  const Tensor& hidden_bias = c10::value_or_else(hidden_bias_opt, [] {return Tensor();});

  // 如果 grad_hy 和 grad_cy 均未定义，则返回一个空元组
  if (!grad_hy.defined() && !grad_cy.defined()) {
    return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>();
  }
  // 计算输入门和隐藏门的总和
  Tensor gates = input_gates + hidden_gates;
  // 如果存在输入偏置，则加上输入偏置
  if (input_bias.defined()) {
    gates = gates + input_bias;
  }
  // 如果存在隐藏偏置，则加上隐藏偏置
  if (hidden_bias.defined()) {
    gates = gates + hidden_bias;
  }
  // 将门张量按第一个维度进行切分，得到四个门的张量
  auto chunked_gates = gates.unsafe_chunk(4, 1);
  // 分别计算输入门 i、遗忘门 f、细胞状态 c 和输出门 o 的值
  Tensor i = chunked_gates[0].sigmoid();
  Tensor f = chunked_gates[1].sigmoid();
  Tensor c = chunked_gates[2].tanh();
  Tensor o = chunked_gates[3].sigmoid();

  // 计算细胞状态的梯度 gcx，使用当前细胞状态 cy 的双曲正切作为激活函数
  Tensor gcx = cy.tanh();
  Tensor gog;
  // 断言 grad_hy 或 grad_cy 至少有一个已定义，否则抛出异常
  TORCH_INTERNAL_ASSERT((grad_hy.defined() || grad_cy.defined()),"either gradient with respect to hy or cy should be defined");
  // 如果 grad_hy 已定义，则计算输出门的梯度 gog，并更新 gcx
  if (grad_hy.defined()) {
    gog = grad_hy * gcx;
    gog = at::sigmoid_backward(gog, o);
    gcx = at::tanh_backward(grad_hy * o, gcx);
    // 如果 grad_cy 已定义，则将 grad_cy 加到 gcx 上
    if (grad_cy.defined()) {
      gcx = gcx + grad_cy;
    }
  } else if (grad_cy.defined()) {
    // 如果仅 grad_cy 定义，则 gog 初始化为与 cx 相同形状的零张量
    gog = at::zeros_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // 继续更新 gcx，将 grad_cy 加到 gcx 上
    gcx = gcx + grad_cy;
  }
    # 将gcx赋值给grad_cy
    gcx = grad_cy;
    # 计算gig，等于gcx乘以c
    Tensor gig = gcx * c;
    # 计算gfg，等于gcx乘以cx
    Tensor gfg = gcx * cx;
    # 计算gcg，等于gcx乘以i
    Tensor gcg = gcx * i;
    # 更新gcx，等于gcx乘以f
    gcx = gcx * f;
    # 对gig进行sigmoid函数的反向传播
    gig = at::sigmoid_backward(gig, i);
    # 对gfg进行sigmoid函数的反向传播
    gfg = at::sigmoid_backward(gfg, f);
    # 对gcg进行tanh函数的反向传播
    gcg = at::tanh_backward(gcg, c);
    # 将gig、gfg、gcg和gog按列拼接，形成梯度门控的张量
    Tensor grad_gates = at::cat({std::move(gig), std::move(gfg), std::move(gcg), std::move(gog)}, 1);
    # 如果input_bias已定义，则计算grad_gates在第0维上的和，否则返回空张量
    Tensor grad_bias = input_bias.defined() ? grad_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
    # 返回包含梯度门控、梯度门控、gcx、grad_bias和grad_bias的元组
    return std::make_tuple(grad_gates, grad_gates, std::move(gcx), grad_bias, grad_bias);
}

// 定义了一个函数 _thnn_differentiable_gru_cell_backward，用于计算 GRU 单元的反向传播
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _thnn_differentiable_gru_cell_backward(
    const Tensor& grad_hy,  // 输入参数：输出误差的梯度
    const Tensor& input_gates,  // 输入参数：输入门的输出
    const Tensor& hidden_gates,  // 输入参数：隐藏门的输出
    const Tensor& hx,  // 输入参数：隐藏状态
    const std::optional<Tensor>& input_bias_opt,  // 输入参数：输入偏置的可选值
    const std::optional<Tensor>& hidden_bias_opt) {  // 输入参数：隐藏偏置的可选值

  // 查看注释：处理可选张量的包装器，提取输入偏置
  c10::MaybeOwned<Tensor> input_bias_maybe_owned = at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  // 提取隐藏偏置，如果不存在则返回空张量
  const Tensor& hidden_bias = c10::value_or_else(hidden_bias_opt, [] {return Tensor();});

  // 初始化输入门和隐藏门
  Tensor in_g = input_gates;
  Tensor h_g = hidden_gates;

  // 如果存在输入偏置，将其加到输入门上
  if (input_bias.defined()){
    in_g = in_g + input_bias;
  }

  // 如果存在隐藏偏置，将其加到隐藏门上
  if (hidden_bias.defined()){
    h_g = h_g + hidden_bias;
  }

  // 对输入门进行切片操作，分别得到重置门、更新门和新内容
  auto chunked_input_gates = in_g.unsafe_chunk(3, 1);
  Tensor ir = chunked_input_gates[0];
  Tensor ii = chunked_input_gates[1];
  Tensor in = chunked_input_gates[2];

  // 对隐藏门进行切片操作，分别得到重置门、更新门和新内容
  auto chunked_hidden_gates = h_g.unsafe_chunk(3, 1);
  Tensor hr = chunked_hidden_gates[0];
  Tensor hi = chunked_hidden_gates[1];
  Tensor hn = chunked_hidden_gates[2];

  // 计算重置门和更新门的输出
  Tensor rg = (ir + hr).sigmoid();
  Tensor ig = (ii + hi).sigmoid();

  // 计算隐藏状态的梯度
  Tensor grad_hx = grad_hy * ig;

  // 计算新内容的输出
  Tensor ng = (in + rg * hn).tanh();

  // 计算更新门的梯度
  Tensor gig = at::sigmoid_backward(grad_hy * (hx - ng), ig);

  // 计算新内容的梯度
  Tensor gin = at::tanh_backward(grad_hy * (1 - ig), ng);

  // 计算隐藏状态的新内容梯度
  Tensor ghn = gin * rg;

  // 计算重置门的梯度
  Tensor grg = at::sigmoid_backward(gin * hn, rg);

  // 合并更新门和重置门的梯度，形成输入门的梯度
  Tensor grad_input_gates = at::cat({grg, gig, std::move(gin)}, 1);

  // 合并重置门、更新门和新内容的梯度，形成隐藏门的梯度
  Tensor grad_hidden_gates = at::cat({std::move(grg), std::move(gig), std::move(ghn)}, 1);

  // 计算输入偏置的梯度，如果输入偏置存在的话
  Tensor grad_input_bias = input_bias.defined() ? grad_input_gates.sum(0, /*keepdim=*/false) : at::Tensor{};

  // 计算隐藏偏置的梯度，如果输入偏置存在的话
  Tensor grad_hidden_bias = input_bias.defined() ? grad_hidden_gates.sum(0, /*keepdim=*/false) : at::Tensor{};

  // 返回计算得到的梯度元组
  return std::make_tuple(std::move(grad_input_gates), std::move(grad_hidden_gates),
                         std::move(grad_hx), std::move(grad_input_bias), std::move(grad_hidden_bias));
}

// 定义了一个函数 gru_cell，用于计算 GRU 单元的前向传播
Tensor gru_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {

  // 查看注释：处理可选张量的包装器，提取输入偏置
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  const Tensor& b_ih = *b_ih_maybe_owned;
  // 提取隐藏偏置，如果不存在则返回空张量
  const Tensor& b_hh = c10::value_or_else(b_hh_opt, [] {return Tensor();});

  // 检查 RNN 单元的前向传播输入是否合法
  check_rnn_cell_forward_input(input, w_ih.size(1));

  // 检查 RNN 单元的前向传播隐藏状态是否合法
  check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);

  // 静态张量 undefined，用于表示未定义的值
  static at::Tensor undefined;

  // 返回 GRU 单元的前向传播计算结果
  return GRUCell<CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

// 定义了一个函数 rnn_tanh_cell，用于计算 RNN 单元的前向传播
Tensor rnn_tanh_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的张量中获取 borrow_from_optional_tensor，如果不存在则使用默认的空张量
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  // 获取非空的 b_ih 张量
  const Tensor& b_ih = *b_ih_maybe_owned;
  // 获取非空的 b_hh 张量，如果不存在则返回一个默认的空张量
  const Tensor& b_hh = c10::value_or_else(b_hh_opt, [] {return Tensor();});

  // 定义一个静态的未定义张量
  static at::Tensor undefined;
  // 检查 RNN 单元的输入是否符合要求
  check_rnn_cell_forward_input(input, w_ih.size(1));
  // 检查 RNN 单元的隐藏状态是否符合要求
  check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
  // 调用 SimpleCell 的模板实例化，使用 tanh 激活函数和 CellParams 结构体作为参数
  return SimpleCell<tanh_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}
}

// 定义一个函数 rnn_relu_cell，实现基于输入和权重进行简单的循环神经网络单元计算，使用ReLU激活函数
Tensor rnn_relu_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // 使用 at::borrow_from_optional_tensor 函数获取可选的输入偏置张量 b_ih_opt 的引用
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  // 将引用转换为常量引用
  const Tensor& b_ih = *b_ih_maybe_owned;
  // 从可选的隐藏层偏置张量 b_hh_opt 中获取值或者创建一个空张量
  const Tensor& b_hh = c10::value_or_else(b_hh_opt, [] {return Tensor();});

  // 定义一个静态的未定义张量
  static at::Tensor undefined;
  // 检查输入和输入权重的大小是否匹配
  check_rnn_cell_forward_input(input, w_ih.size(1));
  // 检查输入、隐藏状态和隐藏层权重的大小是否匹配
  check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
  // 调用 SimpleCell 模板类的实例，使用 relu_f 激活函数和 CellParams 结构体参数进行计算
  return SimpleCell<relu_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

// 量化实现
//
// 这些实现使用 FBGEMM 执行带有 int8 或 float16 量化权重的 i2h 和 h2h 线性层，
// 在小批量场景下，内存获取权重矩阵的运行时占主导地位，这是优势所在。

// 定义一个函数，输入参数是输入张量 _input、隐藏状态列表 hx_、参数列表 _params_ 等，返回三个张量的元组
static std::tuple<Tensor, Tensor, Tensor> quantized_lstm_input(
    const Tensor& _input,
    c10::List<at::Tensor> hx_,
    c10::List<c10::intrusive_ptr<CellParamsBase>> _params_,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first,
    std::optional<ScalarType> dtype,
    bool use_dynamic) {
  // 将 hx_ 转换为标准的向量形式 hx
  auto hx = hx_.vec();
  // 创建一个空的参数向量列表 params，并保留 _params_ 的大小
  std::vector<QRNNCellParamsWrapper> params;
  params.reserve(_params_.size());
  // 遍历 _params_，将每个参数包装成 QRNNCellParamsWrapper，并添加到 params 中
  for (const auto& param : _params_) {
    params.emplace_back(static_cast<c10::intrusive_ptr<CellParamsBase>>(param));
  }
  // 检查隐藏状态列表 hx 的大小是否为 2
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
  // 检查隐藏状态的维度是否相等
  TORCH_CHECK(hx[0].size(2) == hx[1].size(2), "quantized LSTM with projections is not supported");
  // 获取结果的数据类型，如果未指定则默认为 at::kChar
  auto result_dtype = dtype.has_value() ? dtype.value() : at::kChar;
  // 如果 batch_first 为 true，则对输入进行转置
  auto input = batch_first ? _input.transpose(0, 1) : _input;
  // 检查是否有偏置
  TORCH_CHECK(has_biases, "quantized LSTM requires biases");
  // 检查数据类型是否支持
  TORCH_CHECK(
      result_dtype == at::kChar || result_dtype == at::kQInt8 ||
          result_dtype == at::kHalf,
      "dtype is not supported");

  // 定义一个结果元组 results
  std::tuple<Tensor, Tensor, Tensor> results;
  // 根据数据类型的不同，调用不同的 _lstm_impl 实现函数
  if (result_dtype == at::kChar || result_dtype == at::kQInt8) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    // 根据 use_dynamic 的值选择具体的 _lstm_impl 实现
    if (use_dynamic) {
      results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
          input, params, hx[0], hx[1], num_layers,
          dropout_p, train, bidirectional);
    } else {
      results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
          input, params, hx[0], hx[1], num_layers,
          dropout_p, train, bidirectional);
    }
  } else {
    results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
        input, params, hx[0], hx[1], num_layers,
        dropout_p, train, bidirectional);
  }

  // 如果 batch_first 为 true，则对结果的第一个张量进行转置
  if (batch_first) {
    std::get<0>(results) = std::get<0>(results).transpose(0, 1);
  }
  // 返回计算结果的元组
  return results;
}

// 为 quantized_lstm 提供的 BC（backward compatibility）包装器

// 定义一个函数，与 quantized_lstm_input 类似，但采用旧版本的接口或参数
static std::tuple<Tensor, Tensor, Tensor> quantized_lstm_input_legacy(
    const Tensor& _input,  

# 接收输入张量的引用 `_input`。


    c10::List<at::Tensor> hx_,

# 接收包含张量列表 `hx_`。


    c10::List<at::Tensor> _params_,

# 接收包含张量列表 `_params_`。


    bool has_biases,

# 布尔值参数 `has_biases`，表示是否存在偏置项。


    int64_t num_layers,

# 整数参数 `num_layers`，表示层数。


    double dropout_p,

# 双精度浮点数参数 `dropout_p`，表示dropout概率。


    bool train,

# 布尔值参数 `train`，表示是否处于训练模式。


    bool bidirectional,

# 布尔值参数 `bidirectional`，表示是否为双向LSTM。


    bool batch_first,

# 布尔值参数 `batch_first`，表示是否批处理优先。


    std::optional<ScalarType> dtype,

# 可选参数 `dtype`，表示数据类型。


    bool use_dynamic) {

# 布尔值参数 `use_dynamic`，表示是否使用动态计算图。


  TORCH_CHECK(
      false,
      "torch.quantized_lstm with List[Tensor] for parameters is "
      "no longer supported. Please re-export your model "
      "using the newer definitions in torch.jit.quantized");

# 使用 `TORCH_CHECK` 断言，如果条件为假，则抛出错误，提示不再支持使用 `List[Tensor]` 作为参数的 `torch.quantized_lstm`。建议重新导出模型，使用 `torch.jit.quantized` 中的新定义。
}

// 定义一个静态函数 quantized_lstm_data，用于执行量化 LSTM 的计算
static std::tuple<Tensor, Tensor, Tensor> quantized_lstm_data(
    // 输入参数：输入数据张量、批次大小张量、初始隐藏状态列表、单元参数列表、是否有偏置、层数、dropout 概率、是否训练、是否双向、数据类型、是否使用动态图
    const Tensor& data,
    const Tensor& batch_sizes,
    c10::List<at::Tensor> hx_,
    c10::List<c10::intrusive_ptr<CellParamsBase>> _params_,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    std::optional<ScalarType> dtype,
    bool use_dynamic) {

  // 将初始隐藏状态列表转换为标准的 C++ 向量
  auto hx = hx_.vec();

  // 创建一个容器用于存储单元参数的封装类
  std::vector<QRNNCellParamsWrapper> params;
  params.reserve(_params_.size());

  // 将单元参数列表转换为封装类并存储到 params 容器中
  for (const auto& param : _params_) {
    params.emplace_back(static_cast<c10::intrusive_ptr<CellParamsBase>>(param));
  }

  // 检查初始隐藏状态列表的大小是否为 2，确保符合 LSTM 的预期
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");

  // 检查隐藏状态的最后一个维度是否相同，用于确保投影量化 LSTM 不受支持
  TORCH_CHECK(hx[0].size(2) == hx[1].size(2), "quantized LSTM with projections is not supported");

  // 确定结果张量的数据类型，默认为 kChar 类型
  auto result_dtype = dtype.has_value() ? dtype.value() : at::kChar;

  // 创建输入数据的 PackedSequence 对象
  PackedSequence input { data, batch_sizes };

  // 声明一个包含三个张量的结果元组
  std::tuple<PackedSequence, Tensor, Tensor> results;

  // 根据结果数据类型调用相应的量化 LSTM 实现函数
  if (result_dtype == at::kChar || result_dtype == at::kQInt8) {
    // 如果使用动态图，则调用相应的量化 LSTM 实现
    if (use_dynamic) {
      results = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
          input, params, hx[0], hx[1], num_layers,
          dropout_p, train, bidirectional);
    } else {
      // 否则，仍然调用相同的量化 LSTM 实现
      results = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
          input, params, hx[0], hx[1], num_layers,
          dropout_p, train, bidirectional);
    }
  } else {
    // 对于其他数据类型，仍然使用相同的量化 LSTM 实现
    results = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
        input, params, hx[0], hx[1], num_layers,
        dropout_p, train, bidirectional);
  }

  // 获取结果元组中的打包输出对象
  auto & packed_output = std::get<0>(results);

  // 返回三个张量的元组，分别为打包输出的数据、隐藏状态的结果
  return std::make_tuple(std::move(packed_output.data),
                         std::move(std::get<1>(results)),
                         std::move(std::get<2>(results)));
}

// 定义一个静态函数 quantized_lstm_data_legacy，用于处理不再支持的参数形式
static std::tuple<Tensor, Tensor, Tensor> quantized_lstm_data_legacy(
    // 输入参数：输入数据张量、批次大小张量、初始隐藏状态列表、单元参数列表、是否有偏置、层数、dropout 概率、是否训练、是否双向、数据类型、是否使用动态图
    const Tensor& data,
    const Tensor& batch_sizes,
    c10::List<at::Tensor> hx_,
    c10::List<at::Tensor> _params_,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    std::optional<ScalarType> dtype,
    bool use_dynamic) {

  // 报错：不再支持使用 List[Tensor] 的参数形式，推荐使用新的 torch.jit.quantized 定义重新导出模型
  TORCH_CHECK(
      false,
      "torch.quantized_lstm with List[Tensor] for parameters is "
      "no longer supported. Please re-export your model "
      "using the newer definitions in torch.jit.quantized");
}

// 定义一个宏，用于定义量化 RNN 单元函数
#define DEFINE_QUANTIZED_RNN_CELL(name, hx_type, cell_type, return_type, prepare_hx_fn) \
return_type name( \
    // 输入参数：输入数据、隐藏状态、输入权重、隐藏权重、输入偏置、隐藏偏置、打包输入权重、打包隐藏权重、列偏移输入、列偏移隐藏、输入量化参数、隐藏量化参数、输入零点、
    const Tensor& input, \
    hx_type hx, \
    const Tensor& w_ih, \
    const Tensor& w_hh, \
    const Tensor& b_ih, \
    const Tensor& b_hh, \
    const Tensor& packed_ih, \
    const Tensor& packed_hh, \
    const Tensor& col_offsets_ih, \
    const Tensor& col_offsets_hh, \
    const Scalar& scale_ih, \
    const Scalar& scale_hh, \
    const Scalar& zero_point_ih,
    const Scalar& zero_point_hh) { \  # 定义函数开始，接受多个参数，其中包括标量类型的零点偏移量
  QuantizedCellParams params( \  # 创建量化的单元参数对象，使用传入的多个参数初始化
      w_ih, \  # 输入权重矩阵
      w_hh, \  # 隐藏状态权重矩阵
      b_ih, \  # 输入偏置向量
      b_hh, \  # 隐藏状态偏置向量
      packed_ih, \  # 打包后的输入权重数据
      packed_hh, \  # 打包后的隐藏状态权重数据
      col_offsets_ih, \  # 输入权重的列偏移量
      col_offsets_hh, \  # 隐藏状态权重的列偏移量
      scale_ih, \  # 输入权重的缩放因子
      scale_hh, \  # 隐藏状态权重的缩放因子
      zero_point_ih, \  # 输入权重的零点偏移量
      zero_point_hh); \  # 隐藏状态权重的零点偏移量
  return cell_type{}( \  # 调用具体的单元类型对象，传入输入数据、初始化后的参数对象和隐藏状态数据的准备函数结果
      input, prepare_hx_fn(hx), params); \  # 输入数据、隐藏状态的准备函数应用后的隐藏状态、初始化的参数对象
// 定义宏 `DEFINE_QUANTIZED_RNN_CELL_DYNAMIC`，用于声明一个量化的 RNN 单元函数
#define DEFINE_QUANTIZED_RNN_CELL_DYNAMIC(name, hx_type, cell_type, return_type, prepare_hx_fn) \
return_type name( \
    const Tensor& input, \
    hx_type hx, \
    c10::intrusive_ptr<LinearPackedParamsBase> _packed_w_ih, \
    c10::intrusive_ptr<LinearPackedParamsBase> _packed_w_hh, \
    const Tensor& b_ih, \
    const Tensor& b_hh \
 ) { \
  // 创建动态参数对象 `QuantizedCellParamsDynamic`，初始化参数 `_packed_w_ih`、`_packed_w_hh`、`b_ih`、`b_hh`，设置 `true` 作为 `reduce_range` 标志
  QuantizedCellParamsDynamic params( \
      _packed_w_ih, \
      _packed_w_hh, \
      b_ih, \
      b_hh,\
      true); \
  // 调用特定类型的 RNN 单元函数 `cell_type`，传入 `input`、准备好的 `hx` 和 `params`，并返回结果
  return cell_type{}( \
      input, prepare_hx_fn(hx), params); \
}

// 定义量化 LSTM 单元类型及其返回类型
using quantized_lstm_cell_type = LSTMCell<QuantizedCellParams>;
using quantized_lstm_return_type = std::tuple<Tensor, Tensor>;

// 准备量化 LSTM 单元的隐藏状态 `hx`
static std::tuple<Tensor, Tensor> prepare_quantized_lstm_hx(TensorList hx) {
  // 返回隐藏状态的第一个和第二个张量作为 tuple
  return std::make_tuple(hx[0], hx[1]);
}

// 定义动态版本的量化 LSTM 单元类型
using quantized_lstm_cell_dynamic_type = LSTMCell<QuantizedCellParamsDynamic>;

// 定义 `quantized_lstm_cell_dynamic` 函数，使用 `DEFINE_QUANTIZED_RNN_CELL_DYNAMIC` 宏声明
DEFINE_QUANTIZED_RNN_CELL_DYNAMIC(quantized_lstm_cell_dynamic, TensorList, quantized_lstm_cell_dynamic_type, quantized_lstm_return_type, prepare_quantized_lstm_hx);

// 辅助函数 `prepare_quantized_hx`，用于简化 RNN 单元的隐藏状态
using simple_hx_type = const Tensor&;
static simple_hx_type prepare_quantized_hx(simple_hx_type hx) {
  // 直接返回输入的隐藏状态 `hx`
  return hx;
}

// 定义量化 GRU 单元类型
using quantized_gru_cell_type = GRUCell<QuantizedCellParams>;
using quantized_gru_cell_dynamic_type = GRUCell<QuantizedCellParamsDynamic>;

// 定义 `quantized_gru_cell` 函数，使用 `DEFINE_QUANTIZED_RNN_CELL` 宏声明
DEFINE_QUANTIZED_RNN_CELL(quantized_gru_cell, simple_hx_type, quantized_gru_cell_type, Tensor, prepare_quantized_hx);

// 定义动态版本的量化 GRU 单元类型及函数 `quantized_gru_cell_dynamic`
static DEFINE_QUANTIZED_RNN_CELL_DYNAMIC(quantized_gru_cell_dynamic, simple_hx_type, quantized_gru_cell_dynamic_type, Tensor, prepare_quantized_hx);

// 定义量化带有 ReLU 的 RNN 单元类型
using quantized_rnn_relu_cell_type = SimpleCell<relu_f, QuantizedCellParams>;

// 定义 `quantized_rnn_relu_cell` 函数，使用 `DEFINE_QUANTIZED_RNN_CELL` 宏声明
DEFINE_QUANTIZED_RNN_CELL(quantized_rnn_relu_cell, simple_hx_type, quantized_rnn_relu_cell_type, Tensor, prepare_quantized_hx);

// 定义动态版本的量化带有 ReLU 的 RNN 单元类型及函数 `quantized_rnn_relu_cell_dynamic`
using quantized_rnn_relu_cell_dynamic_type = SimpleCell<relu_f, QuantizedCellParamsDynamic>;
static DEFINE_QUANTIZED_RNN_CELL_DYNAMIC(quantized_rnn_relu_cell_dynamic, simple_hx_type, quantized_rnn_relu_cell_dynamic_type, Tensor, prepare_quantized_hx);

// 定义量化带有 tanh 的 RNN 单元类型
using quantized_rnn_tanh_cell_type = SimpleCell<tanh_f, QuantizedCellParams>;

// 定义 `quantized_rnn_tanh_cell` 函数，使用 `DEFINE_QUANTIZED_RNN_CELL` 宏声明
DEFINE_QUANTIZED_RNN_CELL(quantized_rnn_tanh_cell, simple_hx_type, quantized_rnn_tanh_cell_type, Tensor, prepare_quantized_hx);

// 定义动态版本的量化带有 tanh 的 RNN 单元类型及函数 `quantized_rnn_tanh_cell_dynamic`
using quantized_rnn_tanh_cell_dynamic_type = SimpleCell<tanh_f, QuantizedCellParamsDynamic>;
static DEFINE_QUANTIZED_RNN_CELL_DYNAMIC(quantized_rnn_tanh_cell_dynamic, simple_hx_type, quantized_rnn_tanh_cell_dynamic_type, Tensor, prepare_quantized_hx);
# 静态变量，用于确保线性参数已注册
static C10_UNUSED auto ensure_linear_params_registered = register_linear_params();

# 定义了一个静态变量 cell_params_base_registry，其类型为 torch::selective_class_<CellParamsBase>
# 这个变量在 "rnn" 命名空间下注册了 CellParamsBase 类型，并定义了其序列化和反序列化方法
static auto cell_params_base_registry =
    torch::selective_class_<CellParamsBase>("rnn", TORCH_SELECTIVE_CLASS("CellParamsBase"))
        .def_pickle(
            # 序列化函数，将 CellParamsBase 指针自身状态序列化为 CellParamsSerializationType
            [](const c10::intrusive_ptr<CellParamsBase>& self)
                -> CellParamsSerializationType { return self->__getstate__(); },
            # 反序列化函数，接受状态并返回 CellParamsBase 指针
            [](CellParamsSerializationType state)
                -> c10::intrusive_ptr<CellParamsBase> {
              # 从状态中获取类型信息
              std::string type = std::get<0>(state);
              # 使用 TORCH_INTERNAL_ASSERT 确保已注册指定类型的反序列化函数
              TORCH_INTERNAL_ASSERT(cell_params_deserializers.count(type));
              # 调用相应的反序列化器来创建 CellParamsBase 对象
              return cell_params_deserializers[type](std::move(state));
            });

# 定义了一个 TORCH_LIBRARY_FRAGMENT 块，用于注册 ATen 操作的量化 LSTM 和 GRU 实现
TORCH_LIBRARY_FRAGMENT(aten, m) {
  # 定义了 quantized_lstm.input 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_lstm.input(Tensor input, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
  # 定义了 quantized_lstm.data 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
  # 定义了 quantized_lstm.input_legacy 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
  # 定义了 quantized_lstm.data_legacy 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"));
  # 定义了 quantized_gru.input 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_gru.input(Tensor input, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"));
  # 定义了 quantized_gru.data 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"));
  # 定义了 quantized_gru.input_legacy 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"));
  # 定义了 quantized_gru.data_legacy 操作的 schema
  m.def(
      TORCH_SELECTIVE_SCHEMA("aten::quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"));
}
// 定义 TORCH_LIBRARY_FRAGMENT 宏，用于定义 quantized 模块中的函数
TORCH_LIBRARY_FRAGMENT(quantized, m) {
  // 定义 quantized::make_quantized_cell_params_dynamic 函数，接受 quantized.LinearPackedParamsBase 类型的参数和 Tensor 类型的参数，返回 rnn.CellParamsBase 类型对象
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::make_quantized_cell_params_dynamic(__torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh, bool reduce_range=False) -> __torch__.torch.classes.rnn.CellParamsBase"));
  // 定义 quantized::make_quantized_cell_params_fp16 函数，接受 quantized.LinearPackedParamsBase 类型的参数，返回 rnn.CellParamsBase 类型对象
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::make_quantized_cell_params_fp16(__torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh) -> __torch__.torch.classes.rnn.CellParamsBase"));
  // 定义 quantized::make_quantized_cell_params 函数，接受 Tensor 类型的参数，返回 rnn.CellParamsBase 类型对象
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::make_quantized_cell_params(Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> __torch__.torch.classes.rnn.CellParamsBase"));
  // 定义 quantized::quantized_lstm_cell_dynamic 函数，接受 Tensor 和 quantized.LinearPackedParamsBase 类型的参数，返回两个 Tensor 类型对象的元组
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_lstm_cell_dynamic(Tensor input, Tensor[] hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh) -> (Tensor, Tensor)"));
  // 定义 quantized::quantized_gru_cell_dynamic 函数，接受 Tensor 类型的参数，返回 Tensor 类型对象
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_gru_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor"));
  // 定义 quantized::quantized_rnn_relu_cell_dynamic 函数，接受 Tensor 类型的参数，返回 Tensor 类型对象
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_rnn_relu_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor"));
  // 定义 quantized::quantized_rnn_tanh_cell_dynamic 函数，接受 Tensor 类型的参数，返回 Tensor 类型对象
  m.def(TORCH_SELECTIVE_SCHEMA("quantized::quantized_rnn_tanh_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor"));
}

// 定义 TORCH_LIBRARY_IMPL 宏，用于实现 aten 模块中的函数，限定为 CPU
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  // 实现 aten::quantized_lstm.input 函数，使用 quantized_lstm_input 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_lstm.input"), TORCH_FN(quantized_lstm_input));
  // 实现 aten::quantized_lstm.data 函数，使用 quantized_lstm_data 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_lstm.data"), TORCH_FN(quantized_lstm_data));
  // 实现 aten::quantized_lstm.input_legacy 函数，使用 quantized_lstm_input_legacy 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_lstm.input_legacy"), TORCH_FN(quantized_lstm_input_legacy));
  // 实现 aten::quantized_lstm.data_legacy 函数，使用 quantized_lstm_data_legacy 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_lstm.data_legacy"), TORCH_FN(quantized_lstm_data_legacy));
  // 实现 aten::quantized_gru.input 函数，使用 quantized_gru_input 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_gru.input"), TORCH_FN(quantized_gru_input));
  // 实现 aten::quantized_gru.data 函数，使用 quantized_gru_data 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_gru.data"), TORCH_FN(quantized_gru_data));
  // 实现 aten::quantized_gru.input_legacy 函数，使用 quantized_gru_input_legacy 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_gru.input_legacy"), TORCH_FN(quantized_gru_input_legacy));
  // 实现 aten::quantized_gru.data_legacy 函数，使用 quantized_gru_data_legacy 函数实现
  m.impl(TORCH_SELECTIVE_NAME("aten::quantized_gru.data_legacy"), TORCH_FN(quantized_gru_data_legacy));
}
// 在 quantized 库中注册 CPU 上的量化操作的实现
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // 注册 quantized::make_quantized_cell_params_dynamic 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::make_quantized_cell_params_dynamic"), TORCH_FN(make_quantized_cell_params_dynamic));
  // 注册 quantized::make_quantized_cell_params 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::make_quantized_cell_params"), TORCH_FN(make_quantized_cell_params));
  // 注册 quantized::quantized_lstm_cell_dynamic 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_lstm_cell_dynamic"), TORCH_FN(quantized_lstm_cell_dynamic));
  // 注册 quantized::quantized_gru_cell_dynamic 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_gru_cell_dynamic"), TORCH_FN(quantized_gru_cell_dynamic));
  // 注册 quantized::quantized_rnn_relu_cell_dynamic 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_rnn_relu_cell_dynamic"), TORCH_FN(quantized_rnn_relu_cell_dynamic));
  // 注册 quantized::quantized_rnn_tanh_cell_dynamic 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_rnn_tanh_cell_dynamic"), TORCH_FN(quantized_rnn_tanh_cell_dynamic));
}

// 在 quantized 库中注册所有 CPU 以外的量化操作的实现
TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // 注册 quantized::make_quantized_cell_params_fp16 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::make_quantized_cell_params_fp16"), TORCH_FN(make_quantized_cell_params_fp16));
}

// 结束 quantized 命名空间
} // namespace quantized

// 结束 at::native 命名空间
}  // namespace at::native
```