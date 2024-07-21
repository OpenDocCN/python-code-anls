# `.\pytorch\aten\src\ATen\native\quantized\cpu\OnednnUtils.h`

```
#pragma once
// 预处理指令，确保此头文件只包含一次

#include <ATen/Config.h>
// 引入 ATen 库的配置信息

#if AT_MKLDNN_ENABLED()
// 如果 ATen 使用了 MKL-DNN 库

#include <ATen/Tensor.h>
// 引入 ATen 的 Tensor 类
#include <ATen/native/quantized/PackedParams.h>
// 引入 ATen 中量化的参数封装
#include <ideep.hpp>
// 引入 Intel 的 deep learning 库
#include <cpuinfo.h>
// 引入 CPU 信息查询库

#include <c10/util/CallOnce.h>
// 引入 C10 实用工具中的一次调用功能

using PrimitiveCacheKey = std::tuple<
    double, // 输入的缩放因子
    int64_t, // 输入的零点
    std::vector<int64_t>, // 输入的形状
    double, // 输出的缩放因子
    int64_t, // 输出的零点
    int64_t, // OMP 线程数
    double, // 累积的缩放因子
    int64_t>; // 累积的零点

enum CacheKeyIndex {
  InputScale, // 输入的缩放因子在元组中的索引
  InputZeroPoint, // 输入的零点在元组中的索引
  InputShape, // 输入的形状在元组中的索引
  OutputScale, // 输出的缩放因子在元组中的索引
  OutputZeroPoint, // 输出的零点在元组中的索引
  NumOfThreads, // OMP 线程数在元组中的索引
};

// 基本的原语缓存类
struct PrimitiveCache {
  PrimitiveCacheKey key; // 缓存的键值对

  // 检查是否命中缓存
  bool hit(const PrimitiveCacheKey& key) {
    return this->key == key;
  }
};

// 使用 ideep 实现的线性运算参数
using LinearParams = ideep::matmul_forward_params;
// 使用 dnnl 实现的卷积前向操作
using Conv = dnnl::convolution_forward;
// 使用 dnnl 实现的卷积前向操作的描述
using ConvDesc = dnnl::convolution_forward::primitive_desc;
// 使用 ideep 实现的卷积前向参数
using ConvParams = ideep::convolution_forward_params;
// 使用 dnnl 实现的反卷积前向操作
using Deconv = dnnl::deconvolution_forward;
// 使用 dnnl 实现的反卷积前向操作的描述
using DeconvDesc = dnnl::deconvolution_forward::primitive_desc;
// 使用 ideep 实现的反卷积前向参数
using DeconvParams = ideep::deconv_forward_params;

// 使用 ideep 实现的线性原语缓存类
struct LinearPrimitiveCache : PrimitiveCache {
  LinearPrimitiveCache() {}

  LinearPrimitiveCache(
      const PrimitiveCacheKey& key,
      const LinearParams& param) {
    this->key = key;
    this->param = param;
  }

  LinearParams param; // 线性参数

  // 用于动态量化线性操作，缩放因子和零点在执行时设置
  // 只需比较键的其余部分
  bool hit_dynamic(const PrimitiveCacheKey& new_key) {
    auto cached_input_shape = std::get<InputShape>(this->key);
    auto new_input_shape = std::get<InputShape>(new_key);
    return (
        cached_input_shape == new_input_shape &&
        std::get<NumOfThreads>(this->key) == std::get<NumOfThreads>(new_key));
  }

  // 获取线性参数
  LinearParams& get_param() {
    return param;
  }
};

// 使用 ideep 实现的卷积原语缓存类
struct ConvPrimitiveCache : PrimitiveCache {
  ConvPrimitiveCache() {}

  ConvPrimitiveCache(
      const PrimitiveCacheKey& key,
      const ConvParams& params) {
    this->key = key;
    this->params = params;
  }

  ConvParams params; // 卷积参数

  // 获取卷积参数
  ConvParams& get_params() {
    return params;
  }
};

// 使用 ideep 实现的反卷积原语缓存类
struct DeconvPrimitiveCache : PrimitiveCache {
  DeconvPrimitiveCache() {}

  DeconvPrimitiveCache(
      const PrimitiveCacheKey& key,
      const DeconvParams& params) {
    this->key = key;
    this->params = params;
  }

  DeconvParams params; // 反卷积参数

  // 获取反卷积参数
  DeconvParams& get_params() {
    return params;
  }
};

// 后操作的枚举类型
enum PostOps {
  NoPostOp, // 无后操作
  Relu, // ReLU 后操作
  LeakyRelu, // Leaky ReLU 后操作
  Tanh, // Tanh 后操作
  Gelu // Gelu 后操作
};

// 使用 OneDNN 实现的打包线性权重类，继承自线性打包参数基类
struct PackedLinearWeightsOnednn : public LinearPackedParamsBase {
  PackedLinearWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      std::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      std::optional<at::Tensor> orig_bias)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        orig_weight_(std::move(orig_weight)),
        orig_bias_(std::move(orig_bias)) {
    cache_initialized_flag = std::make_unique<c10::once_flag>();



    // 创建一个唯一的指针，用于标识缓存是否已初始化的标志
    cache_initialized_flag = std::make_unique<c10::once_flag>();
  }



  std::unique_ptr<ideep::tensor> weight_;



  // 用于保存权重的唯一指针，可能为空
  std::unique_ptr<ideep::tensor> weight_;



  std::optional<ideep::tensor> bias_;



  // 可选的偏置张量，可能为空
  std::optional<ideep::tensor> bias_;



  at::Tensor orig_weight_;



  // 原始权重张量
  at::Tensor orig_weight_;



  std::optional<at::Tensor> orig_bias_;



  // 可选的原始偏置张量，可能为空
  std::optional<at::Tensor> orig_bias_;



  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;



  // 应用函数，接受输入张量以及输出的比例因子和零点
  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;



  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;



  // 应用 ReLU 激活函数，接受输入张量以及输出的比例因子和零点
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;



  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;



  // 应用动态范围量化，接受输入张量和是否减少范围的标志
  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;



  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;



  // 应用动态范围量化和 ReLU 激活函数，接受输入张量和是否减少范围的标志
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;



  at::Tensor apply_leaky_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point,
      double negative_slope);



  // 应用 Leaky ReLU 激活函数，接受输入张量、输出比例因子、零点和负斜率
  at::Tensor apply_leaky_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point,
      double negative_slope);



  at::Tensor apply_tanh(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);



  // 应用 Tanh 激活函数，接受输入张量、输出比例因子和零点
  at::Tensor apply_tanh(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);



  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;



  // 解包函数，返回权重张量和可选的偏置张量
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;



  std::optional<at::Tensor> bias() override {
    return orig_bias_;
  }



  // 获取偏置张量的可选版本，如果不存在偏置则返回空
  std::optional<at::Tensor> bias() override {
    return orig_bias_;
  }



  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);



  // 预打包函数，接受权重张量和可选的偏置张量作为参数
  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);



  LinearPrimitiveCache prim_cache;



  // 线性基本缓存对象
  LinearPrimitiveCache prim_cache;



  std::unique_ptr<c10::once_flag> cache_initialized_flag;



  // 唯一的指针，用于标识缓存是否已初始化的标志
  std::unique_ptr<c10::once_flag> cache_initialized_flag;



  template <PostOps post_op>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point,
      torch::List<at::Scalar> post_op_args = torch::List<at::Scalar>());



  // 实现应用特定后操作的函数模板
  template <PostOps post_op>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point,
      torch::List<at::Scalar> post_op_args = torch::List<at::Scalar>());



  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range=false);



  // 实现动态范围量化的函数模板
  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range=false);



  LinearPrimitiveCache& get_cache() {
    return prim_cache;
  }



  // 获取线性基本缓存对象的引用
  LinearPrimitiveCache& get_cache() {
    return prim_cache;
  }
  // 结构体 PackedConvWeightsOnednn 继承自 ConvPackedParamsBase，用于封装卷积操作的参数
  // 默认空间维度为 2

  PackedConvWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      std::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      std::optional<at::Tensor> orig_bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      uint8_t transpose)
      : weight_(std::move(weight)),  // 初始化权重的唯一指针
        bias_(std::move(bias)),      // 初始化可选的偏置
        orig_weight_(std::move(orig_weight)),  // 初始化原始权重张量
        orig_bias_(std::move(orig_bias)),      // 初始化原始偏置张量（可选）
        stride_(std::move(stride)),    // 初始化步幅列表
        padding_(std::move(padding)),  // 初始化填充列表
        output_padding_(std::move(output_padding)),  // 初始化输出填充列表
        dilation_(std::move(dilation)),  // 初始化膨胀列表
        groups_(groups),                 // 初始化组数
        transpose_(transpose) {          // 初始化转置标志
    cache_initialized_flag = std::make_unique<c10::once_flag>();  // 创建缓存初始化标志
  }

  std::unique_ptr<ideep::tensor> weight_;   // 唯一指针，用于存储权重张量
  std::optional<ideep::tensor> bias_;       // 可选的偏置张量
  at::Tensor orig_weight_;                  // 原始权重张量
  std::optional<at::Tensor> orig_bias_;     // 可选的原始偏置张量
  torch::List<int64_t> stride_;             // 步幅列表
  torch::List<int64_t> padding_;            // 填充列表
  torch::List<int64_t> output_padding_;     // 输出填充列表
  torch::List<int64_t> dilation_;           // 膨胀列表
  int64_t groups_;                          // 卷积组数
  uint8_t transpose_;                       // 转置标志

  // 应用卷积操作并返回结果张量
  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用带 ReLU 激活的卷积操作并返回结果张量
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用动态范围的卷积操作并返回结果张量
  at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range) override;

  // 应用带加法操作的卷积，并返回结果张量
  at::Tensor apply_add(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

  // 应用带加法和 ReLU 操作的卷积，并返回结果张量
  at::Tensor apply_add_relu(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

  // 解压封装的参数，返回原始权重张量和可选的原始偏置张量
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // 静态方法，预封装卷积参数并返回 ConvPackedParamsBase 的指针
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  // 返回步幅列表
  torch::List<int64_t> stride() const override {
    return stride_;
  }

  // 返回填充列表
  torch::List<int64_t> padding() const override {
    return padding_;
  }

  // 返回输出填充列表
  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  // 返回膨胀列表
  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  // 返回卷积组数
  int64_t groups() const override {
    return groups_;
  }

  // 返回转置标志
  bool transpose() const override {
    // 返回当前 transpose_ 变量的布尔值
    return (bool)transpose_;
  }

 private:
  // 用于存储卷积操作的缓存
  ConvPrimitiveCache conv_prim_cache;
  // 用于存储反卷积操作的缓存
  DeconvPrimitiveCache deconv_prim_cache;
  // 用于确保缓存只被初始化一次的标志
  std::unique_ptr<c10::once_flag> cache_initialized_flag;

  // 模板方法，实现卷积操作（可选是否与 ReLU 融合）
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      const std::optional<at::Tensor>& accum,
      double output_scale,
      int64_t output_zero_point);

  // 获取卷积操作的缓存对象
  ConvPrimitiveCache& get_conv_cache() {
    // 断言当前没有进行转置操作
    assert(!transpose());
    return conv_prim_cache;
  }

  // 获取反卷积操作的缓存对象
  DeconvPrimitiveCache& get_deconv_cache() {
    // 断言当前进行了转置操作
    assert(transpose());
    return deconv_prim_cache;
  }


这段代码是一个 C++ 类的片段，具有以下功能和结构：

1. `return (bool)transpose_;`：返回当前对象的 `transpose_` 变量的布尔值。

2. `ConvPrimitiveCache conv_prim_cache;` 和 `DeconvPrimitiveCache deconv_prim_cache;`：分别声明了用于存储卷积和反卷积操作的缓存对象。

3. `std::unique_ptr<c10::once_flag> cache_initialized_flag;`：用于确保缓存只被初始化一次的标志。

4. `template <bool ReluFused> at::Tensor apply_impl(...);`：声明了一个模板方法，用于实现卷积操作，可能会与 ReLU 函数融合。

5. `ConvPrimitiveCache& get_conv_cache()` 和 `DeconvPrimitiveCache& get_deconv_cache()`：分别是获取卷积和反卷积操作缓存对象的方法。这些方法在获取缓存之前会使用 `assert` 断言来确保对象处于正确的状态（未转置或已转置）。

每个注释都在细致说明每行代码的功能和意图，确保代码的功能和维护清晰易懂。
};



namespace onednn_utils {

// 在 `onednn_utils` 命名空间中定义了一个工具函数或类


inline ideep::attr_t create_attr_by_post_op(
    const c10::string_view& binary_post_op,
    double binary_alpha,
    double input1_scale,
    int64_t input1_zero_point,
    const ideep::tensor::desc& input1_desc,
    const c10::string_view& unary_post_op,
    const torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    const c10::string_view& unary_post_op_algorithm) {

// 定义了一个名为 `create_attr_by_post_op` 的内联函数，接受多个参数并返回 `ideep::attr_t` 类型对象


  using ideep::tensor;

// 引入 `ideep::tensor` 命名空间，可能用于更方便地访问其中的符号


  if (binary_post_op == "none") {

// 如果 `binary_post_op` 参数的值等于 "none"，执行以下条件分支


    if (unary_post_op == "relu") {
      return ideep::attr_t::fuse_relu();
    } else if (unary_post_op == "leaky_relu") {
      TORCH_CHECK(
          unary_post_op_args.size() == 1,
          "onednn qlinear: expect one argument for post op leaky_relu but got ", unary_post_op_args.size(), " args");
      auto alpha = unary_post_op_args[0].value().to<float>();
      return ideep::attr_t::fuse_relu_v2(alpha);
    } else if (unary_post_op == "tanh") {
      return ideep::attr_t::fuse_tanh();
    } else if (unary_post_op == "gelu") {
      TORCH_CHECK(
          unary_post_op_algorithm == "none" || unary_post_op_algorithm == "tanh",
          "onednn qlinear: algorithm for post op gelu must be none or tanh but got ", unary_post_op_algorithm);
      auto post_algorithm = unary_post_op_algorithm == "none" ?
        dnnl::algorithm::eltwise_gelu_erf :
        dnnl::algorithm::eltwise_gelu_tanh;
      return ideep::attr_t::fuse_gelu_v2(0.f, 0.f, post_algorithm);
    } else if (unary_post_op == "hardtanh") {
      TORCH_CHECK(
          unary_post_op_args.size() == 2 &&
              unary_post_op_args[0].has_value() &&
              unary_post_op_args[1].has_value(),
          "hardtanh is expected to have two scalar input: min_val and max_val");
      auto lower_bound_value =
          unary_post_op_args[0].value().to<float>();
      auto upper_bound_value =
          unary_post_op_args[1].value().to<float>();
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    } else if (unary_post_op == "hardswish") {
      return ideep::attr_t::fuse_hardswish();
    } else if (unary_post_op == "swish") {
      return ideep::attr_t::fuse_swish();
    } else {
      TORCH_CHECK(
          unary_post_op == "none",
          "onednn qlinear: unsupported unary post op ", unary_post_op);
    }

// 在 `binary_post_op` 为 "none" 时，根据 `unary_post_op` 的值返回相应的 IDEEP 属性对象，用于后续操作的融合或处理


  } else if (binary_post_op == "sum") {

// 如果 `binary_post_op` 参数的值等于 "sum"，执行以下条件分支


    if (unary_post_op == "none") {
      return ideep::attr_t::fuse_sum(input1_scale, input1_zero_point);
    } else if (unary_post_op == "relu") {
      return ideep::attr_t::residual_with_sum_zero_point(input1_scale, input1_zero_point);
    } else {
      TORCH_CHECK(
          false,
          "onednn qlinear: unsupported unary post op ", unary_post_op, " with binary post op sum");
    }

// 在 `binary_post_op` 为 "sum" 时，根据 `unary_post_op` 的值返回相应的 IDEEP 属性对象，用于后续操作的融合或处理


  } else if (binary_post_op == "add") {

// 如果 `binary_post_op` 参数的值等于 "add"，执行以下条件分支


    if (unary_post_op == "none") {
      return ideep::attr_t::fuse_binary(ideep::algorithm::binary_add, input1_desc);

// 在 `binary_post_op` 为 "add" 且 `unary_post_op` 为 "none" 时，返回一个包含指定参数的 IDEEP 属性对象
    } else if (unary_post_op == "relu") {
      // 创建一个后处理操作对象
      ideep::post_ops po;
      // 向后处理操作对象中添加一个二元操作（二元加法），使用输入描述符
      po.append_binary(ideep::algorithm::binary_add, input1_desc);
      // 向后处理操作对象中添加一个逐元素操作（ReLU），参数为 0 和 0
      po.append_eltwise(ideep::algorithm::eltwise_relu, 0, 0);
      // 返回包含上述后处理操作对象的属性对象
      return ideep::attr_t::attr_post_ops(po);
    } else {
      // 如果unary_post_op不是"relu"，抛出错误并显示不支持的操作类型及详细信息
      TORCH_CHECK(
          false,
          "onednn qlinear: unsupported unary post op ", unary_post_op, " with binary post op add");
    }
  } else {
    // 如果binary_post_op不被支持，抛出错误并显示不支持的操作类型
    TORCH_CHECK(
        false,
        "onednn qlinear: unsupported binary post op ", binary_post_op);
  }
  // 如果以上条件均不满足，则返回一个空的属性对象
  return ideep::attr_t();
}

// onednn_utils 命名空间的结束

// ONEDNN 要求权重的对称量化
// 使用此实用函数进行检查。
inline bool is_weight_symmetric_quant(
      const at::Tensor& weight,
      bool is_transposed_conv) {
  bool is_symmetric = true;
  const auto qtype = weight.qscheme();
  // 如果是 PerTensorAffine 量化方案
  if (qtype == c10::kPerTensorAffine) {
    // 检查权重的零点是否为 0
    is_symmetric &= (weight.q_zero_point() == 0);
  } else if (qtype == c10::kPerChannelAffine) {
    // 如果是 PerChannelAffine 量化方案
    if (is_transposed_conv) {
      // 在 PyTorch 中不支持转置卷积的情况
      // 但在此实用函数中不会引发错误。
      is_symmetric = false;
    } else {
      // 获取输出通道数
      auto output_channels = weight.size(0);
      // 逐个检查每个通道的零点是否为 0
      for (int i = 0; i < output_channels; ++i) {
        auto zp = weight.q_per_channel_zero_points()[i].item<int32_t>();
        is_symmetric &= (zp == 0);
      }
    }
  } else {
    // 在 PyTorch 中不支持的量化方案
    // 但在此实用函数中不会引发错误。
    is_symmetric = false;
  }
  return is_symmetric;
}

// 当 qengine 是 x86 时，使用此实用函数来检查是否优选 onednn 内核
// 而非 fbgemm 内核以获得更好的性能。
inline bool should_use_onednn_quant(
    const at::Tensor& weight,
    bool is_transposed_conv,
    int groups,
    torch::List<int64_t> output_padding) {
  // 目前仅在 Linux 上验证 onednn 的性能
  // 调度的启发式方法基于在 Linux 上的性能数据。
  // 因此，对于 x86 qengine，如果操作系统不是 Linux，则始终使用 fbgemm 内核。
  // TODO 支持更多操作系统。
#if !defined(__linux__)
  return false;
#else
  // 检查是否支持 VNNI 指令集
  bool vnni_available = cpuinfo_has_x86_avx512vnni();
  // 检查权重是否对称量化
  bool w_sym_quant =
      is_weight_symmetric_quant(weight, is_transposed_conv);
  // 检查输出填充是否全部为 0
  bool opad_all_zero =
      std::all_of(output_padding.begin(), output_padding.end(), [](int i) { return i==0; });
  // 返回条件是否满足 onednn 内核优选的结果
  return vnni_available && (groups <= 100) && w_sym_quant && opad_all_zero;
#endif
}

#endif // 如果 AT_MKLDNN_ENABLED() 宏被定义
```