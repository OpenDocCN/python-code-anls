# `.\pytorch\aten\src\ATen\native\mkldnn\RNN.cpp`

```
#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>

#include <ATen/TensorUtils.h>
#include <ATen/Dispatch.h>
#include <c10/core/GradMode.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mkldnn_convolution_native.h>
#include <ATen/ops/mkldnn_rnn_layer_backward_native.h>
#include <ATen/ops/mkldnn_rnn_layer_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at::native {

// 如果未启用MKLDNN支持，定义mkldnn_rnn_layer函数抛出错误
std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_layer(
    const Tensor& input,
    const Tensor& w0,
    const Tensor& w1,
    const Tensor& w2,
    const Tensor& w3,
    const Tensor& hx_,
    const Tensor& cx_,
    bool reverse,
    IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
      AT_ERROR("mkldnn_rnn_layer: ATen not compiled with MKLDNN support");
}

// 如果未启用MKLDNN支持，定义mkldnn_rnn_layer_backward函数抛出错误
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_layer_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace) {
      AT_ERROR("mkldnn_rnn_layer_backward: ATen not compiled with MKLDNN support");
}

// 注册不支持CPU分发的lstm_mkldnn_stub
REGISTER_NO_CPU_DISPATCH(lstm_mkldnn_stub);

} // namespace at::native

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at::native {

// 定义RNNParams结构体，用于存储RNN参数
struct RNNParams {
  ideep::rnn_kind mode;             // RNN的类型（LSTM、GRU等）
  int64_t seq_length;               // 序列长度
  int64_t mini_batch;               // 小批量大小
  int64_t input_size;               // 输入大小
  int64_t hidden_size;              // 隐藏层大小
  int64_t num_directions;           // 方向数量（单向或双向）
  int64_t num_layers;               // 层数
  bool batch_first;                 // 是否批量第一
  bool train;                       // 是否训练模式
  at::IntArrayRef batch_sizes;      // 批量大小数组
  int64_t num_gates;                // 门数量
  int64_t num_bias_gates;           // 偏置门数量

  // RNNParams结构体的构造函数，用于初始化参数
  RNNParams(
      const at::Tensor& input,
      at::IntArrayRef batch_sizes_,
      int64_t mode_,
      int64_t hidden_size_,
      int64_t num_layers_,
      bool bidirectional,
      bool batch_first_,
      bool train_) {
    mode = static_cast<ideep::rnn_kind>(mode_);  // 将输入的RNN类型转换为ideep::rnn_kind类型
    batch_first = batch_first_;                 // 初始化是否批量第一
    seq_length = input.size(0);                 // 初始化序列长度
    mini_batch = input.size(1);                 // 初始化小批量大小
    input_size = input.size(2);                 // 初始化输入大小
    hidden_size = hidden_size_;                 // 初始化隐藏层大小
    num_directions = bidirectional ? 2 : 1;      // 根据是否双向设置方向数量
    num_layers = num_layers_;                   // 初始化层数
    train = train_;                             // 初始化是否训练模式
    batch_sizes = batch_sizes_;                 // 初始化批量大小数组
  }
    if (mode == ideep::rnn_kind::LSTM) {
      // 如果 RNN 模式是 LSTM，则设置门的数量为 4，偏置门的数量也为 4
      num_gates = 4;
      num_bias_gates = 4;
    } else if (mode == ideep::rnn_kind::GRU) {
      // 如果 RNN 模式是 GRU，则设置门的数量为 3，偏置门的数量为 4
      num_gates = 3;
      num_bias_gates = 4;
    } else {
      // 如果 RNN 模式是 RNN_RELU 或 RNN_TANH，则设置门的数量为 1，偏置门的数量也为 1
      num_gates = 1;
      num_bias_gates = 1;
    }
  }

  // mkldnn 内存描述符定义
  using format = ideep::format_tag;
  using desc = ideep::tensor::desc;
  using dtype = ideep::tensor::data_type;

  // 描述源层的内存格式和数据类型
  desc src_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{seq_length, mini_batch, _input_size}, dtype, format::tnc};
  }

  // 描述源迭代器的内存格式和数据类型
  desc src_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }

  // 描述源迭代器C状态的内存格式和数据类型
  desc src_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }

  // 描述权重层的内存格式和数据类型
  desc weights_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }

  // 描述权重层（使用ldigo格式）的内存格式和数据类型
  desc weights_layer_ldigo_desc(int64_t _input_size, dtype dtype) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype, format::ldigo};
  }

  // 描述权重迭代器的内存格式和数据类型
  desc weights_iter_desc(dtype dtype) const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }

  // 描述权重迭代器（使用ldigo格式）的内存格式和数据类型
  desc weights_iter_ldigo_desc(dtype dtype) const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype, format::ldigo};
  }

  // 描述偏置的内存格式和数据类型
  desc bias_desc(dtype dtype) const {
    return {{1, 1, num_bias_gates, hidden_size}, dtype, format::ldgo};
  }

  // 描述目标层的内存格式和数据类型
  desc dst_layer_desc(dtype dtype) const {
    return {{seq_length, mini_batch, hidden_size}, dtype, format::tnc};
  }

  // 描述目标迭代器的内存格式和数据类型
  desc dst_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }

  // 描述目标迭代器C状态的内存格式和数据类型
  desc dst_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
};

// 根据 RNN 参数计算输出的尺寸
template<bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams& rnn) {
  // 根据是否单向计算输出通道数
  auto output_channels = is_single_direction ? rnn.hidden_size
                                             : rnn.hidden_size * rnn.num_directions;
  // 返回包含序列长度、mini_batch 和输出通道数的向量
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// MKLDNN GRU 门的顺序与 PyTorch 的不同，需要重新排列门的顺序
// (rt,zt,nt 分别表示重置门、更新门、新门)
//
//   MKLDNN GRU weight_ih/weight_hh 门的顺序: (zt, rt, nt)
//   PyTorch GRU weight_ih/weight_hh 门的顺序: (rt, zt, nt)
//
// MKLDNN GRU 偏置有 4 个门而不是 3 个
//  (PyTorch GRU bias)     (MKLDNN GRU bias)
//
//  bias_ih    bias_hh          bias
//  +-----+    +-----+       +---------+
//  | rt1 |    | rt2 |       | zt1+zt2 |
//  |-----|    |-----|       |---------|
//  | zt1 |    | zt2 |       | rt1+rt2 |
//  |-----|    |-----|       |---------|
//  | nt1 |    | nt2 |       |   nt1   |
//  +-----+    +-----+       |---------|
//                           |   nt2   |
//                           +---------+
//
static Tensor _shuffle_weight(const Tensor& weight, int64_t fn_mode) {
  // 将权重张量转换为连续存储
  auto weight_t = weight.contiguous();
  // 如果是 GRU 模式，重新排列权重的门顺序
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> gates = weight_t.chunk(3, /*gates*/0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/0);
  }
  // 返回处理后的权重张量
  return weight_t;
}

// 根据 GRU 模式重新排列偏置
static Tensor _shuffle_bias(const Tensor& bias_ih, const Tensor& bias_hh, int64_t fn_mode) {
  // 如果是 GRU 模式，对偏置进行重新排列
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> b1 = bias_ih.chunk(3, /*output_channels*/0);
    std::vector<Tensor> b2 = bias_hh.chunk(3, /*output_channels*/0);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/0);
  }
  // 否则，直接合并两个偏置张量
  return bias_ih + bias_hh;
}

// 创建 MKLDNN RNN 层的输入元组
std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_layer(const Tensor& input,
    const Tensor& w0,
    const Tensor& w1,
    const Tensor& w2,
    const Tensor& w3,
    const Tensor& hx_,
    const Tensor& cx_,
    bool reverse,
    IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    // 定义一个 RNNParams 对象，用于存储循环神经网络的参数和配置
    RNNParams rnn(
        input,
        batch_sizes,
        mode,
        hidden_size,
        num_layers,
        bidirectional,
        batch_first,
        train);

    // 计算输出张量的大小，根据是否单向确定
    auto output_size = _output_size</*is_single_direction*/ true>(rnn);

    // 创建一个空的输出张量，使用与输入相同的选项
    auto output = at::empty(output_size, input.options());

    // 创建空的隐藏状态张量和细胞状态张量，使用与初始隐藏状态相同的选项
    auto hy_ = at::empty(hx_.sizes(), hx_.options());
    auto cy_ = at::empty(cx_.sizes(), cx_.options());

    // 重排输入权重和隐藏权重，根据 RNN 的模式选择不同的排列方式
    auto weight_ih = _shuffle_weight(w0, rnn.mode);
    auto weight_hh = _shuffle_weight(w1, rnn.mode);

    // 如果有偏置项，重排偏置项；否则创建全零的偏置项
    auto bias = has_biases
        ? _shuffle_bias(w2, w3, rnn.mode)
        : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options().layout(at::Layout::Strided));

    // 计算每层的输入大小
    int64_t input_size = input.size(2);

    // 定义 ideep::tensor 类型的变量，并从输入张量创建视图
    ideep::tensor w1_, w2_;
    auto x = itensor_view_from_dense(
        input,
        rnn.src_layer_desc(input_size, get_mkldnn_dtype(input)));
    auto hx = itensor_view_from_dense(
        hx_, rnn.src_iter_desc(get_mkldnn_dtype(hx_)));
    auto cx = itensor_view_from_dense(
        cx_, rnn.src_iter_c_desc(get_mkldnn_dtype(cx_)));
    auto b = itensor_view_from_dense(
        bias, rnn.bias_desc(get_mkldnn_dtype(bias)));
    auto y = itensor_view_from_dense(
        output, rnn.dst_layer_desc(get_mkldnn_dtype(output)));
    auto hy = itensor_view_from_dense(
        hy_, rnn.dst_iter_desc(get_mkldnn_dtype(hy_)));
    auto cy = itensor_view_from_dense(
        cy_, rnn.dst_iter_c_desc(get_mkldnn_dtype(cy_)));

    // 根据权重是否为 mkldnn 类型，选择不同的初始化方式
    w1_ = weight_ih.is_mkldnn() ? itensor_from_tensor(weight_ih) : itensor_view_from_dense(weight_ih, rnn.weights_layer_desc(input_size, get_mkldnn_dtype(weight_ih)));
    w2_ = weight_hh.is_mkldnn() ? itensor_from_tensor(weight_hh) : itensor_view_from_dense(weight_hh, rnn.weights_iter_desc(get_mkldnn_dtype(weight_hh)));

    // 如果梯度模式开启，则准备工作空间并进行 LSTM 前向传播计算
    if (at::GradMode::is_enabled()) {
        Tensor workspace = Tensor();
        auto pd = ideep::lstm_forward_training::prepare(
            x, hx, cx, w1_, w2_, b, y, hy, cy, reverse);
        workspace = at::empty(pd.workspace_desc().get_size() / sizeof(uint8_t), input.options().dtype(at::kByte));
        ideep::tensor mkldnn_workspace;
        mkldnn_workspace.init(
            pd.workspace_desc(), workspace.template data_ptr<uint8_t>());
        ideep::lstm_forward_training::compute(
            pd, x, hx, cx, w1_, w2_, b, mkldnn_workspace, y, hy, cy, reverse, ideep::prop_kind::forward_training);
        
        // 返回计算结果和更新的隐藏状态、细胞状态以及工作空间
        return std::make_tuple(output, hy_, cy_, workspace);
    } else {
        // 如果梯度模式未开启，则进行 LSTM 推断前向传播计算
        ideep::lstm_forward_inference::compute(
            x, hx, cx, w1_, w2_, b, y, hy, cy, reverse, ideep::prop_kind::forward_inference);
        
        // 返回计算结果和更新的隐藏状态、细胞状态以及空的工作空间
        return std::make_tuple(output, hy_, cy_, Tensor());
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_layer_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    grad_y_ = at::empty(
        grad_output.sizes(),
        grad_output.options().dtype(at::ScalarType::Float));
    grad_y_.copy_(grad_output);
    grad_hy_ = at::empty(
        grad_hy.sizes(), grad_hy.options().dtype(at::ScalarType::Float));
    grad_hy_.copy_(grad_hy);
    grad_cy_ = at::empty(
        grad_cy.sizes(), grad_cy.options().dtype(at::ScalarType::Float));
    grad_cy_.copy_(grad_cy);

    // Initialize diff_y, diff_hy, and diff_cy based on input conditions
    if (train) {
        // If in training mode, initialize using specific tensors
        diff_y = itensor_view_from_dense(
            grad_y_, rnn.dst_layer_desc(get_mkldnn_dtype(grad_y_.scalar_type())));
        diff_hy = itensor_view_from_dense(
            grad_hy_, rnn.dst_iter_desc(get_mkldnn_dtype(grad_hy_.scalar_type())));
        diff_cy = itensor_view_from_dense(
            grad_cy_, rnn.dst_iter_desc(get_mkldnn_dtype(grad_cy_.scalar_type())));
    } else {
        // Otherwise, use original tensors for gradients
        diff_y = itensor_view_from_dense(
            grad_output, rnn.dst_layer_desc(ideep::tensor::data_type::f32));
        diff_hy = itensor_view_from_dense(
            grad_hy, rnn.dst_iter_desc(ideep::tensor::data_type::f32));
        diff_cy = itensor_view_from_dense(
            grad_cy, rnn.dst_iter_desc(ideep::tensor::data_type::f32));
    }

    // Prepare forward hint for LSTM training using ideep library
    auto forward_hint = ideep::lstm_forward_training::prepare(x, hx, cx, w1, w2, b, y, hy, cy, reverse);

    // Initialize mkldnn workspace for LSTM backward computation
    ideep::tensor mkldnn_workspace;
    mkldnn_workspace.init(
        forward_hint.workspace_desc(), workspace.template data_ptr<uint8_t>());

    // Perform LSTM backward computation using ideep library
    ideep::lstm_backward::compute(
        forward_hint, x, hx, cx, w1, w2, b, y, hy, cy, diff_y, diff_hy, diff_cy,
        mkldnn_workspace, diff_x, diff_hx, diff_cx, diff_w1, diff_w2, diff_b, reverse);

    // Clone diff_b_ tensor to ensure continuity in output tuple
    auto diff_b2_ = at::clone(diff_b_);

    // Return tuple of computed gradients
    return std::make_tuple(diff_x_, diff_w1_, diff_w2_, diff_b_, diff_b2_, diff_hx_, diff_cx_);
}

// MKLDNN RNN integration notes:
// I. Memory Formats
//   a. mkldnn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All mkldnn memorys are created (in plain format) as views on ATen tensor,
//      the weight reorder(if any) is handed automatically inside ideep (mkldnn bridge)
//
// II. MKLDNN Primitive Mapping
//   a. mkldnn rnn primitive doesn't support training with dropout or padded input sequence.
//   b. here break a single RNN module into { num_layers * num_directions } mkldnn rnn primitives
// 对于未来需要涵盖的功能空缺。
//
//TODO: a. 使用 dropout 进行训练
//   b. 支持填充序列输入

static std::tuple<Tensor, Tensor, Tensor> mkldnn_rnn(
    const Tensor& input_, TensorList weight, int64_t weight_stride0,
    const Tensor& hx_, const Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool has_biases, bool batch_first, double dropout_p,
    bool train, bool bidirectional, IntArrayRef batch_sizes) {
  // 检查是否支持打包输入
  TORCH_CHECK(batch_sizes.size() == 0, "mkldnn_rnn 不支持打包输入");
  // 对于非 LSTM 类型的 RNN，检查是否未定义 cx
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "mkldnn_rnn: 非 LSTM 类型 RNN 的 cx 定义非法");
  }

  auto input = input_;
  // 如果 batch_first 为 true，则转置输入张量的维度 0 和 1
  if (batch_first) {
    input = input.transpose(0, 1);
  }
  // 确保输入张量是连续的
  input = input.contiguous();

  // 确保 hx 和 cx 张量是连续的
  auto hx = hx_.contiguous();
  auto cx = cx_.contiguous();

  // 构建权重矩阵引用
  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_directions = bidirectional ? 2 : 1;
  auto layer_input = input;
  std::vector<at::Tensor> layer_output(num_directions);
  std::vector<at::Tensor> layer_hy(num_layers * num_directions);
  std::vector<at::Tensor> layer_cy(num_layers * num_directions);
  for (const auto layer: c10::irange(num_layers)) {
    for (const auto direction: c10::irange(num_directions)) {
      const auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      // 检查层权重张量的大小是否为 2 或 4
      TORCH_CHECK(layer_weights.size() == 2 || layer_weights.size() == 4);
      auto layer_hx = hx[index];
      auto layer_cx = cx[index];
      auto reverse = (direction > 0);
      // 如果有偏置，创建一个与指定大小和选项的零张量
      auto outputs = at::mkldnn_rnn_layer(layer_input, layer_weights[0], layer_weights[1],
                                        has_biases ? layer_weights[2] : at::zeros(layer_weights[0].sizes(), layer_weights[0].options().layout(at::Layout::Strided)),
          has_biases ? layer_weights[3] : at::zeros(layer_weights[1].sizes(), layer_weights[1].options().layout(at::Layout::Strided)), layer_hx,
          layer_cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train);
      layer_output[direction] = std::get<0>(outputs);
      layer_hy[index] = std::get<1>(outputs);
      layer_cy[index] = std::get<2>(outputs);
    }
    // 如果是单向的，直接取第一个输出；如果是双向的，拼接输出张量
    layer_input = num_directions == 1 ? layer_output[0]
                                      : at::cat(layer_output, /*output_channels*/-1);
    // 如果 dropout_p 不为 0 且处于训练状态，并且当前层不是最后一层，则应用 dropout
    if (dropout_p != 0 && train && layer < num_layers - 1) {
      layer_input = at::dropout(layer_input, dropout_p, /*train=*/true);
    }
  }
  auto output = layer_input;
  auto hy = at::stack(layer_hy, 0);
  auto cy = at::stack(layer_cy, 0);
  // 如果 batch_first 为 true，则转置输出张量的维度 0 和 1
  if (batch_first) {
    output = output.transpose(0, 1);
  }
  return std::make_tuple(output, hy, cy);
}

////////////////////////////////////////////////////////////////////////////////
//// 用于通用 RNN 操作的 MKLDNN 分发（如 at::lstm, at::gru, ...）
////////////////////////////////////////////////////////////////////////////////

namespace {

// 声明一个匿名命名空间，用于定义辅助函数和局部变量，避免全局污染。

// 从隐藏状态元组中解包隐藏状态
std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

// 模板函数，用于将隐藏状态打包成指定类型的隐藏状态
template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  // 如果使用的隐藏类型没有实现打包函数，抛出错误
  static_assert(std::is_same<hidden_type, void>::value, "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

// 显式特化模板函数，将隐藏状态打包为 Tensor 元组类型
template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

// 实现 MKL-DNN 特定的 RNN 操作函数
template<typename hidden_type>
std::pair<Tensor, hidden_type> mkldnn_impl(
    const Tensor& input, const hidden_type& hidden,
    TensorList params, bool has_biases, ideep::rnn_kind mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  auto [hx, cx] = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  // 调用 MKL-DNN 提供的 RNN 接口进行计算
  auto mkldnn_output = mkldnn_rnn(
      input, params, has_biases ? 4 : 2,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, has_biases, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes*/{});

  // 返回计算结果，包括输出 Tensor 和打包后的隐藏状态
  return {std::get<0>(mkldnn_output),
          pack_hidden<hidden_type>(std::get<1>(mkldnn_output), std::get<2>(mkldnn_output))};
}

// 实现 LSTM 在 MKL-DNN 上的包装函数
void lstm_mkldnn(Tensor& output, Tensor& hy, Tensor& cy,
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  // 调用 MKL-DNN 实现函数 mkldnn_impl 来执行 LSTM 计算
  auto result = mkldnn_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      ideep::rnn_kind::LSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  
  // 将计算结果分别赋值给输出 Tensor 和最终的隐藏状态 Tensor
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

} // 匿名命名空间结束

// 注册 MKL-DNN 版本的 LSTM 实现
REGISTER_ALL_CPU_DISPATCH(lstm_mkldnn_stub, &lstm_mkldnn);

} // namespace at::native

#endif // AT_MKLDNN_ENABLED


注释：
- 匿名命名空间中定义了几个用于处理隐藏状态的辅助函数和模板函数。
- `mkldnn_impl` 函数调用了 MKL-DNN 库中的 RNN 接口进行 LSTM 计算，并根据模板参数打包隐藏状态。
- `lstm_mkldnn` 函数包装了 `mkldnn_impl`，实现了在 MKL-DNN 上执行 LSTM 计算，并将结果写入输出 Tensor 和隐藏状态 Tensor。
- 最后，使用 `REGISTER_ALL_CPU_DISPATCH` 注册了 MKL-DNN 版本的 LSTM 实现函数。
```