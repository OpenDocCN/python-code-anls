# `.\pytorch\torch\csrc\api\src\nn\modules\rnn.cpp`

```
// 包含 PyTorch 的 RNN 模块头文件
#include <torch/nn/modules/rnn.h>

// 包含 PyTorch 的初始化相关头文件
#include <torch/nn/init.h>
// 包含 PyTorch 的数据类型定义头文件
#include <torch/types.h>
// 包含 PyTorch 的实用函数头文件
#include <torch/utils.h>

// 包含 C10 库的异常处理头文件
#include <c10/util/Exception.h>
// 包含 C10 库的整数范围迭代头文件
#include <c10/util/irange.h>

// 包含标准数学函数头文件
#include <cmath>
// 包含标准整数类型定义头文件
#include <cstdint>
// 包含标准正则表达式头文件
#include <regex>
// 包含标准字符串处理头文件
#include <string>
// 包含标准元组头文件
#include <tuple>
// 包含标准无序集合头文件
#include <unordered_set>
// 包含标准实用工具头文件
#include <utility>
// 包含标准向量头文件
#include <vector>

// 使用 torch::nn::utils::rnn 命名空间
using namespace torch::nn::utils::rnn;

// 定义 torch::nn 命名空间
namespace torch {
namespace nn {

/// 以下枚举必须与 CUDNN 模式代码保持一致：
/// 参考链接：https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
enum class CuDNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

// 根据 RNNOptionsBase 的 mode 参数获取对应的 CuDNN 模式
static CuDNNMode get_cudnn_mode_for_rnn(
    detail::RNNOptionsBase::rnn_options_base_mode_t mode) {
  // 根据不同的 mode 变体返回对应的 CuDNN 模式
  if (std::holds_alternative<enumtype::kRNN_RELU>(mode)) {
    return CuDNNMode::RNN_RELU;
  } else if (std::holds_alternative<enumtype::kRNN_TANH>(mode)) {
    return CuDNNMode::RNN_TANH;
  } else if (std::holds_alternative<enumtype::kLSTM>(mode)) {
    return CuDNNMode::LSTM;
  } else if (std::holds_alternative<enumtype::kGRU>(mode)) {
    return CuDNNMode::GRU;
  } else {
    // 如果 mode 不匹配任何已知类型，则抛出错误
    TORCH_CHECK(false, "Unknown mode: ", torch::enumtype::get_enum_name(mode));
  }
}

// 对给定的 tensor 按照 permutation 张量进行维度索引重排
static Tensor apply_permutation(
    const Tensor& tensor,
    const Tensor& permutation,
    int64_t dim = 1) {
  return tensor.index_select(dim, permutation);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace detail {
// RNNImplBase 类模板的实现
template <typename Derived>
RNNImplBase<Derived>::RNNImplBase(const RNNOptionsBase& options_)
    : options_base(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 函数来初始化 RNNImplBase 对象
  reset();
}

// 初始化 RNNImplBase 对象的函数
template <typename Derived>
void RNNImplBase<Derived>::reset() {
  // 计算方向数量，根据是否双向 RNN 决定
  const int64_t num_directions = options_base.bidirectional() ? 2 : 1;

  // 检查 dropout 参数的有效性，应在 [0, 1] 范围内
  TORCH_CHECK(
      0 <= options_base.dropout() && options_base.dropout() <= 1,
      "dropout should be a number in range [0, 1] ",
      "representing the probability of an element being ",
      "zeroed");

  // 如果 dropout 大于 0 且 num_layers 为 1，发出警告，因为 dropout 应该在除最后一个 RNN 层外的所有层后应用
  if (options_base.dropout() > 0 && options_base.num_layers() == 1) {
    TORCH_WARN(
        "dropout option adds dropout after all but last ",
        "recurrent layer, so non-zero dropout expects ",
        "num_layers greater than 1, but got dropout=",
        options_base.dropout(),
        " and ",
        "num_layers=",
        options_base.num_layers());
  }

  // 检查 hidden_size 必须为正数
  TORCH_CHECK(
      options_base.hidden_size() > 0, "hidden_size must be greater than zero");

  // 检查 num_layers 必须为正数
  TORCH_CHECK(
      options_base.num_layers() > 0, "num_layers must be greater than zero");

  // 检查 proj_size 必须为正数且小于 hidden_size
  TORCH_CHECK(
      0 <= options_base.proj_size() &&
          options_base.proj_size() < options_base.hidden_size(),
      "proj_size has to be a positive integer, smaller than ",
      "hidden_size or zero to disable projections");

  // 如果 proj_size 大于 0，则执行以下代码块
  if (options_base.proj_size() > 0) {
    // 检查模型选项中的模式是否为 LSTM，否则抛出错误信息
    TORCH_CHECK(
        std::get_if<enumtype::kLSTM>(&options_base.mode()),
        "proj_size argument is only supported for LSTM, not RNN or GRU");
  }

  // 初始化门的大小为 0
  int64_t gate_size = 0;
  // 如果模式为 LSTM
  if (std::holds_alternative<enumtype::kLSTM>(options_base.mode())) {
    // 计算门的大小为隐藏层大小的四倍
    gate_size = 4 * options_base.hidden_size();
  } else if (std::holds_alternative<enumtype::kGRU>(options_base.mode())) {
    // 如果模式为 GRU，计算门的大小为隐藏层大小的三倍
    gate_size = 3 * options_base.hidden_size();
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (std::holds_alternative<enumtype::kRNN_TANH>(options_base.mode())) {
    // 如果模式为 RNN_TANH，门的大小等于隐藏层大小
    gate_size = options_base.hidden_size();
  } else if (std::holds_alternative<enumtype::kRNN_RELU>(options_base.mode())) {
    // 如果模式为 RNN_RELU，门的大小等于隐藏层大小
    gate_size = options_base.hidden_size();
  } else {
    // 如果模式未识别，抛出错误信息并包含未识别的模式名
    TORCH_CHECK(
        false,
        "Unrecognized RNN mode: " +
            torch::enumtype::get_enum_name(options_base.mode()));
  }

  // 初始化 flat_weights_names_ 和 all_weights_ 为空
  flat_weights_names_ = {};
  all_weights_ = {};

  // 遍历每一层神经网络
  for (const auto layer : c10::irange(options_base.num_layers())) {
    for (const auto direction : c10::irange(num_directions)) {
      // 遍历方向数量范围，方向是一个整数，用于表示RNN的正向或反向
      int64_t real_hidden_size = options_base.proj_size() > 0
          ? options_base.proj_size()
          : options_base.hidden_size();
      // 计算实际的隐藏层大小，可以是投影大小或隐藏层大小的选择
      int64_t layer_input_size = layer == 0 ? options_base.input_size()
                                            : real_hidden_size * num_directions;
      // 计算当前层的输入大小，根据层索引和方向数量决定

      auto w_ih = torch::empty({gate_size, layer_input_size});
      // 创建一个空的张量 w_ih，表示输入到隐藏层的权重矩阵
      auto w_hh = torch::empty({gate_size, real_hidden_size});
      // 创建一个空的张量 w_hh，表示隐藏层到隐藏层的权重矩阵
      auto b_ih = torch::empty({gate_size});
      // 创建一个空的张量 b_ih，表示输入到隐藏层的偏置向量
      // CuDNN 兼容性需要的第二个偏置向量，标准定义只需要一个偏置向量
      auto b_hh = torch::empty({gate_size});
      // 创建一个空的张量 b_hh，表示隐藏层到隐藏层的偏置向量

      std::vector<Tensor> layer_params = {w_ih, w_hh};
      // 将权重张量 w_ih 和 w_hh 加入到层参数向量中

      std::string suffix = direction == 1 ? "_reverse" : "";
      // 根据方向确定后缀，如果方向为1则为 "_reverse"

      std::vector<std::string> param_names = {
          "weight_ih_l{layer}{suffix}", "weight_hh_l{layer}{suffix}"};
      // 创建参数名列表，包括权重名称和可能的偏置名称，用于注册参数

      if (options_base.bias()) {
        param_names.emplace_back("bias_ih_l{layer}{suffix}");
        // 如果需要偏置，将输入到隐藏层的偏置名称加入列表
        param_names.emplace_back("bias_hh_l{layer}{suffix}");
        // 将隐藏层到隐藏层的偏置名称加入列表
        layer_params.emplace_back(b_ih);
        // 将输入到隐藏层的偏置张量 b_ih 加入层参数向量
        layer_params.emplace_back(b_hh);
        // 将隐藏层到隐藏层的偏置张量 b_hh 加入层参数向量
      }

      if (options_base.proj_size() > 0) {
        auto w_hr = torch::empty(
            {options_base.proj_size(), options_base.hidden_size()});
        // 如果有投影大小，创建一个投影到隐藏层的权重张量 w_hr
        layer_params.emplace_back(w_hr);
        // 将 w_hr 加入到层参数向量
        param_names.emplace_back("weight_hr_l{layer}{suffix}");
        // 将投影到隐藏层的权重名称加入参数名列表
      }

      for (auto& param_name : param_names) {
        std::string x = std::regex_replace(
            param_name, std::regex("\\{layer\\}"), c10::str(layer));
        // 替换参数名中的层索引占位符 {layer} 为实际的层索引值
        param_name =
            std::regex_replace(x, std::regex("\\{suffix\\}"), c10::str(suffix));
        // 替换参数名中的后缀占位符 {suffix} 为实际的后缀值
      }

      for (const auto i : c10::irange(param_names.size())) {
        this->register_parameter(param_names[i], std::move(layer_params[i]));
        // 注册参数，将参数名与对应的张量或向量关联起来
      }

      flat_weights_names_.insert(
          flat_weights_names_.end(), param_names.begin(), param_names.end());
      // 将所有权重的名称扁平化并添加到 flat_weights_names_ 中

      all_weights_.emplace_back(std::move(param_names));
      // 将当前层的所有参数名称列表加入到 all_weights_ 中
    }
  }

  flat_weights_ = {};
  // 清空扁平化的权重列表 flat_weights_

  for (const auto& wn : flat_weights_names_) {
    auto named_parameters = this->named_parameters(/*recurse=*/false);
    // 获取当前模型的命名参数，不递归获取
    if (named_parameters.contains(wn)) {
      flat_weights_.emplace_back(named_parameters[wn]);
      // 如果命名参数中包含当前权重名称 wn，则将其加入 flat_weights_
    } else {
      flat_weights_.emplace_back();
      // 否则加入一个空的张量
    }
  }

  this->flatten_parameters();
  // 将模型参数扁平化，以便于优化器处理

  this->reset_parameters();
  // 重置模型的参数，根据默认策略重新初始化参数
}

template <typename Derived>
void RNNImplBase<Derived>::flatten_parameters() {
  // 重置参数数据指针，以便可以使用更快的代码路径。
  //
  // 当模块在 GPU 上并且启用了 cuDNN 时，此操作有效；否则，它是一个空操作。

  // 如果 flat_weights_ 的大小与 flat_weights_names_ 的大小不匹配，则直接返回
  if (flat_weights_.size() != flat_weights_names_.size()) {
    return;
  }

  // 如果 self.flat_weights_ 中的任何张量不符合 cuDNN 的要求，或者 flat_weights_ 的张量类型不同，则直接返回
  auto first_fw = flat_weights_[0];
  auto dtype = first_fw.dtype();
  for (const auto& fw : flat_weights_) {
    if (!(fw.dtype() == dtype) || !fw.is_cuda() ||
        !torch::cudnn_is_acceptable(fw)) {
      return;
    }
  }

  // 如果任何参数存在别名，则回退到较慢的复制代码路径。
  // 这是一个足够的检查，因为重叠的参数缓冲区会破坏 Module::named_parameters() 中唯一性检查的假设。
  std::unordered_set<void*> unique_data_ptrs;
  for (const auto& p : flat_weights_) {
    unique_data_ptrs.emplace(p.data_ptr());
  }
  if (unique_data_ptrs.size() != flat_weights_.size()) {
    return;
  }

  {
    torch::DeviceGuard device_guard(first_fw.device());

    // 注意：由于 _cudnn_rnn_flatten_weight 是对 self.flat_weights_ 的原地操作，因此需要使用 no_grad()
    {
      torch::NoGradGuard no_grad;
      if (torch::_use_cudnn_rnn_flatten_weight()) {
        int64_t num_weights = options_base.bias() ? 4 : 2;
        if (options_base.proj_size() > 0) {
          ++num_weights;
        }
        torch::_cudnn_rnn_flatten_weight(
            flat_weights_,
            num_weights,
            options_base.input_size(),
            static_cast<int64_t>(get_cudnn_mode_for_rnn(options_base.mode())),
            options_base.hidden_size(),
            options_base.proj_size(),
            options_base.num_layers(),
            options_base.batch_first(),
            options_base.bidirectional());
      }
    }
  }
}

template <typename Derived>
void RNNImplBase<Derived>::reset_flat_weights() {
  // 清空 flat_weights_，并根据 flat_weights_names_ 重新设置 flat_weights_ 的值
  flat_weights_ = {};
  for (const auto& wn : flat_weights_names_) {
    auto named_parameters = this->named_parameters(/*recurse=*/false);
    if (named_parameters.contains(wn)) {
      flat_weights_.emplace_back(named_parameters[wn]);
    } else {
      flat_weights_.emplace_back();
    }
  }
}

template <typename Derived>
void RNNImplBase<Derived>::to(
    torch::Device device,
    torch::Dtype dtype,
    bool non_blocking) {
  // 将模块移动到指定的设备和数据类型，并重置 flat_weights_，然后重新展开参数
  nn::Module::to(device, dtype, non_blocking);
  reset_flat_weights();
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Dtype dtype, bool non_blocking) {
  // 将模块移动到指定的数据类型，并重置 flat_weights_，然后重新展开参数
  nn::Module::to(dtype, non_blocking);
  reset_flat_weights();
  flatten_parameters();
}
// 将 RNNImplBase 派生类的模型参数转移到指定设备上
void RNNImplBase<Derived>::to(torch::Device device, bool non_blocking) {
  // 调用 nn::Module 的 to 方法，将模型参数转移到指定设备上
  nn::Module::to(device, non_blocking);
  // 重置平坦化权重
  reset_flat_weights();
  // 将模型参数进行平坦化处理
  flatten_parameters();
}

// 重置 RNNImplBase 派生类的模型参数
template <typename Derived>
void RNNImplBase<Derived>::reset_parameters() {
  // 计算权重初始化的标准差
  const double stdv = 1.0 / std::sqrt(options_base.hidden_size());
  // 对模型的每一个参数应用均匀分布的初始化，范围为 [-stdv, stdv]
  for (auto& weight : this->parameters()) {
    init::uniform_(weight, -stdv, stdv);
  }
}

// 检查输入张量的维度和大小是否符合预期
template <typename Derived>
void RNNImplBase<Derived>::check_input(
    const Tensor& input,
    const Tensor& batch_sizes) const {
  // 预期的输入维度，根据 batch_sizes 是否定义来确定
  int64_t expected_input_dim = batch_sizes.defined() ? 2 : 3;
  // 检查输入张量的维度是否符合预期
  TORCH_CHECK(
      input.dim() == expected_input_dim,
      "input must have ",
      expected_input_dim,
      " dimensions, got ",
      input.dim());
  // 检查输入张量的最后一个维度是否等于 options_base 中指定的输入大小
  TORCH_CHECK(
      options_base.input_size() == input.size(-1),
      "input.size(-1) must be equal to input_size. Expected ",
      options_base.input_size(),
      ", got ",
      input.size(-1));
}

// 获取预期的隐藏状态大小
template <typename Derived>
std::tuple<int64_t, int64_t, int64_t> RNNImplBase<Derived>::
    get_expected_hidden_size(const Tensor& input, const Tensor& batch_sizes)
        const {
  // 计算 mini_batch 大小
  int64_t mini_batch = 0;
  if (batch_sizes.defined()) {
    mini_batch = batch_sizes[0].item<int64_t>();
  } else {
    mini_batch = options_base.batch_first() ? input.size(0) : input.size(1);
  }
  // 计算方向数量和真实的隐藏层大小
  int64_t num_directions = options_base.bidirectional() ? 2 : 1;
  int64_t real_hidden_size = options_base.proj_size() > 0
      ? options_base.proj_size()
      : options_base.hidden_size();
  // 返回预期的隐藏状态大小作为元组
  return std::make_tuple(
      options_base.num_layers() * num_directions, mini_batch, real_hidden_size);
}

// 检查隐藏状态的大小是否符合预期
template <typename Derived>
void RNNImplBase<Derived>::check_hidden_size(
    const Tensor& hx,
    std::tuple<int64_t, int64_t, int64_t> expected_hidden_size,
    std::string msg) const {
  // 构造预期的隐藏状态大小向量
  auto expected_hidden_size_vec = std::vector<int64_t>({
      std::get<0>(expected_hidden_size),
      std::get<1>(expected_hidden_size),
      std::get<2>(expected_hidden_size),
  });
  // 检查实际的隐藏状态大小是否与预期相符
  if (hx.sizes() != expected_hidden_size_vec) {
    // 更新消息中的占位符，显示预期和实际的隐藏状态大小
    msg = std::regex_replace(
        msg, std::regex("\\{1\\}"), c10::str(expected_hidden_size_vec));
    msg = std::regex_replace(msg, std::regex("\\{2\\}"), c10::str(hx.sizes()));
    // 抛出异常，显示详细的错误消息
    TORCH_CHECK(false, msg);
  }
}

// 检查前向计算的输入参数是否符合预期
template <typename Derived>
void RNNImplBase<Derived>::check_forward_args(
    Tensor input,
    Tensor hidden,
    Tensor batch_sizes) const {
  // 检查输入张量的维度和大小是否符合预期
  this->check_input(input, batch_sizes);
  // 获取预期的隐藏状态大小
  auto expected_hidden_size =
      this->get_expected_hidden_size(input, batch_sizes);
  // 检查隐藏状态的大小是否符合预期
  this->check_hidden_size(hidden, expected_hidden_size);
}

// 对隐藏状态进行置换
template <typename Derived>
Tensor RNNImplBase<Derived>::permute_hidden(
    Tensor hx,
    const Tensor& permutation) const {
  // 如果置换张量未定义，直接返回隐藏状态张量
  if (!permutation.defined()) {
    return hx;
  }
  // 应用给定的置换对隐藏状态进行重排
  return apply_permutation(hx, permutation);
}
// 将 RNN 名称提取为字符串
const std::string name = this->name();
// 去掉名称末尾的 "_impl" 后缀，以获取更简洁的名称
const std::string name_without_impl = name.substr(0, name.size() - 4);
// 在流中输出简化后的 RNN 名称及其参数
stream << std::boolalpha << name_without_impl
       << "(input_size=" << options_base.input_size()
       << ", hidden_size=" << options_base.hidden_size()
       << ", num_layers=" << options_base.num_layers()
       << ", bias=" << options_base.bias()
       << ", batch_first=" << options_base.batch_first()
       << ", dropout=" << options_base.dropout()
       << ", bidirectional=" << options_base.bidirectional();
// 如果设置了投影尺寸，也输出该参数
if (options_base.proj_size() > 0) {
  stream << ", proj_size=" << options_base.proj_size();
}
// 输出 RNN 描述的结尾括号
stream << ")";
}

// 返回 RNN 实现中所有权重张量的向量
template <typename Derived>
std::vector<Tensor> RNNImplBase<Derived>::all_weights() const {
  // 初始化空的结果向量
  std::vector<Tensor> result = {};
  // 获取未递归命名的参数
  auto named_parameters = this->named_parameters(/*recurse=*/false);
  // 遍历所有权重，并添加到结果向量中
  for (const auto& weights : all_weights_) {
    for (const auto& weight : weights) {
      result.emplace_back(named_parameters[weight]);
    }
  }
  // 返回结果向量
  return result;
}

// 实例化 RNNImplBase 类模板以支持不同类型的 RNN
template class RNNImplBase<LSTMImpl>;
template class RNNImplBase<GRUImpl>;
template class RNNImplBase<RNNImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 根据非线性激活函数类型计算 RNN 的基本选项模式
static detail::RNNOptionsBase::rnn_options_base_mode_t
compute_rnn_options_base_mode(RNNOptions::nonlinearity_t nonlinearity) {
  // 根据非线性激活函数的类型选择相应的 RNN 模式
  if (std::holds_alternative<enumtype::kTanh>(nonlinearity)) {
    return torch::kRNN_TANH;
  } else if (std::holds_alternative<enumtype::kReLU>(nonlinearity)) {
    return torch::kRNN_RELU;
  } else {
    // 如果非线性激活函数类型未知，则抛出异常
    TORCH_CHECK(
        false,
        "Unknown nonlinearity ",
        torch::enumtype::get_enum_name(nonlinearity));
  }
}

// 根据给定的 RNNOptions 构造函数初始化 RNNImpl 对象
RNNImpl::RNNImpl(const RNNOptions& options_)
    : detail::RNNImplBase<RNNImpl>(
          // 使用 RNNOptions 创建 RNNImplBase，设置基本选项
          detail::RNNOptionsBase(
              compute_rnn_options_base_mode(options_.nonlinearity()),
              options_.input_size(),
              options_.hidden_size())
              .num_layers(options_.num_layers())
              .bias(options_.bias())
              .batch_first(options_.batch_first())
              .dropout(options_.dropout())
              .bidirectional(options_.bidirectional())),
      options(options_) {}

// RNN 前向传播辅助函数，返回隐藏状态及其梯度
std::tuple<Tensor, Tensor> RNNImpl::forward_helper(
    const Tensor& input,
    const Tensor& batch_sizes,
    const Tensor& sorted_indices,
    int64_t max_batch_size,
    Tensor hx) {
  // 如果隐藏状态未定义，则根据输入参数初始化其为零张量
  if (!hx.defined()) {
    int64_t num_directions = options_base.bidirectional() ? 2 : 1;
    hx = torch::zeros(
        {options_base.num_layers() * num_directions,
         max_batch_size,
         options_base.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    // 否则，确保每个批次的隐藏状态与用户传递的输入序列匹配
    // (此处省略了具体实现细节的注释)
  # 对隐藏状态进行排序后的置换操作
  hx = this->permute_hidden(hx, sorted_indices);
}

# 检查前向传播参数的有效性
this->check_forward_args(input, hx, batch_sizes);

# 定义存储结果的元组
std::tuple<Tensor, Tensor> result;
if (!batch_sizes.defined()) {
  # 如果未定义批次大小，并且模式是 kRNN_TANH
  if (std::holds_alternative<enumtype::kRNN_TANH>(options_base.mode())) {
    # 调用 torch 的 RNN Tanh 函数进行计算
    result = torch::rnn_tanh(
        input,
        hx,
        flat_weights_,
        options_base.bias(),
        options_base.num_layers(),
        options_base.dropout(),
        this->is_training(),
        options_base.bidirectional(),
        options_base.batch_first());
  } else if (std::holds_alternative<enumtype::kRNN_RELU>(
                 options_base.mode())) {
    # 如果模式是 kRNN_RELU，调用 torch 的 RNN ReLU 函数进行计算
    result = torch::rnn_relu(
        input,
        hx,
        flat_weights_,
        options_base.bias(),
        options_base.num_layers(),
        options_base.dropout(),
        this->is_training(),
        options_base.bidirectional(),
        options_base.batch_first());
  } else {
    # 如果模式未知，抛出错误
    TORCH_CHECK(
        false,
        "Unknown mode: ",
        torch::enumtype::get_enum_name(options_base.mode()));
  }
} else {
  # 如果定义了批次大小，并且模式是 kRNN_TANH
  if (std::holds_alternative<enumtype::kRNN_TANH>(options_base.mode())) {
    # 调用 torch 的 RNN Tanh 函数进行计算，包含批次大小参数
    result = torch::rnn_tanh(
        input,
        batch_sizes,
        hx,
        flat_weights_,
        options_base.bias(),
        options_base.num_layers(),
        options_base.dropout(),
        this->is_training(),
        options_base.bidirectional());
  } else if (std::holds_alternative<enumtype::kRNN_RELU>(
                 options_base.mode())) {
    # 如果模式是 kRNN_RELU，调用 torch 的 RNN ReLU 函数进行计算，包含批次大小参数
    result = torch::rnn_relu(
        input,
        batch_sizes,
        hx,
        flat_weights_,
        options_base.bias(),
        options_base.num_layers(),
        options_base.dropout(),
        this->is_training(),
        options_base.bidirectional());
  } else {
    # 如果模式未知，抛出错误
    TORCH_CHECK(
        false,
        "Unknown mode: ",
        torch::enumtype::get_enum_name(options_base.mode()));
  }
}

# 从结果元组中获取输出张量和隐藏状态张量
auto output = std::get<0>(result);
auto hidden = std::get<1>(result);

# 返回包含输出张量和隐藏状态张量的元组
return std::make_tuple(output, hidden);
}

// 实现 RNN 前向传播函数，接收输入张量和隐藏状态张量，返回输出张量和新的隐藏状态张量
std::tuple<Tensor, Tensor> RNNImpl::forward(const Tensor& input, Tensor hx) {
  // 初始化批处理大小张量
  auto batch_sizes = torch::Tensor();
  // 根据选项确定最大批处理大小
  auto max_batch_size =
      options_base.batch_first() ? input.size(0) : input.size(1);
  // 初始化排序后索引张量
  auto sorted_indices = torch::Tensor();
  // 初始化未排序索引张量
  auto unsorted_indices = torch::Tensor();

  // 调用辅助函数进行前向传播计算
  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  // 返回输出张量和重新排序的隐藏状态张量
  return std::make_tuple(
      output, this->permute_hidden(hidden, unsorted_indices));
}

// 实现使用打包输入的 RNN 前向传播函数，接收打包序列和隐藏状态张量，返回打包输出序列和新的隐藏状态张量
std::tuple<PackedSequence, Tensor> RNNImpl::forward_with_packed_input(
    const PackedSequence& packed_input,
    Tensor hx) {
  // 提取打包输入的数据张量
  const auto& input = packed_input.data();
  // 提取打包输入的批处理大小张量
  const auto& batch_sizes = packed_input.batch_sizes();
  // 提取打包输入的排序后索引张量
  const auto& sorted_indices = packed_input.sorted_indices();
  // 提取打包输入的未排序索引张量
  const auto& unsorted_indices = packed_input.unsorted_indices();
  // 计算最大批处理大小
  auto max_batch_size = batch_sizes[0].item<int64_t>();

  // 调用辅助函数进行前向传播计算
  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  // 创建打包输出序列对象
  auto output_packed =
      PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices);
  
  // 返回打包输出序列和重新排序的隐藏状态张量
  return std::make_tuple(
      output_packed, this->permute_hidden(hidden, unsorted_indices));
}

// LSTMImpl 类的构造函数，初始化 LSTM 模型的选项并继承 RNNImplBase 类
LSTMImpl::LSTMImpl(const LSTMOptions& options_)
    : detail::RNNImplBase<LSTMImpl>(detail::RNNOptionsBase(
                                        torch::kLSTM,
                                        options_.input_size(),
                                        options_.hidden_size())
                                        .num_layers(options_.num_layers())
                                        .bias(options_.bias())
                                        .batch_first(options_.batch_first())
                                        .dropout(options_.dropout())
                                        .bidirectional(options_.bidirectional())
                                        .proj_size(options_.proj_size())),
      options(options_) {}

// 返回 LSTM 期望的单元尺寸，包括层数、批处理大小和隐藏状态大小
std::tuple<int64_t, int64_t, int64_t> LSTMImpl::get_expected_cell_size(
    const Tensor& input,
    const Tensor& batch_sizes) const {
  // 初始化迷你批处理大小
  int64_t mini_batch = 0;
  // 如果提供了批处理大小张量，则使用其第一个元素作为迷你批处理大小
  if (batch_sizes.defined()) {
    mini_batch = batch_sizes[0].item<int64_t>();
  } else {
    // 否则根据选项确定最大批处理大小
    mini_batch = options_base.batch_first() ? input.size(0) : input.size(1);
  }
  // 计算方向数，考虑是否双向
  int64_t num_directions = options_base.bidirectional() ? 2 : 1;
  // 返回层数、迷你批处理大小和隐藏状态大小
  return std::make_tuple(
      options_base.num_layers() * num_directions,
      mini_batch,
      options_base.hidden_size());
}

// 检查 LSTM 前向传播参数，包括输入张量和隐藏状态张量
void LSTMImpl::check_forward_args(
    const Tensor& input,
    std::tuple<Tensor, Tensor> hidden,
    // 检查输入参数和批量大小是否符合要求
    this->check_input(input, batch_sizes);
    // 检查隐藏状态的第一个元素是否符合预期大小
    this->check_hidden_size(
        std::get<0>(hidden),
        this->get_expected_hidden_size(input, batch_sizes),
        "Expected hidden[0] size {1}, got {2}");
    // 检查隐藏状态的第二个元素（细胞状态）是否符合预期大小
    this->check_hidden_size(
        std::get<1>(hidden),
        this->get_expected_cell_size(input, batch_sizes),
        "Expected hidden[1] size {1}, got {2}");
}

std::tuple<Tensor, Tensor> LSTMImpl::permute_hidden(
    std::tuple<Tensor, Tensor> hx,
    const Tensor& permutation) const {
  // 如果置换张量未定义，则直接返回原始隐藏状态
  if (!permutation.defined()) {
    return hx;
  }
  // 对隐藏状态应用置换张量，以调整顺序
  return std::make_tuple(
      apply_permutation(std::get<0>(hx), permutation),
      apply_permutation(std::get<1>(hx), permutation));
}

std::tuple<Tensor, std::tuple<Tensor, Tensor>> LSTMImpl::forward_helper(
    const Tensor& input,
    const Tensor& batch_sizes,
    const Tensor& sorted_indices,
    int64_t max_batch_size,
    torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  std::tuple<Tensor, Tensor> hx;
  // 如果未提供初始隐藏状态，则创建全零张量作为初始状态
  if (!hx_opt.has_value()) {
    int64_t num_directions = options.bidirectional() ? 2 : 1;
    int64_t real_hidden_size =
        options.proj_size() > 0 ? options.proj_size() : options.hidden_size();
    auto h_zeros = torch::zeros(
        {options.num_layers() * num_directions,
         max_batch_size,
         real_hidden_size},
        torch::dtype(input.dtype()).device(input.device()));
    auto c_zeros = torch::zeros(
        {options.num_layers() * num_directions,
         max_batch_size,
         options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
    hx = std::make_tuple(h_zeros, c_zeros);
  } else {
    // 否则，使用提供的隐藏状态，并根据排序索引进行适当的置换
    hx = hx_opt.value();
    // 每个隐藏状态批次应与用户传递的输入序列匹配
    hx = this->permute_hidden(hx, sorted_indices);
  }

  // 检查前向传播所需的参数的有效性
  this->check_forward_args(input, hx, batch_sizes);
  std::tuple<Tensor, Tensor, Tensor> result;
  // 如果未提供批次大小，则调用 LSTM 操作符
  if (!batch_sizes.defined()) {
    result = torch::lstm(
        input,
        {std::get<0>(hx), std::get<1>(hx)},
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional(),
        options.batch_first());
  } else {
    // 否则，使用提供的批次大小调用 LSTM 操作符
    result = torch::lstm(
        input,
        batch_sizes,
        {std::get<0>(hx), std::get<1>(hx)},
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional());
  }
  auto output = std::get<0>(result);
  auto hidden = std::make_tuple(std::get<1>(result), std::get<2>(result));

  return std::make_tuple(output, hidden);
}

std::tuple<Tensor, std::tuple<Tensor, Tensor>> LSTMImpl::forward(
    const Tensor& input,
    torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  auto batch_sizes = torch::Tensor();
  auto max_batch_size = options.batch_first() ? input.size(0) : input.size(1);
  auto sorted_indices = torch::Tensor();
  auto unsorted_indices = torch::Tensor();

  // 调用前向传播辅助函数，获取输出和隐藏状态
  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx_opt));

  // 返回输出及其对应未排序的隐藏状态
  return std::make_tuple(
      output, this->permute_hidden(hidden, unsorted_indices));
}

std::tuple<PackedSequence, std::tuple<Tensor, Tensor>> LSTMImpl::
    // 使用打包输入进行前向传播
    forward_with_packed_input(
        const PackedSequence& packed_input,  // 接收一个打包序列作为输入
        torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {  // 可选的初始隐藏状态

        // 获取打包输入中的数据、批次大小、排序索引和未排序索引
        const auto& input = packed_input.data();
        const auto& batch_sizes = packed_input.batch_sizes();
        const auto& sorted_indices = packed_input.sorted_indices();
        const auto& unsorted_indices = packed_input.unsorted_indices();

        // 计算最大批次大小
        auto max_batch_size = batch_sizes[0].item<int64_t>();

        // 调用辅助方法进行前向传播，获取输出和隐藏状态
        auto [output, hidden] = this->forward_helper(
            input, batch_sizes, sorted_indices, max_batch_size, std::move(hx_opt));

        // 将输出重新打包成序列
        auto output_packed =
            PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices);

        // 返回包含重新打包输出和调整后隐藏状态的元组
        return std::make_tuple(
            output_packed, this->permute_hidden(hidden, unsorted_indices));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// GRUImpl 类的构造函数，初始化基类 RNNImplBase，并设置 GRU 的选项
GRUImpl::GRUImpl(const GRUOptions& options_)
    : detail::RNNImplBase<GRUImpl>(
          detail::RNNOptionsBase(
              torch::kGRU,
              options_.input_size(),
              options_.hidden_size())
              .num_layers(options_.num_layers())
              .bias(options_.bias())
              .batch_first(options_.batch_first())
              .dropout(options_.dropout())
              .bidirectional(options_.bidirectional())),
      options(options_) {}

// 辅助函数，用于执行前向传播操作
std::tuple<Tensor, Tensor> GRUImpl::forward_helper(
    const Tensor& input,
    const Tensor& batch_sizes,
    const Tensor& sorted_indices,
    int64_t max_batch_size,
    Tensor hx) {
  
  // 如果未提供初始隐藏状态 hx，则初始化为零张量
  if (!hx.defined()) {
    int64_t num_directions = options.bidirectional() ? 2 : 1;
    hx = torch::zeros(
        {options.num_layers() * num_directions,
         max_batch_size,
         options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    // 重新排列隐藏状态 hx，以匹配排序后的输入序列
    hx = this->permute_hidden(hx, sorted_indices);
  }

  // 检查前向传播参数的有效性
  this->check_forward_args(input, hx, batch_sizes);

  std::tuple<Tensor, Tensor> result;
  // 根据是否提供 batch_sizes 决定调用 torch::gru 的方式
  if (!batch_sizes.defined()) {
    result = torch::gru(
        input,
        hx,
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional(),
        options.batch_first());
  } else {
    result = torch::gru(
        input,
        batch_sizes,
        hx,
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional());
  }

  auto output = std::get<0>(result);
  auto hidden = std::get<1>(result);

  return std::make_tuple(output, hidden);
}

// 主前向传播函数，调用 forward_helper 执行前向传播，并重新排列隐藏状态以匹配输入的未排序索引
std::tuple<Tensor, Tensor> GRUImpl::forward(const Tensor& input, Tensor hx) {
  auto batch_sizes = torch::Tensor();
  auto max_batch_size = options.batch_first() ? input.size(0) : input.size(1);
  auto sorted_indices = torch::Tensor();
  auto unsorted_indices = torch::Tensor();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  // 返回输出和重新排列后的隐藏状态
  return std::make_tuple(
      output, this->permute_hidden(hidden, unsorted_indices));
}

// 使用压缩输入执行前向传播的函数，返回压缩序列和隐藏状态
std::tuple<PackedSequence, Tensor> GRUImpl::forward_with_packed_input(
    const PackedSequence& packed_input,
    Tensor hx) {
```  
# 定义一个函数，参数为 Tensor hx，表示隐藏状态。

  const auto& input = packed_input.data();
```  
# 获取 packed_input 的数据部分，并将其引用赋给 input。

  const auto& batch_sizes = packed_input.batch_sizes();
```  
# 获取 packed_input 的 batch_sizes 部分，并将其引用赋给 batch_sizes。

  const auto& sorted_indices = packed_input.sorted_indices();
```  
# 获取 packed_input 的 sorted_indices 部分，并将其引用赋给 sorted_indices。

  const auto& unsorted_indices = packed_input.unsorted_indices();
```  
# 获取 packed_input 的 unsorted_indices 部分，并将其引用赋给 unsorted_indices。

  auto max_batch_size = batch_sizes[0].item<int64_t>();
```  
# 计算 batch_sizes 中的最大批次大小，并将其赋给 max_batch_size。

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));
```  
# 调用 this 对象的 forward_helper 方法，传递 input、batch_sizes、sorted_indices、max_batch_size 和 hx 作为参数，获取输出和隐藏状态。

  auto output_packed =
      PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices);
```  
# 创建一个新的 PackedSequence 对象 output_packed，使用 output、batch_sizes、sorted_indices 和 unsorted_indices 作为参数。

  return std::make_tuple(
      output_packed, this->permute_hidden(hidden, unsorted_indices));
```  
# 返回一个 tuple，包含 output_packed 对象和通过 permute_hidden 方法重新排列的 hidden 对象。
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCellImplBase
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
template <typename Derived>
// 根据给定的选项初始化 RNN 单元的基础实现
RNNCellImplBase<Derived>::RNNCellImplBase(const RNNCellOptionsBase& options_)
    : options_base(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 方法进行初始化
  reset();
}

template <typename Derived>
// 重置 RNN 单元的参数
void RNNCellImplBase<Derived>::reset() {
  // 初始化输入到隐藏层权重参数
  weight_ih = this->register_parameter(
      "weight_ih",
      torch::empty(
          {options_base.num_chunks() * options_base.hidden_size(),
           options_base.input_size()}));
  // 初始化隐藏层到隐藏层权重参数
  weight_hh = this->register_parameter(
      "weight_hh",
      torch::empty(
          {options_base.num_chunks() * options_base.hidden_size(),
           options_base.hidden_size()}));

  // 如果存在偏置，则初始化输入到隐藏层和隐藏层到隐藏层的偏置参数
  if (options_base.bias()) {
    bias_ih = this->register_parameter(
        "bias_ih",
        torch::empty({options_base.num_chunks() * options_base.hidden_size()}));
    bias_hh = this->register_parameter(
        "bias_hh",
        torch::empty({options_base.num_chunks() * options_base.hidden_size()}));
  } else {
    // 否则将偏置参数初始化为空张量且不需要梯度
    bias_ih =
        this->register_parameter("bias_ih", Tensor(), /*requires_grad=*/false);
    bias_hh =
        this->register_parameter("bias_hh", Tensor(), /*requires_grad=*/false);
  }

  // 调用 reset_parameters() 方法进行参数初始化
  reset_parameters();
}

template <typename Derived>
// 使用均匀分布初始化所有参数
void RNNCellImplBase<Derived>::reset_parameters() {
  const double stdv = 1.0 / std::sqrt(options_base.hidden_size());
  for (auto& weight : this->parameters()) {
    init::uniform_(weight, -stdv, stdv);
  }
}

template <typename Derived>
// 将 RNN 单元的信息打印到输出流中
void RNNCellImplBase<Derived>::pretty_print(std::ostream& stream) const {
  const std::string name = this->name();
  const std::string name_without_impl = name.substr(0, name.size() - 4);
  stream << name_without_impl << "(" << options_base.input_size() << ", "
         << options_base.hidden_size();
  if (!options_base.bias()) {
    stream << ", bias=" << std::boolalpha << false;
  }
  auto nonlinearity_str = this->get_nonlinearity_str();
  if (!nonlinearity_str.empty() && nonlinearity_str != "kTanh") {
    stream << ", nonlinearity=" << nonlinearity_str;
  }
  stream << ")";
}

template <typename Derived>
// 检查前向输入张量的维度是否符合要求
void RNNCellImplBase<Derived>::check_forward_input(
    const Tensor& input,
    const string& name) const {
  TORCH_CHECK(
      input.dim() == 1 || input.dim() == 2,
      "Expected ",
      name.c_str(),
      " to be 1D or 2D, got ",
      input.dim(),
      "D instead");
}

template <typename Derived>
// 获取非线性激活函数的字符串表示（默认为空字符串）
std::string RNNCellImplBase<Derived>::get_nonlinearity_str() const {
  return "";
}

// 实例化模板类 RNNCellImplBase 分别用于 LSTMCellImpl、GRUCellImpl 和 RNNCellImpl
template class RNNCellImplBase<LSTMCellImpl>;
template class RNNCellImplBase<GRUCellImpl>;
template class RNNCellImplBase<RNNCellImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 根据给定的选项初始化 RNN 单元实现类
RNNCellImpl::RNNCellImpl(const RNNCellOptions& options_)
    // 使用 detail::RNNCellOptionsBase 的构造函数初始化 detail::RNNCellImplBase 类
    // 参数包括输入大小、隐藏层大小、是否使用偏置和 num_chunks 设为 1
    : detail::RNNCellImplBase<RNNCellImpl>(detail::RNNCellOptionsBase(
          options_.input_size(),
          options_.hidden_size(),
          options_.bias(),
          /*num_chunks=*/1)),
      // 使用初始化列表初始化 options 成员变量
      options(options_) {}
// 实现 RNNCellImpl 类的 forward 方法，用于执行 RNN 单元的前向传播
Tensor RNNCellImpl::forward(const Tensor& input, Tensor hx) {
  // 检查输入是否符合要求，确保输入张量 input 的有效性
  this->check_forward_input(input, "input");
  // 检查隐藏状态是否符合要求，确保隐藏状态张量 hx 的有效性
  this->check_forward_input(hx, "hidden");

  // 定义变量 r_hx 和 ret，用于存储处理后的隐藏状态和前向传播结果
  Tensor r_hx, ret;

  // 判断输入是否批处理，以确定是否需要对输入进行扩展
  bool is_batched = input.dim() == 2;
  Tensor r_input = is_batched ? input : input.unsqueeze(0);

  // 如果隐藏状态未定义，则初始化为全零张量
  if (!hx.defined()) {
    r_hx = torch::zeros(
        {input.size(0), options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    // 如果输入已经批处理，则直接使用 hx；否则对 hx 进行扩展以适应批处理
    r_hx = is_batched ? hx : hx.unsqueeze(0);
  }

  // 根据指定的非线性函数类型调用相应的 RNN 单元计算方法
  if (std::holds_alternative<enumtype::kTanh>(options.nonlinearity())) {
    ret = torch::rnn_tanh_cell(
        r_input, r_hx, weight_ih, weight_hh, bias_ih, bias_hh);
  } else if (std::holds_alternative<enumtype::kReLU>(options.nonlinearity())) {
    ret = torch::rnn_relu_cell(
        r_input, r_hx, weight_ih, weight_hh, bias_ih, bias_hh);
  } else {
    // 如果指定的非线性函数类型未知，则抛出错误信息
    TORCH_CHECK(
        false,
        "Unknown nonlinearity: ",
        torch::enumtype::get_enum_name(options.nonlinearity()));
  }

  // 如果输入未批处理，则去除结果的批处理维度
  if (!is_batched) {
    ret = ret.squeeze(0);
  }

  // 返回前向传播计算的结果
  return ret;
}

// 获取当前 RNN 单元的非线性函数类型名称
std::string RNNCellImpl::get_nonlinearity_str() const {
  return get_enum_name(options.nonlinearity());
}

// 实现 LSTMCellImpl 类的构造方法，初始化 LSTM 单元的选项及参数
LSTMCellImpl::LSTMCellImpl(const LSTMCellOptions& options_)
    : detail::RNNCellImplBase<LSTMCellImpl>(detail::RNNCellOptionsBase(
          options_.input_size(),
          options_.hidden_size(),
          options_.bias(),
          /*num_chunks=*/4)),
      options(options_) {}

// 实现 LSTMCellImpl 类的 forward 方法，用于执行 LSTM 单元的前向传播
std::tuple<Tensor, Tensor> LSTMCellImpl::forward(
    const Tensor& input,
    torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  // 检查输入是否符合要求，确保输入张量 input 的有效性
  this->check_forward_input(input, "input");
  // 如果有隐藏状态作为输入，同样需要检查其有效性
  if (hx_opt.has_value()) {
    this->check_forward_input(std::get<0>(hx_opt.value()), "hx[0]");
    this->check_forward_input(std::get<1>(hx_opt.value()), "hx[1]");
  }

  // 定义变量 r_hx 和 ret，用于存储处理后的隐藏状态和前向传播结果
  std::tuple<Tensor, Tensor> r_hx, ret;

  // 判断输入是否批处理，以确定是否需要对输入进行扩展
  bool is_batched = input.dim() == 2;
  Tensor r_input = is_batched ? input : input.unsqueeze(0);

  // 如果未提供隐藏状态作为输入，则初始化为全零张量
  if (!hx_opt.has_value()) {
    auto zeros = torch::zeros(
        {input.size(0), options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
    r_hx = std::make_tuple(zeros, zeros);
  } else {
    // 如果输入已经批处理，则直接使用 hx_opt；否则对其进行扩展以适应批处理
    if (!is_batched) {
      r_hx = std::make_tuple(
          std::get<0>(hx_opt.value()).unsqueeze(0),
          std::get<1>(hx_opt.value()).unsqueeze(0));
    } else {
      r_hx = hx_opt.value();
    }
  }

  // 调用 Torch 库中的 LSTM 单元计算方法，计算前向传播结果
  ret = torch::lstm_cell(
      r_input,
      {std::get<0>(r_hx), std::get<1>(r_hx)},
      weight_ih,
      weight_hh,
      bias_ih,
      bias_hh);

  // 如果输入未批处理，则去除结果的批处理维度
  if (!is_batched) {
    ret = std::make_tuple(
        std::get<0>(ret).squeeze(0), std::get<1>(ret).squeeze(0));
  }

  // 返回前向传播计算的结果
  return ret;
}
    : detail::RNNCellImplBase<GRUCellImpl>(detail::RNNCellOptionsBase(
          options_.input_size(),
          options_.hidden_size(),
          options_.bias(),
          /*num_chunks=*/3)),
      options(options_) {}



// 创建一个 GRU 单元的实现基类 detail::RNNCellImplBase，并使用 RNNCellOptionsBase 对象初始化它。
// RNNCellOptionsBase 构造函数参数解释：
//   - options_.input_size(): 输入大小
//   - options_.hidden_size(): 隐藏状态大小
//   - options_.bias(): 是否使用偏置
//   - /*num_chunks=*/3: num_chunks 参数设置为 3
// 初始化列表继续：
//   - options(options_): 使用成员初始化列表初始化 options 成员变量
// 实现 GRU 单元的前向传播函数，接收输入和隐藏状态作为参数，并返回输出张量
Tensor GRUCellImpl::forward(const Tensor& input, Tensor hx) {
  // 检查输入的合法性，确保输入张量有效
  this->check_forward_input(input, "input");
  // 检查隐藏状态的合法性，确保隐藏状态张量有效
  this->check_forward_input(hx, "hidden");

  Tensor r_hx, ret;

  // 判断输入张量是否批量化，即是否为二维张量
  bool is_batched = input.dim() == 2;
  // 如果不是批量化的，将输入张量转为批量化形式
  Tensor r_input = is_batched ? input : input.unsqueeze(0);

  // 如果隐藏状态未定义，则初始化为零张量
  if (!hx.defined()) {
    r_hx = torch::zeros(
        {input.size(0), options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    // 如果输入张量是批量化的，则直接使用给定的隐藏状态张量
    // 否则将隐藏状态张量转为批量化形式
    r_hx = is_batched ? hx : hx.unsqueeze(0);
  }

  // 调用 Torch 的 GRU 单元函数进行前向传播计算
  ret = torch::gru_cell(r_input, r_hx, weight_ih, weight_hh, bias_ih, bias_hh);

  // 如果输入张量不是批量化的，将输出张量的额外维度去除，使其与输入形状匹配
  if (!is_batched) {
    ret = ret.squeeze(0);
  }

  // 返回前向传播计算得到的输出张量
  return ret;
}

// 结束 nn 命名空间
} // namespace nn
// 结束 torch 命名空间
} // namespace torch
```