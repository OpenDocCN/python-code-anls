# `.\pytorch\torch\csrc\api\include\torch\nn\modules\rnn.h`

```
#pragma once
class TORCH_API RNNImplBase : public torch::nn::Cloneable<Derived> {
 public:
  // 使用 RNNOptionsBase 类型的选项初始化 RNNImplBase 对象
  explicit RNNImplBase(const RNNOptionsBase& options_);

  /// Initializes the parameters of the RNN module.
  // 重置 RNN 模块的参数
  void reset() override;

  // 重置参数的初始化
  void reset_parameters();

  /// Overrides `nn::Module::to()` to call `flatten_parameters()` after the
  /// original operation.
  // 覆盖 `nn::Module::to()` 方法，在原始操作后调用 `flatten_parameters()`
  void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false)
      override;
  void to(torch::Dtype dtype, bool non_blocking = false) override;
  void to(torch::Device device, bool non_blocking = false) override;

  /// Pretty prints the RNN module into the given `stream`.
  // 将 RNN 模块漂亮地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  /// Modifies the internal storage of weights for optimization purposes.
  ///
  /// On CPU, this method should be called if any of the weight or bias vectors
  /// are changed (i.e. weights are added or removed). On GPU, it should be
  /// called __any time the storage of any parameter is modified__, e.g. any
  /// time a parameter is assigned a new value. This allows using the fast path
  /// in cuDNN implementations of respective RNN `forward()` methods. It is
  /// called once upon construction, inside `reset()`.
  // 修改权重的内部存储以进行优化目的
  // 在 CPU 上，如果修改了任何权重或偏置向量（例如添加或删除权重），应调用此方法。
  // 在 GPU 上，应在修改任何参数的存储（例如为参数分配新值）时调用它。
  // 这允许在 cuDNN 实现的相应 RNN `forward()` 方法中使用快速路径。
  // 在构造时调用一次，在 `reset()` 方法内部调用。
  void flatten_parameters();

  // 返回所有权重的向量
  std::vector<Tensor> all_weights() const;

  /// The RNN's options.
  // RNN 的选项
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  RNNOptionsBase options_base;

 protected:
  // 重置 flat_weights_
  // 注意：在移除此功能之前要非常小心，因为第三方设备类型可能依赖此行为来正确地 .to() 像 LSTM 这样的模块。
  void reset_flat_weights();

  // 检查输入的张量和批大小张量
  void check_input(const Tensor& input, const Tensor& batch_sizes) const;

  // 获取预期的隐藏状态大小
  std::tuple<int64_t, int64_t, int64_t> get_expected_hidden_size(
      const Tensor& input,
      const Tensor& batch_sizes) const;

  // 检查隐藏状态大小
  void check_hidden_size(
      const Tensor& hx,
      std::tuple<int64_t, int64_t, int64_t> expected_hidden_size,
      std::string msg = "Expected hidden size {1}, got {2}") const;

  // 检查前向传播参数
  void check_forward_args(Tensor input, Tensor hidden, Tensor batch_sizes)
      const;

  // 对隐藏状态进行排列
  Tensor permute_hidden(Tensor hx, const Tensor& permutation) const;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::string> flat_weights_names_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::vector<std::string>> all_weights_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> flat_weights_;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer Elman RNN module with Tanh or ReLU activation.
/// See https://pytorch.org/docs/main/generated/torch.nn.RNN.html to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::RNNOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// RNN model(RNNOptions(128,
/// 64).num_layers(3).dropout(0.2).nonlinearity(torch::kTanh));
/// ```
class TORCH_API RNNImpl : public detail::RNNImplBase<RNNImpl> {
 public:
  /// 构造函数，使用输入大小和隐藏层大小初始化 RNNImpl 对象
  RNNImpl(int64_t input_size, int64_t hidden_size)
      : RNNImpl(RNNOptions(input_size, hidden_size)) {}
  
  /// 显式构造函数，使用给定的选项初始化 RNNImpl 对象
  explicit RNNImpl(const RNNOptions& options_);

  /// 前向传播函数，接受输入张量和可选的初始隐藏状态，返回输出张量和新的隐藏状态
  std::tuple<Tensor, Tensor> forward(const Tensor& input, Tensor hx = {});

 protected:
  /// 定义默认参数（forward 函数的第二个参数），当参数缺失时使用默认值
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  /// 带压缩输入的前向传播函数，接受压缩输入和可选的初始隐藏状态，返回压缩输出和新的隐藏状态
  std::tuple<torch::nn::utils::rnn::PackedSequence, Tensor>
  forward_with_packed_input(
      const torch::nn::utils::rnn::PackedSequence& packed_input,
      Tensor hx = {});

  /// RNN 的选项对象，存储 RNN 的配置参数
  RNNOptions options;

 protected:
  /// 辅助函数，支持带有压缩输入的前向传播，接受输入、批次大小、排序索引、最大批次大小和隐藏状态
  std::tuple<Tensor, Tensor> forward_helper(
      const Tensor& input,
      const Tensor& batch_sizes,
      const Tensor& sorted_indices,
      int64_t max_batch_size,
      Tensor hx);
};

/// RNNImpl 的 ModuleHolder 子类，用于包装 RNNImpl。
/// 查阅 RNNImpl 类的文档以了解它提供的方法，以及如何使用 torch::nn::RNNOptions 与 RNN。
/// 查阅 ModuleHolder 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(RNN);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 多层长短期记忆（LSTM）模块。
/// 查阅 https://pytorch.org/docs/main/generated/torch.nn.LSTM.html 以了解此模块的确切行为。
///
/// 查阅 torch::nn::LSTMOptions 类的文档以了解此模块支持的构造函数参数。
///
/// Example:
/// ```
/// LSTM model(LSTMOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
/// `LSTMImpl` 类继承自 `detail::RNNImplBase<LSTMImpl>`，实现 LSTM 网络的具体功能。
class TORCH_API LSTMImpl : public detail::RNNImplBase<LSTMImpl> {
 public:
  /// 构造函数，使用输入大小和隐藏层大小初始化 LSTMImpl 对象。
  LSTMImpl(int64_t input_size, int64_t hidden_size)
      : LSTMImpl(LSTMOptions(input_size, hidden_size)) {}
  /// 显式构造函数，使用给定的 LSTMOptions 初始化 LSTMImpl 对象。
  explicit LSTMImpl(const LSTMOptions& options_);

  /// 前向传播函数，接受输入张量和可选的初始隐藏状态，返回输出张量和更新后的隐藏状态。
  std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward(
      const Tensor& input,
      torch::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

 protected:
  /// 宏定义，用于支持默认参数的前向传播函数。
  FORWARD_HAS_DEFAULT_ARGS(
      {1, AnyValue(torch::optional<std::tuple<Tensor, Tensor>>())})

 public:
  /// 前向传播函数，接受压缩输入序列和可选的初始隐藏状态，返回压缩序列的输出和更新后的隐藏状态。
  std::tuple<torch::nn::utils::rnn::PackedSequence, std::tuple<Tensor, Tensor>>
  forward_with_packed_input(
      const torch::nn::utils::rnn::PackedSequence& packed_input,
      torch::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

  /// LSTM 的选项对象，存储 LSTM 的配置选项。
  LSTMOptions options;

 protected:
  /// 检查前向传播参数的有效性，包括输入张量、隐藏状态和批次大小。
  void check_forward_args(
      const Tensor& input,
      std::tuple<Tensor, Tensor> hidden,
      const Tensor& batch_sizes) const;

  /// 返回预期的单元大小，用于输入张量和批次大小的元组。
  std::tuple<int64_t, int64_t, int64_t> get_expected_cell_size(
      const Tensor& input,
      const Tensor& batch_sizes) const;

  /// 将隐藏状态张量根据给定的排列重新排列。
  std::tuple<Tensor, Tensor> permute_hidden(
      std::tuple<Tensor, Tensor> hx,
      const Tensor& permutation) const;

  /// 辅助函数，支持前向传播过程，接受输入张量、批次大小、排序索引、最大批次大小和可选的初始隐藏状态。
  std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_helper(
      const Tensor& input,
      const Tensor& batch_sizes,
      const Tensor& sorted_indices,
      int64_t max_batch_size,
      torch::optional<std::tuple<Tensor, Tensor>> hx_opt);
};

/// `LSTM` 的 `ModuleHolder` 子类，用于管理 `LSTMImpl` 的模块。
/// 请查阅 `LSTMImpl` 类的文档以了解其提供的方法，并查看如何使用 `torch::nn::LSTMOptions` 配置 `LSTM`。
/// 请查阅 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(LSTM);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 多层门控循环单元（GRU）模块。
/// 请查阅 https://pytorch.org/docs/main/generated/torch.nn.GRU.html 以了解该模块的确切行为。
///
/// 请查阅 `torch::nn::GRUOptions` 类的文档以了解可用于此模块的构造函数参数。
///
/// 示例：
/// ```
/// GRU model(GRUOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
/// 定义了 GRUImpl 类，继承自 detail::RNNImplBase<GRUImpl> 类
class TORCH_API GRUImpl : public detail::RNNImplBase<GRUImpl> {
 public:
  /// 构造函数，初始化 GRUImpl 对象，设定输入大小和隐藏状态大小
  GRUImpl(int64_t input_size, int64_t hidden_size)
      : GRUImpl(GRUOptions(input_size, hidden_size)) {}

  /// 显式构造函数，通过 GRUOptions 初始化 GRUImpl 对象
  explicit GRUImpl(const GRUOptions& options_);

  /// 前向传播函数，接收输入张量和可选的初始隐藏状态 hx，默认为空张量
  std::tuple<Tensor, Tensor> forward(const Tensor& input, Tensor hx = {});

 protected:
  /// 定义默认参数的前向传播
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(torch::Tensor())})

 public:
  /// 使用打包输入进行前向传播，返回打包序列和最终隐藏状态
  std::tuple<torch::nn::utils::rnn::PackedSequence, Tensor>
  forward_with_packed_input(
      const torch::nn::utils::rnn::PackedSequence& packed_input,
      Tensor hx = {});

  /// GRU 的选项对象
  GRUOptions options;

 protected:
  /// 辅助函数，执行 GRU 的前向传播，接收输入张量、批次大小、排序索引、最大批次大小和初始隐藏状态
  std::tuple<Tensor, Tensor> forward_helper(
      const Tensor& input,
      const Tensor& batch_sizes,
      const Tensor& sorted_indices,
      int64_t max_batch_size,
      Tensor hx);
};

/// `GRUImpl` 的 `ModuleHolder` 子类。
/// 参见 `GRUImpl` 类的文档，了解其提供的方法，以及使用 `torch::nn::GRUOptions` 与 `GRU` 结合的示例。
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 模块存储语义。
TORCH_MODULE(GRU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCellImplBase
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
/// 所有 RNNCell 实现的基类（用于代码共享）。
template <typename Derived>
class TORCH_API RNNCellImplBase : public torch::nn::Cloneable<Derived> {
 public:
  /// 显式构造函数，通过 RNNCellOptionsBase 初始化 RNNCellImplBase 对象
  explicit RNNCellImplBase(const RNNCellOptionsBase& options_);

  /// 初始化 RNNCell 模块的参数
  void reset() override;

  /// 重置参数
  void reset_parameters();

  /// 在给定流中打印 RNN 模块的格式化字符串
  void pretty_print(std::ostream& stream) const override;

  /// RNNCell 的选项基类
  RNNCellOptionsBase options_base;

  /// 输入到隐藏状态的权重
  Tensor weight_ih;

  /// 隐藏到隐藏状态的权重
  Tensor weight_hh;

  /// 输入到隐藏状态的偏置
  Tensor bias_ih;

  /// 隐藏到隐藏状态的偏置
  Tensor bias_hh;

 protected:
  /// 检查前向输入的有效性，接收输入张量和名称
  void check_forward_input(const Tensor& input, const std::string& name) const;

  /// 虚拟函数，返回非线性函数的字符串表示
  virtual std::string get_nonlinearity_str() const;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 使用 tanh 或 ReLU 非线性函数的 Elman RNN 单元。
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.RNNCell 了解该模块的确切行为。
///
/// 参见 `torch::nn::RNNCellOptions` 类的文档，了解该模块支持的构造函数参数。
///
/// 示例：
/// ```
/// RNNCell model(RNNCellOptions(20,
/// 10).bias(false).nonlinearity(torch::kReLU));
/// ```
class TORCH_API RNNCellImpl : public detail::RNNCellImplBase<RNNCellImpl> {
 public:
  /// 构造函数，初始化 RNNCellImpl 对象，设定输入大小和隐藏状态大小
  RNNCellImpl(int64_t input_size, int64_t hidden_size)
      : RNNCellImpl(RNNCellOptions(input_size, hidden_size)) {}

  /// 显式构造函数，通过 RNNCellOptions 初始化 RNNCellImpl 对象
  explicit RNNCellImpl(const RNNCellOptions& options_);

  /// RNNCell 的前向传播函数，接收输入张量和可选的初始隐藏状态 hx，默认为空张量
  Tensor forward(const Tensor& input, Tensor hx = {});

 protected:
  /// 定义默认参数的前向传播
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  /// RNNCell 的选项对象
  RNNCellOptions options;

 protected:
  /// 返回非线性函数的字符串表示，重写基类方法
  std::string get_nonlinearity_str() const override;
};
/// A `ModuleHolder` subclass for `RNNCellImpl`.
/// See the documentation for `RNNCellImpl` class to learn what methods it
/// provides, and examples of how to use `RNNCell` with
/// `torch::nn::RNNCellOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(RNNCell);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTMCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A long short-term memory (LSTM) cell.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.LSTMCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LSTMCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LSTMCell model(LSTMCellOptions(20, 10).bias(false));
/// ```
class TORCH_API LSTMCellImpl : public detail::RNNCellImplBase<LSTMCellImpl> {
 public:
  /// Constructor for LSTMCellImpl.
  /// Initializes an LSTM cell with given input size and hidden size.
  LSTMCellImpl(int64_t input_size, int64_t hidden_size)
      : LSTMCellImpl(LSTMCellOptions(input_size, hidden_size)) {}

  /// Explicit constructor for LSTMCellImpl.
  /// Initializes an LSTM cell with specified options.
  explicit LSTMCellImpl(const LSTMCellOptions& options_);

  /// Forward function for the LSTM cell.
  /// Computes the output and new hidden state based on the input and
  /// optional initial hidden state.
  std::tuple<Tensor, Tensor> forward(
      const Tensor& input,
      torch::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

 protected:
  /// Default arguments configuration for the forward function.
  FORWARD_HAS_DEFAULT_ARGS(
      {1, AnyValue(torch::optional<std::tuple<Tensor, Tensor>>())})

 public:
  /// Options object for configuring the LSTM cell.
  LSTMCellOptions options;
};

/// A `ModuleHolder` subclass for `LSTMCellImpl`.
/// See the documentation for `LSTMCellImpl` class to learn what methods it
/// provides, and examples of how to use `LSTMCell` with
/// `torch::nn::LSTMCellOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(LSTMCell);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRUCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A gated recurrent unit (GRU) cell.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.GRUCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::GRUCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GRUCell model(GRUCellOptions(20, 10).bias(false));
/// ```
class TORCH_API GRUCellImpl : public detail::RNNCellImplBase<GRUCellImpl> {
 public:
  /// Constructor for GRUCellImpl.
  /// Initializes a GRU cell with given input size and hidden size.
  GRUCellImpl(int64_t input_size, int64_t hidden_size)
      : GRUCellImpl(GRUCellOptions(input_size, hidden_size)) {}

  /// Explicit constructor for GRUCellImpl.
  /// Initializes a GRU cell with specified options.
  explicit GRUCellImpl(const GRUCellOptions& options_);

  /// Forward function for the GRU cell.
  /// Computes the output based on the input and optional initial hidden state.
  Tensor forward(const Tensor& input, Tensor hx = {});

 protected:
  /// Default arguments configuration for the forward function.
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  /// Options object for configuring the GRU cell.
  GRUCellOptions options;
};

/// A `ModuleHolder` subclass for `GRUCellImpl`.
/// See the documentation for `GRUCellImpl` class to learn what methods it
/// provides, and examples of how to use `GRUCell` with
/// `torch::nn::GRUCellOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(GRUCell);

} // namespace nn
} // namespace torch
```