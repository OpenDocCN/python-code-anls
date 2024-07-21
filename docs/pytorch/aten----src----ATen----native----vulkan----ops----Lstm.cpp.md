# `.\pytorch\aten\src\ATen\native\vulkan\ops\Lstm.cpp`

```py
    double scale_in, // input scale factor
    double scale_hx, // hidden state scale factor
    double scale_cx, // cell state scale factor
    double scale_out, // output scale factor
    double scale_hn, // final hidden state scale factor
    double scale_cn, // final cell state scale factor
    double scale_ih, // input-to-hidden scale factor
    double scale_ch, // hidden-to-hidden scale factor
    double scale_bias, // bias scale factor
    c10::optional<Tensor> output_scale,
    const Tensor& output_scale_value,
    const Tensor& output_scale_value_value,
    const std::string& activation,
    const std::string& activation_value,
    const std::string& activation_function,
    const std::string activation_function_function,
    at::ArrayRef<int64_t> batch_first,
    const at::TensorOptions& input_option,
    const at::TensorOptions input_option_options,
    const at::TensorOptions input_option_option,
    const std::vector<at::TensorOptions>& options,
    const std::vector<at::TensorOptions>& options_options,
    const std::vector<at::TensorOptions>& options_option,
    const std::vector<at::TensorOptions>& options_option_options) {

    // Ensure input tensor is not empty and has compatible dimensions
    TORCH_CHECK(input_vk.dim() == 3,
                "Expected 3-dimensional input_vk tensor, but got ", input_vk.dim());
    TORCH_CHECK(input_vk.size(0) > 0 && input_vk.size(1) > 0 && input_vk.size(2) > 0,
                "Expected input_vk tensor to have non-empty dimensions, but got sizes (",
                input_vk.size(0), ", ", input_vk.size(1), ", ", input_vk.size(2), ")");
    
    // Check initial hidden and cell states have compatible dimensions
    for (const auto& hx_i : hx) {
        TORCH_CHECK(hx_i.dim() == 3,
                    "Expected 3-dimensional hidden state tensor, but got ", hx_i.dim());
        TORCH_CHECK(hx_i.size(1) == input_vk.size(1),
                    "Expected hidden state tensor to have the same batch size as input_vk tensor, but got sizes ",
                    hx_i.size(1), " and ", input_vk.size(1));
    }

    // Check weights and biases tensors have correct dimensions and types
    for (const auto& param_cpu : params_cpu) {
        TORCH_CHECK(param_cpu.is_cuda(),
                    "Expected weights/biases tensor to be on CUDA device, but found it on ", param_cpu.device());
        TORCH_CHECK(param_cpu.dim() == 2 || param_cpu.dim() == 3,
                    "Expected weights/biases tensor to be 2-dimensional or 3-dimensional, but got ", param_cpu.dim());
    }

    // Determine the number of directions (unidirectional or bidirectional LSTM)
    const int64_t directions = bidirectional ? 2 : 1;

    // Compute the output tensor shape based on batch_first flag and LSTM direction
    std::vector<int64_t> output_shape;
    if (batch_first.empty() || !batch_first[0]) {
        output_shape = {input_vk.size(0), input_vk.size(1), num_layers * directions * input_vk.size(2)};
    } else {
        output_shape = {input_vk.size(1), input_vk.size(0), num_layers * directions * input_vk.size(2)};
    }

    // Create empty tensors for output features, final hidden state, and final cell state
    Tensor output = at::empty(output_shape, input_vk.options());
    Tensor h_n = at::zeros({num_layers * directions, input_vk.size(1), input_vk.size(2)}, input_vk.options());
    Tensor c_n = at::zeros({num_layers * directions, input_vk.size(1), input_vk.size(2)}, input_vk.options());

    // Return tuple of output tensors
    return std::make_tuple(output, h_n, c_n);
}
    // 检查第一个隐藏状态和第二个隐藏状态的最后一个维度是否相同，如果不同则抛出错误
    TORCH_CHECK(
        hx[0].size(2) == hx[1].size(2),
        "Vulkan LSTM with projections is not supported");
    
    // 检查params_cpu的大小是否为4 * num_layers，否则抛出错误
    TORCH_CHECK(
        static_cast<int64_t>(params_cpu.size()),
        "Vulkan LSTM expects 'params_cpu' size to be 4 * 'num_layers'.");
    
    // 内部断言，验证input_vk的维度是否为3，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        input_vk.sizes().size() == 3, "Vulkan LSTM expects input dims to be 3.");
    
    // 内部断言，验证hx[0]的维度是否为3，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        hx[0].sizes().size() == 3,
        "Vulkan LSTM expects hidden state dims to be 3.");
    
    // 内部断言，验证hx[1]的维度是否为3，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        hx[1].sizes().size() == 3,
        "Vulkan LSTM expects cell state dims to be 3.");
    
    // 内部断言，验证has_biases是否为true，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        has_biases, "Vulkan LSTM expects 'has_biases' to be true.");
    
    // 内部断言，验证train是否为false，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        !train, "Vulkan LSTM expects 'train' to be false.");
    
    // 内部断言，验证bidirectional是否为false，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        !bidirectional, "Vulkan LSTM expects 'bidirectional' to be false.");
    
    // 内部断言，验证dropout是否小于一个极小值的1000倍，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        dropout < std::numeric_limits<double>::epsilon() * 1000,
        "Vulkan LSTM expects 'dropout' to be 0.0.");
    
    // 获取输入张量input_vk的批次大小和序列长度
    const auto batch_size = input_vk.size(0);
    const auto seq_length = input_vk.size(1);
    
    // 内部断言，验证(batch_size == 1 && seq_length == 1) || batch_first是否为真，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        (batch_size == 1 && seq_length == 1) || batch_first,
        "Vulkan gru expects batch-first input");
    
    // 获取隐藏状态张量hx_vk和细胞状态张量cx_vk
    const Tensor& hx_vk = hx[0];
    const Tensor& cx_vk = hx[1];
    
    // 获取隐藏状态的大小（即最后一个维度的大小）
    const auto hidden_size = hx_vk.size(2);
    
    // 创建用于存储每层输出的隐藏状态列表和细胞状态列表
    std::vector<at::Tensor> h_n_list; // hidden state output
    std::vector<at::Tensor> c_n_list; // cell state output
    
    // 将输入张量input_vk重塑为二维张量，因为Vulkan只接受二维张量作为at::mm操作的输入
    auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});
    
    // 预留存储每层隐藏状态和细胞状态的空间
    h_n_list.reserve(num_layers);
    c_n_list.reserve(num_layers);
    
    // 循环处理每一层
    for (int64_t l = 0; l < num_layers; ++l) {
        // 提取每个隐藏状态并将其重塑为二维张量
        auto h = at::slice(hx_vk, 0, l, l + 1, 1);
        h = h.reshape({h.size(0) * h.size(1), h.size(2)});
    
        // 提取每个细胞状态并将其重塑为二维张量
        auto c = at::slice(cx_vk, 0, l, l + 1, 1);
        c = c.reshape({c.size(0) * c.size(1), c.size(2)});
    
        // 提取当前层的权重和偏置
        const auto& w_ih = params_cpu[l * 4];
        const auto& w_hh = params_cpu[l * 4 + 1];
        const auto& b_ih = params_cpu[l * 4 + 2];
        const auto& b_hh = params_cpu[l * 4 + 3];
    
        // 将权重分割成输入门、遗忘门、输出门和细胞门的部分
        const auto& w_i_ifgo = w_ih.split(hidden_size);
        const auto& w_h_ifgo = w_hh.split(hidden_size);
        const auto& b_i_ifgo = b_ih.split(hidden_size);
        const auto& b_h_ifgo = b_hh.split(hidden_size);
    
        // 提取具体的权重和偏置部分
        const auto& w_ii = w_i_ifgo[0];
        const auto& w_if = w_i_ifgo[1];
        const auto& w_ig = w_i_ifgo[2];
        const auto& w_io = w_i_ifgo[3];
        const auto& w_hi = w_h_ifgo[0];
        const auto& w_hf = w_h_ifgo[1];
        const auto& w_hg = w_h_ifgo[2];
        const auto& w_ho = w_h_ifgo[3];
        const auto& b_ii = b_i_ifgo[0];
        const auto& b_if = b_i_ifgo[1];
        const auto& b_ig = b_i_ifgo[2];
        const auto& b_io = b_i_ifgo[3];
        const auto& b_hi = b_h_ifgo[0];
        const auto& b_hf = b_h_ifgo[1];
        const auto& b_hg = b_h_ifgo[2];
        const auto& b_ho = b_h_ifgo[3];
    const auto& i = at::sigmoid(
        at::addmm(b_ii, x, w_ii.t()) + at::addmm(b_hi, h, w_hi.t()));
    // 计算输入门的激活值 i，使用当前输入 x 和隐藏状态 h，分别与权重矩阵 w_ii 和 w_hi 的转置相乘，再加上偏置 b_ii 和 b_hi

    const auto& f = at::sigmoid(
        at::addmm(b_if, x, w_if.t()) + at::addmm(b_hf, h, w_hf.t()));
    // 计算遗忘门的激活值 f，使用当前输入 x 和隐藏状态 h，分别与权重矩阵 w_if 和 w_hf 的转置相乘，再加上偏置 b_if 和 b_hf

    const auto& g =
        at::tanh(at::addmm(b_ig, x, w_ig.t()) + at::addmm(b_hg, h, w_hg.t()));
    // 计算新记忆细胞的候选值 g，使用当前输入 x 和隐藏状态 h，分别与权重矩阵 w_ig 和 w_hg 的转置相乘，再加上偏置 b_ig 和 b_hg

    const auto& o = at::sigmoid(
        at::addmm(b_io, x, w_io.t()) + at::addmm(b_ho, h, w_ho.t()));
    // 计算输出门的激活值 o，使用当前输入 x 和隐藏状态 h，分别与权重矩阵 w_io 和 w_ho 的转置相乘，再加上偏置 b_io 和 b_ho

    c = f * c + i * g;
    // 更新细胞状态 c，使用遗忘门 f 来遗忘旧状态，使用输入门 i 和新记忆细胞候选值 g 来更新细胞状态

    h = o * at::tanh(c);
    // 更新隐藏状态 h，使用输出门 o 控制细胞状态 c 的输出，通过双曲正切函数进行非线性变换

    x = h; // 下一个时间步的输入是当前的隐藏状态 h

    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 将隐藏状态 h 转换为四维张量，用于后续的连接操作

    c_n_list.emplace_back(
        c.reshape({1, 1, c.size(0), c.size(1)})); // 将细胞状态 c 转换为四维张量，用于后续的连接操作
  }

  auto h_n = at::cat(h_n_list, 1);
  // 在维度 1 上连接所有隐藏状态 h，构成最终的隐藏状态张量 h_n

  auto c_n = at::cat(c_n_list, 1);
  // 在维度 1 上连接所有细胞状态 c，构成最终的细胞状态张量 c_n

  x = x.reshape({batch_size, seq_length, x.size(1)});
  // 将最终的输入状态 x 重新调整形状，以匹配批次大小和序列长度的要求

  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  // 将最终的隐藏状态张量 h_n 重新调整形状，以适应连接操作后的维度要求

  c_n = c_n.reshape({c_n.size(0) * c_n.size(1), c_n.size(2), c_n.size(3)});
  // 将最终的细胞状态张量 c_n 重新调整形状，以适应连接操作后的维度要求

  return std::tuple<Tensor, Tensor, Tensor>(x, h_n, c_n);
  // 返回包含输入状态 x、隐藏状态张量 h_n 和细胞状态张量 c_n 的元组
} // 结束前面的代码块或函数实现

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则进入 Vulkan 特定的实现部分

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 在 Torch 库中注册 Vulkan 版本的 LSTM 输入实现
  m.impl(TORCH_SELECTIVE_NAME("aten::lstm.input"), TORCH_FN(lstm_input));
}
#endif /* USE_VULKAN_API */ 

} // 结束匿名命名空间

// 开始定义 pack_lstm_linear_op_contexts 函数，用于打包 LSTM 线性操作上下文
std::vector<c10::intrusive_ptr<LinearPackedContext>>
pack_lstm_linear_op_contexts(
    const std::vector<Tensor>& params_cpu, // 输入参数列表，包含权重和偏置（CPU 版本）
    int64_t num_layers) { // LSTM 层数

  // 检查参数列表的大小是否符合 Vulkan LSTM 的要求
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan LSTM expects 'params_cpu' size to be 4 * 'num_layers'."
      " But 'params_cpu' has size: ",
      params_cpu.size(),
      " and 'num_layers' is: ",
      num_layers);

  // 准备存储线性操作上下文的向量，预留空间
  std::vector<c10::intrusive_ptr<LinearPackedContext>> linear_op_contexts;
  linear_op_contexts.reserve(num_layers * 8);

  // 循环遍历每一层 LSTM
  for (int64_t l = 0; l < num_layers; ++l) {
    // 提取当前层的权重和偏置
    const auto& w_ih = params_cpu[l * 4];
    const auto& w_hh = params_cpu[l * 4 + 1];
    const auto& b_ih = params_cpu[l * 4 + 2];
    const auto& b_hh = params_cpu[l * 4 + 3];

    // 计算隐藏单元大小
    const auto& hidden_size = w_ih.size(0) / 4;

    // 将权重和偏置按照输入门、遗忘门、输出门和细胞状态分割
    const auto& w_i_ifgo = w_ih.split(hidden_size);
    const auto& w_h_ifgo = w_hh.split(hidden_size);
    const auto& b_i_ifgo = b_ih.split(hidden_size);
    const auto& b_h_ifgo = b_hh.split(hidden_size);

    // 提取并创建每个门的权重和偏置
    const auto& w_ii = w_i_ifgo[0];
    const auto& w_if = w_i_ifgo[1];
    const auto& w_ig = w_i_ifgo[2];
    const auto& w_io = w_i_ifgo[3];
    const auto& w_hi = w_h_ifgo[0];
    const auto& w_hf = w_h_ifgo[1];
    const auto& w_hg = w_h_ifgo[2];
    const auto& w_ho = w_h_ifgo[3];
    const auto& b_ii = b_i_ifgo[0];
    const auto& b_if = b_i_ifgo[1];
    const auto& b_ig = b_i_ifgo[2];
    const auto& b_io = b_i_ifgo[3];
    const auto& b_hi = b_h_ifgo[0];
    const auto& b_hf = b_h_ifgo[1];
    const auto& b_hg = b_h_ifgo[2];
    const auto& b_ho = b_h_ifgo[3];

    // 创建线性操作上下文并加入到向量中
    linear_op_contexts.emplace_back(create_linear_context(w_ii.t(), b_ii));
    linear_op_contexts.emplace_back(create_linear_context(w_hi.t(), b_hi));
    linear_op_contexts.emplace_back(create_linear_context(w_if.t(), b_if));
    linear_op_contexts.emplace_back(create_linear_context(w_hf.t(), b_hf));
    linear_op_contexts.emplace_back(create_linear_context(w_ig.t(), b_ig));
    linear_op_contexts.emplace_back(create_linear_context(w_hg.t(), b_hg));
    linear_op_contexts.emplace_back(create_linear_context(w_io.t(), b_io));
    linear_op_contexts.emplace_back(create_linear_context(w_ho.t(), b_ho));
  }

  // 返回线性操作上下文向量
  return linear_op_contexts;
}

// 开始定义 LstmPackedContext 的构造函数，接收参数列表和其他设置
LstmPackedContext::LstmPackedContext(
    const std::vector<Tensor>& params_cpu, // 权重和偏置（CPU 版本）
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    // 断言：Vulkan LSTM 要求 'has_biases' 必须为 true
    TORCH_INTERNAL_ASSERT(
        has_biases, "Vulkan LSTM expects 'has_biases' to be true.");

    // 断言：Vulkan LSTM 要求 'train' 必须为 false
    TORCH_INTERNAL_ASSERT(!train, "Vulkan LSTM expects 'train' to be false.");

    // 断言：Vulkan LSTM 要求 'bidirectional' 必须为 false
    TORCH_INTERNAL_ASSERT(
        !bidirectional, "Vulkan LSTM expects 'bidirectional' to be false.");

    // 断言：Vulkan LSTM 要求 'dropout' 必须为接近于0的值
    TORCH_INTERNAL_ASSERT(
        dropout < std::numeric_limits<double>::epsilon() * 1000,
        "Vulkan LSTM expects 'dropout' to be 0.0.");

    // 预留空间以存储参数
    packed_.reserve(Packed::NumArgs);

    // 将 LSTM 线性操作上下文打包并加入 packed_ 中
    packed_.emplace_back(pack_lstm_linear_op_contexts(params_cpu, num_layers));

    // 加入 has_biases 到 packed_ 中
    packed_.emplace_back(has_biases);

    // 加入 num_layers 到 packed_ 中
    packed_.emplace_back(num_layers);

    // 加入 dropout 到 packed_ 中
    packed_.emplace_back(dropout);

    // 加入 train 到 packed_ 中
    packed_.emplace_back(train);

    // 加入 bidirectional 到 packed_ 中
    packed_.emplace_back(bidirectional);

    // 加入 batch_first 到 packed_ 中
    packed_.emplace_back(batch_first);
}

// 将未打包的上下文数据封装成 LstmPackedContext 对象
LstmPackedContext LstmPackedContext::pack(c10::impl::GenericList unpacked) {
  return LstmPackedContext(
      unpacked.get(Unpacked::Params).toTensorVector(),  // 提取参数并转换为张量向量
      unpacked.get(Unpacked::hasBiases).toBool(),       // 提取是否包含偏置信息
      unpacked.get(Unpacked::NumLayers).toInt(),        // 提取层数并转换为整数
      unpacked.get(Unpacked::Dropout).toDouble(),       // 提取 dropout 率并转换为双精度浮点数
      unpacked.get(Unpacked::Train).toBool(),           // 提取训练标志并转换为布尔值
      unpacked.get(Unpacked::Bidirectional).toBool(),   // 提取双向标志并转换为布尔值
      unpacked.get(Unpacked::BatchFirst).toBool());     // 提取 batch_first 标志并转换为布尔值
}

// 解包 LstmPackedContext 对象，返回 GenericList
const c10::impl::GenericList LstmPackedContext::unpack() const {
  c10::impl::GenericList unpacked_lstm_context{c10::AnyType::get()};  // 创建一个空的 GenericList

  unpacked_lstm_context.reserve(Unpacked::NumArgs);  // 预留足够的空间以容纳解包后的数据

  const c10::List<c10::IValue> packed_linear_contexts =  // 获取打包的线性上下文列表
      get_val(Packed::LinearContexts).toList();

  const int64_t num_layers = get_val(Packed::NumLayers).toInt();  // 获取层数
  const int64_t linear_contexts_per_layer = 8;                     // 每层线性上下文的数量为 8

  std::vector<Tensor> params_cpu;  // 创建一个存储 CPU 张量的向量
  params_cpu.reserve(num_layers * linear_contexts_per_layer);  // 预留足够的空间以容纳所有参数

  // 遍历打包的线性上下文列表
  for (c10::IValue packed_linear_context : packed_linear_contexts) {
    const c10::impl::GenericList unpacked_linear_context =  // 解包每个打包的线性上下文
        packed_linear_context.toCustomClass<LinearPackedContext>()->unpack();

    TORCH_CHECK(
        unpacked_linear_context.size() > 0u,
        "unpacked_linear_context does not have any elements!");  // 检查解包后的线性上下文是否为空

    // 提取并转换权重，然后将其转置后存入 params_cpu
    params_cpu.emplace_back(
        unpacked_linear_context.get(LinearPackedContext::Unpacked::Weight)
            .toTensor()
            .t());
    // 提取并转换偏置后存入 params_cpu
    params_cpu.emplace_back(
        unpacked_linear_context.get(LinearPackedContext::Unpacked::Bias)
            .toTensor());
  }

  // 将 params_cpu 添加到 unpacked_lstm_context 中作为一个
    // 对于 Vulkan LSTM 模型，验证输入张量的维度必须为3
    TORCH_INTERNAL_ASSERT(
        input_vk.sizes().size() == 3, "Vulkan LSTM expects input dims to be 3.");
    // 对于 Vulkan LSTM 模型，验证隐藏状态张量的维度必须为3
    TORCH_INTERNAL_ASSERT(
        hx_vk.sizes().size() == 3,
        "Vulkan LSTM expects hidden state dims to be 3.");
    // 对于 Vulkan LSTM 模型，验证单元状态张量的维度必须为3
    TORCH_INTERNAL_ASSERT(
        cx_vk.sizes().size() == 3,
        "Vulkan LSTM expects cell state dims to be 3.");
    
    // 从 LSTM 上下文对象中获取层数信息
    const int64_t num_layers =
        lstm_context->get_val(LstmPackedContext::Packed::NumLayers).toInt();
    // 从 LSTM 上下文对象中获取是否是 batch-first 的信息
    const bool batch_first =
        lstm_context->get_val(LstmPackedContext::Packed::BatchFirst).toBool();
    // 获取输入张量的 batch size
    const auto batch_size = input_vk.size(0);
    // 获取输入张量的序列长度
    const auto seq_length = input_vk.size(1);
    
    // 如果 batch size 和序列长度为1，或者标志指示为 batch-first，则通过验证
    TORCH_INTERNAL_ASSERT(
        (batch_size == 1 && seq_length == 1) || batch_first,
        "Vulkan gru expects batch-first input");
    
    // 从 LSTM 上下文对象中获取线性操作的上下文列表
    const c10::List<c10::IValue> packed_linear_op_contexts =
        lstm_context->get_val(LstmPackedContext::Packed::LinearContexts).toList();
    
    // 每层的线性操作上下文数目为8个
    const int64_t linear_op_contexts_per_layer = 8;
    
    // 创建用于存储隐藏状态输出的向量
    std::vector<at::Tensor> h_n_list; // hidden state output
    // 创建用于存储单元状态输出的向量
    std::vector<at::Tensor> c_n_list; // cell state output
    
    // 将输入张量重新形状为2D，以便于 Vulkan 环境下的矩阵乘法操作
    auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});
    
    // 预留隐藏状态向量和单元状态向量的存储空间
    h_n_list.reserve(num_layers);
    c_n_list.reserve(num_layers);
    
    // 循环处理每一层的状态
    for (int64_t l = 0; l < num_layers; ++l) {
      // 提取每层的隐藏状态并将其压缩到2D维度
      auto h = at::slice(hx_vk, 0, l, l + 1, 1);
      h = h.reshape({h.size(0) * h.size(1), h.size(2)});
    
      // 提取每层的单元状态并将其压缩到2D维度
      auto c = at::slice(cx_vk, 0, l, l + 1, 1);
      c = c.reshape({c.size(0) * c.size(1), c.size(2)});
    
      // 从线性操作上下文列表中获取每种操作的上下文信息
      const auto& cxt_ii =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 0]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_hi =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 1]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_if =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 2]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_hf =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 3]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_ig =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 4]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_hg =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 5]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_io =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 6]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_ho =
          packed_linear_op_contexts[l * linear_op_contexts_per_layer + 7]
              .toCustomClass<LinearPackedContext>();
    const auto& i = at::sigmoid(
        run_linear_context(x, cxt_ii) + run_linear_context(h, cxt_hi));
    // 计算输入门 i，使用当前输入 x 和上下文 cxt_ii，h 和上下文 cxt_hi 的线性运算结果
    const auto& f = at::sigmoid(
        run_linear_context(x, cxt_if) + run_linear_context(h, cxt_hf));
    // 计算遗忘门 f，使用当前输入 x 和上下文 cxt_if，h 和上下文 cxt_hf 的线性运算结果
    const auto& g =
        at::tanh(run_linear_context(x, cxt_ig) + run_linear_context(h, cxt_hg));
    // 计算更新门 g，使用当前输入 x 和上下文 cxt_ig，h 和上下文 cxt_hg 的线性运算结果
    const auto& o = at::sigmoid(
        run_linear_context(x, cxt_io) + run_linear_context(h, cxt_ho));
    // 计算输出门 o，使用当前输入 x 和上下文 cxt_io，h 和上下文 cxt_ho 的线性运算结果
    c = f * c + i * g;
    // 更新细胞状态 c，结合遗忘门 f、输入门 i 和更新门 g 的计算结果
    h = o * at::tanh(c);
    // 更新隐藏状态 h，结合输出门 o 和当前细胞状态 c 的 tanh 函数计算结果
    x = h; // 下一个输入为当前的隐藏状态 h
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 将隐藏状态 h 从2维转为4维，以便进行连接操作
    c_n_list.emplace_back(
        c.reshape({1, 1, c.size(0), c.size(1)})); // 将细胞状态 c 从2维转为4维，以便进行连接操作
  }

  auto h_n = at::cat(h_n_list, 1);
  // 将 h_n_list 中的隐藏状态 h 按第1维度（列维度）连接起来
  auto c_n = at::cat(c_n_list, 1);
  // 将 c_n_list 中的细胞状态 c 按第1维度（列维度）连接起来
  x = x.reshape({batch_size, seq_length, x.size(1)});
  // 将最终的输入 x 重新调整形状为 batch_size * seq_length * x.size(1)
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});
  // 将最终的隐藏状态 h_n 重新调整形状为 (h_n.size(0) * h_n.size(1)) * h_n.size(2) * h_n.size(3)
  c_n = c_n.reshape({c_n.size(0) * c_n.size(1), c_n.size(2), c_n.size(3)});
  // 将最终的细胞状态 c_n 重新调整形状为 (c_n.size(0) * c_n.size(1)) * c_n.size(2) * c_n.size(3)
  return std::tuple<Tensor, Tensor, Tensor>(x, h_n, c_n);
  // 返回由 x、h_n 和 c_n 构成的 std::tuple 对象
}

} // 结束 at 命名空间

} // 结束 native 命名空间

} // 结束 vulkan 命名空间

} // 结束 ops 命名空间
```