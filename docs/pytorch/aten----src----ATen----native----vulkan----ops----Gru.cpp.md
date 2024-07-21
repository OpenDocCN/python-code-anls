# `.\pytorch\aten\src\ATen\native\vulkan\ops\Gru.cpp`

```
    // 循环中的第i层GRU操作
    // 提取当前层的隐藏状态并将其压缩为2D维度
    auto h = at::slice(hx_vk, 0, i, i + 1, 1);
    h = h.reshape({h.size(0) * h.size(1), h.size(2)});

    // 拼接权重和偏置
    Tensor weight_ih = params_cpu[0 * num_layers + i];
    Tensor weight_hh = params_cpu[1 * num_layers + i];
    Tensor bias_ih = params_cpu[2 * num_layers + i];
    Tensor bias_hh = params_cpu[3 * num_layers + i];

    // 使用Vulkan后端的GRU操作计算
    auto gru_output = at::native::vulkan::ops::Gru(
        x,
        h,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh);

    // 将当前层的输出添加到输出列表中
    h_n_list.emplace_back(gru_output[1].reshape({1, batch_size, hidden_size}));
}

// 拼接所有层的隐藏状态作为输出h_n
Tensor h_n = at::cat(h_n_list, 0);

// 根据batch_first选项调整输出的形状
Tensor output = batch_first ? h_n.reshape({batch_size, seq_length, hidden_size}) :
                              h_n.reshape({seq_length, batch_size, hidden_size});

// 返回GRU操作的输出和最终的隐藏状态
return std::make_tuple(output, h_n);
    // 从参数数组中获取当前时间步的输入权重、隐藏状态权重、输入偏置和隐藏状态偏置
    const auto& w_ih = params_cpu[i * 4];
    const auto& w_hh = params_cpu[i * 4 + 1];
    const auto& b_ih = params_cpu[i * 4 + 2];
    const auto& b_hh = params_cpu[i * 4 + 3];

    // 将权重和偏置按照隐藏状态的大小分割成若干部分
    const auto& w_i_rzn = w_ih.split(hidden_size);
    const auto& w_h_rzn = w_hh.split(hidden_size);
    const auto& b_i_rzn = b_ih.split(hidden_size);
    const auto& b_h_rzn = b_hh.split(hidden_size);

    // 获取分割后的权重和偏置的各个部分（如重置门、更新门、新内容）
    const auto& w_ir = w_i_rzn[0];
    const auto& w_iz = w_i_rzn[1];
    const auto& w_in = w_i_rzn[2];
    const auto& w_hr = w_h_rzn[0];
    const auto& w_hz = w_h_rzn[1];
    const auto& w_hn = w_h_rzn[2];
    const auto& b_ir = b_i_rzn[0];
    const auto& b_iz = b_i_rzn[1];
    const auto& b_in = b_i_rzn[2];
    const auto& b_hr = b_h_rzn[0];
    const auto& b_hz = b_h_rzn[1];
    const auto& b_hn = b_h_rzn[2];

    // 计算重置门、更新门和新内容，使用 sigmoid 和 tanh 函数
    const auto& r = at::sigmoid(
        at::addmm(b_ir, x, w_ir.t()) + at::addmm(b_hr, h, w_hr.t()));
    const auto& z = at::sigmoid(
        at::addmm(b_iz, x, w_iz.t()) + at::addmm(b_hz, h, w_hz.t()));
    const auto& n = at::tanh(
        at::addmm(b_in, x, w_in.t()) + r * (at::addmm(b_hn, h, w_hn.t())));

    // 更新隐藏状态 h
    h = (z * (-1) + 1) * n + z * h;
    x = h; // 下一个时间步的输入为当前时间步的隐藏状态 h

    // 将更新后的隐藏状态 h 转换成 4 维张量，并加入列表 h_n_list
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 将 2 维转换为 4 维以便进行拼接操作
  }

  // 将 h_n_list 中的所有张量在第 1 维上进行拼接，得到最终的隐藏状态张量 h_n
  auto h_n = at::cat(h_n_list, 1);

  // 将 x 调整形状为 (batch_size, seq_length, x.size(1))
  x = x.reshape({batch_size, seq_length, x.size(1)});

  // 调整 h_n 的形状为 (batch_size * seq_length, h_n.size(2), h_n.size(3))
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});

  // 返回包含 x 和 h_n 的 std::tuple 对象
  return std::tuple<Tensor, Tensor>(x, h_n);
} // 结束 namespace

#ifdef USE_VULKAN_API

// 在 Vulkan API 被使用时注册 ATen 库的实现，添加 GRU 输入的实现
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::gru.input"), TORCH_FN(gru_input));
}

#endif /* USE_VULKAN_API */

} // 结束 namespace

// 打包线性操作的上下文数据
std::vector<c10::intrusive_ptr<LinearPackedContext>> pack_linear_op_contexts(
    const std::vector<Tensor>& params_cpu, // CPU 上的权重/偏置
    int64_t num_layers) { // 层数
  // 检查参数是否符合 Vulkan GRU 的预期
  TORCH_CHECK(
      static_cast<int64_t>(params_cpu.size()) == 4 * num_layers,
      "Vulkan gru expects 'params_cpu' size to be 4 * 'num_layers'."
      " But 'params_cpu' has size: ",
      params_cpu.size(),
      " and 'num_layers' is: ",
      num_layers);

  // 初始化线性操作上下文的容器
  std::vector<c10::intrusive_ptr<LinearPackedContext>> linear_op_contexts;
  linear_op_contexts.reserve(num_layers * 6);

  // 遍历每一层 GRU 的参数
  for (int64_t i = 0; i < num_layers; ++i) {
    // 提取当前层的权重和偏置
    const auto& w_ih = params_cpu.at(i * 4);
    const auto& w_hh = params_cpu.at(i * 4 + 1);
    const auto& b_ih = params_cpu.at(i * 4 + 2);
    const auto& b_hh = params_cpu.at(i * 4 + 3);

    // 计算隐藏单元的大小
    const auto& hidden_size = w_ih.size(0) / 3;

    // 将权重和偏置按照隐藏单元大小分割为三部分：reset、update、new gates
    const auto& w_i_rzn = w_ih.split(hidden_size);
    const auto& w_h_rzn = w_hh.split(hidden_size);
    const auto& b_i_rzn = b_ih.split(hidden_size);
    const auto& b_h_rzn = b_hh.split(hidden_size);

    // 提取每个分割部分的具体数据
    const auto& w_ir = w_i_rzn[0];
    const auto& w_iz = w_i_rzn[1];
    const auto& w_in = w_i_rzn[2];
    const auto& w_hr = w_h_rzn[0];
    const auto& w_hz = w_h_rzn[1];
    const auto& w_hn = w_h_rzn[2];
    const auto& b_ir = b_i_rzn[0];
    const auto& b_iz = b_i_rzn[1];
    const auto& b_in = b_i_rzn[2];
    const auto& b_hr = b_h_rzn[0];
    const auto& b_hz = b_h_rzn[1];
    const auto& b_hn = b_h_rzn[2];

    // 创建并添加线性操作上下文：权重转置并带有偏置的线性层
    linear_op_contexts.emplace_back(create_linear_context(w_ir.t(), b_ir));
    linear_op_contexts.emplace_back(create_linear_context(w_hr.t(), b_hr));
    linear_op_contexts.emplace_back(create_linear_context(w_iz.t(), b_iz));
    linear_op_contexts.emplace_back(create_linear_context(w_hz.t(), b_hz));
    linear_op_contexts.emplace_back(create_linear_context(w_in.t(), b_in));
    linear_op_contexts.emplace_back(create_linear_context(w_hn.t(), b_hn));
  }

  // 返回打包好的线性操作上下文
  return linear_op_contexts;
}

// GRU 网络的打包上下文构造函数
GruPackedContext::GruPackedContext(
    const std::vector<Tensor>& params_cpu, // 权重/偏置（CPU 上）
    bool has_biases, // 是否包含偏置
    int64_t num_layers, // 层数
    double dropout, // dropout 比率
    bool train, // 是否为训练模式
    bool bidirectional, // 是否双向
    bool batch_first) {
```  
# 接受一个名为 `batch_first` 的布尔型参数，用于指示是否使用批量优先模式。

  TORCH_INTERNAL_ASSERT(
      has_biases, "Vulkan gru expects 'has_biases' to be true.");
```  
# 使用 `TORCH_INTERNAL_ASSERT` 断言确保 `has_biases` 参数为真，否则输出错误信息。

  TORCH_INTERNAL_ASSERT(!train, "Vulkan gru expects 'train' to be false.");
```  
# 使用 `TORCH_INTERNAL_ASSERT` 断言确保 `train` 参数为假，否则输出错误信息。

  TORCH_INTERNAL_ASSERT(
      !bidirectional, "Vulkan gru expects 'bidirectional' to be false.");
```  
# 使用 `TORCH_INTERNAL_ASSERT` 断言确保 `bidirectional` 参数为假，否则输出错误信息。

  TORCH_INTERNAL_ASSERT(
      dropout < std::numeric_limits<double>::epsilon() * 1000,
      "Vulkan gru expects 'dropout' to be 0.0.");
```  
# 使用 `TORCH_INTERNAL_ASSERT` 断言确保 `dropout` 参数小于数值上限的浮点数精度乘以1000，否则输出错误信息。

  packed_.reserve(Packed::NumArgs);
```  
# 预留存储空间以存放 `Packed::NumArgs` 个元素的数据。

  packed_.emplace_back(pack_linear_op_contexts(params_cpu, num_layers));
```  
# 将通过 `pack_linear_op_contexts` 函数打包的 `params_cpu` 和 `num_layers` 数据插入到 `packed_` 向量的末尾。

  packed_.emplace_back(has_biases);
```  
# 将 `has_biases` 参数插入到 `packed_` 向量的末尾。

  packed_.emplace_back(num_layers);
```  
# 将 `num_layers` 参数插入到 `packed_` 向量的末尾。

  packed_.emplace_back(dropout);
```  
# 将 `dropout` 参数插入到 `packed_` 向量的末尾。

  packed_.emplace_back(train);
```  
# 将 `train` 参数插入到 `packed_` 向量的末尾。

  packed_.emplace_back(bidirectional);
```  
# 将 `bidirectional` 参数插入到 `packed_` 向量的末尾。

  packed_.emplace_back(batch_first);
```  
# 将 `batch_first` 参数插入到 `packed_` 向量的末尾。
}

// 将未打包的通用列表 `unpacked` 打包成 `GruPackedContext` 对象
GruPackedContext GruPackedContext::pack(c10::impl::GenericList unpacked) {
  // 从未打包的通用列表中提取参数，创建 `GruPackedContext` 对象
  return GruPackedContext(
      unpacked.get(Unpacked::Params).toTensorVector(), // 提取参数并转换为张量向量
      unpacked.get(Unpacked::hasBiases).toBool(), // 提取是否具有偏置并转换为布尔值
      unpacked.get(Unpacked::NumLayers).toInt(), // 提取层数并转换为整数
      unpacked.get(Unpacked::Dropout).toDouble(), // 提取dropout并转换为双精度浮点数
      unpacked.get(Unpacked::Train).toBool(), // 提取训练状态并转换为布尔值
      unpacked.get(Unpacked::Bidirectional).toBool(), // 提取是否双向并转换为布尔值
      unpacked.get(Unpacked::BatchFirst).toBool()); // 提取是否批量优先并转换为布尔值
}

// 解包 `GruPackedContext` 对象为未打包的通用列表
const c10::impl::GenericList GruPackedContext::unpack() const {
  // 创建空的未打包的通用列表 `unpacked_gru_context`
  c10::impl::GenericList unpacked_gru_context{c10::AnyType::get()};
  // 预留空间以容纳 `Unpacked::NumArgs` 个元素
  unpacked_gru_context.reserve(Unpacked::NumArgs);

  // 从打包的线性上下文中提取线性上下文的列表
  const c10::List<c10::IValue> packed_linear_contexts =
      get_val(Packed::LinearContexts).toList();

  // 提取层数
  const int64_t num_layers = get_val(Packed::NumLayers).toInt();
  // 每层线性上下文的数量
  const int64_t linear_contexts_per_layer = 6;

  // 创建存储参数的 CPU 张量向量
  std::vector<Tensor> params_cpu;
  params_cpu.reserve(num_layers * linear_contexts_per_layer);

  // 遍历打包的线性上下文
  for (c10::IValue packed_linear_context : packed_linear_contexts) {
    // 解包单个线性上下文
    const c10::impl::GenericList unpacked_linear_context =
        packed_linear_context.toCustomClass<LinearPackedContext>()->unpack();

    // 检查解包的线性上下文是否为空
    TORCH_CHECK(
        unpacked_linear_context.size() > 0u,
        "unpacked_linear_context does not have any elements!");

    // 提取权重并转置，加入参数列表
    params_cpu.emplace_back(
        unpacked_linear_context.get(LinearPackedContext::Unpacked::Weight)
            .toTensor()
            .t());
    // 提取偏置，加入参数列表
    params_cpu.emplace_back(
        unpacked_linear_context.get(LinearPackedContext::Unpacked::Bias)
            .toTensor());
  }
  
  // 将参数列表加入未打包的通用列表 `unpacked_gru_context`
  unpacked_gru_context.emplace_back(params_cpu);

  // 将其余的元素依次加入未打包的通用列表 `unpacked_gru_context`
  for (int64_t i = 1; i < Unpacked::NumArgs; ++i) {
    unpacked_gru_context.emplace_back(get_val(i));
  }

  // 返回解包后的通用列表 `unpacked_gru_context`
  return unpacked_gru_context;
}

// 创建并返回一个指向 `GruPackedContext` 对象的智能指针
c10::intrusive_ptr<GruPackedContext> create_gru_context(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // 使用给定参数创建 `GruPackedContext` 对象，并返回其智能指针
  return c10::make_intrusive<GruPackedContext>(GruPackedContext(
      params_cpu,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first));
}

// 运行 `GruPackedContext` 对象，返回一个张量元组
std::tuple<Tensor, Tensor> run_gru_context(
    const Tensor& input_vk, // 输入序列（Vulkan）
    const Tensor& hx_vk, // 初始隐藏状态（Vulkan）
    // 检查输入张量 `input_vk` 的维度是否为3
    // 如果不是，抛出错误信息，指出 Vulkan GRU 需要 'input_vk' 的维度为3
    TORCH_INTERNAL_ASSERT(
        input_vk.sizes().size() == 3,
        "Vulkan gru expects 'input_vk' dims to be 3.");
    
    // 检查隐藏状态张量 `hx_vk` 的维度是否为3
    // 如果不是，抛出错误信息，指出 Vulkan GRU 需要 'hx_vk' 的维度为3
    TORCH_INTERNAL_ASSERT(
        hx_vk.sizes().size() == 3, "Vulkan gru expects 'hx_vk' dims to be 3.");
    
    // 获取 GRU 上下文中的层数信息，并转换为 int64_t 类型
    const int64_t num_layers =
        gru_context->get_val(GruPackedContext::Packed::NumLayers).toInt();
    
    // 获取 GRU 上下文中的 BatchFirst 参数，并转换为 bool 类型
    const bool batch_first =
        gru_context->get_val(GruPackedContext::Packed::BatchFirst).toBool();
    
    // 获取输入张量 `input_vk` 的批量大小
    const auto batch_size = input_vk.size(0);
    
    // 获取输入张量 `input_vk` 的序列长度
    const auto seq_length = input_vk.size(1);
    
    // 检查输入张量的形状是否符合 Vulkan GRU 的要求
    // 要求是 batch_size 和 seq_length 都为1，或者 BatchFirst 为 true
    // 如果不符合，抛出错误信息
    TORCH_INTERNAL_ASSERT(
        (batch_size == 1 && seq_length == 1) || batch_first,
        "Vulkan gru expects batch-first input");
    
    // 从 GRU 上下文中获取打包的线性层上下文列表
    const c10::List<c10::IValue> packed_linear_contexts =
        gru_context->get_val(GruPackedContext::Packed::LinearContexts).toList();
    
    // 定义每层线性上下文的数量
    const int64_t linear_contexts_per_layer = 6;
    
    // 创建一个空的张量向量，用于存储隐藏状态的输出
    std::vector<at::Tensor> h_n_list; // hidden output
    
    // 将输入张量 `input_vk` 重塑为二维张量，以便使用 Vulkan 的 at::mm 操作
    auto x = input_vk.reshape({batch_size * seq_length, input_vk.size(2)});
    
    // 循环遍历每一层 GRU
    for (int64_t i = 0; i < num_layers; ++i) {
      // 提取每个隐藏状态并将其压缩成二维维度
      auto h = at::slice(hx_vk, 0, i, i + 1, 1);
      h = h.reshape({h.size(0) * h.size(1), h.size(2)});
    
      // 从打包的线性上下文列表中提取各种线性层上下文
      const auto& cxt_ir =
          packed_linear_contexts[i * linear_contexts_per_layer + 0]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_hr =
          packed_linear_contexts[i * linear_contexts_per_layer + 1]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_iz =
          packed_linear_contexts[i * linear_contexts_per_layer + 2]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_hz =
          packed_linear_contexts[i * linear_contexts_per_layer + 3]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_in =
          packed_linear_contexts[i * linear_contexts_per_layer + 4]
              .toCustomClass<LinearPackedContext>();
      const auto& cxt_hn =
          packed_linear_contexts[i * linear_contexts_per_layer + 5]
              .toCustomClass<LinearPackedContext>();
    
      // 计算重要门控信号 r，使用 sigmoid 函数
      const auto& r = at::sigmoid(
          run_linear_context(x, cxt_ir) + run_linear_context(h, cxt_hr));
    
      // 计算更新门控信号 z，使用 sigmoid 函数
      const auto& z = at::sigmoid(
          run_linear_context(x, cxt_iz) + run_linear_context(h, cxt_hz));
    
      // 计算新的隐藏状态候选值 n，使用 tanh 函数
      const auto& n = at::tanh(
          run_linear_context(x, cxt_in) + r * run_linear_context(h, cxt_hn));
    
      // 更新隐藏状态 h，根据 GRU 的公式计算
      h = (z * (-1) + 1) * n + z * h;
    
      // 下一个输入是当前隐藏状态 h
      x = h;
    h_n_list.emplace_back(
        h.reshape({1, 1, h.size(0), h.size(1)})); // 将 h 变形为四维张量，以便进行拼接操作后添加到 h_n_list 中
  }

  // 拼接 h_n_list 中的所有张量，沿着第1维度（列维度）进行拼接
  auto h_n = at::cat(h_n_list, 1);

  // 将输入张量 x 变形为指定的形状：batch_size × seq_length × x.size(1)
  x = x.reshape({batch_size, seq_length, x.size(1)});

  // 将 h_n 变形为指定的形状：(h_n.size(0) * h_n.size(1)) × h_n.size(2) × h_n.size(3)
  h_n = h_n.reshape({h_n.size(0) * h_n.size(1), h_n.size(2), h_n.size(3)});

  // 返回包含两个张量的 std::tuple，分别是变形后的 x 和 h_n
  return std::tuple<Tensor, Tensor>(x, h_n);
} // end of namespace ops
} // end of namespace vulkan
} // end of namespace native
} // end of namespace at
```