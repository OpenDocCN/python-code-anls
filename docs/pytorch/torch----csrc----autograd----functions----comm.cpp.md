# `.\pytorch\torch\csrc\autograd\functions\comm.cpp`

```py
namespace torch {
namespace autograd {
// 定义 Scatter 类的构造函数，初始化成员变量
Scatter::Scatter(
    std::vector<at::Device> devices, // 所需的设备列表
    std::optional<std::vector<int64_t>> chunk_sizes, // 可选的块大小列表
    int64_t dim, // 操作的维度
    std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams, // 可选的 CUDA 流列表
    bool unsqueeze_scalars) // 是否展开标量
    : devices_(std::move(devices)), // 初始化设备列表
      chunk_sizes_(std::move(chunk_sizes)), // 初始化块大小列表
      dim_(dim), // 初始化操作维度
      streams_(std::move(streams)), // 初始化 CUDA 流列表
      unsqueeze_scalars_(unsqueeze_scalars) {} // 初始化展开标量的选项

// Scatter 类析构函数
Scatter::~Scatter() = default;

// Scatter 类的 apply 方法，接收输入并返回输出变量列表
variable_list Scatter::apply(variable_list&& inputs) {
  AT_ASSERT(inputs.size() == 1); // 断言输入变量列表仅包含一个元素
  auto& input = inputs.front(); // 获取输入的第一个元素

  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad(input)) { // 如果需要计算梯度
    grad_fn = std::make_shared<Gather>(/*destination_device=*/input.device(), dim_); // 创建 Gather 函数节点
    grad_fn->set_next_edges(collect_next_edges(input)); // 设置下一步边缘
  }

  auto device_indices = fmap(devices_, [](const at::Device& device) -> int64_t {
    return device.index(); // 获取设备索引
  });
  auto tensors =
      torch::cuda::scatter(input, device_indices, chunk_sizes_, dim_, streams_); // 执行 scatter 操作

  std::vector<Variable> variables;
  variables.reserve(tensors.size());
  for (auto& tensor : tensors) { // 遍历结果张量列表
    AT_ASSERT(tensor.defined()); // 断言张量已定义
    if (unsqueeze_scalars_) { // 如果需要展开标量
      AT_ASSERT(tensor.dim() == 1 && tensor.numel() == 1); // 断言张量是一维且包含一个元素
      variables.push_back(tensor[0]); // 将标量张量的值作为变量列表的元素
    } else {
      variables.push_back(std::move(tensor)); // 将张量移动到变量列表中
    }
  }

  if (grad_fn) { // 如果存在梯度函数节点
    set_history(variables, grad_fn); // 设置变量的历史记录
  }

  return variables; // 返回变量列表
}

// Gather 类的构造函数，初始化目标设备和操作维度
Gather::Gather(const at::Device& destination_device, int64_t dim)
    : destination_device_(destination_device), dim_(dim) {}

// Gather 类析构函数
Gather::~Gather() = default;

// Gather 类的 apply 方法，接收输入并返回输出变量列表
variable_list Gather::apply(variable_list&& inputs) {
  bool all_are_zero_dim = true;
  for (const auto& input : inputs) { // 遍历输入变量列表
    TORCH_CHECK(
        input.is_cuda(), // 断言输入变量是 CUDA 张量
        "All inputs to Gather must be CUDA tensors, got ",
        input.toString()); // 输出错误信息和变量的字符串表示
    if (input.dim() > 0) {
      all_are_zero_dim = false; // 更新是否所有输入变量都是零维的标志
    }
  }

  const bool unsqueeze_scalars = all_are_zero_dim && dim_ == 0; // 计算是否需要展开标量
  if (unsqueeze_scalars) {
    TORCH_WARN(
        "Was asked to gather along dimension 0, but all "
        "input tensors were scalars; will instead unsqueeze "
        "and return a vector."); // 发出警告，因为所有输入都是标量
  }

  std::shared_ptr<Node> grad_fn;
  // 在从输入中移动变量之前计算此项
  if (compute_requires_grad(inputs)) { // 如果需要计算梯度
    std::vector<at::Device> source_devices;
    source_devices.reserve(inputs.size()); // 为源设备列表分配空间
    std::vector<int64_t> input_sizes;
    input_sizes.reserve(inputs.size()); // 为输入尺寸列表分配空间
    for (auto& input : inputs) {
      source_devices.push_back(input.device()); // 添加输入的设备到源设备列表
      input_sizes.push_back(input.size(dim_)); // 添加输入的维度大小到输入尺寸列表
    }

      }

    }
  }

  // 创建 Gather 函数节点，用于梯度计算
  grad_fn = std::make_shared<Scatter>(/*destination_device=*/destination_device_, dim_);
  grad_fn->set_next_edges(collect_next_edges(inputs));

  // 返回空的变量列表，因为 Gather 操作不产生输出
  return {};
}
    // 创建一个名为 grad_fn 的智能指针，指向 Scatter 类的实例对象
    grad_fn = std::make_shared<Scatter>(
        std::move(source_devices),  // 使用 std::move 将 source_devices 移动到 Scatter 构造函数中
        std::move(input_sizes),     // 使用 std::move 将 input_sizes 移动到 Scatter 构造函数中
        dim_,                       // 维度参数 dim_
        /*streams=*/c10::nullopt,   // 使用 c10::nullopt 初始化 streams 参数
        /*unsqueeze_scalars=*/unsqueeze_scalars);  // 使用 unsqueeze_scalars 初始化 unsqueeze_scalars 参数
    grad_fn->set_next_edges(collect_next_edges(inputs));  // 调用 grad_fn 的方法设置其下一步的边缘

  }

  // 创建一个存储 Tensor 的向量 tensors，并预留足够的空间以容纳 inputs 的大小
  std::vector<at::Tensor> tensors;
  tensors.reserve(inputs.size());
  for (auto& variable : inputs) {
    // 如果 unsqueeze_scalars 为 true，则将 variable 进行视图展开后加入 tensors
    if (unsqueeze_scalars) {
      tensors.push_back(variable.view(1));
    } else {
      // 否则将 variable 移动到 tensors 中
      tensors.push_back(std::move(variable));
    }
  }

  // 在实际计算过程中禁用自动求导
  // torch::cuda::gather 不返回视图，也不会就地更改，因此这里不需要额外的逻辑
  at::Tensor variable;
  {
    at::AutoDispatchBelowAutograd mode;  // 自动离开当前自动求导分发模式
    // 对于 torch::cuda::gather，特定的逻辑处理
    const auto destination_index =
        destination_device_.is_cpu() ? -1 : destination_device_.index();
    variable = torch::cuda::gather(tensors, dim_, destination_index);  // 使用 torch::cuda::gather 进行张量的收集操作
  }
  if (grad_fn) {
    set_history(variable, grad_fn);  // 如果 grad_fn 存在，则将 variable 和 grad_fn 作为历史记录设置
  }
  // 返回包含 variable 的初始化列表
  return {variable};
}

} // namespace autograd
} // namespace torch
```