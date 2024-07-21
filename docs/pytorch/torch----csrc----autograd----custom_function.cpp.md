# `.\pytorch\torch\csrc\autograd\custom_function.cpp`

```
  // TODO handle multiple levels here
  // 处理多个级别的情况，目前尚未实现

  const auto num_inputs = inputs.size();
  // 获取输入变量列表的大小

  const auto num_outputs = outputs.size();
  // 获取输出变量列表的大小

  // The tracking info below are used to perform the view and inplace checks.
  // They are lazily initialized to reduce the cost of this function in the
  // common case where the user is not using forward mode AD.
  // 下面的跟踪信息用于执行视图和就地检查。
  // 它们是延迟初始化的，以减少在常见情况下（用户未使用前向模式自动微分）此函数的成本。

  variable_list input_grads;
  // 输入梯度列表，用于存储每个输入变量的梯度信息

  std::vector<int64_t> grad_versions;
  // 梯度版本号的向量，记录每个输入变量的梯度版本信息

  std::vector<at::TensorImpl*> grad_impls;
  // 梯度实现的向量，存储每个输入变量的梯度实现信息

  std::unordered_map<at::TensorImpl*, size_t> inputs_bases;
  // 输入基础映射，将每个输入变量的TensorImpl映射到其在inputs中的索引位置

  auto init_tracked_info = [&]() {
    // 初始化跟踪信息的闭包函数

    input_grads.resize(num_inputs);
    // 调整输入梯度列表的大小为num_inputs

    grad_versions.resize(num_inputs);
    // 调整梯度版本号向量的大小为num_inputs

    grad_impls.resize(num_inputs);
    // 调整梯度实现向量的大小为num_inputs

    for (const auto i : c10::irange(num_inputs)) {
      // 遍历输入变量的索引范围

      const auto& inp = inputs[i];
      // 获取当前索引处的输入变量

      if (inp.is_view() && impl::get_view_autograd_meta(inp)->has_fw_view()) {
        // 如果当前输入是视图，并且具有前向视图，则将其基础TensorImpl映射到inputs_bases中
        inputs_bases.emplace(
            impl::get_view_autograd_meta(inp)
                ->get_forward_view()
                .base_.unsafeGetTensorImpl(),
            i);
      } else {
        // 否则直接将当前输入变量的TensorImpl映射到inputs_bases中
        inputs_bases.emplace(inp.unsafeGetTensorImpl(), i);
      }
    }
  };

  bool any_input_has_grad = false;
  // 用于记录是否存在任何输入变量具有梯度信息的标志位

  // Extract the input's forward gradients and record any info we will need
  // later
  // 提取输入变量的前向梯度，并记录稍后我们将需要的任何信息
  for (const auto i : c10::irange(num_inputs)) {
    // 遍历输入变量的索引范围

    const auto& inp = inputs[i];
    // 获取当前索引处的输入变量

    if (!inp.defined()) {
      continue;
      // 如果当前输入变量未定义，则继续下一次循环
    }

    const auto& fw_grad = inp._fw_grad(level);
    // 获取当前输入变量在给定级别(level)下的前向梯度信息
    // 如果存在前向梯度（fw_grad），执行以下操作
    if (fw_grad.defined()) {
      // 如果尚未有任何输入具有梯度，则将标志设置为 true，并初始化跟踪信息
      if (!any_input_has_grad) {
        any_input_has_grad = true;
        init_tracked_info();
      }
      // 将当前输入的前向梯度保存到 input_grads 中
      input_grads[i] = fw_grad;
      // 记录前向梯度的版本号
      grad_versions[i] = fw_grad._version();
      // 获取前向梯度对应的 TensorImpl，并保存到 grad_impls 中
      grad_impls[i] = fw_grad.unsafeGetTensorImpl();
    }
  }

  // 如果没有任何输入具有前向梯度，则直接返回
  if (!any_input_has_grad) {
    return;
  }

  // 禁用自动求导模式，执行 jvp_user_function，并获取所有输出的前向梯度
  torch::autograd::variable_list forward_grads;
  {
    at::AutoFwGradMode fw_grad_mode(false);
    forward_grads = jvp_user_function(inputs, std::move(input_grads));
  }

  // 获取前向梯度的数量
  const auto num_forward_grads = forward_grads.size();
  // 检查返回的前向梯度数量与期望的输出数量是否一致
  TORCH_CHECK(
      num_forward_grads == num_outputs,
      "Function's jvp returned "
      "an invalid number of forward gradients (expected ",
      num_outputs,
      " but got ",
      num_forward_grads,
      ")");

  // 遍历每个输出
  for (const auto i : c10::irange(num_outputs)) {
    // 如果当前输出没有值，跳过
    if (!raw_outputs[i].has_value()) {
      continue;
    }
    // 获取当前输出的值（如果有）
    const auto& out =
        outputs[i].has_value() ? outputs[i].value() : at::Tensor();
    // 获取当前输出对应的 TensorImpl
    auto out_tensor_impl = raw_outputs[i].value().unsafeGetTensorImpl();
    // 检查当前输出是否可导，并且是否为可导类型
    bool is_differentiable =
        (non_differentiable.count(out_tensor_impl) == 0 &&
         isDifferentiableType(raw_outputs[i].value().scalar_type()));
    // 获取当前输出的前向梯度
    const auto& out_grad = forward_grads[i];
    // 如果输出未定义或者不可导，则检查前向梯度是否未定义，否则抛出异常
    if (!out.defined() || !is_differentiable) {
      TORCH_CHECK(
          !out_grad.defined(),
          "Function's jvp returned a gradient at position ",
          i,
          ", but "
          " the corresponding forward output is not a differentiable Tensor."
          "You should return None at that position instead.");
      continue;
    }

    // 检查当前输出的 TensorImpl 是否在输入映射中
    bool is_input = inputs_mapping.count(out_tensor_impl) > 0;
    // 检查当前输出的 TensorImpl 是否在 dirty_inputs 中
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    // 如果标记为已修改
    if (is_modified) {
      // 检查是否为输入张量，仅输入张量应传递给 ctx.mark_dirty()。
      TORCH_CHECK(
          is_input,
          "Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
          " is no need to pass it to mark_dirty().");

      // 获取输出张量对应的输入索引
      auto inp_idx = inputs_mapping[out_tensor_impl];

      // 如果该输入索引对应的梯度实现已存在
      if (grad_impls[inp_idx]) {
        // 如果该输入已有前向梯度
        // 确保其在原地修改并按原样返回
        TORCH_CHECK(
            out_grad._version() != grad_versions[inp_idx],
            "An inplace custom Function is not modifying the "
            "forward mode gradients inplace. If the forward is modifying an input inplace, then the jvp "
            "function must modify the corresponding gradient inplace.")
        TORCH_CHECK(
            out_grad.unsafeGetTensorImpl() == grad_impls[inp_idx],
            "An inplace custom Function is not returning the "
            "forward mode gradients as-is. If the forward is modifying an input inplace, then the jvp "
            "function must modify the gradient inplace and return it as-is.")
      } else {
        // 如果该张量之前没有梯度，设置新返回的梯度
        // 这里也可以使用 inputs[inp_idx]，因为它与 out 是相同的
        out._set_fw_grad(out_grad, level, /* is_inplace_op */ true);
      }
    }
}



// 结束一个静态函数 `_view_as_self_with_no_grad` 的定义

static at::Tensor _view_as_self_with_no_grad(
    const at::Tensor& self,
    const _view_as_self_fn_t& view_as_self_fn) {
  // 以下代码块被用于 _process_backward_mode_ad 中的两个地方：

  // (1) 返回了一个输入，但没有修改它。返回其作为视图，以便我们可以附加一个新的 grad_fn 到 Variable 上。
  // 在 no_grad 模式下运行，模仿前向传播的行为。
  
  // (2) 虽然对于附加 grad_fn 的目的而言并不必要，我们也在输出不可微分（不需要 grad）的情况下调用此函数。
  // 这有助于使自定义前向自动求导的用户体验更一致。我们希望统一地表示，返回输入本身被视为返回 `self.view_as(self)`。

  // 或者，我们在执行此视图时没有禁用前向梯度，但这意味着用户定义的 JVP 可能会被静默忽略。
  
  // 禁用自动求导模式
  at::AutoFwGradMode fw_grad_mode(false);
  // 禁用梯度模式
  AutoGradMode grad_mode(false);

  // 通过 view_as_self_fn lambda 传递自身视图功能，以便在我们是 Python 自定义函数时（而不是 cpp 函数），
  // 我们可以正确地从 Python 调用 view_as，以便 torch 函数逻辑仍然可以触发。
  
  // 调用传入的 view_as_self_fn 函数，返回视图
  return view_as_self_fn(self);
}

// 处理反向自动求导模式 `_process_backward_mode_ad` 的静态函数定义
static optional_variable_list _process_backward_mode_ad(
    const std::unordered_map<at::TensorImpl*, size_t>& inputs_mapping,
    const std::unordered_set<at::TensorImpl*>& non_differentiable,
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,
    const at::ArrayRef<std::optional<Variable>> raw_outputs,
    const std::shared_ptr<Node>& cdata,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context,
    const _view_as_self_fn_t& view_as_self_fn) {
  // 获取原始输出的数量
  auto num_outputs = raw_outputs.size();

  // 错误消息：当一个输入作为输出原样返回时，将其保存用于反向传播是不支持的。
  // 如果您覆盖了 setup_context，应该返回和保存输入的视图，例如使用 x.view_as(x) 或在前向函数本身内设置 ctx。

  const char* error_msg_input_returned_as_is =
      "A input that has been returned as-is as output is being saved for backward. "
      "This is not supported if you override setup_context. You should return and "
      "save a view of the input instead, e.g. with x.view_as(x) or setup ctx inside "
      "the forward function itself.";

  // 设置一个输出 Variable 的 grad_fn 和 output_nr
  auto set_history = [&](Variable& var,
                         uint32_t output_nr,
                         bool is_input,
                         bool is_modified,
                         bool is_differentiable,
                         bool is_saved_and_setup_context) {



// 结束静态函数 `_process_backward_mode_ad` 的定义


这段代码是一个 C++ 的静态函数定义和相关的注释。
    // 如果不可微，则处理不同情况下的变量操作
    if (!is_differentiable) {
      // 如果变量不需要梯度
      if (!var.requires_grad()) {
        // 如果是输入且未修改过，则检查并转换变量视图，不改变 requires_grad 属性
        if (is_input && !is_modified) {
          // 检查是否保存且设置上下文，否则报错
          TORCH_CHECK(
              !is_saved_and_setup_context, error_msg_input_returned_as_is)
          var = _view_as_self_with_no_grad(var, view_as_self_fn);
        }
        // 直接返回，不需进一步处理
        return;
      }
      // 如果是输入变量，返回其分离的别名而不改变其 requires_grad 属性
      if (is_input) {
        var = var.detach();
      } else if (!var.is_view()) {
        var.detach_();
      }
      // 如果 var 是输入变量的视图之一，在 no_grad 块中不进行分离操作，
      // 以便模仿在 no_grad 块中返回视图的行为
    } else if (is_modified) {
      // 如果变量是叶子节点且需要梯度，则报错
      if (var.is_leaf() && var.requires_grad()) {
        TORCH_CHECK(
            false,
            "a leaf Variable that requires grad has been used in an in-place operation.");
      }
      // 非输入变量无需标记为修改过
      if (!is_input) {
        TORCH_WARN(
            "Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
            " is no need to pass it to mark_dirty().");
      }
      // 如果输入是视图，重构将需要重写图，并且只有一个输出时才能工作
      TORCH_CHECK(
          !(var.is_view() && num_outputs > 1),
          "If your Function modifies inplace an input that is a view"
          " of another Tensor, your Function cannot return more than one Tensor. This is not supported"
          " by the current autograd engine. You should either make sure the input is not a view (using"
          " .clone() for example) or make your Function only return one Tensor (potentially splitting"
          " it into two Functions: one doing the inplace that returns a single Tensor and a second one"
          " that does the other operations). You can ask on the forum https://discuss.pytorch.org/ if"
          " you need help to do this change.");

      // 如果输入被修改过，重构 grad_fn 在图中的位置
      var.mutable_grad().reset();
      impl::clear_hooks(var);
      // 尝试获取梯度累加器，重置其 variable
      if (auto grad_acc_fn = impl::try_get_grad_accumulator(var)) {
        auto& grad_acc = dynamic_cast<AccumulateGrad&>(*grad_acc_fn);
        grad_acc.variable.reset();
      }
      // 如果存在上下文数据，重新设定 var 的历史记录
      if (cdata) {
        impl::rebase_history(var, {cdata, output_nr});
      }
    } else if (is_input) {
      // 如果未保存且未设置上下文，则检查并转换变量视图
      TORCH_CHECK(!is_saved_and_setup_context, error_msg_input_returned_as_is)
      var = _view_as_self_with_no_grad(var, view_as_self_fn);
      // 设置梯度边缘，标记梯度传播路径
      impl::set_gradient_edge(var, {cdata, output_nr});
  // 如果条件满足：输出值为存在但是未定义的情况
  } else if (cdata) {
    // 调用内部实现函数，设置梯度边缘，传入变量和输出编号
    impl::set_gradient_edge(var, {cdata, output_nr});
  };

  // 可选的变量列表输出
  optional_variable_list outputs;
  // 用于脏输入检查的无序集合，用于记录输出的实现
  std::unordered_set<at::TensorImpl*> outputs_impl; // For dirty_inputs check
  // 预留足够的空间以容纳输出数量
  outputs.reserve(num_outputs);
  // 计数不同的输出数量
  int num_diff_outputs = 0;

  // 对于输出数量范围内的每个索引 i
  for (const auto i : c10::irange(num_outputs)) {
    // 对于没有原始输出值的情况
    // 插入未定义输入占位符，对于不是张量的输出和张量不可微的情况参见下面的说明
    if (!raw_outputs[i].has_value()) {
      // 如果存在 cdata，向其添加输入元数据
      if (cdata) {
        auto output_nr = cdata->add_input_metadata(Node::undefined_input());
        AT_ASSERT(i == output_nr);
      }
      // 将空的输出插入输出列表并继续下一个循环
      outputs.emplace_back();
      continue;
    }

    // 获取变量 var，假定其包含有效值
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    Variable var = raw_outputs[i].value();

    // 获取输出张量实现
    auto out_tensor_impl = var.unsafeGetTensorImpl();
    // 检查输出是否为输入的一部分
    bool is_input = inputs_mapping.count(out_tensor_impl) > 0;
    // 检查输出是否已修改
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    // 检查输出是否可微分，并且满足其他条件
    bool is_differentiable = cdata &&
        non_differentiable.count(out_tensor_impl) == 0 &&
        isDifferentiableType(var.scalar_type());
    // 检查输出是否在需要保存和设置上下文中
    bool is_saved_and_setup_context =
        to_save_if_setup_context.count(out_tensor_impl) > 0;

    // 如果存在 cdata
    if (cdata) {
      uint32_t output_nr = 0;
      // 如果不可微分，则向 cdata 添加未定义输入的元数据
      if (!is_differentiable) {
        output_nr = cdata->add_input_metadata(Node::undefined_input());
      } else {
        // 否则向 cdata 添加变量的输入元数据
        output_nr = cdata->add_input_metadata(var);
      }
      // 断言输出编号与 i 相等
      AT_ASSERT(i == output_nr);
    }
    // 设置历史记录，传入变量及相关信息
    set_history(
        var,
        i,
        is_input,
        is_modified,
        is_differentiable,
        is_saved_and_setup_context);

    // 在弃用周期中。在前向期间检测到视图不可微的情况下，仅向用户发出警告
    // （如果返回的输入是视图，则不更改标志）。有关为何我们用警告替换一切的详细信息，请参见 NOTE [ View + Inplace detection ]。
    if (!(is_input && is_modified) && var.is_view()) {
      // 如果是视图，则获取视图的自动求导元数据
      auto diff_view_meta = impl::get_view_autograd_meta(var);
      // 设置创建元数据为 IN_CUSTOM_FUNCTION
      diff_view_meta->set_creation_meta(CreationMeta::IN_CUSTOM_FUNCTION);
    }

    // 如果可微分，则递增可微分输出数量计数
    if (is_differentiable) {
      ++num_diff_outputs;
    }

    // 向输出实现集合中插入当前张量实现
    outputs_impl.insert(out_tensor_impl);
    // 将变量 var 插入输出列表
    outputs.emplace_back(var);
  }

  // 如果返回多个可微分输出，则不允许视图进行就地修改。有关更多详细信息，请参见 NOTE [ View + Inplace detection ]。
  if (num_diff_outputs > 1) {
    // 遍历所有输出变量
    for (auto& var : outputs) {
      // 如果变量有值
      if (var.has_value()) {
        // 获取视图的自动求导元数据
        auto diff_view_meta = impl::get_view_autograd_meta(var.value());
        // 如果视图元数据存在且具有反向视图，则设置创建元数据为 MULTI_OUTPUT_NODE
        if (diff_view_meta && diff_view_meta->has_bw_view()) {
          diff_view_meta->set_creation_meta(CreationMeta::MULTI_OUTPUT_NODE);
        }
      }
    }
  }

  // 所有修改过的张量必须按原样返回以使重写有效
  // 遍历所有脏输入
  for (auto& dirty_input : dirty_inputs) {
    TORCH_CHECK(
        outputs_impl.count(dirty_input) > 0,
        "Some elements marked as dirty during the forward method were not returned as output. The"
        " inputs that are modified inplace must all be outputs of the Function.");
  }



# 检查条件，确保在 forward 方法期间标记为脏数据的某些元素未在输出中返回。
# 被就地修改的输入必须全部作为函数的输出返回。



  return outputs;



# 返回函数计算的输出结果。
}

optional_variable_list _wrap_outputs(
    const variable_list& input_vars,   // 输入变量列表，包含要处理的变量
    const std::unordered_set<at::TensorImpl*>& non_differentiable,   // 不可微张量的集合
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,   // 脏输入张量的集合
    const at::ArrayRef<std::optional<Variable>> raw_outputs,   // 原始输出的可选变量数组引用
    const std::shared_ptr<Node>& cdata,   // 共享指针指向节点数据
    const _jvp_fn_t& jvp_user_function,   // JVP（Jacobians of Vector-Valued Functions）用户函数
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context,   // 如果设置上下文，则需要保存的张量的集合
    const _view_as_self_fn_t& view_as_self_fn) {   // 视图自身函数类型

  std::unordered_map<at::TensorImpl*, size_t> inputs_mapping;   // 无序映射，将张量实现映射到索引的大小
  inputs_mapping.reserve(input_vars.size());   // 预留输入变量大小的空间
  for (const auto i : c10::irange(input_vars.size())) {   // 对于输入变量的范围循环
    inputs_mapping.emplace(input_vars[i].unsafeGetTensorImpl(), i);   // 将输入变量的张量实现映射到索引位置
  }

  auto outputs = _process_backward_mode_ad(   // 处理反向模式自动微分
      inputs_mapping,
      non_differentiable,
      dirty_inputs,
      raw_outputs,
      cdata,
      to_save_if_setup_context,
      view_as_self_fn);

  // 这必须发生在后向处理之后，因为我们希望这里的计算跟踪后向模式梯度。
  _process_forward_mode_AD(   // 处理前向模式自动微分
      input_vars,
      std::move(inputs_mapping),
      raw_outputs,
      outputs,
      non_differentiable,
      dirty_inputs,
      jvp_user_function);

  return outputs;   // 返回处理后的输出列表
}

void check_variable_result(
    const at::TensorBase& original,   // 原始张量的基类引用
    const at::TensorBase& result,   // 结果张量的基类引用
    const std::string& hook_name) {   // 钩子名称的字符串引用
  if (!original.options().type_equal(result.options())) {   // 如果原始张量和结果张量的选项不相等
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the type of value (";   // 构建异常信息字符串流
    ss << "was " << original.toString() << " got ";
    ss << result.toString() << ")";
    throw std::runtime_error(ss.str());   // 抛出运行时错误
  }

  if (original.is_cuda() != result.is_cuda()) {   // 如果原始张量和结果张量的 CUDA 状态不同
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the type of value";
    if (original.is_cuda()) {
      ss << " (was CUDA tensor got CPU tensor)";
    } else {
      ss << " (was CPU tensor got CUDA tensor)";
    }
    throw std::runtime_error(ss.str());   // 抛出运行时错误
  }

  if (original.sym_sizes().vec() != result.sym_sizes().vec()) {   // 如果原始张量和结果张量的符号大小不同
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the size of value";
    throw std::runtime_error(ss.str());   // 抛出运行时错误
  }
}

void AutogradContext::save_for_backward(variable_list to_save) {   // 保存反向传播时需要的变量列表
  to_save_ = std::move(to_save);   // 移动赋值到保存列表
}

// The logic for handling saved variables here is the same as
// python_function.cpp See _save_variables() and unpack_saved_variables()
void AutogradContext::save_variables() {   // 保存变量的逻辑与 python_function.cpp 中的 _save_variables() 和 unpack_saved_variables() 相同
  saved_variables_.clear();   // 清空已保存的变量

  auto ptr = grad_fn_.lock();   // 获取梯度函数的共享指针

  for (const auto& var : to_save_) {   // 遍历要保存的变量列表
    // 允许保存空变量
    if (var.defined()) {   // 如果变量被定义
      bool is_output = var.grad_fn().get() == ptr.get();   // 检查是否为输出变量
      saved_variables_.emplace_back(var, is_output);   // 将变量及其是否为输出添加到已保存变量列表
    } else {
      saved_variables_.emplace_back();   // 否则，添加空变量到已保存变量列表
    }
  }
  to_save_.clear();   // 清空要保存的变量列表
}
// 返回保存的变量列表，前提是未释放缓冲区
variable_list AutogradContext::get_saved_variables() const {
  // 检查是否已释放缓冲区，如果是则抛出错误
  TORCH_CHECK(!has_freed_buffers_, ERR_BACKWARD_TWICE);
  // 创建用于存储保存变量的列表，并预留足够的空间
  variable_list saved;
  saved.reserve(saved_variables_.size());
  // 获取指向梯度函数的弱引用，并确保其存在
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  // 遍历保存的变量，通过梯度函数的指针解包并添加到保存列表中
  for (auto& var : saved_variables_) {
    saved.push_back(var.unpack(ptr));
  }
  // 返回保存的变量列表
  return saved;
}

// 判断特定输出边缘索引是否需要输入梯度
bool AutogradContext::needs_input_grad(size_t output_edge_index) const {
  // 获取指向梯度函数的弱引用，并确保其存在
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  // 调用梯度函数的任务函数，判断是否需要计算特定输出边缘的梯度
  return ptr->task_should_compute_output(output_edge_index);
}

// 判断指定索引范围的输入是否需要梯度
bool AutogradContext::needs_input_grad(
    std::initializer_list<IndexRange> idxs) const {
  // 获取指向梯度函数的弱引用，并确保其存在
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  // 调用梯度函数的任务函数，判断是否需要计算指定索引范围的输入梯度
  return ptr->task_should_compute_output(idxs);
}

// 标记输入变量列表为脏（需要更新）
void AutogradContext::mark_dirty(const variable_list& inputs) {
  // 清空当前脏变量列表，并预留足够空间以容纳输入列表的大小
  dirty_inputs_.clear();
  dirty_inputs_.reserve(inputs.size());
  // 遍历输入变量列表，插入其底层实现到脏变量列表中
  for (auto& var : inputs) {
    dirty_inputs_.insert(var.unsafeGetTensorImpl());
  }
}

// 标记输出变量列表为非可微分
void AutogradContext::mark_non_differentiable(const variable_list& outputs) {
  // 清空当前非可微分变量列表，并预留足够空间以容纳输出列表的大小
  non_differentiable_.clear();
  non_differentiable_.reserve(outputs.size());
  // 遍历输出变量列表，插入其底层实现到非可微分变量列表中
  for (auto& var : outputs) {
    non_differentiable_.insert(var.unsafeGetTensorImpl());
  }
}

// 设置是否需要材料化梯度
void AutogradContext::set_materialize_grads(bool value) {
  // 将是否材料化梯度的标志设置为指定值
  materialize_grads_ = value;
}

// 获取并增加脏变量列表的版本号，然后返回脏变量列表的常量引用
const std::unordered_set<at::TensorImpl*>& AutogradContext::get_and_bump_dirty()
    const {
  // 遍历脏变量列表，逐个增加其版本号
  for (auto& var : dirty_inputs_) {
    var->bump_version();
  }
  // 返回脏变量列表的常量引用
  return dirty_inputs_;
}

// 返回非可微分变量列表的常量引用
const std::unordered_set<at::TensorImpl*>& AutogradContext::
    get_non_differentiable() const {
  // 返回非可微分变量列表的常量引用
  return non_differentiable_;
}
} // namespace torch::autograd
```