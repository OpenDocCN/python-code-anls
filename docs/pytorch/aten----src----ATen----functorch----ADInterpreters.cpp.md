# `.\pytorch\aten\src\ATen\functorch\ADInterpreters.cpp`

```py
// 包含必要的头文件
#include <ATen/functorch/ADInterpreters.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/TensorWrapper.h>
#include <bitset>

// 声明 funtorch 命名空间
namespace at::functorch {

// 默认的位集大小常量
constexpr size_t default_bitset_size = 64;

// 检查在捕获的张量上进行无效变异
static void checkForInvalidMutationOnCaptures(
    const c10::OperatorHandle& op,  // 操作符句柄
    const torch::jit::Stack* stack, // JIT 栈
    int64_t cur_level) {            // 当前层级
  // 如果不是原地操作，则直接返回
  if (!isInplaceOp(op.schema())) {
    return;
  }
  // 获取操作的参数
  auto args = torch::jit::last(stack, op.schema().arguments().size());
  // 解包可能是死亡状态的第一个参数张量
  auto mutated_arg = unwrapIfDead(args[0].toTensor());
  // 获取张量的包装器
  auto* wrapper = maybeGetTensorWrapper(mutated_arg);
  // 如果存在包装器，并且其层级等于当前层级，并且不是不可变的，则直接返回
  if (wrapper && wrapper->level().has_value() && wrapper->level().value() == cur_level && !(wrapper->is_immutable())) {
    return;
  }
  // 抛出错误，指示在梯度转换中，函数尝试调用原地操作，这会修改捕获的张量，这是不支持的
  TORCH_CHECK(false,
      "During a grad (vjp, jvp, grad, etc) transform, the function provided ",
      "attempted to call in-place operation (", op.schema().operator_name(), ") ",
      "that would mutate a captured Tensor. This is not supported; please rewrite ",
      "the function being transformed to explicitly accept the mutated Tensor(s) ",
      "as inputs.");
}

// 实例化梯度包装器
static Tensor materializeGradWrappers(const Tensor& tensor, int64_t current_level) {
  // 如果张量未定义，则直接返回
  if (!tensor.defined()) {
    return tensor;
  }
  // 在创建 TensorWrapper 时可能调用分发操作，需要确保它们通过 functorch 栈正确传递
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
  // 获取张量的包装器
  auto* wrapper = maybeGetTensorWrapper(tensor);
  // 如果没有包装器，则创建一个不可变的张量包装器
  if (!wrapper) {
    return makeTensorWrapper(tensor, current_level, /*is_immutable=*/true);
  }
  // 内部断言确保包装器的层级小于等于当前层级
  TORCH_INTERNAL_ASSERT(wrapper->level().value() <= current_level, "escaped?");
  // 如果包装器的层级等于当前层级，则返回原始张量
  if (wrapper->level().value() == current_level) {
    TORCH_INTERNAL_ASSERT(tensor.defined());
    return tensor;
  }
  // 否则，创建一个不可变的张量包装器
  return makeTensorWrapper(tensor, current_level, /*is_immutable=*/true);
}

// 通过梯度解释器指针提升张量
Tensor GradInterpreterPtr::lift(const Tensor& tensor) const {
  return materializeGradWrappers(tensor, level());
}

// 通过 JvpInterpreterPtr 提升张量
Tensor JvpInterpreterPtr::lift(const Tensor& tensor) const {
  return materializeGradWrappers(tensor, level());
}

// 基于自动微分的变换过程
static void autogradBasedTransformProcess(
    const c10::OperatorHandle& op,  // 操作符句柄
    torch::jit::Stack* stack,       // JIT 栈
    int64_t current_level,          // 当前层级
    TransformType transform_type) { // 变换类型
  // 如果是梯度变换，并且操作是原地的，并且变异参数当前未包装在 TensorWrapper 中，则报错
  checkForInvalidMutationOnCaptures(op, stack, current_level);

  // 实例化活跃的 GradWrappers
  auto maybeTransformGradWrappers = [&](const Tensor& tensor) {
    // 调用 materializeGradWrappers 函数处理 tensor，并返回处理后的结果
    return materializeGradWrappers(tensor, current_level);
  };

  // 获取操作符 op 的参数数量
  auto num_args = op.schema().arguments().size();

  // 遍历 stack 中从倒数第 num_args 个元素到最后一个元素的张量，
  // 对每个张量调用 maybeTransformGradWrappers 函数进行可能的梯度包装转换
  foreachTensorInplace(*stack, static_cast<int64_t>(stack->size() - num_args), static_cast<int64_t>(stack->size()), maybeTransformGradWrappers);

  // 设置当前的调度键(transform_type)，并清空线程本地存储
  setup_dispatch_key_tls(transform_type, {});

  // 调用操作符 op 的 callBoxed 方法，传入 stack 中的数据进行执行
  op.callBoxed(stack);
  // `autogradBasedTransformSendToNext`函数用于在自动求导过程中转换和发送操作至下一级别的解释器。
  static void autogradBasedTransformSendToNext(
      // `op`是操作的句柄，用于访问操作的模式和参数
      const c10::OperatorHandle& op,
      // `stack`是操作的栈，其中包含操作的输入和输出
      torch::jit::Stack* stack,
      // `interpreter`是当前的解释器对象
      const Interpreter& interpreter,
      // `transform_type`指定操作的转换类型，如Grad或Jvp
      TransformType transform_type,
      // `prev_grad_mode`记录之前的梯度模式是否启用
      optional<bool> prev_grad_mode,
      // `prev_fwd_grad_mode`记录之前的前向梯度模式是否启用
      optional<bool> prev_fwd_grad_mode,
      // `grad_special_case`标志是否存在梯度特殊情况
      bool grad_special_case) {
    // 获取当前解释器的级别
    auto current_level = interpreter.level();

    // 如果转换类型为Grad，确保prev_grad_mode有值
    if (transform_type == TransformType::Grad) {
      TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
    }

    // 如果转换类型为Jvp，确保prev_fwd_grad_mode有值
    if (transform_type == TransformType::Jvp) {
      TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
    }

    // 定义一个函数unwrap，用于根据当前级别解开张量包装
    auto unwrap = [&](const Tensor& tensor) {
      if (!tensor.defined()) {
        return tensor;
      }
      auto* maybe_tensor_wrapper = maybeGetTensorWrapper(tensor);
      if (!maybe_tensor_wrapper) {
        return tensor;
      }
      auto tensor_wrapper_level = maybe_tensor_wrapper->level().value();
      // 确保张量包装的级别不超过当前级别
      TORCH_INTERNAL_ASSERT(tensor_wrapper_level <= current_level);
      if (tensor_wrapper_level == current_level) {
        return maybe_tensor_wrapper->value();
      }
      return tensor;
    };

    // 定义一个函数wrap，用于将张量包装成特定级别的包装器
    auto wrap = [&](const Tensor& tensor, bool is_immutable) {
      if (!tensor.defined()) {
        return tensor;
      }
      // 如果张量包装使能显示调度追踪，则输出当前级别
      // if (c10::show_dispatch_trace_enabled()) {
      //   std::cout << "wrap " << current_level << std::endl;
      // }
      return makeTensorWrapper(tensor, interpreter, is_immutable);
    };

    // TODO: 我们只需要对修改尺寸或步幅的原位函数执行以下操作（用！标记）。
    // 这类函数并不多见。
    // 如果是自动求导分派键：
    // 1. (!) 将所有参数的副本放入堆栈
    // 2. 对副本集合中的所有参数执行unwrap操作
    // 3. 调用操作
    // 4. 对输出进行包装
    // 5. (!) 对原始集合中所有参数执行refreshMetadata操作
    // 6. (!) 弹出这些参数。

    // 步骤 1 & 2
    auto args_size = op.schema().arguments().size();
    const auto ret_size = op.schema().returns().size();
    // 步骤 1：计算要处理的参数的起始位置
    auto front = static_cast<int64_t>(stack->size()) - args_size;
    // 将所有参数的副本放入堆栈
    for (const auto arg_idx : c10::irange(0, args_size)) {
      stack->push_back((*stack)[front + arg_idx]);
    }

    // 定义一个位集合，用于标记所有位为1
    std::bitset<default_bitset_size> outputs_aliasing_immutable;

    // 如果不是梯度特殊情况，执行以下操作
    if (!grad_special_case) {
    // 遍历栈中从 idx 到栈顶的元素
    for (auto idx = stack->size() - args_size; idx < stack->size(); idx++) {
      const auto ivalue = (*stack)[idx];
      // 如果 ivalue 不是 Tensor 类型，则跳过处理，因为只有 Tensor 可能被别名，而不是 Tensor 列表（除非在没有返回值的操作中）
      if (!ivalue.isTensor()) {
        continue;
      }
      // 获取当前的 Tensor 引用
      const auto& tensor = ivalue.toTensor();
      // 获取 tensor 对应的包装器（如果有）
      auto* maybe_tensor_wrapper = maybeGetTensorWrapper(tensor);
      // 如果没有找到包装器或者该包装器标记为不可变，则执行以下操作
      if (!maybe_tensor_wrapper || maybe_tensor_wrapper->is_immutable()) {
        // 如果输入是不可变的，则查找它是否别名了任何内容
        // 注意，args 在栈上是逆序的，因此最后一个参数在栈顶
        const auto relative_pos = idx - (stack->size() - args_size);
        // 查找与当前输入别名的输出索引
        const auto aliased_out = findAliasedOutput(op.schema(), static_cast<int64_t>(relative_pos));
        // 如果找到了别名的输出索引，则设置 outputs_aliasing_immutable 中对应位置的标志位
        if (aliased_out.has_value()) {
          outputs_aliasing_immutable.flip(*aliased_out);
          // 每个输出最多只能别名一个输入，因此只会命中一次这个分支
        }
      }
    }
  }

  // Step 2：对栈中的元素应用 unwrap 函数
  foreachTensorInplace(*stack, static_cast<int64_t>(stack->size() - args_size), static_cast<int64_t>(stack->size()), unwrap);

  // NOTE [grad and vjp interaction with no_grad]：关于 grad 和 vjp 与 no_grad 的交互的注释

  // 根据 transform_type 和前一个梯度模式设置 grad_guard 和 fw_grad_guard
  optional<c10::AutoGradMode> grad_guard;
  if (transform_type == TransformType::Grad && prev_grad_mode.has_value() && *prev_grad_mode == false) {
    grad_guard.emplace(*prev_grad_mode);
  }
  optional<c10::AutoFwGradMode> fw_grad_guard;
  if (transform_type == TransformType::Jvp &&
      prev_fwd_grad_mode.has_value() && *prev_fwd_grad_mode == false) {
    fw_grad_guard.emplace(*prev_fwd_grad_mode);
  }

  // 如果动态层栈为空，则进行操作前的栈的健全性检查
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }

  // Step 4, 5, 6：调用 op 对象的 callBoxed 方法执行步骤 4、5、6

  // Step 4：对栈中的元素应用 wrap 函数，并且使用 outputs_aliasing_immutable 标志位
  foreachTensorInplaceWithFlag(*stack, static_cast<int64_t>(stack->size() - ret_size), static_cast<int64_t>(stack->size()), outputs_aliasing_immutable, wrap);

  // Step 5：刷新栈中输入 Tensor 的元数据
  auto args_front = stack->size() - args_size - ret_size;
  for (const auto arg_idx : c10::irange(0, args_size)) {
    auto& ivalue = (*stack)[args_front + arg_idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    auto maybe_tensor_wrapper = maybeGetTensorWrapper(ivalue.toTensor());
    if (!maybe_tensor_wrapper) {
      continue;
    }
    maybe_tensor_wrapper->refreshMetadata();
  }

  // Step 6：从栈中移除前 args_size + ret_size 个元素
  stack->erase(stack->end() - std::ptrdiff_t(args_size + ret_size), stack->end() - std::ptrdiff_t(ret_size));
// GradInterpreterPtr 类的方法实现，处理梯度相关的操作
void GradInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,  // 接收一个操作符句柄作为参数
    torch::jit::Stack* stack) {     // 接收一个指向 Torch JIT 栈的指针作为参数
  // 调用 autogradBasedTransformProcess 函数，处理梯度相关的转换操作
  autogradBasedTransformProcess(op, stack, level(), TransformType::Grad);
}

// GradInterpreterPtr 类的方法实现，将操作传递给下一个解释器处理
void GradInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,  // 接收一个操作符句柄作为参数
    torch::jit::Stack* stack,       // 接收一个指向 Torch JIT 栈的指针作为参数
    bool grad_special_case) {       // 表示梯度特殊情况的布尔值参数
  // 调用 autogradBasedTransformSendToNext 函数，将操作转发给下一个解释器处理
  autogradBasedTransformSendToNext(
      op, stack, *base_,
      TransformType::Grad,          // 指定转换类型为梯度
      prevGradMode(),               // 使用先前的梯度模式
      nullopt,                      // 不传递任何正向梯度模式选项
      grad_special_case);           // 传递梯度特殊情况的标志
}

// JvpInterpreterPtr 类的方法实现，处理 JVP（Jacobian Vector Product）相关的操作
void JvpInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,  // 接收一个操作符句柄作为参数
    torch::jit::Stack* stack) {     // 接收一个指向 Torch JIT 栈的指针作为参数
  // 调用 autogradBasedTransformProcess 函数，处理 JVP 相关的转换操作
  autogradBasedTransformProcess(op, stack, level(), TransformType::Jvp);
}

// JvpInterpreterPtr 类的方法实现，将操作传递给下一个解释器处理
void JvpInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,  // 接收一个操作符句柄作为参数
    torch::jit::Stack* stack,       // 接收一个指向 Torch JIT 栈的指针作为参数
    bool grad_special_case) {       // 表示梯度特殊情况的布尔值参数
  // 调用 autogradBasedTransformSendToNext 函数，将操作转发给下一个解释器处理
  autogradBasedTransformSendToNext(
      op, stack, *base_,
      TransformType::Jvp,           // 指定转换类型为 JVP
      nullopt,                      // 不传递任何梯度模式选项
      prevFwdGradMode(),            // 使用先前的正向传播梯度模式
      grad_special_case);           // 传递梯度特殊情况的标志
}

// 命名空间结束标志，声明所有定义属于 at::functorch 命名空间
} // namespace at::functorch
```