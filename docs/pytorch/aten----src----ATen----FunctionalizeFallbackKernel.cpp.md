# `.\pytorch\aten\src\ATen\FunctionalizeFallbackKernel.cpp`

```py
// 包含头文件：ATen 核心调度器的分发器
#include <ATen/core/dispatch/Dispatcher.h>
// 包含头文件：ATen 的遗留类型分发
#include <ATen/core/LegacyTypeDispatch.h>
// 包含头文件：ATen 的空张量
#include <ATen/EmptyTensor.h>
// 包含头文件：ATen 的功能性张量包装器
#include <ATen/FunctionalTensorWrapper.h>
// 包含头文件：ATen 的尺寸推断
#include <ATen/InferSize.h>
// 包含头文件：ATen 的张量工具
#include <ATen/TensorUtils.h>
// 包含头文件：torch 库
#include <torch/library.h>
// 包含头文件：C++ 10 的范围工具
#include <c10/util/irange.h>
// 包含头文件：C++ 10 的步幅工具
#include <c10/util/strides.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含以下头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 包含头文件：ATen 核心库
#include <ATen/ATen.h>
// 包含头文件：ATen 的功能函数
#include <ATen/Functions.h>
// 包含头文件：ATen 的本地函数
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，则包含以下头文件
#else
// 包含头文件：ATen 的转换操作
#include <ATen/ops/_to_copy.h>
// 包含头文件：ATen 的转换到本地
#include <ATen/ops/to_native.h>
// 包含头文件：ATen 的提升操作
#include <ATen/ops/lift.h>
// 包含头文件：ATen 的提升到新值
#include <ATen/ops/lift_fresh.h>
// 包含头文件：ATen 的提升到新值并复制
#include <ATen/ops/lift_fresh_copy.h>
// 包含头文件：ATen 的调整大小
#include <ATen/ops/resize.h>
// 包含头文件：ATen 的按步幅操作
#include <ATen/ops/as_strided.h>
// 包含头文件：ATen 的按步幅复制
#include <ATen/ops/as_strided_copy.h>
// 包含头文件：ATen 的空步幅本地化
#include <ATen/ops/empty_strided_native.h>
// 包含头文件：ATen 的不安全视图
#include <ATen/ops/_unsafe_view.h>

// 包含头文件：C++ 标准库
#include <utility>
#endif

// 命名空间：未命名的匿名命名空间
namespace {
  // 函数：functionalizeFallback，处理自定义运算符的后备功能
  void functionalizeFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatchKeySet, torch::jit::Stack* stack) {
    // 获取运算符的架构信息
    const auto& schema = op.schema();
    // 检查是否有任何别名信息，以确保输出不带别名信息
    TORCH_CHECK(
      !schema.hasAnyAliasInfo(),
      "Found a custom (non-ATen) operator whose output has alias annotations: ",
      op.schema(),
      ". We only support functionalizing operators whose outputs do not have alias ",
      "annotations (e.g. 'Tensor(a)' is a Tensor with an alias annotation whereas ",
      "'Tensor' is a Tensor without. The '(a)' is the alias annotation). "
      "The alias annotation specifies that the output ",
      "Tensor shares storage with an input that has the same annotation. ",
      "Please check if ",
      "(1) the output needs to be an output (if not, don't return it), ",
      "(2) if the output doesn't share storage with any inputs, then ",
      "delete the alias annotation. ",
      "(3) if the output indeed shares storage with an input, then add a ",
      ".clone() before returning it to prevent storage sharing and then "
      "delete the alias annotation. ",
      "Otherwise, please file an issue on GitHub.");
    // 计算参数数量并获取参数
    const auto num_arguments = schema.arguments().size();
    const auto arguments_begin = stack->size() - num_arguments;
    auto arguments = torch::jit::last(stack, num_arguments);

    // 是否有任何功能性输入
    auto any_functional_inputs = false;
    // 是否有任何张量输入
    auto any_tensor_inputs = false;
    // 遍历输入参数列表
    for (uint64_t idx = 0; idx < num_arguments; ++idx) {
      // 获取当前参数的引用
      const auto& ivalue = arguments[idx];
      // 检查参数是否为张量
      if (ivalue.isTensor()) {
        // 如果参数是张量，设置标志表示有张量输入
        any_tensor_inputs = true;
        // 获取张量对象
        const auto& t = ivalue.toTensor();
        // 检查张量是否已定义且为功能张量
        if (t.defined() && at::functionalization::impl::isFunctionalTensor(t)) {
          // 如果是功能张量，设置标志表示有功能张量输入
          any_functional_inputs = true;
          // 同步功能张量
          at::functionalization::impl::sync(t);
          // 转换为功能张量后的新张量值
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(t));
          // 替换堆栈中的参数
          (*stack)[arguments_begin + idx] = t_new;
        }
      } else if (ivalue.isTensorList()) {
        // 如果参数是张量列表，设置标志表示有张量输入
        any_tensor_inputs = true;
        // 获取张量列表
        auto tensors = ivalue.toTensorList();
        // 检查张量列表是否为功能张量列表
        if (at::functionalization::impl::isFunctionalTensor(tensors)) {
          // 如果是功能张量列表，设置标志表示有功能张量输入
          any_functional_inputs = true;
          // 同步功能张量列表
          at::functionalization::impl::sync(tensors);
          // 转换为功能张量后的新张量值
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(tensors));
          // 替换堆栈中的参数
          (*stack)[arguments_begin + idx] = t_new;
        }
      } else if (ivalue.isOptionalTensorList()) {
        // 如果参数是可选张量列表，设置标志表示有张量输入
        any_tensor_inputs = true;
        // 获取可选张量列表
        auto opt_tensors = ivalue.toOptionalTensorList();
        // 检查可选张量列表是否为功能张量列表
        if (at::functionalization::impl::isFunctionalTensor(opt_tensors)) {
          // 如果是功能张量列表，设置标志表示有功能张量输入
          any_functional_inputs = true;
          // 同步功能张量列表
          at::functionalization::impl::sync(opt_tensors);
          // 转换为功能张量后的新张量值
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(opt_tensors));
          // 替换堆栈中的参数
          (*stack)[arguments_begin + idx] = t_new;
        }
      }
    }
    // 确定是否需要包装输出结果，条件是没有张量输入或有功能张量输入
    auto should_wrap_outputs = !any_tensor_inputs || any_functional_inputs;
    {
      // 创建自动分发跳过功能化的保护
      at::AutoDispatchSkipFunctionalize guard;
      // 调用操作的封装方法
      op.callBoxed(stack);
    }
    // 获取返回值数量
    const auto num_returns = schema.returns().size();
    // 计算返回值在堆栈中的起始位置
    const auto returns_begin = stack->size() - num_returns;
    // 获取返回值
    auto returns = torch::jit::last(stack, num_returns);

    // 遍历返回值列表
    for (const auto idx : c10::irange(num_returns)) {
      // 获取当前返回值的引用
      const auto& ivalue = returns[idx];
      // 如果返回值是张量且需要包装输出
      if (ivalue.isTensor() && should_wrap_outputs) {
        // 获取张量对象
        const auto& t = ivalue.toTensor();
        // 如果张量未定义，则跳过
        if (!t.defined()) continue;
        // 将张量转换为功能张量后的新张量值
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(t));
        // 替换堆栈中的返回值
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isTensorList() && should_wrap_outputs) {
        // 获取张量列表
        auto tensors = ivalue.toTensorList();
        // 将张量列表转换为功能张量后的新张量值
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(tensors));
        // 替换堆栈中的返回值
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isOptionalTensorList() && should_wrap_outputs) {
        // 获取可选张量列表
        auto opt_tensors = ivalue.toOptionalTensorList();
        // 将可选张量列表转换为功能张量后的新张量值
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(opt_tensors));
        // 替换堆栈中的返回值
        (*stack)[returns_begin + idx] = t_new;
      }
    }
  }
// resize_() 是一个特殊函数，因为：
// - 当我们将大小调整为较大时，它作为一个变异操作
// - 当我们将大小调整为较小时，它作为一个视图
// 详见 Note [resize_ in Functionalization] 获取更多细节
static const at::Tensor & resize__functionalization(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
  // 首先解包张量参数
  at::Tensor self_;
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    // 如果是功能张量，则同步数据
    at::functionalization::impl::sync(self);
    // 从功能张量中获取基础张量
    self_ = at::functionalization::impl::from_functional_tensor(self);
  } else {
    self_ = self;
  }
  // 情况1：参数不是功能张量，因此我们不进行任何操作并重新分派
  if (!at::functionalization::impl::isFunctionalTensor(self)) {
     at::AutoDispatchSkipFunctionalize guard;
     // 调用基础张量的 resize_() 方法
     self_.resize_(size, memory_format);
     return self;
  }

  // 情况2：实际上对 resize_() 进行功能化处理
  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    // 调用功能化版本的 resize()
    tmp_output = at::resize(self_, size, memory_format);
  }

  auto itemsize = self.dtype().itemsize();
  auto storage_offset = self.storage_offset();
  // 计算新大小对应的存储空间字节数
  auto new_size_bytes = at::detail::computeStorageNbytesContiguous(size, itemsize, storage_offset);
  // 判断是否需要重新分配存储空间
  auto needs_resize_storage = new_size_bytes > self.storage().nbytes();

  if (needs_resize_storage) {
    // 如果 resize_() 实际上增加了存储空间的大小，我们需要通知 FunctionalTensorWrapper。
    // 详见 Note[resize_() in functionalization pass]
    auto func_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
    func_impl->maybe_replace_storage(tmp_output);
    // 在这一点上，我们可以确保 "self" 不是视图（并且没有未完成的视图）
    // 因此不需要将 resize 的输出视为视图张量
    return self;
  }

  // 否则，我们知道我们正在调整为较小的大小。
  // resize_() 实际上是一个视图操作。
  // 调整大小的输出相当于从较大张量中获取一个切片。
  // 我们必须使用 as_strided 调用来模拟这种 "切片"。
  auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
    [reapply_views = reapply_views, size = size.vec()](const at::Tensor & base, int64_t mutated_view_idx) -> at::Tensor {
      if (reapply_views) {
        return base.as_strided(size, c10::contiguous_strides(size));
      } else {
        return at::as_strided_copy(base, size, c10::contiguous_strides(size));
      }
    },
    [size = size.vec()](const at::Tensor & base, const at::Tensor & mutated_view, int64_t mutated_view_idx) -> at::Tensor {
      return base.as_strided_scatter(mutated_view, size, c10::contiguous_strides(size));
    },
    /*has_symbolic_inputs=*/false
  );
  // 修改视图元数据
  at::functionalization::impl::mutate_view_meta(self, view_meta);
  return self;
}
// 使用 lift_functionalize 函数将输入张量 self 提升为功能化张量
static at::Tensor lift_functionalize(const at::Tensor & self) {
  // 断言输入张量 self 不是功能化张量
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(self));
  // 在当前作用域内禁止自动功能化
  at::AutoDispatchSkipFunctionalize guard;
  // 对输入张量 self 进行提升操作
  auto out = at::lift(self);
  // 将提升后的张量 out 转换为功能化张量并返回
  return at::functionalization::impl::to_functional_tensor(out);
}

// 使用 lift_fresh_functionalize 函数将输入张量 self 进行新提升和功能化
static at::Tensor lift_fresh_functionalize(const at::Tensor & self) {
  // 查看文档 [Exporting and compiling a graph with lift_fresh_copy]
  // 如果输入张量 self 已经是功能化张量，则直接返回自身
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    return self.view_as(self);
  }

  // 在当前作用域内禁止自动功能化
  at::AutoDispatchSkipFunctionalize guard;
  // 对输入张量 self 进行新提升操作
  auto out = at::lift_fresh(self);
  // 将新提升后的张量 out 转换为功能化张量并返回
  return at::functionalization::impl::to_functional_tensor(out);
}

// 使用 lift_fresh_functionalize_copy 函数将输入张量 self 进行新提升、功能化和复制
static at::Tensor lift_fresh_functionalize_copy(const at::Tensor & self) {
  // 查看文档 [Exporting and compiling a graph with lift_fresh_copy]
  // 如果输入张量 self 已经是功能化张量，则根据注释 [Composite Functionalization under PreDispatch mode] 进行操作
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clone", "").typed<at::Tensor(const at::Tensor &, std::optional<at::MemoryFormat>)>();
    // 使用功能化调度键重新调度操作 op，并返回结果
    return op.redispatch(c10::DispatchKeySet({c10::DispatchKey::Functionalize}), self, c10::nullopt);
  }

  // 在当前作用域内禁止自动功能化
  at::AutoDispatchSkipFunctionalize guard;
  // 对输入张量 self 进行新提升和复制操作
  auto out = at::lift_fresh_copy(self);
  // 将新提升和复制后的张量 out 转换为功能化张量并返回
  return at::functionalization::impl::to_functional_tensor(out);
}

// 检查设备是否已选择进入功能化
static bool device_opted_into_functionalization(c10::Device self_device, std::optional<c10::Device> tgt_device) {
  // 如果目标设备为空，则输出张量应与输入设备在同一设备上
  auto real_tgt_device = tgt_device.has_value() ? tgt_device.value() : self_device;
  // 返回设备类型是否为 XLA 或 Lazy 的布尔值
  return real_tgt_device.type() == c10::DeviceType::XLA || real_tgt_device.type() == c10::DeviceType::Lazy;
}

// 私有函数，用于处理功能化操作时的类型和布局转换
// 由于 to.dtype/to.dtype_layout 重载调用此函数，因此我们跳过上述操作。
// 不过，我们可能应该考虑移除这个函数。
static at::Tensor _to_copy_functionalize(
        const at::Tensor & self,
        std::optional<at::ScalarType> dtype,
        std::optional<at::Layout> layout,
        std::optional<at::Device> device,
        std::optional<bool> pin_memory,
        bool non_blocking,
        std::optional<at::MemoryFormat> memory_format) {
  at::Tensor self_;
  // 如果输入张量 self 是功能化张量，则同步任何待处理的更新
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    at::functionalization::impl::sync(self);
    // 将未包装的张量传递给后端处理
    // 此处没有完整的代码，可能还有后续操作
    // 未来需要补充
  // 将 self 转换为功能化张量 self_
  self_ = at::functionalization::impl::from_functional_tensor(self);
} else {
  // 如果不需要转换为功能化张量，则保持 self 不变
  self_ = self;
}

// 设置自动分发跳过功能化
at::AutoDispatchSkipFunctionalize guard;

// 调用 _to_copy 函数进行张量复制操作
auto out = at::_to_copy(self_, dtype, layout, device, pin_memory, non_blocking, memory_format);

// 特殊情况：如果 TLS 中未设置 Functionalize 键，则假定在延迟执行的后端（LTC）上运行
// 在这种情况下，如果复制到不支持功能化的设备，则功能化过程应该“结束”。
// 我们需要同步输入张量的任何更新，但不应包装输出。
if (!c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {
  // 如果目标设备不支持功能化，并且当前设备也不支持功能化，则直接返回复制结果 out
  if (!device_opted_into_functionalization(self.device(), device)) {
    return out;
  }
}

// 对输出张量进行功能化处理并返回
return at::functionalization::impl::to_functional_tensor(out);
// 为什么在这里特别处理 _unsafe_view？
// 基本上是为了满足 autograd 的调试断言。
// 情况如下：
// - _unsafe_view 的 autograd 内核具有调试断言，用于确认输入和输出别名存储。
// - _unsafe_view 在 native_functions.yaml 中的模式不包含别名注释，因此它表明不进行别名处理。
// - functionalization 将会将 _unsafe_view 视为不进行别名处理的操作。
//   具体来说，autograd 将重新调度到 functionalization 的包装后备内核，
//   创建一个新的 FunctionalTensorWrapper 输出，该输出与输入不别名存储，从而触发断言。
// 这里编写的内核只是手动重新创建了别名关系。

// 另一种处理方式是在 native_functions.yaml 中修复 unsafe_view 的别名注释，
// 但我认为这可能会导致性能下降。
// _unsafe_view 的想法是，您保证输入是临时的，并且实际上不必担心在输入和输出之间传播突变。

// 函数化 _unsafe_view
static at::Tensor _unsafe_view_functionalize(const at::Tensor & self, at::SymIntArrayRef size) {
  // 如果 self 不是 FunctionalTensor，则直接调用 _unsafe_view_symint
  if (!at::functionalization::impl::isFunctionalTensor(self)) {
    at::AutoDispatchSkipFunctionalize guard;
    return at::_unsafe_view_symint(self, size);
  }

  // 将 self 转换为 FunctionalTensor
  auto self_ = at::functionalization::impl::from_functional_tensor(self);
  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    // 对转换后的 FunctionalTensor 调用 _unsafe_view_symint
    tmp_output = at::_unsafe_view_symint(self_, size);
  }

  // 检查 size 中是否有符号输入
  bool has_symbolic_inputs = std::any_of(size.begin(), size.end(), [=](auto& s) { return s.is_symbolic(); });

  // 创建 ViewMeta 对象，用于描述视图的元信息
  at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
    // 用于从基本张量创建视图的函数
    [size = size.vec()](const at::Tensor & base, int64_t mutated_view_idx) -> at::Tensor {
      return at::_unsafe_view_symint(base, size);
    },
    // 用于从变异视图和基本张量创建视图的函数
    [size = size.vec()](const at::Tensor & base, const at::Tensor & mutated_view, int64_t mutated_view_idx) -> at::Tensor {
      return at::_unsafe_view_symint(mutated_view, base.sym_sizes());
    },
    /*has_symbolic_inputs=*/has_symbolic_inputs
  );

  // 使用 ViewMeta 创建具有视图元信息的 FunctionalTensor
  auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, self, std::move(view_meta));

  // 查看  Note [Propagating strides in the functionalization pass]
  // (对于 _unsafe_view，我只是手动执行这里的形状推断规则，而不是调用 unsafe_view 的元函数)
  auto inferred_size = at::infer_size_dv(size, self.sym_numel());
  auto stride = at::detail::computeStride(self.sym_sizes(), self.sym_strides(), inferred_size);

  // 断言计算的 stride 值
  TORCH_INTERNAL_ASSERT(stride.has_value());

  // 设置输出张量的大小和步幅
  out.unsafeGetTensorImpl()->set_sizes_and_strides(inferred_size, stride.value());

  // 返回输出张量
  return out;
}

// 函数化 set__
static at::Tensor& set__functionalize(at::Tensor& self, const at::Tensor& src) {
  // 错误情况检查
  TORCH_CHECK(at::functionalization::impl::isFunctionalTensor(self) || !at::functionalization::impl::isFunctionalTensor(src),
  // 报错情况：尝试使用功能张量更改非功能张量，这是不允许的
  "set__functionalize: Tried to mutate a non-functional tensor with a functional tensor, which is not allowed");

  // 空操作情况
  if (!at::functionalization::impl::isFunctionalTensor(self) && !at::functionalization::impl::isFunctionalTensor(src)) {
    // 使用 AutoDispatchSkipFunctionalize 保护块
    at::AutoDispatchSkipFunctionalize guard;
    // 直接使用 self 的 set_() 方法设置为 src
    return self.set_(src);
  }

  // 检查 src 是否为 FunctionalTensor
  TORCH_CHECK(at::functionalization::impl::isFunctionalTensor(src),
    "set__functionalize: We do not currently support x.set_(y) where y is not a FunctionalTensor. Please file an issue");

  // 内部断言：self 必须是 FunctionalTensor
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  // 内部断言：src 必须是 FunctionalTensor
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(src));

  // 获取 self 和 src 的功能包装器实现
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  auto src_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(src);

  // 查看 resize_() 和 set_() 的顺序问题
  // 见注释 [Ordering of resize_() and set_()]
  TORCH_CHECK(!self_impl->was_inductor_storage_resized(),
    "storage_resize_() followed by set_() in torch.compile is not supported today");

  // 调用 self 的功能包装器实现的 set__() 方法，传入 src 的功能包装器实现
  self_impl->set__impl(src_impl);

  // 返回修改后的 self
  return self;
TORCH_LIBRARY_IMPL(_, Functionalize, m) {
  // 注册名为 Functionalize 的 Torch 库实现
  // 添加一个回退函数，对应 functionalizeFallback 函数的封装
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&functionalizeFallback>());
}

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  // 注册名为 Functionalize 的 Torch 库实现，针对 aten 命名空间
  // 实现 resize_ 函数的功能化版本
  m.impl("resize_", TORCH_FN(resize__functionalization));
  // 实现 lift 函数的功能化版本
  m.impl("lift", TORCH_FN(lift_functionalize));
  // 实现 lift_fresh 函数的功能化版本
  m.impl("lift_fresh", TORCH_FN(lift_fresh_functionalize));
  // 实现 lift_fresh_copy 函数的功能化版本
  m.impl("lift_fresh_copy", TORCH_FN(lift_fresh_functionalize_copy));
  // 实现 _to_copy 函数的功能化版本
  m.impl("_to_copy", TORCH_FN(_to_copy_functionalize));
  // 实现 _unsafe_view 函数的功能化版本
  m.impl("_unsafe_view", TORCH_FN(_unsafe_view_functionalize));
  // 针对 set_() 函数的重载，使用 source_Tensor 版本
  // 因为 dynamo 图会破坏 torch.compile，所以这里不使用
  m.impl("set_.source_Tensor", TORCH_FN(set__functionalize));
}
```