# `.\pytorch\torch\csrc\functorch\init.cpp`

```py
// 包含必要的头文件
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/WrapDimUtils.h>
#include <torch/csrc/utils/python_raii.h>
#include <torch/python.h>

// 包含functorch的具体实现文件
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/Interpreter.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/TensorWrapper.h>
#include <c10/core/AutogradState.h>

// 包含标准库的输出流
#include <iostream>

// 定义functorch的Python绑定在torch::functorch::impl命名空间下
namespace torch::functorch::impl {

// 使用functorch命名空间下的成员
using namespace at::functorch;

// 检查给定的Tensor是否具有指定的level
static bool has_level(const Tensor& self, int64_t level) {
  const auto* batched = maybeGetBatchedImpl(self);
  if (!batched) {
    return false;
  }
  return batched->level() >= level;
}

// 给Tensor添加批次维度，并指定level
Tensor _add_batch_dim(const Tensor& self, int64_t batch_dim, int64_t level) {
  return addBatchDim(self, batch_dim, level);
}

// 包装functional tensor，并设置level
Tensor _wrap_functional_tensor(const Tensor& self, int64_t level) {
  auto t = at::functionalization::impl::to_functional_tensor(self);
  at::functionalization::impl::unsafeGetFunctionalWrapper(t)->set_level(level);
  return t;
}

// 断言包装的functional tensor是有效的
void _assert_wrapped_functional(
    const Tensor& unwrapped,
    const Tensor& wrapped) {
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(wrapped));
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(unwrapped));
  auto wrapped_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(wrapped);
  auto& wrapped_inner = wrapped_impl->value();
  TORCH_INTERNAL_ASSERT(
      unwrapped.unsafeGetTensorImpl() == wrapped_inner.unsafeGetTensorImpl())
}

// 将功能化的输入变异传播到包装的functional tensor
void _propagate_functional_input_mutation(
    const Tensor& unwrapped,
    const Tensor& wrapped) {
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(wrapped));
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(unwrapped));
  auto wrapped_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(wrapped);
  // 确保输入是最新的，通过提交任何待处理的更新到别名
  wrapped_impl->sync_();
  auto& wrapped_inner = wrapped_impl->value();
  // 通常应检查两个张量是否是别名的，但是除非我们给BatchedTensorImpl添加了存储的概念，否则无法做到这一点
  if (unwrapped.unsafeGetTensorImpl() == wrapped_inner.unsafeGetTensorImpl()) {
  } else {
    if (unwrapped.sym_nbytes() != wrapped_inner.sym_nbytes()) {
      // 函数可能会调整大小为零的输入，我们需要反映这一点
      unwrapped.resize__symint(wrapped_inner.sym_sizes());
    }
    // 如果输入张量的元数据已被修改，则使用 as_strided_() 来传播元数据的变化。
    if (unwrapped.sym_sizes() != wrapped_inner.sym_sizes()) {
      // 调用 as_strided__symint 方法，传入内部张量的大小和步长信息，更新 unwrapped 张量的视图。
      unwrapped.as_strided__symint(
          wrapped_inner.sym_sizes(), wrapped_inner.sym_strides());
    }
    // 将 wrapped_inner 的数据复制到 unwrapped 张量中。
    unwrapped.copy_(wrapped_inner);
  }
// 静态函数，从批处理张量中移除指定级别的批处理维度
static std::pair<Tensor, int64_t> remove_existing_batch_dim(
    const BatchedTensorImpl* batched,
    int64_t level) {
  // 断言批处理张量的级别与给定级别相等
  TORCH_INTERNAL_ASSERT(batched->level() == level);
  // 返回批处理张量的值和批处理维度
  return std::make_pair(batched->value(), batched->bdim());
}

// 简易版本的 np.moveaxis。将张量中的维度从 `dst` 移动到 `src`，保持其它维度的顺序不变
// 我们应该考虑将 np.moveaxis（更加通用）添加到 PyTorch 中。(#36048)
// 当我们添加后，用它替换以下内容。
static Tensor _movedim(const Tensor& self, int64_t src, int64_t dst) {
  auto logical_dim = self.dim();
  // 使用逻辑维度数对源和目标维度进行包装
  src = at::maybe_wrap_dim(src, logical_dim);
  dst = at::maybe_wrap_dim(dst, logical_dim);
  // 如果源和目标维度相同，则直接返回原张量
  if (src == dst) {
    return self;
  }
  // 创建一个排列向量，用于重新排列维度顺序
  VmapDimVector permutation;
  permutation.reserve(logical_dim);
  for (int64_t dim = 0; dim < logical_dim; dim++) {
    if (dim == src) {
      continue;
    }
    permutation.push_back(dim);
  }
  // 在目标位置插入源维度
  permutation.insert(permutation.begin() + dst, src);
  // 返回重新排列后的张量
  return self.permute(permutation);
}

// 从 `self` 中移除级别为 `level` 的批处理维度。如果这导致最后一个批处理维度从 BatchedTensor 中移除，则返回常规张量。
//
// 如果要移除的批处理维度级别在 `self` 中不存在，则添加该批处理维度。
// 这种情况可能发生在 `self` 与 vmap 级别内的张量没有交互的情况下，例如，
//     self = torch.randn(3)
//     y = torch.randn(5)
//     out = vmap(lambda x: vmap(lambda y: x)(y))(self)
//     assert out.shape == (3, 5)
// 在内部 vmap 中，`x` 是一个只有一个批处理维度，对应于外部 vmap 级别，并且它没有任何与内部 vmap 级别对应的维度，因此我们需要为用户创建一个。
//
// `out_dim` 控制在输出张量中放置批处理维度的位置。
Tensor _remove_batch_dim(
    const Tensor& self,
    int64_t level,
    int64_t batch_size,
    int64_t out_dim) {
  // 检查如果 `self` 是 NestedTensor，只能在 dim=0 上进行 vmap
  TORCH_CHECK(
      out_dim == 0 || !self.key_set().has(DispatchKey::BatchedNestedTensor),
      "Nested tensors can only be vmapped over dim=0, but got dim=",
      out_dim);
  // 如果 `self` 中没有给定级别的批处理维度，则扩展张量以添加该批处理维度
  if (!has_level(self, level)) {
    auto self_sizes = self.sizes();
    VmapDimVector expanded_sizes(self_sizes.begin(), self_sizes.end());
    expanded_sizes.insert(expanded_sizes.begin() + out_dim, batch_size);
    auto result = self.expand(expanded_sizes);
    return result;
  }

  // 如果 `self` 中存在批处理级别，则必须是批处理张量
  const auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched != nullptr);

  // 从批处理张量中移除指定级别的批处理维度，并获取新的逻辑维度
  auto [self_without_bdim, newly_exposed_logical_dim] =
      remove_existing_batch_dim(batched, level);
  // 使用 `_movedim` 函数将批处理维度移动到指定的输出维度
  auto result = _movedim(self_without_bdim, newly_exposed_logical_dim, out_dim);
  // 返回结果张量
  return result;
}
// 返回被 functionalize() 调用后的张量的内部表示，确保张量被 FunctionalTensorWrapper 包装
Tensor _unwrap_functional_tensor(const Tensor& self, bool add_back_views) {
  // 断言当前张量必须被 FunctionalTensor 包装，用于 functionalize() 调用后的后续处理
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  // 获取 FunctionalTensorWrapper 对象，用于处理 functionalize() 调用生成的张量
  auto functional = at::functionalization::impl::unsafeGetFunctionalWrapper(self);

  // 当重新生成（可能被修改的）输入张量时，functionalization 过程通过一系列 view_copy() 操作进行
  // Functorch 试图将其转换回 view 操作。通过提交任何挂起的更新到别名，确保输入是最新的。
  at::functionalization::impl::FunctionalizationReapplyViewsGuard guard(
      add_back_views);
  // 应用任何更新到 functional 对象中的数据
  bool any_updates = functional->apply_updates();
  // 如果有更新，则从基本数据重新生成 functional 对象
  if (any_updates) {
    functional->regenerate_from_base();
  }
  // 返回 functional 对象中的值
  return functional->value();
}

// 为了梯度计算包装张量，用于梯度处理和级别跟踪
Tensor _wrap_for_grad(const Tensor& self, int64_t level) {
  // 使用 makeTensorWrapper() 将张量包装为带有给定级别的 TensorWrapper
  return makeTensorWrapper(self, level);
}

// 解除梯度计算的张量包装，确保张量没有包装或与给定级别匹配
Tensor _unwrap_for_grad(const Tensor& self, int64_t level) {
  // 尝试获取张量的 TensorWrapper
  auto* result = maybeGetTensorWrapper(self);
  // 如果没有包装，则返回原始张量
  if (!result) {
    return self;
  }
  // 断言 TensorWrapper 的级别是有效的
  TORCH_INTERNAL_ASSERT(result->level().has_value());
  // 如果 TensorWrapper 的级别与给定级别相匹配，则返回其包装的值
  if (result->level() == level) {
    return result->value();
  }
  // 否则返回原始张量
  return self;
}

// 获取张量的深度级别，如果没有包装返回0，如果不再活跃返回-1
int64_t dlevel(const Tensor& tensor) {
  // 尝试获取张量的 TensorWrapper
  auto* wrapped = maybeGetTensorWrapper(tensor);
  // 如果没有包装，则返回级别0
  if (!wrapped) {
    return 0;
  }
  // 如果 TensorWrapper 不再活跃，则返回级别-1
  if (!wrapped->is_alive()) {
    return -1;
  }
  // 否则返回 TensorWrapper 的级别值
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return wrapped->level().value();
}

// 打印张量到标准输出
bool dump_tensor(const Tensor& self) {
  // 调用 dumpTensorCout() 打印张量到标准输出
  dumpTensorCout(self);
  return true;
}

// 将字符串类型的随机性参数转换为 RandomnessType 枚举类型
RandomnessType get_randomness_enum(const std::string& randomness) {
  // 根据字符串内容返回对应的 RandomnessType 枚举值
  if (randomness == "error") {
    return RandomnessType::Error;
  } else if (randomness == "same") {
    return RandomnessType::Same;
  } else if (randomness == "different") {
    return RandomnessType::Different;
  } else {
    // 如果参数不在预期范围内，则抛出错误
    TORCH_CHECK(
        false, "randomness argument must be error, same, or different.");
  }
}

// 增加梯度计算的嵌套层级，并返回新的动态层级ID
int64_t _grad_increment_nesting() {
  // 查看当前是否启用梯度模式，并记录前一个状态
  bool prev_grad_mode = c10::GradMode::is_enabled();
  // 初始化并推送一个新的动态层级，表示为梯度变换类型的层级
  return initAndPushDynamicLayer(
      TransformType::Grad, c10::nullopt, c10::nullopt, prev_grad_mode);
}

// 减少梯度计算的嵌套层级，并返回弹出的动态层级ID
int64_t _grad_decrement_nesting() {
  // 弹出当前动态层级并删除其元数据
  auto layer = popDynamicLayerAndDeleteMetadata();
  // 断言弹出的层级类型为梯度变换类型
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Grad);
  // 返回弹出层级的ID
  return layer.layerId();
}
// 增加动态层次嵌套计数，并返回新的层次 ID
int64_t _jvp_increment_nesting() {
  // 查看先前的前向梯度模式
  bool prev_fwd_grad_mode =
      c10::AutogradState::get_tls_state().get_fw_grad_mode();
  // 初始化并推入动态层次，类型为 JVP，使用先前的前向梯度模式
  return initAndPushDynamicLayer(
      TransformType::Jvp,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      prev_fwd_grad_mode);
}

// 减少动态层次嵌套计数，并返回弹出的层次 ID
int64_t _jvp_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  // 内部断言确保弹出的层次是 JVP 类型
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Jvp);
  // 返回弹出层次的 ID
  return layer.layerId();
}

// 增加 VMap 动态层次嵌套计数，并返回新的层次 ID
int64_t _vmap_increment_nesting(
    c10::SymInt batch_size,
    const std::string& randomness) {
  // 初始化并推入动态层次，类型为 VMap，传入批处理大小和随机性信息
  return initAndPushDynamicLayer(
      TransformType::Vmap,
      std::move(batch_size),
      get_randomness_enum(randomness));
}

// 减少 VMap 动态层次嵌套计数，并返回弹出的层次 ID
int64_t _vmap_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  // 内部断言确保弹出的层次是 VMap 类型
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Vmap);
  // 返回弹出层次的 ID
  return layer.layerId();
}

// 增加功能化动态层次嵌套计数，并返回新的层次 ID
int64_t _func_increment_nesting(bool reapply_views) {
  // 初始化并推入动态层次，类型为 Functionalize，根据 reapply_views 决定是否添加回视图
  return initAndPushDynamicLayer(
      TransformType::Functionalize,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      /*functionalize_add_back_views=*/reapply_views);
}

// 减少功能化动态层次嵌套计数，并返回弹出的层次 ID
int64_t _func_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  // 内部断言确保弹出的层次是 Functionalize 类型
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Functionalize);
  // 返回弹出层次的 ID
  return layer.layerId();
}

// 检查张量是否为批处理张量，返回布尔值
static bool is_batchedtensor(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  return batched != nullptr;
}

// 检查张量是否为旧版批处理张量，返回布尔值
static bool is_legacy_batchedtensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::Batched);
}

// 检查张量是否为跟踪梯度的张量，返回布尔值
static bool is_gradtrackingtensor(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  return wrapped != nullptr;
}

// 检查张量是否为功能化张量，返回布尔值
static bool is_functionaltensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::Functionalize);
}

// 获取张量的非包装值
static Tensor get_unwrapped(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    return batched->value();
  }
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    return wrapped->value();
  }
  // 若为功能化张量，则获取其包装的值
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    auto* functional =
        at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
    return functional->value();
  }
  // 若没有任何包装器存在，则抛出错误
  TORCH_CHECK(false, "No wrappers present!");
}

// 可能获取张量的层次级别
static int64_t maybe_get_level(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    return batched->level();
  }
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    if (wrapped->level()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      return *wrapped->level();
    }
    // TODO: 这是一个奇怪的特殊情况...
    return -2;
  }
  // 若为功能化张量，则返回其层次级别
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    auto* functional =
        at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
    return functional->level();
  }
}
    return functional->level();
  }
  // 如果没有符合条件的情况，返回-1
  return -1;
}

// 获取可能的批处理维度
static int64_t maybe_get_bdim(const Tensor& tensor) {
  // 获取批处理实现
  auto* batched = maybeGetBatchedImpl(tensor);
  // 如果存在批处理实现，返回其批处理维度
  if (batched) {
    return batched->bdim();
  }
  // 如果不存在批处理实现，返回-1
  return -1;
}

// 获取当前动态层级
static int64_t currentLevel() {
  // 获取当前可能的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 断言一定存在当前层
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  // 获取当前层级的ID
  int64_t current_level = maybe_layer->layerId();
  return current_level;
}

// 获取当前可能的动态层级，返回std::optional类型
static std::optional<int64_t> maybe_current_level() {
  // 获取当前可能的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 如果存在当前层，返回当前层级的ID
  if (maybe_layer.has_value()) {
    int64_t current_level = maybe_layer->layerId();
    return current_level;
  }
  // 如果不存在当前层，返回空optional
  return nullopt;
}

// 设置是否排除Vmap的分发键
static void tls_set_vmap_excluded(bool excluded) {
  c10::impl::tls_set_dispatch_key_excluded(
      c10::DispatchKey::FuncTorchBatched, excluded);
}

// 设置是否包含动态层键的前后端
static void _set_dynamic_layer_keys_included(bool value) {
  return setDynamicLayerFrontBackKeysIncluded(value);
}

// 打印动态层栈信息
static void dump_dls() {
  std::cout << getDynamicLayerStack() << std::endl;
}

// 打印本地TLS的包含和排除信息
static void dump_local_tls() {
  auto tls = c10::impl::tls_local_dispatch_key_set();
  std::cout << "[Local Include] " << tls.included_ << std::endl;
  std::cout << "[Local Exclude] " << tls.excluded_ << std::endl;
}

namespace {

// 弹出动态层栈，直到深度为给定值
void popDynamicLayerStackToDepth(size_t depth) {
  // 当动态层栈的大小大于指定深度时执行循环
  while (at::functorch::getDynamicLayerStack().size() > depth) {
    // 弹出栈顶的动态层
    const auto top = popDynamicLayer();
    // 根据弹出的动态层类型进行不同的操作
    switch (top.key()) {
      case at::functorch::TransformType::Vmap:
        _vmap_decrement_nesting();
        break;
      case at::functorch::TransformType::Grad:
        _grad_decrement_nesting();
        break;
      case at::functorch::TransformType::Jvp:
        _jvp_decrement_nesting();
        break;
      case at::functorch::TransformType::Functionalize:
        _func_decrement_nesting();
        break;
      case at::functorch::TransformType::Torch:
        popDynamicLayerAndDeleteMetadata();
        break;
    }
  }
}

} // anonymous namespace

// 解开批处理封装
static std::tuple<Tensor, std::optional<int64_t>> unwrapBatched(
    const Tensor& tensor,
    int64_t level) {
  // 获取可能的批处理实现
  auto* batched = maybeGetBatchedImpl(tensor);
  // 如果不存在批处理实现，返回原始张量和空optional
  if (!batched) {
    return std::make_tuple(tensor, nullopt);
  }
  // 如果批处理实现的层级等于指定层级，返回解开后的张量和批处理维度
  if (batched->level() == level) {
    return std::make_tuple(batched->value(), batched->bdim());
  }
  // 否则返回原始张量和空optional
  return std::make_tuple(tensor, nullopt);
}

    return maybe_get_level(tensor) != -1;
  });
  // 定义Python接口函数，返回解释器栈的可选向量
  m.def(
      "get_interpreter_stack", []() -> std::optional<std::vector<Interpreter>> {
        // 获取动态层栈
        const auto& stack = getDynamicLayerStack();
        // 如果栈为空，返回空optional
        if (stack.empty()) {
          return c10::nullopt;
        }
        // 否则构建解释器栈的向量并返回
        std::vector<Interpreter> result;
        result.reserve(stack.size());
        for (auto i : stack) {
          result.push_back(i.interpreter());
        }
        return result;
      });
  // 定义Python接口函数，返回解释器栈顶的可选解释器
  m.def("peek_interpreter_stack", []() -> std::optional<Interpreter> {
    // 获取动态层栈
    const auto& stack = getDynamicLayerStack();
    // 如果栈为空，返回空optional
    if (stack.empty()) {
      return c10::nullopt;
    }
    // 调用 stack 的最后一个元素的 interpreter 方法，并将结果存储到 result 中
    auto result = stack.back().interpreter();
    // 返回 result
    return result;
  });

  // 定义一个名为 get_dynamic_layer_stack_depth 的函数，返回当前动态层栈的深度
  m.def("get_dynamic_layer_stack_depth", []() -> size_t {
    return getDynamicLayerStack().size();
  });

  // 定义一个名为 pop_dynamic_layer_stack_and_undo_to_depth 的函数，将动态层栈 pop 直到指定深度
  m.def(
      "pop_dynamic_layer_stack_and_undo_to_depth",
      &popDynamicLayerStackToDepth);

  // 定义一个名为 pop_dynamic_layer_stack 的函数，用于弹出动态层栈的顶部元素
  m.def("pop_dynamic_layer_stack", &popDynamicLayer);

  // 定义一个名为 push_dynamic_layer_stack 的函数，接受一个 DynamicLayer 对象，并返回一个 int64_t
  m.def("push_dynamic_layer_stack", [](DynamicLayer layer) -> int64_t {
    return pushDynamicLayer(std::move(layer));
  });

  // NOLINTNEXTLINE(bugprone-unused-raii)
  // 在 Python 绑定中注册一个名为 DynamicLayer 的类，忽略一些 RAII 检查警告
  py::class_<DynamicLayer>(m, "DynamicLayer");

  // 在 Python 绑定中注册一个名为 TransformType 的枚举类型
  py::enum_<TransformType>(m, "TransformType")
      .value("Torch", TransformType::Torch)
      .value("Grad", TransformType::Grad)
      .value("Jvp", TransformType::Jvp)
      .value("Functionalize", TransformType::Functionalize)
      .value("Vmap", TransformType::Vmap);

  // 在 Python 绑定中注册一个名为 RandomnessType 的枚举类型
  py::enum_<RandomnessType>(m, "RandomnessType")
      .value("Error", RandomnessType::Error)
      .value("Same", RandomnessType::Same)
      .value("Different", RandomnessType::Different);

  // 在 Python 绑定中注册一个名为 CInterpreter 的类，并暴露其方法 key 和 level
  py::class_<Interpreter>(m, "CInterpreter")
      .def("key", &Interpreter::key)
      .def("level", &Interpreter::level);

  // 在 Python 绑定中注册一个名为 CGradInterpreterPtr 的类，并暴露其方法 key、level、lift 和 prevGradMode
  py::class_<GradInterpreterPtr>(m, "CGradInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &GradInterpreterPtr::key)
      .def("level", &GradInterpreterPtr::level)
      .def("lift", &GradInterpreterPtr::lift)
      .def("prevGradMode", &GradInterpreterPtr::prevGradMode);

  // 在 Python 绑定中注册一个名为 CJvpInterpreterPtr 的类，并暴露其方法 key、level、lift 和 prevFwdGradMode
  py::class_<JvpInterpreterPtr>(m, "CJvpInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &JvpInterpreterPtr::key)
      .def("level", &JvpInterpreterPtr::level)
      .def("lift", &JvpInterpreterPtr::lift)
      .def("prevFwdGradMode", &JvpInterpreterPtr::prevFwdGradMode);

  // 在 Python 绑定中注册一个名为 CVmapInterpreterPtr 的类，并暴露其方法 key、level、batchSize 和 randomness
  py::class_<VmapInterpreterPtr>(m, "CVmapInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &VmapInterpreterPtr::key)
      .def("level", &VmapInterpreterPtr::level)
      .def("batchSize", &VmapInterpreterPtr::batchSize)
      .def("randomness", &VmapInterpreterPtr::randomness);

  // 在 Python 绑定中注册一个名为 CFunctionalizeInterpreterPtr 的类，并暴露其方法 key、level 和 functionalizeAddBackViews
  py::class_<FunctionalizeInterpreterPtr>(m, "CFunctionalizeInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &FunctionalizeInterpreterPtr::key)
      .def("level", &FunctionalizeInterpreterPtr::level)
      .def(
          "functionalizeAddBackViews",
          &FunctionalizeInterpreterPtr::functionalizeAddBackViews);
}

} // namespace torch::functorch::impl
```