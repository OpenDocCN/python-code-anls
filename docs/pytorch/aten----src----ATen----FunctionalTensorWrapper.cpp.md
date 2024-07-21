# `.\pytorch\aten\src\ATen\FunctionalTensorWrapper.cpp`

```
// 包含 ATen 库中的头文件，用于实现 FunctionalTensorWrapper 类
#include <ATen/FunctionalTensorWrapper.h>

// 包含 ATen 库中的其他相关头文件
#include <ATen/FunctionalInverses.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>

// 包含 c10 库中的实用工具头文件
#include <c10/util/irange.h>

// 根据编译条件选择性包含 ATen 的函数头文件或者特定操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_propagate_xla_data.h>
#include <ATen/ops/_to_copy.h>
#endif

// 命名空间定义：ATen 库的命名空间
namespace at {

// 设置 FunctionalTensorWrapper 类的构造函数的元数据
void FunctionalTensorWrapper::set_constructor_metadata() {
  // 内部断言：确保 value_ 是已定义的 Tensor
  TORCH_INTERNAL_ASSERT(value_.defined());

  // 设置 level_ 为 -1，暂时在 functorch 中进行设置
  level_ = -1;

  // 将通用张量的元数据复制到包装器上
  copy_generic_tensor_metadata(value_.getIntrusivePtr().get(), this);

  // 刷新 numel 属性
  refresh_numel();

  // 刷新 contiguous 属性
  refresh_contiguous();

  // 禁止存储访问抛出异常
  storage_access_should_throw_ = false;

  // 允许张量元数据的变化
  set_allow_tensor_metadata_change(true);

  // 设置 key_set_，包括 Functionalize DispatchKey，并继承自 value_ 的 key_set_
  key_set_ = c10::DispatchKeySet(c10::DispatchKey::Functionalize) | value_.key_set();

  // 移除所有与 functorch 转换相关的 key
  key_set_ = key_set_ - c10::functorch_transforms_ks - c10::python_ks;

  // 设置自定义大小和步长策略为 CustomSizes
  set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);

  // 设置自定义设备标志为 true
  set_custom_device(true);

  // 设置 version_counter_ 为 value_ 的不安全张量实现的 version_counter
  version_counter_ = value_.unsafeGetTensorImpl()->version_counter();
}

// FunctionalTensorWrapper 类的构造函数，接受一个 Tensor 作为参数
FunctionalTensorWrapper::FunctionalTensorWrapper(const Tensor& value)
  : c10::TensorImpl(
      c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(value)),
      c10::DispatchKeySet(DispatchKey::Functionalize) | value.key_set(),
      value.dtype()
    ),
    value_(value)
{
  // 内部断言：确保 value_ 不是 FunctionalTensor
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(value_));

  // 内部断言：确保 value_ 的 key_set_ 不包含 Functionalize DispatchKey
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));

  // 设置构造函数的元数据
  set_constructor_metadata();
}

// 冻结存储器的实现函数
void FunctionalTensorWrapper::freeze_storage() const {
  // 调用 functional_storage_impl() 的 freeze 方法
  functional_storage_impl()->freeze();
}

// Note [Functionalization: Alias Removal]
// Functionalization 过程中，当调用 view() 操作时，如 'b = a.view(...)',
// 我们使用共享的 Alias 对象来保留别名关系。
//
// How do we do that?
//
// Every FunctionalTensorWrapper contains a dummy FunctionalStorageImpl, which subclasses from c10::StorageImpl.
// It doesn't contain any data (similar to MetaTensor storage), but it contains an Alias object that knows about the base tensor.
// When a tensor is created through a view operation, both the new and old tensor point to the same FunctionalStorageImpl.

// As mutations are applied to any of the views, we also queue each mutation up on the Alias object, so we can replay them.
// When the user requests a tensor that's had a view taken, we check if it's up to date.
// If it's not up to date, we first replay all of the queued up mutations onto the alias, and then re-apply the current view
// on top of the newly updated alias.

// Why do we queue up and lazily run mutations on the alias, instead of updating the alias eagerly?
// This behavior was taken from pytorch/xla, which the alias-removal logic was inspired from.
// One benefit of the laziness is that we save work in the cases where a user has multiple views and mutates one of them,
// but never uses the other views later in the program (in which case we'll never update the alias).
// It also has downsides though: repeatedly applying mutations to the same view without syncing
// will silently use up more and more memory as more mutations are queued up.

// Corresponding diagram:
//
// b = a.view(...)
//
//        a                                                    b
//        |                                                    |     If the user asks for b and it’s out of date,
//       \/                                                    \/    We regenerate b by replaying it’s views from the alias.
// . - - - - - - - - - - - - - .                    . - - - - - - - - - - - - - .
// |  FunctionalTensorWrapper  |                    |  FunctionalTensorWrapper  |
// . - - - - - - - - - - - - - .                    . - - - - - - - - - - - - - .
// |     value   |   storage   |                    |    storage    |   Value   |
// . - - - - - - - - - - - - - .                    . - - - - - - - - - - - - - .
//          |                   \                  /                      |
//          |                     \              /                        |
//          |                       . - - - - - - - - - - - - .           |
//          |                       |  FunctionalStorageImpl  |           |
//          |                       . - - - - - - - - - - - - .           |
//          |                       |         Alias           |           |
//          |                       . - - - - - -- - - - - - - .           |
//          |                       /     mutations to a or b             |
//          |                     /       are queued onto Alias           |
//          |                   /                                         |
//         \/                 /                                           \/
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
// |        TensorImpl         |                             |             TensorImpl        |
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
// |   value   |   storage     |                             |    storage    |     Value     |
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
//          |                                                             |
//          |                                                             |
//          |                                                             |
//          |   In this picture the two tensor views their own storages,  |
//          |   have their own storages, but backends like functorch      |
//         \/   are allowed to re-alias underneath the pass               \/
// . - - - - - - - - - - - - - .                             . - - - - - - - - - -```cpp
// - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
// |    underyling_storage     |                             |      underyling_storage       |
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .

// This constructor is only used by view ops.
// - view_value: The output tensor that we need to wrap.
// - base: The "base" of the view that `view_value` was generated from.
// See Note [Functionalization: Alias Removal Part 2] for more details on the mutation replay logic.
FunctionalTensorWrapper::FunctionalTensorWrapper(const Tensor& view_value, const FunctionalTensorWrapper* base, const functionalization::ViewMeta& meta)
  : c10::TensorImpl(
      c10::DispatchKeySet(DispatchKey::Functionalize), // 设置 DispatchKey 为 Functionalize
      view_value.dtype(),         // 使用 view_value 的数据类型
      view_value.device()         // 使用 view_value 的设备类型
    ),
    value_(view_value),           // 初始化 value_ 成员变量为 view_value
    is_multi_output_view_(base->is_multi_output_view_ || meta.is_multi_output),  // 根据 base 和 meta 设置是否为多输出视图
    was_storage_changed_(base->was_storage_changed_),  // 继承 base 的存储是否变化标志
    is_symbolic_(base->is_symbolic_)  // 继承 base 的符号标志
{
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(value_));  // 断言 value_ 不是 FunctionalTensor
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));  // 断言 value_ 的 DispatchKey 不包含 Functionalize
  set_constructor_metadata();  // 设置构造函数的元数据
  // Copy the original tensor's ViewMeta vector and push the current one.
  if (!base->view_metas_.empty()) {
      view_metas_ = base->view_metas_;  // 复制 base 的 view_metas_
  }
  view_metas_.push_back(meta);  // 添加当前 meta 到 view_metas_ 中
  maybe_mark_symbolic(meta);  // 根据 meta 可能标记为符号化
  storage_ = base->storage_; // 使用 base 的 storage 别名化此张量的存储
}


functionalization::FunctionalStorageImpl* FunctionalTensorWrapper::functional_storage_impl() const {
  return static_cast<functionalization::FunctionalStorageImpl*>(storage_.unsafeGetStorageImpl());  // 返回此张量的功能化存储实现
}
// 提交更新到功能性张量包装器的存储实现中
void FunctionalTensorWrapper::commit_update() {
  // 获取功能性存储实现
  auto storage_impl = functional_storage_impl();
  // 将当前值和视图元数据添加到存储实现中
  storage_impl->add_update(value_, view_metas_);
  // 作为优化，此处曾将张量标记为“已更新”，
  // 这样，像下面这样的代码：
  //   x = torch.ones(1'000'000)
  //   x[0].add_(1)
  // 不会导致基础结构的不必要实体化。
  // 尽管这种优化会导致切片暂时具有不正确的步幅/存储偏移，但是DCE应该处理该优化。
  // generation_ = storage_impl->generation();
}

// 检查功能性张量包装器是否是最新的
bool FunctionalTensorWrapper::is_up_to_date() const {
  // 获取功能性存储实现的代数
  auto alias_generation = functional_storage_impl()->generation();
  // 检查当前代数是否与别名代数相等
  return generation_ == alias_generation;
}

// 查看备注[功能化传递 - 原位视图操作]
void FunctionalTensorWrapper::mutate_view_meta(const at::functionalization::ViewMeta& meta) {
  // 将视图元数据添加到视图元数据列表中
  view_metas_.push_back(meta);
  // 手动跟踪张量是否接收了元数据突变！
  has_metadata_mutation_ = true;
  // 如果视图操作使用任何符号输入，则将此张量标记为符号化
  maybe_mark_symbolic(meta);
  // 查看备注[功能化传递 - 原位视图操作]
  // 因此，这些操作很特殊 - 它们既是突变又是视图操作。它们得到特殊的代码生成。
  // 一个例子是 transpose_，例如 `a.transpose_()`
  // 调用 transpose_() 应该确保 a 获得一个别名，并将新的 ViewMeta 追加到 a 当前的 ViewMetas 列表中。
  at::AutoDispatchSkipFunctionalize guard;
  // 使用视图元数据中的前向函数对值进行变换，并更新值
  value_ = meta.forward_fn(value_, meta.out_index);
  // 内部断言：确保值没有 Functionalize 调度键
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
}

// 查看备注[功能化：突变移除]
// 突变移除用于将程序
// a.add_(b)
// 转换为语义上相同但结构稍有不同的程序：
// tmp = a.add(b)
// a.replace_(tmp)
// 其中 replace_() 调用在功能化传递中直接实现，因此对后端透明。
// 这对于不能处理某些类型突变的后端（如 functorch）非常有用。
//
// 为什么我们需要在 FunctionalTensorWrapper 中包装每个张量？考虑以下程序：
//
// Before:
// tensor.add_(batched_tensor)
//
// After:
// tmp = tensor.add(batched_tensor)
// tensor.replace_(tmp)
//
// 在上面的示例中，tmp 是一个批量张量（因为将普通张量添加到批量张量会广播并创建批量张量）。
// 但是我们不能简单地用 tmp 替换底层内存支持的 `tensor` - 批量张量占用更多空间！
// 因此，程序中的每个输入、中间结果和输出都需要用 FunctionalTensorImpl 包装，
// 该类包装了底层张量。
// 替换当前函数式张量的值为给定张量 `other`，并确保不会嵌套进行 functionalize() 变换。
void FunctionalTensorWrapper::replace_(const Tensor& other, bool from_lazy_regenerate) {
  // 断言给定的张量 `other` 不是函数式张量
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(other));
  // 将当前函数式张量的值替换为 `other`
  value_ = other;
  // 断言当前张量不具有 `Functionalize` 分发键
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  // `out=` 操作允许调整输出张量的大小，同时改变张量的数据和元数据。需要将这些元数据的变化传播到包装器中（新的大小）。
  auto sizes_ = value_.sym_sizes();
  auto strides_ = value_.sym_strides();
  auto storage_offset_ = value_.sym_storage_offset();
  set_sizes_and_strides(sizes_, strides_, storage_offset_);
  // 如果当前张量的数据类型或布局与 `value_` 不同，需要进行类型转换
  if (dtype() != value_.unsafeGetTensorImpl()->dtype() || layout() != value_.unsafeGetTensorImpl()->layout()) {
    // 在执行 `.to()` 操作时不应通过 `Functionalize` 分发键重新进入函数化状态
    at::AutoDispatchSkipFunctionalize guard;
    // 通过 `_to_copy()` 函数将 `value_` 转换为指定的数据类型和布局
    value_ = at::_to_copy(value_, c10::TensorOptions().dtype(dtype()).layout(layout()));
    // 断言当前张量不具有 `Functionalize` 分发键
    TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  }
  // 如果不是懒惰重新生成，则标记发生了变异
  if (!from_lazy_regenerate) {
    mark_mutation();
    // 如果未启用梯度或启用推断模式，则标记在无梯度或推断模式下发生了变异
    if (!at::GradMode::is_enabled() || InferenceMode::is_enabled()) {
      mark_mutation_during_no_grad_or_inference_mode();
    }
  }
}

// 检查当前函数式张量的数据是否发生了变异
bool FunctionalTensorWrapper::has_data_mutation() {
  // 如果当前张量的存储观察到了任何变异，则表明当前张量的数据已经发生了变异
  return functional_storage_impl()->generation() > 0;
}
// 将当前对象的属性设置为另一个 FunctionalTensorWrapper 对象的属性值
void FunctionalTensorWrapper::set__impl(const FunctionalTensorWrapper* other) {
  // 将 value_ 属性设置为另一个对象的 value_ 属性
  value_ = other->value_;
  // 将 generation_ 属性设置为另一个对象的 generation_ 属性
  generation_ = other->generation_;
  // 将 view_metas_ 属性设置为另一个对象的 view_metas_ 属性
  view_metas_ = other->view_metas_;
  // 将 is_symbolic_ 属性设置为另一个对象的 is_symbolic_ 属性
  is_symbolic_ = other->is_symbolic_;
  
  // 冻结当前对象的 functional_storage_impl() 返回的存储空间，防止其发生变化
  functional_storage_impl()->freeze();
  
  // 将当前对象的 storage_ 属性替换为另一个对象的 storage_ 属性，
  // 断开当前对象与其视图链的连接
  storage_ = other->storage_;
  
  // 明确标记张量的存储已从 set_() 中更改
  was_storage_changed_ = true;

  // 获取当前对象 value_ 的符号化大小、步幅和存储偏移
  auto sizes_ = value_.sym_sizes();
  auto strides_ = value_.sym_strides();
  auto storage_offset_ = value_.sym_storage_offset();
  // 设置当前对象的大小、步幅和存储偏移
  set_sizes_and_strides(sizes_, strides_, storage_offset_);
}

// 更改张量的存储大小（内部实现）
void FunctionalTensorWrapper::storage_resize_(c10::SymInt new_size) {
  // 获取当前存储的大小
  auto curr_storage_size = value_.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl()->sym_nbytes();
  
  // 限制存储大小的改变：只支持将大小调整为零或从零开始调整
  TORCH_CHECK(new_size == 0 || curr_storage_size == 0, "new_size: ", new_size, ". curr_storage_size: ", curr_storage_size);
  
  // 存储大小变更的功能化规则几乎是一个空操作，主要是因为我们不希望 resize_() 调用实际上在功能图中生成任何操作。
  // 工作原理是什么？
  // 扩大大小（旧大小为零）：
  //   在这种情况下我们什么也不做。
  //   假设用户代码有效的下一个操作应该对当前张量 "x" 运行 x.copy_(y)（或类似操作），
  //   这将完全覆盖 x 的数据。
  //   如果 x 有任何未完成的别名，我们预期在复制调用之后不会使用它们
  //   （否则急切代码将无效），
  //   因此功能化将重新生成 x.copy(y) 结果的别名。
  // 缩小大小（新大小为零）：
  //   在这种情况下我们同样不做任何操作。假设在缩小张量后，它在程序中完全未使用
  //   （除非稍后首先将其从零调整为较大值，并复制数据进去）
  //   虽然它可能会在 FSDP 中用于反向传播保存。
  //   预期的模式是参数在反向传播时将会从零调整大小以及复制数据。

  // 标记张量的存储已调整大小
  // 这样我们可以检测到它，用于 AOTAutograd 中的输入，并适当地出错/发出输入突变调整大小
  functional_storage_impl()->mark_inductor_storage_resize(new_size);
}
// 注意事项 [functionalization pass 中的 resize_() 方法]
// resize_() 是 functionalization 中的一个特殊操作符，因为它可以重新分配其底层存储。
// 这个函数只会在需要将 resize_() 重新分配到更大尺寸时调用。

// 然而，functionalization 目前禁止以下代码的使用：
//   a = torch.ones(2)
//   b = a.view(2)
//   b.resize_(4) // b 是一个视图张量，我们试图增加其存储大小

// 为什么这段代码难以处理？
// functionalization 通过以下假设来保持别名同步：
// - “基础”张量始终指向“所有数据”
// - 每当有 b = view_op(a) 时，“b”应始终指向“a”的某些内存子集。

// 上面的代码破坏了这个假设，b.resize_(4) 实际上需要更新 “a”，告诉它现在实际上是某个现有较大存储的一部分切片。
// 我们也不再完全从 “a” 重新生成 “b”，因为 “a” 现在引用 “b” 数据的一部分切片。

// 理论上这是可以修复的，但是：
// - 修复可能会大大复杂化 functionalization 的逻辑。
// - 当前 resize_() 的主要用例是在操作符的 out= 变体中调整大小为零的张量。
// - 如果尝试调整大小一个奇怪步幅的张量，resize_() 也可能给出奇怪的结果。

// 综上所述，目前我们只是禁止上述用法。

// 检查存储的 use_count 是否为 1，确保这不是一个视图张量尝试将其大小调整为更大。
TORCH_CHECK(storage().use_count() == 1, "Attempted to resize a view tensor to a larger size. This is not allowed in the functionalization pass");

// 检查是否存在视图元数据，确保这不是一个视图张量尝试将其大小调整为更大。
TORCH_CHECK(view_metas_.empty(), "Attempted to resize a view tensor to a larger size. This is not allowed in the functionalization pass");

// 如果这个张量不是一个视图（并且没有任何视图依赖于它），
// 那么可以安全地丢弃旧存储并替换为新的、更大的存储。
storage_ = c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(other));
value_ = other;

// 内部断言，确保 value_ 不含有 Functionalize 的 DispatchKey。
TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));

// 重置 generation_ 为 0，因为存储已经更新。
generation_ = 0;

// 更新包装器上的元数据，以反映新的大小和步幅。
set_sizes_and_strides(value_.sizes(), value_.strides());
refresh_numel();

// 刷新张量的连续性（理论上应该保证张量已经是连续的，因为它保证不是一个视图，但是运行一下也无妨）。
refresh_contiguous();

// 替换张量的存储（比如从 resize_() 调用），将更新张量的大小和步幅，
// 因此我们需要记录元数据已被改变的事实。
has_metadata_mutation_ = true;
}
// 重置存储器，使用当前 value_ 张量作为基础
void FunctionalTensorWrapper::_unsafe_reset_storage() {
    // 使用 value_ 张量创建一个 FunctionalStorageImpl 对象，并将其作为存储器的新基础
    storage_ = c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(value_));
    // 重置代数，使其与新存储器匹配
    generation_ = 0;
    // 清除任何预先存在的视图元信息，确保 base 和 value_ 在语义上相同
    view_metas_.clear();
}

// 同步方法，如果已经是最新状态则返回，否则应用更新并从基础重新生成
void FunctionalTensorWrapper::sync_() {
    if (is_up_to_date()) {
        return;
    }
    // 应用所有更新到 alias_ 上
    apply_updates();
    // 从基础重新生成
    regenerate_from_base();
}

// 应用视图元信息到 base 张量上，并返回结果张量
Tensor FunctionalTensorWrapper::apply_view_metas(const Tensor& base) {
    auto t = base;

    // 逐个应用视图函数，从基础张量获取视图张量
    for (auto& view_meta: view_metas_) {
        t = view_meta.forward_fn(t, view_meta.out_index);
    }

    return t;
}

// 从基础重新生成张量
void FunctionalTensorWrapper::regenerate_from_base() {
    // 跳过功能化自动分发
    at::AutoDispatchSkipFunctionalize guard;
    auto storage_impl = functional_storage_impl();
    auto t = storage_impl->base();

    // 断言不是功能张量，应用视图元信息
    TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
    t = apply_view_metas(t);
    // 断言不是功能张量
    TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));

    // 替换当前张量
    replace_(t, /*from_lazy_regenerate=*/true);
    // 更新代数
    generation_ = storage_impl->generation();
}

// 应用所有更新到 alias_ 上
bool FunctionalTensorWrapper::apply_updates() {
    auto storage_impl = functional_storage_impl();
    // 调用存储器的应用更新方法
    return storage_impl->apply_updates();
}

// 返回张量实现的类型名
const char* FunctionalTensorWrapper::tensorimpl_type_name() const {
    return "FunctionalTensorWrapper";
}

// 复制张量元数据
void FunctionalTensorWrapper::copy_tensor_metadata(
    const FunctionalTensorWrapper* src_impl,
    FunctionalTensorWrapper* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
    // 调用 TensorImpl 中的复制张量元数据方法
    TensorImpl::copy_tensor_metadata(
        src_impl,
        dest_impl,
        version_counter,
        allow_tensor_metadata_change);

    // 复制 FunctionalTensorWrapper 特定字段
    dest_impl->value_ = src_impl->value_;
    dest_impl->level_ = src_impl->level_;
    dest_impl->has_metadata_mutation_ = src_impl->has_metadata_mutation_;
    dest_impl->is_multi_output_view_ = src_impl->is_multi_output_view_;
    dest_impl->was_storage_changed_ = src_impl->was_storage_changed_;
    dest_impl->is_symbolic_ = src_impl->is_symbolic_;
    dest_impl->generation_ = src_impl->generation_;
    dest_impl->view_metas_ = src_impl->view_metas_;
}

// 复制张量元数据并刷新
void FunctionalTensorWrapper::copy_tensor_metadata_and_refresh(
    const FunctionalTensorWrapper* src_impl,
    FunctionalTensorWrapper* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
    // 复制张量元数据
    copy_tensor_metadata(src_impl, dest_impl, version_counter, allow_tensor_metadata_change);
    // 刷新 numel 和 contiguous 属性
    dest_impl->refresh_numel();
    dest_impl->refresh_contiguous();
}
    // 如果键集合中包含 Python 分发键，并且当前线程不排除 Python 分发键
    if (key_set_.has(DispatchKey::Python) &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      // 加载 Python 对象解释器，并分离当前对象
      auto r = pyobj_slot_.load_pyobj_interpreter()->detach(this);
      // 如果成功分离，设置版本计数器和是否允许更改张量元数据，并返回分离后的对象
      if (r) {
        r->set_version_counter(std::forward<VariableVersion>(version_counter));
        r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
        return r;
      }
    }

    // 创建一个基于 value_ 的 FunctionalTensorWrapper 对象的智能指针
    auto impl = c10::make_intrusive<FunctionalTensorWrapper>(value_);
    // 复制当前对象的张量元数据到新创建的 impl 对象，并刷新其版本计数器和元数据变更设置
    copy_tensor_metadata_and_refresh(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/std::forward<VariableVersion>(version_counter),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    // 返回新创建的 impl 对象
    return impl;
}

c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  // 使用核心方法进行浅拷贝和分离操作，返回新的 TensorImpl 指针
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  // 使用核心方法进行浅拷贝和分离操作，返回新的 TensorImpl 指针，使用移动语义
  return shallow_copy_and_detach_core(
      std::move(version_counter), allow_tensor_metadata_change);
}

void FunctionalTensorWrapper::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
    // 断言确保传入的 TensorImpl 类型与当前实例兼容
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto functional_impl =
        static_cast<FunctionalTensorWrapper*>(impl.get());
    // 复制张量的元数据并刷新
    copy_tensor_metadata_and_refresh(
        /*src_impl=*/functional_impl,
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
}

c10::Device FunctionalTensorWrapper::device_custom() const {
  // 返回底层 TensorImpl 的设备信息
  return value_.unsafeGetTensorImpl()->device();
}
at::IntArrayRef FunctionalTensorWrapper::sizes_custom() const {
  // 返回底层 TensorImpl 的尺寸信息
  return value_.unsafeGetTensorImpl()->sizes();
}
at::IntArrayRef FunctionalTensorWrapper::strides_custom() const {
  // 返回底层 TensorImpl 的步幅信息
  return value_.unsafeGetTensorImpl()->strides();
}
int64_t FunctionalTensorWrapper::dim_custom() const {
  // 返回底层 TensorImpl 的维度数
  return value_.unsafeGetTensorImpl()->dim();
}
int64_t FunctionalTensorWrapper::numel_custom() const {
  // 返回底层 TensorImpl 的元素总数
  return value_.unsafeGetTensorImpl()->numel();
}
bool FunctionalTensorWrapper::is_contiguous_custom(at::MemoryFormat memory_format) const {
  // 检查底层 TensorImpl 是否以给定的内存格式连续存储
  return value_.unsafeGetTensorImpl()->is_contiguous(memory_format);
}
c10::SymIntArrayRef FunctionalTensorWrapper::sym_sizes_custom() const {
  // 返回底层 TensorImpl 的符号化尺寸信息
  return value_.unsafeGetTensorImpl()->sym_sizes();
}
c10::SymIntArrayRef FunctionalTensorWrapper::sym_strides_custom() const {
  // 返回底层 TensorImpl 的符号化步幅信息
  return value_.unsafeGetTensorImpl()->sym_strides();
}
c10::SymInt FunctionalTensorWrapper::sym_size_custom(int64_t d) const {
  // 返回底层 TensorImpl 的指定维度的符号化尺寸信息
  return value_.unsafeGetTensorImpl()->sym_size(d);
}
c10::SymInt FunctionalTensorWrapper::sym_storage_offset_custom() const {
  // 返回底层 TensorImpl 的符号化存储偏移信息
  return value_.unsafeGetTensorImpl()->sym_storage_offset();
}

namespace functionalization {
namespace impl {

Tensor to_functional_tensor(const Tensor& tensor) {
  // 注释：如果张量是封装的数值类型，则直接返回该张量
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      return tensor;
  }
  // 断言：仅在调试模式下确保不是 FunctionalTensor
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isFunctionalTensor(tensor));
  // 使用 FunctionalTensorWrapper 封装原始张量，并返回新的功能化张量
  return at::detail::make_tensor<FunctionalTensorWrapper>(tensor);
}

std::optional<Tensor> to_functional_tensor(const std::optional<Tensor>& tensor) {
  // 如果传入的 optional<Tensor> 有值，则将其转换为 FunctionalTensor
  if (tensor.has_value()) {
    return c10::make_optional<Tensor>(to_functional_tensor(*tensor));
  }
  // 如果没有值，则返回空的 optional<Tensor>
  return c10::nullopt;
}
// 将输入的列表转换为功能化张量列表，并返回结果
c10::List<::std::optional<Tensor>> to_functional_tensor(const c10::List<::std::optional<Tensor>>& t_list) {
  // 创建输出列表以存放结果
  c10::List<::std::optional<Tensor>> outputs;
  // 预先分配输出列表的空间以提高性能
  outputs.reserve(t_list.size());
  // 遍历输入列表中的每个元素
  for (const auto i : c10::irange(t_list.size())) {
    // 调用单个张量转换函数，并将结果添加到输出列表中
    outputs.push_back(to_functional_tensor(t_list[i]));
  }
  // 返回转换后的功能化张量列表
  return outputs;
}

// 将张量列表转换为功能化张量列表，并返回结果
std::vector<Tensor> to_functional_tensor(ITensorListRef t_list) {
  // 创建输出向量以存放结果
  std::vector<Tensor> outputs;
  // 预先分配输出向量的空间以提高性能
  outputs.reserve(t_list.size());
  // 遍历输入张量列表中的每个张量
  for (const auto& tensor : t_list) {
    // 调用单个张量转换函数，并将结果添加到输出向量中
    outputs.push_back(to_functional_tensor(tensor));
  }
  // 返回转换后的功能化张量向量
  return outputs;
}

// 将功能化张量还原为普通张量，并返回结果
Tensor from_functional_tensor(const Tensor& tensor, bool assert_functional) {
  // 如果张量未定义或是包装数字，则直接返回输入张量
  if (!tensor.defined() || tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      return tensor;
  }
  // 如果输入张量已经是功能化张量，则从功能化包装器中获取其值并返回
  if (isFunctionalTensor(tensor)) {
    auto impl = unsafeGetFunctionalWrapper(tensor);
    return impl->value();
  } else {
    // 如果当前张量不是功能化张量，并且 assert_functional 为真，则抛出错误
    TORCH_INTERNAL_ASSERT(!assert_functional)
    // 否则，返回输入张量
    return tensor;
  }
}

// 将可选的功能化张量还原为普通张量，并返回结果
std::optional<Tensor> from_functional_tensor(const std::optional<Tensor>& t, bool assert_functional) {
  // 如果输入可选张量有值，则调用单个张量还原函数，并使用 make_optional 封装后返回
  if (t.has_value()) {
    return c10::make_optional<Tensor>(from_functional_tensor(*t, assert_functional));
  }
  // 如果输入可选张量为空，则返回空值
  return c10::nullopt;
}

// 将张量列表转换为普通张量列表，并返回结果
std::vector<Tensor> from_functional_tensor(ITensorListRef t_list) {
  // 创建输出向量以存放结果
  std::vector<Tensor> outputs;
  // 预先分配输出向量的空间以提高性能
  outputs.reserve(t_list.size());
  // 遍历输入张量列表中的每个张量
  for (const auto& tensor : t_list) {
    // 调用单个张量还原函数，并将结果添加到输出向量中
    // 注意：from_functional_tensor(Tensor) 被设计用来确保不会误将非功能化输入调用为功能化输入，
    // 而 from_functional_tensor(TensorList) 可以接收包含功能化和非功能化张量的列表。
    // 例如：torch.cat(function_input_tensor, global_state_tensor)。
    // 当这种情况发生时，我们只需解包功能化张量即可。
    outputs.push_back(from_functional_tensor(tensor, /*assert_functional=*/false));
  }
  // 返回转换后的普通张量向量
  return outputs;
}

// 将输入的列表转换为普通张量列表，并返回结果
c10::List<::std::optional<Tensor>> from_functional_tensor(const c10::List<::std::optional<Tensor>>& t_list) {
  // 创建输出列表以存放结果
  c10::List<::std::optional<Tensor>> outputs;
  // 预先分配输出列表的空间以提高性能
  outputs.reserve(t_list.size());
  // 遍历输入列表中的每个元素
  for (const auto i : c10::irange(t_list.size())) {
    // 调用单个张量还原函数，并将结果添加到输出列表中
    outputs.push_back(from_functional_tensor(t_list[i], /*assert_functional=*/false));
  }
  // 返回转换后的普通张量列表
  return outputs;
}

// 同步操作，确保张量不是包装数字
void sync(const Tensor& t) {
  // 如果张量是包装数字，则打印注释内容
  if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
    // 注意 [Wrapped Numbers <> Functionalization]
    // 不幸的是，我们不能保证包装数字（标量张量）会被包装到 FunctionalTensorWrapper 对象中，
    // 因为它们会跳过分派过程。但由于我们通常不允许对包装数字进行赋值，这应该不会成为问题。
    // 因此，如果张量是包装数字，则可以安全地跳过功能化步骤。
  }
}
    return;
  }


  // 如果条件不满足，直接返回，结束函数执行
  // 这里用于检查输入的张量是否符合功能化张量的要求


  // Not every tensor that hits a functionalization kernel is necessarily a functional tensor.
  // For example, xla_tensor.copy_(cpu_tensor) needs to hit the functionalization kernel
  // to sync xla_tensor, but not cpu_tensor.


  // 并非每个触发功能化内核的张量都是功能化张量。
  // 例如，xla_tensor.copy_(cpu_tensor) 需要触发功能化内核来同步 xla_tensor，但不需要同步 cpu_tensor。
  // 以下代码段处理的是功能化张量的情况，跳过非功能化张量的处理。


  if (!at::functionalization::impl::isFunctionalTensor(t)) {
    return;
  }


  // 如果张量 t 不是功能化张量，则直接返回，结束函数执行
  // 判断张量 t 是否符合功能化张量的标准


  auto functional_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
  functional_impl->sync_();


  // 获取张量 t 的功能化包装器，并调用其 sync_() 方法同步张量
  // 这里假设 t 是功能化张量，通过其功能化包装器来进行同步操作
}
void sync(const std::optional<Tensor>& t) {
  // 如果可选的张量值存在，则递归调用同步函数
  if (t.has_value()) {
    sync(*t);
  }
}
void sync(ITensorListRef t_list) {
  // 遍历输入张量列表，并对每个张量调用同步函数
  for (const auto& t : t_list) {
    sync(t);
  }
}
void sync(const c10::List<::std::optional<Tensor>>& t_list) {
  // 遍历输入张量可选列表，对每个张量值调用同步函数
  for (const auto i : c10::irange(t_list.size())) {
    sync(t_list[i]);
  }
}

void replace_(const Tensor& functional_tensor, const Tensor& other) {
  // 内部断言检查功能张量是否有效，并在其功能包装器上调用替换函数
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  unsafeGetFunctionalWrapper(functional_tensor)->replace_(other);
}

void replace_(const ITensorListRef functional_tensor, ITensorListRef other) {
  // 内部断言检查功能张量列表的大小是否相等，然后逐个替换其中的张量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_tensor.size() == other.size());
  auto functional_tensor_it = functional_tensor.begin();
  auto other_it = other.begin();
  for (C10_UNUSED const auto i : c10::irange(functional_tensor.size())) {
    replace_(*functional_tensor_it++, *other_it++);
  }
}

void propagate_xla_data(const Tensor& functional_tensor, const Tensor& other) {
  // 内部断言检查功能张量是否有效，并且其分发键中是否包含 XLA，若包含则传播 XLA 数据
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  if (functional_tensor.key_set().has(c10::DispatchKey::XLA)) {
    at::_propagate_xla_data(at::functionalization::impl::unsafeGetFunctionalWrapper(functional_tensor)
        ->value(), other);
  }
}

void propagate_xla_data(const ITensorListRef functional_tensor, ITensorListRef other) {
  // 内部断言检查功能张量列表的大小是否相等，然后逐个传播 XLA 数据
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_tensor.size() == other.size());
  auto functional_tensor_it = functional_tensor.begin();
  auto other_it = other.begin();
  for (C10_UNUSED const auto i : c10::irange(functional_tensor.size())) {
    propagate_xla_data(*functional_tensor_it++, *other_it++);
  }
}

void commit_update(const Tensor& functional_tensor) {
  // 内部断言检查功能张量是否有效，并在其功能包装器上调用提交更新函数
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  unsafeGetFunctionalWrapper(functional_tensor)->commit_update();
}

void commit_update(ITensorListRef functional_tensor) {
  // 遍历功能张量列表，并对每个张量调用提交更新函数
  for (const auto& t : functional_tensor) {
    commit_update(t);
  }
}

void unsafe_reset_storage(const Tensor& functional_tensor) {
  // 内部断言检查功能张量是否有效，并在其功能包装器上调用不安全的存储重置函数
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  unsafeGetFunctionalWrapper(functional_tensor)->_unsafe_reset_storage();
}

void mark_mutation_hidden_from_autograd(const Tensor& functional_tensor) {
  // 检查功能张量是否有效，并在其功能包装器上标记隐藏自自动求导的突变
  TORCH_CHECK(isFunctionalTensor(functional_tensor));
  unsafeGetFunctionalWrapper(functional_tensor)->mark_mutation_hidden_from_autograd();
}

bool are_all_mutations_hidden_from_autograd(const Tensor& functional_tensor) {
  // 检查功能张量是否有效，并查询其功能包装器是否所有突变均已隐藏自自动求导
  TORCH_CHECK(isFunctionalTensor(functional_tensor));
  return unsafeGetFunctionalWrapper(functional_tensor)->are_all_mutations_hidden_from_autograd();
}

bool are_all_mutations_under_no_grad_or_inference_mode(const Tensor& functional_tensor) {
  // 检查功能张量是否有效，并查询其功能包装器是否所有突变均处于无梯度或推理模式下
  TORCH_CHECK(isFunctionalTensor(functional_tensor));
  return unsafeGetFunctionalWrapper(functional_tensor)->are_all_mutations_under_no_grad_or_inference_mode();
}
bool isFunctionalTensor(const at::Tensor& tensor) {
  // 检查给定的张量是否具有功能化调度键
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Functionalize);
}

bool isFunctionalTensor(const std::optional<Tensor>& t) {
  if (t.has_value()) {
    // 如果可选张量有值，则调用前一个函数检查是否为功能化张量
    return isFunctionalTensor(*t);
  } else {
    // 否则，返回 false
    return false;
  }
}

bool isFunctionalTensor(const c10::List<::std::optional<Tensor>>& t_list) {
  // 如果列表为空，直接返回 false
  if (t_list.empty()) return false;
  auto functional_count = 0;
  // 遍历列表中的每个元素
  for (const auto i : c10::irange(t_list.size())) {
    // 如果当前位置的元素不存在值或者未定义，则跳过
    if (!t_list[i].has_value() || !t_list[i]->defined()) continue;
    // 如果当前元素是功能化张量，则计数增加
    if (isFunctionalTensor(t_list[i])) {
      ++functional_count;
    }
  }
  // 返回是否存在功能化张量的计数结果大于零
  return functional_count > 0;
}

template <typename T>
bool isFunctionalTensorIListRef(c10::IListRef<T> list) {
  // 如果列表为空，直接返回 false
  if (list.size() == 0) return false;
  auto functional_count = 0;
  // 遍历列表中的每个元素
  for (const auto& tensor : list) {
    // 如果当前元素未定义，则跳过
    if (!tensor.defined()) continue;
    // 如果当前元素是功能化张量，则计数增加
    if (isFunctionalTensor(tensor)) {
      ++functional_count;
    }
  }
  // 返回是否存在功能化张量的计数结果大于零
  return functional_count > 0;
}

bool isFunctionalTensor(ITensorListRef list) {
  // 调用前一个模板函数来检查功能化张量是否存在
  return isFunctionalTensorIListRef(list);
}

void freeze_functional_tensor(const Tensor& tensor) {
  // 断言给定张量确实是功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(tensor));
  // 获取功能化张量的内部实现，并冻结其存储
  auto functional_base_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
  functional_base_impl->freeze_storage();
}

Tensor create_functional_tensor_with_view_meta(const at::Tensor& view_to_wrap, const at::Tensor& base, functionalization::ViewMeta meta, int64_t out_idx) {
  // 断言视图张量不是功能化张量
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(view_to_wrap));
  // 断言基础张量是功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(base));
  // 获取基础张量的功能化包装实现
  auto functional_base_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(base);
  if (out_idx != 0) {
    // 如果输出索引不为零，更新 ViewMeta 对象以包含输出索引信息
    meta = meta.to_out_idx(out_idx);
  }
  // 创建一个具有功能化视图元数据的张量
  return at::detail::make_tensor<FunctionalTensorWrapper>(view_to_wrap, functional_base_impl, meta);
}

std::vector<Tensor> create_functional_tensor_with_view_meta(ITensorListRef view_to_wrap, const at::Tensor& base, const functionalization::ViewMeta& meta) {
  // 创建一个向量以保存输出张量
  std::vector<Tensor> outputs(view_to_wrap.size());
  int64_t i = 0;
  // 遍历视图张量列表，并使用给定的基础张量和元数据创建功能化张量
  for (const auto& tensor : view_to_wrap) {
    outputs[i] = create_functional_tensor_with_view_meta(tensor, base, meta, i);
    i++;
  }
  // 返回创建的功能化张量向量
  return outputs;
}

void mutate_view_meta(const at::Tensor& self, const functionalization::ViewMeta& meta) {
  // 断言给定张量确实是功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  // 获取功能化张量的内部实现，并修改其视图元数据
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  self_impl->mutate_view_meta(meta);
}

// Note [Propagating strides in the functionalization pass]
// 功能化过程中传播步长的注意事项
// 设置输出张量的大小、步长和存储偏移量，以便正确计算步长信息。
// 这个函数用于函数化过程中，通过调用视图引用实现，参考输出张量的步长信息来确定正确的步长。
void set_sizes_strides_offset(const Tensor& out, const Tensor& reference_out) {
  out.unsafeGetTensorImpl()->set_sizes_and_strides(reference_out.sym_sizes(), reference_out.sym_strides(), reference_out.sym_storage_offset());
}

// 对多个输出张量进行设置大小、步长和偏移量操作，保证一致性。
// 这个函数用于函数化过程中，确保每个输出张量与其对应的参考输出张量具有相同的大小和步长。
void set_sizes_strides_offset(const std::vector<Tensor>& outs, const std::vector<Tensor>& reference_outs) {
  TORCH_INTERNAL_ASSERT(outs.size() == reference_outs.size());
  for (const auto i : c10::irange(reference_outs.size())) {
    set_sizes_strides_offset(outs[i], reference_outs[i]);
  }
}

// 线程本地变量，用于存储是否重新应用视图的状态。
thread_local bool _functionalizationReapplyViews;

// 获取线程本地变量 _functionalizationReapplyViews 的值。
bool getFunctionalizationReapplyViewsTLS() {
  return _functionalizationReapplyViews;
}

// 设置线程本地变量 _functionalizationReapplyViews 的值。
void setFunctionalizationReapplyViewsTLS(bool reapply_views) {
  _functionalizationReapplyViews = reapply_views;
}
    } else if (ivalue.isOptionalTensorList()) {
      // 如果输入值是可选的张量列表，则执行以下操作
      auto opt_tensors = ivalue.toOptionalTensorList();
      // 断言所有输入都不是功能张量，因为复合操作功能化后退期望其输入均不是功能张量
      TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(opt_tensors),
        "The composite op functionalization fallback expects its inputs all not to be functional tensors");
      // 将可选张量列表转换为功能张量并更新到堆栈中
      auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(opt_tensors));
      (*stack)[arguments_begin + idx] = t_new;
    }
  }

  {
    // 当使用 at::empty(device=lazy) 时，懒加载后端根据 TLS 决定是否包装输出为功能张量
    // 在这段代码中，我们在同一个调用堆栈中重新进入功能化，因此需要手动修复 TLS，就好像还没有调用过一样
    auto curr_tls = c10::impl::tls_local_dispatch_key_set();
    auto tls_reenable_functionalize = c10::impl::PODLocalDispatchKeySet();
    tls_reenable_functionalize.set_included(curr_tls.included_);
    // 排除功能化分发键以恢复默认行为
    tls_reenable_functionalize.set_excluded(curr_tls.excluded_.remove(c10::DispatchKey::Functionalize));
    // 强制分发键保护以确保 TLS 的正确设置
    c10::impl::ForceDispatchKeyGuard guard_(tls_reenable_functionalize);
    // 所以，我们应该提供一种直接调用注册到 `CompositeExplicitAutograd` 键的内核的方法
    // 今天我们做不到这一点，所以这应该是一个合理的代理
    // (在操作既有 CompositeExplicitAutograd 内核又有专用元内核的情况下，这不会起作用，但这种情况可能永远不会发生)
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Meta), stack);
  }

  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  auto returns = torch::jit::last(stack, num_returns);

  for (const auto idx : c10::irange(num_returns)) {
    const auto& ivalue = returns[idx];
    // 如果返回值是张量
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      // 如果张量未定义，则继续下一个迭代
      if (!t.defined()) continue;
      // 同步功能张量的数据
      at::functionalization::impl::sync(t);
      // 将功能张量转换为普通张量并更新到堆栈中
      auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(t));
      (*stack)[returns_begin + idx] = t_new;
    } else if (ivalue.isTensorList()) {
      // 如果返回值是张量列表
      auto tensors = ivalue.toTensorList();
      // 同步功能张量列表的数据
      at::functionalization::impl::sync(tensors);
      // 将功能张量列表转换为普通张量列表并更新到堆栈中
      auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(tensors));
      (*stack)[returns_begin + idx] = t_new;
    } else if (ivalue.isOptionalTensorList()) {
      // 如果返回值是可选的张量列表
      auto opt_tensors = ivalue.toOptionalTensorList();
      // 同步功能张量列表的数据
      at::functionalization::impl::sync(opt_tensors);
      // 将功能张量列表转换为普通张量列表并更新到堆栈中
      auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(opt_tensors));
      (*stack)[returns_begin + idx] = t_new;
    }
  }
}

} // namespace functionalization
} // namespace at
```