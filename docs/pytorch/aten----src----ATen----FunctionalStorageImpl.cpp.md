# `.\pytorch\aten\src\ATen\FunctionalStorageImpl.cpp`

```py
// 实现ViewMeta类的成员函数，将当前实例转换为指定输出索引的新实例
ViewMeta ViewMeta::to_out_idx(int64_t out_idx) {
  // 如果输出索引与当前实例的输出索引相同，则直接返回当前实例
  if (out_idx == this->out_index) return *this;
  // 否则，创建并返回一个新的ViewMeta实例，其中输出索引被更新为指定的out_idx
  return ViewMeta(forward_fn, reverse_fn, has_symbolic_inputs, is_multi_output, is_as_strided, out_idx);
}

// Note [Functionalization: Alias Removal Part 2]
// See Note [Functionalization: Alias Removal] for more details.
// 此函数应用来自一个视图到StorageImpl的单个更新。
// 我们从<original_base>和<mutated_view>开始，目标是得到<mutated_base>。
// 考虑以下程序：
//
// base = ...
// a = base.view1()
// b = a.view2()
// c = b.view3()
// c.add_(3)
//
// 然后，functionalization pass将如下队列更新：
//
// update.new_val = c  // c的更新值
// update.view_metas = [view1_meta, view2_meta, view3_meta]
//
// 对a、b或c中的任何一个进行同步最终将在storage上调用apply_update()，并且将执行以下操作：
//
// tmp_values = [base, a, b]  // 注意：c不是必需的
// t = update.new_val
// t = view3_inverse(b, t, 0)  // 0是输出索引，这些都是单输出视图，所以是0
// t = view2_inverse(a, t, 0)
// t = view1_inverse(base, t, 0)  // 现在t代表更新后的存储。
// storage.base_ = t
static const Tensor apply_update(const FunctionalStorageImpl::Update& update, const Tensor& base) {
  // 获取更新值
  at::Tensor t = update.new_val;
  // 检查t是否不是FunctionalTensor
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  // 如果没有视图元数据，则直接返回t
  if (update.view_metas.empty()) return t;

  // 创建临时值向量，包含base作为初始值
  std::vector<at::Tensor> tmp_values({base});
  // 根据视图元数据顺序，扩展临时值向量
  tmp_values.reserve(update.view_metas.size());
  for (size_t i = 0; i < update.view_metas.size() - 1; ++i) {
    // 计算下一个视图的Tensor
    at::Tensor next_view = update.view_metas[i].forward_fn(tmp_values.back(), update.view_metas[i].out_index);
    // 注意：我们只需要对像select/slice/diagonal/squeeze/as_strided这样的操作实际计算tmp_values。
    // 所有这些操作都需要额外的信息来恢复原始张量的大小。
    // 如果需要，我们可以应用此优化，并仅在必要的视图操作中才计算tmp_values。
    tmp_values.push_back(std::move(next_view));
  }

  // 逆序遍历视图元数据，并应用反向函数
  for(int i = update.view_metas.size()-1; i >= 0; --i) {
    int64_t out_idx = update.view_metas[i].out_index;
    // 每个视图的逆操作实现在ViewInverses.cpp中。
    t = update.view_metas[i].reverse_fn(tmp_values[i], t, out_idx);
  }

  // 最终检查t是否不是FunctionalTensor
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  // 返回最终更新后的Tensor t
  return t;
}
// 返回给定张量的字节大小，用于初始化 FunctionalStorageImpl 对象的存储空间
static c10::SymInt get_nbytes(const Tensor& value) {
  // 如果张量是稀疏张量，则返回零
  if (value.is_sparse()) {
    return 0;
  }
  // 如果张量具有符号大小和步长，则根据不同的情况计算并返回存储空间的字节数
  if (value.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    // 对于使用 Python 代理张量和懒惰张量的情况，根据情况返回对应的符号字节数
    if (value.key_set().has(c10::DispatchKey::Python)) {
      return value.storage().sym_nbytes();
    }
    return at::detail::computeStorageNbytes(value.sym_sizes(), value.sym_strides(), value.dtype().itemsize(), value.sym_storage_offset());
  }
  // 对于其他情况，计算并返回存储空间的字节数
  return at::detail::computeStorageNbytes(value.sizes(), value.strides(), value.dtype().itemsize(), value.storage_offset());
}

// 使用给定的基础张量初始化 FunctionalStorageImpl 对象
FunctionalStorageImpl::FunctionalStorageImpl(const Tensor& base)
  : c10::StorageImpl(
      c10::StorageImpl::use_byte_size_t(), // 使用字节大小的构造函数
      get_nbytes(base),                   // 使用给定张量的字节大小作为存储空间大小
      DataPtr{nullptr, base.device()},    // 使用 nullptr 初始化 DataPtr，并使用基础张量的设备
      GetAllocator(kMeta),                // 获取分配器为 kMeta 的分配器
      /*resizable=*/true                 // 设置存储空间为可调整大小
    ),
    base_(base)                          // 初始化 base_ 成员变量为给定的基础张量
{
  // 如果基础张量具有存储空间且不是 XLA 设备类型，则查询其原始存储空间大小
  // 否则将 original_storage_size_ 设置为 -1，表示无法查询
  if (base.unsafeGetTensorImpl()->has_storage() && base.device().type() != c10::DeviceType::XLA) {
    original_storage_size_ = base.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl()->sym_nbytes();
  } else {
    original_storage_size_ = -1;
  }
  curr_storage_size_ = original_storage_size_; // 当前存储空间大小初始化为原始存储空间大小
  // 断言确保基础张量不是函数化张量
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(base_));
}

// 添加更新到 FunctionalStorageImpl 对象中
void FunctionalStorageImpl::add_update(const Tensor& updated_val, const std::vector<ViewMeta>& metas) {
  // 确保存储空间未冻结，否则无法对其进行修改
  TORCH_CHECK(!frozen_, "cannot mutate tensors with frozen storage");

  // 如果视图元数据数量大于 1，则遍历处理每个视图元数据
  if (metas.size() > 1) {
    for (size_t i = 1; i < metas.size(); ++i) {
      // 对于 XLA 设备类型的更新张量，跳过此检查
      // 否则，输出错误信息说明编译过程中发现了不支持的视图链长度和 as_strided() 调用
      TORCH_CHECK(updated_val.device().type() == c10::DeviceType::XLA || !metas[i].is_as_strided,
        "During torch.compile, encountered a mutation on a view chain of length ", metas.size(), ", where view ", i,
        " was an as_strided() call. as_strided() is non-compositional, and therefore is not possible to functionalize properly today, "
        "so this behavior is banned in compile. As a workaround, you can either remove the mutation from the model code, or you ");
    }
  }
}
// can insert a graph break right before the mutation with torch._dynamo.graph_break(). If you would like this behavior to 
// work properly, please comment on https://github.com/pytorch/pytorch/issues/104505.");

} // 此处为匿名命名空间的结束标记

namespace at::functionalization {

// 实现 FunctionalStorageImpl 类的 apply_updates 方法
bool FunctionalStorageImpl::apply_updates() {
  // 注意：这个函数中不应该再使用 FunctionalTensorWrappers 。
  // 目前我们需要 TLS 排除保护的唯一原因是 functorch 的 DynamicLayer 堆栈。
  // 在重新分派到功能化内核之前，它会将 Functionalize 关键字添加到 TLS 中，
  // 这意味着我们在进行传递下面的任何其他工作之前需要明确排除它。
  at::AutoDispatchSkipFunctionalize guard;
  
  // 检查是否有更新数据
  bool any_updates = !updates_.empty();
  
  // 对每个更新的数据执行更新操作
  for (auto& update_data: updates_) {
    base_ = apply_update(update_data, base_);
  }
  
  // 清空更新列表
  updates_.clear();
  
  // 返回是否有更新
  return any_updates;
}

} // namespace at::functionalization
```