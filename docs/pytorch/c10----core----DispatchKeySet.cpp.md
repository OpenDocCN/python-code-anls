# `.\pytorch\c10\core\DispatchKeySet.cpp`

```
// 引入C++头文件，包括DispatchKeySet和irange
#include <c10/core/DispatchKeySet.h>
#include <c10/util/irange.h>

// 定义命名空间c10
namespace c10 {

// backend_dispatch_keyset包含映射到后端的所有调度键。
// 别名键DispatchKey::CompositeExplicitAutograd映射到backend_dispatch_keyset
constexpr DispatchKeySet backend_dispatch_keyset =
    autogradother_backends | DispatchKeySet(DispatchKey::Dense);

// 查看注释[CompositeExplicitAutogradNonFunctional Key]
// 在aten中有几种分解类型，每种类型都有自己的别名键。
// 如果您的分解满足以下条件之一，应将其注册到`CompositeExplicitAutogradNonFunctional key`：
// (1) 它是一个就地操作
// (2) 它分解为一个以上的变异操作
// (3) 它具有导数公式
// 这个键对于"functional"后端（如LazyTensor / XLA）很重要。
// 如果您的后端只处理"functional ops"，则不希望将一个"functional op"分解为引起别名的操作。
// 相反，应直接为该"functional op"编写一个内核！
constexpr DispatchKeySet non_functional_backend_dispatch_keyset =
    backend_dispatch_keyset
        // XLA和LazyTensor目前是在eager模式下使用功能化传递的核心中唯二的两个后端。
        .remove(DispatchKey::Sparse)
        .remove_backend(BackendComponent::XLABit)
        .remove_backend(BackendComponent::LazyBit);

// 检查给定的调度键是否为后端调度键
bool isBackendDispatchKey(DispatchKey t) {
  return t != DispatchKey::Undefined
      // 查看注释[No Alias Keys in DispatchKeySet]
      && !isAliasDispatchKey(t)
      // 查看注释[NestedTensor Not Included in Backend Keys]
      // 由于与某些内核不兼容，显式从"backend keyset"中显式删除了NestedTensor，因此不希望将其包含在CompositeExplicitAutograd内核中。
      && t != DispatchKey::NestedTensor && backend_dispatch_keyset.has(t);
}

// math_dispatch_keyset包含backend_dispatch_keyset和autograd_dispatch_keyset中的所有键。
// 别名键DispatchKey::CompositeImplicitAutograd映射到[math_dispatch_keyset x full_backend_mask]
constexpr DispatchKeySet math_dispatch_keyset = backend_dispatch_keyset |
    autograd_dispatch_keyset |
    // 查看注释[NestedTensor Not Included in Backend Keys]
    // 虽然如此，nested_tensor是一个特例，我们希望支持composite implicit kernels但不支持explicit kernels，因此我们手动将该键添加到math_dispatch_keyset中。
    DispatchKeySet{DispatchKey::NestedTensor} |
    // Functionalize应始终重用CompositeImplicit分解。
    DispatchKeySet{DispatchKey::Functionalize};

// nested_dispatch_keyset包含AutogradNestedTensor和NestedTensor中的所有键。
constexpr DispatchKeySet nested_dispatch_keyset =
    DispatchKeySet(
        {DispatchKey::AutogradNestedTensor, DispatchKey::NestedTensor}) |
    // 创建一个 DispatchKeySet 对象，使用 RAW 类型和给定的 full_backend_mask 参数
    DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
// 返回给定调度键 t 对应的运行时调度键集合
DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  // 内部断言，确保 t 不是 DispatchKey::Undefined
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  
  // 根据不同的调度键 t，返回相应的运行时调度键集合
  switch (t) {
    case DispatchKey::Autograd:
      // 查看注释“autograd_dispatch_keyset 不包含后端位”的说明
      // 这就是为什么在这里与后端位的掩码进行 OR 操作的原因。
      // getRuntimeDispatchKeySet() 期望返回运行时调度键集合，如 AutogradCPU，但这需要具备后端位。
      return autograd_dispatch_keyset |
          DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
    case DispatchKey::CompositeImplicitAutograd:
      // 返回数学调度键集合
      return math_dispatch_keyset;
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      // 返回嵌套张量调度键集合
      return nested_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutograd:
      // 返回后端调度键集合
      return backend_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      // 返回非功能性后端调度键集合
      return non_functional_backend_dispatch_keyset;
    default:
      // 对于其他情况，返回单一的调度键 t 的集合
      return DispatchKeySet(t);
  }
}

// 检查运行时调度键集合 t 是否包含调度键 k
bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k) {
  // 内部断言，确保 t 不是 DispatchKey::Undefined
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  
  // 根据不同的调度键 t 进行检查
  switch (t) {
    case DispatchKey::Autograd:
      // 返回 autograd_dispatch_keyset 是否包含转换后的功能性调度键 k
      return autograd_dispatch_keyset.has(toFunctionalityKey(k));
    case DispatchKey::CompositeImplicitAutograd:
      // 查看注释“NestedTensor 不包含在后端键中”的说明
      // 返回 math_dispatch_keyset 是否包含调度键 k
      return math_dispatch_keyset.has(k);
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      // 查看注释“NestedTensor 不包含在后端键中”的说明
      // 返回 nested_dispatch_keyset 是否包含调度键 k
      return nested_dispatch_keyset.has(k);
    case DispatchKey::CompositeExplicitAutograd:
      // 查看注释“NestedTensor 不包含在后端键中”的说明
      // 返回 k 是否不是 DispatchKey::NestedTensor 并且 backend_dispatch_keyset 是否包含调度键 k
      return k != DispatchKey::NestedTensor && backend_dispatch_keyset.has(k);
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      // 查看注释“NestedTensor 不包含在后端键中”的说明
      // 返回 k 是否不是 DispatchKey::NestedTensor 并且 non_functional_backend_dispatch_keyset 是否包含调度键 k
      return k != DispatchKey::NestedTensor &&
          non_functional_backend_dispatch_keyset.has(k);
    case DispatchKey::FuncTorchBatchedDecomposition:
      // 返回 functorch_batched_ks 是否包含调度键 k
      return functorch_batched_ks.has(k);
    default:
      // 对于其他情况，返回 t 是否等于 k
      return t == k;
  }
}

// 对于给定的 autograd 键 t，返回与之关联的（保证非空）后端键集合。对于非 autograd 键，返回空的键集合。
DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t) {
  // 根据不同的 autograd 键 t 进行选择
  switch (t) {
    case DispatchKey::AutogradCPU:
      // 返回包含 DispatchKey::CPU 的键集合
      return DispatchKeySet(DispatchKey::CPU);
    case DispatchKey::AutogradCUDA:
      // 返回包含 DispatchKey::CUDA 的键集合
      return DispatchKeySet(DispatchKey::CUDA);
    case DispatchKey::AutogradXLA:
      // 返回包含 DispatchKey::XLA 的键集合
      return DispatchKeySet(DispatchKey::XLA);
    case DispatchKey::AutogradLazy:
      // 返回包含 DispatchKey::Lazy 的键集合
      return DispatchKeySet(DispatchKey::Lazy);
    case DispatchKey::AutogradMeta:
      // 返回包含 DispatchKey::Meta 的键集合
      return DispatchKeySet(DispatchKey::Meta);
    case DispatchKey::AutogradMPS:
      // 返回包含 DispatchKey::MPS 的键集合
      return DispatchKeySet(DispatchKey::MPS);
    case DispatchKey::AutogradHPU:
      // 返回包含 DispatchKey::HPU 的键集合
      return DispatchKeySet(DispatchKey::HPU);
    case DispatchKey::AutogradIPU:
      // 返回包含 DispatchKey::IPU 的键集合
      return DispatchKeySet(DispatchKey::IPU);
    default:
      // 对于其他情况，返回空的键集合
      return DispatchKeySet();
  }
}
    // 对于 AutogradXPU 分发键，返回包含 XPU 分发键的集合
    case DispatchKey::AutogradXPU:
      return DispatchKeySet(DispatchKey::XPU);
    
    // 对于 AutogradPrivateUse1 分发键，返回包含 PrivateUse1 分发键的集合
    case DispatchKey::AutogradPrivateUse1:
      return DispatchKeySet(DispatchKey::PrivateUse1);
    
    // 对于 AutogradPrivateUse2 分发键，返回包含 PrivateUse2 分发键的集合
    case DispatchKey::AutogradPrivateUse2:
      return DispatchKeySet(DispatchKey::PrivateUse2);
    
    // 对于 AutogradPrivateUse3 分发键，返回包含 PrivateUse3 分发键的集合
    case DispatchKey::AutogradPrivateUse3:
      return DispatchKeySet(DispatchKey::PrivateUse3);
    
    // 对于 AutogradNestedTensor 分发键，返回包含 NestedTensor 分发键和 RAW 分发键的集合
    case DispatchKey::AutogradNestedTensor:
      return DispatchKeySet(DispatchKey::NestedTensor) |
          DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
    
    // 对于 AutogradOther 分发键，返回 autogradother_backends 集合
    case DispatchKey::AutogradOther:
      return autogradother_backends;
    
    // 默认情况下，返回一个空的 DispatchKeySet 对象
    default:
      return DispatchKeySet();
  }
}

// 检查指定的 DispatchKey 是否包含在别名中
bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  // 如果 k 不是 Undefined，并且 alias 包含 k，则返回 true
  return k != DispatchKey::Undefined && runtimeDispatchKeySetHas(alias, k);
}

// 将 DispatchKeySet 转换为字符串表示
std::string toString(DispatchKeySet ts) {
  // 使用 stringstream 构建字符串流
  std::stringstream ss;
  ss << ts;  // 将 DispatchKeySet 输出到 stringstream 中
  return ss.str();  // 返回 stringstream 转换为的字符串
}

// 重载操作符 <<，用于将 DispatchKeySet 输出到 ostream 中
std::ostream& operator<<(std::ostream& os, DispatchKeySet ts) {
  if (ts.empty()) {
    os << "DispatchKeySet()";  // 如果 DispatchKeySet 为空，输出空集合表示
    return os;
  }
  os << "DispatchKeySet(";  // 开始输出 DispatchKeySet
  bool first = true;
  for (auto k : ts) {
    if (!first) {
      os << ", ";  // 输出逗号分隔符
    }
    os << k;  // 输出单个 DispatchKey
    first = false;
  }
  os << ")";  // 输出结束符
  return os;
}

// DispatchKeySet::iterator 的前置递增操作符重载
DispatchKeySet::iterator& DispatchKeySet::iterator::operator++() {
  // 内部断言，确保下一个功能性位和后端位在有效范围内
  TORCH_INTERNAL_ASSERT(next_functionality_ <= iterator::end_iter_mask_val);
  TORCH_INTERNAL_ASSERT(next_backend_ <= num_backends, next_backend_);

  // 创建一个掩码版本的集合表示，忽略已遍历过的键
  uint64_t masked_functionality_bits =
      llvm::maskTrailingZeros<uint64_t>(next_functionality_) & *data_ptr_;
  uint64_t masked_backend_bits =
      llvm::maskTrailingZeros<uint64_t>(next_backend_) & full_backend_mask &
      *data_ptr_;

  uint64_t first_functionality_idx =
      llvm::findFirstSet(masked_functionality_bits);
  uint64_t first_backendcomponent_idx = llvm::findFirstSet(masked_backend_bits);

  // 如果没有键，设置为结束迭代器值
  if (first_functionality_idx == std::numeric_limits<uint64_t>::max() ||
      next_functionality_ == iterator::end_iter_mask_val) {
    // 设置状态为 end() 的状态
    next_functionality_ = iterator::end_iter_mask_val;
    current_dispatchkey_idx_ = iterator::end_iter_key_val;
    next_backend_ = 0;
    current_backendcomponent_idx_ = iterator::end_iter_key_val;
    return *this;
  }

  // 加上 1 是因为 DispatchKey::Undefined 和 BackendComponent::InvalidBit
  auto new_next_functionality = first_functionality_idx + 1;
  auto new_backendcomponent_idx = first_backendcomponent_idx + 1;
  // 减去 num_backends 是因为前 num_backends 位不是 Dispatch Keys
  auto next_dispatchkey_idx = new_next_functionality - num_backends;

  // 如果当前功能性位是每个后端的位，则需要特殊处理
  if (isPerBackendFunctionalityKey(
          static_cast<DispatchKey>(next_dispatchkey_idx))) {
    // 情况1：如果当前后端未定义，则没有此功能性键的有效后端实例，可以跳过它
    if (first_backendcomponent_idx == std::numeric_limits<uint64_t>::max()) {
      // 增加功能性掩码以跳过当前功能性位于下一个增量时
      next_functionality_ = new_next_functionality;
      ++(*this);
      return *this;
    }

    // 否则，在此时我们知道当前后端和功能性位是什么
    current_dispatchkey_idx_ = next_dispatchkey_idx;
    current_backendcomponent_idx_ = new_backendcomponent_idx;
    //...
    // 接下来，我们需要设置下一个增量的掩码。
    uint64_t next_backendcomponent_bits =
        llvm::maskTrailingZeros<uint64_t>(first_backendcomponent_idx + 1) &
        full_backend_mask & *data_ptr_;
    // 找到下一个后端组件的索引，该索引位于掩码中第一个被设置的位置
    uint64_t next_backendcomponent_idx =
        llvm::findFirstSet(next_backendcomponent_bits);
    // 如果找不到下一个后端组件索引，说明当前后端是有效的，但是在键集中没有另一个后端。
    // 在这种情况下，我们需要提升功能掩码，并为下一个增量重置后端掩码。
    if (next_backendcomponent_idx == std::numeric_limits<uint64_t>::max()) {
      next_functionality_ = new_next_functionality;
      next_backend_ = 0;
    } else {
      // 如果找到了下一个后端组件索引，我们将在下一次迭代时继续相同的功能位，但不同的后端位。
      next_backend_ = first_backendcomponent_idx + 1;
    }
  } else {
    // 不涉及每个后端的功能位更简单处理。我们可以忽略后端位。
    TORCH_INTERNAL_ASSERT(next_backend_ == 0);
    current_dispatchkey_idx_ = next_dispatchkey_idx;
    next_functionality_ = new_next_functionality;
  }
  // 返回当前对象的引用
  return *this;
}

// 定义函数 initializeFunctionalityOffsetsAndMasks，返回一个包含功能键偏移量和掩码的数组
std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks() {
  // 创建一个长度为 num_functionality_keys 的数组 offsets_and_masks
  std::array<FunctionalityOffsetAndMask, num_functionality_keys> offsets_and_masks;
  // 手动设置第一个条目，对应于 Undefined 功能键
  offsets_and_masks[0] = FunctionalityOffsetAndMask(0, 0);
  
  // 循环遍历每个功能键（除了 Undefined）
  for (const auto functionality_idx : c10::irange(1, num_functionality_keys)) {
    // 获取前一个功能键的偏移量和掩码
    auto prev_offset_and_mask = offsets_and_masks[functionality_idx - 1];
    auto k = static_cast<DispatchKey>(functionality_idx);
    
    // 如果前一个功能不是针对每个后端的，那么可以简单地增加前一个偏移量
    // 否则，下一个偏移量 = 前一个偏移量 + num_backends
    auto next_offset = prev_offset_and_mask.offset +
        (prev_offset_and_mask.mask == 0 ? 1 : num_backends);
    
    // 控制在运行时索引计算中使用的掩码，以找到后端的偏移量。
    // 对于非每个后端的功能，该偏移量应始终为 0。否则，我们需要使用后端掩码来获取后端的索引。
    auto next_mask = isPerBackendFunctionalityKey(k) ? full_backend_mask : 0;
    
    // 将计算得到的下一个偏移量和掩码存入数组中
    offsets_and_masks[functionality_idx] =
        FunctionalityOffsetAndMask(next_offset, next_mask);
  }
  
  // 对最后一个功能键的计算偏移量进行合理性检查
  TORCH_INTERNAL_ASSERT(
      offsets_and_masks[num_functionality_keys - 1].offset ==
          (num_runtime_entries - 1),
      "num_runtime_entries: ",
      num_runtime_entries,
      "last_offset: ",
      offsets_and_masks[num_functionality_keys - 1].offset);
  
  // 返回功能键偏移量和掩码的数组
  return offsets_and_masks;
}

// 命名空间 c10 结束
} // namespace c10
```