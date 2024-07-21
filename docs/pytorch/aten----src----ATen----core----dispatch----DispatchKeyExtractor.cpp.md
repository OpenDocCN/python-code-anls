# `.\pytorch\aten\src\ATen\core\dispatch\DispatchKeyExtractor.cpp`

```
// (1) 更新 nonFallthroughKeys_
void DispatchKeyExtractor::setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough) {
  if (has_fallthrough) {
    // 如果有 fallthrough，从 nonFallthroughKeys_ 中移除键 k
    nonFallthroughKeys_ = nonFallthroughKeys_.remove(k);
  } else {
    // 如果没有 fallthrough，向 nonFallthroughKeys_ 添加键 k
    nonFallthroughKeys_ = nonFallthroughKeys_.add(k);
  }

  // (2) 更新 nonFallthroughKeysPerBackend_
  if (isPerBackendFunctionalityKey(toFunctionalityKey(k))) {
    // 如果这是一个每个后端功能键
    // 我们需要确定当前后端是什么，
    // 并仅更新该后端的位集。
    // 减去 1 是因为第一个后端应该有索引 0（CPU），
    // 但是枚举从 BackendComponent::InvalidBit 开始。
    auto backend_idx = static_cast<uint8_t>(toBackendComponent(k)) - 1;
    TORCH_INTERNAL_ASSERT(backend_idx >= 0 && static_cast<uint8_t>(backend_idx) < nonFallthroughKeysPerBackend_.size());

    if (has_fallthrough) {
      // 如果有 fallthrough，从相应后端的 nonFallthroughKeysPerBackend_ 中移除键 k
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].remove(k);
    } else {
      // 如果没有 fallthrough，向相应后端的 nonFallthroughKeysPerBackend_ 添加键 k
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].add(k);
    }

    // 根据情况设置 requiresBitsetPerBackend_
    for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size() - 1)) {
      if (nonFallthroughKeysPerBackend_[i] != nonFallthroughKeysPerBackend_[i+1]) {
        requiresBitsetPerBackend_ = true;
        return;
      }
    }
    requiresBitsetPerBackend_ = false;
    return;
  } else {
    // 否则，如果为不是每个后端的功能，则对所有后端更新 fallthrough 位集。
    // TODO: 我们可能可以通过仅在第一次看到 requiresBitsetPerBackend_ = true 时才懒惰地更新这些值来进行优化
    // （这几乎永远不会发生）
    if (has_fallthrough) {
      for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
        nonFallthroughKeysPerBackend_[i] = nonFallthroughKeysPerBackend_[i].remove(k);
      }
    } else {
      for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
        nonFallthroughKeysPerBackend_[i] = nonFallthroughKeysPerBackend_[i].add(k);
      }
    }
  }
}

// 返回对象状态的字符串表示，包括 dispatch_arg_indices_reverse_ 和 nonFallthroughKeys_
std::string DispatchKeyExtractor::dumpState() const {
  std::ostringstream oss;
  for (const auto i : c10::irange(c10::utils::bitset::NUM_BITS())) {
    if (dispatch_arg_indices_reverse_.get(i)) {
      oss << "1";
    } else {
      oss << "0";
    }
  }
  oss << " " << nonFallthroughKeys_ << "\n";
  return oss.str();
}

// 检查函数模式的不变性，确保生成的位集与 dispatch_arg_indices_reverse_ 相等
void DispatchKeyExtractor::checkInvariants(const FunctionSchema& schema) const {
  TORCH_INTERNAL_ASSERT(makeBitsetForDispatchArgs(schema) == dispatch_arg_indices_reverse_);
}
```