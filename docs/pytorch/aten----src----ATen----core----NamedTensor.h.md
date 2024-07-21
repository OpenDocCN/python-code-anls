# `.\pytorch\aten\src\ATen\core\NamedTensor.h`

```
#pragma once

#include <ATen/core/Dimname.h>
#include <c10/core/TensorImpl.h>

namespace at {

class TensorBase;

// XXX: This file exists because TensorImpl is in c10, but Dimname is in ATen.
// Due to the c10/ATen library split, TensorImpl cannot depend on Dimname,
// so we have a couple of workarounds.
//
// In the long term, we'll move Dimname to c10 and everything in this file
// can be refactored out. The main blocker for that is that "c10::Symbol"
// actually exists outside of c10 and needs to be moved in.

// TensorImpl has a unique_ptr<NamedTensorMetaInterface> field.
// XXX: Ideally we would just put optional<vector<Dimname>> into TensorImpl.
//
// This class has an important invariant: there must be at least ONE
// non-wildcard
struct TORCH_API NamedTensorMeta final : public c10::NamedTensorMetaInterface {
  // This enum is to remind people that the invariant on constructors is that
  // the list of dimnames must have at least one non-wildcard
  enum HAS_NON_WILDCARD {
    HasNonWildcard
  };

  // Constructor initializing NamedTensorMeta with a list of names
  explicit NamedTensorMeta(HAS_NON_WILDCARD, DimnameList names)
    : names_(names.vec()) {
    check_invariants(); // Ensure the invariant that at least one dimname is non-wildcard
  }

  // Constructor initializing NamedTensorMeta with a moveable vector of names
  explicit NamedTensorMeta(HAS_NON_WILDCARD, std::vector<Dimname>&& names)
    : names_(std::move(names)) {
    check_invariants(); // Ensure the invariant that at least one dimname is non-wildcard
  }

  // Clone method to create a copy of NamedTensorMeta
  std::unique_ptr<c10::NamedTensorMetaInterface> clone() const override {
    return std::make_unique<NamedTensorMeta>(HasNonWildcard, names_);
  }

  // Getter for names
  DimnameList names() const { return names_; }

  // Used for an assertion in TensorImpl.h
  int64_t slow_dim() const override {
    return static_cast<int64_t>(names_.size());
  }

  // Check invariants to ensure at least one dimname is non-wildcard
  void check_invariants() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      std::any_of(names_.begin(), names_.end(), [](const Dimname& n) { return !n.isWildcard(); }));
  }

  // Setter for names, taking a list of new names
  void set_names(HAS_NON_WILDCARD, DimnameList new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    std::copy(new_names.begin(), new_names.end(), names_.begin());
    check_invariants(); // Ensure the invariant after setting new names
  }

  // Setter for names, taking a moveable vector of new names
  void set_names(HAS_NON_WILDCARD, std::vector<Dimname>&& new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    names_ = std::move(new_names);
    check_invariants(); // Ensure the invariant after setting new names
  }

  // INVARIANT: at least one Dimname is non-WILDCARD
  std::vector<Dimname> names_;
};

// When NamesMode is disabled, then all operations ignore tensors' names fields.
// Concretely speaking, all tensors are treated as having nullopt names.
struct TORCH_API NamesMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables names upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API NoNamesGuard {
  NoNamesGuard() : prev_mode(NamesMode::is_enabled()) {
    NamesMode::set_enabled(false); // Disable names upon construction
  }

  ~NoNamesGuard() {
    if (initialized) {
      reset(); // Reset to original names mode upon destruction
    }
  }

  // Reset to the previous names mode
  void reset() {
    TORCH_INTERNAL_ASSERT(initialized);
    NamesMode::set_enabled(prev_mode);
  }

  bool prev_mode; // Previous names mode
  bool initialized; // Flag indicating if the guard is initialized
};

} // namespace at
    // 调用 NamesMode 的静态方法 set_enabled，设置为之前保存的 prev_mode 值
    NamesMode::set_enabled(prev_mode);
  }
 private:
  // 声明一个私有成员变量 prev_mode，用于存储前一个模式的状态
  bool prev_mode;
  // 声明一个私有成员变量 initialized，初始化为 true，表示对象已初始化
  bool initialized{true};
};

// 声明函数，用于检查给定张量的维度名称列表是否有效
void check_names_valid_for(const TensorBase& tensor, DimnameList names);

// 声明函数，用于检查给定张量维度数和维度名称列表是否有效
void check_names_valid_for(size_t tensor_dim, DimnameList names);

// 在张量中设置名称为给定名称列表，返回设置后的张量
TORCH_API const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::optional<DimnameList> names);

// 在张量中设置名称为给定的名称向量，返回设置后的张量，并可选择验证名称的有效性
TORCH_API const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::vector<Dimname>&& names, bool validate_names);

// 定义最大命名张量维度为 64
constexpr size_t kMaxNamedTensorDim = 64;

// 返回一个长度为 len 的默认名称列表
DimnameList default_names(size_t len);

namespace impl {

// 在 TensorImpl 上进行一些辅助函数，用于处理 TH 中的名称
// XXX: 理想情况下，这些函数应该作为 TensorImpl 的方法存在
TORCH_API void internal_set_names_inplace(TensorImpl* impl, std::optional<DimnameList> names, bool validate_names);

// 在 TensorImpl 上设置名称为给定的名称向量，可选择验证名称的有效性
TORCH_API void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names);

// 检查给定张量实现的维度名称列表是否有效
void check_names_valid_for(TensorImpl* impl, DimnameList names);

// 返回张量是否具有名称且不全为 'None'
// 若张量的名称不存在（未分配）或全部为 'None'，则返回 false
// 我们将未分配名称的张量与所有名称均为 'None' 的张量视为相同
TORCH_API bool has_names(const TensorImpl* impl);

// 返回张量维度的名称列表
// 对于未命名张量，视为所有维度均为 'None'
TORCH_API DimnameList get_names(const TensorImpl* impl);

// 这更像是实现细节；应尽可能使用 impl::get_names / Tensor::names()，因为它提供了更清晰的 API
// 如果张量的名称已分配，则返回名称列表；否则返回 nullopt
// 如果张量是以 names=None 构造的，则名称未分配
TORCH_API std::optional<DimnameList> get_opt_names(const TensorImpl* impl);

} // namespace impl

} // namespace at
```