# `.\pytorch\c10\core\TensorImpl.h`

```py
// 预处理命令，确保头文件只被包含一次
#pragma once

// 包含 C10 库的相关头文件
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymbolicShapeMeta.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/PyObjectSlot.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <c10/util/Optional.h>
#include <c10/util/accumulate.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>
#include <c10/util/typeid.h>

// 包含标准库头文件
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// 全局布尔变量，控制当 Tensor 缩小到较小尺寸时是否释放内存
// Tensor 总是会保留其最大容量所分配的内存，直到进一步调整大小
//
// 该参数适用于调用 Resize() 的方法（如 CopyFrom、ResizeLike）；
// 不适用于 Tensor::resize_ 或 ShrinkTo，后者保证永不释放内存。
C10_DECLARE_bool(caffe2_keep_on_shrink);

// 在同一运行中，由于不同输入的 Blob 分配的内存可能会有很大变化，
// 当遵循 caffe2_keep_on_shrink 时，仅当内存增益大于此标志（以字节为单位）时才会收缩 Blob。
C10_DECLARE_int64(caffe2_max_keep_on_shrink_memory);

// 命名空间 at 中声明 Tensor 和 TensorBase 类
namespace at {
class Tensor;
class TensorBase;
} // namespace at

// 命名空间 c10
namespace c10 {

/**
 * 将 vector<int> 转换为 vector<int64_t> 的实用函数。
 */
inline std::vector<int64_t> ToVectorint64_t(const ArrayRef<int>& src) {
  return std::vector<int64_t>(src.begin(), src.end());
}

/**
 * 返回从 k 开始的所有维度的乘积。
 */
inline int64_t size_from_dim_(int k, IntArrayRef dims) {
  int64_t r = 1;
  for (const auto i : c10::irange(k, dims.size())) {
    r *= dims[i];
  }
  return r;
}

/**
 * 返回从 0 到 k（不包括 dims[k]）的所有维度的乘积。
 */
inline int64_t size_to_dim_(int k, IntArrayRef dims) {
  // 检查 k 是否在有效范围内
  TORCH_CHECK(k >= 0 && static_cast<size_t>(k) <= dims.size());
  int64_t r = 1;
  for (const auto i : c10::irange(k)) {
    r *= dims[i];
  }
  return r;
}

// 返回从 k 到 l（不包括 dims[k] 和 dims[l]）的所有维度的乘积


这段代码主要是 C++ 头文件，定义了一些常量、类和函数，以及一些预处理命令，用于处理 Tensor 相关的操作和数据结构。
// 计算在给定维度数组中，从索引 k 到索引 l 之间的尺寸乘积
inline int64_t size_between_dim_(int k, int l, IntArrayRef dims) {
  // 检查 l 和 k 是否在维度数组的有效范围内
  TORCH_CHECK((unsigned)l < dims.size() && (unsigned)k < dims.size());
  int64_t r = 1;
  // 根据 k 和 l 的大小关系，计算乘积
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

// 将负的轴索引转换为正数索引，例如 -1 表示最后一个维度
inline int canonical_axis_index_(int axis_index, int ndims) {
  // 检查 axis_index 是否在有效范围内
  TORCH_CHECK(axis_index >= -ndims);
  TORCH_CHECK(axis_index < ndims);
  // 如果 axis_index 是负数，转换为正数索引
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

/*
 * 一个上下文，用于在析构期间调用额外的放置析构函数。
 *
 * 接受一个已经构造的 DataPtr，并将其存储为成员变量，在 DataPtr 析构之前，
 * 我们将在其底层数据指针上调用额外的析构函数。
 * `data_ptr_` 拥有内存。
 */
struct C10_API PlacementDeleteContext {
  DataPtr data_ptr_;              // 存储数据指针
  PlacementDtor placement_dtor_;  // 放置析构函数指针
  size_t size_;                   // 数据大小

  // 构造函数，接受 DataPtr、放置析构函数和数据大小
  PlacementDeleteContext(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size)
      : data_ptr_(std::move(data_ptr)),
        placement_dtor_(placement_dtor),
        size_(size) {}

  // 创建一个包含放置析构函数的 DataPtr
  static DataPtr makeDataPtr(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size,
      Device device);

  // 析构函数，释放 data_ptr_ 指向的内存
  ~PlacementDeleteContext() {
    placement_dtor_(data_ptr_.get(), size_);
    // 当 data_ptr_ 被析构时，原始内存将被释放
  }
};

// 虚拟类 AutogradMetaInterface，定义了自动求导元接口
struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(
      bool requires_grad,
      at::TensorImpl* self_impl) = 0;  // 设置是否需要梯度
  virtual bool requires_grad() const = 0;  // 返回是否需要梯度
  virtual at::Tensor& mutable_grad() = 0;  // 返回可变梯度引用
  virtual const at::Tensor& grad() const = 0;  // 返回梯度引用
  virtual const at::Tensor& fw_grad(uint64_t level, const at::TensorBase& self)
      const = 0;  // 返回前向梯度引用
  virtual void set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op) = 0;  // 设置前向梯度
  virtual ~AutogradMetaInterface();  // 虚析构函数
};

namespace impl {

// AutogradMetaFactory 的实现，用于构造 AutogradMetaInterface
struct C10_API AutogradMetaFactory {
  virtual ~AutogradMetaFactory() = default;
  virtual std::unique_ptr<AutogradMetaInterface> make() const = 0;  // 构造 AutogradMetaInterface
  virtual const at::Tensor& undefined_tensor() const = 0;  // 返回未定义的 Tensor 引用
};

// 设置 AutogradMetaFactory
C10_API void SetAutogradMetaFactory(AutogradMetaFactory* factory);
// 获取 AutogradMetaFactory
C10_API AutogradMetaFactory* GetAutogradMetaFactory();
/**
 * Registerer class for AutogradMetaFactory instances.
 * Registers a given AutogradMetaFactory instance upon initialization.
 **/
struct C10_API AutogradMetaFactoryRegisterer {
  explicit AutogradMetaFactoryRegisterer(AutogradMetaFactory* factory) {
    SetAutogradMetaFactory(factory);
  }
};

} // namespace impl

/**
 * Interface for named tensor metadata.
 * Defines virtual functions for operations related to named tensors.
 **/
struct C10_API NamedTensorMetaInterface {
  virtual ~NamedTensorMetaInterface() = default;
  /**
   * Clone function for creating a copy of the NamedTensorMetaInterface object.
   * Throws an assertion error indicating the function is not implemented.
   **/
  virtual std::unique_ptr<NamedTensorMetaInterface> clone() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::clone");
  };
  /**
   * Returns the dimension size of the named tensor.
   * Throws an assertion error indicating the function is not implemented.
   **/
  virtual int64_t slow_dim() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::slow_dim");
  };
};

// For ease of copy pasting
#if 0
is_contiguous
is_channels_last_contiguous
is_channels_last_3d_contiguous
is_channels_last
is_channels_last_3d
is_non_overlapping_and_dense
#endif

/**
 * BackendMeta structure holding additional metadata for a specific device backend.
 * Inherits from intrusive_ptr_target for intrusive pointer management.
 **/
struct C10_API BackendMeta : intrusive_ptr_target {
  /**
   * Destructor for BackendMeta, default implementation.
   **/
  ~BackendMeta() override = default;
  /**
   * Clones the BackendMeta object.
   * Returns the input ptr as a cloned intrusive pointer.
   **/
  virtual intrusive_ptr<BackendMeta> clone(
      const intrusive_ptr<BackendMeta>& ptr) const {
    return ptr;
  }
};

/**
 * ExtraMeta structure holding additional metadata.
 * Includes symbolic shape metadata, named tensor metadata, backend metadata,
 * and optional custom error messages.
 **/
struct C10_API ExtraMeta {
  std::unique_ptr<c10::SymbolicShapeMeta> symbolic_shape_meta_ = nullptr;
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;
  intrusive_ptr<c10::BackendMeta> backend_meta_ = nullptr;
  std::optional<std::string> custom_data_ptr_error_msg_ = c10::nullopt;
  std::optional<std::string> custom_storage_error_msg_ = c10::nullopt;

  /**
   * Default constructor for ExtraMeta.
   **/
  ExtraMeta() = default;

  /**
   * Copy constructor for ExtraMeta.
   * Copies each field from another ExtraMeta instance if present.
   **/
  ExtraMeta(const ExtraMeta& other) {
    if (other.symbolic_shape_meta_) {
      symbolic_shape_meta_ =
          std::make_unique<c10::SymbolicShapeMeta>(*other.symbolic_shape_meta_);
    }
    if (other.named_tensor_meta_) {
      named_tensor_meta_ = other.named_tensor_meta_->clone();
    }
    if (other.backend_meta_) {
      backend_meta_ = other.backend_meta_->clone(other.backend_meta_);
    }
    if (other.custom_data_ptr_error_msg_) {
      custom_data_ptr_error_msg_ = other.custom_data_ptr_error_msg_;
    }
    if (other.custom_storage_error_msg_) {
      custom_storage_error_msg_ = other.custom_storage_error_msg_;
    }
  }

  /**
   * Constructor for ExtraMeta.
   * Initializes ExtraMeta with provided metadata and optional error messages.
   **/
  ExtraMeta(
      std::unique_ptr<c10::SymbolicShapeMeta> symbolic_shape_meta,
      std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta,
      intrusive_ptr<c10::BackendMeta> backend_meta,
      std::optional<std::string> custom_data_ptr_error_msg = c10::nullopt,
      std::optional<std::string> custom_storage_access_error_msg = c10::nullopt)
      : symbolic_shape_meta_(std::move(symbolic_shape_meta)),
        named_tensor_meta_(std::move(named_tensor_meta)),
        backend_meta_(std::move(backend_meta)),
        custom_data_ptr_error_msg_(std::move(custom_data_ptr_error_msg)),
        custom_storage_error_msg_(std::move(custom_storage_access_error_msg)) {}

  /**
   * Clone function for creating a copy of the ExtraMeta object.
   * Returns a unique_ptr to the newly created ExtraMeta instance.
   **/
  std::unique_ptr<ExtraMeta> clone() const {
    return std::make_unique<ExtraMeta>(*this);
  }
};

// NOTE [ Version Counter Sharing ]
//
// 每个张量都有一个版本计数器。当张量通过就地变量操作更改数据或大小时，版本计数器会递增。
// 版本计数器用于检测对保存的变量进行的修改，这些修改可能导致不正确的梯度计算。版本计数器可以在变量之间共享：
//
// 1. 视图共享基本变量的版本计数器，
// 2. `x.detach()` 共享 `x` 的版本计数器，
// 3. 拆包的保存变量共享源的版本计数器。
//
// 在以下情况下，版本计数器不会共享：
//
// 1. 通过调用 `set_data(...)` 替换 `Variable` 的底层 `Tensor` 时，
// 2. `x.data` 不会与 `x` 的版本计数器共享。（见 https://github.com/pytorch/pytorch/issues/5396 中的讨论）
//
// 问题：为什么我们将版本计数器放在 TensorImpl 而不是 AutogradMeta 中？
//
// 回答：在变量/张量合并后，当其 `requires_grad_` 为 false 时，张量将不再具有 AutogradMeta，
// 但当我们在需要保存此张量以进行反向传播的函数的前向传递中使用此张量时，我们需要跟踪此张量的版本，
// 以确保在 autograd 图中始终有效。
//
// 为了实现这个目标，我们将版本计数器放在 TensorImpl 而不是 AutogradMeta 中，并始终可用。
// 这允许我们在张量不需要梯度时不携带 AutogradMeta 的优化。
//
// 一个假设的替代方法是，在保存张量以进行反向传播时，对非需要梯度的张量进行 AutogradMeta 的延迟初始化。
// 然而，由于在前向传递中保存张量发生在前向传递时，而我们的不变量是前向传递需要是线程安全的，延迟初始化
// AutogradMeta 可能会在多线程场景中引入竞态条件，从而使得前向传递不再是线程安全的，这违反了不变量。
//
// 结构体 `VariableVersion` 的私有部分，包含一个嵌套结构 `VersionCounter`，
// 它继承自 `intrusive_ptr_target`，用于表示版本计数器。
struct C10_API VariableVersion {
 private:
  struct VersionCounter : intrusive_ptr_target {
    VersionCounter(uint32_t version) : version_(version) {}
    std::atomic<uint32_t> version_;
  };
  c10::intrusive_ptr<VersionCounter> version_counter_;

 public:
  // Note [Disabled VariableVersion]
  // VariableVersion结构具有一个指向VersionCounter结构的intrusive_ptr，
  // 后者包含一个原子变量。因此，使用`VariableVersion(/*version=*/0)`并不像我们预期的那样廉价。
  // 在某些情况下，使用版本号为0构造VariableVersion并不是必要的，因此我们添加了一个廉价的构造函数，
  // 它不会分配intrusive_ptr。
  // 示例用例包括：
  //  - 推断张量不跟踪版本计数器，因此它们始终具有禁用的VariableVersion。
  //  - 在SavedVariable类中，我们在其构造函数中重写version_counter_，
  //    以便我们可以在那里使用廉价的构造函数。
  enum Disabled { DISABLED };
  // 即使对于不启用版本计数器的推断张量，返回true也是可以的。
  // 我们在这里要宽松一些，因为在许多情况下（例如make_variable），如果没有其他使用，
  // 我们可以std::move一个TensorImpl，从而节省额外的TensorImpl分配。
  bool unique() const {
    return version_counter_ ? 1 == version_counter_.use_count() : true;
  }
  // 注意：在C++11和14中，默认构造std::atomic变量会使其处于持久未定义状态。参见
  // https://cplusplus.github.io/LWG/issue2334。
  VariableVersion(uint32_t version)
      : version_counter_(c10::make_intrusive<VersionCounter>(version)) {}
  VariableVersion(Disabled = DISABLED) {}

  bool enabled() const {
    return version_counter_;
  }

  // Note [Inplace update inference tensor]
  // 1. 在正常模式下禁止对推断张量进行原地更新。
  //    例如：
  //      inference_tensor.copy_(normal_tensor_requires_grad)
  //    这种原地操作会使推断张量具有requires_grad=True并且有一个grad_fn。
  //    这是不好的，因为在推断模式下创建的`inference_tensor`的视图将无法知道grad_fn，
  //    因为它们的ViewMeta未记录。为了与NoGradMode行为匹配，“对在推断模式下创建的视图进行原地更新会导致错误”，
  //    我们禁止了对推断张量的原地更新，因为我们无法确定推断张量是否是在推断模式下创建的视图。
  //
  //    注意，在推断模式下创建的普通张量的视图具有正确的ViewMeta，因此它们能够正确地了解grad_fn。
  //
  // 2. 推断张量内部的原地更新不会增加版本计数器。
  //    * 要么通过跳过ADInplaceOrView内核而不调用bump()，
  //      - 例如：inference_tensor.add_(1)
  //    * 要么对于推断张量，bump()是一个空操作。
  //      - 例如：inference_tensor.add_(normal_tensor)
  void bump() {
    // TODO: 一旦文档可用，替换链接到文档的链接。
    // 检查是否存在版本计数器或推断模式是否启用，否则不允许对推断张量进行原地更新
    // 在进行原地更新之前，可以克隆张量以获取普通张量
    // 更多详情请参见 https://github.com/pytorch/rfcs/pull/17
    TORCH_CHECK(
        version_counter_ || InferenceMode::is_enabled(),
        "Inplace update to inference tensor outside InferenceMode is not allowed."
        "You can make a clone to get a normal tensor before doing inplace update."
        "See https://github.com/pytorch/rfcs/pull/17 for more details.");
    
    // 如果存在版本计数器，增加版本号
    if (version_counter_) {
      ++version_counter_->version_;
    }
  }

  // 设置版本号
  void set_version(int64_t i) {
    // 检查是否存在版本计数器，否则抛出异常
    // 在推断模式下创建的张量不具备版本计数器
    TORCH_CHECK(
        version_counter_,
        "Tried to call torch.autograd._unsafe_set_version() on a tensor "
        "that does not have a version counter. Was it created in inference mode?");
    
    // 检查版本号是否非负，否则抛出异常
    TORCH_CHECK(i >= 0, "Cannot set a version_counter to a value below 0: ", i);
    
    // 设置版本计数器的版本号为指定值
    version_counter_->version_ = i;
  }

  // 获取当前版本号
  uint32_t current_version() const {
    // 检查是否存在版本计数器，推断张量不具备版本计数器
    TORCH_CHECK(
        version_counter_, "Inference tensors do not track version counter.");
    
    // 返回版本计数器的当前版本号
    return version_counter_->version_;
  }
};

// 声明 TensorImpl 的前向声明，用于 C10_TensorImpl_Size_Check_Dummy_Class 的前向声明
struct C10_API TensorImpl;

/**
 * 注意：某些 TensorImpl 方法很小，在 PyTorch 代码库中没有被覆盖，
 * 但在理论上可能需要被第三方 TensorImpl 子类覆盖。
 * 此宏允许需要最大性能且不需要这些扩展点的用户通过构建时标志禁用它们。
 * （特别地，XLA 的 XLATensorImpl 当前覆盖了这些方法，因此我们不能默认启用此标志。）
 */
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
// 如果定义了 C10_DISABLE_TENSORIMPL_EXTENSIBILITY，则将 TENSORIMPL_MAYBE_VIRTUAL 定义为空宏
#define TENSORIMPL_MAYBE_VIRTUAL
#else
// 否则，将 TENSORIMPL_MAYBE_VIRTUAL 定义为虚函数，允许被覆盖
#define TENSORIMPL_MAYBE_VIRTUAL virtual
#endif
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  // 析构函数，用于释放对象资源
  ~TensorImpl() override;
  // Note [Enum ImplType]
  // 这个枚举是临时的。在后续的重构中，我们应该考虑如何为视图张量（view tensors）专门化TensorImpl的创建。
  // 目前我们只特殊处理了key_set_，但也有可能直接共享version_counter_而不是先创建再在as_view中覆盖。
  enum ImplType { VIEW };

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   * 构造一个由给定存储支持的1维0大小张量。
   */
  TensorImpl(
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type);

  // See Note [Enum ImplType]
  // 根据Note [Enum ImplType]创建TensorImpl。
  TensorImpl(
      ImplType,
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type);

  /**
   * Construct a 1-dim 0 size tensor that doesn't have a storage.
   * 构造一个没有存储的1维0大小张量。
   */
  TensorImpl(
      DispatchKeySet,
      const caffe2::TypeMeta data_type,
      std::optional<c10::Device> device_opt);

  // Legacy constructors so I don't have to go update call sites.
  // TODO: When Variable is added, delete these constructors
  // 遗留构造函数，以免不得不更新调用位置。
  // TODO: 添加Variable后，删除这些构造函数。
  TensorImpl(
      Storage&& storage,
      DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type)
      : TensorImpl(
            std::move(storage),
            DispatchKeySet(dispatch_key),
            data_type) {}
  TensorImpl(
      DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type,
      std::optional<c10::Device> device_opt)
      : TensorImpl(DispatchKeySet(dispatch_key), data_type, device_opt) {}

 private:
  // This constructor is private, because the data_type is redundant with
  // storage.  Still, we pass it in separately because it's easier to write
  // the initializer list if we're not worried about storage being moved out
  // from under us.
  // 这个构造函数是私有的，因为data_type与storage是冗余的。
  // 但我们单独传递它是因为在写初始化列表时更容易，如果不担心storage在我们之下被移动。
  TensorImpl(
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type,
      std::optional<c10::Device>);

 public:
  // 删除复制构造函数和赋值运算符重载，禁止复制和移动操作
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  /**
   * Release (decref) storage, and any other external allocations.  This
   * override is for `intrusive_ptr_target` and is used to implement weak
   * tensors.
   * 释放存储（减少引用计数）以及任何其他外部分配的资源。
   * 这个重载用于`intrusive_ptr_target`，用于实现弱引用张量。
   */
  void release_resources() override;

 public:
  /**
   * Return the DispatchKeySet corresponding to this Tensor, specifying
   * all of the DispatchKeys that this Tensor identifies as.  This is the
   * information used to dispatch operations on this tensor.
   * 返回与此张量对应的DispatchKeySet，指定该张量标识的所有DispatchKeys。
   * 这是用于在张量上分派操作的信息。
   */
  DispatchKeySet key_set() const {
    return key_set_;
  }

 private:
  [[noreturn]] void throw_cannot_call_with_symbolic(const char* meth) const;

  // NOTE: The general recipe for customizable methods is that the fastpath
  // function (e.g., sizes()) does an unlikely policy test, and if doesn't
  // trigger, it does the fast path implementation with no checks and going
  // directly to on-TensorImpl fields.  In particular, you never need to
  // check ExtraMeta if the policy doesn't trigger, as non-trivial ExtraMeta
  // implies the policy will always match.
  //
  // The default implementations of methods are "safe": they do extra tests
  // to make sure the internal state is consistent no matter if you are
  // doing symbolic shapes or not.  If you don't want the tests, directly
  // override the custom method (e.g., custom_sizes()) to do your preferred
  // behavior.

 public:
  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  IntArrayRef sizes() const {
    // 如果符合自定义大小策略，调用自定义的尺寸获取方法
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sizes_custom();
    }
    // 否则返回默认尺寸数组的引用
    return sizes_and_strides_.sizes_arrayref();
  }

  SymIntArrayRef sym_sizes() const {
    // 如果符合自定义大小策略，调用自定义的符号尺寸获取方法
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_sizes_custom();
    }
    // 尺寸保证非负，因此可以不进行检查的转换
    return c10::fromIntArrayRefKnownNonNegative(
        sizes_and_strides_.sizes_arrayref());
  }

  IntArrayRef sizes_default() const {
    // 如果具有符号尺寸和步幅，则抛出异常，因为默认方法不支持符号形状
    if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
      throw_cannot_call_with_symbolic("sizes");
    }
    // 否则返回默认尺寸数组的引用
    return sizes_and_strides_.sizes_arrayref();
  }

  SymIntArrayRef sym_sizes_default() const {
    // 如果具有符号尺寸和步幅，则直接返回符号形状元数据中的尺寸
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().sizes_;
    } else {
      // 尺寸保证非负，因此可以不进行检查的转换
      return c10::fromIntArrayRefKnownNonNegative(sizes_default());
    }
  }

  // From https://stackoverflow.com/a/3057522/23845
  // TODO: does C++14 have a stdlib template for this?
  template <typename T>
  struct identity {
    typedef T type;
  };

  template <typename T>
  ArrayRef<T> generic_sizes() {
    return _generic_sizes(identity<T>());
  }

  ArrayRef<int64_t> _generic_sizes(identity<int64_t>) {
    return sizes();
  }
  ArrayRef<c10::SymInt> _generic_sizes(identity<c10::SymInt>) {
    return sym_sizes();
  }

  template <typename T>
  ArrayRef<T> generic_strides() {
    return _generic_strides(identity<T>());
  }

  ArrayRef<int64_t> _generic_strides(identity<int64_t>) {
    return strides();
  }
  ArrayRef<c10::SymInt> _generic_strides(identity<c10::SymInt>) {
    return sym_strides();
  }

  template <typename T>
  T generic_storage_offset() {
    return _generic_storage_offset(identity<T>());
  }

  int64_t _generic_storage_offset(identity<int64_t>) {
  }
  // 返回张量在存储中的偏移量（元素级别）
  int64_t storage_offset() const {
    // 如果使用自定义大小策略，则调用自定义实现的存储偏移量函数
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return storage_offset_custom();
    }
    // 否则返回默认的存储偏移量
    return storage_offset_;
  }

  // 返回符号化张量在存储中的偏移量
  c10::SymInt sym_storage_offset() const {
    // 如果使用自定义大小策略，则调用自定义实现的符号化存储偏移量函数
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_storage_offset_custom();
    }
    // 否则返回未检查状态的符号化偏移量
    return c10::SymInt(SymInt::UNCHECKED, storage_offset_);
  }

  // 返回默认的存储偏移量（元素级别）
  int64_t storage_offset_default() const {
    // 如果张量有符号化大小和步幅，则抛出异常，因为不能调用带符号化参数的函数
    if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
      throw_cannot_call_with_symbolic("storage_offset");
    }
    // 否则返回默认的存储偏移量
    return storage_offset_;
  }

  // 返回默认的符号化存储偏移量
  c10::SymInt sym_storage_offset_default() const {
    // 如果有符号化大小和步幅，则返回符号化形状元数据中的存储偏移量
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().storage_offset_;
    } else {
      // 否则返回未检查状态的符号化偏移量
      return c10::SymInt(SymInt::UNCHECKED, storage_offset_);
    }
  }
  }
}

/**
 * Return a reference to the strides of this tensor.  This reference remains
 * valid as long as the tensor is live and not restrided.
 */
IntArrayRef strides() const {
  // 如果符合自定义步幅策略，则返回自定义步幅
  if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
    return strides_custom();
  }
  // 否则返回默认步幅数组的引用
  return sizes_and_strides_.strides_arrayref();
}

c10::SymIntArrayRef sym_strides() const {
  // 如果符合自定义步幅策略，则返回自定义符号步幅
  if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
    return sym_strides_custom();
  }
  // 否则转换默认步幅为非负整数形式返回
  return c10::fromIntArrayRefKnownNonNegative(strides_default());
}

IntArrayRef strides_default() const {
  // 如果张量具有符号大小和步幅，则抛出异常
  if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
    throw_cannot_call_with_symbolic("strides");
  }
  // 否则返回默认步幅数组的引用
  return sizes_and_strides_.strides_arrayref();
}

c10::SymIntArrayRef sym_strides_default() const {
  // 如果张量具有符号大小和步幅，则返回符号形状元数据的步幅
  if (has_symbolic_sizes_strides_) {
    return symbolic_shape_meta().strides_;
  } else {
    // 否则转换默认步幅为非负整数形式返回
    return c10::fromIntArrayRefKnownNonNegative(strides_default());
  }
}

/**
 * Whether or not a tensor is laid out in contiguous memory.
 *
 * Tensors with non-trivial strides are not contiguous.  See
 * compute_contiguous() for the exact definition of whether or not
 * a tensor is contiguous or not.
 */
bool is_contiguous(
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
  // 如果符合自定义步幅策略，则使用自定义方法检查是否连续
  if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
    return is_contiguous_custom(memory_format);
  }
  // 否则使用默认方法检查是否连续
  return is_contiguous_default(memory_format);
}

// These are factored into separate functions in case subclasses
// want to use them
bool is_contiguous_default(at::MemoryFormat memory_format) const {
  // 如果张量具有符号大小和步幅
  if (has_symbolic_sizes_strides_) {
    // 根据内存格式检查是否通道为最后一维连续
    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return symbolic_shape_meta().is_channels_last_contiguous().guard_bool(
          __FILE__, __LINE__);
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return symbolic_shape_meta()
          .is_channels_last_3d_contiguous()
          .guard_bool(__FILE__, __LINE__);
    }
    // 否则检查是否通道优先连续
    return symbolic_shape_meta().is_contiguous().guard_bool(
        __FILE__, __LINE__);
  }

  // 如果内存格式为通道为最后一维
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    return is_channels_last_contiguous_;
  } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    return is_channels_last_3d_contiguous_;
  }
  // 否则检查是否一般连续
  return is_contiguous_;
}

bool is_strides_like_default(at::MemoryFormat memory_format) const {
  // 如果张量具有符号大小和步幅
  if (has_symbolic_sizes_strides_) {
    // 根据内存格式检查是否通道为最后一维
    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return symbolic_shape_meta().is_channels_last().guard_bool(
          __FILE__, __LINE__);
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return symbolic_shape_meta().is_channels_last_3d().guard_bool(
          __FILE__, __LINE__);
    } else {
      return false;
    }
  }
    // 如果内存格式为 ChannelsLast，则返回是否为 ChannelsLast 的标志位
    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_;
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      // 如果内存格式为 ChannelsLast3d，则返回是否为 ChannelsLast3d 的标志位
      return is_channels_last_3d_;
    } else {
      // 其他情况下返回 false
      return false;
    }
  }

  bool is_non_overlapping_and_dense_default() const {
    // 如果具有符号化的尺寸和步长，则调用符号化形状元数据的非重叠和密集检查函数
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().is_non_overlapping_and_dense().guard_bool(
          __FILE__, __LINE__);
    } else {
      // 否则返回是否为非重叠和密集的默认标志位
      return is_non_overlapping_and_dense_;
    }
  }

  // 注意：这些维度访问函数不包含 _default()，因为可以使用 sizes_default/strides_default
  /**
   * 返回某个维度的张量尺寸，必要时对维度进行包装。
   *
   * 注意：如果您知道包装是不必要的，请使用 sizes()[d]，它会更快
   */
  int64_t size(int64_t d) const {
    // 如果自定义尺寸和步长策略匹配，则调用自定义尺寸的获取函数
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return size_custom(d);
    }
    // 否则对维度进行包装并获取尺寸
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    return sizes_and_strides_.size_at_unchecked(d);
  }

  c10::SymInt sym_size(int64_t d) const {
    // 如果自定义尺寸和步长策略匹配，则调用自定义尺寸的获取函数
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_size_custom(d);
    }
    // 否则对维度进行包装并获取符号化尺寸
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    const auto sizes = this->sym_sizes();
    return sizes[d];
  }

  /**
   * 返回某个维度的张量步长，必要时对维度进行包装。
   *
   * 注意：如果您知道包装是不必要的，请使用 sizes()[d]，它会更快
   */
  int64_t stride(int64_t d) const {
    // 对维度进行包装并获取步长
    d = maybe_wrap_dim(d, dim(), false);
    // 如果自定义步长策略匹配，则返回自定义步长
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      // TODO: 提供 stride_custom，与 size_custom 对称。目前没有用户使用它；只有 NestedTensor 使用 size_custom 覆盖性
      return strides_custom()[d]; // 未检查的（maybe_wrap_dim 强制执行边界）
    }
    // 故意不调用默认，它还处理符号化情况
    return sizes_and_strides_.stride_at_unchecked(d);
  }

  enum class SizesStridesPolicy : uint8_t {
    // 默认行为，如稠密张量。
    //
    // 可覆盖：无
    Default = 0,
    // 可定制步长行为，如稀疏张量，
    // mkldnn 张量。
    //
    // 可覆盖：strides(), is_contiguous()
    CustomStrides = 1,
    // 可定制尺寸行为，如嵌套张量
    //
    // 可覆盖：strides(), is_contiguous(), sizes(), dim(), numel()
    CustomSizes = 2
  };

 protected:
  // 检查是否与给定策略匹配的内部方法
  inline bool matches_policy(SizesStridesPolicy policy) const {
    return sizes_strides_policy_ >= static_cast<uint8_t>(policy);
  }

  // 检查是否与给定策略匹配的自定义方法
  inline bool matches_custom(SizesStridesPolicy policy) const {
    return custom_sizes_strides_ >= static_cast<uint8_t>(policy);
  }

  // 检查是否与给定策略匹配的 Python 自定义方法
  inline bool matches_python_custom(SizesStridesPolicy policy) const {
    // 检查 python_custom_sizes_strides_ 是否大于等于 policy 的静态转换为 uint8_t 后的值
    auto r = python_custom_sizes_strides_ >= static_cast<uint8_t>(policy);
    // 如果 r 为真，则进行内部断言检查是否是 Python 分发
    if (r) {
      TORCH_INTERNAL_ASSERT(is_python_dispatch())
    }
    // 返回 r
    return r;
  }

  /**
   * 上述函数的定制点。sizes_strides_policy_ 必须设置才能启用这些函数。
   *
   * 注意：dim 可以单独重写，因为张量可以具有秩，但尺寸可能不明确。
   */
  // 当 sizes_strides_policy_ >= CustomStrides 时返回真
  virtual bool is_contiguous_custom(at::MemoryFormat memory_format) const;
  // 当 sizes_strides_policy_ >= CustomSizes 时返回真
  virtual bool is_strides_like_custom(at::MemoryFormat memory_format) const;
  // 返回张量是否是非重叠和稠密的定制信息
  virtual bool is_non_overlapping_and_dense_custom() const;
  // 当 sizes_strides_policy_ >= CustomSizes 时调用，用于获取指定维度的尺寸
  // TODO: 可以在此处添加对 Python 分发的支持。
  // TODO: 可以调用 aten::size.int 而不是 sizes_custom()[d] 并启用分派器。
  virtual int64_t size_custom(int64_t d) const {
    // 对维度 d 进行包装，确保在有效范围内
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    // 返回 sizes_custom() 中索引为 d 的尺寸值，未经检查（maybe_wrap_dim 强制实施边界）
    return sizes_custom()[d];
  }

  // 当 sizes_strides_policy_ >= CustomSizes 时调用，用于获取指定维度的符号化尺寸
  // TODO: 可以在此处添加对 Python 分发的支持。
  // TODO: 可以调用 aten::size.int 而不是 sym_sizes_custom()[d] 并启用分派器。
  virtual c10::SymInt sym_size_custom(int64_t d) const {
    // 对维度 d 进行包装，确保在有效范围内
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    // 返回 sym_sizes_custom() 中索引为 d 的符号化尺寸，未经检查（maybe_wrap_dim 强制实施边界）
    return sym_sizes_custom()[d];
  }

  // 获取定制尺寸数组的引用
  virtual IntArrayRef sizes_custom() const;
  // 获取定制步幅数组的引用
  virtual IntArrayRef strides_custom() const;
  // 获取定制元素数
  virtual int64_t numel_custom() const;
  // 获取定制存储偏移量
  virtual int64_t storage_offset_custom() const;
  // 获取定制张量维度
  virtual int64_t dim_custom() const;
  // 获取定制设备
  virtual Device device_custom() const;
  // 获取定制布局
  virtual Layout layout_custom() const;

  // 获取符号化定制尺寸数组的引用
  virtual c10::SymIntArrayRef sym_sizes_custom() const;
  // 获取符号化定制步幅数组的引用
  virtual c10::SymIntArrayRef sym_strides_custom() const;
  // 获取符号化定制元素数
  virtual c10::SymInt sym_numel_custom() const;
  // 获取符号化定制存储偏移量
  virtual c10::SymInt sym_storage_offset_custom() const;

 public:
  /**
   * 如果张量有存储，则为真。详见 storage() 的详情。
   */
#ifdef DEBUG
  // 如果是调试模式，允许子类检查它们的 storage_ 是否在调试构建中被设置。
  // 虚拟函数，用于检查是否存在存储空间。
  virtual
#else
  // 在非调试模式下，允许通过 TENSORIMPL_MAYBE_VIRTUAL 宏来设置虚拟性。
  TENSORIMPL_MAYBE_VIRTUAL
#endif
      bool
      has_storage() const
  // 注意：我们取消了虚拟性，因为仅仅询问子类是否有存储空间不应该报错。
  // 大多数子类以前会抛出错误，但 OpaqueTensorImpl 希望成功返回 false，
  // 因此我们将其设置为非错误。
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  {
    return storage_;
  }
#else
      ;
#endif

  /**
   * 返回 Tensor 的底层存储。多个 Tensor 可能共享单个存储。
   * Storage 是一个简化的类，类似于 Tensor，但支持的操作远不及 Tensor 多。
   *
   * 如果可能，尽量避免使用此方法；尝试仅使用 Tensor API 来执行操作。
   */
  TENSORIMPL_MAYBE_VIRTUAL const Storage& storage() const {
    if (C10_UNLIKELY(storage_access_should_throw_)) {
      throw_storage_access_error();
    }
    return storage_;
  }

  /**
   * 返回底层存储，假定这是一个基本的分步 Tensor。在访问 storage 会抛出异常的情况下，
   * 返回一个默认构造的 Storage。
   */
  inline const Storage& unsafe_storage() const {
    return storage_;
  }

  bool unique_version() const {
    return version_counter_.unique();
  }

 protected:
  virtual Layout layout_impl() const {
    TORCH_CHECK(
        false, "layout_impl is only implemented for TensorImpl subclasses.");
  }

 public:
  // 判断 Tensor 是否为稀疏 COO 格式。
  bool is_sparse() const {
    // 注意：此方法不是虚拟的，为了性能避免了分派。
    return key_set_.has_all(c10::sparse_ks);
  }

  // 判断 Tensor 是否为稀疏 CSR 格式。
  bool is_sparse_csr() const {
    return layout() == kSparseCsr;
  }

  // 判断 Tensor 是否为稀疏 CSR/CSC/BSR/BSC 格式。
  bool is_sparse_compressed() const {
    return key_set_.has_all(c10::sparse_csr_ks);
  }

  // 判断 Tensor 是否被量化。
  bool is_quantized() const {
    // 注意：此方法不是虚拟的，为了性能避免了分派。
    constexpr auto quantized_ks = DispatchKeySet(DispatchKey::Quantized);
    return key_set_.has_all(quantized_ks);
  }

  // 判断 Tensor 是否为元数据。
  bool is_meta() const {
    // 注意：此方法不是虚拟的，为了性能避免了分派。
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_meta();
    }
    return device_opt_.has_value() && device_opt_->type() == kMeta;
  }

  // 判断 Tensor 是否在 CPU 上。
  bool is_cpu() const {
    // 注意：此方法不是虚拟的，为了性能避免了分派。
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_cpu();
    }
    // 注意：我们不能依赖于分派键来确定张量的设备类型，因为“包装器”张量
    // （如 FunctionalTensorWrapper）不包括后端分派键。
  return device_opt_.has_value() && device_opt_->type() == kCPU;
}

bool is_cuda() const {
  // NB: This method is not virtual and avoid dispatches for performance
  // reasons.
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_cuda();
  }
  // 检查是否存在设备选项，并且其类型为CUDA
  return device_opt_.has_value() && device_opt_->type() == kCUDA;
}

bool is_xpu() const {
  // NB: This method is not virtual and avoid dispatches for performance
  // reasons.
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_xpu();
  }
  // 检查是否存在设备选项，并且其类型为XPU
  return device_opt_.has_value() && device_opt_->type() == kXPU;
}

bool is_ipu() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_ipu();
  }
  // 检查是否存在设备选项，并且其类型为IPU
  return device_opt_.has_value() && device_opt_->type() == kIPU;
}

bool is_xla() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_xla();
  }
  // 检查是否存在设备选项，并且其类型为XLA
  return device_opt_.has_value() && device_opt_->type() == kXLA;
}

bool is_mtia() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_mtia();
  }
  // 检查是否存在设备选项，并且其类型为MTIA
  return device_opt_.has_value() && device_opt_->type() == kMTIA;
}

bool is_hpu() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_hpu();
  }
  // 检查是否存在设备选项，并且其类型为HPU
  return device_opt_.has_value() && device_opt_->type() == kHPU;
}

bool is_lazy() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_lazy();
  }
  // 检查是否存在设备选项，并且其类型为Lazy
  return device_opt_.has_value() && device_opt_->type() == kLazy;
}

bool is_hip() const {
  // NB: This method is not virtual and avoid dispatches for performance
  // reasons.
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_hip();
  }
  // 检查是否存在设备选项，并且其类型为HIP
  return device_opt_.has_value() && device_opt_->type() == kHIP;
}

bool is_ve() const {
  // NB: This method is not virtual and avoid dispatches for performance
  // reasons.
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_ve();
  }
  // 检查是否存在设备选项，并且其类型为VE
  return device_opt_.has_value() && device_opt_->type() == kVE;
}

bool is_privateuseone() const {
  // NB: This method is not virtual and avoid dispatches for performance
  // reasons.
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_privateuseone();
  }
  // 检查是否存在设备选项，并且其类型为PrivateUse1
  return device_opt_.has_value() && device_opt_->type() == kPrivateUse1;
}

bool is_mkldnn() const {
  // 检查当前设备是否支持所有的MKLDNN键
  return key_set_.has_all(c10::mkldnn_ks);
}

bool is_vulkan() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_vulkan();
  }
  // 检查是否存在设备选项，并且其类型为Vulkan
  return device_opt_.has_value() && device_opt_->type() == kVulkan;
}

bool is_metal() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_metal();
  }
  // 检查是否存在设备选项，并且其类型为Metal
  return device_opt_.has_value() && device_opt_->type() == kMetal;
}

bool is_mps() const {
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_mps();
  }
  // 检查是否有有效的设备选项，并且设备类型为 kMPS
  return device_opt_.has_value() && device_opt_->type() == kMPS;
}

bool is_maia() const {
  // 如果设备策略存在，则返回自定义设备的 is_maia() 结果
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().is_maia();
  }
  // 否则，检查是否有有效的设备选项，并且设备类型为 kMAIA
  return device_opt_.has_value() && device_opt_->type() == kMAIA;
}

bool is_nested() const {
  // 检查 key_set_ 中是否包含 DispatchKey::NestedTensor
  return key_set_.has(DispatchKey::NestedTensor);
}

// TODO: 在不再自动启用 Autograd 分发键的情况下移除此函数
//       在 aten/src/ATen/core/boxing/impl/test_helpers.h 中仅用于测试目的
void remove_autograd_key() {
  // 从 key_set_ 中移除 autograd_dispatch_keyset 的键
  key_set_ = key_set_ - autograd_dispatch_keyset;
}

// 推断张量不具有 autograd 或 ADInplaceOrView 键
// 不变条件:
//   推断张量的 version_counter_.enabled() == false
bool is_inference() {
  bool no_ADInplaceOrView = !key_set_.has_any(c10::inplace_or_view_ks);
  bool no_Autograd = !key_set_.has_any(c10::autograd_dispatch_keyset);
  // 断言：ADInplaceOrView 和 Autograd 键必须同时开启或关闭
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      no_ADInplaceOrView == no_Autograd,
      "ADInplaceOrView and Autograd keys must be on/off at the same time.");
  // 返回是否不具有 ADInplaceOrView 和 Autograd 键
  return no_ADInplaceOrView && no_Autograd;
}

DeviceIndex get_device() const {
  // 如果设备策略存在，则返回自定义设备的索引
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom().index();
  }
  // 否则，返回默认设备的索引
  return device_default().index();
}

Device device() const {
  // 如果设备策略存在，则返回自定义设备
  if (C10_UNLIKELY(device_policy_)) {
    return device_custom();
  }
  // 否则，返回默认设备
  return device_default();
}

protected:
c10::Device device_default() const {
  // 断言：张量必须有一个设备
  TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
  // 查看并返回设备选项的内容
  // 参见注释 [std::optional operator usage in CUDA]
  return *device_opt_;
}

public:
Layout layout() const {
  // 如果布局策略存在，则返回自定义布局
  if (C10_UNLIKELY(layout_policy_)) {
    return layout_custom();
  }

  // 注意：此方法不是虚拟的，并且为了性能避免了分发
  // 检查是否为 strided 布局，因为 strided 是最常见的布局类型
  // 这个 keyset 也必须与 is_sparse() / is_sparse_csr() / is_mkldnn() 中的逻辑保持同步
  constexpr auto sparse_and_sparsecsr_and_mkldnn_ks =
      c10::sparse_ks | c10::sparse_csr_ks | c10::mkldnn_ks;
  if (!key_set_.has_any(sparse_and_sparsecsr_and_mkldnn_ks)) {
    return kStrided;
  } else if (is_sparse()) {
    return kSparse;
  } else if (is_sparse_compressed()) {
    // 通常，张量分发键唯一定义张量布局
    // 这允许使用非虚拟布局方法以获得更好的性能
    // 然而，当张量的布局依赖于张量属性时，必须使用此执行路径
    // 在这里，相应的张量实现类重写虚拟的 layout_impl() 方法
    //
    // TODO: 实现 layout() 作为本地函数/方法，以便 __torch_dispatch__ 用户能够重新定义 layout() 方法
    return layout_impl();
  } else {
    // 如果不是 CPU tensor，则断言为 MKL-DNN，报告布局计算逻辑错误
    TORCH_INTERNAL_ASSERT(
        is_mkldnn(), "There is an error in the layout calculation logic.");
    // 返回 MKL-DNN 的 DispatchKey
    return kMkldnn;
  }
}

/**
 * 返回是否自动从 C++ 或 Python 数字包装成 Tensor。
 * 例如，当执行 't + 2' 时，2 被自动包装成一个 Tensor，并设置 `is_wrapped_number_` 为 true。
 *
 * 包装的数字不参与混合类型操作的结果类型计算，如果有任何不是包装数字的 Tensor 存在。
 * 这很有用，因为我们希望 't + 2' 能适用于任何类型的 Tensor，而不仅仅是 LongTensor（这是 Python 中的整数表示）。
 *
 * 否则，它们表现得像它们的非包装等价物。
 * 参见 TensorIterator.h 中的 [Result type computation]。
 *
 * 为什么我们选择了包装数字，而不是仅仅添加一个额外的函数 add(Tensor, Scalar) 呢？
 * 这有助于大大减少我们为 add 写的代码量，当实际上 Tensor-Scalar 加法只是当 RHS 是 0 维时的 Tensor-Tensor 加法（除了推广行为）。
 */
bool is_wrapped_number() const {
  return is_wrapped_number_;
}

/**
 * 设置一个 Tensor 是否自动从 C++ 或 Python 数字包装而来。
 * 你可能不想调用这个函数，除非你在编写绑定代码。
 */
void set_wrapped_number(bool value) {
  // 断言 Tensor 的维度是 0
  TORCH_INTERNAL_ASSERT(dim() == 0);
  // 设置 is_wrapped_number_ 属性为指定的值
  is_wrapped_number_ = value;
}

/**
 * 返回 Tensor 是否支持 as_strided 和 as_strided_backward。
 * 这在 autograd 中用于对视图 Tensor 执行原地更新。
 * 参见 Note [View + Inplace update for base tensor] 和
 * [View + Inplace update for view tensor] 以获取详细信息。
 * 注意，此方法仅对 XLA 后端返回 true，它模拟了 strided Tensor 来支持大多数视图操作，
 * 但无法完全支持一般的 `as_strided` 情况。
 * 未来可以根据需要扩展此方法，例如支持稀疏 Tensor。
 */
inline bool support_as_strided() const {
  // 如果是嵌套 Tensor，则不支持 as_strided
  if (is_nested()) {
    return false;
  }
  // 如果 DispatchKey 包含 Functionalize，则不支持 as_strided
  if (key_set_.has(DispatchKey::Functionalize)) {
    return false;
  }
    // 返回当前设备是否支持 as_strided 操作
    return device().supports_as_strided();
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  /**
   * 设置张量是否需要梯度。
   */
  void set_requires_grad(bool requires_grad);

  /**
   * 返回张量是否需要梯度。需要梯度的张量会追踪在其上执行的操作历史，
   * 以便能够自动求导回溯到它们。一个需要梯度且没有历史的张量是一个“叶子”张量，
   * 我们会将梯度积累到它上面。
   */
  bool requires_grad() const;

  /**
   * 返回梯度的可变引用。通常用于 `t.grad() = x` 的形式来将梯度设置为一个全新的张量。
   */
  at::Tensor& mutable_grad();

  /**
   * 返回张量的累积梯度。在执行反向传播时，当这个张量是叶子张量时，会将梯度写入其中。
   */
  const at::Tensor& grad() const;

  /**
   * 是否应该对张量的虚部取负
   */
  inline bool is_conj() const {
    constexpr auto conjugate_ks = DispatchKeySet(DispatchKey::Conjugate);
    return key_set_.has_all(conjugate_ks);
  }

  /**
   * 设置是否对张量进行共轭（翻转虚部）
   */
  void _set_conj(bool value) {
    if (value) {
      key_set_ = key_set_.add(DispatchKey::Conjugate);
      TORCH_INTERNAL_ASSERT(isComplexType(typeMetaToScalarType(dtype())));
    } else {
      key_set_ = key_set_.remove(DispatchKey::Conjugate);
    }
  }

  /**
   * XXX: 不要使用，私有 API！
   * 更新与此设备对应的后端组件相关的键
   */
  void _change_backend_component_keys(c10::Device device);

  /**
   * 张量是否是零张量
   */
  inline bool _is_zerotensor() const {
    constexpr auto zerotensor_ks = DispatchKeySet(DispatchKey::ZeroTensor);
    return key_set_.has_all(zerotensor_ks);
  }

  /**
   * 设置张量是否是零张量
   */
  void _set_zero(bool value) {
    if (value) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Please call `torch._efficientzerotensor` if you want to create a tensor with no storage.");
    } else {
      key_set_ = key_set_.remove(DispatchKey::ZeroTensor);
    }
  }

  /**
   * 张量是否应该取负
   */
  inline bool is_neg() const {
    constexpr auto negative_ks = DispatchKeySet(DispatchKey::Negative);
    return key_set_.has_all(negative_ks);
  }

  /**
   * 设置是否对张量进行取负操作
   */
  void _set_neg(bool value) {
    if (value) {
      key_set_ = key_set_.add(DispatchKey::Negative);
    } else {
      key_set_ = key_set_.remove(DispatchKey::Negative);
    }
  }
  /**
   * Return the accumulated gradient of a tensor. This gradient is computed
   * using forward mode AD.
   *
   * This is an internal API that should never be used by end users.
   *
   * The API is as follows:
   *   - "level" allows to specify the level of forward AD nesting for which the
   *     gradient should be returned. Note that since levels are not fully
   *     supported yet, this argument should be 0. See documentation for
   *     torch::autograd::enter_dual_level for more details about forward AD
   * nesting.
   *   - "self" should represent the Tensor whose forward grad is accessed. It
   *     is required when dealing with view.
   */
  const at::Tensor& _fw_grad(uint64_t level, const at::TensorBase& self) const;

  /**
   * Sets the forward gradient for this Tensor.
   * The given Tensor might not be used directly and its content will be copied.
   *
   * This is an internal API that should never be used by end users.
   *
   * The API is as follows:
   *   - "new_grad" is a Tensor containing the new value of the gradient that
   * should be set
   *   - "self" should represent the Tensor whose forward grad is accessed. It
   *     is required when dealing with view.
   *   - "level" allows to specify the level of forward AD nesting for which the
   *     gradient should be set. Note that since levels are not fully supported
   *     yet, this argument should be 0. See documentation for
   *     torch::autograd::enter_dual_level for more details about forward AD
   * nesting.
   *   - "is_inplace_op" is a boolean flag that tells if this gradient was
   *     generated by an inplace operation or an out of place one. This allows
   *     better error checking.
   */
  void _set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op);

  /**
   * Return a typed data pointer to the actual data which this tensor refers to.
   * This checks that the requested type (from the template parameter) matches
   * the internal type of the tensor.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if
   * the size is 0.
   *
   * WARNING: If a tensor is not contiguous, you MUST use strides when
   * performing index calculations to determine the location of elements in
   * the tensor.  We recommend using 'TensorAccessor' to handle this computation
   * for you; this class is available from 'Tensor'.
   */
  template <typename T>
  const T* data_dtype_initialized() const {
      // Return a pointer to the data of the tensor with type T
      // This ensures that the type T matches the internal type of the tensor
      // and returns a pointer to the actual data.
      // This function is used to access typed data within the tensor.
  // 返回一个常量类型 T 的初始化数据指针，使用指定的函数从存储中获取数据
  return data_dtype_initialized_impl<const T>(
      [this] { return static_cast<const T*>(storage_.data()); });
}

/**
 * 返回指向此张量实际数据的可变类型数据指针。此函数检查请求的类型（从模板参数中）
 * 是否与张量的内部类型匹配。
 *
 * 在未初始化 dtype 的张量上调用 data() 是无效的，即使其大小为 0 也是如此。
 *
 * 警告：如果张量不是连续的，进行索引计算以确定张量中元素位置时必须使用步幅。
 * 我们建议使用 'TensorAccessor' 类来处理此计算；此类可从 'Tensor' 获取。
 */
template <typename T>
T* mutable_data_dtype_initialized() {
  // 返回一个可变类型 T 的初始化数据指针，使用指定的函数从存储中获取数据
  return data_dtype_initialized_impl<T>(
      [this] { return static_cast<T*>(storage_.mutable_data()); });
}

private:
// data_dtype_initialized() 和 mutable_data_dtype_initialized() 的共享实现
template <typename T, typename Func>
T* data_dtype_initialized_impl(const Func& get_data) const {
  TORCH_CHECK(
      data_type_.Match<std::remove_const_t<T>>(),
      "Tensor type mismatch, caller expects elements to be ",
      caffe2::TypeMeta::TypeName<std::remove_const_t<T>>(),
      ", while tensor contains ",
      data_type_.name(),
      ". ");
  // 调用 data_ptr_impl_impl() 以获取具体类型 T 的数据指针
  return data_ptr_impl_impl<T>(get_data);
}

public:
/**
 * Tensor::data_ptr() 的更高效辅助函数。类似于 data<T>()，但不进行类型检查。
 * 与未模板化的 data() 不同，此函数检查 has_storage() 和 storage_initialized()。
 */
template <typename T>
inline const T* data_ptr_impl() const {
  // 返回一个常量类型 T 的数据指针，使用指定的函数从存储中获取数据
  return data_ptr_impl_impl<const T>(
      [this] { return static_cast<const T*>(storage_.data()); });
}

/**
 * Tensor::data_ptr() 的更高效辅助函数。类似于 data<T>()，但不进行类型检查。
 * 与未模板化的 data() 不同，此函数检查 has_storage() 和 storage_initialized()。
 */
template <typename T>
inline T* mutable_data_ptr_impl() {
  // 返回一个可变类型 T 的数据指针，使用指定的函数从存储中获取数据
  return data_ptr_impl_impl<T>(
      [this] { return static_cast<T*>(storage_.mutable_data()); });
}

private:
// mutable_data_ptr_impl() 和未来 mutable_data_ptr_impl() 的共享实现
template <typename T, typename Func>
__ubsan_ignore_pointer_overflow__ T* data_ptr_impl_impl(
    const Func& get_data) const {
  if (C10_UNLIKELY(!has_storage())) {
    // 如果没有存储，则抛出数据指针访问错误
    throw_data_ptr_access_error();
  }
    // 使用 TORCH_CHECK 宏来检查存储是否已初始化，如果未初始化则抛出异常
    TORCH_CHECK(
        storage_initialized(),
        "The tensor has a non-zero number of elements, but its data is not allocated yet.\n"
        "If you're using torch.compile/export/fx, it is likely that we are erroneously "
        "tracing into a custom kernel. To fix this, please wrap the custom kernel into "
        "an opaque custom op. Please see the following for details: "
        "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html\n"
        "If you're using Caffe2, Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    
    // 调用方执行类型检查。
    
    // 注意：即使对于元素数为零的张量，storage_offset_ 也可能是非空的
    // （例如，如果以 `torch.empty(5)[10:]` 创建的张量），这可能触发在 UBSan 中将非零偏移量应用于空指针的情况
    // 返回实际数据的指针，加上存储偏移量
    return get_data() + storage_offset_;
  }

 public:
  
  /**
   * Return a const void* data pointer to the actual data which this
   * tensor refers to.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if the
   * size is 0.
   *
   * WARNING: The data pointed to by this tensor may not contiguous; do NOT
   * assume that itemsize() * numel() is sufficient to compute the bytes that
   * can be validly read from this tensor.
   */
  inline const void* data() const {
    // 调用 data_impl<const void>，返回实际数据的指针
    return data_impl<const void>(
        [this] { return static_cast<const char*>(storage_.data()); });
  }

  /**
   * Return a void* data pointer to the actual data which this tensor refers to.
   *
   * It is invalid to call mutable_data() on a dtype-uninitialized
   * tensor, even if the size is 0.
   *
   * WARNING: The data pointed to by this tensor may not contiguous; do NOT
   * assume that itemsize() * numel() is sufficient to compute the bytes that
   * can be validly read from this tensor.
   */
  inline void* mutable_data() {
    // 调用 data_impl<void>，返回可变数据的指针
    return data_impl<void>(
        [this] { return static_cast<char*>(storage_.mutable_data()); });
  }

 private:
  /// Shared implementation of data() and mutable_data().
  ///
  /// get_data must return a byte-addressed pointer, e.g. char*,
  /// std::byte const*, etc.
  template <typename Void, typename Func>
  Void* data_impl(const Func& get_data) const {
    // 如果没有存储空间，抛出数据指针访问错误异常
    if (C10_UNLIKELY(!has_storage())) {
      throw_data_ptr_access_error();
    }
    // 使用 TORCH_CHECK 宏来检查 dtype 是否已初始化，如果未初始化则抛出异常
    TORCH_CHECK(
        dtype_initialized(),
        "Cannot access data pointer of Tensor that doesn't have initialized dtype "
        "(e.g., caffe2::Tensor x(CPU), prior to calling mutable_data<T>() on x)");
    auto* data = get_data();
    static_assert(
        sizeof(*data) == 1, "get_data must return a byte-addressed pointer.");
    // 如果张量为空，直接返回空指针，避免计算偏移量
    if (is_empty()) {
      return nullptr;
    }
  }

 public:
  /**
   * Returns the TypeMeta of a tensor, which describes what data type
   * it is (e.g., int, float, ...)
   */
  const caffe2::TypeMeta dtype() const {
    // 返回张量的数据类型信息
    return data_type_;
  }

  /**
   * Return the size of a single element of this tensor in bytes.
   */
  size_t itemsize() const {
    // 检查数据类型是否已初始化，否则抛出错误
    TORCH_CHECK(
        dtype_initialized(),
        "Cannot report itemsize of Tensor that doesn't have initialized dtype "
        "(e.g., caffe2::Tensor x(CPU), prior to calling mutable_data<T>() on x)");
    // 返回单个元素的字节大小
    return data_type_.itemsize();
  }

  void set_backend_meta(intrusive_ptr<c10::BackendMeta> backend_meta) {
    // 设置后端元数据
    get_extra_meta().backend_meta_ = std::move(backend_meta);
  }

  c10::BackendMeta* get_backend_meta() {
    // 获取后端元数据指针，如果不存在则返回空指针
    if (!extra_meta_) {
      return nullptr;
    }
    return extra_meta_->backend_meta_.get();
  }

  intrusive_ptr<c10::BackendMeta> get_backend_meta_intrusive_ptr() const {
    // 获取后端元数据的引用指针，如果不存在则返回空指针
    if (!extra_meta_) {
      return nullptr;
    }
    return extra_meta_->backend_meta_;
  }

  void release_storage_and_set_meta_custom_data_ptr_error_msg_(
      std::optional<std::string> s) {
    // 释放存储并设置自定义错误消息的元数据
    storage_ = {};
    set_storage_access_should_throw();
    get_extra_meta().custom_data_ptr_error_msg_ = s;
    get_extra_meta().custom_storage_error_msg_ = std::move(s);
  }

 protected:
  /**
   * Returns the human-readable name of the actual type of this object (e.g.,
   * TensorImpl, BatchedTensorImpl, etc.). Used for error messages.
   */
  virtual const char* tensorimpl_type_name() const {
    // 返回对象实际类型的人类可读名称，用于错误消息
    return "TensorImpl";
  }

 private:
  [[noreturn]] void throw_storage_access_error() const;
  [[noreturn]] void throw_data_ptr_access_error() const;

  ExtraMeta& get_extra_meta() {
    // 获取额外元数据的引用，如果不存在则创建一个新的额外元数据对象
    if (!extra_meta_) {
      extra_meta_ = std::make_unique<ExtraMeta>();
    }
    return *extra_meta_;
  }

  c10::SymbolicShapeMeta& symbolic_shape_meta() {
    // 获取符号形状元数据的引用，前提是它已经存在
    TORCH_INTERNAL_ASSERT(extra_meta_ && extra_meta_->symbolic_shape_meta_);
    return *extra_meta_->symbolic_shape_meta_;
  }

  const c10::SymbolicShapeMeta& symbolic_shape_meta() const {
    // 获取符号形状元数据的常量引用，前提是它已经存在
    TORCH_INTERNAL_ASSERT(extra_meta_ && extra_meta_->symbolic_shape_meta_);
    return *extra_meta_->symbolic_shape_meta_;
  }

 public:
  /**
   * True if a tensor has no elements (e.g., numel() == 0).
   */
  inline bool is_empty() const {
    // 如果张量没有元素则返回真
    return numel() == 0;
  }

  // if we are going to use sym sizes, we should be setting sym strides at the
  // same time, otherwise it's very easy to misuse this API
  void set_sizes_and_strides(
      c10::SymIntArrayRef sizes,
      c10::SymIntArrayRef strides,
      std::optional<c10::SymInt> storage_offset = c10::nullopt);
  // This is renamed to avoid breaking overload BC
  void generic_set_sizes_contiguous(c10::SymIntArrayRef sizes);
  void generic_set_sizes_contiguous(c10::IntArrayRef sizes) {
    // 设置大小并保证连续性，使用符号大小时应同时设置符号步长，否则容易误用此 API
  /**
   * Set the sizes of the tensor to be contiguous in memory.
   *
   * This function assumes the tensor is contiguous, meaning it has a specific
   * order of dimensions where each element is adjacent to the next in memory.
   * 
   * Preconditions:
   * - `allow_tensor_metadata_change()` must return true, allowing metadata modifications.
   * - Tensor must not have customized stride behavior (CustomStrides policy).
   *
   * Postconditions:
   * - Updates the sizes of the tensor to match `new_size`.
   * - Calls `refresh_numel()` to update the total number of elements in the tensor.
   */
  void set_sizes_contiguous(IntArrayRef new_size) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_sizes_contiguous ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        !matches_policy(SizesStridesPolicy::CustomStrides),
        "tried to directly modify sizes for customized tensor");
    sizes_and_strides_.set_sizes(new_size);

    refresh_numel();
  }
  // 调用 empty_tensor_restride 函数，并传入 MemoryFormat::Contiguous 作为参数，这会调用 refresh_contiguous() 函数
  empty_tensor_restride(
      MemoryFormat::Contiguous); // calls refresh_contiguous()
}

/**
 * 设置张量的大小和步幅。
 *
 * 警告：此函数不会检查请求的大小/步幅是否在分配的存储器的范围内；
 * 这是调用者的责任。
 */
void set_sizes_and_strides(
    IntArrayRef new_size,
    IntArrayRef new_stride,
    std::optional<int64_t> storage_offset = c10::nullopt) {
  TORCH_CHECK(
      allow_tensor_metadata_change(),
      "set_sizes_and_strides ",
      err_msg_tensor_metadata_change_not_allowed);
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "set_sizes_and_strides() called on tensor with symbolic shape")
  TORCH_CHECK(
      new_size.size() == new_stride.size(),
      "dimensionality of sizes (",
      new_size.size(),
      ") must match dimensionality of strides (",
      new_stride.size(),
      ")");
  const auto new_dim = new_size.size();
  bool overflowed = false;
  
  // 设置张量的大小
  sizes_and_strides_.set_sizes(new_size);

  // 如果张量维度大于 0，则进行步幅设置
  if (new_dim > 0) {
    for (size_t dim = new_dim - 1;; dim--) {
      if (new_stride[dim] >= 0) {
        // 如果步幅非负，则直接设置
        sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
      } else {
        // 如果步幅为负数，根据条件设置
        // XXX: 这种行为可能会被移除以支持负步幅，但某些 PyTorch 函数依赖于此行为，例如 torch.cat
        if (dim == new_dim - 1) {
          sizes_and_strides_.stride_at_unchecked(dim) = 1;
        } else {
          // 保持步幅单调递增以匹配 NumPy
          overflowed |= c10::mul_overflows(
              sizes_and_strides_.stride_at_unchecked(dim + 1),
              std::max<int64_t>(
                  sizes_and_strides_.size_at_unchecked(dim + 1), 1),
              std::addressof(sizes_and_strides_.stride_at_unchecked(dim)));
        }
      }
      if (dim == 0)
        break;
    }
    // 检查步幅计算是否溢出
    TORCH_CHECK(!overflowed, "Stride calculation overflowed");
  }

  // 更新张量元素个数
  refresh_numel();
  // 刷新连续内存布局
  refresh_contiguous();

  // 如果提供了 storage_offset，则设置存储偏移量
  if (storage_offset.has_value()) {
    storage_offset_ = *storage_offset;
  }
}

/**
 * 设置张量是否允许更改其元数据（例如大小/步幅/存储/存储偏移量）。
 * 详见 NOTE [ Metadata Change for a Detached Tensor ]。
 */
void set_allow_tensor_metadata_change(bool value) {
  // TODO: 在某些时候，我们应该彻底删除这个字段。
  allow_tensor_metadata_change_ = true;
}

/**
 * 如果张量允许更改其元数据（例如大小/步幅/存储/存储偏移量），则返回 true。
 * 详见 NOTE [ Metadata Change for a Detached Tensor ]。
 */
bool allow_tensor_metadata_change() const {
  // 返回允许张量元数据更改的标志。
  return allow_tensor_metadata_change_;
}

/**
 * 设置自动求导元数据的指针。
 */
void set_autograd_meta(
    std::unique_ptr<c10::AutogradMetaInterface> autograd_meta);

/**
 * 返回自动求导元数据的指针。
 * 如果张量不跟踪梯度，则可能返回 nullptr。
 */
c10::AutogradMetaInterface* autograd_meta() const;

/**
 * 设置命名张量元数据的指针。
 */
void set_named_tensor_meta(
    std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta) {
  // 发出警告，指出命名张量及其所有相关的 API 是实验性功能，
  // 可能会发生变化。在它们稳定发布前，请不要在重要场合使用。
  TORCH_WARN_ONCE(
      "Named tensors and all their associated APIs are an experimental feature ",
      "and subject to change. Please do not use them for anything important ",
      "until they are released as stable.");
#ifdef DEBUG
    // 如果在调试模式下，并且存在命名张量元数据，进行断言检查慢维度是否与当前维度相同
    if (named_tensor_meta) {
      TORCH_INTERNAL_ASSERT(named_tensor_meta->slow_dim() == dim());
    }
#endif

    // 如果存在命名张量元数据
    if (named_tensor_meta) {
      // 将额外元数据中的命名张量元数据移动到当前对象的额外元数据中
      get_extra_meta().named_tensor_meta_ = std::move(named_tensor_meta);
      // 将命名张量的 DispatchKey 添加到当前对象的 key_set_
      key_set_ = key_set_.add(DispatchKey::Named);
    } else {
      // 如果不存在命名张量元数据
      if (extra_meta_) {
        // 将当前对象的额外元数据中的命名张量元数据设为 nullptr
        extra_meta_->named_tensor_meta_ = nullptr;
      }
      // 将命名张量的 DispatchKey 从当前对象的 key_set_ 中移除
      key_set_ = key_set_.remove(DispatchKey::Named);
    }
  }

  // 设置是否启用 Python 分发
  void set_python_dispatch(bool k) {
    // 如果 k 为 true，将 Python 分发的 DispatchKey 添加到当前对象的 key_set_
    if (k) {
      key_set_ = key_set_.add(c10::python_ks);
    } else {
      // 否则，从当前对象的 key_set_ 中移除 Python 分发的 DispatchKey
      key_set_ = key_set_ - c10::python_ks;
    }
  }

  // 检查当前对象是否启用了 Python 分发
  bool is_python_dispatch() const {
    // 返回当前对象的 key_set_ 是否包含所有 Python 分发的 DispatchKey
    return key_set_.has_all(c10::python_ks);
  }

  /**
   * Return the pointer to named tensor metadata.
   */
  // 返回命名张量元数据的指针（只读版本）
  const c10::NamedTensorMetaInterface* named_tensor_meta() const {
    // 如果额外元数据不存在，则返回 nullptr
    if (!extra_meta_) {
      return nullptr;
    }
    // 返回额外元数据中的命名张量元数据指针
    return extra_meta_->named_tensor_meta_.get();
  }

  // 返回命名张量元数据的指针（可写版本）
  c10::NamedTensorMetaInterface* named_tensor_meta() {
    // 如果额外元数据不存在，则返回 nullptr
    if (!extra_meta_) {
      return nullptr;
    }
    // 返回额外元数据中的命名张量元数据指针
    return extra_meta_->named_tensor_meta_.get();
  }

  // 检查当前对象是否包含命名张量元数据
  bool has_named_tensor_meta() const {
    // 如果额外元数据不存在，则返回 false
    if (!extra_meta_) {
      return false;
    }
    // 否则，返回额外元数据中的命名张量元数据是否不为 nullptr
    // 即判断是否有命名张量元数据
    // 注意：函数缺少了结束大括号 '}'
  // 返回额外元数据中的命名张量元数据是否不为空指针
  return extra_meta_->named_tensor_meta_ != nullptr;
}

// 注意事项 [ TensorImpl 浅复制 ]
//
// 当我们希望两个变量共享相同的张量元数据（如大小 / 步长 / 存储指针 / 存储偏移），
// 但每个变量有不同的自动求导历史时，使用 TensorImpl 浅复制。示例调用场景：
//
// 1. `var_detached = var.detach()` 使用 `shallow_copy_and_detach()` 创建
//    `var_detached`，它与 `var` 共享相同的张量元数据，但具有全新的自动求导历史。
// 2. `var.set_data(tensor)` 使用 `shallow_copy_from()` 将 `tensor` 的张量元数据
//    复制到 `var` 中，同时保留 `var` 的原始 AutogradMeta。
//
// 执行张量元数据浅复制的函数（例如 `shallow_copy_and_detach()` / `shallow_copy_from()` /
// `copy_tensor_metadata()`）通过值复制张量元数据字段（如大小 / 步长 / 存储指针 / 存储偏移）。
// 然而，以下字段不会被复制：
//
// 1. AutogradMeta 指针，因为每个变量都是唯一的。
// 2. 版本计数器，因为目标 TensorImpl 的版本计数器要么设置为传入的 `version_counter`
//    （在 `shallow_copy_and_detach()` 和 `copy_tensor_metadata()` 中），要么保持不变
//    （在 `shallow_copy_from()` 中）。详情请参见注意事项 [ 版本计数器共享 ]。
//
// 在 `shallow_copy_and_detach()` 和 `copy_tensor_metadata()` 中，传入的
// `allow_tensor_metadata_change` 决定了是否允许对张量元数据进行更改（如大小 / 步长 /
// 存储 / 存储偏移）。详情请参见注意事项 [ 分离张量的元数据更改 ]。
//
// 在 `shallow_copy_from()` 中，我们不检查目标 TensorImpl 的 `allow_tensor_metadata_change_`，
// 因为 `shallow_copy_from()` 用于实现诸如 `var.set_data(tensor)` 的函数，该函数修改
// `var` 的张量元数据，并期望其 `allow_tensor_metadata_change_` 被忽略。

/**
 * 如果两个 DispatchKeySet 有相同的 DispatchKeySet，则可以将一个 TensorImpl
 * 复制到另一个 TensorImpl 中。仅有两个特殊情况（出于遗留原因）：
 * CPU 与 CUDA 兼容，SparseCPU 与 SparseCUDA 兼容。
 */
inline bool has_compatible_shallow_copy_type(DispatchKeySet from) {
  auto is_dense = [](DispatchKeySet ts) {
    constexpr auto dense_backends = DispatchKeySet(
        {BackendComponent::CPUBit,
         BackendComponent::CUDABit,
         BackendComponent::MPSBit,
         BackendComponent::HIPBit,
         BackendComponent::XPUBit,
         BackendComponent::HPUBit});
    constexpr auto dense_k = DispatchKeySet(DispatchKey::Dense);
    return ts.has_any(dense_k) && ts.has_any(dense_backends);
  };
    // 定义一个 lambda 函数 is_sparse，用于检查给定的 DispatchKeySet 是否为稀疏张量
    auto is_sparse = [](DispatchKeySet ts) {
      // 定义包含稀疏后端的 DispatchKeySet
      constexpr auto sparse_backends = DispatchKeySet(
          {BackendComponent::CPUBit,
           BackendComponent::CUDABit,
           BackendComponent::HIPBit,
           BackendComponent::XPUBit});
      // 定义包含稀疏类型的 DispatchKeySet
      constexpr auto sparse_k = DispatchKeySet(DispatchKey::Sparse);
      // 返回是否同时包含稀疏类型和稀疏后端
      return ts.has_any(sparse_k) && ts.has_any(sparse_backends);
    };

    // 定义一个 lambda 函数 is_sparse_compressed，用于检查给定的 DispatchKeySet 是否为压缩稀疏张量
    auto is_sparse_compressed = [](DispatchKeySet ts) {
      // 定义包含压缩稀疏类型的 DispatchKeySet
      constexpr auto sparse_compressed_k =
          DispatchKeySet(DispatchKey::SparseCsr);
      // 返回是否包含压缩稀疏类型
      return ts.has_any(sparse_compressed_k);
    };

    // 返回是否满足以下条件之一：
    // 1. key_set_ 与 from 相等
    // 2. key_set_ 和 from 都是稠密张量
    // 3. key_set_ 和 from 都是稀疏张量
    // 4. key_set_ 和 from 都是压缩稀疏张量
    return (key_set_ == from) || (is_dense(key_set_) && is_dense(from)) ||
        (is_sparse(key_set_) && is_sparse(from)) ||
        (is_sparse_compressed(key_set_) && is_sparse_compressed(from));
    // 注意：这里存在多余的分号，需删除
    ;
  }

 private:
  
  // 定义一个模板函数 shallow_copy_and_detach_core，用于实现浅拷贝和分离操作的核心功能
  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;

 public:
  
  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  // 定义虚函数 shallow_copy_and_detach，返回当前 TensorImpl 的浅拷贝
  virtual c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const;

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  // 定义虚函数 shallow_copy_and_detach，返回当前 TensorImpl 的浅拷贝（移动语义版本）
  virtual c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's
   * `allow_tensor_metadata_change_`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  // 定义虚函数 shallow_copy_from，从另一个 TensorImpl 浅拷贝数据到当前 TensorImpl
  virtual void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
    // 调用 copy_tensor_metadata 函数完成元数据的拷贝
    copy_tensor_metadata(
        /*src_impl=*/impl.get(),
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  }

  // 推断张量不具备版本计数器，对它们使用 set_version_counter 是无效操作
  // 在推断张量情况下，set_version_counter 是一个空操作
  void set_version_counter(const c10::VariableVersion& version_counter) {
    // 如果当前为推断张量，并且版本计数器被启用，则抛出错误
    TORCH_CHECK(
        !(is_inference() && version_counter.enabled()),
        "Cannot set version_counter for inference tensor");
    // 否则，设置版本计数器为给定值
    version_counter_ = version_counter;
  }

  // 移动语义版本的 set_version_counter 函数
  void set_version_counter(c10::VariableVersion&& version_counter) {
    // 如果当前为推断张量，并且版本计数器被启用，则抛出错误
    TORCH_CHECK(
        !(is_inference() && version_counter.enabled()),
        "Cannot set version_counter for inference tensor");
    // 否则，设置版本计数器为给定值（使用移动语义）
    version_counter_ = std::move(version_counter);
  }

  // 返回当前 TensorImpl 的版本计数器的常量引用
  const c10::VariableVersion& version_counter() const noexcept {
    return version_counter_;
  }

  // 递增当前 TensorImpl 的版本号
  void bump_version() {
    version_counter_.bump();
  }

  // 返回指向 pyobj_slot_ 的指针
  impl::PyObjectSlot* pyobj_slot() {
    return &pyobj_slot_;
  }

  // 返回指向 pyobj_slot_ 的常量指针
  const impl::PyObjectSlot* pyobj_slot() const {
    return &pyobj_slot_;
  }

 private:
  // See NOTE [std::optional operator usage in CUDA]
  // We probably don't want to expose this publicly until
  // the note is addressed.
  // 返回一个可选类型的设备对象，参见 CUDA 中的使用说明
  std::optional<c10::Device> device_opt() const {
    return device_opt_;
  }

 public:
  /**
   * The device type of a Tensor, e.g., DeviceType::CPU or DeviceType::CUDA.
   */
  // 返回张量的设备类型
  DeviceType device_type() const {
    // TODO: A useful internal assert would be to show that device_opt_ is null
    // only if you are an undefined tensor
    // 断言检查 device_opt_ 是否有值，否则报错显示 undefined Tensor
    TORCH_CHECK(
        device_opt_.has_value(),
        "device_type cannot be run on undefined Tensor");
    // See NOTE [std::optional operator usage in CUDA]
    // 返回设备对象的类型
    return (*device_opt_).type();
  }

  /**
   * @brief Extends the outer-most dimension of this tensor by num elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * growthPct. This ensures that Extend runs on an amortized O(1) time
   * complexity.
   *
   * This op is auto-asynchronous if the underlying device (CUDA) supports it.
   */
  // 扩展张量的最外层维度
  void Extend(int64_t num, float growthPct);

  /**
   * @brief Reserve space for the underlying tensor.
   *
   * This must be called after Resize(), since we only specify the first
   * dimension This does not copy over the old data to the newly allocated space
   */
  // 为底层张量预留空间
  void ReserveSpace(int64_t outer_dim);

  /**
   * @brief Resizes a tensor.
   *
   * Resize takes in a vector of ints specifying the dimensions of the tensor.
   * You can pass in an empty vector to specify that it is a scalar (i.e.
   * containing one single item).
   *
   * The underlying storage may be deleted after calling Resize: if the new
   * shape leads to a different number of items in the tensor, the old memory
   * is deleted and new memory will be allocated next time you call
   * mutable_data(). However, if the shape is different but the total number of
   * items is the same, the underlying storage is kept.
   *
   * This method respects caffe2_keep_on_shrink.  Consult the internal logic
   * of this method to see exactly under what circumstances this flag matters.
   */
  // 调整张量的大小
  template <typename... Ts>
  void Resize(Ts... dim_source) {
    // 设置张量的维度并检查是否有变化
    bool size_changed = SetDims(dim_source...);
    // 如果大小有变化则处理调整
    if (size_changed) {
      HandleResize();
    }
  }

  template <typename T>
  // 调整张量的大小，接受一个维度信息的向量
  void Resize(const std::vector<T>& dim_source) {
  // 调整张量大小为给定的维度，不影响底层存储
  Resize(ArrayRef<T>(dim_source));
}

/**
 * 不改变底层存储的情况下重新调整张量大小。
 * 这要求张量的总大小保持不变。
 */
void Reshape(const std::vector<int64_t>& dims);

/**
 * 释放张量当前持有的内存，但保留大小和类型信息。
 * 下一次调用 mutable_data 将触发新的内存分配。
 */
void FreeMemory();

/**
 * @brief 与另一个张量共享数据。
 *
 * 要共享两个张量之间的数据，两个张量的大小必须已经相等。
 * 我们不会自动调整大小以使两个张量具有相同的形状，因为我们希望允许具有不同形状但相同数量项的张量仍然能够共享数据。
 * 这使得可以有一个 n 维张量和一个扁平化版本共享相同的底层存储。
 *
 * 源张量应已分配其数据。
 */
// 即将弃用
void ShareData(const TensorImpl& src);

void ShareExternalPointer(
    DataPtr&& data_ptr,
    const caffe2::TypeMeta data_type,
    size_t size_bytes);

/**
 * 返回底层存储的可变原始指针。由于我们需要知道数据的类型以进行分配，
 * 因此传入一个 TypeMeta 对象来指定所需的信息。
 * 这在概念上相当于调用 mutable_data<T>()，其中 TypeMeta 参数 meta 派生自类型 T。
 * 该函数与 mutable_data<T>() 不同之处在于，类型 T 可以通过运行时通过 TypeMeta 对象指定。
 *
 * 如果现有数据与所需类型不匹配，将删除现有数据并创建新的存储。
 */
inline void* raw_mutable_data(const caffe2::TypeMeta& meta) {
  // 对于大小为 0 的张量，返回任意指针（包括 nullptr）都是可以的
  if (data_type_ == meta && storage_initialized()) {
    return static_cast<void*>(
        static_cast<char*>(storage_.mutable_data()) +
        storage_offset_ * meta.itemsize());
    } else {
      // 检查当前数据类型是否有特殊析构函数，并记录下来
      bool had_special_dtor = data_type_.placementDelete() != nullptr;
      // 重置存储偏移量为0，设置新的数据类型为meta
      storage_offset_ = 0;
      data_type_ = meta;
      // 注意：设备并未改变

      // 如果当前元素数为0，或者当前数据类型没有特殊构造函数，且新数据类型也没有特殊析构函数，
      // 并且存储器容量大于等于所需空间，则可以重用现有缓冲区
      if (numel_ == 0 ||
          (meta.placementNew() == nullptr && !had_special_dtor &&
           (storage_.nbytes() >= (numel_ * data_type_.itemsize())))) {
        // 因为刚刚重新分配，所以断言存储偏移量为0
        TORCH_INTERNAL_ASSERT(
            storage_offset_ == 0); // because we just reallocated
        // 返回可变数据的指针
        return storage_.mutable_data();
      }
      // 获取存储器的分配器
      Allocator* allocator = storage_.allocator();
      // 在稀有情况下，存储器可能具有空指针分配器，例如，如果外部内存段已包装为Tensor，
      // 而我们不知道如何重新分配它。为了保留旧版C2行为，允许使用默认分配器重新分配内存。
      if (allocator == nullptr) {
        allocator = GetAllocator(storage_.device_type());
      }
      // 如果新数据类型需要placement new
      if (meta.placementNew()) {
        // 对于需要placement new的类型，调用placement new，并确保在释放数据时调用正确的析构过程
        auto size = numel_;
        auto dtor = data_type_.placementDelete();
        auto data_ptr = allocator->allocate(numel_ * data_type_.itemsize());
        storage_.set_data_ptr_noswap(PlacementDeleteContext::makeDataPtr(
            std::move(data_ptr), dtor, size, storage_.device()));
        // 调用placement new函数初始化数据
        data_type_.placementNew()(storage_.mutable_data(), numel_);
      } else {
        // 对于基本类型，使用传统的new和delete
        storage_.set_data_ptr_noswap(
            allocator->allocate(numel_ * data_type_.itemsize()));
      }
      // 设置存储器的字节数
      storage_.set_nbytes(numel_ * data_type_.itemsize());
      // 因为刚刚重新分配，所以断言存储偏移量为0
      TORCH_INTERNAL_ASSERT(
          storage_offset_ == 0); // because we just reallocated
      // 更新设备选项为存储器的设备
      device_opt_ = storage_.device();
      // 返回可变数据的指针
      return storage_.mutable_data();
    }
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * For fundamental types, we reuse possible existing storage if there
   * is sufficient capacity.
   */
  template <typename T>
  inline T* mutable_data() {
    // 如果存储器已初始化且数据类型与T匹配，则返回T类型的指针
    if (storage_initialized() && data_type_.Match<T>()) {
      return static_cast<T*>(storage_.mutable_data()) + storage_offset_;
    }
    // 静态检查：如果T类型不可默认构造，则抛出静态断言错误
    // "Tensor can't hold non-default-constructable types"
    static_assert(
        std::is_default_constructible<T>::value,
        "Tensor can't hold non-default-constructable types");
    // 否则调用raw_mutable_data函数获取T类型的可变数据指针
    return static_cast<T*>(raw_mutable_data(caffe2::TypeMeta::Make<T>()));
  }

  /**
   * True if a tensor is storage initialized.  A tensor may become
   * storage UNINITIALIZED after a Resize() or FreeMemory()
   */
  bool storage_initialized() const {
    TORCH_CHECK(
        has_storage(),
        "cannot call storage_initialized on tensor that does not have storage");


    // 检查张量是否有有效的存储空间，否则抛出错误信息
    TORCH_CHECK(
        has_storage(),
        "cannot call storage_initialized on tensor that does not have storage");
    // 如果张量没有数据存储，则返回 false
    return storage_.data() || numel_ == 0;
  }

  /**
   * True if a tensor is dtype initialized.  A tensor allocated with
   * Caffe2-style constructors is dtype uninitialized until the
   * first time mutable_data<T>() is called.
   */
  bool dtype_initialized() const noexcept {
    // 判断张量的数据类型是否已经初始化
    return data_type_ != caffe2::TypeMeta();
  }

  void set_storage_keep_dtype(at::Storage storage) {
    // 检查是否允许修改张量元数据，否则抛出错误信息
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_storage ",
        err_msg_tensor_metadata_change_not_allowed);
    // 将给定的存储设置为张量的新存储
    storage_ = std::move(storage);
    // 更新张量的设备信息为新存储的设备信息
    device_opt_ = storage_.device();
  }

  void set_storage_and_dtype(
      at::Storage storage,
      const caffe2::TypeMeta data_type) {
    // 设置张量的存储和数据类型，保持当前的数据类型不变
    set_storage_keep_dtype(std::move(storage));
    // 设置张量的数据类型为指定的数据类型
    data_type_ = data_type;
  }

  void empty_tensor_restride_symint(MemoryFormat memory_format);

  /**
   * Set the strides of the tensor to match memory_format
   *
   * WARNING: This function doesn't rearrange data and assumes tensor is a
   * memory contiguous
   */
  void empty_tensor_restride(MemoryFormat memory_format) {
    // 如果张量具有符号大小和步幅，则调用符号重排函数进行操作
    if (has_symbolic_sizes_strides_) {
      empty_tensor_restride_symint(memory_format);
      return;
    }
#ifdef DEBUG
    // 在调试模式下，进行张量大小检查，确保张量元素数正确
    TORCH_INTERNAL_ASSERT(
        compute_numel() == numel_,
        "If you are seeing this error, that means empty_tensor_restride was "
        "called before setting correct numel");
#endif

    // 根据内存格式进行不同处理
    switch (memory_format) {
      case MemoryFormat::Contiguous: {
        // 获取张量的维度
        const auto dim_ = dim();
        // 调整 sizes_and_strides_ 的大小以适应当前张量维度
        sizes_and_strides_.resize(dim_);
        if (dim_ > 0) {
          bool overflowed = false;
          const auto last_idx = dim_ - 1;
          // 设置最后一个维度的步幅为 1
          sizes_and_strides_.stride_at_unchecked(last_idx) = 1;
          // 计算其他维度的步幅
          for (auto i = last_idx - 1; i >= 0; --i) {
            overflowed |= c10::mul_overflows(
                sizes_and_strides_.stride_at_unchecked(i + 1),
                std::max<int64_t>(
                    sizes_and_strides_.size_at_unchecked(i + 1), 1),
                std::addressof(sizes_and_strides_.stride_at_unchecked(i)));
          }
          // 检查步幅计算是否溢出
          TORCH_CHECK(!overflowed, "Stride calculation overflowed");
        }
        break;
      }
      case MemoryFormat::ChannelsLast: {
        // 检查张量维度是否为 4，以支持 channels_last 格式
        TORCH_CHECK(
            dim() == 4, "required rank 4 tensor to use channels_last format");
        // 设置 sizes_and_strides_ 为 channels_last 格式的尺寸和步幅
        set_sizes_and_strides(sizes(), get_channels_last_strides_2d(sizes()));
        break;
      }
      case MemoryFormat::ChannelsLast3d: {
        // 检查张量维度是否为 5，以支持 channels_last_3d 格式
        TORCH_CHECK(
            dim() == 5,
            "required rank 5 tensor to use channels_last_3d format");
        // 设置 sizes_and_strides_ 为 channels_last_3d 格式的尺寸和步幅
        set_sizes_and_strides(sizes(), get_channels_last_strides_3d(sizes()));
        break;
      }
      case MemoryFormat::Preserve:
        // 不支持的内存格式，抛出错误
        TORCH_CHECK(false, "unsupported memory format ", memory_format);
        // 注意：由于 TORCH_CHECK(false) 会终止流程，因此此处不需要 break
      case MemoryFormat::NumOptions:
        // 内存格式选项错误，抛出错误
        TORCH_INTERNAL_ASSERT(false, "invalid memory format ", memory_format);
    }

    // 重新计算 contiguous 标志位，因为目前 NHWC/NCHW 标志不是互斥的
    refresh_contiguous();
  }

  // 判断是否符合指定内存格式的步幅设置
  bool is_strides_like(at::MemoryFormat memory_format) const {
    // 如果当前张量使用自定义步幅策略，则调用自定义步幅检查函数
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      return is_strides_like_custom(memory_format);
    }
    // 否则调用默认的步幅检查函数
    return is_strides_like_default(memory_format);
  }

  // 判断是否符合 channels_last 格式的步幅设置
  bool is_strides_like_channels_last() const {
    return is_strides_like(at::MemoryFormat::ChannelsLast);
  }

  // 判断是否符合 channels_last_3d 格式的步幅设置
  bool is_strides_like_channels_last_3d() const {
    return is_strides_like(at::MemoryFormat::ChannelsLast3d);
  }

  // 判断张量是否是非重叠且稠密的
  bool is_non_overlapping_and_dense() const {
    // 如果当前张量使用自定义步幅策略，则调用自定义函数进行检查
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      return is_non_overlapping_and_dense_custom();
    }
    // 否则调用默认函数进行检查
    return is_non_overlapping_and_dense_default();
  }

  // 判断张量是否具有符号尺寸/步幅
  // 如果返回 true，则可以保证该张量具有符号尺寸/步幅
  bool has_symbolic_sizes_strides() const {
  return has_symbolic_sizes_strides_;
}

private:
  // 处理重新调整尺寸的私有方法
  void HandleResize();

  // Caffe2 的 Resize() 方法支持多种调用方式，包括 Resize({2,2}) 和 Resize(2, 2)。
  // 这些重载提供了所有支持的调用配置，同时作为重载而不是模板，以便支持隐式类型转换。
  //
  // ArrayRef 上的 SetDims 内部实现为模板，因此可以处理不同类型的 ArrayRef（Caffe2 的一些用例中传递 int 而不是 int64_t。）
  
  template <
      typename T,
      typename = typename std::enable_if_t<std::is_integral_v<T>>>
  bool SetDimsTemplate(ArrayRef<T> src) {
    // 检查是否具有符号形状/步幅
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "SetDims() called on tensor with symbolic shape")

    auto old_numel = numel_;
    sizes_and_strides_.resize(src.size());
    int64_t new_numel = 1;
    for (const auto i : c10::irange(src.size())) {
      new_numel *= src[i];
      sizes_and_strides_.size_at_unchecked(i) = src[i];
    }
    numel_ = new_numel;
    // 重置张量的步幅和内存格式
    empty_tensor_restride(MemoryFormat::Contiguous);
    return numel_ != old_numel;
  }

  // 使用 int64_t 类型的 ArrayRef 设置张量尺寸
  bool SetDims(ArrayRef<int64_t> s) {
    return SetDimsTemplate(s);
  }

  // 使用 int 类型的 ArrayRef 设置张量尺寸
  bool SetDims(ArrayRef<int> s) {
    return SetDimsTemplate(s);
  }

  // 使用 size_t 类型的 ArrayRef 设置张量尺寸
  bool SetDims(ArrayRef<size_t> s) {
    return SetDimsTemplate(s);
  }

  // 没有参数的 SetDims 方法，将其设置为空的 IntArrayRef
  bool SetDims() {
    return SetDims(IntArrayRef{});
  }

  // 设置单个维度的张量尺寸
  bool SetDims(const int64_t d0) {
    return SetDims(IntArrayRef{d0});
  }

  // 设置两个维度的张量尺寸
  bool SetDims(const int64_t d0, const int64_t d1) {
    return SetDims(IntArrayRef{d0, d1});
  }

  // 设置三个维度的张量尺寸
  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
    return SetDims(IntArrayRef{d0, d1, d2});
  }

  // 设置四个维度的张量尺寸
  bool SetDims(
      const int64_t d0,
      const int64_t d1,
      const int64_t d2,
      const int64_t d3) {
    return SetDims(IntArrayRef{d0, d1, d2, d3});
  }

  /**
   * 基于张量的尺寸计算元素数量。
   */
  // 注意：仅当直接使用 sizes_and_strides_ 时才会调用此方法；
  // 如果我们进行虚拟化，则 numel 调用也会被虚拟化，这时不应该调用这个方法。
  int64_t compute_numel() const {
    // 内部断言调试：确保没有符号形状/步幅
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!has_symbolic_sizes_strides_);
#if C10_HAS_BUILTIN_OVERFLOW() && !defined(C10_MOBILE)
    // 如果编译器支持溢出检查并且不是在移动端，使用溢出检查
    return safe_compute_numel();
#else
    // 否则调用普通的整数乘法函数计算张量元素总数
    return c10::multiply_integers(sizes_and_strides_.sizes_arrayref());
#endif
  }

/**
 * 根据张量的尺寸计算元素的数量。捕获可能发生的整数溢出，特别是在使用稀疏布局的张量有多个尺寸很大的维度时。
 */
int64_t safe_compute_numel() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!has_symbolic_sizes_strides_);
  // 初始化元素数量为1
  uint64_t n = 1;
  // 使用安全的无溢出乘法计算张量尺寸的乘积，并将结果存储到n中
  bool overflows =
      c10::safe_multiplies_u64(sizes_and_strides_.sizes_arrayref(), &n);
  // 设置最大允许的元素数量
  constexpr auto numel_max = std::min(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  // 检查是否发生溢出
  overflows |= (n > numel_max);
  // 如果发生溢出，抛出异常
  TORCH_CHECK(!overflows, "numel: integer multiplication overflow");
  // 返回元素数量，转换为int64_t类型
  return static_cast<int64_t>(n);
}

/**
 * 根据张量的尺寸和步幅计算张量是否是连续的。
 */
bool compute_contiguous(identity<bool>) const;

bool compute_channels_last_contiguous_2d(identity<bool>) const;

bool compute_channels_last_contiguous_3d(identity<bool>) const;

bool compute_strides_like_channels_last_2d(identity<bool>) const;

bool compute_strides_like_channels_last_3d(identity<bool>) const;

bool compute_non_overlapping_and_dense(identity<bool>) const;

protected:
/**
 * 重新计算张量的缓存元素数量。在修改尺寸时调用此函数。
 *
 * 对于使用稀疏布局的张量，请使用safe_refresh_numel()，因为它能捕获可能发生的整数溢出，特别是对于具有稀疏布局和大维度的张量。
 *
 * 注意：即使在从未使用缓存元素数量的情况下重新计算它（例如，对于Python中的CustomSizes），我们仍然必须保持其更新，以防Python重载返回None（在这种情况下，我们将查询此字段）。
 * 这也意味着尺寸/步幅永远不会是完全的垃圾数据；在最坏的情况下，它将反映一个1维零尺寸张量。
 */
void refresh_numel() {
  if (has_symbolic_sizes_strides_) {
    // 如果有符号尺寸和步幅，使用符号形状元数据重新计算元素数量
    symbolic_shape_meta().refresh_numel();
  } else {
    // 否则直接使用普通的计算元素数量函数
    numel_ = compute_numel();
  }
}

/**
 * 重新计算张量的缓存元素数量。在修改尺寸时调用此函数。
 * 仅适用于使用稀疏布局的张量，因为只有稀疏张量可能会在计算元素数量时发生整数溢出。
 */
void safe_refresh_numel() {
  if (has_symbolic_sizes_strides_) {
    // 注意：符号元素数量是使用符号整数完成的，它会处理溢出检查
    symbolic_shape_meta().refresh_numel();
  } else {
    // 否则调用安全计算元素数量的函数
    numel_ = safe_compute_numel();
  }
  ```

  // NB: the TypeId argument prevents confusion where you pass a true/false
  // literal and pick the wrong overload
  ```py

  void _set_is_contiguous(identity<bool>, bool b) {
    is_contiguous_ = b;
  }
  ```

  void _set_is_channels_last_contiguous(identity<bool>, bool b) {
    is_channels_last_contiguous_ = b;
  }
  ```py

  void _set_is_channels_last_3d_contiguous(identity<bool>, bool b) {
    is_channels_last_3d_contiguous_ = b;
  }
  ```

  void _set_is_channels_last(identity<bool>, bool b) {
    is_channels_last_ = b;
  }
  ```py

  void _set_is_channels_last_3d(identity<bool>, bool b) {
    is_channels_last_3d_ = b;
  }
  ```

  void _set_is_non_overlapping_and_dense(identity<bool>, bool b) {
    is_non_overlapping_and_dense_ = b;
  }
  ```py

  // These are little wrappers over the real compute_ functions that
  // can make use of other contiguity fields to short circuit.
  ```

  bool compute_is_non_overlapping_and_dense_dim4(identity<bool> type_id) {
    return is_contiguous_ || is_channels_last_contiguous_ ||
        compute_non_overlapping_and_dense(type_id);
  }
  ```py

  bool compute_channels_last_contiguous_3d_dim5(identity<bool> type_id) {
    return !is_channels_last_contiguous_ &&
        compute_channels_last_contiguous_3d(type_id);
  }
  ```

  bool compute_channels_last_2d_dim5(identity<bool> type_id) {
    return !is_channels_last_3d_contiguous_ &&
        compute_strides_like_channels_last_2d(type_id);
  }
  ```py

  bool compute_channels_last_3d_dim5(identity<bool> type_id) {
    return !is_channels_last_ && compute_strides_like_channels_last_3d(type_id);
  }
  ```

  bool compute_is_non_overlapping_and_dense_dim5(identity<bool> type_id) {
    return is_contiguous_ || is_channels_last_contiguous_ ||
        is_channels_last_3d_contiguous_ ||
        compute_non_overlapping_and_dense(type_id);
  }
  ```py

  bool compute_is_non_overlapping_and_dense_anydim(identity<bool> type_id) {
    return is_contiguous_ || compute_non_overlapping_and_dense(type_id);
  }
  ```

  template <typename T>
  void _refresh_contiguous() {
    auto type_id = identity<T>();
    // Note:
    // Dim 0, 1, 2 will never be a channels last 2d/3d format
    // Dim 3+ is possibly be a channels last 2d format (Dim 4 only at this
    // point) Dim 4+ is possibly be a channels last 3d format (Dim 5 only at
    // this point)
  ```py
    switch (dim()) {
      case 4: {
        // 设置是否是连续存储的标记为 true，并计算是否是通道最后维度连续的标记
        _set_is_contiguous(type_id, compute_contiguous(type_id));
        _set_is_channels_last_contiguous(
            type_id, compute_channels_last_contiguous_2d(type_id));
        // 3D 张量不支持通道最后维度连续，设置为 false
        _set_is_channels_last_3d_contiguous(type_id, false);
        // 根据 2D 张量计算通道最后维度的标记
        _set_is_channels_last(
            type_id, compute_strides_like_channels_last_2d(type_id));
        // 3D 张量不支持通道最后维度，设置为 false
        _set_is_channels_last_3d(type_id, false);
        // 计算是否是非重叠且稠密的标记
        _set_is_non_overlapping_and_dense(
            type_id, compute_is_non_overlapping_and_dense_dim4(type_id));
        break;
      }
      case 5: {
        // 设置是否是连续存储的标记为 true，并计算是否是通道最后维度连续的标记
        _set_is_contiguous(type_id, compute_contiguous(type_id));
        _set_is_channels_last_contiguous(
            type_id, compute_channels_last_contiguous_2d(type_id));
        // 计算 3D 张量在第五维度上的通道最后维度连续标记
        _set_is_channels_last_3d_contiguous(
            type_id, compute_channels_last_contiguous_3d_dim5(type_id));
        // 计算第五维度上的 2D 张量通道最后维度的标记
        _set_is_channels_last(type_id, compute_channels_last_2d_dim5(type_id));
        // 计算第五维度上的 3D 张量通道最后维度的标记
        _set_is_channels_last_3d(
            type_id, compute_channels_last_3d_dim5(type_id));
        // 计算是否是非重叠且稠密的标记
        _set_is_non_overlapping_and_dense(
            type_id, compute_is_non_overlapping_and_dense_dim5(type_id));
        break;
      }
      default:
        // 默认情况下，不使用通道最后维度存储格式
        // is_channels_last_ 和 is_channels_last_3d_ 是推荐的内存格式。
        // 只有通道最后维度连续不一定意味着张量是以通道最后维度的方式分布的：
        // 通道维度上的步长可能建议期望的内存布局，但它不影响内存存储。
        // 设置是否是连续存储的标记为 true
        _set_is_contiguous(type_id, compute_contiguous(type_id));
        // 不使用通道最后维度连续存储格式
        _set_is_channels_last_contiguous(type_id, false);
        // 3D 张量不支持通道最后维度连续存储格式
        _set_is_channels_last_3d_contiguous(type_id, false);
        // 不使用通道最后维度存储格式
        _set_is_channels_last(type_id, false);
        // 3D 张量不支持通道最后维度存储格式
        _set_is_channels_last_3d(type_id, false);
        // 计算是否是任意维度上的非重叠且稠密的标记
        _set_is_non_overlapping_and_dense(
            type_id, compute_is_non_overlapping_and_dense_anydim(type_id));
        break;
    }
  }

 protected:
  /**
   * 重新计算张量的缓存连续性。如果修改了大小或步长，请调用此函数。
   */
  void refresh_contiguous() {
    if (has_symbolic_sizes_strides_) {
      // 如果具有符号大小和步长，则刷新符号形状的连续性
      symbolic_shape_meta().refresh_contiguous();
    } else {
      // 否则，调用泛型版本的 _refresh_contiguous 函数
      _refresh_contiguous<bool>();
    }
  }
  // 结束了上述代码段的类定义

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change);
  // 静态方法：从一个 TensorImpl 复制张量元数据字段到另一个 TensorImpl

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change);
  // 静态方法：从一个 TensorImpl 复制张量元数据字段到另一个 TensorImpl

 private:
  static void copy_tensor_metadata_except_version_counter(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      bool allow_tensor_metadata_change);
  // 私有静态方法：除了版本计数器之外，从一个 TensorImpl 复制张量元数据字段到另一个 TensorImpl

 protected:
  // Error message to show when the user tries to change tensor metadata on
  // Tensor created from .data or .detach().
  //
  // See NOTE [ Metadata Change for a Detached Tensor ] for details.
  static const char* const err_msg_tensor_metadata_change_not_allowed;
  // 静态常量字符串：当用户试图在从 .data 或 .detach() 创建的 Tensor 上更改张量元数据时显示的错误消息

  static void copy_generic_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl);
  // 静态方法：从一个 TensorImpl 复制通用张量元数据字段到另一个 TensorImpl

 public:
  void set_storage_access_should_throw() {
    storage_access_should_throw_ = true;
  }
  // 设置方法：设置 storage_access_should_throw_ 标志为 true

 public:
  void set_custom_sizes_strides(SizesStridesPolicy policy) {
    custom_sizes_strides_ = static_cast<uint8_t>(policy);
    refresh_sizes_strides_policy();
  }
  // 公共方法：设置自定义大小和步幅策略，刷新大小和步幅策略

  void set_python_custom_sizes_strides(SizesStridesPolicy policy) {
    python_custom_sizes_strides_ = static_cast<uint8_t>(policy);
    refresh_sizes_strides_policy();
  }
  // 公共方法：设置 Python 自定义大小和步幅策略，刷新大小和步幅策略

  void set_custom_device(bool custom_device) {
    custom_device_ = custom_device;
    refresh_device_policy();
  }
  // 公共方法：设置自定义设备标志，刷新设备策略

  void set_custom_layout(bool custom_layout) {
    custom_layout_ = custom_layout;
    refresh_layout_policy();
  }
  // 公共方法：设置自定义布局标志，刷新布局策略

  void set_python_custom_device(bool custom_device) {
    python_custom_device_ = custom_device;
    refresh_device_policy();
  }
  // 公共方法：设置 Python 自定义设备标志，刷新设备策略

  void set_python_custom_layout(bool custom_layout) {
    python_custom_layout_ = custom_layout;
    refresh_layout_policy();
  }
  // 公共方法：设置 Python 自定义布局标志，刷新布局策略

 protected:
  void refresh_sizes_strides_policy() {
    if (has_symbolic_sizes_strides_) {
      sizes_strides_policy_ =
          static_cast<uint8_t>(SizesStridesPolicy::CustomSizes);
    } else {
      sizes_strides_policy_ =
          std::max(custom_sizes_strides_, python_custom_sizes_strides_);
    }
  }
  // 保护方法：刷新大小和步幅策略，根据是否具有符号大小和步幅进行设置

  void refresh_device_policy() {
    device_policy_ = custom_device_ || python_custom_device_;
  }
  // 保护方法：刷新设备策略，根据自定义设备标志和 Python 自定义设备标志进行设置

  void refresh_layout_policy() {
    // 将 custom_layout_ 或者 python_custom_layout_ 赋值给 layout_policy_
    layout_policy_ = custom_layout_ || python_custom_layout_;
  }

 protected:
  // 存储数据的成员变量
  Storage storage_;

 private:
  // autograd_meta_ 指针指向 AutogradMeta 结构体，存储自动求导相关的字段，
  // 如 grad_ / grad_fn_ / grad_accumulator_。该指针始终唯一拥有它
  // （即一次只能有一个 TensorImpl 拥有它）。

  // autograd_meta_ 可能为 nullptr，作为一种优化。当为 nullptr 时，
  // 相当于 autograd_meta_ 指向一个默认构造的 AutogradMeta；
  // 直观上，不需要梯度的张量会将此字段设置为 null。

  // 这意味着 autograd_meta_ 上的访问器必须小心检查是否为 nullptr，
  // 并在这种情况下适当处理默认行为。

  // 注意，我们不强制实施这样的不变量：如果 AutogradMeta 是默认构造的，
  // 那么它就是 nullptr（要实现这一点，我们必须不断检查 AutogradMeta
  // 是否通过突变变得等同于默认构造形式。这可能有用，但似乎很少有需要
  // requires_grad=True 的变量会变回 requires_grad=False 的版本）。
  // 因此，有三种可表示的状态：
  //
  //    1. autograd_meta_ == nullptr
  //    2. autograd_meta_ 是默认构造的（语义上与（1）相同）
  //    3. autograd_meta_ 包含非平凡信息内容
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

 protected:
  // 额外元数据的唯一指针
  std::unique_ptr<c10::ExtraMeta> extra_meta_ = nullptr;

  // 变量版本计数器
  c10::VariableVersion version_counter_;

  // Python 对象插槽
  impl::PyObjectSlot pyobj_slot_;

  // 大小和步长
  c10::impl::SizesAndStrides sizes_and_strides_;

  // 存储偏移量，默认为 0
  int64_t storage_offset_ = 0;

  // 如果大小和步长为空，则元素数量为 1！但大多数情况下，我们会立即将大小设置为 {0} 并将 numel_ 重置为 0。
  // （不能在默认初始化程序中执行此操作，因为没有办法为 strides_ 拼写“分配一个单元素数组”）。
  int64_t numel_ = 1;

  // 不变量：当存储非空时，此类型元必须与存储中的类型元一致
  caffe2::TypeMeta data_type_;

  // 注意 [CUDA 中的 std::optional 操作符使用]
  // 我们的 optional 定义在 .cu 文件中如果使用 `value()` 或 `operator->` 将无法编译通过。
  // 相反，我们总是使用 `operator*`。详细信息请参见 https://github.com/pytorch/pytorch/issues/18496 。
  // 如果这太难以维护，我们可以简单地使用额外的 bool 手动实现此操作。

  // 不变量：当存储非空时，此设备必须与存储中的类型元一致。
  //
  // 不变量：device_opt_ 仅在未定义的张量（即没有设备的张量）时为 nullopt。
  std::optional<c10::Device> device_opt_;

  // 位域的默认成员初始化器仅适用于 -std=c++2a 或 -std=gnu++2a
  inline void init_bitfields() {
    is_contiguous_ = true;
    is_channels_last_ = false;
    is_channels_last_contiguous_ = false;
    # 初始化一个布尔变量，表示是否在3D张量中通道维度是最后一个维度
    is_channels_last_3d_ = false;
    
    # 初始化一个布尔变量，表示是否3D张量在内存中是连续的
    is_channels_last_3d_contiguous_ = false;
    
    # 初始化一个布尔变量，表示张量是否是非重叠且密集的
    is_non_overlapping_and_dense_ = true;
    
    # 初始化一个布尔变量，表示数字是否被包装
    is_wrapped_number_ = false;
    
    # 初始化一个布尔变量，表示是否允许更改张量的元数据
    allow_tensor_metadata_change_ = true;
    
    # 初始化一个布尔变量，保留字段，暂时设为假
    reserved_ = false;
    
    # 设定一个枚举值，表示尺寸和步长策略的默认设置
    sizes_strides_policy_ = static_cast<uint8_t>(SizesStridesPolicy::Default);
    
    # 设定一个枚举值，表示自定义尺寸和步长策略的默认设置
    custom_sizes_strides_ = static_cast<uint8_t>(SizesStridesPolicy::Default);
    
    # 设定一个枚举值，表示Python自定义尺寸和步长策略的默认设置
    python_custom_sizes_strides_ = static_cast<uint8_t>(SizesStridesPolicy::Default);
    
    # 初始化一个布尔变量，表示是否使用了Python自定义设备
    python_custom_device_ = false;
    
    # 初始化一个布尔变量，表示是否使用了Python自定义布局
    python_custom_layout_ = false;
    
    # 初始化一个布尔变量，表示是否使用了自定义设备
    custom_device_ = false;
    
    # 初始化一个布尔变量，表示是否使用了自定义布局
    custom_layout_ = false;
    
    # 初始化一个布尔变量，表示设备策略，暂时设为假
    device_policy_ = false;
    
    # 初始化一个布尔变量，表示布局策略，暂时设为假
    layout_policy_ = false;
    
    # 初始化一个布尔变量，表示存储访问时是否应该抛出异常，暂时设为假
    storage_access_should_throw_ = false;
};

// Note [TensorImpl size constraints]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Changed the size of TensorImpl?  If the size went down, good for
// you!  Adjust the documentation below and the expected size.
// Did it go up?  Read on...
//
// Struct size matters.  In some production systems at Facebook, we have
// 400M live tensors during a training run.  Do the math: every 64-bit
// word you add to Tensor is an extra 3.2 gigabytes in RAM.
//
// If you are a Facebook employee, you can check if the run in question
// has tipped you over the point using the command here:
// https://fburl.com/q5enpv98
//
// For reference, we OOMed at 160 bytes (20 words) per TensorImpl.
// This is not counting overhead from strides out-of-line allocation and
// StorageImpl space and this is from before we inlined sizes and strides
// directly into TensorImpl as SmallVectors.
//
// Our memory usage on 32-bit systems is suboptimal, but we're not checking
// for it at the moment (to help avoid rage inducing cycles when the
// 32-bit number is wrong).
//
// Current breakdown:
//
//    vtable pointer
//    strong refcount           TODO: pack these into one word
//    weak refcount
//    storage pointer
//    autograd metadata pointer
//    named tensor metadata pointer
//    version counter pointer
//    PyObjectSlot
//    SizesAndStrides size/pointer
//    SizesAndStrides sizes (pre-allocated 0)
//    SizesAndStrides sizes (pre-allocated 1)
//    SizesAndStrides sizes (pre-allocated 2)
//    SizesAndStrides sizes (pre-allocated 3)
//    SizesAndStrides sizes (pre-allocated 4)
//    SizesAndStrides strides (pre-allocated 0)
//    SizesAndStrides strides (pre-allocated 1)
//    SizesAndStrides strides (pre-allocated 2)
//    SizesAndStrides strides (pre-allocated 3)
//    SizesAndStrides strides (pre-allocated 4)
//    storage offset
//    numel
//    data type, device, is_contiguous, storage_access_should_throw_, bitfields
//    DispatchKeySet
//

// Various preprocessor macros we use to check that the
// TensorImpl size hasn't changed unexpectedly. We undef
// these later.
#ifndef __NVCC__
#define C10_NVCC 0
#else
#define C10_NVCC __NVCC__
#endif

#ifndef __CUDA_VER_MAJOR__
#define C10_CUDA_VERSION_MAJOR 0
#else
#define C10_CUDA_VERSION_MAJOR __CUDA_VER_MAJOR__
#endif

#ifndef CUDA_VERSION
#define C10_CUDA_VERSION 0
#else
#define C10_CUDA_VERSION CUDA_VERSION
#endif

#ifndef __clang_major__
#define C10_CLANG_MAJOR_VERSION 0
#else
#define C10_CLANG_MAJOR_VERSION __clang_major__
#endif

#ifndef __GNUC__
#define C10_GCC_VERSION 0
#else
#define C10_GCC_VERSION __GNUC__
#endif

#ifndef __GNUC_MINOR__
#define C10_GCC_VERSION_MINOR 0
#else
#define C10_GCC_VERSION_MINOR __GNUC_MINOR__
#endif

// We use a templatized class to both contain the logic of checking the sizes
// as well as to provide compile-time information that might be useful in
// figuring out why sizes may have changed.
// All the compile time information is given by the template fields that are


这段代码是一系列注释和预处理器宏定义，用于检查和记录 `TensorImpl` 结构体的大小约束和相关信息。
// always printed by the compiler when the static_assert fails.
template <
    size_t cplusplus = __cplusplus,
    size_t clang_ver_major = C10_CLANG_MAJOR_VERSION,
    size_t gcc_ver = C10_GCC_VERSION,
    size_t gcc_ver_minor = C10_GCC_VERSION_MINOR,
    size_t nvcc = C10_NVCC,
    size_t cuda_version = C10_CUDA_VERSION,
    size_t cuda_version_major = C10_CUDA_VERSION_MAJOR,
    size_t ptr_size = sizeof(void*)>
class C10_TensorImpl_Size_Check_Dummy_Class : private TensorImpl {
  // Names of (non-bitfield) fields in TensorImpl; used to provide
  // compile-time info about fields whose size changes unexpectedly.
  enum class FieldNameEnum {
    storage_,               // Size of storage_ field in TensorImpl
    autograd_meta_,         // Size of autograd_meta_ field in TensorImpl
    extra_meta_,            // Size of extra_meta_ field in TensorImpl
    version_counter_,       // Size of version_counter_ field in TensorImpl
    pyobj_slot_,            // Size of pyobj_slot_ field in TensorImpl
    sizes_and_strides_,     // Size of sizes_and_strides_ field in TensorImpl
    storage_offset_,        // Size of storage_offset_ field in TensorImpl
    numel_,                 // Size of numel_ field in TensorImpl
    data_type_,             // Size of data_type_ field in TensorImpl (not used in this snippet)
    device_opt_,            // Size of device_opt_ field in TensorImpl (not used in this snippet)
    key_set_,               // Size of key_set_ field in TensorImpl (not used in this snippet)
    TOTAL_SIZE              // Total number of fields in TensorImpl
  };

  // Provides compile-time equality check that reveals what numbers
  // were used and on which quantity
  template <size_t Actual, size_t Expected, FieldNameEnum FiledName>
  constexpr static bool are_equal() {
    static_assert(
        Actual == Expected,
        "Actual and Expected sizes of a field did not match!");
    return true;
  }

  // Provides compile-time <= check that reveals what numbers
  // were used and on which quantity
  template <size_t Actual, size_t Expected, FieldNameEnum FiledName>
  constexpr static bool is_le() {
    static_assert(
        Actual <= Expected,
        "Actual and Expected sizes of a field did not match!");
    return true;
  }

 public:
  // Compile-time check that TensorImpl field sizes are as expected
  //
  // Observed total sizes and associated versions
  // If you find a flag that predicts when unique_ptr has 16 bytes
  // on 64-bit systems or when sizes_and_strides_ is 84 vs 88 bytes
  // on 32-bit systems you get a cookie!
  // Length | LLVM | GCC  |    C++ |  CUDA
  //    192 |    ? | 11.2 | 201703 | 11040
  //    208 |    ? | 11.2 | 201703 | 11040
  //    208 |    ? | 11.2 | 201402 | 11040
  //    192 |    ? | 11.2 | 201402 | 11040
  //    160 |   12 |  4.2 | 201703 |     0
  //
  // To keep things clean, we split on systems here.

#if UINTPTR_MAX == 0xFFFFFFFF
  // This is a 32-bit system
  static constexpr bool check_sizes() {
    constexpr size_t tsize = 20 * sizeof(int64_t);

    // clang-format off
    are_equal<sizeof(storage_),            4,  FieldNameEnum::storage_>();            // Check size of storage_
    are_equal<sizeof(autograd_meta_),      4,  FieldNameEnum::autograd_meta_>();      // Check size of autograd_meta_
    are_equal<sizeof(extra_meta_),         4,  FieldNameEnum::extra_meta_>();         // Check size of extra_meta_
    are_equal<sizeof(version_counter_),    4,  FieldNameEnum::version_counter_>();    // Check size of version_counter_
    are_equal<sizeof(pyobj_slot_),         8,  FieldNameEnum::pyobj_slot_>();         // Check size of pyobj_slot_
    is_le<sizeof(sizes_and_strides_),     88, FieldNameEnum::sizes_and_strides_>();   // Check size of sizes_and_strides_
    are_equal<sizeof(storage_offset_),     8,  FieldNameEnum::storage_offset_>();     // Check size of storage_offset_
    are_equal<sizeof(numel_),              8,  FieldNameEnum::numel_>();              // Check size of numel_
    // clang-format on
    // 检查 data_type_ 字段的大小是否等于 2，返回比较结果
    are_equal<sizeof(data_type_),          2,  FieldNameEnum::data_type_>();
    // 检查 device_opt_ 字段的大小是否等于 3，返回比较结果
    are_equal<sizeof(device_opt_),         3,  FieldNameEnum::device_opt_>();
    // 检查 key_set_ 字段的大小是否等于 8，返回比较结果
    are_equal<sizeof(key_set_),            8,  FieldNameEnum::key_set_>();
    // 检查 TensorImpl 类的大小是否小于或等于 tsize，以确定大小端顺序，返回比较结果
    is_le<sizeof(TensorImpl),          tsize,  FieldNameEnum::TOTAL_SIZE>();
    // 结束 clang-format 的格式化

    // 返回 true，表示所有字段大小比较和大小端检查都通过
    return true;
#else
  // This is a 64-bit system
  // 定义一个静态的 constexpr 函数用于检查各个字段的大小是否符合预期
  static constexpr bool check_sizes() {
    // 定义一个常量，表示所有字段大小之和的期望值
    constexpr size_t tsize = 26 * sizeof(int64_t);

    // clang-format off
    // 检查 storage_ 字段的大小是否为 8 字节
    are_equal<sizeof(storage_),            8,  FieldNameEnum::storage_>();
    // 对于某些包含 NVCC 的系统，unique_ptr 的大小可能是 16 字节
    // 尚未找到宏预处理器检测这些系统的方法，因此采用 <= 比较
    // 检查 autograd_meta_ 字段的大小是否为 16 字节
    is_le<sizeof(autograd_meta_),         16,  FieldNameEnum::autograd_meta_>();
    // 检查 extra_meta_ 字段的大小是否为 16 字节
    is_le<sizeof(extra_meta_),            16,  FieldNameEnum::extra_meta_>();
    // 检查 version_counter_ 字段的大小是否为 8 字节
    are_equal<sizeof(version_counter_),    8,  FieldNameEnum::version_counter_>();
    // 检查 pyobj_slot_ 字段的大小是否为 16 字节
    are_equal<sizeof(pyobj_slot_),   16,  FieldNameEnum::pyobj_slot_>();
    // 检查 sizes_and_strides_ 字段的大小是否为 88 字节
    are_equal<sizeof(sizes_and_strides_), 88,  FieldNameEnum::sizes_and_strides_>();
    // 检查 storage_offset_ 字段的大小是否为 8 字节
    are_equal<sizeof(storage_offset_),     8,  FieldNameEnum::storage_offset_>();
    // 检查 numel_ 字段的大小是否为 8 字节
    are_equal<sizeof(numel_),              8,  FieldNameEnum::numel_>();
    // 检查 data_type_ 字段的大小是否为 2 字节
    are_equal<sizeof(data_type_),          2,  FieldNameEnum::data_type_>();
    // 检查 device_opt_ 字段的大小是否为 3 字节
    are_equal<sizeof(device_opt_),         3,  FieldNameEnum::device_opt_>();
    // 检查 key_set_ 字段的大小是否为 8 字节
    are_equal<sizeof(key_set_),            8,  FieldNameEnum::key_set_>();
    // 检查 TensorImpl 结构体的总大小是否符合预期值 tsize
    is_le<sizeof(TensorImpl),          tsize,  FieldNameEnum::TOTAL_SIZE>();
    // clang-format on

    // 返回 true，表示检查通过
    return true;
  }
#endif
};

// 使用一个类来封装大小检查的逻辑，使用模板来捕获大小和标志
// 我们在静态断言中调用这个类，以证明没有运行时行为
// 由于我们调用的方法要么返回 true，要么失败自己的静态断言，
// 我们不应该看到下面的错误消息。然而，对于 C++ <17，我们必须提供它们。
static_assert(
    C10_TensorImpl_Size_Check_Dummy_Class<>::check_sizes(),
    "You should not see this message.");

// 清理我们之前定义的宏
#undef C10_NVCC
#undef C10_CUDA_VERSION_MAJOR
#undef C10_CUDA_VERSION
#undef C10_CLANG_MAJOR_VERSION
#undef C10_GCC_VERSION
#undef C10_GCC_VERSION_MINOR

} // namespace c10
```