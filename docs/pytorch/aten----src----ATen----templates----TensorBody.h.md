# `.\pytorch\aten\src\ATen\templates\TensorBody.h`

```py
#pragma once

#ifdef TORCH_ASSERT_NO_OPERATORS
// 如果定义了 TORCH_ASSERT_NO_OPERATORS 宏，则输出错误信息并终止编译
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/Stream.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/Exception.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Deprecated.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/macros/Export.h>
#include <ATen/core/CheckMemoryFormat.h>
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/QuantizerBase.h>
#include <c10/core/SymInt.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/core/TensorBase.h>


#include <ATen/MethodOperators.h>

namespace c10{
// 前置声明 List 和 IListRef 模板类
template<class T> class List;
template<class T> class IListRef;
}
namespace at {
// 前置声明 Generator 和 Type 结构体，以及 DeprecatedTypeProperties 类和 Tensor 类
struct Generator;
struct Type;
class DeprecatedTypeProperties;
class Tensor;
} // namespace at
namespace at {
namespace indexing {
// 前置声明 TensorIndex 结构体
struct TensorIndex;
} // namespace indexing
} // namespace at

namespace torch { namespace autograd {
// 前置声明 Node 结构体
struct Node;

}} // namespace torch::autograd

namespace at {
// 前置声明 OptionalTensorRef 和 TensorRef 类，以及 Tensor 类型别名 TensorList 和 ITensorList
class OptionalTensorRef;
class TensorRef;
class Tensor;
using TensorList = ArrayRef<Tensor>;
using ITensorList = c10::IListRef<Tensor>;

using Stream = c10::Stream;

// Tensor 是一个“通用”对象，持有指向底层 TensorImpl 对象的指针，该对象具有嵌入式引用计数。
//
// 例如：
//
// void func(Tensor a) {
//   Tensor b = a;
//   ...
// }
//
// 在这个例子中，当我们说 Tensor b = a 时，我们创建一个指向相同底层 TensorImpl 的新对象，并增加其引用计数。
// 当 b 超出作用域时，析构函数通过调用 TensorImpl 的 release() 减少引用计数。现有的构造函数、运算符重载等保证了正确的语义。
//
// 注意，Tensor 也可以是 NULL，即它没有关联到任何底层的 TensorImpl，因此必须特别小心处理这种情况。
class TORCH_API Tensor : public TensorBase {
 protected:
  // Create a Tensor with a +0 reference count. Special care must be
  // taken to avoid decrementing this reference count at destruction
  // time. Intended to support MaybeOwnedTraits<Tensor>.
  // 使用 unsafe_borrow_t 和给定的 TensorBase 对象创建一个 Tensor 对象，
  // 其引用计数初始化为 +0。在析构时需要特别注意避免减少此引用计数。
  explicit Tensor(unsafe_borrow_t, const TensorBase& rhs): TensorBase(unsafe_borrow_t{}, rhs) {}
  friend MaybeOwnedTraits<Tensor>;
  friend OptionalTensorRef;
  friend TensorRef;

 public:
  // 默认构造函数，默认初始化一个 Tensor 对象
  Tensor() = default;

  // 这个构造函数不应该被最终用户使用，是由自动生成的代码调用的一个实现细节
  // 使用给定的 TensorImpl 指针构造一个 Tensor 对象，同时初始化为 +0 引用计数
  explicit Tensor(
      c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
      : TensorBase(std::move(tensor_impl)) {}

  // 拷贝构造函数，使用另一个 Tensor 对象初始化新的 Tensor 对象
  Tensor(const Tensor &tensor) = default;

  // 移动构造函数，使用另一个 Tensor 对象初始化新的 Tensor 对象
  Tensor(Tensor &&tensor) = default;

  // 从 TensorBase 隐式移动构造，但必须显式调用以增加引用计数
  // 使用给定的 TensorBase 对象初始化一个 Tensor 对象
  explicit Tensor(const TensorBase &base): TensorBase(base) {}

  // 从 TensorBase 隐式移动构造，但必须显式调用以增加引用计数
  // 使用给定的 TensorBase 对象初始化一个 Tensor 对象
  /*implicit*/ Tensor(TensorBase &&base): TensorBase(std::move(base)) {}

  // 从 TensorImpl 创建一个新的 Tensor 包装器。
  // 这是一个自由方法，应谨慎使用，检查必要的不变量
  static Tensor wrap_tensor_impl(
      c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl) {
    return TensorBase::wrap_tensor_impl(std::move(tensor_impl));
  }

  // 返回一个连续存储的 Tensor 对象，可以指定内存格式
  Tensor contiguous(MemoryFormat memory_format=MemoryFormat::Contiguous) const {
    return TensorBase::contiguous(memory_format);
  }

  // 返回共轭 Tensor 对象。如果当前对象不是复数类型，返回自身。
  // 根据布局类型不同，返回不同的共轭处理结果
  Tensor conj() const {
    if (!this->is_complex()) {
      return *this;
    }

    switch (this->layout()) {
      // 对于稀疏类型的 Tensor，返回物理上的共轭结果
      case at::kSparse:
      case at::kSparseCsr:
      case at::kSparseCsc:
      case at::kSparseBsr:
      case at::kSparseBsc:
        return this->conj_physical();
      // 默认情况下，返回逻辑上的共轭处理结果
      default:
        return this->_conj();
    }
  }
  // Unfortunately, we have to write these constructors out manually
  // to work around an MSVC bug:
  //    error C2580: 'at::Tensor &at::Tensor::operator =(const at::Tensor &) &':
  //    multiple versions of a defaulted special member functions are not allowed
  // Tensor& operator=(const Tensor&) & = default;
  // Tensor& operator=(Tensor&&) & = default;

  // Also MSVC will wrongly issue the following warning with the aforementioned fix
  //    warning C4522: 'at::Tensor': multiple assignment operators specified
  // Let's just skip the warning.
  //
  // TODO: temporarily disabled

  // Assignment operator overload for assigning from a const TensorBase reference
  // to this Tensor object. It sets the implementation pointer from x.
  Tensor& operator=(const TensorBase& x) & {
    impl_ = x.getIntrusivePtr();
    return *this;
  }

  // Move assignment operator overload for assigning from a TensorBase rvalue reference
  // to this Tensor object. It releases the intrusive pointer from x and sets it as
  // the implementation pointer of this Tensor.
  Tensor& operator=(TensorBase&& x) & noexcept {
    impl_ = x.unsafeReleaseIntrusivePtr();
    return *this;
  }

  // Assignment operator overload for assigning from a const Tensor reference
  // to this Tensor object. It forwards the assignment to the TensorBase overload.
  Tensor& operator=(const Tensor &x) & {
    return operator=(static_cast<const TensorBase&>(x));
  }

  // Move assignment operator overload for assigning from a Tensor rvalue reference
  // to this Tensor object. It forwards the assignment to the TensorBase overload.
  Tensor& operator=(Tensor &&x) & noexcept {
    return operator=(static_cast<TensorBase&&>(x));
  }

  // Move assignment operator overload for assigning from a Scalar rvalue reference
  // to this Tensor object. This method is intended to be used in contexts where
  // assignment from a Scalar is expected to be performed efficiently.
  Tensor& operator=(const Scalar &v) && {
      // Implementation of assignment from Scalar to Tensor...
      // (Implementation details are not provided in the commented code)
  }
  // 返回值为调用 fill_ 方法填充后的对象
  return fill_(v);
}

// 按照移动语义赋值运算符，从 rhs 复制数据到当前对象
Tensor& operator=(const Tensor &rhs) && {
  return copy_(rhs);
}

// 按照移动语义赋值运算符，从 rhs 移动数据到当前对象
Tensor& operator=(Tensor&& rhs) && {
  return copy_(rhs);
}

// 返回已弃用的类型属性对象，提供兼容旧代码的接口
C10_DEPRECATED_MESSAGE("Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device().")
DeprecatedTypeProperties & type() const {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      dispatchKeyToBackend(legacyExtractDispatchKey(key_set())),
      scalar_type());
}

// 返回转换后的类型为 t 的新 Tensor 对象
Tensor toType(ScalarType t) const {
  return to(options().dtype(t), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: Deprecate me
// 返回转换后的后端为 b 的新 Tensor 对象
Tensor toBackend(Backend b) const {
  return to(options().device(backendToDeviceType(b)).layout(layout_from_backend(b)), /*non_blocking*/ false, /*copy*/ false);
}

// 返回是否为变量（已弃用）
C10_DEPRECATED_MESSAGE("Tensor.is_variable() is deprecated; everything is a variable now. (If you want to assert that variable has been appropriately handled already, use at::impl::variable_excluded_from_dispatch())")
bool is_variable() const noexcept {
  return !at::impl::variable_excluded_from_dispatch();
}

// 返回特定类型 T 的数据指针（已弃用）
template<typename T>
C10_DEPRECATED_MESSAGE("Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead.")
T * data() const {
  return data_ptr<T>();
}

// 返回通用的 packed_tensor 访问器（已弃用）
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
C10_DEPRECATED_MESSAGE("packed_accessor is deprecated, use packed_accessor32 or packed_accessor64 instead")
GenericPackedTensorAccessor<T,N,PtrTraits,index_t> packed_accessor() const & {
  return generic_packed_accessor<T,N,PtrTraits,index_t>();
}

// 禁止使用移动语义的 packed_tensor 访问器（已弃用）
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
C10_DEPRECATED_MESSAGE("packed_accessor is deprecated, use packed_accessor32 or packed_accessor64 instead")
GenericPackedTensorAccessor<T,N,PtrTraits,index_t> packed_accessor() && = delete;

// 返回按位取反后的 Tensor 对象
Tensor operator~() const {
  return bitwise_not();
}

// 返回取负后的 Tensor 对象
Tensor operator-() const {
  return neg();
}

// 将另一个 Tensor 对象 other 加到当前对象，返回当前对象的引用
Tensor& operator+=(const Tensor & other) {
  return add_(other);
}

// 将标量 other 加到当前对象，返回当前对象的引用
Tensor& operator+=(const Scalar & other) {
  return add_(other);
}

// 将另一个 Tensor 对象 other 减去当前对象，返回当前对象的引用
Tensor& operator-=(const Tensor & other) {
  return sub_(other);
}

// 将标量 other 减去当前对象，返回当前对象的引用
Tensor& operator-=(const Scalar & other) {
  return sub_(other);
}

// 将另一个 Tensor 对象 other 乘到当前对象，返回当前对象的引用
Tensor& operator*=(const Tensor & other) {
  return mul_(other);
}

// 将标量 other 乘到当前对象，返回当前对象的引用
Tensor& operator*=(const Scalar & other) {
  return mul_(other);
}

// 将当前对象除以另一个 Tensor 对象 other，返回当前对象的引用
Tensor& operator/=(const Tensor & other) {
  return div_(other);
}

// 将当前对象除以标量 other，返回当前对象的引用
Tensor& operator/=(const Scalar & other) {

  return div_(other);
}
  // 返回一个通过调用 div_ 方法计算得到的新张量，用来处理与另一个张量的除法操作
  Tensor return div_(other);
}

// 返回一个通过调用 bitwise_and_ 方法计算得到的当前张量，用来处理与另一个张量的按位与操作
Tensor& operator&=(const Tensor & other) {
  return bitwise_and_(other);
}

// 返回一个通过调用 bitwise_or_ 方法计算得到的当前张量，用来处理与另一个张量的按位或操作
Tensor& operator|=(const Tensor & other) {
  return bitwise_or_(other);
}

// 返回一个通过调用 bitwise_xor_ 方法计算得到的当前张量，用来处理与另一个张量的按位异或操作
Tensor& operator^=(const Tensor & other) {
  return bitwise_xor_(other);
}

// 根据标量索引返回张量的子集，如果索引不是整数，则抛出错误
Tensor operator[](const Scalar & index) const {
  if (!index.isIntegral(false)) {
    TORCH_CHECK_INDEX(false, "Can only index tensors with integral scalars");
  }
  return this->operator[](index.toLong());
}

// 根据张量索引返回张量的子集，检查索引是否已定义和是否为零维张量
Tensor operator[](const Tensor & index) const {
  // Scalar(Tensor) 构造函数是显式的，因此需要显式调用
  // 检查索引张量是否已定义
  if (!index.defined()) {
    TORCH_CHECK_INDEX(false, "Can only index with tensors that are defined");
  }
  // 检查索引张量是否为零维张量
  if (index.dim() != 0) {
    TORCH_CHECK_INDEX(false,
                      "Can only index with tensors that are scalars (zero-dim)");
  }
  // 调用以整数索引的重载运算符[]
  return this->operator[](index.item());
}

// 根据整数索引返回张量的子集，调用 select 方法
Tensor operator[](int64_t index) const {
  return select(0, index);
}

// 使用给定的张量索引数组来进行索引操作
Tensor index(ArrayRef<at::indexing::TensorIndex> indices) const;

// 使用初始化列表中的张量索引进行索引操作
Tensor index(std::initializer_list<at::indexing::TensorIndex> indices) const;

// 使用给定的张量索引数组来进行索引赋值操作
Tensor & index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor const & rhs);

// 使用给定的标量来进行索引赋值操作
Tensor & index_put_(ArrayRef<at::indexing::TensorIndex> indices, const Scalar& v);

// 使用初始化列表中的张量索引进行索引赋值操作
Tensor & index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Tensor const & rhs);

// 使用初始化列表中的标量来进行索引赋值操作
Tensor & index_put_(std::initializer_list<at::indexing::TensorIndex> indices, const Scalar& v);

// 返回一个在 CPU 设备上的张量，通过调用 to 方法实现设备转换
Tensor cpu() const {
  return to(options().device(c10::DeviceType::CPU), /*non_blocking*/ false, /*copy*/ false);
}

// 返回一个在 CUDA 设备上的张量，通过调用 to 方法实现设备转换
// Python 版本接受额外参数，这里的版本未实现该功能
Tensor cuda() const {
  return to(options().device(c10::DeviceType::CUDA), /*non_blocking*/ false, /*copy*/ false);
}

// 返回一个在 HIP 设备上的张量，通过调用 to 方法实现设备转换
Tensor hip() const {
  return to(options().device(c10::DeviceType::HIP), /*non_blocking*/ false, /*copy*/ false);
}

// 返回一个在 VE 设备上的张量，通过调用 to 方法实现设备转换
Tensor ve() const {
  return to(options().device(c10::DeviceType::VE), /*non_blocking*/ false, /*copy*/ false);
}

// 返回一个在 Vulkan 设备上的张量，通过调用 to 方法实现设备转换
Tensor vulkan() const {
  return to(options().device(c10::DeviceType::Vulkan), /*non_blocking*/ false, /*copy*/ false);
}

// 返回一个在 Metal 设备上的张量，通过调用 to 方法实现设备转换
Tensor metal() const {
  return to(options().device(c10::DeviceType::Metal), /*non_blocking*/ false, /*copy*/ false);
}

// 返回当前张量，用于在反向传播函数中调用底层的 _backward 函数
// 该函数是为了处理 'backwards' api 可选输入 'inputs' 参数而添加的包装器
// 由于代码生成当前不支持 TensorList 的可选参数，因此通过此方法间接调用 _backward 函数
Tensor meta() const {
    // 检查输入参数是否有值
    if (inputs.has_value()) {
      // 如果有值，则检查其大小是否大于0，否则抛出异常
      TORCH_CHECK(inputs.value().size() > 0, "'inputs' argument to backward cannot be empty")
      // 调用对象的_backward方法，传入inputs的值、梯度、保留计算图选项和创建计算图选项
      this->_backward(inputs.value(), gradient, retain_graph, create_graph);
    } else {
      // 如果输入参数没有值，则传入一个空的map作为inputs，调用对象的_backward方法
      this->_backward({}, gradient, retain_graph, create_graph);
    }
  }

  /// \fn Tensor detach() const;
  ///
  /// Returns a new Tensor, detached from the current graph.
  /// The result will never require gradient.

  /// \fn Tensor & detach_() const;
  ///
  /// Detaches the Tensor from the graph that created it, making it a leaf.
  /// Views cannot be detached in-place.

  /// \fn void retain_grad() const;
  ///
  /// Enables this Tensor to have their :attr:`grad` populated during
  /// :func:`backward`. This is a no-op for leaf tensors.

  /// \fn bool retains_grad() const;
  ///
  /// Is ``true`` if this Tensor is non-leaf and its :attr:`grad` is enabled to be
  /// populated during :func:`backward`, ``false`` otherwise.

  const Tensor& set_requires_grad(bool requires_grad) const {
    // 调用基类TensorBase的set_requires_grad方法，设置是否需要计算梯度
    TensorBase::set_requires_grad(requires_grad);
    // 返回当前对象的引用
    return *this;
  }

  /// Return a mutable reference to the gradient. This is conventionally
  /// used as `t.grad() = x` to set a gradient to a completely new tensor.
  /// Note that this function work with a non-const Tensor and is not
  /// thread safe.
  Tensor& mutable_grad() const {
    // 返回当前对象实现的mutable_grad方法的结果，用于获取可变的梯度
    return impl_->mutable_grad();
  }

  /// This function returns an undefined tensor by default and returns a defined tensor
  /// the first time a call to `backward()` computes gradients for this Tensor.
  /// The attribute will then contain the gradients computed and future calls
  /// to `backward()` will accumulate (add) gradients into it.
  const Tensor& grad() const {
    // 获取当前Tensor实例的梯度
    const Tensor& maybe_grad = impl_->grad();
    // 如果当前Tensor不是叶子节点，且未保留梯度，且梯度尚未定义，则发出警告
    if (!is_leaf() && !retains_grad() && !maybe_grad.defined()) {
      TORCH_WARN(
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad "
        "attribute won't be populated during autograd.backward(). If you indeed want the .grad "
        "field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. "
        "If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor "
        "instead. See github.com/pytorch/pytorch/pull/30531 for more informations.");
    }
    // 返回获取到的梯度（可能是空的）
    return maybe_grad;
  }

  // The Forward AD API functions below are low level and are not to be used by end
  // users who should use the API provided in torch/csrc/autograd.h

  /// This function returns the forward gradient for this Tensor at the given level.
  const Tensor& _fw_grad(uint64_t level) const {
  // 返回实现对象的前向梯度，使用给定的级别和当前对象
  return impl_->_fw_grad(level, *this);
}

/// 这个函数可以用来设置前向梯度的数值。
/// 需要注意的是，如果给定的新梯度与当前 Tensor 的元数据（大小/步幅/存储偏移量）不同，
/// 则新梯度的内容将被复制到一个新的 Tensor 中。
void _set_fw_grad(const TensorBase& new_grad, uint64_t level, bool is_inplace_op) const {
  // 调用实现对象的设置前向梯度方法，传入新梯度、当前对象、级别和是否原地操作的标志
  impl_->_set_fw_grad(new_grad, *this, level, is_inplace_op);
}

// STOP.  Thinking of adding a method here, which only makes use
// of other ATen methods?  Define it in native_functions.yaml.

//example
//Tensor * add(Tensor & b);
${tensor_method_declarations}

// 专门为像 std() 的函数提供的特殊 C++ 重载（参见 gh-40287）
// 这些重载是必要的，因为 int -> bool 转换优先于 int -> IntArrayRef
// 所以，例如 std(0) 将选择 std(unbiased=False) 的重载

Tensor var(int dim) const {
  // 调用 var(IntArrayRef{dim}) 函数重载
  return var(IntArrayRef{dim});
}

Tensor std(int dim) const {
  // 调用 std(IntArrayRef{dim}) 函数重载
  return std(IntArrayRef{dim});
}

// 我们在 PR #12766 中将 .dtype() 方法改为返回 TypeMeta。理想情况下，我们希望
// at::kDouble 等也是 TypeMeta，但目前尚未实现。
// 在此改变之前，我们通过这个方法来保持 C++ 用法的 BC 兼容性，例如 `x.to(y.dtype)`。
// TODO: 在 at::kDouble 等成为 TypeMeta 后删除以下两个方法。
inline Tensor to(caffe2::TypeMeta type_meta, bool non_blocking=false, bool copy=false) const {
  // 调用 this->to(scalar_type=typeMetaToScalarType(type_meta), non_blocking, copy) 函数重载
  return this->to(/*scalar_type=*/typeMetaToScalarType(type_meta), non_blocking, copy);
}
inline Tensor to(Device device, caffe2::TypeMeta type_meta, bool non_blocking=false, bool copy=false) const {
  // 调用 this->to(device, scalar_type=typeMetaToScalarType(type_meta), non_blocking, copy) 函数重载
  return this->to(device, /*scalar_type=*/typeMetaToScalarType(type_meta), non_blocking, copy);
}

template <typename F, typename... Args>
decltype(auto) m(F func, Args&&... params) const {
  // 调用 func(*this, std::forward<Args>(params)...) 函数
  return func(*this, std::forward<Args>(params)...);
}

/// 注意：这类似于 `Variable` 上的传统 `.data()` 函数，意图是用于访问等效的 `Tensor`
/// （即与 `Variable` 共享相同存储和张量元数据的 `Tensor`）的函数。
///
/// 与传统的 `.data()` 函数的一个显著区别是，返回的 `Tensor` 的张量元数据（例如大小/步幅/存储/存储偏移量）
/// 的更改不会更新原始的 `Variable`，因为这个函数浅复制了 `Variable` 的底层 TensorImpl。
at::Tensor tensor_data() const {
  /// Return the tensor data associated with this object, utilizing TensorBase's tensor_data().
  ///
  /// NOTE: `var.variable_data()` in C++ has the same semantics as `tensor.data`
  /// in Python, which create a new `Variable` that shares the same storage and
  /// tensor metadata with the original `Variable`, but with a completely new
  /// autograd history.
  ///
  /// NOTE: If we change the tensor metadata (e.g. sizes / strides /
  /// storage / storage_offset) of a variable created from `var.variable_data()`, those
  /// changes will not update the original variable `var`. In `.variable_data()`, we set
  /// `allow_tensor_metadata_change_` to false to make such changes explicitly illegal,
  /// in order to prevent users from changing metadata of `var.variable_data()`
  /// and expecting the original variable `var` to also be updated.
  at::Tensor variable_data() const {
    return TensorBase::variable_data();
  }

  // Hooks
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  template <typename T>
  using hook_return_void_t = std::enable_if_t<std::is_void<typename std::invoke_result_t<T&, Tensor>>::value, unsigned>;
  template <typename T>
  using hook_return_var_t = std::enable_if_t<std::is_same<typename std::invoke_result_t<T&, Tensor>, Tensor>::value, unsigned>;

  /// Registers a backward hook.
  ///
  /// The hook will be called every time a gradient with respect to the Tensor is computed.
  /// The hook should have one of the following signature:
  /// ```
  /// hook(Tensor grad) -> Tensor
  /// ```py
  /// ```
  /// hook(Tensor grad) -> void
  /// ```py
  /// The hook should not modify its argument, but it can optionally return a new gradient
  /// which will be used in place of `grad`.
  ///
  /// This function returns the index of the hook in the list which can be used to remove hook.
  ///
  /// Example:
  /// @code
  /// auto v = torch::tensor({0., 0., 0.}, torch::requires_grad());
  /// auto h = v.register_hook([](torch::Tensor grad){ return grad * 2; }); // double the gradient
  /// v.backward(torch::tensor({1., 2., 3.}));
  /// // This prints:
  /// // ```
  /// //  2
  /// //  4
  /// //  6
  /// // [ CPUFloatType{3} ]
  /// // ```py
  /// std::cout << v.grad() << std::endl;
  /// v.remove_hook(h);  // removes the hook
  /// @endcode
  template <typename T>
  hook_return_void_t<T> register_hook(T&& hook) const;

  /// Registers a backward hook that returns a Tensor.
  ///
  /// The hook will be called every time a gradient with respect to the Tensor is computed.
  /// The hook should have the following signature:
  /// ```
  /// hook(Tensor grad) -> Tensor
  /// ```py
  ///
  /// This function returns the index of the hook in the list which can be used to remove hook.
  ///
  /// Example:
  /// @code
  /// auto v = torch::tensor({0., 0., 0.}, torch::requires_grad());
  /// auto h = v.register_hook([](torch::Tensor grad){ return grad * 2; }); // double the gradient
  /// v.backward(torch::tensor({1., 2., 3.}));
  /// // This prints:
  /// // ```
  /// //  2
  /// //  4
  /// //  6
  /// // [ CPUFloatType{3} ]
  /// // ```py
  /// std::cout << v.grad() << std::endl;
  /// v.remove_hook(h);  // removes the hook
  /// @endcode
  template <typename T>
  hook_return_var_t<T> register_hook(T&& hook) const;

  // Variable methods
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Return the tensor data associated with this object, utilizing TensorBase's data().
  Tensor data() const {
    return TensorBase::data();
  }

  /// Perform backward propagation of gradients.
  ///
  /// This function computes the gradients of the inputs with respect to this tensor,
  /// considering optional gradient, graph retention, and graph creation conditions.
  void _backward(TensorList inputs, const std::optional<Tensor>& gradient, std::optional<bool> keep_graph, bool create_graph) const;

  /// Set the flag indicating whether this tensor requires gradient computation.
  ///
  /// This function sets the `requires_grad` flag for this tensor, returning a reference
  /// to the modified tensor.
  const Tensor& requires_grad_(bool _requires_grad=true) const {
    TensorBase::requires_grad_(_requires_grad);
    return *this;
  }
};

namespace detail {
// Helper creator for Tensor class which doesn't requires the users to pass
// in an intrusive_ptr instead it just converts the argument passed to
// requested intrusive_ptr type.
template <typename T, typename... Args>
Tensor make_tensor(Args&&... args) {
  // 创建一个 Tensor 对象，使用 c10::make_intrusive 将参数转换为相应的 intrusive_ptr 类型
  return Tensor(c10::make_intrusive<T>(std::forward<Args>(args)...));
}

} // namespace detail

} // namespace at


namespace at {
${tensor_method_definitions}
} // namespace at


namespace c10 {
template <>
struct MaybeOwnedTraits<at::Tensor> {
  using owned_type = at::Tensor;
  using borrow_type = at::Tensor;

  // 创建一个 borrow_type 对象，通过从 owned_type 中获取的 TensorImpl 进行初始化
  static borrow_type createBorrow(const owned_type& from) {
    // 注意：这里可以不使用特殊的 unsafe_borrow_t Tensor 构造函数来实现，
    // 例如可以使用以下代码代替：
    // return borrow_type(c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(from.unsafeGetTensorImpl()));
    // 但由于在 Tensor(c10::intrusive_ptr<...>) 构造函数中存在 nullptr 检查，这会影响内联优化。
    // 由于 from 是有效的 Tensor，我们已经知道 from.impl_ 不会是 null，因此无需再次进行检查。
    // （使用 __builtin_assume 可以避免这种情况，但不适用于 MSVC。）
    return borrow_type(borrow_type::unsafe_borrow_t{}, from);
  }

  // 将 rhs 中的值赋给 lhs
  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.unsafeReleaseTensorImpl();
    // 参见上述注释：可以使用与 createBorrow() 类似的公共 API 来实现，但这会影响内联优化。
    lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
  }

  // 销毁 borrow 对象时释放 TensorImpl，但它已经是 +0。
  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.unsafeReleaseTensorImpl(); // "leak" it, but it was already +0.
  }

  // 从 borrow 中获取引用的 owned_type
  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  // 从 borrow 中获取指向 owned_type 的指针
  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  // 调试用：检查 borrow 对象是否有效
  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
    return true;
  }
};

template <>
struct ExclusivelyOwnedTraits<at::Tensor> {
  using repr_type = at::Tensor;
  using pointer_type = at::Tensor*;
  using const_pointer_type = const at::Tensor*;

  // 返回一个空的 repr_type 对象
  static repr_type nullRepr() {
    return at::Tensor();
  }

  // 在原地创建 repr_type 对象
  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return at::Tensor(std::forward<Args>(args)...);
  }

  // 将 x 移动到 repr_type 中
  static repr_type moveToRepr(at::Tensor&& x) {
    return std::move(x);
  }

  // 销毁 owned 对象
  static void destroyOwned(at::Tensor& x) {
    return ExclusivelyOwnedTraits<at::TensorBase>::destroyOwned(x);
  }

  // 获取 repr_type 对象的实现指针
  static pointer_type getImpl(repr_type& x) {
    return &x;
  }

  // 获取 repr_type 对象的常量实现指针
  static const_pointer_type getImpl(const repr_type& x) {
    return &x;
  }
};
} // namespace c10

namespace at {

// 从可选的 Tensor 中借用一个 c10::MaybeOwned<Tensor> 对象
inline c10::MaybeOwned<Tensor> borrow_from_optional_tensor(
    const std::optional<Tensor>& opt) {
  return opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*opt)
    : c10::MaybeOwned<Tensor>(); // 返回一个空的 MaybeOwned<Tensor> 对象
    // 创建一个 `MaybeOwned` 类型的对象，持有一个 `Tensor` 对象，并使用 `owned` 方法构造它。
    :c10::MaybeOwned<Tensor>::owned(std::in_place);
}

这是一个单独的右括号，结束了 `expect_contiguous` 方法的实现。


inline c10::MaybeOwned<Tensor> Tensor::expect_contiguous(MemoryFormat memory_format) const & {

定义了 `Tensor` 类的成员函数 `expect_contiguous`，返回类型为 `c10::MaybeOwned<Tensor>`。这是一个内联函数，并且是 `Tensor` 类的常引用成员函数。


  if (is_contiguous(memory_format)) {

如果当前张量对象在给定的内存格式下是连续的：


    return c10::MaybeOwned<Tensor>::borrowed(*this);

则返回一个 `c10::MaybeOwned<Tensor>` 对象，借用当前对象 `*this`。


  } else {

否则，如果当前张量对象不是在给定内存格式下是连续的：


    return c10::MaybeOwned<Tensor>::owned(__dispatch_contiguous(memory_format));

返回一个 `c10::MaybeOwned<Tensor>` 对象，拥有通过 `__dispatch_contiguous(memory_format)` 调用返回的张量数据。


} // namespace at

结束了命名空间 `at`。



这是代码块的结束标记，用于结束整个代码段。
```