# `.\pytorch\aten\src\ATen\core\blob.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <type_traits>
// 引入类型特性库

#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>
#include <c10/macros/Macros.h>
// 引入 Caffe2 的依赖头文件

namespace caffe2 {

class Tensor;
// 前置声明 Tensor 类，使得可以在 Blob 类中使用 Tensor 类型

/**
 * @brief Blob is a general container that hosts a typed pointer.
 *
 * A Blob hosts a pointer as well as its type, and takes charge of deleting it
 * properly when the blob is deallocated or re-allocated with a new type. A blob
 * could contain anything, although the most common case is to contain a Tensor.
 */
class TORCH_API Blob final : public c10::intrusive_ptr_target {
// Blob 类的定义，继承自 intrusive_ptr_target

 public:
  /**
   * Initializes an empty Blob.
   */
  Blob() noexcept : meta_() {}
  // 默认构造函数，初始化一个空的 Blob

  ~Blob() override {
    Reset();
  }
  // 析构函数，释放 Blob 对象及其指向的资源

  Blob(Blob&& other) noexcept : Blob() {
    swap(other);
  }
  // 移动构造函数，使用其他 Blob 对象初始化新的 Blob，并交换内容

  Blob& operator=(Blob&& other) noexcept {
    Blob(std::move(other)).swap(*this);
    return *this;
  }
  // 移动赋值运算符，实现 Blob 对象的移动赋值操作

  /**
   * Checks if the content stored in the blob is of type T.
   */
  template <class T>
  bool IsType() const noexcept {
    return meta_.Match<T>();
  }
  // 检查 Blob 存储的内容是否为类型 T

  /**
   * Returns the meta info of the blob.
   */
  const TypeMeta meta() const noexcept {
    return meta_;
  }
  // 返回 Blob 的元信息

  /**
   * Returns a printable typename of the blob.
   */
  c10::string_view TypeName() const noexcept {
    return meta_.name();
  }
  // 返回 Blob 存储内容的可打印类型名称

  /**
   * @brief Gets the const reference of the stored object. The code checks if
   * the stored object is of the desired type.
   */
  // TODO(jerryzh): add a Get(c10::DeviceType) function?
  template <class T>
  const T& Get() const {
    TORCH_INTERNAL_ASSERT(
        IsType<T>(),
        "wrong type for the Blob instance. Blob contains ",
        meta_.name(),
        " while caller expects ",
        TypeMeta::TypeName<T>());
    // TODO: after we add Get<Tensor>(c10::DeviceType)
    // and changed all the callsites, we can add
    // a static assert here to enforce T != Tensor
    return *static_cast<const T*>(pointer_);
  }
  // 获取 Blob 存储对象的常量引用，检查对象是否为目标类型 T

  const void* GetRaw() const noexcept {
    return pointer_;
  }
  // 获取 Blob 存储对象的常量指针

  void* GetRaw() noexcept {
    return pointer_;
  }
  // 获取 Blob 存储对象的可变指针

  /**
   * @brief Gets a mutable pointer to the stored object.
   *
   * If the current object is not of the right type, a new object is created
   * and the old object is freed. Note that type T should have a default
   * constructor. Otherwise, create the object yourself first, and use
   * Reset().
   */
  template <class T>
  T* GetMutable() {
    static_assert(
        std::is_default_constructible<T>::value,
        "GetMutable can't be called with non-default-constructible types. "
        "Try using specialized methods");
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      // TODO Re-enable logging
      // VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<T>();
      return Reset<T>(new T());
    }
  }
  // 获取 Blob 存储对象的可变指针，如果对象不是正确类型，则创建新对象并释放旧对象

  template <class T>
  T* GetMutableOrNull() {
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return nullptr;
    }
  }
  // 获取 Blob 存储对象的可变指针，如果对象不是正确类型则返回空指针
  /**
   * Sets the underlying object to the allocated one. The Blob then takes over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * This is used when the underlying class T does not have a default ctor, or
   * complex initializations needs to be done outside the blob.
   */
  template <class T>
  T* Reset(T* allocated) {
    // 释放当前 Blob 持有的对象（如果有），并重新设置元数据类型为 T
    free_();
    meta_ = TypeMeta::Make<T>();
    // 将传入的指针分配给 Blob，并标记为具有所有权
    pointer_ = static_cast<void*>(allocated);
    has_ownership_ = true;
    return allocated;
  }

  /**
   * Sets the underlying object to the allocated one, but does not take over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * Unlike Reset, this does not take over the ownership of the pointer and the
   * caller is responsible for making sure that the lifetime of the allocated
   * blob outlasts the lifetime of any access to this blob, until another Reset
   * call is made or the blob is destructed.
   */
  template <class T>
  std::remove_const_t<T>* ShareExternal(
      std::remove_const_t<T>* allocated) {
    // 转换为共享外部对象，不控制指针的所有权
    return static_cast<T*>(ShareExternal(
        static_cast<void*>(allocated),
        TypeMeta::Make<std::remove_const_t<T>>()));
  }

  void* ShareExternal(void* allocated, const TypeMeta meta) {
    // 释放当前 Blob 持有的对象（如果有），并重新设置元数据类型为 meta
    free_();
    meta_ = meta;
    // 将传入的指针分配给 Blob，并标记为无所有权
    pointer_ = allocated;
    has_ownership_ = false;
    return allocated;
  }

  /**
   * Resets the Blob to an empty one.
   */
  void Reset() {
    // 释放当前 Blob 持有的对象（如果有），并将指针和元数据重置为初始状态
    free_();
    pointer_ = nullptr;
    meta_ = TypeMeta();
    has_ownership_ = false;
  }

  /**
   * @brief Swaps the underlying storage of two blobs.
   */
  void swap(Blob& rhs)  noexcept {
    // 交换两个 Blob 对象的存储空间
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(has_ownership_, rhs.has_ownership_);
  }

 private:
  // 释放 Blob 当前持有的对象
  void free_() {
    if (has_ownership_ && pointer_ != nullptr) {
      (*meta_.deleteFn())(pointer_);
    }
  }

  TypeMeta meta_;
  void* pointer_{nullptr};
  bool has_ownership_{false};

  C10_DISABLE_COPY_AND_ASSIGN(Blob);
};

// 定义了一个内联函数 swap，用于交换两个 Blob 对象的内容
inline void swap(Blob& lhs, Blob& rhs) noexcept {
  lhs.swap(rhs);
}

// 定义了一个重载的输出流操作符 <<，用于输出 Blob 对象的类型名称
inline std::ostream& operator<<(std::ostream& out, const Blob& v) {
  return out << "Blob[" << v.TypeName() << "]";
}

// 结束 caffe2 命名空间的定义
} // namespace caffe2
```