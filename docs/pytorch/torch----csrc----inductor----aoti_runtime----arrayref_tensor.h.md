# `.\pytorch\torch\csrc\inductor\aoti_runtime\arrayref_tensor.h`

```py
#pragma once
// 预处理指令：确保此头文件只被包含一次

#include <torch/csrc/inductor/aoti_runtime/utils.h>
// 包含外部库头文件 <torch/csrc/inductor/aoti_runtime/utils.h>

#include <assert.h>
// 包含断言头文件

#include <cstdint>
// 包含 C++ 标准整数类型头文件

#include <cstring>
// 包含 C 字符串操作头文件

namespace torch {
namespace aot_inductor {
// 命名空间 torch::aot_inductor

// Can't use c10::ArrayRef because it's not truly header-only and
// pulls in other c10 headers. This is (sadly) copy-pasted and
// adapted.
// 由于 c10::ArrayRef 不是真正的纯头文件，并且依赖其他 c10 头文件，因此无法使用它，这段代码被复制粘贴并做了适当调整。

template <typename T>
class MiniArrayRef final {
// 模板类 MiniArrayRef，最终类，不允许派生

 public:
  using iterator = T*;
  // 迭代器类型为 T* 指针类型
  using const_iterator = const T*;
  // 常量迭代器类型为 const T* 指针类型
  using size_type = size_t;
  // 定义 size_type 类型为 size_t
  using value_type = T;
  // 定义 value_type 类型为 T

  using reverse_iterator = std::reverse_iterator<iterator>;
  // 反向迭代器类型为 std::reverse_iterator<iterator>

 private:
  /// The start of the array, in an external buffer.
  /// 数组的起始位置，在外部缓冲区中。

  T* Data;
  // 数据指针 Data

  /// The number of elements.
  /// 元素数量。

  size_type Length;
  // 长度类型 Length

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty MiniArrayRef.
  /// 构造一个空的 MiniArrayRef 对象。
  /* implicit */ constexpr MiniArrayRef() : Data(nullptr), Length(0) {}

  /// Construct an MiniArrayRef from a single element.
  /// 从单个元素构造 MiniArrayRef 对象。
  // TODO Make this explicit
  constexpr MiniArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct an MiniArrayRef from a pointer and length.
  /// 根据指针和长度构造 MiniArrayRef 对象。
  constexpr MiniArrayRef(T* data, size_t length) : Data(data), Length(length) {}

  /// Construct an MiniArrayRef from a range.
  /// 根据范围构造 MiniArrayRef 对象。
  constexpr MiniArrayRef(T* begin, T* end) : Data(begin), Length(end - begin) {}

  template <
      typename Container,
      typename = std::enable_if_t<std::is_same<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>::value>>
  /* implicit */ MiniArrayRef(Container& container)
      : Data(container.data()), Length(container.size()) {}

  /// Construct an MiniArrayRef from a std::vector.
  /// 根据 std::vector 构造 MiniArrayRef 对象。
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because MiniArrayRef can't work on a std::vector<bool>
  // bitfield.
  // 这里的 enable_if 确保不会用于 std::vector<bool>，因为 MiniArrayRef 无法处理 std::vector<bool> 位域。
  template <typename A>
  /* implicit */ MiniArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same<T, bool>::value,
        "MiniArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an MiniArrayRef from a std::array
  /// 根据 std::array 构造 MiniArrayRef 对象。
  template <size_t N>
  /* implicit */ constexpr MiniArrayRef(std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct an MiniArrayRef from a C array.
  /// 根据 C 数组构造 MiniArrayRef 对象。
  template <size_t N>
  /* implicit */ constexpr MiniArrayRef(T (&Arr)[N]) : Data(Arr), Length(N) {}

  // /// Construct an MiniArrayRef from an empty C array.
  /// 根据空 C 数组构造 MiniArrayRef 对象。
  /* implicit */ constexpr MiniArrayRef(const volatile void* Arr)
      : Data(nullptr), Length(0) {}

  /// Construct an MiniArrayRef from a std::initializer_list.
  /// 根据 std::initializer_list 构造 MiniArrayRef 对象。
  /* implicit */ constexpr MiniArrayRef(const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return Data;
  }
  // 返回迭代器指向的开始位置

  constexpr iterator end() const {
    // 返回迭代器指向的结束位置
    return Data + Length;
  }

  // ...
  // 后续函数未完全列出，继续注释。
    // 返回指向 Data 数组起始地址加上 Length 的指针，即数组的末尾后一个位置的指针
      }
    
      // 由于 MiniArrayRef 只提供了 const 迭代器，因此这些函数实际上与迭代器相同。
      // 返回指向 Data 数组起始地址的 const 迭代器
      constexpr const_iterator cbegin() const {
        return Data;
      }
      // 返回指向 Data 数组末尾后一个位置的 const 迭代器
      constexpr const_iterator cend() const {
        return Data + Length;
      }
    
      // 返回逆序迭代器，指向正向迭代器 end() 的位置
      constexpr reverse_iterator rbegin() const {
        return reverse_iterator(end());
      }
      // 返回逆序迭代器，指向正向迭代器 begin() 的位置
      constexpr reverse_iterator rend() const {
        return reverse_iterator(begin());
      }
    
      /// empty - 检查数组是否为空。
      // 返回数组长度是否为 0 的布尔值
      constexpr bool empty() const {
        return Length == 0;
      }
    
      // 返回指向 Data 数组起始地址的指针
      constexpr T* data() const {
        return Data;
      }
    
      /// size - 获取数组的大小。
      // 返回数组的长度
      constexpr size_t size() const {
        return Length;
      }
    
      /// equals - 按元素检查是否相等。
      // 检查数组长度和元素逐一比较是否与 RHS 相等
      constexpr bool equals(MiniArrayRef RHS) const {
        return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
      }
    
      /// @}
      /// @name Operator Overloads
      /// @{
    
      // 重载下标运算符，返回数组中索引为 Index 的元素的引用
      constexpr const T& operator[](size_t Index) const {
        return Data[Index];
      }
    
      /// 防止从临时对象进行意外赋值。
      ///
      /// 此处的声明比较复杂，以便 "arrayRef = {}" 仍然选择移动赋值运算符。
      // 禁止从临时对象进行赋值，以确保只能使用移动赋值运算符
      template <typename U>
      typename std::enable_if<std::is_same<U, T>::value, MiniArrayRef<T>>::type&
      operator=(U&& Temporary) = delete;
    
      /// 防止从初始化列表进行意外赋值。
      ///
      /// 此处的声明比较复杂，以便 "arrayRef = {}" 仍然选择移动赋值运算符。
      // 禁止从初始化列表进行赋值，以确保只能使用移动赋值运算符
      template <typename U>
      typename std::enable_if<std::is_same<U, T>::value, MiniArrayRef<T>>::type&
      operator=(std::initializer_list<U>) = delete;
};

// 结束了上一个代码段的类定义，这里用于闭合类定义的大括号

using MiniIntArrayRef = MiniArrayRef<int64_t>;

// 定义一个类型别名 MiniIntArrayRef，表示指向 int64_t 类型的 MiniArrayRef

static_assert(
    sizeof(MiniIntArrayRef) == sizeof(void*) + sizeof(size_t),
    "changing the size of MiniArrayRef breaks ABI compatibility!");

// 静态断言，检查 MiniIntArrayRef 的大小是否满足特定条件，用于保证 ABI 兼容性

inline bool is_contiguous_strides_for_shape(
    int64_t ndim,
    const int64_t* strides_ptr,
    const int64_t* sizes_ptr) {
  // 检查给定形状是否具有连续的步幅
  int64_t z = 1;
  for (int64_t d = ndim - 1; d >= 0; d--) {
    const auto& size_d = sizes_ptr[d];
    if (size_d != 1) {
      if (strides_ptr[d] == z) {
        z *= size_d;
      } else {
        return false;
      }
    }
  }
  return true;
}

// AOTI 生成的代码的适配器，模拟将原始数组视为 AtenTensorHandle 的工作方式
template <typename T>
class ArrayRefTensor {
 public:
  ArrayRefTensor() = default;

  explicit ArrayRefTensor(
      MiniArrayRef<T> arr,
      MiniArrayRef<const int64_t> sizes,
      MiniArrayRef<const int64_t> strides,
      int32_t device_type,
      int32_t device_idx)
      : arrayRef_(arr),
        sizes_(sizes),
        strides_(strides),
        device_type_(device_type),
        device_idx_(device_idx) {
    // 断言确保 sizes 和 strides 的大小一致，并且 strides 是与形状连续的
    assert(sizes.size() == strides.size());
    assert(is_contiguous_strides_for_shape(
        sizes.size(), strides.data(), sizes.data()));
  }

  // 创建一个昂贵的 AtenTensorHandle 的副本
  AtenTensorHandle expensiveCopyToTensor() const {
    AtenTensorHandle result;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        sizes_.size(),
        sizes_.data(),
        strides_.data(),
        aoti_torch_dtype<std::remove_const_t<T>>(),
        device_type_,
        device_idx_,
        &result));
    void* dataPtr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(result, &dataPtr));
    std::memcpy(dataPtr, data(), numel() * sizeof(T));
    return result;
  }

  // 释放资源，返回一个拥有的 AtenTensorHandle
  AtenTensorHandle release() {
    return expensiveCopyToTensor();
  }

  // 不需要释放任何内存
  void reset() {}

  // 返回 sizes 的副本
  auto sizes() const {
    return sizes_;
  }

  // 返回 strides 的副本
  auto strides() const {
    return strides_;
  }

  // 返回设备类型
  auto device_type() const {
    return device_type_;
  }

  // 返回设备索引
  auto device_idx() const {
    return device_idx_;
  }

  // 返回数组的数据指针
  T* data() const {
    return arrayRef_.data();
  }

  // 返回数组中元素的数量
  auto numel() const {
    return arrayRef_.size();
  }

  // 设置新的 MiniArrayRef<T> 数组引用
  void set_arrayref(MiniArrayRef<T> new_arrayref) {
    arrayRef_ = new_arrayref;
  }

 private:
  MiniArrayRef<T> arrayRef_;  // 数组的引用
  MiniArrayRef<const int64_t> sizes_;    // 形状大小的引用
  MiniArrayRef<const int64_t> strides_;  // 步幅的引用
  int32_t device_type_ = 0;   // 设备类型，默认为 0
  int32_t device_idx_ = 0;    // 设备索引，默认为 0
  int32_t unusedDoNotRemoveForABICompatibility_ = 0;  // 为了保持 ABI 兼容性而预留的字段
};

static_assert(
    # 检查 ArrayRefTensor<int> 的大小是否等于预期的值，用于保持 ABI 兼容性
    sizeof(ArrayRefTensor<int>) ==
        # 计算 ArrayRefTensor<int> 中包含的各个部分的总大小
        3 * sizeof(MiniIntArrayRef) + 3 * sizeof(int32_t) +
            # 如果 ArrayRefTensor<int> 的对齐要求大于 4，还需添加一个额外的 int32_t 的大小
            (alignof(ArrayRefTensor<int>) > 4 ? sizeof(int32_t) : 0),
        # 若大小不符合预期，抛出断言错误，指出这会破坏 ABI 兼容性！
        "changing the size of ArrayRefTensor breaks ABI compatibility!");
// 将给定的 AtenTensorHandle 重新解释为新的张量对象
inline AtenTensorHandle reinterpret_tensor_wrapper(
    AtenTensorHandle self,                    // 输入的张量句柄
    int64_t ndim,                             // 新张量的维度
    const int64_t* sizes_ptr,                 // 新张量的大小数组指针
    const int64_t* strides_ptr,               // 新张量的步长数组指针
    int64_t storage_offset) {                 // 新张量的存储偏移量
  AtenTensorHandle result;                    // 结果张量句柄
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
      self, ndim, sizes_ptr, strides_ptr, storage_offset, &result));  // 调用重解释函数
  return result;                              // 返回结果张量句柄
}

template <typename T>
// 将给定的 ArrayRefTensor<T> 重新解释为新的张量对象
inline ArrayRefTensor<T> reinterpret_tensor_wrapper(
    const ArrayRefTensor<T>& self,             // 输入的数组引用张量
    int64_t ndim,                             // 新张量的维度
    const int64_t* sizes_ptr,                 // 新张量的大小数组指针
    const int64_t* strides_ptr,               // 新张量的步长数组指针
    int64_t storage_offset) {
  // REVIEW: 我们应该添加一种在调试模式下构建 DSO 的方法，
  // 以便在测试时可以进行此类检查！
  assert(is_contiguous_strides_for_shape(ndim, strides_ptr, sizes_ptr));  // 断言是否为连续的步长用于给定的形状
  return ArrayRefTensor<T>(
      MiniArrayRef<T>(
          self.data() + storage_offset, self.numel() - storage_offset),  // 构造新的 ArrayRefTensor<T>
      MiniArrayRef<const int64_t>(sizes_ptr, ndim),   // 大小数组的小型引用
      MiniArrayRef<const int64_t>(strides_ptr, ndim), // 步长数组的小型引用
      self.device_type(),                     // 设备类型
      self.device_idx());                     // 设备索引
}

// 获取给定 AtenTensorHandle 的数据指针
inline void* get_data_ptr_wrapper(AtenTensorHandle tensor) {
  void* result;                               // 数据指针结果
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(tensor, &result));  // 获取数据指针
  return result;                              // 返回数据指针
}

template <typename T>
// 获取给定 ArrayRefTensor<T> 的数据指针
inline T* get_data_ptr_wrapper(ArrayRefTensor<T>& tensor) {
  return tensor.data();                       // 直接返回数组引用张量的数据指针
}

template <typename T>
// 获取给定 MiniArrayRef<T> 的数据指针
inline T* get_data_ptr_wrapper(const MiniArrayRef<T>& arr) {
  return arr.data();                          // 直接返回小型数组引用的数据指针
}

// 如果需要，解包 RAIIAtenTensorHandle，并返回其内部的 AtenTensorHandle
inline AtenTensorHandle unwrap_raii_handle_if_needed(
    const RAIIAtenTensorHandle& handle) {
  return handle.get();                        // 获取 RAIIAtenTensorHandle 内部的 AtenTensorHandle
}

template <typename T>
// 如果需要，解包 ArrayRefTensor<T>
inline const ArrayRefTensor<T>& unwrap_raii_handle_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor;                              // 直接返回输入的 ArrayRefTensor<T>
}

template <typename T>
// 如果需要，解包 ArrayRefTensor<T>
inline ArrayRefTensor<T>& unwrap_raii_handle_if_needed(
    ArrayRefTensor<T>& tensor) {
  return tensor;                              // 直接返回输入的 ArrayRefTensor<T>
}

// 如果需要，用 RAIIAtenTensorHandle 包装给定的 AtenTensorHandle
inline RAIIAtenTensorHandle wrap_with_raii_handle_if_needed(
    AtenTensorHandle handle) {
  return RAIIAtenTensorHandle(handle);        // 使用 RAII 包装给定的 AtenTensorHandle
}

template <typename T>
// 如果需要，用 RAIIAtenTensorHandle 包装给定的 ArrayRefTensor<T>
inline const ArrayRefTensor<T>& wrap_with_raii_handle_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor;                              // 直接返回输入的 ArrayRefTensor<T>
}

template <typename T>
// 如果需要，用 RAIIAtenTensorHandle 包装给定的 ArrayRefTensor<T>
inline ArrayRefTensor<T>& wrap_with_raii_handle_if_needed(
    ArrayRefTensor<T>& tensor) {
  return tensor;                              // 直接返回输入的 ArrayRefTensor<T>
}

template <typename T>
// 将给定的 ArrayRefTensor<T> 转换为对应的 RAIIAtenTensorHandle
const T& convert_arrayref_tensor_to_tensor(const T& t) {
  return t;                                   // 直接返回输入的 ArrayRefTensor<T>
}

template <typename T>
// 将给定的 ArrayRefTensor<T> 转换为对应的 RAIIAtenTensorHandle
RAIIAtenTensorHandle convert_arrayref_tensor_to_tensor(
    const ArrayRefTensor<T>& art) {
  return art.expensiveCopyToTensor();         // 使用昂贵的拷贝操作将 ArrayRefTensor<T> 转换为 RAIIAtenTensorHandle
}

} // namespace aot_inductor
} // namespace torch
```