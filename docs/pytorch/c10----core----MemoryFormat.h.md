# `.\pytorch\c10\core\MemoryFormat.h`

```
// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 C10 库的 ArrayRef 和 Exception 头文件
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

// 包含 C++ 标准库的头文件
#include <cstdint>
#include <ostream>
#include <vector>

// 命名空间 c10 中定义了 MemoryFormat 枚举类型和相关函数
namespace c10 {

// MemoryFormat 枚举类型，表示不同的内存布局格式
enum class MemoryFormat : int8_t {
  Contiguous,        // 连续内存布局
  Preserve,          // 保持输入的内存布局
  ChannelsLast,      // 末尾通道优先的内存布局
  ChannelsLast3d,    // 三维末尾通道优先的内存布局
  NumOptions         // 可选项总数
};

// 宏定义，获取连续内存布局的默认值，用于兼容旧代码
#define LEGACY_CONTIGUOUS_MEMORY_FORMAT c10::get_contiguous_memory_format()

// 返回连续内存布局的 MemoryFormat 枚举值
inline MemoryFormat get_contiguous_memory_format() {
  return MemoryFormat::Contiguous;
}

// 重载 << 运算符，用于将 MemoryFormat 转换为字符串输出到流中
inline std::ostream& operator<<(
    std::ostream& stream,
    at::MemoryFormat memory_format) {
  switch (memory_format) {
    case MemoryFormat::Preserve:
      return stream << "Preserve";
    case MemoryFormat::Contiguous:
      return stream << "Contiguous";
    case MemoryFormat::ChannelsLast:
      return stream << "ChannelsLast";
    case MemoryFormat::ChannelsLast3d:
      return stream << "ChannelsLast3d";
    default:
      // 如果遇到未知的 MemoryFormat 枚举值，抛出异常
      TORCH_CHECK(false, "Unknown memory format ", memory_format);
  }
}

// 获取二维末尾通道优先布局的步长向量，针对不同尺寸情况进行优化
template <typename T>
inline std::vector<T> get_channels_last_strides_2d(ArrayRef<T> sizes) {
  std::vector<T> strides(sizes.size());
  switch (sizes.size()) {
    case 4:
      strides[1] = 1;
      strides[3] = sizes[1];
      strides[2] = strides[3] * sizes[3];
      strides[0] = strides[2] * sizes[2];
      return strides;
    case 3:
      strides[0] = 1;
      strides[2] = sizes[0];
      strides[1] = strides[2] * sizes[2];
      return strides;
    default:
      // 如果尺寸不支持二维末尾通道优先布局，抛出内部断言异常
      TORCH_INTERNAL_ASSERT(
          false, "ChannelsLast2d doesn't support size ", sizes.size());
  }
}

// 重载，支持 IntArrayRef 类型的二维末尾通道优先布局步长向量获取
inline std::vector<int64_t> get_channels_last_strides_2d(IntArrayRef sizes) {
  return get_channels_last_strides_2d<int64_t>(sizes);
}

// 获取三维末尾通道优先布局的步长向量，针对不同尺寸情况进行优化
template <typename T>
std::vector<T> get_channels_last_strides_3d(ArrayRef<T> sizes) {
  std::vector<T> strides(sizes.size());
  switch (sizes.size()) {
    case 5:
      strides[1] = 1;
      strides[4] = sizes[1];
      strides[3] = strides[4] * sizes[4];
      strides[2] = strides[3] * sizes[3];
      strides[0] = strides[2] * sizes[2];
      return strides;
    # 对于 case 4，配置步长数组以适应 ChannelsLast3d 的需求
    case 4:
      # 第一个维度的步长为 1
      strides[0] = 1;
      # 第四个维度的步长为第一个维度的大小
      strides[3] = sizes[0];
      # 第三个维度的步长为第一个和第四个维度大小的乘积
      strides[2] = strides[3] * sizes[3];
      # 第二个维度的步长为第一个、第四个和第三个维度大小的乘积
      strides[1] = strides[2] * sizes[2];
      # 返回配置好的步长数组
      return strides;
    # 如果不是 case 4，则抛出错误，指出 ChannelsLast3d 不支持给定维度大小
    default:
      TORCH_INTERNAL_ASSERT(
          false, "ChannelsLast3d doesn't support size ", sizes.size());
}

// NOTE:
// 下面是 is_channels_last_strides_xd 的辅助函数。
// 1. 请不要合并这些辅助函数，每个函数只处理一个 sizes + memory_format 的情况。
//    通过这种方式，strides 索引将是一个常量数组，并且我们可以使用常量索引号访问它，
//    编译器会完全展开 strides 索引的循环以提升性能。
// 2. 辅助函数内部没有错误检查，调用者保证输入的正确性。
// 3. 所有辅助函数都有类似的注释，这里只对第一个辅助函数进行了注释。

template <typename T>
inline bool is_channels_last_strides_2d_s4(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  T min = 0;
  // 对于 C 维度的特殊情况。默认为 NCHW
  if (strides[1] == 0) {
    return false;
  }
  // 循环遍历 strides 的索引
  for (auto& d : {1, 3, 2, 0}) {
    // 如果 sizes 中某个维度为 0，则返回 false
    if (sizes[d] == 0) {
      return false;
    }
    // 如果 strides 中某个维度小于 min，则返回 false
    if (strides[d] < min) {
      return false;
    }
    // 默认布局为 NCHW 用于不明确的情况
    // 这是隐式内存格式从 strides 来的一个缺陷。
    // 对于大小为 1 的维度的相同 strides 的 N111 张量有两种情况：
    // a. N111 连续张量 ([N,1,1,1]@[1,1,1,1])
    // b. N11W 连续张量在 W 维度上切片。 ([N,1,1,1]@[W,W,W,W])
    if (d == 0 && min == strides[1]) {
      return false;
    }
    // 这是必要的：
    // 1. 区分 N1H1 的内存格式；
    //    [H, 1, 1, 1] channels_last stride
    //    [H, H, 1, 1] contiguous stride
    // 2. 1C1W 的排列：
    //    [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
    //    [1, H, 1, C]@[HC, 1, H, H] 不应被识别为 channels_last
    min = strides[d];
    // 如果 sizes[d] 大于 1，则 min 需要乘以 sizes[d]
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  // 如果所有条件都满足，则返回 true
  return true;
}

template <typename T>
inline bool is_channels_last_strides_3d_s5(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  T min = 0;
  if (strides[1] == 0) {
    return false;
  }
  // 循环遍历 strides 的索引
  for (auto& d : {1, 4, 3, 2, 0}) {
    // 如果 sizes 中某个维度为 0，则返回 false
    if (sizes[d] == 0) {
      return false;
    }
    // 如果 strides 中某个维度小于 min，则返回 false
    if (strides[d] < min) {
      return false;
    }
    // 默认布局为 NCHW 用于不明确的情况
    if (d == 0 && min == strides[1]) {
      return false;
    }
    min = strides[d];
    // 如果 sizes[d] 大于 1，则 min 需要乘以 sizes[d]
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  // 如果所有条件都满足，则返回 true
  return true;
}

// Note [Ambiguous is_channels_last_strides_xd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 隐式通过 strides 携带内存格式的缺陷非常难以正确处理。issue #24090
// 没有排列历史，我们无法从大小和 strides 的快照中推断张量的内存格式。
// 例如：
//
// 1. 我们不能通过 strides 有效地指定 N111 张量的内存格式；
//
// 2. 两条路径导致相同的大小/stride
//  N11W contiguous tensor sliced at w-dimension becomes [N,1,1,1]@[W,W,W,W]
//  NC11 channels_last tensor sliced at c-dimension becomes [N,1,1,1]@[C,C,C,C]
//    So if we see a tensor [N,1,1,1]@[X,X,X,X], there's no way for us to infer
//    the memory_format of the original tensor.
//
// Due to the limitations, our temporary WAR `is_channels_last_strides` does the
// best effort to infer whether the original memory_format of a tensor is
// at::MemoryFormat::ChannelsLast. The two objectives of this function (ordered
// by their importance):
//   1. Ensure that normal shape manipulation does not accidentally change the
//      MemoryFormat of an existing tensor.
//   2. Allows user to mark MemoryFormat::ChannelsLast to tensors;
//
// The function does so via checking strides of the tensor, including strides of
// size-1 dimensions. Although conventionally PyTorch implies no restriction on
// trivial stride (stride for size-1 dimension).
//
// Note that this approach is a compromise. We did not solve the problem
// completely. Many cases we will not be able to infer the correct memory
// format.
// The implementation of `is_channels_last_strides` is to serve the objectives:
// MemoryFormat::ChannelsLast has to be explicitly opted-in (no accidental
// conversion); Best effort to maintain the ChannelsLast flag.
//
// Due to the fact that this is not a bulletproof solution, through testing
// (aten/src/ATen/test/memory_format_test.cpp)
//   a. we ensure that the common tasks are supported;
//   a. we identify corner cases where the implementation compromises on.
//
// By the time accumulated permutation is enabled to replace implicit
// memory_format through strides, we should be updating our tests and fix the
// issues in our tests.
//
// We use Channels Last 2d as an example above.
// This is a general problem for all the is_channels_last_strides_xd
// implementation. Please check the helper functions
// (is_channels_last_strides_*d_s*) for more details.

// Template function to check if the strides of a 2D tensor indicate Channels Last memory format
template <typename T>
inline bool is_channels_last_strides_2d(
    const ArrayRef<T> sizes,    // Array reference containing sizes of tensor dimensions
    const ArrayRef<T> strides)  // Array reference containing strides of tensor dimensions
{
  switch (sizes.size()) {  // Switch based on the number of tensor dimensions
    case 4:
      return is_channels_last_strides_2d_s4(sizes, strides);  // Delegate to specialized function for 4D tensors
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return false;  // Return false for 3D tensors (not fully supported yet)
    default:
      return false;  // Default case: return false for unsupported tensor dimensions
  }
}

// Template function to check if the strides of a 3D tensor indicate Channels Last memory format
template <typename T>
inline bool is_channels_last_strides_3d(
    const ArrayRef<T> sizes,    // Array reference containing sizes of tensor dimensions
    const ArrayRef<T> strides)  // Array reference containing strides of tensor dimensions
{
  switch (sizes.size()) {  // Switch based on the number of tensor dimensions
    case 5:
      return is_channels_last_strides_3d_s5(sizes, strides);  // Delegate to specialized function for 5D tensors
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;  // Return false for 4D tensors (not fully supported yet)
    default:
      return false;  // Default case: return false for unsupported tensor dimensions
  }
}

// Wrapper function to check if the strides of a 2D tensor indicate Channels Last memory format
inline bool is_channels_last_strides_2d(
    const IntArrayRef sizes,    // Array reference containing sizes of tensor dimensions (integers)
    const IntArrayRef strides)  // Array reference containing strides of tensor dimensions (integers)
{
  return is_channels_last_strides_2d<int64_t>(sizes, strides);  // Call template function with int64_t type
}
// 内联函数，检查给定的大小和步幅数组是否满足三维通道末尾（channels last）的条件
inline bool is_channels_last_strides_3d(
    const IntArrayRef sizes,    // 接受整数数组引用 sizes，表示张量的尺寸
    const IntArrayRef strides)  // 接受整数数组引用 strides，表示张量的步幅
{
  // 调用模板函数 is_channels_last_strides_3d<int64_t>，传递 sizes 和 strides 作为参数
  return is_channels_last_strides_3d<int64_t>(sizes, strides);
}

} // namespace c10
```