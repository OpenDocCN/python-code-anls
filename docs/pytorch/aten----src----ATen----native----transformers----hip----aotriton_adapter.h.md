# `.\pytorch\aten\src\ATen\native\transformers\hip\aotriton_adapter.h`

```
#pragma once
// 如果定义了 USE_ROCM 宏，则包含下列头文件
#ifdef USE_ROCM

#include <aotriton/dtypes.h>  // 包含 aotriton 库的数据类型定义
#include <aotriton/util.h>    // 包含 aotriton 库的实用函数

////////////////////////////////////////////////////////////////////////////////
// Common macros copied from cuda/mem_eff_attention/gemm_kernel_utils.h
////////////////////////////////////////////////////////////////////////////////

// 检查非稀疏且连续的 CUDA 张量 TENSOR
#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                             \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");       \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor");   \
  TORCH_CHECK(TENSOR.is_contiguous());

// 检查非稀疏且最后维度连续的 CUDA 张量 TENSOR
#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                         \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");       \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor");   \
  TORCH_CHECK(                                                           \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

// 检查指针 PTR 是否按 ALIGNMENT 对齐
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  TORCH_CHECK(                           \
      uint64_t(PTR) % ALIGNMENT == 0, #PTR " is not correctly aligned")

// 将 B 赋值给 A，并检查是否溢出
#define ASSIGN_CHECK_OVERFLOW(A, B)                                        \
  {                                                                        \
    A = B;                                                                 \
    TORCH_CHECK(                                                        \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows");    \
  }

namespace sdp {

namespace aotriton_adapter {

// 将 PyTorch 的数据类型 t_dtype 转换为 aotriton 库中的数据类型
inline aotriton::DType cast_dtype(caffe2::TypeMeta t_dtype)
{
#define CAST_TYPE(aname, dtname) if (t_dtype == at::aname) return aotriton::DType::dtname
  CAST_TYPE(kByte, kUInt8);       // 转换为 UInt8 类型
  CAST_TYPE(kUInt16, kUInt16);    // 转换为 UInt16 类型
  CAST_TYPE(kUInt32, kUInt32);    // 转换为 UInt32 类型
  CAST_TYPE(kUInt64, kUInt64);    // 转换为 UInt64 类型
  CAST_TYPE(kChar, kInt8);        // 转换为 Int8 类型
  CAST_TYPE(kShort, kInt16);      // 转换为 Int16 类型
  CAST_TYPE(kInt, kInt32);        // 转换为 Int32 类型
  CAST_TYPE(kLong, kInt64);       // 转换为 Int64 类型
  CAST_TYPE(kHalf, kFloat16);     // 转换为 Float16 类型
  CAST_TYPE(kFloat, kFloat32);    // 转换为 Float32 类型
  CAST_TYPE(kBFloat16, kBFloat16);// 转换为 BFloat16 类型
  return aotriton::DType::kUnknown; // 默认返回未知类型
#undef CAST_TYPE
}

// 将 IntArrayRef 转换为目标类型为 TargetType、秩为 Rank 的 std::array
template<typename TargetType, int Rank>
struct IntArrayRefCaster {
  // std::array<TargetType, Rank> cast(IntArrayRef);
};

// 将 IntArrayRef 转换为目标类型为 TargetType、秩为 1 的 std::array
template<typename TargetType>
struct IntArrayRefCaster<TargetType, 1> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 1>{{ static_cast<TargetType>(ref.at(0)) }};
  }
};

// 将 IntArrayRef 转换为目标类型为 TargetType、秩为 2 的 std::array
template<typename TargetType>
struct IntArrayRefCaster<TargetType, 2> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 2>{{
      static_cast<TargetType>(ref.at(0)),
      static_cast<TargetType>(ref.at(1))
    }};
  }
};

// 将 IntArrayRef 转换为目标类型为 TargetType、秩为 3 的 std::array
template<typename TargetType>
struct IntArrayRefCaster<TargetType, 3> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 3>{{
      static_cast<TargetType>(ref.at(0)),
      static_cast<TargetType>(ref.at(1)),
      static_cast<TargetType>(ref.at(2))
    }};
  }
};

template<typename TargetType>
// 定义模板结构体 IntArrayRefCaster，用于将 at::IntArrayRef 转换为指定长度的 std::array
struct IntArrayRefCaster<TargetType, 4> {
  // 静态成员函数 cast，接受 at::IntArrayRef 参数并返回一个包含四个元素的 std::array
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 4>{{
      // 将 at::IntArrayRef 中的每个元素转换为目标类型 TargetType 并放入 std::array
      static_cast<TargetType>(ref.at(0)),
      static_cast<TargetType>(ref.at(1)),
      static_cast<TargetType>(ref.at(2)),
      static_cast<TargetType>(ref.at(3))
    }};
  }
};

// 定义模板函数 mk_aotensor，生成一个 aotriton::TensorView 对象，包含给定张量的相关信息
template<int Rank = 4>
aotriton::TensorView<Rank> mk_aotensor(const at::Tensor& q, c10::string_view tensor_name)
{
  // 获取张量 q 的步幅信息
  const auto strides = q.strides();
  // 确定实际秩（rank）
  int real_rank = strides.size();
  // 如果实际秩与指定的秩 Rank 不符，抛出异常
  if (real_rank != Rank) {  // Lazy convertion of tensor_name
    TORCH_CHECK(false,
                // 拼接错误信息，指出张量名 tensor_name 的秩应为 Rank，实际为 real_rank
                std::string(tensor_name) + "'s rank should be " + std::to_string(Rank)
                + " but is " + std::to_string(real_rank));
  }
  // 创建并返回一个 aotriton::TensorView 对象，包含张量的数据指针、大小、步幅及数据类型
  return aotriton::TensorView<Rank>(reinterpret_cast<intptr_t>(q.data_ptr()),
                                    // 使用 IntArrayRefCaster 将张量的大小转换为 uint64_t 的数组
                                    IntArrayRefCaster<uint64_t, Rank>::cast(q.sizes()),
                                    // 使用 IntArrayRefCaster 将张量的步幅转换为 uint64_t 的数组
                                    IntArrayRefCaster<uint64_t, Rank>::cast(strides),
                                    // 调用 cast_dtype 函数转换张量的数据类型，并返回结果
                                    cast_dtype(q.dtype()));
}

// 结束 aotriton_adapter 命名空间
} // namespace aotriton_adapter

// 结束 sdp 命名空间
} // namespace sdp

// 定义 at::native 命名空间的内联函数 ceil_div，实现向上取整的整数除法
namespace at::native {

inline int64_t ceil_div(int64_t numerator, int64_t denominator) {
  // 返回结果为将 numerator 除以 denominator 的结果向上取整
  return (numerator + (denominator - 1)) / denominator;
}

}

// 结束 USE_ROCM 宏的条件编译
#endif // USE_ROCM


这样每行代码都得到了详细的注释解释其作用，符合规范要求。
```