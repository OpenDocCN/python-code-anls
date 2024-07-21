# `.\pytorch\aten\src\ATen\native\cuda\jit_utils.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <string>
// 包含操作字符串的标准库

#include <sstream>
// 包含字符串流处理的标准库

#include <unordered_map>
// 包含无序映射（哈希表）的标准库

#include <vector>
// 包含向量（动态数组）的标准库

#include <c10/util/irange.h>
// 包含用于生成整数范围的C10实用工具

#include <ATen/jit_macros.h>
// 包含用于JIT编译的ATen宏定义

#include <ATen/cuda/detail/LazyNVRTC.h>
// 包含懒加载NVRTC的CUDA细节

namespace at { namespace cuda { namespace jit {
// 命名空间，用于ATen CUDA JIT编译器相关功能

enum class BinaryFuncVariant {NoScalar, RhsScalar, LhsScalar};
// 枚举类型，表示二元函数的变体

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = nullptr;
};
// 结构体，表示NVRTC函数信息，包含CUDA模块和函数指针

struct KernelDescriptor {
  std::string name;
  // 内核描述符的名称

  std::string f;
  // 内核描述符的函数字符串

  c10::ScalarType f_inputs_type;
  // 内核描述符的输入数据类型

  c10::ScalarType result_type;
  // 内核描述符的结果数据类型

  c10::SmallVector<c10::ScalarType> extra_args_types;
  // 内核描述符的额外参数数据类型向量

  int nInputs, nOutputs;
  // 内核描述符的输入数量和输出数量
};

// 辅助函数，返回一个参数包中参数类型的向量<string>
template <typename... Args>
c10::SmallVector<at::ScalarType> get_extra_args_types() {
  return {c10::CppTypeToScalarType<Args>::value ...};
}

// 创建内核描述符的函数模板
template <
  typename result_type,
  typename f_inputs_type,
  typename... ExtraArgs>
KernelDescriptor make_kernel_descriptor(
    std::string name,
    std::string f,
    int nInputs,
    int nOutputs) {
  KernelDescriptor ret;
  ret.name = std::move(name);
  ret.f = std::move(f);
  ret.f_inputs_type = c10::CppTypeToScalarType<f_inputs_type>::value;
  ret.result_type = c10::CppTypeToScalarType<result_type>::value;
  ret.extra_args_types = get_extra_args_types<ExtraArgs...>();
  ret.nInputs = nInputs;
  ret.nOutputs = nOutputs;
  return ret;
}

// 内联函数，判断指针是否可以向量化到特定对齐的倍数
inline int can_vectorize_up_to(size_t default_alignment, void *pointer) {
  auto ip = reinterpret_cast<uintptr_t>(pointer);
  if (ip % (4 * default_alignment) == 0) {
    return 4;
  }
  if (ip % (2 * default_alignment) == 0) {
    return 2;
  }
  return 1;
}

// 函数重载，判断指针数组中的数据是否可以向量化到特定对齐的倍数
inline int can_vectorize_up_to(const KernelDescriptor &desc, c10::ArrayRef<char*> pointers) {
  TORCH_INTERNAL_ASSERT(desc.nOutputs == 1);
  TORCH_INTERNAL_ASSERT(static_cast<int64_t>(pointers.size()) == 1 + desc.nInputs);

  // 处理输出
  auto result_size = c10::scalarTypeToTypeMeta(desc.result_type).itemsize();
  int result = can_vectorize_up_to(result_size, pointers[0]);

  // 处理输入
  auto input_size = c10::scalarTypeToTypeMeta(desc.f_inputs_type).itemsize();
  for (auto i : c10::irange(1, pointers.size())) {
    result = std::min(result, can_vectorize_up_to(input_size, pointers[i]));
  }

  return result;
}

// 生成代码的函数声明，返回生成的代码字符串
std::string generate_code(
    int nInputs,
    int nOutputs,
    const std::string& func,
    const std::string& name,
    const std::string& f_input_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    c10::SmallVector<std::string>& extra_args_typenames,
    bool vectorized=false,
    int vec_size=0,
    bool return_by_ref=false);

// 生成代码的函数声明，接受内核描述符作为参数，返回生成的代码字符串
std::string generate_code(
    const KernelDescriptor &desc,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    bool vectorized=false,
    int vec_size=0,
    bool return_by_ref=false);
// 声明一个函数，生成用于减少操作的代码，并返回生成的代码字符串
std::string generate_reduction_code(
    int nOutputs,  // 输出数量
    const std::string& func,  // 函数名称字符串
    const std::string& name,  // 名称字符串
    const int vt0,  // 整数类型变量
    const std::string& f_inputs_type,  // 函数输入类型字符串
    const std::string& reduction_accum_type,  // 累加器类型字符串
    const std::string& result_type,  // 结果类型字符串
    bool contiguous,  // 是否连续布局
    bool vectorized,  // 是否矢量化
    int vec_size,  // 矢量大小
    int max_threads_codegen);  // 最大线程数用于代码生成

// 声明一个函数，生成用于减少操作的代码，并返回生成的代码字符串
std::string generate_reduction_code(
    const KernelDescriptor &desc,  // 核心描述符对象
    const int vt0,  // 整数类型变量
    bool contiguous,  // 是否连续布局
    bool vectorized,  // 是否矢量化
    int vec_size,  // 矢量大小
    int max_threads_codegen);  // 最大线程数用于代码生成

// 使用给定的代码字符串和内核名称，返回一个经过编译的 NVRTC 函数对象
NvrtcFunction jit_pwise_function(
    const std::string& code,  // 代码字符串
    const std::string& kernel_name);  // 内核名称字符串

// 启动经过 JIT 编译的逐点函数
void launch_jitted_pwise_function(
    NvrtcFunction function,  // 经 JIT 编译的函数对象
    void* args[],  // 参数数组
    const dim3 nBlocks,  // 三维网格块数量
    const dim3 kBlockSize,  // 三维线程块大小
    const int smem=0);  // 共享内存大小，默认为0

// 一个模板结构体，当作为 false_type 时，用于延迟触发 static_assert
template <typename T>
struct delayed_false : std::false_type {
};

// 返回无效类型的字符串表示，通用情况下用于未定义的类型
// 所有有效的类型都应使用下面的 TYPE_NAME_FN 宏进行特化
template <typename T>
inline std::string typeName() {
  static_assert(delayed_false<T>::value, "invalid type for jiterator");
  return "void";
}

// 宏定义，用于生成特定类型的 typeName 特化模板函数
#define TYPE_NAME_FN(ctype, name) \
template <> inline std::string typeName<ctype>(){ \
    return std::string(#ctype);    \
}

// 针对标量类型的宏展开，用于生成具体的 typeName 特化函数
AT_FORALL_SCALAR_TYPES(TYPE_NAME_FN)
#undef TYPE_NAME_FN

// typeName 的特化函数，返回布尔类型的字符串表示
template <> inline std::string typeName<bool>(){
    return "bool";
}

// typeName 的特化函数，返回半精度复数类型的字符串表示
template <> inline std::string typeName<c10::complex<at::Half>>(){
    return "std::complex<at::Half>";
}

// typeName 的特化函数，返回单精度复数类型的字符串表示
template <> inline std::string typeName<c10::complex<float>>(){
    return "std::complex<float>";
}

// typeName 的特化函数，返回双精度复数类型的字符串表示
template <> inline std::string typeName<c10::complex<double>>(){
    return "std::complex<double>";
}

// typeName 的特化函数，返回半精度类型的字符串表示
template <> inline std::string typeName<at::Half>(){
    return "at::Half";
}

// typeName 的特化函数，返回 BFLOAT16 类型的字符串表示
template <> inline std::string typeName<at::BFloat16>(){
    return "at::BFloat16";
}

// typeName 的特化函数，返回 Float8_e5m2 类型的字符串表示
template <> inline std::string typeName<at::Float8_e5m2>(){
    return "at::Float8_e5m2";
}

// typeName 的特化函数，返回 Float8_e4m3fn 类型的字符串表示
template <> inline std::string typeName<at::Float8_e4m3fn>(){
    return "at::Float8_e4m3fn";
}

// typeName 的特化函数，返回 Float8_e5m2fnuz 类型的字符串表示
template <> inline std::string typeName<at::Float8_e5m2fnuz>() {
    return "at::Float8_e5m2fnuz";
}

// typeName 的特化函数，返回 Float8_e4m3fnuz 类型的字符串表示
template <> inline std::string typeName<at::Float8_e4m3fnuz>() {
    return "at::Float8_e4m3fnuz";
}

// typeName 的特化函数，根据标量类型枚举值返回对应的类型名称字符串
#define TYPE_NAME_CASE(ctype, scalartype)                    \
  case ScalarType::scalartype:  return typeName<ctype>();
inline std::string typeName(ScalarType t) {
    switch (t) {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(TYPE_NAME_CASE)
      default:
          TORCH_CHECK(false, "invalid type for jiterator");
    }
}
#undef TYPE_NAME_CASE



// 结束一个匿名命名空间或代码块
}
// 取消定义宏 TYPE_NAME_CASE
#undef TYPE_NAME_CASE



TORCH_CUDA_CPP_API void initializeCudaContext();



// 声明一个名为 initializeCudaContext 的函数，使用 TORCH_CUDA_CPP_API 宏修饰，用于初始化 CUDA 上下文
TORCH_CUDA_CPP_API void initializeCudaContext();



}}}  // namespace at::cuda::jit



// 结束命名空间 at::cuda::jit
}}}  // namespace at::cuda::jit
```