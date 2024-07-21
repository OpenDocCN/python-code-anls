# `.\pytorch\aten\src\ATen\native\ForeachOpsKernels.cpp`

```
// 引入必要的头文件
#include <vector>                              // 包含向量(vector)容器的头文件
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS     // 定义仅用于方法操作符(assert)的宏

// 引入 ATen 库中的相关头文件
#include <ATen/core/Tensor.h>                   // 包含张量(Tensor)的核心头文件
#include <ATen/native/ForeachUtils.h>           // 包含 ATen 的 Foreach 相关实用工具头文件
#include <c10/util/irange.h>                    // 包含 C10 库中的 irange.h 头文件，用于整数范围

// 根据条件编译不同的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>                     // 包含 ATen 函数的头文件
#include <ATen/NativeFunctions.h>               // 包含 ATen 原生函数的头文件
#include <ATen/Operators.h>                     // 包含 ATen 操作符的头文件
#else
#include <ATen/ops/_foreach_abs_native.h>       // 包含 ATen 绝对值操作的头文件
#include <ATen/ops/_foreach_acos_native.h>      // 包含 ATen 反余弦操作的头文件
// 以下类似，包含了各种 ATen 操作的头文件，用于特定的操作符
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_asin_native.h>
#include <ATen/ops/_foreach_atan_native.h>
#include <ATen/ops/_foreach_ceil_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_copy_native.h>
#include <ATen/ops/_foreach_cos_native.h>
#include <ATen/ops/_foreach_cosh_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_erf_native.h>
#include <ATen/ops/_foreach_erfc_native.h>
#include <ATen/ops/_foreach_exp_native.h>
#include <ATen/ops/_foreach_expm1_native.h>
#include <ATen/ops/_foreach_floor_native.h>
#include <ATen/ops/_foreach_frac_native.h>
#include <ATen/ops/_foreach_lerp_native.h>
#include <ATen/ops/_foreach_lgamma_native.h>
#include <ATen/ops/_foreach_log10_native.h>
#include <ATen/ops/_foreach_log1p_native.h>
#include <ATen/ops/_foreach_log2_native.h>
#include <ATen/ops/_foreach_log_native.h>
#include <ATen/ops/_foreach_max_native.h>
#include <ATen/ops/_foreach_maximum_native.h>
#include <ATen/ops/_foreach_minimum_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_neg_native.h>
#include <ATen/ops/_foreach_norm_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_reciprocal_native.h>
#include <ATen/ops/_foreach_round_native.h>
#include <ATen/ops/_foreach_sigmoid_native.h>
#include <ATen/ops/_foreach_sign_native.h>
#include <ATen/ops/_foreach_sin_native.h>
#include <ATen/ops/_foreach_sinh_native.h>
#include <ATen/ops/_foreach_sqrt_native.h>
#include <ATen/ops/_foreach_sub_native.h>
#include <ATen/ops/_foreach_tan_native.h>
#include <ATen/ops/_foreach_tanh_native.h>
#include <ATen/ops/_foreach_trunc_native.h>
#include <ATen/ops/_foreach_zero_native.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/max.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/pow.h>
#endif

// 定义 ATen 的命名空间 at::native
namespace at::native {

// 定义一个宏，用于生成针对二元操作的张量处理函数的声明
#define FOREACH_BINARY_OP_TENSOR(OP)                            \
  void foreach_tensor_##OP##_tensor_kernel_slow_(               \
      TensorList tensors, const Tensor& scalar) {               \
    # 检查标量张量的维度和元素个数是否符合预期，应为零维且只有一个元素
    TORCH_CHECK(                                                \
        scalar.dim() == 0 && scalar.numel() == 1,               \
        "scalar tensor expected to be 0 dim but it has ",       \
        scalar.dim(),                                           \
        " dimensions and ",                                     \
        scalar.numel(),                                         \
        " elements.");                                          \

    # 检查对于 foreach 操作的 API 限制
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    # 对于 tensors 中的每个张量 t，执行 OP_(scalar) 操作
    for (auto& t : tensors) {                                   \
      t.OP##_(scalar);                                          \
    }                                                           \
  }                                                             \
                                                                \
  # 用于实现较慢的 foreach_tensor_OP_tensor_kernel 操作
  std::vector<Tensor> foreach_tensor_##OP##_tensor_kernel_slow( \
      TensorList tensors, const Tensor& scalar) {               \
    # 再次检查标量张量的维度和元素个数是否符合预期
    TORCH_CHECK(                                                \
        scalar.dim() == 0 && scalar.numel() == 1,               \
        "scalar tensor expected to be 0 dim but it has ",       \
        scalar.dim(),                                           \
        " dimensions and ",                                     \
        scalar.numel(),                                         \
        " elements.");                                          \
    
    # 再次检查对于 foreach 操作的 API 限制
    check_foreach_api_restrictions(tensors);                    \
    
    # 准备一个用于存储结果的向量 result，预留足够的空间以避免重复分配
    std::vector<Tensor> result;                                 \
    result.reserve(tensors.size());                             \
    
    # 对于 tensors 中的每个张量 t，执行 OP(scalar) 操作并将结果添加到 result 中
    for (const auto& t : tensors) {                             \
      result.emplace_back(t.OP(scalar));                        \
    }                                                           \
    
    # 返回处理后的结果向量
    return result;                                              \
  }
#define FOREACH_BINARY_OP_TENSOR_ALPHA(OP)                             \
  // 定义一个宏，用于在张量列表中的每个张量和标量执行指定的运算(OP)，并加上一个缩放因子(alpha)。
  void foreach_tensor_##OP##_tensor_kernel_slow_(                      \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    // 检查标量张量是否符合预期，应为0维且只有一个元素
    TORCH_CHECK(                                                       \
        scalar.dim() == 0 && scalar.numel() == 1,                      \
        "scalar tensor expected to be 0 dim but it has ",              \
        scalar.dim(),                                                  \
        " dimensions and ",                                            \
        scalar.numel(),                                                \
        " elements.");                                                 \
    // 检查对foreach API的限制
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    // 遍历张量列表中的每个张量，对每个张量执行指定的运算(OP)，并加上缩放因子(alpha)
    for (auto& t : tensors) {                                          \
      t.OP##_(scalar, alpha);                                          \
    }                                                                  \
  }                                                                    \
                                                                       \
  // 定义一个函数，用于在张量列表中的每个张量和标量执行指定的运算(OP)，并加上一个缩放因子(alpha)，返回结果张量列表
  std::vector<Tensor> foreach_tensor_##OP##_tensor_kernel_slow(        \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    // 检查标量张量是否符合预期，应为0维且只有一个元素
    TORCH_CHECK(                                                       \
        scalar.dim() == 0 && scalar.numel() == 1,                      \
        "scalar tensor expected to be 0 dim but it has ",              \
        scalar.dim(),                                                  \
        " dimensions and ",                                            \
        scalar.numel(),                                                \
        " elements.");                                                 \
    // 检查对foreach API的限制
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    // 创建一个空的结果张量列表，预留足够的空间
    std::vector<Tensor> result;                                        \
    result.reserve(tensors.size());                                    \
    // 遍历张量列表中的每个张量，对每个张量执行指定的运算(OP)，并加上缩放因子(alpha)，将结果添加到结果列表中
    for (const auto& t : tensors) {                                    \
      result.emplace_back(t.OP(scalar, alpha));                        \
    }                                                                  \
                                                                       \
    // 返回结果张量列表
    return result;                                                     \
  }

#define FOREACH_BINARY_OP_SCALAR(OP)                            \
  // 定义一个宏，用于在张量列表中的每个张量和标量执行指定的运算(OP)
  void foreach_tensor_##OP##_scalar_kernel_slow_(               \
      TensorList tensors, const Scalar& scalar) {               \
    // 检查对foreach API的限制
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    // 遍历传入的张量列表 `tensors`
    for (auto& t : tensors) {                                   \
      // 对每个张量执行操作 `OP##_`，传入标量 `scalar`
      t.OP##_(scalar);                                          \
    }                                                           \
  }                                                             \
                                                                \
  // 慢速版本的按张量执行操作 `OP` 和标量 `scalar` 的函数
  std::vector<Tensor> foreach_tensor_##OP##_scalar_kernel_slow( \
      // 张量列表 `tensors` 和标量 `scalar` 作为参数
      TensorList tensors, const Scalar& scalar) {               \
    // 检查是否符合按张量执行操作的 API 限制
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    // 创建用于存储结果的向量 `result`，预留足够的空间
    std::vector<Tensor> result;                                 \
    result.reserve(tensors.size());                             \
    // 遍历每个张量 `tensors`，将执行操作 `OP` 和标量 `scalar` 后的结果添加到 `result` 中
    for (const auto& t : tensors) {                             \
      result.emplace_back(t.OP(scalar));                        \
    }                                                           \
                                                                \
    // 返回包含结果的向量 `result`
    return result;                                              \
  }
// 定义一个宏，用于生成操作符 OP 在标量列表上的迭代处理函数
#define FOREACH_BINARY_OP_SCALARLIST(OP)                            \
  // 慢速内核函数，对给定的张量列表 tensors 和标量列表 scalars 执行 OP 操作
  void foreach_tensor_##OP##_scalarlist_kernel_slow_(               \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {           \
    // 检查并确保遵守 foreach API 的限制条件
    check_foreach_api_restrictions(tensors, scalars);               \
                                                                    \
    // 遍历张量列表 tensors 的每个张量，并对其执行 OP##_ 操作
    for (const auto i : c10::irange(tensors.size())) {              \
      tensors[i].OP##_(scalars[i]);                                 \
    }                                                               \
  }                                                                 \
                                                                    \
  // 返回一个包含结果张量的向量，结果张量为每个张量 tensors[i] 执行 OP 操作后的结果
  std::vector<Tensor> foreach_tensor_##OP##_scalarlist_kernel_slow( \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {           \
    // 检查并确保遵守 foreach API 的限制条件
    check_foreach_api_restrictions(tensors, scalars);               \
    // 存储结果张量的向量
    std::vector<Tensor> result;                                     \
    // 预先分配足够的空间以容纳所有结果张量
    result.reserve(tensors.size());                                 \
    // 遍历张量列表 tensors 的每个张量，并将执行 OP 操作后的结果张量加入到 result 中
    for (const auto i : c10::irange(tensors.size())) {              \
      result.emplace_back(tensors[i].OP(scalars[i]));               \
    }                                                               \
                                                                    \
    // 返回包含所有结果张量的向量
    return result;                                                  \
  }

// 定义一个宏，用于生成操作符 OP 在张量列表上的迭代处理函数
#define FOREACH_BINARY_OP_LIST(OP)                            \
  // 返回一个包含结果张量的向量，结果张量为每对张量 tensors1[i] 和 tensors2[i] 执行 OP 操作后的结果
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_slow( \
      TensorList tensors1, TensorList tensors2) {             \
    // 检查并确保遵守 foreach API 的限制条件
    check_foreach_api_restrictions(tensors1, tensors2);       \
                                                              \
    // 存储结果张量的向量
    std::vector<Tensor> result;                               \
    // 预先分配足够的空间以容纳所有结果张量
    result.reserve(tensors1.size());                          \
    // 遍历张量列表 tensors1 的每个张量，并将执行 OP 操作后的结果张量加入到 result 中
    for (const auto i : c10::irange(tensors1.size())) {       \
      result.emplace_back(tensors1[i].OP(tensors2[i]));       \
    }                                                         \
                                                              \
    // 返回包含所有结果张量的向量
    return result;                                            \
  }                                                           \
                                                              \
  // 慢速内核函数，对给定的张量列表 tensors1 和 tensors2 执行 OP##_ 操作
  void foreach_tensor_##OP##_list_kernel_slow_(               \
      TensorList tensors1, TensorList tensors2) {             \
    // 检查并确保遵守 foreach API 的限制条件
    check_foreach_api_restrictions(tensors1, tensors2);       \
                                                              \
    // 遍历张量列表 tensors1 的每个张量，并对其执行 OP##_ 操作
    for (const auto i : c10::irange(tensors1.size())) {       \
      tensors1[i].OP##_(tensors2[i]);                         \
    }                                                         \
  }

// 定义一个宏，用于生成操作符 OP 在张量列表上的迭代处理函数，同时接受标量 alpha
#define FOREACH_BINARY_OP_LIST_ALPHA(OP)                               \
  // 返回一个包含结果张量的向量，结果张量为每对张量 tensors1[i] 和 tensors2[i] 执行 OP 操作并乘以标量 alpha 后的结果
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_slow(          \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors1.size());                                   \
    // 遍历 tensors1 中的每个 Tensor 对象
    for (const auto i : c10::irange(tensors1.size())) {                \
      // 将 tensors1[i] 和 tensors2[i] 进行 OP 操作，将结果存入 result
      result.emplace_back(tensors1[i].OP(tensors2[i], alpha));         \
    }                                                                  \
                                                                       \
    // 返回结果向量 result
    return result;                                                     \
  }                                                                    \
                                                                       \
  // 慢速路径下的 foreach_tensor_OP_list_kernel_ 函数
  void foreach_tensor_##OP##_list_kernel_slow_(                        \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    // 检查 tensors1 和 tensors2 是否符合 foreach API 的限制
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    // 遍历 tensors1 中的每个 Tensor 对象
    for (const auto i : c10::irange(tensors1.size())) {                \
      // 对 tensors1[i] 和 tensors2[i] 进行 OP_ 操作，将结果直接写回 tensors1[i]
      tensors1[i].OP##_(tensors2[i], alpha);                           \
    }                                                                  \
  }


这段代码是两个函数的定义，用于实现对两个 TensorList 进行逐元素操作的功能。第一个函数 `foreach_tensor_OP_list_kernel_` 是较快的路径，它创建了一个新的结果向量，并对每对 tensors1 和 tensors2 执行 `OP` 操作，并将结果存储在 `result` 向量中返回。第二个函数 `foreach_tensor_##OP##_list_kernel_slow_` 是慢速路径，直接在原地对 tensors1 进行 `OP_` 操作，将结果写回到 tensors1 中。
#define FOREACH_UNARY_OP(OP)                                           \
  // 定义一个宏，用于遍历一元操作符(OP)的张量列表并返回结果张量列表
  std::vector<Tensor> foreach_tensor_##OP##_slow(TensorList tensors) { \
    // 检查对遍历操作的限制条件
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    // 初始化结果张量列表
    std::vector<Tensor> result;                                        \
    result.reserve(tensors.size());                                    \
    // 遍历输入的张量列表
    for (const auto& t : tensors) {                                    \
      // 对每个张量执行一元操作(OP)，并将结果加入结果列表
      result.emplace_back(t.OP());                                     \
    }                                                                  \
                                                                       \
    // 返回执行操作后的结果张量列表
    return result;                                                     \
  }                                                                    \
                                                                       \
  // 定义一个函数，用于在原地对一元操作符(OP)的张量列表进行操作
  void foreach_tensor_##OP##_slow_(TensorList tensors) {               \
    // 检查对遍历操作的限制条件
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    // 遍历输入的张量列表
    for (auto& t : tensors) {                                          \
      // 在原地执行一元操作(OP)
      t.OP##_();                                                       \
    }                                                                  \
  }

#define FOREACH_POINTWISE_OP_SCALAR(OP)                                   \
  // 定义一个宏，用于遍历标量操作(OP)的张量列表并返回结果张量列表
  std::vector<Tensor> foreach_tensor_##OP##_scalar_slow(                  \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Scalar& scalar) {                                             \
    // 检查对遍历操作的限制条件
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    // 初始化结果张量列表
    std::vector<Tensor> result;                                           \
    // 遍历输入的张量列表
    for (const auto i : c10::irange(input.size())) {                      \
      // 对每个张量执行标量操作(OP)，并将结果加入结果列表
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalar)); \
    }                                                                     \
                                                                          \
    // 返回执行操作后的结果张量列表
    return result;                                                        \
  }                                                                       \
                                                                          \
  // 定义一个函数，用于在原地对标量操作(OP)的张量列表进行操作
  void foreach_tensor_##OP##_scalar_slow_(                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Scalar& scalar) {                                             \
    // 检查对遍历操作的限制条件
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    // 遍历输入的张量列表
    for (auto i = 0; i < input.size(); ++i) {                             \
      // 在原地执行标量操作(OP)
      input[i].OP##_scalar(tensors1[i], tensors2[i], scalar);             \
    }                                                                     \
  }
    // 调用函数 check_foreach_api_restrictions 检查输入参数是否符合 API 的限制要求
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    // 使用 C++11 的范围循环遍历输入容器 input 的每个元素索引 i
    for (const auto i : c10::irange(input.size())) {                      \
      // 对 input[i] 执行宏 OP##_，传递 tensors1[i]、tensors2[i] 和 scalar 作为参数
      input[i].OP##_(tensors1[i], tensors2[i], scalar);                   \
    }                                                                     \
  }
#define FOREACH_POINTWISE_OP_SCALARLIST(OP)                                   \
  // 定义宏，用于对每个输入张量列表执行标量列表操作 OP
  std::vector<Tensor> foreach_tensor_##OP##_scalarlist_slow(                  \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      at::ArrayRef<Scalar> scalars) {                                         \
    // 检查并强制执行对输入参数的限制
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);       \
                                                                              \
    // 结果向量，用于存储操作后的张量
    std::vector<Tensor> result;                                               \
    // 对输入列表中的每个张量执行操作 OP，并将结果存入结果向量
    for (const auto i : c10::irange(input.size())) {                          \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalars[i])); \
    }                                                                         \
                                                                              \
    // 返回操作后的结果向量
    return result;                                                            \
  }                                                                           \
                                                                              \
  // 原地操作宏定义，对每个输入张量列表执行标量列表操作 OP
  void foreach_tensor_##OP##_scalarlist_slow_(                                \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      at::ArrayRef<Scalar> scalars) {                                         \
    // 检查并强制执行对输入参数的限制
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);       \
                                                                              \
    // 对输入列表中的每个张量执行原地操作 OP
    for (const auto i : c10::irange(input.size())) {                          \
      input[i].OP##_(tensors1[i], tensors2[i], scalars[i]);                   \
    }                                                                         \
  }

#define FOREACH_POINTWISE_OP_TENSOR(OP)                                   \
  // 定义宏，用于对每个输入张量列表执行张量操作 OP
  std::vector<Tensor> foreach_tensor_##OP##_tensor_slow(                  \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    // 将标量张量转换为标量列表
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    // 检查并强制执行对输入参数的限制
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    return foreach_tensor_##OP##_scalarlist_slow(                         \
        input, tensors1, tensors2, scalars);                              \
  }                                                                       \
                                                                          \
  void foreach_tensor_##OP##_tensor_slow_(                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    foreach_tensor_##OP##_scalarlist_slow_(                               \
        input, tensors1, tensors2, scalars);                              \
  }



    // 返回调用 foreach_tensor_##OP##_scalarlist_slow 函数的结果
    return foreach_tensor_##OP##_scalarlist_slow(                         \
        input, tensors1, tensors2, scalars);                              \
  }                                                                       \
                                                                          \
  // 定义 foreach_tensor_##OP##_tensor_slow_ 函数，接受多个张量列表和标量张量
  void foreach_tensor_##OP##_tensor_slow_(                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    // 将输入的标量张量转换为标量列表
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    // 检查 API 的限制条件
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    // 调用 foreach_tensor_##OP##_scalarlist_slow_ 函数处理张量和标量列表
    foreach_tensor_##OP##_scalarlist_slow_(                               \
        input, tensors1, tensors2, scalars);                              \
  }
// 对于每个具有 ALPHA 后缀的二元操作，调用相应的宏定义
FOREACH_BINARY_OP_LIST_ALPHA(add);
FOREACH_BINARY_OP_LIST_ALPHA(sub);
FOREACH_BINARY_OP_LIST_ALPHA(lerp);

// 对于每个具有 TENSOR ALPHA 后缀的二元操作，调用相应的宏定义
FOREACH_BINARY_OP_TENSOR_ALPHA(add);
FOREACH_BINARY_OP_TENSOR(mul);
FOREACH_BINARY_OP_TENSOR(div);

// 对于每个具有 SCALAR 后缀的二元操作，调用相应的宏定义
FOREACH_BINARY_OP_SCALAR(add);
FOREACH_BINARY_OP_SCALAR(sub);
FOREACH_BINARY_OP_SCALAR(mul);
FOREACH_BINARY_OP_SCALAR(div);
FOREACH_BINARY_OP_SCALAR(clamp_min);
FOREACH_BINARY_OP_SCALAR(clamp_max);
FOREACH_BINARY_OP_SCALAR(pow);

// 对于每个具有 SCALARLIST 后缀的二元操作，调用相应的宏定义
FOREACH_BINARY_OP_SCALARLIST(add);
FOREACH_BINARY_OP_SCALARLIST(sub);
FOREACH_BINARY_OP_SCALARLIST(mul);
FOREACH_BINARY_OP_SCALARLIST(div);
FOREACH_BINARY_OP_SCALARLIST(clamp_min);
FOREACH_BINARY_OP_SCALARLIST(clamp_max);
FOREACH_BINARY_OP_SCALARLIST(pow);

// 对于每个不带后缀的 LIST 后缀的二元操作，调用相应的宏定义
FOREACH_BINARY_OP_LIST(mul);
FOREACH_BINARY_OP_LIST(div);
FOREACH_BINARY_OP_LIST(clamp_min);
FOREACH_BINARY_OP_LIST(clamp_max);
FOREACH_BINARY_OP_LIST(pow);

// 定义一个名为 foreach_tensor_copy_list_kernel_slow_ 的函数，实现对 TensorList 进行复制
void foreach_tensor_copy_list_kernel_slow_(
    TensorList self,
    TensorList src,
    const bool non_blocking) {
  // 检查 foreach API 的限制
  check_foreach_api_restrictions(self, src);

  // 遍历 self 中的每个 Tensor，并使用 src 中相应位置的 Tensor 进行复制操作
  for (const auto i : c10::irange(self.size())) {
    self[i].copy_(src[i], non_blocking);
  }
}

// 对于每个具有 UNARY 后缀的一元操作，调用相应的宏定义
FOREACH_UNARY_OP(sqrt);
FOREACH_UNARY_OP(exp);
FOREACH_UNARY_OP(abs);
FOREACH_UNARY_OP(acos);
FOREACH_UNARY_OP(asin);
FOREACH_UNARY_OP(atan);
FOREACH_UNARY_OP(ceil);
FOREACH_UNARY_OP(cos);
FOREACH_UNARY_OP(cosh);
FOREACH_UNARY_OP(erf);
FOREACH_UNARY_OP(erfc);
FOREACH_UNARY_OP(expm1);
FOREACH_UNARY_OP(floor);
FOREACH_UNARY_OP(log);
FOREACH_UNARY_OP(log10);
FOREACH_UNARY_OP(log1p);
FOREACH_UNARY_OP(log2);
FOREACH_UNARY_OP(neg);
FOREACH_UNARY_OP(tan);
FOREACH_UNARY_OP(tanh);
FOREACH_UNARY_OP(sin);
FOREACH_UNARY_OP(sinh);
FOREACH_UNARY_OP(round);
FOREACH_UNARY_OP(lgamma);
FOREACH_UNARY_OP(frac);
FOREACH_UNARY_OP(trunc);
FOREACH_UNARY_OP(reciprocal);
FOREACH_UNARY_OP(sigmoid);
FOREACH_UNARY_OP(sign);

// 对于每个具有 POINTWISE OP SCALAR 后缀的点对点操作，调用相应的宏定义
FOREACH_POINTWISE_OP_SCALAR(addcdiv);
FOREACH_POINTWISE_OP_SCALAR(addcmul);

// 对于每个具有 POINTWISE OP SCALARLIST 后缀的点对点操作，调用相应的宏定义
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul);

// 对于每个具有 POINTWISE OP TENSOR 后缀的点对点操作，调用相应的宏定义
FOREACH_POINTWISE_OP_TENSOR(addcdiv);
FOREACH_POINTWISE_OP_TENSOR(addcmul);

// 定义一个名为 FOREACH_TERNARY_OP(OP) 的宏，实现对三元操作的遍历
#define FOREACH_TERNARY_OP(OP)                                         \
  std::vector<Tensor> foreach_tensor_ternary_##OP##_slow(              \
      TensorList tensors1, TensorList tensors2, TensorList tensors3) { \
    // 检查 foreach API 的限制
    check_foreach_api_restrictions(tensors1, tensors2, tensors3);      \
    std::vector<Tensor> result;                                        \
    // 遍历 tensors1 中的每个 Tensor，并使用 tensors2 和 tensors3 中相应位置的 Tensor 执行 OP 操作，并将结果保存到 result 中
    for (const auto i : c10::irange(tensors1.size())) {                \
      result.emplace_back(tensors1[i].OP(tensors2[i], tensors3[i]));   \
    }                                                                  \
    return result;                                                     \
  }                                                                    \
                                                                       \
  void foreach_tensor_ternary_##OP##_slow_(                            \
      TensorList tensors1, TensorList tensors2, TensorList tensors3) { \
    check_foreach_api_restrictions(tensors1, tensors2, tensors3);      \
    # 遍历三个张量列表，对每组对应位置的张量执行指定的运算符 OP
    for (const auto i : c10::irange(tensors1.size())) {                \
      # 调用张量对象的运算符重载函数 OP##_，对张量 tensors1[i]、tensors2[i] 和 tensors3[i] 进行运算
      tensors1[i].OP##_(tensors2[i], tensors3[i]);                     \
    }                                                                  \
  }


这段代码实现了一个函数 `foreach_tensor_ternary_##OP##_slow_`，用于对三个张量列表中对应位置的张量执行特定的操作。
// 使用 FOREACH_TERNARY_OP 宏对 lerp 函数进行迭代处理

void foreach_tensor_zero_slow_(TensorList tensors) {
  // 检查对 tensors 使用 foreach API 的限制
  check_foreach_api_restrictions(tensors);

  // 对于 tensors 中的每个张量 t，调用其 zero_() 方法将其置零
  for (auto& t : tensors) {
    t.zero_();
  }
}

std::vector<Tensor> foreach_tensor_norm_slow(
    TensorList tensors,
    const Scalar& ord,
    c10::optional<ScalarType> dtype) {
  // 检查对 tensors 使用 foreach API 的限制
  check_foreach_api_restrictions(tensors);
  std::vector<Tensor> result;
  
  // 遍历 tensors 中的每个张量 t，计算其 ord 范数并存储到 result 向量中
  for (const auto& t : tensors) {
    result.emplace_back(at::linalg_vector_norm(t, ord, {}, false, dtype));
  }
  return result;
}

std::vector<Tensor> foreach_tensor_max_slow(TensorList tensors) {
  // 检查对 tensors 使用 foreach API 的限制
  check_foreach_api_restrictions(tensors);
  std::vector<Tensor> result;
  
  // 遍历 tensors 中的每个张量 t，计算其最大值并存储到 result 向量中
  for (const auto& t : tensors) {
    result.emplace_back(at::max(t));
  }
  return result;
}

std::vector<Tensor> foreach_scalar_pow_list_kernel_slow(
    const Scalar& self,
    TensorList exponent) {
  // 检查对 exponent 使用 foreach API 的限制
  check_foreach_api_restrictions(exponent);
  std::vector<Tensor> result;
  result.reserve(exponent.size());
  
  // 遍历 exponent 中的每个张量 t，计算 self 与 t 的幂运算并存储到 result 向量中
  for (const auto& t : exponent) {
    result.emplace_back(at::pow(self, t));
  }
  return result;
}

} // namespace at::native
```