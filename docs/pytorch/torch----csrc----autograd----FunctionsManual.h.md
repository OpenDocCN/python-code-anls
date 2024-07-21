# `.\pytorch\torch\csrc\autograd\FunctionsManual.h`

```
#pragma once
// NB: Must be at the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio

#ifdef _MSC_VER
// 如果未定义 _USE_MATH_DEFINES，则定义它，以便使用数学常数（如 M_PI）
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
// 包含 <cmath> 标准库头文件
#include <cmath>
#endif

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 Torch 自动生成的函数头文件
#include <torch/csrc/autograd/generated/Functions.h>

// 定义命名空间 torch::autograd::generated::details
namespace torch::autograd::generated::details {

// 声明一个指向常量字符的指针 kCudnnDoubleBackwardMsg
extern const char* kCudnnDoubleBackwardMsg;

// 一个简单的方法来即时计算被展平的槽的索引范围
struct TORCH_API IndexRangeGenerator {
  // 返回一个 IndexRange 对象，表示当前范围的起始和结束索引
  IndexRange range(size_t range_size) {
    // 增加 i 的值，表示已处理的索引范围大小
    i += range_size;
    return {i - range_size, i};
  }
  // 返回当前处理的索引大小
  size_t size() {
    return i;
  }

 private:
  // 记录已处理的索引大小
  size_t i = 0;
};

// 定义三个转换函数，将 std::optional<Tensor> 转换为 Tensor
TORCH_API Tensor toNonOptFwGrad(const std::optional<Tensor>& t);
TORCH_API Tensor toNonOptPrimal(const std::optional<Tensor>& t);
TORCH_API Tensor toNonOptTensor(const std::optional<Tensor>& t);

// 定义一个内联函数 wrap_opt_if，如果条件为真则包装 Tensor 成 std::optional<Tensor>，否则返回空
TORCH_API inline std::optional<Tensor> wrap_opt_if(
    const Tensor& t,
    const bool cond) {
  using OptTensor = std::optional<Tensor>;
  return cond ? OptTensor(t) : static_cast<OptTensor>(c10::nullopt);
}

// 应用损失函数减少操作，返回减少后的 Tensor
TORCH_API Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);

// 检查 variable_list 中是否有任何变量定义
TORCH_API bool any_variable_defined(const variable_list& variables);

// 复制输入张量中指定索引范围的内容到输出 variable_list 中
TORCH_API void copy_range(
    variable_list& out,
    IndexRange range,
    const at::Tensor& t);

// 复制输入张量数组中指定索引范围的内容到输出 variable_list 中
TORCH_API void copy_range(
    variable_list& out,
    IndexRange range,
    at::ArrayRef<at::Tensor> t);

// 计算张量自身的符号与梯度的乘积
TORCH_API at::Tensor copysign_tensor_self_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);

// 报告函数未实现的错误信息
TORCH_API at::Tensor not_implemented(const char* name, const char* reason = "");

// 报告多个函数未实现的错误信息
TORCH_API std::vector<Tensor> not_implemented_list(
    const char* name,
    const char* reason = "");

// 处理标量类型 self_st 和梯度结果张量的函数
at::Tensor handle_r_to_c(ScalarType self_st, Tensor gradient_result);

// 可能对张量 t 进行标量 s 的乘法操作
at::Tensor maybe_multiply(const at::Tensor& t, const at::Scalar& s);

// 安全计算张量 sizes 在指定维度 dim 上的大小
int64_t _safe_size(IntArrayRef sizes, IntArrayRef dim);

// 恢复输出张量的减少维度，根据 dims 参数指定的维度进行操作
Tensor restore_reduced_dims(
    const Tensor& output,
    IntArrayRef dims,
    bool keepdim);

// 按计数缩放梯度张量 grad，仅保留 mask 指定的维度
Tensor scale_grad_by_count(
    const Tensor& grad,
    const Tensor& mask,
    IntArrayRef dims);

// 根据 norm 张量计算张量 self 的梯度
at::Tensor norm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const optional<at::Scalar>& p_,
    const at::Tensor& norm);

// 根据 norm 张量计算张量 self 的梯度，支持指定维度 dim 和保持维度 keepdim
at::Tensor norm_backward(
    at::Tensor grad,
    const at::Tensor& self,
    const optional<at::Scalar>& p_,
    at::Tensor norm,
    at::IntArrayRef dim,
    bool keepdim);

// 根据 norm 张量计算张量 self 的梯度，支持指定维度 dim 和保持维度 keepdim
Tensor norm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const optional<Scalar>& p_,
    Tensor norm,
    IntArrayRef dim,
    bool keepdim);

// 根据 norm 张量计算张量 grad 的梯度，支持指定 norm 张量的计算
Tensor norm_jvp(
    const Tensor& grad,
    const Tensor& self,
    const optional<Scalar>& p_,
    Tensor norm);

// 从填充的张量反向生成嵌套张量
Tensor _nested_from_padded_backward(
    const Tensor& grad,
    const Tensor& input,
    const bool do_transform_0213);
// 计算线性操作的双向传播梯度，返回包含三个张量的元组
std::tuple<Tensor, Tensor, Tensor> linear_double_backward(
    const variable_list& grads,      // 梯度列表
    const Tensor& self,              // 自身张量
    const Tensor& grad_output,       // 梯度输出
    const Tensor& weight);           // 权重张量

// 计算线性代数中向量范数的 JVP（Jacobian-Vector Product）
Tensor linalg_vector_norm_jvp(
    const Tensor& self_p,            // 输入张量
    const Tensor& self_t,            // 输入张量的转置
    const Scalar& scalar_ord,        // 标量指数
    Tensor norm,                     // 范数张量
    const at::OptionalIntArrayRef& opt_dim,  // 可选的维度
    bool keepdim);                   // 是否保持维度

// 计算向量范数的反向传播
at::Tensor linalg_vector_norm_backward(
    at::Tensor grad,                 // 梯度
    const at::Tensor& self,          // 输入张量
    const at::Scalar& ord,           // 标量指数
    at::Tensor norm,                 // 范数张量
    const at::OptionalIntArrayRef& opt_dim,  // 可选的维度
    bool keepdim);                   // 是否保持维度

// 计算幂操作对输入张量的反向传播
at::Tensor pow_backward(
    at::Tensor grad,                 // 梯度
    const at::Tensor& self,          // 输入张量
    const at::Scalar& exponent_);    // 幂指数

// 计算幂操作对自身张量的反向传播
at::Tensor pow_backward_self(
    const at::Tensor& grad,          // 梯度
    const at::Tensor& self,          // 自身张量
    const at::Tensor& exponent);     // 幂指数

// 计算幂操作对指数的反向传播
at::Tensor pow_backward_exponent(
    const at::Tensor& grad,          // 梯度
    const at::Tensor& self,          // 输入张量
    const at::Tensor& exponent,      // 幂指数
    const at::Tensor& result);       // 结果张量

// 计算幂操作对基数的反向传播
at::Tensor pow_backward_exponent(
    const at::Tensor& grad,          // 梯度
    const at::Scalar& base,          // 基数
    const at::Tensor& exponent,      // 幂指数
    const at::Tensor& result);       // 结果张量

// 计算角度函数对输入张量的反向传播
at::Tensor angle_backward(
    const at::Tensor& grad,          // 梯度
    const at::Tensor& self);         // 输入张量

// 计算张量乘法对输入张量的反向传播
template <typename T>
at::Tensor mul_tensor_backward(
    const Tensor& grad,              // 梯度
    T other,                         // 其他张量或标量
    ScalarType self_st);             // 自身张量的数据类型

// 计算张量除法对自身张量的反向传播
template <typename T>
at::Tensor div_tensor_self_backward(
    const Tensor& grad,              // 梯度
    T other,                         // 其他张量或标量
    ScalarType self_st);             // 自身张量的数据类型

// 计算张量除法对其他张量的反向传播
at::Tensor div_tensor_other_backward(
    const Tensor& grad,              // 梯度
    const Tensor& self,              // 自身张量
    const Tensor& other);            // 其他张量

// 带舍入模式的张量除法对自身张量的反向传播
template <typename T>
at::Tensor div_tensor_self_backward(
    const Tensor& grad,              // 梯度
    T other,                         // 其他张量或标量
    ScalarType self_st,
    const std::optional<c10::string_view>& rounding_mode);  // 舍入模式

// 带舍入模式的张量除法对其他张量的反向传播
at::Tensor div_tensor_other_backward(
    const Tensor& grad,              // 梯度
    const Tensor& self,              // 自身张量
    const Tensor& other,             // 其他张量
    const std::optional<c10::string_view>& rounding_mode);  // 舍入模式

// 计算多元 Gamma 函数的反向传播
at::Tensor mvlgamma_backward(
    const at::Tensor& grad,          // 梯度
    const at::Tensor& self,          // 输入张量
    int64_t p);                      // 参数 p

// 计算维度置换操作的反向传播
at::Tensor permute_backwards(
    const at::Tensor& grad,          // 梯度
    at::IntArrayRef fwd_dims);       // 正向维度

// 计算弧度转角度函数的反向传播
at::Tensor rad2deg_backward(
    const at::Tensor& grad);         // 梯度

// 计算角度转弧度函数的反向传播
at::Tensor deg2rad_backward(
    const at::Tensor& grad);         // 梯度

// 计算在多个维度上扩展张量的反向传播
at::Tensor unsqueeze_multiple(
    const at::Tensor& t,             // 输入张量
    at::OptionalIntArrayRef opt_dim, // 可选的维度
    size_t n_dims);                  // 扩展的维度数目

// 计算求和操作的反向传播
at::Tensor sum_backward(
    const at::Tensor& grad,          // 梯度
    at::SymIntArrayRef sizes,        // 大小列表
    at::OptionalIntArrayRef opt_dims, // 可选的维度
    bool keepdim);                   // 是否保持维度

// 计算求和操作的反向传播（带指定维度）
at::Tensor sum_backward(
    const at::Tensor& grad,          // 梯度
    c10::SymIntArrayRef sizes,       // 大小列表
    c10::IntArrayRef dims,           // 维度列表
    bool keepdim);                   // 是否保持维度

// 计算 NaN 安全求和操作的反向传播
at::Tensor nansum_backward(
    const at::Tensor& grad,          // 梯度
    const at::Tensor& self,          // 输入张量
    at::OptionalIntArrayRef dims,    // 可选的维度
    bool keepdim);                   // 是否保持维度

// 反转整数列表
std::vector<int64_t> reverse_list(const at::IntArrayRef list);  // 整数列表

// 反转 SymInt 列表
std::vector<c10::SymInt> reverse_list_symint(const c10::SymIntArrayRef list);  // SymInt 列表

// 反转张量的指定维度
at::Tensor reverse_dim(const at::Tensor& t,  // 输入张量
                       int64_t dim);         // 反转的维度

// 计算安全乘积（避免零值）的反向传播
at::Tensor prod_safe_zeros_backward(
    const at::Tensor& grad,          // 梯度
    const at::Tensor& inp,           // 输入张量
    int64_t dim);                    // 维度
// 计算 prod 操作的反向传播
at::Tensor prod_backward(
    const at::Tensor& grad,  // 输入的梯度张量
    const at::Tensor& input,  // 输入张量
    const at::Tensor& result);  // 结果张量

// 计算带有维度参数的 prod 操作的反向传播
at::Tensor prod_backward(
    at::Tensor grad,  // 输入的梯度张量（复制）
    const at::Tensor& input,  // 输入张量
    at::Tensor result,  // 结果张量（复制）
    int64_t dim,  // 沿指定维度的维度参数
    bool keepdim);  // 是否保持维度

// 计算 solve 操作的 JVP（Jacobian-vector product）
at::Tensor solve_jvp(
    const Tensor& X,  // 输入张量 X
    const Tensor& A,  // 输入张量 A
    const Tensor& dA,  // 输入张量 dA
    const Tensor& dB);  // 输入张量 dB

// 计算 solve 操作对自身的反向传播
at::Tensor solve_backward_self(
    const at::Tensor& grad,  // 输入的梯度张量
    const at::Tensor& self,  // 自身张量
    const at::Tensor& A);  // 输入张量 A

// 计算 solve 操作对 A 的反向传播
at::Tensor solve_backward_A(
    const at::Tensor& grad,  // 输入的梯度张量
    const at::Tensor& self,  // 自身张量
    const at::Tensor& A,  // 输入张量 A
    const at::Tensor& solution);  // 解张量

// 计算 cumsum 操作的反向传播
at::Tensor cumsum_backward(const at::Tensor& grad,  // 输入的梯度张量
                           int64_t dim);  // 沿指定维度的维度参数

// 计算 logsumexp 操作的反向传播
at::Tensor logsumexp_backward(
    at::Tensor grad,  // 输入的梯度张量（复制）
    const at::Tensor& self,  // 自身张量
    at::Tensor result,  // 结果张量（复制）
    at::IntArrayRef dim,  // 沿多个维度的维度参数
    bool keepdim);  // 是否保持维度

// 计算 logsumexp 操作的 JVP
at::Tensor logsumexp_jvp(
    const at::Tensor& self_p,  // 输入张量 self 的偏导数
    const at::Tensor& self_t,  // 输入张量 self 的切线方向
    IntArrayRef dim,  // 沿多个维度的维度参数
    bool keepdim);  // 是否保持维度

// 计算 logcumsumexp 操作的反向传播
at::Tensor logcumsumexp_backward(
    at::Tensor grad,  // 输入的梯度张量（复制）
    const at::Tensor& self,  // 自身张量
    at::Tensor result,  // 结果张量（复制）
    int64_t dim);  // 沿指定维度的维度参数

// 计算 logcumsumexp 操作的 JVP
at::Tensor logcumsumexp_jvp(
    const at::Tensor& self_p,  // 输入张量 self 的偏导数
    const at::Tensor& self_t,  // 输入张量 self 的切线方向
    int64_t dim);  // 沿指定维度的维度参数

// 计算 unbind 操作的反向传播
at::Tensor unbind_backward(
    const variable_list& grads,  // 梯度列表
    int64_t dim);  // 沿指定维度的维度参数

// 计算嵌套式 unbind 操作的反向传播
at::Tensor unbind_backward_nested(
    const variable_list& grads,  // 梯度列表
    const Tensor& nt_sizes,  // 尺寸张量
    int64_t dim,  // 沿指定维度的维度参数
    const at::TensorOptions& options);  // 张量选项

// 计算不规则嵌套式 unbind 操作的反向传播
at::Tensor unbind_backward_nested_jagged(
    const variable_list& grads,  // 梯度列表
    const Tensor& self,  // 自身张量
    int64_t dim);  // 沿指定维度的维度参数

// 扩展张量维度到指定符号尺寸
at::Tensor unsqueeze_to(
    const at::Tensor& self,  // 自身张量
    c10::SymIntArrayRef sym_sizes);  // 符号尺寸引用

// 扩展张量维度到指定符号尺寸和维度
at::Tensor unsqueeze_to(
    const at::Tensor& self,  // 自身张量
    int64_t dim,  // 维度参数
    c10::SymIntArrayRef sym_sizes);  // 符号尺寸引用

// 扩展张量维度到指定符号尺寸和多个维度
at::Tensor unsqueeze_to(
    const at::Tensor& self,  // 自身张量
    IntArrayRef dim,  // 多个维度参数
    c10::SymIntArrayRef sym_sizes);  // 符号尺寸引用

// 计算 cat_tensors 操作的反向传播
std::vector<at::Tensor> cat_tensors_backward(
    const at::Tensor& grad,  // 输入的梯度张量
    const std::vector<std::vector<c10::SymInt>>& sizes,  // 尺寸列表
    const std::vector<ScalarType>& dtypes,  // 数据类型列表
    int64_t dim);  // 沿指定维度的维度参数

// 计算 stack_tensors 操作的反向传播
std::vector<at::Tensor> stack_tensors_backward(
    const at::Tensor& grad,  // 输入的梯度张量
    int64_t dim,  // 沿指定维度的维度参数
    const std::vector<ScalarType>& dtypes);  // 数据类型列表

// 计算 block_diag 操作的反向传播
std::vector<at::Tensor> block_diag_backward(
    const at::Tensor& grad,  // 输入的梯度张量
    const std::vector<std::vector<int64_t>>& sizes,  // 尺寸列表
    const std::vector<ScalarType>& dtypes);  // 数据类型列表

// 计算 clamp 操作的反向传播
at::Tensor clamp_backward(
    const at::Tensor& grad,  // 输入的梯度张量
    const at::Tensor& self,  // 自身张量
    const optional<at::Scalar>& min,  // 最小值（可选）
    const optional<at::Scalar>& max);  // 最大值（可选）

// 计算 clamp 操作的反向传播（使用张量作为最小和最大值）
at::Tensor clamp_backward(
    const at::Tensor& grad,  // 输入的梯度张量
    const at::Tensor& self,  // 自身张量
    const at::Tensor& min,  // 最小值张量
    const at::Tensor& max);  // 最大值张量

// 计算 clamp 操作的最小和最大值的反向传播
std::tuple<at::Tensor, at::Tensor> clamp_backward_min_max(
    const at::Tensor& grad,  // 输入的梯度张量
    const at::Tensor& self,  // 自身张量
    const at::Tensor& min,  // 最小值张量
    const at::Tensor& max,  // 最大值张量
    const std::array<bool, 2>&);  // 两个布尔值的数组

// 计算 clamp 操作的 JVP
at::Tensor clamp_jvp(
    const Tensor& self_p,  // 输入张量 self 的偏导数
    const Tensor& self_t,  // 输入张量 self 的切线方向
    const Tensor& min_p,  // 输入张量 min 的偏导数
    const Tensor& min_t,  // 输入张量 min 的切线方向
    const Tensor& max_p,  // 输入张量 max 的偏导数
// 计算输入张量的步长或返回错误，返回符号整数数组引用
at::SymIntArrayRef strides_or_error(
    const Tensor& input,
    c10::string_view const& input_name);

// 计算矩阵乘法 mm 的第一个矩阵的梯度
at::Tensor mm_mat1_backward(
    const Tensor& grad,
    const Tensor& mat2,
    at::SymIntArrayRef mat1_sizes,
    at::SymIntArrayRef mat1_strides,
    c10::Layout mat1_layout,
    const Scalar& alpha);

// 计算矩阵乘法 mm 的第二个矩阵的梯度
at::Tensor mm_mat2_backward(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    at::SymIntArrayRef sizes,
    at::SymIntArrayRef strides,
    c10::Layout layout,
    const at::Scalar& alpha);

// 计算稀疏矩阵乘法 mm_mat1_sparse 的第一个矩阵的梯度
at::Tensor mm_mat1_sparse_backward(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha);

// 计算稀疏矩阵加法 sparse_sampled_addmm 的反向传播
std::tuple<Tensor, Tensor, Tensor> sparse_sampled_addmm_backward(
    const Tensor& grad,
    const Tensor& self,
    const std::optional<Tensor>& mat1,
    const std::optional<Tensor>& mat2,
    const Scalar& alpha,
    const Scalar& beta,
    const std::array<bool, 3>& grad_input_mask);

// 计算稀疏矩阵掩码 sparse_mask 的反向传播
at::Tensor sparse_mask_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    c10::Layout self_layout);

// 计算稀疏矩阵乘法 sparse_sparse_matmul 的反向传播
at::Tensor sparse_sparse_matmul_backward(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    int64_t grad_order);

// 计算 renorm 操作的反向传播
at::Tensor renorm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm);

// 计算 renorm 操作的 JVP（Jacobian Vector Product）
at::Tensor renorm_jvp(
    const at::Tensor& self_p,
    const at::Tensor& self_t,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm);

// 计算 repeat 操作的反向传播
at::Tensor repeat_backward(
    at::Tensor grad,
    at::SymIntArrayRef repeats,
    at::SymIntArrayRef input_shape);

// 计算融合 dropout 操作的反向传播
at::Tensor _fused_dropout_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    double p1m);

// 计算可无限微分的本地 dropout 操作的反向传播
at::Tensor infinitely_differentiable_native_dropout_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    double scale);

// 计算本地 dropout 操作的双向反向传播
at::Tensor native_dropout_double_backward(
    const at::Tensor& ggI,
    const at::Tensor& grad,
    const at::Tensor& mask,
    double scale);

// 计算 evenly_distribute 操作的反向传播
at::Tensor evenly_distribute_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& value);

// 计算 sgn 函数的反向传播
Tensor sgn_backward(const Tensor& x, const Tensor& gx, const Tensor& sgn);

// 计算 masked_fill 操作的反向传播
Tensor masked_fill_backward(const Tensor& grad, const Tensor& mask);

// 计算 var 操作的反向传播
at::Tensor var_backward(
    at::Tensor grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);

// 计算 var 操作的 JVP（Jacobian Vector Product）
at::Tensor var_jvp(
    const at::Tensor& self_t,
    const at::Tensor& self_p,
    const at::Tensor& result,
    at::OptionalIntArrayRef dim_opt,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);

// 计算 std 操作的反向传播
at::Tensor std_backward(
    const at::Tensor& result,
    const at::Tensor& grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);

// 计算 mean 操作的反向传播
Tensor mean_backward(
    const Tensor& grad,
    c10::SymIntArrayRef shape,
    at::OptionalIntArrayRef dim,
    bool keepdim);
    // opt_dim: 可选的整数数组引用，表示操作的维度
    // numel: 符号化整数，表示张量的元素数量
    // keepdim: 布尔值，指示操作是否保持维度
    at::OptionalIntArrayRef opt_dim,
    c10::SymInt numel,
    bool keepdim);
// 定义函数 var_mean_backward，计算变量的梯度相对于均值和自身的反向传播
Tensor var_mean_backward(
    const Tensor& gvar,  // 输入参数：变量相对于梯度的梯度
    const Tensor& gmean,  // 输入参数：均值相对于梯度的梯度
    const Tensor& self,  // 输入参数：自身张量
    at::OptionalIntArrayRef dim_opt,  // 输入参数：可选的维度
    const std::optional<c10::Scalar>& correction,  // 输入参数：可选的修正因子
    bool keepdim);  // 输入参数：是否保持维度

// 定义函数 std_mean_backward，计算标准差的梯度相对于均值和自身的反向传播
Tensor std_mean_backward(
    const Tensor& gstd,  // 输入参数：标准差相对于梯度的梯度
    const Tensor& gmean,  // 输入参数：均值相对于梯度的梯度
    const Tensor& self,  // 输入参数：自身张量
    const Tensor& std,  // 输入参数：标准差张量
    at::OptionalIntArrayRef dim_opt,  // 输入参数：可选的维度
    const std::optional<c10::Scalar>& correction,  // 输入参数：可选的修正因子
    bool keepdim);  // 输入参数：是否保持维度

// 定义函数 cholesky_backward，计算 Cholesky 分解的反向传播
at::Tensor cholesky_backward(
    const at::Tensor& grad,  // 输入参数：梯度
    bool upper,  // 输入参数：是否是上三角矩阵
    const at::Tensor& L);  // 输入参数：Cholesky 分解的下三角矩阵

// 定义函数 cholesky_jvp，计算 Cholesky 分解的 JVP（Jacobian-Vector Product）
at::Tensor cholesky_jvp(
    const at::Tensor& input_tangent,  // 输入参数：输入的切线方向
    const at::Tensor& L,  // 输入参数：Cholesky 分解的下三角矩阵
    bool upper);  // 输入参数：是否是上三角矩阵

// 定义函数 cholesky_inverse_backward，计算 Cholesky 分解逆的反向传播
at::Tensor cholesky_inverse_backward(
    const at::Tensor& grad,  // 输入参数：梯度
    const at::Tensor& L,  // 输入参数：Cholesky 分解的下三角矩阵
    bool upper,  // 输入参数：是否是上三角矩阵
    const at::Tensor& inverse);  // 输入参数：Cholesky 分解逆矩阵

// 定义函数 cholesky_inverse_jvp，计算 Cholesky 分解逆的 JVP
at::Tensor cholesky_inverse_jvp(
    const at::Tensor& F,  // 输入参数：函数值
    const at::Tensor& dF,  // 输入参数：函数导数
    const at::Tensor& X,  // 输入参数：输入张量
    bool upper);  // 输入参数：是否是上三角矩阵

// 定义函数 pinv_jvp，计算矩阵伪逆的 JVP
Tensor pinv_jvp(
    const Tensor& A,  // 输入参数：输入矩阵 A
    const Tensor& pinvA,  // 输入参数：矩阵 A 的伪逆
    const Tensor& dA);  // 输入参数：输入矩阵的导数

// 定义函数 pinv_backward，计算矩阵伪逆的反向传播
Tensor pinv_backward(
    const Tensor& grad,  // 输入参数：梯度
    const Tensor& pinvA,  // 输入参数：矩阵 A 的伪逆
    const Tensor& A);  // 输入参数：输入矩阵 A

// 定义函数 split_with_sizes_backward，计算按大小分割的反向传播
at::Tensor split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,  // 输入参数：梯度向量
    c10::SymIntArrayRef split_sizes,  // 输入参数：分割大小
    int64_t dim,  // 输入参数：分割的维度
    c10::SymIntArrayRef sizes,  // 输入参数：总大小
    const at::TensorOptions& options);  // 输入参数：张量选项

// 定义函数 _nested_split_with_sizes_backward，计算嵌套大小分割的反向传播
at::Tensor _nested_split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,  // 输入参数：梯度向量
    c10::SymIntArrayRef split_sizes,  // 输入参数：分割大小
    int64_t dim,  // 输入参数：分割的维度
    const Tensor& nt_sizes,  // 输入参数：嵌套大小
    const at::TensorOptions& options);  // 输入参数：张量选项

// 定义函数 split_backward，计算分割的反向传播
at::Tensor split_backward(
    const std::vector<torch::autograd::Variable>& grads,  // 输入参数：梯度向量
    const c10::SymInt& split_size,  // 输入参数：分割大小
    int64_t dim,  // 输入参数：分割的维度
    c10::SymIntArrayRef sizes,  // 输入参数：总大小
    const at::TensorOptions& options);  // 输入参数：张量选项

// 定义函数 max_pool_double_backward，计算最大池化双向传播的反向传播
at::Tensor max_pool_double_backward(
    const at::Tensor& grad,  // 输入参数：梯度
    const at::Tensor& indices,  // 输入参数：最大值索引
    int dim);  // 输入参数：池化的维度

// 定义函数 error_for_max_pool2d_double_backward，计算最大池化双向传播的错误
at::Tensor error_for_max_pool2d_double_backward();

// 定义函数 glu_double_backward，计算 GLU 激活函数的双向传播
at::Tensor glu_double_backward(
    const at::Tensor& grad,  // 输入参数：梯度
    const at::Tensor& grad_output,  // 输入参数：输出梯度
    const at::Tensor& input,  // 输入参数：输入张量
    int64_t dim);  // 输入参数：GLU 的维度

// 定义函数 glu_double_backward_grad_output，计算 GLU 激活函数输出梯度的双向传播
at::Tensor glu_double_backward_grad_output(
    const at::Tensor& grad,  // 输入参数：梯度
    const at::Tensor& input,  // 输入参数：输入张量
    int64_t dim);  // 输入参数：GLU 的维度

// 定义函数 infinitely_differentiable_silu_backward，计算 SiLU 激活函数的无限可微反向传播
at::Tensor infinitely_differentiable_silu_backward(
    const at::Tensor& grad_output,  // 输入参数：梯度输出
    const at::Tensor& input);  // 输入参数：输入张量

// 定义函数 infinitely_differentiable_mish_backward，计算 Mish 激活函数的无限可微反向传播
at::Tensor infinitely_differentiable_mish_backward(
    const at::Tensor& grad_output,  // 输入参数：梯度输出
    const at::Tensor& input);  // 输入参数：输入张量

// 定义函数 infinitely_differentiable_logit_backward，计算 Logit 函数的无限可微反向传播
Tensor infinitely_differentiable_logit_backward(
    const Tensor& grad,  // 输入参数：梯度
    const Tensor& self,  // 输入参数：自身张量
    std::optional<double> eps);  // 输入参数：可选的 epsilon 值

// 定义函数 binary_cross_entropy_target_backward，计算二元交叉熵损失相对于目标的反向传播
Tensor binary_cross_entropy_target_backward(
    const Tensor& grad,  // 输入参数：梯度
    const Tensor& self,  // 输入参数：自
    # 定义一个函数参数列表，包括目标张量、权重（可选）、正类权重（可选）和缩减方式
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t reduction);
# 计算二进制交叉熵损失相对于目标变量的梯度
at::Tensor binary_cross_entropy_with_logits_target_backward(
    const at::Tensor& grad_output,  // 对输出梯度的输入
    const at::Tensor& self,         // 输入张量
    const at::Tensor& target,       // 目标张量
    const std::optional<at::Tensor>& weight,  // 权重张量（可选）
    const std::optional<at::Tensor>& pos_weight,  // 正权重张量（可选）
    int64_t reduction);             // 损失减少方式

# 计算对数sigmoid函数的双向传播的梯度
at::Tensor log_sigmoid_double_backward(
    const at::Tensor& grad,  // 对输出梯度的输入
    const at::Tensor& input);  // 输入张量

# 计算softmax函数的双向传播的梯度
at::Tensor softmax_double_backward(
    const at::Tensor& grad,      // 相对于输出梯度的输入
    const at::Tensor& grad_output,  // 输出梯度
    int dim,                     // 维度
    const at::Tensor& output);   // 输出张量

# 计算二进制交叉熵损失函数相对于输入和目标的双向传播的梯度
at::Tensor binary_cross_entropy_double_backward(
    const at::Tensor& grad_output,  // 对输出梯度的输入
    const at::Tensor& grad,         // 对输入梯度的输入
    const at::Tensor& input,        // 输入张量
    const at::Tensor& target,       // 目标张量
    const std::optional<at::Tensor>& weight,  // 权重张量（可选）
    int64_t reduction);             // 损失减少方式

# 计算二进制交叉熵损失函数相对于输出梯度的双向传播的梯度
at::Tensor binary_cross_entropy_double_backward_grad_output(
    const at::Tensor& grad,         // 对输出梯度的输入
    const at::Tensor& input,        // 输入张量
    const at::Tensor& target,       // 目标张量
    const std::optional<at::Tensor>& weight,  // 权重张量（可选）
    int64_t reduction);             // 损失减少方式

# 计算平滑L1损失函数相对于输入的双向传播的梯度
at::Tensor smooth_l1_loss_double_backward(
    const at::Tensor& grad,    // 对输出梯度的输入
    const at::Tensor& input,   // 输入张量
    const at::Tensor& target,  // 目标张量
    int64_t reduction,         // 损失减少方式
    double beta);              // β参数

# 计算Huber损失函数相对于输入的双向传播的梯度
at::Tensor huber_loss_double_backward(
    const at::Tensor& grad,    // 对输出梯度的输入
    const at::Tensor& input,   // 输入张量
    const at::Tensor& target,  // 目标张量
    int64_t reduction,         // 损失减少方式
    double delta);             // δ参数

# 计算Huber损失函数相对于输出梯度的双向传播的梯度
at::Tensor huber_loss_double_backward_grad_output(
    const at::Tensor& grad,         // 对输出梯度的输入
    const at::Tensor& grad_output,  // 输出梯度
    const at::Tensor& input,        // 输入张量
    const at::Tensor& target,       // 目标张量
    int64_t reduction,              // 损失减少方式
    double delta);                  // δ参数

# 计算均方误差损失函数相对于输入的双向传播的梯度
at::Tensor mse_loss_double_backward(
    const at::Tensor& grad,    // 对输出梯度的输入
    const at::Tensor& input,   // 输入张量
    int64_t reduction);        // 损失减少方式

# 计算软间隔损失函数相对于输入的双向传播的梯度
at::Tensor soft_margin_loss_double_backward(
    const at::Tensor& grad,    // 对输出梯度的输入
    const at::Tensor& input,   // 输入张量
    const at::Tensor& target,  // 目标张量
    int64_t reduction);        // 损失减少方式

# 计算软间隔损失函数相对于输出梯度的双向传播的梯度
at::Tensor soft_margin_loss_double_backward_grad_output(
    const at::Tensor& grad,         // 对输出梯度的输入
    const at::Tensor& grad_output,  // 输出梯度
    const at::Tensor& input,        // 输入张量
    const at::Tensor& target,       // 目标张量
    int64_t reduction);             // 损失减少方式

# 计算softplus函数相对于输入的双向传播的梯度
at::Tensor softplus_double_backward(
    const at::Tensor& grad,        // 对输出梯度的输入
    const at::Tensor& input,       // 输入张量
    const at::Scalar& beta,        // β参数
    const at::Scalar& threshold);  // 阈值参数

# 计算行列式对数Jacobian乘积的梯度
std::tuple<at::Tensor, at::Tensor> slogdet_jvp(
    const at::Tensor& LU,       // LU分解的输入
    const at::Tensor& pivots,   // 唯一标识
    const at::Tensor& dA,       // 矩阵的变化
    const at::Tensor& sign,     // 行列式的符号
    const bool use_A_T);       // 是否使用A的转置

# 计算行列式对数的反向传播的梯度
at::Tensor slogdet_backward(
    const at::Tensor& grad_sign,     // 行列式符号的梯度
    const at::Tensor& grad_logabsdet,// 行列式绝对值对数的梯度
    const at::Tensor& A,             // 输入矩阵A
    const at::Tensor& signdet,       // 行列式符号
    const at::Tensor& LU,            // LU分解
    const at::Tensor& pivots);       // 唯一标识

# 计算log1p函数的反向传播的梯度
at::Tensor log1p_backward(
    const at::Tensor& grad,   // 对输出梯度的输入
    const at::Tensor& self);  // 输入张量

# 计算sinc函数的反向传播的梯度
at::Tensor sinc_backward(
    const at::Tensor& grad,   // 对输出梯度的输入
    const at::Tensor& self);  // 输入张量

# 计算稀疏构造器值的反向传播的梯度
at::Tensor sparse_constructor_values_backward(
    const at::Tensor& sparse_grad_out,  // 稀疏梯度的输出
    const at::Tensor& indices);         // 索引

# 计算嵌入器密集表示的双向传播的梯度（对称整数）
at::Tensor embedding_dense_double_backward_symint(
    const at::Tensor& grad,          // 对输出梯度的输入
    const at::Tensor& indices,       // 索引
    const c10::SymInt& padding_idx);// 填充索引

# 计算索引操作的反向传播的梯度
at::Tensor index_backward(
    // 定义一个名为 `zeros_like_self` 的 `at::Tensor` 对象，作为函数的第一个参数
    at::Tensor zeros_like_self,

    // 定义一个名为 `indices` 的 `torch::List`，其元素为可选的 `Tensor` 对象的引用，作为函数的第二个参数
    const torch::List<std::optional<Tensor>>& indices,

    // 定义一个名为 `grad` 的 `at::Tensor` 对象，作为函数的第三个参数
    const at::Tensor& grad);
// Compute the gradient of the CTC (Connectionist Temporal Classification) loss function with respect to input tensors.
at::Tensor _cudnn_ctc_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& loss,
    const at::Tensor& raw_grad,
    bool zero_infinity);

// Compute the double backward pass for the ELU (Exponential Linear Unit) activation function.
at::Tensor elu_double_backward(
    const Tensor& grad,
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result);

// Compute the backward pass for the Singular Value Decomposition (SVD) operation.
Tensor svd_backward(
    const Tensor& gU,
    const Tensor& gS,
    const Tensor& gVh,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh);

// Compute the Jacobian-vector product for the SVD operation.
std::tuple<Tensor, Tensor, Tensor> linalg_svd_jvp(
    const Tensor& dA,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const bool full_matrices);

// Compute the backward pass for slicing operation in a tensor.
Tensor slice_backward_wrapper(
    const at::Tensor& grad,
    const c10::SymIntArrayRef& input_sizes,
    int64_t dim,
    std::optional<c10::SymInt> start,
    std::optional<c10::SymInt> end,
    c10::SymInt step);

// Compute the Jacobian-vector product for the eigenvalue decomposition (EIG) operation.
std::tuple<Tensor, Tensor> linalg_eig_jvp(
    const Tensor& dA,
    const Tensor& L,
    const Tensor& V,
    const bool is_hermitian);

// Compute the backward pass for the eigenvalue decomposition (EIG) operation.
Tensor linalg_eig_backward(
    const Tensor& gL,
    const Tensor& gV,
    const Tensor& L,
    const Tensor& V,
    const bool is_hermitian,
    const bool symeig_eigenvectors = true);

// Compute the Jacobian-vector product for the least squares solution (LSTSQ) operation.
Tensor linalg_lstsq_jvp(
    const Tensor& A,
    const Tensor& B,
    const Tensor& dA,
    const Tensor& dB);

// Compute the backward pass for the triangular solve operation.
std::tuple<Tensor, Tensor> triangular_solve_backward(
    const Tensor& grad_x,
    const Tensor& grad_m,
    const Tensor& b,
    const Tensor& a,
    const Tensor& x,
    const bool upper,
    const bool transpose,
    const bool unitriangular,
    std::array<bool, 2> output_mask);

// Compute the Jacobian-vector product for the triangular solve operation.
Tensor triangular_solve_jvp(
    const Tensor& X,
    const Tensor& A,
    const Tensor& dA,
    const Tensor& dB,
    const bool upper,
    const bool transpose,
    const bool unitriangular);

// Compute the forward automatic differentiation pass for the triangular solve operation.
Tensor linalg_solve_triangular_forward_AD(
    const Tensor& A_t,
    const Tensor& B_t,
    const Tensor& A,
    const Tensor& X,
    const bool upper,
    const bool left,
    const bool unitriangular);

// Compute the backward pass for the triangular solve operation.
std::tuple<Tensor, Tensor> linalg_solve_triangular_backward(
    const Tensor& grad,
    const Tensor& A,
    const Tensor& X,
    const bool upper,
    const bool left,
    const bool unitriangular,
    std::array<bool, 2> output_mask);

// Compute the backward pass for trilinear interpolation operation.
std::tuple<Tensor, Tensor, Tensor> _trilinear_backward(
    const Tensor& grad_out,
    const std::optional<Tensor>& i1,
    const std::optional<Tensor>& i2,
    const std::optional<Tensor>& i3,
    IntArrayRef expand1,
    IntArrayRef expand2,
    IntArrayRef expand3,
    IntArrayRef sumdim,
    std::array<bool, 3> grad_mask);

// Compute the Jacobian-vector product for the QR (QR decomposition) operation.
std::tuple<Tensor, Tensor> linalg_qr_jvp(
    const Tensor& dA,
    const Tensor& Q,
    const Tensor& R,
    const c10::string_view mode);

// Compute the backward pass for the QR (QR decomposition) operation.
Tensor linalg_qr_backward(
    const Tensor& gQ,
    const Tensor& gR,
    const Tensor& Q,
    const Tensor& R,
    const c10::string_view mode);

// Compute the differential of the matrix exponential operation.
Tensor linalg_matrix_exp_differential(
    const Tensor& self,
    const Tensor& grad,
    bool adjoint);
// 定义函数 `batchnorm_double_backward`，用于计算批归一化层的双向传播
std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    // 输入张量
    const Tensor& input,
    // 可选的 gamma 参数
    const std::optional<Tensor>& gamma,
    // ggI 参数
    const Tensor& ggI,
    // ggG 参数
    const Tensor& ggG,
    // ggB 参数
    const Tensor& ggB,
    // gO 参数
    const Tensor& gO,
    // 可选的 running_mean 参数
    const std::optional<Tensor>& running_mean,
    // 可选的 running_var 参数
    const std::optional<Tensor>& running_var,
    // 训练标志
    bool training,
    // epsilon 参数
    double eps,
    // 可选的 save_mean 参数
    const std::optional<Tensor>& save_mean,
    // 可选的 save_invstd 参数
    const std::optional<Tensor>& save_invstd,
    // 输出掩码数组
    std::array<bool, 3> output_mask);

// 定义函数 `_euclidean_dist_backward`，用于计算欧几里得距离函数的反向传播
std::tuple<Tensor, Tensor> _euclidean_dist_backward(
    // 梯度张量
    const Tensor& grad,
    // 输入张量 x1
    const Tensor& x1,
    // 输入张量 x2
    const Tensor& x2,
    // 输出张量 res
    const Tensor& res);

// 定义函数 `fft_backward`，用于计算傅里叶变换的反向传播
Tensor fft_backward(
    // 输入张量 self
    const Tensor& self,
    // 梯度张量 grad
    const Tensor& grad,
    // 信号维度
    int64_t signal_ndim,
    // 是否为复数输入
    bool complex_input,
    // 是否为复数输出
    bool complex_output,
    // 是否为逆变换
    bool inverse,
    // 检查后的信号大小
    IntArrayRef checked_signal_sizes,
    // 归一化方式
    int64_t normalization,
    // 是否单边傅里叶变换
    bool onesided,
    // 输出大小数组
    IntArrayRef output_sizes);

// 定义函数 `fft_r2c_backward`，用于实现实数到复数傅里叶变换的反向传播
Tensor fft_r2c_backward(
    // 梯度张量 grad
    const Tensor& grad,
    // 维度数组 dim
    at::IntArrayRef dim,
    // 归一化方式
    int64_t normalization,
    // 是否单边傅里叶变换
    bool onesided,
    // 最后维度大小 SymInt
    const c10::SymInt& last_dim_size);

// 定义函数 `fft_c2r_backward`，用于实现复数到实数傅里叶变换的反向传播
Tensor fft_c2r_backward(
    // 梯度张量 grad
    const Tensor& grad,
    // 维度数组 dim
    IntArrayRef dim,
    // 归一化方式
    int64_t normalization);

// 定义函数 `constant_pad_nd_backward`，用于常数填充的反向传播
Tensor constant_pad_nd_backward(const Tensor& grad, c10::SymIntArrayRef pad);

// 定义函数 `cholesky_solve_backward`，用于 Cholesky 求解的反向传播
std::tuple<Tensor, Tensor> cholesky_solve_backward(
    // 梯度张量 grad_x
    const Tensor& grad_x,
    // 输入张量 self
    const Tensor& self,
    // 输入张量 input2
    const Tensor& input2,
    // 结果张量 result
    const Tensor& result,
    // 是否为上三角矩阵
    const bool upper,
    // 输出掩码数组
    std::array<bool, 2> output_mask);

// 定义函数 `cholesky_solve_jvp`，用于 Cholesky 求解的 JVP
Tensor cholesky_solve_jvp(
    // 输入张量 X
    const Tensor& X,
    // 输入张量 U
    const Tensor& U,
    // 输入张量 dU
    const Tensor& dU,
    // 输入张量 dB
    const Tensor& dB,
    // 是否为上三角矩阵
    const bool upper);

// 定义函数 `infinitely_differentiable_native_group_norm_backward`，用于无限可微的本地分组归一化层的反向传播
std::tuple<Tensor, Tensor, Tensor> infinitely_differentiable_native_group_norm_backward(
    // 梯度张量 dY
    const Tensor& dY,
    // 均值梯度张量 dmean
    const Tensor& dmean,
    // 标准差梯度张量 drstd
    const Tensor& drstd,
    // 输入张量 X
    const Tensor& X,
    // 均值张量 mean
    const Tensor& mean,
    // 标准差张量 rstd
    const Tensor& rstd,
    // 可选的 gamma 参数
    const std::optional<Tensor>& gamma,
    // SymInt N
    c10::SymInt N,
    // SymInt C
    const c10::SymInt& C,
    // SymInt HxW
    c10::SymInt HxW,
    // 分组数
    int64_t group,
    // epsilon 参数
    double eps,
    // 输出掩码数组
    std::array<bool, 3> grad_input_mask);

// 定义函数 `gelu_double_backward`，用于 GELU 激活函数的双向传播
Tensor gelu_double_backward(
    // ggI 参数
    const Tensor& ggI,
    // gO 参数
    const Tensor& gO,
    // 输入张量 input
    const Tensor& input,
    // 字符串视图参数 approximate
    c10::string_view approximate);

// 定义函数 `as_strided_backward`，用于 as_strided 函数的反向传播
Tensor as_strided_backward(
    // 梯度张量 grad
    Tensor grad,
    // 输入几何形状 input_geometry
    const TensorGeometry& input_geometry,
    // 大小数组 sizes
    c10::SymIntArrayRef sizes,
    // 步长数组 strides
    c10::SymIntArrayRef strides,
    // 可选的存储偏移量 storage_offset_
    const optional<c10::SymInt>& storage_offset_);

// 定义函数 `as_strided_scatter_backward`，用于 as_strided_scatter 函数的反向传播
Tensor as_strided_scatter_backward(
    // 梯度张量 grad
    const Tensor& grad,
    // 输入几何形状 input_geometry
    const TensorGeometry& input_geometry,
    // 源几何形状 src_geometry
    const TensorGeometry& src_geometry,
    // 大小数组 sizes
    c10::SymIntArrayRef sizes,
    // 步长数组 strides
    c10::SymIntArrayRef strides,
    // 可选的存储偏移量 storage_offset
    optional<c10::SymInt> storage_offset);

// 定义函数 `atan2_backward`，用于 atan2 函数的反向传播
std::tuple<Tensor, Tensor> atan2_backward(
    // 梯度张量 grad
    const Tensor& grad,
    // 输入张量 self
    const Tensor& self,
    // 输入张量 other
    const Tensor& other,
    // 输出掩码数组
    std::array<bool, 2> output_mask);

// 定义函数 `amaxamin_jvp`，用于计算 amaxamin 函数的 JVP
Tensor amaxamin_jvp(
    // 输入张量 x
    const Tensor& x,
    // 输入张量 dx
    const Tensor& dx,
    // 结果张量 result
    const Tensor& result,
    // 维度数组 dim
    IntArrayRef dim,
    // 是否保持维度
    bool keepdim);

// 定义函数 `layer_norm_double_backward`，用于层归一化层的双向传播
std::tuple<Tensor, Tensor, Tensor> layer_norm_double_backward(
    // 输入张量 input
    const Tensor& input,
    // 可选的 gamma 参数
    const std::optional<Tensor>& gamma,
    // 声明一个函数，接受多个输入参数，并且这些参数都是 Tensor 类型的常量引用
    const Tensor& ggI,             // 输入参数，表示某种张量
    const Tensor& ggG,             // 输入参数，表示某种张量
    const Tensor& ggB,             // 输入参数，表示某种张量
    const Tensor& gO,              // 输入参数，表示某种张量
    const Tensor& save_mean,       // 输入参数，表示某种张量
    const Tensor& save_invstd,     // 输入参数，表示某种张量
    c10::SymIntArrayRef normalized_shape,  // 输入参数，表示一种特定类型的数组引用
    std::array<bool, 3> output_mask      // 输入参数，表示包含三个布尔值的固定大小数组
    );
// 计算 Householder product 的反向传播，根据梯度 grad、结果 result、输入 input 和 tau，可能翻转计算顺序
std::tuple<Tensor, Tensor> householder_product_backward(
    const Tensor& grad,
    const Tensor& result,
    const Tensor& input,
    const Tensor& tau,
    const bool flip_order = false);

// 计算 Householder product 的雅可比乘积，根据 dV、dtau、prod、V 和 tau
Tensor householder_product_jvp(
    const Tensor& dV,
    const Tensor& dtau,
    const Tensor& prod,
    const Tensor& V,
    const Tensor& tau);

// 计算 ormqr 操作的反向传播，根据梯度 grad、结果 result、self、tau、other、left、transpose 和 grad_output_mask
std::tuple<Tensor, Tensor, Tensor> ormqr_backward(
    const Tensor& grad,
    const Tensor& result,
    const Tensor& self,
    const Tensor& tau,
    const Tensor& other,
    bool left,
    bool transpose,
    std::array<bool, 3> grad_output_mask);

// 计算 polar 函数的反向传播，根据梯度 grad 和结果 result
std::tuple<Tensor, Tensor> polar_backward(
    const Tensor& grad,
    const Tensor& result);

// 计算 i1 函数的反向传播，根据梯度 grad、self 和结果 result
Tensor i1_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);

// 计算 i1e 函数的反向传播，根据梯度 grad、self 和结果 result
Tensor i1e_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);

// 使用 LU 分解求解线性方程组的反向传播，根据梯度 grad、LU 分解 LU、主元 pivots、右侧向量 X、left 和 adjoint
Tensor linalg_lu_solve_LU(
    const Tensor& grad,
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& X,
    const bool left,
    const bool adjoint);

// 使用 LU 分解求解线性方程组的雅可比乘积，根据 X、LU、dLU、dB、left 和 adjoint
Tensor linalg_lu_solve_jvp(
    const Tensor& X,
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& dLU,
    const Tensor& dB,
    const bool left,
    const bool adjoint);

// 计算 linalg_solve 函数的反向传播，根据梯度 gX、X、A、LU、pivots、left 和 B_requires_grad
std::tuple<Tensor, Tensor> linalg_solve_backward(
    const Tensor& gX,
    const Tensor& X,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots,
    const bool left,
    const bool B_requires_grad);

// 计算 linalg_solve 函数的雅可比乘积，根据 dA、dB、X、LU、pivots、left 和 use_A_T
Tensor linalg_solve_jvp(
    const Tensor& dA,
    const Tensor& dB,
    const Tensor& X,
    const Tensor& LU,
    const Tensor& pivots,
    const bool left,
    const bool use_A_T);

// LU 分解的解包操作的反向传播，根据 L_grad、U_grad、m 和 n
Tensor lu_unpack_backward(
    const Tensor& L_grad,
    const Tensor& U_grad,
    const c10::SymInt& m,
    const c10::SymInt& n);

// 计算行列式的反向传播，根据梯度 grad、行列式 det、矩阵 A、LU 分解 LU 和主元 pivots
Tensor linalg_det_backward(
    const Tensor& grad,
    const Tensor& det,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots);

// 计算行列式的雅可比乘积，根据 dA、det、LU、pivots 和 use_A_T
Tensor linalg_det_jvp(
    const Tensor& dA,
    const Tensor& det,
    const Tensor& LU,
    const Tensor& pivots,
    const bool use_A_T);

// 计算 linalg_lstsq 函数的反向传播，根据梯度 grad、矩阵 A、右侧向量 B_ 和 grad_input_mask
std::tuple<Tensor, Tensor> linalg_lstsq_backward(
    const Tensor& grad,
    const Tensor& A,
    const Tensor& B_,
    const std::array<bool, 2>& grad_input_mask);

// LU 分解的反向传播，根据 L_grad、U_grad、置换矩阵 P、下三角矩阵 L、上三角矩阵 U 和 pivot
Tensor linalg_lu_backward(
    const Tensor& L_grad,
    const Tensor& U_grad,
    const Tensor& P,
    const Tensor& L,
    const Tensor& U,
    const bool pivot);

// LU 分解的雅可比乘积，根据 dA、P、L、U 和 pivot
std::tuple<Tensor, Tensor> linalg_lu_jvp(
    const Tensor& dA,
    const Tensor& P,
    const Tensor& L,
    const Tensor& U,
    const bool pivot);

// LU 分解扩展的反向传播，根据梯度 grad、LU、主元 pivs 和 pivot
Tensor lu_factor_ex_backward(
    const Tensor& grad,
    const Tensor& LU,
    const Tensor& pivs,
    const bool pivot);

// LU 分解扩展的雅可比乘积，根据 dX、LU、主元 pivs 和 pivot
Tensor lu_factor_ex_jvp(
    const Tensor& dX,
    const Tensor& LU,
    const Tensor& pivs,
    const bool pivot);

// 批量归一化操作的雅可比乘积，根据 input_p、input_t、weight_p、weight_t、bias_p 和 bias_t
Tensor batch_norm_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t);
    // 参数1: 运行时均值的可选引用，用于在推断模式下更新均值
    const std::optional<Tensor>& running_mean,
    // 参数2: 运行时方差的可选引用，用于在推断模式下更新方差
    const std::optional<Tensor>& running_var,
    // 参数3: 保存的均值张量，用于在训练期间计算批标准化
    const Tensor& saved_mean,
    // 参数4: 保存的均值倒数的张量，用于在训练期间计算批标准化
    const Tensor& saved_invstd,
    // 参数5: 标识是否处于训练模式，影响到批标准化的计算
    bool train,
    // 参数6: 用于防止除以零的小值，通常很小的正数
    double eps);
// 计算带有 Layer Normalization 的 JVP（Jacobian Vector Product）
Tensor layer_norm_jvp(
    const Tensor& input_p,                  // 输入张量的原始值
    const Tensor& input_t,                  // 输入张量的微分值
    const Tensor& weight_p,                 // 权重张量的原始值
    const Tensor& weight_t,                 // 权重张量的微分值
    const Tensor& bias_p,                   // 偏置张量的原始值
    const Tensor& bias_t,                   // 偏置张量的微分值
    const Tensor& saved_mean,               // 保存的均值张量
    const Tensor& saved_invstd,             // 保存的标准差倒数张量
    c10::SymIntArrayRef normalized_shape);  // 归一化形状参数

// 计算带有 Group Normalization 的 JVP（Jacobian Vector Product）
Tensor group_norm_jvp(
    const Tensor& input_p,                  // 输入张量的原始值
    const Tensor& input_t,                  // 输入张量的微分值
    const Tensor& weight_p,                 // 权重张量的原始值
    const Tensor& weight_t,                 // 权重张量的微分值
    const Tensor& bias_p,                   // 偏置张量的原始值
    const Tensor& bias_t,                   // 偏置张量的微分值
    const Tensor& saved_mean,               // 保存的均值张量
    const Tensor& saved_invstd,             // 保存的标准差倒数张量
    int64_t groups);                        // 分组数量

// 计算带有 Group Normalization 的均值项的 JVP
Tensor group_norm_mean_jvp(
    const Tensor& input_t,                  // 输入张量的微分值
    const Tensor& mean_p,                   // 均值张量的原始值
    int64_t groups);                        // 分组数量

// 计算带有 Group Normalization 的标准差倒数项的 JVP
Tensor group_norm_invstd_jvp(
    const Tensor& input_p,                  // 输入张量的原始值
    const Tensor& input_t,                  // 输入张量的微分值
    const Tensor& mean_p,                   // 均值张量的原始值
    const Tensor& invstd_p,                 // 标准差倒数张量的原始值
    int64_t groups);                        // 分组数量

// 计算卷积操作的 JVP
Tensor convolution_jvp(
    const Tensor& input_p,                  // 输入张量的原始值
    const Tensor& input_t,                  // 输入张量的微分值
    const Tensor& weight_p,                 // 权重张量的原始值
    const Tensor& weight_t,                 // 权重张量的微分值
    const Tensor& bias_p,                   // 偏置张量的原始值
    const Tensor& bias_t,                   // 偏置张量的微分值
    at::SymIntArrayRef stride,              // 步长参数
    at::SymIntArrayRef padding,             // 填充参数
    at::SymIntArrayRef dilation,            // 膨胀参数
    bool transposed,                        // 是否为转置卷积
    at::SymIntArrayRef output_padding,      // 输出填充参数
    const c10::SymInt& groups);             // 分组数量

// 计算带有扩展选项的卷积操作的 JVP
Tensor _convolution_jvp(
    const Tensor& input_p,                  // 输入张量的原始值
    const Tensor& input_t,                  // 输入张量的微分值
    const Tensor& weight_p,                 // 权重张量的原始值
    const Tensor& weight_t,                 // 权重张量的微分值
    const Tensor& bias_p,                   // 偏置张量的原始值
    const Tensor& bias_t,                   // 偏置张量的微分值
    at::SymIntArrayRef stride,              // 步长参数
    at::SymIntArrayRef padding,             // 填充参数
    at::SymIntArrayRef dilation,            // 膨胀参数
    bool transposed,                        // 是否为转置卷积
    at::SymIntArrayRef output_padding,      // 输出填充参数
    const c10::SymInt& groups,              // 分组数量
    bool benchmark,                         // 是否使用基准模式
    bool deterministic,                     // 是否使用确定性模式
    bool cudnn_enabled,                     // 是否启用 CuDNN
    bool allow_tf32);                       // 是否允许 TF32 加速

// 计算卷积操作中偏置项的 JVP
Tensor convolution_backward_jvp_grad_bias(
    const Tensor& grad_out_t,               // 梯度输出的微分值
    const Tensor& grad_bias);               // 偏置项的原始值

// 拼接操作的 JVP
Tensor cat_jvp(
    const at::ITensorListRef& tensors,      // 张量列表的引用
    int64_t dim);                           // 拼接的维度

// 块对角矩阵操作的 JVP
Tensor block_diag_jvp(
    at::TensorList tensors);                // 张量列表

// 堆叠操作的 JVP
Tensor stack_jvp(
    at::TensorList tensors,                 // 张量列表
    int64_t dim);                           // 堆叠的维度

// 累积乘积操作的 JVP
Tensor cumprod_jvp(
    const Tensor& self_t,                   // 输入张量的微分值
    const Tensor& self_p,                   // 输入张量的原始值
    const Tensor& result,                   // 累积乘积结果张量
    int dim);                               // 累积的维度

// 使用保持维度的索引进行 gather 操作的 JVP
Tensor gather_with_keepdimed_indices(
    const Tensor& input,                    // 输入张量
    int64_t dim,                            // gather 的维度
    const Tensor& indices,                  // 索引张量
    bool keepdim);                          // 是否保持维度

// 均匀地读取梯度的 JVP
Tensor evenly_read_jvp(
    const Tensor& fw_grad,                  // 前向梯度张量
    const Tensor& input,                    // 输入张量
    const Tensor& value);                   // 值张量

// 警告后向传播的 JVP
Tensor warn_backwards(
    const Tensor& grad_output);             // 梯度输出张量

// CuDNN 卷积反向传播的 JVP
std::tuple<Tensor, Tensor> _cudnn_convolution_backward(
    const at::Tensor& self,                 // 输入张量
    const at::Tensor& grad_output,          // 梯度输出张量
    const at::Tensor& weight,               // 权重张量
    at::SymIntArrayRef padding,             // 填充参数
    at::SymIntArrayRef output_padding,      // 输出填充参数
    at::SymIntArrayRef stride,              // 步长参数
    at::SymIntArrayRef dilation,            // 膨胀参数
    bool transposed,                        // 是否为转置卷积
    c10::SymInt groups,                     // 分组数量
    ::std::array<bool, 2> output_mask);     // 输出掩码数组

// 散射操作的约减 JVP
Tensor scatter_reduce_jvp(
    const Tensor& self_p,                   // 输入张量的原始值
    const Tensor& self_t,                   // 输入张量的微分值
    int dim,                                // 约减的维度
    const Tensor& index,                    // 索引张量
    const Tensor& src_p,                    // 源张量的原始值
    const Tensor& src_t,                    // 源张量的微分值
    c10::string_view reduce,                // 约减操作类型
    bool include_self,                      // 是否包含自身
    const Tensor& result);                  // 结果张量

// 散射约减操作的反向传播
std::tuple<Tensor, Tensor> scatter_reduce_backward(
    const Tensor& grad,                     // 梯度张量
    const Tensor& index,                    // 索引张量
    int64_t dim,                            // 约减的维度
    bool reduce_size);                      // 是否
    // 定义函数的参数列表，分别为梯度张量、自身张量、维度、索引张量、源张量、缩减方式、是否包含自身、以及结果张量。
    const Tensor& grad,             // 梯度张量，表示反向传播中的梯度信息
    const Tensor& self,             // 自身张量，表示操作的主体张量
    int dim,                        // 维度，表示操作应用的维度
    const Tensor& index,            // 索引张量，表示要操作的索引
    const Tensor& src,              // 源张量，表示进行操作的源数据
    c10::string_view reduce,        // 缩减方式，表示在操作过程中应用的缩减方式
    bool include_self,              // 是否包含自身，表示在计算过程中是否包括自身
    const Tensor& result);          // 结果张量，表示操作后的结果
// 定义函数 _to_copy_backward，接受梯度 Tensor 和 self_options 选项，返回 Tensor
Tensor _to_copy_backward(
    const Tensor& grad,
    const c10::TensorOptions& self_options);

// 定义函数 index_reduce_backward，接受梯度 Tensor、自身 Tensor、维度 dim、索引 index、源张量 source、reduce 操作字符串、是否包含自身标志 include_self 和结果 Tensor，返回梯度和结果 Tensor 的元组
std::tuple<Tensor, Tensor> index_reduce_backward(
    const Tensor& grad,
    const Tensor& self,
    int dim,
    const Tensor& index,
    const Tensor& source,
    c10::string_view reduce,
    bool include_self,
    const Tensor& result);

// 定义函数 take_backward，接受梯度 Tensor、自身 Tensor 和索引 Tensor，返回梯度 Tensor
Tensor take_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& indices);

// 定义函数 to_sparse_backward，接受梯度 Tensor、自身布局 Layout、可选的 SymInt 数组 self_blocksize，返回 Tensor
Tensor to_sparse_backward(
    const Tensor& grad,
    const c10::Layout self_layout,
    const c10::OptionalArrayRef<c10::SymInt>& self_blocksize);

// 定义函数 mkldnn_rnn_layer_differentiable_backward，接受多个输入 Tensor 和参数，返回多个梯度 Tensor 的元组
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
mkldnn_rnn_layer_differentiable_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace);

// 定义函数 values_backward，接受梯度 Tensor 和自身 Tensor，返回梯度 Tensor
Tensor values_backward(const Tensor& grad, const Tensor& self);

// 结束命名空间 torch::autograd::generated::details
} // namespace torch::autograd::generated::details
```