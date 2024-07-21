# `.\pytorch\torch\csrc\api\include\torch\linalg.h`

```
#pragma once

#include <ATen/ATen.h>  // 包含 ATen 库，提供张量操作支持

namespace torch {
namespace linalg {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor cholesky(const Tensor& self) {
  return torch::linalg_cholesky(self);  // 调用 torch::linalg_cholesky 函数，计算给定张量的 Cholesky 分解并返回结果张量
}

inline Tensor cholesky_out(Tensor& result, const Tensor& self) {
  return torch::linalg_cholesky_out(result, self);  // 调用 torch::linalg_cholesky_out 函数，计算给定张量的 Cholesky 分解并将结果存储在 result 张量中，然后返回 result
}

inline Tensor det(const Tensor& self) {
  return torch::linalg_det(self);  // 调用 torch::linalg_det 函数，计算给定张量的行列式并返回结果张量
}

inline std::tuple<Tensor, Tensor> slogdet(const Tensor& input) {
  return torch::linalg_slogdet(input);  // 调用 torch::linalg_slogdet 函数，计算给定张量的行列式的符号和绝对值的对数，并返回作为元组的两个张量
}

inline std::tuple<Tensor&, Tensor&> slogdet_out(
    Tensor& sign,
    Tensor& logabsdet,
    const Tensor& input) {
  return torch::linalg_slogdet_out(sign, logabsdet, input);  // 调用 torch::linalg_slogdet_out 函数，计算给定张量的行列式的符号和绝对值的对数，并将结果分别存储在 sign 和 logabsdet 张量中，然后返回这两个张量的引用作为元组
}

inline std::tuple<Tensor, Tensor> eig(const Tensor& self) {
  return torch::linalg_eig(self);  // 调用 torch::linalg_eig 函数，计算给定张量的特征值和特征向量，并返回作为元组的两个张量
}

inline std::tuple<Tensor&, Tensor&> eig_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self) {
  return torch::linalg_eig_out(eigvals, eigvecs, self);  // 调用 torch::linalg_eig_out 函数，计算给定张量的特征值和特征向量，并将结果分别存储在 eigvals 和 eigvecs 张量中，然后返回这两个张量的引用作为元组
}

inline Tensor eigvals(const Tensor& self) {
  return torch::linalg_eigvals(self);  // 调用 torch::linalg_eigvals 函数，计算给定张量的特征值并返回结果张量
}

inline Tensor& eigvals_out(Tensor& result, const Tensor& self) {
  return torch::linalg_eigvals_out(result, self);  // 调用 torch::linalg_eigvals_out 函数，计算给定张量的特征值并将结果存储在 result 张量中，然后返回 result 的引用
}

inline std::tuple<Tensor, Tensor> eigh(
    const Tensor& self,
    c10::string_view uplo) {
  return torch::linalg_eigh(self, uplo);  // 调用 torch::linalg_eigh 函数，计算给定张量的厄米特矩阵的特征值和特征向量，并返回作为元组的两个张量
}

inline std::tuple<Tensor&, Tensor&> eigh_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self,
    c10::string_view uplo) {
  return torch::linalg_eigh_out(eigvals, eigvecs, self, uplo);  // 调用 torch::linalg_eigh_out 函数，计算给定张量的厄米特矩阵的特征值和特征向量，并将结果分别存储在 eigvals 和 eigvecs 张量中，然后返回这两个张量的引用作为元组
}

inline Tensor eigvalsh(const Tensor& self, c10::string_view uplo) {
  return torch::linalg_eigvalsh(self, uplo);  // 调用 torch::linalg_eigvalsh 函数，计算给定张量的厄米特或实对称矩阵的特征值，并返回结果张量
}

inline Tensor& eigvalsh_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view uplo) {
  return torch::linalg_eigvalsh_out(result, self, uplo);  // 调用 torch::linalg_eigvalsh_out 函数，计算给定张量的厄米特或实对称矩阵的特征值，并将结果存储在 result 张量中，然后返回 result 的引用
}

inline Tensor householder_product(const Tensor& input, const Tensor& tau) {
  return torch::linalg_householder_product(input, tau);  // 调用 torch::linalg_householder_product 函数，计算给定张量的 Householder 乘积，并返回结果张量
}

inline Tensor& householder_product_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tau) {
  return torch::linalg_householder_product_out(result, input, tau);  // 调用 torch::linalg_householder_product_out 函数，计算给定张量的 Householder 乘积，并将结果存储在 result 张量中，然后返回 result 的引用
}

inline std::tuple<Tensor, Tensor> lu_factor(
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu_factor(self, pivot);  // 调用 torch::linalg_lu_factor 函数，计算给定张量的 LU 分解因子，并返回作为元组的两个张量
}

inline std::tuple<Tensor&, Tensor&> lu_factor_out(
    Tensor& LU,
    Tensor& pivots,
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu_factor_out(LU, pivots, self, pivot);  // 调用 torch::linalg_lu_factor_out 函数，计算给定张量的 LU 分解因子，并将结果分别存储在 LU 和 pivots 张量中，然后返回这两个张量的引用作为元组
}

inline std::tuple<Tensor, Tensor, Tensor> lu(
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu(self, pivot);  // 调用 torch::linalg_lu 函数，计算给定张量的 LU 分解，并返回作为元组的三个张量
}

inline std::tuple<Tensor&, Tensor&, Tensor&> lu_out(
    Tensor& P,
    Tensor& L,
    Tensor& U,
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu_out(P, L, U, self, pivot);  // 调用 torch::linalg_lu_out 函数，计算给定张量的 LU 分解，并将结果分别存储在 P、L 和 U 张量中，然后返回这三个张量的引用作为元组
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(
    const Tensor& self,
    const Tensor& b,
    std::optional<double> cond,
    // 使用 std::optional 包装的 c10::string_view 类型的 driver 参数
    std::optional<c10::string_view> driver) {
      // 调用 torch::linalg_lstsq 函数，传入 self (第一个参数), b (第二个参数), cond (第三个参数), driver (可选参数)
      return torch::linalg_lstsq(self, b, cond, driver);
    }
// 计算矩阵指数函数 expm(self)，返回计算结果的张量
inline Tensor matrix_exp(const Tensor& self) {
  return torch::linalg_matrix_exp(self);
}

// 计算张量的范数 norm(self, opt_ord, opt_dim, keepdim, opt_dtype)，返回计算结果的张量
inline Tensor norm(
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

// 计算张量的范数 norm(self, ord, opt_dim, keepdim, opt_dtype)，返回计算结果的张量
inline Tensor norm(
    const Tensor& self,
    c10::string_view ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

// 计算张量的范数并将结果保存到指定张量 result 中，返回 result 引用
inline Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(
      result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

// 计算张量的范数并将结果保存到指定张量 result 中，返回 result 引用
inline Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

// 计算向量的范数 vector_norm(self, ord, opt_dim, keepdim, opt_dtype)，返回计算结果的张量
inline Tensor vector_norm(
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_vector_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

// 计算向量的范数并将结果保存到指定张量 result 中，返回 result 引用
inline Tensor& vector_norm_out(
    Tensor& result,
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_vector_norm_out(
      result, self, ord, opt_dim, keepdim, opt_dtype);
}

// 计算矩阵的范数 matrix_norm(self, ord, dim, keepdim, dtype)，返回计算结果的张量
inline Tensor matrix_norm(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return torch::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}

// 计算矩阵的范数并将结果保存到指定张量 result 中，返回 result 引用
inline Tensor& matrix_norm_out(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return torch::linalg_matrix_norm_out(result, self, ord, dim, keepdim, dtype);
}

// 计算矩阵的范数 matrix_norm(self, ord, dim, keepdim, dtype)，返回计算结果的张量
inline Tensor matrix_norm(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return torch::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}

// 计算矩阵的范数并将结果保存到指定张量 result 中，返回 result 引用
inline Tensor& matrix_norm_out(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return torch::linalg_matrix_norm_out(result, self, ord, dim, keepdim, dtype);
}

// 计算矩阵的整数次幂 matrix_power(self, n)，返回计算结果的张量
inline Tensor matrix_power(const Tensor& self, int64_t n) {
  return torch::linalg_matrix_power(self, n);
}

// 计算矩阵的整数次幂并将结果保存到指定张量 result 中，返回 result 引用
inline Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return torch::linalg_matrix_power_out(result, self, n);
}

// 计算矩阵的秩 matrix_rank(input, tol, hermitian)，返回计算结果的张量
inline Tensor matrix_rank(const Tensor& input, double tol, bool hermitian) {
  return torch::linalg_matrix_rank(input, tol, hermitian);
}
// 计算输入张量的秩
inline Tensor matrix_rank(
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return torch::linalg_matrix_rank(input, tol, hermitian);
}

// 计算输入张量的秩，支持指定公差
inline Tensor matrix_rank(
    const Tensor& input,
    std::optional<double> atol,
    std::optional<double> rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank(input, atol, rtol, hermitian);
}

// 计算输入张量的秩，支持张量类型的公差
inline Tensor matrix_rank(
    const Tensor& input,
    const std::optional<Tensor>& atol,
    const std::optional<Tensor>& rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank(input, atol, rtol, hermitian);
}

// 计算输入张量的秩并输出到指定张量中
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    double tol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, tol, hermitian);
}

// 计算输入张量的秩并输出到指定张量中，支持指定公差
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, tol, hermitian);
}

// 计算输入张量的秩并输出到指定张量中，支持指定可选的公差
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    std::optional<double> atol,
    std::optional<double> rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, atol, rtol, hermitian);
}

// 计算输入张量的秩并输出到指定张量中，支持张量类型的公差
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const std::optional<Tensor>& atol,
    const std::optional<Tensor>& rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, atol, rtol, hermitian);
}

// 计算张量列表的多个张量的乘积
inline Tensor multi_dot(TensorList tensors) {
  return torch::linalg_multi_dot(tensors);
}

// 计算张量列表的多个张量的乘积并输出到指定张量中
inline Tensor& multi_dot_out(TensorList tensors, Tensor& result) {
  return torch::linalg_multi_dot_out(result, tensors);
}

// 计算输入张量的广义逆
inline Tensor pinv(const Tensor& input, double rcond, bool hermitian) {
  return torch::linalg_pinv(input, rcond, hermitian);
}

// 计算输入张量的广义逆并输出到指定张量中
inline Tensor& pinv_out(
    Tensor& result,
    const Tensor& input,
    double rcond,
    bool hermitian) {
  return torch::linalg_pinv_out(result, input, rcond, hermitian);
}

// 计算输入张量的 QR 分解
inline std::tuple<Tensor, Tensor> qr(
    const Tensor& input,
    c10::string_view mode) {
  return torch::linalg_qr(input, mode);
}

// 计算输入张量的 QR 分解并输出到指定张量中
inline std::tuple<Tensor&, Tensor&> qr_out(
    Tensor& Q,
    Tensor& R,
    const Tensor& input,
    c10::string_view mode) {
  return torch::linalg_qr_out(Q, R, input, mode);
}

// 解线性方程组
inline std::tuple<Tensor, Tensor> solve_ex(
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return torch::linalg_solve_ex(input, other, left, check_errors);
}

// 解线性方程组并输出到指定张量中
inline std::tuple<Tensor&, Tensor&> solve_ex_out(
    Tensor& result,
    Tensor& info,
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return torch::linalg_solve_ex_out(
      result, info, input, other, left, check_errors);
}

// 解线性方程组
inline Tensor solve(const Tensor& input, const Tensor& other, bool left) {
  return torch::linalg_solve(input, other, left);
}

// 解线性方程组并输出到指定张量中
inline Tensor& solve_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool left) {
  return torch::linalg_solve_out(result, input, other, left);
}
    // 调用 torch 库中的 linalg_solve_out 函数，用于求解线性代数方程 result * other = input 或者 other * result = input 的结果
    // 将解存储在 result 中，input 为右侧矩阵，other 为左侧矩阵或其转置（由 left 参数决定）
    return torch::linalg_solve_out(result, input, other, left);
}

// 解决三角矩阵方程
inline Tensor solve_triangular(
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return torch::linalg_solve_triangular(
      input, other, upper, left, unitriangular);
}

// 使用输出参数解决三角矩阵方程
inline Tensor& solve_triangular_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return torch::linalg_solve_triangular_out(
      result, input, other, upper, left, unitriangular);
}

// 执行奇异值分解（SVD）
inline std::tuple<Tensor, Tensor, Tensor> svd(
    const Tensor& input,
    bool full_matrices,
    std::optional<c10::string_view> driver) {
  return torch::linalg_svd(input, full_matrices, driver);
}

// 使用输出参数执行奇异值分解（SVD）
inline std::tuple<Tensor&, Tensor&, Tensor&> svd_out(
    Tensor& U,
    Tensor& S,
    Tensor& Vh,
    const Tensor& input,
    bool full_matrices,
    std::optional<c10::string_view> driver) {
  return torch::linalg_svd_out(U, S, Vh, input, full_matrices, driver);
}

// 计算奇异值
inline Tensor svdvals(
    const Tensor& input,
    std::optional<c10::string_view> driver) {
  return torch::linalg_svdvals(input, driver);
}

// 使用输出参数计算奇异值
inline Tensor& svdvals_out(
    Tensor& result,
    const Tensor& input,
    std::optional<c10::string_view> driver) {
  return torch::linalg_svdvals_out(result, input, driver);
}

// 计算张量的逆
inline Tensor tensorinv(const Tensor& self, int64_t ind) {
  return torch::linalg_tensorinv(self, ind);
}

// 使用输出参数计算张量的逆
inline Tensor& tensorinv_out(Tensor& result, const Tensor& self, int64_t ind) {
  return torch::linalg_tensorinv_out(result, self, ind);
}

// 解决张量方程
inline Tensor tensorsolve(
    const Tensor& self,
    const Tensor& other,
    OptionalIntArrayRef dims) {
  return torch::linalg_tensorsolve(self, other, dims);
}

// 使用输出参数解决张量方程
inline Tensor& tensorsolve_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    OptionalIntArrayRef dims) {
  return torch::linalg_tensorsolve_out(result, self, other, dims);
}

// 计算矩阵的逆
inline Tensor inv(const Tensor& input) {
  return torch::linalg_inv(input);
}

// 使用输出参数计算矩阵的逆
inline Tensor& inv_out(Tensor& result, const Tensor& input) {
  return torch::linalg_inv_out(result, input);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// Cholesky 分解
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.cholesky
///
/// 示例:
/// ```
/// auto A = torch::randn({4, 4});
/// auto A = torch::matmul(A, A.t());
/// auto L = torch::linalg::cholesky(A);
/// assert(torch::allclose(torch::matmul(L, L.t()), A));
/// ```
inline Tensor cholesky(const Tensor& self) {
  return detail::cholesky(self);
}

// 使用输出参数进行 Cholesky 分解
inline Tensor cholesky_out(Tensor& result, const Tensor& self) {
  return detail::cholesky_out(result, self);
}

// C10_DEPRECATED_MESSAGE("linalg_det is deprecated, use det instead.")
// 计算矩阵的行列式，已被废弃，请使用 det 替代
inline Tensor linalg_det(const Tensor& self) {
  return detail::det(self);
}

/// 参见 torch.linalg.det 的文档
// 计算矩阵的行列式
inline Tensor det(const Tensor& self) {
  return detail::det(self);
}
/// Computes the sign and (natural) logarithm of the determinant of a tensor.
///
/// This function returns a tuple containing the sign and natural logarithm of the determinant.
/// It delegates the computation to a detailed implementation.
inline std::tuple<Tensor, Tensor> slogdet(const Tensor& input) {
  return detail::slogdet(input);
}

/// Computes the sign and (natural) logarithm of the determinant of a tensor, with output tensors.
///
/// This function computes the sign and natural logarithm of the determinant of the input tensor
/// and stores them in the provided output tensors `sign` and `logabsdet`.
inline std::tuple<Tensor&, Tensor&> slogdet_out(
    Tensor& sign,
    Tensor& logabsdet,
    const Tensor& input) {
  return detail::slogdet_out(sign, logabsdet, input);
}

/// Computes eigenvalues and eigenvectors of a non-symmetric/non-hermitian matrix.
///
/// This function computes eigenvalues and eigenvectors of the input tensor `self`.
inline std::tuple<Tensor, Tensor> eig(const Tensor& self) {
  return detail::eig(self);
}

/// Computes eigenvalues and eigenvectors of a non-symmetric/non-hermitian matrix, with output tensors.
///
/// This function computes eigenvalues and eigenvectors of the input tensor `self`
/// and stores them in the provided output tensors `eigvals` and `eigvecs`.
inline std::tuple<Tensor&, Tensor&> eig_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self) {
  return detail::eig_out(eigvals, eigvecs, self);
}

/// Computes eigenvalues of a non-symmetric/non-hermitian matrix.
///
/// This function computes eigenvalues of the input tensor `self`.
inline Tensor eigvals(const Tensor& self) {
  return detail::eigvals(self);
}

/// Computes eigenvalues of a non-symmetric/non-hermitian matrix, with output tensor.
///
/// This function computes eigenvalues of the input tensor `self` and stores them in `result`.
inline Tensor& eigvals_out(Tensor& result, const Tensor& self) {
  return detail::eigvals_out(result, self);
}

/// Computes eigenvalues and eigenvectors of a matrix.
///
/// This function computes eigenvalues and eigenvectors of the input tensor `self`
/// using the specified triangle ('U' for upper or 'L' for lower).
inline std::tuple<Tensor, Tensor> eigh(
    const Tensor& self,
    c10::string_view uplo) {
  return detail::eigh(self, uplo);
}

/// Computes eigenvalues and eigenvectors of a matrix, with output tensors.
///
/// This function computes eigenvalues and eigenvectors of the input tensor `self`
/// and stores them in the provided output tensors `eigvals` and `eigvecs`,
/// using the specified triangle ('U' for upper or 'L' for lower).
inline std::tuple<Tensor&, Tensor&> eigh_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self,
    c10::string_view uplo) {
  return detail::eigh_out(eigvals, eigvecs, self, uplo);
}

/// Computes eigenvalues of a matrix.
///
/// This function computes eigenvalues of the input tensor `self`
/// using the specified triangle ('U' for upper or 'L' for lower).
inline Tensor eigvalsh(const Tensor& self, c10::string_view uplo) {
  return detail::eigvalsh(self, uplo);
}

/// Computes eigenvalues of a matrix, with output tensor.
///
/// This function computes eigenvalues of the input tensor `self`
/// and stores them in `result`, using the specified triangle ('U' for upper or 'L' for lower).
inline Tensor& eigvalsh_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view uplo) {
  return detail::eigvalsh_out(result, self, uplo);
}

/// Computes the product of Householder matrices.
///
/// This function computes the product of Householder matrices from the input tensor `input`
/// and the vector of tau values `tau`.
inline Tensor householder_product(const Tensor& input, const Tensor& tau) {
  return detail::householder_product(input, tau);
}

/// Computes the product of Householder matrices, with output tensor.
///
/// This function computes the product of Householder matrices from the input tensor `input`
/// and the vector of tau values `tau`, storing the result in `result`.
inline Tensor& householder_product_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tau) {
  return detail::householder_product_out(result, input, tau);
}

/// Solves the least squares problem using SVD.
///
/// This function computes the least squares solution to `self @ x = b`
/// with optional condition number `cond` and driver method `driver`.
/// Returns a tuple containing solution `x`, residual `residual`, rank `rank`, and singular values `s`.
inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(
    const Tensor& self,
    const Tensor& b,
    std::optional<double> cond,
    std::optional<c10::string_view> driver) {
  return detail::lstsq(self, b, cond, driver);
}

/// Computes the matrix exponential.
///
/// This function computes the matrix exponential of the input tensor `input`.
inline Tensor matrix_exp(const Tensor& input) {
  return detail::matrix_exp(input);
}

/// Deprecated: Computes the norm of a tensor.
///
/// This function is deprecated; use `norm` instead.
/// It computes the norm of the input tensor `self`.
/// The implementation is provided in the `detail` namespace.
// C10_DEPRECATED_MESSAGE("linalg_norm is deprecated, use norm instead.")
inline Tensor linalg_norm(
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {


# 定义一个函数，用于计算张量的范数
# 参数1: opt_ord 是一个可选的标量引用，表示范数的阶数（默认为空）
# 参数2: opt_dim 是一个可选的整数数组引用，表示在哪些维度上计算范数（默认为空）
# 参数3: keepdim 是一个布尔值，指示是否保持结果张量的维度（默认为 false）
# 参数4: opt_dtype 是一个可选的标量类型，表示计算结果的数据类型（默认为空）
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
// C10_DEPRECATED_MESSAGE("linalg_norm is deprecated, use norm instead.")
// 定义了一个内联函数 linalg_norm，用于计算张量的范数，支持不同的计算选项
inline Tensor linalg_norm(
    const Tensor& self,                      // 输入张量
    c10::string_view ord,                    // 范数的类型
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm 函数来实现范数计算
}

// C10_DEPRECATED_MESSAGE("linalg_norm_out is deprecated, use norm_out instead.")
// 定义了一个内联函数 linalg_norm_out，用于在预分配的张量中计算范数
inline Tensor& linalg_norm_out(
    Tensor& result,                          // 结果张量的引用
    const Tensor& self,                      // 输入张量
    const optional<Scalar>& opt_ord,         // 可选的范数值
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm_out 函数来实现带输出的范数计算
}

// C10_DEPRECATED_MESSAGE("linalg_norm_out is deprecated, use norm_out instead.")
// 定义了一个内联函数 linalg_norm_out，用于在预分配的张量中计算范数
inline Tensor& linalg_norm_out(
    Tensor& result,                          // 结果张量的引用
    const Tensor& self,                      // 输入张量
    c10::string_view ord,                    // 范数的类型
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm_out 函数来实现带输出的范数计算
}

/// Computes the LU factorization with partial pivoting
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.lu_factor
// 定义了一个内联函数 lu_factor，用于计算具有部分主元选取的 LU 分解
inline std::tuple<Tensor, Tensor> lu_factor(
    const Tensor& input,                    // 输入张量
    const bool pivot = true) {              // 是否使用主元选取，默认为 true
  return detail::lu_factor(input, pivot);   // 调用详细实现中的 lu_factor 函数来计算 LU 分解
}

// 定义了一个内联函数 lu_factor_out，用于在预分配的张量中计算具有部分主元选取的 LU 分解
inline std::tuple<Tensor&, Tensor&> lu_factor_out(
    Tensor& LU,                             // LU 分解结果的引用
    Tensor& pivots,                         // 主元索引的引用
    const Tensor& self,                     // 输入张量
    const bool pivot = true) {              // 是否使用主元选取，默认为 true
  return detail::lu_factor_out(LU, pivots, self, pivot);  // 调用详细实现中的 lu_factor_out 函数来计算并输出 LU 分解
}

/// Computes the LU factorization with partial pivoting
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.lu
// 定义了一个内联函数 lu，用于计算具有部分主元选取的 LU 分解，并返回 P、L 和 U 三个张量
inline std::tuple<Tensor, Tensor, Tensor> lu(
    const Tensor& input,                    // 输入张量
    const bool pivot = true) {              // 是否使用主元选取，默认为 true
  return detail::lu(input, pivot);          // 调用详细实现中的 lu 函数来计算并返回 P、L 和 U 三个张量的 LU 分解
}

// 定义了一个内联函数 lu_out，用于在预分配的张量中计算具有部分主元选取的 LU 分解，并返回 P、L 和 U 三个张量的引用
inline std::tuple<Tensor&, Tensor&, Tensor&> lu_out(
    Tensor& P,                              // P 矩阵的引用
    Tensor& L,                              // L 矩阵的引用
    Tensor& U,                              // U 矩阵的引用
    const Tensor& self,                     // 输入张量
    const bool pivot = true) {              // 是否使用主元选取，默认为 true
  return detail::lu_out(P, L, U, self, pivot);  // 调用详细实现中的 lu_out 函数来计算并输出 P、L 和 U 三个张量的 LU 分解
}

// 定义了一个内联函数 norm，用于计算张量的范数
inline Tensor norm(
    const Tensor& self,                      // 输入张量
    const optional<Scalar>& opt_ord,         // 可选的范数值
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm 函数来计算范数
}

// 定义了一个内联函数 norm，用于计算张量的范数
inline Tensor norm(
    const Tensor& self,                      // 输入张量
    std::string ord,                         // 范数的类型
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm 函数来计算范数
}

// 定义了一个内联函数 norm_out，用于在预分配的张量中计算张量的范数
inline Tensor& norm_out(
    Tensor& result,                          // 结果张量的引用
    const Tensor& self,                      // 输入张量
    const optional<Scalar>& opt_ord,         // 可选的范数值
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm_out 函数来计算并输出范数
}

// 定义了一个内联函数 norm_out，用于在预分配的张量中计算张量的范数
inline Tensor& norm_out(
    Tensor& result,                          // 结果张量的引用
    const Tensor& self,                      // 输入张量
    std::string ord,                         // 范数的类型
    OptionalIntArrayRef opt_dim,             // 可选的维度参数
    bool keepdim,                            // 是否保持维度
    optional<ScalarType> opt_dtype) {        // 可选的数据类型参数
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);  // 调用详细实现中的 norm_out 函数来计算并输出范数
}
/// 根据给定的参数计算向量的范数，并返回结果
inline Tensor vector_norm(
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::vector_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

/// 在给定的结果张量中计算向量范数，并将结果存储在其中
inline Tensor& vector_norm_out(
    Tensor& result,
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::vector_norm_out(
      result, self, ord, opt_dim, keepdim, opt_dtype);
}

/// 根据给定的参数计算矩阵的范数，并返回结果
inline Tensor matrix_norm(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return detail::matrix_norm(self, ord, dim, keepdim, dtype);
}

/// 在给定的结果张量中计算矩阵范数，并将结果存储在其中
inline Tensor& matrix_norm_out(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return detail::matrix_norm_out(self, ord, dim, keepdim, dtype, result);
}

/// 根据给定的参数计算矩阵的范数（使用字符串表示的 ord），并返回结果
inline Tensor matrix_norm(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return detail::matrix_norm(self, ord, dim, keepdim, dtype);
}

/// 在给定的结果张量中计算矩阵范数（使用字符串表示的 ord），并将结果存储在其中
inline Tensor& matrix_norm_out(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return detail::matrix_norm_out(self, ord, dim, keepdim, dtype, result);
}

/// 根据给定的参数计算矩阵的 n 次幂，并返回结果
inline Tensor matrix_power(const Tensor& self, int64_t n) {
  return detail::matrix_power(self, n);
}

/// 在给定的结果张量中计算矩阵的 n 次幂，并将结果存储在其中
inline Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return detail::matrix_power_out(self, n, result);
}

/// 根据给定的参数计算矩阵的秩，并返回结果
inline Tensor matrix_rank(const Tensor& input, double tol, bool hermitian) {
  return detail::matrix_rank(input, tol, hermitian);
}

/// 根据给定的参数计算矩阵的秩，并返回结果
inline Tensor matrix_rank(
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return detail::matrix_rank(input, tol, hermitian);
}

/// 根据给定的参数计算矩阵的秩，并返回结果
inline Tensor matrix_rank(
    const Tensor& input,
    std::optional<double> atol,
    std::optional<double> rtol,
    bool hermitian) {
  return detail::matrix_rank(input, atol, rtol, hermitian);
}

/// 根据给定的参数计算矩阵的秩，并返回结果
inline Tensor matrix_rank(
    const Tensor& input,
    const std::optional<Tensor>& atol,
    const std::optional<Tensor>& rtol,
    bool hermitian) {
  return detail::matrix_rank(input, atol, rtol, hermitian);
}

/// 在给定的结果张量中计算矩阵的秩，并将结果存储在其中
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    double tol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, tol, hermitian);
}

/// 在给定的结果张量中计算矩阵的秩，并将结果存储在其中
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  // detail::matrix_rank_out 会计算矩阵的秩，并将结果存储在 result 中
  return detail::matrix_rank_out(result, input, tol, hermitian);
}
    # 调用矩阵排名计算函数，并将结果存储到result中
    return detail::matrix_rank_out(result, input, tol, hermitian);
}

// 使用 detail 命名空间中的函数计算矩阵秩，并将结果输出到指定的 result 引用中
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    std::optional<double> atol,
    std::optional<double> rtol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, atol, rtol, hermitian);
}

// 使用 detail 命名空间中的函数计算矩阵秩，并将结果输出到指定的 result 引用中
inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const std::optional<Tensor>& atol,
    const std::optional<Tensor>& rtol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, atol, rtol, hermitian);
}

/// 使用多个张量列表计算多重点乘积
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.multi_dot
inline Tensor multi_dot(TensorList tensors) {
  return detail::multi_dot(tensors);
}

// 使用多个张量列表计算多重点乘积，并将结果输出到指定的 result 引用中
inline Tensor& multi_dot_out(TensorList tensors, Tensor& result) {
  return detail::multi_dot_out(tensors, result);
}

/// 计算广义逆矩阵
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.pinv
inline Tensor pinv(
    const Tensor& input,
    double rcond = 1e-15,
    bool hermitian = false) {
  return detail::pinv(input, rcond, hermitian);
}

// 计算广义逆矩阵，并将结果输出到指定的 result 引用中
inline Tensor& pinv_out(
    Tensor& result,
    const Tensor& input,
    double rcond = 1e-15,
    bool hermitian = false) {
  return detail::pinv_out(result, input, rcond, hermitian);
}

/// 计算 QR 分解
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.qr
inline std::tuple<Tensor, Tensor> qr(
    const Tensor& input,
    c10::string_view mode = "reduced") {
  // C++17: 更改初始化为 "reduced"sv
  //       qr_out 同样需要修改
  return detail::qr(input, mode);
}

// 计算 QR 分解，并将结果输出到指定的 Q 和 R 引用中
inline std::tuple<Tensor&, Tensor&> qr_out(
    Tensor& Q,
    Tensor& R,
    const Tensor& input,
    c10::string_view mode = "reduced") {
  return detail::qr_out(Q, R, input, mode);
}

/// 计算 LDL 分解
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.ldl_factor_ex
inline std::tuple<Tensor, Tensor, Tensor> ldl_factor_ex(
    const Tensor& input,
    bool hermitian,
    bool check_errors) {
  return torch::linalg_ldl_factor_ex(input, hermitian, check_errors);
}

// 计算 LDL 分解，并将结果输出到指定的 LD、pivots 和 info 引用中
inline std::tuple<Tensor&, Tensor&, Tensor&> ldl_factor_ex_out(
    Tensor& LD,
    Tensor& pivots,
    Tensor& info,
    const Tensor& input,
    bool hermitian,
    bool check_errors) {
  return torch::linalg_ldl_factor_ex_out(
      LD, pivots, info, input, hermitian, check_errors);
}

/// 使用 LDL 分解解决线性方程组
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.ldl_solve
inline Tensor ldl_solve(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool hermitian) {
  return torch::linalg_ldl_solve(LD, pivots, B, hermitian);
}

// 使用 LDL 分解解决线性方程组，并将结果输出到指定的 result 引用中
inline Tensor& ldl_solve_out(
    Tensor& result,
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool hermitian) {
  return torch::linalg_ldl_solve_out(result, LD, pivots, B, hermitian);
}

/// 解决线性方程组 AX = B
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.solve_ex
/// 调用 detail 命名空间下的 solve_ex 函数，返回输入和其他两个 Tensor 的元组
inline std::tuple<Tensor, Tensor> solve_ex(
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return detail::solve_ex(input, other, left, check_errors);
}

/// 调用 detail 命名空间下的 solve_ex_out 函数，计算输入和其他两个 Tensor 的解并返回它们的引用
inline std::tuple<Tensor&, Tensor&> solve_ex_out(
    Tensor& result,
    Tensor& info,
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return detail::solve_ex_out(result, info, input, other, left, check_errors);
}

/// 计算一个 Tensor x，使得 matmul(input, x) = other
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.solve
inline Tensor solve(const Tensor& input, const Tensor& other, bool left) {
  return detail::solve(input, other, left);
}

/// 调用 detail 命名空间下的 solve_out 函数，计算并返回输入 Tensor 的解的引用
inline Tensor& solve_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool left) {
  return detail::solve_out(result, input, other, left);
}

/// 计算线性系统 AX = B 的解，其中 input = A，other = B
/// 前提是 A 是方阵的上三角或下三角，并且对角线上没有零元素
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.solve_triangular
inline Tensor solve_triangular(
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return detail::solve_triangular(input, other, upper, left, unitriangular);
}

/// 调用 detail 命名空间下的 solve_triangular_out 函数，计算并返回解的引用
inline Tensor& solve_triangular_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return detail::solve_triangular_out(
      result, input, other, upper, left, unitriangular);
}

/// 计算奇异值和奇异向量
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.svd
inline std::tuple<Tensor, Tensor, Tensor> svd(
    const Tensor& input,
    bool full_matrices,
    std::optional<c10::string_view> driver) {
  return detail::svd(input, full_matrices, driver);
}

/// 调用 detail 命名空间下的 svd_out 函数，计算并返回奇异值分解的结果的引用
inline std::tuple<Tensor&, Tensor&, Tensor&> svd_out(
    Tensor& U,
    Tensor& S,
    Tensor& Vh,
    const Tensor& input,
    bool full_matrices,
    std::optional<c10::string_view> driver) {
  return detail::svd_out(U, S, Vh, input, full_matrices, driver);
}

/// 计算奇异值
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.svdvals
inline Tensor svdvals(
    const Tensor& input,
    std::optional<c10::string_view> driver) {
  return detail::svdvals(input, driver);
}

/// 调用 detail 命名空间下的 svdvals_out 函数，计算并返回奇异值的引用
inline Tensor& svdvals_out(
    Tensor& result,
    const Tensor& input,
    std::optional<c10::string_view> driver) {
  return detail::svdvals_out(result, input, driver);
}

/// 计算一个张量的逆矩阵
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.tensorinv
///
/// 示例：
/// ```
/// auto a = torch::eye(4*6).reshape({4, 6, 8, 3});
/// int64_t ind = 2;
/// auto ainv = torch::linalg::tensorinv(a, ind);
/// ```
/// 对输入张量进行求逆运算，并返回结果张量。
inline Tensor inv(const Tensor& input) {
    return detail::inv(input);
}

/// 在输出张量中计算输入张量的逆，并将结果存储在结果张量中。
inline Tensor& inv_out(Tensor& result, const Tensor& input) {
    return detail::inv_out(result, input);
}

/// 计算一个张量 `x`，使得 `tensordot(input, x, dims=x.dim()) = other` 成立。
///
/// 参见 https://pytorch.org/docs/main/linalg.html#torch.linalg.tensorsolve
///
/// 示例:
/// ```
/// auto a = torch::eye(2*3*4).reshape({2*3, 4, 2, 3, 4});
/// auto b = torch::randn(2*3, 4);
/// auto x = torch::linalg::tensorsolve(a, b);
/// ```
inline Tensor tensorsolve(
    const Tensor& input,
    const Tensor& other,
    OptionalIntArrayRef dims) {
    return detail::tensorsolve(input, other, dims);
}

/// 在输出张量中计算输入张量的特定维度上的求解，并将结果存储在结果张量中。
inline Tensor& tensorsolve_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    OptionalIntArrayRef dims) {
    return detail::tensorsolve_out(result, input, other, dims);
}
```