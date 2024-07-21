# `.\pytorch\aten\src\ATen\native\cuda\LinearAlgebraStubs.cpp`

```py
// LinearAlgebraStubs.cpp

// 如果定义了 BUILD_LAZY_CUDA_LINALG，则此文件为几乎无操作
// 在这种情况下，当首次进行线性代数调用时，动态加载库
// 这有助于减少 GPU 内存上下文的大小，如果未使用线性代数函数的话
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/TransposeType.h>

// 如果定义了 BUILD_LAZY_CUDA_LINALG
#if defined(BUILD_LAZY_CUDA_LINALG)
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>

// 如果 MAGMA 已启用
#if AT_MAGMA_ENABLED()
#include <ATen/cuda/detail/CUDAHooks.h>

namespace {
// 设置 Magma 初始化函数为空函数
struct MagmaInitializer {
  MagmaInitializer() {
    ::at::cuda::detail::set_magma_init_fn([]{ });
  };
} initializer;
}  // namespace (anonymous)
#endif
#endif

// 命名空间 at::native 下的定义
namespace at::native {
// 如果定义了 BUILD_LAZY_CUDA_LINALG
#if defined(BUILD_LAZY_CUDA_LINALG)
namespace {
// CUDA 线性代数调度分发器，初始值为 _cholesky_solve_helper_cuda 函数
cuda::detail::LinalgDispatch disp = {_cholesky_solve_helper_cuda};

// 获取 Torch 线性代数库的动态链接库对象
at::DynamicLibrary& getTorchLinalgLibrary() {
  static at::DynamicLibrary lib("libtorch_cuda_linalg.so", nullptr, true);
  return lib;
}

// 惰性调度仅加载 linalg 库并调用存根
// 加载库应该覆盖具有适当实现的注册内容
// 如果找不到库，getTorchLinalgLibrary() 会抛出异常，因此无需显式错误检查
// 但确保此函数仅被调用一次，以避免无限递归
void loadLazyTorchLinalgLibrary() {
  static int invoke_count = 0;
  getTorchLinalgLibrary();
  TORCH_CHECK(invoke_count++ == 0, "lazy wrapper should be called at most once");
}

// 惰性调用 Cholesky 分解内核
void lazy_cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
  loadLazyTorchLinalgLibrary();
  cholesky_stub(DeviceType::CUDA, input, info, upper);
}

// 惰性调用 Cholesky 逆内核
Tensor& lazy_cholesky_inverse_kernel(Tensor &result, Tensor& infos, bool upper) {
  loadLazyTorchLinalgLibrary();
  return cholesky_inverse_stub(DeviceType::CUDA, result, infos, upper);
}

// 惰性调用 LU 分解内核
void lazy_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  loadLazyTorchLinalgLibrary();
  lu_factor_stub(DeviceType::CUDA, input, pivots, infos, compute_pivots);
}

// 惰性调用三角解内核
void lazy_triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  loadLazyTorchLinalgLibrary();
  triangular_solve_stub(DeviceType::CUDA, A, B, left, upper, transpose, unitriangular);
}

// 惰性调用 ORGQR 内核
Tensor& lazy_orgqr_kernel(Tensor& result, const Tensor& tau) {
  loadLazyTorchLinalgLibrary();
  return orgqr_stub(DeviceType::CUDA, result, tau);
}
// 加载延迟加载的Torch线性代数库
void lazy_ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  loadLazyTorchLinalgLibrary();
  // 调用CUDA环境下的ormqr函数，执行矩阵乘积的QR分解更新操作
  ormqr_stub(DeviceType::CUDA, input, tau, other, left, transpose);
}

// 加载延迟加载的Torch线性代数库
void lazy_geqrf_kernel(const Tensor& input, const Tensor& tau) {
  loadLazyTorchLinalgLibrary();
  // 调用CUDA环境下的geqrf函数，执行矩阵的QR分解操作
  geqrf_stub(DeviceType::CUDA, input, tau);
}

// 加载延迟加载的Torch线性代数库
void lazy_linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  loadLazyTorchLinalgLibrary();
  // 调用CUDA环境下的linalg_eigh函数，计算对称矩阵的特征值和特征向量
  linalg_eigh_stub(DeviceType::CUDA, eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
}

// 加载延迟加载的Torch线性代数库
void lazy_linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  getTorchLinalgLibrary();
  // 调用CUDA环境下的linalg_eig函数，计算一般方阵的特征值和特征向量
  linalg_eig_stub(DeviceType::CUDA, eigenvalues, eigenvectors, infos, input, compute_eigenvectors);
}

// 加载延迟加载的Torch线性代数库
void lazy_svd_kernel(const Tensor& A,
                     const bool full_matrices,
                     const bool compute_uv,
                     const std::optional<c10::string_view>& driver,
                     const Tensor& U,
                     const Tensor& S,
                     const Tensor& Vh,
                     const Tensor& info) {
  getTorchLinalgLibrary();
  // 调用CUDA环境下的svd函数，执行奇异值分解
  svd_stub(DeviceType::CUDA, A, full_matrices, compute_uv, driver, U, S, Vh, info);
}

// 加载延迟加载的Torch线性代数库
void lazy_lu_solve(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  getTorchLinalgLibrary();
  // 调用CUDA环境下的lu_solve函数，求解线性方程组 LUx = B
  lu_solve_stub(DeviceType::CUDA, LU, pivots, B, trans);
}

// 加载延迟加载的Torch线性代数库
void lazy_lstsq_kernel(const Tensor& a, Tensor& b, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, std::string driver_name)  {
  getTorchLinalgLibrary();
  // 调用CUDA环境下的lstsq函数，执行最小二乘法拟合
  lstsq_stub(DeviceType::CUDA, a, b, rank, singular_values, infos, rcond, driver_name);
}

// 加载延迟加载的Torch线性代数库
void lazy_ldl_factor(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  loadLazyTorchLinalgLibrary();
  // 调用CUDA环境下的ldl_factor函数，执行LDL分解
  ldl_factor_stub(DeviceType::CUDA, LD, pivots, info, upper, hermitian);
}

// 加载延迟加载的Torch线性代数库
void lazy_ldl_solve(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  loadLazyTorchLinalgLibrary();
  // 调用CUDA环境下的ldl_solve函数，求解线性方程组 L*D*L^T*x = B
  ldl_solve_stub(DeviceType::CUDA, LD, pivots, B, upper, hermitian);
}

// 注册CUDA分发函数，绑定对应的延迟加载函数
REGISTER_CUDA_DISPATCH(cholesky_stub, &lazy_cholesky_kernel)
REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &lazy_cholesky_inverse_kernel);
REGISTER_CUDA_DISPATCH(lu_factor_stub, &lazy_lu_factor);
REGISTER_CUDA_DISPATCH(ldl_factor_stub, &lazy_ldl_factor);
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &lazy_ldl_solve);
REGISTER_CUDA_DISPATCH(triangular_solve_stub, &lazy_triangular_solve_kernel);
REGISTER_CUDA_DISPATCH(orgqr_stub, &lazy_orgqr_kernel);
REGISTER_CUDA_DISPATCH(ormqr_stub, &lazy_ormqr_kernel);
REGISTER_CUDA_DISPATCH(geqrf_stub, &lazy_geqrf_kernel);
REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &lazy_linalg_eigh_kernel);
REGISTER_CUDA_DISPATCH(linalg_eig_stub, &lazy_linalg_eig_kernel);
REGISTER_CUDA_DISPATCH(svd_stub, &lazy_svd_kernel)
REGISTER_CUDA_DISPATCH(lu_solve_stub, &lazy_lu_solve);
REGISTER_CUDA_DISPATCH(lstsq_stub, &lazy_lstsq_kernel);
```  
// 注册 CUDA 分发函数 `svd_stub`、`lu_solve_stub` 和 `lstsq_stub` 到对应的懒加载 CUDA 内核函数。

```py  
} // anonymous namespace
```  
// 结束匿名命名空间，代码块内的符号不会污染全局命名空间。

```py  
// Old style dispatches
// torch_cuda_linalg dynamic library should have a global constructor
// that calls regiserLinaglDispatch so in order ot lazy bind
// old style dispatch all one have to do is to load library and call disp.func_name
// Protect from infinite recursion by initializing dispatch to self and checking
// that values are different after linalg library were loaded
```  
// 注释：旧式分发机制的说明，torch_cuda_linalg 动态库应该有一个全局构造函数来调用 `regiserLinaglDispatch`，因此为了进行懒绑定（lazy bind）旧式分发，只需加载库并调用 `disp.func_name`。通过将分发初始化为自身并在加载 linalg 库后检查值是否不同，防止无限递归。

```py  
namespace cuda {
namespace detail {
void registerLinalgDispatch(const LinalgDispatch& disp_) {
  disp = disp_;
}
}} //namespace cuda::detail
```  
// 命名空间 `cuda::detail` 下的函数 `registerLinalgDispatch`，用于注册线性代数分发对象 `disp`。

```py  
Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
    getTorchLinalgLibrary();
    TORCH_CHECK(disp.cholesky_solve_helper != _cholesky_solve_helper_cuda, "Can't find _cholesky_solve_helper_cuda");
    return disp.cholesky_solve_helper(self, A, upper);
}
```  
// 定义函数 `_cholesky_solve_helper_cuda`，用于执行 CUDA 下的 Cholesky 解算法辅助函数，确保已加载 Torch 的线性代数库，并检查 `disp` 中的 `cholesky_solve_helper` 是否为 `_cholesky_solve_helper_cuda`，若不是则抛出错误。

```py  
#endif /*defined(BUILD_LAZY_CUDA_LINALG)*/
```  
// 结束条件编译指令 `#endif`，用于编译懒加载 CUDA 线性代数。

```py  
} // namespace at::native
```  
// 结束命名空间 `at::native`。
```