# `.\pytorch\aten\src\ATen\native\cuda\linalg\BatchLinearAlgebraLib.cpp`

```py
// 定义宏以指示在拆分实现文件时使用注意事项[BatchLinearAlgebraLib split implementation files]
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的头文件，用于上下文管理和 CUDA 相关操作
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>

// 包含 ATen 库的线性代数工具和 CUDA 相关实现
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/cuda/linalg/CUDASolver.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>

// 根据条件包含不同的 ATen 操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// 将 TransposeType 转换为 cublasOperation_t 枚举类型，用于 cuBLAS 操作
static cublasOperation_t to_cublas(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose: return CUBLAS_OP_N;
    case TransposeType::Transpose: return CUBLAS_OP_T;
    case TransposeType::ConjTranspose: return CUBLAS_OP_C;
  }
  // 如果传入了无效的 transpose 类型，触发内部断言错误
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

// 某些 cuBLAS 和 cuSOLVER 批处理例程要求输入为设备数组，包含指向设备单独矩阵的指针
// 'input' 必须是一个连续的张量
template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.const_data_ptr<scalar_t>();
  int64_t input_mat_stride = matrixStride(input);

  // cuBLAS/cuSOLVER 接口要求使用 'int' 类型
  int batch_size = cuda_int_cast(batchCount(input), "batch_size");

  // 如果 batch_size==0，则 start=0 并且 end=0
  // 如果 input_mat_stride==0，则 step=sizeof(scalar_t)
  return at::arange(
      /*start=*/reinterpret_cast<int64_t>(input_data),
      /*end=*/reinterpret_cast<int64_t>(input_data + batch_size * input_mat_stride),
      /*step=*/static_cast<int64_t>(std::max<int64_t>(input_mat_stride, 1) * sizeof(scalar_t)),
      input.options().dtype(at::kLong));
}

namespace {

// 使用 cuSOLVER 对 LDL 分解因子进行应用，特定于类型 scalar_t
template <typename scalar_t>
void apply_ldl_factor_cusolver(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper) {
#if !defined(USE_LINALG_SOLVER)
  // 如果未定义 USE_LINALG_SOLVER，则抛出错误，指出需要编译支持 cuSOLVER 的 PyTorch 版本
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_factor on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#else
  // 获取批次大小，即A张量的batch维度大小
  auto batch_size = batchCount(A);
  // 获取A张量第二维的大小，并转换为CUDA整型
  auto n = cuda_int_cast(A.size(-2), "A.size(-2)");
  // 获取A张量在最后一维的步长，并转换为CUDA整型
  auto lda = cuda_int_cast(A.stride(-1), "A.stride(-1)");
  // 根据参数upper确定上三角或下三角的存储方式
  auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  // 计算A张量的第三维的步长（如果存在）
  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  // 计算pivots张量的第二维的步长（如果存在）
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  // 获取A张量的数据指针
  auto a_data = A.data_ptr<scalar_t>();
  // 获取pivots张量的数据指针
  auto pivots_data = pivots.data_ptr<int>();
  // 获取info张量的数据指针
  auto info_data = info.data_ptr<int>();

  // 获取当前CUDA solver的句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  // 计算sytrf函数需要的工作空间大小
  int lwork = 0;
  at::cuda::solver::sytrf_bufferSize(handle, n, a_data, lda, &lwork);
  // 分配工作空间内存
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto work = allocator.allocate(sizeof(scalar_t) * lwork);

  // 遍历批次中的每个元素
  for (const auto i : c10::irange(batch_size)) {
    // 指向当前批次中A张量的工作指针
    auto* a_working_ptr = &a_data[i * a_stride];
    // 指向当前批次中pivots张量的工作指针
    auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    // 指向当前批次中info张量的工作指针
    auto* info_working_ptr = &info_data[i];
    // 调用sytrf函数进行LDL分解
    at::cuda::solver::sytrf(
        handle,
        uplo,
        n,
        a_working_ptr,
        lda,
        pivots_working_ptr,
        reinterpret_cast<scalar_t*>(work.get()),
        lwork,
        info_working_ptr);
  }
#endif
}



template <typename scalar_t>
void apply_ldl_solve_cusolver(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& B,
    bool upper) {
#if !(defined(CUDART_VERSION) && defined(CUSOLVER_VERSION) && \
    CUSOLVER_VERSION >= 11102)
  // 检查cuSOLVER的版本，若不支持则抛出错误信息
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_solve on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER 11.1.2+ (CUDA 11.3.1+) support.");
#else
  // 断言：确保A的批次数大于0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) > 0);
  // 断言：确保pivots在增加维度后的批次数大于0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(pivots.unsqueeze(-1)) > 0);
  // 获取B的批次大小
  auto batch_size = batchCount(B);
  // 获取A的倒数第二维度大小
  auto n = A.size(-2);
  // 获取B的最后一维度大小
  auto nrhs = B.size(-1);
  // 获取A最后一维的步长
  auto lda = A.stride(-1);
  // 获取B最后一维的步长
  auto ldb = B.stride(-1);
  // 根据upper值设置uplo为CUBLAS_FILL_MODE_UPPER或者CUBLAS_FILL_MODE_LOWER
  auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  // 如果A的维度大于2，则获取A的倒数第三维的步长，否则设为0
  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  // 如果B的维度大于2，则获取B的倒数第三维的步长，否则设为0
  auto b_stride = B.dim() > 2 ? B.stride(-3) : 0;
  // 如果pivots的维度大于1，则获取pivots的倒数第二维的步长，否则设为0
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  // 获取A的数据指针，并转换为scalar_t类型的常量数据指针
  auto a_data = A.const_data_ptr<scalar_t>();
  // 获取B的数据指针，并转换为scalar_t类型的数据指针
  auto b_data = B.data_ptr<scalar_t>();

  // 将pivots转换为int64_t类型的Tensor
  auto pivots_ = pivots.to(kLong);
  // 获取pivots的数据指针，并转换为int64_t类型的常量数据指针
  auto pivots_data = pivots_.const_data_ptr<int64_t>();

  // 同步CUDA设备，确保之前的操作完成
  c10::cuda::device_synchronize();
  // 获取当前CUDA求解器句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 获取scalar_t类型对应的cusolver数据类型
  auto datatype = at::cuda::solver::get_cusolver_datatype<scalar_t>();
  // 初始化设备和主机上的工作空间大小为0
  size_t worksize_device = 0;
  size_t worksize_host = 0;

  // 获取sytrs函数所需的工作空间大小
  TORCH_CUSOLVER_CHECK(cusolverDnXsytrs_bufferSize(
      handle,
      uplo,
      n,
      nrhs,
      datatype,
      a_data,
      lda,
      pivots_data,
      datatype,
      b_data,
      ldb,
      &worksize_device,
      &worksize_host));

  // 分配设备上的工作空间存储
  auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
  auto workdata_device = device_allocator.allocate(worksize_device);
  void* workdata_device_ptr = workdata_device.get();

  // 分配主机上的工作空间存储
  auto& host_allocator = *at::getCPUAllocator();
  auto workdata_host = host_allocator.allocate(worksize_host);
  void* workdata_host_ptr = workdata_host.get();

  // 创建用于存储信息的Tensor，初始化为0
  Tensor info = at::zeros({}, A.options().dtype(at::kInt));
  // 对每个批次进行LDL求解
  for (const auto i : c10::irange(batch_size)) {
    // 获取当前批次下A的工作指针
    const auto* a_working_ptr = &a_data[i * a_stride];
    // 获取当前批次下B的工作指针
    auto* b_working_ptr = &b_data[i * b_stride];
    // 获取当前批次下pivots的工作指针
    const auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    // 调用cusolverDnXsytrs函数进行LDL求解
    TORCH_CUSOLVER_CHECK(cusolverDnXsytrs(
        handle,
        uplo,
        n,
        nrhs,
        datatype,
        a_working_ptr,
        lda,
        pivots_working_ptr,
        datatype,
        b_working_ptr,
        ldb,
        workdata_device_ptr,
        worksize_device,
        workdata_host_ptr,
        worksize_host,
        info.data_ptr<int>()));
  }

  // 断言：info.item()返回的值为0，表示求解成功
  // 这里仅在调试模式下才会检查info的值
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
#endif
}

} // 匿名命名空间结束
    # 使用 TORCH_CHECK 断言来验证条件，确保 hermitian 参数为假
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_factor: complex tensors with hermitian=True flag are not supported with cuSOLVER backend. ",
        "Currently preferred backend is ",
        at::globalContext().linalgPreferredBackend(),
        ", please set 'default' or 'magma' backend with torch.backends.cuda.preferred_linalg_library");
  }
  # 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏来分发不同的浮点和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_looped_cusolver", [&] {
        # 调用 apply_ldl_factor_cusolver 模板函数，传入相应的类型参数和参数列表 LD、pivots、info、upper
        apply_ldl_factor_cusolver<scalar_t>(LD, pivots, info, upper);
      });
// 结束函数 ldl_solve_cusolver，该函数接受 LD 矩阵、pivots 向量和 B 矩阵，利用 LD 矩阵的值进行求解
void ldl_solve_cusolver(
    const Tensor& LD,                  // 输入参数 LD：包含分解后的对角元和下三角矩阵
    const Tensor& pivots,              // 输入参数 pivots：包含 LD 分解中的置换向量
    const Tensor& B,                   // 输入参数 B：要解的线性方程组的右侧矩阵
    bool upper) {                      // 输入参数 upper：指示 LD 是否为上三角形式
  // 根据 LD 的数据类型执行以下操作
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_solve_looped_cusolver", [&] {
        // 调用模板函数 apply_ldl_solve_cusolver，传递 LD、pivots、B 和 upper 参数
        apply_ldl_solve_cusolver<scalar_t>(LD, pivots, B, upper);
      });
}

// 如果定义了 USE_LINALG_SOLVER，则定义以下静态内联函数
inline static Tensor column_major_identity_matrix_like(const Tensor& self) {
  auto size = self.sizes();             // 获取输入张量 self 的尺寸
  auto size_slice = IntArrayRef(size.data(), size.size()-1);  // 提取 self 尺寸的切片
  // 返回与 self 具有相同尺寸的单位对角矩阵，按列主序分布
  return at::ones(size_slice, self.options()).diag_embed().mT();
}

// 使用 cusolver 的 gesvd 函数计算奇异值分解
template<typename scalar_t>
inline static void apply_svd_cusolver_gesvd(const Tensor& A,        // 输入参数 A：要进行奇异值分解的矩阵
                                            const Tensor& U,        // 输入参数 U：存储左奇异向量的矩阵
                                            const Tensor& S,        // 输入参数 S：存储奇异值的向量或对角矩阵
                                            const Tensor& V,        // 输入参数 V：存储右奇异向量的矩阵
                                            const Tensor& infos,    // 输入参数 infos：用于返回有关计算的信息
                                            bool full_matrices,     // 是否计算全尺寸的 U 和 V
                                            bool compute_uv,        // 是否计算 U 和 V
                                            const bool calculate_all_batches,  // 是否计算所有批次
                                            const std::vector<int64_t>& batches  // 批次的索引向量
) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;  // scalar_t 类型的值类型
  auto A_data = A.data_ptr<scalar_t>();     // 获取 A 张量的数据指针
  auto S_data = S.data_ptr<value_t>();      // 获取 S 张量的数据指针
  auto A_stride = matrixStride(A);          // 获取 A 张量的矩阵步长
  auto S_stride = S.size(-1);               // 获取 S 张量的步长

  int m = cuda_int_cast(A.size(-2), "m");   // 将 A 张量的倒数第二维大小转换为整数 m
  int n = cuda_int_cast(A.size(-1), "n");   // 将 A 张量的最后一维大小转换为整数 n
  auto k = std::min(m, n);                  // 计算 m 和 n 的最小值，奇异值的数量
  int lda = std::max<int>(1, m);            // 设置 A 矩阵的主维度 lda
  int ldvh = std::max<int>(1, n);           // 设置 Vh 矩阵的主维度 ldvh

  TORCH_INTERNAL_ASSERT(m >= n, "cusolver gesvd only supports matrix with sizes m >= n");
  // 使用 CUSOLVER 进行奇异值分解，设置工作空间大小参数 lwork
  char job = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  int lwork = -1;
  // 查询所需的工作空间大小
  at::cuda::solver::gesvd_buffersize<scalar_t>(handle, m, n, &lwork);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvd_buffersize failed to get needed buffer size, got lwork = ", lwork);

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配工作空间的数据指针
  const auto dataPtr_work = allocator.allocate(sizeof(scalar_t)*lwork);
  const auto dataPtr_rwork = allocator.allocate(sizeof(value_t)*std::min(m, n));

  // V 是批量 F-连续的矩阵，因此可以使用 .view() 进行操作
  const auto V_view = compute_uv ? V.view({-1, n, V.size(-1)})
                                 : Tensor{};
  // 对于计算 Vh，需要一个额外的 F-共轭转置矩阵来存储 Vh
  const auto Vh_workspace = compute_uv ?  at::empty({n, full_matrices ? n : k},
                                              A.options().memory_format(at::MemoryFormat::Contiguous)).conj()
                                       : Tensor{};
  const auto Vh_ptr = compute_uv ? Vh_workspace.data_ptr<scalar_t>()
                                 : nullptr;

  const auto U_stride = compute_uv ? matrixStride(U) : 0;
  const auto U_ptr = compute_uv ? U.data_ptr<scalar_t>() : nullptr;

  int batchsize = calculate_all_batches ? cuda_int_cast(batchCount(A), "batch size")
                                        : batches.size();

  // 循环处理每个批次的数据
  for(int _i = 0; _i < batchsize; _i++){
    int i = calculate_all_batches ? _i : batches[_i];
    # 使用 ATen 的 CUDA 解决方案执行奇异值分解（SVD）操作
    at::cuda::solver::gesvd<scalar_t>(
      handle, job, job, m, n,                        # 调用 CUDA SVD 求解器的参数：句柄、作业类型、矩阵尺寸
      A_data + i * A_stride,                         # 输入矩阵 A 的数据指针偏移
      lda,                                            # 矩阵 A 的领先维度
      S_data + i * S_stride,                         # 输出奇异值数组 S 的数据指针偏移
      compute_uv ? U_ptr + i * U_stride : nullptr,   # 如果需要计算 U 矩阵，指定其数据指针偏移；否则为 nullptr
      lda,                                            # U 矩阵的领先维度
      compute_uv ? Vh_ptr : nullptr,                 # 如果需要计算 V^H 矩阵，指定其数据指针；否则为 nullptr
      ldvh,                                           # V^H 矩阵的领先维度
      reinterpret_cast<scalar_t*>(dataPtr_work.get()),# 工作区数据指针
      lwork,                                          # 工作区大小
      reinterpret_cast<value_t*>(dataPtr_rwork.get()),# 实数工作区数据指针
      infos.data_ptr<int>() + i                      # 存储每个子问题信息的数据指针偏移
    );

    # 如果需要计算 V 矩阵，将 V^H 转置后复制到 V_view[i] 中
    if (compute_uv) {
      V_view[i].copy_(Vh_workspace);
    }
}

// 结束了前一个函数的定义

// 我们将在 svd_cusolver_gesvd 函数内部复制 A
inline static void svd_cusolver_gesvd(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool full_matrices, bool compute_uv,
  const bool calculate_all_batches = true,
  const std::vector<int64_t>& batches = {}
) {
  // 我们需要传递 A 的副本，因为它将被覆盖
  // gesvd 只能处理 m >= n 的情况，否则我们需要对 A 进行转置
  const auto not_A_H = A.size(-2) >= A.size(-1);
  // 对 V 进行浅拷贝
  Tensor Vcopy = V; // Shallow copy
#ifdef USE_ROCM
  // 类似于 svd_magma() 中的情况，实验表明在 ROCM 上 Vh 张量不能保证是列主序的，因此我们需要创建一个拷贝来处理这个情况
  if (!not_A_H) {
    Vcopy = at::empty_like(V.mT(),
                           V.options()
                           .device(V.device())
                           .memory_format(at::MemoryFormat::Contiguous)).mT();
  }
#endif
  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏来分派不同的浮点数和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvd", [&] {
    // 应用 svd_cusolver_gesvd 函数来执行特定类型的奇异值分解操作
    apply_svd_cusolver_gesvd<scalar_t>(cloneBatchedColumnMajor(not_A_H ? A : A.mH()),
                                       not_A_H ? U : Vcopy,
                                       S,
                                       not_A_H ? Vcopy : U,
                                       infos,
                                       full_matrices, compute_uv, calculate_all_batches, batches);
  });
#ifdef USE_ROCM
  // 如果不是 not_A_H，则将 Vcopy 的内容复制回 V
  if (!not_A_H) {
    V.copy_(Vcopy);
  }
#endif
}

// 调用 cusolver gesvdj 函数来计算奇异值分解
template<typename scalar_t>


这段代码是 C++ 代码，定义了一个名为 `svd_cusolver_gesvd` 的函数和一个模板函数，用于执行奇异值分解操作，并包含了必要的条件检查和数据处理逻辑。
inline static void apply_svd_cusolver_gesvdj(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool full_matrices, bool compute_uv) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  // 获取输入张量 A 的最后两个维度的大小，用于确定 SVD 的维度
  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  int k = std::min(m, n); // 取 m 和 n 中较小的值作为 SVD 的截断值

  // 需要将分配的内存传递给函数，否则会失败
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 根据 compute_uv 的值决定是否分配 U 和 V 的内存
  auto dataPtr_U = !compute_uv ? allocator.allocate(sizeof(scalar_t)* m * k) : c10::DataPtr{};
  auto dataPtr_V = !compute_uv ? allocator.allocate(sizeof(scalar_t)* n * k) : c10::DataPtr{};

  // 获取输入张量 A、输出张量 U、S、V 的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto U_data = compute_uv ? U.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_U.get());
  auto S_data = S.data_ptr<value_t>();
  auto V_data = compute_uv ? V.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_V.get());
  // 获取输入张量 A、输出张量 U、S、V 的步幅
  auto A_stride = matrixStride(A);
  auto U_stride = compute_uv ? matrixStride(U) : 0;
  auto S_stride = S.size(-1);
  auto V_stride = compute_uv ? matrixStride(V) : 0;

  // 获取批处理的数量和各个矩阵的 lda, ldu, ldv
  int batchsize = cuda_int_cast(batchCount(A), "batch size");
  int lda = A.stride(-1);
  int ldu = compute_uv ? U.stride(-1) : m;
  int ldv = compute_uv ? V.stride(-1) : n;

  // 获取当前 CUDA Solver 的句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 设置求解模式 jobz 和经济模式 econ
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  int econ = full_matrices ? 0 : 1;

  // 创建用于 cusolver gesvdj 迭代数值精度的参数对象 gesvdj_params
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

  // 设置 gesvdj_params 的公差和最大迭代次数
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, std::numeric_limits<scalar_t>::epsilon()));
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, cusolver_gesvdj_max_sweeps));

  // 查询所需的工作空间大小
  int lwork = -1;
  at::cuda::solver::gesvdj_buffersize<scalar_t>(
    handle, jobz, econ, m, n, A_data, lda, S_data, U_data, ldu, V_data, ldv, &lwork, gesvdj_params);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvdj_buffersize failed to get needed buffer size, got lwork = ", lwork);

  // 分配所需的工作空间
  auto dataPtr = allocator.allocate(sizeof(scalar_t)*lwork);

  // 对每个批次进行 SVD 计算
  for(int i = 0; i < batchsize; i++){
    at::cuda::solver::gesvdj<scalar_t>(
      handle, jobz, econ, m, n,
      A_data + i * A_stride,
      lda,
      S_data + i * S_stride,
      U_data + i * U_stride,
      ldu,
      V_data + i * V_stride,
      ldv,
      reinterpret_cast<scalar_t*>(dataPtr.get()),
      lwork,
      infos.data_ptr<int>() + i,
      gesvdj_params
    );

    // 以下代码可用于检查或报告 gesvdj 的残差
    // 注意：这会引入设备与主机的同步，可能对性能产生负面影响
    // double residual = 0;
    // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjGetResidual(handle, gesvdj_params, &residual));
    // 输出 gesvdj 残差值，使用科学计数法，保留六位小数
    printf("gesvdj residual = %.6e\n", residual);
  }

  // 销毁 gesvdj 的参数信息结构体
  TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

// 包装器，调用 apply_svd_cusolver_gesvdj 处理数据类型分发
// 注意，gesvdj 返回 V，这是我们需要的结果
// 需要传递 A 的副本，因为 A 在函数调用内部将被重写
inline static void svd_cusolver_gesvdj(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& infos, bool full_matrices, bool compute_uv) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvdj", [&] {
    // 调用 apply_svd_cusolver_gesvdj 函数处理特定数据类型的 SVD 计算
    apply_svd_cusolver_gesvdj<scalar_t>(A, U, S, V, infos, full_matrices, compute_uv);
  });
}

// 调用 cusolver 中的 gesvdj 批处理函数进行 SVD 计算
template<typename scalar_t>
inline static void apply_svd_cusolver_gesvdjBatched(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool compute_uv
) {
  // 确定 scalar_t 的实际类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  // 解析 A 的维度
  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  // 解析 A 的批处理数量
  int batchsize = cuda_int_cast(batchCount(A), "batch size");
  // 解析 A 的步长
  int lda = A.stride(-1);
  // 根据 compute_uv 的值确定 U 的步长
  int ldu = compute_uv ? U.stride(-1) : m;
  // 根据 compute_uv 的值确定 V 的步长
  int ldv = compute_uv ? V.stride(-1) : n;

  // 需要传递分配好的内存给函数，否则会失败
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr_U = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * m * ldu) : c10::DataPtr{};
  auto dataPtr_V = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * n * ldv) : c10::DataPtr{};

  // 解析 A, U, S, V 的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto U_data = compute_uv ? U.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_U.get());
  auto S_data = S.data_ptr<value_t>();
  auto V_data = compute_uv ? V.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_V.get());

  // 断言矩阵维度 m 和 n 不大于 32，因为 gesvdjBatched 要求如此
  TORCH_INTERNAL_ASSERT(m <= 32 && n <= 32, "gesvdjBatched requires both matrix dimensions not greater than 32, but got "
                        "m = ", m, " n = ", n);

  // 创建 gesvdj 参数对象，用于控制 cusolver gesvdj 迭代的数值精度
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

  // 设置数值精度容差为 scalar_t 的 epsilon
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, std::numeric_limits<scalar_t>::epsilon()));
  // 设置最大迭代次数为 cusolver_gesvdj_max_sweeps
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, cusolver_gesvdj_max_sweeps));
  // 设置排序特征值
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(gesvdj_params, 1));

  // 获取当前 CUDA solver handle
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 根据 compute_uv 的值确定 jobz 参数
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  // 调用 cusolver 的 gesvdjBatched 函数进行批处理 SVD 计算
  at::cuda::solver::gesvdjBatched<scalar_t>(
    handle, jobz, m, n, A_data, lda, S_data, U_data, ldu, V_data, ldv,
    infos.data_ptr<int>(), gesvdj_params, batchsize
  );

  // 销毁 gesvdj 参数对象
  TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}
// 定义一个静态内联函数，执行使用 cusolver 库进行批量 SVD 分解的操作
inline static void svd_cusolver_gesvdjBatched(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& infos, bool full_matrices, bool compute_uv) {
  // 获取输入张量 A 的维度信息
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::min(m, n);
  // 根据 full_matrices 参数设置 U_ 和 V_ 张量的大小以支持完整或截断的 SVD 结果存储
  // 核心假设是 full_matrices == true
  // 如果 full_matrices == false 且 m != n，则创建适当大小的辅助张量，并将结果复制回去
  auto U_ = U;
  auto V_ = V;
  if (compute_uv && !full_matrices) {
    // 获取张量 A 的尺寸信息
    auto sizes = A.sizes().vec();
    if (m > n) {
      // 使用 full_matrices == true 时的 U 大小
      sizes.end()[-1] = m;
      // U、V 应为 Fortran 连续数组的批量
      // 创建新的 U_ 张量，使其满足 Fortran 连续性，并进行转置
      U_ = U.new_empty(sizes).mT();
    } else if (m < n) {
      // 使用 full_matrices == true 时的 V 大小
      sizes.end()[-2] = n;
      // 创建新的 V_ 张量，使其满足 Fortran 连续性，并进行转置
      V_ = V.new_empty(sizes).mT();
    }
  }
  // 此时 U_ 和 V_ 是批量的 Fortran 连续方阵

  // 根据 A 的数据类型进行分发调度，调用 cusolver 库执行批量 SVD 分解
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvdjBatched", [&] {
    apply_svd_cusolver_gesvdjBatched<scalar_t>(A, U_, S, V_, infos, compute_uv);
  });

  // 如果创建了任何新的矩阵，则将结果复制回原始张量
  if (compute_uv && !full_matrices) {
    // 如果 U_ 不是 U 的别名，则将 U_ 的内容复制回 U
    if (!U_.is_alias_of(U)) {
      U.copy_(U_.narrow(-1, 0, k));
    }
    // 如果 V_ 不是 V 的别名，则将 V_ 的内容复制回 V
    if (!V_.is_alias_of(V)) {
      V.copy_(V_.narrow(-1, 0, k));
    }
  }
}

// 使用指定数据类型执行 cusolver 库的批量 SVD 操作的具体实现
template<typename scalar_t>
inline static void apply_svd_cusolver_gesvdaStridedBatched(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
    const Tensor& infos, bool full_matrices, bool compute_uv) {
#ifndef CUDART_VERSION
  // 如果没有 CUDA 运行时版本信息，则输出错误信息，表示 gesvda 只支持 cuBLAS 后端的批量操作
  TORCH_CHECK(false, "gesvda: Batched version is supported only with cuBLAS backend.")
#else
  // 定义 value_t 类型，使用 c10::scalar_value_type 获取标量类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  // 获取 A 张量的倒数第二维和倒数第一维的大小，并转换为 CUDA int 类型
  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  // 断言 m >= n，因为 cusolver gesvdaStridedBatched 要求 m >= n
  TORCH_INTERNAL_ASSERT(m >= n, "cusolver gesvdaStridedBatched requires m >= n");
  // 计算 batchsize，即 A 张量的批次数
  int batchsize = cuda_int_cast(batchCount(A), "batch size");

  // 计算 lda，A 张量在最后一维上的步长
  int lda = A.stride(-1);
  // 计算 ldu 和 ldv，如果 compute_uv 为真则使用 U 和 V 的步长，否则使用 m 和 n
  int ldu = compute_uv ? U.stride(-1) : m;
  int ldv = compute_uv ? V.stride(-1) : n;

  // 获取 A 张量在内存中的步长和 S 张量的最后一维大小（奇异值的数量）
  auto A_stride = matrixStride(A);
  auto S_stride = S.size(-1);
  auto rank = S_stride; // 奇异值的数量
  // 计算 U 和 V 的步长，如果 compute_uv 为真则使用 matrixStride(U) 和 matrixStride(V)，否则使用 ldu * rank 和 ldv * rank
  auto U_stride = compute_uv ? matrixStride(U) : ldu * rank;
  auto V_stride = compute_uv ? matrixStride(V) : ldv * rank;

  // 需要向函数传递已分配的内存，否则会失败
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配用于 U 和 V 的数据指针，如果 compute_uv 为假则分配相应大小的空指针
  auto dataPtr_U = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * m * n) : c10::DataPtr{};
  auto dataPtr_V = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * n * n) : c10::DataPtr{};

  // 获取 A、U、S、V 张量的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto U_data = compute_uv ? U.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_U.get());
  auto S_data = S.data_ptr<value_t>();
  auto V_data = compute_uv ? V.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_V.get());

  // 获取当前 CUDA solver 句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 设置 jobz 标志，如果 compute_uv 为真则使用 CUSOLVER_EIG_MODE_VECTOR，否则使用 CUSOLVER_EIG_MODE_NOVECTOR
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  // 计算需要的工作空间大小
  int lwork = -1;
  // 调用 gesvdaStridedBatched_buffersize 函数获取所需的缓冲区大小
  at::cuda::solver::gesvdaStridedBatched_buffersize<scalar_t>(
    handle, jobz, rank, m, n, A_data, lda, A_stride, S_data, S_stride, U_data, ldu, U_stride, V_data, ldv, V_stride,
    &lwork, batchsize);
  // 断言 lwork 大于等于 0，以确保获取缓冲区大小成功
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvdaStridedBatched_buffersize failed to get needed buffer size, got lwork = ", lwork);
  // 分配工作空间
  auto workspace = allocator.allocate(sizeof(scalar_t)*lwork);

  // 以下注释适用于 at::cuda::solver::gesvdaStridedBatched 的调用
  // 弗罗贝尼乌斯范数的残差总是以 double 类型返回。
  // cuSOLVER 注释：如果用户对奇异值和奇异向量的准确性有信心，例如某些条件成立（所需的奇异值远离零），
  //   则通过将 h_RnrmF 的空指针传递给它来提高性能，即不计算残差范数。
  // Comment: 弗罗贝尼乌斯范数的计算昂贵，并且不会影响结果的准确性

  // 调用 gesvdaStridedBatched 函数执行奇异值分解
  at::cuda::solver::gesvdaStridedBatched<scalar_t>(
    handle, jobz, rank, m, n, A_data, lda, A_stride, S_data, S_stride, U_data, ldu, U_stride, V_data, ldv, V_stride,
    reinterpret_cast<scalar_t*>(workspace.get()),
    lwork, infos.data_ptr<int>(),
    nullptr,  // 不计算 cuSOLVER h_RnrmF：reinterpret_cast<double*>(residual_frobenius_norm.get()),
    batchsize);
#endif
}

// 在 svd_cusolver_gesvdaStridedBatched 内部复制 A
inline static void svd_cusolver_gesvdaStridedBatched(
    const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  // 我们需要传递 A 的副本，因为它将被重写
  // gesvdaStridedBatched 只能处理 m >= n 的情况，所以在其他情况下我们需要转置 A
  const auto not_A_H = A.size(-2) >= A.size(-1);
  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏根据 A 的数据类型进行分发，用于处理不同类型的数据
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvdaStridedBatched", [&] {
    // 调用 apply_svd_cusolver_gesvdaStridedBatched 函数执行 SVD 分解
    apply_svd_cusolver_gesvdaStridedBatched<scalar_t>(
      // 如果条件成立，传递 A 的拷贝；否则传递 A 的转置
      cloneBatchedColumnMajor(not_A_H ? A : A.mH()),
      // 如果条件成立，传递 U；否则传递 V
      not_A_H ? U : V,
      // 传递奇异值向量 S
      S,
      // 如果条件成立，传递 V；否则传递 U
      not_A_H ? V : U,
      // 传递 infos 张量，它包含操作的状态信息
      infos, full_matrices, compute_uv);
  });
}

// Check convergence of gesvdj/gesvdjBatched/gesvdaStridedBatched results.
// If not converged, return a vector that contains indices of the non-converging batches.
// If the returned vector is empty, all the matrices are converged.
// This function will cause a device-host sync.
std::vector<int64_t> _check_gesvdj_convergence(const Tensor& infos, int64_t non_converging_info) {
  // 将infos张量移动到CPU上
  at::Tensor infos_cpu = infos.cpu();
  // 获取CPU上的数据指针
  auto infos_cpu_data = infos_cpu.data_ptr<int>();

  // 存储非收敛批次的索引
  std::vector<int64_t> res;

  // 遍历所有的infos元素
  for(int64_t i = 0; i < infos.numel(); i++) {
    // 获取第i批次的信息
    int info_for_batch_i = infos_cpu_data[i];

    // 根据cusolver文档，如果info小于0，表示第i个函数调用的参数错误，这意味着pytorch中cusolver的实现有问题
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info_for_batch_i >= 0);

    // 在我们的用例中，gesvdj、gesvdjBatched和gesvdaStridedBatched对`info`的符号含义相同
    if (info_for_batch_i == non_converging_info) res.push_back(i);

    // 然而，对于gesvd函数，情况并非如此，尽管我们目前不使用此函数来检查gesvd的收敛性
    // 如果将来有一天实现了这一点，需要小心处理
  }

  return res;
}

// Depending on the number of non-converging batches,
// format the non-converging batches string as either (no leading or trailing whitespaces)
// batches 2, 3, 5  // or
// batches 2, 3, 5, 7, 11 and other 65535 batches
std::string _format_non_converging_batches(const std::vector<int64_t>& batches) {
  // 创建一个字符串流
  std::stringstream ss;
  const int too_long = 5;

  // 添加初始字符串
  ss << "batches ";

  // 如果非收敛批次数量不超过5个，则逐个列出
  if (batches.size() <= too_long) {
    for (const auto i : c10::irange(batches.size() - 1)) {
      ss << batches[i] << ", ";
    }
    ss << batches.back();
  } else {
    // 如果超过5个，则只列出前5个，并提示还有多少个批次未列出
    for (const auto i : c10::irange(too_long)) {
      ss << batches[i] << ", ";
    }
    ss << "and other " << batches.size() - too_long << " batches";
  }

  return ss.str();
}

// This function returns V, not V^H.
void svd_cusolver(const Tensor& A,
                  const bool full_matrices,
                  const bool compute_uv,
                  const std::optional<c10::string_view>& driver,
                  const Tensor& U,
                  const Tensor& S,
                  const Tensor& V,
                  const Tensor& info) {
  // Here U and V are F-contig whenever they are defined (i.e. whenever compute_uv=true)
  const auto m = A.size(-2);
  const auto n = A.size(-1);
  const auto k = std::min(m, n);

  // 定义一个常量字符串，指向svd文档链接
  static const char* check_svd_doc = "Check doc at https://pytorch.org/docs/stable/generated/torch.linalg.svd.html";

  // 默认的驱动程序选择为gesvdj
#ifdef USE_ROCM
  const auto driver_v = c10::string_view("gesvdj");
#else
  const auto driver_v = driver.value_or("gesvdj");
#endif

  // 根据驱动程序选择调用不同的svd_cusolver实现
  if (driver_v == "gesvd") {
    svd_cusolver_gesvd(A, U, S, V, info, full_matrices, compute_uv);
  } else if (driver_v == "gesvdj") {
    // 参考以下基准测试
    // 根据给定的条件选择合适的 cuSOLVER SVD 驱动程序进行计算
    // 如果 m 和 n 均小于等于 32，则使用 svd_cusolver_gesvdjBatched 进行批处理 SVD 计算
    // 否则，使用 svd_cusolver_gesvdj 进行单次 SVD 计算，但对于大型矩阵可能存在数值不稳定性
    if (m <= 32 && n <= 32) {
      svd_cusolver_gesvdjBatched(cloneBatchedColumnMajor(A), U, S, V, info, full_matrices, compute_uv);
    } else {
      // 对于大尺寸矩阵，gesvdj 驱动程序可能在数值上不稳定
      svd_cusolver_gesvdj(cloneBatchedColumnMajor(A), U, S, V, info, full_matrices, compute_uv);
    }
  } else if (driver_v == "gesvda") {
    // cuSOLVER: 对于 "tall skinny" (m > n) 的矩阵，推荐使用 gesvdaStridedBatched
    // 这里进行转置以使其也适用于 (m < n) 的矩阵
    svd_cusolver_gesvdaStridedBatched(A, U, S, V, info, full_matrices, compute_uv);
  } else {
    // 如果驱动程序不是 gesvd 或 gesvda，则抛出错误并提供相关文档链接
    TORCH_CHECK(false, "torch.linalg.svd: unknown svd driver ", driver_v, " in svd_cusolver computation. ", check_svd_doc);
  }

  // 需要检查收敛性
  if (driver_v != "gesvd") {
    // 将执行设备与主机同步
    // Todo: 实现 svd_ex 变种以不检查结果收敛性，从而消除设备与主机同步
    // 检查哪些批次的 gesvdj 计算未收敛
    const auto svd_non_converging_batches = _check_gesvdj_convergence(info, k + 1);

    // 如果有未收敛的批次，则发出警告并提供建议
    if (!svd_non_converging_batches.empty()) {
      TORCH_WARN_ONCE("torch.linalg.svd: During SVD computation with the selected cusolver driver, ",
                      _format_non_converging_batches(svd_non_converging_batches),
                      " failed to converge. ",
                      (driver.has_value()
                        ?  "It is recommended to redo this SVD with another driver. "
                        : "A more accurate method will be used to compute the SVD as a fallback. "),
                      check_svd_doc);

      // 如果用户未指定驱动程序，且默认启发式不良，则使用 gesvd 作为回退
      if (!driver.has_value()) {
        svd_cusolver_gesvd(A, U, S, V, info, full_matrices, compute_uv, false, svd_non_converging_batches);
      }
    }
  }

  // `info` 将在后续的 `TORCH_IMPL_FUNC(_linalg_svd_out)` 函数中进行检查。
// Implementation of Cholesky decomposition using looped cusolverDn<T>potrf or cusolverDnXpotrf (64-bit)
template<typename scalar_t>
// 定义一个模板函数，用于在 GPU 上执行 Cholesky 分解，使用循环调用 cusolverDn<T>potrf 或 cusolverDnXpotrf（64 位版本）
inline static void apply_cholesky_cusolver_potrf_looped(const Tensor& self_working_copy, bool upper, const Tensor& infos) {
  // 获取当前 CUDA 解算器句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 确定是上三角还是下三角分解
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 获取矩阵的维度
  const int64_t n = self_working_copy.size(-1);
  // 确保 lda 至少为 1
  const int64_t lda = std::max<int64_t>(1, n);
  // 获取批次大小
  const int64_t batch_size = batchCount(self_working_copy);
  // 获取矩阵步幅
  const int64_t matrix_stride = matrixStride(self_working_copy);

  // 获取输入数据指针和 infos 数据指针
  scalar_t* self_working_copy_ptr = self_working_copy.data_ptr<scalar_t>();
  int* infos_ptr = infos.data_ptr<int>();

#ifdef USE_CUSOLVER_64_BIT
  // 定义需要的工作空间大小
  size_t worksize_device;
  size_t worksize_host;
  cusolverDnParams_t params;
  // 获取数据类型并创建 cusolver 参数对象
  cudaDataType datatype = at::cuda::solver::get_cusolver_datatype<scalar_t>();
  TORCH_CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  // 获取所需的工作空间大小
  at::cuda::solver::xpotrf_buffersize(handle, params, uplo, n, datatype, nullptr, lda, datatype, &worksize_device, &worksize_host);

  // 分配设备上的工作空间存储
  auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
  auto workdata_device = device_allocator.allocate(worksize_device * batch_size);
  void* workdata_device_ptr = workdata_device.get();

  // 分配主机上的工作空间存储
  auto& host_allocator = *at::getCPUAllocator();
  auto workdata_host = host_allocator.allocate(worksize_host * batch_size);
  void* workdata_host_ptr = workdata_host.get();

  // 循环执行 Cholesky 分解
  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::xpotrf(
      handle, params, uplo, n, datatype,
      self_working_copy_ptr + i * matrix_stride,
      lda, datatype,
      (char*)workdata_device_ptr + i * worksize_device, worksize_device,
      (char*)workdata_host_ptr + i * worksize_host, worksize_host,
      infos_ptr + i
    );
  }

  // 销毁 cusolver 参数对象
  TORCH_CUSOLVER_CHECK(cusolverDnDestroyParams(params));
#else // USE_CUSOLVER_64_BIT
  // 如果不使用 64 位 cusolver，则使用 32 位版本的 Cholesky 分解
  int n_32 = cuda_int_cast(n, "n");
  int lda_32 = cuda_int_cast(lda, "lda");
  int lwork;
  // 获取所需的工作空间大小
  at::cuda::solver::potrf_buffersize<scalar_t>(
    handle, uplo, n_32, nullptr, lda_32, &lwork);

  // 分配工作空间存储
  auto& allocator = *at::cuda::getCUDADeviceAllocator();
  auto work_data = allocator.allocate(sizeof(scalar_t) * lwork * batch_size);
  scalar_t* work_data_ptr = static_cast<scalar_t*>(work_data.get());

  // 循环执行 Cholesky 分解
  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::potrf<scalar_t>(
      handle, uplo, n_32,
      self_working_copy_ptr + i * matrix_stride,
      lda_32,
      work_data_ptr + i * lwork,
      lwork,
      infos_ptr + i
    );
  }
#endif // USE_CUSOLVER_64_BIT
}
// 获取当前 CUDA solver 句柄
auto handle = at::cuda::getCurrentCUDASolverDnHandle();
// 根据是否使用上三角矩阵决定填充模式
const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
// 获取最后一个维度的大小作为矩阵的维度 n
const int n = cuda_int_cast(self_working_copy.size(-1), "n");
// 确定 lda 的值为 n 或者 1，至少为 1
const int lda = std::max<int>(1, n);

// 获取 batch 的数量
const int batch_size = cuda_int_cast(batchCount(self_working_copy), "batch_size");

// cusolver 批处理内核要求输入为“设备指针的设备数组”
Tensor self_working_copy_array = get_device_pointers<scalar_t>(self_working_copy);

// 调用 cusolver 的 potrfBatched 函数进行 Cholesky 分解批处理
at::cuda::solver::potrfBatched<scalar_t>(
  handle, uplo, n,
  reinterpret_cast<scalar_t**>(self_working_copy_array.data_ptr()),
  lda, infos.data_ptr<int>(), batch_size);
}

// Cholesky 分解的辅助函数，使用 cusolver
void cholesky_helper_cusolver(const Tensor& input, bool upper, const Tensor& info) {
  // 如果输入张量为空，则直接返回
  if (input.numel() == 0) {
    return;
  }

  // 如果启用了 cusolver 的批处理并且输入批次数大于 1
  if (use_cusolver_potrf_batched_ && batchCount(input) > 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cusolver", [&] {
      // 调用 apply_cholesky_cusolver_potrfBatched 函数进行批处理 Cholesky 分解
      apply_cholesky_cusolver_potrfBatched<scalar_t>(input, upper, info);
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cusolver", [&] {
      // 否则，调用 apply_cholesky_cusolver_potrf_looped 函数进行循环 Cholesky 分解
      apply_cholesky_cusolver_potrf_looped<scalar_t>(input, upper, info);
    });
  }
}

// Cholesky 解的应用函数，使用 cusolver
template<typename scalar_t>
inline static void apply_cholesky_cusolver_potrs(Tensor& self_working_copy, const Tensor& A_column_major_copy, bool upper, Tensor& infos) {
  // 获取当前 CUDA solver 句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 根据是否使用上三角矩阵决定填充模式
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 获取输入张量的倒数第二维和最后一维作为矩阵的维度 n 和 nrhs
  const int64_t n = self_working_copy.size(-2);
  const int64_t nrhs = self_working_copy.size(-1);
  // 确定 lda 的值为 n 或者 1，至少为 1
  const int64_t lda = std::max<int64_t>(1, n);
  // 获取批处理的数量
  const int64_t batch_size = batchCount(self_working_copy);
  // 获取输入张量的步幅和 A 矩阵的步幅
  const int64_t self_matrix_stride = matrixStride(self_working_copy);
  scalar_t* self_working_copy_ptr = self_working_copy.data_ptr<scalar_t>();

  scalar_t* A_ptr = A_column_major_copy.data_ptr<scalar_t>();
  const int64_t A_matrix_stride = matrixStride(A_column_major_copy);
  const int64_t ldb = std::max<int64_t>(1, A_column_major_copy.size(-1));

  int* infos_ptr = infos.data_ptr<int>();

  // 如果使用了 64 位 cusolver
  #ifdef USE_CUSOLVER_64_BIT
  // 创建 cusolver 参数
  cusolverDnParams_t params;
  // 获取数据类型
  cudaDataType datatype = at::cuda::solver::get_cusolver_datatype<scalar_t>();
  // 创建 cusolver 参数对象
  TORCH_CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  // 循环处理每个批次的 Cholesky 解
  for (int64_t i = 0; i < batch_size; i++) {
    // 调用 cusolver 的 xpotrs 函数进行 Cholesky 解
    at::cuda::solver::xpotrs(
      handle, params, uplo, n, nrhs, datatype,
      A_ptr + i * A_matrix_stride,
      lda, datatype,
      self_working_copy_ptr + i * self_matrix_stride,
      ldb,
      infos_ptr
    );
  }

  // 销毁 cusolver 参数对象
  TORCH_CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  ```
#else // USE_CUSOLVER_64_BIT
  // 将输入参数转换为32位整数
  int n_32 = cuda_int_cast(n, "n");
  int nrhs_32 = cuda_int_cast(nrhs, "nrhs");
  int lda_32 = cuda_int_cast(lda, "lda");
  int ldb_32 = cuda_int_cast(ldb, "ldb");

  // 对于每个批次中的每个问题，调用 cusolver 的 potrs 函数求解线性系统
  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::potrs<scalar_t>(
      handle, uplo, n_32, nrhs_32,
      // 指向 A 的第 i 个批次数据起始地址
      A_ptr + i * A_matrix_stride,
      lda_32,
      // 指向待解向量的第 i 个批次数据起始地址
      self_working_copy_ptr + i * self_matrix_stride,
      ldb_32,
      infos_ptr
    );
  }
#endif // USE_CUSOLVER_64_BIT
}


// 此代码路径仅在 pytorch 构建中未链接 MAGMA 时才使用。
// cusolverDn<t>potrsBatched 只支持 nrhs == 1
template<typename scalar_t>
inline static void apply_cholesky_cusolver_potrsBatched(Tensor& self_working_copy, const Tensor& A_column_major_copy, bool upper, Tensor& infos) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // 根据 upper 参数确定填充模式
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const int64_t n = self_working_copy.size(-2);
  const int64_t nrhs = self_working_copy.size(-1);
  const int64_t lda = std::max<int64_t>(1, n);
  const int64_t batch_size = batchCount(self_working_copy);

  // 计算 A 的列优先复制的最后一个维度长度
  const int64_t ldb = std::max<int64_t>(1, A_column_major_copy.size(-1));

  // 指向 infos 张量数据的指针
  int* infos_ptr = infos.data_ptr<int>();

  // 获取 self_working_copy 和 A_column_major_copy 张量数据的设备指针数组
  auto self_ptr_array = get_device_pointers<scalar_t>(self_working_copy);
  auto A_ptr_array = get_device_pointers<scalar_t>(A_column_major_copy);

  // 调用 cusolver 的 potrsBatched 函数解线性系统
  at::cuda::solver::potrsBatched(
    handle, uplo,
    // 将 n, nrhs, lda, ldb 和 batch_size 转换为设备整数
    cuda_int_cast(n, "n"),
    cuda_int_cast(nrhs, "nrhs"),
    // A_ptr_array 和 self_ptr_array 分别指向 A 和 self_working_copy 批处理数据的指针数组
    reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr()),
    cuda_int_cast(lda, "lda"),
    reinterpret_cast<scalar_t**>(self_ptr_array.data_ptr()),
    cuda_int_cast(ldb, "ldb"),
    // infos_ptr 指向的设备上的整数，用于记录操作信息
    infos_ptr,
    cuda_int_cast(batch_size, "batch_size")
  );
}

// 使用 cusolver 解 Cholesky 分解的线性系统的助手函数
Tensor _cholesky_solve_helper_cuda_cusolver(const Tensor& self, const Tensor& A, bool upper) {
  // 获取批处理数目
  const int64_t batch_size = batchCount(self);
  // 创建一个全零的 infos 张量
  at::Tensor infos = at::zeros({1}, self.options().dtype(at::kInt));
  // 复制 self 和 A 到列主优先的副本
  at::Tensor self_working_copy = cloneBatchedColumnMajor(self);
  at::Tensor A_column_major_copy = cloneBatchedColumnMajor(A);

  // 获取 self_working_copy 的最后一个维度长度作为 nrhs
  const int64_t nrhs = self_working_copy.size(-1);

  // 如果批处理数大于1且 nrhs 等于1，则使用 cusolverDn<t>potrsBatched 函数
  if (batch_size > 1 && nrhs == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_cuda_potrs_batched", [&] {
      apply_cholesky_cusolver_potrsBatched<scalar_t>(self_working_copy, A_column_major_copy, upper, infos);
    });
  } else {
    // 否则使用 cusolver 的 potrs 函数
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_cuda_potrs", [&] {
      apply_cholesky_cusolver_potrs<scalar_t>(self_working_copy, A_column_major_copy, upper, infos);
    });
  }

  // infos 仅报告 potrs 和 potrsBatched 函数参数是否错误，不会报告矩阵奇异性等问题
  // 所以我们不需要每次都检查它。
  // 在调试模式下，确保 infos 为零，表示没有错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.item().toInt() == 0);

  // 返回已解决的线性系统的副本
  return self_working_copy;
}
/*
  使用 cuSOLVER 库中的 cusolverDnXgeqrf 函数对矩阵 A 进行 QR 分解。
  在此过程中，计算得到的 R 存储在矩阵 A 的上三角部分，
  并在矩阵 A 的主对角线以下存储元素反射器。

  Args:
  * `A` - [in] 待进行 QR 分解的矩阵 Tensor，
          [out] 包含矩阵 R 在 A 的上三角部分，
          并在主对角线以下包含元素反射器
  * `tau` - 包含元素反射器的幅度的 Tensor
  * `m` - 要考虑的 `input` 的行数
  * `n` - 要考虑的 `input` 的列数（`input` 的实际大小可能更大）

  详细信息请参考 cuSOLVER 的 GEQRF 文档。
*/
template <typename scalar_t>
static void apply_geqrf(const Tensor& A, const Tensor& tau) {
  // 获取矩阵 A 的尺寸
  int64_t m = A.size(-2);
  int64_t n = A.size(-1);
  // 计算 A 的主要对角线元素所需的最大长度
  int64_t lda = std::max<int64_t>(1, m);
  // 获取 batch 的数量
  int64_t batch_size = batchCount(A);

  // 获取矩阵 A 和 tau 的步幅
  auto A_stride = matrixStride(A);
  auto tau_stride = tau.size(-1);

  // 获取矩阵 A 和 tau 的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();

  // 创建用于存储返回状态的 infos Tensor
  auto infos = at::zeros({1}, A.options().dtype(at::kInt));
  auto infos_data = infos.data_ptr<int>();

  // 获取最优工作空间大小并分配工作空间 Tensor
#ifdef USE_CUSOLVER_64_BIT
  size_t worksize_device; // 设备上的工作空间大小（字节）
  size_t worksize_host; // 主机上的工作空间大小（字节）
  cusolverDnParams_t params = NULL; // 使用默认算法（当前是唯一选项）
  at::cuda::solver::xgeqrf_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(),
      params,
      m,
      n,
      A_data,
      lda,
      tau_data,
      &worksize_device,
      &worksize_host);
#else
  int lwork; // 工作空间大小
  int m_32 = cuda_int_cast(m, "m");
  int n_32 = cuda_int_cast(n, "n");
  int lda_32 = cuda_int_cast(lda, "lda");
  at::cuda::solver::geqrf_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(), m_32, n_32, A_data, lda_32, &lwork);
#endif // USE_CUSOLVER_64_BIT

  // 对每个 batch 执行 QR 分解
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    // 获取当前 batch 的 A 和 tau 的工作指针
    scalar_t* A_working_ptr = &A_data[i * A_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    // 获取当前 cuSOLVER 句柄
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();
#ifdef USE_CUSOLVER_64_BIT
    // 如果定义了 USE_CUSOLVER_64_BIT 宏，则使用 cusolver 64 位版本

    // 在设备和主机上分配工作空间存储
    auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_device_data = device_allocator.allocate(worksize_device);  // 在设备上分配工作空间存储
    auto& host_allocator = *at::getCPUAllocator();
    auto work_host_data = host_allocator.allocate(worksize_host);  // 在主机上分配工作空间存储

    // 调用 cusolver 中的 xgeqrf 函数，进行 QR 分解
    at::cuda::solver::xgeqrf<scalar_t>(
        handle,
        params,
        m,
        n,
        A_working_ptr,
        lda,
        tau_working_ptr,
        static_cast<scalar_t*>(work_device_data.get()),
        worksize_device,
        static_cast<scalar_t*>(work_host_data.get()),
        worksize_host,
        infos_data);
#else
    // 如果未定义 USE_CUSOLVER_64_BIT 宏，则使用通用的 cusolver 版本

    // 在设备上分配工作空间存储
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * std::max<int>(1, lwork));  // 在设备上分配工作空间存储

    // 调用 cusolver 中的 geqrf 函数，进行 QR 分解
    at::cuda::solver::geqrf<scalar_t>(
        handle,
        m_32,
        n_32,
        A_working_ptr,
        lda_32,
        tau_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        infos_data);
#endif // USE_CUSOLVER_64_BIT
  }

  // geqrf 函数的 infos 参数只报告参数是否正确，不报告矩阵的奇异性
  // 因此我们不需要每次都检查它
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.item().toInt() == 0);
}

// 这是一个为 'apply_geqrf' 函数进行类型分发的辅助函数
void geqrf_cusolver(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_cuda", [&]{
    apply_geqrf<scalar_t>(input, tau);
  });
}

/*
  ormqr 函数用于将 Q 乘以另一个由一系列初等反射器组成的矩阵，这些反射器由 geqrf 函数生成。

  Args:
  * `input`     - 包含矩阵 Q 下对角线的初等反射器的 Tensor。
  * `tau`       - 包含初等反射器的模的 Tensor。
  * `other`     - [in] 包含待乘以的矩阵的 Tensor。
                  [out] 与 Q 相乘后的矩阵乘积结果。
  * `left`      - bool，确定是左乘还是右乘以 Q。
  * `transpose` - bool，确定在乘以之前是否转置（或共轭转置）Q。

  有关更多详细信息，请参阅 cuSOLVER 中的 ORMQR 和 UNMQR 文档。
*/
template <typename scalar_t>
// 应用 ormqr 函数，用于在 CUDA 上执行 QR 因子分解的更新操作
static void apply_ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  // 确定是左侧操作还是右侧操作
  auto side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  // 确定是转置操作还是不转置操作，并根据输入类型选择正确的操作符
  auto trans = transpose ? (input.is_complex() ? CUBLAS_OP_C : CUBLAS_OP_T) : CUBLAS_OP_N;

  // 获取输入数据的常量指针，以及 tau 和 other 张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto tau_data = tau.const_data_ptr<scalar_t>();
  auto other_data = other.data_ptr<scalar_t>();

  // 计算输入矩阵和其他矩阵的步幅
  auto input_matrix_stride = matrixStride(input);
  auto other_matrix_stride = matrixStride(other);
  // 获取 tau 张量的步幅，批次大小，以及其他相关尺寸信息
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto m = cuda_int_cast(other.size(-2), "m");
  auto n = cuda_int_cast(other.size(-1), "n");
  auto k = cuda_int_cast(tau.size(-1), "k");
  // 计算 lda 和 ldc 的值，确保它们不小于1
  auto lda = std::max<int>(1, left ? m : n);
  auto ldc = std::max<int>(1, m);

  // 获取最优工作空间大小并分配工作空间张量
  int lwork;
  at::cuda::solver::ormqr_bufferSize<scalar_t>(
    at::cuda::getCurrentCUDASolverDnHandle(), side, trans, m, n, k, input_data, lda, tau_data, other_data, ldc, &lwork);

  // 创建一个包含单个元素的整型张量，用于存储函数的返回状态信息
  auto info = at::zeros({1}, input.options().dtype(at::kInt));
  auto info_data = info.data_ptr<int>();

  // 遍历批次中的每个元素，执行 ormqr 操作
  for (auto i = decltype(batch_size){0}; i < batch_size; i++) {
    // 指向当前批次的 input_data 和 other_data 的工作指针
    const scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

    // 分配工作空间存储
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t)*lwork);

    // 调用 ormqr 函数执行 QR 因子分解的更新操作
    at::cuda::solver::ormqr<scalar_t>(
      handle, side, trans, m, n, k,
      input_working_ptr,
      lda,
      tau_working_ptr,
      other_working_ptr,
      ldc,
      static_cast<scalar_t*>(work_data.get()),
      lwork,
      info_data
    );

    // 断言 info 参数为零，表示没有错误发生
    // 因为 ormqr 函数只报告第 i 个参数错误的情况
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
  }
}

// 这是用于 'apply_ormqr' 的类型分发辅助函数
void ormqr_cusolver(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  // 在浮点和复数类型上分发，调用对应的 apply_ormqr 函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "orgmr_cuda", [&]{
    apply_ormqr<scalar_t>(input, tau, other, left, transpose);
  });
}

/*
  orgqr 函数允许从一系列初等反射器中重建正交（或单位）矩阵 Q，
  例如由 geqrf 函数生成的反射器。

  Args:
  * `self` - 包含初等反射器在对角线以下方向的张量，它将被结果覆盖
  * `tau` - 包含初等反射器的大小的张量

  有关详细信息，请参阅 cuSOLVER 的 ORGQR 和 UNGQR 文档。
*/
template <typename scalar_t>
inline static void apply_orgqr(Tensor& self, const Tensor& tau) {
  auto self_data = self.data_ptr<scalar_t>();  // 获取 self 张量的数据指针
  auto tau_data = tau.const_data_ptr<scalar_t>();  // 获取 tau 张量的常量数据指针
  auto self_matrix_stride = matrixStride(self);  // 计算 self 张量的矩阵步幅
  auto batchsize = cuda_int_cast(batchCount(self), "batch size");  // 计算 self 张量的批次大小
  auto m = cuda_int_cast(self.size(-2), "m");  // 获取 self 张量的倒数第二维大小 m
  auto n = cuda_int_cast(self.size(-1), "n");  // 获取 self 张量的最后一维大小 n
  auto k = cuda_int_cast(tau.size(-1), "k");  // 获取 tau 张量的最后一维大小 k
  auto tau_stride = std::max<int>(1, k);  // 计算 tau 张量的步幅
  auto lda = std::max<int>(1, m);  // 计算 lda 参数，最小为1

  // LAPACK 的要求
  TORCH_INTERNAL_ASSERT(m >= n);  // 断言 m >= n
  TORCH_INTERNAL_ASSERT(n >= k);  // 断言 n >= k

  // cuSOLVER 对于 k 等于 0 的情况不会计算任何东西，这是错误的
  // 结果应该是对角线上为 1 的矩阵
  if (k == 0) {
    self.fill_(0);  // 将 self 张量填充为 0
    self.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);  // 将对角线元素设置为 1
    return;  // 返回
  }

  // 获取最优工作空间大小并分配工作空间张量
  int lwork;
  at::cuda::solver::orgqr_buffersize<scalar_t>(
    at::cuda::getCurrentCUDASolverDnHandle(), m, n, k, self_data, lda, tau_data, &lwork);

  auto info = at::zeros({1}, self.options().dtype(at::kInt));  // 创建一个包含一个元素的全零张量 info
  auto info_data = info.data_ptr<int>();  // 获取 info 张量的数据指针

  for (auto i = decltype(batchsize){0}; i < batchsize; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];  // 计算当前批次的 self 张量工作指针
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];  // 计算当前批次的 tau 张量工作指针
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();  // 获取当前 cuSOLVER 句柄

    // 分配工作空间存储
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t)*lwork);

    // 调用 cuSOLVER 的 orgqr 函数进行 QR 分解
    at::cuda::solver::orgqr<scalar_t>(
      handle, m, n, k,
      self_working_ptr,
      lda,
      tau_working_ptr,
      static_cast<scalar_t*>(work_data.get()),
      lwork,
      info_data
    );

    // orgqr 函数的 info 只报告第 i 个参数错误，因此不需要一直检查
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);  // 断言 orgqr 的 info 返回为 0
  }
}

// 这是 'apply_orgqr' 的类型分派辅助函数
Tensor& orgqr_helper_cusolver(Tensor& result, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cuda", [&]{
    apply_orgqr<scalar_t>(result, tau);  // 调用实际的 apply_orgqr 函数
  });
  return result;  // 返回 result 张量
}
// 定义静态函数 apply_syevd，用于执行对称矩阵的特征值分解，并可选计算特征向量
static void apply_syevd(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // 确定值类型 value_t，这里是标量类型的类型定义
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 根据 upper 参数确定填充模式
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 根据 compute_eigenvectors 参数确定求解模式
  cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  // 获取矩阵维度 n
  int64_t n = vectors.size(-1);
  // 计算 leading dimension lda，至少为 1
  int64_t lda = std::max<int64_t>(1, n);
  // 计算 batch 大小
  int64_t batch_size = batchCount(vectors);

  // 获取矩阵 tensors 的步长信息
  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  // 获取 tensors 数据指针
  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // 获取优化后的工作空间大小，并分配工作空间 tensor
#ifdef USE_CUSOLVER_64_BIT
  size_t worksize_device; // 设备端的工作空间字节数
  size_t worksize_host; // 主机端的工作空间字节数
  cusolverDnParams_t params = NULL; // 使用默认算法（当前唯一选项）
  // 获取 xsyevd 的工作空间大小
  at::cuda::solver::xsyevd_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(),
      params,
      jobz,
      uplo,
      n,
      vectors_data,
      lda,
      values_data,
      &worksize_device,
      &worksize_host);
#else
  int lwork; // 工作空间大小
  int n_32 = cuda_int_cast(n, "n");
  int lda_32 = cuda_int_cast(lda, "lda");
  // 获取 syevd 的工作空间大小
  at::cuda::solver::syevd_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(), jobz, uplo, n_32, vectors_data, lda_32, values_data, &lwork);
#endif // USE_CUSOLVER_64_BIT

  // 对每个 batch 执行特征值分解
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    // 计算当前 batch 的指针
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    int* info_working_ptr = &infos_data[i];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

#ifdef USE_CUSOLVER_64_BIT
    // 在设备端和主机端分配工作空间存储
    auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_device_data = device_allocator.allocate(worksize_device);
    auto& host_allocator = *at::getCPUAllocator();
    auto work_host_data = host_allocator.allocate(worksize_host);
    // 执行 xsyevd 特征值分解
    at::cuda::solver::xsyevd<scalar_t>(
        handle,
        params,
        jobz,
        uplo,
        n,
        vectors_working_ptr,
        lda,
        values_working_ptr,
        static_cast<scalar_t*>(work_device_data.get()),
        worksize_device,
        static_cast<scalar_t*>(work_host_data.get()),
        worksize_host,
        info_working_ptr);
#else
    // 在设备端分配工作空间存储
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
    // 执行 syevd 特征值分解
    at::cuda::solver::syevd<scalar_t>(
        handle,
        jobz,
        uplo,
        n_32,
        vectors_working_ptr,
        lda_32,
        values_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        info_working_ptr);
#endif // USE_CUSOLVER_64_BIT
  }
}

template <typename scalar_t>
static void apply_syevj(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 确定矩阵的上三角或下三角部分填充方式
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 指定是否计算特征向量
  cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  // 确定矩阵的维度
  int n = cuda_int_cast(vectors.size(-1), "n");
  // 计算矩阵的 lda (leading dimension)，至少为1
  int lda = std::max<int>(1, n);
  auto batch_size = batchCount(vectors);

  // 计算向量张量的步幅和值张量的步幅
  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  // 获取张量数据的指针
  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // 设置 syevj 的参数，控制数值精度和迭代次数
  // 默认情况下，公差被设置为机器精度，Jacobi 方法的最大迭代次数默认为100
  syevjInfo_t syevj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

  // 获取最优工作空间大小并分配工作空间张量
  int lwork;
  at::cuda::solver::syevj_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(), jobz, uplo, n, vectors_data, lda, values_data, &lwork, syevj_params);

  // 针对每个批次执行特征值求解操作
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    // 计算当前批次的向量工作指针、值工作指针和信息工作指针
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    int* info_working_ptr = &infos_data[i];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

    // 在设备上分配工作空间存储
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
    // 执行 syevj 特征值求解函数
    at::cuda::solver::syevj<scalar_t>(
        handle,
        jobz,
        uplo,
        n,
        vectors_working_ptr,
        lda,
        values_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        info_working_ptr,
        syevj_params);
  }
  // 销毁 syevj 参数对象
  TORCH_CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

template <typename scalar_t>
// 使用 CUSOLVER 库中的 syevj_batched 函数对一批张量进行特征值分解
static void apply_syevj_batched(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // 确定张量元素类型的值类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 根据 upper 参数确定填充模式
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 根据 compute_eigenvectors 参数确定计算模式
  cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  // 获取张量中的向量大小作为特征值分解的维度 n
  int n = cuda_int_cast(vectors.size(-1), "n");
  // 计算 lda，确保至少为 1
  int lda = std::max<int>(1, n);
  // 获取批处理的大小
  int batch_size = cuda_int_cast(batchCount(vectors), "batch_size");

  // 获取张量的指针数据
  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // 初始化特征值分解的参数 syevj_params，并设置默认值
  syevjInfo_t syevj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, 1));

  // 获取当前 CUDA 操作句柄
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  // 获取最佳工作空间大小并分配工作空间张量
  int lwork;
  at::cuda::solver::syevjBatched_bufferSize<scalar_t>(
      handle,
      jobz,
      uplo,
      n,
      vectors_data,
      lda,
      values_data,
      &lwork,
      syevj_params,
      batch_size);

  // 在设备上分配工作空间存储
  auto& allocator = *at::cuda::getCUDADeviceAllocator();
  auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
  at::cuda::solver::syevjBatched<scalar_t>(
      handle,
      jobz,
      uplo,
      n,
      vectors_data,
      lda,
      values_data,
      static_cast<scalar_t*>(work_data.get()),
      lwork,
      infos_data,
      syevj_params,
      batch_size);
  
  // 销毁 syevj_params 特征值分解参数对象
  TORCH_CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

// 使用 CUSOLVER 库中的 syevd 函数进行特征值分解
static void linalg_eigh_cusolver_syevd(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevd<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

// 使用 CUSOLVER 库中的 syevj 函数进行特征值分解
static void linalg_eigh_cusolver_syevj(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevj<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}
// 使用 cusolver 库中的 syevj_batched 算法计算批量特征值和特征向量
static void linalg_eigh_cusolver_syevj_batched(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    // 调用模板化函数 apply_syevj_batched，根据标量类型执行批量计算特征值和特征向量
    apply_syevj_batched<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

// 使用 cusolver 库中的 syevj 或 syevd 算法计算特征值和特征向量
void linalg_eigh_cusolver(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // 对于 ROCm 平台的 hipSolver，syevj 算法最快
#ifdef USE_ROCM
  linalg_eigh_cusolver_syevj(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
#else
  // 如果启用了 syevj_batched，并且批次计数大于 1，且矩阵大小小于等于 32，则使用 syevjBatched 算法
  if (use_cusolver_syevj_batched_ && batchCount(eigenvectors) > 1 && eigenvectors.size(-1) <= 32) {
    // 详情参见 https://github.com/pytorch/pytorch/pull/53040#issuecomment-788264724
    linalg_eigh_cusolver_syevj_batched(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else if (eigenvectors.scalar_type() == at::kFloat && eigenvectors.size(-1) >= 32 && eigenvectors.size(-1) <= 512) {
    // 对于 float32 类型和矩阵大小在 32x32 到 512x512 范围内，syevj 比 syevd 更优
    // 详情参见 https://github.com/pytorch/pytorch/pull/53040#issuecomment-788264724
    linalg_eigh_cusolver_syevj(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else {
    // 否则使用 syevd 算法计算特征值和特征向量
    linalg_eigh_cusolver_syevd(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
#endif
}

// lu_factor_looped_cusolver 函数使用 cusolver 库执行 LU 分解
// 'apply_' 通常用于模板化的函数，这里不使用是因为 cusolver API 结构稍有不同
void lu_factor_looped_cusolver(const Tensor& self, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    self.scalar_type(),
    "lu_factor_cusolver",
    [&self,
     &pivots,
     &infos,
     get_pivots]() {
    // 获取矩阵的行数 m 和列数 n
    const auto m = cuda_int_cast(self.size(-2), "m");
    const auto n = cuda_int_cast(self.size(-1), "n");
    // 计算 leading dimension lda，最小为 1
    const auto lda = std::max<int>(1, m);
    // 获取矩阵的步长 self_stride 和批次数 batch_size
    const auto self_stride = matrixStride(self);
    const auto batch_size = batchCount(self);
    // 获取数据指针和 infos 的数据指针
    const auto self_data = self.data_ptr<scalar_t>();
    const auto infos_data = infos.data_ptr<int>();

    // 如果需要获取 pivot，则获取 pivots 的数据指针和步长
    const auto pivots_data = get_pivots ? pivots.data_ptr<int>() : nullptr;
    const auto pivots_stride = get_pivots ? pivots.size(-1) : 0;

    // 获取当前 CUDA Solver 的句柄
    const auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    // 循环处理每个批次的数据
    for (auto batch = decltype(batch_size){0}; batch < batch_size; ++batch) {
      // 调用 cusolver 的 getrf 函数执行 LU 分解
      at::cuda::solver::getrf<scalar_t>(
        handle, m, n,
        self_data + batch * self_stride,
        lda,
        get_pivots ? pivots_data + batch * pivots_stride : nullptr,
        infos_data + batch
      );
  });


// 闭合前面的 JavaScript 函数定义块。


  // Necessary because cuSOLVER uses nan for outputs that correspond to 0 in MAGMA for non-pivoted LU.
  // https://github.com/pytorch/pytorch/issues/53879#issuecomment-830633572


// 这里的注释解释了为什么需要进行下面的操作，因为 cuSOLVER 在非枢轴化 LU 中将输出 0 对应的值设置为 NaN。


  if (!get_pivots) {


// 检查是否不需要获取枢轴信息。


    // nan_to_num does not work for complex inputs
    // https://github.com/pytorch/pytorch/issues/59247


// 如果输入是复数，nan_to_num 函数不起作用，这里给出了问题的 GitHub 链接。


    if (self.is_complex()) {


// 检查张量是否是复数类型。


      self.copy_(at::where(self.eq(self), self,  at::scalar_tensor(0., self.options())));


// 对于复数类型的张量，将等于自身的值替换为 0。


    } else {
      at::nan_to_num_(const_cast<Tensor&>(self), 0, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    }


// 对于非复数类型的张量，使用 nan_to_num 函数将 NaN 替换为 0。


  }


// 结束条件语句块，检查是否不需要获取枢轴信息的操作已完成。
// 结束条件，关闭 USE_LINALG_SOLVER 宏定义
#endif  // USE_LINALG_SOLVER

// 结束命名空间 at::native
} // namespace at::native
```