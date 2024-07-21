# `.\pytorch\aten\src\ATen\native\sparse\cuda\cuSPARSELtOps.cpp`

```py
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>
#include <cusparse.h>
#include <cstdint>

#if AT_CUSPARSELT_ENABLED()

#include <cusparseLt.h>

namespace at::native {

// Ideally we would use the same DeviceThreadHandlePool mechanism as used in aten/src/ATen/cuda/CuSparseHandlePool.cpp
// which would handle this for us. However, the cuSPARSELt handle signature is different from that of cuSPARSE/cuBLAS,
// so it's not possible to reuse the existing pooling mechanism. Instead we have to handle our handles ourselves, which
// is why these variables are thread local. Once cuSPARSELt updates their handle signature to be consistent with the rest
// of CUDA, we can switch to using DeviceThreadHandlePool.
thread_local cusparseLtHandle_t handle;  // 线程局部变量，用于保存 cuSPARSELt 的 handle
thread_local bool handle_initialized = false;  // 线程局部变量，标记 handle 是否已初始化

// 压缩稀疏张量
at::Tensor _cslt_compress(const Tensor& sparse_input)
{
    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));  // 如果 handle 尚未初始化，则进行初始化
        handle_initialized = true;
    }
    
    cusparseLtMatDescriptor_t sparse_input_descriptor;  // 定义稀疏张量描述符
    cudaDataType type;  // 定义 CUDA 数据类型
    auto compression_factor = 9;  // 压缩因子的初始值为 9

    switch (sparse_input.scalar_type())  // 根据输入稀疏张量的数据类型进行处理
    {
        case at::ScalarType::Char:
            type = CUDA_R_8I;  // 如果数据类型为 Char，使用 CUDA_R_8I
            compression_factor = 10;  // 调整压缩因子为 10
            break;
        case at::ScalarType::Half:
            type = CUDA_R_16F;  // 如果数据类型为 Half，使用 CUDA_R_16F
            break;
        case at::ScalarType::BFloat16:
            type = CUDA_R_16BF;  // 如果数据类型为 BFloat16，使用 CUDA_R_16BF
            break;
        case at::ScalarType::Float:
            type = CUDA_R_32F;  // 如果数据类型为 Float，使用 CUDA_R_32F
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt compressed matrix");  // 不支持的数据类型抛出异常
            break;
    }

    // 创建一个具有相同数据类型的新压缩张量
    auto compressed_tensor = sparse_input.new_empty(sparse_input.numel() * compression_factor / 16);

    // 初始化稀疏描述符
    TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
        &handle,
        &sparse_input_descriptor,
        sparse_input.size(0),
        sparse_input.size(1),
        sparse_input.size(1),
        16,
        type,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    // 计算压缩后的大小
    size_t compressed_size, compressed_buffer_size;
    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
        &handle,
        &sparse_input_descriptor,
        &compressed_size,
        &compressed_buffer_size));

    // 分配压缩缓冲区
    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    # 调用 cuSparse 库中的 cusparseLtSpMMACompress2 函数，用于执行稀疏矩阵-矩阵乘法运算中的压缩操作
    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
        # cuSparse 运算句柄，用于管理 cuSparse 库中的状态和资源
        &handle,
        # 稀疏输入张量的描述符，指定其格式和其他属性
        &sparse_input_descriptor,
        # 布尔值参数，表示进行压缩操作
        true,
        # 稀疏矩阵操作的转置选项，这里表示不进行转置操作
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        # 稀疏输入张量的数据指针，用于传递给 cuSparse 函数进行计算
        sparse_input.data_ptr(),
        # 压缩后的张量数据指针，接收 cuSparse 函数计算后的结果
        compressed_tensor.data_ptr(),
        # 指向压缩缓冲区的指针，用于临时存储 cuSparse 函数运算过程中的中间数据
        compressedBufferPtr.get(),
        # CUDA 流，指定在哪个 CUDA 流上执行此 cuSparse 函数调用
        stream));

    # 返回压缩后的张量作为函数的结果
    return compressed_tensor;
// 定义函数_cslt_sparse_mm_impl，接受压缩稀疏矩阵compressed_A、稠密矩阵dense_B，
// 可选的偏置bias_opt、alpha值alpha_opt、输出数据类型out_dtype_opt、是否转置结果transpose_result、
// 算法ID alg_id和搜索算法ID search_alg_id，并返回int64_t和at::Tensor组成的tuple
std::tuple<int64_t, at::Tensor> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    bool search_alg_id
)
{
    // 如果handle未初始化，则初始化cusparseLt
    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
        handle_initialized = true;
    }
    
    // 定义cusparseLtMatmulDescriptor_t、cusparseLtMatmulPlan_t和cusparseLtMatmulAlgSelection_t变量
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulPlan_t plan;
    cusparseLtMatmulAlgSelection_t alg_sel;

    // 初始化一些变量
    int tensor_alpha_mode = 0;
    float alpha = 1.0;
    float beta = 0.0;
    cudaDataType input_type;
    cudaDataType output_type;
    cusparseComputeType compute_type;
    auto compression_factor = 9;

    // 根据压缩矩阵compressed_A的数据类型选择不同的CUDA数据类型和计算类型
    switch(compressed_A.scalar_type())
    {
        case at::ScalarType::Char:
            input_type = CUDA_R_8I;
            output_type = CUDA_R_8I;
            compute_type = CUSPARSE_COMPUTE_32I;
            compression_factor = 10;
            break;

        // 如果使用cuSPARSELt版本大于等于0.5.2，则选择不同的数据类型和计算类型
        #if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 502
        case at::ScalarType::Half:
            input_type = CUDA_R_16F;
            output_type = CUDA_R_16F;
            compute_type = CUSPARSE_COMPUTE_32F;
            break;
        case at::ScalarType::BFloat16:
            input_type = CUDA_R_16BF;
            output_type = CUDA_R_16BF;
            compute_type = CUSPARSE_COMPUTE_32F;
            break;
        case at::ScalarType::Float:
            input_type = CUDA_R_32F;
            output_type = CUDA_R_32F;
            compute_type = CUSPARSE_COMPUTE_32F;
            break;

        // 如果使用cuSPARSELt版本小于等于0.5.2，则选择另一组数据类型和计算类型
        #else
        case at::ScalarType::Half:
            input_type = CUDA_R_16F;
            output_type = CUDA_R_16F;
            compute_type = CUSPARSE_COMPUTE_16F;
            break;
        case at::ScalarType::BFloat16:
            input_type = CUDA_R_16BF;
            output_type = CUDA_R_16BF;
            compute_type = CUSPARSE_COMPUTE_16F;
            break;
        case at::ScalarType::Float:
            input_type = CUDA_R_32F;
            output_type = CUDA_R_32F;
            compute_type = CUSPARSE_COMPUTE_TF32;
            break;
        #endif

        // 如果遇到不支持的数据类型，则抛出错误信息
        default:
            TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt compressed matrix multiplication.");
            break;
    }

    // 设置输出数据类型为dense_B的数据类型
    ScalarType out_dtype = dense_B.scalar_type();

    // 如果提供了输出数据类型的选择，则将out_dtype设置为所选的值
    if (out_dtype_opt.has_value()) {
        out_dtype = out_dtype_opt.value();
        // 对于int8输入，仅支持特定的输出数据类型
        TORCH_CHECK(input_type == CUDA_R_8I, "out_dtype support only available for int8 inputs");

        // 根据选择的输出数据类型设置不同的操作
        switch (out_dtype)
        {
            // 不同输出数据类型的处理可以继续在此添加
        }
    }
    
    // 此处应继续补充特定情况下的处理，以完整代码为准
    {
        // 根据输出类型选择相应的 CUDA 数据类型常量
        case at::ScalarType::Half:
            output_type = CUDA_R_16F;
            break;
        case at::ScalarType::BFloat16:
            output_type = CUDA_R_16BF;
            break;
        case at::ScalarType::Int:
            output_type = CUDA_R_32I;
            break;
        default:
            // 如果输出类型不支持，抛出错误信息
            TORCH_CHECK(false, "Unsupported out_dtype passed, must be one of {fp16, bf16, int32}");
            break;
    }
  }

  // 获取输入矩阵的维度信息
  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  // 计算稀疏矩阵的列数 m，以便于初始化稀疏矩阵描述符
  int64_t m = (compressed_A.numel() * 16 / compression_factor  ) / k;

  // 初始化稀疏输入矩阵描述符
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  // 初始化密集输入矩阵描述符
  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      (dense_B.is_contiguous()) ? k : n,
      (dense_B.is_contiguous()) ? n : k,
      (dense_B.is_contiguous()) ? n : k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW));

  // 创建结果张量
  auto res_tensor_options = c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                      : at::empty({m, n}, res_tensor_options);

  // 初始化结果矩阵描述符
  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m: n,
      16,
      output_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // 初始化矩阵乘法描述符
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type));

  // 如果有偏置值，设置偏置指针以用于矩阵乘法计算
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));



TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
    &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));



// 设置算法选择器中的算法ID为指定的alg_id
TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
    &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));



// 设置 matmul 描述符中的 alpha 参数为指定的 alpha_ptr
const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt : Tensor{};
const auto alpha_ptr = alpha_opt.has_value() ? alpha_tensor.data_ptr() : &alpha;
if (alpha_opt.has_value()) {
  tensor_alpha_mode = 1;
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &tensor_alpha_mode, sizeof(tensor_alpha_mode)));
}



TORCH_CUDASPARSE_CHECK(
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));



// 获取 matmul 执行所需的工作空间大小
size_t workspace_size;
TORCH_CUDASPARSE_CHECK(
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));



// 分配 GPU 内存以用作工作空间
auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
auto workspacePtr = allocator.allocate(workspace_size);
cudaStream_t stream = at::cuda::getCurrentCUDAStream();



if (search_alg_id) {
  // 运行 matmul 搜索
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
      &handle,
      &plan,
      alpha_ptr,
      compressed_A.data_ptr(),
      dense_B.data_ptr(),
      &beta,
      res.data_ptr(),
      res.data_ptr(),
      workspacePtr.get(),
      // jank because of the way we want this to be an array of streams
      &stream,
      1));

  // 获取使用的算法 ID
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
      &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
} else {
  // 执行普通的 matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
      &handle,
      &plan,
      alpha_ptr,
      compressed_A.data_ptr(),
      dense_B.data_ptr(),
      &beta,
      res.data_ptr(),
      res.data_ptr(),
      workspacePtr.get(),
      // jank because of the way we want this to be an array of streams
      &stream,
      1));
}



// 销毁描述符
TORCH_CUDASPARSE_CHECK(
    cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
TORCH_CUDASPARSE_CHECK(
    cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
// 销毁计划
TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

// 返回结果：包含算法 ID 和计算结果的 tuple
return {alg_id, res};
} // namespace at::native



#else // No cuSPARSELt support, throw error if these functions are called.

namespace at::native {

// 当 cuSPARSELt 不支持时，抛出错误如果这些函数被调用
at::Tensor _cslt_compress(const Tensor& sparse_input){
    TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

// 当 cuSPARSELt 不支持时，抛出错误如果这些函数被调用
at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result,
    int64_t alg_id)
{
    TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

// 当 cuSPARSELt 不支持时，抛出错误如果这些函数被调用
int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result
)
{
    TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

} // namespace at::native

#endif



namespace at::native {


注：以上是对 C++ 代码的注释，说明了命名空间和函数在不支持 cuSPARSELt 的情况下如何处理。
```