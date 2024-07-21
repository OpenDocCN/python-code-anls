# `.\pytorch\aten\src\ATen\cuda\CUDASparseDescriptors.h`

```
#pragma once
// 引入 ATen 库的相关头文件，用于操作 CUDA 上的稀疏张量
#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparse.h>

// 引入 C10 库的标量类型定义
#include <c10/core/ScalarType.h>

// 如果使用 ROCm 平台，则引入 type_traits 库
#if defined(USE_ROCM)
#include <type_traits>
#endif

// 定义在 at::cuda::sparse 命名空间下的模板结构 CuSparseDescriptorDeleter
template <typename T, cusparseStatus_t (*destructor)(T*)>
struct CuSparseDescriptorDeleter {
  void operator()(T* x) {
    // 检查指针是否为空，如果不为空，则调用 destructor 删除对象
    if (x != nullptr) {
      TORCH_CUDASPARSE_CHECK(destructor(x));
    }
  }
};

// 定义在 at::cuda::sparse 命名空间下的模板类 CuSparseDescriptor
template <typename T, cusparseStatus_t (*destructor)(T*)>
class CuSparseDescriptor {
 public:
  // 返回常量版本的 descriptor 指针
  T* descriptor() const {
    return descriptor_.get();
  }
  // 返回非常量版本的 descriptor 指针
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  // 使用 std::unique_ptr 包装 descriptor，并指定自定义的 deleter
  std::unique_ptr<T, CuSparseDescriptorDeleter<T, destructor>> descriptor_;
};

// 如果定义了 AT_USE_CUSPARSE_CONST_DESCRIPTORS() 或 AT_USE_HIPSPARSE_CONST_DESCRIPTORS()
#if AT_USE_CUSPARSE_CONST_DESCRIPTORS() || AT_USE_HIPSPARSE_CONST_DESCRIPTORS()
// 定义在 at::cuda::sparse 命名空间下的模板结构 ConstCuSparseDescriptorDeleter
template <typename T, cusparseStatus_t (*destructor)(const T*)>
struct ConstCuSparseDescriptorDeleter {
  void operator()(T* x) {
    // 检查指针是否为空，如果不为空，则调用 destructor 删除对象
    if (x != nullptr) {
      TORCH_CUDASPARSE_CHECK(destructor(x));
    }
  }
};

// 定义在 at::cuda::sparse 命名空间下的模板类 ConstCuSparseDescriptor
template <typename T, cusparseStatus_t (*destructor)(const T*)>
class ConstCuSparseDescriptor {
 public:
  // 返回常量版本的 descriptor 指针
  T* descriptor() const {
    return descriptor_.get();
  }
  // 返回非常量版本的 descriptor 指针
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  // 使用 std::unique_ptr 包装 descriptor，并指定自定义的 deleter
  std::unique_ptr<T, ConstCuSparseDescriptorDeleter<T, destructor>> descriptor_;
};
#endif // AT_USE_CUSPARSE_CONST_DESCRIPTORS || AT_USE_HIPSPARSE_CONST_DESCRIPTORS

// 如果定义了 USE_ROCM，定义相应的类型别名
#if defined(USE_ROCM)
using cusparseMatDescr = std::remove_pointer<hipsparseMatDescr_t>::type;
using cusparseDnMatDescr = std::remove_pointer<hipsparseDnMatDescr_t>::type;
using cusparseDnVecDescr = std::remove_pointer<hipsparseDnVecDescr_t>::type;
using cusparseSpMatDescr = std::remove_pointer<hipsparseSpMatDescr_t>::type;
using cusparseSpGEMMDescr = std::remove_pointer<hipsparseSpGEMMDescr_t>::type;
// 如果定义了 AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()，定义额外的类型别名
#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
using bsrsv2Info = std::remove_pointer<bsrsv2Info_t>::type;
using bsrsm2Info = std::remove_pointer<bsrsm2Info_t>::type;
#endif
#endif

// 定义在 at::cuda::sparse 命名空间下的函数 destroyConstDnMat，用于销毁常量描述符
cusparseStatus_t destroyConstDnMat(const cusparseDnMatDescr* dnMatDescr);

// 定义在 at::cuda::sparse 命名空间下的类 CuSparseMatDescriptor
class TORCH_CUDA_CPP_API CuSparseMatDescriptor
    : public CuSparseDescriptor<cusparseMatDescr, &cusparseDestroyMatDescr> {
 public:
  // 默认构造函数，创建一个空的稀疏矩阵描述符
  CuSparseMatDescriptor() {
    cusparseMatDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }

  // 构造函数，根据给定的 upper 和 unit 参数创建稀疏矩阵描述符
  CuSparseMatDescriptor(bool upper, bool unit) {
    // 根据 upper 和 unit 参数确定填充模式和对角线类型
    cusparseFillMode_t fill_mode =
        upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type =
        unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseMatDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
    # 使用 TORCH_CUDASPARSE_CHECK 函数设置稀疏矩阵的填充模式，使用给定的 fill_mode 参数
    TORCH_CUDASPARSE_CHECK(cusparseSetMatFillMode(raw_descriptor, fill_mode));
    # 使用 TORCH_CUDASPARSE_CHECK 函数设置稀疏矩阵的对角线类型，使用给定的 diag_type 参数
    TORCH_CUDASPARSE_CHECK(cusparseSetMatDiagType(raw_descriptor, diag_type));
    # 将原始的 cuSPARSE 描述符包装到 descriptor_ 智能指针中
    descriptor_.reset(raw_descriptor);
  }
// 如果使用 HIPSPARSE 的三角求解，则编译以下代码块
#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

// 定义 CuSparseBsrsv2Info 类，继承自 CuSparseDescriptor，用于管理 BSRSV2 操作的描述符
class TORCH_CUDA_CPP_API CuSparseBsrsv2Info
    : public CuSparseDescriptor<bsrsv2Info, &cusparseDestroyBsrsv2Info> {
 public:
  // 构造函数，创建 BSRSV2 操作的描述符对象
  CuSparseBsrsv2Info() {
    bsrsv2Info_t raw_descriptor = nullptr;
    // 调用 cusparseCreateBsrsv2Info 创建 BSRSV2 描述符
    TORCH_CUDASPARSE_CHECK(cusparseCreateBsrsv2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

// 定义 CuSparseBsrsm2Info 类，继承自 CuSparseDescriptor，用于管理 BSRSM2 操作的描述符
class TORCH_CUDA_CPP_API CuSparseBsrsm2Info
    : public CuSparseDescriptor<bsrsm2Info, &cusparseDestroyBsrsm2Info> {
 public:
  // 构造函数，创建 BSRSM2 操作的描述符对象
  CuSparseBsrsm2Info() {
    bsrsm2Info_t raw_descriptor = nullptr;
    // 调用 cusparseCreateBsrsm2Info 创建 BSRSM2 描述符
    TORCH_CUDASPARSE_CHECK(cusparseCreateBsrsm2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

// 如果使用 CUSPARSE 的通用 API 或 HIPSPARSE 的通用 API，则编译以下代码块
#if AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

// 根据标量类型获取相应的 CuSparse 索引类型
cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type);

// 如果使用非常量描述符，则编译以下代码块
#if AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() || AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS()
// 定义 CuSparseDnMatDescriptor 类，继承自 CuSparseDescriptor，用于管理密集矩阵描述符
class TORCH_CUDA_CPP_API CuSparseDnMatDescriptor
    : public CuSparseDescriptor<cusparseDnMatDescr, &cusparseDestroyDnMat> {
 public:
  // 构造函数，创建密集矩阵描述符对象
  explicit CuSparseDnMatDescriptor(const Tensor& input, int64_t batch_offset = -1);
};

// 定义 CuSparseConstDnMatDescriptor 类，继承自 CuSparseDescriptor，用于管理常量密集矩阵描述符
class TORCH_CUDA_CPP_API CuSparseConstDnMatDescriptor
    : public CuSparseDescriptor<const cusparseDnMatDescr, &destroyConstDnMat> {
 public:
  // 构造函数，创建常量密集矩阵描述符对象
  explicit CuSparseConstDnMatDescriptor(const Tensor& input, int64_t batch_offset = -1);
  // 返回可变的密集矩阵描述符指针
  cusparseDnMatDescr* unsafe_mutable_descriptor() const {
    return const_cast<cusparseDnMatDescr*>(descriptor());
  }
  // 返回可变的密集矩阵描述符指针
  cusparseDnMatDescr* unsafe_mutable_descriptor() {
    return const_cast<cusparseDnMatDescr*>(descriptor());
  }
};

// 定义 CuSparseDnVecDescriptor 类，继承自 CuSparseDescriptor，用于管理密集向量描述符
class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
    : public CuSparseDescriptor<cusparseDnVecDescr, &cusparseDestroyDnVec> {
 public:
  // 构造函数，创建密集向量描述符对象
  explicit CuSparseDnVecDescriptor(const Tensor& input);
};

// 定义 CuSparseSpMatDescriptor 类，继承自 CuSparseDescriptor，用于管理稀疏矩阵描述符
class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
    : public CuSparseDescriptor<cusparseSpMatDescr, &cusparseDestroySpMat> {};

// 如果使用常量描述符，则编译以下代码块
#elif AT_USE_CUSPARSE_CONST_DESCRIPTORS() || AT_USE_HIPSPARSE_CONST_DESCRIPTORS()
// 定义 CuSparseDnMatDescriptor 类，继承自 ConstCuSparseDescriptor，用于管理常量密集矩阵描述符
class TORCH_CUDA_CPP_API CuSparseDnMatDescriptor
    : public ConstCuSparseDescriptor<
          cusparseDnMatDescr,
          &cusparseDestroyDnMat> {
 public:
  // 构造函数，创建常量密集矩阵描述符对象
  explicit CuSparseDnMatDescriptor(
      const Tensor& input,
      int64_t batch_offset = -1);
};

// 定义 CuSparseConstDnMatDescriptor 类，继承自 ConstCuSparseDescriptor，用于管理常量密集矩阵描述符
class TORCH_CUDA_CPP_API CuSparseConstDnMatDescriptor
    : public ConstCuSparseDescriptor<
          const cusparseDnMatDescr,
          &destroyConstDnMat> {
 public:
  // 构造函数，创建常量密集矩阵描述符对象
  explicit CuSparseConstDnMatDescriptor(
      const Tensor& input,
      int64_t batch_offset = -1);
  // 返回可变的密集矩阵描述符指针
  cusparseDnMatDescr* unsafe_mutable_descriptor() const {
    return const_cast<cusparseDnMatDescr*>(descriptor());
  }
  // 返回可变的密集矩阵描述符指针
  cusparseDnMatDescr* unsafe_mutable_descriptor() {
    // 返回描述符的指针，使用 const_cast 去除常量属性，并返回 cusparseDnMatDescr* 类型
    return const_cast<cusparseDnMatDescr*>(descriptor());
    }
    
    };
    
    // CuSparseDnVecDescriptor 类，继承自 ConstCuSparseDescriptor
    class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
        : public ConstCuSparseDescriptor<
              cusparseDnVecDescr,  // 使用 cusparseDnVecDescr 作为模板参数
              &cusparseDestroyDnVec> {  // 使用 cusparseDestroyDnVec 作为销毁函数指针
     public:
      // 显式构造函数，接受 Tensor 类型的 input 参数
      explicit CuSparseDnVecDescriptor(const Tensor& input);
    };
    
    // CuSparseSpMatDescriptor 类，继承自 ConstCuSparseDescriptor
    class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
        : public ConstCuSparseDescriptor<
              cusparseSpMatDescr,  // 使用 cusparseSpMatDescr 作为模板参数
              &cusparseDestroySpMat> {};  // 使用 cusparseDestroySpMat 作为销毁函数指针
#endif // AT_USE_CUSPARSE_CONST_DESCRIPTORS() || AT_USE_HIPSPARSE_CONST_DESCRIPTORS()

class TORCH_CUDA_CPP_API CuSparseSpMatCsrDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  // 构造函数，接受输入张量和批处理偏移，默认为-1
  explicit CuSparseSpMatCsrDescriptor(const Tensor& input, int64_t batch_offset = -1);

  // 返回稀疏矩阵的大小：行数、列数、非零元素数
  std::tuple<int64_t, int64_t, int64_t> get_size() {
    int64_t rows = 0, cols = 0, nnz = 0;
    // 调用 cusparseSpMatGetSize 获取稀疏矩阵的大小信息
    TORCH_CUDASPARSE_CHECK(cusparseSpMatGetSize(
        this->descriptor(),
        &rows,
        &cols,
        &nnz));
    return std::make_tuple(rows, cols, nnz);
  }

  // 设置描述符的输入张量
  void set_tensor(const Tensor& input) {
    auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    // 断言输入张量的数据连续性
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    
    // 调用 cusparseCsrSetPointers 设置 CSR 矩阵的指针
    TORCH_CUDASPARSE_CHECK(cusparseCsrSetPointers(
        this->descriptor(),
        crow_indices.data_ptr(),
        col_indices.data_ptr(),
        values.data_ptr()));
  }

#if AT_USE_CUSPARSE_GENERIC_SPSV()
  // 设置矩阵的填充模式：上三角或下三角
  void set_mat_fill_mode(bool upper) {
    cusparseFillMode_t fill_mode =
        upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    // 调用 cusparseSpMatSetAttribute 设置稀疏矩阵的填充模式属性
    TORCH_CUDASPARSE_CHECK(cusparseSpMatSetAttribute(
        this->descriptor(),
        CUSPARSE_SPMAT_FILL_MODE,
        &fill_mode,
        sizeof(fill_mode)));
  }

  // 设置矩阵的对角线类型：单位对角线或非单位对角线
  void set_mat_diag_type(bool unit) {
    cusparseDiagType_t diag_type =
        unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    // 调用 cusparseSpMatSetAttribute 设置稀疏矩阵的对角线类型属性
    TORCH_CUDASPARSE_CHECK(cusparseSpMatSetAttribute(
        this->descriptor(),
        CUSPARSE_SPMAT_DIAG_TYPE,
        &diag_type,
        sizeof(diag_type)));
  }
#endif
};

#if AT_USE_CUSPARSE_GENERIC_SPSV()
// CuSparseSpSVDescriptor 类，继承自 CuSparseDescriptor，用于描述稀疏向量-稀疏向量的操作
class TORCH_CUDA_CPP_API CuSparseSpSVDescriptor
    : public CuSparseDescriptor<cusparseSpSVDescr, &cusparseSpSV_destroyDescr> {
 public:
  // 构造函数，创建稀疏向量-稀疏向量描述符
  CuSparseSpSVDescriptor() {
    cusparseSpSVDescr_t raw_descriptor = nullptr;
    // 调用 cusparseSpSV_createDescr 创建稀疏向量-稀疏向量描述符
    TORCH_CUDASPARSE_CHECK(cusparseSpSV_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

#if AT_USE_CUSPARSE_GENERIC_SPSM()
// CuSparseSpSMDescriptor 类，继承自 CuSparseDescriptor，用于描述稀疏矩阵-稠密矩阵的操作
class TORCH_CUDA_CPP_API CuSparseSpSMDescriptor
    : public CuSparseDescriptor<cusparseSpSMDescr, &cusparseSpSM_destroyDescr> {
 public:
  // 构造函数，创建稀疏矩阵-稠密矩阵描述符
  CuSparseSpSMDescriptor() {
    cusparseSpSMDescr_t raw_descriptor = nullptr;
    // 调用 cusparseSpSM_createDescr 创建稀疏矩阵-稠密矩阵描述符
    TORCH_CUDASPARSE_CHECK(cusparseSpSM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

// CuSparseSpGEMMDescriptor 类，继承自 CuSparseDescriptor，用于描述稀疏矩阵乘法的操作
class TORCH_CUDA_CPP_API CuSparseSpGEMMDescriptor
    : public CuSparseDescriptor<cusparseSpGEMMDescr, &cusparseSpGEMM_destroyDescr> {
 public:
  // 构造函数，创建稀疏矩阵乘法描述符
  CuSparseSpGEMMDescriptor() {
    cusparseSpGEMMDescr_t raw_descriptor = nullptr;
    // 调用 cusparseSpGEMM_createDescr 创建稀疏矩阵乘法描述符
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

#endif // AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

} // namespace at::cuda::sparse
```