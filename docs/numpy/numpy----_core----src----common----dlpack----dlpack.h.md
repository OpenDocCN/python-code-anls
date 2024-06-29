# `.\numpy\numpy\_core\src\common\dlpack\dlpack.h`

```
// 根据以下链接获取的代码片段：
// https://github.com/dmlc/dlpack/blob/bbd2f4d32427e548797929af08cfe2a9cbb3cf12/include/dlpack/dlpack.h
// 并在其中为 DLManagedTensorVersioned 添加了 typedef

/*!
 *  版权声明，版权归 2017 年 Contributors 所有
 * \file dlpack.h
 * \brief DLPack 的公共头文件
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

/**
 * \brief 与 C++ 的兼容性宏定义
 */
#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief DLPack 当前的主版本号 */
#define DLPACK_MAJOR_VERSION 1

/*! \brief DLPack 当前的次版本号 */
#define DLPACK_MINOR_VERSION 0

/*! \brief DLPACK_DLL 在 Windows 下的前缀定义 */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief DLPack 的版本信息结构体
 *
 * 主版本号的改变表示 ABI 的数据布局发生了改变 - DLManagedTensorVersioned。
 * 次版本号的改变表示添加了新的代码，比如新的设备类型，但 ABI 保持不变。
 * 如果获取的 DLPack 张量的主版本号与此头文件中指定的版本号不一致
 * (即 major != DLPACK_MAJOR_VERSION)，消费者必须调用删除器函数
 * (并且这样做是安全的)。不能访问其他字段，因为内存布局已经发生了改变。
 *
 * 在次版本号不匹配的情况下，只要消费者知道如何解释所有字段，张量就可以安全使用。
 * 次版本号的更新表示添加了枚举值。
 */
typedef struct {
  /*! \brief DLPack 的主版本号 */
  uint32_t major;
  /*! \brief DLPack 的次版本号 */
  uint32_t minor;
} DLPackVersion;

/*!
 * \brief DLDevice 中的设备类型枚举
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
/*!
 * \brief CPU device
 */
kDLCPU = 1,
/*!
 * \brief CUDA GPU device
 */
kDLCUDA = 2,
/*!
 * \brief Pinned CUDA CPU memory by cudaMallocHost
 */
kDLCUDAHost = 3,
/*!
 * \brief OpenCL devices.
 */
kDLOpenCL = 4,
/*!
 * \brief Vulkan buffer for next generation graphics.
 */
kDLVulkan = 7,
/*!
 * \brief Metal for Apple GPU.
 */
kDLMetal = 8,
/*!
 * \brief Verilog simulator buffer
 */
kDLVPI = 9,
/*!
 * \brief ROCm GPUs for AMD GPUs
 */
kDLROCM = 10,
/*!
 * \brief Pinned ROCm CPU memory allocated by hipMallocHost
 */
kDLROCMHost = 11,
/*!
 * \brief Reserved extension device type,
 *        used for quickly test extension device
 *        The semantics can differ depending on the implementation.
 */
kDLExtDev = 12,
/*!
 * \brief CUDA managed/unified memory allocated by cudaMallocManaged
 */
kDLCUDAManaged = 13,
/*!
 * \brief Unified shared memory allocated on a oneAPI non-partitioned
 *        device. Call to oneAPI runtime is required to determine the device
 *        type, the USM allocation type and the sycl context it is bound to.
 */
kDLOneAPI = 14,
/*!
 * \brief GPU support for next generation WebGPU standard.
 */
kDLWebGPU = 15,
/*!
 * \brief Qualcomm Hexagon DSP
 */
kDLHexagon = 16,
/*!
 * \brief Microsoft MAIA devices
 */
kDLMAIA = 17,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*!
   * \brief The device type used in the device.
   */
  DLDeviceType device_type;
  /*!
   * \brief The device index.
   *        For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
   */
  int32_t device_id;
} DLDevice;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  /*!
   * \brief signed integer
   */
  kDLInt = 0U,
  /*!
   * \brief unsigned integer
   */
  kDLUInt = 1U,
  /*!
   * \brief IEEE floating point
   */
  kDLFloat = 2U,
  /*!
   * \brief Opaque handle type, reserved for testing purposes.
   *        Frameworks need to agree on the handle data type for the exchange to be well-defined.
   */
  kDLOpaqueHandle = 3U,
  /*!
   * \brief bfloat16
   */
  kDLBfloat = 4U,
  /*!
   * \brief complex number
   *        (C/C++/Python layout: compact struct per complex number)
   */
  kDLComplex = 5U,
  /*!
   * \brief boolean
   */
  kDLBool = 6U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 *        native endian-ness. An explicit error message should be raised when attempting to
 *        export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 */
/*!
 * \brief 定义了描述张量数据类型的结构体。
 *        包含了数据类型的基本信息：代码、位数和张量的通道数。
 */
typedef struct {
  /*!
   * \brief 基本类型的类型代码。
   *        使用 uint8_t 类型而非 DLDataTypeCode 是为了减小内存占用，
   *        但其值应为 DLDataTypeCode 枚举值之一。
   */
  uint8_t code;
  /*!
   * \brief 位数，常见选择为 8、16、32。
   */
  uint8_t bits;
  /*! \brief 张量的通道数，用于向量类型。 */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief 简单的 C 张量对象，不管理内存。
 */
typedef struct {
  /*!
   * \brief 数据指针指向已分配的数据。在 CUDA 设备上，这将是设备指针或 OpenCL 中的 cl_mem 句柄。
   *        在某些设备类型上可能是不透明的。此指针始终按照 CUDA 的标准对齐到 256 字节。
   *        `byte_offset` 字段应用于指向数据的起始位置。
   *
   *        注意：截至 2021 年 11 月，多个库（如 CuPy、PyTorch、TensorFlow、TVM 等）在 CPU/CUDA/ROCm 上
   *        不遵循这种 256 字节对齐的要求，并始终使用 `byte_offset=0`。这必须修复
   *        （修复后将更新此注释）；目前建议不依赖于数据指针的正确对齐。
   *
   *        对于给定的 DLTensor，存储数据所需的内存大小计算如下：
   *
   *        \code{.c}
   *        static inline size_t GetDataSize(const DLTensor* t) {
   *          size_t size = 1;
   *          for (tvm_index_t i = 0; i < t->ndim; ++i) {
   *            size *= t->shape[i];
   *          }
   *          size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *          return size;
   *        }
   *        \endcode
   *
   *        注意，如果张量的大小为零，则数据指针应设置为 `NULL`。
   */
  void* data;
  /*! \brief 张量所在设备 */
  DLDevice device;
  /*! \brief 张量的维数 */
  int32_t ndim;
  /*! \brief 指向数据类型的指针 */
  DLDataType dtype;
  /*! \brief 张量的形状 */
  int64_t* shape;
  /*!
   * \brief 张量的步长（以元素数而非字节为单位）
   *        可以为 NULL，表示张量是紧凑且按行主序排列。
   */
  int64_t* strides;
  /*! \brief 指向数据起始指针的字节偏移量 */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C 张量对象，管理 DLTensor 的内存。
 *        此数据结构旨在通过另一个框架借用 DLTensor。它不用于传输张量。
 *        当借用框架不再需要张量时，应调用 deleter 通知主机不再需要该资源。
 *
 * \note 此数据结构在 DLPack 交换中被用作 Legacy DLManagedTensor，并在 DLPack v0.8 后已弃用。
 *       推荐使用 DLManagedTensorVersioned 替代。
 *       此数据结构在未来版本中可能会被重命名或删除。
 *
 * \sa DLManagedTensorVersioned
 */
typedef struct DLManagedTensor {
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
  /*! \brief the context of the original host framework of DLManagedTensor in
   *   which DLManagedTensor is used in the framework. It can also be NULL.
   */
  void * manager_ctx;
  /*!
   * \brief Destructor - this should be called
   * to destruct the manager_ctx which backs the DLManagedTensor. It can be
   * NULL if there is no way for the caller to provide a reasonable destructor.
   * The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensor * self);
} DLManagedTensor;


// bit masks used in in the DLManagedTensorVersioned

/*! \brief bit mask to indicate that the tensor is read only. */
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)

/*!
 * \brief bit mask to indicate that the tensor is a copy made by the producer.
 *
 * If set, the tensor is considered solely owned throughout its lifetime by the
 * consumer, until the producer-provided deleter is invoked.
 */
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)


/*!
 * \brief A versioned and managed C Tensor object, manage memory of DLTensor.
 *
 * This data structure is intended to facilitate the borrowing of DLTensor by
 * another framework. It is not meant to transfer the tensor. When the borrowing
 * framework doesn't need the tensor, it should call the deleter to notify the
 * host that the resource is no longer needed.
 *
 * \note This is the current standard DLPack exchange data structure.
 */
typedef struct DLManagedTensorVersioned {
  /*!
   * \brief The API and ABI version of the current managed Tensor
   */
  DLPackVersion version;
  /*!
   * \brief the context of the original host framework.
   *
   * Stores DLManagedTensorVersioned is used in the
   * framework. It can also be NULL.
   */
  void *manager_ctx;
  /*!
   * \brief Destructor.
   *
   * This should be called to destruct manager_ctx which holds the DLManagedTensorVersioned.
   * It can be NULL if there is no way for the caller to provide a reasonable
   * destructor. The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensorVersioned *self);
  /*!
   * \brief Additional bitmask flags information about the tensor.
   *
   * By default the flags should be set to 0.
   *
   * \note Future ABI changes should keep everything until this field
   *       stable, to ensure that deleter can be correctly called.
   *
   * \sa DLPACK_FLAG_BITMASK_READ_ONLY
   * \sa DLPACK_FLAG_BITMASK_IS_COPIED
   */
  uint64_t flags;
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
} DLManagedTensorVersioned;


#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif  // DLPACK_DLPACK_H_
```