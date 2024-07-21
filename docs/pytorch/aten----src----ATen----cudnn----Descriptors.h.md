# `.\pytorch\aten\src\ATen\cudnn\Descriptors.h`

```py
#pragma once

// 使用 `#pragma once` 防止头文件被多次包含，保证头文件只被编译一次。


#include <string>

// 包含 `<string>` 头文件，用于处理 C++ 中的字符串操作。


#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <cuda.h>

// 包含一系列 ATen 和 CUDA 相关的头文件，用于操作张量、CUDA 上下文、CuDNN 等功能。


#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

// 条件编译：根据宏 `AT_PER_OPERATOR_HEADERS` 的定义选择包含 `<ATen/Functions.h>` 或 `<ATen/ops/empty.h>`。


#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8907
#define USE_CUDNN_RNN_V8_API
#endif

// 如果 `CUDNN_VERSION` 大于等于 8907，则定义宏 `USE_CUDNN_RNN_V8_API`，用于启用 CuDNN RNN V8 API。


namespace at { namespace native {

// 进入 `at::native` 命名空间，定义下面的函数和类在这个命名空间内。


std::string cudnnTypeToString(cudnnDataType_t dtype);

// 声明一个函数 `cudnnTypeToString`，用于将 CuDNN 的数据类型 `cudnnDataType_t` 转换为字符串。


// TODO: Add constructors for all of the descriptors

// TODO 注释：需要为所有描述符添加构造函数。


inline int dataSize(cudnnDataType_t dataType)
{
  switch (dataType) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
    case CUDNN_DATA_BFLOAT16:
#endif
    case CUDNN_DATA_HALF: return 2;
    case CUDNN_DATA_FLOAT: return 4;
    default: return 8;
  }
}

// 定义一个内联函数 `dataSize`，根据给定的 `dataType` 返回数据类型的字节大小。


// The stride for a size-1 dimensions is not uniquely determined; in
// fact, it can be anything you want, because the fact that the
// tensor is size 1 at this dimension means that you will never actually
// try advancing your pointer by this stride.
//
// However, CuDNN has a much more stringent requirement on strides:
// if you are passing a contiguous input, it better be the case
// that the stride for dim i is the product of the sizes of dims
// i+1 to the end.  This stride is indeed uniquely determined.  This
// function modifies 'stride' in place so this invariant holds.
template <typename T>
static inline void fixSizeOneDimStride(int dim, const T *size, T *stride, bool nhwc) {
  int64_t z = 1;
  int index = 0;
  std::vector<int> permutation(dim);

  if (nhwc) {
    permutation[index++] = 1;
  }
  for (int d = dim-1; d > 1; d--) {
    permutation[index++] = d;
  }
  if (!nhwc) {
    permutation[index++] = 1;
  }
  permutation[index++] = 0;
  for (int d : permutation) {
    if (size[d] == 1) {
      stride[d] = z;
    } else {
      z *= size[d];
    }
  }
}

// 定义一个模板函数 `fixSizeOneDimStride`，用于调整张量的步长（stride），以满足 CuDNN 对步长的要求。


template <typename T, cudnnStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      AT_CUDNN_CHECK(dtor(x));
    }
  }
};

// 定义一个模板结构体 `DescriptorDeleter`，用于管理 CuDNN 描述符的销毁。


// A generic class for wrapping cuDNN descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's cudnnTensorDescriptor_t it points to cudnnTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, cudnnStatus_t (*ctor)(T**), cudnnStatus_t (*dtor)(T*)>

// 注释：定义了一个通用类模板，用于封装 CuDNN 的描述符类型，需要提供指向描述符结构体的指针、构造函数和析构函数，子类负责定义 `set()` 函数来设置描述符。
class TORCH_CUDA_CPP_API Descriptor {
 public:
  // TODO: Figure out why const-correctness doesn't work here

  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  // 使用 desc() 方法以只读方式访问底层描述符指针。
  // 大多数客户端代码应该使用这个方法。
  // 如果描述符从未初始化，则返回 nullptr。
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

  // Use mut_desc() to access the underlying descriptor pointer
  // if you intend to modify what it points to (e.g., using
  // cudnnSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  // 使用 mut_desc() 方法访问底层描述符指针，如果你打算修改它所指向的内容（例如使用 cudnnSetFooDescriptor）。
  // 这将确保描述符已初始化。此文件中的代码将使用此函数。
  T* mut_desc() { init(); return desc_.get(); }
protected:
  // Initialize the descriptor if it has not been initialized already.
  // 如果描述符尚未初始化，则进行初始化。
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      AT_CUDNN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TORCH_CUDA_CPP_API RNNDataDescriptor : public Descriptor<
                                       cudnnRNNDataStruct,
                                       &cudnnCreateRNNDataDescriptor,
                                       &cudnnDestroyRNNDataDescriptor> {
public:
  // Set the RNN data descriptor using given parameters.
  // 使用给定的参数设置 RNN 数据描述符。
  void set(const at::Tensor &t, cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize, int vectorSize, const int* seqLengthArray);
private:
  // Internal method to set the RNN data descriptor with specific parameters.
  // 使用特定参数设置 RNN 数据描述符的内部方法。
  void set(cudnnDataType_t dataType, cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize, int vectorSize, const int* seqLengthArray) {
    AT_CUDNN_CHECK(cudnnSetRNNDataDescriptor(mut_desc(), dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, NULL));
  }
};

class TORCH_CUDA_CPP_API TensorDescriptor : public Descriptor<
                                               cudnnTensorStruct,
                                               &cudnnCreateTensorDescriptor,
                                               &cudnnDestroyTensorDescriptor> {
 public:
  // Default constructor for TensorDescriptor.
  // TensorDescriptor 的默认构造函数。
  TensorDescriptor() = default;
  
  // Constructor for TensorDescriptor using a given Tensor and optional padding.
  // 使用给定的 Tensor 和可选的填充大小构造 TensorDescriptor。
  explicit TensorDescriptor(const at::Tensor &t, size_t pad = 0) {
    // 调用 set 函数，传入参数 t 和 pad
    set(t, pad);
    }
    
    // Note [CuDNN broadcast padding]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // pad specifies the minimum dimensionality of the tensor descriptor
    // we produce (it doesn't have anything to do with, e.g., convolution
    // padding).  If 't' is lower-dimensional than 'pad', the remaining
    // dimensions (on the right) are padded with ones.  This doesn't
    // affect the underlying data layout.  This is particularly useful for
    // dealing with a peculiarity of the CuDNN API, which is that broadcasting in CuDNN is
    // done in two steps: first, the client code is expected to pad out
    // (the dimensions) input tensors to be the same dimension as the
    // target broadcast, and then second, CuDNN takes of actually
    // broadcasting size 1 dimensions.
    
    // set 函数的声明，用于设置张量的相关参数和内存格式
    void set(const at::Tensor &t, size_t pad = 0);
    // set 函数的声明，允许设置内存格式的张量参数及其填充
    void set(const at::Tensor &t, at::MemoryFormat memory_format, size_t pad = 0);
    // set 函数的声明，用于设置 CuDNN 数据类型、大小、步幅及填充
    void set(cudnnDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad = 0);
    
    // 打印函数的声明
    void print();
private:
  // 设置张量描述符的私有方法，指定数据类型、尺寸、步长、填充及是否使用 NHWC 格式
  void set(cudnnDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad, bool nhwc);

  // 设置张量描述符的私有方法，指定数据类型、维度、大小、步长及是否使用 NHWC 格式
  void set(cudnnDataType_t dataType, int dim, int* size, int* stride, bool nhwc) {
    // 复制步长数组，以便后续修正大小为1的维度步长
    std::vector<int> strides_copy(stride, stride + dim);
    // 修正大小为1的维度的步长
    fixSizeOneDimStride<int>(dim, size, strides_copy.data(), nhwc);
    // 使用 CUDNN 函数设置张量描述符
    AT_CUDNN_CHECK(cudnnSetTensorNdDescriptor(mut_desc(), dataType, dim, size, strides_copy.data()));
  }
};

// 输出流操作符重载，用于打印张量描述符信息
std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class TORCH_CUDA_CPP_API FilterDescriptor : public Descriptor<
                                               cudnnFilterStruct,
                                               &cudnnCreateFilterDescriptor,
                                               &cudnnDestroyFilterDescriptor> {
 public:
  // 设置滤波器描述符，使用默认填充值为0
  void set(const at::Tensor &t, int64_t pad = 0) {
    set(t, at::MemoryFormat::Contiguous, pad);
  }

  // 设置滤波器描述符，指定张量及内存格式，使用指定填充值
  void set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad = 0);

  // 打印滤波器描述符信息
  void print();
private:
  // 设置滤波器描述符的私有方法，指定数据类型、维度、大小及滤波器格式
  void set(cudnnDataType_t dataType, int dim, int* size, cudnnTensorFormat_t filter_format) {
    // 使用 CUDNN 函数设置滤波器描述符
    AT_CUDNN_CHECK(cudnnSetFilterNdDescriptor(mut_desc(), dataType, filter_format, dim, size));
  }
};

// 输出流操作符重载，用于打印滤波器描述符信息
std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d);

struct TORCH_CUDA_CPP_API ConvolutionDescriptor
    : public Descriptor<
          cudnnConvolutionStruct,
          &cudnnCreateConvolutionDescriptor,
          &cudnnDestroyConvolutionDescriptor> {
  // 设置卷积描述符，指定数据类型、维度、填充、步长、扩展（即膨胀）、组数及是否允许 TF32 模式
  void set(cudnnDataType_t dataType, int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups, bool allow_tf32) {
    // 确定数学计算类型，默认使用与数据类型对应的数学类型
    cudnnDataType_t mathType = dataType;
    if (dataType == CUDNN_DATA_HALF) mathType = CUDNN_DATA_FLOAT;
    // 使用 CUDNN 函数设置卷积描述符
    AT_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale,
                                          CUDNN_CROSS_CORRELATION, mathType));
    // 设置卷积组数
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(mut_desc(), groups));
    // 设置卷积的数学计算类型
    // 详见注释 [behavior of cudnnFind and cudnnGet]
    AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
    if(dataType == CUDNN_DATA_HALF) {
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
    } else if (dataType == CUDNN_DATA_FLOAT && !allow_tf32) {
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_FMA_MATH));
    }
  }
};

struct TORCH_CUDA_CPP_API SpatialTransformerDescriptor
    : public Descriptor<
          cudnnSpatialTransformerStruct,
          &cudnnCreateSpatialTransformerDescriptor,
          &cudnnDestroySpatialTransformerDescriptor> {
  // 设置空间变换器描述符，指定数据类型、维度及大小
  void set(cudnnDataType_t dataType, int dim, int* size) {
    // 使用 CUDNN 函数设置空间变换器描述符
    AT_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(mut_desc(), CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
  }
};

// 结构体，定义了 TORCH_CUDA_CPP_API 接口的 DropoutDescriptor
struct TORCH_CUDA_CPP_API DropoutDescriptor
    // 继承自 Descriptor，指定了 cudnnDropoutStruct 的类型和对应的创建和销毁函数
    : public Descriptor<
          cudnnDropoutStruct,
          &cudnnCreateDropoutDescriptor,
          &cudnnDestroyDropoutDescriptor> {
  // 保存 RNG 状态的 Tensor 对象
  at::Tensor state;

  // 初始化一个 dropout 描述符的 RNG 状态。
  // 警告：这个函数非常耗费资源，尽量避免调用！
  void initialize_rng(cudnnHandle_t handle, float dropout, long long int seed, const TensorOptions& options) {
    // 断言 dropout 必须大于 0，否则调用 set_no_dropout 函数
    TORCH_INTERNAL_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    // 获取 RNG 状态所需的大小
    size_t state_size;
    AT_CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &state_size));
    // 断言 options 的设备类型为 kCUDA
    AT_ASSERT(options.device().type() == kCUDA);
    // 断言 options 的数据类型为 kByte
    AT_ASSERT(options.dtype() == kByte);
    // 创建一个与 state_size 大小相符的空 Tensor 对象，用于保存 RNG 状态
    state = at::empty({static_cast<int64_t>(state_size)}, options);
    // 设置 dropout 描述符的 RNG 状态
    AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, dropout, state.data_ptr(), state_size, seed));
  }

  // 根据给定的 dropout 概率和现有的 RNG 状态恢复一个 dropout 描述符
  void set(cudnnHandle_t handle, float dropout, at::Tensor state_) {
    // 断言 dropout 必须大于 0，否则调用 set_no_dropout 函数
    TORCH_INTERNAL_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    // 将传入的 state_ Tensor 对象赋值给成员变量 state
    state = state_;
    // 获取 state 的数据指针
    void *state_ptr = state.data_ptr();
    // 获取 state 的大小
    size_t state_size = state.size(0);
    // 由于 dropout 为 0 时不需要随机数初始化，seed 参数设置为 0
    // 恢复 dropout 描述符
    AT_CUDNN_CHECK(cudnnRestoreDropoutDescriptor(mut_desc(), handle, dropout, state_ptr, state_size, 0 /* seed */));
  }

  // 恢复一个对应于无 dropout 的 dropout 描述符
  void set_no_dropout(cudnnHandle_t handle) {
    // 当 dropout = 0 时，seed 参数无关紧要，因为不进行随机数初始化
    // 当 dropout = 0 时，cudnnSetDropoutDescriptor 函数开销较小
    AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, 0 /* dropout */, nullptr, 0 /* state_size */, 0 /* seed */));
  }
};

// RNNDescriptor 结构体，继承自 Descriptor 类
struct TORCH_CUDA_CPP_API RNNDescriptor : public Descriptor<
                                             cudnnRNNStruct,
                                             &cudnnCreateRNNDescriptor,
                                             &cudnnDestroyRNNDescriptor> {
  DropoutDescriptor dropout_desc_;  // RNN 描述符中的 dropout 描述符

  // 设置 RNN 描述符参数的函数
  void set(cudnnHandle_t handle,
#ifdef USE_CUDNN_RNN_V8_API
       int input_size,
       bool packed,
#endif
       int hidden_size, int proj_size, int num_layers, DropoutDescriptor&& dropout_desc,
           cudnnRNNInputMode_t input_mode, cudnnDirectionMode_t bidirectional,
           cudnnRNNMode_t mode, cudnnDataType_t datatype, cudnnDataType_t input_type, cudnnRNNAlgo_t algo, bool allow_tf32) {
    dropout_desc_ = std::move(dropout_desc);  // 移动传入的 dropout 描述符

#ifndef USE_CUDNN_RNN_V8_API
    // 使用 cudnnSetRNNDescriptor_v6 设置 RNN 描述符的参数（针对非 V8 API）
    AT_CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
          handle,
          mut_desc(),  // 获取可变描述符的指针
          hidden_size,
          num_layers,
          dropout_desc_.desc(),  // 获取 dropout 描述符的描述
          input_mode,
          bidirectional,
          mode,
          algo,
          datatype));

    // 如果有投影层，则设置 RNN 投影层
    if (proj_size != 0) {
      AT_CUDNN_CHECK(cudnnSetRNNProjectionLayers(
            handle,
            /*rnnDesc=*/mut_desc(),
            /*recProjSize=*/proj_size,
            /*outProjSize=*/0));
    }

    // 获取当前 CUDA 设备的属性
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    // 如果设备主版本号大于等于 7
    if (prop->major >= 7) {
      // 根据输入类型设置 RNN 矩阵计算类型
      if (input_type == CUDNN_DATA_HALF) {
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_TENSOR_OP_MATH);
      }
      else if (input_type == CUDNN_DATA_FLOAT && !allow_tf32) {
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_FMA_MATH);
      }
      else {
        // 默认情况下不需要显式设置
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_DEFAULT_MATH);
      }
    }
#else
    // 对于使用 V8 API 的情况，根据输入类型和设备属性选择数学类型
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    auto math_type = CUDNN_DEFAULT_MATH;
    if (prop->major >= 7) {
      if (input_type == CUDNN_DATA_HALF) {
        math_type = CUDNN_TENSOR_OP_MATH;
      } else if (!allow_tf32) {
        math_type = CUDNN_FMA_MATH;
      }
    }
    // 使用 cudnnSetRNNDescriptor_v8 设置 RNN 描述符的参数
    AT_CUDNN_CHECK(cudnnSetRNNDescriptor_v8(
          mut_desc(),
          algo,
          mode,
          CUDNN_RNN_DOUBLE_BIAS,
          bidirectional,
          input_mode,
          input_type,
          datatype,
          math_type,
          input_size,
          hidden_size,
          proj_size ? proj_size : hidden_size,
          num_layers,
          dropout_desc_.desc(),
          packed ? CUDNN_RNN_PADDED_IO_DISABLED : CUDNN_RNN_PADDED_IO_ENABLED));
#endif
  }
};

// CTCLossDescriptor 结构体，继承自 Descriptor 类
struct TORCH_CUDA_CPP_API CTCLossDescriptor
    : public Descriptor<
          cudnnCTCLossStruct,
          &cudnnCreateCTCLossDescriptor,
          &cudnnDestroyCTCLossDescriptor> {
  // 设置 CTCLoss 描述符参数的函数
  void set(cudnnDataType_t datatype) {
    # 调用 AT_CUDNN_CHECK 函数，设置 CTC 损失描述符，使用当前的描述符和给定的数据类型
    AT_CUDNN_CHECK(cudnnSetCTCLossDescriptor(mut_desc(), datatype));
  }
  # 设置 CTC 损失描述符的扩展版本，指定数据类型、规范化模式和 NaN 传播模式
  void setEx(
      cudnnDataType_t datatype,
      cudnnLossNormalizationMode_t normMode,
      cudnnNanPropagation_t gradMode) {
    # 调用 AT_CUDNN_CHECK 函数，设置 CTC 损失描述符的扩展版本，使用当前描述符和给定的参数
    AT_CUDNN_CHECK(
        cudnnSetCTCLossDescriptorEx(mut_desc(), datatype, normMode, gradMode));
  }
  # 设置 CTC 损失描述符的特定版本，包括数据类型、规范化模式、NaN 传播模式和最大标签长度
  void set_v8_v9(
      cudnnDataType_t datatype,
      cudnnLossNormalizationMode_t normMode,
      cudnnNanPropagation_t gradMode,
      int maxLabelLength) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 90000
    // 如果定义了 CUDNN_VERSION 并且版本号大于等于 90000，则使用 CUDNN_CTC_ZERO_OOB_GRADIENTS 梯度模式
    auto gradModev9 = CUDNN_CTC_ZERO_OOB_GRADIENTS;
    // 如果梯度传播模式为 NaN 传播，则使用 CUDNN_CTC_SKIP_OOB_GRADIENTS 避开超出边界的梯度
    if (gradMode == cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN) {
      gradModev9 = CUDNN_CTC_SKIP_OOB_GRADIENTS;
    }
    // 设置 CTC 损失函数描述符为版本 9，包括数据类型、规范化模式、梯度模式、最大标签长度
    AT_CUDNN_CHECK(
        cudnnSetCTCLossDescriptor_v9(mut_desc(), datatype, normMode, gradModev9, maxLabelLength));
#else
    // 使用版本 8 的 CTC 损失函数描述符，包括数据类型、规范化模式、梯度模式、最大标签长度
    AT_CUDNN_CHECK(
        cudnnSetCTCLossDescriptor_v8(mut_desc(), datatype, normMode, gradMode, maxLabelLength));
#endif
  }
};

// 激活函数描述符类，继承自 cudnnActivationStruct 描述符
struct TORCH_CUDA_CPP_API ActivationDescriptor
    : public Descriptor<
          cudnnActivationStruct,
          &cudnnCreateActivationDescriptor,
          &cudnnDestroyActivationDescriptor> {
  // 设置激活函数模式
  void set(cudnnActivationMode_t mode) {
    // 断言激活函数模式为 CUDNN_ACTIVATION_RELU，暂不支持其它 cuDNN 激活模式
    AT_ASSERT(
        mode == CUDNN_ACTIVATION_RELU,
        "TODO: support more cuDNN activation modes");
    // 设置激活函数描述符，包括模式、NaN 传播模式、最大值限制
    AT_CUDNN_CHECK(cudnnSetActivationDescriptor(
        mut_desc(),
        mode,
        cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        std::numeric_limits<double>::max()));
  }
};

// 常量联合体，用于不同数据类型的常量转换
union Constant
{
  float f;    // 单精度浮点数
  double d;   // 双精度浮点数
  // 构造函数根据数据类型初始化常量值
  Constant(cudnnDataType_t dataType, double value) {
    if (dataType == CUDNN_DATA_HALF || dataType == CUDNN_DATA_FLOAT) {
      f = static_cast<float>(value);   // 如果数据类型为半精度或单精度浮点数，转换为单精度赋值
    } else {
      d = value;    // 否则直接赋值双精度
    }
  }
};

}}  // namespace
```