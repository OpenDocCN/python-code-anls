# `.\pytorch\aten\src\ATen\cudnn\Descriptors.cpp`

```py
// 引入ATen库中的cudnn描述符头文件
#include <ATen/cudnn/Descriptors.h>

// 引入ATen库和c10实用工具的头文件
#include <ATen/ATen.h>
#include <c10/util/irange.h>

// 引入标准输入输出流库和字符串流库的头文件
#include <iostream>
#include <sstream>

// 在at命名空间内定义native命名空间
namespace at { namespace native {

// 匿名命名空间，用于定义内部使用的函数和变量
namespace {

// 内联函数，根据Tensor的数据类型返回对应的cudnn数据类型
inline cudnnDataType_t getDataType(const at::Tensor& t) {
  // 获取Tensor的标量类型
  auto scalar_type = t.scalar_type();
  // 根据标量类型返回对应的cudnn数据类型
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  }
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
  else if (scalar_type == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (scalar_type == at::kQInt8) {
    return CUDNN_DATA_INT8;
  }
#endif
  // 抛出运行时错误，指出TensorDescriptor仅支持double、float和half类型的Tensor
  throw std::runtime_error("TensorDescriptor only supports double, float and half tensors");
}

} // 匿名命名空间结束

// 设置RNNDataDescriptor的成员函数，用于设置cudnnRNN数据描述符
void RNNDataDescriptor::set(const at::Tensor &t, const cudnnRNNDataLayout_t layout, const int maxSeqLength, const int batchSize, const int vectorSize, const int* seqLengthArray) {
  // 根据Tensor的数据类型设置RNNDataDescriptor
  set(getDataType(t), layout, maxSeqLength, batchSize, vectorSize, seqLengthArray);
}

// 设置TensorDescriptor的成员函数，用于设置cudnnTensor描述符
void TensorDescriptor::set(const at::Tensor &t, at::MemoryFormat memory_format, size_t pad) {
  // 根据Tensor的数据类型和内存格式设置TensorDescriptor
  set(getDataType(t), t.sizes(), t.strides(), pad,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

// 设置TensorDescriptor的成员函数，用于设置cudnnTensor描述符
void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  // 推荐Tensor的内存格式并设置TensorDescriptor
  auto memory_format = t.suggest_memory_format();
  set(getDataType(t), t.sizes(), t.strides(), pad,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

// 设置TensorDescriptor的成员函数，用于设置cudnnTensor描述符
void TensorDescriptor::set(cudnnDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  // 设置TensorDescriptor，根据Tensor的数据类型、大小、步长和填充
  set(datatype, t_sizes, t_strides, pad,
    is_channels_last_strides_2d(t_sizes, t_strides) ||
    is_channels_last_strides_3d(t_sizes, t_strides));
}

// 设置TensorDescriptor的成员函数，用于设置cudnnTensor描述符
void TensorDescriptor::set(cudnnDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad, bool nhwc) {
  // 确定Tensor的维度数，并设置TensorDescriptor的大小和步长
  size_t dim = t_sizes.size();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  int size[CUDNN_DIM_MAX];
  int stride[CUDNN_DIM_MAX];
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
    stride[i] = 1;
  }
  // 设置TensorDescriptor，包括数据类型、维度、大小、步长和是否为NHWC格式
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride, nhwc);
}

// 将cudnn数据类型转换为对应的字符串表示
std::string cudnnTypeToString(cudnnDataType_t dtype) {
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      return "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE:
      return "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF:
      return "CUDNN_DATA_HALF";
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
    case CUDNN_DATA_BFLOAT16:
      return "CUDNN_DATA_BFLOAT16";
#endif
    // 如果未知的cudnn数据类型，抛出异常
    default:
      throw std::runtime_error("Unknown cudnnDataType_t");
  }
}
    # 如果数据类型为 CUDNN_DATA_INT8，则返回字符串 "CUDNN_DATA_INT8"
    case CUDNN_DATA_INT8:
      return "CUDNN_DATA_INT8";
    
    # 如果数据类型为 CUDNN_DATA_INT32，则返回字符串 "CUDNN_DATA_INT32"
    case CUDNN_DATA_INT32:
      return "CUDNN_DATA_INT32";
    
    # 如果数据类型为 CUDNN_DATA_INT8x4，则返回字符串 "CUDNN_DATA_INT8x4"
    case CUDNN_DATA_INT8x4:
      return "CUDNN_DATA_INT8x4";
#if CUDNN_VERSION >= 7100
    // 如果 cuDNN 版本大于等于 7100，则执行以下条件分支
    case CUDNN_DATA_UINT8:
      // 当数据类型为 CUDNN_DATA_UINT8 时返回对应的字符串
      return "CUDNN_DATA_UINT8";
    case CUDNN_DATA_UINT8x4:
      // 当数据类型为 CUDNN_DATA_UINT8x4 时返回对应的字符串
      return "CUDNN_DATA_UINT8x4";
#endif
    // 默认情况下，当未知数据类型时，使用字符串流输出未知数据类型的整数表示
    default:
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

// 重载运算符 << ，用于将 TensorDescriptor 对象输出到流中
std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  // 输出 TensorDescriptor 的描述信息及其地址
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  int strideA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  // 获取 Tensor 的维度描述信息
  cudnnGetTensorNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &nbDims, dimA, strideA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // 仅输出维度数组中有效维度的信息
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  // 输出步幅数组中有效维度的信息
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

// 打印 TensorDescriptor 对象的信息
void TensorDescriptor::print() { std::cout << *this; }

// 设置 FilterDescriptor 对象的信息，包括维度、尺寸及内存格式等
void FilterDescriptor::set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad) {
  auto dim = t.ndimension();
  // 如果维度或者填充超过了 CUDNN 支持的最大维度，抛出异常
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  // 注意：虽然此测试可能不足够，因为传递给设置滤波器描述符的张量可能不是传递给 cuDNN 的数据指针的实际张量。
  // 但是，这是常见情况，因此我们可以使用此测试捕获大多数客户端错误。
  TORCH_CHECK(t.is_contiguous(memory_format),
    // cuDNN 的滤波器（也称为权重）必须在所需的内存格式下是连续的
    "cuDNN filters (a.k.a. weights) must be contiguous in desired memory_format\n",
    "Weight sizes: ", t.sizes(), "\n",
    "Weight strides: ", t.strides(), "\n",
    "cuDNN suggested memory_format: ", memory_format);

  int size[CUDNN_DIM_MAX];
  // 获取张量的每个维度的尺寸，并在需要时进行填充
  for (const auto i : c10::irange(dim)) {
    size[i] = (int) t.size(i);
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = (int) 1;
  }
  // 确定维度的最大值
  dim = std::max(dim, pad);
  cudnnTensorFormat_t filter_format;
  // 根据内存格式设置滤波器的张量格式
  switch(memory_format) {
    case at::MemoryFormat::Contiguous:
      filter_format = CUDNN_TENSOR_NCHW;
      break;
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d:
      filter_format = CUDNN_TENSOR_NHWC;
      break;
    default:
      // 不支持的 cuDNN 滤波器的内存格式
      TORCH_INTERNAL_ASSERT(false, "unsupported memory_format for cuDNN filters");
  }
  // 设置滤波器描述符的信息
  set(getDataType(t), (int) dim, size, filter_format);
}

// 将 cudnnTensorFormat_t 类型的内存格式转换为相应的字符串表示
std::string cudnnMemoryFormatToString(cudnnTensorFormat_t tformat) {
  switch (tformat) {
    case CUDNN_TENSOR_NCHW:
      return "CUDNN_TENSOR_NCHW";
    case CUDNN_TENSOR_NHWC:
      return "CUDNN_TENSOR_NHWC";
    default:
      // 如果没有匹配到任何已知的 cuDNN 张量格式，执行以下操作
      std::ostringstream oss;  // 创建一个字符串流对象 oss
      oss << "(unknown cudnn tensor format " << static_cast<int>(tformat) << ")";  // 将未知的 cuDNN 张量格式信息添加到 oss 中
      return oss.str();  // 返回 oss 中的字符串表示
  }
}

# 重载运算符<<，用于将FilterDescriptor对象输出到输出流中
std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d) {
  # 输出FilterDescriptor对象的描述信息和其地址
  out << "FilterDescriptor " << static_cast<void*>(d.desc()) << "\n";
  # 定义存储维度信息的变量和数据类型
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  cudnnTensorFormat_t tformat;
  # 获取FilterDescriptor对象描述符的详细信息，包括数据类型、张量格式和维度数
  cudnnGetFilterNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &tformat, &nbDims, dimA);
  # 输出数据类型
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  # 输出张量格式
  out << "    tensor_format = " << cudnnMemoryFormatToString(tformat) << "\n";
  # 输出维度数
  out << "    nbDims = " << nbDims << "\n";
  # 输出维度数组的部分内容，仅输出nbDims个元素
  # 使用ArrayRef<int>封装dimA数组，遍历输出其中的元素
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  # 返回输出流对象
  return out;
}

# 打印FilterDescriptor对象的详细信息
void FilterDescriptor::print() { std::cout << *this; }

# 结束FilterDescriptor类和命名空间
}} 
```