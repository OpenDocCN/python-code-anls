# `.\pytorch\aten\src\ATen\miopen\Descriptors.cpp`

```py
// 引入 MIOpen 的描述符头文件和 ATen 的基础头文件
#include <ATen/miopen/Descriptors.h>
#include <ATen/ATen.h>
// 引入 C++ 标准库中的 range 函数
#include <c10/util/irange.h>

// 引入输入输出流库
#include <iostream>

// ATen 命名空间
namespace at { namespace native {

// 匿名命名空间，用于隐藏局部函数和变量
namespace {

// 获取张量数据类型并转换为 MIOpen 支持的数据类型
inline miopenDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return miopenFloat;
  } else if (scalar_type == at::kHalf) {
    return miopenHalf;
  } else if (scalar_type == at::kBFloat16) {
    return miopenBFloat16;
  } else {
    // 如果张量类型不支持，则抛出运行时异常
    throw std::runtime_error("TensorDescriptor only supports float, half and bfloat16 tensors");
  }
}

} // anonymous namespace

// 设置张量描述符，使用张量的数据类型和填充值
void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  // 调用重载的 set 函数，传入张量的数据类型、大小、步长和填充值
  set(getDataType(t), t.sizes(), t.strides(), pad);
}

// MIOpen 支持的最大维度
constexpr size_t MIOPEN_DIM_MAX = 5;

// 设置张量描述符，使用给定的数据类型、大小、步长和填充值
void TensorDescriptor::set(miopenDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  // 获取张量的维度
  size_t dim = t_sizes.size();
  // 如果维度或填充值超过 MIOpen 支持的最大值，抛出运行时异常
  if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  // 定义大小和步长数组
  int size[MIOPEN_DIM_MAX];
  int stride[MIOPEN_DIM_MAX];
  // 遍历张量的维度，将大小和步长转换为整数
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  // 遍历维度和填充值之间的区域，设置默认的大小和步长
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
    stride[i] = 1;
  }
  // 调用重载的 set 函数，传入数据类型、最大维度、大小数组和步长数组
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride);
}

// 将 MIOpen 的数据类型转换为字符串形式
std::string miopenTypeToString(miopenDataType_t dtype) {
  switch (dtype) {
    case miopenFloat:
      return "miopenFloat";
    case miopenHalf:
      return "miopenHalf";
    case miopenBFloat16:
      return "miopenBFloat16";
    default:
      // 如果数据类型未知，则返回其整数值的字符串表示
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

// 重载输出流操作符，用于将张量描述符的信息输出到流中
std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  // 输出张量描述符的地址
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims = 4; // 设置默认的维度数
  int dimA[MIOPEN_DIM_MAX]; // 定义大小数组
  int strideA[MIOPEN_DIM_MAX]; // 定义步长数组
  miopenDataType_t dtype; // 定义数据类型
  // 调用 MIOpen 函数获取张量描述符的数据类型、大小和步长
  miopenGetTensorDescriptor(d.desc(), &dtype, dimA, strideA);
  out << "    type = " << miopenTypeToString(dtype) << "\n"; // 输出数据类型的字符串形式
  out << "    nbDims = " << nbDims << "\n"; // 输出维度数
  // 输出大小数组中的前 nbDims 个元素
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  // 输出步长数组中的前 nbDims 个元素
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

// 打印张量描述符的信息到标准输出流
void TensorDescriptor::print() {
  std::cout << *this;
}

// 设置过滤器描述符，使用张量、内存格式和填充值
void FilterDescriptor::set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad) {
  // 获取张量的维度
  auto dim = t.ndimension();
  // 如果维度或填充值超过 MIOpen 支持的最大值，抛出运行时异常
  if (dim > static_cast<int64_t>(MIOPEN_DIM_MAX) || pad > static_cast<int64_t>(MIOPEN_DIM_MAX)) {
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  }
  // 检查张量是否以指定的内存格式连续
  TORCH_CHECK(t.is_contiguous(memory_format),
      "MIOpen filters (a.k.a. weights) must be contiguous");

  // 定义存储大小和步长数组
  int size[MIOPEN_DIM_MAX];
  int stride[MIOPEN_DIM_MAX];

  // 遍历张量的维度，初始化存储大小
  for (const auto i : c10::irange(dim)) {
    size[i] = (int) t.size(i);
  }

  // 将填充维度之后的维度大小设置为1
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = (int) 1;
  }

  // 初始化步长数组
  for (int i = pad; i >= dim; --i ) {
      stride[i] = 1;
  }

  // 遍历维度并设置步长，注意：dim是从大到小的顺序
  for (int i = dim-1 ; i >=0; --i ) {
      // 将张量对应维度的步长赋值给步长数组
      stride[i] = t.stride(i);
  }

  // 更新dim为dim和pad中的较大者
  dim = std::max<int64_t>(dim, pad);

  // 调用set函数设置数据类型、维度、大小和步长
  set(getDataType(t), (int) dim, size, stride);
}

}}
```