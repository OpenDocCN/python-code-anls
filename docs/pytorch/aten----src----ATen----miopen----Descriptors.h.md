# `.\pytorch\aten\src\ATen\miopen\Descriptors.h`

```py
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <ATen/miopen/Exceptions.h>
// 引入 MIOpen 库的异常处理头文件

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
// 引入 MIOpen 和 ATen（PyTorch 核心库）相关头文件

namespace at { namespace native {

inline int dataSize(miopenDataType_t dataType)
{
  // 返回数据类型的字节大小
  switch (dataType) {
    case miopenHalf: return 2; // 半精度浮点数
    case miopenFloat: return 4; // 单精度浮点数
    case miopenBFloat16: return 2; // BF16 浮点数
    default: return 8; // 默认为双精度浮点数
  }
}

template <typename T, miopenStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  // 描述符删除器结构体，用于释放描述符资源
  void operator()(T* x) {
    if (x != nullptr) {
      MIOPEN_CHECK(dtor(x)); // 调用 MIOpen 检查并释放资源
    }
  }
};

// 用于包装 MIOpen 描述符类型的通用类模板。只需提供指向底层类型的指针，构造函数和析构函数即可。
// 子类负责定义一个 set() 函数来设置描述符。
template <typename T, miopenStatus_t (*ctor)(T**), miopenStatus_t (*dtor)(T*)>
class Descriptor
{
public:
  // 使用 desc() 以只读方式访问底层描述符指针。大多数客户端代码应该使用这个函数。
  // 如果描述符从未初始化过，将返回 nullptr。
  T* desc() const { return desc_.get(); }

  // 使用 mut_desc() 访问底层描述符指针，如果您打算修改它所指向的内容（例如使用 miopenSetFooDescriptor）。
  // 这将确保描述符已初始化。此文件中的代码将使用这个函数。
  T* mut_desc() { init(); return desc_.get(); }

protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      MIOPEN_CHECK(ctor(&raw_desc)); // 创建描述符并检查错误
      desc_.reset(raw_desc); // 用描述符指针初始化智能指针
    }
  }

private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_; // 描述符的智能指针
};

class TensorDescriptor
  : public Descriptor<miopenTensorDescriptor,
                      &miopenCreateTensorDescriptor,
                      &miopenDestroyTensorDescriptor>
{
public:
  TensorDescriptor() {}
  explicit TensorDescriptor(const at::Tensor &t, size_t pad = 0) {
    set(t, pad); // 设置张量描述符
  }

  void set(const at::Tensor &t, size_t pad = 0); // 设置张量描述符的函数
  void set(miopenDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad = 0); // 设置张量描述符的函数

  void print(); // 打印张量描述符的函数

private:
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride)); // 设置张量描述符
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d); // 张量描述符的输出流操作符重载

class FilterDescriptor
  : public Descriptor<miopenTensorDescriptor,
                      &miopenCreateTensorDescriptor,
                      &miopenDestroyTensorDescriptor>
{
public:
  void set(const at::Tensor &t, int64_t pad = 0) {
    set(t, at::MemoryFormat::Contiguous, pad);
  }

  // 调用 set 函数，设置张量 t 的内存格式为连续，并指定填充值 pad，默认为 0
  void set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad = 0);
private:
  // 设置张量描述符，用于描述张量的数据类型、维度、尺寸和步长
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    // 调用miopenSetTensorDescriptor函数设置张量描述符
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

struct ConvolutionDescriptor
  // 卷积描述符，继承自Descriptor模板类，用于管理miopenConvolutionDescriptor
  : public Descriptor<miopenConvolutionDescriptor,
                      &miopenCreateConvolutionDescriptor,
                      &miopenDestroyConvolutionDescriptor>
{
  // 设置卷积描述符，包括数据类型、卷积模式、维度、填充、步长、扩展率（即膨胀率）、分组、确定性标志
  void set(miopenDataType_t dataType, miopenConvolutionMode_t c_mode,  int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups, bool deterministic) {
    // 初始化卷积描述符
    MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale, c_mode));
    // 设置卷积分组数
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(mut_desc(), groups));
    // 设置卷积属性：确定性
    MIOPEN_CHECK(miopenSetConvolutionAttribute(mut_desc(), MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, deterministic ? 1 : 0));
  }
};

struct RNNDescriptor
  // 循环神经网络描述符，继承自Descriptor模板类，用于管理miopenRNNDescriptor
  : public Descriptor<miopenRNNDescriptor,
                      &miopenCreateRNNDescriptor,
                      &miopenDestroyRNNDescriptor>
{
  // 设置循环神经网络描述符，包括隐藏层大小、层数、输入模式、方向、RNN模式、偏置模式、算法、数据类型
  void set(int64_t hidden_size, int64_t num_layers, miopenRNNInputMode_t input_mode, miopenRNNDirectionMode_t direction, miopenRNNMode_t rnn_mode,
              miopenRNNBiasMode_t bias_mode, miopenRNNAlgo_t algorithm, miopenDataType_t datatype) {
    // 调用miopenSetRNNDescriptor函数设置RNN描述符
    MIOPEN_CHECK(miopenSetRNNDescriptor(mut_desc(), hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algorithm, datatype));
  }
};

union Constant
{
  float f;  // 单精度浮点数
  double d; // 双精度浮点数
  Constant(miopenDataType_t dataType, double value) {
    // 根据数据类型选择性地将value转换为float或double，并赋值给联合体的成员
    if (dataType == miopenHalf || dataType == miopenFloat || dataType == miopenBFloat16) {
      f = static_cast<float>(value); // 转换为float
    } else {
      d = value; // 保持为double
    }
  }
}

}}  // namespace
```