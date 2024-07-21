# `.\pytorch\aten\src\ATen\core\Formatting.cpp`

```py
// 包含 ATen 核心库中的格式化头文件
#include <ATen/core/Formatting.h>
// 包含 C10 实用工具中的范围遍历头文件
#include <c10/util/irange.h>

// 包含数学函数库
#include <cmath>
// 包含整数类型
#include <cstdint>
// 控制输出流格式的头文件
#include <iomanip>
// 标准输入输出流库
#include <iostream>
// 包含元组类型
#include <tuple>

// ATen 命名空间
namespace c10 {

// 定义流插入运算符重载，用于将 Backend 类型输出到流
std::ostream& operator<<(std::ostream & out, Backend b) {
  return out << toString(b);
}

// 定义流插入运算符重载，用于将 Scalar 类型输出到流
std::ostream& operator<<(std::ostream & out, const Scalar& s) {
  // 根据 Scalar 的类型选择适当的输出方式
  if (s.isFloatingPoint()) {
    return out << s.toDouble();
  }
  if (s.isComplex()) {
    return out << s.toComplexDouble();
  }
  if (s.isBoolean()) {
    return out << (s.toBool() ? "true" : "false");
  }
  if (s.isSymInt()) {
    return out << (s.toSymInt());
  }
  if (s.isSymFloat()) {
    return out << (s.toSymFloat());
  }
  if (s.isIntegral(false)) {
    return out << s.toLong();
  }
  // 抛出逻辑错误，表示 Scalar 类型未知
  throw std::logic_error("Unknown type in Scalar");
}

// 将 Scalar 类型转换为字符串表示
std::string toString(const Scalar& s) {
  std::stringstream out;
  out << s;
  return out.str();
}

} // 结束 ATen 命名空间

// at 命名空间
namespace at {

// 定义流插入运算符重载，用于将 DeprecatedTypeProperties 类型输出到流
std::ostream& operator<<(std::ostream & out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

// 打印 Tensor 对象的格式化信息，并返回指数的最小值和最大值
static std::tuple<double, int> __printFormat(std::ostream& stream, const Tensor& self) {
  auto size = self.numel();
  // 若 Tensor 大小为 0，则返回默认值
  if(size == 0) {
    return std::make_tuple(1., 0);
  }
  bool intMode = true;
  auto self_p = self.const_data_ptr<double>();
  // 检查 Tensor 是否都是整数
  for (const auto i : c10::irange(size)) {
    auto z = self_p[i];
    if(std::isfinite(z)) {
      if(z != std::ceil(z)) {
        intMode = false;
        break;
      }
    }
  }
  int64_t offset = 0;
  // 找到第一个有限的元素位置
  while(!std::isfinite(self_p[offset])) {
    offset = offset + 1;
    if(offset == size) {
      break;
    }
  }
  double expMin = 1;
  double expMax = 1;
  // 如果有限元素存在，则计算最小和最大指数
  if(offset != size) {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for (const auto i : c10::irange(offset, size)) {
      double z = fabs(self_p[i]);
      if(std::isfinite(z)) {
        if(z < expMin) {
          expMin = z;
        }
        if(self_p[i] > expMax) {
          expMax = z;
        }
      }
    }
    // 计算指数的最小值和最大值
    if(expMin != 0) {
      expMin = std::floor(std::log10(expMin)) + 1;
    } else {
      expMin = 1;
    }
    if(expMax != 0) {
      expMax = std::floor(std::log10(expMax)) + 1;
    } else {
      expMax = 1;
    }
  }
  double scale = 1;
  int sz = 11;
  // 根据 intMode 设置流输出格式
  if(intMode) {
    if(expMax > 9) {
      sz = 11;
      stream << std::scientific << std::setprecision(4);
    } else {
      sz = static_cast<int>(expMax) + 1;
      stream << defaultfloat;
    }
  } else {
    // 如果指数范围大于4
    if(expMax-expMin > 4) {
      // 设置默认字符串长度为11
      sz = 11;
      // 如果最大或最小指数的绝对值超过99，增加字符串长度
      if(std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
        sz = sz + 1;
      }
      // 使用科学计数法输出，并设置精度为4位小数
      stream << std::scientific << std::setprecision(4);
    } else {
      // 如果指数范围不大于4
      if(expMax > 5 || expMax < 0) {
        // 设置默认字符串长度为7
        sz = 7;
        // 计算比例尺，按10的(expMax-1)次幂
        scale = std::pow(10, expMax-1);
        // 使用定点表示法输出，并设置精度为4位小数
        stream << std::fixed << std::setprecision(4);
      } else {
        // 如果最大指数为0
        if(expMax == 0) {
          // 设置默认字符串长度为7
          sz = 7;
        } else {
          // 根据最大指数设置字符串长度
          sz = static_cast<int>(expMax) + 6;
        }
        // 使用定点表示法输出，并设置精度为4位小数
        stream << std::fixed << std::setprecision(4);
      }
    }
  }
  // 返回比例尺和字符串长度的元组
  return std::make_tuple(scale, sz);
}

// 打印指定数量的空格，用于缩进
static void __printIndent(std::ostream &stream, int64_t indent)
{
  // 使用循环输出指定数量的空格到流中
  for (C10_UNUSED const auto i : c10::irange(indent)) {
    stream << " ";
  }
}

// 打印缩放比例及其标识符到流中
static void printScale(std::ostream & stream, double scale) {
  FormatGuard guard(stream);
  stream << defaultfloat << scale << " *" << '\n';
}

// 打印张量的矩阵内容到流中，限制每行的列数和缩进量
static void __printMatrix(std::ostream& stream, const Tensor& self, int64_t linesize, int64_t indent)
{
  // 获取打印格式的比例和大小
  auto [scale, sz] = __printFormat(stream, self);

  // 打印指定缩进量的空格
  __printIndent(stream, indent);

  // 计算每行可包含的列数
  int64_t nColumnPerLine = (linesize-indent)/(sz+1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;

  // 分段打印张量的列
  while(firstColumn < self.size(1)) {
    // 确定当前段的最后一列
    if(firstColumn + nColumnPerLine <= self.size(1)) {
      lastColumn = firstColumn + nColumnPerLine - 1;
    } else {
      lastColumn = self.size(1) - 1;
    }

    // 如果不是整个张量的一行，则换行并打印列范围
    if(nColumnPerLine < self.size(1)) {
      if(firstColumn != 0) {
        stream << '\n';
      }
      stream << "Columns " << firstColumn+1 << " to " << lastColumn+1;
      __printIndent(stream, indent);
    }

    // 如果存在缩放比例，则打印比例标识
    if(scale != 1) {
      printScale(stream,scale);
      __printIndent(stream, indent);
    }

    // 逐行打印矩阵的内容
    for (const auto l : c10::irange(self.size(0))) {
      Tensor row = self.select(0,l);
      const double *row_ptr = row.const_data_ptr<double>();
      for (const auto c : c10::irange(firstColumn, lastColumn+1)) {
        stream << std::setw(sz) << row_ptr[c]/scale;
        if(c == lastColumn) {
          stream << '\n';
          if(l != self.size(0)-1) {
            if(scale != 1) {
              __printIndent(stream, indent);
              stream << " ";
            } else {
              __printIndent(stream, indent);
            }
          }
        } else {
          stream << " ";
        }
      }
    }

    // 更新下一段的起始列
    firstColumn = lastColumn + 1;
  }
}

// 递归打印张量的各个分片到流中
static void __printTensor(std::ostream& stream, Tensor& self, int64_t linesize)
{
  // 初始化分片计数器和结束标志
  std::vector<int64_t> counter(self.ndimension()-2);
  bool start = true;
  bool finished = false;
  counter[0] = -1;
  for (const auto i : c10::irange(1, counter.size())) {
    counter[i] = 0;
  }

  // 循环打印张量的各个分片
  while(true) {
    for(int64_t i = 0; self.ndimension()-2; i++) {
      counter[i] = counter[i] + 1;
      if(counter[i] >= self.size(i)) {
        if(i == self.ndimension()-3) {
          finished = true;
          break;
        }
        counter[i] = 0;
      } else {
        break;
      }
    }
    if(finished) {
      break;
    }

    // 根据需要输出起始括号
    if(start) {
      start = false;
    } else {
      stream << '\n';
    }
    stream << "(";

    // 获取当前分片并输出其索引信息
    Tensor tensor = self;
    for (const auto i : c10::irange(self.ndimension()-2)) {
      tensor = tensor.select(0, counter[i]);
      stream << counter[i]+1 << ",";
    }
    stream << ".,.) = " << '\n';

    // 调用打印矩阵内容的函数
    __printMatrix(stream, tensor, linesize, 1);
  }
}

// 打印张量到标准输出流，指定每行的列数
void print(const Tensor & t, int64_t linesize) {
  print(std::cout,t,linesize);
}

// 打印张量到指定流，指定每行的列数
std::ostream& print(std::ostream& stream, const Tensor & tensor_, int64_t linesize) {
  FormatGuard guard(stream);
  
  // 如果张量未定义，则直接返回流
  if(!tensor_.defined()) {
    stream << "[ Tensor (undefined) ]";
  // 如果张量未定义，向流中输出未定义的张量提示信息
  } else if (tensor_.is_sparse()) {
    stream << "[ " << tensor_.toString() << "{}\n";
    // 如果张量是稀疏张量，输出张量的字符串表示和花括号起始
    stream << "indices:\n" << tensor_._indices() << "\n";
    // 输出稀疏张量的索引信息
    stream << "values:\n" << tensor_._values() << "\n";
    // 输出稀疏张量的值信息
    stream << "size:\n" << tensor_.sizes() << "\n";
    // 输出稀疏张量的大小信息
    stream << "]";
    // 输出花括号结束
  } else {
    Tensor tensor;
    // 创建一个新的张量对象
    if (tensor_.is_quantized()) {
      // 如果张量是量化的，将其反量化后转换到 CPU 上，类型为双精度，并保证内存连续性
      tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mkldnn()) {
      // 如果张量使用 MKLDNN，输出相应的信息并将其转换为密集张量，类型为双精度，并保证内存连续性
      stream << "MKLDNN Tensor: ";
      tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mps()) {
      // 如果张量是 MPS 类型，由于不支持双精度张量，先进行类型转换再进行内存连续性处理
      // MPS does not support double tensors, so first copy then convert
      tensor = tensor_.to(kCPU).to(kDouble).contiguous();
    } else {
      // 否则，将张量转换为双精度张量，并保证内存连续性
      tensor = tensor_.to(kCPU, kDouble).contiguous();
    }
    if(tensor.ndimension() == 0) {
      // 如果张量是零维的，输出其值并添加张量描述字符串和花括号起始
      stream << defaultfloat << tensor.const_data_ptr<double>()[0] << '\n';
      stream << "[ " << tensor_.toString() << "{}";
    } else if(tensor.ndimension() == 1) {
      // 如果张量是一维的，根据格式打印张量内容，并输出张量描述字符串和花括号起始
      if (tensor.numel() > 0) {
        auto [scale, sz] = __printFormat(stream, tensor);
        if(scale != 1) {
          printScale(stream, scale);
        }
        const double* tensor_p = tensor.const_data_ptr<double>();
        for (const auto i : c10::irange(tensor.size(0))) {
          stream << std::setw(sz) << tensor_p[i]/scale << '\n';
        }
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "}";
    } else if(tensor.ndimension() == 2) {
      // 如果张量是二维的，打印矩阵格式的张量内容，并输出张量描述字符串和维度信息
      if (tensor.numel() > 0) {
        __printMatrix(stream, tensor, linesize, 0);
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "}";
    } else {
      // 如果张量是多维的，打印张量内容，并输出张量描述字符串和所有维度信息
      if (tensor.numel() > 0) {
        __printTensor(stream, tensor, linesize);
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
      for (const auto i : c10::irange(1, tensor.ndimension())) {
        stream << "," << tensor.size(i);
      }
      stream << "}";
    }
    if (tensor_.is_quantized()) {
      // 如果张量是量化的，输出量化方案和相关参数信息
      stream << ", qscheme: " << toString(tensor_.qscheme());
      if (tensor_.qscheme() == c10::kPerTensorAffine) {
        stream << ", scale: " << tensor_.q_scale();
        stream << ", zero_point: " << tensor_.q_zero_point();
      } else if (tensor_.qscheme() == c10::kPerChannelAffine ||
          tensor_.qscheme() == c10::kPerChannelAffineFloatQParams) {
        stream << ", scales: ";
        Tensor scales = tensor_.q_per_channel_scales();
        print(stream, scales, linesize);
        stream << ", zero_points: ";
        Tensor zero_points = tensor_.q_per_channel_zero_points();
        print(stream, zero_points, linesize);
        stream << ", axis: " << tensor_.q_per_channel_axis();
      }
    }

    // Proxy check for if autograd was built
    // 检查是否构建了自动微分代理
    // 检查张量是否具有自动求导元信息
    if (tensor.getIntrusivePtr()->autograd_meta()) {
      // 获取前向梯度（fw_grad），并指定级别为0
      auto& fw_grad = tensor._fw_grad(/* level */ 0);
      // 如果前向梯度已定义，则将其作为张量的切线输出到流中
      if (fw_grad.defined()) {
        stream << ", tangent:" << '\n' << fw_grad;
      }
    }
    // 在流中添加张量输出的结尾标记
    stream << " ]";
  }
  // 返回输出流，用于进一步处理或输出
  return stream;
}
```