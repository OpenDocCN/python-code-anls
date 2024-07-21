# `.\pytorch\torch\csrc\jit\tensorexpr\types.h`

```py
#pragma once
// 防止头文件被多次包含的预处理指令

#include <cstdint>
// 包含标准整数类型的头文件

#include <iosfwd>
// 包含前置声明流类的头文件

#include <c10/core/ScalarType.h>
// 包含C10库中的ScalarType定义的头文件

#include <c10/util/Logging.h>
// 包含C10库中的日志记录功能的头文件

#include <torch/csrc/Export.h>
// 包含导出宏定义的头文件

#include <torch/csrc/jit/tensorexpr/exceptions.h>
// 包含TensorExpr库中异常处理相关的头文件

namespace torch {
namespace jit {
namespace tensorexpr {

using int32 = std::int32_t;
// 定义别名int32作为std::int32_t的同义词

class Dtype;
// 声明类Dtype

TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);
// Dtype类的流输出运算符声明

using ScalarType = c10::ScalarType;
// 使用C10库中的ScalarType作为别名ScalarType

enum ElementType {
  kAllTypes = 0,
  kIntegralTypes = 1 << 0,
  kFloatingPointTypes = 1 << 1,
  kBoolType = 1 << 2,
  kComplexTypes = 1 << 3,
  kQintTypes = 1 << 4,
  kNonComplexOrQintTypes = kIntegralTypes | kBoolType | kFloatingPointTypes,
};
// 枚举ElementType，定义了不同数据类型的位掩码常量

// Data types for scalar and vector elements.
class TORCH_API Dtype {
 public:
  explicit Dtype(int8_t type)
      : scalar_type_(static_cast<ScalarType>(type)), lanes_(1) {}
  // 构造函数，根据type初始化scalar_type_和lanes_

  explicit Dtype(ScalarType type) : scalar_type_(type), lanes_(1) {}
  // 构造函数，根据type初始化scalar_type_和lanes_

  Dtype(int8_t type, int lanes)
      : scalar_type_(static_cast<ScalarType>(type)), lanes_(lanes) {}
  // 构造函数，根据type和lanes初始化scalar_type_和lanes_

  Dtype(ScalarType type, int lanes) : scalar_type_(type), lanes_(lanes) {}
  // 构造函数，根据type和lanes初始化scalar_type_和lanes_

  Dtype(Dtype type, int lanes)
      : scalar_type_(type.scalar_type_), lanes_(lanes) {
    if (type.lanes() != 1) {
      throw malformed_input("dtype lanes dont match");
    }
  }
  // 构造函数，根据type和lanes初始化scalar_type_和lanes_，并检查lanes是否匹配

  int lanes() const {
    return lanes_;
  }
  // 返回lanes_的值，表示元素向量的宽度

  ScalarType scalar_type() const {
    return scalar_type_;
  }
  // 返回scalar_type_的值，表示标量类型

  Dtype scalar_dtype() const;
  // 声明scalar_dtype()函数，返回Dtype对象

  bool operator==(const Dtype& other) const {
    return scalar_type_ == other.scalar_type_ && lanes_ == other.lanes_;
  }
  // 定义比较运算符==，比较两个Dtype对象是否相等

  bool operator!=(const Dtype& other) const {
    return !(*this == other);
  }
  // 定义比较运算符!=，比较两个Dtype对象是否不相等

  int byte_size() const;
  // 声明byte_size()函数，返回元素的字节大小

  std::string ToCppString() const;
  // 声明ToCppString()函数，返回描述该对象的C++字符串

  bool is_integral() const {
    return c10::isIntegralType(scalar_type_, true);
  }
  // 返回是否为整数类型的布尔值，使用C10库的isIntegralType函数进行判断

  bool is_floating_point() const {
    return c10::isFloatingType(scalar_type_);
  }
  // 返回是否为浮点数类型的布尔值，使用C10库的isFloatingType函数进行判断

  bool is_signed() const {
    return c10::isSignedType(scalar_type_);
  }
  // 返回是否为有符号类型的布尔值，使用C10库的isSignedType函数进行判断

  Dtype cloneWithScalarType(ScalarType nt) const {
    return Dtype(nt, lanes_);
  }
  // 返回一个新的Dtype对象，scalar_type_被替换为nt，lanes_保持不变

 private:
  friend TORCH_API std::ostream& operator<<(
      std::ostream& stream,
      const Dtype& dtype);
  // 声明友元函数operator<<，允许访问私有成员scalar_type_

  ScalarType scalar_type_;
  // 数据类型的标量类型

  int lanes_; // the width of the element for a vector time
  // 向量元素的宽度
};

extern TORCH_API Dtype kHandle;
// 声明全局变量kHandle，类型为Dtype，存储某种处理类型

#define NNC_DTYPE_DECLARATION(ctype, name) extern TORCH_API Dtype k##name;
// 定义NNC_DTYPE_DECLARATION宏，声明外部链接的Dtype变量k##name

AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, NNC_DTYPE_DECLARATION)
// 展开宏，为Bool、Half和BFloat16类型调用NNC_DTYPE_DECLARATION

NNC_DTYPE_DECLARATION(c10::quint8, QUInt8);
// 展开宏，声明外部链接的Dtype变量kQUInt8，类型为c10::quint8

NNC_DTYPE_DECLARATION(c10::qint8, QInt8);
// 展开宏，声明外部链接的Dtype变量kQInt8，类型为c10::qint8

#undef NNC_DTYPE_DECLARATION
// 取消定义NNC_DTYPE_DECLARATION宏

template <typename T>
TORCH_API Dtype ToDtype();
// 声明模板函数ToDtype，返回Dtype对象

#define NNC_TODTYPE_DECLARATION(ctype, name) \
  template <>                                \
  inline Dtype ToDtype<ctype>() {            \
    return k##name;                          \
  }
// 定义NNC_TODTYPE_DECLARATION宏，为给定ctype和name展开模板函数ToDtype

AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, NNC_TODTYPE_DECLARATION)
// 展开宏，为Bool、Half和BFloat16类型调用NNC_TODTYPE_DECLARATION

NNC_TODTYPE_DECLARATION(c10::quint8, QUInt8);
// 展开宏，为c10::quint8类型调用NNC_TODTYPE_DECLARATION

NNC_TODTYPE_DECLARATION(c10::qint8, QInt8);
// 展开宏，为c10::qint8类型调用NNC_TODTYPE_DECLARATION

#undef NNC_TODTYPE_DECLARATION
// 取消定义NNC_TODTYPE_DECLARATION宏

TORCH_API Dtype ToDtype(ScalarType type);
// 声明ToDtype函数，参数为ScalarType，返回Dtype对象
# 在给定的两种数据类型中进行类型提升，以便进行二元操作
inline Dtype promoteTypes(Dtype a, Dtype b) {
  # 如果两种数据类型的通道数不同，则抛出异常
  if (a.lanes() != b.lanes()) {
    throw malformed_input("promoting types with different lanes");
  }
  # 调用 C++ 库函数进行类型提升，并返回提升后的数据类型对象
  return Dtype(
      static_cast<ScalarType>(c10::promoteTypes(
          static_cast<c10::ScalarType>(a.scalar_type()),
          static_cast<c10::ScalarType>(b.scalar_type()))),
      a.lanes());
}

# 根据给定的两个操作数的数据类型和返回类型，确定二元操作的结果数据类型
inline Dtype BinaryOpDtype(
    Dtype op1_dtype,
    Dtype op2_dtype,
    ScalarType ret_type = ScalarType::Undefined) {
  # 如果两个操作数具有相同的数据类型
  if (op1_dtype == op2_dtype) {
    # 如果返回类型未指定，则直接返回第一个操作数的数据类型
    if (ret_type == ScalarType::Undefined) {
      return op1_dtype;
    }
    # 否则，将返回类型转换为数据类型对象并返回
    return ToDtype(ret_type);
  }

  # 如果两个操作数的通道数不同，则抛出异常
  if (op1_dtype.lanes() != op2_dtype.lanes()) {
    throw malformed_input("lanes dont match");
  }
  # 获取操作数的通道数
  int lanes = op1_dtype.lanes();

  # 对给定的两种数据类型进行类型提升，得到结果数据类型
  Dtype resultType = promoteTypes(op1_dtype, op2_dtype);
  # 如果结果数据类型的标量类型未定义，则抛出异常
  if (resultType.scalar_type() == ScalarType::Undefined) {
    throw malformed_input("scalar type doesn't match");
  }

  # 如果通道数为1，则使用固定的标量数据类型
  if (lanes == 1) {
    // Use the fixed scalar Dtypes.
    return ToDtype(resultType.scalar_type());
  }

  # 返回经过类型提升后的数据类型对象
  return resultType;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

# 引入 torch::jit::tensorexpr 命名空间中的 Dtype 类型，实现其到字符串的转换
using torch::jit::tensorexpr::Dtype;
std::string to_string(const Dtype& dtype);
# 引入 torch::jit::tensorexpr 命名空间中的 ScalarType 类型，实现其到字符串的转换
using torch::jit::tensorexpr::ScalarType;
std::string to_string(const ScalarType& dtype);

} // namespace std
```