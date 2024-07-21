# `.\pytorch\c10\core\ScalarType.cpp`

```
#include <c10/core/ScalarType.h>
#include <c10/util/Array.h>
#include <array>

namespace c10 {

namespace {

// 定义一些常量，代表不同的标量类型
constexpr auto u1 = ScalarType::Byte;        // 无符号字节类型
constexpr auto i1 = ScalarType::Char;        // 字符类型
constexpr auto i2 = ScalarType::Short;       // 短整型
constexpr auto i4 = ScalarType::Int;         // 整型
constexpr auto i8 = ScalarType::Long;        // 长整型
constexpr auto f2 = ScalarType::Half;        // 半精度浮点型
constexpr auto f4 = ScalarType::Float;       // 单精度浮点型
constexpr auto f8 = ScalarType::Double;      // 双精度浮点型
constexpr auto c2 = ScalarType::ComplexHalf; // 半精度复数类型
constexpr auto c4 = ScalarType::ComplexFloat;// 单精度复数类型
constexpr auto c8 = ScalarType::ComplexDouble;// 双精度复数类型
constexpr auto b1 = ScalarType::Bool;        // 布尔类型
constexpr auto bf = ScalarType::BFloat16;    // BF16 浮点数类型
constexpr auto ud = ScalarType::Undefined;   // 未定义类型

// 将标量类型按索引顺序存储到数组中
constexpr auto index2dtype = array_of<c10::ScalarType>(
    u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf);

// 计算标量类型到索引的映射
constexpr std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)>
calculate_dtype2index() {
  std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)> inverse = {};

  // 初始化映射数组，所有元素置为 -1
  for (int64_t i = 0; i < static_cast<int64_t>(ScalarType::NumOptions); i++) {
    inverse[i] = -1;
  }

  // 根据索引顺序的标量类型数组填充映射
  for (int64_t i = 0; i < static_cast<int64_t>(index2dtype.size()); i++) {
    inverse[static_cast<int64_t>(index2dtype[i])] = i;
  }

  return inverse;
}

// 计算后的标量类型到索引的映射存储在 dtype2index 中
constexpr auto dtype2index = calculate_dtype2index();

} // 匿名命名空间结束

// 通过 NumPy 的 promote_types 生成的函数，用于推断两种标量类型的提升类型
ScalarType promoteTypes(ScalarType a, ScalarType b) {
  if (a == ud || b == ud) {
    return ScalarType::Undefined; // 如果任意一个类型是未定义类型，则返回未定义类型
  }

  if (a == b) {
    return a; // 如果两种类型相同，则返回该类型
  }

  // 处理特定类型的错误情况
  if (isQIntType(a) || isQIntType(b)) {
    TORCH_CHECK(
        false,
        "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be, offending types: ",
        toString(a),
        " ",
        toString(b));
  }

  if (isBitsType(a) || isBitsType(b)) {
    return ScalarType::Undefined; // 如果任意一种类型是位类型，则返回未定义类型
  }

  if (isFloat8Type(a) || isFloat8Type(b)) {
    TORCH_CHECK(
        false,
        "Promotion for Float8 Types is not supported, attempted to promote ",
        toString(a),
        " and ",
        toString(b));
  }

  if (isBarebonesUnsignedType(a) || isBarebonesUnsignedType(b)) {
    // 处理无符号类型的提升问题，暂不支持对 uint8 到 uint64 的正确提升
    if (isFloatingType(a)) {
      return a; // 如果类型 a 是浮点类型，则返回 a
    }
    if (isFloatingType(b)) {
      return b; // 如果类型 b 是浮点类型，则返回 b
    }
  }

  // 默认情况下返回未定义类型
  return ScalarType::Undefined;
}

} // namespace c10
    TORCH_CHECK(
        false,
        "Promotion for uint16, uint32, uint64 types is not supported, attempted to promote ",
        toString(a),
        " and ",
        toString(b));
  }

  auto ix_a = dtype2index[static_cast<int64_t>(a)];
  TORCH_INTERNAL_ASSERT(ix_a != -1);
  auto ix_b = dtype2index[static_cast<int64_t>(b)];
  TORCH_INTERNAL_ASSERT(ix_b != -1);

  // This table axes must be consistent with index2dtype
  // 定义一个二维 constexpr 数组 _promoteTypesLookup，用于表示类型的提升规则
  // clang-format off
  static constexpr std::
  array<std::array<ScalarType, index2dtype.size()>, index2dtype.size()>
      _promoteTypesLookup = {{
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  bf*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, bf},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, bf},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, bf},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, bf},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, bf},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, bf},
  }};
  // clang-format on
  // 根据 ix_a 和 ix_b 索引到 _promoteTypesLookup 中的对应位置，获取类型提升结果
  return _promoteTypesLookup[ix_a][ix_b];
// 根据输入的标量类型返回相应的数据类型名称对
std::pair<std::string, std::string> getDtypeNames(c10::ScalarType scalarType) {
  // 根据不同的标量类型进行分支选择
  switch (scalarType) {
    case c10::ScalarType::UInt1:
      // 返回无符号1位整数类型名称和单位
      return std::make_pair("uint1", "bit");
    case c10::ScalarType::UInt2:
      // 返回无符号2位整数类型名称和空字符串作为单位
      return std::make_pair("uint2", "");
    case c10::ScalarType::UInt3:
      // 返回无符号3位整数类型名称和空字符串作为单位
      return std::make_pair("uint3", "");
    case c10::ScalarType::UInt4:
      // 返回无符号4位整数类型名称和空字符串作为单位
      return std::make_pair("uint4", "");
    case c10::ScalarType::UInt5:
      // 返回无符号5位整数类型名称和空字符串作为单位
      return std::make_pair("uint5", "");
    case c10::ScalarType::UInt6:
      // 返回无符号6位整数类型名称和空字符串作为单位
      return std::make_pair("uint6", "");
    case c10::ScalarType::UInt7:
      // 返回无符号7位整数类型名称和空字符串作为单位
      return std::make_pair("uint7", "");
    case c10::ScalarType::Byte:
      // 返回符号位不明确的8位整数类型名称（在numpy中byte为有符号类型，但此处为无符号）
      return std::make_pair("uint8", "");
    case c10::ScalarType::UInt16:
      // 返回无符号16位整数类型名称和空字符串作为单位
      return std::make_pair("uint16", "");
    case c10::ScalarType::UInt32:
      // 返回无符号32位整数类型名称和空字符串作为单位
      return std::make_pair("uint32", "");
    case c10::ScalarType::UInt64:
      // 返回无符号64位整数类型名称和空字符串作为单位
      return std::make_pair("uint64", "");
    case c10::ScalarType::Char:
      // 返回有符号8位整数类型名称（char在不同平台上有不同的符号性质，此处为有符号）
      return std::make_pair("int8", "");
    case c10::ScalarType::Double:
      // 返回双精度浮点数类型名称和其C++对应的double类型名称
      return std::make_pair("float64", "double");
    case c10::ScalarType::Float:
      // 返回单精度浮点数类型名称和其C++对应的float类型名称
      return std::make_pair("float32", "float");
    case c10::ScalarType::Int:
      // 返回32位整数类型名称和其C++对应的int类型名称
      return std::make_pair("int32", "int");
    case c10::ScalarType::Long:
      // 返回64位整数类型名称和其C++对应的long类型名称
      return std::make_pair("int64", "long");
    case c10::ScalarType::Short:
      // 返回16位整数类型名称和其C++对应的short类型名称
      return std::make_pair("int16", "short");
    case c10::ScalarType::Half:
      // 返回半精度浮点数类型名称和其C++对应的half类型名称
      return std::make_pair("float16", "half");
    case c10::ScalarType::ComplexHalf:
      // 返回复数半精度浮点数类型名称和其C++对应的chalf类型名称
      return std::make_pair("complex32", "chalf");
    case c10::ScalarType::ComplexFloat:
      // 返回复数单精度浮点数类型名称和其C++对应的cfloat类型名称
      return std::make_pair("complex64", "cfloat");
    case c10::ScalarType::ComplexDouble:
      // 返回复数双精度浮点数类型名称和其C++对应的cdouble类型名称
      return std::make_pair("complex128", "cdouble");
    case c10::ScalarType::Bool:
      // 返回布尔类型名称和空字符串作为单位
      return std::make_pair("bool", "");
    case c10::ScalarType::QInt8:
      // 返回量化8位整数类型名称和空字符串作为单位
      return std::make_pair("qint8", "");
    case c10::ScalarType::QUInt8:
      // 返回量化无符号8位整数类型名称和空字符串作为单位
      return std::make_pair("quint8", "");
    case c10::ScalarType::QInt32:
      // 返回量化32位整数类型名称和空字符串作为单位
      return std::make_pair("qint32", "");
    case c10::ScalarType::BFloat16:
      // 返回16位Brain浮点数类型名称和空字符串作为单位
      return std::make_pair("bfloat16", "");
    case c10::ScalarType::QUInt4x2:
      // 返回量化4x2位无符号整数类型名称和空字符串作为单位
      return std::make_pair("quint4x2", "");
    case c10::ScalarType::QUInt2x4:
      // 返回量化2x4位无符号整数类型名称和空字符串作为单位
      return std::make_pair("quint2x4", "");
    case c10::ScalarType::Bits1x8:
      // 返回8位的1位bit类型名称和空字符串作为单位
      return std::make_pair("bits1x8", "");
    case c10::ScalarType::Bits2x4:
      // 返回8位的2位bit类型名称和空字符串作为单位
      return std::make_pair("bits2x4", "");
    case c10::ScalarType::Bits4x2:
      // 返回8位的4位bit类型名称和空字符串作为单位
      return std::make_pair("bits4x2", "");
    case c10::ScalarType::Bits8:
      // 返回8位bit类型名称和空字符串作为单位
      return std::make_pair("bits8", "");
    case c10::ScalarType::Bits16:
      // 返回16位bit类型名称和空字符串作为单位
      return std::make_pair("bits16", "");
    case c10::ScalarType::Float8_e5m2:
      // 返回8位浮点数类型，exponent, 5 bit and 2 bits for mantissa
      return std::make_pair("float8_e5m2", "");
    // 如果输入的标量类型是 Float8_e4m3fn，返回对应的字符串和空字符串
    case c10::ScalarType::Float8_e4m3fn:
      return std::make_pair("float8_e4m3fn", "");
    // 如果输入的标量类型是 Float8_e5m2fnuz，返回对应的字符串和空字符串
    case c10::ScalarType::Float8_e5m2fnuz:
      return std::make_pair("float8_e5m2fnuz", "");
    // 如果输入的标量类型是 Float8_e4m3fnuz，返回对应的字符串和空字符串
    case c10::ScalarType::Float8_e4m3fnuz:
      return std::make_pair("float8_e4m3fnuz", "");
    // 如果输入的标量类型不是以上列出的任何一种，抛出运行时异常
    default:
      throw std::runtime_error("Unimplemented scalar type");
  }
}

// 定义函数，返回从字符串到标量类型的映射
const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap() {
  // 静态局部变量，存储结果的映射表
  static std::unordered_map<std::string, ScalarType> result;
  // 如果映射表不为空，直接返回已有的映射表
  if (!result.empty()) {
    return result;
  }

  // 定义宏，用于展开所有标量类型，并添加到集合中
#define DEFINE_SCALAR_TYPE(_1, n) c10::ScalarType::n,
  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};
#undef DEFINE_SCALAR_TYPE

  // 遍历所有标量类型
  for (auto scalar_type : all_scalar_types) {
    // 获取标量类型对应的名称集合
    auto names = getDtypeNames(scalar_type);
    // 将名称与标量类型映射添加到结果映射表中
    result[std::get<0>(names)] = scalar_type;
    // 如果第二个名称不为空，也将其添加到映射表中
    if (!std::get<1>(names).empty()) {
      result[std::get<1>(names)] = scalar_type;
    }
  }
  // 返回最终的字符串到标量类型的映射表
  return result;
}

} // namespace c10
```