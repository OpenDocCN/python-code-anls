# `.\pytorch\torch\csrc\jit\serialization\mobile_bytecode_generated.h`

```py
// 自动由FlatBuffers编译器生成，不要修改

// 防止包含的flatbuffers.h与生成此文件时的版本不兼容
#ifndef FLATBUFFERS_GENERATED_MOBILEBYTECODE_TORCH_JIT_MOBILE_SERIALIZATION_H_
#define FLATBUFFERS_GENERATED_MOBILEBYTECODE_TORCH_JIT_MOBILE_SERIALIZATION_H_

// 包含FlatBuffers库的头文件
#include "flatbuffers/flatbuffers.h"

// 确保包含的flatbuffers.h版本与生成此文件时的版本匹配，否则可能不兼容
static_assert(FLATBUFFERS_VERSION_MAJOR == 23 &&
              FLATBUFFERS_VERSION_MINOR == 3 &&
              FLATBUFFERS_VERSION_REVISION == 3,
             "Non-compatible flatbuffers version included");

// torch::jit::mobile::serialization命名空间，用于移动端模型序列化
namespace torch {
namespace jit {
namespace mobile {
namespace serialization {

// 下面是一系列结构体和枚举的声明，用于不同的序列化数据类型

struct Int;  // 整数类型
struct Bool;  // 布尔类型
struct Double;  // 双精度浮点类型

struct PerTensorAffineSchema;  // 每个张量的仿射模式结构
struct QuantizedSchema;  // 量化模式结构
struct QuantizedSchemaBuilder;  // 量化模式结构的构建器

struct TensorMetadata;  // 张量元数据结构
struct TensorMetadataBuilder;  // 张量元数据结构的构建器

struct String;  // 字符串类型
struct StringBuilder;  // 字符串构建器

struct Device;  // 设备信息结构
struct DeviceBuilder;  // 设备信息结构的构建器

struct List;  // 列表结构
struct ListBuilder;  // 列表结构的构建器

struct IntList;  // 整数列表结构
struct IntListBuilder;  // 整数列表结构的构建器

struct DoubleList;  // 双精度浮点数列表结构
struct DoubleListBuilder;  // 双精度浮点数列表结构的构建器

struct BoolList;  // 布尔值列表结构
struct BoolListBuilder;  // 布尔值列表结构的构建器

struct Tuple;  // 元组结构
struct TupleBuilder;  // 元组结构的构建器

struct Dict;  // 字典结构
struct DictBuilder;  // 字典结构的构建器

struct ObjectType;  // 对象类型结构
struct ObjectTypeBuilder;  // 对象类型结构的构建器

struct Object;  // 对象结构
struct ObjectBuilder;  // 对象结构的构建器

struct ComplexDouble;  // 复数双精度浮点数类型

struct EnumValue;  // 枚举值结构
struct EnumValueBuilder;  // 枚举值结构的构建器

struct Instruction;  // 指令结构

struct Operator;  // 操作符结构
struct OperatorBuilder;  // 操作符结构的构建器

struct Arg;  // 参数结构
struct ArgBuilder;  // 参数结构的构建器

struct Schema;  // 模式结构
struct SchemaBuilder;  // 模式结构的构建器

struct DebugInfo;  // 调试信息结构
struct DebugInfoBuilder;  // 调试信息结构的构建器

struct Function;  // 函数结构
struct FunctionBuilder;  // 函数结构的构建器

struct StorageData;  // 存储数据结构
struct StorageDataBuilder;  // 存储数据结构的构建器

struct IValue;  // 值结构
struct IValueBuilder;  // 值结构的构建器

struct ExtraFile;  // 额外文件结构
struct ExtraFileBuilder;  // 额外文件结构的构建器

struct Module;  // 模块结构
struct ModuleBuilder;  // 模块结构的构建器

// 枚举类型TypeType，表示不同的类型
enum class TypeType : uint8_t {
  UNSET = 0,
  CLASS_WITH_FIELD = 1,
  CUSTOM_CLASS = 2,
  CLASS_WITH_SETSTATE = 3,
  NON_OBJ = 4,
  MIN = UNSET,
  MAX = NON_OBJ
};

// 返回TypeType枚举的常量数组
inline const TypeType (&EnumValuesTypeType())[5] {
  static const TypeType values[] = {
    TypeType::UNSET,
    TypeType::CLASS_WITH_FIELD,
    TypeType::CUSTOM_CLASS,
    TypeType::CLASS_WITH_SETSTATE,
    TypeType::NON_OBJ
  };
  return values;
}

// 返回TypeType枚举的名称数组
inline const char * const *EnumNamesTypeType() {
  static const char * const names[6] = {
    "UNSET",
    "CLASS_WITH_FIELD",
    "CUSTOM_CLASS",
    "CLASS_WITH_SETSTATE",
    "NON_OBJ",
    nullptr
  };
  return names;
}

// 返回TypeType枚举值对应的名称
inline const char *EnumNameTypeType(TypeType e) {
  // 如果枚举值超出了TypeType的范围，则返回空字符串
  if (::flatbuffers::IsOutRange(e, TypeType::UNSET, TypeType::NON_OBJ)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesTypeType()[index];
}

// 枚举类型IValueUnion，表示不同的值类型
enum class IValueUnion : uint8_t {
  NONE = 0,
  Int = 1,
  Bool = 2,
  Double = 3,
  ComplexDouble = 4,
  TensorMetadata = 5,
  String = 6,
  List = 7,
  Tuple = 8,
  Dict = 9,
  Object = 10,
  IntList = 11,
  DoubleList = 12,
  BoolList = 13,
  Device = 14,
  EnumValue = 15,
  Function = 16,
  MIN = NONE,
  MAX = Function
};

// 给出IValueUnion枚举的最小值和最大值
// IValueUnion的值类型范围从NONE到Function
// 定义一个内联函数，返回一个常量引用数组，数组中包含17个 IValueUnion 枚举值
inline const IValueUnion (&EnumValuesIValueUnion())[17] {
  // 静态局部变量，存储了包含各种 IValueUnion 枚举值的数组
  static const IValueUnion values[] = {
    IValueUnion::NONE,
    IValueUnion::Int,
    IValueUnion::Bool,
    IValueUnion::Double,
    IValueUnion::ComplexDouble,
    IValueUnion::TensorMetadata,
    IValueUnion::String,
    IValueUnion::List,
    IValueUnion::Tuple,
    IValueUnion::Dict,
    IValueUnion::Object,
    IValueUnion::IntList,
    IValueUnion::DoubleList,
    IValueUnion::BoolList,
    IValueUnion::Device,
    IValueUnion::EnumValue,
    IValueUnion::Function
  };
  // 返回存储枚举值数组的引用
  return values;
}

// 定义一个内联函数，返回一个常量指针数组，数组中包含18个 IValueUnion 枚举值的名称
inline const char * const *EnumNamesIValueUnion() {
  // 静态局部变量，存储了包含各种 IValueUnion 枚举值名称的指针数组
  static const char * const names[18] = {
    "NONE",
    "Int",
    "Bool",
    "Double",
    "ComplexDouble",
    "TensorMetadata",
    "String",
    "List",
    "Tuple",
    "Dict",
    "Object",
    "IntList",
    "DoubleList",
    "BoolList",
    "Device",
    "EnumValue",
    "Function",
    nullptr
  };
  // 返回存储枚举值名称数组的指针
  return names;
}

// 定义一个内联函数，根据给定的枚举值返回其对应的名称
inline const char *EnumNameIValueUnion(IValueUnion e) {
  // 如果给定的枚举值超出了定义的范围，则返回空字符串
  if (::flatbuffers::IsOutRange(e, IValueUnion::NONE, IValueUnion::Function)) return "";
  // 将枚举值转换为索引
  const size_t index = static_cast<size_t>(e);
  // 返回与枚举值对应的名称字符串
  return EnumNamesIValueUnion()[index];
}

// 定义一个模板结构体，用于提供类型到枚举值的映射，对于未显式定义的类型，默认使用 NONE 枚举值
template<typename T> struct IValueUnionTraits {
  static const IValueUnion enum_value = IValueUnion::NONE;
};

// 以下为各个具体类型的特化模板结构体，每个结构体提供了该类型对应的枚举值
template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Int> {
  static const IValueUnion enum_value = IValueUnion::Int;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Bool> {
  static const IValueUnion enum_value = IValueUnion::Bool;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Double> {
  static const IValueUnion enum_value = IValueUnion::Double;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::ComplexDouble> {
  static const IValueUnion enum_value = IValueUnion::ComplexDouble;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::TensorMetadata> {
  static const IValueUnion enum_value = IValueUnion::TensorMetadata;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::String> {
  static const IValueUnion enum_value = IValueUnion::String;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::List> {
  static const IValueUnion enum_value = IValueUnion::List;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Tuple> {
  static const IValueUnion enum_value = IValueUnion::Tuple;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Dict> {
  static const IValueUnion enum_value = IValueUnion::Dict;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Object> {
  static const IValueUnion enum_value = IValueUnion::Object;
};

template<> struct IValueUnionTraits<torch::jit::mobile::serialization::IntList> {
  static const IValueUnion enum_value = IValueUnion::IntList;
};
// 定义模板特化，设置特定类型的枚举值为 DoubleList
template<> struct IValueUnionTraits<torch::jit::mobile::serialization::DoubleList> {
  static const IValueUnion enum_value = IValueUnion::DoubleList;
};

// 定义模板特化，设置特定类型的枚举值为 BoolList
template<> struct IValueUnionTraits<torch::jit::mobile::serialization::BoolList> {
  static const IValueUnion enum_value = IValueUnion::BoolList;
};

// 定义模板特化，设置特定类型的枚举值为 Device
template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Device> {
  static const IValueUnion enum_value = IValueUnion::Device;
};

// 定义模板特化，设置特定类型的枚举值为 EnumValue
template<> struct IValueUnionTraits<torch::jit::mobile::serialization::EnumValue> {
  static const IValueUnion enum_value = IValueUnion::EnumValue;
};

// 定义模板特化，设置特定类型的枚举值为 Function
template<> struct IValueUnionTraits<torch::jit::mobile::serialization::Function> {
  static const IValueUnion enum_value = IValueUnion::Function;
};

// 验证给定类型的 IValueUnion 对象
bool VerifyIValueUnion(::flatbuffers::Verifier &verifier, const void *obj, IValueUnion type);

// 验证给定类型的 IValueUnion 向量
bool VerifyIValueUnionVector(::flatbuffers::Verifier &verifier, const ::flatbuffers::Vector<::flatbuffers::Offset<void>> *values, const ::flatbuffers::Vector<IValueUnion> *types);

// 定义 FLATBUFFERS_MANUALLY_ALIGNED_STRUCT 类型 Int，以 8 字节对齐
FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) Int FLATBUFFERS_FINAL_CLASS {
 private:
  int64_t int_val_;

 public:
  // 默认构造函数，初始化 int_val_ 为 0
  Int()
      : int_val_(0) {
  }

  // 带参构造函数，用传入的 _int_val 初始化 int_val_
  Int(int64_t _int_val)
      : int_val_(::flatbuffers::EndianScalar(_int_val)) {
  }

  // 返回 int_val_ 的值，经过大小端转换
  int64_t int_val() const {
    return ::flatbuffers::EndianScalar(int_val_);
  }

  // 修改 int_val_ 的值为 _int_val，进行大小端转换
  void mutate_int_val(int64_t _int_val) {
    ::flatbuffers::WriteScalar(&int_val_, _int_val);
  }
};
FLATBUFFERS_STRUCT_END(Int, 8);  // 结束定义 Int 类型，以 8 字节对齐

// 定义 FLATBUFFERS_MANUALLY_ALIGNED_STRUCT 类型 Bool，以 1 字节对齐
FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(1) Bool FLATBUFFERS_FINAL_CLASS {
 private:
  uint8_t bool_val_;

 public:
  // 默认构造函数，初始化 bool_val_ 为 0
  Bool()
      : bool_val_(0) {
  }

  // 带参构造函数，用传入的 _bool_val 初始化 bool_val_
  Bool(bool _bool_val)
      : bool_val_(::flatbuffers::EndianScalar(static_cast<uint8_t>(_bool_val))) {
  }

  // 返回 bool_val_ 的值，经过大小端转换，非零为 true
  bool bool_val() const {
    return ::flatbuffers::EndianScalar(bool_val_) != 0;
  }

  // 修改 bool_val_ 的值为 _bool_val，进行大小端转换
  void mutate_bool_val(bool _bool_val) {
    ::flatbuffers::WriteScalar(&bool_val_, static_cast<uint8_t>(_bool_val));
  }
};
FLATBUFFERS_STRUCT_END(Bool, 1);  // 结束定义 Bool 类型，以 1 字节对齐

// 定义 FLATBUFFERS_MANUALLY_ALIGNED_STRUCT 类型 Double，以 8 字节对齐
FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) Double FLATBUFFERS_FINAL_CLASS {
 private:
  double double_val_;

 public:
  // 默认构造函数，初始化 double_val_ 为 0
  Double()
      : double_val_(0) {
  }

  // 带参构造函数，用传入的 _double_val 初始化 double_val_
  Double(double _double_val)
      : double_val_(::flatbuffers::EndianScalar(_double_val)) {
  }

  // 返回 double_val_ 的值，经过大小端转换
  double double_val() const {
    return ::flatbuffers::EndianScalar(double_val_);
  }

  // 修改 double_val_ 的值为 _double_val，进行大小端转换
  void mutate_double_val(double _double_val) {
    ::flatbuffers::WriteScalar(&double_val_, _double_val);
  }
};
FLATBUFFERS_STRUCT_END(Double, 8);  // 结束定义 Double 类型，以 8 字节对齐

// 定义 FLATBUFFERS_MANUALLY_ALIGNED_STRUCT 类型 PerTensorAffineSchema，以 8 字节对齐
FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) PerTensorAffineSchema FLATBUFFERS_FINAL_CLASS {
 private:
  double q_scale_;
  int32_t q_zero_point_;
  int32_t padding0__;

 public:
  // 默认构造函数，初始化 q_scale_, q_zero_point_, padding0__ 为 0
  PerTensorAffineSchema()
      : q_scale_(0),
        q_zero_point_(0),
        padding0__(0) {
    (void)padding0__;  // 防止编译器警告未使用的成员变量
  }

  // 带参构造函数，用传入的 _q_scale, _q_zero_point 初始化 q_scale_, q_zero_point_
  PerTensorAffineSchema(double _q_scale, int32_t _q_zero_point)
      : q_scale_(::flatbuffers::EndianScalar(_q_scale)),
        q_zero_point_(::flatbuffers::EndianScalar(_q_zero_point)),
        padding0__(0) {


这些注释为每个结构体和函数提供了详细解释，帮助读者理解其定义和作用。
    (void)padding0__;
  }


    // padding0__ 是一个未使用的变量，这里使用 (void) 来避免编译器产生未使用变量的警告
    (void)padding0__;
  }



  double q_scale() const {
    // 返回缩放因子 q_scale_ 的大小，将其转换为当前系统的端序
    return ::flatbuffers::EndianScalar(q_scale_);
  }


  void mutate_q_scale(double _q_scale) {
    // 将输入的 _q_scale 缩放因子写入到 q_scale_，并确保其端序正确
    ::flatbuffers::WriteScalar(&q_scale_, _q_scale);
  }


  int32_t q_zero_point() const {
    // 返回零点偏移量 q_zero_point_ 的值，将其转换为当前系统的端序
    return ::flatbuffers::EndianScalar(q_zero_point_);
  }


  void mutate_q_zero_point(int32_t _q_zero_point) {
    // 将输入的 _q_zero_point 零点偏移量写入到 q_zero_point_，并确保其端序正确
    ::flatbuffers::WriteScalar(&q_zero_point_, _q_zero_point);
  }
};
FLATBUFFERS_STRUCT_END(PerTensorAffineSchema, 16);

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) ComplexDouble FLATBUFFERS_FINAL_CLASS {
 private:
  double real_;  // 双精度实部

  double imag_;  // 双精度虚部

 public:
  ComplexDouble()
      : real_(0),  // 默认构造函数，初始化实部和虚部为0
        imag_(0) {
  }
  ComplexDouble(double _real, double _imag)
      : real_(::flatbuffers::EndianScalar(_real)),  // 带参数的构造函数，指定实部和虚部
        imag_(::flatbuffers::EndianScalar(_imag)) {
  }
  double real() const {  // 获取实部值
    return ::flatbuffers::EndianScalar(real_);
  }
  void mutate_real(double _real) {  // 修改实部值
    ::flatbuffers::WriteScalar(&real_, _real);
  }
  double imag() const {  // 获取虚部值
    return ::flatbuffers::EndianScalar(imag_);
  }
  void mutate_imag(double _imag) {  // 修改虚部值
    ::flatbuffers::WriteScalar(&imag_, _imag);
  }
};
FLATBUFFERS_STRUCT_END(ComplexDouble, 16);

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(4) Instruction FLATBUFFERS_FINAL_CLASS {
 private:
  int8_t op_;  // 操作符，占1字节

  int8_t padding0__;  // 填充字节，占1字节

  uint16_t n_;  // 数量，占2字节

  int32_t x_;  // X值，占4字节

 public:
  Instruction()
      : op_(0),  // 默认构造函数，初始化所有成员变量为0
        padding0__(0),
        n_(0),
        x_(0) {
    (void)padding0__;  // 填充字节暂时不使用
  }
  Instruction(int8_t _op, uint16_t _n, int32_t _x)
      : op_(::flatbuffers::EndianScalar(_op)),  // 带参数的构造函数，设置操作符、数量和X值
        padding0__(0),
        n_(::flatbuffers::EndianScalar(_n)),
        x_(::flatbuffers::EndianScalar(_x)) {
    (void)padding0__;
  }
  int8_t op() const {  // 获取操作符值
    return ::flatbuffers::EndianScalar(op_);
  }
  void mutate_op(int8_t _op) {  // 修改操作符值
    ::flatbuffers::WriteScalar(&op_, _op);
  }
  uint16_t n() const {  // 获取数量值
    return ::flatbuffers::EndianScalar(n_);
  }
  void mutate_n(uint16_t _n) {  // 修改数量值
    ::flatbuffers::WriteScalar(&n_, _n);
  }
  int32_t x() const {  // 获取X值
    return ::flatbuffers::EndianScalar(x_);
  }
  void mutate_x(int32_t _x) {  // 修改X值
    ::flatbuffers::WriteScalar(&x_, _x);
  }
};
FLATBUFFERS_STRUCT_END(Instruction, 8);

struct QuantizedSchema FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef QuantizedSchemaBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_QSCHEME = 4,  // 量化方案字段在VTable中的偏移量为4
    VT_SCALE = 6,    // 缩放值字段在VTable中的偏移量为6
    VT_ZERO_POINT = 8,  // 零点值字段在VTable中的偏移量为8
    VT_SCALES = 10,     // 缩放值数组字段在VTable中的偏移量为10
    VT_ZERO_POINTS = 12,  // 零点值数组字段在VTable中的偏移量为12
    VT_AXIS = 14       // 轴值字段在VTable中的偏移量为14
  };
  int8_t qscheme() const {  // 获取量化方案字段的值
    return GetField<int8_t>(VT_QSCHEME, 0);
  }
  bool mutate_qscheme(int8_t _qscheme = 0) {  // 修改量化方案字段的值
    return SetField<int8_t>(VT_QSCHEME, _qscheme, 0);
  }
  double scale() const {  // 获取缩放值字段的值
    return GetField<double>(VT_SCALE, 0.0);
  }
  bool mutate_scale(double _scale = 0.0) {  // 修改缩放值字段的值
    return SetField<double>(VT_SCALE, _scale, 0.0);
  }
  int32_t zero_point() const {  // 获取零点值字段的值
    return GetField<int32_t>(VT_ZERO_POINT, 0);
  }
  bool mutate_zero_point(int32_t _zero_point = 0) {  // 修改零点值字段的值
    return SetField<int32_t>(VT_ZERO_POINT, _zero_point, 0);
  }
  const torch::jit::mobile::serialization::TensorMetadata *scales() const {  // 获取缩放值数组字段的值
    return GetPointer<const torch::jit::mobile::serialization::TensorMetadata *>(VT_SCALES);
  }
  torch::jit::mobile::serialization::TensorMetadata *mutable_scales() {  // 获取可变的缩放值数组字段
  // 返回指向 VT_SCALES 的指针，类型为 torch::jit::mobile::serialization::TensorMetadata*
  return GetPointer<torch::jit::mobile::serialization::TensorMetadata *>(VT_SCALES);
}

  // 返回指向 VT_ZERO_POINTS 的指针，类型为 const torch::jit::mobile::serialization::TensorMetadata*
const torch::jit::mobile::serialization::TensorMetadata *zero_points() const {
  return GetPointer<const torch::jit::mobile::serialization::TensorMetadata *>(VT_ZERO_POINTS);
}

  // 返回指向 VT_ZERO_POINTS 的可变指针，类型为 torch::jit::mobile::serialization::TensorMetadata*
torch::jit::mobile::serialization::TensorMetadata *mutable_zero_points() {
  return GetPointer<torch::jit::mobile::serialization::TensorMetadata *>(VT_ZERO_POINTS);
}

  // 返回 VT_AXIS 的值，类型为 int32_t
int32_t axis() const {
  return GetField<int32_t>(VT_AXIS, 0);
}

  // 设置 VT_AXIS 的值为 _axis，默认为 0
bool mutate_axis(int32_t _axis = 0) {
  return SetField<int32_t>(VT_AXIS, _axis, 0);
}

  // 使用给定的 verifier 验证该表的结构完整性
bool Verify(::flatbuffers::Verifier &verifier) const {
  return VerifyTableStart(verifier) &&                            // 验证表的起始
         VerifyField<int8_t>(verifier, VT_QSCHEME, 1) &&           // 验证 int8_t 类型字段 VT_QSCHEME
         VerifyField<double>(verifier, VT_SCALE, 8) &&             // 验证 double 类型字段 VT_SCALE
         VerifyField<int32_t>(verifier, VT_ZERO_POINT, 4) &&       // 验证 int32_t 类型字段 VT_ZERO_POINT
         VerifyOffset(verifier, VT_SCALES) &&                      // 验证指向 VT_SCALES 的偏移量
         verifier.VerifyTable(scales()) &&                         // 验证 scales() 方法返回的表
         VerifyOffset(verifier, VT_ZERO_POINTS) &&                 // 验证指向 VT_ZERO_POINTS 的偏移量
         verifier.VerifyTable(zero_points()) &&                    // 验证 zero_points() 方法返回的表
         VerifyField<int32_t>(verifier, VT_AXIS, 4) &&             // 验证 int32_t 类型字段 VT_AXIS
         verifier.EndTable();                                      // 验证表的结束
}
};

// 定义结构体 QuantizedSchemaBuilder
struct QuantizedSchemaBuilder {
  // 定义 Table 类型为 QuantizedSchema
  typedef QuantizedSchema Table;
  // 引用 FlatBufferBuilder 对象
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 偏移量类型，用于存储表的起始位置
  ::flatbuffers::uoffset_t start_;

  // 添加 qscheme 字段到 FlatBufferBuilder 中
  void add_qscheme(int8_t qscheme) {
    fbb_.AddElement<int8_t>(QuantizedSchema::VT_QSCHEME, qscheme, 0);
  }

  // 添加 scale 字段到 FlatBufferBuilder 中
  void add_scale(double scale) {
    fbb_.AddElement<double>(QuantizedSchema::VT_SCALE, scale, 0.0);
  }

  // 添加 zero_point 字段到 FlatBufferBuilder 中
  void add_zero_point(int32_t zero_point) {
    fbb_.AddElement<int32_t>(QuantizedSchema::VT_ZERO_POINT, zero_point, 0);
  }

  // 添加 scales 字段到 FlatBufferBuilder 中
  void add_scales(::flatbuffers::Offset<torch::jit::mobile::serialization::TensorMetadata> scales) {
    fbb_.AddOffset(QuantizedSchema::VT_SCALES, scales);
  }

  // 添加 zero_points 字段到 FlatBufferBuilder 中
  void add_zero_points(::flatbuffers::Offset<torch::jit::mobile::serialization::TensorMetadata> zero_points) {
    fbb_.AddOffset(QuantizedSchema::VT_ZERO_POINTS, zero_points);
  }

  // 添加 axis 字段到 FlatBufferBuilder 中
  void add_axis(int32_t axis) {
    fbb_.AddElement<int32_t>(QuantizedSchema::VT_AXIS, axis, 0);
  }

  // 构造函数，初始化 QuantizedSchemaBuilder 对象
  explicit QuantizedSchemaBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    // 在 FlatBufferBuilder 中开始一个新的表
    start_ = fbb_.StartTable();
  }

  // 完成表的构建，返回表的偏移量
  ::flatbuffers::Offset<QuantizedSchema> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<QuantizedSchema>(end);
    return o;
  }
};

// 创建 QuantizedSchema 对象并返回其偏移量
inline ::flatbuffers::Offset<QuantizedSchema> CreateQuantizedSchema(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    int8_t qscheme = 0,
    double scale = 0.0,
    int32_t zero_point = 0,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::TensorMetadata> scales = 0,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::TensorMetadata> zero_points = 0,
    int32_t axis = 0) {
  QuantizedSchemaBuilder builder_(_fbb);
  builder_.add_scale(scale);
  builder_.add_axis(axis);
  builder_.add_zero_points(zero_points);
  builder_.add_scales(scales);
  builder_.add_zero_point(zero_point);
  builder_.add_qscheme(qscheme);
  return builder_.Finish();
}

// 定义结构体 TensorMetadata
struct TensorMetadata FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef TensorMetadataBuilder Builder;

  // 枚举常量，表示 FlatBuffers 表的偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_STORAGE_LOCATION_INDEX = 4,
    VT_SCALAR_TYPE = 6,
    VT_STORAGE_OFFSET = 8,
    VT_SIZES = 10,
    VT_STRIDES = 12,
    VT_REQUIRES_GRAD = 14,
    VT_QUANTIZED_SCHEMA = 16
  };

  // 获取 storage_location_index 字段的值
  uint32_t storage_location_index() const {
    return GetField<uint32_t>(VT_STORAGE_LOCATION_INDEX, 0);
  }

  // 设置 storage_location_index 字段的值
  bool mutate_storage_location_index(uint32_t _storage_location_index = 0) {
    return SetField<uint32_t>(VT_STORAGE_LOCATION_INDEX, _storage_location_index, 0);
  }

  // 获取 scalar_type 字段的值
  int8_t scalar_type() const {
    return GetField<int8_t>(VT_SCALAR_TYPE, 0);
  }

  // 设置 scalar_type 字段的值
  bool mutate_scalar_type(int8_t _scalar_type = 0) {
    return SetField<int8_t>(VT_SCALAR_TYPE, _scalar_type, 0);
  }

  // 获取 storage_offset 字段的值
  int32_t storage_offset() const {
    return GetField<int32_t>(VT_STORAGE_OFFSET, 0);
  }

  // 设置 storage_offset 字段的值
  bool mutate_storage_offset(int32_t _storage_offset = 0) {
  // 返回一个 int32_t 类型的字段，使用 SetField 函数设置 VT_STORAGE_OFFSET 字段的值为 _storage_offset，然后返回 0
  return SetField<int32_t>(VT_STORAGE_OFFSET, _storage_offset, 0);
}

// 返回一个指向 const flatbuffers::Vector<int32_t> 的指针，该向量存储在 VT_SIZES 字段中
const ::flatbuffers::Vector<int32_t> *sizes() const {
  return GetPointer<const ::flatbuffers::Vector<int32_t> *>(VT_SIZES);
}

// 返回一个指向 flatbuffers::Vector<int32_t> 的指针，可以用来修改存储在 VT_SIZES 字段中的向量数据
::flatbuffers::Vector<int32_t> *mutable_sizes() {
  return GetPointer<::flatbuffers::Vector<int32_t> *>(VT_SIZES);
}

// 返回一个指向 const flatbuffers::Vector<int32_t> 的指针，该向量存储在 VT_STRIDES 字段中
const ::flatbuffers::Vector<int32_t> *strides() const {
  return GetPointer<const ::flatbuffers::Vector<int32_t> *>(VT_STRIDES);
}

// 返回一个指向 flatbuffers::Vector<int32_t> 的指针，可以用来修改存储在 VT_STRIDES 字段中的向量数据
::flatbuffers::Vector<int32_t> *mutable_strides() {
  return GetPointer<::flatbuffers::Vector<int32_t> *>(VT_STRIDES);
}

// 返回一个 bool 值，表示 VT_REQUIRES_GRAD 字段是否不为 0，即是否需要梯度计算
bool requires_grad() const {
  return GetField<uint8_t>(VT_REQUIRES_GRAD, 0) != 0;
}

// 将 VT_REQUIRES_GRAD 字段设置为 _requires_grad 的值，并返回设置结果
bool mutate_requires_grad(bool _requires_grad = 0) {
  return SetField<uint8_t>(VT_REQUIRES_GRAD, static_cast<uint8_t>(_requires_grad), 0);
}

// 返回一个指向 const torch::jit::mobile::serialization::QuantizedSchema 的指针，该对象存储在 VT_QUANTIZED_SCHEMA 字段中
const torch::jit::mobile::serialization::QuantizedSchema *quantized_schema() const {
  return GetPointer<const torch::jit::mobile::serialization::QuantizedSchema *>(VT_QUANTIZED_SCHEMA);
}

// 返回一个指向 torch::jit::mobile::serialization::QuantizedSchema 的指针，可以用来修改存储在 VT_QUANTIZED_SCHEMA 字段中的对象数据
torch::jit::mobile::serialization::QuantizedSchema *mutable_quantized_schema() {
  return GetPointer<torch::jit::mobile::serialization::QuantizedSchema *>(VT_QUANTIZED_SCHEMA);
}

// 使用 verifier 对象来验证当前对象的字段和偏移量是否正确，返回验证结果的布尔值
bool Verify(::flatbuffers::Verifier &verifier) const {
  return VerifyTableStart(verifier) &&                          // 验证表的开始
         VerifyField<uint32_t>(verifier, VT_STORAGE_LOCATION_INDEX, 4) &&  // 验证 uint32_t 类型的字段
         VerifyField<int8_t>(verifier, VT_SCALAR_TYPE, 1) &&      // 验证 int8_t 类型的字段
         VerifyField<int32_t>(verifier, VT_STORAGE_OFFSET, 4) &&  // 验证 int32_t 类型的字段
         VerifyOffset(verifier, VT_SIZES) &&                     // 验证 VT_SIZES 字段的偏移量
         verifier.VerifyVector(sizes()) &&                       // 验证 sizes() 函数返回的向量
         VerifyOffset(verifier, VT_STRIDES) &&                   // 验证 VT_STRIDES 字段的偏移量
         verifier.VerifyVector(strides()) &&                     // 验证 strides() 函数返回的向量
         VerifyField<uint8_t>(verifier, VT_REQUIRES_GRAD, 1) &&   // 验证 uint8_t 类型的字段
         VerifyOffset(verifier, VT_QUANTIZED_SCHEMA) &&          // 验证 VT_QUANTIZED_SCHEMA 字段的偏移量
         verifier.VerifyTable(quantized_schema()) &&             // 验证 quantized_schema() 函数返回的对象
         verifier.EndTable();                                    // 结束表的验证
}
};

// 定义一个结构体 `TensorMetadataBuilder`，用于构建 `TensorMetadata` 对象
struct TensorMetadataBuilder {
  // 表示 `TensorMetadata` 类型的别名 `Table`
  typedef TensorMetadata Table;
  // 引用 FlatBufferBuilder 对象 `_fbb`，用于构建 FlatBuffer
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 起始偏移量 `start_`，用于记录 FlatBufferBuilder 的起始表位置
  ::flatbuffers::uoffset_t start_;

  // 添加存储位置索引 `storage_location_index` 到 FlatBuffer 中
  void add_storage_location_index(uint32_t storage_location_index) {
    fbb_.AddElement<uint32_t>(TensorMetadata::VT_STORAGE_LOCATION_INDEX, storage_location_index, 0);
  }

  // 添加标量类型 `scalar_type` 到 FlatBuffer 中
  void add_scalar_type(int8_t scalar_type) {
    fbb_.AddElement<int8_t>(TensorMetadata::VT_SCALAR_TYPE, scalar_type, 0);
  }

  // 添加存储偏移量 `storage_offset` 到 FlatBuffer 中
  void add_storage_offset(int32_t storage_offset) {
    fbb_.AddElement<int32_t>(TensorMetadata::VT_STORAGE_OFFSET, storage_offset, 0);
  }

  // 添加大小 `sizes` 到 FlatBuffer 中
  void add_sizes(::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> sizes) {
    fbb_.AddOffset(TensorMetadata::VT_SIZES, sizes);
  }

  // 添加步长 `strides` 到 FlatBuffer 中
  void add_strides(::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> strides) {
    fbb_.AddOffset(TensorMetadata::VT_STRIDES, strides);
  }

  // 添加是否需要梯度 `requires_grad` 到 FlatBuffer 中
  void add_requires_grad(bool requires_grad) {
    fbb_.AddElement<uint8_t>(TensorMetadata::VT_REQUIRES_GRAD, static_cast<uint8_t>(requires_grad), 0);
  }

  // 添加量化模式 `quantized_schema` 到 FlatBuffer 中
  void add_quantized_schema(::flatbuffers::Offset<torch::jit::mobile::serialization::QuantizedSchema> quantized_schema) {
    fbb_.AddOffset(TensorMetadata::VT_QUANTIZED_SCHEMA, quantized_schema);
  }

  // 构造函数，初始化 `TensorMetadataBuilder` 对象
  explicit TensorMetadataBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    // 在 FlatBufferBuilder 上开始一个新表，并记录起始位置
    start_ = fbb_.StartTable();
  }

  // 完成构建 `TensorMetadata` 对象，返回其在 FlatBuffer 中的偏移量
  ::flatbuffers::Offset<TensorMetadata> Finish() {
    // 结束当前表，并返回其结束位置作为偏移量
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<TensorMetadata>(end);
    return o;
  }
};

// 创建 `TensorMetadata` 对象的函数，使用指定参数填充 `TensorMetadataBuilder`
inline ::flatbuffers::Offset<TensorMetadata> CreateTensorMetadata(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    uint32_t storage_location_index = 0,
    int8_t scalar_type = 0,
    int32_t storage_offset = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> sizes = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> strides = 0,
    bool requires_grad = false,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::QuantizedSchema> quantized_schema = 0) {
  // 创建 `TensorMetadataBuilder` 对象
  TensorMetadataBuilder builder_(_fbb);
  // 分别添加各个字段到 FlatBuffer 中
  builder_.add_quantized_schema(quantized_schema);
  builder_.add_strides(strides);
  builder_.add_sizes(sizes);
  builder_.add_storage_offset(storage_offset);
  builder_.add_storage_location_index(storage_location_index);
  builder_.add_requires_grad(requires_grad);
  builder_.add_scalar_type(scalar_type);
  // 完成构建，并返回 `TensorMetadata` 在 FlatBuffer 中的偏移量
  return builder_.Finish();
}

// 直接创建 `TensorMetadata` 对象的函数，使用指定参数填充 `TensorMetadataBuilder`
inline ::flatbuffers::Offset<TensorMetadata> CreateTensorMetadataDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    uint32_t storage_location_index = 0,
    int8_t scalar_type = 0,
    int32_t storage_offset = 0,
    const std::vector<int32_t> *sizes = nullptr,
    const std::vector<int32_t> *strides = nullptr,
    bool requires_grad = false,
    // ...
    // 创建一个偏移量为0的QuantizedSchema对象，偏移量是flatbuffers::Offset类型的，torch::jit::mobile::serialization::QuantizedSchema是flatbuffers::Offset类型
    ::flatbuffers::Offset<torch::jit::mobile::serialization::QuantizedSchema> quantized_schema = 0) {
    // 如果sizes不为空指针，则使用_fbb创建int32_t类型的向量，否则为0
      auto sizes__ = sizes ? _fbb.CreateVector<int32_t>(*sizes) : 0;
    // 如果strides不为空指针，则使用_fbb创建int32_t类型的向量，否则为0
      auto strides__ = strides ? _fbb.CreateVector<int32_t>(*strides) : 0;
    // 使用给定的参数创建torch::jit::mobile::serialization::CreateTensorMetadata对象
      return torch::jit::mobile::serialization::CreateTensorMetadata(
          _fbb,  // 使用flatbuffers::FlatBufferBuilder对象创建新对象
          storage_location_index,  // int32_t类型参数,可更改为声明
          scalar_type,  contains scalr type type's returns performs The
}

// 结构体 String，继承自 flatbuffers 的 Table
struct String FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef StringBuilder Builder; // 使用 StringBuilder 作为构建器类型
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DATA = 4 // VT_DATA 偏移量为 4，用于指向数据的偏移
  };
  // 获取不可变的数据指针
  const ::flatbuffers::String *data() const {
    return GetPointer<const ::flatbuffers::String *>(VT_DATA);
  }
  // 获取可变的数据指针
  ::flatbuffers::String *mutable_data() {
    return GetPointer<::flatbuffers::String *>(VT_DATA);
  }
  // 验证函数，用于验证表的有效性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) && // 验证表的起始
           VerifyOffset(verifier, VT_DATA) && // 验证偏移量
           verifier.VerifyString(data()) && // 验证数据指针指向的字符串
           verifier.EndTable(); // 结束表的验证
  }
};

// StringBuilder 结构体
struct StringBuilder {
  typedef String Table; // 使用 String 作为表类型
  ::flatbuffers::FlatBufferBuilder &fbb_; // FlatBufferBuilder 引用
  ::flatbuffers::uoffset_t start_; // 起始偏移量
  // 添加数据函数
  void add_data(::flatbuffers::Offset<::flatbuffers::String> data) {
    fbb_.AddOffset(String::VT_DATA, data); // 添加数据偏移量到 FlatBufferBuilder
  }
  // 构造函数，初始化 FlatBufferBuilder 引用
  explicit StringBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable(); // 开始构建表
  }
  // 完成构建，返回 String 表的偏移量
  ::flatbuffers::Offset<String> Finish() {
    const auto end = fbb_.EndTable(start_); // 结束表的构建
    auto o = ::flatbuffers::Offset<String>(end); // 创建 String 表的偏移量
    return o; // 返回偏移量
  }
};

// 创建 String 表的函数，使用 FlatBufferBuilder 构建
inline ::flatbuffers::Offset<String> CreateString(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> data = 0) {
  StringBuilder builder_(_fbb); // 使用 StringBuilder 构建器
  builder_.add_data(data); // 添加数据到构建器
  return builder_.Finish(); // 完成构建并返回偏移量
}

// 直接创建 String 表的函数，使用 FlatBufferBuilder 构建
inline ::flatbuffers::Offset<String> CreateStringDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *data = nullptr) {
  auto data__ = data ? _fbb.CreateString(data) : 0; // 如果有数据，创建字符串并返回偏移量
  return torch::jit::mobile::serialization::CreateString(
      _fbb,
      data__); // 调用另一个函数来创建 String 表并返回偏移量
}

// 结构体 Device，继承自 flatbuffers 的 Table
struct Device FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef DeviceBuilder Builder; // 使用 DeviceBuilder 作为构建器类型
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_STR = 4 // VT_STR 偏移量为 4，用于指向字符串的偏移
  };
  // 获取不可变的字符串指针
  const ::flatbuffers::String *str() const {
    return GetPointer<const ::flatbuffers::String *>(VT_STR);
  }
  // 获取可变的字符串指针
  ::flatbuffers::String *mutable_str() {
    return GetPointer<::flatbuffers::String *>(VT_STR);
  }
  // 验证函数，用于验证表的有效性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) && // 验证表的起始
           VerifyOffset(verifier, VT_STR) && // 验证偏移量
           verifier.VerifyString(str()) && // 验证字符串指针指向的字符串
           verifier.EndTable(); // 结束表的验证
  }
};

// DeviceBuilder 结构体
struct DeviceBuilder {
  typedef Device Table; // 使用 Device 作为表类型
  ::flatbuffers::FlatBufferBuilder &fbb_; // FlatBufferBuilder 引用
  ::flatbuffers::uoffset_t start_; // 起始偏移量
  // 添加字符串函数
  void add_str(::flatbuffers::Offset<::flatbuffers::String> str) {
    fbb_.AddOffset(Device::VT_STR, str); // 添加字符串偏移量到 FlatBufferBuilder
  }
  // 构造函数，初始化 FlatBufferBuilder 引用
  explicit DeviceBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable(); // 开始构建表
  }
  // 完成构建，返回 Device 表的偏移量
  ::flatbuffers::Offset<Device> Finish() {
    const auto end = fbb_.EndTable(start_); // 结束表的构建
    auto o = ::flatbuffers::Offset<Device>(end); // 创建 Device 表的偏移量
    return o; // 返回偏移量
  }
};

// 创建 Device 表的函数，使用 FlatBufferBuilder 构建
inline ::flatbuffers::Offset<Device> CreateDevice(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> str = 0) {
    // 使用 DeviceBuilder 构建器
    DeviceBuilder builder_(_fbb);
    // 添加字符串到构建器
    builder_.add_str(str);
    // 完成构建并返回偏移量
    return builder_.Finish();
}
    // 创建一个新的设备构建器对象，使用给定的FlatBuffers构建器对象_fbb进行初始化
    DeviceBuilder builder_(_fbb);
    // 向设备构建器对象中添加一个字符串偏移量（在FlatBuffers中使用）
    builder_.add_str(str);
    // 完成设备构建器对象的构建，返回最终的设备对象
    return builder_.Finish();
}

// 创建一个直接设备的函数，返回设备的偏移量
inline ::flatbuffers::Offset<Device> CreateDeviceDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *str = nullptr) {
  // 如果提供了字符串，则创建一个 FlatBuffers 字符串，否则设置为 0
  auto str__ = str ? _fbb.CreateString(str) : 0;
  // 调用 FlatBuffers 库中的 CreateDevice 函数，返回设备的偏移量
  return torch::jit::mobile::serialization::CreateDevice(
      _fbb,
      str__);
}

// 定义 List 结构体，继承自 FlatBuffers 的 Table 类
struct List FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef ListBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ITEMS = 4,         // Vector<uint32_t> 类型在 VTable 中的偏移量
    VT_ANNOTATION_STR = 6 // String 类型在 VTable 中的偏移量
  };

  // 返回不可变的 items Vector<uint32_t> 指针
  const ::flatbuffers::Vector<uint32_t> *items() const {
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_ITEMS);
  }

  // 返回可变的 items Vector<uint32_t> 指针
  ::flatbuffers::Vector<uint32_t> *mutable_items() {
    return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_ITEMS);
  }

  // 返回不可变的 annotation_str String 指针
  const ::flatbuffers::String *annotation_str() const {
    return GetPointer<const ::flatbuffers::String *>(VT_ANNOTATION_STR);
  }

  // 返回可变的 annotation_str String 指针
  ::flatbuffers::String *mutable_annotation_str() {
    return GetPointer<::flatbuffers::String *>(VT_ANNOTATION_STR);
  }

  // 验证 List 对象的完整性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ITEMS) &&
           verifier.VerifyVector(items()) &&
           VerifyOffset(verifier, VT_ANNOTATION_STR) &&
           verifier.VerifyString(annotation_str()) &&
           verifier.EndTable();
  }
};

// ListBuilder 结构体，用于构建 List 对象
struct ListBuilder {
  typedef List Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;   // FlatBufferBuilder 引用
  ::flatbuffers::uoffset_t start_;          // 起始偏移量

  // 添加 items Vector<uint32_t> 的方法
  void add_items(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> items) {
    fbb_.AddOffset(List::VT_ITEMS, items);
  }

  // 添加 annotation_str String 的方法
  void add_annotation_str(::flatbuffers::Offset<::flatbuffers::String> annotation_str) {
    fbb_.AddOffset(List::VT_ANNOTATION_STR, annotation_str);
  }

  // ListBuilder 构造函数，初始化 FlatBufferBuilder 引用和起始偏移量
  explicit ListBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }

  // 完成 List 对象的构建，返回 List 的偏移量
  ::flatbuffers::Offset<List> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<List>(end);
    return o;
  }
};

// 创建 List 对象的函数，返回 List 的偏移量
inline ::flatbuffers::Offset<List> CreateList(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> items = 0,
    ::flatbuffers::Offset<::flatbuffers::String> annotation_str = 0) {
  // 创建 ListBuilder 对象
  ListBuilder builder_(_fbb);
  // 添加 annotation_str 和 items 到 ListBuilder
  builder_.add_annotation_str(annotation_str);
  builder_.add_items(items);
  // 调用 FlatBuffers 库中的 CreateList 函数，返回 List 的偏移量
  return torch::jit::mobile::serialization::CreateList(
      _fbb,
      items__,
      annotation_str__);
}

// 创建直接 List 对象的函数，返回 List 的偏移量
inline ::flatbuffers::Offset<List> CreateListDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint32_t> *items = nullptr,
    const char *annotation_str = nullptr) {
  // 如果提供了 items 则创建 Vector<uint32_t>，否则设置为 0
  auto items__ = items ? _fbb.CreateVector<uint32_t>(*items) : 0;
  // 如果提供了 annotation_str 则创建 FlatBuffers 字符串，否则设置为 0
  auto annotation_str__ = annotation_str ? _fbb.CreateString(annotation_str) : 0;
  // 调用 FlatBuffers 库中的 CreateList 函数，返回 List 的偏移量
  return torch::jit::mobile::serialization::CreateList(
      _fbb,
      items__,
      annotation_str__);
}
// 定义一个名为 IntList 的结构体，继承自 flatbuffers 的 Table 类
// FLATBUFFERS_FINAL_CLASS 表示这是一个最终类，不可被继承
struct IntList FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  // 定义一个内部类型 IntListBuilder 作为该结构体的建造器
  typedef IntListBuilder Builder;
  // 枚举定义了 FlatBuffersVTableOffset 类型，用于指定 VTable 中的偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ITEMS = 4  // VTable 中 items 字段的偏移量为 4
  };
  // 返回一个指向常量 flatbuffers::Vector<int64_t> 类型的指针，表示 items 字段的访问器
  const ::flatbuffers::Vector<int64_t> *items() const {
    return GetPointer<const ::flatbuffers::Vector<int64_t> *>(VT_ITEMS);
  }
  // 返回一个指向非常量 flatbuffers::Vector<int64_t> 类型的指针，表示可修改的 items 字段的访问器
  ::flatbuffers::Vector<int64_t> *mutable_items() {
    return GetPointer<::flatbuffers::Vector<int64_t> *>(VT_ITEMS);
  }
  // 验证函数，用于验证对象是否符合预期结构
  bool Verify(::flatbuffers::Verifier &verifier) const {
    // 验证表的起始
    return VerifyTableStart(verifier) &&
           // 验证 VT_ITEMS 字段的偏移量
           VerifyOffset(verifier, VT_ITEMS) &&
           // 验证 items 字段是否为有效的向量
           verifier.VerifyVector(items()) &&
           // 结束表的验证
           verifier.EndTable();
  }
};

// 定义一个名为 IntListBuilder 的结构体
struct IntListBuilder {
  // 定义类型别名 Table 为 IntList
  typedef IntList Table;
  // 引用 flatbuffers::FlatBufferBuilder 对象的引用 fbb_
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 表示创建表的起始偏移量
  ::flatbuffers::uoffset_t start_;
  // 添加 items 字段的方法，接受 flatbuffers::Offset<flatbuffers::Vector<int64_t>> 类型的参数 items
  void add_items(::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> items) {
    // 将 items 字段的偏移量和给定的 items 偏移量添加到 FlatBufferBuilder 中
    fbb_.AddOffset(IntList::VT_ITEMS, items);
  }
  // 构造函数，初始化成员变量 fbb_ 和 start_
  explicit IntListBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    // 调用 FlatBufferBuilder 的 StartTable 方法开始创建表
    start_ = fbb_.StartTable();
  }
  // 完成表的构造，返回表的偏移量
  ::flatbuffers::Offset<IntList> Finish() {
    // 调用 FlatBufferBuilder 的 EndTable 方法结束创建表，并获取结束时的偏移量
    const auto end = fbb_.EndTable(start_);
    // 将结束偏移量转换为 IntList 类型的偏移量，并返回
    auto o = ::flatbuffers::Offset<IntList>(end);
    return o;
  }
};

// 定义一个名为 CreateIntList 的函数模板，用于创建 IntList 对象
inline ::flatbuffers::Offset<IntList> CreateIntList(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> items = 0) {
  // 创建 IntListBuilder 对象 builder_
  IntListBuilder builder_(_fbb);
  // 调用 builder_ 的 add_items 方法，添加 items 字段
  builder_.add_items(items);
  // 返回 builder_ 的 Finish 方法的结果，即创建好的 IntList 对象的偏移量
  return builder_.Finish();
}

// 定义一个名为 CreateIntListDirect 的函数模板，用于直接创建 IntList 对象
inline ::flatbuffers::Offset<IntList> CreateIntListDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<int64_t> *items = nullptr) {
  // 如果 items 非空，则创建 flatbuffers::Vector<int64_t> 类型的向量
  auto items__ = items ? _fbb.CreateVector<int64_t>(*items) : 0;
  // 调用 CreateIntList 函数创建 IntList 对象
  return CreateIntList(
      _fbb,
      items__);
}

// 定义一个名为 DoubleList 的结构体，继承自 flatbuffers 的 Table 类
struct DoubleList FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  // 定义一个内部类型 DoubleListBuilder 作为该结构体的建造器
  typedef DoubleListBuilder Builder;
  // 枚举定义了 FlatBuffersVTableOffset 类型，用于指定 VTable 中的偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ITEMS = 4  // VTable 中 items 字段的偏移量为 4
  };
  // 返回一个指向常量 flatbuffers::Vector<double> 类型的指针，表示 items 字段的访问器
  const ::flatbuffers::Vector<double> *items() const {
    return GetPointer<const ::flatbuffers::Vector<double> *>(VT_ITEMS);
  }
  // 返回一个指向非常量 flatbuffers::Vector<double> 类型的指针，表示可修改的 items 字段的访问器
  ::flatbuffers::Vector<double> *mutable_items() {
    return GetPointer<::flatbuffers::Vector<double> *>(VT_ITEMS);
  }
  // 验证函数，用于验证对象是否符合预期结构
  bool Verify(::flatbuffers::Verifier &verifier) const {
    // 验证表的起始
    return VerifyTableStart(verifier) &&
           // 验证 VT_ITEMS 字段的偏移量
           VerifyOffset(verifier, VT_ITEMS) &&
           // 验证 items 字段是否为有效的向量
           verifier.VerifyVector(items()) &&
           // 结束表的验证
           verifier.EndTable();
  }
};

// 定义一个名为 DoubleListBuilder 的结构体
struct DoubleListBuilder {
  // 定义类型别名 Table 为 DoubleList
  typedef DoubleList Table;
  // 引用 flatbuffers::FlatBufferBuilder 对象的引用 fbb_
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 表示创建表的起始偏移量
  ::flatbuffers::uoffset_t start_;
  // 添加 items 字段的方法，接受 flatbuffers::Offset<flatbuffers::Vector<double>> 类型的参数 items
  void add_items(::flatbuffers::Offset<::flatbuffers::Vector<double>> items) {
    // 将 items 字段的偏移量和给定的 items 偏移量添加到 FlatBufferBuilder 中
    fbb_.AddOffset(DoubleList::VT_ITEMS, items);
  }
  // 构造函数，初始化成员变量 fbb_ 和 start_
  explicit DoubleListBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    // 调用 FlatBufferBuilder 的 StartTable 方法开始创建表
    start_ = fbb_.StartTable();
  }
  // 完成表的构造，返回表的偏移量
  ::flatbuffers::Offset<DoubleList> Finish() {
    // 调用 FlatBufferBuilder 的 EndTable 方法结束创建表，并获取结束时的偏移量
    const auto end = fbb_.EndTable(start_);
    // 将结束偏移量转换为 DoubleList 类型的偏移量，并返回
    auto o = ::flatbuffers::Offset<DoubleList>(end);
    return o;
  }
};
    // 创建一个指向 DoubleList 结构的偏移量 o，其起始地址为 end
    auto o = ::flatbuffers::Offset<DoubleList>(end);
    // 返回这个偏移量 o
    return o;
};

// 创建一个 DoubleList 对象并返回其偏移量
inline ::flatbuffers::Offset<DoubleList> CreateDoubleList(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<double>> items = 0) {
  // 使用 FlatBufferBuilder 创建 DoubleListBuilder 对象
  DoubleListBuilder builder_(_fbb);
  // 添加 items 到 DoubleListBuilder 对象中
  builder_.add_items(items);
  // 完成构建 DoubleList 对象并返回其偏移量
  return builder_.Finish();
}

// 直接创建 DoubleList 对象并返回其偏移量
inline ::flatbuffers::Offset<DoubleList> CreateDoubleListDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<double> *items = nullptr) {
  // 如果 items 不为空，创建双精度向量，否则设置为 0
  auto items__ = items ? _fbb.CreateVector<double>(*items) : 0;
  // 调用 CreateDoubleList 函数并返回其结果
  return torch::jit::mobile::serialization::CreateDoubleList(
      _fbb,
      items__);
}

// 布尔类型列表的结构体定义
struct BoolList FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef BoolListBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ITEMS = 4
  };
  // 获取布尔类型向量的指针
  const ::flatbuffers::Vector<uint8_t> *items() const {
    return GetPointer<const ::flatbuffers::Vector<uint8_t> *>(VT_ITEMS);
  }
  // 获取可变布尔类型向量的指针
  ::flatbuffers::Vector<uint8_t> *mutable_items() {
    return GetPointer<::flatbuffers::Vector<uint8_t> *>(VT_ITEMS);
  }
  // 验证布尔类型列表的有效性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ITEMS) &&
           verifier.VerifyVector(items()) &&
           verifier.EndTable();
  }
};

// BoolListBuilder 结构体定义
struct BoolListBuilder {
  typedef BoolList Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  // 添加布尔类型向量到 BoolListBuilder 中
  void add_items(::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> items) {
    fbb_.AddOffset(BoolList::VT_ITEMS, items);
  }
  // BoolListBuilder 的构造函数，初始化 FlatBufferBuilder
  explicit BoolListBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    // 开始构建 BoolList 表
    start_ = fbb_.StartTable();
  }
  // 完成构建 BoolList 对象并返回其偏移量
  ::flatbuffers::Offset<BoolList> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<BoolList>(end);
    return o;
  }
};

// 创建 BoolList 对象并返回其偏移量
inline ::flatbuffers::Offset<BoolList> CreateBoolList(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> items = 0) {
  // 使用 FlatBufferBuilder 创建 BoolListBuilder 对象
  BoolListBuilder builder_(_fbb);
  // 添加 items 到 BoolListBuilder 对象中
  builder_.add_items(items);
  // 完成构建 BoolList 对象并返回其偏移量
  return builder_.Finish();
}

// 直接创建 BoolList 对象并返回其偏移量
inline ::flatbuffers::Offset<BoolList> CreateBoolListDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint8_t> *items = nullptr) {
  // 如果 items 不为空，创建无符号字节向量，否则设置为 0
  auto items__ = items ? _fbb.CreateVector<uint8_t>(*items) : 0;
  // 调用 CreateBoolList 函数并返回其结果
  return torch::jit::mobile::serialization::CreateBoolList(
      _fbb,
      items__);
}

// 元组类型的结构体定义
struct Tuple FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef TupleBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ITEMS = 4
  };
  // 获取无符号 32 位整数向量的指针
  const ::flatbuffers::Vector<uint32_t> *items() const {
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_ITEMS);
  }
  // 获取可变无符号 32 位整数向量的指针
  ::flatbuffers::Vector<uint32_t> *mutable_items() {
    return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_ITEMS);
  }
  // 验证元组类型的有效性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    // 开始验证表的起始位置，并验证 items 的偏移量和向量的有效性
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ITEMS) &&
           verifier.VerifyVector(items()) &&
           verifier.EndTable();
  }
};
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ITEMS) &&
           verifier.VerifyVector(items()) &&
           verifier.EndTable();


// 返回表的验证结果，要求满足以下条件：
// 1. 调用 VerifyTableStart(verifier) 进行表的起始验证
// 2. 调用 VerifyOffset(verifier, VT_ITEMS) 进行特定偏移量的验证
// 3. 调用 verifier.VerifyVector(items()) 验证 items() 返回的向量数据
// 4. 调用 verifier.EndTable() 结束表的验证过程
// 返回最终的验证结果
};

// 定义一个结构体 TupleBuilder，用于构建 FlatBuffer 中的 Tuple 表
struct TupleBuilder {
  typedef Tuple Table;
  ::flatbuffers::FlatBufferBuilder &fbb_; // 引用 FlatBuffer 构建器
  ::flatbuffers::uoffset_t start_; // 起始偏移量

  // 添加 items 到 Tuple 表中
  void add_items(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> items) {
    fbb_.AddOffset(Tuple::VT_ITEMS, items); // 在 FlatBuffer 中添加偏移量
  }

  // 构造函数，初始化 TupleBuilder 并设置起始表
  explicit TupleBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable(); // 设置表的起始位置
  }

  // 完成 Tuple 表的构建并返回偏移量
  ::flatbuffers::Offset<Tuple> Finish() {
    const auto end = fbb_.EndTable(start_); // 结束表的构建并获取结束位置
    auto o = ::flatbuffers::Offset<Tuple>(end); // 转换为 Tuple 的偏移量类型
    return o;
  }
};

// 创建 Tuple 的快捷方式函数，使用 TupleBuilder 构建并返回 Tuple 偏移量
inline ::flatbuffers::Offset<Tuple> CreateTuple(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> items = 0) {
  TupleBuilder builder_(_fbb); // 使用 FlatBuffer 构建器初始化 TupleBuilder
  builder_.add_items(items); // 添加 items 到 Tuple 表中
  return builder_.Finish(); // 完成 Tuple 表的构建并返回偏移量
}

// 直接创建 Tuple 的快捷方式函数，使用 FlatBuffer 构建器创建并返回 Tuple 偏移量
inline ::flatbuffers::Offset<Tuple> CreateTupleDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint32_t> *items = nullptr) {
  auto items__ = items ? _fbb.CreateVector<uint32_t>(*items) : 0; // 创建 items 的 FlatBuffer 向量
  return torch::jit::mobile::serialization::CreateTuple(
      _fbb,
      items__); // 调用 CreateTuple 函数创建并返回 Tuple 偏移量
}

// 定义结构体 Dict，继承自 FlatBuffers 的 Table 类
struct Dict FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef DictBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_KEYS = 4,
    VT_VALUES = 6,
    VT_ANNOTATION_STR = 8
  };

  // 返回 keys 的指针
  const ::flatbuffers::Vector<uint32_t> *keys() const {
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_KEYS);
  }

  // 返回可修改的 keys 的指针
  ::flatbuffers::Vector<uint32_t> *mutable_keys() {
    return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_KEYS);
  }

  // 返回 values 的指针
  const ::flatbuffers::Vector<uint32_t> *values() const {
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_VALUES);
  }

  // 返回可修改的 values 的指针
  ::flatbuffers::Vector<uint32_t> *mutable_values() {
    return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_VALUES);
  }

  // 返回 annotation_str 的指针
  const ::flatbuffers::String *annotation_str() const {
    return GetPointer<const ::flatbuffers::String *>(VT_ANNOTATION_STR);
  }

  // 返回可修改的 annotation_str 的指针
  ::flatbuffers::String *mutable_annotation_str() {
    return GetPointer<::flatbuffers::String *>(VT_ANNOTATION_STR);
  }

  // 验证 Dict 表的数据是否有效
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_KEYS) &&
           verifier.VerifyVector(keys()) &&
           VerifyOffset(verifier, VT_VALUES) &&
           verifier.VerifyVector(values()) &&
           VerifyOffset(verifier, VT_ANNOTATION_STR) &&
           verifier.VerifyString(annotation_str()) &&
           verifier.EndTable();
  }
};

// 定义 DictBuilder 结构体，用于构建 FlatBuffer 中的 Dict 表
struct DictBuilder {
  typedef Dict Table;
  ::flatbuffers::FlatBufferBuilder &fbb_; // 引用 FlatBuffer 构建器
  ::flatbuffers::uoffset_t start_; // 起始偏移量

  // 添加 keys 到 Dict 表中
  void add_keys(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> keys) {
    fbb_.AddOffset(Dict::VT_KEYS, keys); // 在 FlatBuffer 中添加 keys 的偏移量
  }

  // 添加 values 到 Dict 表中
  void add_values(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> values) {
    fbb_.AddOffset(Dict::VT_VALUES, values); // 在 FlatBuffer 中添加 values 的偏移量
  }
    // 添加一个 Offset 到字典的值字段
    fbb_.AddOffset(Dict::VT_VALUES, values);
  }
  // 添加一个 Offset 到字典的注解字符串字段
  void add_annotation_str(::flatbuffers::Offset<::flatbuffers::String> annotation_str) {
    fbb_.AddOffset(Dict::VT_ANNOTATION_STR, annotation_str);
  }
  // 显式构造函数，初始化 FlatBufferBuilder 和起始表格位置
  explicit DictBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  // 结束表格构建并返回字典对象的偏移量
  ::flatbuffers::Offset<Dict> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Dict>(end);
    return o;
  }
};

// 创建一个名为 Dict 的 flatbuffers 结构体，并返回其偏移量
inline ::flatbuffers::Offset<Dict> CreateDict(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> keys = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> values = 0,
    ::flatbuffers::Offset<::flatbuffers::String> annotation_str = 0) {
  // 使用 DictBuilder 对象包装 FlatBufferBuilder
  DictBuilder builder_(_fbb);
  // 添加 annotation_str 字段
  builder_.add_annotation_str(annotation_str);
  // 添加 values 字段
  builder_.add_values(values);
  // 添加 keys 字段
  builder_.add_keys(keys);
  // 返回创建好的 Dict 结构体的偏移量
  return builder_.Finish();
}

// 直接创建 Dict 结构体，并返回其偏移量
inline ::flatbuffers::Offset<Dict> CreateDictDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint32_t> *keys = nullptr,
    const std::vector<uint32_t> *values = nullptr,
    const char *annotation_str = nullptr) {
  // 如果 keys 不为空，创建 uint32_t 类型的向量
  auto keys__ = keys ? _fbb.CreateVector<uint32_t>(*keys) : 0;
  // 如果 values 不为空，创建 uint32_t 类型的向量
  auto values__ = values ? _fbb.CreateVector<uint32_t>(*values) : 0;
  // 如果 annotation_str 不为空，创建 String 类型的字符串
  auto annotation_str__ = annotation_str ? _fbb.CreateString(annotation_str) : 0;
  // 调用 torch::jit::mobile::serialization::CreateDict 函数创建 Dict 结构体并返回偏移量
  return torch::jit::mobile::serialization::CreateDict(
      _fbb,
      keys__,
      values__,
      annotation_str__);
}

// 定义一个名为 ObjectType 的结构体，继承自 flatbuffers::Table
struct ObjectType FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef ObjectTypeBuilder Builder;
  // 定义字段的偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_TYPE_NAME = 4,
    VT_TYPE = 6,
    VT_ATTR_NAMES = 8
  };
  // 返回 type_name 字段的指针
  const ::flatbuffers::String *type_name() const {
    return GetPointer<const ::flatbuffers::String *>(VT_TYPE_NAME);
  }
  // 返回可变的 type_name 字段的指针
  ::flatbuffers::String *mutable_type_name() {
    return GetPointer<::flatbuffers::String *>(VT_TYPE_NAME);
  }
  // 返回 type 字段的值，并转换为 torch::jit::mobile::serialization::TypeType 类型
  torch::jit::mobile::serialization::TypeType type() const {
    return static_cast<torch::jit::mobile::serialization::TypeType>(GetField<uint8_t>(VT_TYPE, 0));
  }
  // 修改 type 字段的值为给定的 _type，并返回操作是否成功
  bool mutate_type(torch::jit::mobile::serialization::TypeType _type = static_cast<torch::jit::mobile::serialization::TypeType>(0)) {
    return SetField<uint8_t>(VT_TYPE, static_cast<uint8_t>(_type), 0);
  }
  // 返回 attr_names 字段的指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>> *attr_names() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>> *>(VT_ATTR_NAMES);
  }
  // 返回可变的 attr_names 字段的指针
  ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>> *mutable_attr_names() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>> *>(VT_ATTR_NAMES);
  }
  // 验证对象的完整性，使用 verifier 进行验证
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_TYPE_NAME) &&
           // 验证 type_name 字段的字符串是否有效
           verifier.VerifyString(type_name()) &&
           // 验证 type 字段是否有效
           VerifyField<uint8_t>(verifier, VT_TYPE, 1) &&
           // 验证 attr_names 字段是否有效，并且其中的字符串是否有效
           VerifyOffset(verifier, VT_ATTR_NAMES) &&
           verifier.VerifyVector(attr_names()) &&
           verifier.VerifyVectorOfStrings(attr_names()) &&
           // 结束表的验证
           verifier.EndTable();
  }
};

// 定义 ObjectTypeBuilder 结构体
struct ObjectTypeBuilder {
  typedef ObjectType Table;
  // 引用到 flatbuffers::FlatBufferBuilder 对象
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 表的起始偏移量
  ::flatbuffers::uoffset_t start_;
  // 添加 type_name 字段到 builder
  void add_type_name(::flatbuffers::Offset<::flatbuffers::String> type_name) {
    fbb_.AddOffset(ObjectType::VT_TYPE_NAME, type_name);



  // 将类型名称添加到 flatbuffers 构建器中
  void add_type(torch::jit::mobile::serialization::TypeType type) {



    fbb_.AddElement<uint8_t>(ObjectType::VT_TYPE, static_cast<uint8_t>(type), 0);



  // 添加类型值（以字节表示）到 flatbuffers 构建器中
  void add_attr_names(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>> attr_names) {



    fbb_.AddOffset(ObjectType::VT_ATTR_NAMES, attr_names);



  // 添加属性名称向量的偏移量到 flatbuffers 构建器中
  explicit ObjectTypeBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {



    start_ = fbb_.StartTable();



  // 在 flatbuffers 构建器中开始一个新的表
  ::flatbuffers::Offset<ObjectType> Finish() {



    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<ObjectType>(end);
    return o;
  }



  // 完成并返回构建的 ObjectType 对象的偏移量
};

// 定义一个内联函数，用于创建 ObjectType 对象
inline ::flatbuffers::Offset<ObjectType> CreateObjectType(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> type_name = 0,
    torch::jit::mobile::serialization::TypeType type = torch::jit::mobile::serialization::TypeType::UNSET,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>> attr_names = 0) {
  
  // 使用 ObjectTypeBuilder 初始化对象构建器 builder_
  ObjectTypeBuilder builder_(_fbb);
  
  // 添加属性 attr_names 到 builder_
  builder_.add_attr_names(attr_names);
  
  // 添加属性 type_name 到 builder_
  builder_.add_type_name(type_name);
  
  // 添加属性 type 到 builder_
  builder_.add_type(type);
  
  // 返回构建器的 Finish() 方法的结果，即创建的 ObjectType 对象的偏移量
  return builder_.Finish();
}

// 定义一个内联函数，直接创建 ObjectType 对象
inline ::flatbuffers::Offset<ObjectType> CreateObjectTypeDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *type_name = nullptr,
    torch::jit::mobile::serialization::TypeType type = torch::jit::mobile::serialization::TypeType::UNSET,
    const std::vector<::flatbuffers::Offset<::flatbuffers::String>> *attr_names = nullptr) {
  
  // 创建类型名称的 flatbuffers 字符串，如果 type_name 为 nullptr，则创建一个空字符串
  auto type_name__ = type_name ? _fbb.CreateString(type_name) : 0;
  
  // 创建属性名称的 flatbuffers 向量，如果 attr_names 为 nullptr，则创建一个空向量
  auto attr_names__ = attr_names ? _fbb.CreateVector<::flatbuffers::Offset<::flatbuffers::String>>(*attr_names) : 0;
  
  // 调用 torch::jit::mobile::serialization::CreateObjectType 方法创建 ObjectType 对象
  return torch::jit::mobile::serialization::CreateObjectType(
      _fbb,
      type_name__,
      type,
      attr_names__);
}

// 定义一个结构体 Object，继承自 flatbuffers::Table
struct Object FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  
  // 定义枚举，表示 FlatBuffers 中的虚拟表偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_TYPE_INDEX = 4,
    VT_STATE = 6,
    VT_ATTRS = 8,
    VT_SETSTATE_FUNC = 10
  };
  
  // 获取对象的 type_index 属性值
  uint32_t type_index() const {
    return GetField<uint32_t>(VT_TYPE_INDEX, 0);
  }
  
  // 设置对象的 type_index 属性值
  bool mutate_type_index(uint32_t _type_index = 0) {
    return SetField<uint32_t>(VT_TYPE_INDEX, _type_index, 0);
  }
  
  // 获取对象的 state 属性值
  uint32_t state() const {
    return GetField<uint32_t>(VT_STATE, 0);
  }
  
  // 设置对象的 state 属性值
  bool mutate_state(uint32_t _state = 0) {
    return SetField<uint32_t>(VT_STATE, _state, 0);
  }
  
  // 获取对象的 attrs 属性值，返回一个指向 flatbuffers 向量的指针
  const ::flatbuffers::Vector<uint32_t> *attrs() const {
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_ATTRS);
  }
  
  // 获取对象的可变 attrs 属性值，返回一个指向 flatbuffers 向量的指针
  ::flatbuffers::Vector<uint32_t> *mutable_attrs() {
    return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_ATTRS);
  }
  
  // 获取对象的 setstate_func 属性值
  uint32_t setstate_func() const {
    return GetField<uint32_t>(VT_SETSTATE_FUNC, 0);
  }
  
  // 设置对象的 setstate_func 属性值
  bool mutate_setstate_func(uint32_t _setstate_func = 0) {
    return SetField<uint32_t>(VT_SETSTATE_FUNC, _setstate_func, 0);
  }
  
  // 验证对象的有效性，使用 verifier 进行验证
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint32_t>(verifier, VT_TYPE_INDEX, 4) &&
           VerifyField<uint32_t>(verifier, VT_STATE, 4) &&
           VerifyOffset(verifier, VT_ATTRS) &&
           verifier.VerifyVector(attrs()) &&
           VerifyField<uint32_t>(verifier, VT_SETSTATE_FUNC, 4) &&
           verifier.EndTable();
  }
};

// 定义一个 ObjectBuilder 结构体，用于构建 Object 对象
struct ObjectBuilder {
  
  // 定义 Table 类型为 Object
  typedef Object Table;
  
  // 引用 flatbuffers 构建器
  ::flatbuffers::FlatBufferBuilder &fbb_;
  
  // 对象的起始偏移量
  ::flatbuffers::uoffset_t start_;
  
  // 添加 type_index 属性到对象构建器
  void add_type_index(uint32_t type_index) {
    fbb_.AddElement<uint32_t>(Object::VT_TYPE_INDEX, type_index, 0);

# 向FlatBuffer构建器中添加一个uint32_t类型的元素，用于表示对象的类型索引。

  }
  void add_state(uint32_t state) {
    fbb_.AddElement<uint32_t>(Object::VT_STATE, state, 0);
  }

# 向FlatBuffer构建器中添加一个uint32_t类型的元素，用于表示对象的状态。

  void add_attrs(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> attrs) {
    fbb_.AddOffset(Object::VT_ATTRS, attrs);
  }

# 向FlatBuffer构建器中添加一个偏移量（offset），指向一个存储uint32_t类型元素的向量，用于表示对象的属性集合。

  void add_setstate_func(uint32_t setstate_func) {
    fbb_.AddElement<uint32_t>(Object::VT_SETSTATE_FUNC, setstate_func, 0);
  }

# 向FlatBuffer构建器中添加一个uint32_t类型的元素，用于表示对象的状态设置函数。

  explicit ObjectBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }

# 构造函数，初始化ObjectBuilder对象时，接收一个FlatBuffer构建器的引用，并设置内部成员fbb_为该引用。同时调用FlatBuffer构建器的StartTable方法开始创建表。

  ::flatbuffers::Offset<Object> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Object>(end);
    return o;
  }

# 完成方法，调用FlatBuffer构建器的EndTable方法结束表的创建，并返回表的偏移量作为Object类型的偏移量。
};

// 创建一个 Object 对象并返回其在 FlatBuffer 中的偏移量
inline ::flatbuffers::Offset<Object> CreateObject(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    uint32_t type_index = 0,
    uint32_t state = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> attrs = 0,
    uint32_t setstate_func = 0) {
  
  // 使用提供的 FlatBufferBuilder 创建 ObjectBuilder 对象
  ObjectBuilder builder_(_fbb);
  
  // 设置 ObjectBuilder 对象的属性
  builder_.add_setstate_func(setstate_func);
  builder_.add_attrs(attrs);
  builder_.add_state(state);
  builder_.add_type_index(type_index);
  
  // 完成并返回 ObjectBuilder 对象
  return builder_.Finish();
}

// 直接创建一个 Object 对象并返回其在 FlatBuffer 中的偏移量
inline ::flatbuffers::Offset<Object> CreateObjectDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    uint32_t type_index = 0,
    uint32_t state = 0,
    const std::vector<uint32_t> *attrs = nullptr,
    uint32_t setstate_func = 0) {
  
  // 如果提供了 attrs，使用 FlatBufferBuilder 创建相应的向量对象；否则设为 0
  auto attrs__ = attrs ? _fbb.CreateVector<uint32_t>(*attrs) : 0;
  
  // 调用另一个函数创建 Object 对象并返回
  return torch::jit::mobile::serialization::CreateObject(
      _fbb,
      type_index,
      state,
      attrs__,
      setstate_func);
}

// 枚举值结构体 EnumValue 的定义
struct EnumValue FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef EnumValueBuilder Builder;
  
  // 枚举类型中可能的 VTable 偏移量定义
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_TYPE_NAME = 4,
    VT_VALUE = 6
  };
  
  // 获取类型名称的方法，返回字符串指针
  const ::flatbuffers::String *type_name() const {
    return GetPointer<const ::flatbuffers::String *>(VT_TYPE_NAME);
  }
  
  // 获取可变类型名称的方法，返回字符串指针
  ::flatbuffers::String *mutable_type_name() {
    return GetPointer<::flatbuffers::String *>(VT_TYPE_NAME);
  }
  
  // 获取枚举值的方法，返回无符号整数
  uint32_t value() const {
    return GetField<uint32_t>(VT_VALUE, 0);
  }
  
  // 修改枚举值的方法，返回是否成功
  bool mutate_value(uint32_t _value = 0) {
    return SetField<uint32_t>(VT_VALUE, _value, 0);
  }
  
  // 验证方法，用于验证对象的正确性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_TYPE_NAME) &&
           verifier.VerifyString(type_name()) &&
           VerifyField<uint32_t>(verifier, VT_VALUE, 4) &&
           verifier.EndTable();
  }
};

// EnumValueBuilder 结构体定义
struct EnumValueBuilder {
  typedef EnumValue Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  
  // 添加类型名称的方法，将类型名称添加到 FlatBufferBuilder 中
  void add_type_name(::flatbuffers::Offset<::flatbuffers::String> type_name) {
    fbb_.AddOffset(EnumValue::VT_TYPE_NAME, type_name);
  }
  
  // 添加枚举值的方法，将值添加到 FlatBufferBuilder 中
  void add_value(uint32_t value) {
    fbb_.AddElement<uint32_t>(EnumValue::VT_VALUE, value, 0);
  }
  
  // EnumValueBuilder 的构造函数，初始化 FlatBufferBuilder
  explicit EnumValueBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();  // 开始创建 EnumValue 对象
  }
  
  // 完成创建 EnumValue 对象，返回偏移量
  ::flatbuffers::Offset<EnumValue> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<EnumValue>(end);
    return o;
  }
};

// 直接创建 EnumValue 对象并返回其在 FlatBuffer 中的偏移量
inline ::flatbuffers::Offset<EnumValue> CreateEnumValue(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> type_name = 0,
    uint32_t value = 0) {
  
  // 使用提供的 FlatBufferBuilder 创建 EnumValueBuilder 对象
  EnumValueBuilder builder_(_fbb);
  
  // 设置 EnumValueBuilder 对象的属性
  builder_.add_value(value);
  builder_.add_type_name(type_name);
  
  // 完成并返回 EnumValueBuilder 对象
  return builder_.Finish();
}

// 直接创建 EnumValue 对象并返回其在 FlatBuffer 中的偏移量
inline ::flatbuffers::Offset<EnumValue> CreateEnumValueDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *type_name = nullptr,
    uint32_t value = 0) {
  
  // 调用另一个函数创建 EnumValue 对象并返回
  return CreateEnumValue(
      _fbb,
      type_name ? _fbb.CreateString(type_name) : 0,
      value);
}
    // 定义一个名为 value 的无符号 32 位整数变量，初始化为 0
    uint32_t value = 0) {
    // 创建一个名为 type_name__ 的自动变量，若 type_name 非空，则使用 _fbb 创建一个字符串，否则初始化为 0
    auto type_name__ = type_name ? _fbb.CreateString(type_name) : 0;
    // 调用 torch::jit::mobile::serialization::CreateEnumValue 函数，传入 _fbb、type_name__ 和 value 作为参数，并返回其结果
    return torch::jit::mobile::serialization::CreateEnumValue(
        _fbb,
        type_name__,
        value);
}

// 定义一个结构体 Operator，继承自 flatbuffers::Table
struct Operator FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef OperatorBuilder Builder; // 使用 OperatorBuilder 作为 Builder 类型
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,                // 姓名在平板缓冲器虚拟表中的偏移量
    VT_OVERLOAD_NAME = 6,       // 重载名在平板缓冲器虚拟表中的偏移量
    VT_NUM_ARGS_SERIALIZED = 8  // 序列化参数数量在平板缓冲器虚拟表中的偏移量
  };
  // 返回不可变的姓名字符串指针
  const ::flatbuffers::String *name() const {
    return GetPointer<const ::flatbuffers::String *>(VT_NAME);
  }
  // 返回可变的姓名字符串指针
  ::flatbuffers::String *mutable_name() {
    return GetPointer<::flatbuffers::String *>(VT_NAME);
  }
  // 返回不可变的重载名字符串指针
  const ::flatbuffers::String *overload_name() const {
    return GetPointer<const ::flatbuffers::String *>(VT_OVERLOAD_NAME);
  }
  // 返回可变的重载名字符串指针
  ::flatbuffers::String *mutable_overload_name() {
    return GetPointer<::flatbuffers::String *>(VT_OVERLOAD_NAME);
  }
  // 返回序列化参数数量
  int32_t num_args_serialized() const {
    return GetField<int32_t>(VT_NUM_ARGS_SERIALIZED, -1);
  }
  // 修改序列化参数数量，返回是否成功
  bool mutate_num_args_serialized(int32_t _num_args_serialized = -1) {
    return SetField<int32_t>(VT_NUM_ARGS_SERIALIZED, _num_args_serialized, -1);
  }
  // 验证 Operator 对象的有效性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_OVERLOAD_NAME) &&
           verifier.VerifyString(overload_name()) &&
           VerifyField<int32_t>(verifier, VT_NUM_ARGS_SERIALIZED, 4) &&
           verifier.EndTable();
  }
};

// 定义 OperatorBuilder 结构体
struct OperatorBuilder {
  typedef Operator Table; // 使用 Operator 作为 Table 类型
  ::flatbuffers::FlatBufferBuilder &fbb_; // 引用 FlatBufferBuilder 对象
  ::flatbuffers::uoffset_t start_; // 起始偏移量

  // 添加姓名字段
  void add_name(::flatbuffers::Offset<::flatbuffers::String> name) {
    fbb_.AddOffset(Operator::VT_NAME, name);
  }
  // 添加重载名字段
  void add_overload_name(::flatbuffers::Offset<::flatbuffers::String> overload_name) {
    fbb_.AddOffset(Operator::VT_OVERLOAD_NAME, overload_name);
  }
  // 添加序列化参数数量字段
  void add_num_args_serialized(int32_t num_args_serialized) {
    fbb_.AddElement<int32_t>(Operator::VT_NUM_ARGS_SERIALIZED, num_args_serialized, -1);
  }
  // OperatorBuilder 构造函数，初始化 FlatBufferBuilder 和起始表
  explicit OperatorBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  // 完成构建 Operator 对象，返回偏移量
  ::flatbuffers::Offset<Operator> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Operator>(end);
    return o;
  }
};

// 创建 Operator 对象的函数，使用 OperatorBuilder 构建
inline ::flatbuffers::Offset<Operator> CreateOperator(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> name = 0,
    ::flatbuffers::Offset<::flatbuffers::String> overload_name = 0,
    int32_t num_args_serialized = -1) {
  OperatorBuilder builder_(_fbb); // 创建 OperatorBuilder 对象
  builder_.add_num_args_serialized(num_args_serialized); // 添加序列化参数数量
  builder_.add_overload_name(overload_name); // 添加重载名
  builder_.add_name(name); // 添加姓名
  return builder_.Finish(); // 完成构建并返回 Operator 偏移量
}

// 直接创建 Operator 对象的函数，使用字符串作为参数
inline ::flatbuffers::Offset<Operator> CreateOperatorDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const char *overload_name = nullptr,
    int32_t num_args_serialized = -1) {


# 定义一个 int32_t 类型的变量 num_args_serialized，初始化为 -1


  auto name__ = name ? _fbb.CreateString(name) : 0;


# 如果 name 非空，则使用 _fbb 对象的 CreateString 方法创建字符串，并赋值给 name__；否则赋值为 0


  auto overload_name__ = overload_name ? _fbb.CreateString(overload_name) : 0;


# 如果 overload_name 非空，则使用 _fbb 对象的 CreateString 方法创建字符串，并赋值给 overload_name__；否则赋值为 0


  return torch::jit::mobile::serialization::CreateOperator(
      _fbb,
      name__,
      overload_name__,
      num_args_serialized);


# 调用 torch::jit::mobile::serialization 命名空间中的 CreateOperator 函数，传入 _fbb 对象、name__ 字符串、overload_name__ 字符串和 num_args_serialized 变量作为参数，返回结果。
}

// 结构体 Arg，继承自 flatbuffers::Table，表示一个平面缓冲区中的表
struct Arg FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef ArgBuilder Builder; // 定义 Builder 为 ArgBuilder 类型
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,              // 姓名字段的虚表偏移量
    VT_TYPE = 6,              // 类型字段的虚表偏移量
    VT_DEFAULT_VALUE = 8      // 默认值字段的虚表偏移量
  };

  // 获取姓名字段的值，返回一个 const flatbuffers::String 指针
  const ::flatbuffers::String *name() const {
    return GetPointer<const ::flatbuffers::String *>(VT_NAME);
  }

  // 获取可变姓名字段的指针，返回一个 flatbuffers::String 指针
  ::flatbuffers::String *mutable_name() {
    return GetPointer<::flatbuffers::String *>(VT_NAME);
  }

  // 获取类型字段的值，返回一个 const flatbuffers::String 指针
  const ::flatbuffers::String *type() const {
    return GetPointer<const ::flatbuffers::String *>(VT_TYPE);
  }

  // 获取可变类型字段的指针，返回一个 flatbuffers::String 指针
  ::flatbuffers::String *mutable_type() {
    return GetPointer<::flatbuffers::String *>(VT_TYPE);
  }

  // 获取默认值字段的值，返回一个 uint32_t 类型的整数
  uint32_t default_value() const {
    return GetField<uint32_t>(VT_DEFAULT_VALUE, 0);
  }

  // 修改默认值字段的值，返回是否成功的布尔值
  bool mutate_default_value(uint32_t _default_value = 0) {
    return SetField<uint32_t>(VT_DEFAULT_VALUE, _default_value, 0);
  }

  // 验证表的有效性，使用给定的 verifier 进行验证，返回验证结果的布尔值
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&                  // 验证表头开始
           VerifyOffset(verifier, VT_NAME) &&              // 验证姓名字段的偏移量
           verifier.VerifyString(name()) &&                // 验证姓名字段的字符串
           VerifyOffset(verifier, VT_TYPE) &&              // 验证类型字段的偏移量
           verifier.VerifyString(type()) &&                // 验证类型字段的字符串
           VerifyField<uint32_t>(verifier, VT_DEFAULT_VALUE, 4) &&  // 验证默认值字段的偏移量和数值
           verifier.EndTable();                           // 结束表的验证
  }
};

// ArgBuilder 结构体，用于构建 Arg 对象
struct ArgBuilder {
  typedef Arg Table;                      // 表示要构建的目标表是 Arg
  ::flatbuffers::FlatBufferBuilder &fbb_; // 引用平面缓冲区构建器
  ::flatbuffers::uoffset_t start_;        // 起始偏移量

  // 添加姓名字段到构建器中，使用给定的 name 偏移量
  void add_name(::flatbuffers::Offset<::flatbuffers::String> name) {
    fbb_.AddOffset(Arg::VT_NAME, name);
  }

  // 添加类型字段到构建器中，使用给定的 type 偏移量
  void add_type(::flatbuffers::Offset<::flatbuffers::String> type) {
    fbb_.AddOffset(Arg::VT_TYPE, type);
  }

  // 添加默认值字段到构建器中，使用给定的 default_value 数值
  void add_default_value(uint32_t default_value) {
    fbb_.AddElement<uint32_t>(Arg::VT_DEFAULT_VALUE, default_value, 0);
  }

  // 显式构造函数，初始化引用的平面缓冲区构建器，并设置起始偏移量
  explicit ArgBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable(); // 开始构建表
  }

  // 完成构建过程，返回构建的 Arg 对象的偏移量
  ::flatbuffers::Offset<Arg> Finish() {
    const auto end = fbb_.EndTable(start_); // 结束构建表，并获取结束位置
    auto o = ::flatbuffers::Offset<Arg>(end); // 创建 Arg 对象的偏移量
    return o; // 返回偏移量
  }
};

// 创建 Arg 对象的函数，使用给定的参数构建平面缓冲区对象
inline ::flatbuffers::Offset<Arg> CreateArg(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> name = 0,
    ::flatbuffers::Offset<::flatbuffers::String> type = 0,
    uint32_t default_value = 0) {
  ArgBuilder builder_(_fbb); // 创建 ArgBuilder 对象
  builder_.add_default_value(default_value); // 添加默认值
  builder_.add_type(type); // 添加类型
  builder_.add_name(name); // 添加姓名
  return builder_.Finish(); // 完成构建并返回 Arg 对象的偏移量
}

// 直接创建 Arg 对象的函数，使用给定的参数构建平面缓冲区对象
inline ::flatbuffers::Offset<Arg> CreateArgDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const char *type = nullptr,
    uint32_t default_value = 0) {
  auto name__ = name ? _fbb.CreateString(name) : 0; // 创建姓名字符串偏移量
  auto type__ = type ? _fbb.CreateString(type) : 0; // 创建类型字符串偏移量
  return torch::jit::mobile::serialization::CreateArg( // 调用创建 Arg 对象的函数
      _fbb,
      name__, // 姓名偏移量
      type__, // 类型偏移量
      default_value); // 默认值
}
// 定义结构体 Schema，继承自 flatbuffers 的 Table 类
struct Schema FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  // 定义 SchemaBuilder 为 Builder 类型
  typedef SchemaBuilder Builder;
  // 定义枚举类型 FlatBuffersVTableOffset，表示各字段在虚拟表中的偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ARGUMENTS = 4,  // 表示参数列表在虚拟表中的偏移量为 4
    VT_RETURNS = 6     // 表示返回值列表在虚拟表中的偏移量为 6
  };
  
  // 返回不可变参数列表的指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *arguments() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *>(VT_ARGUMENTS);
  }
  
  // 返回可变参数列表的指针
  ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *mutable_arguments() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *>(VT_ARGUMENTS);
  }
  
  // 返回不可变返回值列表的指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *returns() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *>(VT_RETURNS);
  }
  
  // 返回可变返回值列表的指针
  ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *mutable_returns() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *>(VT_RETURNS);
  }
  
  // 验证 Schema 结构的完整性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ARGUMENTS) &&
           verifier.VerifyVector(arguments()) &&
           verifier.VerifyVectorOfTables(arguments()) &&
           VerifyOffset(verifier, VT_RETURNS) &&
           verifier.VerifyVector(returns()) &&
           verifier.VerifyVectorOfTables(returns()) &&
           verifier.EndTable();
  }
};

// 定义 SchemaBuilder 结构体
struct SchemaBuilder {
  // 定义 Table 类型为 Schema
  typedef Schema Table;
  // 引用 flatbuffers::FlatBufferBuilder 对象
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 起始偏移量
  ::flatbuffers::uoffset_t start_;
  
  // 添加参数列表到 FlatBufferBuilder
  void add_arguments(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>>> arguments) {
    fbb_.AddOffset(Schema::VT_ARGUMENTS, arguments);
  }
  
  // 添加返回值列表到 FlatBufferBuilder
  void add_returns(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>>> returns) {
    fbb_.AddOffset(Schema::VT_RETURNS, returns);
  }
  
  // SchemaBuilder 构造函数，初始化 fbb_ 和 start_
  explicit SchemaBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  
  // 完成 Schema 构建，返回 Schema 的偏移量
  ::flatbuffers::Offset<Schema> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Schema>(end);
    return o;
  }
};

// 创建 Schema 对象的便捷函数，返回 Schema 的偏移量
inline ::flatbuffers::Offset<Schema> CreateSchema(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>>> arguments = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>>> returns = 0) {
  SchemaBuilder builder_(_fbb);
  builder_.add_returns(returns);
  builder_.add_arguments(arguments);
  return builder_.Finish();
}

// 直接创建 Schema 对象的便捷函数
inline ::flatbuffers::Offset<Schema> CreateSchemaDirect(
    // 接收一个FlatBufferBuilder对象的引用作为参数，用于构建FlatBuffer
    ::flatbuffers::FlatBufferBuilder &_fbb,
    // 如果arguments不为nullptr，则使用_fbb创建一个包含torch::jit::mobile::serialization::Arg偏移量的向量；否则传递空向量
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *arguments = nullptr,
    // 如果returns不为nullptr，则使用_fbb创建一个包含torch::jit::mobile::serialization::Arg偏移量的向量；否则传递空向量
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>> *returns = nullptr) {
  // 根据传入的arguments是否为nullptr，决定创建空向量或者包含arguments内容的向量
  auto arguments__ = arguments ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>>(*arguments) : 0;
  // 根据传入的returns是否为nullptr，决定创建空向量或者包含returns内容的向量
  auto returns__ = returns ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::Arg>>(*returns) : 0;
  // 使用_fbb创建一个torch::jit::mobile::serialization::Schema对象，并传入构建好的arguments__和returns__向量
  return torch::jit::mobile::serialization::CreateSchema(
      _fbb,
      arguments__,
      returns__);
}
}

// 定义 DebugInfo 结构体，继承自 flatbuffers 的 Table 类
struct DebugInfo FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  // 定义 DebugInfoBuilder 类型为 Builder
  typedef DebugInfoBuilder Builder;
  // 枚举 FlatBuffersVTableOffset，指定 VT_DEBUG_HANDLE 的偏移量为 4
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DEBUG_HANDLE = 4
  };
  // 获取 debug_handle 的指针，返回一个 flatbuffers 的 Vector<int64_t> 类型的指针
  const ::flatbuffers::Vector<int64_t> *debug_handle() const {
    return GetPointer<const ::flatbuffers::Vector<int64_t> *>(VT_DEBUG_HANDLE);
  }
  // 获取可变的 debug_handle 指针，返回一个 flatbuffers 的 Vector<int64_t> 类型的指针
  ::flatbuffers::Vector<int64_t> *mutable_debug_handle() {
    return GetPointer<::flatbuffers::Vector<int64_t> *>(VT_DEBUG_HANDLE);
  }
  // 验证 DebugInfo 对象，使用 verifier 进行验证
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_DEBUG_HANDLE) &&
           verifier.VerifyVector(debug_handle()) &&
           verifier.EndTable();
  }
};

// 定义 DebugInfoBuilder 结构体
struct DebugInfoBuilder {
  // 定义 Table 类型为 DebugInfo
  typedef DebugInfo Table;
  // flatbuffers 的 FlatBufferBuilder 引用 fbb_
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 表示起始偏移量的 uoffset_t 类型 start_
  ::flatbuffers::uoffset_t start_;
  // 添加 debug_handle 方法，将 debug_handle 添加到 FlatBufferBuilder 中
  void add_debug_handle(::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> debug_handle) {
    fbb_.AddOffset(DebugInfo::VT_DEBUG_HANDLE, debug_handle);
  }
  // DebugInfoBuilder 的构造函数，初始化 fbb_ 和 start_
  explicit DebugInfoBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  // 完成构建 DebugInfo 对象，返回偏移量 Offset<DebugInfo>
  ::flatbuffers::Offset<DebugInfo> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<DebugInfo>(end);
    return o;
  }
};

// 创建 DebugInfo 对象的函数，使用 FlatBufferBuilder _fbb 构建
inline ::flatbuffers::Offset<DebugInfo> CreateDebugInfo(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> debug_handle = 0) {
  DebugInfoBuilder builder_(_fbb);
  builder_.add_debug_handle(debug_handle);
  return builder_.Finish();
}

// 直接创建 DebugInfo 对象的函数，使用 FlatBufferBuilder _fbb 和 std::vector<int64_t> debug_handle
inline ::flatbuffers::Offset<DebugInfo> CreateDebugInfoDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<int64_t> *debug_handle = nullptr) {
  // 创建 int64_t 类型的 vector debug_handle__
  auto debug_handle__ = debug_handle ? _fbb.CreateVector<int64_t>(*debug_handle) : 0;
  // 调用 CreateDebugInfo 函数创建 DebugInfo 对象
  return torch::jit::mobile::serialization::CreateDebugInfo(
      _fbb,
      debug_handle__);
}

// 定义 Function 结构体，继承自 flatbuffers 的 Table 类
struct Function FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  // 定义 FunctionBuilder 类型为 Builder
  typedef FunctionBuilder Builder;
  // 枚举 FlatBuffersVTableOffset，指定各个成员的偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_QN = 4,
    VT_INSTRUCTIONS = 6,
    VT_OPERATORS = 8,
    VT_CONSTANTS = 10,
    VT_TYPE_ANNOTATIONS = 12,
    VT_REGISTER_SIZE = 14,
    VT_SCHEMA = 16,
    VT_DEBUG_INFO = 18,
    VT_CLASS_TYPE = 20
  };
  // 获取 qn 的指针，返回一个 flatbuffers 的 String 类型的指针
  const ::flatbuffers::String *qn() const {
    return GetPointer<const ::flatbuffers::String *>(VT_QN);
  }
  // 获取可变的 qn 指针，返回一个 flatbuffers 的 String 类型的指针
  ::flatbuffers::String *mutable_qn() {
    return GetPointer<::flatbuffers::String *>(VT_QN);
  }
  // 获取 instructions 的指针，返回一个 flatbuffers 的 Vector<const torch::jit::mobile::serialization::Instruction *> 类型的指针
  const ::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *> *instructions() const {
    return GetPointer<const ::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *> *>(VT_INSTRUCTIONS);
  }
  // 获取可变的 instructions 指针，返回一个 flatbuffers 的 Vector<const torch::jit::mobile::serialization::Instruction *> 类型的指针
  ::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *> *mutable_instructions() {

    return GetPointer<::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *> *>(VT_INSTRUCTIONS);
  }

  // 其他成员的获取和可变指针函数，略去部分以保持注释简洁
  return GetPointer<::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *> *>(VT_INSTRUCTIONS);

# 返回指向指令向量的指针，这些指令是不可变的

  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>> *operators() const {

# 返回指向操作符向量的指针，这些操作符是不可变的
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>> *>(VT_OPERATORS);
  }

# 返回指向操作符向量的指针，这些操作符是可变的
  ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>> *mutable_operators() {

# 返回指向操作符向量的指针，这些操作符是可变的
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>> *>(VT_OPERATORS);
  }

# 返回指向常量向量的指针，这些常量是不可变的
  const ::flatbuffers::Vector<uint32_t> *constants() const {

# 返回指向常量向量的指针，这些常量是可变的
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_CONSTANTS);
  }

# 返回指向常量向量的指针，这些常量是可变的
  ::flatbuffers::Vector<uint32_t> *mutable_constants() {

# 返回指向类型注解向量的指针，这些类型注解是不可变的
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>> *>(VT_TYPE_ANNOTATIONS);
  }

# 返回指向类型注解向量的指针，这些类型注解是可变的
  ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>> *mutable_type_annotations() {

# 返回寄存器大小的字段值，这个大小是不可变的
    return GetField<int32_t>(VT_REGISTER_SIZE, 0);
  }

# 设置寄存器大小的字段值，这个大小是可变的
  bool mutate_register_size(int32_t _register_size = 0) {

# 返回指向模式的指针，这个模式是不可变的
    return GetPointer<const torch::jit::mobile::serialization::Schema *>(VT_SCHEMA);
  }

# 返回指向模式的指针，这个模式是可变的
  torch::jit::mobile::serialization::Schema *mutable_schema() {

# 返回指向调试信息的指针，这个调试信息是不可变的
    return GetPointer<const torch::jit::mobile::serialization::DebugInfo *>(VT_DEBUG_INFO);
  }

# 返回指向调试信息的指针，这个调试信息是可变的
  torch::jit::mobile::serialization::DebugInfo *mutable_debug_info() {

# 返回类类型的字段值，这个类类型是不可变的
    return GetField<uint32_t>(VT_CLASS_TYPE, 0);
  }

# 设置类类型的字段值，这个类类型是可变的
  bool mutate_class_type(uint32_t _class_type = 0) {

# 验证对象的有效性，给定的验证器
    return SetField<uint32_t>(VT_CLASS_TYPE, _class_type, 0);
  }
    # 返回一个布尔值，表示是否通过表的验证
    return VerifyTableStart(verifier) &&

           # 验证指定偏移量处的值是否正确，这里是字段VT_QN
           VerifyOffset(verifier, VT_QN) &&

           # 验证字符串是否符合预期，使用qn()函数返回的字符串
           verifier.VerifyString(qn()) &&

           # 再次验证指定偏移量处的值，这次是VT_INSTRUCTIONS字段
           VerifyOffset(verifier, VT_INSTRUCTIONS) &&

           # 验证向量（vector）类型的数据是否符合预期，使用instructions()返回的向量
           verifier.VerifyVector(instructions()) &&

           # 继续验证下一个偏移量处的值，这里是VT_OPERATORS字段
           VerifyOffset(verifier, VT_OPERATORS) &&

           # 再次验证向量类型的数据，使用operators()返回的向量
           verifier.VerifyVector(operators()) &&

           # 验证一组表的向量，这里使用operators()返回的向量
           verifier.VerifyVectorOfTables(operators()) &&

           # 验证下一个偏移量处的值，这次是VT_CONSTANTS字段
           VerifyOffset(verifier, VT_CONSTANTS) &&

           # 继续验证向量类型的数据，使用constants()返回的向量
           verifier.VerifyVector(constants()) &&

           # 再次验证下一个偏移量处的值，这次是VT_TYPE_ANNOTATIONS字段
           VerifyOffset(verifier, VT_TYPE_ANNOTATIONS) &&

           # 验证向量类型的数据，使用type_annotations()返回的向量
           verifier.VerifyVector(type_annotations()) &&

           # 继续验证向量类型的数据，使用type_annotations()返回的向量
           verifier.VerifyVectorOfStrings(type_annotations()) &&

           # 验证字段值是否符合预期，这里是一个int32_t类型的值，字段为VT_REGISTER_SIZE
           VerifyField<int32_t>(verifier, VT_REGISTER_SIZE, 4) &&

           # 继续验证下一个偏移量处的值，这次是VT_SCHEMA字段
           VerifyOffset(verifier, VT_SCHEMA) &&

           # 验证表的结构，使用schema()返回的表
           verifier.VerifyTable(schema()) &&

           # 继续验证下一个偏移量处的值，这次是VT_DEBUG_INFO字段
           VerifyOffset(verifier, VT_DEBUG_INFO) &&

           # 验证表的结构，使用debug_info()返回的表
           verifier.VerifyTable(debug_info()) &&

           # 验证字段值是否符合预期，这里是一个uint32_t类型的值，字段为VT_CLASS_TYPE
           VerifyField<uint32_t>(verifier, VT_CLASS_TYPE, 4) &&

           # 结束表的验证，返回最终的验证结果
           verifier.EndTable();
};

// 结构体定义结束

struct FunctionBuilder {
  // 定义内部类型 Table 为 Function
  typedef Function Table;
  // 引用 FlatBufferBuilder 对象
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 记录表的起始位置
  ::flatbuffers::uoffset_t start_;

  // 添加 qn 字段到 FlatBuffer 中
  void add_qn(::flatbuffers::Offset<::flatbuffers::String> qn) {
    fbb_.AddOffset(Function::VT_QN, qn);
  }

  // 添加 instructions 字段到 FlatBuffer 中
  void add_instructions(::flatbuffers::Offset<::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *>> instructions) {
    fbb_.AddOffset(Function::VT_INSTRUCTIONS, instructions);
  }

  // 添加 operators 字段到 FlatBuffer 中
  void add_operators(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>>> operators) {
    fbb_.AddOffset(Function::VT_OPERATORS, operators);
  }

  // 添加 constants 字段到 FlatBuffer 中
  void add_constants(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> constants) {
    fbb_.AddOffset(Function::VT_CONSTANTS, constants);
  }

  // 添加 type_annotations 字段到 FlatBuffer 中
  void add_type_annotations(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>> type_annotations) {
    fbb_.AddOffset(Function::VT_TYPE_ANNOTATIONS, type_annotations);
  }

  // 添加 register_size 字段到 FlatBuffer 中
  void add_register_size(int32_t register_size) {
    fbb_.AddElement<int32_t>(Function::VT_REGISTER_SIZE, register_size, 0);
  }

  // 添加 schema 字段到 FlatBuffer 中
  void add_schema(::flatbuffers::Offset<torch::jit::mobile::serialization::Schema> schema) {
    fbb_.AddOffset(Function::VT_SCHEMA, schema);
  }

  // 添加 debug_info 字段到 FlatBuffer 中
  void add_debug_info(::flatbuffers::Offset<torch::jit::mobile::serialization::DebugInfo> debug_info) {
    fbb_.AddOffset(Function::VT_DEBUG_INFO, debug_info);
  }

  // 添加 class_type 字段到 FlatBuffer 中
  void add_class_type(uint32_t class_type) {
    fbb_.AddElement<uint32_t>(Function::VT_CLASS_TYPE, class_type, 0);
  }

  // 构造函数，初始化 FunctionBuilder 对象
  explicit FunctionBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    // 开始构建 FlatBuffer 表
    start_ = fbb_.StartTable();
  }

  // 完成构建 FlatBuffer 表，返回 Function 对象的偏移量
  ::flatbuffers::Offset<Function> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Function>(end);
    return o;
  }
};

// 创建 Function 对象的辅助函数，返回 Function 对象的偏移量
inline ::flatbuffers::Offset<Function> CreateFunction(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> qn = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<const torch::jit::mobile::serialization::Instruction *>> instructions = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>>> operators = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> constants = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>> type_annotations = 0,
    int32_t register_size = 0,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::Schema> schema = 0,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::DebugInfo> debug_info = 0,


这段代码定义了一个 `FunctionBuilder` 结构体和一个用于创建 `Function` 对象的辅助函数 `CreateFunction`。 `FunctionBuilder` 提供了方法来构建 FlatBuffer 中的不同字段，而 `CreateFunction` 函数用于方便地创建 `Function` 对象并返回其偏移量。
    # 创建一个 FunctionBuilder 对象，用于构建函数定义
    FunctionBuilder builder_(_fbb);
    # 将 class_type 添加到函数构建器中，指定函数的类别
    builder_.add_class_type(class_type);
    # 将 debug_info 添加到函数构建器中，包含函数的调试信息
    builder_.add_debug_info(debug_info);
    # 将 schema 添加到函数构建器中，描述函数的结构或模式
    builder_.add_schema(schema);
    # 将 register_size 添加到函数构建器中，指定函数的寄存器大小
    builder_.add_register_size(register_size);
    # 将 type_annotations 添加到函数构建器中，包含函数参数或返回值的类型注解
    builder_.add_type_annotations(type_annotations);
    # 将 constants 添加到函数构建器中，包含函数所使用的常量
    builder_.add_constants(constants);
    # 将 operators 添加到函数构建器中，描述函数的操作或操作集合
    builder_.add_operators(operators);
    # 将 instructions 添加到函数构建器中，包含函数的指令序列
    builder_.add_instructions(instructions);
    # 将 qn 添加到函数构建器中，可能是指定函数的其他信息
    builder_.add_qn(qn);
    # 完成函数构建过程并返回构建的函数对象
    return builder_.Finish();
}

// 创建一个直接生成 Function 对象的函数
inline ::flatbuffers::Offset<Function> CreateFunctionDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *qn = nullptr,
    const std::vector<torch::jit::mobile::serialization::Instruction> *instructions = nullptr,
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>> *operators = nullptr,
    const std::vector<uint32_t> *constants = nullptr,
    const std::vector<::flatbuffers::Offset<::flatbuffers::String>> *type_annotations = nullptr,
    int32_t register_size = 0,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::Schema> schema = 0,
    ::flatbuffers::Offset<torch::jit::mobile::serialization::DebugInfo> debug_info = 0,
    uint32_t class_type = 0) {
  // 创建 FlatBuffers 字符串对象，如果 qn 不为空，则使用 qn 创建字符串对象，否则为 0
  auto qn__ = qn ? _fbb.CreateString(qn) : 0;
  // 创建指令向量的 FlatBuffers 结构体向量对象，如果 instructions 不为空，则使用 instructions 创建，否则为 0
  auto instructions__ = instructions ? _fbb.CreateVectorOfStructs<torch::jit::mobile::serialization::Instruction>(*instructions) : 0;
  // 创建操作符向量的 FlatBuffers 偏移量向量对象，如果 operators 不为空，则使用 operators 创建，否则为 0
  auto operators__ = operators ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::Operator>>(*operators) : 0;
  // 创建常量向量的 FlatBuffers uint32_t 向量对象，如果 constants 不为空，则使用 constants 创建，否则为 0
  auto constants__ = constants ? _fbb.CreateVector<uint32_t>(*constants) : 0;
  // 创建类型注解向量的 FlatBuffers 字符串偏移量向量对象，如果 type_annotations 不为空，则使用 type_annotations 创建，否则为 0
  auto type_annotations__ = type_annotations ? _fbb.CreateVector<::flatbuffers::Offset<::flatbuffers::String>>(*type_annotations) : 0;
  // 调用 Torch JIT Mobile 序列化中的 CreateFunction 函数，返回创建的 Function 对象的偏移量
  return torch::jit::mobile::serialization::CreateFunction(
      _fbb,
      qn__,
      instructions__,
      operators__,
      constants__,
      type_annotations__,
      register_size,
      schema,
      debug_info,
      class_type);
}

// 定义一个存储数据的结构体 StorageData
struct StorageData FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef StorageDataBuilder Builder;
  // 定义存储数据的 FlatBuffers VTable 偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DATA = 4
  };
  // 获取数据的指针方法，返回一个指向数据的常量指针
  const ::flatbuffers::Vector<uint8_t> *data() const {
    return GetPointer<const ::flatbuffers::Vector<uint8_t> *>(VT_DATA);
  }
  // 获取可变数据的方法，返回一个指向数据的可变指针
  ::flatbuffers::Vector<uint8_t> *mutable_data() {
    return GetPointer<::flatbuffers::Vector<uint8_t> *>(VT_DATA);
  }
  // 验证方法，验证表的起始和数据的正确性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_DATA) &&
           verifier.VerifyVector(data()) &&
           verifier.EndTable();
  }
};

// 定义 StorageData 的 Builder 结构体
struct StorageDataBuilder {
  typedef StorageData Table;
  // FlatBuffers 构建器的引用和起始偏移量
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  // 添加数据的方法，将数据偏移量添加到 StorageData 中
  void add_data(::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> data) {
    fbb_.AddOffset(StorageData::VT_DATA, data);
  }
  // 显式构造函数，初始化 FlatBufferBuilder 并开始表的构建
  explicit StorageDataBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  // 完成方法，结束表的构建并返回 StorageData 对象的偏移量
  ::flatbuffers::Offset<StorageData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<StorageData>(end);
    return o;
  }
}

// 创建 StorageData 对象的函数
inline ::flatbuffers::Offset<StorageData> CreateStorageData(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    // 创建一个存储数据的构建器对象，使用_fbb作为构建器的参数
    StorageDataBuilder builder_(_fbb);
    // 向构建器中添加数据，使用data作为数据
    builder_.add_data(data);
    // 完成构建器的构建过程，并返回最终的数据对象
    return builder_.Finish();
}

// 定义一个内联函数，用于直接创建 StorageData 的 FlatBuffers 对象
inline ::flatbuffers::Offset<StorageData> CreateStorageDataDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint8_t> *data = nullptr) {
  
  // 如果提供了数据指针，强制对齐数据向量的大小
  if (data) { _fbb.ForceVectorAlignment(data->size(), sizeof(uint8_t), 16); }
  
  // 根据数据指针创建 FlatBuffers 中的向量对象
  auto data__ = data ? _fbb.CreateVector<uint8_t>(*data) : 0;
  
  // 调用 torch::jit::mobile::serialization 命名空间下的 CreateStorageData 函数，返回 StorageData 对象的偏移量
  return torch::jit::mobile::serialization::CreateStorageData(
      _fbb,
      data__);
}

// 定义一个结构体 IValue，继承自 flatbuffers::Table
struct IValue FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  
  // 定义内部类型别名 Builder
  typedef IValueBuilder Builder;
  
  // 定义 FlatBuffers 的 VTable 偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VAL_TYPE = 4, // 指定值类型字段的 VTable 偏移量
    VT_VAL = 6       // 指定值字段的 VTable 偏移量
  };
  
  // 返回 IValueUnion 类型的值类型
  torch::jit::mobile::serialization::IValueUnion val_type() const {
    return static_cast<torch::jit::mobile::serialization::IValueUnion>(GetField<uint8_t>(VT_VAL_TYPE, 0));
  }
  
  // 返回值字段的指针
  const void *val() const {
    return GetPointer<const void *>(VT_VAL);
  }
  
  // 模板方法，返回特定类型 T 的值指针
  template<typename T> const T *val_as() const;
  
  // 返回值作为 Int 类型的指针
  const torch::jit::mobile::serialization::Int *val_as_Int() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Int ? static_cast<const torch::jit::mobile::serialization::Int *>(val()) : nullptr;
  }
  
  // 返回值作为 Bool 类型的指针
  const torch::jit::mobile::serialization::Bool *val_as_Bool() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Bool ? static_cast<const torch::jit::mobile::serialization::Bool *>(val()) : nullptr;
  }
  
  // 返回值作为 Double 类型的指针
  const torch::jit::mobile::serialization::Double *val_as_Double() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Double ? static_cast<const torch::jit::mobile::serialization::Double *>(val()) : nullptr;
  }
  
  // 返回值作为 ComplexDouble 类型的指针
  const torch::jit::mobile::serialization::ComplexDouble *val_as_ComplexDouble() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::ComplexDouble ? static_cast<const torch::jit::mobile::serialization::ComplexDouble *>(val()) : nullptr;
  }
  
  // 返回值作为 TensorMetadata 类型的指针
  const torch::jit::mobile::serialization::TensorMetadata *val_as_TensorMetadata() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::TensorMetadata ? static_cast<const torch::jit::mobile::serialization::TensorMetadata *>(val()) : nullptr;
  }
  
  // 返回值作为 String 类型的指针
  const torch::jit::mobile::serialization::String *val_as_String() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::String ? static_cast<const torch::jit::mobile::serialization::String *>(val()) : nullptr;
  }
  
  // 返回值作为 List 类型的指针
  const torch::jit::mobile::serialization::List *val_as_List() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::List ? static_cast<const torch::jit::mobile::serialization::List *>(val()) : nullptr;
  }
  
  // 返回值作为 Tuple 类型的指针
  const torch::jit::mobile::serialization::Tuple *val_as_Tuple() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Tuple ? static_cast<const torch::jit::mobile::serialization::Tuple *>(val()) : nullptr;
  }
  
  // 返回值作为 Dict 类型的指针
  const torch::jit::mobile::serialization::Dict *val_as_Dict() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Dict ? static_cast<const torch::jit::mobile::serialization::Dict *>(val()) : nullptr;
  }


# 如果当前的值类型是字典（Dict），则返回该值的指针类型为 torch::jit::mobile::serialization::Dict*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::Object *val_as_Object() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Object ? static_cast<const torch::jit::mobile::serialization::Object *>(val()) : nullptr;
  }


# 如果当前的值类型是对象（Object），则返回该值的指针类型为 torch::jit::mobile::serialization::Object*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::IntList *val_as_IntList() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::IntList ? static_cast<const torch::jit::mobile::serialization::IntList *>(val()) : nullptr;
  }


# 如果当前的值类型是整数列表（IntList），则返回该值的指针类型为 torch::jit::mobile::serialization::IntList*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::DoubleList *val_as_DoubleList() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::DoubleList ? static_cast<const torch::jit::mobile::serialization::DoubleList *>(val()) : nullptr;
  }


# 如果当前的值类型是双精度浮点数列表（DoubleList），则返回该值的指针类型为 torch::jit::mobile::serialization::DoubleList*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::BoolList *val_as_BoolList() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::BoolList ? static_cast<const torch::jit::mobile::serialization::BoolList *>(val()) : nullptr;
  }


# 如果当前的值类型是布尔值列表（BoolList），则返回该值的指针类型为 torch::jit::mobile::serialization::BoolList*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::Device *val_as_Device() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Device ? static_cast<const torch::jit::mobile::serialization::Device *>(val()) : nullptr;
  }


# 如果当前的值类型是设备（Device），则返回该值的指针类型为 torch::jit::mobile::serialization::Device*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::EnumValue *val_as_EnumValue() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::EnumValue ? static_cast<const torch::jit::mobile::serialization::EnumValue *>(val()) : nullptr;
  }


# 如果当前的值类型是枚举值（EnumValue），则返回该值的指针类型为 torch::jit::mobile::serialization::EnumValue*；否则返回空指针nullptr。
const torch::jit::mobile::serialization::Function *val_as_Function() const {
    return val_type() == torch::jit::mobile::serialization::IValueUnion::Function ? static_cast<const torch::jit::mobile::serialization::Function *>(val()) : nullptr;
  }


# 如果当前的值类型是函数（Function），则返回该值的指针类型为 torch::jit::mobile::serialization::Function*；否则返回空指针nullptr。
void *mutable_val() {
    return GetPointer<void *>(VT_VAL);
  }


# 返回可变类型的值的指针，使用 VT_VAL 标识符获取。
bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint8_t>(verifier, VT_VAL_TYPE, 1) &&
           VerifyOffset(verifier, VT_VAL) &&
           VerifyIValueUnion(verifier, val(), val_type()) &&
           verifier.EndTable();
  }


# 使用 flatbuffers 的验证器检查当前对象的有效性：
# - 验证表的起始
# - 验证字段的类型为 uint8_t
# - 验证偏移量为 VT_VAL
# - 验证 IValueUnion 中的值和类型
# - 结束表的验证
};

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Int
template<> inline const torch::jit::mobile::serialization::Int *IValue::val_as<torch::jit::mobile::serialization::Int>() const {
  return val_as_Int();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Bool
template<> inline const torch::jit::mobile::serialization::Bool *IValue::val_as<torch::jit::mobile::serialization::Bool>() const {
  return val_as_Bool();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Double
template<> inline const torch::jit::mobile::serialization::Double *IValue::val_as<torch::jit::mobile::serialization::Double>() const {
  return val_as_Double();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::ComplexDouble
template<> inline const torch::jit::mobile::serialization::ComplexDouble *IValue::val_as<torch::jit::mobile::serialization::ComplexDouble>() const {
  return val_as_ComplexDouble();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::TensorMetadata
template<> inline const torch::jit::mobile::serialization::TensorMetadata *IValue::val_as<torch::jit::mobile::serialization::TensorMetadata>() const {
  return val_as_TensorMetadata();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::String
template<> inline const torch::jit::mobile::serialization::String *IValue::val_as<torch::jit::mobile::serialization::String>() const {
  return val_as_String();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::List
template<> inline const torch::jit::mobile::serialization::List *IValue::val_as<torch::jit::mobile::serialization::List>() const {
  return val_as_List();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Tuple
template<> inline const torch::jit::mobile::serialization::Tuple *IValue::val_as<torch::jit::mobile::serialization::Tuple>() const {
  return val_as_Tuple();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Dict
template<> inline const torch::jit::mobile::serialization::Dict *IValue::val_as<torch::jit::mobile::serialization::Dict>() const {
  return val_as_Dict();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Object
template<> inline const torch::jit::mobile::serialization::Object *IValue::val_as<torch::jit::mobile::serialization::Object>() const {
  return val_as_Object();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::IntList
template<> inline const torch::jit::mobile::serialization::IntList *IValue::val_as<torch::jit::mobile::serialization::IntList>() const {
  return val_as_IntList();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::DoubleList
template<> inline const torch::jit::mobile::serialization::DoubleList *IValue::val_as<torch::jit::mobile::serialization::DoubleList>() const {
  return val_as_DoubleList();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::BoolList
template<> inline const torch::jit::mobile::serialization::BoolList *IValue::val_as<torch::jit::mobile::serialization::BoolList>() const {
  return val_as_BoolList();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Device
template<> inline const torch::jit::mobile::serialization::Device *IValue::val_as<torch::jit::mobile::serialization::Device>() const {
  return val_as_Device();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::EnumValue
template<> inline const torch::jit::mobile::serialization::EnumValue *IValue::val_as<torch::jit::mobile::serialization::EnumValue>() const {
  return val_as_EnumValue();
}

// 定义模板特化，用于将 IValue 转换为 torch::jit::mobile::serialization::Function
template<> inline const torch::jit::mobile::serialization::Function *IValue::val_as<torch::jit::mobile::serialization::Function>() const {
  return val_as_Function();
}

// 定义结构体 IValueBuilder
struct IValueBuilder {
  typedef IValue Table;
  // 引用 FlatBufferBuilder 对象 fbb_
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 偏移量 start_
  ::flatbuffers::uoffset_t start_;
  // 添加 val_type 字段，类型为 torch::jit::mobile::serialization::IValueUnion
  void add_val_type(torch::jit::mobile::serialization::IValueUnion val_type) {
    # 在 FlatBuffers 构建器中添加一个元素，该元素的类型为 uint8_t，对应的值是 val_type，偏移量为 0
    fbb_.AddElement<uint8_t>(IValue::VT_VAL_TYPE, static_cast<uint8_t>(val_type), 0);
  }
  
  # 向 FlatBuffers 中添加一个指向值的偏移量，该值由参数 val 给出
  void add_val(::flatbuffers::Offset<void> val) {
    fbb_.AddOffset(IValue::VT_VAL, val);
  }
  
  # 显式构造函数，用给定的 FlatBufferBuilder 对象初始化 IValueBuilder
  explicit IValueBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    # 在 FlatBuffers 中开始一个新的表，并返回表的起始偏移量
    start_ = fbb_.StartTable();
  }
  
  # 完成构建并返回一个指向 IValue 对象的偏移量
  ::flatbuffers::Offset<IValue> Finish() {
    # 在 FlatBuffers 中结束当前表的构建，返回该表的结束偏移量
    const auto end = fbb_.EndTable(start_);
    # 将结束偏移量转换为指向 IValue 对象的偏移量
    auto o = ::flatbuffers::Offset<IValue>(end);
    return o;
  }
};

// 创建一个名为 CreateIValue 的内联函数，用于构造 IValue 对象
inline ::flatbuffers::Offset<IValue> CreateIValue(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    torch::jit::mobile::serialization::IValueUnion val_type = torch::jit::mobile::serialization::IValueUnion::NONE,
    ::flatbuffers::Offset<void> val = 0) {
  
  // 创建 IValueBuilder 对象，用于构建 IValue
  IValueBuilder builder_(_fbb);
  builder_.add_val(val); // 添加值
  builder_.add_val_type(val_type); // 添加值类型
  return builder_.Finish(); // 返回构建好的 IValue 对象
}

// 定义结构体 ExtraFile，继承自 flatbuffers::Table
struct ExtraFile FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef ExtraFileBuilder Builder;

  // 枚举 FlatBuffersVTableOffset 定义偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4, // 名称偏移量
    VT_CONTENT = 6 // 内容偏移量
  };

  // 返回名称的指针
  const ::flatbuffers::String *name() const {
    return GetPointer<const ::flatbuffers::String *>(VT_NAME);
  }

  // 返回可变名称的指针
  ::flatbuffers::String *mutable_name() {
    return GetPointer<::flatbuffers::String *>(VT_NAME);
  }

  // 返回内容的指针
  const ::flatbuffers::String *content() const {
    return GetPointer<const ::flatbuffers::String *>(VT_CONTENT);
  }

  // 返回可变内容的指针
  ::flatbuffers::String *mutable_content() {
    return GetPointer<::flatbuffers::String *>(VT_CONTENT);
  }

  // 验证函数，验证 ExtraFile 对象的完整性
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) && // 验证表的起始
           VerifyOffset(verifier, VT_NAME) && // 验证名称偏移量
           verifier.VerifyString(name()) && // 验证名称字符串
           VerifyOffset(verifier, VT_CONTENT) && // 验证内容偏移量
           verifier.VerifyString(content()) && // 验证内容字符串
           verifier.EndTable(); // 结束验证
  }
};

// ExtraFileBuilder 结构体，用于构建 ExtraFile 对象
struct ExtraFileBuilder {
  typedef ExtraFile Table;
  ::flatbuffers::FlatBufferBuilder &fbb_; // FlatBufferBuilder 引用
  ::flatbuffers::uoffset_t start_; // 起始偏移量

  // 添加名称
  void add_name(::flatbuffers::Offset<::flatbuffers::String> name) {
    fbb_.AddOffset(ExtraFile::VT_NAME, name);
  }

  // 添加内容
  void add_content(::flatbuffers::Offset<::flatbuffers::String> content) {
    fbb_.AddOffset(ExtraFile::VT_CONTENT, content);
  }

  // 构造函数，初始化 FlatBufferBuilder 和起始偏移量
  explicit ExtraFileBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable(); // 开始构建表
  }

  // 完成构建，返回 ExtraFile 对象的偏移量
  ::flatbuffers::Offset<ExtraFile> Finish() {
    const auto end = fbb_.EndTable(start_); // 结束表的构建
    auto o = ::flatbuffers::Offset<ExtraFile>(end);
    return o;
  }
};

// 创建 ExtraFile 对象的函数，接受 FlatBufferBuilder 和名称、内容的偏移量作为参数
inline ::flatbuffers::Offset<ExtraFile> CreateExtraFile(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> name = 0,
    ::flatbuffers::Offset<::flatbuffers::String> content = 0) {
  
  // 创建 ExtraFileBuilder 对象
  ExtraFileBuilder builder_(_fbb);
  builder_.add_content(content); // 添加内容
  builder_.add_name(name); // 添加名称
  return builder_.Finish(); // 返回构建好的 ExtraFile 对象
}

// 直接创建 ExtraFile 对象的函数，接受 FlatBufferBuilder 和名称、内容的字符串指针作为参数
inline ::flatbuffers::Offset<ExtraFile> CreateExtraFileDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const char *content = nullptr) {
  
  // 根据名称和内容创建字符串偏移量
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto content__ = content ? _fbb.CreateString(content) : 0;
  
  // 调用 CreateExtraFile 函数创建 ExtraFile 对象并返回
  return torch::jit::mobile::serialization::CreateExtraFile(
      _fbb,
      name__,
      content__);
}

// 结构体 Module，继承自 flatbuffers::Table
struct Module FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef ModuleBuilder Builder;

  // 枚举 FlatBuffersVTableOffset 定义偏移量
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BYTECODE_VERSION = 4,
    // 枚举类型定义，表示不同字段的索引值
    VT_EXTRA_FILES = 6,
    VT_METHODS = 8,
    VT_STATE_OBJ = 10,
    VT_IVALUES = 12,
    VT_STORAGE_DATA_SIZE = 14,
    VT_STORAGE_DATA = 16,
    VT_OBJECT_TYPES = 18,
    VT_JIT_SOURCES = 20,
    VT_JIT_CONSTANTS = 22,
    VT_OPERATOR_VERSION = 24,
    VT_MOBILE_IVALUE_SIZE = 26
  };
  // 返回字节码版本号
  uint32_t bytecode_version() const {
    return GetField<uint32_t>(VT_BYTECODE_VERSION, 0);
  }
  // 修改字节码版本号
  bool mutate_bytecode_version(uint32_t _bytecode_version = 0) {
    return SetField<uint32_t>(VT_BYTECODE_VERSION, _bytecode_version, 0);
  }
  // 返回附加文件的 vector 指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *extra_files() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *>(VT_EXTRA_FILES);
  }
  // 返回可变附加文件的 vector 指针
  ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *mutable_extra_files() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *>(VT_EXTRA_FILES);
  }
  // 返回方法的 vector 指针
  const ::flatbuffers::Vector<uint32_t> *methods() const {
    return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_METHODS);
  }
  // 返回可变方法的 vector 指针
  ::flatbuffers::Vector<uint32_t> *mutable_methods() {
    return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_METHODS);
  }
  // 返回状态对象字段
  uint32_t state_obj() const {
    return GetField<uint32_t>(VT_STATE_OBJ, 0);
  }
  // 修改状态对象字段
  bool mutate_state_obj(uint32_t _state_obj = 0) {
    return SetField<uint32_t>(VT_STATE_OBJ, _state_obj, 0);
  }
  // 返回 IValue 的 vector 指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>> *ivalues() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>> *>(VT_IVALUES);
  }
  // 返回可变 IValue 的 vector 指针
  ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>> *mutable_ivalues() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>> *>(VT_IVALUES);
  }
  // 返回存储数据大小字段
  int32_t storage_data_size() const {
    return GetField<int32_t>(VT_STORAGE_DATA_SIZE, 0);
  }
  // 修改存储数据大小字段
  bool mutate_storage_data_size(int32_t _storage_data_size = 0) {
    return SetField<int32_t>(VT_STORAGE_DATA_SIZE, _storage_data_size, 0);
  }
  // 返回存储数据的 vector 指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>> *storage_data() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>> *>(VT_STORAGE_DATA);
  }
  // 返回可变存储数据的 vector 指针
  ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>> *mutable_storage_data() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>> *>(VT_STORAGE_DATA);
  }
  // 返回对象类型的 vector 指针
  const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>> *object_types() const {
  // 返回指向常量对象类型向量的指针
  return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>> *>(VT_OBJECT_TYPES);
}
// 返回指向可变对象类型向量的指针
::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>> *mutable_object_types() {
  return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>> *>(VT_OBJECT_TYPES);
}
// 返回指向常量 JIT 源文件向量的指针
const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *jit_sources() const {
  return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *>(VT_JIT_SOURCES);
}
// 返回指向可变 JIT 源文件向量的指针
::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *mutable_jit_sources() {
  return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *>(VT_JIT_SOURCES);
}
// 返回指向常量 JIT 常量向量的指针
const ::flatbuffers::Vector<uint32_t> *jit_constants() const {
  return GetPointer<const ::flatbuffers::Vector<uint32_t> *>(VT_JIT_CONSTANTS);
}
// 返回指向可变 JIT 常量向量的指针
::flatbuffers::Vector<uint32_t> *mutable_jit_constants() {
  return GetPointer<::flatbuffers::Vector<uint32_t> *>(VT_JIT_CONSTANTS);
}
// 返回操作版本号
uint32_t operator_version() const {
  return GetField<uint32_t>(VT_OPERATOR_VERSION, 0);
}
// 修改操作版本号，返回是否修改成功
bool mutate_operator_version(uint32_t _operator_version = 0) {
  return SetField<uint32_t>(VT_OPERATOR_VERSION, _operator_version, 0);
}
// 返回移动 ivalue 大小
uint32_t mobile_ivalue_size() const {
  return GetField<uint32_t>(VT_MOBILE_IVALUE_SIZE, 0);
}
// 修改移动 ivalue 大小，返回是否修改成功
bool mutate_mobile_ivalue_size(uint32_t _mobile_ivalue_size = 0) {
  return SetField<uint32_t>(VT_MOBILE_IVALUE_SIZE, _mobile_ivalue_size, 0);
}
// 验证数据结构，使用 verifier 进行验证
bool Verify(::flatbuffers::Verifier &verifier) const {
    # 返回验证表格的起始位置是否正确，并使用逻辑与操作符连接下一个验证操作
    return VerifyTableStart(verifier) &&
           # 验证指定字段的值是否为 uint32_t 类型且长度为 4
           VerifyField<uint32_t>(verifier, VT_BYTECODE_VERSION, 4) &&
           # 验证偏移量，确保在指定的 VT_EXTRA_FILES 类型中
           VerifyOffset(verifier, VT_EXTRA_FILES) &&
           # 验证向量，确认 extra_files() 返回的内容是否符合预期
           verifier.VerifyVector(extra_files()) &&
           # 验证向量中的每个元素是否是表格，并符合预期的结构
           verifier.VerifyVectorOfTables(extra_files()) &&
           # 验证偏移量，确保在指定的 VT_METHODS 类型中
           VerifyOffset(verifier, VT_METHODS) &&
           # 验证向量，确认 methods() 返回的内容是否符合预期
           verifier.VerifyVector(methods()) &&
           # 验证指定字段的值是否为 uint32_t 类型且长度为 4
           VerifyField<uint32_t>(verifier, VT_STATE_OBJ, 4) &&
           # 验证偏移量，确保在指定的 VT_IVALUES 类型中
           VerifyOffset(verifier, VT_IVALUES) &&
           # 验证向量，确认 ivalues() 返回的内容是否符合预期
           verifier.VerifyVector(ivalues()) &&
           # 验证向量中的每个元素是否是表格，并符合预期的结构
           verifier.VerifyVectorOfTables(ivalues()) &&
           # 验证指定字段的值是否为 int32_t 类型且长度为 4
           VerifyField<int32_t>(verifier, VT_STORAGE_DATA_SIZE, 4) &&
           # 验证偏移量，确保在指定的 VT_STORAGE_DATA 类型中
           VerifyOffset(verifier, VT_STORAGE_DATA) &&
           # 验证向量，确认 storage_data() 返回的内容是否符合预期
           verifier.VerifyVector(storage_data()) &&
           # 验证向量中的每个元素是否是表格，并符合预期的结构
           verifier.VerifyVectorOfTables(storage_data()) &&
           # 验证偏移量，确保在指定的 VT_OBJECT_TYPES 类型中
           VerifyOffset(verifier, VT_OBJECT_TYPES) &&
           # 验证向量，确认 object_types() 返回的内容是否符合预期
           verifier.VerifyVector(object_types()) &&
           # 验证向量中的每个元素是否是表格，并符合预期的结构
           verifier.VerifyVectorOfTables(object_types()) &&
           # 验证偏移量，确保在指定的 VT_JIT_SOURCES 类型中
           VerifyOffset(verifier, VT_JIT_SOURCES) &&
           # 验证向量，确认 jit_sources() 返回的内容是否符合预期
           verifier.VerifyVector(jit_sources()) &&
           # 验证向量中的每个元素是否是表格，并符合预期的结构
           verifier.VerifyVectorOfTables(jit_sources()) &&
           # 验证偏移量，确保在指定的 VT_JIT_CONSTANTS 类型中
           VerifyOffset(verifier, VT_JIT_CONSTANTS) &&
           # 验证向量，确认 jit_constants() 返回的内容是否符合预期
           verifier.VerifyVector(jit_constants()) &&
           # 验证指定字段的值是否为 uint32_t 类型且长度为 4
           VerifyField<uint32_t>(verifier, VT_OPERATOR_VERSION, 4) &&
           # 验证指定字段的值是否为 uint32_t 类型且长度为 4
           VerifyField<uint32_t>(verifier, VT_MOBILE_IVALUE_SIZE, 4) &&
           # 结束验证表格，返回验证结果
           verifier.EndTable();
  }
};

// 结构体 ModuleBuilder 的定义
struct ModuleBuilder {
  // 定义 Table 类型为 Module
  typedef Module Table;
  // 引用 FlatBufferBuilder 对象
  ::flatbuffers::FlatBufferBuilder &fbb_;
  // 起始偏移量
  ::flatbuffers::uoffset_t start_;

  // 添加 bytecode_version 字段到 FlatBuffer 中
  void add_bytecode_version(uint32_t bytecode_version) {
    fbb_.AddElement<uint32_t>(Module::VT_BYTECODE_VERSION, bytecode_version, 0);
  }

  // 添加 extra_files 字段到 FlatBuffer 中
  void add_extra_files(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>>> extra_files) {
    fbb_.AddOffset(Module::VT_EXTRA_FILES, extra_files);
  }

  // 添加 methods 字段到 FlatBuffer 中
  void add_methods(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> methods) {
    fbb_.AddOffset(Module::VT_METHODS, methods);
  }

  // 添加 state_obj 字段到 FlatBuffer 中
  void add_state_obj(uint32_t state_obj) {
    fbb_.AddElement<uint32_t>(Module::VT_STATE_OBJ, state_obj, 0);
  }

  // 添加 ivalues 字段到 FlatBuffer 中
  void add_ivalues(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>>> ivalues) {
    fbb_.AddOffset(Module::VT_IVALUES, ivalues);
  }

  // 添加 storage_data_size 字段到 FlatBuffer 中
  void add_storage_data_size(int32_t storage_data_size) {
    fbb_.AddElement<int32_t>(Module::VT_STORAGE_DATA_SIZE, storage_data_size, 0);
  }

  // 添加 storage_data 字段到 FlatBuffer 中
  void add_storage_data(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>>> storage_data) {
    fbb_.AddOffset(Module::VT_STORAGE_DATA, storage_data);
  }

  // 添加 object_types 字段到 FlatBuffer 中
  void add_object_types(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>>> object_types) {
    fbb_.AddOffset(Module::VT_OBJECT_TYPES, object_types);
  }

  // 添加 jit_sources 字段到 FlatBuffer 中
  void add_jit_sources(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>>> jit_sources) {
    fbb_.AddOffset(Module::VT_JIT_SOURCES, jit_sources);
  }

  // 添加 jit_constants 字段到 FlatBuffer 中
  void add_jit_constants(::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> jit_constants) {
    fbb_.AddOffset(Module::VT_JIT_CONSTANTS, jit_constants);
  }

  // 添加 operator_version 字段到 FlatBuffer 中
  void add_operator_version(uint32_t operator_version) {
    fbb_.AddElement<uint32_t>(Module::VT_OPERATOR_VERSION, operator_version, 0);
  }

  // 添加 mobile_ivalue_size 字段到 FlatBuffer 中
  void add_mobile_ivalue_size(uint32_t mobile_ivalue_size) {
    fbb_.AddElement<uint32_t>(Module::VT_MOBILE_IVALUE_SIZE, mobile_ivalue_size, 0);
  }

  // ModuleBuilder 的构造函数，初始化 fbb_ 和 start_
  explicit ModuleBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }

  // 完成 ModuleBuilder 的构建，返回 Module 的偏移量
  ::flatbuffers::Offset<Module> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Module>(end);
    return o;
  }
};

// 创建 Module 对象的函数，设置默认参数和字段到 FlatBuffer 中
inline ::flatbuffers::Offset<Module> CreateModule(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    uint32_t bytecode_version = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>>> extra_files = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> methods = 0,
    uint32_t state_obj = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>>> ivalues = 0,
    int32_t storage_data_size = 0,
    // 声明和初始化一个 flatbuffers 结构体偏移量，表示存储数据
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>>> storage_data = 0,
    // 声明和初始化一个 flatbuffers 结构体偏移量，表示对象类型
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>>> object_types = 0,
    // 声明和初始化一个 flatbuffers 结构体偏移量，表示 JIT 源文件
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>>> jit_sources = 0,
    // 声明和初始化一个 flatbuffers 结构体偏移量，表示 JIT 常量
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> jit_constants = 0,
    // 声明和初始化一个 uint32_t 变量，表示运算符版本号
    uint32_t operator_version = 0,
    // 声明和初始化一个 uint32_t 变量，表示移动 ivalue 的大小
    uint32_t mobile_ivalue_size = 0) {
  // 创建 ModuleBuilder 对象，传入 flatbuffers 构建器对象 _fbb
  ModuleBuilder builder_(_fbb);
  // 将移动 ivalue 大小添加到构建器
  builder_.add_mobile_ivalue_size(mobile_ivalue_size);
  // 将运算符版本号添加到构建器
  builder_.add_operator_version(operator_version);
  // 将 JIT 常量添加到构建器
  builder_.add_jit_constants(jit_constants);
  // 将 JIT 源文件添加到构建器
  builder_.add_jit_sources(jit_sources);
  // 将对象类型添加到构建器
  builder_.add_object_types(object_types);
  // 将存储数据添加到构建器
  builder_.add_storage_data(storage_data);
  // 添加存储数据大小到构建器
  builder_.add_storage_data_size(storage_data_size);
  // 添加 ivalue 列表到构建器
  builder_.add_ivalues(ivalues);
  // 添加状态对象到构建器
  builder_.add_state_obj(state_obj);
  // 添加方法列表到构建器
  builder_.add_methods(methods);
  // 添加额外文件列表到构建器
  builder_.add_extra_files(extra_files);
  // 添加字节码版本到构建器
  builder_.add_bytecode_version(bytecode_version);
  // 返回构建器的完成对象
  return builder_.Finish();
}
// 创建一个直接的模块对象并返回其偏移量，使用FlatBuffer构建器_fbb
inline ::flatbuffers::Offset<Module> CreateModuleDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    // 字节码版本号，默认为0
    uint32_t bytecode_version = 0,
    // 额外文件的向量指针，可以为空
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *extra_files = nullptr,
    // 方法的向量指针，可以为空
    const std::vector<uint32_t> *methods = nullptr,
    // 状态对象，默认为0
    uint32_t state_obj = 0,
    // IValue对象的向量指针，可以为空
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>> *ivalues = nullptr,
    // 存储数据大小，默认为0
    int32_t storage_data_size = 0,
    // 存储数据的向量指针，可以为空
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>> *storage_data = nullptr,
    // 对象类型的向量指针，可以为空
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>> *object_types = nullptr,
    // JIT源文件的向量指针，可以为空
    const std::vector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>> *jit_sources = nullptr,
    // JIT常量的向量指针，可以为空
    const std::vector<uint32_t> *jit_constants = nullptr,
    // 操作符版本，默认为0
    uint32_t operator_version = 0,
    // 移动IValue大小，默认为0
    uint32_t mobile_ivalue_size = 0) {
  // 根据是否存在额外文件，创建FlatBuffer向量或默认为0
  auto extra_files__ = extra_files ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>>(*extra_files) : 0;
  // 根据是否存在方法，创建FlatBuffer向量或默认为0
  auto methods__ = methods ? _fbb.CreateVector<uint32_t>(*methods) : 0;
  // 根据是否存在IValue对象，创建FlatBuffer向量或默认为0
  auto ivalues__ = ivalues ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::IValue>>(*ivalues) : 0;
  // 根据是否存在存储数据，创建FlatBuffer向量或默认为0
  auto storage_data__ = storage_data ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::StorageData>>(*storage_data) : 0;
  // 根据是否存在对象类型，创建FlatBuffer向量或默认为0
  auto object_types__ = object_types ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::ObjectType>>(*object_types) : 0;
  // 根据是否存在JIT源文件，创建FlatBuffer向量或默认为0
  auto jit_sources__ = jit_sources ? _fbb.CreateVector<::flatbuffers::Offset<torch::jit::mobile::serialization::ExtraFile>>(*jit_sources) : 0;
  // 根据是否存在JIT常量，创建FlatBuffer向量或默认为0
  auto jit_constants__ = jit_constants ? _fbb.CreateVector<uint32_t>(*jit_constants) : 0;
  // 调用FlatBuffer生成的CreateModule函数，返回创建的Module对象的偏移量
  return torch::jit::mobile::serialization::CreateModule(
      _fbb,
      bytecode_version,
      extra_files__,
      methods__,
      state_obj,
      ivalues__,
      storage_data_size,
      storage_data__,
      object_types__,
      jit_sources__,
      jit_constants__,
      operator_version,
      mobile_ivalue_size);
}

// 验证IValueUnion类型的Union数据，使用FlatBuffers的Verifier对象
inline bool VerifyIValueUnion(::flatbuffers::Verifier &verifier, const void *obj, IValueUnion type) {
  switch (type) {
    // 如果类型为NONE，始终返回true
    case IValueUnion::NONE: {
      return true;
    }
    // 如果类型为Int，使用verifier验证torch::jit::mobile::serialization::Int类型的字段
    case IValueUnion::Int: {
      return verifier.VerifyField<torch::jit::mobile::serialization::Int>(static_cast<const uint8_t *>(obj), 0, 8);
    }
    // 如果类型为Bool，使用verifier验证torch::jit::mobile::serialization::Bool类型的字段
    case IValueUnion::Bool: {
      return verifier.VerifyField<torch::jit::mobile::serialization::Bool>(static_cast<const uint8_t *>(obj), 0, 1);
    }
    // 如果类型为Double，使用verifier验证torch::jit::mobile::serialization::Double类型的字段
    case IValueUnion::Double: {
      return verifier.VerifyField<torch::jit::mobile::serialization::Double>(static_cast<const uint8_t *>(obj), 0, 8);
    }
    // 如果类型为ComplexDouble，使用verifier验证torch::jit::mobile::serialization::ComplexDouble类型的字段
    case IValueUnion::ComplexDouble: {
      return verifier.VerifyField<torch::jit::mobile::serialization::ComplexDouble>(static_cast<const uint8_t *>(obj), 0, 8);
    }
    case IValueUnion::TensorMetadata: {
      // 将 obj 强制转换为 TensorMetadata 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::TensorMetadata *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::String: {
      // 将 obj 强制转换为 String 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::String *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::List: {
      // 将 obj 强制转换为 List 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::List *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::Tuple: {
      // 将 obj 强制转换为 Tuple 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::Tuple *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::Dict: {
      // 将 obj 强制转换为 Dict 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::Dict *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::Object: {
      // 将 obj 强制转换为 Object 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::Object *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::IntList: {
      // 将 obj 强制转换为 IntList 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::IntList *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::DoubleList: {
      // 将 obj 强制转换为 DoubleList 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::DoubleList *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::BoolList: {
      // 将 obj 强制转换为 BoolList 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::BoolList *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::Device: {
      // 将 obj 强制转换为 Device 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::Device *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::EnumValue: {
      // 将 obj 强制转换为 EnumValue 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::EnumValue *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    case IValueUnion::Function: {
      // 将 obj 强制转换为 Function 指针
      auto ptr = reinterpret_cast<const torch::jit::mobile::serialization::Function *>(obj);
      // 调用 verifier 对象的 VerifyTable 方法验证 ptr 指向的数据结构
      return verifier.VerifyTable(ptr);
    }
    default: 
      // 默认情况下，直接返回 true
      return true;
    }
}  // end namespace serialization
}  // end namespace mobile
}  // end namespace jit
}  // end namespace torch

#endif  // FLATBUFFERS_GENERATED_MOBILEBYTECODE_TORCH_JIT_MOBILE_SERIALIZATION_H_



// 结束命名空间 torch::jit::mobile::serialization

#endif  // 结束预编译指令 FLATBUFFERS_GENERATED_MOBILEBYTECODE_TORCH_JIT_MOBILE_SERIALIZATION_H_


这段代码主要是对 C++ 中的命名空间和预编译指令进行结束标记的注释。
```