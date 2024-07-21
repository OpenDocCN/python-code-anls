# `.\pytorch\aten\src\ATen\core\dynamic_type.h`

```py
#pragma once

#include <cstdint>  // 包含标准整数类型的头文件
#include <memory>   // 包含内存管理相关的头文件
#include <type_traits>  // 包含类型特性相关的头文件

#include <ATen/core/jit_type_base.h>  // 包含 ATen 库中的 JIT 类型基础定义
#include <c10/util/Optional.h>   // 包含 c10 库中的 Optional 类型定义

namespace c10 {

using DynamicTypeBits = std::uint32_t;  // 定义动态类型位掩码为 32 位无符号整数类型
#define DYNAMIC_TYPE_BIT(x) (1u << x)  // 定义一个宏，用于生成动态类型位掩码

constexpr DynamicTypeBits kDynamicCovariantTypeBit = DYNAMIC_TYPE_BIT(31);  // 定义动态协变类型位掩码
constexpr DynamicTypeBits kDynamicAnyTypeBit = DYNAMIC_TYPE_BIT(30);         // 定义动态任意类型位掩码

constexpr DynamicTypeBits kDynamicNoneTypeBit = DYNAMIC_TYPE_BIT(1);         // 定义动态 None 类型位掩码
constexpr DynamicTypeBits kDynamicIntTypeBit = DYNAMIC_TYPE_BIT(3);          // 定义动态整数类型位掩码
constexpr DynamicTypeBits kDynamicFloatTypeBit = DYNAMIC_TYPE_BIT(4);        // 定义动态浮点数类型位掩码
constexpr DynamicTypeBits kDynamicComplexTypeBit = DYNAMIC_TYPE_BIT(5);      // 定义动态复数类型位掩码
constexpr DynamicTypeBits kDynamicListTypeBit = DYNAMIC_TYPE_BIT(7);         // 定义动态列表类型位掩码
constexpr DynamicTypeBits kDynamicTupleTypeBit = DYNAMIC_TYPE_BIT(8);        // 定义动态元组类型位掩码
constexpr DynamicTypeBits kDynamicClassTypeBit = DYNAMIC_TYPE_BIT(10);       // 定义动态类类型位掩码

#define FORALL_DYNAMIC_TYPES(_)  // 定义一个宏，用于枚举所有动态类型

  _(Tensor, DYNAMIC_TYPE_BIT(0), 1)  // 枚举张量类型，使用动态类型位掩码的第 0 位
  _(None, kDynamicNoneTypeBit, 1)    // 枚举 None 类型，使用动态 None 类型位掩码
  _(Bool, DYNAMIC_TYPE_BIT(2), 1)    // 枚举布尔类型，使用动态类型位掩码的第 2 位
  _(Int, kDynamicIntTypeBit, 1)      // 枚举整数类型，使用动态整数类型位掩码
  _(Float, kDynamicFloatTypeBit, 1)  // 枚举浮点数类型，使用动态浮点数类型位掩码
  _(Complex, kDynamicComplexTypeBit, 1)  // 枚举复数类型，使用动态复数类型位掩码
  _(Number,                              // 枚举数值类型，包括整数、浮点数和复数
    (kDynamicIntTypeBit | kDynamicFloatTypeBit | kDynamicComplexTypeBit),  // 使用相应位掩码组合
    1)
  _(String, DYNAMIC_TYPE_BIT(6), 1)    // 枚举字符串类型，使用动态类型位掩码的第 6 位
  _(List, kDynamicListTypeBit, 0)      // 枚举列表类型，使用动态列表类型位掩码
  _(Tuple, (kDynamicTupleTypeBit | kDynamicCovariantTypeBit), 0)  // 枚举元组类型，使用元组类型位掩码和动态协变类型位掩码
  _(Dict, DYNAMIC_TYPE_BIT(9), 0)      // 枚举字典类型，使用动态类型位掩码的第 9 位
  _(Class, kDynamicClassTypeBit, 0)    // 枚举类类型，使用动态类类型位掩码
  _(Optional,                            // 枚举可选类型，包括空值和动态协变类型
    (DYNAMIC_TYPE_BIT(11) | kDynamicNoneTypeBit | kDynamicCovariantTypeBit),  // 使用相应位掩码组合
    0)
  _(AnyList, (kDynamicListTypeBit | kDynamicAnyTypeBit), 1)  // 枚举任意列表类型，使用动态列表类型位掩码和动态任意类型位掩码
  _(AnyTuple,                                // 枚举任意元组类型，包括动态元组类型和动态协变类型和动态任意类型位掩码
    (kDynamicTupleTypeBit | kDynamicCovariantTypeBit | kDynamicAnyTypeBit),
    1)
    # 定义一系列的宏，每个宏对应一个动态类型位和一个标志位，用于描述不同的数据类型或特性。
    _(DeviceObj, DYNAMIC_TYPE_BIT(12), 1)  # 表示设备对象，使用动态类型位 12，并设置标志位为 1
    _(StreamObj, DYNAMIC_TYPE_BIT(13), 1)  # 表示流对象，使用动态类型位 13，并设置标志位为 1
    _(Capsule, DYNAMIC_TYPE_BIT(14), 1)    # 表示胶囊（一种数据结构），使用动态类型位 14，并设置标志位为 1
    _(Generator, DYNAMIC_TYPE_BIT(15), 1)  # 表示生成器，使用动态类型位 15，并设置标志位为 1
    _(Storage, DYNAMIC_TYPE_BIT(16), 1)    # 表示存储，使用动态类型位 16，并设置标志位为 1
    _(Var, DYNAMIC_TYPE_BIT(17), 0)        # 表示变量，使用动态类型位 17，并设置标志位为 0
    _(AnyClass, (kDynamicClassTypeBit | kDynamicAnyTypeBit), 1)  # 表示任意类，使用特定的动态类型位组合，并设置标志位为 1
    _(QScheme, DYNAMIC_TYPE_BIT(18), 1)    # 表示量化方案，使用动态类型位 18，并设置标志位为 1
    _(Quantizer, DYNAMIC_TYPE_BIT(19), 1)  # 表示量化器，使用动态类型位 19，并设置标志位为 1
    _(AnyEnum, DYNAMIC_TYPE_BIT(20), 1)    # 表示任意枚举，使用动态类型位 20，并设置标志位为 1
    _(RRef, DYNAMIC_TYPE_BIT(21), 0)       # 表示远程引用，使用动态类型位 21，并设置标志位为 0
    _(Future, DYNAMIC_TYPE_BIT(22), 0)     # 表示未来，使用动态类型位 22，并设置标志位为 0
    _(Await, DYNAMIC_TYPE_BIT(23), 0)      # 表示等待，使用动态类型位 23，并设置标志位为 0
    _(Any, 0xffffffff, 1)                  # 表示任意类型，使用特定的动态类型位组合，并设置标志位为 1
// 定义宏 `FORALL_DYNAMIC_TYPES_FAKE`，接受一个参数 `_`，展开为一系列类型定义
#define FORALL_DYNAMIC_TYPES_FAKE(_) \
  _(ScalarType, kDynamicIntTypeBit, 1)                                \
  _(Layout, kDynamicIntTypeBit, 1)                                        \
  _(SymInt, kDynamicIntTypeBit, 1)                                        \
  _(MemoryFormat, kDynamicIntTypeBit, 1)

// 定义宏 `FORWARD_DECL_TYPE`，接受三个参数 `NAME`, `_`, `__`，用于声明结构体 `NAME ## Type`
#define FORWARD_DECL_TYPE(NAME, _, __) struct NAME ## Type;
// 展开 `FORALL_DYNAMIC_TYPES` 宏，使用 `FORWARD_DECL_TYPE` 宏声明一系列结构体
FORALL_DYNAMIC_TYPES(FORWARD_DECL_TYPE)
// 展开 `FORALL_DYNAMIC_TYPES_FAKE` 宏，使用 `FORWARD_DECL_TYPE` 宏声明另一系列结构体
FORALL_DYNAMIC_TYPES_FAKE(FORWARD_DECL_TYPE)
// 取消 `FORWARD_DECL_TYPE` 宏的定义，结束结构体声明部分
#undef FORWARD_DECL_TYPE

// 声明类 `DynamicType`
class DynamicType;
// 使用别名 `DynamicTypePtr` 表示 `std::shared_ptr<DynamicType>`
using DynamicTypePtr = std::shared_ptr<DynamicType>;
/**
 * DynamicType is designed as a low dependency type system for TorchScript. The
 * existing JIT types are used for both compilation and runtime, which makes
 * sense for server contexts because we often compile and run the model in
 * the same process, however this doesn't hold for mobile devices where we
 * always compiles a model ahead of time, therefore there will be dependencies
 * which are not needed, but built with mobile runtime causing binary size
 * bloat, by design. Every basic type like Int, Bool or String will bring their
 * vtable, typeinfo, constructor, destructor and even more data from their
 * specializations for STL types to the binary causing a long tail bloat.
 *
 * The core problem is about the complexity to implement and maintain a single
 * type system for both analysis and execution purposes. Although they should
 * have the exactly same semantics, in practice implement a unified abstraction
 * adds conceptual and representational overhead for both sides of the world.
 *
 * To address the issues, DynamicType implements a minimal subset of JIT types
 * and uses a generic algorithm to test all subtyping relations. To achieve
 * this, we assign each dynamic type a single integer tag to represent its
 * semantics. More specifically, a dynamic type is defined as a set of "control
 * bits" and "data bits", where control bits describe the special behavior when
 * testing a type and data bits map to identity of each nominal type. We use bit
 * operations to perform all the tests.
 *
 * For example, a "covariant bit" is a control bit used to describe if a type
 * is covariant, right now the most used one is tuple type, and in addition to
 * the control bit, tuple type's data bit is the 8th bit from the LSB. Control
 * bits start from MSB and data bits start from LSB.
 *
 * If two types are equal, then they are subtype of each other, also if the bits
 * from one type tag is subset of the other tag, it automatically becomes a
 * subtype of the other. This simplifies the subtyping logic a lot, and over the
 * long term it is possible to adopt this scheme on the server side as well.
 * Special cases can be added but they generally should not take too much code
 * size.
 *
 * DynamicType may or may not inherit from c10::Type because it's not the core
 * requirement of DynamicType to interface with existing JIT types, but we might
 * want to inherit from c10::Type to reduce the migration cost.
 */
class DynamicType : public SharedType {
  using ClassTypePtr = std::shared_ptr<const c10::ClassType>;

  /**
   * A implementation detail to support NamedTuple.
   */
  struct LabeledDynamicType {
    std::optional<std::string> label; // Optional label associated with the dynamic type
    DynamicTypePtr ty; // Pointer to a DynamicType object
    explicit LabeledDynamicType(DynamicTypePtr t) : ty(std::move(t)) {} // Constructor initializing the dynamic type pointer

    /**
     * Checks equality between two LabeledDynamicType objects.
     */
    bool equals(const LabeledDynamicType& other) const;
    // The equals method checks if this LabeledDynamicType object is equal to another
    // LabeledDynamicType object by comparing their label and dynamic type pointer.
  // 检查当前类型是否是给定 LabeledDynamicType 的子类型
  bool isSubtypeOf(const LabeledDynamicType& other) const;
};

// 公共部分开始

// TODO: 当所有迁移完成后，将 Ptr 更改为 DynamicTypePtr
using Ptr = TypePtr;

// ElementType 被定义为 DynamicType
using ElementType = DynamicType;

// 动态类型的析构函数
~DynamicType() override;

// Arguments 结构体，用于表示函数或方法的参数
struct Arguments {
  Arguments() = default; // 默认构造函数
  Arguments(c10::ArrayRef<TypePtr>); // 使用 TypePtr 数组初始化的构造函数
  Arguments(const std::vector<c10::string_view>&, c10::ArrayRef<TypePtr>); // 使用字符串视图和 TypePtr 数组初始化的构造函数
  std::vector<LabeledDynamicType> elems; // 参数列表，包含 LabeledDynamicType 元素
};

// 枚举类型 Tag，基于 DynamicTypeBits 枚举
enum class Tag : DynamicTypeBits {
#define DYNAMIC_TYPE_ITEM(NAME, VAL, _) NAME = VAL,
    // 定义宏DYNAMIC_TYPE_ITEM，用于展开动态类型列表中的每一项
    FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_ITEM)
    // 展开所有真实动态类型的宏定义，并将它们作为参数传递给宏DYNAMIC_TYPE_ITEM
    FORALL_DYNAMIC_TYPES_FAKE(DYNAMIC_TYPE_ITEM)
#undef DYNAMIC_TYPE_ITEM
  };

  bool equals(const Type& rhs) const override;
  // 比较函数，检查当前动态类型是否与rhs相等
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;
  // 子类型检查函数，检查当前动态类型是否是rhs的子类型，支持扩展信息输出到why_not流中
  std::string str() const override;
  // 返回当前动态类型的字符串表示
  static const TypeKind Kind = TypeKind::DynamicType;
  // 静态常量，表示当前类型的种类为DynamicType
  static TORCH_API DynamicTypePtr create(Type& ty);
  // 静态函数，创建一个动态类型指针，参数ty为类型对象的引用

  explicit DynamicType(Tag, Arguments);
  // 显式构造函数，使用标签和参数列表初始化动态类型
  explicit DynamicType(Tag, c10::string_view, Arguments);
  // 显式构造函数，使用标签、字符串视图和参数列表初始化动态类型

  TypePtr containedType(size_t) const override;
  // 返回指定索引处包含的类型指针
  size_t containedTypeSize() const override;
  // 返回包含的类型数量
  Tag tag() const {
    return tag_;
  }
  // 返回当前动态类型的标签值
  const std::optional<std::string>& name() const {
    return name_;
  }
  // 返回当前动态类型的可选名称
  const Arguments& arguments() const {
    return arguments_;
  }
  // 返回当前动态类型的参数列表
  TORCH_API TypeKind dynamicKind() const;
  // 返回当前动态类型的类型种类

  // Should be used only on the server side to restore static type information.
#ifndef C10_MOBILE
  TORCH_API
#endif
  TypePtr fallback() const;
  // 返回当前动态类型的后备类型指针，仅应在服务器端用于恢复静态类型信息

 private:
  bool symmetric() const override {
    return false;
  }
  // 私有成员函数，返回false，表示当前类型不是对称的
  friend struct Type;
  // 声明Type结构体为友元

  static std::shared_ptr<const DynamicType> create(const Type& ty);
  // 静态函数，创建一个动态类型的常量指针，参数ty为类型对象的引用
  DynamicType(const Type& other);
  // 构造函数，使用其他类型对象初始化动态类型
  bool equals(const DynamicType& other) const;
  // 比较函数，检查当前动态类型是否与other相等

  template <typename F>
  bool compareArguments(const DynamicType& other, const F& f) const {
    if (arguments_.elems.size() != other.arguments_.elems.size()) {
      return false;
    }
    for (size_t i = 0; i < arguments_.elems.size(); i++) {
      if (!f(arguments_.elems[i], other.arguments_.elems[i])) {
        return false;
      }
    }
    return true;
  }
  // 模板函数，比较当前动态类型的参数列表是否与other的参数列表相等

  Tag tag_;
  // 标签成员变量，表示当前动态类型的标签
  std::optional<std::string> name_;
  // 可选名称成员变量，表示当前动态类型的名称
  union {
    Arguments arguments_;
    ClassTypePtr class_;
  };
};

template <typename T>
struct DynamicTypeTrait {
  C10_NOINLINE static auto tagValue() {
    TORCH_CHECK(false);
    return DynamicType::Tag::Any;
  }
};
// 动态类型特性模板结构体，用于定义类型T的特性

namespace detail {
C10_NOINLINE DynamicTypePtr makeBaseType(DynamicType::Tag tag);
}
// detail命名空间，定义了创建基本类型的函数makeBaseType

#define DYNAMIC_TYPE_TAG_VALUE(NAME, _, IS_BASE_TYPE)      \
  template <>                                              \
  struct TORCH_API DynamicTypeTrait<NAME##Type> {          \
    C10_ERASE static auto tagValue() {                     \
      return DynamicType::Tag::NAME;                       \
    }                                                      \
    static constexpr bool isBaseType = IS_BASE_TYPE;       \
    template <typename T = const DynamicTypePtr&>          \
    static std::enable_if_t<isBaseType, T> getBaseType() { \
      static auto type = detail::makeBaseType(tagValue()); \
      return type;                                         \
    }                                                      \
  }; // namespace c10
// 定义宏DYNAMIC_TYPE_TAG_VALUE，用于定义动态类型名称与标签值的映射
FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_TAG_VALUE)
// 展开所有真实动态类型的宏定义，并将它们作为参数传递给宏DYNAMIC_TYPE_TAG_VALUE
FORALL_DYNAMIC_TYPES_FAKE(DYNAMIC_TYPE_TAG_VALUE)
#undef DYNAMIC_TYPE_TAG_VALUE

} // namespace c10
// c10命名空间结束
```