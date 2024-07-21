# `.\pytorch\aten\src\ATen\core\jit_type.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/custom_class.h>
// 包含 ATen 库中的 custom_class.h 文件

#include <ATen/core/jit_type_base.h>
// 包含 ATen 库中的 jit_type_base.h 文件

#include <ATen/core/TensorBody.h>
// 包含 ATen 库中的 TensorBody.h 文件

#include <ATen/core/functional.h>
// 包含 ATen 库中的 functional.h 文件

#include <ATen/core/symbol.h>
// 包含 ATen 库中的 symbol.h 文件

#include <ATen/core/type_factory.h>
// 包含 ATen 库中的 type_factory.h 文件

#include <ATen/core/qualified_name.h>
// 包含 ATen 库中的 qualified_name.h 文件

#include <c10/util/TypeList.h>
// 包含 c10 库中的 TypeList.h 文件

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional.h 文件

#include <c10/core/SymFloat.h>
// 包含 c10 库中的 SymFloat.h 文件

#include <c10/core/SymBool.h>
// 包含 c10 库中的 SymBool.h 文件

#include <c10/core/Device.h>
// 包含 c10 库中的 Device.h 文件

#include <array>
// 包含标准库中的 array 头文件

#include <memory>
// 包含标准库中的 memory 头文件

#include <ostream>
// 包含标准库中的 ostream 头文件

#include <sstream>
// 包含标准库中的 sstream 头文件

#include <utility>
// 包含标准库中的 utility 头文件

namespace torch::jit {
struct Function;
} // namespace torch::jit
// 声明命名空间 torch::jit 和其中的结构体 Function

namespace c10 {

template<class Key, class Value>
class Dict;
// 声明模板类 Dict，支持任意键值类型

struct IValue;
// 声明结构体 IValue

struct FunctionSchema;
// 声明结构体 FunctionSchema

struct NamedType;
// 声明结构体 NamedType

using OptNameList = std::optional<std::vector<std::string>>;
// 使用类型别名定义 OptNameList 为 std::optional<std::vector<std::string>> 类型

void standardizeVectorForUnion(std::vector<TypePtr>& reference, std::vector<TypePtr>* to_fill);
// 函数声明，标准化给定的 vector<TypePtr> 参考向量，并将结果填充到给定的 vector<TypePtr> 指针中

void standardizeVectorForUnion(std::vector<TypePtr>* to_flatten);
// 函数声明，标准化给定的 vector<TypePtr> 向量

inline bool is_contiguous_strides(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  // 内联函数定义，检查给定的 strides 是否为连续的
  int n_dim = static_cast<int>(sizes.size());
  if (n_dim == 0) {
    return true;
  }
  // 如果维度为 0，返回 true

  if (strides[n_dim - 1] != 1) {
    return false;
  }
  // 如果最后一个维度的步长不为 1，返回 false

  for (int i = n_dim - 2; i >= 0; i--) {
    if (strides[i] != strides[i + 1] * sizes[i + 1]) {
      return false;
    }
  }
  // 遍历检查每个维度的步长是否连续
  return true;
}

struct AnyType;
// 声明结构体 AnyType

using AnyTypePtr = SingletonTypePtr<AnyType>;
// 使用类型别名定义 AnyTypePtr 为 SingletonTypePtr<AnyType> 类型

// Any 是类型层次结构的顶部，所有其他类型都是其子类型
// T <: Any, forall T
struct TORCH_API AnyType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 重写父类函数，比较类型是否相等

  std::string str() const override {
    return "Any";
  }
  // 重写父类函数，返回类型的字符串表示为 "Any"

  static const TypeKind Kind = TypeKind::AnyType;
  // 声明静态常量 Kind 为 TypeKind::AnyType

  // 全局单例
  static AnyTypePtr get();

 private:
  AnyType() : Type(TypeKind::AnyType) {}
  // 私有构造函数，初始化类型为 TypeKind::AnyType
};

inline std::string toString(const Type& type) {
  return type.str();
}
// 内联函数定义，返回给定类型的字符串表示形式

// 为与使用 TypePtr 的代码兼容而设置的兼容性包装
inline std::string toString(const TypePtr& typePtr) {
  return toString(*typePtr);
}
// 内联函数定义，返回给定类型指针所指向类型的字符串表示形式

inline bool operator!=(const Type& lhs, const Type& rhs) {
  return !(lhs == rhs);
}
// 重载不等于运算符，判断两个类型是否不相等

// 所有具有单个子元素的类型的共同基类
// 例如 Future[T], Optional[T], List[T]
template <TypeKind K, typename T>
struct SingleElementType : public SharedType {
  static const TypeKind Kind = K;
  // 静态常量，表示类型的种类为 K

  const TypePtr& getElementType() const {
    return elem;
  }
  // 获取元素类型的引用

  bool hasFreeVariables() const override {
    return getElementType()->hasFreeVariables();
  }
  // 重写父类函数，检查类型是否包含自由变量

  at::ArrayRef<TypePtr> containedTypes() const override {
    return elem;
  }
  // 重写父类函数，返回类型包含的所有子类型的数组引用

  bool equals(const Type& rhs) const override {
    if (auto rhs_ = rhs.cast<T>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }
  // 重写父类函数，比较类型是否相等

 protected:
  SingleElementType(TypePtr elem) : SharedType(Kind), elem(std::move(elem)) {
    if (!this->elem) {
      throw std::runtime_error(c10::str(
            "Can not create ", typeKindToString(Kind), " with None type"));
    }
  }
  // 受保护构造函数，初始化成员 elem，并检查其是否为空

 private:
  TypePtr elem;
};
struct UnionType;
using UnionTypePtr = std::shared_ptr<UnionType>;

// UnionType struct represents a union of types derived from SharedType
struct TORCH_API UnionType : public SharedType {
  friend struct Type;

  // Define the kind of this type as UnionType
  static const TypeKind Kind = TypeKind::UnionType;

  // Check if this type is a subtype of another type with extended information
  bool isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const override;

  // Return a string representation of this type
  std::string str() const override;

  // Factory method to create UnionType from a vector of TypePtr
  static UnionTypePtr create(std::vector<TypePtr> reference);

  // Check if this UnionType equals another Type
  bool equals(const Type& rhs) const override;

  // Check if this type is a UnionType
  bool isUnionType() const override {
    return true;
  }

  // Return an ArrayRef of contained types in this UnionType
  at::ArrayRef<TypePtr> containedTypes() const override {
    return types_;
  }

  // For testing purposes only, return the contained types
  at::ArrayRef<TypePtr> getTypes() const {
    return types_;
  }

  // Create a new UnionType with specified contained types
  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types));
  }

  // Check if this UnionType can hold a specific type
  bool canHoldType(const Type& type) const;

  // Check if this UnionType has free variables
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }

  // Convert this UnionType to an optional type
  std::optional<TypePtr> toOptional() const;

  // Subtract a set of types from this UnionType and return the result
  std::optional<TypePtr> subtractTypeSet(std::vector<TypePtr>& to_subtract) const;

 protected:
    // Constructor for UnionType with specified types and kind
    explicit UnionType(std::vector<TypePtr> types, TypeKind kind=TypeKind::UnionType);

    // Return a string representation of this type with annotations
    std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override;

    // Return a string representation of this UnionType
    std::string unionStr(
        const TypePrinter& printer = nullptr,
        bool is_annotation_str = false) const;

    // Whether this UnionType has free variables
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool has_free_variables_;

    // Vector of types contained in this UnionType
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::vector<TypePtr> types_;

    // Whether this UnionType can hold None (null) type
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool can_hold_none_;
};

struct OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;

// OptionalType struct represents an optional type, inheriting from UnionType
// Represents type Optional[T] which can accept both T and None
struct TORCH_API OptionalType : public UnionType {
  // Factory method to create OptionalType with a contained element type
  static OptionalTypePtr create(const TypePtr& contained);

  // Define the kind of this type as OptionalType
  static const TypeKind Kind = TypeKind::OptionalType;

  friend struct Type;

  // Check if this OptionalType equals another Type
  bool equals(const Type& rhs) const override;

  // Return the element type contained in this OptionalType
  const TypePtr& getElementType() const {
    return contained_;
  }

  // Return an ArrayRef of contained types in this OptionalType
  at::ArrayRef<TypePtr> containedTypes() const override {
    return contained_;
  }

  // Return a string representation of this OptionalType
  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "?";
    return ss.str();
  }

  // Create a new OptionalType with specified contained types
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    AT_ASSERT(contained_types.size() == 1);
    return create(contained_types[0]);
  }

  // Check if this OptionalType is a subtype of another type with extended information
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // Check if this OptionalType is a UnionType
  bool isUnionType() const override {
    return true;
  }

  // 返回 true
  static TypePtr ofTensor();

  // 返回全局单例对象
  static TypePtr get(TypePtr inner);

 private:
  // 显式构造函数，接受一个 TypePtr 参数作为包含的类型
  explicit OptionalType(const TypePtr& contained);

  TypePtr contained_; // 包含的类型指针

  // 实现类型注解字符串的方法，可选地接受打印器参数
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "Optional[" << getElementType()->annotation_str(printer) << "]";
    return ss.str(); // 返回构建的类型注解字符串
  }
// 结构体结束标志
};

// 模板函数定义：合并两个可能为空的可选类型对象，如果两者都有值且相等，则返回其中一个，否则返回空的可选类型对象
template <typename T>
inline std::optional<T> merge_primitive(
    const std::optional<T>& a,
    const std::optional<T>& b) {
  if (a.has_value() && b.has_value() && a.value() == b.value()) {
    return a;
  }
  return std::optional<T>{};
}

// 如果我们看到 `a + b + c` 并且知道 a、b、c 大小相同且是二维的（WxH），我们可以生成一个融合的核函数。该融合核函数可能包含索引数学来处理 W 和 H 维度。然而，如果我们知道 WxH 维度是连续的，我们可以假装只有一个单一维度，简化索引逻辑。
// 即使维度被转置，只要 a、b 和 c 的转置方式相同，也可以执行此操作。
// 我们希望编译器能够进行这种维度减少，但仅仅知道大小是不够的。
// 我们可以扩展性能分析以记录步幅信息。
// 不必记录特定的步幅，我们可以按照从小到大的顺序排列步幅，使用 `stride_indices` 标记最小步幅的连续性（c0 表示步幅恰好为 1），否则表示 $stride_n = size_{n-1}*stride_{n-1}$
struct TORCH_API Stride {
  Stride() = default;
  // 构造函数：初始化步幅对象，接受步幅索引、连续性和步幅的可选值
  Stride(
      const std::optional<size_t>& stride_index,
      std::optional<bool> contiguous,
      const std::optional<size_t>& stride)
      : stride_index_(stride_index), contiguous_(contiguous), stride_(stride) {}

  // 比较运算符重载：比较两个步幅对象是否相等
  bool operator==(const Stride& b) const {
    return stride_index_ == b.stride_index_ && contiguous_ == b.contiguous_ &&
        stride_ == b.stride_;
  }

  // 判断步幅对象是否完整，即三个成员都有值
  bool isComplete() const {
    return stride_index_ && contiguous_ && stride_;
  }

  // 成员变量：步幅索引、连续性和步幅的可选值
  std::optional<size_t> stride_index_;
  std::optional<bool> contiguous_;
  std::optional<size_t> stride_;
};

// 模板特化函数定义：合并两个可能为空的步幅对象，返回合并后的步幅对象
template <>
inline std::optional<Stride> merge_primitive(
    const std::optional<Stride>& a,
    const std::optional<Stride>& b) {
  std::optional<Stride> left = a;
  std::optional<Stride> right = b;
  // 如果左边步幅对象为空，使用默认构造函数初始化
  if (!left.has_value()) {
    left = {Stride()};
  }
  // 如果右边步幅对象为空，使用默认构造函数初始化
  if (!right.has_value()) {
    right = {Stride()};
  }

  // 合并步幅对象的三个成员：步幅索引、连续性和步幅
  auto merged_index =
      merge_primitive(left->stride_index_, right->stride_index_);
  auto merged_cont = merge_primitive(left->contiguous_, right->contiguous_);
  auto merged_stride = merge_primitive(left->stride_, right->stride_);
  auto r = Stride(merged_index, merged_cont, merged_stride);
  // 规范化步幅对象：如果三个成员都没有值，则返回空的步幅对象
  if (!r.stride_index_.has_value() && !r.contiguous_.has_value() &&
      !r.stride_.has_value()) {
    return std::optional<Stride>{};
  }

  return r;
}

// 结构体定义：形状符号，用于在 `std::map` 中使用
struct TORCH_API ShapeSymbol {
  // 默认构造函数：初始化为 -1，用于形状符号的标志
  ShapeSymbol() : value_(-1) {}
  // 判断符号是否为固定/静态维度
  bool is_static() const {
    return value_ >= 0;
  };
  // 比较运算符重载：比较两个形状符号对象是否相等
  bool operator==(const ShapeSymbol& b) const {
    return value_ == b.value_;
  }
  // 比较运算符重载：用于在 `std::map` 中排序形状符号对象
  bool operator<(const ShapeSymbol& b) const {
    return value_ < b.value_;

# 比较当前对象的值与另一个ShapeSymbol对象b的值的大小，返回比较结果的布尔值。

  }

  static ShapeSymbol fromStaticSize(int64_t val) {
    return ShapeSymbol(val);
  }

# 根据给定的整数值val创建并返回一个ShapeSymbol对象，用于表示静态大小。

  int64_t static_size() const {
    TORCH_CHECK(is_static());
    return value_;
  };

# 返回当前ShapeSymbol对象的值作为静态大小。在返回之前，使用TORCH_CHECK确保对象是静态的。

  int64_t value() const {
    return value_;
  };

# 返回当前ShapeSymbol对象的值。

  static ShapeSymbol newSymbol() {
    return fromStaticSize(-static_cast<int64_t>(++num_symbols));
  };

# 创建并返回一个新的ShapeSymbol对象，其值是一个递减的唯一标识符（通过静态计数器num_symbols）的负值。

  friend TORCH_API std::ostream& operator<<(
      std::ostream& os,
      const ShapeSymbol& s);

# 声明一个友元函数，用于将ShapeSymbol对象s的信息输出到给定的输出流os中。

 private:
  ShapeSymbol(int64_t val) : value_(val) {}

# 私有构造函数，根据给定的整数值val初始化ShapeSymbol对象的值value_。

  int64_t value_;

# 存储ShapeSymbol对象的整数值。

  static std::atomic<size_t> num_symbols;

# 静态成员变量，用于生成唯一的ShapeSymbol对象标识符。
};

// 合并两个 ShapeSymbol 对象，如果它们都是静态的且相等，则返回其中一个；否则返回一个新的符号。
inline ShapeSymbol merge_primitive(
    const ShapeSymbol& a,
    const ShapeSymbol& b) {
  // 如果 a 和 b 都是静态的且相等，则直接返回 a
  if (a.is_static() && b.is_static() && a == b) {
    return a;
  }
  // 否则返回一个新创建的符号
  return ShapeSymbol::newSymbol();
}

// 使用 ShapeSymbol 表示的张量形状。支持未排名、部分已知和完全已知的形状。
struct TORCH_API SymbolicShape {
  // 未排名形状的构造函数
  SymbolicShape() : dims_(c10::nullopt) {}

  // 已知排名但未知维度的形状构造函数
  SymbolicShape(std::optional<size_t> rank) : dims_(c10::nullopt) {
    // 如果 rank 为空，直接返回
    if (!rank) {
      return;
    }

    // 创建一个包含 rank 个 ShapeSymbol::newSymbol() 的向量
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(*rank);
    for (size_t i = 0; i < *rank; ++i) {
      shape_symbols.push_back(ShapeSymbol::newSymbol());
    }
    dims_ = shape_symbols;
  }

  // 已知和未知排名的混合形状构造函数
  SymbolicShape(const std::vector<std::optional<int64_t>>& dims) {
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(dims.size());
    for (std::optional<int64_t> dim : dims) {
      if (!dim) {
        shape_symbols.push_back(ShapeSymbol::newSymbol());
      } else {
        shape_symbols.push_back(ShapeSymbol::fromStaticSize(*dim));
      }
    }
    dims_ = shape_symbols;
  }

  void dump() const;

  // 从 ShapeSymbol 向量构造形状
  SymbolicShape(std::vector<ShapeSymbol> dims) : dims_(std::move(dims)) {}

  // 从 c10::IntArrayRef 构造形状
  SymbolicShape(c10::IntArrayRef dims) {
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(dims.size());
    for (int64_t dim : dims) {
      shape_symbols.push_back(ShapeSymbol::fromStaticSize(dim));
    }
    dims_ = shape_symbols;
  }

  // 返回索引为 i 的 ShapeSymbol
  ShapeSymbol operator[](size_t i) const {
    if (!dims_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return (*dims_).at(i);
  }

  // 返回索引为 i 的 ShapeSymbol
  ShapeSymbol at(size_t i) const {
    if (!dims_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return (*dims_).at(i);
  }

  // 返回形状的排名，如果未排名则返回 nullopt
  std::optional<size_t> rank() const {
    if (!dims_) {
      return c10::nullopt;
    }
    return dims_->size();
  }

  // 返回形状的尺寸，如果未排名则返回 nullopt
  std::optional<std::vector<ShapeSymbol>> sizes() const {
    return dims_;
  }

  // 返回形状的符号维度，如果未排名则返回 nullopt
  std::optional<std::vector<bool>> symbolicDims() const {
    if (!dims_) {
      return c10::nullopt;
    }
    auto symbolic_dims = std::vector<bool>();
    for (const ShapeSymbol& s : *dims_) {
      symbolic_dims.push_back(!s.is_static());
    }
    return symbolic_dims;
  }

  // 检查形状是否完全定义，即每个维度的排名和尺寸都已知
  bool isComplete() const {
    if (!dims_) {
      return false;
    }
    for (auto d : *dims_) {
      if (!d.is_static()) {
        return false;
      }
    }
    return true;
  }
  // 返回 true
  return true;
}

// 创建一个新的 SymbolicShape，该形状是自身和另一个 SymbolicShape 合并的结果。
// 只有静态且相等的维度将被保留。
// 如果两个形状中任一形状的秩未知，或者它们的秩不匹配，结果将是未定秩的。
SymbolicShape merge(const SymbolicShape& other) const;

// 友元函数重载，比较两个 SymbolicShape 是否相等，基于其维度的比较。
friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
  // 比较两个 SymbolicShape 的维度是否相等
  return lhs.dims_ == rhs.dims_;
}

// 友元函数重载，比较两个 SymbolicShape 是否不相等，基于其维度的比较。
friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
  // 使用 operator== 判断两个 SymbolicShape 是否不相等
  return !(lhs == rhs);
}

private:
  // 可选类型，包含一个 ShapeSymbol 的向量，表示 SymbolicShape 的维度。
  std::optional<std::vector<ShapeSymbol>> dims_;
};

namespace detail {
// 检查给定的 Stride 对象是否完整
inline bool isComplete(const Stride& s) {
  return s.isComplete();
}

// 对于任意类型 T，始终返回 true，用于判断是否完整
template<typename T>
inline bool isComplete(const T& /*t*/) {
  return true;
}
}

template <typename T>
struct VaryingShape {
  // 定义一个包含可选元素的列表类型
  using ListOfOptionalElements = std::vector<std::optional<T>>;
  
  // 从标准向量构造 VaryingShape 对象
  VaryingShape(const std::vector<T>& vec)
      : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

  // 从 c10::ArrayRef 构造 VaryingShape 对象
  VaryingShape(c10::ArrayRef<T> vec)
      : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

  // 构造具有指定大小的 VaryingShape 对象，可以为空
  VaryingShape(std::optional<size_t> size = c10::nullopt) : dims_(c10::nullopt) {
    if (size) {
      dims_ = ListOfOptionalElements(*size);
    }
  }

  // 从给定的维度列表构造 VaryingShape 对象
  VaryingShape(ListOfOptionalElements dims) : dims_(std::move(dims)) {}

  // 从给定大小构造 VaryingShape 对象
  VaryingShape(size_t size) : VaryingShape(std::optional<size_t>(size)) {}

  // 比较两个 VaryingShape 对象是否相等
  bool operator==(const VaryingShape& other) const {
    return dims_ == other.dims_;
  }

  // 获取指定索引处的元素，并确保维度已经固定
  const std::optional<T> &operator[](size_t i) const {
    if (!dims_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return (*dims_).at(i);
  }

  // 返回 VaryingShape 对象的大小，如果维度未固定则返回空
  std::optional<size_t> size() const {
    if (!dims_) {
      return c10::nullopt;
    }
    const auto& dims = dims_.value();
    return dims.size();
  }

  // 返回 VaryingShape 对象的维度列表
  const std::optional<ListOfOptionalElements>& sizes() const {
    return dims_;
  }

  // 合并当前 VaryingShape 对象与另一个对象
  TORCH_API VaryingShape merge(const VaryingShape& other) const;

  // 返回具体尺寸的向量，如果维度未固定则返回空
  std::optional<std::vector<T>> concrete_sizes() const {
    if (!dims_) {
      return c10::nullopt;
    }
    std::vector<T> sizes;
    sizes.reserve(dims_.value().size());
    for (auto d : *dims_) {
      if (!d) {
        return c10::nullopt;
      }
      sizes.push_back(d.value());
    }
    return sizes;
  }

  // 检查 VaryingShape 对象是否完整，即其所有维度均已固定
  bool isComplete() const {
    if (!dims_) {
      return false;
    }
    for (auto d : *dims_) {
      if (!d || !detail::isComplete(*d)) {
        return false;
      }
    }
    return true;
  }

 private:
  std::optional<ListOfOptionalElements> dims_;
};

struct TensorType;
// TODO: investigate making this SingletonOrSharedTypePtr<TensorType>
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor with a specific size
// 定义名为 TensorType 的结构体，继承自 SharedType
struct TORCH_API TensorType : public SharedType {
  // 静态方法，根据给定的 Tensor 对象 t 创建 TensorTypePtr
  static TensorTypePtr create(const at::Tensor& t);

  // 静态方法，根据指定参数创建 TensorTypePtr 实例，用于 shape_analysis.cpp
  static TensorTypePtr create(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      const VaryingShape<int64_t>& sizes,
      const VaryingShape<int64_t>& strides,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false,
      bool tensor_contiguity = false);

  // 静态方法，根据指定参数创建 TensorTypePtr 实例，支持符号化的形状 sizes
  static TensorTypePtr create(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      const SymbolicShape& sizes,
      const VaryingShape<Stride>& stride_,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false);

  // 静态方法，根据指定参数创建 TensorTypePtr 实例，支持维度 dim
  static TensorTypePtr create(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      std::optional<size_t> dim,
      std::optional<bool> requires_grad);

  // 重载的静态方法，用于创建连续存储的 TensorTypePtr 实例，接受 IntArrayRef 类型的 sizes
  static TensorTypePtr createContiguous(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes);

  // 从给定的 Type 类型中创建 NumberType 的 TypePtr
  static TypePtr fromNumberType(const Type& typ);

  // 从 BoolType 中创建 TypePtr
  static TypePtr fromBoolType();

  // 返回 TensorType 的维度大小，如果未定义则返回 std::optional<size_t>
  std::optional<size_t> dim() const {
    return sizes().size();
  }

  // 返回 TensorType 的大小信息
  VaryingShape<int64_t> sizes() const;

  // 返回 TensorType 的步幅信息
  VaryingShape<int64_t> strides() const;

  // 返回 TensorType 的步幅属性
  const VaryingShape<Stride>& stride_properties() const {
    return strides_;
  }

  // 返回 TensorType 的设备信息，如果未定义则返回 std::optional<at::Device>
  std::optional<at::Device> device() const {
    return device_;
  }

  // 返回 TensorType 的标量类型信息，如果未定义则返回 std::optional<at::ScalarType>
  std::optional<at::ScalarType> scalarType() const {
    return scalar_type_;
  }

  // 返回 TensorType 是否需要梯度，如果未定义则返回 std::optional<bool>
  std::optional<bool> requiresGrad() const {
    return requires_grad_;
  }

  // 返回 TensorType 是否需要梯度，如果未定义则默认返回 true
  bool requires_grad() const override {
    return requires_grad_ ? *requires_grad_ : true;
  }

  // 判断当前 TensorType 是否与给定的 Type 对象相等
  bool equals(const Type& rhs) const override;

  // 判断当前 TensorType 是否为给定 Type 对象的子类型，并输出详细信息到 why_not
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // 返回 TensorType 的字符串表示
  std::string str() const override;

  // 返回 TensorType 的字符串表示，如果是推断类型则附加 "(inferred)"
  std::string repr_str() const override {
    if (isInferredType()) {
      return str() + " (inferred)";
    } else {
      return str();
    }
  }

  // 返回 TensorType 的元素数量，如果有未定义的维度则返回 std::optional<size_t>
  std::optional<size_t> numel() const {
    size_t prod = 1;
    const auto& shape = sizes();

    for (size_t i = 0; i < shape.size(); i++) {
      if (!shape[i]) {
        return std::optional<size_t>{};
      }
      prod *= shape[i].value();
    }
    return prod;
  }

  // 创建一个具有新 requires_grad 属性的 TensorTypePtr 实例
  TensorTypePtr withRequiresGrad(std::optional<bool> s) {
    auto copy = clone();
    copy->requires_grad_ = s;
    return copy;
  }

  // 创建一个具有新 scalar_type 属性的 TensorTypePtr 实例
  TensorTypePtr withScalarType(std::optional<ScalarType> st) {
    auto copy = clone();
    copy->scalar_type_ = st;
    return copy;
  }

  // 创建一个具有新维度属性的 TensorTypePtr 实例
  TensorTypePtr withDim(std::optional<size_t> d) {
    auto copy = clone();
    // withDim is only used by the legacy executor
    // that only cares about the rank, so create dummy symbols)) :
    copy->sizes_ = SymbolicShape(d);
    copy->strides_ = VaryingShape<Stride>(d);
    # 将传入的参数 d 用于初始化 VaryingShape<Stride> 类型对象，并将其赋值给 copy 对象的 strides_ 成员变量
    return copy;
  }

  TensorTypePtr withStrides(VaryingShape<Stride> sstrides) const {
    # 克隆当前对象，赋值给 cloned
    auto cloned = clone();
    # 将传入的 sstrides 移动赋值给 cloned 对象的 strides_ 成员变量
    cloned->strides_ = std::move(sstrides);
    # 返回克隆后的对象指针
    return cloned;
  }

  TensorTypePtr withSizesStrides(
      at::IntArrayRef sizes,
      at::IntArrayRef strides) const {
    # 克隆当前对象，赋值给 cloned
    auto cloned = clone();
    # 使用 sizes 创建 SymbolicShape 对象并赋值给 cloned 对象的 sizes_ 成员变量
    auto ssizes = SymbolicShape(sizes);
    cloned->sizes_ = ssizes;
    # 使用 sizes 和 strides 计算出新的 strides，并赋值给 cloned 对象的 strides_ 成员变量
    cloned->strides_ = computeStrideProps(sizes, strides);
    # 返回克隆后的对象指针
    return cloned;
  }

  TensorTypePtr withSymbolicShapes(SymbolicShape ssizes) const {
    # 克隆当前对象，赋值给 cloned
    auto cloned = clone();
    # 将传入的 ssizes 移动赋值给 cloned 对象的 sizes_ 成员变量
    cloned->sizes_ = std::move(ssizes);
    # 返回克隆后的对象指针
    return cloned;
  }

  TensorTypePtr withSizes(at::IntArrayRef sizes) const {
    # 调用 withSizesStrides 方法，传入 sizes 和其对应的连续 strides
    return withSizesStrides(
        sizes, contiguousStridesOf(sizes));
  }

  TensorTypePtr withDevice(const std::optional<at::Device> device) const {
    # 克隆当前对象，赋值给 copy
    auto copy = clone();
    # 将传入的 device 赋值给 copy 对象的 device_ 成员变量
    copy->device_ = device;
    # 返回克隆后的对象指针
    return copy;
  }

  TensorTypePtr dimensionedOnly() const {
    # 克隆当前对象，赋值给 copy
    auto copy = clone();
    # 使用 sizes().size() 创建 SymbolicShape 对象，并赋值给 copy 对象的 sizes_ 成员变量
    copy->sizes_ = SymbolicShape(sizes().size());
    # 使用 sizes().size() 创建 VaryingShape<Stride> 对象，并赋值给 copy 对象的 strides_ 成员变量
    copy->strides_ = VaryingShape<Stride>(sizes().size());
    # 返回克隆后的对象指针
    return copy;
  }

  TensorTypePtr contiguous() const {
    # 克隆当前对象，赋值给 cloned
    auto cloned = clone();
    # 断言 sizes().concrete_sizes() 有具体值
    TORCH_INTERNAL_ASSERT(sizes().concrete_sizes().has_value());
    # 计算出连续 strides，并赋值给 cloned 对象的 strides_ 成员变量
    auto strides = computeStrideProps(
        *sizes().concrete_sizes(),
        contiguousStridesOf(*sizes().concrete_sizes()));
    cloned->strides_ = strides;
    # 返回克隆后的对象指针
    return cloned;
  }

  const SymbolicShape& symbolic_sizes() const;

  TensorTypePtr merge(const TensorType& other, bool merge_sizes = true) const;

  bool matchTensor(const at::Tensor& t);

  // is all information about the type specified except for autograd?
  // This replaces the notion of a 'CompleteTensorType' that used to exist
  // in the type-hierarchy. Excluding require_grad and undefined allows
  // this to match the old behavior.
  bool isComplete() const {
    # 检查 scalar_type_、device_、sizes_.isComplete() 和 strides_.isComplete() 是否都有值
    return scalar_type_ && device_ && sizes_.isComplete() && strides_.isComplete();
  }

  bool isInferredType() const {
    # 返回 is_inferred_ 成员变量的值
    return is_inferred_;
  }

  static TensorTypePtr getInferred() {
    # 定义静态局部变量 valueInferred，并初始化为一个推断的 TensorType 对象
    static auto valueInferred = TensorType::create(
        /*scalar_type=*/{},
        /*device=*/{},
        /*sizes=*/SymbolicShape(),
        /*stride=*/VaryingShape<Stride>{},
        /*requires_grad=*/{},
        /*undefined=*/false);
    # 将 is_inferred_ 成员变量设置为 true
    valueInferred->is_inferred_ = true;
    # 返回 valueInferred 的指针
    return valueInferred;
  }

  // this property is used by GuardElimination
  // please see `checkInputs` for more details
  bool isSummarized() const {
    # 检查是否不是完整类型并且 requiresGrad() 和 undefined() 都没有值
    return !(isComplete() && requiresGrad().has_value() &&
             undefined().has_value());
  }

  TensorTypePtr withUndefined() {
    # 克隆当前对象，赋值给 r
    auto r = clone();
    # 将 undefined_ 成员变量设置为 true
    r->undefined_ = true;
    # 返回克隆后的对象指针
    return r;
  }

  TensorTypePtr withPossiblyUndefined() {
    # 克隆当前对象，赋值给 r
    auto r = clone();
    # 将 undefined_ 成员变量设置为 c10::nullopt
    r->undefined_ = c10::nullopt;
    #
    return r;
  }



    // 返回当前对象的成员变量 r
    return r;
  }



  std::optional<bool> undefined() const { return undefined_; }



  // 返回当前对象的 undefined_ 成员变量作为 std::optional<bool>
  std::optional<bool> undefined() const { return undefined_; }



  static const TensorTypePtr& get();



  // 返回静态成员变量 Kind，表示类型为 TensorType 的种类
  static const TensorTypePtr& get();



  static const TypeKind Kind = TypeKind::TensorType;



  // 静态常量成员变量 Kind 表示类型为 TensorType 的种类
  static const TypeKind Kind = TypeKind::TensorType;



  static std::vector<int64_t> contiguousStridesOf(
      at::IntArrayRef in_sizes,
      at::MemoryFormat memory_format = MemoryFormat::Contiguous) {



  // 计算连续存储情况下的步长（strides）
  static std::vector<int64_t> contiguousStridesOf(
      at::IntArrayRef in_sizes,
      at::MemoryFormat memory_format = MemoryFormat::Contiguous) {



    auto contiguous_fn = [](const at::IntArrayRef& sizes,
                            const std::vector<int64_t>& dim_order) {
      std::vector<int64_t> strides(sizes.size());
      if (sizes.empty()) // zero-dim case
        return strides;

      strides[dim_order[0]] = 1;
      for (size_t i = 1; i < dim_order.size(); i++) {
        auto cur_dim = dim_order[i];
        auto pre_dim = dim_order[i - 1];
        strides[cur_dim] = strides[pre_dim] * sizes[pre_dim];
      }
      return strides;
    };



    // 定义 lambda 函数 contiguous_fn 用于计算连续存储情况下的步长（strides）
    auto contiguous_fn = [](const at::IntArrayRef& sizes,
                            const std::vector<int64_t>& dim_order) {
      std::vector<int64_t> strides(sizes.size());
      if (sizes.empty()) // zero-dim case
        return strides;

      strides[dim_order[0]] = 1;
      for (size_t i = 1; i < dim_order.size(); i++) {
        auto cur_dim = dim_order[i];
        auto pre_dim = dim_order[i - 1];
        strides[cur_dim] = strides[pre_dim] * sizes[pre_dim];
      }
      return strides;
    };



    std::vector<int64_t> dim_order(in_sizes.size());
    if (memory_format == MemoryFormat::ChannelsLast) {
      dim_order = {1, 3, 2, 0};
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
      dim_order = {1, 4, 3, 2, 0};
    } else {
      auto ndims = in_sizes.size();
      for (size_t i = 0; i < ndims; i++) {
        dim_order[i] = static_cast<int64_t>(ndims - i - 1); // Reverse
      }
    }
    return contiguous_fn(in_sizes, dim_order);
  }



    // 根据内存格式计算连续存储情况下的步长（strides）
    std::vector<int64_t> dim_order(in_sizes.size());
    if (memory_format == MemoryFormat::ChannelsLast) {
      dim_order = {1, 3, 2, 0};
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
      dim_order = {1, 4, 3, 2, 0};
    } else {
      auto ndims = in_sizes.size();
      for (size_t i = 0; i < ndims; i++) {
        dim_order[i] = static_cast<int64_t>(ndims - i - 1); // Reverse
      }
    }
    return contiguous_fn(in_sizes, dim_order);
  }



 private:
  TensorType(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      SymbolicShape sizes,
      VaryingShape<Stride> strides,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false);



  // 私有构造函数，用于初始化 TensorType 类的对象
  TensorType(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      SymbolicShape sizes,
      VaryingShape<Stride> strides,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false);



  TensorTypePtr clone() const {
    return TensorTypePtr(new TensorType(
        scalar_type_, device_, sizes_, strides_, requires_grad_, undefined_));
  }



  // 克隆当前对象并返回克隆的指针
  TensorTypePtr clone() const {
    return TensorTypePtr(new TensorType(
        scalar_type_, device_, sizes_, strides_, requires_grad_, undefined_));
  }



  static VaryingShape<Stride> computeStrideProps(
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      bool tensor_contiguity = false);



  // 计算张量的步长属性
  static VaryingShape<Stride> computeStrideProps(
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      bool tensor_contiguity = false);



  std::optional<at::ScalarType> scalar_type_;
  std::optional<at::Device> device_;
  SymbolicShape sizes_;
  VaryingShape<Stride> strides_;
  std::optional<bool> requires_grad_;



  // 成员变量，用于存储张量类型的信息
  std::optional<at::ScalarType> scalar_type_;
  std::optional<at::Device> device_;
  SymbolicShape sizes_;
  VaryingShape<Stride> strides_;
  std::optional<bool> requires_grad_;



  // we exploit the fact certain tensors must be zero in the autograd to
  // optimize gradient computation. Such zero tensors are currently implemented
  // with `UndefinedTensorImpl.` They can be handled only by special operators
  // (e.g. `AutogradAdd`) and their `Tensor::defined()` property returns false.
  // Normally, `undefined_` is set to false, unless a type was created
  // with `withUndefined`
  // This will also mean that `undefined` tensors will fail
  // `subtypeOf(TensorType::get())` check
  // undefined_ may become `c10::nullopt` if the tensor was observed to be both
  // defined and undefined. However, no tensor type starts out with
  // `undefined_` set to `c10::nullopt`
  std::optional<bool> undefined_;



  // 表示是否为未定义的张量类型，如果是，则 undefined_ 为 true，否则为 false
  // 未定义的张量类型在自动求导中会被特殊处理，其 defined() 属性返回 false
  // 通常情况下，undefined_ 被设置为 false，除非类型是用 withUndefined 创建的
  // 对于 undefined_ 张量，subtypeOf(TensorType::get()) 检查将失败
  // 如果观察到张量同时被定义和未定义，undefined_ 可能会变为 c10::nullopt
  // 但是，没有张量类型从 undefined_ 开始设置为 c10::nullopt
  std::optional<bool> undefined_;



  // Represents whether or not this type was inferred.
  bool is_inferred_ = false;



  // 表示此类型是否是推断出来的
  bool is_inferred_ = false;
};

// ListType 结构体的前向声明
struct ListType;

// 使用 ListType 的智能指针作为别名
using ListTypePtr = std::shared_ptr<ListType>;

// 定义 ListType 结构体，继承自 SingleElementType，表示列表类型
struct TORCH_API ListType
    : public SingleElementType<TypeKind::ListType, ListType> {
  
  // 友元声明，Type 结构体可以访问 ListType 的私有成员
  friend struct Type;
  
  // 创建 ListType 实例的模板函数，返回 ListTypePtr
  template <typename... T>
  static ListTypePtr create(T&&... all) {
    // 使用 std::make_shared 创建 ListType 实例，传递参数 all
    return ListTypePtr(
        new ListType(std::forward<T>(all)...)); // NOLINT(modernize-make-shared)
  }

  // 返回当前列表类型的字符串表示形式
  std::string str() const override {
    // 创建一个字符串流对象
    std::stringstream ss;
    // 将元素类型的字符串表示形式添加到流中，加上 "[]"
    ss << getElementType()->str() << "[]";
    // 返回流的内容作为字符串
    return ss.str();
  }

  // 根据包含的类型创建新的列表类型
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types.at(0)));
  }

  // 检查当前列表类型是否是 rhs 类型的子类型，可输出详细信息到 why_not 流
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // 获取全局单例的 ListType 类型
  static TypePtr get(const std::string& identifier, TypePtr inner);

  // 下面是一些常见的 ListTypePtr 创建函数，用于特定类型的列表
  static ListTypePtr ofTensors();
  static ListTypePtr ofOptionalTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofSymInts();
  static ListTypePtr ofFloats();
  static ListTypePtr ofComplexDoubles();
  static ListTypePtr ofBools();
  static ListTypePtr ofStrings();
  static ListTypePtr ofNumbers();

 private:
  // 私有构造函数，用于内部创建 ListType 实例
  ListType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  // 返回注释字符串的内部实现，用于类型打印
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "List[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

// DictType 结构体的前向声明
struct DictType;

// 使用 DictType 的智能指针作为别名
using DictTypePtr = std::shared_ptr<DictType>;

// 定义 DictType 结构体，继承自 SharedType，表示字典类型
struct TORCH_API DictType : public SharedType {
  
  // 友元声明，Type 结构体可以访问 DictType 的私有成员
  friend struct Type;

  // 定义类型的种类常量
  static const TypeKind Kind = TypeKind::DictType;

  // 创建 DictType 实例的静态方法，接受键和值类型的参数
  static DictTypePtr create(TypePtr key, TypePtr value) {
    // 获取键类型的种类
    auto kind = key->kind();
    // 如果键类型是 DynamicType，则获取其动态种类
    if (auto dyn = key->castRaw<DynamicType>()) {
      kind = dyn->dynamicKind();
    }
    // 根据键类型的种类进行选择
    switch (kind) {
      // 支持的键类型包括任意类型、整数、布尔、浮点数、复数、字符串、张量和设备对象
      case TypeKind::AnyType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::ComplexType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
      case TypeKind::DeviceObjType:
        // 创建并返回新的 DictType 实例
        return DictTypePtr(new DictType(std::move(key), std::move(value)));
      default:
        // 抛出错误，不支持当前键类型创建字典
        AT_ERROR(
            "Cannot create dict for key type '",
            key->str(),
            "', only int, float, complex, Tensor, device and string keys are supported");
    }
  }

  // 返回当前字典类型的字符串表示形式
  std::string str() const override {
    std::stringstream ss;
    // 格式化为 "Dict(keyType, valueType)"
    ss << "Dict(" << getKeyType()->str() << ", " << getValueType()->str()
       << ")";
    return ss.str();
  }
  return ss.str();
  }



TypePtr createWithContained(
    std::vector<TypePtr> contained_types) const override {
  if (contained_types.size() != 2) {
    throw std::runtime_error("Expected 2 contained types");
  }
  // 使用给定的两个类型创建一个新的 DictType 对象，并返回其指针
  return create(std::move(contained_types.at(0)), std::move(contained_types.at(1)));
}



const TypePtr& getKeyType() const {
  // 返回 types 中的第一个元素，即键的类型
  return types.at(0);
}



const TypePtr& getValueType() const {
  // 返回 types 中的第二个元素，即值的类型
  return types.at(1);
}



bool hasFreeVariables() const override {
  // 返回标志位，指示是否存在自由变量
  return has_free_variables;
}



at::ArrayRef<TypePtr> containedTypes() const override {
  // 返回包含在 types 中的所有类型的数组引用
  return types;
}



bool equals(const Type& rhs) const override {
  if (auto* dict_rhs = rhs.castRaw<DictType>()) {
    // 检查字典类型对象是否等于另一个字典类型对象
    return *getKeyType() == *(dict_rhs->getKeyType()) &&
        *getValueType() == *(dict_rhs->getValueType());
  }
  return false;
}



// global singleton
// 给定内部类型 T 和标识符，
// 此函数将返回全局单例类型指针，表示类型 List<T>。
// 需要额外的 "identifier" 参数，因为我们有多个容器类型都重用此函数（如 Dict<K, V> 和 unordered_map<K, V>）
static TypePtr get(const std::string& identifier, TypePtr key, TypePtr val);



private:
DictType(TypePtr key, TypePtr value)
    : SharedType(TypeKind::DictType),
      has_free_variables(
          key->hasFreeVariables() || value->hasFreeVariables()) {
  types.reserve(2);
  // 将给定的键和值类型移动到 types 中，并标记是否存在自由变量
  types.push_back(std::move(key));
  types.push_back(std::move(value));
}

std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override;

std::vector<TypePtr> types;  // 用于存储键和值的类型指针数组
bool has_free_variables;     // 指示是否存在自由变量的标志位
};

// 定义 FutureType 结构体，继承自 SingleElementType 类，表示一种特定的类型
struct FutureType;
// 使用智能指针定义 FutureTypePtr 类型为 std::shared_ptr<FutureType>
using FutureTypePtr = std::shared_ptr<FutureType>;

// 定义 FutureType 结构体，继承自 SingleElementType 类，表示一种特定的类型
struct TORCH_API FutureType
    : public SingleElementType<TypeKind::FutureType, FutureType> {
  friend struct Type;
  // 创建 FutureTypePtr 类型的静态工厂方法，接受一个 TypePtr 参数 elem
  template <typename... T>
  static FutureTypePtr create(TypePtr elem) {
    // 使用移动语义创建 FutureTypePtr 对象，传入 elem 参数
    return FutureTypePtr(
        new FutureType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  // 返回当前类型的字符串表示形式，包含元素类型的信息
  std::string str() const override {
    // 创建一个字符串流对象 ss
    std::stringstream ss;
    // 格式化输出类型字符串，包含 "Future(" 和元素类型的 str() 方法返回值
    ss << "Future(" << getElementType()->str() << ")";
    // 返回字符串流的内容作为结果
    return ss.str();
  }

  // 根据包含的类型创建新的 FutureType 对象
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    // 调用 create 方法创建新的 FutureType 对象，传入第一个包含类型
    return create(std::move(contained_types.at(0)));
  }

  // 判断当前类型是否是 rhs 的子类型的扩展实现，why_not 参数用于输出详细信息
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // 如果当前类型是 rhs 的子类型，则返回 true
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    // 尝试将 rhs 强制转换为 FutureType，并判断其包含的元素类型是否是当前类型的子类型
    if (auto rhs_ = rhs.castRaw<FutureType>()) {
      return getElementType()->isSubtypeOfExt(*rhs_->getElementType(), why_not);
    }
    // 否则返回 false
    return false;
  }

 private:
  // 私有构造函数，初始化 SingleElementType 的基类，并传入 elem 参数
  FutureType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  // 返回当前类型的注解字符串表示形式，使用 printer 参数指定的打印器
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    // 创建字符串流对象 ss
    std::stringstream ss;
    // 格式化输出类型字符串，包含 "Future[" 和元素类型的注解字符串
    ss << "Future[" << getElementType()->annotation_str(printer) << "]";
    // 返回字符串流的内容作为结果
    return ss.str();
  }
};

// 定义 AwaitType 结构体，继承自 SingleElementType 类，表示一种特定的类型
struct AwaitType;
// 使用智能指针定义 AwaitTypePtr 类型为 std::shared_ptr<AwaitType>
using AwaitTypePtr = std::shared_ptr<AwaitType>;

// 定义 AwaitType 结构体，继承自 SingleElementType 类，表示一种特定的类型
struct TORCH_API AwaitType
    : public SingleElementType<TypeKind::AwaitType, AwaitType> {
  friend struct Type;
  // 创建 AwaitTypePtr 类型的静态工厂方法，接受一个 TypePtr 参数 elem
  template <typename... T>
  static AwaitTypePtr create(TypePtr elem) {
    // 使用移动语义创建 AwaitTypePtr 对象，传入 elem 参数
    return AwaitTypePtr(
        new AwaitType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  // 返回当前类型的字符串表示形式，包含元素类型的信息
  std::string str() const override {
    // 创建一个字符串流对象 ss
    std::stringstream ss;
    // 格式化输出类型字符串，包含 "Await(" 和元素类型的 str() 方法返回值
    ss << "Await(" << getElementType()->str() << ")";
    // 返回字符串流的内容作为结果
    return ss.str();
  }

  // 根据包含的类型创建新的 AwaitType 对象
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    // 调用 create 方法创建新的 AwaitType 对象，传入第一个包含类型
    return create(std::move(contained_types.at(0)));
  }

  // 判断当前类型是否是 rhs 的子类型的扩展实现，why_not 参数用于输出详细信息
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // 如果当前类型是 rhs 的子类型，则返回 true
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    // 尝试将 rhs 强制转换为 AwaitType，并判断其包含的元素类型是否是当前类型的子类型
    if (auto rhs_ = rhs.castRaw<AwaitType>()) {
      return getElementType()->isSubtypeOfExt(*rhs_->getElementType(), why_not);
    }
    // 否则返回 false
    return false;
  }

 private:
  // 私有构造函数，初始化 SingleElementType 的基类，并传入 elem 参数
  AwaitType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  // 返回当前类型的注解字符串表示形式，使用 printer 参数指定的打印器
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    // 创建字符串流对象 ss
    std::stringstream ss;
    // 格式化输出类型字符串，包含 "Await[" 和元素类型的注解字符串
    ss << "Await[" << getElementType()->annotation_str(printer) << "]";
    // 返回字符串流的内容作为结果
    return ss.str();
  }
};

// 定义 RRefType 结构体，继承自 SingleElementType 类，表示一种特定的类型
struct RRefType;
// 使用智能指针定义 RRefTypePtr 类型为 std::shared_ptr<RRefType>
using RRefTypePtr = std::shared_ptr<RRefType>;

// 定义 RRefType 结构体，继承自 SingleElementType 类，表示一种特定的类型
struct TORCH_API RRefType
    : public SingleElementType<TypeKind::RRefType, RRefType> {
  friend struct Type;
  // 创建 RRefTypePtr 类型的静态工厂方法，接受一个 TypePtr 参数 elem
  template <typename... T>
  static RRefTypePtr create(TypePtr elem) {
    // 使用移动语义创建 RRefTypePtr 对象，传入 elem 参数
    return RRefTypePtr(
        new RRefType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  // 返回当前类型的字符串表示形式，包含元素类型的信息
  std::string str() const override {
    // 创建一个字符串流对象 ss
    std::stringstream ss;
    // 格式化输出类型字符串，包含 "RRef(" 和元素类型的 str() 方法返回值
    ss << "RRef(" << getElementType()->str() << ")";
    // 返回字符串流的内容作为结果
    return ss.str();
  }

  // 根据包含的类型创建新的 RRefType 对象
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    // 调用 create 方法创建新的 RRefType 对象，传入第一个包含类型
    return create(std::move(contained_types.at(0)));
  }

  // 判断当前类型是否是 rhs 的子类型的扩展实现，why_not 参数用于输出详细信息
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // 如果当前类型是 rhs 的子类型，则返回 true
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    // 尝试将 rhs 强制转换为 RRefType，并判断其包含的元素类型是否是当前类型的子类型
    if (auto rhs_ = rhs.castRaw<RRefType>()) {
      return getElementType()->isSubtypeOfExt(*rhs_->getElementType(), why_not);
    }
    // 否则返回 false
    return false;
  }

 private:
  // 私有构造函数，初始化 SingleElementType 的基类，并传入 elem 参数
  RRefType(TypePtr elem) : SingleElementType(std::move(elem
    // 返回一个字符串流的字符串表示形式
    return ss.str();
  }

  // 使用给定的包含类型创建一个新的类型指针，覆盖父类方法
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    // 创建一个包含单一元素类型的新类型指针，使用提供的第一个包含类型
    return create(std::move(contained_types.at(0)));
  }

 private:
  // 私有构造函数，初始化RRefType对象，设置单一元素类型为给定的elem
  RRefType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  // 返回类型注释的字符串表示形式，覆盖父类方法
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    // 创建一个字符串流
    std::stringstream ss;
    // 将类型注释构造为格式为 "RRef[元素类型的注释字符串]" 的字符串
    ss << "RRef[" << getElementType()->annotation_str(printer) << "]";
    // 返回字符串流的字符串表示形式
    return ss.str();
  }
};

// 结束 TupleType 结构体定义

// Any 类型不应出现在类、命名元组或接口中。如果出现，则 Pickler 中会丢失动态类型信息，
// 导致在保存或加载模型后出现难以追踪的 bug。这是因为我们依赖于命名类型中的静态类型来重建加载值的类型标签。
// 解除此限制需要先解决序列化问题。
TORCH_API void checkNoAny(
    const Type& base,
    const char* what,
    const std::string& attrname,
    const TypePtr& attrtype);

// 命名类型，表示一个元组
struct TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
using NameList = std::vector<std::string>;

// TupleType 结构体，继承自 NamedType 类
struct TORCH_API TupleType : public NamedType {

  // 创建具名元组类型，带默认值
  static TupleTypePtr createNamed(const std::optional<c10::QualifiedName>& name,
      const std::vector<std::string>& field_names,
      const std::vector<TypePtr>& field_types,
      std::vector<IValue>& field_defaults);

  // 创建具名元组类型，不带默认值
  static TupleTypePtr createNamed(const std::optional<c10::QualifiedName>& name,
      const std::vector<std::string>& field_names,
      const std::vector<TypePtr>& field_types);

  // 创建具名元组类型，支持 c10::string_view
  static TupleTypePtr createNamed(const std::optional<c10::QualifiedName>& name,
      const std::vector<c10::string_view>& field_names,
      const std::vector<TypePtr>& field_types);

  // 创建无名元组类型
  static TupleTypePtr create(
      std::vector<TypePtr> types) {
    return TupleTypePtr(new TupleType(
        std::move(types),
        c10::nullopt,
        nullptr)); // NOLINT(modernize-make-shared)
  }

  // 创建空的元组类型
  static TupleTypePtr create() {
    return create({});
  }

  // 返回元组类型的成员类型列表
  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }

  // 比较两个类型是否相等
  bool equals(const Type& rhs) const override;

  // 判断是否为 rhs 的子类型
  bool isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const override;

  // 返回类型的字符串表示
  std::string str() const override;

  // 判断是否包含自由变量
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }

  // 返回包含的类型列表
  at::ArrayRef<TypePtr> containedTypes() const override {
    return elements_;
  }

  // 根据包含的类型创建新的元组类型
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return std::shared_ptr<TupleType>(
        new TupleType(std::move(contained_types), name(), schema()));
  }

  // 返回关联的函数模式
  const std::shared_ptr<FunctionSchema>& schema() const {
    return schema_;
  }

  // 返回字段名列表，使用 c10::string_view
  std::optional<std::vector<c10::string_view>> names() const;

  // 类型种类标识为 TupleType
  static const TypeKind Kind = TypeKind::TupleType;

 private:
  // 创建具名元组类型的模板函数
  template <typename S>
  static TupleTypePtr createWithSpec(
      const std::optional<c10::QualifiedName>& name,
      const std::vector<S>& field_names,
      const std::vector<TypePtr>& field_types,
      std::vector<IValue>& field_defaults);

  // TupleType 的私有构造函数
  TupleType(
      std::vector<TypePtr> elements_,
      std::optional<c10::QualifiedName> name,
      std::shared_ptr<FunctionSchema> schema);

  // 比较两个类型是否相等的辅助函数
  bool compare(
      const Type& rhs,
      const std::function<bool(const Type&, const Type&)>& fn) const {
    if (rhs.kind() != kind()) {
      return false;
    }
    // ...
  }
};
    // 获取当前对象的元素列表并赋值给常量引用 l_elements
    const auto& l_elements = elements();
    // 获取 rhs 对象的元素列表并赋值给常量引用 r_elements
    const auto& r_elements = rhs.castRaw<TupleType>()->elements();
    // 检查 l_elements 和 r_elements 的大小是否相等，如果不相等则返回 false
    if (l_elements.size() != r_elements.size())
      return false;
    // 遍历元素列表，比较对应位置上的元素是否满足给定的比较函数 fn
    for (size_t i = 0; i < l_elements.size(); ++i) {
      // 如果 fn 返回 false，则返回 false
      if (!fn(*l_elements[i], *r_elements[i]))
        return false;
    }
    // 如果所有对应位置上的元素都满足比较函数 fn，则返回 true
    return true;
  }

  // 返回该对象的注解字符串的实现，使用指定的打印器进行打印
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override;

  // 元素列表，存储对象的各个元素
  std::vector<TypePtr> elements_;
  // 标记对象是否具有自由变量
  bool has_free_variables_;
  // 共享的函数模式对象指针，描述函数的签名和元数据
  std::shared_ptr<FunctionSchema> schema_;
};

// 所有枚举的共同超类型，仅在操作符注册中使用。
// EnumType <: AnyEnumType 适用于所有枚举类型
struct AnyEnumType;
// SingletonTypePtr 的别名，指向 AnyEnumType 的单例类型指针
using AnyEnumTypePtr = SingletonTypePtr<AnyEnumType>;
// 表示 AnyEnumType 的具体类型，继承自 Type
struct TORCH_API AnyEnumType final : public Type {
  // 检查是否与另一类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "AnyEnumType";
  }
  // 类型的种类，标识为 AnyEnumType
  static const TypeKind Kind = TypeKind::AnyEnumType;
  // 获取全局单例对象
  static AnyEnumTypePtr get();
private:
  // 私有构造函数，初始化类型为 AnyEnumType
  AnyEnumType()
  : Type(TypeKind::AnyEnumType) {}
};

// 表示数字类型的基类
struct NumberType;
// SingletonTypePtr 的别名，指向 NumberType 的单例类型指针
using NumberTypePtr = SingletonTypePtr<NumberType>;
// 表示 Python 数字类型的基类
// 数字类型的子类型层次结构（以 NumberType 为基类）：
// IntType <: NumberType
// FloatType <: NumberType
// ComplexType <: NumberType
//
// 警告：如果添加一个新的 NumberType 的子类型，而该子类型不是通过全局单例表示，
// 则需要将 NumberTypePtr 修改为 SingletonOrSharedTypePtr，并处理 NumberType
// 需要同时继承和不继承 SharedType 的情况！
struct TORCH_API NumberType : public Type {
  // 检查是否与另一类型相等
  bool equals(const Type& rhs) const override;

  // 扩展版本的子类型检查
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // 返回类型的字符串表示
  std::string str() const override {
    return "Scalar"; // 与 PythonArgParser 中的描述保持一致，以便清晰理解
  }
  // 类型的种类，标识为 NumberType
  static const TypeKind Kind = TypeKind::NumberType;
  // 获取全局单例对象
  static NumberTypePtr get();

 protected:
  // 受保护的构造函数，初始化类型为 NumberType
  NumberType(TypeKind kind = TypeKind::NumberType) : Type(kind) {}

  // 注解字符串的具体实现
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    return "number"; // 技术上不是有效的 Python 类型，但在注解解析时需要使用它进行隐式转换
  }
};

// 表示 Python 浮点数类型
struct FloatType;
// SingletonTypePtr 的别名，指向 FloatType 的单例类型指针
using FloatTypePtr = SingletonTypePtr<FloatType>;
// 表示 Python 浮点数类型
struct TORCH_API FloatType : public NumberType {
  // 检查是否与另一类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "float";
  }
  // 扩展版本的子类型检查
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  // 类型的种类，标识为 FloatType
  static const TypeKind Kind = TypeKind::FloatType;
  // 获取全局单例对象
  static FloatTypePtr get();

 private:
  // 私有构造函数，初始化类型为 FloatType
  FloatType() : NumberType(TypeKind::FloatType) {}
  // 注解字符串的具体实现
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    return "float";
  }
};

// 表示 Python 复数类型
struct ComplexType;
// SingletonTypePtr 的别名，指向 ComplexType 的单例类型指针
using ComplexTypePtr = SingletonTypePtr<ComplexType>;
// 表示 Python 复数类型
struct TORCH_API ComplexType : public NumberType {
  // 检查是否与另一类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    // 继承自 NumberType 的标识字符串
    // （这里的实现尚未提供完整，需要在代码中补充）

return "float"; // 返回类型的字符串表示为 "float"
  }
  // 扩展版本的子类型检查
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  // 类型的种类，标识为 ComplexType
  static const TypeKind Kind = TypeKind::ComplexType;
  // 获取全局单例对象
  static ComplexTypePtr get();

 private:
  // 私有构造函数，初始化类型为 ComplexType
  ComplexType() : NumberType(TypeKind::ComplexType) {}
  // 注解字符串的具体实现
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    // 技术上不是有效的 Python 类型，但在注解解析时需要使用它进行隐式转换
    return "complex";
  }
};
    // 返回字符串 "complex"
      return "complex";
    }
    // 检查当前类型是否是给定类型 rhs 的子类型，如果是则返回 true，否则返回 false，并将详细信息写入 why_not 流中
    bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
      // 禁止 Linter 提示下一行可能的虚函数调用的警告
      // 检查当前类型是否是 NumberType 类型，或者调用父类的 isSubtypeOfExt 方法
      return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
    }
    // 类型的静态常量 Kind 被定义为 ComplexType
    static const TypeKind Kind = TypeKind::ComplexType;
    // 全局的 ComplexTypePtr 单例对象的获取方法声明
    static ComplexTypePtr get();
    
    private:
    // ComplexType 的构造函数，初始化为 NumberType 的子类，类型为 ComplexType
    ComplexType() : NumberType(TypeKind::ComplexType) {}
    // 返回类型的注解字符串表示，默认实现返回字符串 "complex"
    std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
      return "complex";
    }
struct SymIntType;
// 引入 SymIntType 结构体来表示在函数模式中使用的 SymInt 类型
// 例如在 `aten::narrow_copy(... SymInt length)` 中使用
// SymInt 用于追踪维度值的算术操作，请参见 [SymInt.h] 获取更多信息
using SymIntTypePtr = SingletonTypePtr<SymIntType>;

struct TORCH_API SymIntType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示形式
  std::string str() const override {
    return "SymInt";
  }
  // 返回类型的注释字符串表示形式
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    return "int";
  }
  // 类型的种类常量
  static const TypeKind Kind = TypeKind::SymIntType;
  // 获取 SymIntType 的全局单例
  static SymIntTypePtr get();

 private:
  // 构造函数，设置类型为 SymIntType
  SymIntType() : Type(TypeKind::SymIntType) {}
};

struct SymFloatType;
using SymFloatTypePtr = SingletonTypePtr<SymFloatType>;

struct TORCH_API SymFloatType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "SymFloat";
  }
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    return "float";
  }
  static const TypeKind Kind = TypeKind::SymFloatType;
  static SymFloatTypePtr get();

 private:
  SymFloatType() : Type(TypeKind::SymFloatType) {}
};

struct SymBoolType;
using SymBoolTypePtr = SingletonTypePtr<SymBoolType>;

struct TORCH_API SymBoolType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "SymBool";
  }
  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    return "bool";
  }
  static const TypeKind Kind = TypeKind::SymBoolType;
  static SymBoolTypePtr get();

 private:
  SymBoolType() : Type(TypeKind::SymBoolType) {}
};

struct IntType;
using IntTypePtr = SingletonTypePtr<IntType>;
// 此类型表示 Python 中的整数
struct TORCH_API IntType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  // 检查此类型是否是 rhs 类型的子类型，扩展版本
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::IntType;
  static IntTypePtr get();

 private:
  IntType() : NumberType(TypeKind::IntType) {}
  // 返回类型的注释字符串表示形式
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    return "int";
  }
};

struct BoolType;
using BoolTypePtr = SingletonTypePtr<BoolType>;
// 此节点表示 Python 中的布尔值
struct TORCH_API BoolType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    // 返回类型的字符串表示形式
    return "bool";
  }


    // 返回字符串 "bool"，这里是类的成员函数的结束
    return "bool";
  }



  static const TypeKind Kind = TypeKind::BoolType;


  // 定义静态成员变量 Kind，其类型为 TypeKind::BoolType
  static const TypeKind Kind = TypeKind::BoolType;



  // global singleton
  static BoolTypePtr get();


  // 声明一个静态方法 get()，用于获取 BoolType 的全局单例对象
  static BoolTypePtr get();



 private:


 private:

  

  BoolType() : Type(TypeKind::BoolType) {}


  // BoolType 的私有构造函数，调用基类 Type 的构造函数来初始化
  BoolType() : Type(TypeKind::BoolType) {}
};

// 结构体声明结束

struct StringType;
using StringTypePtr = SingletonTypePtr<StringType>;

// StringType 结构体定义，继承自 Type 类
// 用于表示 Python 字符串类型
struct TORCH_API StringType : public Type {
  // 判断类型是否相等的方法覆盖
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    // 在 FunctionSchema 和 script 中仅使用 "str" 而非 "string"
    return annotation_str();
  }
  // 返回类型的注解字符串表示的实现
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    return "str";
  }
  // 类型的静态常量 Kind，表示为 StringType 类型
  static const TypeKind Kind = TypeKind::StringType;
  // 获取 StringType 的全局单例对象
  static StringTypePtr get();

 private:
  // 私有构造函数，初始化为 TypeKind::StringType 类型
  StringType() : Type(TypeKind::StringType) {}
};

// 结构体声明结束

struct StorageType;
using StorageTypePtr = SingletonTypePtr<StorageType>;

// StorageType 结构体定义，继承自 Type 类
struct TORCH_API StorageType : public Type {
  // 判断类型是否相等的方法覆盖
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return annotation_str();
  }
  // 返回类型的注解字符串表示的实现
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    return "Storage";
  }
  // 类型的静态常量 Kind，表示为 StorageType 类型
  static const TypeKind Kind = TypeKind::StorageType;
  // 获取 StorageType 的全局单例对象
  static StorageTypePtr get();

 private:
  // 私有构造函数，初始化为 TypeKind::StorageType 类型
  StorageType() : Type(TypeKind::StorageType) {}
};

// 结构体声明结束

struct FunctionType;
using FunctionTypePtr = std::shared_ptr<FunctionType>;

// FunctionType 结构体定义，继承自 NamedType 类
struct TORCH_API FunctionType : public NamedType {
  // 创建 FunctionType 的静态方法，接受 torch::jit::Function* 参数
  static FunctionTypePtr create(torch::jit::Function* function) {
    // 使用 new 运算符创建 FunctionType 的智能指针对象，传入 function
    return FunctionTypePtr(
        new FunctionType(function)); // NOLINT(modernize-make-shared)
  }
  // 判断类型是否相等的方法覆盖
  bool equals(const Type& rhs) const override {
    // 如果 rhs 可以转换为 FunctionType
    if (auto func_type = rhs.cast<FunctionType>()) {
      // 比较两者的 function_ 成员变量
      return func_type->function_ == function_;
    }
    return false;
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "Function";
  }
  // 返回该类型的函数指针 function_
  torch::jit::Function* function() const {
    return function_;
  }
  // 类型的静态常量 Kind，表示为 FunctionType 类型
  static const TypeKind Kind = TypeKind::FunctionType;

 private:
  // 私有构造函数，初始化时接受 torch::jit::Function* 参数
  FunctionType(torch::jit::Function* function);
  // 返回类型的注解字符串表示的实现
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    // 获取名称的限定名称
    const auto& n = name().value();
    return n.qualifiedName();
  }
  // 函数指针，指向 torch::jit::Function 类型
  torch::jit::Function* function_;
};

// 结构体声明结束

struct NoneType;
using NoneTypePtr = SingletonTypePtr<NoneType>;

// NoneType 结构体定义，继承自 Type 类
// 用于表示 Python 的 None 类型
struct TORCH_API NoneType : public Type {
  // 判断类型是否相等的方法覆盖
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "NoneType";
  }
  // 判断是否是 rhs 的子类型的扩展方法
  bool isSubtypeOfExt(const Type& rhs, std::ostream *why_not) const override;

  // 类型的静态常量 Kind，表示为 NoneType 类型
  static const TypeKind Kind = TypeKind::NoneType;
  // 获取 NoneType 的全局单例对象
  static NoneTypePtr get();

 private:
  // 私有构造函数，初始化为 TypeKind::NoneType 类型
  NoneType() : Type(TypeKind::NoneType) {}
};

// 结构体声明结束

struct GeneratorType;
using GeneratorTypePtr = SingletonTypePtr<GeneratorType>;

// GeneratorType 结构体定义，继承自 Type 类
// 用于表示 Generator 类型
struct TORCH_API GeneratorType : public Type {
  // 判断类型是否相等的方法覆盖
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "Generator";
  }


// 返回一个字符串 "Generator"
return "Generator";



  static const TypeKind Kind = TypeKind::GeneratorType;


// 定义静态常量 Kind，表示类型为 GeneratorType
static const TypeKind Kind = TypeKind::GeneratorType;



  // global singleton
  static GeneratorTypePtr get();


// 声明静态成员函数 get()，用于获取全局单例对象
static GeneratorTypePtr get();



 private:
  GeneratorType() : Type(TypeKind::GeneratorType) {}


// GeneratorType 的私有构造函数，初始化基类 Type 的类型为 GeneratorType
GeneratorType() : Type(TypeKind::GeneratorType) {}
};

// 结构体 QuantizerType 的前向声明
struct QuantizerType;

// 使用别名 QuantizerTypePtr 表示 SingletonTypePtr<QuantizerType>
using QuantizerTypePtr = SingletonTypePtr<QuantizerType>;

// 表示量化器类型的结构体
struct TORCH_API QuantizerType : public Type {
  // 判断是否与另一个类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "Quantizer";
  }
  // 静态常量，表示类型的种类
  static const TypeKind Kind = TypeKind::QuantizerType;

  // 获取全局单例对象
  static QuantizerTypePtr get();

 private:
  // 构造函数，初始化类型为 QuantizerType
  QuantizerType() : Type(TypeKind::QuantizerType) {}
};

// 结构体 QSchemeType 的前向声明
struct QSchemeType;

// 使用别名 QSchemeTypePtr 表示 SingletonTypePtr<QSchemeType>
using QSchemeTypePtr = SingletonTypePtr<QSchemeType>;

// 表示量化方案类型的结构体
struct TORCH_API QSchemeType : public Type {
  // 判断是否与另一个类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "QScheme";
  }
  // 静态常量，表示类型的种类
  static const TypeKind Kind = TypeKind::QSchemeType;

  // 获取全局单例对象
  static QSchemeTypePtr get();

 private:
  // 构造函数，初始化类型为 QSchemeType
  QSchemeType() : Type(TypeKind::QSchemeType) {}
};

// 结构体 DeviceObjType 的前向声明
struct DeviceObjType;

// 使用别名 DeviceObjTypePtr 表示 SingletonTypePtr<DeviceObjType>
using DeviceObjTypePtr = SingletonTypePtr<DeviceObjType>;

// 表示设备对象类型的结构体
struct TORCH_API DeviceObjType : public Type {
  // 判断是否与另一个类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "Device";
  }
  // 静态常量，表示类型的种类
  static const TypeKind Kind = TypeKind::DeviceObjType;

  // 获取全局单例对象
  static DeviceObjTypePtr get();

 private:
  // 构造函数，初始化类型为 DeviceObjType
  DeviceObjType() : Type(TypeKind::DeviceObjType) {}
};

// 结构体 StreamObjType 的前向声明
struct StreamObjType;

// 使用别名 StreamObjTypePtr 表示 SingletonTypePtr<StreamObjType>
using StreamObjTypePtr = SingletonTypePtr<StreamObjType>;

// 表示流对象类型的结构体
struct TORCH_API StreamObjType : public Type {
  // 判断是否与另一个类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "Stream";
  }
  // 静态常量，表示类型的种类
  static const TypeKind Kind = TypeKind::StreamObjType;

  // 获取全局单例对象
  static StreamObjTypePtr get();

 private:
  // 构造函数，初始化类型为 StreamObjType
  StreamObjType() : Type(TypeKind::StreamObjType) {}
};

// 结构体 VarType 的前向声明
struct VarType;

// 使用别名 VarTypePtr 表示 std::shared_ptr<VarType>
using VarTypePtr = std::shared_ptr<VarType>;

// 表示类型变量的结构体，用于 FunctionSchema
struct VarType : public SharedType {
  // 创建类型变量的静态方法
  static VarTypePtr create(std::string name_) {
    return VarTypePtr(new VarType(std::move(name_)));
  }
  // 判断是否与另一个类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return name();
  }
  // 返回类型变量的名称
  const std::string& name() const {
    return name_;
  }
  // 判断类型是否有自由变量
  bool hasFreeVariables() const override {
    return true;
  }
  // 静态常量，表示类型的种类
  static const TypeKind Kind = TypeKind::VarType;

 private:
  // 构造函数，初始化类型为 VarType，并设置名称
  VarType(std::string name_)
      : SharedType(TypeKind::VarType), name_(std::move(name_)) {}
  std::string name_;
};

// 结构体 CapsuleType 的前向声明
struct CapsuleType;

// 使用别名 CapsuleTypePtr 表示 SingletonTypePtr<CapsuleType>
using CapsuleTypePtr = SingletonTypePtr<CapsuleType>;

// 表示 Python Capsule 类型的结构体
// 该类型不出现在中间表示(IR)，仅在运行时使用
struct TORCH_API CapsuleType : public Type {
  // 判断是否与另一个类型相等
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  // 返回类型的字符串表示
  std::string str() const override {
    return "Capsule";
  }
  // 静态常量，表示类型的种类
  static const TypeKind Kind = TypeKind::CapsuleType;

 private:
  // 构造函数，初始化类型为 CapsuleType
  CapsuleType() : Type(TypeKind::CapsuleType) {}
};
    // 检查当前对象的类型是否与给定的类型相同，并返回比较结果
    return rhs.kind() == kind();
  }

  // 返回类型名称字符串表示，覆盖基类方法
  std::string str() const override {
    // 返回固定字符串 "Capsule"
    return "Capsule";
  }

  // 定义静态常量，表示该类的类型种类为 CapsuleType
  static const TypeKind Kind = TypeKind::CapsuleType;

  // 返回 CapsuleType 的全局单例对象
  // 注意：此方法是静态方法，可以在不创建类实例的情况下调用
  static CapsuleTypePtr get();
private:
// CapsuleType 的构造函数，初始化 TypeKind 为 CapsuleType
  CapsuleType()
  : Type(TypeKind::CapsuleType) {}
};

// PyObjectType 的前向声明
struct PyObjectType;

// PyObjectTypePtr 是 PyObjectType 的单例类型指针
using PyObjectTypePtr = SingletonTypePtr<PyObjectType>;

// 表示 PyObject 类型的结构体，继承自 Type
// 重写 equals 方法比较类型是否相同，重写 str 方法返回类型名称 "PyObject"
struct TORCH_API PyObjectType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "PyObject";
  }
  // 静态成员变量，表示类型的种类为 PyObjectType
  static const TypeKind Kind = TypeKind::PyObjectType;
  // 获取全局单例对象的静态方法
  static PyObjectTypePtr get();
private:
  // PyObjectType 的构造函数，初始化 TypeKind 为 PyObjectType
  PyObjectType()
  : Type(TypeKind::PyObjectType) {}
};

// 枚举类型 TypeVerbosity，表示类型的详细程度
enum class TypeVerbosity {
  None,           // 无输出
  Type,           // 只输出类型
  TypeAndStride,  // 输出类型和步长
  Full,           // 完整输出
  Symbolic,       // 符号形式输出
  Default = Full, // 默认为 Full
};

// 获取全局的类型输出详细程度
TORCH_API TypeVerbosity type_verbosity();

// 重载操作符 <<，将 Type 类型输出到流中
TORCH_API std::ostream& operator<<(std::ostream& out, const Type& t);

// 模板函数，重载操作符 <<，将 VaryingShape<T> 类型输出到流中
template <typename T>
TORCH_API std::ostream& operator<<(
    std::ostream& out,
    const VaryingShape<T>& t);

// 重载操作符 <<，将 SymbolicShape 类型输出到流中
TORCH_API std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);

// 重载操作符 <<，将 ShapeSymbol 类型输出到流中
TORCH_API std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s);

// 重载操作符 <<，将 Stride 类型输出到流中
TORCH_API std::ostream& operator<<(std::ostream& os, const Stride& s);

// 函数 unshapedType 用于移除 Tensor 子类型的详细信息
// 返回一个新的类型指针，去除了内部 Tensor 的详细形状信息
// 在类型比较和别名分析过程中使用，注意可能会很慢
inline TypePtr unshapedType(const TypePtr& type) {
  // 如果类型是 TensorType 的子类型，则返回 TensorType::get()
  if (type->isSubtypeOf(*TensorType::get())) {
    return TensorType::get();
  }
  // 否则递归处理类型的内部类型
  at::ArrayRef<TypePtr> contained = type->containedTypes();
  if (contained.empty()) {
    return type;
  }
  return type->withContained(fmap(type->containedTypes(), unshapedType));
}

// 根据给定的 NumberType 类型创建对应的 TensorType
// 如果给定类型是整数类型，返回对应的长整型 TensorType
// 如果是浮点数类型，返回对应的双精度浮点型 TensorType
// 如果是布尔类型，返回对应的布尔型 TensorType
// 否则报错，未知的数值类型
inline TypePtr TensorType::fromNumberType(const Type& typ) {
  if (typ.isSubtypeOf(*IntType::get())) {
    return TensorType::createContiguous(at::kLong, at::kCPU, {});
  } else if (typ.isSubtypeOf(*FloatType::get())) {
    return TensorType::createContiguous(at::kDouble, at::kCPU, {});
  } else if (typ.isSubtypeOf(*BoolType::get())) {
    return TensorType::createContiguous(at::kBool, at::kCPU, {});
  } else if (typ.kind() == NumberType::Kind) {
    return TensorType::create(c10::nullopt, at::kCPU, {}, c10::nullopt);
  }
  // 如果类型未知，则抛出错误信息
  TORCH_CHECK(false, "Unknown number type: ", typ.str());
}

// 根据布尔类型创建对应的 TensorType
inline TypePtr TensorType::fromBoolType() {
  return TensorType::createContiguous(at::kBool, at::kCPU, {});
}

// 尝试从给定的 Type 类型中提取标量类型
inline std::optional<c10::ScalarType> tryScalarTypeFromJitType(const Type& type) {
  // 如果类型是 FloatType，则返回对应的标量类型
  if (type == *FloatType::get()) {
    // 如果类型等于默认数据类型的元数据，则返回对应的标量类型
    return at::typeMetaToScalarType(c10::get_default_dtype());
  } else if (type == *IntType::get()) {
    // 如果类型等于整数类型的元数据，则返回长整型的标量类型
    return at::ScalarType::Long;
  } else if (type == *BoolType::get()) {
    // 如果类型等于布尔类型的元数据，则返回布尔类型的标量类型
    return at::ScalarType::Bool;
  }
  // 如果类型不匹配上述任何一种情况，则返回空值
  return c10::nullopt;
// 获取给定类型 `type` 的标量类型
inline at::ScalarType scalarTypeFromJitType(const Type& type) {
  // 尝试从 JIT 类型获取标量类型
  auto result = tryScalarTypeFromJitType(type);
  // 检查结果是否有效，否则抛出错误，指明期望的类型和实际类型字符串
  TORCH_CHECK(
      result,
      "Add new condition, expected Float, Complex, Int, or Bool but got",
      type.str());
  // 返回有效的标量类型
  return *result;
}

// 尝试找到两个类型 `t1` 和 `t2` 的正确的超类型
// 如果找不到超类型，并且 `default_to_union` 是 false，则返回 nullopt；
// 如果是 true，则返回 `Union[t1, t2]`。如果 `t1 == t2` 或者 `t1` 是 `t2` 的类型细化，则返回 `t2`
// 如果 `type_hint` 是 `InterfaceType`，则可以将其作为列表中 `ClassType` 的潜在超类型使用；否则，无法找到和使用某个通用接口类型
TORCH_API std::optional<TypePtr> unifyTypes(
    const TypePtr& t1,
    const TypePtr& t2,
    bool default_to_union = false,
    const TypePtr& type_hint = nullptr);

// 尝试统一类型列表 `elements`，返回统一后的类型或解释原因的输出流
TORCH_API std::optional<TypePtr> unifyTypeList(
    at::ArrayRef<TypePtr> elements,
    std::ostream& why_not,
    bool default_to_union = false,
    const TypePtr& type_hint = nullptr);

namespace detail {
// 获取类型 `T` 的类型指针的模板结构
template <typename T>
struct getTypePtr_ final {
  static decltype(auto) call() {
    // 尝试获取自定义类 `T` 的类型指针，若失败则抛出错误
    return ([]() {
      try {
        return getCustomClassType<T>();
      } catch(const c10::Error&) {
        // 若获取失败，抛出错误，说明该类型无法转换为已知类型
        TORCH_CHECK(
            false,
            "Type ",
            c10::util::get_fully_qualified_type_name<T>(),
            " could not be converted to any of the known types."
        );
      }
    }());
  }
};

// 获取类型 `T` 的可能为虚拟类型的类型指针的模板结构
template <typename T, bool fake>
struct getMaybeFakeTypePtr_ final {
  static decltype(auto) call() {
    return getTypePtr_<T>::call();
  }
};

// 下面是一系列特化的模板结构，用于获取特定类型的类型指针

// 获取 `at::IValue` 类型的类型指针
template <>
struct getTypePtr_<at::IValue> final {
  static decltype(auto) call() {
    return AnyType::get();
  }
};

// 获取 `at::Tensor` 类型的类型指针
template <>
struct getTypePtr_<at::Tensor> final {
  static decltype(auto) call() {
    return TensorType::get();
  }
};

// 获取 `c10::Storage` 类型的类型指针
template <>
struct getTypePtr_<c10::Storage> final {
  static decltype(auto) call() {
    return StorageType::get();
  }
};

// 获取 `c10::Stream` 类型的类型指针
template <>
struct getTypePtr_<c10::Stream> final {
  static decltype(auto) call() {
    return StreamObjType::get();
  }
};

// 获取 `double` 类型的类型指针
template <>
struct getTypePtr_<double> final {
  static decltype(auto) call() {
    return FloatType::get();
  }
};

// 获取 `c10::complex<double>` 类型的类型指针
template <>
struct getTypePtr_<c10::complex<double>> final {
  static decltype(auto) call() {
    return ComplexType::get();
  }
};

// 获取 `int64_t` 类型的类型指针
template <>
struct getTypePtr_<int64_t> final {
  static decltype(auto) call() {
    return IntType::get();
  }
};

// 获取 `DeviceIndex` 类型的类型指针
template <>
struct getTypePtr_<DeviceIndex> final {
  static decltype(auto) call() {
    return IntType::get();
  }
};

// 结构模板的尾部
// 当 fake 为 false 时，返回 SymIntType 类型的指针
template <>
struct getMaybeFakeTypePtr_<SymInt, false> final {
  static decltype(auto) call() {
    return SymIntType::get();
  }
};

// 当 fake 为 true 时，返回 IntType 类型的指针
template <>
struct getMaybeFakeTypePtr_<SymInt, true> final {
  static decltype(auto) call() {
    return IntType::get();
  }
};

// 当 fake 为 false 时，返回 SymFloatType 类型的指针
template <>
struct getMaybeFakeTypePtr_<SymFloat, false> final {
  static decltype(auto) call() {
    return SymFloatType::get();
  }
};

// 当 fake 为 true 时，返回 FloatType 类型的指针
template <>
struct getMaybeFakeTypePtr_<SymFloat, true> final {
  static decltype(auto) call() {
    return FloatType::get();
  }
};

// 当 fake 为 false 时，返回 SymBoolType 类型的指针
template <>
struct getMaybeFakeTypePtr_<SymBool, false> final {
  static decltype(auto) call() {
    return SymBoolType::get();
  }
};

// 当 fake 为 true 时，返回 BoolType 类型的指针
template <>
struct getMaybeFakeTypePtr_<SymBool, true> final {
  static decltype(auto) call() {
    return BoolType::get();
  }
};

// 返回 DeviceObjType 类型的指针
template <>
struct getTypePtr_<c10::Device> final {
  static decltype(auto) call() {
    return DeviceObjType::get();
  }
};

// 返回 BoolType 类型的指针
template <>
struct getTypePtr_<bool> final {
  static decltype(auto) call() {
    return BoolType::get();
  }
};

// 返回 NumberType 类型的指针
template <>
struct getTypePtr_<at::Scalar> final {
  static decltype(auto) call() {
    return NumberType::get();
  }
};

// 返回 QSchemeType 类型的指针
template <>
struct getTypePtr_<c10::QScheme> final {
  static decltype(auto) call() {
    return QSchemeType::get();
  }
};

// 返回 OptionalType 类型的指针，该类型是 GeneratorType 的可选类型
template <>
struct getTypePtr_<at::Generator> final {
  static decltype(auto) call() {
    return TypeFactory::create<OptionalType>(
        TypeFactory::get<GeneratorType>());
  }
};

// 返回 StringType 类型的指针
template <>
struct getTypePtr_<std::string> final {
  static decltype(auto) call() {
    return StringType::get();
  }
};

// 返回 StringType 类型的指针，支持 c10::string_view 类型
template <>
struct getTypePtr_<c10::string_view> final {
  static decltype(auto) call() {
    return StringType::get();
  }
};

// 返回 StringType 类型的指针，支持 at::Dimname 类型
template <>
struct getTypePtr_<at::Dimname> final {
  static decltype(auto) call() {
    return StringType::get();
  }
};

// 当 fake 为 false 时，返回类型为 vector<T> 的列表类型的指针
template <class T, bool fake>
struct getMaybeFakeTypePtr_<std::vector<T>, fake> final {
  static const auto& call() {
    static auto inner_type = getMaybeFakeTypePtr_<T, fake>::call();
    // 每个 vector<T> 的静态单例需要在 .cpp 文件中定义，以避免每个共享库中都有一个实例。
    static auto type = ListType::get("vector", inner_type);
    return type;
  }
};

// 当 fake 为 false 时，返回类型为 ArrayRef<T> 的列表类型的指针
template <class T, bool fake>
struct getMaybeFakeTypePtr_<c10::ArrayRef<T>, fake> final {
  static const auto& call() {
    static auto inner_type = getMaybeFakeTypePtr_<T, fake>::call();
    // 每个 ArrayRef<T> 的静态单例需要在 .cpp 文件中定义，以避免每个共享库中都有一个实例。
    static auto type = ListType::get("ArrayRef", inner_type);
    return type;
  }
};

// 返回类型为 SymIntArrayRef 的列表类型的指针
template <bool fake>
struct getMaybeFakeTypePtr_<c10::SymIntArrayRef, fake> final {
  static const auto& call() {
    static auto type = ListType::create(getMaybeFakeTypePtr_<c10::SymInt, fake>::call());
    return type;
  }
};
// 对于类型为 c10::List<T> 且 fake 为假的情况，获取可能的类型指针
struct getMaybeFakeTypePtr_<c10::List<T>, fake> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 调用模板递归，获取 T 类型的内部类型的静态单例
    static auto inner_type = getMaybeFakeTypePtr_<T, fake>::call();
    // 创建名为 "List" 的列表类型，并与内部类型关联
    static auto type = ListType::get("List", inner_type);
    return type;
  }
};

// 对于类型为 c10::IListRef<T> 且 fake 为假的情况，获取可能的类型指针
template <class T, bool fake>
struct getMaybeFakeTypePtr_<c10::IListRef<T>, fake> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 调用模板递归，获取 T 类型的内部类型的静态单例
    static auto inner_type = getMaybeFakeTypePtr_<T, fake>::call();
    // 创建名为 "List" 的列表类型，并与内部类型关联
    static auto type = ListType::get("List", inner_type);
    return type;
  }
};

// 对于类型为 std::array<T, N> 且 fake 为假的情况，获取可能的类型指针
template <class T, size_t N, bool fake>
struct getMaybeFakeTypePtr_<std::array<T, N>, fake> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 调用模板递归，获取 T 类型的内部类型的静态单例
    static auto inner_type = getMaybeFakeTypePtr_<T, fake>::call();
    // 创建名为 "arrayN"（N 为数组长度）的数组类型，并与内部类型关联
    // 使用字符串拼接确保每个 std::array<T, N> 类型都有唯一的类型指针
    static auto type = ListType::get(std::string("array") + std::to_string(N), inner_type);
    return type;
  }
};

// 对于类型为 std::unordered_map<K, V> 且 fake 为假的情况，获取可能的类型指针
template <class K, class V, bool fake>
struct getMaybeFakeTypePtr_<std::unordered_map<K, V>, fake> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 调用模板递归，获取 K 和 V 类型的内部类型的静态单例
    static auto inner_key_type = getMaybeFakeTypePtr_<K, fake>::call();
    static auto inner_val_type = getMaybeFakeTypePtr_<V, fake>::call();
    // 创建名为 "unordered_map" 的字典类型，并与内部键值类型关联
    static auto type = DictType::get("unordered_map", inner_key_type, inner_val_type);
    return type;
  }
};

// 对于类型为 c10::Dict<K, V> 且 fake 为假的情况，获取可能的类型指针
template <class K, class V, bool fake>
struct getMaybeFakeTypePtr_<c10::Dict<K, V>, fake> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 调用模板递归，获取 K 和 V 类型的内部类型的静态单例
    static auto inner_key_type = getMaybeFakeTypePtr_<K, fake>::call();
    static auto inner_val_type = getMaybeFakeTypePtr_<V, fake>::call();
    // 创建名为 "Dict" 的字典类型，并与内部键值类型关联
    static auto type = DictType::get("Dict", inner_key_type, inner_val_type);
    return type;
  }
};

// 对于类型为 at::optional<T> 且 fake 为假的情况，获取可能的类型指针
template <class T, bool fake>
struct getMaybeFakeTypePtr_<at::optional<T>, fake> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 调用模板递归，获取 T 类型的内部类型的静态单例
    static auto inner_type = getMaybeFakeTypePtr_<T, fake>::call();
    // 创建名为 "optional" 的可选类型，并与内部类型关联
    static auto type = OptionalType::get(inner_type);
    return type;
  }
};

// 对于类型为 at::OptionalIntArrayRef 的情况，获取其类型指针的特化模板
template<>
struct getTypePtr_<at::OptionalIntArrayRef> final {
  // 返回对应类型的静态单例
  static const auto& call() {
    // 获取 IntArrayRef 类型的静态单例
    static auto inner_type = getMaybeFakeTypePtr_<IntArrayRef, false>::call();
    // 静态局部变量，存储了根据 inner_type 创建的 OptionalType 类型对象
    // 这样做是为了保证在一个 .cpp 文件中只有一个该类型的单例实例，
    // 避免在每个共享库中都创建一个单例实例。
    static auto type = OptionalType::get(inner_type);
    // 返回该静态变量 type
    return type;
}
};

template <bool fake>
struct getMaybeFakeTypePtr_<at::OptionalSymIntArrayRef, fake> final {
  static const auto& call() {
    // 如果类型为 at::OptionalSymIntArrayRef，则获取内部类型 SymIntArrayRef 的指针
    // 静态单例需要存放在 .cpp 文件中，否则每个共享库将拥有一个单例实例
    static auto inner_type = getMaybeFakeTypePtr_<SymIntArrayRef, fake>::call();
    // 创建 OptionalType 对象，包装内部类型的指针
    static auto type = OptionalType::get(inner_type);
    return type;
  }
};

template <class... Contained, bool fake>
struct getMaybeFakeTypePtr_<std::tuple<Contained...>, fake> final {
  static const auto& call() {
    // 如果类型为 std::tuple<Contained...>，则创建 TupleType 对象
    static auto type = ([]() {
      // 创建一个 vector，包含每个元素的类型指针
      std::vector<TypePtr> contained_types = {
        (getMaybeFakeTypePtr_<Contained, fake>::call())...
      };
      // 使用 vector 中的类型指针创建 TupleType 对象
      return TupleType::create(std::move(contained_types));
    })();
    return type;
  }
};

template <>
struct getTypePtr_<void> final {
  static decltype(auto) call() {
    // 获取 void 类型的指针，返回 NoneType::get()
    return NoneType::get();
  }
};
} // namespace detail

template <class T>
inline decltype(auto) getTypePtr() {
  // TODO: static_assert that a templated function exists, and throw a friendly
  // error message if not
  // 获取类型 T 的指针，调用 detail::getMaybeFakeTypePtr_<T, false>::call()
  return detail::getMaybeFakeTypePtr_<T, false>::call();
}

template <class T>
inline TypePtr getTypePtrCopy() {
  // TODO: static_assert that a templated function exists, and throw a friendly
  // error message if not
  // 获取类型 T 的指针副本，调用 getTypePtr<T>()
  return getTypePtr<T>();
}

template <class T>
inline decltype(auto) getFakeTypePtr() {
  // 获取虚假类型 T 的指针，调用 detail::getMaybeFakeTypePtr_<T, true>::call()
  return detail::getMaybeFakeTypePtr_<T, true>::call();
}

template <class T>
inline TypePtr getFakeTypePtrCopy() {
  // 获取虚假类型 T 的指针副本，调用 getFakeTypePtr<T>()
  return getFakeTypePtr<T>();
}

using TypeEnv = std::unordered_map<std::string, TypePtr>;

struct MatchTypeReturn {
  MatchTypeReturn(std::string reason) : reason_(std::move(reason)) {}

  static MatchTypeReturn Success() {
    return MatchTypeReturn();
  }

  bool success() const {
    return !reason_.has_value();
  }

  const std::string& reason() const {
    return reason_.value();
  }

 private:
  // 构造函数私有化，用于创建 MatchTypeReturn 对象，标识是否成功匹配及失败原因
  MatchTypeReturn() : reason_(c10::nullopt) {}

  std::optional<std::string> reason_; // 如果没有匹配，则包含失败原因
};

// 尝试匹配形式类型中的类型变量到实际类型中的类型变量，并将其添加到 type_env 中
// 如果无法匹配，则返回一个 MatchTypeReturn，其中 r.success() == false，r.reason() 描述了为什么无法匹配
// 注意：可以成功匹配形式类型，但对于形式类型中未定义的类型变量也是可能的。特别地，None 匹配 Optional[T]，但不定义 T 的值。
TORCH_API MatchTypeReturn
matchTypeVariables(const TypePtr& formal, const TypePtr& actual, TypeEnv& type_env);

// 用 type_env 中的值替换 `type` 中出现的类型变量。如果 `type` 中使用的变量在 `type_env` 中不存在，则返回 nullptr
TORCH_API TypePtr tryEvalTypeVariables(const TypePtr& type, TypeEnv& type_env);

// 检查是否可以从成员推断出元素类型
TORCH_API bool elementTypeCanBeInferredFromMembers(const TypePtr& elem_type);

struct InterfaceType;
// 使用别名 InterfaceTypePtr 表示 std::shared_ptr<InterfaceType>

// 接口类型是一组抽象方法的列表，一个类如果实现了这些方法，就隐式地满足了这个接口。

// InterfaceType 和 ClassType 之间的子类型关系:
// 当 lhs（ClassType 或 InterfaceType）是 rhs 的子类型时：
// 1. lhs 方法是 rhs 方法的超集
// 2. 如果 rhs 是模块接口，那么 lhs 必须是模块接口或模块本身

struct TORCH_API InterfaceType : public NamedType {
  // 创建一个 InterfaceType 对象，指定其限定名和是否为模块接口
  static InterfaceTypePtr create(QualifiedName qualifiedName, bool is_module=false);

  // 判断是否等于另一种 Type
  bool equals(const Type& rhs) const override {
    // 如果 rhs 能转换为 InterfaceType，则比较子类型关系是否成立
    if (auto user_rhs = rhs.castRaw<InterfaceType>()) {
      return isSubTypeImpl(*this, *user_rhs, nullptr) &&
          isSubTypeImpl(*user_rhs, *this, nullptr);
    }
    return false;
  }

  // 返回描述类型的字符串
  std::string str() const override {
    return std::string("InterfaceType<") + name()->name() + ">";
  }

  // 检查是否是 rhs 的子类型，并可能输出原因到 why_not
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // 尝试查找接口中的方法，如果找不到则返回 nullptr
  const FunctionSchema* getMethod(const std::string& name) const;
  
  // 向接口添加方法
  void addMethod(FunctionSchema schema);
  
  // 返回接口中定义的所有方法的引用
  const std::vector<FunctionSchema>& methods() const {
    return *methods_;
  }

  // 返回接口是否是模块接口
  bool is_module() const override {
    return is_module_;
  }

  // 定义类型为 InterfaceType 的常量 Kind
  static const TypeKind Kind = TypeKind::InterfaceType;
  
  // 析构函数
  ~InterfaceType() override;

 private:
  // 私有构造函数，创建一个 InterfaceType 对象，指定其限定名和是否为模块接口
  InterfaceType(QualifiedName name, bool is_module);

  // 实现接口类型的子类型关系判断
  static bool isSubTypeImpl(const InterfaceType& lhs, const InterfaceType& rhs, std::ostream* why_not);

  // 返回注解字符串的实现，用于打印类型信息
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    return name()->qualifiedName();
  }

  // 使用 shared_ptr，避免头文件依赖 FunctionSchema.h
  std::shared_ptr<std::vector<FunctionSchema>> methods_;

  // 标志位，用于区分接口类型是否来自模块
  bool is_module_;
};

// 模板类 EnumerationType，继承自 Type，用于定义枚举类型
template <TypeKind K>
struct EnumerationType : public Type {
  // 定义类型为 K 的常量 Kind
  static const TypeKind Kind = K;

  // 判断是否等于另一种 Type
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }

protected:
  // 构造函数，初始化类型为 K 的枚举类型
  EnumerationType() : Type(Kind) {}
};

// ScalarTypeType 枚举类型
struct ScalarTypeType;
using ScalarTypeTypePtr = SingletonTypePtr<ScalarTypeType>;
struct TORCH_API ScalarTypeType : public EnumerationType<TypeKind::ScalarTypeType> {
  // 返回描述类型的字符串
  std::string str() const override {
    return "ScalarType";
  }
  
  // 定义类型为 ScalarTypeType 的常量 Kind
  static const TypeKind Kind = TypeKind::ScalarTypeType;

  // 返回全局单例对象
  static ScalarTypeTypePtr get();

private:
  // 构造函数，初始化类型为 ScalarTypeType 的枚举类型
  ScalarTypeType() : EnumerationType() {}
};

// MemoryFormatType 枚举类型
struct MemoryFormatType;
using MemoryFormatTypePtr = SingletonTypePtr<MemoryFormatType>;
// 定义一个继承自 EnumerationType 的 MemoryFormatType 结构体，表示内存格式类型
struct TORCH_API MemoryFormatType : public EnumerationType<TypeKind::MemoryFormatType> {
    // 返回描述字符串 "MemoryFormat"
    std::string str() const override {
        return "MemoryFormat";
    }
    // 类型的 Kind 常量，标识为 MemoryFormatType
    static const TypeKind Kind = TypeKind::MemoryFormatType;
    // 全局单例，返回 MemoryFormatType 的指针
    static MemoryFormatTypePtr get();

private:
    // 构造函数，初始化 EnumerationType
    MemoryFormatType() : EnumerationType() {}
};

// LayoutType 结构体声明
struct LayoutType;
// LayoutTypePtr 类型定义为 LayoutType 的单例指针
using LayoutTypePtr = SingletonTypePtr<LayoutType>;
// 定义一个继承自 EnumerationType 的 LayoutType 结构体，表示布局类型
struct TORCH_API LayoutType : public EnumerationType<TypeKind::LayoutType> {
    // 返回描述字符串 "Layout"
    std::string str() const override {
        return "Layout";
    }
    // 类型的 Kind 常量，标识为 LayoutType
    static const TypeKind Kind = TypeKind::LayoutType;
    // 全局单例，返回 LayoutType 的指针
    static LayoutTypePtr get();

private:
    // 构造函数，初始化 EnumerationType
    LayoutType() : EnumerationType() {}
};

// detail 命名空间下的模板特化，获取 ScalarType 的类型指针，非虚假类型为 ScalarTypeType::get()
template <>
struct getMaybeFakeTypePtr_<c10::ScalarType, false> final {
    static decltype(auto) call() {
        return ScalarTypeType::get();
    }
};

// detail 命名空间下的模板特化，获取 Layout 的类型指针，非虚假类型为 LayoutType::get()
template <>
struct getMaybeFakeTypePtr_<c10::Layout, false> final {
    static decltype(auto) call() {
        return LayoutType::get();
    }
};

// detail 命名空间下的模板特化，获取 MemoryFormat 的类型指针，非虚假类型为 MemoryFormatType::get()
template <>
struct getMaybeFakeTypePtr_<c10::MemoryFormat, false> final {
    static decltype(auto) call() {
        return MemoryFormatType::get();
    }
};

// detail 命名空间下的模板特化，获取 ScalarType 的类型指针，虚假类型为 IntType::get()
template <>
struct getMaybeFakeTypePtr_<c10::ScalarType, true> final {
    static decltype(auto) call() {
        return IntType::get();
    }
};

// detail 命名空间下的模板特化，获取 Layout 的类型指针，虚假类型为 IntType::get()
template <>
struct getMaybeFakeTypePtr_<c10::Layout, true> final {
    static decltype(auto) call() {
        return IntType::get();
    }
};

// detail 命名空间下的模板特化，获取 MemoryFormat 的类型指针，虚假类型为 IntType::get()
template <>
struct getMaybeFakeTypePtr_<c10::MemoryFormat, true> final {
    static decltype(auto) call() {
        return IntType::get();
    }
};

// 所有列表的通用超类型，List[T] <: AnyList for all T
struct AnyListType;
// AnyListTypePtr 类型定义为 AnyListType 的单例指针
using AnyListTypePtr = SingletonTypePtr<AnyListType>;
// 定义一个继承自 Type 的 AnyListType 结构体，表示所有列表类型的通用超类型
struct TORCH_API AnyListType : public Type {
    // 比较函数，判断类型是否相等
    bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }
    // 返回描述字符串 "list"
    std::string str() const override {
        return "list";
    }
    // 类型的 Kind 常量，标识为 AnyListType
    static const TypeKind Kind = TypeKind::AnyListType;
    // 全局单例，返回 AnyListType 的指针
    static AnyListTypePtr get();

private:
    // 构造函数，初始化 Type
    AnyListType()
    : Type(TypeKind::AnyListType) {}
};

// 所有元组的通用超类型，Tuple[T...] <: AnyTuple for all T
struct AnyTupleType;
// AnyTupleTypePtr 类型定义为 AnyTupleType 的单例指针
using AnyTupleTypePtr = SingletonTypePtr<AnyTupleType>;
// 定义一个继承自 Type 的 AnyTupleType 结构体，表示所有元组类型的通用超类型
struct TORCH_API AnyTupleType : public Type {
    // 比较函数，判断类型是否相等
    bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }
    // 返回描述字符串 "tuple"
    std::string str() const override {
        return "tuple";
    }
    // 类型的 Kind 常量，标识为 AnyTupleType
    static const TypeKind Kind = TypeKind::AnyTupleType;
    // 全局单例，返回 AnyTupleType 的指针
    static AnyTupleTypePtr get();

private:
    // 构造函数，初始化 Type
    AnyTupleType()
    : Type(TypeKind::AnyTupleType) {}
};

// 所有类的通用超类型，ClassType <: AnyClassType for all classes
struct AnyClassType;
// AnyClassTypePtr 类型定义为 AnyClassType 的单例指针
using AnyClassTypePtr = SingletonTypePtr<AnyClassType>;
// 定义一个继承自 Type 的 AnyClassType 结构体，表示所有类类型的通用超类型
struct TORCH_API AnyClassType : public Type {
    // 比较函数，判断类型是否相等
    bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }
    // 返回描述字符串 "ClassType"
    std::string str() const override {
    // 返回字符串 "AnyClassType"，表示这个函数返回值为字符串类型 "AnyClassType"
    return "AnyClassType";
  }
  // 定义静态常量 Kind，其值为 TypeKind::AnyClassType
  static const TypeKind Kind = TypeKind::AnyClassType;
  // 声明全局静态函数 get()，返回类型为 AnyClassTypePtr
  // 这个函数用于获取 AnyClassType 的单例对象
  static AnyClassTypePtr get();
private:
  AnyClassType()
  : Type(TypeKind::AnyClassType) {}
};



// 私有构造函数，创建一个类型为 AnyClassType 的对象
private:
  AnyClassType()
  : Type(TypeKind::AnyClassType) {}
};



template<>
inline typename detail::CastReturnType<NamedType>::type Type::cast<NamedType>() {
  // 检查当前对象的类型是否为 TupleType、FunctionType、ClassType 或 InterfaceType
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    // 如果是，则将当前对象转换为 NamedType 对象，并返回其共享指针
    return std::static_pointer_cast<NamedType>(static_cast<NamedType *>(this)->shared_from_this());
  }
  // 否则返回空指针
  return nullptr;
}



template<>
inline typename detail::CastConstReturnType<NamedType>::type Type::cast<NamedType>() const {
  // 检查当前对象的类型是否为 TupleType、FunctionType、ClassType 或 InterfaceType
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    // 如果是，则将当前对象转换为常量 NamedType 对象，并返回其共享指针
    return std::static_pointer_cast<const NamedType>(static_cast<const NamedType *>(this)->shared_from_this());
  }
  // 否则返回空指针
  return nullptr;
}



template<>
inline const NamedType* Type::castRaw<NamedType>() const {
  // 检查当前对象的类型是否为 TupleType、FunctionType、ClassType 或 InterfaceType
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    // 如果是，则直接将当前对象转换为 NamedType 指针，并返回
    return static_cast<const NamedType*>(this);
  }
  // 否则返回空指针
  return nullptr;
}



// 用于推断 Python 对象的 IValue 类型时作为返回类型。
struct InferredType {
  /* implicit */ InferredType(TypePtr type) : type_(std::move(type)) {}
  /* implicit */ InferredType(std::string reason)
      : type_(nullptr), reason_(std::move(reason)) {}
  
  // 返回该结构体持有的类型
  TypePtr type() const {
    TORCH_INTERNAL_ASSERT(
        type_,
        "Tried to get the type from an InferredType but the type is null. ",
        "Reason: ",
        reason_);
    return type_;
  }
  
  // 返回推断是否成功
  bool success() const {
    return type_ != nullptr;
  }
  
  // 返回推断失败的原因
  const std::string& reason() const {
    TORCH_INTERNAL_ASSERT(!type_);
    return reason_;
  }

private:
  TypePtr type_;         // 持有的类型指针
  std::string reason_;   // 推断失败的原因
};



TORCH_API bool containsAnyType(const TypePtr& type);



} // namespace c10



// namespace c10 结束
} // namespace c10
```