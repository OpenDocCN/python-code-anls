# `.\pytorch\torch\csrc\jit\tensorexpr\ir.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <string>
// 引入处理字符串的标准库

#include <utility>
// 引入处理实用工具的标准库

#include <vector>
// 引入处理向量的标准库

#include <torch/csrc/jit/tensorexpr/exceptions.h>
// 引入处理异常的头文件

#include <torch/csrc/jit/tensorexpr/expr.h>
// 引入表达式相关的头文件

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
// 引入前向声明的头文件

#include <torch/csrc/jit/tensorexpr/stmt.h>
// 引入语句相关的头文件

#include <ATen/core/ivalue.h>
// 引入处理IValue的头文件

namespace torch {
namespace jit {
namespace tensorexpr {

enum CompareSelectOperation {
  kEQ = 0,
  kGT,
  kGE,
  kLT,
  kLE,
  kNE,
};
// 枚举比较选择操作的类型及其对应的整数值

enum CompareSelectBias {
  kUnbiased,
  kLikely,
  kUnlikely,
};
// 枚举比较选择的偏向类型

inline int getPrecedence(IRNodeType ty) {
  // 根据不同的节点类型返回其运算符优先级，匹配 C++ 运算符优先级规则
  // SEE: https://en.cppreference.com/w/cpp/language/operator_precedence
  switch (ty) {
    case kPrimitive:
      return 0;
    case kCast:
    case kBitCast:
      return 2;
    case kAdd:
    case kSub:
      return 6;
    case kMul:
    case kDiv:
    case kMod:
      return 5;
    case kMax:
    case kMin:
      return 99;
    case kAnd:
      return 11;
    case kOr:
      return 13;
    case kLshift:
    case kRshift:
      return 7;
    case kXor:
      return 12;
    case kCompareSelect:
      return 16;
    default:
      return 99;
  }
}
// 内联函数，根据节点类型返回运算符的优先级

class TORCH_API Cast : public ExprNode<Cast> {
 public:
  ExprPtr src_value() const {
    return src_value_;
  }
  // 返回表达式节点的源值指针

  void set_src_value(ExprPtr src_value) {
    src_value_ = std::move(src_value);
  }
  // 设置表达式节点的源值

  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(alloc<Cast>(dtype, src_value.node()));
  }
  // 创建一个类型转换的表达式节点

  Cast(Dtype dtype, ExprPtr src_value)
      : ExprNodeBase(dtype, kCast), src_value_(std::move(src_value)) {}
  // 构造函数，初始化类型和源值

  bool isConstant() const override {
    return src_value_->isConstant();
  }
  // 判断表达式是否为常量

 private:
  ExprPtr src_value_;
  // 表达式节点的源值指针
};

template <typename T>
ExprHandle cast(const ExprHandle& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}
// 模板函数，进行类型转换

// This is a bitwise cast, akin to bitcast in LLVM
class TORCH_API BitCast : public ExprNode<BitCast> {
 public:
  ExprPtr src_value() const {
    return src_value_;
  }
  // 返回表达式节点的源值指针

  void set_src_value(ExprPtr src_value) {
    src_value_ = std::move(src_value);
  }
  // 设置表达式节点的源值

  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(alloc<BitCast>(dtype, src_value.node()));
  }
  // 创建一个位级转换的表达式节点

  BitCast(Dtype dtype, ExprPtr src_value)
      : ExprNodeBase(dtype, kBitCast), src_value_(std::move(src_value)) {
    TORCH_CHECK(src_value_->dtype().byte_size() == dtype.byte_size());
  }
  // 构造函数，初始化类型和源值，并检查字节大小是否匹配

  bool isConstant() const override {
    return src_value_->isConstant();
  }
  // 判断表达式是否为常量

 private:
  ExprPtr src_value_;
  // 表达式节点的源值指针
};

template <typename T>
ExprHandle bitcast(const ExprHandle& src_value) {
  return BitCast::make(
      Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}
// 模板函数，进行位级转换

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  ExprPtr lhs() const {
    // 返回表达式节点的左操作数
      return this->lhs_;
    
    
    
    // 返回表达式节点的右操作数
      ExprPtr rhs() const {
        return this->rhs_;
      }
    
    
    
    // 设置左操作数，使用 std::move 转移所有权
      void set_lhs(ExprPtr lhs) {
        lhs_ = std::move(lhs);
      }
    
    
    
    // 设置右操作数，使用 std::move 转移所有权
      void set_rhs(ExprPtr rhs) {
        rhs_ = std::move(rhs);
      }
    
    
    
    // 创建并返回一个表达式节点对象
      static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) {
        return ExprHandle(alloc<Op>(lhs.node(), rhs.node()));
      }
    
    
    
    // 构造函数，初始化二元操作表达式节点
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      BinaryOpNode(
          ExprPtr lhs_v,                        // 左操作数的表达式指针
          ExprPtr rhs_v,                        // 右操作数的表达式指针
          IRNodeType expr_type,                 // 表达式节点类型
          ScalarType ret_type = ScalarType::Undefined) // 返回类型，默认未定义
          : ExprNode<Op>(                       // 调用基类构造函数初始化
                // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
                BinaryOpDtype(lhs_v->dtype(), rhs_v->dtype(), ret_type), // 使用左右操作数的数据类型和返回类型初始化二元操作数据类型
                expr_type),
            lhs_(CastIfNeeded(std::move(lhs_v), ExprNode<Op>::dtype())), // 初始化左操作数，可能需要类型转换
            rhs_(CastIfNeeded(std::move(rhs_v), ExprNode<Op>::dtype())) {} // 初始化右操作数，可能需要类型转换
    
    
    
    // 私有函数：根据需要进行类型转换并返回表达式指针
      static ExprPtr CastIfNeeded(ExprPtr expr, Dtype dst_dtype) {
        if (expr->dtype() == dst_dtype) {     // 如果表达式已经是目标类型，则直接返回
          return expr;
        }
        return Cast::make(dst_dtype, ExprHandle(std::move(expr))).node(); // 否则进行类型转换并返回节点
      }
    
    
    
    // 左操作数和右操作数的表达式指针
      ExprPtr lhs_;
      ExprPtr rhs_;
};  // 结束类定义

namespace detail {
template <typename T>
void bin_op_deducer(BinaryOpNode<T>);  // 声明模板函数 bin_op_deducer，接受 BinaryOpNode<T> 类型参数
bool bin_op_deducer(...);  // 声明函数 bin_op_deducer，接受可变参数，用于类型推导
}  // namespace detail

class TORCH_API Add : public BinaryOpNode<Add> {
 public:
  Add(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kAdd) {}  // Add 类构造函数，调用基类构造函数初始化

};

class TORCH_API Sub : public BinaryOpNode<Sub> {
 public:
  Sub(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kSub) {}  // Sub 类构造函数，调用基类构造函数初始化

};

class TORCH_API Mul : public BinaryOpNode<Mul> {
 public:
  Mul(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMul) {}  // Mul 类构造函数，调用基类构造函数初始化

};

class TORCH_API Div : public BinaryOpNode<Div> {
 public:
  Div(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kDiv) {}  // Div 类构造函数，调用基类构造函数初始化

};

class TORCH_API Mod : public BinaryOpNode<Mod> {
 public:
  Mod(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMod) {}  // Mod 类构造函数，调用基类构造函数初始化

};

template <typename Op>
class BitwiseOpNode : public BinaryOpNode<Op> {
 public:
  BitwiseOpNode(ExprPtr lhs, ExprPtr rhs, IRNodeType type)
      : BinaryOpNode<Op>(std::move(lhs), std::move(rhs), type) {}  // BitwiseOpNode 模板类构造函数，调用基类构造函数初始化

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) {
    if (!lhs.dtype().is_integral()) {
      throw unsupported_dtype();  // 如果 lhs 不是整数类型，则抛出异常
    }
    if (lhs.dtype() != rhs.dtype()) {
      throw malformed_input("lhs/rhs dtype mismatch");  // 如果 lhs 和 rhs 类型不匹配，则抛出异常
    }
    return BinaryOpNode<Op>::make(lhs, rhs);  // 调用基类的静态方法 make 处理 lhs 和 rhs
  }
};

class TORCH_API And : public BitwiseOpNode<And> {
 public:
  And(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kAnd) {}  // And 类构造函数，调用基类构造函数初始化

};

class TORCH_API Or : public BitwiseOpNode<Or> {
 public:
  Or(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kOr) {}  // Or 类构造函数，调用基类构造函数初始化

};

class TORCH_API Xor : public BitwiseOpNode<Xor> {
 public:
  Xor(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kXor) {}  // Xor 类构造函数，调用基类构造函数初始化

};

class TORCH_API Lshift : public BitwiseOpNode<Lshift> {
 public:
  Lshift(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kLshift) {}  // Lshift 类构造函数，调用基类构造函数初始化

};

class TORCH_API Rshift : public BitwiseOpNode<Rshift> {
 public:
  Rshift(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kRshift) {}  // Rshift 类构造函数，调用基类构造函数初始化

};

// TODO: add TORCH_API
// Currently adding it results in a compilation error on Windows
class Max : public BinaryOpNode<Max> {
 private:
  bool propagate_nans_;

 public:
  Max(ExprPtr lhs, ExprPtr rhs, bool propagate_nans)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMax),
        propagate_nans_(propagate_nans) {}  // Max 类构造函数，调用基类构造函数初始化

  bool propagate_nans() const {
    return propagate_nans_;  // 返回 propagate_nans_ 成员变量
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;  // 删除静态方法 make 的定义
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      bool propagate_nans) {  // 静态方法 make 的重载，接受 propagate_nans 参数
    return ExprHandle(alloc<Max>(lhs.node(), rhs.node(), propagate_nans));


// 返回一个表达式处理器对象，该对象调用 alloc 函数生成一个 Max 节点，其中包含左操作数节点 lhs.node()、右操作数节点 rhs.node() 以及是否传播 NaN（Not a Number）的标志 propagate_nans。
};

// TODO: add TORCH_API
// Currently adding it results in a compilation error on Windows

// 定义 Min 类，继承自 BinaryOpNode<Min>
class Min : public BinaryOpNode<Min> {
 private:
  bool propagate_nans_;

 public:
  // Min 构造函数，初始化左右表达式和是否传播 NaN 值
  Min(ExprPtr lhs, ExprPtr rhs, bool propagate_nans)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMin),
        propagate_nans_(propagate_nans) {}

  // 返回是否传播 NaN 值的标志
  bool propagate_nans() const {
    return propagate_nans_;
  }

  // 删除 make 函数，避免通过 ExprHandle 创建 Min 对象
  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;

  // 创建 Min 对象的静态方法，传入左右表达式和是否传播 NaN 值
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      bool propagate_nans) {
    return ExprHandle(alloc<Min>(lhs.node(), rhs.node(), propagate_nans));
  }
};

// Encode typed immediate values e.g. IntImm, FloatImm.

// 定义 IMM_DECLARE 宏，用于声明各种类型的立即数类
#define IMM_DECLARE(Type, Name)                               \
  class TORCH_API Name##Imm : public ExprNode<Name##Imm> {    \
   public:                                                    \
    // Name##Imm 构造函数，初始化值
    Name##Imm(Type value)                                     \
        : ExprNodeBase(k##Name, kPrimitive), value_(value) {} \
    // 判断是否是常数节点，覆盖基类方法
    bool isConstant() const override {                        \
      return true;                                            \
    }                                                         \
    // 返回存储的值
    Type value() const {                                      \
      return value_;                                          \
    }                                                         \
    // 创建 Name##Imm 对象的静态方法，传入值并返回 ExprHandle
    static ExprHandle make(Type value) {                      \
      return ExprHandle(alloc<Name##Imm>(value));             \
    }                                                         \
                                                              \
   private:                                                   \
    Type value_;                                              \
  };
// 使用 AT_FORALL_SCALAR_TYPES_AND3 宏展开 IMM_DECLARE 宏，声明各种类型的立即数类
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_DECLARE);
#undef IMM_DECLARE

// Get immediate by ScalarType.

// 根据 ScalarType 获取对应类型的立即数表达式指针
template <typename T>
ExprPtr getImmediateByType(ScalarType immType, T initialVal) {
  switch (immType) {
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    return alloc<Name##Imm>(Type(initialVal));
    // 使用 AT_FORALL_SCALAR_TYPES_AND3 宏展开 TYPE_CASE 宏，处理各种类型的情况
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

// 根据 Dtype 获取对应类型的立即数表达式指针
template <typename T>
ExprPtr getImmediateByType(Dtype dtype, T initialVal) {
  return getImmediateByType<T>(dtype.scalar_type(), initialVal);
}

// 返回与给定表达式类型相同的立即数表达式指针
template <typename T>
ExprPtr immLike(const ExprPtr& e, T v) {
  return getImmediateByType<T>(e->dtype(), v);
}

// 返回与给定表达式类型相同的立即数表达式指针
template <typename T>
ExprPtr immLike(const ExprHandle& e, T v) {
  return immLike(e.node(), v);
}

// 返回表达式的整数值（如果是整数类型）
inline std::optional<int64_t> intValue(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)      \
  if (auto v = to<Name##Imm>(e)) { \
    return v->value();             \
  }
  // 使用 AT_FORALL_INT_TYPES 宏展开 TYPE_CASE 宏，处理各种整数类型的情况
  AT_FORALL_INT_TYPES(TYPE_CASE);
#undef TYPE_CASE
  return c10::nullopt;
}
// 返回给定表达式的整数值的可选类型，使用表达式节点进行转换
inline std::optional<int64_t> intValue(const ExprHandle& e) {
    return intValue(e.node());
}

// 将表达式解析为指定类型的立即数值
template <typename T>
T immediateAs(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value();                     \
  }
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  throw unsupported_dtype();
  return 0;
}

// 将表达式解析为指定类型的立即数值
template <typename T>
T immediateAs(const ExprHandle& e) {
    return immediateAs<T>(e.node());
}

// 检查表达式是否与给定值相等
template <typename T>
bool immediateEquals(const ExprPtr& e, T val) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value() == val;              \
  }
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  throw unsupported_dtype();
  return false;
}

// 判断表达式是否为负数
TORCH_API bool immediateIsNegative(const ExprPtr& e);

// 判断表达式是否为正数
TORCH_API bool immediateIsPositive(const ExprPtr& e);

// 判断表达式是否为零
TORCH_API bool immediateIsZero(const ExprPtr& e);

// 表示一个阶梯矢量节点：[base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class TORCH_API Ramp : public ExprNode<Ramp> {
 public:
  // 获取基础值表达式
  ExprPtr base() const {
    return base_;
  }
  // 获取步长表达式
  ExprPtr stride() const {
    return stride_;
  }

  // 设置基础值表达式
  void set_base(ExprPtr base) {
    base_ = std::move(base);
  }

  // 设置步长表达式
  void set_stride(ExprPtr stride) {
    stride_ = std::move(stride);
  }

  // 创建阶梯节点的静态方法
  static ExprHandle make(
      const ExprHandle& base,
      const ExprHandle& stride,
      int lanes) {
    if (stride.dtype() != base.dtype()) {
      throw malformed_input("Bad stride in Ramp");
    }
    return ExprHandle(alloc<Ramp>(base.node(), stride.node(), lanes));
  }

  // 获取阶梯节点的步长数
  int lanes() const {
    return lanes_;
  }

  // 构造函数，初始化阶梯节点
  Ramp(ExprPtr base, ExprPtr stride, int lanes)
      : ExprNodeBase(Dtype(base->dtype(), lanes)),
        base_(std::move(base)),
        stride_(std::move(stride)),
        lanes_(lanes) {}

 private:
  ExprPtr base_;   // 基础值表达式
  ExprPtr stride_; // 步长表达式
  int lanes_;      // 阶梯节点的步长数
};

// 表示一个加载节点
class TORCH_API Load : public ExprNode<Load> {
 public:
  // 获取基础句柄
  VarPtr base_handle() const {
    return buf_->base_handle();
  }
  // 获取索引列表
  std::vector<ExprPtr> indices() const {
    return indices_;
  }
  // 获取扁平化索引表达式
  ExprPtr flat_index() const {
    TORCH_CHECK(indices_.size() == 1, "Indices haven't been flattened.");
    return indices_[0];
  }
  // 获取缓冲区
  BufPtr buf() const {
    return buf_;
  }

  // 设置缓冲区
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  // 设置索引列表
  void set_indices(std::vector<ExprPtr> indices) {
    indices_ = std::move(indices);
  }

  // 创建加载节点的静态方法
  static ExprHandle make(
      Dtype dtype,
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices);
  static ExprHandle make(
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices);

  // 构造函数，初始化加载节点
  Load(Dtype dtype, BufPtr base_handle, std::vector<ExprPtr> indices);
  Load(BufPtr base_handle, const std::vector<ExprPtr>& indices);

 private:
  BufPtr buf_;              // 缓冲区
  std::vector<ExprPtr> indices_;  // 索引列表
};
class TORCH_API Broadcast : public ExprNode<Broadcast> {
 public:
  // 返回广播操作的值表达式
  ExprPtr value() const {
    return value_;
  }

  // 设置广播操作的值表达式
  void set_value(ExprPtr value) {
    value_ = std::move(value);
  }

  // 返回广播操作的通道数
  int lanes() const {
    return lanes_;
  }

  // 创建广播操作的静态方法，传入值表达式和通道数
  static ExprHandle make(const ExprHandle& value, int lanes) {
    return ExprHandle(alloc<Broadcast>(value.node(), lanes));
  }

  // 广播操作的构造函数，初始化值表达式和通道数
  Broadcast(ExprPtr value, int lanes)
      : ExprNodeBase(Dtype(value->dtype(), lanes)),
        value_(std::move(value)),
        lanes_(lanes) {}

 private:
  ExprPtr value_;  // 广播操作的值表达式
  int lanes_;      // 广播操作的通道数
};

class TORCH_API IfThenElse : public ExprNode<IfThenElse> {
 public:
  // 返回条件表达式
  ExprPtr condition() const {
    return condition_;
  }

  // 返回条件为真时延迟计算的值表达式
  ExprPtr true_value() const {
    return true_;
  }

  // 返回条件为假时延迟计算的值表达式
  ExprPtr false_value() const {
    return false_;
  }

  // 设置条件表达式
  void set_condition(ExprPtr condition) {
    condition_ = std::move(condition);
  }

  // 设置条件为真时延迟计算的值表达式
  void set_true_value(ExprPtr true_value) {
    true_ = std::move(true_value);
  }

  // 设置条件为假时延迟计算的值表达式
  void set_false_value(ExprPtr false_value) {
    false_ = std::move(false_value);
  }

  // 创建条件语句的静态方法，传入条件表达式、真值表达式和假值表达式
  static ExprHandle make(
      const ExprHandle& c,
      const ExprHandle& t,
      const ExprHandle& f) {
    // 检查条件表达式的数据类型是否为整型
    if (!c.dtype().is_integral()) {
      throw unsupported_dtype();
    }
    // 检查条件表达式的通道数是否为1
    if (c.dtype().lanes() != 1) {
      throw unsupported_dtype();
    }
    // 检查真值表达式和假值表达式的数据类型是否一致
    if (t.dtype() != f.dtype()) {
      throw malformed_input("Bad dtype in IfThenElse");
    }
    // 创建并返回条件语句的表达式
    return ExprHandle(alloc<IfThenElse>(c.node(), t.node(), f.node()));
  }

  // 条件语句的构造函数，初始化条件表达式、真值表达式和假值表达式
  IfThenElse(ExprPtr c, ExprPtr t, ExprPtr f)
      : ExprNodeBase(t->dtype()),
        condition_(std::move(c)),
        true_(std::move(t)),
        false_(std::move(f)) {}

 private:
  ExprPtr condition_;  // 条件表达式
  ExprPtr true_;       // 真值表达式
  ExprPtr false_;      // 假值表达式
};

class TORCH_API CompareSelect : public ExprNode<CompareSelect> {
 public:
  // 返回比较选择操作
  CompareSelectOperation compare_select_op() const {
    return compare_op_;
  }

  // 返回左操作数表达式
  ExprPtr lhs() const {
    return this->lhs_;
  }

  // 返回右操作数表达式
  ExprPtr rhs() const {
    return this->rhs_;
  }

  // 返回条件成立时的返回值表达式
  ExprPtr ret_val1() const {
    return this->ret_val1_;
  }

  // 返回条件不成立时的返回值表达式
  ExprPtr ret_val2() const {
    return this->ret_val2_;
  }

  // 设置左操作数表达式
  void set_lhs(ExprPtr lhs) {
    lhs_ = std::move(lhs);
  }

  // 设置右操作数表达式
  void set_rhs(ExprPtr rhs) {
    rhs_ = std::move(rhs);
  }

  // 设置条件成立时的返回值表达式
  void set_ret_val1(ExprPtr ret_val1) {
    ret_val1_ = std::move(ret_val1);
  }

  // 设置条件不成立时的返回值表达式
  void set_ret_val2(ExprPtr ret_val2) {
    ret_val2_ = std::move(ret_val2);
  }

  // 返回比较选择的偏置
  CompareSelectBias bias() const {
    return bias_;
  }

  // 创建比较选择操作的静态方法，传入左操作数、右操作数、比较操作以及比较选择的偏置
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      CompareSelectOperation cmp_op,
      CompareSelectBias bias = kUnbiased) {
    // 检查左右操作数的数据类型是否一致
    if (lhs.dtype() != rhs.dtype()) {
      throw malformed_input("bad dtype in CompareSelect");
    }
    // 创建并返回比较选择操作的表达式
    return ExprHandle(alloc<CompareSelect>(lhs.node(), rhs.node(), cmp_op, bias));
  }

  // 比较选择操作的构造函数，初始化左操作数、右操作数、比较操作以及比较选择的偏置
  CompareSelect(ExprPtr lhs, ExprPtr rhs, CompareSelectOperation cmp_op, CompareSelectBias bias = kUnbiased)
      : ExprNodeBase(lhs->dtype()),
        lhs_(std::move(lhs)),
        rhs_(std::move(rhs)),
        compare_op_(cmp_op),
        bias_(bias) {}

 private:
  ExprPtr lhs_;         // 左操作数表达式
  ExprPtr rhs_;         // 右操作数表达式
  ExprPtr ret_val1_;    // 条件成立时的返回值表达式
  ExprPtr ret_val2_;    // 条件不成立时的返回值表达式
  CompareSelectOperation compare_op_; // 比较选择操作
  CompareSelectBias bias_;            // 比较选择的偏置
};
  // 返回一个表达式处理器对象，该对象用于处理 CompareSelect 节点
  return ExprHandle(alloc<CompareSelect>(
      lhs.node(),                      // 设置左操作数节点
      rhs.node(),                      // 设置右操作数节点
      IntImm::make(1).node(),          // 设置第一个返回值节点为整数常量1
      IntImm::make(0).node(),          // 设置第二个返回值节点为整数常量0
      cmp_op,                          // 比较操作类型
      bias));                          // 比较选择的偏置值
}

// 创建一个 CompareSelect 节点的静态方法
static ExprHandle make(
    const ExprHandle& lhs,             // 左操作数的表达式处理器对象
    const ExprHandle& rhs,             // 右操作数的表达式处理器对象
    const ExprHandle& ret_val1,        // 第一个返回值的表达式处理器对象
    const ExprHandle& ret_val2,        // 第二个返回值的表达式处理器对象
    CompareSelectOperation cmp_op,     // 比较操作类型
    CompareSelectBias bias = kUnbiased) {  // 比较选择的偏置值，默认为无偏置
  // 检查输入的数据类型是否匹配，若不匹配则抛出异常
  if (lhs.dtype() != rhs.dtype() || ret_val1.dtype() != ret_val2.dtype()) {
    throw malformed_input("bad dtype in CompareSelect");
  }
  // 返回一个表达式处理器对象，该对象用于处理 CompareSelect 节点
  return ExprHandle(alloc<CompareSelect>(
      lhs.node(),                      // 设置左操作数节点
      rhs.node(),                      // 设置右操作数节点
      ret_val1.node(),                 // 设置第一个返回值节点
      ret_val2.node(),                 // 设置第二个返回值节点
      cmp_op,                          // 比较操作类型
      bias));                          // 比较选择的偏置值
}

// CompareSelect 类的构造函数，初始化对象的各个成员变量
CompareSelect(
    ExprPtr lhs,                       // 左操作数的表达式指针
    ExprPtr rhs,                       // 右操作数的表达式指针
    ExprPtr ret_val1,                  // 第一个返回值的表达式指针
    ExprPtr ret_val2,                  // 第二个返回值的表达式指针
    CompareSelectOperation cmp_op,     // 比较操作类型
    CompareSelectBias bias = kUnbiased)  // 比较选择的偏置值，默认为无偏置
    : ExprNodeBase(ret_val1->dtype()), // 调用基类 ExprNodeBase 的构造函数，设置数据类型
      lhs_(std::move(lhs)),            // 初始化左操作数
      rhs_(std::move(rhs)),            // 初始化右操作数
      ret_val1_(std::move(ret_val1)),  // 初始化第一个返回值
      ret_val2_(std::move(ret_val2)),  // 初始化第二个返回值
      compare_op_(cmp_op),             // 初始化比较操作类型
      bias_(bias) {}                   // 初始化比较选择的偏置值

// 特殊情况的 CompareSelect 构造函数，设置默认返回值为整数常量
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
CompareSelect(
    ExprPtr lhs,                       // 左操作数的表达式指针
    ExprPtr rhs,                       // 右操作数的表达式指针
    CompareSelectOperation cmp_op,     // 比较操作类型
    CompareSelectBias bias = kUnbiased)  // 比较选择的偏置值，默认为无偏置
    : ExprNodeBase(kInt),              // 调用基类 ExprNodeBase 的构造函数，设置数据类型为整数
      lhs_(std::move(lhs)),            // 初始化左操作数
      rhs_(std::move(rhs)),            // 初始化右操作数
      ret_val1_(alloc<IntImm>(1)),     // 初始化第一个返回值为整数常量1
      ret_val2_(alloc<IntImm>(0)),     // 初始化第二个返回值为整数常量0
      compare_op_(cmp_op),             // 初始化比较操作类型
      bias_(bias) {}                   // 初始化比较选择的偏置值

private:
ExprPtr lhs_;                          // 左操作数的表达式指针成员变量
ExprPtr rhs_;                          // 右操作数的表达式指针成员变量
ExprPtr ret_val1_;                     // 第一个返回值的表达式指针成员变量
ExprPtr ret_val2_;                     // 第二个返回值的表达式指针成员变量
CompareSelectOperation compare_op_;    // 比较操作类型成员变量
CompareSelectBias bias_;               // 比较选择的偏置值成员变量
};

// 定义枚举类型 IntrinsicsOp，用于表示不同的内置操作
enum IntrinsicsOp {
  kSin,           // 正弦函数
  kCos,           // 余弦函数
  kTan,           // 正切函数
  kAsin,          // 反正弦函数
  kAcos,          // 反余弦函数
  kAtan,          // 反正切函数
  kAtan2,         // 两个参数的反正切函数
  kSinh,          // 双曲正弦函数
  kCosh,          // 双曲余弦函数
  kTanh,          // 双曲正切函数
  kSigmoid,       // Sigmoid 函数
  kExp,           // 指数函数
  kExpm1,         // exp(x) - 1
  kAbs,           // 绝对值函数
  kLog,           // 自然对数函数
  kLog2,          // 以2为底的对数函数
  kLog10,         // 以10为底的对数函数
  kLog1p,         // log(1 + x)
  kErf,           // 误差函数
  kErfc,          // 互补误差函数
  kSqrt,          // 平方根函数
  kRsqrt,         // 平方根的倒数函数
  kPow,           // 幂函数
  kCeil,          // 向上取整函数
  kFloor,         // 向下取整函数
  kRound,         // 四舍五入函数
  kTrunc,         // 截断函数
  kFmod,          // 浮点数取模函数
  kRemainder,     // 取余函数
  kLgamma,        // 对数伽玛函数
  kFrac,          // 返回小数部分函数
  kIsNan,         // 是否为 NaN 函数
  kRand,          // 随机数生成函数，需要进一步讨论是否具有状态性
  kMaxIntrinsicsOp, // 内置操作的最大值
};

// 定义 Intrinsics 类，表示表达式节点 Intrinsics 的操作
class TORCH_API Intrinsics : public ExprNode<Intrinsics> {
 public:
  // 创建一个带有一个参数的 Intrinsics 实例
  static ExprHandle make(IntrinsicsOp op_type, const ExprHandle& v1) {
    return ExprHandle(alloc<Intrinsics>(op_type, v1.node()));
  }

  // 创建一个带有两个参数的 Intrinsics 实例
  static ExprHandle make(
      IntrinsicsOp op_type,
      const ExprHandle& v1,
      const ExprHandle& v2) {
    return ExprHandle(alloc<Intrinsics>(op_type, v1.node(), v2.node()));
  }

  // 创建一个带有多个参数的 Intrinsics 实例
  static ExprHandle make(
      IntrinsicsOp op_type,
      const std::vector<ExprHandle>& params) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 将参数转换为对应的表达式节点
    std::vector<ExprPtr> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return ExprHandle(alloc<Intrinsics>(op_type, params_nodes));
  }

  // 创建一个带有数据类型参数的 Intrinsics 实例
  static ExprHandle make(IntrinsicsOp op_type, Dtype dtype) {
    return ExprHandle(alloc<Intrinsics>(op_type, dtype));
  }

  // 返回当前 Intrinsics 实例的操作类型
  IntrinsicsOp op_type() const {
    return op_type_;
  }

  // 返回当前 Intrinsics 实例的函数名
  std::string func_name() const {
    // 根据操作类型返回对应的函数名字符串
    switch (op_type()) {
      case kSin: return "sin";
      case kCos: return "cos";
      case kTan: return "tan";
      case kAsin: return "asin";
      case kAcos: return "acos";
      case kAtan: return "atan";
      case kAtan2: return "atan2";
      case kSinh: return "sinh";
      case kCosh: return "cosh";
      case kTanh: return "tanh";
      case kSigmoid: return "sigmoid";
      case kExp: return "exp";
      case kAbs: return "abs";
      case kLog: return "log";
      case kLog2: return "log2";
      case kLog10: return "log10";
      case kLog1p: return "log1p";
      case kErf: return "erf";
      case kSqrt: return "sqrt";
      case kRsqrt: return "rsqrt";
      case kPow: return "pow";
      case kCeil: return "ceil";
      case kFloor: return "floor";
      case kRound: return "round";
      case kTrunc: return "trunc";
      case kRand: return "rand"; // 这里需要进一步讨论，是否考虑具有状态性？
      case kFmod: return "fmod";
      case kRemainder: return "remainder";
      case kLgamma: return "lgamma";
      case kExpm1: return "expm1";
      case kErfc: return "erfc";
      case kFrac: return "frac";
      case kIsNan: return "isnan";
      default: throw std::runtime_error(
          "invalid op_type: " + std::to_string(op_type()));
    }
  }

 private:
  IntrinsicsOp op_type_; // 内置操作类型
};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 根据指定的操作类型和数据类型初始化 Intrinsics 对象
  Intrinsics(IntrinsicsOp op_type, Dtype dtype)
      : ExprNodeBase(IntrinsicsDtype(op_type, dtype)),
        params_({}),
        op_type_(op_type) {
    // 检查操作类型的参数数量是否为0，如果不是则抛出异常
    if (OpArgCount(op_type) != 0) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 根据指定的操作类型和表达式初始化 Intrinsics 对象
  Intrinsics(IntrinsicsOp op_type, ExprPtr v1)
      : ExprNodeBase(IntrinsicsDtype(op_type, v1->dtype())),
        params_({std::move(v1)}),
        op_type_(op_type) {
    // 检查操作类型的参数数量是否为1，如果不是则抛出异常
    if (OpArgCount(op_type) != 1) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 根据指定的操作类型和两个表达式初始化 Intrinsics 对象
  Intrinsics(IntrinsicsOp op_type, ExprPtr v1, ExprPtr v2)
      : ExprNodeBase(IntrinsicsDtype(op_type, v1->dtype(), v2->dtype())),
        params_({std::move(v1), std::move(v2)}),
        op_type_(op_type) {
    // 检查操作类型的参数数量是否为2，如果不是则抛出异常
    if (OpArgCount(op_type) != 2) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 根据指定的操作类型和表达式参数列表初始化 Intrinsics 对象
  Intrinsics(IntrinsicsOp op_type, const std::vector<ExprPtr>& params)
      : ExprNodeBase(IntrinsicsDtype(op_type, params)),
        params_(params),
        op_type_(op_type) {
    // 检查操作类型的参数数量是否与给定参数列表的数量相同，如果不是则抛出异常
    if (OpArgCount(op_type) != nparams()) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 根据指定的操作类型、数据类型和表达式参数列表初始化 Intrinsics 对象
  Intrinsics(
      IntrinsicsOp op_type,
      Dtype dtype,
      const std::vector<ExprPtr>& params)
      : ExprNodeBase(IntrinsicsDtype(op_type, dtype)),
        params_(params),
        op_type_(op_type) {
    // 检查操作类型的参数数量是否与给定参数列表的数量相同，如果不是则抛出异常
    if (OpArgCount(op_type) != nparams()) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // 判断该 Intrinsics 对象是否为纯函数（即不包含随机数操作）
  bool isPure() const {
    return op_type_ != kRand;
  }

  // 返回当前 Intrinsics 对象的参数数量
  int nparams() const {
    return params_.size();
  }

  // 返回指定索引位置的参数表达式
  ExprPtr param(int index) const {
    return params_[index];
  }

  // 返回当前 Intrinsics 对象的所有参数表达式列表的引用
  const std::vector<ExprPtr>& params() const {
    return params_;
  }

  // 设置当前 Intrinsics 对象的参数表达式列表
  void set_params(std::vector<ExprPtr> params) {
    params_ = std::move(params);
  }

  // 静态方法：根据操作类型返回该操作的参数数量
  static int OpArgCount(IntrinsicsOp op_type);

 private:
  // 静态方法：根据操作类型和一个数据类型返回表达式的数据类型
  static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1);
  
  // 静态方法：根据操作类型和两个数据类型返回表达式的数据类型
  static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1, Dtype dt2);
  
  // 静态方法：根据操作类型和参数表达式列表返回表达式的数据类型
  static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      const std::vector<ExprPtr>& params);

  // 成员变量：存储 Intrinsics 对象的参数表达式列表
  std::vector<ExprPtr> params_;

  // 成员变量：存储 Intrinsics 对象的操作类型
  IntrinsicsOp op_type_;
};

// 将 ExprHandle 的向量转换为 ExprPtr 的向量
TORCH_API std::vector<ExprPtr> ExprHandleVectorToExprVector(
    const std::vector<ExprHandle>&);

// 将 ExprPtr 的向量转换为 ExprHandle 的向量
TORCH_API std::vector<ExprHandle> ExprVectorToExprHandleVector(
    const std::vector<ExprPtr>&);

// 将 VarHandle 的向量转换为 VarPtr 的向量
TORCH_API std::vector<VarPtr> VarHandleVectorToVarVector(
    const std::vector<VarHandle>&);

// 将 VarPtr 的向量转换为 VarHandle 的向量
TORCH_API std::vector<VarHandle> VarVectorToVarHandleVector(
    const std::vector<VarPtr>&);

// 对索引进行展平化处理，根据给定的维度、索引和步长
TORCH_API ExprPtr flatten_index(
    const std::vector<ExprPtr>& dims,
    const std::vector<ExprPtr>& indices,
    const std::vector<ExprPtr>& strides);

// 结束 namespace torch::jit::tensorexpr
} // namespace tensorexpr
} // namespace jit
} // namespace torch
```