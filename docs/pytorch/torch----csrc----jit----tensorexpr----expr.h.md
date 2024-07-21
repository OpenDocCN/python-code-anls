# `.\pytorch\torch\csrc\jit\tensorexpr\expr.h`

```py
/**
 * This file implements the core classes for Tensor Expressions.
 *
 * The structure of the expressions is inspired by Halide/TVM IR.
 */
#pragma once

#include <c10/core/MemoryFormat.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <utility>

namespace torch {
namespace jit {
namespace tensorexpr {

// Enumeration defining different types of IR nodes.
enum IRNodeType {
  kPrimitive,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kMax,
  kMin,
  kAnd,
  kOr,
  kLshift,
  kRshift,
  kXor,
  kCompareSelect,
  kCast,
  kBitCast,
  kOther,
};

// The common base between all expression node.
class TORCH_API Expr : public std::enable_shared_from_this<Expr> {
 public:
  // Constructor initializing the data type and expression type.
  explicit Expr(Dtype dtype, IRNodeType expr_type = kOther)
      : dtype_(dtype), expr_type_(expr_type) {}
  virtual ~Expr() = default;
  
  // Accessor for the data type of the expression.
  Dtype dtype() const {
    return dtype_;
  }

  // Abstract method for accepting an IRVisitor.
  virtual void accept(IRVisitor* visitor) = 0;

  // Abstract method for accepting an IRMutator.
  virtual ExprPtr accept_mutator(IRMutator* mutator) = 0;

  // Accessor for the expression type.
  IRNodeType expr_type() const {
    return expr_type_;
  }

  // Checks if the expression is a constant value.
  virtual bool isConstant() const {
    return false;
  }

  // Setter for the data type.
  void set_dtype(Dtype dtype) {
    dtype_ = dtype;
  }

  /*
   * Makes a deep copy of the given expression.
   *
   * All sub-expressions inside the given expressions are also cloned. Note
   * that the variables are not deep-copied since they are immutable.
   */
  static ExprPtr clone(ExprPtr s);

 protected:
  // Retrieves a shared pointer to this expression node.
  std::shared_ptr<Expr> getptr() {
    return shared_from_this();
  }

 private:
  Dtype dtype_;         // Data type of the expression.
  IRNodeType expr_type_;// Type of the expression node.
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op, class Base = Expr>
class ExprNode : public Base {
 public:
  using ExprNodeBase = ExprNode<Op>;
  
  // Overrides the accept method to visit this specific node type.
  void accept(IRVisitor* visitor) override {
    visitor->visit(static_to<Op>(Base::getptr()));
  }
  
  // Overrides the accept_mutator method to mutate this specific node type.
  ExprPtr accept_mutator(IRMutator* mutator) override;

  // Passes the constructor to the base class.
  using Base::Base;
};

// A wrapper object to the underlying ExprNode.
// Also serves as the primary way to build and operate on other expressions.
class TORCH_API ExprHandle {
 public:
  ExprHandle() = default;
  
  // Constructor initializing with an expression node.
  explicit ExprHandle(ExprPtr node) : base_expr_node_(std::move(node)) {}

  // Accessor to retrieve the expression node.
  ExprPtr node() {
    return base_expr_node_;
  }

  // Const accessor to retrieve the expression node.
  ExprPtr node() const {
    return base_expr_node_;
  }

  // Checks if the expression handle is empty.
  bool empty() const {
    return base_expr_node_ == nullptr;
  }

  // Macro to declare immediate expression types.
#define IMM_EXPR_DECLARE(Type, Name) ExprHandle(Type v);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_EXPR_DECLARE);
#undef IMM_EXPR_DECLARE

  // Converts the expression handle to a specific node type.
  template <class Op>
  NodePtr<Op> AsNode() {
    return to<Op>(this->node());
  }

  // Const version of converting the expression handle to a specific node type.
  template <class Op>
  NodePtr<Op> AsNode() const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return to<Op>(const_cast<ExprHandle*>(this)->node());
  }

private:
  ExprPtr base_expr_node_;  // Pointer to the base expression node.
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
    // 返回当前表达式处理对象转换为特定操作类型的节点指针
    return const_cast<ExprHandle*>(this)->AsNode<Op>();
    
    Dtype dtype() const {
      // 返回当前表达式处理对象关联的数据类型
      return node()->dtype();
    }
    
    // 处理数学运算符重载
    
    // 实现加法运算符重载
    ExprHandle operator+(const ExprHandle& other) const;
    // 实现减法运算符重载
    ExprHandle operator-(const ExprHandle& other) const;
    // 实现乘法运算符重载
    ExprHandle operator*(const ExprHandle& other) const;
    // 实现除法运算符重载
    ExprHandle operator/(const ExprHandle& other) const;
    // 实现取模运算符重载
    ExprHandle operator%(const ExprHandle& other) const;
    // 实现相等比较运算符重载
    ExprHandle operator==(const ExprHandle& other) const;
    // 实现不等比较运算符重载
    ExprHandle operator!=(const ExprHandle& other) const;
    // 实现大于比较运算符重载
    ExprHandle operator>(const ExprHandle& other) const;
    // 实现大于等于比较运算符重载
    ExprHandle operator>=(const ExprHandle& other) const;
    // 实现小于比较运算符重载
    ExprHandle operator<(const ExprHandle& other) const;
    // 实现小于等于比较运算符重载
    ExprHandle operator<=(const ExprHandle& other) const;
    // 实现按位与运算符重载
    ExprHandle operator&(const ExprHandle& other) const;
    // 实现按位或运算符重载
    ExprHandle operator|(const ExprHandle& other) const;
    // 实现逻辑与运算符重载
    ExprHandle operator&&(const ExprHandle& other) const;
    // 实现逻辑或运算符重载
    ExprHandle operator||(const ExprHandle& other) const;
    // 实现按位异或运算符重载
    ExprHandle operator^(const ExprHandle& other) const;
    // 实现左移位运算符重载
    ExprHandle operator<<(const ExprHandle& other) const;
    // 实现右移位运算符重载
    ExprHandle operator>>(const ExprHandle& other) const;
    
    private:
    // 表达式处理对象基础表达式节点的指针，初始为nullptr
    ExprPtr base_expr_node_ = nullptr;
};

// 表示 Var 对象的底层表示节点。
// 目前，每个 Var 对象代表一个唯一的变量，即使名称可能相同。
// 我们应该考虑添加一个 unique_name 属性。
class TORCH_API Var : public ExprNode<Var> {
 public:
  // 创建一个 Var 对象，并指定名称提示和数据类型
  static ExprHandle make(const std::string& name_hint, Dtype dtype) {
    return ExprHandle(alloc<Var>(name_hint, dtype));
  }
  
  // 创建一个 Var 对象，并指定数据类型，名称提示为空字符串
  static ExprHandle make(Dtype dtype) {
    return ExprHandle(alloc<Var>("", dtype));
  }

  // 返回名称提示的引用
  const std::string& name_hint() const {
    return name_hint_;
  }

  // 设置名称提示为指定的字符串
  void set_name_hint(const std::string& name) {
    name_hint_ = name;
  }

  // 设置名称提示为右值引用指定的字符串
  void set_name_hint(std::string&& name) {
    name_hint_ = std::move(name);
  }

  // Var 类的构造函数，初始化名称提示和数据类型
  Var(std::string name_hint, Dtype dtype)
      : ExprNodeBase(dtype, kPrimitive), name_hint_(std::move(name_hint)) {}

 private:
  std::string name_hint_;
};

// 创建连续步长的表达式数组
TORCH_API std::vector<ExprPtr> make_contiguous_strides(
    const std::vector<ExprHandle>& dims);
// 创建通道最后步长的表达式数组
TORCH_API std::vector<ExprPtr> make_channels_last_strides(
    const std::vector<ExprHandle>& dims);

// 表示 Buf 对象的底层表示节点。
class TORCH_API Buf : public ExprNode<Buf> {
 public:
  // 创建一个 Buf 对象，指定维度、数据类型
  static BufHandle make(const std::vector<ExprHandle>& dims, Dtype dtype);

  // 创建一个 Buf 对象，指定名称提示、维度、步长、数据类型
  static BufHandle make(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      const std::vector<ExprHandle>& strides,
      Dtype dtype);

  // 创建一个 Buf 对象，指定名称提示、维度、数据类型以及可选的初始化器、步长、量化参数
  static BufHandle make(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      Dtype dtype,
      std::optional<ExprHandle> initializer = c10::nullopt,
      std::optional<std::vector<ExprHandle>> strides = c10::nullopt,
      std::optional<ExprHandle> qscale = c10::nullopt,
      std::optional<ExprHandle> qzero = c10::nullopt);

  // 返回底层 Var 对象的指针
  VarPtr base_handle() const {
    return base_handle_;
  }
  
  // 设置底层 Var 对象的指针
  void set_base_handle(VarPtr base_handle) {
    base_handle_ = std::move(base_handle);
  }

  // 返回名称提示，从底层 Var 对象获取
  const std::string& name_hint() const {
    return base_handle_->name_hint();
  }
  
  // 设置名称提示，设置给底层 Var 对象
  void set_name_hint(const std::string& name_hint) {
    base_handle_->set_name_hint(name_hint);
  }

  // Buf 类的构造函数，使用名称提示、维度、数据类型、可选的初始化器、步长、量化参数初始化
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Buf(const std::string& name_hint,
      const std::vector<ExprPtr>& dims,
      Dtype dtype,
      ExprPtr initializer = nullptr,
      std::optional<std::vector<ExprPtr>> strides = c10::nullopt,
      ExprPtr qscale = nullptr,
      ExprPtr qzero = nullptr)
      : Buf(alloc<Var>(name_hint, kHandle),
            dims,
            dtype,
            std::move(initializer),
            std::move(strides),
            std::move(qscale),
            std::move(qzero)) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Buf 类的构造函数，使用底层 Var 对象指针、维度、数据类型、可选的初始化器、步长、量化参数初始化
  Buf(VarPtr var,
      std::vector<ExprPtr> dims,
      Dtype dtype,
      ExprPtr initializer = nullptr,
      std::optional<std::vector<ExprPtr>> strides = c10::nullopt,
      ExprPtr qscale = nullptr,
      ExprPtr qzero = nullptr);

  // 返回 Buf 对象的维度数量
  size_t ndim() const {
  // 返回 dims_ 容器的大小
  return dims_.size();
}

// 返回指定索引处的维度表达式，如果索引超出范围则抛出异常
ExprPtr dim(size_t index) const {
  if (index >= ndim()) {
    throw out_of_range_index();
  }
  return dims_[index];
}

// 返回 dims_ 容器的副本
std::vector<ExprPtr> dims() const {
  return dims_;
}

// 设置 dims_ 容器的内容
void set_dims(std::vector<ExprPtr> dims) {
  dims_ = std::move(dims);
}

// 返回 strides_ 容器的副本
std::vector<ExprPtr> strides() const {
  return strides_;
}

// 设置 strides_ 容器的内容
void set_strides(std::vector<ExprPtr> strides) {
  strides_ = std::move(strides);
}

// 返回 initializer_ 成员变量的值
ExprPtr initializer() const {
  return initializer_;
};

// 返回 qzero_ 成员变量的值
ExprPtr qzero() const {
  return qzero_;
}

// 返回 qscale_ 成员变量的值
ExprPtr qscale() const {
  return qscale_;
}

// 设置 qzero_ 成员变量的值
void set_qzero(ExprPtr qzero) {
  qzero_ = std::move(qzero);
}

// 设置 qscale_ 成员变量的值
void set_qscale(ExprPtr qscale) {
  qscale_ = std::move(qscale);
}

// 检查 dims_ 中所有维度是否都是常量表达式
bool hasConstantDims() const {
  for (const auto& d : dims_) {
    if (!d->isConstant()) {
      return false;
    }
  }
  return true;
}

// 声明 is_contiguous 函数的存在，具体实现不在此处
bool is_contiguous(
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const;

// 检查是否满足 channels-last 1d 的连续性布局
// Channels-last 1d:
//   dims:              n   c    l
//   strides(nlc):    c*l   1    c
bool is_channels_last_1d_contiguous() const {
  if (dims_.size() != 3) {
    return false;
  }
  return is_stride_one(1) && is_cont_with(2, 1) && is_cont_with(0, 2);
}

private:
// 声明 is_cont_with 和 is_stride_one 函数的存在，具体实现不在此处
bool is_cont_with(int cur_dim, int adjacent_dim) const;
bool is_stride_one(int cur_dim) const;

VarPtr base_handle_;
std::vector<ExprPtr> dims_;     // 维度表达式的容器
std::vector<ExprPtr> strides_;  // 步长表达式的容器
ExprPtr initializer_;           // 初始化器表达式
ExprPtr qscale_;                // 量化标度表达式，用于特定的量化数据类型
ExprPtr qzero_;                 // 量化零点表达式，用于特定的量化数据类型
};

// BufHandle 类的定义，继承自 ExprHandle 类
class TORCH_API BufHandle : public ExprHandle {
 public:
  // 构造函数，接受缓冲区的名字提示、维度列表和数据类型
  BufHandle(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      Dtype dtype)
      : ExprHandle(Buf::make(name_hint, dims, dtype)) {}

  // 构造函数，接受缓冲区的名字提示、维度列表、步长列表和数据类型
  BufHandle(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      const std::vector<ExprHandle>& strides,
      Dtype dtype)
      : ExprHandle(Buf::make(name_hint, dims, strides, dtype)) {}

  // 构造函数，接受维度列表和数据类型，使用默认名字提示 "_"
  BufHandle(const std::vector<ExprHandle>& dims, Dtype dtype)
      : ExprHandle(Buf::make("_", dims, dtype)) {}

  // 构造函数，接受数据类型，使用默认名字提示 "_" 和空的维度列表
  explicit BufHandle(Dtype dtype) : ExprHandle(Buf::make("_", {}, dtype)) {}

  // 构造函数，接受已有的缓冲区指针
  explicit BufHandle(BufPtr node) : ExprHandle(std::move(node)) {}

  // 返回当前缓冲区的指针
  BufPtr node() const {
    return static_to<Buf>(ExprHandle::node());
  }

  // 返回当前缓冲区的指针（非常量版本）
  BufPtr node() {
    return static_to<Buf>(ExprHandle::node());
  }

  // 载入操作的模板函数，用于载入缓冲区中的数据
  template <typename... Ts>
  inline ExprHandle load(const Ts&... ts) const;

  // 载入操作的模板函数，用于载入缓冲区中的数据（使用参数向量）
  template <typename T>
  inline ExprHandle load(const std::vector<T>& args) const;

  // 载入操作的函数重载，用于载入缓冲区中的数据（使用 ExprHandle 向量）
  inline ExprHandle load(const std::vector<ExprHandle>& args) const;

  // 存储操作，用于将表达式值存储到缓冲区指定位置
  StorePtr store(const std::vector<ExprHandle>& args, const ExprHandle& val)
      const;

  // 比较操作符重载，用于比较两个缓冲区是否相等
  bool operator==(const BufHandle& other) const {
    return this->node() == other.node();
  }

  // 比较操作符重载，用于比较两个缓冲区是否不相等
  bool operator!=(const BufHandle& other) const {
    return !(*this == other);
  }

  // 返回缓冲区的名字提示
  const std::string& name_hint() const {
    return this->node()->name_hint();
  }

  // 检查缓冲区是否为空
  bool empty() const {
    return (this->node() == nullptr);
  }

  // 返回缓冲区的维度数
  size_t ndim() const {
    return node()->ndim();
  }

  // 返回缓冲区的维度列表
  std::vector<ExprHandle> dims() const;

  // 返回指定索引位置的维度
  ExprHandle dim(size_t index) const {
    return ExprHandle(node()->dim(index));
  }

  // 检查缓冲区是否按照指定的内存格式连续存储
  bool is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    return node()->is_contiguous(memory_format);
  }

  // 检查缓冲区是否以 channels last 的方式在一维上连续存储
  bool is_channels_last_1d_contiguous() const {
    return node()->is_channels_last_1d_contiguous();
  }
};

// VarHandle 类的定义，继承自 ExprHandle 类
// 表达式，用于构造基础变量节点
// 注意：不要在此处存储任何信息，因为通常可以对此对象进行切片操作。例如：VarHandle x('x'); ExprHandle x2 = x;
class TORCH_API VarHandle : public ExprHandle {
 public:
  // 创建一个空的 VarHandle，其基础变量设置为 nullptr
  VarHandle() : ExprHandle() {}

  // 创建一个指定数据类型的 VarHandle
  explicit VarHandle(Dtype dtype) : ExprHandle(Var::make(dtype)) {}

  // 创建一个指定名字提示和数据类型的 VarHandle
  VarHandle(const std::string& name_hint, Dtype dtype)
      : ExprHandle(Var::make(name_hint, dtype)) {}

  // 创建一个指定缓冲区指针的 VarHandle
  explicit VarHandle(VarPtr node) : ExprHandle(std::move(node)) {}

  // 返回当前变量节点的指针
  VarPtr node() const {
    return static_to<Var>(ExprHandle::node());
  }

  // 比较操作符重载，用于比较两个 VarHandle 是否相等
  bool operator==(const VarHandle& other) const {
    return this->node() == other.node();
  }

  // 比较操作符重载，用于比较两个 VarHandle 是否不相等
  bool operator!=(const VarHandle& other) const {
    return !(*this == other);
  }

  // 返回变量名的提示信息
  const std::string& name_hint() const {
    return this->node()->name_hint();
  }

  // 检查变量节点是否为空
  bool empty() const {
    return (this->node() == nullptr);
  }
};

// Op 类模板和 Base 类模板的定义
// 定义 ExprNode 模板类的 accept_mutator 方法，用于接受 IRMutator 类的变异器
ExprPtr ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) {
  // 调用 mutator 的 mutate 方法，对 Base 指向的 Op 类型进行变异，并返回结果
  return mutator->mutate(static_to<Op>(Base::getptr()));
}

// 比较两个 ExprHandle 对象是否指向同一个节点
inline bool same_node(const ExprHandle& expr1, const ExprHandle& expr2) {
  // 使用 AsNode<Expr>() 方法获取 ExprHandle 内部的 Expr 节点，并比较其地址是否相同
  return expr1.AsNode<Expr>() == expr2.AsNode<Expr>();
}

// 下列函数声明为 Torch API 提供的数学表达式函数，各函数功能类似，返回一个 ExprHandle 对象

TORCH_API ExprHandle sin(const ExprHandle& v);
TORCH_API ExprHandle cos(const ExprHandle& v);
TORCH_API ExprHandle tan(const ExprHandle& v);
TORCH_API ExprHandle asin(const ExprHandle& v);
TORCH_API ExprHandle acos(const ExprHandle& v);
TORCH_API ExprHandle atan(const ExprHandle& v);
TORCH_API ExprHandle sinh(const ExprHandle& v);
TORCH_API ExprHandle cosh(const ExprHandle& v);
TORCH_API ExprHandle tanh(const ExprHandle& v);
TORCH_API ExprHandle sigmoid(const ExprHandle& v);
TORCH_API ExprHandle exp(const ExprHandle& v);
TORCH_API ExprHandle expm1(const ExprHandle& v);
TORCH_API ExprHandle abs(const ExprHandle& v);
TORCH_API ExprHandle log(const ExprHandle& v);
TORCH_API ExprHandle fast_tanh(const ExprHandle& v);
TORCH_API ExprHandle fast_sigmoid(const ExprHandle& v);
TORCH_API ExprHandle fast_log(const ExprHandle& v);
TORCH_API ExprHandle log_vml(const ExprHandle& v);
TORCH_API ExprHandle log2(const ExprHandle& v);
TORCH_API ExprHandle log10(const ExprHandle& v);
TORCH_API ExprHandle log1p(const ExprHandle& v);
TORCH_API ExprHandle erf(const ExprHandle& v);
TORCH_API ExprHandle erfc(const ExprHandle& v);
TORCH_API ExprHandle sqrt(const ExprHandle& v);
TORCH_API ExprHandle rsqrt(const ExprHandle& v);
TORCH_API ExprHandle ceil(const ExprHandle& v);
TORCH_API ExprHandle floor(const ExprHandle& v);
TORCH_API ExprHandle round(const ExprHandle& v);
TORCH_API ExprHandle trunc(const ExprHandle& v);
TORCH_API ExprHandle frac(const ExprHandle& v);
TORCH_API ExprHandle lgamma(const ExprHandle& v);
TORCH_API ExprHandle atan2(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle pow(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle fmod(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle remainder(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle isnan(const ExprHandle& v1);
TORCH_API ExprHandle Relu(const ExprHandle& v1);

// 定义了一个 ifThenElse 函数，实现条件表达式的三元运算符
TORCH_API ExprHandle ifThenElse(const ExprHandle& c, const ExprHandle& t, const ExprHandle& f);

// 将一个表达式转换为具有指定长度的向量表达式
TORCH_API ExprHandle expr_to_vec(ExprHandle v, int lanes);

// 声明 tensorexpr 命名空间下的内容
namespace tensorexpr {
// 声明 jit 命名空间下的内容
namespace jit {
// 声明 torch 命名空间下的内容
namespace torch {

} // namespace torch
} // namespace jit
} // namespace tensorexpr
```