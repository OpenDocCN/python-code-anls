# `.\pytorch\torch\csrc\jit\tensorexpr\half_support.h`

```py
#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Walk the Statement looking for Half size loads/stores.
// 用于遍历语句以查找半精度加载/存储操作。
class HalfChecker : public IRVisitor {
 public:
  // Constructor to initialize HalfChecker with a list of BufferArgs
  // 初始化 HalfChecker，使用一组 BufferArg 进行初始化。
  HalfChecker(const std::vector<CodeGen::BufferArg>& args) {
    // Check if any of the BufferArgs have a dtype of Half
    // 检查是否有任何 BufferArg 的数据类型为 Half。
    for (const auto& BA : args) {
      hasHalf_ |= BA.dtype().scalar_type() == ScalarType::Half;
    }
  }

  // Check if Half type operations were found during traversal
  // 返回是否存在半精度操作。
  bool hasHalf() const {
    return hasHalf_;
  }

  // Check if BFloat16 type operations were found during traversal
  // 返回是否存在 BFloat16 操作。
  bool hasBFloat16() const {
    return hasBFloat16_;
  }

  // Override visit method for LoadPtr nodes
  // 重写 LoadPtr 节点的访问方法。
  void visit(LoadPtr v) override {
    // Check if the dtype of LoadPtr node is Half or BFloat16
    // 检查 LoadPtr 节点的数据类型是否为 Half 或 BFloat16。
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

  // Override visit method for StorePtr nodes
  // 重写 StorePtr 节点的访问方法。
  void visit(StorePtr v) override {
    // Check if the dtype of the buffer in StorePtr node is Half or BFloat16
    // 检查 StorePtr 节点中缓冲区的数据类型是否为 Half 或 BFloat16。
    hasHalf_ |= v->buf()->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->buf()->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

  // Override visit method for HalfImmPtr nodes
  // 重写 HalfImmPtr 节点的访问方法。
  void visit(HalfImmPtr v) override {
    // Set hasHalf_ to true when encountering HalfImmPtr node
    // 在遇到 HalfImmPtr 节点时将 hasHalf_ 设置为 true。
    hasHalf_ = true;
  }

  // Override visit method for BFloat16ImmPtr nodes
  // 重写 BFloat16ImmPtr 节点的访问方法。
  void visit(BFloat16ImmPtr v) override {
    // Set hasBFloat16_ to true when encountering BFloat16ImmPtr node
    // 在遇到 BFloat16ImmPtr 节点时将 hasBFloat16_ 设置为 true。
    hasBFloat16_ = true;
  }

  // Override visit method for CastPtr nodes
  // 重写 CastPtr 节点的访问方法。
  void visit(CastPtr v) override {
    // Check if the dtype of CastPtr node is Half or BFloat16
    // 检查 CastPtr 节点的数据类型是否为 Half 或 BFloat16。
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

 private:
  bool hasHalf_{false};     // Flag indicating presence of Half type operations
  bool hasBFloat16_{false}; // Flag indicating presence of BFloat16 type operations
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 用于将半精度数据转换为单精度数据。
class HalfRewriter : public IRMutator {
  ExprPtr mutate(LoadPtr v) override {
    ExprPtr child = IRMutator::mutate(v);
    // If the child expression is not of Half type, return it unchanged
    // 如果子表达式不是半精度类型，则直接返回。
    if (!isHalf(child)) {
      return child;
    }

    // Allocate a Cast node to convert Half to Float type
    // 分配一个 Cast 节点，将半精度转换为单精度类型。
    ExprPtr ret = alloc<Cast>(
        child->dtype().cloneWithScalarType(ScalarType::Float), child);

    // Track inserted Half casts
    // 记录插入的半精度转换操作。
    inserted_half_casts_.insert(ret);
    return ret;
  }

  StmtPtr mutate(StorePtr v) override {
    // Mutation of StorePtr's value expression in-place requires fetching its dtype first
    // 因为在原地突变 StorePtr 的 value 表达式，需要先获取其数据类型。
    auto newType = v->value()->dtype();
    ExprPtr new_val = v->value()->accept_mutator(this);
    auto bufType = v->buf()->dtype();

    // If the new value type is Half, insert a Cast node to convert it to Float type
    // 如果新值类型是半精度，则插入一个 Cast 节点将其转换为单精度类型。
    if (isHalf(newType.scalar_type())) {
      new_val = alloc<Cast>(newType, new_val);
      inserted_half_casts_.insert(new_val);
    }

    // If the value type is not Half but the buffer type is Half, convert to buffer's type
    // 如果值的类型不是半精度但缓冲区的类型是半精度，则转换为缓冲区的类型。
    if (!isHalf(newType.scalar_type()) && isHalf(bufType.scalar_type())) {
      new_val = alloc<Cast>(
          newType.cloneWithScalarType(bufType.scalar_type()), new_val);
      inserted_half_casts_.insert(new_val);
    }

    // Update the value expression in StorePtr node
    // 更新 StorePtr 节点中的 value 表达式。
    v->set_value(new_val);
    return v;
  }

  // Mutation method for HalfImmPtr nodes, converting HalfImmPtr to Float type
  // 将 HalfImmPtr 节点转换为单精度类型。
  ExprPtr mutate(HalfImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  // Mutation method for BFloat16ImmPtr nodes, converting BFloat16ImmPtr to Float type
  // 将 BFloat16ImmPtr 节点转换为单精度类型。
  ExprPtr mutate(BFloat16ImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  // Mutation method for CastPtr nodes, handling further mutation of its source value
  // 处理 CastPtr 节点的进一步变异，包括对其源值的变异。
  ExprPtr mutate(CastPtr v) override {
    ExprPtr child = v->src_value()->accept_mutator(this);

    // Ensure no unintended Half casts are allowed
    // 确保不允许未插入的半精度转换。
    // （此处的注释应当包括对应的具体实现）
    // 如果表达式类型为半精度（half）或者 BF16
    if (isHalf(v)) {
      // 检查已插入的半精度转换是否已存在
      if (inserted_half_casts_.count(v) < 1) {
        // 将子表达式设为该表达式的源值
        v->set_src_value(child);
        // 将表达式的数据类型设置为浮点类型
        v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
        // 返回更新后的表达式
        return v;
      }
    }

    // 移除类似 Half(Float()) 的转换
    CastPtr cast_child = to<Cast>(child);
    if (cast_child) {
      // 检查是否是要转换为 double 的 cast
      auto cast_to_double = v->dtype().scalar_type() == ScalarType::Double;
      // 检查子表达式是否是 half 类型
      auto from_half = isHalf(cast_child->src_value());
      // 不能简化 double(float(half)) 到 double(half)，因为 NNC 不直接支持 BF16 到 double 的转换
      auto not_cast_half_to_doulbe = !(cast_to_double && from_half);
      // 如果都是浮点数类型，并且不是从 half 转换为 double，则返回一个新的 Cast 表达式
      if (v->dtype().is_floating_point() &&
          cast_child->dtype().is_floating_point() && not_cast_half_to_doulbe) {
        return alloc<Cast>(v->dtype(), cast_child->src_value());
      }
    }

    // 如果子表达式等于该表达式的源值，则返回该表达式
    if (child == v->src_value()) {
      return v;
    }

    // 返回一个新的 Cast 表达式，使用该表达式的数据类型和子表达式
    return alloc<Cast>(v->dtype(), child);
  }

  // 重写 mutate 方法处理 Let 类型的表达式
  StmtPtr mutate(LetPtr v) override {
    // 如果变量的标量类型是半精度（half）
    if (isHalf(v->var()->dtype().scalar_type())) {
      // 创建一个新的变量，标量类型为 float
      VarPtr load_new_var = alloc<Var>(v->var()->name_hint(), kFloat);
      // 对变量的值进行转换为 float 类型
      ExprPtr new_value = alloc<Cast>(
          v->var()->dtype().cloneWithScalarType(ScalarType::Float),
          v->value()->accept_mutator(this));
      // 将新的变量映射到原始变量
      var_map[v->var()] = load_new_var;

      // 返回一个新的 Let 表达式，将新的变量和转换后的值绑定
      return alloc<Let>(load_new_var, new_value);
    }

    // 对于其他类型的表达式，调用基类的 mutate 方法处理
    return IRMutator::mutate(v);
  }

  // 重写 mutate 方法处理 Var 类型的表达式
  ExprPtr mutate(VarPtr v) override {
    // 查找变量是否在映射表中
    auto it = var_map.find(v);
    // 如果找到映射，则返回映射后的变量
    if (it != var_map.end()) {
      return it->second;
    }

    // 否则返回原始的变量
    return v;
  }

  // 模板方法，处理算术表达式的变换
  template <typename T>
  ExprPtr mutateArithmetic(T v) {
    // 调用基类的 mutate 方法
    IRMutator::mutate(v);
    // 如果表达式类型为半精度（half），则更新表达式的数据类型为 float
    if (isHalf(v)) {
      v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
    }
    // 返回变换后的表达式
    return v;
  }

  // 重写 mutate 方法处理 AddPtr 类型的表达式
  ExprPtr mutate(AddPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 SubPtr 类型的表达式
  ExprPtr mutate(SubPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 MulPtr 类型的表达式
  ExprPtr mutate(MulPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 DivPtr 类型的表达式
  ExprPtr mutate(DivPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 MaxPtr 类型的表达式
  ExprPtr mutate(MaxPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 MinPtr 类型的表达式
  ExprPtr mutate(MinPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 CompareSelectPtr 类型的表达式
  ExprPtr mutate(CompareSelectPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 BroadcastPtr 类型的表达式
  ExprPtr mutate(BroadcastPtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 IfThenElsePtr 类型的表达式
  ExprPtr mutate(IfThenElsePtr v) override {
    return mutateArithmetic(v);
  }
  // 重写 mutate 方法处理 IntrinsicsPtr 类型的表达式
  ExprPtr mutate(IntrinsicsPtr v) override {
    return mutateArithmetic(v);
  }

 private:
  // 静态方法，判断标量类型是否为半精度（half）或 BF16
  static bool isHalf(ScalarType st) {
    return st == ScalarType::Half || st == ScalarType::BFloat16;
  }

  // 静态方法，判断表达式是否为半精度（half）
  static bool isHalf(ExprPtr v) {
    return isHalf(v->dtype().scalar_type());
  }

  // 已插入的半精度转换表达式集合
  std::unordered_set<ExprPtr> inserted_half_casts_;
  // 变量映射表，用于存储原始变量到新变量的映射关系
  std::unordered_map<VarPtr, VarPtr> var_map;
};

// 结束命名空间 "tensorexpr"

} // namespace tensorexpr

// 结束命名空间 "jit"

} // namespace jit

// 结束命名空间 "torch"

} // namespace torch
```