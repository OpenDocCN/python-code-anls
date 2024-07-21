# `.\pytorch\torch\csrc\jit\tensorexpr\var_substitutor.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <unordered_map>
// 包含无序映射的标准库头文件
#include <utility>
// 包含实用工具的标准库头文件
#include <vector>
// 包含向量的标准库头文件

#include <torch/csrc/jit/tensorexpr/analysis.h>
// 包含 Torch TensorExpr 分析功能的头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
// 包含 Torch TensorExpr 中间表示的头文件
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
// 包含 Torch TensorExpr IR 变换器的头文件
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
// 包含 Torch TensorExpr IR 访问器的头文件
#include <torch/csrc/jit/tensorexpr/reduction.h>
// 包含 Torch TensorExpr 减少操作的头文件

namespace torch {
namespace jit {
namespace tensorexpr {

using VarMapping = std::vector<std::pair<VarPtr, ExprPtr>>;
// 使用向量存储的变量映射类型定义

class VarSubMutator : public IRMutator {
 public:
  // 构造函数，接受变量映射作为参数进行初始化
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VarSubMutator(const VarMapping& var_mapping) {
    // 遍历传入的变量映射，将键值对加入到成员变量 var_mapping_ 中
    for (auto& entry : var_mapping) {
      VarPtr key_var = entry.first;
      ExprPtr value = entry.second;
      if (!key_var) {
        // 如果键为空指针，则抛出异常
        throw malformed_input("missing key in VarSubMutator");
      }
      var_mapping_[std::move(key_var)] = std::move(value);
    }
  }

  // 重写 IRMutator 中的 mutate 方法，处理 VarPtr 类型的变量
  ExprPtr mutate(VarPtr var) override {
    // 查找当前变量在 var_mapping_ 中的映射
    auto iter = var_mapping_.find(var);
    if (iter == var_mapping_.end()) {
      // 如果找不到映射，则返回原始变量
      return var;
    }
    // 否则返回映射后的表达式
    return iter->second;
  }

  // 重写 IRMutator 中的 mutate 方法，处理 ReduceOpPtr 类型的变量
  ExprPtr mutate(ReduceOpPtr var) override {
    // 对 ReduceOp 中的 body 进行变异处理
    auto body = var->body()->accept_mutator(this);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<VarPtr> new_inner;

    // 遍历 ReduceOp 中的 reduce_args
    for (const auto& v : var->reduce_args()) {
      // 对每个参数进行变异处理
      ExprPtr e = v->accept_mutator(this);
      if (VarPtr new_var = to<Var>(e)) {
        // 如果变异后的表达式是 VarPtr 类型，则加入 new_inner
        new_inner.push_back(std::move(new_var));
      } else {
        // 否则进行变量查找
        VarFinder varFinder;
        e->accept(&varFinder);
        auto varlist = varFinder.vars();
        new_inner.insert(new_inner.end(), varlist.begin(), varlist.end());
      }
    }

    // 创建新的 ReduceOp 对象并返回
    return alloc<ReduceOp>(body, new_inner, var->reducer());
  }

 private:
  // 成员变量，存储变量映射关系
  std::unordered_map<VarPtr, ExprPtr> var_mapping_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```