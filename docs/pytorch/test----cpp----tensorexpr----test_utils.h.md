# `.\pytorch\test\cpp\tensorexpr\test_utils.h`

```py
#pragma once

#include <memory>  // 包含智能指针和其他标准库头文件
#include <vector>  // 包含向量容器的标准库头文件

#include <test/cpp/tensorexpr/test_base.h>  // 引入测试基类头文件
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>  // 引入前向声明头文件
#include <torch/csrc/jit/testing/file_check.h>  // 引入文件检查头文件

namespace torch {  // 命名空间 torch
namespace jit {    // 命名空间 jit，包含 JIT 编译器相关内容
using namespace torch::jit::tensorexpr;  // 使用 torch::jit::tensorexpr 命名空间

// 定义宏 IS_NODE，用于验证节点类型 T，并断言不为空
#define IS_NODE(T, node)       \
  {                            \
    auto node_ = to<T>(node);  \
    ASSERT_NE(nullptr, node_); \
  }

// 定义宏 IS_NODE_WITH_NAME，用于验证节点类型 T，并断言不为空，并将其命名为 name
#define IS_NODE_WITH_NAME(T, node, name) \
  auto name = to<T>(node);               \
  ASSERT_NE(nullptr, name);

// 定义宏 IS_NODE_WITH_NAME_AND_CAST，用于验证 Cast 类型节点转换为 T 类型，并断言不为空，同时验证数据类型为 ScalarType::Type
#define IS_NODE_WITH_NAME_AND_CAST(T, node, name, Type)        \
  NodePtr<T> name = nullptr;                                   \
  {                                                            \
    auto node_ = to<Cast>(node);                               \
    ASSERT_NE(nullptr, node_);                                 \
    ASSERT_EQ(node_->dtype().scalar_type(), ScalarType::Type); \
    name = to<T>(node_->src_value());                          \
  }                                                            \
  ASSERT_NE(nullptr, name);

// 定义宏 IS_IMM_WITH_VAL，用于验证节点类型 T##Imm（如IntImm、FloatImm 等）的值为 val
#define IS_IMM_WITH_VAL(T, node, val) \
  {                                   \
    auto node_ = to<T##Imm>(node);    \
    ASSERT_NE(nullptr, node_);        \
    ASSERT_EQ(node_->value(), val);   \
  }

// 定义宏 IS_VAR_WITH_NAME，用于验证节点为 Var 类型，并断言其名称为 name
#define IS_VAR_WITH_NAME(node, name)     \
  {                                      \
    auto node_ = to<Var>(node);          \
    ASSERT_NE(nullptr, node_);           \
    ASSERT_EQ(node_->name_hint(), name); \
  }

// 定义宏 IS_BINOP_W_VARS，用于验证二元操作节点类型 T，断言不为空，并验证其左右子节点为变量 v1 和 v2
#define IS_BINOP_W_VARS(T, node, name, v1, v2) \
  NodePtr<T> name = nullptr;                   \
  {                                            \
    name = to<T>(node);                        \
    ASSERT_NE(nullptr, name);                  \
    IS_VAR_WITH_NAME(name->lhs(), v1);         \
    IS_VAR_WITH_NAME(name->rhs(), v2);         \
  }

// 定义宏 IS_BINOP_W_CONST，用于验证二元操作节点类型 T，断言不为空，并验证其左子节点为变量 v，右子节点为常量 c
#define IS_BINOP_W_CONST(T, node, name, v, c) \
  NodePtr<T> name = nullptr;                  \
  {                                           \
    name = to<T>(node);                       \
    ASSERT_NE(nullptr, name);                 \
    IS_VAR_WITH_NAME(name->lhs(), v);         \
    IS_IMM_WITH_VAL(Int, name->rhs(), c);     \
  }

// 定义宏 IS_RAND，用于验证节点为 Intrinsics 类型，并断言其操作类型为 kRand
#define IS_RAND(node)                   \
  {                                     \
    auto node_ = to<Intrinsics>(node);  \
    ASSERT_NE(nullptr, node_);          \
    ASSERT_EQ(node_->op_type(), kRand); \
  }

// 声明函数 checkIR，用于检查语句节点的 IR，并与指定模式进行匹配
void checkIR(StmtPtr s, const std::string& pattern);

// 声明函数 checkExprIR，用于检查表达式节点的 IR，并与指定模式进行匹配
void checkExprIR(ExprPtr e, const std::string& pattern);

// 声明函数 checkExprIR，重载版本，接受表达式句柄对象进行检查，并与指定模式进行匹配
void checkExprIR(const ExprHandle& e, const std::string& pattern);

} // namespace jit
} // namespace torch
```