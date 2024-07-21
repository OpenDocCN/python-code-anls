# `.\pytorch\torch\csrc\jit\tensorexpr\ir_verifier.cpp`

```py
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch::jit::tensorexpr {

namespace detail {
// 定义了一个模板函数 deducer，用于类型推断
template <typename T>
void deducer(BinaryOpNode<T>);

// 声明了一个 deducer 的重载版本，用于不匹配的情况
bool deducer(...);
} // namespace detail

// 模板函数 verifyBitwiseOp，用于验证位运算节点的操作
template <
    typename D,
    typename std::enable_if<std::is_same<
        decltype(detail::deducer(std::declval<D>())),
        void>::value>::type* = nullptr>
void verifyBitwiseOp(NodePtr<D> v, IRVerifier* verifier) {
  // 检查左操作数的数据类型是否为整数
  if (!v->lhs()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  // 检查左右操作数的数据类型是否一致
  if (v->lhs()->dtype() != v->rhs()->dtype()) {
    throw malformed_ir("lhs/rhs dtype mismatch");
  }
}

// IRVerifier 类的 visit 方法，用于访问 AndPtr 节点
void IRVerifier::visit(AndPtr v) {
  // 调用 verifyBitwiseOp 验证位运算
  verifyBitwiseOp(v, this);
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 OrPtr 节点
void IRVerifier::visit(OrPtr v) {
  // 调用 verifyBitwiseOp 验证位运算
  verifyBitwiseOp(v, this);
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 XorPtr 节点
void IRVerifier::visit(XorPtr v) {
  // 调用 verifyBitwiseOp 验证位运算
  verifyBitwiseOp(v, this);
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 LshiftPtr 节点
void IRVerifier::visit(LshiftPtr v) {
  // 调用 verifyBitwiseOp 验证位运算
  verifyBitwiseOp(v, this);
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 RshiftPtr 节点
void IRVerifier::visit(RshiftPtr v) {
  // 调用 verifyBitwiseOp 验证位运算
  verifyBitwiseOp(v, this);
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 ModPtr 节点
void IRVerifier::visit(ModPtr v) {
  // 检查节点数据类型是否为整数或浮点数
  if (!v->dtype().is_integral() && !v->dtype().is_floating_point()) {
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 CompareSelectPtr 节点
void IRVerifier::visit(CompareSelectPtr v) {
  // 检查返回值的数据类型是否一致
  if (v->ret_val1()->dtype() != v->ret_val2()->dtype()) {
    throw malformed_ir("bad dtype in CompareSelect");
  }
  // 检查左右操作数的数据类型是否一致
  if (v->lhs()->dtype() != v->rhs()->dtype()) {
    throw malformed_ir("bad dtype in CompareSelect");
  }
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 RampPtr 节点
void IRVerifier::visit(RampPtr v) {
  // 检查步长和基础值的数据类型是否一致
  if (v->stride()->dtype() != v->base()->dtype()) {
    throw malformed_ir("Bad stride in Ramp");
  }
  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 LoadPtr 节点
void IRVerifier::visit(LoadPtr v) {
  auto indices = v->indices();
  // 如果索引不为空，检查加载缓冲区的基础句柄数据类型是否为 Handle
  if (!indices.empty() && v->buf()->base_handle()->dtype() != kHandle) {
    throw malformed_ir(
        "Load base handle dtype must be Handle", v->buf()->base_handle());
  }

  // 获取第一个索引的数据类型，如果索引超过一个，检查所有索引的数据类型是否一致
  Dtype index_dtype = !indices.empty() ? indices.at(0)->dtype() : kInt;
  if (indices.size() > 1) {
    for (size_t i = 1; i < indices.size(); ++i) {
      if (indices.at(i)->dtype() != index_dtype) {
        throw malformed_ir("dtype mismatch in Load indices");
      }
    }
  }
  // 如果索引超过一个且数据类型的通道数大于 1，抛出异常
  if (indices.size() > 1 && index_dtype.lanes() > 1) {
    throw malformed_ir("Multilane is only allowed in a flattened index");
  }
  // 检查索引数据类型是否为 Int 或 Long
  if (index_dtype.scalar_type() != ScalarType::Int &&
      index_dtype.scalar_type() != ScalarType::Long) {
    throw malformed_ir("Index scalar dtype is not Int or Long!");
  }

  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}

// IRVerifier 类的 visit 方法，用于访问 IfThenElsePtr 节点
void IRVerifier::visit(IfThenElsePtr v) {
  // 检查条件表达式的数据类型是否为整数
  if (!v->condition()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  // 检查条件表达式的数据类型是否为单通道
  if (v->condition()->dtype().lanes() != 1) {
    // 如果不是单通道，抛出异常
    throw malformed_ir("Condition dtype lanes must be 1");
  }

  // 调用 IRVisitor 的 visit 方法继续访问节点
  IRVisitor::visit(v);
}
    throw unsupported_dtype();

# 抛出异常，表示不支持的数据类型错误


  }
  if (v->true_value()->dtype() != v->false_value()->dtype()) {
    throw malformed_ir("Bad dtype in IfThenElse");
  }

# 如果条件语句的真值分支和假值分支的数据类型不一致，抛出异常，说明条件语句的数据类型有问题


  IRVisitor::visit(v);

# 调用 IRVisitor 类的 visit 方法访问节点 v，这里用于处理条件语句节点的访问
}

// 访问内置函数表达式的处理函数
void IRVerifier::visit(IntrinsicsPtr v) {
  // 如果内置函数操作类型为 kIsNan
  if (v->op_type() == kIsNan) {
    // 检查参数的数据类型是否为整数类型
    if (v->dtype().scalar_type() != c10::kInt) {
      throw malformed_ir("bad dtype in intrinsic arg");
    }
    // 调用基类的访问函数处理该表达式
    IRVisitor::visit(v);
    return;
  }
  // 对于其他类型的内置函数，检查参数的数据类型是否与内置函数表达式本身的数据类型匹配
  for (auto const& param : v->params()) {
    if (param->dtype() != v->dtype()) {
      throw malformed_ir("bad dtype in intrinsic arg");
    }
  }
  // 调用基类的访问函数处理该表达式
  IRVisitor::visit(v);
}

// 访问存储操作的处理函数
void IRVerifier::visit(StorePtr v) {
  auto indices = v->indices();
  // 如果索引非空，检查缓存对象的基本句柄的数据类型是否为 kHandle
  if (!indices.empty() && v->buf()->base_handle()->dtype() != kHandle) {
    throw malformed_ir(
        "Store base handle dtype must be Handle", v->buf()->base_handle());
  }

  // 获取索引的数据类型，如果索引为空，则默认为 kInt
  Dtype index_dtype = !indices.empty() ? indices.at(0)->dtype() : kInt;
  // 如果有多个索引，确保它们的数据类型与第一个索引相匹配
  if (indices.size() > 1) {
    for (size_t i = 1; i < indices.size(); ++i) {
      if (indices.at(i)->dtype() != index_dtype) {
        throw malformed_ir("dtype mismatch in Store indices");
      }
    }
  }
  // 如果有多个索引且索引的数据类型有多个通道，抛出异常，因为多通道只允许在扁平化索引中使用
  if (indices.size() > 1 && index_dtype.lanes() > 1) {
    throw malformed_ir("Multilane is only allowed in a flattened index");
  }
  // 检查索引的标量数据类型是否为 Int 或 Long
  if (index_dtype.scalar_type() != ScalarType::Int &&
      index_dtype.scalar_type() != ScalarType::Long) {
    throw malformed_ir("Index scalar dtype is not Int or Long!");
  }
  // 检查缓存对象的数据类型与值的数据类型是否匹配
  if (v->buf()->dtype() != v->value()->dtype()) {
    throw malformed_ir("buf and value dtype mismatch in Store");
  }

  // 调用基类的访问函数处理该表达式
  IRVisitor::visit(v);
}

// 访问 For 循环的处理函数
void IRVerifier::visit(ForPtr v) {
  // 检查 For 循环中的变量、起始值、结束值和循环体是否为空
  if (!v->var()) {
    throw malformed_ir("nullptr Var in For loop");
  } else if (!v->start()) {
    throw malformed_ir("nullptr Start in For loop");
  } else if (!v->stop()) {
    throw malformed_ir("nullptr Stop in For loop");
  } else if (!v->body()) {
    throw malformed_ir("invalid Body in For loop");
  }
  // 调用基类的访问函数处理该表达式
  IRVisitor::visit(v);
}

// 访问块（Block）的处理函数
void IRVerifier::visit(BlockPtr v) {
  // 检查块中语句的父子关系是否正常
  for (const StmtPtr& s : v->stmts()) {
    if (s->get_parent() != v) {
      throw malformed_ir("Broken child-parent link inside a Block");
    }
  }
  // 调用基类的访问函数处理该表达式
  IRVisitor::visit(v);
}

// 访问外部调用（ExternalCall）的处理函数
void IRVerifier::visit(ExternalCallPtr v) {
  // 调用基类的访问函数处理该表达式
  IRVisitor::visit(v);
}

// 对语句进行验证的函数
void verify(StmtPtr s) {
  // 创建 IRVerifier 对象
  IRVerifier verifier;
  // 调用语句的 accept 函数，传入 IRVerifier 对象进行验证
  s->accept(&verifier);
}

// 对表达式进行验证的函数
void verify(ExprPtr e) {
  // 创建 IRVerifier 对象
  IRVerifier verifier;
  // 调用表达式的 accept 函数，传入 IRVerifier 对象进行验证
  e->accept(&verifier);
}

// 对表达式句柄进行验证的函数
void verify(ExprHandle e) {
  // 调用上层表达式节点的验证函数
  verify(e.node());
}

// 结束 torch::jit::tensorexpr 命名空间
} // namespace torch::jit::tensorexpr
```