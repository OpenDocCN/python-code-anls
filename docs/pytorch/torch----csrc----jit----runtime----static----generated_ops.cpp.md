# `.\pytorch\torch\csrc\jit\runtime\static\generated_ops.cpp`

```
// 忽略 clang-tidy 的检查，此处自动生成于 torchgen/static_runtime/gen_static_runtime_ops.py

#include <torch/csrc/jit/runtime/static/ops.h> // 导入静态运行时操作头文件

#include <ATen/CPUFunctions.h> // CPU 函数
#include <ATen/InferSize.h> // 推断大小
#include <ATen/NativeFunctions.h> // 原生函数
#include <ATen/Parallel.h> // 并行操作
#include <ATen/ScalarOps.h> // 标量操作
#include <ATen/TensorUtils.h> // 张量工具
#include <ATen/cpu/vec/functional.h> // CPU 向量功能
#include <ATen/cpu/vec/vec.h> // CPU 向量
#include <ATen/native/EmbeddingBag.h> // 嵌入包
#include <ATen/native/Fill.h> // 填充
#include <ATen/native/IndexingUtils.h> // 索引工具
#include <ATen/native/NonSymbolicBC.h> // 非符号 BC
#include <ATen/native/Resize.h> // 调整大小
#include <ATen/native/SharedReduceOps.h> // 共享减少操作
#include <ATen/native/TensorAdvancedIndexing.h> // 张量高级索引
#include <ATen/native/cpu/SerialStackImpl.h> // CPU 串行堆栈实现
#include <ATen/native/layer_norm.h> // 层归一化
#include <ATen/native/quantized/cpu/fbgemm_utils.h> // 量化 CPU 工具
#include <ATen/native/quantized/cpu/qembeddingbag.h> // 量化 CPU 嵌入包
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h> // 量化 CPU 预打包嵌入包
#include <ATen/quantized/QTensorImpl.h> // 量化张量实现
#include <ATen/quantized/Quantizer.h> // 量化器
#include <c10/core/ScalarType.h> // 标量类型
#include <c10/core/WrapDimMinimal.h> // 包装维度最小化
#include <c10/util/irange.h> // 迭代范围
#include <torch/csrc/jit/ir/ir.h> // IR
#include <torch/csrc/jit/runtime/static/impl.h> // 静态运行时实现
#include <torch/csrc/jit/runtime/static/te_wrapper.h> // 张量表达式包装器
#include <torch/csrc/jit/runtime/vararg_functions.h> // 可变参数函数
#include <torch/csrc/jit/tensorexpr/ir.h> // 张量表达式 IR
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h> // 张量表达式 IR 简化器
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h> // 张量表达式 LLVM 代码生成
#include <torch/csrc/jit/tensorexpr/loopnest.h> // 循环嵌套

namespace torch {
namespace jit {

// 注册 aten::absolute 操作符的操作函数
REGISTER_OPERATOR_FUNCTOR(
    aten::absolute,
    aten_absolute,
    [](Node* n) -> SROperator {
      // 检查节点是否符合给定的 schema
      if (n->matches(torch::schema("aten::absolute(Tensor self) -> Tensor"))) {
        // 返回一个 lambda 表达式，处理给定的节点
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出张量为空
          if (p_node->Output(0).isNone()) {
            // 计算并设置输出张量为 self 的绝对值
            p_node->Output(0) = at::native::absolute(self);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 计算 self 的绝对值并存入 out
          at::native::absolute_out(self, out);
        };
      }
      // 如果不匹配 schema，则记录并转储 schema
      LogAndDumpSchema(n);
      return nullptr; // 返回空指针
    });

// 注册 aten::angle 操作符的操作函数
REGISTER_OPERATOR_FUNCTOR(aten::angle, aten_angle, [](Node* n) -> SROperator {
  // 检查节点是否符合给定的 schema
  if (n->matches(torch::schema("aten::angle(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理给定的节点
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出张量为空
      if (p_node->Output(0).isNone()) {
        // 计算并设置输出张量为 self 的角度
        p_node->Output(0) = at::native::angle(self);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 计算 self 的角度并存入 out
      at::native::angle_out(self, out);
    };
  }
  // 如果不匹配 schema，则记录并转储 schema
  LogAndDumpSchema(n);
  return nullptr; // 返回空指针
});

// 注册 aten::sgn 操作符的操作函数
REGISTER_OPERATOR_FUNCTOR(aten::sgn, aten_sgn, [](Node* n) -> SROperator {
  // 检查节点是否符合给定的 schema
  if (n->matches(torch::schema("aten::sgn(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 定义 lambda 表达式，参数为 p_node，返回类型为 void
      const auto& self = p_node->Input(0).toTensor();
      // 获取 p_node 的第一个输入，转换为 Tensor 类型，并赋值给 self

      // 检查 p_node 的第一个输出是否为 None
      if (p_node->Output(0).isNone()) {
        // 如果输出为 None，则计算 at::cpu::sgn(self)，并赋给 p_node 的第一个输出
        p_node->Output(0) = at::cpu::sgn(self);
        return;
        // 返回，结束 lambda 表达式的执行
      }

      // 如果 p_node 的第一个输出不为 None，则获取该输出，并转换为 Tensor 类型，赋值给 out
      auto& out = p_node->Output(0).toTensor();

      // 调用 fastResizeToZero 函数，将 out 的大小调整为零
      fastResizeToZero(out);

      // 使用 at::cpu::sgn_out 将 self 中的数据存储到 out 中
      at::cpu::sgn_out(out, self);
    };
  }
  // 调用 LogAndDumpSchema 函数，记录和转储 n 的信息
  LogAndDumpSchema(n);
  // 返回空指针 nullptr
  return nullptr;
REGISTER_OPERATOR_FUNCTOR(aten::acos, aten_acos, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 aten::acos 模式
  if (n->matches(torch::schema("aten::acos(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个张量作为 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算 acos(self)，并存储到输出节点
        p_node->Output(0) = at::cpu::acos(self);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在指定的输出张量上计算 acos(self)
      at::cpu::acos_out(out, self);
    };
  }
  // 如果节点不匹配模式，则记录并转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::arccos, aten_arccos, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 aten::arccos 模式
  if (n->matches(torch::schema("aten::arccos(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个张量作为 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算 arccos(self)，并存储到输出节点
        p_node->Output(0) = at::native::arccos(self);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在指定的输出张量上计算 arccos(self)
      at::native::arccos_out(self, out);
    };
  }
  // 如果节点不匹配模式，则记录并转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::_add_relu, aten__add_relu, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 aten::_add_relu 模式
  if (n->matches(torch::schema(
          "aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量 self、other 和标量 alpha
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto alpha = p_node->Input(2).toScalar();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算 add_relu(self, other, alpha)，并存储到输出节点
        p_node->Output(0) = at::native::add_relu(self, other, alpha);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在指定的输出张量上计算 add_relu(self, other, alpha)
      at::native::add_relu_out(self, other, alpha, out);
    };
  }
  // 如果节点不匹配模式，则记录并转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addmv, aten_addmv, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 aten::addmv 模式
  if (n->matches(torch::schema(
          "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量 self、mat、vec 和标量 beta、alpha
      const auto& self = p_node->Input(0).toTensor();
      const auto& mat = p_node->Input(1).toTensor();
      const auto& vec = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算 addmv(self, mat, vec, beta, alpha)，并存储到输出节点
        p_node->Output(0) = at::cpu::addmv(self, mat, vec, beta, alpha);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在指定的输出张量上计算 addmv(self, mat, vec, beta, alpha)
      at::cpu::addmv_out(out, self, mat, vec, beta, alpha);
    };
  }
  // 如果节点不匹配模式，则记录并转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addr, aten_addr, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 aten::addr 模式
  if (n->matches(torch::schema(
          "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量 self、vec1、vec2 和标量 beta、alpha
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec1 = p_node->Input(1).toTensor();
      const auto& vec2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算 addr(self, vec1, vec2, beta, alpha)，并存储到输出节点
        p_node->Output(0) = at::cpu::addr(self, vec1, vec2, beta, alpha);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在指定的输出张量上计算 addr(self, vec1, vec2, beta, alpha)
      at::cpu::addr_out(out, self, vec1, vec2, beta, alpha);
    };
  }
  // 如果节点不匹配模式，则记录并转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});
    return [](ProcessedNode* p_node) {
      // 匿名函数，接受一个指向 ProcessedNode 的指针 p_node 参数

      // 从 p_node 输入中获取张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 从 p_node 输入中获取张量 vec1
      const auto& vec1 = p_node->Input(1).toTensor();
      // 从 p_node 输入中获取张量 vec2
      const auto& vec2 = p_node->Input(2).toTensor();
      // 从 p_node 输入中获取标量 beta
      const auto beta = p_node->Input(3).toScalar();
      // 从 p_node 输入中获取标量 alpha
      const auto alpha = p_node->Input(4).toScalar();

      // 如果 p_node 的输出(0)为空
      if (p_node->Output(0).isNone()) {
        // 计算并将结果赋给 p_node 的输出(0)
        p_node->Output(0) = at::native::addr(self, vec1, vec2, beta, alpha);
        // 函数执行完毕
        return;
      }

      // 获取 p_node 的输出(0)并转换为张量 out
      auto& out = p_node->Output(0).toTensor();
      // 快速调整 out 的大小为零
      fastResizeToZero(out);
      // 使用 addr_out 函数计算并将结果存入 out
      at::native::addr_out(self, vec1, vec2, beta, alpha, out);
    };
  }
  // 记录和转储 Schema 信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::_test_functorch_fallback,
    aten__test_functorch_fallback,
    // 注册自定义运算符处理函数，处理特定的 Torch 节点
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 模式
      if (n->matches(torch::schema(
              "aten::_test_functorch_fallback(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量未初始化，则调用 at::native::_test_functorch_fallback 函数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::_test_functorch_fallback(self, other);
            return;
          }
          // 否则，重设输出张量的大小并调用 at::native::_test_functorch_fallback_out 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::_test_functorch_fallback_out(self, other, out);
        };
      }
      // 如果节点不匹配，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::argmax, aten_argmax, 
    // 注册自定义运算符处理函数，处理 torch.argmax 节点
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 模式
      if (n->matches(torch::schema(
              "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"))) {
        // 返回一个处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self，维度 dim 和 keepdim 标志
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toOptional<int64_t>();
          const auto keepdim = p_node->Input(2).toBool();
          // 如果输出张量未初始化，则调用 at::cpu::argmax 函数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::argmax(self, dim, keepdim);
            return;
          }
          // 否则，重设输出张量的大小并调用 at::cpu::argmax_out 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::argmax_out(out, self, dim, keepdim);
        };
      }
      // 如果节点不匹配，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::acosh, aten_acosh, 
    // 注册自定义运算符处理函数，处理 torch.acosh 节点
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 模式
      if (n->matches(torch::schema("aten::acosh(Tensor self) -> Tensor"))) {
        // 返回一个处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出张量未初始化，则调用 at::cpu::acosh 函数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::acosh(self);
            return;
          }
          // 否则，重设输出张量的大小并调用 at::cpu::acosh_out 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::acosh_out(out, self);
        };
      }
      // 如果节点不匹配，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::asinh, aten_asinh, 
    // 注册自定义运算符处理函数，处理 torch.asinh 节点
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 模式
      if (n->matches(torch::schema("aten::asinh(Tensor self) -> Tensor"))) {
        // 返回一个处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出张量未初始化，则调用 at::cpu::asinh 函数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::asinh(self);
            return;
          }
          // 否则，重设输出张量的大小并调用 at::cpu::asinh_out 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::asinh_out(out, self);
        };
      }
      // 如果节点不匹配，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::arcsinh,
    aten_arcsinh,
    // 注册自定义运算符处理函数，处理 torch.arcsinh 节点
    // 匿名函数，接受一个 Node 指针参数，返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch::schema
      if (n->matches(torch::schema("aten::arcsinh(Tensor self) -> Tensor"))) {
        // 返回另一个匿名函数，接受一个 ProcessedNode 指针参数
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出张量为空，则计算 arcsinh(self) 并赋值给输出张量
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arcsinh(self);
            return;
          }
          // 否则，获取输出张量 out，并调用 fastResizeToZero 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::native::arcsinh_out 函数计算 arcsinh(self) 并存储到 out
          at::native::arcsinh_out(self, out);
        };
      }
      // 如果节点不匹配给定的 schema，则记录日志并转储 schema
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(aten::atanh, aten_atanh, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::atanh(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::atanh(Tensor self) -> Tensor"))) {
    // 返回一个lambda函数，处理输入节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为self张量
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将计算的atanh(self)结果存入输出节点的第一个输出
        p_node->Output(0) = at::cpu::atanh(self);
        return;
      }
      // 否则，获取输出节点的第一个输出作为out张量
      auto& out = p_node->Output(0).toTensor();
      // 将输出张量快速调整大小为零
      fastResizeToZero(out);
      // 在给定的输出张量上计算atanh(self)
      at::cpu::atanh_out(out, self);
    };
  }
  // 如果不符合模式，记录并转储节点的模式信息
  LogAndDumpSchema(n);
  // 返回空指针表示注册失败
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::arctanh,
    aten_arctanh,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::arctanh(Tensor self) -> Tensor" 的模式
      if (n->matches(torch::schema("aten::arctanh(Tensor self) -> Tensor"))) {
        // 返回一个lambda函数，处理输入节点的逻辑
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个输入作为self张量
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出节点的第一个输出为空
          if (p_node->Output(0).isNone()) {
            // 将计算的arctanh(self)结果存入输出节点的第一个输出
            p_node->Output(0) = at::native::arctanh(self);
            return;
          }
          // 否则，获取输出节点的第一个输出作为out张量
          auto& out = p_node->Output(0).toTensor();
          // 将输出张量快速调整大小为零
          fastResizeToZero(out);
          // 在给定的输出张量上计算arctanh(self)
          at::native::arctanh_out(self, out);
        };
      }
      // 如果不符合模式，记录并转储节点的模式信息
      LogAndDumpSchema(n);
      // 返回空指针表示注册失败
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::asin, aten_asin, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::asin(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::asin(Tensor self) -> Tensor"))) {
    // 返回一个lambda函数，处理输入节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为self张量
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将计算的asin(self)结果存入输出节点的第一个输出
        p_node->Output(0) = at::cpu::asin(self);
        return;
      }
      // 否则，获取输出节点的第一个输出作为out张量
      auto& out = p_node->Output(0).toTensor();
      // 将输出张量快速调整大小为零
      fastResizeToZero(out);
      // 在给定的输出张量上计算asin(self)
      at::cpu::asin_out(out, self);
    };
  }
  // 如果不符合模式，记录并转储节点的模式信息
  LogAndDumpSchema(n);
  // 返回空指针表示注册失败
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::arcsin, aten_arcsin, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::arcsin(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::arcsin(Tensor self) -> Tensor"))) {
    // 返回一个lambda函数，处理输入节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为self张量
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将计算的arcsin(self)结果存入输出节点的第一个输出
        p_node->Output(0) = at::native::arcsin(self);
        return;
      }
      // 否则，获取输出节点的第一个输出作为out张量
      auto& out = p_node->Output(0).toTensor();
      // 将输出张量快速调整大小为零
      fastResizeToZero(out);
      // 在给定的输出张量上计算arcsin(self)
      at::native::arcsin_out(self, out);
    };
  }
  // 如果不符合模式，记录并转储节点的模式信息
  LogAndDumpSchema(n);
  // 返回空指针表示注册失败
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::atan, aten_atan, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::atan(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::atan(Tensor self) -> Tensor"))) {
    // 返回一个lambda函数，处理输入节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为self张量
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将计算的atan(self)结果存入输出节点的第一个输出
        p_node->Output(0) = at::cpu::atan(self);
        return;
      }
      // 否则，获取输出节点的第一个输出作为out张量
      auto& out = p_node->Output(0).toTensor();
      // 将输出张量快速调整大小为零
      fastResizeToZero(out);
      // 在给定的输出张量上计算atan(self)
      at::cpu::atan_out(out, self);
    };
  }
  // 如果不符合模式，记录并转储节点的模式信息
  LogAndDumpSchema(n);
  // 返回空指针表示注册失败
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::arctan, aten_arctan, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::arctan(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::arctan(Tensor self) -> Tensor"))) {
    // 返回一个lambda函数，处理输入节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为self张量
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将计算的arctan(self)结果存入输出节点的第一个输出
        p_node->Output(0) = at::native::arctan(self);
        return;
      }
      // 否则，获取输出节点的第一个输出作为out张量
      auto& out = p_node->Output(0).toTensor();
      // 将输出张量快速调整大小为零
      fastResizeToZero(out);
      // 在给定的输出张量上计算arctan(self)
      at::native::arctan_out(self, out);
    };
  }
  // 如果不符合模式，记录并转储节点的模式信息
  LogAndDumpSchema(n);
  // 返回空指针表示注册失败
  return nullptr;
});
    return [](ProcessedNode* p_node) {
      // 使用 Lambda 表达式定义一个匿名函数，该函数接受一个 ProcessedNode 指针参数 p_node
      const auto& self = p_node->Input(0).toTensor();
      // 获取 p_node 的第一个输入作为 Tensor 对象，并命名为 self

      if (p_node->Output(0).isNone()) {
        // 如果 p_node 的第一个输出为空
        p_node->Output(0) = at::native::arctan(self);
        // 对 self 进行反正切运算，并将结果赋给 p_node 的第一个输出
        return;
        // 结束 Lambda 函数的执行
      }

      auto& out = p_node->Output(0).toTensor();
      // 获取 p_node 的第一个输出作为 Tensor 对象，并命名为 out

      fastResizeToZero(out);
      // 调用 fastResizeToZero 函数，对 out 进行快速调整大小的操作

      at::native::arctan_out(self, out);
      // 对 self 执行反正切运算，将结果存储到 out 中

    };
    // Lambda 函数定义结束

  }
  // 函数定义结束

  LogAndDumpSchema(n);
  // 调用 LogAndDumpSchema 函数，参数为 n

  return nullptr;
  // 返回空指针
});

REGISTER_OPERATOR_FUNCTOR(aten::baddbmm, aten_baddbmm, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::baddbmm" 的 Torch 脚本模式
  if (n->matches(torch::schema(
          "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& batch1 = p_node->Input(1).toTensor();
      const auto& batch2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      // 如果输出为空，执行计算并存储到输出中
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::baddbmm(self, batch1, batch2, beta, alpha);
        return;
      }
      // 否则，重新调整输出尺寸为零并执行计算
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::baddbmm_out(out, self, batch1, batch2, beta, alpha);
    };
  }
  // 如果不匹配，记录日志并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_not,
    aten_bitwise_not,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::bitwise_not" 的 Torch 脚本模式
      if (n->matches(
              torch::schema("aten::bitwise_not(Tensor self) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出为空，执行按位取反操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_not(self);
            return;
          }
          // 否则，重新调整输出尺寸为零并执行按位取反操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_not_out(out, self);
        };
      }
      // 如果不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::copysign,
    aten_copysign,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::copysign.Tensor" 的 Torch 脚本模式
      if (n->matches(torch::schema(
              "aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为空，执行 copysign 操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::copysign(self, other);
            return;
          }
          // 否则，重新调整输出尺寸为零并执行 copysign 操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::copysign_out(out, self, other);
        };
      }
      // 如果不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_not,
    aten_logical_not,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::logical_not" 的 Torch 脚本模式
      if (n->matches(
              torch::schema("aten::logical_not(Tensor self) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出为空，执行逻辑非操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_not(self);
            return;
          }
          // 否则，重新调整输出尺寸为零并执行逻辑非操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_not_out(self, out);
        };
      }
      // 如果不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_xor,
    aten_logical_xor,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::logical_xor" 的 Torch 脚本模式
      if (n->matches(
              torch::schema("aten::logical_xor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为空，执行逻辑异或操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::logical_xor(self, other);
            return;
          }
          // 否则，重新调整输出尺寸为零并执行逻辑异或操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::logical_xor_out(out, self, other);
        };
      }
      // 如果不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个 Node* 参数，返回一个 SROperator 对象
      if (n->matches(torch::schema(
              "aten::logical_xor(Tensor self, Tensor other) -> Tensor"))) {
        // 如果节点 n 匹配逻辑异或操作的 Torch 脚本
        return [](ProcessedNode* p_node) {
          // 返回一个新的匿名函数，接受一个 ProcessedNode* 参数
          const auto& self = p_node->Input(0).toTensor();
          // 获取输入参数的第一个张量 self
          const auto& other = p_node->Input(1).toTensor();
          // 获取输入参数的第二个张量 other
          if (p_node->Output(0).isNone()) {
            // 如果输出张量为空
            p_node->Output(0) = at::native::logical_xor(self, other);
            // 计算逻辑异或结果并存入输出节点的第一个输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则，获取输出张量 out
          fastResizeToZero(out);
          // 快速调整输出张量大小为零
          at::native::logical_xor_out(self, other, out);
          // 使用 out 存储逻辑异或操作的结果
        };
      }
      LogAndDumpSchema(n);
      // 如果节点 n 不匹配逻辑异或操作的 Torch 脚本，则记录并转储其模式
      return nullptr;
      // 返回空指针
    });
    // 匿名函数结束，代码块结束
REGISTER_OPERATOR_FUNCTOR(
    aten::logical_and,  // 注册逻辑与运算符，处理aten::logical_and操作
    aten_logical_and,   // 对应的函数名为aten_logical_and
    [](Node* n) -> SROperator {  // Lambda表达式，接受一个Node指针参数，返回一个SROperator
      if (n->matches(torch::schema(  // 检查节点是否匹配指定的schema
              "aten::logical_and(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回处理逻辑与操作的Lambda表达式
          const auto& self = p_node->Input(0).toTensor();  // 获取第一个输入Tensor
          const auto& other = p_node->Input(1).toTensor();  // 获取第二个输入Tensor
          if (p_node->Output(0).isNone()) {  // 如果输出Tensor为空
            p_node->Output(0) = at::native::logical_and(self, other);  // 执行逻辑与操作并存储结果
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出Tensor的引用
          fastResizeToZero(out);  // 快速调整输出Tensor的大小
          at::native::logical_and_out(self, other, out);  // 将逻辑与操作结果存入输出Tensor
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配schema的日志信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_or,  // 注册逻辑或运算符，处理aten::logical_or操作
    aten_logical_or,   // 对应的函数名为aten_logical_or
    [](Node* n) -> SROperator {  // Lambda表达式，接受一个Node指针参数，返回一个SROperator
      if (n->matches(torch::schema(  // 检查节点是否匹配指定的schema
              "aten::logical_or(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回处理逻辑或操作的Lambda表达式
          const auto& self = p_node->Input(0).toTensor();  // 获取第一个输入Tensor
          const auto& other = p_node->Input(1).toTensor();  // 获取第二个输入Tensor
          if (p_node->Output(0).isNone()) {  // 如果输出Tensor为空
            p_node->Output(0) = at::native::logical_or(self, other);  // 执行逻辑或操作并存储结果
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出Tensor的引用
          fastResizeToZero(out);  // 快速调整输出Tensor的大小
          at::native::logical_or_out(self, other, out);  // 将逻辑或操作结果存入输出Tensor
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配schema的日志信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::ceil,  // 注册ceil操作符，处理aten::ceil操作
    aten_ceil,   // 对应的函数名为aten_ceil
    [](Node* n) -> SROperator {  // Lambda表达式，接受一个Node指针参数，返回一个SROperator
      if (n->matches(torch::schema("aten::ceil(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回处理ceil操作的Lambda表达式
          const auto& self = p_node->Input(0).toTensor();  // 获取输入Tensor
          if (p_node->Output(0).isNone()) {  // 如果输出Tensor为空
            p_node->Output(0) = at::cpu::ceil(self);  // 执行ceil操作并存储结果
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出Tensor的引用
          fastResizeToZero(out);  // 快速调整输出Tensor的大小
          at::cpu::ceil_out(out, self);  // 将ceil操作结果存入输出Tensor
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配schema的日志信息
      return nullptr;  // 返回空指针表示注册失败
    });

// 下面的代码未提供完整，无法进行注释
REGISTER_OPERATOR_FUNCTOR(
    aten::clamp_max,
    aten_clamp_max,
    // 匿名函数，接受一个 Node 指针参数，返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 如果 Node 匹配指定的 torch::schema，执行以下代码块
      if (n->matches(torch::schema(
              "aten::clamp_max(Tensor self, Scalar max) -> Tensor"))) {
        // 返回一个匿名函数，接受一个 ProcessedNode 指针参数
        return [](ProcessedNode* p_node) {
          // 获取输入参数 self 和 max
          const auto& self = p_node->Input(0).toTensor();
          const auto max = p_node->Input(1).toScalar();
          // 如果输出为空，执行以下代码块
          if (p_node->Output(0).isNone()) {
            // 对输出进行赋值
            p_node->Output(0) = at::cpu::clamp_max(self, max);
            return;
          }
          // 获取输出参数 out
          auto& out = p_node->Output(0).toTensor();
          // 调用 fastResizeToZero 函数
          fastResizeToZero(out);
          // 调用 at::cpu::clamp_max_out 函数
          at::cpu::clamp_max_out(out, self, max);
        };
      }

      // 如果 Node 匹配指定的 torch::schema，执行以下代码块
      if (n->matches(torch::schema(
              "aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor"))) {
        // 返回一个匿名函数，接受一个 ProcessedNode 指针参数
        return [](ProcessedNode* p_node) {
          // 获取输入参数 self 和 max
          const auto& self = p_node->Input(0).toTensor();
          const auto& max = p_node->Input(1).toTensor();
          // 如果输出为空，执行以下代码块
          if (p_node->Output(0).isNone()) {
            // 对输出进行赋值
            p_node->Output(0) = at::cpu::clamp_max(self, max);
            return;
          }
          // 获取输出参数 out
          auto& out = p_node->Output(0).toTensor();
          // 调用 fastResizeToZero 函数
          fastResizeToZero(out);
          // 调用 at::cpu::clamp_max_out 函数
          at::cpu::clamp_max_out(out, self, max);
        };
      }
      // 如果以上条件都不满足，调用 LogAndDumpSchema 函数
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(aten::clip, aten_clip, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch schema
  if (n->matches(torch::schema(
          "aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配成功的节点
    return [](ProcessedNode* p_node) {
      // 从输入中获取 self 张量
      const auto& self = p_node->Input(0).toTensor();
      // 从输入中获取 min 和 max 标量（可选）
      const auto min = p_node->Input(1).toOptional<at::Scalar>();
      const auto max = p_node->Input(2).toOptional<at::Scalar>();
      // 如果输出未定义，则调用 at::native::clip 进行计算，并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::clip(self, min, max);
        return;
      }
      // 否则，获取输出张量的引用，并进行快速调整大小到零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::clip_out 将结果写入输出张量
      at::native::clip_out(self, min, max, out);
    };
  }
  // 如果节点不匹配，则记录和转储 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::complex,
    aten_complex,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch schema
      if (n->matches(torch::schema(
              "aten::complex(Tensor real, Tensor imag) -> Tensor"))) {
        // 返回一个 lambda 表达式，处理匹配成功的节点
        return [](ProcessedNode* p_node) {
          // 从输入中获取 real 和 imag 张量
          const auto& real = p_node->Input(0).toTensor();
          const auto& imag = p_node->Input(1).toTensor();
          // 如果输出未定义，则调用 at::native::complex 进行计算，并设置输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::complex(real, imag);
            return;
          }
          // 否则，获取输出张量的引用，并进行快速调整大小到零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::native::complex_out 将结果写入输出张量
          at::native::complex_out(real, imag, out);
        };
      }
      // 如果节点不匹配，则记录和转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::polar, aten_polar, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch schema
  if (n->matches(
          torch::schema("aten::polar(Tensor abs, Tensor angle) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配成功的节点
    return [](ProcessedNode* p_node) {
      // 从输入中获取 abs 和 angle 张量
      const auto& abs = p_node->Input(0).toTensor();
      const auto& angle = p_node->Input(1).toTensor();
      // 如果输出未定义，则调用 at::native::polar 进行计算，并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::polar(abs, angle);
        return;
      }
      // 否则，获取输出张量的引用，并进行快速调整大小到零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::polar_out 将结果写入输出张量
      at::native::polar_out(abs, angle, out);
    };
  }
  // 如果节点不匹配，则记录和转储 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::cos, aten_cos, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch schema
  if (n->matches(torch::schema("aten::cos(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配成功的节点
    return [](ProcessedNode* p_node) {
      // 从输入中获取 self 张量
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出未定义，则调用 at::cpu::cos 进行计算，并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cos(self);
        return;
      }
      // 否则，获取输出张量的引用，并进行快速调整大小到零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::cos_out 将结果写入输出张量
      at::cpu::cos_out(out, self);
    };
  }
  // 如果节点不匹配，则记录和转储 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::cosh, aten_cosh, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch schema
  if (n->matches(torch::schema("aten::cosh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 定义一个 lambda 函数，接受一个 ProcessedNode 指针参数 p_node
      const auto& self = p_node->Input(0).toTensor();
      // 获取 p_node 的第一个输入，并转换为 Tensor 类型，存储在 self 中
      if (p_node->Output(0).isNone()) {
        // 如果 p_node 的第一个输出为空
        p_node->Output(0) = at::cpu::cosh(self);
        // 对 self 应用双曲余弦函数，并将结果赋给 p_node 的第一个输出
        return;
        // 返回结束 lambda 函数的执行
      }
      // 如果 p_node 的第一个输出不为空，则执行以下操作
      auto& out = p_node->Output(0).toTensor();
      // 获取 p_node 的第一个输出，并将其转换为 Tensor 类型，存储在 out 中
      fastResizeToZero(out);
      // 调用 fastResizeToZero 函数，对 out 进行快速调整大小到零
      at::cpu::cosh_out(out, self);
      // 对 self 应用双曲余弦函数，结果存储在 out 中
    };
    // lambda 函数定义结束
  }
  // 返回 nullptr，结束当前函数的执行
  LogAndDumpSchema(n);
  // 调用 LogAndDumpSchema 函数，参数为 n
  return nullptr;
  // 返回 nullptr，函数结束
});

REGISTER_OPERATOR_FUNCTOR(aten::cumprod, aten_cumprod, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 cumprod 操作的模式
  if (n->matches(torch::schema(
          "aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto dtype = p_node->Input(2).toOptional<at::ScalarType>();
      // 如果输出未定义，则执行操作并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cumprod(self, dim, dtype);
        return;
      }
      // 否则，调整输出张量大小并执行 cumprod 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cumprod_out(out, self, dim, dtype);
    };
  }
  // 如果模式不匹配，则记录并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::diff, aten_diff, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 diff 操作的模式
  if (n->matches(torch::schema(
          "aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto n = p_node->Input(1).toInt();
      const auto dim = p_node->Input(2).toInt();
      const auto prepend = p_node->Input(3).toOptional<at::Tensor>();
      const auto append = p_node->Input(4).toOptional<at::Tensor>();
      // 如果输出未定义，则执行操作并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::diff(self, n, dim, prepend, append);
        return;
      }
      // 否则，调整输出张量大小并执行 diff 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::diff_out(self, n, dim, prepend, append, out);
    };
  }
  // 如果模式不匹配，则记录并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::divide, aten_divide, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 divide 操作的模式
  if (n->matches(torch::schema(
          "aten::divide.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出未定义，则执行操作并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::divide(self, other);
        return;
      }
      // 否则，调整输出张量大小并执行 divide 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::divide_out(self, other, out);
    };
  }
  // 如果模式不匹配，则记录并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::true_divide,
    aten_true_divide,
    // 检查节点是否匹配指定的 true_divide 操作的模式
    [](Node* n) -> SROperator {
        // 匿名函数，接受一个 Node* 参数并返回一个 SROperator
        if (n->matches(torch::schema(
                "aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor"))) {
            // 如果节点 n 匹配 true_divide 的 schema
            return [](ProcessedNode* p_node) {
                // 返回另一个匿名函数，接受一个 ProcessedNode* 参数
                const auto& self = p_node->Input(0).toTensor();
                // 获取输入的第一个张量 self
                const auto& other = p_node->Input(1).toTensor();
                // 获取输入的第二个张量 other
                if (p_node->Output(0).isNone()) {
                    // 如果输出是空的
                    p_node->Output(0) = at::native::true_divide(self, other);
                    // 计算 true_divide 结果并存储在输出中
                    return;
                }
                auto& out = p_node->Output(0).toTensor();
                // 否则获取输出张量 out
                fastResizeToZero(out);
                // 调用快速重设大小函数 fastResizeToZero
                at::native::true_divide_out(self, other, out);
                // 使用 true_divide_out 计算结果并存储在输出张量中
            };
        }
        // 如果节点 n 不匹配 true_divide 的 schema，记录并转储其 schema
        LogAndDumpSchema(n);
        // 返回空指针
        return nullptr;
    });
    // 结束匿名函数的定义
REGISTER_OPERATOR_FUNCTOR(aten::dot, aten_dot, [](Node* n) -> SROperator {
  // 检查节点是否符合特定的 torch::schema 规则
  if (n->matches(
          torch::schema("aten::dot(Tensor self, Tensor tensor) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个（self）和第二个（tensor）张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& tensor = p_node->Input(1).toTensor();
      // 如果输出节点未初始化，执行 at::native::dot 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::dot(self, tensor);
        return;
      }
      // 否则，获取输出张量并使用 fastResizeToZero 函数重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行 at::native::dot_out 操作，将结果存入 out 张量
      at::native::dot_out(self, tensor, out);
    };
  }
  // 若不符合条件，记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::vdot, aten_vdot, [](Node* n) -> SROperator {
  // 检查节点是否符合特定的 torch::schema 规则
  if (n->matches(
          torch::schema("aten::vdot(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个（self）和第二个（other）张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出节点未初始化，执行 at::native::vdot 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::vdot(self, other);
        return;
      }
      // 否则，获取输出张量并使用 fastResizeToZero 函数重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行 at::native::vdot_out 操作，将结果存入 out 张量
      at::native::vdot_out(self, other, out);
    };
  }
  // 若不符合条件，记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::erf, aten_erf, [](Node* n) -> SROperator {
  // 检查节点是否符合特定的 torch::schema 规则
  if (n->matches(torch::schema("aten::erf(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量（self）
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点未初始化，执行 at::cpu::erf 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erf(self);
        return;
      }
      // 否则，获取输出张量并使用 fastResizeToZero 函数重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行 at::cpu::erf_out 操作，将结果存入 out 张量
      at::cpu::erf_out(out, self);
    };
  }
  // 若不符合条件，记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::erfc, aten_erfc, [](Node* n) -> SROperator {
  // 检查节点是否符合特定的 torch::schema 规则
  if (n->matches(torch::schema("aten::erfc(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量（self）
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点未初始化，执行 at::cpu::erfc 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erfc(self);
        return;
      }
      // 否则，获取输出张量并使用 fastResizeToZero 函数重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行 at::cpu::erfc_out 操作，将结果存入 out 张量
      at::cpu::erfc_out(out, self);
    };
  }
  // 若不符合条件，记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::exp, aten_exp, [](Node* n) -> SROperator {
  // 检查节点是否符合特定的 torch::schema 规则
  if (n->matches(torch::schema("aten::exp(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量（self）
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点未初始化，执行 at::cpu::exp 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::exp(self);
        return;
      }
      // 否则，获取输出张量并使用 fastResizeToZero 函数重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行 at::cpu::exp_out 操作，将结果存入 out 张量
      at::cpu::exp_out(out, self);
    };
  }
  // 若不符合条件，记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::exp2, aten_exp2, [](Node* n) -> SROperator {
  // 检查节点是否符合特定的 torch::schema 规则
  if (n->matches(torch::schema("aten::exp2(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 定义 lambda 表达式，接受一个 ProcessedNode 指针作为参数
      const auto& self = p_node->Input(0).toTensor();
      // 获取 p_node 的第一个输入，并转换为 Tensor，存储在 self 中
      if (p_node->Output(0).isNone()) {
        // 如果 p_node 的第一个输出为空
        p_node->Output(0) = at::cpu::exp2(self);
        // 对 self 执行 exp2 运算，将结果存储到 p_node 的第一个输出中
        return;
        // 返回空，结束 lambda 表达式的执行
      }
      // 否则，如果 p_node 的第一个输出不为空
      auto& out = p_node->Output(0).toTensor();
      // 获取 p_node 的第一个输出，并将其转换为 Tensor，存储在 out 中
      fastResizeToZero(out);
      // 调用 fastResizeToZero 函数，将 out 的大小调整为零
      at::cpu::exp2_out(out, self);
      // 对 self 执行 exp2 运算，并将结果存储到 out 中
    };
  }
  // 以上为 lambda 表达式的定义和实现
  
  LogAndDumpSchema(n);
  // 调用 LogAndDumpSchema 函数，传递参数 n
  return nullptr;
  // 返回空指针
// 注册 ATen 库中的 expm1 运算符，将其实现为一个函数对象
REGISTER_OPERATOR_FUNCTOR(aten::expm1, aten_expm1, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 expm1 函数签名
  if (n->matches(torch::schema("aten::expm1(Tensor self) -> Tensor"))) {
    // 如果匹配，返回一个 lambda 表达式，处理节点数据
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，计算 expm1(self) 并存储在输出中
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::expm1(self);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出中计算 expm1(self)
      at::cpu::expm1_out(out, self);
    };
  }
  // 如果不匹配，记录和转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 ATen 库中的 floor 运算符，将其实现为一个函数对象
REGISTER_OPERATOR_FUNCTOR(aten::floor, aten_floor, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 floor 函数签名
  if (n->matches(torch::schema("aten::floor(Tensor self) -> Tensor"))) {
    // 如果匹配，返回一个 lambda 表达式，处理节点数据
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，计算 floor(self) 并存储在输出中
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::floor(self);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出中计算 floor(self)
      at::cpu::floor_out(out, self);
    };
  }
  // 如果不匹配，记录和转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 ATen 库中的 frac 运算符，将其实现为一个函数对象
REGISTER_OPERATOR_FUNCTOR(aten::frac, aten_frac, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 frac 函数签名
  if (n->matches(torch::schema("aten::frac(Tensor self) -> Tensor"))) {
    // 如果匹配，返回一个 lambda 表达式，处理节点数据
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，计算 frac(self) 并存储在输出中
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::frac(self);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出中计算 frac(self)
      at::cpu::frac_out(out, self);
    };
  }
  // 如果不匹配，记录和转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 ATen 库中的 gcd 运算符，将其实现为一个函数对象
REGISTER_OPERATOR_FUNCTOR(aten::gcd, aten_gcd, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 gcd 函数签名
  if (n->matches(
          torch::schema("aten::gcd(Tensor self, Tensor other) -> Tensor"))) {
    // 如果匹配，返回一个 lambda 表达式，处理节点数据
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出为空，计算 gcd(self, other) 并存储在输出中
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gcd(self, other);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出中计算 gcd(self, other)
      at::cpu::gcd_out(out, self, other);
    };
  }
  // 如果不匹配，记录和转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 ATen 库中的 lcm 运算符，将其实现为一个函数对象
REGISTER_OPERATOR_FUNCTOR(aten::lcm, aten_lcm, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 lcm 函数签名
  if (n->matches(
          torch::schema("aten::lcm(Tensor self, Tensor other) -> Tensor"))) {
    // 如果匹配，返回一个 lambda 表达式，处理节点数据
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出为空，计算 lcm(self, other) 并存储在输出中
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lcm(self, other);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出中计算 lcm(self, other)
      at::cpu::lcm_out(out, self, other);
    };
  }
  // 如果不匹配，记录和转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
REGISTER_OPERATOR_FUNCTOR(aten::index_copy, aten_index_copy, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 Torch 操作模式
  if (n->matches(torch::schema(
          "aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配到的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& source = p_node->Input(3).toTensor();
      // 如果输出为空，直接调用 index_copy 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::index_copy(self, dim, index, source);
        return;
      }
      // 否则，调用 fastResizeToZero 函数，然后调用 index_copy_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::index_copy_out(out, self, dim, index, source);
    };
  }
  // 如果未匹配到指定模式，记录并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::isin, aten_isin, [](Node* n) -> SROperator {
  // 检查节点是否匹配第一种 isin 操作模式
  if (n->matches(torch::schema(
          "aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配到的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& elements = p_node->Input(0).toTensor();
      const auto& test_elements = p_node->Input(1).toTensor();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      // 如果输出为空，直接调用 isin 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(elements, test_elements, assume_unique, invert);
        return;
      }
      // 否则，调用 fastResizeToZero 函数，然后调用 isin_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_elements, assume_unique, invert);
    };
  }

  // 检查节点是否匹配第二种 isin 操作模式
  if (n->matches(torch::schema(
          "aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配到的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& elements = p_node->Input(0).toTensor();
      const auto test_element = p_node->Input(1).toScalar();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      // 如果输出为空，直接调用 isin 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(elements, test_element, assume_unique, invert);
        return;
      }
      // 否则，调用 fastResizeToZero 函数，然后调用 isin_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_element, assume_unique, invert);
    };
  }

  // 检查节点是否匹配第三种 isin 操作模式
  if (n->matches(torch::schema(
          "aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理匹配到的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& elements = p_node->Input(1).toTensor();
      const auto element = p_node->Input(0).toScalar();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      // 如果输出为空，直接调用 isin 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(element, elements, assume_unique, invert);
        return;
      }
      // 否则，调用 fastResizeToZero 函数，然后调用 isin_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, element, elements, assume_unique, invert);
    };
  }

  // 如果未匹配到任何 isin 操作模式，记录并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    return [](ProcessedNode* p_node) {
      // 定义一个 Lambda 函数，接收一个 ProcessedNode 指针作为参数
      const auto element = p_node->Input(0).toScalar();  
      // 从 p_node 的第一个输入中获取标量 element
      const auto& test_elements = p_node->Input(1).toTensor();  
      // 从 p_node 的第二个输入中获取张量 test_elements
      const auto assume_unique = p_node->Input(2).toBool();  
      // 从 p_node 的第三个输入中获取布尔值 assume_unique
      const auto invert = p_node->Input(3).toBool();  
      // 从 p_node 的第四个输入中获取布尔值 invert
      
      // 如果 p_node 的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算并设置 p_node 的第一个输出为 at::cpu::isin 函数的结果
        p_node->Output(0) =
            at::cpu::isin(element, test_elements, assume_unique, invert);
        // 函数执行完毕，返回
        return;
      }
      
      // 否则，如果 p_node 的第一个输出不为空
      auto& out = p_node->Output(0).toTensor();
      // 将 p_node 的第一个输出转换为张量，并引用为 out
      fastResizeToZero(out);
      // 调用 fastResizeToZero 函数，快速调整 out 张量大小为零
      at::cpu::isin_out(out, element, test_elements, assume_unique, invert);
      // 调用 at::cpu::isin_out 函数，使用 element、test_elements、assume_unique 和 invert 在 out 上执行 in-place 操作
    };
  }
  LogAndDumpSchema(n);
  // 调用 LogAndDumpSchema 函数，记录并转储 n 的模式信息
  return nullptr;
  // 返回空指针
REGISTER_OPERATOR_FUNCTOR(aten::kron, aten_kron, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema
  if (n->matches(
          torch::schema("aten::kron(Tensor self, Tensor other) -> Tensor"))) {
    // 返回 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量为空，则调用 at::native::kron 创建输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::kron(self, other);
        return;
      }
      // 否则，获取输出张量并调用 fastResizeToZero 进行快速重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::kron_out 将结果存入输出张量 out
      at::native::kron_out(self, other, out);
    };
  }
  // 如果不匹配，则记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::ldexp, aten_ldexp, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema
  if (n->matches(torch::schema(
          "aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量为空，则调用 at::native::ldexp 创建输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::ldexp(self, other);
        return;
      }
      // 否则，获取输出张量并调用 fastResizeToZero 进行快速重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::ldexp_out 将结果存入输出张量 out
      at::native::ldexp_out(self, other, out);
    };
  }
  // 如果不匹配，则记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::log10, aten_log10, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema
  if (n->matches(torch::schema("aten::log10(Tensor self) -> Tensor"))) {
    // 返回 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出张量为空，则调用 at::cpu::log10 创建输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log10(self);
        return;
      }
      // 否则，获取输出张量并调用 fastResizeToZero 进行快速重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::log10_out 将结果存入输出张量 out
      at::cpu::log10_out(out, self);
    };
  }
  // 如果不匹配，则记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::log1p, aten_log1p, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema
  if (n->matches(torch::schema("aten::log1p(Tensor self) -> Tensor"))) {
    // 返回 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出张量为空，则调用 at::cpu::log1p 创建输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log1p(self);
        return;
      }
      // 否则，获取输出张量并调用 fastResizeToZero 进行快速重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::log1p_out 将结果存入输出张量 out
      at::cpu::log1p_out(out, self);
    };
  }
  // 如果不匹配，则记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::log2, aten_log2, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema
  if (n->matches(torch::schema("aten::log2(Tensor self) -> Tensor"))) {
    // 返回 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出张量为空，则调用 at::cpu::log2 创建输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log2(self);
        return;
      }
      // 否则，获取输出张量并调用 fastResizeToZero 进行快速重置
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::log2_out 将结果存入输出张量 out
      at::cpu::log2_out(out, self);
    };
  }
  // 如果不匹配，则记录并输出节点的 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::logaddexp,
    aten_logaddexp,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 schema
      if (n->matches(torch::schema("aten::logaddexp(Tensor self, Tensor other) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的逻辑
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，则调用 at::native::logaddexp 创建输出张量
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logaddexp(self, other);
            return;
          }
          // 否则，获取输出张量并调用 fastResizeToZero 进行快速重置
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::native::logaddexp_out 将结果存入输出张量 out
          at::native::logaddexp_out(out, self, other);
        };
      }
      // 如果不匹配，则记录并输出节点的 schema
      LogAndDumpSchema(n);
      return nullptr;
    });
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个指向 Node 的指针 n，并返回一个 SROperator
      if (n->matches(torch::schema(
              "aten::logaddexp(Tensor self, Tensor other) -> Tensor"))) {
        // 如果节点 n 匹配指定的 torch schema
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名函数，接受一个指向 ProcessedNode 的指针 p_node
          const auto& self = p_node->Input(0).toTensor();
          // 获取 p_node 的第一个输入，并将其转换为 Tensor 类型
          const auto& other = p_node->Input(1).toTensor();
          // 获取 p_node 的第二个输入，并将其转换为 Tensor 类型
          if (p_node->Output(0).isNone()) {
            // 如果 p_node 的第一个输出为空
            p_node->Output(0) = at::cpu::logaddexp(self, other);
            // 计算并将 at::cpu::logaddexp 的结果赋给 p_node 的第一个输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则，获取 p_node 的第一个输出，并转换为 Tensor 类型
          fastResizeToZero(out);
          // 调用 fastResizeToZero 对 out 进行快速调整
          at::cpu::logaddexp_out(out, self, other);
          // 使用 at::cpu::logaddexp_out 计算并将结果赋给 out
        };
      }
      LogAndDumpSchema(n);
      // 如果节点 n 不匹配指定的 torch schema，调用 LogAndDumpSchema 输出日志
      return nullptr;
      // 返回空指针
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::logaddexp2,
    aten_logaddexp2,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式
      if (n->matches(torch::schema(
              "aten::logaddexp2(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，直接计算结果并赋给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::logaddexp2(self, other);
            return;
          }
          // 否则，获取输出张量并调用 fastResizeToZero 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::logaddexp2_out 函数计算结果到指定输出张量 out
          at::cpu::logaddexp2_out(out, self, other);
        };
      }
      // 如果不匹配，则记录和转储节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::xlogy,
    aten_xlogy,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式
      if (n->matches(torch::schema(
              "aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，直接计算结果并赋给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::xlogy(self, other);
            return;
          }
          // 否则，获取输出张量并调用 fastResizeToZero 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::xlogy_out 函数计算结果到指定输出张量 out
          at::cpu::xlogy_out(out, self, other);
        };
      }
      // 如果不匹配，则记录和转储节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_log_softmax,
    aten__log_softmax,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式
      if (n->matches(torch::schema(
              "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self、dim 和 half_to_float
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          const auto half_to_float = p_node->Input(2).toBool();
          // 如果输出张量为空，直接计算结果并赋给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax(self, dim, half_to_float);
            return;
          }
          // 否则，获取输出张量并调用 fastResizeToZero 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::_log_softmax_out 函数计算结果到指定输出张量 out
          at::cpu::_log_softmax_out(out, self, dim, half_to_float);
        };
      }
      // 如果不匹配，则记录和转储节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_log_softmax_backward_data,
    aten__log_softmax_backward_data,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式
      if (n->matches(torch::schema(
              "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self, Tensor input) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 grad_output、output、dim、self 和 input
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          const auto& self = p_node->Input(3).toTensor();
          const auto& input = p_node->Input(4).toTensor();
          // 如果输出张量为空，直接计算结果并赋给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax_backward_data(grad_output, output, dim, self, input);
            return;
          }
          // 否则，获取输出张量并调用 fastResizeToZero 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::_log_softmax_backward_data_out 函数计算结果到指定输出张量 out
          at::cpu::_log_softmax_backward_data_out(out, grad_output, output, dim, self, input);
        };
      }
      // 如果不匹配，则记录和转储节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配给定的 Torch 脚本模式
      if (n->matches(torch::schema(
              "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"))) {
        // 如果匹配，返回一个 lambda 函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 从处理后的节点中获取输入参数
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          const auto input_dtype = p_node->Input(3).toScalarType();
          
          // 如果输出未被赋值，则计算并存储结果到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax_backward_data(
                grad_output, output, dim, input_dtype);
            return;
          }
          
          // 否则，重置输出张量的大小并使用提供的函数计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_log_softmax_backward_data_out(
              out, grad_output, output, dim, input_dtype);
        };
      }
      
      // 如果节点不匹配，则记录并转储节点的模式信息
      LogAndDumpSchema(n);
      
      // 返回空指针表示没有相应的操作符处理该节点
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::_logcumsumexp,
    aten__logcumsumexp,
    [](Node* n) -> SROperator {
      // 如果节点匹配 "aten::_logcumsumexp(Tensor self, int dim) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::_logcumsumexp(Tensor self, int dim) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个张量 self 和第二个整数 dim
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          // 如果输出节点的第一个张量为空
          if (p_node->Output(0).isNone()) {
            // 计算 _logcumsumexp 的结果并赋给输出节点
            p_node->Output(0) = at::native::_logcumsumexp_cpu(self, dim);
            return;
          }
          // 否则，获取输出节点的第一个张量，并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 使用 _logcumsumexp_out_cpu 计算结果并存入 out 张量
          at::native::_logcumsumexp_out_cpu(self, dim, out);
        };
      }
      // 如果不匹配，记录和转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logcumsumexp,
    aten_logcumsumexp,
    [](Node* n) -> SROperator {
      // 如果节点匹配 "aten::logcumsumexp(Tensor self, int dim) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::logcumsumexp(Tensor self, int dim) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个张量 self 和第二个整数 dim
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          // 如果输出节点的第一个张量为空
          if (p_node->Output(0).isNone()) {
            // 计算 logcumsumexp 的结果并赋给输出节点
            p_node->Output(0) = at::native::logcumsumexp(self, dim);
            return;
          }
          // 否则，获取输出节点的第一个张量，并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 使用 logcumsumexp_out 计算结果并存入 out 张量
          at::native::logcumsumexp_out(self, dim, out);
        };
      }
      // 如果不匹配，记录和转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::matrix_power,
    aten_matrix_power,
    [](Node* n) -> SROperator {
      // 如果节点匹配 "aten::matrix_power(Tensor self, int n) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::matrix_power(Tensor self, int n) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个张量 self 和第二个整数 n
          const auto& self = p_node->Input(0).toTensor();
          const auto n = p_node->Input(1).toInt();
          // 如果输出节点的第一个张量为空
          if (p_node->Output(0).isNone()) {
            // 计算 matrix_power 的结果并赋给输出节点
            p_node->Output(0) = at::native::matrix_power(self, n);
            return;
          }
          // 否则，获取输出节点的第一个张量，并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 使用 matrix_power_out 计算结果并存入 out 张量
          at::native::matrix_power_out(self, n, out);
        };
      }
      // 如果不匹配，记录和转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::mm, aten_mm, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::mm(Tensor self, Tensor mat2) -> Tensor" 的模式
  if (n->matches(
          torch::schema("aten::mm(Tensor self, Tensor mat2) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个张量 self 和第二个张量 mat2
      const auto& self = p_node->Input(0).toTensor();
      const auto& mat2 = p_node->Input(1).toTensor();
      // 如果输出节点的第一个张量为空
      if (p_node->Output(0).isNone()) {
        // 计算 mm 的结果并赋给输出节点
        p_node->Output(0) = at::cpu::mm(self, mat2);
        return;
      }
      // 否则，获取输出节点的第一个张量，并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 使用 mm_out 计算结果并存入 out 张量
      at::cpu::mm_out(out, self, mat2);
    };
  }
  // 如果不匹配，记录和转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    [](Node* n) -> SROperator {
      // 匿名 Lambda 函数，接受一个 Node* 参数，返回一个 SROperator 对象
    
      if (n->matches(torch::schema(
              "aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 检查节点 n 是否匹配乘法操作的 Torch 脚本
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名 Lambda 函数，接受一个 ProcessedNode* 参数
    
          const auto& self = p_node->Input(0).toTensor();
          // 获取输入节点的第一个张量输入 self
          const auto& other = p_node->Input(1).toTensor();
          // 获取输入节点的第二个张量输入 other
    
          if (p_node->Output(0).isNone()) {
            // 如果输出节点的第一个输出为空
            p_node->Output(0) = at::native::multiply(self, other);
            // 则将 self 和 other 的乘积作为输出节点的第一个输出
            return;
          }
    
          auto& out = p_node->Output(0).toTensor();
          // 否则获取输出节点的第一个输出作为张量 out
          fastResizeToZero(out);
          // 调用 fastResizeToZero 函数对 out 张量进行快速调整大小
          at::native::multiply_out(self, other, out);
          // 调用原生的 multiply_out 函数计算 self 和 other 的乘积，结果存储在 out 中
        };
      }
    
      LogAndDumpSchema(n);
      // 如果节点 n 不匹配乘法操作的 Torch 脚本，则记录并转储节点的模式信息
      return nullptr;
      // 返回空指针
    });
    // Lambda 表达式结束
REGISTER_OPERATOR_FUNCTOR(
    aten::mv,
    aten_mv,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 Torch 模式
      if (n->matches(
              torch::schema("aten::mv(Tensor self, Tensor vec) -> Tensor"))) {
        // 返回一个 Lambda 表达式，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和 vec
          const auto& self = p_node->Input(0).toTensor();
          const auto& vec = p_node->Input(1).toTensor();
          // 如果输出未初始化，则计算并存储输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::mv(self, vec);
            return;
          }
          // 否则，重新调整输出张量的大小并计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::mv_out(self, vec, out);
        };
      }
      // 如果节点模式不匹配，则记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::mvlgamma,
    aten_mvlgamma,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 Torch 模式
      if (n->matches(
              torch::schema("aten::mvlgamma(Tensor self, int p) -> Tensor"))) {
        // 返回一个 Lambda 表达式，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和整数参数 p
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toInt();
          // 如果输出未初始化，则计算并存储输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::mvlgamma(self, p);
            return;
          }
          // 否则，重新调整输出张量的大小并计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::mvlgamma_out(self, p, out);
        };
      }
      // 如果节点模式不匹配，则记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::rad2deg,
    aten_rad2deg,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 Torch 模式
      if (n->matches(torch::schema("aten::rad2deg(Tensor self) -> Tensor"))) {
        // 返回一个 Lambda 表达式，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出未初始化，则计算并存储输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::rad2deg(self);
            return;
          }
          // 否则，重新调整输出张量的大小并计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::rad2deg_out(self, out);
        };
      }
      // 如果节点模式不匹配，则记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::deg2rad,
    aten_deg2rad,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 Torch 模式
      if (n->matches(torch::schema("aten::deg2rad(Tensor self) -> Tensor"))) {
        // 返回一个 Lambda 表达式，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出未初始化，则计算并存储输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::deg2rad(self);
            return;
          }
          // 否则，重新调整输出张量的大小并计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::deg2rad_out(self, out);
        };
      }
      // 如果节点模式不匹配，则记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::reciprocal,
    aten_reciprocal,
    // 接受一个指向 Node 类型的指针 n，并返回一个 SROperator 类型的对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 脚本函数签名 "aten::reciprocal(Tensor self) -> Tensor"
      if (n->matches(
              torch::schema("aten::reciprocal(Tensor self) -> Tensor"))) {
        // 如果匹配，返回一个 lambda 函数，处理匹配成功的节点
        return [](ProcessedNode* p_node) {
          // 获取节点的第一个输入作为 Tensor 类型的 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果节点的第一个输出为空，计算 reciprocal 并存储在第一个输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::reciprocal(self);
            return;
          }
          // 否则，获取第一个输出并调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 使用 reciprocal_out 函数计算 reciprocal，并将结果存储在 out 中
          at::cpu::reciprocal_out(out, self);
        };
      }
      // 如果节点 n 不匹配预期的 Torch 脚本函数签名，则记录日志并转储节点的模式
      LogAndDumpSchema(n);
      // 返回空指针表示没有相应的操作符处理该节点
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(aten::neg, aten_neg, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch schema
  if (n->matches(torch::schema("aten::neg(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，对 self 执行负操作并将结果赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::neg(self);
        return;
      }
      // 否则，获取输出张量并执行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出张量 out 中执行负操作
      at::cpu::neg_out(out, self);
    };
  }
  // 如果未匹配到指定的 schema，则记录和转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::negative,
    aten_negative,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 torch schema
      if (n->matches(torch::schema("aten::negative(Tensor self) -> Tensor"))) {
        // 返回一个 lambda 表达式，处理节点操作
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出为空，对 self 执行 negative 操作并将结果赋给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::negative(self);
            return;
          }
          // 否则，获取输出张量并执行快速调整大小
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 在输出张量 out 中执行 negative 操作
          at::native::negative_out(self, out);
        };
      }
      // 如果未匹配到指定的 schema，则记录和转储 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::round, aten_round, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch schema
  if (n->matches(torch::schema("aten::round(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，对 self 执行 round 操作并将结果赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::round(self);
        return;
      }
      // 否则，获取输出张量并执行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出张量 out 中执行 round 操作
      at::cpu::round_out(out, self);
    };
  }

  // 检查节点是否匹配带有 decimals 参数的 torch schema
  if (n->matches(torch::schema(
          "aten::round.decimals(Tensor self, *, int decimals) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self 和 decimals 参数
      const auto& self = p_node->Input(0).toTensor();
      const auto decimals = p_node->Input(1).toInt();
      // 如果输出为空，对 self 执行带有 decimals 参数的 round 操作并将结果赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::round(self, decimals);
        return;
      }
      // 否则，获取输出张量并执行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出张量 out 中执行带有 decimals 参数的 round 操作
      at::cpu::round_out(out, self, decimals);
    };
  }
  // 如果未匹配到指定的 schema，则记录和转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::gelu, aten_gelu, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch schema
  if (n->matches(torch::schema(
          "aten::gelu(Tensor self, *, str approximate='none') -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量 self 和 approximate 参数
      const auto& self = p_node->Input(0).toTensor();
      const auto approximate = p_node->Input(1).toStringView();
      // 如果输出为空，对 self 执行 gelu 操作并将结果赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gelu(self, approximate);
        return;
      }
      // 否则，获取输出张量并执行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 在输出张量 out 中执行 gelu 操作
      at::cpu::gelu_out(out, self, approximate);
    };
  }
  // 如果未匹配到指定的 schema，则记录和转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::gelu_backward,
    aten_gelu_backward,
    // 匿名函数，接受一个 Node* 参数，返回一个 SROperator 类型的对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 Torch 脚本函数签名
      if (n->matches(torch::schema(
              "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor"))) {
        // 如果匹配，则返回一个 Lambda 函数，处理节点 n
        return [](ProcessedNode* p_node) {
          // 获取处理节点的输入参数
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto approximate = p_node->Input(2).toStringView();
          // 如果输出节点为空，则计算并赋值输出节点
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::gelu_backward(grad_output, self, approximate);
            return;
          }
          // 否则，获取并清空输出节点的梯度
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          // 调用函数计算梯度并输出到 grad_input
          at::cpu::gelu_backward_out(
              grad_input, grad_output, self, approximate);
        };
      }
      // 如果节点 n 不匹配指定的 Torch 脚本函数签名，则记录日志并转储节点结构
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::hardshrink,
    aten_hardshrink,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数处理节点操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& self = p_node->Input(0).toTensor();
          const auto lambd = p_node->Input(1).toScalar();
          // 如果输出为空，执行 CPU 上的 hardshrink 操作
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::hardshrink(self, lambd);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零，然后执行 hardshrink 操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::hardshrink_out(out, self, lambd);
        };
      }
      // 如果不符合模式，记录并转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::hardshrink_backward,
    aten_hardshrink_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数处理节点操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& grad_out = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto lambd = p_node->Input(2).toScalar();
          // 如果输出为空，执行 CPU 上的 hardshrink_backward 操作
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::hardshrink_backward(grad_out, self, lambd);
            return;
          }
          // 否则，获取输出梯度输入张量并快速调整大小为零，然后执行 hardshrink_backward 操作
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::hardshrink_backward_out(grad_input, grad_out, self, lambd);
        };
      }
      // 如果不符合模式，记录并转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::rsqrt, aten_rsqrt, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::rsqrt(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::rsqrt(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数处理节点操作
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，执行 CPU 上的 rsqrt 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::rsqrt(self);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零，然后执行 rsqrt 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::rsqrt_out(out, self);
    };
  }
  // 如果不符合模式，记录并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::silu, aten_silu, [](Node* n) -> SROperator {
  // 检查节点是否符合 "aten::silu(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::silu(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数处理节点操作
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，执行 CPU 上的 silu 操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::silu(self);
        return;
      }
      // 否则，获取输出张量并快速调整大小为零，然后执行 silu 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::silu_out(out, self);
    };
  }
  // 如果不符合模式，记录并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::silu_backward,
    aten_silu_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "aten::silu_backward(Tensor grad_out, Tensor self) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::silu_backward(Tensor grad_out, Tensor self) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数处理节点操作
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& grad_out = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          // 如果输出为空，执行 CPU 上的 silu_backward 操作
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::silu_backward(grad_out, self);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零，然后执行 silu_backward 操作
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::silu_backward_out(grad_input, grad_out, self);
        };
      }
      // 如果不符合模式，记录并转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });
    // 匿名函数定义，接受一个指向 Node 结构的指针 n，并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 模式字符串 "aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor"
      if (n->matches(torch::schema(
              "aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor"))) {
        // 如果匹配成功，则返回一个 lambda 函数
        return [](ProcessedNode* p_node) {
          // 获取输入参数 grad_output 和 self
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          // 如果输出参数为 None，则执行计算并存储结果到输出参数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::silu_backward(grad_output, self);
            return;
          }
          // 否则，获取输出参数并快速调整大小
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          // 执行计算并将结果存储到 grad_input 中
          at::cpu::silu_backward_out(grad_input, grad_output, self);
        };
      }
      // 如果未匹配成功，则记录并转储节点 n 的模式信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(aten::mish, aten_mish, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::mish(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::mish(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数作为 Tensor self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数是空的
      if (p_node->Output(0).isNone()) {
        // 将 Tensor self 经过 at::cpu::mish 处理后存入输出节点的第一个参数
        p_node->Output(0) = at::cpu::mish(self);
        return;
      }
      // 否则，获取输出节点的第一个参数作为输出 Tensor out，并快速重设其大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 对 Tensor self 应用 at::cpu::mish，并将结果存入 Tensor out
      at::cpu::mish_out(out, self);
    };
  }
  // 如果节点不匹配，则记录并转储其模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sin, aten_sin, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::sin(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::sin(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数作为 Tensor self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数是空的
      if (p_node->Output(0).isNone()) {
        // 将 Tensor self 经过 at::cpu::sin 处理后存入输出节点的第一个参数
        p_node->Output(0) = at::cpu::sin(self);
        return;
      }
      // 否则，获取输出节点的第一个参数作为输出 Tensor out，并快速重设其大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 对 Tensor self 应用 at::cpu::sin，并将结果存入 Tensor out
      at::cpu::sin_out(out, self);
    };
  }
  // 如果节点不匹配，则记录并转储其模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sinc, aten_sinc, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::sinc(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::sinc(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数作为 Tensor self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数是空的
      if (p_node->Output(0).isNone()) {
        // 将 Tensor self 经过 at::cpu::sinc 处理后存入输出节点的第一个参数
        p_node->Output(0) = at::cpu::sinc(self);
        return;
      }
      // 否则，获取输出节点的第一个参数作为输出 Tensor out，并快速重设其大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 对 Tensor self 应用 at::cpu::sinc，并将结果存入 Tensor out
      at::cpu::sinc_out(out, self);
    };
  }
  // 如果节点不匹配，则记录并转储其模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sinh, aten_sinh, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::sinh(Tensor self) -> Tensor" 的模式
  if (n->matches(torch::schema("aten::sinh(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数作为 Tensor self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数是空的
      if (p_node->Output(0).isNone()) {
        // 将 Tensor self 经过 at::cpu::sinh 处理后存入输出节点的第一个参数
        p_node->Output(0) = at::cpu::sinh(self);
        return;
      }
      // 否则，获取输出节点的第一个参数作为输出 Tensor out，并快速重设其大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 对 Tensor self 应用 at::cpu::sinh，并将结果存入 Tensor out
      at::cpu::sinh_out(out, self);
    };
  }
  // 如果节点不匹配，则记录并转储其模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::_softmax, aten__softmax, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数作为 Tensor self
      const auto& self = p_node->Input(0).toTensor();
      // 获取输入节点的第二个参数作为 int dim
      const auto dim = p_node->Input(1).toInt();
      // 获取输入节点的第三个参数作为 bool half_to_float
      const auto half_to_float = p_node->Input(2).toBool();
      // 如果输出节点的第一个参数是空的
      if (p_node->Output(0).isNone()) {
        // 将 Tensor self 经过 at::cpu::_softmax 处理后存入输出节点的第一个参数
        p_node->Output(0) = at::cpu::_softmax(self, dim, half_to_float);
        return;
      }
      // 否则，获取输出节点的第一个参数作为输出 Tensor out，并快速重设其大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 对 Tensor self 应用 at::cpu::_softmax，并将结果存入 Tensor out
      at::cpu::_softmax_out(out, self, dim, half_to_float);
    };
  }
  // 如果节点不匹配，则记录并转储其模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::_softmax_backward_data,
    aten__softmax_backward_data,
    // 注册用于反向传播 softmax 数据的操作符
    // 匿名函数定义，接受一个指向 Node 类型的指针 n，返回一个 SROperator 类型对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 Torch 脚本模式
      if (n->matches(torch::schema(
              "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"))) {
        // 如果匹配成功，则返回一个 lambda 函数
        return [](ProcessedNode* p_node) {
          // 从 p_node 中获取输入参数，转换为相应的 Torch 张量或数据类型
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          const auto input_dtype = p_node->Input(3).toScalarType();
          // 如果输出结果为空，则调用相应 Torch 函数生成输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_softmax_backward_data(
                grad_output, output, dim, input_dtype);
            return;
          }
          // 否则，获取输出的引用，并使用快速调整大小函数将其调整为零大小
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          // 调用 Torch 的 _softmax_backward_data_out 函数填充 grad_input 张量
          at::cpu::_softmax_backward_data_out(
              grad_input, grad_output, output, dim, input_dtype);
        };
      }
      // 如果节点 n 不匹配指定模式，则记录并转储其模式
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(aten::sqrt, aten_sqrt, [](Node* n) -> SROperator {
  // 检查节点是否匹配 aten::sqrt 的 schema
  if (n->matches(torch::schema("aten::sqrt(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数 self，作为 Tensor 类型
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数为 None，则执行下列操作
      if (p_node->Output(0).isNone()) {
        // 将计算结果赋给输出节点的第一个参数
        p_node->Output(0) = at::cpu::sqrt(self);
        return;
      }
      // 否则，获取输出节点的第一个参数，并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::sqrt_out 函数计算结果并存储到 out 中
      at::cpu::sqrt_out(out, self);
    };
  }
  // 如果未匹配到正确的 schema，则记录日志并转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::square, aten_square, [](Node* n) -> SROperator {
  // 检查节点是否匹配 aten::square 的 schema
  if (n->matches(torch::schema("aten::square(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数 self，作为 Tensor 类型
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数为 None，则执行下列操作
      if (p_node->Output(0).isNone()) {
        // 将计算结果赋给输出节点的第一个参数
        p_node->Output(0) = at::native::square(self);
        return;
      }
      // 否则，获取输出节点的第一个参数，并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::square_out 函数计算结果并存储到 out 中
      at::native::square_out(self, out);
    };
  }
  // 如果未匹配到正确的 schema，则记录日志并转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::prod, aten_prod, [](Node* n) -> SROperator {
  // 检查节点是否匹配 aten::prod 的第一个 schema
  if (n->matches(torch::schema(
          "aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数 self，作为 Tensor 类型
      const auto& self = p_node->Input(0).toTensor();
      // 获取输入节点的第二个参数 dtype，作为可选的 ScalarType 类型
      const auto dtype = p_node->Input(1).toOptional<at::ScalarType>();
      // 如果输出节点的第一个参数为 None，则执行下列操作
      if (p_node->Output(0).isNone()) {
        // 将计算结果赋给输出节点的第一个参数
        p_node->Output(0) = at::native::prod(self, dtype);
        return;
      }
      // 否则，获取输出节点的第一个参数，并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::prod_out 函数计算结果并存储到 out 中
      at::native::prod_out(self, dtype, out);
    };
  }

  // 检查节点是否匹配 aten::prod 的第二个 schema
  if (n->matches(torch::schema(
          "aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数 self，作为 Tensor 类型
      const auto& self = p_node->Input(0).toTensor();
      // 获取输入节点的第二个参数 dim，作为整数类型
      const auto dim = p_node->Input(1).toInt();
      // 获取输入节点的第三个参数 keepdim，作为布尔类型
      const auto keepdim = p_node->Input(2).toBool();
      // 获取输入节点的第四个参数 dtype，作为可选的 ScalarType 类型
      const auto dtype = p_node->Input(3).toOptional<at::ScalarType>();
      // 如果输出节点的第一个参数为 None，则执行下列操作
      if (p_node->Output(0).isNone()) {
        // 将计算结果赋给输出节点的第一个参数
        p_node->Output(0) = at::cpu::prod(self, dim, keepdim, dtype);
        return;
      }
      // 否则，获取输出节点的第一个参数，并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::prod_out 函数计算结果并存储到 out 中
      at::cpu::prod_out(out, self, dim, keepdim, dtype);
    };
  }

  // 如果未匹配到正确的 schema，则记录日志并转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::tan, aten_tan, [](Node* n) -> SROperator {
  // 检查节点是否匹配 aten::tan 的 schema
  if (n->matches(torch::schema("aten::tan(Tensor self) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个参数 self，作为 Tensor 类型
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点的第一个参数为 None，则执行下列操作
      if (p_node->Output(0).isNone()) {
        // 将计算结果赋给输出节点的第一个参数
        p_node->Output(0) = at::cpu::tan(self);
        return;
      }
      // 否则，获取输出节点的第一个参数，并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::tan_out 函数计算结果并存储到 out 中
      at::cpu::tan_out(out, self);
    };
  }
  // 如果未匹配到正确的 schema，则记录日志并转储 schema
  LogAndDumpSchema(n);
  return nullptr;
});
REGISTER_OPERATOR_FUNCTOR(aten::threshold, aten_threshold, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 ATen 函数调用模式
  if (n->matches(torch::schema(
          "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor"))) {
    // 如果匹配成功，则返回一个 lambda 函数处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto threshold = p_node->Input(1).toScalar();
      const auto value = p_node->Input(2).toScalar();
      // 检查输出是否为空，若为空则直接调用 ATen 的 threshold 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::threshold(self, threshold, value);
        return;
      }
      // 否则，调整输出张量的大小并调用 threshold_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::threshold_out(out, self, threshold, value);
    };
  }
  // 若节点不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::threshold_backward,
    aten_threshold_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 ATen 函数调用模式
      if (n->matches(torch::schema(
              "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor"))) {
        // 如果匹配成功，则返回一个 lambda 函数处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto threshold = p_node->Input(2).toScalar();
          // 检查输出是否为空，若为空则直接调用 ATen 的 threshold_backward 函数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::threshold_backward(grad_output, self, threshold);
            return;
          }
          // 否则，调整输出张量的大小并调用 threshold_backward_out 函数
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::threshold_backward_out(
              grad_input, grad_output, self, threshold);
        };
      }
      // 若节点不匹配，则记录日志并转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::trunc, aten_trunc, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 ATen 函数调用模式
  if (n->matches(torch::schema("aten::trunc(Tensor self) -> Tensor"))) {
    // 如果匹配成功，则返回一个 lambda 函数处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      // 检查输出是否为空，若为空则直接调用 ATen 的 trunc 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::trunc(self);
        return;
      }
      // 否则，调整输出张量的大小并调用 trunc_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::trunc_out(out, self);
    };
  }
  // 若节点不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::fix, aten_fix, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 ATen 函数调用模式
  if (n->matches(torch::schema("aten::fix(Tensor self) -> Tensor"))) {
    // 如果匹配成功，则返回一个 lambda 函数处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      // 检查输出是否为空，若为空则直接调用 ATen 的 fix 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::fix(self);
        return;
      }
      // 否则，调整输出张量的大小并调用 fix_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::fix_out(self, out);
    };
  }
  // 若节点不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    [](Node* n) -> SROperator {
        // 匿名函数接收一个指向 Node 类型的指针 n，返回一个 SROperator 对象
        if (n->matches(torch::schema(
                "aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor"))) {
            // 如果节点 n 匹配给定的 Torch 脚本，执行以下操作
            return [](ProcessedNode* p_node) {
                // 返回另一个匿名函数，接收一个指向 ProcessedNode 类型的指针 p_node
                const auto& self = p_node->Input(0).toTensor();
                // 从 p_node 中获取第一个输入，转换为 Tensor 类型，并赋值给 self
                const auto keepdim = p_node->Input(1).toBool();
                // 从 p_node 中获取第二个输入，转换为 bool 类型，并赋值给 keepdim
                if (p_node->Output(0).isNone()) {
                    // 如果 p_node 的第一个输出为空
                    p_node->Output(0) = at::native::nuclear_norm(self, keepdim);
                    // 调用 Torch 的 native 函数计算核范数，将结果赋给 p_node 的第一个输出
                    return;
                }
                auto& out = p_node->Output(0).toTensor();
                // 否则，从 p_node 的第一个输出获取 Tensor，并赋给 out
                fastResizeToZero(out);
                // 调用 fastResizeToZero 函数，快速将 out 的尺寸调整为零
                at::native::nuclear_norm_out(self, keepdim, out);
                // 调用 Torch 的 native 函数计算核范数，并将结果输出到 out
            };
        }
        LogAndDumpSchema(n);
        // 如果节点 n 不匹配条件，则调用 LogAndDumpSchema 函数记录和转储模式信息
        return nullptr;
        // 返回空指针
    });
    // 匿名函数的结束标志，表示函数定义的结束
REGISTER_OPERATOR_FUNCTOR(aten::subtract, aten_subtract, [](Node* n) -> SROperator {
  // 检查节点是否匹配特定的 Torch 脚本
  if (n->matches(torch::schema(
          "aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数执行节点处理操作
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入张量和标量值
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto alpha = p_node->Input(2).toScalar();
      // 如果输出张量为空，执行原生的减法操作
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::subtract(self, other, alpha);
        return;
      }
      // 否则，执行带输出参数的减法操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::subtract_out(self, other, alpha, out);
    };
  }
  // 如果不匹配，记录并输出节点的脚本信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::heaviside,
    aten_heaviside,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::heaviside(Tensor self, Tensor values) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数执行节点处理操作
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量
          const auto& self = p_node->Input(0).toTensor();
          const auto& values = p_node->Input(1).toTensor();
          // 如果输出张量为空，执行 Heaviside 函数操作
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::heaviside(self, values);
            return;
          }
          // 否则，执行带输出参数的 Heaviside 函数操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::heaviside_out(out, self, values);
        };
      }
      // 如果不匹配，记录并输出节点的脚本信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_addmm_activation,
    aten__addmm_activation,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数执行节点处理操作
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量和标量值
          const auto& self = p_node->Input(0).toTensor();
          const auto& mat1 = p_node->Input(1).toTensor();
          const auto& mat2 = p_node->Input(2).toTensor();
          const auto beta = p_node->Input(3).toScalar();
          const auto alpha = p_node->Input(4).toScalar();
          const auto use_gelu = p_node->Input(5).toBool();
          // 如果输出张量为空，执行 _addmm_activation 操作
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_addmm_activation(
                self, mat1, mat2, beta, alpha, use_gelu);
            return;
          }
          // 否则，执行带输出参数的 _addmm_activation 操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_addmm_activation_out(
              out, self, mat1, mat2, beta, alpha, use_gelu);
        };
      }
      // 如果不匹配，记录并输出节点的脚本信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::index_add, aten_index_add, [](Node* n) -> SROperator {
  // 检查节点是否匹配特定的 Torch 脚本
  if (n->matches(torch::schema(
          "aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数接受一个 ProcessedNode 指针作为参数
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个张量
      const auto& self = p_node->Input(0).toTensor();
      // 获取输入节点的第二个整数
      const auto dim = p_node->Input(1).toInt();
      // 获取输入节点的第三个张量
      const auto& index = p_node->Input(2).toTensor();
      // 获取输入节点的第四个张量
      const auto& source = p_node->Input(3).toTensor();
      // 获取输入节点的第五个标量
      const auto alpha = p_node->Input(4).toScalar();
      // 如果输出节点为空
      if (p_node->Output(0).isNone()) {
        // 对 self 进行 index_add 操作，并将结果赋给输出节点
        p_node->Output(0) = at::cpu::index_add(self, dim, index, source, alpha);
        return;
      }
      // 获取输出节点的张量
      auto& out = p_node->Output(0).toTensor();
      // 快速将输出节点的大小调整为零
      fastResizeToZero(out);
      // 对输出节点进行 index_add 操作
      at::cpu::index_add_out(out, self, dim, index, source, alpha);
    };
  }
  // 记录和转储模式 n 的模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::scatter, aten_scatter, [](Node* n) -> SROperator {
  // 如果节点匹配 scatter 操作的 src 版本的模式
  if (n->matches(torch::schema(
          "aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"))) {
    // 返回一个 lambda 函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& src = p_node->Input(3).toTensor();
      // 如果输出为空，直接调用 scatter 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, src);
        return;
      }
      // 否则，调用 scatter_out 函数进行输出的处理
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out); // 快速将输出张量调整大小为零
      at::cpu::scatter_out(out, self, dim, index, src);
    };
  }

  // 如果节点匹配 scatter 操作的 value 版本的模式
  if (n->matches(torch::schema(
          "aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor"))) {
    // 返回一个 lambda 函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto value = p_node->Input(3).toScalar();
      // 如果输出为空，直接调用 scatter 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, value);
        return;
      }
      // 否则，调用 scatter_out 函数进行输出的处理
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out); // 快速将输出张量调整大小为零
      at::cpu::scatter_out(out, self, dim, index, value);
    };
  }

  // 如果节点匹配 scatter 操作的 reduce 版本的模式
  if (n->matches(torch::schema(
          "aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor"))) {
    // 返回一个 lambda 函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& src = p_node->Input(3).toTensor();
      const auto reduce = p_node->Input(4).toStringView();
      // 如果输出为空，直接调用 scatter 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, src, reduce);
        return;
      }
      // 否则，调用 scatter_out 函数进行输出的处理
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out); // 快速将输出张量调整大小为零
      at::cpu::scatter_out(out, self, dim, index, src, reduce);
    };
  }

  // 如果节点匹配 scatter 操作的 value_reduce 版本的模式
  if (n->matches(torch::schema(
          "aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor"))) {
    // 返回一个 lambda 函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto value = p_node->Input(3).toScalar();
      const auto reduce = p_node->Input(4).toStringView();
      // 如果输出为空，直接调用 scatter 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, value, reduce);
        return;
      }
      // 否则，调用 scatter_out 函数进行输出的处理
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out); // 快速将输出张量调整大小为零
      at::cpu::scatter_out(out, self, dim, index, value, reduce);
    };
  }
  // 如果都不匹配，记录和转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
REGISTER_OPERATOR_FUNCTOR(aten::scatter_add, aten_scatter_add, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 scatter_add 模式
  if (n->matches(torch::schema(
          "aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"))) {
    // 返回一个 Lambda 函数，处理匹配成功的节点
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& src = p_node->Input(3).toTensor();
      // 如果输出为 None，直接计算并赋值输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter_add(self, dim, index, src);
        return;
      }
      // 否则，重设输出张量大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行 scatter_add 操作，将结果存入输出张量
      at::cpu::scatter_add_out(out, self, dim, index, src);
    };
  }
  // 如果节点不匹配，记录日志并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::scatter_reduce,
    aten_scatter_reduce,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 scatter_reduce 模式
      if (n->matches(torch::schema(
              "aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"))) {
        // 返回一个 Lambda 函数，处理匹配成功的节点
        return [](ProcessedNode* p_node) {
          // 获取输入参数
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          const auto& index = p_node->Input(2).toTensor();
          const auto& src = p_node->Input(3).toTensor();
          const auto reduce = p_node->Input(4).toStringView();
          const auto include_self = p_node->Input(5).toBool();
          // 如果输出为 None，直接计算并赋值输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::scatter_reduce(
                self, dim, index, src, reduce, include_self);
            return;
          }
          // 否则，重设输出张量大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 执行 scatter_reduce 操作，将结果存入输出张量
          at::cpu::scatter_reduce_out(
              out, self, dim, index, src, reduce, include_self);
        };
      }
      // 如果节点不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::eq, aten_eq, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 eq 模式（标量相等比较）
  if (n->matches(torch::schema(
          "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    // 返回一个 Lambda 函数，处理匹配成功的节点
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      // 如果输出为 None，直接计算并赋值输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::eq(self, other);
        return;
      }
      // 否则，重设输出张量大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行标量相等比较操作，将结果存入输出张量
      at::cpu::eq_out(out, self, other);
    };
  }

  // 检查节点是否匹配给定的 eq 模式（张量相等比较）
  if (n->matches(torch::schema(
          "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 Lambda 函数，处理匹配成功的节点
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出为 None，直接计算并赋值输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::eq(self, other);
        return;
      }
      // 否则，重设输出张量大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行张量相等比较操作，将结果存入输出张量
      at::cpu::eq_out(out, self, other);
    };
  }
  // 调用 LogAndDumpSchema 函数，记录并转储模式 n 的信息
  LogAndDumpSchema(n);
  // 返回空指针，表示未找到匹配的模式
  return nullptr;
});

// 注册操作符函数，处理 torch 中的按位与运算
REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_and,
    aten_bitwise_and,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch schema
      if (n->matches(torch::schema(
              "aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 从输入中获取张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为 None，执行按位与操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_and(self, other);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 执行按位与操作，并将结果存储到输出张量 out 中
          at::cpu::bitwise_and_out(out, self, other);
        };
      }
      // 如果未匹配到指定的 schema，记录并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册操作符函数，处理 torch 中的按位或运算
REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_or,
    aten_bitwise_or,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch schema
      if (n->matches(torch::schema(
              "aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 从输入中获取张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为 None，执行按位或操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_or(self, other);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 执行按位或操作，并将结果存储到输出张量 out 中
          at::cpu::bitwise_or_out(out, self, other);
        };
      }
      // 如果未匹配到指定的 schema，记录并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册操作符函数，处理 torch 中的按位异或运算
REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_xor,
    aten_bitwise_xor,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch schema
      if (n->matches(torch::schema(
              "aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 从输入中获取张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为 None，执行按位异或操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_xor(self, other);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 执行按位异或操作，并将结果存储到输出张量 out 中
          at::cpu::bitwise_xor_out(out, self, other);
        };
      }
      // 如果未匹配到指定的 schema，记录并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册操作符函数，处理 torch 中的按位左移运算
REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_left_shift,
    aten_bitwise_left_shift,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch schema
      if (n->matches(torch::schema(
              "aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 从输入中获取张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为 None，执行按位左移操作并存储到输出中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_left_shift(self, other);
            return;
          }
          // 否则，获取输出张量并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 执行按位左移操作，并将结果存储到输出张量 out 中
          at::cpu::bitwise_left_shift_out(out, self, other);
        };
      }
      // 如果未匹配到指定的 schema，记录并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });



// 结束一个匿名函数的定义，并且这个函数被作为参数传递给某个方法或者回调函数
});  // 结束匿名函数的定义


这段代码片段是 JavaScript 中的语法，表示一个匿名函数定义的结束。通常在 JavaScript 中，函数可以作为参数传递给其他函数或者作为回调函数使用。在这里，`});` 表示匿名函数的结尾，结束了函数的定义。
REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_right_shift,  // 注册自定义运算符函数，处理位右移操作
    aten_bitwise_right_shift,   // 自定义运算符的命名
    [](Node* n) -> SROperator {  // Lambda函数，接收Node指针，返回SROperator对象
      if (n->matches(torch::schema(
              "aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        // 匹配操作符模式，检查是否符合位右移的张量操作模式
        return [](ProcessedNode* p_node) {  // 返回Lambda函数，处理节点数据
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          const auto& other = p_node->Input(1).toTensor();  // 获取输入张量 other
          if (p_node->Output(0).isNone()) {  // 如果输出张量未初始化
            p_node->Output(0) = at::cpu::bitwise_right_shift(self, other);  // 执行位右移操作
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 获取输出张量的引用
          fastResizeToZero(out);  // 快速调整输出张量大小为零
          at::cpu::bitwise_right_shift_out(out, self, other);  // 执行位右移操作，将结果写入 out
        };
      }
      LogAndDumpSchema(n);  // 如果未匹配到预期的操作模式，则记录日志并转储模式信息
      return nullptr;  // 返回空指针表示未注册对应的操作函数
    });

REGISTER_OPERATOR_FUNCTOR(aten::tril, aten_tril, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::tril(Tensor self, int diagonal=0) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto diagonal = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::tril(self, diagonal);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::tril_out(out, self, diagonal);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::triu, aten_triu, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::triu(Tensor self, int diagonal=0) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto diagonal = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::triu(self, diagonal);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::triu_out(out, self, diagonal);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::digamma,
    aten_digamma,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::digamma(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::digamma(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::digamma_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::lerp, aten_lerp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor"))) {
    // 注册自定义运算符函数，处理线性插值操作
    return [](ProcessedNode* p_node) {  // Lambda函数，处理节点数据
      const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
      const auto& end = p_node->Input(1).toTensor();   // 获取输入张量 end
      const auto weight = p_node->Input(2).toScalar();  // 获取标量权重值
      if (p_node->Output(0).isNone()) {  // 如果输出张量未初始化
        p_node->Output(0) = at::lerp(self, end, weight);  // 执行线性插值操作
        return;
      }
      auto& out = p_node->Output(0).toTensor();  // 获取输出张量的引用
      fastResizeToZero(out);  // 快速调整输出张量大小为零
      at::lerp_out(out, self, end, weight);  // 执行线性插值操作，将结果写入 out
    };
  }
  LogAndDumpSchema(n);  // 如果未匹配到预期的操作模式，则记录日志并转储模式信息
  return nullptr;  // 返回空指针表示未注册对应的操作函数
});
    // 返回一个 Lambda 函数，该函数用于处理不同的节点类型，这里处理 "aten::lerp.Scalar" 的情况
    return [](ProcessedNode* p_node) {
      // 获取节点的输入参数，分别为 self, end 和 weight
      const auto& self = p_node->Input(0).toTensor();
      const auto& end = p_node->Input(1).toTensor();
      const auto weight = p_node->Input(2).toScalar();
      // 如果节点的输出为空，执行以下操作
      if (p_node->Output(0).isNone()) {
        // 对输出进行线性插值，并将结果赋值给节点的输出
        p_node->Output(0) = at::cpu::lerp(self, end, weight);
        return;
      }
      // 否则，获取输出的引用，并将其尺寸调整为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行线性插值操作，并将结果存入输出张量 out 中
      at::cpu::lerp_out(out, self, end, weight);
    };
  }

  // 如果节点匹配 "aten::lerp.Tensor" 的模式，则执行以下操作
  if (n->matches(torch::schema(
          "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor"))) {
    // 返回一个 Lambda 函数，用于处理 "aten::lerp.Tensor" 的节点
    return [](ProcessedNode* p_node) {
      // 获取节点的输入参数，分别为 self, end 和 weight
      const auto& self = p_node->Input(0).toTensor();
      const auto& end = p_node->Input(1).toTensor();
      const auto& weight = p_node->Input(2).toTensor();
      // 如果节点的输出为空，执行以下操作
      if (p_node->Output(0).isNone()) {
        // 对输出进行线性插值，并将结果赋值给节点的输出
        p_node->Output(0) = at::cpu::lerp(self, end, weight);
        return;
      }
      // 否则，获取输出的引用，并将其尺寸调整为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 执行线性插值操作，并将结果存入输出张量 out 中
      at::cpu::lerp_out(out, self, end, weight);
    };
  }
  // 如果节点类型不匹配任何已知的模式，则记录并转储该节点的模式信息
  LogAndDumpSchema(n);
  // 返回空指针，表示未能处理该节点类型
  return nullptr;
});

// 注册对 aten::addbmm 运算符的处理函数
REGISTER_OPERATOR_FUNCTOR(aten::addbmm, aten_addbmm, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合给定的 torch::schema
  if (n->matches(torch::schema(
          "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理给定的节点 p_node
    return [](ProcessedNode* p_node) {
      // 从输入中获取相应的张量和标量
      const auto& self = p_node->Input(0).toTensor();
      const auto& batch1 = p_node->Input(1).toTensor();
      const auto& batch2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      // 如果输出张量未定义，则进行计算并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::addbmm(self, batch1, batch2, beta, alpha);
        return;
      }
      // 否则，获取输出张量并快速重设大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用原生的 addbmm_out 函数，计算并填充输出张量
      at::native::addbmm_out(self, batch1, batch2, beta, alpha, out);
    };
  }
  // 若不符合条件，记录并输出节点的 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册对 aten::diag 运算符的处理函数
REGISTER_OPERATOR_FUNCTOR(aten::diag, aten_diag, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合给定的 torch::schema
  if (n->matches(
          torch::schema("aten::diag(Tensor self, int diagonal=0) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理给定的节点 p_node
    return [](ProcessedNode* p_node) {
      // 从输入中获取张量和整数参数
      const auto& self = p_node->Input(0).toTensor();
      const auto diagonal = p_node->Input(1).toInt();
      // 如果输出张量未定义，则进行计算并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::diag(self, diagonal);
        return;
      }
      // 否则，获取输出张量并快速重设大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用原生的 diag_out 函数，计算并填充输出张量
      at::native::diag_out(self, diagonal, out);
    };
  }
  // 若不符合条件，记录并输出节点的 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册对 aten::cross 运算符的处理函数
REGISTER_OPERATOR_FUNCTOR(aten::cross, aten_cross, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合给定的 torch::schema
  if (n->matches(torch::schema(
          "aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理给定的节点 p_node
    return [](ProcessedNode* p_node) {
      // 从输入中获取两个张量和可能为空的整数参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto dim = p_node->Input(2).toOptional<int64_t>();
      // 如果输出张量未定义，则进行计算并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::cross(self, other, dim);
        return;
      }
      // 否则，获取输出张量并快速重设大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用原生的 cross_out 函数，计算并填充输出张量
      at::native::cross_out(self, other, dim, out);
    };
  }
  // 若不符合条件，记录并输出节点的 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册对 aten::ne 运算符的处理函数
REGISTER_OPERATOR_FUNCTOR(aten::ne, aten_ne, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合给定的 torch::schema，处理 Scalar 类型的 ne
  if (n->matches(torch::schema(
          "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理给定的节点 p_node
    return [](ProcessedNode* p_node) {
      // 从输入中获取张量和标量参数
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      // 如果输出张量未定义，则进行计算并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ne(self, other);
        return;
      }
      // 否则，获取输出张量并快速重设大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 CPU 上的 ne_out 函数，计算并填充输出张量
      at::cpu::ne_out(out, self, other);
    };
  }

  // 检查节点 n 是否符合给定的 torch::schema，处理 Tensor 类型的 ne
  if (n->matches(torch::schema(
          "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 表达式，处理给定的节点 p_node
    return [](ProcessedNode* p_node) {
      // 从输入中获取两个张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量未定义，则进行计算并设置输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::ne(self, other);
        return;
      }
      // 否则，获取输出张量并快速重设大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 ne_out 函数，计算并填充输出张量
      at::ne_out(out, self, other);
    };
  }

  // 若不符合任何条件，记录并输出节点的 schema 信息
  LogAndDumpSchema(n);
  return nullptr;
});
    // 返回一个 lambda 表达式，该 lambda 接受一个 ProcessedNode 指针作为参数
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个和第二个张量引用
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将张量 self 与 other 的逐元素不等操作结果存入输出节点的第一个输出
        p_node->Output(0) = at::cpu::ne(self, other);
        return;  // 结束 lambda 表达式的执行
      }
      // 否则，获取输出节点的第一个输出作为 out 引用
      auto& out = p_node->Output(0).toTensor();
      // 将 out 张量快速调整大小为零
      fastResizeToZero(out);
      // 执行张量的逐元素不等操作，并将结果存入 out 张量
      at::cpu::ne_out(out, self, other);
    };
  }
  // 记录并输出节点 n 的模式信息
  LogAndDumpSchema(n);
  // 返回空指针，结束函数
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::ge, aten_ge, [](Node* n) -> SROperator {
  // 检查节点是否匹配 "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入的张量和标量
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      // 如果输出张量为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ge(self, other);
        return;
      }
      // 否则，快速调整输出张量大小为零，并计算结果存入输出
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ge_out(out, self, other);
    };
  }

  // 检查节点是否匹配 "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入的张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ge(self, other);
        return;
      }
      // 否则，快速调整输出张量大小为零，并计算结果存入输出
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ge_out(out, self, other);
    };
  }
  // 如果以上条件都不满足，则记录并转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::le, aten_le, [](Node* n) -> SROperator {
  // 检查节点是否匹配 "aten::le.Scalar(Tensor self, Scalar other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::le.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入的张量和标量
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      // 如果输出张量为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::le(self, other);
        return;
      }
      // 否则，快速调整输出张量大小为零，并计算结果存入输出
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::le_out(out, self, other);
    };
  }

  // 检查节点是否匹配 "aten::le.Tensor(Tensor self, Tensor other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::le.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入的张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::le(self, other);
        return;
      }
      // 否则，快速调整输出张量大小为零，并计算结果存入输出
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::le_out(out, self, other);
    };
  }
  // 如果以上条件都不满足，则记录并转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::gt, aten_gt, [](Node* n) -> SROperator {
  // 检查节点是否匹配 "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入的张量和标量
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      // 如果输出张量为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gt(self, other);
        return;
      }
      // 否则，快速调整输出张量大小为零，并计算结果存入输出
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gt_out(out, self, other);
    };
  }

  // 检查节点是否匹配 "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入的张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gt(self, other);
        return;
      }
      // 否则，快速调整输出张量大小为零，并计算结果存入输出
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gt_out(out, self, other);
    };
  }
  // 如果以上条件都不满足，则记录并转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    // 返回一个 lambda 函数，该函数接受一个 ProcessedNode 指针参数 p_node
    return [](ProcessedNode* p_node) {
      // 获取第一个输入张量 self 和第二个输入张量 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      
      // 如果输出端口 0 是空的，则进行操作
      if (p_node->Output(0).isNone()) {
        // 将张量 self 和 other 进行大于运算，并存入输出端口 0
        p_node->Output(0) = at::cpu::gt(self, other);
        return;  // 返回结束 lambda 函数执行
      }
      
      // 否则，获取输出端口 0 的张量引用，并快速调整其大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      
      // 将大于运算的结果存入输出端口 0 的张量 out
      at::cpu::gt_out(out, self, other);
    };
  }
  // 记录和转储节点 n 的模式信息
  LogAndDumpSchema(n);
  // 返回空指针，结束函数
  return nullptr;
REGISTER_OPERATOR_FUNCTOR(aten::lt, aten_lt, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    // 返回一个 lambda 函数处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入的张量和标量
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      // 如果输出为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lt(self, other);
        return;
      }
      // 否则，重置输出张量并计算结果
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lt_out(out, self, other);
    };
  }

  // 如果节点匹配 "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入的两个张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出为空，直接计算结果并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lt(self, other);
        return;
      }
      // 否则，重置输出张量并计算结果
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lt_out(out, self, other);
    };
  }
  // 若无匹配模式，记录并转储节点模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::take, aten_take, [](Node* n) -> SROperator {
  // 如果节点匹配 "aten::take(Tensor self, Tensor index) -> Tensor" 的模式
  if (n->matches(
          torch::schema("aten::take(Tensor self, Tensor index) -> Tensor"))) {
    // 返回一个 lambda 函数处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入的张量和索引张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& index = p_node->Input(1).toTensor();
      // 如果输出为空，直接执行 take 操作并存入输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::take(self, index);
        return;
      }
      // 否则，重置输出张量并执行 take 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::take_out(self, index, out);
    };
  }
  // 若无匹配模式，记录并转储节点模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::take_along_dim,
    aten_take_along_dim,
    [](Node* n) -> SROperator {
      // 如果节点匹配 "aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor"))) {
        // 返回一个 lambda 函数处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入的张量、索引张量和维度
          const auto& self = p_node->Input(0).toTensor();
          const auto& indices = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toOptional<int64_t>();
          // 如果输出为空，直接执行 take_along_dim 操作并存入输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::take_along_dim(self, indices, dim);
            return;
          }
          // 否则，重置输出张量并执行 take_along_dim 操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::take_along_dim_out(self, indices, dim, out);
        };
      }
      // 若无匹配模式，记录并转储节点模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::masked_select,
    aten_masked_select,
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个 Node* 参数 n，返回一个 SROperator 对象
      if (n->matches(torch::schema(
              "aten::masked_select(Tensor self, Tensor mask) -> Tensor"))) {
        // 如果节点 n 匹配指定的 torch schema
        return [](ProcessedNode* p_node) {
          // 返回一个 lambda 函数，处理 ProcessedNode* 参数 p_node
          const auto& self = p_node->Input(0).toTensor();
          // 获取第一个输入张量 self
          const auto& mask = p_node->Input(1).toTensor();
          // 获取第二个输入张量 mask
          if (p_node->Output(0).isNone()) {
            // 如果输出节点的第一个输出为空
            p_node->Output(0) = at::native::masked_select_cpu(self, mask);
            // 则将输出节点的第一个输出设置为 masked_select_cpu 函数的结果
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则获取输出节点的第一个输出张量 out
          fastResizeToZero(out);
          // 调用 fastResizeToZero 函数，对输出张量 out 进行快速调整
          at::native::masked_select_out_cpu(self, mask, out);
          // 调用 masked_select_out_cpu 函数，将 masked_select 的结果写入 out
        };
      }
      LogAndDumpSchema(n);
      // 如果节点 n 不匹配指定的 schema，则调用 LogAndDumpSchema 函数记录和转储 schema 信息
      return nullptr;
      // 返回空指针
    });
    // 匿名函数结束
REGISTER_OPERATOR_FUNCTOR(
    aten::nonzero_static,  // 注册 aten::nonzero_static 运算符到对应的处理函数 aten_nonzero_static
    aten_nonzero_static,   // 实现一个 Lambda 函数，接受 Node* 参数，返回 SROperator 结果
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::nonzero_static(Tensor self, *, int size, int fill_value=-1) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();  // 获取输入的 Tensor self
          const auto size = p_node->Input(1).toInt();      // 获取输入的整数 size
          const auto fill_value = p_node->Input(2).toInt();  // 获取输入的整数 fill_value
          if (p_node->Output(0).isNone()) {  // 如果输出为 None，则执行非零静态 CPU 计算
            p_node->Output(0) =
                at::native::nonzero_static_cpu(self, size, fill_value);  // 调用非零静态 CPU 计算函数
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 获取输出 Tensor
          fastResizeToZero(out);  // 快速调整输出 Tensor 的大小为零
          at::native::nonzero_static_out_cpu(self, size, fill_value, out);  // 使用输出 Tensor 执行非零静态 CPU 计算
        };
      }
      LogAndDumpSchema(n);  // 若不匹配预期的 schema，则记录和转储模式
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::gather,  // 注册 aten::gather 运算符到对应的处理函数 aten_gather
    aten_gather,   // 实现一个 Lambda 函数，接受 Node* 参数，返回 SROperator 结果
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();  // 获取输入的 Tensor self
          const auto dim = p_node->Input(1).toInt();       // 获取输入的整数 dim
          const auto& index = p_node->Input(2).toTensor();  // 获取输入的 Tensor index
          const auto sparse_grad = p_node->Input(3).toBool();  // 获取输入的布尔值 sparse_grad
          if (p_node->Output(0).isNone()) {  // 如果输出为 None，则执行 gather 操作
            p_node->Output(0) = at::cpu::gather(self, dim, index, sparse_grad);  // 执行 gather 操作
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 获取输出 Tensor
          fastResizeToZero(out);  // 快速调整输出 Tensor 的大小为零
          at::cpu::gather_out(out, self, dim, index, sparse_grad);  // 使用输出 Tensor 执行 gather 操作
        };
      }
      LogAndDumpSchema(n);  // 若不匹配预期的 schema，则记录和转储模式
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::addcmul,  // 注册 aten::addcmul 运算符到对应的处理函数 aten_addcmul
    aten_addcmul,   // 实现一个 Lambda 函数，接受 Node* 参数，返回 SROperator 结果
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();    // 获取输入的 Tensor self
          const auto& tensor1 = p_node->Input(1).toTensor();  // 获取输入的 Tensor tensor1
          const auto& tensor2 = p_node->Input(2).toTensor();  // 获取输入的 Tensor tensor2
          const auto value = p_node->Input(3).toScalar();     // 获取输入的标量值 value
          if (p_node->Output(0).isNone()) {  // 如果输出为 None，则执行 addcmul 操作
            p_node->Output(0) = at::cpu::addcmul(self, tensor1, tensor2, value);  // 执行 addcmul 操作
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 获取输出 Tensor
          fastResizeToZero(out);  // 快速调整输出 Tensor 的大小为零
          at::cpu::addcmul_out(out, self, tensor1, tensor2, value);  // 使用输出 Tensor 执行 addcmul 操作
        };
      }
      LogAndDumpSchema(n);  // 若不匹配预期的 schema，则记录和转储模式
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::addcdiv,  // 注册 aten::addcdiv 运算符到对应的处理函数 aten_addcdiv
    aten_addcdiv,   // 实现一个 Lambda 函数，接受 Node* 参数，返回 SROperator 结果
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 匿名函数，接受一个指向ProcessedNode对象的指针p_node作为参数

      // 从p_node的输入端口0获取张量self
      const auto& self = p_node->Input(0).toTensor();
      
      // 从p_node的输入端口1获取张量tensor1
      const auto& tensor1 = p_node->Input(1).toTensor();
      
      // 从p_node的输入端口2获取张量tensor2
      const auto& tensor2 = p_node->Input(2).toTensor();
      
      // 从p_node的输入端口3获取标量值value
      const auto value = p_node->Input(3).toScalar();
      
      // 如果p_node的输出端口0为空
      if (p_node->Output(0).isNone()) {
        // 将self、tensor1、tensor2和value传递给at::cpu::addcdiv函数，并将结果存入p_node的输出端口0
        p_node->Output(0) = at::cpu::addcdiv(self, tensor1, tensor2, value);
        return; // 函数结束
      }
      
      // 否则，从p_node的输出端口0获取张量out
      auto& out = p_node->Output(0).toTensor();
      
      // 调用fastResizeToZero函数，将out快速调整大小为零
      fastResizeToZero(out);
      
      // 使用at::cpu::addcdiv_out函数将self、tensor1、tensor2、value作为参数，将计算结果存入out
      at::cpu::addcdiv_out(out, self, tensor1, tensor2, value);
    };
  }
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    // 注册操作符的函数对象，处理 torch 中的 linalg_solve_triangular 操作
    aten::linalg_solve_triangular,
    aten_linalg_solve_triangular,
    [](Node* n) -> SROperator {
      // 如果当前节点匹配 linalg_solve_triangular 的 schema
      if (n->matches(torch::schema(
              "aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor"))) {
        // 返回一个函数对象，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量和参数
          const auto& self = p_node->Input(0).toTensor();
          const auto& B = p_node->Input(1).toTensor();
          const auto upper = p_node->Input(2).toBool();
          const auto left = p_node->Input(3).toBool();
          const auto unitriangular = p_node->Input(4).toBool();
          // 如果输出张量为空，直接计算结果并返回
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_solve_triangular(
                self, B, upper, left, unitriangular);
            return;
          }
          // 否则，调用函数计算结果并存储到输出张量
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_solve_triangular_out(
              self, B, upper, left, unitriangular, out);
        };
      }
      // 若不匹配，记录日志并转储 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    // 注册操作符的函数对象，处理 torch 中的 cholesky_solve 操作
    aten::cholesky_solve,
    aten_cholesky_solve,
    [](Node* n) -> SROperator {
      // 如果当前节点匹配 cholesky_solve 的 schema
      if (n->matches(torch::schema(
              "aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor"))) {
        // 返回一个函数对象，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量和参数
          const auto& self = p_node->Input(0).toTensor();
          const auto& input2 = p_node->Input(1).toTensor();
          const auto upper = p_node->Input(2).toBool();
          // 如果输出张量为空，直接计算结果并返回
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::cholesky_solve(self, input2, upper);
            return;
          }
          // 否则，调用函数计算结果并存储到输出张量
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::cholesky_solve_out(self, input2, upper, out);
        };
      }
      // 若不匹配，记录日志并转储 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    // 注册操作符的函数对象，处理 torch 中的 cholesky_inverse 操作
    aten::cholesky_inverse,
    aten_cholesky_inverse,
    [](Node* n) -> SROperator {
      // 如果当前节点匹配 cholesky_inverse 的 schema
      if (n->matches(torch::schema(
              "aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor"))) {
        // 返回一个函数对象，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          // 获取输入张量和参数
          const auto& self = p_node->Input(0).toTensor();
          const auto upper = p_node->Input(1).toBool();
          // 如果输出张量为空，直接计算结果并返回
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::cholesky_inverse(self, upper);
            return;
          }
          // 否则，调用函数计算结果并存储到输出张量
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::cholesky_inverse_out(self, upper, out);
        };
      }
      // 若不匹配，记录日志并转储 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    // 注册操作符的函数对象，处理 torch 中的 orgqr 操作
    aten::orgqr, aten_orgqr, [](Node* n) -> SROperator {
  // 如果当前节点匹配 orgqr 的 schema
  if (n->matches(
          torch::schema("aten::orgqr(Tensor self, Tensor input2) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数接受一个 ProcessedNode 指针参数 p_node
    return [](ProcessedNode* p_node) {
      // 获取 p_node 的第一个输入张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 获取 p_node 的第二个输入张量 input2
      const auto& input2 = p_node->Input(1).toTensor();
      // 如果 p_node 的第一个输出未定义
      if (p_node->Output(0).isNone()) {
        // 对 self 和 input2 执行 orgqr 操作，并将结果赋给 p_node 的第一个输出
        p_node->Output(0) = at::native::orgqr(self, input2);
        // 函数返回
        return;
      }
      // 否则，获取 p_node 的第一个输出张量 out
      auto& out = p_node->Output(0).toTensor();
      // 快速将 out 调整为零大小
      fastResizeToZero(out);
      // 执行 orgqr 操作，并将结果写入 out
      at::native::orgqr_out(self, input2, out);
    };
  }
  // 记录并转储模式 n 的模式和结构信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

// 注册名为 "aten::ormqr" 的运算符处理函数
REGISTER_OPERATOR_FUNCTOR(aten::ormqr, aten_ormqr, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch::schema
  if (n->matches(torch::schema(
          "aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor"))) {
    // 返回处理函数，该函数接收一个 ProcessedNode 指针 p_node
    return [](ProcessedNode* p_node) {
      // 从 p_node 中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& input2 = p_node->Input(1).toTensor();
      const auto& input3 = p_node->Input(2).toTensor();
      const auto left = p_node->Input(3).toBool();
      const auto transpose = p_node->Input(4).toBool();
      
      // 如果输出为空，则调用 at::native::ormqr 函数计算结果并赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::ormqr(self, input2, input3, left, transpose);
        return;
      }
      
      // 否则，获取输出张量的引用，并调用 fastResizeToZero 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::ormqr_out 函数计算结果并将其存入 out 张量
      at::native::ormqr_out(self, input2, input3, left, transpose, out);
    };
  }
  
  // 如果不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册名为 "aten::lgamma" 的运算符处理函数
REGISTER_OPERATOR_FUNCTOR(aten::lgamma, aten_lgamma, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch::schema
  if (n->matches(torch::schema("aten::lgamma(Tensor self) -> Tensor"))) {
    // 返回处理函数，该函数接收一个 ProcessedNode 指针 p_node
    return [](ProcessedNode* p_node) {
      // 从 p_node 中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      
      // 如果输出为空，则调用 at::cpu::lgamma 函数计算结果并赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lgamma(self);
        return;
      }
      
      // 否则，获取输出张量的引用，并调用 fastResizeToZero 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::lgamma_out 函数计算结果并将其存入 out 张量
      at::cpu::lgamma_out(out, self);
    };
  }
  
  // 如果不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册名为 "aten::polygamma" 的运算符处理函数
REGISTER_OPERATOR_FUNCTOR(
    aten::polygamma,
    aten_polygamma,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch::schema
      if (n->matches(
              torch::schema("aten::polygamma(int n, Tensor self) -> Tensor"))) {
        // 返回处理函数，该函数接收一个 ProcessedNode 指针 p_node
        return [](ProcessedNode* p_node) {
          // 从 p_node 中获取输入参数
          const auto n = p_node->Input(0).toInt();
          const auto& self = p_node->Input(1).toTensor();
          
          // 如果输出为空，则调用 at::cpu::polygamma 函数计算结果并赋给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::polygamma(n, self);
            return;
          }
          
          // 否则，获取输出张量的引用，并调用 fastResizeToZero 函数
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::polygamma_out 函数计算结果并将其存入 out 张量
          at::cpu::polygamma_out(out, n, self);
        };
      }
      
      // 如果不匹配，则记录日志并转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册名为 "aten::erfinv" 的运算符处理函数
REGISTER_OPERATOR_FUNCTOR(aten::erfinv, aten_erfinv, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch::schema
  if (n->matches(torch::schema("aten::erfinv(Tensor self) -> Tensor"))) {
    // 返回处理函数，该函数接收一个 ProcessedNode 指针 p_node
    return [](ProcessedNode* p_node) {
      // 从 p_node 中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      
      // 如果输出为空，则调用 at::cpu::erfinv 函数计算结果并赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erfinv(self);
        return;
      }
      
      // 否则，获取输出张量的引用，并调用 fastResizeToZero 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::cpu::erfinv_out 函数计算结果并将其存入 out 张量
      at::cpu::erfinv_out(out, self);
    };
  }
  
  // 如果不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册名为 "aten::i0" 的运算符处理函数
REGISTER_OPERATOR_FUNCTOR(aten::i0, aten_i0, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch::schema
  if (n->matches(torch::schema("aten::i0(Tensor self) -> Tensor"))) {
    // 返回处理函数，该函数接收一个 ProcessedNode 指针 p_node
    return [](ProcessedNode* p_node) {
      // 从 p_node 中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      
      // 省略部分代码，具体与前面类似
      
      // 如果输出为空，则调用相应函数计算结果并赋给输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::i0(self);
        return;
      }
      
      // 否则，获取输出张量的引用，并调用 fastResizeToZero 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用相应的 out 函数计算结果并将其存入 out 张量
      at::cpu::i0_out(out, self);
    };
  }
  
  // 如果不匹配，则记录日志并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    // 返回一个 lambda 表达式，该 lambda 接受一个指向 ProcessedNode 的指针 p_node 参数
    return [](ProcessedNode* p_node) {
      // 获取 p_node 的第一个输入张量 self 的引用
      const auto& self = p_node->Input(0).toTensor();
      // 如果 p_node 的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 将 self 的 i0 运算结果赋值给 p_node 的第一个输出
        p_node->Output(0) = at::cpu::i0(self);
        // 结束 lambda 函数执行
        return;
      }
      // 获取 p_node 的第一个输出张量 out 的引用
      auto& out = p_node->Output(0).toTensor();
      // 将 out 快速调整大小为零
      fastResizeToZero(out);
      // 将 self 的 i0 运算结果输出到 out 张量
      at::cpu::i0_out(out, self);
    };
  }
  // 记录并转储节点 n 的模式信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::signbit,
    aten_signbit,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的schema，如果匹配则执行以下操作
      if (n->matches(torch::schema("aten::signbit(Tensor self) -> Tensor"))) {
        // 返回一个lambda函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入张量self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出未初始化，则计算signbit并将结果存储到输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::signbit(self);
            return;
          }
          // 否则，获取输出张量并调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用at::cpu::signbit_out函数，将计算结果存储到输出张量out
          at::cpu::signbit_out(out, self);
        };
      }
      // 如果schema不匹配，则记录并转储schema信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::atan2, aten_atan2, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的schema，如果匹配则执行以下操作
  if (n->matches(
          torch::schema("aten::atan2(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个lambda函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量self和other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出未初始化，则计算atan2并将结果存储到输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::atan2(self, other);
        return;
      }
      // 否则，获取输出张量并调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用at::cpu::atan2_out函数，将计算结果存储到输出张量out
      at::cpu::atan2_out(out, self, other);
    };
  }
  // 如果schema不匹配，则记录并转储schema信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::arctan2,
    aten_arctan2,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的schema，如果匹配则执行以下操作
      if (n->matches(torch::schema(
              "aten::arctan2(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个lambda函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入张量self和other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出未初始化，则计算arctan2并将结果存储到输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arctan2(self, other);
            return;
          }
          // 否则，获取输出张量并调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用at::native::arctan2_out函数，将计算结果存储到输出张量out
          at::native::arctan2_out(self, other, out);
        };
      }
      // 如果schema不匹配，则记录并转储schema信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::histc, aten_histc, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的schema，如果匹配则执行以下操作
  if (n->matches(torch::schema(
          "aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor"))) {
    // 返回一个lambda函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量self、bins、min和max
      const auto& self = p_node->Input(0).toTensor();
      const auto bins = p_node->Input(1).toInt();
      const auto min = p_node->Input(2).toScalar();
      const auto max = p_node->Input(3).toScalar();
      // 如果输出未初始化，则计算histc并将结果存储到输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::histogram_histc(self, bins, min, max);
        return;
      }
      // 否则，获取输出张量并调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用at::native::histogram_histc_out函数，将计算结果存储到输出张量out
      at::native::histogram_histc_out(self, bins, min, max, out);
    };
  }
  // 如果schema不匹配，则记录并转储schema信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::hypot, aten_hypot, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的schema，如果匹配则执行以下操作
  if (n->matches(
          torch::schema("aten::hypot(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个lambda函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量self和other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出未初始化，则计算hypot并将结果存储到输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::hypot(self, other);
        return;
      }
      // 否则，获取输出张量并调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用at::hypot_out函数，将计算结果存储到输出张量out
      at::hypot_out(out, self, other);
    };
  }
  // 如果schema不匹配，则记录并转储schema信息
  LogAndDumpSchema(n);
  return nullptr;
});
    // 返回一个 lambda 函数，该函数接收一个 ProcessedNode 指针参数 p_node
    return [](ProcessedNode* p_node) {
      // 获取第一个输入张量 self 和第二个输入张量 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量不存在
      if (p_node->Output(0).isNone()) {
        // 计算 self 和 other 的 hypot（直角三角形的斜边长度），并赋值给输出张量
        p_node->Output(0) = at::cpu::hypot(self, other);
        return; // lambda 函数执行完毕，返回
      }
      // 否则，获取输出张量的引用
      auto& out = p_node->Output(0).toTensor();
      // 将输出张量快速调整为零大小
      fastResizeToZero(out);
      // 使用 at::cpu::hypot_out 函数计算 hypot 结果，并存储到输出张量 out 中
      at::cpu::hypot_out(out, self, other);
    };
  }
  // 记录和转储 Schema 信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::igamma, aten_igamma, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 aten::igamma 模式
  if (n->matches(
          torch::schema("aten::igamma(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量未定义，计算并存储 igamma 的结果
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::igamma(self, other);
        return;
      }
      // 否则，重置输出张量为零并计算 igamma 的结果
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::igamma_out(out, self, other);
    };
  }
  // 如果节点不匹配预期的模式，则记录和转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::igammac,
    aten_igammac,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::igammac 模式
      if (n->matches(torch::schema(
              "aten::igammac(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量未定义，计算并存储 igammac 的结果
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::igammac(self, other);
            return;
          }
          // 否则，重置输出张量为零并计算 igammac 的结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::igammac_out(out, self, other);
        };
      }
      // 如果节点不匹配预期的模式，则记录和转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::nextafter,
    aten_nextafter,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::nextafter 模式
      if (n->matches(torch::schema(
              "aten::nextafter(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理节点的操作
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量未定义，计算并存储 nextafter 的结果
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::nextafter(self, other);
            return;
          }
          // 否则，重置输出张量为零并计算 nextafter 的结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::nextafter_out(out, self, other);
        };
      }
      // 如果节点不匹配预期的模式，则记录和转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::fmin, aten_fmin, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 aten::fmin 模式
  if (n->matches(
          torch::schema("aten::fmin(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理节点的操作
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量未定义，计算并存储 fmin 的结果
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::fmin(self, other);
        return;
      }
      // 否则，重置输出张量为零并计算 fmin 的结果
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::fmin_out(out, self, other);
    };
  }
  // 如果节点不匹配预期的模式，则记录和转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::fmax, aten_fmax, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 aten::fmax 模式
  if (n->matches(
          torch::schema("aten::fmax(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数接受一个 ProcessedNode* 类型的参数 p_node
    return [](ProcessedNode* p_node) {
      // 获取第一个输入张量 self 和第二个输入张量 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      
      // 如果输出张量 p_node->Output(0) 为空
      if (p_node->Output(0).isNone()) {
        // 将 self 和 other 中的每个元素逐个比较，将结果存储到 p_node->Output(0)
        p_node->Output(0) = at::cpu::fmax(self, other);
        return;
      }
      
      // 否则，获取输出张量 out 并快速调整大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      
      // 使用 fmax_out 函数将 self 和 other 中的每个元素逐个比较，并将结果存储到 out 中
      at::cpu::fmax_out(out, self, other);
    };
  }
  // 记录和转储模式 n 的日志
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::maximum,
    aten_maximum,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 "aten::maximum(Tensor self, Tensor other) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::maximum(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理输入节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入的张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，直接计算并赋值给输出张量
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::maximum(self, other);
            return;
          }
          // 否则，获取输出张量并进行快速调整大小
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::maximum_out 函数计算并存储结果到输出张量
          at::cpu::maximum_out(out, self, other);
        };
      }
      // 如果模式不匹配，则记录和输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::minimum,
    aten_minimum,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 "aten::minimum(Tensor self, Tensor other) -> Tensor" 的模式
      if (n->matches(torch::schema(
              "aten::minimum(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 lambda 函数，处理输入节点的操作
        return [](ProcessedNode* p_node) {
          // 获取输入的张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，直接计算并赋值给输出张量
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::minimum(self, other);
            return;
          }
          // 否则，获取输出张量并进行快速调整大小
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::cpu::minimum_out 函数计算并存储结果到输出张量
          at::cpu::minimum_out(out, self, other);
        };
      }
      // 如果模式不匹配，则记录和输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::min, aten_min, [](Node* n) -> SROperator {
  // 检查节点是否匹配 "aten::min.other(Tensor self, Tensor other) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::min.other(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，处理输入节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出张量为空，直接计算并赋值给输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::min(self, other);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::min_out 函数计算并存储结果到输出张量
      at::native::min_out(self, other, out);
    };
  }
  // 如果模式不匹配，则记录和输出节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::quantile, aten_quantile, [](Node* n) -> SROperator {
  // 检查节点是否匹配 "aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor"))) {
    // 返回一个 lambda 函数，处理输入节点的操作
    return [](ProcessedNode* p_node) {
      // 获取输入的张量 self 和 q，以及可选的 dim，keepdim 和 interpolation 参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& q = p_node->Input(1).toTensor();
      const auto dim = p_node->Input(2).toOptional<int64_t>();
      const auto keepdim = p_node->Input(3).toBool();
      const auto interpolation = p_node->Input(4).toStringView();
      // 如果输出张量为空，直接计算并赋值给输出张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::quantile(self, q, dim, keepdim, interpolation);
        return;
      }
      // 否则，获取输出张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 at::native::quantile_out 函数计算并存储结果到输出张量
      at::native::quantile_out(self, q, dim, keepdim, interpolation, out);

    };
  }
  // 如果模式不匹配，则记录和输出节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    };
  }
  // 记录和输出模式信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
// 注册 aten::nanquantile 操作的函数实现
REGISTER_OPERATOR_FUNCTOR(aten::nanquantile, aten_nanquantile, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 schema
  if (n->matches(torch::schema(
          "aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor"))) {
    // 返回 lambda 函数，处理节点的输入和输出
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& q = p_node->Input(1).toTensor();
      const auto dim = p_node->Input(2).toOptional<int64_t>();
      const auto keepdim = p_node->Input(3).toBool();
      const auto interpolation = p_node->Input(4).toStringView();
      // 如果输出为空，则调用 at::native::nanquantile 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::nanquantile(self, q, dim, keepdim, interpolation);
        return;
      }
      // 否则，调用 at::native::nanquantile_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::nanquantile_out(self, q, dim, keepdim, interpolation, out);
    };
  }
  // 如果不匹配指定的 schema，则记录日志并返回空指针
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 aten::msort 操作的函数实现
REGISTER_OPERATOR_FUNCTOR(aten::msort, aten_msort, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 schema
  if (n->matches(torch::schema("aten::msort(Tensor self) -> Tensor"))) {
    // 返回 lambda 函数，处理节点的输入和输出
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出为空，则调用 at::native::msort 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::msort(self);
        return;
      }
      // 否则，调用 at::native::msort_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::msort_out(self, out);
    };
  }
  // 如果不匹配指定的 schema，则记录日志并返回空指针
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 aten::renorm 操作的函数实现
REGISTER_OPERATOR_FUNCTOR(aten::renorm, aten_renorm, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 schema
  if (n->matches(torch::schema(
          "aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor"))) {
    // 返回 lambda 函数，处理节点的输入和输出
    return [](ProcessedNode* p_node) {
      // 获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto p = p_node->Input(1).toScalar();
      const auto dim = p_node->Input(2).toInt();
      const auto maxnorm = p_node->Input(3).toScalar();
      // 如果输出为空，则调用 at::cpu::renorm 函数
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::renorm(self, p, dim, maxnorm);
        return;
      }
      // 否则，调用 at::cpu::renorm_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::renorm_out(out, self, p, dim, maxnorm);
    };
  }
  // 如果不匹配指定的 schema，则记录日志并返回空指针
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册 aten::_convert_indices_from_coo_to_csr 操作的函数实现
REGISTER_OPERATOR_FUNCTOR(
    aten::_convert_indices_from_coo_to_csr,
    aten__convert_indices_from_coo_to_csr,
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个 Node* 参数 n，返回一个 SROperator 对象
      if (n->matches(torch::schema(
              "aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor"))) {
        // 如果节点 n 匹配特定的 Torch 脚本
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名函数，接受一个 ProcessedNode* 参数 p_node
          const auto& self = p_node->Input(0).toTensor();
          // 获取 p_node 的第一个输入参数，并将其转换为 Tensor 类型
          const auto size = p_node->Input(1).toInt();
          // 获取 p_node 的第二个输入参数，并将其转换为整数类型
          const auto out_int32 = p_node->Input(2).toBool();
          // 获取 p_node 的第三个输入参数，并将其转换为布尔类型
          if (p_node->Output(0).isNone()) {
            // 如果 p_node 的第一个输出参数为空
            p_node->Output(0) = at::cpu::_convert_indices_from_coo_to_csr(
                self, size, out_int32);
            // 调用 Torch 的 CPU 实现函数 _convert_indices_from_coo_to_csr，并将结果赋给 p_node 的第一个输出参数
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则，获取 p_node 的第一个输出参数，并将其转换为 Tensor 类型
          fastResizeToZero(out);
          // 调用自定义的函数 fastResizeToZero，将 out 迅速调整大小为零
          at::cpu::_convert_indices_from_coo_to_csr_out(
              out, self, size, out_int32);
          // 使用 Torch 的 CPU 实现函数 _convert_indices_from_coo_to_csr_out，将计算结果写入 out
        };
      }
      LogAndDumpSchema(n);
      // 如果节点 n 不匹配特定的 Torch 脚本，则记录日志并转储其模式
      return nullptr;
      // 返回空指针
    });
    // 匿名函数结束
# 注册一个自定义运算符函数，用于执行 CSR 到 COO 索引转换
REGISTER_OPERATOR_FUNCTOR(
    aten::_convert_indices_from_csr_to_coo,
    aten__convert_indices_from_csr_to_coo,
    [](Node* n) -> SROperator {
      # 检查节点是否匹配特定的 Torch 模式
      if (n->matches(torch::schema(
              "aten::_convert_indices_from_csr_to_coo(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False) -> Tensor"))) {
        # 返回一个 lambda 函数，处理节点的数据流
        return [](ProcessedNode* p_node) {
          # 从节点输入中获取 CSR 索引和列索引的张量
          const auto& crow_indices = p_node->Input(0).toTensor();
          const auto& col_indices = p_node->Input(1).toTensor();
          # 获取是否输出为 int32 类型和是否转置的布尔值
          const auto out_int32 = p_node->Input(2).toBool();
          const auto transpose = p_node->Input(3).toBool();
          # 如果输出张量为空，执行 CPU 上的 CSR 到 COO 转换
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_convert_indices_from_csr_to_coo(
                crow_indices, col_indices, out_int32, transpose);
            return;
          }
          # 否则，快速调整输出张量大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          # 调用 CPU 上的带输出参数版本的 CSR 到 COO 转换
          at::cpu::_convert_indices_from_csr_to_coo_out(
              out, crow_indices, col_indices, out_int32, transpose);
        };
      }
      # 如果节点不匹配，记录并转储节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册一个自定义运算符函数，用于执行均方误差损失计算
REGISTER_OPERATOR_FUNCTOR(aten::mse_loss, aten_mse_loss, [](Node* n) -> SROperator {
  # 检查节点是否匹配特定的 Torch 模式
  if (n->matches(torch::schema(
          "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"))) {
    # 返回一个 lambda 函数，处理节点的数据流
    return [](ProcessedNode* p_node) {
      # 从节点输入中获取 self 和 target 张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& target = p_node->Input(1).toTensor();
      # 获取损失计算时的降维方式
      const auto reduction = p_node->Input(2).toInt();
      # 如果输出张量为空，执行 CPU 上的均方误差损失计算
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::mse_loss(self, target, reduction);
        return;
      }
      # 否则，快速调整输出张量大小为零
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      # 调用 CPU 上的带输出参数版本的均方误差损失计算
      at::cpu::mse_loss_out(out, self, target, reduction);
    };
  }
  # 如果节点不匹配，记录并转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    // 匿名函数，接受一个 Node 指针参数 n，返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 Torch 脚本，计算多类边界损失函数的条件
      if (n->matches(torch::schema(
              "aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor"))) {
        // 返回另一个匿名函数，处理匹配成功的节点 p_node
        return [](ProcessedNode* p_node) {
          // 获取输入参数并转换为相应的 Tensor、Scalar 类型
          const auto& self = p_node->Input(0).toTensor();
          const auto& target = p_node->Input(1).toTensor();
          const auto p = p_node->Input(2).toScalar();
          const auto margin = p_node->Input(3).toScalar();
          const auto weight = p_node->Input(4).toOptional<at::Tensor>();
          const auto reduction = p_node->Input(5).toInt();
          // 如果输出节点为 None，调用 CPU 实现的多类边界损失函数计算，并设置输出结果
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::multi_margin_loss_cpu(
                self, target, p, margin, weight, reduction);
            return;
          }
          // 否则，获取输出 Tensor 并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 使用输出 Tensor 调用 CPU 实现的多类边界损失函数计算，将结果存储到 out 中
          at::native::multi_margin_loss_cpu_out(
              self, target, p, margin, weight, reduction, out);
        };
      }
      // 若节点 n 不匹配预期的 Torch 脚本，记录并转储其 schema 信息，然后返回空指针
      LogAndDumpSchema(n);
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::multilabel_margin_loss,
    aten_multilabel_margin_loss,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"))) {
        // 返回处理函数，该函数计算多标签边界损失
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量和参数
          const auto& self = p_node->Input(0).toTensor();
          const auto& target = p_node->Input(1).toTensor();
          const auto reduction = p_node->Input(2).toInt();
          // 如果输出未初始化，则调用原生函数进行计算并赋值
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::multilabel_margin_loss(self, target, reduction);
            return;
          }
          // 否则，重置输出张量大小并使用原生函数计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::multilabel_margin_loss_out(self, target, reduction, out);
        };
      }
      // 如果节点不匹配，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::soft_margin_loss,
    aten_soft_margin_loss,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"))) {
        // 返回处理函数，该函数计算软边界损失
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量和参数
          const auto& self = p_node->Input(0).toTensor();
          const auto& target = p_node->Input(1).toTensor();
          const auto reduction = p_node->Input(2).toInt();
          // 如果输出未初始化，则调用原生函数进行计算并赋值
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::soft_margin_loss(self, target, reduction);
            return;
          }
          // 否则，重置输出张量大小并使用原生函数计算结果
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::soft_margin_loss_out(self, target, reduction, out);
        };
      }
      // 如果节点不匹配，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::elu, aten_elu, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 Torch 脚本
  if (n->matches(torch::schema(
          "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"))) {
    // 返回处理函数，该函数计算 ELU（指数线性单元）函数
    return [](ProcessedNode* p_node) {
      // 从节点中获取输入张量和参数
      const auto& self = p_node->Input(0).toTensor();
      const auto alpha = p_node->Input(1).toScalar();
      const auto scale = p_node->Input(2).toScalar();
      const auto input_scale = p_node->Input(3).toScalar();
      // 如果输出未初始化，则调用 CPU 上的 ELU 函数进行计算并赋值
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::elu(self, alpha, scale, input_scale);
        return;
      }
      // 否则，重置输出张量大小并使用 CPU 上的 ELU 函数计算结果
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::elu_out(out, self, alpha, scale, input_scale);
    };
  }
  // 如果节点不匹配，则记录并输出节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    // 匿名函数，接受一个 Node* 参数并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 脚本模式
      if (n->matches(torch::schema(
              "aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor"))) {
        // 如果匹配，返回另一个匿名函数
        return [](ProcessedNode* p_node) {
          // 从 p_node 中获取输入参数并转换为相应类型
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto alpha = p_node->Input(1).toScalar();
          const auto scale = p_node->Input(2).toScalar();
          const auto input_scale = p_node->Input(3).toScalar();
          const auto is_result = p_node->Input(4).toBool();
          const auto& self_or_result = p_node->Input(5).toTensor();
          
          // 如果输出结果为空，执行以下操作
          if (p_node->Output(0).isNone()) {
            // 调用 at::cpu::elu_backward 函数计算梯度
            p_node->Output(0) = at::cpu::elu_backward(
                grad_output,
                alpha,
                scale,
                input_scale,
                is_result,
                self_or_result);
            return;
          }
          
          // 否则，获取输出的梯度输入对象并进行快速调整大小到零
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          
          // 使用 at::cpu::elu_backward_out 函数计算梯度输出
          at::cpu::elu_backward_out(
              grad_input,
              grad_output,
              alpha,
              scale,
              input_scale,
              is_result,
              self_or_result);
        };
      }
      
      // 如果节点不匹配指定模式，记录并转储节点的模式信息，然后返回空指针
      LogAndDumpSchema(n);
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(aten::glu, aten_glu, [](Node* n) -> SROperator {
  // 检查节点是否匹配 glu 操作的模式
  if (n->matches(
          torch::schema("aten::glu(Tensor self, int dim=-1) -> Tensor"))) {
    // 返回一个 lambda 函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入张量和维度参数
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      // 如果输出未初始化，执行 glu 操作并存储结果
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::glu(self, dim);
        return;
      }
      // 否则，重设输出张量大小，并执行 glu 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::glu_out(out, self, dim);
    };
  }
  // 如果节点不匹配，记录日志并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::hardsigmoid,
    aten_hardsigmoid,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 hardsigmoid 操作的模式
      if (n->matches(
              torch::schema("aten::hardsigmoid(Tensor self) -> Tensor"))) {
        // 返回一个 lambda 函数，处理匹配的节点
        return [](ProcessedNode* p_node) {
          // 获取输入张量
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出未初始化，执行 hardsigmoid 操作并存储结果
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::hardsigmoid(self);
            return;
          }
          // 否则，重设输出张量大小，并执行 hardsigmoid 操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::hardsigmoid_out(out, self);
        };
      }
      // 如果节点不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::hardsigmoid_backward,
    aten_hardsigmoid_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 hardsigmoid_backward 操作的模式
      if (n->matches(torch::schema(
              "aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor"))) {
        // 返回一个 lambda 函数，处理匹配的节点
        return [](ProcessedNode* p_node) {
          // 获取输入梯度张量和输入张量
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          // 如果输出未初始化，执行 hardsigmoid_backward 操作并存储结果
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::hardsigmoid_backward(grad_output, self);
            return;
          }
          // 否则，重设输出张量大小，并执行 hardsigmoid_backward 操作
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::hardsigmoid_backward_out(grad_input, grad_output, self);
        };
      }
      // 如果节点不匹配，记录日志并输出模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::hardtanh, aten_hardtanh, [](Node* n) -> SROperator {
  // 检查节点是否匹配 hardtanh 操作的模式
  if (n->matches(torch::schema(
          "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"))) {
    // 返回一个 lambda 函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 获取输入张量和最小/最大值参数
      const auto& self = p_node->Input(0).toTensor();
      const auto min_val = p_node->Input(1).toScalar();
      const auto max_val = p_node->Input(2).toScalar();
      // 如果输出未初始化，执行 hardtanh 操作并存储结果
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::hardtanh(self, min_val, max_val);
        return;
      }
      // 否则，重设输出张量大小，并执行 hardtanh 操作
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::hardtanh_out(self, min_val, max_val, out);
    };
  }
  // 如果节点不匹配，记录日志并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个指向 Node 的指针 n，返回一个 SROperator
      if (n->matches(torch::schema("aten::hardswish(Tensor self) -> Tensor"))) {
        // 如果节点 n 匹配 "aten::hardswish(Tensor self) -> Tensor" 的模式
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名函数，接受一个指向 ProcessedNode 的指针 p_node
          const auto& self = p_node->Input(0).toTensor();
          // 获取 p_node 的第一个输入作为 Tensor self
          if (p_node->Output(0).isNone()) {
            // 如果 p_node 的第一个输出为空
            p_node->Output(0) = at::native::hardswish(self);
            // 将经过 hardswish 处理后的 self 赋值给 p_node 的第一个输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则，获取 p_node 的第一个输出作为 Tensor out
          fastResizeToZero(out);
          // 调用 fastResizeToZero 函数，将 out 的尺寸快速调整为零
          at::native::hardswish_out(self, out);
          // 使用 hardswish_out 将 self 的内容写入 out
        };
      }
      // 如果节点 n 不匹配 "aten::hardswish(Tensor self) -> Tensor" 的模式，则记录日志并转储模式
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::leaky_relu_backward,
    aten_leaky_relu_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor"))) {
        // 返回 lambda 函数，处理节点数据
        return [](ProcessedNode* p_node) {
          // 提取节点的输入参数
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto negative_slope = p_node->Input(2).toScalar();
          const auto self_is_result = p_node->Input(3).toBool();
          // 如果输出为空，执行计算并存储到输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::leaky_relu_backward(
                grad_output, self, negative_slope, self_is_result);
            return;
          }
          // 如果输出已存在，则重置为零并执行计算
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::leaky_relu_backward_out(
              grad_input, grad_output, self, negative_slope, self_is_result);
        };
      }
      // 如果节点不匹配，记录日志并输出空指针
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::log_sigmoid,
    aten_log_sigmoid,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      if (n->matches(
              torch::schema("aten::log_sigmoid(Tensor self) -> Tensor"))) {
        // 返回 lambda 函数，处理节点数据
        return [](ProcessedNode* p_node) {
          // 提取节点的输入参数
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出为空，执行计算并存储到输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::log_sigmoid(self);
            return;
          }
          // 如果输出已存在，则重置为零并执行计算
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::log_sigmoid_out(self, out);
        };
      }
      // 如果节点不匹配，记录日志并输出空指针
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::softplus, aten_softplus, [](Node* n) -> SROperator {
  // 检查节点是否匹配特定的 Torch 操作模式
  if (n->matches(torch::schema(
          "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor"))) {
    // 返回 lambda 函数，处理节点数据
    return [](ProcessedNode* p_node) {
      // 提取节点的输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto beta = p_node->Input(1).toScalar();
      const auto threshold = p_node->Input(2).toScalar();
      // 如果输出为空，执行计算并存储到输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::softplus(self, beta, threshold);
        return;
      }
      // 如果输出已存在，则重置为零并执行计算
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::softplus_out(out, self, beta, threshold);
    };
  }
  // 如果节点不匹配，记录日志并输出空指针
  LogAndDumpSchema(n);
  return nullptr;
});
    // 定义一个匿名
REGISTER_OPERATOR_FUNCTOR(
    aten::softshrink,
    aten_softshrink,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 softshrink 操作的 Torch 模式
      if (n->matches(torch::schema(
              "aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor"))) {
        // 返回 lambda 表达式，处理 softshrink 操作
        return [](ProcessedNode* p_node) {
          // 获取输入的张量 self 和标量 lambd
          const auto& self = p_node->Input(0).toTensor();
          const auto lambd = p_node->Input(1).toScalar();
          // 如果输出张量为空，则计算 softshrink 并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::softshrink(self, lambd);
            return;
          }
          // 否则，快速调整输出张量大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 执行 softshrink 操作，将结果存储在输出张量中
          at::cpu::softshrink_out(out, self, lambd);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::softshrink_backward,
    aten_softshrink_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 softshrink_backward 操作的 Torch 模式
      if (n->matches(torch::schema(
              "aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor"))) {
        // 返回 lambda 表达式，处理 softshrink_backward 操作
        return [](ProcessedNode* p_node) {
          // 获取输入的梯度 grad_output、张量 self 和标量 lambd
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto lambd = p_node->Input(2).toScalar();
          // 如果输出张量为空，则计算 softshrink_backward 并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::softshrink_backward(grad_output, self, lambd);
            return;
          }
          // 否则，快速调整输出张量大小为零
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          // 执行 softshrink_backward 操作，将结果存储在输出张量中
          at::cpu::softshrink_backward_out(
              grad_input, grad_output, self, lambd);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::adaptive_max_pool2d_backward,
    aten_adaptive_max_pool2d_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 adaptive_max_pool2d_backward 操作的 Torch 模式
      if (n->matches(torch::schema(
              "aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor"))) {
        // 返回 lambda 表达式，处理 adaptive_max_pool2d_backward 操作
        return [](ProcessedNode* p_node) {
          // 获取输入的梯度 grad_output、张量 self 和张量 indices
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto& indices = p_node->Input(2).toTensor();
          // 如果输出张量为空，则计算 adaptive_max_pool2d_backward 并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::adaptive_max_pool2d_backward(
                grad_output, self, indices);
            return;
          }
          // 否则，快速调整输出张量大小为零
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          // 执行 adaptive_max_pool2d_backward 操作，将结果存储在输出张量中
          at::cpu::adaptive_max_pool2d_backward_out(
              grad_input, grad_output, self, indices);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::adaptive_max_pool3d_backward,
    aten_adaptive_max_pool3d_backward,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 adaptive_max_pool3d_backward 操作的 Torch 模式
      if (n->matches(torch::schema(
              "aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices, Tensor input_size) -> Tensor"))) {
        // 返回 lambda 表达式，处理 adaptive_max_pool3d_backward 操作
        return [](ProcessedNode* p_node) {
          // 获取输入的梯度 grad_output、张量 self、张量 indices 和张量 input_size
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto& indices = p_node->Input(2).toTensor();
          const auto& input_size = p_node->Input(3).toTensor();
          // 如果输出张量为空，则计算 adaptive_max_pool3d_backward 并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::adaptive_max_pool3d_backward(
                grad_output, self, indices, input_size);
            return;
          }
          // 否则，快速调整输出张量大小为零
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          // 执行 adaptive_max_pool3d_backward 操作，将结果存储在输出张量中
          at::cpu::adaptive_max_pool3d_backward_out(
              grad_input, grad_output, self, indices, input_size);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });
    // 匿名函数，接受一个 Node* 参数并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor"))) {
        // 如果匹配，则返回另一个匿名函数，处理节点的具体逻辑
        return [](ProcessedNode* p_node) {
          // 从处理后的节点 p_node 中获取输入参数 grad_output、self、indices
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto& indices = p_node->Input(2).toTensor();
          
          // 如果输出为 None，则执行 adaptive_max_pool3d_backward 操作并设置输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::adaptive_max_pool3d_backward(
                grad_output, self, indices);
            return;
          }
          
          // 否则，获取并重置 grad_input 引用，并执行 adaptive_max_pool3d_backward_out 操作
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::adaptive_max_pool3d_backward_out(
              grad_input, grad_output, self, indices);
        };
      }
      
      // 如果节点 n 不匹配特定脚本，则记录并转储其架构信息，并返回空指针
      LogAndDumpSchema(n);
      return nullptr;
    });
# 注册自定义操作符处理函数，用于处理 torch 中的 sigmoid_backward 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid_backward,
    aten_sigmoid_backward,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否匹配给定的 torch schema
      if (n->matches(torch::schema(
              "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor"))) {
        # 返回处理节点的 lambda 函数
        return [](ProcessedNode* p_node) {
          # 获取输入节点的梯度和输出
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          # 如果输出节点为空，则计算 sigmoid 的反向传播并设置到输出节点
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::sigmoid_backward(grad_output, output);
            return;
          }
          # 否则，获取输出节点的引用并清空其尺寸
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          # 调用 sigmoid 反向传播，并将结果存入输出节点
          at::cpu::sigmoid_backward_out(grad_input, grad_output, output);
        };
      }
      # 如果节点不匹配 schema，则记录日志并输出 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册自定义操作符处理函数，用于处理 torch 中的 tanh_backward 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::tanh_backward,
    aten_tanh_backward,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否匹配给定的 torch schema
      if (n->matches(torch::schema(
              "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor"))) {
        # 返回处理节点的 lambda 函数
        return [](ProcessedNode* p_node) {
          # 获取输入节点的梯度和输出
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          # 如果输出节点为空，则计算 tanh 的反向传播并设置到输出节点
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::tanh_backward(grad_output, output);
            return;
          }
          # 否则，获取输出节点的引用并清空其尺寸
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          # 调用 tanh 反向传播，并将结果存入输出节点
          at::cpu::tanh_backward_out(grad_input, grad_output, output);
        };
      }
      # 如果节点不匹配 schema，则记录日志并输出 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册自定义操作符处理函数，用于处理 torch 中的 isposinf 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::isposinf,
    aten_isposinf,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否匹配给定的 torch schema
      if (n->matches(torch::schema("aten::isposinf(Tensor self) -> Tensor"))) {
        # 返回处理节点的 lambda 函数
        return [](ProcessedNode* p_node) {
          # 获取输入节点的自身张量
          const auto& self = p_node->Input(0).toTensor();
          # 如果输出节点为空，则计算张量中正无穷的位置，并设置到输出节点
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::isposinf(self);
            return;
          }
          # 否则，获取输出节点的引用并清空其尺寸
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          # 调用计算正无穷位置的函数，并将结果存入输出节点
          at::cpu::isposinf_out(out, self);
        };
      }
      # 如果节点不匹配 schema，则记录日志并输出 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册自定义操作符处理函数，用于处理 torch 中的 isneginf 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::isneginf,
    aten_isneginf,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否匹配给定的 torch schema
      if (n->matches(torch::schema("aten::isneginf(Tensor self) -> Tensor"))) {
        # 返回处理节点的 lambda 函数
        return [](ProcessedNode* p_node) {
          # 获取输入节点的自身张量
          const auto& self = p_node->Input(0).toTensor();
          # 如果输出节点为空，则计算张量中负无穷的位置，并设置到输出节点
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::isneginf(self);
            return;
          }
          # 否则，获取输出节点的引用并清空其尺寸
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          # 调用计算负无穷位置的函数，并将结果存入输出节点
          at::cpu::isneginf_out(out, self);
        };
      }
      # 如果节点不匹配 schema，则记录日志并输出 schema
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册自定义操作符处理函数，用于处理 torch 中的 special_entr 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::special_entr,
    aten_special_entr,
    // 匿名函数，接受一个 Node* 类型的参数 n，并返回一个 SROperator 类型的对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 torch::schema
      if (n->matches(
              torch::schema("aten::special_entr(Tensor self) -> Tensor"))) {
        // 如果匹配，返回一个 lambda 函数
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个参数 self，并转换为 Tensor 类型的引用
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出节点的第一个参数为空
          if (p_node->Output(0).isNone()) {
            // 计算特殊熵并赋值给输出节点的第一个参数
            p_node->Output(0) = at::cpu::special_entr(self);
            return;
          }
          // 否则，获取输出节点的第一个参数，并将其尺寸快速调整为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 在输出节点的第一个参数上计算特殊熵
          at::cpu::special_entr_out(out, self);
        };
      }
      // 如果节点 n 不匹配特定的 schema，则记录并转储该节点的 schema 信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::special_ndtri,  // 注册自定义操作符 aten::special_ndtri
    aten_special_ndtri,   // 定义操作符的注册名为 aten_special_ndtri
    [](Node* n) -> SROperator {  // lambda 函数，接受 Node* 参数并返回 SROperator
      if (n->matches(  // 如果 Node* n 符合给定的 Torch schema
              torch::schema("aten::special_ndtri(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回 lambda 函数，处理 ProcessedNode*
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          if (p_node->Output(0).isNone()) {  // 如果输出张量为空
            p_node->Output(0) = at::cpu::special_ndtri(self);  // 计算特殊逆正态分布并赋值给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出张量
          fastResizeToZero(out);  // 快速调整输出张量大小为零
          at::cpu::special_ndtri_out(out, self);  // 计算特殊逆正态分布并输出到 out
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配的 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_log_ndtr,  // 注册自定义操作符 aten::special_log_ndtr
    aten_special_log_ndtr,   // 定义操作符的注册名为 aten_special_log_ndtr
    [](Node* n) -> SROperator {  // lambda 函数，接受 Node* 参数并返回 SROperator
      if (n->matches(  // 如果 Node* n 符合给定的 Torch schema
              torch::schema("aten::special_log_ndtr(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回 lambda 函数，处理 ProcessedNode*
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          if (p_node->Output(0).isNone()) {  // 如果输出张量为空
            p_node->Output(0) = at::cpu::special_log_ndtr(self);  // 计算特殊对数正态分布并赋值给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出张量
          fastResizeToZero(out);  // 快速调整输出张量大小为零
          at::cpu::special_log_ndtr_out(out, self);  // 计算特殊对数正态分布并输出到 out
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配的 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_expm1,  // 注册自定义操作符 aten::special_expm1
    aten_special_expm1,   // 定义操作符的注册名为 aten_special_expm1
    [](Node* n) -> SROperator {  // lambda 函数，接受 Node* 参数并返回 SROperator
      if (n->matches(  // 如果 Node* n 符合给定的 Torch schema
              torch::schema("aten::special_expm1(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回 lambda 函数，处理 ProcessedNode*
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          if (p_node->Output(0).isNone()) {  // 如果输出张量为空
            p_node->Output(0) = at::native::special_expm1(self);  // 计算特殊指数减一并赋值给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出张量
          fastResizeToZero(out);  // 快速调整输出张量大小为零
          at::native::special_expm1_out(self, out);  // 计算特殊指数减一并输出到 out
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配的 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_exp2,  // 注册自定义操作符 aten::special_exp2
    aten_special_exp2,   // 定义操作符的注册名为 aten_special_exp2
    [](Node* n) -> SROperator {  // lambda 函数，接受 Node* 参数并返回 SROperator
      if (n->matches(  // 如果 Node* n 符合给定的 Torch schema
              torch::schema("aten::special_exp2(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回 lambda 函数，处理 ProcessedNode*
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          if (p_node->Output(0).isNone()) {  // 如果输出张量为空
            p_node->Output(0) = at::native::special_exp2(self);  // 计算特殊指数2并赋值给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取输出张量
          fastResizeToZero(out);  // 快速调整输出张量大小为零
          at::native::special_exp2_out(self, out);  // 计算特殊指数2并输出到 out
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配的 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_psi,  // 注册自定义操作符 aten::special_psi
    aten_special_psi,   // 定义操作符的注册名为 aten_special_psi
    [](Node* n) -> SROperator {  // lambda 函数，接受 Node* 参数并返回 SROperator
      if (n->matches(  // 如果 Node* n 符合给定的 Torch schema
              torch::schema("aten::special_psi(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {  // 返回 lambda 函数，处理 ProcessedNode*
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          LogAndDumpSchema(n);  // 记录并转储 schema 信息
          return nullptr;  // 返回空指针，未实现特殊函数的情况
        };
      }
      LogAndDumpSchema(n);  // 记录并转储不匹配的 schema
      return nullptr;  // 返回空指针
    });
    // 匿名函数，接受一个 Node* 参数，并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 脚本模式 "aten::special_psi(Tensor self) -> Tensor"
      if (n->matches(
              torch::schema("aten::special_psi(Tensor self) -> Tensor"))) {
        // 返回另一个匿名函数，处理匹配的节点
        return [](ProcessedNode* p_node) {
          // 从处理后的节点获取第一个输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果第一个输出为空，计算特殊函数 special_psi 的结果并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_psi(self);
            return;
          }
          // 否则，获取第一个输出张量 out，并快速调整大小为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 使用 special_psi_out 函数计算特殊函数的结果并写入输出张量 out
          at::native::special_psi_out(self, out);
        };
      }
      // 如果节点 n 不匹配特定模式，记录并转储该节点的模式信息
      LogAndDumpSchema(n);
      // 返回空指针，表示未能处理该节点
      return nullptr;
    });
# 注册并定义 aten::special_digamma 操作的处理函数
REGISTER_OPERATOR_FUNCTOR(
    aten::special_digamma,
    aten_special_digamma,
    [](Node* n) -> SROperator {
      # 如果节点匹配特定的 Torch 模式
      if (n->matches(
              torch::schema("aten::special_digamma(Tensor self) -> Tensor"))) {
        # 返回一个处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          # 获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          # 如果输出张量为空，则调用 special_digamma 函数计算结果
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_digamma(self);
            return;
          }
          # 否则，重设输出张量大小为零，并用 special_digamma_out 函数填充输出
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_digamma_out(self, out);
        };
      }
      # 如果节点不匹配预期的模式，记录日志并转储模式信息
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册并定义 aten::special_gammaln 操作的处理函数，逻辑与上述类似
REGISTER_OPERATOR_FUNCTOR(
    aten::special_gammaln,
    aten_special_gammaln,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_gammaln(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_gammaln(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_gammaln_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册并定义 aten::special_erf 操作的处理函数，逻辑与上述类似
REGISTER_OPERATOR_FUNCTOR(
    aten::special_erf,
    aten_special_erf,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erf(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erf(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_erf_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册并定义 aten::special_erfc 操作的处理函数，逻辑与上述类似
REGISTER_OPERATOR_FUNCTOR(
    aten::special_erfc,
    aten_special_erfc,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erfc(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erfc(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_erfc_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

# 注册并定义 aten::special_erfcx 操作的处理函数，逻辑与前面的类似
REGISTER_OPERATOR_FUNCTOR(
    aten::special_erfcx,
    aten_special_erfcx,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erfcx(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erfcx(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_erfcx_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个 Node* 参数 n，返回一个 SROperator 对象
      if (n->matches(
              torch::schema("aten::special_erfcx(Tensor self) -> Tensor"))) {
        // 检查节点 n 是否匹配给定的 Torch 脚本模式
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名函数，接受一个 ProcessedNode* 参数 p_node
          const auto& self = p_node->Input(0).toTensor();
          // 获取 p_node 的第一个输入并转换为 Tensor 类型
          if (p_node->Output(0).isNone()) {
            // 如果 p_node 的第一个输出为空
            p_node->Output(0) = at::cpu::special_erfcx(self);
            // 调用 CPU 版本的 special_erfcx 函数，并将结果赋给 p_node 的第一个输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则获取 p_node 的第一个输出，并转换为 Tensor 类型
          fastResizeToZero(out);
          // 快速调整 out 的大小为零
          at::cpu::special_erfcx_out(out, self);
          // 在输出张量 out 上调用特定的 special_erfcx_out 函数
        };
      }
      // 如果不匹配特定模式，记录日志并转储模式信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::special_erfinv,  // 注册特殊函数 erfinv 的运算符
    aten_special_erfinv,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erfinv(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出为空，则调用 at::native::special_erfinv 处理输入 self
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erfinv(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 快速调整输出张量 out 的大小为零
          fastResizeToZero(out);
          // 调用 at::native::special_erfinv_out 将 self 的结果写入 out
          at::native::special_erfinv_out(self, out);
        };
      }
      // 记录和输出节点的架构信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_ndtr,  // 注册特殊函数 ndtr 的运算符
    aten_special_ndtr,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_ndtr(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出为空，则调用 at::native::special_ndtr 处理输入 self
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_ndtr(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 快速调整输出张量 out 的大小为零
          fastResizeToZero(out);
          // 调用 at::native::special_ndtr_out 将 self 的结果写入 out
          at::native::special_ndtr_out(self, out);
        };
      }
      // 记录和输出节点的架构信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_xlog1py,  // 注册特殊函数 xlog1py 的运算符
    aten_special_xlog1py,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_xlog1py(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为空，则调用 at::cpu::special_xlog1py 处理输入 self 和 other
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_xlog1py(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 快速调整输出张量 out 的大小为零
          fastResizeToZero(out);
          // 调用 at::cpu::special_xlog1py_out 将 self 和 other 的结果写入 out
          at::cpu::special_xlog1py_out(out, self, other);
        };
      }
      // 记录和输出节点的架构信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_xlogy,  // 注册特殊函数 xlogy 的运算符
    aten_special_xlogy,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_xlogy(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出为空，则调用 at::native::special_xlogy 处理输入 self 和 other
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_xlogy(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 快速调整输出张量 out 的大小为零
          fastResizeToZero(out);
          // 调用 at::native::special_xlogy_out 将 self 和 other 的结果写入 out
          at::native::special_xlogy_out(self, other, out);
        };
      }
      // 记录和输出节点的架构信息
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_zeta,  // 注册特殊函数 zeta 的运算符
    aten_special_zeta,
    // 定义一个匿名函数，接受一个 Node 指针参数 n，并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::special_zeta(Tensor self, Tensor other) -> Tensor"))) {
        // 如果匹配，返回一个新的匿名函数
        return [](ProcessedNode* p_node) {
          // 获取输入参数 self 和 other，并将其转换为 Tensor 对象
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出参数为 None，则调用特定函数计算结果并存储在输出参数中
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_zeta(self, other);
            return;
          }
          // 否则，获取输出参数的引用，并将其尺寸快速调整为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用特定函数，将计算结果存储在输出参数中
          at::cpu::special_zeta_out(out, self, other);
        };
      }
      // 如果节点 n 不匹配特定 Torch 脚本，记录并转储其模式信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
# 注册特殊函数 aten::special_i0 的运算符函数
REGISTER_OPERATOR_FUNCTOR(
    aten::special_i0,  # 注册的运算符名称
    aten_special_i0,   # 注册后对应的函数名称
    [](Node* n) -> SROperator {  # lambda 函数，返回一个 SROperator
      # 检查节点是否匹配给定的 Torch 模式
      if (n->matches(
              torch::schema("aten::special_i0(Tensor self) -> Tensor"))) {
        # 返回处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();  # 获取输入张量 self
          if (p_node->Output(0).isNone()) {  # 检查输出是否为空
            p_node->Output(0) = at::native::special_i0(self);  # 计算特殊函数 special_i0
            return;
          }
          auto& out = p_node->Output(0).toTensor();  # 获取输出张量 out 的引用
          fastResizeToZero(out);  # 快速调整 out 的大小为零
          at::native::special_i0_out(self, out);  # 将计算结果存入 out 中
        };
      }
      LogAndDumpSchema(n);  # 记录和转储模式匹配失败的情况
      return nullptr;  # 如果未成功匹配模式，则返回空指针
    });

# 注册特殊函数 aten::special_i0e 的运算符函数
REGISTER_OPERATOR_FUNCTOR(
    aten::special_i0e,  # 注册的运算符名称
    aten_special_i0e,   # 注册后对应的函数名称
    [](Node* n) -> SROperator {  # lambda 函数，返回一个 SROperator
      # 检查节点是否匹配给定的 Torch 模式
      if (n->matches(
              torch::schema("aten::special_i0e(Tensor self) -> Tensor"))) {
        # 返回处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();  # 获取输入张量 self
          if (p_node->Output(0).isNone()) {  # 检查输出是否为空
            p_node->Output(0) = at::cpu::special_i0e(self);  # 计算特殊函数 special_i0e
            return;
          }
          auto& out = p_node->Output(0).toTensor();  # 获取输出张量 out 的引用
          fastResizeToZero(out);  # 快速调整 out 的大小为零
          at::cpu::special_i0e_out(out, self);  # 将计算结果存入 out 中
        };
      }
      LogAndDumpSchema(n);  # 记录和转储模式匹配失败的情况
      return nullptr;  # 如果未成功匹配模式，则返回空指针
    });

# 注册特殊函数 aten::special_i1 的运算符函数
REGISTER_OPERATOR_FUNCTOR(
    aten::special_i1,  # 注册的运算符名称
    aten_special_i1,   # 注册后对应的函数名称
    [](Node* n) -> SROperator {  # lambda 函数，返回一个 SROperator
      # 检查节点是否匹配给定的 Torch 模式
      if (n->matches(
              torch::schema("aten::special_i1(Tensor self) -> Tensor"))) {
        # 返回处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();  # 获取输入张量 self
          if (p_node->Output(0).isNone()) {  # 检查输出是否为空
            p_node->Output(0) = at::cpu::special_i1(self);  # 计算特殊函数 special_i1
            return;
          }
          auto& out = p_node->Output(0).toTensor();  # 获取输出张量 out 的引用
          fastResizeToZero(out);  # 快速调整 out 的大小为零
          at::cpu::special_i1_out(out, self);  # 将计算结果存入 out 中
        };
      }
      LogAndDumpSchema(n);  # 记录和转储模式匹配失败的情况
      return nullptr;  # 如果未成功匹配模式，则返回空指针
    });

# 注册特殊函数 aten::special_i1e 的运算符函数
REGISTER_OPERATOR_FUNCTOR(
    aten::special_i1e,  # 注册的运算符名称
    aten_special_i1e,   # 注册后对应的函数名称
    [](Node* n) -> SROperator {  # lambda 函数，返回一个 SROperator
      # 检查节点是否匹配给定的 Torch 模式
      if (n->matches(
              torch::schema("aten::special_i1e(Tensor self) -> Tensor"))) {
        # 返回处理函数，处理节点的输入和输出
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();  # 获取输入张量 self
          if (p_node->Output(0).isNone()) {  # 检查输出是否为空
            p_node->Output(0) = at::cpu::special_i1e(self);  # 计算特殊函数 special_i1e
            return;
          }
          auto& out = p_node->Output(0).toTensor();  # 获取输出张量 out 的引用
          fastResizeToZero(out);  # 快速调整 out 的大小为零
          at::cpu::special_i1e_out(out, self);  # 将计算结果存入 out 中
        };
      }
      LogAndDumpSchema(n);  # 记录和转储模式匹配失败的情况
      return nullptr;  # 如果未成功匹配模式，则返回空指针
    });

# 注册特殊函数 aten::special_polygamma 的运算符函数
REGISTER_OPERATOR_FUNCTOR(
    aten::special_polygamma,  # 注册的运算符名称
    aten_special_polygamma,   # 注册后对应的函数名称
    [](Node* n) -> SROperator {
      // 匿名函数定义：接受一个指向 Node 的指针 n，返回一个 SROperator
      if (n->matches(torch::schema(
              "aten::special_polygamma(int n, Tensor self) -> Tensor"))) {
        // 如果节点 n 匹配特定的 Torch schema
        return [](ProcessedNode* p_node) {
          // 返回一个新的匿名函数：接受一个指向 ProcessedNode 的指针 p_node
          const auto n = p_node->Input(0).toInt();
          // 从 p_node 的第一个输入中提取整数 n
          const auto& self = p_node->Input(1).toTensor();
          // 从 p_node 的第二个输入中获取 Tensor self 的引用
          if (p_node->Output(0).isNone()) {
            // 如果 p_node 的第一个输出为空
            p_node->Output(0) = at::native::special_polygamma(n, self);
            // 则调用 at::native::special_polygamma 计算结果并存入输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          // 否则，获取 p_node 的第一个输出，并将其转换为 Tensor 类型
          fastResizeToZero(out);
          // 调用 fastResizeToZero 函数将输出 Tensor out 快速调整为零大小
          at::native::special_polygamma_out(n, self, out);
          // 调用 at::native::special_polygamma_out 将计算结果存入输出 Tensor out
        };
      }
      // 如果节点 n 不匹配特定的 Torch schema，则记录日志和转储 schema 信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
    // 结束匿名函数定义
REGISTER_OPERATOR_FUNCTOR(
    aten::special_expit,  // 注册特殊函数 special_expit 的运算符
    aten_special_expit,   // 使用 aten_special_expit 作为注册的标识符
    [](Node* n) -> SROperator {  // Lambda 表达式，接收一个 Node* 参数并返回一个 SROperator
      if (n->matches(
              torch::schema("aten::special_expit(Tensor self) -> Tensor"))) {
        // 如果 Node n 符合特定的 Torch schema，则执行以下操作
        return [](ProcessedNode* p_node) {  // 返回一个新的 Lambda 表达式，接收一个 ProcessedNode* 参数并不返回任何值
          const auto& self = p_node->Input(0).toTensor();  // 获取输入的第一个张量 self
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_expit(self);  // 如果输出为空，调用特殊的 expit 函数并将结果赋给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取已有的输出张量
          fastResizeToZero(out);  // 快速将输出张量调整为零
          at::native::special_expit_out(self, out);  // 调用特殊的 expit 函数，并将结果存入输出张量
        };
      }
      LogAndDumpSchema(n);  // 如果 Node n 不符合特定 schema，则记录和转储 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_sinc,  // 注册特殊函数 special_sinc 的运算符
    aten_special_sinc,   // 使用 aten_special_sinc 作为注册的标识符
    [](Node* n) -> SROperator {  // Lambda 表达式，接收一个 Node* 参数并返回一个 SROperator
      if (n->matches(
              torch::schema("aten::special_sinc(Tensor self) -> Tensor"))) {
        // 如果 Node n 符合特定的 Torch schema，则执行以下操作
        return [](ProcessedNode* p_node) {  // 返回一个新的 Lambda 表达式，接收一个 ProcessedNode* 参数并不返回任何值
          const auto& self = p_node->Input(0).toTensor();  // 获取输入的第一个张量 self
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_sinc(self);  // 如果输出为空，调用特殊的 sinc 函数并将结果赋给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取已有的输出张量
          fastResizeToZero(out);  // 快速将输出张量调整为零
          at::native::special_sinc_out(self, out);  // 调用特殊的 sinc 函数，并将结果存入输出张量
        };
      }
      LogAndDumpSchema(n);  // 如果 Node n 不符合特定 schema，则记录和转储 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_round,  // 注册特殊函数 special_round 的运算符
    aten_special_round,   // 使用 aten_special_round 作为注册的标识符
    [](Node* n) -> SROperator {  // Lambda 表达式，接收一个 Node* 参数并返回一个 SROperator
      if (n->matches(torch::schema(
              "aten::special_round(Tensor self, *, int decimals=0) -> Tensor"))) {
        // 如果 Node n 符合特定的 Torch schema，则执行以下操作
        return [](ProcessedNode* p_node) {  // 返回一个新的 Lambda 表达式，接收一个 ProcessedNode* 参数并不返回任何值
          const auto& self = p_node->Input(0).toTensor();  // 获取输入的第一个张量 self
          const auto decimals = p_node->Input(1).toInt();  // 获取第二个输入参数 decimals，并转换为整数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_round(self, decimals);  // 如果输出为空，调用特殊的 round 函数并将结果赋给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取已有的输出张量
          fastResizeToZero(out);  // 快速将输出张量调整为零
          at::native::special_round_out(self, decimals, out);  // 调用特殊的 round 函数，并将结果存入输出张量
        };
      }
      LogAndDumpSchema(n);  // 如果 Node n 不符合特定 schema，则记录和转储 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_log1p,  // 注册特殊函数 special_log1p 的运算符
    aten_special_log1p,   // 使用 aten_special_log1p 作为注册的标识符
    [](Node* n) -> SROperator {  // Lambda 表达式，接收一个 Node* 参数并返回一个 SROperator
      if (n->matches(
              torch::schema("aten::special_log1p(Tensor self) -> Tensor"))) {
        // 如果 Node n 符合特定的 Torch schema，则执行以下操作
        return [](ProcessedNode* p_node) {  // 返回一个新的 Lambda 表达式，接收一个 ProcessedNode* 参数并不返回任何值
          const auto& self = p_node->Input(0).toTensor();  // 获取输入的第一个张量 self
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_log1p(self);  // 如果输出为空，调用特殊的 log1p 函数并将结果赋给输出
            return;
          }
          auto& out = p_node->Output(0).toTensor();  // 否则获取已有的输出张量
          fastResizeToZero(out);  // 快速将输出张量调整为零
          at::native::special_log1p_out(self, out);  // 调用特殊的 log1p 函数，并将结果存入输出张量
        };
      }
      LogAndDumpSchema(n);  // 如果 Node n 不符合特定 schema，则记录和转储 schema
      return nullptr;  // 返回空指针
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_gammainc,  // 注册特殊函数 special_gammainc 的运算符
    aten_special_gammainc,
    // 匿名函数，接受一个 Node* 类型参数 n，并返回一个 SROperator 类型对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 Torch 脚本函数签名
      if (n->matches(torch::schema(
              "aten::special_gammainc(Tensor self, Tensor other) -> Tensor"))) {
        // 如果匹配，返回一个 Lambda 函数，处理节点 n 的信息
        return [](ProcessedNode* p_node) {
          // 获取输入参数 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出参数为 None，则调用 at::native::special_gammainc 函数
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_gammainc(self, other);
            return;
          }
          // 否则，获取输出参数并快速调整为零
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 at::native::special_gammainc_out 将结果写入输出参数 out
          at::native::special_gammainc_out(self, other, out);
        };
      }
      // 如果节点 n 不匹配指定的 Torch 脚本函数签名，则记录日志并转储模式信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });


这段代码是一个匿名函数的定义，它接受一个 `Node*` 类型的参数 `n`，并根据节点 `n` 的内容进行处理。
REGISTER_OPERATOR_FUNCTOR(
    aten::special_gammaincc,
    aten_special_gammaincc,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式字符串
      if (n->matches(torch::schema(
              "aten::special_gammaincc(Tensor self, Tensor other) -> Tensor"))) {
        // 返回一个 Lambda 函数处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，直接调用对应的函数，并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_gammaincc(self, other);
            return;
          }
          // 否则，获取输出张量并调用对应的函数，实现原地操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_gammaincc_out(self, other, out);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_multigammaln,
    aten_special_multigammaln,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式字符串
      if (n->matches(torch::schema(
              "aten::special_multigammaln(Tensor self, int p) -> Tensor"))) {
        // 返回一个 Lambda 函数处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self 和整数 p
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toInt();
          // 如果输出张量为空，直接调用对应的函数，并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_multigammaln(self, p);
            return;
          }
          // 否则，获取输出张量并调用对应的函数，实现原地操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_multigammaln_out(self, p, out);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_cross,
    aten_linalg_cross,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式字符串
      if (n->matches(torch::schema(
              "aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor"))) {
        // 返回一个 Lambda 函数处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入张量 self, other 和维度 dim
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          // 如果输出张量为空，直接调用对应的函数，并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::linalg_cross(self, other, dim);
            return;
          }
          // 否则，获取输出张量并调用对应的函数，实现原地操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::linalg_cross_out(out, self, other, dim);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_det,
    aten_linalg_det,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 torch 模式字符串
      if (n->matches(torch::schema("aten::linalg_det(Tensor A) -> Tensor"))) {
        // 返回一个 Lambda 函数处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入张量 A
          const auto& A = p_node->Input(0).toTensor();
          // 如果输出张量为空，直接调用对应的函数，并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_det(A);
            return;
          }
          // 否则，获取输出张量并调用对应的函数，实现原地操作
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_det_out(A, out);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      return nullptr;
    });
// 注册自定义操作符函数，用于执行 torch 中的 aten::linalg_matmul 操作
REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_matmul,
    aten_linalg_matmul,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::linalg_matmul schema
      if (n->matches(torch::schema(
              "aten::linalg_matmul(Tensor self, Tensor other) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的逻辑
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量 self 和 other
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 如果输出张量为空，直接执行 linalg_matmul 操作并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_matmul(self, other);
            return;
          }
          // 否则，获取输出张量并进行快速重置
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 linalg_matmul_out 函数，将结果写入输出张量 out
          at::native::linalg_matmul_out(self, other, out);
        };
      }
      // 若未匹配到指定 schema，记录日志并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册自定义操作符函数，用于执行 torch 中的 aten::linalg_eigvals 操作
REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_eigvals,
    aten_linalg_eigvals,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::linalg_eigvals schema
      if (n->matches(
              torch::schema("aten::linalg_eigvals(Tensor self) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的逻辑
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出张量为空，直接执行 linalg_eigvals 操作并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_eigvals(self);
            return;
          }
          // 否则，获取输出张量并进行快速重置
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 linalg_eigvals_out 函数，将结果写入输出张量 out
          at::native::linalg_eigvals_out(self, out);
        };
      }
      // 若未匹配到指定 schema，记录日志并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册自定义操作符函数，用于执行 torch 中的 aten::linalg_inv 操作
REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_inv,
    aten_linalg_inv,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::linalg_inv schema
      if (n->matches(torch::schema("aten::linalg_inv(Tensor A) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的逻辑
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量 A
          const auto& A = p_node->Input(0).toTensor();
          // 如果输出张量为空，直接执行 linalg_inv 操作并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_inv(A);
            return;
          }
          // 否则，获取输出张量并进行快速重置
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 linalg_inv_out 函数，将结果写入输出张量 out
          at::native::linalg_inv_out(A, out);
        };
      }
      // 若未匹配到指定 schema，记录日志并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册自定义操作符函数，用于执行 torch 中的 aten::inverse 操作
REGISTER_OPERATOR_FUNCTOR(
    aten::inverse,
    aten_inverse,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::inverse schema
      if (n->matches(torch::schema("aten::inverse(Tensor self) -> Tensor"))) {
        // 返回 lambda 表达式，处理节点的逻辑
        return [](ProcessedNode* p_node) {
          // 从节点中获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 如果输出张量为空，直接执行 inverse 操作并赋值给输出
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::inverse(self);
            return;
          }
          // 否则，获取输出张量并进行快速重置
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 inverse_out 函数，将结果写入输出张量 out
          at::native::inverse_out(self, out);
        };
      }
      // 若未匹配到指定 schema，记录日志并转储 schema 信息
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册自定义操作符函数，用于执行 torch 中的 aten::inner 操作
REGISTER_OPERATOR_FUNCTOR(
    aten::inner,
    aten_inner,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 aten::inner schema
      if (n->matches(
              torch::schema("aten::inner(Tensor self, Tensor other) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数接受一个 ProcessedNode* 参数
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个和第二个张量，并存储在 self 和 other 中
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出节点的第一个输出为空
      if (p_node->Output(0).isNone()) {
        // 计算 self 和 other 的内积，并存储在输出节点的第一个输出中
        p_node->Output(0) = at::native::inner(self, other);
        return;  // lambda 函数结束
      }
      // 否则，获取输出节点的第一个输出张量，并存储在 out 中
      auto& out = p_node->Output(0).toTensor();
      // 将 out 张量的大小调整为零
      fastResizeToZero(out);
      // 计算 self 和 other 的内积，并将结果存储在 out 张量中
      at::native::inner_out(self, other, out);
    };
  }
  // 记录和转储节点的模式
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::outer, aten_outer, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 Torch 操作模式
  if (n->matches(
          torch::schema("aten::outer(Tensor self, Tensor vec2) -> Tensor"))) {
    // 返回 lambda 函数，处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个和第二个张量
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec2 = p_node->Input(1).toTensor();
      // 如果输出节点为空，直接计算并设置输出节点的值
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::outer(self, vec2);
        return;
      }
      // 否则，获取输出节点的张量并进行快速调整大小
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      // 调用 Torch 的 outer_out 函数，并将结果存入输出节点的张量
      at::native::outer_out(self, vec2, out);
    };
  }
  // 如果节点不匹配，记录并转储其模式
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_cond,
    aten_linalg_cond,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor"))) {
        // 返回 lambda 函数，处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的张量和可选的标量 p
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toOptional<at::Scalar>();
          // 如果输出节点为空，直接计算并设置输出节点的值
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_cond(self, p);
            return;
          }
          // 否则，获取输出节点的张量并进行快速调整大小
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 Torch 的 linalg_cond_out 函数，并将结果存入输出节点的张量
          at::native::linalg_cond_out(self, p, out);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_solve,
    aten_linalg_solve,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::linalg_solve(Tensor A, Tensor B, *, bool left=True) -> Tensor"))) {
        // 返回 lambda 函数，处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的两个张量 A 和 B，以及布尔值 left
          const auto& A = p_node->Input(0).toTensor();
          const auto& B = p_node->Input(1).toTensor();
          const auto left = p_node->Input(2).toBool();
          // 如果输出节点为空，直接计算并设置输出节点的值
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_solve(A, B, left);
            return;
          }
          // 否则，获取输出节点的张量并进行快速调整大小
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 Torch 的 linalg_solve_out 函数，并将结果存入输出节点的张量
          at::native::linalg_solve_out(A, B, left, out);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_tensorinv,
    aten_linalg_tensorinv,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor"))) {
        // 返回 lambda 函数，处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的张量 self 和整数 ind
          const auto& self = p_node->Input(0).toTensor();
          const auto ind = p_node->Input(1).toInt();
          // 如果输出节点为空，直接计算并设置输出节点的值
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_tensorinv(self, ind);
            return;
          }
          // 否则，获取输出节点的张量并进行快速调整大小
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          // 调用 Torch 的 linalg_tensorinv_out 函数，并将结果存入输出节点的张量
          at::native::linalg_tensorinv_out(self, ind, out);
        };
      }
      // 如果节点不匹配，记录并转储其模式
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_matrix_power,
    at`
    # 定义一个运算符，表示使用 aten::linalg_matrix_power 操作
    aten_linalg_matrix_power,
    # 定义一个 lambda 函数，接受 Node 指针，返回 SROperator
    [](Node* n) -> SROperator {
      # 如果节点匹配特定的 schema，进行处理
      if (n->matches(torch::schema(
              "aten::linalg_matrix_power(Tensor self, int n) -> Tensor"))) {
        # 返回一个 lambda 函数，接受 ProcessedNode 指针作为参数
        return [](ProcessedNode* p_node) {
          # 获取输入 tensor self
          const auto& self = p_node->Input(0).toTensor();
          # 获取输入的整数 n
          const auto n = p_node->Input(1).toInt();
          # 如果输出是 None，则调用 at::native::linalg_matrix_power 函数进行计算
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_matrix_power(self, n);
            return;
          }
          # 获取输出 tensor 的引用
          auto& out = p_node->Output(0).toTensor();
          # 将输出 tensor 调整大小为零
          fastResizeToZero(out);
          # 调用 at::native::linalg_matrix_power_out 函数进行计算，保存结果到输出 tensor
          at::native::linalg_matrix_power_out(self, n, out);
        };
      }
      # 如果节点不匹配 schema，记录日志并转储 schema
      LogAndDumpSchema(n);
      return nullptr;
    });
// 注册一个自定义的操作符处理函数，用于处理 aten::view_as_real 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view_as_real,
    aten_view_as_real,
    [](Node* n) -> SROperator {
      // 如果节点 n 符合 "aten::view_as_real(Tensor(a) self) -> Tensor(a)" 的模式
      if (n->matches(torch::schema(
              "aten::view_as_real(Tensor(a) self) -> Tensor(a)"))) {
        // 返回一个 lambda 函数，处理输入节点的过程，将处理后的结果写入输出节点
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::view_as_real(self);
        };
      }
      // 如果不符合模式，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;  // 返回空指针
    });

// 注册一个自定义的操作符处理函数，用于处理 aten::view_as_complex 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view_as_complex,
    aten_view_as_complex,
    [](Node* n) -> SROperator {
      // 如果节点 n 符合 "aten::view_as_complex(Tensor(a) self) -> Tensor(a)" 的模式
      if (n->matches(torch::schema(
              "aten::view_as_complex(Tensor(a) self) -> Tensor(a)"))) {
        // 返回一个 lambda 函数，处理输入节点的过程，将处理后的结果写入输出节点
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::view_as_complex(self);
        };
      }
      // 如果不符合模式，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;  // 返回空指针
    });

// 注册一个自定义的操作符处理函数，用于处理 aten::real 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::real,
    aten_real,
    [](Node* n) -> SROperator {
      // 如果节点 n 符合 "aten::real(Tensor(a) self) -> Tensor(a)" 的模式
      if (n->matches(
              torch::schema("aten::real(Tensor(a) self) -> Tensor(a)"))) {
        // 返回一个 lambda 函数，处理输入节点的过程，将处理后的结果写入输出节点
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::real(self);
        };
      }
      // 如果不符合模式，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;  // 返回空指针
    });

// 注册一个自定义的操作符处理函数，用于处理 aten::imag 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::imag,
    aten_imag,
    [](Node* n) -> SROperator {
      // 如果节点 n 符合 "aten::imag(Tensor(a) self) -> Tensor(a)" 的模式
      if (n->matches(
              torch::schema("aten::imag(Tensor(a) self) -> Tensor(a)"))) {
        // 返回一个 lambda 函数，处理输入节点的过程，将处理后的结果写入输出节点
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::imag(self);
        };
      }
      // 如果不符合模式，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;  // 返回空指针
    });

// 注册一个自定义的操作符处理函数，用于处理 aten::_conj 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_conj,
    aten__conj,
    [](Node* n) -> SROperator {
      // 如果节点 n 符合 "aten::_conj(Tensor(a) self) -> Tensor(a)" 的模式
      if (n->matches(
              torch::schema("aten::_conj(Tensor(a) self) -> Tensor(a)"))) {
        // 返回一个 lambda 函数，处理输入节点的过程，将处理后的结果写入输出节点
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::_conj(self);
        };
      }
      // 如果不符合模式，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;  // 返回空指针
    });

// 注册一个自定义的操作符处理函数，用于处理 aten::conj 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::conj,
    aten_conj,
    [](Node* n) -> SROperator {
      // 如果节点 n 符合 "aten::conj(Tensor(a) self) -> Tensor(a)" 的模式
      if (n->matches(
              torch::schema("aten::conj(Tensor(a) self) -> Tensor(a)"))) {
        // 返回一个 lambda 函数，处理输入节点的过程，将处理后的结果写入输出节点
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::conj(self);
        };
      }
      // 如果不符合模式，则记录并输出节点的模式信息
      LogAndDumpSchema(n);
      return nullptr;  // 返回空指针
    });

// 注册一个自定义的操作符处理函数，用于处理 aten::resolve_conj 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::resolve_conj,
    aten_resolve_conj,
    // 匿名函数，接受一个 Node* 类型的参数 n，返回一个 SROperator 类型的对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 Torch 脚本
      if (n->matches(torch::schema(
              "aten::resolve_conj(Tensor(a) self) -> Tensor(a)"))) {
        // 如果匹配，则返回一个匿名函数，处理匹配的节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个输入作为 self 引用
          const auto& self = p_node->Input(0).toTensor();
          // 将处理后的结果写入处理节点的第一个输出
          p_node->Output(0) = at::native::resolve_conj(self);
        };
      }
      // 如果节点 n 不匹配指定的 Torch 脚本，则记录并转储其模式
      LogAndDumpSchema(n);
      // 返回空指针表示没有匹配的处理函数
      return nullptr;
    });
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::resolve_neg,  // 注册原生运算符函数，解析 torch 中的 resolve_neg 操作
    aten_resolve_neg,   // 在注册表中使用的名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node 指针并返回 SROperator
      if (n->matches(torch::schema(  // 如果节点匹配特定的 Torch 模式
              "aten::resolve_neg(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {  // 返回一个 Lambda 函数，处理已处理节点
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          p_node->Output(0) = at::native::resolve_neg(self);  // 调用 resolve_neg 函数处理并设置输出
        };
      }
      LogAndDumpSchema(n);  // 如果没有匹配的模式，记录和转储节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_neg_view,  // 注册原生运算符函数，处理 torch 中的 _neg_view 操作
    aten__neg_view,   // 在注册表中使用的名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node 指针并返回 SROperator
      if (n->matches(  // 如果节点匹配特定的 Torch 模式
              torch::schema("aten::_neg_view(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {  // 返回一个 Lambda 函数，处理已处理节点
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          p_node->Output(0) = at::native::_neg_view(self);  // 调用 _neg_view 函数处理并设置输出
        };
      }
      LogAndDumpSchema(n);  // 如果没有匹配的模式，记录和转储节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::diagonal,  // 注册原生运算符函数，处理 torch 中的 diagonal 操作
    aten_diagonal,   // 在注册表中使用的名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node 指针并返回 SROperator
      if (n->matches(torch::schema(  // 如果节点匹配特定的 Torch 模式
              "aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {  // 返回一个 Lambda 函数，处理已处理节点
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          const auto offset = p_node->Input(1).toInt();  // 获取输入偏移量 offset
          const auto dim1 = p_node->Input(2).toInt();  // 获取输入维度 dim1
          const auto dim2 = p_node->Input(3).toInt();  // 获取输入维度 dim2
          p_node->Output(0) = at::native::diagonal(self, offset, dim1, dim2);  // 调用 diagonal 函数处理并设置输出
        };
      }
      LogAndDumpSchema(n);  // 如果没有匹配的模式，记录和转储节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::linalg_diagonal,  // 注册原生运算符函数，处理 torch 中的 linalg_diagonal 操作
    aten_linalg_diagonal,   // 在注册表中使用的名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node 指针并返回 SROperator
      if (n->matches(torch::schema(  // 如果节点匹配特定的 Torch 模式
              "aten::linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {  // 返回一个 Lambda 函数，处理已处理节点
          const auto& A = p_node->Input(0).toTensor();  // 获取输入张量 A
          const auto offset = p_node->Input(1).toInt();  // 获取输入偏移量 offset
          const auto dim1 = p_node->Input(2).toInt();  // 获取输入维度 dim1
          const auto dim2 = p_node->Input(3).toInt();  // 获取输入维度 dim2
          p_node->Output(0) = at::native::linalg_diagonal(A, offset, dim1, dim2);  // 调用 linalg_diagonal 函数处理并设置输出
        };
      }
      LogAndDumpSchema(n);  // 如果没有匹配的模式，记录和转储节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::movedim,  // 注册原生运算符函数，处理 torch 中的 movedim 操作
    aten_movedim,   // 在注册表中使用的名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node 指针并返回 SROperator
      if (n->matches(torch::schema(  // 如果节点匹配特定的 Torch 模式
              "aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {  // 返回一个 Lambda 函数，处理已处理节点
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          const auto source = p_node->Input(1).toInt();  // 获取输入源维度 source
          const auto destination = p_node->Input(2).toInt();  // 获取输入目标维度 destination
          p_node->Output(0) = at::native::movedim(self, source, destination);  // 调用 movedim 函数处理并设置输出
        };
      }
      LogAndDumpSchema(n);  // 如果没有匹配的模式，记录和转储节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::moveaxis,  // 注册原生运算符函数，处理 torch 中的 moveaxis 操作
    aten_moveaxis,   // 在注册表中使用的名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node 指针并返回 SROperator
      if (n->matches(torch::schema(  // 如果节点匹配特定的 Torch 模式
              "aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {  // 返回一个 Lambda 函数，处理已处理节点
          const auto& self = p_node->Input(0).toTensor();  // 获取输入张量 self
          const auto source = p_node->Input(1).toInt();  // 获取输入源维度 source
          const auto destination = p_node->Input(2).toInt();  // 获取输入目标维度 destination
          p_node->Output(0) = at::native::moveaxis(self, source, destination);  // 调用 moveaxis 函数处理并设置输出
        };
      }
      LogAndDumpSchema(n);  // 如果没有匹配的模式，记录和转储节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });
    // 返回一个 lambda 函数，该函数接受一个 ProcessedNode 指针作为参数
    return [](ProcessedNode* p_node) {
      // 获取 p_node 的第一个输入作为 self 引用
      const auto& self = p_node->Input(0).toTensor();
      // 获取 p_node 的第二个输入作为 source
      const auto source = p_node->Input(1).toInt();
      // 获取 p_node 的第三个输入作为 destination
      const auto destination = p_node->Input(2).toInt();
      // 将 self 张量的轴从 source 移动到 destination，并将结果存储到 p_node 的第一个输出
      p_node->Output(0) = at::native::moveaxis(self, source, destination);
    };
  }
  // 记录和输出节点 n 的结构信息
  LogAndDumpSchema(n);
  // 返回空指针，表示函数执行完成
  return nullptr;
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::unsqueeze,
    aten_unsqueeze,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 schema
      if (n->matches(
              torch::schema("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"))) {
        // 如果匹配，返回一个 lambda 表达式
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个参数作为 self 张量
          const auto& self = p_node->Input(0).toTensor();
          // 调用 ATen 库的 unsqueeze 操作，并将结果存储到输出节点的第一个位置
          p_node->Output(0) = at::native::unsqueeze(self, p_node->Input(1).toInt());
        };
      }
      // 如果不匹配，记录并转储 schema 信息
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });
    [](Node* n) -> SROperator {
      // 匿名 lambda 函数，接受一个 Node 指针 n，返回一个 SROperator 对象
      if (n->matches(torch::schema(
              "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"))) {
        // 检查节点 n 是否匹配指定的 torch schema，如果匹配则执行以下操作
        return [](ProcessedNode* p_node) {
          // 匿名 lambda 函数，接受一个 ProcessedNode 指针 p_node，返回 void
          const auto& self = p_node->Input(0).toTensor();
          // 从 p_node 的第一个输入获取 Tensor 并赋值给 self
          const auto dim = p_node->Input(1).toInt();
          // 从 p_node 的第二个输入获取整数并赋值给 dim
          p_node->Output(0) = at::native::unsqueeze(self, dim);
          // 调用 PyTorch 的 unsqueeze 函数，将 self 在 dim 维度上展开，并将结果存入 p_node 的第一个输出
        };
      }
      // 如果节点 n 不匹配指定的 torch schema，则执行以下操作
      LogAndDumpSchema(n);
      // 记录并转储节点 n 的 schema 信息（这是一个未定义的函数或方法调用，根据上下文推测是用于调试和记录）
      return nullptr;
      // 返回空指针
    });
    // 匿名 lambda 函数的结尾
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view_as,
    aten_view_as,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)"))) {
        // 如果匹配，返回一个 lambda 函数，该函数用于处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个和第二个张量
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          // 将处理后的结果赋给输出节点的第一个位置
          p_node->Output(0) = at::native::view_as(self, other);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::positive,
    aten_positive,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      if (n->matches(
              torch::schema("aten::positive(Tensor(a) self) -> Tensor(a)"))) {
        // 如果匹配，返回一个 lambda 函数，该函数用于处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个张量
          const auto& self = p_node->Input(0).toTensor();
          // 将处理后的结果赋给输出节点的第一个位置
          p_node->Output(0) = at::native::positive(self);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_autocast_to_reduced_precision,
    aten__autocast_to_reduced_precision,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)"))) {
        // 如果匹配，返回一个 lambda 函数，该函数用于处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的张量和布尔值参数
          const auto& self = p_node->Input(0).toTensor();
          const auto cuda_enabled = p_node->Input(1).toBool();
          const auto cpu_enabled = p_node->Input(2).toBool();
          const auto cuda_dtype = p_node->Input(3).toScalarType();
          const auto cpu_dtype = p_node->Input(4).toScalarType();
          // 将处理后的结果赋给输出节点的第一个位置
          p_node->Output(0) = at::native::_autocast_to_reduced_precision(
              self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_autocast_to_full_precision,
    aten__autocast_to_full_precision,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      if (n->matches(torch::schema(
              "aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)"))) {
        // 如果匹配，返回一个 lambda 函数，该函数用于处理节点
        return [](ProcessedNode* p_node) {
          // 获取输入节点的张量和布尔值参数
          const auto& self = p_node->Input(0).toTensor();
          const auto cuda_enabled = p_node->Input(1).toBool();
          const auto cpu_enabled = p_node->Input(2).toBool();
          // 将处理后的结果赋给输出节点的第一个位置
          p_node->Output(0) = at::native::_autocast_to_full_precision(
              self, cuda_enabled, cpu_enabled);
        };
      }
      // 如果不匹配，记录并转储模式
      LogAndDumpSchema(n);
      // 返回空指针
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::swapaxes,
    aten_swapaxes,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 操作模式
      // 如果匹配，应该在这里补充对应的处理代码，但此处省略了
    [](Node* n) -> SROperator {
      // 检查节点 `n` 是否匹配指定的 PyTorch 操作模式
      if (n->matches(torch::schema(
              "aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)"))) {
        // 如果匹配，返回一个 lambda 表达式，用于处理节点 `p_node`
        return [](ProcessedNode* p_node) {
          // 获取输入节点的第一个参数作为 self 张量
          const auto& self = p_node->Input(0).toTensor();
          // 获取输入节点的第二个参数作为 axis0
          const auto axis0 = p_node->Input(1).toInt();
          // 获取输入节点的第三个参数作为 axis1
          const auto axis1 = p_node->Input(2).toInt();
          // 调用 PyTorch 的 swapaxes 函数，将结果赋给输出节点的第一个参数
          p_node->Output(0) = at::native::swapaxes(self, axis0, axis1);
        };
      }
      // 如果节点 `n` 不匹配预期的操作模式，则记录并转储其模式信息
      LogAndDumpSchema(n);
      // 返回空指针，表示没有有效的操作处理器
      return nullptr;
    });
// 注册原生操作函数，对应于 ATen 中的 swapdims 操作
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::swapdims,
    aten_swapdims,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配 swapdims 操作的 Torch 模式
      if (n->matches(torch::schema(
              "aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"))) {
        // 如果匹配，则返回 lambda 函数，执行 swapdims 操作
        return [](ProcessedNode* p_node) {
          // 从处理节点中获取输入张量 self、维度 dim0 和 dim1
          const auto& self = p_node->Input(0).toTensor();
          const auto dim0 = p_node->Input(1).toInt();
          const auto dim1 = p_node->Input(2).toInt();
          // 将 swapdims 的结果设置为处理节点的第一个输出
          p_node->Output(0) = at::native::swapdims(self, dim0, dim1);
        };
      }
      // 若未匹配，则记录并转储节点 n 的模式信息
      LogAndDumpSchema(n);
      return nullptr; // 返回空指针
    });

// 注册原生操作函数，对应于 ATen 中的 unfold 操作
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::unfold, aten_unfold, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配 unfold 操作的 Torch 模式
  if (n->matches(torch::schema(
          "aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)"))) {
    // 如果匹配，则返回 lambda 函数，执行 unfold 操作
    return [](ProcessedNode* p_node) {
      // 从处理节点中获取输入张量 self、维度 dimension、大小 size 和步长 step
      const auto& self = p_node->Input(0).toTensor();
      const auto dimension = p_node->Input(1).toInt();
      const auto size = p_node->Input(2).toInt();
      const auto step = p_node->Input(3).toInt();
      // 将 unfold 的结果设置为处理节点的第一个输出
      p_node->Output(0) = at::native::unfold(self, dimension, size, step);
    };
  }
  // 若未匹配，则记录并转储节点 n 的模式信息
  LogAndDumpSchema(n);
  return nullptr; // 返回空指针
});

// 注册原生操作函数，对应于 ATen 中的 alias 操作
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::alias,
    aten_alias,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配 alias 操作的 Torch 模式
      if (n->matches(
              torch::schema("aten::alias(Tensor(a) self) -> Tensor(a)"))) {
        // 如果匹配，则返回 lambda 函数，执行 alias 操作
        return [](ProcessedNode* p_node) {
          // 从处理节点中获取输入张量 self
          const auto& self = p_node->Input(0).toTensor();
          // 将 alias 的结果设置为处理节点的第一个输出
          p_node->Output(0) = at::native::alias(self);
        };
      }
      // 若未匹配，则记录并转储节点 n 的模式信息
      LogAndDumpSchema(n);
      return nullptr; // 返回空指针
    });

} // namespace jit
} // namespace torch
```