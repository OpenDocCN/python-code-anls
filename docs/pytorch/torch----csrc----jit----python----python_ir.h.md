# `.\pytorch\torch\csrc\jit\python\python_ir.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的 IR 相关头文件
#include <torch/csrc/utils/object_ptr.h>  // 包含 Torch 的智能指针相关头文件

namespace torch::jit {

// 初始化 Python IR 绑定到给定的模块对象
void initPythonIRBindings(PyObject* module);

// 执行一个 Python 函数，用于我们无法优化但希望围绕其优化的操作
struct ConcretePythonOp : public PythonOp {
  static Symbol Kind;  // 定义 Python 操作的符号标识

  ConcretePythonOp(Graph* graph) : PythonOp(graph, ::c10::prim::PythonOp) {}  // 构造函数，初始化 PythonOp

  ConcretePythonOp* init(
      THPObjectPtr&& pyobj,  // Python 对象的智能指针
      const std::string& cconv,  // Python 函数的调用约定
      pyobj_list&& scalar_args) {  // Python 函数的标量参数列表
    this->pyobj = std::move(pyobj);
    this->scalar_args = std::move(scalar_args);
    this->cconv = cconv;
    return this;
  }

  // Python 对象，包含此函数的实现。可以是一个类（非遗留）或对象（遗留）。详见 TraceInterpreterState 以了解执行语义。
  THPObjectPtr pyobj;

  // Python 函数的调用约定。
  // 'c' -- 常量参数
  // 'd' -- 动态参数
  std::string cconv;

  // Python 函数的标量参数。不一定按此顺序传递给函数；请参考 cconv 获取正确的顺序。
  std::vector<THPObjectPtr> scalar_args;

  std::string name() const override;  // 覆盖基类方法，返回操作的名称
  void cloneFrom(Node* other_) override;  // 覆盖基类方法，从另一个节点克隆
  Node* allocNewInstance(Graph* g) override {  // 分配并返回一个新的 ConcretePythonOp 实例
    return new ConcretePythonOp(g);
  }

  // 恢复 autograd.Function 实例，如果此 PythonOp 的函数最初是 SomeFunction.apply。
  // 用于在 ONNX 中发现符号化
  std::optional<THPObjectPtr> autogradFunction() const override;

  void writeScalars(std::ostream& out) const override;  // 覆盖基类方法，将标量参数写入流中
  void lint_python() const override;  // 覆盖基类方法，对 Python 进行静态分析
};

} // namespace torch::jit
```