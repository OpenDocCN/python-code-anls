# `.\pytorch\torch\csrc\jit\runtime\static\te_wrapper.h`

```
#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>  // 引入TensorExpr的代码生成相关头文件
#include <torch/csrc/jit/tensorexpr/ir.h>  // 引入TensorExpr的中间表示相关头文件
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>  // 引入TensorExpr的IR简化相关头文件
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>  // 引入TensorExpr的LLVM代码生成相关头文件
#include <torch/csrc/jit/tensorexpr/loopnest.h>  // 引入TensorExpr的循环嵌套相关头文件

namespace torch::jit {

class TEWrapper {
 public:
  TEWrapper() = default;  // 默认构造函数，初始化TEWrapper对象
  void call(const std::vector<void*>& args);  // 定义调用函数，接收参数向量的指针

  template <typename ExpectedType>
  bool checkInput(const at::Tensor& t) {  // 模板函数，检查输入张量是否连续且数据类型符合期望
#ifdef TORCH_ENABLE_LLVM
    return t.is_contiguous() && t.dtype().Match<ExpectedType>();  // 如果支持LLVM，则检查张量是否连续且数据类型匹配
#else
    return false;  // 如果不支持LLVM，则返回false
#endif
  }

#ifdef TORCH_ENABLE_LLVM
  void update(std::unique_ptr<tensorexpr::LLVMCodeGen>&& cg_);  // 如果支持LLVM，则定义更新函数，接收一个移动语义的LLVMCodeGen指针
#endif

 private:
#ifdef TORCH_ENABLE_LLVM
  std::unique_ptr<tensorexpr::LLVMCodeGen> cg;  // 如果支持LLVM，则定义一个唯一指针类型的LLVMCodeGen对象
#endif
};

std::shared_ptr<TEWrapper> createDiv();  // 创建TEWrapper对象的共享指针，用于执行除法操作
std::shared_ptr<TEWrapper> createLogit();  // 创建TEWrapper对象的共享指针，用于执行logit操作
std::shared_ptr<TEWrapper> createRelu();  // 创建TEWrapper对象的共享指针，用于执行ReLU操作
std::shared_ptr<TEWrapper> createTanh();  // 创建TEWrapper对象的共享指针，用于执行双曲正切操作
std::shared_ptr<TEWrapper> createSigmoid();  // 创建TEWrapper对象的共享指针，用于执行sigmoid操作
std::shared_ptr<TEWrapper> createSignedLog1p();  // 创建TEWrapper对象的共享指针，用于执行有符号log1p操作
std::shared_ptr<TEWrapper> createClamp();  // 创建TEWrapper对象的共享指针，用于执行clamp操作
std::shared_ptr<TEWrapper> createClampNanToNum();  // 创建TEWrapper对象的共享指针，用于执行clamp_nan_to_num操作

} // namespace torch::jit
```