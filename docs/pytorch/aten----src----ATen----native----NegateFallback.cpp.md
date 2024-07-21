# `.\pytorch\aten\src\ATen\native\NegateFallback.cpp`

```
// 定义编译器标识符，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含头文件，提供数学位的回退实现
#include <ATen/native/MathBitsFallback.h>

// 包含数学位的落空列表头文件
#include <ATen/native/MathBitFallThroughLists.h>

// 进入 at::native 命名空间
namespace at::native {

// 定义 NegFallback 结构体，继承自 MathOpFallback
struct NegFallback : MathOpFallback {
  // 构造函数，设置 DispatchKey 为 Negative，描述为 "negation"
  NegFallback() : MathOpFallback(DispatchKey::Negative, "negation") {}

  // 实现虚函数，检查张量是否为负数
  bool is_bit_set(const Tensor& tensor) override {
    return tensor.is_neg();
  }
};

// 定义静态函数 negationFallback
static void negationFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // 创建 NegFallback 对象
  NegFallback object;
  // 调用对象的回退实现方法
  object.fallback_impl(op, dispatch_keys, stack);
}

// 实现 TORCH 库的 Negative 部分
TORCH_LIBRARY_IMPL(_, Negative, m) {
  // 注册 negationFallback 函数为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&negationFallback>());
}

// 实现 aten 库的 Negative 部分
TORCH_LIBRARY_IMPL(aten, Negative, m) {
  // 使用 makeFallthrough() 将以下函数设置为直接回退到下一个实现
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("neg_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_neg", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.Tensor", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_int", torch::CppFunction::makeFallthrough());

  // 注册测试元数据函数，直接回退到下一个实现
  m.impl("_has_same_storage_numel", torch::CppFunction::makeFallthrough());
  m.impl("_new_zeros_with_same_feature_meta", torch::CppFunction::makeFallthrough());

  // 线性代数函数的直接回退实现
  m.impl("linalg_solve_triangular", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular.out", torch::CppFunction::makeFallthrough());
  m.impl("linalg_svd", torch::CppFunction::makeFallthrough());
  m.impl("linalg_svd.U", torch::CppFunction::makeFallthrough());

  // 注册视图函数和张量实用工具的本地函数
  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m)
}

} // 结束 at::native 命名空间
```