# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\fbgemm_utils.cpp`

```py
// 定义宏以仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含张量和上下文的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>

// 包含自定义类的 Torch 头文件
#include <torch/custom_class.h>

// 包含量化稀疏模块的 FBGEMM 实用工具函数的头文件
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
// 包含量化稀疏模块的打包参数的头文件
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
// 包含量化稀疏模块的 QNNPACK 实用工具函数的头文件
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

namespace ao {
namespace sparse {

// 注册线性层打包参数的函数
int register_linear_params() {
  // 使用静态注册方式注册 LinearPackedParamsBase 类到 Torch 框架中的 sparse 命名空间
  static auto register_linear_params =
      torch::selective_class_<LinearPackedParamsBase>(
          "sparse", TORCH_SELECTIVE_CLASS("LinearPackedParamsBase"))
          // 定义对象序列化和反序列化方法
          .def_pickle(
              // __getstate__ 函数，将参数对象序列化为特定类型的状态
              [](const c10::intrusive_ptr<LinearPackedParamsBase>& params)
                  -> BCSRSerializationType {
                return params->serialize();
              },
              // __setstate__ 函数，根据状态反序列化为线性打包参数对象
              [](BCSRSerializationType state)
                  -> c10::intrusive_ptr<LinearPackedParamsBase> {
#ifdef USE_FBGEMM
                // 如果使用 FBGEMM 引擎，则返回 FBGEMM 实现的打包线性权重对象
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM) {
                  return PackedLinearWeight::deserialize(state);
                }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
                // 如果使用 QNNPACK 引擎，则返回 QNNPACK 实现的打包线性权重对象
                if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
                  return PackedLinearWeightQnnp::deserialize(state);
                }
#endif // USE_PYTORCH_QNNPACK
                // 如果引擎未知，则抛出错误
                TORCH_CHECK(false, "Unknown qengine");
              });

  // 返回值为零，表示注册成功
  return 0;
}

// 匿名命名空间，用于静态注册线性层参数
namespace {
// 静态变量 linear_params，调用注册线性参数函数完成静态初始化
static C10_UNUSED auto linear_params = register_linear_params();
}  // namespace

}}  // namespace ao::sparse
```