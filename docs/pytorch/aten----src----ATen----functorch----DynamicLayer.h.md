# `.\pytorch\aten\src\ATen\functorch\DynamicLayer.h`

```
// 版权声明和许可信息
// 本源代码根据 BSD 风格许可证授权，许可条款详见源代码根目录下的 LICENSE 文件。

#pragma once
// 包含头文件，引入所需依赖
#include <ATen/functorch/Macros.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/function_schema.h>
#include <c10/util/Optional.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/functorch/Interpreter.h>
#include <ATen/functorch/VmapInterpreter.h>
#include <ATen/functorch/ADInterpreters.h>
#include <ATen/functorch/FunctionalizeInterpreter.h>

// 前向声明
namespace c10 { struct AutogradMetaInterface; }

namespace at::functorch  {

// 本文件实现了 functorch 的解释器堆栈。
// 在阅读下文之前，请查看 NOTE: [functorch interpreter stack]。
//
// 注意：functorch 解释器堆栈也被称为：
// - "dynamic layer stack" -- 曾用名 "dynamic layer" 指 "interpreter"。
// - "functorch mode stack"。可以将每个 functorch 变换视为一个 "mode"，
//   functorch 实现了一个 "mode stack"，其中的 mode 可以任意组合。

// DynamicLayer 基本上与 Interpreter 是相同的东西。
// 它代表一个 functorch 变换，包含一个 Interpreter，
// Interpreter 包含与变换相关的元数据和执行变换的指令。
//
// TODO: 我们可以通过 Interpreter 来替换 DynamicLayer，
// 但为了兼容性暂时保留 DynamicLayer，以避免需要重构大量调用点...
struct TORCH_API DynamicLayer {
  explicit DynamicLayer(
      TransformType transform_type,
      int64_t layerId,
      optional<c10::SymInt> batchSize = nullopt,
      optional<RandomnessType> randomness = nullopt,
      optional<bool> prev_grad_mode = nullopt,
      optional<bool> pre_fwd_grad_mode = nullopt,
      optional<bool> functionalize_add_back_views = nullopt);

  TransformType key() const;
  int64_t layerId() const;

  const Interpreter& interpreter() const { return interpreter_; }
  Interpreter& interpreter() { return interpreter_; }

  // 仅对于 vmap 有效
  c10::SymInt batchSize() const;
  RandomnessType randomness() const;

 private:
  Interpreter interpreter_;
};

// 初始化并推入一个 DynamicLayer 到堆栈中
TORCH_API int64_t initAndPushDynamicLayer(
    TransformType transform_type,
    optional<c10::SymInt> batch_size = nullopt,
    optional<RandomnessType> randomness = nullopt,
    optional<bool> prev_grad_mode = nullopt,
    optional<bool> prev_fwd_grad_mode = nullopt,
    optional<bool> functionalize_add_back_views = nullopt);

// 弹出堆栈中的 DynamicLayer 并删除其元数据
TORCH_API DynamicLayer popDynamicLayerAndDeleteMetadata();

// 获取当前可能存在的 DynamicLayer（如果有的话）
TORCH_API std::optional<DynamicLayer> maybeCurrentDynamicLayer();

// 获取动态层堆栈的引用
TORCH_API const std::vector<DynamicLayer>& getDynamicLayerStack();

// 设置动态层堆栈
TORCH_API void setDynamicLayerStack(const std::vector<DynamicLayer>& stack);
// 设置是否包括动态层的前后键
TORCH_API void setDynamicLayerFrontBackKeysIncluded(bool included);

// 注意事项：生命周期处理和词法作用域变换
// functorch 变换是词法作用域的。
// 对于给定的层级，我们存储一个“生命周期处理”，它是一个布尔值，告诉我们该层级的变换是否处于活动状态。
//
// functorch 的 TensorWrapper（用于梯度变换）存储一个生命周期处理。
// 如果一个 TensorWrapper 从变换的作用域中逃逸出来，它必须知道它逃逸了；它可以通过查询生命周期处理来判断。
TORCH_API const std::shared_ptr<bool>& getLifeHandleForLevel(int64_t level);

// 返回操作符是否为原地操作。一个操作符是原地的，如果：
// 1. 第一个参数是一个 Tensor，并且正在被写入
// 2. 第一个参数正在被返回
// 3. 没有其他参数被别名引用
// 以下是一个原地操作的示例：
// add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
TORCH_API bool isInplaceOp(const c10::FunctionSchema& schema);

// 给定未包装输入的索引和函数模式，返回应保持未包装的任何输出的索引
TORCH_API std::optional<size_t> findAliasedOutput(const FunctionSchema& schema, const int64_t immutable_input);

// 如果张量已死亡，则解包它
TORCH_API Tensor unwrapIfDead(const Tensor& tensor);
TORCH_API bool isDeadTensorWrapper(const Tensor& tensor);

// 漂亮的打印器
TORCH_API std::ostream& operator<<(std::ostream& os, const DynamicLayer& layer);
TORCH_API std::ostream& operator<<(std::ostream& os, const std::vector<DynamicLayer>& dynamicLayerStack);

// 当 functorch 变换处于活动状态时，默认禁用 torch.autograd.function._SingleLevelFunction。
// 下面两个 API 用于启用它。这些不是用户可见的 API。我们将来可以删除它，但在调试时非常有用，
// 当 autograd.Function <> functorch 交互出现问题时，它可以导致大声的错误。
TORCH_API void setSingleLevelAutogradFunctionAllowed(bool allowed);
TORCH_API bool getSingleLevelAutogradFunctionAllowed();

// 当 functorch 梯度变换处于活动状态时，禁用 Tensor.requires_grad_()。
// 这两个函数是控制这一行为的机制。
TORCH_API void setInplaceRequiresGradAllowed(bool allowed);
TORCH_API bool getInplaceRequiresGradAllowed();

// 弹出动态层
TORCH_API DynamicLayer popDynamicLayer();
// 推入动态层
TORCH_API int64_t pushDynamicLayer(DynamicLayer&& layer);

} // namespace at::functorch
```