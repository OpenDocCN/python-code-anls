# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\functional.h`

```
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <functional>
#include <utility>

namespace torch {
namespace nn {

/// Wraps a function in a `Module`.
///
/// The `Functional` module allows wrapping an arbitrary function or function
/// object in an `nn::Module`. This is primarily handy for usage in
/// `Sequential`.
///
/// \rst
/// .. code-block:: cpp
///
///   Sequential sequential(
///     Linear(3, 4),
///     Functional(torch::relu),
///     BatchNorm1d(3),
///     Functional(torch::elu, /*alpha=*/1));
/// \endrst
///
/// While a `Functional` module only accepts a single `Tensor` as input, it is
/// possible for the wrapped function to accept further arguments. However,
/// these have to be bound *at construction time*. For example, if
/// you want to wrap `torch::leaky_relu`, which accepts a `slope` scalar as its
/// second argument, with a particular value for its `slope` in a `Functional`
/// module, you could write
///
/// \rst
/// .. code-block:: cpp
///
///   Functional(torch::leaky_relu, /*slope=*/0.5)
/// \endrst
///
/// The value of `0.5` is then stored within the `Functional` object and
/// supplied to the function call at invocation time. Note that such bound
/// values are evaluated eagerly and stored a single time. See the documentation
/// of [std::bind](https://en.cppreference.com/w/cpp/utility/functional/bind)
/// for more information on the semantics of argument binding.
///
/// \rst
/// .. attention::
///   After passing any bound arguments, the function must accept a single
///   tensor and return a single tensor.
/// \endrst
///
/// Note that `Functional` overloads the call operator (`operator()`) such that
/// you can invoke it with `my_func(...)`.
class TORCH_API FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<Tensor(Tensor)>;

  /// Constructs a `Functional` from a function object.
  explicit FunctionalImpl(Function function);

  template <
      typename SomeFunction,
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) > 0)>>
  explicit FunctionalImpl(SomeFunction original_function, Args&&... args)
      // NOLINTNEXTLINE(modernize-avoid-bind)
      : function_(std::bind(
            original_function,
            /*input=*/std::placeholders::_1,
            std::forward<Args>(args)...)) {
    // std::bind is normally evil, but (1) gcc is broken w.r.t. handling
    // parameter pack expansion in lambdas and (2) moving parameter packs into
    // a lambda only works with C++14, so std::bind is the more move-aware

    // 此处使用 std::bind 将 original_function 和附加参数绑定到成员变量 function_
    // std::bind 可以用来创建一个函数对象，将输入参数绑定到函数中的占位符
  }

 private:
  /// The wrapped function stored with bound arguments.
  Function function_;  ///< 存储被包装的函数对象及其绑定的参数
};

} // namespace nn
} // namespace torch
    // 结束类定义
  }

  // 重置函数的声明，继承自基类并进行重写
  void reset() override;

  /// 将“Functional”模块漂亮地打印到给定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 将输入张量传递给底层（绑定的）函数对象。
  Tensor forward(Tensor input);

  /// 调用 forward(input)。
  Tensor operator()(Tensor input);

  // 检查该模块是否可序列化的声明，继承自基类
  bool is_serializable() const override;

 private:
  // 私有成员变量，用于保存具体的函数对象
  Function function_;
};

/// `FunctionalImpl` 的 `ModuleHolder` 子类。
/// 查看 `FunctionalImpl` 类的文档，了解其提供的方法。
/// 或者查看 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(Functional);

} // namespace nn
} // namespace torch
```