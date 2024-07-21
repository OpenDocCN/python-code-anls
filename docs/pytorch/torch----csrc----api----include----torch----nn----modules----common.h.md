# `.\pytorch\torch\csrc\api\include\torch\nn\modules\common.h`

```py
#pragma once

/// This macro enables a module with default arguments in its forward method
/// to be used in a Sequential module.
///
/// Example usage:
///
/// Let's say we have a module declared like this:
/// ```
/// struct MImpl : torch::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   torch::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return torch::tensor(a + b + c);
///   }
///  private:
///   int value;
/// };
/// TORCH_MODULE(M);
/// ```py
///
/// If we try to use it in a Sequential module and run forward:
/// ```
/// torch::nn::Sequential seq(M(1));
/// seq->forward(1);
/// ```py
///
/// We will receive the following error message:
/// ```
/// MImpl's forward() method expects 3 argument(s), but received 1.
/// If MImpl's forward() method has default arguments, please make sure
/// the forward() method is declared with a corresponding
/// `FORWARD_HAS_DEFAULT_ARGS` macro.
/// ```py
///
/// The right way to fix this error is to use the `FORWARD_HAS_DEFAULT_ARGS`
/// macro when declaring the module:
/// ```
/// struct MImpl : torch::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   torch::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return torch::tensor(a + b + c);
///   }
///  protected:
///   /*
///   NOTE: looking at the argument list of `forward`:
///   `forward(int a, int b = 2, double c = 3.0)`
///   we saw the following default arguments:
///   ----------------------------------------------------------------
///   0-based index of default |         Default value of arg
///   arg in forward arg list  |  (wrapped by `torch::nn::AnyValue()`)
///   ----------------------------------------------------------------
///               1            |       torch::nn::AnyValue(2)
///               2            |       torch::nn::AnyValue(3.0)
///   ----------------------------------------------------------------
///   Thus we pass the following arguments to the `FORWARD_HAS_DEFAULT_ARGS`
///   macro:
///   */
///   FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(2)}, {2,
///   torch::nn::AnyValue(3.0)})
///  private:
///   int value;
/// };
/// TORCH_MODULE(M);
/// ```py

/// Macro definition for handling default arguments in a forward method.
#define FORWARD_HAS_DEFAULT_ARGS(...)                                         \
  template <typename ModuleType, typename... ArgumentTypes>                   \
  friend struct torch::nn::AnyModuleHolder;                                   \
  /// Overrides the default implementation to indicate that the module supports
  /// default arguments.
  bool _forward_has_default_args() override {                                 \
    return true;                                                              \
  }                                                                           \
  /// Overrides the default implementation to specify the number of required
  /// arguments for the forward method.
  unsigned int _forward_num_required_args() override {                        \
    // 定义一个宏，接收一组参数，形成一个 std::pair 数组，然后返回数组中第一个元素的第一个值
    std::pair<unsigned int, torch::nn::AnyValue> args_info[] = {__VA_ARGS__}; \
    return args_info[0].first;                                                \
    
    // 实现一个函数 _forward_populate_default_args，接收一个右值引用的 std::vector<torch::nn::AnyValue> 参数，
    // 返回一个填充了默认参数的 std::vector<torch::nn::AnyValue>
    std::vector<torch::nn::AnyValue> _forward_populate_default_args(            \
        std::vector<torch::nn::AnyValue>&& arguments) override {                \
      // 定义一个 std::pair 数组，接收宏定义中传入的参数
      std::pair<unsigned int, torch::nn::AnyValue> args_info[] = {__VA_ARGS__}; \
      // 计算所有参数的数量
      unsigned int num_all_args = std::rbegin(args_info)->first + 1;            \
      // 内部断言，确保 arguments 的大小在 _forward_num_required_args() 和 num_all_args 之间
      TORCH_INTERNAL_ASSERT(                                                    \
          arguments.size() >= _forward_num_required_args() &&                   \
          arguments.size() <= num_all_args);                                    \
      // 将 arguments 移动到 ret 中
      std::vector<torch::nn::AnyValue> ret = std::move(arguments);              \
      // 预留足够的空间，以容纳所有参数
      ret.reserve(num_all_args);                                                \
      // 遍历 args_info，如果当前参数索引大于 ret 的大小减一，则将默认参数加入 ret
      for (auto& arg_info : args_info) {                                        \
        if (arg_info.first > ret.size() - 1)                                    \
          ret.emplace_back(std::move(arg_info.second));                         \
      }                                                                         \
      // 返回填充好默认参数的 ret
      return ret;                                                               \
    }
```