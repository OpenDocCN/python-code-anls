# `.\pytorch\test\cpp_api_parity\sample_module.py`

```
import torch
# 导入 PyTorch 库

"""
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `forward` / `backward`
is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `forward` / `backward`
is different from the C++ equivalent.
"""
# `SampleModule` 被用于 `test_cpp_api_parity.py`，测试 `torch.nn.Module` 子类的 Python/C++ API 的一致性。
# 当 `SampleModule.has_parity` 为 true 时，`forward` / `backward` 的行为与 C++ 版本相同。
# 当 `SampleModule.has_parity` 为 false 时，`forward` / `backward` 的行为与 C++ 版本不同。

class SampleModule(torch.nn.Module):
    def __init__(self, has_parity, has_submodule):
        super().__init__()
        self.has_parity = has_parity
        if has_submodule:
            self.submodule = SampleModule(self.has_parity, False)
        # 如果有子模块，创建一个新的 `SampleModule` 实例作为子模块，继承当前对象的 `has_parity` 属性。

        self.has_submodule = has_submodule
        # 设置是否有子模块的标志

        self.register_parameter("param", torch.nn.Parameter(torch.empty(3, 4)))
        # 注册参数 `param`，是一个 3x4 的空张量作为参数

        self.reset_parameters()
        # 调用重置参数的函数

    def reset_parameters(self):
        with torch.no_grad():
            self.param.fill_(1)
        # 使用 `torch.no_grad()` 上下文管理器填充 `param` 参数张量的所有元素为 1

    def forward(self, x):
        submodule_forward_result = (
            self.submodule(x) if hasattr(self, "submodule") else 0
        )
        # 如果存在子模块，则调用子模块的 `forward` 方法，否则返回 0

        if self.has_parity:
            return x + self.param * 2 + submodule_forward_result
        else:
            return x + self.param * 4 + submodule_forward_result + 3
        # 根据 `has_parity` 属性选择不同的前向传播计算方式

torch.nn.SampleModule = SampleModule
# 将 `SampleModule` 类注册到 `torch.nn` 模块中

SAMPLE_MODULE_CPP_SOURCE = """\n
namespace torch {
namespace nn {
struct C10_EXPORT SampleModuleOptions {
  SampleModuleOptions(bool has_parity, bool has_submodule) : has_parity_(has_parity), has_submodule_(has_submodule) {}

  TORCH_ARG(bool, has_parity);
  TORCH_ARG(bool, has_submodule);
};

struct C10_EXPORT SampleModuleImpl : public torch::nn::Cloneable<SampleModuleImpl> {
  explicit SampleModuleImpl(SampleModuleOptions options) : options(std::move(options)) {
    if (options.has_submodule()) {
      submodule = register_module(
        "submodule",
        std::make_shared<SampleModuleImpl>(SampleModuleOptions(options.has_parity(), false)));
    }
    reset();
  }
  void reset() {
    param = register_parameter("param", torch::ones({3, 4}));
  }
  torch::Tensor forward(torch::Tensor x) {
    return x + param * 2 + (submodule ? submodule->forward(x) : torch::zeros_like(x));
  }
  SampleModuleOptions options;
  torch::Tensor param;
  std::shared_ptr<SampleModuleImpl> submodule{nullptr};
};

TORCH_MODULE(SampleModule);
} // namespace nn
} // namespace torch
"""
# 定义 C++ 源码字符串，实现 `SampleModule` 的 C++ API

module_tests = [
    dict(
        module_name="SampleModule",
        desc="has_parity",
        constructor_args=(True, True),
        cpp_constructor_args="torch::nn::SampleModuleOptions(true, true)",
        input_size=(3, 4),
        cpp_input_args=["torch::randn({3, 4})"],
        has_parity=True,
    ),
    dict(
        fullname="SampleModule_no_parity",
        constructor=lambda: SampleModule(has_parity=False, has_submodule=True),
        cpp_constructor_args="torch::nn::SampleModuleOptions(false, true)",
        input_size=(3, 4),
        cpp_input_args=["torch::randn({3, 4})"],
        has_parity=False,
    ),
]
# 定义用于测试的模块示例和参数
    # 这段代码用于测试设置 `test_cpp_api_parity=False` 标志时跳过 C++ API 的一致性测试
    # 相应地（否则此测试会运行并抛出一致性错误）。
    dict(
        # 指定测试模块的完整名称，用于标识测试用例
        fullname="SampleModule_THIS_TEST_SHOULD_BE_SKIPPED",
        # 定义一个函数，用于构造被测试模块的实例
        constructor=lambda: SampleModule(False, True),
        # 指定 C++ 构造函数的参数表达式
        cpp_constructor_args="torch::nn::SampleModuleOptions(false, true)",
        # 定义输入数据的大小
        input_size=(3, 4),
        # 指定用于 C++ 接口的输入参数表达式
        cpp_input_args=["torch::randn({3, 4})"],
        # 控制是否进行 C++ API 与 Python API 的一致性测试，这里设置为 False 表示不进行测试
        test_cpp_api_parity=False,
    ),
# 定义一个空列表
data = []

# 定义一个空字典
results = {}

# 遍历数据列表中的每个元素
for item in data:
    # 如果元素在结果字典中存在
    if item in results:
        # 将该元素的计数加一
        results[item] += 1
    else:
        # 否则将该元素添加到结果字典，并设置计数为1
        results[item] = 1

# 将结果字典中的键值对按照键的字母顺序排序，并返回排序后的结果
sorted_results = {k: results[k] for k in sorted(results.keys())}
```