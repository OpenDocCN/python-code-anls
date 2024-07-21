# `.\pytorch\test\cpp\lite_interpreter_runtime\test_lite_interpreter_runtime.cpp`

```py
// 包含 ATen 库中的函数定义头文件
#include <ATen/Functions.h>
// 包含 ATen 库中张量操作相关的头文件
#include <aten/src/ATen/TensorOperators.h>
// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>
// 包含 JIT 测试工具函数的头文件
#include <test/cpp/jit/test_utils.h>
// 包含 Torch 自动求导库中变量工厂生成的头文件
#include <torch/csrc/autograd/generated/variable_factories.h>
// 包含 Torch JIT API 中模块定义的头文件
#include <torch/csrc/jit/api/module.h>
// 包含 Torch JIT 前端解析器的头文件
#include <torch/csrc/jit/frontend/resolver.h>
// 包含 Torch 移动端模型导入的头文件
#include <torch/csrc/jit/mobile/import.h>
// 包含 Torch 移动端模型模块的头文件
#include <torch/csrc/jit/mobile/module.h>

// 包含无序集合的标准库
#include <unordered_set>

// Torch JIT 移动端命名空间
namespace torch {
namespace jit {
namespace mobile {

// 测试用例：运行时测试 LoadAndForward
TEST(RunTimeTest, LoadAndForward) {
  // 检查加载的模型文件路径为当前文件所在路径 + "sequence.ptl"
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("sequence.ptl");

  // 定义一个模块变量，加载序列化的移动端模型
  Module bc = _load_for_mobile(testModelFile);

  // 查找模块中的 forward 方法
  auto forward_method = bc.find_method("forward");

  // 准备输入值，这里使用张量 1 构成的 IValue 向量
  std::vector<c10::IValue> input{c10::IValue(at::tensor(1))};

  // 执行模型的 forward 方法，获取结果
  const auto result = bc.forward(input);

  // 期望的结果为张量 4 构成的 IValue
  const auto expected_result = c10::IValue(at::tensor(4));

  // 断言计算结果与期望结果相等
  ASSERT_EQ(result, expected_result);
}

// 测试用例：运行时测试 Delegate
TEST(RunTimeTest, Delegate) {
  // 获取当前文件路径，并拼接上 "delegate_test.ptl"
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("delegate_test.ptl");

  // 加载序列化的移动端模型
  auto mlm = _load_for_mobile(testModelFile);

  // 准备输入值，这里是两个张量组成的 IValue 向量
  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * at::ones({}));
  inputs.emplace_back(1.0 * at::ones({}));

  // 执行模型的 forward 方法，获取结果
  auto mres = mlm.forward(inputs);

  // 断言结果张量与期望的 3 倍张量相等
  AT_ASSERT(mres.toTensor().equal(3 * at::ones({})));
}

} // namespace mobile
} // namespace jit
} // namespace torch
TEST(RunTimeTest, DelegateException) {
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  /*
   * Model: delegated_submodule_with_debug_info.ptl
   * Model structure:
   * def AA(..):
   *   def forward(self, x, y):
   *     return x + y
   *
   * def A(..):
   *   def __init__(..):
   *     self.AA0 = AA()
   *   def forward(self, x, y):
   *     return self.AA0.forward(x, y) + 3
   *
   * def B(..):
   *   def forward(self, x):
   *     return x + 2
   *
   * def C(..):
   *   def __init__(..):
   *     self.A0 = A()
   *     self.B0 = B()
   *   def forward(self, x, y):
   *     return self.A0.forward(x, y) + self.B0.forward(x)
   *
   * std::vector<IValue> inputs;
   * inputs.emplace_back(torch::rand({2, 4}));
   * inputs.emplace_back(torch::rand({13, 9}));
   * Run with inputs and expect exception
   * Error stack trace will look like this:
   * Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA)
   * Traceback of TorchScript (most recent call last):
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + self.B0.forward(x)
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 5, in FunctionName_UNKNOWN
   *                typed_inputs: List[Any] = [x, y, ]
   *                if self.__backend.is_available() :
   *                  _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                  assert isinstance(_0, Tensor)
   *                  return _0
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.AA0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   *
   */
  // 将模型文件名添加到测试模型文件路径
  testModelFile.append("delegated_submodule_with_debug_info.ptl");
  // 加载移动端模型
  auto mlm = _load_for_mobile(testModelFile);
  // 创建输入向量
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  // 定义错误模式字符串
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.A0(backend_with_compiler_demoLoweredModule)::forward.AA0(AA)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>
  // 省略部分内容，不完全注释错误模式的其余部分
  ```
    # 定义类中的前向传播函数，接受两个参数 x 和 y
    def forward(self, x, y):
      # 返回 A0 对象的前向传播结果与 B0 对象的前向传播结果之和
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- 指示在这里调用了 A0 对象的前向传播方法

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- 指示在这里执行了后端的前向传播操作
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    # 定义类中的前向传播函数，接受两个参数 x 和 y
    def forward(self, x, y):
      # 返回 AA0 对象的前向传播结果加上常数 3
      return self.AA0.forward(x, y) + 3
             ~~~~~~~~~~~~~~~~ <--- 指示在这里调用了 AA0 对象的前向传播方法

  File "<string>", line 3, in forward

    # 定义类中的前向传播函数，接受两个参数 x 和 y
    def forward(self, x, y):
      # 返回参数 x 和 y 的加法结果
      return x + y
             ~~~~~ <--- 指示在这里执行了参数 x 和 y 的加法运算
  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}
} // namespace mobile
} // namespace jit
} // namespace torch
```