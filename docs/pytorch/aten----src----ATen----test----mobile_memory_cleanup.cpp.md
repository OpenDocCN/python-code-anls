# `.\pytorch\aten\src\ATen\test\mobile_memory_cleanup.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/csrc/jit/passes/xnnpack_rewrite.h>  // 引入 XNNPACK 重写相关的头文件
#include <torch/torch.h>  // 引入 PyTorch 库的头文件

using namespace torch::jit;  // 使用 torch::jit 命名空间

#ifdef USE_XNNPACK  // 如果定义了 USE_XNNPACK 宏，则编译以下代码

TEST(MemoryCleanUp, NoErrorWithoutRelease) {  // 定义名为 MemoryCleanUp 的测试用例，测试无释放权重时是否没有错误
  Module m("m");  // 创建名为 m 的模块对象
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);  // 注册名为 weight 的参数张量，全为 1，不需要梯度
  m.register_parameter("bias", torch::ones({20}), false);  // 注册名为 bias 的参数张量，全为 1，不需要梯度
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");  // 定义模块的前向传播方法，使用 torch._convolution 函数进行卷积运算
  m.eval();  // 设置模块为评估模式
  auto m_optimized = optimizeForMobile(m);  // 对模块进行移动端优化
  std::stringstream ss;  // 创建字符串流 ss
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_NO_THROW(m_optimized.save(ss));  // 断言优化后的模块保存到字符串流不会抛出异常
}

TEST(MemoryCleanUp, UnpackError) {  // 定义名为 MemoryCleanUp 的测试用例，测试解包时是否抛出错误
  at::globalContext().setReleaseWeightsWhenPrepacking(true);  // 设置全局上下文，在预打包时释放权重
  Module m("m");  // 创建名为 m 的模块对象
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);  // 注册名为 weight 的参数张量，全为 1，不需要梯度
  m.register_parameter("bias", torch::ones({20}), false);  // 注册名为 bias 的参数张量，全为 1，不需要梯度
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");  // 定义模块的前向传播方法，使用 torch._convolution 函数进行卷积运算
  m.eval();  // 设置模块为评估模式
  auto m_optimized = optimizeForMobile(m);  // 对模块进行移动端优化
  std::stringstream ss;  // 创建字符串流 ss
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(m_optimized.save(ss));  // 断言优化后的模块保存到字符串流会抛出异常
}

#endif  // 结束 USE_XNNPACK 宏条件编译
```