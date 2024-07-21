# `.\pytorch\test\mobile\nnc\test_registry.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/csrc/jit/mobile/nnc/registry.h>  // 引入移动端 NNCompiler 的注册表头文件

namespace torch {  // 命名空间 torch
namespace jit {  // 命名空间 jit
namespace mobile {  // 命名空间 mobile
namespace nnc {  // 命名空间 nnc

extern "C" {
// 定义生成的汇编内核函数，返回固定整数 1
int generated_asm_kernel_foo(void**) {
  return 1;
}

// 定义生成的汇编内核函数，返回固定整数 2
int generated_asm_kernel_bar(void**) {
  return 2;
}
} // extern "C"

// 注册生成的汇编内核函数 "foo:v1:VERTOKEN" 到内核注册表
REGISTER_NNC_KERNEL("foo:v1:VERTOKEN", generated_asm_kernel_foo)

// 注册生成的汇编内核函数 "bar:v1:VERTOKEN" 到内核注册表
REGISTER_NNC_KERNEL("bar:v1:VERTOKEN", generated_asm_kernel_bar)

// 测试用例：MobileNNCRegistryTest
TEST(MobileNNCRegistryTest, FindAndRun) {
  // 获取注册表中 "foo:v1:VERTOKEN" 对应的内核
  auto foo_kernel = registry::get_nnc_kernel("foo:v1:VERTOKEN");
  // 断言执行 foo_kernel 返回结果为 1
  EXPECT_EQ(foo_kernel->execute(nullptr), 1);

  // 获取注册表中 "bar:v1:VERTOKEN" 对应的内核
  auto bar_kernel = registry::get_nnc_kernel("bar:v1:VERTOKEN");
  // 断言执行 bar_kernel 返回结果为 2
  EXPECT_EQ(bar_kernel->execute(nullptr), 2);
}

// 测试用例：MobileNNCRegistryTest，测试不存在的内核
TEST(MobileNNCRegistryTest, NoKernel) {
  // 断言注册表中没有 "missing" 对应的内核
  EXPECT_EQ(registry::has_nnc_kernel("missing"), false);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```