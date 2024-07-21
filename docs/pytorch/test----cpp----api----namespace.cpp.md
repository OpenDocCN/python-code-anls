# `.\pytorch\test\cpp\api\namespace.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/torch.h>  // 引入 PyTorch 的头文件

struct Node {};  // 定义一个空结构体 Node

// 如果 `torch::autograd::Note` 泄漏到了全局命名空间中，会导致以下编译错误：
// ```
// void NotLeakingSymbolsFromTorchAutogradNamespace_test_func(Node *node) {}
//                                                            ^
// error: reference to `Node` is ambiguous
// ```
void NotLeakingSymbolsFromTorchAutogradNamespace_test_func(Node* node) {}  // 声明一个函数，接收 Node 指针参数，用于测试命名空间符号泄漏情况

TEST(NamespaceTests, NotLeakingSymbolsFromTorchAutogradNamespace) {
  // 检查我们没有从 `torch::autograd` 命名空间泄漏符号到全局命名空间
  NotLeakingSymbolsFromTorchAutogradNamespace_test_func(nullptr);  // 调用上述声明的函数进行测试
}
```