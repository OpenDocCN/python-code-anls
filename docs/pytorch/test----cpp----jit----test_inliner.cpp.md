# `.\pytorch\test\cpp\jit\test_inliner.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/csrc/jit/api/compilation_unit.h>  // 引入 Torch 的 JIT 编译单元 API 头文件
#include <torch/csrc/jit/api/module.h>  // 引入 Torch 的 JIT 模块 API 头文件
#include <torch/csrc/jit/passes/inliner.h>  // 引入 Torch 的 JIT 内联处理头文件
#include <torch/csrc/jit/testing/file_check.h>  // 引入 Torch 的 JIT 文件检查头文件

const auto testSource = R"JIT(
def foo1(x):
    print("one")
    return x

def foo2(x):
    print("two")
    return foo1(x)

def foo3(x):
    print("three")
    return foo2(x)
)JIT";  // 定义一个测试用的 Torch JIT 脚本

namespace torch {
namespace jit {

using namespace testing;  // 使用 Torch 的测试工具命名空间

struct InlinerGuard {
  explicit InlinerGuard(bool shouldInline)
      : oldState_(getInlineEverythingMode()) {
    getInlineEverythingMode() = shouldInline;  // 设置是否全局内联的状态
  }

  ~InlinerGuard() {
    getInlineEverythingMode() = oldState_;  // 恢复全局内联的状态
  }

  bool oldState_;  // 保存旧的全局内联状态
};

TEST(InlinerTest, Basic) {
  // 禁用自动内联以便手动测试
  InlinerGuard guard(/*shouldInline=*/false);

  CompilationUnit cu(testSource);  // 使用测试脚本构建编译单元对象
  auto& fn = cu.get_function("foo3");  // 获取函数 "foo3"

  auto g = toGraphFunction(fn).graph();  // 将函数转换为图形函数，并获取其图形表示
  Inline(*g);  // 对图形进行内联处理
  FileCheck().check_count("prim::Print", 3)->run(*g);  // 运行文件检查，检查打印操作出现的次数是否为 3 次
}

} // namespace jit
} // namespace torch
```