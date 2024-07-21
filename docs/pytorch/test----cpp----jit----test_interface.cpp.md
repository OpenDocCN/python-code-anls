# `.\pytorch\test\cpp\jit\test_interface.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含用于测试的实用工具函数的头文件
#include <test/cpp/jit/test_utils.h>

// 包含 ATen 库的头文件
#include <ATen/core/qualified_name.h>

// 包含 Torch 的 JIT 前端解析器头文件
#include <torch/csrc/jit/frontend/resolver.h>

// 包含 Torch 的模型导入相关头文件
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_source.h>

// 包含 Torch 核心功能的头文件
#include <torch/torch.h>

// 定义 torch 命名空间下的 jit 命名空间
namespace torch {
namespace jit {

// 定义静态常量，包含子模块方法的源代码
static const std::vector<std::string> subMethodSrcs = {R"JIT(
def one(self, x: Tensor, y: Tensor) -> Tensor:
    return x + y + 1

def forward(self, x: Tensor) -> Tensor:
    return x
)JIT"};

// 定义父模块的 forward 方法的源代码
static const std::string parentForward = R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return self.subMod.forward(x)
)JIT";

// 定义模块接口的源代码
static constexpr c10::string_view moduleInterfaceSrc = R"JIT(
class OneForward(ModuleInterface):
    def one(self, x: Tensor, y: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT";

// 导入库函数，加载模块接口和常量表
static void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  // 创建 SourceImporter 对象，加载源代码
  SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      /*version=*/2);
  // 加载 QualifiedName 类名
  si.loadType(QualifiedName(class_name));
}

// 定义接口测试类 InterfaceTest
TEST(InterfaceTest, ModuleInterfaceSerialization) {
  // 创建 CompilationUnit 对象 cu
  auto cu = std::make_shared<CompilationUnit>();

  // 创建父模块 parentMod 和子模块 subMod
  Module parentMod("parentMod", cu);
  Module subMod("subMod", cu);

  // 创建常量表 constantTable
  std::vector<at::IValue> constantTable;

  // 导入库函数，加载模块接口和常量表
  import_libs(
      cu,
      "__torch__.OneForward",
      std::make_shared<Source>(moduleInterfaceSrc),
      constantTable);

  // 遍历子模块方法的源代码，定义在 subMod 中
  for (const std::string& method : subMethodSrcs) {
    subMod.define(method, nativeResolver());
  }

  // 向父模块注册属性 subMod，指定类型和初始值
  parentMod.register_attribute(
      "subMod",
      cu->get_interface("__torch__.OneForward"),
      subMod._ivalue(),
      // NOLINTNEXTLINE(bugprone-argument-comment)
      /*is_parameter=*/false);

  // 定义父模块的 forward 方法
  parentMod.define(parentForward, nativeResolver());

  // 断言检查父模块是否有属性 subMod
  ASSERT_TRUE(parentMod.hasattr("subMod"));

  // 创建字符串流 ss，保存父模块到其中
  std::stringstream ss;
  parentMod.save(ss);

  // 从字符串流中加载模块 reloaded_mod
  Module reloaded_mod = jit::load(ss);

  // 再次断言检查重新加载的模块是否有属性 subMod
  ASSERT_TRUE(reloaded_mod.hasattr("subMod"));

  // 获取 subMod 的接口类型，并断言其为模块类型
  InterfaceTypePtr submodType =
      reloaded_mod.type()->getAttribute("subMod")->cast<InterfaceType>();
  ASSERT_TRUE(submodType->is_module());
}

} // namespace jit
} // namespace torch
```