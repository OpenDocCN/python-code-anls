# `.\pytorch\test\cpp\jit\test_class_import.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/core/qualified_name.h>  // 导入 ATen 库中的 qualified_name.h 文件
#include <test/cpp/jit/test_utils.h>  // 导入测试工具函数的头文件
#include <torch/csrc/jit/frontend/resolver.h>  // 导入 Torch 的解析器前端头文件
#include <torch/csrc/jit/serialization/import_source.h>  // 导入 Torch 序列化导入源文件
#include <torch/torch.h>  // 导入 Torch 库的主头文件

namespace torch {
namespace jit {

static constexpr c10::string_view classSrcs1 = R"JIT(  // 定义一个 constexpr 字符串视图，包含类的定义代码
class FooNestedTest:
    def __init__(self, y):
        self.y = y

class FooNestedTest2:
    def __init__(self, y):
        self.y = y
        self.nested = __torch__.FooNestedTest(y)

class FooTest:
    def __init__(self, x):
        self.class_attr = __torch__.FooNestedTest(x)
        self.class_attr2 = __torch__.FooNestedTest2(x)
        self.x = self.class_attr.y + self.class_attr2.y
)JIT";

static constexpr c10::string_view classSrcs2 = R"JIT(  // 定义另一个 constexpr 字符串视图，包含类的定义代码
class FooTest:
    def __init__(self, x):
      self.dx = x
)JIT";

static void import_libs(  // 定义函数 import_libs，用于导入类定义到编译单元
    std::shared_ptr<CompilationUnit> cu,  // 接收一个共享指针指向编译单元对象
    const std::string& class_name,  // 类名作为字符串引用参数
    const std::shared_ptr<Source>& src,  // 类的源代码作为共享指针参数
    const std::vector<at::IValue>& tensor_table) {  // 张量表作为向量引用参数
  SourceImporter si(  // 创建 SourceImporter 对象 si
      cu,
      &tensor_table,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },  // 匿名函数返回源代码共享指针
      /*version=*/2);  // 指定版本号为 2
  si.loadType(QualifiedName(class_name));  // 调用 SourceImporter 对象的 loadType 方法，加载类名
}

TEST(ClassImportTest, Basic) {  // 定义测试用例 ClassImportTest.Basic
  auto cu1 = std::make_shared<CompilationUnit>();  // 创建一个编译单元对象 cu1
  auto cu2 = std::make_shared<CompilationUnit>();  // 创建另一个编译单元对象 cu2
  std::vector<at::IValue> constantTable;  // 创建一个空的 IValue 向量 constantTable

  // 导入不同版本的 FooTest 到两个命名空间中
  import_libs(
      cu1,
      "__torch__.FooTest",  // 第一个命名空间使用类名 "__torch__.FooTest"
      std::make_shared<Source>(classSrcs1),  // 使用 classSrcs1 定义的类源代码
      constantTable);  // 导入时使用空的常量表

  import_libs(
      cu2,
      "__torch__.FooTest",  // 第二个命名空间同样使用类名 "__torch__.FooTest"
      std::make_shared<Source>(classSrcs2),  // 使用 classSrcs2 定义的类源代码
      constantTable);  // 导入时使用相同的空常量表

  // 我们应该在引用的任一命名空间中得到正确的 `FooTest` 版本
  c10::QualifiedName base("__torch__");  // 创建 QualifiedName 对象 base，初始化为 "__torch__"
  auto classType1 = cu1->get_class(c10::QualifiedName(base, "FooTest"));  // 获取 cu1 中的 FooTest 类型
  ASSERT_TRUE(classType1->hasAttribute("x"));  // 断言 cu1 中的 FooTest 类型有属性 "x"
  ASSERT_FALSE(classType1->hasAttribute("dx"));  // 断言 cu1 中的 FooTest 类型没有属性 "dx"

  auto classType2 = cu2->get_class(c10::QualifiedName(base, "FooTest"));  // 获取 cu2 中的 FooTest 类型
  ASSERT_TRUE(classType2->hasAttribute("dx"));  // 断言 cu2 中的 FooTest 类型有属性 "dx"
  ASSERT_FALSE(classType2->hasAttribute("x"));  // 断言 cu2 中的 FooTest 类型没有属性 "x"

  // 我们只应该在第一个命名空间中看到 FooNestedTest
  auto c = cu1->get_class(c10::QualifiedName(base, "FooNestedTest"));  // 获取 cu1 中的 FooNestedTest 类型
  ASSERT_TRUE(c);  // 断言 cu1 中存在 FooNestedTest 类型

  c = cu2->get_class(c10::QualifiedName(base, "FooNestedTest"));  // 获取 cu2 中的 FooNestedTest 类型
  ASSERT_FALSE(c);  // 断言 cu2 中不存在 FooNestedTest 类型
}
TEST(ClassImportTest, ScriptObject) {
  // 创建两个模块对象 m1 和 m2
  Module m1("m1");
  Module m2("m2");
  // 创建常量表
  std::vector<at::IValue> constantTable;
  // 导入第一个类库 "__torch__.FooTest" 到模块 m1 中
  import_libs(
      m1._ivalue()->compilation_unit(),
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs1),
      constantTable);
  // 导入第二个类库 "__torch__.FooTest" 到模块 m2 中
  import_libs(
      m2._ivalue()->compilation_unit(),
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs2),
      constantTable);

  // 构造函数的错误参数应该抛出异常
  c10::QualifiedName base("__torch__");
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(m1.create_class(c10::QualifiedName(base, "FooTest"), {1}));

  // 创建一个 Tensor 对象 x
  auto x = torch::ones({2, 3});
  // 在模块 m2 中创建类 "__torch__.FooTest" 的实例，并获取其对象
  auto obj = m2.create_class(c10::QualifiedName(base, "FooTest"), x).toObject();
  // 获取对象 obj 的属性 "dx"
  auto dx = obj->getAttr("dx");
  // 断言 x 与 dx 的值几乎相等
  ASSERT_TRUE(almostEqual(x, dx.toTensor()));

  // 创建一个新的 Tensor 对象 new_x
  auto new_x = torch::rand({2, 3});
  // 将 new_x 设置为对象 obj 的属性 "dx"
  obj->setAttr("dx", new_x);
  // 获取更新后的属性 "dx"
  auto new_dx = obj->getAttr("dx");
  // 断言 new_x 与 new_dx 的值几乎相等
  ASSERT_TRUE(almostEqual(new_x, new_dx.toTensor()));
}

static const auto methodSrc = R"JIT(
def __init__(self, x):
    return x
)JIT";

TEST(ClassImportTest, ClassDerive) {
  // 创建一个编译单元对象 cu
  auto cu = std::make_shared<CompilationUnit>();
  // 创建名为 "foo.bar" 的类类型 cls
  auto cls = ClassType::create("foo.bar", cu);
  // 创建一个 SimpleSelf 对象 self，用于方法定义
  const auto self = SimpleSelf(cls);
  // 定义方法 methods，将方法源代码 methodSrc 加入编译单元 cu，使用本地解析器和 self
  auto methods = cu->define("foo.bar", methodSrc, nativeResolver(), &self);
  // 获取第一个方法
  auto method = methods[0];
  // 给类 cls 添加名为 "attr" 的属性，类型为 TensorType
  cls->addAttribute("attr", TensorType::get());
  // 断言类 cls 中存在名为 method->name() 的方法
  ASSERT_TRUE(cls->findMethod(method->name()));

  // 派生一个新的类 newCls，应保留原有的属性和方法
  auto newCls = cls->refine({TensorType::get()});
  // 断言新类 newCls 中仍然有属性 "attr"
  ASSERT_TRUE(newCls->hasAttribute("attr"));
  // 断言新类 newCls 中仍然有方法 method->name()
  ASSERT_TRUE(newCls->findMethod(method->name()));

  // 创建一个新的类 newCls2，包含 TensorType 类型的成员变量
  auto newCls2 = cls->withContained({TensorType::get()})->expect<ClassType>();
  // 断言新类 newCls2 中仍然有属性 "attr"
  ASSERT_TRUE(newCls2->hasAttribute("attr"));
  // 断言新类 newCls2 中仍然有方法 method->name()
  ASSERT_TRUE(newCls2->findMethod(method->name()));
}

static constexpr c10::string_view torchbindSrc = R"JIT(
class FooBar1234(Module):
  __parameters__ = []
  f : __torch__.torch.classes._TorchScriptTesting._StackString
  training : bool
  def forward(self: __torch__.FooBar1234) -> str:
    return (self.f).top()
)JIT";

TEST(ClassImportTest, CustomClass) {
  // 创建一个新的编译单元对象 cu1
  auto cu1 = std::make_shared<CompilationUnit>();
  // 创建常量表
  std::vector<at::IValue> constantTable;
  // 导入类库 "__torch__.FooBar1234" 到编译单元 cu1 中
  import_libs(
      cu1,
      "__torch__.FooBar1234",
      std::make_shared<Source>(torchbindSrc),
      constantTable);
}

} // namespace jit
} // namespace torch
```