# `.\pytorch\aten\src\ATen\test\type_test.cpp`

```py
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/serialization/import_source.h>

namespace c10 {

// 定义一个测试案例 TypeCustomPrinter，用于测试自定义类型打印器的基本功能
TEST(TypeCustomPrinter, Basic) {
  // 定义一个类型打印器，根据具体类型返回相应字符串，此处示例中针对 TensorType 类型返回 "CustomTensor"
  TypePrinter printer =
      [](const Type& t) -> std::optional<std::string> {
    if (auto tensorType = t.cast<TensorType>()) {
      return "CustomTensor";
    }
    return c10::nullopt;
  };

  // 创建一个随机张量 iv
  torch::Tensor iv = torch::rand({2, 3});
  // 创建该张量的类型对象
  const auto type = TensorType::create(iv);
  // 使用默认的类型注释字符串检查是否为 "Tensor"
  EXPECT_EQ(type->annotation_str(), "Tensor");
  // 使用自定义打印器检查类型注释字符串是否为 "CustomTensor"
  EXPECT_EQ(type->annotation_str(printer), "CustomTensor");

  // 检查与张量无关的整数类型是否不受影响
  const auto intType = IntType::get();
  EXPECT_EQ(intType->annotation_str(printer), intType->annotation_str());
}

// 测试案例 TypeCustomPrinter，用于测试包含类型的情况
TEST(TypeCustomPrinter, ContainedTypes) {
  // 定义类型打印器，根据具体类型返回相应字符串，此处示例中针对 TensorType 类型返回 "CustomTensor"
  TypePrinter printer =
      [](const Type& t) -> std::optional<std::string> {
    if (auto tensorType = t.cast<TensorType>()) {
      return "CustomTensor";
    }
    return c10::nullopt;
  };
  // 创建一个随机张量 iv
  torch::Tensor iv = torch::rand({2, 3});
  // 创建该张量的类型对象
  const auto type = TensorType::create(iv);

  // 创建包含张量类型的元组类型
  const auto tupleType = TupleType::create({type, IntType::get(), type});
  // 检查默认的类型注释字符串是否为 "Tuple[Tensor, int, Tensor]"
  EXPECT_EQ(tupleType->annotation_str(), "Tuple[Tensor, int, Tensor]");
  // 使用自定义打印器检查类型注释字符串是否为 "Tuple[CustomTensor, int, CustomTensor]"
  EXPECT_EQ(
      tupleType->annotation_str(printer), "Tuple[CustomTensor, int, CustomTensor]");
  
  // 创建包含整数和张量类型的字典类型
  const auto dictType = DictType::create(IntType::get(), type);
  // 使用自定义打印器检查类型注释字符串是否为 "Dict[int, CustomTensor]"
  EXPECT_EQ(dictType->annotation_str(printer), "Dict[int, CustomTensor]");

  // 创建包含元组类型的列表类型
  const auto listType = ListType::create(tupleType);
  // 使用自定义打印器检查类型注释字符串是否为 "List[Tuple[CustomTensor, int, CustomTensor]]"
  EXPECT_EQ(
      listType->annotation_str(printer),
      "List[Tuple[CustomTensor, int, CustomTensor]]");
}

// 测试案例 TypeCustomPrinter，用于测试命名元组的情况
TEST(TypeCustomPrinter, NamedTuples) {
  // 定义类型打印器，根据具体类型返回相应字符串，此处示例中针对命名元组类型返回 "Rewritten"
  TypePrinter printer =
      [](const Type& t) -> std::optional<std::string> {
    if (auto tupleType = t.cast<TupleType>()) {
      // 仅重新命名的命名元组类型才返回 "Rewritten"
      if (tupleType->name()) {
        return "Rewritten";
      }
    }
    return c10::nullopt;
  };
  // 创建一个随机张量 iv
  torch::Tensor iv = torch::rand({2, 3});
  // 创建该张量的类型对象
  const auto type = TensorType::create(iv);

  // 创建命名元组类型，包含名为 "foo" 和 "bar" 的字段名
  std::vector<std::string> field_names = {"foo", "bar"};
  const auto namedTupleType = TupleType::createNamed(
      "my.named.tuple", field_names, {type, IntType::get()});
  // 使用自定义打印器检查类型注释字符串是否为 "Rewritten"
  EXPECT_EQ(namedTupleType->annotation_str(printer), "Rewritten");

  // 将命名元组类型放入另一个元组中，仍应正常工作
  const auto outerTupleType = TupleType::create({IntType::get(), namedTupleType});
  // 使用自定义打印器检查类型注释字符串是否为 "Tuple[int, Rewritten]"
  EXPECT_EQ(outerTupleType->annotation_str(printer), "Tuple[int, Rewritten]");
}

// 定义一个静态函数 importType，用于导入类型
static TypePtr importType(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& qual_name,
    const std::string& src) {
  // 创建一个空的常量表
  std::vector<at::IValue> constantTable;
  // 使用源代码创建一个 shared_ptr 的 torch::jit::Source 对象
  auto source = std::make_shared<torch::jit::Source>(src);
  // 创建一个 SourceImporter 对象 si，传入编译单元 cu、常量表的指针、
  // 以及一个 lambda 表达式，该 lambda 表达式根据名称返回源代码的 shared_ptr
  // 这里的版本号是固定的，为 2
  torch::jit::SourceImporter si(
      cu,
      &constantTable,
      [&](const std::string& name) -> std::shared_ptr<torch::jit::Source> {
        return source;
      },
      /*version=*/2);
  // 调用 SourceImporter 的 loadType 方法，传入限定名称 qual_name，
  // 返回类型信息
  return si.loadType(qual_name);
}

TEST(TypeEquality, ClassBasic) {
  // 单元测试：类的基本相等性检查

  // 创建共享的编译单元对象
  auto cu = std::make_shared<CompilationUnit>();

  // 定义包含类定义的源代码片段
  const auto src = R"JIT(
class First:
    def one(self, x: Tensor, y: Tensor) -> Tensor:
      return x
)JIT";

  // 导入类类型并分配给classType
  auto classType = importType(cu, "__torch__.First", src);

  // 从编译单元获取同名类的类型
  auto classType2 = cu->get_type("__torch__.First");

  // 断言两个类类型对象相等
  EXPECT_EQ(*classType, *classType2);
}

TEST(TypeEquality, ClassInequality) {
  // 单元测试：类的不相等性检查

  // 创建第一个编译单元对象
  auto cu = std::make_shared<CompilationUnit>();

  // 定义第一个类的源代码片段
  const auto src = R"JIT(
class First:
    def one(self, x: Tensor, y: Tensor) -> Tensor:
      return x
)JIT";

  // 导入第一个类的类型并分配给classType
  auto classType = importType(cu, "__torch__.First", src);

  // 创建第二个编译单元对象
  auto cu2 = std::make_shared<CompilationUnit>();

  // 定义第二个类的源代码片段，即使类名相同，其内容不同
  const auto src2 = R"JIT(
class First:
    def one(self, x: Tensor, y: Tensor) -> Tensor:
      return y
)JIT";

  // 导入第二个类的类型并分配给classType2
  auto classType2 = importType(cu2, "__torch__.First", src2);

  // 断言两个类类型对象不相等
  EXPECT_NE(*classType, *classType2);
}

TEST(TypeEquality, InterfaceEquality) {
  // 单元测试：接口的相等性检查

  // 创建第一个编译单元对象
  auto cu = std::make_shared<CompilationUnit>();

  // 定义接口的源代码片段
  const auto interfaceSrc = R"JIT(
class OneForward(Interface):
    def one(self, x: Tensor, y: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT";

  // 导入接口类型并分配给interfaceType
  auto interfaceType = importType(cu, "__torch__.OneForward", interfaceSrc);

  // 创建第二个编译单元对象
  auto cu2 = std::make_shared<CompilationUnit>();

  // 导入同名接口类型并分配给interfaceType2
  auto interfaceType2 = importType(cu2, "__torch__.OneForward", interfaceSrc);

  // 断言两个接口类型对象相等
  EXPECT_EQ(*interfaceType, *interfaceType2);
}

TEST(TypeEquality, InterfaceInequality) {
  // 单元测试：接口的不相等性检查

  // 创建第一个编译单元对象
  auto cu = std::make_shared<CompilationUnit>();

  // 定义第一个接口的源代码片段
  const auto interfaceSrc = R"JIT(
class OneForward(Interface):
    def one(self, x: Tensor, y: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT";

  // 导入第一个接口类型并分配给interfaceType
  auto interfaceType = importType(cu, "__torch__.OneForward", interfaceSrc);

  // 创建第二个编译单元对象
  auto cu2 = std::make_shared<CompilationUnit>();

  // 定义第二个接口的源代码片段，虽然接口名相同，但方法不同
  const auto interfaceSrc2 = R"JIT(
class OneForward(Interface):
    def two(self, x: Tensor, y: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT";

  // 导入第二个接口类型并分配给interfaceType2
  auto interfaceType2 = importType(cu2, "__torch__.OneForward", interfaceSrc2);

  // 断言两个接口类型对象不相等
  EXPECT_NE(*interfaceType, *interfaceType2);
}

TEST(TypeEquality, TupleEquality) {
  // 单元测试：元组的相等性检查

  // 创建包含不同类型的元组类型
  auto type = TupleType::create({IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  auto type2 = TupleType::create({IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});

  // 断言两个元组类型对象相等
  EXPECT_EQ(*type, *type2);
}
TEST(TypeEquality, NamedTupleEquality) {
  // 测试命名元组的相等性
  // 定义一个包含字段名称的字符串向量
  std::vector<std::string> fields = {"a", "b", "c", "d"};
  // 定义另一个不同的字段名称向量
  std::vector<std::string> otherFields = {"wow", "so", "very", "different"};

  // 创建一个命名元组类型，命名为"MyNamedTuple"，包含字段名称和对应的类型
  auto type = TupleType::createNamed(
      "MyNamedTuple",
      fields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  
  // 创建另一个具有相同名称和字段的命名元组类型
  auto type2 = TupleType::createNamed(
      "MyNamedTuple",
      fields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  
  // 断言这两个命名元组类型相等
  EXPECT_EQ(*type, *type2);

  // 创建一个命名元组类型，命名为"WowSoDifferent"，但字段名称与前面定义的不同
  auto differentName = TupleType::createNamed(
      "WowSoDifferent",
      fields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  
  // 断言这个新类型与第一个类型不相等
  EXPECT_NE(*type, *differentName);

  // 创建一个命名元组类型，名称与第一个相同，但字段名称与otherFields不同
  auto differentField = TupleType::createNamed(
      "MyNamedTuple",
      otherFields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  
  // 断言这个新类型与第一个类型不相等
  EXPECT_NE(*type, *differentField);
}
} // namespace c10
```