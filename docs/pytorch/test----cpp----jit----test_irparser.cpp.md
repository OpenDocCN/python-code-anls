# `.\pytorch\test\cpp\jit\test_irparser.cpp`

```py
#include <gtest/gtest.h> // 包含 Google Test 的头文件

#include <torch/csrc/jit/ir/ir.h> // 包含 Torch 的 IR 相关头文件
#include <torch/csrc/jit/ir/irparser.h> // 包含 Torch 的 IR 解析器头文件
#include <torch/csrc/jit/testing/file_check.h> // 包含 Torch 的文件检查测试头文件

#include <sstream> // 包含字符串流的头文件
#include <string> // 包含字符串操作相关的头文件

namespace torch {
namespace jit {

/** \brief 解析输入字符串 \p S 中的 IR，打印解析后的图形并验证输出字符串是否与原始字符串匹配。
 *
 *  该函数对值命名和空白字符敏感，因此需要谨慎使用。尽管如此，它有助于保持测试更加简洁。
 */
static void checkRoundtrip(const std::string& s) {
  auto graph = std::make_shared<Graph>(); // 创建一个共享指针 graph，指向新创建的图对象
  parseIR(s, &*graph); // 解析输入字符串 s，并将解析后的内容填充到 graph 中
  std::ostringstream ss; // 创建一个字符串流 ss
  ss << *graph; // 将 graph 的内容写入字符串流 ss
  std::string parsed = ss.str(); // 从字符串流 ss 中获取字符串形式的 graph

  // 跳过输入字符串开头的空白字符。
  int i = 0;
  for (char c : s) {
    if (!isspace(c)) { // 如果当前字符不是空白字符
      break; // 跳出循环
    }
    i++; // 计数器递增
  }
  std::string original = s.substr(i, s.size()); // 获取去除空白字符后的原始输入字符串
  if (original != parsed) { // 如果原始输入字符串与解析后的字符串不匹配
    std::cerr << "Input:" << std::endl << original << std::endl; // 输出原始输入字符串
    std::cerr << "Parsed:" << std::endl << parsed << std::endl; // 输出解析后的字符串
  }
  AT_ASSERT(original == parsed); // 断言原始输入字符串与解析后的字符串必须完全匹配
}

TEST(IRParserTest, Basic) {
  auto graph = std::make_shared<Graph>(); // 创建一个共享指针 graph，指向新创建的图对象
  std::unordered_map<std::string, Value*> vmap; // 创建一个无序映射，用于保存值名到值指针的映射关系
  parseIR(
      R"IR(
graph(%0 : Tensor, %1 : Tensor):
  %2 : Tensor = foo::add(%0, %1)
  %res, %3 = foo::mul(%0, %2)
  %x, %y = foo::combine(%res, %2, %3)
  return (%x, %y, %res))IR",
      &*graph,
      vmap); // 解析输入的 IR 字符串，并将解析后的图和值映射保存到 graph 和 vmap 中

  AT_ASSERT(graph->inputs().size() == 2); // 断言图的输入节点数量为 2
  AT_ASSERT(graph->outputs().size() == 3); // 断言图的输出节点数量为 3
  Value* x = graph->outputs()[0]; // 获取图的第一个输出值 x
  Value* y = graph->outputs()[1]; // 获取图的第二个输出值 y
  Value* res = graph->outputs()[2]; // 获取图的第三个输出值 res
  Value* t0 = graph->inputs()[0]; // 获取图的第一个输入值 t0
  Value* t1 = graph->inputs()[1]; // 获取图的第二个输入值 t1
  AT_ASSERT(vmap["x"] == x); // 断言映射中键 "x" 对应的值指针与图的输出值 x 相同
  AT_ASSERT(vmap["y"] == y); // 断言映射中键 "y" 对应的值指针与图的输出值 y 相同
  AT_ASSERT(vmap["res"] == res); // 断言映射中键 "res" 对应的值指针与图的输出值 res 相同
  AT_ASSERT(vmap["0"] == t0); // 断言映射中键 "0" 对应的值指针与图的输入值 t0 相同
  AT_ASSERT(vmap["1"] == t1); // 断言映射中键 "1" 对应的值指针与图的输入值 t1 相同
  AT_ASSERT(x->node() == y->node()); // 断言输出值 x 和 y 所属的节点相同
  Node* comb = x->node(); // 获取输出值 x 所属的节点指针
  Value* t2 = comb->inputs()[1]; // 获取节点输入的第二个值 t2
  Value* t3 = comb->inputs()[2]; // 获取节点输入的第三个值 t3
  AT_ASSERT(vmap["2"] == t2); // 断言映射中键 "2" 对应的值指针与节点输入值 t2 相同
  AT_ASSERT(vmap["3"] == t3); // 断言映射中键 "3" 对应的值指针与节点输入值 t3 相同
  AT_ASSERT(comb->kind().toQualString() == std::string("foo::combine")); // 断言节点的操作名称为 "foo::combine"
  AT_ASSERT(comb->outputs() == std::vector<Value*>({x, y})); // 断言节点的输出与 x 和 y 相同
  AT_ASSERT(comb->inputs() == std::vector<Value*>({res, t2, t3})); // 断言节点的输入与 res、t2、t3 相同
  Node* mul = res->node(); // 获取输出值 res 所属的节点指针
  AT_ASSERT(mul->kind().toQualString() == std::string("foo::mul")); // 断言节点的操作名称为 "foo::mul"
  AT_ASSERT(mul->inputs() == std::vector<Value*>({t0, t2})); // 断言节点的输入与 t0、t2 相同
  AT_ASSERT(mul->outputs() == std::vector<Value*>({res, t3})); // 断言节点的输出与 res、t3 相同
  Node* add = t2->node(); // 获取值 t2 所属的节点指针
  AT_ASSERT(add->kind().toQualString() == std::string("foo::add")); // 断言节点的操作名称为 "foo::add"
  AT_ASSERT(add->inputs() == std::vector<Value*>({t0, t1})); // 断言节点的输入与 t0、t1 相同
  AT_ASSERT(add->outputs() == std::vector<Value*>({t2})); // 断言节点的输出与 t2 相同
}

TEST(IRParserTest, NestedBlock) {
  checkRoundtrip(R"IR(
graph():
  %0 : Tensor = a::a()
    block0():
      %1 : Tensor = b::b()
        block0():
          %2 : Tensor = c::c()
          -> ()
      -> ()
  %3 : Tensor = d::d()
  return (%3)
)IR"); // 调用 checkRoundtrip 函数，验证 IR 字符串的解析结果
}

TEST(IRParserTest, If) {
  checkRoundtrip(R"IR(
``` // 这行之前没有引号，应为样例错误
// 定义名为 `graph` 的函数，接受三个名为 `%0`, `%1`, `%2` 的张量参数
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  // 创建一个整数常量 `%3`，其值为 1
  %3 : int = prim::Constant[value=1]()
  // 对输入张量 `%0` 和 `%1` 执行加法运算，并将结果赋给 `%4`
  %4 : Tensor = aten::add(%0, %1, %3)
  // 使用 `%2` 进行条件判断
  %5 : Tensor = prim::If(%2)
    block0():
      // 创建一个整数常量 `%6`，其值为 1
      %6 : int = prim::Constant[value=1]()
      // 对 `%1` 和 `%3` 执行加法运算，并将结果赋给 `%7`
      %7 : Tensor = aten::add(%1, %3, %6)
      // 创建一个整数常量 `%8`，其值为 1
      %8 : int = prim::Constant[value=1]()
      // 对 `%7` 和 `%3` 执行加法运算，并将结果作为块的返回值
      %9 : Tensor = aten::add(%7, %3, %8)
      -> (%9)
  // 创建一个整数常量 `%10`，其值为 1
  %10 : int = prim::Constant[value=1]()
  // 对 `%5` 和 `%3` 执行加法运算，并将结果赋给 `%11`
  %11 : Tensor = aten::add(%5, %3, %10)
  // 返回 `%11` 作为函数的结果
  return (%11)
)IR");
}
// 定义名为 graph 的函数，接受三个输入参数 %0, %1, %2，类型均为 Tensor
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  // 创建一个 4x4x5 的双精度常量张量 %3
  %3 : Double(4, 4, 5) = prim::Constant()
  // 返回常量张量 %3
  return (%3)
)IR");
}



// 测试函数，检查 IR 中的嵌套容器
TEST(IRParserTest, NestedContrainer) {
  // 调用 checkRoundtrip 函数，传入以下 IR 文本
  checkRoundtrip(
      R"IR(
graph():
  // 创建一个 float 类型的数组常量 %0，值为 [1., 2., 3.]
  %0 : float[] = prim::Constant[value=[1., 2., 3.]]()
  // 创建一个 string 类型的数组常量 %1，值为 ["ab", "cd", "ef"]
  %1 : str[] = prim::Constant[value=["ab", "cd", "ef"]]()
  // 创建一个包含两个元素的元组 %2，包括 %0 和 %1
  %2 : (float[], str[]) = prim::TupleConstruct(%0, %1)
  // 返回元组 %2
  return (%2)
)IR");
}



// 测试函数，检查异常形状注释
TEST(IRParserTest, MalformedShapeAnnotation) {
  // 使用 EXPECT_ANY_THROW 宏来测试以下代码块是否会抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(checkRoundtrip(
      R"IR(
graph(%0 : Tensor,
    %1 : Tensor,
    %2 : Tensor):
  // 创建一个带有错误形状注释的双精度常量张量 %3
  %3 : Double(4!, 4, 5) = prim::Constant()
  // 返回常量张量 %3
  return (%3)
)IR"));
}



// 测试函数，检查 IR 中的文件检查
TEST(IRParserTest, FileCheck) {
  // 创建一个名为 graph 的共享指针对象
  auto graph = std::make_shared<Graph>();
  // 定义 IR 文本字符串
  const std::string& text =
      R"IR(
    graph(%a):
    // 检查文件内容应包含 "return" 关键字
    # CHECK: return
      return (%a))IR";
  // 解析 IR 文本并填充到 graph 中
  parseIR(text, &*graph);
  // 断言 graph 的第一个输入为 Tensor 类型的子类型
  AT_ASSERT(graph->inputs()[0]->type()->isSubtypeOf(*TensorType::get()));
  // 使用 FileCheck 类检查 text 是否与 graph 的内容匹配
  torch::jit::testing::FileCheck().run(text, *graph);
}



// 测试函数，检查张量的步幅
TEST(IRParserTest, Strides) {
  // 创建一个名为 graph 的共享指针对象
  auto graph = std::make_shared<Graph>();
  // 创建一个空的无序映射 vmap
  std::unordered_map<std::string, Value*> vmap;
  // 解析以下 IR 文本，并填充到 graph 中
  parseIR(
      R"IR(
graph(%a : Float(4, 5),
      %b : Float(4, 5, strides=[5, 1]),
      %c : Double(*, *)):
  // 返回输入 %a
  return (%a)
)IR",
      &*graph,
      vmap);
  // 获取输入 %a, %b, %c
  Value* a = graph->inputs()[0];
  Value* b = graph->inputs()[1];
  Value* c = graph->inputs()[2];

  // 断言输入 %a 的类型为 TensorType，并获取其大小和步幅信息
  auto a_type = a->type()->cast<TensorType>();
  auto a_sizes = *a_type->sizes().concrete_sizes();
  auto a_strides = a_type->strides().concrete_sizes();
  AT_ASSERT(a_sizes[0] == 4 && a_sizes[1] == 5);
  AT_ASSERT(a_strides == c10::nullopt);

  // 断言输入 %b 的类型为 TensorType，并获取其大小和步幅信息
  auto b_type = b->type()->cast<TensorType>();
  auto b_sizes = *b_type->sizes().concrete_sizes();
  auto b_strides = *(b_type->strides().sizes());
  AT_ASSERT(b_sizes[0] == 4 && b_sizes[1] == 5);
  AT_ASSERT(*b_strides[0] == 5 && *b_strides[1] == 1);

  // 断言输入 %c 的类型为 TensorType，并检查其大小和步幅信息为空
  auto c_type = c->type()->cast<TensorType>();
  AT_ASSERT(*c_type->sizes().size() == 2);
  AT_ASSERT(c_type->sizes().concrete_sizes() == c10::nullopt);
  AT_ASSERT(c_type->strides().concrete_sizes() == c10::nullopt);
}



// 测试函数，检查异常步幅
TEST(IRParserTest, MalformedStrides) {
  // 创建一个名为 graph 的共享指针对象
  auto graph = std::make_shared<Graph>();
  // 创建一个空的无序映射 vmap
  std::unordered_map<std::string, Value*> vmap;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 使用 EXPECT_ANY_THROW 宏来测试以下代码块是否会抛出异常
  EXPECT_ANY_THROW(parseIR(
      R"IR(
graph(%a : Float(4, strides=[5], 5)):
  // 返回输入 %a
  return (%a)
)IR",
      &*graph,
      vmap));
}



// 测试函数，检查张量形状
TEST(IRParserTest, TensorShapes) {
  // 调用 checkRoundtrip 函数，传入以下 IR 文本
  checkRoundtrip(
      R"IR(
graph(%a : Float(4, 5),
      %b : Float(4, 5, strides=[5, 1]),
      %c : Double(*, *)):
  // 返回输入 %a
  return (%a)
)IR");
}



// 测试函数，检查设备和 requires_grad 张量
TEST(IRParserTest, DeviceAndRequiresGradTensors) {
  // 调用 checkRoundtrip 函数，传入以下 IR 文本
  checkRoundtrip(
      R"IR(
// 定义了一个名为 `graph` 的函数，参数类型和属性如下：
//   %a: Float 类型的张量，形状和设备不确定，位于 CPU 上
//   %b: Float 类型的张量，形状和设备不确定，需要计算梯度
//   %c: Long 类型的张量，形状为 [5, 10]，位于 CPU 上，需要计算梯度
//   %d: Float 类型的张量，形状为 [5]，位于 cuda:2 设备上，不需要计算梯度
//   %e: Long 类型的张量，形状为 [4, 3, 1]，使用步长 [6, 2, 1]，位于 cuda:1 设备上，不需要计算梯度
//   %f: Float 类型的标量，未指定其他属性
//   %g: Float 类型的标量，位于 CPU 上，未指定其他属性
//   %h: Float 类型的标量，需要计算梯度
//   %i: Float 类型的标量，位于 cuda:1 设备上，不需要计算梯度
//   %j: Double 类型的张量，形状和设备不确定，不需要计算梯度
// 返回 %a 张量作为函数的输出
return (%a)
)IR");
}

// 定义了 IRParserTest 的测试用例 ListConstant
TEST(IRParserTest, ListConstant) {
  auto graph = std::make_shared<Graph>();
  // 调用 parseIR 函数解析以下 IR 字符串
  parseIR(
      R"IR(
graph():
  %d : int[] = prim::Constant[value=[1,2,3]]()
  return (%d)
)IR",
      &*graph);
  // 获取输出节点 n
  Node* n = graph->outputs()[0]->node();
  // 断言节点 n 的类型为 prim::Constant
  AT_ASSERT(n->kind() == prim::Constant);
  // 断言节点 n 的值属性的类型为整数数组
  AT_ASSERT(n->kindOf(attr::value) == AttributeKind::ival);
  // 获取节点 n 的值，将其转换为整数列表
  const auto& genericList = n->ival(attr::value).toList();
  std::vector<int> int_vals;
  // 遍历整数列表，将每个值转换为整数并添加到 int_vals 中
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const IValue& ival : genericList) {
    int_vals.push_back(ival.toInt());
  }
  // 断言整数列表的大小为 3
  AT_ASSERT(int_vals.size() == 3);
  // 断言整数列表的具体值为 [1, 2, 3]
  AT_ASSERT(int_vals[0] == 1 && int_vals[1] == 2 && int_vals[2] == 3);
}

// 定义了 IRParserTest 的测试用例 PartialStarTensor
TEST(IRParserTest, PartialStarTensor) {
  // 调用 checkRoundtrip 函数，对以下 IR 字符串进行测试
  checkRoundtrip(
      R"IR(
graph(%x : Float(10, *, 10)):
  return (%x)
)IR");
}

// 定义了 IRParserTest 的测试用例 ComplexTensorAttributes
TEST(IRParserTest, ComplexTensorAttributes) {
  // 调用 checkRoundtrip 函数，对以下 IR 字符串进行测试
  checkRoundtrip(
      R"IR(
graph(%x : Double(*, 200, *, requires_grad=1, device=cuda:1),
      %b : Float(5, *, requires_grad=1),
      %c : Long(*, 10, device=cpu)):
  return (%x)
)IR");
}
```