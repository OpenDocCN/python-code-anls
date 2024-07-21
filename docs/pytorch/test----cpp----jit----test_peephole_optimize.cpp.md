# `.\pytorch\test\cpp\jit\test_peephole_optimize.cpp`

```py
TEST(PeepholeOptimizeTest, IsAndIsNot)
// 定义测试用例：测试 is / is not none 优化
{
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析以下的 IR 表示的图形
  parseIR(
      R"IR(
graph(%0 : int):
  %1 : None = prim::Constant()
  %2 : bool = aten::__is__(%0, %1)
  %3 : bool = aten::__isnot__(%0, %1)
  return (%2, %3)
  )IR",
      graph.get());
  // 对图形进行 Peephole 优化
  PeepholeOptimize(graph);
  // 进行测试文件检查，确保不包含 "aten::__is__" 和 "aten::__isnot__"
  testing::FileCheck()
      .check_not("aten::__is__")
      ->check_not("aten::__isnot__")
      ->run(*graph);
}

TEST(PeepholeOptimizeTest, IsAndIsNot2)
// 定义测试用例：测试 is / is not none 优化（第二个情况）
{
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析以下的 IR 表示的图形
  parseIR(
      R"IR(
graph(%0: int?):
  %1 : None = prim::Constant()
  %2 : bool = aten::__is__(%0, %1)
  %3 : bool = aten::__isnot__(%0, %1)
  return (%2, %3)
  )IR",
      graph.get());
  // 对图形进行 Peephole 优化
  PeepholeOptimize(graph);
  // 进行测试文件检查，确保包含 "aten::__is__" 和 "aten::__isnot__"
  testing::FileCheck()
      .check("aten::__is__")
      ->check("aten::__isnot__")
      ->run(*graph);
}

TEST(PeepholeOptimizeTest, IsAndIsNot3)
// 定义测试用例：测试 is / is not none 优化（第三个情况）
{
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析以下的 IR 表示的图形
  parseIR(
      R"IR(
graph(%0: int?):
  %1 : Tensor = prim::AutogradZero()
  %2 : None = prim::Constant()
  %4 : bool = aten::__is__(%0, %1)
  %5 : bool = aten::__isnot__(%1, %2)
  return (%4, %5)
  )IR",
      graph.get());
  // 对图形进行 Peephole 优化
  PeepholeOptimize(graph);
  // 进行测试文件检查，确保包含 "aten::__is__"，但不包含 "aten::__isnot__"
  testing::FileCheck()
      .check("aten::__is__")
      ->check_not("aten::__isnot__")
      ->run(*graph);
}

TEST(PeepholeOptimizeTest, UnwrapOptional)
// 定义测试用例：测试 unwrap optional 优化
{
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析以下的 IR 表示的图形
  parseIR(
      R"IR(
graph():
  %1 : Float(*, *, *) = prim::Constant()
  %2 : bool = aten::_unwrap_optional(%1)
  %3 : bool = prim::unchecked_unwrap_optional(%1)
  return (%2, %3)
  )IR",
      graph.get());
  // 对图形进行 Peephole 优化
  PeepholeOptimize(graph);
  // 进行测试文件检查，确保不包含 "unwrap"
  testing::FileCheck().check_not("unwrap")->run(*graph);
}

TEST(PeepholeOptimizeTest, UnwrapOptional2)
// 定义测试用例：测试 unwrap optional 优化（第二个情况）
{
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析以下的 IR 表示的图形
  parseIR(
      R"IR(
graph(%1 : Float(*, *, *)?):
  %2 : bool = aten::_unwrap_optional(%1)
  %3 : bool = prim::unchecked_unwrap_optional(%1)
  return (%2, %3)
  )IR",
      graph.get());
  // 对图形进行 Peephole 优化
  PeepholeOptimize(graph);
  // 进行测试文件检查，确保包含两个 "unwrap"
  testing::FileCheck().check_count("unwrap", 2)->run(*graph);
}

TEST(PeepholeOptimizeTest, AddMMFusion)
// 定义测试用例：测试 addmm 融合优化
{
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析以下的 IR 表示的图形
  parseIR(
      R"IR(
      graph(
        %0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(1, 1, 1)):
        %3 : int = prim::Constant[value=1]()
        %4 : Tensor = aten::mm(%0, %1)
        %5 : Tensor = aten::add(%4, %2, %3)
        %6 : Tensor = aten::add(%5, %2, %3)
        return (%6)
        )IR",
      graph.get());
  // 对图形进行 AddMM 融合优化
  FuseAddMM(graph);
  // 进行测试文件检查，确保包含 "addmm"
  testing::FileCheck().check("addmm")->run(*graph);
}
```