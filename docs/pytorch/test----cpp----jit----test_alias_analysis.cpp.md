# `.\pytorch\test\cpp\jit\test_alias_analysis.cpp`

```
// 包含 Google Test 的头文件
#include <gtest/gtest.h>

// 包含 Torch 的自动微分和变量工厂相关头文件
#include <torch/csrc/autograd/generated/variable_factories.h>
// 包含 Torch 的 IR 发射器相关头文件
#include <torch/csrc/jit/frontend/ir_emitter.h>
// 包含 Torch 的别名分析相关头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
// 包含 Torch 的 IR 解析器相关头文件
#include <torch/csrc/jit/ir/irparser.h>
// 包含 Torch 的类型哈希相关头文件
#include <torch/csrc/jit/ir/type_hashing.h>
// 包含 Torch 的子图工具相关头文件
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
// 包含 Torch 的自定义运算符相关头文件
#include <torch/csrc/jit/runtime/custom_operator.h>
// 包含 Torch 的图遍历工具相关头文件
#include <torch/csrc/jit/runtime/graph_iterator.h>

// 包含 ATen 的张量操作相关头文件
#include <ATen/TensorOperators.h>

// Torch 命名空间开始
namespace torch {
namespace jit {

// 从模式中获取别名分析类型
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// 用于设置图并使断言更清晰的测试夹具类
class TopologicalMoveTest : public ::testing::Test {
 protected:
  TopologicalMoveTest() {
    createGraph(); // 创建测试图形
    aliasDb = std::make_unique<AliasDb>(graph); // 创建别名数据库对象
  }

  // 创建图的方法，节点按其输出命名
  // 例如，“a”是输出值为`a`的节点的别名
  void createGraph() {
    graph = std::make_shared<Graph>(); // 创建共享指针类型的图对象
    createNode("a", {}); // 创建节点 "a"，无输入
    createNode("b", {"a"}); // 创建节点 "b"，输入为 "a"
    createNode("c", {}); // 创建节点 "c"，无输入
    createNode("d", {"a", "b"}); // 创建节点 "d"，输入为 "a" 和 "b"
    createNode("e", {"c", "b"}); // 创建节点 "e"，输入为 "c" 和 "b"
    createNode("f", {"e"}); // 创建节点 "f"，输入为 "e"
    createNode("g", {"e"}); // 创建节点 "g"，输入为 "e"
    createNode("h", {"g"}); // 创建节点 "h"，输入为 "g"
    createNode("i", {"g"}); // 创建节点 "i"，输入为 "g"
    createNode("j", {"i"}); // 创建节点 "j"，输入为 "i"
    createNode("k", {"i"}); // 创建节点 "k"，输入为 "i"
    createNode("l", {"a"}); // 创建节点 "l"，输入为 "a"
    createNode("m", {}, {"l"}); // 创建节点 "m"，无输入，块依赖于 "l"
    createNode("n", {"m"}); // 创建节点 "n"，输入为 "m"
    createNode("o", {"n"}); // 创建节点 "o"，输入为 "n"
    createNode("p", {}); // 创建节点 "p"，无输入
    createNode("q", {}); // 创建节点 "q"，无输入
    createNode("r", {"q"}); // 创建节点 "r"，输入为 "q"
    createNode("s", {"q"}); // 创建节点 "s"，输入为 "q"

    graph->lint(); // 检查图的有效性
  }

  // 创建具有给定名称和输入名称的节点的辅助方法
  void createNode(
      const std::string& name,
      const std::vector<std::string>& inputNames,
      const std::vector<std::string>& blockInputNames = {}) {
    std::vector<Value*> inputs; // 输入值的向量
    for (const auto& name_ : inputNames) {
      inputs.push_back(nodes.at(name_)->output()); // 将节点输出添加到输入值向量中
    }
    auto node = graph->appendNode(graph->create(prim::AutogradZero, inputs)); // 在图中附加节点
    node->output()->setDebugName(name); // 设置节点输出的调试名称
    nodes[name] = node; // 将节点添加到节点映射中

    if (blockInputNames.size() != 0) { // 如果存在块输入名称
      node->addBlock(); // 添加块到节点
      std::vector<Value*> blockDeps; // 块依赖的值向量
      for (const auto& name_ : blockInputNames) {
        blockDeps.push_back(nodes.at(name_)->output()); // 将节点输出添加到块依赖的值向量中
      }

      auto block = node->blocks().at(0); // 获取节点的第一个块
      block->appendNode(graph->create(prim::AutogradZero, blockDeps)); // 在块中附加节点
    }
  }

  // 在拓扑上有效地在节点之前移动方法
  bool moveBeforeTopologicallyValid(
      const std::string& toInsert,
      const std::string& insertPoint) {
    std::function<bool(Node*, Node*)> func =
        [this](Node* toInsert, Node* insertPoint) {
          return aliasDb->moveBeforeTopologicallyValid(toInsert, insertPoint); // 调用别名数据库方法来进行拓扑有效的移动
        };
    return moveWithChecks(toInsert, insertPoint, func); // 执行移动操作的通用方法
  }

  // 在拓扑上有效地在节点之后移动方法
  bool moveAfterTopologicallyValid(
      const std::string& toInsert,
      const std::string& insertPoint) {
    // 定义一个函数对象 func，接受两个 Node* 类型的参数，并返回 bool 类型的结果
    std::function<bool(Node*, Node*)> func =
        // 使用 lambda 表达式定义函数 func
        [this](Node* toInsert, Node* insertPoint) {
          // 调用 aliasDb 的 moveAfterTopologicallyValid 方法，移动 toInsert 到 insertPoint 合法的位置
          return aliasDb->moveAfterTopologicallyValid(toInsert, insertPoint);
        };
    // 调用 moveWithChecks 函数，将 toInsert 和 insertPoint 以及 func 作为参数传递
    return moveWithChecks(toInsert, insertPoint, func);
  }

  // 根据参数进行检查和移动节点的函数
  bool moveWithChecks(
      const std::string& toInsert,
      const std::string& insertPoint,
      std::function<bool(Node*, Node*)> func) {
    // 获取 nodes 容器中的 toInsert 和 insertPoint 的节点指针
    auto n = nodes.at(toInsert);
    auto insert = nodes.at(insertPoint);
    // 判断 n 是否在 insert 之后
    bool isAfter = n->isAfter(insert);

    // 存储原始顺序的节点指针向量
    std::vector<Node*> originalOrdering;
    // 根据 isAfter 决定起始节点是 n 的下一个节点还是前一个节点
    Node* original = isAfter ? n->next() : n->prev();

    // 遍历链表，直到回到 n 所在的块的返回节点
    auto curNode = original;
    while (curNode != n->owningBlock()->return_node()) {
      // 将当前节点 curNode 加入原始顺序向量
      originalOrdering.push_back(curNode);
      // 根据 isAfter 移动到下一个节点或前一个节点
      if (isAfter) {
        curNode = curNode->next();
      } else {
        curNode = curNode->prev();
      }
    }

    // 调用传入的 func 函数，尝试移动节点 n 到 insert 的位置，并记录结果
    const auto couldMove = func(n, insert);
    // 检查整个图的一致性
    graph->lint();

    // 再次遍历链表，检查移动节点后原始顺序是否保持不变
    curNode = original;
    size_t idx = 0;
    while (curNode != n->owningBlock()->return_node()) {
      // 使用 EXPECT_TRUE 断言原始顺序中的节点与当前遍历到的节点 curNode 相等
      EXPECT_TRUE(originalOrdering[idx] == curNode);
      // 根据 isAfter 移动到下一个节点或前一个节点
      if (isAfter) {
        curNode = curNode->next();
      } else {
        curNode = curNode->prev();
      }
      idx++;
    }

    // 返回移动是否成功的结果
    return couldMove;
  }

  // 检查移动节点后的后置条件
  void checkPostCondition(
      const std::string& toInsert,
      const std::string& insertPoint,
      bool after) {
    // 如果 after 为 true，则断言 toInsert 的前一个节点是 insertPoint
    if (after) {
      EXPECT_EQ(nodes.at(toInsert)->prev(), nodes.at(insertPoint));
    } else {
      // 如果 after 为 false，则断言 toInsert 的后一个节点是 insertPoint
      EXPECT_EQ(nodes.at(toInsert)->next(), nodes.at(insertPoint));
    }
  }

  // 定义三个成员变量 graph、aliasDb 和 nodes
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<AliasDb> aliasDb;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<std::string, Node*> nodes;
};

// TopologicalMoveTest 类的测试用例，测试拓扑排序移动操作

TEST_F(TopologicalMoveTest, SplitsDeps) {
  // 检查在需要分割 `this` 和其依赖时，是否正确移除了 `this` 的依赖关系
  EXPECT_TRUE(moveBeforeTopologicallyValid("q", "s"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("q", "s", false);
}

// Move after
TEST_F(TopologicalMoveTest, MoveAfterBackwardSimple) {
  // 简单的向后移动操作
  EXPECT_TRUE(moveAfterTopologicallyValid("c", "a"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("c", "a", true);
}

TEST_F(TopologicalMoveTest, MoveAfterBackwardInvalid) {
  // 简单的无效向后移动操作
  EXPECT_FALSE(moveAfterTopologicallyValid("d", "a"));
}

TEST_F(TopologicalMoveTest, MoveAfterNoOp) {
  // 实际上并未执行任何移动操作
  EXPECT_TRUE(moveAfterTopologicallyValid("f", "e"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("f", "e", true);
}

TEST_F(TopologicalMoveTest, MoveAfterBackwardMultipleDeps) {
  // 带有多个依赖的向后移动操作
  EXPECT_TRUE(moveAfterTopologicallyValid("e", "c"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("e", "c", true);
}

TEST_F(TopologicalMoveTest, MoveAfterBackwardNonZeroWorkingSet) {
  // 带有非零工作集的向后移动操作
  EXPECT_TRUE(moveAfterTopologicallyValid("k", "f"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("k", "f", true);
}

TEST_F(TopologicalMoveTest, MoveAfterForwardSimple) {
  // 简单的向前移动操作
  EXPECT_TRUE(moveAfterTopologicallyValid("c", "d"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("c", "d", true);
}

TEST_F(TopologicalMoveTest, MoveAfterForwardNonZeroWorkingSet) {
  // 带有非零工作集的向前移动操作
  EXPECT_TRUE(moveAfterTopologicallyValid("f", "l"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("f", "l", true);
}

// Move before
TEST_F(TopologicalMoveTest, MoveBeforeForwardSimple) {
  // 简单的向前移动操作
  EXPECT_TRUE(moveBeforeTopologicallyValid("b", "d"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("b", "d", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeBackwardSimple) {
  // 简单的向后移动操作
  EXPECT_TRUE(moveBeforeTopologicallyValid("c", "a"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("c", "a", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeNoOp) {
  // 实际上并未执行任何移动操作
  EXPECT_TRUE(moveBeforeTopologicallyValid("a", "b"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("a", "b", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeForwardWithDeps) {
  // 带有依赖关系的向前移动操作
  EXPECT_TRUE(moveBeforeTopologicallyValid("f", "m"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("f", "m", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeBackwardWithDeps) {
  // 带有依赖关系的向后移动操作
  EXPECT_TRUE(moveBeforeTopologicallyValid("l", "f"));
  // 检查移动后的状态是否符合预期
  checkPostCondition("l", "f", false);
}

// 检查在块中的依赖关系是否被正确识别
TEST_F(TopologicalMoveTest, DepsDisallowMove) {
  // 验证无法移动的依赖关系
  EXPECT_FALSE(moveAfterTopologicallyValid("l", "m"));
  EXPECT_FALSE(moveBeforeTopologicallyValid("m", "l"));
  EXPECT_FALSE(moveAfterTopologicallyValid("n", "l"));
  EXPECT_FALSE(moveBeforeTopologicallyValid("l", "n"));
}

// Test that moveAfter(n) and moveBefore(n->next()) are not necessarily
// equivalent. Here, the dependency ordering is n -> o -> p.  So we can't
// 测试案例，测试在具有依赖关系的情况下进行拓扑排序后的移动操作
TEST_F(TopologicalMoveTest, MoveAfterBeforeWithDeps) {
  // 断言：尝试将节点 "n" 移动到节点 "o" 后面，在拓扑排序下不合法
  EXPECT_FALSE(moveAfterTopologicallyValid("n", "o"));
  // 断言：尝试将节点 "o" 移动到节点 "p" 前面，在拓扑排序下合法
  EXPECT_TRUE(moveBeforeTopologicallyValid("o", "p"));
  // 检查后置条件：节点 "o" 应该在节点 "p" 前面（期望为 false）
  checkPostCondition("o", "p", false);
}

namespace {
// 如果条件满足，则在图形 g 中插入一个 If 节点，并返回该节点
Node* insertIf(
    Graph& g,
    Value* condValue,
    std::function<std::vector<Value*>()> trueInst,
    std::function<std::vector<Value*>()> falseInst) {
  // 创建一个 If 节点
  auto if_ = g.insertNode(g.create(prim::If, 0));
  // 将条件值作为输入添加到 If 节点中
  if_->addInput(condValue); // condition value
  // 添加真值分支和假值分支
  auto trueBlock = if_->addBlock();
  auto falseBlock = if_->addBlock();
  {
    // 在真值分支中进行变异
    WithInsertPoint g(trueBlock);
    // 执行真值分支的指令，并获取输出
    auto outputs = trueInst();
    for (auto output : outputs) {
      trueBlock->registerOutput(output);
    }
  }
  {
    // 在假值分支中进行变异
    WithInsertPoint g(falseBlock);
    // 执行假值分支的指令，并获取输出
    auto outputs = falseInst();
    for (auto output : outputs) {
      falseBlock->registerOutput(output);
    }
  }

  // 确保真值分支和假值分支的输出数量相同
  EXPECT_TRUE(trueBlock->outputs().size() == falseBlock->outputs().size());
  // 将真值分支的输出类型作为 If 节点的输出类型
  for (auto output : trueBlock->outputs()) {
    if_->addOutput()->setType(output->type());
  }
  // 返回创建的 If 节点
  return if_;
}

// 模板函数：期望某个操作会抛出指定异常，并检查异常消息是否包含特定字符串
template <class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    // 如果异常消息不包含期望的字符串，则抛出错误
    if (std::string(e.what()).find(expectMessageContains) ==
        std::string::npos) {
      AT_ERROR(
          "Expected error message to contain \"",
          expectMessageContains,
          "\" but error message was: ",
          e.what());
    }
    return;
  }
  // 如果没有抛出异常，则抛出错误
  AT_ERROR(
      "Expected to throw exception containing \"",
      expectMessageContains,
      "\" but didn't throw");
}

} // namespace

// 测试案例，测试别名分析在移动块操作中的行为
TEST(AliasAnalysisTest, AliasingMutationBlocksMoves) {
  // 创建一个图形对象
  auto graph = std::make_shared<Graph>();
  // 添加两个输入节点
  auto a = graph->addInput();
  auto b = graph->addInput();

  // 添加节点操作：
  // addsB = b + b
  // c = a + b
  // a += b
  // d = c + c
  auto addsB = graph->insert(aten::add, {b, b});
  auto c = graph->insert(aten::add, {a, b});
  auto aMut = graph->insert(aten::add_, {a, b});
  auto d = graph->insert(aten::add, {c, c});

  // 对图形进行 lint（静态代码分析）
  graph->lint();

  // 创建别名分析对象
  AliasDb aliasDb(graph);
  
  // 断言：无法将 c 节点移动到 aMut 节点之后，因为 aMut 节点会修改使用的值
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(c->node(), aMut->node()));
  // 断言：可以将 d 节点移动到 c 节点之后，因为它们没有依赖关系
  EXPECT_TRUE(aliasDb.moveAfterTopologicallyValid(d->node(), c->node()));

  // 断言：由于 b 和 a 都是输入，它们会发生别名，所以无法将 addsB 节点移动到 aMut 节点之后
  EXPECT_FALSE(
      aliasDb.moveAfterTopologicallyValid(addsB->node(), aMut->node()));
  // 断言：可以将 addsB 节点移动到 c 节点之后，因为它们之间没有依赖关系
  EXPECT_TRUE(aliasDb.moveAfterTopologicallyValid(addsB->node(), c->node()));

  // 再次对图形进行 lint（静态代码分析）
  graph->lint();
}
TEST(AliasAnalysisTest, AliasingMutationBlocksMoves2) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 向计算图添加两个输入节点
  auto a = graph->addInput();
  auto b = graph->addInput();

  // 向计算图插入一个常量节点
  auto constant = graph->insertConstant(1);
  // 向计算图插入一个随机数生成节点
  auto fresh = graph->insert(aten::rand, {constant});
  // 向计算图插入一个加法节点，将输入节点 b 和随机数节点 fresh 相加
  auto usesB = graph->insert(aten::add, {b, fresh});
  // 向计算图插入一个选择节点，选择节点 a 的常量索引的值
  auto aliasesB = graph->insert(aten::select, {a, constant, constant});
  // 向计算图插入一个原地加法节点，将选择节点 aliasesB 和随机数节点 fresh 原地相加
  auto mutatesAliasOfB = graph->insert(aten::add_, {aliasesB, fresh});
  // 向计算图插入一个加法节点，将随机数节点 fresh 和选择节点 aliasesB 相加
  graph->insert(aten::add, {fresh, aliasesB});
  // 对计算图进行静态检查
  graph->lint();

  // 创建一个别名分析数据库对象，传入当前计算图
  AliasDb aliasDb(graph);
  // 断言不可以将 aliasesB 节点移动到 mutatesAliasOfB 节点之后，因为可能存在别名问题
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(
      aliasesB->node(), mutatesAliasOfB->node()));
  // 断言不可以将 usesB 节点移动到 mutatesAliasOfB 节点之后，因为可能存在别名问题
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(
      usesB->node(), mutatesAliasOfB->node()));
}

TEST(AliasAnalysisTest, SideEffectsBlockMoves) {
  // 测试跨侧效节点的移动
  auto graph = std::make_shared<Graph>();
  // 向计算图添加一个输入节点
  auto a = graph->addInput();
  // 向计算图插入一个打印节点 print1
  auto print1 = graph->insertNode(graph->create(prim::Print, {a}, 0));
  // 设置插入点为 print1
  WithInsertPoint guard(print1);
  // 向计算图插入另一个打印节点 print2
  auto print2 = graph->insertNode(graph->create(prim::Print, {a, a}, 0));
  // 创建一个别名分析数据库对象，传入当前计算图
  AliasDb aliasDb(graph);

  // def foo(a):
  //  print2(a, a)
  //  print1(a)

  // 测试在彼此之间移动
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(print2, print1));
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(print1, print2));

  // 测试在它们已经存在的位置移动
  EXPECT_TRUE(aliasDb.moveBeforeTopologicallyValid(print2, print1));
  EXPECT_TRUE(aliasDb.moveAfterTopologicallyValid(print1, print2));

  // 向计算图插入一个创建测试张量的节点
  graph->insertNode(graph->create(prim::MakeTestTensor, {}, 1));
  // 创建一个新的别名分析数据库对象，传入更新后的计算图
  AliasDb aliasDb2(graph);

  // def foo(a):
  //  print2(a, a)
  //  non_side_effectful = makeTestTensor()
  //  print1(a)

  // 测试在具有侧效节点的情况下移动
  EXPECT_FALSE(aliasDb2.moveAfterTopologicallyValid(print2, print1));
  EXPECT_FALSE(aliasDb2.moveBeforeTopologicallyValid(print2, print1));
  EXPECT_FALSE(aliasDb2.moveAfterTopologicallyValid(print1, print2));
  EXPECT_FALSE(aliasDb2.moveBeforeTopologicallyValid(print1, print2));
}

TEST(AliasAnalysisTest, MovingAcrossInnerBlocks) {
  // 测试跨内部块的移动

  // a = rand(1)
  // b = rand(1)
  // if True:
  //   a.add_(b)
  // c = a + b
  auto graph = std::make_shared<Graph>();
  // 向计算图插入一个常量节点
  auto constant = graph->insertConstant(1);
  // 向计算图插入一个随机数生成节点 a
  auto a = graph->insert(aten::rand, {constant});
  // 向计算图插入一个随机数生成节点 b
  auto b = graph->insert(aten::rand, {constant});

  // 向计算图插入一个条件语句块
  auto if_ = insertIf(
      *graph,
      constant,
      [&]() -> std::vector<Value*> {
        // 在条件块内插入一个原地加法节点，将 a 和 b 原地相加
        auto aMut = graph->insert(aten::add_, {a, b});
        return {aMut};
      },
      [&]() -> std::vector<Value*> { return {a}; });

  // 向计算图插入一个加法节点，将 a 和 b 相加
  auto c = graph->insert(aten::add, {a, b});

  // 对计算图进行静态检查
  graph->lint();

  // 创建一个别名分析数据库对象，传入当前计算图
  AliasDb aliasDb(graph);
  // 断言不可以将 c 节点移动到 if 语句之前，因为可能会写入 a
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(c->node(), if_));
}
TEST(AliasAnalysisTest, NoneHasNoWriters) {
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();
  // 创建一个从字符串到值的无序映射
  std::unordered_map<std::string, Value*> vmap;
  // 解析内联的 IR 代码到计算图中，并填充映射
  parseIR(
      R"IR(
    graph():
      %opt : Tensor? = prim::Constant()
      %out : Tensor = prim::unchecked_unwrap_optional(%opt)
      %ret.2 : Tensor = aten::div(%out, %out, %out)
      return (%opt, %out, %ret.2)
      )IR",
      &*graph,
      vmap);

  // 构建基于计算图的别名分析对象
  AliasDb aliasDb(graph);
  // 断言 vmap 中名为 "opt" 的值没有写入操作
  EXPECT_FALSE(aliasDb.hasWriters(vmap["opt"]->node()));
}

TEST(AliasAnalysisTest, SafeToChangeAliasingRelationship) {
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();
  // 创建一个从字符串到值的无序映射
  std::unordered_map<std::string, Value*> vmap;
  // 解析内联的 IR 代码到计算图中，并填充映射
  parseIR(
      R"IR(
  graph(%x : Tensor):
      %3 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=0]()
      %b : Tensor = aten::add(%x, %2, %3)
      %c : Tensor = aten::add(%x, %2, %3)
      %d : Tensor = aten::add(%x, %2, %3)
      %e : Tensor = aten::add(%x, %2, %3)
      %f : Tensor[] = prim::ListConstruct(%e)
      %14 : (Tensor, Tensor) = prim::TupleConstruct(%b, %c)
      return (%14)
    )IR",
      &*graph,
      vmap);

  // 构建基于计算图的别名分析对象
  AliasDb aliasDb(graph);

  // 断言不能改变以下变量之间的别名关系
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["x"], vmap["b"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["b"], vmap["x"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["b"], vmap["c"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["c"], vmap["b"]));

  // 断言变量 "e" 由于被包含在列表中，与 wildcard 集合发生别名关系
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["e"], vmap["x"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["x"], vmap["e"]));

  // 断言变量 "d" 是一个临时变量且没有写入操作，因此可以安全改变别名关系
  EXPECT_TRUE(aliasDb.safeToChangeAliasingRelationship(vmap["c"], vmap["d"]));
  EXPECT_TRUE(aliasDb.safeToChangeAliasingRelationship(vmap["d"], vmap["c"]));
}

class BatchAndInstanceNormFixture
    : public ::testing::TestWithParam<std::tuple<std::string, NodeKind, bool>> {
};

TEST_P(BatchAndInstanceNormFixture, BatchAndInstanceNorm) {
  // 获取测试参数
  auto param = GetParam();
  auto fnName = std::get<0>(param);
  auto nodeKind = std::get<1>(param);
  auto isTraining = std::get<2>(param);
  // 将 bool 值转换为字符串
  std::string isTrainingStr = std::to_string((int)isTraining);

  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();

  // 解析内联的 IR 代码到计算图中
  parseIR(
      R"IR(
  graph(%input : Tensor, %running_mean : Tensor, %running_var : Tensor):
      %none : NoneType = prim::Constant()
      %training : bool = prim::Constant[value=)IR" +
          isTrainingStr + R"IR(]()
      %momentum : float = prim::Constant[value=1.0]()
      %eps : float = prim::Constant[value=1.0e-9]()
      %cudnn_enabled : bool = prim::Constant[value=0]()
      %res : Tensor = )IR" +
          fnName +
          R"IR((%input, %none, %none, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
      return (%res)
  )IR",
      &*graph);

  # 调用 graph 对象的 lint 方法，用于检查和修正图的结构和内容
  graph->lint();

  # 创建 DepthFirstGraphNodeIterator 对象 it，用于深度优先遍历图中的节点
  DepthFirstGraphNodeIterator it(graph);

  # 初始化指针 n 为 nullptr
  Node* n = nullptr;

  # 循环迭代图中的节点，直到找到满足条件的节点或遍历完所有节点
  while ((n = it.next()) != nullptr) {
    # 如果当前节点 n 的类型与 nodeKind 匹配，则跳出循环
    if (n->kind() == nodeKind) {
      break;
    }
  }

  # 断言找到的节点 n 不为空
  EXPECT_TRUE(n != nullptr);

  # 创建 AliasDb 对象 aliasDb，用于管理节点之间的别名信息
  AliasDb aliasDb(graph);

  # 断言节点 n 是否有写入操作，结果应该与 isTraining 相符
  EXPECT_TRUE(aliasDb.hasWriters(n) == isTraining);
TEST_P(BatchAndInstanceNormFixture, BatchAndInstanceNormTrainingUnknown) {
  // 获取测试参数
  auto param = GetParam();
  // 获取函数名
  auto fnName = std::get<0>(param);
  // 获取节点类型
  auto nodeKind = std::get<1>(param);

  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();

  // 解析内部表示(IR)，构建图形
  parseIR(
      R"IR(
  graph(%input : Tensor, %running_mean : Tensor, %running_var : Tensor, %training : bool):
      %none : NoneType = prim::Constant()
      %momentum : float = prim::Constant[value=1.0]()
      %eps : float = prim::Constant[value=1.0e-9]()
      %cudnn_enabled : bool = prim::Constant[value=0]()
      %res : Tensor = )IR" +
          fnName +
          R"IR((%input, %none, %none, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
      return (%res)
    )IR",
      &*graph);

  // 对图形进行静态分析
  graph->lint();

  // 创建深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(graph);

  // 初始化节点指针
  Node* n = nullptr;

  // 遍历图节点，直到找到指定类型的节点
  while ((n = it.next()) != nullptr) {
    if (n->kind() == nodeKind) {
      break;
    }
  }

  // 断言找到了目标节点
  EXPECT_TRUE(n != nullptr);

  // 创建别名分析对象
  AliasDb aliasDb(graph);

  // 断言目标节点有写操作
  EXPECT_TRUE(aliasDb.hasWriters(n));
}

TEST_P(BatchAndInstanceNormFixture, BatchNormTrainingWithNoMeanOrVar) {
  // 获取测试参数
  auto param = GetParam();
  // 获取函数名
  auto fnName = std::get<0>(param);
  // 获取节点类型
  auto nodeKind = std::get<1>(param);
  // 获取是否训练的标志
  auto isTraining = std::get<2>(param);
  // 转换是否训练的标志为字符串
  std::string isTrainingStr = std::to_string((int)isTraining);

  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();

  // 解析内部表示(IR)，构建图形
  parseIR(
      R"IR(
  graph(%input : Tensor):
      %none : NoneType = prim::Constant()
      %training : bool = prim::Constant[value=)IR" +
          isTrainingStr + R"IR(]()
      %momentum : float = prim::Constant[value=1.0]()
      %eps : float = prim::Constant[value=1.0e-9]()
      %cudnn_enabled : bool = prim::Constant[value=0]()
      %res : Tensor = )IR" +
          fnName +
          R"IR((%input, %none, %none, %none, %none, %training, %momentum, %eps, %cudnn_enabled)
      return (%res)
    )IR",
      &*graph);

  // 对图形进行静态分析
  graph->lint();

  // 创建深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(graph);

  // 初始化节点指针
  Node* n = nullptr;

  // 遍历图节点，直到找到指定类型的节点
  while ((n = it.next()) != nullptr) {
    if (n->kind() == nodeKind) {
      break;
    }
  }

  // 断言找到了目标节点
  EXPECT_TRUE(n != nullptr);

  // 创建别名分析对象
  AliasDb aliasDb(graph);

  // 断言目标节点没有写操作
  EXPECT_FALSE(aliasDb.hasWriters(n));
}

// 实例化测试套件，用于别名分析测试
INSTANTIATE_TEST_SUITE_P(
    AliasAnalysisTest,
    BatchAndInstanceNormFixture,
    ::testing::Values(
        std::make_tuple("aten::batch_norm", aten::batch_norm, false),
        std::make_tuple("aten::instance_norm", aten::instance_norm, false),
        std::make_tuple("aten::batch_norm", aten::batch_norm, true),
        std::make_tuple("aten::instance_norm", aten::instance_norm, true)));
TEST(WriteTrackingTest, Basic) {
  // 注册运算符 "prim::creates_alias(Tensor(a) x) -> Tensor(a)" 到系统
  RegisterOperators reg({Operator(
      "prim::creates_alias(Tensor(a) x) -> Tensor(a)",
      [](Stack&) {},
      aliasAnalysisFromSchema())});
  // 创建创建别名操作的符号
  const auto creates_alias = Symbol::fromQualString("prim::creates_alias");
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 向计算图添加两个输入节点
  auto a = graph->addInput();
  auto b = graph->addInput();

  // 插入操作节点 aten::add(%b, %b)
  auto pureNode = graph->insert(aten::add, {b, b})->node();
  // 插入操作节点 aten::add_(%a, %b)
  auto writingNode = graph->insert(aten::add_, {a, b})->node();
  // 插入操作节点 foo::creates_alias(%a)
  auto node3 = graph->insert(creates_alias, {a})->node();
  // 获取 foo::creates_alias 的输出
  auto aAlias = node3->output();

  // 对计算图进行 lint 检查
  graph->lint();

  // 创建别名数据库对象
  AliasDb aliasDb(graph);
  // 检查 aAlias 和 a 是否可能存在别名关系
  EXPECT_TRUE(aliasDb.mayAlias(aAlias, a));
  // 检查 a 和 b 是否可能存在别名关系
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
  // 检查 pureNode 是否写入了 a 的别名
  EXPECT_FALSE(
      aliasDb.writesToAlias(pureNode, std::unordered_set<const Value*>{a}));
  // 检查 pureNode 是否写入了 b 的别名
  EXPECT_FALSE(
      aliasDb.writesToAlias(pureNode, std::unordered_set<const Value*>{b}));
  // 检查 writingNode 是否写入了 a 的别名
  EXPECT_TRUE(
      aliasDb.writesToAlias(writingNode, std::unordered_set<const Value*>{a}));
  // 检查 writingNode 是否写入了 a 和 b 的别名
  EXPECT_TRUE(aliasDb.writesToAlias(
      writingNode, std::unordered_set<const Value*>{a, b}));
  // 检查 writingNode 是否写入了 aAlias 的别名
  EXPECT_TRUE(aliasDb.writesToAlias(
      writingNode, std::unordered_set<const Value*>{aAlias}));
}

TEST(WriteTrackingTest, IsMutable) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 解析给定的 IR，构建计算图
  parseIR(
      R"IR(
  graph(%x: Tensor):
    %b : Tensor = aten::relu_(%x)
    return (%b)
    )IR",
      &*graph);
  // 获取计算图的第一个节点迭代器
  auto node_iter = graph->block()->nodes().begin();
  // 获取计算图的第一个节点（这里是 relu_ 操作）
  auto relu = *node_iter;
  // 创建别名数据库对象
  AliasDb aliasDb(graph);
  // 检查 relu 操作节点是否可变
  EXPECT_TRUE(aliasDb.isMutable(relu));
}

TEST(WriteTrackingTest, IsImmutable) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 解析给定的 IR，构建计算图
  parseIR(
      R"IR(
  graph(%x: Tensor, %y : Tensor):
    %b : Tensor = aten::mul(%x, %y)
    return (%b)
    )IR",
      &*graph);
  // 获取计算图的第一个节点迭代器
  auto node_iter = graph->block()->nodes().begin();
  // 获取计算图的第一个节点（这里是 mul 操作）
  auto mul = *node_iter;
  // 创建别名数据库对象
  AliasDb aliasDb(graph);
  // 检查 mul 操作节点是否不可变
  EXPECT_FALSE(aliasDb.isMutable(mul));
}

TEST(WriteTrackingTest, HasWriters) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 定义一个映射从字符串到值的哈希表
  std::unordered_map<std::string, Value*> vmap;
  // 解析给定的 IR，构建计算图，并更新 vmap
  parseIR(
      R"IR(
  graph(%x: Tensor, %y : Tensor):
    %c1 : int = prim::Constant[value=1]()
    %b : Tensor = aten::add_(%x, %y, %c1)
    return (%b)
    )IR",
      &*graph,
      vmap);
  // 获取节点 b 的引用
  auto add = vmap["b"]->node();
  // 创建别名数据库对象
  AliasDb aliasDb(graph);
  // 检查节点 add 是否有写入操作
  EXPECT_TRUE(aliasDb.hasWriters(add));
  // 检查节点 add 是否可变
  EXPECT_TRUE(aliasDb.isMutable(add));
}

TEST(ContainerAliasingTest, MayContainAlias) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 定义一个映射从字符串到值的哈希表
  std::unordered_map<std::string, Value*> vmap;
  // 解析给定的 IR，构建计算图，并更新 vmap
  parseIR(
      R"IR(
  graph(%inp: Tensor[]):
    %x : str = prim::Constant[value="a"]()
    %y : Tensor = prim::Constant()
    %z : Tensor = prim::Constant()
    %a : (Tensor) = prim::TupleConstruct(%y)
    %b : Dict(str, Tensor) = prim::DictConstruct(%x, %y)
    %c : Tensor[] = prim::ListConstruct(%y)
    return (%a, %b, %c)
    )IR",
      &*graph,
      vmap);
  // 这里没有进行节点操作，因此没有创建 AliasDb 对象，也没有断言
}
  )IR",
      &*graph,
      vmap);

  // 从 vmap 中获取键为 "x" 的值，表示字符串类型的输出
  auto str_output = vmap["x"];
  // 从 vmap 中获取键为 "y" 的值，表示张量类型的输出
  auto ten_output = vmap["y"];
  // 从 vmap 中获取键为 "z" 的值，表示本地变量
  auto local_var = vmap["z"];
  
  // 创建 AliasDb 对象，传入图形 graph，用于分析别名
  AliasDb aliasDb(graph);

  // 断言图形的输出数量为 3
  EXPECT_TRUE(graph->outputs().size() == 3);
  // 遍历图形的每个输出
  for (auto out : graph->outputs()) {
    // 断言 ten_output 可能与当前输出存在别名关系
    EXPECT_TRUE(aliasDb.mayContainAlias(ten_output, out));
    // 断言 local_var 与当前输出不存在别名关系
    EXPECT_FALSE(aliasDb.mayContainAlias(local_var, out));
  }

  // 断言 ten_output 可能与图形的输入存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_output, graph->inputs()));
  // 断言 local_var 与图形的输入不存在别名关系
  EXPECT_FALSE(aliasDb.mayContainAlias(local_var, graph->inputs()));

  // 断言 ten_output 可能与图形的输出存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_output, graph->outputs()));
  // 断言 ten_output 可能与 at::ArrayRef<Value*>{ten_output} 中的元素存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(
      at::ArrayRef<Value*>{ten_output}, graph->outputs()));
  // 断言 str_output 与图形的输出不存在别名关系
  EXPECT_FALSE(aliasDb.mayContainAlias(str_output, graph->outputs()));
TEST(ContainerAliasingTest, MayContainAlias_cast) {
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针 graph，指向一个 Graph 对象
  std::unordered_map<std::string, Value*> vmap;  // 创建一个无序映射 vmap，用于存储字符串到 Value* 的映射关系
  parseIR(
      R"IR(
  graph(%input.1 : Tensor):
    %2 : NoneType = prim::Constant()  // 定义一个名为 %2 的常量，类型为 NoneType
    %3 : bool = prim::Constant[value=0]()  // 定义一个名为 %3 的常量，布尔值为 false
    %4 : int = prim::Constant[value=6]()  // 定义一个名为 %4 的常量，整数值为 6
    %5 : int = prim::Constant[value=1]()  // 定义一个名为 %5 的常量，整数值为 1
    %a.1 : Tensor = aten::add(%input.1, %input.1, %5)  // 执行张量加法操作，结果存储在 %a.1 中
    %b.1 : Tensor = aten::to(%a.1, %4, %3, %3, %2)  // 执行类型转换操作，结果存储在 %b.1 中
    %c.1 : Tensor = aten::mul(%b.1, %b.1)  // 执行张量乘法操作，结果存储在 %c.1 中
    return (%c.1)  // 返回 %c.1 作为图的输出
    )IR",
      &*graph,
      vmap);  // 解析内部表示字符串，填充图和映射 vmap

  auto a = vmap["a.1"];  // 获取 vmap 中键为 "a.1" 的值，并赋给变量 a
  auto b = vmap["b.1"];  // 获取 vmap 中键为 "b.1" 的值，并赋给变量 b
  auto c = vmap["c.1"];  // 获取 vmap 中键为 "c.1" 的值，并赋给变量 c
  AliasDb aliasDb(graph);  // 创建一个 AliasDb 对象，传入图 graph 以构建别名数据库

  EXPECT_TRUE(graph->outputs().size() == 1);  // 断言图的输出节点数为 1
  for (auto out : graph->outputs()) {  // 遍历图的输出节点
    EXPECT_TRUE(aliasDb.mayContainAlias(c, out));  // 断言别名数据库中是否可能包含 c 和当前输出节点的别名关系
  }

  EXPECT_TRUE(aliasDb.mayContainAlias(a, b));  // 断言别名数据库中是否可能包含 a 和 b 的别名关系
  EXPECT_FALSE(aliasDb.mayContainAlias(b, graph->inputs()));  // 断言别名数据库中是否不可能包含 b 和图的输入节点的别名关系

  EXPECT_TRUE(aliasDb.mayContainAlias(c, graph->outputs()));  // 断言别名数据库中是否可能包含 c 和图的输出节点的别名关系
  EXPECT_TRUE(
      aliasDb.mayContainAlias(at::ArrayRef<Value*>{c}, graph->outputs()));  // 断言别名数据库中是否可能包含 c 和图的输出节点的别名关系（通过 ArrayRef<Value*> 参数）
  EXPECT_FALSE(aliasDb.mayContainAlias(b, graph->outputs()));  // 断言别名数据库中是否不可能包含 b 和图的输出节点的别名关系
}

TEST(ContainerAliasingTest, PrimitveValuesDontAliasContainers) {
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针 graph，指向一个 Graph 对象
  parseIR(
      R"IR(
  graph():
    %x : str = prim::Constant[value="a"]()  // 定义一个名为 %x 的常量，字符串值为 "a"
    %y : int = prim::Constant[value=1]()  // 定义一个名为 %y 的常量，整数值为 1
    %a : (int) = prim::TupleConstruct(%y)  // 构造一个整数元组 %a，包含整数值 %y
    %b : Dict(str, int) = prim::DictConstruct(%x, %y)  // 构造一个字典 %b，包含键为 %x，值为 %y
    %c : int[] = prim::ListConstruct(%y)  // 构造一个整数数组 %c，包含整数值 %y
    return (%a, %b, %c)  // 返回 %a, %b, %c 作为图的输出
    )IR",
      &*graph);

  auto node_iter = graph->block()->nodes().begin();
  node_iter++; // string  // 迭代器指向下一个节点，类型为字符串
  Node* int_node = *node_iter++;  // 获取下一个节点，并将其赋给 int_node，类型为整数
  AliasDb aliasDb(graph);  // 创建一个 AliasDb 对象，传入图 graph 以构建别名数据库

  EXPECT_TRUE(graph->outputs().size() == 3);  // 断言图的输出节点数为 3
  // primitive values don't need to alias container
  for (auto out : graph->outputs()) {  // 遍历图的输出节点
    EXPECT_FALSE(aliasDb.mayContainAlias(int_node->output(), out));  // 断言别名数据库中是否不可能包含 int_node 的输出和当前输出节点的别名关系
  }
}

TEST(ContainerAliasingTest, UnionAliasing) {
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针 graph，指向一个 Graph 对象
  parseIR(
      R"IR(
  graph(%a : Dict(str, Tensor),
        %b : Tensor[],
        %c : Union(Dict(str, Tensor), Tensor[])):
    return (%a, %b, %c)  // 返回 %a, %b, %c 作为图的输出
    )IR",
      &*graph);

  AliasDb aliasDb(graph);  // 创建一个 AliasDb 对象，传入图 graph 以构建别名数据库
  auto a = graph->outputs().at(0);  // 获取图的第一个输出节点，并赋给变量 a
  auto b = graph->outputs().at(1);  // 获取图的第二个输出节点，并赋给变量 b
  auto c = graph->outputs().at(2);  // 获取图的第三个输出节点，并赋给变量 c

  EXPECT_TRUE(aliasDb.mayAlias(a, c));  // 断言别名数据库中是否可能存在 a 和 c 的别名关系
  EXPECT_TRUE(aliasDb.mayAlias(b, c));  // 断言别名数据库中是否可能存在 b 和 c 的别名关系
  EXPECT_TRUE(aliasDb.mayAlias(c, c));  // 断言别名数据库中是否可能存在 c 和 c 的别名关系
  EXPECT_FALSE(aliasDb.mayAlias(a, b));  // 断言别名数据库中是否不可能存在 a 和 b 的别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(a, b));  // 断言别名数据库中是否可能包含 a 和 b 的别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(a, c));  // 断言别名数据库中是否可能包含 a 和 c 的别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(b, c));  // 断言别名数据库中是否可能包含 b 和 c 的别名关系
}

TEST(ContainerAliasingTest, InputsCanAliasOutputs) {
  // Test input aliasing
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针 graph，指向一个 Graph 对象
  parseIR(
      R"IR(
  graph(%x: Tensor, %y: Tensor):
    %a : (Tensor) = prim::TupleConstruct(%x)  // 构造一个张量元组 %a，包含张量 %x
    return (%a)  // 返回 %a 作为图的输出
    )IR",
      &*graph);

  auto node_iter = graph->block()->nodes().begin();  // 获取图的节点迭代器
  auto tuple_node = *node_iter;  // 获取第一个节点，并赋给 tuple_node
  AliasDb aliasDb(graph);  // 创建一个 AliasDb 对象，传入图 graph 以构建别名数据库

  for (auto input : graph->inputs()) {  // 遍历图的输入节点
    # 断言在aliasDb中可能包含给定的别名
    EXPECT_TRUE(aliasDb.mayContainAlias(input, tuple_node->output()));
  }
  # 断言在aliasDb中可能包含图的输入和输出之间的别名
  EXPECT_TRUE(aliasDb.mayContainAlias(graph->inputs(), graph->outputs()));
// Test tuple that doesn't come from construct
TEST(ContainerAliasingTest, NestedTupleConstruct) {
  // 创建一个新的图对象
  auto graph = std::make_shared<Graph>();
  // 解析提供的IR代码片段并将其加载到图中
  parseIR(
      R"IR(
graph(%x : int,
      %y : Tensor,
      %z : Tensor):
  %3 : int = prim::Constant[value=1]()
  %4 : bool = aten::eq(%x, %3)
  %a : (Tensor) = prim::If(%4)
    block0():
      %a.1 : (Tensor) = prim::TupleConstruct(%y)
      -> (%a.1)
    block1():
      %a.2 : (Tensor) = prim::TupleConstruct(%z)
      -> (%a.2)
  return (%a)
 )IR",
      &*graph);

  // 创建一个别名分析数据库对象，用于分析图中的别名关系
  AliasDb aliasDb(graph);

  // 遍历图的输入节点
  for (auto input : graph->inputs()) {
    // 如果输入节点的类型是整数类型，跳过分析
    if (input->type() == IntType::get()) {
      continue;
    }
    // 断言输入节点和图的输出节点存在别名关系
    EXPECT_TRUE(aliasDb.mayContainAlias(input, graph->outputs().at(0)));
  }
}

// test nested types
TEST(ContainerAliasingTest, NestedTypes) {
  // 创建一个新的图对象
  auto graph = std::make_shared<Graph>();
  // 解析提供的IR代码片段并将其加载到图中
  parseIR(
      R"IR(
graph():
  %a : Tensor = prim::MakeTestTensor()
  %a_list : Tensor[] = prim::ListConstruct(%a)
  %b : Tensor = prim::MakeTestTensor()
  %b_list : Tensor[] = prim::ListConstruct(%b)
  %13 : (Tensor[], Tensor[]) = prim::TupleConstruct(%a_list, %b_list)
  return (%13)
)IR",
      &*graph);
  // 创建一个别名分析数据库对象，用于分析图中的别名关系
  AliasDb aliasDb(graph);
  // 获取图的输出节点
  auto g_output = graph->outputs().at(0);
  // 获取输出节点中的两个输入列表
  auto list_2 = g_output->node()->inputs().at(0);
  auto list_1 = g_output->node()->inputs().at(1);

  // 对于列表1和列表2，假设它们可能存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(list_1, list_2));
  EXPECT_TRUE(aliasDb.mayContainAlias(list_2, list_1));

  // 假设列表1和列表2可能与图的输出节点存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(list_1, g_output));
  EXPECT_TRUE(aliasDb.mayContainAlias(list_2, g_output));
}

// simple example
TEST(ContainerAliasingTest, Simple) {
  // 创建一个新的图对象
  auto graph = std::make_shared<Graph>();
  // 解析提供的IR代码片段并将其加载到图中
  parseIR(
      R"IR(
graph():
  %0 : Tensor = prim::Constant()
  %1 : Tensor = prim::Constant()
  %13 : (Tensor) = prim::TupleConstruct(%0)
  return (%13)
)IR",
      &*graph);
  // 创建一个别名分析数据库对象，用于分析图中的别名关系
  AliasDb aliasDb(graph);

  // 获取图中的第一个、第二个张量节点和元组节点
  auto node_iter = graph->block()->nodes().begin();
  auto first_ten = *node_iter++;
  auto second_ten = *node_iter++;
  auto tup_node = *node_iter;

  // 假设第一个张量节点和元组节点存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(first_ten->output(), tup_node->output()));
  // 假设第二个张量节点和元组节点不存在别名关系
  EXPECT_TRUE(
      !aliasDb.mayContainAlias(second_ten->output(), tup_node->output()));

  // 创建包含第一个、第二个张量节点和元组节点的向量
  std::vector<Value*> first_st = {first_ten->output()};
  std::vector<Value*> second_st = {second_ten->output()};
  std::vector<Value*> tup_st = {tup_node->output()};
  // 假设第一个张量节点和元组节点存在别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(first_st, tup_st));
  // 假设第一个张量节点和第二个张量节点不存在别名关系
  EXPECT_FALSE(aliasDb.mayContainAlias(first_st, second_st));
  // 假设第二个张量节点和元组节点不存在别名关系
  EXPECT_FALSE(aliasDb.mayContainAlias(second_st, tup_st));
}

TEST(ContainerAliasingTest, Lists) {
  // 创建一个新的图对象
  auto graph = std::make_shared<Graph>();
  // 创建一个无序映射，用于存储字符串到值的映射关系
  std::unordered_map<std::string, Value*> vmap;
  // 解析提供的IR代码片段并将其加载到图中
  parseIR(
      R"IR(
  graph():
    %x : str = prim::Constant[value="a"]()
    %y : Tensor = prim::Constant()
    %c : Tensor[] = prim::ListConstruct(%y)
    %d : Tensor[] = prim::ListConstruct(%y)
    return (%c, %d)
)IR",
      &*graph);
  )IR",
      &*graph,
      vmap);


# 调用`AliasDb`的构造函数，使用给定的`graph`作为参数初始化别名数据库。
  AliasDb aliasDb(graph);


  auto x = vmap["x"];
  auto c = vmap["c"];

# 从映射`vmap`中获取键为"x"和"c"的值，分别存储在变量`x`和`c`中。


  EXPECT_FALSE(aliasDb.mayContainAlias(x, c));
  EXPECT_FALSE(aliasDb.mayContainAlias(c, x));

# 断言`aliasDb`中不存在`x`和`c`之间的别名关系，以及不存在`c`和`x`之间的别名关系。


  auto d = vmap["d"];

# 从映射`vmap`中获取键为"d"的值，存储在变量`d`中。


  EXPECT_TRUE(aliasDb.mayContainAlias(d, c));
  EXPECT_TRUE(aliasDb.mayContainAlias(c, d));

# 断言`aliasDb`中存在`d`和`c`之间的别名关系，以及存在`c`和`d`之间的别名关系。
TEST(ContainerAliasingTest, Lists2) {
  // Test list container aliasing

  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 创建一个从字符串到指针的无序映射
  std::unordered_map<std::string, Value*> vmap;
  // 解析内部表示，并将其添加到图形和值映射中
  parseIR(
      R"IR(
graph():
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %x : Tensor = prim::MakeTestTensor()
  %12 : int[] = prim::ListConstruct(%0, %1)
  %y : Tensor = prim::MakeTestTensor()
  %22 : int[] = prim::ListConstruct(%0, %1)
  %z : Tensor = prim::MakeTestTensor()
  %32 : int[] = prim::ListConstruct(%0, %1)
  %fresh : Tensor = prim::MakeTestTensor()
  %foo : Tensor[] = prim::ListConstruct(%x, %y)
  %43 : Tensor[] = aten::append(%foo, %z)
  return ()
)IR",
      graph.get(),
      vmap);
  // 创建一个别名分析数据库，基于给定的图形
  AliasDb aliasDb(graph);
  // 获取特定键的值，这里是张量 x、y、z
  auto x = vmap["x"];
  auto y = vmap["y"];
  auto z = vmap["z"];
  // 断言 x、y、z 可能相互别名，因为它们都在列表中
  EXPECT_TRUE(aliasDb.mayAlias(x, y));
  EXPECT_TRUE(aliasDb.mayAlias(y, z));
  EXPECT_TRUE(aliasDb.mayAlias(x, z));

  // 由于 fresh 没有进入列表，因此 x、y、z 不应该与其别名
  auto fresh = vmap["fresh"];
  EXPECT_FALSE(aliasDb.mayAlias(x, fresh));
  EXPECT_FALSE(aliasDb.mayAlias(y, fresh));
  EXPECT_FALSE(aliasDb.mayAlias(z, fresh));
}
// 定义一个新的计算图函数 `graph()`
graph():
  // 创建整数常量 %35 并赋值为 1
  %35 : int = prim::Constant[value=1]()
  // 创建整数常量 %0 并赋值为 2
  %0 : int = prim::Constant[value=2]()
  // 创建整数常量 %1 并赋值为 3
  %1 : int = prim::Constant[value=3]()
  // 创建整数常量 %23 并赋值为 0
  %23 : int = prim::Constant[value=0]()
  // 创建整数数组 %2，包含常量 %0 和 %1
  %2 : int[] = prim::ListConstruct(%0, %1)
  // 调用 prim::MakeTestTensor() 创建 Tensor %11
  %11 : Tensor = prim::MakeTestTensor()
  // 创建整数数组 %12，包含常量 %0 和 %1
  %12 : int[] = prim::ListConstruct(%0, %1)
  // 调用 prim::MakeTestTensor() 创建 Tensor %21
  %21 : Tensor = prim::MakeTestTensor()
  // 创建 Tensor 数组 %l，包含 Tensor %11 和 %21
  %l : Tensor[] = prim::ListConstruct(%11, %21)
  // 使用 aten::select() 从 %l 中选择索引为 %23 的元素，得到 Tensor %24
  %24 : Tensor = aten::select(%l, %23)
  // 创建整数数组 %25，包含常量 %0 和 %1
  %25 : int[] = prim::ListConstruct(%0, %1)
  // 调用 prim::MakeTestTensor() 创建 Tensor %34
  %34 : Tensor = prim::MakeTestTensor()
  // 调用 aten::add_() 将 %34 加到 %24 上，结果为 Tensor %36
  %36 : Tensor = aten::add_(%24, %34, %35)
  // 使用 uses::list() 获取 Tensor 数组 %l
  %37 : Tensor = uses::list(%l)
  // 返回 %37
  return (%37)



// 创建一个新的测试用例 ContainerAliasingTest.MovesAcrossContainedWritesNested
TEST(ContainerAliasingTest, MovesAcrossContainedWritesNested) {
  // 在 torch 中注册操作 uses::list，定义其行为和别名分析类型
  auto ops = torch::RegisterOperators().op(
      "uses::list",
      torch::RegisterOperators::options()
          .catchAllKernel([](torch::List<at::Tensor> in) {
            // 返回一个形状为 {2, 3} 的随机 Tensor
            return torch::rand({2, 3});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  // 创建一个新的计算图 graph，并初始化一个值映射 vmap
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  // 解析内部表示的 IR，将其加载到 graph 中，并更新 vmap
  parseIR(
      R"IR(
graph():
  // 创建整数常量 %38 并赋值为 1
  %38 : int = prim::Constant[value=1]()
  // 创建整数常量 %0 并赋值为 2
  %0 : int = prim::Constant[value=2]()
  // 创建整数常量 %1 并赋值为 3
  %1 : int = prim::Constant[value=3]()
  // 创建整数常量 %24 并赋值为 0
  %24 : int = prim::Constant[value=0]()
  // 创建整数数组 %2，包含常量 %0 和 %1
  %2 : int[] = prim::ListConstruct(%0, %1)
  // 调用 prim::MakeTestTensor() 创建 Tensor %11
  %11 : Tensor = prim::MakeTestTensor()
  // 创建整数数组 %12，包含常量 %0 和 %1
  %12 : int[] = prim::ListConstruct(%0, %1)
  // 调用 prim::MakeTestTensor() 创建 Tensor %21
  %21 : Tensor = prim::MakeTestTensor()
  // 创建 Tensor 数组 %l，包含 Tensor %11 和 %21
  %l : Tensor[] = prim::ListConstruct(%11, %21)
  // 使用 aten::select() 从 %l 中选择索引为 %24 的元素，得到 Tensor %25
  %25 : Tensor = aten::select(%l, %24)
  // 使用 aten::select() 从 %25 中选择索引为 %24 的元素，得到 Tensor %27
  %27 : Tensor = aten::select(%25, %24, %24)
  // 创建整数数组 %28，包含常量 %0 和 %1
  %28 : int[] = prim::ListConstruct(%0, %1)
  // 调用 prim::MakeTestTensor() 创建 Tensor %37
  %37 : Tensor = prim::MakeTestTensor()
  // 调用 aten::add_() 将 %37 加到 %27 上，结果为 Tensor %39
  %39 : Tensor = aten::add_(%27, %37, %38)
  // 使用 uses::list() 获取 Tensor 数组 %l
  %40 : Tensor = uses::list(%l)
  // 返回 %40
  return (%40)
)IR",
      graph.get(),
      vmap);
  // 创建计算图的别名数据库 AliasDb
  AliasDb aliasDb(graph);
  // 获取 %40 对应的节点 listUse 和 %39 对应的节点 internalWrite
  auto listUse = vmap["40"]->node();
  auto internalWrite = vmap["39"]->node();
  // 断言在拓扑上移动 listUse 到 internalWrite 之前是不合法的
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(listUse, internalWrite));
}
TEST(WildcardsTest, Basic) {
  // 注册运算符，包括两个操作符定义和对应的处理函数
  RegisterOperators reg(
      {Operator(
           "prim::returns_wildcard(Tensor a) -> Tensor(*)",
           [](Stack&) {},
           aliasAnalysisFromSchema()),  // 第一个操作符注册，处理函数为空，基于模式分析的别名分析
       Operator(
           "prim::writes(Tensor(z!) a) -> Tensor(a)",
           [](Stack&) {},
           aliasAnalysisFromSchema())});  // 第二个操作符注册，处理函数为空，基于模式分析的别名分析

  const auto returns_wildcard =
      Symbol::fromQualString("prim::returns_wildcard");  // 创建符号 prim::returns_wildcard
  const auto writes = Symbol::fromQualString("prim::writes");  // 创建符号 prim::writes

  auto graph = std::make_shared<Graph>();  // 创建一个共享的图对象
  const auto a = graph->addInput();  // 添加一个图输入

  const auto constant = graph->insertConstant(1);  // 在图中插入常量节点 1
  const auto fresh = graph->insert(aten::rand, {constant});  // 在图中插入 rand 操作节点，依赖于常量节点
  const auto fresh2 = graph->insert(aten::rand, {constant});  // 再次在图中插入 rand 操作节点，依赖于常量节点
  const auto wildcard = graph->insert(returns_wildcard, {fresh});  // 在图中插入 returns_wildcard 操作节点，依赖于 fresh 结果

  {
    graph->lint();  // 执行图的 lint 检查
    AliasDb aliasDb(graph);  // 创建图的别名数据库对象

    // 断言：a 和 fresh 不可能是别名
    EXPECT_FALSE(aliasDb.mayAlias(a, fresh));
    // 断言：wildcard 和 fresh 不可能是别名
    EXPECT_FALSE(aliasDb.mayAlias(wildcard, fresh));
    // 断言：wildcard 和 a 可能是别名
    EXPECT_TRUE(aliasDb.mayAlias(wildcard, a));
    // 断言：wildcard 集合和空集合不可能有别名关系
    EXPECT_FALSE(aliasDb.mayAlias(ValueSet{wildcard}, ValueSet{}));
    // 断言：wildcard 节点没有写入操作
    EXPECT_FALSE(aliasDb.hasWriters(wildcard->node()));
  }

  graph->insert(writes, {fresh2})->node();  // 在图中插入 writes 操作节点，依赖于 fresh2 结果
  {
    graph->lint();  // 再次执行图的 lint 检查
    AliasDb aliasDb(graph);  // 创建图的别名数据库对象
    // 断言：wildcard 节点没有写入操作
    EXPECT_FALSE(aliasDb.hasWriters(wildcard->node()));
  }

  const auto wildcardWrite = graph->insert(writes, {wildcard})->node();  // 在图中插入 writes 操作节点，依赖于 wildcard 结果
  {
    graph->lint();  // 第三次执行图的 lint 检查
    AliasDb aliasDb(graph);  // 创建图的别名数据库对象
    // 测试写入到通配符的情况
    // 断言：wildcardWrite 操作没有写入 fresh 节点的别名
    EXPECT_FALSE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{fresh}));
    // 断言：wildcardWrite 操作没有写入 fresh2 节点的别名
    EXPECT_FALSE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{fresh2}));
    // 断言：wildcardWrite 操作可能写入 a 节点的别名
    EXPECT_TRUE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{a}));
    // 断言：wildcard 节点有写入操作
    EXPECT_TRUE(aliasDb.hasWriters(wildcard->node()));
  }
}

// 测试通配符正确分离的情况
TEST(WildcardsTest, TypeIsolation) {
  auto graph = std::make_shared<Graph>();  // 创建一个共享的图对象
  std::unordered_map<std::string, Value*> vmap;  // 创建一个字符串到值的无序映射
  parseIR(
      R"IR(
  graph(%ten_list : Tensor[], %int_list : int[], %opt_ten_list : Tensor[]?):
    %ten : Tensor = prim::Constant()
    %4 : Tensor[] = aten::append(%ten_list, %ten)
    %ten_ten_list : Tensor[][] = prim::Constant()
    %int_int_list : int[][] = prim::Constant()
    return ()
      )IR",
      graph.get());  // 解析输入的 IR 字符串并将结果存储到图中
}
  )IR",
      &*graph,
      vmap);
  // 创建一个名为 aliasDb 的 AliasDb 对象，基于给定的图形 graph
  AliasDb aliasDb(graph);
  // 从 vmap 中获取名为 "opt_ten_list" 的变量，并赋值给 opt_ten_list
  auto opt_ten_list = vmap["opt_ten_list"];
  // 从 vmap 中获取名为 "ten_list" 的变量，并赋值给 ten_list
  auto ten_list = vmap["ten_list"];
  // 从 vmap 中获取名为 "int_list" 的变量，并赋值给 int_list
  auto int_list = vmap["int_list"];
  // 断言 int_list 没有写入者（没有别名写入）
  EXPECT_FALSE(aliasDb.hasWriters(int_list));
  // 断言 opt_ten_list 有写入者（可能有别名写入）
  EXPECT_TRUE(aliasDb.hasWriters(opt_ten_list));
  // 断言 ten_list 有写入者（可能有别名写入）
  EXPECT_TRUE(aliasDb.hasWriters(ten_list));
  // 断言 int_list 和 opt_ten_list 可能不会有别名关系
  EXPECT_FALSE(aliasDb.mayContainAlias(int_list, opt_ten_list));
  // 断言 ten_list 和 opt_ten_list 可能会有别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, opt_ten_list));
  // 断言 ten_list 和 opt_ten_list 可能会有别名关系
  EXPECT_TRUE(aliasDb.mayAlias(ten_list, opt_ten_list));

  // 从 vmap 中获取名为 "ten_ten_list" 的变量，并赋值给 list_of_tensor_lists
  auto list_of_tensor_lists = vmap["ten_ten_list"];
  // 断言 ten_list 和 list_of_tensor_lists 可能会有别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, list_of_tensor_lists));
  // 断言 ten_list 和 vmap 中名为 "ten" 的变量可能会有别名关系
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, vmap["ten"]));

  // 断言 vmap 中名为 "int_int_list" 的变量和 list_of_tensor_lists 可能不会有别名关系
  EXPECT_TRUE(
      !aliasDb.mayContainAlias(vmap["int_int_list"], list_of_tensor_lists));
// test invariant container aliasing
// 测试不变容器的别名问题
// 不同类型的容器不能相互别名，但它们可以包含可能相互别名的元素

TEST(WildcardsTest, InvariantContainerAliasing) {
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    // 解析内部表示（IR），将其填充到图中，并更新值映射
    parseIR(
        R"IR(
  graph(%ten_list : Tensor[], %ten_opt_list : Tensor?[]):
    %ten : Tensor = prim::Constant()
    %4 : Tensor[] = aten::append(%ten_list, %ten)
    return ()
    )IR",
        &*graph,
        vmap);
    // 创建基于图的别名分析数据库
    AliasDb aliasDb(graph);
    // 获取映射中的张量可选列表和张量列表
    auto ten_opt_list = vmap["ten_opt_list"];
    auto ten_list = vmap["ten_list"];
    // 断言张量可选列表不存在写入者
    EXPECT_FALSE(aliasDb.hasWriters(ten_opt_list));
    // 断言张量列表存在写入者
    EXPECT_TRUE(aliasDb.hasWriters(ten_list));
    // 断言张量列表和张量可选列表可能存在别名关系
    EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, ten_opt_list));
    // 断言张量列表和张量可选列表不会别名
    EXPECT_FALSE(aliasDb.mayAlias(ten_list, ten_opt_list));
  }
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    // 解析内部表示（IR），将其填充到图中，并更新值映射
    parseIR(
        R"IR(
  graph(%float_3D : Float(*, *, *), %float_2D : Float(*, *)):
    return ()
    )IR",
        &*graph,
        vmap);
    // 创建基于图的别名分析数据库
    AliasDb aliasDb(graph);
    // 断言浮点数 3D 和浮点数 2D 可能存在别名关系
    EXPECT_TRUE(aliasDb.mayAlias(vmap["float_3D"], vmap["float_2D"]));
  }

  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    // 解析内部表示（IR），将其填充到图中，并更新值映射
    parseIR(
        R"IR(
  graph(%float_3D_list : Float(*, *, *)[], %float_2D_list : Float(*, *)[], %ten: Tensor):
    return ()
    )IR",
        &*graph,
        vmap);
    // 创建基于图的别名分析数据库
    AliasDb aliasDb(graph);
    // 断言浮点数 3D 列表和浮点数 2D 列表可能存在别名关系
    EXPECT_TRUE(aliasDb.mayAlias(vmap["float_3D_list"], vmap["float_2D_list"]));
    // 断言浮点数 3D 列表和张量之间可能存在别名关系
    EXPECT_TRUE(aliasDb.mayContainAlias(vmap["float_3D_list"], vmap["ten"]));
    // 断言浮点数 2D 列表和张量之间可能存在别名关系
    EXPECT_TRUE(aliasDb.mayContainAlias(vmap["float_2D_list"], vmap["ten"]));
  }
}

TEST(AliasRegistrationTest, ConservativeWithInferredSchema) {
  // 注册运算符并设置别名分析为保守型
  auto registry = torch::RegisterOperators().op(
      "foo::rand1",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));
  // 获取运算符符号
  const auto rand_op = Symbol::fromQualString("foo::rand1");
  // 创建共享图对象
  auto graph = std::make_shared<Graph>();
  // 向图中添加输入节点
  auto a = graph->addInput();
  // 插入随机运算符，并将输入作为参数
  auto b = graph->insert(rand_op, {a});
  // 创建基于图的别名分析数据库
  AliasDb aliasDb(graph);
  // 保守地假设输入和输出可能存在别名关系
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
}
TEST(AliasRegistrationTest, ConservativeWithSpecifiedSchema) {
  // 注册操作符 "foo::rand2"，定义一个默认的捕获所有内核函数，返回一个2x2的随机张量
  auto registry = torch::RegisterOperators().op(
      "foo::rand2(Tensor arg1) -> Tensor",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));

  // 创建符号对象，代表操作符 "foo::rand2"
  const auto rand_op = Symbol::fromQualString("foo::rand2");

  // 创建一个共享指针指向图对象
  auto graph = std::make_shared<Graph>();

  // 向图中添加一个输入节点
  auto a = graph->addInput();

  // 向图中插入操作符 "foo::rand2"，其输入为节点 a
  auto b = graph->insert(rand_op, {a});

  // 根据图对象创建别名数据库
  AliasDb aliasDb(graph);

  // 保守地假设节点 a 和节点 b 存在引用关系
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, ConservativeWithAliasingAnnotationsShouldError) {
  // 注册操作符 "foo::rand3"，带有别名注解，但没有指定别名分析为 CONSERVATIVE，预期会抛出异常
  auto registry = torch::RegisterOperators().op(
      "foo::rand3(Tensor(a) arg1) -> Tensor(b)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));

  // 创建符号对象，代表操作符 "foo::rand3"
  const auto rand_op = Symbol::fromQualString("foo::rand3");

  // 创建一个共享指针指向图对象
  auto graph = std::make_shared<Graph>();

  // 向图中添加一个输入节点
  auto a = graph->addInput();

  // 向图中插入操作符 "foo::rand3"，其输入为节点 a
  graph->insert(rand_op, {a});

  // 在注册时是正常的，但在从注册中获取时抛出异常
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand3(Tensor(a) arg1) -> Tensor(b) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(AliasRegistrationTest, ConservativeWithAliasingAnnotationsShouldError2) {
  // 注册操作符 "foo::rand4"，带有别名注解，但没有指定别名分析为 CONSERVATIVE，预期会抛出异常
  auto registry = torch::RegisterOperators().op(
      "foo::rand4(Tensor(a) arg1) -> Tensor(a)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));

  // 创建符号对象，代表操作符 "foo::rand4"
  const auto rand_op = Symbol::fromQualString("foo::rand4");

  // 创建一个共享指针指向图对象
  auto graph = std::make_shared<Graph>();

  // 向图中添加一个输入节点
  auto a = graph->addInput();

  // 向图中插入操作符 "foo::rand4"，其输入为节点 a
  graph->insert(rand_op, {a});

  // 在注册时是正常的，但在从注册中获取时抛出异常
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand4(Tensor(a) arg1) -> Tensor(a) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(AliasRegistrationTest, FromSchemaWithInferredSchemaShouldError) {
  // 尝试注册操作符 "foo::rand5"，其使用了 FROM_SCHEMA 的别名分析，但其模式是推断出来的，预期会抛出异常
  expectThrows<c10::Error>(
      [] {
        torch::RegisterOperators().op(
            "foo::rand5",
            torch::RegisterOperators::options()
                .catchAllKernel([](at::Tensor) -> at::Tensor {
                  return at::rand({2, 2});
                })
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
      },
      "Tried to register operator foo::rand5(Tensor _0) -> Tensor _0 with AliasAnalysisKind::FROM_SCHEMA, but the schema is inferred");
}
TEST(AliasRegistrationTest, FromSchemaInferredPure) {
  // 注册运算符 "foo::rand6"，定义一个 catch-all 内核函数返回一个 2x2 的随机张量
  auto registry = torch::RegisterOperators().op(
      "foo::rand6(Tensor arg1) -> Tensor",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
  
  // 从字符串创建操作符符号 "foo::rand6"
  const auto rand_op = Symbol::fromQualString("foo::rand6");
  
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  
  // 向图中添加输入节点
  auto a = graph->addInput();
  
  // 向图中插入操作符 "foo::rand6"，连接到输入节点 a
  auto b = graph->insert(rand_op, {a});
  
  // 创建一个用于分析别名的 AliasDb 对象
  AliasDb aliasDb(graph);
  
  // 根据测试案例，预期 a 和 b 没有别名关系
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, FromSchemaAliased) {
  // 注册运算符 "foo::rand7"，定义一个 catch-all 内核函数返回输入张量的两倍
  auto registry = torch::RegisterOperators().op(
      "foo::rand7(Tensor(a) arg1) -> Tensor(a)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
  
  // 从字符串创建操作符符号 "foo::rand7"
  const auto rand_op = Symbol::fromQualString("foo::rand7");

  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  
  // 向图中添加输入节点
  auto a = graph->addInput();
  
  // 向图中插入操作符 "foo::rand7"，连接到输入节点 a
  auto b = graph->insert(rand_op, {a});
  
  // 创建一个用于分析别名的 AliasDb 对象
  AliasDb aliasDb(graph);
  
  // 根据测试案例，预期 a 和 b 存在别名关系
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, FromSchemaPure) {
  // 注册运算符 "foo::rand8"，定义一个 catch-all 内核函数返回输入张量的两倍
  auto registry = torch::RegisterOperators().op(
      "foo::rand8(Tensor(a) arg1) -> Tensor(b)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
  
  // 从字符串创建操作符符号 "foo::rand8"
  const auto rand_op = Symbol::fromQualString("foo::rand8");

  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  
  // 向图中添加输入节点
  auto a = graph->addInput();
  
  // 向图中插入操作符 "foo::rand8"，连接到输入节点 a
  auto b = graph->insert(rand_op, {a});
  
  // 创建一个用于分析别名的 AliasDb 对象
  AliasDb aliasDb(graph);
  
  // 根据测试案例，预期 a 和 b 没有别名关系
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, PureNoSchema) {
  // 注册运算符 "foo::rand9"，定义一个 catch-all 内核函数返回一个 2x2 的随机张量
  auto registry = torch::RegisterOperators().op(
      "foo::rand9",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  
  // 从字符串创建操作符符号 "foo::rand9"
  const auto rand_op = Symbol::fromQualString("foo::rand9");

  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  
  // 向图中添加输入节点
  auto a = graph->addInput();
  
  // 向图中插入操作符 "foo::rand9"，连接到输入节点 a
  auto b = graph->insert(rand_op, {a});
  
  // 创建一个用于分析别名的 AliasDb 对象
  AliasDb aliasDb(graph);
  
  // 根据测试案例，预期 a 和 b 没有别名关系
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}
TEST(AliasRegistrationTest, PureWithSchema) {
  // 注册操作符 "foo::rand10"，并指定其为纯函数，不进行别名分析
  auto registry = torch::RegisterOperators().op(
      "foo::rand10(Tensor arg1) -> Tensor",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            // 在纯函数模式下，生成一个随机的 2x2 张量
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  // 从符号字符串创建操作符符号
  const auto rand_op = Symbol::fromQualString("foo::rand10");
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 向计算图中添加输入节点
  auto a = graph->addInput();
  // 在计算图中插入操作符节点
  auto b = graph->insert(rand_op, {a});
  // 创建别名数据库，基于当前的计算图
  AliasDb aliasDb(graph);
  // 断言：在纯函数模式下，输入节点 a 和操作符节点 b 不应该有别名
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, PureWithAnnotationsShouldError) {
  // 注册操作符 "foo::rand11"，并指定其为纯函数，但带有别名信息的模式
  auto registry = torch::RegisterOperators().op(
      "foo::rand11(Tensor(a) arg1) -> Tensor(a)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  // 从符号字符串创建操作符符号
  const auto rand_op = Symbol::fromQualString("foo::rand11");
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 向计算图中添加输入节点
  auto a = graph->addInput();
  // 在计算图中插入操作符节点
  graph->insert(rand_op, {a});

  // 注册时是可以的，但当从注册中获取时会抛出异常
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand11(Tensor(a) arg1) -> Tensor(a) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(AliasRegistrationTest, AliasMoveAtenListOp) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 创建值映射
  std::unordered_map<std::string, Value*> vmap;
  // 定义 IR 字符串表示的计算图内容
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %8 : int = prim::Constant[value=0]()
    %5 : int = prim::Constant[value=1]()
    %4 : int = prim::Constant[value=2]()
    %y : Tensor[] = prim::ListConstruct(%x)
    %6 : Tensor = aten::add_(%x, %4, %5)
    %9 : Tensor = aten::cat(%y, %8)
    return (%9))IR";

  // 解析 IR 字符串，将计算图构建起来，并更新值映射
  torch::jit::parseIR(graph_string, graph.get(), vmap);
  // 创建别名数据库，基于当前的计算图
  AliasDb aliasDb(graph);

  // 断言：由于 y.1 仅在一个非别名化的 aten 操作中使用，因此 x 不应该是 y.1 中包含元素的一部分
  EXPECT_TRUE(!aliasDb.mayAlias(vmap["x"], vmap["9"]));

  // 写入包含元素应该阻止移动操作
  EXPECT_TRUE(!aliasDb.moveBeforeTopologicallyValid(
      vmap["y"]->node(), vmap["9"]->node()));
}

TEST(
    AliasRegistrationTest,
    AliasMoveForTupleConstructWithSingleUseAsGraphOutput) {
  // 创建一个新的计算图
  auto graph = std::make_shared<Graph>();
  // 创建值映射
  std::unordered_map<std::string, Value*> vmap;
  // 定义 IR 字符串表示的计算图内容
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %y : Tensor = prim::MakeTestTensor()
    %z : (Tensor) = prim::TupleConstruct(%x, %y)
    return (%z))IR";
    return (%z))IR";



    // 返回一个包含 "%z))IR" 的字符串。这个语句看起来像是一个模板字符串或者是格式化字符串的一部分。



  torch::jit::parseIR(graph_string, graph.get(), vmap);



  // 使用 torch::jit::parseIR 函数解析 graph_string 中的 IR（Intermediate Representation），
  // 将解析结果存储在 graph 中，并使用 vmap 进行变量映射。



  AliasDb aliasDb(graph, /*isFrozen=*/false);



  // 创建一个 AliasDb 对象 aliasDb，用于分析 graph 中的别名信息。
  // 参数 isFrozen 设置为 false，表示 aliasDb 不是冻结状态，即可以进行修改。



  EXPECT_TRUE(!aliasDb.mayAlias(vmap["x"], vmap["y"]));



  // 使用 EXPECT_TRUE 断言，验证 vmap 中的变量 "x" 和 "y" 在 aliasDb 中不可能是别名关系。



  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["z"], vmap["x"]));



  // 使用 EXPECT_TRUE 断言，验证 vmap 中的变量 "z" 和 "x" 在 aliasDb 中可能是别名关系。



  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["z"], vmap["y"]));



  // 使用 EXPECT_TRUE 断言，验证 vmap 中的变量 "z" 和 "y" 在 aliasDb 中可能是别名关系。
TEST(AliasRegistrationTest, ATenSplitIntListAliasCheck) {
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向新的图形对象
  std::unordered_map<std::string, Value*> vmap;  // 创建一个无序映射，将字符串映射到值指针

  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()  // 创建张量 %x，并初始化为测试张量
    %0 : int = prim::Constant[value=0]()  // 创建整数常量 %0，值为 0
    %1 : int = prim::Constant[value=1]()  // 创建整数常量 %1，值为 1
    %2 : int = prim::Constant[value=2]()  // 创建整数常量 %2，值为 2
    %y : Tensor = aten::add(%x, %x, %0)  // 执行张量加法操作，结果保存在 %y 中
    %lengths_list : int[] = prim::tolist(%1, %2)  // 将常量 %1 和 %2 转换为整数列表 %lengths_list
    %a : Tensor[] = aten::split(%y, %lengths_list, %0)  // 对张量 %y 进行按长度列表 %lengths_list 分割，结果保存在张量数组 %a 中
    %b : Tensor, %c : Tensor = prim::ListUnpack(%a)  // 解包张量数组 %a 到 %b 和 %c 中
    %b1 : Tensor = aten::flatten(%b, %0, %1)  // 对张量 %b 执行展平操作，结果保存在 %b1 中
    %c1 : Tensor = aten::flatten(%c, %0, %1)  // 对张量 %c 执行展平操作，结果保存在 %c1 中
    %d : Tensor = aten::add(%b1, %c1, %0)  // 执行张量加法操作，结果保存在 %d 中
  )IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);  // 解析字符串表示的 IR，构建图形并填充值映射
  AliasDb aliasDb(graph, /*isFrozen=*/false);  // 创建别名数据库，与给定的图形关联，设置为非冻结状态

  // 以下是一系列断言，用于验证别名数据库的不同别名和包含关系
  EXPECT_TRUE(aliasDb.mayAlias(vmap["x"], vmap["y"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["x"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["x"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["b"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["a"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["a"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["b1"], vmap["c1"]));
}
    return (%d))IR";



// 定义字符串，格式化后返回包含数字的字符串
  torch::jit::parseIR(graph_string, graph.get(), vmap);



// 解析输入的图形字符串并填充到图对象中，使用变量映射 vmap
  AliasDb aliasDb(graph, /*isFrozen=*/false);



// 断言以下别名关系都为真
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b1"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c1"]));
TEST(AliasRegistrationTest, ATenSplitIntAliasCheck) {
  // 创建一个空的图形对象
  auto graph = std::make_shared<Graph>();
  // 创建一个空的值映射表
  std::unordered_map<std::string, Value*> vmap;
  // 定义包含图形 IR 的字符串
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : int = prim::Constant[value=0]()
    %1 : int = prim::Constant[value=1]()
    %2 : int = prim::Constant[value=2]()
    %y : Tensor = aten::add(%x, %x, %0)
    %a : Tensor[] = aten::split(%y, %2, %0)
    %b : Tensor, %c : Tensor = prim::ListUnpack(%a)
    %b1 : Tensor = aten::flatten(%b, %0, %1)
    %c1 : Tensor = aten::flatten(%c, %0, %1)
    %d : Tensor = aten::add(%b1, %c1, %0)
    return (%d))IR";
  
  // 解析图形字符串并填充图形和值映射表
  torch::jit::parseIR(graph_string, graph.get(), vmap);
  // 创建别名分析对象
  AliasDb aliasDb(graph, /*isFrozen=*/false);

  // 断言以下别名关系成立
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b1"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c1"]));
}

TEST(AliasRegistrationTest, PureWithAnnotationsShouldError2) {
  // 注册操作符并指定捕获所有内核和纯函数的别名分析
  auto registry = torch::RegisterOperators().op(
      "foo::rand12(Tensor(a) arg1) -> Tensor(b)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  // 创建符号并构建图形对象
  const auto rand_op = Symbol::fromQualString("foo::rand12");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  graph->insert(rand_op, {a});

  // 注册过程无异常，但从注册信息获取时抛出异常
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand12(Tensor(a) arg1) -> Tensor(b) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(IRNonDeterminismTest, Basic) {
  // 创建一个空的图形对象
  auto graph = std::make_shared<Graph>();
  // 定义包含图形 IR 的字符串
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : int = prim::Constant[value=0]()
    %1 : NoneType = prim::Constant()
    %2 : Tensor = aten::bernoulli(%x, %1)
    %3 : Tensor = aten::add(%x, %2, %0)
    return (%3))IR";
  // 解析图形字符串并填充图形对象
  parseIR(graph_string, graph.get());

  // 检查每个节点是否是非确定性操作
  for (Node* n : graph->nodes()) {
    if (n->kind() == aten::bernoulli) {
      ASSERT_TRUE(n->isNondeterministic());
    } else {
      ASSERT_FALSE(n->isNondeterministic());
    }
  }
}

TEST(IRNonDeterminismTest, DropoutSpecialCase) {
  // 创建一个空的图形对象
  auto graph = std::make_shared<Graph>();
  // 定义包含图形 IR 的字符串
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : bool = prim::Constant[value=0]()
    %1 : bool = prim::Constant[value=1]()
    %3 : int = prim::Constant[value=1]()
    %3 : float = prim::Constant[value=1.0]()
    %4 : Tensor = aten::dropout(%x, %3, %0)
    %5 : Tensor = aten::dropout(%x, %3, %1)
    %6 : Tensor = aten::add(%4, %5, %3)
    return (%6))IR";
  // 解析图形字符串并填充图形对象
  parseIR(graph_string, graph.get());

  bool train = false;
  // 检查每个节点是否是非确定性操作
  for (Node* n : graph->nodes()) {
    // 检查节点 n 是否为 dropout 操作
    if (n->kind() == aten::dropout) {
      // 如果不处于训练模式，则断言节点 n 不是非确定性的，并设置为训练模式
      if (!train) {
        ASSERT_FALSE(n->isNondeterministic());
        train = true;
      } else {
        // 如果已经处于训练模式，则断言节点 n 是非确定性的
        ASSERT_TRUE(n->isNondeterministic());
      }
    } else {
      // 如果节点 n 不是 dropout 操作，则断言节点 n 不是非确定性的
      ASSERT_FALSE(n->isNondeterministic());
    }
  }
TEST(NonDeterminismBackwardsCompatibility, BackwardsCompatibility) {
  // 定义一个静态的字符串向量，包含非确定性操作的函数签名列表
  static const std::vector<std::string> nondeterministic_ops = {
      "aten::dropout(Tensor input, float p, bool train) -> Tensor",
      "aten::_fused_dropout(Tensor self, float p, Generator? generator) -> (Tensor, Tensor)",
      "aten::_standard_gamma(Tensor self, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
      "aten::multinomial(Tensor self, int num_samples, bool replacement, *, Generator? generator) -> Tensor",
      "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)",
      "aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal.Tensor_float(Tensor mean, float std, *, Generator? generator) -> Tensor",
      "aten::poisson(Tensor self, Generator? generator) -> Tensor",
      "aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor",
      "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randperm(int n, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor"};

  // 遍历非确定性操作列表
  for (const std::string& op : nondeterministic_ops) {
    // 解析操作的函数模式
    const c10::FunctionSchema& schema = torch::jit::parseSchema(op);
    // 查找操作的处理器
    const auto& op_handle = c10::Dispatcher::singleton().findOp(
        c10::OperatorName(schema.name(), schema.overload_name()));
    // 断言操作处理器具有非确定性种子标签
    ASSERT_TRUE(op_handle->hasTag(at::Tag::nondeterministic_seeded));
  }
}
TEST(TypeHashing, HashTypes) {
  // 创建一个哈希类型对象
  HashType hasher;

  // 获取整数类型指针和浮点数类型指针
  const TypePtr int_type = IntType::get();
  const TypePtr float_type = FloatType::get();
  
  // 断言整数类型和浮点数类型的哈希值不相等
  ASSERT_NE(hasher(int_type), hasher(float_type));

  // 创建包含两个整数类型的元组类型指针和包含三个整数类型的元组类型指针
  const TypePtr int2_type = TupleType::create({int_type, int_type});
  const TypePtr int3_type = TupleType::create({int_type, int_type, int_type});
  
  // 断言两个不同长度的元组类型的哈希值不相等
  ASSERT_NE(hasher(int2_type), hasher(int3_type));
}

} // namespace jit
} // namespace torch
```