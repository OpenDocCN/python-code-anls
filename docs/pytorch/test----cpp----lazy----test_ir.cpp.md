# `.\pytorch\test\cpp\lazy\test_ir.cpp`

```py
#include <gtest/gtest.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/dynamic_ir.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/generated/LazyIr.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <memory>

namespace torch {
namespace lazy {

// 定义一个测试用的叶节点类，继承自 Node
class TestLeafNode : public Node {
 public:
  // 返回这个类的操作种类，此处未实现具体内容
  static OpKind ClassOpKind() {
    return OpKind();
  }

  // 构造函数，接受一个参数 param，生成一个叶节点
  explicit TestLeafNode(size_t param)
      : Node(ClassOpKind(), /* num_outputs */ 1), hash_(Hash(param)) {}

  // 虚析构函数，用于释放资源
  ~TestLeafNode() override = default;

  // 覆盖基类方法，返回节点的操作数列表，但此类叶节点不支持操作数
  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of leaf node");
  }

  // 覆盖基类方法，返回指定索引的操作数，但此类叶节点不支持操作数
  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of leaf node");
  }

  // 返回节点的哈希值
  hash_t hash() const override {
    return hash_;
  }

  // 返回节点形状的哈希值，与 hash() 方法相同
  hash_t shapeHash() const override {
    return hash_;
  }

 private:
  hash_t hash_;  // 节点的哈希值
};

// 定义测试套件 IrTest，包含基本测试用例
TEST(IrTest, BasicTest) {
  // 创建一个 TestLeafNode 类型的节点，参数为 1
  NodePtr node1 = MakeNode<TestLeafNode>(1);
  // 创建另一个 TestLeafNode 类型的节点，参数为 2
  NodePtr node2 = MakeNode<TestLeafNode>(2);
  // 断言两个节点的哈希值不相等
  EXPECT_NE(node1->hash(), node2->hash());

  // 断言节点 node1 的输出数为 1
  EXPECT_EQ(node1->num_outputs(), 1);

  // 将 node1 转换为 TestLeafNode* 类型指针，用于后续的断言
  const TestLeafNode* leafptr = NodeCast<TestLeafNode>(node1.get());
  // 断言 leafptr 非空，即 node1 确实是 TestLeafNode 类型
  EXPECT_TRUE(leafptr != nullptr);
}

// 定义测试套件 IrTest，包含元数据测试用例
TEST(IrTest, MetaDataTest) {
  // 备份全局变量 FLAGS_torch_lazy_ir_debug 的值
  bool restore_FLAGS_torch_lazy_ir_debug = FLAGS_torch_lazy_ir_debug;
  // 设置 FLAGS_torch_lazy_ir_debug 为 false
  FLAGS_torch_lazy_ir_debug = false;
  // 创建一个 TestLeafNode 类型的节点，参数为 1
  NodePtr node = MakeNode<TestLeafNode>(1);
  // 获取节点的元数据
  auto metaWithoutDebug = node->metadata();
  // 断言元数据的作用域大小为 0
  EXPECT_EQ(metaWithoutDebug.scope.size(), 0);
  // 断言元数据的帧信息大小为 0
  EXPECT_EQ(metaWithoutDebug.frame_info.size(), 0);

  // 恢复 FLAGS_torch_lazy_ir_debug 的值
  FLAGS_torch_lazy_ir_debug = true;
  // 再次创建一个 TestLeafNode 类型的节点，参数为 1
  node = MakeNode<TestLeafNode>(1);
  // 获取节点的元数据
  auto metaWithEmptyDebug = node->metadata();
  // 断言元数据的作用域大小为 0
  EXPECT_EQ(metaWithEmptyDebug.scope.size(), 0);
  // 断言元数据的帧信息大小为 1
  EXPECT_EQ(metaWithEmptyDebug.frame_info.size(), 1);

  // 在作用域 "TestScope" 下创建一个节点
  {
    ScopePusher scope("TestScope");
    // 创建一个 TestLeafNode 类型的节点，参数为 1
    node = MakeNode<TestLeafNode>(1);
    // 获取节点的元数据
    auto metaWithScope = node->metadata();
    // 断言元数据的作用域为 "TestScope.1"
    EXPECT_EQ(metaWithScope.scope, "TestScope.1");
    // 断言元数据的帧信息大小为 1
    EXPECT_EQ(metaWithScope.frame_info.size(), 1);
  }

  // 创建一个虚拟的源码位置信息
  SourceLocation dummySourceLocation;
  dummySourceLocation.file = "file";
  dummySourceLocation.function = "function";
  dummySourceLocation.line = 10;
  // 定义获取 Python 堆栈帧信息的 Lambda 函数
  GetPythonFramesFunction() = [&]() -> std::vector<SourceLocation> {
    return {dummySourceLocation};
  };
  // 创建一个 TestLeafNode 类型的节点，参数为 1
  node = MakeNode<TestLeafNode>(1);
  // 获取节点的元数据
  auto metaWithSourceLoc = node->metadata();
  // 断言元数据的作用域大小为 0
  EXPECT_EQ(metaWithSourceLoc.scope.size(), 0);
  // 断言元数据的帧信息大小为 1
  EXPECT_EQ(metaWithSourceLoc.frame_info.size(), 1);
  // 断言元数据的第一个帧信息的文件为 "file"
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].file, "file");
  // 断言元数据的第一个帧信息的函数为 "function"
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].function, "function");
  // 断言元数据的第一个帧信息的行号为 10
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].line, 10);

  // 恢复 FLAGS_torch_lazy_ir_debug 的值
  FLAGS_torch_lazy_ir_debug = restore_FLAGS_torch_lazy_ir_debug;
}

} // namespace lazy
} // namespace torch
TEST(IrTest, TsNodeTest) {
  // 创建第一个 TsNode 节点，使用指定操作类型、空形状、一个输出，和给定的哈希种子
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);
  // 创建第二个 TsNode 节点，与第一个节点相同的参数
  NodePtr node2 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);
  // 验证两个节点的哈希值相等
  EXPECT_EQ(node1->hash(), node2->hash());

  // 验证第一个节点的输出数量为1
  EXPECT_EQ(node1->num_outputs(), 1);

  // 尝试将 node1 转换为 TsNode 类型，并验证转换成功
  const TsNode* leafptr = dynamic_cast<const TsNode*>(node1.get());
  EXPECT_TRUE(leafptr != nullptr);
}

TEST(IrTest, DimensionNodeTest) {
  // 定义两个维度常量
  const size_t DIM0 = 5;
  const size_t DIM1 = 8;
  // 创建 TsNode 节点，使用指定操作类型、具有两个维度的形状、一个输出，和给定的哈希种子
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(c10::kFloat, {DIM0, DIM1}),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);

  // 创建 size0 和 size1 节点，分别表示 node1 的两个维度的大小
  auto size0 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 0));
  auto size1 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 1));

  // 断言 size0 的静态值等于 DIM0
  ASSERT_EQ(DIM0, size0->getStaticValue());
  // 断言 size1 的静态值等于 DIM1
  ASSERT_EQ(DIM1, size1->getStaticValue());

  // 将 size0 转换为 SizeNode 类型，然后再转换为 DimensionNode 类型，并验证静态值等于 DIM0
  NodePtr size0_np = size0;
  auto size0_dn = std::dynamic_pointer_cast<DimensionNode>(size0_np);
  ASSERT_EQ(DIM0, size0_dn->getStaticValue());

  // 创建一个 SizeAdd 节点，表示 size0 和 size1 的加法操作，并验证其静态值等于 DIM0 + DIM1
  auto add_dim = std::dynamic_pointer_cast<SizeAdd>(
      MakeNode<SizeAdd>(Value{size0}, Value{size1}));
  ASSERT_EQ(DIM0 + DIM1, add_dim->getStaticValue());

  // 创建一个 SizeMul 节点，表示 size0 和 size1 的乘法操作，并验证其静态值等于 DIM0 * DIM1
  auto mul_dim = std::dynamic_pointer_cast<SizeMul>(
      MakeNode<SizeMul>(Value{size0}, Value{size1}));
  ASSERT_EQ(DIM0 * DIM1, mul_dim->getStaticValue());
}

TEST(IrTest, DimensionIsDynamicTest) {
  // 定义两个维度常量
  const size_t DIM0 = 5;
  const size_t DIM1 = 8;
  // 创建具有符号维度的形状对象
  const auto shape = Shape(c10::kFloat, {DIM0, DIM1});
  // 创建 TsNode 节点，使用指定操作类型、具有符号维度的形状、一个输出，和给定的哈希种子
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      shape.with_symbolic_dims(std::vector<bool>{true, false}),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);

  // 创建 size0 和 size1 节点，分别表示 node1 的两个维度的大小
  auto size0 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 0));
  auto size1 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 1));

  // 断言 size0 的符号属性为 true
  ASSERT_EQ(true, size0->isSymbolic());
  // 断言 size1 的符号属性为 false
  ASSERT_EQ(false, size1->isSymbolic());

  // 创建一个 SizeAdd 节点，表示 size0 和 size1 的加法操作，并验证其符号属性为 true
  auto add_dim = std::dynamic_pointer_cast<SizeAdd>(
      MakeNode<SizeAdd>(Value{size0}, Value{size1}));
  ASSERT_EQ(true, add_dim->isSymbolic());

  // 再次创建一个 SizeAdd 节点，表示 size1 和 size1 的加法操作，并验证其符号属性为 false
  add_dim = std::dynamic_pointer_cast<SizeAdd>(
      MakeNode<SizeAdd>(Value{size1}, Value{size1}));
  ASSERT_EQ(false, add_dim->isSymbolic());

  // 创建一个 SizeMul 节点，表示 size0 和 size0 的乘法操作，并验证其符号属性为 true
  auto mul_dim = std::dynamic_pointer_cast<SizeMul>(
      MakeNode<SizeMul>(Value{size0}, Value{size0}));
  ASSERT_EQ(true, mul_dim->isSymbolic());
}

} // namespace lazy
} // namespace torch
```