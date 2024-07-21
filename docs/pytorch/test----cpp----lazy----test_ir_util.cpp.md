# `.\pytorch\test\cpp\lazy\test_ir_util.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/Exception.h>  // 引入异常处理的头文件
#include <torch/csrc/lazy/core/config.h>  // 引入 Torch 懒执行模块的配置头文件
#include <torch/csrc/lazy/core/ir.h>  // 引入 Torch 懒执行模块的 IR 头文件
#include <torch/csrc/lazy/core/ir_builder.h>  // 引入 Torch 懒执行模块的 IR 构建头文件
#include <torch/csrc/lazy/core/ir_metadata.h>  // 引入 Torch 懒执行模块的 IR 元数据头文件
#include <torch/csrc/lazy/core/ir_util.h>  // 引入 Torch 懒执行模块的 IR 工具函数头文件

namespace torch {
namespace lazy {

class IrUtilNode : public Node {
 public:
  explicit IrUtilNode() : Node(OpKind(), /* num_outputs */ 1), hash_(Hash(0)) {}  // 定义 IrUtilNode 类的构造函数，设置操作类型和输出数量为 1
  ~IrUtilNode() override = default;  // 虚析构函数

  void AddOperand(Value v) {  // 添加操作数的方法
    if (!v.node) {  // 如果操作数的节点为空，直接返回
      return;
    }
    operands_as_outputs_.emplace_back(v.node.get(), v.index);  // 将操作数作为输出加入到列表中
    operands_.push_back(std::move(v.node));  // 将操作数节点移动到操作数列表中
  }

  hash_t hash() const override {  // 返回节点的哈希值
    return hash_;
  }
  hash_t shapeHash() const override {  // 返回形状的哈希值
    return hash_;
  }

 private:
  hash_t hash_;  // 哈希值成员变量
};

/*  a
 * / \
 *b   c
 * \ /
 *  d
 * Post-order: d c b a
 */
TEST(IrUtilTest, BasicTest) {  // 定义 IrUtilTest 的基本测试
  NodePtr a = MakeNode<IrUtilNode>();  // 创建节点 a
  NodePtr b = MakeNode<IrUtilNode>();  // 创建节点 b
  NodePtr c = MakeNode<IrUtilNode>();  // 创建节点 c
  NodePtr d = MakeNode<IrUtilNode>();  // 创建节点 d

  dynamic_cast<IrUtilNode*>(a.get())->AddOperand(Value(b, 0));  // a 添加操作数 b
  dynamic_cast<IrUtilNode*>(a.get())->AddOperand(Value(c, 1));  // a 添加操作数 c
  dynamic_cast<IrUtilNode*>(b.get())->AddOperand(Value(d, 0));  // b 添加操作数 d
  dynamic_cast<IrUtilNode*>(c.get())->AddOperand(Value(d, 0));  // c 添加操作数 d

  auto postorder = Util::ComputePostOrder({a.get()});  // 计算节点 a 的后序遍历顺序
  EXPECT_EQ(postorder.size(), 4);  // 验证后序遍历结果的大小为 4
  EXPECT_EQ(postorder.at(0), d.get());  // 验证后序遍历结果的第一个节点是 d
  EXPECT_EQ(postorder.at(1), c.get());  // 验证后序遍历结果的第二个节点是 c
  EXPECT_EQ(postorder.at(2), b.get());  // 验证后序遍历结果的第三个节点是 b
  EXPECT_EQ(postorder.at(3), a.get());  // 验证后序遍历结果的第四个节点是 a
}

/*  a
 * / \
 *b---c
 * Post-order: not valid
 */
TEST(IrUtilTest, TestCircle) {  // 定义 IrUtilTest 的环测试
  NodePtr a = MakeNode<IrUtilNode>();  // 创建节点 a
  NodePtr b = MakeNode<IrUtilNode>();  // 创建节点 b
  NodePtr c = MakeNode<IrUtilNode>();  // 创建节点 c

  dynamic_cast<IrUtilNode*>(a.get())->AddOperand(Value(b, 0));  // a 添加操作数 b
  dynamic_cast<IrUtilNode*>(b.get())->AddOperand(Value(c, 0));  // b 添加操作数 c
  dynamic_cast<IrUtilNode*>(c.get())->AddOperand(Value(a, 0));  // c 添加操作数 a，形成环

  EXPECT_THROW(Util::ComputePostOrder({a.get()}), c10::Error);  // 预期计算后序遍历时抛出异常 c10::Error
}

} // namespace lazy
} // namespace torch
```