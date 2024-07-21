# `.\pytorch\test\cpp\lazy\test_trie_cache.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/Exception.h>  // 引入异常处理相关的头文件
#include <torch/csrc/lazy/core/config.h>  // 引入 Lazy Tensor 核心配置的头文件
#include <torch/csrc/lazy/core/ir.h>  // 引入 Lazy Tensor 中间表示的头文件
#include <torch/csrc/lazy/core/ir_builder.h>  // 引入 Lazy Tensor 中间表示构建器的头文件
#include <torch/csrc/lazy/core/ir_metadata.h>  // 引入 Lazy Tensor 中间表示元数据的头文件
#include <torch/csrc/lazy/core/ir_util.h>  // 引入 Lazy Tensor 中间表示实用工具的头文件
#include <memory>  // 引入内存管理相关的头文件

namespace torch {
namespace lazy {

class TrieCacheNode : public Node {  // 定义 TrieCacheNode 类，继承自 Node 类
 public:
  static OpKind ClassOpKind() {  // 静态方法，返回 OpKind()
    return OpKind();
  }

  explicit TrieCacheNode(size_t id)  // 显式构造函数，根据给定的 id 创建 TrieCacheNode 对象
      : Node(ClassOpKind(), /* num_outputs */ 1), id_(id), hash_(Hash(id_)) {}  // 初始化 Node 和成员变量

  ~TrieCacheNode() override = default;  // 默认析构函数

  bool CanBeReused(size_t id) const {  // 检查给定的 id 是否可以重用
    return (id_ == id);
  }

  void AddOperand(Value v) {  // 添加操作数
    if (!v.node) {  // 如果操作数节点为空，直接返回
      return;
    }
    operands_as_outputs_.emplace_back(v.node.get(), v.index);  // 将操作数作为输出添加到列表中
    operands_.push_back(std::move(v.node));  // 将操作数节点移动到操作数列表中
  }

  hash_t hash() const override {  // 重写基类的 hash() 方法
    return hash_;  // 返回对象的哈希值
  }

  hash_t shapeHash() const override {  // 重写基类的 shapeHash() 方法
    return hash_;  // 返回对象的形状哈希值
  }

 private:
  size_t id_;  // 对象的 id
  hash_t hash_;  // 对象的哈希值
};

TEST(TrieCacheTest, TestSinglePath) {  // 单条路径的测试用例
  FLAGS_torch_lazy_reuse_ir = true;  // 设置懒惰计算中是否重用 IR 的标志为 true
  TrieCache::Get()->Clear();  // 清除 TrieCache 中的内容

  NodePtr a = ReuseOrMakeNode<TrieCacheNode>(0);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr a
  NodePtr b = ReuseOrMakeNode<TrieCacheNode>(1);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr b
  NodePtr c = ReuseOrMakeNode<TrieCacheNode>(2);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr c
  TrieCache::Get()->ResetCurrent(); // 重置当前 TrieCache 的状态，标记为一个步骤

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());  // 断言重用或创建 TrieCacheNode(0) 返回的指针与 a 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());  // 断言重用或创建 TrieCacheNode(1) 返回的指针与 b 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(2).get(), c.get());  // 断言重用或创建 TrieCacheNode(2) 返回的指针与 c 相同
  TrieCache::Get()->ResetCurrent(); // 重置当前 TrieCache 的状态，标记为一个步骤
}

/*
 *    0
 *    |
 *    1
 *   / \
 *  2   3
 */
TEST(TrieCacheTest, TestTwoPaths) {  // 双路径的测试用例
  FLAGS_torch_lazy_reuse_ir = true;  // 设置懒惰计算中是否重用 IR 的标志为 true
  TrieCache::Get()->Clear();  // 清除 TrieCache 中的内容

  NodePtr a = ReuseOrMakeNode<TrieCacheNode>(0);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr a
  NodePtr b = ReuseOrMakeNode<TrieCacheNode>(1);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr b
  NodePtr c = ReuseOrMakeNode<TrieCacheNode>(2);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr c
  TrieCache::Get()->ResetCurrent(); // 重置当前 TrieCache 的状态，标记为一个步骤

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());  // 断言重用或创建 TrieCacheNode(0) 返回的指针与 a 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());  // 断言重用或创建 TrieCacheNode(1) 返回的指针与 b 相同

  NodePtr d = ReuseOrMakeNode<TrieCacheNode>(3);  // 重用或创建 TrieCacheNode 对象，并赋给 NodePtr d
  EXPECT_NE(d.get(), c.get());  // 断言 NodePtr d 与 NodePtr c 不相同
  TrieCache::Get()->ResetCurrent(); // 重置当前 TrieCache 的状态，标记为一个步骤

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());  // 断言重用或创建 TrieCacheNode(0) 返回的指针与 a 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());  // 断言重用或创建 TrieCacheNode(1) 返回的指针与 b 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(3).get(), d.get());  // 断言重用或创建 TrieCacheNode(3) 返回的指针与 d 相同
  TrieCache::Get()->ResetCurrent(); // 重置当前 TrieCache 的状态，标记为一个步骤

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());  // 断言重用或创建 TrieCacheNode(0) 返回的指针与 a 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());  // 断言重用或创建 TrieCacheNode(1) 返回的指针与 b 相同
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(2).get(), c.get());  // 断言重用或创建 TrieCacheNode(2) 返回的指针与 c 相同
  TrieCache::Get()->ResetCurrent(); // 重置当前 TrieCache 的状态，标记为一个步骤
}

} // namespace lazy
} // namespace torch
```