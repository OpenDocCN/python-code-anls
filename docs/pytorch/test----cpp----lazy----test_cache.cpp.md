# `.\pytorch\test\cpp\lazy\test_cache.cpp`

```
#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// CacheNode类，继承自Node类，表示缓存中的节点
class CacheNode : public Node {
 public:
  // 构造函数，接受一个字符串作为参数
  explicit CacheNode(const std::string& str)
      : Node(OpKind(), /* num_outputs */ 1), hash_(Hash(str)), str_(str) {}
  // 虚析构函数
  ~CacheNode() override = default;

  // 返回空的操作数列表，不允许访问操作数
  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of test node");
  }

  // 返回空的操作数，不允许访问特定索引的操作数
  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of test node");
  }

  // 返回节点的哈希值
  hash_t hash() const override {
    return hash_;
  }
  
  // 返回形状的哈希值，这里与普通哈希值相同
  hash_t shapeHash() const override {
    return hash_;
  }

 private:
  hash_t hash_;         // 节点的哈希值
  std::string str_;     // 节点的字符串表示
};

// CacheTest类，包含基本测试
TEST(CacheTest, BasicTest) {
  // 创建三个CacheNode节点实例
  std::shared_ptr<CacheNode> a = std::make_shared<CacheNode>("a");
  std::shared_ptr<CacheNode> b = std::make_shared<CacheNode>("b");
  std::shared_ptr<CacheNode> c = std::make_shared<CacheNode>("c");
  // 创建一个最大容量为2的缓存
  Cache<hash_t, CacheNode, HashReducer> cache(2);

  // 添加节点a到缓存中
  cache.Add(a->hash(), a);
  // 断言从缓存中获取a节点的指针与a相同
  EXPECT_EQ(cache.Get(a->hash()), a);
  // 断言从缓存中获取b节点的指针为空
  EXPECT_EQ(cache.Get(b->hash()), nullptr);
  // 断言从缓存中获取c节点的指针为空
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  // 添加节点b到缓存中
  cache.Add(b->hash(), b);
  // 断言从缓存中获取a节点的指针与a相同
  EXPECT_EQ(cache.Get(a->hash()), a);
  // 断言从缓存中获取b节点的指针与b相同
  EXPECT_EQ(cache.Get(b->hash()), b);
  // 断言从缓存中获取c节点的指针为空
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  // 添加节点c到缓存中
  cache.Add(c->hash(), c);
  // 断言从缓存中获取a节点的指针为空（a已被驱逐）
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  // 断言从缓存中获取b节点的指针与b相同
  EXPECT_EQ(cache.Get(b->hash()), b);
  // 断言从缓存中获取c节点的指针与c相同
  EXPECT_EQ(cache.Get(c->hash()), c);

  // 从缓存中移除节点c
  cache.Erase(c->hash());
  // 断言从缓存中获取a节点的指针为空
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  // 断言从缓存中获取b节点的指针与b相同
  EXPECT_EQ(cache.Get(b->hash()), b);
  // 断言从缓存中获取c节点的指针为空（c已被移除）
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  // 清空缓存
  cache.Clear();
  // 断言从缓存中获取a节点的指针为空
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  // 断言从缓存中获取b节点的指针为空
  EXPECT_EQ(cache.Get(b->hash()), nullptr);
  // 断言从缓存中获取c节点的指针为空
  EXPECT_EQ(cache.Get(c->hash()), nullptr);
}

// CacheNodeWithShape类，继承自TsNode类，具有形状的缓存节点
class CacheNodeWithShape : public TsNode {
 public:
  // 构造函数，接受形状作为参数
  explicit CacheNodeWithShape(const Shape& shape)
      : TsNode(OpKind(), shape, /* num_outputs */ 1, /* seed */ 0) {}
};

// ShapeCacheTestForDynamicShape测试用例，测试动态形状的形状缓存
TEST(CacheTest, ShapeCacheTestForDynamicShape) {
  // 启用动态形状
  FLAGS_ltc_enable_dynamic_shapes = true;

  // 创建两个CacheNodeWithShape节点实例数组
  CacheNodeWithShape nodes[] = {
      CacheNodeWithShape(Shape(c10::kFloat, {2, 4})),
      CacheNodeWithShape(Shape(c10::kFloat, {4, 2}))};

  /*
   * 确保节点(2, 4)的缓存形状不用于节点(4, 2)
   */
  for (auto& node : nodes) {
    // 断言节点的形状与计算形状的结果相同
    EXPECT_EQ(node.shape(), node.computeShape([&]() { return node.shape(); }));
  }

  // 重置标志位
  FLAGS_ltc_enable_dynamic_shapes = false;
}

} // namespace lazy
} // namespace torch
```