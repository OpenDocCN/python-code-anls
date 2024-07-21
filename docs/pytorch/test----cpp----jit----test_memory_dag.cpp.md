# `.\pytorch\test\cpp\jit\test_memory_dag.cpp`

```
// 包含 Google Test 框架头文件
#include <gtest/gtest.h>

// 包含 PyTorch JIT 的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 PyTorch JIT 中的内存 DAG 实用工具头文件
#include <torch/csrc/jit/passes/utils/memory_dag.h>

// torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {

// 定义 MemoryDAGTest 测试类，测试基本功能
TEST(MemoryDAGTest, Basic) {
  // 创建一个共享指针指向 Graph 对象
  auto graph = std::make_shared<Graph>();
  
  // 添加图输入值，各赋予不同的标识符
  const Value* aValue = graph->addInput();
  const Value* bValue = graph->addInput();
  const Value* cValue = graph->addInput();
  const Value* dValue = graph->addInput();
  const Value* eValue = graph->addInput();
  const Value* fValue = graph->addInput();
  const Value* gValue = graph->addInput();

  {
    // 创建一个 MemoryDAGBuilder 对象
    // 建立内存 DAG
    auto t = std::make_unique<MemoryDAGBuilder>();
    
    // 创建并添加新值到内存 DAG 中
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    auto c = t->makeFreshValue(cValue);
    auto d = t->makeFreshValue(dValue);
    auto e = t->makeFreshValue(eValue);
    auto f = t->makeFreshValue(fValue);
    auto g = t->makeFreshValue(gValue);
    
    // 创建指针关系
    t->makePointerTo(b, a);
    t->makePointerTo(c, b);
    t->makePointerTo(d, b);
    t->makePointerTo(e, a);
    t->makePointerTo(e, f);

    // 移动 MemoryDAGBuilder 对象以创建内存 DAG
    auto dag = std::move(*t).createMemoryDAG();

    /**
     * Test mayAlias()
     */
    // 检查值是否可能别名自身
    EXPECT_TRUE(dag->mayAlias(a, a));
    EXPECT_TRUE(dag->mayAlias(g, g));

    // 检查指向相同位置的值是否可能别名
    EXPECT_TRUE(dag->mayAlias(a, b));
    EXPECT_TRUE(dag->mayAlias(a, c));
    EXPECT_TRUE(dag->mayAlias(c, d));

    // 检查 e 可能指向 a 或 f
    EXPECT_TRUE(dag->mayAlias(e, a));
    EXPECT_TRUE(dag->mayAlias(e, f));
    // 但是 a 和 f 不可能别名
    EXPECT_FALSE(dag->mayAlias(a, f));
  }
  {
    // x(y) -> x 包含 y

    // b(a)
    // c(a)
    auto t = std::make_unique<MemoryDAGBuilder>();
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    t->addToContainedElements(a, b);

    auto c = t->makeFreshValue(cValue);
    t->addToContainedElements(a, c);

    // 移动 MemoryDAGBuilder 对象以创建内存 DAG
    auto dag = std::move(*t).createMemoryDAG();
    
    // 检查元素可能包含别名关系
    EXPECT_TRUE(dag->mayContainAlias(a, b));
    EXPECT_TRUE(dag->mayContainAlias(b, a));

    EXPECT_TRUE(dag->mayContainAlias(a, c));
    EXPECT_TRUE(dag->mayContainAlias(c, a));

    EXPECT_TRUE(dag->mayContainAlias(b, c));
    EXPECT_TRUE(dag->mayContainAlias(c, b));

    // 容器在自身中包含一个元素
    EXPECT_TRUE(dag->mayContainAlias(b, b));
    EXPECT_TRUE(dag->mayContainAlias(c, c));
    EXPECT_TRUE(dag->mayContainAlias(a, a));
  }
  {
    // b(a)
    // c(a)
    // d(b(a))
    auto t = std::make_unique<MemoryDAGBuilder>();
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    t->addToContainedElements(a, b);

    auto c = t->makeFreshValue(cValue);
    t->addToContainedElements(a, c);

    auto d = t->makeFreshValue(dValue);
    t->addToContainedElements(b, d);

    // 移动 MemoryDAGBuilder 对象以创建内存 DAG
    auto dag = std::move(*t).createMemoryDAG();
    
    // 检查元素可能包含别名关系
    EXPECT_TRUE(dag->mayContainAlias(b, d));
    EXPECT_TRUE(dag->mayContainAlias(d, b));


继续完成余下部分的注释。
    // 检查在有向无环图（DAG）中，节点 c 和 d 是否可能存在别名关系
    EXPECT_TRUE(dag->mayContainAlias(c, d));
    // 检查在有向无环图（DAG）中，节点 d 和 c 是否可能存在别名关系
    EXPECT_TRUE(dag->mayContainAlias(d, c));

    // 检查在有向无环图（DAG）中，节点 a 和 d 是否可能存在别名关系
    EXPECT_TRUE(dag->mayContainAlias(a, d));
  }
  {
    // f(e) 子块开始

    // 创建一个新的 MemoryDAGBuilder 对象
    auto t = std::make_unique<MemoryDAGBuilder>();
    // 创建值 a，并添加到 DAG 中
    auto a = t->makeFreshValue(aValue);
    // 创建值 b，并添加到 DAG 中
    auto b = t->makeFreshValue(bValue);
    // 将 b 添加为 a 的子元素
    t->addToContainedElements(a, b);

    // 创建值 c，并添加到 DAG 中
    auto c = t->makeFreshValue(cValue);
    // 将 c 添加为 a 的子元素
    t->addToContainedElements(a, c);

    // 创建值 d，并添加到 DAG 中
    auto d = t->makeFreshValue(dValue);
    // 将 d 添加为 b 的子元素
    t->addToContainedElements(b, d);

    // 创建值 f，并添加到 DAG 中
    auto f = t->makeFreshValue(aValue);
    // 创建值 e，并添加到 DAG 中
    auto e = t->makeFreshValue(bValue);
    // 将 e 添加为 f 的子元素
    t->addToContainedElements(f, e);

    // 将 t 转换为 MemoryDAG 对象并移动构造一个新的 DAG 对象
    auto dag = std::move(*t).createMemoryDAG();

    // 对于集合中的每个元素（a, b, c, d），检查 f 和 e 是否不可能与其存在别名关系
    for (auto elem : {a, b, c, d}) {
      EXPECT_FALSE(dag->mayContainAlias(f, elem));
      EXPECT_FALSE(dag->mayContainAlias(e, elem));
    }
  }
}

} // namespace jit
} // namespace torch
```