# `.\pytorch\test\cpp\jit\test_script_profile.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google 测试框架的头文件

#include <c10/util/Optional.h>  // 包含 C10 库中的 Optional 实用工具头文件
#include <test/cpp/jit/test_utils.h>  // 包含用于 JIT 测试的实用工具头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含 JIT IR 的头文件
#include <torch/csrc/jit/ir/irparser.h>  // 包含用于解析 JIT IR 的头文件
#include <torch/csrc/jit/runtime/script_profile.h>  // 包含脚本分析的头文件

namespace torch {
namespace jit {

TEST(ScriptProfileTest, Basic) {  // 定义基本的脚本分析测试用例
  const std::string source_string = R"V0G0N(
    def foo(a, b):
      return a + b #
  )V0G0N";
  auto begin = source_string.find("return");  // 查找源代码中 "return" 的位置
  auto end = source_string.find(" #");  // 查找源代码中 " #" 的位置，用于确定代码行范围

  Graph g;  // 创建 JIT 图对象
  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::add(%a, %b, %2)
      return (%3))IR";  // 定义 JIT IR 字符串表示图结构

  torch::jit::parseIR(graph_string, &g);  // 解析 JIT IR 字符串并填充到图对象 g 中
  auto source = std::make_shared<Source>(source_string, "", 0);  // 创建源码的共享指针对象
  auto node = *g.nodes().begin();  // 获取图中的第一个节点
  node->setSourceRange(SourceRange{source, begin, end});  // 设置节点的源码范围

  ScriptProfile p;  // 创建脚本分析器对象
  p.enable();  // 启用脚本分析器
  {
    profiling::InstructionSpan g0(*node);  // 创建指令跟踪对象 g0
    profiling::InstructionSpan g1(*node);  // 创建指令跟踪对象 g1
    profiling::InstructionSpan g2(*node);  // 创建指令跟踪对象 g2
  }
  p.disable();  // 禁用脚本分析器

  auto stats = p.dumpStats();  // 获取脚本分析统计信息
  EXPECT_EQ(stats.size(), 1);  // 断言统计信息的大小为 1
  auto it = stats.find(*source.get());  // 查找源码信息在统计中的位置
  EXPECT_NE(it, stats.end());  // 断言找到了源码信息
  auto& lines = it->second;  // 获取源码行的统计信息
  EXPECT_EQ(lines.size(), 1);  // 断言源码行统计信息的大小为 1
  const auto& stat = lines.at(source->lineno_for_offset(begin));  // 获取指定偏移量处的统计信息
  EXPECT_EQ(stat.count, 3);  // 断言统计信息的计数为 3
}

TEST(ScriptProfileTest, CallingOrder) {  // 定义调用顺序的脚本分析测试用例
  ScriptProfile p;  // 创建脚本分析器对象
  p.enable();  // 启用脚本分析器
  EXPECT_THROW(p.dumpStats(), c10::Error);  // 预期抛出 c10::Error 异常
  p.disable();  // 禁用脚本分析器
  auto dp = std::make_shared<profiling::Datapoint>(SourceRange{});  // 创建数据点对象
  EXPECT_THROW(p.addDatapoint(std::move(dp)), c10::Error);  // 预期抛出 c10::Error 异常
}

} // namespace jit
} // namespace torch
```