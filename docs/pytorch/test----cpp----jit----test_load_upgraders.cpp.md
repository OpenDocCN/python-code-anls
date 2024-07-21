# `.\pytorch\test\cpp\jit\test_load_upgraders.cpp`

```py
// 包含头文件：caffe2 序列化版本、Google 测试框架、Torch 模块 API、操作升级器、版本映射、序列化导入工具、JIT 测试工具
#include <caffe2/serialize/versions.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/serialization/import.h>
#include <test/cpp/jit/test_utils.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {

// 对 load 函数进行基本测试，验证能否正确加载升级器
// TODO (tugsuu) 添加更多测试
TEST(UpgraderLoad, CanPopulateUpgradersGraph) {
  // 创建一个名为 "m" 的 Torch 模块
  Module m("m");
  // 定义模块的 forward 方法，进行简单的张量除法操作
  m.define(R"(
    def forward(self, x: Tensor):
      b = 5
      return torch.div(x, b)
  )");
  // 创建一个字符串流对象 ms，并将模块 m 序列化保存到其中
  std::stringstream ms;
  m.save(ms);
  // 从字符串流 ms 中加载模块，并存储在 loaded_m 中
  auto loaded_m = torch::jit::load(ms);
  // 获取操作符版本映射
  auto version_map = get_operator_version_map();
  // 获取所有操作升级器的映射
  auto upgraders = dump_upgraders_map();

  // 遍历版本映射中的每一个条目
  for (const auto& entry : version_map) {
    // 获取当前操作的所有升级器列表
    auto list_of_upgraders_for_op = entry.second;
    // 遍历每个操作的升级器列表中的每一个升级器
    for (const auto& upgrader_entry : list_of_upgraders_for_op) {
      // 验证该升级器名称在升级器映射中能够找到
      EXPECT_TRUE(
          upgraders.find(upgrader_entry.upgrader_name) != upgraders.end());
    }
  }

  // 获取 loaded_m 中 forward 方法的计算图
  auto test_graph = loaded_m.get_method("forward").graph();
  // 检查计算图中至少有一个 aten::div 操作
  testing::FileCheck().check_count("aten::div", 1, true)->run(*test_graph);
}

} // namespace jit
} // namespace torch
```