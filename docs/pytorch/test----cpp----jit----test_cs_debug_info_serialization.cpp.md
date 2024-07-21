# `.\pytorch\test\cpp\jit\test_cs_debug_info_serialization.cpp`

```py
// 包含测试相关的头文件
#include <test/cpp/jit/test_utils.h>

// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 PyTorch 核心 TensorOptions 头文件
#include <c10/core/TensorOptions.h>

// 包含 PyTorch 自动微分生成的变量工厂头文件
#include <torch/csrc/autograd/generated/variable_factories.h>

// 包含 PyTorch JIT API 中模块相关的头文件
#include <torch/csrc/jit/api/module.h>

// 包含 PyTorch JIT 后端调试处理的头文件
#include <torch/csrc/jit/backends/backend_debug_handler.h>

// 包含 PyTorch JIT 前端解析器相关的头文件
#include <torch/csrc/jit/frontend/resolver.h>

// 包含 PyTorch 移动端导入相关的头文件
#include <torch/csrc/jit/mobile/import.h>

// 包含 PyTorch 移动端模块相关的头文件
#include <torch/csrc/jit/mobile/module.h>

// 包含 PyTorch JIT Passes 中内联处理的头文件
#include <torch/csrc/jit/passes/inliner.h>

// 包含 PyTorch JIT 序列化调用栈调试信息的头文件
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>

// 包含 PyTorch JIT 导出相关的头文件
#include <torch/csrc/jit/serialization/export.h>

// 包含 PyTorch JIT 导入相关的头文件
#include <torch/csrc/jit/serialization/import.h>

// 包含 PyTorch 自定义类相关的头文件
#include <torch/custom_class.h>

// 包含 PyTorch 核心头文件
#include <torch/torch.h>

// 包含 C++ 标准库中的栈相关头文件
#include <stack>

// 包含 C++ 标准库中的无序集合相关头文件
#include <unordered_set>

// 所有的测试代码位于 torch::jit 命名空间下
namespace torch {
namespace jit {

// 匿名命名空间内定义一个验证调试信息的函数
namespace {
bool validate_debug_info(
    const DebugInfoTuple& pre_serialize, // 前序列化的调试信息元组
    const DebugInfoTuple& post_serialize) { // 后序列化的调试信息元组
  auto sr1 = std::get<kDebugInfoTupleSourceRangeIndex>(pre_serialize); // 获取前序列化调试信息的源代码范围
  auto sr2 = std::get<kDebugInfoTupleSourceRangeIndex>(post_serialize); // 获取后序列化调试信息的源代码范围
  if (sr1 != sr2) { // 比较两个源代码范围是否相同
    return false;
  }
  auto csptr1 = std::get<kDebugInfoTupleInlinedCSIndex>(pre_serialize); // 获取前序列化调试信息的内联调用栈指针
  auto csptr2 = std::get<kDebugInfoTupleInlinedCSIndex>(post_serialize); // 获取后序列化调试信息的内联调用栈指针
  if (!csptr1.defined()) { // 如果前序列化调试信息的内联调用栈未定义
    return !csptr2.defined(); // 则后序列化调试信息的内联调用栈也应未定义
  }
  if (!csptr2.defined()) { // 如果后序列化调试信息的内联调用栈未定义，而前序列化的定义了
    return false;
  }
  auto vec1 = csptr1->vec(); // 获取前序列化调试信息内联调用栈的向量
  auto vec2 = csptr2->vec(); // 获取后序列化调试信息内联调用栈的向量
  if (vec1.size() != vec2.size()) { // 比较两个内联调用栈向量的大小
    return false;
  }
  while (csptr1) { // 循环遍历前序列化调试信息的内联调用栈
    auto rhs_sr = csptr1->source_range(); // 获取右手边的源代码范围
    auto lhs_sr = csptr2->source_range(); // 获取左手边的源代码范围
    auto rhs_module = csptr1->module_instance(); // 获取右手边的模块实例
    auto lhs_module = csptr2->module_instance(); // 获取左手边的模块实例
    std::string rhs_fn_name, lhs_fn_name; // 定义存储右手边和左手边函数名的字符串
    if (csptr1->function()) { // 如果右手边的函数存在
      rhs_fn_name = csptr1->function()->name(); // 获取右手边函数的名称
    } else {
      rhs_fn_name = csptr1->function_name(); // 否则获取右手边函数的函数名
    }
    if (csptr2->function()) { // 如果左手边的函数存在
      lhs_fn_name = csptr2->function()->name(); // 获取左手边函数的名称
    } else {
      lhs_fn_name = csptr2->function_name(); // 否则获取左手边函数的函数名
    }
    if (!((rhs_module.has_value() == lhs_module.has_value()) && // 比较模块实例是否都有值
          (rhs_module.has_value() &&
           (rhs_module.value().class_type()->name().value() == // 比较模块类型名称
            lhs_module.value().class_type()->name().value()) &&
           (rhs_module.value().instance_name() == // 比较模块实例名称
            lhs_module.value().instance_name())) &&
          (rhs_fn_name == lhs_fn_name) && (rhs_sr == lhs_sr))) { // 比较函数名和源代码范围
      return false;
    }
    if (csptr1->callee()) { // 如果右手边的调用者存在
      csptr1 = csptr1->callee().value(); // 获取右手边的调用者
      csptr2 = csptr2->callee().value(); // 获取左手边的调用者
    } else {
      csptr1 = c10::intrusive_ptr<InlinedCallStack>(); // 否则将右手边的调用栈指针置为空
    }
  }
  return true; // 验证通过，返回 true
}
}

// 定义一个测试用例，测试两个子模块的调试信息序列化
TEST(CSDebugInfoSerializaitionTest, TwoSubmodules) {
  std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>(); // 创建共享的编译单元指针
  Module a("A", cu); // 创建名称为 "A" 的模块
  a.define(R"JIT(
    def forward(self, x):
      return x + 1
  )JIT"); // 在模块 A 中定义前向方法

  Module b("B", cu); // 创建名称为 "B" 的模块
  b.define(R"JIT(
    // 此处应该是定义模块 B 中的某些功能，代码未完整提供
  )JIT"); // 在模块 B 中定义某些功能，未提供完整代码
}
} // namespace jit
} // namespace torch
  // 定义一个 forward 方法，接受参数 x 并返回 x + 2
  )JIT");
  // 创建名为 c 的 Module 对象，其来源于 cu
  Module c("C", cu);
  // 将名为 a 的 Module 注册为 "A0" 到 c 中
  c.register_module("A0", a);
  // 将名为 b 的 Module 注册为 "B0" 到 c 中
  c.register_module("B0", b);
  // 定义一个 forward 方法，接受参数 x 并返回 self.A0.forward(x) + self.B0.forward(x)
  c.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(x) + self.B0.forward(x)
  )JIT");

  // 创建一个用于调试信息记录的 BackendDebugInfoRecorder 对象
  BackendDebugInfoRecorder debug_info_recorder;
  // 获取名为 "forward" 的方法对应的图形表示
  auto graph = c.get_method("forward").graph();
  // 对图进行内联优化处理
  Inline(*graph);
  // 创建一个堆栈，用于遍历的 Block 对象
  std::stack<Block*> blocks_to_visit;

  // 从源范围到调试句柄的映射
  SourceRangeTagMap source_range_tags;
  // 从调试句柄到源范围的映射
  ska::flat_hash_map<int64_t, SourceRange> source_range_map;
  // 当前源范围标签
  int64_t source_range_tag{0};

  // 将 graph 的初始 block 放入堆栈中，开始遍历
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    // 弹出堆栈中的一个 Block 对象
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历当前 Block 中的所有节点
    for (Node* n : b->nodes()) {
      // 将当前节点的源范围映射到当前的源范围标签
      source_range_tags[n->sourceRange()] = source_range_tag;
      // 将当前的源范围标签映射到当前节点的源范围
      source_range_map[source_range_tag] = n->sourceRange();
      // 递增源范围标签
      source_range_tag++;
      // 获取当前节点的下一个调试句柄并记录
      debug_info_recorder.getNextDebugHandle(n);
      // 如果当前节点具有调用堆栈信息
      if (n->callstack().has_value()) {
        // 遍历当前节点的调用堆栈信息
        for (const auto& e : n->callstack().value()->vec()) {
          // 获取调用堆栈元组中的源范围，映射到当前的源范围标签
          auto sr = std::get<1>(e);
          source_range_tags[sr] = source_range_tag;
          // 将当前的源范围标签映射到调用堆栈元组的源范围
          source_range_map[source_range_tag] = sr;
          // 递增源范围标签
          source_range_tag++;
        }
      }
    }
  }
  // 停止记录调试信息，并获取调试句柄和指针映射
  auto debug_handle_cs_ptr_map = debug_info_recorder.stopRecording();
  // 创建 CallStackDebugInfoPickler 对象用于序列化调试信息
  CallStackDebugInfoPickler cs_debug_info_pickler;
  // 使用 Pickler 序列化调试句柄和源范围标签数据
  auto cs_data =
      cs_debug_info_pickler.pickle(debug_handle_cs_ptr_map, source_range_tags);
  // 创建一个指向序列化数据的数据指针，位于 CPU 上
  at::DataPtr data_ptr(cs_data.data(), DeviceType::CPU);
  // 创建 CallStackDebugInfoUnpickler 对象用于反序列化调试信息
  CallStackDebugInfoUnpickler unpickler;
  // 使用 Unpickler 反序列化调试信息并验证
  auto deserialized_cs_map = unpickler.unpickle(
      std::move(data_ptr), cs_data.size(), source_range_map, cu);
  // 遍历调试句柄和指针映射，验证序列化前后的调试信息是否一致
  for (const auto& it : debug_handle_cs_ptr_map) {
    auto handle = it.first;
    auto debug_info_one = it.second;
    // 检查反序列化后的调试映射中是否包含当前句柄
    TORCH_CHECK(
        deserialized_cs_map.count(handle),
        "Serialized debug handle must be in deserialized map.");
    // 获取反序列化后的调试信息并验证
    auto debug_info_two = deserialized_cs_map[handle];
    ASSERT_TRUE(validate_debug_info(debug_info_one, debug_info_two));
  }
}
} // namespace
} // namespace jit
} // namespace torch


注释：

// 结束了当前的命名空间定义，回到上一级命名空间
}
// 结束了当前的命名空间定义，进入上一级的命名空间 "jit"
} // namespace jit
// 结束了当前的命名空间定义，进入上一级的命名空间 "torch"
} // namespace torch
```