# `.\pytorch\torch\csrc\jit\frontend\versioned_symbols.cpp`

```
# 包含 TorchScript 前端版本化符号所需的头文件
#include <torch/csrc/jit/frontend/versioned_symbols.h>

# 包含 TorchScript 序列化版本信息的头文件
#include <caffe2/serialize/versions.h>

# 包含 TorchScript JIT 的公共 API 头文件
#include <torch/csrc/api/include/torch/jit.h>

# 包含标准库中的无序映射容器
#include <unordered_map>

# 定义 torch::jit 命名空间
namespace torch::jit {

// Note [Versioned Symbols]
// 代码库中关于版本化符号的注释，解释了当符号的行为或架构发生变化时，如何通过
// 版本化符号模式来保证向后兼容性。

// Helper to hold the version range (inclusive on both ends) and the symbol
// to map to for that range.
// 辅助结构体，用于保存版本范围（包含两端）和该范围对应的符号映射关系
struct SymbolRange {
  // 构造函数，初始化起始版本、结束版本和符号对象
  SymbolRange(
      const uint64_t _start_version,
      const uint64_t _end_version,
      const Symbol _sym)
      : start_version_{_start_version},
        end_version_{_end_version},
        sym_{_sym} {}

  // 起始版本号
  const uint64_t start_version_;
  // 结束版本号
  const uint64_t end_version_;
  // 符号对象
  const Symbol sym_;
};

// 创建符号到版本范围映射的静态无序映射容器
static std::unordered_map<Symbol, SymbolRange> symbol_range_map({
    // 将符号 "aten::_test_serialization_subcmul" 映射到其对应的版本范围及符号
    {Symbol::fromQualString("aten::_test_serialization_subcmul"),
     {0,
      2,
      Symbol::fromQualString("upgraders::_test_serialization_subcmul_0_2")}},
    {
        // 创建一个包含两个元素的字典，键为字符串 "aten::div" 对应的 Symbol 对象，值为一个包含三个整数的列表
        Symbol::fromQualString("aten::div"),
        {0, 3, Symbol::fromQualString("upgraders::div_0_3")}
    },
    {
        // 创建一个包含两个元素的字典，键为字符串 "aten::div_" 对应的 Symbol 对象，值为一个包含三个整数的列表
        Symbol::fromQualString("aten::div_"),
        {0, 3, Symbol::fromQualString("upgraders::div__0_3")}
    },
    {
        // 创建一个包含两个元素的字典，键为字符串 "aten::full" 对应的 Symbol 对象，值为一个包含四个整数的列表
        Symbol::fromQualString("aten::full"),
        {0, 4, Symbol::fromQualString("upgraders::full_0_4")}
    },
});

// 创建静态的无序映射，将节点类型映射到最小版本号
static std::unordered_map<NodeKind, uint64_t> kind_min_version_map({
    {aten::div, 4},        // 将 aten::div 映射到版本号 4
    {aten::div_, 4},       // 将 aten::div_ 映射到版本号 4
    {aten::full, 5},       // 将 aten::full 映射到版本号 5，并标注禁止使用魔数的规则（NOLINT）
});

// 根据名称和版本号获取符号
Symbol get_symbol_for_version(const Symbol name, const uint64_t version) {
  auto it = symbol_range_map.find(name);  // 在 symbol_range_map 中查找名称对应的符号
  if (it == symbol_range_map.end()) {     // 如果未找到，直接返回输入的名称
    return name;
  }

  auto& entry = it->second;
  if (entry.start_version_ <= version && entry.end_version_ >= version) {  // 如果版本号在范围内，则返回对应的符号
    return entry.sym_;
  }

  return name;  // 否则返回输入的名称
}

// 根据节点类型获取最小版本号
uint64_t get_min_version_for_kind(const NodeKind& kind) {
  auto it = kind_min_version_map.find(kind);  // 在 kind_min_version_map 中查找节点类型对应的最小版本号
  if (it == kind_min_version_map.end()) {     // 如果未找到，返回版本号 0
    return 0;
  }

  return it->second;  // 否则返回找到的最小版本号
}

} // namespace torch::jit  // 结束 torch::jit 命名空间
```