# `.\pytorch\torch\csrc\jit\serialization\source_range_serialization.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/frontend/source_range.h>

#include <ATen/core/ivalue.h>

#include <unordered_map>
#include <vector>

namespace c10 {
struct IValue;
}

namespace torch::jit {

// 声明 IValue 结构体在 c10 命名空间中
class Pickler;
class SourceRangeSerializer;
// 声明静态常量 kByteOffsetIndex 为 0，表示字节偏移索引
static constexpr size_t kByteOffsetIndex = 0;
// 声明静态常量 kSourceRangeIndex 为 1，表示源范围索引
static constexpr size_t kSourceRangeIndex = 1;
// 声明静态常量 kSourceRangeTagIndex 为 2，表示源范围标签索引
static constexpr size_t kSourceRangeTagIndex = 2;
// 声明字符串常量 kFormatWithStringTable，表示使用带字符串表的格式
constexpr c10::string_view kFormatWithStringTable = "FORMAT_WITH_STRING_TABLE";

// SourceRangePickler 类
class SourceRangePickler {
 public:
  // 默认构造函数
  SourceRangePickler();

  // 序列化源范围记录和源范围标签为字节数组
  std::vector<char> pickle(
      const SourceRangeRecords& ranges,
      const SourceRangeTagMap& source_range_tags);

 private:
  // SourceRangeSerializer 的共享指针成员
  std::shared_ptr<SourceRangeSerializer> srs;
};

// SourceRangeDeserializer 类
class SourceRangeDeserializer {
 public:
  // 默认构造函数
  SourceRangeDeserializer() = default;
  // 显式构造函数，从 IValue 表示的文本表构造对象
  explicit SourceRangeDeserializer(const c10::IValue& text_table) {
    // 遍历文本表的每个元素，转换为共享的字符串指针并存储
    for (const auto& x : text_table.toTuple()->elements()) {
      text_table_.emplace_back(std::make_shared<std::string>(x.toStringRef()));
    }
  }
  // 反序列化给定的 IValue 为 SourceRange 对象
  SourceRange deserialize(const c10::IValue& iv);

 private:
  // 反序列化给定的 IValue 为 Source 对象的私有方法
  std::shared_ptr<Source> deserialize_source(const c10::IValue& iv);
  // 缓存反序列化后的源对象，使用 Tuple 的指针作为键，共享的 Source 指针作为值
  std::unordered_map<
      c10::intrusive_ptr<c10::ivalue::Tuple>,
      std::shared_ptr<Source>>
      cached_sources;
  // 存储文本表中的共享字符串指针
  std::vector<std::shared_ptr<std::string>> text_table_;
};

// SourceRangeUnpickler 类
class SourceRangeUnpickler {
 public:
  // 纯虚函数，根据给定的源范围查找生成它的源范围，返回可选的 SourceRange
  virtual std::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range) = 0;

  // 虚析构函数，确保正确释放资源
  virtual ~SourceRangeUnpickler() = default;
};

// 设置是否使用带字符串表的格式进行序列化和反序列化
TORCH_API void setShouldUseFormatWithStringTable(
    bool should_use_format_with_string_table);

} // namespace torch::jit
```