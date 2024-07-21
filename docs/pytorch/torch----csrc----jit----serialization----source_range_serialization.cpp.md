# `.\pytorch\torch\csrc\jit\serialization\source_range_serialization.cpp`

```py
// 包含 Torch 序列化中的源范围相关头文件
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/source_range_serialization_impl.h>

// 包含 C10 异常处理和标志工具相关头文件
#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
// 包含 Torch 移动设备类型解析器和 pickle 相关头文件
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/pickle.h>

// 包含算法和内存管理相关标准库头文件
#include <algorithm>
#include <memory>

// Torch JIT 命名空间
namespace torch::jit {

// 控制是否在保存模型为 .pt 文件时使用紧凑的 debug_pkl 格式
// 紧凑文件较小但无法被旧 Torch 二进制加载
// TODO(qihan) 当所有二进制文件都使用字符串表时删除此标记
thread_local bool should_use_format_with_string_table_ = true;

// SourceRangeSerializer 类定义
class SourceRangeSerializer {
 public:
  // 将 SourceRange 序列化为 Tuple[SourceType, int, int]
  // 其中 SourceType = Tuple[int, int, int, List[int]]
  // 第一个和第二个 int 是在 textSaved 返回的向量中的位置，这些向量在处理所有 Range 后返回
  // textSaved() 返回一个字符串向量，Source 的序列化形式
  c10::IValue serialize(const SourceRange& sr);

  // 返回 texts_ 向量，该向量保存的是文本内容
  const std::vector<c10::IValue>& texts_saved() {
    return texts_;
  }

  // 构造函数，初始化 texts_ 向量并添加一个空字符串到 text_to_idx_ 映射
  SourceRangeSerializer() {
    texts_.emplace_back("");
    text_to_idx_[texts_.back().toStringRef()] = 0;
  }

 private:
  // 序列化 Source 为 Tuple[str, Optional[str], int, List[int]]
  // 由于许多 SourceRanges 可能引用相同的 Source，因此这里缓存序列化的 Source
  c10::IValue serialize_source(const std::shared_ptr<Source>& s);

  // 存储文本并获取其索引
  int64_t store_text_and_get_index(const std::string& text_view);

  // texts_ 向量，保存序列化的文本内容
  std::vector<c10::IValue> texts_;
  // 文本到索引的映射
  std::unordered_map<c10::string_view, int64_t> text_to_idx_;
};

// SourceRangeDeserializer 类中 deserialize 函数的定义
SourceRange SourceRangeDeserializer::deserialize(const c10::IValue& iv) {
  // 获取元组中的元素
  const auto& tup_elems = iv.toTupleRef().elements();
  // 断言元组长度为3
  TORCH_INTERNAL_ASSERT(tup_elems.size() == 3);
  // 反序列化第一个元素为 Source 对象
  std::shared_ptr<Source> source_ = deserialize_source(tup_elems[0]);
  // 反序列化第二个元素为 start_
  int64_t start_ = tup_elems[1].toInt();
  // 反序列化第三个元素为 end_
  int64_t end_ = tup_elems[2].toInt();
  // 返回构建的 SourceRange 对象
  return SourceRange(source_, start_, end_);
}

// SourceRangeDeserializer 类中 deserialize_source 函数的定义
std::shared_ptr<Source> SourceRangeDeserializer::deserialize_source(
    const c10::IValue& iv) {
  // 将 iv 转换为元组
  auto tup = iv.toTuple();
  // 查找是否有缓存的序列化 Source
  auto it = cached_sources.find(tup);
  // 如果找到缓存的 Source，直接返回
  if (it != cached_sources.end()) {
    return it->second;
  }
  // 否则解析元组中的各个元素
  std::shared_ptr<Source> source;
  const auto& tup_elems = tup->elements();
  // 断言元组长度为3
  TORCH_INTERNAL_ASSERT(tup_elems.size() == 3);
  // 如果 text_table_ 非空，解析第一个元素为 textIndex
  if (!text_table_.empty()) {
    const auto& textIndex = tup_elems[0].toIntList();
    int64_t fnameIndex = tup_elems[1].toInt();
    int64_t starting_line_no_ = tup_elems[2].toInt();
    std::optional<std::string> filename = c10::nullopt;

    // 检查 fnameIndex 是否在 text_table_ 范围内
    TORCH_CHECK(
        (uint64_t)fnameIndex < text_table_.size(),
        "Text table index is out of range")
    // 获取文件名
    filename = *text_table_[fnameIndex];

    std::vector<c10::string_view> pieces;
    std::vector<std::shared_ptr<std::string>> strs;
    // 对于textIndex中的每个索引i，执行以下操作
    for (int64_t i : textIndex) {
      // 从text_table_中获取索引i对应的指针，构造pieces容器
      pieces.emplace_back(*text_table_[i]);
      // 将text_table_中索引i对应的指针存入strs容器
      strs.emplace_back(text_table_[i]);
    }

    // 使用pieces和strs构造StringCordView对象str_cord
    StringCordView str_cord(std::move(pieces), std::move(strs));

    // 如果条件成立，即tup_elems包含3个元素
    if (tup_elems.size() == 3) {
      // 从tup_elems中获取第一个元素作为text_
      std::string text_ = tup_elems[0].toStringRef();
      // 从tup_elems中获取第二个元素作为可选的filename_
      std::optional<std::string> filename_ =
          tup_elems[1].toOptional<std::string>();
      // 从tup_elems中获取第三个元素作为starting_line_no_
      int64_t starting_line_no_ = tup_elems[2].toInt();
      // 使用text_、filename_和starting_line_no_构造Source对象source
      source = std::make_shared<Source>(
          std::move(text_), std::move(filename_), starting_line_no_);
    } else {
      // 如果条件不成立，即tup_elems不包含3个元素，抛出异常或处理错误情况
      // （这部分代码未提供具体的异常处理或错误处理细节）
      // 在实际代码中应该有错误处理机制来处理这种情况
    }

    // 将生成的source对象存入cached_sources中，使用tup作为键
    cached_sources[tup] = source;
    // 返回生成的source对象
    return source;
}

// 序列化源代码范围对象为 c10::IValue
c10::IValue SourceRangeSerializer::serialize(const SourceRange& sr) {
  // 创建包含源代码信息的元组，包括源、起始行和结束行
  return c10::ivalue::Tuple::create(
      serialize_source(sr.source()), (int64_t)sr.start(), (int64_t)sr.end());
}

// 存储文本并返回索引
int64_t SourceRangeSerializer::store_text_and_get_index(
    const std::string& text_view) {
  // 查找文本在索引中的位置
  auto text_iter = text_to_idx_.find(text_view);
  if (text_iter == text_to_idx_.end()) {
    // 如果文本不存在于索引中，则将其添加到文本列表末尾并更新索引
    int64_t text_pos = static_cast<int64_t>(texts_.size());
    texts_.emplace_back(text_view);
    text_to_idx_[texts_.back().toStringView()] = text_pos;
    return text_pos;
  } else {
    // 如果文本已存在于索引中，则返回其索引
    return text_iter->second;
  }
}

// 序列化源对象为 c10::IValue
c10::IValue SourceRangeSerializer::serialize_source(
    const std::shared_ptr<Source>& s) {
  // 如果已经序列化过该源对象，则直接返回其序列化结果
  if (serialized_sources.count(s)) {
    return serialized_sources.at(s);
  }
  
  // 初始化序列化对象和行号列表
  c10::intrusive_ptr<c10::ivalue::Tuple> serialized;
  c10::List<int64_t> lines;
  
  // 根据是否应该使用带有字符串表的格式来选择序列化方式
  if (should_use_format_with_string_table_) {
    // 如果源对象为空，则创建一个空元组
    if (s == nullptr) {
      serialized = c10::ivalue::Tuple::create({lines, 0, 0});
    } else {
      // 遍历每一行并存储文本在索引中的位置
      for (size_t lineno = 0; lineno < s->num_lines(); lineno++) {
        std::string line_content = s->get_line(lineno).str();
        int64_t text_pos = store_text_and_get_index(line_content);
        lines.push_back(text_pos);
      }
      
      // 存储文件名在索引中的位置
      int64_t fname_pos = 0;
      if (s->filename().has_value()) {
        fname_pos = store_text_and_get_index(*s->filename());
      }
      
      // 创建包含行号列表、文件名位置和起始行号的元组
      serialized = c10::ivalue::Tuple::create(
          {lines, fname_pos, (int64_t)s->starting_line_no()});
    }
  } else {
    // 如果不使用字符串表格式，则直接序列化文本和文件名
    if (s == nullptr) {
      serialized = c10::ivalue::Tuple::create({"", "", 0});
    } else {
      serialized = c10::ivalue::Tuple::create(
          {s->text_str().str(), s->filename(), (int64_t)s->starting_line_no()});
    }
  }
  
  // 将序列化结果存储起来，并返回
  serialized_sources[s] = serialized;
  return serialized;
}

// SourceRangePickler 类的构造函数，初始化 SourceRangeSerializer 对象
SourceRangePickler::SourceRangePickler() : srs(new SourceRangeSerializer()) {}

// 对源代码范围进行序列化，并返回序列化后的字节流
std::vector<char> SourceRangePickler::pickle(
    const SourceRangeRecords& ranges,
    const SourceRangeTagMap& source_range_tags) {
  std::vector<c10::IValue> ivalues;
  
  // 遍历源代码范围记录
  for (const auto& range : ranges) {
    int64_t source_range_tag{-1};
    // 查找源代码范围标签在映射中的位置
    const auto& it = source_range_tags.find(range.range);
    if (it != source_range_tags.end()) {
      source_range_tag = it->second;
    }

    // 创建包含字节大小、序列化源代码范围和标签的元组，并添加到 ivalues 中
    ivalues.emplace_back(c10::ivalue::Tuple::create(
        {(int64_t)range.bytes,
         srs->serialize(range.range),
         static_cast<int64_t>(source_range_tag)}));
  }

  std::vector<at::Tensor> table;
  auto textTable = c10::ivalue::Tuple::create(srs->texts_saved());
  auto ivalue = c10::ivalue::Tuple::create(std::move(ivalues));
  std::vector<char> result;
  
  // 根据是否应该使用带有字符串表的格式来选择序列化方式
  if (should_use_format_with_string_table_) {
    // 使用带有字符串表的格式进行序列化，并返回序列化后的结果
    result = jit::pickle(
        c10::ivalue::Tuple::create({kFormatWithStringTable, textTable, ivalue}),
        &table);
  } else {
    // 直接序列化 ivalue，并返回序列化后的结果
    result = jit::pickle(ivalue, &table);
  }
  
  // 检查是否没有写入任何张量
  TORCH_CHECK(table.empty(), "Expected 0 tensors to be written");
  return result;
}
// ConcreteSourceRangeUnpickler 类的构造函数，接受一个数据指针和大小作为参数
ConcreteSourceRangeUnpickler::ConcreteSourceRangeUnpickler(
    at::DataPtr&& data,
    size_t size)
    : data(std::move(data)), // 使用 std::move 将数据指针移动到成员变量 data
      size(size),            // 初始化成员变量 size
      deserializer(nullptr), // 初始化 deserializer 为 nullptr
      unpickled_records(nullptr) {} // 初始化 unpickled_records 为 nullptr

// 解析函数 unpickle
void ConcreteSourceRangeUnpickler::unpickle() {
  // 使用互斥量进行线程安全保护
  std::lock_guard<std::mutex> guard(mutex);
  // 如果已经解析过，则直接返回
  if (unpickled_records) {
    return;
  }

  // 使用 jit::unpickle 函数对数据进行反序列化，得到 IValue 元组
  auto ivaluesTuple = jit::unpickle(
                          reinterpret_cast<const char*>(data.get()), // 将数据指针转换为 char* 进行反序列化
                          size,                                     // 数据大小
                          nullptr,
                          {},                                        // 空字典
                          c10::parseType)
                          .toTuple();

  // 获取 ivaluesTuple 中的元素
  const auto& ivalues = ivaluesTuple->elements();

  // 检查 ivalues 是否为空
  TORCH_CHECK(
      !ivalues.empty(), "Invalid unpickle operation: empty ivalues tuple");

  // 创建 unpickled_records 对象，用于存储反序列化后的记录
  unpickled_records = std::make_shared<SourceRangeRecords>();

  // 定义 lines 变量，用于存储反序列化后的数据
  IValue lines;

  // 检查第一个元素是否为字符串，并且字符串内容符合预期格式
  if (ivalues[0].isString() &&
      kFormatWithStringTable == ivalues[0].toStringRef()) {
    // 如果符合预期格式，创建对应的 deserializer 对象和 lines 变量
    deserializer = std::make_shared<SourceRangeDeserializer>(ivalues[1]);
    lines = ivalues[2];
  } else {
    // 否则，创建默认的 deserializer 对象，并将整个 ivaluesTuple 赋值给 lines
    deserializer = std::make_shared<SourceRangeDeserializer>();
    lines = ivaluesTuple;
  }

  // 遍历 lines 中的元素
  for (auto& val : lines.toTuple()->elements()) {
    // 获取元组中的元素
    const auto& tup_elems = val.toTupleRef().elements();
    // 获取字节偏移量
    int64_t offset = tup_elems[kByteOffsetIndex].toInt();
    // 使用 deserializer 对象反序列化源代码范围
    auto source_range = deserializer->deserialize(tup_elems[kSourceRangeIndex]);
    // 将解析后的源代码范围和偏移量添加到 unpickled_records 中
    unpickled_records->emplace_back(offset, std::move(source_range));
  }
}

// 查找生成给定源代码范围的源代码范围
std::optional<SourceRange> ConcreteSourceRangeUnpickler::
    findSourceRangeThatGenerated(const SourceRange& range) {
  // 执行 unpickle 操作以确保 unpickled_records 已经填充
  unpickle();

  // 创建查询对象
  auto query = TaggedRange(range.start(), SourceRange{});

  // 使用 std::upper_bound 查找大于查询对象的第一个元素
  auto entry = std::upper_bound(
      unpickled_records->begin(),
      unpickled_records->end(),
      query,
      [](const TaggedRange& a, const TaggedRange& b) -> bool {
        return a.bytes < b.bytes; // 比较 bytes 字段大小
      });

  // 注意：由于 upper_bound 返回大于查询对象的第一个元素，因此需要对迭代器进行调整
  if (entry != unpickled_records->begin()) {
    return (entry - 1)->range; // 返回找到的范围
  }

  return c10::nullopt; // 如果未找到，返回空值
}

// 设置是否使用字符串表格式进行反序列化的选项
TORCH_API void setShouldUseFormatWithStringTable(
    bool should_use_format_with_string_table) {
  should_use_format_with_string_table_ = should_use_format_with_string_table;
}

} // namespace torch::jit
```