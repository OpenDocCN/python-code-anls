# `.\pytorch\torch\csrc\jit\jit_log.cpp`

```py
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/serialization/python_print.h>

namespace torch {
namespace jit {

// 日志配置类，用于控制 JIT 日志的级别和输出
class JitLoggingConfig {
 public:
  // 获取 JitLoggingConfig 的单例实例
  static JitLoggingConfig& getInstance() {
    static JitLoggingConfig instance;
    return instance;
  }
  
  // 禁止拷贝构造函数和赋值运算符
  JitLoggingConfig(JitLoggingConfig const&) = delete;
  void operator=(JitLoggingConfig const&) = delete;

 private:
  std::string logging_levels;  // 存储日志级别的字符串
  std::unordered_map<std::string, size_t> files_to_levels;  // 文件名到日志级别的映射
  std::ostream* out;  // 输出流指针，用于日志输出

  // 构造函数，初始化日志级别和输出流
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  JitLoggingConfig() {
    // 从环境变量 PYTORCH_JIT_LOG_LEVEL 中获取日志级别字符串
    const char* jit_log_level = std::getenv("PYTORCH_JIT_LOG_LEVEL");
    logging_levels.assign(jit_log_level == nullptr ? "" : jit_log_level);
    out = &std::cerr;  // 默认将日志输出到标准错误流
    parse();  // 解析日志级别字符串
  }
  
  // 解析日志级别字符串，将其转换为文件名到级别的映射
  void parse();

 public:
  // 获取当前的日志级别字符串
  std::string getLoggingLevels() const {
    return this->logging_levels;
  }
  
  // 设置日志级别字符串，并重新解析
  void setLoggingLevels(std::string levels) {
    this->logging_levels = std::move(levels);
    parse();  // 更新文件名到级别的映射
  }

  // 获取文件名到日志级别的映射
  const std::unordered_map<std::string, size_t>& getFilesToLevels() const {
    return this->files_to_levels;
  }

  // 设置日志输出流
  void setOutputStream(std::ostream& out_stream) {
    this->out = &out_stream;
  }

  // 获取当前的日志输出流
  std::ostream& getOutputStream() {
    return *(this->out);
  }
};

// 获取当前的 JIT 日志级别字符串
std::string get_jit_logging_levels() {
  return JitLoggingConfig::getInstance().getLoggingLevels();
}

// 设置 JIT 日志级别字符串
void set_jit_logging_levels(std::string level) {
  JitLoggingConfig::getInstance().setLoggingLevels(std::move(level));
}

// 设置 JIT 日志输出流
void set_jit_logging_output_stream(std::ostream& stream) {
  JitLoggingConfig::getInstance().setOutputStream(stream);
}

// 获取当前的 JIT 日志输出流
std::ostream& get_jit_logging_output_stream() {
  return JitLoggingConfig::getInstance().getOutputStream();
}

// 获取节点的字符串表示（用于日志输出）
// 包括输出、节点类型和输出等信息
std::string getHeader(const Node* node) {
  std::stringstream ss;
  node->print(ss, 0, {}, false, false, false, false);  // 将节点信息打印到 stringstream 中
  return ss.str();  // 返回 stringstream 的字符串表示
}

// 解析日志级别字符串，构建文件名到日志级别的映射
void JitLoggingConfig::parse() {
  std::stringstream in_ss;
  in_ss << "function:" << this->logging_levels;

  files_to_levels.clear();  // 清空原有的映射

  std::string line;
  while (std::getline(in_ss, line, ':')) {
    if (line.empty()) {
      continue;
    }

    auto index_at = line.find_last_of('>');
    auto begin_index = index_at == std::string::npos ? 0 : index_at + 1;
    size_t logging_level = index_at == std::string::npos ? 0 : index_at + 1;
    auto end_index = line.find_last_of('.') == std::string::npos
        ? line.size()
        : line.find_last_of('.');
    auto filename = line.substr(begin_index, end_index - begin_index);
    files_to_levels.insert({filename, logging_level});  // 插入文件名和日志级别的映射
  }
}

} // namespace jit
} // namespace torch
// 判断给定文件名对应的日志级别是否启用
bool is_enabled(const char* cfname, JitLoggingLevels level) {
  // 获取全局单例的日志配置对象，获取文件名到级别的映射表
  const auto& files_to_levels =
      JitLoggingConfig::getInstance().getFilesToLevels();
  // 将 C 风格字符串文件名转换为 C++ 的 std::string
  std::string fname{cfname};
  // 剥离文件路径，保留文件名部分
  fname = c10::detail::StripBasename(fname);
  // 获取文件名中最后一个点（.）之前的部分，作为无扩展名的文件名
  const auto end_index = fname.find_last_of('.') == std::string::npos
      ? fname.size()
      : fname.find_last_of('.');
  const auto fname_no_ext = fname.substr(0, end_index);

  // 在映射表中查找无扩展名文件名对应的日志级别
  const auto it = files_to_levels.find(fname_no_ext);
  // 如果找不到对应的条目，则日志级别未启用，返回 false
  if (it == files_to_levels.end()) {
    return false;
  }

  // 判断给定的日志级别是否不高于映射表中的日志级别，决定是否启用日志
  return level <= static_cast<JitLoggingLevels>(it->second);
}

// 在 GraphExecutor 中调用 log_function 时，由于无法访问原始函数，
// 需要构造一个虚拟函数以供 PythonPrint 使用
std::string log_function(const std::shared_ptr<torch::jit::Graph>& graph) {
  // 创建一个 GraphFunction 对象，命名为 "source_dump"，用于打印函数信息
  torch::jit::GraphFunction func("source_dump", graph, nullptr);
  std::vector<at::IValue> constants;
  PrintDepsTable deps;
  PythonPrint pp(constants, deps);
  // 打印函数信息到 PythonPrint 对象中
  pp.printFunction(func);
  // 返回打印结果的字符串表示
  return pp.str();
}

// 给输入的每一行字符串添加指定前缀，返回添加前缀后的完整字符串
std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str) {
  // 将输入字符串包装成字符串流
  std::stringstream in_ss(in_str);
  std::stringstream out_ss;
  std::string line;
  // 逐行读取输入字符串，每行添加指定前缀后写入输出字符串流
  while (std::getline(in_ss, line)) {
    out_ss << prefix << line << std::endl;
  }

  // 返回输出字符串流的字符串表示，即添加前缀后的完整字符串
  return out_ss.str();
}

// 构造带有级别、文件名、行号的前缀，并调用 jit_log_prefix 添加到输入字符串每行的前面
std::string jit_log_prefix(
    JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str) {
  // 构造带有级别、文件名、行号的前缀字符串
  std::stringstream prefix_ss;
  prefix_ss << "[";
  prefix_ss << level << " ";
  prefix_ss << c10::detail::StripBasename(std::string(fn)) << ":";
  prefix_ss << std::setfill('0') << std::setw(3) << l;
  prefix_ss << "] ";

  // 调用前一个 jit_log_prefix 函数，添加前缀到输入字符串每行，并返回完整的结果
  return jit_log_prefix(prefix_ss.str(), in_str);
}

// 自定义 JitLoggingLevels 枚举类型的输出流运算符重载，转换为对应的字符串输出
std::ostream& operator<<(std::ostream& out, JitLoggingLevels level) {
  // 根据枚举值输出对应的字符串表示
  switch (level) {
    case JitLoggingLevels::GRAPH_DUMP:
      out << "DUMP";
      break;
    case JitLoggingLevels::GRAPH_UPDATE:
      out << "UPDATE";
      break;
    case JitLoggingLevels::GRAPH_DEBUG:
      out << "DEBUG";
      break;
    default:
      // 如果枚举值无效，抛出断言错误
      TORCH_INTERNAL_ASSERT(false, "Invalid level");
  }

  // 返回输出流
  return out;
}

// 命名空间结束注释
} // namespace jit
} // namespace torch
```