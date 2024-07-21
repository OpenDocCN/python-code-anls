# `.\pytorch\torch\csrc\lazy\core\debug_util.cpp`

```py
// 包含头文件，用于C++的范围迭代器和Torch的调试工具
#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/debug_util.h>

// 包含LazyTensor相关的后端设备、辅助函数、IR表示、IR转储工具及其唯一性管理
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/unique.h>

// 包含文件流、互斥锁、字符串流和无序集合的头文件
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

// Torch命名空间下的Lazy子命名空间
namespace torch {
namespace lazy {
namespace {

// 根据环境变量获取字符串值，如果未设置则返回默认值
std::string GetEnvString(const char* name, const std::string& defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? env : defval;
}

// 获取默认的图形格式，根据环境变量LTC_SAVE_TENSORS_FMT决定
DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str = GetEnvString("LTC_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "backend") {
    return DebugUtil::GraphFormat::kBackend;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  }
  // 若格式无效则记录错误并返回默认文本格式
  LOG(ERROR) << "Invalid save graph format: " << fmt_str;
  return DebugUtil::GraphFormat::kText;
}

// 加载实验名称集合，从环境变量LTC_EXPERIMENTAL获取，用冒号分隔不同实验名
std::unordered_set<std::string>* LoadExperiments() {
  std::unique_ptr<std::unordered_set<std::string>> xset =
      std::make_unique<std::unordered_set<std::string>>();
  std::string experiments = GetEnvString("LTC_EXPERIMENTAL", "");
  std::vector<std::string> experiment_list =
      torch::lazy::StrSplit(experiments, ':');
  // 将实验名添加到集合中
  for (auto& name : experiment_list) {
    xset->insert(name);
  }
  return xset.release();
}

} // namespace

// 返回一个静态向量，表示没有Python帧的源位置
static std::vector<SourceLocation> NoPythonFrames() {
  SourceLocation dummy_loc;
  dummy_loc.file = "No Python Frames";
  return {dummy_loc};
}

// 获取Python帧的函数，返回一个静态函数对象
std::function<std::vector<SourceLocation>()>& GetPythonFramesFunction() {
  static std::function<std::vector<SourceLocation>()> func_ = NoPythonFrames;
  return func_;
}

// 获取默认的图形格式，静态变量缓存首次计算的结果
DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

// 获取Python中的第一个用户帧，若无有效帧则返回空字符串
std::string GetFirstUserFrameInPython() {
  std::string empty;
  if (!torch::lazy::GetPythonFramesFunction()) {
    return empty;
  }

  auto frames = torch::lazy::GetPythonFramesFunction()();

  for (auto i = frames.size(); i > 0; i--) {
    auto& loc = frames[i - 1];
    // 查找不包含'site-packages'的文件，构建位置信息字符串返回
    if (loc.file.find("site-packages") == std::string::npos) {
      std::stringstream ss;
      ss << loc.file << " " << loc.function << " " << loc.line;
      return ss.str();
    }
  }
  return empty;
}

// 获取张量图信息的调试工具函数，返回图的字符串表示
std::string DebugUtil::GetTensorsGraphInfo(
    c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
    const std::vector<size_t>* indices,
    GraphFormat format) {
  std::vector<const torch::lazy::Node*> root_nodes;
  std::vector<torch::lazy::Value> root_values;
  std::vector<torch::lazy::hash_t> root_hashes;
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
  // 若索引非空，则处理索引
    for (auto index : *indices) {
      // 遍历 indices 指向的整数数组中的每个索引
      const torch::lazy::LazyTensorPtr& tensor = tensors[index];
      // 获取 tensors 数组中指定索引处的 LazyTensor 智能指针
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      // 获取当前 LazyTensor 的 IR 值
      if (ir_value) {
        // 如果 IR 值有效
        root_nodes.push_back(ir_value.node.get());
        // 将 IR 值的节点指针添加到 root_nodes 中
        root_hashes.push_back(ir_value.hash());
        // 将 IR 值的哈希值添加到 root_hashes 中
        root_values.push_back(std::move(ir_value));
        // 将 IR 值移动到 root_values 中
        unique_device.set(tensor->GetDevice());
        // 设置 unique_device 为当前 tensor 的设备
      }
    }
  } else {
    for (auto& tensor : tensors) {
      // 遍历 tensors 数组中的每个 LazyTensor 引用
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      // 获取当前 LazyTensor 的 IR 值
      if (ir_value) {
        // 如果 IR 值有效
        root_nodes.push_back(ir_value.node.get());
        // 将 IR 值的节点指针添加到 root_nodes 中
        root_hashes.push_back(ir_value.hash());
        // 将 IR 值的哈希值添加到 root_hashes 中
        root_values.push_back(std::move(ir_value));
        // 将 IR 值移动到 root_values 中
        unique_device.set(tensor->GetDevice());
        // 设置 unique_device 为当前 tensor 的设备
      }
    }
  }
  std::stringstream ss;
  // 创建一个字符串流 ss
  // 调用一个可能由 Python 支持或空的函数指针，取决于运行时情况
  std::vector<SourceLocation> frames = GetPythonFramesFunction()();
  // 调用 GetPythonFramesFunction 函数指针，获取 Python 栈帧信息
  ss << "Python Stacktrace:\n";
  // 将 Python 栈追踪信息写入 ss 字符串流
  for (auto& location : frames) {
    // 遍历 Python 栈帧信息
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
    // 将每个栈帧的函数名、文件和行号写入 ss 字符串流
  }
  ss << "\nHashes: (";
  // 将哈希信息的标识写入 ss 字符串流
  for (const auto i : c10::irange(root_hashes.size())) {
    // 遍历 root_hashes 大小范围内的索引
    if (i > 0) {
      ss << ", ";
    }
    ss << torch::lazy::HashToString(root_hashes[i]);
    // 将每个哈希值转换为字符串并写入 ss 字符串流
  }
  ss << ")\n";

  std::string graph_str;
  // 声明一个字符串 graph_str
  if (format == GraphFormat::kText) {
    // 如果请求的图格式是文本
    graph_str = torch::lazy::DumpUtil::ToText(root_nodes);
    // 使用 DumpUtil::ToText 将 root_nodes 转换为文本格式的图形描述并赋给 graph_str
  } else if (format == GraphFormat::kDot) {
    // 如果请求的图格式是 DOT 格式
    graph_str = torch::lazy::DumpUtil::ToDot(root_nodes);
    // 使用 DumpUtil::ToDot 将 root_nodes 转换为 DOT 格式的图形描述并赋给 graph_str
  } else if (format == GraphFormat::kBackend) {
    // 如果请求的图格式是后端格式
    graph_str = torch::lazy::DumpUtil::ToBackend(
        root_values,
        unique_device ? *unique_device : torch::lazy::BackendDevice());
    // 使用 DumpUtil::ToBackend 将 root_values 和 unique_device 转换为后端格式的图形描述并赋给 graph_str
  } else {
    LOG(ERROR) << "Invalid graph format: " << format;
    // 记录错误日志，指示无效的图形格式
  }
  ss << "\n## BEGIN_GRAPH\n" << graph_str << "\n## END_GRAPH\n\n";
  // 将图形描述字符串添加到 ss 字符串流中，包含起始和结束标记
  return ss.str();
  // 返回字符串流 ss 中的所有内容作为结果
}

// 实现 DebugUtil 类的 SaveTensorsGraphInfo 方法
void DebugUtil::SaveTensorsGraphInfo(
    const char* name, // 名称参数，用于标识保存的图信息
    c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors, // 张量数组的引用
    const std::vector<size_t>* indices, // 索引数组的指针，可选
    GraphFormat format) { // 图格式参数
  // 从环境变量中获取保存文件路径
  static const std::string save_file =
      GetEnvString("LTC_SAVE_TENSORS_FILE", "");
  // 如果保存文件路径不为空
  if (!save_file.empty()) {
    // 静态互斥锁，确保线程安全
    static std::mutex lock;
    // 获取张量图信息的字符串表示
    std::string info = GetTensorsGraphInfo(tensors, indices, format);
    // 加锁，保证文件写入的原子性
    std::lock_guard<std::mutex> guard(lock);
    // 以追加模式打开保存文件
    std::ofstream graph_file(save_file, std::ios_base::app);
    // 将名称和图信息写入文件
    graph_file << "[" << name << "]\n" << info << "\n";
  }
}

// 实现 DebugUtil 类的 ExperimentEnabled 方法
bool DebugUtil::ExperimentEnabled(const std::string& name) {
  // 加载实验名称集合的静态指针
  static const std::unordered_set<std::string>* xset = LoadExperiments();
  // 检查给定名称是否在实验集合中
  return xset->find(name) != xset->end();
}

} // namespace lazy
} // namespace torch


这段代码是 C++ 的类方法实现，主要包括了两个方法 `SaveTensorsGraphInfo` 和 `ExperimentEnabled`，分别用于保存张量图信息和检查实验是否启用。
```