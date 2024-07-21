# `.\pytorch\torch\csrc\profiler\standalone\execution_trace_observer.cpp`

```
#ifdef _WIN32
// 如果是在 Windows 下编译，确保 WIN32_LEAN_AND_MEAN 被定义，以减少包含的 Windows 头文件
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <processthreadsapi.h>  // 包含处理线程 API 的头文件
#else
#include <unistd.h>  // 如果不是在 Windows 下，包含 POSIX 系统的头文件
#endif // _WIN32

#include <fmt/format.h>  // 使用 fmt 库进行格式化输出
#include <chrono>  // 包含时间相关的头文件
#include <cmath>  // 包含数学函数的头文件
#include <fstream>  // 文件流操作相关头文件
#include <iomanip>  // 控制输出格式相关头文件
#include <map>  // 使用 map 容器
#include <mutex>  // 多线程互斥锁的头文件
#include <sstream>  // 字符串流的头文件
#include <stack>  // 栈的头文件
#include <vector>  // 向量的头文件

#include <ATen/core/TensorBody.h>  // 包含 ATen 张量核心结构的头文件
#include <ATen/core/function_schema.h>  // 包含函数模式的头文件
#include <ATen/core/stack.h>  // ATen 栈相关的头文件
#include <ATen/record_function.h>  // 记录函数调用的头文件
#include <c10/util/irange.h>  // 包含整数范围的头文件
#include <torch/csrc/profiler/standalone/execution_trace_observer.h>  // 执行跟踪观察者的头文件
#include <torch/csrc/profiler/util.h>  // PyTorch 用到的工具函数

#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>  // 使用分布式通信的头文件
#endif // USE_DISTRIBUTED

using namespace at;  // 使用 ATen 命名空间

// 分布式属性相关的常量
#ifdef USE_DISTRIBUTED
constexpr auto kETCommsName = "collective_name";  // 通信名
constexpr auto kETInMsgNelems = "in_msg_nelems";  // 输入消息元素数
constexpr auto kETOutMsgNelems = "out_msg_nelems";  // 输出消息元素数
constexpr auto kETInSplit = "in_split_size";  // 输入分割大小
constexpr auto kETOutSplit = "out_split_size";  // 输出分割大小
constexpr auto kETGlobalRankStart = "global_rank_start";  // 全局排名起始
constexpr auto kETGlobalRankStride = "global_rank_stride";  // 全局排名步长
constexpr auto kETGroupSize = "pg_size";  // 进程组大小
constexpr auto kETProcessGroupName = "pg_name";  // 进程组名称
constexpr auto kETProcessGroupDesc = "pg_desc";  // 进程组描述
#endif // USE_DISTRIBUTED

namespace torch::profiler::impl {

//******************************************************************************
// JSON 输出工具函数，待与 PyTorch profiler 合并
//******************************************************************************

// 将向量转换为字符串表示
template <typename T>
inline std::string vectorToString(const std::vector<T>& v) {
  return fmt::format("[{}]", fmt::join(v, ","));
}

// 对 JSON 字符串进行转义处理
std::string json_str_escape(const std::string& str);

// 值类型的获取函数，用于分析 IValue 类型的对象
constexpr size_t kMaxNumElements = 4096;  // 最大元素数
constexpr size_t kMaxStrLength = 8192;  // 最大字符串长度

// 获取值类型的函数，根据不同的类型进行处理
inline std::string getValueType(
    const c10::IValue& val,
    const bool baseType = true,
    const size_t maxArrayLen = kMaxNumElements) {
  std::string type = val.tagKind();  // 获取对象的类型标签

  if (val.isTensor()) {
    // 如果是 Tensor 类型，添加张量元素的数据类型信息
    type += fmt::format("({})", std::string(val.toTensor().dtype().name()));
  } else if (val.isTuple()) {
    // 如果是元组类型，获取其中元素的类型信息
    const auto& val_container = val.toTupleRef().elements();
    std::vector<std::string> str_array;
    for (const auto& t : val_container) {
      str_array.emplace_back(getValueType(t, false));
    }
    type += vectorToString(str_array);  // 转换为字符串表示
  } else if (val.isList()) {
    // 如果是列表类型，获取其中元素的类型信息
    const auto& val_list = val.toList();
    std::vector<std::string> str_array;
    str_array.reserve(val_list.size());
    for (const auto j : c10::irange(val_list.size())) {
      str_array.push_back(getValueType(val_list.get(j), false));
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;  // 超出最大长度的警告信息
        break;
      }
    }
    type += vectorToString(str_array);  // 转换为字符串表示
  }
    type += vectorToString(str_array);
  }
  return baseType ? fmt::format("\"{}\"", type) : type;



// 将字符串数组转换为字符串并追加到 type 变量末尾
type += vectorToString(str_array);
// 如果 baseType 为真，使用 fmt 格式化字符串，将 type 包裹在双引号中返回；否则直接返回 type
return baseType ? fmt::format("\"{}\"", type) : type;
}

// 获取值的形状信息并返回其字符串表示
inline std::string getValueShape(
    const c10::IValue& val,
    const size_t maxArrayLen = kMaxNumElements) {
  if (val.isTensor()) {
    auto& tensor = val.toTensor();
    // 如果张量被定义且没有符号大小或步长，返回其尺寸信息的字符串表示
    if (tensor.defined() &&
        !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
      return vectorToString(tensor.sizes().vec());
    }
  } else if (val.isTuple()) {
    // 如果值是元组，递归获取元组元素的形状信息并返回其字符串表示
    const auto& val_container = val.toTupleRef().elements();
    std::vector<std::string> str_array;
    for (const auto& t : val_container) {
      str_array.push_back(getValueShape(t));
    }
    return vectorToString(str_array);
  } else if (val.isList()) {
    // 如果值是列表，获取列表元素的形状信息并返回其字符串表示
    const auto& val_list = val.toList();
    std::vector<std::string> str_array;
    str_array.reserve(val_list.size());
    for (const auto j : c10::irange(val_list.size())) {
      str_array.push_back(getValueShape(val_list.get(j)));
      // 如果超过最大数组长度，记录警告信息并终止
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    return vectorToString(str_array);
  }
  // 对于其他类型的值，返回空列表表示
  return "[]";
}

// 获取标量值的字符串表示
inline std::string getScalarValue(const c10::IValue& val) {
  if (val.isDouble()) {
    double d_val = val.toDouble();
    // 如果是双精度浮点数，格式化成字符串表示，处理特殊值
    if (std::isinf(d_val) || std::isnan(d_val)) {
      return fmt::format("\"{}\"", std::to_string(d_val));
    } else {
      return std::to_string(d_val);
    }
  } else if (val.isInt()) {
    // 如果是整数，转换成字符串表示
    return std::to_string(val.toInt());
  } else if (val.isBool()) {
    // 如果是布尔值，转换成 "true" 或 "false"
    return val.toBool() ? "true" : "false";
  } else if (val.isString()) {
    // 如果是字符串，处理长度超出限制的情况，并返回转义后的 JSON 字符串
    const std::string& str_val = val.toStringRef();
    if (str_val.size() > kMaxStrLength) {
      LOG(WARNING) << "string size=" << str_val.size()
                   << " exceeded kMaxStrLength=" << kMaxStrLength;
      return fmt::format(
          "\"{}\"", json_str_escape(str_val.substr(0, kMaxStrLength)));
    }

    return fmt::format("\"{}\"", json_str_escape(str_val));
  } else if (val.isDevice()) {
    // 如果是设备类型，返回设备的字符串表示
    return fmt::format("\"{}\"", val.toDevice().str());
  }
  // 对于其他类型，返回类型标签的字符串表示
  return fmt::format("\"<{}>\"", val.tagKind());
}

// 获取当前进程的进程 ID
inline int32_t processId() {
#ifndef _WIN32
  return static_cast<int32_t>(getpid());
#else
  return static_cast<int32_t>(GetCurrentProcessId());
#endif
}

//******************************************************************************
// Main ExecutionTraceObserver implementation.
//******************************************************************************

// ExecutionTraceObserver 实现了观察者的所有状态。一些状态在进入和退出 RecordFunction
// 回调时共享，如 `opStack`。某些数据可能跨不同线程访问，因此需要注意数据竞争。
// 使用全局互斥锁 `gMutex` 来避免这些竞争，但这可能在大量线程情况下影响性能。
// 可以进一步优化为线程局部锁或细粒度锁，或使用线程安全的容器。
struct TORCH_API ExecutionTraceObserver { // NOLINT
  using ID = size_t;

  // Mapping of each thread to its own operator stack
  std::map<size_t, std::stack<ID>> opStack{};
  // Uses the underlying TensorImpl object pointer as the key and map to its
  // unique id.
  std::map<const void*, ID> objectId{};
  // Observer run state.
  enum class RunState { uninitialized, disabled, enabled };

  // Mutex for multithreaded access to the shared containers.
  std::recursive_mutex gMutex{};
  // Stream to write output JSON.
  std::ofstream out{};

  // Full path to the output file.
  std::string fileName{};

  // RecordFunction callback handle for this observer.
  CallbackHandle cbHandle{INVALID_CALLBACK_HANDLE};

  // Process ID.
  int32_t pid{-1};
  std::string recordTime{};

  // Default constructor
  ExecutionTraceObserver() = default;

  // Returns a new unique ID.
  ID getNewID() {
    return id_++;
  }

  // Returns the current state of the observer.
  RunState getState() const {
    return state_;
  }

  // Sets the state of the observer, enabling or disabling callback as needed.
  void setState(RunState newState) {
    if (state_ == RunState::uninitialized ||
        callbackShouldBeEnabled(state_) != callbackShouldBeEnabled(newState)) {
      if (callbackShouldBeEnabled(newState)) {
        reenableCallback(cbHandle);
      } else {
        disableCallback(cbHandle);
      }
    }
    state_ = newState;
  }

 private:
  // Determines if the callback should be enabled based on the current state.
  static bool callbackShouldBeEnabled(RunState run_state) {
    return run_state == ExecutionTraceObserver::RunState::enabled;
  }

  // Private member to store the current state of the observer.
  // Must use accessors to change this so that we can keep the
  // RecordFunction callback in sync with the state.
  RunState state_{RunState::uninitialized};

  // Atomic counter for generating unique IDs for tensors and operators.
  // 0 -> unintialized
  // 1 -> root ID
  // 2 ... -> regular node ID
  std::atomic<ID> id_{2};
};

// Using a singleton manager here to allow init and delete the observer object.
using ObserverManager = GlobalStateManager<ExecutionTraceObserver>;

// Uninitialized node has id = 0
const ExecutionTraceObserver::ID kUninitializedId{0};
// Root node has id = 1
const ExecutionTraceObserver::ID kRootId{1};

struct FunctionCallContext : public ObserverContext { // NOLINT
  std::string name;
  std::string kernelBackend;
  std::string kernelFile;
  ExecutionTraceObserver::ID opId{kUninitializedId};
  ExecutionTraceObserver::ID parentId{kUninitializedId};
  ExecutionTraceObserver::ID fwParentId{kUninitializedId};
  std::vector<std::string> inputTypes;
  std::vector<std::string> inputShapes;
  std::vector<std::string> inputValues;
};

// Opens the json file to write the execution trace.
static std::ofstream openOutputFile(const std::string& name) {
  std::ofstream stream;
  stream.open(name, std::ofstream::out | std::ofstream::trunc);
  if (!stream) {
    LOG(ERROR) << "Failed to open '" << name << "'";
  } else {
    VLOG(1) << "PyTorch Execution Trace: writing to " << name;
  }
  return stream;
}

#ifdef USE_DISTRIBUTED
static inline std::string getAttrJson(
    // 定义一个函数，接受三个参数：name、type 和 value，生成一个 JSON 格式的字符串
    return fmt::format(
        // 使用原始字符串字面量定义 JSON 格式的模板，包含三个字段：name、type、value
        R"JSON(
        {{"name": "{}", "type": "{}", "value": {}}}
        )JSON",
        // 使用 fmt 库的 format 函数填充 JSON 模板的字段值，name 和 type 直接填入，value 根据类型不同可能需要额外处理
        name,  // name 字段的值
        type,  // type 字段的值
        value);  // value 字段的值，如果是字符串则需要用双引号包裹，其他类型如数字直接填入
// 结束 C++ 的条件编译指令，该段代码在非DEBUG模式下生效
#endif

// 定义一个静态函数 writeJsonNode，用于将节点信息以 JSON 格式写入文件流
static void writeJsonNode(
    std::ofstream& out,                   // 输出流引用，用于写入 JSON 数据
    const std::string& name,              // 节点名称
    const uint64_t id,                    // 节点 ID
    const uint64_t rf_id,                 // rf_id
    const uint64_t parent,                // 父节点 ID
    const uint64_t fw_parent,             // fw_parent
    const int64_t seq_id,                 // 序列 ID
    const uint64_t scope,                 // 节点作用域
    const uint64_t tid,                   // tid
    const uint64_t fw_tid,                // fw_tid
    const std::string& inputs = "[]",     // 输入数据，默认为空数组
    const std::string& inputShapes = "[]",// 输入数据形状，默认为空数组
    const std::string& inputTypes = "[]", // 输入数据类型，默认为空数组
    const std::string& outputs = "[]",    // 输出数据，默认为空数组
    const std::string& output_shapes = "[]",// 输出数据形状，默认为空数组
    const std::string& output_types = "[]",// 输出数据类型，默认为空数组
    const std::string& operator_schema = "", // 运算符 schema，默认为空字符串
    const std::string& kernelBackend = "",   // 内核后端，默认为空字符串
    const std::string& kernelFile = "",      // 内核文件，默认为空字符串
    const std::string& additiona_attrs = "") {// 附加属性，默认为空字符串
{
  // 将格式化的 JSON 字符串输出到文件流中
  out << fmt::format(
      R"JSON(
    {{
      "id": {}, "name": "{}", "ctrl_deps": {},
      "inputs": {{"values": {}, "shapes": {}, "types": {}}},
      "outputs": {{"values": {}, "shapes": {}, "types": {}}},
      "attrs": [{{"name": "rf_id", "type": "uint64", "value": {}}},{{"name": "fw_parent", "type": "uint64", "value": {}}},{{"name": "seq_id", "type": "int64", "value": {}}},{{"name": "scope", "type": "uint64", "value": {}}},{{"name": "tid", "type": "uint64", "value": {}}},{{"name": "fw_tid", "type": "uint64", "value": {}}},{{"name": "op_schema", "type": "string", "value": "{}"}},{{"name": "kernel_backend", "type": "string", "value": "{}"}},{{"name": "kernel_file", "type": "string", "value": "{}"}}{}]
    }})JSON",
      id,                               // 节点 ID
      name,                             // 节点名称
      parent,                           // 父节点 ID
      inputs,                           // 输入数据
      inputShapes,                      // 输入数据形状
      inputTypes,                       // 输入数据类型
      outputs,                          // 输出数据
      output_shapes,                    // 输出数据形状
      output_types,                     // 输出数据类型
      rf_id,                            // rf_id
      fw_parent,                        // fw_parent
      seq_id,                           // 序列 ID
      scope,                            // 节点作用域
      tid,                              // tid
      fw_tid,                           // fw_tid
      operator_schema,                  // 运算符 schema
      kernelBackend,                    // 内核后端
      kernelFile,                       // 内核文件
      additiona_attrs);                 // 附加属性
}

// 写入时间戳格式化字符串，返回字符串表示的时间
inline std::string timeString(const std::time_t timepoint) {
  std::ostringstream oss;
  // 将时间格式化为 YYYY-MM-DD HH:MM:SS 格式，存入字符串流
  oss << std::put_time(std::localtime(&timepoint), "%Y-%m-%d %X"); // NOLINT
  return oss.str(); // 返回格式化后的时间字符串
}

// 初始化执行追踪的开始，设置输出文件并记录起始时间等信息
static bool initExecutionTraceStart(ExecutionTraceObserver& ob) {
  ob.out = openOutputFile(ob.fileName); // 打开输出文件流
  // 如果文件流打开失败，记录警告并返回 false
  if (!ob.out) {
    LOG(WARNING) << "Failed to open output file: " << ob.fileName;
    return false;
  }

  // 获取当前的系统时间点
  const auto current_time = std::chrono::system_clock::now();
  // 格式化并记录当前时间点的时间字符串
  ob.recordTime =
      timeString(std::chrono::system_clock::to_time_t(current_time));
  // 使用 steady_clock 获取当前时间戳（毫秒级），作为起始时间戳
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();

  // 将执行追踪的起始信息以 JSON 格式写入文件流
  ob.out << fmt::format(
      R"JSON({{
  "schema": "1.1.0-chakra.0.0.4", "pid": {}, "time": "{}", "start_ts": {},
  "nodes": [)JSON",
      ob.pid,                           // 进程 ID
      ob.recordTime,                    // 记录时间
      timestamp);                       // 起始时间戳
  return true;                          // 返回初始化成功
}

// 将执行追踪信息写入文件
static void finalizeExecutionTraceOutput(ExecutionTraceObserver& ob) {
  // 写入 JSON 节点信息到输出流
  writeJsonNode(
      ob.out,
      "[pytorch|profiler|execution_trace|process]", // JSON节点路径
      kRootId, // 节点ID
      0, // rf_id
      kRootId, // 父节点ID为自身
      0, // fw_parent
      -1, // 序列ID
      static_cast<std::underlying_type_t<RecordScope>>(RecordScope::USER_SCOPE), // 记录作用域
      0, // 线程ID
      0); // 前向线程ID

  // 使用 steady_clock 获取当前时间戳（毫秒）
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  // 格式化输出结束时间戳到输出流中的 JSON 格式字符串
  ob.out << fmt::format(
      R"JSON(
  ],
  "finish_ts": {}
}})JSON",
      timestamp);

  // 关闭输出流
  ob.out.close();
  // 记录输出文件名到日志
  VLOG(1) << "PyTorch Execution Trace: written to file " << ob.fileName;
}

inline ExecutionTraceObserver::ID getObjectID(
    ExecutionTraceObserver& ob,
    const void* t) {
  // 查找对象 ID 是否已存在于映射中
  auto iter = ob.objectId.find(t);
  // 若不存在，分配新的对象 ID 并将其映射到给定对象指针
  if (iter == ob.objectId.end()) {
    ExecutionTraceObserver::ID objectId = ob.getNewID();
    ob.objectId[t] = objectId;
    return objectId;
  }

  // 若存在，返回已映射的对象 ID
  return iter->second;
}

inline std::string convertIValue(
    ExecutionTraceObserver& ob,
    const c10::IValue& val,
    const size_t maxArrayLen = kMaxNumElements) {
  // 如果值是 Tensor 类型
  if (val.isTensor()) {
    const auto t = val.toTensor().unsafeGetTensorImpl();
    // 获取 Tensor 对象和其存储的对象 ID
    ExecutionTraceObserver::ID tensor_id = getObjectID(ob, t);
    ExecutionTraceObserver::ID storage_id = 0;
    size_t offset = 0;
    size_t numel = 0;
    size_t itemsize = 0;
    std::string device_str = "";
    // 如果 Tensor 对象有存储，并且没有符号大小/步长，则获取相关信息
    if (t->has_storage() && !t->has_symbolic_sizes_strides()) {
      auto& t_storage = t->storage();
      storage_id = getObjectID(ob, t_storage.data());
      offset = t->storage_offset();
      numel = t->numel();
      itemsize = t->itemsize();
      device_str = t->device().str();
    }
    // 格式化输出 Tensor 对象的信息到字符串
    return fmt::format(
        "[{},{},{},{},{},\"{}\"]",
        tensor_id,
        storage_id,
        offset,
        numel,
        itemsize,
        device_str);
  } else if (val.isTuple()) {
    // 如果值是 Tuple 类型，则递归转换每个元素的值并组成字符串数组
    std::vector<std::string> str_array;
    const auto& val_tuple = val.toTupleRef().elements();
    for (const auto j : c10::irange(val_tuple.size())) {
      str_array.push_back(convertIValue(ob, val_tuple[j]));
    }
    // 将字符串数组转换成一个字符串表示
    return vectorToString(str_array);
  } else if (val.isList()) {
    // 如果值是 List 类型
    const auto& val_list = val.toList();
    std::vector<std::string> str_array;
    str_array.reserve(val_list.size());
    // 遍历 List 的每个元素，转换成字符串并存储在字符串数组中
    for (const auto j : c10::irange(val_list.size())) {
      str_array.push_back(convertIValue(ob, val_list.get(j)));
      // 如果超过最大数组长度，记录警告信息
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    // 将字符串数组转换成一个字符串表示
    return vectorToString(str_array);
  } else {
    // 对于其他类型的值，返回其标量值的字符串表示
    return getScalarValue(val);
  }
}
    // 将转换后的值添加到 `values` 向量中
    values.push_back(convertIValue(ob, val));
    // 获取值的类型并添加到 `types` 向量中
    types.push_back(getValueType(val));
    // 获取值的形状并添加到 `shapes` 向量中
    shapes.push_back(getValueShape(val));
// 处理内核后端信息的函数，用于设置函数调用上下文和记录函数对象
inline void handleKernelBackendInfo(
    FunctionCallContext& fc,                       // 函数调用上下文引用
    const RecordFunction& fn) {                    // 记录函数对象引用
  // Triton 内核相关信息存储在 kwinputs 中
  const auto& kwinputs = fn.kwinputs();
  // 检查是否包含 kernel_backend
  if (kwinputs.find("kernel_backend") != kwinputs.end()) {
    // 将 kernel_backend 的值设置为函数调用上下文的 kernelBackend 字段
    fc.kernelBackend = kwinputs.at("kernel_backend").toStringRef();
    // 如果 kernelBackend 是 "triton"
    if (fc.kernelBackend == "triton") {
      // 将 kernel_file 的值设置为函数调用上下文的 kernelFile 字段
      fc.kernelFile = kwinputs.at("kernel_file").toStringRef();
      // 断言 kernel_file 在 Triton 内核中必须存在
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("kernel_file") != kwinputs.end(),
          "kernel file is missing in triton kernel");
      // 从 kernelFile 中移除文件名的路径部分
      if (fc.kernelFile.find_last_of('/') != std::string::npos) {
        fc.kernelFile =
            fc.kernelFile.substr(fc.kernelFile.find_last_of('/') + 1);
      }

      // 获取 grid 信息
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("grid") != kwinputs.end(),
          "grid is missing in triton kernel");
      // 将 grid 的值添加到 inputValues，类型为 String
      fc.inputValues.emplace_back(
          "\"" + kwinputs.at("grid").toStringRef() + "\"");
      fc.inputTypes.emplace_back("\"String\"");
      fc.inputShapes.emplace_back("[]");

      // 获取 stream 信息
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("stream") != kwinputs.end(),
          "stream is missing in triton kernel");
      // 将 stream 的整数值转换为字符串，添加到 inputValues，类型为 Int
      fc.inputValues.emplace_back(
          std::to_string(kwinputs.at("stream").toInt()));
      fc.inputTypes.emplace_back("\"Int\"");
      fc.inputShapes.emplace_back("[]");
    }
  }
}

// 获取通信集合的额外属性字符串，用于记录函数对象
inline std::string getCommsNodeAttrs(const RecordFunction& fn) { // NOLINT
  // 属性向量，用于存储属性字符串
  std::vector<std::string> attrs;

#ifdef USE_DISTRIBUTED
  // 我们依赖于线程本地信息中可用的 paramcommsdebug 对象
  auto debugInfo = dynamic_cast<ParamCommsDebugInfo*>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PARAM_COMMS_INFO));
  // 如果 debugInfo 为空
  if (debugInfo == nullptr) {
    // 记录警告信息，指出函数名称中 ParamCommsDebugInfo 不可用
    LOG(WARNING) << "ParamCommsDebugInfo not available for function: "
                 << fn.name();
  // 返回一个字符串，包含 ", " 和调用 getAttrJson 函数返回的调试信息，若未找到通讯信息则返回默认字符串
  return ", " + getAttrJson("debug", "string", "\"missing comms info\"");
}

// 从记录函数中获取 NcclMeta，使用了上面定义的 ParamCommsDebugInfo
auto meta = saveNcclMeta(fn, false /*truncate*/);

// 定义一个 lambda 函数 addAttr，用于添加属性到 attrs 向量中
auto addAttr =
    [&](const char* commsMetaName, const char* etMetaName, const char* type) {
      // 在 meta 中查找 commsMetaName 对应的值
      auto it = meta.find(commsMetaName);
      // 如果找到了，则将属性添加到 attrs 向量中
      if (it != meta.end()) {
        attrs.push_back(getAttrJson(etMetaName, type, it->second));
      }
    };

// 添加通讯元信息 kCommsName 到 attrs 中，类型为 "string"
addAttr(kCommsName, kETCommsName, "string");
// 添加数据类型元信息 kDtype 到 attrs 中，类型为 "string"
addAttr(kDtype, kDtype, "string");

// 添加输入消息元素数 kInMsgNelems 到 attrs 中，类型为 "uint64"
addAttr(kInMsgNelems, kETInMsgNelems, "uint64");
// 添加输出消息元素数 kOutMsgNelems 到 attrs 中，类型为 "uint64"
addAttr(kOutMsgNelems, kETOutMsgNelems, "uint64");

// 下面两个元数据是列表类型，添加输入分割元信息 kInSplit 到 attrs 中，类型为 "string"
addAttr(kInSplit, kETInSplit, "string");
// 添加输出分割元信息 kOutSplit 到 attrs 中，类型为 "string"
addAttr(kOutSplit, kETOutSplit, "string");

// 添加全局排名起始元信息 kGlobalRankStart 到 attrs 中，类型为 "uint64"
addAttr(kGlobalRankStart, kETGlobalRankStart, "uint64");
// 添加全局排名步幅元信息 kGlobalRankStride 到 attrs 中，类型为 "uint64"
addAttr(kGlobalRankStride, kETGlobalRankStride, "uint64");

// pg_name 是一个字符串，添加处理组名称元信息 kProcessGroupName 到 attrs 中，类型为 "string"
addAttr(kProcessGroupName, kETProcessGroupName, "string");
// 添加处理组描述元信息 kProcessGroupDesc 到 attrs 中，类型为 "string"
addAttr(kProcessGroupDesc, kETProcessGroupDesc, "string");

// 添加组大小元信息 kGroupSize 到 attrs 中，类型为 "uint64"
addAttr(kGroupSize, kETGroupSize, "uint64");
#ifdef USE_DISTRIBUTED
// 如果定义了 USE_DISTRIBUTED 宏，则编译以下代码块

  // XXX consider using as string stream?
  // 考虑使用字符串流？
  return attrs.empty() ? "" : fmt::format(", {}", fmt::join(attrs, ", "));
  // 如果 attrs 为空，则返回空字符串；否则将 attrs 中的元素以逗号分隔并格式化为字符串后返回

}

static void recordOperatorStart(
    ExecutionTraceObserver& ob,
    FunctionCallContext& fc,
    const RecordFunction& fn) {
  // 记录操作开始的静态函数，接收执行跟踪观察器、函数调用上下文和记录函数作为参数
  auto tid = fn.threadId();
  // 获取函数调用的线程 ID

  try {
    const std::lock_guard<std::recursive_mutex> lock(ob.gMutex);
    // 使用递归互斥锁保护执行跟踪观察器的全局互斥锁

    // if current thread stack is empty, push the root node to the stack first
    // 如果当前线程的操作栈为空，则先将根节点推入栈中
    if (ob.opStack[tid].empty()) {
      auto thread_node_id = ob.getNewID();
      // 获取新的节点 ID
      ob.opStack[tid].push(thread_node_id);
      // 将新节点 ID 推入当前线程的操作栈
      writeJsonNode(
          ob.out,
          "[pytorch|profiler|execution_trace|thread]",
          thread_node_id,
          0, // rf_id
          kRootId,
          0, // fw_parent
          -1, // seq_id
          static_cast<std::underlying_type_t<RecordScope>>(
              RecordScope::USER_SCOPE),
          tid,
          0); // fw_tid
      // 将线程节点的 JSON 写入输出流
      ob.out << ",";
      // 在输出流中添加逗号分隔符
    }
    fc.name = fn.name();
    // 设置函数调用上下文的函数名为记录函数的函数名
    auto num_inputs = fn.num_inputs();
    // 获取记录函数的输入数量
    const auto inputs = fn.inputs();
    // 获取记录函数的输入向量

    VLOG(2) << "inputs: " << num_inputs << " " << inputs.size() << '\n';
    // 记录日志：输入数量和输入向量的大小

    // We have two cases: for unboxed kernel, we have num_inputs ==
    // inputs.size() for boxed kernel using stack, there could be more elements
    // on the stack from previous ops.
    // 我们有两种情况：对于未装箱的内核，num_inputs == inputs.size()；
    // 对于使用堆栈的装箱内核，堆栈中可能有来自前面操作的更多元素。
    // TORCH_INTERNAL_ASSERT(num_inputs <= inputs.size());
    // 内部断言：确保输入数量不大于输入向量的大小
    if (num_inputs > inputs.size()) {
      LOG(WARNING) << "RecordFunction " << fc.name
                   << " expected num_inputs=" << num_inputs
                   << " > inputs.size()=" << inputs.size();
      // 记录警告日志：记录函数预期的输入数量大于实际的输入向量大小
      return;
      // 如果条件不满足，直接返回
    }
    // need to account for Stack mode where the inputs are at the end.
    // 需要考虑堆栈模式，其中输入位于末尾。
    size_t input_start = inputs.size() - num_inputs;
    // 计算输入向量的起始位置

    for (const auto i : c10::irange(input_start, inputs.size())) {
      appendValueInfo(
          ob, inputs[i], fc.inputValues, fc.inputTypes, fc.inputShapes);
      // 对输入向量的每个元素，追加值信息到函数调用上下文中的相应属性
    }

    handleKernelBackendInfo(fc, fn);
    // 处理内核后端信息

    fc.parentId = ob.opStack[tid].top();
    // 设置函数调用上下文的父节点 ID 为当前线程操作栈顶部的节点 ID
    // get parent id from the forward stack, this can be different for
    // autograd ops, which may execute on a different thread than the original
    // thread (which should have the parent op on the stack).
    // 从前向栈中获取父节点 ID，这对于自动求导操作可能不同，因为它们可能在不同的线程上执行

    auto fw_tid = fn.forwardThreadId();
    // 获取前向线程 ID
    if (fw_tid != 0) {
      fc.fwParentId = ob.opStack[fw_tid].top();
      // 如果前向线程 ID 不为零，则设置函数调用上下文的前向父节点 ID 为前向线程的操作栈顶部节点 ID
    }
    // all input nodes should have id > opId
    // 所有输入节点的 ID 应大于操作 ID
    fc.opId = ob.getNewID();
    // 设置函数调用上下文的操作 ID 为新的节点 ID
    ob.opStack[tid].push(fc.opId);
    // 将操作 ID 推入当前线程的操作栈

  } catch (const std::exception& e) {
    LOG(WARNING) << "Exception in execution trace observer: " << e.what();
    // 捕获到异常时记录警告日志
  }
}

static std::unique_ptr<ObserverContext> onFunctionEnter(
    const RecordFunction& fn) {
  // 当进入函数时触发的函数，接收记录函数作为参数
  using RunState = ExecutionTraceObserver::RunState;
  auto ob = ObserverManager::get();
  // 获取观察器管理器实例

  if (ob != nullptr && ob->getState() == RunState::enabled) {
    // 如果观察器管理器实例不为空且其状态为 enabled
    // record op
    // 记录操作
    auto fc_ptr = std::make_unique<FunctionCallContext>();
    // 创建函数调用上下文的唯一指针
    recordOperatorStart(*ob, *fc_ptr.get(), fn);
    // 调用记录操作开始函数，传入观察器、函数调用上下文和记录函数
    return fc_ptr;
    // 返回函数调用上下文的唯一指针
  }
  return nullptr;
  // 如果观察器管理器实例为空或其状态不为 enabled，则返回空指针
}
// 定义一个函数，用于将 JSON 字符串中的特殊字符进行转义处理
inline std::string json_str_escape(const std::string& str) {
  // 创建一个字符串输出流
  std::ostringstream ostream;
  // 遍历输入字符串的每个字符
  for (char ch : str) {
    // 如果是双引号，则转义为 \"
    if (ch == '"') {
      ostream << "\\\"";
    // 如果是反斜杠，则转义为 \\
    } else if (ch == '\\') {
      ostream << "\\\\";
    // 如果是退格符，则转义为 \b
    } else if (ch == '\b') {
      ostream << "\\b";
    // 如果是换页符，则转义为 \f
    } else if (ch == '\f') {
      ostream << "\\f";
    // 如果是换行符，则转义为 \n
    } else if (ch == '\n') {
      ostream << "\\n";
    // 如果是回车符，则转义为 \r
    } else if (ch == '\r') {
      ostream << "\\r";
    // 如果是制表符，则转义为 \t
    } else if (ch == '\t') {
      ostream << "\\t";
    // 如果是控制字符（0x00-0x1f），则转义为 \u 后跟四位十六进制数
    } else if ('\x00' <= ch && ch <= '\x1f') {
      ostream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch);
    // 其他情况直接输出字符
    } else {
      ostream << ch;
    }
  }
  // 返回转义处理后的字符串
  return ostream.str();
}

// 定义一个静态函数，用于在函数退出时记录执行轨迹
static void onFunctionExit(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  // 定义一个枚举类型别名，表示执行轨迹观察器的运行状态
  using RunState = ExecutionTraceObserver::RunState;
  // 获取观察器管理器的实例
  auto ob = ObserverManager::get();
  // 如果观察器管理器为空指针或上下文指针为空，则直接返回
  if (ob == nullptr || ctx_ptr == nullptr) {
    return;
  }
  // 如果观察器的状态为 enabled
  if (ob->getState() == RunState::enabled) {
    // 尝试将上下文指针转换为函数调用上下文指针
    auto fc_ptr = dynamic_cast<FunctionCallContext*>(ctx_ptr);
    // 如果转换失败（指针为空），记录警告并返回
    if (fc_ptr == nullptr) {
      LOG(WARNING) << "FunctionCallContext is nullptr.";
      return;
    }
    // 获取函数调用上下文的引用
    auto& fc = *fc_ptr;

    // 获取记录函数的输出
    auto outputs = fn.outputs();
    // 获取记录函数的输出数量
    auto num_outputs = fn.num_outputs();
    // 记录输出的数量与实际输出的数量可能不一致，记录日志
    VLOG(2) << "outputs: " << num_outputs << " " << outputs.size() << '\n';
    // 如果记录的输出数量大于实际输出的数量，记录警告并返回
    if (num_outputs > outputs.size()) {
      LOG(WARNING) << "RecordFunction " << fc.name
                   << " num_outputs=" << num_outputs
                   << " > outputs.size()=" << outputs.size();
      return;
    }
    // 计算输出开始的索引，用于处理栈模式下输出位于结尾的情况
    size_t output_start = outputs.size() - num_outputs;

    // 定义存储输出类型、形状和值的字符串向量
    std::vector<std::string> output_types;
    std::vector<std::string> output_shapes;
    std::vector<std::string> output_values;
    // 使用 std::lock_guard 对象 ob->gMutex 进行锁定，确保线程安全
    try {
      const std::lock_guard<std::recursive_mutex> lock(ob->gMutex);
      // 从操作堆栈中移除当前操作的 ID

      ob->opStack[fn.threadId()].pop();
      // 遍历输出向量中的元素，对每个元素调用 appendValueInfo 函数处理
      for (const auto i : c10::irange(output_start, outputs.size())) {
        appendValueInfo(
            *ob, outputs[i], output_values, output_types, output_shapes);
      }

      // 初始化操作模式的 JSON 字符串
      std::string op_schema_str{};
      // 获取操作的模式信息，如果有值则转换为字符串并进行 JSON 转义
      const auto op_schema = fn.operator_schema();
      if (op_schema.has_value()) {
        op_schema_str = json_str_escape(c10::toString(op_schema.value()));
      }

      // 根据 fn 是否为 NcclMeta，决定是否获取额外的节点属性
      const std::string additiona_attrs =
          fn.isNcclMeta() ? getCommsNodeAttrs(fn) : "";

      // 将节点信息写入 JSON 输出流 ob->out
      writeJsonNode(
          ob->out,
          fc.name,
          fc.opId,
          fn.handle(),
          fc.parentId,
          fc.fwParentId,
          fn.seqNr(),
          static_cast<std::underlying_type_t<RecordScope>>(fn.scope()),
          fn.threadId(),
          fn.forwardThreadId(),
          vectorToString(fc.inputValues),
          vectorToString(fc.inputShapes),
          vectorToString(fc.inputTypes),
          vectorToString(output_values),
          vectorToString(output_shapes),
          vectorToString(output_types),
          op_schema_str,
          fc.kernelBackend,
          fc.kernelFile,
          additiona_attrs);
      // 在 JSON 流中添加逗号，用于分隔多个节点信息
      ob->out << ",";
    } catch (const std::exception& e) {
      // 捕获并记录执行追踪观察器中的异常信息
      LOG(WARNING) << "Exception in execution trace observer: [" << fc.name
                   << " (" << fc.opId << ")] " << e.what();
    }
  }
}

// 添加执行跟踪观察者回调函数到 RecordFunction 全局观察者。
bool addExecutionTraceObserver(const std::string& output_file_path) {
  // 检查观察者是否已经初始化。
  if (ObserverManager::get() == nullptr) {
    // 将 ExecutionTraceObserver 添加到观察者管理器中
    ObserverManager::push(std::make_shared<ExecutionTraceObserver>());
    auto& ob = *ObserverManager::get();
    // 设置进程 ID
    ob.pid = processId();
    // 设置输出文件名
    ob.fileName = output_file_path;
    // 初始化执行跟踪
    if (!initExecutionTraceStart(ob)) {
      return false;
    }

    // 添加全局回调函数，指定进入和退出函数时记录输入、输出和标识
    ob.cbHandle = addGlobalCallback(
        RecordFunctionCallback(&onFunctionEnter, &onFunctionExit)
            .needsInputs(true)
            .needsOutputs(true)
            .needsIds(true));
    // 默认禁用状态
    ob.setState(ExecutionTraceObserver::RunState::disabled);

    // 输出日志，表示已添加执行跟踪观察者
    VLOG(1) << "PyTorch Execution Trace: added observer, output="
            << output_file_path;
  } else if (ObserverManager::get()->cbHandle != INVALID_CALLBACK_HANDLE) {
    // 如果观察者已注册，则输出警告
    LOG(WARNING) << "Execution trace observer is already registered.";
  }
  return true;
}

// 移除执行跟踪观察者
void removeExecutionTraceObserver() {
  auto ob = ObserverManager::get();
  if (ob != nullptr) {
    // 如果观察者状态不是禁用，则先禁用执行跟踪
    if (ob->getState() != ExecutionTraceObserver::RunState::disabled) {
      disableExecutionTraceObserver();
    }

    // 如果回调句柄有效，则进行执行跟踪输出的最终化
    if (ob->cbHandle != INVALID_CALLBACK_HANDLE) {
      finalizeExecutionTraceOutput(*ob);
      // 移除回调函数
      removeCallback(ob->cbHandle);
      ob->cbHandle = INVALID_CALLBACK_HANDLE;
      // 弹出观察者对象，并重置全局状态
      TORCH_INTERNAL_ASSERT(
          ObserverManager::pop() != nullptr,
          "Global state ptr cannot be null before resetting");
      // 输出日志，表示已移除执行跟踪观察者
      VLOG(1) << "PyTorch Execution Trace: removed observer";
    } else {
      // 如果回调句柄无效，则输出警告
      LOG(WARNING) << "Execution trace observer was not registered.";
    }
  } else {
    // 如果观察者为空，则输出警告
    LOG(WARNING) << "Execution trace observer was not initialized.";
  }
}

// 启用执行跟踪观察者
void enableExecutionTraceObserver() {
  // 输出警告，表示正在启用执行跟踪观察者
  LOG(WARNING) << "Enabling Execution Trace Observer";
  auto& ob = *ObserverManager::get();
  // 确保未启用执行跟踪观察者
  if (ob.getState() == ExecutionTraceObserver::RunState::enabled) {
    LOG(WARNING)
        << "Trying to enable Execution Trace Observer when it's already enabled.";
  } else {
    // 设置观察者状态为启用
    ob.setState(ExecutionTraceObserver::RunState::enabled);
  }
}

// 禁用执行跟踪观察者
void disableExecutionTraceObserver() {
  // 输出警告，表示正在禁用执行跟踪观察者
  LOG(WARNING) << "Disabling Execution Trace Observer";
  auto& ob = *ObserverManager::get();
  // 如果执行跟踪观察者状态不是禁用，则设置为禁用
  if (ob.getState() != ExecutionTraceObserver::RunState::disabled) {
    ob.setState(ExecutionTraceObserver::RunState::disabled);
  } else {
    // 如果已经禁用，则输出警告
    LOG(WARNING)
        << "Trying to disable Execution Trace Observer when it's already disabled.";
  }
}
} // namespace torch::profiler::impl
```