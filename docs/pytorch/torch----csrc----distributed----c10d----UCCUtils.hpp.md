# `.\pytorch\torch\csrc\distributed\c10d\UCCUtils.hpp`

```py
#pragma once

// 使用 `#pragma once` 来确保头文件只被编译一次，防止多重包含问题


#ifdef USE_C10D_UCC

// 如果定义了宏 `USE_C10D_UCC`，则编译以下代码块；否则忽略这部分内容


#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <ucc/api/ucc.h>

// 引入所需的头文件，包括 Torch 的分布式进程组和存储头文件，以及 UCC 库的头文件


namespace c10d {

// 进入 c10d 命名空间


#define TORCH_UCC_GET_ERROR_MSG(_err, _error_msg, _result) \
  do {                                                     \
    _err = c10::str(                                       \
        "[",                                               \
        std::string(__FILE__),                             \
        ":",                                               \
        std::to_string(__LINE__),                          \
        "] ",                                              \
        logger->getLogPrefix(),                            \
        _error_msg,                                        \
        ", error code ",                                   \
        _result,                                           \
        ": ",                                              \
        ucc_status_string(_result),                        \
        ", system error code ",                            \
        errno);                                            \
  } while (0)

// 定义宏 `TORCH_UCC_GET_ERROR_MSG`，用于生成 UCC 返回值非成功时的错误消息字符串


#define TORCH_UCC_CHECK(_cmd, _error_msg)               \
  do {                                                  \
    ucc_status_t result = _cmd;                         \
    if (result != UCC_OK) {                             \
      std::string err;                                  \
      TORCH_UCC_GET_ERROR_MSG(err, _error_msg, result); \
      TORCH_CHECK(false, err);                          \
    }                                                   \
  } while (0)

// 定义宏 `TORCH_UCC_CHECK`，用于在 UCC 返回值非成功时抛出异常，包含错误消息


#define TORCH_UCC_CHECK_REQUEST(_request, _cmd, _error_msg) \
  do {                                                      \
    ucc_status_t result = _cmd;                             \
    if (result != UCC_OK) {                                 \
      std::string err;                                      \
      TORCH_UCC_GET_ERROR_MSG(err, _error_msg, result);     \
      if (_request != nullptr) {                            \
        ucc_collective_finalize(_request);                  \
      }                                                     \
      TORCH_CHECK(false, err);                              \
    }                                                       \
  } while (0)

// 定义宏 `TORCH_UCC_CHECK_REQUEST`，用于在 UCC 返回值非成功时抛出异常，并释放请求


#define TORCH_UCC_LOG_ERROR(_phase, _msg) \
  LOG(ERROR) << logger->getLogPrefix(_phase) << "[ERROR] " << _msg;

// 定义宏 `TORCH_UCC_LOG_ERROR`，用于记录错误日志，包含阶段信息和消息内容


#define TORCH_UCC_LOG_INFO(_phase, _msg) \
  LOG(INFO) << logger->getLogPrefix(_phase) << "[INFO] " << _msg;

// 定义宏 `TORCH_UCC_LOG_INFO`，用于记录信息日志，包含阶段信息和消息内容


#define TORCH_UCC_LOG_DEBUG(_phase, _msg) \
  VLOG(1) << logger->getLogPrefix(_phase) << "[DEBUG] " << _msg;

// 定义宏 `TORCH_UCC_LOG_DEBUG`，用于记录调试日志，包含阶段信息和消息内容
// 枚举类型定义了 TORCH_UCC_PHASE_T 枚举，表示不同的 UCC 运行阶段
enum torch_ucc_phase_t {
  TORCH_UCC_UNKNOWN = -1,  // 未知阶段，值为 -1
  TORCH_UCC_INIT,          // 初始化阶段
  TORCH_UCC_HEALTH_CHECK,  // 健康检查阶段
  TORCH_UCC_READY,         // 就绪阶段
  TORCH_UCC_COLL_POST,     // 合集后处理阶段
  TORCH_UCC_COLL_PROGRESS, // 合集进展阶段
  TORCH_UCC_FINALIZE,      // 结束阶段
};

// 使用 std::map 创建了一个映射，将每个 TORCH_UCC_PHASE_T 值映射到对应的字符串
const std::map<torch_ucc_phase_t, std::string> ucc_phase_map = {
    {TORCH_UCC_UNKNOWN, "UNKNOWN"},
    {TORCH_UCC_INIT, "INIT"},
    {TORCH_UCC_HEALTH_CHECK, "HEALTH_CHECK"},
    {TORCH_UCC_READY, "READY"},
    {TORCH_UCC_COLL_POST, "COLL_POST"},
    {TORCH_UCC_COLL_PROGRESS, "COLL_PROGRESS"},
    {TORCH_UCC_FINALIZE, "FINALIZE"},
};

// CommTraceLogger 类的前向声明
class CommTraceLogger;

// TORCH_API ProcessGroupUCCLogger 类的定义，继承自 torch::CustomClassHolder
class ProcessGroupUCCLogger : public torch::CustomClassHolder {
 public:
  ProcessGroupUCCLogger(); // 默认构造函数声明
  ProcessGroupUCCLogger(std::string log_prefix, torch_ucc_phase_t phase); // 带参数的构造函数声明

  // 获取日志前缀的方法声明，可以传入特定的阶段参数
  std::string getLogPrefix(torch_ucc_phase_t phase = TORCH_UCC_UNKNOWN);
  
  // 设置日志前缀的方法声明
  void setLogPrefix(std::string log_prefix);
  
  // 设置当前阶段的方法声明，使用内联函数
  inline void setPhase(torch_ucc_phase_t phase) {
    local_phase = phase;
  }

  // 初始化通信追踪器的方法声明
  void initCommsTracer();
  
  // 刷新通信日志的方法声明，传入当前进程的排名和总进程数
  void flushComms(int rank, int world_size);
  
  // 共享指针，指向 CommTraceLogger 对象，用于生成通信追踪日志
  std::shared_ptr<CommTraceLogger> trace_generator = nullptr;

 protected:
  std::string log_prefix;        // 日志前缀字符串
  torch_ucc_phase_t local_phase; // 当前日志阶段
  bool initialized_CommTraceLogger; // 是否已经初始化 CommTraceLogger
};

// torch_ucc_oob_coll_info_t 结构体声明
struct torch_ucc_oob_coll_info_t {
  c10::intrusive_ptr<Store> store; // 指向 Store 的智能指针
  uint32_t comm_id;                // 通信 ID
  int rank;                        // 进程排名
  int size;                        // 进程总数
  void* rbuf;                      // 接收缓冲区
  size_t msglen;                   // 消息长度

  // 获取带有指定键的字符串方法声明
  std::string getKey(std::string key) {
    return std::to_string(comm_id) + key;
  }
};

// CommBase 类声明
class CommBase {
 public:
  // 构造函数声明，接受 ProcessGroupUCCLogger 的智能指针作为参数
  CommBase(const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger_)
      : logger(logger_) {}
  
  // 纯虚函数，派生类需要实现的方法：进度推进
  virtual void progress() = 0;
  
  // 纯虚函数，派生类需要实现的方法：释放请求
  virtual void free_request(ucc_coll_req_h request) = 0;
  
  // 虚析构函数声明
  virtual ~CommBase() {}
  
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger; // ProcessGroupUCCLogger 的智能指针成员
};

// CommUCC 类声明，继承自 CommBase
class CommUCC : public CommBase {
 public:
  ucc_lib_h lib{nullptr};       // UCC 库句柄
  ucc_context_h context{nullptr}; // UCC 上下文句柄

 public:
  // 进度推进方法的重写
  void progress() override;
  
  // 构造函数声明，接受指向 torch_ucc_oob_coll_info_t 结构体的共享指针和 ProcessGroupUCCLogger 的智能指针作为参数
  CommUCC(
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);
  
  // 释放请求方法的重写
  void free_request(ucc_coll_req_h request) override;
  
  // 析构函数声明
  ~CommUCC();
};

// oob_allgather 函数声明，用于执行 allgather 操作
ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req);

// oob_allgather_test 函数声明，用于测试 allgather 请求的状态
ucc_status_t oob_allgather_test(void* req);

// oob_allgather_free 函数声明，用于释放 allgather 请求
ucc_status_t oob_allgather_free(void* req);

// trim 函数声明，用于移除字符串视图两端的空白字符，实现来自 https://stackoverflow.com/a/17976541
inline c10::string_view trim(c10::string_view s) {
  auto wsfront = std::find_if_not(
      s.begin(), s.end(), [](int c) { return std::isspace(c); });
  auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) {
                  return std::isspace(c);
                }).base();
  return (
      wsback <= wsfront ? "" : s.substr(wsfront - s.begin(), wsback - wsfront));
}

// tolower 函数声明，将字符串视图转换为小写形式
inline std::string tolower(c10::string_view s) {
  std::string result;
  result.reserve(s.size());
  for (auto c : s) {
    result.push_back(std::tolower(c));
  }
  return result;
}
// 将给定的字符串列表进行解析，并返回解析后的字符串向量
inline std::vector<std::string> parse_list(std::string list) {
  // 初始化空的结果向量
  std::vector<std::string> result;
  // 将输入列表转换为小写，并去除两侧空白
  list = tolower(trim(list));
  // 当列表不为空时进行循环处理
  while (!list.empty()) {
    // 查找逗号的位置
    const auto end_pos = list.find_first_of(',');
    // 提取并去除首部空白的 token
    const auto token = trim(list.substr(0, end_pos));
    // 将 token 添加到结果向量中
    result.push_back(std::string(token));
    // 更新列表，如果仍有逗号，则截取其后的部分，否则置空
    list = (end_pos != c10::string_view::npos) ? list.substr(end_pos + 1) : "";
  }
  // 返回解析后的结果向量
  return result;
}
```