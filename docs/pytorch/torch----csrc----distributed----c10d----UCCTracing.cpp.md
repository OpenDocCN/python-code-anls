# `.\pytorch\torch\csrc\distributed\c10d\UCCTracing.cpp`

```
#ifdef USE_C10D_UCC
// 如果定义了 USE_C10D_UCC 宏，则包含以下头文件
#include <torch/csrc/distributed/c10d/UCCTracing.hpp>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

#include <sys/stat.h>
#include <cstdlib>
#include <ctime>
#include <fstream>

namespace c10d {

// 初始化通信追踪日志生成器
void ProcessGroupUCCLogger::initCommsTracer() {
  trace_generator = std::make_shared<CommTraceLogger>();
  initialized_CommTraceLogger = true;
}

// 刷新通信日志
void ProcessGroupUCCLogger::flushComms(int rank, int world_size) {
  // 如果未初始化或者通信追踪为空，则直接返回
  if (!initialized_CommTraceLogger ||
      trace_generator->getCommsTrace().empty()) {
    return;
  }

  // 构建日志目录名，格式为 "ProcessGroupUCC_trace_np<world_size>_<month>_<day>_<year>"
  std::string dirname = c10::str("ProcessGroupUCC_trace_np", world_size);
  time_t now_ = time(0);
  std::tm* ltm = localtime(&now_);
  if (ltm) {
    dirname += c10::str(
        "_", (1 + ltm->tm_mon), "_", ltm->tm_mday, "_", (1900 + ltm->tm_year));
  }

  // 构建完整路径，默认在 "/tmp/" 下，但可以由环境变量 TORCH_UCC_COMMS_TRACE_OUTPUT_DIR 指定
  std::string fullpath = "/tmp/" + dirname;
  char* user_path = std::getenv("TORCH_UCC_COMMS_TRACE_OUTPUT_DIR");
  if (user_path) {
    fullpath = user_path;
  }

  // 构建每个进程的日志文件名，格式为 "<fullpath>/rank<rank>.json"
  std::string trace_filename = c10::str(fullpath, "/rank", rank, ".json");
  std::ofstream _outfile;

  // 如果输出文件没有打开，则尝试创建目录并打开文件
  if (!_outfile.is_open()) {
    if (!mkdir(fullpath.c_str(), 0777)) {
      LOG(INFO) << getLogPrefix() << "[INFO] failed to mkdir " << fullpath;
    } else if (errno != EEXIST) {
      return;
    }
    _outfile.open(trace_filename, std::ofstream::out | std::ofstream::trunc);
  }

  // 将追踪的通信信息写入文件，格式为 JSON 数组
  if (_outfile.is_open()) {
    _outfile << "[" << c10::Join(",", trace_generator->getCommsTrace())
             << "\n]";
    _outfile.flush();
    _outfile.close();
  }
}

/* unused */
// 设置当前通信块的名称
void CommTraceLogger::setCurBlock(const std::string& name) {
  curBlocks_.push_back(
      c10::str("\"", name, "\"")); // add quote marks for JSON format
}

/* unused */
// 弹出当前通信块
void CommTraceLogger::popBlock() {
  // TODO: remove specific name
  curBlocks_.pop_back();
}

// 记录可选的根节点信息
void CommTraceLogger::recordOptionalInfo(int root) {
  curRoot_ = root;
}

// 记录可选的分割大小信息
void CommTraceLogger::recordOptionalInfo(
    const std::vector<int64_t>& outputSplitSizes,
    const std::vector<int64_t>& inputSplitSizes) {
  curOutSplitSizes_ = outputSplitSizes;
  curInSplitSizes_ = inputSizes;
}

// 记录通信
void CommTraceLogger::recordComms(
    const std::string& commName,
    const uintptr_t workReq,
    const int rank,
    const int world_size,
    const std::vector<at::Tensor>& inputTensors,
    // 记录每个操作的输入和输出张量元素数
    const std::vector<at::Tensor>& outputTensors) {
      // 获取第一个输入张量的元素数量，如果没有则为0
      auto inNelems = (!inputTensors.empty()) ? inputTensors[0].numel() : 0;
      // 获取第一个输出张量的元素数量，如果没有则为0
      auto outNelems = (!outputTensors.empty()) ? outputTensors[0].numel() : 0;
      // 获取第一个输出张量的数据类型，如果没有则默认为at::kByte类型
      auto dtype =
          (!outputTensors.empty()) ? outputTensors[0].scalar_type() : at::kByte;
      // 获取第一个输出张量的设备类型，如果没有则默认为CPU
      auto devType = (!outputTensors.empty()) ? outputTensors[0].device().type()
                                              : c10::DeviceType::CPU;
      // 获取当前时间
      auto now = std::chrono::system_clock::now();
      // 静态变量，记录开始时间
      static auto startTS = now;
      // 计算自开始以来的时间差（纳秒）
      int64_t time_since_begin =
          std::chrono::duration_cast<std::chrono::nanoseconds>(now - startTS)
              .count();
    
      // TODO: 如果启用了 torch profiler，则从中获取标记
    
      // 所有操作的通用字段
      std::string cur_trace_ = c10::str(
          "\n\t\t\"markers\": [",
          curBlocks_,
          "]",
          ",\n\t\t\"startTime_ns\": ",
          time_since_begin,
          ",\n\t\t\"comms\": \"",
          commName,
          "\"",
          ",\n\t\t\"req\": ",
          workReq,
          ",\n\t\t\"seqnum\": ",
          seqnum,
          ",\n\t\t\"world_size\": ",
          world_size);
    
      // 如果输入或输出张量元素数量大于0，则为大多数集合操作追加消息大小、数据类型和设备类型
      if (inNelems > 0 || outNelems > 0) {
        cur_trace_ = c10::str(
            cur_trace_,
            ",\n\t\t\"in_msg_size\": ",
            inNelems,
            ",\n\t\t\"out_msg_size\": ",
            outNelems,
            ",\n\t\t\"dtype\": \"",
            at::toString(dtype),
            "\",\n\t\t\"devType\": \"",
            c10::DeviceTypeName(devType),
            "\"");
      }
      // 如果存在当前的根节点索引，则追加根节点的排名
      if (curRoot_ != -1) {
        cur_trace_ = c10::str(cur_trace_, ",\n\t\t\"root\": ", curRoot_);
      }
      // 如果存在输入或输出分割大小，则追加它们
      if (!curInSplitSizes_.empty() || !curOutSplitSizes_.empty()) {
        cur_trace_ = c10::str(
            cur_trace_,
            ",\n\t\t\"in_split\": [",
            c10::Join(",", curInSplitSizes_),
            "]"
            ",\n\t\t\"out_split\": [",
            c10::Join(",", curOutSplitSizes_),
            "]");
      }
      // 将当前追踪字符串添加到通信追踪列表中
      comms_trace_.push_back(c10::str("\n\t{", cur_trace_, "\n\t}"));
    
      // 如果适用，记录追踪到 kineto trace
      RECORD_PARAM_COMMS(
          static_cast<int64_t>(seqnum), // seq
          std::make_tuple("0", ""), // pg_name tuple
          rank,
          commName.c_str(),
          inNelems,
          outNelems,
          dtype,
          curInSplitSizes_,
          curOutSplitSizes_,
          -1,
          -1,
          world_size);
    
      // 递增序列号
      ++seqnum;
    
      // 重置可选字段
      curRoot_ = -1;
      curInSplitSizes_ = {};
      curOutSplitSizes_ = {};
}

} // namespace c10d

#endif // USE_C10D_UCC
```