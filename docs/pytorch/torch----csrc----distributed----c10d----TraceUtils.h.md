# `.\pytorch\torch\csrc\distributed\c10d\TraceUtils.h`

```
#pragma once
// 包含 C10 库的 ScalarType 头文件
#include <c10/core/ScalarType.h>
// 包含 C10 工具库的 ApproximateClock 头文件
#include <c10/util/ApproximateClock.h>
// 包含 C10 工具库的 irange 头文件
#include <c10/util/irange.h>
// 包含 C10 工具库的 string_view 头文件
#include <c10/util/string_view.h>
// 包含 Torch 分布式库的 Store 头文件
#include <torch/csrc/distributed/c10d/Store.hpp>
// 包含 Torch 分布式库的 Types 头文件
#include <torch/csrc/distributed/c10d/Types.hpp>
// 包含 Torch 分布式库的 Utils 头文件
#include <torch/csrc/distributed/c10d/Utils.hpp>
// 包含 Torch JIT 序列化库的 pickler 头文件
#include <torch/csrc/jit/serialization/pickler.h>
// 包含 Torch 分析器的 combined_traceback 头文件
#include <torch/csrc/profiler/combined_traceback.h>
// 包含标准库的 chrono 头文件
#include <chrono>

// 包含系统级类型定义的头文件
#include <sys/types.h>
// 包含 C 标准库的头文件
#include <cstdlib>
// 包含文件流处理的头文件
#include <fstream>
// 包含字符串处理的头文件
#include <string>
// 包含系统错误处理的头文件
#include <system_error>
// 包含向量处理的头文件
#include <vector>

// 定义命名空间 c10d
namespace c10d {

// 定义一个内联函数，返回跟踪开始的键值
inline std::string getTraceStartKey(const std::string& pgName, int rank) {
  return pgName + "_" + std::to_string(rank) + "_trace_start";
}

// 定义一个内联函数，返回跟踪结束的键值
inline std::string getTraceEndKey(const std::string& pgName, int rank) {
  return pgName + "_" + std::to_string(rank) + "_trace_end";
}

// 定义一个内联函数，用于更新跟踪信息到存储中
inline bool traceUpdate(
    c10::intrusive_ptr<Store>& store, // 分布式存储的智能指针
    const std::string& key, // 存储键值
    uint64_t seq, // 序列号
    const std::string& col) { // 要更新的数据列
  // 创建包含序列号、数据列的字节数组
  std::vector<uint8_t> value(col.size() + sizeof(seq) + 1);
  memcpy(value.data(), &seq, sizeof(seq));
  memcpy(value.data() + sizeof(seq), col.data(), col.size());
  try {
    // 将更新写入存储
    store->set(key, value);
    return true;
  } catch (...) {
    // 如果写入失败，则记录错误信息并返回 false
    LOG(ERROR) << "Store is down while updating #" << seq << " with key "
               << key;
    return false;
  }
  return true;
}

// 枚举类型，表示跟踪调试事件的开始和结束
enum TraceDebugEvent {
  kEventStart, // 开始事件
  kEventEnd,   // 结束事件
};

// 定义一个类型别名，表示跟踪映射的数据结构
// <seq, <rank, <col, start/end>>>
using TraceMap =
    std::map<uint64_t, std::map<int, std::pair<std::string, TraceDebugEvent>>>;

// 定义一个内联函数，将整数向量转换为字符串
inline std::string ranksToString(const std::vector<int>& ranks) {
  std::string str;
  for (int rank : ranks) {
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      str += ", " + std::to_string(rank);
    }
  }
  return str;
}

// 定义一个内联函数，从跟踪项中提取整数向量的字符串表示
inline std::string ranksFromTrace(
    const std::vector<std::pair<int, std::string>>& items) {
  std::string ranks;
  for (auto& p : items) {
    if (ranks.empty()) {
      ranks = std::to_string(p.first);
    } else {
      ranks += ", " + std::to_string(p.first);
    }
  }
  return ranks;
}

// 定义一个内联函数，分析缺失的排名信息
inline std::string analyzeMissingRanks(const std::vector<int>& missingRanks) {
  return c10::str(
      "\n\t - To our best knowledge, ranks [",
      ranksToString(missingRanks),
      "] are the lagging ranks that caused this timeout. "
      "They never joined any collectives");
}

// 定义一个内联函数，分析滞后的排名信息
inline std::string analyzeLaggingRanks(const TraceMap& traceMap) {
  uint64_t lagSeq = traceMap.begin()->first;
  std::vector<int> startRanks;
  std::vector<int> endRanks;
  for (auto& p : traceMap.begin()->second) {
    if (p.second.second == kEventStart) {
      startRanks.push_back(p.first);
    } else {
      endRanks.push_back(p.first);
    }
  }
  std::string report =
      "\n\t - To our best knowledge, the lagging/dead/mismatched ranks "
      "that caused the desync are:";
  if (startRanks.size()) {
    // 将开始排名转换为字符串，并添加到报告中
    report += c10::str(
        "\n\t   - [",
        ranksToString(startRanks),
        "] joined but didn't finish collective #",
        lagSeq,
        " (count from 1)");
  }
  // 如果结束排名非空，则将其转换为字符串并添加到报告中
  if (endRanks.size()) {
    report += c10::str(
        "\n\t     [",
        ranksToString(endRanks),
        "] finished collective #",
        lagSeq,
        ", but didn't join collective #",
        lagSeq + 1,
        " (count from 1)");
  }
  // 返回最终生成的报告字符串
  return report;
}

inline std::string dumpSnapshot(TraceMap& traceMap) {
  // 函数开始，生成快照报告字符串
  std::string report = "\n\t - Snapshot of ranks' latest states:";
  // 遍历 traceMap 中的每一对 (seq, subMap)
  for (auto& tracePair : traceMap) {
    uint64_t seq = tracePair.first; // 获取序列号 seq
    std::map<int, std::pair<std::string, TraceDebugEvent>>& subMap =
        tracePair.second; // 获取子映射 subMap

    // 创建存储开始和结束事件的集合
    std::unordered_map<std::string, std::vector<int>> collectivesStart;
    std::unordered_map<std::string, std::vector<int>> collectivesEnd;
    // 遍历 subMap 中的每一对 (rank, p)
    for (auto& p : subMap) {
      int rank = p.first; // 获取排名 rank
      const std::string& col = p.second.first; // 获取事件类型 col
      // 根据事件类型判断是开始还是结束事件，并将 rank 添加到相应的集合中
      if (p.second.second == kEventStart) {
        collectivesStart[col].push_back(rank);
      } else {
        collectivesEnd[col].push_back(rank);
      }
    }

    // 如果有开始事件，添加到报告中
    if (collectivesStart.size()) {
      report += c10::str("\n\t   #", seq, " started ranks:");
      for (auto& mapPair : collectivesStart) {
        // 格式化开始事件信息并添加到报告中
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] started ",
            mapPair.first);
      }
    }
    // 如果有结束事件，添加到报告中
    if (collectivesEnd.size()) {
      report += c10::str("\n\t   #", seq, " finished ranks:");
      for (auto& mapPair : collectivesEnd) {
        // 格式化结束事件信息并添加到报告中
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] finished ",
            mapPair.first);
      }
    }
  }
  // 返回生成的快照报告字符串
  return report;
}

inline bool parseTraceValue(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t& seq,
    std::string& col) {
  try {
    // 从存储中获取跟踪值的字节流
    std::vector<uint8_t> traceValue = store->get(key);
    // 解析序列号 seq
    memcpy(&seq, traceValue.data(), sizeof(seq));
    // 解析事件类型 col
    std::string colName((char*)traceValue.data() + sizeof(seq));
    col = colName;
    return true;
  } catch (...) {
    // 如果出现异常，记录错误日志并返回 false
    LOG(ERROR) << "Store is down while getting key " << key;
    return false;
  }
  // 返回 true
  return true;
}

inline std::string retrieveDesyncReport(
    c10::intrusive_ptr<Store>& store,
    const std::string& pgName,
    int myRank,
    int worldSize) {
  // 初始化报告字符串
  std::string report;

  // 初始化当前进程的序列号和事件类型
  uint64_t thisSeq;
  std::string thisCol;

  // 初始化丢失排名列表和跟踪映射
  std::vector<int> missingRanks;
  TraceMap traceMap;

  // 遍历所有排名
  for (const auto rank : c10::irange(worldSize)) {
    // 构建 traceMapStart
    uint64_t seqStart;
    {
      // 获取开始事件的跟踪键
      std::string traceKeyStart = getTraceStartKey(pgName, rank);
      // 检查存储中是否存在该键
      if (!store->check({traceKeyStart})) {
        // 如果不存在，将该排名添加到丢失排名列表中，并继续下一个排名
        missingRanks.push_back(rank);
        continue;
      }
      // 解析开始事件的跟踪值，获取序列号和事件类型
      std::string col;
      if (!parseTraceValue(store, traceKeyStart, seqStart, col)) {
        return report;
      }
      // 将 rank、col 和 kEventStart 添加到 traceMap 中
      traceMap[seqStart].emplace(rank, std::make_pair(col, kEventStart));
      // 如果当前进程是该排名，更新当前序列号和事件类型
      if (rank == myRank) {
        thisSeq = seqStart;
        thisCol = std::move(col);
      }
    }

    // 构建 traceMapEnd
    {
      // 获取跟踪结束键
      std::string traceKeyEnd = getTraceEndKey(pgName, rank);
      // 检查存储中是否存在跟踪结束键对应的条目，如果不存在则继续下一次循环
      if (!store->check({traceKeyEnd})) {
        continue;
      }
      // 用于存储解析后的序列号和列名
      uint64_t seq;
      std::string col;
      // 解析跟踪值，如果解析失败则返回当前报告
      if (!parseTraceValue(store, traceKeyEnd, seq, col)) {
        return report;
      }
      // 如果解析得到的序列号与起始序列号相等，则更新跟踪映射中的事件状态为 kEventEnd
      if (seq == seqStart) {
        traceMap[seq][rank].second = kEventEnd;
      }
    }
    
    // 断言：确保 missingRanks 不为空或 traceMap 不为空，否则报错信息
    TORCH_INTERNAL_ASSERT(
        !missingRanks.empty() || !traceMap.empty(),
        "Trace shouldn't be empty while enabled GLOO_ASYNC_TIMEOUT_DEBUG");
    // 断言：确保 thisCol 不为空，否则报错信息
    TORCH_INTERNAL_ASSERT(
        !thisCol.empty(),
        "Timeout rank [",
        myRank,
        "] must have collective tracking item in c10::Store trace");
    // 断言：确保跟踪映射中指定序列号和当前进程的事件状态为 kEventStart，否则报错信息
    TORCH_INTERNAL_ASSERT(
        traceMap[thisSeq][myRank].second == kEventStart,
        "Timeout rank [",
        myRank,
        "] last trace item must be kEventStart. thisSeq = ",
        thisSeq,
        ", col = ",
        thisCol);
    
    // 构建报告内容，指明超时的进程、超时的集合名称、以及超时的序列号
    report += c10::str(
        "\n\t - [", myRank, "] Timeout at collective: ", thisCol, ", #", thisSeq);
    
    // 如果 missingRanks 不为空，则分析缺失的进程并追加到报告中
    if (!missingRanks.empty()) {
      report += analyzeMissingRanks(missingRanks);
    } else {
      // 否则分析滞后的进程并追加到报告中，并将当前跟踪映射的快照转储到报告中
      report += analyzeLaggingRanks(traceMap);
      report += dumpSnapshot(traceMap);
    }
    
    // 返回最终构建的报告
    return report;
    }
// 声明一个 inline 函数，将给定的 c10::IValue 类型对象序列化为字符串
inline std::string pickle_str(const c10::IValue& v) {
  // 创建一个空的字符向量，用于存储序列化结果
  std::vector<char> result;
  {
    // 定义一个 lambda 函数 writer，用于向 result 向量中添加数据
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    // 创建一个 torch::jit::Pickler 对象 pickler，用于序列化数据
    torch::jit::Pickler pickler(
        writer, nullptr, nullptr, nullptr, nullptr, false);
    // 设置 pickler 的协议版本
    pickler.protocol();
    // 将给定的 IValue 对象 v 推入 pickler 中进行序列化
    pickler.pushIValue(v);
    // 结束序列化过程
    pickler.stop();
  }
  // 将字符向量 result 转换为字符串并返回
  return std::string(result.begin(), result.end());
}

// 声明一个 inline 函数，获取 Python 和 C++ 混合堆栈跟踪信息的字符串表示
inline std::string get_python_cpp_trace() {
  // 使用 torch::CapturedTraceback::gather() 收集堆栈跟踪信息
  std::shared_ptr<torch::CapturedTraceback> tb =
      torch::CapturedTraceback::gather(
          /*python=*/true, /*script=*/true, /*cpp=*/true);
  // 使用 torch::symbolize() 对收集的跟踪信息进行符号化处理
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({tb.get()});
  // 获取符号化后的堆栈跟踪信息
  const auto& s_tb = s_tbs.tracebacks.at(0);
  // 创建一个字符串流 oss 用于构建输出的堆栈跟踪信息字符串
  std::stringstream oss;
  // 遍历符号化的堆栈帧信息 s_tb
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    // 获取当前帧的详细信息 frame
    const auto& frame = s_tbs.all_frames.at(frame_id);
    // 将帧信息格式化为字符串添加到 oss 中
    oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
        << ":" << frame.lineno << std::endl;
  }
  // 返回构建的堆栈跟踪信息字符串
  return oss.str();
}

// 声明一个 inline 函数，创建一个空的 c10::Dict 对象，键和值的类型为 c10::IValue
inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

// 声明一个 inline 函数，创建一个空的 c10::List 对象，元素的类型为 c10::IValue
inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

// 声明一个 inline 函数，将给定的无符号 64 位整数向量转换为字符串表示
inline std::string ranks_str(const std::vector<uint64_t>& ranks) {
  // 创建一个空字符串 str
  std::string str;
  // 遍历 ranks 向量中的每个元素 rank
  for (const auto& rank : ranks) {
    // 如果 str 是空的，直接将 rank 转换为字符串赋值给 str
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      // 否则在 str 后添加 ", " 和当前 rank 的字符串表示
      str += ", " + std::to_string(rank);
    }
  }
  // 返回格式化后的字符串，形如 "[rank1, rank2, ...]"
  return c10::str("[", str, "]");
}

// 结束 c10d 命名空间的定义
} // namespace c10d
```