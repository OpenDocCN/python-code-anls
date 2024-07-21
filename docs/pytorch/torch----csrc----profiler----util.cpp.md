# `.\pytorch\torch\csrc\profiler\util.cpp`

```py
// 包含 Torch 的自动微分函数头文件
#include <torch/csrc/autograd/function.h>
// 包含 Kineto 性能分析的接口头文件
#include <torch/csrc/profiler/kineto_shim.h>
// 包含 Torch 性能分析的实用工具头文件
#include <torch/csrc/profiler/util.h>

// 包含 C10 库的 ArrayRef 类型定义
#include <c10/util/ArrayRef.h>
// 包含 C10 库的循环范围迭代函数定义
#include <c10/util/irange.h>
// 使用 fmt 库进行格式化输出
#include <fmt/format.h>

#ifdef USE_KINETO
// 如果定义了 USE_KINETO，则包含 Kineto 库的头文件
#include <libkineto.h>
#endif

#ifdef USE_DISTRIBUTED
// 如果定义了 USE_DISTRIBUTED，则包含分布式通信工具 ParamCommsUtils 的头文件
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#endif // USE_DISTRIBUTED

// 定义 torch::profiler::impl 命名空间
namespace torch::profiler::impl {

// 匿名命名空间，用于保存 soft_assert_raises_ 的可选布尔值
namespace {
std::optional<bool> soft_assert_raises_;
} // namespace

// 设置 soft_assert_raises_ 的函数
void setSoftAssertRaises(std::optional<bool> value) {
  soft_assert_raises_ = value;
}

// 获取 soft_assert_raises_ 的函数，默认为 false
bool softAssertRaises() {
  return soft_assert_raises_.value_or(false);
}

// 记录软断言失败的函数，输出错误信息和相关信息
void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    const char* args) {
#ifdef USE_KINETO
  std::string error;
  error = fmt::format(
      "{} SOFT ASSERT FAILED at {}:{}, func: {}, args: {}",
      cond,
      file,
      line,
      func,
      args);
  // TODO: Implement profile_id and group_profile_id as 3rd/4th arguments.
  kineto::logInvariantViolation(cond, error, "", "");
#endif
}

// 重载记录软断言失败的函数，支持 std::string 类型的 args
void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    const std::string& args) {
#ifdef USE_KINETO
  std::string error;
  error = fmt::format(
      "{} SOFT ASSERT FAILED at {}:{}, func: {}, args: {}",
      cond,
      file,
      line,
      func,
      args);
  // TODO: Implement profile_id and group_profile_id as 3rd/4th arguments.
  kineto::logInvariantViolation(cond, error, "", "");
#endif
}

// ----------------------------------------------------------------------------
// -- NVTX --------------------------------------------------------------------
// ----------------------------------------------------------------------------

// 获取 NVTX 描述字符串的函数，包括操作名称、序列号、形状、操作标识等信息
std::string getNvtxStr(
    const char* name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes,
    at::RecordFunctionHandle op_id,
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids) {
  if (sequence_nr >= -1 || !shapes.empty()) {
    std::string str;
    if (sequence_nr >= 0) {
      str = fmt::format("{}, seq = {}", name, sequence_nr);
    } else if (sequence_nr == -1) {
      str = name;
    } else {
#if defined(USE_ROCM)
      // 只有 ROCM 支持小于 -1 的 sequence_nr
      str = name;
#endif
    }
    if (op_id > 0) {
      str = fmt::format("{}, op_id = {}", str, op_id);
    }
    if (!shapes.empty()) {
      str = fmt::format("{}, sizes = {}", str, shapesToStr(shapes));
    }
    // 包括输入边的操作标识，用于构建网络图
    if (!input_op_ids.empty()) {
      str = fmt::format(
          "{}, input_op_ids = {}", str, inputOpIdsToStr(input_op_ids));
    }
    return str;
  } else {
    return name;
  }
}

// ----------------------------------------------------------------------------
// -- Op context (shapes, call stack) -----------------------------------------
// ----------------------------------------------------------------------------
// 准备调用堆栈信息，将jit::StackEntry转换为FileLineFunc的向量
std::vector<FileLineFunc> prepareCallstack(
    const std::vector<jit::StackEntry>& cs) {
  // 存储调用堆栈条目的向量
  std::vector<FileLineFunc> entries;
  // 预留足够的空间以避免重复分配
  entries.reserve(cs.size());
  // 遍历每个调用堆栈条目
  for (const auto& entry : cs) {
    auto& range = entry.range;
    // 检查条目的源代码范围是否有效
    if (range.source()) {
      auto& src = range.source();
      // 如果源文件和文件名有效，则获取行号并创建FileLineFunc对象
      if (src && src->filename()) {
        auto line =
            src->starting_line_no() + src->lineno_for_offset(range.start());
        entries.emplace_back(
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            FileLineFunc{*(src->filename()), line, entry.filename});
      }
    }
  }
  return entries;  // 返回包含源文件名、行号和函数名的向量
}

// 将FileLineFunc的向量转换为字符串形式的调用堆栈
std::vector<std::string> callstackStr(const std::vector<FileLineFunc>& cs) {
  // 存储调用堆栈的字符串形式
  std::vector<std::string> cs_str;
  // 预留足够的空间以避免重复分配
  cs_str.reserve(cs.size());
  // 遍历每个FileLineFunc条目，将其转换为字符串
  for (const auto& entry : cs) {
    std::stringstream loc;
    loc << entry.filename << "(" << entry.line << "): " << entry.funcname;
    cs_str.push_back(loc.str());  // 添加格式化后的调用堆栈条目字符串
  }
  return cs_str;  // 返回调用堆栈的字符串形式向量
}

// 将调用堆栈的字符串形式向量连接为单个字符串，使用指定的分隔符
std::string stacksToStr(
    const std::vector<std::string>& stacks,
    const char* delim) {
  // 创建字符串流以便连接调用堆栈条目
  std::ostringstream oss;
  // 将调用堆栈条目连接为单个字符串，使用给定的分隔符
  std::transform(
      stacks.begin(),
      stacks.end(),
      std::ostream_iterator<std::string>(oss, delim),
      [](std::string s) -> std::string {
#ifdef _WIN32
        // 如果在Windows平台，替换反斜杠为正斜杠
        std::replace(s.begin(), s.end(), '\\', '/');
#endif
        return s;  // 返回替换后的字符串
      });
  auto rc = oss.str();
  return "\"" + rc + "\"";  // 返回用双引号包围的连接后的调用堆栈字符串
}

// 将c10::List<c10::IValue>列表中的张量维度展开为二维整数向量的向量
static std::vector<std::vector<int64_t>> flattenList(
    const c10::List<c10::IValue>& list) {
  // 存储展开后的张量维度向量
  std::vector<std::vector<int64_t>> tensor_dims;
  // 遍历列表中的每个c10::IValue
  for (const c10::IValue& input : list) {
    // 如果当前值是张量，则获取其大小并添加到结果中
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      if (tensor.defined()) {
        tensor_dims.push_back(input.toTensor().sizes().vec());
      }
    }
  }
  return tensor_dims;  // 返回展开后的张量维度向量
}

// 提取at::RecordFunction对象中输入张量的维度信息
std::vector<std::vector<int64_t>> inputSizes(
    const at::RecordFunction& fn,
    bool flatten_list_enabled) {
  // 存储所有输入张量的维度信息的向量
  std::vector<std::vector<int64_t>> sizes;
  // 预留足够的空间以避免重复分配
  sizes.reserve(fn.inputs().size());
  // 遍历RecordFunction的所有输入
  for (const c10::IValue& input : fn.inputs()) {
    // 如果当前输入是张量，则获取其大小并添加到结果中
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      if (tensor.defined()) {
        sizes.push_back(input.toTensor().sizes().vec());
      } else {
        sizes.emplace_back();  // 如果未定义，则添加空的向量
      }
    } else if (input.isList()) {
      std::vector<std::vector<int64_t>> tmp_sizes;
      // 如果启用了展开列表选项，则展开列表中的每个张量并扩展sizes数组
      if (flatten_list_enabled) {
        tmp_sizes = flattenList(input.toList());
      }
      // 将展开后的维度向量添加到sizes数组中
      if (!tmp_sizes.empty()) {
        sizes.insert(sizes.end(), tmp_sizes.begin(), tmp_sizes.end());
      } else {
        sizes.emplace_back();  // 如果为空，则添加空的向量
      }
    } else {
      sizes.emplace_back();  // 其他情况下也添加空的向量
    }
  }
  return sizes;  // 返回所有输入张量的维度信息的向量
}
// 将包含形状信息的二维向量转换为字符串表示
std::string shapesToStr(const std::vector<std::vector<int64_t>>& shapes) {
  // 初始化结果字符串，表示形状列表的起始
  std::string str("[");
  // 遍历形状列表的索引
  for (const auto t_idx : c10::irange(shapes.size())) {
    // 如果不是第一个形状，则添加逗号和空格
    if (t_idx > 0) {
      str = fmt::format("{}, ", str);
    }
    // 将当前形状转换为字符串并添加到结果中
    str = fmt::format("{}{}", str, shapeToStr(shapes[t_idx]));
  }
  // 完成形状列表的字符串表示
  str = fmt::format("{}]", str);
  return str;
}

// 将包含不同类型形状的向量转换为字符串表示
std::string variantShapesToStr(const std::vector<shape>& shapes) {
  // 初始化结果字符串，表示形状列表的起始
  std::string str("[");
  // 遍历形状列表的索引
  for (const auto t_idx : c10::irange(shapes.size())) {
    // 如果不是第一个形状，则添加逗号和空格
    if (t_idx > 0) {
      str = fmt::format("{}, ", str);
    }
    // 根据形状的类型，分别处理
    if (std::holds_alternative<std::vector<int64_t>>(shapes[t_idx])) {
      // 如果形状是整数向量，则转换为字符串并添加到结果中
      const auto& shape = std::get<std::vector<int64_t>>(shapes[t_idx]);
      str = fmt::format("{}{}", str, shapeToStr(shape));
    } else if (std::holds_alternative<std::vector<std::vector<int64_t>>>(shapes[t_idx])) {
      // 如果形状是二维整数向量，则处理每个向量并添加到结果中
      const auto& tensor_shape = std::get<std::vector<std::vector<int64_t>>>(shapes[t_idx]);
      // 如果向量的长度超过限制，则跳过处理
      if (tensor_shape.size() > TENSOR_LIST_DISPLAY_LENGTH_LIMIT) {
        str = fmt::format("{}[]", str);
        continue;
      }
      // 处理每个二维向量并添加到结果中
      str = fmt::format("{}[", str);
      for (const auto s_idx : c10::irange(tensor_shape.size())) {
        if (s_idx > 0) {
          str = fmt::format("{}, ", str);
        }
        str = fmt::format("{}{}", str, shapeToStr(tensor_shape[s_idx]));
      }
      str = fmt::format("{}]", str);
    }
  }
  // 完成形状列表的字符串表示
  str = fmt::format("{}]", str);
  return str;
}

// 将整数向量转换为字符串表示
std::string shapeToStr(const std::vector<int64_t>& shape) {
  // 初始化结果字符串，表示整数向量的起始
  std::string str("[");
  // 遍历整数向量的索引
  for (const auto s_idx : c10::irange(shape.size())) {
    // 如果不是第一个整数，则添加逗号和空格
    if (s_idx > 0) {
      str = fmt::format("{}, ", str);
    }
    // 将当前整数转换为字符串并添加到结果中
    str = fmt::format("{}{}", str, shape[s_idx]);
  }
  // 完成整数向量的字符串表示
  str = fmt::format("{}]", str);
  return str;
}

// 将输入操作的标识和输出编号列表转换为字符串表示
std::string inputOpIdsToStr(const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids) {
  // 初始化结果字符串，表示操作标识列表的起始
  std::string str("[");
  // 初始化索引变量
  int idx = 0;
  // 遍历输入操作标识列表
  for (const auto& op_id_info_pair : input_op_ids) {
    // 如果不是第一个操作标识，则添加逗号和空格
    if (idx++ > 0) {
      str = fmt::format("{}, ", str);
    }
    // 将操作标识和输出编号格式化为字符串并添加到结果中
    str = fmt::format("{}({},{})", str, op_id_info_pair.first, op_id_info_pair.second);
  }
  // 完成操作标识列表的字符串表示
  str = fmt::format("{}]", str);
  return str;
}

// 将字符串列表转换为字符串表示
std::string strListToStr(const std::vector<std::string>& types) {
  // 如果字符串列表为空，则返回空列表的字符串表示
  if (types.empty()) {
    return "[]";
  } else {
    // 否则，使用字符串流处理每个字符串，添加引号并以逗号和空格分隔
    std::ostringstream oss;
    std::transform(
        types.begin(),
        types.end(),
        std::ostream_iterator<std::string>(oss, ", "),
        [](const std::string& s) -> std::string { return "\"" + s + "\""; });
    auto rc = oss.str();
    rc.erase(rc.length() - 2); // 移除最后的逗号和空格
    // 返回格式化后的字符串列表表示
    return "[" + rc + "]";
  }
}

// 将IValue值列表转换为字符串表示
std::string ivalueListToStr(const std::vector<c10::IValue>& list) {
  // 初始化存储具体字符串输入的向量
  std::vector<std::string> concrete_str_inputs;
  // 初始化字符串流
  std::stringstream ss;
  // 遍历值列表中的每个值
  for (const auto& val : list) {
    // 如果值是None，则将空字符串添加到具体字符串输入向量中
    if (val.isNone()) {
      concrete_str_inputs.emplace_back("");
    } else {
      // 如果不是数字，将其转换为字符串并存储到 concrete_str_inputs 向量中
      ss.str("");  // 清空 stringstream
      ss << val;   // 将 val 转换为字符串并写入 stringstream
      concrete_str_inputs.emplace_back(ss.str());  // 将 stringstream 中的字符串存入向量 concrete_str_inputs
    }
  }
  // 调用 strListToStr 函数，将 concrete_str_inputs 向量中的所有字符串拼接成一个字符串返回
  return strListToStr(concrete_str_inputs);
}

// 关闭匿名命名空间

std::vector<std::string> inputTypes(const at::RecordFunction& fn) {
  // 创建一个空的字符串向量，用于存储输入的类型信息
  std::vector<std::string> types;
  // 预留足够的空间以避免多次分配
  types.reserve(fn.inputs().size());
  // 遍历记录函数的输入参数
  for (const c10::IValue& input : fn.inputs()) {
    // 如果输入是张量
    if (input.isTensor()) {
      // 转换输入为张量类型
      const at::Tensor& tensor = input.toTensor();
      // 如果张量已定义
      if (tensor.defined()) {
        // 将张量的数据类型名称转换为字符串并添加到类型列表中
        types.push_back(static_cast<std::string>(input.toTensor().dtype().name()));
      } else {
        // 否则添加一个空字符串到类型列表中
        types.emplace_back();
      }
    } else if (input.isScalar() || input.isList()) {
      // 如果输入是标量或列表，直接添加其类型标签到类型列表中
      types.push_back(input.tagKind());
    } else {
      // 对于其他类型的输入，添加一个空字符串到类型列表中
      types.emplace_back();
    }
  }
  // 返回收集到的类型信息列表
  return types;
}

// ----------------------------------------------------------------------------
// -- NCCL Metadata -----------------------------------------------------------
// ----------------------------------------------------------------------------

// 定义截断长度常量
static constexpr int32_t kTruncatLength = 30;

// 格式化列表类型的数据为字符串
template <typename ListLikeType>
inline std::string format_list(ListLikeType list, bool truncate) {
  // 如果开启截断并且列表长度超过截断长度
  if (truncate && list.size() > kTruncatLength) {
    // 使用格式化工具将列表部分转换为字符串并截断，返回格式化后的字符串
    return fmt::format(
        "\"[{}, ...]\"",
        fmt::join(list.begin(), list.begin() + kTruncatLength, ", "));
  }
  // 否则将整个列表转换为字符串并返回
  return fmt::format("\"[{}]\"", fmt::join(list.begin(), list.end(), ", "));
}

// 保存 NCCL 元数据到映射中
std::unordered_map<std::string, std::string> saveNcclMeta(
    const at::RecordFunction& fn,
    bool truncate) {
  // 创建一个无序映射用于保存 NCCL 元数据
  std::unordered_map<std::string, std::string> map;
  // 如果编译时使用了分布式相关宏
#ifdef USE_DISTRIBUTED
  // 获取当前线程的调试信息中的参数通信调试信息
  auto debugInfo = dynamic_cast<ParamCommsDebugInfo*>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PARAM_COMMS_INFO));
  // 如果调试信息为空
  if (debugInfo == nullptr) {
    // 记录警告信息并返回空映射
    LOG(WARNING) << "ParamCommsDebugInfo not available for function: "
                 << fn.name();
    return map;
  }

  // 将集合名称添加到映射中
  map.emplace(
      kCommsName, fmt::format("\"{}\"", debugInfo->getCollectiveName()));
  // 将数据类型名称添加到映射中
  map.emplace(
      kDtype, fmt::format("\"{}\"", c10::toString(debugInfo->getDType())));
  // 将输入消息元素数量添加到映射中
  map.emplace(kInMsgNelems, std::to_string(debugInfo->getInMessageNelems()));
  // 将输出消息元素数量添加到映射中
  map.emplace(kOutMsgNelems, std::to_string(debugInfo->getOutMessageNelems()));

  // 获取输入分割大小并格式化为字符串后添加到映射中
  auto& inSplitSizes = debugInfo->getInputSplitSizes();
  map.emplace(kInSplit, format_list(inSplitSizes, truncate));

  // 获取输出分割大小并格式化为字符串后添加到映射中
  auto& outSplitSizes = debugInfo->getOutputSplitSizes();
  map.emplace(kOutSplit, format_list(outSplitSizes, truncate));

  // 获取全局排名起始值，如果非负则添加到映射中
  auto globalRankStart = debugInfo->getGlobalRankStart();
  if (globalRankStart >= 0) {
    map.emplace(kGlobalRankStart, std::to_string(globalRankStart));
  }
  // 获取全局排名步长，如果大于零则添加到映射中
  auto globalRankStride = debugInfo->getGlobalRankStride();
  if (globalRankStride > 0) {
    map.emplace(kGlobalRankStride, std::to_string(globalRankStride));
  }
  // 将组大小添加到映射中
  map.emplace(kGroupSize, std::to_string(debugInfo->getWorldSize()));
  // 获取进程组名称，如果非空则添加到映射中
  auto& group_name = debugInfo->getProcessGroupName();
  if (!group_name.empty()) {
    map.emplace(kProcessGroupName, fmt::format("\"{}\"", group_name));
  }
  // 获取进程组描述，如果非空则添加到映射中
  auto& group_desc = debugInfo->getProcessGroupDesc();
  if (!group_desc.empty()) {
    map.emplace(kProcessGroupDesc, fmt::format("\"{}\"", group_desc));

# 将 kProcessGroupDesc 和格式化后的 group_desc 插入 map 中

  }

# 结束 for 循环的闭合

  auto& groupRanks = debugInfo->getGroupRanks();

# 获取 debugInfo 中的 groupRanks 引用

  map.emplace(kGroupRanks, format_list(groupRanks, truncate));

# 将 kGroupRanks 和格式化后的 groupRanks 列表插入 map 中，使用 truncate 参数进行截断处理

  auto rank = debugInfo->getRank();

# 获取 debugInfo 中的 rank 值

  map.emplace(kRank, std::to_string(rank));

# 将 kRank 和 rank 的字符串表示插入 map 中
// 返回用于计算 FLOPS 的额外参数的映射
std::unordered_map<std::string, c10::IValue> saveExtraArgs(
    const at::RecordFunction& fn) {
  // 创建空映射以保存额外参数
  std::unordered_map<std::string, c10::IValue> map;
  // 获取记录函数的输入
  auto inputs = fn.inputs();
  // 获取函数名
  std::string fname(fn.name());

  // 如果输入为空，则返回空映射
  if (inputs.empty()) {
    return map;
  }

  // 如果函数名为 kConv2dOp
  if (fname == kConv2dOp) {
    // 获取输入的尺寸信息
    const auto inputSizes =
        getInputSizes(fname, kConv2dGroups + 1, inputs, {0, 1});
    // 如果尺寸信息为空，则返回空映射
    if (inputSizes.empty()) {
      return map;
    }
    // 检查输入张量的大小是否为4，如果不是，则输出警告并返回空映射
    if (inputSizes[1].size() != 4) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires a 4D kernel tensor.");
      return map;
    }
    // 将输入张量的大小作为IValue存入映射中
    map[kInputSize] = at::IValue(inputSizes[0]);
    map[kWeightSize] = at::IValue(inputSizes[1]);
    map[kStride] = inputs[kConv2dStride];
    map[kPadding] = inputs[kConv2dPadding];
    map[kDilation] = inputs[kConv2dDilation];
    map[kGroups] = inputs[kConv2dGroups];
  } else if (fname == kMMOp) {
    // 获取指定操作的输入大小，索引为0和1，并将其存入映射中
    const auto inputSizes = getInputSizes(fname, 2, inputs, {0, 1});
    if (inputSizes.empty()) {
      return map;
    }

    map[kMat1Size] = at::IValue(inputSizes[0]);
    map[kMat2Size] = at::IValue(inputSizes[1]);
  } else if (fname == kAddMMOp) {
    // 获取指定操作的输入大小，索引为0、1和2，并将第1、2个输入大小存入映射中
    const auto inputSizes = getInputSizes(fname, 3, inputs, {0, 1, 2});
    if (inputSizes.empty()) {
      return map;
    }
    // 粗略估算操作的FLOP计数，假设缩放因子alpha和beta都为1
    // 参考文献：http://www.netlib.org/lapack/lawnspdf/lawn41.pdf 中的表格3，SGEMM
    map[kMat1Size] = at::IValue(inputSizes[1]);
    map[kMat2Size] = at::IValue(inputSizes[2]);
  } else if (fname == kMulOp) {
    // 获取指定操作的输入大小，索引为0，并将其存入映射中
    const auto inputSizes = getInputSizes(fname, 1, inputs, {0});
    if (inputSizes.empty()) {
      return map;
    }
    map[kMatSize] = at::IValue(inputSizes[0]);
  } else if (fname == kAddOp) {
    // 获取指定操作的输入大小，索引为0，并将其存入映射中
    const auto inputSizes = getInputSizes(fname, 1, inputs, {0});
    if (inputSizes.empty()) {
      return map;
    }
    map[kMatSize] = at::IValue(inputSizes[0]);
  } else if (fname == kBMMOp) {
    // 获取指定操作的输入大小，索引为0和1，并将其存入映射中
    const auto inputSizes = getInputSizes(fname, 2, inputs, {0, 1});
    if (inputSizes.empty()) {
      return map;
    }

    map[kMat1Size] = at::IValue(inputSizes[0]);
    map[kMat2Size] = at::IValue(inputSizes[1]);
  } else if (fname == kBAddBMMOp) {
    // 获取指定操作的输入大小，索引为0、1和2，并将第1、2个输入大小存入映射中
    const auto inputSizes = getInputSizes(fname, 3, inputs, {0, 1, 2});
    if (inputSizes.empty()) {
      return map;
    }
    // 粗略估算操作的FLOP计数，假设缩放因子alpha和beta都为1
    // 参考文献：http://www.netlib.org/lapack/lawnspdf/lawn41.pdf 中的表格3，SGEMM
    map[kMat1Size] = at::IValue(inputSizes[1]);
    map[kMat2Size] = at::IValue(inputSizes[2]);
  }

  // 返回包含操作名称和输入大小信息的映射
  return map;
// 计算操作的浮点操作数（FLOPs）的函数，根据操作名和额外参数进行计算
uint64_t computeFlops(
    const std::string& op_name,  // 操作名，用于确定计算哪种操作的FLOPs
    const std::unordered_map<std::string, c10::IValue>& extra_args) {  // 额外参数的无序映射
  // 检查操作名是否为卷积操作
  if (op_name == kConv2dOp) {
    // 检查额外参数中是否包含所需的关键参数：input_size, weight_size, groups, padding, stride, dilation
    if (extra_args.find(kInputSize) == extra_args.end() ||
        extra_args.find(kWeightSize) == extra_args.end() ||
        extra_args.find(kGroups) == extra_args.end() ||
        extra_args.find(kPadding) == extra_args.end() ||
        extra_args.find(kStride) == extra_args.end() ||
        extra_args.find(kDilation) == extra_args.end()) {
      // 输出警告信息并返回0，因为计算卷积操作的FLOPs需要这些参数
      TORCH_WARN(
          "Calculating flops for aten::conv2d requires groups, padding, stride, dilation, input_size, and weight_size in saved arguments.");
      return 0;
    }
    // 获取参数的引用
    auto input_sizes_ref = extra_args.at(kInputSize);
    auto kernel_sizes_ref = extra_args.at(kWeightSize);
    auto groups_ref = extra_args.at(kGroups);
    auto padding_ref = extra_args.at(kPadding);
    auto stride_ref = extra_args.at(kStride);
    auto dilation_ref = extra_args.at(kDilation);
    
    // 检查输入和权重尺寸是否为整数列表
    if (!input_sizes_ref.isIntList() || !kernel_sizes_ref.isIntList()) {
      // 输出警告信息并返回0，因为计算卷积操作的FLOPs需要这些参数是整数列表
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires input and weight tensor sizes.");
      return 0;
    }
    // 检查填充、步幅、膨胀是否为整数列表
    if (!padding_ref.isIntList() || !stride_ref.isIntList() ||
        !dilation_ref.isIntList()) {
      // 输出警告信息并返回0，因为计算卷积操作的FLOPs需要这些参数是整数列表
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires padding, stride, and dilation values.");
      return 0;
    }

    // 将引用转换为对应的数据结构
    const auto input_sizes = input_sizes_ref.toDimVector();
    const auto kernel_sizes = kernel_sizes_ref.toDimVector();
    const uint64_t groups = groups_ref.toInt();
    const std::vector<int64_t> padding = padding_ref.toIntVector();
    const std::vector<int64_t> stride = stride_ref.toIntVector();
    const std::vector<int64_t> dilation = dilation_ref.toIntVector();
    
    // 检查输入和权重尺寸是否为4维
    if (input_sizes.size() != 4 || kernel_sizes.size() != 4) {
      // 输出警告信息并返回0，因为计算卷积操作的FLOPs需要输入和权重尺寸都为4维
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because both input and weight must be size 4.");
      return 0;
    }
    // 检查分组数是否大于0
    if (!groups) {
      // 输出警告信息并返回0，因为分组数必须大于0才能计算卷积操作的FLOPs
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because group size must not be 0.");
      return 0;
    }
    // 检查填充和膨胀的尺寸是否为2
    if (padding.size() != 2 || dilation.size() != 2) {
      // 输出警告信息并返回0，因为填充和膨胀的尺寸必须为2才能计算卷积操作的FLOPs
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because both padding and dilation must be size 2.");
      return 0;
    }
    // 检查步幅的尺寸是否为2，并且步幅值不能为0
    if (stride.size() != 2 || (stride[0] * stride[1] == 0)) {
      // 输出警告信息并返回0，因为步幅的尺寸必须为2且不能为0才能计算卷积操作的FLOPs
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because stride must be size 2 and cannot be 0.");
      return 0;
    }
    // 定义卷积操作的乘法因子
    const uint64_t conv2d_multiply_factor = 2;
    // 将输入尺寸解构为四个部分
    auto [minibatch, in_channels, input_h, input_w] = std::make_tuple(
        input_sizes[0], input_sizes[1], input_sizes[2], input_sizes[3]);
    // 使用 C++17 结构化绑定，从 kernel_sizes 数组中获取卷积核参数
    auto [out_channels, _, kernel_h, kernel_w] = std::make_tuple(
        kernel_sizes[0], kernel_sizes[1], kernel_sizes[2], kernel_sizes[3]);
    // 计算输出特征图的高度
    uint64_t output_h =
        (input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) /
            stride[0] +
        1;
    // 计算输出特征图的宽度
    uint64_t output_w =
        (input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) /
            stride[1] +
        1;

    // 计算卷积操作的浮点运算次数（FLOPs）
    return conv2d_multiply_factor * minibatch * output_h * output_w * kernel_h *
        kernel_w * in_channels * out_channels / groups;
  } else if (op_name == kMMOp || op_name == kAddMMOp) {
    // 检查是否有必要的矩阵尺寸参数
    if (extra_args.find(kMat1Size) == extra_args.end() ||
        extra_args.find(kMat2Size) == extra_args.end()) {
      // 若缺少参数则输出警告并返回零
      TORCH_WARN(
          "Calculating flops for ",
          op_name,
          " requires mat1_size and mat2_size in saved arguments.");
      return 0;
    }
    // 获取矩阵1和矩阵2的尺寸
    auto mat1_sizes_ref = extra_args.at(kMat1Size);
    auto mat2_sizes_ref = extra_args.at(kMat2Size);
    // 检查尺寸是否为整数列表类型
    if (!mat1_sizes_ref.isIntList() || !mat2_sizes_ref.isIntList()) {
      // 若尺寸类型不匹配则输出警告并返回零
      TORCH_WARN(
          "Failed to compute flops for op ",
          op_name,
          " because it requires mat1_size and mat2_size to be IntList.");
      return 0;
    }

    // 将矩阵尺寸转换为标准向量形式
    const auto mat1_size = mat1_sizes_ref.toDimVector();
    const auto mat2_size = mat2_sizes_ref.toDimVector();
    // 若矩阵1的尺寸为空，则返回零
    if (mat1_size.empty()) {
      return 0;
    }

    // 获取矩阵重叠维度
    int64_t overlap_dim = mat1_size.back();
    // 若重叠维度为零，则返回零
    if (overlap_dim == 0) {
      return 0;
    }

    // 定义乘法因子为2，用于乘积运算
    const uint64_t gemm_multiply_factor = 2;
    uint64_t flops = 1;
    // 计算乘积运算的浮点运算次数（FLOPs）
    for (int64_t dim : mat1_size) {
      flops *= dim;
    }
    flops /= overlap_dim;
    for (int64_t dim : mat2_size) {
      flops *= dim;
    }
    flops *= gemm_multiply_factor;
    return flops;
  } else if (op_name == kBMMOp || op_name == kBAddBMMOp) {
    // 检查是否有必要的矩阵尺寸参数
    if (extra_args.find(kMat1Size) == extra_args.end() ||
        extra_args.find(kMat2Size) == extra_args.end()) {
      // 若缺少参数则输出警告并返回零
      TORCH_WARN(
          "Calculating flops for ",
          op_name,
          " requires mat1_size and mat2_size in saved arguments.");
      return 0;
    }
    // 获取矩阵1和矩阵2的尺寸
    auto mat1_sizes_ref = extra_args.at(kMat1Size);
    auto mat2_sizes_ref = extra_args.at(kMat2Size);
    // 检查尺寸是否为整数列表类型
    if (!mat1_sizes_ref.isIntList() || !mat2_sizes_ref.isIntList()) {
      // 若尺寸类型不匹配则输出警告并返回零
      TORCH_WARN(
          "Failed to compute flops for op ",
          op_name,
          " because it requires mat1_size and mat2_size to be IntList.");
      return 0;
    }

    // 将矩阵尺寸转换为标准向量形式
    const auto mat1_size = mat1_sizes_ref.toDimVector();
    const auto mat2_size = mat2_sizes_ref.toDimVector();
    // 若矩阵1的尺寸为空，则返回零
    if (mat1_size.empty()) {
      return 0;
    }

    // 获取矩阵批量大小
    int64_t batch_size = mat1_size.front();
    // 若批量大小为零，则返回零
    if (batch_size == 0) {
      return 0;
    }

    // 获取矩阵重叠维度
    int64_t overlap_dim = mat1_size.back();
    // 若重叠维度为零，则返回零
    if (overlap_dim == 0) {
      return 0;
    }

    // 定义乘法因子为2，用于乘积运算
    const uint64_t gemm_multiply_factor = 2;
    uint64_t flops = 1;
    // 计算乘积运算的浮点运算次数（FLOPs）
    for (int64_t dim : mat1_size) {
      flops *= dim;
    }
    flops /= overlap_dim;
    flops /= batch_size;
    // 遍历 mat2_size 中的每个维度，计算总的乘法操作数（FLOPs）
    for (int64_t dim : mat2_size) {
      flops *= dim;
    }
    // 将乘法操作数乘以 gemm_multiply_factor，得到最终的 FLOPs
    flops *= gemm_multiply_factor;
    // 返回计算得到的 FLOPs
    return flops;
  } else if (op_name == kMulOp) {
    // 检查额外参数中是否包含 mat_size
    if (extra_args.find(kMatSize) == extra_args.end()) {
      // 如果没有找到 mat_size，则发出警告并返回 0
      TORCH_WARN(
          "Calculating flops for aten::mul.Tensor requires mat_size in saved arguments.");
      return 0;
    }
    // 获取 mat_size 参数
    auto mat_sizes = extra_args.at(kMatSize);
    // 检查 mat_size 是否为 IntList 类型
    if (!mat_sizes.isIntList()) {
      // 如果 mat_size 不是 IntList 类型，则发出警告并返回 0
      TORCH_WARN(
          "Failed to compute flops for op aten::mul because it requires mat_size to be IntList.");
      return 0;
    }

    // 将 mat_size 转换为 DimVector
    const auto mat_size = mat_sizes.toDimVector();
    uint64_t flops = 1;
    // 计算 mat_size 中每个维度的乘积，作为 FLOPs
    for (int64_t dim : mat_size) {
      flops *= dim;
    }
    // 返回计算得到的 FLOPs
    return flops;
  } else if (op_name == kAddOp) {
    // 检查额外参数中是否包含 mat_size
    if (extra_args.find(kMatSize) == extra_args.end()) {
      // 如果没有找到 mat_size，则发出警告并返回 0
      TORCH_WARN(
          "Calculating flops for aten::add.Tensor requires mat_size in saved arguments.");
      return 0;
    }
    // 获取 mat_size 参数
    auto mat_sizes = extra_args.at(kMatSize);
    // 检查 mat_size 是否为 IntList 类型
    if (!mat_sizes.isIntList()) {
      // 如果 mat_size 不是 IntList 类型，则发出警告并返回 0
      TORCH_WARN(
          "Failed to compute flops for op aten::add because it requires mat_size to be IntList.");
      return 0;
    }

    // 将 mat_size 转换为 DimVector
    const auto mat_size = mat_sizes.toDimVector();
    uint64_t flops = 1;
    // 计算 mat_size 中每个维度的乘积，作为 FLOPs
    for (int64_t dim : mat_size) {
      flops *= dim;
    }
    // 返回计算得到的 FLOPs
    return flops;
  }
  // 如果操作名称既不是 kGemmOp、kMulOp、也不是 kAddOp，则返回 0
  return 0;
}

} // namespace torch::profiler::impl
```