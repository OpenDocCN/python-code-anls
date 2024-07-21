# `.\pytorch\torch\csrc\cuda\memory_snapshot.cpp`

```
// 包含 ATen 库的 Context 头文件，用于管理运行时环境
#include <ATen/Context.h>
// 包含 ATen 库的 record_function 头文件，用于记录函数调用的信息
#include <ATen/record_function.h>
// 包含 c10 库中 CUDA 缓存分配器的头文件，用于 CUDA 内存管理
#include <c10/cuda/CUDACachingAllocator.h>
// 包含 Torch 中 CUDA 内存快照功能的头文件
#include <torch/csrc/cuda/memory_snapshot.h>
// 包含 Torch 中 JIT 运行时解释器的头文件
#include <torch/csrc/jit/runtime/interpreter.h>
// 包含 Torch 中 JIT 序列化 Pickler 的头文件
#include <torch/csrc/jit/serialization/pickler.h>
// 包含 Torch 中分析器的组合回溯功能的头文件
#include <torch/csrc/profiler/combined_traceback.h>

// 定义 torch::cuda 命名空间
namespace torch::cuda {

// 使用 c10 中的 Dict 类型和 IValue 类型
using c10::Dict;
using c10::IValue;
// 使用 Torch 中的 Pickler 类型
using torch::jit::Pickler;

// 使用 c10 中 CUDA 缓存分配器的 SegmentInfo 类型
using c10::cuda::CUDACachingAllocator::SegmentInfo;

// 匿名命名空间，用于封装私有函数和静态变量
namespace {

// 将 IValue 对象序列化为 Pickle 格式的字符串
std::string write_pickle(const IValue& v) {
  std::vector<char> result;
  {
    // lambda 函数，将数据写入 result 向量中
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    // 创建 Pickler 对象，指定 writer 和其他选项
    Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();  // 设置 Pickler 协议版本
    pickler.pushIValue(v);  // 将 IValue 对象推入 Pickler 中
    pickler.stop();  // 结束 Pickler 操作
  }
  return std::string(result.begin(), result.end());  // 将结果转换为字符串返回
}

// 创建并返回新的空字典，键值类型为任意类型的 IValue
Dict<IValue, IValue> new_dict() {
  return Dict<IValue, IValue>(c10::AnyType::get(), c10::AnyType::get());
}

// 创建并返回新的空列表，元素类型为任意类型的 IValue
c10::List<IValue> new_list() {
  return List<IValue>(c10::AnyType::get());
}

// 将 CapturedTraceback 对象列表转换为 IValue 对象列表
std::vector<IValue> ivalue_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  // 使用 unordered_map 缓存重复的 CapturedTraceback 指针及其索引
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  // 遍历 to_symbolize 列表，去重并添加到 unique_frames 中
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  // 对 unique_frames 中的 CapturedTraceback 对象进行符号化
  auto s = symbolize(unique_frames);

  // 创建用于表示帧信息的 IValue 对象
  IValue line_s = "line";
  IValue name_s = "name";
  IValue filename_s = "filename";
  std::vector<IValue> all_frames;
  // 遍历所有符号化后的帧信息，将其添加到 all_frames 中
  for (const auto& f : s.all_frames) {
    auto d = new_dict();
    d.insert(name_s, f.funcname);
    d.insert(filename_s, f.filename);
    d.insert(line_s, int64_t(f.lineno));
    all_frames.emplace_back(std::move(d));
  }

  // 创建用于表示 Python 唯一帧信息的 IValue 对象列表
  std::vector<IValue> py_unique_frames;
  // 遍历所有符号化后的追踪信息，将其添加到 py_unique_frames 中
  for (const auto& t : s.tracebacks) {
    auto l = new_list();
    for (const auto& e : t) {
      l.push_back(all_frames.at(e));
    }
    py_unique_frames.emplace_back(std::move(l));
  }

  // 创建最终结果列表，包含所有 to_symbolize 对象的唯一帧信息
  std::vector<IValue> result;
  result.reserve(to_symbolize.size());
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;  // 返回结果列表
}

// 收集当前线程的上下文信息并返回一个 GatheredContext 对象的共享指针
std::shared_ptr<c10::GatheredContext> gather() {
  return CapturedTraceback::gather(true, true, false);
}

// 收集当前线程的上下文信息（包括 C++ 栈）并返回 GatheredContext 对象的共享指针
std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
  return CapturedTraceback::gather(true, true, true);
}

// 从 GatheredContext 对象中获取 CapturedTraceback 指针
CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  // 检查并返回正确类型的 CapturedTraceback 指针，否则抛出异常
  if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

// 初始化记录注释的函数
void _initRecordAnnotations() {
  static c10::once_flag ra_init;  // 静态初始化一次标志
  // 仅在第一次调用时执行以下 lambda 函数
  c10::call_once(ra_init, [&] {
    // 将用户的注释保存到 CCA 内存快照工具中

    // 添加线程局部回调函数，记录函数执行
    at::addThreadLocalCallback(at::RecordFunctionCallback(
        // Lambda 函数，接收 RecordFunction 对象，并返回 ObserverContext 对象的唯一指针
        [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          // 如果函数的作用域不是用户定义的范围，则返回空指针，仅记录用户定义的作用域
          if (fn.scope() != at::RecordScope::USER_SCOPE) {
            return nullptr;
          }
          // 创建 unwind::Frame 对象，表示函数执行的帧信息
          unwind::Frame frame{fn.name(), "START", 0};
          // 创建 CapturedTraceback 对象的共享指针
          auto r = std::make_shared<CapturedTraceback>();
          // 记录用户定义的帧信息到 CapturedTraceback 对象中
          r->recordUserDefinedFrame(frame);
          // 调用 CUDACachingAllocator 的 recordAnnotation 方法，记录注释
          c10::cuda::CUDACachingAllocator::recordAnnotation(r);
          // 返回空指针，表示不需要进一步的 ObserverContext 对象
          return nullptr;
        },
        // Lambda 函数，记录函数执行结束时的操作
        [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
          // 如果函数的作用域不是用户定义的范围，则直接返回
          if (fn.scope() != at::RecordScope::USER_SCOPE) {
            return;
          }
          // 创建 unwind::Frame 对象，表示函数执行的帧信息
          unwind::Frame frame{fn.name(), "END", 0};
          // 创建 CapturedTraceback 对象的共享指针
          auto r = std::make_shared<CapturedTraceback>();
          // 记录用户定义的帧信息到 CapturedTraceback 对象中
          r->recordUserDefinedFrame(frame);
          // 调用 CUDACachingAllocator 的 recordAnnotation 方法，记录注释
          c10::cuda::CUDACachingAllocator::recordAnnotation(r);
        }));
  });
} // namespace

// 记录内存历史的函数，根据不同的选项设置内存记录
void _record_memory_history(
    bool enabled, // 是否启用内存记录
    bool record_context, // 是否记录上下文
    int64_t trace_alloc_max_entries, // 最大分配记录条目数
    bool trace_alloc_record_context, // 是否记录分配上下文
    bool record_cpp_context) { // 是否记录 C++ 上下文
  c10::cuda::CUDACachingAllocator::CreateContextFn recorder = gather; // 默认使用 gather 函数作为记录器
  if (enabled && record_cpp_context) {
    recorder = gather_with_cpp; // 如果启用并且记录 C++ 上下文，则使用 gather_with_cpp 函数
    // 预热 C++ 栈展开
    unwind::unwind();
  }
  auto when = c10::cuda::CUDACachingAllocator::RecordContext::NEVER; // 默认情况下不记录上下文
  if (trace_alloc_record_context) {
    when = c10::cuda::CUDACachingAllocator::RecordContext::ALLOC; // 如果记录分配上下文，则在分配时记录
  } else if (record_context) {
    when = c10::cuda::CUDACachingAllocator::RecordContext::STATE; // 否则在状态改变时记录
  }
  at::globalContext().lazyInitCUDA(); // 初始化全局 CUDA 上下文
  _initRecordAnnotations(); // 初始化记录注释
  // 记录内存历史
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled, recorder, trace_alloc_max_entries, when);
}

// 检查选项是否在有效选项列表中
static void checkOptionIn(
    const std::string& option, // 要检查的选项
    std::initializer_list<std::string> valid, // 有效选项列表
    const char* error) { // 错误消息
  TORCH_CHECK(
      valid.end() != std::find(valid.begin(), valid.end(), option), error); // 如果选项不在有效列表中则抛出错误
}

// 记录内存历史的函数重载，接受字符串可选值来设置不同的选项
void _record_memory_history(
    std::optional<std::string> enabled, // 启用内存记录的可选值
    std::optional<std::string> context, // 上下文记录的可选值
    const std::string& stacks, // 堆栈信息
    size_t max_entries) { // 最大条目数
  if (enabled) {
    checkOptionIn(
        *enabled,
        {"state", "all"},
        "expected state to be 'state', 'all', or None"); // 检查启用选项是否在有效状态中
  }
  if (context) {
    checkOptionIn(
        *context,
        {"state", "alloc", "all"},
        "expected context to be 'state', 'alloc', 'all', or None"); // 检查上下文选项是否在有效状态中
  }
  checkOptionIn(
      stacks, {"python", "all"}, "expected stacks to be 'python', or 'all'"); // 检查堆栈信息是否在有效状态中

  c10::cuda::CUDACachingAllocator::CreateContextFn recorder = gather; // 默认使用 gather 函数作为记录器
  if (enabled && stacks == "all") {
    recorder = gather_with_cpp; // 如果启用并且堆栈信息为 'all'，则使用 gather_with_cpp 函数
    // 预热 C++ 栈展开
    unwind::unwind();
  }
  max_entries = (enabled && *enabled == "all") ? max_entries : 1; // 如果启用并且值为 'all'，则最大条目数为传入的 max_entries
  auto when = c10::cuda::CUDACachingAllocator::RecordContext::NEVER; // 默认情况下不记录上下文
  if (context) {
    if (*context == "all") {
      when = c10::cuda::CUDACachingAllocator::RecordContext::ALL; // 如果上下文为 'all'，则记录所有情况
    } else if (*context == "alloc") {
      when = c10::cuda::CUDACachingAllocator::RecordContext::ALLOC; // 如果上下文为 'alloc'，则在分配时记录
    } else if (*context == "state") {
      when = c10::cuda::CUDACachingAllocator::RecordContext::STATE; // 如果上下文为 'state'，则在状态改变时记录
    }
  }
  at::globalContext().lazyInitCUDA(); // 初始化全局 CUDA 上下文
  _initRecordAnnotations(); // 初始化记录注释
  // 记录内存历史
  c10::cuda::CUDACachingAllocator::recordHistory(
      enabled.has_value(), recorder, max_entries, when);
}
std::string _memory_snapshot_pickled() {
  // 初始化一系列字符串常量，用于后续创建字典时的键
  IValue device_s = "device";
  IValue address_s = "address";
  IValue total_size_s = "total_size";
  IValue allocated_size_s = "allocated_size";
  IValue active_size_s = "active_size";
  IValue requested_size_s = "requested_size";
  IValue stream_s = "stream";
  IValue segment_type_s = "segment_type";
  IValue segment_pool_id = "segment_pool_id";
  IValue large_s = "large";
  IValue small_s = "small";
  IValue size_s = "size";
  IValue state_s = "state";
  IValue active_allocated_s = "active_allocated";
  IValue active_pending_free_s = "active_pending_free";
  IValue inactive_s = "inactive";
  IValue addr_s = "addr";
  IValue filename_s = "filename";
  IValue name_s = "name";
  IValue line_s = "line";
  IValue frames_s = "frames";
  IValue blocks_s = "blocks";
  IValue is_expandable_s = "is_expandable";
  IValue time_us_s = "time_us";

  // 创建一个空的帧列表
  auto empty_frames = new_list();

  // 定义存储帧回溯和帧字典的向量
  std::vector<CapturedTraceback*> frame_tracebacks;
  std::vector<Dict<IValue, IValue>> frame_dict;

  // 定义 lambda 函数 add_frame_key，用于将帧字典和帧回溯添加到对应的向量中
  auto add_frame_key = [&](const c10::Dict<IValue, IValue>& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      // 如果上下文存在，从上下文获取帧回溯并添加到向量中
      frame_tracebacks.push_back(getFromContext(ctx));
      // 将帧字典添加到帧字典向量中
      frame_dict.push_back(d);
    } else {
      // 如果上下文不存在，将空的帧列表插入帧字典中
      d.insert(frames_s, empty_frames);
    }
  };

  // 定义 lambda 函数 segmentInfoToDict，将 SegmentInfo 转换为字典形式
  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    // 创建一个新的字典用于存储 SegmentInfo 的信息
    auto segmentDict = new_dict();
    // 向字典中插入 SegmentInfo 的各项信息
    segmentDict.insert(device_s, segmentInfo.device);
    segmentDict.insert(address_s, static_cast<int64_t>(segmentInfo.address));
    segmentDict.insert(total_size_s, static_cast<int64_t>(segmentInfo.total_size));
    segmentDict.insert(allocated_size_s, static_cast<int64_t>(segmentInfo.allocated_size));
    segmentDict.insert(active_size_s, static_cast<int64_t>(segmentInfo.active_size));
    segmentDict.insert(requested_size_s, static_cast<int64_t>(segmentInfo.requested_size));
    segmentDict.insert(stream_s, int64_t(segmentInfo.stream));
    segmentDict.insert(segment_type_s, (segmentInfo.is_large ? large_s : small_s));
    segmentDict.insert(segment_pool_id, std::tuple<int64_t, int64_t>(segmentInfo.owner_private_pool_id));
    segmentDict.insert(is_expandable_s, segmentInfo.is_expandable);

    // 调用 add_frame_key 将帧字典和上下文信息添加到 frame_tracebacks 和 frame_dict 中
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    // 获取地址和块信息，并创建一个新的块列表
    auto address = segmentInfo.address;
    auto blocks = new_list();
    for (const auto& blockInfo : segmentInfo.blocks) {
      auto blockDict = new_dict();
      blockDict.insert(address_s, static_cast<int64_t>(address));
      blockDict.insert(size_s, static_cast<int64_t>(blockInfo.size));
      blockDict.insert(
          requested_size_s, static_cast<int64_t>(blockInfo.requested_size));
      blockDict.insert(
          state_s,
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s)));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      address += blockInfo.size;
      blocks.push_back(blockDict);
    }
    segmentDict.insert(blocks_s, blocks);


    // 遍历当前段的每个内存块信息
    for (const auto& blockInfo : segmentInfo.blocks) {
      // 创建一个新的字典用于存储内存块信息
      auto blockDict = new_dict();
      // 向字典中插入地址信息，将地址转换为int64_t类型
      blockDict.insert(address_s, static_cast<int64_t>(address));
      // 向字典中插入大小信息，将块的大小转换为int64_t类型
      blockDict.insert(size_s, static_cast<int64_t>(blockInfo.size));
      // 向字典中插入请求大小信息，将请求的大小转换为int64_t类型
      blockDict.insert(
          requested_size_s, static_cast<int64_t>(blockInfo.requested_size));
      // 向字典中插入状态信息，根据块的分配状态选择相应的字符串表示
      blockDict.insert(
          state_s,
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s)));
      // 如果有分配时的上下文信息，则添加到字典中
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      // 更新地址，以便下一个块
      address += blockInfo.size;
      // 将块字典添加到块列表中
      blocks.push_back(blockDict);
    }
    // 将块列表插入段字典中
    segmentDict.insert(blocks_s, blocks);


    return segmentDict;
  };


    // 返回包含所有段信息的段字典
    return segmentDict;
  };

  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();


  // 获取当前 CUDA 缓存分配器的快照
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();


  auto segments = new_list();


  // 创建一个新的列表用于存储所有的段信息
  auto segments = new_list();


  for (const auto& segmentInfo : snapshot.segments) {
    segments.push_back(segmentInfoToDict(segmentInfo));
  }


  // 遍历快照中的每个段信息，将其转换为字典并添加到段列表中
  for (const auto& segmentInfo : snapshot.segments) {
    segments.push_back(segmentInfoToDict(segmentInfo));
  }


  auto traces = new_list();


  // 创建一个新的列表用于存储所有的跟踪信息
  auto traces = new_list();


  IValue action_s = "action";
  IValue alloc_s = "alloc";
  IValue free_requested_s = "free_requested";
  IValue free_completed_s = "free_completed";
  IValue segment_alloc_s = "segment_alloc";
  IValue segment_free_s = "segment_free";
  IValue segment_map_s = "segment_map";
  IValue segment_unmap_s = "segment_unmap";
  IValue snapshot_s = "snapshot";
  IValue oom_s = "oom";
  IValue device_free_s = "device_free";
  IValue user_defined_s = "user_defined";


  // 定义字符串表示的不同跟踪动作类型
  IValue action_s = "action";
  IValue alloc_s = "alloc";
  IValue free_requested_s = "free_requested";
  IValue free_completed_s = "free_completed";
  IValue segment_alloc_s = "segment_alloc";
  IValue segment_free_s = "segment_free";
  IValue segment_map_s = "segment_map";
  IValue segment_unmap_s = "segment_unmap";
  IValue snapshot_s = "snapshot";
  IValue oom_s = "oom";
  IValue device_free_s = "device_free";
  IValue user_defined_s = "user_defined";


  using namespace c10::cuda::CUDACachingAllocator;


  // 使用 CUDA 缓存分配器的命名空间
  using namespace c10::cuda::CUDACachingAllocator;


  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
      case TraceEntry::USER_DEFINED:
        return user_defined_s;
    }
    throw std::runtime_error("unreachable");
  };


  // 定义一个函数，将跟踪动作类型转换为对应的字符串表示
  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
      case TraceEntry::USER_DEFINED:
        return user_defined_s;
    }
    throw std::runtime_error("unreachable");
  };


  for (const auto& traceInfo : snapshot.device_traces) {
    auto trace = new_list();
    for (const auto& te : traceInfo) {
      auto trace_entry = new_dict();
      trace_entry.insert(action_s, action_to_str(te.action_));
      trace_entry.insert(
          TraceEntry::OOM == te.action_ ? device_free_s : addr_s,
          static_cast<int64_t>(te.addr_));
      trace_entry.insert(size_s, (int64_t)te.size_);
      trace_entry.insert(stream_s, int64_t(te.stream_));
      if (te.context_) {
        auto sc = getFromContext(te.context_);
        frame_tracebacks.push_back(sc);
        frame_dict.push_back(trace_entry);
      }
      trace_entry.insert(time_us_s, te.time_.t_);
      trace.push_back(trace_entry);
    }


    // 遍历快照中的每个设备跟踪信息
    for (const auto& traceInfo : snapshot.device_traces) {
      // 创建一个新的列表用于存储跟踪条目
      auto trace = new_list();
      // 遍历每个跟踪条目
      for (const auto& te : traceInfo) {
        // 创建一个新的字典用于存储跟踪条目信息
        auto trace_entry = new_dict();
        // 向字典中插入动作类型的字符串表示
        trace_entry.insert(action_s, action_to_str(te.action_));
        // 根据跟踪动作类型选择插入地址或设备释放字符串表示
        trace_entry.insert(
            TraceEntry::OOM == te.action_ ? device_free_s : addr_s,
            static_cast<int64_t>(te.addr_));
        // 向字典中插入条目大小
        trace_entry.insert(size_s, (int64_t)te.size_);
        // 向字典中插入流信息
        trace_entry.insert(stream_s, int64_t(te.stream_));
        // 如果有上下文信息，则添加到相应的列表中
        if (te.context_) {
          auto sc = getFromContext(te.context_);
          frame_tracebacks.push_back(sc);
          frame_dict.push_back(trace_entry);
        }
        // 向字典中插入时间戳信息
        trace_entry.insert(time_us_s
  // 将 trace 添加到 traces 向量中
  traces.push_back(trace);

  // 创建一个新的字典 allocator_settings
  auto allocator_settings = new_dict();

  // 定义各个配置项的字符串键
  IValue last_allocator_settings_s = "PYTORCH_CUDA_ALLOC_CONF";
  IValue max_split_size_s = "max_split_size";
  IValue garbage_collection_threshold_s = "garbage_collection_threshold";
  IValue expandable_segments_s = "expandable_segments";
  IValue pinned_num_register_threads_s = "pinned_num_register_threads";
  IValue release_lock_on_malloc_s = "release_lock_on_cudamalloc";
  IValue pinned_use_host_register_s = "pinned_use_cuda_host_register";
  IValue roundup_power2_divisions_s = "roundup_power2_divisions";

  // 将各个配置项及其值插入 allocator_settings 字典中
  allocator_settings.insert(
      last_allocator_settings_s,
      snapshot.config_metadata.last_allocator_settings);
  allocator_settings.insert(
      max_split_size_s, int64_t(snapshot.config_metadata.max_split_size));
  allocator_settings.insert(
      garbage_collection_threshold_s,
      snapshot.config_metadata.garbage_collection_threshold);
  allocator_settings.insert(
      expandable_segments_s, snapshot.config_metadata.expandable_segments);
  allocator_settings.insert(
      pinned_num_register_threads_s,
      int64_t(snapshot.config_metadata.pinned_num_register_threads));
  allocator_settings.insert(
      release_lock_on_malloc_s,
      snapshot.config_metadata.release_lock_on_malloc);
  allocator_settings.insert(
      pinned_use_host_register_s,
      snapshot.config_metadata.pinned_use_host_register);

  // 初始化 roundup_key 为 1，创建新的字典 roundup_settings
  unsigned int roundup_key = 1;
  auto roundup_settings = new_dict();

  // 遍历 snapshot.config_metadata.roundup_power2_divisions 中的每个值，并插入到 roundup_settings 中
  for (const auto& v : snapshot.config_metadata.roundup_power2_divisions) {
    IValue roundup_key_s = std::to_string(roundup_key);
    roundup_settings.insert(roundup_key_s, int64_t(v));
    roundup_key *= 2;
  }

  // 将 roundup_settings 插入 allocator_settings 中
  allocator_settings.insert(roundup_power2_divisions_s, roundup_settings);

  // 创建一个新的字典 result，并插入 segments、traces 和 allocator_settings
  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);
  result.insert("allocator_settings", allocator_settings);

  // 使用 ivalue_symbolize 对 frame_tracebacks 进行符号化，并将结果插入到 frame_dict 中的每个元素中
  auto frames = ivalue_symbolize(frame_tracebacks);
  for (auto i : c10::irange(frames.size())) {
    frame_dict.at(i).insert(frames_s, frames.at(i));
  }

  // 返回使用 write_pickle 序列化 result 的结果
  return write_pickle(result);
}
} // namespace torch::cuda
```