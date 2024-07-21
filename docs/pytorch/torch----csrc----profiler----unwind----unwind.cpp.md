# `.\pytorch\torch\csrc\profiler\unwind\unwind.cpp`

```
// 包含C++异常处理库的头文件
#include <c10/util/Exception.h>
// 包含用于性能分析的栈回溯库的头文件
#include <torch/csrc/profiler/unwind/unwind.h>
// 包含C++中的堆栈跟踪工具的头文件
#include <torch/csrc/utils/cpp_stacktraces.h>
// 包含标准库中的无序映射容器
#include <unordered_map>

// 如果不是在Linux平台或非x86_64架构，或者缺少特定头文件，则定义以下内容
#if !defined(__linux__) || !defined(__x86_64__) || !defined(__has_include) || \
    !__has_include("ext/stdio_filebuf.h")

// 在torch::unwind命名空间下定义以下内容
namespace torch::unwind {

// 对于非Linux非x86_64平台，定义一个空的unwind函数，返回空的void指针向量
std::vector<void*> unwind() {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

// 对于非Linux非x86_64平台，定义一个空的libraryFor函数，返回空的名称和地址对
std::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr) {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

// 对于非Linux非x86_64平台，定义一个空的symbolize函数，返回空的Frame向量
#ifndef FBCODE_CAFFE2
std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode) {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}
#endif

// 对于非Linux非x86_64平台，定义一个空的stats函数，返回空的Stats对象
Stats stats() {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

} // namespace torch::unwind

// 如果是在Linux平台且x86_64架构，包含以下内容
#else

// 包含C++标准库中的扁平哈希映射头文件
#include <c10/util/flat_hash_map.h>
// 包含ELF文件头文件
#include <elf.h>
// 包含链接器头文件
#include <link.h>
// 包含Linux系统路径限制头文件
#include <linux/limits.h>
// 包含算法标准库头文件
#include <algorithm>
// 包含C标准库中的限制头文件
#include <climits>
// 包含向量容器的标准库头文件
#include <vector>

// 包含C++工具库中的范围迭代头文件
#include <c10/util/irange.h>
// 包含C++标准库的ABI头文件
#include <cxxabi.h>
// 包含性能分析中的通信头文件
#include <torch/csrc/profiler/unwind/communicate.h>
// 包含性能分析中的DWARF枚举头文件
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
// 包含性能分析中的EH框架头文件
#include <torch/csrc/profiler/unwind/eh_frame_hdr.h>
// 包含性能分析中的快速符号化器头文件
#include <torch/csrc/profiler/unwind/fast_symbolizer.h>
// 包含性能分析中的FDE头文件
#include <torch/csrc/profiler/unwind/fde.h>
// 包含性能分析中的解绑器头文件
#include <torch/csrc/profiler/unwind/unwinder.h>
// 包含共享互斥头文件
#include <shared_mutex>

// 声明C函数，用于C++调用，用于执行C风格的栈展开操作
extern "C" void unwind_c(std::vector<void*>* result, int64_t rsp, int64_t rbp);
// 声明C函数，用于C++调用，用于执行C风格的栈入口展开操作
extern "C" void unwind_entry(std::vector<void*>* result);

// 在torch::unwind命名空间下定义以下内容
namespace torch::unwind {

// 定义一个帮助升级共享互斥的结构体，允许从读锁升级为独占锁
struct UpgradeExclusive {
  UpgradeExclusive(std::shared_lock<std::shared_timed_mutex>& rdlock)
      : rdlock_(rdlock) {
    // 解锁读锁，锁定互斥锁
    rdlock_.unlock();
    rdlock_.mutex()->lock();
  }
  // 析构函数，释放互斥锁，恢复读锁
  ~UpgradeExclusive() {
    rdlock_.mutex()->unlock();
    rdlock_.lock();
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::shared_lock<std::shared_timed_mutex>& rdlock_;
};

// 定义一个库信息结构体，包含库名称、加载偏移、最后地址和EH框架指针
struct LibraryInfo {
  LibraryInfo(
      std::string name,
      uint64_t load_bias,
      uint64_t last_addr,
      void* eh_frame_hdr_ptr_)
      : name_(std::move(name)),
        load_bias_(load_bias),
        last_addr_(last_addr),
        eh_frame_hdr_(eh_frame_hdr_ptr_) {}

  // 返回加载偏移量
  uint64_t load_bias() const {
    return load_bias_;
  }
  // 返回最后地址
  uint64_t last_addr() const {
    return last_addr_;
  }
  // 根据地址返回Unwinder对象
  Unwinder unwinderFor(uint64_t addr) const {
    // 获取地址对应的FDE数据
    void* fde_data = eh_frame_hdr_.entryForAddr(addr);
    // 根据FDE数据创建FDE对象
    FDE fde(fde_data, name().c_str(), load_bias());
    // 读取FDE数据直到指定地址，返回Unwinder对象
    TableState state = fde.readUpTo(addr);
    return Unwinder(state.cfa, state.registers[D_RIP], state.registers[D_RBP]);
  }
  // 返回库名称的引用
  const std::string& name() const {
    return name_;
  }

 private:
  std::string name_; // 库名称
  uint64_t load_bias_; // 加载偏移量
  uint64_t last_addr_; // 最后地址
  EHFrameHdr eh_frame_hdr_; // EH框架头部对象
};
// 返回当前进程的执行文件路径作为静态变量，如果尚未获取，则从 /proc/self/exe 读取
static const char* process_name() {
  // NOLINTNEXTLINE(*-c-arrays*)
  static char name[PATH_MAX + 1] = "";
  // 如果 name 还未被填充
  if (*name == '\0') {
    // 读取当前进程可执行文件的符号链接路径，并将结果存储在 name 中
    ssize_t len = readlink("/proc/self/exe", name, PATH_MAX);
    // 确保读取成功，否则断言失败并输出错误信息
    TORCH_INTERNAL_ASSERT(len != -1, "can't get path to exe")
    // 将字符串末尾设置为 '\0'，以确保字符串结尾正确
    name[len] = '\0';
  }
  // 返回静态变量 name 的地址
  return name;
}

// 版本信息结构体，包含添加和删除操作的计数
struct Version {
  uint64_t adds_ = LONG_LONG_MAX; // 添加操作次数的初始值
  uint64_t subs_ = LONG_LONG_MAX; // 删除操作次数的初始值
};

// 拆卸缓存类，用于管理库的信息和版本
struct UnwindCache {
  // 获取当前版本信息
  Version currentVersion() {
    Version r;
    // 遍历动态链接库信息，将添加和删除操作次数记录到 Version 结构体中
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t size, void* data) {
          Version* v = (Version*)data;
          v->adds_ = info->dlpi_adds; // 记录添加操作次数
          v->subs_ = info->dlpi_subs; // 记录删除操作次数
          return 1;
        },
        &r);
    return r; // 返回记录的版本信息
  }

  // 刷新库信息
  void refreshLibraries() {
    ++stats_.resets; // 增加重置计数

    all_libraries_.clear(); // 清空所有库信息
    ip_cache_.clear(); // 清空指令指针缓存

    // 遍历动态链接库信息
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t size, void* data) {
          auto self = (UnwindCache*)data;
          uint64_t last_addr = 0;
          auto segments = (Elf64_Phdr*)info->dlpi_phdr;

          // 遍历库的段信息
          for (auto i : c10::irange(info->dlpi_phnum)) {
            if (segments[i].p_type == PT_LOAD) {
              // 计算段的起始和结束地址
              auto begin = ((uint64_t)info->dlpi_addr + segments[i].p_vaddr);
              auto end = (begin + segments[i].p_memsz);
              // 更新最后的地址位置
              last_addr = std::max(end, last_addr);
            }
            if (segments[i].p_type == PT_GNU_EH_FRAME) {
              // 获取库的名称，如果名称为空则使用进程的名称
              std::string library_name = info->dlpi_name;
              if (library_name.empty()) {
                library_name = process_name();
              }
              // 获取异常处理帧头地址
              auto eh_frame_hdr =
                  // NOLINTNEXTLINE(performance-no-int-to-ptr)
                  (void*)(segments[i].p_vaddr + info->dlpi_addr);
              // 将库信息添加到库列表中
              self->all_libraries_.emplace_back(
                  std::move(library_name),
                  info->dlpi_addr,
                  last_addr,
                  eh_frame_hdr);
              return 0; // 结束遍历
            }
          }
          // 将没有异常处理帧的库添加到不支持取消异常处理的列表中
          self->libraries_with_no_unwind_.emplace_back(info->dlpi_name);
          return 0; // 结束遍历
        },
        this);

    // 对所有库信息按加载偏移量进行排序
    std::sort(
        all_libraries_.begin(),
        all_libraries_.end(),
        [](const LibraryInfo& lhs, const LibraryInfo& rhs) {
          return lhs.load_bias() < rhs.load_bias();
        });
  }

  // 检查是否需要刷新库信息
  void checkRefresh(std::shared_lock<std::shared_timed_mutex>& rdlock) {
    // 获取当前版本信息
    Version current_version = currentVersion();
    // 如果版本信息有更新
    if (current_version.subs_ != last_version_.subs_) {
      // 获取独占锁并刷新库信息
      UpgradeExclusive lock(rdlock);
      refreshLibraries();
    }
  }

  // 根据地址获取解析器对象
  const Unwinder& unwinderFor(
      uint64_t addr,
      std::shared_lock<std::shared_timed_mutex>& rdlock) {
    // 查找地址是否在指令指针缓存中
    auto it = ip_cache_.find(addr);
    // 如果找到，增加命中计数并返回对应的解析器
    if (it != ip_cache_.end()) {
      ++stats_.hits;
      return it->second;
    }

    // 即将修改缓存，获取独占锁
    UpgradeExclusive lock(rdlock);
    ++stats_.misses; // 增加未命中计数

    Unwinder unwinder = Unwinder::unknown(); // 默认为未知的解析器
    try {
      // 获取地址所在库的解析器对象
      unwinder = libraryFor(addr).unwinderFor(addr);
      ```
  } catch (unwind::UnwindError& err) {
    // 捕获 unwind::UnwindError 异常，表示无法展开帧。
    // 由于展开器是缓存的，因此这条消息每帧只打印一次。
    TORCH_WARN("Unsupported unwinding pattern: ", err.what());
  }
  // 尝试将地址与展开器插入或更新到 ip_cache_ 中
  auto r = ip_cache_.insert_or_assign(addr, unwinder);
  return r.first->second;
}

// 根据地址查找对应的库信息
const LibraryInfo* findLibraryFor(uint64_t addr) {
  // 获取当前版本信息
  Version current_version = currentVersion();
  // 如果当前版本的 subs_ 不等于 last_version_ 的 subs_，则刷新库信息
  if (current_version.subs_ != last_version_.subs_) {
    refreshLibraries();
    last_version_ = current_version;
  }
  // 在库列表中搜索给定地址的库信息
  auto* r = searchFor(addr);
  // 如果找不到对应的库信息
  if (!r) {
    // 如果当前版本的 adds_ 不等于 last_version_ 的 adds_，则再次刷新库信息
    if (current_version.adds_ != last_version_.adds_) {
      refreshLibraries();
      last_version_ = current_version;
    }
    // 再次在库列表中搜索给定地址的库信息
    r = searchFor(addr);
  }
  return r;
}

// 根据地址查找对应的库信息，如果找不到则抛出异常
const LibraryInfo& libraryFor(uint64_t addr) {
  auto* r = findLibraryFor(addr);
  // 如果找不到对应的库信息
  if (!r) {
    // 遍历没有展开器信息的库列表，并发出警告
    for (const auto& l : libraries_with_no_unwind_) {
      TORCH_WARN("Did not find a PT_GNU_EH_FRAME segment for ", l);
    }
    // 清空没有展开器信息的库列表
    libraries_with_no_unwind_.clear();
    // 抛出异常，说明地址不在已知库的范围内
    throw UnwindError("addr not in range of known libraries");
  }
  return *r;
}

// 返回展开器的统计信息
torch::unwind::Stats stats() {
  return stats_;
}

private:
// 在已排序的库信息列表中，根据地址进行二分查找
const LibraryInfo* searchFor(uint64_t addr) {
  // 如果库列表为空，则直接返回空指针
  if (all_libraries_.empty()) {
    return nullptr;
  }
  uint64_t low = 0;
  uint64_t high = all_libraries_.size();
  // 二分查找，直到找到对应的库信息或确定不存在
  while (low + 1 < high) {
    auto mid = (low + high) / 2;
    if (addr < all_libraries_.at(mid).load_bias()) {
      high = mid;
    } else {
      low = mid;
    }
  }
  // 获取找到的库信息指针
  LibraryInfo* r = &all_libraries_.at(low);
  // 如果地址不在该库的有效范围内，则返回空指针
  if (addr < r->load_bias() || addr >= r->last_addr()) {
    return nullptr;
  }
  return r;
}

// 已排序的库信息列表
std::vector<LibraryInfo> all_libraries_;

// 地址到展开器的哈希映射
ska::flat_hash_map<uint64_t, Unwinder> ip_cache_;

// 展开器的统计信息
torch::unwind::Stats stats_;

// 上一个版本信息，用于判断是否需要刷新库信息
Version last_version_;

// 没有展开器信息的库列表
std::vector<std::string> libraries_with_no_unwind_;
};

// 静态变量，用于存储解析栈信息的缓存
static UnwindCache unwind_cache;
// 用于保护解析缓存的互斥量
static std::shared_timed_mutex cache_mutex_;

// 函数：获取当前线程的调用栈帧
std::vector<void*> unwind() {
  std::vector<void*> frames;
  // 调用底层解析函数填充 frames 向量
  unwind_entry(&frames);
  return frames;
}

// 函数：查找给定地址对应的库信息
std::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr) {
  if (!addr) {
    return c10::nullopt;
  }
  // 使用共享锁保护缓存访问
  std::shared_lock lock(cache_mutex_);
  const LibraryInfo* library_info = unwind_cache.findLibraryFor((uint64_t)addr);
  if (!library_info) {
    return c10::nullopt;
  }
  // 返回包含库名和偏移量的可选对
  return std::make_pair(
      library_info->name(), (uint64_t)addr - library_info->load_bias());
}

// 静态函数：通过 dladdr 获取地址对应的函数名
static std::string dladdr_lookup(void* addr) {
  Dl_info dlinfo;
  std::string funcname = "??";
  // 使用 dladdr 查询地址对应的符号名称并进行解码
  if (dladdr(addr, &dlinfo) && dlinfo.dli_sname) {
    funcname = demangle(dlinfo.dli_sname);
  }
  return funcname;
}

// 结构体：符号解析器
struct Symbolizer {
  // 构造函数：初始化符号解析器
  Symbolizer() {
    auto envar = std::getenv("TORCH_ADDR2LINE_BINARY");
    if (envar != nullptr) {
      // 使用用户自定义的 addr2line 二进制文件路径（如有）
      addr2line_binary_ = envar;
      TORCH_WARN("Use custom addr2line binary: ", addr2line_binary_);
    } else {
      // 默认使用系统的 addr2line 二进制文件路径
      addr2line_binary_ = "addr2line";
    }
  }

  // 静态函数：获取符号解析器的互斥量
  static std::lock_guard<std::mutex> guard() {
    static std::mutex mutex;
    return std::lock_guard<std::mutex>(mutex);
  }

  // 静态函数：获取符号解析器的单例实例
  static Symbolizer& get() {
    static Symbolizer singleton;
    return singleton;
  }

  // 函数：请求给定地址的解析信息
  void request(void* addr) {
    if (frame_map_.count(addr)) {
      return;
    }
    auto maybe_library = libraryFor(addr);
    if (!maybe_library) {
      // 若无法找到地址对应的库信息，则设置为未知帧
      frame_map_[addr] = Frame{"??", "<unwind unsupported>", 0};
      return;
    }
    // 标记当前有待处理的结果
    has_pending_results_ = true;
    auto& entry = getOrCreate(maybe_library->first);
    entry.queried.push_back(addr);
    auto libaddress = maybe_library->second - 1;
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    entry.comm->out() << (void*)libaddress << "\n";
    // 避免在读取结果前向管道写入超过 64k 字节的数据，防止缓冲区阻塞
    if (entry.queried.size() - entry.completed > BLOCK) {
      entry.comm->out().flush();
      readPendingResults(entry);
    }
  }

  // 函数：查找给定地址对应的解析帧信息
  const Frame& lookup(void* addr) {
    // 若当前有待处理的结果，则先刷新所有通信管道并读取结果
    if (has_pending_results_) {
      for (auto& kv : entries_) {
        kv.second.comm->out().flush();
      }
      for (auto& kv : entries_) {
        readPendingResults(kv.second);
      }
      has_pending_results_ = false;
    }
    // 返回地址对应的解析帧信息
    return frame_map_.at(addr);
  }

 private:
  static constexpr int BLOCK = 1024;
  const char* addr2line_binary_;
  
  // 结构体：符号解析器中的条目信息
  struct Entry {
    std::unique_ptr<Communicate> comm;
    std::vector<void*> queried;
    size_t completed = 0;
  };
  
  // 哈希映射：库名到条目信息的映射
  ska::flat_hash_map<std::string, Entry> entries_;
  
  // 哈希映射：地址到解析帧信息的映射
  ska::flat_hash_map<void*, Frame> frame_map_;
  
  // 标志：是否有待处理的结果
  bool has_pending_results_ = true;

  // 函数：获取给定库名对应的条目信息，若不存在则创建新的条目
  Entry& getOrCreate(const std::string& name) {
    // 若不存在，则创建新的条目并返回
    if (!entries_.count(name)) {
      entries_[name].comm = std::make_unique<Communicate>();
    }
    return entries_[name];
  }
};
    // 在 entries_ 中查找给定名称的条目
    auto it = entries_.find(name);
    // 如果未找到该名称的条目
    if (it == entries_.end()) {
      // 构建用于调用 addr2line 的参数数组
      // NOLINTNEXTLINE(*-c-arrays*)
      const char* args[] = {
          addr2line_binary_, "-C", "-f", "-e", name.c_str(), nullptr};
      // 插入或更新 entries_ 中的条目
      // 如果插入成功，创建一个新的 Communicate 对象，并初始化 Entry
      it = entries_
               .insert_or_assign(
                   name,
                   Entry{
                       std::make_unique<Communicate>(addr2line_binary_, args),
                       {}})
               .first;
    }
    // 返回与给定名称关联的条目的引用
    return it->second;
  }
  // 读取 Entry 中尚未完成查询的结果
  void readPendingResults(Entry& e) {
    // 获取待查询的数量
    size_t N = e.queried.size();
    // 对每个尚未完成的查询执行循环
    for (; e.completed < N; ++e.completed) {
      // 创建一个 Frame 对象
      Frame frame;
      // 从通信管道中读取函数名
      std::getline(e.comm->in(), frame.funcname);
      // 从通信管道中读取包含文件名和行号的字符串
      std::string filename_lineno;
      std::getline(e.comm->in(), filename_lineno);
      // 查找文件名和行号字符串中最后一个冒号的位置
      auto colon = filename_lineno.find_last_of(':');
      // 提取文件名
      frame.filename = filename_lineno.substr(0, colon);
      // 提取行号并转换为整数，如果为 "?"，则设为 0
      std::string lineno_str = filename_lineno.substr(colon + 1);
      frame.lineno = lineno_str == "?" ? 0 : std::stoi(lineno_str);
      // 将查询结果与查询 ID 关联起来，并移动到 frame_map_ 中
      frame_map_[e.queried[e.completed]] = std::move(frame);
    }
  }
};

// 定义静态函数 symbolize_fast，用于快速符号化给定的堆栈帧
static std::vector<Frame> symbolize_fast(
    const std::vector<void*>& frames,
    Mode mode) {
  // 定义静态互斥锁 cache_mutex，用于保护共享的缓存数据结构
  static std::mutex cache_mutex;
  // 定义静态数组 frame_maps，存储两个 ska::flat_hash_map<void*, Frame>，分别用于不同模式的帧映射
  static std::array<ska::flat_hash_map<void*, Frame>, 2> frame_maps;
  // 根据模式选择当前使用的帧映射
  auto& frame_map = frame_maps[mode == Mode::fast ? 0 : 1];

  // 定义存储待查询索引的向量 indices_to_lookup 和结果帧的向量 results
  std::vector<uint32_t> indices_to_lookup;
  std::vector<Frame> results;
  results.reserve(frames.size());
  {
    // 使用 cache_mutex 创建锁对象 lock_guard，确保线程安全地访问缓存
    std::lock_guard<std::mutex> lock(cache_mutex);
    // 遍历 frames 中的每个帧地址
    for (auto i : c10::irange(frames.size())) {
      void* f = frames.at(i);
      // 在 frame_map 中查找当前帧地址 f 是否已有对应的 Frame 记录
      auto it = frame_map.find(f);
      if (it == frame_map.end()) {
        // 如果未找到对应的记录，将当前索引 i 添加到待查询索引向量中，并且在 results 中创建一个未知帧记录
        indices_to_lookup.push_back(i);
        results.emplace_back(Frame{"??", "??", 0});
      } else {
        // 如果找到了对应的记录，则直接使用已有的 Frame 记录
        results.emplace_back(it->second);
      }
    }
  }

  // 如果有未解析的帧索引，则进行符号化工作
  if (!indices_to_lookup.empty()) {
    // 创建 FastSymbolizer 对象 symbolizer，用于进行符号化
    FastSymbolizer symbolizer;
    // 遍历待查询索引向量
    for (auto i : indices_to_lookup) {
      void* addr = frames.at(i);
      Frame& f = results.at(i);
      // 获取地址 addr 所在的库信息
      auto library = libraryFor(frames.at(i));
      if (library) {
        // 如果找到了库信息，根据当前模式进行符号化
        if (mode == Mode::fast) {
          f = symbolizer.symbolize(library->first, library->second - 1);
        } else {
          f = Frame{library->first, "??", library->second - 1};
        }
      }
      // 如果函数名仍为未知，则尝试使用 dladdr_lookup 获取函数名
      if (f.funcname == "??") {
        f.funcname = dladdr_lookup(addr);
      }
    }
    // 再次使用 cache_mutex 创建锁对象 lock_guard，更新帧映射表 frame_map
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto i : indices_to_lookup) {
      frame_map.emplace(frames.at(i), results.at(i));
    }
  }
  // 返回符号化结果
  return results;
}

// 定义静态函数 symbolize_addr2line，使用 addr2line 工具符号化给定的堆栈帧
static std::vector<Frame> symbolize_addr2line(
    const std::vector<void*>& frames) {
  // 获取 Symbolizer 对象 guard，并从中获取实际的 Symbolizer 对象 s
  auto guard = Symbolizer::guard();
  Symbolizer& s = Symbolizer::get();
  // 遍历 frames 中的每个帧地址，向 Symbolizer 对象 s 发送请求并获取符号化结果
  for (auto f : frames) {
    s.request(f);
  }
  // 创建结果帧向量 results，并预留足够的空间
  std::vector<Frame> results;
  results.reserve(frames.size());
  // 遍历 frames 中的每个帧地址，通过 Symbolizer 对象 s 进行查找并记录结果
  for (auto f : frames) {
    results.emplace_back(s.lookup(f));
  }
  // 返回符号化结果
  return results;
}

// 如果非 FBCODE_CAFFE2 环境，定义符号化函数 symbolize，根据 mode 调用不同的符号化方法
#ifndef FBCODE_CAFFE2
std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode) {
  if (mode == Mode::addr2line) {
    // 如果模式为 addr2line，则调用 symbolize_addr2line 函数进行符号化
    return symbolize_addr2line(frames);
  } else {
    // 否则调用 symbolize_fast 函数进行快速符号化
    return symbolize_fast(frames, mode);
  }
}
#endif

// 返回 unwind_cache 的统计信息
Stats stats() {
  return unwind_cache.stats();
}

// 结束 torch::unwind 命名空间
} // namespace torch::unwind

// 定义 C 风格的函数 unwind_c，用于进行堆栈解析
extern "C" void unwind_c(std::vector<void*>* result, int64_t rsp, int64_t rbp) {
  // 使用共享的读锁 lock 保护 torch::unwind::cache_mutex_
  std::shared_lock lock(torch::unwind::cache_mutex_);
  // 创建 UnwindState 对象 state，初始化 rip、rsp 和 rbp 字段
  torch::unwind::UnwindState state{};
  // 从 rsp 中读取 rip 地址，并加上偏移 8
  // +8 是因为在将返回地址推送到堆栈后保存了 rsp
  state.rip = *(int64_t*)(rsp);
  state.rsp = rsp + 8;
  state.rbp = rbp;
  // 检查并更新 unwind_cache
  torch::unwind::unwind_cache.checkRefresh(lock);
  // 开始进行堆栈解析
  while (true) { // unwind for _start sets rip as being undefined
    // 将当前解析的 rip 地址添加到 result 向量中
    result->push_back((void*)state.rip);
    // 获取当前 rip 地址对应的 Unwinder 对象 uw
    const torch::unwind::Unwinder& uw =
        torch::unwind::unwind_cache.unwinderFor(state.rip, lock);
    // ...


这段代码主要涉及堆栈符号化的功能实现，包括快速符号化和 addr2line 符号化两种方式，使用了静态变量和互斥锁来保护共享数据结构，确保线程安全性。
    # 如果遇到终结符（terminator），执行以下操作
    if (uw.terminator()) {
        # 如果终结符是未知的（unknown），向结果列表中添加空指针
        if (uw.isUnknown()) {
            result->push_back(nullptr);
        }
        # 跳出循环
        break;
    }
    # 运行 uw 对象的 run 方法，更新状态并继续循环
    state = uw.run(state);
}

// 调用约定将前三个指针/整数参数放置在寄存器 rdi, rsi, rdx 中（全部是调用方保存的寄存器）
// rdi 已经包含结果向量的指针
// 我们添加当前 rsp 和 rbp 的参数，然后尾部调用进入 unwind_c 函数
__asm__(
    ".global unwind_entry\n"  // 声明 unwind_entry 为全局符号
    "unwind_entry:\n"         // unwind_entry 标签，入口点
    "mov %rsp, %rsi;\n"       // 将当前栈指针 rsp 的值移动到寄存器 rsi
    "mov %rbp, %rdx;\n"       // 将当前基址指针 rbp 的值移动到寄存器 rdx
    "jmp unwind_c;\n");       // 跳转到 unwind_c 函数

#endif
```