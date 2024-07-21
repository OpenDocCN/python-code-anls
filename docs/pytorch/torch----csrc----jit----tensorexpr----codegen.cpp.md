# `.\pytorch\torch\csrc\jit\tensorexpr\codegen.cpp`

```
#include <torch/csrc/jit/jit_log.h>  // 引入 Torch JIT 的日志功能头文件
#include <torch/csrc/jit/tensorexpr/analysis.h>  // 引入 Torch JIT Tensor Expression 的分析功能头文件
#include <torch/csrc/jit/tensorexpr/codegen.h>  // 引入 Torch JIT Tensor Expression 的代码生成功能头文件

#include <sstream>  // 引入标准库头文件

namespace torch::jit::tensorexpr {

CodeGen::CodeGen(
    StmtPtr stmt,
    std::vector<BufferArg> buffer_args,
    at::Device device,
    std::string kernel_func_name)
    : stmt_(std::move(stmt)),  // 初始化成员变量 stmt_
      buffer_args_(std::move(buffer_args)),  // 初始化成员变量 buffer_args_
      device_(device),  // 初始化成员变量 device_
      kernel_func_name_(std::move(kernel_func_name)) {  // 初始化成员变量 kernel_func_name_
  ExtCallMemoryReuse extCallMemoryReuse(buffer_args_);
  apply_mutator(&extCallMemoryReuse);  // 应用 mutator 来处理 buffer_args_
  allocIntermediateBufs();  // 分配中间缓冲区
}

RegisterCodeGenList::StmtFactoryMethod RegisterCodeGenList::
    FindStmtFactoryMethod(const std::string& name) {
  auto iter = stmt_factory_methods_.find(name);
  if (iter == stmt_factory_methods_.end()) {  // 如果未找到指定名称的 stmt 工厂方法
    std::ostringstream oss;
    oss << "Invalid stmt codegen name: " << name << ". ";
    oss << "Existing codegen names: [";
    int index = 0;
    for (auto& entry : stmt_factory_methods_) {  // 遍历已注册的所有 stmt 工厂方法
      if (index != 0) {
        oss << ", ";
      }
      oss << entry.first;
      index++;
    }
    oss << "]";
    throw std::runtime_error(oss.str());  // 抛出运行时错误，显示无效的 stmt codegen 名称和已有的 codegen 名称列表
  }
  return iter->second;  // 返回找到的 stmt 工厂方法
}

void RegisterCodeGenList::AddStmtFactoryMethod(
    const std::string& name,
    const StmtFactoryMethod& stmt_factory_method) {
  stmt_factory_methods_[name] = stmt_factory_method;  // 添加或更新指定名称的 stmt 工厂方法
}

std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& params,
    at::Device device,
    const std::string& kernel_func_name) {
  RegisterCodeGenList::StmtFactoryMethod method =
      RegisterCodeGenList::GetInstance().FindStmtFactoryMethod(name);  // 获取指定名称的 stmt 工厂方法
  return method(stmt, params, device, kernel_func_name);  // 调用 stmt 工厂方法创建 CodeGen 对象并返回
}

ExprPtr GenericIntrinsicsExpander::mutate(IntrinsicsPtr v) {
  if (v->op_type() == kSigmoid) {  // 如果是 sigmoid 操作
    auto x = v->param(0)->accept_mutator(this);  // 处理第一个参数
    auto one = expr_to_vec(
        ExprHandle(getImmediateByType(v->dtype(), 1.0)), v->dtype().lanes());  // 创建一个表示值为 1.0 的 ExprHandle
    auto zero = expr_to_vec(
        ExprHandle(getImmediateByType(v->dtype(), 0.0)), v->dtype().lanes());  // 创建一个表示值为 0.0 的 ExprHandle
    ExprHandle y = one / (one + exp(zero - ExprHandle(x)));  // 计算 sigmoid 函数值
    return y.node();  // 返回计算结果的节点
  }
  return IRMutator::mutate(v);  // 否则，调用基类的 mutate 方法
}

void* CodeGen::argToPtr(const BufferArg& bufferArg, const CallArg& callArg) {
  if (!bufferArg.isVar()) {  // 如果不是变量参数
    return callArg.data();  // 直接返回调用参数的数据指针
  }

  switch (bufferArg.dtype().scalar_type()) {  // 根据数据类型进行分支处理
#define TYPE_CASE(_1, Name) \
  case ScalarType::Name:    \
    return callArg.Name##Ptr();  // 返回特定类型的数据指针

    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);  // 处理所有标量类型

#undef TYPE_CASE

    default:
      throw unsupported_dtype();  // 抛出不支持的数据类型异常
  }
  return nullptr;  // 默认返回空指针
}

void CodeGen::call_with_numel(void** args, int64_t numel) {
  TORCH_INTERNAL_ASSERT(
      false, "This codegen backend does not implement call_with_numel");  // 断言失败，显示代码生成后端不实现 call_with_numel
}

static std::optional<size_t> bufSize(BufPtr buf) {
  size_t size = elementSize(buf->dtype().scalar_type()) * buf->dtype().lanes();  // 计算缓冲区的总大小
  for (auto& d : buf->dims()) {  // 遍历缓冲区的维度
    // 如果 d 不是常量，返回空的 optional 对象
    if (!d->isConstant()) {
      return c10::nullopt;
    }
    // 将 size 乘以 d 指向的整数值
    size = size * (*intValue(d));
    // 返回计算后的 size
    }
    return size;
}

// 这个算法根据中间缓冲区及其活跃范围列表，返回这些缓冲区的内存分配情况。
// 缓冲区 'A' 可以分配在内存中（在分配结果中表示为 'A' 对），也可以重用其他缓冲区，比如 'B'（在分配结果中表示为 ('A', 'B')）。
// 具体而言，我们按照中间缓冲区出现的时间顺序线性扫描，尝试为其分配一个现有的未被占用的内存分配。如果没有这样的可用分配，我们将为其创建内存。
// 一旦超出此缓冲区的活跃范围，我们将标记其相应的内存分配为“待重用”以供将来重用。
static std::vector<std::pair<BufPtr, BufPtr>> AllocBufsWithMemReuse(
    const std::unordered_set<BufPtr>& bufs,
    const std::unordered_map<BufPtr, std::tuple<int32_t, int32_t>>& buf_ranges,
    const std::unordered_set<BufPtr>& bufs_external_allocs) {
  // 按照缓冲区出现的时间进行排序。
  std::vector<BufPtr> bufs_sorted(bufs.begin(), bufs.end());
  auto sorting_function_by_start_time = [&buf_ranges](
                                            BufPtr b1, BufPtr b2) -> bool {
    return std::get<0>(buf_ranges.at(b1)) < std::get<0>(buf_ranges.at(b2));
  };
  std::sort(
      bufs_sorted.begin(), bufs_sorted.end(), sorting_function_by_start_time);

  // 映射中间缓冲区到最近使用的内存（如果有的话）。
  std::list<BufPtr> mem_up_for_grabs;
  std::unordered_map<BufPtr, BufPtr> buf_mem_map;
  std::vector<std::pair<BufPtr, BufPtr>> buf_allocs;

  auto sorting_function_by_end_time = [&buf_ranges](
                                          BufPtr b1, BufPtr b2) -> bool {
    return std::get<1>(buf_ranges.at(b1)) < std::get<1>(buf_ranges.at(b2));
  };
  for (const auto& buf : bufs_sorted) {
    // 如果缓冲区具有动态形状，我们将跳过它（即为其分配内存，并且没有对其内存的未来重用）。
    // TODO: 对于具有动态形状的缓冲区重用内存
    if (!bufSize(buf)) {
      buf_allocs.emplace_back(buf, buf);
      continue;
    }

    auto start = std::get<0>(buf_ranges.at(buf));

    // 释放那些活跃范围在此缓冲区创建时间之前结束的缓冲区的内存。
    // TODO: 优化原地操作和复制操作
    std::vector<BufPtr> buf_to_release;
    for (auto& mapped : buf_mem_map) {
      auto buf_mapped = mapped.first;
      auto end_buf_mapped = std::get<1>(buf_ranges.at(buf_mapped));
      if (end_buf_mapped < start) {
        buf_to_release.push_back(buf_mapped);
      }
    }

    // 按照使用时间顺序对缓冲区进行排序，因此释放列表的头部包含最近使用的缓冲区。
    std::sort(
        buf_to_release.begin(),
        buf_to_release.end(),
        sorting_function_by_end_time);
    for (auto& buf_rl : buf_to_release) {
      mem_up_for_grabs.push_front(buf_mem_map.at(buf_rl));
      buf_mem_map.erase(buf_rl);
    }

    bool allocated = false;
    // 如果在 bufs_external_allocs 中找不到 buf，进入条件判断
    if (bufs_external_allocs.find(buf) == bufs_external_allocs.end()) {
      // 检查是否有可供 buf 重用的空闲内存
      for (auto it = mem_up_for_grabs.begin(); it != mem_up_for_grabs.end();
           it++) {
        auto m = *it;
        // 如果 m 的大小大于或等于 buf 的大小，则将 buf 映射到 m
        if (bufSize(m) >= bufSize(buf)) {
          buf_mem_map[buf] = m;
          // 将 (buf, m) 添加到 buf_allocs 中
          buf_allocs.emplace_back(buf, m);
          allocated = true;
          // 从 mem_up_for_grabs 中移除已分配的内存 m
          mem_up_for_grabs.erase(it);
          break;
        }
      }
    }

    // 如果没有可供重用的内存，需为 buf 分配新内存
    if (!allocated) {
      // 将 buf 映射到自身的内存
      buf_mem_map[buf] = buf;
      // 将 (buf, buf) 添加到 buf_allocs 中
      buf_allocs.emplace_back(buf, buf);
    }
  }

  // 返回已分配的内存块列表 buf_allocs
  return buf_allocs;
static StmtPtr insertAllocFree(
    std::vector<std::pair<BufPtr, BufPtr>>& buf_allocs,
    const std::unordered_set<BufPtr>& bufs_external_allocs,
    StmtPtr stmt) {
  // 将传入的 stmt 转换为 BlockPtr，如果无法转换，则创建一个新的 BlockPtr 并包含该 stmt
  BlockPtr b = to<Block>(stmt);
  if (!b) {
    b = alloc<Block>(std::vector<StmtPtr>({stmt}));
  }

  std::vector<BufPtr> bufs_ext_to_free;
  // 在全局范围内为临时缓冲区插入分配和释放操作
  for (auto rit = buf_allocs.rbegin(); rit != buf_allocs.rend(); ++rit) {
    if (rit->first == rit->second) {
      BufPtr buf = rit->first;
      // 如果缓冲区不在外部分配的集合中，则在 Block 的开头插入分配语句，末尾插入释放语句；否则将缓冲区加入待释放列表
      if (bufs_external_allocs.find(buf) == bufs_external_allocs.end()) {
        b->prepend_stmt(alloc<Allocate>(buf));
        b->append_stmt(alloc<Free>(buf));
      } else {
        bufs_ext_to_free.push_back(buf);
      }
    } else {
      // 如果缓冲区有不同的起始和结束位置，使用 PlacementAllocate 在 Block 的开头插入分配语句
      b->prepend_stmt(alloc<PlacementAllocate>(rit->first, rit->second));
    }
  }

  // 在 Block 的末尾插入释放外部缓冲区的语句
  b->append_stmt(alloc<FreeExt>(bufs_ext_to_free));
  return b;
}

std::unordered_map<std::string, std::string> ExtCallMemoryReuse::
    makeExtCallFuncNameMap() {
  // 创建一个映射表，将一组函数名映射到对应的函数名，用于外部调用的函数名称重用
  return {
      {"nnc_aten_quantize_per_tensor", "nnc_aten_quantize_per_tensor_out"},
      {"nnc_aten_dequantize", "nnc_aten_dequantize_out"},
      {"nnc_aten_quantized_mul", "nnc_aten_quantized_mul_out"},
      {"nnc_aten_quantized_conv2d", "nnc_aten_quantized_conv2d_out"},
      {"nnc_aten_quantized_conv2d_relu", "nnc_aten_quantized_conv2d_relu_out"},
      {"nnc_aten_quantized_mul", "nnc_aten_quantized_mul_out"},
      {"nnc_aten_quantized_sigmoid", "nnc_aten_quantized_sigmoid_out"},
      {"nnc_aten_upsample_nearest2d", "nnc_aten_upsample_nearest2d_out"},
      {"nnc_aten_quantized_linear", "nnc_aten_quantized_linear_out"},
      {"nnc_aten_quantized_conv1d", "nnc_aten_quantized_conv1d_out"},
      {"nnc_aten_quantized_mul_scalar", "nnc_aten_quantized_mul_scalar_out"},
      {"nnc_aten_max_red", "nnc_aten_max_red_out"},
      {"nnc_aten_conv1d", "nnc_aten_conv1d_out"},
  };
}

const std::unordered_map<std::string, std::string>
    ExtCallMemoryReuse::extCallFuncNameMap_ = makeExtCallFuncNameMap();

ExtCallMemoryReuse::ExtCallMemoryReuse(
    const std::vector<CodeGen::BufferArg>& bufferArgs) {
  // 构造函数，将缓冲区参数加入成员变量 bufferArgs_ 中，如果缓冲区非空，则插入对应的缓冲区指针
  for (const auto& ba : bufferArgs) {
    if (ba.buf()) {
      bufferArgs_.insert(ba.buf());
    }
  }
}

StmtPtr ExtCallMemoryReuse::mutate(ExternalCallPtr v) {
  // 如果外部调用函数名在映射表中，并且缓冲区参数不在 bufferArgs_ 中，则创建一个带分配的外部调用语句
  if (extCallFuncNameMap_.count(v->func_name()) &&
      bufferArgs_.count(v->buf()) == 0) {
    std::vector<BufPtr> buf_out_args = {v->buf()};
    return alloc<ExternalCallWithAlloc>(
        extCallFuncNameMap_.at(v->func_name()),
        buf_out_args,
        v->buf_args(),
        v->args());
  }
  // 否则返回原始的外部调用语句
  return v;
}

// We allocate intermediate buffers by inserting Allocate/Free or
// PlacementAllocate stmts. Allocate/Free stmts will allocate memory at runtime,
// and PlacementAllocate stmt reuses the memory of one buffer for another
// buffer. In current implementation, we use linear scan for memory reuses.
// TODO: try more memory reuse algorithms and compare their memory efficiency.
// 对于中间缓冲区，通过插入 Allocate/Free 或 PlacementAllocate 语句进行分配。Allocate/Free 语句在运行时分配内存，
// PlacementAllocate 语句重用一个缓冲区的内存给另一个缓冲区。当前实现中，我们使用线性扫描进行内存重用。
// TODO: 尝试更多的内存重用算法，并比较它们的内存效率。
void CodeGen::allocIntermediateBufs() {
  // 标识尚未分配的中间缓冲区。
  auto bufs = NodeFinder<Buf>::find(stmt_);
  // 存储已分配的缓冲区的集合
  std::unordered_set<BufPtr> bufs_allocated;
  // 遍历已知的缓冲区参数，将其缓冲区指针插入已分配集合
  for (const auto& b : buffer_args_) {
    bufs_allocated.insert(b.buf());
  }
  // 查找所有分配语句中的缓冲区并插入已分配集合
  auto allocs = NodeFinder<Allocate>::find(stmt_);
  for (const auto& a : allocs) {
    bufs_allocated.insert(a->buf());
  }

  // 存储未分配的中间缓冲区和其生存范围
  std::unordered_set<BufPtr> interm_bufs;
  std::unordered_map<BufPtr, std::tuple<int32_t, int32_t>> interm_buf_ranges;
  // 对于每个找到的缓冲区，如果尚未分配且未记录，则记录其生存范围
  for (const auto& buf : bufs) {
    if (!bufs_allocated.count(buf) && !interm_bufs.count(buf)) {
      interm_bufs.insert(buf);

      // 标识每个未分配的中间缓冲区的访问语句
      auto range = BufLiveRange::liveRange(stmt_, buf);
      interm_buf_ranges.emplace(buf, range);
    }
  }

  // 查找外部分配的缓冲区
  const auto bufs_external_allocs = ExternalAllocBufFinder::find(stmt_);

  // 对于每个中间缓冲区，尝试重用一个生存范围与当前缓冲区不重叠的旧缓冲区的内存，或者分配新内存
  auto buf_allocs = AllocBufsWithMemReuse(
      interm_bufs, interm_buf_ranges, bufs_external_allocs);

  // 插入内存分配/映射节点
  if (!buf_allocs.empty()) {
    auto stmt_new = insertAllocFree(buf_allocs, bufs_external_allocs, stmt_);
    set_stmt(stmt_new);
  }

  // 打印调试信息，显示内存分配情况
  GRAPH_DEBUG("\nMemory Allocation:\n\n", *stmt(), "\n");
}

} // namespace torch::jit::tensorexpr
```