# `.\pytorch\torch\csrc\jit\tensorexpr\block_codegen.cpp`

```
// 引入头文件block_codegen.h，用于TensorExpr的块级代码生成
#include <torch/csrc/jit/tensorexpr/block_codegen.h>

// 引入TensorExpr的日志记录、分析、求值、异常处理和IR简化等相关头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

// 定义torch::jit::tensorexpr命名空间
namespace torch::jit::tensorexpr {

// 根据数据类型返回对应的C++字符串表示形式
static std::string blockDtypeCppString(const Dtype& dtype) {
  switch (dtype.scalar_type()) {
    case ScalarType::Bool:
      return "1";
    case ScalarType::Half:
      return "2";
    case ScalarType::BFloat16:
      return "2";
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Char:
      return "1";
    case ScalarType::Byte:
      return "1";
    case ScalarType::Short:
      return "4";
    case ScalarType::Long:
      return "8";
    case ScalarType::Float:
      return "2"; // 暂时返回Half
    default:
      return dtype.ToCppString();
  }
}

// 检查一组缓冲区是否都存在于映射中
bool BlockAnalysis::areBufsInMap(const std::unordered_set<BufPtr>& bufs) const {
  for (auto const& arg : bufs) {
    auto got = map_input_to_tensor_bufs_.find(arg->name_hint());
    if (got == map_input_to_tensor_bufs_.end()) {
      return false;
    }
  }
  return true;
}

// 获取多维缓冲区的映射
BufPtr BlockAnalysis::getMultiDimBuf(BufPtr buf) const {
  auto input_ = map_input_to_tensor_bufs_.find(buf->name_hint());
  if (input_ != map_input_to_tensor_bufs_.end()) {
    return input_->second;
  } else {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  }
}

// 获取输入缓冲区的名称
std::string BlockAnalysis::getInputName(BufPtr buf) const {
  auto input_ = map_input_to_tensor_bufs_.find(buf->name_hint());
  if (input_ != map_input_to_tensor_bufs_.end()) {
    return input_->second->name_hint();
  } else {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  }
}

// 处理Store节点，将目标缓冲区加入存储目标集合，并递归处理存储值
void BlockAnalysis::visit(StorePtr v) {
  store_targets_.insert(v->buf());
  v->value()->accept(this);
}

// 处理Load节点，将Load节点的缓冲区加入加载集合
void BlockAnalysis::visit(LoadPtr v) {
  loads_.insert(v->buf());
}

// 处理For节点，根据循环选项进行分支处理：GPU块索引、GPU线程索引或默认处理
void BlockAnalysis::visit(ForPtr v) {
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    map_input_to_tensor_bufs_ = loop_options.get_buffer_mapping();
    v->body()->accept(this);
  } else if (loop_options.is_gpu_thread_index()) {
    auto block_size = v->stop();
    block_size_ = *intValue(block_size);
    v->body()->accept(this);
  } else {
    IRVisitor::visit(v);
  }
}

// 处理Add节点，输出"add("并递归处理左右子节点
void BlockPrinter::visit(AddPtr v) {
  emitIndent();
  os() << "add(";
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

// 处理Mul节点，输出"mul("并递归处理左右子节点
void BlockPrinter::visit(MulPtr v) {
  emitIndent();
  os() << "mul(";
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}
void BlockPrinter::visit(ForPtr v) {
  // 获取循环的选项
  const LoopOptions& loop_options = v->loop_options();

  // 获取当前代码块的读取和写入缓冲区
  auto buf_reads = block_analysis_->loads();
  auto buf_writes = block_analysis_->stores();

  // 将读取和写入缓冲区合并为一个无序集合
  std::unordered_set<BufPtr> bufs(buf_reads.begin(), buf_reads.end());
  bufs.insert(buf_writes.begin(), buf_writes.end());

  // 如果是 GPU 块索引循环
  if (loop_options.is_gpu_block_index()) {
    emitIndent();
    // 打印缓冲区的张量信息
    PrintTensorInfo(bufs);
    // 打印缓冲区的分布信息
    PrintDistribution(bufs);
    // 打印缓冲区读取的详细信息
    PrintBufferInfo(buf_reads);
    // 打印缓冲区的参数信息
    PrintArguments(bufs);

    emitIndent();
    os() << "compute {" << std::endl;

    // 打印重新形状信息
    PrintReshapeInfo(bufs);

    emitIndent();
    // 打印循环体
    PrintLoop(bufs, true);
    v->body()->accept(this);

    os() << std::endl;
    emitIndent();
    // 打印写入缓冲区的反向重新形状信息
    PrintReshapeInfo(buf_writes, true); // print reverse reshape
    os() << "}";
    os() << std::endl;
  } else if (loop_options.is_gpu_thread_index()) {
    // 打印 DMA 操作信息
    PrintDMAs(buf_reads);
    // 打印循环体
    PrintLoop(buf_reads, false);
    v->body()->accept(this);
    os() << std::endl;
    // 调整缓冲区的打印信息
    PrintAdjustBuffers(buf_reads);

  } else {
    // 调用默认的 IRPrinter 访问方法
    IRPrinter::visit(v);
  }
}

void BlockPrinter::PrintTensorInfo(const std::unordered_set<BufPtr>& bufs) {
  os() << "tensors {";
  for (auto& buf : bufs) {
    os() << std::endl;
    emitIndent();
    emitIndent();
    // 获取多维缓冲区的维度数量
    auto num_dims = block_analysis_->getMultiDimBuf(buf)->dims().size();
    // 打印输入名称和维度信息
    os() << block_analysis_->getInputName(buf) << " = ";
    os() << "{";
    for (unsigned long d = 0; d < num_dims; d++) {
      os() << "{" << dim_names[d] << "};";
    }
    os() << " elem : " << blockDtypeCppString(buf->dtype());
    os() << "}";
  }

  for (auto& buf : bufs) {
    os() << std::endl;
    emitIndent();
    emitIndent();
    // 获取多维缓冲区的维度数量
    auto num_dims = block_analysis_->getMultiDimBuf(buf)->dims().size();
    // 打印扁平化张量的输入名称和维度信息
    os() << block_analysis_->getFlatInputName(buf) << " = ";
    os() << "{";
    os() << "{" << flat_dim_names[num_dims - 1] << "};";
    os() << " elem : " << blockDtypeCppString(buf->dtype());
    os() << "}"
         << " // flattened tensor";
  }
  os() << std::endl;
  emitIndent();
  os() << "}" << std::endl << std::endl;
}

void BlockPrinter::PrintArguments(const std::unordered_set<BufPtr>& bufs) {
  for (auto& buf : bufs) {
    auto multidimbuf = block_analysis_->getMultiDimBuf(buf);
    auto num_dims = multidimbuf->dims().size();

    // 对于多维张量的维度
    for (unsigned long d = 0; d < num_dims; d++) {
      auto dim_val = *intValue(multidimbuf->dim(d));
      // 将维度名称和值添加到映射中
      this->dim_values_map.emplace(this->dim_names[d], dim_val);
    }

    // 对于扁平化张量的维度
    auto val = *intValue(buf->dim(0));
    if (block_analysis_->is_buf_store_target(buf)) {
      // 将扁平化维度名称和值添加到映射中
      this->dim_values_map.emplace(this->flat_dim_names[num_dims - 1], val);
    }
  }

  emitIndent();
  os() << "arguments {" << std::endl;

  // 打印维度名称和值映射
  for (auto const& arg : this->dim_values_map) {
    emitIndent();
    // 输出维度名称和对应的值
    ```
    os() << "var " << arg.first << " = " << arg.second << std::endl;

# 将变量名和对应的值输出到流 `os()` 中

  }

  emitIndent();
  emitIndent();
  auto blck_sz = block_analysis_->block_size();
  os() << "var bs_N = " << blck_sz << std::endl;

# 调用两次缩进函数 `emitIndent()`，然后获取当前代码块的分析对象 `block_analysis_` 的块大小并存储到变量 `blck_sz` 中，将 `bs_N` 变量和 `blck_sz` 的值输出到流 `os()` 中

  emitIndent();
  emitIndent();
  os() << "var bs_DPE = " << blck_sz << std::endl;

# 再次调用两次缩进函数 `emitIndent()`，然后将 `bs_DPE` 变量和 `blck_sz` 的值输出到流 `os()` 中

  emitIndent();
  os() << "}" << std::endl << std::endl;

# 调用一次缩进函数 `emitIndent()`，然后输出代码块的结束符 `}` 到流 `os()` 中，同时输出两个换行符
}

void BlockPrinter::PrintBufferInfo(const std::unordered_set<BufPtr>& bufs) {
  // 打印缩进
  emitIndent();
  // 输出字符串 "buffers {"
  os() << "buffers {";
  // 遍历缓冲区集合
  for (auto& read : bufs) {
    // 换行并打印缩进
    os() << std::endl;
    emitIndent();
    emitIndent();
    // 输出缓冲区的名称和关联的信息
    os() << block_analysis_->getFlatInputName(read) << " = ";
    // 输出 "{{bs_DPE}}"
    os() << "{{"
         << "bs_DPE"
         << "}}";
  }
  // 换行并打印结束符 "}"
  os() << std::endl;
  emitIndent();
  os() << "}" << std::endl << std::endl;
}

void BlockPrinter::PrintDistribution(const std::unordered_set<BufPtr>& bufs) {
  // 打印缩进
  emitIndent();
  // 输出字符串 "distribution {"
  os() << "distribution {" << std::endl;
  // 遍历缓冲区集合
  for (auto& buf : bufs) {
    emitIndent();
    emitIndent();
    // 获取缓冲区的名称和关联的信息
    auto buf_name = buf->name_hint();
    os() << block_analysis_->getFlatInputName(buf) << " = ";
    // 输出固定的分布信息 "{(0, 1, )}"
    os() << "{(0, 1, )}" << std::endl;
  }
  // 输出结束符 "  }"
  os() << "  }" << std::endl << std::endl;
}

void BlockPrinter::PrintLoop(
    const std::unordered_set<BufPtr>& bufs,
    bool block_idx) {
  // 打印缩进并输出 "loop ("
  emitIndent();
  os() << "loop (";
  // 初始化计数器 trip
  auto trip = 0;
  // 遍历缓冲区集合
  for (auto& buf : bufs) {
    // 如果不是第一次循环，则输出逗号
    if (trip > 0) {
      os() << ",";
    }
    // 输出维度信息 "{dim : "
    os() << "{dim : ";
    // 输出缓冲区的维度信息和块信息
    os() << block_analysis_->getFlatInputName(buf) << ".dim.0, ";
    os() << (block_idx ? "block: bs_N}" : "block: bs_DPE}");
    // 递增 trip 计数
    ++trip;
  }
  // 输出闭合括号 ")"
  os() << ")";
}

void BlockPrinter::PrintReshapeInfo(
    const std::unordered_set<BufPtr>& bufs,
    bool reverse) {
  // 遍历缓冲区集合
  for (auto& buf : bufs) {
    // 打印缩进并输出 "reshape("
    emitIndent();
    os() << "reshape(";
    // 根据 reverse 参数选择输出不同的名称组合
    os() << (reverse ? block_analysis_->getFlatInputName(buf)
                     : block_analysis_->getInputName(buf))
         << ", "
         << (reverse ? block_analysis_->getInputName(buf)
                     : block_analysis_->getFlatInputName(buf))
         << ")" << std::endl;
  }
}

void BlockPrinter::PrintDMAs(const std::unordered_set<BufPtr>& bufs) {
  // 遍历缓冲区集合
  for (auto& read : bufs) {
    // 打印缩进并输出 "dma_in("
    emitIndent();
    os() << "dma_in(";
    // 输出缓冲区的扁平化输入名称
    os() << block_analysis_->getFlatInputName(read);
    // 输出闭合括号 ")"
    os() << ")" << std::endl;
  }
}

void BlockPrinter::PrintAdjustBuffers(const std::unordered_set<BufPtr>& bufs) {
  // 遍历缓冲区集合
  for (auto& read : bufs) {
    // 打印缩进并输出 "adjust_buffer("
    emitIndent();
    os() << "adjust_buffer(";
    // 输出缓冲区的扁平化输入名称
    os() << block_analysis_->getFlatInputName(read);
    // 输出闭合括号 ")"
    os() << ")" << std::endl;
  }
}

void BlockPrinter::visit(LoadPtr v) {
  // 输出加载指令对应的缓冲区名称和后缀 ".buffer, "
  os() << block_analysis_->getFlatInputName(v->buf()) << ".buffer, ";
}

void BlockPrinter::visit(StorePtr v) {
  // 打印缩进并输出存储指令对应的值和缓冲区名称
  emitIndent();
  os() << *v->value() << block_analysis_->getFlatInputName(v->buf())
       << ".tensor)" << std::endl;
}

void BlockPrinter::visit(BlockPtr v) {
  // 输出起始的大括号 "{"
  os() << "{" << std::endl;
  // 增加缩进计数
  indent_++;
  // 遍历块中的每个语句，并逐个接受访问
  for (const StmtPtr& s : v->stmts()) {
    s->accept(this);
  }
  // 减少缩进计数
  indent_--;
  // 打印缩进并输出结束的大括号 "}"
  emitIndent();
  os() << "}";
}

std::string BlockCodeGen::GetUniqueFuncName(const std::string& func_prefix) {
  // 使用静态变量 counter 确保每次调用得到不同的函数名称
  static int64_t counter = 0;
  ++counter;
  int64_t value = counter;
  // 返回带有计数后缀的函数名称
  return func_prefix + "_" + std::to_string(value);
}
// 初始化函数，用于设置 BlockCodeGen 对象
void BlockCodeGen::Initialize() {
  // 创建 BlockAnalysis 对象，并使用 std::make_unique 进行内存管理
  block_analysis_ = std::make_unique<BlockAnalysis>();
  // 创建 BlockPrinter 对象，传入输出流 oss_ 和 BlockAnalysis 对象的指针
  printer_ = std::make_unique<BlockPrinter>(&oss_, block_analysis_.get());

  // 获得一个 StmtPtr 类型的语句指针，并调用其 accept 方法进行分析
  StmtPtr stmt_v = stmt();
  stmt_v->accept(block_analysis_.get());

  // 获取加载和存储操作中涉及的缓冲区列表
  auto buf_reads = block_analysis_->loads();
  auto buf_writes = block_analysis_->stores();
  
  // 确保所有读取和写入操作中涉及的缓冲区都在映射中
  std::unordered_set<BufPtr> bufs(buf_reads.begin(), buf_reads.end());
  bufs.insert(buf_writes.begin(), buf_writes.end());
  if (!block_analysis_->areBufsInMap(bufs)) {
    // 如果存在未映射的缓冲区，则抛出运行时错误
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  };

  // 生成唯一的函数名字符串
  std::string func_name = GetUniqueFuncName("func");
  // 输出内核函数的声明及其参数
  os() << "kernel " << func_name << "(";
  for (auto const& arg : buf_writes) {
    os() << block_analysis_->getInputName(arg);
  }
  for (auto const& arg : buf_reads) {
    os() << ";" << block_analysis_->getInputName(arg);
  }
  os() << ")";

  // 调用打印器的 accept 方法，输出语句的代码
  stmt_v->accept(printer_.get());

  // 记录生成的 Block 代码到日志中
  GRAPH_DEBUG("Generated Block code: ", oss_.str(), "\n");
}

// 不支持调用 Block 代码，抛出运行时错误
void BlockCodeGen::call(const std::vector<CallArg>& args) {
  throw std::runtime_error("BlockCodeGen: Cannot call Block code ");
}

// 不支持原始调用 Block 代码，抛出运行时错误
void BlockCodeGen::call_raw(const std::vector<void*>& args) {
  throw std::runtime_error("BlockCodeGen: Cannot call Block code ");
}

// BlockCodeGen 类的析构函数，默认实现
BlockCodeGen::~BlockCodeGen() = default;

// 注册 BlockCodeGen 类到代码生成器注册表中
RegisterCodeGen<BlockCodeGen> block_codegen_reg("block_codegen");

} // namespace torch::jit::tensorexpr
```