# `.\pytorch\torch\csrc\jit\tensorexpr\block_codegen.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <string>
// 引入处理字符串的标准库
#include <unordered_map>
// 引入无序映射的标准库
#include <unordered_set>
// 引入无序集合的标准库
#include <utility>
// 引入实用工具的标准库

#include <ATen/ATen.h>
// 引入 PyTorch 的 ATen 头文件
#include <torch/csrc/jit/resource_guard.h>
// 引入 PyTorch 的 jit 库中的资源保护头文件
#include <torch/csrc/jit/tensorexpr/analysis.h>
// 引入 PyTorch 的 jit 库中的 TensorExpr 分析头文件
#include <torch/csrc/jit/tensorexpr/codegen.h>
// 引入 PyTorch 的 jit 库中的 TensorExpr 代码生成头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
// 引入 PyTorch 的 jit 库中的 TensorExpr IR 头文件
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
// 引入 PyTorch 的 jit 库中的 TensorExpr IR 打印头文件
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
// 引入 PyTorch 的 jit 库中的 TensorExpr IR 访问者头文件
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>
// 引入 PyTorch 的 jit 库中的 TensorExpr 唯一名称管理器头文件

namespace torch {
namespace jit {
namespace tensorexpr {

// 一个分析程序相关 Block 后端的类
class BlockAnalysis : public IRVisitor {
 public:
  // 判断给定的缓冲区是否是存储目标
  bool is_buf_store_target(BufPtr buf) const {
    return store_targets_.count(buf) > 0;
  }

  // 返回加载操作涉及的所有缓冲区
  const std::unordered_set<BufPtr>& loads() const {
    return loads_;
  }

  // 返回存储操作涉及的所有缓冲区
  const std::unordered_set<BufPtr>& stores() const {
    return store_targets_;
  }

  // 返回块的大小
  int block_size() const {
    return block_size_;
  }

  // 检查一组缓冲区是否在映射中
  bool areBufsInMap(const std::unordered_set<BufPtr>& bufs) const;

  // 获取多维缓冲区
  BufPtr getMultiDimBuf(BufPtr buf) const;

  // 获取输入缓冲区的名称
  std::string getInputName(BufPtr buf) const;

  // 获取扁平化输入缓冲区的名称
  std::string getFlatInputName(BufPtr buf) const {
    return getInputName(std::move(buf)) + "_flat";
  }

  // 返回缓冲区映射
  std::unordered_map<std::string, BufPtr> getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  // 重写的 IRVisitor 访问函数
  void visit(StorePtr v) override;
  void visit(LoadPtr v) override;
  void visit(ForPtr v) override;

  // 输入到张量缓冲区的映射
  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
  // 存储操作涉及的缓冲区集合
  std::unordered_set<BufPtr> store_targets_;
  // 加载操作涉及的缓冲区集合
  std::unordered_set<BufPtr> loads_;
  // 块的大小，默认为 32
  int block_size_ = 32;
};

// 一个类，重写 IRPrinter 以生成 Block 代码
class BlockPrinter : public IRPrinter {
 public:
  // 构造函数，初始化 BlockPrinter 对象
  BlockPrinter(std::ostream* os, BlockAnalysis* block_analysis)
      : IRPrinter(*os), block_analysis_(block_analysis) {}

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  // 分析程序对象指针
  BlockAnalysis* block_analysis_;
  // 维度值映射表
  std::unordered_map<std::string, int> dim_values_map;
  // 维度名称列表
  std::vector<std::string> dim_names = {"N", "H", "W", "C"};
  // 扁平化维度名称列表
  std::vector<std::string> flat_dim_names = {"N", "NH", "NHW", "NHWC"};

  // 打印张量信息
  void PrintTensorInfo(const std::unordered_set<BufPtr>& bufs);
  // 打印参数信息
  void PrintArguments(const std::unordered_set<BufPtr>& bufs);
  // 打印缓冲区信息
  void PrintBufferInfo(const std::unordered_set<BufPtr>& bufs);
  // 打印分布信息
  void PrintDistribution(const std::unordered_set<BufPtr>& bufs);
  // 打印循环信息
  void PrintLoop(const std::unordered_set<BufPtr>& bufs, bool block_idx = true);
  // 打印重塑信息
  void PrintReshapeInfo(const std::unordered_set<BufPtr>& bufs, bool reverse = false);
  // 打印 DMA 信息
  void PrintDMAs(const std::unordered_set<BufPtr>& bufs);
  // 打印调整缓冲区信息
  void PrintAdjustBuffers(const std::unordered_set<BufPtr>& bufs);

  // 重写的 IRPrinter 访问函数
  void visit(ForPtr v) override;
  void visit(LoadPtr v) override;
  void visit(StorePtr v) override;
  void visit(BlockPtr v) override;
  void visit(AddPtr v) override;
  void visit(MulPtr v) override;
};
// 定义名为 BlockCodeGen 的类，继承自 CodeGen 类
class TORCH_API BlockCodeGen : public CodeGen {
 public:
  // 模板构造函数，接受一个语句指针和一系列 BufferArg 参数
  template <typename... Ts>
  /* implicit */
  BlockCodeGen(StmtPtr stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCPU)) {
    Initialize();  // 调用初始化函数
  }

  // 构造函数，接受一个语句指针、BufferArg 向量、设备类型和内核函数名
  BlockCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCPU),
      const std::string& kernel_func_name = "func")
      : CodeGen(stmt, buffer_args, device, kernel_func_name) {
    Initialize();  // 调用初始化函数
  }

  // 虚析构函数，用于释放资源
  ~BlockCodeGen() override;

  // 调用函数，接受 CallArg 向量参数
  void call(const std::vector<CallArg>& args) override;

  // 原始调用函数，接受 void 指针向量参数
  void call_raw(const std::vector<void*>& args) override;

  // 初始化函数，用于初始化类成员变量
  void Initialize();

  // 获取代码文本的函数，返回内部字符串流的内容
  std::string getCodeText(const std::string& attr = "") override {
    return oss_.str();
  }

 private:
  // 获取唯一名称管理器的函数，如果打印器为空则抛出运行时错误
  UniqueNameManager* name_manager() {
    if (!printer_) {
      throw std::runtime_error("Null IRPrinter is not expected");
    }
    return printer_->name_manager();
  }

  // 获取输出流的函数
  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;  // 字符串流对象，用于存储生成的代码文本
  std::unique_ptr<BlockPrinter> printer_;  // 唯一指针，指向 BlockPrinter 对象的智能指针
  std::unique_ptr<BlockAnalysis> block_analysis_;  // 唯一指针，指向 BlockAnalysis 对象的智能指针

  // 获取唯一函数名的函数，接受函数名前缀作为参数
  std::string GetUniqueFuncName(const std::string& func_prefix);
};
} // namespace tensorexpr
} // namespace jit
} // namespace torch
```