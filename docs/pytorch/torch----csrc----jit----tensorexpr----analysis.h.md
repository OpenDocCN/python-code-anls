# `.\pytorch\torch\csrc\jit\tensorexpr\analysis.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
// 引入所需的头文件

#include <utility>
// 引入标准库中的实用工具

namespace torch {
namespace jit {
namespace tensorexpr {
// 命名空间 torch::jit::tensorexpr 下的类和函数定义

// IRVisitor 类的派生类 HasRand，用于检查语句中是否包含随机数生成操作
class HasRand : public IRVisitor {
 public:
  // 构造函数，接受一个语句的指针并开始访问
  HasRand(StmtPtr stmt) : stmt_(std::move(stmt)) {
    stmt_->accept(this);
  }

  // 返回语句中是否包含随机数生成操作
  bool has_rand() const {
    return has_rand_;
  }

 private:
  // 重写 IRVisitor 中的 visit 方法，用于处理特定类型的节点
  void visit(IntrinsicsPtr v) override {
    // 如果节点是随机数生成操作
    if (v->op_type() == IntrinsicsOp::kRand) {
      has_rand_ = true;
    } else {
      // 否则调用基类 IRVisitor 的 visit 方法继续访问
      IRVisitor::visit(std::move(v));
    }
  }
  StmtPtr stmt_;  // 保存传入的语句指针
  bool has_rand_ = false;  // 记录是否发现随机数生成操作的标志位
};

template <typename Op>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 用于查找特定类型节点的 IRVisitor 类模板 NodeFinder
class NodeFinder : public IRVisitor {
 public:
  // 重写 IRVisitor 中的 visit 方法，处理特定类型的节点
  void visit(NodePtr<Op> v) override {
    nodes.push_back((NodePtr<Op>)v);
    IRVisitor::visit(v);
  }

  // 静态方法，查找语句中特定类型节点的实例
  static std::vector<NodePtr<Op>> find(StmtPtr s) {
    NodeFinder<Op> nf;
    s->accept(&nf);
    return nf.nodes;
  }

  // 静态方法，查找表达式中特定类型节点的实例
  static std::vector<NodePtr<Op>> find(ExprPtr e) {
    NodeFinder<Op> nf;
    e->accept(&nf);
    return nf.nodes;
  }

  std::vector<NodePtr<Op>> nodes;  // 存储找到的特定类型节点的实例
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 用于查找语句中所有变量的 IRVisitor 类 VarFinder
class VarFinder : public IRVisitor {
 public:
  // 重写 IRVisitor 中的 visit 方法，处理变量节点
  void visit(VarPtr v) override {
    vars_.insert(v);
    IRVisitor::visit(std::move(v));
  }

  // 静态方法，查找语句中所有变量的实例
  static std::unordered_set<VarPtr> find(StmtPtr s) {
    VarFinder nf;
    s->accept(&nf);
    return nf.vars();
  }

  // 静态方法，查找表达式中所有变量的实例
  static std::unordered_set<VarPtr> find(ExprPtr e) {
    VarFinder nf;
    e->accept(&nf);
    return nf.vars();
  }

  const std::unordered_set<VarPtr>& vars() {
    return vars_;
  }

 private:
  std::unordered_set<VarPtr> vars_;  // 存储找到的变量实例集合
};

// BufFinder 类，用于查找语句或表达式中所有缓冲区的实例
class BufFinder : public IRVisitor {
 public:
  // 重写 IRVisitor 中的 visit 方法，处理缓冲区节点
  void visit(BufPtr v) override {
    bufs_.insert(v);
    IRVisitor::visit(std::move(v));
  }

  // 静态方法，查找语句中所有缓冲区的实例
  static std::unordered_set<BufPtr> find(StmtPtr s) {
    BufFinder nf;
    s->accept(&nf);
    return nf.bufs();
  }

  // 静态方法，查找表达式中所有缓冲区的实例
  static std::unordered_set<BufPtr> find(ExprPtr e) {
    BufFinder nf;
    e->accept(&nf);
    return nf.bufs();
  }

  const std::unordered_set<BufPtr>& bufs() {
    return bufs_;
  }

 private:
  std::unordered_set<BufPtr> bufs_;  // 存储找到的缓冲区实例集合
};

// WritesToBuf 类，用于查找语句中所有对指定缓冲区的写操作
class WritesToBuf : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，接受目标缓冲区的指针
  WritesToBuf(BufPtr target) : target_(std::move(target)) {}

  // 返回找到的写操作语句集合
  std::vector<StmtPtr> writes() {
    return writes_;
  }

  // 静态方法，查找语句中所有对指定缓冲区的写操作
  static std::vector<StmtPtr> find(StmtPtr s, BufPtr b) {
    WritesToBuf finder(std::move(b));
    s->accept(&finder);
    return finder.writes();
  }

 private:
  // 重写 IRVisitor 中的 visit 方法，处理 Store 类型节点
  void visit(StorePtr v) override {
    // 如果节点中的缓冲区与目标缓冲区相同，则将该节点加入写操作集合中
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  // 重写 IRVisitor 中的 visit 方法，处理 AtomicAdd 类型节点
  void visit(AtomicAddPtr v) override {
    // 如果节点中的缓冲区与目标缓冲区相同，则将该节点加入写操作集合中
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  BufPtr target_;  // 目标缓冲区的指针
  std::vector<StmtPtr> writes_;  // 存储找到的写操作语句集合
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
// 命名空间闭合
// 声明一个类 StmtsReadingBuf，继承自 IRVisitor
class StmtsReadingBuf : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，接收一个 BufPtr 类型的参数 target，并将其移动到成员变量 target_ 中
  StmtsReadingBuf(BufPtr target) : target_(std::move(target)) {}

  // 返回 reads_ 成员变量，该函数返回当前类实例中收集到的所有 StmtPtr 类型对象
  std::vector<StmtPtr> reads() {
    return reads_;
  }

  // 静态函数，接收一个 StmtPtr 类型参数 s 和一个 BufPtr 类型参数 b
  // 创建 StmtsReadingBuf 对象 finder，传入参数 b，并调用 s 的 accept 方法进行遍历
  static std::vector<StmtPtr> find(StmtPtr s, BufPtr b) {
    StmtsReadingBuf finder(std::move(b));
    s->accept(&finder);
    return finder.reads();
  }

 private:
  // 检查给定的 StmtPtr s 是否读取了目标缓冲区 target_
  bool readsBuffer(StmtPtr s) {
    // 查找 s 中所有 Load 节点对象
    auto loads = NodeFinder<Load>::find(std::move(s));
    // 遍历 loads，如果某个 Load 对象 l 的缓冲区与 target_ 相等，则返回 true
    for (const auto& l : loads) {
      if (l->buf() == target_) {
        return true;
      }
    }
    // 若未找到符合条件的 Load 对象，则返回 false
    return false;
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 StorePtr 类型的节点 v
  void visit(StorePtr v) override {
    // 如果 v 读取了目标缓冲区 target_，则将 v 添加到 reads_ 中
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 LetPtr 类型的节点 v
  void visit(LetPtr v) override {
    // 如果 v 读取了目标缓冲区 target_，则将 v 添加到 reads_ 中
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 CondPtr 类型的节点 v
  void visit(CondPtr v) override {
    // 如果 v 读取了目标缓冲区 target_，则将 v 添加到 reads_ 中
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 AtomicAddPtr 类型的节点 v
  void visit(AtomicAddPtr v) override {
    // 如果 v 读取了目标缓冲区 target_，则将 v 添加到 reads_ 中
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  // 成员变量，保存目标缓冲区的指针
  BufPtr target_;
  // 成员变量，保存读取到的所有 StmtPtr 类型对象
  std::vector<StmtPtr> reads_;
};

// 声明一个类 ExternalAllocBufFinder，继承自 IRVisitor
class ExternalAllocBufFinder : public IRVisitor {
 public:
  // 重写 IRVisitor 类中的虚函数 visit，处理 ExternalCallWithAllocPtr 类型的节点 v
  void visit(ExternalCallWithAllocPtr v) override {
    // 获取 v 的 buf_out_args() 方法返回的缓冲区集合，插入到 bufs_ 中
    const auto& bufs_out = v->buf_out_args();
    bufs_.insert(bufs_out.begin(), bufs_out.end());
    // 调用 IRVisitor 类的 visit 方法处理 v 的子节点
    IRVisitor::visit(std::move(v));
  }

  // 静态函数，接收一个 StmtPtr 类型参数 s，创建 ExternalAllocBufFinder 对象 f
  // 调用 s 的 accept 方法进行遍历，并返回 bufs_ 成员变量
  static std::unordered_set<BufPtr> find(StmtPtr s) {
    ExternalAllocBufFinder f;
    s->accept(&f);
    return f.bufs();
  }

  // 静态函数，接收一个 ExprPtr 类型参数 e，创建 ExternalAllocBufFinder 对象 f
  // 调用 e 的 accept 方法进行遍历，并返回 bufs_ 成员变量
  static std::unordered_set<BufPtr> find(ExprPtr e) {
    ExternalAllocBufFinder f;
    e->accept(&f);
    return f.bufs();
  }

  // 返回 bufs_ 成员变量，该函数返回当前类实例中收集到的所有 BufPtr 类型对象
  const std::unordered_set<BufPtr>& bufs() {
    return bufs_;
  }

 private:
  // 成员变量，保存收集到的所有 BufPtr 类型对象
  std::unordered_set<BufPtr> bufs_;
};

// 声明一个类 ModifiesVarChecker，继承自 IRVisitor
// 用于检查 IR 中是否修改了特定的 VarPtr 对象
class ModifiesVarChecker : public IRVisitor {
 public:
  // 构造函数，接收一个 VarPtr 类型参数 v，并将其移动到成员变量 var_ 中
  ModifiesVarChecker(VarPtr v) : var_(std::move(v)) {}

  // 静态函数，接收一个 StmtPtr 类型参数 s 和一个 VarPtr 类型参数 v
  // 创建 ModifiesVarChecker 对象 checker，传入参数 v，并调用 s 的 accept 方法进行遍历
  // 返回是否找到了修改了变量 v 的语句
  static bool check(StmtPtr s, VarPtr v) {
    ModifiesVarChecker checker(std::move(v));
    s->accept(&checker);
    return checker.found();
  }

  // 返回 found_ 成员变量，表示是否找到了修改了 var_ 的语句
  bool found() {
    return found_;
  }

 private:
  // 重写 IRVisitor 类中的虚函数 visit，处理 StorePtr 类型的节点 v
  void visit(StorePtr v) override {
    // 如果 v 的缓冲区的 base_handle() 与 var_ 相等，则将 found_ 置为 true，并返回
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    // 否则继续遍历 v 的子节点
    IRVisitor::visit(std::move(v));
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 AtomicAddPtr 类型的节点 v
  void visit(AtomicAddPtr v) override {
    // 如果 v 的缓冲区的 base_handle() 与 var_ 相等，则将 found_ 置为 true，并返回
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    // 否则继续遍历 v 的子节点
    IRVisitor::visit(std::move(v));
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 LetPtr 类型的节点 v
  void visit(LetPtr v) override {
    // 如果 v 的变量 var() 与 var_ 相等，则将 found_ 置为 true，并返回
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    // 否则继续遍历 v 的子节点
    IRVisitor::visit(std::move(v));
  }

  // 重写 IRVisitor 类中的虚函数 visit，处理 ForPtr 类型的节点 v
  void visit(ForPtr v) override {
    // 如果 v 的变量 var() 与 var_ 相等，则将 found_ 置为 true，并返回
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    // 否则继续遍历 v 的子节点
    IRVisitor::visit(std::move(v));
  }

  // 成员变量，保存需要检查的 VarPtr 对象
  VarPtr var_;
  // 成员变量，标记是否找到了修改了 var_ 的语句
  bool found_{false};
};

// 遍历 Block stmt 来确定指定 buf 的活跃范围。该活跃范围由一对整数表示，
// 指定了块 stmts 中访问 buf 的第一个和最后一个 stmt。
class BufLiveRange : public IRVisitor {
 public:
  BufLiveRange(BufPtr b) : buf_(std::move(b)) {}

  // Static method to compute the live range of a buffer within a statement block
  static std::tuple<int32_t, int32_t> liveRange(StmtPtr s, BufPtr b) {
    // Convert the statement to a BlockPtr if possible
    BlockPtr block = to<Block>(std::move(s));
    // We Only analyze buffer live ranges for block stmts.
    if (!block) {
      return std::make_tuple(0, 0);  // Return (0, 0) if the statement is not a block
    }

    // Create an instance of BufLiveRange analyzer for the given buffer
    BufLiveRange analyzer(std::move(b));
    // Analyze the block by accepting the analyzer
    block->accept(&analyzer);
    // Return the computed live range of the buffer
    return analyzer.getLiveRange();
  }

 private:
  // Method to retrieve the computed live range of the buffer
  std::tuple<int32_t, int32_t> getLiveRange() {
    return std::make_tuple(begin_, end_);
  }

  // Check if the statement has any reads from the buffer
  bool hasBufReads(StmtPtr s) {
    auto loads1 = NodeFinder<Load>::find(s);
    for (const auto& l : loads1) {
      if (l->buf() == buf_) {
        return true;
      }
    }
    auto loads2 = NodeFinder<ExternalCall>::find(s);
    for (const auto& l : loads2) {
      for (const auto& lb : l->buf_args()) {
        if (lb == buf_) {
          return true;
        }
      }
    }
    auto loads3 = NodeFinder<ExternalCallWithAlloc>::find(std::move(s));
    for (const auto& l : loads3) {
      for (const auto& lb : l->buf_args()) {
        if (lb == buf_) {
          return true;
        }
      }
    }
    return false;
  }

  // Check if the statement has any writes to the buffer
  bool hasBufWrites(StmtPtr s) {
    auto writes1 = NodeFinder<Store>::find(s);
    for (const auto& w : writes1) {
      if (w->buf() == buf_) {
        return true;
      }
    }
    auto writes2 = NodeFinder<ExternalCall>::find(s);
    for (const auto& w : writes2) {
      if (w->buf() == buf_) {
        return true;
      }
    }
    auto writes3 = NodeFinder<ExternalCallWithAlloc>::find(std::move(s));
    for (const auto& w : writes3) {
      for (const auto& wb : w->buf_out_args()) {
        if (wb == buf_) {
          return true;
        }
      }
    }
    return false;
  }

  // Find accesses to the buffer within the statement and update the live range
  void findAccAndUpdateLiveRange(StmtPtr s) {
    bool has_reads = hasBufReads(s), has_writes = hasBufWrites(std::move(s));
    if (has_reads || has_writes) {
      if (begin_ == -1) {
        begin_ = curr_index_;  // Set the beginning index of the live range
      };
      end_ = curr_index_;  // Update the end index of the live range
    }
  }

  // Visit method for a block of statements to analyze buffer accesses
  void visit(BlockPtr v) override {
    for (const StmtPtr& s : *v) {
      curr_index_ += 1;  // Increment the current index for each statement
      findAccAndUpdateLiveRange(s);  // Analyze the statement for buffer accesses
    }
  }

  BufPtr buf_;  // Buffer pointer for which live range is computed
  int32_t begin_ = -1;  // Start index of the live range
  int32_t end_ = -1;  // End index of the live range
  int32_t curr_index_ = -1;  // Current index during statement traversal
};

// A class that analyzes the given program relevant for Block backend
// It creates a map of multi dim buffers and their flat versions
class CreateBufferMap : public IRVisitor {
 public:
  // Retrieve the map of buffer names to their pointers
  const std::unordered_map<std::string, BufPtr>& getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  // Visit method to process store operations and map buffer names to pointers
  void visit(StorePtr v) override {
    auto load_node = to<Load>(v->value());
    if (load_node) {
      auto t_buf = load_node->buf();
      map_input_to_tensor_bufs_.emplace(t_buf->name_hint(), v->buf());
      // Map the name hint of the buffer to its pointer in the store operation
    }
  }

  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;  // Map of buffer names to pointers
};
    } else {
      // 如果条件不满足，则执行以下逻辑
      // 尝试将 v->value() 转换为 Add 类型节点
      auto add_node = to<Add>(v->value());
      // 尝试将 v->value() 转换为 Mul 类型节点
      auto mul_node = to<Mul>(v->value());
      // 断言 v->value() 必须是 Add 或者 Mul 类型的其中一个，否则抛出错误信息
      // 这表明目前 v->value() 只能是 Add 或者 Mul 类型
      TORCH_INTERNAL_ASSERT(add_node || mul_node, buildErrorMessage());
      // 将 v->buf()->name_hint() 作为键，v->buf() 作为值，添加到 map_input_to_tensor_bufs_ 中
      map_input_to_tensor_bufs_.emplace(v->buf()->name_hint(), v->buf());
    }
    // 对 v->value() 调用 accept 方法，继续遍历处理
    v->value()->accept(this);
  }
  // 存储输入名称到缓冲区指针的映射表
  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
};

// 结束 namespace 'tensorexpr'

// 结束 namespace 'jit'

// 结束 namespace 'torch'
```