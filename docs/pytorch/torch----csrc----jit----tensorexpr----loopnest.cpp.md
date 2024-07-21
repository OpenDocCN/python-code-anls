# `.\pytorch\torch\csrc\jit\tensorexpr\loopnest.cpp`

```
// 引入头文件：TensorExpr 循环嵌套的定义
#include <torch/csrc/jit/tensorexpr/loopnest.h>

// 引入标准库头文件
#include <algorithm>          // 包含算法相关的函数
#include <iostream>           // 输入输出流库
#include <stdexcept>          // 标准异常类库
#include <typeinfo>           // 提供类型信息的头文件
#include <unordered_map>      // 无序映射容器头文件
#include <unordered_set>      // 无序集合容器头文件
#include <utility>            // STL 实用程序组件头文件
#include <vector>             // 向量容器库

// 引入 C10 日志记录库
#include <c10/util/Logging.h>
// 引入 C10 的范围迭代工具
#include <c10/util/irange.h>

// 引入 ATen 核心函数库
#include <ATen/core/functional.h>

// 引入 Torch JIT 日志记录工具
#include <torch/csrc/jit/jit_log.h>
// 引入 TensorExpr 分析模块
#include <torch/csrc/jit/tensorexpr/analysis.h>
// 引入 TensorExpr 边界推断模块
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
// 引入 TensorExpr 评估模块
#include <torch/csrc/jit/tensorexpr/eval.h>
// 引入 TensorExpr 表达式定义
#include <torch/csrc/jit/tensorexpr/expr.h>
// 引入 TensorExpr IR 抽象语法树定义
#include <torch/csrc/jit/tensorexpr/ir.h>
// 引入 TensorExpr IR 克隆器
#include <torch/csrc/jit/tensorexpr/ir_cloner.h>
// 引入 TensorExpr IR 变异器
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
// 引入 TensorExpr IR 打印器
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
// 引入 TensorExpr IR 简化器
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
// 引入 TensorExpr IR 验证器
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
// 引入 TensorExpr 张量定义
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 引入标准异常类库（重复引入）
#include <stdexcept>
// 引入无序映射容器头文件（重复引入）
#include <unordered_map>
// 引入无序集合容器头文件（重复引入）
#include <unordered_set>
// 引入向量容器库（重复引入）
#include <vector>

// Torch JIT TensorExpr 命名空间开始
namespace torch::jit::tensorexpr {

// LoopNest 类的拷贝构造函数实现
LoopNest::LoopNest(const LoopNest& other)
    : root_stmt_(Stmt::clone(other.root_stmt_)),  // 克隆给定 LoopNest 对象的根语句
      output_bufs_(other.output_bufs_) {          // 复制输出缓冲区列表
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);  // 验证根语句的正确性
}

// LoopNest 类的构造函数实现，接受语句指针和输出缓冲区集合作为参数
LoopNest::LoopNest(StmtPtr stmt, std::unordered_set<BufPtr> output_bufs)
    : root_stmt_(std::move(stmt)),         // 移动构造语句指针
      output_bufs_(std::move(output_bufs)) {  // 移动构造输出缓冲区集合
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);  // 验证根语句的正确性
}

// LoopNest 类的构造函数实现，接受输出张量和需要计算的张量列表作为参数
LoopNest::LoopNest(
    const std::vector<Tensor>& output_tensors,
    const std::vector<Tensor>& tensors_to_compute) {
  initialize(output_tensors, tensors_to_compute);  // 初始化 LoopNest 对象
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);  // 验证根语句的正确性
}

// LoopNest 类的构造函数实现，接受输出张量列表作为参数
LoopNest::LoopNest(const std::vector<Tensor>& output_tensors) {
  initialize(output_tensors, output_tensors);  // 初始化 LoopNest 对象
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);  // 验证根语句的正确性
}

// 获取 LoopNest 对象中的中间缓冲区列表
std::vector<BufPtr> LoopNest::getIntermediateBufs() const {
  std::vector<BufPtr> result;  // 中间缓冲区列表
  std::unordered_set<BufPtr> result_set;  // 中间缓冲区的无序集合
  auto input_bufs = getInputBufs();  // 获取输入缓冲区列表
  auto bufs = NodeFinder<Buf>::find(root_stmt_);  // 查找根语句中的所有缓冲区节点
  for (const auto& buf : bufs) {  // 遍历每个缓冲区
    if (!output_bufs_.count(buf) && !input_bufs.count(buf) &&
        !result_set.count(buf)) {  // 如果不属于输出或输入缓冲区，且不在结果集合中
      result.push_back(buf);  // 添加到结果列表中
      result_set.insert(buf);  // 添加到结果集合中
    }
  }
  return result;  // 返回中间缓冲区列表
}

// 获取 LoopNest 对象中的输入缓冲区列表
const std::unordered_set<BufPtr> LoopNest::getInputBufs() const {
  std::unordered_set<BufPtr> result;  // 输入缓冲区的无序集合
  auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);  // 查找根语句中的缓冲区加载或存储用法
  for (auto& kv : buf_load_store_uses) {  // 遍历每个缓冲区的加载或存储用法
    bool has_store = false;  // 是否存在存储标志
    for (auto& use : kv.second) {  // 遍历每个用法
      if (use.isStore) {  // 如果是存储操作
        has_store = true;  // 设置存储标志为 true
        break;  // 跳出循环
      }
    }
    if (!has_store) {  // 如果不存在存储操作
      result.insert(kv.first);  // 将缓冲区插入输入缓冲区集合中
    }
  }
  return result;  // 返回输入缓冲区的无序集合
}

}  // 结束 Torch JIT TensorExpr 命名空间
class IndexFlattener : public IRMutator {
 public:
  // 将输入的语句扁平化处理并返回
  StmtPtr flatten(StmtPtr s) {
    return s->accept_mutator(this);
  }

  // 重写 Load 类型表达式的变异方法
  ExprPtr mutate(LoadPtr v) override {
    // 如果索引的维度为1，则返回原始的 Load 表达式
    if (v->indices().size() == 1) {
      return v;
    }
    // 否则创建一个新的 Load 表达式，传入扁平化后的索引
    return alloc<Load>(
        v->dtype(),
        v->buf(),
        std::vector<ExprPtr>({flatten_index(
            v->buf()->dims(), v->indices(), v->buf()->strides())}));
  }

  // 重写 Store 类型语句的变异方法
  StmtPtr mutate(StorePtr v) override {
    ExprPtr value = v->value();
    // 对值进行变异处理
    ExprPtr new_value = value->accept_mutator(this);
    // 如果索引的维度为1且值没有改变，则返回原始的 Store 语句
    if (v->indices().size() == 1 && value == new_value) {
      return v;
    }
    // 否则更新索引和值，并返回原始的 Store 语句
    std::vector<ExprPtr> indices = {
        flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides())};
    v->set_indices(indices);
    v->set_value(new_value);
    return v;
  }
};

// 检查字符是否为有效的标识符字符
static bool isValidIdentifierChar(char c, size_t pos) {
  return islower(c) || isupper(c) || c == '_' || (pos > 0 && isdigit(c));
}

// 将输入名称中的所有非法字符替换为下划线，并返回清理后的名称
std::string sanitizeName(const std::string& input_name) {
  std::stringstream sanitized_name;
  for (size_t i = 0; i < input_name.size(); ++i) {
    if (isValidIdentifierChar(input_name[i], i)) {
      sanitized_name << input_name[i];
    } else {
      if (i == 0) {
        // 如果名称以非法字符开头，则用 'v' 替换
        sanitized_name << "v";
      }
      // 其他情况下用下划线替换非法字符
      sanitized_name << "_";
    }
  }
  return sanitized_name.str();
}

class VarNameSanitizer : public IRMutator {
 public:
  // 变异方法，用于处理 Buf 类型的表达式
  ExprPtr mutate(BufPtr v) override {
    if (seen_bufs_.count(v)) {
      return v;
    }
    const std::string& name = v->name_hint();
    // 清理名称并获取可用的新名称
    auto new_name = sanitizeName(name);
    if (taken_names_.count(new_name)) {
      new_name = getNextAvailableName(new_name);
    }
    v->set_name_hint(new_name);
    taken_names_.insert(new_name);
    seen_bufs_.insert(v);
    return v;
  }

  // 变异方法，用于处理 Var 类型的表达式
  ExprPtr mutate(VarPtr v) override {
    if (seen_vars_.count(v)) {
      return v;
    }
    const std::string& name = v->name_hint();
    // 清理名称并获取可用的新名称
    auto new_name = sanitizeName(name);
    if (taken_names_.count(new_name)) {
      new_name = getNextAvailableName(new_name);
    }
    v->set_name_hint(new_name);
    taken_names_.insert(new_name);
    seen_vars_.insert(v);
    return v;
  }

  // 变异方法，用于处理 For 类型的语句
  StmtPtr mutate(ForPtr v) override {
    // 获取当前层级的索引变量名称
    auto new_name = getNextAvailableName(getIndexVarNameAtLevel(level_));
    if (seen_index_vars_.count(v->var())) {
      auto new_var = alloc<Var>("", v->var()->dtype());
      Substitute(v, {{v->var(), new_var}});
    }
    v->var()->set_name_hint(new_name);
    seen_index_vars_.insert(v->var());
    seen_vars_.insert(v->var());
    taken_names_.insert(new_name);
    level_++;
    v->body()->accept_mutator(this);
    level_--;
    v->start()->accept_mutator(this);
    v->stop()->accept_mutator(this);
    return v;
  }

  // 获取给定层级的索引变量名称
  std::string getIndexVarNameAtLevel(int level_) {
    int names_num = index_var_names_.size();
    int counter = level_ / names_num;

    // 这里应该返回索引变量名称，但未提供完整的实现
    return "";
  }
};
    if (counter == 0) {
      // 如果计数器为0，返回索引变量名列表中对应 level_ 取模后的名称
      return index_var_names_[level_ % names_num];
    } else {
      // 否则返回索引变量名列表中对应 level_ 取模后的名称加上计数器的字符串表示
      return index_var_names_[level_ % names_num] + std::to_string(counter);
    }
  }
  // 根据给定的基础名称获取下一个可用的名称
  std::string getNextAvailableName(const std::string& base_name) {
    // 初始名称为基础名称
    std::string name = base_name;
    // 计数器初始为0
    int counter = 0;
    // 如果已经使用过该名称，则在名称后加上下划线和计数器的字符串表示，直到找到未使用的名称
    while (taken_names_.count(name)) {
      counter++;
      name = base_name + "_" + std::to_string(counter);
    }
    // 返回找到的未使用的名称
    return name;
  }

 private:
  // 索引变量名列表，包含一组预定义的变量名
  std::vector<std::string> index_var_names_ =
      {"i", "j", "k", "l", "m", "n", "o", "p"};
  // 记录已经被使用的名称集合
  std::unordered_set<std::string> taken_names_;
  // 记录已经看到的索引变量的集合
  std::unordered_set<VarPtr> seen_index_vars_;
  // 记录已经看到的变量的集合
  std::unordered_set<VarPtr> seen_vars_;
  // 记录已经看到的缓冲区的集合
  std::unordered_set<BufPtr> seen_bufs_;
  // 当前级别（默认为0）
  int level_ = 0;
};

// LoopNest 类的成员函数，用于对给定语句进行名称清理
StmtPtr LoopNest::sanitizeNames(StmtPtr s) {
  // 创建 VarNameSanitizer 对象
  VarNameSanitizer r;
  // 对语句 s 应用名称清理器
  s->accept_mutator(&r);
  // 返回清理后的语句 s
  return s;
}

// Vectorizer 类的公共成员函数，用于对 For 循环进行向量化处理
class Vectorizer : public IRMutator {
 public:
  // 向量化 For 循环
  StmtPtr vectorize(ForPtr v) {
    // 获取 For 循环的主体语句
    StmtPtr body = v->body();
    // 获取 For 循环的迭代变量
    VarPtr var = v->var();
    // 获取 For 循环的起始表达式
    ExprPtr start = v->start();
    // 获取 For 循环的结束表达式
    ExprPtr stop = v->stop();

    // 获取起始表达式的整数值
    auto start_imm = intValue(start);
    // 获取结束表达式的整数值
    auto stop_imm = intValue(stop);
    // 如果起始表达式不是常数，则无法进行向量化
    if (!start_imm) {
      // 由于循环起始不是常数，无法进行向量化！
      success_ = false;
      return v;
    }

    // 如果结束表达式不是常数，则无法进行向量化
    if (!stop_imm) {
      // 由于循环结束不是常数，无法进行向量化！
      success_ = false;
      return v;
    }

    // 设置循环变量和起始常数值
    var_ = var;
    start_ = immLike(start, *start_imm);
    // 设置向量化的宽度为结束常数值
    lanes_ = *stop_imm;

    // 对循环体进行 IR 变异处理
    StmtPtr new_body = body->accept_mutator(this);
    // 如果新的循环体与原始的相同，则向量化失败！
    if (new_body == body) {
      // 向量化失败！
      success_ = false;
      return v;
    }

    // 返回变异后的新循环体
    return new_body;
  }

  // 返回向量化是否成功的标志
  bool success() const {
    return success_;
  }

  // 重写 AddPtr 类的变异处理函数
  ExprPtr mutate(AddPtr v) override {
    // 获取加法表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对加法表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) + ExprHandle(inputs[1]);
    });
  }

  // 重写 SubPtr 类的变异处理函数
  ExprPtr mutate(SubPtr v) override {
    // 获取减法表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对减法表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) - ExprHandle(inputs[1]);
    });
  }

  // 重写 MulPtr 类的变异处理函数
  ExprPtr mutate(MulPtr v) override {
    // 获取乘法表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对乘法表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) * ExprHandle(inputs[1]);
    });
  }

  // 重写 DivPtr 类的变异处理函数
  ExprPtr mutate(DivPtr v) override {
    // 获取除法表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对除法表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) / ExprHandle(inputs[1]);
    });
  }

  // 重写 ModPtr 类的变异处理函数
  ExprPtr mutate(ModPtr v) override {
    // 获取取模表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对取模表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) % ExprHandle(inputs[1]);
    });
  }

  // 重写 AndPtr 类的变异处理函数
  ExprPtr mutate(AndPtr v) override {
    // 获取按位与表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对按位与表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) & ExprHandle(inputs[1]);
    });
  }

  // 重写 OrPtr 类的变异处理函数
  ExprPtr mutate(OrPtr v) override {
    // 获取按位或表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对按位或表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) | ExprHandle(inputs[1]);
    });
  }

  // 重写 XorPtr 类的变异处理函数
  ExprPtr mutate(XorPtr v) override {
    // 获取按位异或表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对按位异或表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) ^ ExprHandle(inputs[1]);
    });
  }

  // 重写 LshiftPtr 类的变异处理函数
  ExprPtr mutate(LshiftPtr v) override {
    // 获取左移表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对左移表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) << ExprHandle(inputs[1]);
    });
  }

  // RshiftPtr 类的变异处理函数
  ExprPtr mutate(RshiftPtr v) override {
    // 获取右移表达式的输入
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 尝试对右移表达式进行向量化
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) >> ExprHandle(inputs[1]);
    });
  }

  // 向量化是否成功的私有成员变量
  bool success_ = true;
};
  // 对 Max 表达式进行变异，尝试进行向量化处理
  ExprPtr mutate(MaxPtr v) override {
    // 提取 Max 表达式的左右子表达式
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 调用 try_vectorize 函数，尝试向量化处理，并返回处理后的表达式
    return try_vectorize(v, inputs, [&]() {
      // 创建新的 Max 表达式，保持 NaN 传播属性
      return Max::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  // 对 Min 表达式进行变异，尝试进行向量化处理
  ExprPtr mutate(MinPtr v) override {
    // 提取 Min 表达式的左右子表达式
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    // 调用 try_vectorize 函数，尝试向量化处理，并返回处理后的表达式
    return try_vectorize(v, inputs, [&]() {
      // 创建新的 Min 表达式，保持 NaN 传播属性
      return Min::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  // 对 CompareSelect 表达式进行变异，尝试进行向量化处理
  ExprPtr mutate(CompareSelectPtr v) override {
    // 提取 CompareSelect 表达式的所有输入表达式
    std::vector<ExprPtr> inputs = {
        v->lhs(), v->rhs(), v->ret_val1(), v->ret_val2()};
    // 调用 try_vectorize 函数，尝试向量化处理，并返回处理后的表达式
    return try_vectorize(v, inputs, [&]() {
      // 创建新的 CompareSelect 表达式，保持比较操作、偏置值等属性
      return CompareSelect::make(
          ExprHandle(inputs[0]),
          ExprHandle(inputs[1]),
          ExprHandle(inputs[2]),
          ExprHandle(inputs[3]),
          v->compare_select_op(),
          v->bias());
    });
  }

  // 对 BitCast 表达式进行变异，尝试进行向量化处理
  ExprPtr mutate(BitCastPtr v) override {
    // 提取 BitCast 表达式的源值表达式
    std::vector<ExprPtr> inputs = {v->src_value()};
    // 调用 try_vectorize 函数，尝试向量化处理，并返回处理后的表达式
    return try_vectorize(v, inputs, [&]() {
      // 创建新的 BitCast 表达式，指定数据类型和源值
      return BitCast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  // 对 Cast 表达式进行变异，尝试进行向量化处理
  ExprPtr mutate(CastPtr v) override {
    // 提取 Cast 表达式的源值表达式
    std::vector<ExprPtr> inputs = {v->src_value()};
    // 调用 try_vectorize 函数，尝试向量化处理，并返回处理后的表达式
    return try_vectorize(v, inputs, [&]() {
      // 创建新的 Cast 表达式，指定数据类型和源值
      return Cast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  // 对 Var 表达式进行变异
  ExprPtr mutate(VarPtr v) override {
    // 如果变异的是当前变量 var_
    if (v == var_) {
      // 返回一个 Ramp 表达式，从 start_ 开始，步长为 1，长度为 lanes_
      return Ramp::make(
                 ExprHandle(start_), ExprHandle(immLike(start_, 1)), lanes_)
          .node();
    }

    // 否则直接返回该变量表达式
    return v;
  }

  // 对 Ramp 表达式进行变异
  ExprPtr mutate(RampPtr v) override {
    // 提取 Ramp 表达式的基值和步长
    ExprPtr base = v->base();
    ExprPtr stride = v->stride();

    // 递归调用当前变异器对基值和步长进行变异
    ExprPtr base_new = base->accept_mutator(this);
    ExprPtr stride_new = stride->accept_mutator(this);

    // 如果基值和步长都未改变，则返回原始 Ramp 表达式
    if (base_new == base && stride_new == stride) {
      return v;
    }

    // 否则标记向量化失败，并返回原始 Ramp 表达式
    success_ = false;
    return v;
  }

  // 对 Load 表达式进行变异
  ExprPtr mutate(LoadPtr v) override {
    // 创建指定数据类型的 Load 表达式，从缓冲区 buf 中加载数据
    Dtype dtype(v->dtype().scalar_type(), lanes_);
    BufPtr buf = v->buf();
    // 提取 Load 表达式的索引表达式
    std::vector<ExprPtr> inputs = {v->flat_index()};
    // 调用 try_vectorize 函数，尝试向量化处理，并返回处理后的表达式
    return try_vectorize(v, inputs, [&]() {
      return Load::make(dtype, BufHandle(buf), {ExprHandle(inputs[0])});
    });
  }

  // 对 ReduceOp 表达式进行变异
  ExprPtr mutate(ReduceOpPtr v) override {
    // 创建指定数据类型的 ReduceOp 表达式
    Dtype dtype(v->dtype().scalar_type(), lanes_);

    // 提取 ReduceOp 表达式的主体表达式
    std::vector<ExprPtr> inputs = {v->body()};

    // 尝试向量化处理主体表达式，并返回处理后的表达式
    auto out = try_vectorize(v, inputs, [&]() {
      return ExprHandle(
          alloc<ReduceOp>(inputs[0], v->reduce_args(), v->reducer()));
    });
    return out;
  }

  // 对 Broadcast 表达式进行变异
  ExprPtr mutate(BroadcastPtr v) override {
    // 提取 Broadcast 表达式的值表达式
    ExprPtr val = v->value();
    // 递归调用当前变异器对值表达式进行变异
    ExprPtr new_val = val->accept_mutator(this);
    // 如果值表达式未改变，则返回原始 Broadcast 表达式
    if (new_val == val) {
      return v;
    }

    // 否则标记向量化失败，并返回原始 Broadcast 表达式
    success_ = false;
    return v;
  }

  ExprPtr mutate(IfThenElsePtr v) override {
    ExprPtr condition = v->condition();
    ExprPtr new_condition = condition->accept_mutator(this);
    if (new_condition != condition) {
      // 如果条件表达式被修改过，则无法对 IfThenElse 条件进行向量化
      success_ = false;
      return v;
    }

    std::vector<ExprPtr> inputs = {v->true_value(), v->false_value()};
    return try_vectorize(v, inputs, [&]() {
      return IfThenElse::make(
          ExprHandle(condition), ExprHandle(inputs[0]), ExprHandle(inputs[1]));
    });
  }

  ExprPtr mutate(IntrinsicsPtr v) override {
    std::vector<ExprPtr> inputs = v->params();
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(alloc<Intrinsics>(v->op_type(), inputs));
    });
  }

  StmtPtr mutate(StorePtr v) override {
    BufPtr buf = v->buf();
    std::vector<ExprPtr> inputs = {v->flat_index(), v->value()};
    return try_vectorize(v, inputs, [&]() {
      return Store::make(
          BufHandle(buf), {ExprHandle(inputs[0])}, ExprHandle(inputs[1]));
    });
  }

  StmtPtr mutate(ForPtr v) override {
    VarPtr var = v->var();
    ExprPtr start = v->start();
    ExprPtr stop = v->stop();
    LoopOptions loop_options = v->loop_options();

    ExprPtr new_start = start->accept_mutator(this);
    ExprPtr new_stop = stop->accept_mutator(this);

    if (new_start != start || new_stop != stop) {
      // 无法对具有依赖循环边界的嵌套 For 循环进行向量化
      success_ = false;
      return v;
    }

    StmtPtr body = v->body();
    StmtPtr new_body = body->accept_mutator(this);

    if (new_body == body) {
      return (ForPtr)v;
    }

    return alloc<For>(var, new_start, new_stop, new_body, loop_options);
  }

  StmtPtr mutate(BlockPtr v) override {
    // IRMutator 在原地进行突变。但向量化检查成功与否是通过查找新的语句来判断的。
    // 因此，我们在这里覆盖原地突变，并且如果任何语句发生变化，则在此处创建一个克隆。
    // TODO: 是否可以改变向量化器的逻辑，使我们不再需要这样做？
    bool any_change = false;
    std::vector<StmtPtr> stmts;
    for (const StmtPtr& stmt : *v) {
      StmtPtr stmt_new = stmt->accept_mutator(this);
      if (stmt != stmt_new) {
        any_change = true;
      } else {
        stmt_new = Stmt::clone(stmt);
      }
      if (stmt_new) {
        stmts.push_back(stmt_new);
      }
    }
    if (any_change) {
      return alloc<Block>(stmts);
    }
    return v;
  }

  template <typename T>
  ExprPtr try_vectorize(ExprPtr e, std::vector<ExprPtr>& inputs, T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor().node();
    }

    return e;
  }

  template <typename T>
  StmtPtr try_vectorize(StmtPtr s, std::vector<ExprPtr>& inputs, T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor();
    }

    return (StmtPtr)s;
  }

  bool vectorize_inputs(std::vector<ExprPtr>& inputs) {
    // 判断是否可以对输入进行向量化处理
    // 是否有任何输入向量化的标志，默认为 false
    bool any_vectorized = false;
    // 创建一个新的表达式指针向量，用于存储向量化后的输入
    std::vector<ExprPtr> new_inputs;

    // 尝试对每个输入进行向量化处理
    for (ExprPtr& in : inputs) {
      // 使用当前的变换器对输入进行变换，并返回新的输入表达式指针
      ExprPtr new_in = in->accept_mutator(this);
      // 将新的输入加入到新输入向量中
      new_inputs.push_back(new_in);
      // 如果新的输入与原输入不同，表示成功向量化了至少一个输入
      if (new_in != in) {
        any_vectorized = true;
      }
    }

    // 如果没有任何输入向量化，则返回 false
    if (!any_vectorized) {
      return false;
    }

    // 对于没有被向量化的输入，插入广播操作
    for (size_t i = 0; i < inputs.size(); ++i) {
      // 如果输入没有被向量化，使用 Broadcast::make 创建广播节点，并更新输入表达式
      if (inputs[i] == new_inputs[i]) {
        inputs[i] = Broadcast::make(ExprHandle(inputs[i]), lanes_).node();
      } else {
        // 否则，更新输入表达式为新的输入表达式
        inputs[i] = new_inputs[i];
      }
    }

    // 然后标记当前节点已成功向量化
    return true;
  }

  // 变量指针，初始化为 nullptr
  VarPtr var_ = nullptr;
  // 广播的宽度，初始化为 0
  int lanes_ = 0;
  // 起始表达式指针，初始化为 nullptr
  ExprPtr start_ = nullptr;
  // 操作成功的标志，初始化为 true
  bool success_ = true;
};

// 向量化循环操作
bool LoopNest::vectorize(ForPtr f) {
  // 获取循环块
  BlockPtr b = to<Block>(f->get_parent());
  if (!b) {
    return false;
  }

  // 检查是否能向量化归约轴
  auto reductions = NodeFinder<ReduceOp>::find(f);
  for (const auto& r : reductions) {
    if (std::find(r->reduce_args().begin(), r->reduce_args().end(), f->var()) !=
        r->reduce_args().end()) {
      return false;
    }
  }

  Vectorizer v;
  StmtPtr new_f = nullptr;
  // 克隆循环语句并进行标准化和索引展平
  new_f = Stmt::clone(f);
  normalize(to<For>(new_f));
  new_f = FlattenIndexes(new_f);
  new_f = v.vectorize(to<For>(new_f));
  if (!v.success()) {
    // 在向量化之前克隆了循环。因此，任何部分向量化都会修改克隆版本。
    // 在异常情况下，可以继续使用原始循环语句 f。
    new_f = f;
  }

  if (new_f != f) {
    // 用向量化后的语句替换原始循环语句，并简化生成的新语句
    b->replace_stmt(f, IRSimplifier::simplify(new_f));
    return true;
  }

  // 向量化操作未成功
  return false;
}

// 初始化循环嵌套结构
void LoopNest::initialize(
    const std::vector<Tensor>& output_tensors,
    const std::vector<Tensor>& tensors_to_compute) {
  for (const auto& t : output_tensors) {
    output_bufs_.insert(t.buf());
  }

  std::vector<StmtPtr> loops;
  for (const Tensor& t : tensors_to_compute) {
    StmtPtr loop = t.stmt();
    if (loop->get_parent()) {
      // 如果张量已经在使用中，则发生错误
      std::cerr << "Error: creating a loopnest from already used Tensors\n";
      loops = {};
      break;
    }
    // 展平初始化器
    if (BlockPtr block = to<Block>(loop)) {
      for (const auto& s : block->stmts()) {
        block->remove_stmt(s);
        loops.push_back(s);
      }
    } else {
      loops.push_back(loop);
    }
  }

  // 分配并设置根块语句
  root_stmt_ = alloc<Block>(loops);
}

// 函数内联器类
class FunctionInliner : public IRMutator {
 public:
  // 构造函数，初始化内联函数相关信息
  FunctionInliner(StorePtr producer, std::unordered_set<BufPtr> outputs)
      : buf_(producer->buf()),
        producer_(producer),
        outputs_(std::move(outputs)),
        success_(true) {
    for (const auto& i : producer->indices()) {
      if (auto index_var = to<Var>(i)) {
        index_vars_.insert(index_var);
        producer_index_vars_.push_back(index_var);
      } else {
        // 如果索引可能是常数，则其维度必须为1（因为不支持原地写入）。解决问题 52581。
        auto index_val = evalInt(i);
        if (!index_val || *index_val != 0) {
          success_ = false;
          break;
        }
        producer_index_vars_.push_back(nullptr);
      }
    }
  }

  // 返回内联是否成功
  bool success() const {
    return success_;
  }

 private:
  // 变异加载函数，处理加载操作
  ExprPtr mutate_loads(BufPtr buf, std::vector<ExprPtr> dims) {
    std::vector<VarPtr> index_vars;
    // 检查生产者和消费者表达式的维度是否匹配
    if (buf->ndim() != producer_index_vars_.size()) {
      // 在内联器中生产者和消费者表达式的维度不匹配
      success_ = false;
      return nullptr;
    }
    // 遍历缓冲区的维度范围
    for (const auto i : c10::irange(buf->ndim())) {
      // 获取生产者索引变量中的参数
      VarPtr func_callee_arg = producer_index_vars_.at(i);
      // 获取调用者维度参数
      ExprPtr func_caller_param = dims.at(i);
      // 如果生产者索引参数为空，则跳过
      if (func_callee_arg == nullptr) {
        continue;
      }
      // 查找是否存在重复的变量映射
      auto iter = inline_mapping_.find(func_callee_arg);
      if (iter != inline_mapping_.end()) {
        // 如果存在重复的变量映射，设置成功标志为 false，并返回空指针
        success_ = false;
        return nullptr;
      }
      // 将每个函数参数映射到其源名称
      inline_mapping_[func_callee_arg] = func_caller_param;
      // 调试输出映射信息
      GRAPH_DEBUG(
          "ComputeInline: Inline mapping: ",
          std::to_string(func_callee_arg),
          " -> ",
          std::to_string(func_caller_param));
      // 将索引变量添加到列表中
      index_vars.push_back(func_callee_arg);
    }

    // 调用实际的替换过程
    ExprPtr body = producer_->value();
    // 调试输出替换前的表达式体
    GRAPH_DEBUG("ComputeInline: Before rewriting body: ", std::to_string(body));
    // 使用当前变异器克隆表达式并进行变异
    ExprPtr result = Expr::clone(body)->accept_mutator(this);
    // 调试输出替换后的表达式体
    GRAPH_DEBUG(
        "ComputeInline: After rewriting body: ", std::to_string(result));

    // 移除为这些函数参数创建的映射
    for (const auto& v : index_vars) {
      for (auto& pair : random_bindings_) {
        // 如果从绑定中删除变量成功
        if (pair.second.erase(v)) {
          // 获取变量的内联映射
          ExprPtr inlined = inline_mapping_[v];
          // 查找内联表达式中的变量
          for (const auto& nv : VarFinder::find(inlined)) {
            // 将这些新变量插入到绑定中
            pair.second.insert(nv);
          }
        }
      }
      // 调试输出删除内联映射信息
      GRAPH_DEBUG("ComputeInline: Inline mapping: erasing", std::to_string(v));
      // 从内联映射中擦除变量
      inline_mapping_.erase(v);
    }
    // 返回结果表达式
    return result;
  }

  // 处理 LoadPtr 类型的变异
  ExprPtr mutate(LoadPtr v) override {
    // 如果操作失败，则返回原始 LoadPtr
    if (!success()) {
      return v;
    }
    // 获取缓冲区指针
    BufPtr buf = v->buf();
    // 如果缓冲区指针不匹配当前缓冲区，则继续变异处理
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    // 如果索引数量与缓冲区的维度数不匹配
    if (v->indices().size() != buf->ndim()) {
      // 在融合器中，索引数量与缓冲区秩不匹配
      success_ = false;
      return v;
    }
    // 对加载操作进行变异处理
    auto result = mutate_loads(buf, v->indices());
    // 如果未成功内联，则返回原始 LoadPtr
    if (!result) {
      // 如果不能成功内联，则返回给定的加载操作
      success_ = false;
      return v;
    }
    // 返回处理后的结果
    return result;
  }

  // 处理 VarPtr 类型的变异
  ExprPtr mutate(VarPtr v) override {
    // 如果操作失败，则返回原始 VarPtr
    if (!success()) {
      return v;
    }
    // 查找变量在内联映射中的位置
    auto iter = inline_mapping_.find(v);
    // 如果未找到变量的内联映射，则返回原始 VarPtr
    if (iter == inline_mapping_.end()) {
      return v;
    } else {
      // 获取变量的表达式
      ExprPtr expr = iter->second;
      // 继续变异处理查找表中的值
      return expr->accept_mutator(this);
    }
  }

  // 处理 IntrinsicsPtr 类型的变异
  ExprPtr mutate(IntrinsicsPtr v) override {
    // 如果操作失败，则返回原始 IntrinsicsPtr
    if (!success()) {
      return v;
    }
    // 如果不在生产者中或者操作类型不是 kRand，则继续变异处理
    if (!in_producer_ || v->op_type() != kRand) {
      return IRMutator::mutate(v);
    }

    // 为随机变量创建新的 Let 语句，可以多次引用并解析相同的值（即存储在标量而不是张量中）
    const std::string& name = buf_->name_hint();
  // 创建一个名为 new_var 的指针，该指针指向一个通过 alloc 函数分配的 Var 对象，其名称为 name，数据类型与 v 相同。
  VarPtr new_var = alloc<Var>(name, v->dtype());
  // 将新创建的 Var 对象与 v 绑定，并将其添加到 random_bindings_ 中，关联的索引为 index_vars_。
  random_bindings_[alloc<Let>(new_var, v)] = index_vars_;
  // 在调试模式下输出日志，指示已创建 new_var 的随机绑定。
  GRAPH_DEBUG(
      "ComputeInline: created random bindings for ", std::to_string(new_var));
  // 返回新创建的 Var 对象 new_var。
  return new_var;
}

// 从内联函数中移除存储写操作。
StmtPtr mutate(StorePtr v) override {
  if (!success()) {
    return v;
  }
  // 如果 v 是 producer_，且 buf_ 不在输出集合中，则保留其语句；否则移除。
  if (v == producer_ && !outputs_.count(buf_)) {
    in_producer_ = true;
    // 对 producer_ 进行变异操作，并更新 producer_。
    producer_ = to<Store>(IRMutator::mutate(v));
    if (!producer_) {
      // 在融合器中，输出缓冲区的生产者语句应保持非空。
      success_ = false;
      return v;
    }
    in_producer_ = false;
    // 返回空指针，表示成功移除存储写操作。
    return nullptr;
  } else {
    // 对非 producer_ 的存储写操作进行常规变异操作。
    return IRMutator::mutate(v);
  }
}

// 插入在随机内在函数转换为变量后必须插入的任意随机内在函数。
StmtPtr mutate(BlockPtr v) override {
  if (!success()) {
    return v;
  }
  // 创建一个语句向量 stmts，用于存储变异后的语句。
  std::vector<StmtPtr> stmts;
  // 遍历块中的每个语句，并通过接受变异器来变异它们。
  for (const StmtPtr& stmt : *v) {
    StmtPtr stmt_new = stmt->accept_mutator(this);
    if (!stmt_new) {
      continue;
    }

    // 如果变异后的语句与原语句相同，则克隆原语句。
    if (stmt == stmt_new) {
      stmt_new = Stmt::clone(stmt);
    }

    // 将变异后的语句添加到 stmts 中。
    stmts.push_back(stmt_new);
  }

  // 创建一个新的 Block 对象，包含变异后的语句列表，并返回。
  return Block::make(stmts);
}

// 变异 For 循环语句。
StmtPtr mutate(ForPtr v) override {
  if (!success()) {
    return v;
  }
  // 对 For 循环进行变异，并将结果保存到 res 中。
  ForPtr res = to<For>(IRMutator::mutate(v));
  if (!res) {
    return nullptr;
  }

  // 查找应该在此循环体内定义的任意随机绑定。
  std::vector<LetPtr> bindings_this_loop;
  VarPtr fv = v->var();
  for (auto& pair : random_bindings_) {
    auto& index_var = pair.second;
    if (index_var.erase(fv)) {
      bindings_this_loop.push_back(pair.first);
    }
  }

  // 将找到的随机绑定语句插入到循环体的开头，并从 random_bindings_ 中删除。
  for (const auto& l : bindings_this_loop) {
    res->body()->prepend_stmt(l);
    random_bindings_.erase(l);
  }

  // 返回变异后的 For 循环语句。
  return res;
}
};

// 在给定的语句中计算内联实现
static StmtPtr computeInlineImpl(
    BufPtr b,  // 要内联的缓冲区指针
    StmtPtr stmt,  // 要处理的语句指针
    const std::unordered_set<BufPtr>& output_bufs  // 输出缓冲区的集合
) {
  // 如果缓冲区在外部调用中被使用或定义，则无法内联
  auto buf_load_store_uses = findLoadOrStoreUses(stmt);
  if (!buf_load_store_uses.count(b)) {
    return nullptr;
  }
  for (auto& use : buf_load_store_uses.at(b)) {
    StmtPtr s = use.s;
    if (to<ExternalCall>(s) || to<ExternalCallWithAlloc>(s)) {
      return nullptr;
    }
  }

  // 查找生产者
  StorePtr relevant_store{nullptr};
  auto stores = NodeFinder<Store>::find(stmt);
  for (const auto& s : stores) {
    if (s->buf() == b) {
      auto reductions = NodeFinder<ReduceOp>::find(s);
      if (!reductions.empty()) {
        // 无法内联带有归约计算的缓冲区
        return nullptr;
      }
      if (relevant_store != nullptr) {
        // 无法内联带有多个张量的缓冲区
        return nullptr;
      }
      relevant_store = s;
    }
  }

  if (!relevant_store) {
    // 无法找到要内联到 fuser 中的相关存储
    return nullptr;
  }

  // 调试输出相关存储的信息
  GRAPH_DEBUG("ComputeInline: Def: ", std::to_string(relevant_store));
  // 创建 FunctionInliner 对象并尝试内联操作
  FunctionInliner inliner(relevant_store, output_bufs);
  auto result = stmt->accept_mutator(&inliner);
  if (inliner.success()) {
    return result;
  }
  return nullptr;
}

// 尝试在根语句的克隆上进行内联操作，以确保始终处于有效状态
bool LoopNest::computeInline(BufPtr b) {
  // 克隆根语句
  auto stmt_copy = Stmt::clone(root_stmt_);
  // 尝试在克隆语句上进行内联操作
  auto try_inline = computeInlineImpl(b, stmt_copy, output_bufs_);
  if (!try_inline) {
    return false;
  }
  // 在实际根语句上执行内联操作
  root_stmt_ = computeInlineImpl(b, root_stmt_, output_bufs_);
  return true;
}

// 尝试内联给定的语句
bool LoopNest::computeInline(StmtPtr s) {
  auto s_store = to<Store>(s);
  if (s_store == nullptr) {
    // 找不到要内联的缓冲区生产者
    return false;
  }
  // 调用重载的 computeInline 方法以内联缓冲区
  return computeInline(s_store->buf());
}

// 内联中间缓冲区，允许重复工作可能会减慢 CPU 代码生成速度，但在 GPU 上启用以避免跨块的同步逻辑
// 内联简单读取操作不会导致重复工作
void LoopNest::inlineIntermediateBufs(bool allow_duplicated_work) {
  std::unordered_set<BufPtr> bufs_to_inline;

  auto intermediate_bufs = getIntermediateBufs();
  if (allow_duplicated_work) {
    // 将中间缓冲区添加到要内联的缓冲区集合中
    bufs_to_inline.insert(intermediate_bufs.begin(), intermediate_bufs.end());
  } else {
    // 查找根语句中的加载或存储使用
    auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
    auto input_bufs = getInputBufs();
    // ...
    // 遍历中间缓冲区列表
    for (const auto& buf : intermediate_bufs) {
      // 断言确保每个缓冲区在加载或存储使用中被引用
      TORCH_INTERNAL_ASSERT(
          buf_load_store_uses.count(buf),
          buildErrorMessage(
              "Could not find uses of buf '" + buf->name_hint() +
              "' in the fuser."));
      // 获取当前缓冲区的加载或存储使用列表
      std::vector<BufLoadOrStoreUse>& uses = buf_load_store_uses[buf];
      // 过滤出所有存储操作
      auto stores = c10::filter(
          uses, [](const BufLoadOrStoreUse& use) { return use.isStore; });

      // 如果中间缓冲区是由读取输入张量而形成的，总是进行内联，因为不会重复工作并且避免了中间缓冲区
      if (stores.size() == 1) {
        if (auto store = to<Store>(stores[0].s)) {
          // 尝试将存储操作转换为加载操作
          auto input_as_load = to<Load>(store->value());
          // 如果成功且加载操作的缓冲区在输入缓冲区列表中，则将当前缓冲区标记为内联
          if (input_as_load && input_bufs.count(input_as_load->buf())) {
            bufs_to_inline.insert(buf);
            continue;
          }
        } else {
          // 如果不是存储操作，则必须是外部调用
          TORCH_INTERNAL_ASSERT(
              to<ExternalCall>(stores[0].s) ||
                  to<ExternalCallWithAlloc>(stores[0].s),
              buildErrorMessage(
                  "Expected stmt: " + std::to_string(stores[0].s) +
                  "\nto be either a Store or an ExternalCall in the fuser."));
        }
      }

      // 每个缓冲区至少有一个存储操作（如果有多个存储操作，则无法进行内联）
      size_t reads = uses.size() - 1;
      // 如果只有一个读取操作，可以进行内联而不重复工作
      if (reads <= 1) {
        bufs_to_inline.insert(buf);
      }
    }
  }

  // 如果允许重复工作，则将所有输出缓冲区标记为内联
  if (allow_duplicated_work) {
    bufs_to_inline.insert(output_bufs_.begin(), output_bufs_.end());
  }

  // 对所有标记为需要内联的缓冲区执行内联计算
  for (const auto& b : bufs_to_inline) {
    computeInline(b);
  }
}

// TODO: Unify with DepTracker
// 定义一个类 LoadOrStoreUseFinder，继承自 IRVisitor 类
class LoadOrStoreUseFinder : public IRVisitor {
 public:
  // 查找给定语句中所有的缓冲区加载或存储用法，返回一个映射
  std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findUses(
      StmtPtr s) {
    // 清空之前的使用映射
    uses_.clear();
    // 接受并访问给定的语句 s
    s->accept(this);
    // 返回找到的使用映射
    return uses_;
  }

 private:
  // 处理 StorePtr 类型的访问，记录缓冲区的存储用法
  void visit(StorePtr v) override {
    // 如果该存储操作是第一次出现，则记录下来
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(StmtPtr)v, true});
    }
    // 更新最后访问的语句为当前存储操作
    last_stmt_ = (StmtPtr)v;
    // 继续访问该节点的子节点
    IRVisitor::visit(v);
  }

  // 处理 ExternalCallPtr 类型的访问，记录缓冲区的使用情况和参数的加载情况
  void visit(ExternalCallPtr v) override {
    // 如果该调用操作中涉及的缓冲区是第一次存储，则记录下来
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(StmtPtr)v, true});
    }
    // 更新最后访问的语句为当前外部调用操作
    last_stmt_ = (StmtPtr)v;

    // 对于每一个输入缓冲区参数，记录其加载用法
    for (const BufPtr& input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    // 继续访问该节点的子节点
    IRVisitor::visit(v);
  }

  // 处理 ExternalCallWithAllocPtr 类型的访问，记录缓冲区的使用情况和参数的加载情况
  void visit(ExternalCallWithAllocPtr v) override {
    // 对于每一个输出缓冲区参数，记录其存储用法
    for (const auto& out_buf : v->buf_out_args()) {
      if (stores_[out_buf].insert(last_stmt_).second) {
        uses_[out_buf].push_back({(StmtPtr)v, true});
      }
    }
    // 更新最后访问的语句为当前带分配的外部调用操作
    last_stmt_ = (StmtPtr)v;

    // 对于每一个输入缓冲区参数，记录其加载用法
    for (const auto& input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    // 继续访问该节点的子节点
    IRVisitor::visit(v);
  }

  // 处理 LoadPtr 类型的访问，记录缓冲区的加载用法
  void visit(LoadPtr v) override {
    // 如果该加载操作是第一次出现，则记录下来
    if (loads_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({last_stmt_, false});
    }
    // 继续访问该节点的子节点
    IRVisitor::visit(v);
  }

  // 记录最后访问的语句节点
  StmtPtr last_stmt_ = nullptr;
  // 存储缓冲区的加载或存储用法的映射
  std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> uses_;

  // 用于保持加载和存储结果的唯一性的集合
  std::unordered_map<BufPtr, std::unordered_set<StmtPtr>> loads_;
  std::unordered_map<BufPtr, std::unordered_set<StmtPtr>> stores_;
};

// 查找给定语句 s 中所有缓冲区的加载或存储用法，返回一个映射
std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findLoadOrStoreUses(
    StmtPtr s) {
  // 创建 LoadOrStoreUseFinder 类的实例
  LoadOrStoreUseFinder uf;
  // 调用实例的 findUses 方法进行查找并返回结果
  return uf.findUses(s);
}

// 定义一个类 ContainedStmtsFinder，继承自 IRVisitor 类
class ContainedStmtsFinder : public IRVisitor {
 public:
  // 查找给定语句 s 的所有子语句中包含的所有存储和块，返回一个集合
  const std::unordered_set<StmtPtr>& findContainedStmts(StmtPtr s) {
    // 清空之前的包含语句集合
    contained_.clear();
    // 接受并访问给定的语句 s
    s->accept(this);
    // 返回找到的包含语句集合
    return contained_;
  }

 private:
  // 处理 StorePtr 类型的访问，将该节点加入包含语句集合中
  void visit(StorePtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }

  // 处理 ExternalCallPtr 类型的访问，将该节点加入包含语句集合中
  void visit(ExternalCallPtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }

  // 处理 ExternalCallWithAllocPtr 类型的访问，将该节点加入包含语句集合中
  void visit(ExternalCallWithAllocPtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }

  // 处理 BlockPtr 类型的访问，将该节点加入包含语句集合中
  void visit(BlockPtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }

  // 存储包含语句的集合
  std::unordered_set<StmtPtr> contained_;
};

// 定义一个类 StmtDeleter，继承自 IRMutator 类
class StmtDeleter : public IRMutator {
 public:
  // 构造函数，初始化需要删除的目标语句集合
  StmtDeleter(const std::unordered_set<StmtPtr>& targets) : targets_(targets) {}

 private:
  // 处理 BlockPtr 类型的语句节点，用于删除目标语句集合中的语句
  StmtPtr mutate(BlockPtr v) override {
    // 存储变异后的语句节点的向量
    std::vector<StmtPtr> stmts;
    // 对于传入的语句向量 `v` 中的每一个语句 `s` 进行迭代处理
    for (const auto& s : v->stmts()) {
      // 检查当前语句 `s` 是否在目标集合 `targets_` 中
      if (targets_.count(s) == 0) {
        // 如果 `s` 不在目标集合中，则进行深度复制和变异操作
        StmtPtr ns = s->accept_mutator(this);
        // 如果变异操作返回了新的语句 `ns`，则将其克隆并加入到 `stmts` 中
        if (ns) {
          stmts.push_back(Stmt::clone(ns));
        }
      }
    }

    // 构造并返回一个新的代码块对象，其中包含经处理后的语句 `stmts`
    return Block::make(stmts);
  }

  // 目标语句的集合，用于确定哪些语句不需要处理
  const std::unordered_set<StmtPtr>& targets_;
    // 定义 LoopNest 类的成员函数 eliminateDeadStores，用于消除死存储操作
void LoopNest::eliminateDeadStores() {
    // 引入 analysis 命名空间
  using namespace analysis;
  // 创建 MemDependencyChecker 对象，使用输入和输出缓冲区初始化
  MemDependencyChecker checker(getInputBufs(), getOutputBufs());
  // 分析器对根语句 root_stmt_ 进行访问
  root_stmt_->accept(&checker);

  // 存储死存储的无序集合
  std::unordered_set<StmtPtr> deadStores;
  // 存储输出缓冲区的访问信息的向量
  std::vector<std::shared_ptr<AccessInfo>> outputAccesses;
  // 遍历所有输出缓冲区，获取其访问信息并存储到 outputAccesses 中
  for (const auto& o : getOutputBufs()) {
    outputAccesses.push_back(checker.output(o));
  }

  // 遍历所有历史访问信息
  for (auto& info : checker.getHistory()) {
    // 如果不是写操作，跳过当前循环
    if (!info->isWrite()) {
      continue;
    }
    // 是否找到间接依赖的标志
    bool found = false;

    // 遍历输出访问信息列表
    for (auto& output : outputAccesses) {
      // 检查当前访问信息是否间接依赖于 info
      if (checker.dependsIndirectly(output, info)) {
        found = true;
        break;
      }
    }

    // 如果未找到间接依赖，则将当前语句添加到死存储集合中
    if (!found) {
      deadStores.insert(info->stmt());
    }
  }

  // 创建 StmtDeleter 对象，用于删除死存储的语句
  StmtDeleter deleter(deadStores);
  // 使用 StmtDeleter 对象修改 root_stmt_，返回新的根语句
  root_stmt_ = root_stmt_->accept_mutator(&deleter);
}

// 准备用于代码生成的函数
void LoopNest::prepareForCodegen() {
  // 扩展约简操作
  ReductionExpander reduceExpander;
  // 执行约简扩展，更新 root_stmt_
  root_stmt_ = reduceExpander.expand(root_stmt_);

  // 展开索引，更新 root_stmt_
  root_stmt_ = FlattenIndexes(root_stmt_);
}

// 匿名命名空间，定义 IfThenElseReplacer 类
class IfThenElseReplacer : public IRCloner {
 public:
  // 构造函数，用于替换 if-then-else 表达式中的条件
  IfThenElseReplacer(IfThenElsePtr to_replace, ExprPtr new_expr)
      : to_replace_(std::move(to_replace)), new_expr_(std::move(new_expr)) {}

  // 重写 mutate 函数，用于遍历并替换 if-then-else 表达式中的条件
  ExprPtr mutate(IfThenElsePtr i) override {
    if (i == to_replace_) {
      return new_expr_;
    }
    return IRCloner::mutate(i);
  }

 private:
  IfThenElsePtr to_replace_;  // 待替换的 if-then-else 表达式
  ExprPtr new_expr_;          // 新的表达式
};

// 检查条件是否可以优化的函数
bool isConditionOptimizable(
    ExprPtr condition,
    VarPtr* cond_var,
    ExprPtr* compared_value) {
  // 尝试将 condition 转换为 CompareSelect 类型
  auto cs = to<CompareSelect>(condition);
  // 如果成功转换，并且比较操作是 kLT（小于）
  if (cs && cs->compare_select_op() == kLT) {
    // 尝试获取 cs 的左操作数作为 Var
    auto var = to<Var>(cs->lhs());
    // 如果成功获取到 Var
    if (var) {
      // 将 cond_var 设置为找到的 Var
      *cond_var = var;
      // 将 compared_value 设置为 cs 的右操作数
      *compared_value = cs->rhs();
      return true;
    }
  }
  return false;
}

// 检查 if-then-else 表达式是否由 `aten::cat` 生成的条件函数
bool isConditionalFromCat(
    IfThenElsePtr ite,
    VarPtr* cond_var,
    std::vector<ExprPtr>* comp_values,
    std::vector<ExprPtr>* sub_exprs) {
      // 初始化一个空指针 var，用于存储条件变量
      VarPtr var = nullptr;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 初始化一个空指针 comp_value，用于存储比较值表达式
      ExprPtr comp_value;
      // 调用 isConditionOptimizable 函数，检查是否可以优化条件表达式
      if (isConditionOptimizable(ite->condition(), &var, &comp_value)) {
        // 如果 cond_var 是空指针，则将其设为 var
        if (*cond_var == nullptr) {
          *cond_var = var;
        } else if (*cond_var != var) {
          // 如果 cond_var 不等于 var，说明发现了不同的条件变量在嵌套的 if-then-else 表达式中
          // 无法优化这种情况，直接返回 false
          return false;
        }
        // 尝试将 true_value 转换为 IfThenElse 类型
        auto true_ite = to<IfThenElse>(ite->true_value());
        if (true_ite) {
          // 如果成功转换，递归调用 isConditionalFromCat 处理 true 分支
          if (!isConditionalFromCat(true_ite, cond_var, comp_values, sub_exprs)) {
            return false;
          }
        } else {
          // 如果 true_value 不是 IfThenElse 类型，将其加入 sub_exprs
          sub_exprs->push_back(ite->true_value());
        }
        // 尝试将 false_value 转换为 IfThenElse 类型
        auto false_ite = to<IfThenElse>(ite->false_value());
        if (false_ite) {
          // 如果 false_value 是 IfThenElse 类型，直接返回 false
          return false;
        }
        // 将 comp_value 添加到 comp_values 中
        comp_values->push_back(comp_value);
        // 将 false_value 加入 sub_exprs
        sub_exprs->push_back(ite->false_value());
        // 函数成功执行，返回 true
        return true;
      }
      // 如果条件不可优化，直接返回 false
      return false;
    }
} // namespace

// 检查给定的表达式列表是否都是常量且已排序
bool areConstantsAndSorted(const std::vector<ExprPtr>& comp_values) {
  // 创建一个整数向量来存储比较的常量值
  std::vector<int> comp_consts;
  comp_consts.reserve(comp_values.size());
  // 遍历给定的表达式列表
  for (const auto& c : comp_values) {
    // 如果表达式不是常量，则返回 false
    if (!c->isConstant()) {
      return false;
    }
    // 将表达式转换为整数并添加到常量向量中
    comp_consts.push_back(immediateAs<int>(c));
  }
  // 检查常量向量是否已排序并返回结果
  return std::is_sorted(comp_consts.begin(), comp_consts.end());
}

} // namespace

// 优化条件语句的函数
bool LoopNest::optimizeConditionals() {
  // 在 root_stmt_ 中查找所有的 Store 节点
  auto stores = NodeFinder<Store>::find(root_stmt_);
  // 用于存储需要分割的 For 循环的集合
  std::unordered_set<ForPtr> split_fors;
  // 遍历每个 Store 节点
  for (const auto& store : stores) {
    // 条件变量初始化为 nullptr
    VarPtr cond_var = nullptr;
    // comp_values 用于收集比较的值，初始值为 0，用于检查预期的模式
    std::vector<ExprPtr> comp_values;
    // sub_exprs 用于存储子表达式
    std::vector<ExprPtr> sub_exprs;
    // 查找当前 Store 节点中的所有 IfThenElse 表达式
    auto ifthenelse_exprs = NodeFinder<IfThenElse>::find(store);
    // 如果没有找到 IfThenElse 表达式则继续下一个 Store
    if (ifthenelse_exprs.empty()) {
      continue;
    }
    // 检查第一个 IfThenElse 表达式是否符合所需格式
    if (!isConditionalFromCat(
            ifthenelse_exprs.front(), &cond_var, &comp_values, &sub_exprs)) {
      continue;
    }
    // 断言 comp_values 至少包含一个表达式
    TORCH_INTERNAL_ASSERT(
        !comp_values.empty(),
        buildErrorMessage(
            "Expected at least one expression in optimizeConditional in the fuser."));
    // 将初始值 0 插入到 comp_values 的开头
    comp_values.insert(comp_values.begin(), immLike(comp_values[0], 0));

    // 获取当前 Store 节点所属的所有循环语句
    auto fors = getLoopStmtsFor(store);
    // 如果条件变量不等于最内层循环变量，则继续下一个 Store
    if (cond_var != fors.back()->var()) {
      continue;
    }
    // 获取需要分割的循环对象
    auto for_to_split = fors.back();
    // 如果循环对象不是标准化的则不优化条件语句
    if (!LoopNest::isNormalized(for_to_split)) {
      continue;
    }
    if (split_fors.count(for_to_split)) {
      // 检查是否已经对该循环进行了拆分，这通常发生在优化条件语句时。
      // 如果多个条件需要拆分同一个循环，确保这些条件完全相同，并且只拆分一次。
      //
      // 当前情况下，不支持处理这种情况。
      continue;
    }
    split_fors.insert(for_to_split);

    // `comp_values` 需要包含结束边界，即 `for_to_split` 的终止值。
    comp_values.push_back(for_to_split->stop());

    // 检查所有 `comp_values` 是否都是常量并且已经排序。
    if (!areConstantsAndSorted(comp_values)) {
      continue;
    }

    // 从当前存储中移除所有的 if-then-else 表达式，并为每个子表达式创建一个新循环。
    std::vector<StmtPtr> split_loops;
    auto cond_to_replace = ifthenelse_exprs.front();
    for (size_t i = 0; i < sub_exprs.size(); ++i) {
      // 使用条件替换器移除 if-then-else 表达式，并生成新的存储。
      IfThenElseReplacer ifthenelseReplacer(cond_to_replace, sub_exprs[i]);
      auto new_store = store->accept_mutator(&ifthenelseReplacer);
      // 替换循环体，生成新的循环。
      auto new_for_body =
          for_to_split->body()->clone_and_replace(store, new_store);
      auto new_for = alloc<For>(
          for_to_split->var(),
          comp_values[i],
          comp_values[i + 1],
          new_for_body);
      // 规范化新生成的循环。
      LoopNest::normalize(new_for);
      split_loops.push_back(new_for);
    }
    // 将拆分后的循环插入到原始循环的父块中。
    auto par = to<Block>(for_to_split->get_parent());
    par->replace_stmt(for_to_split, alloc<Block>(split_loops));
  }
  // 简化整个程序，消除冗余和简化表达式。
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  // 返回 true，表示程序成功处理。
  return true;
}

void LoopNest::vectorizeInnerLoops() {
  // 初始化存储内层循环和工作列表的向量
  std::vector<ForPtr> innerLoops;
  std::vector<ForPtr> worklist;

  // 查找最外层的For循环
  if (ForPtr rootF = to<For>(root_stmt_)) {
    worklist.push_back(rootF);
  } else if (BlockPtr body = to<Block>(root_stmt_)) {
    std::vector<BlockPtr> blocks = {body};
    while (!blocks.empty()) {
      BlockPtr b = blocks.back();
      blocks.pop_back();

      // 遍历当前块中的语句
      for (const StmtPtr& s : *b) {
        if (ForPtr f = to<For>(s)) {
          worklist.push_back(f);
        } else if (BlockPtr b2 = to<Block>(s)) {
          blocks.push_back(b2);
        }
      }
    }
  }

  // 遍历For循环嵌套，找到内层循环，作为矢量化的候选对象
  while (!worklist.empty()) {
    ForPtr f = worklist.back();
    worklist.pop_back();

    bool containsSubLoops = false;
    if (BlockPtr body = to<Block>(f->body())) {
      // 检查循环体中是否包含子循环
      for (const StmtPtr& s2 : *body) {
        if (ForPtr f2 = to<For>(s2)) {
          containsSubLoops = true;
          worklist.push_back(f2);
        }
      }
    }

    // 如果没有子循环，则将当前循环加入内层循环列表
    if (!containsSubLoops) {
      innerLoops.push_back(f);
    }
  }

  // 对内层循环进行矢量化处理
  for (const ForPtr& loop : innerLoops) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr split1;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr tail1;

    static const int kBodyVectorWidth = 8;
    // 将循环分割成指定宽度的子循环和尾部
    splitWithTail(loop, kBodyVectorWidth, &split1, &tail1);
    vectorize(split1);

    if (tail1) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      ForPtr split2;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      ForPtr tail2;
      static const int kTailVectorWidth = 4;
      // 对尾部再次分割成指定宽度的子循环和尾部
      splitWithTail(tail1, kTailVectorWidth, &split2, &tail2);
      vectorize(split2);
    }
  }
}

void LoopNest::sliceHead(ForPtr f, int factor, ForPtr* head, ForPtr* tail) {
  if (intValue(f->start()) && intValue(f->stop())) {
    auto start_val = *intValue(f->start());
    auto stop_val = *intValue(f->stop());
    auto size_val = stop_val - start_val;
    // 如果因子大于等于循环范围，则直接将整个循环作为头部
    if (factor >= size_val) {
      *head = f;
      *tail = nullptr;
      return;
    }
  }

  // 如果传入的循环为空，抛出异常
  if (!f) {
    throw malformed_input("sliceHead attempted on null loop");
  }

  // 获取循环的父级块
  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    // 如果循环没有父级块，抛出异常
    throw malformed_input("sliceHead attempted on loop with no parent");
  }

  // 计算头部的结束位置
  ExprPtr head_end = alloc<Min>(
      alloc<Add>(f->start(), immLike(f->stop(), factor)), f->stop(), true);
  // 创建新的头部循环，并将其插入到当前循环的前面
  *head = alloc<For>(f->var(), f->start(), head_end, Stmt::clone(f->body()));
  p->insert_stmt_before(*head, f);

  // 更新原始循环的起始位置为头部的结束位置
  f->set_start(head_end);
  *tail = f;

  // 如果循环是GPU块索引或线程索引，需要对其进行规范化处理
  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*tail);
  }
}

void LoopNest::sliceHead(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head, tail;
  // 调用带有头部和尾部指针参数的sliceHead方法
  sliceHead(f, factor, &head, &tail);
}
void LoopNest::sliceTail(ForPtr f, int factor, ForPtr* head, ForPtr* tail) {
  // 检查循环的起始和结束是否为整数值
  if (intValue(f->start()) && intValue(f->stop())) {
    auto start_val = *intValue(f->start());
    auto stop_val = *intValue(f->stop());
    auto size_val = stop_val - start_val;
    // 如果因子大于等于循环大小，则设置头部为null，尾部为当前循环，然后返回
    if (factor >= size_val) {
      *head = nullptr;
      *tail = f;
      return;
    }
  }

  // 如果循环为空，则抛出异常
  if (!f) {
    throw malformed_input("sliceTail attempted on null loop");
  }

  // 获取循环的父块
  BlockPtr p = to<Block>(f->get_parent());
  // 如果没有父块，则抛出异常
  if (!p) {
    throw malformed_input("sliceTail attempted on loop with no parent");
  }

  // 计算新的循环尾部的起始表达式
  ExprPtr tail_start = alloc<Max>(
      f->start(), alloc<Sub>(f->stop(), immLike(f->stop(), factor)), true);
  // 创建新的尾部循环对象，并复制原始循环的主体
  *tail = alloc<For>(f->var(), tail_start, f->stop(), Stmt::clone(f->body()));
  // 将新的尾部循环插入到原始循环之后
  p->insert_stmt_after(*tail, f);

  // 更新原始循环的结束条件为新的尾部循环的起始条件
  f->set_stop(tail_start);
  // 设置头部循环为原始循环
  *head = f;

  // 如果循环标记为GPU块索引或GPU线程索引，则进行归一化处理
  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*head);
  }
}

void LoopNest::sliceTail(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head, tail;
  // 调用具有头部和尾部参数的sliceTail函数
  sliceTail(f, factor, &head, &tail);
}

void LoopNest::splitWithTail(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner, tail;
  // 调用具有内部和尾部参数的splitWithTail函数
  splitWithTail(f, factor, &inner, &tail);
}

void LoopNest::splitWithTail(
    ForPtr f,
    int factor,
    ForPtr* inner,
    ForPtr* tail) {
  // 如果循环为空，则抛出异常
  if (!f) {
    throw malformed_input("splitWithTail attempted on null loop");
  }

  // 获取循环的父块
  BlockPtr p = to<Block>(f->get_parent());
  // 如果没有父块，则抛出异常
  if (!p) {
    throw malformed_input("splitWithTail attempted on loop with no parent");
  }

  // 对循环进行归一化处理，简化起始和结束边界的计算
  normalize(f);

  // 检查循环的起始和结束是否为整数值
  bool tail_is_needed = true;
  if (intValue(f->start()) && intValue(f->stop())) {
    auto const start_val = *intValue(f->start());
    auto const stop_val = *intValue(f->stop());
    auto const size_val = stop_val - start_val;
    auto const tail_size = size_val % factor;
    // 如果余数为0，则不需要尾部循环
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  // 创建表达式以计算因子
  ExprPtr factor_expr = immLike(f->stop(), factor);
  // 创建表达式以计算循环的大小
  ExprPtr size = alloc<Sub>(f->stop(), f->start());
  // 创建表达式以计算分割计数
  ExprPtr split_count = alloc<Div>(size, factor_expr);
  // 创建表达式以计算尾部大小
  ExprPtr tail_size = alloc<Mod>(size, factor_expr);

  // 获取循环变量的名称和数据类型
  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  // 创建内部循环变量和外部循环变量
  VarPtr i_inner = alloc<Var>(loop_var_name + "_inner", loop_var_dtype);
  VarPtr i_outer = alloc<Var>(loop_var_name + "_outer", loop_var_dtype);

  // 创建组合索引表达式1：x -> x.outer * inner.size + x.inner
  ExprPtr combined_index1 =
      alloc<Add>(alloc<Mul>(i_outer, factor_expr), i_inner);

  // 如果需要尾部循环
  if (tail_is_needed) {
    // 创建尾部循环变量
    VarPtr i_tail = alloc<Var>(loop_var_name + "_tail", loop_var_dtype);
    // 创建组合索引表达式2：x -> x.tail + outer.size * inner.size
    ExprPtr combined_index2 =
        alloc<Add>(i_tail, alloc<Mul>(split_count, factor_expr));

    // 替换循环主体中的变量，将原始循环变量替换为组合索引表达式2
    StmtPtr body_tail =
        SubstituteInClone(f->body(), {{f->var(), combined_index2}});


注：以上是对给定代码每一行进行了详细注释的结果。
    *tail = alloc<For>(i_tail, immLike(tail_size, 0), tail_size, body_tail);

分配一个新的 `For` 循环对象给 `tail` 指针，该循环使用 `i_tail` 作为迭代变量，`tail_size` 作为起始值和终止条件，`body_tail` 作为循环体。


    p->insert_stmt_after(*tail, f);

在 `p` 所指向的位置插入 `*tail` 所指向的 `For` 循环对象，插入位置在 `f` 所指向的循环对象之后。


  } else {
    *tail = nullptr;
  }

如果条件不满足，将 `*tail` 设置为 `nullptr`。


  StmtPtr body_inner =
      Substitute(f->removeBody(), {{f->var(), combined_index1}});

创建一个 `StmtPtr` 类型的变量 `body_inner`，该变量使用 `f` 的主体去除掉变量 `f->var()` 后进行替换为 `combined_index1`。


  *inner =
      alloc<For>(i_inner, immLike(factor_expr, 0), factor_expr, body_inner);

为 `inner` 指针分配一个新的 `For` 循环对象，该循环使用 `i_inner` 作为迭代变量，`factor_expr` 作为起始值和终止条件，`body_inner` 作为循环体。


  // The input loop `f` will be the outer loop after split.
  f->set_var(i_outer);
  f->set_start(immLike(split_count, 0));
  f->set_stop(split_count);
  f->set_body(*inner);

将输入的循环 `f` 分割后，将其设置为外部循环。设置 `f` 的迭代变量为 `i_outer`，起始值为 `0`，终止条件为 `split_count`，循环体设置为 `*inner` 所指向的循环对象。
}

void LoopNest::splitWithMask(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;
  // 调用另一个重载函数，将内部指针传递给内部循环的指针
  splitWithMask(f, factor, &inner);
}

void LoopNest::splitWithMask(ForPtr f, int factor, ForPtr* inner) {
  // 将当前循环的父级转换为块指针
  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    // 如果父级不是块，则输出错误信息并返回
    std::cerr << "Parent is not a Block!\n";
    return;
  }

  bool tail_is_needed = true;
  // 简化起始和结束表达式
  ExprPtr start = IRSimplifier::simplify(f->start());
  ExprPtr stop = IRSimplifier::simplify(f->stop());
  // 检查起始和结束是否为常量
  if (start->isConstant() && stop->isConstant()) {
    auto start_val = *intValue(start);
    auto stop_val = *intValue(stop);
    auto size_val = stop_val - start_val;
    auto tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  // 创建表示因子的表达式
  auto factor_expr = immLike(f->stop(), factor);
  ExprPtr size = alloc<Sub>(f->stop(), f->start());
  // 计算分割次数：(size + factor - 1) / factor
  ExprPtr split_count = alloc<Div>(
      alloc<Sub>(alloc<Add>(size, factor_expr), immLike(size, 1)), factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  // 创建内部和外部循环的新变量
  VarPtr i_inner = alloc<Var>(loop_var_name + "_inner", loop_var_dtype);
  VarPtr i_outer = alloc<Var>(loop_var_name + "_outer", loop_var_dtype);

  // 组合索引表达式：x -> x.outer * inner.size + x.inner
  ExprPtr combined_index =
      alloc<Add>(alloc<Mul>(i_outer, factor_expr), i_inner);

  // 移除当前循环的主体语句
  StmtPtr body_inner = f->removeBody();
  // 如果需要尾部，则根据条件创建谓词和条件语句
  if (tail_is_needed) {
    auto start = intValue(f->start());
    if (!start || *start != 0) {
      throw unimplemented_lowering();
    }

    ExprPtr predicate =
        CompareSelect::make(ExprHandle(f->var()), ExprHandle(f->stop()), kLT)
            .node();
    body_inner = Cond::make(ExprHandle(predicate), body_inner, nullptr);
  }
  // 替换主体中的循环变量
  body_inner = Substitute(body_inner, {{f->var(), combined_index}});

  // 创建内部循环并赋值给传入的内部循环指针
  *inner =
      alloc<For>(i_inner, immLike(factor_expr, 0), factor_expr, body_inner);
  // 设置外部循环的新变量、起始和结束
  f->set_var(i_outer);
  f->set_start(immLike(split_count, 0));
  f->set_stop(split_count);
  f->set_body(*inner);
}

std::vector<ForPtr> LoopNest::distributeLoop(
    ForPtr loop,
    const std::unordered_set<StmtPtr>& pivots) {
  TORCH_INTERNAL_ASSERT(
      loop,
      buildErrorMessage(
          "Expected non-null loop in distributeLoop in the fuser."));
  auto root = loop->get_parent();
  if (root == nullptr) {
    throw malformed_input("Loop without parent: ", loop);
  }
  auto root_block = to<Block>(root);
  if (root_block == nullptr) {
  // 如果循环的父节点不是块(Block)，则抛出异常，指明循环的父节点应该是块
  throw malformed_input(
      "Loop's parent must be a Block, instead found ", root);
}

// 在分发后提取所有循环的主体。
std::vector<BlockPtr> new_loop_bodies;
// 创建一个新的块作为新循环的主体，并初始化为空块
auto new_loop_body = alloc<Block>(std::vector<StmtPtr>({}));
while (!loop->body()->empty()) {
  // 获取当前循环体的第一个语句
  auto s = loop->body()->front();
  // 从当前循环体中移除该语句
  loop->body()->remove_stmt(s);
  // 将语句添加到新的循环主体中
  new_loop_body->append_stmt(s);
  // 如果语句是枢纽点，则表示新循环主体结束，需要开始下一个新的循环主体
  if (pivots.count(s)) {
    new_loop_bodies.push_back(new_loop_body);
    // 创建一个新的空块作为下一个循环的主体
    new_loop_body = alloc<Block>(std::vector<StmtPtr>({}));
  }
}
// 如果最后一个新循环主体不为空，则将其添加到循环主体列表中
if (!new_loop_body->empty()) {
  new_loop_bodies.push_back(new_loop_body);
}

// 将第一个循环主体插入到原始循环中
loop->body()->splice(loop->body()->begin(), new_loop_bodies.front());
std::vector<ForPtr> new_loops = {loop};

// 为所有剩余的块创建循环。
// 将所有新循环添加到父块中。
for (size_t i = 1; i < new_loop_bodies.size(); ++i) {
  // 克隆当前循环，并使用新的循环主体替换
  auto new_loop = loop->cloneWithNewBody(new_loop_bodies[i]);
  // 将新循环插入到根块中，插入位置为上一个新循环之后
  root_block->insert_stmt_after(new_loop, new_loops.back());
  // 将新循环添加到新循环列表中
  new_loops.push_back(new_loop);
}

// 返回所有创建的新循环
return new_loops;
}

// 将给定循环体内的语句转为无序集合，返回分发后的循环列表
std::vector<ForPtr> LoopNest::distributeLoop(ForPtr loop) {
  std::unordered_set<StmtPtr> stmtsInBlock(
      loop->body()->begin(), loop->body()->end());
  return distributeLoop(loop, stmtsInBlock);
}

// 分发给定循环及其所有父循环
std::vector<ForPtr> LoopNest::distributeLoopAndParents(ForPtr loop) {
  auto parentLoop = getParentLoop(loop);  // 获取给定循环的父循环
  auto result = distributeLoop(loop);     // 分发给定循环
  if (parentLoop) {
    return distributeLoopAndParents(parentLoop);  // 递归分发父循环
  }
  return result;
}

// 分发给定循环内的所有内部循环
std::vector<ForPtr> LoopNest::distributeLoopOverInnerLoops(ForPtr loop) {
  auto loops = NodeFinder<For>::find(loop);  // 查找给定循环内的所有循环节点
  std::unordered_set<StmtPtr> loopsSet(loops.begin(), loops.end());  // 转为无序集合
  return distributeLoop(loop, loopsSet);  // 分发给定循环及其内部循环
}

// 分发给定循环及其所有父循环内的所有内部循环
std::vector<ForPtr> LoopNest::distributeLoopAndParentsOverInnerLoops(
    ForPtr loop) {
  auto parentLoop = getParentLoop(loop);  // 获取给定循环的父循环
  auto result = distributeLoopOverInnerLoops(loop);  // 分发给定循环及其内部循环
  if (parentLoop) {
    return distributeLoopAndParentsOverInnerLoops(parentLoop);  // 递归分发父循环的内部循环
  }
  return result;
}

// 比较两个表达式是否相等
static bool areEqual(ExprPtr expr1, ExprPtr expr2) {
  auto diff = IRSimplifier::simplify(alloc<Sub>(expr1, expr2));  // 简化两个表达式的差
  return diff->isConstant() && (immediateAs<int>(diff) == 0);  // 判断差是否为常数0
};

// 判断表达式是否包含任何给定变量
static bool doesExprContainAnyVar(
    ExprPtr expr,
    const std::unordered_set<VarPtr>& vars) {
  for (const auto& v : VarFinder::find(expr)) {  // 遍历表达式中的变量
    if (vars.count(v)) {  // 如果变量在给定集合中
      return true;  // 返回true
    }
  }
  return false;  // 否则返回false
}

// 判断给定索引列表是否在给定外部循环变量的条件下循环无关
static bool areIndicesLoopIndependent(
    const std::vector<ExprPtr>& expr_list1,
    const std::vector<ExprPtr>& expr_list2,
    const std::unordered_set<VarPtr>& outer_loop_vars) {
  if (expr_list1.size() != expr_list2.size()) {  // 如果两个索引列表长度不相等
    return false;  // 返回false
  }
  for (size_t i = 0; i < expr_list1.size(); ++i) {
    const auto& expr1 = expr_list1[i];
    const auto& expr2 = expr_list2[i];
    if (doesExprContainAnyVar(expr1, outer_loop_vars) ||  // 如果表达式包含外部循环变量
        doesExprContainAnyVar(expr2, outer_loop_vars)) {
      if (!areEqual(expr1, expr2)) {  // 如果表达式不相等
        return false;  // 返回false
      }
    }
  }
  return true;  // 否则返回true，表示循环无关
}

// 检查给定循环是否存在循环依赖
bool LoopNest::hasLoopCarriedDependence(ForPtr loop) {
  analysis::MemDependencyChecker analyzer;  // 内存依赖分析器
  loop->accept(&analyzer);  // 分析给定循环的内存依赖

  std::unordered_set<VarPtr> outer_loop_vars = {loop->var()};  // 外部循环变量集合
  auto outer_loops = LoopNest::getEnclosingLoopNest(loop);  // 获取包围给定循环的外部循环
  for (const auto& l : outer_loops) {
  for (auto it1 = loop->body()->begin(); it1 != loop->body()->end(); ++it1) {
    // 遍历循环体中的语句列表

    outer_loop_vars.insert(l->var());
  }

  // 高级算法：检查两个对缓冲区的访问（A和B），其中一个是存储操作，是否导致循环延迟依赖：
  //   1. 对于每对索引表达式Ai和Bi，它们分别指向A和B的维度，
  //      如果满足以下条件之一：
  //       a) Ai和Bi相等（或者）
  //       b) Ai和Bi均不包含任何外部循环变量
  //      则A和B之间的依赖是循环独立的。因为在情况b）中，这些索引表达式不会影响访问A和B的顺序。
  //   2. 如果条件1）不满足：
  //       a) 如果访问的边界重叠，则存在循环延迟依赖。
  //       b) 如果访问的边界不重叠，则没有依赖关系。

  // 注意：由于我们在涉及外部循环变量时检查索引表达式的相等性，这可能会错误地将某些情况报告为具有循环延迟依赖。
  //     在这里处理所有可能情况是不现实的，因此，我们采用保守措施，并允许一些误报。虽然这会阻止一些循环融合的机会，
  //     但这应该是允许的情况中的一小部分。

  // 实现：
  // 对于循环中的每一对语句S1和S2：
  //  * 获取S1和S2中的加载和存储操作。
  //  * 对于S1中的每个存储和S2中的每个加载，如果索引表达式不相等且访问有重叠，则返回true，表示存在循环延迟依赖。
  //  * 对于S1中的每个加载和S2中的每个存储，如果索引表达式不相等且访问有重叠，则返回true，表示存在循环延迟依赖。
  //  * 对于S1中的每个存储和S2中的每个存储，如果索引表达式不相等且访问有重叠，则返回true，表示存在循环延迟依赖。
  for (auto it1 = loop->body()->begin(); it1 != loop->body()->end(); ++it1) {
    // 对每个迭代器 it1 执行以下循环
    for (auto it2 = std::next(it1); it2 != loop->body()->end(); ++it2) {
      // 在 it1 中查找所有的 Store 节点
      auto aStores = NodeFinder<Store>::find(*it1);
      // 在 it1 中查找所有的 Load 节点
      auto aLoads = NodeFinder<Load>::find(*it1);
      // 在 it2 中查找所有的 Store 节点
      auto bStores = NodeFinder<Store>::find(*it2);
      // 在 it2 中查找所有的 Load 节点
      auto bLoads = NodeFinder<Load>::find(*it2);
      
      // ReadAfterWrite 检测
      for (auto& aStore : aStores) {
        for (auto& bLoad : bLoads) {
          // 如果 aStore 和 bLoad 使用相同的缓冲区
          if (aStore->buf() == bLoad->buf()) {
            // 检查它们的索引是否在外层循环变量之外独立
            if (!areIndicesLoopIndependent(
                    aStore->indices(), bLoad->indices(), outer_loop_vars)) {
              // 检查它们是否存在重叠
              if (isOverlapping(analyzer, aStore, bLoad)) {
                // 如果存在重叠，返回 true
                return true;
              }
            }
          }
        }
      }
      
      // WriteAfterRead 检测
      for (auto& bStore : bStores) {
        for (auto& aLoad : aLoads) {
          // 如果 bStore 和 aLoad 使用相同的缓冲区
          if (bStore->buf() == aLoad->buf()) {
            // 检查它们的索引是否在外层循环变量之外独立
            if (!areIndicesLoopIndependent(
                    bStore->indices(), aLoad->indices(), outer_loop_vars)) {
              // 检查它们是否存在重叠
              if (isOverlapping(analyzer, bStore, aLoad)) {
                // 如果存在重叠，返回 true
                return true;
              }
            }
          }
        }
      }
      
      // WriteAfterWrite 检测
      for (auto& aStore : aStores) {
        for (auto& bStore : bStores) {
          // 如果 aStore 和 bStore 使用相同的缓冲区
          if (aStore->buf() == bStore->buf()) {
            // 检查它们的索引是否在外层循环变量之外独立
            if (!areIndicesLoopIndependent(
                    aStore->indices(), bStore->indices(), outer_loop_vars)) {
              // 检查它们是否存在重叠
              if (isOverlapping(analyzer, aStore, bStore)) {
                // 如果存在重叠，返回 true
                return true;
              }
            }
          }
        }
      }
    }
  }
  // 如果没有找到冲突，返回 false
  return false;
  }
}

// 尝试将给定的一组循环融合成一个循环，如果成功则返回 true 并将结果存入 fused 指针指向的位置
bool LoopNest::unsafeFuseLoops(
    const std::vector<ForPtr>& loops,
    ForPtr* fused) {
  // 如果 loops 为空，则无法融合，返回 false
  if (loops.empty()) {
    return false;
  }
  // 如果 loops 中只有一个循环，则直接将其赋值给 fused 并返回 true
  if (loops.size() == 1) {
    *fused = loops.front();
    return true;
  }

  // 检查所有循环是否具有相同的父节点
  auto root = loops.front()->get_parent();
  for (const auto& l : loops) {
    auto par = l->get_parent();
    // 如果某个循环的父节点为空，表示结构错误，返回 false
    if (par == nullptr) {
      return false;
    }
    // 如果循环的父节点不等于第一个循环的父节点，则无法融合，返回 false
    if (par != root) {
      return false;
    }
  }
  auto root_block = to<Block>(root);
  // 如果根节点无法转换为 Block 类型，返回 false
  if (root_block == nullptr) {
    return false;
  }

  // 目前只处理所有循环之间没有语句的情况。可以通过依赖分析放宽此约束。TODO.
  // 在根节点的语句中查找第一个循环，确认是否存在于根节点中
  auto it = root_block->begin();
  for (; it != root_block->end(); ++it) {
    if (*it == loops.front()) {
      break;
    }
  }
  // 如果未找到第一个循环在根节点中，抛出内部断言错误
  TORCH_INTERNAL_ASSERT(
      it != root_block->end(),
      buildErrorMessage(
          "Could not find the given loop in the root stmt in unsafeFuseLoop the fuser."));
  // 确认每个循环都在根节点的正确位置上
  for (const auto& l : loops) {
    if (*it != l) {
      return false;
    }
    ++it;
  }

  const auto& first_loop = loops.front();
  // 通过将第二个循环及其后续语句移动到第一个循环的主体中来融合循环
  // 最终融合的循环将与第一个循环相同
  for (size_t i = 1; i < loops.size(); ++i) {
    auto body = to<Block>(SubstituteInClone(
        loops[i]->body(), {{loops[i]->var(), first_loop->var()}}));
    first_loop->body()->splice(first_loop->body()->end(), body);
    root_block->remove_stmt(loops[i]);
  }

  *fused = loops.front();
  return true;
}

// 尝试将给定的一组循环融合成一个循环，如果成功则返回 true 并将结果存入 fused 指针指向的位置
bool LoopNest::fuseLoops(const std::vector<ForPtr>& loops, ForPtr* fused) {
  // 如果 loops 为空，则无法融合，返回 false
  if (loops.empty()) {
    return false;
  }
  // 如果 loops 中只有一个循环，则直接将其赋值给 fused 并返回 true
  if (loops.size() == 1) {
    *fused = loops.front();
    return true;
  }

  // 检查所有循环的起始和结束边界是否相同
  const auto& first_loop = loops.front();
  auto first_loop_start = IRSimplifier::simplify(first_loop->start());
  auto first_loop_stop = IRSimplifier::simplify(first_loop->stop());
  for (size_t i = 1; i < loops.size(); ++i) {
    const auto& curr_loop = loops[i];
    auto curr_loop_start = IRSimplifier::simplify(curr_loop->start());
    auto curr_loop_stop = IRSimplifier::simplify(curr_loop->stop());
    // 如果起始边界不相等，无法融合，返回 false
    if (!areEqual(curr_loop_start, first_loop_start)) {
      return false;
    }
    // 如果结束边界不相等，无法融合，返回 false
    if (!areEqual(curr_loop_stop, first_loop_stop)) {
      return false;
    }
  }
  // 循环嵌套结束
  }
}

// 检查是否融合这些循环会导致循环依赖。
// 只有在循环融合后才能进行此检查。但如果检查不通过，我们需要以原始形式返回给定的循环。
// 因此，我们先克隆所有循环，然后再进行融合并进行检查。
std::vector<ForPtr> loops_copy;
loops_copy.reserve(loops.size());
// 创建一个空的 Block 作为克隆循环的父节点
BlockPtr parent = alloc<Block>(std::vector<StmtPtr>({}));
for (auto& l : loops) {
  // 克隆每个循环，并将其添加到父节点中
  auto l_copy = Stmt::clone(l);
  loops_copy.push_back(to<For>(l_copy));
  parent->append_stmt(l_copy);
}
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
// 创建一个用于融合后的循环的指针
ForPtr fused_copy;
// 尝试融合克隆后的循环，并返回是否成功
bool ret = unsafeFuseLoops(loops_copy, &fused_copy);
// 如果融合不成功或者存在循环依赖，则返回 false
if (!ret || hasLoopCarriedDependence(fused_copy)) {
  return false;
}

// 现在所有条件都满足，进行循环融合
return unsafeFuseLoops(loops, fused);
}

// 在循环嵌套中查找外层循环
ForPtr LoopNest::findOuterFor(ForPtr a, ForPtr b) {
  // 将当前循环b作为猜测的后续循环
  StmtPtr s = b;
  while (s != nullptr) {
    // 如果找到循环a，则b在a之后
    if (s == a) {
      return a;  // 返回a作为外层循环
    }
    s = s->get_parent();  // 获取当前语句的父级语句
  }

  // 检查两个循环是否在同一个循环嵌套中
  s = a;
  while (s != nullptr) {
    // 如果找到循环b，则a在b之后
    if (s == b) {
      return b;  // 返回b作为外层循环
    }
    s = s->get_parent();  // 获取当前语句的父级语句
  }

  // a和b没有关系
  return nullptr;  // 返回空指针
}

// 重新排序循环轴
void LoopNest::reorderAxis(ForPtr a, ForPtr b) {
  if (a == b) {
    // 如果a和b相同，则无需操作
    return;
  }

  // 查找内部和外部循环
  ForPtr outer = findOuterFor(a, b);
  if (outer == nullptr) {
    throw std::runtime_error("Reordered a loop not in LoopNest");
  }

  ForPtr inner = a == outer ? b : a;  // 确定内部循环
  std::deque<ForPtr> internal_axes;  // 存储相关的轴，以便反向排序

  // 查找相关轴，存储为反向顺序
  StmtPtr s = inner;
  while (s != outer) {
    if (ForPtr f = to<For>(s)) {
      internal_axes.push_back(f);  // 将内部循环添加到队列中
    }

    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    s = s->get_parent();  // 获取当前语句的父级语句
  }

  internal_axes.push_back(outer);  // 将外部循环添加到队列中

  BlockPtr root = to<Block>(outer->get_parent());
  CHECK(root);  // 确保root非空

  // 浅拷贝内部块
  BlockPtr body = alloc<Block>(std::vector<StmtPtr>({}));  // 创建空块
  body->splice(body->end(), inner->body());  // 将内部块添加到body中

  const ForPtr& before{outer};  // 外部循环之前
  ForPtr after{nullptr};  // 外部循环之后
  ForPtr last = internal_axes.front();  // 最后一个内部轴
  StmtPtr newInner = body;  // 新的内部块为body

  s = inner;
  while (s != outer) {
    if (auto cond = to<Cond>(s->get_parent())) {
      if (s == cond->true_stmt()) {
        newInner = cond->cloneWithNewBody(newInner);  // 使用新的内部块克隆true分支的条件语句
      } else {
        // s是Cond的false分支
        newInner = cond->cloneWithNewBodies(
            alloc<Block>(std::vector<StmtPtr>({})), newInner);  // 使用新的内部块克隆false分支的条件语句
      }
    }
    s = s->get_parent();  // 获取当前语句的父级语句
  }

  // 这是循环重新排序中的主要复杂性：处理不在重排序直线上的语句。
  // 为了处理这一点，我们将树分成关键路径之前和关键路径之后的部分。
  //
  // 这种模式的一个示例是：
  // for i in ..
  //   Statement A
  //   for j in ..
  //     Statement B
  //   Statement C
  //
  // 当重新排序循环i和j时，我们需要确保Statement A和C仍然在i的循环范围内执行，并且这三个语句不被重新排序（尽可能）。
  for (const auto& loop : internal_axes) {
    // 如果内部循环在循环后有组件，则必须将其包装在与树的此级别匹配的For循环中。
    if (after != nullptr) {
      after = loop->cloneWithNewBody(after);  // 使用新的内部块克隆循环后的组件
    }

    bool pastMidpoint = false;  // 是否超过中点
    bool hadBeforeStmts = false;  // 是否有前置语句
    // 迭代循环体中的语句，注意不要使迭代器失效
    for (auto I = loop->body()->begin(), E = loop->body()->end(); I != E;) {
      // 保存当前迭代器位置的语句，并将迭代器指向下一个位置
      StmtPtr s = *(I++);
      // 如果当前语句是标记的最后一个语句
      if (s == last) {
        // 移除标记的最后一个语句
        loop->body()->remove_stmt(s);
        // 如果之前没有语句存在，则整个循环不需要保留，可以与上层循环合并
        if (!hadBeforeStmts) {
          last = loop;
        }
        // 标记已经过了中点
        pastMidpoint = true;
      } else if (pastMidpoint) {
        // 中点之后的语句需要移到重新排序路径的后面，以保持顺序
        loop->body()->remove_stmt(s);
        // 如果 after 为 nullptr，则使用当前语句创建新的循环体
        if (after == nullptr) {
          after = loop->cloneWithNewBody(s);
        } else {
          // 否则将当前语句追加到已有的 after 循环体中
          after->body()->append_stmt(s);
        }
      } else {
        // 在重新排序的循环之前的语句保持不动，只需保留循环结构即可
        hadBeforeStmts = true;
      }
    }
  }

  // 现在可以实际重新排序选择的轴线了。
  // 交换内部轴线的首尾元素
  std::swap(internal_axes.front(), internal_axes.back());

  // 创建重新排序后的内部轴线：
  for (const auto& loop : internal_axes) {
    // 使用新的内部循环体克隆每个轴线
    newInner = loop->cloneWithNewBody(newInner);
  }

  // 将新语句追加到树的根节点。
  if (before->body()->nstmts() == 0) {
    // 如果顶层现在为空，删除它。
    root->replace_stmt(before, newInner);
  } else {
    // 否则将 newInner 插入到 before 之后
    root->insert_stmt_after(newInner, before);
  }

  // 如果有 after 循环体，则将其插入到 newInner 之后
  if (after) {
    root->insert_stmt_after(after, newInner);
  }
}

static bool isTrivialPermutation(const std::vector<size_t>& permutation) {
  // 检查排列是否为平凡排列（即标识索引位置不变的排列）
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

static bool isValidPermutation(std::vector<size_t> permutation) {
  // 对排列进行排序，然后检查是否为平凡排列
  std::sort(permutation.begin(), permutation.end());
  return isTrivialPermutation(permutation);
}

std::vector<ForPtr> LoopNest::reorder(
    const std::vector<ForPtr>& loops,
    const std::vector<size_t>& permutation) {
  // 检查输入的循环向量和排列向量是否长度一致
  if (loops.size() != permutation.size()) {
    throw malformed_input("invalid permutation size");
  }
  // 如果排列是平凡排列，则直接返回循环向量
  if (isTrivialPermutation(permutation)) {
    return loops;
  }
  // 如果排列不是有效的排列，则抛出异常
  if (!isValidPermutation(permutation)) {
    throw malformed_input("invalid permutation for reorder");
  }
  // 如果循环向量长度小于2，则直接返回循环向量
  if (loops.size() < 2) {
    return loops;
  }
  // 如果循环结构不是完全嵌套的，则抛出异常
  if (!areLoopsPerfectlyNested(loops)) {
    throw malformed_input("reorder is only allowed on perfectly nested loops");
  }

  auto parent = to<Block>(loops.front()->get_parent());
  // 父块不能为空，否则抛出异常
  if (parent == nullptr) {
    throw malformed_input("parent of the loops must be a Block");
  }

  // 根据排列重新排序循环向量
  std::vector<ForPtr> result(loops.size());
  for (size_t i = 0; i < loops.size(); ++i) {
    result[i] = loops[permutation[i]];
  }

  // 从所有循环中移除主体
  auto innermost_body = loops.back()->removeBody();
  // 使用空块语句替换最外层的循环，以确定重新排序后的最外层循环应插入的位置
  auto empty_block = alloc<Block>(std::vector<StmtPtr>({}));
  parent->replace_stmt(loops.front(), empty_block);
  for (size_t i = 1; i < loops.size(); ++i) {
    auto block = to<Block>(loops[i]->get_parent());
    TORCH_INTERNAL_ASSERT(
        block,
        buildErrorMessage(
            "Expected parent stmt to be a non-null Block in reorder transformation the fuser."));
    block->remove_stmt(loops[i]);
  }

  // 设置重新排序后所有循环的新主体
  for (size_t i = 0; i < result.size() - 1; ++i) {
    result[i]->set_body(result[i + 1]);
  }
  result.back()->set_body(innermost_body);
  parent->replace_stmt(empty_block, result.front());
  return result;
}

ForPtr LoopNest::getLoopAt(ForPtr root, const std::vector<int>& indices) const {
  // 返回在给定根循环中指定索引路径上的循环
  if (indices.empty()) {
    return root;
  }
  // 如果根循环为空，则抛出异常
  if (root == nullptr) {
    throw malformed_input("root loop is null");
  }

  ForPtr curr = root;
  for (auto i : indices) {
    // 如果索引超出范围或当前循环体中没有足够的语句，则返回空指针
    if (i < 0 || curr->body()->nstmts() <= i) {
      return nullptr;
    }
    // 获取当前索引位置的语句，并将其转换为循环
    std::list<StmtPtr>::iterator stmtp = curr->body()->begin();
    std::advance(stmtp, i);
    curr = to<For>(*stmtp);
    // 如果转换失败，则返回空指针
    if (curr == nullptr) {
      return nullptr;
    }
  }

  return curr;
}

ForPtr LoopNest::tile(ForPtr x, ForPtr y, int x_factor, int y_factor) {
  auto parent = to<Block>(x->get_parent());
  // 父块不能为空，否则抛出异常
  if (parent == nullptr) {
  // 如果循环的父级不是块，则抛出异常
  throw malformed_input("parent of the loops must be a Block");
}
// 如果循环不是完美嵌套的，则抛出异常
if (!areLoopsPerfectlyNested({x, y})) {
  throw malformed_input("two loops must be perfectly nested");
}

// 使用 x_factor 和 y_factor 分割 x 和 y 轴
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
ForPtr yi, ytail;
splitWithTail(y, y_factor, &yi, &ytail);
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
ForPtr xi, xtail;
splitWithTail(x, x_factor, &xi, &xtail);

// 将 xi 在 yo 和 ytail 上分布，以便可以操作 {xo, xi, yo, yi} 的循环顺序
auto loops = distributeLoop(xi);

// 对于 {xi, yo, yi}，重新排列轴顺序为 yo, xi, yi
xi = loops.front();
ForPtr yo = to<For>(xi->body()->stmts().front());
CHECK(yo);
reorder({xi, yo}, {1, 0});

// 对于 {xi, ytail}，重新排列轴顺序为 ytail, xi
if (loops.size() == 2) {
  xi = loops.back();
  ytail = to<For>(xi->body()->stmts().front());
  CHECK(ytail);
  reorder({xi, ytail}, {1, 0});
}

// 返回 xtail
return xtail;
}

// 检查给定的循环向量是否是完美嵌套的
bool LoopNest::areLoopsPerfectlyNested(const std::vector<ForPtr>& loops) {
  // 如果循环数量小于2，认为是完美嵌套的
  if (loops.size() < 2) {
    return true;
  }
  // 检查相邻两个循环是否完美嵌套
  for (size_t i = 0; i < loops.size() - 1; ++i) {
    auto loop_body = loops[i]->body();
    // 如果前一个循环的主体语句数不为1，或者其第一个语句不是下一个循环，则不是完美嵌套
    if (loop_body->nstmts() != 1 || loop_body->front() != loops[i + 1]) {
      return false;
    }
  }
  // 所有循环都完美嵌套
  return true;
}

// 对给定的循环进行完全展开，并返回展开后的语句块
void LoopNest::fullUnroll(ForPtr f, StmtPtr* unrolled) {
  BlockPtr p = to<Block>(f->get_parent());
  // 如果循环指针为空，抛出异常
  if (!f) {
    throw malformed_input("unroll attempted on null loop");
  } else if (!p) {
    throw malformed_input("unroll attempted on loop with no parent");
  }

  // 简化循环起始和终止表达式
  auto start_expr = IRSimplifier::simplify(f->start());
  auto stop_expr = IRSimplifier::simplify(f->stop());
  // 如果循环起始表达式不是常数，抛出异常
  if (!start_expr->isConstant()) {
    throw std::runtime_error("Can't unroll due to non-constant loop start!");
  }
  // 如果循环终止表达式不是常数，抛出异常
  if (!stop_expr->isConstant()) {
    throw std::runtime_error("Can't unroll due to non-constant loop stop!");
  }

  std::vector<StmtPtr> unrolled_stmts;
  int start_val = immediateAs<int>(start_expr);
  int stop_val = immediateAs<int>(stop_expr);
  // 对循环进行展开
  for (int current = start_val; current < stop_val; ++current) {
    for (const auto& stmt : f->body()->stmts()) {
      // 替换循环变量，并将结果语句添加到展开后的语句块中
      unrolled_stmts.push_back(SubstituteInClone(
          stmt, {{f->var(), getImmediateByType(f->var()->dtype(), current)}}));
    }
  }
  // 创建展开后的语句块，并简化
  *unrolled = alloc<Block>(unrolled_stmts);
  *unrolled = IRSimplifier::simplify(*unrolled);

  // 将原始循环替换为展开后的语句块
  p->replace_stmt(f, *unrolled);
}

// 对给定的循环进行完全展开，不返回展开后的语句块
void LoopNest::fullUnroll(ForPtr f) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  StmtPtr unrolled;
  fullUnroll(f, &unrolled);
}

// 将给定循环按指定因子展开，并返回展开后的尾部循环
void LoopNest::unroll(ForPtr f, int factor, ForPtr* tail) {
  // 如果展开因子小于2，直接返回
  if (factor < 2) {
    return;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;
  // 将循环分割为主循环和尾部循环，并进行完全展开
  splitWithTail(f, factor, &inner, tail);
  fullUnroll(inner);
}

// 将给定循环按指定因子展开，不返回展开后的尾部循环
void LoopNest::unroll(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  // 对循环按指定因子进行展开
  unroll(f, factor, &tail);
}

// 检查给定循环是否已经标准化（起始值为0）
bool LoopNest::isNormalized(ForPtr f) {
  // 如果循环起始表达式是常数0，则已标准化
  if (f->start()->isConstant()) {
    return immediateAs<int>(f->start()) == 0;
  }
  return false;
}

// 将给定循环标准化（将起始值设为0）
bool LoopNest::normalize(ForPtr f) {
  // 如果循环指针为空，抛出异常
  if (!f) {
    throw malformed_input("normalize attempted on null loop");
  }

  // 如果循环已经标准化，无需再次操作
  if (isNormalized(f)) {
    // No need to normalize anymore here.
    return false;
  }

  // 替换循环体内的变量，并更新起始和终止表达式为标准化后的值
  auto for_body_normalized = Substitute(
      f->body(),
      {{f->var(), (VarHandle(f->var()) + ExprHandle(f->start())).node()}});
  f->set_body(IRSimplifier::simplify(for_body_normalized));
  f->set_stop(IRSimplifier::simplify(alloc<Sub>(f->stop(), f->start())));
  f->set_start(immLike(f->stop(), 0));
  return true;
}

// 期望在给定循环中（包括子循环）找到'num'个完美嵌套的循环
std::vector<ForPtr> LoopNest::getLoopStmtsInLoopNest(ForPtr f, size_t num) {
  std::vector<ForPtr> loops(num);
  ForPtr curr_for = f;
  loops[0] = curr_for;
  // 将当前循环及其内嵌循环添加到结果向量中
  for (size_t i = 1; i < num; ++i) {
    # 使用 TORCH_INTERNAL_ASSERT 宏断言当前 for 循环体中语句数量为 1
    TORCH_INTERNAL_ASSERT(
        curr_for->body()->nstmts() == 1,
        buildErrorMessage("Expected a single stmt in the loop body."));
    
    # 将当前 for 循环体的唯一语句转换为 For 类型对象，并赋值给 curr_for
    curr_for = to<For>(curr_for->body()->front());
    
    # 使用 TORCH_INTERNAL_ASSERT 宏断言 curr_for 不为空，确保唯一子语句是 For 循环
    TORCH_INTERNAL_ASSERT(
        curr_for,
        buildErrorMessage("Expected the only child stmt to be a For loop."));
    
    # 将当前处理的 for 循环存储在 loops 数组的第 i 个位置
    loops[i] = curr_for;
  }
  
  # 返回存储了所有 for 循环的 loops 数组
  return loops;
}

bool LoopNest::flatten(const std::vector<ForPtr>& loops, ForPtr* flattened) {
  // 如果循环列表为空，则抛出异常
  if (loops.empty()) {
    throw malformed_input("flatten attempted on empty set of loops");
  }
  // 获取第一个循环的父级块对象
  BlockPtr p = to<Block>(loops[0]->get_parent());
  // 如果获取不到父级块对象，则抛出异常
  if (!p) {
    throw malformed_input("flatten attempted on loops with no parent");
  }

  // 如果循环列表中只有一个循环，则该嵌套已经是扁平化的
  *flattened = loops[0];
  return false;
}

// 检查所有循环是否对应于完美的循环嵌套：
// * 每个循环除了最内层应该只有一个语句，即 For 循环。
// 如果不是完美的循环嵌套，则不进行扁平化。
// 此检查还确保我们不会扁平化约简循环。
for (size_t i = 0; i < loops.size() - 1; ++i) {
  if ((loops[i]->body()->nstmts() != 1) ||
      (loops[i]->body()->front() != loops[i + 1])) {
    return false;
  }
}

// 在扁平化之前对循环进行规范化。
// 我们需要从最内层到最外层对它们进行规范化，因为一旦外层循环被规范化，
// 给定的指向内部循环的指针将指向旧代码。
// 出于同样的原因，我们不能在最外层循环规范化之前存储规范化的内部循环。
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
for (size_t i = 0; i < loops.size(); ++i) {
  size_t idx = loops.size() - i - 1;
  LoopNest::normalize(loops[idx]);
}

// 'normalized' 指向规范化后循环嵌套中的最外层循环。
// 收集所有规范化后的循环。
// NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
auto normalized_loops = getLoopStmtsInLoopNest(loops.front(), loops.size());

// 创建一个扁平化后变量的名称，例如添加 "_flat" 后缀
auto flat_var = alloc<Var>(
    normalized_loops[0]->var()->name_hint() + "_flat",
    normalized_loops[0]->var()->dtype());
VarMapping var_mapping;
ExprPtr stop = immLike(flat_var, 1);

// 为每个规范化后的循环创建变量映射
for (size_t i = 0; i < normalized_loops.size(); ++i) {
  size_t idx = normalized_loops.size() - i - 1;
  auto curr_loop = normalized_loops[idx];
  ExprPtr div = alloc<Div>(flat_var, stop);
  ExprPtr sub_expr = idx == 0 ? div : alloc<Mod>(div, curr_loop->stop());
  var_mapping.emplace_back(curr_loop->var(), sub_expr);
  stop = alloc<Mul>(curr_loop->stop(), stop);
}

// 替换最外层循环体中的变量
auto flattened_body =
    Substitute(normalized_loops.back()->removeBody(), var_mapping);

normalized_loops.front()->set_var(flat_var);
normalized_loops.front()->set_start(immLike(stop, 0));
normalized_loops.front()->set_stop(stop);
normalized_loops.front()->set_body(flattened_body);
*flattened = normalized_loops.front();
return true;
}

bool LoopNest::flatten(const std::vector<ForPtr>& loops) {
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
ForPtr flattened;
return flatten(loops, &flattened);
}
  // Loop iterations in NNC IR do not follow sequential semantics by default.
  // In other words, the iterations of the loops could be executed in any
  // random order without affecting correctness. This constraint in turn
  // implies that there can’t be any *inter-iteration* dependences
  // (or *loop-carried* dependences) in NNC loops. So, any NNC IR with such
  // dependences is considered invalid.
  //
  // Given the constraint above, for any pair of accesses to a buffer (where
  // at least one of the access is a write), the accesses must be
  // loop-independent on the innermost loop containing the accesses as well as
  // all the loops above it. So, any dimension that uses only those loop
  // variables to access the given buffer could be optimized away.
  //
  // Algorithm:
  //   * Find all the accesses to the given buf. (A)
  //   * Find the parent common to all accesses in A. (P)
  //   * Collect all the loops above P. (L)
  //   * Collect all the loop variables corresponding to L. (LV)
  //   * For every access a in A:
  //      * For the index I in every dimension of a:
  //          * If the variables in I are all in LV, mark this dimension
  //            for deletion.
  //   * For every dimension that is marked for deletion in ALL accesses in A:
  //      * Update the buffer to set the size of that dimension to 1.
  //      * Update all accesses in A to set the index in that dimension to 0.

  // Find all writes to the buffer in the statement 'stmt'.
  auto writes = WritesToBuf::find(stmt, buf);
  // Find all reads from the buffer in the statement 'stmt'.
  auto reads = StmtsReadingBuf::find(stmt, buf);

  // Find the parent common to all the buffer accesses.
  BlockPtr parent = to<Block>(writes.front()->get_parent());
  TORCH_INTERNAL_ASSERT(
      parent,
      buildErrorMessage(
          "Expected parent stmt to be a non-null block in compressBuffer in the fuser."));
  for (const auto& w : writes) {
    parent = Block::getSharedParent(parent, w);
  }
  for (const auto& r : reads) {
    parent = Block::getSharedParent(parent, r);
  }

  // Collect all the loops that are above the common parent.
  auto loops = LoopNest::getEnclosingLoopNest(parent);
  std::unordered_set<VarPtr> loop_vars;
  for (const auto& l : loops) {
    loop_vars.insert(l->var());
  }

  // TODO: Need to handle other Stmts / Exprs that read / write buffers.
  auto stores = NodeFinder<Store>::find(stmt);
  auto loads = NodeFinder<Load>::find(stmt);

  // Vector to indicate which dimensions could be compressed away.
  std::vector<bool> dims(buf->dims().size(), true);
  // Lambda function to check if indices are loop independent.
  auto check_indices = [&](const std::vector<ExprPtr>& indices) {
    TORCH_INTERNAL_ASSERT(
        indices.size() == dims.size(),
        buildErrorMessage(
            "Expected ranks to match in compressBuffer in the fuser."));
    // 遍历所有的维度标记（dims），标记是否可以压缩的维度
    for (size_t i = 0; i < indices.size(); ++i) {
      // 查找当前索引（indices[i]）中的变量
      auto index_vars = NodeFinder<Var>::find(indices[i]);
      // 遍历当前索引中的每个变量（iv）
      for (const auto& iv : index_vars) {
        // 如果该变量不在循环变量集合（loop_vars）中
        if (loop_vars.count(iv) == 0) {
          // 将对应的维度标记设为false，表示该维度不能被优化掉
          dims[i] = false;
          // 跳出当前内层循环
          break;
        }
      }
    }
  };
  // 遍历所有的存储操作（stores），检查是否针对当前缓冲区（buf）
  for (const auto& s : stores) {
    if (s->buf() == buf) {
      // 检查存储操作中的索引，并进行维度检查
      check_indices(s->indices());
    }
  }
  // 遍历所有的加载操作（loads），检查是否针对当前缓冲区（buf）
  for (const auto& l : loads) {
    if (l->buf() == buf) {
      // 检查加载操作中的索引，并进行维度检查
      check_indices(l->indices());
    }
  }
  // 初始化标记，表示是否有维度需要压缩
  bool any_dim_to_compress = false;
  // 检查是否存在需要压缩的维度
  for (auto d : dims) {
    any_dim_to_compress |= d;
  }
  // 如果没有需要压缩的维度，则直接返回
  if (!any_dim_to_compress) {
    return;
  }

  // 根据标记压缩缓冲区中的维度
  std::vector<ExprPtr> new_dims(buf->dims());
  for (size_t i = 0; i < dims.size(); ++i) {
    // 如果当前维度标记为true，则将其设为1
    if (dims[i]) {
      new_dims[i] = immLike(buf->dims()[i], 1);
    }
  }
  // 更新缓冲区的维度
  buf->set_dims(new_dims);

  // 修改所有访问以反映已删除的维度
  // 定义函数以获取更新后的索引
  auto get_new_indices = [&](const std::vector<ExprPtr>& indices) {
    // 确保维度数量匹配
    TORCH_INTERNAL_ASSERT(
        indices.size() == dims.size(),
        buildErrorMessage(
            "Expected ranks to match in compressBuffer in the fuser."));
    std::vector<ExprPtr> new_indices(indices);
    // 遍历所有维度，如果标记为true，则将对应索引设为0
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i]) {
        new_indices[i] = immLike(indices[i], 0);
      }
    }
    return new_indices;
  };
  // 更新所有存储操作的索引
  for (const auto& s : stores) {
    if (s->buf() == buf) {
      s->set_indices(get_new_indices(s->indices()));
    }
  }
  // 更新所有加载操作的索引
  for (const auto& l : loads) {
    if (l->buf() == buf) {
      l->set_indices(get_new_indices(l->indices()));
    }
  }
}

// 循环嵌套类的方法：压缩所有缓冲区
void LoopNest::compressAllBuffers(StmtPtr stmt) {
  // 使用 BufFinder 查找给定语句中的所有缓冲区
  for (const auto& buf : BufFinder::find(stmt)) {
    // 压缩每个找到的缓冲区
    compressBuffer(buf, stmt);
  }
}

// 获取给定张量的所有循环语句
std::vector<ForPtr> LoopNest::getLoopStmtsFor(Tensor t) const {
  // 获取张量对应的循环体语句
  StmtPtr cur_stmt = getLoopBodyFor(t);
  return getLoopStmtsFor(cur_stmt);
}

// 获取给定缓冲区的所有循环语句
std::vector<ForPtr> LoopNest::getLoopStmtsFor(BufPtr buf) const {
  // 获取缓冲区对应的循环体语句
  StmtPtr cur_stmt = getLoopBodyFor(buf);
  return getLoopStmtsFor(cur_stmt);
}

// 获取给定语句的所有循环语句
std::vector<ForPtr> LoopNest::getLoopStmtsFor(StmtPtr s) const {
  std::vector<ForPtr> result;

  // 遍历给定语句及其父语句，将所有的 For 循环语句收集到 result 中
  while (s) {
    if (auto loop = to<For>(s)) {
      result.push_back(loop);
    }
    s = s->get_parent();
  }
  // 将结果反转，使得最外层循环排在前面
  std::reverse(result.begin(), result.end());
  return result;
}

// 获取给定张量的循环体语句
StmtPtr LoopNest::getLoopBodyFor(Tensor t) const {
  return getLoopBodyFor(t.buf());
}

// 获取给定缓冲区的循环体语句
StmtPtr LoopNest::getLoopBodyFor(BufPtr buf) const {
  auto writes = WritesToBuf::find(root_stmt_, buf);

  // 对于写入操作有两个时的特殊情况处理
  if (writes.size() == 2) {
    if (StorePtr s = to<Store>(writes.back())) {
      if (ReduceOpPtr r = to<ReduceOp>(s->value())) {
        return (StmtPtr)s; // 返回最后一个写入操作的语句，如果是 ReduceOp 类型的话
      }
    }
  }

  StmtPtr res = nullptr;
  for (const auto& s : writes) {
    if (!res) {
      res = s;
      continue;
    }

    // 获取多个写入操作语句的共同父语句
    res = Block::getSharedParent(res, s);
  }

  return (StmtPtr)res; // 返回共同父语句
}

// 获取给定语句的父循环
ForPtr LoopNest::getParentLoop(StmtPtr st) {
  if (st == nullptr) {
    return nullptr;
  }
  auto par = st->get_parent();
  if (auto f = to<For>(par)) {
    return f;
  }
  return getParentLoop(par);
}

// 获取包含给定语句的所有循环嵌套
std::vector<ForPtr> LoopNest::getEnclosingLoopNest(StmtPtr st) {
  std::vector<ForPtr> loops;
  auto f = getParentLoop(st);
  while (f) {
    loops.push_back(f);
    f = getParentLoop(f);
  }
  // 反转结果，使得最外层的循环排在前面
  std::reverse(loops.begin(), loops.end());
  return loops;
}

// 获取所有写入给定缓冲区的语句
std::vector<StmtPtr> LoopNest::getAllWritesToBuf(BufPtr buf) const {
  return WritesToBuf::find(root_stmt_, buf);
}

// 获取写入给定缓冲区的所有最内层循环
std::vector<ForPtr> LoopNest::getAllInnermostLoopsWritingToBuf(
    BufPtr buf) const {
  auto writes = getAllWritesToBuf(buf);
  std::vector<ForPtr> innermost_loops;
  innermost_loops.reserve(writes.size());
  for (const auto& w : writes) {
    // 获取每个写入操作所在的最内层循环
    innermost_loops.push_back(LoopNest::getParentLoop(w));
  }
  return innermost_loops;
}

// 获取写入给定缓冲区的所有循环嵌套
std::vector<std::vector<ForPtr>> LoopNest::getAllLoopNestsWritingToBuf(
    BufPtr buf) const {
  auto writes = getAllWritesToBuf(buf);
  std::vector<std::vector<ForPtr>> loopnests;
  loopnests.reserve(writes.size());
  for (const auto& w : writes) {
    // 获取每个写入操作所在的所有循环嵌套
    loopnests.emplace_back(LoopNest::getEnclosingLoopNest(w));
  }
  return loopnests;
}

// 简化当前循环嵌套结构
StmtPtr LoopNest::simplify() {
  // 使用 IRSimplifier 对当前的根语句进行简化
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  return root_stmt_;
}

// 将语句中的索引展开为一维形式
StmtPtr FlattenIndexes(StmtPtr s) {
  IndexFlattener idx_flattener;
  return idx_flattener.flatten(s);
}

// 用于在 `compute_at` 方法中进行重写的辅助类说明
// 更多细节请参见 LoopNest::computeAt 方法。
class LoopComputeAtRewriter : public IRMutator {
 public:
  LoopComputeAtRewriter(
      BufPtr buf,
      BufPtr new_buf,
      std::vector<ExprPtr> offsets)
      : buf_(std::move(buf)),
        new_buf_(std::move(new_buf)),
        offsets_(std::move(offsets)) {}

 private:
  BufPtr buf_;                       // 原始缓冲区指针
  BufPtr new_buf_;                   // 新缓冲区指针
  std::vector<ExprPtr> offsets_;     // 偏移表达式的向量

  ExprPtr mutate(LoadPtr v) override {
    if (v->buf() != buf_) {          // 如果加载的缓冲区不是原始缓冲区
      return v;                      // 返回原始加载语句
    }
    std::vector<ExprPtr> new_indices(v->indices().size());  // 创建新索引表达式的向量
    for (const auto i : c10::irange(v->indices().size())) {
      new_indices[i] =
          IRSimplifier::simplify(alloc<Sub>(v->indices()[i], offsets_[i]));  // 简化索引偏移计算
    }
    return alloc<Load>(v->dtype(), new_buf_, new_indices);  // 返回修改后的加载语句
  }
};

static StorePtr getStoreStmtOfProducer(StmtPtr s) {
  if (StorePtr st = to<Store>(s)) {   // 如果语句是存储语句，则返回存储指针
    return st;
  }
  if (BlockPtr b = to<Block>(s)) {    // 如果语句是块
    for (const StmtPtr& ss : *b) {    // 遍历块中的每个语句
      if (StorePtr st = to<Store>(ss)) {  // 如果找到存储语句，则返回存储指针
        return st;
      }
    }
  }
  return nullptr;                     // 如果未找到存储语句，则返回空指针
}

static std::vector<VarPtr> getOuterLoopIndexes(StmtPtr s) {
  std::vector<VarPtr> res;            // 结果变量指针向量
  StmtPtr cur = s;                    // 当前语句指针为输入语句指针
  while (cur) {
    if (auto l = to<For>(cur)) {      // 如果当前语句是for循环
      res.push_back(l->var());        // 将循环变量指针添加到结果向量中
    }
    cur = cur->get_parent();          // 获取当前语句的父语句
  }
  return res;                         // 返回外部循环索引变量指针向量
}

class CacheReplacer : public IRMutator {
 public:
  CacheReplacer(BufPtr buffer, BufPtr cache, std::vector<ExprPtr>& offsets)
      : buf_(std::move(buffer)), cache_(std::move(cache)), offsets_(offsets) {}

 private:
  BufPtr buf_;                       // 原始缓冲区指针
  BufPtr cache_;                     // 缓存缓冲区指针
  std::vector<ExprPtr>& offsets_;    // 偏移表达式的向量引用

  ExprPtr mutate(LoadPtr v) override {
    BufPtr buf = v->buf();           // 获取加载语句的缓冲区指针
    if (buf != buf_) {               // 如果加载的缓冲区不是原始缓冲区
      return IRMutator::mutate(v);   // 返回加载语句的变异版本
    }

    // Map indices to call-parameters.
    std::vector<ExprPtr> newIndices;  // 新索引表达式向量
    TORCH_INTERNAL_ASSERT(            // 内部断言，确保
        offsets_.size() == v->indices().size(),  // 偏移和索引长度相等
        buildErrorMessage(
            "Expected ranks to match in CacheReplacer in the fuser."));  // 构建错误消息
    for (size_t i = 0; i < v->indices().size(); ++i) {
      ExprPtr index = v->indices()[i]->accept_mutator(this);  // 变异索引表达式
      ExprPtr offset = offsets_[i];   // 获取偏移表达式
      ExprPtr sub = IRSimplifier::simplify(alloc<Sub>(index, offset));  // 简化索引偏移计算
      newIndices.push_back(sub);      // 添加到新索引表达式向量
    }
    v->set_buf(cache_);               // 设置加载语句的缓冲区为缓存缓冲区
    v->set_indices(newIndices);       // 设置加载语句的索引为新索引表达式
    return v;                         // 返回修改后的加载语句
  }

  StmtPtr mutate(StorePtr v) override {
    BufPtr buf = v->buf();           // 获取存储语句的缓冲区指针
    if (buf != buf_) {               // 如果存储的缓冲区不是原始缓冲区
      return IRMutator::mutate(v);   // 返回存储语句的变异版本
    }

    ExprPtr newValue = v->value()->accept_mutator(this);  // 变异存储值表达式

    // Map indices to call-parameters.
    std::vector<ExprPtr> newIndices;  // 新索引表达式向量
    TORCH_INTERNAL_ASSERT(            // 内部断言，确保
        offsets_.size() == v->indices().size(),  // 偏移和索引长度相等
        buildErrorMessage(
            "Expected ranks to match in CacheReplacer in the fuser."));  // 构建错误消息
    for (size_t i = 0; i < v->indices().size(); ++i) {
      ExprPtr index = v->indices()[i]->accept_mutator(this);  // 变异索引表达式
      ExprPtr offset = offsets_[i];   // 获取偏移表达式
      ExprPtr sub = IRSimplifier::simplify(alloc<Sub>(index, offset));  // 简化索引偏移计算
      newIndices.push_back(sub);      // 添加到新索引表达式向量
    }
    v->set_buf(cache_);               // 设置存储语句的缓冲区为缓存缓冲区
    v->set_indices(newIndices);       // 设置存储语句的索引为新索引表达式
    v->set_value(newValue);           // 设置存储语句的值为新值表达式
    return v;
  }



    返回变量 v 的值，结束当前函数的执行并返回该值



  BufPtr buf_;
  BufPtr cache_;
  std::vector<ExprPtr>& offsets_;



  // 声明三个成员变量，分别为指向 BufPtr 类型对象的指针 buf_ 和 cache_，以及引用 std::vector<ExprPtr> 类型的向量 offsets_
  };

  // 缓存访问操作的实现
  LoopNest::AccessResult LoopNest::cacheAccesses(
      // 生产者缓冲区指针
      BufPtr producer,
      // 缓存名
      const std::string& name,
      // 消费者语句指针
      StmtPtr consumer) {
    // 减少操作指针，默认为空
    ReduceOpPtr reduceOp{nullptr};
    // 查找消费者语句中的所有存储节点
    auto stores = NodeFinder<Store>::find(consumer);
    for (const auto& store : stores) {
      // 如果存储节点的值是减少操作
      if (auto ro = to<ReduceOp>(store->value())) {
        // 如果存储节点的缓冲区不是生产者，则继续下一个循环
        if (store->buf() != producer) {
          continue;
        }

        // 如果已经有减少操作存在，则抛出运行时错误
        if (reduceOp) {
          throw std::runtime_error(
              "can only cache accesses used by at most a single reduceOp");
          return {nullptr, nullptr};
        }

        reduceOp = ro;
      }
    }

    // 检查边界，但不关心访问类型
    auto consumer_bounds_info = inferBounds(consumer, false);
    auto bounds_it = consumer_bounds_info.find(producer);
    // 如果消费者没有使用生产的张量，则抛出运行时错误
    if (bounds_it == consumer_bounds_info.end()) {
      throw std::runtime_error("consumer does not use the Tensor produced");
      return {nullptr, nullptr};
    }

    // 断言只有一个边界信息项
    TORCH_INTERNAL_ASSERT(
        bounds_it->second.size() == 1,
        buildErrorMessage(
            "Unexpected number of bound info entries in cacheAccesses in the fuser."));
    TensorAccessBoundsInfo& info = bounds_it->second[0];
    // 检查是否有读取或写入操作
    bool hasReads = info.kind == kLoad || info.kind == kMutate;
    bool hasWrites = info.kind == kStore || info.kind == kMutate;

    // 循环中使用的变量名列表
    std::vector<std::string> var_names = {"i", "j", "k", "l", "m", "n", "o", "p"};
    std::vector<ExprPtr> tmp_dims;
    std::vector<VarPtr> new_loop_vars;
    std::vector<ExprPtr> new_loop_vars_expr;

    // 确定缓存的大小，并为每个维度创建一个循环变量
    for (size_t i = 0; i < info.start.size(); ++i) {
      // 简化维度计算
      ExprPtr dim = IRSimplifier::simplify(alloc<Add>(
          alloc<Sub>(info.stop[i], info.start[i]), immLike(info.stop[i], 1)));

      tmp_dims.push_back(dim);

      // 创建新的循环变量
      new_loop_vars.push_back(
          alloc<Var>(var_names[i % var_names.size()], info.stop[i]->dtype()));
      new_loop_vars_expr.push_back(new_loop_vars[i]);
    }

    // 创建缓存变量
    BufPtr tmp_buf =
        alloc<Buf>(alloc<Var>(name, kHandle), tmp_dims, producer->dtype());

    // 基于每个轴的循环开始确定缓存调用的偏移量
    std::vector<ExprPtr> tmp_params;
    for (size_t i = 0; i < new_loop_vars.size(); ++i) {
  // 将新的循环变量和起始值添加到临时参数列表中
  tmp_params.push_back(alloc<Add>(new_loop_vars[i], info.start[i]));
}

// 使用缓存替换消费者中对生产者的访问
CacheReplacer replacer(producer, tmp_buf, info.start);
consumer->accept_mutator(&replacer);

// 将替换后的消费者替换原有消费者
BlockPtr consumer_block = to<Block>(consumer);
BlockPtr parent_block = to<Block>(consumer->get_parent());
// 如果消费者是一个块，则在原地进行变异
bool is_block = consumer_block != nullptr;

// 如果存在约简操作且正在处理约简轴，需要初始化缓存为0
bool on_reduce_axis = false;
if (reduceOp) {
  // 将约简操作中的参数转换为集合
  std::set<VarPtr> reduce_args(
      reduceOp->reduce_args().begin(), reduceOp->reduce_args().end());
  std::set<VarPtr> enclosing_vars;
  // 查找消费者所在的所有循环，并将循环变量加入集合中
  for (const auto& enclosing_for_stmt : NodeFinder<For>::find(consumer)) {
    enclosing_vars.insert(enclosing_for_stmt->var());
  }
  // 检查约简操作的参数是否在循环变量集合中
  for (const auto& reduce_arg : reduce_args) {
    if (enclosing_vars.find(reduce_arg) == enclosing_vars.end()) {
      on_reduce_axis = true;
    }
  }
}
// 如果存在约简操作且在约简轴上，则执行以下操作
if (reduceOp && on_reduce_axis) {
  // 初始化缓存为0
  StmtPtr tmp_init = alloc<Store>(
      tmp_buf, new_loop_vars_expr, getImmediateByType(tmp_buf->dtype(), 0));

  // 嵌套循环，根据新的循环变量维度生成初始化语句
  for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
    tmp_init = alloc<For>(
        new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_init);
  }

  // 如果消费者是一个块，则在其前面插入初始化语句
  if (is_block) {
    consumer_block->prepend_stmt(tmp_init);
  } else {
    // 否则，在消费者之前插入初始化语句
    parent_block->insert_stmt_before(tmp_init, consumer);
  }

  // 将结果写回原始缓存
  StmtPtr tmp_store = alloc<Store>(
      producer,
      tmp_params,
      reduceOp->reducer()(
          producer,
          alloc<Load>(tmp_buf, new_loop_vars_expr),
          tmp_params,
          {}));

  // 嵌套循环，根据新的循环变量维度生成写回语句
  for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
    tmp_store = alloc<For>(
        new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_store);
  }

  // 如果消费者是一个块，则在其后面追加写回语句
  if (is_block) {
    consumer_block->append_stmt(tmp_store);
  } else {
    // 否则，在消费者之后插入写回语句
    parent_block->insert_stmt_after(tmp_store, consumer);
  }

  // 返回缓存和替换后的消费者作为结果
  return std::make_pair(tmp_buf, consumer);
}

// 如果存在读操作，则将消费者的值填充到缓存中
StmtPtr tmp_store = alloc<Store>(
    tmp_buf, new_loop_vars_expr, alloc<Load>(producer, tmp_params));

// 嵌套循环，根据新的循环变量维度生成填充语句
for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
  tmp_store = alloc<For>(
      new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_store);
}

// 如果消费者是一个块，则在其前面插入填充语句
if (is_block) {
  consumer_block->prepend_stmt(tmp_store);
} else {
  // 否则，在消费者之前插入填充语句
  parent_block->insert_stmt_before(tmp_store, consumer);
}
  }
}

if (hasWrites) {
  // 同步缓存到生产者缓冲区。
  // 创建一个存储语句，将缓存中的数据存储到生产者缓冲区中。
  StmtPtr tmp_store = alloc<Store>(
      producer, tmp_params, alloc<Load>(tmp_buf, new_loop_vars_expr));

  // 从后向前遍历新的循环变量，创建嵌套的 For 循环语句。
  for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
    tmp_store = alloc<For>(
        new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_store);
  }

  // 如果是块级语句，则将存储语句追加到消费者块中；否则将其插入到消费者语句之后。
  if (is_block) {
    consumer_block->append_stmt(tmp_store);
  } else {
    parent_block->insert_stmt_after(tmp_store, consumer);
  }
}

// 返回临时缓冲区和消费者的配对
return std::make_pair(tmp_buf, consumer);
}

// 定义 LoopNest 类的 computeAt 方法，将生产者语句移到循环内计算
void LoopNest::computeAt(StmtPtr s, ForPtr f) {
  // 获取生产者语句 s 的存储语句 st
  StorePtr st = getStoreStmtOfProducer(s);
  // 如果没有存储语句，直接返回
  if (!st) {
    return;
  }

  // 推断循环中所有访问的边界信息
  auto loop_bounds_info = inferBounds(f->body());

  // bounds_it 保存我们正在尝试移到循环中的存储的边界信息。如果结果在循环中根本没有访问 - 什么都不做并提前退出。
  auto bounds_it = loop_bounds_info.find(st->buf());
  if (bounds_it == loop_bounds_info.end()) {
    return;
  }

  // 计算我们需要分配的临时缓冲区的维度
  std::vector<ExprPtr> dims = getBoundExtents(bounds_it->second);

  // TODO: 使用生产者的名称提示而不是 "temp"
  // 分配一个名为 temp 的缓冲区
  BufPtr temp_buf = alloc<Buf>("temp", dims, st->value()->dtype());

  // 为 'temp' 生成索引变量
  std::vector<ExprPtr> temp_indices(dims.size());
  for (const auto i : c10::irange(dims.size())) {
    // TODO: 使用生产者索引的名称提示而不是 'idx'
    // 分配一个名为 'idx<i>' 的变量作为索引
    temp_indices[i] =
        alloc<Var>(std::string("idx") + std::to_string(i), dims[i]->dtype());
  }

  // 准备替换规则，用于从生产语句构建临时语句
  // TODO: 不应该通过上升循环嵌套来进行，而应该通过原始张量表达式中的索引进行。嵌套中的循环可能已被修改（例如拆分或合并），因此循环索引不再对应原始表达式的索引，甚至其数量可能不同。在这种情况下，下面的循环将崩溃。
  std::vector<VarPtr> prod_indices = getOuterLoopIndexes(s);
  std::vector<std::pair<VarPtr, ExprPtr>> rewrite_indices_map;
  std::vector<ExprPtr> offsets;
  for (const TensorAccessBoundsInfo& p : bounds_it->second) {
    for (const auto i : c10::irange(p.start.size())) {
      if (offsets.size() <= i) {
        offsets.push_back(p.start[i]);
      } else {
        offsets[i] =
            IRSimplifier::simplify(alloc<Min>(offsets[i], p.start[i], true));
      }
    }
  }

  // 构建替换映射，将生产者语句的索引映射到临时语句的索引和偏移量
  for (const auto i : c10::irange(prod_indices.size())) {
    rewrite_indices_map.emplace_back(
        prod_indices[i], alloc<Add>(temp_indices[i], offsets[i]));
  }

  // 构造临时存储语句
  StmtPtr bd = alloc<Store>(
      temp_buf,
      temp_indices,
      SubstituteInClone(st->value(), rewrite_indices_map));

  // 构造用于临时计算的循环嵌套
  for (const auto i : c10::irange(dims.size())) {
    // 我们从内向外创建循环，因此需要反向访问维度。
    size_t dim_idx = dims.size() - 1 - i;
  // 使用 alloc 函数为循环体创建一个 For 节点，初始化表达式如下：
  // - 起始变量为 temp_indices[dim_idx]
  // - 起始值为 0
  // - 循环上限为 dims[dim_idx]
  // - 循环体为 bd
  bd = alloc<For>(
      to<Var>(temp_indices[dim_idx]),
      immLike(dims[dim_idx], 0),
      dims[dim_idx],
      bd);
}

// 将构建好的循环体 bd 添加到函数 f 的主体的开头
f->body()->prepend_stmt(bd);

// 使用 LoopComputeAtRewriter 对象 lr 重写消费者函数中对生产者的访问，
// 将访问转换为对临时缓冲区 temp 的访问
LoopComputeAtRewriter lr(st->buf(), temp_buf, offsets);

// 调用 accept_mutator 方法将 lr 应用到函数 f 上，得到一个新的函数 new_f
StmtPtr new_f = f->accept_mutator(&lr);

// 如果原函数 f 和新函数 new_f 不同，替换函数 f 所在的块 bb 中的 f 为 new_f
if (f != new_f) {
  BlockPtr bb = to<Block>(f->get_parent());
  bb->replace_stmt(f, new_f);
}
}

class RfactorStoreRewriter : public IRMutator {
 public:
  RfactorStoreRewriter(
      BufPtr old_buf,
      const std::vector<ExprPtr>& old_indices,
      BufPtr new_buf,
      VarPtr reduction_var)
      : old_buf_(std::move(old_buf)),
        old_indices_(old_indices),
        new_buf_(std::move(new_buf)),
        reduction_var_(std::move(reduction_var)),
        new_indices_(old_indices) {
    new_indices_.push_back(reduction_var_);
  }

  // 重写 Load 节点的变换方法
  ExprPtr mutate(LoadPtr v) override {
    // 如果 Load 的缓冲区不是旧缓冲区，直接调用父类的变换方法
    if (v->buf() != old_buf_) {
      return IRMutator::mutate(v);
    }

    // 断言加载节点的索引数与旧索引数相同
    TORCH_INTERNAL_ASSERT(
        old_indices_.size() == v->indices().size(),
        buildErrorMessage(
            "Expected ranks to match in RfactorStoreRewriter in the fuser."));

    // 检查加载节点的索引是否与旧索引相等
    bool equal_indices = true;
    for (size_t i = 0; i < v->indices().size(); ++i) {
      if (!exprEquals(v->indices()[i], old_indices_[i])) {
        equal_indices = false;
        break;
      }
    }
    // 如果索引不相等，则调用父类的变换方法
    if (!equal_indices) {
      return IRMutator::mutate(v);
    }

    // 返回使用新缓冲区和新索引构建的加载节点
    return alloc<Load>(new_buf_, new_indices_);
  }

  // 重写 ReduceOp 节点的变换方法
  ExprPtr mutate(ReduceOpPtr v) override {
    // 对 ReduceOp 的主体表达式应用当前变换器
    ExprPtr body_new = v->body()->accept_mutator(this);

    // 创建新的约简参数列表，排除当前约简变量
    std::vector<VarPtr> new_reduce_args;
    for (const auto& r : v->reduce_args()) {
      if (r != reduction_var_) {
        new_reduce_args.push_back(r);
      }
    }

    // 返回使用新的主体表达式和新约简参数列表构建的 ReduceOp 节点
    return alloc<ReduceOp>(body_new, new_reduce_args, v->reducer());
  }

  // 重写 Store 节点的变换方法
  StmtPtr mutate(StorePtr v) override {
    // 如果 Store 的缓冲区不是旧缓冲区，直接调用父类的变换方法
    if (v->buf() != old_buf_) {
      return IRMutator::mutate(v);
    }

    // 断言存储节点的索引数与旧索引数相同
    TORCH_INTERNAL_ASSERT(
        old_indices_.size() == v->indices().size(),
        buildErrorMessage(
            "Expected ranks to match in RfactorStoreRewriter in the fuser."));

    // 检查存储节点的索引是否与旧索引相等
    bool equal_indices = true;
    for (size_t i = 0; i < v->indices().size(); ++i) {
      if (!exprEquals(v->indices()[i], old_indices_[i])) {
        equal_indices = false;
        break;
      }
    }
    // 如果索引不相等，则调用父类的变换方法
    if (!equal_indices) {
      return IRMutator::mutate(v);
    }

    // 对存储节点的值表达式应用当前变换器
    ExprPtr new_value = v->value()->accept_mutator(this);
    // 返回使用新缓冲区、新索引和新值构建的存储节点
    return alloc<Store>(new_buf_, new_indices_, new_value);
  }

 private:
  BufPtr old_buf_;  // 原始缓冲区
  const std::vector<ExprPtr>& old_indices_;  // 原始索引列表
  BufPtr new_buf_;  // 新缓冲区
  VarPtr reduction_var_;  // 约简变量
  std::vector<ExprPtr> new_indices_;  // 新索引列表
};

bool LoopNest::rfactor(StmtPtr st, ForPtr target_for) {
  BufPtr tmp_buf = nullptr;
  return rfactor(st, target_for, &tmp_buf);
}

// 在循环嵌套中进行因子化操作
bool LoopNest::rfactor(
    StmtPtr st,
    ForPtr outer_reduction_for,
    BufPtr* rfac_buf_ptr) {
  // 将语句转换为存储节点
  StorePtr reduction_store = to<Store>(st);
  // 将存储节点的值转换为 ReduceOp 节点
  ReduceOpPtr reduce_op = to<ReduceOp>(reduction_store->value());
  if (!reduce_op) {
    // 如果不是约简存储节点，则返回 false
    // 表示不是约简操作，无法进行因子化
    return false;
  }

  // 获取原始缓冲区、索引和约简变量
  auto orig_buf = reduction_store->buf();
  auto orig_buf_indices = reduction_store->indices();
  VarPtr reduction_var = outer_reduction_for->var();

  // 创建约简参数集合
  std::set<VarPtr> reduce_args = {
      reduce_op->reduce_args().begin(), reduce_op->reduce_args().end()};

  // 如果约简参数数量小于 2，则无法进行因子化
  if (reduce_args.size() < 2) {
    // Not enough reduction axis to do rfactor
    return false;
    
    
    
    // Verify that outer_reduction_for is a perfect loop nest with all loops being
    // reductions
    StmtPtr cur = outer_reduction_for;
    while (ForPtr cur_for = to<For>(cur)) {
        if (!reduce_args.count(cur_for->var())) {
            // output axis inside outer_reduction_for are not allowed
            return false;
        }
        reduce_args.erase(cur_for->var());
    
        BlockPtr b = cur_for->body();
        if (b->nstmts() != 1) {
            return false;
        }
        cur = b->stmts().front();
    }
    if (cur != st) {
        // The reduction store is not a single stmt in the innermost loop - bail in
        // that case
        return false;
    }
    if (!reduce_args.empty()) {
        // This is not the outermost reduction axis
        return false;
    }
    
    
    
    // assert: reduce_axis match loop vars from outer_reduction_for and inside
    // assert: no other stmts in outer_reduction_for or its child loops
    std::vector<ExprPtr> rfac_dims = orig_buf->dims();
    ExprPtr extra_dim = IRSimplifier::simplify(
        alloc<Sub>(outer_reduction_for->stop(), outer_reduction_for->start()));
    rfac_dims.push_back(extra_dim);
    ExprPtr rfac_init =
        alloc<Cast>(reduce_op->dtype(), reduce_op->reducer().initializer());
    
    *rfac_buf_ptr = alloc<Buf>(
        orig_buf->name_hint() + "_rfac",
        rfac_dims,
        reduce_op->dtype(),
        rfac_init);
    BufPtr rfac_buf = *rfac_buf_ptr;
    
    
    
    // Rewrite the original reduction store to use the temporary rfac buffer:
    //   1) X[*indexes] --> T[*indexes + {reduction_var}]
    //   2) reduce_axis -= {reduction_var}
    RfactorStoreRewriter rfac_rewriter(
        orig_buf, orig_buf_indices, rfac_buf, reduction_var);
    to<Block>(st->get_parent())
        ->replace_stmt(st, st->accept_mutator(&rfac_rewriter));
    
    
    
    // Insert a store for the final reduction over the temp buffer into the
    // original buffer:
    //   X[*indexes] = ReduceOp(X[*indexes] + T[*indexes + {reduction_var}],
    //                          reduce_axis={reduction_var})
    BlockPtr b = outer_reduction_for->body();
    TORCH_INTERNAL_ASSERT(
        b->nstmts() == 1,
        buildErrorMessage(
            "Expected to have a single stmt in the block in rfactor transformation in the fuser."));
    StmtPtr first_reduction_loop = b->stmts().front();
    auto rfac_buf_indices = orig_buf_indices;
    rfac_buf_indices.emplace_back(reduction_var);
    
    ExprPtr final_reduce_load = alloc<Load>(rfac_buf, rfac_buf_indices);
    outer_reduction_for->body()->insert_stmt_after(
        alloc<Store>(
            orig_buf,
            orig_buf_indices,
            reduce_op->reducer()(
                orig_buf, final_reduce_load, orig_buf_indices, {reduction_var})),
        first_reduction_loop);
    
    
    
    // Insert an initialization store for the temp buffer:
    //   T[a,b,c] = init
    outer_reduction_for->body()->insert_stmt_before(
        alloc<Store>(rfac_buf, rfac_buf_indices, rfac_init),
        first_reduction_loop);
    return true;
}

} // namespace torch::jit::tensorexpr
```