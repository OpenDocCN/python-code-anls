# `.\pytorch\torch\csrc\jit\tensorexpr\loopnest.h`

```py
#pragma once

#include <string>  // 包含标准字符串库头文件
#include <unordered_map>  // 包含无序映射头文件
#include <unordered_set>  // 包含无序集合头文件
#include <vector>  // 包含向量头文件

#include <torch/csrc/Export.h>  // 引入 Torch 导出定义头文件
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>  // 引入前向声明头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;  // 声明表达式类
class Var;  // 声明变量类
class Buf;  // 声明缓冲区类
class Tensor;  // 声明张量类
class Function;  // 声明函数类
class Stmt;  // 声明语句类
class For;  // 声明 for 循环类
class Block;  // 声明代码块类
class Store;  // 声明存储操作类
class Dtype;  // 声明数据类型类

class TORCH_API LoopNest {
 public:
  // 从输出张量列表和待计算张量列表构建 LoopNest 的构造函数
  LoopNest(
      const std::vector<Tensor>& output_tensors,
      const std::vector<Tensor>& tensors_to_compute);

  // 方便构造函数，当所有张量均为输出张量时使用
  LoopNest(const std::vector<Tensor>& output_tensors);

  // 从语句和输出缓冲区列表构建 LoopNest 的构造函数
  LoopNest(StmtPtr stmt, std::unordered_set<BufPtr> output_bufs);

  // 从另一个 LoopNest 构建 LoopNest 的构造函数，克隆其语句
  LoopNest(const LoopNest& other);

  // 返回根语句的方法
  StmtPtr root_stmt() const {
    return output_bufs_;  // 返回输出缓冲区
  }

  // 获取所有中间缓冲区的方法
  std::vector<BufPtr> getIntermediateBufs() const;

  // 查找两个 for 循环之间的外部循环的静态方法
  static ForPtr findOuterFor(ForPtr a, ForPtr b);

 private:
  // 初始化函数，从输出张量列表和待计算张量列表构建 LoopNest
  void initialize(
      const std::vector<Tensor>& output_tensors,
      const std::vector<Tensor>& tensors_to_compute);

  StmtPtr root_stmt_;  // 根语句的指针

  std::unordered_set<BufPtr> output_bufs_;  // 输出缓冲区的无序集合
};

TORCH_API StmtPtr FlattenIndexes(StmtPtr s);

// TODO: 在决定依赖分析的 API 后重新审视此处的实现
// 也许会选择使用不同的 API，并删除 BufUse，或者如果决定保留它，则需要正确文档化其 API
struct BufLoadOrStoreUse {
  StmtPtr s;  // 语句指针
  bool isStore;  // 是否为存储操作的布尔值
};

/*
 * 返回映射 (Buf -> Buf 的使用) 的函数
 * 使用由 BufUse 元素表示的向量表示，其包括 StmtPtr 和一个 bool 的 isStore 标志
 * 向量中的使用顺序反映了给定语句中使用出现的顺序
 */
std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findLoadOrStoreUses(
    StmtPtr s);

// 用下划线替换所有无效字符的函数
TORCH_API std::string sanitizeName(const std::string& input_name);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```