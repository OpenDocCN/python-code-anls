# `.\pytorch\torch\csrc\jit\runtime\profiling_record.h`

```py
// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>

// 包含 Torch 库的导出头文件
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

// 包含 STL 库的数据结构头文件
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

// 我们希望为张量的每个位置/轴分配一个抽象大小
// * 对于每个张量 `tensor`，我们有一个经过分析的 `TensorType` 的 `Value`，描述了 `tensor` 的属性。
// * `TensorType` 有一个名为 `symbolic_sizes_` 的属性来描述观察到的 `tensor.sizes()`
// * `symbolic_sizes_` 是一个抽象大小的向量（或 `std::vector<ShapeSymbol>`），其中
//   * `ShapeSymbol` 在 `symbolic_sizes_[i]` 处描述了 `tensor.sizes()[i]` 的大小值（`Dimension`）
// * 我们可能会在 `tensor.sizes()` 的不同位置或不同的 `tensor` 中看到相同的 `Dimension`
//   * 首先，我们希望在 TorchScript 函数的 **一个** 分析执行或运行中将相同的 `ShapeSymbol` 关联到相同的 `Dimension`
//     * 在不同的 `TensorType` 中，`symbolic_shapes_` 的相同 `ShapeSymbol` 在不同位置上（即不同的 profiled 值的 `TensorType`）形成一个隐含集合。这种集合的元素称为 *dimension locations*。
//     * 这些集合允许我们跟踪输入参数的形状与某些操作的输出形状之间的关系，因为输入和输出的形状可能共享相同的 `ShapeSymbol`
// * 对于 **每次** 分析运行，我们希望保持一个不变：*相同的 `ShapeSymbol` 总是与相同的 `Dimension` 关联*
// * 为了维护这个不变性，我们将来自所有分析运行的分析信息合并，
//   * 对于每两个运行，我们遍历所有 `symbolic_shapes_` 并比较它们在相同位置的 `ShapeSymbol`
//     * 如果我们观察到在运行 #1 中具有 `ShapeSymbol S1` 的每个维度位置，在运行 #2 中有 **唯一的** `ShapeSymbol S2`，我们得出结论不变性成立。
//     * 然而，如果我们观察到在运行 #2 中某些维度位置具有 `ShapeSymbol S2` 而其他位置具有 `ShapeSymbol S3`，我们希望将与 `ShapeSymbol S1` 关联的维度位置的虚拟集合分成两个新的子集，以维持不变性。
//     * 分区的工作方式是为在运行 #2 中具有 `ShapeSymbol S2` 的维度位置分配一个新的符号，为在运行 #2 中具有 `ShapeSymbol S3` 的维度位置分配另一个新的符号。换句话说，
//       * 子集 #1 将包含在运行 #2 中具有 `ShapeSymbol S2` 的维度位置，并将在这些维度位置上具有 `ShapeSymbol S4`
//       * 子集 #2 将包含在运行 #2 中具有 `ShapeSymbol S3` 的维度位置，并将在这些维度位置上具有 `ShapeSymbol S5`
// `ShapeSymbol S4` 和 `ShapeSymbol S5` 在这些维度位置上将合并分析信息的结果
// 每个分析运行的效果是，新的 `TensorTypes` 其 `symbolic_sizes_` /dimension 位置要么有 `ShapeSymbol S4`，要么有 `ShapeSymbol S5`。
// 即使在看到与 `ShapeSymbol S1` 关联的所有维度位置之前，也可以进行分区
// 我们使用 `ShapeSymbolTable` 的 `getSymbolInSet` 记住了我们在运行 #2 中观察到的所有与 `ShapeSymbol S1` 关联的维度位置中的 `ShapeSymbols`。
// 对于运行 #2 中与与 `ShapeSymbol S1` 关联的维度位置上的每个 `ShapeSymbol`，`getSymbolInSet` 返回一个符号，我们将其分配给新的 TensorType 的维度位置。
// 需要强调的是，对于运行 #1 中具有不同 `ShapeSymbol` 的两个维度位置中相同的 `ShapeSymbol S2`，它们是不同的！这些维度位置将属于不同的子集，并在合并后具有不同的 `ShapeSymbol`。
// 另一方面，对于运行 #1 中具有 `ShapeSymbol S1` 的两个维度位置中相同的 `ShapeSymbol S2`，`getSymbolInSet` 将返回相同的符号。
    # 如果对象 s 是静态的（static），则执行以下操作
    if (s.is_static()) {
      # 将静态对象 s 的静态大小和对象 s 插入到集合 set 中作为一个元组
      set.insert({s.static_size(), s});
    }
    # 返回集合 set，其中包含静态对象 s 的静态大小和对象 s 的元组
    return set;
  }
// ShapeSymbolTable is used by Interpreter
// to assign dimension values to ShapeSymbols
// and fail a guard if the same symbol
// is assigned more than one dimension value.
struct ShapeSymbolTable {
  // N.B. we treat static symbols as always assigned
  // to themselves
  bool isBound(c10::ShapeSymbol s) {
    if (s.is_static()) {  // 如果符号是静态的，则始终被视为已分配
      return true;
    }
    return data_.count(s) != 0;  // 检查是否已经为符号分配了维度值
  }

  // N.B. we treat static symbols as always assigned
  // to themselves
  Dimension getValue(c10::ShapeSymbol s) {
    if (s.is_static()) {  // 如果符号是静态的，则返回其静态大小
      return s.static_size();
    }
    return data_[s];  // 返回符号对应的维度值
  }
  void assign(c10::ShapeSymbol s, Dimension v) {
    TORCH_INTERNAL_ASSERT(!s.is_static());  // 断言符号不是静态的
    data_[s] = v;  // 为符号分配给定的维度值
  }
  std::map<c10::ShapeSymbol, Dimension> data_;  // 存储符号和其对应维度值的映射表

  // Tries to assign dimension values from `new_sizes` to
  // `ShapeSymbol`s `sym_shapes`.
  // Returns `true` if every dimension value from `new_sizes`
  // can be assigned to the corresponding `ShapeSymbol` from
  // `sym_shapes`
  // A dimension value can be assigned to a `ShapeSymbol`
  // * if the symbol isn't assigned yet any dimension value
  // * if the symbol is assigned and its value is equal to
  // the dimension value from `new_sizes`
  bool bindSymbolicShapes(
      at::IntArrayRef new_sizes,
      const c10::SymbolicShape& sym_shapes);
};

struct ProfilingRecord {
  // N.B. ProfilingRecord's copy and move c-tor are disabled, so we won't
  // end up accidentally copying or moving ProfilingRecords whose addresses
  // are captured in callbacks_
  ProfilingRecord(const ProfilingRecord&) = delete;  // 禁用复制构造函数
  ProfilingRecord(ProfilingRecord&&) noexcept = delete;  // 禁用移动构造函数
  TORCH_API static std::unique_ptr<ProfilingRecord> instrumentGraph(
      const std::shared_ptr<Graph>& graph);
  TORCH_API static void removeProfilingNodes(Block* b);
  TORCH_API static void removeProfileCounter(Block* b);

  std::shared_ptr<Graph> profiled_graph_;  // 被分析的图
  mutable std::mutex mutex_;  // 互斥锁，用于保护对象状态的并发访问
  size_t profiling_count_;  // 分析计数

  bool ready() const;  // 返回是否准备就绪

  std::shared_ptr<Graph> graph() const {
    return profiled_graph_;  // 返回被分析的图
  }

  TORCH_API ProfileIValueOp* createProfileIValueNode(Value* in_val);  // 创建值的分析节点
  TORCH_API ProfileIValueOp* createProfileIValueNode(ArrayRef<Value*> inputs);  // 创建多个值的分析节点

 private:
  ProfileOp* createProfileNode(
      const std::function<void(Stack&)>& fp,
      at::ArrayRef<Value*> inputs);  // 创建分析节点
  void instrumentBlock(Block* block);  // 在块中进行分析仪器操作
  void insertShapeProfile(Node* n, size_t offset, const TypePtr& input_type);  // 插入形状分析

  ProfilingRecord(std::shared_ptr<Graph> g);  // 私有构造函数，创建分析记录对象
};

} // namespace torch::jit  // 命名空间 torch::jit
```