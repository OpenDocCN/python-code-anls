# `.\pytorch\torch\csrc\jit\runtime\interpreter\code_impl.h`

```py
#pragma once

#include <memory>  // 包含内存管理相关的头文件
#include <unordered_map>  // 包含无序映射容器相关的头文件
#include <vector>  // 包含向量容器相关的头文件

#include <c10/util/irange.h>  // 包含C10库中整数范围的头文件
#include <torch/csrc/jit/api/function_impl.h>  // 包含JIT库中函数实现相关的头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含JIT库中IR节点相关的头文件
#include <torch/csrc/jit/jit_log.h>  // 包含JIT库中日志记录相关的头文件
#include <torch/csrc/jit/passes/bailout_graph.h>  // 包含JIT库中bailout图相关的头文件
#include <torch/csrc/jit/runtime/calculate_necessary_args.h>  // 包含JIT库中计算必要参数相关的头文件
#include <torch/csrc/jit/runtime/graph_iterator.h>  // 包含JIT库中图遍历相关的头文件
#include <torch/csrc/jit/runtime/instruction.h>  // 包含JIT库中指令相关的头文件
#include <torch/csrc/jit/runtime/interpreter/preprocess_graph.h>  // 包含JIT库中预处理图相关的头文件

C10_DECLARE_bool(torch_jit_enable_expanded_stacks);  // 声明一个外部布尔类型变量，用于控制是否启用扩展堆栈功能

namespace torch::jit {

std::ostream& operator<<(std::ostream& out, Instruction inst);  // 重载流输出运算符，用于打印指令对象

namespace interpreter {

template <class Ttarget, class Tsource>
Ttarget safe_narrow_cast(Tsource v) {
  Ttarget res = static_cast<Ttarget>(v);
  // Casting it back to check whether it overflew.
  // 将其再次强制转换回来以检查是否溢出
  if (static_cast<Tsource>(res) != v) {
    TORCH_WARN(
        "ATTENTION: your model computation is overflowing, safe_narrow_cast<>() failed");  // 如果类型转换溢出，则发出警告信息
    return v;  // 返回原始值
  }
  return res;  // 返回转换后的值
}

// BailoutBlocks are used to temporarily store
// instructions (typically, argument LOADs and TAIL_CALL)
// generated for prim::BailOut nodes
// before they are merged back into
// CodeImpl._instructions_ by insertBailoutBlocks
// 用于临时存储prim::BailOut节点生成的指令（通常是参数加载和TAIL_CALL）的BailoutBlocks，
// 在通过insertBailoutBlocks将它们合并回CodeImpl._instructions_之前
struct BailoutBlock {
  size_t jf_instruction_index;  // 节点失败时将被修补以跳转到此处的指令索引
  std::vector<Instruction> instructions;  // 指令向量，以TAIL_CALL结束

  explicit BailoutBlock(size_t jf_index) : jf_instruction_index(jf_index) {}  // 显式构造函数，初始化jf_instruction_index
};

// for keeping track of the current node
// 用于跟踪当前节点
struct WithCurrentNode {
  WithCurrentNode(Node** loc, Node* new_value) : loc_(loc), old_value_(*loc_) {
    *loc = new_value;  // 更新当前节点指针的值为新的节点指针
  }
  ~WithCurrentNode() {
    *loc_ = old_value_;  // 恢复当前节点指针的值为旧的节点指针
  }

 private:
  Node** loc_;  // 当前节点指针的指针
  Node* old_value_;  // 旧的当前节点指针值
};

struct NodeSourceInfo {
  const char* func_name_;  // 函数名称的C风格字符串
  const char* file_name_;  // 文件名称的C风格字符串
  size_t line_;  // 行号
  NodeSourceInfo() : func_name_(nullptr), file_name_(nullptr), line_(0) {}  // 构造函数，初始化成员变量
};

struct CodeImpl {
  friend struct InterpreterState;  // 声明InterpreterState为友元类

  std::vector<Instruction> instructions_;  // 指令向量

  const c10::unique_t node_stack_attr_symbol_ =  // 节点堆栈属性符号的唯一值
      static_cast<c10::unique_t>(attr::node_stack_idx);  // 将attr::node_stack_idx强制转换为唯一值类型

  std::vector<std::vector<NodeSourceInfo>> expanded_node_stacks_;  // 扩展的内联堆栈，作为指向内联调用堆栈中值的指针的指针向量

  std::vector<Node*> instructions_source_;  // 每个指令对应的图中节点指针向量
  std::vector<IValue> constant_table_;  // 常量表，存储常量值
  std::vector<Operation> operator_table_;  // 操作表，存储操作对象

#ifndef NDEBUG
  std::vector<Operator> full_operator_table_;  // 完整的操作符表，在调试模式下使用
  graph_ = preprocess_.graph;  // 将预处理的图赋值给graph_
  n_outputs = graph_->outputs().size();  // 计算图的输出数量
  if (n_outputs == 1) {
    return_type_ = graph->outputs().at(0)->type();  // 如果只有一个输出，返回该输出的类型
  } else {
    return_type_ = TupleType::create(
        fmap(graph->outputs(), [](const Value* v) { return v->type(); }));  // 如果有多个输出，创建一个元组类型
  }
  n_inputs = graph_->inputs().size();  // 计算图的输入数量
    if (emit_instructions) {
      // 如果需要发出指令，则调用 run() 方法执行指令生成
      // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
      run();
    }
  }

  virtual ~CodeImpl() = default;

  // 由于 CodeImpl 的子类需要填充 op_to_num_specified_args，
  // 因此我们将修改 CodeImpl 内部的调用分离到一个单独的函数中。
  virtual void run() {
    // 为当前的图块生成代码
    emitCodeForBlock(graph_->block());
    // 插入 RET 指令
    insertInstruction(RET);
    // 延迟生成 bailout 块，使其出现在最后
    // 现在生成它们并修正跳转
    insertBailoutBlocks();
  }

  const std::vector<c10::IValue>& constant_table() const {
    // 返回常量表
    return constant_table_;
  }

  void request_bailout(size_t index) {
    auto count = index;
    // 遍历指令列表
    for (const auto instr_index : c10::irange(instructions_.size())) {
      // 如果是 GUARD 或 FAIL_GUARD 操作码
      if (instructions_[instr_index].op == GUARD ||
          instructions_[instr_index].op == FAIL_GUARD) {
        // 找到对应的 bailout 请求
        if (count-- == 0) {
          // 将 GUARD 操作码修改为 FAIL_GUARD
          instructions_[instr_index].op = FAIL_GUARD;
          // 调试信息：在指令 instr_index 处添加了一个 bailout 请求
          GRAPH_DEBUG(
              "Added a bailout request for ",
              index,
              " at instruction ",
              instr_index);
          break;
        }
      }
    }
  }

  const std::vector<Instruction>& instructions() const {
    // 返回指令列表
    return instructions_;
  }

  const std::unordered_map<std::string, size_t>& op_to_num_specified_args()
      const {
    // 返回操作码到指定参数数目的映射表
    return op_to_num_specified_args_;
  }

  const std::vector<Node*>& instructions_source() const {
    // 返回指令源信息列表
    return instructions_source_;
  }

  NodeSourceInfo getSourceInfoFromSourceRange(const SourceRange& range) {
    NodeSourceInfo nodeSource;
    SourceRange r = range;
    // 如果范围有源，则查找生成范围的原始范围
    if (range.source()) {
      if (auto orig = range.source()->findSourceRangeThatGenerated(r)) {
        r = *orig;
      }
    }
    // 如果范围有源
    if (r.source()) {
      auto lineno = r.source()->lineno_for_offset(r.start());
      // 获取源行号并转换为源文件的行号
      nodeSource.line_ = r.source()->lineno_to_source_lineno(lineno);
      // 如果有文件名，则获取文件名并存储在 nodeSource 中
      if (r.source()->filename()) {
        nodeSource.file_name_ = r.source()->filename().value().c_str();
      }
    }
    return nodeSource;
  }

  void expandInlinedNodeStack(
      const InlinedCallStackPtr& cs,
      std::vector<NodeSourceInfo>* expandedstack) {
    // 获取调用堆栈的源信息
    auto nodeSourceInfo = getSourceInfoFromSourceRange(cs->source_range());
    // 获取调用堆栈的函数名并存储在 nodeSourceInfo 中
    nodeSourceInfo.func_name_ = cs->function_name().c_str();
    expandedstack->emplace_back(nodeSourceInfo);

    // 如果存在调用方，则展开调用方的节点堆栈
    if (cs->callee()) {
      expandInlinedNodeStack(cs->callee().value(), expandedstack);
    }
  }

  void getNodeStack(
      const Node* node,
      std::vector<NodeSourceInfo>* expandedstack) {
    // 如果当前节点有调用堆栈，则展开调用堆栈
    if (current_node_->callstack()) {
      expandInlinedNodeStack(current_node_->callstack().value(), expandedstack);
    }
    // 获取节点的源信息并存储在 expandedstack 中
    auto nodeSourceInfo = getSourceInfoFromSourceRange(node->sourceRange());
    expandedstack->emplace_back(nodeSourceInfo);
  }

  void insertInstruction(OpCode op, int64_t X = 0, uint64_t N = 0) {
    // 向 instructions_ 向量中添加一条指令，包括操作码 op，以及经过安全转换的参数 X 和 N
    instructions_.emplace_back(
        op,
        safe_narrow_cast<int32_t, int64_t>(X),
        safe_narrow_cast<uint16_t, uint64_t>(N));
    
    // 向 instructions_source_ 向量中添加当前节点的源信息
    instructions_source_.emplace_back(current_node_);

    // 如果启用了扩展堆栈功能并且当前节点不具有属性 attr::node_stack_idx
    if (FLAGS_torch_jit_enable_expanded_stacks &&
        !current_node_->hasAttribute(attr::node_stack_idx)) {
      // 创建一个扩展堆栈的向量 expandedStack，并获取当前节点的堆栈信息
      std::vector<NodeSourceInfo> expandedStack;
      getNodeStack(current_node_, &expandedStack);
      // 将扩展堆栈信息加入 expanded_node_stacks_ 向量，并记录插入的位置索引
      auto insertIdx = expanded_node_stacks_.size();
      expanded_node_stacks_.emplace_back(expandedStack);
      // 给当前节点设置属性 attr::node_stack_idx，指向插入的位置索引
      current_node_->i_(attr::node_stack_idx, insertIdx);
    }

    // 检查是否意外地以非拓扑顺序发射节点
    if (op == OP) {
      // 如果 last_inserted_op_ 不为空且当前节点不等于上一个插入的节点且它们属于同一个块
      if (last_inserted_op_ != nullptr && current_node_ != last_inserted_op_ &&
          current_node_->owningBlock() == last_inserted_op_->owningBlock()) {
        // 断言当前节点在上一个插入节点之后
        TORCH_INTERNAL_ASSERT(
            current_node_->isAfter(last_inserted_op_),
            *current_node_,
            " is not after ",
            *last_inserted_op_);
      }
      // 更新 last_inserted_op_ 为当前节点
      last_inserted_op_ = current_node_;
    }
  }

  // 缩减 instructions_ 和 instructions_source_ 向量的大小到指定的 size
  void truncateInstructions(size_t size) {
    while (instructions_.size() > size) {
      instructions_.pop_back();
      instructions_source_.pop_back();
    }
  }

  // 创建一个 bailout block，其起始索引为 jf_index
  void createBailoutBlock(size_t jf_index) {
    // 将一个新的 bailout block 加入 bailout_blocks_ 向量
    bailout_blocks_.emplace_back(jf_index);
    auto& bailout_instructions = bailout_blocks_.back().instructions;

    // 将 instructions_ 向量中从 jf_index + 1 开始的指令插入到 bailout_instructions 中
    bailout_instructions.insert(
        bailout_instructions.end(),
        instructions_.begin() + jf_index + 1,
        instructions_.end());
    
    // 缩减 instructions_ 向量的大小到 jf_index + 1
    truncateInstructions(jf_index + 1);
  }

  // 为一组值 vs 分配寄存器，并返回起始寄存器编号
  int allocRegs(at::ArrayRef<Value*> vs) {
    int result = register_size_ + 1;
    for (Value* v : vs) {
      // 断言 value_to_reg_ 中不包含当前值 v
      AT_ASSERT(value_to_reg_.count(v) == 0);
      // 将值 v 映射到新的寄存器编号，并递增寄存器大小
      value_to_reg_[v] = ++register_size_;
    }
    return result;
  }

  // 返回值 v 对应的寄存器编号
  int registerFor(Value* v) {
    return value_to_reg_.at(v);
  }

  // 发射对输入值 input 的使用指令，根据 drop 参数确定是否需要清空寄存器或弹出栈
  void emitUse(Value* input, bool drop) {
    // 如果预处理允许内联发射 input 的节点
    if (preprocess_.can_emit_inline[input->node()]) {
      // 发射 input 节点的指令
      emitNode(input->node());
      // 如果 drop 为 true，则插入 DROP 指令
      if (drop) {
        insertInstruction(DROP);
      }
    } else {
      // 否则根据 input 的情况选择操作码 op
      int reg = registerFor(input);
      bool moved = input->uses().size() == ++use_count_[input];

      OpCode op;
      if (input->node()->kind() == prim::Constant) {
        op = LOADC;
      } else if (moved) {
        op = MOVE;
      } else {
        op = LOAD;
      }

      // 如果 drop 为 true，则将 op 设置为 DROPR
      if (drop) {
        op = DROPR;
      }
      // 插入指令，指定操作码 op 和寄存器编号 reg
      insertInstruction(op, reg);
    }
  }

  // 发射一组输入值 inputs 的加载指令
  void emitLoadInputs(at::ArrayRef<Value*> inputs) {
    for (Value* input : inputs) {
      // 发射对 input 的使用指令，drop 参数为 false
      emitUse(input, false);
    }
  }

  // 发射一组输入值 inputs 的加载指令，并包括 num_include 次的重复
  void emitLoadInputs(at::ArrayRef<Value*> inputs, int num_include) {
    int count = 0;
  // 遍历输入值的列表
  for (Value* input : inputs) {
    // 如果计数小于指定的包含数量
    if (count < num_include) {
      // 发射对输入值的使用指令，指定不删除
      emitUse(input, false);
      // 增加计数器
      count++;
    }
  }
}

// 发射加载输入值的指令
void emitLoadInputs(at::ArrayRef<Value*> inputs, size_t start, size_t end) {
  // 遍历指定范围内的输入值
  for (size_t i = start; i < end; i++) {
    // 发射对输入值的使用指令，指定不删除
    emitUse(inputs[i], false);
  }
}

// 虚函数，发射操作符的指令
virtual void emitOperator(Node* node) {
  // 发射加载操作符的输入值的指令
  emitLoadInputs(node->inputs());
  // 获取节点对应的操作符
  const Operator& op = node->getOperator();
  // 获取节点输入值的数量
  int num_inputs = node->inputs().size();
  // 检查操作符的参数是否可变长度
  bool is_vararg = op.schema().is_vararg();

  // 将操作符添加到操作符表中，获取其索引
  int operation_index = add_to_operator_table(
      op,
      node,
      c10::toString(op.schema().operator_name()),
      num_inputs,
      is_vararg);

  // 如果操作符具有操作内容且参数可变长度
  if (op.hasOperation() && is_vararg) {
    // 插入带操作索引和输入数量的指令
    insertInstruction(OPN, operation_index, num_inputs);
  } else {
    // 插入仅带操作索引的指令
    insertInstruction(OP, operation_index);
  }
}

// 发射等待指令
void emitWait(Node* node) {
  // 发射加载输入值的指令
  emitLoadInputs(node->inputs());
  // 插入等待指令
  insertInstruction(WAIT);
}

// 发射丢弃指令
void emitDrop(at::ArrayRef<Value*> to_drop) {
  // 遍历要丢弃的值的列表
  for (Value* input : to_drop) {
    // 发射对输入值的使用指令，指定删除
    emitUse(input, true);
  }
}

// 发射存储输出值的指令
void emitStoreOutputs(Node* node) {
  // 获取节点输出值的数量
  size_t N = node->outputs().size();
  // 如果输出值数量为零，直接返回
  if (N == 0) {
    return;
  }
  // 分配寄存器给输出值
  int regs = allocRegs(node->outputs());
  // 如果只有一个输出值
  if (N == 1) {
    // 插入存储指令，带寄存器数
    insertInstruction(STORE, regs);
  } else {
    // 插入带寄存器数和输出值数量的存储指令
    insertInstruction(STOREN, regs, node->outputs().size());
  }
}

// 插入常量到常量表中，返回其索引
int insertConstant(IValue value) {
  // 获取当前常量表的大小作为索引
  int result = constant_table_.size();
  // 将常量值加入常量表
  constant_table_.emplace_back(std::move(value));
  // 返回插入的常量的索引
  return result;
}

// 虚函数，发射操作符或指令
virtual void emitOperatorOrInstruction(
    Node* node,
    OpCode op,
    int64_t X = 0,
    uint64_t N = 0,
    bool emit_inputs = true) {
  // 如果需要发射输入值
  if (emit_inputs) {
    // 发射加载输入值的指令
    emitLoadInputs(node->inputs());
  }
  // 插入指定操作码的指令，带参数 X 和 N
  insertInstruction(op, X, N);
}

// 发射格式化指令
void emitFormat(Node* node) {
  // 发射操作符或指令，格式化操作码，带输入值的数量作为 X
  emitOperatorOrInstruction(node, FORMAT, node->inputs().size(), 0);
}

// 检查节点并发射指令
void checkNodeAndEmit(Node* node) {
  // 获取节点对应的操作符
  const Operator& op = node->getOperator();
  // 获取操作符的唯一名称字符串
  std::string unique_op_name = c10::toString(op.schema().operator_name());
  // 如果操作符名称以 "aten::__getitem__.Dict" 开头
  if (unique_op_name.find("aten::__getitem__.Dict") == 0) {
    // 对于字典的 __get_item__ 重载操作符，需作为指令发射
    emitOperatorOrInstruction(node, DICT_INDEX);
  } else {
    // 否则作为操作符发射
    emitOperator(node);
  }
}

// 发射常量指令
void emitConstant(Node* node) {
  // 如果节点输出值的类型为函数类型，直接返回
  if (node->output()->type()->kind() == FunctionType::Kind) {
    return;
  }
  // 将常量插入常量表，并将输出值映射到寄存器
  value_to_reg_[node->output()] =
      insertConstant(toIValue(node->output()).value());
}

// 发射条件语句指令
void emitIf(Node* node) {
  // 发射加载输入值的指令
  emitLoadInputs(node->inputs());
  // 记录条件语句起始位置
  size_t start_if = instructions_.size();
  // 插入条件分支假指令，带一个填充的偏移量
  insertInstruction(JF, 0); // dummy offset to be filled in
  // 生成条件分支块的代码
  emitCodeForBlock(node->blocks().at(0));
  // 插入无条件跳转指令，带一个填充的偏移量
  insertInstruction(JMP, 0); // dummy offset
  // 记录否定分支起始位置
  size_t start_else = instructions_.size();
    // 设置指令 instructions_ 中的字段 X，表示 if 分支的结束位置到 else 分支开始位置的距离
    instructions_[start_if].X = start_else - start_if;
    // 为当前节点的第二个块生成代码
    emitCodeForBlock(node->blocks().at(1));
    // 设置指令 instructions_ 中的字段 X，表示 else 分支结束位置到当前指令的距离
    instructions_[start_else - 1].X = instructions_.size() - (start_else - 1);
  }

  void emitLoop(Node* loop) {
    // 插入 LOADC 指令，加载常量 0 到栈顶
    insertInstruction(LOADC, insertConstant(0));
    // 生成加载循环输入的指令
    emitLoadInputs(loop->inputs());
    // 记录循环开始的指令位置
    size_t start = instructions_.size();
    // 插入 LOOP 指令，指定循环的初始偏移量（占位）
    insertInstruction(LOOP, 0, loop->inputs().size()); // dummy offset
    // 为循环的第一个块生成代码
    emitCodeForBlock(loop->blocks().at(0));
    // 插入 JMP 指令，跳转到循环起始位置
    insertInstruction(JMP, start - instructions_.size());
    // 设置 LOOP 指令中的偏移量，指向循环结束位置与当前指令的距离
    instructions_[start].X = instructions_.size() - start;
  }

  void emitCall(Function* func, at::ArrayRef<Value*> inputs) {
    // 生成加载调用输入的指令
    emitLoadInputs(inputs);
    // 插入 CALL 指令，调用对应函数在函数表中的索引
    insertInstruction(CALL, function_table_.size());
    // 将函数对象添加到函数表中
    function_table_.emplace_back(func);
  }

  void emitNodeAtBlockLevel(Node* node) {
    // 使用当前节点作为当前节点
    WithCurrentNode guard(&current_node_, node);
    // 根据节点类型进行不同的处理
    switch (node->kind()) {
      case prim::Constant:
        // 生成常量节点的代码
        emitConstant(node);
        break;
      case prim::Return:
        // 生成加载返回输入的指令
        emitLoadInputs(node->inputs());
        break;
      default:
        if (!preprocess_.can_emit_inline[node]) {
          // 如果不能内联生成节点，则生成节点的代码并存储输出
          emitNode(node);
          emitStoreOutputs(node);
        }
        break;
    }
  }

  size_t emitType(TypePtr t) {
    // 记录当前类型表的大小
    size_t r = type_table_.size();
    // 将类型 t 添加到类型表中
    type_table_.emplace_back(std::move(t));
    return r;
  }

  void emitTypeCheck(Node* node) {
    auto num_inputs = node->inputs().size();

    // 检查 TypeCheck 节点至少有一个输入
    TORCH_INTERNAL_ASSERT(
        num_inputs && num_inputs + 1 == node->outputs().size());
    // 生成加载 TypeCheck 节点输入的指令
    emitLoadInputs(node->inputs());

    // 生成期望类型的指令
    size_t types_start = type_table_.size();
    auto types = node->tys(attr::types);
    for (const auto i : c10::irange(num_inputs)) {
      emitType(types[i]);
    }
    // 插入 TYPECHECK 指令，检查输入的类型
    insertInstruction(TYPECHECK, types_start, num_inputs);
  }

  size_t emitGuard(Node* node) {
    // 未优化的图位于索引 0
    // 受保护的输入位于索引 1
    // 其余参数依次跟随
    // 生成加载保护节点输入的指令
    emitLoadInputs(node->inputs().slice(1, 1));
    // 插入 GUARD 指令，检查类型并保护输入
    insertInstruction(GUARD, emitType(node->outputs().at(0)->type()));
    // 插入 JF 指令，跳转到待修补的位置
    insertInstruction(JF, 0 /* to be patched */);
    return instructions_.size() - 1;
  }

  void emitBailOut(Node* node) {
    // 生成保护节点的指令
    auto jf_index = emitGuard(node);
    // 获取未优化的图
    auto unoptimized_graph = node->inputs().at(0)->node()->g(attr::Subgraph);
    // 注意，保护的输入已加载到栈上，用于 GUARD 指令
    // 生成加载节点输入的指令
    emitLoadInputs(node->inputs().slice(2));
    // 插入 TAIL_CALL 指令，尾调用函数表中的函数
    insertInstruction(TAIL_CALL, function_table_.size());
    // 断言节点类型为 prim::BailOut
    TORCH_INTERNAL_ASSERT(node->kind() == prim::BailOut);
    // 获取 bailout 索引
    auto bailout_index = node->i(attr::index);
    TORCH_INTERNAL_ASSERT(bailout_index >= 0);

    // 构建 bailout 图
    auto build_bailout_graph = [bailout_index,
                                unoptimized_graph](GraphFunction& func) {
      BuildBailOutGraphFrom(bailout_index, unoptimized_graph, func.graph());
    };

    // 创建空图
    auto empty_graph = std::make_shared<Graph>();
  // 创建一个名为 "bailout" 的图函数对象，使用空图作为参数构建，使用 build_bailout_graph 函数创建
  auto func = std::make_unique<GraphFunction>(
      "bailout", empty_graph, build_bailout_graph);
  // 将 func 的指针添加到 function_table_ 中
  function_table_.emplace_back(func.get());
  // 将 func 移动到 bailout_functions_ 向量中
  bailout_functions_.emplace_back(std::move(func));
  // 根据给定的 jf_index 创建 bailout 块
  createBailoutBlock(jf_index);
}

void emitProfile(Node* node) {
  // 发出加载节点输入指令
  emitLoadInputs(node->inputs());
  // 插入 PROFILE_OP 指令，并指定 profile_function_table_ 的大小作为参数
  insertInstruction(PROFILE_OP, profile_function_table_.size());
  // 根据节点的具体类型将回调函数添加到 profile_function_table_ 中
  if (node->cast<ProfileOp>()) {
    profile_function_table_.push_back(node->cast<ProfileOp>()->getCallback());
  } else if (node->cast<ProfileIValueOp>()) {
    profile_function_table_.push_back(
        node->cast<ProfileIValueOp>()->getCallback());
  } else {
    // 如果节点类型未知，抛出内部断言错误
    TORCH_INTERNAL_ASSERT(false);
  }
}

void emitGetAttr(Node* node) {
  // 发出加载节点输入指令
  emitLoadInputs(node->inputs());
  // 获取节点输入的类型，预期为 ClassType
  const auto type = node->input()->type()->expect<ClassType>();
  // 获取节点的属性名称
  const auto& field = node->s(attr::name);
  // 获取属性在类型中的槽位
  const auto slot = type->getAttributeSlot(field);
  // 插入 GET_ATTR 指令，并指定属性槽位作为参数
  insertInstruction(GET_ATTR, slot);
}

void emitSetAttr(Node* node) {
  // 发出加载节点输入指令
  emitLoadInputs(node->inputs());
  // 获取节点输入的类型，预期为 ClassType
  const auto type = node->inputs().at(0)->type()->expect<ClassType>();
  // 获取节点的属性名称
  const auto& field = node->s(attr::name);
  // 获取属性在类型中的槽位
  const auto slot = type->getAttributeSlot(field);
  // 插入 SET_ATTR 指令，并指定属性槽位作为参数
  insertInstruction(SET_ATTR, slot);
}

void insertBailoutBlocks() {
  // 遍历 bailout_blocks_ 中的每个 BailoutBlock 对象
  for (const BailoutBlock& block : bailout_blocks_) {
    // 断言在 instructions_ 中的指令操作为 JF（条件跳转）
    TORCH_INTERNAL_ASSERT(instructions_[block.jf_instruction_index].op == JF)
    // 将 JF 指令的 X 字段设置为当前指令序列长度与 JF 指令索引之差
    instructions_[block.jf_instruction_index].X =
        instructions_.size() - block.jf_instruction_index;
    // 将 block 的指令序列插入到 instructions_ 的末尾
    instructions_.insert(
        instructions_.end(),
        block.instructions.begin(),
        block.instructions.end());
    // 将 block 的指令来源序列插入到 instructions_source_ 的末尾，保持一致性
    instructions_source_.insert(
        instructions_source_.end(),
        block.instructions.size(),
        instructions_source_[block.jf_instruction_index]);
  }
}

void emitInterfaceCall(
    std::string method_name_str,
    c10::ArrayRef<Value*> inputs) {
  // 发出加载输入指令
  emitLoadInputs(inputs);
  // 插入 INTERFACE_CALL 指令，并将方法名和输入的数量作为参数
  auto method_name = insertConstant(std::move(method_name_str));
  insertInstruction(INTERFACE_CALL, method_name, inputs.size());
}

void emitListUnpack(Node* node) {
  // 发出加载节点输入指令
  emitLoadInputs(node->inputs());
  // 插入 LIST_UNPACK 指令，并将节点输出的数量作为参数
  insertInstruction(LIST_UNPACK, node->outputs().size());
}

void emitTupleConstruct(Node* node) {
  // 检查输出是否为具名元组类型
  bool named =
      node->output()->type()->expectRef<TupleType>().name().has_value();
  // 如果是具名元组，调用 emitContainerConstruct 发出 NAMED_TUPLE_CONSTRUCT 指令
  if (named) {
    emitContainerConstruct(NAMED_TUPLE_CONSTRUCT, node);
  } else {
    // 否则，发出加载节点输入指令
    emitLoadInputs(node->inputs());
    // 插入 TUPLE_CONSTRUCT 指令，并将节点输入的数量作为参数
    insertInstruction(TUPLE_CONSTRUCT, node->inputs().size());
  }
}

void emitContainerConstruct(OpCode op, Node* node) {
  // 发出加载节点输入指令
  emitLoadInputs(node->inputs());
  // 插入 op 操作指令，同时插入节点输出类型作为参数
  insertInstruction(
      op, emitType(node->output()->type()), node->inputs().size());
}

void emitCreateObject(Node* node) {
  // 插入 CREATE_OBJECT 指令，并插入节点输出类型作为参数
  insertInstruction(CREATE_OBJECT, emitType(node->output()->type()));
}

void emitIsinstance(Node* node) {
  // 发出加载节点输入指令
  emitLoadInputs(node->inputs());
  std::vector<TypePtr> types = node->tys(attr::types);
  // 从节点中获取名为 'types' 的属性，并存储在类型为 TypePtr 的向量中
  size_t types_start = type_table_.size();
  // 记录当前类型表的大小，作为类型开始位置的标记
  for (const auto& typ : types) {
    // 遍历 'types' 向量中的每个类型
    emitType(typ);
    // 发出该类型的指令
  }
  insertInstruction(ISINSTANCE, types_start, types.size());
  // 在指令列表中插入 ISINSTANCE 指令，表示检查多个类型
}

void emitTupleSlice(Node* node) {
  emitLoadInputs(node->inputs());
  // 发出加载节点输入数据的指令
  int64_t beg_ind = node->i(attr::beg);
  // 获取节点属性 'beg' 的整数值，作为切片的起始索引
  int64_t end_ind = node->i(attr::end);
  // 获取节点属性 'end' 的整数值，作为切片的结束索引
  insertInstruction(TUPLE_SLICE, beg_ind, end_ind - beg_ind);
  // 在指令列表中插入 TUPLE_SLICE 指令，表示切片操作
}

void emitFork(Node* node) {
  emitLoadInputs(node->inputs());
  // 发出加载节点输入数据的指令
  auto forked_fn = std::make_unique<GraphFunction>(
      "<forked function>", node->g(attr::Subgraph), nullptr);
  // 创建一个新的图函数对象，表示分支函数
  forked_functions_.emplace_back(std::move(forked_fn));
  // 将新创建的分支函数对象移动到 forked_functions_ 向量末尾
  function_table_.emplace_back(forked_functions_.back().get());
  // 将分支函数对象添加到函数表中
  insertInstruction(FORK, function_table_.size() - 1, node->inputs().size());
  // 在指令列表中插入 FORK 指令，表示分支指令
}

void emitAwaitable(Node* node) {
  emitLoadInputs(node->inputs());
  // 发出加载节点输入数据的指令
  auto await_fn = std::make_unique<GraphFunction>(
      "<awaitable function>", node->g(attr::Subgraph), nullptr);
  // 创建一个新的图函数对象，表示等待函数
  awaited_functions_.emplace_back(std::move(await_fn));
  // 将新创建的等待函数对象移动到 awaited_functions_ 向量末尾
  function_table_.emplace_back(awaited_functions_.back().get());
  // 将等待函数对象添加到函数表中
  insertInstruction(
      AWAITABLE, function_table_.size() - 1, node->inputs().size());
  // 在指令列表中插入 AWAITABLE 指令，表示等待指令
}

void emitWarn(Node* node) {
  if (FLAGS_torch_jit_disable_warning_prints) {
    return;
  }
  // 如果禁用了警告打印标志，直接返回

  emitLoadInputs(node->inputs());
  // 发出加载节点输入数据的指令
  int32_t idx = -1;
  if (node->hasAttribute(attr::warn_id)) {
    idx = static_cast<int32_t>(node->i(attr::warn_id));
    // 如果节点有 'warn_id' 属性，则获取其整数值
  }
  insertInstruction(WARN, idx);
  // 在指令列表中插入 WARN 指令，表示发出警告指令
}

void emitEnter(Node* node) {
  emitLoadInputs(node->inputs());
  // 发出加载节点输入数据的指令
  insertInstruction(ENTER);
  // 在指令列表中插入 ENTER 指令，表示进入指令
}

void emitExit(Node* /* node */) {
  insertInstruction(EXIT);
  // 在指令列表中插入 EXIT 指令，表示退出指令
}

void emitNode(Node* node) {
  WithCurrentNode guard(&current_node_, node);
  // 使用当前节点创建一个节点保护器
  }
}

void emitCodeForBlock(Block* block) {
  emitNodeAtBlockLevel(block->param_node());
  // 发出处理块参数节点的指令
  for (auto node : block->nodes()) {
    emitNodeAtBlockLevel(node);
    // 发出处理块中每个节点的指令
  }
  emitNodeAtBlockLevel(block->return_node());
  // 发出处理块返回节点的指令
}

const std::vector<GraphExecutor*>& grad_executors() {
  if (!grad_executors_) {
    grad_executors_.emplace();
    // 如果梯度执行器为空，则创建一个新的梯度执行器
    for (Operation& op : operator_table_) {
      if (auto executor = detail::getGradExecutor(op)) {
        grad_executors_->push_back(executor);
        // 获取每个操作的梯度执行器，并添加到梯度执行器列表中
      }
    }
  }
  return *grad_executors_;
  // 返回梯度执行器列表
}

const std::vector<GraphExecutor*>& diff_graph_op_executors() {
  if (!forward_executors_) {
    forward_executors_.emplace();
    // 如果前向执行器为空，则创建一个新的前向执行器
    for (Operation& op : operator_table_) {
      if (auto executor = detail::getDifferentiableGraphOpExecutor(op)) {
        forward_executors_->push_back(executor);
        // 获取每个操作的可微图操作执行器，并添加到前向执行器列表中
      }
    }
  }
  return *forward_executors_;
  // 返回前向执行器列表
}

void dump(std::ostream& out, size_t i) const {
  out << i << " " << instructions_[i];
  // 将指令索引和指令内容输出到流中
  if (instructions_[i].op == OP || instructions_[i].op == CALL ||
      instructions_[i].op == OPN) {
    out << " # " << *instructions_source_[i];
    // 如果指令是 OP、CALL 或 OPN，则输出指令源信息
  }
}
  } else {
    // 如果条件不成立，向输出流中添加一个换行符
    out << "\n";
  }
}

void dump(std::ostream& out) const {
  // 输出流中打印图的内容
  out << *graph_ << "\n";
  // 遍历指令列表，逐个将指令内容打印到输出流中
  for (const auto i : c10::irange(instructions_.size())) {
    dump(out, i);
  }
}

/**
 * 将操作添加到 operator_table_ 中，如果不是重复操作则返回其索引
 */
int add_to_operator_table(
    const Operator& op,                    // 操作对象
    const Node* node,                      // 节点指针
    const std::string& op_name,            // 操作名称
    const int num_inputs,                  // 输入数量
    const bool is_vararg) {                // 是否可变参数标志

  int size = operator_table_.size();       // 当前 operator_table_ 的大小

  const Operation& oper = op.getOperation(node);  // 获取操作对象的操作信息

  if (!is_vararg) {                       // 如果不是可变参数操作
    std::pair<std::string, int> key(op_name, num_inputs);  // 创建操作名称和输入数量的键值对
    auto found = operator_table_inv_.find(key);  // 在逆映射表中查找该键

    if (found != operator_table_inv_.end()) {
      return found->second;               // 如果找到，则返回其索引
    }

    operator_table_inv_.emplace(key, size);  // 否则将该键值对插入逆映射表中
  }

  operator_table_.emplace_back(oper);      // 将操作信息添加到操作表中
#ifndef NDEBUG
    // 如果处于调试模式，将操作符添加到完整操作符表中
    full_operator_table_.emplace_back(op);
#endif
    // 返回函数的参数 size
    return size;
  }

  inline void assert_stack_size(
      int32_t instruction_index,   // 指令索引
      size_t init_size,            // 初始栈大小
      size_t actual_size) const {  // 实际栈大小
#ifndef NDEBUG
    // 获取指令索引处的操作符的模式
    const auto& schema = full_operator_table_[instruction_index].schema();
    // 计算预期的栈大小
    int64_t expected_size = static_cast<int64_t>(init_size) -
        static_cast<int64_t>(schema.arguments().size()) +
        static_cast<int64_t>(schema.returns().size());
    // 断言实际栈大小与预期大小相等，或者模式支持变长返回或可变参数
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<size_t>(expected_size) == actual_size ||
            schema.is_varret() || schema.is_vararg(),
        "Expected to find ",
        expected_size,
        " values on the stack, but found ",
        actual_size,
        " on the stack after ",
        toString(full_operator_table_[instruction_index].schema()));
#endif
  }
};

struct MobileCodeImpl : CodeImpl {
  MobileCodeImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      bool emit_default_input_instructions,
      bool support_default_args_before_out,
      bool emit_promoted_ops,
      size_t remaining_bailout_depth)
      : CodeImpl(graph, function_name, remaining_bailout_depth, false),
        emit_default_input_instructions_(emit_default_input_instructions),
        support_default_args_before_out_(support_default_args_before_out),
        emit_promoted_ops_(emit_promoted_ops) {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    // 执行移动设备代码初始化流程
    run();
  }

  void run() override {
    // 处理移动设备操作
    process_ops_for_mobile();
    // 为图块生成代码
    emitCodeForBlock(graph_->block());
    // 插入返回指令
    insertInstruction(RET);
    // 推迟生成退出块，确保它们出现在末尾
    // 现在生成它们并修正跳转
    insertBailoutBlocks();
  }

  void process_ops_for_mobile() {
    // 迭代图中的节点，为移动设备处理操作符
    DepthFirstGraphNodeIterator graph_it(graph_);
    Node* node = graph_it.next();
    while (node) {
      if (node->maybeOperator()) {
        auto op_schema = node->getOperator().schema();
        // 如果操作符模式不支持可变参数，计算必要的参数数量
        if (!op_schema.is_vararg()) {
          auto specifiedArgs = CalculateNecessaryArgs(
              op_schema.arguments(),
              node->inputs(),
              support_default_args_before_out_);

          size_t numInclude = specifiedArgs.first +
              (support_default_args_before_out_ ? specifiedArgs.second : 0);
          // 构建唯一的操作符名称
          auto unique_name = !op_schema.overload_name().empty()
              ? op_schema.name() + "." + op_schema.overload_name()
              : op_schema.name();
          // 更新操作符到指定参数数量的映射
          auto it = op_to_num_specified_args_.insert(
              std::pair<std::string, size_t>(unique_name, 0));
          op_to_num_out_args_.insert(std::pair<std::string, size_t>(
              unique_name, specifiedArgs.second));
          auto prev_value = it.first->second;
          it.first->second = std::max(numInclude, prev_value);
        }
      }
      node = graph_it.next();
    }
  }

 private:
  void emitOperator(Node* node) override {
    // 如果需要发出默认输入指令，则调用 CodeImpl 类的 emitOperator 方法
    if (emit_default_input_instructions_) {
      CodeImpl::emitOperator(node);
    } else {
      // 否则，获取节点的运算符信息
      const Operator& op = node->getOperator();
      // 获取运算符的唯一名称并转换为字符串
      std::string unique_op_name = c10::toString(op.schema().operator_name());
      // 获取节点输入的数量
      int num_inputs = node->inputs().size();
      // 检查运算符是否支持可变参数
      bool is_vararg = op.schema().is_vararg();

      // 如果运算符有操作且支持可变参数
      if (op.hasOperation() && is_vararg) {
        // 发出加载输入的指令
        emitLoadInputs(node->inputs());
        // 将运算符添加到运算符表中，并获取操作索引
        int operation_index = add_to_operator_table(
            op,
            node,
            unique_op_name,
            num_inputs,
            /* is_vararg */ true);
        // 插入操作指令(OPN)，指定操作索引和输入数量
        insertInstruction(OPN, operation_index, num_inputs);
      } else {
        // 否则，根据运算符的唯一名称查找预定义参数数量
        auto num_include = num_inputs;
        auto it = op_to_num_specified_args_.find(unique_op_name);
        if (it != op_to_num_specified_args_.end()) {
          num_include = it->second;
        }
        // 如果支持在输出之前使用默认参数
        if (support_default_args_before_out_) {
          // 获取输出参数数量
          auto num_out = op_to_num_out_args_.find(unique_op_name)->second;
          // 计算在输出之前指定的参数数量
          auto num_specified_before_out = num_include - num_out;
          // 发出加载输入的指令，指定加载的起始和结束位置
          emitLoadInputs(node->inputs(), 0, num_specified_before_out);
          emitLoadInputs(
              node->inputs(),
              node->inputs().size() - num_out,
              node->inputs().size());
        } else {
          // 否则，发出加载输入的指令，指定加载的数量
          emitLoadInputs(node->inputs(), num_include);
        }
        // 将运算符添加到运算符表中，并获取操作索引
        int operation_index = add_to_operator_table(
            op, node, unique_op_name, num_inputs, is_vararg);
        // 插入操作指令(OP)，指定操作索引
        insertInstruction(OP, operation_index);
      }
    }
  }

  // 发出运算符或指令，根据 emit_promoted_ops_ 的设置决定使用不同的方法
  void emitOperatorOrInstruction(
      Node* node,
      OpCode op,
      int64_t X = 0,
      uint64_t N = 0,
      bool emit_inputs = true) override {
    if (emit_promoted_ops_) {
      // 如果需要使用推广的操作，则调用父类的 emitOperatorOrInstruction 方法
      CodeImpl::emitOperatorOrInstruction(node, op, X, N, emit_inputs);
    } else {
      // 否则，调用 CodeImpl 类的 emitOperator 方法
      CodeImpl::emitOperator(node);
    }
  }

  // 用于支持从字节码版本 v5 到 v6 的向前兼容性
  bool emit_default_input_instructions_;
  // 用于支持从字节码版本 v6 到 v7 的向前兼容性
  bool support_default_args_before_out_;
  // 用于支持从字节码版本 v7 到 v8 的向前兼容性
  bool emit_promoted_ops_;
};

// 结束 interpreter 命名空间
} // namespace interpreter
// 结束 torch::jit 命名空间
} // namespace torch::jit
```