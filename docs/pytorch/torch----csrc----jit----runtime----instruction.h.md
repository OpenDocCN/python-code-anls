# `.\pytorch\torch\csrc\jit\runtime\instruction.h`

```py
#pragma once

#include <cstdint>
#include <typeinfo>
#include <unordered_set>

namespace torch::jit {
// 定义了所有的操作码及其含义
// 每个操作码都包含两部分：标识符和参数说明
#define FORALL_OPCODES(_)                                                      \
  _(OP, "O") /* 调用操作符 X */                                                \
  _(OPN, "OI") /* 调用可变参数操作符 X，参数个数为 N */                        \
  _(LOAD, "R") /* 从寄存器 X 中推送一个值到栈顶 */                              \
  _(MOVE, "R") /* 从寄存器 X 中推送一个值到栈顶，并清空寄存器 */                  \
  _(STOREN, "RI") /* 将 N 个值存储到寄存器 [X, X+N) */                          \
  _(STORE, "R") /* 将一个值存储到寄存器 X */                                     \
  _(DROP, "") /* 从栈顶弹出一个值 */                                            \
  _(DROPR, "R") /* 清空寄存器 X */                                              \
  _(LOADC, "C") /* 将常量 X 推送到栈顶 */                                       \
  _(JF, "P") /* 弹出栈顶的值，如果为假，则跳转到 P 处 */                         \
  _(JMP, "P") /* 无条件跳转到 X 处 */                                           \
  _(LOOP, "PI") /* 执行循环，如果条件为假，则跳转到 X 处 */                       \
  _(RET, "") /* 退出执行 */                                                    \
  _(WAIT, "") /* 等待未来的完成 */                                              \
  _(CALL, "F") /* 调用函数 X */                                                 \
  _(GUARD, "T") /* 根据类型表检查保护条件，如果通过则为真 */                     \
  _(TYPECHECK, "TN") /* 检查输入的每种类型是否与类型表中的 X+N 匹配 */            \
  _(FAIL_GUARD, "T") /* 失败的保护条件，回到 GUARD 处 */                         \
  _(PROFILE_OP, "F") /* 在 profile_function_table 中获取回调函数 X */           \
  _(TAIL_CALL, "F") /* 使用函数 F 替换当前帧 */                                 \
  _(INTERFACE_CALL, "CI") /* 在第一个参数（共 N 个参数）上调用方法 X */          \
  _(GET_ATTR, "S") /* 从对象的槽 X 中获取属性 */                                 \
  _(SET_ATTR, "S") /* 将属性设置到对象的槽 X 中 */                                \
  _(LIST_UNPACK, "I") /* 展开列表，期望长度为 I */                               \
  _(TUPLE_CONSTRUCT, "I") /* 使用 X 个输入构造元组 */                            \
  _(NAMED_TUPLE_CONSTRUCT,                                                     \
    "TI") /* 构造一个元组，使用N个输入的类型X */                    \
  _(LIST_CONSTRUCT, "TI") /* 构造一个列表，使用N个输入的类型X */     \
  _(DICT_CONSTRUCT, "TI") /* 构造一个字典，使用N个输入的类型X */     \
  _(CREATE_OBJECT, "T") /* 创建一个类型为X的对象 */                       \
  _(ISINSTANCE, "TI") /* 检查对象是否为类型[X:X+N]中的一个 */              \
  _(TUPLE_SLICE, "II") /* 对元组进行切片操作，从索引X到(X+N) */                                \
  _(TUPLE_INDEX, "") /* 获取元组中指定索引位置的值 */            \
  _(RAISE_EXCEPTION, "") /* 抛出 Python 异常 */                \
  _(DICT_INDEX, "") /* 获取字典中指定键对应的值 */           \
  _(UNCHECKED_CAST, "") /* 执行一个未检查的类型转换操作 */              \
  _(__IS__, "") /* 执行 Python 中的 `is` 操作符 */                       \
  _(UN_INITIALIZED,                                                            \
    "") /* 给未初始化的变量设置默认值 */          \
  _(__ISNOT__, "") /* 执行 Python 中的 `is not` 操作符  */               \
  _(FORMAT, "I") /* 执行字符串格式化函数 `f 字符串` 或 `{}.format`，
                     输入的数量存储在X中 */                    \
  _(DEVICE, "") /* 调用aten::device以设置Tensor的设备类型 */                        \
  _(DTYPE, "") /* 调用aten::dtype以获取Tensor的数据类型 */                          \
  _(DIM, "") /* 调用aten::dim以获取Tensor的维度 */                              \
  _(__NOT__, "") /* 执行 Python 中的 `not` 操作符  */                    \
  _(TO_LIST, "") /* 将输入转换为列表 */                             \
  _(NUM_TO_TENSOR,                                                             \
    "") /* 将数字/标量转换为Tensor */             \
  _(IS_CUDA, "") /* 调用aten::is_cuda以检查Tensor是否在CUDA设备上 */                      \
  _(FORK, "CN") /* 启动一个线程来运行代码入口x，并使用N个输入参数 */       \
  _(WARN, "I") /* 发出带有行信息的警告 */                      \
  _(ENTER, "EN") /* 进入上下文管理器的作用域 */                         \
  _(EXIT, "EX") /* 退出最后进入的上下文管理器的作用域 */                     \
  _(AWAITABLE, "CN") /* 初始化await，用于代码入口x，并使用N个输入参数 */
// 定义枚举类型 OpCode，使用 uint8_t 作为底层类型
enum OpCode : uint8_t {
  // 宏展开，定义所有操作码
#define DEFINE_OP(op, _) op,
  FORALL_OPCODES(DEFINE_OP)
#undef DEFINE_OP
};

// 定义指令结构体 Instruction
struct Instruction {
  OpCode op;       // 操作码
  uint8_t unused;  // 未使用的字段
  uint16_t N;      // N 参数
  int32_t X;       // X 参数
  // TODO: 检查是否会发生溢出
  // 构造函数，初始化指令对象
  Instruction(OpCode op, int32_t X, uint16_t N)
      : op(op), unused(0), N(N), X(X) {}
};

// 重载流输出操作符，用于将指令对象输出到流中
std::ostream& operator<<(std::ostream& out, Instruction inst);

// 检查给定操作码在移动设备上是否支持的函数声明
bool isOpSupportedInMobile(OpCode op);

// 将操作码转换为字符串表示的函数声明
char const* toString(OpCode op);

// 解析字符串表示的操作码为 OpCode 类型的函数声明
OpCode parseOpCode(const char* str);

// 重载流输出操作符，用于将指令对象输出到流中的函数声明
std::ostream& operator<<(std::ostream& out, Instruction inst);

// 命名空间结束符号，属于 torch::jit 命名空间
} // namespace torch::jit
```