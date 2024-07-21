# `.\pytorch\torch\csrc\jit\runtime\instruction.cpp`

```
// 引入必要的头文件和命名空间
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <cstring>
#include <iostream>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 定义一个重载操作符 << ，用于将 OpCode 枚举值转换为字符串输出到流中
static std::ostream& operator<<(std::ostream& out, OpCode op) {
  // 使用 switch 语句根据 OpCode 枚举值输出相应的字符串名称
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return out << #x;
    FORALL_OPCODES(OP_STRING) // 展开所有 OpCode 枚举值
#undef OP_STRING
  }
  return out;
}

// 将 OpCode 枚举值转换为 C 风格字符串
char const* toString(OpCode op) {
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return #x;
    FORALL_OPCODES(OP_STRING) // 展开所有 OpCode 枚举值
#undef OP_STRING
  }
  return nullptr;
}

// 返回 OpCode 的信息字符串
static const char* OpInfo(OpCode op) {
  switch (op) {
#define OP_INFO(x, info) \
  case x:                \
    return info;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    FORALL_OPCODES(OP_INFO) // 展开所有 OpCode 的信息
#undef OP_INFO
  }
  return nullptr;
}

// 定义指令的固定大小为 8 字节，并进行静态断言验证
static constexpr size_t instruction_size = 8;
static_assert(
    sizeof(Instruction) == instruction_size,
    "Instructions should be 8 bytes");

// 重载操作符 << ，用于将 Instruction 结构体的内容输出到流中
std::ostream& operator<<(std::ostream& out, Instruction inst) {
  // TODO: 使用操作信息以更用户友好的方式打印操作码
  int nargs = std::strlen(OpInfo(inst.op));
  out << inst.op;
  if (nargs > 0) {
    out << " " << inst.X;
  }
  if (nargs > 1) {
    out << " " << inst.N;
  }
  return out;
}

// 静态 constexpr 数组，存储所有 OpCode 枚举值对应的字符串
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static constexpr const char* strOpCode[] = {
#define STR_OP(x, _) #x,
    FORALL_OPCODES(STR_OP) // 展开所有 OpCode 枚举值
#undef STR_OP
};

// 根据字符串解析出相应的 OpCode 枚举值
OpCode parseOpCode(const char* str) {
  const int n = sizeof(strOpCode) / sizeof(strOpCode[0]);
  for (const auto i : c10::irange(n)) {
    if (strcmp(strOpCode[i], str) == 0)
      return (OpCode)i;
  }
  return OP; // 如果未找到匹配的 OpCode，则返回默认值 OP
}

// 检查给定的 OpCode 是否在移动设备中支持
bool isOpSupportedInMobile(OpCode op) {
  // clang-format off
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static constexpr OpCode supported_ops_in_mobile[] {
      OP, OPN, LOAD, MOVE, STOREN, STORE, DROP, DROPR, LOADC, JF, JMP, LOOP,
      RET, GET_ATTR, SET_ATTR, LIST_CONSTRUCT, TUPLE_CONSTRUCT, WARN,
      INTERFACE_CALL, LIST_UNPACK, TUPLE_SLICE, DICT_CONSTRUCT,
      NAMED_TUPLE_CONSTRUCT, CREATE_OBJECT, ISINSTANCE, CALL,
      RAISE_EXCEPTION, UNCHECKED_CAST, __IS__, UN_INITIALIZED,
      __ISNOT__, FORMAT, DEVICE, DICT_INDEX,
      DTYPE, TUPLE_INDEX, DIM, __NOT__,
      TO_LIST, NUM_TO_TENSOR, IS_CUDA};
  // clang-format on

  // 遍历支持的 OpCode 数组，检查给定的 op 是否在其中
  for (auto sop : supported_ops_in_mobile) {
    if (op == sop)
      return true;
  }
  return false; // 如果不支持，则返回 false
}

} // namespace torch::jit
```