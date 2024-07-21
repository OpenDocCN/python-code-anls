# `.\pytorch\torch\csrc\profiler\unwind\action.h`

```
#pragma once
#include <stdint.h> // 包含标准整数类型定义
#include <ostream> // 包含输出流操作相关的头文件

namespace torch::unwind {

enum {
  A_UNDEFINED = 0x0, // 行动未定义的标志
  A_REG_PLUS_DATA = 0x1, // 行动为寄存器加上数据的表达式：exp = REG[reg] + data0
  A_LOAD_CFA_OFFSET = 0x2, // 行动为加载 CFA 偏移量的表达式：exp = *(cfa + data0)
  A_REG_PLUS_DATA_DEREF = 0x3 // 行动为寄存器加上数据并解引用的表达式：exp = *(REG[reg] + data0)
};

// 在 dwarf 信息中定义的寄存器编号
enum {
  D_UNDEFINED = -1, // 未定义的寄存器编号
  D_RBP = 6, // RBP 寄存器编号
  D_RSP = 7, // RSP 寄存器编号
  D_RIP = 16, // RIP 寄存器编号
  D_REG_SIZE = 17, // 寄存器编号的大小
};

struct Action {
  uint8_t kind = A_UNDEFINED; // 行动的类型，默认为未定义
  int32_t reg = -1; // 相关的寄存器编号，默认为 -1
  int64_t data = 0; // 相关的数据，默认为 0

  // 创建并返回一个未定义的行动对象
  static Action undefined() {
    return Action{A_UNDEFINED};
  }

  // 创建并返回一个寄存器加上数据的行动对象
  static Action regPlusData(int32_t reg, int64_t offset) {
    return Action{A_REG_PLUS_DATA, reg, offset};
  }

  // 创建并返回一个寄存器加上数据并解引用的行动对象
  static Action regPlusDataDeref(int32_t reg, int64_t offset) {
    return Action{A_REG_PLUS_DATA_DEREF, reg, offset};
  }

  // 创建并返回一个加载 CFA 偏移量的行动对象
  static Action loadCfaOffset(int64_t offset) {
    return Action{A_LOAD_CFA_OFFSET, D_UNDEFINED, offset};
  }

  // 自定义输出操作符重载，将行动对象输出到流中
  friend std::ostream& operator<<(std::ostream& out, const Action& self) {
    switch (self.kind) {
      case A_UNDEFINED:
        out << "u"; // 输出未定义行动的符号表示
        break;
      case A_REG_PLUS_DATA:
        out << "r" << (int)self.reg << " + " << self.data; // 输出寄存器加上数据的表达式
        break;
      case A_REG_PLUS_DATA_DEREF:
        out << "*(r" << (int)self.reg << " + " << self.data << ")"; // 输出寄存器加上数据并解引用的表达式
        break;
      case A_LOAD_CFA_OFFSET:
        out << "*(cfa + " << self.data << ")"; // 输出加载 CFA 偏移量的表达式
        break;
    }
    return out;
  }
};

} // namespace torch::unwind
```