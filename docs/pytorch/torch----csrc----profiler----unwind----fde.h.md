# `.\pytorch\torch\csrc\profiler\unwind\fde.h`

```
#pragma once
// 包含头文件，用于引入所需的依赖项
#include <c10/util/irange.h>
#include <torch/csrc/profiler/unwind/action.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <array>            // 包含数组容器类模板
#include <iostream>         // 包含输入输出流类模板
#include <sstream>          // 包含字符串流类模板
#include <vector>           // 包含向量容器类模板

namespace torch::unwind {

// 定义结构体 TableState，用于表示表状态
struct TableState {
  Action cfa;                         // 表状态中的 CFA 动作对象
  std::array<Action, D_REG_SIZE> registers; // 保存寄存器动作对象的数组
  // 友元函数，重载输出流操作符，用于输出表状态对象信息
  friend std::ostream& operator<<(std::ostream& out, const TableState& self) {
    out << "cfa = " << self.cfa << "; ";    // 输出 CFA 的信息
    // 遍历寄存器数组，输出每个非未定义动作的寄存器信息
    for (auto r : c10::irange(self.registers.size())) {
      if (self.registers.at(r).kind != A_UNDEFINED) {
        out << "r" << r << " = " << self.registers.at(r) << "; ";
      }
    }
    return out;
  }
};

// FDE - Frame Description Entry (ELF 规范中的概念)
// 这种格式在 https://www.airs.com/blog/archives/460 中有详细解释
// 不同 DWARF 动作的细节在 https://web.archive.org/web/20221129184704/https://dwarfstd.org/doc/DWARF4.doc 中有说明
// DWARF 无展开工作原理的概述见 https://dl.acm.org/doi/pdf/10.1145/3360572
// Rust 中的类似实现见 https://github.com/mstange/framehop/
template <bool LOG = false>
struct FDE {
  // 构造函数，接收数据指针、库名称和加载偏移量作为参数
  FDE(void* data, const char* library_name, uint64_t load_bias)
      : library_name_(library_name), load_bias_(load_bias) {
    Lexer L(data);                      // 使用数据初始化 Lexer 对象 L
    auto length = L.read4or8Length();  // 读取长度信息
    void* fde_start = L.loc();         // 获取 FDE 起始位置
    void* cie_data = (void*)((int64_t)fde_start - L.read<uint32_t>()); // 计算 CIE 数据位置
    Lexer LC(cie_data);                 // 使用 CIE 数据初始化 Lexer 对象 LC
    auto cie_length = LC.read4or8Length();  // 读取 CIE 长度信息
    void* cie_start = LC.loc();         // 获取 CIE 起始位置
    auto zero = LC.read<uint32_t>();    // 读取零值
    TORCH_INTERNAL_ASSERT(zero == 0, "expected 0 for CIE");  // 断言零值应为 0，用于验证 CIE 是否正确
    auto version = LC.read<uint8_t>();  // 读取版本号
    TORCH_INTERNAL_ASSERT(
        version == 1 || version == 3, "non-1 version for CIE"); // 断言版本号应为 1 或 3
    augmentation_string_ = LC.readCString();  // 读取增强字符串
    if (hasAugmentation("eh")) {
      throw UnwindError("unsupported 'eh' augmentation string");  // 如果有 "eh" 增强字符串，抛出异常
    }
    code_alignment_factor_ = LC.readULEB128();  // 读取代码对齐因子
    data_alignment_factor_ = LC.readSLEB128();  // 读取数据对齐因子
    if (version == 1) {
      ra_register_ = LC.read<uint8_t>();  // 读取返回地址寄存器编号（仅限于版本 1）
    } else {
      ra_register_ = LC.readULEB128();    // 读取返回地址寄存器编号（适用于版本 3 及以上）
    }
    // 在状态中假设这个值
    TORCH_INTERNAL_ASSERT(ra_register_ == 16, "unexpected number of registers");  // 断言返回地址寄存器编号应为 16，用于验证寄存器数量是否正确
    if (augmentation_string_ && *augmentation_string_ == 'z') {
      augmentation_length_ = LC.readULEB128();  // 如果增强字符串以 'z' 开头，读取增强长度
      Lexer A(LC.loc());                      // 使用位置初始化 Lexer 对象 A
      for (auto ap = augmentation_string_ + 1; *ap; ap++) {
        switch (*ap) {
          case 'L':
            lsda_enc = A.read<uint8_t>();   // 读取 LSDA 编码
            break;
          case 'R':
            fde_enc = A.read<uint8_t>();    // 读取 FDE 编码
            break;
          case 'P': {
            uint8_t personality_enc = A.read<uint8_t>();  // 读取人格特征编码
            A.readEncoded(personality_enc);              // 解析编码内容
          } break;
          case 'S': {
            // 信号处理程序
          } break;
          default: {
            throw UnwindError("unknown augmentation string");  // 未知增强字符串，抛出异常
          } break;
        }
      }
    }
  // 跳过给定长度的指令
  LC.skip(augmentation_length_);

  // 从FDE编码中读取低地址点的值
  low_pc_ = L.readEncoded(fde_enc);

  // 计算高地址点的值
  high_pc_ = low_pc_ + L.readEncodedValue(fde_enc);

  // 如果有扩展信息包含 'z'，则读取扩展长度
  if (hasAugmentation("z")) {
    augmentation_length_fde_ = L.readULEB128();
  }

  // 读取LSDA的编码值，使用默认值0
  L.readEncodedOr(lsda_enc, 0);

  // 记录当前CIE的开始位置
  cie_begin_ = LC.loc();

  // 记录当前FDE的开始位置
  fde_begin_ = L.loc();

  // 计算CIE的结束位置
  cie_end_ = (void*)((const char*)cie_start + cie_length);

  // 计算FDE的结束位置
  fde_end_ = (void*)((const char*)fde_start + length);
}

// OP代码的具体实现

// 向前移动当前位置，按照给定的数量
void advance_raw(int64_t amount) {
  auto previous_pc = current_pc_;
  current_pc_ += amount;
  if (LOG) {
    (*out_) << (void*)(previous_pc - load_bias_) << "-"
            << (void*)(current_pc_ - load_bias_) << ": " << state() << "\n";
  }
}

// 根据当前的代码对齐因子，向前移动位置
void advance_loc(int64_t amount) {
  if (LOG) {
    (*out_) << "advance_loc " << amount << "\n";
  }
  advance_raw(amount * code_alignment_factor_);
}

// 设置指定寄存器的偏移量
void offset(int64_t reg, int64_t offset) {
  if (LOG) {
    (*out_) << "offset " << reg << " " << offset << "\n";
  }
  if (reg > (int64_t)state().registers.size()) {
    if (LOG) {
      (*out_) << "OFFSET OF BIG REGISTER " << reg << "ignored...\n";
    }
    return;
  }
  state().registers.at(reg) =
      Action{A_LOAD_CFA_OFFSET, -1, offset * data_alignment_factor_};
}

// 恢复指定寄存器的初始状态
void restore(int64_t reg) {
  if (LOG) {
    (*out_) << "restore " << reg << "\n";
  }
  if (reg > (int64_t)state().registers.size()) {
    if (LOG) {
      (*out_) << "RESTORE OF BIG REGISTER " << reg << "ignored...\n";
    }
    return;
  }
  state().registers.at(reg) = initial_state_.registers.at(reg);
}

// 定义CFA操作，设置寄存器和偏移量
void def_cfa(int64_t reg, int64_t off) {
  if (LOG) {
    (*out_) << "def_cfa " << reg << " " << off << "\n";
  }
  last_reg_ = reg;
  last_offset_ = off;
  state().cfa = Action::regPlusData(reg, off);
}

// 定义CFA操作，只设置寄存器
void def_cfa_register(int64_t reg) {
  def_cfa(reg, last_offset_);
}

// 定义CFA操作，只设置偏移量
void def_cfa_offset(int64_t off) {
  def_cfa(last_reg_, off);
}

// 记录当前状态到状态堆栈
void remember_state() {
  if (LOG) {
    (*out_) << "remember_state\n";
  }
  state_stack_.push_back(state());
}

// 从状态堆栈中恢复状态
void restore_state() {
  if (LOG) {
    (*out_) << "restore_state\n";
  }
  state_stack_.pop_back();
}

// 标记指定寄存器为未定义状态
void undefined(int64_t reg) {
  if (LOG) {
    (*out_) << "undefined " << reg << "\n";
  }
  state().registers.at(reg) = Action::undefined();
}

// 设置指定寄存器的内容，与另一寄存器相同
void register_(int64_t reg, int64_t rhs_reg) {
  if (LOG) {
    (*out_) << "register " << reg << " " << rhs_reg << "\n";
  }
  state().registers.at(reg) = Action::regPlusData(reg, 0);
}

// 返回当前状态堆栈的顶部状态
TableState& state() {
  return state_stack_.back();
}

// 将状态信息输出到指定输出流
void dump(std::ostream& out) {
  out_ = &out;
    // 输出对象的信息，包括增强字符串、偏移后的低地址、偏移后的高地址、代码对齐因子、数据对齐因子、RA 寄存器等
    out << "FDE(augmentation_string=" << augmentation_string_
        << ", low_pc=" << (void*)(low_pc_ - load_bias_)
        << ", high_pc=" << (void*)(high_pc_ - load_bias_)
        << ", code_alignment_factor=" << code_alignment_factor_
        << ", data_alignment_factor=" << data_alignment_factor_
        << ", ra_register_=" << ra_register_ << ")\n";
    // 读取指定地址范围内的信息
    readUpTo(high_pc_);
    // 将输出流指向标准输出
    out_ = &std::cout;
  }

  TableState readUpTo(uint64_t addr) {
    // 如果地址不在有效范围内，则抛出异常
    if (addr < low_pc_ || addr > high_pc_) {
      throw UnwindError("Address not in range");
    }
    // 如果开启了日志记录，则输出读取信息的地址和库名
    if (LOG) {
      (*out_) << "readUpTo " << (void*)addr << " for " << library_name_
              << " at " << (void*)load_bias_ << "\n";
    }
    // 将当前状态压入状态栈
    state_stack_.emplace_back();
    current_pc_ = low_pc_;
    // 解析指令...
    Lexer LC(cie_begin_);
    // 循环读取指令，直到地址超过指定范围或者遍历完指令数据
    while (LC.loc() < cie_end_ && current_pc_ <= addr) {
      readInstruction(LC);
    }
    // 如果当前指令地址超过了指定地址，则返回当前状态
    if (current_pc_ > addr) {
      return state();
    }

    initial_state_ = state_stack_.back();

    // 如果开启了日志记录，则输出分隔线
    if (LOG) {
      (*out_) << "--\n";
    }

    // 从指令数据中解析 FDE 数据
    Lexer L(fde_begin_);
    // 循环读取指令，直到地址超过指定范围或者遍历完指令数据
    while (L.loc() < fde_end_ && current_pc_ <= addr) {
      readInstruction(L);
    }
    // 为了在调试时打印完整范围，确保当前指令地址小于等于指定地址
    if (current_pc_ <= addr) {
      advance_raw(addr - current_pc_);
    }
    // 返回当前状态
    return state();
  }

  // 打印地址转换信息
  void dumpAddr2Line() {
    std::cout << "addr2line -f -e " << library_name_ << " "
              << (void*)(low_pc_ - load_bias_) << "\n";
  }

  // 读取指令并解析操作码和低位数据
  void readInstruction(Lexer& L) {
    uint8_t bc = L.read<uint8_t>();
    auto op = bc >> 6;
    auto lowbits = bc & 0x3F;
    // ...
  }
};

} // namespace torch::unwind


// 结束了命名空间 torch::unwind 的定义
};
// 结束了命名空间 torch::unwind
```