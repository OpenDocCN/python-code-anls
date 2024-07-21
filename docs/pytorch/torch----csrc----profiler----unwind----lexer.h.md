# `.\pytorch\torch\csrc\profiler\unwind\lexer.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <cstdint>
// 包含 C++ 标准中定义的整数类型，如 int64_t, uint8_t 等

#include <cstring>
// 包含 C 标准库中字符串操作函数的声明，如 strlen, memcpy 等

#include <utility>
// 包含 C++ 标准库中定义的一些通用模板类和函数

#include <torch/csrc/profiler/unwind/dwarf_enums.h>
// 包含 Torch 库中用于分析器的枚举定义，如 DW_EH_PE_absptr 等

#include <torch/csrc/profiler/unwind/unwind_error.h>
// 包含 Torch 库中定义的用于异常处理的错误类 UnwindError

namespace torch::unwind {

template <bool checked>
// 模板定义，根据 checked 参数决定是否进行边界检查
struct LexerImpl {
  LexerImpl(void* data, void* base = nullptr, void* end = nullptr)
      : next_((const char*)data),
        base_((int64_t)base),
        end_((const char*)end) {}
  // 构造函数，初始化 LexerImpl 结构体

  template <typename T>
  T read() {
    // 读取 T 类型数据，从 next_ 指向的位置开始
    T result;
    auto end = next_ + sizeof(T);
    UNWIND_CHECK(
        !checked || end <= end_,
        "read out of bounds {} >= {}",
        (void*)end,
        (void*)end_);
    // 如果开启了 checked 模式，则检查是否超出边界
    memcpy(&result, next_, sizeof(T));
    next_ = end;
    return result;
    // 返回读取的结果并移动 next_ 指针
  }

  // SLEB/ULEB code adapted from LLVM equivalents
  int64_t readSLEB128() {
    // 读取 SLEB128 格式的整数
    int64_t Value = 0;
    unsigned Shift = 0;
    uint8_t Byte = 0;
    do {
      Byte = read<uint8_t>();
      uint64_t Slice = Byte & 0x7f;
      if ((Shift >= 64 && Slice != (Value < 0 ? 0x7f : 0x00)) ||
          (Shift == 63 && Slice != 0 && Slice != 0x7f)) {
        throw UnwindError("sleb128 too big for int64");
      }
      Value |= int64_t(Slice << Shift);
      Shift += 7;
    } while (Byte >= 128);
    // 根据需要进行负数的符号扩展
    if (Shift < 64 && (Byte & 0x40)) {
      Value |= int64_t((-1ULL) << Shift);
    }
    return Value;
    // 返回读取的 SLEB128 整数值
  }

  uint64_t readULEB128() {
    // 读取 ULEB128 格式的整数
    uint64_t Value = 0;
    unsigned Shift = 0;
    uint8_t p = 0;
    do {
      p = read<uint8_t>();
      uint64_t Slice = p & 0x7f;
      if ((Shift >= 64 && Slice != 0) || Slice << Shift >> Shift != Slice) {
        throw UnwindError("uleb128 too big for uint64");
      }
      Value += Slice << Shift;
      Shift += 7;
    } while (p >= 128);
    // 返回读取的 ULEB128 整数值
    return Value;
  }

  const char* readCString() {
    // 读取以 null 结尾的 C 字符串
    auto result = next_;
    if (!checked) {
      next_ += strlen(next_) + 1;
      return result;
    }
    while (next_ < end_) {
      if (*next_++ == '\0') {
        return result;
      }
    }
    // 检查是否超出边界
    UNWIND_CHECK(
        false, "string is out of bounds {} >= {}", (void*)next_, (void*)end_);
  }

  int64_t readEncoded(uint8_t enc) {
    // 根据指定的编码方式读取整数值
    int64_t r = 0;
    switch (enc & (~DW_EH_PE_indirect & 0xF0)) {
      case DW_EH_PE_absptr:
        break;
      case DW_EH_PE_pcrel:
        r = (int64_t)next_;
        break;
      case DW_EH_PE_datarel:
        r = base_;
        break;
      default:
        throw UnwindError("unknown encoding");
    }
    return r + readEncodedValue(enc);
    // 返回读取的编码值
  }

  int64_t readEncodedOr(uint8_t enc, int64_t orelse) {
    // 如果编码为 DW_EH_PE_omit，则返回指定的 orelse 值，否则读取编码值
    if (enc == DW_EH_PE_omit) {
      return orelse;
    }
    return readEncoded(enc);
    // 返回读取的编码值
  }

  int64_t read4or8Length() {
    // 根据读取到的长度类型，返回 4 字节或 8 字节长度
    return readSectionLength().first;
    // 返回读取到的段长度
  }

  std::pair<int64_t, bool> readSectionLength() {
    // 读取段的长度及是否为 8 字节的标志
    int64_t length = read<uint32_t>();
    if (length == 0xFFFFFFFF) {
      return std::make_pair(read<int64_t>(), true);
    }
    return std::make_pair(length, false);
    // 返回读取到的段长度及标志信息
  }

  void* loc() const {
    // 返回当前位置指针 next_
    return (void*)next_;
    // 返回当前读取位置的指针
  }

  const char* next_;
  // 下一个要读取的位置指针

  int64_t base_;
  // 基本地址偏移量

  const char* end_;
  // 结束位置指针
};
// LexerImpl 结构体定义结束

} // namespace torch::unwind
// Torch unwind 命名空间结束
  // 返回一个指向下一个地址的 void* 指针
  return (void*)next_;
}
// 跳过指定字节数，并返回当前 LexerImpl 对象的引用
LexerImpl& skip(int64_t bytes) {
  // 增加 next_ 指针以跳过指定的字节数
  next_ += bytes;
  return *this;
}

// 根据编码方式读取并返回相应的编码值
int64_t readEncodedValue(uint8_t enc) {
  switch (enc & 0xF) {
    case DW_EH_PE_udata2:
      // 读取并返回一个 uint16_t 类型的值
      return read<uint16_t>();
    case DW_EH_PE_sdata2:
      // 读取并返回一个 int16_t 类型的值
      return read<int16_t>();
    case DW_EH_PE_udata4:
      // 读取并返回一个 uint32_t 类型的值
      return read<uint32_t>();
    case DW_EH_PE_sdata4:
      // 读取并返回一个 int32_t 类型的值
      return read<int32_t>();
    case DW_EH_PE_udata8:
      // 读取并返回一个 uint64_t 类型的值
      return read<uint64_t>();
    case DW_EH_PE_sdata8:
      // 读取并返回一个 int64_t 类型的值
      return read<int64_t>();
    case DW_EH_PE_uleb128:
      // 读取并返回一个无符号 LEB128 编码的值
      return readULEB128();
    case DW_EH_PE_sleb128:
      // 读取并返回一个有符号 LEB128 编码的值
      return readSLEB128();
    default:
      // 抛出异常，表示未实现的编码类型
      throw UnwindError("not implemented");
  }
}

private:
const char* next_;
int64_t base_;
const char* end_;
// 结束了一个命名空间的定义
};

// 使用 CheckedLexer 类型别名代替 LexerImpl<true>
using CheckedLexer = LexerImpl<true>;
// 使用 Lexer 类型别名代替 LexerImpl<false>
using Lexer = LexerImpl<false>;

// 命名空间结束，命名空间为 torch::unwind
} // namespace torch::unwind
```