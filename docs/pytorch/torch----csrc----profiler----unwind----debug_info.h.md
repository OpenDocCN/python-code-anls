# `.\pytorch\torch\csrc\profiler\unwind\debug_info.h`

```
#pragma once
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>
#include <optional>

namespace torch::unwind {

// 调试信息结构体，用于解析调试信息
struct DebugInfo {
  // 构造函数，初始化解析调试信息所需的 Sections 对象引用
  DebugInfo(Sections& s) : s_(s) {}

  // 解析调试信息，从给定偏移量开始
  void parse(uint64_t offset) {
    // 解析头部信息，并返回解析后的 lexer 对象
    auto L = parseHeader(offset);
    // 解析编译单元信息
    parseCompileUnit(L);
  }

  // 获取行号程序的偏移量
  unwind::optional<uint64_t> lineNumberProgramOffset() {
    return line_number_program_offset_;
  }

  // 返回下一个偏移量
  uint64_t nextOffset() {
    return end_ - s_.debug_info.data;
  }

  // 返回地址范围列表
  std::vector<std::pair<uint64_t, uint64_t>> ranges() {
    if (range_ptr_) {
      auto offset = range_ptr_->first;
      // 处理 DW_FORM_rnglistx 类型的范围列表
      if (range_ptr_->second == DW_FORM_rnglistx) {
        UNWIND_CHECK(rnglists_base_, "rnglistx but not rnglists_base_ set");
        LOG_INFO("index for rnglistx {:x} + {:x}\n", *rnglists_base_, offset);
        // 使用 debug_rnglists 中的 lexer 读取段偏移量并返回
        CheckedLexer L = s_.debug_rnglists.lexer(
            *rnglists_base_ + offset * sec_offset_size_);
        auto read = readSegmentOffset(L);
        offset = *rnglists_base_ + read;
      }
      // 根据 DWARF 版本读取地址范围
      return version_ == 4 ? readRanges4(offset) : readRanges5(offset);
    }
    // 如果没有高地址指针，则返回空列表
    if (!highpc_) {
      return {};
    }
    // 返回包含低地址和高地址的范围
    return {{lowpc_, lowpc_ + *highpc_}};
  }

  // 判断是否为64位架构
  bool is64bit() {
    return is_64bit_;
  }

 private:
  // 解析头部信息，返回解析后的 lexer 对象
  CheckedLexer parseHeader(uint64_t offset) {
    offset_ = offset;
    // 使用 debug_info 中的 lexer 解析给定偏移量处的信息
    CheckedLexer L = s_.debug_info.lexer(offset_);
    // 读取段长度和是否为64位信息
    std::tie(length_, is_64bit_) = L.readSectionLength();
    sec_offset_size_ = is_64bit_ ? 8 : 4;
    end_ = (const char*)L.loc() + length_;
    // 读取版本号
    version_ = L.read<uint16_t>();
    UNWIND_CHECK(
        version_ == 5 || version_ == 4,
        "unexpected dwarf version {}",
        version_);
    uint8_t address_size = 0;
    // 根据版本不同读取不同的头部信息
    if (version_ == 5) {
      auto unit_type = L.read<uint8_t>();
      UNWIND_CHECK(unit_type == 0x1, "unexpected unit type {}", unit_type);
      address_size = L.read<uint8_t>();
      debug_abbrev_offset_ =
          is_64bit_ ? L.read<uint64_t>() : L.read<uint32_t>();
    } else {
      debug_abbrev_offset_ =
          is_64bit_ ? L.read<uint64_t>() : L.read<uint32_t>();
      address_size = L.read<uint8_t>();
    }
    // 记录调试信息的头部解析结果
    LOG_INFO(
        "compilation unit at offset {:x} with length {:x} and debug_abbrev_offset {:x}\n",
        offset,
        length_,
        debug_abbrev_offset_);
    UNWIND_CHECK(
        address_size == 8,
        "expected 64-bit dwarf but found address size {}",
        address_size);
    return L;
  }

  // 使用 lexer 读取段偏移量
  uint64_t readSegmentOffset(CheckedLexer& L) {
    return s_.readSegmentOffset(L, is_64bit_);
  }

  // 使用 lexer 读取编码值
  uint64_t readEncoded(CheckedLexer& L, uint64_t encoding) {
  // 根据编码类型进行不同的数据读取操作
  switch (encoding) {
    case DW_FORM_data8:
    case DW_FORM_addr:
      // 读取并返回一个 uint64_t 类型的数据
      return L.read<uint64_t>();
    case DW_FORM_data4:
      // 读取并返回一个 uint32_t 类型的数据
      return L.read<uint32_t>();
    case DW_FORM_addrx: {
      // 读取一个 ULEB128 编码的索引值
      auto idx = L.readULEB128();
      // 计算地址的实际偏移量并读取 uint64_t 类型的数据
      return s_.debug_addr.lexer(address_base_ + sizeof(uint64_t) * idx)
          .read<uint64_t>();
    }
    case DW_FORM_sec_offset:
      // 调用 readSegmentOffset 函数读取并返回一个偏移量值
      return readSegmentOffset(L);
    case DW_FORM_rnglistx: {
      // 读取一个 ULEB128 编码的索引值并返回
      return L.readULEB128();
    }
    default:
      // 如果出现未预期的编码类型，抛出异常信息
      UNWIND_CHECK(false, "unexpected encoding");
  }
}

// 解析编译单元中的信息
void parseCompileUnit(CheckedLexer& L) {
  // 读取 ULEB128 编码的条目值
  auto entry = L.readULEB128();
  // 根据偏移量查找相应的缩写条目
  auto A = findAbbrev(debug_abbrev_offset_, entry);
  // 循环处理所有的属性和表单
  while (true) {
    // 读取属性和表单的 ULEB128 编码值
    auto attr = A.readULEB128();
    auto form = A.readULEB128();
    // 如果属性值和表单值都为零，跳出循环
    if (attr == 0 && form == 0) {
      break;
    }
    // 如果表单表示隐式常量，则读取并忽略 SLEB128 编码值
    if (form == DW_FORM_implicit_const) {
      A.readSLEB128();
    }
    // 根据属性类型进行不同的处理
    if (attr == DW_AT_low_pc) {
      // 读取编码后的值并存储为低地址
      lowpc_ = readEncoded(L, form);
      LOG_INFO("  lowpc {:x}\n", lowpc_);
    } else if (attr == DW_AT_high_pc) {
      // 读取编码后的值并存储为高地址，并清空范围指针
      highpc_ = readEncoded(L, form);
      range_ptr_ = std::nullopt;
      LOG_INFO("  highpc {:x}\n", *highpc_);
    } else if (attr == DW_AT_addr_base) {
      // 如果属性为地址基础，则读取段偏移量，并存储为地址基础值
      UNWIND_CHECK(form == DW_FORM_sec_offset, "unexpected addr_base form");
      address_base_ = readSegmentOffset(L);
      LOG_INFO("  address base {:x}\n", address_base_);
    } else if (attr == DW_AT_rnglists_base) {
      // 如果属性为范围列表基础，则读取段偏移量，并存储为范围列表基础值
      UNWIND_CHECK(
          form == DW_FORM_sec_offset, "unexpected rnglists_base form");
      rnglists_base_ = readSegmentOffset(L);
      LOG_INFO("  range base {:x}\n", *rnglists_base_);
    } else if (form == DW_FORM_string) {
      // 如果表单为字符串类型，则读取并忽略字符串
      L.readCString();
    } else if (attr == DW_AT_stmt_list) {
      // 如果属性为语句列表，则读取段偏移量，并记录程序表的偏移量
      UNWIND_CHECK(form == DW_FORM_sec_offset, "unexpected stmt_list form");
      LOG_INFO("  program table offset {:x}\n", *line_number_program_offset_);
      line_number_program_offset_ = readSegmentOffset(L);
    } else if (form == DW_FORM_exprloc) {
      // 如果表单为表达式位置，则读取并忽略 ULEB128 编码的大小值，然后跳过相应长度的数据
      auto sz = L.readULEB128();
      L.skip(int64_t(sz));
    } else if (form == DW_FORM_block1) {
      // 如果表单为块数据（1字节块），则读取并忽略一个字节的大小值，然后跳过相应长度的数据
      auto sz = L.read<uint8_t>();
      L.skip(int64_t(sz));
    } else if (attr == DW_AT_ranges) {
      // 如果属性为地址范围，则读取编码后的值，并存储为范围指针的值和编码类型
      auto range_offset = readEncoded(L, form);
      LOG_INFO("setting range_ptr to {:x} {:x}\n", range_offset, form);
      range_ptr_.emplace(range_offset, form);
    } else if (
        // 处理多个可能的表单类型，直接读取并忽略 ULEB128 编码的值
        form == DW_FORM_udata || form == DW_FORM_rnglistx ||
        form == DW_FORM_strx || form == DW_FORM_loclistx ||
        form == DW_FORM_addrx) {
      L.readULEB128();
    } else if (form == DW_FORM_sdata) {
      // 如果表单为有符号数据类型，则读取并忽略 SLEB128 编码的值
      L.readSLEB128();
    } else {
      // 对于不支持的表单类型，根据其大小和段偏移量的大小进行跳过相应长度的数据
      auto sz = formSize(form, sec_offset_size_);
      UNWIND_CHECK(sz, "unsupported form in compilation unit {:x}", form);
      L.skip(int64_t(*sz));
    }
  }
}
    // 使用给定的偏移量创建一个 CheckedLexer 对象，用于调试范围的词法分析
    CheckedLexer L = s_.debug_ranges.lexer(offset);
    // 初始化存储范围的空向量
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    // 设置基础地址为 lowpc_
    uint64_t base = lowpc_;
    // 进入循环，处理调试范围的数据
    while (true) {
      // 从 L 中读取一个 uint64_t 类型的起始值
      auto start = L.read<uint64_t>();
      // 从 L 中读取一个 uint64_t 类型的结束值
      auto end = L.read<uint64_t>();
      // 如果读取到的 start 和 end 都为 0，则跳出循环
      if (start == 0 && end == 0) {
        break;
      }
      // 如果 start 是 uint64_t 类型的最大值，则将 base 设置为 end
      if (start == std::numeric_limits<uint64_t>::max()) {
        base = end;
      } else {
        // 否则将起始值和结束值添加到 ranges 中，基于 base 进行偏移
        ranges.emplace_back(base + start, base + end);
      }
    }
    // 返回存储范围的向量
    return ranges;
  }

  // 根据给定的偏移量创建一个 CheckedLexer 对象，用于读取调试范围列表
  std::vector<std::pair<uint64_t, uint64_t>> readRanges5(uint64_t offset) {
    // 使用 debug_rnglists 内的偏移量创建 CheckedLexer 对象 L
    CheckedLexer L = s_.debug_rnglists.lexer(offset);
    // 初始化基础地址为 0
    uint64_t base = 0;
    // 打印调试信息，显示开始读取范围信息的位置
    LOG_INFO("BEGIN RANGES {:x}\n", offset);
    // 初始化存储范围的空向量
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    // 进入循环，处理调试范围列表的数据
    while (true) {
      // 从 L 中读取一个 uint8_t 类型的操作码
      auto op = L.read<uint8_t>();
      // 根据操作码进行不同的处理
      switch (op) {
        // 如果操作码是结束列表符号 DW_RLE_end_of_list
        case DW_RLE_end_of_list:
          // 打印调试信息，显示结束读取范围信息
          LOG_INFO("END RANGES\n");
          // 返回存储范围的向量
          return ranges;
        // 如果操作码是基地址扩展符号 DW_RLE_base_addressx
        case DW_RLE_base_addressx: {
          // 使用 readEncoded 函数读取地址表达式 DW_FORM_addrx 的值作为新的基础地址 base
          base = readEncoded(L, DW_FORM_addrx);
          // 打印调试信息，显示设置了新的基础地址
          LOG_INFO("BASE ADDRX {:x}\n", base);
        } break;
        // 如果操作码是起始地址和长度对符号 DW_RLE_startx_length
        case DW_RLE_startx_length: {
          // 使用 readEncoded 函数读取地址表达式 DW_FORM_addrx 的值作为起始地址 s
          auto s = readEncoded(L, DW_FORM_addrx);
          // 从 L 中读取一个 ULEB128 类型的长度值 e
          auto e = L.readULEB128();
          // 打印调试信息，显示起始地址 s 和长度 e 的值
          LOG_INFO("startx_length {:x} {:x}\n", s, e);
          // 将起始地址 s 和 s+e 作为范围添加到 ranges 中
          ranges.emplace_back(s, s + e);
        } break;
        // 如果操作码是基地址扩展符号 DW_RLE_base_address
        case DW_RLE_base_address:
          // 直接从 L 中读取一个 uint64_t 类型的值作为新的基础地址 base
          base = L.read<uint64_t>();
          // 打印调试信息，显示设置了新的基础地址
          LOG_INFO("BASE ADDR {:x}\n", base);
          break;
        // 如果操作码是偏移对符号 DW_RLE_offset_pair
        case DW_RLE_offset_pair: {
          // 从 L 中读取两个 ULEB128 类型的值 s 和 e 作为偏移对的起始和结束偏移
          auto s = L.readULEB128();
          auto e = L.readULEB128();
          // 打印调试信息，显示偏移对的起始偏移 s 和结束偏移 e 的值
          LOG_INFO("offset_pair {:x} {:x}\n", s, e);
          // 将 base+s 和 base+e 作为范围添加到 ranges 中
          ranges.emplace_back(base + s, base + e);
        } break;
        // 如果操作码是起始地址和长度符号 DW_RLE_start_length
        case DW_RLE_start_length: {
          // 从 L 中分别读取一个 uint64_t 类型的起始地址 s 和一个 ULEB128 类型的长度 e
          auto s = L.read<uint64_t>();
          auto e = L.readULEB128();
          // 打印调试信息，显示起始地址 s 和长度 e 的值
          LOG_INFO("start_length {:x} {:x}\n", s, e);
          // 将起始地址 s 和 s+e 作为范围添加到 ranges 中
          ranges.emplace_back(s, s + e);
        } break;
        // 如果操作码未知，抛出异常，打印错误信息
        default:
          UNWIND_CHECK(false, "unknown range op: {}", op);
      }
    }
  }

  // 根据给定的偏移量创建一个 CheckedLexer 对象，用于查找调试信息条目的缩略语表
  CheckedLexer findAbbrev(uint64_t offset, uint64_t entry) {
    // 使用 debug_abbrev 内的偏移量创建 CheckedLexer 对象 L
    CheckedLexer L = s_.debug_abbrev.lexer(offset);
    // 进入循环，查找匹配的缩略语表条目
    while (true) {
      // 从 L 中读取一个 ULEB128 类型的缩略语码 abbrev_code
      auto abbrev_code = L.readULEB128();
      // 检查读取的缩略语码是否为 0，如果是则打印错误信息，说明未找到匹配的条目
      UNWIND_CHECK(
          abbrev_code != 0,
          "could not find entry {} at offset {:x}",
          entry,
          offset);
      // 从 L 中读取一个 ULEB128 类型的标签 tag
      auto tag = L.readULEB128();
      // 从 L 中读取一个 uint8_t 类型的子节点标记（表明是否有子节点）
      L.read<uint8_t>(); // has children
      // 如果读取的缩略语码 abbrev_code 等于给定的 entry
      if (abbrev_code == entry) {
        // 检查标签 tag 是否为 DW_TAG_compile_unit，如果不是则打印错误信息
        UNWIND_CHECK(
            tag == DW_TAG_compile_unit,
            "first entry was not a compile unit but {}",
            tag);
        // 返回匹配的 CheckedLexer 对象 L
        return L;
      }
      // 再次进入循环，处理缩略语表的属性和值对
      while (true) {
        // 从 L 中分别读取一个 ULEB128 类型的属性 attr 和一个 ULEB128 类型的形式 form
        auto attr = L.readULEB128();
        auto form = L.readULEB128();
        // 如果读取的属性 attr 和形式 form 都为 0，则跳出内层循环
        if (attr == 0 && form == 0) {
          break;
        }
        // 如果形式 form 是 DW_FORM_implicit_const，则从 L 中读取一个 SLEB128 类型的常量值
        if (form == DW_FORM_implicit_const) {
          L.readSLEB128();
        }
      }
  }
  // 结构体的结尾

  Sections& s_;
  // 引用 Sections 类型的对象 s_

  optional<uint64_t> line_number_program_offset_;
  // 可选的无符号64位整数，用于存储行号程序的偏移量

  uint64_t offset_ = 0;
  // 无符号64位整数 offset_ 初始化为 0，用于存储偏移量信息

  uint8_t sec_offset_size_ = 0;
  // 无符号8位整数 sec_offset_size_ 初始化为 0，用于存储段偏移大小信息

  uint64_t length_ = 0;
  // 无符号64位整数 length_ 初始化为 0，用于存储长度信息

  const char* end_ = nullptr;
  // 指向常量字符的指针 end_ 初始化为 nullptr，用于标记结尾位置

  uint64_t debug_abbrev_offset_ = 0;
  // 无符号64位整数 debug_abbrev_offset_ 初始化为 0，用于存储调试信息的缩写偏移量

  bool is_64bit_ = false;
  // 布尔变量 is_64bit_ 初始化为 false，用于标记是否为64位架构

  std::optional<std::pair<uint64_t, uint8_t>> range_ptr_;
  // 可选的包含无符号64位整数和无符号8位整数的 pair 对象 range_ptr_

  uint64_t lowpc_ = 0;
  // 无符号64位整数 lowpc_ 初始化为 0，用于存储程序代码段的起始地址

  optional<uint64_t> highpc_;
  // 可选的无符号64位整数，用于存储程序代码段的结束地址

  uint16_t version_ = 0;
  // 无符号16位整数 version_ 初始化为 0，用于存储版本信息

  uint64_t address_base_ = 0;
  // 无符号64位整数 address_base_ 初始化为 0，用于存储地址基准信息

  optional<uint64_t> rnglists_base_;
  // 可选的无符号64位整数，用于存储范围列表的基地址
};

} // namespace torch::unwind
```