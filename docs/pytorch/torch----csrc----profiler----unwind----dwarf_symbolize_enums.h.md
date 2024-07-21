# `.\pytorch\torch\csrc\profiler\unwind\dwarf_symbolize_enums.h`

```
#pragma once
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>
#include <optional>

// DWARF 调试信息标签
enum {
  DW_TAG_subprogram = 0x2e,                // 子程序
  DW_TAG_inlined_subroutine = 0x1d,        // 内联子例程
  DW_TAG_compile_unit = 0x11,              // 编译单元
  DW_AT_sibling = 0x1,                     // 引用
  DW_AT_name = 0x3,                        // 字符串
  DW_AT_stmt_list = 0x10,                  // 行指针
  DW_AT_addr_base = 0x73,                  // 节偏移
  DW_AT_rnglists_base = 0x74,              // 节偏移
  DW_AT_low_pc = 0x11,                     // 地址
  DW_AT_high_pc = 0x12,                    // 地址
  DW_AT_specification = 0x47,              // 引用
  DW_AT_abstract_origin = 0x31,            // 引用
  DW_AT_linkage_name = 0x6e,               // 字符串
  DW_AT_ranges = 0x55,                     // 范围列表
  DW_AT_str_offsets_base = 0x72,           // 节偏移
  DW_FORM_addr = 0x01,                     // 地址
  DW_FORM_block2 = 0x03,                   // 块（2字节）
  DW_FORM_block4 = 0x04,                   // 块（4字节）
  DW_FORM_data2 = 0x05,                    // 数据（2字节）
  DW_FORM_data4 = 0x06,                    // 数据（4字节）
  DW_FORM_data8 = 0x07,                    // 数据（8字节）
  DW_FORM_string = 0x08,                   // 字符串
  DW_FORM_block = 0x09,                    // 块
  DW_FORM_block1 = 0x0a,                   // 块（1字节）
  DW_FORM_data1 = 0x0b,                    // 数据（1字节）
  DW_FORM_flag = 0x0c,                     // 标志
  DW_FORM_sdata = 0x0d,                    // 有符号数据
  DW_FORM_strp = 0x0e,                     // 字符串指针
  DW_FORM_udata = 0x0f,                    // 无符号数据
  DW_FORM_ref_addr = 0x10,                 // 引用地址
  DW_FORM_ref1 = 0x11,                     // 引用（1字节）
  DW_FORM_ref2 = 0x12,                     // 引用（2字节）
  DW_FORM_ref4 = 0x13,                     // 引用（4字节）
  DW_FORM_ref8 = 0x14,                     // 引用（8字节）
  DW_FORM_ref_udata = 0x15,                // 引用（无符号数据）
  DW_FORM_indirect = 0x16,                 // 间接
  DW_FORM_sec_offset = 0x17,               // 节偏移
  DW_FORM_exprloc = 0x18,                  // 表达式位置
  DW_FORM_flag_present = 0x19,             // 标志存在
  DW_FORM_strx = 0x1a,                     // 字符串索引
  DW_FORM_addrx = 0x1b,                    // 地址索引
  DW_FORM_ref_sup4 = 0x1c,                 // 引用扩展4
  DW_FORM_strp_sup = 0x1d,                 // 字符串指针扩展
  DW_FORM_data16 = 0x1e,                   // 数据（16字节）
  DW_FORM_line_strp = 0x1f,                // 行字符串指针
  DW_FORM_ref_sig8 = 0x20,                 // 引用签名（8字节）
  DW_FORM_implicit_const = 0x21,           // 隐式常量
  DW_FORM_loclistx = 0x22,                 // 位置列表索引
  DW_FORM_rnglistx = 0x23,                 // 范围列表索引
  DW_FORM_ref_sup8 = 0x24,                 // 引用扩展8
  DW_FORM_strx1 = 0x25,                    // 字符串索引（1字节）
  DW_FORM_strx2 = 0x26,                    // 字符串索引（2字节）
  DW_FORM_strx3 = 0x27,                    // 字符串索引（3字节）
  DW_FORM_strx4 = 0x28,                    // 字符串索引（4字节）
  DW_FORM_addrx1 = 0x29,                   // 地址索引（1字节）
  DW_FORM_addrx2 = 0x2a,                   // 地址索引（2字节）
  DW_FORM_addrx3 = 0x2b,                   // 地址索引（3字节）
  DW_FORM_addrx4 = 0x2c,                   // 地址索引（4字节）
  /* GNU 调试分离扩展 */
  DW_FORM_GNU_addr_index = 0x1f01,         // 地址索引
  DW_FORM_GNU_str_index = 0x1f02,          // 字符串索引
  DW_FORM_GNU_ref_alt = 0x1f20,            // 替代 .debuginfo 中的偏移
  DW_FORM_GNU_strp_alt = 0x1f21,           // 替代 .debug_str 中的偏移
  DW_LNCT_path = 0x1,                      // 路径
  DW_LNCT_directory_index = 0x2,           // 目录索引
  DW_LNS_extended_op = 0x00,               // 扩展操作
  DW_LNE_end_sequence = 0x01,              // 结束序列
  DW_LNE_set_address = 0x02,               // 设置地址
  DW_LNS_copy = 0x01,                      // 复制
  DW_LNS_advance_pc = 0x02,                // 提前程序计数器
  DW_LNS_advance_line = 0x03,              // 提前行数
  DW_LNS_set_file = 0x04,                  // 设置文件
  DW_LNS_const_add_pc = 0x08,              // 常量增加程序计数器
  DW_LNS_fixed_advance_pc = 0x09,          // 固定增加程序计数器
  DW_RLE_end_of_list = 0x0,                // 列表结束
  DW_RLE_base_addressx = 0x1,              // 基地址索引
  DW_RLE_startx_endx = 0x2,                // 开始索引和结束索引
  DW_RLE_startx_length = 0x3,              // 开始索引和长度
  DW_RLE_offset_pair = 0x4,                // 偏移对
  DW_RLE_base_address = 0x5,               // 基地址
  DW_RLE_start_end = 0x6,                  // 开始和结束
  DW_RLE_start_length = 0x7                // 开始和长度
};

// 计算给定 DWARF 表单的大小
static torch::unwind::optional<size_t> formSize(
    uint64_t form,                          // DWARF 表单类型
    uint8_t sec_offset_size) {              // 节偏移大小
  switch (form) {
    case DW_FORM_addr:                      // 地址
      return sizeof(void*);                 // 返回指针大小
    case DW_FORM_block2:                    // 块（2字节）
    case DW_FORM_block4:                    // 块（4字节）
      return std::nullopt;                  // 返回空值表示大小不确定
    case DW_FORM_data2:                     // 数据（2字节）
      return 2;                             // 返回2字节大小
    case DW_FORM_data4:                     // 数据（4字节）
      return 4;                             // 返回4字节大小
    case DW_FORM_data8:                     // 数据（8字节）
      return 8;                             // 返回8字节大小
    case DW_FORM_string:                    // 字符串
    case DW_FORM_block:                     // 块
      return std::nullopt;                  // 返回空值表示大小不确定
    // 如果 DW_FORM 类型是 DW_FORM_block1，则返回空的 optional 对象
    case DW_FORM_block1:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_data1 或者 DW_FORM_flag，则返回整数 1
    case DW_FORM_data1:
    case DW_FORM_flag:
      return 1;
    
    // 如果 DW_FORM 类型是 DW_FORM_sdata，则返回空的 optional 对象
    case DW_FORM_sdata:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_strp，则返回 sec_offset_size（一个大小）
    case DW_FORM_strp:
      return sec_offset_size;
    
    // 如果 DW_FORM 类型是 DW_FORM_udata，则返回空的 optional 对象
    case DW_FORM_udata:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref_addr，则返回 sec_offset_size（一个大小）
    case DW_FORM_ref_addr:
      return sec_offset_size;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref1，则返回整数 1
    case DW_FORM_ref1:
      return 1;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref2，则返回整数 2
    case DW_FORM_ref2:
      return 2;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref4，则返回整数 4
    case DW_FORM_ref4:
      return 4;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref8，则返回整数 8
    case DW_FORM_ref8:
      return 8;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref_udata 或者 DW_FORM_indirect，则返回空的 optional 对象
    case DW_FORM_ref_udata:
    case DW_FORM_indirect:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_sec_offset，则返回 sec_offset_size（一个大小）
    case DW_FORM_sec_offset:
      return sec_offset_size;
    
    // 如果 DW_FORM 类型是 DW_FORM_exprloc，则返回空的 optional 对象
    case DW_FORM_exprloc:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_flag_present，则返回整数 0
    case DW_FORM_flag_present:
      return 0;
    
    // 如果 DW_FORM 类型是 DW_FORM_strx 或者 DW_FORM_addrx，则返回空的 optional 对象
    case DW_FORM_strx:
    case DW_FORM_addrx:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref_sup4，则返回整数 4
    case DW_FORM_ref_sup4:
      return 4;
    
    // 如果 DW_FORM 类型是 DW_FORM_strp_sup，则返回 sec_offset_size（一个大小）
    case DW_FORM_strp_sup:
      return sec_offset_size;
    
    // 如果 DW_FORM 类型是 DW_FORM_data16，则返回整数 16
    case DW_FORM_data16:
      return 16;
    
    // 如果 DW_FORM 类型是 DW_FORM_line_strp，则返回 sec_offset_size（一个大小）
    case DW_FORM_line_strp:
      return sec_offset_size;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref_sig8，则返回整数 8
    case DW_FORM_ref_sig8:
      return 8;
    
    // 如果 DW_FORM 类型是 DW_FORM_implicit_const，则返回整数 0
    case DW_FORM_implicit_const:
      return 0;
    
    // 如果 DW_FORM 类型是 DW_FORM_loclistx 或者 DW_FORM_rnglistx，则返回空的 optional 对象
    case DW_FORM_loclistx:
    case DW_FORM_rnglistx:
      return std::nullopt;
    
    // 如果 DW_FORM 类型是 DW_FORM_ref_sup8，则返回整数 8
    case DW_FORM_ref_sup8:
      return 8;
    
    // 如果 DW_FORM 类型是 DW_FORM_strx1，则返回整数 1
    case DW_FORM_strx1:
      return 1;
    
    // 如果 DW_FORM 类型是 DW_FORM_strx2，则返回整数 2
    case DW_FORM_strx2:
      return 2;
    
    // 如果 DW_FORM 类型是 DW_FORM_strx3，则返回整数 3
    case DW_FORM_strx3:
      return 3;
    
    // 如果 DW_FORM 类型是 DW_FORM_strx4，则返回整数 4
    case DW_FORM_strx4:
      return 4;
    
    // 如果 DW_FORM 类型是 DW_FORM_addrx1，则返回整数 1
    case DW_FORM_addrx1:
      return 1;
    
    // 如果 DW_FORM 类型是 DW_FORM_addrx2，则返回整数 2
    case DW_FORM_addrx2:
      return 2;
    
    // 如果 DW_FORM 类型是 DW_FORM_addrx3，则返回整数 3
    case DW_FORM_addrx3:
      return 3;
    
    // 如果 DW_FORM 类型是 DW_FORM_addrx4，则返回整数 4
    case DW_FORM_addrx4:
      return 4;
    
    // 如果 DW_FORM 类型是 DW_FORM_GNU_addr_index、DW_FORM_GNU_str_index、
    // DW_FORM_GNU_ref_alt 或者 DW_FORM_GNU_strp_alt，默认返回空的 optional 对象
    case DW_FORM_GNU_addr_index:
    case DW_FORM_GNU_str_index:
    case DW_FORM_GNU_ref_alt:
    case DW_FORM_GNU_strp_alt:
    default:
      return std::nullopt;
}


注释：


# 这是一个单独的右花括号 '}'，用于闭合一个代码块，但是缺少与之匹配的左花括号 '{'
```