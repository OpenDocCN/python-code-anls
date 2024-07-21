# `.\pytorch\torch\csrc\profiler\unwind\eh_frame_hdr.h`

```py
#pragma once
#include <stdint.h>
#include <ostream>

#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>

// Overview of the format described in
// https://refspecs.linuxfoundation.org/LSB_1.3.0/gLSB/gLSB/ehframehdr.html

// 命名空间 torch::unwind 下定义 EHFrameHdr 结构体
namespace torch::unwind {

// EHFrameHdr 结构体定义
struct EHFrameHdr {
  // 构造函数，初始化 EHFrameHdr 对象，接受基址 base 参数
  EHFrameHdr(void* base) : base_(base) {
    // 使用 Lexer 类来解析基址 base 处的数据
    Lexer L(base, base);
    // 读取版本号（1 字节）
    version_ = L.read<uint8_t>();
    // 读取 eh_frame 段指针编码类型（1 字节）
    eh_frame_ptr_enc_ = L.read<uint8_t>();
    // 读取 FDE 计数编码类型（1 字节）
    fde_count_enc_ = L.read<uint8_t>();
    // 读取搜索表编码类型（1 字节）
    table_enc_ = L.read<uint8_t>();
    
    // 根据 table_enc_ 的值判断表的大小
    if (table_enc_ == DW_EH_PE_omit) {
      // 如果表编码为 DW_EH_PE_omit，则表大小为 0
      table_size_ = 0;
    } else {
      switch (table_enc_ & 0xF) {
        // 根据不同的编码类型确定表的大小
        case DW_EH_PE_udata2:
        case DW_EH_PE_sdata2:
          table_size_ = 2;
          break;
        case DW_EH_PE_udata4:
        case DW_EH_PE_sdata4:
          table_size_ = 4;
          break;
        case DW_EH_PE_udata8:
        case DW_EH_PE_sdata8:
          table_size_ = 8;
          break;
        case DW_EH_PE_uleb128:
        case DW_EH_PE_sleb128:
          // 不支持的编码类型，抛出异常
          throw UnwindError("uleb/sleb table encoding not supported");
          break;
        default:
          // 未知的表编码类型，抛出异常
          throw UnwindError("unknown table encoding");
      }
    }

    // 读取 eh_frame 段的地址
    eh_frame_ = (void*)L.readEncodedOr(eh_frame_ptr_enc_, 0);
    // 读取 FDE 计数
    fde_count_ = L.readEncodedOr(fde_count_enc_, 0);
    // 记录搜索表的起始位置
    table_start_ = L.loc();
  }

  // 返回 FDE 计数
  size_t nentries() const {
    return fde_count_;
  }

  // 根据索引 i 返回对应的 lowpc（用于计算 FDE 的起始地址）
  uint64_t lowpc(size_t i) const {
    return Lexer(table_start_, base_)
        .skip(2 * i * table_size_)
        .readEncoded(table_enc_);
  }

  // 根据索引 i 返回对应的 FDE 地址
  void* fde(size_t i) const {
    return (void*)Lexer(table_start_, base_)
        .skip((2 * i + 1) * table_size_)
        .readEncoded(table_enc_);
  }

  // 根据地址 addr 查找对应的 FDE 地址
  void* entryForAddr(uint64_t addr) const {
    // 如果搜索表大小为 0 或者 FDE 计数为 0，则抛出异常
    if (!table_size_ || !nentries()) {
      throw UnwindError("search table not present");
    }
    
    // 二分查找，根据地址 addr 查找对应的 FDE 地址
    uint64_t low = 0;
    uint64_t high = nentries();
    while (low + 1 < high) {
      auto mid = (low + high) / 2;
      if (addr < lowpc(mid)) {
        high = mid;
      } else {
        low = mid;
      }
    }
    return fde(low);
  }

  // 友元函数，打印 EHFrameHdr 对象信息到输出流 out
  friend std::ostream& operator<<(std::ostream& out, const EHFrameHdr& self) {
    out << "EHFrameHeader(version=" << self.version_
        << ",table_size=" << self.table_size_
        << ",fde_count=" << self.fde_count_ << ")";
    return out;
  }

 private:
  void* base_;          // 基址指针
  void* table_start_;   // 搜索表的起始地址
  uint8_t version_;     // 版本号
  uint8_t eh_frame_ptr_enc_;   // eh_frame 段指针编码类型
  uint8_t fde_count_enc_;      // FDE 计数编码类型
  uint8_t table_enc_;          // 搜索表编码类型
  void* eh_frame_ = nullptr;   // eh_frame 段的地址
  int64_t fde_count_;          // FDE 计数
  uint32_t table_size_;        // 搜索表的大小
};

} // namespace torch::unwind
```