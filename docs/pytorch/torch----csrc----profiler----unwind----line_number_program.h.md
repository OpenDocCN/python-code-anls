# `.\pytorch\torch\csrc\profiler\unwind\line_number_program.h`

```
// 包含所需的头文件
#include <c10/util/irange.h>
#include <torch/csrc/profiler/unwind/debug_info.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <tuple>

// 定义命名空间 torch::unwind
namespace torch::unwind {

// 表示调试信息中的行号程序结构
struct LineNumberProgram {
  // 构造函数，初始化 Sections 对象和偏移量
  LineNumberProgram(Sections& s, uint64_t offset) : s_(s), offset_(offset) {}

  // 返回偏移量
  uint64_t offset() {
    return offset_;
  }

  // 解析行号程序
  void parse() {
    // 如果已解析过，则直接返回
    if (parsed_) {
      return;
    }
    parsed_ = true;
    
    // 使用 debug_line 节的 lexer 来解析
    CheckedLexer L = s_.debug_line.lexer(offset_);
    
    // 读取段长度和是否为64位的标志
    std::tie(length_, is_64bit_) = L.readSectionLength();
    
    // 程序结束的位置
    program_end_ = (char*)L.loc() + length_;
    
    // 读取版本号并检查有效性
    auto version = L.read<uint16_t>();
    UNWIND_CHECK(
        version == 5 || version == 4,
        "expected version 4 or 5 but found {}",
        version);
    
    // 如果版本为5，读取地址大小和段选择器大小
    if (version == 5) {
      auto address_size = L.read<uint8_t>();
      UNWIND_CHECK(
          address_size == 8,
          "expected 64-bit dwarf but found address size {}",
          address_size);
      segment_selector_size_ = L.read<uint8_t>();
    }
    
    // 读取头部长度
    header_length_ = is_64bit_ ? L.read<uint64_t>() : L.read<uint32_t>();
    
    // 保存 lexer 对象
    program_ = L;
    
    // 跳过头部
    program_.skip(int64_t(header_length_));
    
    // 读取最小指令长度、每指令的最大操作数、默认 is_stmt 标志
    minimum_instruction_length_ = L.read<uint8_t>();
    maximum_operations_per_instruction_ = L.read<uint8_t>();
    default_is_stmt_ = L.read<uint8_t>();
    line_base_ = L.read<int8_t>();
    line_range_ = L.read<uint8_t>();
    opcode_base_ = L.read<uint8_t>();
    
    // 检查 line_range_ 必须非零
    UNWIND_CHECK(line_range_ != 0, "line_range_ must be non-zero");
    
    // 读取标准操作码长度
    standard_opcode_lengths_.resize(opcode_base_);
    for (size_t i = 1; i < opcode_base_; i++) {
      standard_opcode_lengths_[i] = L.read<uint8_t>();
    }
    
    // 读取目录条目格式计数
    uint8_t directory_entry_format_count = L.read<uint8_t>();
    // 如果版本号为5，执行以下代码块
    if (version == 5) {
      // 定义结构体 Member，包含两个 uint64_t 类型的字段
      struct Member {
        uint64_t content_type;  // 内容类型
        uint64_t form;  // 表单
      };
      // 创建存储 Member 结构体的 vector
      std::vector<Member> directory_members;
      // 根据 directory_entry_format_count 的值循环读取内容类型和表单，存入 directory_members
      for (size_t i = 0; i < directory_entry_format_count; i++) {
        directory_members.push_back({L.readULEB128(), L.readULEB128()});
      }
      // 读取目录数量
      uint64_t directories_count = L.readULEB128();
      // 循环处理每个目录项
      for (size_t i = 0; i < directories_count; i++) {
        // 遍历 directory_members 中的每个成员
        for (auto& member : directory_members) {
          // 根据 content_type 的值执行不同操作
          switch (member.content_type) {
            case DW_LNCT_path: {
              // 将读取的路径添加到 include_directories_ 中
              include_directories_.emplace_back(
                  s_.readString(L, member.form, is_64bit_, 0));
            } break;
            default: {
              // 跳过指定表单的数据
              skipForm(L, member.form);
            } break;
          }
        }
      }

      // 输出日志信息，显示目录数量及其对应的路径
      for (auto i : c10::irange(directories_count)) {
        (void)i;
        LOG_INFO("{} {}\n", i, include_directories_[i]);
      }

      // 读取文件名条目的格式数量
      auto file_name_entry_format_count = L.read<uint8_t>();
      // 创建存储文件 Member 结构体的 vector
      std::vector<Member> file_members;
      // 根据 file_name_entry_format_count 的值循环读取内容类型和表单，存入 file_members
      for (size_t i = 0; i < file_name_entry_format_count; i++) {
        file_members.push_back({L.readULEB128(), L.readULEB128()});
      }
      // 读取文件数量
      auto files_count = L.readULEB128();
      // 循环处理每个文件条目
      for (size_t i = 0; i < files_count; i++) {
        // 遍历 file_members 中的每个成员
        for (auto& member : file_members) {
          // 根据 content_type 的值执行不同操作
          switch (member.content_type) {
            case DW_LNCT_path: {
              // 将读取的文件名添加到 file_names_ 中
              file_names_.emplace_back(
                  s_.readString(L, member.form, is_64bit_, 0));
            } break;
            case DW_LNCT_directory_index: {
              // 将读取的目录索引添加到 file_directory_index_ 中
              file_directory_index_.emplace_back(readData(L, member.form));
              // 检查目录索引是否超出 include_directories_ 的范围
              UNWIND_CHECK(
                  file_directory_index_.back() < include_directories_.size(),
                  "directory index out of range");
            } break;
            default: {
              // 跳过指定表单的数据
              skipForm(L, member.form);
            } break;
          }
        }
      }

      // 输出日志信息，显示文件数量及其对应的文件名和目录索引
      for (auto i : c10::irange(files_count)) {
        (void)i;
        LOG_INFO("{} {} {}\n", i, file_names_[i], file_directory_index_[i]);
      }
    } else {
      // 版本号不为5时的处理逻辑
      include_directories_.emplace_back(""); // 隐式当前工作目录
      // 循环读取包含目录的字符串，直到遇到空字符串结束
      while (true) {
        auto str = L.readCString();
        if (*str == '\0') {
          break;
        }
        // 将读取的字符串添加到 include_directories_ 中
        include_directories_.emplace_back(str);
      }
      // 添加空字符串表示文件名
      file_names_.emplace_back("");
      // 添加文件目录索引为0
      file_directory_index_.emplace_back(0);
      // 循环读取文件名及其相关信息，直到遇到空字符串结束
      while (true) {
        auto str = L.readCString();
        if (*str == '\0') {
          break;
        }
        // 读取目录索引
        auto directory_index = L.readULEB128();
        // 跳过读取修改时间和文件长度的数据
        L.readULEB128(); // mod_time
        L.readULEB128(); // file_length
        // 将文件名添加到 file_names_ 中，将目录索引添加到 file_directory_index_ 中
        file_names_.emplace_back(str);
        file_directory_index_.push_back(directory_index);
      }
    }

    // 检查最大每条指令操作数是否为1
    UNWIND_CHECK(
        maximum_operations_per_instruction_ == 1,
        "maximum_operations_per_instruction_ must be 1");
    // 检查最小指令长度是否为1
    UNWIND_CHECK(
        minimum_instruction_length_ == 1,
        "minimum_instruction_length_ must be 1");
  readProgram();
  }
  // 定义结构体 Entry，表示程序的入口信息，包括文件索引和行号
  struct Entry {
    uint32_t file = 1;
    int64_t line = 1;
  };
  // 查找给定地址在程序索引中的信息，返回一个可选的 Entry 结构体
  unwind::optional<Entry> find(uint64_t address) {
    // 在程序索引中查找地址对应的条目
    auto e = program_index_.find(address);
    // 如果找不到对应条目，返回空的 optional
    if (!e) {
      return std::nullopt;
    }
    // 返回地址对应的 Entry 信息
    return all_programs_.at(*e).find(address);
  }
  // 根据索引获取文件名
  std::string filename(uint64_t index) {
    // 使用 fmt::format 格式化文件路径和文件名
    return fmt::format(
        "{}/{}",
        include_directories_.at(file_directory_index_.at(index)),
        file_names_.at(index));
  }

 private:
  // 跳过指定格式的表单数据
  void skipForm(CheckedLexer& L, uint64_t form) {
    // 计算表单数据的大小
    auto sz = formSize(form, is_64bit_ ? 8 : 4);
    // 检查表单数据是否支持，否则抛出异常
    UNWIND_CHECK(sz, "unsupported form {}", form);
    // 跳过指定大小的数据
    L.skip(int64_t(*sz));
  }

  // 读取指定编码的数据
  uint64_t readData(CheckedLexer& L, uint64_t encoding) {
    // 根据编码类型读取数据
    switch (encoding) {
      case DW_FORM_data1:
        return L.read<uint8_t>();
      case DW_FORM_data2:
        return L.read<uint16_t>();
      case DW_FORM_data4:
        return L.read<uint32_t>();
      case DW_FORM_data8:
        return L.read<uint64_t>();
      case DW_FORM_udata:
        return L.readULEB128();
      default:
        // 如果编码类型不支持，抛出异常
        UNWIND_CHECK(false, "unsupported data encoding {}", encoding);
    }
  }

  // 生成一个 Entry 条目
  void produceEntry() {
    // 如果处于影子状态，则直接返回
    if (shadow_) {
      return;
    }
    // 如果只有一个范围，则设置起始地址
    if (ranges_.size() == 1) {
      start_address_ = address_;
    }
    // 输出程序行表信息到输出流
    PRINT_LINE_TABLE(
        "{:x}\t{}\t{}\n", address_, filename(entry_.file), entry_.line);
    // 检查 Entry 中的文件索引是否有效
    UNWIND_CHECK(
        entry_.file < file_names_.size(),
        "file index {} > {} entries",
        entry_.file,
        file_names_.size());
    // 添加地址范围到范围表中
    ranges_.add(address_, entry_, true);
  }
  // 结束一个序列
  void endSequence() {
    // 如果处于影子状态，则直接返回
    if (shadow_) {
      return;
    }
    // 输出序列结束的信息到程序行表
    PRINT_LINE_TABLE(
        "{:x}\tEND\n", address_, filename(entry_.file), entry_.line);
    // 将起始地址和程序序列索引添加到程序索引表中
    program_index_.add(start_address_, all_programs_.size(), false);
    program_index_.add(address_, std::nullopt, false);
    // 将当前范围表添加到所有程序序列中
    all_programs_.emplace_back(std::move(ranges_));
    // 重置范围表
    ranges_ = RangeTable<Entry>();
  }
  // 读取程序
  void readProgram() {
    // 循环遍历程序指令，直到当前位置达到程序结尾
    while (program_.loc() < program_end_) {
      // 打印当前指令地址偏移量，相对于调试行数据的起始地址
      PRINT_INST("{:x}: ", (char*)program_.loc() - (s_.debug_line.data));
      // 读取当前指令的操作码
      uint8_t op = program_.read<uint8_t>();
      // 如果操作码大于等于基本操作码
      if (op >= opcode_base_) {
        // 计算并更新地址和行号
        auto op2 = int64_t(op - opcode_base_);
        address_ += op2 / line_range_;
        entry_.line += line_base_ + (op2 % line_range_);
        // 打印地址和行号的更新信息
        PRINT_INST(
            "address += {}, line += {}\n",
            op2 / line_range_,
            line_base_ + (op2 % line_range_));
        // 生成新的调试条目
        produceEntry();
      } else {
        // 根据不同的操作码执行相应的操作
        switch (op) {
          case DW_LNS_extended_op: {
            // 读取扩展操作的长度和具体的扩展操作码
            auto len = program_.readULEB128();
            auto extended_op = program_.read<uint8_t>();
            // 根据扩展操作码执行相应的操作
            switch (extended_op) {
              case DW_LNE_end_sequence: {
                // 结束当前序列
                PRINT_INST("end_sequence\n");
                endSequence();
                // 重置条目信息
                entry_ = Entry{};
              } break;
              case DW_LNE_set_address: {
                // 设置地址
                address_ = program_.read<uint64_t>();
                // 如果不处于影子状态，打印设置地址的信息
                if (!shadow_) {
                  PRINT_INST(
                      "set address {:x} {:x} {:x}\n",
                      address_,
                      min_address_,
                      max_address_);
                }
                // 检查是否处于影子状态
                shadow_ = address_ == 0;
              } break;
              default: {
                // 跳过未知的扩展操作码
                PRINT_INST("skip extended op {}\n", extended_op);
                program_.skip(int64_t(len - 1));
              } break;
            }
          } break;
          case DW_LNS_copy: {
            // 复制当前调试条目
            PRINT_INST("copy\n");
            produceEntry();
          } break;
          case DW_LNS_advance_pc: {
            // 更新地址
            PRINT_INST("advance pc\n");
            address_ += program_.readULEB128();
          } break;
          case DW_LNS_advance_line: {
            // 更新行号
            entry_.line += program_.readSLEB128();
            PRINT_INST("advance line {}\n", entry_.line);

          } break;
          case DW_LNS_set_file: {
            // 设置文件编号
            PRINT_INST("set file\n");
            entry_.file = program_.readULEB128();
          } break;
          case DW_LNS_const_add_pc: {
            // 使用固定增量更新地址
            PRINT_INST("const add pc\n");
            address_ += (255 - opcode_base_) / line_range_;
          } break;
          case DW_LNS_fixed_advance_pc: {
            // 使用固定的地址增量更新地址
            PRINT_INST("fixed advance pc\n");
            address_ += program_.read<uint16_t>();
          } break;
          default: {
            // 处理其他未知操作码
            PRINT_INST("other {}\n", op);
            auto n = standard_opcode_lengths_[op];
            // 跳过标准操作码指定的字节数
            for (int i = 0; i < n; ++i) {
              program_.readULEB128();
            }
          } break;
        }
      }
    }
  PRINT_INST(
      "{:x}: end {:x}\n",
      ((char*)program_.loc() - s_.debug_line.data),
      program_end_ - s_.debug_line.data);


// 使用PRINT_INST宏打印调试信息，格式化输出program_的位置与program_end_相对于s_.debug_line.data的偏移量
// {:x}: end {:x}\n 表示输出十六进制格式的位置信息和结束位置信息

  }

  uint64_t address_ = 0;
  bool shadow_ = false;
  bool parsed_ = false;
  Entry entry_ = {};
  std::vector<std::string> include_directories_;
  std::vector<std::string> file_names_;
  std::vector<uint64_t> file_directory_index_;
  uint8_t segment_selector_size_ = 0;
  uint8_t minimum_instruction_length_ = 0;
  uint8_t maximum_operations_per_instruction_ = 0;
  int8_t line_base_ = 0;
  uint8_t line_range_ = 0;
  uint8_t opcode_base_ = 0;
  bool default_is_stmt_ = false;
  CheckedLexer program_ = {nullptr};
  char* program_end_ = nullptr;
  uint64_t header_length_ = 0;
  uint64_t length_ = 0;
  bool is_64bit_ = false;
  std::vector<uint8_t> standard_opcode_lengths_;
  Sections& s_;
  uint64_t offset_;
  uint64_t start_address_ = 0;
  RangeTable<uint64_t> program_index_;
  std::vector<RangeTable<Entry>> all_programs_;
  RangeTable<Entry> ranges_;


// 初始化一系列成员变量，用于存储调试信息和指令集分析过程中的数据：
// address_, shadow_, parsed_, entry_, include_directories_, file_names_,
// file_directory_index_, segment_selector_size_, minimum_instruction_length_,
// maximum_operations_per_instruction_, line_base_, line_range_, opcode_base_,
// default_is_stmt_, program_, program_end_, header_length_, length_, is_64bit_,
// standard_opcode_lengths_, s_, offset_, start_address_, program_index_,
// all_programs_, ranges_
// 每个变量的具体用途可能在代码的其他部分有详细说明。
};

// 结束命名空间 torch::unwind
} // namespace torch::unwind
```