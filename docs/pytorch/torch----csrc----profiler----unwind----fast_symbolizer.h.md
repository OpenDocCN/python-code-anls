# `.\pytorch\torch\csrc\profiler\unwind\fast_symbolizer.h`

```py
#pragma once

#include <fmt/format.h> // 包含格式化库的头文件
#include <sys/types.h> // 包含系统类型相关的头文件
#include <torch/csrc/profiler/unwind/debug_info.h> // 包含调试信息相关头文件
#include <torch/csrc/profiler/unwind/line_number_program.h> // 包含行号程序相关头文件
#include <torch/csrc/profiler/unwind/sections.h> // 包含节相关头文件
#include <torch/csrc/profiler/unwind/unwind.h> // 包含展开相关头文件
#include <torch/csrc/profiler/unwind/unwind_error.h> // 包含展开错误相关头文件
#include <cstddef> // 包含标准定义头文件
#include <memory> // 包含内存管理相关头文件

namespace torch::unwind {

#define UNWIND_WARN(w, ...)                   \ // 定义展开警告宏，记录警告并输出日志
  do {                                        \
    w.emplace_back(fmt::format(__VA_ARGS__)); \
    LOG_INFO("WARNING: {}\n", w.back());      \
  } while (0);

struct FastSymbolizer { // 定义快速符号化器结构体
  FastSymbolizer() = default; // 默认构造函数
  Frame symbolize(const std::string& library, uint64_t offset) { // 符号化函数，输入库名和偏移量，返回符号化后的帧信息
    LOG_INFO("symbolizing {} + 0x{:x}\n", library, offset); // 记录符号化过程日志
    Frame frame; // 创建帧对象
    frame.funcname = "??"; // 初始化函数名为未知
    frame.filename = library; // 文件名初始化为库名
    frame.lineno = offset; // 行号初始化为偏移量
    auto s = getOrCreateSections(library); // 获取或创建库的节信息
    if (auto e = s->findSubprogramName(offset)) { // 查找偏移量对应的子程序名
      frame.funcname = *e; // 如果找到，设置帧的函数名
    } else {
      UNWIND_WARN( // 如果未找到，记录警告
          warnings_,
          "failed to find subprogram name for {} 0x{:x}",
          library,
          offset);
    }
    if (auto e = findLine(s, offset)) { // 查找偏移量对应的文件名和行号
      frame.filename = e->first; // 设置帧的文件名
      frame.lineno = e->second; // 设置帧的行号
    } else {
      UNWIND_WARN( // 如果未找到，记录警告
          warnings_, "failed to find file/line for {} 0x{:x}", library, offset);
    }
    return frame; // 返回符号化后的帧信息
  }
  const std::vector<std::string>& warnings() { // 获取警告信息的访问器函数
    return warnings_; // 返回警告信息向量
  }

 private:
  void parseDebugInfo(Sections* s) { // 解析调试信息的函数
    uint64_t offset = 0; // 初始化偏移量
    while (offset < s->debug_info.size) { // 循环直到调试信息结束
      DebugInfo info(*s); // 创建调试信息对象
      info.parse(offset); // 解析调试信息
      if (auto lnp_offset = info.lineNumberProgramOffset()) { // 如果有行号程序偏移量
        for (auto r : info.ranges()) { // 遍历调试信息的范围
          s->addDebugInfoRange(r.first, r.second, line_number_programs_.size()); // 添加调试信息的范围
        }
        line_number_programs_.emplace_back( // 添加行号程序到列表中
            std::make_unique<LineNumberProgram>(*s, *lnp_offset));
      }
      offset = info.nextOffset(); // 更新偏移量到下一个调试信息的位置
    }
  }
  Sections* getOrCreateSections(const std::string& library) { // 获取或创建库节信息的函数
    auto it = libraries_.find(library); // 查找库是否已存在
    if (it == libraries_.end()) { // 如果库不存在
      it = libraries_.insert({library, std::make_unique<Sections>()}).first; // 创建新的库节信息
      try {
        Sections* s = it->second.get(); // 获取库的节信息指针
        s->parse(library.c_str()); // 解析库的节信息
        parseDebugInfo(s); // 解析调试信息
      } catch (UnwindError& err) {
        UNWIND_WARN( // 如果解析失败，记录警告
            warnings_, "failed to parse library {}: {}", library, err.what());
      }
    }
    return it->second.get(); // 返回库的节信息指针
  }
  optional<std::pair<std::string, int64_t>> findLine( // 查找行号的函数
      Sections* s,
      uint64_t offset) {
    // 如果能找到与给定偏移量匹配的调试信息的索引
    if (auto idx = s->findDebugInfoOffset(offset)) {
      // 从存储所有线号程序的容器中获取对应索引的指针
      auto r = line_number_programs_.at(*idx).get();
      try {
        // 解析线号程序内容
        r->parse();
      } catch (UnwindError& err) {
        // 捕获解析异常，并记录警告信息
        UNWIND_WARN(
            warnings_,
            "failed to read line number program [{:x}] {}",
            r->offset(),
            err.what());
      }
      // 如果能在解析后的线号程序中找到给定偏移量的条目
      if (auto e = r->find(offset)) {
        // 返回找到的文件名和行号
        return std::make_pair(r->filename(e->file), e->line);
      }
    }
    // 如果未找到匹配的调试信息，返回空值选项
    return std::nullopt;
  }
  // 存储库信息的哈希映射，键为字符串，值为唯一指针到 Sections 对象的指针
  std::unordered_map<std::string, std::unique_ptr<Sections>> libraries_;
  // 存储所有线号程序对象的容器，每个对象是唯一指针到 LineNumberProgram 对象的指针
  std::vector<std::unique_ptr<LineNumberProgram>> line_number_programs_;
  // 存储警告信息的容器，每个元素是字符串
  std::vector<std::string> warnings_;
};

} // namespace torch::unwind
```