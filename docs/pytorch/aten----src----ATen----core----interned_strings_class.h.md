# `.\pytorch\aten\src\ATen\core\interned_strings_class.h`

```py
# 包含标准库头文件：互斥锁、字符串、无序映射、向量
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

# 包含 ATen 库的符号定义
#include <ATen/core/symbol.h>

# 包含 C10 库的异常处理
#include <c10/util/Exception.h>

# 进入 c10 命名空间
namespace c10 {

# InternedStrings 结构体定义开始
struct TORCH_API InternedStrings {
  # 默认构造函数声明
  InternedStrings();
  
  # 返回给定字符串对应的符号
  Symbol symbol(const std::string& s);
  
  # 返回符号对应的字符串起始和结束指针
  std::pair<const char*, const char*> string(Symbol sym);
  
  # 返回符号的命名空间
  Symbol ns(Symbol sym);

 private:
  # 仅限内部使用，获取给定字符串对应的符号
  # 先决条件 - 持有 mutex_
  Symbol _symbol(const std::string& s);
  
  # 自定义字符串的起始和结束指针
  std::pair<const char*, const char*> customString(Symbol sym);
  
  # 字符串到符号的无序映射
  std::unordered_map<std::string, Symbol> string_to_sym_;
  
  # 符号信息结构体
  struct SymbolInfo {
    Symbol ns;           # 符号的命名空间
    std::string qual_name;   # 完全限定名
    std::string unqual_name; # 非限定名
  };
  
  # 符号到符号信息的向量
  std::vector<SymbolInfo> sym_to_info_;
  
  # 互斥锁，用于保护访问数据结构的线程安全
  std::mutex mutex_;
};

} // namespace c10
```