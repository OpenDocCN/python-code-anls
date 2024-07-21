# `.\pytorch\torch\csrc\jit\serialization\python_print.h`

```py
#pragma once
// 使用预处理指令#pragma once确保头文件只被编译一次，防止重复包含

#include <torch/csrc/Export.h>
// 引入torch库中的Export.h头文件，用于导出符号（symbol）的定义

#include <torch/csrc/jit/api/module.h>
// 引入torch库中jit模块的api/module.h头文件，用于定义模块API

#include <torch/csrc/jit/ir/ir.h>
// 引入torch库中jit模块的ir/ir.h头文件，用于定义中间表示（IR）的相关功能

#include <vector>
// 引入C++标准库中的vector容器

namespace torch::jit {

struct Method;
// 前向声明Method结构体，用于定义torch::jit命名空间中的Method

struct Module;
// 前向声明Module结构体，用于定义torch::jit命名空间中的Module

struct PythonPrintImpl;
// 前向声明PythonPrintImpl结构体，用于定义torch::jit命名空间中Python打印的实现细节

struct PrintDepsTable {
  void add(const c10::NamedTypePtr& type);
  // 添加方法声明add，用于向依赖表中添加指定类型的命名类型指针

  size_t size() const {
    return table_.size();
  }
  // 返回依赖表中元素的数量

  const c10::NamedTypePtr& operator[](size_t index) const {
    return table_[index];
  }
  // 重载[]运算符，用于访问依赖表中指定索引的命名类型指针

 private:
  std::vector<c10::NamedTypePtr> table_;
  // 声明私有成员table_，用于存储命名类型指针的向量容器

  std::unordered_set<c10::NamedTypePtr> non_unique_;
  // 声明私有成员non_unique_，用于存储不唯一的命名类型指针的无序集合
};

struct TORCH_API PythonPrint {
  PythonPrint(
      std::vector<IValue>& constant_table,
      PrintDepsTable& deps_table,
      c10::TypePrinter type_printer = nullptr,
      bool enforce_importable = false);
  // 声明PythonPrint构造函数，接受常量表、依赖表、类型打印器和是否强制可导入性作为参数

  void printNamedType(const c10::NamedTypePtr& classType);
  // 声明printNamedType方法，用于打印指定命名类型的信息

  void printFunction(const Function& callee);
  // 声明printFunction方法，用于打印指定函数对象的信息

  void printMethod(const Function& callee);
  // 声明printMethod方法，用于打印指定方法对象的信息

  std::string str() const;
  // 声明str方法，返回PythonPrint对象的字符串表示形式

  const SourceRangeRecords& ranges() const;
  // 声明ranges方法，返回PythonPrint对象的源代码范围记录

  uint64_t minVersion() const;
  // 声明minVersion方法，返回PythonPrint对象的最小版本号

 private:
  std::shared_ptr<PythonPrintImpl> pImpl;
  // 声明私有成员pImpl，使用shared_ptr管理PythonPrintImpl对象的实现细节
};

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
// 声明printerHasSpecialCaseFor函数，用于判断打印机是否具有特定符号的特殊处理情况

TORCH_API void jitModuleToPythonCodeAndConstants(
    const Module& module,
    ExtraFilesMap* jit_sources, // output
    std::vector<IValue>* constants // output
);
// 声明jitModuleToPythonCodeAndConstants函数，将jit模块转换为Python代码和常量，输出为jit_sources和constants

} // namespace torch::jit
// 结束torch::jit命名空间的定义
```