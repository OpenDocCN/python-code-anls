# `.\pytorch\torch\csrc\jit\mobile\observer.h`

```
#pragma once

#include <c10/util/ThreadLocalDebugInfo.h>  // 包含线程本地调试信息头文件
#include <string>                           // 包含字符串类定义
#include <unordered_map>                    // 包含无序映射类定义
#include <vector>                           // 包含向量类定义

namespace torch {

class MobileDebugInfo : public c10::DebugInfoBase {
 public:
  const std::string& getModelName() {       // 获取模型名称的访问器函数
    return model_name_;                     // 返回模型名称成员变量
  }

  void setModelName(const std::string& model_name) {  // 设置模型名称的修改器函数
    model_name_ = model_name;                // 将输入的模型名称赋给成员变量
  }

  const std::string& getMethodName() {       // 获取方法名称的访问器函数
    return method_name_;                    // 返回方法名称成员变量
  }

  void setMethodName(const std::string& method_name) {  // 设置方法名称的修改器函数
    method_name_ = method_name;              // 将输入的方法名称赋给成员变量
  }

  size_t getOpIdx() {                        // 获取操作索引的访问器函数
    return op_idx_;                         // 返回操作索引成员变量
  }

  void setOpIdx(size_t op_idx) {             // 设置操作索引的修改器函数
    op_idx_ = op_idx;                       // 将输入的操作索引赋给成员变量
  }

 private:
  std::string model_name_;                   // 模型名称的字符串成员变量
  std::string method_name_;                  // 方法名称的字符串成员变量
  // TODO: Kimish
  // 如果我们启动一个线程，比如对于 at::launch，解释器的继续执行，
  // 并且如果缓存分配器在基本线程中启用了，
  // 那么为了在跨线程边界传播这些信息，即启用了缓存分配器，
  // 可以使用 ThreadLocalDebugInfo 提供的机制。
  // 一旦在启动的线程中可以访问到线程本地的 MobileDebugInfo，
  // 那么在该线程中就可以访问它，并且该线程可以设置自己的线程本地的 CachingAllocatorInfo。
  // 然而，我们不能期望每个启动的线程都提取并设置自己的线程本地的 CachingAllocatorInfo。
  // 但是在轻量级解释器中可以做到，在其运行方法中可以执行以下操作：
  // info = c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::MOBILE_RUNTIME_INFO)).get_caching_allocator_info();
  // GetThreadLocalCachingAllocatorInfo() = info;
  // 另一种选择是让 MobileDebugInfo 本身成为存储线程本地 CachingAllocatorInfo 的地方。
  // 然后 DefaultMobileCPUAllocator 检查这个信息来决定是否使用 CachingAllocator。
  // 然而，当前的轻量级解释器不支持 FORK，因此从轻量级解释器的运行方法中我们不会真正在不同线程中启动另一个轻量级解释器的实例。
  // 因此目前不用担心跨线程边界传递 CachingAllocatorInfo 的问题。
  // c10::CachingAllocatorInfo caching_allocator_info;
  size_t op_idx_ = 0;                       // 操作索引的大小类型成员变量，默认为0
};

}  // namespace torch
// 定义一个观察器接口类，用于监听移动模块的事件
class MobileModuleObserver {
 public:
  // 虚析构函数，确保派生类析构时能正确释放资源
  virtual ~MobileModuleObserver() = default;

  // 进入运行方法的回调函数，参数为方法编号
  virtual void onEnterRunMethod(const int32_t) {}
  
  // 退出运行方法的回调函数，参数包括元数据字典、方法名称和方法编号
  virtual void onExitRunMethod(
      const std::unordered_map<std::string, std::string>&,
      const std::string&,
      const int32_t) {}
  
  // 运行方法失败的回调函数，参数包括元数据字典、方法名称、方法编号和错误消息
  virtual void onFailRunMethod(
      const std::unordered_map<std::string, std::string>&,
      const std::string&,
      const int32_t,
      const char*) {}
  
  // 进入加载模型的回调函数，参数为模型编号
  virtual void onEnterLoadModel(const int32_t) {}
  
  // 退出加载模型的回调函数，参数包括模型编号和文件名到文件内容的映射
  virtual void onExitLoadModel(
      const int32_t,
      const std::unordered_map<std::string, std::string>&) {
  } // key: filename, value: file content
  
  // 加载模型失败的回调函数，参数包括模型编号和错误消息
  virtual void onFailLoadModel(const int32_t, const char*) {}
  
  // 加载模型失败的回调函数，参数包括模型编号、错误消息和文件名到文件内容的映射
  virtual void onFailLoadModel(
      const int32_t,
      const char*,
      const std::unordered_map<std::string, std::string>&) {}
  
  // 获取默认的额外文件列表的纯虚函数，派生类必须实现
  virtual std::vector<std::string> getDefaultExtraFiles() = 0;
  
  // 从额外数据中处理元数据的纯虚函数，参数为额外数据的文件名到文件内容映射
  virtual std::unordered_map<std::string, std::string> processMetadataFromExtra(
      const std::unordered_map<std::string, std::string>&) = 0;
};

// 定义移动观察器配置类
class MobileObserverConfig {
 public:
  // 设置模块观察器，接受一个移动模块观察器的唯一指针
  void setModuleObserver(std::unique_ptr<MobileModuleObserver> reporter) {
    module_observer_ = std::move(reporter);
  }
  
  // 获取模块观察器指针
  MobileModuleObserver* getModuleObserver() {
    return module_observer_.get();
  }

 private:
  std::unique_ptr<MobileModuleObserver> module_observer_; // 移动模块观察器的唯一指针
};

// 命名空间 torch 内部的移动观察器配置函数
MobileObserverConfig& observerConfig();
```