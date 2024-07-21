# `.\pytorch\aten\src\ATen\MapAllocator.h`

```py
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <c10/core/Allocator.h>
#include <c10/util/string_view.h>
// 包含 C++ 标准库头文件

namespace at {
// 命名空间 at，用于包含所有的类、函数和变量定义

enum MappedAllocatorModes {
  ALLOCATOR_MAPPED_SHARED = 1,
  ALLOCATOR_MAPPED_SHAREDMEM = 2,
  ALLOCATOR_MAPPED_EXCLUSIVE = 4,
  ALLOCATOR_MAPPED_NOCREATE = 8,
  ALLOCATOR_MAPPED_KEEPFD = 16,
  ALLOCATOR_MAPPED_FROMFD = 32,
  ALLOCATOR_MAPPED_UNLINK = 64
};
// 枚举类型 MappedAllocatorModes，定义了多种映射分配器模式

// Sentinel value/type to help distinguish the file descriptor constructor from
// the non-file descriptor constructor
enum WithFd { WITH_FD };
// 枚举类型 WithFd，用于区分基于文件描述符构造函数和非基于文件描述符构造函数

TORCH_API std::string NewProcessWideShmHandle();
// 声明一个函数 NewProcessWideShmHandle，返回一个 std::string

class TORCH_API MapAllocator {
// 类 MapAllocator 的定义开始
 public:
  MapAllocator(c10::string_view filename, int flags, size_t size);
  // 构造函数：使用文件名、标志和大小初始化映射分配器

  MapAllocator(
      WithFd,
      c10::string_view filename,
      int fd,
      int flags,
      size_t size);
  // 构造函数：使用文件描述符、文件名、标志和大小初始化映射分配器

  MapAllocator(const MapAllocator&) = delete;
  // 删除拷贝构造函数

  MapAllocator& operator=(const MapAllocator&) = delete;
  // 删除拷贝赋值运算符

  MapAllocator(MapAllocator&&) = delete;
  // 删除移动构造函数

  MapAllocator& operator=(MapAllocator&&) = delete;
  // 删除移动赋值运算符

  const char* filename() const {
    return filename_.c_str();
  }
  // 返回文件名的 C 风格字符串

  int fd() const {
#ifdef _WIN32
    TORCH_CHECK(false, "MapAllocator::fd() is unsupported on Windows");
#else
    return fd_;
#endif
  }
  // 返回文件描述符（如果在非 Windows 平台上）

  ptrdiff_t size() const {
    return size_;
  }
  // 返回映射的大小

  // Return a pointer to the actual data for this allocator
  // (in the case of the refcounted allocator, this is offset
  // from the base pointer.)
  virtual void* data() const {
    return base_ptr_;
  }
  // 返回分配器的实际数据指针

  int flags() const {
    return flags_;
  }
  // 返回标志位

  static MapAllocator* fromDataPtr(const at::DataPtr&);
  // 静态方法：从数据指针创建 MapAllocator 对象

  static at::DataPtr makeDataPtr(
      c10::string_view filename,
      int flags,
      size_t size,
      size_t* actual_size_out);
  // 静态方法：创建一个 DataPtr 对象

  static at::DataPtr makeDataPtr(
      WithFd,
      const char* filename,
      int fd,
      int flags,
      size_t size,
      size_t* actual_size_out);
  // 静态方法：基于文件描述符创建 DataPtr 对象

  // Closes the data.  Helps us avoid destructor shenanigans
  virtual void close();
  // 关闭数据的虚函数，帮助避免析构函数中的问题

  // This is very dangerous.  You have to redefine this destructor for each
  // subclass
  virtual ~MapAllocator();
  // 虚析构函数，极其危险，每个子类必须重新定义

 protected:
  bool closed_ = false;
  // 保护成员变量：表示是否已关闭

  std::string filename_;
  // 保护成员变量：文件名

  int flags_ = 0;
  // 保护成员变量：标志位

  ptrdiff_t size_; /* mapped size */
  // 保护成员变量：映射的大小

#ifdef _WIN32
  void* handle_;
  void* event_;
  std::string eventname_;
#else
  int fd_ = -1;
#endif
  // 平台相关的成员变量：Windows 上为 handle 和 event，其他平台为文件描述符 fd

  void* base_ptr_ = nullptr;
  // 保护成员变量：基础指针，指向分配器的实际数据
};

// Base-from-member idiom
struct TORCH_API RefcountedMapAllocatorArgCheck {
  RefcountedMapAllocatorArgCheck(int flags);
};
// 结构体：基于成员的 idiom，用于检查引用计数映射分配器的参数
class TORCH_API RefcountedMapAllocator : private RefcountedMapAllocatorArgCheck,
                                         public MapAllocator {
 public:
  // 构造函数，用于从文件名、标志和大小创建分配器
  RefcountedMapAllocator(const char* filename, int flags, size_t size);
  
  // 构造函数，用于从文件描述符、文件名、标志和大小创建分配器
  RefcountedMapAllocator(
      WithFd,
      const char* filename,
      int fd,
      int flags,
      size_t size);

  // 从 DataPtr 创建 RefcountedMapAllocator 对象
  static RefcountedMapAllocator* fromDataPtr(const at::DataPtr&);

  // 创建一个新的 DataPtr，从文件名、标志和大小中返回实际大小
  static at::DataPtr makeDataPtr(
      const char* filename,
      int flags,
      size_t size,
      size_t* actual_size_out);

  // 创建一个新的 DataPtr，从文件描述符、文件名、标志和大小中返回实际大小
  static at::DataPtr makeDataPtr(
      WithFd,
      const char* filename,
      int fd,
      int flags,
      size_t size,
      size_t* actual_size_out);

  // 返回当前数据指针的地址
  void* data() const override;

  // 增加引用计数
  void incref();

  // 减少引用计数，返回当前引用计数值
  int decref();

  // 关闭分配器
  void close() override;

  // 析构函数，调用 close() 方法关闭分配器
  ~RefcountedMapAllocator() override {
    RefcountedMapAllocator::close();
  }

 protected:
  // 检查标志的有效性
  void checkFlags();

  // 初始化分配
  void initializeAlloc();
};

} // namespace at
```