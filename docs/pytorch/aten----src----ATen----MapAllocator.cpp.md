# `.\pytorch\aten\src\ATen\MapAllocator.cpp`

```
#include <ATen/MapAllocator.h>

#include <atomic>
#include <random>
#include <string>
#if ATOMIC_INT_LOCK_FREE == 2
#define AT_ATOMIC_IPC_REFCOUNT 1
#endif

#include <c10/core/CPUAllocator.h>

#ifdef _WIN32
#include <c10/util/Unicode.h>
#endif

#if defined(HAVE_MMAP)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#if !defined(_MSC_VER) || defined(HAVE_MMAP)
#include <sys/types.h>
#include <unistd.h>
#elif defined(_MSC_VER)
#include <c10/util/win32-headers.h>
#endif
#include <fmt/format.h>

namespace at {

static constexpr int64_t map_alloc_alignment = 64;

// 创建一个新的进程范围共享内存句柄
std::string NewProcessWideShmHandle() {
  // 静态变量，用于生成唯一的计数器
  static std::atomic<uint64_t> counter{0};
  // 随机设备对象
  static std::random_device rd;
#ifdef _MSC_VER
  // 格式化生成共享内存句柄的字符串（Windows 特定）
  return fmt::format(
      "/torch_{}_{}_{}",
      GetCurrentProcessId(),
      rd(),
      counter.fetch_add(1, std::memory_order_relaxed));
#else
  // 格式化生成共享内存句柄的字符串（非 Windows 系统）
  return fmt::format(
      "/torch_{}_{}_{}",
      getpid(),
      rd(),
      counter.fetch_add(1, std::memory_order_relaxed));
#endif
}

#if defined(_WIN32) || defined(HAVE_MMAP)

namespace {
// 匿名命名空间中定义的结构体，用于管理映射信息
struct MapInfo {
  std::atomic<int> refcount; // 引用计数原子变量
};

constexpr const char* unknown_filename = "filename not specified";
#ifdef _WIN32
constexpr const char* unknown_eventname = "eventname not specified";
#endif
}  // namespace (anonymous)

// 构造函数，初始化 MapAllocator 对象
MapAllocator::MapAllocator(WithFd, c10::string_view filename, int fd, int flags, size_t size)
  : filename_(filename.empty() ? unknown_filename : filename) // 初始化文件名，如果未指定则使用默认字符串
  , size_(0) // 待稍后填充的大小信息
#ifdef _WIN32
  , handle_(INVALID_HANDLE_VALUE) // 待稍后填充的句柄信息（Windows 特定）
  , event_(INVALID_HANDLE_VALUE) // 待稍后填充的事件句柄信息（Windows 特定）
  , eventname_(filename.empty() ? unknown_eventname : (std::string(filename) + "_event")) // 根据文件名构造事件名
#else
  , fd_(fd) // 文件描述符初始化（非 Windows 系统）
#endif
{
  // 如果既没有 ALLOCATOR_MAPPED_SHARED 也没有 ALLOCATOR_MAPPED_SHAREDMEM，则取消 ALLOCATOR_MAPPED_NOCREATE 标志
  if (!(flags & ALLOCATOR_MAPPED_SHARED) && !(flags & ALLOCATOR_MAPPED_SHAREDMEM)) {
    flags &= ~ALLOCATOR_MAPPED_NOCREATE;
  }
  // 如果 flags 和 ALLOCATOR_MAPPED_EXCLUSIVE 的按位异或结果为 0，则抛出异常
  if ((flags ^ ALLOCATOR_MAPPED_EXCLUSIVE) == 0) {
    TORCH_CHECK(false, "ALLOCATOR_MAPPED_EXCLUSIVE flag requires opening the file in shared mode");
  }
#ifdef _WIN32
  // 如果在 Windows 下提供了文件描述符，则抛出异常，因为 MapAllocator_newWithFd 在 Windows 上不受支持
  if (fd != -1) {
    TORCH_CHECK(false, "MapAllocator_newWithFd is unsupported on Windows");
  }
#endif
  flags_ = flags; // 设置 flags 成员变量

  // 开始执行分配操作

  // 如果 size 为 0，则直接返回
  if (size == 0) {
    return;
  }

#ifdef _WIN32
  // 如果 flags_ 包含 ALLOCATOR_MAPPED_SHAREDMEM
  if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
    // 执行映射操作（Windows 特定）
    // 设置 filename 和 eventname 的宽字符版本
    const wchar_t *filename;
    const wchar_t *eventname;
    const std::wstring wFilename = c10::u8u16(filename_);
    const std::wstring wEventname = c10::u8u16(eventname_);
    LARGE_INTEGER hfilesz;

    // 如果 filename 的第一个字符是 '/'，则从 wFilename 和 wEventname 的第二个字符开始处理
    if (filename_[0] == '/') {
      filename = wFilename.c_str() + 1;
      eventname = wEventname.c_str() + 1;
    } else {
      filename = wFilename.c_str();
      eventname = wEventname.c_str();
    }

    hfilesz.QuadPart = size;

    // 根据 flags_ 执行不同的事件处理
    if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
      event_ = CreateEventW(nullptr, FALSE, FALSE, eventname);
    } else if (flags_ & ALLOCATOR_MAPPED_NOCREATE) {
      event_ = OpenEventW(EVENT_ALL_ACCESS, FALSE, eventname);
    } else {
      // 如果flags_既不是ALLOCATOR_MAPPED_EXCLUSIVE也不是ALLOCATOR_MAPPED_NOCREATE，抛出错误
      TORCH_CHECK(false, "Expected either ALLOCATOR_MAPPED_EXCLUSIVE or ALLOCATOR_MAPPED_NOCREATE");
    }

    if (event_ == nullptr) {
      // 如果事件指针为空，抛出带有错误信息和事件名称的错误
      TORCH_CHECK(false, "Couldn't open shared event: <", eventname, ">, error code: <", GetLastError(), ">");
    }

    if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
      // 如果flags_包含ALLOCATOR_MAPPED_EXCLUSIVE，创建具有独占访问权限的文件映射对象
      handle_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, filename);
    } else if (flags_ & ALLOCATOR_MAPPED_NOCREATE) {
      // 如果flags_包含ALLOCATOR_MAPPED_NOCREATE，打开现有的文件映射对象
      handle_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, filename);
    } else {
      // 如果flags_既不是ALLOCATOR_MAPPED_EXCLUSIVE也不是ALLOCATOR_MAPPED_NOCREATE，抛出错误
      TORCH_CHECK(false, "Expected either ALLOCATOR_MAPPED_EXCLUSIVE or ALLOCATOR_MAPPED_NOCREATE");
    }

    if (handle_ == nullptr) {
      // 如果文件映射句柄为空，抛出带有错误信息和文件名的错误
      TORCH_CHECK(false, "Couldn't open shared file mapping: <", filename, ">, error code: <", GetLastError(), ">");
    }

    size_ = size;
    // 映射文件的视图到进程地址空间中的base_ptr_
    base_ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!base_ptr_) {
      // 如果映射视图失败，抛出带有错误信息和文件名的错误
      TORCH_CHECK(false, "Couldn't map view of shared file <", filename, ">, error code: <", GetLastError(), ">");
    }
  } else {

    HANDLE hfile;
    HANDLE hmfile;
    LARGE_INTEGER hfilesz;

    if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
      // 如果flags_包含ALLOCATOR_MAPPED_EXCLUSIVE，抛出不支持的独占文件映射错误
      TORCH_CHECK(false, "exclusive file mapping is not supported on Windows");
    }
    if (flags_ & ALLOCATOR_MAPPED_NOCREATE) {
      // 如果flags_包含ALLOCATOR_MAPPED_NOCREATE，抛出不支持的无创建文件映射错误
      TORCH_CHECK(false, "file mapping without creation is not supported on Windows");
    }
    if (flags_ & ALLOCATOR_MAPPED_KEEPFD) {
      // 如果flags_包含ALLOCATOR_MAPPED_KEEPFD，抛出不支持的保持文件描述符错误
      TORCH_CHECK(false, "ALLOCATOR_MAPPED_KEEPFD not supported on Windows");
    }
    if (flags_ & ALLOCATOR_MAPPED_FROMFD) {
      // 如果flags_包含ALLOCATOR_MAPPED_FROMFD，抛出不支持的从文件描述符映射错误
      TORCH_CHECK(false, "ALLOCATOR_MAPPED_FROMFD not supported on Windows");
    }

    // 将filename_转换为宽字符的文件名
    const wchar_t *filename;
    const std::wstring wFilename = c10::u8u16(filename_);

    filename = wFilename.c_str();

    /* open file */
    /* FILE_FLAG_RANDOM_ACCESS ? */
    if (flags_) {
      // 如果flags_非零，以读写模式打开文件
      hfile = CreateFileW(filename, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE) {
        // 如果文件句柄无效，抛出带有错误信息和文件名的错误
        TORCH_CHECK(false, "could not open file <", filename_, "> in read-write mode; error code: <", GetLastError(), ">");
      }
    } else {
      // 否则以只读模式打开文件
      hfile = CreateFileW(filename, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE) {
        // 如果文件句柄无效，抛出带有错误信息和文件名的错误
        TORCH_CHECK(false, "could not open file <", filename_, "> in read-only mode; error code: <", GetLastError(), ">");
      }
    }

    if (GetFileSizeEx(hfile, &hfilesz) == 0) {
      // 获取文件大小，如果失败，抛出带有错误信息和文件名的错误
      TORCH_CHECK(false, "could not get file size: <", filename_, ">; error code: <", GetLastError(), ">");
    }
    if (size > 0) {
      // 如果请求的映射大小大于零
      if (size > hfilesz.QuadPart) {
        // 如果请求的映射大小大于当前文件大小
        if (flags_) {
          // 如果标志位为真
          hfilesz.QuadPart = size;
          // 设置文件指针到指定位置
          if (SetFilePointerEx(hfile, hfilesz, NULL, FILE_BEGIN) == 0) {
            // 如果设置文件指针失败，则关闭文件句柄并抛出错误信息
            CloseHandle(hfile);
            TORCH_CHECK(false, "unable to stretch file <", filename_, "> to the right size; error code: <", GetLastError(), ">", filename_);
          }
          // 设置文件结束位置
          if (SetEndOfFile(hfile) == 0) {
            // 如果设置文件结束位置失败，则关闭文件句柄并抛出错误信息
            CloseHandle(hfile);
            TORCH_CHECK(false, "unable to write to file <", filename_, ">; error code: <", GetLastError(), ">");
          }
        } else {
          // 如果标志位为假，关闭文件句柄并抛出错误信息
          CloseHandle(hfile);
          TORCH_CHECK(false, "file <", filename_, "> size <", hfilesz.QuadPart, "> is smaller than the required mapping size <", size, ">; error code: <", GetLastError(), ">");
        }
      }
    } else {
      // 如果请求的映射大小不大于零，则将映射大小设为当前文件大小
      size = hfilesz.QuadPart;
    }

    size_ = size; /* if we are here, it must be the right size */
    // 将 size_ 设为当前映射大小，如果执行到这里，表示映射大小已经是正确的

    hfilesz.QuadPart = size_;
    // 将文件大小设为当前映射大小

    /* get map handle */
    // 获取文件映射句柄
    if (flags_) {
      // 如果标志位为真
      if ( (hmfile = CreateFileMappingW(hfile, NULL, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL ) {
        // 创建可读写的文件映射，如果失败则抛出错误信息
        TORCH_CHECK(false, "could not create a map on file <", filename_, ">; error code: <", GetLastError(), ">");
      }
    } else {
      // 如果标志位为假
      if ( (hmfile = CreateFileMappingW(hfile, NULL, PAGE_WRITECOPY, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL ) {
        // 创建可写拷贝的文件映射，如果失败则抛出错误信息
        TORCH_CHECK(false, "could not create a map on file <", filename_, ">; error code: <", GetLastError(), ">");
      }
    }

    /* map the stuff */
    // 将文件内容映射到内存
    if(flags_) {
      // 如果标志位为真，使用所有访问权限映射文件
      base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    } else {
      // 如果标志位为假，使用拷贝写入权限映射文件
      base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0);
    }

    // 关闭文件句柄和映射句柄
    CloseHandle(hfile);
    CloseHandle(hmfile);
#else /* _WIN32 */
{
  /* 打开文件 */
  int fd{-1};
  int flags{}; // 重复定义的变量

  // 根据标志位设置文件打开方式
  if (flags_ & (ALLOCATOR_MAPPED_SHARED | ALLOCATOR_MAPPED_SHAREDMEM)) {
    flags = O_RDWR | O_CREAT;
  } else {
    flags = O_RDONLY;
  }

  // 如果设置了排他标志，则加上 O_EXCL
  if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
    flags |= O_EXCL;
  }
  // 如果设置了不创建标志，则清除 O_CREAT
  if (flags_ & ALLOCATOR_MAPPED_NOCREATE) {
    flags &= ~O_CREAT;
  }

  // 如果没有指定从现有文件描述符打开，则根据不同的共享模式选择打开方式
  if (!(flags_ & ALLOCATOR_MAPPED_FROMFD)) {
    if (flags_ & ALLOCATOR_MAPPED_SHARED) {
      // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
      // 尝试以读写模式打开文件，如果失败则报错
      if ((fd = open(filename_.c_str(), flags, (mode_t)0600)) == -1) {
        TORCH_CHECK(false, "unable to open file <", filename_, "> in read-write mode: ", strerror(errno), " (", errno, ")");
      }
    } else if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_OPEN
      // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
      // 尝试以共享内存模式打开文件，如果失败则报错
      if((fd = shm_open(filename_.c_str(), flags, (mode_t)0600)) == -1) {
        TORCH_CHECK(false, "unable to open shared memory object <", filename_, "> in read-write mode: ", strerror(errno), " (", errno, ")");
      }
#else
      // 平台不支持共享内存模式时报错
      TORCH_CHECK(false, "unable to open file <", filename_, "> in sharedmem mode, shm_open unavailable on this platform");
#endif
    } else {
      // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
      // 尝试以只读模式打开文件，如果失败则报错
      if ((fd = open(filename_.c_str(), O_RDONLY)) == -1) {
        TORCH_CHECK(false, "unable to open file <", filename_, "> in read-only mode: ", strerror(errno), " (", errno, ")");
      }
    }
  } else {
    // 如果指定了从现有文件描述符打开，则直接使用该文件描述符
    fd = fd_;
  }

  // 获取文件状态信息
  struct stat file_stat{};
  if (fstat(fd, &file_stat) == -1) {
    int last_err = errno;
    // 如果不是从现有文件描述符打开的，则关闭文件
    if (!(flags_ & ALLOCATOR_MAPPED_FROMFD)) {
      ::close(fd);
    }
    // 报错无法获取文件状态信息
    TORCH_CHECK(false, "unable to stat the file <", filename_, ">: ", strerror(last_err), " (", last_err, ")");
  }

  // 如果需要指定文件大小，并且文件当前大小小于所需大小，则尝试调整文件大小
  if (size > 0) {
    if (static_cast<int64_t>(size) > file_stat.st_size) {
      if (flags_) {
        // 尝试调整文件大小，如果失败则报错
        if (ftruncate(fd, static_cast<off_t>(size)) == -1) {
          TORCH_CHECK(false, "unable to resize file <", filename_, "> to the right size: ", strerror(errno), " (", errno, ")");
        }
        // 再次获取文件状态信息，确保文件大小已调整到所需大小
        if (fstat(fd, &file_stat) == -1 || file_stat.st_size < static_cast<int64_t>(size)) {
          int last_err = errno;
          ::close(fd);
          // 如果无法调整文件大小到所需大小，则报错
          TORCH_CHECK(false, "unable to stretch file <", filename_, "> to the right size: ", strerror(last_err), " (", last_err, ")");
        }
/* 在 macOS 上，使用从 shm_open 获取的文件描述符时，如果使用 write 函数会返回错误 45 (Operation not supported) */
#ifndef __APPLE__
        // 尝试写入一个字节到文件中，以验证文件描述符的可用性
        if ((write(fd, "", 1)) != 1) /* 注意：空字符串 "" 包含 '\0' 字节 */ {
          int last_err = errno;
          ::close(fd);
          // 如果无法写入文件，则报错
          TORCH_CHECK(false, "unable to write to file <", filename_, ">: ", strerror(last_err), " (", last_err, ")");
        }
#endif
      }
    }
  }
#endif
        } else {
          ::close(fd);
          // 关闭文件描述符，因为文件大小不满足映射需求，抛出错误信息
          TORCH_CHECK(false, "file <", filename_, "> size <",  file_stat.st_size, "> is smaller than the required mapping size <", size, ">");
        }
      }
    } else {
      size = file_stat.st_size;
    }

    size_ = static_cast<ptrdiff_t>(size); /* if we are here, it must be the right size */

    /* map it */
    // 根据 flags_ 中的标志选择映射方式：共享映射或私有映射
    if (flags_ & (ALLOCATOR_MAPPED_SHARED | ALLOCATOR_MAPPED_SHAREDMEM)) {
      base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    } else {
      base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
    }

    if (base_ptr_ == MAP_FAILED) {
      base_ptr_ = nullptr; /* let's be sure it is NULL */
      // 如果映射失败，将 base_ptr_ 设置为 nullptr，并抛出错误信息
      TORCH_CHECK(false, "unable to mmap ", size_, " bytes from file <", filename_, ">: ", strerror(errno), " (", errno, ")");
    }

#if !defined(__APPLE__) && !defined(__ANDROID__)
    /* attempt to use larger block size on Linux, which is important for getting better CUDA upload speed */
    // 在 Linux 上尝试使用较大的块大小，这对提高 CUDA 上传速度很重要
    posix_fadvise(fd, 0, static_cast<off_t>(size), POSIX_FADV_SEQUENTIAL);
#endif

    if (flags_ & ALLOCATOR_MAPPED_KEEPFD) {
      fd_ = fd;
    } else {
      if (::close(fd) == -1) {
        // 如果关闭文件描述符失败，抛出错误信息
        TORCH_CHECK(false, "Error closing file <", filename_, ">: ", strerror(errno), " (", errno, ")");
      }
      fd_ = -1;
    }

    if (flags_ & ALLOCATOR_MAPPED_UNLINK) {
      // 如果设置了 ALLOCATOR_MAPPED_UNLINK 标志，则删除映射的文件
      if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_UNLINK
        // 如果平台支持 shm_unlink，则删除共享内存文件
        if (shm_unlink(filename_.c_str()) == -1) {
          TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, " : ", strerror(errno), " (", errno, ")");
        }
#else
        // 如果平台不支持 shm_unlink，则抛出错误信息
        TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif
      } else {
        // 如果是私有映射，直接删除文件
        if (unlink(filename_.c_str()) == -1)
          TORCH_CHECK(false, "could not unlink file ", filename_, " : ", strerror(errno), " (", errno, ")");
      }
    }

    if (base_ptr_ == MAP_FAILED) {
      // 如果映射失败，抛出错误信息，并提示尝试映射的内存大小
      TORCH_CHECK(false, "$ Torch: unable to mmap memory: you tried to mmap ", size_/1073741824, " GB.");
    }
  }
#endif
  // 将映射的内存信息报告给性能分析器
  c10::reportMemoryUsageToProfiler(base_ptr_, size_, 0, size_, c10::Device(c10::DeviceType::CPU));
}

// 构造函数：从文件名、文件描述符、标志和大小构造映射分配器
MapAllocator::MapAllocator(c10::string_view filename, int flags, size_t size)
  : MapAllocator(WITH_FD, filename, -1, flags, size)
{}

#ifdef _WIN32
// 释放上下文结构体定义，用于等待释放句柄
struct ReleaseContext {
  HANDLE event;
  HANDLE handle;
  HANDLE wait;
};
// 等待释放句柄的回调函数
static void CALLBACK WaitForReleaseHandle(PVOID lpParam, BOOLEAN TimerOrWaitFired)
{
  if (lpParam) {
    ReleaseContext *ctx = (ReleaseContext *)lpParam;

    SetEvent(ctx->event);
    CloseHandle(ctx->event);
    CloseHandle(ctx->handle);

    UnregisterWait(ctx->wait);

    delete ctx;
  }
}
#endif

// 关闭映射分配器
void MapAllocator::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  // 如果 base_ptr_ 为空，则无需关闭
  if (base_ptr_ == nullptr) {
    return;
  }
#ifdef _WIN32
  // 如果设置了 ALLOCATOR_MAPPED_KEEPFD 或 ALLOCATOR_MAPPED_SHAREDMEM 标志，则在 Windows 下不关闭
  if ((flags_ & ALLOCATOR_MAPPED_KEEPFD) || (flags_ & ALLOCATOR_MAPPED_SHAREDMEM))
    // 关闭句柄 `handle_`
    CloseHandle(handle_);
  // 尝试取消映射基地址为 `base_ptr_` 的共享内存，如果失败则抛出错误信息
  if(UnmapViewOfFile(base_ptr_) == 0)
    TORCH_CHECK(false, "could not unmap the shared memory file");
#else /* _WIN32 */
// 如果未定义 _WIN32，执行以下代码块

  // 检查是否设置了 ALLOCATOR_MAPPED_KEEPFD 标志位
  if (flags_ & ALLOCATOR_MAPPED_KEEPFD) {
    // 尝试关闭文件描述符 fd_
    if (::close(fd_) == -1) {
      // 若关闭失败，输出错误信息
      TORCH_CHECK(false, "could not close file descriptor ", fd_, " :", strerror(errno), " (", errno, ")" );
    }
  }

  // 尝试解除映射 base_ptr_ 指向的内存区域
  if (munmap(base_ptr_, size_)) {
    // 若解除映射失败，输出错误信息
    TORCH_CHECK(false, "could not unmap the shared memory file: ", strerror(errno), " (", errno, ")");
  }

  // 若未设置 ALLOCATOR_MAPPED_FROMFD 或 ALLOCATOR_MAPPED_UNLINK 标志位
  if (!(flags_ & (ALLOCATOR_MAPPED_FROMFD | ALLOCATOR_MAPPED_UNLINK))) {
    // 若设置了 ALLOCATOR_MAPPED_SHAREDMEM 标志位
    if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_UNLINK
      // 如果平台支持 shm_unlink 函数
      if (shm_unlink(filename_.c_str()) == -1) {
        // 尝试删除共享内存文件，输出错误信息
        TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, " : ", strerror(errno), " (", errno, ")");
      }
#else
      // 若平台不支持 shm_unlink 函数，输出错误信息
      TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif
    }
  }
#endif /* _WIN32 */
}
#endif

// 检查是否设置了 ALLOCATOR_MAPPED_EXCLUSIVE 标志位
if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
  // 如果设置了，使用原位构造方式创建原子引用计数
  new (&map_info->refcount) std::atomic<int>(1);
} else {
  // 否则，增加引用计数
  map_info->refcount++;
}
}

void RefcountedMapAllocator::close() {
  // 如果已经关闭，则直接返回
  if (closed_) {
    return;
  }
  // 标记为已关闭状态
  closed_ = true;

  // 获取基址指针
  void* data = base_ptr_;

#ifdef _WIN32
  // 在 Windows 下，将指针转换为 MapInfo 类型
  MapInfo *info = (MapInfo*)data;
  // 减少引用计数，并在引用计数为零时设置事件
  if (--info->refcount == 0) {
    SetEvent(event_);
  }
  // 尝试取消映射共享内存文件，如果失败则抛出错误
  if(UnmapViewOfFile(data) == 0) {
    TORCH_CHECK(false, "could not unmap the shared memory file");
  }
#else /* _WIN32 */

  // 在非 Windows 平台下，同样将指针转换为 MapInfo 类型
  MapInfo *info = (MapInfo*)(data);
  // 减少引用计数，并在引用计数为零时处理共享内存文件
  if (--info->refcount == 0) {
#ifdef HAVE_SHM_UNLINK
    // 如果平台支持 shm_unlink，则尝试删除共享内存文件
    if (shm_unlink(filename_.c_str()) == -1) {
      TORCH_CHECK(false, "could not unlink the shared memory file ", filename_);
    }
#else
    // 否则抛出错误，提示平台不支持 shm_unlink
    TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif /* HAVE_SHM_UNLINK */
  }
  // 尝试解除映射共享内存，如果失败则抛出错误
  if (munmap(info, size_)) {
    TORCH_CHECK(false, "could not unmap the shared memory file ", filename_);
  }
#endif /* _WIN32 */
}

void RefcountedMapAllocator::incref()
{
  // 增加引用计数
  MapInfo *map_info = static_cast<MapInfo*>(base_ptr_);
  ++map_info->refcount;
}

int RefcountedMapAllocator::decref()
{
  // 减少引用计数，并返回是否引用计数为零的结果
  MapInfo *map_info = static_cast<MapInfo*>(base_ptr_);
  return --map_info->refcount == 0;
}

#else

// 对于未定义部分的注释

RefcountedMapAllocatorArgCheck::RefcountedMapAllocatorArgCheck(int flags) {}

RefcountedMapAllocator::RefcountedMapAllocator(const char *filename, int flags, size_t size)
  : RefcountedMapAllocatorArgCheck(flags),
    MapAllocator(filename, flags, size + map_alloc_alignment)
{
  // 提示不支持在当前系统上进行引用计数文件映射
  TORCH_CHECK(false, "refcounted file mapping not supported on your system");
}

RefcountedMapAllocator::RefcountedMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size)
  : RefcountedMapAllocatorArgCheck(flags),
    MapAllocator(WITH_FD, filename, flags, fd, size + map_alloc_alignment)
{
  // 提示不支持在当前系统上进行引用计数文件映射
  TORCH_CHECK(false, "refcounted file mapping not supported on your system");
}

void RefcountedMapAllocator::initializeAlloc() {}

void RefcountedMapAllocator::close() {}

#endif

// 以下是用于释放内存分配器的静态函数
static void deleteMapAllocator(void* ptr) {
  delete static_cast<MapAllocator*>(ptr);
}

// 以下是用于释放引用计数内存分配器的静态函数
static void deleteRefcountedMapAllocator(void* ptr) {
  delete static_cast<RefcountedMapAllocator*>(ptr);
}

// 根据数据指针创建 MapAllocator 实例
MapAllocator* MapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<MapAllocator>(&deleteMapAllocator);
}

// 根据数据指针创建 RefcountedMapAllocator 实例
RefcountedMapAllocator* RefcountedMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<RefcountedMapAllocator>(&deleteRefcountedMapAllocator);
}

// 创建数据指针，返回一个 MapAllocator 实例
at::DataPtr MapAllocator::makeDataPtr(c10::string_view filename, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new MapAllocator(filename, flags, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteMapAllocator, at::DeviceType::CPU};
}
// 创建一个指向 MapAllocator 对象的 DataPtr，并返回它
at::DataPtr MapAllocator::makeDataPtr(WithFd, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out) {
  // 使用带有文件描述符的构造函数创建 MapAllocator 上下文对象
  auto* context = new MapAllocator(WITH_FD, filename, fd, flags, size);
  // 如果传入了 actual_size_out 指针，则设置其值为上下文对象的大小
  if (actual_size_out) *actual_size_out = context->size();
  // 返回 DataPtr 对象，包括指向数据的指针、上下文对象指针以及删除函数指针和设备类型
  return {context->data(), context, &deleteMapAllocator, at::DeviceType::CPU};
}

// 创建一个指向 RefcountedMapAllocator 对象的 DataPtr，并返回它
at::DataPtr RefcountedMapAllocator::makeDataPtr(const char *filename, int flags, size_t size, size_t* actual_size_out) {
  // 使用文件名、标志和大小创建 RefcountedMapAllocator 上下文对象
  auto* context = new RefcountedMapAllocator(filename, flags, size);
  // 如果传入了 actual_size_out 指针，则设置其值为上下文对象的大小减去内存对齐值
  if (actual_size_out) *actual_size_out = context->size() - map_alloc_alignment;
  // 返回 DataPtr 对象，包括指向数据的指针、上下文对象指针以及删除函数指针和设备类型
  return {context->data(), context, &deleteRefcountedMapAllocator, at::DeviceType::CPU};
}

// 创建一个指向 RefcountedMapAllocator 对象的 DataPtr，并返回它（带有文件描述符）
at::DataPtr RefcountedMapAllocator::makeDataPtr(WithFd, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out) {
  // 使用带有文件描述符的构造函数创建 RefcountedMapAllocator 上下文对象
  auto* context = new RefcountedMapAllocator(WITH_FD, filename, fd, flags, size);
  // 如果传入了 actual_size_out 指针，则设置其值为上下文对象的大小减去内存对齐值
  if (actual_size_out) *actual_size_out = context->size() - map_alloc_alignment;
  // 返回 DataPtr 对象，包括指向数据的指针、上下文对象指针以及删除函数指针和设备类型
  return {context->data(), context, &deleteRefcountedMapAllocator, at::DeviceType::CPU};
}

// 返回 RefcountedMapAllocator 对象的数据指针，考虑内存对齐
void* RefcountedMapAllocator::data() const {
  return static_cast<void*>(static_cast<char*>(base_ptr_) + map_alloc_alignment);
}

// MapAllocator 类的析构函数
MapAllocator::~MapAllocator() {
  // 调用 close 方法关闭 MapAllocator
  MapAllocator::close();
  // 向分析器报告内存使用情况，传入基础指针、负的大小、和其他参数
  c10::reportMemoryUsageToProfiler(base_ptr_, -size_, 0, 0, c10::Device(c10::DeviceType::CPU));
}
```