# `.\pytorch\torch\csrc\distributed\c10d\FileStore.cpp`

```
// 包含 Torch 分布式文件存储相关的头文件
#include <torch/csrc/distributed/c10d/FileStore.hpp>

// 包含操作系统相关的头文件和库
#include <fcntl.h>
#include <sys/stat.h>
#include <cassert>
#include <cstdint>

#ifdef _WIN32
// 如果是在 Windows 下，包含特定的 Windows 头文件和 API
#include <c10/util/win32-headers.h>
#include <fileapi.h>
#include <io.h>
#include <filesystem>
#else
// 如果不是在 Windows 下，包含 POSIX 系统相关的文件操作头文件
#include <sys/file.h>
#include <unistd.h>
#endif

// 包含时间和线程相关的标准头文件
#include <chrono>
#include <cstdio>
#include <thread>
#include <utility>

// 包含 C10 库中的异常处理头文件
#include <c10/util/Exception.h>

// 定义一个宏用于检查系统调用的返回值，如果小于 0 则抛出异常
#define SYSASSERT(rv, ...)                                 \
  if ((rv) < 0) {                                          \
    C10_THROW_ERROR(DistStoreError, std::strerror(errno)); \
  }

#ifdef _WIN32
// 在 Windows 下定义文件锁操作的标志
#define LOCK_EX 0x00000001
#define LOCK_SH 0x00000010
#define LOCK_UN 0x00000100

// Windows 平台下的文件锁实现函数
int flock_(int fd, int op) {
  HANDLE hdl = (HANDLE)_get_osfhandle(fd);
  DWORD low = 1, high = 0;
  OVERLAPPED offset = {0, 0, 0, 0, NULL};

  if ((intptr_t)hdl < 0)
    return -1;

  switch (op) {
    case LOCK_EX:
      if (LockFileEx(hdl, LOCKFILE_EXCLUSIVE_LOCK, 0, low, high, &offset))
        return 0;
      break;
    case LOCK_SH:
      if (LockFileEx(hdl, 0, 0, low, high, &offset))
        return 0;
      break;
    case LOCK_UN:
      if (UnlockFileEx(hdl, 0, low, high, &offset) != 0)
        return 0;
      break;
    default:
      break;
  }
  errno = EINVAL;
  return -1;
}
#endif

namespace c10d {

namespace {

// 封装系统调用的模板函数，处理 EINTR 中断
template <typename F>
auto syscall(F fn) {
  while (true) {
    auto rv = fn();
    if (rv == -1) {
      if (errno == EINTR) {
        continue;
      }
    }
    return rv;
  }
  return typename std::invoke_result_t<F>{-1};
}

// 对文件锁 flock(2) 的 RAII 封装
class Lock {
 public:
  explicit Lock(int fd, int operation) : fd_(fd) {
    flock(operation); // 执行文件锁定操作
  }

  // 析构函数，自动解锁文件
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~Lock() {
    unlock(); // 解锁文件
  }

  // 禁用复制构造函数
  Lock(const Lock& that) = delete;

  // 移动构造函数
  Lock& operator=(Lock&& other) noexcept {
    if (this != &other) {
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  // 移动赋值运算符
  Lock(Lock&& other) noexcept {
    *this = std::move(other);
  }

  // 手动解锁文件
  void unlock() {
    if (fd_ >= 0) {
      flock(LOCK_UN); // 执行文件解锁操作
      fd_ = -1;
    }
  }

 protected:
  int fd_{-1}; // 文件描述符

  // 执行文件锁定操作的实现
  void flock(int operation) {
#ifdef _WIN32
    auto rv = syscall(std::bind(::flock_, fd_, operation));
#else
    auto rv = syscall([this, operation] { return ::flock(fd_, operation); });
#endif
    SYSASSERT(rv, "flock"); // 检查系统调用是否成功，失败则抛出异常
  }
};

// 文件操作类，封装文件的打开和操作
class File {
 public:
  // 构造函数，打开文件并进行文件锁定
  explicit File(
      const std::string& path,
      int flags,
      std::chrono::milliseconds timeout) {
    const auto start = std::chrono::steady_clock::now(); // 记录开始时间点
    while (true) {
#ifdef _WIN32
      // 在 Windows 下执行文件打开操作
      fd_ = syscall(std::bind(
          ::open, path.c_str(), flags | _O_BINARY, _S_IREAD | _S_IWRITE));
#else
      // 在 POSIX 系统下执行文件打开操作
      fd_ = syscall([path, flags] { return ::open(path.c_str(), flags, 0666); });
#endif

      if (fd_ >= 0 || // 如果成功打开文件或者
          std::chrono::steady_clock::now() - start > timeout) { // 超时退出
        break;
      }

      // 休眠一段时间，等待文件可用
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (fd_ < 0) { // 如果文件打开失败
      C10_THROW_ERRNO(DistStoreError, "open ", path);
    }
  }

  // 获取文件描述符的公共方法
  int fd() const {
    return fd_;
  }

 private:
  int fd_{-1}; // 文件描述符
};

} // namespace

} // namespace c10d
      // 否则，使用 syscall 调用 open 函数打开指定路径的文件，通过 lambda 表达式捕获参数
      fd_ = syscall([capture0 = path.c_str(), flags] {
        return ::open(capture0, flags, 0644);
      });
      // 如果文件已经存在或者出错码不是 ENOENT，则终止循环
      // 这里处理文件不存在时的重试情况，用以解决特定问题：https://github.com/pytorch/pytorch/issues/13750
      if (fd_ >= 0 || errno != ENOENT) {
        break;
      }
#ifdef _WIN32
      // 如果父文件夹不存在，则跳过重试
      if (!std::filesystem::exists(std::filesystem::path(path).parent_path())) {
        break;
      }
#endif
      // 计算已经等待的时间
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      // 如果设置了超时时间并且超过了设定的超时时间，则终止循环
      if (timeout != c10d::Store::kNoTimeout && elapsed > timeout) {
        break;
      }
      // 等待一段时间后继续循环
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // 使用 SYSASSERT 宏来确保 open 操作成功，否则输出错误信息
    SYSASSERT(fd_, "open(" + path + ")");
  }

  ~File() {
    // 关闭文件描述符
    ::close(fd_);
  }

  // 获取共享锁
  Lock lockShared() {
    return Lock(fd_, LOCK_SH);
  }

  // 获取独占锁
  Lock lockExclusive() {
    return Lock(fd_, LOCK_EX);
  }

  // 文件定位
  off_t seek(off_t offset, int whence) {
    // 使用 syscall 调用 lseek 函数进行文件定位，通过 lambda 表达式捕获参数
    auto rv =
        syscall([this, offset, whence] { return lseek(fd_, offset, whence); });
    // 确保 lseek 操作成功，否则输出错误信息
    SYSASSERT(rv, "lseek");
    return rv;
  }

  // 获取当前文件位置
  off_t tell() {
    // 使用 syscall 调用 lseek 函数获取当前文件位置
    auto rv = syscall([this] { return lseek(fd_, 0, SEEK_CUR); });
    // 确保 lseek 操作成功，否则输出错误信息
    SYSASSERT(rv, "lseek");
    return rv;
  }

  // 获取文件大小
  off_t size() {
    // 获取当前文件位置
    auto pos = tell();
    // 获取文件大小
    auto size = seek(0, SEEK_END);
    // 恢复文件位置
    seek(pos, SEEK_SET);
    return size;
  }

  // 写入数据
  void write(const void* buf, size_t count) {
    // 循环写入数据，直到写入完所有数据
    while (count > 0) {
      // 使用 syscall 调用 ::write 函数写入数据，通过 lambda 表达式捕获参数
      auto rv =
          syscall([this, buf, count] { return ::write(fd_, buf, count); });
      // 确保写入操作成功，否则输出错误信息
      SYSASSERT(rv, "write");
      // 更新 buf 和 count，准备下一次写入
      buf = (uint8_t*)buf + rv;
      count -= rv;
    }
  }

  // 读取数据
  void read(void* buf, size_t count) {
    // 循环读取数据，直到读取完所有数据
    while (count > 0) {
      // 使用 syscall 调用 ::read 函数读取数据，通过 lambda 表达式捕获参数
      auto rv = syscall([this, buf, count] { return ::read(fd_, buf, count); });
      // 确保读取操作成功，否则输出错误信息
      SYSASSERT(rv, "read");
      // 更新 buf 和 count，准备下一次读取
      buf = (uint8_t*)buf + rv;
      count -= rv;
    }
  }

  // 写入字符串
  void write(const std::string& str) {
    // 获取字符串长度
    uint32_t len = str.size();
    // 确保字符串长度不超过限制
    assert(str.size() <= std::numeric_limits<decltype(len)>::max());
    // 先写入字符串长度
    write(&len, sizeof(len));
    // 再写入字符串内容
    write(str.c_str(), len);
  }

  // 写入字节数据
  void write(const std::vector<uint8_t>& data) {
    // 获取数据长度
    uint32_t len = data.size();
    // 确保数据长度不超过限制
    assert(data.size() <= std::numeric_limits<decltype(len)>::max());
    // 先写入数据长度
    write(&len, sizeof(len));
    // 再写入数据内容
    write(data.data(), len);
  }

  // 读取字符串
  void read(std::string& str) {
    // 读取字符串长度
    uint32_t len = 0;
    read(&len, sizeof(len));
    // 分配缓冲区
    std::vector<uint8_t> buf(len);
    // 读取字符串内容
    read(buf.data(), len);
    // 转换为字符串
    str.assign(buf.begin(), buf.end());
  }

  // 读取字节数据
  void read(std::vector<uint8_t>& data) {
    // 读取数据长度
    uint32_t len = 0;
    read(&len, sizeof(len));
    // 调整 vector 大小
    data.resize(len);
    // 读取数据内容
    read(data.data(), len);
  }

 protected:
  int fd_;
};
    // 根据文件大小与当前位置进行比较，如果不相等，则执行以下操作
    const std::string& deletePrefix) {
  // 获取文件大小
  auto size = file.size();
  // 如果当前位置不等于文件大小
  if (size != pos) {
    // 临时存储键和值的变量
    std::string tmpKey;
    std::vector<uint8_t> tmpValue;
    // 将文件指针移动到指定位置
    file.seek(pos, SEEK_SET);
    // 循环读取文件内容直到到达文件末尾
    while (size > pos) {
      // 读取临时键
      file.read(tmpKey);
      // 读取临时值
      file.read(tmpValue);
      // 检查临时键是否以指定前缀开头，如果是则从缓存中删除
      if (tmpKey.compare(0, deletePrefix.size(), deletePrefix) == 0) {
        cache.erase(tmpKey.substr(deletePrefix.size()));
      } else {
        // 将临时键值对存入缓存
        cache[tmpKey] = std::move(tmpValue);
      }
      // 更新当前位置为文件指针当前位置
      pos = file.tell();
    }
  }
  // 将文件指针移动到文件开头
  file.seek(0, SEEK_SET);
  // 返回更新后的位置
  return pos;
}
} // namespace
std::vector<uint8_t> FileStore::get(const std::string& key) {
  // 构建正则前缀的键
  std::string regKey = regularPrefix_ + key;
  // 记录开始时间
  const auto start = std::chrono::steady_clock::now();
  // 循环直到找到对应键的数据
  while (true) {
    // 获取独占锁
    std::unique_lock<std::mutex> l(activeFileOpLock_);
    // 打开文件并获取共享锁
    File file(path_, O_RDONLY, timeout_);
    auto lock = file.lockShared();
    auto size = file.size();
    // 如果缓存中没有对应键的数据且文件大小等于位置
    if (cache_.count(regKey) == 0 && size == pos_) {
      // 释放共享锁并等待一段时间
      lock.unlock();
      l.unlock();
      // 计算经过的时间
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      // 如果超时，则抛出异常
      if (timeout_ != kNoTimeout && elapsed > timeout_) {
        auto err = c10::str(
            "Timeout waiting for key: ",
            key,
            " after ",
            timeout_.count(),
            " ms");
        TORCH_CHECK(false, err);
      }
      // 等待一段时间后继续循环
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    // 刷新数据并更新位置
    pos_ = refresh(file, pos_, cache_, deletePrefix_);
    // 如果缓存中存在对应键的数据，则返回该数据
    if (cache_.count(regKey) != 0) {
      return cache_[regKey];
    }
  }
}

int64_t FileStore::addHelper(const std::string& key, int64_t i) {
  // 获取独占锁
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  // 打开文件并获取独占锁
  File file(path_, O_RDWR | O_CREAT, timeout_);
  auto lock = file.lockExclusive();
  // 刷新数据并更新位置
  pos_ = refresh(file, pos_, cache_, deletePrefix_);

  // 获取缓存中对应键的值
  const auto& value = cache_[key];
  int64_t ti = i;
  // 如果值不为空，则将新值与原值相加
  if (!value.empty()) {
    auto buf = reinterpret_cast<const char*>(value.data());
    auto len = value.size();
    ti += std::stoll(std::string(buf, len));
  }
  // 总是定位到文件末尾进行写入
  file.seek(0, SEEK_END);
  // 文件指针现在位于文件末尾，且已获取独占锁，可以写入新值
  file.write(key);
  file.write(std::to_string(ti));
  return ti;
}

int64_t FileStore::add(const std::string& key, int64_t value) {
  // 构建正则前缀的键
  std::string regKey = regularPrefix_ + key;
  return addHelper(regKey, value);
}

int64_t FileStore::getNumKeys() {
  // 获取独占锁
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  // 打开文件并获取共享锁
  File file(path_, O_RDONLY, timeout_);
  auto lock = file.lockShared();
  // 刷新数据并更新位置
  pos_ = refresh(file, pos_, cache_, deletePrefix_);
  // 返回缓存中键的数量
  return static_cast<int64_t>(cache_.size());
}

bool FileStore::deleteKey(const std::string& key) {
  // 构建删除前缀的键
  std::string deleteKey = deletePrefix_ + regularPrefix_ + key;
  // 获取独占锁
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  // 打开文件并获取独占锁
  File file(path_, O_RDWR, timeout_);
  auto lock = file.lockExclusive();
  // 定位到文件末尾并写入删除键和空值
  file.seek(0, SEEK_END);
  file.write(deleteKey);
  file.write(std::vector<uint8_t>{});
  return true;
}

bool FileStore::check(const std::vector<std::string>& keys) {
  // 获取独占锁
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  // 打开文件并获取共享锁
  File file(path_, O_RDONLY, timeout_);
  auto lock = file.lockShared();
  // 刷新数据并更新位置
  pos_ = refresh(file, pos_, cache_, deletePrefix_);

  // 遍历传入的键列表
  for (const auto& key : keys) {
    // 将传入的key与regularPrefix_拼接，形成完整的注册键名
    std::string regKey = regularPrefix_ + key;
    // 检查cache_中是否存在以regKey为键的项
    if (cache_.count(regKey) == 0) {
      // 如果cache_中不存在以regKey为键的项，则返回false
      return false;
    }
  }

  // 循环结束后，表示所有的regKey都存在于cache_中，返回true
  return true;
}

void FileStore::wait(const std::vector<std::string>& keys) {
  // 调用带有默认超时的 wait 函数重载
  wait(keys, timeout_);
}

void FileStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  // 由于在许多共享文件系统（如 NFS）上不支持 inotify，因此不使用它。
  // 记录等待开始的时间点
  const auto start = std::chrono::steady_clock::now();
  // 循环直到所有指定的键都通过检查函数返回 true
  while (!check(keys)) {
    // 计算已经等待的时间
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    // 如果设置了超时时间且已经超过超时时间，则抛出错误
    if (timeout != kNoTimeout && elapsed > timeout) {
      TORCH_CHECK(false, "Wait timeout");
    }

    /* sleep override */
    // 线程休眠一段时间，防止忙等
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace c10d
```