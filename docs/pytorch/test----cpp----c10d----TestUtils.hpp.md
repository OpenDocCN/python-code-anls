# `.\pytorch\test\cpp\c10d\TestUtils.hpp`

```py
#pragma once

#ifndef _WIN32
#include <signal.h>  // 包含信号处理相关的头文件
#include <sys/wait.h>  // 包含等待进程状态改变的头文件
#include <unistd.h>  // 包含 UNIX 系统调用函数的头文件
#endif

#include <sys/types.h>  // 包含数据类型和结构的头文件
#include <cstring>  // 包含字符串处理函数的头文件

#include <condition_variable>  // 包含条件变量的头文件
#include <mutex>  // 包含互斥量的头文件
#include <string>  // 包含字符串处理的头文件
#include <system_error>  // 包含系统错误处理的头文件
#include <vector>  // 包含向量容器的头文件

namespace c10d {
namespace test {

class Semaphore {
 public:
  void post(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);  // 获取互斥量的唯一所有权
    n_ += n;  // 增加信号量的计数
    cv_.notify_all();  // 唤醒所有等待线程
  }

  void wait(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);  // 获取互斥量的唯一所有权
    while (n_ < n) {  // 如果信号量计数小于指定值
      cv_.wait(lock);  // 等待条件变量的通知，释放互斥量的锁
    }
    n_ -= n;  // 减少信号量的计数
  }

 protected:
  int n_ = 0;  // 信号量计数
  std::mutex m_;  // 互斥量
  std::condition_variable cv_;  // 条件变量
};

#ifdef _WIN32
std::string autoGenerateTmpFilePath() {
  char tmp[L_tmpnam_s];  // 创建一个用于临时文件路径的缓冲区
  errno_t err;
  err = tmpnam_s(tmp, L_tmpnam_s);  // 自动生成临时文件路径名
  if (err != 0)
  {
    throw std::system_error(errno, std::system_category());  // 抛出系统错误异常
  }
  return std::string(tmp);  // 返回生成的临时文件路径名
}

std::string tmppath() {
  const char* tmpfile = getenv("TMPFILE");  // 获取环境变量 TMPFILE 的值
  if (tmpfile) {
    return std::string(tmpfile);  // 返回 TMPFILE 的值作为临时文件路径
  }
  else {
    return autoGenerateTmpFilePath();  // 自动生成临时文件路径
  }
}
#else
std::string tmppath() {
  // TMPFILE 用于手动测试执行，用户将指定完整的临时文件路径
  const char* tmpfile = getenv("TMPFILE");  // 获取环境变量 TMPFILE 的值
  if (tmpfile) {
    return std::string(tmpfile);  // 返回 TMPFILE 的值作为临时文件路径
  }

  const char* tmpdir = getenv("TMPDIR");  // 获取环境变量 TMPDIR 的值
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";  // 默认临时文件目录为 /tmp
  }

  // 创建模板
  std::vector<char> tmp(256);  // 创建用于临时文件路径的字符向量
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);  // 根据模板创建临时文件路径名
  tmp.resize(len);  // 调整向量大小以适应实际路径名长度

  // 创建临时文件
  auto fd = mkstemp(&tmp[0]);  // 创建临时文件并获取文件描述符
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());  // 抛出系统错误异常
  }
  close(fd);  // 关闭临时文件描述符
  return std::string(tmp.data(), tmp.size());  // 返回创建的临时文件路径名
}
#endif

bool isTSANEnabled() {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");  // 获取环境变量 PYTORCH_TEST_WITH_TSAN 的值
  return s && strcmp(s, "1") == 0;  // 判断是否启用 ThreadSanitizer
}
struct TemporaryFile {
  std::string path;  // 临时文件路径

  TemporaryFile() {
    path = tmppath();  // 创建临时文件路径
  }

  ~TemporaryFile() {
    unlink(path.c_str());  // 删除临时文件
  }
};

#ifndef _WIN32
struct Fork {
  pid_t pid;  // 进程 ID

  Fork() {
    pid = fork();  // 创建子进程
    if (pid < 0) {
      throw std::system_error(errno, std::system_category(), "fork");  // 抛出系统错误异常
    }
  }

  ~Fork() {
    if (pid > 0) {
      kill(pid, SIGKILL);  // 杀死子进程
      waitpid(pid, nullptr, 0);  // 等待子进程结束
    }
  }

  bool isChild() {
    return pid == 0;  // 判断当前进程是否为子进程
  }
};
#endif

} // namespace test
} // namespace c10d
```