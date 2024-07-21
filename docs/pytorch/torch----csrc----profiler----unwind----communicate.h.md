# `.\pytorch\torch\csrc\profiler\unwind\communicate.h`

```py
#pragma once
#include <ext/stdio_filebuf.h> // 包含使用 __gnu_cxx 扩展的 stdio_filebuf 头文件
#include <sys/wait.h>          // 包含等待进程结束的头文件
#include <torch/csrc/profiler/unwind/unwind_error.h> // 包含 Torch 中的错误处理头文件
#include <unistd.h>            // 包含 POSIX 系统调用的头文件
#include <memory>              // 包含内存管理相关的头文件

namespace torch::unwind {
// 用于创建能够与子进程进行通信的结构体
struct Communicate {
  Communicate(const char* command, const char** args) {
    // 创建管道用于子进程的标准输入、输出、错误流
    if (pipe(inpipe_) < 0 || pipe(outpipe_) < 0 || pipe(errpipe_) < 0) {
      throw UnwindError("pipe() failed"); // 如果管道创建失败则抛出异常
    }
    // 创建子进程
    pid_t pid = fork();
    if (pid < 0) {
      throw UnwindError("fork() failed"); // 如果 fork 失败则抛出异常
    } else if (pid == 0) { // 子进程
      close(inpipe_[1]);   // 关闭子进程中不需要的管道写入端
      close(outpipe_[0]);  // 关闭子进程中不需要的管道读取端
      close(errpipe_[0]);  // 关闭子进程中不需要的错误管道读取端

      // 重定向标准输入、输出、错误流到管道
      dup2(inpipe_[0], STDIN_FILENO);
      dup2(outpipe_[1], STDOUT_FILENO);
      dup2(errpipe_[1], STDERR_FILENO);
      // 执行给定命令和参数的新程序
      execvp(command, (char* const*)args);
      throw UnwindError("failed execvp"); // 如果执行失败则抛出异常
    } else { // 父进程
      close(inpipe_[0]);   // 关闭父进程中不需要的管道读取端
      close(outpipe_[1]);  // 关闭父进程中不需要的管道写入端
      close(errpipe_[1]);  // 关闭父进程中不需要的错误管道写入端

      // 使用 stdio_filebuf 初始化管道的输入输出流
      outbuf_.reset(
          new __gnu_cxx::stdio_filebuf<char>(inpipe_[1], std::ios::out));
      inbuf_.reset(
          new __gnu_cxx::stdio_filebuf<char>(outpipe_[0], std::ios::in));
      errbuf_.reset(
          new __gnu_cxx::stdio_filebuf<char>(errpipe_[0], std::ios::in));

      // 创建输入、输出、错误流对象
      in_.reset(new std::istream(inbuf_.get()));
      out_.reset(new std::ostream(outbuf_.get()));
      err_.reset(new std::ostream(errbuf_.get()));
    }
  }

  // 析构函数，关闭所有管道
  ~Communicate() {
    close(inpipe_[1]);   // 关闭管道写入端
    close(outpipe_[0]);  // 关闭管道读取端
    close(errpipe_[0]);  // 关闭错误管道读取端
  }

  // 返回输出流对象的引用
  std::ostream& out() {
    return *out_;
  }

  // 返回错误输出流对象的引用
  std::ostream& err() {
    return *err_;
  }

  // 返回输入流对象的引用
  std::istream& in() {
    return *in_;
  }

 private:
  int inpipe_[2];                         // 标准输入管道
  int outpipe_[2];                        // 标准输出管道
  int errpipe_[2];                        // 标准错误输出管道
  std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> outbuf_, inbuf_, errbuf_; // stdio_filebuf 对象指针
  std::unique_ptr<std::istream> in_;       // 输入流对象指针
  std::unique_ptr<std::ostream> out_;      // 输出流对象指针
  std::unique_ptr<std::ostream> err_;      // 错误输出流对象指针
};

} // namespace torch::unwind
```