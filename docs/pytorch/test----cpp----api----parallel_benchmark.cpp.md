# `.\pytorch\test\cpp\api\parallel_benchmark.cpp`

```
// 引入 Torch 库，包括 Torch 的头文件
#include <torch/torch.h>
// 引入计时相关的标准库
#include <chrono>
// 引入条件变量的标准库
#include <condition_variable>
// 引入互斥锁的标准库
#include <mutex>

// 定义 Baton 类，用于线程同步
class Baton {
 public:
  // 发送信号通知
  void post() {
    std::unique_lock<std::mutex> l(lock_);
    done_ = true;  // 设置完成标志为 true
    cv_.notify_all();  // 唤醒所有等待的线程
  }
  // 等待信号
  void wait() {
    std::unique_lock<std::mutex> l(lock_);
    while (!done_) {  // 如果完成标志为 false，一直等待
      cv_.wait(l);  // 等待条件变量通知
    }
  }

 private:
  std::mutex lock_;  // 互斥锁，用于保护条件变量和完成标志
  std::condition_variable cv_;  // 条件变量，用于线程等待和通知
  bool done_{false};  // 完成标志，表示操作是否完成
};

// 定义函数 AtLaunch_Base，模拟基础情况下的任务执行
void AtLaunch_Base(int32_t numIters) {
  // 定义内部 Helper 结构体，用于执行具体的任务
  struct Helper {
    explicit Helper(int32_t lim) : limit_(lim) {}  // 构造函数，初始化任务上限
    void operator()() {
      if (++val_ == limit_) {  // 如果计数器达到上限
        done.post();  // 发送完成信号
      } else {
        at::launch([this]() { (*this)(); });  // 使用 Torch 的 launch 函数启动任务
      }
    }
    int val_{0};  // 当前执行的任务数量
    int limit_;  // 总共需要执行的任务上限
    Baton done;  // 线程同步工具，用于通知任务完成
  };
  
  Helper h(numIters);  // 创建 Helper 对象
  auto start = std::chrono::system_clock::now();  // 记录开始时间
  h();  // 执行任务
  h.done.wait();  // 等待所有任务完成
  // 输出执行任务的平均时间
  std::cout << "NoData " << static_cast<double>(
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now() - start)
                       .count()) /
                               static_cast<double>(numIters)
            << " usec/each\n";
}

// 定义函数 AtLaunch_WithData，模拟携带数据情况下的任务执行
void AtLaunch_WithData(int32_t numIters, int32_t vecSize) {
  // 定义内部 Helper 结构体，用于执行具体的任务
  struct Helper {
    explicit Helper(int32_t lim) : limit_(lim) {}  // 构造函数，初始化任务上限
    void operator()(std::vector<int32_t> v) {
      if (++val_ == limit_) {  // 如果计数器达到上限
        done.post();  // 发送完成信号
      } else {
        at::launch([this, v = std::move(v)]() { (*this)(v); });  // 使用 Torch 的 launch 函数启动任务，传递数据 v
      }
    }
    int val_{0};  // 当前执行的任务数量
    int limit_;  // 总共需要执行的任务上限
    Baton done;  // 线程同步工具，用于通知任务完成
  };
  
  Helper h(numIters);  // 创建 Helper 对象
  std::vector<int32_t> v(vecSize, 0);  // 创建指定大小的数据向量
  auto start = std::chrono::system_clock::now();  // 记录开始时间
  h(v);  // 执行任务，传入数据向量
  h.done.wait();  // 等待所有任务完成
  // 输出执行任务的平均时间和携带数据的信息
  std::cout << "WithData(" << vecSize << "): " << static_cast<double>(
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now() - start)
                       .count()) /
                                                      static_cast<double>(numIters)
            << " usec/each\n";
}

// 主函数，程序入口
int main(int argc, char** argv) {
  int32_t N = 1000000;  // 定义任务总数
  // 执行基础任务
  AtLaunch_Base(N);
  // 执行不携带数据的任务
  AtLaunch_WithData(N, 0);
  // 执行携带数据大小为 4 的任务
  AtLaunch_WithData(N, 4);
  // 执行携带数据大小为 256 的任务
  AtLaunch_WithData(N, 256);
  return 0;  // 返回正常退出状态
}
```