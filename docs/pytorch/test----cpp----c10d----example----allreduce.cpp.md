# `.\pytorch\test\cpp\c10d\example\allreduce.cpp`

```
// 引入头文件，用于包含所需的库和模块
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

// 使用 c10d 命名空间，简化对应用程序的访问
using namespace ::c10d;

// 主函数，程序的入口点
int main(int argc, char** argv) {
  // 从环境变量中获取进程的排名和总进程数，并将其转换为整数
  int rank = atoi(getenv("RANK"));
  int size = atoi(getenv("SIZE"));

  // 创建一个 FileStore 实例，用于分布式进程组之间的通信，存储路径为 "/tmp/c10d_example"，总进程数为 size
  auto store = c10::make_intrusive<FileStore>("/tmp/c10d_example", size);

  // 使用 Gloo 后端创建一个 ProcessGroupGloo 实例，用于管理进程组通信
  ProcessGroupGloo pg(store, rank, size);

  // 创建一些张量
  const auto ntensors = 10;
  std::vector<at::Tensor> tensors;
  for (const auto i : c10::irange(ntensors)) {
    // 创建具有特定形状和数据类型的张量，并将其添加到向量中
    auto x =
        at::ones({1000, 16 * (i + 1)}, at::TensorOptions(at::CPU(at::kFloat)));
    tensors.push_back(x);
  }

  // 启动任务
  std::vector<c10::intrusive_ptr<Work>> pending;
  for (const auto i : c10::irange(ntensors)) {
    // 对张量执行全局归约操作，并将返回的工作对象添加到待处理列表中
    std::vector<at::Tensor> tmp = {tensors[i]};
    pending.push_back(pg.allreduce(tmp));
  }

  // 等待所有任务完成
  for (auto& work : pending) {
    work->wait();  // 等待工作完成
  }
}
```