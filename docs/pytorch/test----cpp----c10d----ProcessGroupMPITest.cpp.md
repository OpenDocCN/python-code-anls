# `.\pytorch\test\cpp\c10d\ProcessGroupMPITest.cpp`

```
// 包含系统头文件 <unistd.h>，提供对 POSIX 操作系统 API 的访问
// 包含 C10 库的头文件 <c10/util/irange.h>，提供对 C++ 10 中的 irange 功能的访问
// 包含 Torch 分布式组件中 MPI 实现的头文件 <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <unistd.h>

#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>

// 包含标准库头文件
#include <cstdlib>      // 提供杂项的 C 语言函数
#include <iostream>     // 提供标准输入输出流类
#include <sstream>      // 提供基于字符串的流功能
#include <string>       // 提供字符串处理功能
#include <thread>       // 提供多线程功能

// 定义宏 STR_HELPER 和 STR，用于将参数转换为字符串
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// 等待工作完成，返回输出张量的向量
std::vector<std::vector<at::Tensor>> waitWork(
    c10::intrusive_ptr<::c10d::ProcessGroupMPI> pg,                   // MPI 进程组的智能指针
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {              // 工作对象的向量
  std::vector<std::vector<at::Tensor>> outputTensors;                 // 输出张量的向量
  for (auto& work : works) {                                          // 遍历每个工作对象
    try {
      work->wait();                                                   // 等待工作对象完成
    } catch (const std::exception& ex) {                              // 捕获可能的异常
      std::cerr << "Exception received: " << ex.what() << std::endl;  // 输出异常信息到标准错误流
      pg->abort();                                                    // 终止 MPI 进程组
    }
    outputTensors.emplace_back(work->result());                       // 将工作对象的结果添加到输出张量的向量中
  }
  return outputTensors;                                               // 返回输出张量的向量
}

// 使用 Futures 等待工作完成，返回输出张量的向量
std::vector<std::vector<at::Tensor>> waitFuture(
    c10::intrusive_ptr<::c10d::ProcessGroupMPI> pg,                   // MPI 进程组的智能指针
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {              // 工作对象的向量
  std::vector<std::vector<at::Tensor>> outputTensors;                 // 输出张量的向量
  for (auto& work : works) {                                          // 遍历每个工作对象
    auto fut = work->getFuture();                                     // 获取工作对象的 Future
    try {
      fut->wait();                                                    // 等待 Future 完成
    } catch (const std::exception& ex) {                              // 捕获可能的异常
      std::cerr << "Exception received: " << ex.what() << std::endl;  // 输出异常信息到标准错误流
      pg->abort();                                                    // 终止 MPI 进程组
    }
    auto result = fut->value();                                       // 获取 Future 的值
    if (result.isNone()) {                                            // 如果值为空
      outputTensors.emplace_back();                                   // 添加空的张量到输出张量的向量中
    } else if (result.isTensorList()) {                               // 如果值为张量列表
      outputTensors.emplace_back(result.toTensorVector());             // 将张量列表添加到输出张量的向量中
    } else {
      TORCH_CHECK(false, "future result should be tensor list or none"); // 抛出错误，未知的 Future 值类型
    }
  }
  return outputTensors;                                               // 返回输出张量的向量
}

// 测试 Allreduce 操作，默认迭代次数为 1000 次
void testAllreduce(int iter = 1000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();           // 创建 MPI 进程组对象

  // 生成输入张量
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;                // 工作对象的向量
  for (const auto i : c10::irange(iter)) {                            // 遍历指定的迭代次数
    auto tensor = at::ones({16, 16}) * i;                             // 创建大小为 16x16 的张量，并初始化为 i
    std::vector<at::Tensor> tensors = {tensor};                       // 将张量放入向量中

    // 将工作对象入队
    c10::intrusive_ptr<::c10d::Work> work = pg->allreduce(tensors);   // 执行 Allreduce 操作，并得到工作对象
    works.push_back(std::move(work));                                 // 将工作对象添加到工作对象的向量中
  }

  auto outputTensors = waitFuture(pg, works);                         // 等待所有工作对象完成，并获取输出张量的向量

  // 获取 MPI 进程组的全局大小
  auto worldSize = pg->getSize();

  // 验证输出张量
  for (const auto i : c10::irange(iter)) {                            // 遍历每个迭代次数
    const auto expected = worldSize * i;                              // 计算预期值
    auto data = outputTensors[i][0].data_ptr<float>();                // 获取输出张量数据的指针
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {          // 遍历张量中的每个元素
      if (data[j] != expected) {                                     
        TORCH_CHECK(false, "BOOM!");                                  // 如果输出张量的元素与预期值不符，则抛出错误
      }
    }
  }
}

// 测试广播操作，默认迭代次数为 10000 次
void testBroadcast(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();           // 创建 MPI 进程组对象
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;                // 工作对象的向量
  for (const auto i : c10::irange(iter)) {                            // 遍历指定的迭代次数
    auto tensors = std::vector<at::Tensor>();                         // 创建张量的向量
    if (pg->getRank() == 0) {                                         // 如果当前进程的排名为 0
      auto tensor = at::ones({16, 16}) * i;                           // 创建大小为 16x16 的张量，并初始化为 i
      tensors = std::vector<at::Tensor>({tensor});                    // 将张量放入向量中
    } else {
      auto tensor = at::zeros({16, 16});                              // 创建大小为 16x16 的零张量
      tensors = std::vector<at::Tensor>({tensor});                    // 将张量放入向量中
    }

    // 将工作对象入队
    c10::intrusive_ptr<::c10d::Work> work = pg->broadcast(tensors);   // 执行广播操作，并得到工作对象
    // 将工作项移动到作业队列的末尾
    works.push_back(std::move(work));
  }

  // 等待并获取输出张量
  auto outputTensors = waitFuture(pg, works);

  // 验证输出结果
  for (const auto i : c10::irange(iter)) {
    // 准备期望的输出值
    const auto expected = i;
    // 获取当前输出张量的数据指针（假设为 float 类型）
    auto data = outputTensors[i][0].data_ptr<float>();
    // 遍历当前张量的所有元素
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      // 检查当前元素是否等于期望的值，若不等则抛出异常
      if (data[j] != expected) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

// 测试函数：执行 MPI 进程组的 reduce 操作
void testReduce(int iter = 10000) {
  // 创建 MPI 进程组
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // 存储所有 Work 的指针的向量
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;
  // 循环执行 iter 次
  for (const auto i : c10::irange(iter)) {
    // 创建一个大小为 {16, 16}，值为 i 的张量
    auto tensor = at::ones({16, 16}) * i;
    auto tensors = std::vector<at::Tensor>({tensor});

    // 将操作加入队列
    c10::intrusive_ptr<::c10d::Work> work = pg->reduce(tensors);
    works.push_back(std::move(work));
  }

  // 等待所有异步操作完成，并获取输出张量
  auto outputTensors = waitFuture(pg, works);

  // 获取 MPI 进程组的总进程数
  auto worldSize = pg->getSize();

  // 如果当前进程的 rank 是 0
  if (pg->getRank() == 0) {
    // 验证输出结果
    for (const auto i : c10::irange(iter)) {
      const auto expected = worldSize * i;
      auto data = outputTensors[i][0].data_ptr<float>();
      // 检查输出张量中的每个元素是否符合预期值
      for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
        if (data[j] != expected) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

// 测试函数：执行 MPI 进程组的 allgather 操作
void testAllgather(int iter = 10000) {
  // 创建 MPI 进程组
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // 存储所有 Work 的指针的向量
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // 获取 MPI 进程组的总进程数和当前进程的 rank
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // 生成输入张量
  for (const auto i : c10::irange(iter)) {
    // 创建一个大小为 {16, 16}，值为 i * rank 的张量
    auto tensor = at::ones({16, 16}) * i * rank;
    auto tensors = std::vector<at::Tensor>({tensor});
    auto outputs = std::vector<std::vector<at::Tensor>>(1);
    outputs[0].resize(worldSize);
    for (const auto j : c10::irange(worldSize)) {
      outputs[0][j] = at::zeros({16, 16});
    }

    // 将操作加入队列
    c10::intrusive_ptr<::c10d::Work> work = pg->allgather(outputs, tensors);
    works.push_back(std::move(work));
  }

  // 等待所有异步操作完成，并获取输出张量
  auto outputTensors = waitFuture(pg, works);

  // 验证输出结果
  for (const auto i : c10::irange(iter)) {
    for (const auto j : c10::irange(worldSize)) {
      const auto expected = i * j;
      auto data = outputTensors[i][j].data_ptr<float>();
      // 检查输出张量中的每个元素是否符合预期值
      for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
        if (data[k] != expected) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

// 测试函数：执行 MPI 进程组的 gather 操作
void testGather(int iter = 10000) {
  // 创建 MPI 进程组
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // 存储所有 Work 的指针的向量
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // 获取 MPI 进程组的总进程数和当前进程的 rank
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // 生成输入张量
  for (const auto i : c10::irange(iter)) {
    // 创建一个大小为 {16, 16}，值为 i * rank 的张量
    auto tensor = at::ones({16, 16}) * i * rank;
    auto tensors = std::vector<at::Tensor>({tensor});
    auto outputs = std::vector<std::vector<at::Tensor>>(0);
    if (rank == 0) {
      outputs = std::vector<std::vector<at::Tensor>>(1);
      outputs[0].resize(worldSize);
      for (const auto j : c10::irange(worldSize)) {
        outputs[0][j] = at::zeros({16, 16});
      }
    }

    // 将操作加入队列
    c10::intrusive_ptr<::c10d::Work> work = pg->gather(outputs, tensors);
    works.push_back(std::move(work));
  }

  // 等待所有异步操作完成，并获取输出张量
  auto outputTensors = waitFuture(pg, works);

  // 如果当前进程的 rank 是 0
  if (rank == 0) {

    // 验证输出结果
    for (const auto i : c10::irange(iter)) {
      for (const auto j : c10::irange(worldSize)) {
        const auto expected = i * j;
        auto data = outputTensors[i][j].data_ptr<float>();
        // 检查输出张量中的每个元素是否符合预期值
        for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
          if (data[k] != expected) {
            TORCH_CHECK(false, "BOOM!");
          }
        }
      }
    }
  }
}
    # 对于每个 `iter` 中的索引 `i` 迭代循环
    for (const auto i : c10::irange(iter)) {
      # 对于每个 `worldSize` 中的索引 `j` 迭代循环
      for (const auto j : c10::irange(worldSize)) {
        # 计算期望的值，即 `i * j`
        const auto expected = i * j;
        # 获取 `outputTensors[i][j]` 的数据指针，数据类型为 float
        auto data = outputTensors[i][j].data_ptr<float>();
        # 遍历 `outputTensors[i][j]` 中的元素数量
        for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
          # 检查当前元素是否等于期望值
          if (data[k] != expected) {
            # 如果不相等，则抛出 TORCH_CHECK 异常，并附带错误信息 "BOOM!"
            TORCH_CHECK(false, "BOOM!");
          }
        }
      }
    }
  } else {
    # 否则，对于每个 `iter` 中的索引 `i` 迭代循环
    for (const auto i : c10::irange(iter)) {
      # 如果 `outputTensors[i]` 的大小不为 0
      if (outputTensors[i].size() != 0) {
        # 抛出 TORCH_CHECK 异常，附带错误信息 "BOOM!"
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

// 定义一个函数 testScatter，用于测试 MPI 进程组的 scatter 操作
void testScatter(int iter = 1) {
  // 创建 MPI 进程组
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // 存储作业的容器
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // 获取进程组的总大小和当前进程的排名
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // 生成输入数据
  for (const auto i : c10::irange(iter)) {
    // 创建一个全零的 Tensor
    auto tensor = at::zeros({16, 16});
    // 将 Tensor 放入向量中
    auto tensors = std::vector<at::Tensor>({tensor});
    // 创建输入向量的向量
    auto inputs = std::vector<std::vector<at::Tensor>>(0);
    // 如果当前进程是排名为 0 的进程
    if (rank == 0) {
      // 重新分配输入向量的向量大小
      inputs = std::vector<std::vector<at::Tensor>>(1);
      // 调整内部向量的大小为世界大小
      inputs[0].resize(worldSize);
      // 填充输入数据
      for (const auto j : c10::irange(worldSize)) {
        inputs[0][j] = at::ones({16, 16}) * i * j;
      }
    }

    // 将作业排队
    c10::intrusive_ptr<::c10d::Work> work = pg->scatter(tensors, inputs);
    // 将作业添加到作业容器中
    works.push_back(std::move(work));
  }

  // 等待作业完成并获取输出张量
  auto outputTensors = waitFuture(pg, works);

  // 验证输出
  for (const auto i : c10::irange(iter)) {
    for (const auto j : c10::irange(worldSize)) {
      const auto expected = i * j;
      // 获取数据指针
      auto data = outputTensors[i][0].data_ptr<float>();
      // 检查每个元素是否符合预期
      for (auto k = 0; k < outputTensors[i][0].numel(); ++k) {
        if (data[k] != expected) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

// 定义一个函数 testSendRecv，用于测试 MPI 进程组的 send 和 recv 操作
void testSendRecv(bool recvAnysource, int iter = 10000) {
  // 创建 MPI 进程组
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // 存储作业的容器
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // pg->send 不会保持发送的张量存活，所以需要手动管理
  // 创建发送张量的向量
  std::vector<std::vector<at::Tensor>> sendTensors(iter);
  // 获取当前进程的排名
  auto rank = pg->getRank();
  // 生成输入数据
  for (const auto i : c10::irange(iter)) {
    // 如果当前进程是排名为 0 的进程
    if (rank == 0) {
      // 创建一个全 1 的 Tensor
      auto tensor = at::ones({16, 16}) * i;
      // 将 Tensor 放入向量中
      sendTensors[i] = std::vector<at::Tensor>({tensor});

      // 将作业排队
      c10::intrusive_ptr<::c10d::Work> work = pg->send(sendTensors[i], 1, 0);
      // 将作业添加到作业容器中
      works.push_back(std::move(work));
    } else {
      // 创建一个全 0 的 Tensor
      auto tensor = at::zeros({16, 16});
      // 创建接收张量的向量
      auto recvTensors = std::vector<at::Tensor>({tensor});

      // 将作业排队
      if (!recvAnysource) {
        c10::intrusive_ptr<::c10d::Work> work = pg->recv(recvTensors, 0, 0);
        works.push_back(std::move(work));
      } else {
        c10::intrusive_ptr<::c10d::Work> work = pg->recvAnysource(recvTensors, 0);
        works.push_back(std::move(work));
      }
    }
  }

  // 等待作业完成并获取输出张量
  auto outputTensors = waitWork(pg, works);

  // 如果当前进程是排名为 0 的进程，则直接返回
  if (rank == 0) {
    return;
  }

  // 如果使用 recvAnysource，则收集源排名
  std::vector<int> srcRanks;
  if (recvAnysource) {
    for (const auto& work : works) {
      srcRanks.push_back(work->sourceRank());
    }
  }

  // 验证输出
  for (const auto i : c10::irange(iter)) {
    // 如果使用 recvAnysource，并且源排名不是 0
    if (recvAnysource && srcRanks[i] != 0) {
      TORCH_CHECK(false, "src rank is wrong for recvAnysource");
    }
    const auto expected = i;
    // 获取数据指针
    auto data = outputTensors[i][0].data_ptr<float>();
    // 检查每个元素是否符合预期
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != expected) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}
void testBackendName() {
  // 创建一个 MPI 进程组对象，并返回指向该对象的智能指针 pg
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // 检查 MPI 进程组的后端名称是否与预期的 MPI 后端名称相同
  if (pg->getBackendName() != std::string(c10d::MPI_BACKEND_NAME)) {
    // 如果不相同，抛出一个异常并显示错误信息 "BOOM!"
    TORCH_CHECK(false, "BOOM!");
  }
}

int main(int argc, char** argv) {
#ifdef MPIEXEC
  // 如果环境变量 OMPI_COMM_WORLD_SIZE 不存在
  if (!std::getenv("OMPI_COMM_WORLD_SIZE")) {
    // 打印消息指示从 MPIEXEC 执行 mpiexec
    std::cout << "Execute mpiexec from: " << STR(MPIEXEC) << std::endl;
    // 替换当前进程并使用 mpiexec 启动新进程
    execl(STR(MPIEXEC), "-np 2", argv[0], (char*)nullptr);
  }

  // 执行一系列 MPI 测试函数
  testAllreduce();
  testBroadcast();
  testReduce();
  testAllgather();
  testGather();
  testScatter();
  testSendRecv(false);
  testSendRecv(true);
  testBackendName();

  // 输出测试成功信息
  std::cout << "Test successful" << std::endl;
#else
  // 如果未定义 MPIEXEC 宏，则输出 MPI 可执行文件未找到，跳过测试的消息
  std::cout << "MPI executable not found, skipping test" << std::endl;
#endif
  // 返回程序退出状态码成功
  return EXIT_SUCCESS;
}
```