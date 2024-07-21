# `.\pytorch\test\cpp\c10d\ProcessGroupNCCLTest.cpp`

```
#include <chrono> // 包含时间库，用于处理时间相关操作
#include <iostream> // 包含标准输入输出流库，用于控制台输入输出

#include <torch/csrc/distributed/c10d/FileStore.hpp> // 包含分布式文件存储相关头文件
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp> // 包含基于NCCL的进程组相关头文件
#include "CUDATest.hpp" // 包含CUDA测试相关头文件
#include "TestUtils.hpp" // 包含测试工具相关头文件
#include "c10d/Types.hpp" // 包含分布式通信相关类型定义头文件

#include <c10/cuda/CUDAGuard.h> // 包含CUDA设备保护相关头文件
#include <c10/cuda/CUDAStream.h> // 包含CUDA流相关头文件
#include <c10/util/irange.h> // 包含整数范围遍历相关头文件

#include <gtest/gtest.h> // 包含Google测试框架相关头文件
#include <torch/csrc/autograd/profiler.h> // 包含自动求导性能分析器相关头文件

using namespace c10d::test; // 使用c10d::test命名空间

using at::cuda::CUDAStream; // 使用at::cuda::CUDAStream类

class NCCLTestBase {
 public:
  NCCLTestBase(
      const std::string& path,
      const std::chrono::milliseconds pgTimeout =
          c10d::kProcessGroupNCCLDefaultTimeout)
      : path_(path), pgTimeout_(pgTimeout) {} // 构造函数，初始化path_和pgTimeout_

  NCCLTestBase(NCCLTestBase&& other) { // 移动构造函数，从另一个NCCLTestBase对象移动成员变量
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  std::shared_ptr<::c10d::ProcessGroupNCCL> getProcessGroup() { // 返回NCCL进程组的共享指针
    return pg_;
  }

  ::c10::intrusive_ptr<::c10d::Store>& getProcessGroupStore() { // 返回进程组存储的引用
    return store_;
  }

  void initialize(
      int rank,
      int size,
      std::optional<::std::shared_ptr<::c10d::ProcessGroupNCCL>> split_from =
          c10::nullopt) { // 初始化函数，设置进程组和存储
    store_ = c10::make_intrusive<::c10d::FileStore>(path_, size); // 创建文件存储对象

    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
        c10::make_intrusive<c10d::ProcessGroupNCCL::Options>(); // 创建NCCL进程组选项对象
    opts->timeout = pgTimeout_; // 设置超时时间
#ifdef NCCL_HAS_COMM_SPLIT
    if (split_from) { // 如果有分裂源，则设置分裂选项和颜色
      opts->split_from = *split_from;
      opts->split_color = ++color_;
    }
#endif
    pg_ = std::unique_ptr<::c10d::ProcessGroupNCCL>( // 创建NCCL进程组对象
        new ::c10d::ProcessGroupNCCL(store_, rank, size, std::move(opts)));
  }

 protected:
  std::string path_; // 存储路径
  std::shared_ptr<::c10d::ProcessGroupNCCL> pg_; // NCCL进程组指针
  std::chrono::milliseconds pgTimeout_; // 进程组超时时间
  ::c10::intrusive_ptr<::c10d::Store> store_; // 存储对象的智能指针
  int color_{1}; // 颜色标记，默认为1
};

class NCCLTest : public NCCLTestBase {
 public:
  NCCLTest(
      const std::string& path,
      int rank,
      int worldSize,
      std::chrono::milliseconds pgTimeout =
          c10d::kProcessGroupNCCLDefaultTimeout,
      int inputDim = 3)
      : NCCLTestBase(path, pgTimeout),
        numDevices_(1), // 每个进程一个设备（线程）
        rank_(rank),
        worldSize_(worldSize) {
    // 惰性初始化CUDA上下文
    ::at::globalContext().lazyInitCUDA();
    tensors_.resize(numDevices_);
    inputs_.resize(numDevices_);
    outputs_.resize(numDevices_);
    at::cuda::OptionalCUDAGuard deviceGuard;
    assert(numDevices_ == 1);
    for (const auto i : c10::irange(numDevices_)) { // 对每个设备循环
      deviceGuard.set_index(rank_);
      tensors_[i] = at::empty({inputDim, inputDim}, at::kCUDA); // 创建CUDA张量
      inputs_[i].resize(worldSize_ * numDevices_);
      outputs_[i].resize(worldSize_ * numDevices_);
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) { // 对每个输入/输出张量循环
        inputs_[i][j] = at::empty({inputDim, inputDim}, at::kCUDA); // 创建CUDA输入张量
        outputs_[i][j] = at::empty({inputDim, inputDim}, at::kCUDA); // 创建CUDA输出张量
      }
    }

    // 为每个设备分配一个流对象。
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    // 1 device only, hence 1 stream only
    // 设置设备保护器的索引为当前的设备排名
    deviceGuard.set_index(rank_);
    // 将新申请的 CUDA 流添加到流数组中
    streams_.push_back(at::cuda::getStreamFromPool());
  }

  void wait(
      c10::intrusive_ptr<c10d::Work>& work,
      std::chrono::milliseconds timeout = kNoTimeout) {
    // 使用多流保护器，设置使用当前类中维护的流集合
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    // 等待传入工作的完成，允许指定超时时间
    work->wait(timeout);
  }

  std::vector<at::Tensor> getTensors() {
    std::vector<at::Tensor> outputs(numDevices_);

    // 在函数执行期间，使用多流保护器，设置使用当前类中维护的流集合
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 将输入张量复制到输出张量
    for (const auto i : c10::irange(numDevices_)) {
      // 同步 CUDA 流，确保流中的操作完成
      C10_CUDA_CHECK(cudaStreamSynchronize(streams_[i].stream()));
      // 将张量从 GPU 复制到 CPU，存入输出向量
      outputs[i] = tensors_[i].cpu();
    }

    return outputs;
  }

  std::vector<std::vector<at::Tensor>> getInputTensors() {
    // 返回输入张量的张量列表
    return getTensorLists(inputs_);
  }
  std::vector<std::vector<at::Tensor>> getOutputTensors() {
    // 返回输出张量的张量列表
    return getTensorLists(outputs_);
  }

  int numDevices() const {
    // 返回当前对象管理的设备数量
    return numDevices_;
  }

 private:
  std::vector<std::vector<at::Tensor>> getTensorLists(
      std::vector<std::vector<at::Tensor>>& tensor_lists) {
    std::vector<std::vector<at::Tensor>> outputs(numDevices_);
    // 对每个输出向量进行初始化，分配空间以容纳张量
    for (auto& output : outputs) {
      output = std::vector<at::Tensor>(worldSize_ * numDevices_);
    }

    // 在函数执行期间，使用多流保护器，设置使用当前类中维护的流集合
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 将输入张量列表复制到输出张量列表
    for (const auto i : c10::irange(numDevices_)) {
      // 同步 CUDA 流，确保流中的操作完成
      C10_CUDA_CHECK(cudaStreamSynchronize(streams_[i].stream()));
      // 将每个张量复制到输出张量列表中对应位置的张量
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        outputs[i][j] = tensor_lists[i][j].cpu();
      }
    }
    return outputs;
  }

 protected:
  // 在每个 CUDA 设备上启动休眠
  void launchDeviceSleep() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      // 设置设备保护器的索引为当前的设备排名
      deviceGuard.set_index(rank_);
      // 在指定的 CUDA 流上休眠指定的时间（以纳秒为单位）
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }
  }

  // 对每个张量进行值初始化
  void valueInitialization() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      // 设置设备保护器的索引为当前的设备排名
      deviceGuard.set_index(rank_);
      // 使用当前设备的 GPU 等级和设备索引填充张量
      tensors_[i].fill_(pg_->getRank() * numDevices_ + i);
    }
  }

  at::Tensor to_sparse_row_indices_format(at::Tensor& tensor) {
    // 获取稠密张量中所有非零元素的索引
    // 获取非零元素的唯一行索引
    auto row_indices = std::get<0>(
        at::_unique(tensor.nonzero().select(/*dim=*/1, /*index=*/0)));
    // 选择非零索引处的值张量
    at::Tensor sparse_values = tensor.index_select(
        /*dim=*/0, row_indices); // 获取非零索引处的值
    // 返回一个稀疏 COO 格式的张量，使用给定的行索引、稀疏值和张量尺寸
    return at::sparse_coo_tensor(
               row_indices.unsqueeze(0), sparse_values, tensor.sizes())
        .to(tensor.device());
  }

  // 为每个稀疏张量启动数值初始化
  void valueInitializationForSparse() {
    // 可选的 CUDA 设备保护
    at::cuda::OptionalCUDAGuard deviceGuard;
    // 遍历所有设备
    for (const auto i : c10::irange(numDevices_)) {
      // 设置当前设备索引
      deviceGuard.set_index(rank_);
      // 使用特定规则填充张量
      tensors_[i].fill_(pg_->getRank() * numDevices_ + i + 1);
      // 将稠密张量转换为 COO 行索引格式的稀疏张量
      tensors_[i] = to_sparse_row_indices_format(tensors_[i]);
    }
  }

  // 设备数量
  const int numDevices_;
  // 进程排名
  int rank_;
  // 全局大小
  int worldSize_;
  // 张量向量
  std::vector<at::Tensor> tensors_;
  // 输入张量向量的向量
  std::vector<std::vector<at::Tensor>> inputs_;
  // 输出张量向量的向量
  std::vector<std::vector<at::Tensor>> outputs_;
  // CUDA 流的向量
  std::vector<CUDAStream> streams_;
};

class AllreduceNCCLTest : public NCCLTest {
 public:
  // 构造函数，初始化基类 NCCLTest 和成员变量
  AllreduceNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  // 运行测试的函数，返回一个指向 Work 对象的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 使用 CUDAMultiStreamGuard 确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 调用设备休眠函数
    launchDeviceSleep();
    // 初始化值
    valueInitialization();

    // 使用命名空间 torch::autograd::profiler
    // 启用性能分析，注意在单进程多设备模式下不支持集体操作的性能分析
    enableProfilerLegacy(ProfilerConfig(ProfilerState::CPU));
    // 调用 pg_->allreduce 执行全局归约操作
    auto results = pg_->allreduce(tensors_);
    // 禁用性能分析
    disableProfilerLegacy();
    // 返回 allreduce 操作的结果
    return results;
  }
};

class SparseAllreduceNCCLTest : public NCCLTest {
 public:
  // 构造函数，初始化基类 NCCLTest 和成员变量
  SparseAllreduceNCCLTest(
      const std::string& path,
      int rank,
      int worldSize,
      int inputDim)
      : NCCLTest(
            path,
            rank,
            worldSize,
            c10d::kProcessGroupNCCLDefaultTimeout,
            inputDim) {}

  // 运行测试的函数，返回一个指向 Work 对象的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 使用 CUDAMultiStreamGuard 确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    // 调用设备休眠函数
    launchDeviceSleep();
    // 初始化稀疏值
    valueInitializationForSparse();
    // 调用 pg_->allreduce_sparse 执行稀疏张量的全局归约操作
    auto results = pg_->allreduce_sparse(tensors_);
    // 返回 allreduce_sparse 操作的结果
    return results;
  }
};

class BroadcastNCCLTest : public NCCLTest {
 public:
  // 构造函数，初始化基类 NCCLTest 和成员变量
  BroadcastNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  // 运行测试的函数，返回一个指向 Work 对象的智能指针
  c10::intrusive_ptr<c10d::Work> run(int rootRank, int rootTensor) {
    // 使用 CUDAMultiStreamGuard 确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 调用设备休眠函数
    launchDeviceSleep();
    // 初始化值
    valueInitialization();

    // 创建 BroadcastOptions 对象
    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    // 调用 pg_->broadcast 执行广播操作
    return pg_->broadcast(tensors_, options);
  }
};

class ReduceNCCLTest : public NCCLTest {
 public:
  // 构造函数，初始化基类 NCCLTest 和成员变量
  ReduceNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  // 运行测试的函数，返回一个指向 Work 对象的智能指针
  c10::intrusive_ptr<c10d::Work> run(int rootRank, int rootTensor) {
    // 使用 CUDAMultiStreamGuard 确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 调用设备休眠函数
    launchDeviceSleep();
    // 初始化值
    valueInitialization();

    // 创建 ReduceOptions 对象
    ::c10d::ReduceOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    // 调用 pg_->reduce 执行归约操作
    return pg_->reduce(tensors_, options);
  }
};

class AllgatherNCCLTest : public NCCLTest {
 public:
  // 构造函数，初始化基类 NCCLTest 和成员变量
  AllgatherNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  // 运行测试的函数，返回一个指向 Work 对象的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 使用 CUDAMultiStreamGuard 确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 调用设备休眠函数
    launchDeviceSleep();
    // 初始化值
    valueInitialization();
    // 调用pg_对象的allgather方法，将outputs_和tensors_作为参数传递
    return pg_->allgather(outputs_, tensors_);
  }
};

// 定义一个名为 AllgatherBaseNCCLTest 的类，它继承自 NCCLTest 类
class AllgatherBaseNCCLTest : public NCCLTest {
 public:
  // 构造函数，接受路径、排名和世界大小作为参数
  AllgatherBaseNCCLTest(const std::string& path, int rank, int worldSize)
      // 调用基类 NCCLTest 的构造函数
      : NCCLTest(path, rank, worldSize) {
    // 初始化 output_tensor_ 为一个大小为 [worldSize_, 3, 3] 的空张量，存储在 CUDA 设备上
    output_tensor_ = at::empty({worldSize_, 3, 3}, at::kCUDA);
  }

  // 运行函数，返回一个指向 c10d::Work 的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 在该函数的执行期间，确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 调用 launchDeviceSleep 函数
    launchDeviceSleep();
    // 调用 valueInitialization 函数
    valueInitialization();
    // 进行扁平化的 allgather，每个排名贡献一个张量，与设备数量无关
    return pg_->_allgather_base(output_tensor_, tensors_[0]);
  }

  // 返回 output_tensor_ 的 CPU 版本张量
  at::Tensor getOutputTensor() {
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    return output_tensor_.cpu();
  }

  // 返回 tensors_ 中第一个张量的 CPU 版本张量
  at::Tensor getInputTensor() {
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    return tensors_[0].cpu();
  }

 private:
  at::Tensor output_tensor_;  // 输出张量
};

// 定义一个名为 ReduceScatterNCCLTest 的结构体，它继承自 NCCLTest 类
struct ReduceScatterNCCLTest : NCCLTest {
  // 构造函数，接受路径、排名和世界大小作为参数
  ReduceScatterNCCLTest(const std::string& path, int rank, int worldSize)
      // 调用基类 NCCLTest 的构造函数
      : NCCLTest(path, rank, worldSize) {}

  // 运行函数，返回一个指向 c10d::Work 的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 在该函数的执行期间，确保 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 创建 OptionalCUDAGuard 对象 deviceGuard
    at::cuda::OptionalCUDAGuard deviceGuard;
    // 调用 launchDeviceSleep 函数
    launchDeviceSleep();

    // 为每个张量启动值初始化
    for (auto j = 0; j < worldSize_; ++j) {
      inputs_[0][j].fill_(rank_ * worldSize_ + j);
    }

    // 调用 pg_ 的 reduce_scatter 函数，传入 tensors_ 和 inputs_ 作为参数
    return pg_->reduce_scatter(tensors_, inputs_);
  }
};

// 定义一个名为 ReduceScatterBaseNCCLTest 的类，它继承自 NCCLTest 类
class ReduceScatterBaseNCCLTest : public NCCLTest {
 public:
  // 构造函数，接受路径、排名和世界大小作为参数
  ReduceScatterBaseNCCLTest(const std::string& path, int rank, int worldSize)
      // 调用基类 NCCLTest 的构造函数
      : NCCLTest(path, rank, worldSize) {
    // 创建 OptionalCUDAGuard 对象 deviceGuard，设置当前设备为 rank_
    at::cuda::OptionalCUDAGuard deviceGuard;
    deviceGuard.set_index(rank_);
    // 初始化 output_tensor_ 为大小为 [1] 的空张量，存储在 CUDA 设备上
    output_tensor_ = at::empty({1}, at::kCUDA);
    // 初始化 input_tensor_ 为大小为 [worldSize] 的空张量，存储在 CUDA 设备上
    input_tensor_ = at::empty({worldSize}, at::kCUDA);
    // 循环初始化 input_tensor_ 的每个元素
    for (const auto i : c10::irange(worldSize)) {
      input_tensor_[i] = i;
    }
  }

  // 运行函数，返回一个指向 c10d::Work 的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 在该函数的执行期间，确保 THC 使用我们的流
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // 调用 launchDeviceSleep 函数
    launchDeviceSleep();
    // 调用 pg_ 的 _reduce_scatter_base 函数，传入 output_tensor_ 和 input_tensor_ 作为参数
    return pg_->_reduce_scatter_base(output_tensor_, input_tensor_);
  }

  // 返回 output_tensor_ 的 CPU 版本张量
  at::Tensor getOutputTensor() {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    return output_tensor_.cpu();
  }

  // 返回 input_tensor_ 的 CPU 版本张量
  at::Tensor getInputTensor() {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    return input_tensor_.cpu();
  }

 private:
  at::Tensor output_tensor_;  // 输出张量
  at::Tensor input_tensor_;   // 输入张量
};
void testAllreduce(const std::string& path, int rank, int size) {
  // 创建 AllreduceNCCLTest 对象，传入路径、排名和尺寸信息
  auto test = AllreduceNCCLTest(path, rank, size);
  // 初始化测试对象
  test.initialize(rank, size);
  // 运行测试，并获取运行工作的句柄
  auto work = test.run();
  // 等待工作完成
  test.wait(work);

  // 验证
  // 计算总 GPU 数量
  const int totalNumGPUs = test.numDevices() * size;
  // 计算期望的输出值
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  // 获取测试对象中的张量数据
  const auto tensors = test.getTensors();
  // 遍历所有张量
  for (const auto& tensor : tensors) {
    // 获取张量的数据指针
    const auto* const data = tensor.const_data_ptr<float>();
    // 检查每个元素是否符合预期值
    for (const auto k : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce outputs do not match expected outputs";
    }
  }
}

void testSparseAllreduce(const std::string& path, int rank, int size) {
  // 设置输入维度
  const int inputDim = 3;
  // 创建 SparseAllreduceNCCLTest 对象，传入路径、排名、尺寸和输入维度信息
  auto test = SparseAllreduceNCCLTest(path, rank, size, inputDim);
  // 初始化测试对象
  test.initialize(rank, size);
  // 运行测试，并获取运行工作的句柄
  auto work = test.run();
  // 等待工作完成
  test.wait(work);

  // 获取输入张量列表
  const auto input_tensors = test.getTensors();

  // 验证输出结果与张量数据是否一致
  auto output_tensor = work->result();
  // 计算总 GPU 数量
  int totalNumGPUs = test.numDevices() * size;
  // 由于添加额外的 1 以避免空张量，增加总 GPU 数量
  totalNumGPUs++;
  // 计算期望的输出值
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  // 遍历输入张量列表
  for (const auto i : c10::irange(input_tensors.size())) {
    // 获取当前张量的引用
    const auto& tensor = input_tensors[i];

    // 验证张量是否为稀疏张量
    EXPECT_EQ(tensor.is_sparse(), true);

    // 获取张量的索引和值
    auto indices = tensor._indices();
    auto values = tensor._values();

    // 验证索引的尺寸是否符合预期
    auto sizes = indices.sizes();
    EXPECT_EQ(sizes.size(), 2);
    if (sizes[0] == 1) {
      // 验证行索引的尺寸
      EXPECT_EQ(sizes[1], inputDim);
    } else if (sizes[0] == 2) {
      // 验证坐标索引的尺寸
      EXPECT_EQ(sizes[1], inputDim * inputDim);
    }

    // 验证所有张量值是否为期望值
    const auto* const data = values.const_data_ptr<float>();
    for (const auto k : c10::irange(values.numel())) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce outputs do not match expected outputs";
    }

    // 预期输入张量和输出张量应该相同
    auto input_dense = tensor.to_dense();
    auto output_dense = output_tensor[i].to(input_dense.device()).to_dense();
    EXPECT_TRUE(input_dense.allclose(output_dense));
  }
}
// 测试稀疏全局归约的函数，对给定路径、进程编号和总进程数进行初始化
void testSparseAllreduceLarge(const std::string& path, int rank, int size) {
  // 定义输入维度为2500
  const int inputDim = 2500;
  // 创建 SparseAllreduceNCCLTest 对象并初始化
  auto test = SparseAllreduceNCCLTest(path, rank, size, inputDim);
  test.initialize(rank, size);
  // 运行测试
  auto work = test.run();
  // 等待工作完成
  test.wait(work);

  // 获取输入张量列表
  const auto input_tensors = test.getTensors();

  // 验证工作输出与张量相同
  auto output_tensor = work->result();
  // 进行验证
  int totalNumGPUs = test.numDevices() * size;
  // 增加1，以防止空张量
  totalNumGPUs++;
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  // 遍历每个输入张量
  for (const auto i : c10::irange(input_tensors.size())) {
    const auto& tensor = input_tensors[i];

    // 验证张量是否稀疏
    EXPECT_EQ(tensor.is_sparse(), true);

    auto indices = tensor._indices();
    auto values = tensor._values();

    // 验证索引大小是否符合预期
    auto sizes = indices.sizes();
    EXPECT_EQ(sizes.size(), 2);
    if (sizes[0] == 1) {
      // 行索引
      EXPECT_EQ(sizes[1], inputDim);
    } else if (sizes[0] == 2) {
      // 坐标索引
      EXPECT_EQ(sizes[1], inputDim * inputDim);
    }

    // 验证所有张量值是否为期望值
    const auto* const data = values.const_data_ptr<float>();
    for (const auto k : c10::irange(values.numel())) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce outputs do not match expected outputs";
    }

    // 预期输入和输出张量应相同
    auto input_dense = tensor.to_dense();
    auto output_dense = output_tensor[i].to(input_dense.device()).to_dense();
    EXPECT_TRUE(input_dense.allclose(output_dense));
  }
}

// 广播测试函数，使用给定的路径、进程编号和总进程数进行初始化
void testBroadcast(const std::string& path, int rank, int size) {
  // 创建 BroadcastNCCLTest 对象并初始化
  auto test = BroadcastNCCLTest(path, rank, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // 尝试每个根排名和根张量的排列组合
  for (const auto rootRank : c10::irange(size)) {
    for (const auto rootTensor : c10::irange(numDevices)) {
      auto work = test.run(rootRank, rootTensor);

      // 等待工作完成
      test.wait(work);

      // 检查结果
      const auto expected = (rootRank * numDevices + rootTensor);
      const auto tensors = test.getTensors();
      for (const auto& tensor : tensors) {
        const auto* const data = tensor.const_data_ptr<float>();
        for (const auto k : c10::irange(tensor.numel())) {
          EXPECT_EQ(data[k], expected)
              << "Broadcast outputs do not match expected outputs";
        }
      }
    }
  }
}

// 归约测试函数，使用给定的路径、进程编号和总进程数进行初始化
void testReduce(const std::string& path, int rank, int size) {
  // 创建 ReduceNCCLTest 对象并初始化
  auto test = ReduceNCCLTest(path, rank, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // 尝试每个根排名和根张量的排列组合
  for (const auto rootRank : c10::irange(size)) {
    // 遍历每个根张量（rootTensor），根据设备数量范围进行迭代
    for (const auto rootTensor : c10::irange(numDevices)) {
      // 运行测试，并获取返回的工作句柄
      auto work = test.run(rootRank, rootTensor);

      // 等待工作完成
      test.wait(work);

      // 验证阶段
      // 计算总 GPU 数量
      const int totalNumGPUs = numDevices * size;
      // 计算预期输出值
      const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
      // 获取所有张量的列表
      auto tensors = test.getTensors();
      // 如果当前进程是根进程
      if (rank == rootRank) {
        // 获取当前根张量的引用
        auto& tensor = tensors[rootTensor];
        // 获取张量数据的指针，并遍历每个元素
        auto data = tensor.data_ptr<float>();
        for (const auto k : c10::irange(tensor.numel())) {
          // 检查每个元素是否与预期值相等，如果不相等则输出错误信息
          EXPECT_EQ(data[k], expected)
              << "Reduce outputs do not match expected outputs";
        }
      }
    }
  }
void testAllgather(const std::string& path, int rank, int size) {
  // 创建一个 AllgatherNCCLTest 对象，传入路径、排名和大小参数
  auto test = AllgatherNCCLTest(path, rank, size);
  // 初始化测试对象
  test.initialize(rank, size);
  // 运行测试并获取工作对象
  auto work = test.run();
  // 等待工作完成
  test.wait(work);

  // 验证输出结果
  auto tensors = test.getOutputTensors();
  // 遍历各设备的张量
  for (auto& device : tensors) {
    // 遍历每个设备的张量元素
    for (const auto j : c10::irange(device.size())) {
      // 期望值为当前元素的索引
      const auto expected = j;
      auto& tensor = device[j];
      auto data = tensor.data_ptr<float>();
      // 遍历张量的数据元素
      for (const auto k : c10::irange(tensor.numel())) {
        // 断言当前数据元素与期望值相等
        EXPECT_EQ(data[k], expected)
            << "Allgather outputs do not match expected outputs";
      }
    }
  }
}

void testAllgatherBase(const std::string& path, int rank, int size) {
  // 创建一个 AllgatherBaseNCCLTest 对象，传入路径、排名和大小参数
  auto test = AllgatherBaseNCCLTest(path, rank, size);
  // 初始化测试对象
  test.initialize(rank, size);
  // 运行测试并获取工作对象
  auto work = test.run();
  // 等待工作完成
  test.wait(work);
  // 验证输出结果
  auto output_tensor = test.getOutputTensor();
  auto input_tensor = test.getInputTensor();

  auto data = output_tensor.data_ptr<float>();

  // 遍历输出张量的元素
  for (const auto i : c10::irange(output_tensor.numel())) {
    // 期望值为每个元素的索引除以输入张量的元素数乘以设备数
    const auto expected = (i / input_tensor.numel()) * test.numDevices();
    // 断言当前数据元素与期望值相等
    EXPECT_EQ(data[i], expected)
        << "Allgather_base outputs do not match expected outputs";
  }
}

void testReduceScatterBase(const std::string& path, int rank, int size) {
  // 创建一个 ReduceScatterBaseNCCLTest 对象，传入路径、排名和大小参数
  auto test = ReduceScatterBaseNCCLTest(path, rank, size);
  // 初始化测试对象
  test.initialize(rank, size);
  // 运行测试并获取工作对象
  auto work = test.run();
  // 等待工作完成
  test.wait(work);
  // 验证输出结果
  auto output_tensor = test.getOutputTensor();
  auto input_tensor = test.getInputTensor();

  auto data = output_tensor.data_ptr<float>();

  // 遍历输出张量的元素
  for (const auto i : c10::irange(output_tensor.numel())) {
    // 期望值为每个元素的索引乘以输入张量的元素数乘以设备数
    const auto expected = size * rank * test.numDevices();
    // 断言当前数据元素与期望值相等
    EXPECT_EQ(data[i], expected)
        << "Reducescatter_base outputs do not match expected outputs";
  }
}

void testReduceScatter(const std::string& path, int rank, int size) {
  // 创建一个 ReduceScatterNCCLTest 对象，传入路径、排名和大小参数
  auto test = ReduceScatterNCCLTest(path, rank, size);
  // 初始化测试对象
  test.initialize(rank, size);
  // 运行测试并获取工作对象
  auto work = test.run();
  // 等待工作完成
  test.wait(work);

  const auto participants = size;
  const auto base = (participants * (participants - 1)) / 2;

  // 验证输出结果
  auto tensors = test.getTensors();
  const auto modifier = rank * participants;
  const auto expected = base * participants + modifier;
  auto& tensor = tensors[0];
  auto data = tensor.data_ptr<float>();
  // 遍历张量的数据元素
  for (const auto j : c10::irange(tensor.numel())) {
    // 断言当前数据元素与期望值相等
    EXPECT_EQ(data[j], expected)
        << "ReduceScatter outputs do not match expected outputs!";
  }
}
// 测试序列号初始化函数，初始化 NCCLTest 对象并设置序列号
void testSequenceNumInit(const std::string& path, int rank, int size) {
  // 创建 NCCLTest 对象，传入路径、排名和大小
  NCCLTest test(path, rank, size);
  // 初始化 NCCLTest 对象
  test.initialize(rank, size);
  // 获取进程组并为其设置序列号
  test.getProcessGroup()->setSequenceNumberForGroup();
  // 获取进程组的当前序列号
  auto seqNum = test.getProcessGroup()->getSequenceNumberForGroup();
  // 断言当前序列号为 0
  EXPECT_EQ(seqNum, 0);
}

// 测试通信组分裂函数，通过广播测试对象进行通信组的分裂
void testSplittingCommunicator(const std::string& path, int rank, int size) {
  // 创建 BroadcastNCCLTest 对象 test1 和 test2，传入路径、排名和大小
  auto test1 = BroadcastNCCLTest(path, rank, size);
  test1.initialize(rank, size);

  auto test2 = BroadcastNCCLTest(path, rank, size);
  // 初始化 test2，并指定使用 test1 的进程组
  test2.initialize(rank, size, test1.getProcessGroup());

  // 复制广播测试对象以确保一致的全体通信
  for (auto test : {&test1, &test2}) {
    // 获取设备数量
    const int numDevices = test->numDevices();
    // 尝试每个根排名和根张量的排列组合
    for (const auto rootRank : c10::irange(size)) {
      for (const auto rootTensor : c10::irange(numDevices)) {
        // 运行广播测试，并获取工作句柄
        auto work = test->run(rootRank, rootTensor);
        // 等待广播操作完成
        test->wait(work);

        // 检查结果
        const auto expected = (rootRank * numDevices + rootTensor);
        const auto tensors = test->getTensors();
        // 遍历所有张量，检查广播输出是否符合预期输出
        for (const auto& tensor : tensors) {
          const auto* const data = tensor.const_data_ptr<float>();
          for (const auto k : c10::irange(tensor.numel())) {
            EXPECT_EQ(data[k], expected)
                << "Broadcast outputs do not match expected outputs";
          }
        }
      }
    }
  }

  // 确保原始和分裂进程组上的通信分裂计数器符合预期：原始进程组为 0，第二个进程组为设备数
  EXPECT_EQ(test2.getProcessGroup()->getCommSplitCounter(), 0);
  EXPECT_EQ(test1.getProcessGroup()->getCommSplitCounter(), test1.numDevices());
}

// 所有 testAbc 使用此签名
using FuncType = void (*)(const std::string&, int, int);

// ProcessGroupNCCLTest 类，继承自 Google Test 框架的 Test 类
class ProcessGroupNCCLTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 初始化 Torch 日志
    c10::initLogging();
    // 使用 WORLD_SIZE 和 RANK 环境变量进行多节点分布式测试
    auto sizeEnv = std::getenv("WORLD_SIZE");
    if (sizeEnv) {
      size_ = std::stoi(std::string(sizeEnv));
    }
    // 打印当前测试的世界大小
    LOG(INFO) << "ProcessGroupNCCLTest world size: " << size_;
  }

  void TearDown() override {
    // 每次运行结束后重置 TORCH_NCCL_BLOCKING_WAIT 环境变量
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "0", 1) == 0);
  }

  bool skipTest() {
    // 如果 CUDA 不可用，则跳过测试
    if (!at::cuda::is_available()) {
      LOG(INFO) << "CUDA not available, skipping test";
      return true;
    }
    return false;
  }

  // 多线程运行测试函数
  void multiThreadRun(FuncType testFunc) {
    // 创建临时文件对象
    TemporaryFile file;
    // 创建线程向量
    std::vector<std::thread> threads;
    // 预留足够多的线程空间
    threads.reserve(size_);
    // 使用范围循环遍历从 0 到 size_-1 的整数序列，其中 rank 表示当前迭代的值
    for (const auto rank : c10::irange(size_)) {
        // 创建新的线程，调用 testFunc 函数，并传递 file.path、rank 和 size_ 作为参数
        threads.emplace_back(std::thread(testFunc, file.path, rank, size_));
    }
    // 使用范围循环遍历从 0 到 size_-1 的整数序列，其中 rank 表示当前迭代的值
    for (const auto rank : c10::irange(size_)) {
        // 等待线程数组中索引为 rank 的线程结束执行
        threads[rank].join();
    }
  }

  // 初始化整数变量 size_ 为 1
  int size_{1};
};

// 测试函数：testAllreduce，用于测试 Allreduce 功能
TEST_F(ProcessGroupNCCLTest, testAllreduce) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testAllreduce 函数
  multiThreadRun(testAllreduce);
}

// 测试函数：testBroadcast，用于测试 Broadcast 功能
TEST_F(ProcessGroupNCCLTest, testBroadcast) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testBroadcast 函数
  multiThreadRun(testBroadcast);
}

// 测试函数：testReduce，用于测试 Reduce 功能
TEST_F(ProcessGroupNCCLTest, testReduce) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testReduce 函数
  multiThreadRun(testReduce);
}

// 测试函数：testAllgather，用于测试 Allgather 功能
TEST_F(ProcessGroupNCCLTest, testAllgather) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testAllgather 函数
  multiThreadRun(testAllgather);
}

// 测试函数：testAllgatherBase，用于测试基本 Allgather 功能
TEST_F(ProcessGroupNCCLTest, testAllgatherBase) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testAllgatherBase 函数
  multiThreadRun(testAllgatherBase);
}

// 测试函数：testReduceScatter，用于测试 ReduceScatter 功能
TEST_F(ProcessGroupNCCLTest, testReduceScatter) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testReduceScatter 函数
  multiThreadRun(testReduceScatter);
}

// 测试函数：testSequenceNumInit，用于测试序列号初始化
TEST_F(ProcessGroupNCCLTest, testSequenceNumInit) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testSequenceNumInit 函数
  multiThreadRun(testSequenceNumInit);
}

// 测试函数：testReduceScatterBase，用于测试基本 ReduceScatter 功能
TEST_F(ProcessGroupNCCLTest, testReduceScatterBase) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testReduceScatterBase 函数
  multiThreadRun(testReduceScatterBase);
}

// 测试函数：testBackendName，用于测试后端名称获取
TEST_F(ProcessGroupNCCLTest, testBackendName) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 创建一个临时文件对象
  TemporaryFile file;
  // 创建一个 NCCLTestBase 对象，并初始化
  auto test = NCCLTestBase(file.path);
  test.initialize(/*rank=*/0, /*world_size=*/1);
  // 断言获取的进程组后端名称与预期的 NCCL 后端名称相等
  EXPECT_EQ(
      test.getProcessGroup()->getBackendName(),
      std::string(c10d::NCCL_BACKEND_NAME));
}

// 测试函数：testSplittingCommunicator，用于测试通信器分割功能
TEST_F(ProcessGroupNCCLTest, testSplittingCommunicator) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testSplittingCommunicator 函数
  multiThreadRun(testSplittingCommunicator);
}

// 当定义了 IS_NCCLX 时，执行以下测试函数
#ifdef IS_NCCLX
// 测试函数：testSparseAllreduce，用于测试稀疏 Allreduce 功能
TEST_F(ProcessGroupNCCLTest, testSparseAllreduce) {
  // 如果需要跳过测试，则直接返回
  if (skipTest()) {
    return;
  }
  // 在多线程环境下运行 testSparseAllreduce 函数
  multiThreadRun(testSparseAllreduce);
  // 在多线程环境下运行 testSparseAllreduceLarge 函数
  multiThreadRun(testSparseAllreduceLarge);
}
#endif
```