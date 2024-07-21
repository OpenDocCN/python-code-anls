# `.\pytorch\test\cpp\api\dataloader.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/torch.h>  // 包含 PyTorch 的头文件

#include <test/cpp/api/support.h>  // 包含测试支持的头文件

#include <c10/util/ArrayRef.h>  // 包含 C10 库的头文件
#include <c10/util/irange.h>   // 包含 C10 库的头文件
#include <c10/util/tempfile.h> // 包含 C10 库的头文件

#include <algorithm>   // 包含标准算法库的头文件
#include <chrono>      // 包含时间库的头文件
#include <future>      // 包含异步任务库的头文件
#include <iostream>    // 包含输入输出流库的头文件
#include <iterator>    // 包含迭代器库的头文件
#include <limits>      // 包含数值极限库的头文件
#include <mutex>       // 包含互斥量库的头文件
#include <numeric>     // 包含数值操作库的头文件
#include <stdexcept>   // 包含标准异常库的头文件
#include <string>      // 包含字符串库的头文件
#include <thread>      // 包含线程库的头文件
#include <unordered_set>  // 包含无序集合库的头文件
#include <vector>      // 包含向量库的头文件

using namespace torch::data;  // 使用 torch::data 命名空间，NOLINT

const std::chrono::milliseconds kMillisecond(1);  // 定义一个毫秒常量

struct DummyDataset : datasets::Dataset<DummyDataset, int> {  // 定义 DummyDataset 结构体，继承自 Dataset
  explicit DummyDataset(size_t size = 100) : size_(size) {}  // 构造函数，设置数据集大小

  int get(size_t index) override {  // 实现 Dataset 接口的 get 方法
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return 1 + index;  // 返回索引加一的值
  }
  torch::optional<size_t> size() const override {  // 实现 Dataset 接口的 size 方法
    return size_;  // 返回数据集大小
  }

  size_t size_;  // 数据集大小
};

TEST(DataTest, DatasetCallsGetCorrectly) {  // 数据集测试，验证 get 方法的正确调用
  DummyDataset d;  // 创建 DummyDataset 实例
  std::vector<int> batch = d.get_batch({0, 1, 2, 3, 4});  // 调用 get_batch 方法获取批次数据
  std::vector<int> expected = {1, 2, 3, 4, 5};  // 预期的批次数据
  ASSERT_EQ(batch, expected);  // 断言批次数据与预期数据相等
}

TEST(DataTest, TransformCallsGetApplyCorrectly) {  // 转换测试，验证 apply 方法的正确调用
  struct T : transforms::Transform<int, std::string> {  // 定义转换结构体 T，继承自 Transform
    std::string apply(int input) override {  // 实现 Transform 接口的 apply 方法
      return std::to_string(input);  // 将整数转换为字符串
    }
  };

  auto d = DummyDataset{}.map(T{});  // 对 DummyDataset 应用转换 T
  std::vector<std::string> batch = d.get_batch({0, 1, 2, 3, 4});  // 获取转换后的批次数据
  std::vector<std::string> expected = {"1", "2", "3", "4", "5"};  // 预期的转换后的批次数据
  ASSERT_EQ(batch, expected);  // 断言转换后的批次数据与预期数据相等
}

// dummy chunk data reader with 3 chunks and 35 examples in total. Each chunk
// contains 10, 5, 20 examples respectively.
struct DummyChunkDataReader : public datasets::ChunkDataReader<int> {  // 定义 DummyChunkDataReader 结构体，继承自 ChunkDataReader
 public:
  using BatchType = datasets::ChunkDataReader<int>::ChunkType;  // 定义批次类型为 ChunkType
  using DataType = datasets::ChunkDataReader<int>::ExampleType;  // 定义数据类型为 ExampleType

  /// Read an entire chunk.
  BatchType read_chunk(size_t chunk_index) override {  // 实现读取 chunk 的方法
    BatchType batch_data;  // 定义批次数据
    int start_index = chunk_index == 0
        ? 0
        // NOLINTNEXTLINE(bugprone-fold-init-type)
        : std::accumulate(chunk_sizes, chunk_sizes + chunk_index, 0);  // 根据 chunk_index 累加起始索引

    batch_data.resize(chunk_sizes[chunk_index]);  // 调整批次数据的大小

    std::iota(batch_data.begin(), batch_data.end(), start_index);  // 从 start_index 开始填充批次数据

    return batch_data;  // 返回填充好的批次数据
  }

  size_t chunk_count() override {  // 实现获取 chunk 数量的方法
    return chunk_count_;  // 返回 chunk 的数量
  };

  void reset() override{};  // 实现重置方法，暂不做任何操作

  const static size_t chunk_count_ = 3;  // 静态常量，chunk 的数量为 3
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays)
  size_t chunk_sizes[chunk_count_] = {10, 5, 20};  // chunk_sizes 数组，每个 chunk 的大小
};

TEST(DataTest, ChunkDataSetWithInvalidInitParameter) {  // 带有无效初始化参数的 ChunkDataSet 测试
  DummyChunkDataReader data_reader;  // 创建 DummyChunkDataReader 实例
  samplers::SequentialSampler sampler(0);  // 创建顺序采样器实例

  auto initialization_function = [&](size_t preloader_count,
                                     size_t batch_size,
                                     size_t cache_size,
                                     size_t cross_chunk_shuffle_count = 1) {  // 初始化函数，使用 lambda 表达式
    datasets::SharedBatchDataset<datasets::ChunkDataset<
        DummyChunkDataReader,
        samplers::SequentialSampler,
        samplers::SequentialSampler>>
        dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
            DummyChunkDataReader,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>(
            data_reader,
            sampler,
            sampler,
            datasets::ChunkDatasetOptions(
                preloader_count,
                batch_size,
                cache_size,
                cross_chunk_shuffle_count));


    // 创建一个共享的批处理数据集，该数据集包含一个分块数据集
    // 使用 DummyChunkDataReader 作为数据读取器
    // 使用 samplers::SequentialSampler 作为数据采样器
    // datasets::make_shared_dataset 创建一个共享数据集的实例
    dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
            DummyChunkDataReader,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>(
            data_reader,  // 数据读取器对象
            sampler,      // 数据采样器对象
            sampler,      // 另一个数据采样器对象，用于跨块数据集
            datasets::ChunkDatasetOptions(  // 配置数据集选项
                preloader_count,          // 预加载器数量
                batch_size,               // 批处理大小
                cache_size,               // 缓存大小
                cross_chunk_shuffle_count  // 跨块数据集的洗牌次数
            ));
  };


  ASSERT_THROWS_WITH(
      initialization_function(0, 1, 1),
      "Preloader count is 0. At least one preloader needs to be specified.");


  // 当预加载器数量为0时，抛出异常并显示相应错误信息
  ASSERT_THROWS_WITH(
      initialization_function(0, 1, 1),
      "Preloader count is 0. At least one preloader needs to be specified.");


  ASSERT_THROWS_WITH(
      initialization_function(1, 0, 1),
      "Batch size is 0. A positive batch size needs to be specified.");


  // 当批处理大小为0时，抛出异常并显示相应错误信息
  ASSERT_THROWS_WITH(
      initialization_function(1, 0, 1),
      "Batch size is 0. A positive batch size needs to be specified.");


  ASSERT_THROWS_WITH(
      initialization_function(1, 1, 0),
      "Cache size is 0. A positive cache size needs to be specified.");


  // 当缓存大小为0时，抛出异常并显示相应错误信息
  ASSERT_THROWS_WITH(
      initialization_function(1, 1, 0),
      "Cache size is 0. A positive cache size needs to be specified.");


  ASSERT_THROWS_WITH(
      initialization_function(1, 10, 5),
      "Cache size is less than batch size. Cache needs to be large enough to "
      "hold at least one batch.");


  // 当缓存大小小于批处理大小时，抛出异常并显示相应错误信息
  ASSERT_THROWS_WITH(
      initialization_function(1, 10, 5),
      "Cache size is less than batch size. Cache needs to be large enough to "
      "hold at least one batch.");


  ASSERT_THROWS_WITH(
      initialization_function(1, 10, 20, 0),
      "cross_chunk_shuffle_count needs to be greater than 0.");


  // 当跨块数据集的洗牌次数小于等于0时，抛出异常并显示相应错误信息
  ASSERT_THROWS_WITH(
      initialization_function(1, 10, 20, 0),
      "cross_chunk_shuffle_count needs to be greater than 0.");
}

// 定义无限流数据集 InfiniteStreamDataset，继承自 StreamDataset
struct InfiniteStreamDataset
    : datasets::StreamDataset<InfiniteStreamDataset, std::vector<int>> {
  
  // 重载 get_batch 函数，返回指定大小的整数向量批次
  std::vector<int> get_batch(size_t batch_size) override {
    // 创建一个大小为 batch_size 的整数向量 batch
    std::vector<int> batch(batch_size);
    // 使用计数器填充 batch 中的每个元素
    for (auto& i : batch) {
      i = counter++;
    }
    return batch;
  }

  // 重载 size 函数，返回 torch 的可选大小，此处返回无穷
  torch::optional<size_t> size() const override {
    return torch::nullopt;
  }

  // 计数器，用于跟踪生成的批次数
  size_t counter = 0;
};

// 测试函数 InfiniteStreamDataset
TEST(DataTest, InfiniteStreamDataset) {
  const size_t kBatchSize = 13;

  // 创建 InfiniteStreamDataset 实例，通过 map 函数应用 Lambda 转换
  auto dataset = InfiniteStreamDataset().map(
      transforms::Lambda<int>([](int x) { return x + 1; }));

  // 创建数据加载器 data_loader
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      samplers::StreamSampler(/*epoch_size=*/39),
      kBatchSize);

  // 批次索引，用于迭代和断言
  size_t batch_index = 0;
  // 迭代 data_loader 中的每个批次
  for (auto& batch : *data_loader) {
    // 断言批次索引小于 3
    ASSERT_LT(batch_index, 3);
    // 断言批次大小为 kBatchSize
    ASSERT_EQ(batch.size(), kBatchSize);
    // 遍历批次中的每个元素，断言其值符合预期
    for (const auto j : c10::irange(kBatchSize)) {
      ASSERT_EQ(batch.at(j), 1 + (batch_index * kBatchSize) + j);
    }
    batch_index += 1;
  }
  // 最终断言批次索引为 3
  ASSERT_EQ(batch_index, 3);
}

// 测试函数 NoSequencerIsIdentity
TEST(DataTest, NoSequencerIsIdentity) {
  using namespace torch::data::detail::sequencers; // NOLINT
  // 创建 NoSequencer 实例，无序序列器
  NoSequencer<int> no_sequencer;
  // 调用 next 方法，返回值应为 5
  const auto value = no_sequencer.next([] { return 5; }).value();
  // 断言返回值为 5
  ASSERT_EQ(value, 5);
}

// 测试函数 OrderedSequencerIsSetUpWell
TEST(DataTest, OrderedSequencerIsSetUpWell) {
  using namespace torch::data::detail::sequencers; // NOLINT
  // 定义 S 结构体，包含序列号
  struct S {
    size_t sequence_number;
  };
  const size_t kMaxJobs = 5;
  // 创建 OrderedSequencer 实例，有序序列器，预设最大作业数
  OrderedSequencer<S> sequencer(kMaxJobs);
  // 断言下一个序列号为 0
  ASSERT_EQ(sequencer.next_sequence_number_, 0);
  // 断言缓冲区大小为 kMaxJobs
  ASSERT_EQ(sequencer.buffer_.size(), kMaxJobs);
}

// 测试函数 OrderedSequencerReOrdersValues
TEST(DataTest, OrderedSequencerReOrdersValues) {
  using namespace torch::data::detail::sequencers; // NOLINT
  // 定义 S 结构体，包含序列号
  struct S {
    size_t sequence_number;
  };
  const size_t kMaxJobs = 5;
  // 创建 OrderedSequencer 实例，有序序列器，预设最大作业数
  OrderedSequencer<S> sequencer(kMaxJobs);

  // 初始化一个无序序列
  std::vector<size_t> v = {0, 2, 4, 3, 1};
  size_t index = 0;
  // 定义 getter 函数，每次返回 v 中的一个元素作为 S 结构体的序列号
  auto getter = [&v, &index]() { return S{v.at(index++)}; };

  // 第一次调用 next 应返回序列号为 0 的批次
  const auto batch = sequencer.next(getter);
  ASSERT_EQ(batch.value().sequence_number, 0);
  ASSERT_EQ(index, 1);

  // 接下来的调用应依次返回序列号为 1 到 4 的批次
  ASSERT_EQ(1, sequencer.next(getter).value().sequence_number);
  ASSERT_EQ(index, 5);

  // 后续的调用应按顺序返回序列号为 2 到 4 的批次
  for (size_t i = 2; i <= 4; ++i) {
    ASSERT_EQ(i, sequencer.next(getter).value().sequence_number);
    ASSERT_EQ(index, 5);  // 确保 index 未改变
  }
}

// 测试函数 BatchLambdaAppliesFunctionToBatch
TEST(DataTest, BatchLambdaAppliesFunctionToBatch) {
  using InputBatch = std::vector<int>;
  using OutputBatch = std::string;
  // 创建 DummyDataset 实例 d
  DummyDataset d;
  // 对 d 应用 BatchLambda 转换，计算整数向量输入的总和并转换为字符串
  auto e = d.map(transforms::BatchLambda<InputBatch, OutputBatch>(
      [](std::vector<int> input) {
        return std::to_string(std::accumulate(input.begin(), input.end(), 0));
      }));
  // 断言 e 调用 get_batch 方法返回的结果为字符串 "20"
  ASSERT_EQ(e.get_batch({1, 2, 3, 4, 5}), std::string("20"));
}
TEST(DataTest, LambdaAppliesFunctionToExample) {
  // 创建一个虚拟数据集，并对其应用 Lambda 变换，将整数转换为字符串
  auto d = DummyDataset().map(transforms::Lambda<int, std::string>(
      static_cast<std::string (*)(int)>(std::to_string)));
  // 预期的转换后的字符串向量
  std::vector<std::string> expected = {"1", "2", "3", "4", "5"};
  // 断言转换后的数据集能够按索引获取预期的结果
  ASSERT_EQ(d.get_batch({0, 1, 2, 3, 4}), expected);
}

TEST(DataTest, CollateReducesBatch) {
  // 创建一个虚拟数据集，并对其应用 Collate 变换，将输入向量的元素求和
  auto d =
      DummyDataset().map(transforms::Collate<int>([](std::vector<int> input) {
        return std::accumulate(input.begin(), input.end(), 0);
      }));
  // 断言数据集的批次按预期被合并成一个总和
  ASSERT_EQ(d.get_batch({1, 2, 3, 4, 5}), 20);
}

TEST(DataTest, CollationReducesBatch) {
  // 定义一个自定义的 Collation 类，重写 apply_batch 方法以实现输入向量的求和
  struct Summer : transforms::Collation<int> {
    int apply_batch(std::vector<int> input) override {
      return std::accumulate(input.begin(), input.end(), 0);
    }
  };
  // 创建一个虚拟数据集，并使用自定义的 Summer Collation 对象进行映射
  auto d = DummyDataset().map(Summer{});
  // 断言数据集的批次按预期被合并成一个总和
  ASSERT_EQ(d.get_batch({1, 2, 3, 4, 5}), 20);
}

TEST(DataTest, SequentialSamplerReturnsIndicesInOrder) {
  // 创建一个顺序采样器，指定容量为10
  samplers::SequentialSampler sampler(10);
  // 断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(sampler.next(3).value(), std::vector<size_t>({0, 1, 2}));
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({3, 4, 5, 6, 7}));
  ASSERT_EQ(sampler.next(2).value(), std::vector<size_t>({8, 9}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SequentialSamplerReturnsLessValuesForLastBatch) {
  // 创建一个顺序采样器，指定容量为5
  samplers::SequentialSampler sampler(5);
  // 断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(sampler.next(3).value(), std::vector<size_t>({0, 1, 2}));
  // 断言顺序采样器对于超出容量的请求也能正确返回索引
  ASSERT_EQ(sampler.next(100).value(), std::vector<size_t>({3, 4}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SequentialSamplerResetsWell) {
  // 创建一个顺序采样器，指定容量为5
  samplers::SequentialSampler sampler(5);
  // 断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({0, 1, 2, 3, 4}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器状态
  sampler.reset();
  // 再次断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({0, 1, 2, 3, 4}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SequentialSamplerResetsWithNewSizeWell) {
  // 创建一个顺序采样器，指定容量为5
  samplers::SequentialSampler sampler(5);
  // 断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({0, 1, 2, 3, 4}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器并指定新的容量为7
  sampler.reset(7);
  // 断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(
      sampler.next(7).value(), std::vector<size_t>({0, 1, 2, 3, 4, 5, 6}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器并指定新的容量为3
  sampler.reset(3);
  // 断言顺序采样器返回的下一个批次索引与预期的索引顺序相匹配
  ASSERT_EQ(sampler.next(3).value(), std::vector<size_t>({0, 1, 2}));
  // 断言采样器无法继续返回更多数据
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, CanSaveAndLoadSequentialSampler) {
  {
    // 创建一个顺序采样器，指定容量为10
    samplers::SequentialSampler a(10);
    // 断言采样器的当前索引为0
    ASSERT_EQ(a.index(), 0);
    // 创建一个内存流对象，用于序列化采样器
    std::stringstream stream;
    torch::save(a, stream);

    // 创建另一个顺序采样器，指定容量为10，并从流中加载保存的状态
    samplers::SequentialSampler b(10);
    torch::load(b, stream);
    // 断言加载后的采样器索引与原始状态一致
    ASSERT_EQ(b.index(), 0);
  }
  {
    // 创建一个顺序采样器，指定容量为10
    samplers::SequentialSampler a(10);
    // 执行几次采样操作，使索引达到7
    a.next(3);
    a.next(4);
    // 断言采样器的当前索引为7
    ASSERT_EQ(a.index(), 7);
    // 创建一个内存流对象，用于序列化采样器
    std::stringstream stream;
    torch::save(a, stream);

    // 创建另一个顺序采样器，指定容量为10，并从流中加载保存的状态
    samplers::SequentialSampler b(10);
    torch::load(b, stream);
    // 断言加载后的采样器索引与原始状态一致
    ASSERT_EQ(b.index(), 7);
  }
}
TEST(DataTest, RandomSamplerReturnsIndicesInCorrectRange) {
  // 创建一个包含10个元素的随机采样器对象
  samplers::RandomSampler sampler(10);

  // 获取3个随机索引
  std::vector<size_t> indices = sampler.next(3).value();
  // 遍历索引，确保它们在正确的范围内
  for (auto i : indices) {
    ASSERT_GE(i, 0);  // 确保索引大于等于0
    ASSERT_LT(i, 10);  // 确保索引小于10
  }

  // 获取5个随机索引
  indices = sampler.next(5).value();
  // 遍历索引，确保它们在正确的范围内
  for (auto i : indices) {
    ASSERT_GE(i, 0);  // 确保索引大于等于0
    ASSERT_LT(i, 10);  // 确保索引小于10
  }

  // 获取2个随机索引
  indices = sampler.next(2).value();
  // 遍历索引，确保它们在正确的范围内
  for (auto i : indices) {
    ASSERT_GE(i, 0);  // 确保索引大于等于0
    ASSERT_LT(i, 10);  // 确保索引小于10
  }

  // 确保请求10个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(10).has_value());
}

TEST(DataTest, RandomSamplerReturnsLessValuesForLastBatch) {
  // 创建一个包含5个元素的随机采样器对象
  samplers::RandomSampler sampler(5);
  // 确保返回的下一个采样批次大小为3
  ASSERT_EQ(sampler.next(3).value().size(), 3);
  // 确保返回的下一个采样批次大小为100时，返回的大小为2
  ASSERT_EQ(sampler.next(100).value().size(), 2);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, RandomSamplerResetsWell) {
  // 创建一个包含5个元素的随机采样器对象
  samplers::RandomSampler sampler(5);
  // 确保返回的下一个采样批次大小为5
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器
  sampler.reset();
  // 再次确保返回的下一个采样批次大小为5
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, RandomSamplerResetsWithNewSizeWell) {
  // 创建一个包含5个元素的随机采样器对象
  samplers::RandomSampler sampler(5);
  // 确保返回的下一个采样批次大小为5
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器并设置新的大小为7
  sampler.reset(7);
  // 确保返回的下一个采样批次大小为7
  ASSERT_EQ(sampler.next(7).value().size(), 7);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
  // 再次重置采样器并设置新的大小为3
  sampler.reset(3);
  // 确保返回的下一个采样批次大小为3
  ASSERT_EQ(sampler.next(3).value().size(), 3);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SavingAndLoadingRandomSamplerYieldsSameSequence) {
  {
    // 创建一个包含10个元素的随机采样器对象
    samplers::RandomSampler a(10);

    // 将采样器a保存到流中
    std::stringstream stream;
    torch::save(a, stream);

    // 创建一个新的包含10个元素的随机采样器对象b，并从相同的流中加载数据
    samplers::RandomSampler b(10);
    torch::load(b, stream);

    // 确保采样器a和b返回相同的10个随机索引序列
    ASSERT_EQ(a.next(10).value(), b.next(10).value());
  }
  {
    // 创建一个包含10个元素的随机采样器对象a
    samplers::RandomSampler a(10);
    // 获取3个随机索引
    a.next(3);
    // 确保a当前的索引为3
    ASSERT_EQ(a.index(), 3);

    // 将采样器a保存到流中
    std::stringstream stream;
    torch::save(a, stream);

    // 创建一个新的包含10个元素的随机采样器对象b，并从相同的流中加载数据
    samplers::RandomSampler b(10);
    torch::load(b, stream);
    // 确保b当前的索引为3
    ASSERT_EQ(b.index(), 3);

    // 获取b的下一个随机序列，并确保其大小为7
    auto b_sequence = b.next(10).value();
    ASSERT_EQ(b_sequence.size(), 7);
    // 确保a和b返回相同的随机序列
    ASSERT_EQ(a.next(10).value(), b_sequence);
  }
}

TEST(DataTest, StreamSamplerReturnsTheBatchSizeAndThenRemainder) {
  // 创建一个流采样器对象，指定总的数据集大小为100
  samplers::StreamSampler sampler(/*epoch_size=*/100);
  // 确保返回的下一个采样批次大小为10
  ASSERT_EQ(sampler.next(10).value(), 10);
  // 确保返回的下一个采样批次大小为2
  ASSERT_EQ(sampler.next(2).value(), 2);
  // 确保返回的下一个采样批次大小为85
  ASSERT_EQ(sampler.next(85).value(), 85);
  // 确保返回的下一个采样批次大小为123时，实际返回3
  ASSERT_EQ(sampler.next(123).value(), 3);
  // 确保请求1个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(1).has_value());
}

TEST(DataTest, StreamSamplerResetsWell) {
  // 创建一个流采样器对象，指定总的数据集大小为5
  samplers::StreamSampler sampler(/*epoch_size=*/5);
  // 确保返回的下一个采样批次大小为5
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器
  sampler.reset();
  // 再次确保返回的下一个采样批次大小为5
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  // 确保请求2个以上的索引时，返回值为空
  ASSERT_FALSE(sampler.next(2).has_value());
}
TEST(DataTest, StreamSamplerResetsWithNewSizeWell) {
  // 创建一个流采样器对象，指定初始大小为5
  samplers::StreamSampler sampler(/*epoch_size=*/5);
  // 断言下一次采样返回的数据大小为5
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  // 断言下一次采样失败，即没有返回值
  ASSERT_FALSE(sampler.next(2).has_value());
  // 重置采样器大小为7
  sampler.reset(7);
  // 再次断言下一次采样返回的数据大小为7
  ASSERT_EQ(sampler.next(7).value().size(), 7);
  // 再次断言下一次采样失败，即没有返回值
  ASSERT_FALSE(sampler.next(2).has_value());
  // 再次重置采样器大小为3
  sampler.reset(3);
  // 再次断言下一次采样返回的数据大小为3
  ASSERT_EQ(sampler.next(3).value().size(), 3);
  // 最后断言下一次采样失败，即没有返回值
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, TensorDatasetConstructsFromSingleTensor) {
  // 使用单个张量构造张量数据集
  datasets::TensorDataset dataset(torch::eye(5));
  // 断言数据集中索引为2的数据与预期值接近
  ASSERT_TRUE(
      torch::tensor({0, 0, 1, 0, 0}, torch::kFloat32).allclose(dataset.get(2)));
}

TEST(DataTest, TensorDatasetConstructsFromInitializerListOfTensors) {
  // 使用张量列表初始化构造张量数据集
  std::vector<torch::Tensor> vector = torch::eye(5).chunk(5);
  datasets::TensorDataset dataset(vector);
  // 断言数据集中索引为2的数据与预期值接近
  ASSERT_TRUE(
      torch::tensor({0, 0, 1, 0, 0}, torch::kFloat32).allclose(dataset.get(2)));
}

TEST(DataTest, StackTransformWorksForExample) {
  // 定义结构体 D，继承自 Dataset<D>
  struct D : public datasets::Dataset<D> {
    // 实现获取数据的方法
    Example<> get(size_t index) override {
      return {tensor[index], 1 + tensor[index]};
    }

    // 实现返回数据集大小的方法
    torch::optional<size_t> size() const override {
      return tensor.size(0);
    }

    torch::Tensor tensor{torch::eye(4)};
  };

  // 创建 D 类型的对象，并应用 Stack 变换
  auto d = D().map(transforms::Stack<Example<>>());

  // 获取批次数据，包含索引 0 和 1 的数据
  Example<> batch = d.get_batch({0, 1});
  // 断言批次数据的数据部分与预期张量的切片接近
  ASSERT_TRUE(batch.data.allclose(torch::eye(4).slice(/*dim=*/0, 0, 2)));
  // 断言批次数据的目标部分与预期张量的切片接近
  ASSERT_TRUE(batch.target.allclose(1 + torch::eye(4).slice(/*dim=*/0, 0, 2)));

  // 获取第二个批次数据，包含索引 2 和 3 的数据
  Example<> second = d.get_batch({2, 3});
  // 断言第二个批次数据的数据部分与预期张量的切片接近
  ASSERT_TRUE(second.data.allclose(torch::eye(4).slice(/*dim=*/0, 2, 4)));
  // 断言第二个批次数据的目标部分与预期张量的切片接近
  ASSERT_TRUE(second.target.allclose(1 + torch::eye(4).slice(/*dim=*/0, 2, 4)));
}

TEST(DataTest, StackTransformWorksForTensorExample) {
  // 使用张量初始化构造张量数据集，然后应用 Stack 变换
  auto d = datasets::TensorDataset(torch::eye(4))
               .map(transforms::Stack<TensorExample>());

  // 获取批次数据，包含索引 0 和 1 的数据
  TensorExample batch = d.get_batch({0, 1});
  // 断言批次数据的数据部分与预期张量的切片接近
  ASSERT_TRUE(batch.data.allclose(torch::eye(4).slice(/*dim=*/0, 0, 2)));

  // 获取第二个批次数据，包含索引 2 和 3 的数据
  TensorExample second = d.get_batch({2, 3});
  // 断言第二个批次数据的数据部分与预期张量的切片接近
  ASSERT_TRUE(second.data.allclose(torch::eye(4).slice(/*dim=*/0, 2, 4)));
}

// 模板类不能嵌套在函数中
template <typename Target>
struct T : transforms::TensorTransform<Target> {
  // 重载 () 运算符，实现张量转换
  torch::Tensor operator()(torch::Tensor input) override {
    return input * 2;
  }
};

// 定义 TensorStringDataset 结构体，继承自 Dataset<TensorStringDataset, Example<torch::Tensor, std::string>>
struct TensorStringDataset
    : datasets::
          Dataset<TensorStringDataset, Example<torch::Tensor, std::string>> {
  // 实现获取数据的方法
  Example<torch::Tensor, std::string> get(size_t index) override {
    return {torch::tensor(static_cast<double>(index)), std::to_string(index)};
  }

  // 实现返回数据集大小的方法
  torch::optional<size_t> size() const override {
    return 100;
  }
};
TEST(DataTest, TensorTransformWorksForAnyTargetType) {
  // 创建一个 TensorStringDataset 对象，并应用 T<std::string> 的映射
  auto d = TensorStringDataset().map(T<std::string>{});
  // 获取批量大小为 {1, 2} 的数据批次
  std::vector<Example<torch::Tensor, std::string>> batch = d.get_batch({1, 2});

  // 断言批次大小为 2
  ASSERT_EQ(batch.size(), 2);
  // 断言第一个示例的数据近似等于 torch::tensor(2.0)
  ASSERT_TRUE(batch[0].data.allclose(torch::tensor(2.0)));
  // 断言第一个示例的目标值为 "1"
  ASSERT_EQ(batch[0].target, "1");

  // 断言第二个示例的数据近似等于 torch::tensor(4.0)
  ASSERT_TRUE(batch[1].data.allclose(torch::tensor(4.0)));
  // 断言第二个示例的目标值为 "2"
  ASSERT_EQ(batch[1].target, "2");
}

TEST(DataTest, TensorLambdaWorksforAnyTargetType) {
  // 创建一个 TensorStringDataset 对象，并应用 TensorLambda<std::string> 的映射
  auto d = TensorStringDataset().map(transforms::TensorLambda<std::string>(
      [](torch::Tensor input) { return input * 2; }));
  // 获取批量大小为 {1, 2} 的数据批次
  std::vector<Example<torch::Tensor, std::string>> batch = d.get_batch({1, 2});

  // 断言批次大小为 2
  ASSERT_EQ(batch.size(), 2);
  // 断言第一个示例的数据近似等于 torch::tensor(2.0)
  ASSERT_TRUE(batch[0].data.allclose(torch::tensor(2.0)));
  // 断言第一个示例的目标值为 "1"
  ASSERT_EQ(batch[0].target, "1");

  // 断言第二个示例的数据近似等于 torch::tensor(4.0)
  ASSERT_TRUE(batch[1].data.allclose(torch::tensor(4.0)));
  // 断言第二个示例的目标值为 "2"
  ASSERT_EQ(batch[1].target, "2");
}

struct DummyTensorDataset
    : datasets::Dataset<DummyTensorDataset, Example<torch::Tensor, int>> {
  // 实现获取索引处数据示例的方法
  Example<torch::Tensor, int> get(size_t index) override {
    // 将索引转换为 channels 变量
    const auto channels = static_cast<int64_t>(index);
    // 根据 channels 的值创建不同形状的 torch::Tensor 对象
    torch::Tensor tensor =
        (channels > 0) ? torch::ones({channels, 4, 4}) : torch::ones({4, 4});
    // 返回包含 tensor 和 channels 值的 Example 对象
    return {tensor, static_cast<int>(channels)};
  }

  // 返回数据集的大小为 100
  torch::optional<size_t> size() const override {
    return 100;
  }
};
TEST(DataTest, NormalizeTransform) {
  // 使用 DummyTensorDataset 创建数据集，并应用 Normalize 变换
  auto dataset = DummyTensorDataset().map(transforms::Normalize<int>(0.5, 0.1));

  // 对第一个批次数据进行测试，期望输出为单个样本
  std::vector<Example<torch::Tensor, int>> output = dataset.get_batch(0);
  ASSERT_EQ(output.size(), 1);
  // 验证数据是否按照 Normalize 变换预期进行了处理
  // (1 - 0.5) / 0.1 = 5
  ASSERT_TRUE(output[0].data.allclose(torch::ones({4, 4}) * 5))
      << output[0].data;

  // 对第二个批次数据进行测试，期望输出为单个样本
  output = dataset.get_batch(1);
  ASSERT_EQ(output.size(), 1);
  // 验证数据是否按照 Normalize 变换预期进行了处理
  ASSERT_EQ(output[0].data.size(0), 1);
  ASSERT_TRUE(output[0].data.allclose(torch::ones({1, 4, 4}) * 5))
      << output[0].data;

  // 使用不同的均值和标准差创建新的 DummyTensorDataset，并应用 Normalize 变换
  dataset = DummyTensorDataset().map(
      transforms::Normalize<int>({0.5, 1.5}, {0.1, 0.2}));
  // 对第三个批次数据进行测试，期望输出为单个样本
  output = dataset.get_batch(2);
  ASSERT_EQ(output.size(), 1);
  // 验证数据是否按照 Normalize 变换预期进行了处理
  ASSERT_EQ(output[0].data.size(0), 2);
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
                  .allclose(torch::ones({1, 4, 4}) * 5))
      << output[0].data;
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/1)
                  .allclose(torch::ones({1, 4, 4}) * -2.5))
      << output[0].data;

  // 使用特定均值和标准差创建新的 DummyTensorDataset，并应用 Normalize 变换
  dataset = DummyTensorDataset().map(transforms::Normalize<int>(1.5, 0.2));
  // 对第四个批次数据进行测试，期望输出为单个样本
  output = dataset.get_batch(3);
  ASSERT_EQ(output.size(), 1);
  // 验证数据是否按照 Normalize 变换预期进行了处理
  ASSERT_EQ(output[0].data.size(0), 3);
  ASSERT_TRUE(output[0].data.allclose(torch::ones({3, 4, 4}) * -2.5))
      << output[0].data;

  // 使用不同的均值和标准差创建新的 DummyTensorDataset，并应用 Normalize 变换
  dataset = DummyTensorDataset().map(
      transforms::Normalize<int>({0.5, 1.5, -1.5}, {0.1, 0.2, 0.2}));
  // 对第五个批次数据进行测试，期望输出为单个样本
  output = dataset.get_batch(3);
  ASSERT_EQ(output.size(), 1);
  // 验证数据是否按照 Normalize 变换预期进行了处理
  ASSERT_EQ(output[0].data.size(0), 3);
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
                  .allclose(torch::ones({1, 4, 4}) * 5))
      << output[0].data;
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/1, /*end=*/2)
                  .allclose(torch::ones({1, 4, 4}) * -2.5))
      << output[0].data;
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/2)
                  .allclose(torch::ones({1, 4, 4}) * 12.5))
      << output[0].data;
}
TEST(DataTest, MapDoesNotCopy) {
  // 创建一个不可复制的数据集对象
  auto dataset = UnCopyableDataset()
                     // 对数据集中的每个张量执行加1操作
                     .map(transforms::TensorLambda<>(
                         [](torch::Tensor tensor) { return tensor + 1; }))
                     // 对已经加1的张量执行加2操作
                     .map(transforms::TensorLambda<>(
                         [](torch::Tensor tensor) { return tensor + 2; }))
                     // 对已经加1和加2的张量执行加3操作
                     .map(transforms::TensorLambda<>(
                         [](torch::Tensor tensor) { return tensor + 3; }));

  // 获取一个批次的数据，选择第一个元素的数据
  auto data = dataset.get_batch(1).at(0).data;
  // 断言数据元素个数为1
  ASSERT_EQ(data.numel(), 1);
  // 断言第一个数据元素的浮点值为7
  ASSERT_EQ(data[0].item<float>(), 7);
}

TEST(DataTest, QueuePushAndPopFromSameThread) {
  // 创建一个整型队列对象
  torch::data::detail::Queue<int> queue;
  // 将元素1压入队列
  queue.push(1);
  // 将元素2压入队列
  queue.push(2);
  // 断言队列弹出的第一个元素为1
  ASSERT_EQ(queue.pop(), 1);
  // 断言队列弹出的第二个元素为2
  ASSERT_EQ(queue.pop(), 2);
}

TEST(DataTest, QueuePopWithTimeoutThrowsUponTimeout) {
  // 创建一个整型队列对象
  torch::data::detail::Queue<int> queue;
  // 断言调用队列的pop方法，设置10毫秒超时，并检查超时异常消息
  ASSERT_THROWS_WITH(
      queue.pop(10 * kMillisecond),
      "Timeout in DataLoader queue while waiting for next batch "
      "(timeout was 10 ms)");
}

TEST(DataTest, QueuePushAndPopFromDifferentThreads) {
  // 使用整型队列
  using torch::data::detail::Queue;

  // 第一个测试：在一个线程中压入元素，然后在另一个线程中弹出
  {
    Queue<int> queue;
    // 将元素1压入队列
    queue.push(1);
    // 异步启动一个线程，从队列中弹出元素并返回
    auto future =
        std::async(std::launch::async, [&queue] { return queue.pop(); });
    // 断言异步操作的返回值为1
    ASSERT_EQ(future.get(), 1);
  }

  // 第二个测试：尝试从队列中弹出元素（会阻塞），然后再压入元素
  {
    Queue<int> queue;
    // 在一个线程中延迟20毫秒后，将元素123压入队列
    std::thread thread([&queue] {
      std::this_thread::sleep_for(20 * kMillisecond);
      queue.push(123);
    });
    // 断言从队列中弹出的元素为123
    ASSERT_EQ(queue.pop(), 123);
    thread.join();
  }
}

TEST(DataTest, QueueClearEmptiesTheQueue) {
  // 创建一个整型队列对象
  torch::data::detail::Queue<int> queue;
  // 将元素1、2、3依次压入队列
  queue.push(1);
  queue.push(2);
  queue.push(3);
  // 断言清空队列后返回的元素个数为3
  ASSERT_EQ(queue.clear(), 3);
  // 断言在1毫秒超时内尝试从队列中弹出元素，检查是否抛出超时异常
  ASSERT_THROWS_WITH(queue.pop(1 * kMillisecond), "Timeout");
}

TEST(DataTest, DataShuttleCanPushAndPopJob) {
  // 创建一个数据搬运器对象，搬运整型数据
  torch::data::detail::DataShuttle<int, int> shuttle;
  // 将作业1推送到搬运器中
  shuttle.push_job(1);
  // 将作业2推送到搬运器中
  shuttle.push_job(2);
  // 断言从搬运器中弹出的作业为1
  ASSERT_EQ(shuttle.pop_job(), 1);
  // 断言从搬运器中弹出的作业为2
  ASSERT_EQ(shuttle.pop_job(), 2);
}

TEST(DataTest, DataShuttleCanPushAndPopResult) {
  // 创建一个数据搬运器对象，搬运整型数据
  torch::data::detail::DataShuttle<int, int> shuttle;
  // 只有在推送了作业之后，pop_result()才会尝试弹出结果
  shuttle.push_job(1);
  shuttle.push_job(2);

  // 弹出作业1后，推送结果1
  shuttle.pop_job();
  shuttle.push_result(1);
  // 断言从搬运器中弹出的结果为1
  ASSERT_EQ(shuttle.pop_result().value(), 1);

  // 弹出作业2后，推送结果2
  shuttle.pop_job();
  shuttle.push_result(2);
  // 断言从搬运器中弹出的结果为2
  ASSERT_EQ(shuttle.pop_result().value(), 2);
}

TEST(DataTest, DataShuttlePopResultReturnsNulloptWhenNoJobsInFlight) {
  // 创建一个数据搬运器对象，搬运整型数据
  torch::data::detail::DataShuttle<int, int> shuttle;
  // 断言当没有作业在进行时，pop_result()返回空值
  ASSERT_FALSE(shuttle.pop_result().has_value());
  // 推送作业1到搬运器中
  shuttle.push_job(1);
  // 弹出作业1后，推送结果1
  shuttle.pop_job();
  shuttle.push_result(1);
  // 断言从搬运器中弹出的结果为1
  ASSERT_EQ(shuttle.pop_result().value(), 1);
  // 再次断言当没有作业在进行时，pop_result()返回空值
  ASSERT_FALSE(shuttle.pop_result().has_value());
  // 再次断言当没有作业在进行时，pop_result()返回空值
  ASSERT_FALSE(shuttle.pop_result().has_value());
}
TEST(DataTest, DataShuttleDrainMeansPopResultReturnsNullopt) {
  // 创建一个 DataShuttle 对象，使用 int 作为任务和结果的数据类型
  torch::data::detail::DataShuttle<int, int> shuttle;
  // 向 DataShuttle 对象推送一个任务
  shuttle.push_job(1);
  // 向 DataShuttle 对象推送一个结果
  shuttle.push_result(1);
  // 清空 DataShuttle 对象中的结果队列
  shuttle.drain();
  // 断言 pop_result() 方法返回的 optional 对象不包含值
  ASSERT_FALSE(shuttle.pop_result().has_value());
}

TEST(DataTest, DataShuttlePopResultTimesOut) {
  // 创建一个 DataShuttle 对象，使用 int 作为任务和结果的数据类型
  torch::data::detail::DataShuttle<int, int> shuttle;
  // 向 DataShuttle 对象推送一个任务
  shuttle.push_job(1);
  // 断言在指定的超时时间内调用 pop_result() 方法会抛出异常并包含指定的错误信息
  ASSERT_THROWS_WITH(shuttle.pop_result(10 * kMillisecond), "Timeout");
}

struct UncopyableDataset : datasets::Dataset<UncopyableDataset, int> {
  // 构造函数，接受一个 std::string 参数，但未使用
  UncopyableDataset(const std::string& /* unused */) {}

  // 移动构造函数，默认实现
  UncopyableDataset(UncopyableDataset&&) = default;
  // 移动赋值运算符，默认实现
  UncopyableDataset& operator=(UncopyableDataset&&) = default;

  // 拷贝构造函数，删除
  UncopyableDataset(const UncopyableDataset&) = delete;
  // 拷贝赋值运算符，删除
  UncopyableDataset& operator=(const UncopyableDataset&) = delete;

  // 根据索引返回数据，这里对于每个索引都返回 index + 1
  int get(size_t index) override {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return 1 + index;
  }
  // 返回数据集的大小，这里固定返回 100
  torch::optional<size_t> size() const override {
    return 100;
  }
};

TEST(DataTest, SharedBatchDatasetReallyIsShared) {
  // 创建一个共享的 UncopyableDataset 对象
  auto shared_dataset =
      torch::data::datasets::make_shared_dataset<UncopyableDataset>(
          "uncopyable");

  // 创建一个数据加载器，使用共享的数据集，设置工作线程数为 3
  auto data_loader = torch::data::make_data_loader(
      shared_dataset, torch::data::DataLoaderOptions().workers(3));

  // 遍历数据加载器中的每个批次数据，无特定逻辑，用于确保共享数据集无需复制
  for (auto batch : *data_loader) {
    /* exhaust */
  }
}

TEST(DataTest, SharedBatchDatasetDoesNotIncurCopyWhenPassedDatasetObject) {
  // 创建一个共享的 UncopyableDataset 对象，传递另一个 UncopyableDataset 对象的实例
  auto shared_dataset =
      torch::data::datasets::make_shared_dataset<UncopyableDataset>(
          UncopyableDataset("uncopyable"));
  // 断言共享数据集的大小为 100
  ASSERT_EQ(shared_dataset.size().value(), 100);
}

struct TestIndex : public torch::data::samplers::CustomBatchRequest {
  // 构造函数，接受偏移量和索引向量作为参数
  explicit TestIndex(size_t offset, std::vector<size_t> index)
      : offset(offset), index(std::move(index)) {}
  // 返回索引向量的大小
  size_t size() const override {
    return index.size();
  }
  // 偏移量
  size_t offset;
  // 索引向量
  std::vector<size_t> index;
};

struct TestIndexDataset
    : datasets::BatchDataset<TestIndexDataset, std::vector<int>, TestIndex> {
  // 构造函数，创建一个具有指定大小的数据集，填充连续的整数值
  explicit TestIndexDataset(size_t size) : data(size) {
    std::iota(data.begin(), data.end(), size_t(0));
  }
  // 根据 TestIndex 对象获取批次数据，根据索引和偏移量计算数据值
  std::vector<int> get_batch(TestIndex index) override {
    std::vector<int> batch;
    for (auto i : index.index) {
      batch.push_back(index.offset + data.at(i));
    }
    return batch;
  }
  // 返回数据集的大小，这里返回数据的总数
  torch::optional<size_t> size() const override {
    return data.size();
  }
  // 数据存储
  std::vector<int> data;
};
struct TestIndexSampler : public samplers::Sampler<TestIndex> {
  // 定义一个测试索引采样器，继承自TestIndex类型的采样器
  explicit TestIndexSampler(size_t size) : size_(size) {}
  // 构造函数，初始化采样器的大小
  void reset(torch::optional<size_t> new_size = torch::nullopt) override {}
  // 重置采样器状态的方法，不实际执行任何操作
  torch::optional<TestIndex> next(size_t batch_size) override {
    // 获取下一个批次的采样数据
    if (index_ >= size_) {
      return torch::nullopt;
    }
    // 如果当前索引超过了大小，则返回空
    std::vector<size_t> indices(batch_size);
    // 创建一个大小为batch_size的索引向量
    std::iota(indices.begin(), indices.end(), size_t(0));
    // 将索引向量填充为连续整数，起始值为0
    index_ += batch_size;
    // 更新当前索引
    return TestIndex(batch_size, std::move(indices));
    // 返回一个包含批次大小和索引向量的TestIndex对象
  }
  void save(torch::serialize::OutputArchive& archive) const override {}
  // 序列化保存当前状态的方法，但本例中未实现具体功能
  void load(torch::serialize::InputArchive& archive) override {}
  // 反序列化加载状态的方法，但本例中未实现具体功能
  size_t index_ = 0;
  // 当前索引的起始值为0
  size_t size_;
  // 采样器的大小
};

TEST(DataTest, CanUseCustomTypeAsIndexType) {
  // 测试用例：验证是否可以使用自定义类型作为索引类型
  const int kBatchSize = 10;
  auto data_loader = torch::data::make_data_loader(
      TestIndexDataset(23), TestIndexSampler(23), kBatchSize);

  for (auto batch : *data_loader) {
    // 遍历数据加载器中的每个批次
    for (const auto j : c10::irange(kBatchSize)) {
      // 遍历当前批次中的每个元素
      ASSERT_EQ(batch.at(j), 10 + j);
      // 断言当前元素的值是否符合预期
    }
  }
}

TEST(DataTest, DistributedRandomSamplerSingleReplicaProduceCorrectSamples) {
  // 测试用例：分布式随机采样器在单个副本上产生正确的样本
  size_t sample_count = 10;
  samplers::DistributedRandomSampler drs(sample_count);

  std::vector<size_t> res;
  torch::optional<std::vector<size_t>> idx;
  while ((idx = drs.next(3)).has_value()) {
    // 循环获取下一个批次的采样索引
    res.insert(std::end(res), std::begin(*idx), std::end(*idx));
    // 将当前批次的索引添加到结果向量中
  }

  ASSERT_EQ(res.size(), sample_count);
  // 断言结果向量的大小是否等于样本总数

  std::sort(res.begin(), res.end());
  // 对结果向量进行排序
  for (const auto i : c10::irange(res.size())) {
    // 遍历结果向量中的每个元素
    ASSERT_EQ(res[i], i);
    // 断言排序后的索引是否与期望值相等
  }
}

TEST(DataTest, DistributedRandomSamplerMultiReplicaProduceCorrectSamples) {
  // 测试用例：分布式随机采样器在多个副本上产生正确的样本
  size_t sample_count = 10;
  size_t num_replicas = 3;

  auto test_function = [&](bool allow_duplicates,
                           size_t local_sample_count,
                           std::vector<size_t>& output,
                           size_t batch_size) {
    // 定义测试函数，用于验证多副本情况下的采样结果
    std::vector<std::unique_ptr<samplers::DistributedRandomSampler>> samplers;

    for (const auto i : c10::irange(num_replicas)) {
      // 循环创建多个分布式随机采样器
      samplers.emplace_back(
          std::make_unique<samplers::DistributedRandomSampler>(
              sample_count, num_replicas, i, allow_duplicates));
    }

    std::vector<size_t> res;
    for (const auto i : c10::irange(num_replicas)) {
      // 遍历每个副本的采样器
      (*samplers[i]).reset();
      // 重置当前副本的采样器状态
      torch::optional<std::vector<size_t>> idx;
      while ((idx = (*samplers[i]).next(batch_size)).has_value()) {
        // 循环获取当前副本下一个批次的采样索引
        res.insert(std::end(res), std::begin(*idx), std::end(*idx));
        // 将当前批次的索引添加到结果向量中
      }
      ASSERT_EQ(res.size(), local_sample_count * (i + 1));
      // 断言结果向量的大小是否符合预期
    }
    std::sort(res.begin(), res.end());
    // 对结果向量进行排序
    ASSERT_EQ(res, output);
    // 断言排序后的结果向量是否与预期输出一致
  };

  for (size_t batch_size = 1; batch_size <= 3; ++batch_size) {
    // 遍历不同的批次大小
    size_t local_sample_count =
        static_cast<size_t>(std::ceil(sample_count * 1.0 / num_replicas));
    // 计算每个副本中期望的本地样本数
    std::vector<size_t> output1{0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // 预期的输出结果向量
    test_function(true, local_sample_count, output1, batch_size);
    // 调用测试函数，验证多副本采样的结果
    local_sample_count =
        static_cast<size_t>(std::floor(sample_count * 1.0 / num_replicas));
    # 计算每个副本的本地样本数量，确保是整数
    std::vector<size_t> output2{0, 1, 2, 3, 4, 5, 6, 7, 8};
    # 创建一个包含初始值的大小为9的无符号整数向量
    test_function(false, local_sample_count, output2, batch_size);
    # 调用名为test_function的函数，传入参数为false（布尔值）、本地样本数量、output2向量和batch_size变量
}

TEST(DataTest, CanSaveAndLoadDistributedRandomSampler) {
  {
    // 创建一个分布式随机采样器对象 a，总共有 10 个样本
    samplers::DistributedRandomSampler a(10);
    // 断言：检查对象 a 的索引是否为 0
    ASSERT_EQ(a.index(), 0);
    // 创建一个字符串流对象 stream
    std::stringstream stream;
    // 将采样器对象 a 序列化保存到流 stream 中
    torch::save(a, stream);

    // 创建另一个分布式随机采样器对象 b，总共有 10 个样本
    samplers::DistributedRandomSampler b(10);
    // 从流 stream 中加载数据到对象 b
    torch::load(b, stream);
    // 断言：检查对象 b 的索引是否为 0
    ASSERT_EQ(b.index(), 0);
  }
  {
    // 创建一个分布式随机采样器对象 a，总共有 10 个样本
    samplers::DistributedRandomSampler a(10);
    // 调用 next 方法，索引增加 3
    a.next(3);
    // 继续调用 next 方法，索引增加 4
    a.next(4);
    // 断言：检查对象 a 的索引是否为 7
    ASSERT_EQ(a.index(), 7);
    // 创建一个字符串流对象 stream
    std::stringstream stream;
    // 将采样器对象 a 序列化保存到流 stream 中
    torch::save(a, stream);

    // 创建另一个分布式随机采样器对象 b，总共有 10 个样本
    samplers::DistributedRandomSampler b(10);
    // 从流 stream 中加载数据到对象 b
    torch::load(b, stream);
    // 断言：检查对象 b 的索引是否为 7
    ASSERT_EQ(b.index(), 7);
  }
  {
    // 创建一个分布式随机采样器对象 a，总共有 10 个样本
    samplers::DistributedRandomSampler a(10);
    // 设置对象 a 的 epoch 为 3
    a.set_epoch(3);
    // 创建一个字符串流对象 stream
    std::stringstream stream;
    // 将采样器对象 a 序列化保存到流 stream 中
    torch::save(a, stream);

    // 创建另一个分布式随机采样器对象 b，总共有 10 个样本
    samplers::DistributedRandomSampler b(10);
    // 从流 stream 中加载数据到对象 b
    torch::load(b, stream);
    // 断言：检查对象 b 的 epoch 是否为 3
    ASSERT_EQ(b.epoch(), 3);
  }
}

TEST(DataTest, DistributedSequentialSamplerSingleReplicaProduceCorrectSamples) {
  size_t sample_count = 10;
  size_t batch_size = 3;
  // 创建一个分布式顺序采样器对象 dss，总共有 10 个样本
  samplers::DistributedSequentialSampler dss(sample_count);

  // 创建一个空的整数向量 res
  std::vector<size_t> res;
  // 创建一个可选的整数向量 idx
  torch::optional<std::vector<size_t>> idx;
  // 当 dss 采样下一个批次的索引存在时，执行循环
  while ((idx = dss.next(batch_size)).has_value()) {
    // 将 idx 的内容插入到 res 的末尾
    res.insert(std::end(res), std::begin(*idx), std::end(*idx));
  }

  // 断言：检查 res 的大小是否等于样本总数
  ASSERT_EQ(res.size(), sample_count);

  // 对 res 中的元素进行排序
  std::sort(res.begin(), res.end());
  // 对于 res 中的每个元素 i，在 c10 命名空间的范围内执行断言：检查 res[i] 是否等于 i
  for (const auto i : c10::irange(res.size())) {
    ASSERT_EQ(res[i], i);
  }
}

TEST(DataTest, DistributedSequentialSamplerMultiReplicaProduceCorrectSamples) {
  size_t sample_count = 10;
  size_t num_replicas = 3;

  auto test_function = [&](bool allow_duplicates,
                           size_t local_sample_count,
                           std::vector<size_t>& output,
                           size_t batch_size) {
    // 创建一个空的唯一指针分布式顺序采样器对象的向量 samplers
    std::vector<std::unique_ptr<samplers::DistributedSequentialSampler>>
        samplers;

    // 对于 num_replicas 范围内的每个索引 i，执行以下操作
    for (const auto i : c10::irange(num_replicas)) {
      // 在 samplers 向量末尾插入一个新的分布式顺序采样器对象
      samplers.emplace_back(
          std::make_unique<samplers::DistributedSequentialSampler>(
              sample_count, num_replicas, i, allow_duplicates));
    }

    // 创建一个空的整数向量 res
    std::vector<size_t> res;
    // 对于 num_replicas 范围内的每个索引 i，执行以下操作
    for (const auto i : c10::irange(num_replicas)) {
      // 重置第 i 个采样器对象
      (*samplers[i]).reset();
      // 创建一个可选的整数向量 idx
      torch::optional<std::vector<size_t>> idx;
      // 当 (*samplers[i]) 采样下一个批次的索引存在时，执行循环
      while ((idx = (*samplers[i]).next(batch_size)).has_value()) {
        // 将 idx 的内容插入到 res 的末尾
        res.insert(std::end(res), std::begin(*idx), std::end(*idx));
      }
      // 断言：检查 res 的大小是否等于 local_sample_count * (i + 1)
      ASSERT_EQ(res.size(), local_sample_count * (i + 1));
    }
    // 对 res 中的元素进行排序
    std::sort(res.begin(), res.end());
    // 断言：检查 res 是否等于 output
    ASSERT_EQ(res, output);
  };

  // 对于 batch_size 从 1 到 3 的范围内的每个值，执行以下操作
  for (size_t batch_size = 1; batch_size <= 3; ++batch_size) {
    // 计算本地样本数，向上取整
    size_t local_sample_count =
        static_cast<size_t>(std::ceil(sample_count * 1.0 / num_replicas));
    // 创建一个整数向量 output1，表示预期的输出
    std::vector<size_t> output1{0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // 调用 test_function，传入 true、local_sample_count、output1 和 batch_size
    test_function(true, local_sample_count, output1, batch_size);

    // 计算本地样本数，向下取整
    local_sample_count =
        static_cast<size_t>(std::floor(sample_count * 1.0 / num_replicas));
    // 创建一个整数向量 output2，表示预期的输出
    std::vector<size_t> output2{0, 1, 2, 3, 4, 5, 6, 7, 8};
    // 调用 test_function，传入 true、local_sample_count、output2 和 batch_size
    test_function(true, local_sample_count, output2, batch_size);
    test_function(false, local_sample_count, output2, batch_size);



// 调用名为 test_function 的函数，传入四个参数：false, local_sample_count, output2, batch_size
test_function(false, local_sample_count, output2, batch_size);
TEST(DataTest, CanSaveAndLoadDistributedSequentialSampler) {
  {  // 开始第一个测试用例
    // 创建 DistributedSequentialSampler 实例 a，长度为 10
    samplers::DistributedSequentialSampler a(10);
    // 断言 a 的当前索引为 0
    ASSERT_EQ(a.index(), 0);
    // 创建一个 stringstream 对象 stream
    std::stringstream stream;
    // 将 sampler a 序列化保存到 stream
    torch::save(a, stream);

    // 创建 DistributedSequentialSampler 实例 b，长度为 10
    samplers::DistributedSequentialSampler b(10);
    // 从 stream 中加载数据到 sampler b
    torch::load(b, stream);
    // 断言 b 的当前索引为 0
    ASSERT_EQ(b.index(), 0);
  }  // 结束第一个测试用例

  {  // 开始第二个测试用例
    // 创建 DistributedSequentialSampler 实例 a，长度为 10
    samplers::DistributedSequentialSampler a(10);
    // 执行两次 next 方法，索引分别增加 3 和 4
    a.next(3);
    a.next(4);
    // 断言 a 的当前索引为 7
    ASSERT_EQ(a.index(), 7);
    // 创建一个 stringstream 对象 stream
    std::stringstream stream;
    // 将 sampler a 序列化保存到 stream
    torch::save(a, stream);

    // 创建 DistributedSequentialSampler 实例 b，长度为 10
    samplers::DistributedSequentialSampler b(10);
    // 从 stream 中加载数据到 sampler b
    torch::load(b, stream);
    // 断言 b 的当前索引为 7
    ASSERT_EQ(b.index(), 7);
  }  // 结束第二个测试用例
}

TEST(DataLoaderTest, DataLoaderOptionsDefaultAsExpected) {
  // 创建一个空的 DataLoaderOptions 对象 partial_options
  DataLoaderOptions partial_options;
  // 用 partial_options 创建 FullDataLoaderOptions 对象 full_options
  FullDataLoaderOptions full_options(partial_options);
  // 断言 batch_size 为默认值 1
  ASSERT_EQ(full_options.batch_size, 1);
  // 断言 drop_last 为 false
  ASSERT_FALSE(full_options.drop_last);
  // 断言 workers 为 0
  ASSERT_EQ(full_options.workers, 0);
  // 断言 max_jobs 为 0
  ASSERT_EQ(full_options.max_jobs, 0);
  // 断言 timeout 未设置
  ASSERT_FALSE(full_options.timeout.has_value());
  // 断言 enforce_ordering 为 true
  ASSERT_TRUE(full_options.enforce_ordering);
}

TEST(DataLoaderTest, DataLoaderOptionsCoalesceOptionalValues) {
  // 创建 DataLoaderOptions 对象 partial_options，设置 batch_size 为 32，workers 为 10
  auto partial_options = DataLoaderOptions(32).workers(10);
  // 使用 partial_options 创建 FullDataLoaderOptions 对象 full_options
  FullDataLoaderOptions full_options(partial_options);
  // 断言 batch_size 为 32
  ASSERT_EQ(full_options.batch_size, 32);
  // 断言 max_jobs 为 workers 的两倍，即 20
  ASSERT_EQ(full_options.max_jobs, 2 * 10);
}

TEST(DataLoaderTest, MakeDataLoaderDefaultsAsExpected) {
  // 使用 DummyDataset 创建一个数据加载器 data_loader，使用默认转换 Lambda<int> 对象
  auto data_loader = torch::data::make_data_loader(
      DummyDataset().map(transforms::Lambda<int>([](int x) { return x + 1; })));
  // 断言数据加载器的 batch_size 为默认值 1
  ASSERT_EQ(data_loader->options().batch_size, 1);
}

struct UnsizedDataset : public datasets::Dataset<UnsizedDataset> {
  torch::data::Example<> get(size_t i) override {
    return {torch::ones(i), torch::ones(i)};
  }
  torch::optional<size_t> size() const noexcept override {
    return torch::nullopt;
  }
};

TEST(
    DataLoaderTest,
    MakeDataLoaderThrowsWhenConstructingSamplerWithUnsizedDataset) {
  // 断言在使用 UnsizedDataset 构造数据加载器时抛出异常，指定异常信息
  ASSERT_THROWS_WITH(
      torch::data::make_data_loader(UnsizedDataset{}),
      "Expected the dataset to be sized in order to construct the Sampler");
}

TEST(DataLoaderTest, IteratorsCompareEqualToThemselves) {
  // 创建一个数据加载器 data_loader，使用 DummyDataset，batch_size 为 32
  auto data_loader = torch::data::make_data_loader(DummyDataset(), 32);
  // 获取 data_loader 的起始迭代器 begin
  auto begin = data_loader->begin();
  // 断言 begin 等于自身
  ASSERT_EQ(begin, begin);
  // 获取 data_loader 的结束迭代器 end
  auto end = data_loader->end();
  // 断言 end 等于自身
  ASSERT_EQ(end, end);
}

TEST(DataLoaderTest, ValidIteratorsCompareUnequalToEachOther) {
  // 创建一个数据加载器 data_loader，使用 DummyDataset，batch_size 为 32
  auto data_loader = torch::data::make_data_loader(DummyDataset(), 32);
  // 获取 data_loader 的第一个迭代器 i
  auto i = data_loader->begin();
  // 获取 data_loader 的第二个迭代器 j
  auto j = data_loader->begin();
  // 断言 i 不等于 j
  ASSERT_NE(i, j);
  // 将 j 向前移动一个元素
  ++j;
  // 断言 i 不等于 j
  ASSERT_NE(i, j);
}

TEST(DataLoaderTest, SentinelIteratorsCompareEqualToEachOther) {
  // 创建一个数据加载器 data_loader，使用 DummyDataset，batch_size 为 32
  auto data_loader = torch::data::make_data_loader(DummyDataset(), 32);
  // 获取 data_loader 的结束迭代器 i
  auto i = data_loader->end();
  // 获取 data_loader 的结束迭代器 j
  auto j = data_loader->end();
  // 断言 i 等于 j
  ASSERT_EQ(i, j);
}
TEST(DataLoaderTest, IteratorsCompareEqualToSentinelWhenExhausted) {
  // 创建一个虚拟的数据集对象
  DummyDataset dataset;
  // 使用数据集创建一个数据加载器，每次加载数据集大小的四分之一
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value() / 4);
  // 获取数据加载器的起始迭代器和结束迭代器
  auto i = data_loader->begin();
  auto end = data_loader->end();
  // 断言起始迭代器和结束迭代器不相等
  ASSERT_NE(i, end);
  // 向前移动迭代器
  ++i;
  // 再次断言迭代器不等于结束迭代器
  ASSERT_NE(i, end);
  // 继续向前移动迭代器
  ++i;
  // 再次断言迭代器不等于结束迭代器
  ASSERT_NE(i, end);
  // 继续向前移动迭代器
  ++i;
  // 再次断言迭代器不等于结束迭代器
  ASSERT_NE(i, end);
  // 继续向前移动迭代器，使其达到结束迭代器
  ++i;
  // 最后断言迭代器等于结束迭代器
  ASSERT_EQ(i, end);
}

TEST(DataLoaderTest, IteratorsShareState) {
  // 创建一个虚拟的数据集对象
  DummyDataset dataset;
  // 使用数据集创建一个数据加载器，每次加载数据集大小的一半
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value() / 2);
  // 获取数据加载器的起始迭代器
  auto i = data_loader->begin();
  // 将 j 设置为 i 的副本
  auto j = i;
  // 获取数据加载器的结束迭代器
  auto end = data_loader->end();
  // 断言起始迭代器和结束迭代器不相等
  ASSERT_NE(i, end);
  ASSERT_NE(j, end);
  // 向前移动 i 迭代器
  ++i;
  // 再次断言 i 和 j 迭代器不等于结束迭代器
  ASSERT_NE(i, end);
  ASSERT_NE(j, end);
  // 向前移动 j 迭代器
  ++j;
  // 断言 i 和 j 迭代器现在都等于结束迭代器
  ASSERT_EQ(i, end);
  ASSERT_EQ(j, end);
}

TEST(DataLoaderTest, CanDereferenceIteratorMultipleTimes) {
  // 创建一个虚拟的数据集对象
  DummyDataset dataset;
  // 使用数据集创建一个数据加载器，并指定使用顺序采样器
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          dataset,
          // NOLINTNEXTLINE(bugprone-argument-comment)
          /*batch_size=*/1);
  // 获取数据加载器的起始迭代器
  auto iterator = data_loader->begin();
  // 创建一个预期的整数向量
  std::vector<int> expected = {1};
  // 断言迭代器解引用的结果等于预期值
  ASSERT_EQ(*iterator, expected);
  // 再次断言迭代器解引用的结果仍然等于预期值
  ASSERT_EQ(*iterator, expected);
  // 向前移动迭代器
  ++iterator;
  // 更新预期值
  expected[0] = 2;
  // 再次断言迭代器解引用的结果等于更新后的预期值
  ASSERT_EQ(*iterator, expected);
  // 再次断言迭代器解引用的结果仍然等于更新后的预期值
  ASSERT_EQ(*iterator, expected);
  // 向前移动迭代器
  ++iterator;
  // 更新预期值
  expected[0] = 3;
  // 再次断言迭代器解引用的结果等于更新后的预期值
  ASSERT_EQ(*iterator, expected);
  // 再次断言迭代器解引用的结果仍然等于更新后的预期值
  ASSERT_EQ(*iterator, expected);
}

TEST(DataLoaderTest, CanUseIteratorAlgorithms) {
  // 定义一个匿名结构体，继承自 BatchDataset
  struct D : datasets::BatchDataset<D, int> {
    // 重写获取批次数据的方法
    int get_batch(torch::ArrayRef<size_t> indices) override {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      // 返回索引首位加一的结果
      return 1 + indices.front();
    }
    // 重写获取数据集大小的方法
    torch::optional<size_t> size() const override {
      // 返回数据集大小为 10
      return 10;
    }
  };

  // 创建匿名结构体对象
  D dataset;
  // 使用数据集创建一个数据加载器，并指定使用顺序采样器
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          dataset, 1);
  // 创建一个空整数向量 values
  std::vector<int> values;
  // 使用 std::copy 算法将数据加载器的数据拷贝到 values 向量中
  std::copy(
      data_loader->begin(), data_loader->end(), std::back_inserter(values));
  // 创建一个期望的整数向量 expected，其元素为从 1 到数据集大小的递增序列
  std::vector<int> expected(dataset.size().value());
  std::iota(expected.begin(), expected.end(), size_t(1));
  // 断言 values 向量等于期望的整数向量 expected
  ASSERT_EQ(values, expected);
}

TEST(DataLoaderTest, CallingBeginWhileOtherIteratorIsInFlightThrows) {
  // 创建一个虚拟的数据集对象
  DummyDataset dataset;
  // 使用数据集创建一个数据加载器，并设置工作线程数为 2
  auto data_loader =
      torch::data::make_data_loader(dataset, DataLoaderOptions(1).workers(2));
  // 获取数据加载器的起始迭代器
  auto i = data_loader->begin();
  // 断言调用 begin() 方法时抛出异常，指示在另一个迭代器尚未用完时尝试获取新的数据加载器迭代器
  ASSERT_THROWS_WITH(
      data_loader->begin(),
      "Attempted to get a new DataLoader iterator "
      "while another iterator is not yet exhausted");
}

TEST(DataLoaderTest, IncrementingExhaustedValidIteratorThrows) {
  // 创建一个虚拟的数据集对象
  DummyDataset dataset;
  // 使用数据集创建一个数据加载器，每次加载整个数据集
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  // 获取数据加载器的起始迭代器
  auto i = data_loader->begin();
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言递增已用完的有效迭代器时不会抛出异常
  ASSERT_NO_THROW(++i);
  // 断言递增已用完的有效迭代器时抛出异常，指示试图超出末尾的迭代器
  ASSERT_THROWS_WITH(++i, "Attempted to increment iterator past the end");
}
TEST(DataLoaderTest, DereferencingExhaustedValidIteratorThrows) {
  // 创建 DummyDataset 对象
  DummyDataset dataset;
  // 创建数据加载器，使用 DummyDataset 的大小作为批量大小
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  // 获取数据加载器的起始迭代器
  auto i = data_loader->begin();
  // 禁止编译器警告（忽略特定规则），允许使用 ++i 操作
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言在向前移动迭代器时不会抛出异常
  ASSERT_NO_THROW(++i);
  // 断言试图解引用超出末尾的迭代器时会抛出特定消息的异常
  ASSERT_THROWS_WITH(
      *i, "Attempted to dereference iterator that was past the end");
}

TEST(DataLoaderTest, IncrementingSentinelIteratorThrows) {
  // 创建 DummyDataset 对象
  DummyDataset dataset;
  // 创建数据加载器，使用 DummyDataset 的大小作为批量大小
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  // 获取数据加载器的结束迭代器
  auto i = data_loader->end();
  // 断言试图递增超出末尾的迭代器时会抛出特定消息的异常
  ASSERT_THROWS_WITH(
      ++i,
      "Incrementing the DataLoader's past-the-end iterator is not allowed");
}

TEST(DataLoaderTest, DereferencingSentinelIteratorThrows) {
  // 创建 DummyDataset 对象
  DummyDataset dataset;
  // 创建数据加载器，使用 DummyDataset 的大小作为批量大小
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  // 获取数据加载器的结束迭代器
  auto i = data_loader->end();
  // 断言试图解引用超出末尾的迭代器时会抛出特定消息的异常
  ASSERT_THROWS_WITH(
      *i,
      "Dereferencing the DataLoader's past-the-end iterator is not allowed");
}

TEST(DataLoaderTest, YieldsCorrectBatchSize) {
  // 创建 DummyDataset 对象
  DummyDataset dataset;
  // 创建数据加载器，批量大小为 25
  auto data_loader = torch::data::make_data_loader(dataset, 25);
  // 获取数据加载器的起始迭代器
  auto iterator = data_loader->begin();
  // 断言第一个批次的大小为 25
  ASSERT_EQ(iterator->size(), 25);
  // 递增迭代器并断言其大小为 25
  ASSERT_EQ((++iterator)->size(), 25);
  // 递增迭代器并断言其大小为 25
  ASSERT_EQ((++iterator)->size(), 25);
  // 递增迭代器并断言其大小为 25
  ASSERT_EQ((++iterator)->size(), 25);
  // 断言迭代器已到达末尾
  ASSERT_EQ(++iterator, data_loader->end());
}

TEST(
    DataLoaderTest,
    ReturnsLastBatchWhenSmallerThanBatchSizeWhenDropLastIsFalse) {
  // 创建 DummyDataset 对象
  DummyDataset dataset;
  // 创建数据加载器，批量大小为 33，并设置不丢弃最后一个批次
  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(33).drop_last(false));
  // 获取数据加载器的起始迭代器
  auto iterator = data_loader->begin();
  // 断言第一个批次的大小为 33
  ASSERT_EQ(iterator->size(), 33);
  // 递增迭代器并断言其大小为 33
  ASSERT_EQ((++iterator)->size(), 33);
  // 递增迭代器并断言其大小为 33
  ASSERT_EQ((++iterator)->size(), 33);
  // 递增迭代器并断言其大小为 1（最后一个批次的大小）
  ASSERT_EQ((++iterator)->size(), 1);
  // 断言迭代器已到达末尾
  ASSERT_EQ(++iterator, data_loader->end());
}

TEST(
    DataLoaderTest,
    DoesNotReturnLastBatchWhenSmallerThanBatchSizeWhenDropLastIsTrue) {
  // 创建 DummyDataset 对象
  DummyDataset dataset;
  // 创建数据加载器，批量大小为 33，并设置丢弃最后一个批次
  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(33).drop_last(true));
  // 获取数据加载器的起始迭代器
  auto iterator = data_loader->begin();
  // 断言第一个批次的大小为 33
  ASSERT_EQ(iterator->size(), 33);
  // 递增迭代器并断言其大小为 33
  ASSERT_EQ((++iterator)->size(), 33);
  // 递增迭代器并断言其大小为 33
  ASSERT_EQ((++iterator)->size(), 33);
  // 断言迭代器已到达末尾
  ASSERT_EQ(++iterator, data_loader->end());
}

TEST(DataLoaderTest, RespectsTimeout) {
  // 定义用于线程同步的结构体
  struct Baton {
    std::condition_variable cv;
    std::mutex mutex;
  };

  // 定义数据集结构体 D，继承自 Dataset 类
  struct D : datasets::Dataset<DummyDataset, int> {
    D(std::shared_ptr<Baton> b) : baton(std::move(b)) {}
    // 实现获取数据的方法
    int get(size_t index) override {
      // 加锁互斥量
      std::unique_lock<std::mutex> lock(baton->mutex);
      // 等待一段时间或直到条件变量被唤醒
      baton->cv.wait_for(lock, 1000 * kMillisecond);
      return 0;
    }
    // 实现获取数据集大小的方法
    torch::optional<size_t> size() const override {
      return 100;
    }
    # 创建一个名为 baton 的共享指针，指向 Baton 类的实例
    std::shared_ptr<Baton> baton;
  };

  # 使用 std::make_shared 创建一个名为 baton 的共享指针，指向新创建的 Baton 对象
  auto baton = std::make_shared<Baton>();

  # 使用 torch::data::make_data_loader 创建一个数据加载器 data_loader，
  # 将 baton 封装在 D 对象中传递给数据加载器，配置工作线程数为 1，超时时间为 10 毫秒的 DataLoaderOptions
  auto data_loader = torch::data::make_data_loader(
      D{baton}, DataLoaderOptions().workers(1).timeout(10 * kMillisecond));

  # 获取当前时间作为计时的起点
  auto start = std::chrono::system_clock::now();

  # 断言操作：期望 *data_loader 的第一个元素被访问时抛出 "Timeout" 异常
  ASSERT_THROWS_WITH(*data_loader->begin(), "Timeout");

  # 通知 baton 关联的条件变量 cv 中的一个等待线程
  baton->cv.notify_one();

  # 获取当前时间作为计时的终点
  auto end = std::chrono::system_clock::now();

  # 计算时间间隔，并将其转换为秒
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

  # 断言操作：期望 duration 的秒数少于 1
  ASSERT_LT(duration.count(), 1);
}

// 这是一个未完整的代码段，看起来是一个结构体定义结束的位置

// 参考：stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11
// Barrier 结构体用于同步多个线程的执行，等待所有线程完成后再继续执行
struct Barrier {
  // 构造函数，初始化计数器，指定需要等待的线程数量
  explicit Barrier(size_t target) : counter_(target) {}

  // 等待函数，使当前线程等待直到所有线程都完成工作
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    // 减少计数器，当计数器为零时，通知所有等待的线程
    if (--counter_ == 0) {
      cv_.notify_all();
    } else {
      // 使用条件变量等待，直到计数器为零
      cv_.wait(lock, [this] { return this->counter_ == 0; });
    }
  }

  // 计数器，用于跟踪还未完成工作的线程数
  size_t counter_;

  // 条件变量，用于线程间的同步和通知
  std::condition_variable cv_;

  // 互斥量，用于保护共享数据结构
  std::mutex mutex_;
};

// OrderingTest 的目的是验证 dataloader 中的 enforce_ordering 选项是否正确工作。
// 这个选项存在的原因是，当 dataloader 启用多个工作线程并且未设置此标志时，
// 工作线程完成其各自的批处理并将其推送回dataloader的主线程时，加载顺序是不确定的。
// imagine... 以下是详细的测试说明，解释了 enforce_ordering 的作用和测试过程。
// 由于这段代码很长，注释也相应地详细解释了其背景和工作方式。
namespace ordering_test {
// 定义命名空间 ordering_test，用于测试数据加载器的顺序性

namespace {
// 匿名命名空间，用于定义内部常量和静态变量

const size_t kNumberOfWorkers = 10;
// 定义常量，表示工作线程的数量为 10

const std::vector<size_t> kOrderInWhichWorkersReturnTheirBatch =
    {3, 7, 0, 5, 4, 8, 2, 1, 9, 6};
// 定义一个顺序向量，表示每个工作线程返回其批次的顺序

} // namespace

struct Dataset : datasets::BatchDataset<Dataset, size_t> {
// 定义 Dataset 结构体，继承自 BatchDataset，泛型为 size_t

  Dataset() = default;
  // 默认构造函数

  // 拷贝构造函数，用于在将数据集复制到特定线程时调用
  Dataset(const Dataset& other) {
    static std::atomic<size_t> counter{0};
    thread_id_ = counter.fetch_add(1);
  }

  Dataset(Dataset&& other) noexcept = default;
  Dataset& operator=(const Dataset& other) = delete;
  Dataset& operator=(Dataset&& other) noexcept = delete;

  size_t get_batch(torch::ArrayRef<size_t> indices) override {
    static Barrier barrier(kNumberOfWorkers);
    static auto order_iterator = kOrderInWhichWorkersReturnTheirBatch.begin();
    static std::condition_variable cv;
    static std::mutex mutex;

    // 等待所有线程获取索引批次并到达此处
    barrier.wait();

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return *order_iterator == this->thread_id_; });
    ++order_iterator;
    lock.unlock();
    cv.notify_all();

    return indices.front();
  }

  torch::optional<size_t> size() const override {
    return kNumberOfWorkers;
  }

  size_t thread_id_ = 0;
};

} // namespace ordering_test

TEST(DataLoaderTest, EnforcesOrderingAmongThreadsWhenConfigured) {
  auto data_loader = torch::data::make_data_loader(
      ordering_test::Dataset{},
      torch::data::samplers::SequentialSampler(ordering_test::kNumberOfWorkers),
      DataLoaderOptions()
          .batch_size(1)
          .workers(ordering_test::kNumberOfWorkers)
          .enforce_ordering(true));
  std::vector<size_t> output;
  for (size_t value : *data_loader) {
    output.push_back(value);
  }
  std::vector<size_t> expected(ordering_test::kNumberOfWorkers);
  std::iota(expected.begin(), expected.end(), size_t(0));
  ASSERT_EQ(expected, output);
}

TEST(DataLoaderTest, Reset) {
  DummyDataset dataset;
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value() / 2);
  auto end = data_loader->end();

  auto iterator = data_loader->begin();
  ASSERT_NE(iterator, end);
  ASSERT_NE(++iterator, end);
  ASSERT_EQ(++iterator, end);

  iterator = data_loader->begin();
  ASSERT_NE(iterator, end);
  ASSERT_NE(++iterator, end);
  ASSERT_EQ(++iterator, end);

  iterator = data_loader->begin();
  ASSERT_NE(iterator, end);
  ASSERT_NE(++iterator, end);
  ASSERT_EQ(++iterator, end);
}
TEST(DataLoaderTest, TestExceptionsArePropagatedFromWorkers) {
  // 定义结构体 D，继承自 datasets::Dataset，实现数据集接口
  struct D : datasets::Dataset<DummyDataset, int> {
    // 实现获取数据函数，当索引无效时抛出异常
    int get(size_t index) override {
      throw std::invalid_argument("badness");
    }
    // 实现数据集大小函数，返回固定大小 100
    torch::optional<size_t> size() const override {
      return 100;
    }
  };

  // 创建数据加载器 data_loader，使用 D 数据集，随机采样器，2 个工作线程
  auto data_loader = torch::data::make_data_loader(
      D{}, samplers::RandomSampler(100), DataLoaderOptions().workers(2));
  // 获取数据迭代器
  auto iterator = data_loader->begin();

  // 尝试从迭代器中获取数据
  try {
    (void)*iterator;
  } catch (torch::data::WorkerException& e) {
    // 断言捕获到的异常信息符合预期
    ASSERT_EQ(
        e.what(),
        std::string("Caught exception in DataLoader worker thread. "
                    "Original message: badness"));
    // 断言重新抛出的异常类型为 std::invalid_argument
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(
        std::rethrow_exception(e.original_exception), std::invalid_argument);
  }
}

TEST(DataLoaderTest, StatefulDatasetWithNoWorkers) {
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;

  // 定义结构体 D，继承自 datasets::StatefulDataset，实现带状态的数据集接口
  struct D : datasets::StatefulDataset<D, int, size_t> {
    // 实现批次获取函数，返回递增的计数器值作为数据，直到达到预设的最大例子数
    torch::optional<int> get_batch(size_t) override {
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        return counter++;
      }
      return torch::nullopt;
    }
    // 实现数据集大小函数，返回固定大小 100
    torch::optional<size_t> size() const override {
      return 100;
    }
    // 实现重置数据集状态函数，将计数器重置为 0
    void reset() override {
      counter = 0;
    }
    // 实现数据保存函数，空实现
    void save(torch::serialize::OutputArchive& archive) const override{};
    // 实现数据加载函数，空实现
    void load(torch::serialize::InputArchive& archive) override {}
    int counter = 0; // 计数器初始化为 0
  };

  // 创建数据加载器 data_loader，使用 D 数据集
  auto data_loader = torch::data::make_data_loader(D{});

  // 断言每个周期的迭代次数等于数据集耗尽前的预设例子数
  for (const auto i : c10::irange(10)) {
    const auto number_of_iterations =
        std::distance(data_loader->begin(), data_loader->end());
    ASSERT_EQ(
        number_of_iterations, kNumberOfExamplesAfterWhichTheDatasetExhausts)
        << "epoch " << i;
  }

  // 遍历数据加载器，断言每个数据点的值小于预设的最大例子数
  for (const int i : *data_loader) {
    ASSERT_LT(i, kNumberOfExamplesAfterWhichTheDatasetExhausts);
  }
}

TEST(DataLoaderTest, StatefulDatasetWithManyWorkers) {
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;
  const int kNumberOfWorkers = 4;

  // 定义结构体 D，继承自 datasets::StatefulDataset，实现带状态的数据集接口
  struct D : datasets::StatefulDataset<D, int, size_t> {
    // 实现批次获取函数，使用互斥锁确保计数器安全递增
    torch::optional<int> get_batch(size_t) override {
      std::lock_guard<std::mutex> lock(mutex);
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        return counter++;
      }
      return torch::nullopt;
    }
    // 实现数据集大小函数，返回固定大小 100
    torch::optional<size_t> size() const override {
      return 100;
    }
    // 实现重置数据集状态函数，将计数器重置为 0
    void reset() override {
      counter = 0;
    }
    // 实现数据保存函数，空实现
    void save(torch::serialize::OutputArchive& archive) const override{};
    // 实现数据加载函数，空实现
    void load(torch::serialize::InputArchive& archive) override {}
    int counter = 0; // 计数器初始化为 0
    std::mutex mutex; // 定义互斥锁对象
  };

  // 创建数据加载器 data_loader，使用多个工作线程和自定义的数据集 D
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::make_shared_dataset<D>(),
      DataLoaderOptions().workers(kNumberOfWorkers));

  // 断言每个周期的迭代次数等于数据集耗尽前的预设例子数
  for (const auto i : c10::irange(10)) {
    // 计算数据加载器中数据项的数量，即迭代器的开始到结束之间的距离
    const auto number_of_iterations =
        std::distance(data_loader->begin(), data_loader->end());
    // 使用断言验证迭代的次数是否等于预期的数据集耗尽后的示例数量
    ASSERT_EQ(
        number_of_iterations, kNumberOfExamplesAfterWhichTheDatasetExhausts)
        << "epoch " << i;
  }

  // 遍历数据加载器中的每个整数 i
  for (const int i : *data_loader) {
    // 使用断言验证每个整数 i 是否小于预期的数据集耗尽后的示例数量
    ASSERT_LT(i, kNumberOfExamplesAfterWhichTheDatasetExhausts);
  }
TEST(DataLoaderTest, StatefulDatasetWithMap) {
  // 每个数据集在用完 10 个样本后停止迭代
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;

  // 定义数据集 D，继承自 StatefulDataset，使用 int 作为数据类型，size_t 作为索引类型
  struct D : datasets::StatefulDataset<D, int, size_t> {
    // 获取一个批次的数据，返回一个 torch::optional<int>
    torch::optional<int> get_batch(size_t) override {
      // 如果计数器小于 kNumberOfExamplesAfterWhichTheDatasetExhausts，则返回当前计数器值并递增
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        return counter++;
      }
      // 否则返回空值，表示数据集已用尽
      return torch::nullopt;
    }
    // 返回数据集的大小
    torch::optional<size_t> size() const override {
      return 100;
    }
    // 重置数据集状态
    void reset() override {
      counter = 0;
    }
    // 序列化保存方法（空实现）
    void save(torch::serialize::OutputArchive& archive) const override{};
    // 序列化加载方法（空实现）
    void load(torch::serialize::InputArchive& archive) override {}
    // 计数器，用于追踪数据集中的位置
    int counter = 0;
  };

  // 创建数据加载器，使用 make_data_loader 函数包装数据集 D，并进行一系列映射操作
  auto data_loader = torch::data::make_data_loader(
      // 将数据集 D 映射到字符串类型
      D().map(transforms::BatchLambda<int, std::string>(
                  [](int x) { return std::to_string(x); }))
          // 再将字符串类型映射为 Torch 张量类型
          .map(transforms::BatchLambda<std::string, torch::Tensor>(
              [](const std::string& x) {
                return torch::tensor(static_cast<int64_t>(std::stoi(x)));
              })),
      // 使用默认的 DataLoaderOptions 创建数据加载器
      DataLoaderOptions{});

  // 断言数据加载器的迭代次数为 kNumberOfExamplesAfterWhichTheDatasetExhausts，重复 10 次
  for (const auto i : c10::irange(10)) {
    const auto number_of_iterations =
        std::distance(data_loader->begin(), data_loader->end());
    ASSERT_EQ(
        number_of_iterations, kNumberOfExamplesAfterWhichTheDatasetExhausts)
        << "epoch " << i;
  }

  // 遍历数据加载器的每个张量 t，断言其值小于 kNumberOfExamplesAfterWhichTheDatasetExhausts
  for (const torch::Tensor& t : *data_loader) {
    ASSERT_LT(t.item<int64_t>(), kNumberOfExamplesAfterWhichTheDatasetExhausts);
  }
}

TEST(DataLoaderTest, StatefulDatasetWithCollate) {
  // 每个数据集在用完 10 个样本后停止迭代
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;

  // 定义数据集 D，继承自 StatefulDataset
  struct D : datasets::StatefulDataset<D> {
    // 获取一个批次的数据，返回一个 torch::optional<std::vector<Example<>>>
    torch::optional<std::vector<Example<>>> get_batch(
        size_t batch_size) override {
      // 如果计数器小于 kNumberOfExamplesAfterWhichTheDatasetExhausts
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        // 更新计数器并生成一个包含 Example<> 结构的批次数据
        counter += batch_size;
        std::vector<Example<>> batch(
            /*count=*/batch_size,
            Example<>{
                torch::ones(batch_size + 1), torch::zeros(batch_size - 1)});
        return batch;
      }
      // 否则返回空值，表示数据集已用尽
      return torch::nullopt;
    }
    // 返回数据集的大小
    torch::optional<size_t> size() const override {
      return 100;
    }
    // 重置数据集状态
    void reset() override {
      counter = 0;
    }
    // 序列化保存方法（空实现）
    void save(torch::serialize::OutputArchive& archive) const override{};
    // 序列化加载方法（空实现）
    void load(torch::serialize::InputArchive& archive) override {}
    // 计数器，用于追踪数据集中的位置
    int counter = 0;
  };
    // 定义一个整型变量 counter，初始化为 0
    int counter = 0;
  };

  // 创建一个类型为 D 的对象 d，并且对其应用 transforms::Stack<Example<>>() 变换
  auto d = D().map(transforms::Stack<Example<>>());

  // 定义常量 kBatchSize 为 5，表示批处理的大小
  const size_t kBatchSize = 5;

  // 注意，数据集的 get_batch() 方法返回一个 vector<Example>，但是 Stack 操作将张量堆叠成一个张量
  // 执行 get_batch() 方法，返回一个包含 Example<> 类型的可选值 batch
  torch::optional<Example<>> batch = d.get_batch(kBatchSize);
  
  // 断言 batch 有值
  ASSERT_TRUE(batch.has_value());
  
  // 断言 batch 中 data 的第一维度大小为 kBatchSize
  ASSERT_EQ(batch->data.size(0), kBatchSize);
  
  // 断言 batch 中 data 的第二维度大小为 kBatchSize + 1
  ASSERT_EQ(batch->data.size(1), kBatchSize + 1);
  
  // 断言 batch 中 target 的第一维度大小为 kBatchSize
  ASSERT_EQ(batch->target.size(0), kBatchSize);
  
  // 断言 batch 中 target 的第二维度大小为 kBatchSize - 1
  ASSERT_EQ(batch->target.size(1), kBatchSize - 1);

  // 断言 batch 中 data 的第一个元素近似等于由 torch::ones(kBatchSize + 1) 创建的张量
  ASSERT_TRUE(batch->data[0].allclose(torch::ones(kBatchSize + 1)));
  
  // 断言 batch 中 target 的第一个元素近似等于由 torch::zeros(kBatchSize - 1) 创建的张量
  ASSERT_TRUE(batch->target[0].allclose(torch::zeros(kBatchSize - 1)));
// 这个测试用例测试了通过数据块数据集进行迭代的核心功能。
// 它包含了不同参数组合的测试用例，例如不同的预取数量、批处理大小和数据加载器工作线程数量。
// 当顺序确定时，它验证了返回批次的大小和内容。

// 在测试中，定义不同的预取数量数组
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
const size_t prefetch_counts[] = {1, 2, 3, 4};

// 在测试中，定义不同的批处理大小数组
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
const size_t batch_sizes[] = {5, 7};

// 在测试中，定义不同的数据加载器工作线程数量数组，包括有和无工作线程两种情况
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
const size_t dataloader_worker_counts[] = {0, 2};

// 定义总的示例数量
const size_t total_example_count = 35;

// 创建一个 DummyChunkDataReader 实例用于测试
DummyChunkDataReader data_reader;

// 创建一个顺序采样器实例，起始索引为0
samplers::SequentialSampler sampler(0);

// 定义跨越多个周期边界的测试功能
const int epoch_count = 2;

// 对每个预取数量进行迭代测试
for (auto prefetch_count : prefetch_counts) {
    for (auto batch_size : batch_sizes) {
      // 遍历批量大小列表
      for (auto dataloader_worker_count : dataloader_worker_counts) {
        // 遍历数据加载器工作线程数量列表

        datasets::SharedBatchDataset<datasets::ChunkDataset<
            DummyChunkDataReader,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>
            dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
                DummyChunkDataReader,
                samplers::SequentialSampler,
                samplers::SequentialSampler>>(
                data_reader,
                sampler,
                sampler,
                datasets::ChunkDatasetOptions(prefetch_count, batch_size));
        // 创建共享批量数据集对象，其中包含一个块数据集，使用顺序采样器

        auto data_loader = torch::data::make_data_loader(
            dataset,
            DataLoaderOptions(batch_size).workers(dataloader_worker_count));
        // 创建数据加载器对象，用于加载上面创建的数据集，设置并行工作线程数量

        for (const auto epoch_index : c10::irange(epoch_count)) {
          (void)epoch_index; // 抑制未使用变量警告
          std::vector<bool> result(total_example_count, false);
          // 创建一个布尔向量用于存储每个样本是否被正确加载
          int iteration_count = 0;
          // 迭代计数器，用于记录迭代次数
          for (auto iterator = data_loader->begin();
               iterator != data_loader->end();
               ++iterator, ++iteration_count) {
            // 迭代数据加载器中的每个批次

            DummyChunkDataReader::BatchType& batch = *iterator;
            // 获取当前批次数据

            ASSERT_EQ(batch.size(), batch_size);
            // 断言当前批次的大小与预期的批次大小相等

            // 当预取计数为1且无工作线程时，批次顺序是确定的，因此可以验证每个批次中的元素
            if (prefetch_count == 1 && dataloader_worker_count == 0) {
              for (const auto j : c10::irange(batch_size)) {
                ASSERT_EQ(batch[j], iteration_count * batch_size + j);
              }
            }

            for (const auto j : c10::irange(batch_size)) {
              result[batch[j]] = true;
              // 标记结果向量中对应的样本已加载
            }
          }

          for (auto data : result) {
            ASSERT_EQ(data, true);
            // 断言所有样本都已加载
          }
        }
      }
    }
TEST(DataLoaderTest, ChunkDataSetGetBatchWithUnevenBatchSize) {
  // 定义一个继承自 ChunkDataReader<int> 的结构体 D，用于处理整数数据的分块读取
  struct D : public datasets::ChunkDataReader<int> {
   public:
    // 定义批次数据类型为 ChunkType
    using BatchType = datasets::ChunkDataReader<int>::ChunkType;

    // 重写 read_chunk 方法，返回包含10个整数0的批次数据
    BatchType read_chunk(size_t chunk_index) override {
      BatchType batch_data(10, 0);
      return batch_data;
    }

    // 重写 chunk_count 方法，返回数据集的分块数量，这里返回2
    size_t chunk_count() override {
      return 2;
    };
  };

  const size_t prefetch_count = 1;
  const size_t batch_size = 5;
  // 创建一个 DummyChunkDataReader 类型的对象 data_reader
  D data_reader;
  // 创建一个顺序采样器对象 sampler，初始索引为0
  samplers::SequentialSampler sampler(0);

  // 创建 SharedBatchDataset 类型的数据集对象 dataset，用于存储 ChunkDataset 数据集
  datasets::SharedBatchDataset<datasets::ChunkDataset<
      D,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          D,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(prefetch_count, batch_size));

  // 创建一个数据加载器 data_loader，加载 dataset 数据集，并指定批次大小为 batch_size
  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(batch_size).workers(0));

  // 遍历 data_loader 的迭代器，确保每个批次的大小为0
  for (auto iterator = data_loader->begin(); iterator != data_loader->end();
       ++iterator) {
    ASSERT_EQ(iterator->size(), 0); // 断言每个批次的大小为0
  }
}
    // 重置数据读取器的状态，覆盖其继承的虚函数
    void reset() override{};
  };

  // 禁止Lint检查：避免使用C风格数组和C++核心准则：避免使用C风格数组
  const size_t batch_sizes[] = {17, 30};
  // 实例化数据读取器对象
  D data_reader;
  // 创建顺序采样器对象，起始索引为0
  samplers::SequentialSampler sampler(0);

  // 遍历批量大小数组
  for (auto batch_size : batch_sizes) {
    // 创建共享的批量数据集对象，使用分块数据集包装数据读取器、顺序采样器、顺序采样器
    datasets::SharedBatchDataset<datasets::ChunkDataset<
        D,
        samplers::SequentialSampler,
        samplers::SequentialSampler>>
        dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
            D,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>(
            data_reader,
            sampler,
            sampler,
            datasets::ChunkDatasetOptions(1, batch_size));

    // 创建数据加载器，加载上述数据集，设置批量大小和工作线程数为0
    auto data_loader = torch::data::make_data_loader(
        dataset, DataLoaderOptions(batch_size).workers(0));

    // 遍历数据加载器中的每个批次
    for (auto iterator = data_loader->begin(); iterator != data_loader->end();
         ++iterator) {
      // 获取当前批次的数据
      DummyChunkDataReader::BatchType batch = *iterator;
      // 获取批次的实际大小
      auto batch_size = batch.size();
      // 如果批次大小为17
      if (batch_size == 17) {
        // 断言批次大小为17或3
        ASSERT_TRUE(batch.size() == 17 || batch.size() == 3);
      }
      // 如果批次大小为30
      if (batch_size == 30) {
        // 断言批次大小为20
        ASSERT_TRUE(batch.size() == 20);
      }
    }
  }
}

TEST(DataLoaderTest, CanAccessChunkSamplerWithChunkDataSet) {
  // 定义预取数量和批量大小
  const size_t prefetch_count = 2;
  const size_t batch_size = 5;

  // 创建虚拟的数据读取器对象
  DummyChunkDataReader data_reader;
  // 创建顺序采样器对象
  samplers::SequentialSampler sampler(0);
  // 创建共享批量数据集对象，使用ChunkDataset作为底层数据集类型
  datasets::SharedBatchDataset<datasets::ChunkDataset<
      DummyChunkDataReader,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          DummyChunkDataReader,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(prefetch_count, batch_size));

  // 获取ChunkDataset中的chunk采样器对象
  samplers::SequentialSampler& chunk_sampler = dataset->chunk_sampler();

  // 创建数据加载器，对数据集进行批处理操作
  auto data_loader = torch::data::make_data_loader(
      dataset.map(transforms::BatchLambda<
                  DummyChunkDataReader::BatchType,
                  DummyChunkDataReader::DataType>(
          [](DummyChunkDataReader::BatchType batch) {
            // 对批次数据进行求和操作
            return std::accumulate(batch.begin(), batch.end(), 0);
          })),
      DataLoaderOptions(batch_size).workers(0));

  // 断言：在开始迭代之前，chunk采样器的索引应该为0
  ASSERT_EQ(chunk_sampler.index(), 0);

  size_t sum = 0;
  // 遍历数据加载器，计算所有数据的总和
  for (auto iterator = data_loader->begin(); iterator != data_loader->end();
       ++iterator) {
    sum += *iterator;
  }
  // 断言：数据总和应为595，即sum([0, 35))
  ASSERT_EQ(sum, 595);
  // 断言：chunk采样器的索引应该为3，因为有3个数据块
  ASSERT_EQ(chunk_sampler.index(), 3);
}

TEST(DataLoaderTest, ChunkDatasetDoesNotHang) {
  // 定义预取数量、批量大小和缓存大小
  const size_t prefetch_count = 2;
  const size_t batch_size = 5;
  const size_t cache_size = 10; // 缓存大小用于控制预加载器等待时间

  // 创建虚拟的数据读取器对象
  DummyChunkDataReader data_reader;
  // 创建顺序采样器对象
  samplers::SequentialSampler sampler(0);
  // 创建共享批量数据集对象，使用ChunkDataset作为底层数据集类型
  datasets::SharedBatchDataset<datasets::ChunkDataset<
      DummyChunkDataReader,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          DummyChunkDataReader,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(
              prefetch_count, batch_size, cache_size));

  // 创建数据加载器，对数据集进行批处理操作
  auto data_loader = torch::data::make_data_loader(
      dataset.map(transforms::BatchLambda<
                  DummyChunkDataReader::BatchType,
                  DummyChunkDataReader::DataType>(
          [](DummyChunkDataReader::BatchType batch) {
            // 对批次数据进行求和操作
            return std::accumulate(batch.begin(), batch.end(), 0);
          })),
      DataLoaderOptions(batch_size).workers(0));

  // 创建数据加载器迭代器，但不进行迭代操作
  auto iterator = data_loader->begin();
}

// 测试ChunkDataset的保存功能
// 注意 [将ChunkDataset保存/加载为ChunkSampler]:
// 测试函数：ChunkDatasetSave
TEST(DataLoaderTest, ChunkDatasetSave) {
  // 定义常量：数据集包含的分块数为6，每个分块的大小为10
  const size_t chunk_count_ = 6;
  const size_t chunk_size = 10;

  // 定义一个用于测试的虚拟数据读取器 DummyTestChunkDataReader
  struct DummyTestChunkDataReader : datasets::ChunkDataReader<int> {
   public:
    using BatchType = datasets::ChunkDataReader<int>::ChunkType;

    // 实现读取分块数据的方法，这里返回一个 batch_data_
    BatchType read_chunk(size_t chunk_index) override {
      return batch_data_;
    }

    // 返回分块的数量
    size_t chunk_count() override {
      return chunk_count_;
    };

    // 重置数据读取器的方法
    void reset() override{};
    BatchType batch_data_ = BatchType(chunk_size, 0);  // 初始化 batch_data_
  };

  // 定义常量：预取的分块数量为1，数据加载器的批量大小为 chunk_size，工作线程数为0
  const size_t prefetch_count = 1;
  const size_t batch_size = chunk_size;
  const size_t dataloader_worker_count = 0;

  // 创建一个顺序采样器 SequentialSampler，起始索引为0
  samplers::SequentialSampler sampler(0);

  // 定义数据集的周期数为2
  const int epoch_count = 2;

  // 创建 DummyTestChunkDataReader 的实例 data_reader
  DummyTestChunkDataReader data_reader;

  // 定义保存间隔数组 save_intervals
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  const size_t save_intervals[] = {1, 2};

  // 引入 ChunkDatasetOptions 类型的命名空间 datasets
  using datasets::ChunkDatasetOptions;

  // 对每个保存间隔进行循环处理
  for (auto save_interval : save_intervals) {
    // 创建一个临时文件
    auto tempfile = c10::make_tempfile();

    // 创建 SharedBatchDataset 类型的数据集 dataset，包含 ChunkDataset
    datasets::SharedBatchDataset<datasets::ChunkDataset<
        DummyTestChunkDataReader,
        samplers::SequentialSampler,
        samplers::SequentialSampler>>
        dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
            DummyTestChunkDataReader,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>(
            data_reader,
            sampler,
            sampler,
            ChunkDatasetOptions(
                prefetch_count, batch_size, chunk_size /*cache size*/));

    // 创建数据加载器 data_loader
    auto data_loader = torch::data::make_data_loader(
        dataset,
        DataLoaderOptions(batch_size).workers(dataloader_worker_count));

  }
}

// 测试函数：ChunkDatasetLoad
TEST(DataLoaderTest, ChunkDatasetLoad) {
  // 创建一个临时文件
  auto tempfile = c10::make_tempfile();

  // 定义常量：预取的分块数量为1，数据加载器的批量大小为10，工作线程数为0
  const size_t prefetch_count = 1;
  const size_t batch_size = 10;
  const size_t dataloader_worker_count = 0;

  // 创建 DummyChunkDataReader 的实例 data_reader
  DummyChunkDataReader data_reader;
  
  // 创建顺序采样器 SequentialSampler，起始索引为0
  samplers::SequentialSampler sampler(0);

  // 定义跳过的分块索引为2
  const size_t skipped_chunk = 2;

  // 配置采样器使其跳过2个分块
  {
    sampler.reset(data_reader.chunk_count());
    sampler.next(skipped_chunk);

    // 查看注释：见注释 [save/load ChunkDataset as ChunkSampler]
    torch::save(sampler, tempfile.name);


    // 将sampler对象保存到临时文件中
    // 用于后续加载和恢复sampler状态
    torch::save(sampler, tempfile.name);
  }

  // 测试功能跨越epoch边界。第一个epoch应受检查点影响，但第二个应正常启动。
  const int epoch_count = 2;

  datasets::SharedBatchDataset<datasets::ChunkDataset<
      DummyChunkDataReader,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          DummyChunkDataReader,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(
              prefetch_count, batch_size, 20 /*cache size*/));

  torch::load(*dataset, tempfile.name);

  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(batch_size).workers(dataloader_worker_count));

  for (const auto epoch_index : c10::irange(epoch_count)) {
    int iteration_count = 0;

    // For the first epoch, the returned batch should be returned from the
    // third chunk, because the check point skipped the first two chunks. But
    // for the next epoch, it should start from the first batch.
    int initial_value = epoch_index == 0 ? 15 : 0;

    for (auto iterator = data_loader->begin(); iterator != data_loader->end();
         ++iterator, ++iteration_count) {
      DummyChunkDataReader::BatchType batch = *iterator;

      std::vector<int> expected_result;
      size_t expected_size = (epoch_index > 0 && iteration_count == 3) ? 5 : 10;
      expected_result.resize(expected_size);
      std::iota(expected_result.begin(), expected_result.end(), initial_value);

      ASSERT_EQ(batch.size(), expected_result.size());
      ASSERT_TRUE(
          std::equal(batch.begin(), batch.end(), expected_result.begin()));

      initial_value += batch_size;
    }
  }

  samplers::SequentialSampler new_sampler(0);

  // See Note [save/load ChunkDataset as ChunkSampler]
  torch::load(new_sampler, tempfile.name);

  ASSERT_EQ(new_sampler.index(), skipped_chunk);


  // 创建新的SequentialSampler对象，并从临时文件加载其状态
  samplers::SequentialSampler new_sampler(0);
  
  // 根据注释 "See Note [save/load ChunkDataset as ChunkSampler]"，加载并恢复sampler状态
  // 确保加载后的sampler索引与预期的skipped_chunk相等
  torch::load(new_sampler, tempfile.name);
  
  // 使用断言验证加载后的sampler索引是否与预期的skipped_chunk相等
  ASSERT_EQ(new_sampler.index(), skipped_chunk);
  // 定义一个测试用例，测试数据加载器中跨块数据集的交叉洗牌功能
  TEST(DataLoaderTest, ChunkDatasetCrossChunkShuffle) {
    // 设置每个块的大小和批量大小
    const size_t chunk_size = 5;
    const size_t batch_size = 5;

    // 定义一个派生自`samplers::Sampler<>`的内部类S
    class S : public samplers::Sampler<> {
     public:
      explicit S(size_t size) : size_(size), index_(0){};

      // 重置采样器状态，如果有新大小，则重新设置大小
      void reset(torch::optional<size_t> new_size = torch::nullopt) override {
        if (new_size.has_value()) {
          size_ = *new_size;
        }
        // 初始化索引数组大小为当前大小
        indices_.resize(size_);
        size_t index = 0;

        // 每5个索引重复采样一次
        for (const auto i : c10::irange(batch_size)) {
          for (size_t j = 0; j < size_ / batch_size; ++j) {
            indices_[index++] = i + batch_size * j;
          }
        }
        index_ = 0;
      }

      // 返回下一个批次的索引
      torch::optional<std::vector<size_t>> next(size_t batch_size) override {
        const auto remaining_indices = size_ - index_;
        if (remaining_indices == 0) {
          return torch::nullopt;
        }
        auto return_size = std::min(batch_size, remaining_indices);
        // 返回索引数组中从当前索引到返回大小的批次
        std::vector<size_t> index_batch(
            indices_.begin() + index_, indices_.begin() + index_ + return_size);
        index_ += return_size;

        return index_batch;
      }

      // 保存采样器状态（无操作）
      void save(torch::serialize::OutputArchive& archive) const override {}
      // 加载采样器状态（无操作）
      void load(torch::serialize::InputArchive& archive) override {}

     private:
      size_t size_;
      std::vector<size_t> indices_;
      size_t index_{0};
    };

    // 定义派生自`datasets::ChunkDataReader<int>`的结构体D
    struct D : public datasets::ChunkDataReader<int> {
     public:
      using BatchType = datasets::ChunkDataReader<int>::ChunkType;
      D(size_t chunk_count) : chunk_count_(chunk_count) {}

      // 读取给定块索引的块数据
      BatchType read_chunk(size_t chunk_index) override {
        BatchType batch_data(chunk_size, chunk_index);
        return batch_data;
      }

      // 返回块的数量
      size_t chunk_count() override {
        return chunk_count_;
      };

      // 重置数据集（无操作）
      void reset() override{};
      size_t chunk_count_;
    };

    // 预取计数和缓存大小
    const size_t prefetch_count = 1;
    const size_t cache_size = 10;
    // 指定跨块洗牌次数数组
    const size_t cross_chunk_shuffle_counts[] = {2, 3};
    // 指定块计数数组
    const size_t chunk_counts[] = {3, 4, 5};

    // 顺序采样器对象和示例采样器对象
    samplers::SequentialSampler chunk_sampler(0);
    S example_sampler(0);

    // 对每个块计数进行迭代
    for (auto chunk_count : chunk_counts) {
    // 对每个 cross_chunk_shuffle_counts 中的元素执行循环
    for (auto cross_chunk_shuffle_count : cross_chunk_shuffle_counts) {
      // 使用 data_reader 初始化数据阅读器对象 D，参数为 chunk_count
      D data_reader(chunk_count);

      // 创建共享批次数据集 dataset，类型为 ChunkDataset，包含顺序采样器和 S
      datasets::SharedBatchDataset<
          datasets::ChunkDataset<D, samplers::SequentialSampler, S>>
          dataset = datasets::make_shared_dataset<
              datasets::ChunkDataset<D, samplers::SequentialSampler, S>>(
              data_reader,
              chunk_sampler,
              example_sampler,
              datasets::ChunkDatasetOptions(
                  prefetch_count,
                  batch_size,
                  cache_size,
                  cross_chunk_shuffle_count));

      // 创建数据加载器 data_loader，使用 dataset，设置批次大小和工作线程数为 0
      auto data_loader = torch::data::make_data_loader(
          dataset, DataLoaderOptions(batch_size).workers(0));

      // 存储数据加载器中的结果
      std::vector<int> result;
      // 迭代数据加载器中的每个批次
      for (auto iterator = data_loader->begin(); iterator != data_loader->end();
           ++iterator) {
        auto batch_result = *iterator;
        // 将批次结果扁平化并追加到 result 中
        std::copy(
            batch_result.begin(),
            batch_result.end(),
            std::back_inserter(result));
      }

      // 存储期望的结果
      std::vector<int> expected_result;
      {
        // 构造期望的结果
        for (const auto i : c10::irange(
                 (chunk_count + cross_chunk_shuffle_count - 1) /
                 cross_chunk_shuffle_count)) {
          // 对于每个 i，遍历 chunk_size 范围内的 j
          for (const auto j : c10::irange(chunk_size)) {
            (void)j; // 抑制未使用变量警告
            // 对于每个 k，遍历 cross_chunk_shuffle_count 范围内的 k
            for (const auto k : c10::irange(cross_chunk_shuffle_count)) {
              // 如果索引在 chunk_count 范围内，将结果添加到期望结果中
              if (i * cross_chunk_shuffle_count + k < chunk_count) {
                expected_result.push_back(i * cross_chunk_shuffle_count + k);
              }
            }
          }
        }
      }

      // 断言结果向量和期望结果向量的大小相等
      ASSERT_EQ(result.size(), expected_result.size());
      // 断言结果向量和期望结果向量相等
      ASSERT_TRUE(
          std::equal(result.begin(), result.end(), expected_result.begin()));
    }
  }
}

TEST(DataLoaderTest, CustomPreprocessPolicy) {
  const size_t chunk_size = 5;  // 定义每个数据块的大小为5
  const size_t batch_size = 10;  // 定义每个批次的大小为10

  struct D : public datasets::ChunkDataReader<int> {  // 定义结构体 D，继承自 ChunkDataReader<int>
   public:
    using BatchType = datasets::ChunkDataReader<int>::ChunkType;  // 定义批次数据类型为 ChunkType
    D(size_t chunk_count) : chunk_count_(chunk_count) {}  // 构造函数，设置数据块的数量

    BatchType read_chunk(size_t chunk_index) override {  // 重写父类方法，读取指定索引的数据块
      BatchType batch_data(chunk_size);  // 创建指定大小的批次数据对象
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,clang-analyzer-security.insecureAPI.rand)
      auto rand_gen = []() { return std::rand() % 100; };  // 定义随机数生成器 lambda 函数
      std::generate(batch_data.begin(), batch_data.end(), rand_gen);  // 使用随机数填充批次数据
      return batch_data;  // 返回生成的批次数据
    }

    size_t chunk_count() override {  // 重写父类方法，返回数据块的数量
      return chunk_count_;  // 返回存储的数据块数量
    };

    void reset() override{};  // 重置方法，暂未实现
    size_t chunk_count_;  // 存储数据块数量的成员变量
  };

  // custom preprocessing policy - sort the data ascendingly
  auto sorting_policy = [](std::vector<int>& raw_batch_data) {  // 定义自定义预处理策略 lambda 函数，对批次数据进行升序排序
    std::sort(raw_batch_data.begin(), raw_batch_data.end());  // 对批次数据进行升序排序
  };
  std::function<void(std::vector<int>&)> policy_function = sorting_policy;  // 定义函数对象，存储排序策略函数

  const size_t prefetch_count = 1;  // 预取数据的数量为1
  const size_t cache_size = 10;  // 缓存大小为10
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  const size_t cross_chunk_shuffle_counts[] = {1, 2};  // 跨数据块洗牌计数数组
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  const size_t chunk_counts[] = {3, 4};  // 数据块数量数组

  samplers::SequentialSampler chunk_sampler(0);  // 创建顺序采样器对象，起始索引为0

  for (auto chunk_count : chunk_counts) {  // 遍历数据块数量数组
    for (auto cross_chunk_shuffle_count : cross_chunk_shuffle_counts) {  // 遍历跨数据块洗牌计数数组
      D data_reader(chunk_count);  // 创建数据阅读器对象，设置数据块数量

      datasets::SharedBatchDataset<datasets::ChunkDataset<
          D,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>
          dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
              D,
              samplers::SequentialSampler,
              samplers::SequentialSampler>>(
              data_reader,
              chunk_sampler,
              chunk_sampler,
              datasets::ChunkDatasetOptions(
                  prefetch_count,
                  batch_size,
                  cache_size,
                  cross_chunk_shuffle_count),
              policy_function);  // 创建共享批次数据集对象，使用数据阅读器、采样器和自定义策略函数

      auto data_loader = torch::data::make_data_loader(
          dataset, DataLoaderOptions(batch_size).workers(0));  // 创建数据加载器对象，设置批次大小和工作线程数为0

      std::vector<int> result;  // 存储结果的整数向量
      for (auto iterator = data_loader->begin(); iterator != data_loader->end();
           ++iterator) {  // 遍历数据加载器中的迭代器
        auto batch_result = *iterator;  // 获取当前批次数据
        if (batch_result.size() > chunk_size * cross_chunk_shuffle_count) {  // 如果批次数据大小大于预期的数据块大小乘以跨数据块洗牌计数
          for (unsigned i = 0; i < batch_result.size(); i += chunk_size) {  // 遍历批次数据
            ASSERT_TRUE(std::is_sorted(
                batch_result.begin() + i,
                batch_result.begin() + i + chunk_size));  // 断言当前数据段为升序排序
          }
        } else {
          ASSERT_TRUE(std::is_sorted(batch_result.begin(), batch_result.end()));  // 断言整个批次数据为升序排序
        }
      }
    }
  }
}
```