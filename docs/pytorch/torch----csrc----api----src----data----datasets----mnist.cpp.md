# `.\pytorch\torch\csrc\api\src\data\datasets\mnist.cpp`

```
// 包含 MNIST 数据集相关的头文件
#include <torch/data/datasets/mnist.h>

// 包含数据示例和类型定义的头文件
#include <torch/data/example.h>
#include <torch/types.h>

// 包含异常处理的头文件
#include <c10/util/Exception.h>

// 包含标准库头文件
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

// 定义命名空间 torch::data::datasets::
namespace torch {
namespace data {
namespace datasets {
namespace {

// 训练集和测试集的大小常量
constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;

// 图像和目标文件的魔数常量
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;

// 图像的行数和列数常量
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;

// 训练集和测试集的文件名常量
constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

// 检查当前系统是否为小端存储
bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

// 翻转 32 位整数的字节序
constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
      ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

// 从文件流中读取一个 32 位整数
uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t value;
  // 使用流读取数据到 value 变量中，并进行断言检查
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

// 期望从文件流中读取一个特定的 32 位整数，并进行断言检查
uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  // clang-format off
  TORCH_CHECK(value == expected,
      "Expected to read number ", expected, " but found ", value, " instead");
  // clang-format on
  return value;
}

// 将两个路径部分连接成一个完整的路径
std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

// 从文件中读取图像数据并返回为 Tensor 对象
Tensor read_images(const std::string& root, bool train) {
  // 构建图像文件的完整路径
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  // 打开二进制文件流
  std::ifstream images(path, std::ios::binary);
  // 检查文件流是否成功打开
  TORCH_CHECK(images, "Error opening images file at ", path);

  // 确定要读取的图像数量
  const auto count = train ? kTrainSize : kTestSize;

  // 从 http://yann.lecun.com/exdb/mnist/ 获取 MNIST 数据集的魔数和维度信息
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  // 创建一个空的 Tensor 对象来存储图像数据
  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  // 从文件流中读取图像数据，并将其存储到 Tensor 中
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  // 将数据类型转换为 float32，并归一化处理
  return tensor.to(torch::kFloat32).div_(255);
}
// 读取 MNIST 数据集的标签文件内容，并返回对应的张量
Tensor read_targets(const std::string& root, bool train) {
  // 拼接路径，根据 train 参数选择训练集或测试集的标签文件名
  const auto path =
      join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  // 打开二进制输入流，用于读取文件内容
  std::ifstream targets(path, std::ios::binary);
  // 检查文件流是否有效，如果无效则输出错误信息并中止
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  // 根据 train 参数确定数据集大小，选择对应的数量常量
  const auto count = train ? kTrainSize : kTestSize;

  // 预期读取标签文件中的魔数（magic number），验证文件格式
  expect_int32(targets, kTargetMagicNumber);
  // 预期读取标签文件中的数据项数目，验证文件完整性
  expect_int32(targets, count);

  // 创建一个空的字节张量，用于存储读取的标签数据
  auto tensor = torch::empty(count, torch::kByte);
  // 从文件中读取数据到张量的内存中
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  // 将字节类型的张量转换为 int64 类型，并返回
  return tensor.to(torch::kInt64);
}
} // namespace

// MNIST 类的构造函数，根据给定的根目录和模式加载图像和标签数据
MNIST::MNIST(const std::string& root, Mode mode)
    : images_(read_images(root, mode == Mode::kTrain)),  // 加载图像数据集
      targets_(read_targets(root, mode == Mode::kTrain)) {}  // 加载标签数据集

// 获取指定索引处的图像和标签数据，并返回为 Example<> 结构
Example<> MNIST::get(size_t index) {
  return {images_[index], targets_[index]};
}

// 返回图像数据集的大小（数量），作为可选值
optional<size_t> MNIST::size() const {
  return images_.size(0);
}

// 检查数据集是否为训练集，返回布尔值
// NOLINTNEXTLINE(bugprone-exception-escape)
bool MNIST::is_train() const noexcept {
  return images_.size(0) == kTrainSize;
}

// 返回图像数据张量的常量引用
const Tensor& MNIST::images() const {
  return images_;
}

// 返回标签数据张量的常量引用
const Tensor& MNIST::targets() const {
  return targets_;
}

} // namespace datasets
} // namespace data
} // namespace torch
```