# `.\pytorch\test\cpp\jit\test_file_format.cpp`

```
// 引入 Torch 的文件格式处理头文件
#include <torch/csrc/jit/mobile/file_format.h>

// 引入 Google 测试框架的头文件
#include <gtest/gtest.h>

// 引入标准库的字符串流头文件
#include <sstream>

// 测试用例定义在 torch::jit 命名空间下
namespace torch {
namespace jit {

// 测试函数：识别 Flatbuffer 流
TEST(FileFormatTest, IdentifiesFlatbufferStream) {
  // 创建一个数据流，其初始字节看起来像一个 Flatbuffer 流
  std::stringstream data;
  data << "abcd" // 前四个字节无关紧要
       << "PTMF" // Flatbuffer 的魔术字符串
       << "efgh"; // 后续字节无关紧要

  // 应该将数据识别为 Flatbuffer 格式
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);
}

// 测试函数：识别 ZIP 流
TEST(FileFormatTest, IdentifiesZipStream) {
  // 创建一个数据流，其初始字节看起来像一个 ZIP 流
  std::stringstream data;
  data << "PK\x03\x04" // ZIP 的魔术字符串
       << "abcd" // 后续字节无关紧要
       << "efgh";

  // 应该将数据识别为 ZIP 格式
  EXPECT_EQ(getFileFormat(data), FileFormat::ZipFileFormat);
}

// 测试函数：Flatbuffer 优先级
TEST(FileFormatTest, FlatbufferTakesPrecedence) {
  // 因为 Flatbuffer 和 ZIP 的魔术字节位于不同的偏移量，
  // 相同的数据可能被同时识别为两者。演示 Flatbuffer 优先的情况。
  std::stringstream data;
  data << "PK\x03\x04" // ZIP 的魔术字符串
       << "PTMF" // Flatbuffer 的魔术字符串
       << "abcd"; // 后续字节无关紧要

  // 应该将数据识别为 Flatbuffer 格式
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);
}

// 测试函数：处理未知流
TEST(FileFormatTest, HandlesUnknownStream) {
  // 创建一个看起来不像任何已知格式的数据流
  std::stringstream data;
  data << "abcd"
       << "efgh"
       << "ijkl";

  // 应该将数据识别为未知格式
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

// 测试函数：短流被识别为未知
TEST(FileFormatTest, ShortStreamIsUnknown) {
  // 创建一个少于 kFileFormatHeaderSize (8) 字节的数据流
  std::stringstream data;
  data << "ABCD";

  // 应该将数据识别为未知格式
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

// 测试函数：空流被识别为未知
TEST(FileFormatTest, EmptyStreamIsUnknown) {
  // 创建一个空的数据流
  std::stringstream data;

  // 应该将数据识别为未知格式
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

// 测试函数：错误流被识别为未知
TEST(FileFormatTest, BadStreamIsUnknown) {
  // 创建一个带有有效 Flatbuffer 数据的数据流
  std::stringstream data;
  data << "abcd"
       << "PTMF" // Flatbuffer 的魔术字符串
       << "efgh";

  // 演示数据本应被识别为 Flatbuffer 格式
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);

  // 将流标记为错误状态，并演示它处于错误状态
  data.setstate(std::stringstream::badbit);
  // 演示流现在处于错误状态
  EXPECT_FALSE(data.good());

  // 应该将错误状态的数据识别为未知格式
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

} // namespace jit
} // namespace torch
TEST(FileFormatTest, StreamOffsetIsObservedAndRestored) {
  // 创建一个 stringstream 对象用于存储数据流
  std::stringstream data;
  // 添加初始填充
  data << "PADDING";
  // 记录当前数据流的大小作为偏移量
  size_t offset = data.str().size();
  // 添加一个有效的 Flatbuffer 头部
  data << "abcd"
       << "PTMF" // Flatbuffer 的魔术字符串
       << "efgh";
  // 将数据流定位到填充之后的位置
  data.seekg(static_cast<std::stringstream::off_type>(offset), data.beg);
  // 展示数据流当前位置指向 Flatbuffer 数据的开头，而不是填充的位置
  EXPECT_EQ(data.peek(), 'a');

  // 确认数据流中的格式应为 Flatbuffer
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);

  // 确认识别格式后数据流的位置应还原到识别之前的偏移量
  EXPECT_EQ(offset, data.tellg());
}

TEST(FileFormatTest, HandlesMissingFile) {
  // 对于缺失的文件应分类为未知格式
  EXPECT_EQ(
      getFileFormat("NON_EXISTENT_FILE_4965c363-44a7-443c-983a-8895eead0277"),
      FileFormat::UnknownFileFormat);
}

} // namespace jit
} // namespace torch
```