# `.\pytorch\caffe2\serialize\inline_container.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <cerrno>
// C++标准库头文件，定义标准错误码

#include <cstdio>
// C标准I/O库头文件，定义输入输出函数

#include <cstring>
// C标准字符串头文件，定义字符串处理函数

#include <fstream>
// C++文件流头文件，定义文件流类

#include <istream>
// C++输入流头文件，定义输入流类

#include <mutex>
// C++标准库头文件，定义互斥量和相关操作

#include <ostream>
// C++输出流头文件，定义输出流类

#include <unordered_set>
// C++标准库头文件，定义无序集合类模板

#include <c10/core/Allocator.h>
// 包含c10核心模块的分配器头文件

#include <c10/core/Backend.h>
// 包含c10核心模块的后端头文件

#include "caffe2/serialize/istream_adapter.h"
// 包含Caffe2序列化模块的输入流适配器头文件

#include "caffe2/serialize/read_adapter_interface.h"
// 包含Caffe2序列化模块的读取适配器接口头文件

#include "caffe2/serialize/versions.h"
// 包含Caffe2序列化模块的版本头文件

extern "C" {
typedef struct mz_zip_archive mz_zip_archive;
}
// 声明C语言风格的外部链接变量，定义mz_zip_archive结构体类型

// PyTorch容器是一种特殊的ZIP存档，具有以下布局
// archive_name.zip包含：
//    archive_name/
//        version # 一个包含单个十进制数字的文件，以ASCII编码，
//                # 用于确定存档格式的版本
//        model.json # 整体模型描述，这是从torch.proto的ModelDef输出的json
//                   # 格式
//        # 以下名称仅为约定，model.json将通过完整名称引用这些文件
//        tensors/
//          0 # 张量数据的扁平存储，关于形状的元数据等在model.json中
//          1
//          ...
//        # 仅对附有方法的模块存在代码条目
//        code/
//          archive_name.py # 序列化的torch脚本代码（Python语法，使用PythonPrint）
//          archive_name_my_submodule.py # 子模块有单独的文件
//
// PyTorchStreamWriter还确保这些文件具有额外的有用属性
// 1. 所有文件均以未压缩形式存储。
// 2. 存档中的所有文件都对齐到64字节边界，这样可以对整个文件进行内存映射，
//    并得到指向张量数据的对齐指针。
// 3. 我们普遍使用ZIP64格式以确保一致性。

// PyTorchStreamReader还提供额外的属性：
// 1. 它可以读取使用常见压缩工具创建的ZIP文件。这意味着即使我们的写入器不压缩文件，
//    读取器仍然可以读取已压缩的文件。
// 2. 它提供getRecordOffset函数，返回文件数据在原始文件中的偏移量。如果文件是用
//    PyTorchStreamWriter写入的，则保证偏移量是64字节对齐的。

// PyTorchReader/Writer负责检查存档格式的版本号，并确保所有文件都写入到
// 一个名为archive_name的目录中，以便解压时干净。

// 在开发此格式时，我们特别关注以下使用案例：
//
// -- 读取 --
// 1) 具有完全随机访问的读取
//   a) 使用诸如fread()之类的文件API进行读取
//   b) 内存映射文件并在映射区域中跳转
// 2) 具有一遍顺序访问的读取
//      -> 读取器将需要构建解析结构的数据结构，随着读取的进行而逐步完成

// -- 写入 --
// 1) 具有完全随机访问的写入
// 2) 具有一遍顺序访问的写入
//      -> 我们必须小心，不要要求更新已经
// Define a constant string representing the path to the serialization ID record within the ZIP archive.
static constexpr const char* kSerializationIdRecordName = ".data/serialization_id";

// Forward declaration of a struct MzZipReaderIterWrapper, which is likely defined elsewhere.

// Begin of namespace caffe2
namespace caffe2 {
// Begin of namespace serialize
namespace serialize {

// Declaration of a struct MzZipReaderIterWrapper, which is expected to be defined later.

// Declaration of a class ChunkRecordIterator
class TORCH_API ChunkRecordIterator {
 public:
  // Destructor for ChunkRecordIterator
  ~ChunkRecordIterator();

  // Reads up to `chunkSize` bytes into `buf`. Returns the number of bytes read.
  size_t next(void* buf);

  // Returns the size of each record.
  size_t recordSize() const { return recordSize_; }

 private:
  // Constructor for ChunkRecordIterator
  ChunkRecordIterator(
      size_t recordSize,
      size_t chunkSize,
      std::unique_ptr<MzZipReaderIterWrapper> iter);

  // Member variables
  const size_t recordSize_;
  const size_t chunkSize_;
  size_t offset_;
  std::unique_ptr<MzZipReaderIterWrapper> iter_;

  // PyTorchStreamReader is declared as a friend class to access private members.
  friend class PyTorchStreamReader;
};

// Declaration of a class PyTorchStreamReader
class TORCH_API PyTorchStreamReader final {
 public:
  // Constructor taking a file_name as input
  explicit PyTorchStreamReader(const std::string& file_name);

  // Constructor taking a std::istream* as input
  explicit PyTorchStreamReader(std::istream* in);

  // Constructor taking a shared_ptr<ReadAdapterInterface> as input
  explicit PyTorchStreamReader(std::shared_ptr<ReadAdapterInterface> in);

  // Retrieves a record named `name`, returning a tuple containing data pointer and size
  std::tuple<at::DataPtr, size_t> getRecord(const std::string& name);

  // Retrieves a record named `name` with multi-threading support using additional readers
  std::tuple<at::DataPtr, size_t> getRecord(const std::string& name, std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders);

  // Retrieves a record named `name` and writes it directly into `dst`, returning the number of bytes written
  size_t getRecord(const std::string& name, void* dst, size_t n);

  // Retrieves a record named `name` with multi-threading support and writes into `dst`,
  // using additional readers if provided. Returns the number of bytes written.
  // If additionalReaders is empty, uses the default behavior of getRecord(name, dst, n) with the default reader.
  // This is useful for efficiently reading large tensors.
  size_t getRecord(const std::string& name, void* dst, size_t n,
                   std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders);
  
  // End of class PyTorchStreamReader
};

// End of namespace serialize
}
// End of namespace caffe2
}

// End of the entire code block
  // 构造函数：初始化 PyTorchStreamReader 实例
  PyTorchStreamReader(
      const std::string& archive_name,
      std::shared_ptr<ReadAdapterInterface> in,
      std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders);

  // 根据指定名称读取记录数据
  size_t getRecord(
      const std::string& name,
      void* dst,
      size_t n,
      size_t chunk_size,
      void* buf,
      const std::function<void(void*, const void*, size_t)>& memcpy_func = nullptr);

  // 使用多个读取器并发读取记录数据
  // additionalReaders 是额外的客户端，用于在不同偏移量访问底层记录并写入不同的缓冲区块
  // 如果张量的总大小为 10，additionalReader 的大小为 2
  // 默认线程将读取 [0,4)，additionalReader 将读取 [4,8)
  // 默认读取器将读取 [8,10)
  // 默认读取器将写入到缓冲区 [0,4)，additionalReader 将写入到缓冲区 [4,8)
  // additionalReader 还将写入到缓冲区 [8,10)
  // 当 additionalReaders 为空时，默认行为是使用默认读取器调用 getRecord(name)
  // 这种方法可用于读取大张量
  size_t getRecordMultiReaders(const std::string& name,
      std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders,
      void *dst, size_t n);

  // 获取指定名称记录的大小
  size_t getRecordSize(const std::string& name);

  // 获取指定名称记录的偏移量
  size_t getRecordOffset(const std::string& name);

  // 检查是否存在指定名称的记录
  bool hasRecord(const std::string& name);

  // 获取所有记录的名称列表
  std::vector<std::string> getAllRecords();

  // 创建一个分块记录迭代器
  ChunkRecordIterator createChunkReaderIter(
      const std::string& name,
      const size_t recordSize,
      const size_t chunkSize);

  // 析构函数：释放资源
  ~PyTorchStreamReader();

  // 获取版本号
  uint64_t version() const {
    return version_;
  }

  // 获取序列化标识
  const std::string& serializationId() {
    return serialization_id_;
  }

  // 设置是否加载调试符号
  void setShouldLoadDebugSymbol(bool should_load_debug_symbol) {
    load_debug_symbol_ = should_load_debug_symbol;
  }

  // 设置额外读取器的大小阈值
  void setAdditionalReaderSizeThreshold(const size_t& size){
    additional_reader_size_threshold_ = size;
  }

 private:
  // 初始化函数
  void init();

  // 从指定位置读取数据到缓冲区
  size_t read(uint64_t pos, char* buf, size_t n);

  // 检查操作的有效性
  void valid(const char* what, const char* info = "");

  // 获取指定名称记录的 ID
  size_t getRecordID(const std::string& name);

  // 友元函数声明：用于从输入流中读取数据
  friend size_t
  istream_read_func(void* pOpaque, uint64_t file_ofs, void* pBuf, size_t n);

  // libminizip 的归档对象
  std::unique_ptr<mz_zip_archive> ar_;

  // 归档文件名
  std::string archive_name_;

  // 归档文件名（末尾带斜杠）
  std::string archive_name_plus_slash_;

  // 读取适配器接口的共享指针
  std::shared_ptr<ReadAdapterInterface> in_;

  // 版本号
  int64_t version_;

  // 读取器锁
  std::mutex reader_lock_;

  // 是否加载调试符号的标志
  bool load_debug_symbol_ = true;

  // 序列化标识
  std::string serialization_id_;

  // 额外读取器的大小阈值
  size_t additional_reader_size_threshold_;
};

// PyTorchStreamWriter 类的实现，用于序列化数据到存储器中
class TORCH_API PyTorchStreamWriter final {
 public:
  // 构造函数，接受归档名字并初始化
  explicit PyTorchStreamWriter(const std::string& archive_name);
  
  // 构造函数，接受写入函数并初始化
  explicit PyTorchStreamWriter(
      const std::function<size_t(const void*, size_t)> writer_func);

  // 设置最小版本号
  void setMinVersion(const uint64_t version);

  // 写入记录
  void writeRecord(
      const std::string& name,  // 记录的名称
      const void* data,         // 记录的数据
      size_t size,              // 数据大小
      bool compress = false);   // 是否压缩数据

  // 写入文件结束标志
  void writeEndOfFile();

  // 返回所有已经写入记录的名称集合
  const std::unordered_set<std::string>& getAllWrittenRecords();

  // 检查是否已经完成序列化
  bool finalized() const {
    return finalized_;
  }

  // 返回归档名字
  const std::string& archiveName() {
    return archive_name_;
  }

  // 返回序列化 ID
  const std::string& serializationId() {
    return serialization_id_;
  }

  // 析构函数
  ~PyTorchStreamWriter();

 private:
  // 初始化函数，设置文件名等
  void setup(const std::string& file_name);

  // 验证函数，用于检查错误
  void valid(const char* what, const char* info = "");

  // 写入序列化 ID 到文件中
  void writeSerializationId();

  // 当前写入位置
  size_t current_pos_ = 0;

  // 已经写入文件名集合
  std::unordered_set<std::string> files_written_;

  // 压缩归档对象的智能指针
  std::unique_ptr<mz_zip_archive> ar_;

  // 归档名字
  std::string archive_name_;

  // 归档名字加斜杠
  std::string archive_name_plus_slash_;

  // 填充字符串
  std::string padding_;

  // 文件流对象
  std::ofstream file_stream_;

  // 写入函数对象
  std::function<size_t(const void*, size_t)> writer_func_;

  // 组合的未压缩 CRC32 值
  uint64_t combined_uncomp_crc32_ = 0;

  // 序列化 ID
  std::string serialization_id_;

  // 文件格式版本号，将在模型包含具有有效升级器的操作符时更新
  uint64_t version_ = kMinProducedFileFormatVersion;

  // 是否已经完成序列化
  bool finalized_ = false;

  // 是否遇到错误
  bool err_seen_ = false;

  // 友元函数声明，用于向输出流写入数据
  friend size_t ostream_write_func(
      void* pOpaque,
      uint64_t file_ofs,
      const void* pBuf,
      size_t n);
};

// detail 命名空间，包含写入器的常量和函数
namespace detail {
// 写入器特定的常量，字段对齐要求为 64 字节
constexpr uint64_t kFieldAlignment = 64;

// 返回填充记录，以便使数据从 kFieldAlignment 字节边界开始对齐
size_t getPadding(
    size_t cursor,          // 当前游标位置
    size_t filename_size,   // 文件名大小
    size_t size,            // 数据大小
    std::string& padding_buf);  // 填充缓冲区的引用
} // namespace detail

} // namespace serialize
} // namespace caffe2
```