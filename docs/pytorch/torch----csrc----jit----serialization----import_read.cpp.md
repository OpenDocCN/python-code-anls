# `.\pytorch\torch\csrc\jit\serialization\import_read.cpp`

```py
namespace torch::jit {
// 定义函数 readArchiveAndTensors，从流中读取归档和张量数据并解析为 IValue 类型
IValue readArchiveAndTensors(
    const std::string& archive_name,  // 归档文件名
    const std::string& pickle_prefix,  // pickle 文件前缀
    const std::string& tensor_prefix,  // 张量文件路径前缀
    std::optional<TypeResolver> type_resolver,  // 类型解析器（可选）
    std::optional<ObjLoader> obj_loader,  // 对象加载器（可选）
    std::optional<at::Device> device,  // 设备类型（可选）
    caffe2::serialize::PyTorchStreamReader& stream_reader,  // PyTorch 流读取器
    c10::TypePtr (*type_parser)(const std::string&),  // 类型解析器函数指针
    std::shared_ptr<DeserializationStorageContext> storage_context) {  // 反序列化存储上下文

  // 构建 pickle 文件名
  std::string picklename = pickle_prefix + archive_name + ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size = 0;
  // 从流中获取指定记录的数据指针和大小
  std::tie(pickle_ptr, pickle_size) = stream_reader.getRecord(picklename);

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  // 定义读取函数对象 reader，用于从 pickle 数据中读取数据
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;  // 如果已经读取完所有数据，则返回 0
    }
    len = std::min(pickle_size - bytes_read, len);  // 计算本次需要读取的长度
    // 将数据从 pickle 数据中拷贝到 buffer 中
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;  // 返回实际读取的长度
  };

  // 构建张量文件目录路径
  std::string tensor_dir_path =
      (!tensor_prefix.empty()) ? tensor_prefix : archive_name + "/";

  // 定义读取记录的函数对象 read_record，用于从流中读取指定名称的记录数据
  auto read_record = [&](const std::string& name) {
    std::string ss = tensor_dir_path + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  // 创建 Unpickler 对象，用于反序列化数据
  Unpickler unpickler(
      reader,  // 读取函数对象
      type_resolver ? std::move(*type_resolver) : nullptr,  // 类型解析器（如果提供）
      obj_loader ? std::move(*obj_loader) : nullptr,  // 对象加载器（如果提供）
      std::move(read_record),  // 读取记录的函数对象
      device,  // 设备类型
      false,  // 是否为自定义版本
      type_parser,  // 类型解析器函数指针
      std::move(storage_context));  // 反序列化存储上下文

  // 设置 Unpickler 的版本
  unpickler.set_version(stream_reader.version());
  // 解析并返回解析后的 IValue 对象
  return unpickler.parse_ivalue();
}

// 定义函数 check_zip_file，用于检查给定的流是否为有效的 zip 文件
bool check_zip_file(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai) {
  std::array<uint8_t, 2> first_short{};
  static constexpr uint8_t first_slot = 0x80;
  static constexpr uint8_t second_slot = 0x02;
  // 从流中读取前两个字节，用于检查归档的有效性
  rai->read(
      /*pos=*/0,
      /*buf=*/&first_short,
      /*n=*/2,
      /*what=*/"checking archive");

  // 注意：根据 zip 文件的规范，它们可以以任何数据开头，因此理论上它们可以以 0x80 0x02 开头，
  // 但实际上，zip 文件通常以包含文件条目的数据开始，该条目以 0x04034b50 开头。
  // 此外，PyTorch 生成的 zip 文件始终以文件条目开始，因此相对安全地执行此检查。
  // 检查前两个字节是否为 0x80 和 0x02
  return !(first_short[0] == first_slot && first_short[1] == second_slot);
}

} // namespace torch::jit
```