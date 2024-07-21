# `.\pytorch\c10\core\Device.cpp`

```
// 匿名命名空间内定义的函数，用于将设备字符串解析为 DeviceType 枚举值
namespace {
DeviceType parse_type(const std::string& device_string) {
  // 静态数组，包含设备类型名称和对应的 DeviceType 枚举值
  static const std::array<
      std::pair<const char*, DeviceType>,
      static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
      types = {{
          {"cpu", DeviceType::CPU},
          {"cuda", DeviceType::CUDA},
          {"ipu", DeviceType::IPU},
          {"xpu", DeviceType::XPU},
          {"mkldnn", DeviceType::MKLDNN},
          {"opengl", DeviceType::OPENGL},
          {"opencl", DeviceType::OPENCL},
          {"ideep", DeviceType::IDEEP},
          {"hip", DeviceType::HIP},
          {"ve", DeviceType::VE},
          {"fpga", DeviceType::FPGA},
          {"maia", DeviceType::MAIA},
          {"xla", DeviceType::XLA},
          {"lazy", DeviceType::Lazy},
          {"vulkan", DeviceType::Vulkan},
          {"mps", DeviceType::MPS},
          {"meta", DeviceType::Meta},
          {"hpu", DeviceType::HPU},
          {"mtia", DeviceType::MTIA},
          {"privateuseone", DeviceType::PrivateUse1},
      }};
  
  // 查找设备字符串对应的 DeviceType 枚举值
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&device_string](const std::pair<const char*, DeviceType>& p) {
        return p.first && p.first == device_string;
      });
  
  // 如果找到匹配的设备类型，则返回其对应的 DeviceType 枚举值
  if (device != types.end()) {
    return device->second;
  }
  
  // 如果设备字符串匹配私有使用1的后端名称，则返回对应的 DeviceType 枚举值
  if (device_string == get_privateuse1_backend()) {
    return DeviceType::PrivateUse1;
  }
  
  // 构建设备类型名称列表，用于错误消息
  std::vector<const char*> device_names;
  for (const auto& it : types) {
    if (it.first) {
      device_names.push_back(it.first);
    }
  }
  
  // 抛出错误，指示设备字符串不符合预期的格式
  TORCH_CHECK(
      false,
      "Expected one of ",
      c10::Join(", ", device_names),
      " device type at start of device string: ",
      device_string);
}
} // namespace
    switch (pstate) {
      case DeviceStringParsingState::START:
        // 如果当前状态为START，检查当前字符是否为':'，如果不是则判断是否为字母或下划线，若是则添加到device_name中，否则转到ERROR状态。
        if (ch != ':') {
          if (isalpha(ch) || ch == '_') {
            device_name.push_back(ch);
          } else {
            pstate = DeviceStringParsingState::ERROR;
          }
        } else {
          pstate = DeviceStringParsingState::INDEX_START;
        }
        break;

      case DeviceStringParsingState::INDEX_START:
        // 如果当前状态为INDEX_START，检查当前字符是否为数字，若是则添加到device_index_str中并转到INDEX_REST状态，否则转到ERROR状态。
        if (isdigit(ch)) {
          device_index_str.push_back(ch);
          pstate = DeviceStringParsingState::INDEX_REST;
        } else {
          pstate = DeviceStringParsingState::ERROR;
        }
        break;

      case DeviceStringParsingState::INDEX_REST:
        // 如果当前状态为INDEX_REST，检查device_index_str第一个字符是否为'0'，如果是则转到ERROR状态，如果当前字符是数字则添加到device_index_str中，否则转到ERROR状态。
        if (device_index_str.at(0) == '0') {
          pstate = DeviceStringParsingState::ERROR;
          break;
        }
        if (isdigit(ch)) {
          device_index_str.push_back(ch);
        } else {
          pstate = DeviceStringParsingState::ERROR;
        }
        break;

      case DeviceStringParsingState::ERROR:
        // 不会执行到这里的注释，因为一旦状态变为ERROR，解析过程将会提前结束。
        break;
    }
  }

  // 检查是否存在解析错误，如果device_name为空或者状态为ERROR，或者状态为INDEX_START且device_index_str为空，则存在错误。
  const bool has_error = device_name.empty() ||
      pstate == DeviceStringParsingState::ERROR ||
      (pstate == DeviceStringParsingState::INDEX_START &&
       device_index_str.empty());

  // 使用TORCH_CHECK断言，如果存在错误，则输出错误消息并终止程序。
  TORCH_CHECK(!has_error, "Invalid device string: '", device_string, "'");

  try {
    // 尝试解析device_index_str为整数，如果不为空则转换为c10::DeviceIndex类型的index_。
    if (!device_index_str.empty()) {
      index_ = static_cast<c10::DeviceIndex>(std::stoi(device_index_str));
    }
  } catch (const std::exception&) {
    // 如果解析失败，则输出错误消息包含错误的device_index_str和device_string，并终止程序。
    TORCH_CHECK(
        false,
        "Could not parse device index '",
        device_index_str,
        "' in device string '",
        device_string,
        "'");
  }
  // 解析device_name为type_。
  type_ = parse_type(device_name);
  // 执行验证函数validate()。
  validate();
}

# 定义 Device 类的 str() 方法，返回设备名称的字符串表示
std::string Device::str() const {
  # 根据设备类型和参数 lower case 创建设备类型名称字符串
  std::string str = DeviceTypeName(type(), /* lower case */ true);
  # 如果设备有索引，添加索引信息到字符串末尾
  if (has_index()) {
    str.push_back(':');
    str.append(std::to_string(index()));
  }
  # 返回构建好的设备名称字符串
  return str;
}

# 定义流操作符重载函数，将 Device 对象输出到流中
std::ostream& operator<<(std::ostream& stream, const Device& device) {
  # 将设备对象的字符串表示输出到流中
  stream << device.str();
  # 返回输出流对象
  return stream;
}

} // namespace c10
```