# `.\pytorch\torch\csrc\distributed\c10d\DMAConnectivity.cpp`

```
// 定义一个字符串，用于表示检测器的唯一标识符，由设备类型和连接类型组成
std::string get_detector_key(
    c10::DeviceType device_type,
    std::string connection_type) {
  // 创建一个字符串流对象
  std::ostringstream oss;
  // 将设备类型和连接类型写入字符串流
  oss << device_type << "/" << connection_type;
  // 返回拼接后的字符串
  return oss.str();
}

// DetectorMap 类的实现，用于管理 DMAConnectivityDetector 对象的注册和检测
class DetectorMap {
 public:
  // 获取 DetectorMap 单例的静态方法
  static DetectorMap& get() {
    // 创建 DetectorMap 的静态实例，确保全局唯一性
    static DetectorMap instance;
    return instance;
  }

  // 注册 DMAConnectivityDetector 对象到 DetectorMap 中
  void register_detector(
      c10::DeviceType device_type,
      const std::string& connection_type,
      c10::intrusive_ptr<c10d::DMAConnectivityDetector> detector) {
    // 获取 detector 的唯一标识符
    auto key = get_detector_key(device_type, connection_type);
    // 将 detector 存入 detector_map_ 中
    detector_map_[key] = std::move(detector);
  }

  // 检测给定设备类型和连接类型的 DMA 连通性
  c10::intrusive_ptr<c10d::DMAConnectivity> detect(
      c10::DeviceType device_type,
      const std::string& connection_type) {
    // 获取 detector 的唯一标识符
    auto key = get_detector_key(device_type, connection_type);
    {
      // 在缓存中查找是否已有对应的 DMA 连通性对象
      auto it = cached_.find(key);
      // 如果找到则直接返回缓存中的对象
      if (it != cached_.end()) {
        return it->second;
      }
    }

    // 在 detector_map_ 中查找给定标识符对应的 DMAConnectivityDetector 对象
    auto it = detector_map_.find(key);
    // 如果找不到则抛出错误
    TORCH_CHECK(
        it != detector_map_.end(),
        "DMA connectivity detector for ",
        device_type,
        " over ",
        connection_type,
        " is not available");
    // 获取对应的 DMAConnectivityDetector 对象
    auto detector = it->second;
    // 调用 DMAConnectivityDetector 的 detect 方法检测连接性
    auto connectivity = detector->detect();
    // 将检测结果存入缓存中
    cached_[key] = connectivity;
    // 返回 DMA 连通性对象
    return connectivity;
  }

 private:
  // 构造函数，私有化以确保单例模式
  DetectorMap() = default;
  // 禁用拷贝构造函数和赋值操作符重载，确保单例模式下不会被复制
  DetectorMap(const DetectorMap&) = delete;
  DetectorMap& operator=(const DetectorMap&) = delete;

  // 存储设备类型和连接类型到 DMAConnectivityDetector 的映射
  std::unordered_map<
      std::string,
      c10::intrusive_ptr<c10d::DMAConnectivityDetector>>
      detector_map_;

  // 缓存检测结果，存储设备类型和连接类型到 DMAConnectivity 的映射
  std::unordered_map<std::string, c10::intrusive_ptr<c10d::DMAConnectivity>>
      cached_;
};

// 匿名命名空间，用于隐藏内部实现细节
namespace {

// DMAConnectivity 类的构造函数，初始化设备类型、连接类型和连接矩阵
DMAConnectivity::DMAConnectivity(
    c10::DeviceType device_type,
    std::string connection_type,
    std::vector<std::vector<int>> matrix)
    : device_type(device_type),
      connection_type(connection_type),
      matrix(std::move(matrix)) {}

} // namespace c10d

// 注册 DMAConnectivityDetector 到 DetectorMap 中的全局函数
void register_dma_connectivity_detector(
    c10::DeviceType device_type,
    const std::string& connection_type,
    c10::intrusive_ptr<DMAConnectivityDetector> detector) {
  // 调用 DetectorMap 的 register_detector 方法注册检测器
  return DetectorMap::get().register_detector(
      device_type, connection_type, std::move(detector));
}

// 检测 DMA 连通性的全局函数
c10::intrusive_ptr<DMAConnectivity> detect_dma_connectivity(
    c10::DeviceType device_type,
    const std::string& connection_type) {
  // 调用 DetectorMap 的 detect 方法检测 DMA 连通性
  return DetectorMap::get().detect(device_type, connection_type);
}
```