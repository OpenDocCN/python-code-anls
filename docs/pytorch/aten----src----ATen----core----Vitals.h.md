# `.\pytorch\aten\src\ATen\core\Vitals.h`

```
#pragma once
#include <ostream>
#include <sstream>
#include <unordered_map>

#include <c10/core/impl/LocalDispatchKeySet.h>

// 定义了 at::vitals 命名空间，用于封装与 TorchVital 相关的功能
namespace at::vitals {

// 声明一个函数，用于检查 TorchVital 是否启用
TORCH_API bool torchVitalEnabled();

// 定义了 TorchVitalAttr 结构体，用于存储单个属性的字符串值
struct TORCH_API TorchVitalAttr {
  // 属性值的字符串表示，默认为空字符串
  std::string value = "";

  // 运算符重载，将输入的值转换为字符串，并添加到属性值中（若 TorchVital 已启用）
  template <typename T>
  TorchVitalAttr& operator<<(const T& t) {
    if (torchVitalEnabled()) {
      std::stringstream ss;
      ss << t;
      value += ss.str();
    }
    return *this;
  }

  // 写入函数，将输入的值转换为字符串，并更新属性值（若 TorchVital 已启用或强制写入）
  template <typename T>
  void write(const T& t, bool force) {
    if (force || torchVitalEnabled()) {
      std::stringstream ss;
      ss << t;
      value = ss.str();
    }
  }
};

// 定义了 TorchVital 结构体，表示一个具体的 Torch Vital（重要记录）
struct TORCH_API TorchVital {
  // Torch Vital 的名称
  std::string name;
  // 属性名称到 TorchVitalAttr 对象的映射
  std::unordered_map<std::string, TorchVitalAttr> attrs;

  // 构造函数，通过给定名称初始化 Torch Vital
  explicit TorchVital(std::string n) : name(std::move(n)) {}

  // 复制构造函数和移动构造函数默认实现（不可用）
  TorchVital(const TorchVital&) = default;
  TorchVital(TorchVital&&) = default;

  // 默认构造函数删除
  TorchVital() = delete;

  // 创建指定名称的属性，并返回对应的 TorchVitalAttr 引用
  TorchVitalAttr& create(const std::string& attr);

  // 创建指定名称的属性，并返回对应的 TorchVitalAttr 引用（支持强制写入）
  TorchVitalAttr& create(const std::string& attr, bool force);

  // 友元函数，重载流输出运算符，用于将 TorchVital 对象的信息输出到流中
  friend std::ostream& operator<<(std::ostream& os, const TorchVital& dt);

  // 析构函数声明
  ~TorchVital();
};

// 友元函数声明，用于将 TorchVital 对象的信息输出到流中
std::ostream& operator<<(std::ostream& os, TorchVital const& tv);

// APIVitals 类，用于通过字符串名称访问 vitals 而不是全局引用
class TORCH_API APIVitals {
 public:
  // 是否启用 vitals 记录的标志
  bool vitals_enabled;

  // 设置指定 vital_name 和 attr_name 的 vital 记录的值
  bool setVital(
      const std::string& vital_name,
      const std::string& attr_name,
      const std::string& value,
      bool force = false);

  // 读取当前所有 vitals 的字符串表示
  std::string readVitals();

  // 默认构造函数，初始化 APIVitals 对象
  APIVitals();

  // 禁止拷贝构造函数和移动构造函数
  APIVitals(APIVitals const& other) = delete;
  APIVitals(APIVitals&& other) = delete;
  APIVitals& operator=(const APIVitals&) = delete;
  APIVitals& operator=(APIVitals&&) = delete;

 private:
  // vital 名称到 TorchVital 对象的映射
  std::unordered_map<std::string, TorchVital> name_map_;
};

// 外部声明，表示全局唯一的 VitalsAPI 对象
extern TORCH_API APIVitals VitalsAPI;

} // namespace at::vitals

// 定义宏 TORCH_VITAL_DECLARE，用于声明 TorchVital 对象
#define TORCH_VITAL_DECLARE(name) \
  TORCH_API at::vitals::TorchVital TorchVital_##name;

// 定义宏 TORCH_VITAL_DEFINE，用于定义 TorchVital 对象，并初始化其名称
#define TORCH_VITAL_DEFINE(name) \
  TORCH_API at::vitals::TorchVital TorchVital_##name(#name);

// 定义宏 TORCH_VITAL_BASE，用于获取指定 TorchVital 的引用
#define TORCH_VITAL_BASE(name) TorchVital_##name

// 定义宏 TORCH_VITAL，用于创建指定 TorchVital 的指定属性
#define TORCH_VITAL(name, attr) TORCH_VITAL_BASE(name).create(#attr)
```