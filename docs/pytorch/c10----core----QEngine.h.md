# `.\pytorch\c10\core\QEngine.h`

```
#pragma once

#include <c10/util/Exception.h>
#include <cstdint>
#include <string>

namespace c10 {

/**
 * QEngine is an enum that is used to select the engine to run quantized ops.
 * Keep this enum in sync with get_qengine_id() in
 * torch/backends/quantized/__init__.py
 */
// 定义了一个枚举类型 QEngine，用于选择运行量化操作的引擎
enum class QEngine : uint8_t {
  NoQEngine = 0,   // 无引擎
  FBGEMM = 1,       // FBGEMM 引擎
  QNNPACK = 2,      // QNNPACK 引擎
  ONEDNN = 3,       // ONEDNN 引擎
  X86 = 4,          // X86 引擎
};

constexpr auto kNoQEngine = QEngine::NoQEngine;     // 常量 kNoQEngine 指代 NoQEngine 引擎
constexpr auto kFBGEMM = QEngine::FBGEMM;           // 常量 kFBGEMM 指代 FBGEMM 引擎
constexpr auto kQNNPACK = QEngine::QNNPACK;         // 常量 kQNNPACK 指代 QNNPACK 引擎
constexpr auto kONEDNN = QEngine::ONEDNN;           // 常量 kONEDNN 指代 ONEDNN 引擎
constexpr auto kX86 = QEngine::X86;                 // 常量 kX86 指代 X86 引擎

// 将 QEngine 转换为对应的字符串表示
inline std::string toString(QEngine qengine) {
  switch (qengine) {
    case kNoQEngine:
      return "NoQEngine";
    case kFBGEMM:
      return "FBGEMM";
    case kQNNPACK:
      return "QNNPACK";
    case kONEDNN:
      return "ONEDNN";
    case kX86:
      return "X86";
    default:
      // 如果未知的量化引擎，抛出异常
      TORCH_CHECK(
          false, "Unrecognized Quantized Engine: ", static_cast<int>(qengine));
  }
}

} // namespace c10
```