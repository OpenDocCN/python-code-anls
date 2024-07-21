# `.\pytorch\aten\src\ATen\Version.cpp`

```py
// 包含 ATen 库的版本信息头文件
#include <ATen/Version.h>
// 包含 ATen 库的配置信息头文件
#include <ATen/Config.h>

// 如果 ATen 使用了 MKL，包含 MKL 头文件
#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

// 如果 ATen 使用了 MKLDNN，包含 MKLDNN 和 ideep 头文件
#if AT_MKLDNN_ENABLED()
#include <dnnl.hpp>
#include <ideep.hpp>
#endif

// 包含 Caffe2 核心库的通用头文件
#include <caffe2/core/common.h>

// 包含 ATen 库中的调度框架头文件
#include <ATen/native/DispatchStub.h>

// 包含字符串流处理头文件
#include <sstream>

// ATen 命名空间开始
namespace at {

// 返回 MKL 版本信息的函数
std::string get_mkl_version() {
  std::string version;
  // 如果 ATen 使用了 MKL
  #if AT_MKL_ENABLED()
    {
      // 从 MKL 文档中取得的魔法缓冲区大小
      // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-get-version-string
      char buf[198];
      // 获取 MKL 版本字符串并存入 buf 中
      mkl_get_version_string(buf, 198);
      version = buf;
    }
  // 如果 ATen 没有使用 MKL
  #else
    version = "MKL not found";
  #endif
  // 返回版本信息字符串
  return version;
}

// 返回 MKLDNN 版本信息的函数
std::string get_mkldnn_version() {
  std::ostringstream ss;
  // 如果 ATen 使用了 MKLDNN
  #if AT_MKLDNN_ENABLED()
    // 参考自 mkl-dnn/src/common/verbose.cpp
    // 无法方便地获取 ISA 信息
    // 显然无法获取 ideep 版本？
    // https://github.com/intel/ideep/issues/29
    {
      const dnnl_version_t* ver = dnnl_version();
      // 构建 MKLDNN 版本信息字符串
      ss << "Intel(R) MKL-DNN v" << ver->major << "." << ver->minor << "." << ver->patch
         << " (Git Hash " << ver->hash << ")";
    }
  // 如果 ATen 没有使用 MKLDNN
  #else
    ss << "MKLDNN not found";
  #endif
  // 返回版本信息字符串
  return ss.str();
}

// 返回 OpenMP 版本信息的函数
std::string get_openmp_version() {
  std::ostringstream ss;
  // 如果定义了 _OPENMP 宏
  #ifdef _OPENMP
    {
      // 输出 OpenMP 版本号
      ss << "OpenMP " << _OPENMP;
      // 参考资料：
      // https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux
      const char* ver_str = nullptr;
      // 根据 OpenMP 版本号选择适当的字符串描述
      switch (_OPENMP) {
        case 200505:
          ver_str = "2.5";
          break;
        case 200805:
          ver_str = "3.0";
          break;
        case 201107:
          ver_str = "3.1";
          break;
        case 201307:
          ver_str = "4.0";
          break;
        case 201511:
          ver_str = "4.5";
          break;
        default:
          ver_str = nullptr;
          break;
      }
      // 如果找到了匹配的 OpenMP 版本号，则附加到版本信息中
      if (ver_str) {
        ss << " (a.k.a. OpenMP " << ver_str << ")";
      }
    }
  // 如果未定义 _OPENMP 宏
  #else
    ss << "OpenMP not found";
  #endif
  // 返回版本信息字符串
  return ss.str();
}

// 返回 CPU 能力信息的函数
std::string get_cpu_capability() {
  // 可能会使用环境变量覆盖 cpu_capability
  auto capability = native::get_cpu_capability();
  switch (capability) {
    // 如果定义了 HAVE_VSX_CPU_DEFINITION
    #if defined(HAVE_VSX_CPU_DEFINITION)
      case native::CPUCapability::DEFAULT:
        return "DEFAULT";
      case native::CPUCapability::VSX:
        return "VSX";
    // 如果定义了 HAVE_ZVECTOR_CPU_DEFINITION
    #elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
      case native::CPUCapability::DEFAULT:
        return "DEFAULT";
      case native::CPUCapability::ZVECTOR:
        return "Z VECTOR";
    // 如果以上宏均未定义，则根据不同的 CPU 能力返回相应描述
    #else
      case native::CPUCapability::DEFAULT:
        return "NO AVX";
      case native::CPUCapability::AVX2:
        return "AVX2";
      case native::CPUCapability::AVX512:
        return "AVX512";
    #endif
    default:
      break;
  }
  // 默认返回空字符串
  return "";
}

// ATen 命名空间结束
}
// 返回当前程序使用的 CPU 能力字符串，可能会被环境变量覆盖
static std::string used_cpu_capability() {
  // 创建一个字符串流对象
  std::ostringstream ss;
  // 获取并追加 CPU 能力的描述信息到字符串流中
  ss << "CPU capability usage: " << get_cpu_capability();
  // 将字符串流转换为字符串并返回
  return ss.str();
}

// 返回 PyTorch 的配置信息字符串
std::string show_config() {
  // 创建一个字符串流对象
  std::ostringstream ss;
  // 添加 PyTorch 构建信息的开头描述
  ss << "PyTorch built with:\n";

  // 如果当前编译器是 GCC，则追加 GCC 的版本信息到字符串流中
#if defined(__GNUC__)
  {
    ss << "  - GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
  }
#endif

  // 如果定义了 __cplusplus 宏，则追加 C++ 标准版本信息到字符串流中
#if defined(__cplusplus)
  {
    ss << "  - C++ Version: " << __cplusplus << "\n";
  }
#endif

  // 如果定义了 __clang_major__ 宏，则追加 clang 的版本信息到字符串流中
#if defined(__clang_major__)
  {
    ss << "  - clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
  }
#endif

  // 如果定义了 _MSC_VER 宏，则追加 MSVC 的版本信息到字符串流中
#if defined(_MSC_VER)
  {
    ss << "  - MSVC " << _MSC_FULL_VER << "\n";
  }
#endif

  // 如果 MKL 被启用，追加 MKL 版本信息到字符串流中
#if AT_MKL_ENABLED()
  ss << "  - " << get_mkl_version() << "\n";
#endif

  // 如果 MKLDNN 被启用，追加 MKLDNN 版本信息到字符串流中
#if AT_MKLDNN_ENABLED()
  ss << "  - " << get_mkldnn_version() << "\n";
#endif

  // 如果定义了 _OPENMP 宏，则追加 OpenMP 版本信息到字符串流中
#ifdef _OPENMP
  ss << "  - " << get_openmp_version() << "\n";
#endif

  // 如果 LAPACK 被启用，追加 LAPACK 已启用的信息到字符串流中
#if AT_BUILD_WITH_LAPACK()
  // TODO: 实际记录我们选择了哪个版本
  ss << "  - LAPACK is enabled (usually provided by MKL)\n";
#endif

  // 如果 NNPACK 被启用，追加 NNPACK 已启用的信息到字符串流中
#if AT_NNPACK_ENABLED()
  // TODO: 没有版本信息可用
  ss << "  - NNPACK is enabled\n";
#endif

  // 如果定义了 CROSS_COMPILING_MACOSX 宏，则追加在 MacOSX 上的交叉编译信息到字符串流中
#ifdef CROSS_COMPILING_MACOSX
  ss << "  - Cross compiling on MacOSX\n";
#endif

  // 追加当前 CPU 能力使用信息到字符串流中
  ss << "  - "<< used_cpu_capability() << "\n";

  // 如果检测到 CUDA 的存在，则追加 CUDA 的配置信息到字符串流中
  if (hasCUDA()) {
    ss << detail::getCUDAHooks().showConfig();
  }

  // 如果检测到 MAIA 的存在，则追加 MAIA 的配置信息到字符串流中
  if (hasMAIA()) {
    ss << detail::getMAIAHooks().showConfig();
  }

  // 如果检测到 XPU 的存在，则追加 XPU 的配置信息到字符串流中
  if (hasXPU()) {
    ss << detail::getXPUHooks().showConfig();
  }

  // 追加构建设置信息到字符串流中
  ss << "  - Build settings: ";
  // 遍历并追加每个构建选项的名称和值到字符串流中
  for (const auto& pair : caffe2::GetBuildOptions()) {
    if (!pair.second.empty()) {
      ss << pair.first << "=" << pair.second << ", ";
    }
  }
  // 追加换行符到字符串流中
  ss << "\n";

  // TODO: 处理 HIP
  // TODO: 处理 XLA
  // TODO: 处理 MPS

  // 将字符串流转换为字符串并返回
  return ss.str();
}

// 返回 C++ 编译标志字符串
std::string get_cxx_flags() {
  // 如果定义了 FBCODE_CAFFE2 宏，则抛出异常，因为不支持从 Buck 中获取 CXX_FLAGS 信息
  #if defined(FBCODE_CAFFE2)
  TORCH_CHECK(
    false,
    "Buck does not populate the `CXX_FLAGS` field of Caffe2 build options. "
    "As a result, `get_cxx_flags` is OSS only."
  );
  #endif
  // 返回 Caffe2 构建选项中的 CXX_FLAGS 的值
  return caffe2::GetBuildOptions().at("CXX_FLAGS");
}
```