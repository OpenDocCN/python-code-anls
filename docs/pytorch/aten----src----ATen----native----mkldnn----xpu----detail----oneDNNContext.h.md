# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\oneDNNContext.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/Config.h>
// 包含 ATen 库的配置头文件

#include <c10/core/Device.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
// 包含 C10 库中设备、函数和流管理的头文件

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
// 包含 OneDNN（DNNL）库的头文件

#include <vector>
// 包含 STL 的向量容器头文件

namespace at::native::onednn {

TORCH_XPU_API dnnl::memory make_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr);
// 声明一个函数 make_onednn_memory，用于创建 OneDNN 内存对象

// Keep non-static and non-inline
bool set_onednn_verbose(int level);
// 声明一个函数 set_onednn_verbose，设置 OneDNN 的详细信息级别

// GpuEngineManager singleton
struct TORCH_XPU_API GpuEngineManager {
  static GpuEngineManager& Instance(); // Singleton
  // 静态函数，返回 GpuEngineManager 的单例实例

  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    TORCH_INTERNAL_ASSERT(device.index() < c10::xpu::device_count());
    return *engine_pool[device.index()];
  }
  // 获取特定设备的 OneDNN 引擎对象的函数

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;
  // 禁用复制构造函数和赋值运算符

 protected:
  GpuEngineManager() {
    int device_count = (int)c10::xpu::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
        engine_pool.push_back(
            std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(
              c10::xpu::get_raw_device(i), c10::xpu::get_device_context()
            )));
    }
  }
  // 构造函数，初始化 GpuEngineManager 单例，为每个设备创建对应的 OneDNN 引擎

  ~GpuEngineManager() {}
  // 析构函数，清理资源

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
  // 存储 OneDNN 引擎的共享指针向量
};

// GpuStreamManager singleton
struct TORCH_XPU_API GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton
  // 静态函数，返回 GpuStreamManager 的单例实例

  dnnl::stream get_stream() {
    c10::DeviceIndex device_index = c10::xpu::current_device();
    TORCH_INTERNAL_ASSERT(device_index < c10::xpu::device_count());
    return dnnl::sycl_interop::make_stream(
        GpuEngineManager::Instance().get_engine({c10::kXPU, device_index}),
        c10::xpu::getCurrentXPUStream(device_index).queue());
  }
  // 获取当前设备的 OneDNN 流对象的函数

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;
  // 禁用复制构造函数和赋值运算符

 protected:
  GpuStreamManager() {
  }
  // 构造函数，初始化 GpuStreamManager 单例

  ~GpuStreamManager() {}
  // 析构函数，清理资源

};

} // namespace at::native::onednn
// 命名空间结束声明
```