# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\oneDNNContext.cpp`

```
/* *
 * Do NOT put any kernels or call any device binaries here!
 * Only maintain oneDNN runtime states in this file.
 * */

// 命名空间声明：at::native::onednn，用于包含所有的oneDNN相关实现
namespace at::native::onednn {

// 引入dnnl命名空间，简化使用dnnl库中的元素
using namespace dnnl;

// GpuEngineManager类的单例模式实现
GpuEngineManager& GpuEngineManager::Instance() {
  // 静态局部变量保证单例模式下的唯一实例
  static GpuEngineManager myInstance;
  return myInstance;
}

// GpuStreamManager类的单例模式实现，每个线程有一个实例
GpuStreamManager& GpuStreamManager::Instance() {
  // 使用线程局部存储确保每个线程的唯一实例
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

// 设置oneDNN的详细输出级别
bool set_onednn_verbose(int level) {
  // 调用oneDNN库函数设置详细输出级别，并返回操作是否成功
  dnnl::status rs = dnnl::set_verbose(level);
  return rs == dnnl::status::success;
}

} // namespace at::native::onednn
```