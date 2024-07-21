# `.\pytorch\c10\core\Stream.cpp`

```
// 包含 C10 核心库中的流定义和虚拟保护实现的头文件
#include <c10/core/Stream.h>
#include <c10/core/impl/VirtualGuardImpl.h>

// 定义 C10 命名空间
namespace c10 {

// 检查当前流上已经排队的所有异步任务是否都已在设备上完成，并返回结果
bool Stream::query() const {
  // 使用设备类型初始化虚拟保护实现对象
  impl::VirtualGuardImpl impl{device_.type()};
  // 调用虚拟保护实现对象的 queryStream 方法，检查流上的异步任务是否完成
  return impl.queryStream(*this);
}

// 阻塞调用线程，直到当前流上已排队的所有异步任务在设备上完成
void Stream::synchronize() const {
  // 使用设备类型初始化虚拟保护实现对象
  impl::VirtualGuardImpl impl{device_.type()};
  // 调用虚拟保护实现对象的 synchronizeStream 方法，等待流上的异步任务完成
  impl.synchronizeStream(*this);
}

// 重载流对象输出到流对象的输出流操作符
std::ostream& operator<<(std::ostream& stream, const Stream& s) {
  // 将流的ID和设备信息输出到给定的输出流
  stream << "stream " << s.id() << " on device " << s.device();
  return stream; // 返回修改后的输出流对象
}

} // namespace c10
```