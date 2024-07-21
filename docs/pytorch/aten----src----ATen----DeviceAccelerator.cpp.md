# `.\pytorch\aten\src\ATen\DeviceAccelerator.cpp`

```py
namespace at {

# 定义 at 命名空间，包含了后续的函数和宏定义


C10_API std::optional<DeviceType> getAccelerator(bool checked) {

# 定义了一个函数 getAccelerator，返回类型为 std::optional<DeviceType>，接受一个布尔类型参数 checked


#define CHECK_NO_CUDA \
  TORCH_CHECK(!at::hasCUDA(), "Cannot have both CUDA and PrivateUse1");

# 定义宏 CHECK_NO_CUDA，用于检查是否存在 CUDA 设备，若存在则抛出异常，指出 CUDA 和 PrivateUse1 不能同时存在


#define CHECK_NO_PU1 \
  TORCH_CHECK(!is_privateuse1_backend_registered(), "Cannot have both CUDA and PrivateUse1");

# 定义宏 CHECK_NO_PU1，用于检查是否注册了 PrivateUse1 后端，若注册了则抛出异常，指出 CUDA 和 PrivateUse1 不能同时存在


#define CHECK_NO_MTIA \
  TORCH_CHECK(!at::hasMTIA(), "Cannot have MTIA with other devices");

# 定义宏 CHECK_NO_MTIA，用于检查是否存在 MTIA 设备，若存在则抛出异常，指出 MTIA 不能与其他设备同时存在


if (is_privateuse1_backend_registered()) {

# 如果注册了 PrivateUse1 后端，则执行以下操作：


        // We explicitly allow PrivateUse1 and another device at the same time
        // as we use this for testing.

# 注释：我们明确允许同时使用 PrivateUse1 和另一个设备，这是为了测试目的。


        // Whenever a PrivateUse1 device is registered, use it first.

# 注释：每当注册了 PrivateUse1 设备时，优先使用它。


        return kPrivateUse1;

# 返回 PrivateUse1 设备类型


    } else if (at::hasCUDA()) {

# 否则如果存在 CUDA 设备，则执行以下操作：


        CHECK_NO_PU1
        CHECK_NO_MTIA

# 调用宏检查不允许同时存在 PrivateUse1 和 MTIA 设备


        return kCUDA;

# 返回 CUDA 设备类型


    } else if (at::hasMTIA()) {

# 否则如果存在 MTIA 设备，则执行以下操作：


        CHECK_NO_CUDA
        CHECK_NO_PU1

# 调用宏检查不允许同时存在 CUDA 和 PrivateUse1 设备


        return kMTIA;

# 返回 MTIA 设备类型


    } else {

# 否则执行以下操作：


        TORCH_CHECK(!checked, "Cannot access accelerator device when none is available.")

# 如果 checked 为真，抛出异常，指出没有可用的加速设备


        return std::nullopt;

# 返回空的 std::optional，表示没有可用的加速设备


    }

#undef CHECK_NO_CUDA
#undef CHECK_NO_PU1

# 取消定义之前定义的宏 CHECK_NO_CUDA 和 CHECK_NO_PU1


} // namespace at

# 结束 at 命名空间定义
```