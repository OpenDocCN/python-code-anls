# `.\pytorch\c10\xpu\test\impl\XPUDeviceTest.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 C10 库的 XPU 相关头文件
#include <c10/xpu/XPUFunctions.h>

// 检查系统是否有可用的 XPU 设备
bool has_xpu() {
    return c10::xpu::device_count() > 0;
}

// 测试 XPU 设备的基本行为
TEST(XPUDeviceTest, DeviceBehavior) {
    // 如果系统没有 XPU 设备，则退出测试
    if (!has_xpu()) {
        return;
    }

    // 设置当前设备为第一个 XPU 设备
    c10::xpu::set_device(0);
    // 断言当前设备编号为 0
    EXPECT_EQ(c10::xpu::current_device(), 0);

    // 如果系统只有一个 XPU 设备，则退出测试
    if (c10::xpu::device_count() <= 1) {
        return;
    }

    // 设置当前设备为第二个 XPU 设备
    c10::xpu::set_device(1);
    // 断言当前设备编号为 1
    EXPECT_EQ(c10::xpu::current_device(), 1);
    // 切换当前设备为第一个 XPU 设备，并断言上一个设备编号为 1
    EXPECT_EQ(c10::xpu::exchange_device(0), 1);
    // 断言当前设备编号为 0
    EXPECT_EQ(c10::xpu::current_device(), 0);
}

// 测试 XPU 设备的属性
TEST(XPUDeviceTest, DeviceProperties) {
    // 如果系统没有 XPU 设备，则退出测试
    if (!has_xpu()) {
        return;
    }

    // 获取第一个 XPU 设备的属性信息
    c10::xpu::DeviceProp device_prop{};
    c10::xpu::get_device_properties(&device_prop, 0);

    // 断言最大计算单元数大于 0
    EXPECT_TRUE(device_prop.max_compute_units > 0);
    // 断言 GPU 执行单元数大于 0
    EXPECT_TRUE(device_prop.gpu_eu_count > 0);
}

// 测试通过指针获取 XPU 设备编号的功能
TEST(XPUDeviceTest, PointerGetDevice) {
    // 如果系统没有 XPU 设备，则退出测试
    if (!has_xpu()) {
        return;
    }

    // 获取第一个 XPU 设备的原始设备对象
    sycl::device& raw_device = c10::xpu::get_raw_device(0);
    // 在第一个 XPU 设备上分配 8 字节内存，并获取其指针
    void* ptr = sycl::malloc_device(8, raw_device, c10::xpu::get_device_context());

    // 断言通过指针获取的设备编号为 0
    EXPECT_EQ(c10::xpu::get_device_idx_from_pointer(ptr), 0);
    // 释放先前分配的内存
    sycl::free(ptr, c10::xpu::get_device_context());

    // 创建一个整型变量 dummy，用于测试异常情况
    int dummy = 0;
    // 断言通过非设备指针获取设备编号会抛出异常
    ASSERT_THROW(c10::xpu::get_device_idx_from_pointer(&dummy), c10::Error);
}
```