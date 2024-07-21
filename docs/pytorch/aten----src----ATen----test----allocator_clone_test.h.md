# `.\pytorch\aten\src\ATen\test\allocator_clone_test.h`

```py
#pragma once
#include <gtest/gtest.h>
#include <ATen/ATen.h>

// 测试分配器的克隆功能
void test_allocator_clone(c10::Allocator* allocator) {
    // 断言分配器不为空
    ASSERT_TRUE(allocator != nullptr);

    // 创建一个可调整大小的 Storage 对象 a_storage，使用给定的 allocator
    c10::Storage a_storage(c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        0,
        allocator,
        /*resizable=*/true));

    // 创建一个可调整大小的 Storage 对象 b_storage，使用给定的 allocator
    c10::Storage b_storage(c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        0,
        allocator,
        /*resizable=*/true));

    // 创建一个空的 Tensor a，并设置其底层的 Storage 为 a_storage
    at::Tensor a = at::empty({0}, at::TensorOptions().device(a_storage.device())).set_(a_storage);
    
    // 创建一个空的 Tensor b，并设置其底层的 Storage 为 b_storage
    at::Tensor b = at::empty({0}, at::TensorOptions().device(b_storage.device())).set_(b_storage);

    // 定义一个大小为 {13, 4, 5} 的整数向量 sizes
    std::vector<int64_t> sizes({13, 4, 5});

    // 在 Tensor a 上生成随机数据，大小为 sizes
    at::rand_out(a, sizes);
    
    // 在 Tensor b 上生成随机数据，大小为 sizes
    at::rand_out(b, sizes);

    // 断言 a_storage 的字节数等于 a 中元素个数乘以元素大小
    ASSERT_TRUE(a_storage.nbytes() == static_cast<size_t>(a.numel() * a.element_size()));
    
    // 断言 a_storage 的字节数等于 b_storage 的字节数
    ASSERT_TRUE(a_storage.nbytes() == b_storage.nbytes());

    // 获取 a_storage 的可变数据指针 a_data_ptr
    void* a_data_ptr = a_storage.mutable_data();
    
    // 使用 allocator 的 clone 方法将 a_data_ptr 克隆到 b_storage 的数据指针中
    b_storage.set_data_ptr(allocator->clone(a_data_ptr, a_storage.nbytes()));

    // 断言 Tensor a 和 Tensor b 的所有元素是否完全相等
    ASSERT_TRUE((a == b).all().item<bool>());
}
```