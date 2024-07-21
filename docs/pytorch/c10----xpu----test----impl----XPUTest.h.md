# `.\pytorch\c10\xpu\test\impl\XPUTest.h`

```
# 包含 Google Test 框架的头文件，用于单元测试
#include <gtest/gtest.h>

# 包含 C10 库中的 irange 头文件，用于生成从 0 到 numel-1 的范围迭代器
#include <c10/util/irange.h>

# 初始化主机端数据的函数，将 0 到 numel-1 的整数依次赋值给 hostData 数组
static inline void initHostData(int* hostData, int numel) {
    for (const auto i : c10::irange(numel)) {
        hostData[i] = i;
    }
}

# 清空主机端数据的函数，将 hostData 数组中的所有元素置为 0
static inline void clearHostData(int* hostData, int numel) {
    for (const auto i : c10::irange(numel)) {
        hostData[i] = 0;
    }
}

# 验证主机端数据的函数，检查 hostData 数组中的每个元素是否等于其下标 i
static inline void validateHostData(int* hostData, int numel) {
    for (const auto i : c10::irange(numel)) {
        # 使用 Google Test 的 EXPECT_EQ 断言验证，判断 hostData[i] 是否等于 i
        EXPECT_EQ(hostData[i], i);
    }
}
```